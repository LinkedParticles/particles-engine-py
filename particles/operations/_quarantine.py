"""Quarantined-loser resolution shared by Review and the trust cascade.

A *quarantined* particle is the losing candidate of a §6.6 INCONSISTENT
verdict, persisted at extract time born ``PROVENANCE_STALE`` with
``status_reason = CONFLICT_PENDING``. Two resolution moves exist over it:

- the conflict resolves *against* it → a reason-only flip to
  ``CONFLICT_RESOLVED`` (``store.particle_store.update_status_reason``; no
  status transition — stale stays stale);
- the conflict resolves *for* it (Review PREFER_B / BOTH_VALID, or a trust
  cascade in its favour) → :func:`promote_quarantined` mints a **new ACTIVE
  particle**, mirroring the Reindex pattern: the quarantined row transitions
  ``PROVENANCE_STALE → SUPERSEDED`` and the minted particle records
  ``supersedes``. There is deliberately no ``PROVENANCE_STALE → ACTIVE`` edge
  (see ``core/status.py``) — promotion never reactivates a stale particle.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle, UncertaintyNature
from particles.core.status import Status, StatusReason, validate_transition
from particles.store.particle_store import (
    copy_particle_embedding,
    insert_particle,
    update_particle_status,
)

log = logging.getLogger(__name__)


def is_quarantined(particle: Particle) -> bool:
    """True for a §6.6 quarantined conflict loser."""
    return (
        particle.status is Status.PROVENANCE_STALE
        and particle.status_reason is StatusReason.CONFLICT_PENDING
    )


async def promote_quarantined(
    session: AsyncSession,
    quarantined: Particle,
    *,
    uncertainty_nature: UncertaintyNature | None = None,
) -> Particle:
    """Mint a new ACTIVE particle from a quarantined conflict loser.

    Fresh id, ``supersedes`` → the quarantined row; content, confidence,
    provenance, subjects, and embedding carried over verbatim. The quarantined
    row transitions ``PROVENANCE_STALE → SUPERSEDED`` (CONFLICT_RESOLVED).

    Args:
        session: Active session — the caller commits.
        quarantined: The quarantined particle; must satisfy
            :func:`is_quarantined`.
        uncertainty_nature: Optional override on the minted particle —
            BOTH_VALID promotes with ``ALEATORY``.

    Returns:
        The minted ACTIVE particle.

    Raises:
        ValueError: If ``quarantined`` is not a quarantined conflict loser.
    """
    if not is_quarantined(quarantined):
        raise ValueError(
            f"Particle {quarantined.id} is not a quarantined conflict loser"
            " (expected PROVENANCE_STALE / CONFLICT_PENDING)"
        )
    update: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "status": Status.ACTIVE,
        "status_reason": None,
        "supersedes": quarantined.id,
    }
    if uncertainty_nature is not None:
        update["uncertainty_nature"] = uncertainty_nature
    minted = quarantined.model_copy(update=update)
    validate_transition(None, Status.ACTIVE)
    await insert_particle(session, minted)
    # The vector is copied, not recomputed — its model marker travels with it.
    await copy_particle_embedding(session, quarantined.id, minted.id)
    await update_particle_status(
        session, quarantined.id, Status.SUPERSEDED, StatusReason.CONFLICT_RESOLVED
    )
    log.info(
        "Promoted quarantined particle %s → new ACTIVE %s (supersedes recorded)",
        quarantined.id[:8],
        minted.id[:8],
    )
    return minted
