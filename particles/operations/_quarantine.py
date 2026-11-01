# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
  The minted particle is asserted at promotion time, as every successor is:
  the store did not hold the claim before.

The two closures that free a held claim with no review, the extraction-record
withdrawal and the held-claim release, share one
further move, :func:`restore_held_claim`: fold the claim into an ACTIVE twin
that already states it, or else promote it.

A held claim is not ACTIVE, so no writer reconciles against it while it is
held. Before any of these moves returns it to the store, the writer asks the
§6.6 search again under the writer lock (:func:`gather_held_search`, then
:func:`conflict_under_lock`), and a claim the search chooses holds it again
behind a new record (:func:`hold_behind`) instead of beside it (security
review 2026-10-31-3, AF1).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.conflict_resolution import build_inconsistency_particle
from particles.core.conflict_review import Demotion, is_quarantined
from particles.core.schema import Particle, ProvenanceRefType, UncertaintyNature
from particles.core.status import Status, StatusReason, validate_transition
from particles.corpus.store import get_entry
from particles.extraction.registry import infer_domain
from particles.ingest.duplicate_suppression import build_duplicate_index
from particles.ingest.pipeline import (
    ConflictWatermark,
    _nearest_conflict,
    _trigger_ref_for,
    load_active_conflict_candidates,
    read_conflict_watermark,
)
from particles.store.particle_store import (
    append_provenance_ref,
    copy_modality_stamp,
    copy_particle_embedding,
    get_embeddings_by_ids,
    insert_particle,
    update_particle_status,
    update_status_reason,
)

__all__ = [
    "HeldSearch",
    "apply_demotion",
    "conflict_under_lock",
    "gather_held_search",
    "hold_behind",
    "is_quarantined",
    "promote_quarantined",
    "promote_unless_contradicted",
    "restore_held_claim",
]

log = logging.getLogger(__name__)


async def apply_demotion(session: AsyncSession, loser_id: str, demotion: Demotion) -> None:
    """Write the demotion :func:`~particles.core.conflict_review.decide_demotion` chose.

    Args:
        session: Active session — the caller commits.
        loser_id: The losing claim.
        demotion: The decided move; ``NONE`` writes nothing.
    """
    match demotion:
        case Demotion.REASON_FLIP:
            await update_status_reason(session, loser_id, StatusReason.CONFLICT_RESOLVED)
        case Demotion.TRANSITION:
            await update_particle_status(
                session, loser_id, Status.PROVENANCE_STALE, StatusReason.CONFLICT_RESOLVED
            )
        case Demotion.NONE:
            log.debug("Loser %s needs no demotion", loser_id)


async def promote_quarantined(
    session: AsyncSession,
    quarantined: Particle,
    *,
    uncertainty_nature: UncertaintyNature | None = None,
) -> Particle:
    """Mint a new ACTIVE particle from a quarantined conflict loser.

    Fresh id, ``supersedes`` → the quarantined row, ``asserted_at`` now;
    content, confidence, provenance, subjects, and embedding carried over
    verbatim. The fresh ``asserted_at`` is what the as-of lens reads as the
    start of the belief, and the successor's ``asserted_at`` is its
    predecessor's retirement instant (rung 1); a copied value
    would date the belief from the quarantined candidate's extraction, before
    any review had accepted it. Decay is unaffected: it reads the
    source's ``content_published_at``, which the provenance carries over. The quarantined
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
        "asserted_at": datetime.now(UTC),
    }
    if uncertainty_nature is not None:
        update["uncertainty_nature"] = uncertainty_nature
    minted = quarantined.model_copy(update=update)
    validate_transition(None, Status.ACTIVE)
    await insert_particle(session, minted)
    # The vector is copied, not recomputed — its model marker travels with it.
    await copy_particle_embedding(session, quarantined.id, minted.id)
    # So does the modality stamp: the value is copied, not reclassified.
    await copy_modality_stamp(session, quarantined.id, minted.id)
    await update_particle_status(
        session, quarantined.id, Status.SUPERSEDED, StatusReason.CONFLICT_RESOLVED
    )
    log.info(
        "Promoted quarantined particle %s → new ACTIVE %s (supersedes recorded)",
        quarantined.id[:8],
        minted.id[:8],
    )
    return minted


async def promote_unless_contradicted(
    session: AsyncSession,
    quarantined: Particle,
    *,
    exclude: Collection[str],
    uncertainty_nature: UncertaintyNature | None = None,
) -> tuple[Particle | None, Particle | None]:
    """Promote a quarantined claim a resolution prefers, unless §6.6 now pairs it.

    Review's ``PREFER_B`` and ``BOTH_VALID`` and the trust cascade settle the
    record the claim was held behind, and no other: nothing reconciled the
    claim against what was committed while it was held. The search runs
    again (:func:`conflict_under_lock`, the caller holding the writer lock),
    with ``exclude`` (the record's members) left out; a claim it chooses keeps
    ``quarantined`` held behind a new record naming that claim instead of
    minting it ACTIVE beside it (security review 2026-10-31-3, AF1).

    Returns:
        ``(minted, None)`` when promoted, ``(None, record)`` when held again.
    """
    search = await gather_held_search(session, quarantined, exclude=exclude)
    conflict = await conflict_under_lock(session, quarantined, search)
    if conflict is not None:
        return None, await hold_behind(session, quarantined, conflict)
    return (
        await promote_quarantined(session, quarantined, uncertainty_nature=uncertainty_nature),
        None,
    )


async def restore_held_claim(session: AsyncSession, held: Particle) -> tuple[str, str]:
    """Return a held claim to the store once its record closed without a review.

    The shape the withdrawal and the release both close
    a record in. A claim an ACTIVE particle already states verbatim with the
    same subjects (the exact-duplicate index) has its provenance
    folded into that particle, and the held row moves to ``SUPERSEDED`` /
    ``CONFLICT_RESOLVED``; otherwise :func:`promote_quarantined` mints a
    successor, which enters without a §6.6 check, and leaves the held row in
    the same state. The caller has checked that ``held`` is quarantined and
    commits.

    Returns:
        ``("folded_into", twin id)`` or ``("successor", minted id)``, the key
        and value the closing event's payload records.
    """
    twin = (await build_duplicate_index(session, [held.content])).find(held)
    if twin is not None:
        for ref in held.provenance:
            await append_provenance_ref(session, twin.id, ref)
        await update_particle_status(
            session, held.id, Status.SUPERSEDED, StatusReason.CONFLICT_RESOLVED
        )
        return "folded_into", twin.id
    successor = await promote_quarantined(session, held)
    return "successor", successor.id


@dataclass(frozen=True)
class HeldSearch:
    """What a writer read, before the lock, about a held claim it may return to the store."""

    embedding: np.ndarray[Any, np.dtype[np.float32]] | None
    watermark: ConflictWatermark
    """The store's change marker, read before the candidates."""
    neighbourhood: frozenset[str]
    """Every claim the search could have chosen (the pipeline's neighbourhood)."""
    exclude: frozenset[str]
    """The claims the held one is never reconciled against here: its record's
    members, which the writer is resolving, and an ACTIVE twin stating it."""


async def gather_held_search(
    session: AsyncSession, held: Particle, *, exclude: Collection[str]
) -> HeldSearch:
    """Read the §6.6 search's inputs for ``held``; writes nothing and calls no model.

    The held claim's stored vector, the store's change marker and the
    neighbourhood the pipeline's conflict search would choose from, so the
    search under the lock (:func:`conflict_under_lock`) reads only that
    neighbourhood and what changed since (security review 2026-10-31-2, AE5).
    """
    skip = set(exclude) | {held.id}
    twin = (await build_duplicate_index(session, [held.content])).find(held)
    if twin is not None:
        skip.add(twin.id)  # a verbatim twin states the claim; it cannot contradict it
    embedding = (await get_embeddings_by_ids(session, [held.id])).get(held.id)
    watermark = await read_conflict_watermark(session)
    pairs = await load_active_conflict_candidates(session)
    _, neighbourhood = await _nearest_conflict(session, held, embedding, pairs, frozenset(skip))
    return HeldSearch(
        embedding=embedding,
        watermark=watermark,
        neighbourhood=neighbourhood,
        exclude=frozenset(skip),
    )


async def conflict_under_lock(
    session: AsyncSession, held: Particle, gathered: HeldSearch
) -> Particle | None:
    """The ACTIVE claim §6.6 would reconcile ``held`` against now, or ``None``.

    Runs with the writer lock held: the store as it stands, bounded by the
    gathered neighbourhood and the claims changed since its watermark
    (``search_store_under_lock``), read whole only when the change journal
    cannot say what changed. The held claim was never reconciled against a
    claim committed while it was held, so a claim found here is one no check
    ever paired with it (security review 2026-10-31-3, AF1).
    """
    # Deferred import: tests/test_writer_lock_guard.py wraps the search at its
    # source module (tests/AGENTS.md § Mocking strategy).
    from particles.ingest import pipeline

    found = await pipeline.search_store_under_lock(
        session, gathered.watermark, gathered.neighbourhood
    )
    pairs = found[0] if found is not None else await load_active_conflict_candidates(session)
    conflict, _ = await _nearest_conflict(
        session, held, gathered.embedding, pairs, gathered.exclude
    )
    return conflict


async def hold_behind(session: AsyncSession, held: Particle, existing: Particle) -> Particle:
    """Keep ``held`` quarantined behind a new INCONSISTENCY record naming ``existing``.

    The fail-closed move when :func:`conflict_under_lock` finds a claim: the
    pair goes to review unprobed, as the write path holds an unprobed pair
    (security review 2026-10-31-1, AD1). ``held`` keeps its status; the
    record is unstamped, since no rung decided it. The caller commits.

    Returns:
        The new record.
    """
    corpus_entry_id, snapshot_id, trigger_ref_type = _trigger_ref_for(held)
    entry = (
        await get_entry(session, corpus_entry_id)
        if trigger_ref_type is ProvenanceRefType.SOURCE
        else None
    )
    record = build_inconsistency_particle(
        existing,
        held,
        corpus_entry_id=corpus_entry_id,
        snapshot_id=snapshot_id,
        trigger_ref_type=trigger_ref_type,
    )
    validate_transition(None, Status.INCONSISTENCY)
    await insert_particle(
        session, record, domain_hint=infer_domain(entry.source_type) if entry else None
    )
    log.info(
        "Held claim %s kept quarantined behind %s (INCONSISTENCY %s): a claim committed "
        "while it was held",
        held.id[:8],
        existing.id[:8],
        record.id[:8],
    )
    return record
