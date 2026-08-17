"""Composed contested badge — one basis-carrying disclosure per claim.

A claim renders *contested* iff at least one of three named bases fires (§1):

- ``stance`` — the claim's query-time stance distribution (over its
  CO_EVIDENTIAL group, dangling edges excluded) contains at least one
  ``DISPUTES`` position. Endorsements alone never fire — agreement is not
  contest. When this basis fires, the M6 unverified-holder caveat
  travels with the badge (the MUST that names this consumer).
- ``divergence`` — the claim's :class:`ContestednessReading` spread
  is at least ``contestedness.callout_threshold`` (the existing knob; the badge
  gate, the prose callout, and the lint "highly contested" count stay a single
  threshold).
- ``inconsistency`` — an open INCONSISTENCY particle references
  the claim (``get_inconsistency_backrefs``). The badge subsumes the existing
  marker; the INCONSISTENCY id remains the basis's drill-down payload.

Absence semantics (§3): a basis that *cannot be measured* is absent from the
composition, not a non-firing vote — divergence is absent below two policies.
A claim with no available basis fired carries no badge (``None``), never an
explicit "uncontested".

Invariants (§4): the badge is computed at read time and never stored; it MUST
NOT feed ``effective_confidence``, ranking, ``min_confidence`` filtering, or
§6.6 conflict resolution — disclosure, not discount.

Cost (§5): the inconsistency backref is one status scan, stance presence is a
relation lookup over the rendered claims, and divergence reuses
``load_member_policies`` / ``compute_contestedness`` — free in the common
zero-lens store where the basis is absent.
"""

from __future__ import annotations

from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ContestedBadge, ContestednessReading, Particle, RelationType
from particles.core.stance import stance_holder
from particles.core.status import Status
from particles.store.particle_store import get_inconsistency_backrefs, get_particle
from particles.store.relation_store import get_co_evidential_group, get_incoming

from .stance import AGREEMENT_CAVEAT

#: The three named basis labels — the badge's only vocabulary.
BasisLabel = Literal["stance", "divergence", "inconsistency"]


def compose_badge(
    *,
    has_dispute: bool,
    reading: ContestednessReading | None,
    inconsistency_id: str | None,
) -> ContestedBadge | None:
    """Apply the §2 basis gates and compose one claim's badge (pure).

    ``reading=None`` means the divergence basis is *absent* (unmeasurable, §3)
    — distinct from a present reading below threshold, which is a non-firing
    basis. Returns ``None`` when no available basis fired.
    """
    bases: list[BasisLabel] = []
    if has_dispute:
        bases.append("stance")
    if reading is not None and reading.spread >= get_config().contestedness.callout_threshold:
        bases.append("divergence")
    if inconsistency_id is not None:
        bases.append("inconsistency")
    if not bases:
        return None
    return ContestedBadge(
        bases=bases,
        inconsistency_id=inconsistency_id,
        caveat=AGREEMENT_CAVEAT if has_dispute else None,
    )


async def _dispute_presence(session: AsyncSession, targets: list[Particle]) -> list[bool]:
    """Per-target: does the query-time stance distribution contain ≥1 DISPUTES?

    Mirrors ``stance.compute_stance_distribution``'s membership rules — the
    ``DISPUTES`` edges into the target's CO_EVIDENTIAL group, keeping only
    ACTIVE stance particles that carry a ``stance:holder`` attribution (a
    dangling or holder-less edge contributes no position) — but
    answers only the presence question, so it never scores the stances.
    """
    per_target: list[set[str]] = []
    for t in targets:
        ids: set[str] = set()
        for member in await get_co_evidential_group(session, t.id):
            ids.update(await get_incoming(session, member, RelationType.DISPUTES))
        per_target.append(ids)

    live: set[str] = set()
    for sid in {sid for ids in per_target for sid in ids}:
        p = await get_particle(session, sid)
        if p is not None and p.status == Status.ACTIVE and stance_holder(p) is not None:
            live.add(sid)
    return [bool(ids & live) for ids in per_target]


async def compute_contested_badges(
    session: AsyncSession,
    targets: list[Particle],
    *,
    backrefs: dict[str, str] | None = None,
    readings: list[ContestednessReading] | None = None,
) -> list[ContestedBadge | None]:
    """Compose the per-claim contested badge for each target.

    ``badges[i]`` is the badge for ``targets[i]``, or ``None`` when no
    available basis fired. ``backrefs`` may be supplied when the caller has
    already paid the INCONSISTENCY scan (digest / projection). ``readings``
    may be supplied when the caller already computed the readings
    (``include_contestedness``); an *empty* list means the divergence basis is
    absent (fewer than two policies, §3), while ``None`` means "not computed"
    and the divergence path runs here — skipped for free when the viewer has
    fewer than two policies.
    """
    if not targets:
        return []
    if backrefs is None:
        backrefs = await get_inconsistency_backrefs(session)
    disputes = await _dispute_presence(session, targets)

    if readings is None:
        from .contestedness import compute_contestedness, load_member_policies

        members = await load_member_policies(session)
        # Returns [] below two policies — the §3 absent divergence basis.
        readings = await compute_contestedness(session, targets, members)
    per_reading: list[ContestednessReading | None] = (
        list(readings) if len(readings) == len(targets) else [None] * len(targets)
    )

    return [
        compose_badge(
            has_dispute=has_dispute,
            reading=reading,
            inconsistency_id=backrefs.get(t.id),
        )
        for t, has_dispute, reading in zip(targets, disputes, per_reading, strict=True)
    ]
