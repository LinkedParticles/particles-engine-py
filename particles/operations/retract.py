"""Corpus retract operation.

The non-destructive sibling of ``corpus delete``: bulk-transition every *live*
particle sourced from a corpus entry to ``RETRACTED`` (reason
``SOURCE_RETRACTED``) while leaving the corpus entry, its snapshots, and the
particles' subjects intact — so the audit trail *"we believed X based on
source Y, then Y was retracted"* survives.

Lives in the operation layer (not the CLI body) so the event fires and the
transitions run identically regardless of front-end. Caller commits.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.status import Status, StatusReason
from particles.store.event_store import EventRefKind, OperatorEventType, record_event
from particles.store.particle_store import get_particles_for_entry, update_particle_status

# Only these statuses are the "live belief" a source retraction withdraws.
# SUPERSEDED / PROVENANCE_STALE / RETRACTED are skipped (not live; and
# SUPERSEDED -> RETRACTED is not a permitted §6.6 transition anyway).
_LIVE = frozenset({Status.ACTIVE, Status.INCONSISTENCY})


class RetractionItem(BaseModel):
    """One particle in a retraction plan."""

    particle_id: str
    status: Status


class RetractionPlan(BaseModel):
    """What a retraction would do — read-only, no DB writes."""

    entry_id: str
    to_retract: list[RetractionItem] = []
    skipped: dict[str, int] = {}  # status value -> count of skipped particles


class RetractionResult(BaseModel):
    """Outcome of an applied retraction."""

    entry_id: str
    retracted_ids: list[str] = []
    skipped: dict[str, int] = {}


async def plan_retraction(session: AsyncSession, entry_id: str) -> RetractionPlan:
    """Partition the entry's particles into live (to retract) vs skipped.

    Read-only: makes no writes, so it backs both ``--dry-run`` and the
    pre-apply plan display.
    """
    particles = await get_particles_for_entry(session, entry_id)
    to_retract: list[RetractionItem] = []
    skipped: Counter[str] = Counter()
    for p in particles:
        if p.status in _LIVE:
            to_retract.append(RetractionItem(particle_id=p.id, status=p.status))
        else:
            skipped[p.status.value] += 1
    return RetractionPlan(entry_id=entry_id, to_retract=to_retract, skipped=dict(skipped))


async def retract_entry(
    session: AsyncSession,
    entry_id: str,
    *,
    reason: str | None = None,
    actor: str = "corpus-retract",
) -> RetractionResult:
    """Retract every live particle from ``entry_id``; record one event.

    Each transition goes through ``set_particle_status`` so the §6.6 validator
    and the co-evidential relation cleanup run. The corpus entry and
    its snapshots are not touched. Idempotent: a second run finds nothing live.
    Caller commits.
    """
    plan = await plan_retraction(session, entry_id)
    retracted_ids: list[str] = []
    for item in plan.to_retract:
        await update_particle_status(
            session,
            item.particle_id,
            Status.RETRACTED,
            StatusReason.SOURCE_RETRACTED,
        )
        retracted_ids.append(item.particle_id)

    if retracted_ids:
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.SOURCE_RETRACTED,
            reason=reason,
            refs=[
                (EventRefKind.CORPUS_ENTRY, entry_id),
                *[(EventRefKind.PARTICLE, pid) for pid in retracted_ids],
            ],
            payload={"retracted": len(retracted_ids), "skipped": plan.skipped},
        )
    return RetractionResult(entry_id=entry_id, retracted_ids=retracted_ids, skipped=plan.skipped)
