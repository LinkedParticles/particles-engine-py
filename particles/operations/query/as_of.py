"""As-of read lens — visibility predicate + retirement-instant ladder.

The Engine helper behind ``QueryRequest.as_of``: an :class:`AsOfView` is
loaded **once per query** (per store — the shape ``DecayPolicy`` /
``TrustPolicy`` established) and evaluates each widened-load candidate against
the visibility predicate — *believed at T*:

1. **It had been asserted:** ``asserted_at <= T`` (naive timestamps assumed
   UTC, matching the existing query-path comparison).
2. **It had not yet been retired:** the particle is currently ``ACTIVE``, or
   its retirement instant R — stored (§2a) or reconstructed (§2b) —
   satisfies ``R > T``.

``INCONSISTENCY`` particles are never visible (they are never ACTIVE; the
§6.6 ledger is a different surface). Born-retired rows (quarantine
losers, ``status_reason = CONFLICT_PENDING``) were never believed — never
visible and never counted. ``valid_until`` is evaluated against T, not now:
a claim valid until 2007 was in force in 2000.

The retirement instant resolves through the §2b ladder, each rung exact:

0. **Stored** — the §2a ``retired_at`` storage column (every post-migration
   retirement lands here).
1. **Successor pointer** — a particle whose ``supersedes`` names P; its
   ``asserted_at`` is P's retirement instant (retire + assert share one
   transaction on every pointer-setting path).
2. **Operator event** — the latest ``PARTICLE_SUPERSEDED`` /
   ``PARTICLE_RETRACTED`` / ``SOURCE_RETRACTED`` / ``REVIEW_RESOLVED`` event
   holding a ref to P (indexed refs table, batched per query).
3. **Validity expiry** — ``status_reason = VALIDITY_EXPIRED`` retires at the
   stored ``valid_until``, by definition.
4. **Otherwise unknown → fail-closed**: never visible at any T, and the
   response **discloses the exclusion count**. An audit lens that
   manufactures instants is worse than one that discloses a gap.

Nothing here is stored; the lens never transitions anything (the status
machine is untouched) — it is a pure read-time parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import AsOfNote, AsOfSuccessor, Particle
from particles.core.status import Status, StatusReason
from particles.store.event_store import EventRefKind, OperatorEventRefRow, OperatorEventRow
from particles.store.particle_store import ParticleRow

#: Operator event types whose refs date a particle's retirement exactly
#: (rung 2). Automated pipeline/lint transitions do not emit
#: events (inclusion criterion), so the log covers deliberate acts.
RETIREMENT_EVENT_TYPES: tuple[str, ...] = (
    "PARTICLE_SUPERSEDED",
    "PARTICLE_RETRACTED",
    "SOURCE_RETRACTED",
    "REVIEW_RESOLVED",
)

#: The §2b ladder rung that dated a retirement — carried on the AsOfNote so
#: every displayed instant is itself auditable.
RetirementBasis = Literal["stored", "successor", "event", "valid_until"]


def ensure_utc(dt: datetime) -> datetime:
    """Assume UTC for a naive timestamp (the store's existing normalization)."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def is_once_believed_retirement(status: Status, status_reason: StatusReason | None) -> bool:
    """True when a non-ACTIVE row records a belief that was once held.

    The shared exclusion-set helper for the query disclosure count and the
    ``UNDATED_RETIREMENT`` lint finding: born-retired rows
    quarantine losers (``status_reason = CONFLICT_PENDING``) and INCONSISTENCY
    records — were never believed, so their lack of a retirement instant is
    correct, not a gap; counting them would overstate the store's undatable
    history.
    """
    if status in (Status.ACTIVE, Status.INCONSISTENCY):
        return False
    return status_reason is not StatusReason.CONFLICT_PENDING


@dataclass(frozen=True)
class SuccessorRef:
    """A ``supersedes``-pointer successor, as loaded for the rung-1 map."""

    id: str
    content: str
    asserted_at: datetime


@dataclass(frozen=True)
class RetirementResolution:
    """One resolved retirement instant plus the ladder rung that answered."""

    retired_at: datetime
    basis: RetirementBasis


@dataclass(frozen=True)
class AsOfEvaluation:
    """The visibility verdict for one candidate at the reference instant."""

    visible: bool
    #: The supersession crossing for a visible hit retired after T; None for a
    #: still-ACTIVE hit (and always None when not visible).
    note: AsOfNote | None
    #: True when the candidate was excluded fail-closed because its retirement
    #: instant is not reconstructible (rung 4) — feeds the disclosure count.
    excluded_undatable: bool


@dataclass(frozen=True)
class RetirementIndex:
    """The per-store rung 1–2 lookup maps, loaded once per query.

    ``successor_by_predecessor`` maps each superseded particle id to its
    earliest ``supersedes``-pointer successor; ``event_retired_at`` maps a
    particle id to the latest retirement-dating operator event instant.
    """

    successor_by_predecessor: dict[str, SuccessorRef]
    event_retired_at: dict[str, datetime]

    def resolve(
        self,
        particle_id: str,
        *,
        status_reason: StatusReason | None,
        valid_until: datetime | None,
        stored_retired_at: datetime | None,
    ) -> RetirementResolution | None:
        """Run the §2b ladder for one retired particle; ``None`` = rung 4 (unknown)."""
        # Rung 0 — stored (exact): every post-migration retirement lands here.
        if stored_retired_at is not None:
            return RetirementResolution(ensure_utc(stored_retired_at), "stored")
        # Rung 1 — successor pointer (exact).
        successor = self.successor_by_predecessor.get(particle_id)
        if successor is not None:
            return RetirementResolution(ensure_utc(successor.asserted_at), "successor")
        # Rung 2 — operator event (exact).
        event_at = self.event_retired_at.get(particle_id)
        if event_at is not None:
            return RetirementResolution(ensure_utc(event_at), "event")
        # Rung 3 — validity expiry (exact by definition).
        if status_reason is StatusReason.VALIDITY_EXPIRED and valid_until is not None:
            return RetirementResolution(ensure_utc(valid_until), "valid_until")
        # Rung 4 — unknown: fail-closed, the caller discloses.
        return None


@dataclass(frozen=True)
class AsOfView:
    """One store's as-of lens for a single reference instant."""

    as_of: datetime  # timezone-aware UTC
    index: RetirementIndex

    def _successor_payload(self, particle_id: str) -> AsOfSuccessor | None:
        successor = self.index.successor_by_predecessor.get(particle_id)
        if successor is None:
            return None
        return AsOfSuccessor(
            id=successor.id,
            content=successor.content,
            asserted_at=ensure_utc(successor.asserted_at),
        )

    def evaluate(self, particle: Particle, stored_retired_at: datetime | None) -> AsOfEvaluation:
        """Apply the §1 visibility predicate to one candidate.

        ``stored_retired_at`` is the §2a storage column threaded beside the
        model by the widened loader (it is not a ``Particle`` field).
        """
        not_visible = AsOfEvaluation(visible=False, note=None, excluded_undatable=False)

        # §1 condition 1: it had been asserted by T.
        if ensure_utc(particle.asserted_at) > self.as_of:
            return not_visible

        # INCONSISTENCY records are never visible; born-retired quarantine
        # losers were never believed — neither is counted as undatable.
        if particle.status is Status.INCONSISTENCY:
            return not_visible
        if particle.status_reason is StatusReason.CONFLICT_PENDING:
            return not_visible

        # ``valid_until`` is evaluated against T, not now: a claim whose
        # validity had already expired by T was not in force at T.
        if particle.valid_until is not None and ensure_utc(particle.valid_until) < self.as_of:
            return not_visible

        if particle.status is Status.ACTIVE:
            return AsOfEvaluation(visible=True, note=None, excluded_undatable=False)

        # §1 condition 2 for a retired particle: believed at T iff R > T.
        resolution = self.index.resolve(
            particle.id,
            status_reason=particle.status_reason,
            valid_until=particle.valid_until,
            stored_retired_at=stored_retired_at,
        )
        if resolution is None:
            # Rung 4: unknown retirement instant → fail-closed + disclosed.
            return AsOfEvaluation(visible=False, note=None, excluded_undatable=True)
        if resolution.retired_at <= self.as_of:
            return not_visible
        note = AsOfNote(
            status=particle.status,
            status_reason=particle.status_reason,
            retired_at=resolution.retired_at,
            basis=resolution.basis,
            successor=self._successor_payload(particle.id),
        )
        return AsOfEvaluation(visible=True, note=note, excluded_undatable=False)


async def load_retirement_index(session: AsyncSession) -> RetirementIndex:
    """Load one store's rung 1–2 maps: successor pointers + retirement events.

    One in-memory successor map (``SELECT id, content, supersedes, asserted_at
    WHERE supersedes IS NOT NULL`` — no new index; the store's stated scale is
    ≤10⁵ particles) and one batched lookup over the indexed
    ``operator_event_refs`` table. Shared by the query lens and the
    ``UNDATED_RETIREMENT`` lint finding.
    """
    successor_by_predecessor: dict[str, SuccessorRef] = {}
    result = await session.execute(
        select(
            ParticleRow.id,
            ParticleRow.content,
            ParticleRow.supersedes,
            ParticleRow.asserted_at,
        ).where(ParticleRow.supersedes.isnot(None))
    )
    for succ_id, content, predecessor_id, asserted_at in result.all():
        current = successor_by_predecessor.get(predecessor_id)
        # Multiple successors are unexpected but legal data; the retirement
        # instant is the *first* replacement.
        if current is None or ensure_utc(asserted_at) < ensure_utc(current.asserted_at):
            successor_by_predecessor[predecessor_id] = SuccessorRef(
                id=succ_id, content=content, asserted_at=asserted_at
            )

    event_retired_at: dict[str, datetime] = {}
    result = await session.execute(
        select(OperatorEventRefRow.ref_id, OperatorEventRow.occurred_at)
        .join(OperatorEventRow, OperatorEventRow.event_id == OperatorEventRefRow.event_id)
        .where(
            OperatorEventRefRow.ref_kind == EventRefKind.PARTICLE.value,
            OperatorEventRow.event_type.in_(RETIREMENT_EVENT_TYPES),
        )
    )
    for ref_id, occurred_at in result.all():
        occurred = ensure_utc(occurred_at)
        current_at = event_retired_at.get(ref_id)
        # "The latest such event": a re-touched particle dates by the last
        # deliberate act that could have retired it.
        if current_at is None or occurred > current_at:
            event_retired_at[ref_id] = occurred

    return RetirementIndex(
        successor_by_predecessor=successor_by_predecessor,
        event_retired_at=event_retired_at,
    )


async def load_as_of_view(session: AsyncSession, as_of: datetime) -> AsOfView:
    """Snapshot one store's as-of lens for the reference instant ``as_of``."""
    return AsOfView(as_of=ensure_utc(as_of), index=await load_retirement_index(session))
