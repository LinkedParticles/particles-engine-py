"""Store for per-belief utility evidence — the usefulness lens.

The *utility evidence* half of two-quantity split: per-belief,
store-local — not a policy, not portable (the portable *judgment* half is the
lens ``utility_rules`` layer). A **utility event** records that a belief earned
its place, and reaches this table through one of two channels:

- ``source = "mined"`` (§1/§3) — the agent demonstrably *acted* on the
  belief in a harvested session. Credit action, not attention.
- ``source = "explicit"`` — an operator said so, with
  ``particles memory useful <id>``. This channel exists because action mining is
  structurally blind to prohibitive and stance guidelines: compliance with
  "never do X" is the *absence* of a tool call, and a miner reading the actions
  that happened cannot observe the action that didn't.

**This table is a derived index, not the system of record for the explicit
channel.** Every explicit row is reconstructible from its ``BELIEF_MARKED_USEFUL``
operator event, which is append-only and authoritative. That is what
lets :func:`clear_utility_events` stay a blunt reset: ``rebuild-utility``
re-derives *both* channels — mined rows from the harvested transcripts, explicit
rows from the event log — rather than special-casing preservation.

Events are keyed on ``(particle_id, session_id)`` so re-mining a session is
idempotent (the corpus is the harvest state; no side-car
high-water mark to drift). The explicit channel reuses that same natural key by
synthesising :func:`explicit_credit_key` — ``explicit:<actor>:<date>`` — which
is what bounds it to **one credit per belief per principal per day**. :func:`get_reinforcement_scores` reads the event ages and channels and
returns the weighted reinforcement count (``core.scoring.utility``) the
query-time :class:`~particles.operations.query.utility_policy.UtilityPolicy`
turns into an additive rank-lift.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import cast

from sqlalchemy import CursorResult, DateTime, String, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.config import get_config
from particles.core.scoring.utility import reinforcement_score
from particles.db import Base

SOURCE_MINED = "mined"
"""Channel for events produced by the transcript action-miner."""

SOURCE_EXPLICIT = "explicit"
"""Channel for events produced by the explicit operator gesture."""


def explicit_credit_key(actor: str, when: datetime | date | None = None) -> str:
    """The synthetic ``session_id`` for an explicit gesture: ``explicit:<actor>:<date>``.

    The rate bound, expressed as a natural key rather than as a
    counter. Because the table is already unique on ``(particle_id,
    session_id)``, folding the principal and the UTC date into that key makes a
    second press on the same belief by the same principal on the same day a
    no-op *by construction* — there is no window to reset and no counter to
    drift. Pressing on N distinct days still accrues N credits, which is the
    genuine signal (repeated endorsement over time), and decay is what keeps it
    honest.

    Args:
        actor: The pressing principal (e.g. ``cli:memory-useful``).
        when: The moment of the gesture; defaults to now. A datetime is reduced
            to its UTC calendar date.
    """
    if when is None:
        stamp = datetime.now(UTC).date()
    elif isinstance(when, datetime):
        stamp = (
            (when if when.tzinfo is not None else when.replace(tzinfo=UTC)).astimezone(UTC).date()
        )
    else:
        stamp = when
    return f"{SOURCE_EXPLICIT}:{actor}:{stamp.isoformat()}"


class UtilityEventRow(Base):
    """One utility event: ``particle_id`` earned credit in ``session_id``.

    ``source`` (:data:`SOURCE_MINED` | :data:`SOURCE_EXPLICIT`) is the channel
    discriminator — it decides the event's weight at read time and scopes the
    miner's re-mine delete so a re-mine can never destroy an operator's gesture.
    ``match_basis`` (``"literal"`` | ``"behavioural"``) records how
    the *miner* matched and is ``NULL`` for explicit rows: it names a miner tier
    and has no meaning off the miner, so it is not overloaded with a third value.

    The ``(particle_id, session_id)`` uniqueness is enforced both at write time
    (idempotent re-mine; the daily credit key) and by a DB
    constraint, so a belief credited in N distinct sessions accrues N events
    while a repeat within one session — or one operator-day — is a no-op.
    """

    __tablename__ = "utility_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    particle_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    match_basis: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False, default=SOURCE_MINED)


def channel_weight(source: str, explicit_weight: float) -> float:
    """The read-time weight one event contributes, by channel."""
    return explicit_weight if source == SOURCE_EXPLICIT else 1.0


async def record_utility_events(
    session: AsyncSession,
    session_id: str,
    events: dict[str, str],
    observed_at: datetime | None = None,
) -> int:
    """Record *mined* utility events for one harvested session, idempotently.

    ``events`` maps ``particle_id → match_basis``. Every existing **mined** event
    for ``session_id`` is cleared first, so re-mining the same session replaces
    its contribution rather than duplicating it (idempotence).
    Returns the number of events written.

    The delete is scoped to :data:`SOURCE_MINED`: an unscoped
    delete-then-insert would silently destroy an explicit operator credit that
    happened to share a ``session_id``, which is the single most damaging way a
    second event producer could have been bolted onto this table.

    Args:
        session: Active store session.
        session_id: The harvested session the actions came from.
        events: ``{particle_id: "literal" | "behavioural"}`` for each belief the
            agent demonstrably acted on in that session.
        observed_at: Event timestamp (the session's time); defaults to now.
    """
    await session.execute(
        delete(UtilityEventRow).where(
            UtilityEventRow.session_id == session_id,
            UtilityEventRow.source == SOURCE_MINED,
        )
    )
    ts = observed_at or datetime.now(UTC)
    rows = [
        UtilityEventRow(
            particle_id=pid,
            session_id=session_id,
            observed_at=ts,
            match_basis=basis,
            source=SOURCE_MINED,
        )
        for pid, basis in events.items()
    ]
    session.add_all(rows)
    await session.flush()
    return len(rows)


async def record_explicit_utility_event(
    session: AsyncSession,
    particle_id: str,
    credit_key: str,
    observed_at: datetime | None = None,
) -> bool:
    """Credit one belief from the explicit operator channel, idempotently.

    Returns ``True`` when a new credit was written and ``False`` when this
    ``(particle_id, credit_key)`` was already credited — the daily
    bound. The caller records the operator event either way: the log is the
    audit trail of what the operator *did*, this table is what it was *worth*.

    Deliberately not routed through :func:`record_utility_events`: that path
    deletes every row for the session key before inserting, which for a
    per-operator-day key would wipe the day's earlier credits on other beliefs.

    The check-then-insert is backed by the ``(particle_id, session_id)`` unique
    index, so two concurrent presses race to the same outcome rather than
    double-crediting: the loser's ``IntegrityError`` is the same "already
    credited today" answer the check would have given, and is reported as such.

    Args:
        session: Active store session.
        particle_id: The belief being credited.
        credit_key: The synthetic session key from :func:`explicit_credit_key`.
        observed_at: Event timestamp; defaults to now.
    """
    existing = (
        await session.execute(
            select(UtilityEventRow.id).where(
                UtilityEventRow.particle_id == particle_id,
                UtilityEventRow.session_id == credit_key,
            )
        )
    ).first()
    if existing is not None:
        return False
    savepoint = await session.begin_nested()
    session.add(
        UtilityEventRow(
            particle_id=particle_id,
            session_id=credit_key,
            observed_at=observed_at or datetime.now(UTC),
            match_basis=None,
            source=SOURCE_EXPLICIT,
        )
    )
    try:
        await session.flush()
    except IntegrityError:
        # Lost the race. Roll back only this insert — the caller's operator
        # event still belongs in the transaction, since the gesture happened.
        await savepoint.rollback()
        return False
    await savepoint.commit()
    return True


async def get_reinforcement_scores(
    session: AsyncSession,
    particle_ids: list[str],
    half_life_uses_days: float,
    now: datetime | None = None,
    explicit_weight: float | None = None,
) -> dict[str, float]:
    """Return ``{particle_id: reinforcement_score}`` for the given beliefs.

    The recency- and channel-weighted event count
    (``core.scoring.utility.reinforcement_score``) under the resolved
    reinforcement half-life. A belief with no utility events is absent from the
    result (the caller treats absence as ``0.0`` → ``+0`` bonus, the cold-start
    posture). One query over the id set, so projection / digest
    scoring stays free of per-particle round trips.

    Args:
        session: Active store session.
        particle_ids: The beliefs to score.
        half_life_uses_days: Reinforcement half-life (a lens judgment).
        now: Override for current time (used in tests).
        explicit_weight: What one explicit gesture is worth relative to
            one mined event; defaults to ``utility.explicit_weight`` from config.
    """
    if not particle_ids:
        return {}
    weight = (
        explicit_weight if explicit_weight is not None else get_config().utility.explicit_weight
    )
    rows = (
        (
            await session.execute(
                select(
                    UtilityEventRow.particle_id,
                    UtilityEventRow.observed_at,
                    UtilityEventRow.source,
                ).where(UtilityEventRow.particle_id.in_(particle_ids))
            )
        )
        .tuples()
        .all()
    )
    by_particle: dict[str, list[tuple[datetime, float]]] = {}
    for pid, observed, source in rows:
        by_particle.setdefault(pid, []).append((observed, channel_weight(source, weight)))
    return {
        pid: reinforcement_score(events, half_life_uses_days, now)
        for pid, events in by_particle.items()
    }


async def clear_utility_events(session: AsyncSession, source: str | None = None) -> int:
    """Delete utility events (the ``rebuild-utility`` backfill's reset step).

    With ``source`` omitted this clears **every** channel, which is safe
    precisely because both are re-derivable: mined rows from the harvested
    transcripts, explicit rows from the append-only ``BELIEF_MARKED_USEFUL``
    operator event log. Pass a channel to scope the reset.
    """
    stmt = delete(UtilityEventRow)
    if source is not None:
        stmt = stmt.where(UtilityEventRow.source == source)
    result = cast("CursorResult[object]", await session.execute(stmt))
    await session.flush()
    return result.rowcount or 0
