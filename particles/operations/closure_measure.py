# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The closure measure: gather, record, and read back.

Each consolidation run measures two numbers in its zero-LLM ``measure`` pass
and writes them to its ``CONSOLIDATION_RUN`` event under the ``closure`` key
(additive, no payload format bump):

- the contested fraction, from one store-wide :func:`compute_store_contested`
  pass (the same finder the census and lint share), so every run
  reads the whole store whatever its census cadence or delta scope; and
- the window's lifecycle transitions split autonomous versus gesture, read from
  the write-once ``retired_at`` stamps and the operator event log.
  The definitions and the split are the pure
  :mod:`particles.core.closure_measure`.

The window opens where the previous run's window closed (its recorded
``window_end``, or its ``completed_at`` for a run that predates the measure)
and closes when this run measures, so consecutive windows tile the log with no
gap and no overlap. A store's first run opens at the start of the store.

The measure is disclosure only. It changes no status, feeds no score, and moves
no closure default: what weight automated closure may carry is the owner's call,
and this is the number that call reads.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.closure_measure import (
    GESTURE_RETIREMENT_EVENTS,
    TRANSITION_KINDS,
    GestureEvent,
    Retirement,
    contested_fraction,
    share,
    tally_transitions,
)
from particles.operations.abstraction import ABSTRACTION_ACTOR
from particles.operations.lint.contestedness import compute_store_contested
from particles.store.event_store import (
    EventRefKind,
    OperatorEvent,
    OperatorEventType,
    list_events_in_range,
)
from particles.store.particle_store import (
    list_ids_asserted_by_in_range,
    list_retirements_in_range,
)

#: The run-record payload key the measure rides under.
PAYLOAD_KEY = "closure"


class ClosureMeasure(BaseModel):
    """One run's contested fraction and transition split."""

    #: Where the transition window opened; ``None`` = the start of the store.
    window_start: datetime | None = None
    window_end: datetime
    #: ACTIVE beliefs, and those carrying the composed contested badge.
    active: int = 0
    contested: int = 0
    #: Contested beliefs per badge basis; a belief firing two counts under both.
    contested_by_basis: dict[str, int] = Field(default_factory=dict)
    #: kind → {"autonomous": n, "gesture": n}.
    transitions: dict[str, dict[str, int]] = Field(default_factory=dict)
    autonomous_by_reason: dict[str, int] = Field(default_factory=dict)
    gesture_by_actor: dict[str, int] = Field(default_factory=dict)

    @property
    def contested_fraction(self) -> float | None:
        return contested_fraction(self.contested, self.active)

    @property
    def autonomous(self) -> int:
        return sum(k.get("autonomous", 0) for k in self.transitions.values())

    @property
    def gesture(self) -> int:
        return sum(k.get("gesture", 0) for k in self.transitions.values())

    @property
    def autonomous_share(self) -> float | None:
        return share(self.autonomous, self.autonomous + self.gesture)

    def payload(self) -> dict[str, Any]:
        """The ``closure`` payload: the stored counts plus both derived ratios."""
        data = self.model_dump(mode="json")
        data["contested_fraction"] = self.contested_fraction
        data["autonomous"] = self.autonomous
        data["gesture"] = self.gesture
        data["autonomous_share"] = self.autonomous_share
        return data


def measure_from_payload(payload: dict[str, Any] | None) -> ClosureMeasure | None:
    """The measure a run record carries, or ``None`` for a run that predates it."""
    raw = (payload or {}).get(PAYLOAD_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        return ClosureMeasure.model_validate(raw)
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    """SQLite hands timestamps back naive; every window bound here is UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def window_start_after(prior: OperatorEvent | None) -> datetime | None:
    """Where the next window opens: the prior run's ``window_end``, else its end."""
    if prior is None:
        return None
    measured = measure_from_payload(prior.payload)
    if measured is not None:
        return _as_utc(measured.window_end)
    completed = (prior.payload or {}).get("completed_at")
    if isinstance(completed, str):
        try:
            return _as_utc(datetime.fromisoformat(completed))
        except ValueError:
            pass
    return _as_utc(prior.occurred_at)


def _gesture(event: OperatorEvent) -> GestureEvent:
    return GestureEvent(
        event_type=event.event_type.value,
        actor=event.actor,
        particle_ids=tuple(r.ref_id for r in event.refs if r.ref_kind is EventRefKind.PARTICLE),
    )


async def _events(
    session: AsyncSession,
    event_type: OperatorEventType,
    since: datetime | None,
    until: datetime,
) -> list[OperatorEvent]:
    return await list_events_in_range(session, event_type=event_type, since=since, until=until)


async def measure_closure(
    session: AsyncSession, *, window_start: datetime | None, window_end: datetime
) -> ClosureMeasure:
    """Measure the store now and tally the transitions in ``[window_start, window_end)``.

    Reads only, and makes no LLM call. The contested pass is store-wide; the
    transition reads are bounded by the window.
    """
    census = await compute_store_contested(session)
    by_basis: Counter[str] = Counter()
    for badge in census.badges.values():
        by_basis.update(badge.bases)

    retirements = [
        Retirement(particle_id=pid, reason=reason)
        for pid, reason in await list_retirements_in_range(
            session, since=window_start, until=window_end
        )
    ]
    gestures: list[GestureEvent] = []
    for name in sorted(GESTURE_RETIREMENT_EVENTS):
        gestures.extend(
            _gesture(e)
            for e in await _events(session, OperatorEventType(name), window_start, window_end)
        )
    closed = await _events(
        session, OperatorEventType.INCONSISTENCY_CLOSED, window_start, window_end
    )
    accepted = [
        _gesture(e)
        for e in await _events(
            session, OperatorEventType.ABSTRACTION_RESOLVED, window_start, window_end
        )
        if (e.payload or {}).get("resolution") == "accepted"
    ]
    promoted = await list_ids_asserted_by_in_range(
        session, ABSTRACTION_ACTOR, since=window_start, until=window_end
    )
    tally = tally_transitions(
        retirements=retirements,
        gestures=gestures,
        autonomous_closures=len(closed),
        promoted_ids=promoted,
        accepted_promotions=accepted,
    )
    return ClosureMeasure(
        window_start=window_start,
        window_end=window_end,
        active=census.active_count,
        contested=len(census.badges),
        contested_by_basis=dict(sorted(by_basis.items())),
        transitions={k: tally.by_kind[k] for k in TRANSITION_KINDS},
        autonomous_by_reason=tally.autonomous_by_reason,
        gesture_by_actor=tally.gesture_by_actor,
    )


# ---------------------------------------------------------------------------
# History: the series over past run records
# ---------------------------------------------------------------------------


class ClosureHistoryRow(BaseModel):
    """One measured run in the series."""

    event_id: str
    run_at: datetime
    measure: ClosureMeasure


class ClosureHistory(BaseModel):
    """Every measured run by one actor, oldest first."""

    store: str
    actor: str
    runs: list[ClosureHistoryRow] = Field(default_factory=list)
    #: Runs by the same actor whose record predates the measure.
    unmeasured_runs: int = 0

    def payload(self) -> dict[str, Any]:
        """The ``--format json`` shape: each row with both derived ratios."""
        return {
            "store": self.store,
            "actor": self.actor,
            "unmeasured_runs": self.unmeasured_runs,
            "runs": [
                {"event_id": r.event_id, "run_at": r.run_at.isoformat(), **r.measure.payload()}
                for r in self.runs
            ],
        }


async def closure_history(session: AsyncSession, *, store: str, actor: str) -> ClosureHistory:
    """Read the measure back from every ``CONSOLIDATION_RUN`` this actor recorded.

    The interactive audit writes the same event type under its own actor and
    never measures, so it is filtered out rather than counted as unmeasured.
    """
    history = ClosureHistory(store=store, actor=actor)
    events = await list_events_in_range(session, event_type=OperatorEventType.CONSOLIDATION_RUN)
    for event in events:
        if event.actor != actor:
            continue
        measured = measure_from_payload(event.payload)
        if measured is None:
            history.unmeasured_runs += 1
            continue
        history.runs.append(
            ClosureHistoryRow(
                event_id=event.event_id, run_at=_as_utc(event.occurred_at), measure=measured
            )
        )
    return history


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_closure_lines(measure: ClosureMeasure, previous: ClosureMeasure | None) -> list[str]:
    """The run report's two lines: the contested fraction, then the transition split."""
    bases = ", ".join(f"{b} {n}" for b, n in measure.contested_by_basis.items())
    contested = (
        f"  contested        {measure.contested} of {measure.active} active beliefs "
        f"({_percent(measure.contested_fraction)})"
    )
    if bases:
        contested += f" [{bases}]"
    if previous is not None:
        contested += f"; previous run {_percent(previous.contested_fraction)}"
    since = (
        "since the previous run"
        if measure.window_start is not None
        else "since the store began (first measured run)"
    )
    total = measure.autonomous + measure.gesture
    transitions = (
        f"  transitions      {total} {since}: {measure.autonomous} autonomous, "
        f"{measure.gesture} by gesture"
    )
    if total:
        transitions += f" ({_percent(measure.autonomous_share)} autonomous)"
    return [contested, transitions]


def render_closure_history(history: ClosureHistory) -> str:
    """The ``--history`` table: one row per measured run, oldest first."""
    lines = [
        f"Contested fraction and transition split per consolidation run, store '{history.store}'",
        "",
    ]
    if not history.runs:
        lines.append(
            "  No consolidation run has recorded the measure yet; the next "
            "'particles memory consolidate' records the first."
        )
    else:
        header = (
            f"  {'run (UTC)':<16}  {'active':>7}  {'contested':>9}  {'fraction':>8}  "
            f"{'transitions':>11}  {'autonomous':>10}  {'gesture':>7}  {'auto share':>10}"
        )
        lines.append(header)
        for row in history.runs:
            m = row.measure
            lines.append(
                f"  {row.run_at:%Y-%m-%d %H:%M}  {m.active:>7}  {m.contested:>9}  "
                f"{_percent(m.contested_fraction):>8}  {m.autonomous + m.gesture:>11}  "
                f"{m.autonomous:>10}  {m.gesture:>7}  {_percent(m.autonomous_share):>10}"
            )
    if history.unmeasured_runs:
        lines.append("")
        lines.append(
            f"  {history.unmeasured_runs} earlier run(s) by {history.actor} predate the measure."
        )
    return "\n".join(lines) + "\n"
