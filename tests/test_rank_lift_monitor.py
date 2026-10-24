# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The nightly rank-lift monitor.

The pure reductions are tested on synthetic rows shaped like the three ways
the target left the head: a uniform decay of use, a change of regime,
and a named belief that stopped earning use. The Engine half is tested on a
store: the off-head cohort read from utility events against the
record, the run, its record block, and the card the queue raises from it.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from typing import Any

from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.scoring.utility import (
    OUT_OF_BAND,
    LiftRanker,
    MonitorThresholds,
    SweepRow,
    evaluate_rank_lift_monitor,
    plan_alert,
    rank_lift_grid,
    sweep_rank_lift,
    target_transitions,
)
from particles.operations.curation.cards import CardKind, CurationCard
from particles.operations.curation.collect import rank_lift_card
from particles.operations.rank_lift_monitor import (
    PAYLOAD_KEY,
    STORE_WIDE,
    off_head_sessions,
    render_rank_lift_lines,
    run_rank_lift_monitor,
)
from particles.store.event_store import OperatorEventType, record_event
from particles.store.session_exposure_store import (
    SessionExposure,
    ShownBelief,
    record_session_exposure,
)
from particles.store.utility_store import record_utility_events
from tests._observer_scope import belief, source

GRID = rank_lift_grid(0.12, 120)
DEFAULT = MonitorThresholds()


def _row(pid: str, eff: float, r: float = 0.0, key: str | None = None) -> SweepRow:
    return SweepRow(
        particle_id=pid, effective_confidence=eff, reinforcement=r, content_key=key or pid
    )


def _plateau(n: int = 50) -> list[SweepRow]:
    """Beliefs at the confidence cap with no use: the head at ``λ = 0``."""
    return [_row(f"z{i:02d}", 0.7) for i in range(n)]


#: Four beliefs used outside the head, below the cap: they need the lift.
COHORT_R = {"c0": 8.0, "c1": 6.0, "c2": 4.0, "c3": 3.0}


def _cohort(scale: float = 1.0) -> list[SweepRow]:
    return [_row(pid, 0.6, r * scale) for pid, r in COHORT_R.items()]


def _judge(rows: list[SweepRow], lam: float, cohort: set[str] | None = None) -> Any:
    (head,) = evaluate_rank_lift_monitor(
        rows,
        cohort_ids=set(COHORT_R) if cohort is None else cohort,
        grid=GRID,
        head_sizes=[20],
        configured_rank_lift=lam,
        thresholds=DEFAULT,
    )
    return head


# --------------------------------------------------------------- the floor


def test_a_lift_too_small_to_reach_the_cohort_is_below_the_floor() -> None:
    head = _judge(_plateau() + _cohort(), 0.015)

    assert head.verdict == "below_floor"
    assert head.captured == 0 and head.cohort_size == 4
    # Three of four need R >= 4: 0.6 + λ·ln 5 > 0.7.
    assert head.floor.low is not None and 0.06 <= head.floor.low <= 0.065
    assert head.ceiling.high == GRID[-1]


def test_a_lift_that_reaches_three_of_four_is_in_band() -> None:
    head = _judge(_plateau() + _cohort(), 0.1)

    assert head.verdict == "in_band" and head.captured == 4


def test_a_cohort_below_the_minimum_gets_no_verdict() -> None:
    head = _judge(_plateau() + _cohort(), 0.1, cohort={"c0", "c1"})

    assert head.verdict == "cold" and head.cohort_size == 2


def test_credit_earned_only_through_the_head_makes_no_cohort() -> None:
    """Circularity (finding 4): with no off-head use there is nothing to judge."""
    head = _judge(_plateau() + _cohort(), 0.1, cohort=set())

    assert head.verdict == "cold" and head.cohort_size == 0


# ------------------------------------------------------------- the ceiling


def test_a_head_with_no_room_for_unused_beliefs_is_above_the_ceiling() -> None:
    cohort = [_row(pid, 0.7, r) for pid, r in COHORT_R.items()]  # in at any λ > 0
    used = [_row(f"u{i:02d}", 0.66, 1.0) for i in range(15)]  # enter past λ ≈ 0.058
    head = _judge(_plateau() + cohort + used, 0.1)

    assert head.verdict == "above_ceiling"
    assert head.entry_share < 0.25
    assert head.band.low is not None and head.band.high is not None and head.band.high < 0.06


def test_floor_above_ceiling_is_no_band() -> None:
    used = [_row(f"u{i:02d}", 0.69, 1.0) for i in range(15)]  # enter past λ ≈ 0.0145
    head = _judge(_plateau() + _cohort() + used, 0.1)

    assert head.band.empty
    assert head.verdict == "no_band"


# ------------------------------------------------- the three excursion shapes


def test_uniform_decay_raises_the_floor_and_keeps_the_cohort() -> None:
    """Shape (a): a quiet month halves every R; same cohort, the floor moves past λ."""
    lam = 0.08
    fresh = _judge(_plateau() + _cohort(), lam)
    decayed = _judge(_plateau() + _cohort(0.5), lam)

    assert fresh.verdict == "in_band"
    assert decayed.verdict == "below_floor"
    assert decayed.cohort_size == fresh.cohort_size == 4
    assert decayed.floor.low is not None and fresh.floor.low is not None
    assert decayed.floor.low > lam > fresh.floor.low
    # A diversity-only check sees neither state as a problem.
    for rows in (_plateau() + _cohort(), _plateau() + _cohort(0.5)):
        sweep = sweep_rank_lift(rows, grid=GRID, head_sizes=[20], configured_rank_lift=lam)
        assert sweep.configured_admissible


def test_a_regime_change_is_reported_on_the_first_run_after_it() -> None:
    """Shape (b): the signal is replaced; the reading moves out of band and an alert is new."""
    before = {"store-wide@20": "in_band"}
    after = {"store-wide@20": "below_floor"}

    assert plan_alert(after, before, None, "run-2") == "run-2"
    # Carried while it persists, cleared when it ends.
    assert plan_alert(after, after, "run-2", "run-3") == "run-2"
    assert plan_alert(before, after, "run-2", "run-4") is None


def test_a_named_target_losing_its_use_is_disclosed_and_never_judged() -> None:
    """Shape (c): the target leaves the head; the verdict does not move."""
    lam = 0.08
    used = _plateau() + _cohort() + [_row("t", 0.66, 5.0)]
    unused = _plateau() + _cohort() + [_row("t", 0.66, 0.0)]

    def in_heads(rows: list[SweepRow]) -> list[int]:
        rank = [r.particle_id for r in LiftRanker(rows).ordered(lam)].index("t") + 1
        return [n for n in (20,) if rank <= n]

    assert target_transitions({"t": in_heads(unused)}, {"t": in_heads(used)}, [20]) == [
        ("t", 20, "left")
    ]
    assert _judge(used, lam).verdict == _judge(unused, lam).verdict == "in_band"


# --------------------------------------------------------------- the alert


def test_no_alert_without_a_previous_reading() -> None:
    """A standing verdict known at activation raises nothing."""
    assert plan_alert({"store-wide@60": "below_floor"}, None, None, "run-1") is None


def test_a_new_kind_of_out_of_band_verdict_is_a_new_alert() -> None:
    previous = {"store-wide@60": "below_floor"}
    current = {"store-wide@60": "above_ceiling"}

    assert plan_alert(current, previous, "run-1", "run-2") == "run-2"


def test_a_reading_that_turns_cold_clears_the_alert() -> None:
    assert (
        plan_alert({"store-wide@60": "cold"}, {"store-wide@60": "below_floor"}, "r1", "r2") is None
    )


def test_a_target_new_to_the_record_has_no_transition() -> None:
    assert target_transitions({"t": [60]}, {}, [60]) == []
    assert target_transitions({"t": [60]}, None, [60]) == []


# ------------------------------------------------------------------ ranking


def test_the_fast_ranker_orders_exactly_as_the_sweep_does() -> None:
    rng = random.Random(327)
    rows = [
        _row(f"p{i:03d}", rng.choice([0.7, 0.7, 0.65, 0.6]), rng.choice([0.0, 0.0, 1.0, 3.5]))
        for i in range(300)
    ]
    for lam in (0.0, 0.015, 0.05, 0.12):
        brute = sorted(
            rows,
            key=lambda r: (
                -(r.effective_confidence + lam * math.log1p(r.reinforcement)),
                r.particle_id,
            ),
        )
        assert [r.particle_id for r in LiftRanker(rows).ordered(lam)] == [
            r.particle_id for r in brute
        ]


# ----------------------------------------------------- the Engine half


_NOW = datetime.now(UTC)


async def _exposure(session: Any, sid: str, shown: list[str], key: str = "-proj") -> None:
    await record_session_exposure(
        session,
        SessionExposure(
            session_id=sid,
            recorded_at=_NOW - timedelta(days=1),
            source="startup",
            action="full",
            project_key=key,
            observer_scope_applied=True,
            beliefs=[ShownBelief(particle_id=pid, shown_as="projection") for pid in shown],
        ),
    )


async def _store(session: Any) -> dict[str, str]:
    """Shown, off-head-used and unrecorded beliefs, and four unused ones."""
    entry = await source(session, "notes", ["web"])
    ids = {
        name: (await belief(session, f"{name} belief.", entry)).id
        for name in ("shown", "used", "dark", "other", "x1", "x2", "x3")
    }
    when = _NOW - timedelta(days=1)
    for sid in ("s1", "s2"):
        await _exposure(session, sid, [ids["shown"]])
        await record_utility_events(
            session, sid, {ids["shown"]: "literal", ids["used"]: "behavioural"}, observed_at=when
        )
    # A session with no start-context record: unknown, so it counts for nobody.
    await record_utility_events(session, "s3", {ids["dark"]: "literal"}, observed_at=when)
    await record_utility_events(session, "s4", {ids["dark"]: "literal"}, observed_at=when)
    await session.flush()
    return ids


async def test_only_sessions_that_did_not_show_a_belief_count_for_it(db_session: Any) -> None:
    ids = await _store(db_session)

    counts, coverage = await off_head_sessions(
        db_session, set(ids.values()), since=_NOW - timedelta(days=30)
    )

    assert counts == {ids["used"]: 2}
    assert coverage == 4 / 6


async def test_a_run_judges_the_store_and_records_a_block(db_session: Any) -> None:
    ids = await _store(db_session)

    report = await run_rank_lift_monitor(
        db_session,
        run_id="run-1",
        thresholds=MonitorThresholds(capture=1.0, min_cohort=1, entry_share=0.25),
        head_sizes=[4],
    )

    assert [m.particle_id for m in report.cohort] == [ids["used"]]
    (view,) = report.views
    assert view.view == STORE_WIDE
    (head,) = view.heads
    assert head.verdict == "in_band" and head.captured == 1
    assert report.baseline and report.alert is None
    assert report.record_coverage is not None
    payload = report.payload()
    assert payload["views"][0]["heads"][0]["verdict"] == "in_band"


async def test_the_second_run_compares_with_the_first(db_session: Any) -> None:
    await _store(db_session)
    thresholds = MonitorThresholds(capture=1.0, min_cohort=1, entry_share=0.25)
    first = await run_rank_lift_monitor(
        db_session, run_id="run-1", thresholds=thresholds, head_sizes=[4]
    )

    second = await run_rank_lift_monitor(
        db_session,
        run_id="run-2",
        previous=first.payload(),
        thresholds=MonitorThresholds(capture=1.0, min_cohort=1, entry_share=0.9),
        head_sizes=[4],
    )

    assert not second.baseline
    assert second.views[0].heads[0].verdict in OUT_OF_BAND
    assert second.alert == "run-2" and second.alert_new
    line, command = render_rank_lift_lines(second)
    assert "[new]" in line
    assert command.strip().startswith("reproduce: particles memory sweep-rank-lift")


async def test_an_alert_on_the_run_record_becomes_one_card(db_session: Any) -> None:
    assert await rank_lift_card(db_session) is None
    await _store(db_session)
    thresholds = MonitorThresholds(capture=1.0, min_cohort=1, entry_share=0.9)
    report = await run_rank_lift_monitor(
        db_session,
        run_id="run-1",
        previous={"views": [{"view": STORE_WIDE, "heads": []}]},
        thresholds=thresholds,
        head_sizes=[4],
    )
    await record_event(
        db_session,
        actor="memory-consolidate",
        event_type=OperatorEventType.CONSOLIDATION_RUN,
        payload={PAYLOAD_KEY: report.payload()},
    )

    card = await rank_lift_card(db_session)

    assert card is not None and card.kind is CardKind.RANK_LIFT
    assert card.key == "rank_lift:run-1"
    assert card.suggested_gestures == ["snooze"]
    assert "sweep-rank-lift" in card.diagnostic
    again = CurationCard.from_key(card.key)
    assert again.kind is CardKind.RANK_LIFT and again.rank_lift_alert == "run-1"


def test_the_threshold_flags_need_the_monitor_reading() -> None:
    result = CliRunner().invoke(app, ["memory", "sweep-rank-lift", "--capture", "0.5"])

    assert result.exit_code == 2
    assert "--off-head-cohort" in result.output
