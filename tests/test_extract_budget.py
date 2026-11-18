# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Pass 1's nightly extraction budget and the transcript order.

Covers ``particles/operations/extract_budget.py`` and the round loop and queue
order it feeds in ``particles/operations/consolidation.py``. The pooled pass,
the read plan and the usage scope are faked: no model is called, and spend is
whatever the fake pass says a round cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import ProviderSelection, get_config
from particles.core.spend_budget import RoundCandidate
from particles.corpus.store import PendingSnapshot
from particles.llm.batch_budget import BatchWaitBudget, batch_wait_budget
from particles.operations import consolidation as consolidation_mod
from particles.operations import consolidation_plan as plan_mod
from particles.operations import extract_budget as budget_mod
from particles.operations.consolidation import ConsolidationReport

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def _snap(
    entry: str,
    sid: str,
    minutes: int,
    *,
    attempts: int = 0,
    source_type: str | None = "CONVERSATION",
    tags: frozenset[str] = frozenset(),
) -> PendingSnapshot:
    return PendingSnapshot(
        entry,
        sid,
        T0 + timedelta(minutes=minutes),
        attempts,
        tags=tags,
        source_type=source_type,
    )


# ---------------------------------------------------------------------------
# The order
# ---------------------------------------------------------------------------


class TestTranscriptOrder:
    def test_transcripts_are_read_newest_session_first_each_session_oldest_first(self) -> None:
        pending = [
            _snap("old", "o1", 0),
            _snap("old", "o2", 10),
            _snap("new", "n1", 20),
            _snap("new", "n2", 30),
            _snap("mid", "m1", 25),
        ]
        ordered = consolidation_mod._prioritised(pending)
        assert [p.snapshot_id for p in ordered] == ["n1", "n2", "m1", "o1", "o2"]

    def test_least_tried_still_comes_first(self) -> None:
        pending = [_snap("tried", "t1", 50, attempts=2), _snap("fresh", "f1", 0)]
        ordered = consolidation_mod._prioritised(pending)
        assert [p.snapshot_id for p in ordered] == ["f1", "t1"]

    def test_priority_class_first_then_transcripts_then_the_rest(self) -> None:
        pending = [
            _snap("vault", "v1", 0, source_type="LOCAL_MARKDOWN"),
            _snap("chat", "c1", 5),
            _snap("rules", "r1", 1, tags=frozenset({"rule-file"}), source_type="LOCAL_MARKDOWN"),
        ]
        ordered = consolidation_mod._prioritised(pending)
        assert [p.snapshot_id for p in ordered] == ["r1", "c1", "v1"]

    def test_the_setting_off_restores_the_listing_order(self) -> None:
        get_config().consolidation.transcripts_newest_first = False
        pending = [_snap("old", "o1", 0), _snap("new", "n1", 20)]
        assert consolidation_mod._prioritised(pending) == pending

    def test_round_heads_are_each_entrys_oldest_pending_snapshot(self) -> None:
        pending = [_snap("a", "a2", 10), _snap("a", "a1", 0), _snap("b", "b1", 5)]
        heads = consolidation_mod._round_heads(pending, finished={"x"})
        assert [h.snapshot_id for h in heads] == ["a1", "b1"]
        assert consolidation_mod._round_heads(pending, finished={"a"})[0].snapshot_id == "b1"


# ---------------------------------------------------------------------------
# The round loop
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self) -> None:
        self.total = 0.0

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(priced_cost_usd=self.total)


class _Queue:
    """A pending queue the fake pooled pass drains, charging a fixed cost per snapshot."""

    def __init__(
        self,
        pending: list[PendingSnapshot],
        usage: _FakeUsage,
        *,
        cost: float = 0.045,
        hand_back: frozenset[str] = frozenset(),
        wait: BatchWaitBudget | None = None,
        wait_per_round: float = 0.0,
    ) -> None:
        self.pending = pending
        self.usage = usage
        self.cost = cost
        self.hand_back = hand_back
        self.rounds: list[list[str]] = []
        self.wait = wait
        self.wait_per_round = wait_per_round

    async def pooled(
        self,
        chosen: Sequence[tuple[str, str]],
        report: ConsolidationReport,
        *,
        account_level_out: list[bool] | None = None,
    ) -> int:
        self.rounds.append([sid for _, sid in chosen])
        self.usage.total += self.cost * len(chosen)
        if self.wait is not None:
            self.wait.spent_seconds += self.wait_per_round
        done = {sid for _, sid in chosen} - self.hand_back
        self.pending = [p for p in self.pending if p.snapshot_id not in done]
        report.pending_extracted += len(done)
        return report.pending_extracted

    async def listing(self, _session: Any) -> tuple[list[PendingSnapshot], int]:
        return list(self.pending), 0


async def _price(
    _session: Any,
    heads: Sequence[PendingSnapshot],
    cache: dict[str, RoundCandidate],
    *,
    batched: bool,
) -> list[RoundCandidate]:
    return [RoundCandidate(h.snapshot_id, budget_mod.class_of(h), 0.18, 0.2) for h in heads]


@pytest.fixture
def loop(monkeypatch: pytest.MonkeyPatch) -> Any:
    def build(queue: _Queue) -> None:
        monkeypatch.setattr(consolidation_mod, "_pass_extract_pooled", queue.pooled)
        monkeypatch.setattr(consolidation_mod, "catchup_listing", queue.listing)
        monkeypatch.setattr(consolidation_mod, "price_candidates", _price)

    return build


def _session() -> MagicMock:
    session = MagicMock(spec=AsyncSession)
    session.commit = AsyncMock()
    return session


async def _run(queue: _Queue, usage: _FakeUsage, limit: float = 4.0) -> ConsolidationReport:
    report = ConsolidationReport()
    report.pending_total = len(queue.pending)
    await consolidation_mod._pass_extract_budgeted(
        _session(),
        report,
        list(queue.pending),
        limit,
        usage,  # type: ignore[arg-type]
    )
    return report


class TestBudgetedRounds:
    @pytest.fixture(autouse=True)
    def _budget(self) -> None:
        cfg = get_config().consolidation
        cfg.extract_budget_usd = 4.0
        cfg.max_pending_entries = 200

    @pytest.mark.asyncio
    async def test_a_night_fills_the_budget_in_rounds(self, loop: Any) -> None:
        usage = _FakeUsage()
        queue = _Queue([_snap(f"e{i}", f"s{i}", i) for i in range(300)], usage)
        loop(queue)
        report = await _run(queue, usage)

        summary = report.extract_budget
        assert summary is not None
        assert [len(r) for r in queue.rounds][:2] == [20, 31]
        assert summary.rounds[0].bounded and not summary.rounds[1].bounded
        assert summary.stopped_by == "rounds"
        assert 80 <= summary.admitted <= 90
        assert summary.spent_usd <= 4.0
        assert summary.factor["CONVERSATION"] == pytest.approx(0.25)
        assert summary.overshoot_usd == 0.0
        payload = summary.payload()
        assert payload["rounds_run"] == 6 and payload["rounds_bounded"] == 1

    @pytest.mark.asyncio
    async def test_an_entrys_snapshots_are_read_one_a_round_oldest_first(self, loop: Any) -> None:
        usage = _FakeUsage()
        queue = _Queue([_snap("s", f"s{i}", i) for i in range(3)], usage)
        loop(queue)
        await _run(queue, usage)
        assert queue.rounds == [["s0"], ["s1"], ["s2"]]

    @pytest.mark.asyncio
    async def test_a_snapshot_handed_back_ends_its_entrys_run(self, loop: Any) -> None:
        usage = _FakeUsage()
        queue = _Queue(
            [_snap("a", "a1", 0), _snap("a", "a2", 1), _snap("b", "b1", 2)],
            usage,
            hand_back=frozenset({"a1"}),
        )
        loop(queue)
        report = await _run(queue, usage)
        assert queue.rounds == [["b1", "a1"]]  # b holds the newest snapshot
        assert report.extract_budget is not None
        assert report.extract_budget.stopped_by == "queue"

    @pytest.mark.asyncio
    async def test_the_ceiling_stops_the_pass(self, loop: Any) -> None:
        get_config().consolidation.max_pending_entries = 25
        usage = _FakeUsage()
        queue = _Queue([_snap(f"e{i}", f"s{i}", i) for i in range(100)], usage)
        loop(queue)
        report = await _run(queue, usage)
        assert report.extract_budget is not None
        assert report.extract_budget.admitted == 25
        assert report.extract_budget.stopped_by == "ceiling"

    @pytest.mark.asyncio
    async def test_a_metered_principals_share_is_summed_over_the_rounds(self, loop: Any) -> None:
        """a budgeted run admits the share once, not once a round."""
        from dataclasses import replace

        usage = _FakeUsage()
        metered = [
            replace(_snap(f"a{i}", f"a{i}", 100 + i), metered_principal="agent:a")
            for i in range(12)
        ]
        queue = _Queue(metered + [_snap(f"e{i}", f"s{i}", i) for i in range(40)], usage)
        loop(queue)
        report = await _run(queue, usage)

        read = [sid for round_ in queue.rounds for sid in round_]
        assert len([sid for sid in read if sid.startswith("a")]) == 5
        assert len([sid for sid in read if sid.startswith("s")]) == 40
        assert (report.pending_deferred_metered, report.metered_principals_deferred) == (7, 1)

    @pytest.mark.asyncio
    async def test_a_head_set_aside_then_admitted_is_not_counted_deferred(
        self, loop: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A head too large to read frees its principal's slot for one set aside earlier."""
        from dataclasses import replace

        get_config().consolidation.max_pending_per_metered_principal = 2
        usage = _FakeUsage()
        metered = [
            replace(_snap(f"a{i}", f"a{i}", 100 - i), metered_principal="agent:a") for i in range(3)
        ]
        queue = _Queue(metered, usage)
        loop(queue)
        priced = _price

        async def too_large_a0(*args: Any, **kwargs: Any) -> list[RoundCandidate]:
            out = await priced(*args, **kwargs)
            return [replace(c, worst_case_usd=99.0) if c.snapshot_id == "a0" else c for c in out]

        monkeypatch.setattr(consolidation_mod, "price_candidates", too_large_a0)
        report = await _run(queue, usage)

        read = [sid for round_ in queue.rounds for sid in round_]
        assert sorted(read) == ["a1", "a2"]
        assert report.pending_deferred_metered == 0

    @pytest.mark.asyncio
    async def test_pass_one_stops_at_its_share_of_the_batch_wait(self, loop: Any) -> None:
        usage = _FakeUsage()
        wait = BatchWaitBudget(budget_seconds=3600, min_remaining_seconds=300)
        queue = _Queue(
            [_snap(f"e{i}", f"s{i}", i) for i in range(300)],
            usage,
            wait=wait,
            wait_per_round=850.0,
        )
        loop(queue)
        with batch_wait_budget(wait):
            report = await _run(queue, usage)
        assert report.extract_budget is not None
        assert len(queue.rounds) == 2
        assert report.extract_budget.stopped_by == "batch_wait"

    @pytest.mark.asyncio
    async def test_a_factor_fitted_on_cheap_snapshots_cannot_double_the_limit(
        self, loop: Any
    ) -> None:
        """Security review S1: cheap first rounds, then snapshots that cost their worst case."""
        usage = _FakeUsage()
        queue = _Queue([_snap(f"e{i}", f"s{i}", -i) for i in range(400)], usage)
        cheap = {f"s{i}" for i in range(20)}

        async def pooled(
            chosen: Sequence[tuple[str, str]],
            report: ConsolidationReport,
            *,
            account_level_out: list[bool] | None = None,
        ) -> int:
            queue.rounds.append([sid for _, sid in chosen])
            usage.total += sum(0.01 if sid in cheap else 0.2 for _, sid in chosen)
            done = {sid for _, sid in chosen}
            queue.pending = [p for p in queue.pending if p.snapshot_id not in done]
            report.pending_extracted += len(done)
            return report.pending_extracted

        queue.pooled = pooled  # type: ignore[method-assign]
        loop(queue)
        report = await _run(queue, usage)
        assert report.extract_budget is not None
        cheap_fitted = report.extract_budget.rounds[1]
        assert not cheap_fitted.bounded
        # It cost its worst case, far past its 0.1-scaled charge, and still no more
        # than twice what was left.
        assert cheap_fitted.spent_usd > 5 * cheap_fitted.charged_usd
        assert cheap_fitted.worst_case_usd <= 2 * (4.0 - 0.2) + 1e-9
        assert report.extract_budget.spent_usd <= 2 * 4.0

    @pytest.mark.asyncio
    async def test_a_round_that_costs_more_than_it_was_charged_is_disclosed(
        self, loop: Any
    ) -> None:
        usage = _FakeUsage()
        queue = _Queue([_snap(f"e{i}", f"s{i}", i) for i in range(30)], usage, cost=0.3)
        loop(queue)
        report = await _run(queue, usage)
        assert report.extract_budget is not None
        assert report.extract_budget.overshoot_usd == pytest.approx(2.0)
        assert "over by US$2.00" in budget_mod.render_budget_clause(report.extract_budget)


# ---------------------------------------------------------------------------
# The limit, the unpriced route and the warnings (§§1, 3)
# ---------------------------------------------------------------------------


def test_the_limit_is_the_smaller_of_the_two_budgets() -> None:
    cfg = get_config().consolidation
    assert budget_mod.pass_one_limit(None) is None
    cfg.extract_budget_usd = 4.0
    assert budget_mod.pass_one_limit(None) == 4.0
    assert budget_mod.pass_one_limit(2.5) == 2.5
    assert budget_mod.pass_one_limit(-1.0) == 0.0


@pytest.mark.asyncio
async def test_a_budget_on_an_unpriced_route_runs_unbudgeted(db_session: AsyncSession) -> None:
    get_config().consolidation.extract_budget_usd = 4.0
    get_config().llm.extraction = ProviderSelection(provider="anthropic", model="no-such-model")
    assert not budget_mod.extraction_route_priced()
    report = ConsolidationReport()
    token = consolidation_mod._active_usage.set(_FakeUsage())  # type: ignore[arg-type]
    try:
        await consolidation_mod._pass_extract(db_session, report)
    finally:
        consolidation_mod._active_usage.reset(token)
    assert report.extract_budget is not None
    assert report.extract_budget.stopped_by == "unpriced"
    assert report.extract_budget.rounds == []


def test_the_warnings_name_an_incoherent_setting() -> None:
    cfg = get_config().consolidation
    assert budget_mod.extract_budget_warnings() == []
    cfg.extract_budget_usd = 4.0
    cfg.budget_usd = 3.0
    warnings = budget_mod.extract_budget_warnings()
    assert any("not below consolidation.budget_usd" in w for w in warnings)
    assert any("max_pending_entries is 20" in w for w in warnings)
    cfg.budget_usd = None
    cfg.max_pending_entries = 200
    assert budget_mod.extract_budget_warnings() == []


def test_class_of_keeps_the_priority_class_and_else_the_source_type() -> None:
    assert budget_mod.class_of(_snap("a", "a1", 0, tags=frozenset({"memory-file"}))) == "priority"
    assert (
        budget_mod.class_of(_snap("a", "a1", 0, source_type="LOCAL_MARKDOWN")) == "LOCAL_MARKDOWN"
    )


def test_a_snapshot_is_priced_twice_and_the_worst_case_is_higher() -> None:
    from particles.operations.audit import ExtractionCall
    from particles.operations.spend_estimate import snapshot_price

    price = snapshot_price([ExtractionCall(chars=4000)], batched=True)
    full = snapshot_price([ExtractionCall(chars=4000)], batched=False)
    assert price.estimate_usd is not None and price.worst_case_usd is not None
    assert 0 < price.estimate_usd < price.worst_case_usd
    assert full.worst_case_usd is not None and full.worst_case_usd > price.worst_case_usd
    assert snapshot_price([], batched=True).worst_case_usd == 0.0


@pytest.mark.asyncio
async def test_the_dry_run_shows_the_bounded_first_round(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = get_config().consolidation
    cfg.extract_budget_usd = 4.0
    cfg.max_pending_entries = 200
    monkeypatch.setattr(plan_mod, "price_candidates", _price)
    monkeypatch.setattr(plan_mod, "latest_run_event", AsyncMock(return_value=None))
    pending = [_snap(f"e{i}", f"s{i}", i) for i in range(40)]
    planned = await plan_mod._plan_budgeted_extract(_session(), pending, 4.0)
    assert "round 1 would read 20 snapshots of class CONVERSATION" in planned.detail
    assert "not projected" in planned.detail
