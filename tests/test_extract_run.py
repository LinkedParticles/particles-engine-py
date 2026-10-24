# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The priced, ordered, budgeted ``extract --all-pending``.

The decisions are pure and pinned here with plain values: the reading order,
the confirmation gate, and the budget check with its boundary, its scaling and
its overshoot bound. ``plan_run`` is checked against a real store.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import ProviderSelection, get_config
from particles.llm.usage import LLMUsage, UsageRow
from particles.operations.extract_run import (
    BudgetGate,
    PendingSnapshot,
    decide_confirmation,
    extraction_route,
    order_pending,
    parse_model_route,
    plan_run,
)
from particles.operations.spend_estimate import planned_extraction_estimate


def _p(entry: str, snap: str, *, absent: bool = False, read: bool = False) -> PendingSnapshot:
    return PendingSnapshot(entry, snap, absent=absent, read_before=read)


class TestOrderPending:
    def test_edits_of_read_entries_come_before_never_read_ones(self) -> None:
        items = [_p("a", "a1"), _p("b", "b1", read=True), _p("c", "c1"), _p("d", "d1", read=True)]
        assert [i.snapshot_id for i in order_pending(items)] == ["b1", "d1", "a1", "c1"]

    def test_absences_go_last_in_their_own_order(self) -> None:
        items = [_p("x", "x1", absent=True), _p("a", "a1"), _p("y", "y1", absent=True, read=True)]
        assert [i.snapshot_id for i in order_pending(items)] == ["a1", "x1", "y1"]

    def test_one_entrys_snapshots_keep_their_order(self) -> None:
        """The class is the entry's, so generations are never read out of turn."""
        items = [_p("a", "a1", read=True), _p("b", "b1"), _p("a", "a2", read=True)]
        ordered = [i.snapshot_id for i in order_pending(items)]
        assert ordered.index("a1") < ordered.index("a2")


class TestDecideConfirmation:
    @pytest.mark.parametrize(
        ("calls", "yes", "budget", "interactive", "expected"),
        [
            (50, False, None, False, "run"),
            (51, False, None, True, "ask"),
            (51, False, None, False, "refuse"),
            (51, True, None, False, "run"),
            (51, False, 3.0, False, "run"),
            (5000, False, 0.0, False, "run"),
        ],
    )
    def test_matrix(
        self, calls: int, yes: bool, budget: float | None, interactive: bool, expected: str
    ) -> None:
        verdict = decide_confirmation(
            calls=calls, threshold=50, yes=yes, budget_usd=budget, interactive=interactive
        )
        assert verdict == expected


def _usage(extraction: float, other: float = 0.0) -> LLMUsage:
    rows = [UsageRow(purpose="extraction", provider="anthropic", model="m", cost_usd=extraction)]
    if other:
        rows.append(
            UsageRow(purpose="semantic_lint", provider="anthropic", model="m", cost_usd=other)
        )
    return LLMUsage(rows=rows, cost_usd=extraction + other)


class TestBudgetGate:
    def test_the_records_example_reads_four_of_ten(self) -> None:
        """Estimated at US$0.50, billed US$0.40, budget US$2.00 (§ Acceptance)."""
        gate = BudgetGate(2.0)
        spent = 0.0
        read = 0
        for _ in range(10):
            if not gate.allows(spent_usd=spent, estimate_usd=0.50):
                break
            spent += 0.40
            read += 1
        assert read == 4
        assert spent == pytest.approx(1.60)

    def test_an_estimate_landing_on_the_budget_runs(self) -> None:
        assert BudgetGate(2.0).allows(spent_usd=1.50, estimate_usd=0.50)

    def test_scaling_by_the_observed_ratio(self) -> None:
        gate = BudgetGate(2.0)
        assert gate.allows(spent_usd=0.6, estimate_usd=0.70)
        gate.observe(_usage(0.30, 0.30))
        assert gate.ratio == pytest.approx(2.0)
        # 0.6 + 0.7 × 2 = 2.0 runs; 0.6 + 0.71 × 2 does not.
        assert gate.allows(spent_usd=0.6, estimate_usd=0.70)
        assert not gate.allows(spent_usd=0.6, estimate_usd=0.71)

    def test_the_ratio_never_relaxes_the_check(self) -> None:
        gate = BudgetGate(2.0)
        gate.observe(_usage(0.50))
        assert gate.ratio == 1.0

    def test_overshoot_is_the_last_snapshots_excess_and_nothing_follows(self) -> None:
        gate = BudgetGate(2.0)
        assert gate.allows(spent_usd=1.40, estimate_usd=0.50)
        spent = 1.40 + 0.90  # the snapshot billed past its estimate
        assert spent == pytest.approx(2.30)
        assert not gate.allows(spent_usd=spent, estimate_usd=0.01)

    def test_no_budget_always_runs(self) -> None:
        assert BudgetGate(None).allows(spent_usd=1e9, estimate_usd=1e9)

    def test_an_unpriced_estimate_runs_until_spend_reaches_the_budget(self) -> None:
        gate = BudgetGate(1.0)
        assert gate.allows(spent_usd=0.0, estimate_usd=None)
        assert not gate.allows(spent_usd=1.0, estimate_usd=None)


class TestModelRoute:
    def test_parse(self) -> None:
        selection = parse_model_route("anthropic:claude-haiku-5-5")
        assert selection == ProviderSelection(provider="anthropic", model="claude-haiku-5-5")

    @pytest.mark.parametrize("text", ["claude-haiku-5-5", "anthropic:", ":m", "nowhere:m"])
    def test_parse_refuses(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_model_route(text)

    def test_routes_extraction_and_subject_resolution_and_restores(self) -> None:
        llm = get_config().llm
        before = (llm.for_purpose("extraction"), llm.for_purpose("subject_resolution"))
        other = llm.for_purpose("semantic_lint")
        haiku = ProviderSelection(provider="anthropic", model="claude-haiku-5-5")
        with pytest.raises(RuntimeError), extraction_route(haiku):
            assert llm.for_purpose("extraction") == haiku
            assert llm.for_purpose("subject_resolution") == haiku
            assert llm.for_purpose("semantic_lint") == other
            raise RuntimeError
        assert (llm.for_purpose("extraction"), llm.for_purpose("subject_resolution")) == before

    def test_none_changes_nothing(self) -> None:
        before = get_config().llm.for_purpose("extraction")
        with extraction_route(None):
            assert get_config().llm.for_purpose("extraction") == before


async def _seed(session: AsyncSession, texts: list[str]) -> list[PendingSnapshot]:
    from particles.core.schema import CorpusEntry, ExtractionStatus, Snapshot, WarcRecordType
    from particles.corpus.deposit import save_blob, sha256
    from particles.corpus.store import CorpusEntryRow, SnapshotRow

    out: list[PendingSnapshot] = []
    for i, text in enumerate(texts):
        content = text.encode()
        digest = sha256(content)
        save_blob(content, digest)
        entry = CorpusEntry(
            entry_id=str(uuid.uuid4()),
            source_type="LOCAL_MARKDOWN",
            uri_r=f"file:///tmp/plan-{i}.md",
            deposited_by="test",
        )
        snap = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            captured_at=datetime.now(UTC),
            content_hash=digest,
            archive_path=str(digest),
            extraction_status=ExtractionStatus.PENDING,
            warc_record_type=WarcRecordType.RESPONSE,
        )
        session.add(CorpusEntryRow.from_model(entry))
        session.add(SnapshotRow.from_model(snap, entry.entry_id))
        out.append(PendingSnapshot(entry.entry_id, snap.snapshot_id))
    await session.flush()
    return out


class TestPlanRun:
    async def test_prices_each_snapshot_at_the_high_end(self, db_session: AsyncSession) -> None:
        pending = await _seed(db_session, ["A short note.\n", "A longer note. " * 200])
        plan = await plan_run(db_session, pending)
        assert plan.total is not None and plan.total.calls >= 2
        for item in pending:
            calls = plan.calls[item.snapshot_id]
            assert plan.high_usd[item.snapshot_id] == pytest.approx(
                planned_extraction_estimate(calls, entries=1, batched=False).usd
            )
        # The run's high end prices the same calls together.
        assert plan.total.high_usd == pytest.approx(
            planned_extraction_estimate(
                [c for item in pending for c in plan.calls[item.snapshot_id]],
                entries=2,
                batched=False,
            ).usd
        )

    async def test_prices_the_alternative_model_beside_the_configured_one(
        self, db_session: AsyncSession
    ) -> None:
        pending = await _seed(db_session, ["A note about tools. " * 50])
        plan = await plan_run(db_session, pending)
        assert plan.alternative is not None
        assert plan.alternative.model == "claude-haiku-5-5"
        assert plan.alternative_route == "anthropic:claude-haiku-5-5"
        assert plan.total is not None and plan.alternative.high_usd is not None
        assert plan.alternative.high_usd < (plan.total.high_usd or 0)

    async def test_no_alternative_when_it_is_the_configured_model(
        self, db_session: AsyncSession
    ) -> None:
        pending = await _seed(db_session, ["A note. " * 10])
        with extraction_route(parse_model_route("anthropic:claude-haiku-5-5")):
            plan = await plan_run(db_session, pending)
        assert plan.alternative is None

    async def test_remaining_prices_what_was_not_attempted(self, db_session: AsyncSession) -> None:
        pending = await _seed(db_session, ["One. " * 40, "Two. " * 40, "Three. " * 40])
        plan = await plan_run(db_session, pending)
        count, chars, rest = plan.remaining({pending[0].snapshot_id})
        assert count == 2
        assert chars == sum(c.chars for item in pending[1:] for c in plan.calls[item.snapshot_id])
        assert rest.high_usd is not None and plan.total is not None
        assert rest.high_usd < (plan.total.high_usd or 0)

    async def test_absences_are_not_planned(self, db_session: AsyncSession) -> None:
        plan = await plan_run(db_session, [PendingSnapshot("e", "s", absent=True)])
        assert plan.calls == {}
        assert plan.total is not None and plan.total.calls == 0
