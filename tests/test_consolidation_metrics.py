# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The gauges a finished consolidation run reports."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, get_args

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.llm.usage import LLMUsage
from particles.operations.consolidation import ConsolidationPass, ConsolidationReport
from particles.operations.consolidation_metrics import (
    GAUGES,
    record_run_metrics,
    run_readings,
    run_totals,
)
from particles.operations.extract_budget import ExtractBudgetSummary, ExtractRound, StopReason
from particles.store.event_store import OperatorEventType, record_event


def _round() -> ExtractRound:
    return ExtractRound(
        klass="CONVERSATION",
        bounded=False,
        admitted=4,
        charged_usd=0.7,
        estimated_usd=0.8,
        worst_case_usd=4.1,
        spent_usd=0.65,
        wait_seconds=400.0,
    )


def _report(*, extract: str = "ran", budget: bool = True) -> ConsolidationReport:
    report = ConsolidationReport(
        completed_at=datetime(2026, 11, 17, 14, 39, tzinfo=UTC),
        llm_usage=LLMUsage(cost_usd=3.15),
        pending_extracted=22,
        pending_remaining=587,
    )
    report.passes.append(ConsolidationPass(name="extract", status=extract))  # type: ignore[arg-type]
    if budget:
        report.extract_budget = ExtractBudgetSummary(
            budget_usd=4.0,
            limit_usd=4.0,
            spent_usd=2.81,
            rounds=[_round(), _round()],
            stopped_by="rounds",
        )
    return report


def _by_name(report: ConsolidationReport) -> dict[str, list[tuple[float, dict[str, str]]]]:
    out: dict[str, list[tuple[float, dict[str, str]]]] = {}
    for r in run_readings(report, "default"):
        out.setdefault(r.name, []).append((r.value, r.attributes))
    return out


def test_a_budgeted_run_reports_every_figure() -> None:
    got = _by_name(_report())
    assert got["particles.consolidation.backlog"] == [(587.0, {"store": "default"})]
    assert got["particles.consolidation.extracted"] == [(22.0, {"store": "default"})]
    assert got["particles.consolidation.run_cost"] == [(3.15, {"store": "default"})]
    assert got["particles.consolidation.extract_spent"] == [(2.81, {"store": "default"})]
    assert got["particles.consolidation.extract_budget"] == [(4.0, {"store": "default"})]
    assert got["particles.consolidation.extract_rounds"] == [(2.0, {"store": "default"})]
    stamp = datetime(2026, 11, 17, 14, 39, tzinfo=UTC).timestamp()
    assert got["particles.consolidation.completed_at"] == [(stamp, {"store": "default"})]
    assert set(got) <= set(GAUGES)


def test_only_the_reason_that_stopped_the_run_reads_one() -> None:
    stops = _by_name(_report())["particles.consolidation.extract_stop"]
    assert {a["reason"] for _, a in stops} == set(get_args(StopReason))
    assert [a["reason"] for v, a in stops if v == 1.0] == ["rounds"]


def test_a_night_that_skipped_extraction_reports_no_backlog() -> None:
    # A false zero would read as a drained backlog on the dashboard.
    got = _by_name(_report(extract="skipped"))
    assert "particles.consolidation.backlog" not in got
    assert "particles.consolidation.extract_spent" not in got
    assert "particles.consolidation.run_cost" in got


def test_an_egress_refused_night_still_reports_its_backlog() -> None:
    report = _report(extract="skipped", budget=False)
    report.pending_total = 590
    report.pending_remaining = 590
    got = _by_name(report)
    assert got["particles.consolidation.backlog"] == [(590.0, {"store": "default"})]
    assert "particles.consolidation.extracted" not in got


def test_no_budget_reports_the_backlog_without_budget_figures() -> None:
    got = _by_name(_report(budget=False))
    assert got["particles.consolidation.backlog"] == [(587.0, {"store": "default"})]
    assert "particles.consolidation.extract_budget" not in got
    assert "particles.consolidation.extract_stop" not in got


class _Sink:
    def __init__(self) -> None:
        self.calls: list[tuple[float, dict[str, str] | None]] = []

    def set(self, amount: float, attributes: dict[str, str] | None = None) -> None:
        self.calls.append((amount, attributes))


def test_totals_sum_every_run_and_skip_what_is_missing() -> None:
    payloads: list[dict[str, Any]] = [
        {
            "llm_usage": {"cost_usd": 3.0},
            "extract_budget": {"spent_usd": 2.5},
            "census": {"pending_extracted": 20},
        },
        # An unpriced run, and one from before the extraction budget existed.
        {"llm_usage": {"cost_usd": None}, "census": {"pending_extracted": 14}},
        {"llm_usage": {"cost_usd": 1.25}},
    ]
    got = {r.name: r.value for r in run_totals(payloads, "default")}
    assert got == {
        "particles.consolidation.run_cost_to_date": 4.25,
        "particles.consolidation.extract_spent_to_date": 2.5,
        "particles.consolidation.extracted_to_date": 34.0,
    }


async def _seed_run(session: AsyncSession, cost: float, spent: float, read: int) -> None:
    await record_event(
        session,
        actor="consolidation",
        event_type=OperatorEventType.CONSOLIDATION_RUN,
        payload={
            "llm_usage": {"cost_usd": cost},
            "extract_budget": {"spent_usd": spent},
            "census": {"pending_extracted": read},
        },
    )


async def test_record_sets_each_gauge_with_totals_from_the_store(
    db_session: AsyncSession,
) -> None:
    await _seed_run(db_session, 3.0, 2.5, 20)
    await _seed_run(db_session, 2.0, 1.5, 10)
    sinks = {name: _Sink() for name in GAUGES}
    await record_run_metrics(db_session, _report(), "default", gauges=sinks)  # type: ignore[arg-type]
    assert sinks["particles.consolidation.backlog"].calls == [(587.0, {"store": "default"})]
    assert sinks["particles.consolidation.run_cost_to_date"].calls == [(5.0, {"store": "default"})]
    assert sinks["particles.consolidation.extracted_to_date"].calls == [
        (30.0, {"store": "default"})
    ]
    assert len(sinks["particles.consolidation.extract_stop"].calls) == len(get_args(StopReason))


async def test_a_metrics_fault_never_fails_the_run(
    db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    # A missing gauge: logged, not raised.
    with caplog.at_level(logging.WARNING, logger="particles.operations.consolidation_metrics"):
        await record_run_metrics(db_session, _report(), "default", gauges={})
    assert "consolidation metrics were not recorded" in caplog.text


async def test_the_default_gauges_take_every_reading(db_session: AsyncSession) -> None:
    # The OTel API without an SDK: no-op gauges, but every name must exist.
    await record_run_metrics(db_session, _report(), "default")
