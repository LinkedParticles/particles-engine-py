# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The judged benchmark verbs' spend stop, one test per acceptance criterion.

Every LLM seam is scripted at the provider port; a metered provider reports
usage as an adapter does, so the stop reads the same accumulator a live run
fills. No key, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli.benchmark import _benchmark_leakage, _Format
from particles.benchmark.ceiling import (
    STOP_EXCLUSION,
    CeilingReached,
    InFlight,
    Role,
    SpendCeiling,
    StoppedAtCeiling,
    in_flight_usd,
    render_stop_terms,
    stop_refusal,
)
from particles.benchmark.leakage import LeakageReport, build_report, render_report, run_leakage
from particles.benchmark.memory.runner import _scored_call
from particles.benchmark.relevance_floor import RelevanceFloorReport, run_judged, write_heldout
from particles.benchmark.relevance_floor import build_report as build_floor_report
from particles.benchmark.relevance_floor.render import render_report as render_floor_report
from particles.config import TokenPrice, get_config
from particles.llm import override_providers
from particles.llm.usage import UsageAccumulator, record_usage, track_usage
from tests.test_benchmark_leakage import (
    _ANSWER,
    _OFF_TOPIC,
    _ON_TOPIC,
    _Metered,
    _Scripted,
    _selection,
    _verdicts,
    leakage_store,  # noqa: F401 — the scripted store fixture
)
from tests.test_benchmark_relevance_floor import _Metered as _FloorMetered
from tests.test_benchmark_relevance_floor import _selection as _floor_selection
from tests.test_benchmark_relevance_floor import floor_store  # noqa: F401 — the fixture

cli = CliRunner()


def _price(*names: str) -> None:
    for name in names:
        get_config().llm.price_per_mtok[f"scripted:{name}"] = TokenPrice(input=2.0, output=10.0)


def _roles(answer_priced: bool, judge_priced: bool) -> list[Role]:
    price = (2.0, 10.0)
    return [
        Role("answer", "query_response", "scripted:composer", price if answer_priced else None),
        Role("judge", "benchmark", "scripted:judge", price if judge_priced else None),
    ]


def _rows(results: list[Any]) -> list[dict[str, Any]]:
    return [r.model_dump(mode="json") for r in results]


# ---------------------------------------------------------------------------
# 1. A run without the stops its roles need refuses before any call
# ---------------------------------------------------------------------------


class TestGate:
    @pytest.mark.parametrize(
        ("answer_priced", "judge_priced", "flags", "refused"),
        [
            (True, True, {}, True),
            (True, True, {"max_usd": 1.0}, False),
            (True, True, {"no_ceiling": True}, True),
            (False, False, {}, True),
            (False, False, {"max_calls": 10}, False),
            (False, False, {"no_ceiling": True}, False),
            (False, False, {"max_usd": 1.0}, True),
            (True, False, {"max_usd": 1.0}, True),
            (True, False, {"max_calls": 10}, True),
            (True, False, {"max_usd": 1.0, "max_calls": 10}, False),
            (True, False, {"max_usd": 1.0, "no_ceiling": True}, False),
            (True, False, {"no_ceiling": True}, True),
            (False, False, {"max_calls": 10, "no_ceiling": True}, True),
        ],
    )
    def test_each_role_needs_its_own_stop(
        self, answer_priced: bool, judge_priced: bool, flags: dict[str, Any], refused: bool
    ) -> None:
        args: dict[str, Any] = {"max_usd": None, "max_calls": None, "no_ceiling": False, **flags}
        refusal = stop_refusal(_roles(answer_priced, judge_priced), **args)
        assert (refusal is not None) is refused, refusal

    @pytest.mark.parametrize(
        ("priced", "flags", "message"),
        [
            (True, [], "needs --max-usd"),
            (True, ["--no-ceiling"], "needs --max-usd"),
            (False, [], "needs --max-calls"),
            (True, ["--max-usd", "0"], "must be above zero"),
        ],
    )
    def test_leakage_exits_2_before_any_call(
        self, priced: bool, flags: list[str], message: str
    ) -> None:
        if priced:
            _price("composer", "judge")
        composer = _Scripted("composer", _ANSWER)
        judge = _Scripted("judge", _verdicts)
        with override_providers({"query_response": composer, "benchmark": judge}):
            result = cli.invoke(app, ["benchmark", "leakage", "-q", "What?", "--yes", *flags])
        assert result.exit_code == 2, result.output
        assert message in result.output and "Nothing was spent" in result.output
        assert composer.calls == judge.calls == 0

    def test_relevance_floor_exits_2_before_the_replay(self, tmp_path: Path) -> None:
        _price("answer", "judge")
        heldout = tmp_path / "h.jsonl"
        write_heldout(heldout, [_ON_TOPIC])
        answer = _Scripted("answer", "unused")
        judge = _Scripted("judge", "yes")
        with override_providers({"query_response": answer, "benchmark": judge}):
            result = cli.invoke(
                app, ["benchmark", "relevance-floor", "--heldout", str(heldout), "--judge", "--yes"]
            )
        assert result.exit_code == 2, result.output
        assert "needs --max-usd" in result.output
        assert answer.calls == judge.calls == 0

    def test_estimate_needs_no_stop(self, tmp_path: Path) -> None:
        _price("composer", "judge")
        with override_providers(
            {"query_response": _Scripted("composer", _ANSWER), "benchmark": _Scripted("judge", "")}
        ):
            result = cli.invoke(app, ["benchmark", "leakage", "-q", "What?", "--estimate"])
        assert result.exit_code == 0, result.output
        assert "output worst case, every projected call at its max_tokens" in result.output


# ---------------------------------------------------------------------------
# 2. A stop that is never reached changes nothing
# ---------------------------------------------------------------------------


class TestUnreachedStop:
    async def test_leakage_rows_and_checkpoint_are_identical(
        self,
        leakage_store: None,  # noqa: F811
        tmp_path: Path,
    ) -> None:
        _price("composer", "judge")

        async def _run(tag: str, ceiling_kind: str) -> tuple[list[Any], str]:
            composer = _Metered("composer", _ANSWER, input_tokens=1_000, output_tokens=200)
            judge = _Metered("judge", _verdicts, input_tokens=500, output_tokens=300)
            if ceiling_kind == "unpriced":
                get_config().llm.price_per_mtok.clear()
            checkpoint = tmp_path / f"{tag}.jsonl"
            with (
                override_providers({"query_response": composer, "benchmark": judge}),
                track_usage() as usage,
            ):
                ceiling = (
                    None
                    if ceiling_kind == "none"
                    else SpendCeiling(usage, max_usd=None if ceiling_kind == "unpriced" else 1e6)
                )
                results = await run_leakage(
                    [_ON_TOPIC, _OFF_TOPIC],
                    selection=_selection(),
                    checkpoint_path=checkpoint,
                    ceiling=ceiling,
                )
            assert ceiling is None or ceiling.report is None
            return results, checkpoint.read_text()

        plain, plain_ck = await _run("plain", "none")
        priced, priced_ck = await _run("priced", "priced")
        unpriced, unpriced_ck = await _run("unpriced", "unpriced")
        assert _rows(plain) == _rows(priced) == _rows(unpriced)
        assert plain_ck == priced_ck == unpriced_ck
        report = build_report(_selection(), plain)
        again = build_report(_selection(), priced)
        assert report.model_dump(exclude={"results"}) == again.model_dump(exclude={"results"})
        assert report.stopped_at_ceiling is None


# ---------------------------------------------------------------------------
# 3. Spend over the ceiling stops before the next call, retries included
# ---------------------------------------------------------------------------


class _FailsOnce:
    """A metered provider whose first call is billed and then fails."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_model(self) -> str:
        return "scripted:judge"

    async def complete(self, prompt: str, **_: object) -> str:
        self.calls += 1
        record_usage(self.provider_model, input_tokens=1_000, output_tokens=1_000)
        if self.calls == 1:
            raise RuntimeError("overloaded")
        return "yes"


class TestStop:
    async def test_a_retry_is_not_started_past_the_ceiling(self) -> None:
        _price("judge")
        judge = _FailsOnce()
        with override_providers({"benchmark": judge}), track_usage() as usage:
            ceiling = SpendCeiling(usage, max_usd=0.005)
            with pytest.raises(CeilingReached):
                await _scored_call(
                    "benchmark",
                    "prompt",
                    max_tokens=10,
                    retries=3,
                    backoff_seconds=0.0,
                    before_attempt=ceiling.check,
                )
        assert judge.calls == 1  # $0.012 spent, so the retry never started
        assert ceiling.stopped
        with pytest.raises(CeilingReached):
            ceiling.check()  # a reached stop stays reached

    async def test_the_next_call_never_starts_and_the_terms_are_stated(
        self,
        leakage_store: None,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
        _price("composer", "judge")
        composer = _Metered("composer", _ANSWER, input_tokens=1_000, output_tokens=200)
        judge = _Metered("judge", _verdicts, input_tokens=500, output_tokens=300)
        out = tmp_path / "report.json"
        with override_providers({"query_response": composer, "benchmark": judge}):
            await _benchmark_leakage(
                questions=[_ON_TOPIC.question],
                heldout=None,
                store=None,
                top_k=5,
                limit=None,
                estimate_only=False,
                yes=True,
                per_question=None,
                output=out,
                output_format=_Format.json,
                checkpoint=None,
                allow_in_repo=True,
                max_usd=0.001,
            )
        # The answer cost $0.004, past the $0.001 ceiling: no judge call started.
        assert (composer.calls, judge.calls) == (1, 0)
        err = capsys.readouterr().err
        assert "ceiling US$0.00 at list price (not your account balance)" in err
        assert "in flight at the stop (output hard, US$" in err
        assert "input projected, US$" in err
        assert "priced at US$2 in / US$10 out per million tokens" in err
        assert err.index("ceiling US$") < err.index("Spend: estimated")
        assert "Stopped at the spend ceiling" in err
        report = LeakageReport.model_validate_json(out.read_text())
        assert report.stopped_at_ceiling is not None
        assert report.excluded == {STOP_EXCLUSION: 1}

    def test_the_overshoot_bound_prices_output_at_max_tokens(self) -> None:
        roles = _roles(True, True)
        flying = [InFlight(roles[0], 4, 1_000, 4_096), InFlight(roles[1], 8, 2_000, 1_024)]
        output, inputs = in_flight_usd(flying)
        assert output == pytest.approx((4 * 4_096 + 8 * 1_024) * 10 / 1e6)
        assert inputs == pytest.approx((4 * 1_000 + 8 * 2_000) * 2 / 1e6)
        text = render_stop_terms(
            roles, max_usd=5.0, max_calls=None, in_flight=flying, worst_case_usd=1.0
        )
        assert "ceiling US$5.00 at list price (not your account balance)" in text
        assert "(output hard, US$0.25; input projected, US$0.04)" in text

    def test_an_unpriced_role_states_its_call_ceiling(self) -> None:
        roles = _roles(True, False)
        flying = [InFlight(roles[0], 4, 1_000, 4_096), InFlight(roles[1], 8, 2_000, 1_024)]
        text = render_stop_terms(
            roles, max_usd=1.0, max_calls=50, in_flight=flying, worst_case_usd=None
        )
        assert "ceiling 50 billed call(s) of the unpriced role(s) (judge)" in text
        assert "plus up to 8 in flight at the stop" in text
        waived = render_stop_terms(
            roles, max_usd=1.0, max_calls=None, in_flight=flying, worst_case_usd=None
        )
        assert "no call ceiling on the unpriced role(s) (judge): --no-ceiling" in waived


# ---------------------------------------------------------------------------
# 4. An interrupted question never reaches the checkpoint and is re-answered
# ---------------------------------------------------------------------------


class TestInterrupted:
    async def test_leakage_discards_and_a_rerun_finishes(
        self,
        leakage_store: None,  # noqa: F811
        tmp_path: Path,
    ) -> None:
        _price("composer", "judge")
        composer = _Metered("composer", _ANSWER, input_tokens=1_000, output_tokens=200)
        judge = _Metered("judge", _verdicts, input_tokens=500, output_tokens=300)
        checkpoint = tmp_path / "ck.jsonl"
        with override_providers({"query_response": composer, "benchmark": judge}):
            with track_usage() as usage:
                ceiling = SpendCeiling(usage, max_usd=0.001)
                stopped = await run_leakage(
                    [_ON_TOPIC, _OFF_TOPIC],
                    selection=_selection(),
                    checkpoint_path=checkpoint,
                    ceiling=ceiling,
                )
            assert [r.excluded for r in stopped] == [STOP_EXCLUSION, STOP_EXCLUSION]
            assert "interrupted" in stopped[0].excluded_detail
            assert "not started" in stopped[1].excluded_detail
            assert (composer.calls, judge.calls) == (1, 0)
            assert not checkpoint.exists() or checkpoint.read_text() == ""
            with track_usage() as usage:
                finished = await run_leakage(
                    [_ON_TOPIC, _OFF_TOPIC],
                    selection=_selection(),
                    checkpoint_path=checkpoint,
                    ceiling=SpendCeiling(usage, max_usd=100.0),
                )
        assert [r.excluded for r in finished] == ["", ""]
        assert composer.calls == 3  # the interrupted question was answered afresh
        assert len(checkpoint.read_text().splitlines()) == 2

    async def test_relevance_floor_discards_and_a_rerun_finishes(
        self,
        floor_store: None,  # noqa: F811
        tmp_path: Path,
    ) -> None:
        _price("answer", "judge")
        answer = _FloorMetered(
            "answer", "The default is 0.25.", input_tokens=800, output_tokens=100
        )
        judge = _FloorMetered("judge", "yes", input_tokens=900, output_tokens=40)
        checkpoint = tmp_path / "ck.jsonl"
        with override_providers({"query_response": answer, "benchmark": judge}):
            with track_usage() as usage:
                ceiling = SpendCeiling(usage, max_usd=0.001)
                stopped = await run_judged(
                    [_ON_TOPIC, _OFF_TOPIC],
                    selection=_floor_selection(),
                    checkpoint_path=checkpoint,
                    ceiling=ceiling,
                )
            assert [r.excluded for r in stopped] == [STOP_EXCLUSION, STOP_EXCLUSION]
            assert (answer.calls, judge.calls) == (1, 0)
            assert not checkpoint.exists()
            assert ceiling.report is not None
            assert (ceiling.report.unstarted, ceiling.report.interrupted) == (1, 1)
            assert ceiling.report.interrupted_sentences == [1]
            with track_usage() as usage:
                finished = await run_judged(
                    [_ON_TOPIC, _OFF_TOPIC],
                    selection=_floor_selection(),
                    checkpoint_path=checkpoint,
                    ceiling=SpendCeiling(usage, max_usd=100.0),
                )
        assert all(r.excluded == "" for r in finished)
        assert (answer.calls, judge.calls) == (3, 2)


# ---------------------------------------------------------------------------
# 5. The stopped report says what stopped, and says it is incomplete
# ---------------------------------------------------------------------------


class TestStoppedReport:
    async def test_leakage_records_the_stop(self, leakage_store: None) -> None:  # noqa: F811
        _price("composer", "judge")
        composer = _Metered("composer", _ANSWER, input_tokens=1_000, output_tokens=200)
        judge = _Metered("judge", _verdicts, input_tokens=500, output_tokens=300)
        with (
            override_providers({"query_response": composer, "benchmark": judge}),
            track_usage() as usage,
        ):
            # The first question finishes ($0.004 + 3 × $0.004); the second's answer
            # pushes past $0.017, so its judge calls never start; the third never starts.
            ceiling = SpendCeiling(usage, max_usd=0.017)
            third = _OFF_TOPIC.model_copy(update={"question_id": "third"})
            results = await run_leakage(
                [_ON_TOPIC, _OFF_TOPIC, third], selection=_selection(), ceiling=ceiling
            )
        stop = ceiling.report
        assert stop is not None
        assert [r.excluded for r in results] == ["", STOP_EXCLUSION, STOP_EXCLUSION]
        assert stop.max_usd == 0.017
        assert stop.spend_at_stop_usd == pytest.approx(0.020)
        assert stop.overshoot_usd == 0.0 and stop.overshoot_calls == 0
        assert (stop.unstarted, stop.interrupted) == (1, 1)
        assert stop.discarded_usd == pytest.approx(0.004) and stop.discarded_calls == 1
        assert stop.finished_by_source == {"mcp_query": 1}
        assert stop.unstarted_by_source == {"user_prompt": 1}
        assert stop.finished_sentences == [3]
        assert stop.interrupted_sentences == [3]

        report = build_report(_selection(questions=3), results, stopped=stop)
        assert report.stopped_at_ceiling == stop
        assert report.quality_notes[0].startswith("INCOMPLETE: the run stopped at its spend")
        table = render_report(report)
        assert table.startswith("INCOMPLETE (stopped at the spend ceiling): Model-prior leakage")
        assert "STOPPED AT THE SPEND CEILING — INCOMPLETE" in table
        roundtrip = LeakageReport.model_validate_json(report.model_dump_json())
        assert roundtrip.stopped_at_ceiling == stop

    def test_relevance_floor_marks_a_stopped_report(self) -> None:
        stop = StoppedAtCeiling(max_usd=1.0, unstarted=2, unstarted_by_source={"mcp_query": 2})
        report = build_floor_report(_floor_selection(), [], stopped=stop)
        assert report.quality_notes[0].startswith("INCOMPLETE")
        assert render_floor_report(report).startswith("INCOMPLETE (stopped at the spend ceiling)")
        complete = build_floor_report(_floor_selection(), [])
        assert complete.stopped_at_ceiling is None
        assert "INCOMPLETE" not in render_floor_report(complete)
        loaded = RelevanceFloorReport.model_validate_json(report.model_dump_json())
        assert loaded.stopped_at_ceiling == stop


# ---------------------------------------------------------------------------
# 6. --max-calls counts the unpriced roles' billed calls; replays count for neither
# ---------------------------------------------------------------------------


def _add(usage: UsageAccumulator, model: str, *, replayed: bool = False) -> None:
    usage.add(
        purpose="benchmark",
        provider="scripted",
        model=model,
        batched=False,
        input_tokens=1_000,
        output_tokens=1_000,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        hit_max_tokens=False,
        replayed=replayed,
    )


class TestReading:
    def test_max_calls_counts_only_unpriced_billed_calls(self) -> None:
        _price("priced")
        usage = UsageAccumulator()
        ceiling = SpendCeiling(usage, max_calls=2)
        _add(usage, "priced")
        _add(usage, "priced")
        _add(usage, "local", replayed=True)
        _add(usage, "local")
        ceiling.check()  # one unpriced billed call
        _add(usage, "local")
        with pytest.raises(CeilingReached):
            ceiling.check()

    def test_replayed_rows_never_reach_the_dollar_stop(self) -> None:
        _price("priced")
        usage = UsageAccumulator()
        ceiling = SpendCeiling(usage, max_usd=0.02)
        for _ in range(5):
            _add(usage, "priced", replayed=True)
        ceiling.check()  # $0.06 replayed, $0 spent
        _add(usage, "priced")
        ceiling.check()  # $0.012 spent
        _add(usage, "priced")
        with pytest.raises(CeilingReached):
            ceiling.check()


# ---------------------------------------------------------------------------
# 7. The lexical-leg script: tests/test_bench_lexical_leg.py, which loads the
#    script by path and so is internal tooling the export does not ship.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8. Stopped, then resumed under a higher ceiling: the same rows as uninterrupted
# ---------------------------------------------------------------------------


class TestResume:
    async def test_a_resumed_run_matches_an_uninterrupted_one(
        self,
        leakage_store: None,  # noqa: F811
        tmp_path: Path,
    ) -> None:
        _price("composer", "judge")
        questions = [_ON_TOPIC, _OFF_TOPIC, _OFF_TOPIC.model_copy(update={"question_id": "3"})]

        async def _run(checkpoint: Path, max_usd: float) -> list[Any]:
            composer = _Metered("composer", _ANSWER, input_tokens=1_000, output_tokens=200)
            judge = _Metered("judge", _verdicts, input_tokens=500, output_tokens=300)
            with (
                override_providers({"query_response": composer, "benchmark": judge}),
                track_usage() as usage,
            ):
                return await run_leakage(
                    questions,
                    selection=_selection(),
                    checkpoint_path=checkpoint,
                    ceiling=SpendCeiling(usage, max_usd=max_usd),
                )

        whole = await _run(tmp_path / "whole.jsonl", 100.0)
        first = await _run(tmp_path / "resumed.jsonl", 0.017)
        assert STOP_EXCLUSION in {r.excluded for r in first}
        resumed = await _run(tmp_path / "resumed.jsonl", 100.0)
        assert _rows(resumed) == _rows(whole)
        a = build_report(_selection(questions=3), whole)
        b = build_report(_selection(questions=3), resumed)
        assert a.model_dump() == b.model_dump()
