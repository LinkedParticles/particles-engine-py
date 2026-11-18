# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The projection beside the measured usage of a judged benchmark run."""

from __future__ import annotations

from particles.benchmark.leakage import LeakageEstimate
from particles.benchmark.relevance_floor import JudgedStageEstimate
from particles.benchmark.spend import RunSpend, measure_spend, render_spend
from particles.llm.usage import LLMUsage, UsageRow


def _estimate() -> LeakageEstimate:
    return LeakageEstimate(
        questions=10,
        answer_calls=10,
        judge_calls=80,
        answer_input_tokens=20_000,
        answer_output_tokens=5_000,
        judge_input_tokens=160_000,
        judge_output_tokens=9_600,
        cost_usd=1.00,
    )


def _usage() -> LLMUsage:
    return LLMUsage(
        rows=[
            UsageRow(
                purpose="query_response",
                provider="anthropic",
                model="claude-sonnet-5-5",
                calls=10,
                input_tokens=25_000,
                cache_read_tokens=1_000,
                output_tokens=6_000,
                cost_usd=0.12,
            ),
            UsageRow(
                purpose="benchmark",
                provider="anthropic",
                model="claude-opus-5-5",
                calls=150,
                input_tokens=300_000,
                cache_creation_tokens=2_000,
                output_tokens=90_000,
                cost_usd=3.00,
            ),
            # A replayed slice bought nothing this run and is not measured spend.
            UsageRow(
                purpose="benchmark",
                provider="anthropic",
                model="claude-opus-5-5",
                replayed=True,
                calls=40,
                input_tokens=80_000,
                output_tokens=20_000,
                cost_usd=0.80,
            ),
        ],
        cost_usd=3.12,
    )


class TestMeasureSpend:
    def test_each_role_pairs_its_projection_with_its_billed_usage(self) -> None:
        spend = measure_spend(_estimate(), _usage(), restored_questions=2)
        answer, judge = spend.roles
        assert (answer.role, answer.purpose) == ("answer", "query_response")
        assert (answer.estimated_calls, answer.measured_calls) == (10, 10)
        # Every billed input token counts, cache reads and writes included.
        assert answer.measured_input_tokens == 26_000
        assert (judge.estimated_calls, judge.measured_calls) == (80, 150)
        assert judge.measured_input_tokens == 302_000
        assert judge.measured_output_tokens == 90_000
        assert (spend.estimated_cost_usd, spend.measured_cost_usd) == (1.00, 3.12)
        assert spend.ratio == 3.12
        assert spend.restored_questions == 2

    def test_both_harness_estimates_fit(self) -> None:
        floor = JudgedStageEstimate(answer_calls=3, judge_calls=3, cost_usd=None)
        spend = measure_spend(floor, LLMUsage())
        assert [r.estimated_calls for r in spend.roles] == [3, 3]
        assert [r.measured_calls for r in spend.roles] == [0, 0]
        assert spend.ratio is None

    def test_a_spend_round_trips_in_a_report(self) -> None:
        spend = measure_spend(_estimate(), _usage())
        again = RunSpend.model_validate_json(spend.model_dump_json())
        assert again == spend


class TestRenderSpend:
    def test_the_block_names_both_figures_the_ratio_and_the_usage_line(self) -> None:
        text = render_spend(measure_spend(_estimate(), _usage(), restored_questions=2))
        assert text.startswith("Spend: estimated ~US$1.00, measured US$3.12 at list price")
        assert "(3.12× the estimate)" in text
        assert "judge   estimated    80 calls" in text
        assert "measured   150 calls, 302k in / 90k out" in text
        assert "2 question(s) restored from the checkpoint" in text
        assert "LLM usage:" in text

    def test_an_unpriced_run_says_so_rather_than_guessing(self) -> None:
        usage = LLMUsage(
            rows=[UsageRow(purpose="benchmark", provider="local", model="m", calls=1)],
            cost_usd=None,
            unpriced=["local:m"],
        )
        floor = JudgedStageEstimate(judge_calls=1, cost_usd=None)
        text = render_spend(measure_spend(floor, usage))
        assert text.startswith(
            "Spend: estimated without a dollar figure (no price configured), "
            "measured without a dollar total (a model is unpriced)."
        )
        assert "×" not in text.splitlines()[0]
