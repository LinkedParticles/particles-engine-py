# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The dollar-budget decision and the batch chunks it declines.

Covers ``particles/core/spend_budget.py`` (the pure run-or-skip decision) and
``particles/llm/spend_budget.py`` (the run-scoped budget the batch adapter
consults between chunks). The Anthropic SDK is mocked through the
``particles.llm.set_client`` seam, so no request leaves the process.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pytest

from particles import llm
from particles.config import reset_config
from particles.core.spend_budget import decide_spend
from particles.llm import CompletionRequest
from particles.llm.registry import complete_many_with_provider_model
from particles.llm.spend_budget import SpendBudget, current_spend_budget, spend_budget

# ---------------------------------------------------------------------------
# The pure decision
# ---------------------------------------------------------------------------


class TestDecideSpend:
    def test_no_budget_always_runs(self) -> None:
        assert decide_spend(spent_usd=1e6, budget_usd=None, estimate_usd=1e6) == "run"

    def test_a_unit_that_fits_runs(self) -> None:
        assert decide_spend(spent_usd=0.20, budget_usd=1.00, estimate_usd=0.50) == "run"

    def test_landing_exactly_on_the_budget_runs(self) -> None:
        assert decide_spend(spent_usd=0.50, budget_usd=1.00, estimate_usd=0.50) == "run"

    def test_a_unit_that_would_pass_the_budget_is_skipped(self) -> None:
        assert decide_spend(spent_usd=0.60, budget_usd=1.00, estimate_usd=0.50) == "skip"

    def test_a_spent_budget_skips_even_a_free_unit(self) -> None:
        assert decide_spend(spent_usd=1.00, budget_usd=1.00, estimate_usd=0.0) == "skip"

    def test_an_unpriced_unit_runs_until_the_budget_is_spent(self) -> None:
        """An estimate that cannot be priced is never the reason for a skip."""
        assert decide_spend(spent_usd=0.99, budget_usd=1.00, estimate_usd=None) == "run"
        assert decide_spend(spent_usd=1.01, budget_usd=1.00, estimate_usd=None) == "skip"

    def test_a_zero_budget_skips_everything(self) -> None:
        assert decide_spend(spent_usd=0.0, budget_usd=0.0, estimate_usd=0.01) == "skip"


# ---------------------------------------------------------------------------
# The run-scoped budget
# ---------------------------------------------------------------------------


def _estimate(cost: float | None) -> object:
    def _chunk(
        _purpose: str | None, _model: str, _requests: Sequence[CompletionRequest], _max: int
    ) -> float | None:
        return cost

    return _chunk


class TestSpendBudget:
    def test_no_budget_outside_a_scope(self) -> None:
        assert current_spend_budget() is None
        budget = SpendBudget(budget_usd=1.0, spent=lambda: 0.0)
        with spend_budget(budget):
            assert current_spend_budget() is budget
        assert current_spend_budget() is None

    def test_a_none_scope_installs_nothing(self) -> None:
        with spend_budget(None) as installed:
            assert installed is None
            assert current_spend_budget() is None

    def test_a_declined_chunk_is_counted_with_the_pass_it_happened_in(self) -> None:
        budget = SpendBudget(
            budget_usd=1.0,
            spent=lambda: 0.80,
            estimate_chunk=_estimate(0.30),  # type: ignore[arg-type]
        )
        budget.pass_name = "census"
        requests = [CompletionRequest(prompt="p")] * 3
        assert not budget.allows_chunk(
            purpose="semantic_lint", provider_model="anthropic:m", requests=requests, max_tokens=9
        )
        budget.pass_name = "utility"
        assert not budget.allows_chunk(
            purpose="semantic_lint", provider_model="anthropic:m", requests=requests, max_tokens=9
        )
        assert (budget.skipped_chunks, budget.skipped_requests) == (2, 6)
        assert budget.exhausted_in_pass == "census"

    def test_a_chunk_that_fits_is_allowed_and_not_counted(self) -> None:
        budget = SpendBudget(
            budget_usd=1.0,
            spent=lambda: 0.10,
            estimate_chunk=_estimate(0.30),  # type: ignore[arg-type]
        )
        assert budget.allows_chunk(
            purpose=None, provider_model="anthropic:m", requests=[], max_tokens=1
        )
        assert budget.skipped_chunks == 0


# ---------------------------------------------------------------------------
# The batch adapter consults it between chunks
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_client_around_each_test() -> Generator[None, None, None]:
    llm.set_client(None)
    yield
    llm.set_client(None)
    reset_config()


def _batch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  batch:\n    min_requests: 2\n    max_requests_per_batch: 2\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()


def _succeeded(custom_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(content=[SimpleNamespace(text=text)], stop_reason="end_turn"),
        ),
    )


_REQUESTS = [CompletionRequest(prompt=f"probe {i}", system=f"sys {i}") for i in range(4)]


@pytest.mark.asyncio
async def test_a_chunk_past_the_budget_is_not_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first chunk runs; the second would pass the budget and reads unavailable."""
    _batch_config(tmp_path, monkeypatch)
    spent = {"usd": 0.0}
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.batches.create.side_effect = [SimpleNamespace(id="msgbatch_0")]

    def _retrieve(_id: str) -> SimpleNamespace:
        spent["usd"] = 0.90  # the first chunk was billed
        return SimpleNamespace(processing_status="ended")

    client.messages.batches.retrieve.side_effect = _retrieve
    client.messages.batches.results.side_effect = [
        iter([_succeeded("0", "reply 0"), _succeeded("1", "reply 1")])
    ]
    llm.set_client(client)

    budget = SpendBudget(
        budget_usd=1.0,
        spent=lambda: spent["usd"],
        estimate_chunk=_estimate(0.25),  # type: ignore[arg-type]
    )
    budget.pass_name = "census"
    failures: list[llm.RequestFailure | None] = []
    with spend_budget(budget):
        out, _ = await complete_many_with_provider_model(
            "semantic_lint",
            _REQUESTS,
            max_tokens=10,
            latency_tolerant=True,
            failures_out=failures,
        )

    assert out == ["reply 0", "reply 1", None, None]
    assert failures[2:] == [llm.RequestFailure.UNAVAILABLE] * 2
    assert client.messages.batches.create.call_count == 1  # the second chunk never went
    assert client.messages.create.call_count == 0  # nor did a sequential fallback
    assert (budget.skipped_chunks, budget.skipped_requests) == (1, 2)
    assert budget.exhausted_in_pass == "census"


@pytest.mark.asyncio
async def test_without_a_budget_every_chunk_is_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _batch_config(tmp_path, monkeypatch)
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.batches.create.side_effect = [
        SimpleNamespace(id="msgbatch_0"),
        SimpleNamespace(id="msgbatch_1"),
    ]
    client.messages.batches.retrieve.side_effect = lambda _id: SimpleNamespace(
        processing_status="ended"
    )
    client.messages.batches.results.side_effect = [
        iter([_succeeded("0", "a"), _succeeded("1", "b")]),
        iter([_succeeded("0", "c"), _succeeded("1", "d")]),
    ]
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == ["a", "b", "c", "d"]
    assert client.messages.batches.create.call_count == 2


# ---------------------------------------------------------------------------
# admit_round: pass 1's nightly extraction budget
# ---------------------------------------------------------------------------

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from particles.core.spend_budget import (  # noqa: E402
    MIN_CALIBRATION_SNAPSHOTS,
    ClassRate,
    RoundCandidate,
    admit_round,
)


def _c(
    sid: str, *, est: float = 0.05, worst: float = 0.2, klass: str = "CONVERSATION"
) -> RoundCandidate:
    return RoundCandidate(snapshot_id=sid, klass=klass, estimate_usd=est, worst_case_usd=worst)


def _measured(spent: float, estimated: float, n: int = MIN_CALIBRATION_SNAPSHOTS) -> ClassRate:
    return ClassRate(snapshots=n, spent_usd=spent, estimated_usd=estimated)


class TestAdmitRound:
    def test_round_one_is_bounded_by_worst_cases_within_the_whole_remainder(self) -> None:
        candidates = [_c(f"s{i}") for i in range(30)]
        admission = admit_round(candidates, remaining_usd=4.0, limit_usd=4.0, rates={}, slots=200)
        assert admission.bounded
        assert len(admission.admitted) == 20  # 20 x US$0.20 = US$4.00
        assert admission.worst_case_usd == pytest.approx(4.0)
        assert admission.estimated_usd == pytest.approx(1.0)

    def test_a_measured_class_is_calibrated_within_half_the_remainder(self) -> None:
        candidates = [_c(f"s{i}") for i in range(100)]
        rates = {"CONVERSATION": _measured(spent=0.9, estimated=1.0)}  # factor 0.9
        admission = admit_round(
            candidates, remaining_usd=3.0, limit_usd=4.0, rates=rates, slots=200
        )
        assert not admission.bounded
        assert admission.factor == pytest.approx(0.9)
        # share US$1.50 at 0.9 x 0.05 = US$0.045 each would be 33; the worst-case
        # ceiling (2 x US$3.00 at US$0.20 each) holds it to 30.
        assert len(admission.admitted) == 30
        assert admission.charged_usd <= 1.5

    def test_a_class_with_too_few_priced_snapshots_stays_bounded(self) -> None:
        rates = {"CONVERSATION": _measured(spent=0.05, estimated=0.2, n=1)}
        admission = admit_round([_c("s1")], remaining_usd=4.0, limit_usd=4.0, rates=rates, slots=9)
        assert admission.bounded
        assert admission.factor is None

    def test_a_round_reads_one_class_the_first_that_fits(self) -> None:
        candidates = [
            _c("p1", klass="priority", worst=0.1),
            _c("t1"),
            _c("p2", klass="priority", worst=0.1),
            _c("t2"),
        ]
        admission = admit_round(candidates, remaining_usd=4.0, limit_usd=4.0, rates={}, slots=9)
        assert admission.klass == "priority"
        assert admission.admitted == ("p1", "p2")

    def test_a_snapshot_that_does_not_fit_is_passed_over_not_waited_on(self) -> None:
        candidates = [_c("big", worst=3.0), _c("s1"), _c("s2")]
        admission = admit_round(candidates, remaining_usd=1.0, limit_usd=4.0, rates={}, slots=9)
        assert admission.admitted == ("s1", "s2")
        assert admission.passed_over == ("big",)

    def test_a_snapshot_larger_than_the_whole_limit_is_never_admitted(self) -> None:
        candidates = [_c("huge", worst=5.0), _c("s1")]
        admission = admit_round(candidates, remaining_usd=4.0, limit_usd=4.0, rates={}, slots=9)
        assert admission.too_large == ("huge",)
        assert admission.admitted == ("s1",)

    def test_a_lone_first_snapshot_may_take_all_of_what_is_left(self) -> None:
        rates = {"CONVERSATION": _measured(spent=1.0, estimated=1.0)}
        candidates = [_c("s1", est=1.5, worst=3.5), _c("s2")]
        admission = admit_round(candidates, remaining_usd=2.0, limit_usd=4.0, rates=rates, slots=9)
        assert admission.admitted == ("s1",)
        assert admission.charged_usd == pytest.approx(1.5)

    def test_the_ceiling_and_a_spent_budget_admit_nothing(self) -> None:
        assert not admit_round(
            [_c("s1")], remaining_usd=4.0, limit_usd=4.0, rates={}, slots=0
        ).admitted
        assert not admit_round(
            [_c("s1")], remaining_usd=0.0, limit_usd=4.0, rates={}, slots=9
        ).admitted
        assert admit_round(
            [_c(f"s{i}") for i in range(9)], remaining_usd=4.0, limit_usd=4.0, rates={}, slots=3
        ).admitted == ("s0", "s1", "s2")

    def test_the_factor_is_clamped(self) -> None:
        assert _measured(spent=0.001, estimated=10.0).factor() == pytest.approx(0.1)
        assert _measured(spent=100.0, estimated=1.0).factor() == pytest.approx(10.0)
        assert ClassRate().factor() is None

    @given(
        worsts=st.lists(st.floats(min_value=0.0, max_value=3.0), min_size=1, max_size=40),
        remaining=st.floats(min_value=0.0, max_value=4.0),
    )
    def test_a_bounded_round_never_charges_more_than_the_remainder(
        self, worsts: list[float], remaining: float
    ) -> None:
        candidates = [_c(f"s{i}", worst=w) for i, w in enumerate(worsts)]
        admission = admit_round(
            candidates, remaining_usd=remaining, limit_usd=4.0, rates={}, slots=100
        )
        assert admission.worst_case_usd <= remaining + 1e-9

    @given(
        ests=st.lists(st.floats(min_value=0.0, max_value=2.0), min_size=1, max_size=40),
        remaining=st.floats(min_value=0.01, max_value=4.0),
        factor=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_a_calibrated_round_charges_at_most_its_share(
        self, ests: list[float], remaining: float, factor: float
    ) -> None:
        rates = {"CONVERSATION": _measured(spent=factor, estimated=1.0)}
        candidates = [_c(f"s{i}", est=e, worst=min(4.0, e * 4)) for i, e in enumerate(ests)]
        admission = admit_round(
            candidates, remaining_usd=remaining, limit_usd=4.0, rates=rates, slots=100
        )
        if len(admission.admitted) > 1:
            assert admission.charged_usd <= remaining / 2 + 1e-9
        assert admission.charged_usd <= remaining + 1e-9


def test_a_calibrated_round_holds_its_worst_case_to_twice_what_is_left() -> None:
    """Security review S1: a factor fitted on cheap snapshots cannot admit without bound."""
    rates = {"CONVERSATION": _measured(spent=0.01, estimated=10.0)}  # clamps to 0.1
    candidates = [_c(f"s{i}", est=0.18, worst=0.2) for i in range(200)]
    admission = admit_round(candidates, remaining_usd=3.0, limit_usd=4.0, rates=rates, slots=200)
    assert admission.factor == pytest.approx(0.1)
    assert admission.worst_case_usd <= 6.0 + 1e-9
    assert len(admission.admitted) == 30


@given(
    worsts=st.lists(st.floats(min_value=0.0, max_value=4.0), min_size=1, max_size=40),
    remaining=st.floats(min_value=0.01, max_value=4.0),
)
def test_no_calibrated_round_plans_more_than_twice_the_remainder(
    worsts: list[float], remaining: float
) -> None:
    rates = {"CONVERSATION": _measured(spent=0.01, estimated=10.0)}
    candidates = [_c(f"s{i}", est=w / 20, worst=w) for i, w in enumerate(worsts)]
    admission = admit_round(
        candidates, remaining_usd=remaining, limit_usd=4.0, rates=rates, slots=100
    )
    assert admission.worst_case_usd <= 2 * remaining + 1e-9
