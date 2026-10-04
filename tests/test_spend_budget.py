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
