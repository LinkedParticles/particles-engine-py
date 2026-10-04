# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""List-price estimates of LLM work before it is sent.

Pins ``particles/operations/spend_estimate.py``: each figure is list price
(``llm.price_per_mtok``) over the per-call token assumptions the audit
estimate carries, so every expected value below is arithmetic over config.
"""

from __future__ import annotations

import pytest

from particles.config import ProviderSelection, get_config
from particles.llm import CompletionRequest
from particles.operations.spend_estimate import (
    PassEstimate,
    context_call_estimate,
    estimate_batch_chunk,
    extraction_estimate,
    probe_estimate,
    requests_estimate,
)

#: claude-haiku-4-5 lists at $1 / $5 per MTok.
_HAIKU = "claude-haiku-4-5"


@pytest.fixture
def haiku_everywhere() -> None:
    llm = get_config().llm
    for purpose in ("extraction", "semantic_lint", "verification", "abstraction"):
        setattr(llm, purpose, ProviderSelection(provider="anthropic", model=_HAIKU))


def test_probe_estimate_prices_the_probe_figures(haiku_everywhere: None) -> None:
    audit = get_config().audit
    estimate = probe_estimate(10)
    per_call = (
        audit.estimate_probe_input_tokens * 1 + audit.estimate_probe_output_tokens * 5
    ) / 1e6
    assert estimate.calls == 10
    assert estimate.usd == pytest.approx(10 * per_call)
    assert "audit.estimate_probe_input_tokens" in estimate.basis


def test_no_calls_cost_nothing(haiku_everywhere: None) -> None:
    assert probe_estimate(0) == PassEstimate(
        calls=0, usd=0.0, basis="audit.estimate_probe_input_tokens / estimate_probe_output_tokens"
    )


def test_an_unpriced_model_has_no_dollar_figure() -> None:
    get_config().llm.verification = ProviderSelection(provider="anthropic", model="not-a-model")
    estimate = context_call_estimate("verification", 3)
    assert estimate.calls == 3
    assert estimate.usd is None


def test_adding_estimates_sums_calls_and_keeps_unpriced_unpriced(haiku_everywhere: None) -> None:
    priced = probe_estimate(2)
    total = priced + priced
    assert total.calls == 4
    assert total.usd == pytest.approx(2 * (priced.usd or 0))
    assert (priced + PassEstimate(calls=1, usd=None)).usd is None


def test_extraction_estimate_reuses_the_audit_chunking(haiku_everywhere: None) -> None:
    """One call per small source, plus the expected retries and §6.6 probes."""
    estimate = extraction_estimate([4_000, 4_000, 0])
    assert estimate.calls >= 2  # the empty source costs nothing
    assert estimate.usd is not None and estimate.usd > 0
    assert "extraction.estimate_prompt_overhead_tokens" in estimate.basis
    assert extraction_estimate([]) == PassEstimate(calls=0, usd=0.0, basis="")


def test_a_batch_chunk_is_priced_at_the_batch_discount(haiku_everywhere: None) -> None:
    llm = get_config().llm
    requests = [CompletionRequest(prompt="x" * 4_000, system="y" * 400)] * 2
    cost = estimate_batch_chunk("semantic_lint", f"anthropic:{_HAIKU}", requests, 10_000)
    output = get_config().audit.estimate_probe_output_tokens  # under max_tokens
    full = (2 * 4_400 / 4 * 1 + 2 * output * 5) / 1e6
    assert cost == pytest.approx(full * (1.0 - llm.batch_discount))


def test_a_chunk_output_never_exceeds_its_max_tokens(haiku_everywhere: None) -> None:
    llm = get_config().llm
    requests = [CompletionRequest(prompt="")]
    cost = estimate_batch_chunk("extraction", f"anthropic:{_HAIKU}", requests, 100)
    assert cost == pytest.approx(100 * 5 / 1e6 * (1.0 - llm.batch_discount))


def test_an_unpriced_chunk_has_no_estimate() -> None:
    requests = [CompletionRequest(prompt="p")]
    assert estimate_batch_chunk("semantic_lint", "anthropic:not-a-model", requests, 10) is None


def test_requests_are_priced_from_their_own_size_at_list_price(haiku_everywhere: None) -> None:
    """the utility re-mine is priced from the prompts it will send."""
    requests = [CompletionRequest(prompt="x" * 4_000, system="y" * 400)] * 2
    estimate = requests_estimate("semantic_lint", requests, 400)
    output = get_config().audit.estimate_probe_output_tokens
    assert estimate.calls == 2
    assert estimate.usd == pytest.approx((2 * 4_400 / 4 * 1 + 2 * output * 5) / 1e6)


def test_no_requests_cost_nothing() -> None:
    assert requests_estimate("semantic_lint", [], 400) == PassEstimate(calls=0, usd=0.0)
