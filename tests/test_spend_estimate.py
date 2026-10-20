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
from particles.operations.audit import ExtractionCall, output_density
from particles.operations.spend_estimate import (
    PassEstimate,
    chars_per_token,
    context_call_estimate,
    estimate_batch_chunk,
    extraction_estimate,
    planned_extraction_estimate,
    probe_estimate,
    requests_estimate,
    rides_batches,
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


# --- consolidation pricing: batches, transcripts, the cached prefix ---------

_SONNET = "claude-sonnet-5-5"


@pytest.fixture
def sonnet_extraction() -> None:
    get_config().llm.extraction = ProviderSelection(provider="anthropic", model=_SONNET)


def _calls(n: int, *, chars: int = 6_000, density: float = 1.0) -> list[ExtractionCall]:
    return [ExtractionCall(chars=chars, density=density) for _ in range(n)]


def test_a_batched_extraction_is_priced_at_the_batch_discount(sonnet_extraction: None) -> None:
    calls = _calls(10)
    batched = planned_extraction_estimate(calls, entries=10, batched=True)
    get_config().llm.batch.enabled = False
    sequential = planned_extraction_estimate(calls, entries=10, batched=True)
    assert batched.usd is not None and sequential.usd is not None
    assert batched.usd < sequential.usd
    assert "llm.batch_discount" in batched.basis
    assert "llm.batch_discount" not in sequential.basis


def test_only_the_native_adapter_with_enough_requests_rides_a_batch(
    sonnet_extraction: None,
) -> None:
    llm = get_config().llm
    assert rides_batches("extraction", llm.batch.min_requests)
    assert not rides_batches("extraction", llm.batch.min_requests - 1)
    llm.extraction = ProviderSelection(provider="local", model="qwen3:14b")
    assert not rides_batches("extraction", 100)


def test_a_transcript_writes_less_per_character(sonnet_extraction: None) -> None:
    scale = get_config().audit.estimate_output_scale_by_source_type["CONVERSATION"]
    prose = planned_extraction_estimate(_calls(4), entries=4, batched=False)
    transcript = planned_extraction_estimate(
        _calls(4, density=output_density("CONVERSATION")), entries=4, batched=False
    )
    assert output_density("CONVERSATION") == scale < 1.0
    assert output_density("LOCAL_MARKDOWN") == 1.0
    assert transcript.usd is not None and prose.usd is not None
    assert transcript.usd < prose.usd
    assert "audit.estimate_output_scale_by_source_type" in transcript.basis


def test_delta_context_is_billed_as_input(sonnet_extraction: None) -> None:
    bare = planned_extraction_estimate([ExtractionCall(chars=6_000)], entries=1, batched=False)
    with_context = planned_extraction_estimate(
        [ExtractionCall(chars=6_000, context_chars=4_000)], entries=1, batched=False
    )
    price = get_config().llm.price_for(get_config().llm.for_purpose("extraction"))
    assert price is not None and bare.usd is not None and with_context.usd is not None
    context_tokens = 4_000 / chars_per_token(get_config().llm.for_purpose("extraction"))
    # The context is input on the call and on its share of the expected retries.
    assert with_context.usd - bare.usd == pytest.approx(context_tokens * price.input / 1e6, rel=0.1)
    assert with_context.usd > bare.usd


def test_the_cached_prefix_is_read_at_the_models_multiplier(sonnet_extraction: None) -> None:
    llm = get_config().llm
    calls = _calls(10)
    cached = planned_extraction_estimate(calls, entries=10, batched=False)
    llm.cache_read_price_multiplier_by_model = {_SONNET: 0.5}
    dearer = planned_extraction_estimate(calls, entries=10, batched=False)
    llm.prompt_cache.enabled = False
    uncached = planned_extraction_estimate(calls, entries=10, batched=False)
    assert cached.usd is not None and dearer.usd is not None and uncached.usd is not None
    prefix = get_config().extraction.estimate_cached_prefix_tokens
    price = llm.price_for(llm.for_purpose("extraction"))
    assert price is not None
    # Every call after the first reads the prefix (retries included), at 0.05x
    # and then at 0.5x.
    reads = cached.calls - 1
    assert dearer.usd - cached.usd == pytest.approx(reads * prefix * price.input * 0.45 / 1e6)
    assert uncached.usd > dearer.usd


def test_input_is_counted_at_the_models_chars_per_token() -> None:
    assert chars_per_token(ProviderSelection(provider="anthropic", model=_SONNET)) == 2.98
    assert chars_per_token(ProviderSelection(provider="anthropic", model=_HAIKU)) == 4.0


def test_the_judge_is_priced_at_its_own_chars_per_token() -> None:
    get_config().llm.use_judge = ProviderSelection(provider="anthropic", model=_SONNET)
    requests = [CompletionRequest(prompt="x" * 9_000)] * 10
    judge = requests_estimate("use_judge", requests, 1024)
    figure = get_config().utility.mining.estimate_chars_per_token
    output = get_config().audit.estimate_probe_output_tokens
    assert judge.usd == pytest.approx((10 * 9_000 / figure * 2 + 10 * output * 10) / 1e6)
    assert "utility.mining.estimate_chars_per_token" in judge.basis
    batched = requests_estimate("use_judge", requests, 1024, batched=True)
    discount = get_config().llm.batch_discount
    assert batched.usd == pytest.approx((judge.usd or 0) * (1 - discount))
