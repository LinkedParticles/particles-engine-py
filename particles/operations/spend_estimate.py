# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""List-price estimates of LLM work before it is sent.

The consolidation cycle's dollar budget (``consolidation.budget_usd``) and its
``--dry-run`` both need what a unit of work would cost before any call is made.
Nothing here is measured: every figure is list price (``llm.price_per_mtok``)
over *estimated* tokens, from the per-call assumptions the audit estimate
already carries (``audit.estimate_*``, ``extraction.estimate_prompt_overhead_tokens``).
The figure a run prints afterwards (``render_usage_line``, "LLM usage: … ≈ $X
at list price") is the measured one, priced from the tokens the provider
reported; the two are meant to be compared.

Two consumers:

- :class:`PassEstimate` prices one pass from the counts its gather found
  (snapshots to extract, pairs to probe, groups to judge). A pass whose calls
  ride a Message Batches job in the consolidation cycle (pass 1's pooled
  extraction and the utility judge) is priced at ``llm.batch_discount`` when
  :func:`rides_batches` says the job would be submitted as a batch; every
  other pass at full list price. Extraction's cached instruction prefix is
  priced at the model's cache-read multiplier
  (``llm.cache_read_price_multiplier_by_model``).
- :func:`estimate_batch_chunk` prices one Message Batches chunk for the budget
  the batch adapter consults between chunks; there the batch discount is
  applied, because the chunk is a batch by construction.

Each stays an upper bound on what the run is billed. The output figures lean
high (the extraction estimate prices the high end of its band), and a set the
run's batch-wait budget sends sequentially pays full price inside that margin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from particles.config import ProviderSelection, TokenPrice, get_config, lookup_by_model
from particles.operations.audit import (
    AuditEstimate,
    ExtractionCall,
    chunk_sizes,
    estimate_extraction_calls,
    extraction_output_model,
)

if TYPE_CHECKING:
    from particles.llm.registry import CompletionRequest


@dataclass(frozen=True)
class PassEstimate:
    """What one pass would send, priced at list price."""

    #: LLM calls the pass would make (an upper bound where the count depends
    #: on answers not yet given, such as second readings of flags).
    calls: int = 0
    #: US$ at list price; ``None`` when a model it would spend on is unpriced.
    usd: float | None = 0.0
    #: The config keys the per-call token figures came from.
    basis: str = ""

    def __add__(self, other: PassEstimate) -> PassEstimate:
        usd = None if self.usd is None or other.usd is None else self.usd + other.usd
        basis = "; ".join(b for b in (self.basis, other.basis) if b)
        return PassEstimate(calls=self.calls + other.calls, usd=usd, basis=basis)


def _price(price: TokenPrice | None, input_tokens: float, output_tokens: float) -> float | None:
    if price is None:
        return None
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


def chars_per_token(selection: ProviderSelection) -> float:
    """Characters per input token for ``selection``'s tokenizer.

    The factor ``benchmark_memory.chars_per_token_by_model`` holds for the
    benchmark harnesses, keyed like ``llm.price_per_mtok`` (claude-sonnet-5-5
    ships at 2.98); a model with no entry gets the 4.0 rule of thumb. A newer
    tokenizer sends about a third more tokens for the same text, so dividing
    by 4 would price its input low.
    """
    cfg = get_config().benchmark_memory
    factor, _ = lookup_by_model(cfg.chars_per_token_by_model, selection.provider, selection.model)
    return cfg.chars_per_token if factor is None else factor


def _prompt_chars_per_token(purpose: str | None, selection: ProviderSelection) -> float:
    """:func:`chars_per_token`, or the judge's denser figure for a ``use_judge`` prompt."""
    factor = chars_per_token(selection)
    if purpose == "use_judge":
        return min(factor, get_config().utility.mining.estimate_chars_per_token)
    return factor


def rides_batches(purpose: str, requests: int) -> bool:
    """Whether ``requests`` latency-tolerant calls on ``purpose`` would go as a batch.

    The registry's own test (``complete_many``): ``llm.batch.enabled``, at least
    ``llm.batch.min_requests`` requests, and a provider with a batch API, which
    today is the native Anthropic adapter alone.
    """
    llm = get_config().llm
    return (
        llm.batch.enabled
        and requests >= llm.batch.min_requests
        and llm.for_purpose(purpose).provider == "anthropic"
    )


def _batch_factor(batched: bool) -> float:
    return 1.0 - get_config().llm.batch_discount if batched else 1.0


def price_calls(
    purpose: str, calls: int, *, input_tokens: int, output_tokens: int, basis: str
) -> PassEstimate:
    """``calls`` calls on ``purpose``'s model at the given per-call token figures."""
    if calls <= 0:
        return PassEstimate(calls=0, usd=0.0, basis=basis)
    llm = get_config().llm
    price = llm.price_for(llm.for_purpose(purpose))
    return PassEstimate(
        calls=calls,
        usd=_price(price, calls * input_tokens, calls * output_tokens),
        basis=basis,
    )


def probe_estimate(calls: int) -> PassEstimate:
    """Contradiction-shaped probes on ``llm.semantic_lint``."""
    audit = get_config().audit
    return price_calls(
        "semantic_lint",
        calls,
        input_tokens=audit.estimate_probe_input_tokens,
        output_tokens=audit.estimate_probe_output_tokens,
        basis="audit.estimate_probe_input_tokens / estimate_probe_output_tokens",
    )


def context_call_estimate(purpose: str, calls: int) -> PassEstimate:
    """Context-rich calls (passages, groups of claims) at the second-reading figures."""
    audit = get_config().audit
    return price_calls(
        purpose,
        calls,
        input_tokens=audit.estimate_verify_input_tokens,
        output_tokens=audit.estimate_verify_output_tokens,
        basis="audit.estimate_verify_input_tokens / estimate_verify_output_tokens",
    )


def extraction_estimate(source_chars: Sequence[int]) -> PassEstimate:
    """Extracting sources of these sizes whole, unbatched, with the §6.6 probes.

    Each source is cut by the audit's chunk rule. The consolidation cycle plans
    its calls instead (:func:`planned_extraction_estimate`).
    """
    calls = [ExtractionCall(chars=size) for n in source_chars if n > 0 for size in chunk_sizes(n)]
    return planned_extraction_estimate(
        calls, entries=sum(1 for n in source_chars if n > 0), batched=False
    )


def planned_extraction_estimate(
    calls: Sequence[ExtractionCall], *, entries: int, batched: bool
) -> PassEstimate:
    """Extraction calls a caller planned, with the §6.6 probes extraction makes.

    Reuses the audit's estimate (``audit.estimate_extraction_calls``): its
    output model (a per-call base plus a rate per source character, scaled for
    the call's source type, with the retries a call near
    ``extraction.max_tokens`` makes) and its expected reconcile probes per
    call. Input is counted at the extraction model's characters per token
    (:func:`chars_per_token`), plus the cached instruction prefix
    (``extraction.estimate_cached_prefix_tokens``) at the model's cache-read
    price. ``batched`` says the calls are submitted as one latency-tolerant
    set (the cycle's pooled pass 1); the batch discount then applies when
    :func:`rides_batches` agrees. The probes are not batched. The audit's own
    census probes are left out; the consolidation census is its own pass.
    """
    selection = get_config().llm.for_purpose("extraction")
    estimate = estimate_extraction_calls(
        calls, entries=entries, chars_per_token=chars_per_token(selection)
    )
    if estimate.estimated_llm_calls == 0:
        return PassEstimate(calls=0, usd=0.0, basis="")
    calls_made = estimate.estimated_llm_calls + round(estimate.expected_retries)
    batch = batched and rides_batches("extraction", calls_made)
    extraction = _price(
        estimate.extraction_price,
        estimate.input_tokens + estimate.retry_input_tokens,
        estimate.output_tokens_high,
    )
    prefix = _cached_prefix_usd(estimate, selection, calls_made)
    if extraction is not None and prefix is not None:
        extraction = (extraction + prefix) * _batch_factor(batch)
    audit = get_config().audit
    probes = _price(
        estimate.probe_price,
        estimate.reconcile_probes * audit.estimate_probe_input_tokens,
        estimate.reconcile_probes * audit.estimate_probe_output_tokens,
    )
    usd = (
        None
        if extraction is None or (estimate.reconcile_probes and probes is None)
        else extraction + (probes or 0.0)
    )
    return PassEstimate(
        calls=estimate.estimated_llm_calls + round(estimate.expected_retries),
        usd=usd,
        basis=(
            f"{estimate.output_basis}, "
            + (
                "audit.estimate_output_scale_by_source_type, "
                if any(call.density != 1.0 for call in calls)
                else ""
            )
            + "extraction.estimate_prompt_overhead_tokens, "
            "extraction.estimate_cached_prefix_tokens, "
            "audit.estimate_extraction_retry_rate, "
            "audit.estimate_reconcile_probes_per_extraction_call"
            + (", llm.batch_discount" if batch else "")
        ),
    )


@dataclass(frozen=True)
class ExtractionRange:
    """An unbatched extraction run's band at list price, as a person reads it.

    ``high_usd`` is :func:`planned_extraction_estimate`'s figure for the same
    calls; ``low_usd`` spends no retries and no probes, and ``ceiling_usd``
    bills every call at its full output budget. All three are ``None`` when
    the extraction or probe model is unpriced.
    """

    calls: int
    model: str
    low_usd: float | None
    high_usd: float | None
    ceiling_usd: float | None
    wall_seconds: float


def extraction_range(calls: Sequence[ExtractionCall], *, entries: int) -> ExtractionRange:
    """The band of :func:`planned_extraction_estimate`, unbatched, with its wall time."""
    selection = get_config().llm.for_purpose("extraction")
    estimate = estimate_extraction_calls(
        calls, entries=entries, chars_per_token=chars_per_token(selection)
    )
    calls_made = estimate.estimated_llm_calls + round(estimate.expected_retries)
    audit = get_config().audit
    wall = (
        estimate.output_tokens_expected / audit.estimate_extraction_output_tokens_per_second
        + estimate.reconcile_probes * audit.estimate_seconds_per_probe
    )
    if calls_made == 0:
        return ExtractionRange(0, selection.model, 0.0, 0.0, 0.0, 0.0)
    prefix = _cached_prefix_usd(estimate, selection, calls_made)
    probes = _price(
        estimate.probe_price,
        estimate.reconcile_probes * audit.estimate_probe_input_tokens,
        estimate.reconcile_probes * audit.estimate_probe_output_tokens,
    )
    with_retries = estimate.input_tokens + estimate.retry_input_tokens
    low = _price(estimate.extraction_price, estimate.input_tokens, estimate.output_tokens_low)
    high = _price(estimate.extraction_price, with_retries, estimate.output_tokens_high)
    ceiling = _price(estimate.extraction_price, with_retries, estimate.output_tokens_ceiling)
    if prefix is None or (estimate.reconcile_probes and probes is None):
        low = high = ceiling = None
    else:
        extra = prefix + (probes or 0.0)
        low = None if low is None else low + prefix
        high = None if high is None else high + extra
        ceiling = None if ceiling is None else ceiling + extra
    return ExtractionRange(calls_made, selection.model, low, high, ceiling, wall)


def _cached_prefix_usd(
    estimate: AuditEstimate, selection: ProviderSelection, calls: int
) -> float | None:
    """The instruction prefix every extraction call sends, at its cached price.

    With prompt caching on (``llm.prompt_cache.enabled``, Anthropic only) one
    call writes the prefix and the rest read it; otherwise each call sends it
    as plain input.
    """
    price = estimate.extraction_price
    if price is None:
        return None
    llm = get_config().llm
    tokens = get_config().extraction.estimate_cached_prefix_tokens
    if not (llm.prompt_cache.enabled and selection.provider == "anthropic"):
        return calls * tokens * price.input / 1_000_000
    write = tokens * price.input * llm.cache_write_price_multiplier
    reads = max(0, calls - 1) * tokens * price.input * llm.cache_read_multiplier_for(selection)
    return (write + reads) / 1_000_000


def _output_tokens_per_request(
    purpose: str | None, provider: str, model: str, cap: int, prompt_chars: int
) -> int:
    """The expected output of one request on ``purpose``, never above its ``max_tokens``.

    An extraction request's output scales with the source it carries; its
    prompt's length stands in for that, instructions included, so it leans high.
    """
    audit = get_config().audit
    match purpose:
        case "extraction":
            figure = round(extraction_output_model(provider, model).expected(prompt_chars))
        case "semantic_lint" | "use_judge":
            figure = audit.estimate_probe_output_tokens
        case "verification":
            figure = audit.estimate_verify_output_tokens
        case _:
            figure = cap
    return min(figure, cap)


def estimate_batch_chunk(
    purpose: str | None,
    provider_model: str,
    requests: Sequence[CompletionRequest],
    max_tokens: int,
) -> float | None:
    """US$ one batched chunk would cost: input at the model's chars per token, expected output.

    The :class:`~particles.llm.spend_budget.SpendBudget` estimator for the
    consolidation cycle. The batch discount (``llm.batch_discount``) is
    applied; prompt-cache reads are not, so a cached prefix reads high.
    """
    provider, _, model = provider_model.partition(":")
    llm = get_config().llm
    price = llm.price_for(ProviderSelection(provider=provider, model=model))
    if price is None:
        return None
    chars = sum(len(r.prompt) + len(r.system or "") + len(r.cache_prefix or "") for r in requests)
    output = sum(
        _output_tokens_per_request(purpose, provider, model, max_tokens, len(r.prompt))
        for r in requests
    )
    selection = ProviderSelection(provider=provider, model=model)
    cost = _price(price, chars / _prompt_chars_per_token(purpose, selection), output)
    return None if cost is None else cost * (1.0 - llm.batch_discount)


def requests_estimate(
    purpose: str,
    requests: Sequence[CompletionRequest],
    max_tokens: int,
    *,
    batched: bool = False,
) -> PassEstimate:
    """These exact requests at list price: input at the model's chars per token, expected output.

    For a pass whose prompts are built before it runs (the utility judge),
    so the estimate reads the real prompt sizes rather than a
    per-call assumption. ``batched`` says the requests go as one
    latency-tolerant set; the batch discount then applies when
    :func:`rides_batches` agrees. Prompt-cache reads are not discounted.
    """
    if not requests:
        return PassEstimate(calls=0, usd=0.0, basis="")
    llm = get_config().llm
    selection = llm.for_purpose(purpose)
    price = llm.price_for(selection)
    chars = sum(len(r.prompt) + len(r.system or "") + len(r.cache_prefix or "") for r in requests)
    output = sum(
        _output_tokens_per_request(
            purpose, selection.provider, selection.model, max_tokens, len(r.prompt)
        )
        for r in requests
    )
    batch = batched and rides_batches(purpose, len(requests))
    cost = _price(price, chars / _prompt_chars_per_token(purpose, selection), output)
    return PassEstimate(
        calls=len(requests),
        usd=None if cost is None else cost * _batch_factor(batch),
        basis="prompt size at "
        + (
            "utility.mining.estimate_chars_per_token; "
            if purpose == "use_judge"
            else "benchmark_memory.chars_per_token_by_model; "
        )
        + "audit.estimate_probe_output_tokens"
        + ("; llm.batch_discount" if batch else ""),
    )
