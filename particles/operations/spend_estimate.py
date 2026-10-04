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
  (snapshots to extract, pairs to probe, groups to judge). Batch and
  prompt-cache discounts are not applied, so it reads high for a batched pass.
- :func:`estimate_batch_chunk` prices one Message Batches chunk for the budget
  the batch adapter consults between chunks; there the batch discount is
  applied, because the chunk is a batch by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from particles.config import ProviderSelection, TokenPrice, get_config, lookup_by_model
from particles.operations.audit import estimate_extraction

if TYPE_CHECKING:
    from particles.llm.registry import CompletionRequest

#: ~4 characters per token, the conversion every estimate here uses.
_CHARS_PER_TOKEN = 4


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
    """Extracting sources of these sizes, with the §6.6 probes extraction makes.

    Reuses the audit's estimate (``audit.estimate_extraction``): its chunking
    rule, its per-call output figure and retry rate, and its expected
    reconcile probes per call. The audit's own census probes are left out;
    the consolidation census is estimated as its own pass.
    """
    estimate = estimate_extraction(source_chars)
    if estimate.estimated_llm_calls == 0:
        return PassEstimate(calls=0, usd=0.0, basis="")
    extraction = _price(
        estimate.extraction_price,
        estimate.input_tokens + estimate.retry_input_tokens,
        estimate.output_tokens_high,
    )
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
            f"{estimate.output_tokens_per_call_source}, "
            "extraction.estimate_prompt_overhead_tokens, "
            "audit.estimate_extraction_retry_rate, "
            "audit.estimate_reconcile_probes_per_extraction_call"
        ),
    )


def _output_tokens_per_request(purpose: str | None, provider: str, model: str, cap: int) -> int:
    """The expected output of one request on ``purpose``, never above its ``max_tokens``."""
    audit = get_config().audit
    match purpose:
        case "extraction":
            per_call, _ = lookup_by_model(
                audit.estimate_output_tokens_per_extraction_call_by_model, provider, model
            )
            figure = (
                per_call
                if per_call is not None
                else audit.estimate_output_tokens_per_extraction_call
            )
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
    """US$ one batched chunk would cost: input at ~4 chars per token, expected output.

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
    output = len(requests) * _output_tokens_per_request(purpose, provider, model, max_tokens)
    cost = _price(price, chars / _CHARS_PER_TOKEN, output)
    return None if cost is None else cost * (1.0 - llm.batch_discount)


def requests_estimate(
    purpose: str, requests: Sequence[CompletionRequest], max_tokens: int
) -> PassEstimate:
    """These exact requests at list price: input at ~4 chars per token, expected output.

    For a pass whose prompts are built before it runs (the utility judge),
    so the estimate reads the real prompt sizes rather than a
    per-call assumption. Batch and prompt-cache discounts are not applied.
    """
    if not requests:
        return PassEstimate(calls=0, usd=0.0, basis="")
    llm = get_config().llm
    selection = llm.for_purpose(purpose)
    price = llm.price_for(selection)
    chars = sum(len(r.prompt) + len(r.system or "") + len(r.cache_prefix or "") for r in requests)
    per_call = _output_tokens_per_request(purpose, selection.provider, selection.model, max_tokens)
    return PassEstimate(
        calls=len(requests),
        usd=_price(price, chars / _CHARS_PER_TOKEN, len(requests) * per_call),
        basis="prompt size at ~4 chars per token; audit.estimate_probe_output_tokens",
    )
