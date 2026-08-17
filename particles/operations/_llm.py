"""Shared LLM helper for LLM-assisted operations (lint, links suggest).

A thin wrapper around the ``semantic_lint`` completion purpose that
swallows API errors and returns the plain-text reply (or ``None`` on failure).
Consumers import ``_llm_call`` from here so the configured provider (and any
``set_client`` injection in tests) is honoured uniformly.

Relocated from ``operations/lint/_llm.py`` in 0.46.0 once the
``links_suggest`` operation became a second consumer — per the hub-and-spoke
package-layout convention in the root ``AGENTS.md``. The hardcoded model string
this module used to carry was removed; the model now resolves from
``config.llm.semantic_lint`` (falling back to ``config.llm.default``).

Circuit breaker
--------------------------
An *account-level* failure — bad / missing key (401), no permission (403), or an
out-of-credits billing error (a 400 "credit balance too low") — fails **every**
call, so the seam trips a process-wide breaker: subsequent calls short-circuit
to ``None`` without touching the API for ``config.llm.unavailable_backoff_seconds``
(then one half-open probe), and the condition is logged **once** rather than once
per call. A *per-call* failure (one bad prompt, a transient blip) does not trip —
it logs and returns ``None`` as before. The breaker is transparent: every
consumer still just sees ``None`` and degrades gracefully; a read surface can call
:func:`llm_circuit_open` to report that semantic analysis was skipped.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from particles.config import get_config, register_reset_hook
from particles.llm.errors import is_account_level_failure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from particles.llm import CompletionRequest, LLMPurpose

log = logging.getLogger(__name__)

# Process-wide circuit-breaker deadline (``time.monotonic`` seconds). While
# ``time.monotonic() < _tripped_until`` the seam short-circuits semantic LLM
# calls after an account-level failure.
_tripped_until: float = 0.0

# Process-wide count of failed semantic LLM calls (per-call failures AND
# account-level trips). Read surfaces diff this across an operation to
# disclose how many probes were skipped (honesty stance) — a
# refusal or transient failure otherwise silently under-counts findings
# without tripping the breaker or setting ``semantic_skipped``.
_failure_count: int = 0


def llm_failure_count() -> int:
    """Monotonic count of failed semantic LLM calls in this process.

    Diff it around an operation (``before = llm_failure_count()`` … ``delta =
    llm_failure_count() - before``) rather than resetting, so concurrent
    readers cannot clobber each other. Reset (with the breaker) only via
    ``reset_config()``.
    """
    return _failure_count


def _reset_llm_state() -> None:
    """Reset the breaker + failure counter (uniform test isolation)."""
    global _tripped_until, _failure_count
    _tripped_until = 0.0
    _failure_count = 0


register_reset_hook(_reset_llm_state)


#: The account-level predicate now lives in the Client-layer ``llm`` package
#: (``particles.llm.errors``) so the extraction seam — which cannot import
#: Engine code — can classify the same failures and abort a bulk run instead of
#: repeating a doomed call per snapshot. Re-exported under the historical
#: private name so this module's call site and its tests are unchanged.
_is_account_level = is_account_level_failure


def llm_circuit_open() -> bool:
    """True while the account-level circuit breaker is tripped.

    A read surface uses this to set ``semantic_skipped`` — telling the client
    semantic analysis was skipped because the LLM is unavailable, rather than
    silently returning fewer results.
    """
    return time.monotonic() < _tripped_until


async def _llm_call(
    prompt: str,
    max_tokens: int = 200,
    system: str | None = None,
    response_schema: dict[str, Any] | None = None,
    purpose: LLMPurpose = "semantic_lint",
) -> str | None:
    # Portability note on the small ``max_tokens`` defaults used by the probe
    # call sites (100–200 tokens): safe on models where omitting ``thinking``
    # runs without thinking (e.g. claude-sonnet-4-6), but on adaptive-by-default
    # models (Sonnet 5, Opus 4.7+, Fable 5) thinking spend counts against
    # ``max_tokens`` and can consume the whole budget before any text block —
    # repointing ``llm.semantic_lint`` at one of those needs a bigger budget or
    # an explicit thinking-off configuration.
    global _tripped_until, _failure_count
    if time.monotonic() < _tripped_until:
        # Breaker open: an account-level failure is in effect. Do not hammer a
        # dead API — the trip was already logged once.
        return None
    try:
        from particles.llm import complete

        # ``system`` carries trusted instructions when the caller fences
        # untrusted content into ``prompt`` (F3 hardening); it defaults to None
        # so existing call sites are unchanged. ``response_schema`` (
        # §5) rides through for the JSON-shaped probes; None for text protocols.
        # ``purpose`` selects the per-purpose provider routing;
        # the abstraction pass passes "abstraction", every prior call
        # site keeps the default. The circuit breaker is deliberately shared
        # across purposes — an account-level failure fails them all.
        result = await complete(
            purpose,
            prompt,
            max_tokens=max_tokens,
            system=system,
            response_schema=response_schema,
        )
        _tripped_until = 0.0  # success (incl. the half-open probe) clears the breaker
        return result
    except Exception as exc:
        _failure_count += 1
        if _is_account_level(exc):
            backoff = get_config().llm.unavailable_backoff_seconds
            _tripped_until = time.monotonic() + backoff
            log.error(
                "LLM unavailable (account-level: %s) — suppressing semantic LLM "
                "calls for %ss; fix the API key / credit balance to re-enable.",
                exc,
                backoff,
            )
        else:
            log.error("LLM call failed: %s", exc)
        return None


async def _llm_call_many(
    requests: Sequence[CompletionRequest],
    max_tokens: int = 200,
    response_schema: dict[str, Any] | None = None,
    purpose: LLMPurpose = "semantic_lint",
    *,
    latency_tolerant: bool = False,
) -> list[str | None]:
    """The :func:`_llm_call` contract for a *set* of independent prompts.

    Same breaker, same failure counting, same "``None`` means degrade" promise —
    but the whole set goes to :func:`particles.llm.complete_many`, which submits
    it as one half-price batch when ``latency_tolerant`` is set and the
    configured provider supports batching. Results are positionally aligned with
    ``requests``.

    ``latency_tolerant`` is the caller's assertion that no human is waiting:
    the nightly consolidation cycle sets it; ``particles lint`` and the
    interactive audit do not, and get the sequential path unchanged.

    An open breaker short-circuits the whole set without touching the API, and
    an account-level failure raised mid-set trips the breaker once — the
    per-set analogue of not hammering a dead API once per probe.
    """
    global _tripped_until, _failure_count
    if not requests:
        return []
    if time.monotonic() < _tripped_until:
        return [None] * len(requests)
    try:
        from particles.llm import complete_many

        results = await complete_many(
            purpose,
            requests,
            max_tokens=max_tokens,
            response_schema=response_schema,
            latency_tolerant=latency_tolerant,
        )
    except Exception as exc:
        _failure_count += len(requests)
        if _is_account_level(exc):
            backoff = get_config().llm.unavailable_backoff_seconds
            _tripped_until = time.monotonic() + backoff
            log.error(
                "LLM unavailable (account-level: %s) — suppressing semantic LLM "
                "calls for %ss; fix the API key / credit balance to re-enable.",
                exc,
                backoff,
            )
        else:
            log.error("Batched LLM call failed for %d request(s): %s", len(requests), exc)
        return [None] * len(requests)
    # The job ran. Individual dead requests still count toward the disclosure
    # counter so a read surface can report how many probes went unanswered.
    failed = sum(1 for r in results if r is None)
    _failure_count += failed
    if failed < len(results):
        _tripped_until = 0.0  # at least one success clears the breaker
    return results
