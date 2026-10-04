# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""First-run memory audit — the activation-moment census.

``run_memory_audit`` composes the **existing** machinery and adds no detection
of its own: the standard extract pipeline over the harvested snapshots, then
``collect_cards`` finder-normalization seam — **uncapped and
snooze-unfiltered** (an audit is a census, not a worklist) — leverage-ranked
exemplars per class, and ``get_quality_report`` header counts, assembled into
an :class:`AuditReport`.

The harvest itself (file walk, sentinel filter, transcript distillation) is
Surface-side — it reuses the helpers in
``particles/api/cli/_claude_code.py`` and lives with the ``particles audit``
verb (``particles/api/cli/audit.py``); this Engine module receives the
harvested entry ids. ``estimate_extraction`` is the §4 dry-run cost estimate
(byte counts × the extraction chunker), computed before anything touches the
store.

The renderer bakes the ADR's honesty stance into the copy (§5/§6, approved
verbatim 2026-07-11): hedged class labels ("potential", "likely-",
"probably-"), the uncalibrated-confidence footnote, disclosed skips, and a
next verb per class — the report is a door into the existing loops, not a
dead end.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import TokenPrice, get_config, lookup_by_model
from particles.core.contradiction_disclosure import census_sides, group_pairs
from particles.core.progress import ProgressEvent
from particles.core.schema import (
    ExtractionStatus,
    JudgeVerdictKind,
    QualityReport,
    SuggestMode,
)
from particles.core.status import Status
from particles.llm.usage import LLMUsage, format_usd, render_usage_line, track_usage
from particles.operations._llm import (
    llm_circuit_open,
    llm_failure_count,
    llm_trip_count,
    llm_unavailable_cause,
)
from particles.operations.curation.cards import CardKind, CurationCard
from particles.operations.curation.collect import collect_cards
from particles.operations.curation.leverage import contested_ids_from, score_cards
from particles.operations.curation.session import _attach_particle_briefs
from particles.operations.lint import ContradictionProbeControl
from particles.operations.lint.contradictions import one_line
from particles.operations.quality import get_quality_report
from particles.store.particle_store import get_census_records, get_particle_ids_for_entries

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Cost estimate — pure, computed before anything is deposited
# ---------------------------------------------------------------------------


class AuditCost(BaseModel):
    """A dollar range at list price: expected low/high and a hard ceiling (US$)."""

    expected_low_usd: float
    expected_high_usd: float
    ceiling_usd: float


def price_range(
    *,
    input_tokens: int,
    output_tokens_low: int,
    output_tokens_high: int,
    output_tokens_ceiling: int,
    probe_input_tokens: int,
    probe_output_tokens: int,
    extraction_price: TokenPrice | None,
    probe_price: TokenPrice | None,
    retry_input_tokens: int = 0,
    verify_input_tokens: int = 0,
    verify_output_tokens: int = 0,
    verify_price: TokenPrice | None = None,
) -> AuditCost | None:
    """Price the audit's token projection, or ``None`` when any spending model is unpriced.

    Pure arithmetic. The low end spends no retries and no probes; the high end
    and the ceiling add the retries' re-sent input and every probe. The probe
    tokens are the audit cap plus the expected §6.6 probes, so the expected
    range spans no probes (low) to all of them (high). The second readings of
    flagged contradictions are priced the same way: none (low) to
    every one spent (high). A partial figure would read as a total, so a single
    unpriced component that is actually spent on prices nothing.
    """
    probes_spent = probe_input_tokens > 0 or probe_output_tokens > 0
    verify_spent = verify_input_tokens > 0 or verify_output_tokens > 0
    if (
        extraction_price is None
        or (probes_spent and probe_price is None)
        or (verify_spent and verify_price is None)
    ):
        return None

    def _extraction(inputs: int, output_tokens: int) -> float:
        return (
            inputs * extraction_price.input + output_tokens * extraction_price.output
        ) / 1_000_000

    probes = 0.0
    if probe_price is not None:
        probes = (
            probe_input_tokens * probe_price.input + probe_output_tokens * probe_price.output
        ) / 1_000_000
    if verify_price is not None:
        probes += (
            verify_input_tokens * verify_price.input + verify_output_tokens * verify_price.output
        ) / 1_000_000
    with_retries = input_tokens + retry_input_tokens
    return AuditCost(
        expected_low_usd=_extraction(input_tokens, output_tokens_low),
        expected_high_usd=_extraction(with_retries, output_tokens_high) + probes,
        ceiling_usd=_extraction(with_retries, output_tokens_ceiling) + probes,
    )


class AuditEstimate(BaseModel):
    """Dry-run extraction cost and time estimate from harvested byte counts.

    Mirrors the general extractor's chunking: one LLM call per source at or
    under ``extraction.html_chunk_size`` characters, else one call per chunk,
    bounded by ``extraction.max_llm_calls_per_source``. Output is a per-call
    figure for the extraction model (``audit.estimate_output_tokens_per_extraction_call``
    and its per-model mapping), with the expected retries at
    ``extraction.retry_max_tokens``. Contradiction probes are density-dependent,
    so they enter the dollar range as zero to the audit cap
    (``audit.max_contradiction_probes``) plus the expected §6.6 probes the
    extraction pipeline makes.
    """

    entries: int = 0
    total_chars: int = 0
    #: First-attempt extraction calls; ``expected_retries`` rides on top.
    estimated_llm_calls: int = 0
    #: Source text only, at ~4 characters per token.
    estimated_tokens: int = 0
    #: Resolved models the estimate is priced against (``llm.extraction`` and
    #: ``llm.semantic_lint``, which carries both contradiction probes).
    extraction_model: str = ""
    probe_model: str = ""
    #: Extraction input: source text plus the per-call prompt overhead, first
    #: attempts only. ``retry_input_tokens`` is what the expected retries re-send.
    input_tokens: int = 0
    retry_input_tokens: int = 0
    #: Calls expected to be retried at ``extraction.retry_max_tokens`` (0 when
    #: the retry is disabled).
    expected_retries: float = 0.0
    #: Expected output tokens per extraction call and the config key it came from.
    output_tokens_per_call: int = 0
    output_tokens_per_call_source: str = ""
    #: Extraction output: the expected band and the ceiling (first attempts at
    #: ``extraction.max_tokens`` plus the expected retries at their budget).
    output_tokens_low: int = 0
    output_tokens_high: int = 0
    output_tokens_ceiling: int = 0
    #: Audit contradiction probes at the cap, plus the expected §6.6 probes the
    #: extraction pipeline makes. The token totals cover both.
    probe_cap: int = 0
    reconcile_probes: int = 0
    probe_input_tokens: int = 0
    probe_output_tokens: int = 0
    #: Second readings of flagged contradictions at their cap; 0
    #: when ``audit.verify_contradictions`` is off. Priced on ``llm.verification``.
    verify_cap: int = 0
    verify_model: str = ""
    verify_input_tokens: int = 0
    verify_output_tokens: int = 0
    #: Expected wall time in seconds: every call, retries, probes and second
    #: readings included, in turn.
    wall_seconds: float = 0.0
    #: List prices used, ``None`` for an unpriced model.
    extraction_price: TokenPrice | None = None
    probe_price: TokenPrice | None = None
    verify_price: TokenPrice | None = None
    #: ``None`` when a model that would be spent on has no price entry.
    cost: AuditCost | None = None
    #: ``provider:model (llm.<purpose>)`` for each unpriced model.
    unpriced: list[str] = Field(default_factory=list)


def estimate_extraction(char_counts: Sequence[int]) -> AuditEstimate:
    """Estimate extraction LLM calls, tokens, dollars, and wall time for texts of these sizes."""
    config = get_config()
    cfg = config.extraction
    audit_cfg = config.audit
    extraction_sel = config.llm.for_purpose("extraction")
    probe_sel = config.llm.for_purpose("semantic_lint")
    verify_sel = config.llm.for_purpose("verification")

    per_call, key = lookup_by_model(
        audit_cfg.estimate_output_tokens_per_extraction_call_by_model,
        extraction_sel.provider,
        extraction_sel.model,
    )
    if per_call is None:
        per_call = audit_cfg.estimate_output_tokens_per_extraction_call
        per_call_source = "audit.estimate_output_tokens_per_extraction_call"
    else:
        per_call_source = f"audit.estimate_output_tokens_per_extraction_call_by_model[{key}]"
    # A call's output can never exceed its own budget.
    per_call = min(per_call, cfg.max_tokens)

    calls = 0
    entries = 0
    total = 0
    for n in char_counts:
        if n <= 0:
            continue
        entries += 1
        total += n
        if n <= cfg.html_chunk_size:
            calls += 1
        else:
            calls += min(math.ceil(n / cfg.html_chunk_size), cfg.max_llm_calls_per_source)
    # ~4 chars per token is the standard rough conversion; the estimate is a
    # magnitude signal for the confirm gate, not a billing quote.
    source_tokens_total = total // 4
    input_tokens = source_tokens_total + calls * cfg.estimate_prompt_overhead_tokens

    retry_enabled = cfg.retry_max_tokens > cfg.max_tokens
    retries = calls * audit_cfg.estimate_extraction_retry_rate if retry_enabled else 0.0
    # A retry re-sends the same call: the average call's input again.
    retry_input = round(retries * input_tokens / calls) if calls else 0

    spread = audit_cfg.estimate_output_spread
    output_low = round(calls * per_call * (1 - spread))
    output_high = round((calls + retries) * per_call * (1 + spread))
    # The ceiling counts every expected retry at its full budget, rounded up.
    output_ceiling = calls * cfg.max_tokens + math.ceil(retries) * cfg.retry_max_tokens

    probe_cap = audit_cfg.max_contradiction_probes if entries else 0
    reconcile_probes = round(calls * audit_cfg.estimate_reconcile_probes_per_extraction_call)
    probes = probe_cap + reconcile_probes
    probe_input = probes * audit_cfg.estimate_probe_input_tokens
    probe_output = probes * audit_cfg.estimate_probe_output_tokens
    # A second reading follows only an audit-probe flag, so the audit
    # probe cap bounds the verification cap too.
    verify_cap = (
        min(audit_cfg.max_contradiction_verifications, probe_cap)
        if audit_cfg.verify_contradictions
        else 0
    )
    verify_input = verify_cap * audit_cfg.estimate_verify_input_tokens
    verify_output = verify_cap * audit_cfg.estimate_verify_output_tokens

    wall_seconds = (
        (calls + retries) * audit_cfg.estimate_seconds_per_extraction_call
        + probes * audit_cfg.estimate_seconds_per_probe
        + verify_cap * audit_cfg.estimate_seconds_per_verification
    )

    extraction_price = config.llm.price_for(extraction_sel)
    probe_price = config.llm.price_for(probe_sel)
    verify_price = config.llm.price_for(verify_sel)
    unpriced: list[str] = []
    if extraction_price is None and calls:
        unpriced.append(f"{extraction_sel.provider}:{extraction_sel.model} (llm.extraction)")
    if probe_price is None and probes and probe_sel != extraction_sel:
        unpriced.append(f"{probe_sel.provider}:{probe_sel.model} (llm.semantic_lint)")
    if verify_price is None and verify_cap and verify_sel not in (extraction_sel, probe_sel):
        unpriced.append(f"{verify_sel.provider}:{verify_sel.model} (llm.verification)")

    return AuditEstimate(
        entries=entries,
        total_chars=total,
        estimated_llm_calls=calls,
        estimated_tokens=source_tokens_total,
        extraction_model=extraction_sel.model,
        probe_model=probe_sel.model,
        input_tokens=input_tokens,
        retry_input_tokens=retry_input,
        expected_retries=retries,
        output_tokens_per_call=per_call,
        output_tokens_per_call_source=per_call_source,
        output_tokens_low=output_low,
        output_tokens_high=output_high,
        output_tokens_ceiling=output_ceiling,
        probe_cap=probe_cap,
        reconcile_probes=reconcile_probes,
        probe_input_tokens=probe_input,
        probe_output_tokens=probe_output,
        verify_cap=verify_cap,
        verify_model=verify_sel.model,
        verify_input_tokens=verify_input,
        verify_output_tokens=verify_output,
        wall_seconds=wall_seconds,
        extraction_price=extraction_price,
        probe_price=probe_price,
        verify_price=verify_price,
        cost=price_range(
            input_tokens=input_tokens,
            retry_input_tokens=retry_input,
            output_tokens_low=output_low,
            output_tokens_high=output_high,
            output_tokens_ceiling=output_ceiling,
            probe_input_tokens=probe_input,
            probe_output_tokens=probe_output,
            extraction_price=extraction_price,
            probe_price=probe_price,
            verify_input_tokens=verify_input,
            verify_output_tokens=verify_output,
            verify_price=verify_price,
        ),
        unpriced=unpriced,
    )


def _per_mtok(price: TokenPrice) -> str:
    return f"${price.input:g}/${price.output:g} per MTok"


def _duration(seconds: float) -> str:
    """``45 s``, ``12 min``, or ``1.6 h``."""
    if seconds < 120:
        return f"{max(1, round(seconds))} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{round(minutes)} min"
    return f"{minutes / 60:.1f} h"


def time_summary(estimate: AuditEstimate) -> str | None:
    """``about 1.6 h, ~60 s per file``, or ``None`` when there is nothing to run."""
    if not estimate.entries or estimate.wall_seconds <= 0:
        return None
    per_file = _duration(estimate.wall_seconds / estimate.entries)
    return f"about {_duration(estimate.wall_seconds)}, ~{per_file} per file"


def cost_summary(estimate: AuditEstimate) -> str | None:
    """``≈ $5.63–10.33 expected, $19 ceiling``, or ``None`` when unpriced."""
    cost = estimate.cost
    if cost is None:
        return None
    low, high = format_usd(cost.expected_low_usd), format_usd(cost.expected_high_usd)
    if cost.expected_low_usd < 10 <= cost.expected_high_usd:
        # One precision across the range: ``$5.63–10.33``, not ``$5.63–10``.
        high = f"${cost.expected_high_usd:,.2f}"
    expected = low if low == high else f"{low}–{high.removeprefix('$')}"
    return f"≈ {expected} expected, {format_usd(cost.ceiling_usd)} ceiling"


def render_cost(estimate: AuditEstimate) -> str:
    """The dollar line under the estimate: a list-price range, or why there is none.

    Shared by the ``--estimate`` output and the confirmation prompt, so the
    user sees the same figures before saying yes.
    """
    cost = estimate.cost
    if cost is None or estimate.extraction_price is None:
        models = "; ".join(estimate.unpriced)
        key = estimate.unpriced[0].split(" ", 1)[0] if estimate.unpriced else "<provider>:<model>"
        return (
            f"Cost: not priced. No list price is configured for {models}. "
            "Add one under llm.price_per_mtok in config.yaml, for example "
            f'`price_per_mtok: {{"{key}": {{input: 0.5, output: 1.5}}}}` '
            "(US$ per million tokens), to see a dollar range here."
        )
    expected = cost_summary(estimate)
    prices = f"{estimate.extraction_model} list price ({_per_mtok(estimate.extraction_price)})"
    if (
        (estimate.probe_cap or estimate.reconcile_probes)
        and estimate.probe_price is not None
        and (
            estimate.probe_model != estimate.extraction_model
            or estimate.probe_price != estimate.extraction_price
        )
    ):
        prices += f", probes at {estimate.probe_model} ({_per_mtok(estimate.probe_price)})"
    if (
        estimate.verify_cap
        and estimate.verify_price is not None
        and (
            estimate.verify_model != estimate.extraction_model
            or estimate.verify_price != estimate.extraction_price
        )
    ):
        prices += (
            f", second readings at {estimate.verify_model} ({_per_mtok(estimate.verify_price)})"
        )
    return (
        f"Cost: {expected} at {prices}. "
        "Actual cost depends on output length. This is an estimate from list price; "
        "prompt-cache and batch discounts are not applied."
    )


def render_estimate(estimate: AuditEstimate) -> str:
    """Human rendering of the §4 estimate (always printed pre-extraction)."""
    config = get_config()
    cfg = config.extraction
    audit_cfg = config.audit
    retries = round(estimate.expected_retries)
    retry_clause = (
        f" plus ~{retries} expected retr{'y' if retries == 1 else 'ies'} at "
        f"extraction.retry_max_tokens"
        if retries
        else ""
    )
    lines = [
        f"Estimate: {estimate.entries} entr{'y' if estimate.entries == 1 else 'ies'} to "
        f"extract → ~{estimate.estimated_llm_calls} extraction LLM call(s){retry_clause}, "
        f"~{estimate.estimated_tokens:,} tokens of source text. Contradiction probes are "
        f"similarity-gated and scale with near-duplicate density, not n²: up to "
        f"{estimate.probe_cap} in the audit (audit.max_contradiction_probes) and "
        f"~{estimate.reconcile_probes} expected while extraction reconciles new beliefs"
        + (
            f", and up to {estimate.verify_cap} second reading(s) of the audit's flags "
            f"(audit.max_contradiction_verifications). "
            if estimate.verify_cap
            else ". "
        )
        + "Already-captured content is skipped automatically.",
        render_cost(estimate),
    ]
    elapsed = time_summary(estimate)
    if elapsed is not None:
        lines.append(
            f"Time: {elapsed}. Calls run one at a time, at "
            f"~{audit_cfg.estimate_seconds_per_extraction_call:g} s per extraction call and "
            f"~{audit_cfg.estimate_seconds_per_probe:g} s per probe "
            f"(audit.estimate_seconds_per_extraction_call, audit.estimate_seconds_per_probe)"
            + (
                f", ~{audit_cfg.estimate_seconds_per_verification:g} s per second reading "
                f"(audit.estimate_seconds_per_verification)."
                if estimate.verify_cap
                else "."
            )
        )
    ceiling = f"{estimate.estimated_llm_calls} call(s) × extraction.max_tokens {cfg.max_tokens:,}"
    ceiling_retries = math.ceil(estimate.expected_retries)
    if ceiling_retries:
        ceiling += (
            f", plus {ceiling_retries} × extraction.retry_max_tokens {cfg.retry_max_tokens:,}"
        )
    spread = round(audit_cfg.estimate_output_spread * 100)
    lines.append(
        f"  Tokens: ~{estimate.input_tokens + estimate.retry_input_tokens:,} extraction "
        f"input (source plus the per-call prompt, retries included); "
        f"~{estimate.output_tokens_low:,} to ~{estimate.output_tokens_high:,} output "
        f"expected (~{estimate.output_tokens_per_call:,} per call for "
        f"{estimate.extraction_model}, {estimate.output_tokens_per_call_source}, "
        f"±{spread}%), at most {estimate.output_tokens_ceiling:,} ({ceiling}); "
        f"up to {estimate.probe_cap + estimate.reconcile_probes} probe(s) at "
        f"~{audit_cfg.estimate_probe_input_tokens} in / "
        f"{audit_cfg.estimate_probe_output_tokens} out tokens each"
        + (
            f"; up to {estimate.verify_cap} second reading(s) of flagged contradictions at "
            f"~{audit_cfg.estimate_verify_input_tokens} in / "
            f"{audit_cfg.estimate_verify_output_tokens} out tokens each."
            if estimate.verify_cap
            else "."
        )
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The report model (SDK-internal, like SuggestReport / QualityReport)
# ---------------------------------------------------------------------------


class AuditBucket(BaseModel):
    """One card class in the census: the full count plus a few exemplars."""

    kind: CardKind
    count: int
    # Top ``audit.exemplars_per_class`` cards by leverage, briefs attached
    # so every exemplar carries its claim text.
    exemplars: list[CurationCard] = Field(default_factory=list)
    # for the CONTESTED class, how many cards each contested basis
    # fired on (a card firing two bases counts under both, so these need not
    # sum to ``count``). Empty for every other kind. The class composes three
    # incommensurable signals, so reporting one unattributed total would hide
    # which instrument produced the census — the same auditability
    # built into the badge itself.
    bases: dict[str, int] = Field(default_factory=dict)


class AuditReport(BaseModel):
    """The census output of ``run_memory_audit``."""

    generated_at: datetime = Field(default_factory=_utcnow)
    store: str = "default"
    # Harvest header. ``files_audited`` is None on a re-audit (no PATH).
    files_audited: int | None = None
    transcripts_audited: int = 0
    harvested_new: int = 0
    harvested_unchanged: int = 0
    # Per-file extraction outcome (§6). ``extracted_snapshots`` counts the
    # snapshots that wrote their claims, ``extraction_failures`` the ones that
    # produced nothing (an LLM call failed or returned nothing usable; the
    # snapshot is left PENDING so a re-run retries it). A file in
    # ``extraction_partial_files`` extracted, but a reply was cut at the output
    # budget and the claims after the cut were lost. Files are named by their
    # URI tail (filename or session id).
    extracted_snapshots: int = 0
    extraction_failures: int = 0
    extraction_failed_files: list[str] = Field(default_factory=list)
    extraction_partial_files: list[str] = Field(default_factory=list)
    # Store-level counts (the get_quality_report header).
    beliefs: int = 0
    subjects: int = 0
    snapshots_failed: int = 0
    # The census: complete per-kind counts + leverage-ranked exemplars.
    buckets: list[AuditBucket] = Field(default_factory=list)
    # ``--judge``: duplicate pairs carry LLM_JUDGE verdicts and the
    # label upgrades to "verified duplicates".
    judged: bool = False
    # §6/§7 disclosure flags: the contradiction probe did not run (no API key,
    # or the circuit breaker is open) — never a silent clean bill.
    semantic_skipped: bool = False
    semantic_skip_reason: str | None = None
    # Per-probe failures during the finding scan (refusals, transient errors):
    # each skips one candidate pair without tripping the breaker, so counts can
    # read low — disclosed, never silent (§6).
    semantic_probe_failures: int = 0
    # contradiction-probe census. ``scope`` is the
    # candidate-pair scope the probe ran under ("harvested" = at least one side
    # of every pair traces to this harvest's entries; "store" = store-wide);
    # None when the probe didn't run. When ``probes_run`` <
    # ``candidate_pairs``, the ``audit.max_contradiction_probes`` cap bound and
    # the report discloses "probed X of Y candidate pairs" (§6) — the
    # contradiction count is a lower bound, not a census. Under harvested
    # scope, ``intra_scope_pairs`` is the both-sides-in-scope subset of the
    # candidates — probed first, and named in the capped
    # disclosure's tier split ("N intra-harvest, M cross-store").
    contradiction_probe_scope: Literal["harvested", "store"] | None = None
    contradiction_candidate_pairs: int = 0
    contradiction_intra_scope_pairs: int = 0
    contradiction_probes_run: int = 0
    # pairs not probed because the probe-verdict ledger already holds
    # a NO for them under the current prompt. Outside ``candidate_pairs``, so
    # they never bind the cap; disclosed beside the probe counts.
    contradiction_previously_cleared: int = 0
    # the probe's flags and their second reading. ``flagged`` pairs
    # got a YES from the probe; under ``contradiction_verified`` only the
    # ``confirmed`` ones became findings, and ``unverified`` flags (past
    # ``audit.max_contradiction_verifications``, or a failed reading) were
    # dropped uncounted. ``same_source`` counts the reported pairs whose two
    # claims came from one note, which is what the headline's "across files"
    # split subtracts.
    contradiction_verified: bool = False
    contradiction_verify_model: str | None = None
    contradiction_flagged: int = 0
    contradiction_confirmed: int = 0
    contradiction_unverified: int = 0
    contradiction_same_source: int = 0
    # the probe's reported pairs grouped into disagreements (pairs
    # connected through a shared claim count once). None when the probe did not
    # run, and the headline then counts contradiction cards as before.
    contradiction_disagreements: int | None = None
    contradiction_disagreements_within_one_note: int = 0
    contradiction_grouped_pairs: int = 0
    # what the agent is already shown. ``disclosure_open_records``
    # counts the open census records (each one cross-source disagreement the
    # session digest flags); ``contested_census_claims`` the contested claims
    # sitting in them, which the headline counts through their record rather
    # than one by one; ``disclosure_not_yet`` the confirmed cross-source
    # disagreements this audit found that no record covers yet. The audit
    # reports these and opens no record.
    disclosure_open_records: int = 0
    contested_census_claims: int = 0
    disclosure_not_yet: int = 0
    # duplicate-scan census — the DUPLICATE class analogue of the
    # contradiction fields above. ``duplicate_scope`` is "harvested" when the
    # headline / exemplars were filtered to pairs touching this harvest,
    # "store" when store-wide (re-audit / ``--scope store``).
    # ``duplicate_candidate_pairs_total`` is the store-wide candidate count M
    # (before scoping); when it exceeds the harvest-scoped headline count, the
    # report discloses the store-wide tail so the scan's full reach — e.g.
    # pre-existing store pollution — is never hidden behind the scoped count.
    duplicate_scope: Literal["harvested", "store"] | None = None
    duplicate_candidate_pairs_total: int = 0
    # The §4 estimate the run was gated on, when a harvest happened.
    estimate: AuditEstimate | None = None
    # ``--judge`` verdict split over ALL duplicate cards (not just exemplars):
    # verdict value → count. Empty unless ``judged``.
    duplicate_verdicts: dict[str, int] = Field(default_factory=dict)
    # Set by the CLI when the projection cycle re-rendered MEMORY.md
    # at the end of the run.
    projection_rendered: bool = False
    # Measured LLM usage for this run, from the providers' own ``usage`` fields
    # and priced at list price. Recorded on the CONSOLIDATION_RUN event too.
    llm_usage: LLMUsage | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def complete(self) -> bool:
        """False when the audit did less than asked.

        Two causes, each disclosed on its own: the semantic phase was skipped
        (the contradiction check did not run), or a harvested file produced no
        beliefs because its extraction failed (the census never saw what it
        says). The structural findings still stand. The CLI turns this into its
        exit status and headline, and ``--format json`` carries it next to
        ``semantic_skip_reason`` and ``extraction_failed_files``.
        """
        return not self.semantic_skipped and not self.extraction_failures

    @computed_field  # type: ignore[prop-decorator]
    @property
    def extracted_fully(self) -> int:
        """Snapshots whose every reply parsed whole."""
        return max(0, self.extracted_snapshots - len(self.extraction_partial_files))

    def count(self, kind: CardKind) -> int:
        """The census count for one card class (0 when absent)."""
        for bucket in self.buckets:
            if bucket.kind is kind:
                return bucket.count
        return 0

    def bucket(self, kind: CardKind) -> AuditBucket | None:
        """The bucket for one card class, or None when the class is empty."""
        for b in self.buckets:
            if b.kind is kind:
                return b
        return None

    def contested_split(self) -> tuple[int, int]:
        """(open conflict records, observer-contested beliefs).

        the first stays in the "potential contradictions" headline
        — an open INCONSISTENCY *is* the store contradicting itself. Since
        it counts records (one ``INCONSISTENCY`` card each), not the
        claims in them. The second is every ``CONTESTED`` card: a belief the
        badge fired on for a lens spread or a declared DISPUTES, which is not a
        contradiction and must not inflate the headline the wedge's trust claim
        rests on. A contested belief that also sits in a conflict is counted
        once in each, because each is a different question.
        """
        return self.count(CardKind.INCONSISTENCY), self.count(CardKind.CONTESTED)


# ---------------------------------------------------------------------------
# Assembly (pure given cards + quality) — unit-testable without a store
# ---------------------------------------------------------------------------

# Stable bucket order: headline classes first, then the secondary kinds in the
# §5 "Also:" order.
_BUCKET_ORDER: tuple[CardKind, ...] = (
    CardKind.CONTRADICTION,
    CardKind.INCONSISTENCY,
    CardKind.CONTESTED,
    CardKind.DUPLICATE_PAIR,
    CardKind.STALE,
    CardKind.RECENCY_DECAY,
    CardKind.CONFIDENCE_DECAY,
    CardKind.UNCITED_URL,
    CardKind.NO_SUBJECT,
    CardKind.GATED_SUBJECTS,
    CardKind.RETRACTION_CASCADE,
    CardKind.BROKEN_PROVENANCE,
    CardKind.FAILED_SNAPSHOTS,
)


def build_buckets(cards: Sequence[CurationCard], exemplars_per_class: int) -> list[AuditBucket]:
    """Group scored cards by kind: full counts, top-N exemplars by leverage."""
    by_kind: dict[CardKind, list[CurationCard]] = {}
    for card in cards:
        by_kind.setdefault(card.kind, []).append(card)

    buckets: list[AuditBucket] = []
    kinds = [k for k in _BUCKET_ORDER if k in by_kind]
    kinds += [k for k in by_kind if k not in _BUCKET_ORDER]  # future kinds never vanish
    for kind in kinds:
        group = sorted(by_kind[kind], key=lambda c: (-c.leverage, c.key))
        bases: dict[str, int] = {}
        if kind is CardKind.CONTESTED:
            for card in group:
                for basis in card.contested_bases or ():
                    bases[basis] = bases.get(basis, 0) + 1
        # A batch card is one card over many beliefs; the census
        # counts the beliefs, so folding orphans into it never shrinks the count.
        count = (
            sum(len(c.particle_ids) for c in group)
            if kind is CardKind.GATED_SUBJECTS
            else len(group)
        )
        buckets.append(
            AuditBucket(
                kind=kind,
                count=count,
                exemplars=group[:exemplars_per_class],
                bases=bases,
            )
        )
    return buckets


# ---------------------------------------------------------------------------
# Progress events (rendered by the CLI; the operation itself never prints)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditProgress(ProgressEvent):
    """One progress event from :func:`run_memory_audit`.

    The extraction phase can take many minutes (one LLM call per pending
    snapshot, plus subject resolution); without feedback the activation-moment
    audit is indistinguishable from a hang. The same goes for the
    contradiction probe on a populated store, so it emits per-pair ``probe``
    events (``done``/``total`` over the planned probe set
    — proposed). The Engine emits these events and the Surface renders them —
    the operation itself never prints (AGENTS.md § Code conventions).

    The shared :class:`~particles.core.progress.ProgressEvent` carries the
    four common fields; this subclass narrows ``phase`` and adds the
    per-unit extraction outcome.
    """

    phase: Literal["extract", "census", "probe"]
    #: Beliefs written for this unit (``extract`` phase, successful units only).
    particles: int | None = None
    failed: bool = False
    #: The unit's reply was cut at the output budget: its claims after the cut were lost.
    partial: bool = False


ProgressCallback = Callable[[AuditProgress], None]


def record_contradiction_census(
    report: AuditReport, control: ContradictionProbeControl | None
) -> None:
    """Copy the flag census from a probe control onto ``report``.

    Shared by the interactive audit and the nightly re-audit, which
    build their own controls but report the same census. ``None`` (the probe
    did not run) leaves the report's empty-run defaults.
    """
    if control is None:
        return
    report.contradiction_previously_cleared = control.previously_cleared
    report.contradiction_verified = control.verify
    report.contradiction_verify_model = (
        get_config().llm.for_purpose("verification").model if control.verify else None
    )
    report.contradiction_flagged = control.flagged
    report.contradiction_confirmed = control.confirmed
    report.contradiction_unverified = control.unverified
    report.contradiction_same_source = control.same_source_findings
    grouped = control.disagreements()
    report.contradiction_disagreements = grouped.groups
    report.contradiction_disagreements_within_one_note = grouped.within_one_note
    report.contradiction_grouped_pairs = grouped.pairs
    # The confirmed pairs a nightly cycle would open: cross-source ones,
    # grouped by shared claim. Covered pairs were never probed.
    report.disclosure_not_yet = len(
        group_pairs([p for p in control.confirmed_pairs if not p.same_source])
    )


async def record_disclosure_state(
    session: AsyncSession, report: AuditReport, cards: list[CurationCard]
) -> None:
    """Record what the agent is already shown about contradictions (§7).

    Reads the open census records and counts the contested claims in them
    among ``cards``. Read-only: the audit never opens a record (§6).
    """
    members: set[str] = set()
    for record in await get_census_records(session):
        if record.status is Status.INCONSISTENCY and (sides := census_sides(record)) is not None:
            report.disclosure_open_records += 1
            members |= set(sides.members)
    report.contested_census_claims = sum(
        1
        for c in cards
        if c.kind is CardKind.CONTESTED
        and "inconsistency" in (c.contested_bases or ())
        and c.particle_ids
        and c.particle_ids[0] in members
    )


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


async def run_memory_audit(
    session: AsyncSession,
    *,
    store: str = "default",
    files_audited: int | None = None,
    transcripts_audited: int = 0,
    harvested_new: int = 0,
    harvested_unchanged: int = 0,
    harvested_entry_ids: Sequence[str] | None = None,
    semantic: bool = True,
    judge: bool = False,
    semantic_skip_reason: str | None = None,
    estimate: AuditEstimate | None = None,
    agent_id: str = "audit",
    on_progress: ProgressCallback | None = None,
    contradiction_scope: Literal["harvested", "store"] = "store",
) -> AuditReport:
    """Extract the harvested entries, census the findings, assemble the report.

    Composes existing pieces only: the standard extract pipeline
    over PENDING snapshots of ``harvested_entry_ids`` (COMPLETE snapshots skip
    — idempotence against the harvest is structural), then the
    ``collect_cards(semantic=...)`` **uncapped and snooze-unfiltered**,
    leverage scoring to rank exemplars *within* each class, brief attachment,
    and ``get_quality_report`` for the header.

    ``semantic`` gates the contradiction probe; ``judge`` runs the
    duplicate finder in ``LLM_JUDGE`` mode (default ``REPORT`` — unjudged
    similarity candidates). Commits after each extracted snapshot (a later
    failure must not roll back completed extraction work); otherwise the
    caller owns the transaction.

    ``contradiction_scope`` bounds the probe's candidate
    pairs: ``"harvested"`` keeps only pairs where at least one side's particle
    traces to ``harvested_entry_ids`` (the CLI's default on a harvest run);
    ``"store"`` is the store-wide set (the re-audit default).
    ``"harvested"`` with no harvested entries probes nothing — the caller
    should not ask for it. The ``audit.max_contradiction_probes`` cap applies
    in both scopes; the census fields on the report disclose what was probed.
    Under ``"harvested"``, both the contradiction probe and the ``--judge``
    duplicate pass consume their candidates in two tiers:
    intra-harvest pairs (both sides in scope) first, mixed pairs second,
    highest similarity within each — so a binding cap never starves the
    harvest's own pairs behind coincidental cross-store neighbours.

    Each run also records one ``CONSOLIDATION_RUN`` operator event
    (``actor: audit``) so the interactive audit contributes to —
    and its census deltas are readable from — the same delta chain the
    scheduled consolidation cycle keys off.
    """
    # Count every completion this run pays for, per purpose and model, for the
    # report's usage line and the run record.
    with track_usage() as usage:
        started_at = _utcnow()
        # 1. Extract — PENDING snapshots scoped to this harvest's entries.
        extracted = 0
        failures = 0
        failed_labels: list[str] = []
        partial_labels: list[str] = []
        if harvested_entry_ids:
            # Deferred import: the pipeline pulls the extractor registry / LLM
            # stack; load it only when there is something to extract (AGENTS.md
            # deferred-import case 2).
            from particles.corpus.store import get_entry, list_snapshots_for_entry
            from particles.operations.extract import (
                SnapshotOutcome,
                collapse_superseded_pending,
                extract_snapshot,
            )

            # a re-run audit over memory files edited since the last
            # extraction pays for the newest generation of each, not every one.
            collapse = await collapse_superseded_pending(
                session, entry_ids=list(dict.fromkeys(harvested_entry_ids))
            )
            collapse_line = collapse.summary()
            if collapse_line:
                log.info("%s", collapse_line)

            pending: list[tuple[str, str, str]] = []  # (entry_id, snapshot_id, label)
            for entry_id in dict.fromkeys(harvested_entry_ids):  # de-dupe, keep order
                label: str | None = None
                for snap in await list_snapshots_for_entry(session, entry_id):
                    if snap.extraction_status is not ExtractionStatus.PENDING:
                        continue
                    if label is None:
                        entry = await get_entry(session, entry_id)
                        # Human handle: the URI tail (filename / session id), never
                        # the opaque entry uuid.
                        uri = entry.uri_r if entry is not None else None
                        label = uri.rstrip("/").rsplit("/", 1)[-1] if uri else entry_id[:8]
                    pending.append((entry_id, snap.snapshot_id, label))

            for index, (entry_id, snapshot_id, label) in enumerate(pending, start=1):
                outcome = SnapshotOutcome()
                try:
                    written = await extract_snapshot(
                        session,
                        entry_id,
                        snapshot_id,
                        agent_id=agent_id,
                        skip_if_superseded=True,
                        outcome_out=outcome,
                    )
                    await session.commit()
                    # An empty write is not proof of an empty source: a call that
                    # produced nothing usable left the snapshot PENDING and wrote
                    # nothing, and a reply cut at the budget lost its tail. Both are
                    # disclosed per file (§6), never counted as a clean extraction.
                    match outcome.extracted:
                        case "none":
                            failures += 1
                            failed_labels.append(label)
                        case "partial":
                            extracted += 1
                            partial_labels.append(label)
                        case "full":
                            extracted += 1
                    if on_progress is not None:
                        on_progress(
                            AuditProgress(
                                phase="extract",
                                done=index,
                                total=len(pending),
                                label=label,
                                particles=len(written),
                                failed=outcome.extracted == "none",
                                partial=outcome.extracted == "partial",
                            )
                        )
                except Exception as exc:  # noqa: BLE001 — census the rest; disclose the failure
                    await session.rollback()
                    failures += 1
                    failed_labels.append(label)
                    log.warning(
                        "audit: extraction failed for %s/%s: %s",
                        entry_id[:8],
                        snapshot_id[:8],
                        exc,
                    )
                    if on_progress is not None:
                        on_progress(
                            AuditProgress(
                                phase="extract",
                                done=index,
                                total=len(pending),
                                label=label,
                                failed=True,
                            )
                        )

        # 2–3. Collect findings through the shared finder-normalization seam.
        duplicate_mode = SuggestMode.LLM_JUDGE if (judge and semantic) else SuggestMode.REPORT
        if on_progress is not None:
            census_label = (
                "contradiction probe + duplicate scan"
                if semantic and not llm_circuit_open()
                else "structural checks + duplicate scan"
            )
            on_progress(AuditProgress(phase="census", done=0, total=1, label=census_label))

        # Harvested-scope particle-id set: pairs where at least one side traces to
        # this harvest's entries. Shared by the contradiction probe and the
        # duplicate scan (computed once, independent of ``semantic`` since
        # the duplicate scan runs even when semantic is off). ``None`` ⇒ store-wide
        # (the re-audit / ``--scope store`` path).
        harvested_scope_ids: frozenset[str] | None = None
        if contradiction_scope == "harvested":
            harvested_scope_ids = frozenset(
                await get_particle_ids_for_entries(session, list(harvested_entry_ids or []))
            )

        # Bound the probe: the audit — unlike lint —
        # is cost-gated, so it always caps, optionally scopes to this harvest's
        # beliefs, and streams per-pair progress. The control carries the
        # candidate-pair census back out for the §6 "probed X of Y" disclosure.
        probe_control: ContradictionProbeControl | None = None
        if semantic:

            def _probe_progress(done: int, total: int) -> None:
                if on_progress is not None:
                    on_progress(
                        AuditProgress(
                            phase="probe", done=done, total=total, label="contradiction probe"
                        )
                    )

            def _verify_progress(done: int, total: int) -> None:
                if on_progress is not None:
                    on_progress(
                        AuditProgress(
                            phase="probe",
                            done=done,
                            total=total,
                            label="second reading of flagged pairs",
                        )
                    )

            audit_cfg = get_config().audit
            probe_control = ContradictionProbeControl(
                max_probes=audit_cfg.max_contradiction_probes,
                scope_particle_ids=harvested_scope_ids,
                on_progress=_probe_progress if on_progress is not None else None,
                # a flag counts in the headline only once a second,
                # context-rich reading confirms it.
                verify=audit_cfg.verify_contradictions,
                max_verifications=audit_cfg.max_contradiction_verifications,
                on_verify_progress=_verify_progress if on_progress is not None else None,
            )

        failures_before = llm_failure_count()
        trips_before = llm_trip_count()
        cards = await collect_cards(
            session,
            semantic=semantic,
            duplicate_mode=duplicate_mode,
            contradiction_probe=probe_control,
            duplicate_scope_ids=harvested_scope_ids,
        )
        probe_failures = llm_failure_count() - failures_before
        # read the composed-badge set off the full collection, before
        # the duplicate harvest-scoping below narrows it.
        contested_ids = contested_ids_from(cards)
        disclosure_cards = list(cards)

        # harvest-scope the duplicate headline/exemplars. The scan ran
        # store-wide (REPORT enumeration is pure cosine); keep the store-wide total
        # M for the tail disclosure, then drop DUPLICATE_PAIR cards that don't touch
        # this harvest so the headline / exemplars / verdict census describe the
        # harvest — never store-wide pollution mislabelled as a harvest finding.
        duplicate_total = sum(1 for c in cards if c.kind is CardKind.DUPLICATE_PAIR)
        if harvested_scope_ids is not None:
            cards = [
                c
                for c in cards
                if c.kind is not CardKind.DUPLICATE_PAIR
                or any(pid in harvested_scope_ids for pid in c.particle_ids)
            ]

        await score_cards(session, cards, contested_ids=contested_ids)

        # 4. Assemble: full per-kind counts, top exemplars with claim text.
        exemplars_per_class = get_config().audit.exemplars_per_class
        buckets = build_buckets(cards, exemplars_per_class)
        await _attach_particle_briefs(
            session, [card for bucket in buckets for card in bucket.exemplars]
        )
        quality: QualityReport = await get_quality_report(session)

        # The breaker half-opens after ``llm.unavailable_backoff_seconds``, so a
        # trip during a long census may already have expired here; the trip count
        # remembers it.
        llm_lost = llm_circuit_open() or llm_trip_count() > trips_before
        skipped = (not semantic) or llm_lost
        reason = semantic_skip_reason
        if skipped and reason is None:
            if llm_lost:
                cause = llm_unavailable_cause()
                reason = f"LLM unavailable ({cause})" if cause else "LLM unavailable"
            else:
                reason = "semantic checks disabled"

        verdict_counts: dict[str, int] = {}
        if judge and semantic:
            for card in cards:
                if card.kind is not CardKind.DUPLICATE_PAIR:
                    continue
                value = (
                    card.verdict.verdict.value if card.verdict else JudgeVerdictKind.UNSURE.value
                )
                verdict_counts[value] = verdict_counts.get(value, 0) + 1

        report = AuditReport(
            store=store,
            files_audited=files_audited,
            transcripts_audited=transcripts_audited,
            harvested_new=harvested_new,
            harvested_unchanged=harvested_unchanged,
            extracted_snapshots=extracted,
            extraction_failures=failures,
            extraction_failed_files=failed_labels,
            extraction_partial_files=partial_labels,
            beliefs=quality.active_particles,
            subjects=quality.total_subjects,
            snapshots_failed=quality.snapshots_failed,
            buckets=buckets,
            judged=judge and semantic,
            duplicate_verdicts=verdict_counts,
            semantic_skipped=skipped,
            semantic_skip_reason=reason if skipped else None,
            semantic_probe_failures=probe_failures,
            contradiction_probe_scope=contradiction_scope if probe_control is not None else None,
            contradiction_candidate_pairs=(
                probe_control.candidate_pairs if probe_control is not None else 0
            ),
            contradiction_intra_scope_pairs=(
                probe_control.intra_scope_pairs if probe_control is not None else 0
            ),
            contradiction_probes_run=probe_control.probes_run if probe_control is not None else 0,
            duplicate_scope="harvested" if harvested_scope_ids is not None else "store",
            duplicate_candidate_pairs_total=duplicate_total,
            estimate=estimate,
        )
    record_contradiction_census(report, probe_control)
    await record_disclosure_state(session, report, disclosure_cards)
    report.llm_usage = usage.snapshot()

    # the run-record fold. Deferred import — the
    # consolidation module pulls the reconcile/ingest stack this census path
    # otherwise never loads (AGENTS.md deferred-import case 2).
    from particles.operations.consolidation import record_audit_run

    await record_audit_run(session, report, started_at=started_at, actor=agent_id)
    return report


# ---------------------------------------------------------------------------
# The renderer (§6) — one renderer for terminal and --output
# ---------------------------------------------------------------------------

# Per-class next verbs (§5): every class ends with its door into an existing loop.
_NEXT_VERBS: dict[str, str] = {
    "contradictions": (
        "next: particles curate --kind inconsistency · particles curate --kind contradiction"
    ),
    "duplicates": "next: particles links suggest --judge · particles curate --kind duplicate_pair",
    "stale": "next: particles curate --kind stale",
    # a divergence- or stance-only card has no INCONSISTENCY for
    # `review` to resolve, so its door is the drill-down that shows the reading.
    "contested": ("next: particles query --contestedness · particles curate --kind contested"),
}

_STALE_KINDS = (CardKind.STALE, CardKind.RECENCY_DECAY, CardKind.CONFIDENCE_DECAY)


def _short(pid: str) -> str:
    return f"{pid[:8]}…"


def _exemplar_lines(card: CurationCard) -> list[str]:
    """Render one exemplar: claim text, partner claim for pairs, short id."""
    lines: list[str] = []
    briefs = {b.particle_id: b for b in card.particles}
    if card.particle_ids:
        first = card.particle_ids[0]
        content = briefs[first].content if first in briefs else "(claim text unavailable)"
        lines.append(f'  • "{content}"  [{_short(first)}]')
        for partner in card.particle_ids[1:]:
            partner_content = (
                briefs[partner].content if partner in briefs else "(claim text unavailable)"
            )
            lines.append(f'    ↔ "{partner_content}"  [{_short(partner)}]')
    elif card.corpus_url is not None:
        lines.append(f"  • {card.corpus_url}")
    if card.diagnostic:
        # One line, cut at a word with an ellipsis: a model-written description
        # can run to a paragraph, and a finding line is a pointer, not the text.
        lines.append(f"    — {one_line(card.diagnostic, 200)}")
    return lines


def _judged_duplicate_split(report: AuditReport) -> tuple[int, int, int]:
    """(paraphrase, distinct, unsure) counts from the report's verdict census."""
    counts = report.duplicate_verdicts
    paraphrase = counts.get(JudgeVerdictKind.PARAPHRASE.value, 0)
    distinct = counts.get(JudgeVerdictKind.DISTINCT.value, 0)
    unsure = report.count(CardKind.DUPLICATE_PAIR) - paraphrase - distinct
    return paraphrase, distinct, max(0, unsure)


def _incomplete_label(report: AuditReport) -> str:
    """``INCOMPLETE: contradiction check skipped, LLM unavailable`` for the header.

    The header names the cause *class* only; the lines below it carry the
    detail (``LLM unavailable (credit balance too low)``, the files that were
    not extracted).
    """
    causes: list[str] = []
    if report.extraction_failures:
        n = report.extraction_failures
        causes.append(f"{n} {_unit_noun(report, n)} not extracted")
    if report.semantic_skipped:
        reason = (report.semantic_skip_reason or "LLM unavailable").split(" (", 1)[0]
        causes.append(f"contradiction check skipped, {reason}")
    return "INCOMPLETE: " + "; ".join(causes)


#: How many file names an extraction disclosure line spells out before "and N more".
_NAMED_FILES = 5


def _unit_noun(report: AuditReport, n: int) -> str:
    """``file`` for a memory-file harvest, ``entry`` once transcripts ride along."""
    if report.transcripts_audited:
        return "entry" if n == 1 else "entries"
    return "file" if n == 1 else "files"


def _name_files(labels: Sequence[str]) -> str:
    named = ", ".join(labels[:_NAMED_FILES])
    rest = len(labels) - _NAMED_FILES
    return f"{named} and {rest} more" if rest > 0 else named


def _extraction_lines(report: AuditReport) -> list[str]:
    """The per-file extraction outcome, printed directly under the header (§6).

    Empty when this run extracted nothing (a re-audit, or a harvest whose
    every file was already extracted). Otherwise it always prints, including
    the all-clear, so a reader never has to infer that no file was lost.
    """
    attempted = report.extracted_snapshots + report.extraction_failures
    if not attempted:
        return []
    full = report.extracted_fully
    partial = len(report.extraction_partial_files)
    failed = report.extraction_failures
    if not partial and not failed:
        return [f"  Extraction: all {attempted} {_unit_noun(report, attempted)} extracted in full."]
    lines = [
        f"  Extraction: {full} of {attempted} {_unit_noun(report, attempted)} in full, "
        f"{partial} cut short at the output-token limit, {failed} not at all."
    ]
    if failed:
        lines.append(
            f"    not extracted, so absent from every count below: "
            f"{_name_files(report.extraction_failed_files)}"
        )
    if partial:
        lines.append(
            f"    cut short, so claims after the cut are missing: "
            f"{_name_files(report.extraction_partial_files)}"
        )
    return lines


def _count(n: int, singular: str, plural: str | None = None) -> str:
    """``1 belief pair`` / ``2 belief pairs``: a count with its noun in agreement."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def render_audit_report(report: AuditReport) -> str:
    """Render the census in the §5 approved shape — terminal and Markdown share it."""
    lines: list[str] = []

    # --- Header -----------------------------------------------------------
    # An incomplete run says so where the eye lands first, not only in a
    # disclosure line below the counts.
    incomplete = "" if report.complete else f" ({_incomplete_label(report)})"
    if report.files_audited is not None:
        sources = f"{report.files_audited} memory file{'s' if report.files_audited != 1 else ''}"
        if report.transcripts_audited:
            sources += (
                f" + {report.transcripts_audited} "
                f"transcript{'s' if report.transcripts_audited != 1 else ''}"
            )
        lines.append(
            f"Audited {sources}{incomplete} → {report.beliefs} beliefs "
            f"about {report.subjects} subjects."
        )
    else:
        lines.append(
            f"Re-audited store '{report.store}'{incomplete} → {report.beliefs} beliefs "
            f"about {report.subjects} subjects."
        )
    lines.extend(_extraction_lines(report))
    lines.append("")

    # --- Headline classes (always shown, zero or not; §5) ------------------
    contradiction_n = report.count(CardKind.CONTRADICTION)
    # The open-record term counts records, one INCONSISTENCY card each.
    # An open census record is one cross-source disagreement,
    # so it counts once, across files; the rest were recorded at
    # extract time.
    records_n, observer_contested_n = report.contested_split()
    census_n = min(report.disclosure_open_records, records_n)
    contested_n = records_n - census_n
    dup_n = report.count(CardKind.DUPLICATE_PAIR)
    expired_n = report.count(CardKind.STALE)
    aged_n = report.count(CardKind.RECENCY_DECAY) + report.count(CardKind.CONFIDENCE_DECAY)

    # the class counts disagreements, not claim pairs; pairs
    # connected through a shared claim count once. A contradiction card the
    # probe did not report (a recorded CONTRADICTS edge) counts as
    # its own disagreement. "Across files" is computed per group.
    if report.contradiction_disagreements is not None:
        unprobed = max(0, contradiction_n - report.contradiction_grouped_pairs)
        contradiction_n = report.contradiction_disagreements + unprobed
        within_file_n = report.contradiction_disagreements_within_one_note
    else:
        within_file_n = min(report.contradiction_same_source, contradiction_n)
    by_origin = [f"{contradiction_n - within_file_n + census_n} across files"]
    if within_file_n:
        by_origin.append(f"{within_file_n} within one file")
    by_origin.append(f"{contested_n} contested at extract time")
    headline: list[tuple[str, str]] = []
    headline.append(
        (
            _count(contradiction_n + contested_n + census_n, "potential contradiction"),
            f"({', '.join(by_origin)})",
        )
    )
    if report.judged:
        paraphrase, distinct, unsure = _judged_duplicate_split(report)
        headline.append(
            (
                _count(dup_n, "verified duplicate belief pair"),
                f"(LLM-judged: {paraphrase} paraphrase, {distinct} distinct, {unsure} unsure)",
            )
        )
    else:
        headline.append(
            (
                _count(dup_n, "likely-duplicate belief pair"),
                "(unjudged similarity candidates; --judge to verify)",
            )
        )
    headline.append(
        (
            _count(expired_n + aged_n, "probably-stale fact"),
            f"({aged_n} aged past their source's decay horizon, {expired_n} expired)",
        )
    )
    width = max(len(label) for label, _ in headline)
    for label, detail in headline:
        lines.append(f"  {label.ljust(width)}  {detail}")

    if report.semantic_skipped:
        reason = report.semantic_skip_reason or "LLM unavailable"
        lines.append(f"  contradiction check skipped: {reason}")
    elif report.semantic_probe_failures:
        # Only when the phase ran: the call that trips the breaker also counts
        # as a failure, and "skipped" already says everything it would.
        n = report.semantic_probe_failures
        if n == 1:
            what = "probe failed or was declined by the model and was skipped"
        else:
            what = "probes failed or were declined by the model and were skipped"
        lines.append(f"  {n} semantic {what} — contradiction and duplicate counts may read low")
    # the headline counts confirmed pairs, so say how many the first
    # pass flagged and what the second reading kept.
    if report.contradiction_verified and not report.semantic_skipped:
        model = report.contradiction_verify_model or "llm.verification"
        line = (
            f"  contradiction check: the first pass flagged {report.contradiction_flagged} "
            f"of {report.contradiction_probes_run} probed pairs; a second reading "
            f"({model}) confirmed {report.contradiction_confirmed}"
        )
        if report.contradiction_unverified:
            line += (
                f"; {report.contradiction_unverified} not read a second time "
                f"(audit.max_contradiction_verifications or a failed call) and not counted"
            )
        lines.append(line)
    if report.disclosure_open_records or report.disclosure_not_yet:
        total = report.disclosure_open_records + report.disclosure_not_yet
        noun = "disagreement is" if total == 1 else "disagreements are"
        lines.append(
            f"  agent disclosure: {report.disclosure_open_records} of {total} confirmed "
            f"cross-source {noun} an open inconsistency the session digest flags; "
            f"{report.disclosure_not_yet} not yet (the nightly consolidation opens them)"
        )
    if (
        report.contradiction_disagreements is not None
        and report.contradiction_grouped_pairs > report.contradiction_disagreements
    ):
        lines.append(
            f"  {_count(report.contradiction_grouped_pairs, 'contradiction claim pair')} "
            f"count as {_count(report.contradiction_disagreements, 'disagreement')}: "
            f"pairs that share a claim count once"
        )
    # Proposed disclosures: a scoped or capped probe is a lower
    # bound, never a silent partial census (§6).
    if report.contradiction_probe_scope == "harvested" and not report.semantic_skipped:
        lines.append(
            "  contradiction probe scoped to this harvest's beliefs (each probed "
            "pair touches at least one; intra-harvest pairs probed first) — "
            "pass --scope store to probe the whole store"
        )
    # the saving from the probe-verdict ledger, as a number.
    if report.contradiction_previously_cleared and not report.semantic_skipped:
        lines.append(
            f"  contradiction probe: {_count(report.contradiction_previously_cleared, 'pair')} "
            f"skipped as previously cleared (both claims unchanged since a probe answered "
            f"no; not counted against audit.max_contradiction_probes)"
        )
    if (
        not report.semantic_skipped
        and report.contradiction_probes_run < report.contradiction_candidate_pairs
    ):
        cap = get_config().audit.max_contradiction_probes
        # under harvested scope, name the tier split so the operator
        # can see whether the cap ever reached the cross-store tier.
        split = ""
        if report.contradiction_probe_scope == "harvested":
            intra = report.contradiction_intra_scope_pairs
            cross = report.contradiction_candidate_pairs - intra
            split = f"{intra} intra-harvest, {cross} cross-store; "
        lines.append(
            f"  contradiction probe capped: probed {report.contradiction_probes_run} of "
            f"{report.contradiction_candidate_pairs} candidate pairs "
            f"({split}audit.max_contradiction_probes = {cap}) — the contradiction count "
            f"may read low"
        )
    # a harvest-scoped duplicate headline discloses the store-wide
    # total so the scan's full reach (e.g. pre-existing store pollution) is
    # never hidden behind the harvest-scoped count.
    if (
        report.duplicate_scope == "harvested"
        and report.duplicate_candidate_pairs_total > report.count(CardKind.DUPLICATE_PAIR)
    ):
        harvested_dups = report.count(CardKind.DUPLICATE_PAIR)
        store_dups = report.duplicate_candidate_pairs_total
        lines.append(
            f"  duplicate scan is store-wide; {store_dups} candidate pairs total, "
            f"{harvested_dups} involve this harvest — pass --scope store to surface "
            f"all {store_dups}"
        )

    # --- Secondary line (only nonzero kinds; §5) ----------------------------
    also: list[str] = []
    uncited_n = report.count(CardKind.UNCITED_URL)
    no_subject_n = report.count(CardKind.NO_SUBJECT)
    cascade_n = report.count(CardKind.RETRACTION_CASCADE)
    provenance_n = report.count(CardKind.BROKEN_PROVENANCE)
    if uncited_n:
        also.append(f"{uncited_n} cited sources never captured")
    gated_n = report.count(CardKind.GATED_SUBJECTS)
    if no_subject_n or gated_n:
        line = f"{no_subject_n + gated_n} beliefs have no resolvable subject"
        if gated_n:
            line += f" ({gated_n} recoverable by `particles subjects relink-gated`)"
        also.append(line)
    if cascade_n:
        also.append(f"{cascade_n} beliefs depend on a retracted belief")
    if provenance_n:
        also.append(f"{provenance_n} beliefs cite a missing corpus entry")
    if report.snapshots_failed:
        also.append(f"{report.snapshots_failed} snapshots failed extraction")
    if observer_contested_n:
        contested_bucket = report.bucket(CardKind.CONTESTED)
        counted = contested_bucket.bases if contested_bucket is not None else {}
        by_basis = ", ".join(
            f"{counted[b]} {b}" for b in ("stance", "divergence") if counted.get(b)
        )
        also.append(f"{observer_contested_n} beliefs contested by observer signal ({by_basis})")
    if also:
        lines.append("")
        lines.append("  Also: " + " · ".join(also))
        if uncited_n:
            lines.append("  next: particles deposit <url> · particles curate --kind uncited_url")

    # --- Exemplars per headline class (§5) ----------------------------------
    def _class_block(
        title: str,
        kinds: Sequence[CardKind],
        verbs: str,
        where: Callable[[CurationCard], bool] | None = None,
    ) -> None:
        exemplars: list[CurationCard] = []
        for kind in kinds:
            bucket = report.bucket(kind)
            if bucket is not None:
                exemplars.extend(c for c in bucket.exemplars if where is None or where(c))
        if not exemplars:
            return
        exemplars.sort(key=lambda c: (-c.leverage, c.key))
        limit = get_config().audit.exemplars_per_class
        lines.append("")
        lines.append(title)
        for card in exemplars[:limit]:
            lines.extend(_exemplar_lines(card))
        lines.append(f"  {verbs}")

    dup_title = "Verified duplicates" if report.judged else "Likely-duplicate belief pairs"
    _class_block(
        "Potential contradictions",
        (CardKind.CONTRADICTION, CardKind.INCONSISTENCY),
        _NEXT_VERBS["contradictions"],
    )
    if observer_contested_n:
        # every CONTESTED card now fires an observer-signal basis.
        _class_block(
            "Contested by observer signal", (CardKind.CONTESTED,), _NEXT_VERBS["contested"]
        )
    _class_block(dup_title, (CardKind.DUPLICATE_PAIR,), _NEXT_VERBS["duplicates"])
    _class_block("Probably-stale facts", _STALE_KINDS, _NEXT_VERBS["stale"])

    # --- Disclosures + footer (§5/§6) ---------------------------------------
    if report.extraction_failures:
        n = report.extraction_failures
        lines.append("")
        lines.append(
            f"  {n} {_unit_noun(report, n)} produced no beliefs because extraction failed. "
            f"They are still pending: re-run this audit to retry them."
        )
    lines.append("")
    if report.beliefs:
        lines.append(
            "note: confidence on this content is self-reported and capped, "
            "not benchmark-calibrated."
        )
    lines.append("Run 'particles curate' to work these down a few at a time.")
    if report.projection_rendered:
        lines.append(
            "MEMORY.md was re-projected from the audited store — its memory-index "
            "region now reflects these beliefs."
        )
    if report.llm_usage is not None:
        lines.append(render_usage_line(report.llm_usage))
    return "\n".join(lines) + "\n"
