# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``memory consolidate --dry-run``: each pass's gather, priced, nothing sent.

:func:`plan_consolidation` runs the gather half of every pass in the cycle's
order and stops before the first LLM call: the snapshots pass 1
would extract, the candidate pairs passes 2 and 2b would probe, the triggers
pass 2c would examine, the candidate pairs the census would probe, and the
matcher groups the utility pass would judge. Each LLM-priced pass is priced at
list price over estimated tokens (:mod:`particles.operations.spend_estimate`).
It writes nothing, takes no cycle lock and makes no LLM call; the shape copies
``particles structure --dry-run``, whose dry run doubles as the coverage probe.

Where a dry gather cannot see what the run would see, the plan says so rather
than guess: the run computes the delta scope after pass 1, so the plan's scope
omits the beliefs pass 1 would mint, and pass 1's listing is taken before the
run's stale-claim reset and superseded-generation collapse.

``consolidation.budget_usd`` is simulated with the run's own decision
(:func:`~particles.core.spend_budget.decide_spend`): spend so far is the sum of
the estimates of the passes before, so the plan names the passes the budget
would skip at these estimates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.consolidation_cadence import decide_census
from particles.core.schema import SuggestMode
from particles.core.spend_budget import decide_spend
from particles.llm.usage import format_usd
from particles.operations.consolidation import (
    PASS_ORDER,
    _abstraction_cap_estimate,
    _delta_scope_ids,
    _event_started_at,
    _extract_selection,
    _plan_utility,
    _reanchor_estimate,
    _semantic_availability,
    _snapshot_sizes,
    _utcnow,
    latest_census_event,
    latest_run_event,
)
from particles.operations.contradiction_disclosure import covered_pair_set
from particles.operations.curation.collect import collect_cards
from particles.operations.lint import ContradictionProbeControl
from particles.operations.reanchor import prior_cursor, run_reanchor
from particles.operations.reconcile import count_supersession_candidates, count_update_candidates
from particles.operations.spend_estimate import (
    PassEstimate,
    context_call_estimate,
    extraction_estimate,
    probe_estimate,
)
from particles.operations.utility_mining import estimate_judge_calls

#: The line a metered run prints afterwards, named so the two can be compared.
MEASURED_LINE = "LLM usage: … ≈ $X at list price."


class PlannedPass(BaseModel):
    """One pass of the cycle as the dry run found it."""

    name: str
    position: int
    action: Literal["run", "skip"]
    #: What the gather found, or why the pass would be skipped.
    detail: str
    #: True for a pass that pays for LLM calls when it runs.
    llm_priced: bool = False
    calls: int = 0
    #: List price over estimated tokens in US$; ``None`` when a model is unpriced.
    estimate_usd: float | None = 0.0
    #: Config keys the per-call token figures came from.
    basis: str = ""
    #: Set when ``consolidation.budget_usd`` would skip the pass at these estimates.
    budget_skip: str | None = None


class ConsolidationPlan(BaseModel):
    """The output of :func:`plan_consolidation`."""

    store: str = "default"
    scope: Literal["delta", "store"] = "delta"
    effective_scope: Literal["delta", "store"] = "delta"
    watermark: datetime | None = None
    scope_particle_count: int | None = None
    #: Why the LLM passes would be skipped (``--structural-only``, no key, …).
    semantic_degraded_reason: str | None = None
    budget_usd: float | None = None
    passes: list[PlannedPass] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_calls(self) -> int:
        """LLM calls across the passes that would run."""
        return sum(p.calls for p in self._spending())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_usd(self) -> float | None:
        """US$ across the passes that would run; ``None`` when any is unpriced."""
        total = 0.0
        for p in self._spending():
            if p.estimate_usd is None:
                return None
            total += p.estimate_usd
        return total

    def _spending(self) -> list[PlannedPass]:
        return [p for p in self.passes if p.action == "run" and p.budget_skip is None]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _planned(
    name: str,
    detail: str,
    estimate: PassEstimate | None = None,
    *,
    action: Literal["run", "skip"] = "run",
) -> PlannedPass:
    return PlannedPass(
        name=name,
        position=PASS_ORDER.index(name) + 1,
        action=action,
        detail=detail,
        llm_priced=estimate is not None,
        calls=estimate.calls if estimate is not None else 0,
        estimate_usd=estimate.usd if estimate is not None else 0.0,
        basis=estimate.basis if estimate is not None else "",
    )


def _skipped(name: str, reason: str) -> PlannedPass:
    return _planned(name, reason, action="skip")


async def plan_consolidation(  # noqa: PLR0915 — one linear walk of the pass list
    session: AsyncSession,
    *,
    store: str = "default",
    scope: Literal["delta", "store"] = "delta",
    structural_only: bool = False,
    actor: str = "memory-consolidate",
    projection_skip_reason: str | None = None,
) -> ConsolidationPlan:
    """Each pass's gather and estimate; no write, no LLM call, no lock.

    ``projection_skip_reason`` is the Surface's verdict on pass 6, as
    :func:`~particles.operations.consolidation.run_consolidation` receives it.
    """
    from particles.corpus.store import (
        list_pending_snapshots_for_catchup,
        list_refreshable_local_entries,
    )

    config = get_config()
    cfg = config.consolidation
    plan = ConsolidationPlan(
        store=store, scope=scope, effective_scope=scope, budget_usd=cfg.budget_usd
    )
    semantic_ok, degrade_reason = _semantic_availability(structural_only)
    plan.semantic_degraded_reason = None if semantic_ok else degrade_reason
    passes = plan.passes

    # pass 0.5 — local refresh (zero-LLM)
    if not config.local_refresh.enabled:
        passes.append(_skipped("refresh", "local_refresh.enabled is false"))
    else:
        entries = await list_refreshable_local_entries(session)
        checked = min(len(entries), config.local_refresh.max_entries)
        passes.append(
            _planned("refresh", f"would check {_plural(checked, 'local source')}; no LLM call")
        )

    # pass 1 — extract catch-up
    pending = await list_pending_snapshots_for_catchup(session)
    if not cfg.extract_pending:
        passes.append(_skipped("extract", "consolidation.extract_pending is false"))
    elif not semantic_ok:
        passes.append(
            _skipped(
                "extract",
                f"extraction is LLM-priced ({degrade_reason}); "
                f"{_plural(len(pending), 'snapshot')} pending",
            )
        )
    else:
        chosen = _extract_selection(pending, batching=cfg.extract_batching)
        sizes = await _snapshot_sizes(session, [snapshot_id for _, snapshot_id in chosen])
        estimate = extraction_estimate(sizes)
        passes.append(
            _planned(
                "extract",
                f"would extract {len(chosen)} of {_plural(len(pending), 'pending snapshot')} "
                f"(consolidation.max_pending_entries = {cfg.max_pending_entries}), "
                f"{sum(sizes):,} source bytes; listed before the run's stale-claim reset "
                "and superseded-generation collapse",
                estimate,
            )
        )

    # §4 delta scope. The run computes it after pass 1; the plan cannot.
    scope_ids: frozenset[str] | None = None
    watermark: datetime | None = None
    if scope == "delta":
        prior = await latest_run_event(
            session, actor=actor, successful_only=True, exclude_degraded=True
        )
        if prior is None:
            plan.effective_scope = "store"
        else:
            watermark = _event_started_at(prior)
            plan.watermark = watermark
            scope_ids = await _delta_scope_ids(session, watermark)
            plan.scope_particle_count = len(scope_ids)

    # pass 2 — reconcile sweep
    if not semantic_ok:
        passes.append(
            _skipped("reconcile", f"replacement-signal probes are LLM-priced ({degrade_reason})")
        )
    else:
        pairs = await count_supersession_candidates(session)
        probes = min(pairs, cfg.max_reconcile_probes)
        passes.append(
            _planned(
                "reconcile",
                f"{_plural(pairs, 'candidate pair')}, would probe {probes} "
                f"(consolidation.max_reconcile_probes = {cfg.max_reconcile_probes})",
                probe_estimate(probes),
            )
        )

    # pass 2b — update supersession
    if not semantic_ok:
        passes.append(
            _skipped(
                "reconcile_updates", f"update-supersession probes are LLM-priced ({degrade_reason})"
            )
        )
    elif not config.reconciliation.update_supersession.enabled:
        passes.append(
            _skipped("reconcile_updates", "reconciliation.update_supersession.enabled is false")
        )
    else:
        pairs = await count_update_candidates(session, scope_ids)
        probes = min(pairs, cfg.max_update_probes)
        passes.append(
            _planned(
                "reconcile_updates",
                f"{_plural(pairs, 'qualifying pair')}, would probe {probes} "
                f"(consolidation.max_update_probes = {cfg.max_update_probes}); "
                "up to a second probe each",
                probe_estimate(2 * probes),
            )
        )

    # pass 2c — re-anchor (its own dry run: no call, no write)
    if not semantic_ok:
        passes.append(_skipped("reanchor", f"re-anchor probes are LLM-priced ({degrade_reason})"))
    elif not cfg.reanchor.enabled:
        passes.append(_skipped("reanchor", "consolidation.reanchor.enabled is false"))
    else:
        cursor = await prior_cursor(session, actor)
        reanchor = await run_reanchor(session, cursor=cursor, actor=actor, dry_run=True)
        detail = (
            reanchor.skipped_reason
            if reanchor.skipped_reason
            else (
                f"{_plural(reanchor.triggers, 'update retirement')}, "
                f"{_plural(reanchor.candidates, 'candidate claim')}; "
                "up to a second reading each"
            )
        )
        passes.append(_planned("reanchor", detail, _reanchor_estimate(reanchor)))

    # pass 3 — census: its own cadence first, then the probe's
    # candidate pairs, gathered with a cap of 0, over changes since the last
    # census (store-wide when none is on record or on --scope store).
    census_event = await latest_census_event(session, actor=actor)
    census = decide_census(
        enabled=cfg.census.enabled,
        interval_hours=cfg.census.interval_hours,
        last_ran=_event_started_at(census_event) if census_event is not None else None,
        now=_utcnow(),
        store_wide=scope == "store",
    )
    census_scope_ids: frozenset[str] | None = None
    if census.run and scope == "delta" and census.last_ran is not None:
        census_scope_ids = await _delta_scope_ids(session, census.last_ran)
    if not census.run:
        passes.append(_skipped("census", census.reason or "census skipped"))
    elif not semantic_ok:
        passes.append(
            _planned(
                "census",
                f"structural finders only; contradiction probe not run ({degrade_reason})",
            )
        )
    else:
        audit = config.audit
        control = ContradictionProbeControl(
            max_probes=0,
            scope_particle_ids=census_scope_ids,
            exclude_pairs=await covered_pair_set(session),
        )
        await collect_cards(
            session,
            semantic=True,
            duplicate_mode=SuggestMode.REPORT,
            contradiction_probe=control,
            duplicate_scope_ids=census_scope_ids,
        )
        probes = min(control.candidate_pairs, audit.max_contradiction_probes)
        readings = (
            min(probes, audit.max_contradiction_verifications) if audit.verify_contradictions else 0
        )
        passes.append(
            _planned(
                "census",
                f"{_plural(control.candidate_pairs, 'candidate pair')}, would probe {probes} "
                f"(audit.max_contradiction_probes = {audit.max_contradiction_probes})"
                + (
                    f" and read up to {readings} flag(s) a second time "
                    f"(audit.max_contradiction_verifications)"
                    if readings
                    else ""
                ),
                probe_estimate(probes) + context_call_estimate("verification", readings),
            )
        )

    # passes 3b and 4 — zero-LLM
    passes.append(_planned("disclose", "opens, closes and regroups census records; no LLM call"))
    passes.append(_planned("curation", "persists the card collection; no LLM call"))

    # pass 5 — utility mining: the plan IS the pass's gather
    if not config.utility.mining.enabled:
        passes.append(_skipped("utility", "utility.mining.enabled is false"))
    else:
        mines = await _plan_utility(
            session, watermark=watermark if scope_ids is not None else None, behavioural=semantic_ok
        )
        groups = sum(len(mine.groups) for mine in mines)
        shown = sum(mine.candidates for mine in mines)
        nominated = sum(len(mine.nominated) for mine in mines)
        passes.append(
            _planned(
                "utility",
                f"{_plural(len(mines), 'session')} to mine, {shown} shown belief(s), "
                f"{nominated} literal nomination(s); "
                f"would make {_plural(groups, 'judge call')} "
                f"(utility.mining.max_behavioural_calls = "
                f"{config.utility.mining.max_behavioural_calls})",
                estimate_judge_calls(mines),
            )
        )

    # pass 5b — abstraction, at its cap (its gather is the promotion itself)
    if not cfg.abstraction.enabled:
        passes.append(_skipped("abstraction", "consolidation.abstraction.enabled is false"))
    elif not semantic_ok:
        passes.append(
            _skipped("abstraction", f"synthesis + judges are LLM-priced ({degrade_reason})")
        )
    else:
        passes.append(
            _planned(
                "abstraction",
                f"up to {cfg.abstraction.max_promotions_per_run} promotion(s) "
                "(consolidation.abstraction.max_promotions_per_run), three calls each, at the cap",
                _abstraction_cap_estimate(),
            )
        )

    # pass 6 — projection (Surface-injected; deposits and renders, no LLM call)
    if projection_skip_reason is not None:
        passes.append(_skipped("projection", projection_skip_reason))
    else:
        passes.append(
            _planned(
                "projection",
                "harvests memory files and re-renders the projection; no LLM call "
                "(a later run's pass 1 extracts what it deposits)",
            )
        )

    # pass 6b — the closure measure: reads only, never priced
    passes.append(
        _planned(
            "measure",
            "counts contested beliefs store-wide and the lifecycle transitions since the "
            "previous run; no LLM call",
        )
    )

    _simulate_budget(plan)
    return plan


def _simulate_budget(plan: ConsolidationPlan) -> None:
    """Mark the passes ``consolidation.budget_usd`` would skip at these estimates."""
    if plan.budget_usd is None:
        return
    spent = 0.0
    for planned in plan.passes:
        if planned.action != "run" or not planned.llm_priced or planned.calls == 0:
            continue
        verdict = decide_spend(
            spent_usd=spent, budget_usd=plan.budget_usd, estimate_usd=planned.estimate_usd
        )
        if verdict == "skip":
            estimate = (
                "unpriced" if planned.estimate_usd is None else format_usd(planned.estimate_usd)
            )
            planned.budget_skip = (
                f"would be skipped, US{format_usd(spent)} of US{format_usd(plan.budget_usd)} "
                f"spent before it and US{estimate} more"
                if planned.estimate_usd is not None
                else f"would be skipped, US{format_usd(spent)} of "
                f"US{format_usd(plan.budget_usd)} spent before it"
            )
            continue
        spent += planned.estimate_usd or 0.0


def _usd(value: float | None) -> str:
    return "unpriced" if value is None else f"≈ {format_usd(value)}"


def render_consolidation_plan(plan: ConsolidationPlan) -> str:
    """The human dry-run report: one line per pass, the total, and how to read it."""
    lines = [
        f"Consolidation dry run for store '{plan.store}': nothing written, no LLM call made.",
    ]
    if plan.effective_scope == "delta" and plan.watermark is not None:
        lines.append(
            f"Semantic scope: delta since {plan.watermark:%Y-%m-%d %H:%M} UTC "
            f"({plan.scope_particle_count or 0} beliefs, before pass 1 adds its extractions)."
        )
    else:
        lines.append("Semantic scope: the whole store.")
    if plan.semantic_degraded_reason:
        lines.append(f"LLM passes would be skipped: {plan.semantic_degraded_reason}.")
    lines.append("")
    for planned in plan.passes:
        head = f"  {planned.position:>2} {planned.name:<18}"
        if planned.action == "skip":
            lines.append(f"{head}skipped: {planned.detail}")
            continue
        line = f"{head}{planned.detail}"
        if planned.llm_priced:
            line += f"; ~{_plural(planned.calls, 'LLM call')}, {_usd(planned.estimate_usd)}"
        if planned.budget_skip:
            line += f" [budget: {planned.budget_skip}]"
        lines.append(line)
    lines.append("")
    lines.append(
        f"Estimated total: ~{_plural(plan.total_calls, 'LLM call')}, "
        f"{_usd(plan.total_usd)} at list price."
    )
    if plan.budget_usd is not None:
        skipped = [p.name for p in plan.passes if p.budget_skip]
        lines.append(
            f"Budget: consolidation.budget_usd = US{format_usd(plan.budget_usd)}"
            + (
                f"; at these estimates the run would skip {', '.join(skipped)}."
                if skipped
                else "; every pass fits at these estimates."
            )
        )
    bases: list[str] = []
    for planned in plan.passes:
        if planned.action != "run":
            continue
        for key in planned.basis.replace("; ", ", ").split(", "):
            if key and key not in bases:
                bases.append(key)
    lines.append(
        "These are list-price estimates over estimated tokens (llm.price_per_mtok and the "
        "per-call figures "
        + (", ".join(bases) if bases else "audit.estimate_*")
        + "), with no batch or prompt-cache discount. A metered run prints its measured "
        f'cost afterwards as the "{MEASURED_LINE}" line and records it on the '
        "CONSOLIDATION_RUN event; compare the two."
    )
    return "\n".join(lines) + "\n"
