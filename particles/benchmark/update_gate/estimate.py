# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What a run will cost, printed before any call.

The end-to-end case takes its estimate from the first of three sources that
applies, and says which:

1. ``profile``: a measured usage profile shipped beside this module, one entry
   per case and model a report of record measured (calls and tokens per
   purpose for the largest run of each variant), priced at each pinned model's
   ``llm.price_per_mtok`` entry. The entry for the chosen case and model is
   used when there is one. A case no report has measured yet takes the same
   model's mid-trip entry, and any other model the first entry;
   either is an approximation and the line says so: the input tokens come
   mostly from the fixture, the output tokens and the number of probes from
   the model and notes that ran.
2. ``record``: the largest measured per-run cost of each variant in the
   2026-10-12 record, for the two models it measured, one model on every
   purpose, and the mid-trip case only. It cannot be re-priced for another
   model: the record kept costs, not token rows.
3. ``bound``: an upper bound from the configuration (the audit's extraction
   estimate over the notes, each audit's probe cap, and the consolidation
   probe caps), for a priced model with neither. It reads high.

An unpriced model gets a call count and no dollar figure. The pair replay's
estimate is exact in calls (an upper bound: rows × samples × readings).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from particles.benchmark.update_gate.fixtures import CASES, LEGACY_CASE, case_note
from particles.benchmark.update_gate.schema import TripVsMoveReport
from particles.config import ProviderSelection, TokenPrice, get_config
from particles.llm.usage import format_usd

#: The profile file; absent until the first report of record supplies it.
PROFILE_PATH = Path(__file__).with_name("usage_profile.json")

#: The largest measured run of each variant on 2026-10-12, every purpose on one
#: model (``docs/benchmarks/trip-vs-move-end-to-end-2026-10-12.json``).
RECORD_RUN_USD: dict[str, dict[str, float]] = {
    "anthropic:claude-sonnet-5-5": {"move": 0.1384, "trip-alone": 0.0757},
    "anthropic:claude-sonnet-4-6": {"move": 0.2406, "trip-alone": 0.1112},
}
#: What each pair set cost per model on 2026-10-12, 3 samples (the gate sets
#: were recorded together), printed beside the replay's own estimate.
RECORD_PAIRS_USD: dict[str, dict[str, float]] = {
    "anthropic:claude-sonnet-5-5": {
        "gate+gate-heldout": 1.0776,
        "past": 0.3319,
        "past-marker": 0.3419,
    },
    "anthropic:claude-sonnet-4-6": {
        "gate+gate-heldout": 1.1401,
        "past": 0.3501,
        "past-marker": 0.3594,
    },
}


@dataclass(frozen=True)
class CaseEstimate:
    """The case's estimate: its source, per-run figures, and the total."""

    source: str
    #: Variant → US$ per run (``None`` when a pinned model is unpriced).
    per_run_usd: Mapping[str, float | None]
    #: Variant → LLM calls per run (``None`` when the source does not count calls).
    per_run_calls: Mapping[str, int | None]
    runs: Mapping[str, int]
    note: str

    @property
    def total_usd(self) -> float | None:
        """US$ for every run, or ``None`` when any variant is unpriced."""
        total = 0.0
        for variant, n in self.runs.items():
            usd = self.per_run_usd.get(variant)
            if n and usd is None:
                return None
            total += (usd or 0.0) * n
        return total

    @property
    def total_calls(self) -> int | None:
        """LLM calls for every run, or ``None`` when the source does not count them."""
        total = 0
        for variant, n in self.runs.items():
            calls = self.per_run_calls.get(variant)
            if n and calls is None:
                return None
            total += (calls or 0) * n
        return total

    def run_usd(self, variant: str) -> float | None:
        """US$ for one run of ``variant``."""
        return self.per_run_usd.get(variant)


def _price(
    price: TokenPrice | None,
    row: Mapping[str, Any],
    batched: bool = False,
    selection: ProviderSelection | None = None,
) -> float | None:
    """A token row at list price, the arithmetic ``particles.llm.usage`` prices with.

    ``selection`` picks the model's cache-read multiplier; without it the
    standard ``llm.cache_read_price_multiplier`` applies.
    """
    if price is None:
        return None
    llm = get_config().llm
    cache_read = (
        llm.cache_read_price_multiplier
        if selection is None
        else llm.cache_read_multiplier_for(selection)
    )
    cost = (
        float(
            row.get("input_tokens", 0) * price.input
            + row.get("cache_creation_tokens", 0) * price.input * llm.cache_write_price_multiplier
            + row.get("cache_read_tokens", 0) * price.input * cache_read
            + row.get("output_tokens", 0) * price.output
        )
        / 1_000_000
    )
    return cost * (1.0 - llm.batch_discount) if batched else cost


def _key(selection: ProviderSelection) -> str:
    return f"{selection.provider}:{selection.model}"


def load_profile(path: Path | None = None) -> dict[str, Any] | None:
    """The shipped usage profile, or ``None`` before the first report of record."""
    path = path or PROFILE_PATH
    if not path.exists():
        return None
    profile: dict[str, Any] = json.loads(path.read_text())
    return profile


def _profile_entries(profile: Mapping[str, Any], case: str) -> tuple[Mapping[str, Any], str]:
    """The profile's model entries for ``case``, or the mid-trip ones, and which case."""
    cases: Mapping[str, Any] = profile.get("cases", {})
    if cases.get(case, {}).get("models"):
        return cases[case]["models"], case
    return cases.get(LEGACY_CASE, {}).get("models", {}), LEGACY_CASE


def _from_profile(
    profile: Mapping[str, Any], pins: Mapping[str, ProviderSelection], case: str
) -> tuple[dict[str, float | None], dict[str, int | None], str]:
    llm = get_config().llm
    usd: dict[str, float | None] = {}
    calls: dict[str, int | None] = {}
    entries, measured_case = _profile_entries(profile, case)
    measured = _key(pins["default"])
    if measured not in entries:
        measured = next(iter(entries))
    for variant, rows in entries[measured]["variants"].items():
        total: float | None = 0.0
        n = 0
        for row in rows:
            selection = pins.get(row["purpose"]) or pins["default"]
            cost = _price(llm.price_for(selection), row, bool(row.get("batched")), selection)
            total = None if cost is None or total is None else total + cost
            n += int(row.get("calls", 0))
        usd[variant] = total
        calls[variant] = n
    models = {_key(s) for s in pins.values()}
    note = f"measured usage from {entries[measured].get('source', 'the report of record')}"
    if models != {measured}:
        note += f", measured on {measured}: an approximation for {', '.join(sorted(models))}"
    if measured_case != case:
        note += (
            f", measured on the {measured_case} case: an approximation for the {case} case, "
            "which no report has measured yet"
        )
    return usd, calls, note


def profile_from_reports(reports: Sequence[TripVsMoveReport]) -> dict[str, Any]:
    """The usage profile the reports of record supply: one entry per case and model.

    For each case, model and variant, the run that cost the most at list price
    (the most tokens when unpriced), as calls and tokens per purpose. Live runs
    on one model on every purpose only: a mixture or a scripted run measures
    something else.
    """
    cases: dict[str, Any] = {}
    for report in reports:
        if report.arm != "live" or report.mixed:
            continue
        models = cases.setdefault(report.case, {"models": {}})["models"]
        entry = models.setdefault(report.model, {"variants": {}})
        entry["source"] = (
            f"the trip-vs-move report run at {report.run_at.strftime('%Y-%m-%d %H:%M')} UTC"
        )
        for variant in ("move", "trip-alone"):
            runs = [r for r in report.runs if r.variant == variant and r.usage is not None]
            if not runs:
                continue

            def weight(run: Any) -> float:
                usage = run.usage
                return float(usage.priced_cost_usd) or float(
                    sum(row.input_tokens + row.output_tokens for row in usage.rows)
                )

            largest = max(runs, key=weight).usage
            assert largest is not None  # filtered above
            entry["variants"][variant] = [
                {
                    "purpose": row.purpose,
                    "batched": row.batched,
                    "calls": row.calls,
                    "input_tokens": row.input_tokens,
                    "output_tokens": row.output_tokens,
                    "cache_creation_tokens": row.cache_creation_tokens,
                    "cache_read_tokens": row.cache_read_tokens,
                }
                for row in largest.rows
                if not row.replayed
            ]
    return {"cases": cases}


def _bound(
    pins: Mapping[str, ProviderSelection], case: str
) -> tuple[dict[str, float | None], dict[str, int | None]]:
    # Deferred import: the audit estimate pulls in the extraction stack.
    from particles.operations.audit import estimate_extraction

    cfg = get_config()
    llm = cfg.llm
    usd: dict[str, float | None] = {}
    calls: dict[str, int | None] = {}

    def at(purpose: str) -> TokenPrice | None:
        return llm.price_for(pins.get(purpose) or pins["default"])

    for variant, notes in CASES[case].variants.items():
        sizes = [len(case_note(n).read_text()) for n in notes]
        est = estimate_extraction(sizes)
        audits = len(notes)
        consolidation_probes = (
            cfg.consolidation.max_reconcile_probes
            + cfg.consolidation.max_update_probes
            + cfg.audit.max_contradiction_probes
        )
        probe_row = {
            "input_tokens": cfg.audit.estimate_probe_input_tokens,
            "output_tokens": cfg.audit.estimate_probe_output_tokens,
        }
        parts = [
            _price(
                at("extraction"),
                {
                    "input_tokens": est.input_tokens + est.retry_input_tokens,
                    "output_tokens": est.output_tokens_high,
                },
            ),
            # Every audit pays its probe cap; the estimate counts one.
            _price(
                at("semantic_lint"),
                {
                    "input_tokens": est.probe_input_tokens * audits,
                    "output_tokens": est.probe_output_tokens * audits,
                },
            ),
            _price(
                at("verification"),
                {
                    "input_tokens": est.verify_input_tokens * audits,
                    "output_tokens": est.verify_output_tokens * audits,
                },
            ),
            _price(
                at("semantic_lint"),
                {k: v * consolidation_probes for k, v in probe_row.items()},
            ),
        ]
        usd[variant] = None if any(p is None for p in parts) else sum(p or 0.0 for p in parts)
        calls[variant] = (
            est.estimated_llm_calls
            + round(est.expected_retries)
            + (est.probe_cap + est.reconcile_probes + est.verify_cap) * audits
            + consolidation_probes
        )
    return usd, calls


def estimate_case(
    pins: Mapping[str, ProviderSelection],
    runs: Mapping[str, int],
    *,
    mixed: bool,
    case: str,
) -> CaseEstimate:
    """The ``case``'s estimate for ``runs`` per variant under ``pins``."""
    profile = load_profile()
    if profile is not None and _profile_entries(profile, case)[0]:
        usd, calls, note = _from_profile(profile, pins, case)
        return CaseEstimate("profile", usd, calls, runs, note)
    model = _key(pins["default"])
    if not mixed and case == LEGACY_CASE and model in RECORD_RUN_USD:
        per_run: dict[str, float | None] = dict(RECORD_RUN_USD[model])
        return CaseEstimate(
            "record",
            per_run,
            dict.fromkeys(per_run),
            runs,
            "the largest measured run of each variant on 2026-10-12, at that day's list prices",
        )
    usd, calls = _bound(pins, case)
    return CaseEstimate(
        "bound",
        usd,
        calls,
        runs,
        "an upper bound from the configured caps, not a measurement; it reads high",
    )


def default_ceiling(estimate: CaseEstimate, variant: str, multiplier: float) -> float | None:
    """The per-run ceiling ``--max-usd`` defaults to.

    ``multiplier`` times a measured estimate (``profile`` or ``record``); the
    bound itself under ``bound``, since it is already a ceiling; ``None`` (no
    dollar ceiling) for an unpriced model.
    """
    usd = estimate.run_usd(variant)
    if usd is None:
        return None
    return usd if estimate.source == "bound" else usd * multiplier


def render_case_estimate(estimate: CaseEstimate, model: str) -> str:
    """The line printed before any call."""
    parts = []
    for variant, n in estimate.runs.items():
        if not n:
            continue
        usd = estimate.run_usd(variant)
        each = "no dollar figure" if usd is None else f"about US{format_usd(usd)} a run"
        parts.append(f"{n} {variant} run(s), {each}")
    total = estimate.total_usd
    head = (
        f"Estimate for {model}: "
        + "; ".join(parts)
        + ". "
        + (
            "No dollar total: a pinned model has no llm.price_per_mtok entry."
            if total is None
            else f"Total about US{format_usd(total)}."
        )
    )
    calls = estimate.total_calls
    tail = (
        f" At most {calls} LLM calls." if calls is not None and estimate.source == "bound" else ""
    )
    return f"{head}{tail}\nSource: {estimate.source} ({estimate.note})."


def pairs_estimate(selection: ProviderSelection, calls: int) -> tuple[float | None, str]:
    """The pair replay's upper bound in US$ at ``selection``'s price, and its basis."""
    audit = get_config().audit
    price = get_config().llm.price_for(selection)
    usd = _price(
        price,
        {
            "input_tokens": calls * audit.estimate_probe_input_tokens,
            "output_tokens": calls * audit.estimate_probe_output_tokens,
        },
    )
    return usd, "audit.estimate_probe_input_tokens / estimate_probe_output_tokens"


def render_pairs_estimate(selection: ProviderSelection, sets: Sequence[str], calls: int) -> str:
    """The pair replay's line printed before any call."""
    usd, basis = pairs_estimate(selection, calls)
    model = _key(selection)
    money = (
        "no dollar figure (no llm.price_per_mtok entry)"
        if usd is None
        else (f"at most about US{format_usd(usd)}")
    )
    line = f"Estimate for {model}: at most {calls} LLM calls, {money} ({basis})."
    record = RECORD_PAIRS_USD.get(model)
    if record:
        measured = ", ".join(f"{k} US{format_usd(v)}" for k, v in record.items())
        line += f"\nMeasured on 2026-10-12 with 3 samples: {measured}."
    return line
