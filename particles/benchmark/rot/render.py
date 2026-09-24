# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plain-text rendering of a memory-rot report.

The three metric families are headed separately and there is no aggregate
score line — the rule. Every rate prints with its numerator and
denominator, and an empty denominator prints ``n/a``, never a number.
"""

from __future__ import annotations

from particles.benchmark.rot.schema import Rate, RotBenchmarkReport, RotMetrics

_FIRST_HIT_ORDER = (
    "current",
    "mixed",
    "stale",
    "history",
    "poison_asserted",
    "poison_attributed",
    "miss",
)


def _rate(r: Rate) -> str:
    v = r.value
    return "n/a (0 eligible)" if v is None else f"{v:6.1%}  ({r.numerator}/{r.denominator})"


def _families(m: RotMetrics, indent: str = "  ") -> list[str]:
    return [
        f"{indent}CURRENCY (↑)",
        f"{indent}  recall_current@k        {_rate(m.recall_current_at_k)}",
        f"{indent}  current_first           {_rate(m.current_first)}",
        f"{indent}SUPERSESSION (↓)",
        f"{indent}  stale_over_current      {_rate(m.stale_over_current)}",
        f"{indent}    …with no contested badge {_rate(m.stale_over_current_unflagged)}",
        f"{indent}  stale_retained@k        {_rate(m.stale_retained_at_k)}",
        f"{indent}POISON (↓)",
        f"{indent}  poison_leak@k           {_rate(m.poison_leak_at_k)}",
        f"{indent}  poison_first            {_rate(m.poison_first)}",
        f"{indent}  poison_surfaced@k       {_rate(m.poison_surfaced_at_k)}",
    ]


def _short(m: RotMetrics) -> str:
    def v(r: Rate) -> str:
        x = r.value
        return "  n/a" if x is None else f"{x:5.0%}"

    return (
        f"recall {v(m.recall_current_at_k)} | first {v(m.current_first)} | "
        f"stale-first {v(m.stale_over_current)} | stale@k {v(m.stale_retained_at_k)} | "
        f"poison@k {v(m.poison_leak_at_k)}"
    )


def render_report(report: RotBenchmarkReport) -> str:
    """The report as a terminal table."""
    s = report.selection
    lines = [
        f"Memory-rot benchmark — arm {s.arm.upper()}",
        f"  seeds {s.seeds} · {s.days} days · checkpoints {s.checkpoints} · top_k {s.top_k}",
        f"  trust policy {'on' if s.trust_policy else 'OFF'} "
        f"(untrusted domain trust {s.untrusted_domain_trust}) · "
        f"generator v{s.generator_version} · scorer v{s.scorer_version}",
        f"  extraction={s.extraction_model_id} · probe={s.semantic_lint_model_id} · "
        f"embedding={s.embedding_model_id}",
        "",
        "Pooled over all worlds:",
        *_families(report.metrics),
        "",
        "First value-bearing hit (currency probes):",
    ]
    counts = report.metrics.first_hit_counts
    total = sum(counts.values()) or 1
    for key in _FIRST_HIT_ORDER:
        n = counts.get(key, 0)
        lines.append(f"  {key:<18} {n:4d}  ({n / total:5.1%})")
    lines += ["", "By checkpoint:"]
    for cp in sorted(report.by_checkpoint):
        lines.append(f"  day {cp:>3}: {_short(report.by_checkpoint[cp])}")
    lines += ["", "By update phrasing (changed slots):"]
    for key in sorted(report.by_phrasing):
        lines.append(f"  {key:<12} {_short(report.by_phrasing[key])}")
    lines += ["", "By poison channel:"]
    for key in sorted(report.by_channel):
        m = report.by_channel[key]
        lines.append(
            f"  {key:<10} leak@k {_rate(m.poison_leak_at_k)} · "
            f"surfaced@k {_rate(m.poison_surfaced_at_k)}"
        )
    lines += ["", "Per world:"]
    for w in report.worlds:
        census = ", ".join(f"{k}={v}" for k, v in sorted(w.store_census.items()))
        lines.append(f"  seed {w.seed} ({w.sessions_deposited} sessions): {_short(w.metrics)}")
        lines.append(f"    store: {census}")
    lines += [
        "",
        "Relevance-floor sweep (retrieval-grounded proxy on synthetic data):",
        "  floor  answerable-but-refused        unanswerable-but-passed",
    ]
    for row in report.floor_sweep:
        lines.append(
            f"  {row.floor:.2f}   {_rate(row.answerable_refused):<28}  "
            f"{_rate(row.unanswerable_passed)}"
        )
    if report.quality_notes:
        lines += ["", "Notes:"]
        lines += [f"  - {n}" for n in report.quality_notes]
    return "\n".join(lines)
