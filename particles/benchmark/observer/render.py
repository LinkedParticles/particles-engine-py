# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Markdown rendering of an observer-fixture report."""

from __future__ import annotations

from particles.benchmark.observer.schema import (
    ObserverArm,
    ObserverMetrics,
    ObserverReport,
    WriteCensus,
)
from particles.benchmark.rot.schema import Rate


def _rate(rate: Rate) -> str:
    if rate.value is None:
        return "n/a (0 eligible)"
    return f"{100 * rate.value:.1f}% ({rate.numerator} / {rate.denominator})"


def _metrics_rows(m: ObserverMetrics) -> list[str]:
    rows = [
        "| Own lines in view for their project | " + _rate(m.own_visible) + " |",
        "| … the same lines ACTIVE anywhere (no lens) | " + _rate(m.own_visible_store_wide) + " |",
        "| Own lines retired by the other project's activity | "
        + _rate(m.vanished_cross_project)
        + " |",
        "| Winner in view after a cross-project supersession | " + _rate(m.winner_in_view) + " |",
        "| Lines in view a project never stated (lens leak) | " + _rate(m.leaked) + " |",
    ]
    for kind in ("fact", "rule", "own"):
        if kind in m.by_kind:
            rows.append(f"| In view — `{kind}` lines | {_rate(m.by_kind[kind])} |")
    return rows


def _census_rows(c: WriteCensus, arm: ObserverArm) -> list[str]:
    rows = [
        f"| Own-value supersessions (`SUPERSEDED_BY_UPDATE`) | {c.own_supersessions} |",
        f"| Cross-project supersessions | {c.cross_project_supersessions} |",
        f"| Own generation-cascade retirements | {c.own_cascades} |",
        f"| Cross-project cascade retirements | {c.cross_project_cascades} |",
        f"| Candidates born superseded (rung 2.5 mirror) | {c.candidates_born_superseded} |",
        f"| Pairs declined by the observer precondition | {c.declined_pairs} |",
        f"| `CONTRADICTS` relations (`OBSERVER_DIVERGENCE`) | {c.divergences_recorded} |",
        f"| Declined pairs with no recorded relation | {c.declined_without_relation} |",
        "| Global line contested by a project → held in review | "
        + _rate(c.global_contests)
        + " |",
    ]
    if arm is ObserverArm.CHUNKED:
        total = c.chunks_carried + c.chunks_extracted
        rows.append(f"| Chunks carried forward / all chunks | {c.chunks_carried} / {total} |")
    return rows


def render_report(report: ObserverReport) -> str:
    """The report as Markdown: pooled metrics, causes, and one line per world."""
    m = report.metrics
    lines = [
        "# Two-project observer fixture (gate B)",
        "",
        f"Seeds {', '.join(str(s) for s in report.seeds)} · {report.days} days · "
        f"arm `{report.arm.value}` · "
        f"store mode `{report.store_mode}` · generator v{report.generator_version} · "
        f"zero LLM calls by construction.",
        "",
        "| Measure | Rate |",
        "|---|---|",
        *_metrics_rows(m),
        "",
        "**Why an own line was not in view** (counts over own-line checkpoints):",
        "",
    ]
    if m.vanished_by_cause:
        for cause, n in sorted(m.vanished_by_cause.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{cause}`: {n}")
    else:
        lines.append("- none")
    lines += [
        "",
        "**What the write path did**, each retirement attributed by the retired claim's "
        "scope just before the extraction that retired it:",
        "",
        "| Measure | Count |",
        "|---|---|",
        *_census_rows(report.write_census, report.arm),
    ]
    lines += [
        "",
        "| Seed | Deposits | Own in view | Store-wide | Causes |",
        "|---|---|---|---|---|",
    ]
    for w in report.worlds:
        causes = ", ".join(f"{k} {n}" for k, n in sorted(w.metrics.vanished_by_cause.items()))
        lines.append(
            f"| {w.seed} | {w.deposits} | {_rate(w.metrics.own_visible)} | "
            f"{_rate(w.metrics.own_visible_store_wide)} | {causes or '—'} |"
        )
    if report.refused_llm_calls:
        lines += [
            "",
            "Refused LLM calls: "
            + ", ".join(f"{k} ×{n}" for k, n in sorted(report.refused_llm_calls.items())),
        ]
    for note in report.quality_notes:
        lines += ["", f"_{note}_"]
    return "\n".join(lines) + "\n"
