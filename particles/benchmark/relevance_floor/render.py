# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plain-text rendering of a relevance-floor report.

**Aggregate-only by construction**: nothing here prints a question, an answer,
or a verdict, so the rendered table is safe to publish while the report of
record (``--format json``) stays private. The refusal curve and the judged
sweep are headed separately and there is no aggregate score line. Every rate
prints with its numerator and denominator; an empty denominator prints
``n/a``, never a number.
"""

from __future__ import annotations

from particles.benchmark.rot.schema import Rate

from .schema import RelevanceFloorReport


def _rate(r: Rate) -> str:
    v = r.value
    return (
        f"{'n/a':>6}  (0 eligible)" if v is None else f"{v:6.1%}  ({r.numerator}/{r.denominator})"
    )


def _marker(floor: float, configured: float) -> str:
    return "  ← configured" if abs(floor - configured) < 1e-9 else ""


def render_report(report: RelevanceFloorReport) -> str:
    """The aggregate table — never a question, an answer, or a verdict."""
    sel = report.selection
    sources = ", ".join(f"{name} {count}" for name, count in sorted(sel.heldout_by_source.items()))
    sample = (
        f" (sample limit {sel.sample_limit}, seed {sel.sample_seed})" if sel.sample_limit else ""
    )
    lines = [
        "Relevance-floor benchmark — the query gate on real questions",
        f"  held-out set   {sel.heldout_questions} questions{sample} [{sources}]",
        f"                 fingerprint {sel.heldout_fingerprint}",
        f"  store          {sel.store_active_particles:,} ACTIVE particles",
        f"  encoder        {sel.embedding_model_id}   top_k {sel.top_k}   "
        f"configured floor {sel.configured_floor:.2f}",
    ]
    if sel.judged:
        lines.append(
            f"  answer model   {sel.answer_model_id}   judge {sel.judge_model_id} "
            f"(protocol {sel.judge_protocol})"
        )
    if report.similarity_quantiles:
        quantiles = "  ".join(f"{k} {v:.3f}" for k, v in report.similarity_quantiles.items())
        lines += ["", "MAX COSINE OVER THE RENDERED TOP-K", f"  {quantiles}"]

    source_names = sorted({s for row in report.refusal_curve for s in row.refused_by_source})
    lines += ["", "REFUSAL CURVE — what the gate does (unjudged, no LLM call)"]
    header = f"  {'floor':<6} {'refused':<24}" + "".join(f"{s:<24}" for s in source_names)
    lines.append(header)
    for row in report.refusal_curve:
        per_source = "".join(
            f"{_rate(row.refused_by_source.get(s, Rate())):<24}" for s in source_names
        )
        lines.append(
            f"  {row.floor:<6.2f} {_rate(row.refused):<24}{per_source}"
            f"{_marker(row.floor, sel.configured_floor)}"
        )

    lines += ["", "JUDGED SWEEP — whether the gate was right"]
    if not report.sweep:
        lines.append("  not run (pass --judge; it is LLM-priced and estimate-gated)")
    else:
        lines.append(f"  judged answerable (gate off): {_rate(report.answerable).strip()}")
        lines.append(
            f"  {'floor':<6} {'answerable→refused ↓':<24}{'unanswerable→passed ↓':<24}"
            f"{'refused that were ans. ↓':<26}{'passed that were unans. ↓':<26}"
        )
        for sweep_row in report.sweep:
            lines.append(
                f"  {sweep_row.floor:<6.2f} {_rate(sweep_row.answerable_refused):<24}"
                f"{_rate(sweep_row.unanswerable_passed):<24}"
                f"{_rate(sweep_row.refused_were_answerable):<26}"
                f"{_rate(sweep_row.passed_were_unanswerable):<26}"
                f"{_marker(sweep_row.floor, sel.configured_floor)}"
            )

    if report.quality_notes:
        lines += ["", "NOTES"]
        lines.extend(f"  · {note}" for note in report.quality_notes)
    return "\n".join(lines) + "\n"
