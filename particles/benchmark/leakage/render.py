# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plain-text rendering of a leakage report.

**No question, answer, sentence, or judge reason is ever printed here**, so
the table is publishable while the report of record (``--format json``) stays
private. Per-question rows are keyed by question id. Every rate prints with
its numerator and denominator; an empty denominator prints ``n/a``.
"""

from __future__ import annotations

from particles.benchmark.rot.schema import Rate

from .schema import LeakageReport


def _rate(r: Rate) -> str:
    v = r.value
    return "n/a (0 eligible)" if v is None else f"{v:.1%} ({r.numerator}/{r.denominator})"


def render_report(report: LeakageReport, *, per_question: bool = False) -> str:
    """The aggregate table, plus one row per question when ``per_question``."""
    sel = report.selection
    sources = ", ".join(f"{name} {count}" for name, count in sorted(sel.by_source.items()))
    origin = (
        "ad hoc (--question)"
        if sel.ad_hoc
        else (
            f"held-out set, sample limit {sel.sample_limit} (seed {sel.sample_seed})"
            if sel.sample_limit
            else "held-out set"
        )
    )
    judge = sel.judge_model_id + ("" if sel.distinct_judge else "  (SAME MODEL AS COMPOSER)")
    counts = report.sentences
    mean = "n/a" if report.mean_answer_fraction is None else f"{report.mean_answer_fraction:.1%}"
    lines = [
        "Model-prior leakage — answer sentences the retrieved particles do not support",
        f"  questions      {sel.questions} [{sources}] from {origin}",
        f"                 fingerprint {sel.fingerprint}",
        f"  store          {sel.store} — {sel.store_active_particles:,} ACTIVE particles",
        f"  read           top_k {sel.top_k}   audience {sel.audience}   "
        f"floor {sel.configured_floor:.2f}   encoder {sel.embedding_model_id}",
        f"  composer       {sel.composer_model_id}"
        + ("   (grounded: cites ids, labels its own sentences)" if sel.grounded else ""),
        f"  judge          {judge} (protocol {sel.judge_protocol}, "
        f"{sel.context_sentences} context sentence(s))",
        "",
        "SILENT UNSUPPORTED SENTENCES" if sel.grounded else "UNSUPPORTED SENTENCES",
        f"  pooled over claim-bearing sentences   {_rate(report.unsupported_sentences)}",
    ]
    for source, rate in report.unsupported_sentences_by_source.items():
        lines.append(f"    {source:<36}{_rate(rate)}")
    lines += [
        f"  mean per-answer fraction              {mean}",
        f"  answers with ≥1 unsupported sentence  {_rate(report.answers_with_unsupported)}",
        f"  refused answers (never judged)        {_rate(report.refused)}",
    ]
    if sel.grounded:
        lines += [
            "",
            "LABELLED BY THE COMPOSER (disclosed, not silent)",
            f"  inference                             {_rate(report.inference_sentences)}"
            f"   judge found {counts.inference_entailed} supported",
            f"  background                            {_rate(report.background_sentences)}"
            f"   judge found {counts.background_entailed} supported",
            "",
            "ATTRIBUTION",
            f"  cited {counts.cited} ({counts.unsupported_cited} unsupported by the ids they "
            f"cite), unattributed {counts.unattributed} ({counts.unsupported_unattributed} "
            f"unsupported), invalid citations {counts.invalid_citations}",
        ]
    lines += [
        "",
        "SENTENCES",
        f"  {counts.total} total: {counts.supported} supported, {counts.unsupported} "
        f"unsupported, {counts.no_claim} no claim, {counts.judge_excluded} unscored",
    ]
    if per_question:
        lines += [
            "",
            "PER QUESTION",
            f"  {'question_id':<14}{'source':<13}{'hits':>5}  {'sup':>4}{'unsup':>6}"
            f"{'none':>5}{'n/s':>5}  fraction",
        ]
        for row in report.results:
            if row.excluded:
                outcome = f"excluded ({row.excluded})"
            elif row.refused:
                outcome = "refused"
            else:
                frac = row.unsupported_fraction
                outcome = "n/a" if frac is None else f"{frac:.1%}"
            lines.append(
                f"  {row.question_id:<14}{row.source.value:<13}{row.hit_count:>5}  "
                f"{row.supported:>4}{row.unsupported:>6}{row.no_claim:>5}"
                f"{row.judge_excluded:>5}  {outcome}"
            )
    if report.quality_notes:
        lines += ["", "NOTES"]
        lines.extend(f"  · {note}" for note in report.quality_notes)
    return "\n".join(lines) + "\n"
