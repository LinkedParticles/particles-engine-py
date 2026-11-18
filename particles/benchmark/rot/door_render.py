# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Markdown rendering of a door report.

The disclosures come first. Every metric renders in one table per family,
the expected-loss axes in the same table as the rest and never dropped; each
cell is the mean over builds with its range when there was more than one
build. The answer family renders below the door-text tables, headed with its
reader and prompt, never in their place. There is no aggregate score and no
ranking.
"""

from __future__ import annotations

from particles.benchmark.rot.answer import (
    ANSWER_EXPECTED_LOSSES,
    ANSWER_METRIC_DIRECTIONS,
    DOOR_COUNTERPART,
    HEADLINE_READER,
    PROBE_SETS,
    READINGS,
    SENSITIVITY_READER,
    ReplyVerdict,
    answer_key,
)
from particles.benchmark.rot.door import EXPECTED_LOSSES, METRIC_DIRECTIONS
from particles.benchmark.rot.door_schema import (
    AnswerMetrics,
    DoorMemoryResult,
    DoorReport,
    MetricRange,
)

_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Currency",
        ("current_in_door", "current_first", "recall_current_at_2k", "recall_current_at_8k"),
    ),
    ("Supersession", ("stale_over_current", "stale_in_door")),
    ("Source trust", ("poison_first", "poison_leak", "poison_surfaced")),
    ("Abstention", ("negative_silent", "negative_chars")),
    ("Cost and speed", ("chars_per_door", "write_ms", "door_ms", "build_minutes", "build_usd")),
)
_RATES = frozenset(
    {
        "current_in_door",
        "current_first",
        "recall_current_at_2k",
        "recall_current_at_8k",
        "stale_over_current",
        "stale_in_door",
        "poison_first",
        "poison_leak",
        "poison_surfaced",
        "negative_silent",
    }
)


#: The answer table's columns; ``answer_poison_disclosed`` is
#: reported beside the ranked metrics and has no direction.
_ANSWER_COLUMNS: tuple[str, ...] = (
    "answer_current",
    "answer_stale",
    "answer_hedged",
    "answer_poison",
    "answer_poison_disclosed",
    "answer_abstain",
    "answer_false_abstain",
    "reader_missed",
    "reader_tokens_in",
    "reader_usd",
)
_ANSWER_RATES = frozenset(_ANSWER_COLUMNS) - {"reader_tokens_in", "reader_usd"}


def _fmt(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    if metric in _RATES or metric in _ANSWER_RATES or metric == "reader_overflow":
        return f"{value:.1%}"
    if metric in ("build_usd", "reader_usd"):
        return f"${value:,.2f}"
    if metric == "build_minutes":
        return f"{value:,.1f}"
    return f"{value:,.0f}"


def _cell(metric: str, r: MetricRange | None) -> str:
    if r is None or r.mean is None:
        return "n/a"
    if r.builds > 1 and r.lo is not None and r.hi is not None and r.lo != r.hi:
        return f"{_fmt(metric, r.mean)} ({_fmt(metric, r.lo)}–{_fmt(metric, r.hi)})"
    return _fmt(metric, r.mean)


def _row_label(m: DoorMemoryResult) -> str:
    label = m.name
    if m.variant_note:
        label += f" ({m.variant_note})"
    if m.role != "headline":
        label += f" [{m.role}]"
    return label


def _rows(report: DoorReport) -> list[DoorMemoryResult]:
    """Headline and baseline rows, then every variant in the same table.

    A variant is never dropped from a family: the trust-rule row in particular
    sits at equal prominence in the source-trust table.
    """
    return [m for m in report.memories if m.role != "variant"] + [
        m for m in report.memories if m.role == "variant"
    ]


def _answer_cell(metric: str, m: AnswerMetrics, r: MetricRange | None) -> str:
    if metric == "answer_poison":
        rate = m.answer_poison
        if rate.value is None:
            return "n/a"
        return (
            f"{_cell(metric, r) if r is not None else _fmt(metric, rate.value)} "
            f"({m.answer_poison_asserted.numerator} asserted / "
            f"{m.answer_poison_adopted.numerator} adopted)"
        )
    if metric == "answer_poison_disclosed":
        return _fmt(metric, m.answer_poison_disclosed.value)
    if metric == "reader_overflow":
        return _fmt(metric, m.reader_overflow.value)
    return _cell(metric, r)


def _verdicts(m: AnswerMetrics, probe_set: str) -> str:
    counts = m.verdicts.get(probe_set, {})
    if not counts:
        return "n/a"
    order = [v.value for v in ReplyVerdict]
    return " · ".join(f"{v} {counts[v]}" for v in order if v in counts)


def _answer_table(
    rows: list[tuple[str, AnswerMetrics, dict[str, MetricRange]]], columns: tuple[str, ...]
) -> list[str]:
    header = [c + (" †" if c in ANSWER_EXPECTED_LOSSES else "") for c in columns]
    lines = ["| row | " + " | ".join(header) + " |", "|---|" + "---|" * len(columns)]
    for label, m, ranges in rows:
        cells = [_answer_cell(c, m, ranges.get(c)) for c in columns]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _answer_section(report: DoorReport) -> list[str]:  # noqa: PLR0912 — one table per block
    """The answer family, below the door-text tables and never in their place."""
    pin = report.selection.answer
    if pin is None:
        return []
    lines = ["", "## Answer stage", ""]
    for reader in pin.readers:
        lines.append(
            f"- {reader.role} reader `{reader.provider_model}`"
            + (f" digest `{reader.digest}`" if reader.digest else "")
            + f", temperature {reader.temperature}, max {reader.max_tokens} tokens, "
            f"reasoning {reader.reasoning_effort or 'provider default'}"
            + (f", context {reader.context_tokens:,} tokens" if reader.context_tokens else "")
        )
    lines += [
        f"- prompt v{pin.reader_prompt_version} `{pin.reader_prompt_sha256[:12]}`, reply scorer "
        f"v{pin.reply_scorer_version}, answer strip v{pin.answer_strip_version} "
        f"({', '.join(pin.strip_fields)})",
        "",
        "The reader is a model: these numbers depend on it, and the door-text tables above "
        "remain the primary instrument. Each verdict set sums to its scored replies.",
    ]
    headline = answer_key(HEADLINE_READER, "door")
    rows = [
        (_row_label(m), m.answers[headline], m.answer_ranges.get(headline, {}))
        for m in _rows(report)
        if headline in m.answers
    ]
    if rows:
        lines += ["", "### Headline reader, reading `door`", ""]
        lines += _answer_table(rows, _ANSWER_COLUMNS)
        lines += ["", "#### Verdicts", ""]
        lines.append("| row | " + " | ".join(PROBE_SETS) + " |")
        lines.append("|---|" + "---|" * len(PROBE_SETS))
        for label, m, _ in rows:
            lines.append(f"| {label} | " + " | ".join(_verdicts(m, s) for s in PROBE_SETS) + " |")
    particles = next(
        (m for m in report.memories if m.name == "particles" and m.role == "headline"), None
    )
    if particles is not None:
        diag = [
            (f"`{reading}`", particles.answers[k], particles.answer_ranges.get(k, {}))
            for reading in READINGS
            if (k := answer_key(HEADLINE_READER, reading)) in particles.answers
        ]
        if len(diag) > 1:
            lines += [
                "",
                "### Particles' readings (diagnostic: Particles against itself, never a "
                "comparative row)",
                "",
            ]
            lines += _answer_table(diag, _ANSWER_COLUMNS)
            skipped = {r: m.skipped for r, m, _ in diag if m.skipped}
            if skipped:
                lines.append("")
                lines.append(
                    "Readings not derived (no JSON door or no answer field): "
                    + ", ".join(f"{r} {n}" for r, n in skipped.items())
                )
        if headline in particles.answers:
            pm = particles.answers[headline]
            pairs = []
            for metric, door_metric in DOOR_COUNTERPART.items():
                answer_value = (
                    pm.reader_tokens_in.median
                    if metric == "reader_tokens_in"
                    else getattr(pm, metric).value
                )
                door_range = particles.ranges.get(door_metric)
                pairs.append(
                    f"{metric} {_fmt(metric, answer_value)} beside {door_metric} "
                    f"{_cell(door_metric, door_range)}"
                )
            lines += ["", "Particles, each answer metric beside its door-text counterpart: "]
            lines += [f"- {p}" for p in pairs]
    sensitivity = answer_key(SENSITIVITY_READER, "door")
    srows = [
        (_row_label(m), m.answers[sensitivity], m.answer_ranges.get(sensitivity, {}))
        for m in _rows(report)
        if sensitivity in m.answers
    ]
    if srows:
        lines += ["", "### Sensitivity reader, reading `door` (never a headline)", ""]
        lines += _answer_table(
            srows, (*_ANSWER_COLUMNS[:-2], "reader_overflow", "reader_tokens_in")
        )
    tallies = []
    for memory in report.memories:
        for key, metrics in memory.answers.items():
            if metrics.errors or metrics.cached:
                tallies.append(
                    f"{_row_label(memory)} {key}: {metrics.cached} cached, {metrics.errors} failed"
                )
    if tallies:
        lines += ["", "Reader calls: " + "; ".join(tallies) + "."]
    return lines


#: The rates a compact Particles door moves by rendering, not by ranking or
#: trust: the full record printed the bare structured value.
SHAPE_SENSITIVE = frozenset({"poison_leak", "stale_in_door", "stale_over_current"})

SHAPE_FOOTNOTE = (
    "‡ A Particles row marked ‡ reads the compact `query` result: the hit "
    "list and the answer's composition changed in that version. Most of its "
    "`poison_leak` drop against a full-record row is the structured value no longer "
    "being printed, not a change in ranking or trust."
)


def _compact_door(m: DoorMemoryResult) -> bool:
    """Whether a Particles row read a door shaped other than the full record."""
    shape = m.configuration.get("mcp_query")
    return m.name == "particles" and isinstance(shape, dict) and shape.get("detail") != "full"


def render_door_report(report: DoorReport) -> str:
    """The report as Markdown: disclosures, tables, predictions, notes."""
    s = report.selection
    lines = [f"> {d}" for d in report.disclosures]
    lines += [
        "",
        f"# Rot door arm, tier {s.tier}",
        "",
        f"World `{s.profile}`, seeds {s.seeds}"
        + (
            f" (held out, commitment `{s.seed_commitment[:12]}`)"
            if s.held_out and s.seed_commitment
            else ""
        )
        + f"; generator v{s.generator_version}, scorer v{s.scorer_version}, "
        f"segmenter v{s.door_segmenter_version}; adebench `{s.adebench_sha or 'none'}`; "
        f"Particles {s.particles_version}.",
        "",
        "Each cell is the mean over builds, with the range when there was more than one. "
        "Overlapping ranges are not a ranking. Expected losses for Particles are marked †.",
    ]
    for family, metrics in _FAMILIES:
        header = [m + (" †" if m in EXPECTED_LOSSES else "") for m in metrics]
        lines += ["", f"## {family}", "", "| memory | " + " | ".join(header) + " |"]
        lines.append("|---|" + "---|" * len(metrics))
        footnote = False
        for m in _rows(report):
            cells = [_cell(metric, m.ranges.get(metric)) for metric in metrics]
            mark = _compact_door(m) and bool(SHAPE_SENSITIVE.intersection(metrics))
            footnote = footnote or mark
            label = _row_label(m) + (" ‡" if mark else "")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        # printed by the renderer, not left to the writer.
        if footnote:
            lines += ["", SHAPE_FOOTNOTE]
    lines += _answer_section(report)
    lines += ["", "## Memories", ""]
    for m in report.memories:
        failed = [b for b in m.builds if not b.probes]
        cached = sum(1 for b in m.builds if b.cached)
        extra = f", {len(failed)} failed" if failed else ""
        extra += f", {cached} from cache" if cached else ""
        review = f"; maintainer review: {m.maintainer_review}" if m.maintainer_review else ""
        lines.append(
            f"- **{_row_label(m)}**: {m.adapter} {m.version}, model {m.model or 'none'}, "
            f"door `{m.builds[0].probes[0].door if m.builds and m.builds[0].probes else '?'}`, "
            f"{len(m.builds)} build(s){extra}{review}"
        )
        for b in m.builds:
            lines += [f"  - seed {b.seed} build {b.build}: {n}" for n in b.notes]
    if report.prediction_results:
        lines += ["", "## Pre-registered predictions (Particles against the median competitor)", ""]
        lines.append("| metric | predicted | observed | verdict |")
        lines.append("|---|---|---|---|")
        for p in report.prediction_results:
            lines.append(f"| {p.metric} | {p.predicted} | {p.observed} | {p.verdict} |")
    for prior in report.prior_scorings:
        lines += [
            "",
            f"## Headline under scorer v{prior.scorer_version}, segmenter "
            f"v{prior.door_segmenter_version}",
            "",
        ]
        keys = [k for k in METRIC_DIRECTIONS if k in _RATES]
        lines.append("| memory | " + " | ".join(keys) + " |")
        lines.append("|---|" + "---|" * len(keys))
        for name, figures in prior.figures.items():
            lines.append(f"| {name} | " + " | ".join(_fmt(k, figures.get(k)) for k in keys) + " |")
        if prior.answer_figures:
            akeys = [k for k in ANSWER_METRIC_DIRECTIONS if k in _ANSWER_RATES]
            lines += [
                "",
                f"Answer family under reply scorer v{prior.reply_scorer_version}:",
                "",
                "| memory | " + " | ".join(akeys) + " |",
                "|---|" + "---|" * len(akeys),
            ]
            for name, figures in prior.answer_figures.items():
                lines.append(
                    f"| {name} | " + " | ".join(_fmt(k, figures.get(k)) for k in akeys) + " |"
                )
    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]
    return "\n".join(lines) + "\n"
