# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plain-text tables for the two reports."""

from __future__ import annotations

from particles.benchmark.update_gate.checks import case_rows
from particles.benchmark.update_gate.schema import CaseRun, TripVsMoveReport, UpdateGateReport
from particles.benchmark.update_gate.scoring import GATES
from particles.llm.usage import format_usd


def _state(run: CaseRun) -> str:
    if run.invalid:
        return f"INVALID: {run.invalid}"
    if run.incomplete:
        return f"INCOMPLETE: {run.incomplete}"
    return "PASS" if run.passed else "FAIL"


def _money(value: float | None) -> str:
    return "unpriced" if value is None else f"US{format_usd(value)}"


def _spend_line(report: TripVsMoveReport) -> str:
    if report.spend.estimate_source == "none":
        return "Spend: nothing (scripted arm)."
    return (
        f"Spend: {_money(report.spend.cost_usd)} at list price (estimated "
        f"{_money(report.spend.estimate_usd)} from the {report.spend.estimate_source})."
    )


def render_case(report: TripVsMoveReport, *, rows: bool = True) -> str:
    """The trip-versus-move report as text: one block per run, then the verdict."""
    out = [
        f"Trip versus move: {report.model}"
        + (
            " (mixed: "
            + ", ".join(f"{p}={m}" for p, m in sorted(report.purposes.items()) if m != report.model)
            + ")"
            if report.mixed
            else ""
        ),
        f"arm {report.arm}, same_slot_admits {str(report.same_slot_admits).lower()}, "
        f"checks v{report.checks_version}, SDK {report.sdk_version}",
        "",
    ]
    for run in report.runs:
        cost = (
            ""
            if run.usage is None or report.arm == "scripted"
            else f"  {_money(run.usage.cost_usd)}"
        )
        out.append(f"{run.variant} run {run.index}: {_state(run)}{cost}")
        for check, offending in run.checks.items():
            if (
                check != "pass"
                and isinstance(offending, list)
                and offending
                and check != "residence_claims"
            ):
                out.append(f"  {check}:")
                out.extend(f"    {text}" for text in offending)
        if run.refused_calls:
            out.append(f"  refused calls: {run.refused_calls}")
        if rows:
            for row in case_rows([r.model_dump() for r in run.rows]):
                mark = "  (restatement)" if row["restatement"] else ""
                out.append(
                    f"    {row['status']:18} {row['reason'] or '':26} {row['content']}{mark}"
                )
    v = report.verdict
    bar = (
        "not stated (run counts differ from the record's 3 and 2)"
        if v.bar_met is None
        else ("met" if v.bar_met else "not met")
    )
    out += [
        "",
        f"Move case: {v.move_passed} of {v.move_runs} passed. "
        f"Trip alone: {v.trip_alone_passed} of {v.trip_alone_runs} passed. Bar: {bar}.",
        _spend_line(report),
    ]
    out += [f"Note: {n}" for n in report.notes]
    return "\n".join(out)


def render_pairs(report: UpdateGateReport) -> str:
    """The update-gate report as text: each set's gates, readings and unbarred rows."""
    out = [
        f"Update gate: {report.model}, {report.samples_per_pair} samples per pair, "
        f"SDK {report.sdk_version}",
        "",
    ]
    for name, result in report.sets.items():
        bars = result.bars
        out.append(
            f"{name}: bar {'met' if bars['bar'] else 'not met'} (production gate), "
            f"regression {'none' if bars['regression'] else 'found'}"
        )
        for gate in GATES:
            g = bars["gates"][gate]
            misses = f", missed {', '.join(g['positive_misses'])}" if g["positive_misses"] else ""
            rows = (
                f", controls retired {', '.join(g['control_rows_retired'])}"
                if g["control_rows_retired"]
                else ""
            )
            out.append(
                f"  {gate:13} updates retired {g['positives_retired']} of {g['positives']}, "
                f"control retirements {g['control_retirements']} of {g['control_samples']}"
                f"{misses}{rows}"
            )
        if bars["controls_added_by_production"]:
            out.append(
                "  controls the production gate retires that before-1.187 keeps: "
                + ", ".join(bars["controls_added_by_production"])
            )
        for row_id, counts in bars["reported"].items():
            tally = ", ".join(f"{g} {counts[g]}" for g in GATES)
            out.append(
                f"  reported, never barred: {row_id} ({counts['class']}): {tally} of {counts['of']}"
            )
    out += [
        "",
        f"Spend: {_money(report.spend.cost_usd)} at list price.",
    ]
    out += [f"Note: {n}" for n in report.notes]
    return "\n".join(out)
