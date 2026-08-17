"""lint verb — structural + optional semantic checks over the particle store."""

from __future__ import annotations

import json

import typer

from particles.api.cli import app, run
from particles.api.client import get_backend
from particles.core.schema import LintReport

_DEFAULT_VERBOSE_CAP_PER_CATEGORY = 50


@app.command("lint")
def lint_cmd(
    fix: bool = typer.Option(
        False,
        help=(
            "Apply auto-fixable status transitions "
            "(STALENESS, RETRACTION_CASCADE, CORPUS_LINK_INTEGRITY)."
        ),
    ),
    semantic: bool = typer.Option(False, help="Run LLM-assisted semantic checks (slower)"),
    output_format: str = typer.Option("markdown", help="Output format: markdown or json"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all findings in full"),
    category: str | None = typer.Option(
        None,
        "--category",
        help=(
            "Restrict --verbose output to one finding_type "
            "(e.g. STALENESS, GRANULARITY_VIOLATION_CANDIDATE)"
        ),
    ),
    limit_per_category: int = typer.Option(
        _DEFAULT_VERBOSE_CAP_PER_CATEGORY,
        "--limit-per-category",
        help=(
            "Cap verbose findings per category to keep output manageable; "
            "remainder is summarised. 0 disables the cap."
        ),
    ),
    low_coverage_threshold: int = typer.Option(
        3, help="Subjects with fewer ACTIVE CLAIM particles are flagged"
    ),
) -> None:
    """Run lint checks over the particle store."""
    report = run(
        get_backend().lint(
            fix=fix, semantic=semantic, low_coverage_threshold=low_coverage_threshold
        )
    )

    if category is not None:
        if category not in report.summary:
            available = ", ".join(sorted(report.summary.keys())) or "(none)"
            typer.echo(
                f"No findings with finding_type={category!r}. Categories in this run: {available}",
                err=True,
            )
            raise typer.Exit(code=1)
        report = _filter_report_by_category(report, category)

    if output_format == "json":
        if verbose:
            typer.echo(report.model_dump_json(indent=2))
        else:
            typer.echo(_lint_summary_json(report))
        return

    if verbose:
        from particles.render.markdown import render_lint_report

        typer.echo(
            render_lint_report(
                _cap_findings_for_render(report, limit_per_category),
            )
        )
    else:
        typer.echo(_lint_summary_text(report, fix_applied=fix))


def _filter_report_by_category(report: LintReport, category: str) -> LintReport:
    """Return a LintReport restricted to one finding_type.

    ``summary`` keeps the full counts so the operator still sees the global
    shape; ``findings`` is filtered.
    """
    return LintReport(
        run_at=report.run_at,
        findings=[f for f in report.findings if f.finding_type == category],
        summary=report.summary,
        fixed_counts=report.fixed_counts,
    )


def _cap_findings_for_render(report: LintReport, limit_per_category: int) -> LintReport:
    """Truncate findings per category and append a synthetic INFO note when capped.

    Markdown rendering of 100k lines is hostile; this caps the rendered list
    while preserving the totals in ``summary`` and leaving a breadcrumb for
    how to drill deeper.
    """
    if limit_per_category <= 0:
        return report

    from particles.core.schema import LintFinding

    seen: dict[str, int] = {}
    kept: list[LintFinding] = []
    dropped: dict[str, int] = {}
    for f in report.findings:
        ft = f.finding_type
        seen[ft] = seen.get(ft, 0) + 1
        if seen[ft] <= limit_per_category:
            kept.append(f)
        else:
            dropped[ft] = dropped.get(ft, 0) + 1

    if not dropped:
        return report

    for ft, n in sorted(dropped.items()):
        kept.append(
            LintFinding(
                finding_type=ft,
                severity="INFO",
                detail=(
                    f"…and {n} more {ft} finding(s) suppressed by "
                    f"--limit-per-category={limit_per_category}. "
                    f"Re-run with --category {ft} --limit-per-category 0 to see all."
                ),
                recommended_action=None,
            )
        )
    return LintReport(
        run_at=report.run_at,
        findings=kept,
        summary=report.summary,
        fixed_counts=report.fixed_counts,
    )


def _lint_summary_text(report: LintReport, fix_applied: bool) -> str:
    fix_line = _render_fix_line(report, fix_applied)
    if not report.findings:
        head = f"✓ Lint clean  ({report.run_at.strftime('%Y-%m-%d %H:%M')} UTC)"
        return head if fix_line is None else f"{head}\n{fix_line}"
    errors = sum(1 for f in report.findings if f.severity == "ERROR")
    warnings = sum(1 for f in report.findings if f.severity == "WARNING")
    info = sum(1 for f in report.findings if f.severity == "INFO")
    parts = []
    if errors:
        parts.append(f"{errors} error{'s' if errors != 1 else ''}")
    if warnings:
        parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
    if info:
        parts.append(f"{info} info")
    summary_line = ", ".join(parts)
    lines = [f"✗ Lint: {summary_line}  (run with --verbose for details)"]
    if fix_line is not None:
        lines.append(fix_line)
    for finding_type, count in sorted(report.summary.items()):
        severity = next(
            (f.severity for f in report.findings if f.finding_type == finding_type), "INFO"
        )
        icon = "✗" if severity == "ERROR" else "⚠" if severity == "WARNING" else "·"
        lines.append(f"  {icon} {finding_type}: {count}")
    return "\n".join(lines)


def _render_fix_line(report: LintReport, fix_applied: bool) -> str | None:
    """One-line summary of what --fix did. None when fix was disabled."""
    if not fix_applied:
        return None
    from particles.core.schema import FIX_CAPABLE_CATEGORIES

    total = sum(report.fixed_counts.values())
    if total == 0:
        considered = ", ".join(FIX_CAPABLE_CATEGORIES)
        return f"  Auto-fixed: 0  (categories considered: {considered})"
    per_cat = ", ".join(f"{cat}: {n}" for cat, n in sorted(report.fixed_counts.items()) if n > 0)
    return f"  Auto-fixed: {total}  ({per_cat})"


def _lint_summary_json(report: LintReport) -> str:
    return json.dumps(
        {
            "run_at": report.run_at.isoformat(),
            "summary": report.summary,
            "fixed_counts": report.fixed_counts,
            "error_count": sum(1 for f in report.findings if f.severity == "ERROR"),
            "warning_count": sum(1 for f in report.findings if f.severity == "WARNING"),
        },
        indent=2,
    )
