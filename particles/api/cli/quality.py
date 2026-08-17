"""quality verb — instant extraction-quality dashboard (no LLM calls)."""

from __future__ import annotations

import typer

from particles.api.cli import app, run
from particles.api.client import get_backend


@app.command("quality")
def quality_cmd() -> None:
    """Show the extraction quality dashboard.

    \b
    Displays calibration source distribution, corpus snapshot status,
    and subject coverage metrics. No LLM calls — instant read from the DB.
    For full structural and semantic diagnostics use: particles lint
    """
    report = run(get_backend().quality())

    ts = report.generated_at.strftime("%Y-%m-%d %H:%M")
    typer.echo(f"Extraction Quality Dashboard  ({ts})\n")

    typer.echo("Particles")
    typer.echo(f"  Active:          {report.active_particles:>6,}")
    if report.inconsistency_particles:
        typer.echo(f"  Inconsistency:   {report.inconsistency_particles:>6,}  (needs review)")
    typer.echo("  Calibration:")
    for bucket in report.calibration:
        flag = (
            "  ⚠ above 50% threshold"
            if (bucket.source == "EXTRACTOR_DIRECT" and bucket.fraction > 0.5)
            else ""
        )
        typer.echo(f"    {bucket.source:<26} {bucket.count:>6,}  ({bucket.fraction:.1%}){flag}")
    if not report.calibration:
        typer.echo("    (no particles)")

    typer.echo("\nCorpus")
    typer.echo(f"  Entries:         {report.total_entries:>6,}")
    typer.echo(f"  Complete:        {report.snapshots_complete:>6,}")
    if report.snapshots_pending:
        typer.echo(f"  Pending:         {report.snapshots_pending:>6,}")
    if report.snapshots_failed:
        typer.echo(f"  Failed:          {report.snapshots_failed:>6,}  (run: particles reindex)")
    if report.snapshots_in_progress:
        typer.echo(f"  In progress:     {report.snapshots_in_progress:>6,}")

    typer.echo("\nSubjects")
    typer.echo(f"  Total:           {report.total_subjects:>6,}")
    if report.subjects_without_particles:
        typer.echo(
            f"  Without particles:{report.subjects_without_particles:>5,}  (run: particles lint)"
        )

    # coverage is reported, never enforced — an un-annotated
    # particle is in a legal permanent state, so there is no warning flag here.
    typer.echo("\nStructured claims")
    annotated = report.structured_claims
    fraction = annotated / report.active_particles if report.active_particles else 0.0
    typer.echo(f"  Annotated:       {annotated:>6,}  ({fraction:.1%} of ACTIVE)")
    for stamp, count in sorted(
        report.structured_claims_by_structurizer.items(), key=lambda x: -x[1]
    ):
        typer.echo(f"    {stamp:<26} {count:>6,}")
    if not annotated:
        typer.echo("    (none yet — run: particles structure)")
