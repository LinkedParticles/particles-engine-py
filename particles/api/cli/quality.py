# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""quality verb — instant extraction-quality dashboard (no LLM calls)."""

from __future__ import annotations

from particles.api.cli import app, run
from particles.api.cli._terminal import terminal_echo, terminal_field
from particles.api.client import get_backend
from particles.llm.usage import render_store_spend_line
from particles.operations.curation.precision import render_precision_lines


@app.command("quality")
def quality_cmd() -> None:
    """Show the extraction quality dashboard.

    \b
    Displays calibration source distribution, corpus snapshot status,
    subject coverage metrics, recorded LLM spend, and the curation queue's
    precision over the default window. No LLM calls, just an instant read
    from the DB.
    For full structural and semantic diagnostics use: particles lint
    """
    report = run(get_backend().quality())

    ts = report.generated_at.strftime("%Y-%m-%d %H:%M")
    terminal_echo(f"Extraction Quality Dashboard  ({ts})\n")

    terminal_echo("Particles")
    terminal_echo(f"  Active:          {report.active_particles:>6,}")
    if report.inconsistency_particles:
        terminal_echo(f"  Inconsistency:   {report.inconsistency_particles:>6,}  (needs review)")
    terminal_echo("  Calibration:")
    for bucket in report.calibration:
        flag = (
            "  ⚠ above 50% threshold"
            if (bucket.source == "EXTRACTOR_DIRECT" and bucket.fraction > 0.5)
            else ""
        )
        terminal_echo(
            f"    {terminal_field(bucket.source):<26} {bucket.count:>6,}  "
            f"({bucket.fraction:.1%}){flag}"
        )
    if not report.calibration:
        terminal_echo("    (no particles)")

    terminal_echo("\nCorpus")
    terminal_echo(f"  Entries:         {report.total_entries:>6,}")
    terminal_echo(f"  Complete:        {report.snapshots_complete:>6,}")
    if report.snapshots_pending:
        held = (
            f"  ({report.snapshots_held_refused:,} held: refused, see particles curate)"
            if report.snapshots_held_refused
            else ""
        )
        terminal_echo(f"  Pending:         {report.snapshots_pending:>6,}{held}")
    if report.snapshots_failed:
        terminal_echo(f"  Failed:          {report.snapshots_failed:>6,}  (run: particles reindex)")
    if report.snapshots_in_progress:
        terminal_echo(f"  In progress:     {report.snapshots_in_progress:>6,}")

    terminal_echo("\nSubjects")
    terminal_echo(f"  Total:           {report.total_subjects:>6,}")
    if report.subjects_without_particles:
        terminal_echo(
            f"  Without particles:{report.subjects_without_particles:>5,}  (run: particles lint)"
        )

    # coverage is reported, never enforced — an un-annotated
    # particle is in a legal permanent state, so there is no warning flag here.
    terminal_echo("\nStructured claims")
    annotated = report.structured_claims
    fraction = annotated / report.active_particles if report.active_particles else 0.0
    terminal_echo(f"  Annotated:       {annotated:>6,}  ({fraction:.1%} of ACTIVE)")
    for stamp, count in sorted(
        report.structured_claims_by_structurizer.items(), key=lambda x: -x[1]
    ):
        terminal_echo(f"    {terminal_field(stamp):<26} {count:>6,}")
    if not annotated:
        terminal_echo("    (none yet — run: particles structure)")

    # list price over recorded token counts, since the first
    # recorded run; absent until a run has recorded usage.
    if report.llm_spend is not None:
        terminal_echo(f"\n{render_store_spend_line(report.llm_spend)}")

    # how often the curation queue was right over the default window
    # (curation.precision_window_days); absent until the store has a collection
    # or a gesture. `particles curate --precision` prints the full report.
    if report.curation_precision is not None:
        terminal_echo("")
        for line in render_precision_lines(report.curation_precision, meanings=False):
            terminal_echo(line)
        terminal_echo("  Full report with what each outcome means: particles curate --precision")
