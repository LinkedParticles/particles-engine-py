# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""reconcile verb — cross-entry document-supersession sweep.

Runs the §6.6 rung-1.5 document-supersession prior over already-extracted
ACTIVE particles, demoting superseded claims that the intra-entry extract path
never reconciles. ``--updates`` selects the second mode: the
same-subject update sweep, which retires a value a later claim from the same
source lineage replaced. Further reconciliation modes (corpus-wide
corroboration / contradiction) extend this verb rather than adding
new ones.
"""

from __future__ import annotations

import json

import typer

from particles.api.cli import app, run
from particles.api.cli._output import PROGRESS_OPTION, QUIET_OPTION, configure_output
from particles.api.cli._progress import progress_line
from particles.db import session_scope


@app.command("reconcile")
def reconcile_cmd(
    updates: bool = typer.Option(
        False,
        "--updates",
        help="Run the same-subject update sweep instead of the "
        "document-supersession sweep: retire values a later claim from the same "
        "source lineage replaced.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would be demoted without mutating the store.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print scope size and per-demotion progress.",
    ),
    quiet: bool = QUIET_OPTION,
    progress: bool | None = PROGRESS_OPTION,
) -> None:
    """Demote superseded claims across corpus entries (supersession sweeps)."""
    configure_output(verbose, quiet=quiet, progress=progress)
    summary = run(_reconcile(dry_run, verbose, updates=updates))
    typer.echo(json.dumps(summary, indent=2))


async def _reconcile(dry_run: bool, verbose: bool, *, updates: bool = False) -> dict[str, object]:
    from particles.operations.reconcile import reconcile_supersession, reconcile_updates

    progress = progress_line if verbose else None
    async with session_scope() as session:
        if updates:
            return await reconcile_updates(session, dry_run=dry_run, progress=progress)
        return await reconcile_supersession(session, dry_run=dry_run, progress=progress)
