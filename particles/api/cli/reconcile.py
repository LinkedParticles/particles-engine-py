"""reconcile verb — cross-entry document-supersession sweep.

Runs the §6.6 rung-1.5 document-supersession prior over already-extracted
ACTIVE particles, demoting superseded claims that the intra-entry extract path
never reconciles. v1 covers the document-supersession mode only; future
reconciliation modes (corpus-wide corroboration / contradiction)
extend this verb rather than adding new ones.
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
    """Demote superseded claims across corpus entries (document-supersession sweep)."""
    configure_output(verbose, quiet=quiet, progress=progress)
    summary = run(_reconcile(dry_run, verbose))
    typer.echo(json.dumps(summary, indent=2))


async def _reconcile(dry_run: bool, verbose: bool) -> dict[str, object]:
    from particles.operations.reconcile import reconcile_supersession

    progress = progress_line if verbose else None
    async with session_scope() as session:
        return await reconcile_supersession(session, dry_run=dry_run, progress=progress)
