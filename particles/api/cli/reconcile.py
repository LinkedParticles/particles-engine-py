# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""reconcile verb — cross-entry document-supersession sweep.

Runs the §6.6 rung-1.5 document-supersession prior over already-extracted
ACTIVE particles, demoting superseded claims that the intra-entry extract path
never reconciles. ``--updates`` selects the second mode: the
same-subject update sweep, which retires a value a later claim from the same
source lineage replaced, then the paraphrases pass, which retires the
other phrasings of each value an update replaced. ``--dependents`` selects the
third: claims that relied on a state an update retired are restated,
anchored to that state. Further reconciliation modes (corpus-wide
corroboration / contradiction) extend this verb rather than adding
new ones.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import typer
from sqlalchemy.ext.asyncio import AsyncSession

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
        "document-supersession sweep: retire a value when a later claim from the "
        "same source lineage gives a new value for the same attribute. The "
        "paraphrases pass follows it, retiring the other phrasings of "
        "each replaced value, from the nightly cycle's position without moving it.",
    ),
    dependents: bool = typer.Option(
        False,
        "--dependents",
        help="Run the re-anchor pass: restate claims that relied on a "
        "state an update retired, such as a place described as near the user's "
        "old flat. Resumes from the nightly cycle's position without moving it.",
    ),
    scope: str = typer.Option(
        "cursor",
        "--scope",
        help="With --dependents: 'cursor' examines update retirements after the "
        "nightly position; 'store' examines every one.",
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
    if updates and dependents:
        raise typer.BadParameter("Choose one of --updates and --dependents.")
    if scope not in ("cursor", "store"):
        raise typer.BadParameter("--scope must be 'cursor' or 'store'.")
    summary = run(_reconcile(dry_run, verbose, updates=updates, dependents=dependents, scope=scope))
    typer.echo(json.dumps(summary, indent=2))


async def _reconcile(
    dry_run: bool,
    verbose: bool,
    *,
    updates: bool = False,
    dependents: bool = False,
    scope: str = "cursor",
) -> dict[str, object]:
    from particles.operations.reconcile import reconcile_supersession, reconcile_updates

    progress = progress_line if verbose else None
    async with session_scope() as session:
        if dependents:
            return await _reanchor(session, dry_run=dry_run, scope=scope, progress=progress)
        if updates:
            summary = await reconcile_updates(session, dry_run=dry_run, progress=progress)
            summary["paraphrases"] = await _paraphrases(session, dry_run=dry_run, progress=progress)
            return summary
        return await reconcile_supersession(session, dry_run=dry_run, progress=progress)


async def _paraphrases(
    session: AsyncSession,
    *,
    dry_run: bool,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    """Run the paraphrases pass after the sweep. The nightly position is read, never moved.

    A dry run gathers its candidates and asks nothing, since the sweep's dry
    run retired nothing for it to follow.
    """
    from particles.operations.paraphrases import run_paraphrases
    from particles.operations.reanchor import prior_cursor

    cursor = await prior_cursor(session, "memory-consolidate", "paraphrases")
    report = await run_paraphrases(session, cursor=cursor, dry_run=dry_run, progress=progress)
    if not dry_run:
        await session.commit()
    return report.model_dump(mode="json")


async def _reanchor(
    session: AsyncSession,
    *,
    dry_run: bool,
    scope: str,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    """Run the re-anchor pass by hand. The nightly position is read, never moved."""
    from particles.operations.reanchor import prior_cursor, run_reanchor

    cursor = await prior_cursor(session, "memory-consolidate")
    report = await run_reanchor(
        session,
        cursor=cursor,
        scope="store" if scope == "store" else "cursor",
        dry_run=dry_run,
        actor="reconcile",
        progress=progress,
    )
    if not dry_run:
        await session.commit()
    return report.model_dump(mode="json")
