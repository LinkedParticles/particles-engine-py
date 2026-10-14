# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""validity verb — the deterministic prose-date default for valid_from."""

from __future__ import annotations

import json

import typer

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.api.cli._progress import progress_line
from particles.db import session_scope


@app.command("validity")
def validity_cmd(
    limit: int | None = typer.Option(
        None,
        help="Max claims to date this run. Default: the whole scope, which is safe, "
        "because the pass commits as it goes.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report the scope, what the cue rules would set, and the coverage census; "
        "write nothing.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print per-claim progress."),
    debug: bool = typer.Option(False, "--debug", help="Debug logging."),
) -> None:
    """Date claims from the absolute dates already in their prose.

    Each ACTIVE claim with no `valid_from` is read for one anchoring date
    behind a recognised cue (`on <date>`, `since <date>`, `as of <date>`,
    `in <year>`), and the cue fixes the kind: a moment (EVENT) or a span
    (STATE). The pass makes no model call and never changes a claim's
    content, confidence, provenance, status or `valid_until`. A value the
    extractor stated or an operator set is never overwritten; a default from
    an earlier rule is regenerated. The window route (`particles query
    --during`) reads what this writes; an undated claim is disclosed there,
    never placed.
    """
    configure_logging(verbose, debug)
    run(_validity(limit=limit, dry_run=dry_run, verbose=verbose))


async def _validity(*, limit: int | None, dry_run: bool, verbose: bool) -> None:
    from particles.api.client import get_backend

    # Local-only, as `modality` is: a write loop over one store has no
    # sensible HTTP analogue.
    if get_backend().remote:
        typer.echo(
            "Error: `particles validity` dates one local store per invocation; "
            "run it on the machine that holds the store.",
            err=True,
        )
        raise typer.Exit(2)

    # Deferred import: the operation pulls the store stack, and tests patch it
    # at call time (tests/AGENTS.md § Mocking strategy).
    from particles.operations.validity import regenerate_validity

    progress = progress_line if verbose else None
    async with session_scope(write=not dry_run) as session:
        summary = await regenerate_validity(
            session, limit=limit, dry_run=dry_run, progress=progress
        )
    typer.echo(json.dumps(summary.as_dict(), indent=2))
