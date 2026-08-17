"""structure verb — backfill the derived S-P-O annotation."""

from __future__ import annotations

import json

import typer

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.api.cli._progress import progress_line
from particles.db import session_scope


@app.command("structure")
def structure_cmd(
    limit: int | None = typer.Option(
        None,
        help="Max particles to annotate this run (default: structured_claim."
        "backfill_batch_limit). Use 0 for the whole backlog in one run — safe, "
        "because the pass commits as it goes.",
    ),
    rate_limit_per_minute: int | None = typer.Option(
        None,
        help="Max structurizer calls per minute (default: structured_claim."
        "backfill_rate_limit_per_minute); 0 disables the delay.",
    ),
    structurizer_version: str | None = typer.Option(
        None,
        help="Regenerate annotations stamped with a version OTHER than this one, "
        "instead of annotating unannotated particles (mirrors "
        "`reindex --extractor-version`).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report the whole backlog (not the batch cap), the runs it implies, "
        "and current coverage; write nothing.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print per-particle progress."),
    debug: bool = typer.Option(False, "--debug", help="Debug logging."),
) -> None:
    """Annotate particles with a structured (subject-predicate-object) claim.

    The annotation is derived from `content` and is never an assertion: this
    verb cannot change a claim, its confidence, or its provenance. Particles
    extracted since landed are annotated at extraction time for free;
    this pass is for the ones that predate it, and it pays one LLM call each —
    hence the rate limit and the resumable batch cap.

    Particles whose prose has no honest triple are *skipped*, permanently and
    without complaint. Absence of an annotation is a legal state.
    """
    configure_logging(verbose, debug)
    run(
        _structure(
            limit=limit,
            rate_limit_per_minute=rate_limit_per_minute,
            structurizer_version=structurizer_version,
            dry_run=dry_run,
            verbose=verbose,
        )
    )


async def _structure(
    *,
    limit: int | None,
    rate_limit_per_minute: int | None,
    structurizer_version: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    from particles.api.client import get_backend

    # Local-only, on the `memory consolidate` precedent: the pass walks one
    # store's particles and pays an LLM call each. There is no HTTP analogue
    # and inventing one would put a long, rate-limited write loop behind a
    # request/response boundary.
    if get_backend().remote:
        typer.echo(
            "Error: `particles structure` annotates one local store per invocation; "
            "run it on the machine that holds the store.",
            err=True,
        )
        raise typer.Exit(2)

    # Deferred import: the operation pulls the LLM + store stack, and tests
    # patch it at call time (tests/AGENTS.md § Mocking strategy).
    from particles.operations.structure import backfill_structured_claims

    progress = progress_line if verbose else None
    async with session_scope() as session:
        summary = await backfill_structured_claims(
            session,
            limit=limit,
            rate_limit_per_minute=rate_limit_per_minute,
            structurizer_version=structurizer_version,
            dry_run=dry_run,
            progress=progress,
        )
    typer.echo(json.dumps(summary, indent=2))
