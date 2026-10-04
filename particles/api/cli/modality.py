# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""modality verb — regenerate stale adjudicability defaults."""

from __future__ import annotations

import json

import typer

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.api.cli._progress import progress_line
from particles.db import session_scope
from particles.llm.usage import render_usage_line
from particles.operations.llm_spend import MeteredExtractRun


@app.command("modality")
def modality_cmd(
    limit: int | None = typer.Option(
        None,
        help="Max claims to reclassify this run (default: modality_regeneration."
        "batch_limit). Use 0 for the whole backlog in one run; that is safe, because "
        "the pass commits as it goes.",
    ),
    rate_limit_per_minute: int | None = typer.Option(
        None,
        help="Max classifier calls per minute (default: modality_regeneration."
        "rate_limit_per_minute); 0 disables the delay.",
    ),
    include_unclassified: bool = typer.Option(
        False,
        "--include-unclassified",
        help="Include claims no classifier ever ran on: claims extracted with "
        "classification off, claims from structured extractors, and direct assertions.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report the whole backlog and the stamp census by state and classifier; "
        "write nothing.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print per-claim progress."),
    debug: bool = typer.Option(False, "--debug", help="Debug logging."),
) -> None:
    """Reclassify claims whose adjudicability default came from an outdated rule.

    Each claim's `assertion_modality` is stamped with the classifier rule that
    set it. When that rule has since changed, the stamp is stale, and `particles
    lint` reports it as MODALITY_CLASSIFIER_STALE. This verb runs today's rule
    over each stale claim, one LLM call each, and rewrites the value and its
    stamp. It never changes a claim's content, confidence, or provenance.

    It never makes a claim adjudicable on its own: a verdict that would flip a
    claim to FALSIFIABLE is queued instead, and `particles lint` lists it as
    MODALITY_GRANT_PENDING for `particles particle reclassify`. Verdicts that
    withdraw adjudication, or confirm the stored value, are written. An
    operator verdict is never overwritten, even one recorded during the run.
    Journal-extractor claims are skipped, because the general rule would
    replace the journal prompt's better-informed verdict; re-extraction
    reclassifies them. A reply with no valid verdict leaves the claim as it
    was. A claim whose value changed is re-paired by the next `particles
    memory consolidate` run, under its new default.
    """
    configure_logging(verbose, debug)
    run(
        _modality(
            limit=limit,
            rate_limit_per_minute=rate_limit_per_minute,
            include_unclassified=include_unclassified,
            dry_run=dry_run,
            verbose=verbose,
        )
    )


async def _modality(
    *,
    limit: int | None,
    rate_limit_per_minute: int | None,
    include_unclassified: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    from particles.api.client import get_backend

    # Local-only, as `structure` is: a long, rate-limited write loop over one
    # store has no sensible HTTP analogue.
    if get_backend().remote:
        typer.echo(
            "Error: `particles modality` reclassifies one local store per invocation; "
            "run it on the machine that holds the store.",
            err=True,
        )
        raise typer.Exit(2)

    # Deferred import: the operation pulls the LLM + store stack, and tests
    # patch it at call time (tests/AGENTS.md § Mocking strategy).
    from particles.operations.modality import regenerate_modality

    progress = progress_line if verbose else None

    async def _run() -> dict[str, object]:
        async with session_scope(write=not dry_run) as session:
            summary = await regenerate_modality(
                session,
                limit=limit,
                rate_limit_per_minute=rate_limit_per_minute,
                include_unclassified=include_unclassified,
                dry_run=dry_run,
                progress=progress,
            )
            return summary.as_dict()

    if dry_run:
        typer.echo(json.dumps(await _run(), indent=2))
        return
    # one classifier call per claim, metered as extract is metered.
    meter = MeteredExtractRun(actor="cli:modality", route="modality")
    try:
        async with meter:
            summary = await _run()
    finally:
        if meter.llm_usage is not None:
            typer.echo(render_usage_line(meter.llm_usage), err=True)
    typer.echo(json.dumps(summary, indent=2))
