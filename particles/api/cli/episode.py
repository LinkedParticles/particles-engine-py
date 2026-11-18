# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""episode verb — read the episode question over stored transcript claims (§4)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.api.cli._progress import progress_line
from particles.db import session_scope
from particles.llm.usage import render_usage_line
from particles.operations.llm_spend import MeteredExtractRun


@app.command("episode")
def episode_cmd(
    scope: str = typer.Option(
        "cards",
        help="Which claims to read: 'cards' (the sides of the newest curation "
        "collection's pair cards, the claims a fold acts on) or 'all' (every "
        "transcript-only active claim).",
    ),
    limit: int | None = typer.Option(
        None,
        help="Max claims to read this run (default: curation.episode.batch_limit). "
        "Use 0 for the whole backlog; the pass commits as it goes.",
    ),
    rate_limit_per_minute: int | None = typer.Option(
        None,
        help="Max reader calls per minute (default: curation.episode."
        "rate_limit_per_minute); 0 disables the delay.",
    ),
    restamp: bool = typer.Option(
        False,
        "--restamp",
        help="Rewrite every reading but an operator's under the current regeneration "
        "stamp, including readings under an admitted stamp.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report the backlog, the reader calls it needs, their list-price estimate "
        "and the stored readings by stamp; write nothing and call no model.",
    ),
    sheet: Path | None = typer.Option(
        None,
        "--sheet",
        help="Write the labelling sheet (JSON Lines) to this path instead of reading: "
        "per stratum and stamp combination, a sample of the pair cards it would "
        "withhold. Calls no model and writes nothing to the store.",
    ),
    sheet_size: int | None = typer.Option(
        None,
        "--sheet-size",
        help="Cards per stratum and stamp combination on the sheet "
        "(default: curation.episode.sheet_size).",
    ),
    seed: int = typer.Option(20261108, "--seed", help="The sheet's sampling seed."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print per-call progress."),
    debug: bool = typer.Option(False, "--debug", help="Debug logging."),
) -> None:
    """Read whether transcript claims report only their session's moment.

    The episode reading asks of each claim whose every source is a session
    transcript whether it is true only of the moment its session reports (a
    state then, or an act performed in it) or holds beyond the session. The
    extractor answers it for new claims; this verb answers it for stored ones,
    several claims of one transcript per model call, each beside its source
    passage, and stores the answer with the stamp of the prompt that gave it.
    It never changes a claim's content, confidence, status or provenance, and
    never overwrites an operator's reading.

    No reading withholds a curation card yet: each curation build only counts,
    in shadow, the pair cards a reading would fold into one card per session.
    `--sheet` writes the sample the owner labels to admit a reading's stamp.
    """
    configure_logging(verbose, debug)
    run(
        _episode(
            scope=scope,
            limit=limit,
            rate_limit_per_minute=rate_limit_per_minute,
            restamp=restamp,
            dry_run=dry_run,
            sheet=sheet,
            sheet_size=sheet_size,
            seed=seed,
            verbose=verbose,
        )
    )


def _estimate_line(label: str, estimate: object) -> str:
    """One stderr line pricing a set of reader calls at list price (an upper bound)."""
    if not isinstance(estimate, dict):
        return ""
    usd = estimate.get("usd")
    price = f"at most ${usd:,.2f}" if isinstance(usd, int | float) else "unpriced"
    return (
        f"{label}: {estimate.get('calls', 0):,} calls on {estimate.get('model')}, "
        f"about {estimate.get('input_tokens', 0):,} input and "
        f"{estimate.get('output_tokens', 0):,} output tokens, {price} at list price."
    )


async def _episode(
    *,
    scope: str,
    limit: int | None,
    rate_limit_per_minute: int | None,
    restamp: bool,
    dry_run: bool,
    sheet: Path | None,
    sheet_size: int | None,
    seed: int,
    verbose: bool,
) -> None:
    from particles.api.client import get_backend

    if get_backend().remote:
        typer.echo(
            "Error: `particles episode` reads one local store per invocation; "
            "run it on the machine that holds the store.",
            err=True,
        )
        raise typer.Exit(2)
    if scope not in {"cards", "all"}:
        typer.echo("Error: --scope is 'cards' or 'all'.", err=True)
        raise typer.Exit(2)

    # Deferred import: the operation pulls the LLM + store stack, and tests
    # patch it at call time (tests/AGENTS.md § Mocking strategy).
    from particles.operations.episode import regenerate_episode, write_sheet

    if sheet is not None:
        async with session_scope() as session:
            written = await write_sheet(session, sheet, size=sheet_size, seed=seed)
        typer.echo(json.dumps(written.as_dict(), indent=2))
        return

    progress = progress_line if verbose else None

    async def _run() -> dict[str, object]:
        async with session_scope(write=not dry_run) as session:
            summary = await regenerate_episode(
                session,
                scope=scope,
                limit=limit,
                rate_limit_per_minute=rate_limit_per_minute,
                restamp=restamp,
                dry_run=dry_run,
                progress=progress,
            )
            return summary.as_dict()

    if dry_run:
        report = await _run()
        for key, label in (("estimate", "This run"), ("backlog_estimate", "Whole backlog")):
            line = _estimate_line(label, report.get(key))
            if line:
                typer.echo(line, err=True)
        typer.echo(json.dumps(report, indent=2))
        return
    meter = MeteredExtractRun(actor="cli:episode", route="episode")
    try:
        async with meter:
            summary = await _run()
    finally:
        if meter.llm_usage is not None:
            typer.echo(render_usage_line(meter.llm_usage), err=True)
    typer.echo(json.dumps(summary, indent=2))
