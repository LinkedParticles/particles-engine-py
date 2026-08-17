"""reindex verb — re-extract particles for stale or failed corpus entries."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import cast

import typer

from particles.api.cli import app, run
from particles.api.cli._progress import progress_line, set_heartbeat_status
from particles.api.client import get_backend


class _ReindexFormat(StrEnum):
    human = "human"
    json = "json"


@app.command("reindex")
def reindex_cmd(
    entry_ids: str | None = typer.Option(
        None,
        help="Comma-separated entry IDs (full or unambiguous prefix); omit for auto. "
        "Combines with --extractor-version / --extractor-id / --provider-model by "
        "intersection: only the named entries that also match the filter are "
        "reindexed, and any that don't are reported.",
    ),
    extractor_version: str | None = typer.Option(None, help="Old extractor version to replace"),
    extractor_id: str | None = typer.Option(
        None,
        help="Extractor name (e.g. github-repo-extractor) — re-extract all of its "
        "particles regardless of version. Useful when a shared upstream change "
        "(e.g. a prompt revision in general.py) affects delegating extractors.",
    ),
    provider_model: str | None = typer.Option(
        None,
        help='"<provider>:<model>" pairing (e.g. openai:gpt-5.6-luna) — re-extract '
        "every particle that pairing produced. The handle for undoing an "
        "uncalibrated provider swap. Matched exactly, and the scope unit is the "
        "snapshot, so a snapshot with a model-mixed population is re-extracted "
        "whole. Particles with no recorded pairing never match.",
    ),
    no_failed: bool = typer.Option(
        False,
        help="Skip FAILED snapshot entries. Applies to auto-discovery only — "
        "--entry-ids resolves each entry to its latest COMPLETE snapshot.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the work plan — entries / snapshots / particles in scope, "
        "with per-snapshot counts and any known-missing blobs — and exit "
        "without extracting: zero LLM calls, zero writes.",
    ),
    output_format: _ReindexFormat = typer.Option(
        _ReindexFormat.human,
        "--format",
        help="Output format: a short human summary (default), or the full JSON "
        "result envelope including the per-snapshot plan.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print scope size and per-entry progress while reindexing.",
    ),
) -> None:
    """Re-extract particles for stale or failed corpus entries."""
    eids = [e.strip() for e in entry_ids.split(",")] if entry_ids else None
    summary = run(
        _reindex(
            eids,
            extractor_version,
            extractor_id,
            not no_failed,
            provider_model,
            dry_run,
            verbose,
            output_format,
        )
    )
    if output_format is _ReindexFormat.json:
        typer.echo(json.dumps(summary, indent=2))
    else:
        _echo_human_summary(summary, dry_run)


async def _reindex(
    entry_ids: list[str] | None,
    extractor_version: str | None,
    extractor_id: str | None,
    include_failed: bool,
    provider_model: str | None,
    dry_run: bool,
    verbose: bool,
    output_format: _ReindexFormat,
) -> dict[str, object]:
    # --verbose per-entry progress is a local stderr stream; the remote backend
    # ignores the callbacks (no HTTP streaming) but still returns the summary.
    # The work plan (on_plan) prints unconditionally — the whole point is that
    # a bare `particles reindex` says what it's about to sweep BEFORE the first
    # LLM call — while per-entry progress stays opt-in behind --verbose.
    # Exception: a human-format dry run renders the plan itself on stdout (the
    # plan IS the artifact there), so it skips the stderr stream.
    progress = progress_line if verbose else None
    on_plan = None if (dry_run and output_format is _ReindexFormat.human) else progress_line
    return await get_backend().reindex(
        entry_ids=entry_ids,
        extractor_version=extractor_version,
        extractor_id=extractor_id,
        include_failed=include_failed,
        provider_model=provider_model,
        progress=progress,
        dry_run=dry_run,
        on_plan=on_plan,
        # Per-item position for the heartbeat line: "snapshot 12/89 (entry
        # 0a8fb1a9…) — 3 failed" instead of the bare time-only "working".
        on_status=set_heartbeat_status,
    )


def _echo_human_summary(summary: dict[str, object], dry_run: bool) -> None:
    """The human stdout artifact: one-line plan / counts, never the raw envelope."""
    # Branch-local (AGENTS.md § Deferred imports case 4): only the human format
    # arm re-validates the plan; --format json echoes the envelope untouched.
    from particles.operations.reindex import ReindexPlan

    plan = ReindexPlan.model_validate(summary["plan"])
    if dry_run:
        typer.echo(plan.format_line())
        for line in plan.format_missing_blob_lines():
            typer.echo(line)
        typer.echo("Dry run — nothing extracted.")
        return

    typer.echo(
        f"Reindex complete: {summary['succeeded']} succeeded, "
        f"{summary['failed']} failed (scope: {summary['scope']} snapshot(s))."
    )
    failed_entries = cast(list[str], summary.get("failed_entries") or [])
    if failed_entries:
        shown = ", ".join(f"{e[:8]}…" for e in failed_entries[:5])
        remainder = len(failed_entries) - 5
        suffix = f" … and {remainder} more" if remainder > 0 else ""
        typer.echo(f"Failed entries: {shown}{suffix} (see --format json)")
    lint_summary = cast(dict[str, int], summary.get("lint_summary") or {})
    if lint_summary:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(lint_summary.items()))
        typer.echo(f"Post-reindex lint: {rendered}")
