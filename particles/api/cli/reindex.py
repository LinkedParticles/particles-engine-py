# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""reindex verb — re-extract particles for stale or failed corpus entries."""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from typing import cast

import typer

from particles.api.cli import app, run
from particles.api.cli._progress import progress_line, set_heartbeat_status
from particles.api.cli._remote import refuse_remote_sync
from particles.api.client import get_backend
from particles.llm.usage import LLMUsage, render_usage_line
from particles.operations.reindex_scope import (
    ONLY_CHANGED_COMPONENTS_ENABLED,
    ONLY_CHANGED_COMPONENTS_REFUSAL,
)

#: Exit code when a flag combination or a declined spend stops the verb early.
_EXIT_NOT_STARTED = 2


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
        help="Extractor name (e.g. github-repo-extractor): re-extract all of its "
        "particles regardless of version. Useful when a shared upstream change "
        "(e.g. a prompt revision in general.py) affects delegating extractors.",
    ),
    provider_model: str | None = typer.Option(
        None,
        help='"<provider>:<model>" pairing (e.g. openai:gpt-5.6-luna): re-extract '
        "every particle that pairing produced. The handle for undoing an "
        "uncalibrated provider swap. Matched exactly, and the scope unit is the "
        "snapshot, so a snapshot with a model-mixed population is re-extracted "
        "whole. Particles with no recorded pairing never match.",
    ),
    no_failed: bool = typer.Option(
        False,
        help="Skip FAILED snapshot entries. Applies to auto-discovery only; "
        "--entry-ids resolves each entry to its latest COMPLETE snapshot.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the work plan (entries / snapshots / particles in scope, "
        "with per-snapshot counts and any known-missing blobs) and exit "
        "without extracting: zero LLM calls, zero writes.",
    ),
    estimate: bool = typer.Option(
        False,
        "--estimate",
        help="Measure what an --extractor-version reindex would change before "
        "running it: re-extract a seeded sample of the snapshots it would sweep, "
        "judge each sample's new claims against its stored ones, and report the "
        "projected share of changed snapshots and the projected cost of the full "
        "sweep. Spends only on the sample and writes nothing. Prints the plan and "
        "asks before spending; a non-interactive run needs --yes. Local store only.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="With --estimate: spend on the sample without asking.",
    ),
    sample_size: int | None = typer.Option(
        None,
        "--sample-size",
        min=1,
        help="With --estimate: snapshots to sample (default: reindex.estimate_sample_size).",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="With --estimate: the sample's random seed (default: reindex.estimate_seed).",
    ),
    only_changed_components: bool = typer.Option(
        False,
        "--only-changed-components",
        help="Narrow an --extractor-version scope to the snapshots whose recorded "
        "extraction components changed. Not enabled yet: refused until an "
        "extractor version bump has run over a store whose snapshots record "
        "their components.",
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
    """Re-extract particles for stale or failed corpus entries.

    With --estimate, measure on a sample what an --extractor-version reindex
    would change and cost, and stop there. Exit codes for --estimate: 0 when
    the estimate ran, 2 when it did not start (invalid options, the spend
    declined, or no --yes in a non-interactive run).
    """
    eids = [e.strip() for e in entry_ids.split(",")] if entry_ids else None
    if only_changed_components and not ONLY_CHANGED_COMPONENTS_ENABLED:
        typer.echo(f"Error: {ONLY_CHANGED_COMPONENTS_REFUSAL}", err=True)
        raise typer.Exit(_EXIT_NOT_STARTED)
    if estimate:
        _refuse_estimate_misuse(extractor_version, dry_run, only_changed_components)
        refuse_remote_sync("reindex --estimate")
        run(
            _estimate(
                eids,
                extractor_version or "",
                extractor_id,
                not no_failed,
                provider_model,
                sample_size=sample_size,
                seed=seed,
                yes=yes,
                output_format=output_format,
            )
        )
        return
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
            only_changed_components,
        )
    )
    if output_format is _ReindexFormat.json:
        typer.echo(json.dumps(summary, indent=2))
    else:
        _echo_human_summary(summary, dry_run)
    _echo_usage(summary)


def _echo_usage(summary: dict[str, object]) -> None:
    """The run's measured LLM usage on stderr, as ``extract`` prints it.

    Absent on a dry run, which makes no call, and over HTTP, where the engine
    made and recorded the calls.
    """
    raw = summary.get("llm_usage")
    if isinstance(raw, dict):
        typer.echo(render_usage_line(LLMUsage.model_validate(raw)), err=True)


async def _reindex(
    entry_ids: list[str] | None,
    extractor_version: str | None,
    extractor_id: str | None,
    include_failed: bool,
    provider_model: str | None,
    dry_run: bool,
    verbose: bool,
    output_format: _ReindexFormat,
    only_changed_components: bool = False,
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
        only_changed_components=only_changed_components,
    )


def _refuse_estimate_misuse(
    extractor_version: str | None, dry_run: bool, only_changed_components: bool
) -> None:
    """Refuse the flag combinations --estimate cannot honour (exit 2)."""
    problem: str | None = None
    if not extractor_version:
        problem = (
            "--estimate measures an extractor version bump; pass the superseded "
            "version with --extractor-version."
        )
    elif dry_run:
        problem = "--estimate and --dry-run are separate reports; pass one."
    elif only_changed_components:
        problem = "--estimate samples the full version scope; drop --only-changed-components."
    if problem is not None:
        typer.echo(f"Error: {problem}", err=True)
        raise typer.Exit(_EXIT_NOT_STARTED)


async def _estimate(
    entry_ids: list[str] | None,
    extractor_version: str,
    extractor_id: str | None,
    include_failed: bool,
    provider_model: str | None,
    *,
    sample_size: int | None,
    seed: int | None,
    yes: bool,
    output_format: _ReindexFormat,
) -> None:
    """Plan the sample, ask, spend on it, and report.

    The plan is free and always printed first; the spend waits for a yes. In
    the JSON format the plan and the prompt go to stderr so stdout stays one
    parseable document.
    """
    # Branch-local (AGENTS.md § Deferred imports case 4): only --estimate
    # reaches the estimate operation, the store session, and the meter.
    from particles.db import session_scope
    from particles.operations.llm_spend import MeteredExtractRun
    from particles.operations.reindex_estimate import (
        plan_reindex_estimate,
        run_reindex_estimate,
    )

    as_json = output_format is _ReindexFormat.json
    async with session_scope() as session:
        plan = await plan_reindex_estimate(
            session,
            extractor_version=extractor_version,
            entry_ids=entry_ids,
            extractor_id=extractor_id,
            include_failed=include_failed,
            provider_model=provider_model,
            sample_size=sample_size,
            seed=seed,
        )
    for line in plan.format_lines():
        typer.echo(line, err=as_json)
    if not plan.sample:
        typer.echo(
            "Nothing to sample: no snapshot in scope has stored claims to compare.",
            err=as_json,
        )
        return
    if not yes:
        if not sys.stdin.isatty():
            typer.echo(
                "Error: --estimate spends on its sample and asks first; pass --yes "
                "in a non-interactive run. Nothing was spent.",
                err=True,
            )
            raise typer.Exit(_EXIT_NOT_STARTED)
        if not typer.confirm(
            f"Re-extract and judge {len(plan.sample)} sampled snapshot(s)?", err=as_json
        ):
            typer.echo("Aborted: nothing was spent.", err=as_json)
            raise typer.Exit(_EXIT_NOT_STARTED)

    # the sample is metered and recorded like any reindex spend.
    meter = MeteredExtractRun(actor="cli:reindex", route="reindex")
    async with meter:
        async with session_scope() as session:
            report = await run_reindex_estimate(session, plan)
        meter.snapshots = len(plan.sample)
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
    else:
        for line in report.format_lines():
            typer.echo(line)
    if meter.llm_usage is not None:
        typer.echo(render_usage_line(meter.llm_usage), err=True)


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
