# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""extract verb — run extraction over PENDING snapshots."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import typer

from particles.api.cli import app, run
from particles.api.cli._output import (
    PROGRESS_OPTION,
    QUIET_OPTION,
    configure_output,
    current_output,
)
from particles.api.client import get_backend
from particles.core.schema import Particle
from particles.db import session_scope
from particles.llm.usage import LLMUsage, render_usage_line
from particles.operations.llm_spend import MeteredExtractRun


def _echo_page_stats(
    page_stats: list[Any],
    carry_forward_ids: list[str],
    *,
    label: str = "Pages",
    indent: str = "    ",
) -> None:
    """Report per-page yield: aggregated by default, itemised under ``--verbose``.

    The ``--verbose`` flag is the un-aggregate knob, and this listing is
    exactly what it is for: an 83-page PDF printed 83 lines unconditionally,
    which buried the four other snapshots in the same run.

    The ``⚠ zero yield`` flag is also suppressed when the snapshot had
    carry-forward hits. A page whose chunks matched an existing ACTIVE particle
    yields no *new* candidates **by design** — the cache working, not an
    anomaly — so flagging it as one was a false warning on the most common
    re-extraction path. Without carry-forward, an empty page is still worth
    flagging (an unparseable scan, a chart-only page).
    """
    if not page_stats:
        return
    singular = label[:-1] if label.endswith("s") else label
    carried = bool(carry_forward_ids)
    empty = sum(1 for ps in page_stats if ps.candidate_count == 0)

    if not current_output().verbose:
        if empty:
            # The caller's summary line already carries the page count, so this
            # reports the split rather than restating it.
            reason = (
                " (the rest matched already-extracted content)"
                if carried
                else " — the rest yielded none"
            )
            typer.echo(
                f"{indent}{len(page_stats) - empty} of {len(page_stats)} "
                f"{label.lower()} produced particles{reason}"
            )
        return

    typer.echo(f"{indent}{label}: {len(page_stats)}")
    for ps in page_stats:
        flag = "  ⚠ zero yield" if ps.candidate_count == 0 and not carried else ""
        typer.echo(
            f"{indent}  {singular} {ps.page_number:3d}: {ps.candidate_count:4d} particles{flag}"
        )


@app.command("extract")
def extract_cmd(
    entry_id: str | None = typer.Argument(None, help="Corpus entry ID (omit with --all-pending)"),
    snapshot_id: str | None = typer.Option(None, help="Snapshot ID; defaults to latest PENDING"),
    agent_id: str = typer.Option("cli-user", help="Asserted-by agent ID"),
    all_pending: bool = typer.Option(
        False, "--all-pending", help="Extract all PENDING snapshots in deposit order"
    ),
    tag: str | None = typer.Option(
        None,
        "--tag",
        help=(
            "With --all-pending, extract only snapshots of corpus entries that carry this "
            "tag (for example memory-file for Claude Code memory files)."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show quality notes and INFO logs"),
    debug: bool = typer.Option(
        False, "--debug", help="Show raw LLM prompt/response and DEBUG logs"
    ),
    quiet: bool = QUIET_OPTION,
    progress: bool | None = PROGRESS_OPTION,
) -> None:
    """Extract particles from a corpus entry, or all PENDING entries at once."""
    configure_output(verbose, debug, quiet, progress)

    if tag is not None and not all_pending:
        typer.echo("--tag filters --all-pending; add --all-pending.", err=True)
        raise typer.Exit(1)

    if all_pending:
        if get_backend().remote:
            typer.echo(
                "--all-pending iterates the local store and is not available "
                "against a remote engine. Extract entries by ID instead.",
                err=True,
            )
            raise typer.Exit(1)
        run(_extract_all_pending(agent_id, tag=tag))
        return

    if not entry_id:
        typer.echo("Provide an entry ID or use --all-pending.", err=True)
        raise typer.Exit(1)

    entry_id, particles, page_stats, carry_forward_ids, suppressed_ids, llm_usage = run(
        _extract(entry_id, snapshot_id, agent_id)
    )
    summary = f"Extracted {len(particles)} particles"
    extras: list[str] = []
    if carry_forward_ids:
        # carry-forward: existing particles whose chunk_hash
        # matched were reused without an LLM call. Surface the count so
        # operators see the work that ran versus the work that was
        # skipped — silently dropping this leaves them thinking
        # extraction did nothing on a re-deposit.
        extras.append(f"{len(carry_forward_ids)} already extracted")
    if suppressed_ids:
        # suppression is never silent. Without this line a
        # re-harvest reads as "extracted 3 particles" when it in fact
        # recognised 40 claims the store already held.
        extras.append(f"{len(suppressed_ids)} duplicate(s) suppressed")
    if extras:
        summary += f" ({', '.join(extras)})"
    typer.echo(summary + ".")
    _echo_page_stats(
        page_stats,
        carry_forward_ids,
        label="Pages" if _is_pdf_entry(entry_id) else "Chunks",
        indent="  ",
    )
    for p in particles:
        typer.echo(f"  [{p.status}] {p.id[:8]}… {p.content[:80]}")
    _echo_usage(llm_usage)


@contextlib.asynccontextmanager
async def _metered_run(route: str) -> AsyncIterator[MeteredExtractRun]:
    """Meter one extract run and print its usage line when it ends, however it ends.

    The calls made before an account-level stop were still billed, so the line
    and the ``EXTRACT_RUN`` record both survive an early exit.
    """
    meter = MeteredExtractRun(actor="cli:extract", route=route)
    try:
        async with meter:
            yield meter
    finally:
        _echo_usage(meter.llm_usage)


def _echo_usage(llm_usage: LLMUsage | None) -> None:
    """The run's measured LLM usage, on stderr like ``audit``'s.

    ``None`` is a remote run: the engine made and recorded the calls, so the
    laptop has nothing to report.
    """
    if llm_usage is not None:
        typer.echo(render_usage_line(llm_usage), err=True)


async def _extract_all_pending(agent_id: str, *, tag: str | None = None) -> None:
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import select
    from sqlalchemy.exc import OperationalError

    from particles.config import get_config
    from particles.core.schema import ExtractionStatus
    from particles.corpus.store import (
        CorpusEntryRow,
        SnapshotRow,
        count_snapshots_by_extraction_status,
        reset_stale_in_progress,
    )
    from particles.ingest.pipeline import SnapshotOutcome
    from particles.llm import AccountLevelLLMError, get_client
    from particles.operations.extract import collapse_superseded_pending, extract_snapshot
    from particles.operations.version_guard import assert_store_schema_current

    # Fail fast if ANTHROPIC_API_KEY is unset — otherwise we'd run through every
    # pending snapshot reporting the same auth error.
    try:
        get_client()
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    # Fail fast on schema_version mismatch. extract_snapshot calls
    # the guard per snapshot too, but the per-snapshot path catches the
    # exception inside the loop and prints N copies of the same message.
    # Calling the guard upfront lets the CLI's run() helper translate one
    # raise into one stderr line + exit 1.
    async with session_scope() as session:
        await assert_store_schema_current(session)

    # 0.42.2: reset IN_PROGRESS snapshots stranded by a SIGKILL / segfault
    # whose try/except cleanup in extract_snapshot didn't run. The Ctrl+C
    # path is handled by the pipeline's cleanup; this is the defence-in-
    # depth pass for everything else.
    threshold_minutes = get_config().extraction.stale_in_progress_minutes
    cutoff = datetime.now(UTC) - timedelta(minutes=threshold_minutes)
    async with session_scope() as session:
        stale = await reset_stale_in_progress(session, older_than=cutoff)
        await session.commit()
    if stale:
        typer.echo(
            f"Reset {len(stale)} stale IN_PROGRESS snapshot(s) to PENDING "
            f"(older than {threshold_minutes:g} min)."
        )

    # a MUTABLE source edited N times since the last pass has N
    # pending snapshots; only the newest is worth a call. Runs before the
    # listing so the skipped generations never appear in it, and is never
    # silent — the operator sees what was not extracted and why.
    async with session_scope() as session:
        collapse = await collapse_superseded_pending(session)
    collapse_line = collapse.summary()
    if collapse_line:
        typer.echo(collapse_line)

    async with session_scope() as session:
        result = await session.execute(
            select(SnapshotRow.entry_id, SnapshotRow.snapshot_id, CorpusEntryRow.tags_json)
            .where(SnapshotRow.extraction_status == ExtractionStatus.PENDING.value)
            .join(CorpusEntryRow, SnapshotRow.entry_id == CorpusEntryRow.entry_id)
            .order_by(CorpusEntryRow.created_at)
        )
        pending = [
            (entry_id, snapshot_id)
            for entry_id, snapshot_id, tags_json in result.all()
            if tag is None or entry_has_tag(tags_json, tag)
        ]
        # Counts for the no-pending message below. "No PENDING snapshots
        # found" alone is ambiguous after a failed run — an operator whose
        # previous extraction printed FAILED lines reads it as "my snapshots
        # vanished", when a concurrent runner usually just completed them.
        counts = await count_snapshots_by_extraction_status(session) if not pending else {}

    if not pending:
        breakdown = ", ".join(
            f"{counts[s.value]} {s.value}" for s in ExtractionStatus if counts.get(s.value)
        )
        suffix = f" ({breakdown})" if breakdown else ""
        scope = f" tagged {tag}" if tag is not None else ""
        typer.echo(f"No PENDING snapshots{scope} found{suffix}.")
        return

    typer.echo(f"Extracting {len(pending)} pending snapshot(s)…")
    failures = 0
    left_pending = 0
    # one metered run around the whole pass, one EXTRACT_RUN record
    # and one usage line.
    async with _metered_run("all-pending") as meter:
        for index, (entry_id, snapshot_id) in enumerate(pending):
            meter.snapshots += 1
            # ``id_prefix`` disambiguates multiple snapshots of the same
            # corpus entry — a single ``--all-pending`` run after a scrap-
            # and-re-extract commonly queues every historical
            # snapshot of every entry, and bare ``entry_id[:8]`` lines
            # printed three or four times with different particle counts
            # are confusing.
            id_prefix = f"{entry_id[:8]}…/{snapshot_id[:8]}…"
            page_stats: list[Any] = []
            carry_forward_ids: list[str] = []
            suppressed_ids: list[str] = []
            outcome = SnapshotOutcome()
            async with session_scope() as session:
                try:
                    particles = await extract_snapshot(
                        session,
                        entry_id,
                        snapshot_id,
                        agent_id=agent_id,
                        page_stats_out=page_stats,
                        carry_forward_ids_out=carry_forward_ids,
                        suppressed_ids_out=suppressed_ids,
                        skip_if_superseded=True,
                        outcome_out=outcome,
                    )
                    await session.commit()
                    if outcome.skipped == "waiting":
                        held = (outcome.waiting_on or "")[:8]
                        remedy = (
                            "it is FAILED: restore its blob, or run `particles reindex` "
                            "on the entry"
                            if outcome.waiting_on_failed
                            else "it is read once that snapshot completes"
                        )
                        typer.echo(
                            f"  {id_prefix}  skipped (waiting on {held}…, which holds a "
                            f"partial whole read; {remedy})"
                        )
                        continue
                    if outcome.skipped is not None:
                        typer.echo(f"  {id_prefix}  skipped ({outcome.skipped})")
                        continue
                    if outcome.failed_calls and outcome.kept_calls:
                        # A partly failed append-only read kept what it answered;
                        # the snapshot stays PENDING for the rest.
                        left_pending += 1
                        resent = (
                            f"; a retry sends {outcome.rebilled_calls} other answered call(s) again"
                            if outcome.rebilled_calls
                            else ""
                        )
                        typer.echo(
                            f"  {id_prefix}  left PENDING: {outcome.failed_calls} LLM call(s) "
                            f"produced nothing usable; kept {outcome.kept_calls} answered "
                            f"call(s) ({len(particles)} particles), and a retry reads only "
                            f"the rest{resent}",
                            err=True,
                        )
                        continue
                    if outcome.failed_calls:
                        # The pipeline handed the snapshot back PENDING with nothing
                        # written. Printing "0 particles" here read as a finished
                        # extraction of an empty source.
                        left_pending += 1
                        resent = (
                            f"; a retry sends its {outcome.rebilled_calls} answered call(s) again"
                            if outcome.rebilled_calls
                            else ""
                        )
                        typer.echo(
                            f"  {id_prefix}  left PENDING: {outcome.failed_calls} LLM call(s) "
                            f"produced nothing usable, so nothing was written{resent}",
                            err=True,
                        )
                        continue
                    # Merge ADR-0057 carry-forward count and the existing
                    # page-count suffix into one parenthesised group, so
                    # the line stays compact when both fire:
                    #   ``0 particles (31 already extracted, 3 pages)``.
                    # "already extracted" is plain-English for "matched the
                    # chunk_hash of an existing ACTIVE particle, no LLM call
                    # needed" — without it, the CLI shows "0 particles" for
                    # a snapshot whose entire content was actually reused
                    # from a sibling snapshot.
                    extras: list[str] = []
                    if carry_forward_ids:
                        extras.append(f"{len(carry_forward_ids)} already extracted")
                    if suppressed_ids:
                        # same reasoning as "already extracted": a
                        # re-harvest whose claims the store already holds must not
                        # read as "0 particles" with no explanation.
                        extras.append(f"{len(suppressed_ids)} duplicate(s) suppressed")
                    if outcome.conflicts_cleared:
                        # a contradiction the probe flagged and a second
                        # reading did not confirm wrote no record.
                        extras.append(
                            f"{outcome.conflicts_cleared} contradiction flag(s) cleared on a "
                            "second reading"
                        )
                    if outcome.conflict_unread:
                        extras.append(f"{outcome.conflict_unread} contradiction flag(s) unread")
                    if page_stats:
                        extras.append(f"{len(page_stats)} pages")
                    summary = f"{len(particles)} particles"
                    if extras:
                        summary += f" ({', '.join(extras)})"
                    typer.echo(f"  {id_prefix}  {summary}")
                    _echo_page_stats(page_stats, carry_forward_ids)
                except AccountLevelLLMError as exc:
                    # Bad / missing key, no permission, or no credit: every
                    # remaining snapshot would fail the same way, so stop here
                    # instead of restating one billing error per snapshot (and per
                    # PDF page). The pipeline already reset this snapshot
                    # IN_PROGRESS → PENDING, and the rest were never claimed, so
                    # the whole queue survives for a retry.
                    remaining = len(pending) - index
                    typer.echo(
                        f"\nLLM unavailable (account-level): {exc}",
                        err=True,
                    )
                    typer.echo(
                        f"Stopped after {index} of {len(pending)} snapshot(s); "
                        f"{remaining} still PENDING. Fix the API key or credit balance, "
                        "then re-run `particles extract --all-pending`.",
                        err=True,
                    )
                    raise typer.Exit(1) from exc
                except Exception as exc:
                    failures += 1
                    if (
                        isinstance(exc, OperationalError)
                        and "database is locked" in str(exc).lower()
                    ):
                        # The per-snapshot handler swallows the exception before
                        # it reaches run()'s lock translation, so give the same
                        # operator-friendly message here instead of the raw
                        # SQLAlchemy dump.
                        typer.echo(
                            f"  {id_prefix}  FAILED: database is locked — another "
                            "particles process is holding a writer transaction. "
                            "Re-run `particles extract --all-pending` once it "
                            "finishes to retry this snapshot.",
                            err=True,
                        )
                    else:
                        typer.echo(f"  {id_prefix}  FAILED: {exc}", err=True)

    if left_pending:
        typer.echo(
            f"{left_pending} of {len(pending)} snapshot(s) left PENDING after failed LLM "
            "calls. Re-run `particles extract --all-pending` to retry them.",
            err=True,
        )
    if failures:
        typer.echo(
            f"Extraction failed for {failures} of {len(pending)} snapshot(s).",
            err=True,
        )
    if failures or left_pending:
        raise typer.Exit(1)


def entry_has_tag(tags_json: str | None, tag: str) -> bool:
    """Whether a corpus entry's stored tag list names ``tag`` exactly."""
    import json

    try:
        tags = json.loads(tags_json or "[]")
    except ValueError:
        return False
    return isinstance(tags, list) and tag in tags


def _is_pdf_entry(entry_id: str) -> bool:
    """Quick check whether an entry is PDF-sourced, for CLI label selection."""
    try:
        import asyncio

        from particles.corpus.store import get_entry

        async def _get() -> bool:
            async with session_scope() as s:
                e = await get_entry(s, entry_id)
                return e is not None and e.source_type == "PDF"

        return asyncio.run(_get())
    except Exception:
        return False


async def _resolve_local(entry_id: str, snapshot_id: str | None) -> tuple[str, str]:
    """Resolve an entry-ID / snapshot-ID prefix against the local store.

    Accepts the 8-char prefix form ``deposit`` / ``extract`` print, the way
    ``subjects show`` and the other verbs do. Local-only: ``extract_snapshot``
    itself only takes full UUIDs, and a remote engine resolves nothing for the
    laptop — remote mode requires full IDs.
    """
    from particles.corpus.store import (
        list_snapshots_for_entry,
        resolve_entry_id,
        resolve_snapshot_id,
    )

    async with session_scope() as session:
        try:
            resolved_entry = await resolve_entry_id(session, entry_id)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        if resolved_entry is None:
            typer.echo(f"Corpus entry {entry_id!r} not found.", err=True)
            raise typer.Exit(1)
        entry_id = resolved_entry

        if snapshot_id is None:
            snaps = await list_snapshots_for_entry(session, entry_id)
            if not snaps:
                typer.echo("No snapshots found for entry.", err=True)
                raise typer.Exit(1)
            snapshot_id = snaps[-1].snapshot_id
        else:
            try:
                resolved_snap = await resolve_snapshot_id(session, entry_id, snapshot_id)
            except ValueError as exc:
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(1) from exc
            if resolved_snap is None:
                typer.echo(f"Snapshot {snapshot_id!r} not found in entry.", err=True)
                raise typer.Exit(1)
            snapshot_id = resolved_snap
    return entry_id, snapshot_id


async def _extract(
    entry_id: str, snapshot_id: str | None, agent_id: str
) -> tuple[str, list[Particle], list[Any], list[str], list[str], LLMUsage | None]:
    backend = get_backend()
    if backend.remote:
        # The engine takes full UUIDs and does not infer the latest snapshot;
        # the laptop has no store to resolve a prefix against.
        if snapshot_id is None:
            typer.echo(
                "In remote mode, --snapshot-id is required (the engine does not "
                "infer the latest snapshot). Use the IDs printed by `deposit`.",
                err=True,
            )
            raise typer.Exit(1)
        resolved_snapshot = snapshot_id
    else:
        entry_id, resolved_snapshot = await _resolve_local(entry_id, snapshot_id)

    outcome = await backend.extract(entry_id, resolved_snapshot, agent_id=agent_id)
    return (
        outcome.entry_id,
        outcome.particles,
        outcome.page_stats,
        outcome.carry_forward_ids,
        outcome.suppressed_ids,
        outcome.llm_usage,
    )
