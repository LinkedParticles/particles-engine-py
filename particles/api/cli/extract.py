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
from particles.api.cli._estimate_text import duration_text, range_text, usd
from particles.api.cli._output import (
    PROGRESS_OPTION,
    QUIET_OPTION,
    configure_output,
    current_output,
)
from particles.api.cli._progress import confirm
from particles.api.client import get_backend
from particles.core.schema import Particle
from particles.db import session_scope
from particles.llm.usage import LLMUsage, render_usage_line
from particles.operations.extract_run import RunPlan, extraction_route, parse_model_route
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
    retry_refused: bool = typer.Option(
        False,
        "--retry-refused",
        help=(
            "With --all-pending, also send the sources the extraction model refused, "
            "which a run otherwise holds until the model or the content changes or "
            "extraction.refusal_retry_days passes."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "With --snapshot-id, extract a snapshot older than the source's current "
            "version. Its claims the current version does not state are retired "
            "again at once, and for a source recorded as no longer found, the "
            "claims its deletion retired come back."
        ),
    ),
    estimate: bool = typer.Option(
        False,
        "--estimate",
        help=(
            "With --all-pending, print what the run would read and cost, then exit without "
            "a model call. Needs no API key."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="With --all-pending, skip the question asked above extraction.confirm_call_threshold.",
    ),
    budget_usd: float | None = typer.Option(
        None,
        "--budget-usd",
        min=0.0,
        help=(
            "With --all-pending, stop before a snapshot whose estimate would carry the run's "
            "spend past this many US dollars (list price). What is left stays pending."
        ),
    ),
    extraction_model: str | None = typer.Option(
        None,
        "--extraction-model",
        help=(
            "Extract this run with another model, as provider:model (for example "
            "anthropic:claude-haiku-5-5). Routes extraction and subject resolution only."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show quality notes and INFO logs"),
    debug: bool = typer.Option(
        False, "--debug", help="Show raw LLM prompt/response and DEBUG logs"
    ),
    quiet: bool = QUIET_OPTION,
    progress: bool | None = PROGRESS_OPTION,
) -> None:
    """Extract particles from a corpus entry, or all PENDING entries at once.

    With ``--all-pending`` the run prints its estimate first and asks before
    spending above ``extraction.confirm_call_threshold`` calls. ``--budget-usd``
    stops it between snapshots and reports what is left; edits of sources the
    store has already read are extracted before sources it has never read.
    """
    configure_output(verbose, debug, quiet, progress)

    if tag is not None and not all_pending:
        typer.echo("--tag filters --all-pending; add --all-pending.", err=True)
        raise typer.Exit(1)
    if (estimate or yes or budget_usd is not None) and not all_pending:
        typer.echo("--estimate, --yes and --budget-usd apply to --all-pending.", err=True)
        raise typer.Exit(1)
    if retry_refused and not all_pending:
        typer.echo(
            "--retry-refused applies to --all-pending; naming an entry always sends it.",
            err=True,
        )
        raise typer.Exit(1)
    route = None
    if extraction_model is not None:
        try:
            route = parse_model_route(extraction_model)
        except ValueError as exc:
            typer.echo(f"--extraction-model: {exc}", err=True)
            raise typer.Exit(2) from exc
        if get_backend().remote:
            typer.echo(
                "--extraction-model routes this process's calls; against a remote engine "
                "the engine's configuration chooses the model.",
                err=True,
            )
            raise typer.Exit(1)

    with extraction_route(route):
        if all_pending:
            if get_backend().remote:
                typer.echo(
                    "--all-pending iterates the local store and is not available "
                    "against a remote engine. Extract entries by ID instead.",
                    err=True,
                )
                raise typer.Exit(1)
            run(
                _extract_all_pending(
                    agent_id,
                    tag=tag,
                    estimate_only=estimate,
                    yes=yes,
                    budget_usd=budget_usd,
                    retry_refused=retry_refused,
                )
            )
            return

        if not entry_id:
            typer.echo("Provide an entry ID or use --all-pending.", err=True)
            raise typer.Exit(1)

        (
            entry_id,
            particles,
            page_stats,
            carry_forward_ids,
            suppressed_ids,
            settled,
            restated,
            llm_usage,
        ) = run(_extract(entry_id, snapshot_id, agent_id, force=force))
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
    if settled:
        # a revision pair settled by the generation leaves no
        # record, so the count is the disclosure.
        extras.append(_revision_settled_extra(settled))
    if restated:
        # a restatement folded into its earlier claim mints
        # nothing, so the count is the disclosure, as for a suppression.
        extras.append(_restated_extra(restated))
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


def _restated_extra(count: int) -> str:
    """The summary-line part for restatements carried into the new generation."""
    return f"{count} restated claim(s) carried forward"


def _revision_settled_extra(count: int) -> str:
    """The summary-line part for revision pairs settled by the generation."""
    return f"{count} revision pair(s) settled by the generation"


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


async def _extract_all_pending(
    agent_id: str,
    *,
    tag: str | None = None,
    estimate_only: bool = False,
    yes: bool = False,
    budget_usd: float | None = None,
    retry_refused: bool = False,
) -> None:
    import sys
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import select
    from sqlalchemy.exc import OperationalError

    from particles.config import get_config
    from particles.core.schema import ExtractionStatus, WarcRecordType
    from particles.corpus.store import (
        CorpusEntryRow,
        SnapshotRow,
        count_snapshots_by_extraction_status,
        entry_ids_ever_extracted,
        reset_stale_in_progress,
    )
    from particles.ingest.pipeline import SnapshotOutcome
    from particles.ingest.refusal_hold import held_snapshots
    from particles.llm import AccountLevelLLMError, get_client
    from particles.operations.extract import collapse_superseded_pending, extract_snapshot
    from particles.operations.extract_run import (
        BudgetGate,
        PendingSnapshot,
        decide_confirmation,
        order_pending,
        plan_run,
    )
    from particles.operations.version_guard import assert_store_schema_current

    if not estimate_only:
        # a refused extraction endpoint claims nothing, so every
        # snapshot stays PENDING with its attempt count unchanged.
        _refuse_if_extraction_refused()
        # Fail fast if ANTHROPIC_API_KEY is unset — otherwise we'd run through every
        # pending snapshot reporting the same auth error. An estimate needs no key:
        # it is read before anyone has decided to spend.
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

    if not estimate_only:
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
    # silent — the operator sees what was not extracted and why. An estimate
    # only previews it, and leaves the previewed snapshots out of the price.
    async with session_scope() as session:
        collapse = await collapse_superseded_pending(session, dry_run=estimate_only)
    collapse_line = collapse.summary()
    if collapse_line:
        typer.echo(collapse_line)
    collapsed = {snapshot_id for _, snapshot_id, _ in collapse.collapsed}

    async with session_scope() as session:
        result = await session.execute(
            select(
                SnapshotRow.entry_id,
                SnapshotRow.snapshot_id,
                CorpusEntryRow.tags_json,
                SnapshotRow.warc_record_type,
            )
            .where(SnapshotRow.extraction_status == ExtractionStatus.PENDING.value)
            .join(CorpusEntryRow, SnapshotRow.entry_id == CorpusEntryRow.entry_id)
            .order_by(CorpusEntryRow.created_at, SnapshotRow.captured_at)
        )
        listed = [
            (entry_id, snapshot_id, record_type == WarcRecordType.ABSENT.value)
            for entry_id, snapshot_id, tags_json, record_type in result.all()
            if (tag is None or entry_has_tag(tags_json, tag)) and snapshot_id not in collapsed
        ]
        # a source whose every call the model refused is not sent
        # again until the model, the content or the retry window changes.
        refused_held = (
            []
            if retry_refused
            else await held_snapshots(session, snapshot_ids=[s for _, s, _ in listed])
        )
        held_ids = {h.snapshot_id for h in refused_held}
        listed = [item for item in listed if item[1] not in held_ids]
        read_before = await entry_ids_ever_extracted(session, {e for e, _, _ in listed})
        # Edits of already-read sources first, then never-read ones, each in
        # deposit order. Absences go after the RESPONSE work, so
        # a renamed and edited note is read under its new entry before its old
        # one retires.
        pending = order_pending(
            [PendingSnapshot(e, s, absent=a, read_before=e in read_before) for e, s, a in listed]
        )
        # Counts for the no-pending message below. "No PENDING snapshots
        # found" alone is ambiguous after a failed run — an operator whose
        # previous extraction printed FAILED lines reads it as "my snapshots
        # vanished", when a concurrent runner usually just completed them.
        counts = await count_snapshots_by_extraction_status(session) if not pending else {}
        plan = await plan_run(session, pending) if pending else None

    if refused_held:
        typer.echo(
            f"Holding {len(refused_held)} snapshot(s) the extraction model refused "
            f"({refused_held[0].provider_model}); --retry-refused sends them, and "
            "`particles curate` lists them."
        )
    if not pending or plan is None:
        breakdown = ", ".join(
            f"{counts[s.value]} {s.value}" for s in ExtractionStatus if counts.get(s.value)
        )
        suffix = f" ({breakdown})" if breakdown else ""
        scope = f" tagged {tag}" if tag is not None else ""
        typer.echo(f"No PENDING snapshots{scope} found{suffix}.")
        return

    for line in _run_estimate_lines(plan, budget_usd):
        typer.echo(line)
    if estimate_only:
        typer.echo("--estimate: nothing was extracted.")
        return
    assert plan.total is not None
    verdict = decide_confirmation(
        calls=plan.total.calls,
        threshold=get_config().extraction.confirm_call_threshold,
        yes=yes,
        budget_usd=budget_usd,
        interactive=sys.stdin.isatty(),
    )
    if verdict == "refuse":
        typer.echo(
            f"Estimated LLM calls ({plan.total.calls}) exceed "
            f"extraction.confirm_call_threshold "
            f"({get_config().extraction.confirm_call_threshold}) and there is no terminal "
            "to ask. Re-run with --budget-usd <dollars> to cap the spend, or --yes to "
            "read everything.",
            err=True,
        )
        raise typer.Exit(2)
    if verdict == "ask" and not confirm(
        f"Extract {len(pending)} snapshot(s), about {plan.total.calls} LLM call(s)?",
        default=False,
    ):
        typer.echo("Nothing was extracted. --budget-usd <dollars> reads the newest first.")
        raise typer.Exit(2)

    typer.echo(f"Extracting {len(pending)} pending snapshot(s)…")
    failures = 0
    left_pending = 0
    gate = BudgetGate(budget_usd)
    budget_stopped = False
    attempted: set[str] = set()
    # one metered run around the whole pass, one EXTRACT_RUN record
    # and one usage line.
    async with _metered_run("all-pending") as meter:
        for index, item in enumerate(pending):
            entry_id, snapshot_id = item.entry_id, item.snapshot_id
            if not item.absent:
                # a snapshot starts only when the spend so far plus
                # its estimate fits the budget. The run never skips ahead to a
                # smaller one; absences cost nothing and are still processed.
                if budget_stopped:
                    continue
                gate.observe(meter.usage_so_far())
                if not gate.allows(
                    spent_usd=meter.usage_so_far().priced_cost_usd,
                    estimate_usd=plan.high_usd.get(snapshot_id),
                ):
                    budget_stopped = True
                    continue
            attempted.add(snapshot_id)
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
                    if outcome.absence_retired is not None:
                        typer.echo(
                            f"  {id_prefix}  source no longer found: "
                            f"{len(outcome.absence_retired)} claim(s) retired, no model call"
                        )
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
                    if outcome.failed_calls and outcome.refusal_recorded:
                        left_pending += 1
                        category = (
                            f" (category {outcome.refusal_category})"
                            if outcome.refusal_category
                            else ""
                        )
                        typer.echo(
                            f"  {id_prefix}  refused: the extraction model declined every "
                            f"call{category}, so nothing was written; later runs hold it "
                            "until the model or the content changes",
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
                    if outcome.revision_settled:
                        extras.append(_revision_settled_extra(outcome.revision_settled))
                    if outcome.restated:
                        extras.append(_restated_extra(outcome.restated))
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

    if budget_stopped and budget_usd is not None:
        spent = meter.llm_usage.priced_cost_usd if meter.llm_usage is not None else 0.0
        for line in _budget_stop_lines(plan, attempted, spent, budget_usd):
            typer.echo(line)
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


def _run_estimate_lines(plan: RunPlan, budget_usd: float | None) -> list[str]:
    """The estimate ``extract --all-pending`` prints before its first call."""
    assert plan.total is not None
    reads = sum(1 for item in plan.snapshots if not item.absent)
    absences = len(plan.snapshots) - reads
    line = (
        f"Estimate: {reads} snapshot(s) to read, about {plan.total.calls} LLM call(s), "
        f"{range_text(plan.total)}, {duration_text(plan.total.wall_seconds)}."
    )
    if absences:
        line += f" {absences} source(s) no longer found are processed with no model call."
    lines = [line]
    alt = plan.alternative
    if (
        alt is not None
        and alt.high_usd is not None
        and plan.total.high_usd is not None
        and alt.high_usd < plan.total.high_usd
    ):
        lines.append(
            f"  On {alt.model}: {range_text(alt, name_model=False)}; add --extraction-model "
            f"{plan.alternative_route} to use it for this run."
        )
    if budget_usd is not None:
        lines.append(
            f"  Budget: {usd(budget_usd)}. The run stops before a snapshot whose estimate "
            "would pass it, and what is left stays pending."
        )
    return lines


def _budget_stop_lines(
    plan: RunPlan, attempted: set[str], spent: float, budget_usd: float
) -> list[str]:
    """What a budget stop read, cost, and left."""
    count, chars, rest = plan.remaining(attempted)
    return [
        f"Budget reached: spent {usd(spent)} of {usd(budget_usd)}. {count} snapshot(s) "
        f"({chars:,} characters) left PENDING; finishing them is {range_text(rest)}.",
        "  They are not in `query` answers until read. `particles extract --all-pending` "
        "continues with them (add --budget-usd for the next part), and a scheduled "
        "`particles memory consolidate` reads up to consolidation.max_pending_entries a night.",
    ]


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


async def _resolve_local(
    entry_id: str, snapshot_id: str | None, *, force: bool = False
) -> tuple[str, str]:
    """Resolve an entry-ID / snapshot-ID prefix against the local store.

    Accepts the 8-char prefix form ``deposit`` / ``extract`` print, the way
    ``subjects show`` and the other verbs do. Local-only: ``extract_snapshot``
    itself only takes full UUIDs, and a remote engine resolves nothing for the
    laptop — remote mode requires full IDs.

    A named older snapshot of an entry recorded as no longer found is refused
    without ``force``: extracting it would bring a deleted
    note's claims back. So is any snapshot of a ``MUTABLE`` entry captured
    before its current generation: it would retire nothing the
    current version states and pay to extract claims it no longer makes.
    """
    from particles.core.schema import WarcRecordType
    from particles.corpus.store import (
        get_snapshot,
        list_absent_entry_ids,
        list_snapshots_for_entry,
        resolve_entry_id,
        resolve_snapshot_id,
    )
    from particles.ingest.generation import current_generation

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
            named = await get_snapshot(session, snapshot_id)
            if (
                not force
                and named is not None
                and named.warc_record_type is not WarcRecordType.ABSENT
                and entry_id in await list_absent_entry_ids(session, [entry_id])
            ):
                typer.echo(
                    f"Refusing: entry {entry_id[:8]}… is recorded as no longer found, and "
                    "extracting an older snapshot would bring back the claims its deletion "
                    "retired. Pass --force to extract it anyway.",
                    err=True,
                )
                raise typer.Exit(1)
            if (
                not force
                and named is not None
                and named.warc_record_type is WarcRecordType.RESPONSE
                and (await current_generation(session, entry_id, snapshot_id)).late
            ):
                typer.echo(
                    f"Refusing: snapshot {snapshot_id[:8]}… is older than entry "
                    f"{entry_id[:8]}…'s current version, so extracting it would retire "
                    "nothing the current version states and pay to extract claims it no "
                    "longer makes. Pass --force to extract it anyway.",
                    err=True,
                )
                raise typer.Exit(1)
    return entry_id, snapshot_id


def _refuse_if_extraction_refused() -> None:
    """Raise before any snapshot is claimed when the extraction endpoint is refused.

    The engine's own posture governs a remote run, so this is a local-mode
    check only. ``run()`` prints the refusal as one line.
    """
    from particles.llm import PurposeRefused, purpose_allowed
    from particles.llm.registry import purpose_endpoint

    if not purpose_allowed("extraction"):
        from particles.egress import current_policy

        host, port, _ = purpose_endpoint("extraction")
        raise PurposeRefused("extraction", host, port, current_policy().posture)


async def _extract(
    entry_id: str, snapshot_id: str | None, agent_id: str, *, force: bool = False
) -> tuple[str, list[Particle], list[Any], list[str], list[str], int, int, LLMUsage | None]:
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
        _refuse_if_extraction_refused()
        entry_id, resolved_snapshot = await _resolve_local(entry_id, snapshot_id, force=force)

    outcome = await backend.extract(entry_id, resolved_snapshot, agent_id=agent_id)
    return (
        outcome.entry_id,
        outcome.particles,
        outcome.page_stats,
        outcome.carry_forward_ids,
        outcome.suppressed_ids,
        outcome.revision_settled,
        outcome.restated,
        outcome.llm_usage,
    )
