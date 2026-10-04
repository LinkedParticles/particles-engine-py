# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""audit verb — the first-run memory audit.

``particles audit [PATH]`` harvests an agent-memory directory (or file) into
the user's real store, extracts it, and renders one census report — *"your
agent's memory contains 4 potential contradictions, 11 likely-duplicate
beliefs, 7 probably-stale facts."* Without ``PATH`` it re-audits the existing
store (no harvest). The same flow is the closing step of
``particles init claude-code`` via :func:`run_first_run_audit`.

Division of labour: the harvest reuses helpers
(``filter_memory_file_for_deposit`` sentinel strip, ``distill_transcript`` +
``redact_secrets``) and the URI scheme, so the audit and the
SessionEnd hook are mutually idempotent through corpus dedup; the census +
report live in ``particles.operations.audit``.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particles.operations.audit import AuditProgress, AuditReport

import typer

from particles.api.cli import app, run
from particles.api.cli._claude_code import (
    distill_transcript,
    filter_memory_file_for_deposit,
    is_claude_code_memory_dir,
    projection_enabled,
    redact_secrets,
    stray_memory_dirs,
    transcript_project_key,
)
from particles.api.cli._logging import configure_logging
from particles.api.cli._output import current_output
from particles.api.cli._progress import heartbeat_paused, progress_line, set_heartbeat_status
from particles.config import get_config
from particles.core.observer_scope import project_tag
from particles.db import session_scope
from particles.llm.usage import render_usage_line
from particles.secrets import get_anthropic_api_key_optional

_FORMATS = ("markdown", "json")
_SCOPES = ("harvested", "store")

# Exit codes, aligned with ``particles memory consolidate``: the
# report was written but the run did less than asked, or the run never started.
EXIT_INCOMPLETE = 1
EXIT_NOT_STARTED = 2

#: Off a terminal, probe progress prints at most this many lines.
_PROBE_LINES_OFF_TTY = 10


@app.command("audit")
def audit_cmd(
    path: Path | None = typer.Argument(
        None,
        help=(
            "Memory directory (or single file) to harvest + audit. Omit to re-audit "
            "the existing store without harvesting."
        ),
    ),
    transcripts: Path | None = typer.Option(
        None,
        "--transcripts",
        help=(
            "Opt-in: also harvest session transcripts (*.jsonl) from DIR, newest "
            "first, capped at audit.transcript_max_entries (--max-entries overrides)."
        ),
    ),
    max_entries: int | None = typer.Option(
        None,
        "--max-entries",
        help=(
            "Cap harvested entries (default: audit.transcript_max_entries for "
            "transcripts; unlimited for memory files)."
        ),
    ),
    estimate: bool = typer.Option(
        False,
        "--estimate",
        help=(
            "Print the cost estimate (calls, tokens, a dollar range at the configured "
            "model's list price, and the expected wall time) and exit: no deposit, no LLM "
            "call. With --format json "
            "the estimate is printed as JSON."
        ),
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost-confirmation prompt."),
    judge: bool = typer.Option(
        False,
        "--judge",
        help="LLM-judge duplicate pairs (verified duplicates) instead of REPORT-mode candidates.",
    ),
    scope: str | None = typer.Option(
        None,
        "--scope",
        help=(
            "Semantic-finding scope (contradiction probe + duplicate scan): "
            "'harvested' (default with PATH; headline counts only pairs touching this "
            "harvest's beliefs; the store-wide duplicate total is still disclosed) or "
            "'store' (the whole store; the re-audit default)."
        ),
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Write the Markdown report to FILE as well."
    ),
    format_: str = typer.Option(
        "markdown", "--format", help="Terminal format: markdown (default) or json."
    ),
    store: str = typer.Option(
        "default", "--store", help="Audit a named store (default: the default store)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Audit an agent-memory directory: harvest, extract, and report the rot census.

    The line under the header says how many files extracted in full, how
    many were cut short at the output-token limit, and how many produced
    nothing, naming the last.

    Exit codes: 0 means the audit completed. 1 means the report was written
    but the audit is incomplete: a harvested file produced no beliefs because
    its extraction failed (it stays pending, and re-running the same command
    retries it), or the contradiction check was skipped (no API key, or the
    LLM became unavailable, for example an exhausted credit balance). The
    findings in the report still stand. 2 means the audit did not start
    (invalid options, no API key for a harvest, a missing path, nothing to
    audit, or the cost confirmation was declined).
    """
    configure_logging(verbose, debug)
    if format_ not in _FORMATS:
        typer.echo(f"Error: --format must be one of: {', '.join(_FORMATS)}.", err=True)
        raise typer.Exit(EXIT_NOT_STARTED)
    if scope is not None and scope not in _SCOPES:
        typer.echo(f"Error: --scope must be one of: {', '.join(_SCOPES)}.", err=True)
        raise typer.Exit(EXIT_NOT_STARTED)
    harvesting = path is not None or transcripts is not None
    if scope == "harvested" and not harvesting:
        typer.echo(
            "Error: --scope harvested needs a harvest (PATH or --transcripts) — a "
            "re-audit has no harvested entries to scope to. Omit --scope for the "
            "store-wide re-audit probe.",
            err=True,
        )
        raise typer.Exit(EXIT_NOT_STARTED)
    report = run(
        _audit_impl(
            paths=[path] if path is not None else [],
            transcripts_dir=transcripts,
            max_entries=max_entries,
            estimate_only=estimate,
            yes=yes,
            judge=judge,
            scope=scope,
            output=output,
            fmt=format_,
            store=store,
        )
    )
    if report is not None and not report.complete:
        typer.echo(_incomplete_notice(report, [path] if path is not None else None), err=True)
        raise typer.Exit(EXIT_INCOMPLETE)


def _incomplete_notice(report: AuditReport, paths: list[Path] | None = None) -> str:
    """The line an incomplete audit ends with: what was skipped, how to finish it.

    ``paths`` are the harvested inputs. A file that was not extracted is still
    PENDING, so re-running the same harvest retries it; a re-audit without the
    paths would not.
    """
    store_flag = "" if report.store == "default" else f" --store {report.store}"
    notices: list[str] = []
    if report.extraction_failures:
        n = report.extraction_failures
        noun = "file" if n == 1 else "files"
        rerun = "particles audit" + "".join(f" {p}" for p in paths or []) + store_flag
        notices.append(
            f"audit incomplete: {n} {noun} not extracted, so the census does not "
            f"cover them. Run `{rerun}` to retry them."
        )
    if report.semantic_skipped:
        reason = report.semantic_skip_reason or "LLM unavailable"
        when = (
            "once ANTHROPIC_API_KEY is set"
            if reason == "no API key"
            else "once the LLM is available again"
        )
        finish = "particles audit" + store_flag
        notices.append(
            f"audit incomplete: contradiction check skipped, {reason}. "
            f"Run `{finish}` {when} to finish it."
        )
    return "\n".join(notices)


def _make_progress_renderer() -> Callable[[AuditProgress], None]:
    """Progress for the long extraction phase and the contradiction probe.

    A real memory directory is 10–20 minutes of sequential LLM calls; without
    per-entry feedback the activation-moment audit is indistinguishable from a
    hang (owner-reported on the first dogfood run, 2026-07-11). The Engine
    emits ``AuditProgress`` events; this closure renders them.

    Extraction keeps one line per entry: each names the file and what it
    yielded, which is worth scrollback. The probe is up to
    ``audit.max_contradiction_probes`` identical steps, so it renders in place
    on the heartbeat line when one is running, and otherwise as at most
    ``_PROBE_LINES_OFF_TTY`` milestone lines. Every line goes through
    ``progress_line`` so none lands on the heartbeat's open line.
    """
    start = time.monotonic()
    in_place = get_config().cli.heartbeat_seconds > 0 and current_output().show_progress()

    def _elapsed() -> str:
        minutes, seconds = divmod(int(time.monotonic() - start), 60)
        return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"

    def _render(event: AuditProgress) -> None:
        if event.phase == "census":
            progress_line(f"  Scanning findings — {event.label}… ({_elapsed()} elapsed)")
        elif event.phase == "probe":
            finished = event.done >= event.total
            if in_place:
                set_heartbeat_status(
                    None if finished else f"{event.label} {event.done}/{event.total}"
                )
                return
            step = max(1, math.ceil(event.total / _PROBE_LINES_OFF_TTY))
            if finished or event.done % step == 0:
                progress_line(
                    f"  [{event.done}/{event.total}] {event.label}… ({_elapsed()} elapsed)"
                )
        elif event.failed:
            progress_line(
                f"  [{event.done}/{event.total}] {event.label} → extraction failed "
                f"(disclosed in the report; {_elapsed()} elapsed)"
            )
        elif event.partial:
            noun = "belief" if event.particles == 1 else "beliefs"
            progress_line(
                f"  [{event.done}/{event.total}] {event.label} → "
                f"{event.particles} {noun}, reply cut short at the output-token limit "
                f"(disclosed in the report; {_elapsed()} elapsed)"
            )
        else:
            noun = "belief" if event.particles == 1 else "beliefs"
            progress_line(
                f"  [{event.done}/{event.total}] {event.label} → "
                f"{event.particles} {noun} ({_elapsed()} elapsed)"
            )

    return _render


def run_first_run_audit(store: str, only_memory_dir: Path | None = None) -> None:
    """The ``init claude-code`` closing step.

    Audits every Claude Code memory directory under ``~/.claude/projects/``
    into the freshly registered store — or, for a ``--project`` install, only
    ``only_memory_dir``, that project's own — with the same
    estimate/confirm gate as the standalone verb. Raises ``typer.Exit`` on
    refusal/abort — the caller (init) catches it so a declined audit never fails
    the install. An *incomplete* audit does not raise: its report is already
    printed and the install succeeded, so it ends with the notice naming what
    was skipped and the command that finishes it.
    """
    root = Path.home() / ".claude" / "projects"
    # Linked worktrees' memory directories are this SDK's own stray output,
    # never Claude Code's; auditing them would ingest the projection.
    strays = set(stray_memory_dirs(root))
    dirs = (
        sorted(p for p in root.glob("*/memory") if p.is_dir() and p not in strays)
        if root.is_dir()
        else []
    )
    if only_memory_dir is not None:
        dirs = [d for d in dirs if d == only_memory_dir]
    if not dirs:
        typer.echo(
            "\nFirst-run memory audit: no memory directories found under "
            f"{root} — nothing to audit yet. The SessionEnd hook will harvest "
            "future sessions; run `particles audit <dir>` any time."
        )
        return
    noun = "directory" if len(dirs) == 1 else "directories"
    typer.echo(f"\nFirst-run memory audit over {len(dirs)} memory {noun}…")
    report = run(
        _audit_impl(
            paths=list(dirs),
            transcripts_dir=None,
            max_entries=None,
            estimate_only=False,
            yes=False,
            judge=False,
            scope=None,
            output=None,
            fmt="markdown",
            store=store,
        )
    )
    if report is not None and not report.complete:
        typer.echo(_incomplete_notice(report, list(dirs)))


# ---------------------------------------------------------------------------
# Harvest plan — reads files only; nothing touches the store
# ---------------------------------------------------------------------------


@dataclass
class _PlannedDeposit:
    """One deposit the audit will perform, mirroring the harvest shape."""

    uri_r: str
    text: str
    source_type: str
    mutability: str
    tags: list[str]
    content_published_at: datetime | None


@dataclass
class _HarvestPlan:
    deposits: list[_PlannedDeposit] = field(default_factory=list)
    memory_files: int = 0
    transcripts: int = 0
    # (memory_dir, raw MEMORY.md text) pairs for the projection cycle
    # that ends a successful harvest+extract pass.
    memory_dirs: list[tuple[Path, str | None]] = field(default_factory=list)


def _project_tag(memory_dir: Path) -> list[str]:
    """The ``project:<key>`` tag the SessionEnd hook stamps, when derivable.

    Only for a memory directory Claude Code itself reads, whose parent is the
    ``~/.claude/projects/<key>`` directory the key names. Any other directory
    named ``memory`` (a copy under ``/tmp/rung1``) would otherwise be tagged
    with its parent's name, which two unrelated directories can share, and
    that tag decides which project observes the beliefs. Such a
    harvest is left unattributed, which the observer-scope read treats as
    in view for no project rather than for every project.
    """
    if is_claude_code_memory_dir(memory_dir):
        return [project_tag(memory_dir.resolve().parent.name)]
    return []


def _plan_memory_file(
    md: Path, tags: list[str], memory_dir: Path | None = None
) -> _PlannedDeposit | None:
    """Filter + shape one memory-file deposit.

    ``content_published_at`` uses the canonical date ladder (leading date
    line › file mtime) rather than bare mtime, so a dated memory file carries
    its content date and the age-discount lens sees the real age.

    ``memory_dir`` is the memory directory ``md`` was found under, so its
    projected region is compared with that project's own render; a
    file audited by bare path has none.
    """
    # The precedence ladder deposit_file uses; imported from its home
    # module (the audit harvests via deposit_text_versioned, which takes the
    # resolved date rather than re-deriving it).
    from particles.corpus.deposit import _resolve_content_published_at

    raw = md.read_text(encoding="utf-8", errors="replace")
    text = filter_memory_file_for_deposit(raw, memory_dir=memory_dir, source_dir=md.parent)
    if not text.strip():
        return None
    return _PlannedDeposit(
        uri_r=md.resolve().as_uri(),
        text=text,
        source_type="LOCAL_MARKDOWN",
        mutability="MUTABLE",
        tags=tags,
        content_published_at=_resolve_content_published_at(md, raw.encode("utf-8"), None),
    )


def _plan_transcript(jsonl: Path) -> _PlannedDeposit | None:
    """Distill + redact one session transcript (same URI identity)."""
    session_id = jsonl.stem
    text = distill_transcript(jsonl.read_text(encoding="utf-8", errors="replace"), session_id)
    if not text.strip():
        return None
    return _PlannedDeposit(
        uri_r=f"claude-code://session/{session_id}",
        text=redact_secrets(text),
        source_type="CONVERSATION",
        mutability="APPEND_ONLY",
        # The same project key the SessionEnd hook stamps, so a transcript
        # audited before any hook saw it is attributed too.
        tags=[
            "claude-code",
            f"session:{session_id}",
            "audit",
            project_tag(transcript_project_key(jsonl)),
        ],
        content_published_at=None,
    )


def build_harvest_plan(
    paths: list[Path],
    transcripts_dir: Path | None,
    max_entries: int | None,
) -> _HarvestPlan:
    """Walk the inputs into a deposit plan without touching the store.

    Memory files are unlimited by default (they are the distilled, high-signal
    input); transcripts are capped at ``audit.transcript_max_entries`` newest
    first — ``max_entries`` overrides both.
    """
    plan = _HarvestPlan()

    memory_files: list[tuple[Path, list[str], Path | None]] = []
    for path in paths:
        if path.is_dir():
            tags = ["claude-code", "memory-file", *_project_tag(path)]
            memory_files.extend((md, tags, path) for md in sorted(path.rglob("*.md")))
            memory_md = path / "MEMORY.md"
            plan.memory_dirs.append(
                (
                    path,
                    memory_md.read_text(encoding="utf-8", errors="replace")
                    if memory_md.is_file()
                    else None,
                )
            )
        elif path.suffix == ".jsonl":
            planned = _plan_transcript(path)
            if planned is not None:
                plan.deposits.append(planned)
                plan.transcripts += 1
        else:
            memory_files.append((path, ["claude-code", "memory-file"], None))

    if max_entries is not None:
        memory_files = memory_files[:max_entries]
    for md, tags, memory_dir in memory_files:
        planned = _plan_memory_file(md, tags, memory_dir)
        if planned is not None:
            plan.deposits.append(planned)
            plan.memory_files += 1

    if transcripts_dir is not None:
        cap = max_entries if max_entries is not None else get_config().audit.transcript_max_entries
        candidates = [p for p in transcripts_dir.glob("*.jsonl") if p.is_file()]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for jsonl in candidates[:cap]:
            planned = _plan_transcript(jsonl)
            if planned is not None:
                plan.deposits.append(planned)
                plan.transcripts += 1

    return plan


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def _refuse_without_key() -> None:
    """harvest+extract with no key refuses BEFORE touching the store."""
    typer.echo(
        "Error: ANTHROPIC_API_KEY is not set. Extraction is the audit's substance — "
        "a structural-only audit of an empty store would report nothing — so nothing "
        "was deposited.\n"
        "Fix: export ANTHROPIC_API_KEY=sk-... and re-run.",
        err=True,
    )
    raise typer.Exit(EXIT_NOT_STARTED)


async def _perform_deposits(store: str, plan: _HarvestPlan) -> tuple[list[str], int, int]:
    """Deposit the plan; returns (entry_ids, new, unchanged). Corpus dedup makes
    this idempotent against the SessionEnd harvest."""
    from particles.core.schema import Mutability

    # Deferred import: tests patch ``particles.operations.deposit
    # .deposit_text_versioned`` (see tests/AGENTS.md § Mocking strategy);
    # a module-top import would freeze the binding past the patch.
    from particles.operations.deposit import deposit_text_versioned

    entry_ids: list[str] = []
    new = 0
    unchanged = 0
    async with session_scope(store, write=True) as session:
        for planned in plan.deposits:
            entry_id, _snapshot_id, was_unchanged = await deposit_text_versioned(
                session,
                text=planned.text,
                uri_r=planned.uri_r,
                source_type=planned.source_type,
                mutability=Mutability(planned.mutability),
                tags=planned.tags,
                deposited_by="audit",
                content_published_at=planned.content_published_at,
            )
            entry_ids.append(entry_id)
            if was_unchanged:
                unchanged += 1
            else:
                new += 1
        await session.commit()
    return entry_ids, new, unchanged


async def _run_projection_cycles(store: str, plan: _HarvestPlan) -> bool:
    """End the pass with the first render: the audited store projects
    straight back into each harvested MEMORY.md (render-after-successful-harvest).

    Only into a directory Claude Code itself reads, ``~/.claude/projects/<key>/memory``
    (the set ``init claude-code`` audits). A verb named ``audit`` pointed at any
    other directory, even one named ``memory``, leaves it untouched: the
    projection writes MEMORY.md there and keeps per-project state keyed on the
    parent directory's name, which two unrelated directories can share.
    """
    if not projection_enabled():
        return False
    from particles.api.cli._memory_projection import run_projection_cycle

    rendered = False
    for memory_dir, raw_text in plan.memory_dirs:
        if not is_claude_code_memory_dir(memory_dir):
            continue
        outcome = await run_projection_cycle(store, memory_dir, raw_text)
        if outcome.get("outcome") in ("rendered", "created"):
            rendered = True
    return rendered


async def _audit_impl(
    *,
    paths: list[Path],
    transcripts_dir: Path | None,
    max_entries: int | None,
    estimate_only: bool,
    yes: bool,
    judge: bool,
    scope: str | None,
    output: Path | None,
    fmt: str,
    store: str,
) -> AuditReport | None:
    """Run the flow and print the report; returns it, or None when nothing was audited."""
    from particles.api.client import get_backend

    if get_backend().remote:
        typer.echo(
            "Error: `particles audit` is a local, interactive, cost-gated flow "
            " and is not available against a remote engine. Run it "
            "on the machine that holds the store.",
            err=True,
        )
        raise typer.Exit(EXIT_NOT_STARTED)

    # Deferred import: the operation pulls the curation/lint stack — and tests
    # patch ``particles.operations.audit.run_memory_audit`` at call time
    # (tests/AGENTS.md § Mocking strategy).
    from particles.operations.audit import (
        cost_summary,
        estimate_extraction,
        render_audit_report,
        render_estimate,
        run_memory_audit,
        time_summary,
    )

    on_progress = _make_progress_renderer()

    have_key = get_anthropic_api_key_optional() is not None
    harvesting = bool(paths) or transcripts_dir is not None

    if harvesting:
        # §7: refuse before touching the store — extraction is the substance.
        if not have_key:
            _refuse_without_key()

        for path in [*paths, *([transcripts_dir] if transcripts_dir else [])]:
            if not path.exists():
                typer.echo(f"Error: {path} does not exist.", err=True)
                raise typer.Exit(EXIT_NOT_STARTED)

        plan = build_harvest_plan(paths, transcripts_dir, max_entries)
        if not plan.deposits:
            typer.echo("No auditable content found (no non-empty *.md or *.jsonl files).")
            raise typer.Exit(EXIT_NOT_STARTED)

        # §4: estimate ALWAYS printed before extraction.
        cost = estimate_extraction([len(d.text) for d in plan.deposits])
        if estimate_only and fmt == "json":
            # stdout stays one parseable document; the note goes to stderr.
            typer.echo(cost.model_dump_json(indent=2))
            typer.echo("--estimate: nothing was deposited.", err=True)
            return None
        typer.echo(render_estimate(cost))
        if estimate_only:
            typer.echo("--estimate: nothing was deposited.")
            return None
        threshold = get_config().audit.confirm_call_threshold
        if cost.estimated_llm_calls > threshold and not yes:
            if not sys.stdin.isatty():
                typer.echo(
                    f"Estimated LLM calls ({cost.estimated_llm_calls}) exceed "
                    f"audit.confirm_call_threshold ({threshold}) and no --yes was "
                    "given in a non-interactive run. Nothing was deposited.",
                    err=True,
                )
                raise typer.Exit(EXIT_NOT_STARTED)
            dollars = cost_summary(cost) or "unpriced model"
            duration = time_summary(cost)
            if duration is not None:
                dollars += f", {duration}"
            if not typer.confirm(
                f"Proceed with ~{cost.estimated_llm_calls} extraction LLM calls ({dollars})?"
            ):
                typer.echo("Aborted — nothing was deposited.")
                raise typer.Exit(EXIT_NOT_STARTED)

        entry_ids, new, unchanged = await _perform_deposits(store, plan)
        typer.echo(
            f"Harvested {len(entry_ids)} entr{'y' if len(entry_ids) == 1 else 'ies'} "
            f"({new} new, {unchanged} unchanged). Extracting…"
        )
        async with session_scope(store) as session:
            report = await run_memory_audit(
                session,
                store=store,
                files_audited=plan.memory_files,
                transcripts_audited=plan.transcripts,
                harvested_new=new,
                harvested_unchanged=unchanged,
                harvested_entry_ids=entry_ids,
                semantic=True,
                judge=judge,
                estimate=cost,
                on_progress=on_progress,
                # Proposed: a harvest run probes this harvest's beliefs
                # by default; --scope store opts into the store-wide set.
                contradiction_scope="store" if scope == "store" else "harvested",
            )
            await session.commit()
        report.projection_rendered = await _run_projection_cycles(store, plan)
    else:
        if estimate_only:
            typer.echo(
                "--estimate applies to a harvest; a re-audit (no PATH) deposits and "
                "extracts nothing."
            )
            return None
        # §7 re-audit degradation: structural finders + REPORT-mode duplicates
        # run without an LLM; the contradiction probe is skipped WITH a line.
        async with session_scope(store) as session:
            report = await run_memory_audit(
                session,
                store=store,
                semantic=have_key,
                judge=judge and have_key,
                semantic_skip_reason=None if have_key else "no API key",
                on_progress=on_progress,
                # A re-audit is the deliberate whole-store census (--scope
                # harvested is rejected up front — nothing was harvested).
                contradiction_scope="store",
            )
            await session.commit()

    rendered = render_audit_report(report)
    # The heartbeat's in-place line is still open on the terminal; the report
    # must start on a fresh line, not after "… contradiction probe 185/200".
    with heartbeat_paused():
        if fmt == "json":
            typer.echo(report.model_dump_json(indent=2))
            # stdout stays one parseable document (``llm_usage`` is in it); the
            # human line the markdown report ends with goes to stderr.
            if report.llm_usage is not None:
                typer.echo(render_usage_line(report.llm_usage), err=True)
        else:
            typer.echo(rendered)
    if output is not None:
        from particles.render.markdown import atomic_write_text

        atomic_write_text(output, rendered)
        typer.echo(f"Report written to {output}.")
    return report
