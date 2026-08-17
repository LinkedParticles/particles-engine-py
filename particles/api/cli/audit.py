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

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particles.operations.audit import AuditProgress

import typer

from particles.api.cli import app, run
from particles.api.cli._claude_code import (
    distill_transcript,
    filter_memory_file_for_deposit,
    projection_enabled,
    redact_secrets,
)
from particles.api.cli._logging import configure_logging
from particles.config import get_config
from particles.db import session_scope
from particles.secrets import get_anthropic_api_key_optional

_FORMATS = ("markdown", "json")
_SCOPES = ("harvested", "store")


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
        help="Print the extraction cost estimate and exit — no deposit, no LLM call.",
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
            "'harvested' (default with PATH — headline counts only pairs touching this "
            "harvest's beliefs; the store-wide duplicate total is still disclosed) or "
            "'store' (the whole store; the re-audit default)."
        ),
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Also write the Markdown report to FILE."
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
    """Audit an agent-memory directory: harvest, extract, and report the rot census."""
    configure_logging(verbose, debug)
    if format_ not in _FORMATS:
        typer.echo(f"Error: --format must be one of: {', '.join(_FORMATS)}.", err=True)
        raise typer.Exit(1)
    if scope is not None and scope not in _SCOPES:
        typer.echo(f"Error: --scope must be one of: {', '.join(_SCOPES)}.", err=True)
        raise typer.Exit(1)
    harvesting = path is not None or transcripts is not None
    if scope == "harvested" and not harvesting:
        typer.echo(
            "Error: --scope harvested needs a harvest (PATH or --transcripts) — a "
            "re-audit has no harvested entries to scope to. Omit --scope for the "
            "store-wide re-audit probe.",
            err=True,
        )
        raise typer.Exit(1)
    run(
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


def _make_progress_renderer() -> Callable[[AuditProgress], None]:
    """Progress lines for the long extraction phase.

    A real memory directory is 10–20 minutes of sequential LLM calls; without
    per-entry feedback the activation-moment audit is indistinguishable from a
    hang (owner-reported on the first dogfood run, 2026-07-11). The Engine
    emits ``AuditProgress`` events; this closure renders them.
    """
    start = time.monotonic()

    def _elapsed() -> str:
        minutes, seconds = divmod(int(time.monotonic() - start), 60)
        return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"

    def _render(event: AuditProgress) -> None:
        if event.phase == "census":
            typer.echo(f"  Scanning findings — {event.label}… ({_elapsed()} elapsed)")
        elif event.phase == "probe":
            typer.echo(f"  [{event.done}/{event.total}] {event.label}… ({_elapsed()} elapsed)")
        elif event.failed:
            typer.echo(
                f"  [{event.done}/{event.total}] {event.label} → extraction failed "
                f"(disclosed in the report; {_elapsed()} elapsed)"
            )
        else:
            noun = "belief" if event.particles == 1 else "beliefs"
            typer.echo(
                f"  [{event.done}/{event.total}] {event.label} → "
                f"{event.particles} {noun} ({_elapsed()} elapsed)"
            )

    return _render


def run_first_run_audit(store: str) -> None:
    """The ``init claude-code`` closing step.

    Audits every Claude Code memory directory under ``~/.claude/projects/``
    into the freshly registered store, with the same estimate/confirm gate as
    the standalone verb. Raises ``typer.Exit`` on refusal/abort — the caller
    (init) catches it so a declined audit never fails the install.
    """
    root = Path.home() / ".claude" / "projects"
    dirs = sorted(p for p in root.glob("*/memory") if p.is_dir()) if root.is_dir() else []
    if not dirs:
        typer.echo(
            "\nFirst-run memory audit: no memory directories found under "
            f"{root} — nothing to audit yet. The SessionEnd hook will harvest "
            "future sessions; run `particles audit <dir>` any time."
        )
        return
    noun = "directory" if len(dirs) == 1 else "directories"
    typer.echo(f"\nFirst-run memory audit over {len(dirs)} memory {noun}…")
    run(
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
    """The ``project:<slug>`` tag the SessionEnd hook stamps, when derivable."""
    if memory_dir.name == "memory" and memory_dir.parent.name:
        return [f"project:{memory_dir.parent.name}"]
    return []


def _plan_memory_file(md: Path, tags: list[str]) -> _PlannedDeposit | None:
    """Filter + shape one memory-file deposit.

    ``content_published_at`` uses the canonical date ladder (leading date
    line › file mtime) rather than bare mtime, so a dated memory file carries
    its content date and the age-discount lens sees the real age.
    """
    # The precedence ladder deposit_file uses; imported from its home
    # module (the audit harvests via deposit_text_versioned, which takes the
    # resolved date rather than re-deriving it).
    from particles.corpus.deposit import _resolve_content_published_at

    raw = md.read_text(encoding="utf-8", errors="replace")
    text = filter_memory_file_for_deposit(raw)
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
        tags=["claude-code", f"session:{session_id}", "audit"],
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

    memory_files: list[tuple[Path, list[str]]] = []
    for path in paths:
        if path.is_dir():
            tags = ["claude-code", "memory-file", *_project_tag(path)]
            memory_files.extend((md, tags) for md in sorted(path.rglob("*.md")))
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
            memory_files.append((path, ["claude-code", "memory-file"]))

    if max_entries is not None:
        memory_files = memory_files[:max_entries]
    for md, tags in memory_files:
        planned = _plan_memory_file(md, tags)
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
    raise typer.Exit(1)


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
    straight back into each harvested MEMORY.md (render-after-successful-harvest)."""
    if not projection_enabled():
        return False
    from particles.api.cli._memory_projection import run_projection_cycle

    rendered = False
    for memory_dir, raw_text in plan.memory_dirs:
        if raw_text is None and memory_dir.name != "memory":
            # No MEMORY.md was harvested and this is not a Claude-shaped memory
            # dir — don't mint a MEMORY.md into an arbitrary notes directory.
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
) -> None:
    from particles.api.client import get_backend

    if get_backend().remote:
        typer.echo(
            "Error: `particles audit` is a local, interactive, cost-gated flow "
            " and is not available against a remote engine. Run it "
            "on the machine that holds the store.",
            err=True,
        )
        raise typer.Exit(1)

    # Deferred import: the operation pulls the curation/lint stack — and tests
    # patch ``particles.operations.audit.run_memory_audit`` at call time
    # (tests/AGENTS.md § Mocking strategy).
    from particles.operations.audit import (
        estimate_extraction,
        render_audit_report,
        render_estimate,
        run_memory_audit,
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
                raise typer.Exit(1)

        plan = build_harvest_plan(paths, transcripts_dir, max_entries)
        if not plan.deposits:
            typer.echo("No auditable content found (no non-empty *.md or *.jsonl files).")
            raise typer.Exit(1)

        # §4: estimate ALWAYS printed before extraction.
        cost = estimate_extraction([len(d.text) for d in plan.deposits])
        typer.echo(render_estimate(cost))
        if estimate_only:
            typer.echo("--estimate: nothing was deposited.")
            return
        threshold = get_config().audit.confirm_call_threshold
        if cost.estimated_llm_calls > threshold and not yes:
            if not sys.stdin.isatty():
                typer.echo(
                    f"Estimated LLM calls ({cost.estimated_llm_calls}) exceed "
                    f"audit.confirm_call_threshold ({threshold}) and no --yes was "
                    "given in a non-interactive run. Nothing was deposited.",
                    err=True,
                )
                raise typer.Exit(1)
            if not typer.confirm(f"Proceed with ~{cost.estimated_llm_calls} extraction LLM calls?"):
                typer.echo("Aborted — nothing was deposited.")
                raise typer.Exit(1)

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
            return
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
    if fmt == "json":
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(rendered)
    if output is not None:
        from particles.render.markdown import atomic_write_text

        atomic_write_text(output, rendered)
        typer.echo(f"Report written to {output}.")
