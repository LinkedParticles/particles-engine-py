"""hook sub-Typer — the machine-facing Claude Code lifecycle verbs.

``particles init claude-code`` installs these into Claude Code's settings;
Claude Code runs them on lifecycle events with a JSON payload on stdin:

  hook session-start  — push the store's memory digest into the new session's
                        context, gated by the sources-
                        trailer freshness check when the MEMORY.md projection
                        is enabled
  hook session-end    — harvest the session transcript + memory files into the
                        corpus, with the level-triggered
                        catch-up sweep (§3c), then re-render + splice the
                        MEMORY.md projected region (§6/§7)
  hook log            — print recent hook-log entries (§6)

The group is kept **visible**, not hidden: ``particles hook session-start
< sample.json`` is the debug loop.

**Contract: the hook verbs must degrade to exit 0 with no output on ANY
failure** (§2/§3) — a memory outage costs the user an empty digest or
a skipped harvest, never a hung or noisy session. They therefore do *not* use
the shared ``run()`` helper (its operational-error translation raises
``typer.Exit(1)``); every failure is caught, logged to the hook log, and
swallowed. Exit 2 is never used.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from particles.api.cli import app
from particles.api.cli._claude_code import (
    append_hook_log,
    distill_transcript,
    filter_memory_file_for_deposit,
    hook_log_path,
    projection_enabled,
    read_hook_log_tail,
    redact_secrets,
    truncate_on_line_boundary,
)
from particles.api.cli._memory_projection import DigestDecision
from particles.config import get_config
from particles.corpus.rule_sources import refresh_policy_for

if TYPE_CHECKING:
    from particles.api.client.base import TextDepositOutcome

log = logging.getLogger(__name__)

hook_app = typer.Typer(
    help=(
        "Machine-facing Claude Code lifecycle hooks. "
        "Reads the hook JSON from stdin; degrades to exit 0 on any failure."
    ),
    no_args_is_help=True,
)
app.add_typer(hook_app, name="hook")

_HookImpl = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[str | None]]


# ---------------------------------------------------------------------------
# Guard rail: never fail the session
# ---------------------------------------------------------------------------


def _run_hook(event: str, store: str, impl: _HookImpl) -> None:
    """Run a hook body under the degrade-to-nothing contract.

    ``impl(store, payload, record)`` is an async callable returning the string
    to print on stdout (or ``None`` for no output). Any failure — bad stdin,
    store missing, engine unreachable, ``WriteLockTimeout``, or the
    ``claude_code.hook_deadline_seconds`` internal deadline — is logged to the
    hook log and swallowed; the process exits 0 either way.
    """
    started = time.monotonic()
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        "store": store,
    }
    output: str | None = None
    try:
        payload = _read_stdin_json()
        record["session_id"] = payload.get("session_id")
        record["source"] = payload.get("source") or payload.get("reason")
        deadline = get_config().claude_code.hook_deadline_seconds
        output = asyncio.run(asyncio.wait_for(impl(store, payload, record), timeout=deadline))
        record["outcome"] = "ok"
    except Exception as exc:  # noqa: BLE001 — the degrade-to-nothing contract
        record["outcome"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        if _looks_like_uninitialized_store(exc):
            # The single most common silent failure: the hook ran
            # from a directory where the store's DB does not resolve. Name the
            # path + cwd so `particles hook log` is actionable, not a raw
            # OperationalError. `particles hook doctor` is the deeper check.
            record["hint"] = (
                f"store {store!r} not initialized at {_resolved_store_location(store)} "
                f"(cwd={os.getcwd()}); run `particles hook doctor --store {store}`"
            )
    finally:
        record["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
        append_hook_log(record)
    if output:
        typer.echo(output)
    # Fall through: exit 0. Exit 2 is never used.


def _read_stdin_json() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    loaded = json.loads(raw)
    return loaded if isinstance(loaded, dict) else {}


# ---------------------------------------------------------------------------
# Store-resolution diagnostics
# ---------------------------------------------------------------------------


def _looks_like_uninitialized_store(exc: BaseException) -> bool:
    """True when ``exc`` is the "store DB has no tables / does not resolve" shape."""
    from sqlalchemy.exc import OperationalError

    if not isinstance(exc, OperationalError):
        return False
    message = str(exc).lower()
    return "no such table" in message or "does not exist" in message


def _resolve_store_dsn(store: str) -> str | None:
    """The DSN the ``store`` handle resolves to under the *current* config, or None.

    Mirrors ``particles.db._store_dsn`` but never raises — an unknown named
    store (not declared in ``storage.stores``) returns ``None`` rather than
    ``KeyError``, so the diagnostic can say "not declared" instead of blowing up.
    """
    from particles.db import DEFAULT_STORE

    storage = get_config().storage
    if store == DEFAULT_STORE:
        return storage.database_url
    return storage.stores.get(store)


def _resolved_store_location(store: str) -> str:
    """A human-readable location for the store DB (SQLite file path or DSN)."""
    dsn = _resolve_store_dsn(store)
    if dsn is None:
        return f"<store {store!r} not declared in storage.stores>"
    if dsn.startswith("sqlite"):
        from particles.api.cli._claude_code import absolutize_sqlite_dsn

        path = re.sub(r"^sqlite[^:]*:///+", "/", absolutize_sqlite_dsn(dsn))
        return path
    return dsn


# ---------------------------------------------------------------------------
# session-start — the digest push
# ---------------------------------------------------------------------------


@hook_app.command("session-start")
def session_start_cmd(
    store: str = typer.Option(..., "--store", help="Memory store whose digest is pushed."),
) -> None:
    """SessionStart hook: push the memory digest into the session's context."""
    _run_hook("session-start", store, _session_start)


async def _session_start(store: str, payload: dict[str, Any], record: dict[str, Any]) -> str | None:
    source = str(payload.get("source") or "startup")
    if source == "resume":
        # A resumed session replays its prior context, digest included;
        # re-injecting would duplicate it. startup / clear / compact (context
        # rebuilt) all get a fresh push. The source test lives here, not in a
        # settings matcher, so the policy upgrades with the SDK.
        record["skipped"] = "resume"
        return None

    decision = await _digest_decision(store, payload)
    if decision.action == "skip":
        # The MEMORY.md projected region the harness just loaded IS the
        # current view (trailer fingerprint matched) — injecting the digest
        # would duplicate it.
        record["skipped"] = "projection-current"
        return None

    if decision.action == "diff" and decision.content is not None:
        # Mismatch: the store moved since the last render — top up with only
        # the difference (new / changed / newly-contested lines).
        digest = decision.content
        record["digest_mode"] = "projection-diff"
    else:
        from particles.api.client import get_backend

        digest = await get_backend().digest(store)

    max_bytes = get_config().claude_code.digest_max_bytes
    digest = truncate_on_line_boundary(digest, max_bytes)
    record["digest_bytes"] = len(digest.encode("utf-8"))
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": digest,
            }
        }
    )


async def _digest_decision(store: str, payload: dict[str, Any]) -> DigestDecision:
    """The double-occupancy freshness check (implemented).

    Reads the projected region's sources trailer (``<!-- sources: … -->``) from the MEMORY.md the harness will load, computes the live
    selection fingerprint, and returns ``skip`` on a match (the loaded file
    *is* the current view), the difference-only push on a mismatch, and the
    full digest whenever the projection is disabled for the store or the
    region / trailer cannot be parsed. Its own failure must never cost the
    session its digest, so any error degrades to the full push.
    """
    from particles.api.cli._memory_projection import digest_decision

    try:
        return await digest_decision(store, payload)
    except Exception:  # noqa: BLE001 — freshness is best-effort; digest must survive
        log.debug("projection freshness check failed; pushing the full digest", exc_info=True)
        return DigestDecision("full")


# ---------------------------------------------------------------------------
# session-end — the harvest
# ---------------------------------------------------------------------------


@hook_app.command("session-end")
def session_end_cmd(
    store: str = typer.Option(..., "--store", help="Memory store the harvest deposits into."),
) -> None:
    """SessionEnd hook: harvest the session transcript + memory files."""
    _run_hook("session-end", store, _session_end)


async def _session_end(store: str, payload: dict[str, Any], record: dict[str, Any]) -> str | None:
    cfg = get_config()
    harvest_cfg = cfg.claude_code.harvest

    if cfg.engine.base_url and not harvest_cfg.allow_remote:
        # Local-only by default: transcripts are the most
        # sensitive payload the SDK touches; shipping them off-machine is an
        # explicit opt-in. The refusal is logged; the catch-up sweep back-fills
        # once claude_code.harvest.allow_remote is enabled.
        record["skipped"] = "remote-harvest-disabled"
        return None

    session_id = str(payload.get("session_id") or "")
    transcript_path = Path(str(payload.get("transcript_path") or ""))
    project = transcript_path.parent.name if transcript_path.name else ""

    deposited: list[tuple[str, str]] = []  # (entry_id, snapshot_id) of NEW snapshots
    unchanged = 0

    # (a) The current session's transcript — distilled, not raw (§3a).
    if harvest_cfg.transcripts and session_id and transcript_path.is_file():
        outcome = await _harvest_transcript(store, transcript_path, session_id, project)
        if outcome is not None:
            if outcome.unchanged:
                unchanged += 1
            else:
                deposited.append((outcome.entry_id, outcome.snapshot_id))

    # (b) Changed memory files beside the transcript (§3b).
    memory_dir = transcript_path.parent / "memory" if transcript_path.name else None
    memory_md_text: str | None = None
    if memory_dir is not None and memory_dir.is_dir():
        harvested, skipped, memory_md_text = await _harvest_memory_files(store, memory_dir, project)
        deposited.extend(harvested)
        unchanged += skipped
        record["memory_files"] = len(harvested) + skipped

    # (b′) The fold-and-archive file is itself corpus input:
    # lines folded out of MEMORY.md on a *previous* cycle land here, so the
    # archive is harvested level-triggered like everything else.
    if projection_enabled():
        outcome = await _harvest_archive(store, project)
        if outcome is not None:
            if outcome.unchanged:
                unchanged += 1
            else:
                deposited.append((outcome.entry_id, outcome.snapshot_id))

    # (c) Catch-up sweep: SessionEnd does not fire on SIGKILL / crash, but the
    # transcript persists on disk — harvest is level-triggered, and the corpus
    # is the harvest state (§3c). Failures here were logged last time and are
    # simply retried now.
    if harvest_cfg.transcripts and transcript_path.name and transcript_path.parent.is_dir():
        swept = await _catchup_sweep(
            store, transcript_path.parent, exclude=transcript_path, project=project
        )
        record["swept"] = len(swept)
        deposited.extend(swept)

    record["deposited"] = len(deposited)
    record["unchanged"] = unchanged

    # (§4) Default deferred: the deposit is the hook's whole job and the
    # corpus is the buffer. extract_inline closes the loop at the session
    # boundary for operators who opt in to the LLM cost.
    if harvest_cfg.extract_inline and deposited:
        record["extracted"] = await _extract_inline(store, deposited)

    # mine the session's actions for utility evidence (credit action,
    # not attention; the reliable signal already in the harvest). Guarded like
    # everything in the hook: any failure degrades to a no-op, the next session's
    # harvest retries. Local store only — utility events live in the store the
    # projection ranks; the remote-engine path is deferred with cross-machine
    # memory.
    if (
        get_config().utility.mining.enabled
        and harvest_cfg.transcripts
        and session_id
        and transcript_path.is_file()
    ):
        from particles.api.client import get_backend

        if not get_backend().remote:
            with contextlib.suppress(Exception):
                record["utility"] = await _mine_utility(store, transcript_path, session_id)

    # the render-splice tail of the cycle. Ordering is the
    # safety property: this point is reached only when every deposit above
    # succeeded (harvest → extract → render → splice), so the splice can never
    # overwrite content that hasn't been harvested. Remote engines are skipped:
    # the memory directory is machine-local by platform design.
    if memory_dir is not None and projection_enabled():
        from particles.api.cli._memory_projection import run_projection_cycle
        from particles.api.client import get_backend

        if not get_backend().remote:
            record["projection"] = await run_projection_cycle(store, memory_dir, memory_md_text)
    return None


async def _mine_utility(store: str, transcript_path: Path, session_id: str) -> dict[str, int]:
    """Mine the current session's actions into utility evidence.

    Re-distills the transcript (deterministic, cheap) and mines it against the
    store's current ACTIVE beliefs. Returns disclosure counts for the hook log.
    """
    from particles.operations.utility_mining import mine_session_from_transcript

    text = distill_transcript(
        transcript_path.read_text(encoding="utf-8", errors="replace"), session_id
    )
    if not text:
        return {}
    result = await mine_session_from_transcript(store, text, session_id)
    return {
        "literal": result.literal,
        "behavioural": result.behavioural,
        "candidates": result.candidates,
    }


async def _harvest_transcript(
    store: str, transcript_path: Path, session_id: str, project: str
) -> TextDepositOutcome | None:
    """Distill + redact + deposit one session transcript (APPEND_ONLY)."""
    from particles.api.client import get_backend

    text = distill_transcript(
        transcript_path.read_text(encoding="utf-8", errors="replace"), session_id
    )
    if not text:
        return None
    text = redact_secrets(text)
    tags = ["claude-code", f"session:{session_id}"]
    if project:
        tags.append(f"project:{project}")
    return await get_backend().deposit_text_at_uri(
        text=text,
        uri_r=f"claude-code://session/{session_id}",
        source_type="CONVERSATION",
        mutability="APPEND_ONLY",
        tags=tags,
        deposited_by="claude-code-hook",
        content_published_at=None,
        store=store,
    )


async def _harvest_memory_files(
    store: str, memory_dir: Path, project: str
) -> tuple[list[tuple[str, str]], int, str | None]:
    """Deposit each memory-file ``*.md`` (LOCAL_MARKDOWN / MUTABLE, §3b).

    Content passes through :func:`filter_memory_file_for_deposit` — the sentinel strip (pristine projected regions never reach the
    corpus; a dirtied region rides along as authored input). Unchanged files
    are content-hash no-ops. Also returns the **raw** (pre-strip) text of the
    top-level ``MEMORY.md``, if present: the projection cycle refuses to
    splice a file that changed after this harvest read it.
    """
    from particles.api.client import get_backend

    tags = ["claude-code", "memory-file"]
    if project:
        tags.append(f"project:{project}")
    harvested: list[tuple[str, str]] = []
    skipped = 0
    memory_md_text: str | None = None
    for md in sorted(memory_dir.rglob("*.md")):
        raw = md.read_text(encoding="utf-8", errors="replace")
        if md.parent == memory_dir and md.name == "MEMORY.md":
            memory_md_text = raw
        text = filter_memory_file_for_deposit(raw)
        if not text.strip():
            continue
        mtime = datetime.fromtimestamp(md.stat().st_mtime, tz=UTC)
        outcome = await get_backend().deposit_text_at_uri(
            text=text,
            uri_r=md.resolve().as_uri(),
            source_type="LOCAL_MARKDOWN",
            mutability="MUTABLE",
            tags=tags,
            deposited_by="claude-code-hook",
            content_published_at=mtime,
            store=store,
            # keeps the promise: a memory file is
            # MUTABLE, so it belongs in the nightly refresh loop rather than
            # depending on another SessionEnd in this project to notice an edit.
            # Reconciled even on an unchanged re-deposit, which is how the
            # entries that predate 0207 enrol.
            #
            # §5a: but only when the deposited body IS the file's bytes. The
            # top-level MEMORY.md is deposited sentinel-stripped, so its
            # snapshot hash can never equal the file's — and the byte-level tier
            # would "correct" that by archiving the raw file, putting the
            # store's own rendered output into the corpus (the
            # belt-1 violation). Those files keep NEVER and stay refreshed by
            # this harvest, which applies the same strip.
            fetch_policy=refresh_policy_for(raw, text).value,
        )
        if outcome.unchanged:
            skipped += 1
        else:
            harvested.append((outcome.entry_id, outcome.snapshot_id))
    return harvested, skipped, memory_md_text


async def _harvest_archive(store: str, project: str) -> TextDepositOutcome | None:
    """Deposit the fold-and-archive file (APPEND_ONLY — it only grows)."""
    from particles.api.cli._claude_code import memory_archive_path
    from particles.api.client import get_backend

    archive = memory_archive_path()
    if not archive.is_file():
        return None
    text = archive.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None
    tags = ["claude-code", "memory-file", "memory-archive"]
    if project:
        tags.append(f"project:{project}")
    mtime = datetime.fromtimestamp(archive.stat().st_mtime, tz=UTC)
    return await get_backend().deposit_text_at_uri(
        text=text,
        uri_r=archive.resolve().as_uri(),
        source_type="LOCAL_MARKDOWN",
        mutability="APPEND_ONLY",
        tags=tags,
        deposited_by="claude-code-hook",
        content_published_at=mtime,
        store=store,
    )


async def _catchup_sweep(
    store: str, transcript_dir: Path, *, exclude: Path, project: str
) -> list[tuple[str, str]]:
    """Harvest up to ``catchup_limit`` recent transcripts whose content moved (§3c).

    The distilled-content-hash comparison is the engine-side unchanged check in
    ``deposit_text_versioned`` — no side-car high-water-mark file to drift.
    Subagent transcripts (subdirectories) are deliberately not walked
    (§ Deferred).
    """
    limit = get_config().claude_code.harvest.catchup_limit
    if limit <= 0:
        return []
    candidates = [
        p
        for p in transcript_dir.glob("*.jsonl")
        if p.is_file() and p.resolve() != exclude.resolve()
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    swept: list[tuple[str, str]] = []
    for path in candidates[:limit]:
        outcome = await _harvest_transcript(store, path, path.stem, project)
        if outcome is not None and not outcome.unchanged:
            swept.append((outcome.entry_id, outcome.snapshot_id))
    return swept


async def _extract_inline(store: str, deposited: list[tuple[str, str]]) -> int:
    """Opt-in inline extraction (§4), bounded by ``max_extract_entries_per_session``.

    Current-session-first ordering is the caller's deposit order. Remote mode
    extracts nothing here — the engine owns extraction cadence on its side.
    """
    from particles.api.client import get_backend
    from particles.db import session_scope
    from particles.operations.extract import extract_snapshot

    if get_backend().remote:
        return 0
    max_entries = get_config().claude_code.harvest.max_extract_entries_per_session
    extracted = 0
    for entry_id, snapshot_id in deposited[:max_entries]:
        async with session_scope(store) as session:
            await extract_snapshot(session, entry_id, snapshot_id, agent_id="claude-code-hook")
            await session.commit()
        extracted += 1
    return extracted


# ---------------------------------------------------------------------------
# hook log — "is this thing on?"
# ---------------------------------------------------------------------------


@hook_app.command("log")
def hook_log_cmd(
    tail: int = typer.Option(20, "--tail", "-n", help="How many recent entries to print."),
) -> None:
    """Print recent hook-log entries (one JSONL line per hook invocation)."""
    lines = read_hook_log_tail(tail)
    if not lines:
        typer.echo(f"No hook log entries yet at {hook_log_path()}.")
        return
    for line in lines:
        typer.echo(line)


# ---------------------------------------------------------------------------
# hook doctor — "why did my harvest not land?" (§1/§6)
# ---------------------------------------------------------------------------


@hook_app.command("doctor")
def hook_doctor_cmd(
    store: str = typer.Option("default", "--store", help="Store handle to check."),
) -> None:
    """Diagnose whether ``particles hook`` resolves ``store`` from the current directory.

    The lifecycle hooks degrade to exit 0 on any failure, so a mis-resolved
    store fails silently. This verb makes that resolution visible:
    which config.yaml is found, which DSN the handle resolves to, whether the DB
    file exists and carries the corpus tables. Exits non-zero when the store is
    unusable, so it can gate an operator's "is this thing on?" check.
    """
    ok = _run_doctor(store)
    raise typer.Exit(0 if ok else 1)


def _run_doctor(store: str) -> bool:
    from particles.config import validate_config

    typer.echo(f"particles hook doctor — store '{store}'")
    typer.echo(f"  cwd:            {os.getcwd()}")

    config_path, _ = validate_config()
    if config_path is None:
        typer.echo("  config.yaml:    NOT FOUND (compiled defaults + env overrides only)")
    else:
        typer.echo(f"  config.yaml:    {config_path}")
    if os.environ.get("PARTICLES_CONFIG"):
        typer.echo(f"  PARTICLES_CONFIG env: {os.environ['PARTICLES_CONFIG']}")
    if os.environ.get("DATABASE_URL"):
        typer.echo(f"  DATABASE_URL env:     {os.environ['DATABASE_URL']}")

    dsn = _resolve_store_dsn(store)
    if dsn is None:
        typer.echo(f"  store DSN:      MISSING — '{store}' is not declared in storage.stores")
        typer.echo("  ✗ Store not resolvable. Declare it in config.yaml or bake the pins with")
        typer.echo("    `particles init claude-code --store <handle>`.")
        return False
    typer.echo(f"  store DSN:      {dsn}")

    if dsn.startswith("sqlite"):
        location = _resolved_store_location(store)
        cwd_relative = _dsn_is_cwd_relative(dsn)
        exists = Path(location).exists()
        typer.echo(f"  database file:  {location} ({'exists' if exists else 'MISSING'})")
        if cwd_relative:
            typer.echo(
                "  ✗ DSN path is working-directory-relative — the hook resolves a DIFFERENT "
                "file from each directory. Bake an absolute path via `init claude-code`."
            )
            return False
        if not exists:
            typer.echo("  ✗ Database file does not exist. Run `particles db init` for this store.")
            return False

    tables_ok, detail = _check_tables(store)
    typer.echo(f"  corpus tables:  {detail}")
    if not tables_ok:
        return False

    blob_lines = _check_blobs(store)
    for line in blob_lines:
        typer.echo(f"  {line}")

    typer.echo("  ✓ Store resolves and is initialized from this directory.")
    return True


def _check_blobs(store: str) -> list[str]:
    """Report whether the store's snapshots resolve to blobs from *this* directory.

    The same cwd-sensitivity this verb exists to expose applies to the blob tree
    as well as the DSN, and a hook that resolved the right
    database from the wrong directory would still fail at extraction time with
    `Blob not found for hash …`. Reported, never fatal: unreachable content is
    an existing-data problem, not a "this store is unusable from here" verdict,
    so it does not flip the verb's exit code.
    """
    from particles.corpus.blob_health import check_blob_reachability
    from particles.db import session_scope

    async def _probe() -> list[str]:
        async with session_scope(store) as session:
            report = await check_blob_reachability(session)
        if report.sampled == 0:
            return [f"blobs:          nothing deposited yet ({report.blob_dir})"]
        if report.healthy:
            return [f"blobs:          reachable ({report.sampled} sampled under {report.blob_dir})"]
        return report.warning_lines()

    try:
        return asyncio.run(_probe())
    except Exception as exc:  # noqa: BLE001 — doctor reports, never crashes
        return [f"blobs:          could not check ({type(exc).__name__}: {exc})"]


def _dsn_is_cwd_relative(dsn: str) -> bool:
    """True when a SQLite DSN is working-directory-relative (would resolve per-cwd)."""
    from particles.api.cli._claude_code import absolutize_sqlite_dsn

    return dsn.startswith("sqlite") and absolutize_sqlite_dsn(dsn) != dsn


def _check_tables(store: str) -> tuple[bool, str]:
    """Confirm the corpus tables exist in ``store`` (the harvest deposit target)."""
    from sqlalchemy import func, select
    from sqlalchemy.exc import OperationalError

    async def _count() -> int:
        from particles.corpus.store import CorpusEntryRow
        from particles.db import session_scope

        async with session_scope(store) as session:
            return await session.scalar(select(func.count()).select_from(CorpusEntryRow)) or 0

    try:
        count = asyncio.run(_count())
    except OperationalError:
        return False, "MISSING — run `particles db init` (the harvest target has no tables)"
    except Exception as exc:  # noqa: BLE001 — doctor reports, never crashes
        return False, f"could not read ({type(exc).__name__}: {exc})"
    return True, f"present ({count} corpus entries)"
