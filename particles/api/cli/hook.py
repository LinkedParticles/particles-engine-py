# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
import shlex
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer

from particles.api.cli import app
from particles.api.cli._claude_code import (
    InstallPin,
    append_hook_log,
    claude_project_slug,
    command_env_pins,
    distill_transcript,
    enclosing_work_tree,
    filter_memory_file_for_deposit,
    hook_log_path,
    install_report,
    kept_prefix_length,
    local_harvest_roots,
    memory_load_window_warning,
    observer_project_for,
    particles_hook_commands,
    particles_mcp_executables,
    particles_mcp_registrations,
    pinned_default_dsn,
    pinned_store_dsn,
    projection_enabled,
    projection_refusal,
    read_hook_log_tail,
    read_memory_files,
    record_project_root,
    redact_secrets,
    repository_root,
    resolve_session_project,
    session_start_budget,
    stray_memory_dirs,
    transcript_started_at,
    truncate_on_line_boundary,
    untimed_session_end_hooks,
    walk_memory_dir,
)
from particles.api.cli._memory_projection import (
    DigestDecision,
    _read_memory_md,
    bullet_beliefs,
)
from particles.config import (
    config_resolution_warnings,
    environment_engine_url,
    get_config,
    relative_config_pin,
    reset_config,
    sqlite_file_path,
)
from particles.corpus.rule_sources import refresh_policy_for
from particles.egress import run_armed
from particles.secrets import get_engine_token_optional, redact_known_secrets
from particles.store.session_exposure_store import SessionExposure, ShownBelief
from particles.url_safety import GIT_LOCAL_ARGS, pinned_subprocess_env, url_without_userinfo

if TYPE_CHECKING:
    from particles.api.cli._claude_code import MemoryDirWalk
    from particles.api.client.base import TextDepositOutcome
    from particles.operations.memory_absence import MemoryAbsenceReport

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
    explicit = relative_config_pin()
    if explicit is not None:
        # The hook runs in the session's directory, where a relative pin is a file
        # a cloned repository can ship (security review N15, P14). Nothing reads
        # it: the pin is dropped for this process, so even the log goes where
        # the user file says, and the body does not run.
        os.environ.pop("PARTICLES_CONFIG", None)
        reset_config()
        record["outcome"] = "refused"
        record["error"] = (
            f"PARTICLES_CONFIG names a relative path ({explicit}); set an absolute path "
            "in the hook command or re-run `particles init claude-code`"
        )
        record["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
        append_hook_log(record)
        return
    refusal = _working_directory_store_refusal(store)
    if refusal is not None:
        record["outcome"] = "refused"
        record["error"] = refusal
        record["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
        append_hook_log(record)
        return
    try:
        payload = _read_stdin_json()
        record["session_id"] = payload.get("session_id")
        record["source"] = payload.get("source") or payload.get("reason")
        deadline = get_config().claude_code.hook_deadline_seconds

        async def _body() -> str | None:
            return await asyncio.wait_for(impl(store, payload, record), timeout=deadline)

        # Arm the no-egress backstop and refuse a loop it cannot see (
        # §3; D18, U13): a hook runs its body under its own loop, not the CLI's
        # run(). A refusal is an error like any other, so the hook still exits 0.
        output = run_armed(_body())
        record["outcome"] = "ok"
    except Exception as exc:  # noqa: BLE001 — the degrade-to-nothing contract
        record["outcome"] = "error"
        record["error"] = redact_known_secrets(f"{type(exc).__name__}: {exc}")
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
        # The hooks print nothing but their own output, so a config-resolution
        # warning (a missing PARTICLES_CONFIG file, a nearby config left unread;
        # §2, §3) goes to the hook log instead.
        warnings = config_resolution_warnings()
        if warnings:
            record["config_warnings"] = warnings
        append_hook_log(record)
    if output:
        typer.echo(output)
    # Fall through: exit 0. Exit 2 is never used.


def _working_directory_store_refusal(store: str) -> str | None:
    """Why a hook must not open ``store``, when its SQLite path is relative; else ``None``.

    Claude Code runs the hooks in the session's directory, which a cloned
    repository supplies, so a relative store there is whatever file the clone
    ships: SessionStart would push its beliefs into the session and SessionEnd
    would harvest into it instead of the real store (security review
    2026-10-25, Q5). The stdio MCP servers already refuse such a store.
    A hook has no zero-config launch to keep, so it refuses whichever
    pin named the path, a relative ``DATABASE_URL`` included, and the body does
    not run. The config has already expanded a leading ``~`` (P13). A thin
    client reads no local store, but only an engine the config file names makes
    one: ``PARTICLES_ENGINE_BASE_URL`` can come from the harness's environment
    rather than the hook command, so it does not lift the rule (security
    review 2026-10-25-2, R41).
    """
    from particles.db import DEFAULT_STORE

    inherited = environment_engine_url()
    if get_config().engine.base_url and inherited is None:
        return None
    dsn = _resolve_store_dsn(store)
    db_file = sqlite_file_path(dsn) if dsn is not None else None
    if db_file is None or Path(db_file).is_absolute():
        return None
    if store == DEFAULT_STORE:
        fix = (
            "set an absolute DATABASE_URL in the hook command or re-run "
            "`particles init claude-code`"
        )
    else:
        fix = f"write an absolute path under storage.stores.{store} in the config file"
    if inherited is not None:
        fix += (
            f"; PARTICLES_ENGINE_BASE_URL ({url_without_userinfo(inherited)}) lifts this "
            "rule only when the "
            "pinned config names the engine as engine.base_url"
        )
    return (
        f"the store {store!r} ({dsn}) resolves against the session's working directory, "
        f"so the hook would open whatever file that directory holds; {fix}"
    )


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
    started = time.monotonic()
    source = str(payload.get("source") or "startup")
    if source == "resume":
        # A resumed session replays its prior context, digest included;
        # re-injecting would duplicate it. startup / clear / compact (context
        # rebuilt) all get a fresh push. The source test lives here, not in a
        # settings matcher, so the policy upgrades with the SDK.
        # Its context was recorded when first delivered, so no row.
        record["skipped"] = "resume"
        return None

    try:
        resolved = resolve_session_project(payload)
        project_key = resolved.key
        # which rule gave the key, so `hook doctor` can show a
        # session whose digest key came from a fallback a bound server never takes.
        record["project_key"] = project_key
        record["project_resolved_from"] = resolved.resolved_from
        record["cwd"] = str(payload.get("cwd") or "")
    except Exception:  # noqa: BLE001 — an unresolved project is never an observer
        log.debug("session project resolution failed", exc_info=True)
        project_key = ""
    # under `claude_code.observer_scope: project` the session
    # reads the store through its own project; otherwise the whole store.
    observer = observer_project_for(project_key)
    decision = await _digest_decision(store, payload)
    shown: list[ShownBelief] = list(decision.loaded)
    # deletions the hook held back are named at session start.
    notice = await _absence_notice(store, payload, record)
    if decision.action == "skip":
        # The MEMORY.md projected region the harness just loaded IS the
        # current view (trailer fingerprint matched) — injecting the digest
        # would duplicate it.
        record["skipped"] = "projection-current"
        await _record_exposure(
            store, payload, record, "skip", shown, project_key, observer, started
        )
        return _session_start_output(notice.lstrip("\n")) if notice else None

    # Clamped under Claude Code's additionalContext cap: over it the model
    # sees only a 2 KB preview, and `shown` would credit lines it never read.
    # The notice is kept out of the budget the digest may fill.
    max_bytes = session_start_budget(get_config().claude_code.digest_max_bytes)
    max_bytes = max(0, max_bytes - len(notice.encode("utf-8")))
    if decision.action == "diff" and decision.content is not None:
        # Mismatch: the store moved since the last render — top up with only
        # the difference (new / changed / newly-contested lines).
        digest = truncate_on_line_boundary(decision.content, max_bytes)
        record["digest_mode"] = "projection-diff"
        action: Literal["full", "diff"] = "diff"
        # Diff lines carry their own `p-` ids; read them from what survived.
        shown += bullet_beliefs(digest)
    else:
        from particles.api.client import get_backend

        if observer is not None:
            record["observer_project"] = observer
        rendered = await get_backend().digest_located(store, observer)
        kept = kept_prefix_length(rendered.markdown, max_bytes)
        digest = truncate_on_line_boundary(rendered.markdown, max_bytes)
        action = "full"
        # A belief is credited as shown only when its whole line was delivered,
        # not merely its start (security review 2026-10-26-1, W33).
        shown += [
            ShownBelief(
                particle_id=line.particle_id,
                shown_as="digest",
                contested_bases=line.contested_bases,
            )
            for line in rendered.lines
            if _line_end(rendered.markdown, line.offset) <= kept
        ]

    record["digest_bytes"] = len(digest.encode("utf-8"))
    record["digest_chars"] = len(digest)
    await _record_exposure(store, payload, record, action, shown, project_key, observer, started)
    return _session_start_output(digest + notice)


def _session_start_output(context: str) -> str:
    """The SessionStart hook's JSON reply carrying ``context``."""
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": context,
            }
        }
    )


def _line_end(text: str, offset: int) -> int:
    """The offset just past the line of ``text`` that starts at ``offset``, its newline included."""
    newline = text.find("\n", offset)
    return len(text) if newline < 0 else newline + 1


#: The share of the hook deadline the freshness check may spend before it
#: degrades to the full push, which then has the rest.
_FRESHNESS_DEADLINE_SHARE = 0.5

#: Time the exposure write leaves the hook before its deadline, so a slow write
#: (a held writer lock) can never cost the session its context.
_EXPOSURE_DEADLINE_MARGIN_SECONDS = 1.0


async def _record_exposure(
    store: str,
    payload: dict[str, Any],
    record: dict[str, Any],
    action: Literal["full", "diff", "skip"],
    shown: list[ShownBelief],
    project_key: str,
    observer: str | None,
    started: float,
) -> None:
    """Record what the session was shown, through the client backend.

    Fail-open, like every hook step: a failed or timed-out write
    is logged and the session still receives its context. The write gets what
    is left of the hook deadline, less a margin, so it can never trip the
    deadline that would drop the context.
    """
    from particles.api.client import get_backend

    exposure = SessionExposure(
        session_id=str(payload.get("session_id") or ""),
        recorded_at=datetime.now(UTC),
        source=str(payload.get("source") or "startup"),
        action=action,
        project_key=project_key,
        observer_scope_applied=observer is not None,
        beliefs=shown,
    )
    remaining = (
        get_config().claude_code.hook_deadline_seconds
        - (time.monotonic() - started)
        - _EXPOSURE_DEADLINE_MARGIN_SECONDS
    )
    try:
        if remaining <= 0:
            raise TimeoutError("no time left before the hook deadline")
        await asyncio.wait_for(get_backend().record_session_exposure(store, exposure), remaining)
    except Exception as exc:  # noqa: BLE001 — fail-open: exposure unknown, context delivered
        log.debug("session exposure write failed", exc_info=True)
        record["exposure_error"] = f"{type(exc).__name__}: {exc}"
        return
    record["exposure_beliefs"] = len(shown)


async def _digest_decision(store: str, payload: dict[str, Any]) -> DigestDecision:
    """The double-occupancy freshness check (implemented).

    Reads the projected region's sources trailer (``<!-- sources: … -->``)
    from the MEMORY.md the harness will load, computes the live
    selection fingerprint, and returns ``skip`` on a match (the loaded file
    *is* the current view), the difference-only push on a mismatch, and the
    full digest whenever the projection is disabled for the store or the
    region / trailer cannot be parsed. Its own failure must never cost the
    session its digest, so any error degrades to the full push, and so does a
    check that outlives its share of the hook deadline: on a remote engine the
    fresh render is a request, and a slow engine must leave the full push the
    time it needs.
    """
    from particles.api.cli._memory_projection import digest_decision

    budget = get_config().claude_code.hook_deadline_seconds * _FRESHNESS_DEADLINE_SHARE
    try:
        return await asyncio.wait_for(digest_decision(store, payload), budget)
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
    # The project is the repository the session's auto-memory is keyed on, not
    # the directory its transcript sits in — in a linked worktree those differ,
    # and the transcript's directory holds no memory Claude Code reads.
    session_project = resolve_session_project(payload)
    project = session_project.key
    record["project"] = project
    record["project_resolved_from"] = session_project.resolved_from
    if project and session_project.root is not None:
        _note_project_key_collision(project, session_project.root, record)

    deposited: list[tuple[str, str]] = []  # (entry_id, snapshot_id) of NEW snapshots
    unchanged = 0
    # Memory and archive files the engine refused one by one (rule 5).
    refused: list[dict[str, str]] = []

    # (a) The current session's transcript — distilled, not raw (§3a).
    if harvest_cfg.transcripts and session_id and transcript_path.is_file():
        outcome = await _harvest_transcript(store, transcript_path, session_id, project)
        if outcome is not None:
            if outcome.unchanged:
                unchanged += 1
            else:
                deposited.append((outcome.entry_id, outcome.snapshot_id))

    # (b) Changed memory files in the project's memory directory (§3b) — only
    # when this store is the one the installed hooks name for it. A hook run
    # against any other store (a test, a stray DATABASE_URL) must neither
    # ingest the user's memory files nor render into them.
    memory_dir = session_project.memory_dir
    memory_md_text: str | None = None
    project_roots = _session_project_roots(payload)
    if memory_dir is not None and _memory_dir_refused(store, memory_dir, project_roots, record):
        memory_dir = None
    if memory_dir is not None and memory_dir.is_dir():
        absence_log: dict[str, Any] = {}
        harvested, skipped, memory_md_text = await _harvest_memory_files(
            store, memory_dir, project, refused=refused, absence_log=absence_log
        )
        deposited.extend(harvested)
        unchanged += skipped
        record["memory_files"] = len(harvested) + skipped
        if absence_log.get("missing") or "skipped" in absence_log or "error" in absence_log:
            record["memory_absences"] = absence_log

    # (b′) The fold-and-archive file is itself corpus input:
    # lines folded out of MEMORY.md on a *previous* cycle land here, so the
    # archive is harvested level-triggered like everything else.
    if projection_enabled():
        outcome = await _harvest_archive(store, memory_dir, project, refused=refused)
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
    if refused:
        record["refused"] = refused

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
                record["utility"] = await _mine_utility(store, transcript_path, session_id, project)

    # the render-splice tail of the cycle. Ordering is the
    # safety property: this point is reached only when every deposit above
    # succeeded (harvest → extract → render → splice), so the splice can never
    # overwrite content that hasn't been harvested. Against a remote engine the
    # engine renders and this client splices.
    # A refused memory or archive file is a deposit that did not succeed, so
    # the splice waits for a harvest that sends every file.
    if memory_dir is not None and projection_enabled() and refused:
        record["projection"] = {"skipped": "a memory or archive file was refused"}
    elif memory_dir is not None and projection_enabled():
        from particles.api.cli._memory_projection import run_projection_cycle

        record["projection"] = await run_projection_cycle(
            store, memory_dir, memory_md_text, project_roots=project_roots
        )
    return None


# ---------------------------------------------------------------------------
# stop — retired (deprecated)
# ---------------------------------------------------------------------------


@hook_app.command("stop", hidden=True)
def stop_cmd(
    store: str = typer.Option("default", "--store", help="Ignored; kept for old installs."),
) -> None:
    """Retired Stop hook: exits 0 and does nothing.

    Agent-side extraction was deprecated and removed. An install
    made while it was on still runs this verb at the end of every turn until
    `particles init claude-code` is re-run, which drops the entry; until then
    it must not fail or block, so it returns at once.
    """
    del store


def _session_project_roots(payload: dict[str, Any]) -> tuple[Path, ...]:
    """The session's repository, whose project-scope install may bind its memory."""
    cwd = str(payload.get("cwd") or "")
    return (repository_root(Path(cwd)),) if cwd else ()


def _memory_dir_refused(
    store: str, memory_dir: Path, project_roots: tuple[Path, ...], record: dict[str, Any]
) -> bool:
    """Whether ``store`` must leave ``memory_dir`` alone; the reason goes to the hook log.

    Against a remote engine the binding is the engine and the handle the
    installed hooks pin, not a local database.
    """
    refusal = projection_refusal(store, memory_dir, project_roots)
    if refusal is None:
        return False
    record["memory_skipped"] = refusal
    return True


async def _mine_utility(
    store: str, transcript_path: Path, session_id: str, project: str
) -> dict[str, int]:
    """Mine the current session's actions into utility evidence.

    Re-distills the transcript (deterministic, cheap) and mines it against what
    the session was shown, read as of the session's start: the
    first timestamp in the raw transcript, which the distilled copy drops.
    The text is masked exactly as the deposit is, since the behavioural judge
    sends its action lines to a model (security review 2026-10-25-2, R22).
    Returns disclosure counts for the hook log.
    """
    from particles.operations.utility_mining import mine_session_from_transcript

    raw = transcript_path.read_text(encoding="utf-8", errors="replace")
    text = _distilled_session(raw, session_id)
    if not text:
        return {}
    result = await mine_session_from_transcript(
        store,
        text,
        session_id,
        started_at=transcript_started_at(raw),
        project_key=project or None,
    )
    return {
        "literal": result.literal,
        "behavioural": result.behavioural,
        "candidates": result.candidates,
        "nominated": result.literal_nominated,
        "judge_calls": result.behavioural_calls,
    }


def _distilled_session(jsonl_text: str, session_id: str) -> str:
    """The text a harvest deposits for a session: distilled, then redacted."""
    text = distill_transcript(jsonl_text, session_id)
    return redact_secrets(text) if text else ""


def _note_project_key_collision(key: str, root: Path, record: dict[str, Any]) -> None:
    """Record the session's repository root for its key; flag a shared key.

    Never a failure surface: an unwritable state directory costs the
    disclosure, not the harvest.
    """
    try:
        roots = record_project_root(key, root)
    except OSError:
        log.debug("could not record the project root for %s", key, exc_info=True)
        return
    if len(roots) > 1:
        record["project_key_collision"] = roots


async def _harvest_transcript(
    store: str, transcript_path: Path, session_id: str, project: str
) -> TextDepositOutcome | None:
    """Distill + redact + deposit one session transcript (APPEND_ONLY)."""
    from particles.api.client import get_backend

    jsonl_text = transcript_path.read_text(encoding="utf-8", errors="replace")
    text = _distilled_session(jsonl_text, session_id)
    if not text:
        return None
    tags = ["claude-code", f"session:{session_id}"]
    if project:
        tags.append(f"project:{project}")
    # the session's own start is the reference date its claims
    # resolve against. Left unset, the extractor anchors on its wall-clock,
    # which the deferred extract pass makes later than the session.
    return await get_backend().deposit_text_at_uri(
        text=text,
        uri_r=f"claude-code://session/{session_id}",
        source_type="CONVERSATION",
        mutability="APPEND_ONLY",
        tags=tags,
        deposited_by="claude-code-hook",
        content_published_at=transcript_started_at(jsonl_text),
        store=store,
    )


async def _harvest_memory_files(
    store: str,
    memory_dir: Path,
    project: str,
    *,
    refused: list[dict[str, str]] | None = None,
    absence_log: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, str]], int, str | None]:
    """Deposit each memory-file ``*.md`` (LOCAL_MARKDOWN / MUTABLE, §3b).

    Content passes through :func:`filter_memory_file_for_deposit` — the
    sentinel strip (pristine projected regions never reach the
    corpus; a dirtied region rides along as authored input). Unchanged files
    are content-hash no-ops. Also returns the **raw** (pre-strip) text of the
    top-level ``MEMORY.md``, if present: the projection cycle refuses to
    splice a file that changed after this harvest read it.

    A file the engine refuses (a hook key whose roots miss it) is recorded in
    ``refused`` and the rest are still sent (rule 5; security review
    2026-10-30-1, AB49).

    The walk is also the directory's deletion check: after the
    deposits, each previously harvested memory file it no longer lists gets an
    ``ABSENT`` record, unless a guard holds the walk's absences back. What the
    step did lands in ``absence_log``, for the hook log or pass 6's telemetry.
    """

    if refused is None:
        refused = []
    refused_before = len(refused)

    tags = ["claude-code", "memory-file"]
    if project:
        tags.append(f"project:{project}")
    harvested: list[tuple[str, str]] = []
    skipped = 0
    memory_md_text: str | None = None
    walk = walk_memory_dir(memory_dir)
    for md, raw, st in read_memory_files(memory_dir, walk):
        if md.parent == memory_dir and md.name == "MEMORY.md":
            memory_md_text = raw
        text = filter_memory_file_for_deposit(raw, memory_dir=memory_dir, source_dir=md.parent)
        if not text.strip():
            continue
        mtime = datetime.fromtimestamp(st.st_mtime, tz=UTC)
        uri = md.resolve().as_uri()
        outcome = await _deposit_or_record_refusal(
            refused,
            uri,
            text=text,
            uri_r=uri,
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
        if outcome is None:
            continue
        if outcome.unchanged:
            skipped += 1
        else:
            harvested.append((outcome.entry_id, outcome.snapshot_id))
    absences = await _record_memory_absences(
        store, memory_dir, project, walk, refused_files=len(refused) - refused_before
    )
    if absence_log is not None:
        absence_log.update(absences)
    return harvested, skipped, memory_md_text


async def _record_memory_absences(
    store: str, memory_dir: Path, project: str, walk: MemoryDirWalk, *, refused_files: int
) -> dict[str, Any]:
    """Record what the walk no longer finds; the hook-log record of it.

    Local store only: against a remote engine nothing is recorded and the
    record says so. A failure here is logged and costs the absences,
    never the harvest that preceded it.
    """
    from particles.api.client import get_backend

    if get_backend().remote:
        return {"skipped": "remote"}
    if not project:
        return {"skipped": "no_project"}
    from particles.db import session_scope
    from particles.operations.memory_absence import MemoryWalk, record_memory_absences

    facts = MemoryWalk(
        directory=memory_dir.resolve(),
        project_key=project,
        listed=walk.listed_uris(),
        errors=walk.errors,
        refused_files=refused_files,
    )
    try:
        async with session_scope(store) as session:
            report = await record_memory_absences(session, facts)
    except Exception as exc:  # noqa: BLE001 — the deposits already landed
        log.warning("memory absences for %s not recorded: %s", memory_dir, exc)
        return {"error": redact_known_secrets(str(exc))[:300]}
    return report.log_record()


async def _survey_memory_absences(
    store: str, memory_dir: Path, project: str | None = None
) -> MemoryAbsenceReport | None:
    """The unrecorded deletions in ``memory_dir``, read only; ``None`` against a remote engine.

    ``project`` is the key the harvest stamps on the directory's files: the
    session's key at session start, else the directory's parent name, which
    is how pass 6 keys it.
    """
    from particles.api.client import get_backend

    if get_backend().remote or not memory_dir.is_dir():
        return None
    from particles.db import session_scope
    from particles.operations.memory_absence import MemoryWalk, survey_memory_absences

    walk = walk_memory_dir(memory_dir)
    facts = MemoryWalk(
        directory=memory_dir.resolve(),
        project_key=project or memory_dir.parent.name,
        listed=walk.listed_uris(),
        errors=walk.errors,
    )
    async with session_scope(store) as session:
        return await survey_memory_absences(session, facts)


def absence_notice(unrecorded: int) -> str:
    """The SessionStart line naming deletions the hook held back."""
    files = "memory file" if unrecorded == 1 else "memory files"
    verb = "is" if unrecorded == 1 else "are"
    return (
        f"\n_{unrecorded} {files} no longer found {verb} not recorded, so their beliefs are "
        "still listed: run `particles memory absences`._\n"
    )


#: The share of the hook deadline the absence notice may spend.
_ABSENCE_NOTICE_DEADLINE_SHARE = 0.2


async def _absence_notice(store: str, payload: dict[str, Any], record: dict[str, Any]) -> str:
    """The notice line for the session's memory directory, or ``""``.

    Shown while the hook holds deletions back (the first ever in the
    directory, or a mass deletion), so the next session's agent sees them and
    can relay them. A read, no model call, best-effort: any failure or a slow
    store costs the line, never the digest.
    """
    try:
        session_project = resolve_session_project(payload)
        memory_dir = session_project.memory_dir
        if memory_dir is None or projection_refusal(
            store, memory_dir, _session_project_roots(payload)
        ):
            return ""
        budget = get_config().claude_code.hook_deadline_seconds * _ABSENCE_NOTICE_DEADLINE_SHARE
        report = await asyncio.wait_for(
            _survey_memory_absences(store, memory_dir, session_project.key or None), budget
        )
    except Exception:  # noqa: BLE001 — the notice is best-effort; the digest must survive
        log.debug("memory absence survey failed", exc_info=True)
        return ""
    if report is None or report.skipped is not None or report.refusal is None:
        return ""
    record["absences_unrecorded"] = len(report.missing)
    return absence_notice(len(report.missing))


async def _deposit_or_record_refusal(
    refused: list[dict[str, str]], uri: str, **kwargs: Any
) -> TextDepositOutcome | None:
    """Deposit one harvested file; record and swallow the engine's refusal of it.

    Only a refusal of this one file is swallowed: an engine that cannot be
    reached, or that refuses the caller outright (503), still ends the harvest.
    """
    from particles.api.client import get_backend
    from particles.api.client.http import (
        EngineHttpError,
        EngineRefusedError,
        EngineUnreachableError,
    )

    try:
        return await get_backend().deposit_text_at_uri(**kwargs)
    except (EngineRefusedError, EngineUnreachableError):
        raise
    except EngineHttpError as exc:
        log.warning("the engine refused %s: %s", uri, exc)
        refused.append({"uri": uri, "error": redact_known_secrets(str(exc))[:300]})
        return None


async def _harvest_archive(
    store: str,
    memory_dir: Path | None,
    project: str,
    *,
    refused: list[dict[str, str]] | None = None,
) -> TextDepositOutcome | None:
    """Deposit this project's fold-and-archive file (APPEND_ONLY).

    The archive is per project: lines folded out of one
    project's ``MEMORY.md`` are harvested under that project's key, not under
    whichever project happened to deposit a shared file first. Its lines are
    ``MEMORY.md`` lines, so they pass the same mask as that file's deposit
    (security review 2026-10-25-1, Q30). A refusal is recorded in ``refused``
    rather than raised, as a memory file's is (AB49).
    """
    from particles.api.cli._claude_code import memory_archive_path

    if memory_dir is None:
        return None
    archive = memory_archive_path(memory_dir)
    if not archive.is_file():
        return None
    text = redact_secrets(archive.read_text(encoding="utf-8", errors="replace"))
    if not text.strip():
        return None
    tags = ["claude-code", "memory-file", "memory-archive"]
    if project:
        tags.append(f"project:{project}")
    mtime = datetime.fromtimestamp(archive.stat().st_mtime, tz=UTC)
    uri = archive.resolve().as_uri()
    return await _deposit_or_record_refusal(
        [] if refused is None else refused,
        uri,
        text=text,
        uri_r=uri,
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
    from particles.ingest.pipeline import SnapshotOutcome
    from particles.operations.extract import extract_snapshot

    if get_backend().remote:
        return 0
    max_entries = get_config().claude_code.harvest.max_extract_entries_per_session
    extracted = 0
    for entry_id, snapshot_id in deposited[:max_entries]:
        outcome = SnapshotOutcome()
        async with session_scope(store) as session:
            await extract_snapshot(
                session, entry_id, snapshot_id, agent_id="claude-code-hook", outcome_out=outcome
            )
            await session.commit()
        # A snapshot handed back PENDING (every LLM call failed) or never
        # claimed is not extracted; the consolidation catch-up owns it.
        if outcome.skipped is None and not outcome.failed_calls:
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


def _engine_line() -> str:
    """Where the hooks write, and whether a keyless engine would refuse them.

    Reads local state only: ``engine.base_url``, any ``PARTICLES_ENGINE_TOKEN``
    the installed hook commands pin, and whether this environment holds one.
    It does not probe the engine, so it cannot
    say whether the engine has a key; it says what this install would need.
    """
    base_url = get_config().engine.base_url
    if not base_url:
        return "engine:         none; the hooks write the store in-process (no key needed)"
    base_url = url_without_userinfo(base_url)
    from particles.api._keys import Role, client_token_warning, role_from_prefix
    from particles.secrets import get_particles_api_key

    api_key = get_particles_api_key()
    legacy = None if api_key == "dev-key" else api_key
    # The installed hooks run with the token their settings entries carry,
    # which need not be the one in this shell (security review 2026-10-27-1,
    # W68): a token pinned there is judged first, and when the hooks pin one
    # this shell's is not judged at all (security review 2026-10-28-1, O65).
    pinned_tokens = _hook_pinned_tokens()
    for pinned in pinned_tokens:
        warning = client_token_warning(pinned, legacy, agent_side=True)
        if warning is not None:
            return (
                f"engine:         ⚠ {base_url}: the installed hooks pin a PARTICLES_ENGINE_TOKEN "
                f"in their settings entries, and {warning}"
            )
        if role_from_prefix(pinned) is Role.READER:
            return (
                f"engine:         ⚠ {base_url}: the installed hooks pin a PARTICLES_ENGINE_TOKEN "
                "in their settings entries that is a reader key, so the engine refuses the "
                "SessionEnd harvest and the session exposure; mint the hooks their own key "
                "with `particles engine keys create --role agent --principal claude-code-hook "
                "--reserved-principal`"
            )
    if pinned_tokens:
        return (
            f"engine:         {base_url}, with the PARTICLES_ENGINE_TOKEN the installed hooks "
            "pin an agent key; the harvest writes memory files only under a harvester "
            "principal (claude-code-hook)"
        )
    if get_engine_token_optional() is None:
        return (
            f"engine:         ⚠ {base_url} with no PARTICLES_ENGINE_TOKEN; a keyless "
            "engine serves this install reads only, so the SessionEnd harvest and the "
            "session exposure are refused"
        )
    token = get_engine_token_optional()
    # The hooks are agent-side: an agent with a shell reads their environment,
    # so an operator key there is the operator's authority handed to the agent.
    # The prefix is read locally; the engine is not asked.
    warning = client_token_warning(token, legacy, agent_side=True)
    if warning is not None:
        return f"engine:         ⚠ {base_url}: {warning}"
    # The role is read from the prefix of the token in this process's own
    # environment; the hooks' environment is what settings.json gives them.
    # Every token past the warning above is a per-client key, so the prefix
    # names a role (security review 2026-10-26-1, W77).
    if role_from_prefix(token or "") is Role.READER:
        return (
            f"engine:         ⚠ {base_url}: PARTICLES_ENGINE_TOKEN in this environment is "
            "a reader key, so the engine refuses the SessionEnd harvest and the session "
            "exposure; mint the hooks their own key with `particles engine keys create "
            "--role agent --principal claude-code-hook --reserved-principal`"
        )
    return (
        f"engine:         {base_url}, with PARTICLES_ENGINE_TOKEN in this environment an "
        "agent key; the harvest writes memory files only under a harvester principal "
        "(claude-code-hook)"
    )


def _harvest_roots_line() -> str | None:
    """Whether the engine key the harvest uses records this client's directories.

    ``None`` when the hooks write in-process, or the key is not a harvester's.
    It asks the engine (``/auth/whoami``) for the roots the key records and
    names any directory this client's harvest writes from that the key lacks,
    since the engine refuses a memory or archive file outside every recorded
    root. It judges the key in this environment, so when the installed hooks
    pin a different one it says so and judges nothing.
    """
    if not get_config().engine.base_url:
        return None
    token = get_engine_token_optional()
    pinned = _hook_pinned_tokens()
    if token is None or (pinned and pinned != [token]):
        return (
            "harvest roots: not checked; run the doctor with the PARTICLES_ENGINE_TOKEN the "
            "installed hooks use"
        )
    from particles.api.client.http import EngineHttpError, HttpBackend

    try:
        who = run_armed(HttpBackend().whoami())
    except (EngineHttpError, OSError, RuntimeError) as exc:
        return f"harvest roots: not checked; the engine did not answer ({type(exc).__name__})"
    if who.get("principal") not in {"claude-code-hook", "audit"} or who.get("role") != "agent":
        return None
    memory, archive = local_harvest_roots()
    missing = [
        root
        for wanted, recorded in (
            (memory, who.get("memory_roots") or []),
            (archive, who.get("archive_roots") or []),
        )
        for root in wanted
        if root not in recorded
    ]
    if not missing:
        return "harvest roots: the hook key records this client's memory and archive directories"
    return (
        f"harvest roots: ⚠ the hook key records none of {', '.join(missing)}, so the engine "
        "refuses the memory and archive files there; mint the hooks a key with "
        f"{_harvest_mint_flags()}"
    )


def _harvest_mint_flags() -> str:
    """The ``keys create`` flags that record this client's harvest directories."""
    memory, archive = local_harvest_roots()
    flags = [f"--memory-root {shlex.quote(root)}" for root in memory]
    flags += [f"--archive-root {shlex.quote(root)}" for root in archive]
    return " ".join(flags)


def _hook_settings_paths() -> list[Path]:
    """The user settings and this repository's project-scope settings files, without repeats."""
    root = repository_root(Path.cwd())
    paths = [Path.home() / ".claude" / "settings.json"] + [
        root / ".claude" / name for name in ("settings.local.json", "settings.json")
    ]
    return list(dict.fromkeys(paths))


def _read_doctor_settings(path: Path) -> dict[str, Any] | None:
    """A settings file's JSON object as the doctor reads it, or ``None``.

    The user file is the operator's own and may be a deliberate link (a
    dotfiles checkout). A project-scope file sits in a repository a clone may
    supply, so it is read as the install reads it: through one descriptor that
    follows no link and never blocks, a regular file of one name that resolves
    inside the repository, owned by this account, and capped at
    ``local_files.max_bytes``. Anything else is skipped, so a FIFO or a link
    a clone ships never holds the doctor and a planted file never stands for
    the installed hooks (security review 2026-10-28-2, AA20).
    """
    from particles.corpus.local_read import read_local_file, within

    try:
        if path == Path.home() / ".claude" / "settings.json":
            text = path.read_text(encoding="utf-8")
        else:
            read = read_local_file(
                path,
                max_bytes=get_config().local_files.max_bytes,
                allowed=within(path.parent.parent),
            )
            getuid = getattr(os, "getuid", None)
            if getuid is not None and read.stat.st_uid != getuid():
                return None
            text = read.content.decode("utf-8")
        settings = json.loads(text)
    except (OSError, ValueError):
        return None
    return settings if isinstance(settings, dict) else None


def _hook_pinned_tokens() -> list[str]:
    """The distinct ``PARTICLES_ENGINE_TOKEN`` values the installed Particles hooks pin."""
    tokens: dict[str, None] = {}
    for path in _hook_settings_paths():
        settings = _read_doctor_settings(path)
        if settings is None:
            continue
        for command in particles_hook_commands(settings):
            token = command_env_pins(command).get("PARTICLES_ENGINE_TOKEN")
            if token:
                tokens[token] = None
    return list(tokens)


def _run_doctor(store: str) -> bool:
    from particles.config import (
        describe_config_resolution,
        environment_notes,
        user_config_path,
        validate_config,
    )

    typer.echo(f"particles hook doctor — store '{store}'")
    typer.echo(f"  cwd:            {os.getcwd()}")

    validate_config()
    resolution = describe_config_resolution()
    if resolution.missing is not None:
        typer.echo(
            f"  config:         ⚠ PARTICLES_CONFIG names {resolution.missing}, which does not "
            "exist (compiled defaults + env overrides only)"
        )
    elif resolution.path is None:
        typer.echo("  config:         none (compiled defaults + env overrides only)")
    else:
        typer.echo(f"  config:         {resolution.origin} {resolution.path}")
    if resolution.ignored is not None:
        typer.echo(
            f"  ignored:        {resolution.ignored} is not read; set "
            f"PARTICLES_CONFIG={resolution.ignored} or move it to {user_config_path()}"
        )
    if os.environ.get("PARTICLES_CONFIG"):
        typer.echo(f"  PARTICLES_CONFIG env: {os.environ['PARTICLES_CONFIG']}")
    if os.environ.get("DATABASE_URL"):
        typer.echo(f"  DATABASE_URL env:     {os.environ['DATABASE_URL']}")
    for note in environment_notes():
        typer.echo(f"  env:            {note}")

    for line in _check_hook_timeouts():
        typer.echo(f"  {line}")

    for line in _check_config_pins():
        typer.echo(f"  {line}")

    for line in _check_install():
        typer.echo(f"  {line}")

    typer.echo(f"  {_engine_line()}")
    roots_line = _harvest_roots_line()
    if roots_line is not None:
        typer.echo(f"  {roots_line}")

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
                "  ✗ DSN path is working-directory-relative — it names a DIFFERENT file in "
                "each directory, so the lifecycle hooks refuse this store and do nothing. "
                "Bake an absolute path via `init claude-code`."
            )
            return False
        if not exists:
            typer.echo("  ✗ Database file does not exist. Run `particles db init` for this store.")
            return False

    tables_ok, detail = _check_tables(store)
    typer.echo(f"  corpus tables:  {detail}")
    if not tables_ok:
        return False

    schema_ok, schema_lines = _check_schema(store)
    for line in schema_lines:
        typer.echo(f"  {line}")

    blob_lines = _check_blobs(store)
    for line in blob_lines:
        typer.echo(f"  {line}")

    for line in _check_memory_dirs(store):
        typer.echo(f"  {line}")

    for line in _check_observer_scope(store):
        typer.echo(f"  {line}")

    for line in _check_refresh_roots(store):
        typer.echo(f"  {line}")

    for line in _check_retired_stop_hook():
        typer.echo(f"  {line}")

    for line in _check_model(store):
        typer.echo(f"  {line}")

    for line in _check_egress():
        typer.echo(f"  {line}")

    if not schema_ok:
        typer.echo("  ✗ Store resolves but is behind this install's migrations. Run the")
        typer.echo("    `db init` command on the schema: line above, then re-run the doctor.")
        return False
    typer.echo("  ✓ Store resolves and is initialized from this directory.")
    return True


def _check_config_pins() -> list[str]:
    """Warn about an installed hook command or MCP registration with no absolute config pin.

    Since the loader reads ``PARTICLES_CONFIG`` or the user file and
    never the working directory, and ``init claude-code`` pins
    ``PARTICLES_CONFIG`` into every hook and the MCP registration so each one
    states its config. An entry without the pin, or with a relative one, is
    a pre-upgrade or hand-made install whose config depends on the
    environment it happens to start in. Reported, never fatal.
    """
    fix = "Re-run `particles init claude-code` to pin it."
    lines: list[str] = []
    for path in _hook_settings_paths():
        settings = _read_doctor_settings(path)
        if settings is None:
            continue
        for command in particles_hook_commands(settings):
            pins = command_env_pins(command)
            problem = _config_pin_problem(pins.get("PARTICLES_CONFIG"))
            hook_fix = fix
            if problem is None:
                handle = _command_store(command)
                if handle == "default":
                    problem = _store_pin_problem(pins)
                else:
                    # A named store's path comes from the pinned config, which
                    # no env pin overrides, so the hook refuses it (security
                    # review 2026-10-25, Q5) until that config names it absolutely.
                    problem = _named_store_pin_problem(pins, handle)
                    hook_fix = (
                        f"Write an absolute path under storage.stores.{handle} in that config."
                    )
            if problem is None:
                problem = _engine_pin_problem(pins, "writes to")
            if problem is not None:
                lines.append(f"config pin:     ⚠ a Particles hook in {path} {problem}.")
                lines.append(f"                {hook_fix}")
                break
    for registration in particles_mcp_registrations():
        problem = _config_pin_problem(registration.env.get("PARTICLES_CONFIG"))
        if problem is None:
            # The MCP read tools always read the default store (security review N17).
            problem = _store_pin_problem(registration.env)
        if problem is None:
            problem = _engine_pin_problem(registration.env, "reads from and writes to")
        if problem is not None:
            lines.append(f"config pin:     ⚠ the MCP server '{registration.name}' {problem}.")
            lines.append(f"                {fix}")
    return lines


def _engine_pin_problem(pins: dict[str, str], use: str) -> str | None:
    """A ``PARTICLES_ENGINE_BASE_URL`` pin that is not the empty one ``init`` writes, if so.

    Unpinned, a hook or server follows an engine the harness's environment
    names (security review 2026-10-25-2, R41; E2). Pinned to a URL, it follows
    the engine an install between 1.220.2 and 1.221.3 inherited from its own
    environment, not the pinned config's (2026-10-27-1, W27).
    """
    if "PARTICLES_ENGINE_BASE_URL" not in pins:
        return (
            f"does not pin PARTICLES_ENGINE_BASE_URL, so the engine it {use} "
            "can come from the session's environment"
        )
    pinned = pins["PARTICLES_ENGINE_BASE_URL"]
    if pinned:
        return (
            f"pins PARTICLES_ENGINE_BASE_URL={url_without_userinfo(pinned)}, so the engine it "
            f"{use} is that one, not the pinned config's engine.base_url"
        )
    return None


def _command_store(command: str) -> str:
    """The store a hook command names (``--store <handle>``), ``default`` when it names none."""
    tokens = command.split()
    for i, token in enumerate(tokens[:-1]):
        if token == "--store":
            return tokens[i + 1]
    return "default"


def _store_pin_problem(env: dict[str, str]) -> str | None:
    """A ``default`` store that resolves against the working directory, if it does."""
    dsn = pinned_default_dsn(env)
    if _dsn_is_cwd_relative(dsn):
        return (
            f"has no DATABASE_URL pin, so its default store ({dsn}) is read from each "
            "session's working directory"
        )
    # An install run inside a clone pins the clone's own ./particles.db
    # absolutely, as it once did a config file (security review N3, W16). The
    # judgement is the local refresh's own, on the resolved file, so the doctor
    # and the refresh agree on a pin that links into a clone and on a store
    # under a home-directory .git (2026-10-27-1, W16), and word the cause, and
    # its fix, alike (2026-10-28-1, O60).
    cause = _store_file_cause(dsn)
    return None if cause is None else f"pins its default store to {cause}"


def _named_store_pin_problem(env: dict[str, str], handle: str) -> str | None:
    """A named store the local refresh would not trust as the operator's, if it is one.

    One whose path in the hook's config resolves against the working directory,
    or whose file the refresh refuses: inside a git work tree, writable by other
    users, or another account's (security review 2026-10-28-1, O30, E23).
    """
    dsn = pinned_store_dsn(env, handle)
    if dsn is None:
        return None
    if _dsn_is_cwd_relative(dsn):
        return (
            f"names the store '{handle}', whose path ({dsn}) is read from each session's "
            "working directory, so the hook refuses it"
        )
    db_file = sqlite_file_path(dsn)
    if db_file is None:
        return None
    repository = enclosing_work_tree(Path(db_file))
    if repository is not None and _store_file_cause(dsn) is not None:
        return (
            f"names the store '{handle}', whose file {db_file} is inside the git work tree "
            f"{repository}, which a clone can supply"
        )
    # Every other cause is worded as the refresh judges it, with its fix: the
    # file, a SQLite sidecar or the directory holding it (2026-10-28-1, O60).
    cause = _store_file_cause(dsn)
    return None if cause is None else f"names the store '{handle}', whose file is {cause}"


def _store_file_cause(dsn: str) -> str | None:
    """``"<file>, <why>"`` for an absolute SQLite DSN the local refresh refuses, else ``None``."""
    # Deferred like the doctor's other refresh reads: the lifecycle hooks
    # import this module and never reach the corpus fetch layer.
    from particles.corpus.fetch import store_file_judgement

    db_file = sqlite_file_path(dsn)
    if db_file is None:
        return None
    judgement = store_file_judgement(Path(db_file))
    return None if judgement is None else f"{db_file}, {judgement.cause}"


def _config_pin_problem(pin: str | None) -> str | None:
    if pin is None:
        return "has no PARTICLES_CONFIG pin"
    if not Path(pin).is_absolute():
        return f"pins a relative PARTICLES_CONFIG ({pin})"
    # An install once turned a relative pin into an absolute pin of the
    # working directory's file (security review 2026-10-25-2, R10, R19). The
    # config chooses the store, the engine and its endpoints, so it is judged
    # as the store file is: owner, mode, ACL and directory as well as the work
    # tree (2026-10-28-1, G55).
    from particles.corpus.fetch import store_file_judgement

    judgement = store_file_judgement(Path(pin), sidecars=())
    if judgement is not None:
        return f"pins PARTICLES_CONFIG={pin}, {judgement.cause}"
    return None


def _check_hook_timeouts() -> list[str]:
    """Warn about an installed Particles SessionEnd hook that still has no ``timeout``.

    Without one Claude Code cancels every SessionEnd hook after a shared 1.5 s
    budget, and a cancelled harvest writes no hook-log record, so the loss is
    invisible from ``particles hook log``. Checks the user-level settings and
    this repository's project-scope files. Reported, never fatal: the store
    itself is usable, and the hook's catch-up sweep back-fills the missed
    transcripts once a run is allowed to finish.
    """
    root = repository_root(Path.cwd())
    user_fix = "Re-run `particles init claude-code` to add one."
    project_fix = "Re-run `particles init claude-code --project` to add one."
    paths = [(Path.home() / ".claude" / "settings.json", user_fix)] + [
        (root / ".claude" / name, project_fix) for name in ("settings.local.json", "settings.json")
    ]
    lines: list[str] = []
    seen: set[Path] = set()
    for path, fix in paths:
        if path in seen:  # the repository is $HOME; the user-level reading wins
            continue
        seen.add(path)
        settings = _read_doctor_settings(path)
        if settings is not None and untimed_session_end_hooks(settings):
            lines.append(f"hook timeout:   ⚠ the SessionEnd hook in {path} has no timeout.")
            lines.append(
                "                Claude Code cancels it after 1.5 s, before the harvest runs."
            )
            lines.append(f"                {fix}")
    return lines


def _git(root: Path, *args: str) -> str | None:
    """One ``git -C root`` query's stripped stdout, or ``None`` when it fails."""
    # Drop a caller's GIT_DIR / GIT_WORK_TREE (set inside git hooks), which
    # would override ``-C`` and describe some other repository. ``PATH`` keeps
    # only the entries no other user may write, as the projection's and the
    # subjects' git do, so a relative or shared entry supplies no ``git``
    # (security review 2026-10-30-1, E31).
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["PATH"] = pinned_subprocess_env({"PATH": os.environ.get("PATH", os.defpath)})["PATH"]
    try:
        done = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", *GIT_LOCAL_ARGS, "-C", str(root), *args],  # noqa: S607 — git from PATH
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = done.stdout.strip()
    return out if done.returncode == 0 and out else None


def _install_pin(root: Path) -> InstallPin | None:
    """Where the running install's source checkout sits, or ``None`` when it is not one."""
    if not (root / ".git").exists():
        return None
    head = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if head is None:
        return None
    tags = _git(root, "tag", "--list", "v*", "--sort=-v:refname")
    return InstallPin(
        branch=None if head == "HEAD" else head,
        tag=_git(root, "describe", "--tags", "--exact-match", "HEAD"),
        newest_tag=tags.splitlines()[0] if tags else None,
    )


def _check_install() -> list[str]:
    """Report the install the hooks and MCP server run, and whether it has gone stale.

    Reads the user-level settings and this repository's project-scope files for
    the hook commands, ``~/.claude.json`` for the MCP server, and, when the
    running install is a git checkout, its branch and release tags (local refs
    only; the doctor never fetches). Reported, never fatal.
    """
    import particles

    commands: list[str] = []
    for path in _hook_settings_paths():
        settings = _read_doctor_settings(path)
        if settings is not None:
            commands.extend(particles_hook_commands(settings))
    argv0 = Path(sys.argv[0])
    install_root = Path(particles.__file__).resolve().parent.parent
    return install_report(
        root=install_root,
        version=particles.__version__,
        pin=_install_pin(install_root),
        hook_commands=commands,
        mcp_exes=particles_mcp_executables(),
        running_exe=str(argv0) if argv0.name == "particles" and argv0.is_absolute() else None,
    )


def _check_schema(store: str) -> tuple[bool, list[str]]:
    """Compare the store's migration with this install's head, in both directions.

    Behind is the failure: the hooks, the MCP server and scheduled
    consolidation never migrate, so code that reads a column a pending
    migration adds fails until ``db init`` runs with this install (an install
    advanced past a release with a migration, 2026-10-17). Ahead is advisory: a
    newer Particles sharing the store (a development checkout, say) upgraded
    it, and an older install then reads tables it does not understand. A store
    with no ``alembic_version`` table reports nothing.
    """
    from particles.api.cli._schema import behind_summary, migrate_command
    from particles.db import schema_state

    try:
        state = run_armed(schema_state(store))
    except Exception:  # noqa: BLE001 — doctor reports, never crashes
        return True, []
    if state is None:
        return True, []
    if state.unknown:
        return True, [
            f"schema:         ⚠ the store is at migration {', '.join(state.unknown)}, which this "
            "install does not know.",
            "                A newer Particles migrated it. Upgrade the install the hooks and the "
            "MCP server run.",
        ]
    if not state.pending:
        return True, [f"schema:         migration {', '.join(state.stored)}, this install's head"]
    pad = " " * 16
    return False, [
        f"schema:         ✗ {behind_summary(state)}.",
        f"{pad}Scheduled consolidation, the hooks and the MCP server fail on what",
        f"{pad}{', '.join(state.pending)} {'adds' if len(state.pending) == 1 else 'add'} "
        "until this install migrates the store:",
        f"{pad}  {migrate_command(store)}",
    ]


def _check_model(store: str) -> list[str]:
    """Name the extraction model the hooks run and whether a calibration reaches it.

    The compiled default can move in a release, and a calibration applies only
    to its own pairing and extractor version, so this says which model is in
    force, where it came from, and what the general extractor's claims will be
    stamped with. Reported, never fatal.
    """
    from particles.db import session_scope
    from particles.extraction.calibration import scaler_for_record, version_mismatch_reason
    from particles.extraction.general import EXTRACTOR_ID, EXTRACTOR_VERSION
    from particles.store.extractor_store import get_calibration

    llm = get_config().llm
    sel = llm.for_purpose("extraction")
    pairing = f"{sel.provider}:{sel.model}"
    lines = [f"model:          extraction {pairing} ({llm.selection_source('extraction')})"]

    async def _probe() -> Any:
        async with session_scope(store) as session:
            return await get_calibration(session, EXTRACTOR_ID, pairing)

    try:
        calibration = run_armed(_probe())
    except Exception as exc:  # noqa: BLE001 — a diagnostic must not fail the doctor
        lines.append(f"                calibration: could not check the store ({exc})")
        return lines
    if calibration is None:
        status = "none for this pairing, so its claims are EXTRACTOR_DIRECT"
    elif scaler_for_record(calibration) is None:
        status = "NOT APPLIED (fitted before), so its claims are EXTRACTOR_DIRECT"
    elif (reason := version_mismatch_reason(calibration, EXTRACTOR_VERSION)) is not None:
        status = f"NOT APPLIED ({reason}), so its claims are EXTRACTOR_DIRECT"
    else:
        status = "applies"
    lines.append(f"                {EXTRACTOR_ID} calibration: {status}")
    return lines


def _check_retired_stop_hook() -> list[str]:
    """Warn about a Stop entry left by an install made while agent extraction was on.

    The verb it runs is a retired no-op (deprecated), so the entry
    costs a process start at the end of every turn and nothing else. Re-running
    the installer drops it. Reported, never fatal.
    """
    try:
        settings = json.loads((Path.home() / ".claude" / "settings.json").read_text("utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(settings, dict):
        return []
    if not any(" hook stop " in f"{c} " for c in particles_hook_commands(settings)):
        return []
    return [
        "stop hook:      ⚠ a retired Particles Stop hook is still installed (it does nothing).",
        "                Re-run `particles init claude-code` to remove it.",
    ]


#: The LLM purposes the hooks themselves depend on, and what each feeds.
_HOOK_PURPOSES = {
    "extraction": "background extraction of harvested transcripts",
    "use_judge": "SessionEnd utility mining",
}


def _check_egress() -> list[str]:
    """Report the ``network.egress`` posture as the hooks will meet it.

    One summary line from the posture report ``config validate`` prints, plus a
    ⚠ for each refused purpose the hooks use and a ✗ when the encoder is not
    cached under a refusing posture (every SessionEnd would run without it).
    Reported, never fatal: a refusing posture is a choice, not a broken store.
    """
    from particles.llm.registry import purpose_allowed, purpose_endpoint
    from particles.network_posture import posture

    report = posture()
    lines = [f"egress:         {report.summary()}"]
    if report.posture == "open":
        return lines
    for purpose, feeds in _HOOK_PURPOSES.items():
        if not purpose_allowed(purpose):
            host, port, _ = purpose_endpoint(purpose)
            lines.append(
                f"                ⚠ {purpose} refused ({host}:{port}): {feeds} is unavailable"
            )
    for item in report.items:
        if item.kind == "encoder" and item.severity != "ok":
            lines.append(f"                ✗ encoder {item.name}: {item.note}")
    return lines


def _check_refresh_roots(store: str) -> list[str]:
    """Report enrolled local sources the refresh would refuse to read.

    The refresh reads rule files only under recorded roots, so after the
    upgrade that introduced them, and until a ``rules sync``, the project rule
    files are refused. An entry a depositor the refresh does not trust enrolled
    is refused inside those roots too (security review N21). Each cause gets
    its own line and fix. Reported, never fatal.
    """
    from particles.corpus.fetch import RefusalCounts, count_refused_local_entries
    from particles.db import session_scope

    async def _count() -> RefusalCounts:
        async with session_scope(store) as session:
            return await count_refused_local_entries(session)

    try:
        refused = run_armed(_count())
    except Exception as exc:  # noqa: BLE001 — a diagnostic must not fail the doctor
        return [f"refresh roots:  could not check the store: {exc}"]
    # One line per cause, each with its own fix (an outside-root entry needs a
    # root recorded; an untrusted depositor's does not change on re-enrolment).
    return [
        f"{'refresh roots:  ⚠ ' if i == 0 else '                ⚠ '}{line}"
        for i, line in enumerate(refused.lines())
    ]


def _check_observer_scope(store: str) -> list[str]:
    """Report what a session here reads: the whole store, or its project's view.

    The one non-obvious state is worth a line of its own: ``project`` is set but
    the store has never been rescoped, so the surfaces are still store-wide.
    Reported, never fatal.
    """
    from particles.db import session_scope
    from particles.operations.query.observer_scope import lens_may_engage

    mode = get_config().claude_code.observer_scope
    if mode != "project":
        return [f"observer scope: {mode} (every session sees the whole store)"]

    async def _probe() -> bool:
        async with session_scope(store) as session:
            return await lens_may_engage(session)

    try:
        rescoped = run_armed(_probe())
    except Exception as exc:  # noqa: BLE001 — a diagnostic must not fail the doctor
        return [f"observer scope: project (could not check the store: {exc})"]
    if not rescoped:
        return [
            "observer scope: project, but NOT in effect — this store has not been rescoped.",
            f"                Run `particles memory rescope --store {store}` (it only adds tags).",
        ]
    return [
        "observer scope: project (a session sees global beliefs plus its project's)",
        *_observer_binding_lines(Path.cwd()),
    ]


def _observer_binding_lines(cwd: Path) -> list[str]:
    """Where a session's digest key and its MCP server's key can part.

    A bound server keys on the directory it starts in; the hooks key on the
    session's launch directory, and fall back to the transcript directory's
    name when no recorded ``cwd`` reproduces it. The fallbacks are reported
    after the fact, and so is a server key that differs from the hooks' last
    key for this directory. Reported, never fatal.
    """
    from particles.api.cli._claude_code import (
        particles_mcp_registrations,
        project_key_collision_note,
        read_hook_log_tail,
        recorded_project_roots,
        server_project_key,
    )

    pad = " " * 16
    registrations = particles_mcp_registrations()
    if not registrations:
        lines = [f"{pad}MCP server: not registered at user scope"]
    elif all(r.bound for r in registrations):
        lines = [f"{pad}MCP server: bound to each session's project (--project-observer cwd)"]
    else:
        lines = [
            f"{pad}⚠ MCP server is not bound, so its reads are store-wide; re-run "
            "`particles init claude-code` to add --project-observer cwd"
        ]
    server_key = server_project_key(cwd)
    shared_by = recorded_project_roots(server_key) if server_key else []
    fallbacks: dict[str, str] = {}
    hooks_key: str | None = None
    for raw in read_hook_log_tail(2000):
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if entry.get("event") != "session-start" or "project_resolved_from" not in entry:
            continue
        key = str(entry.get("project_key") or "")
        if entry.get("project_resolved_from") != "repository" and key:
            fallbacks[key] = str(entry["project_resolved_from"])
        if entry.get("cwd") == str(cwd):
            hooks_key = key
    lines.append(f"{pad}key here: a bound server takes {server_key or '(none: unbound at /)'}")
    if server_key and len(shared_by) > 1:
        # a key two repositories share is disclosed, never fixed here.
        lines.append(f"{pad}⚠ {project_key_collision_note(server_key, shared_by)}")
    if hooks_key is not None:
        lines.append(f"{pad}          the hooks last resolved {hooks_key or '(none)'}")
        if server_key is not None and hooks_key != server_key:
            lines.append(
                f"{pad}⚠ the digest and the MCP server read different projects here; "
                "the server's key comes from the directory it really started in"
            )
    if fallbacks:
        shown = ", ".join(f"{k} ({how})" for k, how in sorted(fallbacks.items())[:3])
        lines.append(
            f"{pad}⚠ {len(fallbacks)} recent session key(s) came from a fallback, where a "
            f"bound server may read another project: {shown}"
        )
    return lines


def _last_harvest_refusals() -> int:
    """How many files the engine refused at the newest logged session end (AB49)."""
    from particles.api.cli._claude_code import read_hook_log_tail

    for raw in reversed(read_hook_log_tail(2000)):
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("event") == "session-end":
            refused = entry.get("refused")
            return len(refused) if isinstance(refused, list) else 0
    return 0


def _check_memory_absences(store: str) -> list[str]:
    """One line per bound memory directory holding unrecorded deletions.

    Read only. Reported, never fatal.
    """
    from particles.api.cli._claude_code import discover_memory_dirs

    lines: list[str] = []
    try:
        dirs = discover_memory_dirs(store).bound
        reports = [run_armed(_survey_memory_absences(store, d)) for d in dirs]
    except Exception as exc:  # noqa: BLE001 — a diagnostic must not fail the doctor
        return [f"memory absences: could not check the store: {exc}"]
    for d, report in zip(dirs, reports, strict=True):
        if report is None or not report.missing:
            continue
        if report.skipped == "walk_incomplete":
            why = "the directory could not be read in full, so none is recorded"
        elif report.refusal == "first":
            why = "the first deletion here, which the hook leaves to you"
        elif report.refusal == "mass":
            why = f"{report.absent_fraction:.0%} of the directory at once, held back"
        else:
            why = "recorded at the next session end"
        lines.append(
            f"memory absences: ⚠ {len(report.missing)} file(s) in {d} no longer found ({why}); "
            "`particles memory absences` lists them"
        )
    return lines


def _check_memory_dirs(store: str) -> list[str]:
    """Report which memory directory a session here writes to, whether ``store``
    serves it, and any strays.

    A directory is harvested and rendered only by the store the installed hooks
    name (``_claude_code.projection_refusal``); a refusal is reported with its
    reason, because the SessionEnd hook records it only in the hook log.

    A linked worktree's sessions key their memory on the main checkout, so the
    directory beside their transcripts is never read by Claude Code; older
    versions had the projection create one there anyway. Reported,
    never fatal, and never deleted: the files are small, and removing a user's
    file is not this verb's call.
    """
    cwd = Path.cwd()
    projects_root = Path.home() / ".claude" / "projects"
    key = claude_project_slug(repository_root(cwd))
    memory_dir = projects_root / key / "memory"
    state = "exists" if memory_dir.is_dir() else "not created yet"
    lines = [f"memory dir:     {memory_dir} ({state})"]
    if claude_project_slug(cwd) != key:
        lines.append("                (a session here shares its repository's memory directory)")
    # Read as the harvest reads it: no link followed, never blocking, capped
    # (security review 2026-10-28-2, AA20).
    _, memory_bytes = _read_memory_md(memory_dir / "MEMORY.md")
    if memory_bytes is not None:
        # the index sits below the region, so an overgrown file
        # loses its tail from every session with no error anywhere.
        warning = memory_load_window_warning(memory_bytes.decode("utf-8", errors="replace"))
        if warning is not None:
            lines.append(f"memory size:    {warning}")
    refusal = projection_refusal(store, memory_dir, (repository_root(cwd),))
    if refusal is None:
        lines.append(f"memory binding: store '{store}' is the store the installed hooks name")
    else:
        lines.append(f"memory binding: NOT SERVED, {refusal}")
    lines.extend(_check_memory_absences(store))
    refused = _last_harvest_refusals()
    if refused:
        lines.append(
            f"harvest:        ⚠ the engine refused {refused} file(s) at the last session end; "
            "`particles hook log` names them (a hook key whose roots miss this client's "
            "memory directories; mint it with the command `particles init claude-code` prints)"
        )
    strays = stray_memory_dirs(projects_root)
    if strays:
        lines.append(
            f"stray memory:   {len(strays)} director{'y' if len(strays) == 1 else 'ies'} "
            "Claude Code never reads (created for linked worktrees; safe to delete):"
        )
        lines.extend(f"                {stray}" for stray in strays)
    return lines


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
        return run_armed(_probe())
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
        count = run_armed(_count())
    except OperationalError:
        return False, "MISSING — run `particles db init` (the harvest target has no tables)"
    except Exception as exc:  # noqa: BLE001 — doctor reports, never crashes
        return False, f"could not read ({type(exc).__name__}: {exc})"
    return True, f"present ({count} corpus entries)"
