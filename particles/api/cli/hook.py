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
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer

from particles.api.cli import _thin_client as thin
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
    project_transcript_dirs,
    projection_enabled,
    projection_refusal,
    read_hook_log_tail,
    read_memory_files,
    record_project_root,
    redact_secrets,
    repository_root,
    resolve_session_project,
    session_start_budget,
    state_dir,
    stray_memory_dirs,
    transcript_started_at,
    truncate_on_line_boundary,
    untimed_session_end_hooks,
    walk_memory_dir,
    write_private_text,
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
from particles.config_tuning import TuningReport, compare_tuning, env_overridden_tuning, preview
from particles.corpus.harvest_marks import HARNESS_SOURCES
from particles.corpus.rule_sources import refresh_policy_for
from particles.egress import run_armed
from particles.secrets import (
    EngineTokenFileError,
    check_engine_token_file,
    get_engine_token_optional,
    redact_known_secrets,
)
from particles.store.session_exposure_store import SessionExposure, ShownBelief
from particles.url_safety import GIT_LOCAL_ARGS, pinned_subprocess_env, url_without_userinfo

if TYPE_CHECKING:
    import httpx

    from particles.api.cli._claude_code import MemoryDirWalk
    from particles.api.client.base import TextDepositOutcome
    from particles.api.client.http import HttpBackend
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


@contextlib.contextmanager
def _phase(record: dict[str, Any], name: str) -> Iterator[None]:
    """Time one hook step into the hook log's ``phases_ms``.

    A step the deadline cuts short is also named as ``interrupted_in``, so the
    hook log says where a timed-out run spent its time, not only that it ran out.
    """
    started = time.monotonic()
    try:
        yield
    except asyncio.CancelledError:
        # The hook deadline cancels the body; a step's own timeout is not this.
        record["interrupted_in"] = name
        raise
    finally:
        phases = record.setdefault("phases_ms", {})
        phases[name] = round((time.monotonic() - started) * 1000, 1)


# ---------------------------------------------------------------------------
# Guard rail: never fail the session
# ---------------------------------------------------------------------------


def _token_file_refusal() -> EngineTokenFileError | None:
    """The token-file refusal a hook against a remote engine meets, if any."""
    try:
        if get_config().engine.base_url:
            get_engine_token_optional()
    except EngineTokenFileError as exc:
        return exc
    except Exception:  # noqa: BLE001 — anything else is the body's to meet and log
        return None
    return None


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
    if _refused(store, record, started):
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
        # a SessionStart that kept the loaded region is degraded,
        # reported apart from an error.
        record["outcome"] = "degraded" if record.get("degraded") else "ok"
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


def _refused(store: str, record: dict[str, Any], started: float) -> bool:
    """Log and return ``True`` when a hook process must not run its body at all.

    Shared by the hook verbs and the SessionEnd worker: a relative config pin,
    a store that resolves against the working directory, or an engine token
    file that fails its checks.
    """
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
    elif (refusal := _working_directory_store_refusal(store)) is not None:
        record["outcome"] = "refused"
        record["error"] = refusal
    elif (token_refusal := _token_file_refusal()) is not None:
        # A thin client whose key file fails a check sends nothing.
        record["outcome"] = "refused"
        record["reason"] = "token-file-refused"
        record["check"] = token_refusal.check
        record["error"] = str(token_refusal)
    else:
        return False
    record["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
    append_hook_log(record)
    return True


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
    with _phase(record, "freshness"):
        decision = await _digest_decision(store, payload)
    shown: list[ShownBelief] = list(decision.loaded)
    # deletions the hook held back are named at session start.
    with _phase(record, "absence_notice"):
        notice = await _absence_notice(store, payload, record)
    # claims an automatic rung retired since this observer's
    # previous digest are named whichever way the session starts.
    retired_since = _previous_render(store, observer)
    rendered_at = datetime.now(UTC)
    if decision.action == "degraded":
        # the freshness check ran out its share. The region the
        # harness loaded is the store's view as of the last render, so the
        # session keeps it rather than wait on a full digest a busy machine
        # does not finish. The retirement window stays where it was, so the
        # next session names what this one could not.
        record["degraded"] = "freshness-timeout"
        with contextlib.suppress(OSError):
            record["load_1m"] = round(os.getloadavg()[0], 2)
        await _record_exposure(
            store, payload, record, "skip", shown, project_key, observer, started
        )
        return _session_start_output(notice.lstrip("\n")) if notice else None
    if decision.action != "full":
        with _phase(record, "retirement_notice"):
            retired = await _retirement_notice(store, observer, retired_since)
        if retired is not None:
            _record_render(store, observer, rendered_at)
            notice = retired + notice if retired else notice
            if retired:
                record["retirements_disclosed"] = True
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
        if notice and not notice.startswith("\n"):
            # The retirement section leads; the digest's top-up follows it.
            digest, notice = notice + "\n" + digest, ""
        record["digest_mode"] = "projection-diff"
        action: Literal["full", "diff"] = "diff"
        # Diff lines carry their own `p-` ids; read them from what survived.
        shown += bullet_beliefs(digest)
    else:
        from particles.api.client import get_backend

        if observer is not None:
            record["observer_project"] = observer
        with _phase(record, "digest"):
            rendered = await get_backend().digest_located(store, observer, retired_since)
        _record_render(store, observer, rendered_at)
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
        with _phase(record, "exposure"):
            await asyncio.wait_for(
                get_backend().record_session_exposure(store, exposure), remaining
            )
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
    session its digest, so any error degrades to the full push. A check that
    outlives its share of the hook deadline returns ``degraded`` when the
    session loaded the region the projection last rendered, which it keeps,
    and the full push otherwise.
    """
    from particles.api.cli._memory_projection import degraded_decision, digest_decision

    budget = get_config().claude_code.hook_deadline_seconds * _FRESHNESS_DEADLINE_SHARE
    try:
        return await asyncio.wait_for(digest_decision(store, payload), budget)
    except TimeoutError:
        # out of its share, keep a current loaded region if there
        # is one; the full digest after a slow check rarely finishes either.
        log.debug("projection freshness check ran out of time", exc_info=True)
        try:
            kept = degraded_decision(payload)
        except Exception:  # noqa: BLE001 — a file read; the digest must survive
            kept = None
        return kept if kept is not None else DigestDecision("full")
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


@dataclass(frozen=True)
class RenderPlan:
    """The MEMORY.md render a harvest earned: run only after every deposit landed."""

    memory_dir: Path
    memory_md_text: str | None
    project_roots: tuple[Path, ...]


@hook_app.command("session-end-worker")
def session_end_worker_cmd(
    store: str = typer.Option(..., "--store", help="Memory store the harvest deposits into."),
) -> None:
    """Harvest every session the SessionEnd hand-off queued, then re-render MEMORY.md.

    The installed SessionEnd command starts this worker detached;
    it is safe to run by hand, and exits at once when another worker runs.
    """
    from particles.api.cli import _session_end_worker as worker
    from particles.api.cli._claude_code import session_end_spool_dir

    started = time.monotonic()
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": "session-end",
        "mode": "worker",
        "store": store,
    }
    if _refused(store, record, started):
        return
    deadline = get_config().claude_code.harvest.worker_deadline_seconds

    def write_log(entry: dict[str, Any]) -> None:
        # As the inline hooks do: a config-resolution warning reaches the log.
        warnings = config_resolution_warnings()
        if warnings:
            entry["config_warnings"] = warnings
        append_hook_log(entry)

    def cycle(batch: list[Any]) -> None:
        run_armed(
            worker.run_cycle(
                store,
                batch,
                harvest=harvest_session,
                render=render_session,
                write_log=write_log,
                deadline=deadline,
            )
        )

    try:
        worker.run_worker(store, session_end_spool_dir(), cycle=cycle, write_log=write_log)
    except Exception as exc:  # noqa: BLE001 — the worker never raises past its log
        record["outcome"] = "error"
        record["error"] = redact_known_secrets(f"{type(exc).__name__}: {exc}")[:300]
        record["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
        append_hook_log(record)


async def _session_end(store: str, payload: dict[str, Any], record: dict[str, Any]) -> str | None:
    """The inline SessionEnd body: harvest, then render (§6/§7).

    Installs from before run this verb; a current install hands off
    to :func:`session_end_worker_cmd`, which runs the same two halves.
    """
    plan = await harvest_session(store, payload, record)
    if plan is not None:
        await render_session(store, plan, record)
    return None


async def harvest_session(
    store: str, payload: dict[str, Any], record: dict[str, Any]
) -> RenderPlan | None:
    """Harvest one ended session; the render it earned, or ``None``."""
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
        with _phase(record, "transcript"):
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
        with _phase(record, "memory_files"):
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
        with _phase(record, "archive"):
            outcome = await _harvest_archive(store, memory_dir, project, refused=refused)
        if outcome is not None:
            if outcome.unchanged:
                unchanged += 1
            else:
                deposited.append((outcome.entry_id, outcome.snapshot_id))

    # (b″) Against a remote engine, push the session project's rule files and
    # the two user-level ones: the engine never reads this
    # client's paths from its own disk. Never fatal, inside a share of the
    # deadline, and nothing is sent when nothing changed. A rule file is not in
    # the memory directory, so a refused one does not hold the splice below.
    if get_config().rule_sources.enabled:
        from particles.api.client import get_backend

        if get_backend().remote:
            with _phase(record, "rule_push"):
                record["rule_push"] = await _push_session_rules(
                    store, project_roots[0] if project_roots else None
                )

    # (c) Catch-up sweep: SessionEnd does not fire on SIGKILL / crash, but the
    # transcript persists on disk — harvest is level-triggered, and the corpus
    # is the harvest state (§3c). Failures here were logged last time and are
    # simply retried now.
    if harvest_cfg.transcripts and transcript_path.name and transcript_path.parent.is_dir():
        with _phase(record, "sweep"):
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
        with _phase(record, "extract"):
            record["extracted"] = await _extract_inline(store, deposited)

    # Use evidence is mined by consolidation's pass 5 alone: it mines every
    # session harvested since its last run, pooled into one batch, with a
    # credential bound for it.
    if get_config().utility.mining.enabled and harvest_cfg.transcripts and session_id:
        record["utility"] = {"deferred": "consolidation"}

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
        return RenderPlan(memory_dir, memory_md_text, project_roots)
    return None


async def render_session(store: str, plan: RenderPlan, record: dict[str, Any]) -> None:
    """Render and splice the MEMORY.md region a harvest earned (§6/§7)."""
    from particles.api.cli._memory_projection import run_projection_cycle

    with _phase(record, "projection"):
        record["projection"] = await run_projection_cycle(
            store, plan.memory_dir, plan.memory_md_text, project_roots=plan.project_roots
        )


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


#: The share of the hook deadline the SessionEnd rule push may spend.
_RULE_PUSH_DEADLINE_SHARE = 0.2


async def _push_session_rules(store: str, project_root: Path | None) -> dict[str, Any]:
    """Push the session's rule files to the remote engine; the hook-log record of it.

    The project's files only when the session ran inside a repository (a
    directory with no ``.git`` above it is no project); the two user-level
    files whatever the session's directory.
    """
    from particles.api.cli._claude_code import find_repository_root
    from particles.api.cli._rule_push import (
        global_rule_files,
        push_rule_files,
        session_rule_files,
        user_claude_dir,
    )
    from particles.api.cli.rules import projected_region_filter

    try:
        repo = find_repository_root(project_root) if project_root is not None else None
        if repo is not None:
            files, roots = session_rule_files(repo)
        else:
            files, roots = global_rule_files(), [user_claude_dir()]
        report = await push_rule_files(
            files,
            store=store,
            roots=roots,
            filter_text=projected_region_filter(),
            budget_seconds=get_config().claude_code.hook_deadline_seconds
            * _RULE_PUSH_DEADLINE_SHARE,
        )
    except Exception as exc:  # noqa: BLE001 — the push must never cost the harvest
        log.warning("the rule-file push failed: %s", exc)
        return {"error": redact_known_secrets(str(exc))[:300]}
    out: dict[str, Any] = {k: v for k, v in report.summary().items() if v}
    if report.refused:
        out["refused_files"] = [
            {"path": str(path), "error": redact_known_secrets(error)}
            for path, error in report.refused
        ]
    return out


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
        uri_r=HARNESS_SOURCES["claude-code"].session_prefix + session_id,
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


#: Where the SessionStart hook records each observer's last digest, the start
#: of the next session's automatic-retirement window. Kept in
#: the integration's state directory, beside the per-project render snapshots.
_RETIREMENT_WINDOW_FILE = "digest-retirement-window.json"

#: The share of the hook deadline the retirement notice may spend on a session
#: that injects no digest.
_RETIREMENT_NOTICE_DEADLINE_SHARE = 0.2


def _window_key(store: str, observer: str | None) -> str:
    return f"{store}|{observer or '*'}"


def _read_windows() -> dict[str, str]:
    try:
        data = json.loads((state_dir() / _RETIREMENT_WINDOW_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _previous_render(store: str, observer: str | None) -> datetime | None:
    """When this observer last had a digest, or ``None`` (the configured look-back)."""
    raw = _read_windows().get(_window_key(store, observer))
    try:
        return datetime.fromisoformat(raw) if raw else None
    except ValueError:
        return None


def _record_render(store: str, observer: str | None, at: datetime) -> None:
    """Record a digest render; best-effort, since a lost record only widens the next window."""
    try:
        windows = _read_windows()
        windows[_window_key(store, observer)] = at.isoformat()
        write_private_text(
            state_dir() / _RETIREMENT_WINDOW_FILE,
            json.dumps(windows, indent=2, sort_keys=True) + "\n",
        )
    except Exception:  # noqa: BLE001 — the window record must never cost the session
        log.debug("could not record the digest render", exc_info=True)


async def _retirement_notice(
    store: str, observer: str | None, since: datetime | None
) -> str | None:
    """The retirement section for a session that injects no digest.

    ``""`` when nothing was retired; ``None`` when it could not be read, so the
    window is left where it was and the next session names them.
    """
    from particles.api.client import get_backend

    try:
        budget = get_config().claude_code.hook_deadline_seconds * _RETIREMENT_NOTICE_DEADLINE_SHARE
        return await asyncio.wait_for(
            get_backend().retirement_notice(store, observer, since), budget
        )
    except Exception:  # noqa: BLE001 — best-effort; the session start must survive
        log.debug("retirement notice failed", exc_info=True)
        return None


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
    reached, that refuses the caller outright (503) or that rate-limits it
    (429) still ends the harvest.
    """
    from particles.api.client import get_backend
    from particles.api.client.http import (
        EngineHttpError,
        EngineRefusedError,
        EngineUnreachableError,
    )
    from particles.corpus.origin import OtherHostEntryRefused

    try:
        return await get_backend().deposit_text_at_uri(**kwargs)
    except (EngineRefusedError, EngineUnreachableError):
        raise
    except EngineHttpError as exc:
        if exc.status_code == 429:
            # Rate limited: "not now", about no file in particular, so every
            # later deposit would be refused too. The harvest ends here and the
            # next session end, level-triggered, sends what is left.
            raise
        log.warning("the store refused %s: %s", uri, exc)
        refused.append({"uri": uri, "error": redact_known_secrets(str(exc))[:300]})
        return None
    except OtherHostEntryRefused as exc:
        # The in-process twin of the engine's 409: the store tracks this path
        # for another host, so this file is skipped, not the
        # harvest.
        log.warning("the store refused %s: %s", uri, exc)
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

    The transcripts come from every directory whose sessions belong to the
    ending session's project, the main checkout's and each linked worktree's
    alike, so a worktree session that never ran its SessionEnd is swept by the
    next session end in the project (the match is the adapter's,
    :func:`project_transcript_dirs`). The distilled-content-hash comparison is
    the engine-side unchanged check in ``deposit_text_versioned`` — no side-car
    high-water-mark file to drift. Subagent transcripts (subdirectories) are
    deliberately not walked (§ Deferred).
    """
    limit = get_config().claude_code.harvest.catchup_limit
    if limit <= 0:
        return []
    excluded = exclude.resolve()
    # A symlinked transcript is skipped, as _recorded_cwds skips it, and one the
    # platform prunes mid-walk is passed over rather than failing the hook.
    dated: list[tuple[float, Path]] = []
    for directory in project_transcript_dirs(transcript_dir, project):
        for p in directory.glob("*.jsonl"):
            try:
                if p.is_symlink() or not p.is_file() or p.resolve() == excluded:
                    continue
                dated.append((p.stat().st_mtime, p))
            except OSError:
                continue
    dated.sort(key=lambda item: item[0], reverse=True)
    swept: list[tuple[str, str]] = []
    for _mtime, path in dated[:limit]:
        try:
            outcome = await _harvest_transcript(store, path, path.stem, project)
        except FileNotFoundError:
            continue
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
    harness: str = typer.Option("claude-code", "--harness", help="claude-code or codex."),
    store: str = typer.Option("default", "--store", help="Store handle for Codex diagnostics."),
) -> None:
    """Print recent hook-log entries (one JSONL line per hook invocation)."""
    if harness == "codex":
        from particles.api.cli.codex import codex_doctor

        try:
            _ok, entries = run_armed(codex_doctor(store))
        except (ValueError, OSError) as exc:
            typer.echo(f"Codex doctor: {exc}", err=True)
            raise typer.Exit(1) from None
        for entry in entries:
            typer.echo(entry)
        return
    if harness != "claude-code":
        raise typer.BadParameter("harness must be claude-code or codex")
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
    harness: str = typer.Option("claude-code", "--harness", help="claude-code or codex."),
) -> None:
    """Diagnose whether ``particles hook`` resolves ``store`` from the current directory.

    The lifecycle hooks degrade to exit 0 on any failure, so a mis-resolved
    store fails silently. This verb makes that resolution visible:
    which config.yaml is found, which DSN the handle resolves to, whether the DB
    file exists and carries the corpus tables. Exits non-zero when the store is
    unusable, so it can gate an operator's "is this thing on?" check.
    """
    if harness == "codex":
        from particles.api.cli.codex import codex_doctor

        try:
            ok, lines = run_armed(codex_doctor(store))
            for line in lines:
                typer.echo(line)
        except (ValueError, OSError) as exc:
            typer.echo(f"Codex doctor: {exc}", err=True)
            ok = False
        raise typer.Exit(0 if ok else 1)
    if harness != "claude-code":
        raise typer.BadParameter("harness must be claude-code or codex")
    ok = _run_doctor(store)
    raise typer.Exit(0 if ok else 1)


def _engine_line() -> str:
    """Where the hooks write, and whether a keyless engine would refuse them.

    Reads local state only: ``engine.base_url`` and, when the installed hooks
    pin no key of their own, the key this environment holds. A key the hooks
    pin is judged by :func:`_engine_token_lines` instead. It does
    not probe the engine, so it cannot say whether the engine has a key; it
    says what this install would need.
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
    # W68): when the hooks pin one, this shell's is not judged at all (security
    # review 2026-10-28-1, O65), and theirs is judged on its own engine token:
    # line.
    if _token_pins(thin.HOOKS):
        return (
            f"engine:         {base_url}; the engine token: lines judge the key each consumer pins"
        )
    try:
        own = get_engine_token_optional()
    except EngineTokenFileError as exc:
        return f"engine:         ✗ {base_url}: {exc}"
    if own is None:
        return (
            f"engine:         ⚠ {base_url} with no PARTICLES_ENGINE_TOKEN; a keyless "
            "engine serves this install reads only, so the SessionEnd harvest and the "
            "session exposure are refused"
        )
    token = own
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
    token, pinned = _probe_token()
    if token is None or (pinned and pinned != [token]):
        return (
            "harvest roots: not checked; run the doctor with the PARTICLES_ENGINE_TOKEN the "
            "installed hooks use"
        )
    from particles.api.client.http import EngineHttpError, HttpBackend

    try:
        who = run_armed(HttpBackend(token=token).whoami())
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
    from particles.host_identity import local_host_id

    host = local_host_id()
    if who.get("host_id") != host:
        # The engine stamps every file the key deposits with the key's host id
        # and refuses one tracked for another host.
        recorded = who.get("host_id") or "none"
        return (
            f"harvest roots: ⚠ the hook key records client host {recorded}, but this client's "
            f"host id is {host or 'unusable'}, so the engine refuses its memory and archive "
            f"files; mint the hooks a key with {_harvest_mint_flags()}"
        )
    from particles.api.cli._thin_client import client_rule_roots

    rule_roots, global_rule_roots = client_rule_roots()
    missing += [
        root
        for wanted, recorded in (
            (rule_roots, who.get("rule_roots") or []),
            (global_rule_roots, who.get("global_rule_roots") or []),
        )
        for root in wanted
        if root not in recorded
    ]
    if not missing:
        return (
            "harvest roots: the hook key records this client's host id and its memory, "
            "archive and rule directories"
        )
    return (
        f"harvest roots: ⚠ the hook key records none of {', '.join(missing)}, so the engine "
        "refuses the memory, archive and rule files there; mint the hooks a key with "
        f"{_harvest_mint_flags()}"
    )


def _probe_token() -> tuple[str | None, list[str]]:
    """The key the doctor's engine probes send, and the keys the installed hooks run with.

    This environment's key when it holds one. Otherwise the hooks' own key when
    they run with exactly one, so a thin install's doctor judges the key its
    hooks send rather than reporting none; never when this shell's
    ``PARTICLES_ENGINE_BASE_URL`` names the engine, since the hooks' key goes
    only to the engine their pinned config names (security review R41).
    """
    try:
        token = get_engine_token_optional()
    except EngineTokenFileError:
        token = None
    pinned = _hook_pinned_tokens()
    if token is None and len(pinned) == 1 and environment_engine_url() is None:
        token = pinned[0]
    return token, pinned


def _harvest_mint_flags() -> str:
    """The ``keys create`` flags that record this client's harvest directories."""
    from particles.host_identity import local_host_id

    memory, archive = local_harvest_roots()
    host = local_host_id()
    flags = [f"--host-id {host}"] if host is not None else []
    flags += [f"--memory-root {shlex.quote(root)}" for root in memory]
    flags += [f"--archive-root {shlex.quote(root)}" for root in archive]
    from particles.api.cli._thin_client import client_rule_roots

    rule_roots, global_rule_roots = client_rule_roots()
    flags += [f"--rule-root {shlex.quote(root)}" for root in rule_roots]
    flags += [f"--global-rule-root {shlex.quote(root)}" for root in global_rule_roots]
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


@dataclass(frozen=True)
class _TokenPin:
    """Where one consumer's engine key comes from, as its install pins it.

    ``kind`` is ``file`` (a ``PARTICLES_ENGINE_TOKEN_FILE`` pin, ``where`` the
    token file) or ``plaintext`` (a ``PARTICLES_ENGINE_TOKEN`` value, ``where``
    the file that stores it). ``value`` holds a plaintext key so its role can be
    judged; it is never printed. ``readable`` is false for a token file a
    project-scope settings file names, which the doctor does not open.
    ``command`` marks a pin in a hook command itself, the one the hooks run with.
    ``both_set`` marks a pin that names a token file beside a plaintext key,
    which the token getter refuses.
    """

    consumer: str
    label: str
    kind: Literal["file", "plaintext"]
    where: str
    value: str | None = field(default=None, repr=False)
    readable: bool = True
    command: bool = True
    both_set: bool = False


def _shown_path(path: Path | str) -> str:
    """``path`` with the home directory written as ``~``."""
    text, home = str(path), str(Path.home())
    return "~" + text[len(home) :] if text == home or text.startswith(home + os.sep) else text


def _token_pins(consumer: str) -> list[_TokenPin]:
    """The engine keys ``consumer`` (``thin.HOOKS`` or ``thin.MCP``) is installed with.

    The hooks' come from their settings entries' ``env`` pins and the settings
    file's ``env`` block; the MCP server's from its ``~/.claude.json``
    registration. A key with no per-client file of its own is a plaintext pin.
    """
    found: dict[tuple[str, str, str, str], _TokenPin] = {}

    def add(pin: _TokenPin) -> None:
        found.setdefault((pin.label, pin.kind, pin.where, pin.value or ""), pin)

    if consumer == thin.HOOKS:
        for path in _hook_settings_paths():
            settings = _read_doctor_settings(path)
            if settings is None:
                continue
            shown = _shown_path(path)
            # A token file is read only for the user's own settings: a
            # repository's project-scope file could name any one-line file of
            # this account.
            own = path == Path.home() / ".claude" / "settings.json"
            commands = particles_hook_commands(settings)
            pins_token = False
            for command in commands:
                pins = command_env_pins(command)
                pins_token = pins_token or "PARTICLES_ENGINE_TOKEN" in pins
                if pins.get("PARTICLES_ENGINE_TOKEN"):
                    add(
                        _TokenPin(
                            consumer,
                            "hooks",
                            "plaintext",
                            shown,
                            pins["PARTICLES_ENGINE_TOKEN"],
                            both_set=bool(pins.get("PARTICLES_ENGINE_TOKEN_FILE")),
                        )
                    )
                elif pins.get("PARTICLES_ENGINE_TOKEN_FILE"):
                    add(
                        _TokenPin(
                            consumer,
                            "hooks",
                            "file",
                            pins["PARTICLES_ENGINE_TOKEN_FILE"],
                            readable=own,
                        )
                    )
            env = settings.get("env")
            value = env.get("PARTICLES_ENGINE_TOKEN") if isinstance(env, dict) else None
            if commands and isinstance(value, str) and value.strip():
                # The settings env block reaches every hook command that does not
                # pin the variable itself, so it is a delivery only then; stored
                # there, it is plaintext either way.
                add(
                    _TokenPin(
                        consumer, "hooks", "plaintext", shown, value.strip(), command=not pins_token
                    )
                )
        return list(found.values())
    for reg in particles_mcp_registrations():
        label = f"MCP server {reg.name}"
        shown = _shown_path(Path.home() / ".claude.json")
        if reg.env.get("PARTICLES_ENGINE_TOKEN", "").strip():
            add(
                _TokenPin(
                    consumer,
                    label,
                    "plaintext",
                    shown,
                    reg.env["PARTICLES_ENGINE_TOKEN"].strip(),
                    both_set=bool(reg.env.get("PARTICLES_ENGINE_TOKEN_FILE")),
                )
            )
        elif reg.env.get("PARTICLES_ENGINE_TOKEN_FILE"):
            add(_TokenPin(consumer, label, "file", reg.env["PARTICLES_ENGINE_TOKEN_FILE"]))
    return list(found.values())


def _hook_pinned_tokens() -> list[str]:
    """The distinct engine keys the installed Particles hook commands run with.

    A key pinned in plaintext (``PARTICLES_ENGINE_TOKEN``) or read from the
    token file a thin-client install pins (``PARTICLES_ENGINE_TOKEN_FILE``).
    A file that fails a check contributes nothing here; its
    ``engine token:`` line names the check.
    """
    tokens: dict[str, None] = {}
    for pin in _token_pins(thin.HOOKS):
        if not pin.command or pin.both_set:
            continue
        token = pin.value
        if pin.kind == "file" and pin.readable:
            try:
                token = check_engine_token_file(pin.where)
            except EngineTokenFileError:
                token = None
        if token:
            tokens[token] = None
    return list(tokens)


_ROLE_PHRASES = {"reader": "a reader key", "agent": "an agent key", "operator": "an operator key"}


def _engine_token_lines(base_url: str) -> tuple[bool, list[str]]:
    """One ``engine token:`` line per consumer of the engine key.

    Each names where the key comes from, the token-file checks, the role its
    prefix carries, and, when the engine answers, the principal and role
    ``GET /auth/whoami`` returns against the ones the consumer needs. A key
    stored in plaintext is a warning with its fix, and a hook entry that runs
    the lifecycle verbs through another executable is named. No line ever
    carries a token, a hash of one, or its length. ``False`` when a consumer's
    key cannot be used at all: a token file a check refuses, or a key the
    engine does not accept.
    """
    ok = True
    lines: list[str] = []
    for consumer, none_label in ((thin.HOOKS, "hooks"), (thin.MCP, "MCP server")):
        pins = _token_pins(consumer)
        if not pins:
            lines.append(f"engine token:   {none_label}: none pinned")
            continue
        for pin in pins:
            pin_ok, pin_lines = _judge_token_pin(pin, base_url)
            ok = ok and pin_ok
            lines += pin_lines
    lines += _wrapper_install_lines(base_url)
    return ok, lines


def _judge_token_pin(pin: _TokenPin, base_url: str) -> tuple[bool, list[str]]:
    """The lines for one pin, and whether its consumer can use the key at all."""
    from particles.api._keys import Role, role_from_prefix
    from particles.api.client.http import (
        EngineHttpError,
        EngineUnreachableError,
        HttpBackend,
        PlaintextTokenRefused,
    )

    head = f"engine token:   {pin.label}:"
    if pin.both_set:
        return False, [
            f"{head} ✗ PARTICLES_ENGINE_TOKEN and PARTICLES_ENGINE_TOKEN_FILE are both pinned "
            f"in {pin.where}, which is refused, so it sends nothing; re-run particles init "
            "claude-code to keep the token file only"
        ]
    if pin.kind == "plaintext":
        source = f"plaintext in {pin.where}"
        token = pin.value
        if not pin.command:
            source += " (the settings env block; every hook command pins its own)"
        notes = [
            f"{_PAD}⚠ the engine key is stored in plaintext in {pin.where}; re-run "
            "particles init claude-code to move it to a token file"
        ]
    elif not pin.readable:
        return True, [
            f"{head} ⚠ file {pin.where}, named by a project settings file, so the doctor "
            "does not read it; install the hooks at user scope"
        ]
    else:
        notes = []
        try:
            token = check_engine_token_file(pin.where)
        except EngineTokenFileError as exc:
            return False, [
                f"{head} ✗ file {pin.where} fails the {exc.check} check, so it sends nothing: {exc}"
            ]
        source = f"file {pin.where}, which passes the token-file checks"
    if token is None or not pin.command:
        return True, [f"{head} {source}", *notes]
    refusal = thin.key_refusal(pin.consumer, token)
    role = role_from_prefix(token)
    if refusal is not None or role is None:
        # key_refusal refuses every key with no per-client prefix.
        return True, [f"{head} ⚠ {source}; {refusal}", *notes]
    line = f"{head} {source}; {_ROLE_PHRASES[role.value]}"
    if role is Role.READER:
        line += " (the MCP write tools answer 403)"
    expected = (
        thin.HOOK_PRINCIPAL
        if pin.consumer == thin.HOOKS
        else get_config().mcp.write.asserter_identity
    )
    if environment_engine_url() is not None:
        # The key goes only to the engine the pinned config names, never one
        # this shell's PARTICLES_ENGINE_BASE_URL names (security review R41).
        return True, [
            f"{line}; its principal is not checked, since PARTICLES_ENGINE_BASE_URL names "
            "the engine here rather than the pinned config",
            *notes,
        ]
    try:
        who = run_armed(HttpBackend(token=token, base_url=base_url).whoami())
    except EngineUnreachableError:
        return True, [
            f"{line}; the engine was not reached, so its principal is not checked",
            *notes,
        ]
    except PlaintextTokenRefused as exc:
        return False, [f"{head} ✗ {source}; {redact_known_secrets(str(exc))[:300]}", *notes]
    except EngineHttpError as exc:
        return False, [
            f"{head} ✗ {source}; the engine does not accept the key: "
            f"{redact_known_secrets(str(exc))[:300]}",
            *notes,
        ]
    except Exception as exc:  # noqa: BLE001 — the doctor reports, never crashes
        return True, [
            f"{line}; its principal is not checked ({type(exc).__name__}: "
            f"{redact_known_secrets(str(exc))[:200]})",
            *notes,
        ]
    principal = str(who.get("principal") or "anonymous")
    engine_role = str(who.get("role") or "none")
    if principal != expected or engine_role != role.value:
        return True, [
            f"{head} ⚠ {source}; the engine says principal {principal}, role {engine_role}, "
            f"where this consumer needs principal {expected}; mint it its own key",
            *notes,
        ]
    return True, [f"{line}; the engine says principal {principal}, role {engine_role}", *notes]


def _wrapper_install_lines(base_url: str) -> list[str]:
    """A hook entry that runs the lifecycle verbs through a non-``particles`` executable.

    That is the retired wrapper install, which the installer does not
    recognise and will not add its own entries beside. Only the
    executable is named, since the rest of the command may carry a key.
    """
    lines: list[str] = []
    for path in _hook_settings_paths():
        settings = _read_doctor_settings(path)
        if settings is None:
            continue
        for command in thin.wrapper_hook_entries(settings):
            try:
                words = shlex.split(command)
            except ValueError:
                # wrapper_hook_entries keeps only commands that split, but the
                # doctor reports and never crashes.
                words = []
            start = 0
            if words and words[0] == "env":
                start = 1
                while start < len(words) and re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*=.*", words[start]
                ):
                    start += 1
            exe = words[start] if start < len(words) else "?"
            lines.append(
                f"engine token:   ⚠ {_shown_path(path)} runs the lifecycle hooks through "
                f"{shlex.quote(exe)}, an unrecognised install; the wrapper install is retired: "
                "delete those entries, then run particles init claude-code --engine "
                f"{shlex.quote(url_without_userinfo(base_url))}, which moves its key to a "
                "token file"
            )
            break
    return lines


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

    for line in _check_session_end_hand_off():
        typer.echo(f"  {line}")

    for line in _check_config_pins():
        typer.echo(f"  {line}")

    for line in _check_install():
        typer.echo(f"  {line}")

    from particles.operations.extract_budget import extract_budget_warnings

    # A thin client's consolidation runs on the engine, whose own warnings
    # follow the tuning: line; this file's would judge the wrong config.
    if not get_config().engine.base_url:
        for warning in extract_budget_warnings():
            typer.echo(f"  extract budget: ⚠ {warning}")

    typer.echo(f"  {_engine_line()}")
    base_url = get_config().engine.base_url
    tokens_ok = True
    if base_url:
        tokens_ok, token_lines = _engine_token_lines(base_url)
        for line in token_lines:
            typer.echo(f"  {line}")
    roots_line = _harvest_roots_line()
    if roots_line is not None:
        typer.echo(f"  {roots_line}")

    if base_url:
        return _run_remote_doctor(store, tokens_ok=tokens_ok)

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


# ---------------------------------------------------------------------------
# hook doctor against a remote engine
# ---------------------------------------------------------------------------

_PAD = " " * 16


def _run_remote_doctor(store: str, *, tokens_ok: bool = True) -> bool:
    """The doctor's checks when ``engine.base_url`` is set: the engine, not a local file.

    The hooks write the engine, so a local database file is beside the point:
    absent on a thin client (which is not a failure) and stale when present
    (which is no success). The engine probes decide the exit code; the lines
    that judge a local store say they do not apply.
    """
    ok, lines = _check_remote_engine(store)
    # A consumer whose own key is unusable fails whatever this shell's token
    # proves; its engine token: line above says why.
    ok = ok and tokens_ok
    # The engine named only by PARTICLES_ENGINE_BASE_URL does not lift the
    # relative-store rule, so the hooks may refuse before reaching the engine.
    refusal = _working_directory_store_refusal(store)
    if refusal is not None:
        ok = False
        lines.append(f"hook refusal:   ✗ every hook refuses and does nothing: {refusal}")
    lines += [
        "schema:         not checked from this client; the engine migrates its own store, and "
        "the version: line compares the two installs",
        "local store:    not checked; the hooks write the engine, so the store DSN, database "
        "file, corpus tables, blobs, refresh roots and calibration checks do not apply here",
    ]
    lines += _check_remote_hook_steps(store)
    lines += _check_session_end_hand_off()
    lines += _check_memory_dirs(store)
    lines += _check_remote_observer_scope()
    lines += _check_egress()
    for line in lines:
        typer.echo(f"  {line}")
    if not ok:
        typer.echo(f"  ✗ The hooks cannot use store '{store}' on the engine; see the ✗ line above.")
        return False
    typer.echo(f"  ✓ The engine answers and serves store '{store}' to this client's token.")
    return True


def _check_remote_engine(store: str) -> tuple[bool, list[str]]:
    """Probe the engine the hooks write: reachability, version, an authenticated read.

    ``GET /health`` is timed against ``claude_code.hook_deadline_seconds``, the
    budget inside which every hook gives up, and its version is compared with
    this install's, since the client and the engine ship in lockstep.
    ``GET /digest/{store}`` is the SessionStart read and always
    takes the bearer, so its status tells a missing or refused token apart
    from a store the engine does not configure. Every request goes through the
    engine client's checked transport. ``False`` when the hooks
    cannot reach the engine or read the store.
    """
    import particles
    from particles.api.client.http import HttpBackend as _HttpBackend

    cfg = get_config()
    deadline = cfg.claude_code.hook_deadline_seconds
    backend = _HttpBackend()
    started = time.monotonic()
    try:
        health = run_armed(backend.probe_health())
    except Exception as exc:  # noqa: BLE001 — the doctor reports, never crashes
        return False, [f"reachability:   ✗ {redact_known_secrets(str(exc))}"]
    elapsed = time.monotonic() - started
    if health.status_code >= 400:
        return False, [
            f"reachability:   ✗ GET /health answered {health.status_code}: "
            f"{_response_detail(health)}"
        ]
    timing = f"GET /health answered in {elapsed:.2f} s"
    if elapsed >= deadline:
        lines = [
            f"reachability:   ⚠ {timing}, over the hooks' {deadline:g} s deadline "
            "(claude_code.hook_deadline_seconds), so a hook gives up before the engine answers"
        ]
    else:
        lines = [f"reachability:   {timing} (the hooks' deadline is {deadline:g} s)"]
    body = _response_json(health)
    status = str(body.get("status") or "")
    if status and status != "ok":
        lines.append(f"{_PAD}⚠ the engine reports status {status!r}: a background task failed")
    engine_version = str(body.get("version") or "")
    client_version = particles.__version__
    if not engine_version:
        lines.append(
            f"version:        ⚠ the engine's /health names no version; this client {client_version}"
        )
    elif engine_version != client_version:
        lines.append(
            f"version:        ⚠ engine {engine_version}, this client {client_version}. The "
            "client and the engine ship in lockstep, so upgrade the older side"
        )
    else:
        lines.append(f"version:        engine and client both {client_version}")
    digest_ok, digest_lines = _check_remote_digest(backend, store)
    lines += digest_lines
    if digest_ok:
        lines += _check_remote_writes(backend, store)
        lines += _check_remote_tuning(backend)
    return digest_ok, lines


#: Said when the engine's settings could not be read: this client's own
#: extraction-budget warnings would judge a config consolidation does not run with.
_BUDGET_UNCHECKED = (
    "extract budget: not checked; consolidation runs on the engine, whose settings "
    "could not be read"
)


def _check_remote_tuning(backend: HttpBackend) -> list[str]:
    """Whether this client's tuning matches the engine's, and the engine's budget warnings.

    The engine applies most of what an operator tunes (extraction,
    consolidation, query, the model routes), so a setting changed in this
    client's file only has no effect. ``GET /config/tuning`` serves
    a digest of each matched setting of the engine's effective config; the
    same digests of this client's config name each one that differs. A
    difference is a warning: the hooks still work, and the exit code stays.
    The extraction-budget warnings come from the engine too, since
    consolidation runs there.
    """
    probe = _probe_key(backend)
    try:
        resp = run_armed(probe.backend.probe_tuning())
    except Exception as exc:  # noqa: BLE001 — the doctor reports, never crashes
        return [
            f"tuning:         ⚠ not compared; GET /config/tuning failed "
            f"({type(exc).__name__}: {redact_known_secrets(str(exc))[:200]})",
            _BUDGET_UNCHECKED,
        ]
    if resp.status_code == 404:
        return [
            "tuning:         not compared; the engine predates GET /config/tuning, so its "
            "settings cannot be read from here",
            _BUDGET_UNCHECKED,
        ]
    if resp.status_code >= 400:
        return [
            f"tuning:         ⚠ not compared; GET /config/tuning answered {resp.status_code}: "
            f"{_response_detail(resp)}",
            _BUDGET_UNCHECKED,
        ]
    try:
        report = TuningReport.model_validate(resp.json())
    except ValueError as exc:
        return [
            f"tuning:         ⚠ not compared; the engine's answer did not parse ({exc})"[:300],
            _BUDGET_UNCHECKED,
        ]

    cfg = get_config()
    comparison = compare_tuning(cfg, report.digests, cfg.engine.tuning_drift_ignore)
    compared = comparison.matching + len(comparison.differing) + len(comparison.declared)
    notes: list[str] = []
    if comparison.declared:
        notes.append(f"{len(comparison.declared)} declared in engine.tuning_drift_ignore")
    if comparison.one_sided:
        notes.append(
            f"{comparison.one_sided} exist on one side only and are not compared "
            "(the two releases differ; see the version: line)"
        )
    tail = f"; {'; '.join(notes)}" if notes else ""
    if not comparison.differing:
        lines = [f"tuning:         {compared} settings match the engine's{tail}"]
    else:
        lines = [
            f"tuning:         ⚠ {len(comparison.differing)} of {compared} settings differ from "
            f"the engine's, which runs consolidation, extraction and queries with its own{tail}"
        ]
        here_env = env_overridden_tuning()
        engine_env = set(report.env_overrides)
        for path, value in comparison.differing.items():
            source = f" (from ${here_env[path]})" if path in here_env else ""
            there = (
                "its environment"
                if path in engine_env
                else "its config file or left at its compiled default"
            )
            lines.append(
                f"{_PAD}⚠ drift: {path} is {preview(value)} here{source}; the engine runs "
                f"another value, set in {there}"
            )
        lines.append(
            f"{_PAD}set each in the engine's config and restart the engine, or list a "
            "deliberate difference in engine.tuning_drift_ignore"
        )
    lines += [f"extract budget: ⚠ (the engine's config) {w}" for w in report.warnings]
    return lines


def _check_remote_writes(backend: HttpBackend, store: str) -> list[str]:
    """Whether the engine accepts the hooks' belief writes on ``store``.

    An engine that does not list the store in ``mcp.write.enabled_stores``
    answers 403 to the SessionStart exposure write, the MCP write tools and the
    curation writes, while the digest read passes. ``GET /auth/whoami`` names
    the stores that accept the key's belief writes, so the doctor reads the
    switch without making a write. Only a warning: the read path and the
    harvest still work. The key is the one the digest probe sends.
    """
    probe = _probe_key(backend)
    whose = probe.whose
    try:
        who = run_armed(probe.backend.whoami())
    except Exception as exc:  # noqa: BLE001 — the doctor reports, never crashes
        return [
            f"writes:         ⚠ not checked; GET /auth/whoami failed "
            f"({type(exc).__name__}: {redact_known_secrets(str(exc))[:200]})"
        ]
    stores = who.get("belief_write_stores")
    if not isinstance(stores, list):
        return [
            "writes:         not checked; the engine's /auth/whoami does not name the stores "
            "that accept belief writes (an engine older than this client)"
        ]
    if store in stores:
        return [f"writes:         the engine accepts belief writes on store '{store}' from {whose}"]
    role = str(who.get("role") or "")
    if who.get("anonymous") or role == "reader":
        cause = (
            "the engine has no key for the caller"
            if who.get("anonymous")
            else "the key is a reader key"
        )
        return [
            f"writes:         ⚠ the engine refuses belief writes from {whose}: {cause}, so the "
            "SessionStart exposure write and the MCP write tools answer 403"
        ]
    return [
        f"writes:         ⚠ the engine refuses belief writes on store '{store}': it is not in "
        "the engine's mcp.write.enabled_stores, so the SessionStart exposure write, the MCP "
        "write tools and the curation writes answer 403; add it there in the engine's config "
        "and restart the engine"
    ]


@dataclass(frozen=True)
class _ProbeKey:
    """The key the doctor's engine probes send, and how its lines name it."""

    backend: HttpBackend
    token: str | None
    pinned: list[str]
    hooks_key: bool

    @property
    def whose(self) -> str:
        if self.hooks_key:
            return "the installed hooks' key"
        return "this environment's token" if self.token else "no token (the dev-key bounds)"


def _probe_key(backend: HttpBackend) -> _ProbeKey:
    """``backend``, or one carrying the hooks' own key when :func:`_probe_token` chose it."""
    from particles.api.client.http import HttpBackend as _HttpBackend

    token, pinned = _probe_token()
    hooks_key = token is not None and pinned == [token]
    return _ProbeKey(
        backend=_HttpBackend(token=token) if hooks_key else backend,
        token=token,
        pinned=pinned,
        hooks_key=hooks_key,
    )


def _check_remote_digest(backend: HttpBackend, store: str) -> tuple[bool, list[str]]:
    """The authenticated ``GET /digest/{store}`` probe, as the SessionStart hook reads it."""
    probe = _probe_key(backend)
    token, pinned = probe.token, probe.pinned
    try:
        resp = run_armed(probe.backend.probe_digest(store))
    except Exception as exc:  # noqa: BLE001 — the doctor reports, never crashes
        return False, [f"digest:         ✗ {redact_known_secrets(str(exc))}"]
    code = resp.status_code
    detail = _response_detail(resp)
    if code < 400:
        size = len(str(_response_json(resp).get("markdown") or "").encode("utf-8"))
        lines = [f"digest:         GET /digest/{store} served {size} bytes to {probe.whose}"]
        ok = True
    elif code in (401, 403):
        if token is None:
            cause = (
                "this environment holds no PARTICLES_ENGINE_TOKEN; set the one the installed "
                "hooks carry and re-run"
            )
        elif code == 401:
            cause = "the engine does not accept the PARTICLES_ENGINE_TOKEN here (wrong or revoked)"
        else:
            cause = (
                "the engine refused the request; its detail below says why (the key's "
                "role, or a Host name the engine does not serve)"
            )
        lines = [f"digest:         ✗ GET /digest/{store} refused ({code}): {cause}"]
        lines.append(f"{_PAD}engine said: {detail}")
        ok = False
    elif code == 404:
        lines = [
            f"digest:         ✗ GET /digest/{store} answered 404: the engine does not "
            f"configure store '{store}'; install the hooks with a handle it serves"
        ]
        ok = False
    elif code == 503:
        lines = [
            f"digest:         ✗ GET /digest/{store} answered 503: the engine refuses this "
            f"client outright ({detail})"
        ]
        ok = False
    else:
        lines = [f"digest:         ✗ GET /digest/{store} answered {code}: {detail}"]
        ok = False
    if pinned and pinned != [token]:
        lines.append(
            f"{_PAD}⚠ the probe used this environment's token, not the one the installed "
            "hooks pin; run the doctor with the hooks' environment to judge theirs"
        )
    return ok, lines


def _response_json(resp: httpx.Response) -> dict[str, Any]:
    """A response's JSON object, or an empty one."""
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _response_detail(resp: httpx.Response) -> str:
    """The engine's ``detail`` for a refusal, redacted and capped."""
    detail = _response_json(resp).get("detail")
    text = str(detail) if detail is not None else (resp.text.strip() or resp.reason_phrase)
    return redact_known_secrets(text)[:300]


def _check_remote_hook_steps(store: str) -> list[str]:
    """What the SessionEnd harvest, the Stop hook and the projection do against the engine.

    The harvest ships off-machine only on the operator's opt-in
    (``claude_code.harvest.allow_remote``), and the SessionEnd
    hook returns before the MEMORY.md render while it is off. With it on, the
    engine renders the region and this client splices it.
    Utility mining and absence recording run on a local store only.
    """
    harvest = get_config().claude_code.harvest
    if harvest.allow_remote:
        what = "transcripts and memory files" if harvest.transcripts else "memory files only"
        lines = [
            f"harvest:        SessionEnd deposits {what} on the engine "
            "(claude_code.harvest.allow_remote: true)",
            f"{_PAD}utility mining and memory-absence recording need a local store and are skipped",
        ]
    else:
        lines = [
            "harvest:        ⚠ off: claude_code.harvest.allow_remote is false, so every "
            "SessionEnd skips the harvest (logged as remote-harvest-disabled); set it true "
            "to deposit transcripts and memory files on the engine"
        ]
    stop = _check_retired_stop_hook()
    if stop:
        lines += stop
    else:
        lines.append("stop hook:      none installed (it is retired and did nothing remotely)")
    if not projection_enabled():
        lines.append(
            "projection:     off (agent_memory.projection.enabled is false, or "
            "`particles init claude-code` wrote no manifest)"
        )
    elif not harvest.allow_remote:
        lines.append(
            "projection:     ⚠ MEMORY.md is never re-rendered: the SessionEnd hook stops before "
            "the render while the harvest is off"
        )
        lines += _check_remote_block_texts(allow_remote=False)
    else:
        lines.append(
            f"projection:     the engine renders the MEMORY.md region "
            f"(POST /memory/projection/{store}) and this client splices it"
        )
        lines += _check_remote_block_texts(allow_remote=True)
    return lines


def _check_remote_block_texts(*, allow_remote: bool) -> list[str]:
    """What the manifest's file sections send with each remote render.

    Reads only this client's manifest and the files its blocks name, under the
    render's own containment rules, and sends nothing. Says nothing for a
    manifest with no file section. A refusal names the reason, never a block's
    path or text (security review 2026-10-24-2, Q51).
    """
    from particles.api.cli._claude_code import memory_manifest_path
    from particles.corpus.local_read import LocalReadRefused
    from particles.operations.projection import load_manifest
    from particles.operations.projection.manifest import manifest_error_text
    from particles.operations.projection.memory_region import (
        MAX_REMOTE_BLOCK_BYTES,
        ClientFileSectionError,
        block_texts_size,
        has_client_file_section,
        read_block_texts,
    )
    from particles.operations.projection.paths import ManifestPathError

    path = memory_manifest_path()
    if not path.is_file():
        return []
    try:
        manifest = load_manifest(path)
    except Exception as exc:  # noqa: BLE001 — a diagnostic line, never a failure
        return [f"{_PAD}⚠ the memory manifest does not load: {manifest_error_text(exc)}"]
    if not has_client_file_section(manifest):
        return []
    if not allow_remote:
        return [
            f"{_PAD}file sections: the manifest has some, and their text is sent only with "
            "claude_code.harvest.allow_remote on, so each SessionStart pushes the full digest"
        ]
    try:
        texts = read_block_texts(manifest, path.parent)
    except FileNotFoundError:
        reason = "a file section names a file that does not exist"
    except ManifestPathError as exc:
        reason = str(exc)
    except LocalReadRefused:
        reason = "a file section's file was refused by the contained read"
    except ClientFileSectionError as exc:
        reason = str(exc)
    except (OSError, UnicodeDecodeError) as exc:
        reason = f"a file section's file could not be read ({type(exc).__name__})"
    else:
        return [
            f"{_PAD}file sections: {len(texts)} file(s), {block_texts_size(texts)} bytes sent "
            f"with each render (limit {MAX_REMOTE_BLOCK_BYTES})"
        ]
    return [f"{_PAD}⚠ file sections: the render is refused until this is fixed: {reason}"]


def _check_remote_observer_scope() -> list[str]:
    """The observer scope a session here asks the engine for.

    The mode and the MCP binding are this client's; whether the engine's store
    has been rescoped is the engine's, and is not asked.
    """
    mode = get_config().claude_code.observer_scope
    if mode != "project":
        return [f"observer scope: {mode} (every session sees the whole store)"]
    return [
        "observer scope: project (a session reads through its project; whether the engine's "
        "store is rescoped is not checked from here)",
        *_observer_binding_lines(Path.cwd()),
    ]


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


#: How many recent hook-log records the hand-off and degraded summaries read.
_DOCTOR_LOG_WINDOW = 2000

#: The share of SessionStarts that may be degraded before the doctor warns
#: (the owner's M10 rule, 2026-11-07).
_DEGRADED_SHARE_WARN = 0.05

#: Below this one-minute load average a degraded SessionStart is a fault, not load.
_DEGRADED_LOAD_FLOOR = 4.0


def _check_session_end_hand_off() -> list[str]:
    """The lines: the hand-off installed, the spool, the worker, degraded starts.

    Reported, never fatal. A SessionEnd that still harvests inside the hook is
    told to re-run the installer; a session left in the spool past the
    worker's bound, or taken by a worker that is gone, is named; the newest
    worker record and the recent share of degraded SessionStarts are shown.
    """
    from particles.api.cli import _session_end_worker as worker
    from particles.api.cli._claude_code import (
        session_end_hands_off,
        session_end_spool_dir,
    )
    from particles.api.cli._claude_launcher import LOCK_NAME
    from particles.api.cli._hook_state import advisory_lock

    lines: list[str] = []
    settings = _read_doctor_settings(Path.home() / ".claude" / "settings.json")
    ends = (
        [c for c in particles_hook_commands(settings) if "hook session-end" in c]
        if settings is not None
        else []
    )
    spool = session_end_spool_dir()
    if ends and not any(session_end_hands_off(c) for c in ends):
        lines.append("hand-off:       ⚠ the SessionEnd hook harvests inside the hook.")
        lines.append(
            "                Re-run `particles init claude-code` to install the detached worker."
        )
    for command in (c for c in ends if session_end_hands_off(c)):
        pinned = _command_option(command, "--spool")
        if pinned is not None and Path(pinned) != spool.absolute():
            lines.append(
                f"hand-off:       ⚠ the SessionEnd hook queues sessions in {pinned}, but the "
                f"worker reads {spool}; re-run `particles init claude-code`."
            )
    bound = get_config().claude_code.harvest.worker_deadline_seconds
    now = time.time()
    stale = [p for p in worker.pending(spool) if now - _mtime(p, now) > bound]
    if stale:
        lines.append(
            f"hand-off:       ⚠ {len(stale)} ended session(s) waited over {bound:g} s for "
            "the worker; run `particles hook session-end-worker --store <handle>`."
        )
    if worker.taken(spool) and spool.is_dir():
        with advisory_lock(spool / LOCK_NAME) as held:
            if held:
                lines.append(
                    "hand-off:       ⚠ a worker exited mid-harvest; the next one logs it as "
                    "`worker lost` and harvests again."
                )
    records = []
    for raw in read_hook_log_tail(_DOCTOR_LOG_WINDOW):
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if isinstance(entry, dict):
            records.append(entry)
    workers = [r for r in records if r.get("event") == "session-end" and r.get("mode")]
    if workers:
        last = workers[-1]
        detail = f": {last.get('error')}" if last.get("outcome") != "ok" else ""
        lines.append(f"worker:         last run {last.get('outcome')} at {last.get('ts')}{detail}")
    starts = [r for r in records if r.get("event") == "session-start"][-200:]
    degraded = [r for r in starts if r.get("outcome") == "degraded"]
    if degraded:
        share = len(degraded) / len(starts)
        light = [r for r in degraded if float(r.get("load_1m") or 0.0) < _DEGRADED_LOAD_FLOOR]
        mark = "⚠" if share > _DEGRADED_SHARE_WARN or light else "✓"
        lines.append(
            f"session start:  {mark} {len(degraded)} of the last {len(starts)} kept the loaded "
            f"region (degraded); {len(light)} at load under {_DEGRADED_LOAD_FLOOR:g}"
        )
    return lines


def _mtime(path: Path, default: float) -> float:
    """``path``'s modification time, or ``default`` when a worker took it meanwhile."""
    try:
        return path.stat().st_mtime
    except OSError:
        return default


def _command_option(command: str, option: str) -> str | None:
    """The value after ``option`` in an installed hook command, if any."""
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    for i, word in enumerate(words[:-1]):
        if word == option:
            return words[i + 1]
    return None


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
    from particles.corpus.origin import OriginCounts, count_file_entries_by_origin
    from particles.db import session_scope

    async def _count() -> tuple[RefusalCounts, OriginCounts]:
        async with session_scope(store) as session:
            return (
                await count_refused_local_entries(session),
                await count_file_entries_by_origin(session),
            )

    try:
        refused, by_origin = run_armed(_count())
    except Exception as exc:  # noqa: BLE001 — a diagnostic must not fail the doctor
        return [f"refresh roots:  could not check the store: {exc}"]
    # One line per cause, each with its own fix (an outside-root entry needs a
    # root recorded; an untrusted depositor's does not change on re-enrolment).
    lines = [
        f"{'refresh roots:  ⚠ ' if i == 0 else '                ⚠ '}{line}"
        for i, line in enumerate(refused.lines())
    ]
    # Whose files the store tracks, by origin: a store stamped
    # on the wrong host shows here before the first push.
    if by_origin.this_host or by_origin.other_hosts or by_origin.unstamped:
        lines.append(f"file origins:   {by_origin.line()}")
    return lines


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
