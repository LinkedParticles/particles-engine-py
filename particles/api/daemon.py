"""Resident daemon mode — the in-process background tasks of ``engine serve``.

Running Particles as a service used to take four hand-assembled parts: the
engine, a loop over ``extract --all-pending``, ``inbox watch``, and a
launchd/cron entry for ``memory consolidate --if-due``. In a container none of
that is available — there is no launchd, no cron, and the consolidation lock's
``os.kill(pid, 0)`` stale-reclaim is meaningless across pid namespaces. So the
engine grows an **opt-in** residency: ``engine serve --daemon`` (or
``daemon.enabled``) starts background asyncio tasks in the existing FastAPI
lifespan. Off by default; without it ``engine serve`` behaves exactly as before.

Two design rules the module exists to hold:

* **One scheduler, one writer.** The consolidation tick is the *only* periodic
  writer. Pending extraction rides consolidation pass 1, as it does under cron —
  there is deliberately **no** separate extract-drain loop, because a second
  drain path would reintroduce the multi-writer shape daemon mode exists to
  remove.
* **A crashed task is never silently dead.** Every task runs on one shared poll
  seam (:func:`_poll_loop`) that catches, logs, and *records* a failing
  iteration, then keeps ticking; a task that dies outright is marked ``crashed``
  and disclosed through :func:`daemon_status` — which ``GET /health`` renders.
* **Intake watchers, designed once** (the fold). The inbox watcher and
  the web-clipper watcher are two ``Work`` callables on that same seam, not two
  bespoke loops — and both are mtime-polls, so rejection of
  filesystem-event watchers stands untouched.

The tick is a *scheduler*, not a worker pool: each iteration awaits the ordinary
async operation, so nothing here wedges the event loop.

Layering: this module is a Surface (it is imported by ``particles.api.app``'s
lifespan and by ``engine serve``). It keeps its Engine imports deferred inside
the task bodies so importing it stays cheap, and it never imports the CLI — the
pass-6 projection tail is *injected* by ``engine serve`` through
:func:`set_projection_runner_factory`, mirroring how the consolidation design
 has the Surface
inject that callback into the Engine operation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from particles.config import get_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from particles.operations.consolidation import ProjectionRunner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The pass-6 projection seam (reused)
# ---------------------------------------------------------------------------

#: Builds the consolidation pass-6 callback for a store handle, returning
#: ``(runner, skip_reason)`` exactly like the CLI's own builder. Registered by
#: ``engine serve`` so this module never imports ``particles.api.cli``.
ProjectionRunnerFactory = Callable[[str], "tuple[ProjectionRunner | None, str | None]"]

_projection_runner_factory: ProjectionRunnerFactory | None = None


def set_projection_runner_factory(factory: ProjectionRunnerFactory | None) -> None:
    """Register (or clear) the pass-6 projection-runner factory.

    ``engine serve --daemon`` registers the CLI's builder before uvicorn starts,
    so a host daemon renders ``MEMORY.md`` exactly as the launchd recipe does.
    With nothing registered — a raw ``uvicorn particles.api.app:app`` launch, or
    the container image, whose baked config disables the projection anyway —
    pass 6 records a disclosed skip rather than running.
    """
    global _projection_runner_factory
    _projection_runner_factory = factory


def _projection_runner(store: str) -> tuple[ProjectionRunner | None, str | None]:
    if _projection_runner_factory is None:
        return None, "no projection runner registered (daemon mode)"
    return _projection_runner_factory(store)


# ---------------------------------------------------------------------------
# Status — what ``GET /health`` discloses
# ---------------------------------------------------------------------------

TaskState = Literal["idle", "running", "crashed", "stopped"]


class DaemonTaskStatus(BaseModel):
    """Live state of one daemon task (Consequences: never silently dead)."""

    name: str
    #: Nominal seconds between iterations, as resolved when the task started.
    interval_seconds: float
    state: TaskState = "idle"
    #: Completed iterations, successful or not.
    runs: int = 0
    #: Iterations that raised. A non-zero count with ``state != "crashed"`` means
    #: the task is still ticking but something inside it is failing.
    failures: int = 0
    last_started_at: datetime | None = None
    last_finished_at: datetime | None = None
    #: Short human summary of the last iteration (e.g. ``"skipped: not due…"``).
    last_outcome: str | None = None
    #: ``"ExcType: message"`` of the most recent failure; cleared on success.
    last_error: str | None = None


class DaemonStatus(BaseModel):
    """Aggregate daemon state embedded in the ``/health`` payload."""

    enabled: bool = False
    #: False as soon as any task has died outright — the signal an operator (or
    #: a Kubernetes readiness gate) acts on.
    healthy: bool = True
    tasks: list[DaemonTaskStatus] = Field(default_factory=list)


class _Runtime:
    """Module-level daemon state. One per process; the engine runs one daemon."""

    def __init__(self) -> None:
        self.enabled = False
        self.statuses: dict[str, DaemonTaskStatus] = {}
        self.tasks: list[asyncio.Task[None]] = []

    def reset(self) -> None:
        self.enabled = False
        self.statuses = {}
        self.tasks = []


_runtime = _Runtime()


def daemon_status() -> DaemonStatus:
    """Snapshot the daemon's state for ``GET /health``.

    Returns ``enabled=False`` with no tasks when daemon mode is off, which is
    what a plain ``engine serve`` reports.
    """
    tasks = [status.model_copy() for status in _runtime.statuses.values()]
    healthy = all(status.state != "crashed" for status in tasks)
    return DaemonStatus(enabled=_runtime.enabled, healthy=healthy, tasks=tasks)


# ---------------------------------------------------------------------------
# The shared poll seam — one loop shape every daemon task runs on
# ---------------------------------------------------------------------------

#: An iteration's work: returns a short outcome string for the status block.
Work = Callable[[], Awaitable[str]]
#: Resolves the sleep between iterations, called each time so a config reload
#: takes effect without a restart (read config *inside* the function).
IntervalFn = Callable[[], float]


def _utcnow() -> datetime:
    return datetime.now(UTC)


async def _poll_loop(status: DaemonTaskStatus, interval: IntervalFn, work: Work) -> None:
    """Run ``work`` forever, one iteration per ``interval`` seconds.

    The seam every daemon task shares. A failing iteration is caught, logged
    with its traceback, and recorded on ``status`` — the loop then keeps its
    cadence rather than dying, because every task here is level-triggered
    (the corpus / inbox / captures dir is the state, so the next tick simply
    sees the same work again). Cancellation propagates so lifespan shutdown is
    prompt.

    The first iteration runs immediately: a container restarted after a day of
    downtime should catch up now, and the consolidation tick's ``--if-due``
    guard makes an unnecessary immediate run a no-op.
    """
    while True:
        status.state = "running"
        status.last_started_at = _utcnow()
        try:
            outcome = await work()
        except asyncio.CancelledError:
            status.state = "stopped"
            raise
        except Exception as exc:  # noqa: BLE001 — a failing tick must not kill the task
            status.failures += 1
            status.last_error = f"{type(exc).__name__}: {exc}"
            status.last_outcome = "failed"
            log.exception("daemon task %r failed; continuing on its schedule", status.name)
        else:
            status.last_outcome = outcome
            status.last_error = None
        status.runs += 1
        status.last_finished_at = _utcnow()
        status.state = "idle"
        try:
            await asyncio.sleep(interval())
        except asyncio.CancelledError:
            status.state = "stopped"
            raise


def _on_task_done(status: DaemonTaskStatus, task: asyncio.Task[None]) -> None:
    """Mark a task that ended outside cancellation as ``crashed`` (never silently dead)."""
    if task.cancelled():
        status.state = "stopped"
        return
    exc = task.exception()
    if exc is None:
        # A poll loop never returns normally; if one does, treat it as a crash
        # so /health does not keep reporting a task that is no longer running.
        status.state = "crashed"
        status.last_error = status.last_error or "task exited unexpectedly"
        log.error("daemon task %r exited unexpectedly", status.name)
        return
    status.state = "crashed"
    status.last_error = f"{type(exc).__name__}: {exc}"
    log.error("daemon task %r crashed: %s", status.name, status.last_error, exc_info=exc)


# ---------------------------------------------------------------------------
# The tasks
# ---------------------------------------------------------------------------


async def consolidation_tick() -> str:
    """One consolidation tick — the operation, with ``--if-due`` semantics.

    The hard part already shipped: ``consolidation.min_interval_hours``
    is the real cadence, so ticking hourly against a 20-hour interval is harmless.
    Pending extraction rides pass 1; there is no separate drain loop.
    """
    from particles.db import session_scope
    from particles.operations.consolidation import run_consolidation

    store = get_config().daemon.store
    runner, skip_reason = _projection_runner(store)
    async with session_scope(store) as session:
        report = await run_consolidation(
            session,
            store=store,
            if_due=True,
            actor="memory-consolidate",
            projection_runner=runner,
            projection_skip_reason=skip_reason,
        )
    if report.outcome == "skipped":
        reason = report.skip_reason or "skipped"
        log.info("daemon consolidation tick: %s", reason)
        return f"skipped: {reason}"
    failed = report.failed_passes()
    if failed:
        log.warning("daemon consolidation tick: pass(es) failed: %s", ", ".join(failed))
        return f"completed, failed pass(es): {', '.join(failed)}"
    log.info("daemon consolidation tick: completed")
    return "completed"


# ---------------------------------------------------------------------------
# Intake watchers, designed once (fold)
#
# Both are mtime-polls on the same seam as the tick. No watchdog / FSEvents
# dependency: rejection of filesystem-event watchers stands
# untouched, as does its decision that mutable-source refresh stays a
# consolidation pass. Each watcher stats first and only pays for the expensive
# half (read, hash, deposit) when something actually moved.
# ---------------------------------------------------------------------------


def _mtime_or_none(path: Path) -> float | None:
    try:
        return path.stat().st_mtime if path.exists() else None
    except OSError as exc:
        log.warning("daemon: stat failed for %s: %s", path, exc)
        return None


def _tree_mtime(root: Path) -> float | None:
    """Newest mtime anywhere under ``root``, or ``None`` when it does not exist.

    Directory mtimes are included so a capture *deleted* from the tree also
    registers as a change — the scan is idempotent either way, but the
    poll should not go blind to an edit that leaves file mtimes alone.
    """
    if not root.is_dir():
        return None
    newest: float | None = None
    for dirpath, _dirnames, filenames in os.walk(root):
        for candidate in (dirpath, *(os.path.join(dirpath, name) for name in filenames)):
            try:
                mtime = os.stat(candidate).st_mtime
            except OSError:
                continue
            if newest is None or mtime > newest:
                newest = mtime
    return newest


def _make_inbox_tick(path: Path) -> Work:
    """Inbox watcher — the ``inbox watch`` loop body, on the shared seam.

    Active whenever ``inbox.file_path`` resolves; reuses
    ``inbox.poll_interval_seconds`` rather than minting a daemon-local knob.
    """
    state: dict[str, float | None] = {"last_mtime": None}

    async def _tick() -> str:
        from particles.db import session_scope
        from particles.operations.inbox import process_inbox

        current = _mtime_or_none(path)
        if current is None or current == state["last_mtime"]:
            return "unchanged"
        store = get_config().daemon.store
        async with session_scope(store) as session:
            summary = await process_inbox(session, path, deposited_by="inbox")
        # Re-stat after the in-place rewrite so its own mtime bump does not
        # trigger an immediate second pass (the `inbox watch` behaviour).
        state["last_mtime"] = _mtime_or_none(path)
        processed, failed = len(summary["processed"]), len(summary["failed"])
        log.info("daemon inbox: processed %d URL(s), %d failed", processed, failed)
        return f"processed {processed}, failed {failed}"

    return _tick


def _make_web_clipper_tick(captures_dir: Path) -> Work:
    """Web-clipper watcher — periodically re-runs the idempotent one-shot scan.

    Active whenever ``daemon.web_clipper_dir`` is set. The scan dedups on body
    hash and clipped URL, so re-running it is cheap and safe; the mtime gate is
    there to skip the walk-and-hash cost, not for correctness.
    """
    state: dict[str, float | None] = {"last_mtime": None}

    async def _tick() -> str:
        from particles.db import session_scope
        from particles.operations.deposit import deposit_web_clipper

        current = _tree_mtime(captures_dir)
        if current is None:
            return "captures directory absent"
        if current == state["last_mtime"]:
            return "unchanged"
        store = get_config().daemon.store
        async with session_scope(store) as session:
            results = await deposit_web_clipper(session, captures_dir)
        state["last_mtime"] = _tree_mtime(captures_dir)
        log.info("daemon web-clipper: scanned %s, %d capture(s)", captures_dir, len(results))
        return f"deposited {len(results)}"

    return _tick


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _register(name: str, interval: IntervalFn, work: Work) -> None:
    status = DaemonTaskStatus(name=name, interval_seconds=interval())
    _runtime.statuses[name] = status
    task = asyncio.create_task(_poll_loop(status, interval, work), name=f"particles-daemon:{name}")
    task.add_done_callback(lambda t: _on_task_done(status, t))
    _runtime.tasks.append(task)


def start_daemon() -> bool:
    """Start the daemon tasks if daemon mode is on. Returns whether it started.

    Called from the FastAPI lifespan **after** ``create_tables()``, so the first
    tick never races schema creation. A no-op (returning ``False``) when
    ``daemon.enabled`` is false — which is the default, and is what keeps a plain
    ``engine serve`` byte-identical to its earlier behaviour.

    Which watchers run is *derived from configuration*, not from extra switches
    : the inbox watcher is active whenever ``inbox.file_path``
    resolves, the web-clipper watcher whenever ``daemon.web_clipper_dir`` is set.
    An inactive watcher says so in the log, so "why isn't my inbox draining?"
    is answered at startup rather than by silence.
    """
    if _runtime.tasks:
        log.debug("daemon already started; ignoring duplicate start_daemon()")
        return _runtime.enabled
    cfg = get_config()
    if not cfg.daemon.enabled:
        return False

    _runtime.enabled = True
    _register(
        "consolidation",
        lambda: get_config().daemon.consolidation_tick_minutes * 60.0,
        consolidation_tick,
    )

    from particles.operations.inbox import resolve_inbox_path

    inbox_path = resolve_inbox_path()
    if inbox_path is not None:
        _register(
            "inbox",
            lambda: float(get_config().inbox.poll_interval_seconds),
            _make_inbox_tick(inbox_path),
        )
    else:
        log.info("daemon: inbox watcher inactive (inbox.file_path is not set)")

    clipper_dir = cfg.daemon.web_clipper_dir
    if clipper_dir:
        _register(
            "web-clipper",
            lambda: get_config().daemon.web_clipper_poll_minutes * 60.0,
            _make_web_clipper_tick(Path(clipper_dir).expanduser()),
        )
    else:
        log.info("daemon: web-clipper watcher inactive (daemon.web_clipper_dir is not set)")

    log.info(
        "Daemon mode ON: %s",
        ", ".join(f"{n} every {s.interval_seconds:.0f}s" for n, s in _runtime.statuses.items()),
    )
    return True


async def stop_daemon() -> None:
    """Cancel every daemon task and wait for it to unwind. Idempotent."""
    tasks = list(_runtime.tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("Daemon stopped (%d task(s)).", len(tasks))
    _runtime.reset()
