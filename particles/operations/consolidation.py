# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Dream cycle — the scheduled consolidation operation.

``run_consolidation`` composes the **existing** engine passes in the fixed §3
order and adds no detection of its own: extract catch-up (capped), the
reconcile sweep, one ``collect_cards(semantic=…)`` census pass under
the probe cap + the §4 delta scope, the curation-queue
refresh over the *same* card collection, the utility-mining pass, and
the projection re-render (injected by the Surface caller — the
render-splice tail is CLI-side, so the Engine never imports upward).

Each completed run writes one ``CONSOLIDATION_RUN`` operator event (
§7 — the fold): a versioned payload carrying per-pass status /
durations / LLM call counts, the machine-readable census, degradation
disclosures, and ``started_at`` — the instant the next run's delta window
opens from (correction, v1.74.1: the window opens at the previous
run's *start*, not its completion, so particles minted mid-cycle — by pass 1
itself or by an interleaving harvest — are re-probed by the next run rather
than falling into a permanent gap; overlap re-probes are idempotent and
cheap). Only a successful, non-degraded run by this verb's own actor is
watermark-eligible. ``particles audit`` records the same event shape via
:func:`record_audit_run` (``actor: audit``) for the §7 delta report, but an
audit event neither advances the watermark nor satisfies ``--if-due``.

The cron contract (§8): one cycle at a time (the ``consolidate.lock`` file in
the integration state directory, stale-reclaimed), ``--if-due`` cadence guard,
and continue-and-report on pass failure — a flaky night never leaves the
zero-LLM passes unrun or the run record unwritten.

Progress: a caller that passes ``on_progress`` hears each pass start
(``pass`` N of M), the pass's own counter where it has one (snapshots, probes,
readings, sessions), every change to a pending Message Batches wait with the
budget left, and each pass's end with its status and a one-line
tally (:class:`PassEnded`), all as the shared progress event. The
cycle never prints; the CLI renders the events. The daemon passes nothing and
nothing is computed.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.consolidation_cadence import CensusDecision, decide_census, is_due
from particles.core.contradiction_disclosure import ConfirmedPair, census_sides
from particles.core.progress import ProgressCallback, ProgressEvent
from particles.core.schema import SuggestMode
from particles.core.spend_budget import decide_spend
from particles.core.status import Status
from particles.db import write_lock, write_transaction
from particles.llm.batch_budget import (
    BatchWaitBudget,
    batch_wait_budget,
    current_batch_wait_budget,
)
from particles.llm.errors import AccountLevelLLMError
from particles.llm.spend_budget import SpendBudget, current_spend_budget, spend_budget
from particles.llm.usage import (
    LLMUsage,
    UsageAccumulator,
    format_usd,
    render_usage_line,
    track_usage,
)
from particles.operations._llm import llm_circuit_open
from particles.operations.abstraction import AbstractionReport, run_abstraction_pass
from particles.operations.closure_measure import (
    PAYLOAD_KEY as CLOSURE_PAYLOAD_KEY,
)
from particles.operations.closure_measure import (
    ClosureMeasure,
    measure_closure,
    measure_from_payload,
    render_closure_lines,
    window_start_after,
)
from particles.operations.contradiction_disclosure import (
    DisclosureReport,
    covered_pair_set,
    run_disclosure,
)
from particles.operations.curation.cards import CardKind, CurationCard
from particles.operations.curation.collect import cards_from_findings, collect_cards
from particles.operations.curation.session import _suppressed_keys, build_curation_queue
from particles.operations.curation.snapshot import QueueSource, collect_and_persist
from particles.operations.lint import ContradictionProbeControl
from particles.operations.lint.contestedness import contested_findings_for
from particles.operations.lint.contradictions import contradiction_partner, count_disagreements
from particles.operations.lint.open_inconsistency import open_inconsistency_findings
from particles.operations.modality import reclassified_particle_ids_since
from particles.operations.reanchor import ReanchorReport, prior_cursor, run_reanchor
from particles.operations.reconcile import (
    count_supersession_candidates,
    count_update_candidates,
    reconcile_supersession,
    reconcile_updates,
)
from particles.operations.spend_estimate import (
    PassEstimate,
    context_call_estimate,
    estimate_batch_chunk,
    extraction_estimate,
    probe_estimate,
)
from particles.operations.utility_exposure import ExposureReader
from particles.operations.utility_mining import (
    SessionMine,
    estimate_judge_calls,
    harvested_sessions,
    judge_sessions,
    plan_harvested_session,
    record_session_mine,
)
from particles.store.curation_snapshot_store import CollectionScope, latest_snapshot
from particles.store.event_store import (
    OperatorEvent,
    OperatorEventType,
    list_events,
    record_event,
)
from particles.store.particle_store import (
    get_census_records,
    get_particle_ids_changed_since,
    get_particle_ids_for_entries,
    get_particles_by_ids,
    get_particles_by_status,
)

if sys.platform != "win32":
    import fcntl

if TYPE_CHECKING:
    from particles.core.schema import Particle
    from particles.corpus.store import PendingSnapshot
    from particles.ingest.pipeline import SnapshotOutcome
    from particles.operations.audit import AuditReport

log = logging.getLogger(__name__)

#: Version stamp of the ``CONSOLIDATION_RUN`` event payload.
RUN_PAYLOAD_FORMAT = 1

#: The LLM purposes the cycle's passes route through (§5: existing purposes,
#: no new "consolidation" purpose) — recorded on the run record so a routing
#: change is visible in the audit trail.
_RUN_PURPOSES: tuple[str, ...] = ("extraction", "semantic_lint", "abstraction", "use_judge")

ProjectionRunner = Callable[[], Awaitable[dict[str, Any]]]

#: The actor every scheduled run records under; the cadence, the watermark and
#: the closure measure's history all key on it.
CONSOLIDATION_ACTOR = "memory-consolidate"

#: The §3 passes in run order: the denominator of the ``pass N/M`` progress.
PASS_ORDER: tuple[str, ...] = (
    "refresh",
    "extract",
    "reconcile",
    "reconcile_updates",
    "reanchor",
    "census",
    "disclose",
    "curation",
    "utility",
    "abstraction",
    "projection",
    "measure",
)


# ---------------------------------------------------------------------------
# Progress events, rendered by the CLI
# ---------------------------------------------------------------------------
#
# Phases the cycle emits:
#
# - ``pass``: a pass started. ``done`` is its position in :data:`PASS_ORDER`
#   (1-based), ``total`` the pass count, ``label`` its name.
# - ``<pass name>``: the running pass's own counter, e.g. ``extract`` 3 of 20
#   ``snapshots``. ``label`` names the unit.
# - ``batch_wait``: a Message Batches wait moved (:class:`BatchWaiting`).
# - ``pass_end``: a pass finished or was skipped (:class:`PassEnded`).

PassOutcome = Literal["ok", "degraded", "failed", "skipped"]


@dataclass(frozen=True)
class PassEnded(ProgressEvent):
    """A pass finished (or was skipped), with the tally to show for it.

    ``done`` / ``total`` / ``label`` are the pass's position and name, as on
    its ``pass`` event. ``degraded`` means the pass ran but left work undone
    (a batch cut short or cancelled, snapshots left for retry, a structural-only
    census); the report says what.
    """

    outcome: PassOutcome = "ok"
    duration_seconds: float = 0.0
    summary: str = ""


@dataclass(frozen=True)
class BatchWaiting(ProgressEvent):
    """A Message Batches wait moved.

    ``done`` is the seconds the longest-open batch has waited and ``total`` its
    cap in seconds; both are 0 once no batch is pending. ``label`` is the pass
    that submitted it. ``budget_left_seconds`` is the run's balance less the
    waits still open: the only honest upper bound on the waiting left, since
    the cycle cannot know when the batch will end.
    """

    budget_left_seconds: float = 0.0
    budget_seconds: float = 0.0


@dataclass
class _RunProgress:
    """The progress sink for one run, and the last counter each pass reported."""

    emit: ProgressCallback
    counters: dict[str, tuple[int, int]] = field(default_factory=dict)

    def send(self, event: ProgressEvent) -> None:
        # A rendering failure must never fail the cycle.
        try:
            self.emit(event)
        except Exception as exc:  # noqa: BLE001 — display only
            log.debug("consolidation: progress callback failed: %s", exc)


#: The running cycle's progress sink; ``None`` when the caller asked for none.
_active_progress: ContextVar[_RunProgress | None] = ContextVar("_active_progress", default=None)


def _pass_position(name: str) -> int:
    return PASS_ORDER.index(name) + 1 if name in PASS_ORDER else 0


def _count(pass_name: str, done: int, total: int, unit: str) -> None:
    """Report the running pass's own counter (``done`` of ``total`` ``unit``)."""
    progress = _active_progress.get()
    if progress is None:
        return
    progress.counters[pass_name] = (done, total)
    progress.send(ProgressEvent(phase=pass_name, done=done, total=total, label=unit))


def _batch_wait_publisher(progress: _RunProgress) -> Callable[[BatchWaitBudget], None]:
    """The budget's ``on_wait`` hook: turn its wait state into a progress event."""

    def _publish(budget: BatchWaitBudget) -> None:
        pending = budget.pending_wait()
        elapsed, cap = pending if pending is not None else (0.0, 0.0)
        progress.send(
            BatchWaiting(
                phase="batch_wait",
                done=int(elapsed),
                total=int(cap),
                label=budget.pass_name or "",
                budget_left_seconds=budget.live_remaining(),
                budget_seconds=budget.budget_seconds,
            )
        )

    return _publish


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_utc(dt: datetime) -> datetime:
    """Normalize a possibly-naive stored datetime to UTC for comparisons."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# The cycle lockfile — one cycle at a time
# ---------------------------------------------------------------------------
#
# The lock is an OS advisory lock (``flock``) on the lock path, held for the
# whole cycle. The kernel releases it when the holder exits by any route, so a
# crashed run frees the cadence at once and a live run is never reclaimed,
# however long it runs. The file carries a JSON description of
# the holder under today's ``pid`` / ``started_at`` names, so a pre-change
# binary still honours it. Where ``flock`` is unavailable the
# lock falls back to the pid-and-age rule.

#: Written into a description by a holder of the kernel lock. A description
#: without it was written by a pre-change binary.
_KERNEL_MARK = "kernel"
_FALLBACK_MARK = "fallback"
_NO_FLOCK_ERRNOS = frozenset(
    {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOLCK, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}
)
#: Lock paths already warned about falling back, so the warning is logged once.
_fallback_warned: set[str] = set()
#: Reads of a description that fails to parse before it is judged malformed: a
#: torn read clears within microseconds, a corrupt file costs ~50 ms once.
_DESCRIPTION_READ_ATTEMPTS = 5
_DESCRIPTION_RETRY_SECONDS = 0.01


@dataclass(eq=False)
class CycleLock:
    """A held cycle lock; release with :func:`release_cycle_lock`.

    ``fd`` is the descriptor holding the kernel lock, or ``None`` for a
    pid-and-age fallback lock. ``info`` is the holder description written into
    the lock file.
    """

    path: Path
    fd: int | None = None
    info: dict[str, Any] = field(default_factory=dict)
    _mutex: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _replaced_logged: bool = field(default=False, repr=False)

    def write_description(self) -> None:
        """Rewrite the holder description in place.

        Through the held descriptor, never ``os.replace``: a replace would swap
        the inode out from under the kernel lock. The fallback lock is rewritten
        in place too, because a Windows replace fails while a contender has the
        file open. See :func:`_overwrite_description` for why a reader never
        sees a partial description.
        """
        data = json.dumps(self.info).encode("utf-8")
        with self._mutex:
            if self.fd is not None:
                _overwrite_description(self.fd, data)
                return
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                _overwrite_description(fd, data)
            finally:
                os.close(fd)

    def set_pass(self, name: str) -> None:
        """Record the pass now running."""
        now = _utcnow().isoformat()
        self.info["pass"] = name
        self.info["pass_started_at"] = now
        self.info["heartbeat_at"] = now
        self.write_description()

    def beat(self) -> None:
        """Refresh ``heartbeat_at`` and notice a lock file replaced under us."""
        if self.fd is None:
            return
        if not _fd_names_path(self.fd, self.path) and not self._replaced_logged:
            self._replaced_logged = True
            other = _read_description(self.path) or {}
            log.error(
                "consolidation lock file %s was replaced while this cycle held it "
                "(pid now named there: %s); a pre-change binary may have reclaimed it "
                "",
                self.path,
                other.get("pid", "unknown"),
            )
        self.info["heartbeat_at"] = _utcnow().isoformat()
        self.write_description()

    def _start_heartbeat(self, interval_seconds: float) -> None:
        def _loop() -> None:
            while not self._stop.wait(interval_seconds):
                try:
                    self.beat()
                except OSError as exc:
                    log.warning("consolidation lock heartbeat failed: %s", exc)

        self._thread = threading.Thread(
            target=_loop, name="consolidation-lock-heartbeat", daemon=True
        )
        self._thread.start()


@dataclass(frozen=True)
class LockHeld:
    """The lock is held by a live cycle: the caller skips.

    ``reason`` is the skip message naming the holder. ``warning`` is set when
    the holder has run longer than ``consolidation.lock_timeout_minutes``, for
    the caller to put on stderr.
    """

    reason: str
    warning: str | None = None


#: The lock the current cycle holds, for ``_run_pass`` to record the pass on.
#: A context variable, not a global: the benchmark runs several cycles
#: concurrently in one process, each on its own lock path.
_active_lock: ContextVar[CycleLock | None] = ContextVar("_active_lock", default=None)


def cycle_lock_path() -> Path:
    """``consolidate.lock`` in the integration state directory."""
    return Path(get_config().claude_code.state_dir).expanduser() / "consolidate.lock"


def _host() -> str:
    return socket.gethostname()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _parse_time(value: object) -> datetime | None:
    try:
        return _as_utc(datetime.fromisoformat(str(value)))
    except (TypeError, ValueError):
        return None


def _read_description(path: Path) -> dict[str, Any] | None:
    """The holder description in ``path``, or ``None`` when absent or unparseable.

    The holder rewrites the description in place while contenders read it, and
    a read that races the write can return new bytes followed by old ones. So a
    non-empty file that fails to parse is read again before it is judged
    malformed. An empty file is an unheld lock and is not retried, since the
    holder's rewrite never leaves the file empty (see
    :func:`_overwrite_description`).
    """
    for _attempt in range(_DESCRIPTION_READ_ATTEMPTS):
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        if not raw.strip():
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            time.sleep(_DESCRIPTION_RETRY_SECONDS)
            continue
        return data if isinstance(data, dict) else None
    return None


def _overwrite_description(fd: int, data: bytes) -> None:
    """Replace the file's contents with ``data`` without an empty or cut-off window.

    Contenders read the description while the holder rewrites it, and an
    unparseable one reads as "no holder" (the cross-host check) or "stale"
    (the fallback). Truncating first would leave the file empty until the
    write lands, and an empty file is an unheld lock. Instead the new
    description goes over the old one, padded with spaces to at least the old
    length so no stale tail survives, and only then is the file cut to size.
    So the file is never empty or shorter than a whole description, and once
    the write lands it parses, since JSON permits trailing whitespace. A read
    racing the write itself can still see a mix of old and new bytes, which
    :func:`_read_description` re-reads.
    """
    size = os.fstat(fd).st_size
    os.lseek(fd, 0, os.SEEK_SET)
    view = memoryview(data.ljust(size))
    while view:
        view = view[os.write(fd, view) :]
    if size > len(data):
        os.ftruncate(fd, len(data))
    os.fsync(fd)


def _fd_names_path(fd: int, path: Path) -> bool:
    """True when ``fd`` is still the file at ``path`` (not unlinked or replaced)."""
    held = os.fstat(fd)
    if held.st_nlink == 0:
        return False
    try:
        current = path.stat()
    except OSError:
        return False
    return (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino)


def _legacy_live(data: dict[str, Any], timeout_minutes: int) -> bool:
    """§8's rule: live when the pid is alive and younger than the timeout."""
    pid = data.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return False
    started = _parse_time(data.get("started_at"))
    if started is None:
        return False
    return _utcnow() - started <= timedelta(minutes=timeout_minutes)


def _lock_is_stale(path: Path, timeout_minutes: int) -> bool:
    """The fallback's staleness test, used only without ``flock``.

    An unreadable / malformed lock is stale by definition — a crashed run must
    never wedge the cadence forever.
    """
    data = _read_description(path)
    return data is None or not _legacy_live(data, timeout_minutes)


def _reclaim_stale_lock(path: Path, timeout_minutes: int) -> bool:
    """Unlink ``path`` only if it is stale AND unchanged since it was judged.

    The fallback only. Closes the reclaim TOCTOU: two contenders
    can both judge the same old lock stale; without the re-verify, the slower
    one would unlink the faster one's FRESH lock. Returns ``True`` when the
    caller may retry the exclusive create.
    """
    try:
        before = path.stat()
    except OSError:
        return True  # gone already — the exclusive-create retry decides
    if not _lock_is_stale(path, timeout_minutes):
        return False
    try:
        after = path.stat()
    except OSError:
        return True
    if (after.st_ino, after.st_mtime_ns) != (before.st_ino, before.st_mtime_ns):
        # Replaced mid-judgement: a rival contender reclaimed first and holds
        # a fresh lock. Do NOT unlink it — lost the race.
        return False
    log.warning("Reclaiming stale consolidation lock at %s", path)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    return True


def _format_held(data: dict[str, Any] | None, timeout_minutes: int) -> LockHeld:
    """The skip message naming the holder, plus the long-hold warning (§3/§4)."""
    if not data:
        return LockHeld("consolidation already running — skipped")
    parts: list[str] = []
    pid = data.get("pid")
    host = data.get("host")
    if pid is not None:
        parts.append(f"pid {pid}" + (f" on {host}" if host else ""))
    started = _parse_time(data.get("started_at"))
    if started is not None:
        parts.append(f"started {started:%Y-%m-%d %H:%M} UTC")
    current = data.get("pass")
    if current:
        since = _parse_time(data.get("pass_started_at"))
        parts.append(f"pass {current}" + (f" since {since:%H:%M} UTC" if since else ""))
    reason = "consolidation already running"
    if parts:
        reason += f" ({', '.join(parts)})"
    warning = None
    if started is not None:
        held = _utcnow() - started
        if held > timedelta(minutes=timeout_minutes):
            minutes = int(held.total_seconds() // 60)
            warning = f"consolidation lock held for {minutes // 60}h {minutes % 60:02d}m"
            if pid is not None:
                warning += f"; if it is hung, stop pid {pid}"
            reason += f"; {warning}"
            log.warning("%s", warning)
    return LockHeld(f"{reason} — skipped", warning)


def _new_description(mark: str) -> dict[str, Any]:
    now = _utcnow().isoformat()
    return {
        "pid": os.getpid(),
        "started_at": now,
        "host": _host(),
        "heartbeat_at": now,
        "lock": mark,
    }


def _acquire_fallback(path: Path, timeout_minutes: int) -> CycleLock | LockHeld:
    """§8's pid-and-age lock, where ``flock`` is unavailable."""
    key = str(path)
    if key not in _fallback_warned:
        _fallback_warned.add(key)
        log.warning(
            "consolidation: no advisory file locks at %s; using the pid-and-age cycle lock",
            path,
        )
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt > 0 or not _reclaim_stale_lock(path, timeout_minutes):
                return _format_held(_read_description(path), timeout_minutes)
            continue
        lock = CycleLock(path, None, _new_description(_FALLBACK_MARK))
        # Through the creating descriptor, so the empty file a contender would
        # judge stale exists for one write, not a close and a reopen.
        try:
            _overwrite_description(fd, json.dumps(lock.info).encode("utf-8"))
        finally:
            os.close(fd)
        return lock
    return _format_held(_read_description(path), timeout_minutes)


def _unlock(fd: int) -> None:
    with contextlib.suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def acquire_cycle_lock(
    path: Path,
    *,
    timeout_minutes: int,
    heartbeat_seconds: float = 60.0,
    heartbeat_stale_minutes: int = 10,
) -> CycleLock | LockHeld:
    """Acquire the cycle lock, or report the live holder.

    Returns a :class:`CycleLock` for the caller to hold for the cycle, or a
    :class:`LockHeld` when a live cycle holds it — the caller exits 0 with
    its ``reason`` (contention is normal under cron, not an alarm).

    The order is load-bearing: open without truncating, take the
    kernel lock, confirm the descriptor still names the path, read the
    existing description, and only then write this holder's.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        return _acquire_fallback(path, timeout_minutes)
    host = _host()
    for _attempt in range(2):
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return _format_held(_read_description(path), timeout_minutes)
        except OSError as exc:
            os.close(fd)
            if exc.errno in _NO_FLOCK_ERRNOS:
                return _acquire_fallback(path, timeout_minutes)
            raise
        if not _fd_names_path(fd, path):
            # A rival replaced or removed the file between open and flock.
            _unlock(fd)
            continue
        existing = _read_description(path)
        if existing is not None:
            if existing.get("lock") is None:
                # A pre-change holder takes no kernel lock.
                if _legacy_live(existing, timeout_minutes):
                    _unlock(fd)
                    return _format_held(existing, timeout_minutes)
            elif existing.get("host") not in (None, host):
                # A holder on another host whose kernel lock this mount does not
                # share: judged by its heartbeat.
                beat = _parse_time(existing.get("heartbeat_at"))
                fresh = beat is not None and _utcnow() - beat <= timedelta(
                    minutes=heartbeat_stale_minutes
                )
                if fresh:
                    _unlock(fd)
                    return _format_held(existing, timeout_minutes)
                log.warning(
                    "consolidation: overwriting lock of pid %s on %s, whose heartbeat stopped",
                    existing.get("pid"),
                    existing.get("host"),
                )
        lock = CycleLock(path, fd, _new_description(_KERNEL_MARK))
        lock.write_description()
        lock._start_heartbeat(heartbeat_seconds)
        return lock
    return _format_held(_read_description(path), timeout_minutes)


def release_cycle_lock(lock: CycleLock) -> None:
    """Release a held cycle lock (idempotent).

    A kernel lock file is truncated, never unlinked: an empty file is an unheld
    lock, and unlinking would let a contender lock an orphaned inode.
    """
    lock._stop.set()
    if lock._thread is not None:
        lock._thread.join(timeout=5)
        lock._thread = None
    with lock._mutex:
        fd, lock.fd = lock.fd, None
        if fd is None:
            if lock.info.get("lock") == _FALLBACK_MARK:
                lock.info["lock"] = None
                with contextlib.suppress(FileNotFoundError):
                    lock.path.unlink()
            return
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
        _unlock(fd)


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class ConsolidationPass(BaseModel):
    """One §3 pass: what ran, for how long, at what LLM spend."""

    name: str
    status: Literal["ran", "skipped", "failed"]
    #: Skip reason or error text; None for a clean "ran".
    detail: str | None = None
    duration_seconds: float = 0.0
    #: The §11 spend side. Extraction counts extracted snapshots (a lower
    #: bound: a chunked source makes one call per chunk); the census counts
    #: probes run; reconcile counts replacement-signal probes; utility counts
    #: behavioural matcher calls.
    llm_calls: int = 0
    #: What the run's batch-wait budget did to this pass: batches
    #: cut short, sets moved to sequential calls, and what each cancelled
    #: batch kept and lost. None when it did nothing.
    batch_wait: str | None = None

    def payload_status(self) -> str:
        """The run-record status string: ``ran | skipped(<r>) | failed(<e>)``."""
        if self.status == "ran":
            return "ran"
        return f"{self.status}({self.detail or 'unknown'})"


class BatchWaitSummary(BaseModel):
    """The run-level batch-wait budget's account for one run."""

    budget_seconds: float
    spent_seconds: float = 0.0
    batches: int = 0
    batches_cut_short: int = 0
    cut_short_requests: int = 0
    sequential_sets: int = 0
    sequential_requests: int = 0
    #: Every batch cancelled in the run, whatever cancelled it, and what became
    #: of its requests: read back, answered by the sequential re-run, or
    #: unavailable.
    cancelled_batches: int = 0
    cancelled_kept: int = 0
    cancelled_rerun: int = 0
    cancelled_lost: int = 0
    #: The pass during which the balance first fell below the floor.
    exhausted_in_pass: str | None = None

    @classmethod
    def from_budget(cls, budget: BatchWaitBudget) -> BatchWaitSummary:
        """Read the counters off a finished run's budget."""
        return cls(
            budget_seconds=budget.budget_seconds,
            spent_seconds=round(budget.spent_seconds, 3),
            batches=budget.batches,
            batches_cut_short=budget.batches_cut_short,
            cut_short_requests=budget.cut_short_requests,
            sequential_sets=budget.sequential_sets,
            sequential_requests=budget.sequential_requests,
            cancelled_batches=budget.cancelled_batches,
            cancelled_kept=budget.cancelled_kept,
            cancelled_rerun=budget.cancelled_rerun,
            cancelled_lost=budget.cancelled_lost,
            exhausted_in_pass=budget.exhausted_in_pass,
        )

    @property
    def moved_work(self) -> bool:
        """True when a batch was cut short or cancelled, or a set was sent sequential."""
        return bool(self.batches_cut_short or self.sequential_sets or self.cancelled_batches)


class BudgetSummary(BaseModel):
    """The run's dollar budget and what it spent.

    Written on every run record, budgeted or not, so the spend sits beside the
    budget it was held to. ``spent_usd`` is list price over the priced rows of
    the run's measured usage (an unpriced model's tokens are disclosed by the
    usage line, never folded in at zero).
    """

    #: ``consolidation.budget_usd`` at run time; ``None`` is unbounded.
    budget_usd: float | None = None
    spent_usd: float = 0.0
    #: Passes (or a pass's LLM tier) the budget skipped, in run order.
    skipped_passes: list[str] = Field(default_factory=list)
    #: Message Batches chunks not submitted, and their requests.
    skipped_chunks: int = 0
    skipped_requests: int = 0
    #: The pass during which the first chunk was declined.
    exhausted_in_pass: str | None = None

    @property
    def limited(self) -> bool:
        """True when the budget left LLM work undone this run."""
        return bool(self.skipped_passes or self.skipped_requests)


class ConsolidationReport(BaseModel):
    """The output of :func:`run_consolidation`."""

    store: str = "default"
    actor: str = CONSOLIDATION_ACTOR
    outcome: Literal["ran", "skipped"] = "ran"
    #: Set when ``outcome == "skipped"`` (lock held / --if-due not due).
    skip_reason: str | None = None
    #: Set on a lock skip when the holder has run past
    #: ``consolidation.lock_timeout_minutes``: the "stop pid N" hint the CLI
    #: writes to stderr.
    lock_warning: str | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None

    # §4 scope. ``scope`` records what was asked for; ``effective_scope`` what
    # actually ran ("store" on a first run with no watermark). ``watermark``
    # is the instant the delta scope was computed FROM: the previous
    # watermark-eligible run's ``started_at`` (correction v1.74.1 — see
    # :func:`latest_run_event`).
    scope: Literal["delta", "store"] = "delta"
    effective_scope: Literal["delta", "store"] = "delta"
    watermark: datetime | None = None
    scope_particle_count: int | None = None

    passes: list[ConsolidationPass] = Field(default_factory=list)

    # §6 degradation disclosure — mirrors ``LintReport.semantic_skipped``.
    semantic_degraded: bool = False
    semantic_degraded_reason: str | None = None
    #: provider:model per purpose, so a routing change is visible (§7).
    providers: dict[str, str] = Field(default_factory=dict)

    # Pass 0.5 — local-source refresh. Zero-LLM, so it runs on
    # degraded nights too. ``refresh_unchanged_mtime`` is the no-I/O tier-1
    # short-circuit; ``refresh_unchanged_hash`` read the bytes and matched.
    refresh_checked: int = 0
    refresh_unchanged_mtime: int = 0
    refresh_unchanged_hash: int = 0
    refresh_updated: int = 0
    refresh_missing: int = 0
    refresh_remaining: int = 0

    # Pass 1 — extract catch-up.
    pending_total: int = 0
    #: Snapshots extracted to COMPLETE this run (``pending_empty`` included).
    pending_extracted: int = 0
    #: Of those, snapshots that completed with no belief written, carried
    #: forward or folded into an existing one. Disclosed, never silent.
    pending_empty: int = 0
    #: Snapshots with a failed LLM call: the pipeline handed them back PENDING.
    #: Not extracted; the next run retries them behind every snapshot not yet
    #: tried.
    pending_retry: int = 0
    #: Of those, the snapshots that kept part of their read: an APPEND_ONLY
    #: read's answered chunks were written, and the retry reads only the rest
    #:.
    pending_partial: int = 0
    #: Snapshots not tried because an earlier snapshot of the entry holds a
    #: partial whole read: they wait until it completes.
    pending_waiting: int = 0
    pending_failed: int = 0
    pending_remaining: int = 0
    #: Listed as PENDING, then found superseded when its task reached it
    #: (another runner collapsed it). No longer owed; not in the remainder.
    pending_superseded_late: int = 0
    #: Snapshots stranded IN_PROGRESS by a killed run, reset to PENDING before
    #: the pass listed its work.
    pending_reset_stale: int = 0
    #: Capture time of the oldest snapshot still PENDING after the pass: how
    #: long the backlog has been waiting, which the count alone does not show.
    pending_oldest_at: datetime | None = None
    # Superseded generations of MUTABLE entries skipped before the pass listed
    # its work. Not part of ``pending_total``: they were never owed.
    pending_collapsed: int = 0

    # Pass 2 — reconcile sweep (probe-bearing: one semantic_lint call per
    # candidate pair, capped at consolidation.max_reconcile_probes).
    reconcile_demoted: int = 0
    # the same-subject update sweep's counters.
    update_demoted: int = 0
    update_candidate_pairs: int = 0
    update_probes_run: int = 0
    # qualifying pairs the probe-verdict ledger had already cleared.
    update_previously_cleared: int = 0
    # confirmed pairs that give a fixed slot two values, each opened
    # for review with both claims left ACTIVE instead of retired.
    update_fixed_slot: int = 0
    # Pass 2c: the re-anchor pass, its own sub-report.
    reanchor: ReanchorReport | None = None
    reconcile_candidate_pairs: int = 0
    reconcile_probes_run: int = 0

    # Pass 3 cadence: the census runs every
    # ``consolidation.census.interval_hours``, not every night. ``census_skipped``
    # is the disclosure when this run did not run it; ``census_last_ran_at`` and
    # ``census_next_due_at`` come from the last census's run record.
    census_skipped: str | None = None
    census_last_ran_at: datetime | None = None
    census_next_due_at: datetime | None = None
    #: The last census's headline counts, read from its run record and shown
    #: on a night that skipped it. Never this night's measurement.
    census_last_headline: dict[str, int] | None = None
    # The census's own scope. Its delta window opens at the last census's
    # ``started_at``, not the last run's: a weekly census scoped to one night's
    # changes would never probe the six nights it skipped.
    census_scope: Literal["delta", "store"] | None = None
    census_watermark: datetime | None = None
    census_scope_particle_count: int | None = None

    # Pass 3 — census (the machine-readable fields).
    card_counts: dict[str, int] = Field(default_factory=dict)
    # the CONTESTED class basis, so a delta of "+2
    # contested" is attributable to the instrument that produced it rather
    # than being one unattributed total. A card firing two bases counts under
    # both. Additive on the versioned run-record payload — no format bump.
    contested_bases: dict[str, int] = Field(default_factory=dict)
    contradiction_candidate_pairs: int = 0
    contradiction_intra_scope_pairs: int = 0
    contradiction_probes_run: int = 0
    # candidate pairs not probed because the probe-verdict ledger
    # already cleared them; outside ``contradiction_candidate_pairs``.
    contradiction_previously_cleared: int = 0
    # the second reading of each flag, as in the interactive audit.
    # ``contradiction_verified`` False means every flag was counted.
    contradiction_verified: bool = False
    contradiction_flagged: int = 0
    contradiction_confirmed: int = 0
    contradiction_unverified: int = 0
    contradiction_verifications_run: int = 0
    # reported pairs grouped by shared claim. None when the probe
    # did not run; the headline then counts contradiction cards.
    contradiction_disagreements: int | None = None
    contradiction_grouped_pairs: int = 0
    duplicate_candidate_pairs_total: int = 0
    duplicate_in_scope: int = 0
    # Pass 3's reported and confirmed pairs, handed to pass 3b.
    # Working state, not run-record fields.
    contradiction_finding_pairs: list[tuple[str, str, bool]] = Field(
        default_factory=list, exclude=True
    )
    contradiction_confirmed_pairs: list[ConfirmedPair] = Field(default_factory=list, exclude=True)

    # Pass 3b — the contradiction disclosure.
    disclosure: DisclosureReport | None = None
    #: Open census records after the pass: each one disagreement in the
    #: contradictions headline.
    disclosure_open_records: int = 0
    #: CONTESTED cards whose claim sits in an open census record. Recorded for
    #: the run record only: since the headline counts each open
    #: record through its INCONSISTENCY card and never counts claims.
    contested_census_claims: int = 0

    # Pass 4 — curation-queue refresh. Since the card collection is
    # **persisted**, so this pass is what makes the morning's `GET /curation`
    # fast rather than a report line that is thrown away. The rendered lines
    # below stay in the run record; the cards live in `curation_snapshots`.
    curation_queue: list[str] = Field(default_factory=list)
    curation_queue_total: int = 0
    # the collection this run wrote. A pointer, not the blob —
    # the audit trail says which snapshot the night produced without carrying
    # megabytes of derived cards in an append-only event.
    curation_snapshot_id: str | None = None
    #: When the served collection was built, on a night that skipped the census
    #: and served the last stored one. None when this run wrote it.
    curation_snapshot_built_at: datetime | None = None

    # Pass 5 — utility mining. ``utility_behavioural_exhausted_after`` is the
    # session count at which the shared per-run behavioural budget ran out
    # (None = never) — the truncation disclosure's numerator.
    utility_literal: int = 0
    utility_behavioural: int = 0
    utility_behavioural_calls: int = 0
    utility_sessions_mined: int = 0
    utility_behavioural_exhausted_after: int | None = None

    # Pass 5b — abstraction promotion: the pass's own sub-report
    # (clusters, promotions/proposals, the §5 revalidation ladder outcomes).
    abstraction: AbstractionReport | None = None

    # Pass 6 — projection re-render (the runner's telemetry dict).
    projection: dict[str, Any] | None = None

    # Pass 6b — the closure measure: the store-wide contested
    # fraction and the window's autonomous versus gesture transitions. Read
    # only; recorded under the run record's ``closure`` key.
    closure: ClosureMeasure | None = None
    #: The previous run's measure, for the report's comparison. Not recorded.
    closure_previous: ClosureMeasure | None = Field(default=None, exclude=True)

    # The run-level batch-wait budget's account; None for a run
    # recorded without one (the interactive audit).
    batch_wait: BatchWaitSummary | None = None

    # The run's dollar budget and spend; None for a run recorded
    # without one (the interactive audit).
    budget: BudgetSummary | None = None
    #: Why the census ran without its contradiction probe this run, when the
    #: budget skipped it; the headline then reads "not probed".
    census_probe_skipped: str | None = None
    #: Why the utility pass credited nothing, when the budget skipped its judge
    #: (no ruling, no event).
    utility_behavioural_skipped: str | None = None

    # §7 delta report against the most recent prior CONSOLIDATION_RUN.
    previous_run_at: datetime | None = None
    deltas: dict[str, int] = Field(default_factory=dict)

    #: Measured LLM usage for the run (the providers' own ``usage`` fields,
    #: priced at list price); also written to the run record.
    llm_usage: LLMUsage | None = None

    #: Event id of the written run record.
    event_id: str | None = None

    def count(self, kind: CardKind) -> int:
        """The census count for one card class (0 when absent)."""
        return self.card_counts.get(kind.value, 0)

    @property
    def headline_contradictions(self) -> int:
        return _contradiction_headline(
            self.count(CardKind.CONTRADICTION),
            self.count(CardKind.INCONSISTENCY),
            self.contradiction_disagreements,
            self.contradiction_grouped_pairs,
        )

    @property
    def headline_stale(self) -> int:
        return (
            self.count(CardKind.STALE)
            + self.count(CardKind.RECENCY_DECAY)
            + self.count(CardKind.CONFIDENCE_DECAY)
        )

    def failed_passes(self) -> list[str]:
        """Names of the passes that failed (drives the CLI's exit 1)."""
        return [p.name for p in self.passes if p.status == "failed"]


# ---------------------------------------------------------------------------
# Run-record helpers
# ---------------------------------------------------------------------------


def _event_completed_at(event: OperatorEvent) -> datetime | None:
    """The run's ``completed_at`` (cadence-age basis), falling back to ``occurred_at``."""
    raw = (event.payload or {}).get("completed_at")
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            return _as_utc(datetime.fromisoformat(raw))
    return _as_utc(event.occurred_at)


def _event_started_at(event: OperatorEvent) -> datetime:
    """The run's ``started_at`` — the §4 delta-watermark basis (correction v1.74.1).

    The next run's delta window opens here, not at ``completed_at``: particles
    minted while the run was executing (pass 1's own output, an interleaving
    SessionEnd harvest) must land in *some* run's scope. Overlap re-probes are
    idempotent and cheap; permanent gaps are not. Falls back to ``occurred_at``
    (conservative — earlier than any payload timestamp is impossible, and the
    event row is written at completion).
    """
    raw = (event.payload or {}).get("started_at")
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            return _as_utc(datetime.fromisoformat(raw))
    return _as_utc(event.occurred_at)


def _event_successful(event: OperatorEvent) -> bool:
    """A run record with no ``failed(...)`` pass counts as successful."""
    passes = (event.payload or {}).get("passes")
    if not isinstance(passes, list):
        return True
    return not any(
        isinstance(p, dict) and str(p.get("status", "")).startswith("failed") for p in passes
    )


def _event_degraded(event: OperatorEvent) -> bool:
    """True when the run record disclosed a degraded run.

    Degraded is structural-only, or a run whose dollar budget left LLM work
    undone: either way some of its delta was not probed, so the
    next run's window must still open before it.
    """
    payload = event.payload or {}
    if payload.get("semantic_degraded"):
        return True
    budget = payload.get("budget")
    return isinstance(budget, dict) and bool(budget.get("limited"))


async def latest_run_event(
    session: AsyncSession,
    *,
    actor: str | None = None,
    successful_only: bool = False,
    exclude_degraded: bool = False,
) -> OperatorEvent | None:
    """The most recent ``CONSOLIDATION_RUN`` event passing the given filters.

    Correction (v1.74.1) — eligibility is filtered, not blanket:

    - The ``--if-due`` guard (§2) reads the last *successful* run **by this
      verb's own actor** — an interactive ``particles audit`` writes the same
      event type but runs none of the cross-session passes, so it must not
      satisfy the cadence. A *degraded* consolidation run still satisfies
      cadence — deliberate, so a key-less setup does not hot-loop.
    - The §4 watermark additionally requires ``exclude_degraded``: a
      structural-only night that advanced the watermark would silently convert
      its disclosed "not probed this run" into "never probed".
    - The delta report (§7) compares against the most recent prior run with
      no filters at all.
    """
    events = await list_events(session, event_type=OperatorEventType.CONSOLIDATION_RUN, limit=100)
    for event in events:
        if actor is not None and event.actor != actor:
            continue
        if successful_only and not _event_successful(event):
            continue
        if exclude_degraded and _event_degraded(event):
            continue
        return event
    return None


def _census_ran(event: OperatorEvent) -> bool:
    """Whether a run record's census pass ran (not skipped, not failed)."""
    passes = (event.payload or {}).get("passes")
    if not isinstance(passes, list):
        return False
    return any(
        isinstance(p, dict) and p.get("name") == "census" and p.get("status") == "ran"
        for p in passes
    )


async def latest_census_event(session: AsyncSession, *, actor: str) -> OperatorEvent | None:
    """The run record of the last census this verb ran on the store.

    Read the way the §4 watermark is read: this actor's runs only, so an
    interactive ``particles audit`` neither satisfies the cadence nor moves the
    census window, and never a run whose census did not probe in full
    (:func:`_census_probed`). Unlike the watermark, a run where another pass failed
    still counts: its census ran and its probes were paid for.
    """
    events = await list_events(session, event_type=OperatorEventType.CONSOLIDATION_RUN, limit=100)
    for event in events:
        if event.actor != actor or not _census_probed(event):
            continue
        if _census_ran(event):
            return event
    return None


def _census_probed(event: OperatorEvent) -> bool:
    """Whether a run record's census could have probed in full.

    Not on a structural-only run, nor when the dollar budget skipped the
    census's probe or declined batch requests. A budget that only
    skipped another pass leaves the census's own probing whole, so that run
    still satisfies the cadence.
    """
    payload = event.payload or {}
    if payload.get("semantic_degraded"):
        return False
    budget = payload.get("budget")
    if not isinstance(budget, dict):
        return True
    skipped = budget.get("skipped_passes") or []
    return "census (contradiction probe)" not in skipped and not budget.get("skipped_requests")


async def latest_measured_run_event(session: AsyncSession) -> OperatorEvent | None:
    """The most recent run record carrying census counts: the §7 delta basis.

    A night that skipped the census records none, so the next census compares
    against the last one that measured rather than against an empty night.
    """
    events = await list_events(session, event_type=OperatorEventType.CONSOLIDATION_RUN, limit=100)
    for event in events:
        if _headline_from_payload(event.payload or {}) is not None:
            return event
    return None


def build_run_payload(
    *,
    store: str,
    actor: str,
    scope: str,
    watermark: datetime | None,
    started_at: datetime,
    completed_at: datetime,
    semantic_degraded: bool,
    semantic_degraded_reason: str | None,
    providers: dict[str, str],
    passes: list[ConsolidationPass],
    census: dict[str, Any],
    llm_usage: LLMUsage | None = None,
    batch_wait: BatchWaitSummary | None = None,
    budget: BudgetSummary | None = None,
    closure: ClosureMeasure | None = None,
) -> dict[str, Any]:
    """The versioned ``CONSOLIDATION_RUN`` payload (``format: 1``).

    Shared verbatim by the scheduled cycle and the interactive audit's
    recording (:func:`record_audit_run`) so the delta chain reads one shape.
    ``llm_usage`` is additive (no format bump): the run's measured token
    totals per purpose and model plus the list-price total, so a cost history
    reads from the event log rather than a billing export. ``None`` (a run
    recorded without a usage scope) omits the key. ``batch_wait`` is additive
    in the same way: the run's batch-wait budget account, and a
    per-pass ``batch_wait`` clause on each pass the budget changed. ``budget``
    is additive too: ``consolidation.budget_usd`` and the run's
    spend, the passes it skipped, and ``limited``, which keeps a run that left
    LLM work undone from advancing the delta watermark. ``closure`` is additive
    too: the run's contested fraction and transition split, omitted
    for a run that did not measure.
    """
    payload: dict[str, Any] = {
        "format": RUN_PAYLOAD_FORMAT,
        "store": store,
        "actor": actor,
        "scope": scope,
        "watermark": watermark.isoformat() if watermark is not None else None,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "semantic_degraded": semantic_degraded,
        "semantic_degraded_reason": semantic_degraded_reason,
        "providers": providers,
        "passes": [
            {
                "name": p.name,
                "status": p.payload_status(),
                "duration_seconds": round(p.duration_seconds, 3),
                "llm_calls": p.llm_calls,
                **({"batch_wait": p.batch_wait} if p.batch_wait else {}),
            }
            for p in passes
        ],
        "census": census,
    }
    if llm_usage is not None:
        payload["llm_usage"] = llm_usage.model_dump(mode="json")
    if batch_wait is not None:
        payload["batch_wait"] = batch_wait.model_dump(mode="json")
    if budget is not None:
        payload["budget"] = {**budget.model_dump(mode="json"), "limited": budget.limited}
    if closure is not None:
        payload[CLOSURE_PAYLOAD_KEY] = closure.payload()
    return payload


def _current_providers() -> dict[str, str]:
    """provider:model per cycle purpose — a config read, no client construction."""
    llm_cfg = get_config().llm
    out: dict[str, str] = {}
    for purpose in _RUN_PURPOSES:
        selection = llm_cfg.for_purpose(purpose)
        out[purpose] = f"{selection.provider}:{selection.model}"
    return out


def _semantic_availability(structural_only: bool) -> tuple[bool, str | None]:
    """(available, degradation reason) for the cycle's LLM passes (§6).

    Degraded when the operator asked (``--structural-only``), when
    ``consolidation.semantic`` is off (the §11 demotion switch), when the
    breaker is open, or when an Anthropic-routed purpose has no key.
    A purpose routed to the local provider needs no Anthropic key.
    """
    if structural_only:
        return False, "--structural-only"
    cfg = get_config()
    if not cfg.consolidation.semantic:
        return False, "consolidation.semantic is false"
    if llm_circuit_open():
        return False, "LLM unavailable (circuit breaker open)"
    needs_key = any(cfg.llm.for_purpose(p).provider == "anthropic" for p in _RUN_PURPOSES)
    if needs_key:
        from particles.secrets import get_anthropic_api_key_optional

        if get_anthropic_api_key_optional() is None:
            return False, "no API key (structural-only run)"
    return True, None


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


async def run_consolidation(
    session: AsyncSession,
    *,
    store: str = "default",
    scope: Literal["delta", "store"] = "delta",
    structural_only: bool = False,
    if_due: bool = False,
    actor: str = CONSOLIDATION_ACTOR,
    projection_runner: ProjectionRunner | None = None,
    projection_skip_reason: str | None = None,
    lock_path: Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> ConsolidationReport:
    """Run the fixed consolidation pass list and write the run record.

    Composes existing operations only; the ordering is
    load-bearing — reconcile before census so the census sees the demoted
    state, everything before the projection render so ``MEMORY.md`` reflects
    this run's work. Pass failures are caught, recorded, and the cycle
    continues (§8); the run record is written even for a partially-failed run.
    Commits are per-pass (a later failure must not roll back completed work);
    the final run-record commit happens here too, so a cron run persists its
    record regardless of the caller.

    ``projection_runner`` is the Surface-injected harvest-then-render tail
    (the SessionEnd cycle's, reused) — the Engine cannot import the CLI-side
    projection helpers without inverting the layer contract. ``None`` records
    pass 6 as skipped with ``projection_skip_reason``.

    ``lock_path`` overrides the cycle lock's location. The lock exists to stop
    two cycles running against **one store** (§8), so its natural scope is the
    store — the default global path is merely the right answer when there is
    one. A caller holding many independent stores at once (the
    benchmark's per-question scratch stores) passes a per-store path; sharing
    the global one would make every concurrent store after the first record
    ``skipped``, silently turning a consolidation-on arm into a
    consolidation-off arm.

    ``on_progress`` receives the cycle's progress events (see
    :class:`PassEnded` and the phase list above it). It never changes what the
    cycle does or reports.
    """
    cfg = get_config().consolidation
    report = ConsolidationReport(store=store, actor=actor, scope=scope, effective_scope=scope)

    # ---------------------------------------------------------- pass 0: gate
    # §2 cadence — only this verb's own successful runs count (an interactive
    # audit writes the same event type but runs none of the cross-session
    # passes). A degraded run still satisfies cadence: deliberate, so a
    # key-less setup retries next interval instead of hot-looping.
    if if_due:
        last_for_cadence = await latest_run_event(session, actor=actor, successful_only=True)
        last_completed = (
            _event_completed_at(last_for_cadence) if last_for_cadence is not None else None
        )
        if not is_due(last_completed, _utcnow(), cfg.min_interval_hours):
            report.outcome = "skipped"
            report.skip_reason = (
                f"not due: last successful run {last_completed:%Y-%m-%d %H:%M} UTC is "
                f"younger than consolidation.min_interval_hours ({cfg.min_interval_hours}h)"
            )
            return report

    lock = acquire_cycle_lock(
        lock_path if lock_path is not None else cycle_lock_path(),
        timeout_minutes=cfg.lock_timeout_minutes,
        heartbeat_seconds=cfg.lock_heartbeat_seconds,
        heartbeat_stale_minutes=cfg.lock_heartbeat_stale_minutes,
    )
    if isinstance(lock, LockHeld):
        report.outcome = "skipped"
        report.skip_reason = lock.reason
        report.lock_warning = lock.warning
        return report
    lock_token = _active_lock.set(lock)
    progress = _RunProgress(on_progress) if on_progress is not None else None
    progress_token = _active_progress.set(progress)

    # Count every completion the cycle pays for, per purpose and model, for the
    # report's usage line and the run record.
    wait_cfg = cfg.batch_wait
    budget = BatchWaitBudget(
        budget_seconds=wait_cfg.budget_seconds,
        min_remaining_seconds=wait_cfg.min_remaining_seconds,
    )
    if progress is not None:
        budget.on_wait = _batch_wait_publisher(progress)
    # the dollar budget reads the spend off this run's usage scope,
    # so it is opened inside it; with no budget configured nothing is installed.
    report.budget = BudgetSummary(budget_usd=cfg.budget_usd)
    with (
        track_usage() as usage,
        batch_wait_budget(budget),
        spend_budget(_run_spend_budget(cfg.budget_usd, usage)) as spend,
    ):
        try:
            semantic_ok, degrade_reason = _semantic_availability(structural_only)
            report.semantic_degraded = not semantic_ok
            report.semantic_degraded_reason = degrade_reason
            report.providers = _current_providers()

            # §4 watermark basis (correction v1.74.1): the previous
            # watermark-eligible run — same actor, successful, NOT degraded (a
            # structural-only night must not convert its disclosed "not probed
            # this run" into "never probed") — and its *started_at*, so nothing
            # written mid-cycle ever falls between two runs' windows.
            watermark_event = await latest_run_event(
                session, actor=actor, successful_only=True, exclude_degraded=True
            )
            prior_started = (
                _event_started_at(watermark_event) if watermark_event is not None else None
            )

            # ---------------------------------------- pass 0.5: local refresh
            # Placed BEFORE extract so a rule file edited today is
            # re-snapshotted, extracted by pass 1, and reconciled by pass 2 in the
            # SAME run — anywhere later and a change would take three nights to
            # reach the projection. Zero-LLM, so it is not gated on ``semantic_ok``:
            # a degraded night still notices that the rules changed.
            if not get_config().local_refresh.enabled:
                _skip(report, "refresh", "local_refresh.enabled is false")
            else:
                await _run_pass(session, report, "refresh", lambda: _pass_refresh(session, report))

            # ------------------------------------------------ pass 1: extract
            if not get_config().consolidation.extract_pending:
                _skip(report, "extract", "consolidation.extract_pending is false")
            elif not semantic_ok:
                # §6: extraction is LLM-priced; the backlog is disclosed instead.
                _skip(report, "extract", f"extraction is LLM-priced ({degrade_reason})")
                with contextlib.suppress(Exception):
                    await _count_backlog(session, report)
            else:
                await _run_pass(session, report, "extract", lambda: _pass_extract(session, report))

            # §4 delta scope — computed AFTER pass 1 (correction v1.74.1), so the
            # particles pass 1 just minted (asserted_at > the previous run's
            # started_at) are in THIS run's census scope rather than in no run's
            # scope ever. Particles from corpus entries deposited since the
            # watermark are folded in. Store-wide on the first run or --scope
            # store.
            scope_ids: frozenset[str] | None = None
            if scope == "delta":
                if prior_started is None:
                    report.effective_scope = "store"  # first run: nothing to delta against
                else:
                    report.watermark = prior_started
                    scope_ids = await _delta_scope_ids(session, prior_started)
                    report.scope_particle_count = len(scope_ids)

            # ---------------------------------------------- pass 2: reconcile
            if not semantic_ok:
                # §6 (correction v1.74.1): the sweep makes one replacement-signal
                # probe per candidate pair — LLM-priced, so a degraded run skips
                # it with a disclosure rather than fail-opening every probe to
                # "keep both" behind a clean-looking bill.
                _skip(
                    report,
                    "reconcile",
                    f"replacement-signal probes are LLM-priced ({degrade_reason})",
                )
            else:
                await _run_pass(
                    session, report, "reconcile", lambda: _pass_reconcile(session, report)
                )

            # ------------------------------------- pass 2b: update supersession
            if not semantic_ok:
                _skip(
                    report,
                    "reconcile_updates",
                    f"update-supersession probes are LLM-priced ({degrade_reason})",
                )
            elif not get_config().reconciliation.update_supersession.enabled:
                _skip(
                    report,
                    "reconcile_updates",
                    "reconciliation.update_supersession.enabled is false",
                )
            else:
                await _run_pass(
                    session,
                    report,
                    "reconcile_updates",
                    lambda: _pass_reconcile_updates(session, report, scope_ids),
                )

            # ------------------------------ pass 2c: re-anchor
            # After both sweeps, so every update retirement of the night exists,
            # and before the census, which then sees the restatements.
            if not semantic_ok:
                _skip(
                    report,
                    "reanchor",
                    f"re-anchor probes are LLM-priced ({degrade_reason})",
                )
            elif not get_config().consolidation.reanchor.enabled:
                _skip(report, "reanchor", "consolidation.reanchor.enabled is false")
            else:
                await _run_pass(
                    session,
                    report,
                    "reanchor",
                    lambda: _pass_reanchor(session, report, actor=actor),
                )

            # ------------------------------------------------- pass 3: census
            # the census has its own cadence. Gather the last census's
            # run record, decide (pure), then run it or disclose the skip.
            # a census whose contradiction probe would pass the
            # budget still runs, structurally (its other finders cost nothing),
            # exactly as a degraded night's census does; 3b and 4 follow suit.
            census_semantic = semantic_ok
            census_degrade_reason = degrade_reason
            cards: list[CurationCard] = []
            census_event = await latest_census_event(session, actor=actor)
            census = decide_census(
                enabled=cfg.census.enabled,
                interval_hours=cfg.census.interval_hours,
                last_ran=_event_started_at(census_event) if census_event is not None else None,
                now=_utcnow(),
                store_wide=scope == "store",
            )
            report.census_last_ran_at = census.last_ran
            report.census_next_due_at = census.next_due
            census_scope_ids: frozenset[str] | None = None
            if census.run:
                census_scope_ids = await _census_scope(session, report, census, scope, scope_ids)

                if semantic_ok:
                    probe_skip = _budget_skip_reason(
                        "census", _census_cap_estimate(), what="contradiction probe"
                    )
                    if probe_skip is not None:
                        census_semantic = False
                        census_degrade_reason = probe_skip
                        report.census_probe_skipped = probe_skip
                        _note_budget_skip(report, "census (contradiction probe)")

                async def _census() -> int:
                    nonlocal cards
                    cards = await _pass_census(
                        session, report, semantic=census_semantic, scope_ids=census_scope_ids
                    )
                    return report.contradiction_probes_run + report.contradiction_verifications_run

                await _run_pass(session, report, "census", _census)
            else:
                report.census_skipped = census.reason
                if census_event is not None:
                    report.census_last_headline = _headline_from_payload(census_event.payload or {})
                _skip(report, "census", census.reason or "census skipped")
            census_entry = report.passes[-1]
            census_ran = census_entry.status == "ran"

            # --------------------------------- pass 3b: disclose
            # Zero-LLM. The lapse sweep runs on every night that reaches it,
            # degraded and census-failed nights included; only minting needs
            # this run's confirmed pairs (or a waiting list from the last run).
            covered_after: frozenset[frozenset[str]] = frozenset()

            async def _disclose() -> int:
                nonlocal cards, covered_after
                cards, covered_after = await _pass_disclose(
                    session,
                    report,
                    cards,
                    census_ran=census_ran,
                    semantic=census_semantic,
                    degrade_reason=census_degrade_reason,
                    actor=actor,
                    scope_ids=scope_ids,
                )
                return 0

            await _run_pass(session, report, "disclose", _disclose)

            # ----------------------------------------- pass 4: curation queue
            # The census, not whatever pass ran last, decides whether there is
            # a collection to persist: 3b's own failure does not
            # stop pass 4.
            if census_entry.status == "failed":
                _skip(report, "curation", "census failed — no card collection to rank")
            elif report.census_skipped is not None:
                # no census tonight, so serve the last one's stored
                # collection through the live session half (suppression, status
                # re-validation, post-build resolutions) rather than rank an
                # empty one. Never a live collection: that would pay for the
                # census the cadence just skipped.
                if not get_config().curation.snapshot_enabled:
                    _skip(
                        report,
                        "curation",
                        "the census did not run and curation.snapshot_enabled is false, "
                        "so no stored collection is served",
                    )
                elif await latest_snapshot(session) is None:
                    _skip(
                        report,
                        "curation",
                        "the census did not run and no card collection is stored yet",
                    )
                else:
                    await _run_pass(
                        session, report, "curation", lambda: _pass_curation_stored(session, report)
                    )
            else:
                # the snapshot records the scope its finders ran under,
                # so tonight's delta-scoped contradiction set unions with (rather
                # than erases) the store-wide picture earlier runs built.
                collection_scope = (
                    CollectionScope.STORE if census_scope_ids is None else CollectionScope.DELTA
                )
                await _run_pass(
                    session,
                    report,
                    "curation",
                    lambda: _pass_curation(
                        session,
                        report,
                        cards,
                        scope=collection_scope,
                        semantic=census_semantic,
                        covered=covered_after,
                    ),
                )

            # ------------------------------------------------ pass 5: utility
            if not get_config().utility.mining.enabled:
                _skip(report, "utility", "utility.mining.enabled is false")
            else:
                await _run_pass(
                    session,
                    report,
                    "utility",
                    lambda: _pass_utility(
                        session,
                        report,
                        watermark=report.watermark,
                        behavioural=semantic_ok,
                    ),
                )

            # ------------------------------------- pass 5b: abstraction
            # Between utility mining (whose signals inform future eligibility) and
            # the projection render (so the projection reflects this run's
            # promotions), ahead of the reserved retention slot.
            if not get_config().consolidation.abstraction.enabled:
                _skip(report, "abstraction", "consolidation.abstraction.enabled is false")
            elif not semantic_ok:
                _skip(
                    report,
                    "abstraction",
                    f"synthesis + judges are LLM-priced ({degrade_reason})",
                )
            elif (
                abstraction_skip := _budget_skip_reason("abstraction", _abstraction_cap_estimate())
            ) is not None:
                _skip(report, "abstraction", abstraction_skip)
                _note_budget_skip(report, "abstraction")
            else:
                await _run_pass(
                    session,
                    report,
                    "abstraction",
                    lambda: _pass_abstraction(session, report, scope_ids=scope_ids),
                )

            # --------------------------------------------- pass 6: projection
            if projection_runner is None:
                _skip(
                    report,
                    "projection",
                    projection_skip_reason or "projection disabled or no memory directory",
                )
            else:
                runner: ProjectionRunner = projection_runner
                await _run_pass(
                    session, report, "projection", lambda: _pass_projection(report, runner)
                )

            # ---------------------------------------- pass 6b: measure
            # Zero-LLM and read-only, so it runs on every night, degraded ones
            # included. Last, so it reads the store this run leaves behind.
            await _run_pass(
                session, report, "measure", lambda: _pass_measure(session, report, actor=actor)
            )

            # ---------------------------------------- pass 7: record + report
            report.completed_at = _utcnow()
            report.llm_usage = usage.snapshot()
            report.batch_wait = BatchWaitSummary.from_budget(budget)
            report.budget.spent_usd = report.llm_usage.priced_cost_usd
            if spend is not None:
                report.budget.skipped_chunks = spend.skipped_chunks
                report.budget.skipped_requests = spend.skipped_requests
                report.budget.exhausted_in_pass = spend.exhausted_in_pass
            prior = await latest_run_event(session)
            if prior is not None:
                report.previous_run_at = _event_completed_at(prior)
            # a skipped census measured nothing, so it reports no
            # deltas, and a census compares with the last run that measured.
            if report.census_skipped is None:
                measured = await latest_measured_run_event(session)
                if measured is not None:
                    report.deltas = _compute_deltas(report, measured)
            payload = build_run_payload(
                store=store,
                actor=actor,
                scope=report.effective_scope,
                watermark=report.watermark,
                started_at=report.started_at,
                completed_at=report.completed_at,
                semantic_degraded=report.semantic_degraded,
                semantic_degraded_reason=report.semantic_degraded_reason,
                providers=report.providers,
                passes=report.passes,
                census=_census_payload(report),
                llm_usage=report.llm_usage,
                batch_wait=report.batch_wait,
                budget=report.budget,
                closure=report.closure,
            )
            async with write_transaction(session):
                event = await record_event(
                    session,
                    actor=actor,
                    event_type=OperatorEventType.CONSOLIDATION_RUN,
                    payload=payload,
                )
            report.event_id = event.event_id
        finally:
            _active_progress.reset(progress_token)
            _active_lock.reset(lock_token)
            release_cycle_lock(lock)
    return report


# ---------------------------------------------------------------------------
# Pass bodies — each an existing verb's operation, composed (§3)
# ---------------------------------------------------------------------------


class _BudgetSkip(Exception):  # noqa: N818 — a control-flow signal, not an error
    """Raised by a pass body whose gathered work would pass the run's budget.

    :func:`_run_pass` records the pass as skipped with the message, the way a
    probe cap is disclosed. The body raises it after its gather and
    before any LLM call, so nothing it would have paid for has been sent.
    """


#: LLM calls one abstraction promotion makes: the synthesis plus the
#: entailment and duplicate judges. The budget's cap-bound estimate.
_ABSTRACTION_CALLS_PER_PROMOTION = 3


def _run_spend_budget(budget_usd: float | None, usage: UsageAccumulator) -> SpendBudget | None:
    """The run's dollar budget over its usage scope, or ``None`` when unbounded."""
    if budget_usd is None:
        return None
    return SpendBudget(
        budget_usd=budget_usd,
        spent=lambda: usage.snapshot().priced_cost_usd,
        estimate_chunk=estimate_batch_chunk,
    )


def _budget_skip_reason(
    name: str, estimate: PassEstimate, *, what: str | None = None
) -> str | None:
    """The disclosure for a pass the budget skips, or ``None`` when it runs.

    Gathers the run's spend so far from the open usage scope, asks the pure
    :func:`~particles.core.spend_budget.decide_spend`, and renders its "skip"
    as "spent US$X of US$Y; pass N skipped". No budget in scope runs everything.
    """
    spend = current_spend_budget()
    if spend is None or estimate.calls == 0:
        # No budget, or nothing to send: a pass with no work is never skipped.
        return None
    spent = spend.spent()
    verdict = decide_spend(spent_usd=spent, budget_usd=spend.budget_usd, estimate_usd=estimate.usd)
    if verdict == "run":
        return None
    position = _pass_position(name)
    target = f"pass {position}'s {what}" if what else f"pass {position}"
    reason = f"spent US{format_usd(spent)} of US{format_usd(spend.budget_usd)}; {target} skipped"
    if estimate.usd is not None and spent < spend.budget_usd:
        reason += (
            f" (estimated US{format_usd(estimate.usd)} for "
            f"{_plural(estimate.calls, 'LLM call')} at list price)"
        )
    return reason + " (consolidation.budget_usd)"


def _note_budget_skip(report: ConsolidationReport, label: str) -> None:
    if report.budget is not None:
        report.budget.skipped_passes.append(label)


def _census_cap_estimate() -> PassEstimate:
    """The census probe at its caps: every probe and every second reading spent.

    Cap-bound rather than gathered, because the census's gather is the whole
    lint pass; the caps are small, so the bound is close. ``--dry-run``
    gathers the real candidate count.
    """
    audit = get_config().audit
    probes = audit.max_contradiction_probes
    verifications = (
        min(audit.max_contradiction_verifications, probes) if audit.verify_contradictions else 0
    )
    return probe_estimate(probes) + context_call_estimate("verification", verifications)


def _abstraction_cap_estimate() -> PassEstimate:
    """Abstraction at its cap: every promotion slot spent."""
    cap = get_config().consolidation.abstraction.max_promotions_per_run
    return context_call_estimate("abstraction", cap * _ABSTRACTION_CALLS_PER_PROMOTION)


def _skip(report: ConsolidationReport, name: str, reason: str) -> None:
    report.passes.append(ConsolidationPass(name=name, status="skipped", detail=reason))
    progress = _active_progress.get()
    if progress is not None:
        progress.send(
            PassEnded(
                phase="pass_end",
                done=_pass_position(name),
                total=len(PASS_ORDER),
                label=name,
                outcome="skipped",
                summary=reason,
            )
        )


async def _run_pass(
    session: AsyncSession,
    report: ConsolidationReport,
    name: str,
    body: Callable[[], Awaitable[int]],
) -> None:
    """Run one pass under the §8 continue-and-report contract.

    ``body`` returns the pass's LLM call count. A failure is caught, the
    session rolled back, and the pass recorded as ``failed(<error>)`` — the
    cycle continues, so the zero-LLM tail (projection) still runs on a flaky
    night and the run record is written regardless.
    """
    start = time.monotonic()
    held = _active_lock.get()
    if held is not None:
        # Descriptive only: a failed write never fails the pass.
        try:
            held.set_pass(name)
        except OSError as exc:
            log.warning("consolidation: could not record pass %s on the lock: %s", name, exc)
    entry = ConsolidationPass(name=name, status="ran")
    report.passes.append(entry)
    progress = _active_progress.get()
    if progress is not None:
        progress.send(
            ProgressEvent(
                phase="pass", done=_pass_position(name), total=len(PASS_ORDER), label=name
            )
        )
    budget = current_batch_wait_budget()
    before = (budget.counters(), budget.cancellations()) if budget is not None else None
    if budget is not None:
        budget.pass_name = name
    spend = current_spend_budget()
    if spend is not None:
        spend.pass_name = name
    try:
        entry.llm_calls = await body()
    except _BudgetSkip as skipped:
        # the body's gather found more than the budget has left.
        entry.status = "skipped"
        entry.detail = str(skipped)
        _note_budget_skip(report, name)
    except Exception as exc:  # noqa: BLE001 — §8: continue and report
        with contextlib.suppress(Exception):
            await session.rollback()
        entry.status = "failed"
        entry.detail = f"{type(exc).__name__}: {exc}"
        log.warning("consolidation: pass %s failed: %s", name, exc)
    finally:
        entry.duration_seconds = time.monotonic() - start
        if budget is not None and before is not None:
            clause = _batch_wait_clause(
                before[0], budget.counters(), before[1], budget.cancellations()
            )
            if clause is not None:
                entry.batch_wait = clause
                entry.detail = clause if entry.detail is None else f"{entry.detail}; {clause}"
        if progress is not None:
            progress.send(
                PassEnded(
                    phase="pass_end",
                    done=_pass_position(name),
                    total=len(PASS_ORDER),
                    label=name,
                    outcome=_pass_outcome(report, entry),
                    duration_seconds=entry.duration_seconds,
                    summary=_pass_summary(report, entry, progress),
                )
            )


def _pass_outcome(report: ConsolidationReport, entry: ConsolidationPass) -> PassOutcome:
    """Whether a finished pass did all it set out to (the ``pass_end`` mark)."""
    if entry.status == "failed":
        return "failed"
    if entry.status == "skipped":
        return "skipped"
    if entry.batch_wait is not None:
        return "degraded"
    if entry.name == "extract" and (report.pending_retry or report.pending_failed):
        return "degraded"
    if entry.name == "census" and (report.semantic_degraded or report.census_probe_skipped):
        return "degraded"
    if entry.name == "utility" and report.utility_behavioural_skipped:
        return "degraded"
    return "ok"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _pass_summary(
    report: ConsolidationReport, entry: ConsolidationPass, progress: _RunProgress
) -> str:
    """One line on what a finished pass did, for its ``pass_end`` event.

    Built from the report fields the pass wrote; the full report still prints
    at the end. A failed pass reports its error.
    """
    if entry.status == "failed":
        return entry.detail or "failed"
    if entry.status == "skipped":
        return entry.detail or "skipped"
    parts: list[str]
    match entry.name:
        case "refresh":
            parts = [f"{report.refresh_checked} checked", f"{report.refresh_updated} changed"]
            if report.refresh_missing:
                parts.append(f"{report.refresh_missing} missing")
        case "extract":
            _done, attempted = progress.counters.get("extract", (0, 0))
            parts = [f"{report.pending_extracted} of {attempted} extracted"]
            if report.pending_retry:
                parts.append(f"{report.pending_retry} left for retry")
            if report.pending_partial:
                parts.append(f"{report.pending_partial} kept part of their read")
            if report.pending_waiting:
                parts.append(f"{report.pending_waiting} waiting on a partial whole read")
            if report.pending_failed:
                parts.append(f"{report.pending_failed} failed")
            if report.pending_empty:
                parts.append(f"{report.pending_empty} with no beliefs")
            if report.pending_remaining:
                parts.append(f"{report.pending_remaining} still pending")
        case "reconcile":
            parts = [
                f"demoted {report.reconcile_demoted}",
                f"probed {report.reconcile_probes_run} of "
                f"{_plural(report.reconcile_candidate_pairs, 'pair')}",
            ]
        case "reconcile_updates":
            parts = [
                f"demoted {report.update_demoted}",
                f"probed {report.update_probes_run} of "
                f"{_plural(report.update_candidate_pairs, 'pair')}",
            ]
            if report.update_fixed_slot:
                parts.append(f"{report.update_fixed_slot} fixed-slot pair(s) sent to review")
            if report.update_previously_cleared:
                parts.append(f"{report.update_previously_cleared} previously cleared")
        case "reanchor" if report.reanchor is not None:
            parts = [
                render_reanchor_line(report.reanchor).strip().removeprefix("re-anchor").strip()
            ]
        case "census":
            parts = [_plural(report.headline_contradictions, "contradiction")]
            if report.semantic_degraded or report.census_probe_skipped:
                parts[0] += " (not probed)"
            elif report.contradiction_candidate_pairs:
                parts.append(
                    f"probed {report.contradiction_probes_run} of "
                    f"{_plural(report.contradiction_candidate_pairs, 'pair')}"
                )
            if report.contradiction_previously_cleared and not report.semantic_degraded:
                parts.append(f"{report.contradiction_previously_cleared} previously cleared")
            parts.append(_plural(report.duplicate_candidate_pairs_total, "duplicate pair"))
            parts.append(f"{report.headline_stale} stale")
        case "disclose" if report.disclosure is not None:
            parts = [
                render_disclosure_line(report.disclosure)
                .strip()
                .removeprefix("contradiction disclosure:")
                .strip()
            ]
        case "curation":
            parts = [f"{_plural(report.curation_queue_total, 'card')} queued"]
        case "utility":
            events = report.utility_literal + report.utility_behavioural
            parts = [
                f"{_plural(report.utility_sessions_mined, 'session')} mined",
                f"+{_plural(events, 'utility event')}",
            ]
        case "abstraction" if report.abstraction is not None:
            parts = [
                f"{len(report.abstraction.proposed_event_ids)} proposed",
                f"{len(report.abstraction.promoted_particle_ids)} promoted",
            ]
        case "projection" if report.projection is not None:
            parts = [f"re-rendered {_plural(report.projection.get('rendered', 0), 'memory file')}"]
        case "measure" if report.closure is not None:
            parts = [line.strip() for line in render_closure_lines(report.closure, None)]
        case _:
            parts = [_plural(entry.llm_calls, "LLM call")]
    line = ", ".join(parts)
    if entry.batch_wait is not None:
        line += f" ({entry.batch_wait})"
    return line


def _batch_wait_clause(
    before: tuple[int, int, int, int],
    after: tuple[int, int, int, int],
    cancel_before: tuple[int, int, int, int],
    cancel_after: tuple[int, int, int, int],
) -> str | None:
    """What the batch-wait budget did during one pass, or None.

    A cancelled batch reports what it kept, re-ran and lost rather than
    calling its whole set unavailable.
    """
    cut, _cut_requests, seq_sets, seq_requests = (a - b for a, b in zip(after, before, strict=True))
    cancelled, kept, rerun, lost = (a - b for a, b in zip(cancel_after, cancel_before, strict=True))
    parts: list[str] = []
    if cut:
        parts.append(f"{cut} batch(es) cut short by the batch-wait budget")
    if cancelled:
        parts.append(
            f"{cancelled} batch(es) cancelled ({kept} kept, {rerun} re-run, {lost} unavailable)"
        )
    if seq_sets:
        parts.append(
            f"batch-wait budget spent: {seq_sets} set(s), "
            f"{seq_requests} request(s) sequential at full price"
        )
    return "; ".join(parts) if parts else None


async def _delta_scope_ids(session: AsyncSession, watermark: datetime) -> frozenset[str]:
    """The §4 delta scope: particles changed since ``watermark`` + particles
    from corpus entries deposited since + particles whose adjudicability
    default was rewritten since, threaded through the
    at-least-one-side-in-scope seams unchanged.

    The last term reads ``MODALITY_RECLASSIFIED`` events rather than the
    stamp's timestamp: an event is written for an operator verdict and for a
    value a regeneration run changed, never for a restamp that confirmed the
    stored value, so a confirming run queues nothing to re-pair.
    """
    from particles.corpus.store import list_entry_ids_created_since

    changed = await get_particle_ids_changed_since(session, watermark)
    entry_ids = await list_entry_ids_created_since(session, watermark)
    if entry_ids:
        changed |= await get_particle_ids_for_entries(session, entry_ids)
    changed |= await reclassified_particle_ids_since(session, watermark)
    return frozenset(changed)


async def _count_backlog(session: AsyncSession, report: ConsolidationReport) -> int:
    """Disclose the PENDING backlog without extracting (the degraded pass 1)."""
    from particles.corpus.store import list_pending_snapshots_for_catchup

    pending = await list_pending_snapshots_for_catchup(session)
    report.pending_total = len(pending)
    report.pending_remaining = len(pending)
    report.pending_oldest_at = min((p.captured_at for p in pending), default=None)
    return 0


async def _pass_refresh(session: AsyncSession, report: ConsolidationReport) -> int:
    """Pass 0.5: re-check every LAZY ``file://`` entry against the file on disk.

    The pass — the wire that makes the loop close unattended: edit
    ``AGENTS.md`` → tonight's run stats a changed mtime → the hash differs → a
    RESPONSE snapshot lands PENDING → pass 1 extracts it → the §2 generation
    cascade retires the prior generation → pass 2 sweeps cross-entry → pass 6
    re-renders the projection.

    Returns 0: the pass makes no LLM calls at all — its cost is a ``stat`` per
    entry plus a read + SHA-256 only for the files whose mtime moved.
    Per-entry failures are disclosed and never fatal (§8).
    """
    from particles.corpus.refresh_outcome import RefreshOutcome
    from particles.corpus.store import list_refreshable_local_entries
    from particles.operations.corpus_refresh import refresh_entries

    entries = await list_refreshable_local_entries(session)
    cap = get_config().local_refresh.max_entries
    report.refresh_remaining = max(0, len(entries) - cap)
    planned = min(len(entries), cap)
    _count("refresh", 0, planned, "entries")

    async for result in refresh_entries(session, (entry_id for entry_id, _ in entries[:cap])):
        report.refresh_checked += 1
        _count("refresh", report.refresh_checked, planned, "entries")
        match result.outcome:
            case None:  # §8: continue the sweep, disclose
                log.warning(
                    "consolidation: refresh failed for %s: %s", result.entry_id[:8], result.error
                )
            case RefreshOutcome.MISSING:
                report.refresh_missing += 1
            case RefreshOutcome.UNCHANGED_MTIME:
                report.refresh_unchanged_mtime += 1
            case RefreshOutcome.UNCHANGED_HASH:
                report.refresh_unchanged_hash += 1
            case RefreshOutcome.CHANGED:
                report.refresh_updated += 1
    return 0


async def _pass_extract(session: AsyncSession, report: ConsolidationReport) -> int:
    """Pass 1: extract PENDING snapshots, least-tried then oldest, capped per run (§3.1).

    Level-triggered like the harvest — the corpus is the state; a
    capped run discloses the remainder and the next run continues. Per-snapshot
    failures are disclosed, never fatal (mirroring the audit's extract loop).

    Under ``consolidation.extract_batching`` (default on) the capped
    set runs as concurrent per-snapshot tasks whose LLM requests merge into
    one pooled batch; ``false`` restores this serial loop exactly.
    The pooled cap counts distinct corpus entries, so a transcript with six
    pending snapshots takes one slot of the cap rather than six.
    """
    # Deferred import: the pipeline pulls the extractor registry / LLM stack;
    # load it only when there is something to extract (AGENTS.md case 2).
    from particles.corpus.store import (
        list_pending_snapshots_for_catchup,
        reset_stale_in_progress,
    )
    from particles.operations.extract import collapse_superseded_pending, extract_snapshot

    # A run killed mid-extraction leaves its claims IN_PROGRESS, where no
    # PENDING-filtered listing sees them again. ``extract --all-pending`` has
    # reset them since 0.42.2; the nightly pass never did, so a claim stranded
    # in July was still stranded in September.
    stale_minutes = get_config().extraction.stale_in_progress_minutes
    stale = await reset_stale_in_progress(
        session, older_than=_utcnow() - timedelta(minutes=stale_minutes)
    )
    await session.commit()
    report.pending_reset_stale = len(stale)
    if stale:
        log.warning(
            "consolidation: reset %d stale IN_PROGRESS snapshot(s) to PENDING "
            "(claimed more than %g min ago)",
            len(stale),
            stale_minutes,
        )

    # collapse before listing. Oldest-first plus one snapshot per
    # entry per pooled run otherwise drains a 35-edit MEMORY.md over 35 nights,
    # every one of them extracting a generation the file no longer holds.
    report.pending_collapsed = (await collapse_superseded_pending(session)).snapshots

    pending = await list_pending_snapshots_for_catchup(session)
    report.pending_total = len(pending)
    batching = get_config().consolidation.extract_batching
    chosen = _extract_selection(pending, batching=batching)
    if current_spend_budget() is not None:
        # price the chosen snapshots before the first call.
        sizes = await _snapshot_sizes(session, [snapshot_id for _, snapshot_id in chosen])
        reason = _budget_skip_reason("extract", extraction_estimate(sizes))
        if reason is not None:
            report.pending_remaining = report.pending_total
            report.pending_oldest_at = await _oldest_pending_at(session)
            await session.commit()
            raise _BudgetSkip(reason)
    if batching:
        # The pooled tasks open sessions of their own; end this one's read so
        # it does not pin a connection for the whole pass.
        await session.commit()
        extracted = await _pass_extract_pooled(chosen, report)
    else:
        cap = get_config().consolidation.max_pending_entries
        extracted = await _pass_extract_serial(session, pending[:cap], report, extract_snapshot)
    report.pending_oldest_at = await _oldest_pending_at(session)
    return extracted


def _extract_selection(
    pending: Sequence[PendingSnapshot], *, batching: bool
) -> list[tuple[str, str]]:
    """``(entry_id, snapshot_id)`` pass 1 would extract from the catch-up queue.

    The pooled pass takes one snapshot per entry up to the cap;
    the serial loop takes the first ``cap`` snapshots as listed.
    """
    cap = get_config().consolidation.max_pending_entries
    if batching:
        return _one_per_entry(pending, cap)
    return [(item.entry_id, item.snapshot_id) for item in pending[:cap]]


async def _snapshot_sizes(session: AsyncSession, snapshot_ids: Sequence[str]) -> list[int]:
    """Source size in bytes of each snapshot's blob, for an extraction estimate.

    Bytes stand in for characters: an HTML or PDF source reads larger than the
    text extraction sees, so the estimate errs high. A missing blob counts 0.
    """
    from particles.corpus.deposit import blob_path
    from particles.corpus.store import get_snapshot_content_hashes

    hashes = await get_snapshot_content_hashes(session, snapshot_ids)
    sizes: list[int] = []
    for snapshot_id in snapshot_ids:
        content_hash = hashes.get(snapshot_id)
        try:
            sizes.append(blob_path(content_hash).stat().st_size if content_hash else 0)
        except (OSError, ValueError):
            sizes.append(0)
    return sizes


def _one_per_entry(pending: Sequence[PendingSnapshot], cap: int) -> list[tuple[str, str]]:
    """The first ``cap`` distinct entries of the catch-up queue, one snapshot each.

    A pooled run takes at most one snapshot per entry. Applying
    that rule after slicing the queue to ``cap`` spent a slot on every later
    snapshot of an entry already chosen: on the owner's store twenty slots
    bought thirteen extractions a night. Choosing entries first fills the cap.
    """
    seen: set[str] = set()
    chosen: list[tuple[str, str]] = []
    for item in pending:
        if len(chosen) >= cap:
            break
        if item.entry_id in seen:
            continue
        seen.add(item.entry_id)
        chosen.append((item.entry_id, item.snapshot_id))
    return chosen


async def _oldest_pending_at(session: AsyncSession) -> datetime | None:
    from particles.corpus.store import list_pending_snapshots_for_catchup

    pending = await list_pending_snapshots_for_catchup(session)
    return min((p.captured_at for p in pending), default=None)


@dataclass
class _SnapshotRun:
    """What one catch-up extraction wrote, read back to classify it."""

    outcome: SnapshotOutcome
    carried: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)


def _record_extraction(
    report: ConsolidationReport,
    entry_id: str,
    snapshot_id: str,
    written: Sequence[object],
    run: _SnapshotRun,
) -> None:
    """Count one returned extraction as extracted, empty, retry, or skipped.

    ``extract_snapshot`` returns normally in four different situations, and an
    empty list reads the same in each. Before this, the pass counted every
    normal return as extracted: a snapshot whose LLM calls all failed, handed
    back PENDING with nothing written, was reported as done, night after night.
    """
    outcome = run.outcome
    if outcome.skipped is not None:
        if outcome.skipped == "superseded":
            report.pending_superseded_late += 1
        elif outcome.skipped == "waiting":
            report.pending_waiting += 1
            holder = (outcome.waiting_on or "")[:8]
            if outcome.waiting_on_failed:
                log.warning(
                    "consolidation: %s/%s waits on %s, a FAILED snapshot holding a partial "
                    "whole read (its blob is missing): restore the blob, or run "
                    "`particles reindex %s` to retire its claims and release the wait",
                    entry_id[:8],
                    snapshot_id[:8],
                    holder,
                    entry_id[:8],
                )
            else:
                log.warning(
                    "consolidation: %s/%s waits on %s, which holds a partial whole read; "
                    "it is read once that snapshot completes (`particles reindex %s` "
                    "releases it by hand)",
                    entry_id[:8],
                    snapshot_id[:8],
                    holder,
                    entry_id[:8],
                )
        return
    if outcome.failed_calls and outcome.kept_calls:
        report.pending_retry += 1
        report.pending_partial += 1
        log.warning(
            "consolidation: %s/%s left PENDING: %d LLM call(s) produced nothing usable; "
            "kept %d answered call(s), and the retry sends %d other answered call(s) again",
            entry_id[:8],
            snapshot_id[:8],
            outcome.failed_calls,
            outcome.kept_calls,
            outcome.rebilled_calls,
        )
        return
    if outcome.failed_calls:
        report.pending_retry += 1
        log.warning(
            "consolidation: %s/%s left PENDING: %d LLM call(s) produced nothing usable; "
            "retried on a later run, behind snapshots not yet tried, which sends its "
            "%d answered call(s) again",
            entry_id[:8],
            snapshot_id[:8],
            outcome.failed_calls,
            outcome.rebilled_calls,
        )
        return
    report.pending_extracted += 1
    if not written and not run.carried and not run.suppressed:
        report.pending_empty += 1
        log.warning(
            "consolidation: %s/%s completed with no beliefs (nothing extracted, carried "
            "forward or matched); check the source with `particles corpus cat`",
            entry_id[:8],
            snapshot_id[:8],
        )


def _finish_pending_counts(report: ConsolidationReport) -> None:
    # A snapshot superseded after listing is no longer owed; everything else
    # not extracted is still PENDING.
    report.pending_remaining = (
        report.pending_total - report.pending_extracted - report.pending_superseded_late
    )


async def _pass_extract_serial(
    session: AsyncSession,
    batch: Sequence[PendingSnapshot],
    report: ConsolidationReport,
    extract_snapshot: Callable[..., Awaitable[list[Particle]]],
) -> int:
    """The serial extract loop (``consolidation.extract_batching: false``)."""
    from particles.ingest.pipeline import SnapshotOutcome

    _count("extract", 0, len(batch), "snapshots")
    for done, item in enumerate(batch, start=1):
        entry_id, snapshot_id = item.entry_id, item.snapshot_id
        run = _SnapshotRun(SnapshotOutcome())
        try:
            written = await extract_snapshot(
                session,
                entry_id,
                snapshot_id,
                agent_id=report.actor,
                skip_if_superseded=True,
                carry_forward_ids_out=run.carried,
                suppressed_ids_out=run.suppressed,
                outcome_out=run.outcome,
            )
            await session.commit()
            _record_extraction(report, entry_id, snapshot_id, written, run)
        except AccountLevelLLMError as exc:
            # §8 says continue-and-disclose on a per-snapshot failure, but an
            # account-level failure is not per-snapshot: every remaining
            # extraction in the cap would fail identically. Stop the pass and
            # disclose once — the untried snapshots are still PENDING, so the
            # next night (or the next `extract --all-pending`) resumes.
            await session.rollback()
            report.pending_failed += 1
            log.error("consolidation: extraction unavailable (account-level): %s", exc)
            break
        except Exception as exc:  # noqa: BLE001 — §8: continue the batch, disclose
            await session.rollback()
            report.pending_failed += 1
            log.warning(
                "consolidation: extraction failed for %s/%s: %s",
                entry_id[:8],
                snapshot_id[:8],
                exc,
            )
        _count("extract", done, len(batch), "snapshots")
    _finish_pending_counts(report)
    # A lower bound on the spend (a chunked source makes one call per chunk);
    # token telemetry lives with the OTel spans.
    return report.pending_extracted


class _StoreSlot:
    """One pooled extraction task's turn at the store (``extract_db_concurrency``).

    Every snapshot task in a pooled pass joins one batch, so a pass can run
    twenty of them at once, and a SQLite engine pools fifteen connections. The
    slot bounds the tasks that touch the store at the same time, and it is
    handed back, with the session's connection, for as long as the task waits
    on the pool (:meth:`idle`): gather, then wait holding nothing, then take a
    slot again to apply. Holding the connection across the batch wait (minutes
    to an hour) is what timed out the tasks behind it on 2026-09-27.
    """

    def __init__(self, slots: asyncio.Semaphore) -> None:
        self._slots = slots
        self._held = False

    async def __aenter__(self) -> _StoreSlot:
        await self._slots.acquire()
        self._held = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._release()

    def _release(self) -> None:
        if self._held:
            self._held = False
            self._slots.release()

    def idle(
        self, session: AsyncSession
    ) -> Callable[[], contextlib.AbstractAsyncContextManager[None]]:
        """The pool's idle hook for ``session``: what the task gives up while it waits."""

        @contextlib.asynccontextmanager
        async def _idle() -> AsyncIterator[None]:
            # Ends the read transaction the extractor's carry-forward lookups
            # opened; with it goes the pooled connection. Nothing uncommitted
            # is at stake: the claim was committed before extraction began.
            await session.commit()
            self._release()
            try:
                yield
            finally:
                # A cancellation here leaves ``_held`` False, so __aexit__
                # never releases a slot this task does not hold.
                await self._slots.acquire()
                self._held = True

        return _idle


async def _pass_extract_pooled(batch: list[tuple[str, str]], report: ConsolidationReport) -> int:
    """The pooled twin of the serial extract loop.

    One asyncio task per snapshot, each on its own session, all registered on
    one :class:`~particles.llm.CompletionPool` — so every snapshot's chunk
    requests land in the same nightly batch and the turnarounds
    overlap instead of serialising. Concurrency leans entirely on machinery
    that already serves the multi-process case: the IN_PROGRESS claim
    protocol and the write lock around the pipeline's write phase.

    Two rules:

    * **At most one snapshot per corpus entry per run** — a second pending
      snapshot of the same entry stays PENDING for the next run
      (level-triggered), preserving the ordering where a later snapshot's
      carry-forward lookup sees the earlier one's persisted chunk hashes.
    * **An account-level failure stops the pass once.** The pool raises it in
      every parked task; each task's cleanup has already reset its snapshot
      IN_PROGRESS → PENDING, and it is counted as one failure here, exactly
      as the serial loop's single ``break`` discloses it.
    """
    # Deferred import: the pipeline pulls the extractor registry / LLM stack
    # (AGENTS.md case 2), mirroring the serial loop above.
    from particles.db import session_scope
    from particles.ingest.pipeline import SnapshotOutcome
    from particles.llm import CompletionPool
    from particles.operations.extract import extract_snapshot

    seen_entries: set[str] = set()
    chosen: list[tuple[str, str]] = []
    for entry_id, snapshot_id in batch:
        if entry_id in seen_entries:
            continue
        seen_entries.add(entry_id)
        chosen.append((entry_id, snapshot_id))
    if len(chosen) < len(batch):
        log.info(
            "consolidation: %d pending snapshot(s) deferred to the next run "
            "(one snapshot per corpus entry per pooled pass)",
            len(batch) - len(chosen),
        )

    # expected_participants closes the startup race: no wave dispatches until
    # every task below has registered, however quickly the first ones park.
    pool = CompletionPool("extraction", expected_participants=len(chosen))
    runs = [_SnapshotRun(SnapshotOutcome()) for _ in chosen]
    # The explicit bound on store concurrency. Registration comes first and
    # needs no slot, so a task queued for one still counts toward the
    # expected participants, and the slot holders reach the pool and hand
    # their slots on. A task waiting for a slot is not parked, so the wave
    # waits for it: every snapshot still rides the one batch.
    slots = asyncio.Semaphore(get_config().consolidation.extract_db_concurrency)
    finished = 0
    _count("extract", 0, len(chosen), "snapshots")

    async def _run_one(entry_id: str, snapshot_id: str, run: _SnapshotRun) -> list[Particle]:
        nonlocal finished
        slot = _StoreSlot(slots)
        try:
            async with (
                session_scope() as task_session,
                pool.participant(idle=slot.idle(task_session)),
                slot,
            ):
                written = await extract_snapshot(
                    task_session,
                    entry_id,
                    snapshot_id,
                    agent_id=report.actor,
                    completion_pool=pool,
                    skip_if_superseded=True,
                    carry_forward_ids_out=run.carried,
                    suppressed_ids_out=run.suppressed,
                    outcome_out=run.outcome,
                )
                await task_session.commit()
                return written
        finally:
            finished += 1
            _count("extract", finished, len(chosen), "snapshots")

    outcomes = await asyncio.gather(
        *(
            _run_one(entry_id, snapshot_id, run)
            for (entry_id, snapshot_id), run in zip(chosen, runs, strict=True)
        ),
        return_exceptions=True,
    )

    account_level_seen = False
    for (entry_id, snapshot_id), run, outcome in zip(chosen, runs, outcomes, strict=True):
        if isinstance(outcome, AccountLevelLLMError):
            # Not per-snapshot: every task failed identically and each
            # snapshot was already reset to PENDING on the way out. Disclose
            # once below, mirroring the serial loop's single break.
            account_level_seen = True
        elif isinstance(outcome, BaseException):
            report.pending_failed += 1
            log.warning(
                "consolidation: extraction failed for %s/%s: %s",
                entry_id[:8],
                snapshot_id[:8],
                outcome,
            )
        else:
            _record_extraction(report, entry_id, snapshot_id, outcome or [], run)
    if account_level_seen:
        report.pending_failed += 1
        log.error(
            "consolidation: extraction unavailable (account-level); "
            "untried snapshots remain PENDING for the next run"
        )
    _finish_pending_counts(report)
    # A lower bound on the spend, as in the serial loop; per-chunk counts ride
    # the OTel spans.
    return report.pending_extracted


async def _pass_reconcile(session: AsyncSession, report: ConsolidationReport) -> int:
    """Pass 2: the cross-entry document-supersession sweep. Idempotent.

    Probe-bearing (one ``semantic_lint`` call per candidate pair, routed
    through the breaker seam and capped at
    ``consolidation.max_reconcile_probes``, highest-similarity-first) — so the
    caller only runs it when ``semantic_ok``, and a capped run's truncation is
    disclosed in the report ("probed X of Y candidate pairs").

    Under a dollar budget the sweep's gather runs first and its capped probe
    count is priced; a sweep that would pass the budget is skipped.
    """
    if current_spend_budget() is not None:
        cap = get_config().consolidation.max_reconcile_probes
        pairs = await count_supersession_candidates(session)
        reason = _budget_skip_reason("reconcile", probe_estimate(min(pairs, cap)))
        if reason is not None:
            raise _BudgetSkip(reason)
    summary = await reconcile_supersession(session)
    demoted = summary.get("demoted", 0)
    probed = summary.get("probed", 0)
    candidates = summary.get("candidate_pairs", 0)
    report.reconcile_demoted = demoted if isinstance(demoted, int) else 0
    report.reconcile_candidate_pairs = candidates if isinstance(candidates, int) else 0
    report.reconcile_probes_run = probed if isinstance(probed, int) else 0
    return probed if isinstance(probed, int) else 0


async def _pass_reconcile_updates(
    session: AsyncSession,
    report: ConsolidationReport,
    scope_ids: frozenset[str] | None,
) -> int:
    """Pass 2b: the same-subject update sweep. Idempotent.

    Probe-bearing like pass 2 — one contradiction probe per *qualifying* pair
    (the sweep pre-filters by lineage and date, so no probe is spent on a pair
    rung 2.5 could not act on), capped at ``consolidation.max_update_probes``
    and delta-scoped to the cycle's window, plus one update probe per pair the
    first confirms. The return value counts both as LLM calls.

    Under a dollar budget the sweep's gather runs first and is priced at two
    probes per capped pair, the most it can spend.
    """
    if current_spend_budget() is not None:
        cap = get_config().consolidation.max_update_probes
        pairs = await count_update_candidates(session, scope_ids)
        reason = _budget_skip_reason("reconcile_updates", probe_estimate(2 * min(pairs, cap)))
        if reason is not None:
            raise _BudgetSkip(reason)
    summary = await reconcile_updates(session, scope_ids=scope_ids)
    demoted = summary.get("demoted", 0)
    probed = summary.get("probed", 0)
    update_probed = summary.get("update_probed", 0)
    report.update_demoted = demoted if isinstance(demoted, int) else 0
    report.update_probes_run = probed if isinstance(probed, int) else 0
    cleared = summary.get("previously_cleared", 0)
    report.update_previously_cleared = cleared if isinstance(cleared, int) else 0
    fixed = summary.get("fixed_slot", 0)
    report.update_fixed_slot = fixed if isinstance(fixed, int) else 0
    candidates = summary.get("candidate_pairs", 0)
    report.update_candidate_pairs = candidates if isinstance(candidates, int) else 0
    return report.update_probes_run + (update_probed if isinstance(update_probed, int) else 0)


async def _pass_reanchor(session: AsyncSession, report: ConsolidationReport, *, actor: str) -> int:
    """Pass 2c: restate claims that relied on a state an update retired.

    Probe-bearing: one probe per update retirement past the cursor, a second
    reading per restatement, and the duplicate and contradiction checks each
    restatement needs, all under ``consolidation.reanchor``'s caps. Resumes
    from the cursor the last run of this actor recorded.

    Under a dollar budget the pass's own dry run (no call, no write) prices it
    first: one probe per trigger and, at most, one second reading per
    candidate.
    """
    cursor = await prior_cursor(session, actor)
    if current_spend_budget() is not None:
        planned = await run_reanchor(session, cursor=cursor, actor=actor, dry_run=True)
        reason = _budget_skip_reason("reanchor", _reanchor_estimate(planned))
        if reason is not None:
            raise _BudgetSkip(reason)
    report.reanchor = await run_reanchor(session, cursor=cursor, actor=actor)
    await session.commit()
    return report.reanchor.llm_calls


def _reanchor_estimate(planned: ReanchorReport) -> PassEstimate:
    """A re-anchor dry run priced: probes on ``extraction``, readings on ``verification``."""
    return context_call_estimate("extraction", planned.probes) + context_call_estimate(
        "verification", planned.candidates
    )


async def _census_scope(
    session: AsyncSession,
    report: ConsolidationReport,
    census: CensusDecision,
    scope: Literal["delta", "store"],
    nightly_ids: frozenset[str] | None,
) -> frozenset[str] | None:
    """The census's probe scope: changes since the last census started.

    The nightly passes scope to changes since the last *run*; the census, run
    less often, scopes to changes since the last *census*, so a belief written
    on a night the census skipped is still probed by the next one. Store-wide
    on ``--scope store`` and when no census is on record. Reuses the nightly
    scope when both windows open at the same instant.
    """
    if scope == "store" or census.last_ran is None:
        report.census_scope = "store"
        return None
    report.census_scope = "delta"
    report.census_watermark = census.last_ran
    if nightly_ids is not None and report.watermark == census.last_ran:
        ids = nightly_ids
    else:
        ids = await _delta_scope_ids(session, census.last_ran)
    report.census_scope_particle_count = len(ids)
    return ids


async def _pass_census(
    session: AsyncSession,
    report: ConsolidationReport,
    *,
    semantic: bool,
    scope_ids: frozenset[str] | None,
) -> list[CurationCard]:
    """Pass 3: one ``collect_cards`` pass, capped + scoped (§3.3).

    The re-audit composition: the probe control carries
    ``audit.max_contradiction_probes`` and the §4 delta scope; the
    duplicate partition keeps the store-wide tail count beside the in-scope
    headline. Duplicates run in REPORT mode (unjudged candidates — the audit's
    default; ``--judge`` remains an interactive choice). Granularity probes
    stay off on the card path (via ``collect_cards``).
    """
    probe_control: ContradictionProbeControl | None = None
    if semantic:
        probe_control = ContradictionProbeControl(
            max_probes=get_config().audit.max_contradiction_probes,
            scope_particle_ids=scope_ids,
            # The dream cycle runs unattended at 03:30: nobody is
            # waiting on these probes, so they go out as one half-price batch.
            # ``particles lint`` and the interactive first-run
            # audit leave this off and keep the sequential loop.
            latency_tolerant=True,
            # count a flag only once a second, context-rich reading
            # confirms it, so the nightly census and the curation queue it
            # persists agree with the interactive audit. The readings run one
            # at a time after the batch lands, bounded by their own cap.
            verify=get_config().audit.verify_contradictions,
            max_verifications=get_config().audit.max_contradiction_verifications,
            # a disagreement a census record already discloses is
            # reported through the record, not probed and paid for again.
            exclude_pairs=await covered_pair_set(session),
        )
        if _active_progress.get() is not None:
            probe_control.on_progress = lambda done, total: _count("census", done, total, "probes")
            probe_control.on_verify_progress = lambda done, total: _count(
                "census", done, total, "readings"
            )
    cards = await collect_cards(
        session,
        semantic=semantic,
        duplicate_mode=SuggestMode.REPORT,
        contradiction_probe=probe_control,
        duplicate_scope_ids=scope_ids,
    )

    _count_cards(report, cards, scope_ids)

    if probe_control is not None:
        report.contradiction_candidate_pairs = probe_control.candidate_pairs
        report.contradiction_intra_scope_pairs = probe_control.intra_scope_pairs
        report.contradiction_probes_run = probe_control.probes_run
        report.contradiction_previously_cleared = probe_control.previously_cleared
        report.contradiction_verified = probe_control.verify
        report.contradiction_flagged = probe_control.flagged
        report.contradiction_confirmed = probe_control.confirmed
        report.contradiction_unverified = probe_control.unverified
        report.contradiction_verifications_run = probe_control.verifications_run
        grouped = probe_control.disagreements()
        report.contradiction_disagreements = grouped.groups
        report.contradiction_grouped_pairs = grouped.pairs
        report.contradiction_finding_pairs = list(probe_control.finding_pairs)
        report.contradiction_confirmed_pairs = list(probe_control.confirmed_pairs)
    return cards


def _count_cards(
    report: ConsolidationReport, cards: list[CurationCard], scope_ids: frozenset[str] | None
) -> None:
    """The census's per-class counts off one card collection (§7)."""
    # always recorded, so the next run's contradictions delta can
    # tell "no open conflicts" from "written before the kind existed".
    counts: dict[str, int] = {CardKind.INCONSISTENCY.value: 0}
    bases: dict[str, int] = {}
    for card in cards:
        counts[card.kind.value] = counts.get(card.kind.value, 0) + 1
        for basis in card.contested_bases or ():
            bases[basis] = bases.get(basis, 0) + 1
    report.card_counts = counts
    report.contested_bases = bases

    # hybrid: in-scope headline, store-wide tail always disclosed.
    dup_total = sum(1 for c in cards if c.kind is CardKind.DUPLICATE_PAIR)
    report.duplicate_candidate_pairs_total = dup_total
    if scope_ids is None:
        report.duplicate_in_scope = dup_total
    else:
        report.duplicate_in_scope = sum(
            1
            for c in cards
            if c.kind is CardKind.DUPLICATE_PAIR and any(pid in scope_ids for pid in c.particle_ids)
        )


async def _pass_disclose(
    session: AsyncSession,
    report: ConsolidationReport,
    cards: list[CurationCard],
    *,
    census_ran: bool,
    semantic: bool,
    degrade_reason: str | None,
    actor: str,
    scope_ids: frozenset[str] | None,
) -> tuple[list[CurationCard], frozenset[frozenset[str]]]:
    """Pass 3b: open, close and regroup census records, then re-count.

    Minting needs this run's confirmed pairs, so it runs only when the census
    ran with the second reading on; the lapse sweep and any pairs waiting from
    the last run are handled either way. Afterwards the card collection is
    brought in line with the records: a disclosed pair's CONTRADICTION card
    gives way to the record's INCONSISTENCY card (§7, as amended),
    every conflict card is rebuilt from the records as they now stand,
    and the census counts are taken again so the headline counts each
    disagreement once.

    Returns the adjusted collection and the pairs now covered, which pass 4's
    carry-forward drops.
    """
    mint = census_ran and semantic and report.contradiction_verified
    reason: str | None = None
    if not census_ran:
        reason = "the census did not run this night"
    elif not semantic:
        reason = f"no census probe this run ({degrade_reason or 'semantic passes skipped'})"
    elif not report.contradiction_verified:
        reason = "audit.verify_contradictions is false, so no pair is confirmed"
    disclosure = await run_disclosure(
        session,
        confirmed=report.contradiction_confirmed_pairs,
        mint=mint,
        not_minting_reason=reason,
        unconfirmed=max(0, report.contradiction_flagged - report.contradiction_confirmed),
        actor=actor,
        reread=mint,
    )
    report.disclosure = disclosure
    if not disclosure.enabled:
        return cards, frozenset()

    covered = await covered_pair_set(session)
    census_members = {
        pid
        for record in await get_census_records(session)
        if record.status is Status.INCONSISTENCY and (sides := census_sides(record)) is not None
        for pid in sides.members
    }
    report.disclosure_open_records = disclosure.open_records
    if not census_ran:
        return cards, covered

    touched = set(disclosure.touched_ids)
    swapped: list[CurationCard] = []
    for card in cards:
        if card.kind is CardKind.INCONSISTENCY and touched:
            # rebuilt below from the records as they now stand,
            # since this pass opened, closed or regrouped some.
            continue
        if card.kind is CardKind.CONTRADICTION and card.particle_ids:
            partner = contradiction_partner(card.diagnostic)
            if partner is not None and frozenset((card.particle_ids[0], partner)) in covered:
                continue
        if (
            card.kind is CardKind.CONTESTED
            and card.particle_ids
            and card.particle_ids[0] in touched
        ):
            continue
        swapped.append(card)
    if touched:
        loaded = await get_particles_by_ids(session, sorted(touched))
        swapped.extend(
            cards_from_findings(await contested_findings_for(session, list(loaded.values())))
        )
        swapped.extend(cards_from_findings(await open_inconsistency_findings(session)))

    _count_cards(report, swapped, scope_ids)
    report.contested_census_claims = sum(
        1
        for c in swapped
        if c.kind is CardKind.CONTESTED
        and "inconsistency" in (c.contested_bases or ())
        and c.particle_ids
        and c.particle_ids[0] in census_members
    )
    if report.contradiction_disagreements is not None:
        remaining = [
            p for p in report.contradiction_finding_pairs if frozenset(p[:2]) not in covered
        ]
        grouped = count_disagreements(remaining)
        report.contradiction_disagreements = grouped.groups
        report.contradiction_grouped_pairs = grouped.pairs
    return swapped, covered


async def _pass_curation(
    session: AsyncSession,
    report: ConsolidationReport,
    cards: list[CurationCard],
    *,
    scope: CollectionScope,
    semantic: bool,
    covered: frozenset[frozenset[str]] = frozenset(),
) -> int:
    """Pass 4: persist the collection pass 3 paid for, and report the top worklist.

    **The original §3.4 honesty note is superseded.** The queue used to be
    computed on demand and persist nothing, so this pass ended with a rendered
    worklist and threw 13,000+ fully-formed cards away — while `GET /curation`
    rebuilt them from scratch on every request (172 s measured). The collection
    is now stored, so the night's work is what the morning's queue serves.

    ``scope`` is this run's §4 scope, which drives the §4 per-kind
    replace-vs-carry-forward rule: a delta run must not let its narrower
    contradiction set erase the store-wide picture built by earlier runs.
    """
    merged, snapshot_id = await collect_and_persist(
        session, semantic=semantic, scope=scope, cards=cards, covered_pairs=covered
    )
    report.curation_snapshot_id = snapshot_id

    suppressed = await _suppressed_keys(session)
    eligible = [c for c in merged if c.key not in suppressed]
    result = await build_curation_queue(session, cards=eligible)
    report.curation_queue_total = len(eligible)
    report.curation_queue = [_queue_line(card) for card in result.cards]
    return 0


async def _pass_curation_stored(session: AsyncSession, report: ConsolidationReport) -> int:
    """Pass 4 on a night the census skipped: serve the last stored collection.

    ``build_curation_queue`` reads the newest snapshot and runs only the live
    session half over it, so a card resolved, snoozed or retired
    since the census is not served. Writes no snapshot: nothing was collected.
    """
    result = await build_curation_queue(session, source=QueueSource.SNAPSHOT)
    report.curation_snapshot_id = result.snapshot_id
    report.curation_snapshot_built_at = result.built_at
    report.curation_queue_total = result.open_count
    report.curation_queue = [_queue_line(card) for card in result.cards]
    return 0


def _queue_line(card: CurationCard) -> str:
    """One rendered worklist line: kind, claim text (when briefed), diagnostic."""
    text = card.particles[0].content if card.particles else None
    body = f'"{text}"' if text else (card.corpus_url or card.key)
    detail = f" — {card.diagnostic}" if card.diagnostic else ""
    return f"[{card.kind.value}] {body}{detail}"


async def _pass_utility(
    session: AsyncSession,
    report: ConsolidationReport,
    *,
    watermark: datetime | None,
    behavioural: bool,
) -> int:
    """Pass 5: the utility-mining pass over CONVERSATION entries.

    Delta runs mine transcripts harvested since the last run (idempotent
    regardless — events are keyed on ``(particle_id, session_id)``); a
    store-wide / first run re-mines everything reachable. Only beliefs a
    session was shown are candidates, and only the judge's "applied" ruling
    credits one, so a degraded run (§6), which makes no LLM call,
    credits nothing.

    ONE judge budget for the whole pass (correction v1.74.1):
    ``utility.mining.max_behavioural_calls`` is a per-*run* cap, so the
    remaining budget is threaded into every session's plan in entry order and
    judging stops when it is spent. Exhaustion is disclosed in the report
    ("use-judge budget exhausted after N of M sessions").

    Gather / decide / apply: every session is planned
    first, reads only; then all of their matcher groups go out as one pooled
    ``complete_many`` submission, so a night's sessions wait on one batch
    rather than one each; then each session's evidence is recorded under the
    store write lock, which is never held across the batch wait.
    A group the batch could not answer credits nothing in its own session, as
    a per-session batch would.
    """
    mines = await _plan_utility(session, watermark=watermark, behavioural=behavioural)
    planned = sum(len(mine.groups) for mine in mines)

    # the judge is the pass's only LLM spend. When it would pass the
    # budget, every group reads as unanswered, exactly as after a failed batch,
    # and nothing is credited.
    skip = (
        _budget_skip_reason("utility", estimate_judge_calls(mines), what="use judge")
        if planned
        else None
    )
    replies: list[list[str | None]]
    if skip is not None:
        report.utility_behavioural_skipped = skip
        _note_budget_skip(report, "utility (use judge)")
        replies = [[None] * len(mine.groups) for mine in mines]
    else:
        # Unattended run — the pooled groups may be batched.
        replies = await judge_sessions(mines, latency_tolerant=True)

    calls = 0
    async with write_lock():
        for mine, session_replies in zip(mines, replies, strict=True):
            result = await record_session_mine(session, mine, session_replies)
            report.utility_literal += result.literal
            report.utility_behavioural += result.behavioural
            report.utility_sessions_mined += 1
            _count("utility", report.utility_sessions_mined, len(mines), "sessions")
            calls += result.behavioural_calls
            if (
                behavioural
                and result.behavioural_truncated
                and report.utility_behavioural_exhausted_after is None
            ):
                report.utility_behavioural_exhausted_after = report.utility_sessions_mined
        await session.commit()
    report.utility_behavioural_calls = calls
    return calls


async def _plan_utility(
    session: AsyncSession, *, watermark: datetime | None, behavioural: bool
) -> list[SessionMine]:
    """The utility pass's gather: one :class:`SessionMine` per transcript to mine; reads only.

    Shared by the pass and ``memory consolidate --dry-run``, whose
    count of matcher groups is the behavioural calls the pass would make.
    """
    actives = await get_particles_by_status(session, Status.ACTIVE)
    reader = await ExposureReader.load(session)
    sessions, skipped = await harvested_sessions(
        session,
        since=watermark,
        on_read=lambda read, total: _count("utility", read, total, "transcripts read"),
    )
    if skipped:
        log.warning(
            "consolidation: %d harvested transcript(s) with a missing corpus blob skipped", skipped
        )
    budget = get_config().utility.mining.max_behavioural_calls
    planned = 0
    mines: list[SessionMine] = []
    for harvested in sessions:
        mine = await plan_harvested_session(
            session,
            reader,
            harvested,
            actives,
            behavioural_matching=None if behavioural else False,
            max_behavioural_calls=max(0, budget - planned),
        )
        planned += len(mine.groups)
        mines.append(mine)
    return mines


async def _pass_abstraction(
    session: AsyncSession,
    report: ConsolidationReport,
    *,
    scope_ids: frozenset[str] | None,
) -> int:
    """Pass 5b: the abstraction-promotion pass.

    Revalidation ladder first, then new-cluster promotion, both under
    ``consolidation.abstraction.max_promotions_per_run``. Delta scope gates
    cluster discovery only (revalidation is store-wide by design — a premise
    change outside the window still needs repair).
    """
    report.abstraction = await run_abstraction_pass(session, scope_ids=scope_ids)
    await session.commit()
    return report.abstraction.llm_calls


async def _pass_projection(report: ConsolidationReport, runner: ProjectionRunner) -> int:
    """Pass 6: the render via the Surface-injected harvest-then-render tail."""
    report.projection = await runner()
    return 0


async def _pass_measure(session: AsyncSession, report: ConsolidationReport, *, actor: str) -> int:
    """Pass 6b: the closure measure over the window since this actor's last run.

    The window opens where the previous run's closed, so consecutive runs tile
    the event log (see :mod:`particles.operations.closure_measure`). Reads only.
    """
    prior = await latest_run_event(session, actor=actor)
    report.closure_previous = measure_from_payload(prior.payload) if prior is not None else None
    report.closure = await measure_closure(
        session, window_start=window_start_after(prior), window_end=_utcnow()
    )
    return 0


# ---------------------------------------------------------------------------
# Delta report (§7 — the fold)
# ---------------------------------------------------------------------------

#: Headline keys the delta report tracks, and how each is computed.
_DELTA_KEYS = ("contradictions", "duplicates", "stale")


def _contradiction_headline(
    cards: int,
    open_records: int,
    disagreements: int | None,
    grouped_pairs: int,
) -> int:
    """The contradictions headline: disagreements plus open conflict records.

    Pairs connected through a shared claim count once; a contradiction card the
    probe did not report (a recorded edge) counts as its own. A run record
    written before 1.152.0 carries no grouping and counts cards. Each open
    INCONSISTENCY record counts once, census or extract-time, because it has
    one card; a night that turns a confirmed pair into a census
    record therefore reads no change.
    """
    if disagreements is None:
        return cards + open_records
    return disagreements + max(0, cards - grouped_pairs) + open_records


def _headline_values(report: ConsolidationReport) -> dict[str, int]:
    return {
        "contradictions": report.headline_contradictions,
        "duplicates": report.duplicate_candidate_pairs_total,
        "stale": report.headline_stale,
    }


def _headline_from_payload(payload: dict[str, Any]) -> dict[str, int] | None:
    census = payload.get("census")
    if not isinstance(census, dict) or census.get("skipped"):
        # a run that skipped the census measured nothing.
        return None
    cards = census.get("cards")
    if not isinstance(cards, dict):
        return None

    def _count(kind: CardKind) -> int:
        value = cards.get(kind.value, 0)
        return int(value) if isinstance(value, int) else 0

    grouped = census.get("contradiction_disagreements")
    pairs = census.get("contradiction_grouped_pairs", 0)
    headline: dict[str, int] = {
        "duplicates": int(census.get("duplicate_candidate_pairs_total", 0) or 0),
        "stale": (
            _count(CardKind.STALE)
            + _count(CardKind.RECENCY_DECAY)
            + _count(CardKind.CONFIDENCE_DECAY)
        ),
    }
    # a record written before the INCONSISTENCY card kind counted
    # conflicts by claim, and its per-record count cannot be recovered, so the
    # contradictions delta is omitted for that one run rather than reported as
    # a spurious rise.
    if CardKind.INCONSISTENCY.value in cards:
        headline["contradictions"] = _contradiction_headline(
            _count(CardKind.CONTRADICTION),
            _count(CardKind.INCONSISTENCY),
            grouped if isinstance(grouped, int) else None,
            pairs if isinstance(pairs, int) else 0,
        )
    return headline


def _compute_deltas(report: ConsolidationReport, prior: OperatorEvent) -> dict[str, int]:
    """Per-headline-class delta against the prior run's recorded census."""
    previous = _headline_from_payload(prior.payload or {})
    if previous is None:
        return {}
    current = _headline_values(report)
    if report.census_probe_skipped:
        # an unprobed census counts no new contradiction pairs, so a
        # delta against a probed night would read as conflicts resolved.
        previous.pop("contradictions", None)
    return {key: current[key] - previous[key] for key in _DELTA_KEYS if key in previous}


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _census_payload(report: ConsolidationReport) -> dict[str, Any]:
    """The machine-readable census block of the run record (§7)."""
    return {
        # the cadence. ``skipped`` marks a run that did not run the
        # census, so no later run reads its empty counts as a measurement.
        "skipped": report.census_skipped,
        "last_ran_at": _iso(report.census_last_ran_at),
        "next_due_at": _iso(report.census_next_due_at),
        "scope": report.census_scope,
        "watermark": _iso(report.census_watermark),
        "scope_particle_count": report.census_scope_particle_count,
        "cards": dict(report.card_counts),
        "contested_bases": dict(report.contested_bases),
        "contradiction_candidate_pairs": report.contradiction_candidate_pairs,
        "contradiction_intra_scope_pairs": report.contradiction_intra_scope_pairs,
        "contradiction_probes_run": report.contradiction_probes_run,
        "contradiction_previously_cleared": report.contradiction_previously_cleared,
        "contradiction_verified": report.contradiction_verified,
        "contradiction_flagged": report.contradiction_flagged,
        "contradiction_confirmed": report.contradiction_confirmed,
        "contradiction_unverified": report.contradiction_unverified,
        "contradiction_disagreements": report.contradiction_disagreements,
        "contradiction_grouped_pairs": report.contradiction_grouped_pairs,
        "disclosure_open_records": report.disclosure_open_records,
        "contested_census_claims": report.contested_census_claims,
        **({"disclosure": report.disclosure.payload()} if report.disclosure is not None else {}),
        "duplicate_candidate_pairs_total": report.duplicate_candidate_pairs_total,
        "duplicate_in_scope": report.duplicate_in_scope,
        "pending_backlog": report.pending_remaining,
        "pending_extracted": report.pending_extracted,
        "pending_empty": report.pending_empty,
        "pending_retry": report.pending_retry,
        "pending_partial": report.pending_partial,
        "pending_waiting": report.pending_waiting,
        "pending_failed": report.pending_failed,
        "pending_reset_stale": report.pending_reset_stale,
        "pending_oldest_at": (
            report.pending_oldest_at.isoformat() if report.pending_oldest_at else None
        ),
        "pending_collapsed": report.pending_collapsed,
        "refresh_checked": report.refresh_checked,
        "refresh_updated": report.refresh_updated,
        "refresh_missing": report.refresh_missing,
        "reconcile_demoted": report.reconcile_demoted,
        "reconcile_candidate_pairs": report.reconcile_candidate_pairs,
        "reconcile_probes_run": report.reconcile_probes_run,
        "update_previously_cleared": report.update_previously_cleared,
        **({"reanchor": report.reanchor.payload()} if report.reanchor is not None else {}),
        "utility_literal": report.utility_literal,
        "utility_behavioural": report.utility_behavioural,
        "utility_behavioural_calls": report.utility_behavioural_calls,
        "utility_behavioural_exhausted_after": report.utility_behavioural_exhausted_after,
        "curation_queue_total": report.curation_queue_total,
        "curation_snapshot_id": report.curation_snapshot_id,
        "curation_snapshot_built_at": _iso(report.curation_snapshot_built_at),
        **(
            {
                "abstraction_clusters": report.abstraction.clusters_found,
                "abstraction_promoted": len(report.abstraction.promoted_particle_ids),
                "abstraction_proposed": len(report.abstraction.proposed_event_ids),
                "abstraction_rejected_entailment": report.abstraction.rejected_entailment,
                "abstraction_rejected_population_scope": (
                    report.abstraction.rejected_population_scope
                ),
                "abstraction_rejected_duplicate": report.abstraction.rejected_duplicate,
                # Same-text successors minted by rungs 1–3.
                "abstraction_revalidated": (
                    report.abstraction.revalidation.refreshed_structural
                    + report.abstraction.revalidation.refreshed_entailed
                    + report.abstraction.revalidation.refreshed_paraphrase
                ),
                "abstraction_superseded": report.abstraction.revalidation.superseded,
                "abstraction_retired": report.abstraction.revalidation.retired,
                "abstraction_deferred_in_review": (
                    report.abstraction.revalidation.deferred_in_review
                ),
            }
            if report.abstraction is not None
            else {}
        ),
    }


# ---------------------------------------------------------------------------
# The interactive audit's recording (§7 — ``actor: audit``)
# ---------------------------------------------------------------------------


async def record_audit_run(
    session: AsyncSession,
    audit_report: AuditReport,
    *,
    started_at: datetime,
    actor: str = "audit",
) -> None:
    """Record an interactive audit as a ``CONSOLIDATION_RUN`` event (§7).

    The audit contributes to the §7 *delta report* (the most-recent-prior-run
    comparison) — a request asked for exactly this record and is discharged
    here. Correction (v1.74.1): an ``actor: audit`` event is **not**
    watermark-eligible and does **not** satisfy ``--if-due`` — the audit runs
    neither reconcile, utility mining, curation refresh, nor projection, so it
    must not stand in for a consolidation run (see :func:`latest_run_event`).
    Flushes via
    ``record_event``; the caller owns the commit, exactly like the audit's
    other writes. (``AuditReport`` is imported under ``TYPE_CHECKING`` only:
    ``operations.audit`` imports this module at runtime for the shared payload
    shape, so a runtime import here would be a cycle.)
    """
    completed_at = _utcnow()
    harvested = audit_report.files_audited is not None or audit_report.transcripts_audited > 0
    extract_pass = ConsolidationPass(
        name="extract",
        status="ran" if harvested else "skipped",
        detail=None if harvested else "re-audit — no harvest",
        llm_calls=int(audit_report.extracted_snapshots),
    )
    census_pass = ConsolidationPass(
        name="census",
        status="ran",
        llm_calls=int(audit_report.contradiction_probes_run),
    )
    counts = {bucket.kind.value: bucket.count for bucket in audit_report.buckets}
    report = ConsolidationReport(
        store=audit_report.store,
        actor=actor,
        card_counts=counts,
        contradiction_candidate_pairs=audit_report.contradiction_candidate_pairs,
        contradiction_intra_scope_pairs=audit_report.contradiction_intra_scope_pairs,
        contradiction_probes_run=audit_report.contradiction_probes_run,
        contradiction_previously_cleared=audit_report.contradiction_previously_cleared,
        contradiction_verified=audit_report.contradiction_verified,
        contradiction_flagged=audit_report.contradiction_flagged,
        contradiction_confirmed=audit_report.contradiction_confirmed,
        contradiction_unverified=audit_report.contradiction_unverified,
        contradiction_disagreements=audit_report.contradiction_disagreements,
        contradiction_grouped_pairs=audit_report.contradiction_grouped_pairs,
        duplicate_candidate_pairs_total=audit_report.duplicate_candidate_pairs_total,
        pending_extracted=audit_report.extracted_snapshots,
    )
    payload = build_run_payload(
        store=audit_report.store,
        actor=actor,
        scope=audit_report.contradiction_probe_scope or "store",
        watermark=None,
        started_at=started_at,
        completed_at=completed_at,
        semantic_degraded=bool(audit_report.semantic_skipped),
        semantic_degraded_reason=audit_report.semantic_skip_reason,
        providers=_current_providers(),
        passes=[extract_pass, census_pass],
        census=_census_payload(report),
        llm_usage=audit_report.llm_usage,
    )
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.CONSOLIDATION_RUN,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# The renderer (§7) — one renderer for terminal and --output
# ---------------------------------------------------------------------------


def _fmt_delta(deltas: dict[str, int], key: str) -> str:
    if key not in deltas:
        return ""
    value = deltas[key]
    sign = "+" if value > 0 else ""
    # Deltas compare with the last run that measured, which since is
    # the last census rather than last night.
    return f"  ({sign}{value} since the last census)"


def _pending_line(report: ConsolidationReport) -> str:
    """The pass 1 headline: what was extracted, and how much is still waiting."""
    parts = [f"extracted {report.pending_extracted}"]
    if report.pending_retry:
        parts.append(f"{report.pending_retry} left for retry")
    if report.pending_partial:
        parts.append(f"{report.pending_partial} kept part of their read")
    if report.pending_waiting:
        parts.append(f"{report.pending_waiting} waiting")
    if report.pending_empty:
        parts.append(f"{report.pending_empty} with no beliefs")
    line = "  pending          " + ", ".join(parts)
    if report.pending_remaining:
        waiting = f"{report.pending_remaining} remain"
        if report.pending_oldest_at is not None:
            waiting += f", oldest waiting since {report.pending_oldest_at:%Y-%m-%d}"
        line += f" ({waiting}; the next run continues)"
    if report.pending_collapsed:
        line += f"; skipped {report.pending_collapsed} superseded snapshot(s)"
    return line


def render_reanchor_line(reanchor: ReanchorReport) -> str:
    """The report line for pass 2c."""
    if reanchor.skipped_reason is not None:
        return f"  re-anchor        skipped ({reanchor.skipped_reason})"
    parts = [f"{reanchor.triggers} update(s) examined"]
    if reanchor.restated:
        parts.append(f"{len(reanchor.restated)} dependent claim(s) restated")
    if reanchor.matched_existing:
        parts.append(f"{len(reanchor.matched_existing)} matched to an existing claim")
    if reanchor.unrestated:
        parts.append(f"{len(reanchor.unrestated)} kept for review (stale_basis)")
    if reanchor.waiting:
        parts.append(f"{reanchor.waiting} waiting")
    return "  re-anchor        " + ", ".join(parts)


def render_disclosure_line(disclosure: DisclosureReport) -> str:
    """The pass 3b report line: opened, closed, waiting, and why not."""
    if not disclosure.enabled:
        return f"  contradiction disclosure: off ({disclosure.not_minting_reason})"
    parts: list[str] = []
    if disclosure.opened:
        records = disclosure.opened_records
        claims = sum(int(o.get("members", 0)) for o in disclosure.opened)
        noun = "inconsistency" if records == 1 else "inconsistencies"
        parts.append(
            f"opened {records} {noun} ({claims} claims) for the agent (cap {disclosure.cap})"
        )
    elif disclosure.not_minting_reason:
        parts.append(f"opened none ({disclosure.not_minting_reason})")
    else:
        parts.append(f"opened none (cap {disclosure.cap})")
    lapsed = sum(1 for c in disclosure.closed if c.get("cause") == "lapsed")
    regrouped = sum(1 for c in disclosure.closed if c.get("cause") == "regrouped")
    withdrawn = sum(1 for c in disclosure.closed if c.get("cause") == "withdrawn")
    if lapsed:
        parts.append(f"closed {lapsed} whose side is no longer stated")
    if withdrawn:
        parts.append(f"closed {withdrawn} that a re-reading no longer confirms")
    if regrouped:
        parts.append(f"regrouped {regrouped}")
    parts.append(f"{len(disclosure.waiting)} waiting")
    if disclosure.dropped_waiting:
        parts.append(f"{len(disclosure.dropped_waiting)} waiting pair(s) no longer qualify")
    if disclosure.unsided:
        parts.append(f"{disclosure.unsided} group(s) could not be split into two sides")
    reread = disclosure.reread
    if reread.get("read") or reread.get("deferred"):
        note = (
            f"re-read {reread.get('read', 0)} pair(s) confirmed under an earlier instruction, "
            f"withdrew {reread.get('withdrawn', 0)}"
        )
        if reread.get("failed"):
            note += f", {reread['failed']} unread"
        if reread.get("deferred"):
            note += (
                f", {reread['deferred']} left for a later night "
                "(consolidation.contradiction_disclosure.max_rereadings_per_run)"
            )
        parts.append(note)
    extract = disclosure.extract_reread
    if extract.get("read") or extract.get("deferred"):
        # records extraction opened on the probe alone.
        note = (
            f"read {extract.get('read', 0)} record(s) extraction opened without a second "
            f"reading, withdrew {extract.get('withdrawn', 0)}"
        )
        if extract.get("failed"):
            note += f", {extract['failed']} unread"
        if extract.get("deferred"):
            note += f", {extract['deferred']} left for a later night"
        parts.append(note)
    return "  contradiction disclosure: " + "; ".join(parts)


def _budget_lines(report: ConsolidationReport) -> list[str]:
    """The dollar budget's disclosure; nothing when the run had none."""
    budget = report.budget
    if budget is None or budget.budget_usd is None:
        return []
    lines = [
        "",
        f"  spend budget: spent US{format_usd(budget.spent_usd)} of "
        f"US{format_usd(budget.budget_usd)} (consolidation.budget_usd)"
        + (f"; skipped {', '.join(budget.skipped_passes)}" if budget.skipped_passes else ""),
    ]
    if budget.skipped_requests:
        where = f" during {budget.exhausted_in_pass}" if budget.exhausted_in_pass else ""
        lines.append(
            f"  spend budget: {budget.skipped_requests} request(s) in "
            f"{budget.skipped_chunks} batch chunk(s) not submitted{where}; they read as "
            "unavailable and a later run retries them"
        )
    if report.census_probe_skipped:
        lines.append(f"  contradiction probe not run: {report.census_probe_skipped}")
    if report.utility_behavioural_skipped:
        lines.append(
            f"  use judge not run (no utility events credited): "
            f"{report.utility_behavioural_skipped}"
        )
    return lines


def render_consolidation_report(report: ConsolidationReport) -> str:
    """Render the §7 report shape — headline deltas, disclosures, the queue."""
    lines: list[str] = []
    when = (report.completed_at or report.started_at).strftime("%Y-%m-%d %H:%M")
    if report.previous_run_at is not None:
        prev = report.previous_run_at.strftime("%Y-%m-%d %H:%M")
        lines.append(f"Consolidated store '{report.store}' — {when} (previous run: {prev})")
    else:
        lines.append(f"Consolidated store '{report.store}' — {when} (first recorded run)")
    lines.append("")

    # --- Headline counts + deltas (a degraded run never reads "0"; §6) -----
    if report.census_skipped is not None:
        # no census tonight. Say when it last ran and when it is next
        # due, and show its counts as that census's, never as tonight's.
        lines.append(f"  {report.census_skipped}")
        last = report.census_last_headline
        if last:
            lines.append(
                f"  as of that census: {_plural(last.get('contradictions', 0), 'contradiction')}, "
                f"{_plural(last.get('duplicates', 0), 'duplicate pair')}, "
                f"{last.get('stale', 0)} stale"
            )
    elif report.semantic_degraded or report.census_probe_skipped:
        reason = report.semantic_degraded_reason or report.census_probe_skipped or "LLM unavailable"
        lines.append(f"  contradictions   not probed this run ({reason})")
    else:
        contradictions_line = (
            f"  contradictions   {report.headline_contradictions}"
            f"{_fmt_delta(report.deltas, 'contradictions')}"
        )
        if report.contradiction_verified:
            # the count is confirmed flags; say what the probe flagged.
            contradictions_line += (
                f"  [first pass flagged {report.contradiction_flagged} claim pairs, "
                f"a second reading confirmed {report.contradiction_confirmed}"
                + (
                    f", {report.contradiction_unverified} not read"
                    if report.contradiction_unverified
                    else ""
                )
                + "]"
            )
        lines.append(contradictions_line)
    if report.disclosure is not None:
        lines.append(render_disclosure_line(report.disclosure))
    if report.census_skipped is None:
        dup_line = f"  duplicates       {report.duplicate_candidate_pairs_total}"
        dup_line += _fmt_delta(report.deltas, "duplicates")
        if (
            report.census_scope == "delta"
            and report.duplicate_in_scope < report.duplicate_candidate_pairs_total
        ):
            dup_line += f"  [{report.duplicate_in_scope} touch the census's delta]"
        lines.append(dup_line)
        lines.append(
            f"  stale            {report.headline_stale}{_fmt_delta(report.deltas, 'stale')}"
        )
    if report.closure is not None:
        lines.extend(render_closure_lines(report.closure, report.closure_previous))
    if report.refresh_checked:
        refresh_line = f"  local sources    {report.refresh_checked} checked"
        if report.refresh_updated:
            refresh_line += f", {report.refresh_updated} changed → re-extracting"
        else:
            refresh_line += ", none changed"
        if report.refresh_missing:
            refresh_line += f", {report.refresh_missing} missing"
        lines.append(refresh_line)
    lines.append(_pending_line(report))
    if report.reconcile_demoted:
        lines.append(f"  reconcile        demoted {report.reconcile_demoted} superseded claim(s)")
    if report.reanchor is not None:
        lines.append(render_reanchor_line(report.reanchor))
    utility = f"  utility events   +{report.utility_literal + report.utility_behavioural}"
    if report.utility_behavioural:
        utility += f" ({report.utility_behavioural} behavioural)"
    lines.append(utility)
    if report.abstraction is not None:
        ab = report.abstraction
        parts: list[str] = []
        if ab.proposed_event_ids:
            parts.append(f"{len(ab.proposed_event_ids)} proposed")
        if ab.promoted_particle_ids:
            parts.append(f"{len(ab.promoted_particle_ids)} promoted")
        reval = ab.revalidation
        repaired = (
            reval.refreshed_structural + reval.refreshed_entailed + reval.refreshed_paraphrase
        )
        if repaired:
            parts.append(f"{repaired} revalidated")
        if reval.superseded:
            parts.append(f"{reval.superseded} superseded")
        if reval.retired:
            parts.append(f"{reval.retired} retired")
        if reval.deferred_in_review:
            parts.append(f"{reval.deferred_in_review} held for review")
        if not parts:
            parts.append("nothing to do")
        lines.append(f"  abstraction      {', '.join(parts)}")
    if report.projection is not None:
        rendered = report.projection.get("rendered", 0)
        lines.append(f"  projection       re-rendered ({rendered} memory file(s))")

    # --- Disclosures (§6: every skip, cap, and degradation named) ----------
    if report.batch_wait is not None and report.batch_wait.moved_work:
        bw = report.batch_wait
        where = f", spent during {bw.exhausted_in_pass}" if bw.exhausted_in_pass else ""
        lines.append("")
        lines.append(
            f"  batch-wait budget: waited {bw.spent_seconds:.0f}s of {bw.budget_seconds:.0f}s"
            f"{where}; {bw.batches_cut_short} batch(es) cut short, "
            f"{bw.cancelled_batches} cancelled ({bw.cancelled_kept} request(s) kept, "
            f"{bw.cancelled_rerun} re-run, {bw.cancelled_lost} unavailable), "
            f"{bw.sequential_sets} set(s) "
            f"sequential at full price ({bw.sequential_requests} request(s)) "
            f"(consolidation.batch_wait.budget_seconds)"
        )
    if report.semantic_degraded:
        reason = report.semantic_degraded_reason or "LLM unavailable"
        lines.append("")
        lines.append(f"  semantic passes skipped: {reason}")
    lines.extend(_budget_lines(report))
    if (
        not report.semantic_degraded
        and report.census_skipped is None
        and report.contradiction_probes_run < report.contradiction_candidate_pairs
    ):
        cap = get_config().audit.max_contradiction_probes
        lines.append("")
        lines.append(
            f"  contradiction probe capped: probed {report.contradiction_probes_run} of "
            f"{report.contradiction_candidate_pairs} candidate pairs "
            f"(audit.max_contradiction_probes = {cap})"
        )
    # the saving from the probe-verdict ledger, as a number.
    if not report.semantic_degraded and report.contradiction_previously_cleared:
        lines.append("")
        lines.append(
            f"  contradiction probe: {report.contradiction_previously_cleared} pair(s) "
            f"skipped as previously cleared (both claims unchanged since a probe answered "
            f"no; not counted against audit.max_contradiction_probes)"
        )
    if not report.semantic_degraded and report.update_previously_cleared:
        lines.append("")
        lines.append(
            f"  update sweep: {report.update_previously_cleared} pair(s) skipped as "
            f"previously cleared (not counted against consolidation.max_update_probes)"
        )
    if (
        not report.semantic_degraded
        and report.reconcile_probes_run < report.reconcile_candidate_pairs
    ):
        reconcile_cap = get_config().consolidation.max_reconcile_probes
        lines.append("")
        lines.append(
            f"  reconcile probe capped: probed {report.reconcile_probes_run} of "
            f"{report.reconcile_candidate_pairs} candidate pairs "
            f"(consolidation.max_reconcile_probes = {reconcile_cap})"
        )
    if report.refresh_remaining:
        refresh_cap = get_config().local_refresh.max_entries
        lines.append("")
        lines.append(
            f"  local refresh capped: checked {report.refresh_checked} entries, "
            f"{report.refresh_remaining} not reached this run "
            f"(local_refresh.max_entries = {refresh_cap})"
        )
    if report.utility_behavioural_exhausted_after is not None:
        behavioural_cap = get_config().utility.mining.max_behavioural_calls
        lines.append("")
        lines.append(
            f"  use-judge budget exhausted after "
            f"{report.utility_behavioural_exhausted_after} of "
            f"{report.utility_sessions_mined} sessions "
            f"(utility.mining.max_behavioural_calls = {behavioural_cap})"
        )
    if report.reanchor is not None and report.reanchor.warnings:
        lines.append("")
        for warning in report.reanchor.warnings:
            lines.append(f"  re-anchor: {warning}")
    if report.abstraction is not None and report.abstraction.warnings:
        lines.append("")
        for warning in report.abstraction.warnings:
            lines.append(f"  abstraction: {warning}")
    if report.effective_scope == "delta" and report.watermark is not None:
        lines.append("")
        lines.append(
            f"  semantic scope: delta since {report.watermark:%Y-%m-%d %H:%M} UTC "
            f"({report.scope_particle_count or 0} beliefs) — pass --scope store for "
            f"the whole store"
        )
    if (
        report.census_scope == "delta"
        and report.census_watermark is not None
        and report.census_watermark != report.watermark
    ):
        lines.append(
            f"  census scope: delta since the last census, "
            f"{report.census_watermark:%Y-%m-%d %H:%M} UTC "
            f"({report.census_scope_particle_count or 0} beliefs)"
        )
    for entry in report.passes:
        if entry.name == "census" and report.census_skipped is not None:
            continue  # disclosed in the headline above
        if entry.status == "skipped":
            lines.append(f"  pass skipped: {entry.name} — {entry.detail}")
        elif entry.status == "failed":
            lines.append(f"  pass FAILED: {entry.name} — {entry.detail}")
    if report.pending_failed:
        lines.append(
            f"  {report.pending_failed} snapshot(s) failed to extract; "
            f"the log names each one and why."
        )
    if report.pending_retry:
        kept = (
            f" {report.pending_partial} of them kept the chunks that answered, and their "
            f"retry reads only the rest."
            if report.pending_partial
            else ""
        )
        lines.append(
            f"  {report.pending_retry} snapshot(s) left PENDING after failed LLM calls. "
            f"A later run retries them after the snapshots not yet tried.{kept}"
        )
    if report.pending_waiting:
        lines.append(
            f"  {report.pending_waiting} snapshot(s) wait on an earlier snapshot of their "
            f"entry that holds a partial whole read; the log names each one. "
            f"`particles reindex <entry>` releases a wait by hand."
        )
    if report.pending_empty:
        lines.append(
            f"  {report.pending_empty} snapshot(s) completed with no beliefs. "
            f"`particles lint` lists them as EMPTY_COMPLETE_SNAPSHOT."
        )
    if report.pending_reset_stale:
        lines.append(
            f"  {report.pending_reset_stale} snapshot(s) stranded IN_PROGRESS by an "
            f"interrupted run were returned to PENDING."
        )

    # --- The morning's curation queue (§3.4) --------------------------------
    if report.curation_queue:
        lines.append("")
        shown = len(report.curation_queue)
        source = (
            f" (from the census of {report.curation_snapshot_built_at:%Y-%m-%d %H:%M} UTC)"
            if report.curation_snapshot_built_at is not None
            else ""
        )
        lines.append(f"Curation queue — top {shown} of {report.curation_queue_total}{source}:")
        for line in report.curation_queue:
            lines.append(f"  • {line}")

    lines.append("")
    lines.append("Run 'particles curate' to work these down.")
    if report.llm_usage is not None:
        lines.append(render_usage_line(report.llm_usage))
    return "\n".join(lines) + "\n"
