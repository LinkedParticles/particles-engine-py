# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The detached SessionEnd worker: harvest every handed-off session, then render.

The launcher (``_claude_launcher.py``) leaves one pending spool file per ended
session and starts this worker unless one already holds the spool's lock. The
worker takes every pending file (renaming it ``.taken``), harvests each
session in arrival order, renders each memory directory once with the newest
harvested text, logs one ``session-end`` record per session with
``"mode": "worker"``, and deletes the file. It then looks again, and after
releasing the lock looks once more, so a session handed off while it ran is
never left behind.

A ``.taken`` file found while holding the lock was left by a worker that
exited before logging it (sleep, logout, a crash). It is logged as a
``worker lost`` error, which counts as a SessionEnd error, and harvested
again; after :data:`MAX_ATTEMPTS` such losses the session is dropped and the
level-triggered catch-up sweep is left to it.

The harvest and render steps are passed in (``hook.harvest_session`` and
``hook.render_session``), so this module carries the spool protocol only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from particles.api.cli._claude_launcher import LOCK_NAME, PENDING_SUFFIX, private_directory
from particles.api.cli._hook_state import advisory_lock
from particles.secrets import redact_known_secrets

log = logging.getLogger(__name__)

#: The suffix of a spool file a worker has taken and not yet logged.
TAKEN_SUFFIX = ".taken"

#: How many times a session may be lost with its worker before it is dropped.
MAX_ATTEMPTS = 3

#: Passes over the spool while holding the lock, and lock acquisitions per run.
_MAX_ROUNDS = 5


class _Plan(Protocol):
    @property
    def memory_dir(self) -> Path: ...


Harvest = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[Any]]
Render = Callable[[str, Any, dict[str, Any]], Awaitable[None]]
Log = Callable[[dict[str, Any]], None]


@dataclass
class _Taken:
    path: Path
    envelope: dict[str, Any]

    @property
    def payload(self) -> dict[str, Any]:
        payload = self.envelope.get("payload")
        return payload if isinstance(payload, dict) else {}


def _read(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _record(store: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": datetime.now(UTC).isoformat(),
        "event": "session-end",
        "mode": "worker",
        "store": store,
        "session_id": payload.get("session_id"),
        "source": payload.get("reason") or payload.get("source"),
    }


def pending(spool: Path) -> list[Path]:
    """The spool files no worker has taken, oldest first."""
    if not spool.is_dir():
        return []
    return sorted(p for p in spool.glob(f"*{PENDING_SUFFIX}") if p.is_file())


def taken(spool: Path) -> list[Path]:
    """The spool files a worker has taken and not yet logged."""
    if not spool.is_dir():
        return []
    return sorted(p for p in spool.glob(f"*{TAKEN_SUFFIX}") if p.is_file())


def recover_lost(spool: Path, store: str, write_log: Log) -> None:
    """Log each lost worker's session as an error and return it to the queue.

    Called only while holding the lock, so every ``.taken`` file is one a
    worker that no longer runs left behind.
    """
    for path in taken(spool):
        envelope = _read(path)
        attempts = int(envelope.get("attempts") or 0) + 1
        record = _record(store, envelope.get("payload") or {})
        record["outcome"] = "error"
        if attempts >= MAX_ATTEMPTS:
            record["error"] = (
                f"worker lost {attempts} times; dropped, the catch-up sweep retries the session"
            )
            write_log(record)
            path.unlink(missing_ok=True)
            continue
        record["error"] = "worker lost: it exited before logging this session; harvesting again"
        write_log(record)
        envelope["attempts"] = attempts
        path.write_text(json.dumps(envelope, sort_keys=True) + "\n", encoding="utf-8")
        path.rename(path.with_suffix(PENDING_SUFFIX))


def take_all(spool: Path) -> list[_Taken]:
    """Take every pending spool file (rename to ``.taken``), oldest first."""
    out: list[_Taken] = []
    for path in pending(spool):
        target = path.with_suffix(TAKEN_SUFFIX)
        try:
            path.rename(target)
        except OSError:
            continue
        out.append(_Taken(target, _read(target)))
    return out


async def run_cycle(
    store: str,
    batch: list[_Taken],
    *,
    harvest: Harvest,
    render: Render,
    write_log: Log,
    deadline: float,
) -> None:
    """Harvest each session, render each memory directory once, then log and release."""
    done: list[tuple[_Taken, dict[str, Any], float]] = []
    plans: dict[Path, tuple[_Plan, dict[str, Any]]] = {}
    for item in batch:
        record = _record(store, item.payload)
        started = time.monotonic()
        try:
            plan = await asyncio.wait_for(harvest(store, item.payload, record), deadline)
            record["outcome"] = "ok"
            if plan is not None:
                # The newest harvest of a directory renders it, with its text.
                plans.pop(plan.memory_dir, None)
                plans[plan.memory_dir] = (plan, record)
        except Exception as exc:  # noqa: BLE001 — one session's failure is its own record
            record["outcome"] = "error"
            record["error"] = redact_known_secrets(f"{type(exc).__name__}: {exc}")[:300]
        done.append((item, record, time.monotonic() - started))
    render_seconds: dict[int, float] = {}
    for plan, record in plans.values():
        started = time.monotonic()
        try:
            await asyncio.wait_for(render(store, plan, record), deadline)
        except Exception as exc:  # noqa: BLE001 — the harvest landed; the next run renders
            record["outcome"] = "error"
            record["error"] = redact_known_secrets(f"{type(exc).__name__}: {exc}")[:300]
        render_seconds[id(record)] = time.monotonic() - started
    for item, record, seconds in done:
        seconds += render_seconds.get(id(record), 0.0)
        record["duration_ms"] = round(seconds * 1000, 1)
        write_log(record)
        item.path.unlink(missing_ok=True)


def run_worker(
    store: str,
    spool: Path,
    *,
    cycle: Callable[[list[_Taken]], None],
    write_log: Log,
) -> int:
    """Drain the spool under its lock; the number of sessions handled.

    ``cycle(batch)`` runs one :func:`run_cycle` (the caller supplies the event
    loop). After releasing the lock the spool is checked once more and the
    lock taken again if a session arrived meanwhile and no other worker took
    it.
    """
    private_directory(spool)
    handled = 0
    for _ in range(_MAX_ROUNDS):
        with advisory_lock(spool / LOCK_NAME) as held:
            if not held:
                return handled
            recover_lost(spool, store, write_log)
            for _ in range(_MAX_ROUNDS):
                batch = take_all(spool)
                if not batch:
                    break
                cycle(batch)
                handled += len(batch)
        if not pending(spool):
            break
    return handled
