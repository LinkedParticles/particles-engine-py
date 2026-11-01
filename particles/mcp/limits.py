# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-store rate cap on the MCP tool calls that can drive a paid model call.

Security review D30: nothing on either MCP surface bounded how often an agent
could make the store call a model. The HTTP routes have a per-client token
bucket (``particles.api._rate_limit``); a stdio MCP server has one client, so
its bucket is keyed to the store it serves. :func:`admit` spends one token or
raises ``ValueError``, which both servers already return to the client as a
tool error. The cap is read from ``mcp.rate_limit_per_minute`` at call time,
and ``0`` disables it.

**One allowance per store, however many servers** (security review
2026-10-25-2, U11). A client may start several server processes on one store,
one per agent session, and a bucket held in process memory gave each its own
allowance. For a file-backed SQLite store the bucket lives beside the store
file (``<db>.mcp-rate``), read and written under a file lock, so every server
process on that store spends from one bucket. A store with no file (in-memory
SQLite, PostgreSQL) and a server routed to a remote engine keep the bucket in
process memory: the engine's own HTTP rate limit bounds the routed calls, and
an in-memory store is not shared across processes. Process state is cleared on
every ``reset_config()``; the shared file is not, since it belongs to the store.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from particles.config import get_config, register_reset_hook, sqlite_file_path

log = logging.getLogger(__name__)

#: The suffix of the shared bucket file beside a file-backed store.
STATE_SUFFIX = ".mcp-rate"

#: How long :func:`admit` waits for another server's update of the shared
#: bucket before it refuses the call (the update itself is a few microseconds).
_LOCK_TIMEOUT_SECONDS = 5.0

# Process-local buckets for stores with no file: key -> (tokens, last_refill).
# Mutated synchronously, with no ``await`` in the critical section, so the
# event loop cannot interleave two updates.
_buckets: dict[str, tuple[float, float]] = {}


def reset_limits() -> None:
    """Drop the process-local bucket state. Wired to ``reset_config`` (and used by tests)."""
    _buckets.clear()


def _store_dsn(store: str) -> str | None:
    storage = get_config().storage
    if store == "default":
        return storage.database_url
    return storage.stores.get(store)


def shared_state_path(store: str = "default") -> Path | None:
    """The shared bucket file for ``store``, or ``None`` when the bucket is per process.

    ``None`` for a server routed to a remote engine (``engine.base_url``), for a
    store with no file, and for a store whose directory does not exist yet (the
    server cannot open such a store either, and the cap is not where to create it).
    """
    if get_config().engine.base_url:
        return None
    dsn = _store_dsn(store)
    db_file = sqlite_file_path(dsn) if dsn else None
    if db_file is None:
        return None
    # Keyed on the store file itself, not the DSN's spelling of it: a relative
    # path, a ``..`` or a link to the same file shares one bucket (security
    # review 2026-10-26-1, R24).
    resolved = Path(db_file).resolve()
    path = resolved.with_name(resolved.name + STATE_SUFFIX)
    return path if path.parent.is_dir() else None


def _refill(tokens: float, last: float, now: float, rate: int) -> float:
    # A clock that stepped backwards refills nothing rather than draining.
    return min(float(rate), tokens + max(0.0, now - last) * rate / 60.0)


def _spend(
    state: tuple[float, float] | None, cost: int, rate: int, now: float
) -> tuple[float, bool]:
    """The tokens left after a refill, and whether ``cost`` of them can be spent."""
    tokens, last = state if state is not None else (float(rate), now)
    tokens = _refill(tokens, last, now, rate)
    return tokens, tokens >= cost


def _read_state(path: Path) -> tuple[float, float] | None:
    """The stored ``(tokens, last_refill)``, or ``None`` for a missing or unreadable file.

    The file is opened without following a link, so a link planted at its name
    reads as an unreadable file (a full bucket) rather than as another file.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, encoding="utf-8") as fh:
            raw = json.load(fh)
        return float(raw["tokens"]), float(raw["last"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_state(path: Path, tokens: float, now: float) -> None:
    """Replace the bucket file with a fresh temporary file of its own.

    The temporary file is created exclusively under a random name in the
    store's directory, so a link planted beside the store is never opened, and
    the rename replaces the name rather than writing through what it points at
    (security review 2026-10-26-1, W40).
    """
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"tokens": tokens, "last": now}, fh)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _admit_shared(path: Path, cost: int, rate: int) -> bool:
    """Spend from the bucket file under its lock; ``True`` when the spend went through."""
    try:
        with FileLock(str(path) + ".lock", timeout=_LOCK_TIMEOUT_SECONDS):
            now = time.time()
            tokens, ok = _spend(_read_state(path), cost, rate, now)
            _write_state(path, tokens - cost if ok else tokens, now)
            return ok
    except FileLockTimeout:
        log.warning("MCP rate cap: the shared bucket %s stayed locked; refusing the call", path)
        return False


def _admit_local(key: str, cost: int, rate: int) -> bool:
    now = time.monotonic()
    tokens, ok = _spend(_buckets.get(key), cost, rate, now)
    _buckets[key] = (tokens - cost if ok else tokens, now)
    return ok


def admit(tool: str, cost: int = 1, *, store: str = "default") -> None:
    """Spend ``cost`` tokens of ``store``'s allowance for ``tool``; raise ``ValueError`` when short.

    Capacity and refill are both ``mcp.rate_limit_per_minute``: a one-minute
    burst, refilling to full over a minute. ``cost`` is the number of
    paid-capable operations the call can run: a memory-server façade write runs
    one §6.6 assertion per observation or relation it lists, so it is charged
    for each (security review 2026-10-24-2, P26). The spend is all or nothing,
    and a refused call runs none of them.
    """
    rate = get_config().mcp.rate_limit_per_minute
    if rate <= 0:
        return
    cost = max(1, cost)
    if cost > rate:
        raise ValueError(
            f"Rate limit: this {tool} call runs {cost} assertions that can each drive a "
            f"paid model call, more than the {rate} a minute this store allows "
            "(mcp.rate_limit_per_minute). Split it into smaller calls."
        )
    path = shared_state_path(store)
    ok = _admit_shared(path, cost, rate) if path is not None else _admit_local(store, cost, rate)
    if not ok:
        raise ValueError(
            f"Rate limit reached: {tool} can drive a paid model call, and this store "
            f"allows {rate} such calls a minute across all its MCP servers "
            "(mcp.rate_limit_per_minute). Retry after a short wait."
        )


register_reset_hook(reset_limits)
