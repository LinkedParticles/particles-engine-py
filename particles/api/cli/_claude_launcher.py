# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The Claude Code SessionEnd hand-off: stdlib only, run by absolute path.

The installed SessionEnd command runs this file as ``python -I -S <file>
--spool <dir> --log <file> -- <particles> hook session-end-worker --store
<handle>``. It imports nothing from Particles, so it starts in milliseconds
on a loaded laptop. It writes the hook payload to a private spool file,
starts the worker command after ``--`` detached (unless a worker already
holds the spool's lock, in which case that worker takes the file), appends
one ``session-end-handoff`` record to the hook log and exits 0 whatever
happens: a session never waits on, or fails because of, its harvest.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The largest hook payload read from stdin; a Claude Code payload is a few hundred bytes.
MAX_PAYLOAD_BYTES = 1 << 20

#: The lock a running worker holds on its spool directory.
LOCK_NAME = "worker.lock"

#: The suffix of a spool file no worker has taken yet.
PENDING_SUFFIX = ".json"


def private_directory(path: Path) -> None:
    """Create ``path`` owner-only; refuse a linked or shared directory."""
    missing: list[Path] = []
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("linked_spool_directory")
        if not parent.exists():
            missing.append(parent)
    for parent in reversed(missing):
        parent.mkdir(mode=0o700, exist_ok=True)
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ValueError("shared_spool_directory")


def write_spool_file(spool: Path, payload: dict[str, Any]) -> Path:
    """Write one pending spool file atomically, owner-only; return its path."""
    private_directory(spool)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    final = spool / f"{stamp}-{uuid.uuid4().hex}{PENDING_SUFFIX}"
    temporary = spool / f".{final.name}.tmp"
    envelope = {
        "version": 1,
        "received_at": datetime.now(UTC).isoformat(),
        "attempts": 0,
        "payload": payload,
    }
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(envelope, sort_keys=True).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, final)
    finally:
        if temporary.exists():
            temporary.unlink()
    return final


def worker_running(spool: Path) -> bool:
    """Whether a worker holds the spool's lock now; ``False`` when it cannot tell."""
    try:
        import fcntl
    except ImportError:
        return False
    fd = os.open(spool / LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def absolute_path_env() -> dict[str, str]:
    """This process's environment with ``PATH`` cut to its absolute entries.

    An empty or relative entry would look a program up in the working
    directory (security review 2026-11-17-1, RA2). The SDK's own spawns judge
    the entries further (``url_safety.pinned_subprocess_env``); this file
    imports nothing from Particles, so it drops only what names a relative
    directory.
    """
    env = dict(os.environ)
    if "PATH" in env:
        kept = [p for p in env["PATH"].split(os.pathsep) if os.path.isabs(p)]
        env["PATH"] = os.pathsep.join(kept) if kept else os.defpath
    return env


def start_worker(argv: list[str], cwd: Path) -> None:
    """Start ``argv`` in its own session, detached from the hook's streams."""
    subprocess.Popen(  # noqa: S603 — the installer wrote this argv; nothing here comes from the payload
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=cwd,
        start_new_session=True,
        close_fds=True,
        env=absolute_path_env(),
    )


def append_log(log: Path, record: dict[str, Any]) -> None:
    """Append one JSONL record to the hook log, owner-only; never raises."""
    try:
        private_directory(log.parent)
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001, S110 — the log must never break a hook
        pass


def parse_args(argv: list[str]) -> tuple[Path, Path, list[str]]:
    """``(spool dir, hook log, worker argv)`` from ``--spool D --log F -- WORKER…``."""
    if "--" not in argv:
        raise ValueError("no worker command after --")
    split = argv.index("--")
    own, worker = argv[:split], argv[split + 1 :]
    options = dict(zip(own[0::2], own[1::2], strict=False))
    if not worker or "--spool" not in options or "--log" not in options:
        raise ValueError("usage: --spool DIR --log FILE -- WORKER ...")
    return Path(options["--spool"]), Path(options["--log"]), worker


def _store_of(worker: list[str]) -> str:
    return worker[worker.index("--store") + 1] if "--store" in worker[:-1] else "default"


def main(argv: list[str]) -> int:
    """Hand the session off; always 0."""
    started = time.monotonic()
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": "session-end-handoff",
    }
    log: Path | None = None
    try:
        spool, log, worker = parse_args(argv)
        record["store"] = _store_of(worker)
        raw = sys.stdin.buffer.read(MAX_PAYLOAD_BYTES + 1)
        if len(raw) > MAX_PAYLOAD_BYTES:
            raise ValueError("hook payload over 1 MiB")
        loaded = json.loads(raw) if raw.strip() else {}
        payload = loaded if isinstance(loaded, dict) else {}
        record["session_id"] = payload.get("session_id")
        record["source"] = payload.get("reason") or payload.get("source")
        write_spool_file(spool, payload)
        if worker_running(spool):
            record["worker"] = "running"
        else:
            start_worker(worker, spool)
            record["worker"] = "started"
        record["outcome"] = "handed_off"
    except Exception as exc:  # noqa: BLE001 — the hand-off degrades to exit 0
        record["outcome"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    record["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
    if log is not None:
        append_log(log, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
