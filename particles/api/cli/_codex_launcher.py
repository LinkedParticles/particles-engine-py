# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only installed hook entry point; hand off before importing Particles.

This file is executed by absolute path, never as a package module. Payload
paths are ignored. All destination paths and commands are installer pins.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def private_directory(path: Path) -> None:
    """Create every missing state component privately, not only the leaf."""
    missing: list[Path] = []
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("linked_state_directory")
        if not parent.exists():
            missing.append(parent)
    for parent in reversed(missing):
        parent.mkdir(mode=0o700, exist_ok=True)
    st = path.lstat()
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ValueError("shared_state_directory")


def atomic_metadata(path: Path, value: Any) -> None:
    """Write only private metadata, refusing shared or linked state directories."""
    private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(value, sort_keys=True).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_metadata(path: Path) -> Any:
    """Read a bounded regular private file; never follow a pending-hint link."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        st = os.fstat(stream.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise ValueError("invalid_metadata_permissions")
        if st.st_size > 4096:
            raise ValueError("oversized_metadata")
        return json.load(stream)


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


def child_env(token_file: str) -> dict[str, str]:
    """Give a thin child exactly its pinned credential source."""
    env = {
        key: value
        for key, value in absolute_path_env().items()
        if key
        not in {
            "PARTICLES_ENGINE_TOKEN",
            "PARTICLES_ENGINE_TOKEN_FILE",
            "PARTICLES_API_KEY",
        }
    }
    env["PARTICLES_ENGINE_TOKEN"] = ""
    env["PARTICLES_ENGINE_TOKEN_FILE"] = token_file
    return env


def handoff(args: argparse.Namespace, event: dict[str, Any]) -> None:
    """Durably hint an end, detach capture; independently start repair on starts."""
    identity = event.get("session_id")
    if not isinstance(identity, str) or str(uuid.UUID(identity)) != identity:
        raise ValueError("invalid_session_id")
    cwd = event.get("cwd")
    if not isinstance(cwd, str) or not Path(cwd).is_absolute() or len(cwd) > 2048:
        raise ValueError("invalid_event_cwd")
    if event.get("hook_event_name") != args.event:
        raise ValueError("wrong_hook_event")
    root = Path(args.state) / "stores" / args.store_id
    private_directory(root)
    thin = getattr(args, "thin", False)
    capture_token = getattr(args, "capture_token_file", None)
    can_capture = not thin or capture_token is not None
    if args.event == "SessionEnd" and can_capture:
        pending = root / "pending" / f"{identity}.json"
        now = time.time()
        try:
            previous = read_metadata(pending)
        except FileNotFoundError:
            previous = {}
        hint = {
            "session_id": identity,
            "store_id": args.store_id,
            "home": args.home,
            "first_attempt": previous.get("first_attempt", now),
            "last_attempt": now,
            "failure": "pending",
            "cwd": cwd,
        }
        if len(json.dumps(hint).encode()) > 4096:
            raise ValueError("hint_oversized")
        atomic_metadata(pending, hint)
    worker = [
        args.command,
        "hook",
        "codex-worker",
        "--store",
        args.store,
        "--session",
        identity,
        "--cwd",
        cwd,
    ]
    if args.event == "SessionStart":
        worker.append("--catch-up")
    if thin:
        worker.extend(["--destination", args.store_id])
    if can_capture:
        log_fd = os.open(
            root / "worker.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
        )
        try:
            info = os.fstat(log_fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise ValueError("unsafe_worker_log")
            subprocess.Popen(
                worker,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log_fd,
                close_fds=True,
                start_new_session=True,
                env=child_env(capture_token)
                if thin and capture_token is not None
                else absolute_path_env(),
            )
            if thin:
                atomic_metadata(
                    root / "handoff.json",
                    {
                        "at": time.time(),
                        "event": args.event,
                        "result": "worker launched",
                    },
                )
        finally:
            os.close(log_fd)

    if args.event == "SessionStart":
        reason = event.get("source")
        if not isinstance(reason, str) or reason not in {
            "startup",
            "clear",
            "compact",
            "fork",
            "resume",
        }:
            reason = "unknown"
        result = subprocess.run(
            [
                args.command,
                "hook",
                "codex-recall",
                "--reason",
                reason,
                "--store",
                args.store,
                "--session",
                identity,
                "--cwd",
                cwd,
                *(["--destination", args.store_id] if thin else []),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=child_env(args.recall_token_file) if thin else absolute_path_env(),
            timeout=args.deadline,
            check=False,
        )
        if result.stderr or result.returncode:
            print(
                "particles codex: recall child reported failure; delivery unknown", file=sys.stderr
            )
        if result.returncode == 0:
            sys.stdout.buffer.write(result.stdout)


def main() -> None:
    """Fail softly for the host, with a reason-only diagnostic on stderr."""
    parser = argparse.ArgumentParser()
    for name in ("event", "state", "home", "store-id", "store", "command"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--deadline", type=float, default=10)
    parser.add_argument("--thin", action="store_true")
    parser.add_argument("--capture-token-file")
    parser.add_argument("--recall-token-file")
    args = parser.parse_args()
    if args.thin and args.recall_token_file is None:
        parser.error("thin mode requires a recall token file pin")
    try:
        if not re.fullmatch(r"[a-f0-9]{64}", args.store_id):
            raise ValueError("invalid_store_id")
        if not os.path.isabs(args.command):
            # The installer pins an absolute command; a relative one would be
            # looked up on PATH from the session's directory.
            raise ValueError("relative_command")
        payload = sys.stdin.buffer.read(65537)
        if len(payload) > 65536:
            raise ValueError("event_oversized")
        event = json.loads(payload)
        if not isinstance(event, dict):
            raise ValueError("invalid_event")
        handoff(args, event)
    except Exception as exc:  # hook failure must never block the session
        # Exception text may contain an event path, OS environment or source;
        # disclose only a stable class, never the payload or raw exception.
        print(f"particles codex: handoff failed ({type(exc).__name__})", file=sys.stderr)


if __name__ == "__main__":
    main()
