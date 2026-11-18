# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Private atomic hook metadata and kernel-released advisory locks."""

from __future__ import annotations

import contextlib
import json
import os
import stat
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def private_dir(path: Path) -> None:
    """Create private parents; refuse symlink or shared state directories."""
    missing = []
    for parent in (path, *path.parents):
        if parent.exists():
            break
        missing.append(parent)
    for parent in reversed(missing):
        parent.mkdir(mode=0o700, exist_ok=True)
    path.lstat()  # a path removed since the walk fails here, as it always did
    if private_dir_refusal(path) is not None:
        raise ValueError("hook_state_directory_not_private")


def private_dir_refusal(path: Path) -> str | None:
    """Say why ``private_dir`` would refuse ``path``, without creating anything.

    A missing directory is created private, so only an existing one is refused.
    """
    try:
        st = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        return f"{path} is not a directory (a link is refused too)"
    if st.st_uid != os.getuid():
        return f"{path} belongs to another account"
    if st.st_mode & 0o077:
        mode = stat.S_IMODE(st.st_mode)
        return (
            f"{path} is open to other accounts (mode {mode:04o}); run `chmod 700 {path}` and retry"
        )
    return None


def read_json(path: Path, default: Any = None) -> Any:
    """Read bounded private regular metadata without following a file link."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return default
    with os.fdopen(fd, "rb") as stream:
        st = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(st.st_mode)
            or st.st_uid != os.getuid()
            or st.st_mode & 0o077
            or st.st_size > 4 * 1024 * 1024
        ):
            raise ValueError("hook_state_file_not_private")
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    """Durably replace owner-only metadata; no plaintext source payloads belong here."""
    private_dir(path.parent)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temp.unlink(missing_ok=True)


def _lock_file_is_private(st: os.stat_result) -> bool:
    """One regular file, one link, this account's, closed to everyone else."""
    return (
        stat.S_ISREG(st.st_mode)
        and st.st_nlink == 1
        and st.st_uid == os.getuid()
        and not st.st_mode & 0o077
    )


@contextlib.contextmanager
def advisory_lock(path: Path, *, blocking: bool = False) -> Iterator[bool]:
    """Hold a process lock; contention returns False and crashes release in kernel."""
    try:
        import fcntl
    except ImportError:
        yield False
        return
    private_dir(path.parent)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    acquired = False
    try:
        if not _lock_file_is_private(os.fstat(fd)):
            raise ValueError("hook_lock_not_private")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            acquired = True
        except BlockingIOError:
            pass
        yield acquired
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def installer_lock_path() -> Path:
    """One cross-harness installer lock beside the shared identity lock."""
    from particles.config import get_config

    return (
        Path(get_config().codex.identity_lock_path).expanduser().with_name("harness-installer.lock")
    )


def installer_lock_refusal() -> str | None:
    """Say why an installer could not take its lock, without creating anything.

    Both installers ask this before they plan, so a dry run reports what the
    real run would refuse instead of passing over the lock it skips.
    """
    path = installer_lock_path()
    refusal = private_dir_refusal(path.parent)
    if refusal is not None:
        return refusal
    try:
        st = path.lstat()
    except FileNotFoundError:
        return None
    if not _lock_file_is_private(st):
        return f"{path} is not a private file of this account; remove it and retry"
    return None
