# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Read one local file without being redirected to another (security review N22, N24, D23, D25).

Every path the corpus reads from the local disk on its own initiative (a vault,
web-clipper or project walk, a rule-file intake, the nightly refresh of a
``file://`` entry) was chosen by a check on the *name*, and then read by a call
that re-walked the name. A symlink planted in a walked folder, or a parent
directory swapped for one between the check and the read, therefore delivered
the bytes of a file the check never admitted.

:func:`read_local_file` closes that by reading through one file descriptor and
judging that descriptor, never the name:

* the open refuses a symlinked final component unless the caller follows links
  (``O_NOFOLLOW``), and never blocks on a FIFO (``O_NONBLOCK``);
* the descriptor must be a regular file, so a device or a pipe is refused;
* its size is checked against the caller's cap before a byte is read, and the
  read itself stops one byte past the cap, so a file that grows mid-read is
  refused too;
* the descriptor's own path (``F_GETPATH`` on macOS, ``/proc/self/fd`` on
  Linux) is what the caller's containment predicate judges, so a parent swapped
  after any earlier check is caught here.

A refusal raises :class:`LocalReadRefused`, an ``OSError``, so every caller's
existing "the file is unreadable" branch handles it unchanged.
"""

from __future__ import annotations

import errno
import os
import stat as stat_module
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: The read block size. The last block may run one byte past the cap.
_CHUNK = 1024 * 1024

#: ``MAXPATHLEN`` on macOS, the buffer ``F_GETPATH`` fills.
_MAXPATHLEN = 1024


class LocalReadRefused(OSError):
    """A local read refused on a symlink, a non-regular file, the size cap, or containment."""


@dataclass(frozen=True)
class LocalRead:
    """What one :func:`read_local_file` call read."""

    content: bytes
    stat: os.stat_result
    """``fstat`` of the descriptor the bytes came from."""
    real_path: Path
    """The descriptor's own path, with every symlink resolved."""


def descriptor_path(fd: int, path: Path, st: os.stat_result) -> Path:
    """The path of the file ``fd`` has open, resolved; raise when it cannot be named.

    Asks the kernel where it can (``F_GETPATH``, ``/proc/self/fd``). Elsewhere it
    falls back to resolving ``path`` and accepts the result only when it names
    the same inode as the descriptor, which leaves a race the kernel answer does
    not have, and so is the last resort rather than the method.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - not a POSIX host
        fcntl = None  # type: ignore[assignment]
    get_path = getattr(fcntl, "F_GETPATH", None) if fcntl is not None else None
    if fcntl is not None and get_path is not None:
        raw = fcntl.fcntl(fd, get_path, b"\0" * _MAXPATHLEN)
        return Path(os.fsdecode(raw.split(b"\0", 1)[0]))
    proc = Path(f"/proc/self/fd/{fd}")
    if proc.parent.is_dir():
        return Path(os.readlink(proc))
    real = Path(os.path.realpath(path))  # pragma: no cover - neither macOS nor Linux
    other = os.stat(real)  # pragma: no cover
    if (other.st_dev, other.st_ino) == (st.st_dev, st.st_ino):  # pragma: no cover
        return real
    raise LocalReadRefused(  # pragma: no cover
        errno.EXDEV, "cannot confirm which file was opened", str(path)
    )


def read_local_file(
    path: Path,
    *,
    follow_symlinks: bool = False,
    max_bytes: int | None = None,
    allowed: Callable[[Path], bool] | None = None,
) -> LocalRead:
    """Read ``path`` through one descriptor, judged as described in the module docstring.

    Args:
        path: The file to read.
        follow_symlinks: Whether a symlinked final component may be opened.
            Off, the open fails on one. Either way ``allowed`` judges the file
            actually opened.
        max_bytes: Refuse a file larger than this. ``None`` reads any size.
        allowed: Containment predicate over the opened file's resolved path.

    Raises:
        LocalReadRefused: a symlink with ``follow_symlinks`` off, a non-regular
            file, a file over ``max_bytes``, or a path ``allowed`` rejects.
        OSError: the file is missing or cannot be opened.
    """
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    if not follow_symlinks:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP and not follow_symlinks:
            raise LocalReadRefused(errno.ELOOP, "refusing to follow a symlink", str(path)) from exc
        raise
    try:
        st = os.fstat(fd)
        if not stat_module.S_ISREG(st.st_mode):
            raise LocalReadRefused(errno.EINVAL, "not a regular file", str(path))
        if max_bytes is not None and st.st_size > max_bytes:
            raise LocalReadRefused(
                errno.EFBIG, f"{st.st_size} bytes is over the {max_bytes}-byte cap", str(path)
            )
        real = descriptor_path(fd, path, st)
        if allowed is not None and not allowed(real):
            raise LocalReadRefused(
                errno.EACCES, f"opened file {real} is outside the root", str(path)
            )
        limit = max_bytes + 1 if max_bytes is not None else None
        chunks: list[bytes] = []
        total = 0
        while limit is None or total < limit:
            block = os.read(fd, _CHUNK if limit is None else min(_CHUNK, limit - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
        if max_bytes is not None and total > max_bytes:
            raise LocalReadRefused(
                errno.EFBIG, f"grew past the {max_bytes}-byte cap while read", str(path)
            )
        return LocalRead(content=b"".join(chunks), stat=st, real_path=real)
    finally:
        os.close(fd)


def within(root: Path) -> Callable[[Path], bool]:
    """A containment predicate: the opened file must resolve under ``root``."""
    resolved = root.resolve()
    return lambda real: real.is_relative_to(resolved)
