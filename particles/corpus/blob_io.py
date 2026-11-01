# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Read and write one blob without being redirected (security review 2026-10-25, Q4).

A blob lives at ``<root>/<hh>/<hash>``. With the compiled defaults and no user
config, ``<root>`` is ``./corpus_blobs`` of whatever directory a verb or
``engine serve`` runs in, which a cloned repository can ship. A symlink planted
at a blob's place, at its shard directory or at the root itself then sent the
write to a file the depositor never named and the read (including the open
``GET /corpus/blob`` route) to a host file.

Every blob access therefore walks the tree by descriptor:

* the root is refused when it is a symlink, and when it is relative (so it
  resolves against the working directory), when any component of it is;
* the root and the shard are opened ``O_DIRECTORY | O_NOFOLLOW``, and the blob
  is opened relative to the shard's descriptor with ``O_NOFOLLOW``, so no name
  is re-walked after it was judged;
* a write creates a fresh temporary file ``O_CREAT | O_EXCL`` in the shard,
  which never writes through anything already there, and links it to the
  blob's name only once its bytes are complete; it accepts an existing entry
  only when it is a regular file whose bytes hash to the name; a regular file
  with other bytes is unlinked and the blob linked in its place, and anything
  else is refused;
* a read accepts a regular file only, and returns its bytes only when they hash
  to the blob's name, so a planted file (hard link included) never stands in
  for a blob; the hash is taken in blocks before the file is buffered, so a
  planted file of any size is refused without being read into memory.

A refusal raises :class:`BlobRefused`, a ``FileNotFoundError``, so every
caller's missing-blob branch handles it unchanged: a blob that cannot be read
safely is not there.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import secrets
import stat as stat_module
from pathlib import Path

_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _NOFOLLOW | _CLOEXEC


class BlobRefused(FileNotFoundError):
    """A blob path refused on a symlink, a non-regular file, or bytes that miss their hash."""


def check_blob_root(root: Path) -> None:
    """Refuse a blob root that is a symlink, or a relative root with a symlinked component.

    An absolute root is the operator's choice, so only its last component is
    judged. A relative one resolves against the working directory, which
    nobody may have chosen for its contents, so every component of it is.
    """
    parts = root.parts
    candidates = (
        [root] if root.is_absolute() else [Path(*parts[: i + 1]) for i in range(len(parts))]
    )
    for candidate in candidates:
        if candidate.is_symlink():
            raise BlobRefused(
                errno.ELOOP, "refusing a blob directory reached through a symlink", str(candidate)
            )


def _open_dir(name: str | Path, dir_fd: int | None = None) -> int:
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
            raise BlobRefused(
                exc.errno,
                "refusing a blob directory that is a symlink or not a directory",
                str(name),
            ) from exc
        raise


def write_blob(root: Path, content_hash: str, content: bytes) -> Path:
    """Create ``<root>/<hh>/<hash>`` holding ``content`` unless the blob is already there.

    The bytes are written to a fresh temporary name in the shard and only then
    linked to the blob's name, so the name only ever holds a complete blob. A
    concurrent writer of the same bytes therefore never finds a partial file
    there to judge as other bytes and remove, and a write that returns leaves
    its blob readable whatever another writer did meanwhile (security review
    2026-10-27-1, G53).

    Returns the blob's path. The caller has validated ``content_hash``.

    Raises:
        BlobRefused: the root, the shard or the blob's place is a symlink or
            not what it should be.
    """
    check_blob_root(root)
    shard = content_hash[:2]
    path = root / shard / content_hash
    root.mkdir(parents=True, exist_ok=True)
    root_fd = _open_dir(root)
    try:
        with contextlib.suppress(FileExistsError):
            os.mkdir(shard, 0o777, dir_fd=root_fd)
        shard_fd = _open_dir(shard, root_fd)
        try:
            if _exists(shard_fd, content_hash) and _holds_blob(shard_fd, content_hash, path):
                return path
            temp = f".{content_hash}.{secrets.token_hex(8)}.tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC
            fd = os.open(temp, flags, 0o666, dir_fd=shard_fd)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                _install(shard_fd, temp, content_hash, path)
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(temp, dir_fd=shard_fd)
        finally:
            os.close(shard_fd)
    finally:
        os.close(root_fd)
    return path


def _exists(shard_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=shard_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _install(shard_fd: int, temp: str, content_hash: str, path: Path) -> None:
    """Link the complete file at ``temp`` to the blob's name in the shard.

    A name already holding the blob is kept. A regular file there with other
    bytes (a planted placeholder, or a torn write from before this rule) has
    only its name removed, and the link is made once more (security review
    2026-10-25-2, R40). A symlink or non-regular file there is refused.
    """
    for attempt in range(2):
        try:
            os.link(temp, content_hash, src_dir_fd=shard_fd, dst_dir_fd=shard_fd)
            return
        except FileExistsError:
            if _holds_blob(shard_fd, content_hash, path):
                return
            if attempt == 0:
                os.unlink(content_hash, dir_fd=shard_fd)
        except OSError as exc:
            if exc.errno not in _NO_HARD_LINKS:
                raise
            # A filesystem without hard links: an atomic rename still
            # installs only the complete file.
            if _exists(shard_fd, content_hash) and _holds_blob(shard_fd, content_hash, path):
                return
            os.rename(temp, content_hash, src_dir_fd=shard_fd, dst_dir_fd=shard_fd)
            return
    raise BlobRefused(errno.EEXIST, "another file took this blob's place", str(path))


_NO_HARD_LINKS = frozenset(
    {errno.EPERM, errno.EXDEV, getattr(errno, "ENOTSUP", errno.EPERM), errno.EOPNOTSUPP}
)


def _holds_blob(shard_fd: int, content_hash: str, path: Path) -> bool:
    """Whether the entry at ``content_hash`` in the shard is a regular file of those bytes.

    Opened like a read (``O_NOFOLLOW``, ``O_NONBLOCK``, judged on the
    descriptor), so a symlink or a FIFO at the blob's place is refused rather
    than followed or waited on.

    Raises:
        BlobRefused: a symlink or non-regular file holds the blob's place.
    """
    flags = os.O_RDONLY | _NOFOLLOW | _CLOEXEC | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(content_hash, flags, dir_fd=shard_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise BlobRefused(
                errno.EEXIST, "a symlink or non-regular file holds this blob's place", str(path)
            ) from exc
        raise
    with os.fdopen(fd, "rb") as handle:
        if not stat_module.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise BlobRefused(
                errno.EEXIST, "a symlink or non-regular file holds this blob's place", str(path)
            )
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == content_hash


def read_blob(root: Path, content_hash: str) -> bytes:
    """Return the bytes of ``<root>/<hh>/<hash>``, read as the module docstring describes.

    The caller has validated ``content_hash``.

    Raises:
        FileNotFoundError: no blob exists for ``content_hash``.
        BlobRefused: a symlink or non-regular file on the path, or bytes that
            do not hash to ``content_hash``.
    """
    check_blob_root(root)
    path = root / content_hash[:2] / content_hash
    root_fd = _open_dir(root)
    try:
        shard_fd = _open_dir(content_hash[:2], root_fd)
        try:
            flags = os.O_RDONLY | _NOFOLLOW | _CLOEXEC | getattr(os, "O_NONBLOCK", 0)
            try:
                fd = os.open(content_hash, flags, dir_fd=shard_fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise BlobRefused(errno.ELOOP, "refusing a symlinked blob", str(path)) from exc
                raise
        finally:
            os.close(shard_fd)
    finally:
        os.close(root_fd)
    with os.fdopen(fd, "rb") as handle:
        if not stat_module.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise BlobRefused(errno.EINVAL, "a blob must be a regular file", str(path))
        # Hash the file in blocks before buffering any of it, so a planted
        # file that cannot be the blob costs no memory of the planter's
        # choosing (security review 2026-10-30, AB15). The buffered read is hashed
        # again, so a file swapped between the two passes is still refused.
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
        if digest.hexdigest() != content_hash:
            raise BlobRefused(
                errno.EIO, "the file at this blob's place does not match its hash", str(path)
            )
        handle.seek(0)
        data = handle.read()
    if hashlib.sha256(data).hexdigest() != content_hash:
        raise BlobRefused(
            errno.EIO, "the file at this blob's place does not match its hash", str(path)
        )
    return data
