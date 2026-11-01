# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Where a projection manifest may read and write (security review 2026-10-24-2, P3).

A manifest is data, and a repository can ship one. Its ``output:`` field and its
mechanical blocks' paths therefore stay inside the manifest's own tree unless
the operator says otherwise on the command line:

* a manifest-named path must be relative, with no ``..`` component and no
  leading ``~``;
* it must also *resolve* inside the containment root, so a symlink inside the
  tree cannot lead out of it. The root is the repository that holds the
  manifest (the nearest ancestor with a ``.git`` entry), or the manifest's own
  directory when there is none. The home directory and its ancestors are never
  a root, so a dotfiles repository at ``~`` does not widen it to all of home,
  and a manifest kept directly in one of them has no root at all: every path
  it names is refused (security review 2026-10-25-1, Q52);
* a block is read through :func:`particles.corpus.local_read.read_local_file`:
  one descriptor, no symlinked final component, a regular file, the
  ``local_files.max_bytes`` cap, and the containment check on the file actually
  opened;
* every file the projection writes is opened with ``O_NOFOLLOW``, so a symlink
  committed where the document or its snapshot goes is refused, not written
  through. A write inside the tree walks down from the root one directory at a
  time, each opened with ``O_NOFOLLOW``, and checks the directory it opened is
  still inside the root, so a parent swapped for a symlink while the render
  ran is refused too (security review 2026-10-25-1, Q29);
* a refusal names the field and the reason only. It never repeats the
  manifest's own text, which a reader (often an agent) would take as output,
  and never names the flag that lifts the check (Q51).

``allow_outside`` (the CLI's ``--allow-outside-paths``) lifts the first three
for a manifest the operator wrote. It never lifts the symlink refusal on write.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from pathlib import Path, PurePath

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from particles.config import get_config
from particles.corpus.local_read import LocalReadRefused, read_local_file


class ManifestPathError(ValueError):
    """A manifest-named path is absolute, climbs out, or resolves outside its tree."""


def containment_root(base_dir: Path) -> Path | None:
    """The tree a manifest in ``base_dir`` may read from and write into.

    The nearest ancestor of ``base_dir`` (itself included) holding a ``.git``
    entry, resolved, provided it lies strictly below the home directory;
    otherwise ``base_dir`` itself, resolved. ``None`` when ``base_dir`` is the
    home directory or one of its ancestors, which are never a root (Q52).
    """
    start = Path(os.path.realpath(base_dir))
    home = Path(os.path.realpath(Path.home()))
    if start == home or start in home.parents:
        return None
    for candidate in (start, *start.parents):
        if candidate == home or candidate in home.parents:
            break
        if os.path.lexists(candidate / ".git"):
            return candidate
    return start


def _lexical_refusal(raw: str) -> str | None:
    if raw.startswith("~"):
        return "starts with '~'"
    pure = PurePath(raw)
    if pure.is_absolute():
        return "is absolute"
    if ".." in pure.parts:
        return "contains '..'"
    return None


def resolve_manifest_path(
    raw: str, *, against: Path, root: Path | None, allow_outside: bool, what: str
) -> Path:
    """Resolve a path a manifest names, refusing one that leaves ``root``.

    ``against`` is the directory a relative path is joined to. With
    ``allow_outside`` the path is only ``~``-expanded and joined. A ``root`` of
    ``None`` (:func:`containment_root` of the home directory) refuses every
    path. A refusal names ``what`` and the reason, never ``raw`` (Q51).

    Raises:
        ManifestPathError: the path is absolute, starts with ``~``, contains
            ``..``, or resolves (symlinks followed) outside ``root``.
    """
    if allow_outside:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else against / path
    reason = _lexical_refusal(raw)
    if reason is not None:
        raise ManifestPathError(
            f"the manifest's {what} path {reason}; a manifest's paths stay inside its own tree"
        )
    if root is None:
        raise ManifestPathError(
            f"the manifest's {what} path was refused: a manifest kept in the home "
            "directory or above it has no tree to contain its paths"
        )
    path = against / raw
    real = Path(os.path.realpath(path))
    if not real.is_relative_to(Path(os.path.realpath(root))):
        raise ManifestPathError(f"the manifest's {what} path resolves outside the manifest's tree")
    return path


def read_block_text(raw: str, *, base_dir: Path, allow_outside: bool) -> str:
    """Read a mechanical block's fragment under the rules in the module docstring.

    Raises:
        ManifestPathError: the block path is refused before any read.
        FileNotFoundError: the fragment does not exist.
        LocalReadRefused: the open found a symlink, a non-regular file, a file
            over the cap, or a file outside the root.
    """
    root = containment_root(base_dir)
    path = resolve_manifest_path(
        raw, against=base_dir, root=root, allow_outside=allow_outside, what="block"
    )
    real_root = Path(os.path.realpath(root if root is not None else base_dir))
    read = read_local_file(
        path,
        follow_symlinks=allow_outside,
        max_bytes=get_config().local_files.max_bytes,
        allowed=None if allow_outside else (lambda real: real.is_relative_to(real_root)),
    )
    return read.content.decode("utf-8")


def read_text_no_follow(path: Path) -> str:
    """Read a file the projection is about to rewrite, refusing a symlink."""
    return read_local_file(path, max_bytes=get_config().local_files.max_bytes).content.decode(
        "utf-8"
    )


def write_root(base_dir: Path) -> Path:
    """The root a file the projection writes beside the manifest is contained in.

    The manifest's own tree, or its directory when it has none (a manifest in
    the home directory): the snapshot and the corpus bundle are named by the
    projection, not by the manifest, so they are always written.
    """
    return containment_root(base_dir) or Path(os.path.realpath(base_dir))


#: Whether this platform can walk a directory chain by descriptor.
_DIR_FD_WRITES = (
    os.open in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
)


#: A flag that opens a directory for path lookups only, without read permission.
_SEARCH_ONLY = getattr(os, "O_PATH", 0) or getattr(os, "O_SEARCH", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _symlink_refusal(path: Path, exc: OSError | None = None) -> LocalReadRefused:
    refused = LocalReadRefused(errno.ELOOP, "refusing to write through a symlink", str(path))
    if exc is not None:
        refused.__cause__ = exc
    return refused


def _descriptor_path(fd: int) -> Path | None:
    """The path the kernel reports for an open directory, where it reports one."""
    get_path = getattr(fcntl, "F_GETPATH", None) if fcntl is not None else None
    try:
        if get_path is not None:
            raw = fcntl.fcntl(fd, get_path, b"\0" * 1024)
            return Path(os.fsdecode(raw.split(b"\0", 1)[0]))
        proc = Path(f"/proc/self/fd/{fd}")
        if proc.is_symlink():
            return Path(os.readlink(proc))
    except OSError:
        return None
    return None


def _open_root_no_follow(real_root: Path, path: Path, flags: int) -> int:
    """A descriptor on ``real_root``, reached from ``/`` without following a link.

    ``real_root`` is already resolved, so no component of it is a link unless
    one was swapped in after the resolve. Opening it by name would follow that
    link, and only the kernel-path recheck would catch it, which a platform
    with neither ``F_GETPATH`` nor ``/proc`` cannot run (security review
    2026-10-25-2, critic item on Q29). An ancestor this user may search but not
    read cannot be opened for reading, so it is opened search-only (``O_PATH``
    on Linux, ``O_SEARCH`` elsewhere), which still anchors the next step
    (security review 2026-10-27-1, Z16). Where the platform has neither, only
    the stretch below the last ancestor opened is resolved by name, relative to
    that ancestor's descriptor, and the root is then held to the kernel-path
    recheck; a platform that cannot run the recheck refuses the write instead
    (security review 2026-10-28-1, O67).
    """
    try:
        fd = os.open(real_root.anchor or os.sep, flags)
    except PermissionError:
        return _open_rest_by_name(None, real_root, real_root, path, flags)
    parts = real_root.parts[1:]
    for index, part in enumerate(parts):
        try:
            try:
                child = os.open(part, flags | os.O_NOFOLLOW, dir_fd=fd)
            except PermissionError:
                if not _SEARCH_ONLY or index == len(parts) - 1:
                    raise
                child = os.open(part, _SEARCH_ONLY | _DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        except PermissionError:
            try:
                return _open_rest_by_name(fd, Path(*parts[index:]), real_root, path, flags)
            finally:
                os.close(fd)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise _symlink_refusal(path, exc) from exc
            raise
        os.close(fd)
        fd = child
    return fd


def _open_rest_by_name(
    dir_fd: int | None, rest: Path, real_root: Path, path: Path, flags: int
) -> int:
    """Open ``rest`` by name (below ``dir_fd`` when given), held to the kernel-path recheck.

    A component of ``rest`` could be a link swapped in after the resolve, so
    the descriptor opened must report ``real_root`` itself; where the platform
    reports no path for a descriptor, the write is refused.
    """
    fd = os.open(rest, flags | os.O_NOFOLLOW, dir_fd=dir_fd)
    opened = _descriptor_path(fd)
    if opened is None or Path(os.path.realpath(opened)) != real_root:
        os.close(fd)
        raise _symlink_refusal(path)
    return fd


def _open_contained_parent(path: Path, root: Path) -> int:
    """A descriptor on ``path``'s parent, reached from ``root`` without following a link.

    Missing directories are created on the way down. Each component is opened
    ``O_DIRECTORY | O_NOFOLLOW`` relative to the one above it, so a directory
    swapped for a symlink after the containment check stops the walk; the
    directory finally opened must still report a path inside ``root``.
    """
    real_root = Path(os.path.realpath(root))
    try:
        relative = Path(os.path.realpath(path.parent)).relative_to(real_root)
    except ValueError:
        raise _symlink_refusal(path) from None
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    fd = _open_root_no_follow(real_root, path, flags)
    try:
        for part in relative.parts:
            with contextlib.suppress(FileExistsError):
                os.mkdir(part, 0o777, dir_fd=fd)
            try:
                child = os.open(part, flags | os.O_NOFOLLOW, dir_fd=fd)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise _symlink_refusal(path, exc) from exc
                raise
            os.close(fd)
            fd = child
        opened = _descriptor_path(fd)
        if opened is not None and not Path(os.path.realpath(opened)).is_relative_to(real_root):
            raise _symlink_refusal(path)
    except BaseException:
        os.close(fd)
        raise
    return fd


def write_text_no_follow(path: Path, text: str, *, root: Path | None = None) -> None:
    """Write ``text`` to ``path`` without following a symlink planted there.

    With ``root`` the write is also contained: ``path``'s parent is reached
    from ``root`` one ``O_NOFOLLOW`` directory at a time (creating what is
    missing), so a directory replaced by a symlink after the containment check
    is refused rather than written through (Q29). Without ``root`` (an
    operator's own output path) only the final component is checked.

    The file is opened without truncating it and refused when it has more than
    one link: a hard link planted at the output name would otherwise carry the
    write to a file outside ``root`` (security review 2026-10-26-1, W28). It is
    refused too when it is not a regular file, and a FIFO is refused without
    blocking on it (2026-10-27-1, W28).

    Raises:
        LocalReadRefused: ``path`` is a symlink or a hard link to another
            name, or with ``root`` a directory on the way to it is one or has
            left ``root`` (an ``OSError``, as the open's own failures are).
    """
    # Non-blocking, so a FIFO at the output name is refused at once (ENXIO
    # with no reader) instead of holding the run until one appears; any other
    # file that is not a regular one is refused below (2026-10-27-1, W28).
    nonblock = getattr(os, "O_NONBLOCK", 0)
    flags = os.O_WRONLY | os.O_CREAT | nonblock | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd: int | None = None
    if root is not None and _DIR_FD_WRITES:
        parent_fd = _open_contained_parent(path, root)
    try:
        if parent_fd is None:
            fd = os.open(path, flags, 0o644)
        else:
            fd = os.open(path.name, flags, 0o644, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _symlink_refusal(path, exc) from exc
        raise
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise LocalReadRefused(
                errno.EINVAL, "refusing to write to a non-regular file", str(path)
            )
        if st.st_nlink > 1:
            raise LocalReadRefused(errno.EMLINK, "refusing to write through a hard link", str(path))
        if nonblock:
            os.set_blocking(fd, True)
        os.ftruncate(fd, 0)
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(text)
