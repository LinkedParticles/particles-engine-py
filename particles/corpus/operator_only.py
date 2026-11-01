# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Whether a file the local refresh trusts is one only the operator can write.

The property, stated once: the file is the operator's own and no other user can
change it, and no user but the operator and root can change any directory its
name resolves through. Every check in the refresh that asks "may another user
write this" (the store and its SQLite sidecars, the config file, a rule file
under a recorded root, and the counts that predict the read) calls
:func:`writable_only_by_operator`.

It is split into a gather step and a pure decision (D2).
:func:`gather_write_facts` does the I/O and returns plain records, one per file
and directory; :func:`judge_write_facts` decides from them alone, so every
mechanism is a row of facts a test can feed it on any host.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import functools
import os
import stat as stat_module
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from particles.config import is_broad_root

#: Mode bits that let a user other than the owner write a file.
WRITABLE_BY_OTHERS = stat_module.S_IWGRP | stat_module.S_IWOTH

#: ``ACL_TYPE_EXTENDED``, the one ACL type macOS keeps.
_ACL_TYPE_EXTENDED = 0x00000100

#: Permissions a macOS ACL entry can allow that change a file's bytes, or on a
#: directory which files it holds (``add_file`` and ``add_subdirectory`` print
#: as ``write`` and ``append`` there), or that take the ownership or the ACL
#: that would. ``delete``, ``readattr``, ``writeattr`` and ``writeextattr`` change
#: neither the bytes nor which file a name opens, so they are not counted.
_ACL_WRITE_PERMS = frozenset({"write", "append", "delete_child", "writesecurity", "chown"})

#: ``MNT_IGNORE_OWNERSHIP`` (also spelt ``MNT_UNKNOWNPERMISSIONS``): a macOS
#: volume mounted with ownership ignored reports every file as the caller's and
#: lets every user act as its owner.
_MNT_IGNORE_OWNERSHIP = 0x00200000

#: What ``acl_get_fd_np`` sets when there is no ACL to read.
_NO_ACL_ERRNOS = frozenset(
    {errno.ENOENT, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP), errno.EOPNOTSUPP}
)


class AclState(StrEnum):
    """What reading a node's ACL found."""

    NONE = "none"
    TEXT = "text"
    UNREADABLE = "unreadable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class NodeFacts:
    """What the gather step saw at one path: a file, a sidecar or a directory.

    ``exists`` is whether the name is there at all; ``opened`` whether it could
    be opened and ``fstat``'ed. ``uid`` and ``mode`` come from the open
    descriptor, or from an ``lstat`` when the open failed, and are ``None`` when
    neither worked. ``owners_ignored`` says the node lies on a volume that
    ignores ownership, where every user acts as each file's owner, or that
    whether it does could not be read; its ``uid`` and ``mode`` then say nothing
    about who may write it.
    """

    path: Path
    exists: bool
    opened: bool
    uid: int | None = None
    mode: int | None = None
    acl: AclState = AclState.NONE
    acl_text: str = ""
    owners_ignored: bool = False


@dataclass(frozen=True)
class LinkFacts:
    """A symlink the name resolved through: where it lives and who owns it (its ``lstat``)."""

    path: Path
    uid: int | None


@dataclass(frozen=True)
class WriteFacts:
    """Everything :func:`judge_write_facts` reads about one file.

    ``holding`` is the directory that holds the file, or, for a file that does
    not exist yet, the nearest directory on its path that does. ``named`` are
    the other directories in which a name standing for the file itself was
    looked up: where its final component, as named, is a symlink, and where
    each further link it resolves through lives. ``above`` are the other
    directories the name resolves through, up to the first broad root.
    ``links`` are the symlinks met on the way, outside a broad root.
    """

    file: NodeFacts
    sidecars: tuple[NodeFacts, ...]
    holding: NodeFacts
    above: tuple[NodeFacts, ...]
    named: tuple[NodeFacts, ...] = ()
    links: tuple[LinkFacts, ...] = ()


class ExposureKind(StrEnum):
    """Which part of the property failed."""

    FILE_NOT_OWNED = "file_not_owned"
    FILE_WRITABLE = "file_writable"
    FILE_UNJUDGEABLE = "file_unjudgeable"
    PLACE_OPEN = "place_open"


@dataclass(frozen=True)
class Exposure:
    """Why the property fails, and where.

    ``at`` is the path at fault: the file, a sidecar, or a directory.
    ``sidecar`` says a sidecar failed, ``owner`` is the uid of the file at
    fault when it was read, ``no_uid`` says the host reports no effective
    uid, so nothing could be judged, and ``owners_ignored``
    says the file lies on a volume that ignores ownership.
    """

    kind: ExposureKind
    at: Path
    sidecar: bool = False
    owner: int | None = None
    no_uid: bool = False
    owners_ignored: bool = False


# ---------------------------------------------------------------------------
# ACLs
# ---------------------------------------------------------------------------


@functools.cache
def _acl_library() -> Any:
    """The C library's ACL calls, typed, or ``None`` when they cannot be loaded."""
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        lib.acl_get_fd_np.restype = ctypes.c_void_p
        lib.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.acl_to_text.restype = ctypes.c_void_p
        lib.acl_to_text.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.acl_free.restype = ctypes.c_int
        lib.acl_free.argtypes = [ctypes.c_void_p]
    except (OSError, AttributeError):
        return None
    return lib


def read_acl(fd: int) -> tuple[AclState, str]:
    """The descriptor's ACL as ``acl_to_text`` prints it, or why there is none to judge.

    macOS keeps write grants outside the mode bits. A POSIX ACL on Linux keeps
    its named grants inside the group class bits, which the mode already shows,
    so nothing more is read there.
    """
    if sys.platform != "darwin":
        return AclState.NONE, ""
    lib = _acl_library()
    if lib is None:
        return AclState.UNAVAILABLE, ""
    ctypes.set_errno(0)
    acl = lib.acl_get_fd_np(fd, _ACL_TYPE_EXTENDED)
    if not acl:
        if ctypes.get_errno() in _NO_ACL_ERRNOS:
            return AclState.NONE, ""
        return AclState.UNREADABLE, ""
    try:
        text_p = lib.acl_to_text(acl, None)
        if not text_p:
            return AclState.UNREADABLE, ""
        try:
            return AclState.TEXT, ctypes.string_at(text_p).decode("utf-8", "replace")
        finally:
            lib.acl_free(text_p)
    finally:
        lib.acl_free(acl)


def acl_line_grants_write(line: str, owner_uid: int) -> bool:
    """One ``acl_to_text`` entry, ``tag:uuid:name:id:allow|deny[,flags]:perms``.

    The kind field carries the entry's flags after a comma: an inherited entry
    reads ``allow,inherited`` and an inheritable one on a directory
    ``allow,file_inherit,directory_inherit``. Both grant the write they name,
    so the kind is its first token. An ``only_inherit`` entry applies to the
    files later created below a directory, never to the directory itself, and
    is not counted here: each such file carries it as an inherited entry
    (security review 2026-10-28-1, O13, O22). Only an entry naming the owner's
    own uid is exempt, so a grant to root counts.
    """
    fields = line.split(":")
    if len(fields) < 6:
        return False
    kind, *flags = fields[-2].split(",")
    if kind != "allow" or "only_inherit" in flags:
        return False
    if fields[0] == "user" and fields[3] == str(owner_uid):
        return False
    return bool(_ACL_WRITE_PERMS & set(fields[-1].split(",")))


def _acl_grants_write(node: NodeFacts) -> bool:
    """Whether the node's ACL lets anyone but its owner write it; unjudged counts as yes."""
    if node.acl is AclState.NONE:
        return False
    if node.acl is not AclState.TEXT or node.uid is None:
        return True
    return any(acl_line_grants_write(line, node.uid) for line in node.acl_text.splitlines()[1:])


def acl_lets_others_write(fd: int, owner_uid: int) -> bool:
    """Whether the descriptor's ACL allows anyone but ``owner_uid`` to write it.

    An ACL that exists but cannot be read, or that cannot be read for want of the
    C library's ACL calls, counts as one that allows writing.
    """
    state, text = read_acl(fd)
    return _acl_grants_write(NodeFacts(Path(), True, True, owner_uid, 0, state, text))


# ---------------------------------------------------------------------------
# Volumes that ignore ownership
# ---------------------------------------------------------------------------


class _StatFs(ctypes.Structure):
    """macOS ``struct statfs`` in its 64-bit-inode layout (``sys/mount.h``)."""

    _fields_ = [
        ("f_bsize", ctypes.c_uint32),
        ("f_iosize", ctypes.c_int32),
        ("f_blocks", ctypes.c_uint64),
        ("f_bfree", ctypes.c_uint64),
        ("f_bavail", ctypes.c_uint64),
        ("f_files", ctypes.c_uint64),
        ("f_ffree", ctypes.c_uint64),
        ("f_fsid", ctypes.c_int32 * 2),
        ("f_owner", ctypes.c_uint32),
        ("f_type", ctypes.c_uint32),
        ("f_flags", ctypes.c_uint32),
        ("f_fssubtype", ctypes.c_uint32),
        ("f_fstypename", ctypes.c_char * 16),
        ("f_mntonname", ctypes.c_char * 1024),
        ("f_mntfromname", ctypes.c_char * 1024),
        ("f_flags_ext", ctypes.c_uint32),
        ("f_reserved", ctypes.c_uint32 * 7),
    ]


@functools.cache
def _fstatfs_call() -> Any:
    """The C library's ``fstatfs`` with the 64-bit-inode layout, typed, or ``None``.

    On Intel the call that fills that layout is ``fstatfs$INODE64``; on Apple
    silicon plain ``fstatfs`` already does.
    """
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    except OSError:
        return None
    for name in ("fstatfs$INODE64", "fstatfs"):
        try:
            call = getattr(lib, name)
        except AttributeError:
            continue
        call.restype = ctypes.c_int
        call.argtypes = [ctypes.c_int, ctypes.POINTER(_StatFs)]
        return call
    return None


def volume_statfs(fd: int) -> _StatFs | None:
    """The descriptor's volume as ``fstatfs`` reports it, or ``None`` when it cannot be read."""
    call = _fstatfs_call()
    if call is None:
        return None
    buf = _StatFs()
    if call(fd, ctypes.byref(buf)) != 0:
        return None
    return buf


def ownership_ignored(fd: int) -> bool:
    """Whether the descriptor's volume ignores ownership, so every user acts as each file's owner.

    macOS mounts many external and disk-image volumes this way ("Ignore
    ownership on this volume"): ``fstat`` then reports the caller's uid for
    every file and the kernel lets every user write as the owner, so neither
    the owner nor the mode bits say who may write it. A volume whose flags
    cannot be read counts as one that ignores ownership. Other platforms keep
    a mount's fixed owner and mode in force, which the mode already shows.
    """
    if sys.platform != "darwin":
        return False
    found = volume_statfs(fd)
    return found is None or bool(found.f_flags & _MNT_IGNORE_OWNERSHIP)


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


def _node_from_fd(path: Path, fd: int) -> NodeFacts:
    st = os.fstat(fd)
    state, text = read_acl(fd)
    ignored = ownership_ignored(fd)
    return NodeFacts(path, True, True, st.st_uid, st.st_mode, state, text, ignored)


def _node_at(path: Path, *, directory: bool) -> NodeFacts:
    """Open ``path`` without following a final symlink and record what it is."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) if directory else getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return NodeFacts(path, False, False)
    except OSError:
        try:
            st = os.lstat(path)
        except OSError:
            return NodeFacts(path, True, False)
        return NodeFacts(path, True, False, st.st_uid, st.st_mode)
    try:
        return _node_from_fd(path, fd)
    except OSError:
        return NodeFacts(path, True, False)
    finally:
        os.close(fd)


#: How many symlinks one resolution follows before it stops, as the kernel does.
_MAX_SYMLINK_HOPS = 40


@dataclass(frozen=True)
class _Lookup:
    """What one component-wise resolution saw (:func:`_lookup_directories`)."""

    resolved: Path
    looked: list[Path]
    named: list[Path]
    links: list[LinkFacts]


def _lookup_directories(path: Path) -> _Lookup:
    """Resolve ``path`` one component at a time, as ``realpath`` does, recording each lookup.

    Returns the resolved path and every directory a name was looked up in, in
    order and once each: the named path's directories and those of every
    symlink's target. Whoever can write one of them can choose what the name
    opens, so each is judged. It also returns the directories
    in which a name standing for the file itself was looked up (the final
    component, and the final component of each link that replaced it), and
    every symlink met with its ``lstat`` owner: a link's owner may replace it
    in a sticky directory, where nobody else may. A component that does not
    exist, or a resolution past the symlink limit, ends the walk with the rest
    of the path taken as named.
    """
    if not path.is_absolute():
        path = Path(os.getcwd()) / path
    current = Path(path.anchor)
    # Each pending component carries whether it names the file itself.
    parts = list(path.parts[1:])
    pending = [(name, i == len(parts) - 1) for i, name in enumerate(parts)]
    looked: list[Path] = []
    named: list[Path] = []
    links: list[LinkFacts] = []
    hops = 0

    def done(resolved: Path) -> _Lookup:
        return _Lookup(resolved, looked, named, links)

    while pending:
        name, final = pending.pop(0)
        if name in ("", "."):
            continue
        if name == "..":
            current = current.parent
            continue
        if current not in looked:
            looked.append(current)
        if final and current not in named:
            named.append(current)
        candidate = current / name
        try:
            st = os.lstat(candidate)
        except OSError:
            return done(candidate.joinpath(*(n for n, _ in pending)))
        if stat_module.S_ISLNK(st.st_mode):
            links.append(LinkFacts(candidate, st.st_uid))
            hops += 1
            if hops > _MAX_SYMLINK_HOPS:
                return done(candidate.joinpath(*(n for n, _ in pending)))
            target = Path(os.readlink(candidate))
            if target.is_absolute():
                current = Path(target.anchor)
                inner = list(target.parts[1:])
            else:
                inner = list(target.parts)
            # The link's last component stands for whatever the link did.
            pending = [(n, final and i == len(inner) - 1) for i, n in enumerate(inner)] + pending
            continue
        current = candidate
    return done(current)


def gather_write_facts(
    path: Path,
    *,
    sidecars: Sequence[str] = (),
    fd: int | None = None,
    resolved: Path | None = None,
) -> WriteFacts:
    """Record what :func:`judge_write_facts` needs about ``path``; decide nothing.

    ``path`` is the name as the caller holds it, symlinks unresolved: every
    directory its resolution looks a name up in is recorded.
    ``fd``, when given, is a descriptor the caller holds on the file, and the
    file's record is taken from it, so a read judges the bytes it returns;
    ``resolved`` is that descriptor's own path. Sidecars are the names SQLite
    opens beside the resolved file (part 3).
    """
    lookup = _lookup_directories(path)
    real = resolved if resolved is not None else lookup.resolved
    if fd is not None:
        try:
            file = _node_from_fd(real, fd)
        except OSError:
            file = NodeFacts(real, True, False)
    else:
        file = _node_at(real, directory=False)
    sides = tuple(
        _node_at(real.with_name(real.name + suffix), directory=False) for suffix in sidecars
    )
    holding = real.parent
    while not os.path.lexists(holding) and holding != holding.parent:
        holding = holding.parent
    # The directories other than the holding one in which a name for the file
    # itself was looked up: a symlink standing for the file lives there, and
    # for a file named ahead its creator may have chosen it.
    named = [d for d in lookup.named if d != holding and not is_broad_root(d)]
    # The resolved file's own ancestors first, nearest first, as the walks
    # before judged them; then every other directory the name was
    # looked up in, such as the one holding a symlinked component (§4.1).
    above: list[Path] = []
    for parent in holding.parents:
        if is_broad_root(parent):
            break
        if parent not in named:
            above.append(parent)
    for directory in lookup.looked:
        if (
            directory != holding
            and directory not in above
            and directory not in named
            and not is_broad_root(directory)
        ):
            above.append(directory)
    return WriteFacts(
        file=file,
        sidecars=sides,
        holding=_node_at(holding, directory=True),
        above=tuple(_node_at(d, directory=True) for d in above),
        named=tuple(_node_at(d, directory=True) for d in named),
        links=tuple(link for link in lookup.links if not is_broad_root(link.path.parent)),
    )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def _judge_file(node: NodeFacts, operator_uid: int | None, *, sidecar: bool) -> Exposure | None:
    if operator_uid is not None and node.uid is not None and node.uid != operator_uid:
        return Exposure(ExposureKind.FILE_NOT_OWNED, node.path, sidecar, node.uid)
    if not node.opened:
        return Exposure(ExposureKind.FILE_UNJUDGEABLE, node.path, sidecar, node.uid)
    if node.owners_ignored:
        return Exposure(
            ExposureKind.FILE_UNJUDGEABLE, node.path, sidecar, node.uid, owners_ignored=True
        )
    if (node.mode or 0) & WRITABLE_BY_OTHERS or _acl_grants_write(node):
        return Exposure(ExposureKind.FILE_WRITABLE, node.path, sidecar, node.uid)
    return None


def _directory_open(node: NodeFacts, operator_uid: int, *, sticky_closes: bool) -> bool:
    """Whether a user other than the operator may add, remove or rename the directory's entries."""
    if not node.opened or node.mode is None or node.owners_ignored:
        return True
    if node.uid not in (operator_uid, 0):
        return True
    sticky = sticky_closes and node.mode & stat_module.S_ISVTX
    if node.mode & WRITABLE_BY_OTHERS and not sticky:
        return True
    return _acl_grants_write(node)


def judge_write_facts(
    facts: WriteFacts, *, operator_uid: int | None, named_ahead: bool
) -> Exposure | None:
    """Apply the property to the gathered facts, with no I/O.

    The file, then each sidecar, then the place: the directory holding the file,
    and each directory where a symlink standing for it lives (with the sticky
    bit counting for nothing when ``named_ahead``, since a missing name there
    can be created whatever it says), then each directory above, where the
    sticky bit closes. A symlink on the way that a user other than the operator
    and root owns opens the directory it lives in, since its owner may replace
    it there even when the sticky bit stops everyone else. A node on a volume
    that ignores ownership cannot be judged. A file that does not exist is
    judged by its place alone. With no ``operator_uid`` (a host with no
    effective uid) the property cannot be shown, so every file fails it.
    """
    if operator_uid is None:
        return Exposure(ExposureKind.FILE_UNJUDGEABLE, facts.file.path, no_uid=True)
    if facts.file.exists and (found := _judge_file(facts.file, operator_uid, sidecar=False)):
        return found
    for side in facts.sidecars:
        if side.exists and (found := _judge_file(side, operator_uid, sidecar=True)):
            return found
    for node in (facts.holding, *facts.named):
        if _directory_open(node, operator_uid, sticky_closes=not named_ahead):
            return Exposure(ExposureKind.PLACE_OPEN, node.path)
    for link in facts.links:
        if link.uid not in (operator_uid, 0):
            return Exposure(ExposureKind.PLACE_OPEN, link.path.parent)
    for node in facts.above:
        if _directory_open(node, operator_uid, sticky_closes=True):
            return Exposure(ExposureKind.PLACE_OPEN, node.path)
    return None


def writable_only_by_operator(
    path: Path,
    *,
    named_ahead: bool,
    operator_uid: int | None,
    sidecars: Sequence[str] = (),
    fd: int | None = None,
    resolved: Path | None = None,
) -> Exposure | None:
    """Why ``path`` fails the property, or ``None`` when only the operator can write it.

    :func:`judge_write_facts` over :func:`gather_write_facts`. ``named_ahead``:
    the file is named before it exists and may be read by that name (the store,
    its sidecars, the config file), so its holding directory must be closed to
    creation. ``operator_uid`` is the uid the file must belong to.
    """
    facts = gather_write_facts(path, sidecars=sidecars, fd=fd, resolved=resolved)
    return judge_write_facts(facts, operator_uid=operator_uid, named_ahead=named_ahead)
