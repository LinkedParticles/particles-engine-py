# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""This installation's host id: which machine a ``file://`` path names.

A ``file://`` URI-R carries no host, and a store moves between machines as a
file, so the corpus records on each ``file://`` entry the id of the host whose
filesystem its path names (``corpus_entries.origin_host``). This module owns
that id for the running installation: a random UUID kept in the file
``host.id_file`` names, beside the operating system's own machine identifier as
read when the id was minted.

A file whose recorded machine identifier differs from the running machine's is
not believed, and a fresh id replaces it: a ``~/.particles`` directory copied
from one machine to another would otherwise make the second claim the first's
files. The id is not a secret and grants nothing by itself; its one use is
equality.

Store-free and network-free, so the Client layer may use it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import re
import stat
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from particles.config import get_config

log = logging.getLogger(__name__)

#: The ``ioreg`` line that carries a Mac's hardware UUID.
_IOPLATFORM_UUID = re.compile(r'"IOPlatformUUID"\s*=\s*"([0-9A-Fa-f-]+)"')

#: Where Linux keeps its machine identifier, in the order systemd reads them.
_LINUX_MACHINE_ID_FILES = ("/etc/machine-id", "/var/lib/dbus/machine-id")

#: The longest id file this module reads; a real one is under 200 bytes.
_MAX_ID_FILE_BYTES = 4096


class HostIdUnusable(Exception):
    """The id file cannot be used: a relative path, another user's file, or unwritable."""


@dataclass(frozen=True)
class HostIdentity:
    """The host id this installation uses, and how it came to be.

    ``minted`` is set when this call created the id; ``replaced`` holds the id
    a file copied from another machine carried, when that is why it was minted.
    """

    host_id: str
    path: Path
    machine_id: str
    minted: bool = False
    replaced: str | None = None


_cache: dict[str, HostIdentity] = {}


def reset_host_id_cache() -> None:
    """Forget the cached identity, so the next read consults the file again (tests)."""
    _cache.clear()
    _unusable.clear()


def machine_identifier() -> str:
    """The operating system's own identifier for this machine, or ``""`` when it supplies none.

    macOS: the hardware ``IOPlatformUUID``. Linux: ``/etc/machine-id``. A
    container usually has neither, which is fine: its id file then lives in its
    data volume and is believed as it is.
    """
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.run(  # noqa: S603 — fixed argv, no shell
                ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
        match = _IOPLATFORM_UUID.search(out)
        return match.group(1).lower() if match else ""
    for name in _LINUX_MACHINE_ID_FILES:
        with contextlib.suppress(OSError, UnicodeDecodeError):
            value = Path(name).read_text(encoding="ascii").strip()
            if value:
                return value
    return ""


def host_id_file() -> Path:
    """The id file ``host.id_file`` names, ``~`` expanded.

    Raises :class:`HostIdUnusable` for a relative setting: read against the
    working directory, an id file a cloned repository ships would decide which
    host's files this one claims.
    """
    path = Path(get_config().host.id_file).expanduser()
    if not path.is_absolute():
        raise HostIdUnusable(
            f"host.id_file ({str(path)!r}) is a relative path, which would be read "
            "against the working directory; set it to an absolute path"
        )
    return path


def host_identity() -> HostIdentity:
    """This installation's identity, minting one on first need.

    Cached per id-file path for the life of the process. Raises
    :class:`HostIdUnusable` when the file cannot be used; a caller that needs
    the id then treats this host as having none (:func:`local_host_id`).
    """
    path = host_id_file()
    cached = _cache.get(str(path))
    if cached is not None:
        return cached
    identity = _read_or_mint(path, machine_identifier())
    _cache[str(path)] = identity
    return identity


def local_host_id() -> str | None:
    """This host's id, or ``None`` when its id file is unusable (logged once per process).

    ``None`` fails closed: a deposit stamps no origin, and the local refresh
    reads no ``file://`` entry, since none is known to be this host's.
    """
    try:
        path = host_id_file()
    except HostIdUnusable as exc:
        _note_unusable("relative", exc)
        return None
    if str(path) in _unusable:
        return None
    try:
        return host_identity().host_id
    except HostIdUnusable as exc:
        _note_unusable(str(path), exc)
        return None


#: Id files found unusable this process: not read (or ``ioreg`` run) again
#: until :func:`reset_host_id_cache`.
_unusable: set[str] = set()


def _note_unusable(key: str, exc: HostIdUnusable) -> None:
    if key not in _unusable:
        log.warning("This host has no usable host id, so no local file is refreshed: %s", exc)
    _unusable.add(key)


class _Malformed(Exception):
    """The id file exists but holds no id; it is replaced."""


def _read_or_mint(path: Path, machine_id: str) -> HostIdentity:
    try:
        recorded = _read(path)
    except _Malformed:
        new = _write(path, machine_id, replaced=None, exclusive=False)
        return HostIdentity(new, path, machine_id, minted=True)
    if recorded is not None:
        host_id, recorded_machine = recorded
        if not machine_id or recorded_machine == machine_id:
            # An empty read is "unknown", never "another machine": an ``ioreg``
            # that timed out on a loaded host must not rotate the id and orphan
            # every entry this host stamped.
            return HostIdentity(host_id=host_id, path=path, machine_id=recorded_machine)
        if not recorded_machine:
            # Minted while the identifier could not be read; record it now.
            _write(path, machine_id, replaced=None, exclusive=False, host_id=host_id)
            return HostIdentity(host_id=host_id, path=path, machine_id=machine_id)
        log.warning(
            "The host id file %s was made on another machine (its machine identifier is "
            "not this one's); minting a new host id. Files the old id tracked are now "
            "another host's here.",
            path,
        )
        new = _write(path, machine_id, replaced=host_id, exclusive=False)
        return HostIdentity(new, path, machine_id, minted=True, replaced=host_id)
    try:
        new = _write(path, machine_id, replaced=None, exclusive=True)
    except FileExistsError:
        # Another process minted first (the hook and the MCP server start
        # together after an upgrade); its id is the one every process uses.
        try:
            again = _read(path)
        except _Malformed:
            again = None
        if again is None:
            raise HostIdUnusable(f"{path} appeared but cannot be read") from None
        return HostIdentity(again[0], path, machine_id)
    return HostIdentity(new, path, machine_id, minted=True)


def _read(path: Path) -> tuple[str, str] | None:
    """``(host_id, machine_id)`` from the file, ``None`` when there is none.

    Raises :class:`_Malformed` for a file that holds no id, and
    :class:`HostIdUnusable` for one this user cannot trust.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise HostIdUnusable(f"{path} cannot be opened: {exc}") from exc
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise HostIdUnusable(f"{path} is not a regular file")
        if st.st_uid != os.geteuid():
            raise HostIdUnusable(f"{path} is owned by uid {st.st_uid}, not this user")
        if st.st_mode & 0o022:
            raise HostIdUnusable(f"other users may write {path}")
        raw = handle.read(_MAX_ID_FILE_BYTES + 1)
    if len(raw) > _MAX_ID_FILE_BYTES:
        raise HostIdUnusable(f"{path} is larger than an id file")
    try:
        data = json.loads(raw)
        host_id = str(uuid.UUID(str(data["host_id"])))
        machine = data.get("machine_id", "")
        if not isinstance(machine, str):
            raise ValueError("machine_id is not a string")
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("The host id file %s is malformed (%s); minting a new host id", path, exc)
        raise _Malformed from exc
    return host_id, machine


def _write(
    path: Path,
    machine_id: str,
    *,
    replaced: str | None,
    exclusive: bool,
    host_id: str | None = None,
) -> str:
    """Write a fresh id beside ``machine_id``, mode 0600. Returns the id.

    ``exclusive`` links the file into place and raises :class:`FileExistsError`
    when one is already there, so two processes minting at once agree on one
    id; otherwise the file is replaced by rename.
    """
    host_id = host_id or str(uuid.uuid4())
    body = {"host_id": host_id, "machine_id": machine_id}
    if replaced is not None:
        body["replaced"] = replaced
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".host-id.", suffix=".tmp")
    except OSError as exc:
        raise HostIdUnusable(f"{path} cannot be written: {exc}") from exc
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(body, handle)
            handle.write("\n")
        if exclusive:
            try:
                os.link(tmp, path)
            finally:
                os.unlink(tmp)
        else:
            os.replace(tmp, path)
    except FileExistsError:
        raise
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise HostIdUnusable(f"{path} cannot be written: {exc}") from exc
    return host_id
