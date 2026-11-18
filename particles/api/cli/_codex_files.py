# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Contained rollout resolution; event transcript paths are never opened."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from particles.extraction.codex_transcript import UnsupportedTranscript, session_uuid


@dataclass(frozen=True)
class Rollout:
    """A complete bounded read of one identity-checked regular rollout."""

    path: Path
    content: bytes
    metadata: dict[str, Any]


def inventory(home: Path) -> list[Path]:
    """Enumerate only active/archive logs, never following directory/file links."""
    paths: list[Path] = []
    for name in ("sessions", "archived_sessions"):
        root = home / name
        if root.is_symlink() or not root.is_dir():
            continue
        for parent, directories, files in os.walk(root, followlinks=False):
            directories[:] = sorted(d for d in directories if not (Path(parent) / d).is_symlink())
            for filename in sorted(files):
                if filename.startswith("rollout-") and filename.endswith(".jsonl"):
                    candidate = Path(parent) / filename
                    if not candidate.is_symlink():
                        paths.append(candidate)
    return sorted(paths)


def read_contained(home: Path, path: Path, maximum: int) -> bytes:
    """Open every path component relative to a retained directory descriptor."""
    try:
        relative = path.relative_to(home)
    except ValueError:
        raise UnsupportedTranscript("rollout_outside_codex_home") from None
    if (
        not relative.parts
        or relative.parts[0] not in {"sessions", "archived_sessions"}
        or any(part in {".", ".."} for part in relative.parts)
    ):
        raise UnsupportedTranscript("invalid_rollout_path")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(home, flags | os.O_DIRECTORY)
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, flags | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        leaf = relative.parts[-1]
        opened = os.open(leaf, flags, dir_fd=fd)
        try:
            st = os.fstat(opened)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise UnsupportedTranscript("rollout_not_regular")
            if st.st_size > maximum:
                raise UnsupportedTranscript("rollout_oversized")
            chunks: list[bytes] = []
            count = 0
            while count <= maximum:
                chunk = os.read(opened, min(1024 * 1024, maximum + 1 - count))
                if not chunk:
                    break
                count += len(chunk)
                chunks.append(chunk)
            if count > maximum:
                raise UnsupportedTranscript("rollout_oversized")
            current = os.stat(leaf, dir_fd=fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (st.st_dev, st.st_ino):
                raise UnsupportedTranscript("rollout_changed_during_read")
            return b"".join(chunks)
        finally:
            os.close(opened)
    finally:
        os.close(fd)


def resolve_rollout(
    home: Path, identity: str, maximum: int, event_cwd: str | None = None
) -> Rollout:
    """Find a session by UUID in either approved root, including archive moves."""
    identity = session_uuid(identity)
    found: list[Rollout] = []
    for _attempt in range(2):
        for path in inventory(home):
            if not path.name.endswith(f"-{identity}.jsonl"):
                continue
            try:
                content = read_contained(home, path, maximum)
            except FileNotFoundError:
                continue  # archive move; scan again if no candidate survives
            try:
                first = json.loads(content.split(b"\n", 1)[0])
                meta = first["payload"]
                if first["type"] != "session_meta" or not isinstance(meta, dict):
                    raise ValueError
                actual = session_uuid(meta.get("session_id") or meta.get("id"))
                if actual != identity or meta.get("id") not in (None, identity):
                    raise ValueError
                cwd = meta.get("cwd")
                if not isinstance(cwd, str) or not Path(cwd).is_absolute():
                    raise ValueError
                if event_cwd is not None and os.path.normpath(cwd) != os.path.normpath(event_cwd):
                    raise UnsupportedTranscript("event_metadata_cwd_conflict")
            except (KeyError, TypeError, ValueError, UnicodeDecodeError):
                raise UnsupportedTranscript("rollout_identity_or_cwd_conflict") from None
            found.append(Rollout(path, content, meta))
        if found:
            break
    if not found:
        raise FileNotFoundError("session_rollout_missing")
    if any(item.content != found[0].content for item in found[1:]):
        raise UnsupportedTranscript("ambiguous_session_copies")
    return found[0]
