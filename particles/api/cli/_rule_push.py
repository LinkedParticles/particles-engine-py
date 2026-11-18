# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The client pushes its own rule files to a remote engine.

A remote engine never reads a client's ``file://`` path from its own disk, so
the client sends its ``AGENTS.md`` and ``CLAUDE.md`` files itself. The push
resolves, reads and filters each file exactly as a local ``rules sync``
deposits it, and sends it through ``Backend.deposit_text_at_uri`` as
``MUTABLE`` / ``NEVER``. A push record in the client's state directory keeps a
quiet run from sending anything: a file whose mtime is the recorded one is not
read, and one whose bytes hash to the recorded hash is not sent.

Two callers: the SessionEnd harvest (the session's own project and the two
user-level files, inside a share of the hook deadline) and ``particles rules
sync`` against a remote engine (the whole resolved set, no deadline).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from particles.api.cli._claude_code import rule_file_project_key, state_dir
from particles.api.cli._hook_state import read_json, write_json
from particles.config import get_config
from particles.core.observer_scope import project_tag
from particles.corpus.local_read import read_local_file
from particles.corpus.rule_sources import (
    RULE_SOURCE_TAG,
    resolve_rule_sources,
    rule_file_admitted,
)

log = logging.getLogger(__name__)

#: The source type and principal every pushed rule file is deposited under.
_SOURCE_TYPE = "LOCAL_MARKDOWN"
_PRINCIPAL = "claude-code-hook"


@dataclass
class PushReport:
    """What one push sent, skipped and could not send."""

    pushed: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    #: Rule files below the user-level ``~/.claude`` (a plugin's, a project
    #: directory's), which no global rule root admits.
    skipped_nested: list[Path] = field(default_factory=list)
    skipped_empty: list[Path] = field(default_factory=list)
    refused: list[tuple[Path, str]] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    #: Files left for the next run because the budget ran out.
    deferred: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "pushed": len(self.pushed),
            "unchanged": len(self.unchanged),
            "skipped_nested": len(self.skipped_nested),
            "refused": len(self.refused),
            "failed": len(self.failed),
            "deferred": self.deferred,
        }


def user_claude_dir() -> Path:
    """The client's user-level Claude Code directory, the global rule root."""
    return (Path.home() / ".claude").resolve()


def session_rule_files(project_root: Path) -> tuple[list[Path], list[Path]]:
    """The rule files a SessionEnd pushes: the project's, and the two user-level ones.

    ``project_root`` is the session's repository as Claude Code keys it, which
    for a linked worktree is its verified main checkout (``repository_root``),
    so a branch's copies are never pushed. Returns ``(files,
    roots)``, the roots being what a read is contained to.
    """
    resolution = resolve_rule_sources([str(project_root)])
    files = list(resolution.files) + global_rule_files()
    return sorted(set(files)), [*resolution.roots, user_claude_dir()]


def global_rule_files() -> list[Path]:
    """The rule files directly in ``~/.claude``: the only global ones a push sends."""
    root = user_claude_dir()
    found = []
    for name in get_config().rule_sources.filenames:
        candidate = root / name
        if candidate.is_file() and not candidate.is_symlink():
            found.append(candidate)
    return found


def split_nested(files: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    """``(pushable, nested)``: a file below ``~/.claude`` but not directly in it is nested."""
    root = user_claude_dir()
    pushable: list[Path] = []
    nested: list[Path] = []
    for path in files:
        if root in path.parents and path.parent != root:
            nested.append(path)
        else:
            pushable.append(path)
    return pushable, nested


def push_record_path() -> Path:
    return state_dir() / "rule-pushes.json"


def push_record_key(base_url: str, store: str, path: Path) -> str:
    return f"{base_url}\x00{store}\x00{path}"


def load_push_record() -> dict[str, Any]:
    try:
        data = read_json(push_record_path(), default={})
    except (OSError, ValueError):
        # A cache, never a source of truth: an unreadable record costs one
        # request per file, which the engine answers as unchanged (§4.2).
        return {}
    return data if isinstance(data, dict) else {}


async def push_rule_files(
    files: Sequence[Path],
    *,
    store: str,
    roots: Sequence[Path],
    filter_text: Callable[[str], str],
    budget_seconds: float | None = None,
) -> PushReport:
    """Push each changed rule file to the configured engine; skip the unchanged ones.

    A refusal of one file is recorded and the rest still go; an engine that
    cannot be reached, or that refuses the caller outright, ends the push. With
    ``budget_seconds`` the push stops when it is spent, and what is left waits
    for the next run. Each read is contained to ``roots`` (a rule file of those
    roots, as a local ``rules sync`` admits it) and must be this user's file,
    writable by no one else, so a push sends no more than
    a local sync would deposit.
    """
    from particles.api.client import get_backend
    from particles.api.client.http import (
        EngineHttpError,
        EngineRefusedError,
        EngineUnreachableError,
    )

    report = PushReport()
    pushable, report.skipped_nested = split_nested(files)
    base_url = get_config().engine.base_url or ""
    record = load_push_record()
    before = dict(record)
    started = time.monotonic()
    max_bytes = get_config().local_files.max_bytes
    admitted = rule_file_admitted(list(roots))
    backend = get_backend()
    try:
        for i, path in enumerate(pushable):
            if budget_seconds is not None and time.monotonic() - started > budget_seconds:
                report.deferred = len(pushable) - i
                break
            key = push_record_key(base_url, store, path)
            seen = record.get(key) if isinstance(record.get(key), dict) else None
            try:
                st = path.stat()
            except OSError as exc:
                report.failed.append((path, str(exc)))
                continue
            if (
                seen is not None
                and seen.get("mtime_ns") == st.st_mtime_ns
                and seen.get("refused_by", _key_fingerprint()) == _key_fingerprint()
            ):
                report.unchanged.append(path)
                continue
            try:
                read = read_local_file(
                    path,
                    follow_symlinks=True,
                    max_bytes=max_bytes,
                    allowed=admitted,
                    owner=os.geteuid(),
                )
            except OSError as exc:
                report.failed.append((path, str(exc)))
                continue
            raw = read.content.decode("utf-8", errors="replace")
            text = filter_text(raw)
            if not text.strip():
                report.skipped_empty.append(path)
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            entry = {"mtime_ns": read.stat.st_mtime_ns, "sha256": digest}
            if (
                seen is not None
                and seen.get("sha256") == digest
                and seen.get("refused_by", _key_fingerprint()) == _key_fingerprint()
            ):
                record[key] = {**seen, "mtime_ns": entry["mtime_ns"]}
                report.unchanged.append(path)
                continue
            project = rule_file_project_key(path)
            try:
                await backend.deposit_text_at_uri(
                    text=text,
                    uri_r=path.as_uri(),
                    source_type=_SOURCE_TYPE,
                    mutability="MUTABLE",
                    tags=[RULE_SOURCE_TAG, *([project_tag(project)] if project else [])],
                    deposited_by=_PRINCIPAL,
                    content_published_at=datetime.fromtimestamp(read.stat.st_mtime, tz=UTC),
                    store=store,
                    fetch_policy="NEVER",
                )
            except (EngineRefusedError, EngineUnreachableError):
                raise
            except EngineHttpError as exc:
                if exc.status_code == 429:
                    # Rate limited: "not now", about no file in particular. The
                    # rest wait for the next run, and nothing is recorded, so this
                    # file is sent again then rather than held as refused.
                    log.warning("the engine rate-limited the rule push at %s: %s", path, exc)
                    report.deferred = len(pushable) - i
                    break
                if exc.status_code is not None and exc.status_code >= 500:
                    # The engine failed, not the file: retried at the next run.
                    report.failed.append((path, str(exc)[:300]))
                    continue
                log.warning("the engine refused the rule file %s: %s", path, exc)
                report.refused.append((path, str(exc)[:300]))
                # Not sent again while the file and the key stay as they are: a
                # refusal is retried after an edit or under another key, never
                # every session.
                record[key] = {**entry, "refused_by": _key_fingerprint()}
                continue
            record[key] = entry
            report.pushed.append(path)
    finally:
        if record != before:
            try:
                write_json(push_record_path(), record)
            except OSError:
                log.debug("could not write the rule push record", exc_info=True)
    return report


def _key_fingerprint() -> str | None:
    """The id of the engine key this process presents, so a refusal is retried under another.

    Read from the key's prefix (``ptk_<role>_<id>_…``), never its secret; the
    legacy bearer, or no key, reads as ``None``.
    """
    from particles.secrets import get_engine_token_optional

    token = get_engine_token_optional() or ""
    parts = token.split("_")
    return parts[2] if token.startswith("ptk_") and len(parts) > 3 else None
