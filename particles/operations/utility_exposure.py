# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What a session was shown, read as of its start.

A mined utility event credits a belief only when the session was shown it.
Exposure is the union of three sets, each read **as of the
session's start**, never as of the mining run:

1. **Recorded.** The beliefs the session-start exposure rows say were
   delivered: the projected ``MEMORY.md`` region and the injected digest or
   diff.
2. **Rule files the harness loaded.** For each project the session belongs to
   (and for the keyless, user-level rule files), the root rule files always,
   and a subdirectory's rule file only when the session's action lines name a
   path under that directory. A project's deposited ``MEMORY.md`` counts here
   too: the harness loads it whole, and its deposit already has the projected
   region stripped.
3. **Files the session read.** The beliefs stated by a deposited local file
   that a ``Read`` action line names, which covers memory topic files.

For sets 2 and 3, "stated by" is the as-of form of the currently-stated test:
the belief was believed at the session's start, by the as-of
visibility predicate, and its own provenance names a snapshot of
the source that was current then.
For a ``MUTABLE`` source that is the latest extracted generation captured by
that instant; for any other source, any extracted snapshot captured by then.

Gather / decide: the path rules are pure functions over strings;
:class:`ExposureReader` does the reads, once per run for what does not vary by
session.

Nothing here writes. Query and MCP results are not exposure.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.observer_scope import project_keys
from particles.core.schema import ExtractionStatus, Mutability, ProvenanceRefType, WarcRecordType
from particles.corpus.rule_sources import RULE_SOURCE_TAG
from particles.corpus.store import CorpusEntryRow, SnapshotRow
from particles.operations.query.as_of import (
    AsOfView,
    RetirementIndex,
    ensure_utc,
    load_retirement_index,
)
from particles.store.particle_store import ParticleRow, ProvenanceEdgeRow
from particles.store.session_exposure_store import session_exposures_for

__all__ = [
    "Exposure",
    "ExposureReader",
    "RuleFile",
    "SessionFrame",
    "read_file_paths",
    "rule_files_in_view",
]

_IN_CHUNK = 500

#: A file the harness loads whole at session start whenever it is deposited
#: for the session's project: the auto-memory index. Its
#: projected region is stripped at deposit, so what it states here is the
#: hand-written rest.
_SESSION_START_FILES = frozenset({"MEMORY.md"})

#: A distilled ``Read`` action line: ``[tool: Read — /a/b.md]``.
_READ_LINE_RE = re.compile(r"^\[tool: Read — (?P<path>.+)\]$")

#: A path-like token in an action line: at least one ``/`` between word
#: characters. Quotes and shell punctuation around it are stripped.
_PATH_TOKEN_RE = re.compile(r"[\w.~-]*/[\w./~-]+")


@dataclass(frozen=True)
class RuleFile:
    """One deposited rule file: its entry and its absolute path."""

    entry_id: str
    path: str


@dataclass(frozen=True)
class SessionFrame:
    """Who a session was and when it started: the inputs exposure is read against."""

    session_id: str
    started_at: datetime
    #: The session's project keys. Usually one; empty when the
    #: session could not be attributed, in which case only keyless rule files
    #: count for set 2.
    project_keys: frozenset[str]
    #: Set 1: the belief ids recorded for the session, every row.
    recorded: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Exposure:
    """The three exposure sets for one session."""

    recorded: frozenset[str] = frozenset()
    rule_files: frozenset[str] = frozenset()
    read_files: frozenset[str] = frozenset()

    @property
    def shown(self) -> frozenset[str]:
        """Every belief the session was shown."""
        return self.recorded | self.rule_files | self.read_files


# ---------------------------------------------------------------------------
# Pure path rules
# ---------------------------------------------------------------------------


def read_file_paths(action_lines: Iterable[str]) -> list[str]:
    """The absolute paths the session's ``Read`` calls opened, in order, deduplicated.

    A path the distiller cut at its line cap (it ends in ``…``) names no file
    and is skipped.
    """
    seen: dict[str, None] = {}
    for line in action_lines:
        m = _READ_LINE_RE.match(line.strip())
        if m is None:
            continue
        path = m.group("path").strip()
        if path.endswith("…") or not path.startswith("/"):
            continue
        seen.setdefault(path, None)
    return list(seen)


def _action_path_tokens(action_lines: Iterable[str]) -> list[str]:
    tokens: dict[str, None] = {}
    for line in action_lines:
        for m in _PATH_TOKEN_RE.finditer(line):
            tokens.setdefault(m.group(0).rstrip(".,:;"), None)
    return list(tokens)


def _parent(path: str) -> str:
    return str(PurePosixPath(path).parent)


def _common_dir(dirs: Sequence[str]) -> str:
    parts = [PurePosixPath(d).parts for d in dirs]
    common: list[str] = []
    for column in zip(*parts, strict=False):
        if len(set(column)) != 1:
            break
        common.append(column[0])
    return str(PurePosixPath(*common)) if common else "/"


def _names_path_under(rel_dir: str, tokens: Sequence[str]) -> bool:
    """Whether any token names a path under ``rel_dir``, a directory relative to the root.

    Matches the directory as a whole run of path segments: at the start of a
    relative token (a path in a shell command), or after a ``/`` in an
    absolute one. The second form also catches another checkout of the same
    repository, such as a worktree. It can over-match a nested directory that
    shares the name, which only adds candidates: the judge still rules on
    every one.
    """
    prefix = rel_dir.rstrip("/") + "/"
    return any(t.removeprefix("./").startswith(prefix) or f"/{prefix}" in t for t in tokens)


def rule_files_in_view(rule_files: Sequence[RuleFile], action_lines: Sequence[str]) -> list[str]:
    """The entry ids of one project's rule files the harness loaded for this session.

    The project's root is the deepest directory every one of its rule files
    sits under. A rule file in the root is loaded always; one in a
    subdirectory only when the session's actions name a path under that
    subdirectory, which is when the harness loads it.
    """
    if not rule_files:
        return []
    dirs = [_parent(r.path) for r in rule_files]
    root = _common_dir(dirs)
    tokens = _action_path_tokens(action_lines)
    loaded: list[str] = []
    for rule, directory in zip(rule_files, dirs, strict=True):
        if directory == root:
            loaded.append(rule.entry_id)
            continue
        rel_dir = str(PurePosixPath(directory).relative_to(root))
        if _names_path_under(rel_dir, tokens):
            loaded.append(rule.entry_id)
    return loaded


def _file_path(uri_r: str | None) -> str | None:
    if not uri_r or not uri_r.startswith("file://"):
        return None
    return unquote(urlparse(uri_r).path) or None


# ---------------------------------------------------------------------------
# The reads
# ---------------------------------------------------------------------------


@dataclass
class ExposureReader:
    """Reads exposure for many sessions; what does not vary by session is loaded once.

    Build it with :meth:`load` at the start of a mining run. It holds the
    store's retirement index, the deposited local files by path,
    and the rule files grouped by project, all of which a run's sessions share.
    """

    index: RetirementIndex
    #: Absolute path → entry id, for every deposited ``file://`` entry.
    files: dict[str, str]
    #: Project key (``None`` for keyless, user-level files) → its rule files.
    rule_files: dict[str | None, list[RuleFile]]
    #: Project key → the entries the harness loads whole at session start.
    session_start_files: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    async def load(cls, session: AsyncSession) -> ExposureReader:
        """Load the per-run state."""
        files: dict[str, str] = {}
        rule_files: dict[str | None, list[RuleFile]] = defaultdict(list)
        start_files: dict[str, list[str]] = defaultdict(list)
        rows = await session.execute(
            select(CorpusEntryRow.entry_id, CorpusEntryRow.uri_r, CorpusEntryRow.tags_json).where(
                CorpusEntryRow.uri_r.like("file://%")
            )
        )
        for entry_id, uri_r, tags_json in rows.all():
            path = _file_path(uri_r)
            if path is None:
                continue
            files[path] = entry_id
            tags: list[str] = json.loads(tags_json or "[]")
            keys = project_keys(tags)
            if RULE_SOURCE_TAG in tags:
                for key in keys or {None}:
                    rule_files[key].append(RuleFile(entry_id=entry_id, path=path))
            elif PurePosixPath(path).name in _SESSION_START_FILES:
                for key in keys:
                    start_files[key].append(entry_id)
        return cls(
            index=await load_retirement_index(session),
            files=files,
            rule_files=dict(rule_files),
            session_start_files=dict(start_files),
        )

    def rule_file_entries(self, frame: SessionFrame, action_lines: Sequence[str]) -> set[str]:
        """Set 2's sources for one session: the entry ids, not yet the beliefs."""
        entries: set[str] = set()
        for key in (None, *sorted(frame.project_keys)):
            entries.update(rule_files_in_view(self.rule_files.get(key, []), action_lines))
            if key is not None:
                entries.update(self.session_start_files.get(key, []))
        return entries

    def read_file_entries(self, action_lines: Sequence[str]) -> set[str]:
        """Set 3's sources for one session: the deposited files its ``Read`` calls opened."""
        return {self.files[p] for p in read_file_paths(action_lines) if p in self.files}

    async def exposure(
        self, session: AsyncSession, frame: SessionFrame, action_lines: Sequence[str]
    ) -> Exposure:
        """The session's three exposure sets, each read as of ``frame.started_at``."""
        rule_entries = self.rule_file_entries(frame, action_lines)
        read_entries = self.read_file_entries(action_lines)
        stated = await stated_by(
            session,
            rule_entries | read_entries,
            AsOfView(as_of=ensure_utc(frame.started_at), index=self.index),
        )
        return Exposure(
            recorded=frame.recorded,
            rule_files=frozenset(p for p, es in stated.items() if es & rule_entries),
            read_files=frozenset(p for p, es in stated.items() if es & read_entries),
        )


async def stated_by(
    session: AsyncSession, entry_ids: Collection[str], view: AsOfView
) -> dict[str, set[str]]:
    """Beliefs the entries stated at ``view.as_of`` → the entries that stated each.

    A belief counts when it was believed at that instant and one
    of its own non-derivation provenance refs names a snapshot of the entry
    current then: for a ``MUTABLE`` entry the latest extracted generation
    captured by that instant, otherwise any extracted snapshot captured by it.
    A ref without a snapshot id counts when the entry had any. An entry with
    no extracted snapshot by then stated nothing.
    """
    ids = list(entry_ids)
    if not ids:
        return {}
    as_of = view.as_of

    mutable: set[str] = set()
    snapshots: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for start in range(0, len(ids), _IN_CHUNK):
        chunk = ids[start : start + _IN_CHUNK]
        for entry_id, mutability in (
            await session.execute(
                select(CorpusEntryRow.entry_id, CorpusEntryRow.mutability).where(
                    CorpusEntryRow.entry_id.in_(chunk)
                )
            )
        ).all():
            if mutability == Mutability.MUTABLE.value:
                mutable.add(entry_id)
        for entry_id, snapshot_id, captured_at in (
            await session.execute(
                select(
                    SnapshotRow.entry_id, SnapshotRow.snapshot_id, SnapshotRow.captured_at
                ).where(
                    SnapshotRow.entry_id.in_(chunk),
                    SnapshotRow.extraction_status == ExtractionStatus.COMPLETE.value,
                    SnapshotRow.warc_record_type == WarcRecordType.RESPONSE.value,
                    SnapshotRow.superseded_by_snapshot_id.is_(None),
                )
            )
        ).all():
            captured = ensure_utc(captured_at)
            if captured <= as_of:
                snapshots[entry_id].append((captured, snapshot_id))

    current: dict[str, set[str]] = {}
    for entry_id, held in snapshots.items():
        if entry_id in mutable:
            current[entry_id] = {max(held)[1]}
        else:
            current[entry_id] = {snapshot_id for _, snapshot_id in held}
    if not current:
        return {}

    live = list(current)
    candidates: set[str] = set()
    for start in range(0, len(live), _IN_CHUNK):
        rows = await session.execute(
            select(ProvenanceEdgeRow.particle_id).where(
                ProvenanceEdgeRow.corpus_entry_id.in_(live[start : start + _IN_CHUNK])
            )
        )
        candidates.update(pid for (pid,) in rows.all())

    stated: dict[str, set[str]] = {}
    batch = list(candidates)
    for start in range(0, len(batch), _IN_CHUNK):
        particle_rows = await session.execute(
            select(ParticleRow).where(ParticleRow.id.in_(batch[start : start + _IN_CHUNK]))
        )
        for row in particle_rows.scalars():
            particle = row.to_model()
            if not view.evaluate(particle, row.retired_at).visible:
                continue
            by: set[str] = set()
            for ref in particle.provenance:
                if ref.type is ProvenanceRefType.PARTICLE:
                    continue
                snaps = current.get(ref.corpus_entry_id)
                if snaps and (ref.snapshot_id is None or ref.snapshot_id in snaps):
                    by.add(ref.corpus_entry_id)
            if by:
                stated[particle.id] = by
    return stated


async def session_frame(
    session: AsyncSession,
    session_id: str,
    *,
    entry_tags: Collection[str] = (),
    first_captured_at: datetime | None = None,
    started_at: datetime | None = None,
    project_key: str | None = None,
) -> SessionFrame | None:
    """The frame exposure is read against for one session, or ``None`` if it has no start time.

    The start is the earliest of: an explicit ``started_at`` (the SessionEnd
    hook reads it from the raw transcript), the session's earliest recorded
    exposure row, and ``first_captured_at`` (the first harvest of
    the session's transcript). The last is an upper bound, used only for a
    session with neither, since the distilled transcript carries no timestamps.

    The project keys come from ``project_key``, the recorded rows, and the
    session transcript's entry tags, whichever are present.
    """
    rows = await session_exposures_for(session, session_id)
    starts = [ensure_utc(t) for t in (started_at, first_captured_at) if t is not None]
    starts.extend(ensure_utc(r.recorded_at) for r in rows)
    if not starts:
        return None
    keys = set(project_keys(entry_tags))
    keys.update(r.project_key for r in rows if r.project_key)
    if project_key:
        keys.add(project_key)
    return SessionFrame(
        session_id=session_id,
        started_at=min(starts),
        project_keys=frozenset(keys),
        recorded=frozenset(b.particle_id for r in rows for b in r.beliefs),
    )
