# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""A memory-directory walk records what it no longer finds.

The memory harvest walks a Claude Code memory directory and deposits each
memory file it lists. This module is the other half of that walk: for each
previously harvested memory file of the directory that the walk no longer
lists, it writes an ``ABSENT`` record, so the deleted file's
claims retire at the next extraction pass (``ingest.absence``).

Gather / decide / apply (D2). The Surface walks the directory and
hands over what it listed (:class:`MemoryWalk`); the qualification of an entry
(:func:`qualifies`) and the guards (:func:`structural_skip`, then
``vault_import.decide_absences``) are pure; the gather and the writes run in
one writer-lock transaction (:func:`record_memory_absences`), so two sessions
ending together write one record per file.

Which directory is a memory directory, and what its project key is, are the
harness's conventions; the Engine receives the resolved directory and an
opaque key.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.observer_scope import project_tag
from particles.core.schema import Mutability, SourceType, WarcRecordType
from particles.corpus.deposit import record_absence
from particles.corpus.harvest_marks import MEMORY_ARCHIVE_TAG, MEMORY_FILE_TAG
from particles.corpus.store import (
    FileEntryRow,
    get_tags_for_entries,
    list_entry_ids_with_absences,
    list_file_entries_under,
)
from particles.db import write_transaction
from particles.ingest.absence import AbsenceHolder, memory_absence_holder, process_absence
from particles.ingest.generation import claims_absences_would_retire
from particles.operations.vault_import import (
    MissingNote,
    Refusal,
    decide_absences,
    path_of_file_uri,
)

log = logging.getLogger(__name__)

#: The file the projection writes into every memory directory.
#: Never given an absence: the projection recreates it.
PROJECTED_FILE = "MEMORY.md"

#: The suffix the harvest's walk lists.
MEMORY_SUFFIX = ".md"

#: Why a walk recorded no absence before any judgement guard was read (§2).
Skip = Literal["disabled", "walk_incomplete", "empty_listing", "file_refused"]

#: Missing files a report names one by one; the rest are counted.
NAMED_MISSING = 20


@dataclass(frozen=True)
class MemoryWalk:
    """What one walk of a memory directory found, as the Surface saw it."""

    directory: Path
    """The memory directory, resolved, so a symlinked directory is its target."""
    project_key: str
    """The project the harvest stamps on the directory's files."""
    listed: frozenset[str]
    """The ``file://`` URI of every ``.md`` path the walk enumerated, read or not."""
    errors: tuple[str, ...] = ()
    """Each enumeration error the walk met; any one makes it incomplete."""
    refused_files: int = 0
    """Files of this walk the engine refused to take (rule 5)."""

    @property
    def complete(self) -> bool:
        return not self.errors

    @property
    def projected_uri(self) -> str:
        return (self.directory / PROJECTED_FILE).as_uri()


@dataclass
class MemoryAbsenceReport:
    """What one walk's absence step found, decided and wrote."""

    directory: Path
    project_key: str
    listed: int = 0
    """Memory files the walk listed, ``MEMORY.md`` aside."""
    previously_harvested: int = 0
    """Qualifying entries under the directory whose latest record is not ``ABSENT``."""
    missing: list[MissingNote] = field(default_factory=list)
    absences: list[str] = field(default_factory=list)
    """The ``ABSENT`` records written, one per missing file when recorded."""
    skipped: Skip | None = None
    refusal: Refusal | None = None
    """``first`` or ``mass`` when a judgement guard held the walk's absences back."""
    walk_errors: list[str] = field(default_factory=list)

    @property
    def absent_fraction(self) -> float:
        if not self.previously_harvested:
            return 0.0
        return len(self.missing) / self.previously_harvested

    @property
    def recorded(self) -> bool:
        return bool(self.absences)

    @property
    def unrecorded(self) -> int:
        """Missing files this walk found and did not record."""
        return 0 if self.recorded else len(self.missing)

    def log_record(self) -> dict[str, Any]:
        """The hook-log and consolidation telemetry for this walk (§4)."""
        record: dict[str, Any] = {"missing": len(self.missing)}
        if self.absences:
            record["recorded"] = len(self.absences)
        if self.skipped is not None:
            record["skipped"] = self.skipped
        if self.refusal is not None and self.missing:
            record["absence_refused"] = {
                "reason": self.refusal,
                "count": len(self.missing),
                "fraction": round(self.absent_fraction, 3),
            }
        if self.walk_errors:
            record["walk_errors"] = self.walk_errors[:5]
        return record


@dataclass
class MemoryAbsencePreview:
    """What ``particles memory absences`` reports before it writes (§5)."""

    report: MemoryAbsenceReport
    would_retire: list[str] = field(default_factory=list)
    """The ``ACTIVE`` claims the absences would retire, read through the cascade's test."""
    holder: AbsenceHolder | None = None
    """An unextracted memory file of the project the absences would wait on."""


def qualifies(
    row: FileEntryRow, tags: Collection[str], *, directory: Path, project_key: str
) -> bool:
    """Whether an entry under ``directory`` is a memory file of its project (§1). Pure.

    ``MUTABLE`` ``LOCAL_MARKDOWN``, a ``.md`` URI-R under the resolved
    directory, tagged ``memory-file`` and ``project:<key>``, and neither the
    projected ``MEMORY.md`` nor an entry tagged ``memory-archive``. The
    depositor is not read: the first-run audit deposits the same files.
    """
    prefix = directory.as_uri().rstrip("/") + "/"
    return (
        row.uri_r.startswith(prefix)
        and row.uri_r.endswith(MEMORY_SUFFIX)
        and row.uri_r != (directory / PROJECTED_FILE).as_uri()
        and row.mutability is Mutability.MUTABLE
        and row.source_type == SourceType.LOCAL_MARKDOWN
        and MEMORY_FILE_TAG in tags
        and project_tag(project_key) in tags
        and MEMORY_ARCHIVE_TAG not in tags
    )


def structural_skip(walk: MemoryWalk, *, enabled: bool) -> Skip | None:
    """The §2 guard that keeps a walk from recording any absence, or ``None``. Pure.

    Checked before any judgement guard: the knob is off, the walk met an
    enumeration error, it listed no memory file besides ``MEMORY.md``, or the
    engine refused one of its files.
    """
    if not enabled:
        return "disabled"
    if not walk.complete:
        return "walk_incomplete"
    if not walk.listed - {walk.projected_uri}:
        return "empty_listing"
    if walk.refused_files:
        return "file_refused"
    return None


async def _gather(
    session: AsyncSession, walk: MemoryWalk, report: MemoryAbsenceReport
) -> tuple[list[FileEntryRow], bool]:
    """The qualifying entries, the missing ones into ``report``, and whether the
    directory has never recorded an absence."""
    rows = await list_file_entries_under(session, walk.directory.as_uri())
    tags: Mapping[str, list[str]] = await get_tags_for_entries(
        session, [row.entry_id for row in rows]
    )
    qualifying = [
        row
        for row in rows
        if qualifies(
            row,
            tags.get(row.entry_id, []),
            directory=walk.directory,
            project_key=walk.project_key,
        )
    ]
    living = [row for row in qualifying if row.latest_record_type is not WarcRecordType.ABSENT]
    report.previously_harvested = len(living)
    report.missing = [
        MissingNote(
            entry_id=row.entry_id,
            uri_r=row.uri_r,
            path=path_of_file_uri(row.uri_r),
            refers_to=row.last_response_id,
        )
        for row in living
        if row.uri_r not in walk.listed
    ]
    first = bool(report.missing) and not await list_entry_ids_with_absences(
        session, [row.entry_id for row in qualifying]
    )
    return qualifying, first


def _new_report(walk: MemoryWalk) -> MemoryAbsenceReport:
    return MemoryAbsenceReport(
        directory=walk.directory,
        project_key=walk.project_key,
        listed=len(walk.listed - {walk.projected_uri}),
        walk_errors=list(walk.errors),
    )


async def record_memory_absences(
    session: AsyncSession, walk: MemoryWalk, *, confirmed: bool = False
) -> MemoryAbsenceReport:
    """Record an ``ABSENT`` for each memory file the walk no longer lists (§1, §2, §4).

    ``confirmed`` is the operator's ``particles memory absences --record``: the
    structural guards still apply and the judgement guards do not. Unconfirmed
    (the SessionEnd hook, consolidation pass 6), the first absence ever under
    the directory and a mass absence are refused and reported, never written.
    The gather, the decision and the writes run in one writer-lock transaction,
    which commits.
    """
    report = _new_report(walk)
    cfg = get_config().claude_code.harvest
    report.skipped = structural_skip(walk, enabled=cfg.record_absence or confirmed)
    async with write_transaction(session):
        _, first = await _gather(session, walk, report)
        if report.skipped is not None:
            return report
        decision = decide_absences(
            missing=len(report.missing),
            previously_imported=report.previously_harvested,
            first_under_root=first,
            walk_complete=True,
            confirmed=confirmed,
            keep_missing=False,
            confirm_policy="first",
            max_fraction=cfg.max_absent_fraction,
        )
        report.refusal = decision.refusal
        if decision.record:
            for note in report.missing:
                report.absences.append(
                    await record_absence(session, entry_id=note.entry_id, refers_to=note.refers_to)
                )
    if report.missing:
        log.info(
            "memory directory %s: %d file(s) no longer found, %.1f%% of %d previously "
            "harvested; %s",
            walk.directory,
            len(report.missing),
            100 * report.absent_fraction,
            report.previously_harvested,
            "recorded" if report.recorded else f"not recorded ({report.refusal})",
        )
    return report


async def survey_memory_absences(session: AsyncSession, walk: MemoryWalk) -> MemoryAbsenceReport:
    """What the walk would find and the hook decide, read only (§4, §5).

    ``missing`` is every unrecorded deletion; ``refusal`` names the judgement
    guard that would hold the hook back, and ``skipped`` the structural guard
    that would stop any recording. Nothing is written. ``hook doctor``, the
    SessionStart notice and the verb's dry run read it.
    """
    report = _new_report(walk)
    report.skipped = structural_skip(walk, enabled=True)
    _, first = await _gather(session, walk, report)
    if report.skipped is None and report.missing:
        report.refusal = decide_absences(
            missing=len(report.missing),
            previously_imported=report.previously_harvested,
            first_under_root=first,
            walk_complete=True,
            confirmed=False,
            keep_missing=False,
            confirm_policy="first",
            max_fraction=get_config().claude_code.harvest.max_absent_fraction,
        ).refusal
    return report


async def preview_memory_absences(session: AsyncSession, walk: MemoryWalk) -> MemoryAbsencePreview:
    """What ``--record`` would write and retire, read only (§5).

    The structural guards are applied as ``--record`` applies them; the
    judgement guards are named in ``report.refusal`` but, as for ``--record``,
    do not stop the count.
    """
    report = await survey_memory_absences(session, walk)
    preview = MemoryAbsencePreview(report=report)
    if report.skipped is not None or not report.missing:
        return preview
    preview.would_retire = await claims_absences_would_retire(
        session, [note.entry_id for note in report.missing]
    )
    preview.holder = await project_holder(session, report)
    return preview


async def project_holder(
    session: AsyncSession, report: MemoryAbsenceReport
) -> AbsenceHolder | None:
    """The unextracted memory file the report's absences would wait on, if any (§3)."""
    from particles.corpus.store import get_entry

    if not report.missing:
        return None
    entry = await get_entry(session, report.missing[0].entry_id)
    if entry is None:
        return None
    # The absences are, or are about to be, captured now: every unextracted
    # snapshot already captured is at or before them.
    return await memory_absence_holder(session, entry, datetime.now(UTC))


async def process_recorded_absences(
    session: AsyncSession, report: MemoryAbsenceReport
) -> tuple[list[str], AbsenceHolder | None]:
    """Process the absences ``report`` recorded at once, with no model call (§5).

    Left pending, and the holder returned, while an unextracted memory file of
    the project could be the merge target that restates what a deleted file
    said; the next extraction pass orders them after it. Returns the ids
    retired and the holder.
    """
    if not report.absences:
        return [], None
    holder = await project_holder(session, report)
    if holder is not None:
        return [], holder
    retired: list[str] = []
    for note, snapshot_id in zip(report.missing, report.absences, strict=True):
        retired.extend(
            await process_absence(session, entry_id=note.entry_id, snapshot_id=snapshot_id)
        )
    return retired, None
