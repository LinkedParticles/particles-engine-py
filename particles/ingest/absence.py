# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Processing a pending ``ABSENT`` corpus record.

A completed ``import vault`` walk that no longer finds a previously imported
note writes an ``ABSENT`` record on the note's entry, ``PENDING``, because it
owes one piece of work: retiring the claims the note stated. That work is done
here, by the extraction step that processes pending snapshots, and it is the
generation cascade with the absence as the entry's current
generation:

    cascade_superseded_generation(entry, current_snapshot_id=<the ABSENT id>)

No claim cites an absence, so the cascade's candidate query returns every
``ACTIVE`` claim with a provenance edge to the entry, and absence processing
takes the snapshot-less edges too. The own-entry attestation test (a ref naming
the current snapshot attests) never holds, and other entries are read through
``get_entry_currency`` as before, so a claim another entry's current record
still states is kept. Everything else is retired
``PROVENANCE_STALE`` / ``RETRACTED_DEPENDENCY``, the reason an edit to an empty
note would give, and the record is marked ``COMPLETE``, in one transaction, so
no reader sees the note's claims retired on one axis and stated on the other.

**No model call**, by construction: the cascade reads provenance and statuses
only, and a ``RETRACTED_DEPENDENCY`` retirement triggers none of the passes
that call a model. The suite counts completion calls to hold this.

A memory-directory walk writes the same record for a deleted memory file.
Such an absence waits while another memory file of its project
still owes extraction (:func:`memory_absence_holder`), since that file may be
the merge target restating what the deleted one said.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles import db
from particles.core.observer_scope import PROJECT_TAG_PREFIX
from particles.core.schema import (
    CorpusEntry,
    ExtractionStatus,
    Mutability,
    WarcRecordType,
)
from particles.core.validity import ensure_utc
from particles.corpus.harvest_marks import MEMORY_ARCHIVE_TAG, MEMORY_FILE_TAG
from particles.corpus.store import CorpusEntryRow, SnapshotRow, update_extraction_status
from particles.db import store_of
from particles.ingest.generation import cascade_superseded_generation

log = logging.getLogger(__name__)

T = TypeVar("T")


async def process_absence(session: AsyncSession, *, entry_id: str, snapshot_id: str) -> list[str]:
    """Retire what ``entry_id`` stated and mark its ``ABSENT`` record ``COMPLETE``.

    Returns the ids retired. **Commits**, under the store's writer lock,
    as the extraction write phase does, so the caller's later
    commit is a no-op.
    """
    await session.commit()
    async with db.write_lock(store_of(session)):
        retired = await cascade_superseded_generation(
            session,
            entry_id=entry_id,
            current_snapshot_id=snapshot_id,
            include_unsnapshotted=True,
            actor="extract-absence",
        )
        await update_extraction_status(session, snapshot_id, ExtractionStatus.COMPLETE)
        await session.commit()
    log.info(
        "Entry %s: absence %s processed; %d claim(s) retired, no model call",
        entry_id[:8],
        snapshot_id[:8],
        len(retired),
    )
    return retired


#: The statuses of a ``RESPONSE`` that still owes extraction: a memory-file
#: absence waits behind one. ``IN_PROGRESS`` is included so a
#: pooled pass never processes an absence while its merge target is being read.
_OWES_EXTRACTION = (
    ExtractionStatus.PENDING.value,
    ExtractionStatus.IN_PROGRESS.value,
    ExtractionStatus.FAILED.value,
)


@dataclass(frozen=True)
class AbsenceHolder:
    """The unextracted memory-file snapshot a memory-file absence waits on."""

    entry_id: str
    snapshot_id: str
    failed: bool
    """The snapshot is ``FAILED`` (its blob is missing), so a retry alone will not release it."""


async def memory_absence_holder(
    session: AsyncSession, entry: CorpusEntry, captured_at: datetime
) -> AbsenceHolder | None:
    """The snapshot a memory-file absence must wait for, or ``None``.

    An absence on a memory-file entry is not processed while another
    memory file of the same project (a ``MUTABLE`` entry with the same
    ``memory-file`` and ``project:<key>`` tags, never the ``memory-archive``
    entry, which is append-only and no merge target) holds an uncollapsed
    ``RESPONSE`` that owes extraction and was captured at or before the
    absence (``captured_at``; the verb, about to write absences, passes the
    present). That file may be the
    merge target: extracted first, a claim it restates folds into the existing
    particle and survives the absence; processed after, the claim
    is retired and minted again under a new id. Every other absence returns
    ``None`` at once. The oldest holder is returned.
    """
    if MEMORY_FILE_TAG not in entry.tags:
        return None
    projects = [tag for tag in entry.tags if tag.startswith(PROJECT_TAG_PREFIX)]
    if not projects:
        return None
    rows = await session.execute(
        select(
            SnapshotRow.entry_id,
            SnapshotRow.snapshot_id,
            SnapshotRow.captured_at,
            SnapshotRow.extraction_status,
        )
        .join(CorpusEntryRow, CorpusEntryRow.entry_id == SnapshotRow.entry_id)
        .where(
            CorpusEntryRow.entry_id != entry.entry_id,
            CorpusEntryRow.mutability == Mutability.MUTABLE.value,
            ~CorpusEntryRow.tags_json.contains(json.dumps(MEMORY_ARCHIVE_TAG), autoescape=True),
            *(
                CorpusEntryRow.tags_json.contains(json.dumps(tag), autoescape=True)
                for tag in (MEMORY_FILE_TAG, *projects)
            ),
            SnapshotRow.warc_record_type == WarcRecordType.RESPONSE.value,
            SnapshotRow.extraction_status.in_(_OWES_EXTRACTION),
            SnapshotRow.superseded_by_snapshot_id.is_(None),
        )
    )
    cutoff = ensure_utc(captured_at)
    held = sorted(
        (ensure_utc(captured_at), snapshot_id, entry_id, status)
        for entry_id, snapshot_id, captured_at, status in rows.all()
        if ensure_utc(captured_at) <= cutoff
    )
    if not held:
        return None
    # A file deleted after that snapshot restates nothing any more: its own
    # absence collapses the snapshot at the next pass, so it holds
    # no other absence back.
    deleted = await session.execute(
        select(SnapshotRow.entry_id, SnapshotRow.captured_at).where(
            SnapshotRow.entry_id.in_({row[2] for row in held}),
            SnapshotRow.warc_record_type == WarcRecordType.ABSENT.value,
        )
    )
    deleted_at: dict[str, datetime] = {}
    for entry_id, captured_at in deleted.all():
        at = ensure_utc(captured_at)
        deleted_at[entry_id] = max(at, deleted_at.get(entry_id, at))
    held = [row for row in held if not (row[2] in deleted_at and deleted_at[row[2]] > row[0])]
    if not held:
        return None
    _, snapshot_id, entry_id, status = held[0]
    return AbsenceHolder(
        entry_id=str(entry_id),
        snapshot_id=str(snapshot_id),
        failed=status == ExtractionStatus.FAILED.value,
    )


def absences_last(items: Sequence[T], is_absent: Sequence[bool]) -> list[T]:
    """``items`` with the absences moved after everything else, each half in order.

    Every extract loop processes absences after its ``RESPONSE`` work,
    so a note renamed and edited in one sync is extracted
    under its new entry first, and a claim the new text restates is folded
    into the existing particle instead of being retired by the old
    entry's absence and minted again. ``is_absent`` runs parallel to ``items``.
    """
    flags = list(is_absent)
    return [i for i, a in zip(items, flags, strict=True) if not a] + [
        i for i, a in zip(items, flags, strict=True) if a
    ]
