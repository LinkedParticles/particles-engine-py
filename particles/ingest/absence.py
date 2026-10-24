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
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import ExtractionStatus
from particles.corpus.store import update_extraction_status
from particles.db import write_lock
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
    async with write_lock():
        retired = await cascade_superseded_generation(
            session,
            entry_id=entry_id,
            current_snapshot_id=snapshot_id,
            include_unsnapshotted=True,
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
