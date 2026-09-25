# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Collapse superseded, unextracted snapshots of ``MUTABLE`` entries.

Extraction is deferred by default, so a file edited in N sessions between two
extraction passes holds N pending snapshots, and every bulk path used to
extract all N. For a ``MUTABLE`` entry that is wasted work — the generation
cascade retires whatever the older extractions wrote as soon as the newest one
lands — and, out of order, harmful: the cascade keys on "a snapshot other than
the one just extracted", so an older generation extracted after a newer one
retires the *current* beliefs.

The rule: a ``PENDING`` or ``FAILED`` RESPONSE snapshot *S* of a ``MUTABLE``
entry is superseded when the entry has a RESPONSE snapshot *S′* that sorts
after it, whose status is ``PENDING``, ``IN_PROGRESS`` or ``COMPLETE``, which
is not itself collapsed, and whose blob is on disk. *S* is then marked
``COMPLETE`` ("no extraction work is owed") and ``superseded_by_snapshot_id``
records *S′* — the column, not the status, is what says "collapsed".

Only ``MUTABLE`` entries are touched. ``APPEND_ONLY`` content is additive, so
extracting an older snapshot is never wrong there, only redundant; ``STABLE``
entries hold versions the operator deposited on purpose; ``EPHEMERAL`` entries
have no blob.

The four bulk paths call :func:`collapse_superseded_pending` before they list
what to extract, and pass ``skip_if_superseded=True`` to the pipeline so a
runner holding a pending list taken *before* a collapse still skips the rows it
marked. An explicit ``extract <entry> --snapshot-id`` does neither: naming a
snapshot is an operator instruction, and it un-collapses that generation.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ExtractionStatus
from particles.corpus.deposit import blob_path
from particles.corpus.store import (
    GenerationRow,
    list_generations_with_unextracted_snapshots,
    mark_snapshot_superseded,
)
from particles.db import write_lock

log = logging.getLogger(__name__)

#: Statuses an older snapshot must hold to be collapsible: work is still owed.
_UNEXTRACTED = frozenset({ExtractionStatus.PENDING, ExtractionStatus.FAILED})
#: Statuses a newer snapshot must hold to supersede: it is, or can become, the
#: entry's current generation. ``FAILED`` is absent on purpose — collapsing in
#: favour of a snapshot that cannot be extracted leaves the entry with nothing.
_CAN_SUPERSEDE = frozenset(
    {ExtractionStatus.PENDING, ExtractionStatus.IN_PROGRESS, ExtractionStatus.COMPLETE}
)


@dataclass
class CollapseReport:
    """What one :func:`collapse_superseded_pending` call marked."""

    #: ``(entry_id, snapshot_id, superseded_by)`` for each snapshot collapsed.
    collapsed: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def snapshots(self) -> int:
        return len(self.collapsed)

    @property
    def entries(self) -> int:
        return len({entry_id for entry_id, _, _ in self.collapsed})

    def summary(self) -> str | None:
        """The one-line disclosure, or None when nothing was collapsed."""
        if not self.collapsed:
            return None
        noun = "entry" if self.entries == 1 else "entries"
        return (
            f"Skipped {self.snapshots} superseded snapshot(s) across {self.entries} "
            f"MUTABLE {noun}: a newer snapshot of the same source replaces them."
        )


def plan_collapse(
    generations: list[GenerationRow], present_blobs: AbstractSet[str]
) -> list[tuple[str, str]]:
    """``(snapshot_id, superseded_by)`` for one entry's RESPONSE snapshots, oldest first.

    Pure (D2): ``present_blobs`` is the set of content hashes whose
    blob the caller found on disk, so the decision reads no filesystem. The
    superseding snapshot is the entry's **newest** eligible generation whose
    blob is present, so every collapsed row names the snapshot that will
    actually be extracted rather than an intermediate one that is itself about
    to be collapsed.
    """
    superseder: GenerationRow | None = None
    for row in reversed(generations):
        if _is_eligible_superseder(row) and row.content_hash in present_blobs:
            superseder = row
            break
    if superseder is None:
        return []
    cut = generations.index(superseder)
    return [
        (row.snapshot_id, superseder.snapshot_id)
        for row in generations[:cut]
        if row.extraction_status in _UNEXTRACTED and not row.collapsed
    ]


def _is_eligible_superseder(row: GenerationRow) -> bool:
    """Whether ``row`` may supersede older snapshots, blob presence aside."""
    return row.extraction_status in _CAN_SUPERSEDE and not row.collapsed


async def collapse_superseded_pending(
    session: AsyncSession,
    *,
    entry_ids: Collection[str] | None = None,
    dry_run: bool = False,
) -> CollapseReport:
    """Mark every superseded, unextracted snapshot of a ``MUTABLE`` entry collapsed.

    A no-op returning an empty report when
    ``extraction.collapse_superseded_pending`` is off or nothing qualifies.
    ``entry_ids`` scopes the pass (the audit's harvested entries). ``dry_run``
    reports what would be collapsed and writes nothing, for callers that
    promise zero writes (``reindex --dry-run``).

    **Commits**, unlike most helpers here, and deliberately: the marks are
    written under the store's writer lock, and that lock guards a
    write *transaction* — releasing it before the commit would guard nothing.
    Call it before starting other work on ``session``.
    """
    report = CollapseReport()
    if not get_config().extraction.collapse_superseded_pending:
        return report
    if entry_ids is not None and not entry_ids:
        return report

    grouped = await list_generations_with_unextracted_snapshots(session, entry_ids=entry_ids)
    # Gather: stat only the rows that could supersede, as the decision needs.
    present_blobs = {
        row.content_hash
        for generations in grouped.values()
        for row in generations
        if _is_eligible_superseder(row) and blob_path(row.content_hash).exists()
    }
    planned = [
        (entry_id, snapshot_id, superseded_by)
        for entry_id, generations in grouped.items()
        for snapshot_id, superseded_by in plan_collapse(generations, present_blobs)
    ]
    if not planned:
        return report
    if dry_run:
        report.collapsed.extend(planned)
        return report

    async with write_lock():
        for entry_id, snapshot_id, superseded_by in planned:
            if await mark_snapshot_superseded(session, snapshot_id, superseded_by=superseded_by):
                report.collapsed.append((entry_id, snapshot_id, superseded_by))
        await session.commit()

    per_entry: dict[str, int] = {}
    for entry_id, _, _ in report.collapsed:
        per_entry[entry_id] = per_entry.get(entry_id, 0) + 1
    for entry_id, count in per_entry.items():
        log.info(
            "Entry %s: collapsed %d superseded snapshot(s) (MUTABLE; a newer generation "
            "replaces them)",
            entry_id[:8],
            count,
        )
    return report
