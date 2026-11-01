# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Corpus refresh operation: re-check entries against their sources.

The one per-entry loop behind ``particles corpus refresh`` and consolidation
pass 0.5. For each entry it gathers the latest snapshot id, calls
:func:`~particles.corpus.fetch.maybe_refetch`, and classifies the result with
the pure :func:`~particles.corpus.refresh_outcome.classify_refresh`. It hands
the importer registry to the ladder, so an entry an importer owns is re-fetched
by that importer rather than by a GET of its URI-R.

Each entry commits on its own, and a failure rolls back that entry only and is
reported on its result, so one bad source never stops the sweep. Results are
yielded as they land, which lets a caller stream progress.

A local entry the root allowlist refuses is not handed to the ladder at all: it
yields ``REFUSED`` with the cause, and the loop logs one warning per cause per
run with the count and the fix that matches it, rather than a
line per entry. On a store that resolves against the working directory every
local entry is refused that way, unread.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import WarcRecordType
from particles.corpus.fetch import (
    RefreshRefusal,
    RefusalCounts,
    entry_refusal,
    maybe_refetch,
    store_refusal,
)
from particles.corpus.refresh_outcome import RefreshOutcome, classify_refresh
from particles.corpus.rule_roots import list_rule_source_roots
from particles.corpus.store import CorpusEntryRow, list_snapshots_for_entry
from particles.ingest.importers.registry import get_importers

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefreshResult:
    """One entry's refresh: an outcome, or the error that stopped it.

    ``refusal`` names the cause when ``outcome`` is ``REFUSED``.
    """

    entry_id: str
    outcome: RefreshOutcome | None
    snapshot_id: str | None = None
    error: Exception | None = None
    refusal: RefreshRefusal | None = None


async def refresh_entries(
    session: AsyncSession, entry_ids: Iterable[str], *, force: bool = False
) -> AsyncIterator[RefreshResult]:
    """Re-check each entry and yield its result, committing per entry.

    Args:
        session: A write session. It is committed after each entry, or rolled
            back when that entry fails.
        entry_ids: The entries to re-check, in order.
        force: The tier-3 override passed through to ``maybe_refetch``.

    Yields:
        One :class:`RefreshResult` per entry. ``outcome`` is None exactly when
        ``error`` is set.
    """
    importers = get_importers()
    recorded = await list_rule_source_roots(session)
    cwd_store = store_refusal(session) or False
    refused: list[RefreshRefusal] = []
    for entry_id in entry_ids:
        try:
            if (reason := await _refusal(session, entry_id, recorded, cwd_store)) is not None:
                refused.append(reason)
                yield RefreshResult(
                    entry_id=entry_id, outcome=RefreshOutcome.REFUSED, refusal=reason
                )
                continue
            before_id, before_absent = await _latest_snapshot(session, entry_id)
            snap = await maybe_refetch(session, entry_id, force=force, importers=importers)
            outcome = classify_refresh(before_id, snap, before_absent=before_absent)
            await session.commit()
        except Exception as exc:  # noqa: BLE001 — one bad source must not stop the sweep
            await session.rollback()
            yield RefreshResult(entry_id=entry_id, outcome=None, error=exc)
            continue
        yield RefreshResult(
            entry_id=entry_id,
            outcome=outcome,
            snapshot_id=snap.snapshot_id if snap is not None else None,
        )
    for line in RefusalCounts.tally(refused).lines():
        log.warning("%s", line)


async def _refusal(
    session: AsyncSession,
    entry_id: str,
    recorded: Sequence[Path],
    cwd_store: bool | RefreshRefusal,
) -> RefreshRefusal | None:
    """Why the local tier would refuse this entry's ``file://`` source."""
    row = await session.get(CorpusEntryRow, entry_id)
    return entry_refusal(row, recorded, cwd_store=cwd_store) if row is not None else None


async def _latest_snapshot(session: AsyncSession, entry_id: str) -> tuple[str | None, bool]:
    """The newest snapshot id for an entry and whether it is an ``ABSENT`` record.

    ``(None, False)`` when the entry has no snapshot yet.
    """
    snapshots = await list_snapshots_for_entry(session, entry_id)
    if not snapshots:
        return None, False
    latest = max(snapshots, key=lambda s: s.captured_at)
    return latest.snapshot_id, latest.warc_record_type is WarcRecordType.ABSENT
