# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Corpus refresh operation: re-check entries against their sources.

The one per-entry loop behind ``particles corpus refresh`` and consolidation
pass 0.5. For each entry it gathers the latest snapshot id, calls
:func:`~particles.corpus.fetch.maybe_refetch`, and classifies the result with
the pure :func:`~particles.corpus.refresh_outcome.classify_refresh`.

Each entry commits on its own, and a failure rolls back that entry only and is
reported on its result, so one bad source never stops the sweep. Results are
yielded as they land, which lets a caller stream progress.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from particles.corpus.fetch import maybe_refetch
from particles.corpus.refresh_outcome import RefreshOutcome, classify_refresh
from particles.corpus.store import list_snapshots_for_entry


@dataclass(frozen=True)
class RefreshResult:
    """One entry's refresh: an outcome, or the error that stopped it."""

    entry_id: str
    outcome: RefreshOutcome | None
    snapshot_id: str | None = None
    error: Exception | None = None


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
    for entry_id in entry_ids:
        try:
            before_id = await _latest_snapshot_id(session, entry_id)
            snap = await maybe_refetch(session, entry_id, force=force)
            outcome = classify_refresh(before_id, snap)
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


async def _latest_snapshot_id(session: AsyncSession, entry_id: str) -> str | None:
    """The newest snapshot id for an entry, or None when it has none yet."""
    snapshots = await list_snapshots_for_entry(session, entry_id)
    if not snapshots:
        return None
    return max(snapshots, key=lambda s: s.captured_at).snapshot_id
