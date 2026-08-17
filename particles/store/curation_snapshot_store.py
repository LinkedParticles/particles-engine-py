"""Persisted curation-card collections.

`build_curation_queue` used to run every finder on every request.
On the 2026-08-02 dogfood store that was 137 s of collection to return 7 cards,
which made `GET /curation` and `particles curate` unusable interactively. ADR
0238 splits the operation: the expensive **collection** is persisted here and
the cheap **session** half (suppression, status re-validation, slice, briefs)
re-runs live on every read.

This table is a **cache**, not a record. It holds no belief, no provenance and
no operator decision — every row is re-derivable by running the finders again —
so it is safe to ``DELETE FROM``: reads then fall back to collecting live until
the next build (the nightly cycle, or an explicit rebuild, never a read —
which would make a GET commit). That is the
``synthesis_cache`` / ``wikidata_label_cache`` precedent, and it is
why the
collection does **not** live in the append-only operator event log: the
``CONSOLIDATION_RUN`` record keeps a pointer (``curation_snapshot_id``) instead
of a multi-megabyte derived blob.

A snapshot records **per-kind scope** because the nightly cycle runs
its semantic passes delta-scoped. Store-wide kinds replace on the next build;
delta-scoped kinds carry forward — without that, an unresolved
contradiction found last night would vanish tonight, since contradiction
findings never persist as INCONSISTENCY particles.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import DateTime, Integer, String, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.db import Base

log = logging.getLogger(__name__)


class CollectionScope(StrEnum):
    """Whether a build's finders saw the whole store or only the delta.

    Mirrors the scope of the run that produced the collection.
    """

    STORE = "store"
    DELTA = "delta"


class CurationSnapshotRow(Base):
    """One persisted, scored curation-card collection."""

    __tablename__ = "curation_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    # The instant the collection finished — the staleness stamp clients render,
    # and the cutoff the live post-snapshot event filter compares against.
    built_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    # The build's overall scope (``CollectionScope``); per-kind detail below.
    scope: Mapped[str] = mapped_column(String, nullable=False)
    # Whether the LLM-assisted finders ran, and whether they were skipped
    # because the circuit breaker was open at build time.
    semantic: Mapped[bool] = mapped_column(nullable=False, default=False)
    semantic_degraded: Mapped[bool] = mapped_column(nullable=False, default=False)
    # {CardKind value: CollectionScope value} — drives replacement.
    per_kind_scope_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # The scored ``CurationCard[]``, serialised whole. Stored as one blob
    # deliberately: a per-kind table would let kinds drift out of
    # freshness-sync with each other and make the cross-kind leverage ranking
    # incomparable.
    cards_json: Mapped[str] = mapped_column(Text, nullable=False)
    card_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


async def write_snapshot(
    session: AsyncSession,
    *,
    cards_json: str,
    card_count: int,
    scope: CollectionScope,
    per_kind_scope: dict[str, str],
    semantic: bool = False,
    semantic_degraded: bool = False,
    built_at: datetime | None = None,
    retain: int = 3,
) -> str:
    """Persist a collection and prune to the newest ``retain`` rows.

    Returns the new ``snapshot_id``. Flushes but does not commit — the caller
    owns the transaction, as everywhere else in ``store/``.
    """
    snapshot_id = str(uuid.uuid4())
    session.add(
        CurationSnapshotRow(
            snapshot_id=snapshot_id,
            built_at=built_at or datetime.now(UTC),
            scope=scope.value,
            semantic=semantic,
            semantic_degraded=semantic_degraded,
            per_kind_scope_json=json.dumps(per_kind_scope),
            cards_json=cards_json,
            card_count=card_count,
        )
    )
    await session.flush()
    await _prune(session, retain=retain)
    return snapshot_id


async def latest_snapshot(session: AsyncSession) -> CurationSnapshotRow | None:
    """The newest collection, or ``None`` on a store that has never built one."""
    result = await session.execute(
        select(CurationSnapshotRow).order_by(CurationSnapshotRow.built_at.desc()).limit(1)
    )
    return result.scalars().first()


async def get_snapshot(session: AsyncSession, snapshot_id: str) -> CurationSnapshotRow | None:
    """Look up one collection by id."""
    result = await session.execute(
        select(CurationSnapshotRow).where(CurationSnapshotRow.snapshot_id == snapshot_id)
    )
    return result.scalars().first()


async def clear_snapshots(session: AsyncSession) -> int:
    """Drop every collection. Returns the number removed.

    The cache's escape hatch: a suspect snapshot is one call from gone, and
    reads degrade to collecting live — slow, never wrong — until the next
    build. Nothing else in the store depends on these rows.
    """
    rows = (await session.execute(select(CurationSnapshotRow.snapshot_id))).scalars().all()
    if rows:
        await session.execute(delete(CurationSnapshotRow))
    return len(rows)


async def _prune(session: AsyncSession, *, retain: int) -> None:
    """Keep the newest ``retain`` collections; delete the rest."""
    if retain < 1:
        retain = 1
    result = await session.execute(
        select(CurationSnapshotRow.snapshot_id)
        .order_by(CurationSnapshotRow.built_at.desc())
        .offset(retain)
    )
    stale = list(result.scalars().all())
    if not stale:
        return
    await session.execute(
        delete(CurationSnapshotRow).where(CurationSnapshotRow.snapshot_id.in_(stale))
    )
    log.debug("curation snapshot: pruned %d old collection(s)", len(stale))


def parse_per_kind_scope(row: CurationSnapshotRow) -> dict[str, CollectionScope]:
    """Decode ``per_kind_scope_json``, tolerating a legacy / corrupt blob.

    A snapshot whose per-kind map cannot be read is treated as fully
    store-wide, which is the conservative reading: every kind replaces on the
    next build and nothing is carried forward stale.
    """
    try:
        raw = json.loads(row.per_kind_scope_json)
    except (TypeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, CollectionScope] = {}
    for kind, value in raw.items():
        try:
            out[str(kind)] = CollectionScope(str(value))
        except ValueError:
            continue
    return out
