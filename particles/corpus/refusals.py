# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Sources the extraction model refused, and the hold that stops re-sending them.

A refusal is a judgement about content by one model, so it is keyed by the
snapshot's ``content_hash``, the LLM ``purpose`` that refused, and the
``provider_model`` pairing: ``extraction_refusals`` holds one row
per key. The pipeline writes a row only when every call of a pass was refused
(§3). A ``PENDING`` snapshot of a non-``APPEND_ONLY`` entry is **held** while a
row matches its content hash, the row's pairing is what its purpose resolves to
now, and the row's last refusal is inside the retry window (§4). The bulk
extraction paths skip held snapshots; naming the entry, or ``--retry-refused``,
sends them.

The table is SDK-internal, like the snapshot columns:
not a ``Snapshot`` model field, not serialised, not carried by the store
export. The caller passes the current ``{purpose: provider_model}`` pairings,
so this module reads no LLM configuration.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import DateTime, Integer, String, and_, delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.elements import ColumnElement

from particles.core.schema import ExtractionStatus, Mutability
from particles.corpus.store import CorpusEntryRow, SnapshotRow
from particles.db import Base


class ExtractionRefusalRow(Base):
    """One ``(content_hash, purpose, provider_model)`` the extraction model refused."""

    __tablename__ = "extraction_refusals"

    content_hash: Mapped[str] = mapped_column(String, primary_key=True)
    purpose: Mapped[str] = mapped_column(String, primary_key=True)
    provider_model: Mapped[str] = mapped_column(String, primary_key=True)
    # The snapshot and entry that last met the refusal: what the curate card
    # and the run report name.
    snapshot_id: Mapped[str] = mapped_column(String, nullable=False)
    entry_id: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    first_refused_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_refused_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    refusals: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


@dataclass(frozen=True)
class RefusalKey:
    """The purpose and pairing one refused call names, with its category."""

    purpose: str
    provider_model: str
    category: str | None = None


@dataclass(frozen=True)
class HeldSnapshot:
    """A ``PENDING`` snapshot the bulk paths skip, with the refusal holding it."""

    entry_id: str
    snapshot_id: str
    uri_r: str | None
    tags_json: str
    provider_model: str
    category: str | None
    last_refused_at: datetime

    @property
    def tags(self) -> list[str]:
        """The corpus entry's tags."""
        try:
            parsed = json.loads(self.tags_json or "[]")
        except ValueError:
            return []
        return [t for t in parsed if isinstance(t, str)] if isinstance(parsed, list) else []


async def record_refusals(
    session: AsyncSession,
    *,
    content_hash: str,
    snapshot_id: str,
    entry_id: str,
    keys: Iterable[RefusalKey],
    now: datetime | None = None,
) -> None:
    """Write or refresh one row per distinct key; the caller commits."""
    at = now or datetime.now(UTC)
    seen: dict[tuple[str, str], RefusalKey] = {}
    for key in keys:
        seen.setdefault((key.purpose, key.provider_model), key)
    for (purpose, provider_model), key in seen.items():
        row = await session.get(ExtractionRefusalRow, (content_hash, purpose, provider_model))
        if row is None:
            session.add(
                ExtractionRefusalRow(
                    content_hash=content_hash,
                    purpose=purpose,
                    provider_model=provider_model,
                    snapshot_id=snapshot_id,
                    entry_id=entry_id,
                    category=key.category,
                    first_refused_at=at,
                    last_refused_at=at,
                    refusals=1,
                )
            )
        else:
            row.snapshot_id = snapshot_id
            row.entry_id = entry_id
            row.category = key.category or row.category
            row.last_refused_at = at
            row.refusals = (row.refusals or 0) + 1
    await session.flush()


async def clear_refusals(
    session: AsyncSession, content_hash: str, pairings: Mapping[str, str]
) -> int:
    """Delete the rows a successful extraction under the current pairings answers (§4)."""
    match = _pairing_clause(pairings)
    if match is None:
        return 0
    result = await session.execute(
        delete(ExtractionRefusalRow).where(ExtractionRefusalRow.content_hash == content_hash, match)
    )
    return int(getattr(result, "rowcount", 0) or 0)


async def list_held_snapshots(
    session: AsyncSession,
    pairings: Mapping[str, str],
    retry_days: float,
    *,
    snapshot_ids: Sequence[str] | None = None,
    now: datetime | None = None,
) -> list[HeldSnapshot]:
    """Every held ``PENDING`` snapshot, oldest refusal first.

    ``retry_days <= 0`` turns the hold off. ``snapshot_ids`` narrows the
    search to a caller's work list.
    """
    match = _pairing_clause(pairings)
    if retry_days <= 0 or match is None:
        return []
    cutoff = (now or datetime.now(UTC)) - timedelta(days=retry_days)
    query = (
        select(
            SnapshotRow.entry_id,
            SnapshotRow.snapshot_id,
            CorpusEntryRow.uri_r,
            CorpusEntryRow.tags_json,
            ExtractionRefusalRow.provider_model,
            ExtractionRefusalRow.category,
            ExtractionRefusalRow.last_refused_at,
        )
        .join(CorpusEntryRow, CorpusEntryRow.entry_id == SnapshotRow.entry_id)
        .join(
            ExtractionRefusalRow,
            ExtractionRefusalRow.content_hash == SnapshotRow.content_hash,
        )
        .where(
            SnapshotRow.extraction_status == ExtractionStatus.PENDING.value,
            CorpusEntryRow.mutability != Mutability.APPEND_ONLY.value,
            ExtractionRefusalRow.last_refused_at >= cutoff,
            match,
        )
        .order_by(ExtractionRefusalRow.last_refused_at, SnapshotRow.snapshot_id)
    )
    if snapshot_ids is not None:
        if not snapshot_ids:
            return []
        query = query.where(SnapshotRow.snapshot_id.in_(list(snapshot_ids)))
    held: dict[str, HeldSnapshot] = {}
    for row in (await session.execute(query)).all():
        held.setdefault(
            str(row[1]),
            HeldSnapshot(
                entry_id=str(row[0]),
                snapshot_id=str(row[1]),
                uri_r=row[2],
                tags_json=str(row[3] or "[]"),
                provider_model=str(row[4]),
                category=row[5],
                last_refused_at=row[6],
            ),
        )
    return list(held.values())


def _pairing_clause(pairings: Mapping[str, str]) -> ColumnElement[bool] | None:
    """``(purpose, provider_model)`` equal to one of ``pairings``; ``None`` for none."""
    clauses = [
        and_(
            ExtractionRefusalRow.purpose == purpose,
            ExtractionRefusalRow.provider_model == provider_model,
        )
        for purpose, provider_model in pairings.items()
    ]
    if not clauses:
        return None
    return or_(*clauses)
