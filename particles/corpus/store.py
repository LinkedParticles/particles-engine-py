# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy ORM models for the source corpus (CorpusEntry + Snapshot)."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, aliased, mapped_column
from sqlalchemy.orm.attributes import flag_modified

from particles.core.schema import (
    ContributorRef,
    CorpusEntry,
    ExtractionStatus,
    FetchPolicy,
    Mutability,
    Snapshot,
    WarcRecordType,
)
from particles.db import Base


class CorpusEntryRow(Base):
    __tablename__ = "corpus_entries"

    entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    uri_r: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    mutability: Mapped[str] = mapped_column(String, nullable=False)
    fetch_policy: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deposited_by: Mapped[str] = mapped_column(String, nullable=False)
    tags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    # Extension D/E contributor attribution. NULL ≡ none recorded.
    contributors_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # cap. 2: document-supersession relation captured at ingest by a
    # genre adapter (the ADR adapter reads `supersedes:` / `superseded_by:`
    # frontmatter). A small JSON object — ``{"key": "adr:0116",
    # "supersedes": ["adr:0017"], "superseded_by": []}`` — naming this
    # document's canonical key and the document keys it (directly) supersedes /
    # is superseded by. The §6.6 rung-1.5 prior follows it transitively. NULL ≡
    # no genre relation (every non-genre source). Engine-internal: deliberately
    # *not* on the pydantic ``CorpusEntry`` model — it is read only by
    # ``particles.corpus.supersession`` at reconciliation time, never serialized.
    document_supersession_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_model(self, snapshots: list[Snapshot] | None = None) -> CorpusEntry:
        return CorpusEntry(
            entry_id=self.entry_id,
            uri_r=self.uri_r,
            source_type=self.source_type,
            mutability=Mutability(self.mutability),
            fetch_policy=FetchPolicy(self.fetch_policy),
            created_at=self.created_at,
            deposited_by=self.deposited_by,
            tags=json.loads(self.tags_json),
            snapshots=snapshots or [],
            contributors=(
                [ContributorRef(**c) for c in json.loads(self.contributors_json)]
                if self.contributors_json
                else None
            ),
        )

    @classmethod
    def from_model(cls, entry: CorpusEntry) -> CorpusEntryRow:
        return cls(
            entry_id=entry.entry_id,
            uri_r=entry.uri_r,
            source_type=entry.source_type,
            mutability=entry.mutability.value,
            fetch_policy=entry.fetch_policy.value,
            created_at=entry.created_at,
            deposited_by=entry.deposited_by,
            tags_json=json.dumps(entry.tags),
            contributors_json=(
                json.dumps([c.model_dump(mode="json") for c in entry.contributors])
                if entry.contributors is not None
                else None
            ),
        )


class SnapshotRow(Base):
    __tablename__ = "snapshots"

    snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    entry_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)
    etag: Mapped[str | None] = mapped_column(String, nullable=True)
    last_modified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    warc_record_type: Mapped[str] = mapped_column(String, nullable=False)
    archive_path: Mapped[str | None] = mapped_column(String, nullable=True)
    refers_to: Mapped[str | None] = mapped_column(String, nullable=True)
    extraction_status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # Set when extraction claims a snapshot (IN_PROGRESS write); cleared on
    # transition away from IN_PROGRESS. Used by ``extract --all-pending`` to
    # detect snapshots stranded mid-extraction (SIGKILL, segfault, oom)
    # whose try/finally cleanup didn't run. Nullable so historical rows
    # don't need a backfill.
    extraction_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    author_id: Mapped[str | None] = mapped_column(String, nullable=True)
    author_role: Mapped[str | None] = mapped_column(String, nullable=True)
    content_published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Set when the bulk extraction paths skip this snapshot because a newer
    # generation of the same MUTABLE entry superseded it: the id of that newer
    # snapshot. A snapshot is collapsed if and only if this is non-null; its
    # ``extraction_status`` is COMPLETE ("no extraction work is owed"), which
    # alone cannot tell a skipped generation from an extracted one. SDK-internal
    # by design: not a Snapshot-model field and not serialised.
    superseded_by_snapshot_id: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (Index("ix_snapshots_entry_extraction", "entry_id", "extraction_status"),)

    def to_model(self) -> Snapshot:
        return Snapshot(
            snapshot_id=self.snapshot_id,
            captured_at=self.captured_at,
            content_hash=self.content_hash,
            etag=self.etag,
            last_modified=self.last_modified_at,
            warc_record_type=WarcRecordType(self.warc_record_type),
            archive_path=self.archive_path,
            refers_to=self.refers_to,
            extraction_status=ExtractionStatus(self.extraction_status),
            author_id=self.author_id,
            author_role=self.author_role,
            content_published_at=self.content_published_at,
        )

    @classmethod
    def from_model(cls, snap: Snapshot, entry_id: str) -> SnapshotRow:
        return cls(
            snapshot_id=snap.snapshot_id,
            entry_id=entry_id,
            captured_at=snap.captured_at,
            content_hash=snap.content_hash,
            etag=snap.etag,
            last_modified_at=snap.last_modified,
            warc_record_type=snap.warc_record_type.value,
            archive_path=snap.archive_path,
            refers_to=snap.refers_to,
            extraction_status=snap.extraction_status.value,
            author_id=snap.author_id,
            author_role=snap.author_role,
            content_published_at=snap.content_published_at,
        )


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------


async def get_entry(session: AsyncSession, entry_id: str) -> CorpusEntry | None:
    row = await session.get(CorpusEntryRow, entry_id)
    if row is None:
        return None
    snap_rows_result = await session.execute(
        select(SnapshotRow)
        .where(SnapshotRow.entry_id == entry_id)
        .order_by(SnapshotRow.captured_at)
    )
    snaps = [r.to_model() for r in snap_rows_result.scalars()]
    return row.to_model(snaps)


async def get_entry_by_uri(session: AsyncSession, uri_r: str) -> CorpusEntry | None:
    result = await session.execute(select(CorpusEntryRow).where(CorpusEntryRow.uri_r == uri_r))
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return await get_entry(session, row.entry_id)


async def get_snapshot(session: AsyncSession, snapshot_id: str) -> Snapshot | None:
    row = await session.get(SnapshotRow, snapshot_id)
    return row.to_model() if row else None


async def list_snapshots_for_entry(session: AsyncSession, entry_id: str) -> list[Snapshot]:
    result = await session.execute(
        select(SnapshotRow)
        .where(SnapshotRow.entry_id == entry_id)
        .order_by(SnapshotRow.captured_at)
    )
    return [r.to_model() for r in result.scalars()]


async def update_extraction_status(
    session: AsyncSession, snapshot_id: str, status: ExtractionStatus
) -> None:
    row = await session.get(SnapshotRow, snapshot_id)
    if row is not None:
        row.extraction_status = status.value
        # Clear the claim timestamp on any transition away from IN_PROGRESS
        # so a later stale-detector can't see a stale value paired with a
        # non-IN_PROGRESS status.
        if status is not ExtractionStatus.IN_PROGRESS:
            row.extraction_started_at = None
        await session.flush()


async def claim_snapshot_for_extraction(
    session: AsyncSession, snapshot_id: str, *, started_at: datetime
) -> None:
    """Mark a snapshot as IN_PROGRESS and stamp the claim time.

    Paired with the ``extract_started_at`` column added in 0.42.2 so
    ``extract --all-pending`` can detect rows stranded by a SIGKILL /
    segfault / oom whose try/finally cleanup didn't run. Caller commits.
    """
    row = await session.get(SnapshotRow, snapshot_id)
    if row is not None:
        row.extraction_status = ExtractionStatus.IN_PROGRESS.value
        row.extraction_started_at = started_at
        # A claim un-collapses. The bulk paths never claim a
        # collapsed snapshot (``skip_if_superseded``), so reaching here with the
        # mark set means an operator named this generation explicitly; from now
        # on it is an extracted generation, or on failure a PENDING one the
        # next bulk pass is free to collapse again.
        row.superseded_by_snapshot_id = None
        # Forced into the UPDATE: the mark may have been written by another
        # process's bulk UPDATE, leaving this row's loaded value a stale None
        # that the assignment above would not register as a change.
        flag_modified(row, "superseded_by_snapshot_id")
        await session.flush()


async def reset_stale_in_progress(session: AsyncSession, *, older_than: datetime) -> list[str]:
    """Reset IN_PROGRESS snapshots whose claim is older than ``older_than``.

    Returns the list of snapshot IDs that were reset. Caller commits.

    Rows without ``extraction_started_at`` (legacy IN_PROGRESS from before
    0.42.2) are also reset — they predate the timestamp and the only honest
    interpretation of "no timestamp" is "nobody owns this claim".
    """
    selector = select(SnapshotRow.snapshot_id).where(
        SnapshotRow.extraction_status == ExtractionStatus.IN_PROGRESS.value,
        or_(
            SnapshotRow.extraction_started_at.is_(None),
            SnapshotRow.extraction_started_at < older_than,
        ),
    )
    result = await session.execute(selector)
    stale_ids = [r for r in result.scalars()]
    if not stale_ids:
        return []
    await session.execute(
        update(SnapshotRow)
        .where(SnapshotRow.snapshot_id.in_(stale_ids))
        .values(
            extraction_status=ExtractionStatus.PENDING.value,
            extraction_started_at=None,
        )
    )
    return stale_ids


async def snapshot_exists_by_hash(session: AsyncSession, content_hash: str) -> bool:
    result = await session.execute(
        select(SnapshotRow.snapshot_id).where(SnapshotRow.content_hash == content_hash)
    )
    return result.scalar_one_or_none() is not None


async def get_entry_by_content_hash(session: AsyncSession, content_hash: str) -> CorpusEntry | None:
    """Return the corpus entry whose most recent snapshot matches content_hash."""
    result = await session.execute(
        select(SnapshotRow.entry_id).where(SnapshotRow.content_hash == content_hash).limit(1)
    )
    entry_id = result.scalar_one_or_none()
    if entry_id is None:
        return None
    return await get_entry(session, entry_id)


async def list_entry_ids_with_extraction_status(
    session: AsyncSession, statuses: list[ExtractionStatus]
) -> list[str]:
    """Return distinct entry_ids that have at least one snapshot in the given statuses."""
    result = await session.execute(
        select(SnapshotRow.entry_id)
        .where(SnapshotRow.extraction_status.in_([s.value for s in statuses]))
        .distinct()
    )
    return list(result.scalars())


async def get_snapshot_source_meta(
    session: AsyncSession, snapshot_ids: list[str]
) -> dict[str, tuple[datetime | None, str | None]]:
    """Batch-load (content_published_at, author_id) keyed by snapshot_id.

    One SELECT for all ids — the query ranker and the §6.4 AUTHOR trust
    tier both read from this map, so the hot path stays free of
    per-particle snapshot round trips.
    """
    if not snapshot_ids:
        return {}
    rows = (
        await session.execute(
            select(
                SnapshotRow.snapshot_id,
                SnapshotRow.content_published_at,
                SnapshotRow.author_id,
            ).where(SnapshotRow.snapshot_id.in_(snapshot_ids))
        )
    ).all()
    return {snap_id: (pub_at, author_id) for snap_id, pub_at, author_id in rows}


async def get_source_types_for_entries(
    session: AsyncSession, entry_ids: list[str]
) -> dict[str, str]:
    """Batch-load source_type keyed by entry_id."""
    if not entry_ids:
        return {}
    rows = (
        await session.execute(
            select(CorpusEntryRow.entry_id, CorpusEntryRow.source_type).where(
                CorpusEntryRow.entry_id.in_(entry_ids)
            )
        )
    ).all()
    return {entry_id: source_type for entry_id, source_type in rows}


async def list_entry_tag_rows(session: AsyncSession) -> list[tuple[str, str | None, list[str]]]:
    """``(entry_id, uri_r, tags)`` for every corpus entry, oldest first.

    What ``rescope`` walks.
    """
    rows = (
        await session.execute(
            select(
                CorpusEntryRow.entry_id, CorpusEntryRow.uri_r, CorpusEntryRow.tags_json
            ).order_by(CorpusEntryRow.created_at)
        )
    ).all()
    return [(entry_id, uri_r, json.loads(tags_json or "[]")) for entry_id, uri_r, tags_json in rows]


async def get_tags_for_entries(
    session: AsyncSession, entry_ids: Collection[str]
) -> dict[str, list[str]]:
    """Batch-load tags keyed by entry_id (where a belief was observed).

    Ids that name no corpus entry are simply absent from the result, which is
    what a caller wants when some of its "entry ids" came from ``PARTICLE``
    provenance refs.
    """
    ids = list(entry_ids)
    tags: dict[str, list[str]] = {}
    for start in range(0, len(ids), 500):
        rows = (
            await session.execute(
                select(CorpusEntryRow.entry_id, CorpusEntryRow.tags_json).where(
                    CorpusEntryRow.entry_id.in_(ids[start : start + 500])
                )
            )
        ).all()
        tags.update({entry_id: json.loads(tags_json or "[]") for entry_id, tags_json in rows})
    return tags


@dataclass(frozen=True)
class EntryCurrency:
    """What the observer-scope join needs to know about one attesting entry."""

    tags: list[str]
    mutable: bool
    latest_snapshot_id: str | None
    """For a ``MUTABLE`` entry: its latest *extracted* generation (the
    :func:`get_latest_completed_snapshot_id` reading, batched), or ``None`` when
    nothing has been extracted yet. Always ``None`` for other mutabilities."""


async def get_entry_currency(
    session: AsyncSession, entry_ids: Collection[str], *, as_of: datetime | None = None
) -> dict[str, EntryCurrency]:
    """Tags, mutability and latest extracted snapshot per entry, in two batched reads.

    ``as_of`` restricts "latest" to snapshots captured at or before the instant,
    so an as-of read sees which generation was current then. Ids
    that name no corpus entry are absent from the result, as in
    :func:`get_tags_for_entries`.
    """
    ids = list(entry_ids)
    base: dict[str, tuple[list[str], bool]] = {}
    for start in range(0, len(ids), 500):
        rows = await session.execute(
            select(
                CorpusEntryRow.entry_id, CorpusEntryRow.tags_json, CorpusEntryRow.mutability
            ).where(CorpusEntryRow.entry_id.in_(ids[start : start + 500]))
        )
        for entry_id, tags_json, mutability in rows.all():
            base[entry_id] = (json.loads(tags_json or "[]"), mutability == Mutability.MUTABLE.value)

    mutable = [e for e, (_tags, is_mutable) in base.items() if is_mutable]
    latest: dict[str, tuple[datetime, str]] = {}
    for start in range(0, len(mutable), 500):
        stmt = select(SnapshotRow.entry_id, SnapshotRow.snapshot_id, SnapshotRow.captured_at).where(
            SnapshotRow.entry_id.in_(mutable[start : start + 500]),
            SnapshotRow.extraction_status == ExtractionStatus.COMPLETE.value,
            SnapshotRow.warc_record_type == WarcRecordType.RESPONSE.value,
            SnapshotRow.superseded_by_snapshot_id.is_(None),
        )
        if as_of is not None:
            stmt = stmt.where(SnapshotRow.captured_at <= as_of)
        for entry_id, snapshot_id, captured_at in (await session.execute(stmt)).all():
            held = latest.get(entry_id)
            if held is None or captured_at > held[0]:
                latest[entry_id] = (captured_at, snapshot_id)

    return {
        entry_id: EntryCurrency(
            tags=tags,
            mutable=is_mutable,
            latest_snapshot_id=latest[entry_id][1] if entry_id in latest else None,
        )
        for entry_id, (tags, is_mutable) in base.items()
    }


async def add_entry_tags(
    session: AsyncSession, entry_id: str, new_tags: Sequence[str]
) -> list[str]:
    """Append tags an entry does not already carry; returns the ones actually added.

    Additive only — the corpus never loses a tag this way. Flushes; the caller
    owns the transaction.
    """
    row = await session.get(CorpusEntryRow, entry_id)
    if row is None:
        return []
    current: list[str] = json.loads(row.tags_json or "[]")
    added = [tag for tag in dict.fromkeys(new_tags) if tag not in current]
    if added:
        row.tags_json = json.dumps([*current, *added])
        await session.flush()
    return added


async def get_entry_uri_map(
    session: AsyncSession, entry_ids: set[str] | None = None
) -> dict[str, str | None]:
    """Batch-load ``uri_r`` keyed by entry_id.

    Pass ``entry_ids=None`` to load every corpus entry's URI (used by the
    Obsidian exporter, which renders source links for every ACTIVE
    particle). Pass a non-empty set to filter — the wiki exporter only
    needs URIs for entries actually cited by the qualifying particles.
    Returns ``{}`` when ``entry_ids`` is the empty set, skipping the
    round-trip; otherwise issues one ``SELECT``.

    The value is ``str | None`` because ``CorpusEntryRow.uri_r`` is
    nullable for synthetic / local-only entries that have no public URL.
    """
    if entry_ids is not None and not entry_ids:
        return {}
    stmt = select(CorpusEntryRow.entry_id, CorpusEntryRow.uri_r)
    if entry_ids is not None:
        stmt = stmt.where(CorpusEntryRow.entry_id.in_(entry_ids))
    rows = (await session.execute(stmt)).all()
    return {entry_id: uri for entry_id, uri in rows}


async def get_document_supersession_map(
    session: AsyncSession, entry_ids: set[str]
) -> dict[str, str | None]:
    """Batch-load ``document_supersession_json`` keyed by entry_id.

    One ``SELECT`` for the given entries. The value is the raw JSON string the
    cap. 2 genre adapter stamped at deposit (``{"key": "adr:0166",
    …}``) or ``None`` for any entry that is not a recognised genre. The
    document-precedence tie-break reads the ``key`` from this map to recover the
    ADR id ordinal without re-parsing the source blob. Returns ``{}`` for an
    empty input set, skipping the round-trip.
    """
    if not entry_ids:
        return {}
    rows = (
        await session.execute(
            select(
                CorpusEntryRow.entry_id,
                CorpusEntryRow.document_supersession_json,
            ).where(CorpusEntryRow.entry_id.in_(entry_ids))
        )
    ).all()
    return {entry_id: raw for entry_id, raw in rows}


async def count_snapshots_by_extraction_status(session: AsyncSession) -> dict[str, int]:
    """Snapshot counts grouped by extraction_status."""
    from sqlalchemy import func

    result = await session.execute(
        select(SnapshotRow.extraction_status, func.count(SnapshotRow.snapshot_id)).group_by(
            SnapshotRow.extraction_status
        )
    )
    return {row[0]: row[1] for row in result}


async def count_entries(session: AsyncSession) -> int:
    """Total corpus entry count."""
    from sqlalchemy import func

    result = await session.execute(select(func.count(CorpusEntryRow.entry_id)))
    return int(result.scalar_one())


async def list_entries(
    session: AsyncSession,
    *,
    limit: int = 100,
    source_type: str | None = None,
) -> list[CorpusEntry]:
    """List corpus entries, most-recently-deposited first.

    Used by the MCP server's ``list_corpus_entries`` tool and
    by any agent that wants to browse what's been deposited. Optional
    ``source_type`` filter; ``limit`` caps the result set.
    """
    stmt = select(CorpusEntryRow).order_by(CorpusEntryRow.created_at.desc())
    if source_type is not None:
        stmt = stmt.where(CorpusEntryRow.source_type == source_type)
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return [row.to_model() for row in result.scalars()]


async def find_entry_ids_by_prefix(session: AsyncSession, prefix: str) -> list[str]:
    """Return all entry_ids whose ID starts with the given prefix (for CLI shortening)."""
    from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern

    pattern = f"{escape_like_pattern(prefix)}%"
    result = await session.execute(
        select(CorpusEntryRow.entry_id).where(
            CorpusEntryRow.entry_id.like(pattern, escape=LIKE_ESCAPE)
        )
    )
    return [str(row) for row in result.scalars()]


async def resolve_entry_id(session: AsyncSession, entry_id: str) -> str | None:
    """Resolve a full entry_id or unambiguous prefix to a full entry_id.

    Mirrors the prefix handling in
    :func:`particles.store.subject_store.get_subject`: an exact UUID wins;
    otherwise a unique prefix resolves; a prefix matching more than one entry
    raises ``ValueError``. Returns ``None`` if nothing matches. Lets the
    ``extract`` CLI verb accept the truncated 8-char ID the deposit / extract
    output displays, the way the other verbs already do.

    Raises:
        ValueError: If the prefix matches more than one entry.
    """
    if await session.get(CorpusEntryRow, entry_id) is not None:
        return entry_id
    # Escape user-input LIKE wildcards so '%' / '_' don't broaden the match.
    from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern

    pattern = f"{escape_like_pattern(entry_id)}%"
    result = await session.execute(
        select(CorpusEntryRow.entry_id).where(
            CorpusEntryRow.entry_id.like(pattern, escape=LIKE_ESCAPE)
        )
    )
    rows = [str(row) for row in result.scalars()]
    if len(rows) == 1:
        return rows[0]
    if len(rows) > 1:
        raise ValueError(
            f"Ambiguous entry prefix {entry_id!r} matches {len(rows)} entries; use more characters."
        )
    return None


async def resolve_snapshot_id(session: AsyncSession, entry_id: str, snapshot_id: str) -> str | None:
    """Resolve a full snapshot_id or unambiguous prefix *within one entry*.

    Same contract as :func:`resolve_entry_id` but scoped to the snapshots of a
    single corpus entry: an exact match wins; otherwise a unique prefix among
    that entry's snapshots resolves; a prefix matching more than one raises
    ``ValueError``. Returns ``None`` if nothing matches.

    Raises:
        ValueError: If the prefix matches more than one snapshot.
    """
    exact = await session.get(SnapshotRow, snapshot_id)
    if exact is not None and exact.entry_id == entry_id:
        return snapshot_id
    from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern

    pattern = f"{escape_like_pattern(snapshot_id)}%"
    result = await session.execute(
        select(SnapshotRow.snapshot_id).where(
            SnapshotRow.entry_id == entry_id,
            SnapshotRow.snapshot_id.like(pattern, escape=LIKE_ESCAPE),
        )
    )
    rows = [str(row) for row in result.scalars()]
    if len(rows) == 1:
        return rows[0]
    if len(rows) > 1:
        raise ValueError(
            f"Ambiguous snapshot prefix {snapshot_id!r} matches {len(rows)} snapshots;"
            " use more characters."
        )
    return None


async def resolve_snapshot_for_blob(session: AsyncSession, selector: str) -> Snapshot | None:
    """Resolve a ``corpus cat`` selector to the snapshot whose blob to read.

    ``selector`` is a full ID or unambiguous prefix of *either* a snapshot or a
    corpus entry — ``corpus cat`` accepts both. Resolution tries the snapshot
    interpretation first (exact ``snapshot_id``, then a unique snapshot-id
    prefix across all entries), and falls back to the entry interpretation
    (exact ``entry_id`` or unique entry-id prefix), resolving that entry to its
    most-recently-captured snapshot. Returns ``None`` if nothing matches.

    Raises:
        ValueError: If the prefix is ambiguous — it matches more than one
            snapshot, or (in the fallback) more than one entry.
    """
    from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern

    # 1. Snapshot interpretation: exact match, then unique prefix (global).
    exact_snap = await session.get(SnapshotRow, selector)
    if exact_snap is not None:
        return exact_snap.to_model()

    pattern = f"{escape_like_pattern(selector)}%"
    snap_rows = list(
        (
            await session.execute(
                select(SnapshotRow).where(SnapshotRow.snapshot_id.like(pattern, escape=LIKE_ESCAPE))
            )
        ).scalars()
    )
    if len(snap_rows) == 1:
        return snap_rows[0].to_model()
    if len(snap_rows) > 1:
        raise ValueError(
            f"Ambiguous snapshot prefix {selector!r} matches {len(snap_rows)} snapshots;"
            " use more characters."
        )

    # 2. Entry interpretation: exact match, then unique prefix → latest snapshot.
    entry_id: str | None = None
    if await session.get(CorpusEntryRow, selector) is not None:
        entry_id = selector
    else:
        entry_ids = [
            str(row)
            for row in (
                await session.execute(
                    select(CorpusEntryRow.entry_id).where(
                        CorpusEntryRow.entry_id.like(pattern, escape=LIKE_ESCAPE)
                    )
                )
            ).scalars()
        ]
        if len(entry_ids) == 1:
            entry_id = entry_ids[0]
        elif len(entry_ids) > 1:
            raise ValueError(
                f"Ambiguous entry prefix {selector!r} matches {len(entry_ids)} entries;"
                " use more characters."
            )
    if entry_id is None:
        return None

    latest = (
        (
            await session.execute(
                select(SnapshotRow)
                .where(SnapshotRow.entry_id == entry_id)
                .order_by(SnapshotRow.captured_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    return latest.to_model() if latest is not None else None


async def list_entry_snapshot_pairs_with_extraction_status(
    session: AsyncSession, statuses: list[ExtractionStatus]
) -> list[tuple[str, str]]:
    """Return (entry_id, snapshot_id) tuples for snapshots in any of the given statuses."""
    result = await session.execute(
        select(SnapshotRow.entry_id, SnapshotRow.snapshot_id).where(
            SnapshotRow.extraction_status.in_([s.value for s in statuses])
        )
    )
    return [(str(row[0]), str(row[1])) for row in result.all()]


async def list_pending_snapshots_oldest_first(
    session: AsyncSession,
) -> list[tuple[str, str]]:
    """(entry_id, snapshot_id) pairs of PENDING snapshots, oldest capture first.

    The extract catch-up order: a capped pass drains the backlog
    front-to-back, so the oldest deposits are never starved by newer ones.
    ``snapshot_id`` breaks capture-time ties deterministically.
    """
    result = await session.execute(
        select(SnapshotRow.entry_id, SnapshotRow.snapshot_id)
        .where(SnapshotRow.extraction_status == ExtractionStatus.PENDING.value)
        .order_by(SnapshotRow.captured_at, SnapshotRow.snapshot_id)
    )
    return [(str(row[0]), str(row[1])) for row in result.all()]


async def list_entry_ids_created_since(session: AsyncSession, since: datetime) -> list[str]:
    """Entry ids deposited at/after ``since`` (delta scope)."""
    result = await session.execute(
        select(CorpusEntryRow.entry_id).where(CorpusEntryRow.created_at >= since)
    )
    return [str(eid) for eid in result.scalars()]


async def list_complete_response_snapshots(
    session: AsyncSession,
) -> list[tuple[str, str]]:
    """Return (entry_id, snapshot_id) for COMPLETE snapshots that hold their own blob.

    Excludes REVISIT snapshots (``warc_record_type == REVISIT``): those are
    COMPLETE and produce zero particles *by design* — they inherit content
    lazily from the snapshot they ``refers_to`` and are never extracted. Only
    RESPONSE snapshots are expected to yield particles, so they are the ones a
    zero-particle audit (lint ``EMPTY_COMPLETE_SNAPSHOT``) should consider.
    """
    result = await session.execute(
        select(SnapshotRow.entry_id, SnapshotRow.snapshot_id).where(
            SnapshotRow.extraction_status == ExtractionStatus.COMPLETE.value,
            SnapshotRow.warc_record_type == WarcRecordType.RESPONSE.value,
        )
    )
    return [(str(row[0]), str(row[1])) for row in result.all()]


async def get_latest_completed_snapshot_id(session: AsyncSession, entry_id: str) -> str | None:
    """Return the entry's newest *extracted* generation: its latest COMPLETE RESPONSE snapshot.

    REVISIT snapshots are excluded. The fetch ladder writes every REVISIT
    ``COMPLETE`` ("no new extraction needed"), so without the filter the newest
    COMPLETE row is usually a REVISIT — blob-less, with no particles of its own.
    Both callers mean "the generation whose beliefs are in the store": the
    generation backfill would otherwise take the REVISIT as current and demote
    every ACTIVE particle of the entry, including the generation the REVISIT
    refers to, and ``reindex <entry>`` would hand the pipeline a snapshot it
    skips for having no ``archive_path``.
    """
    result = await session.execute(
        select(SnapshotRow.snapshot_id)
        .where(
            SnapshotRow.entry_id == entry_id,
            SnapshotRow.extraction_status == ExtractionStatus.COMPLETE.value,
            SnapshotRow.warc_record_type == WarcRecordType.RESPONSE.value,
            SnapshotRow.superseded_by_snapshot_id.is_(None),
        )
        .order_by(SnapshotRow.captured_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


@dataclass(frozen=True)
class GenerationRow:
    """One RESPONSE snapshot of a MUTABLE entry, as the collapse reads it."""

    entry_id: str
    snapshot_id: str
    captured_at: datetime
    extraction_status: ExtractionStatus
    content_hash: str
    collapsed: bool


async def list_generations_with_unextracted_snapshots(
    session: AsyncSession, *, entry_ids: Collection[str] | None = None
) -> dict[str, list[GenerationRow]]:
    """RESPONSE snapshots of every MUTABLE entry that still owes an extraction.

    The candidate read. An entry qualifies when it has at least one
    RESPONSE snapshot that is ``PENDING`` or ``FAILED`` and not already
    collapsed; for each such entry **all** of its RESPONSE snapshots are
    returned, oldest first on ``(captured_at, snapshot_id)`` — the same
    tie-break :func:`list_pending_snapshots_oldest_first` uses — because
    whether a snapshot is superseded depends on its newer siblings, whatever
    their status. REVISIT snapshots are excluded: a REVISIT records that the
    content did *not* change, so it is never a generation.

    ``entry_ids`` scopes the read (the audit's harvest); ``None`` is store-wide.
    """
    owing = select(SnapshotRow.entry_id).where(
        SnapshotRow.warc_record_type == WarcRecordType.RESPONSE.value,
        SnapshotRow.extraction_status.in_(
            [ExtractionStatus.PENDING.value, ExtractionStatus.FAILED.value]
        ),
        SnapshotRow.superseded_by_snapshot_id.is_(None),
    )
    stmt = (
        select(SnapshotRow)
        .join(CorpusEntryRow, CorpusEntryRow.entry_id == SnapshotRow.entry_id)
        .where(
            CorpusEntryRow.mutability == Mutability.MUTABLE.value,
            SnapshotRow.warc_record_type == WarcRecordType.RESPONSE.value,
            SnapshotRow.entry_id.in_(owing),
        )
        .order_by(SnapshotRow.entry_id, SnapshotRow.captured_at, SnapshotRow.snapshot_id)
    )
    if entry_ids is not None:
        stmt = stmt.where(SnapshotRow.entry_id.in_(list(entry_ids)))
    grouped: dict[str, list[GenerationRow]] = {}
    for row in (await session.execute(stmt)).scalars():
        grouped.setdefault(row.entry_id, []).append(
            GenerationRow(
                entry_id=row.entry_id,
                snapshot_id=row.snapshot_id,
                captured_at=row.captured_at,
                extraction_status=ExtractionStatus(row.extraction_status),
                content_hash=row.content_hash,
                collapsed=row.superseded_by_snapshot_id is not None,
            )
        )
    return grouped


async def mark_snapshot_superseded(
    session: AsyncSession, snapshot_id: str, *, superseded_by: str
) -> bool:
    """Collapse one snapshot: ``COMPLETE`` plus the id of the generation that replaced it.

    Conditional on the row still being ``PENDING`` / ``FAILED`` and not yet
    collapsed, so a snapshot another runner claimed between the candidate read
    and this write is left alone. Returns whether the row was marked. Caller
    commits.

    ``synchronize_session="fetch"`` keeps this session's identity map in step
    with the write: the session factory does not expire on commit, so a
    ``SnapshotRow`` already loaded here would otherwise keep reading as
    uncollapsed, and a later in-session write of ``superseded_by = None``
    would compare equal to that stale value and be dropped from the flush.
    """
    result = await session.execute(
        update(SnapshotRow)
        .where(
            SnapshotRow.snapshot_id == snapshot_id,
            SnapshotRow.extraction_status.in_(
                [ExtractionStatus.PENDING.value, ExtractionStatus.FAILED.value]
            ),
            SnapshotRow.superseded_by_snapshot_id.is_(None),
        )
        .values(
            extraction_status=ExtractionStatus.COMPLETE.value,
            extraction_started_at=None,
            superseded_by_snapshot_id=superseded_by,
        )
        .execution_options(synchronize_session="fetch")
    )
    return bool(getattr(result, "rowcount", 0) == 1)


async def get_snapshot_superseded_by(session: AsyncSession, snapshot_id: str) -> str | None:
    """The id of the generation that collapsed this snapshot, or None.

    Read fresh from the database rather than the identity map: the mark is
    written by a conditional bulk UPDATE, possibly from another process.
    """
    result = await session.execute(
        select(SnapshotRow.superseded_by_snapshot_id)
        .where(SnapshotRow.snapshot_id == snapshot_id)
        .execution_options(populate_existing=True)
    )
    value = result.scalar_one_or_none()
    return str(value) if value is not None else None


async def list_collapsed_snapshot_ids(session: AsyncSession) -> set[str]:
    """Ids of every collapsed snapshot, for the empty-COMPLETE lint."""
    result = await session.execute(
        select(SnapshotRow.snapshot_id).where(SnapshotRow.superseded_by_snapshot_id.is_not(None))
    )
    return {str(sid) for sid in result.scalars()}


async def list_refreshable_local_entries(session: AsyncSession) -> list[tuple[str, str]]:
    """``(entry_id, uri_r)`` for every LAZY corpus entry with a ``file://`` URI-R.

    The refresh pass's work list. The ``LAZY`` filter is the
    gate, reused deliberately: ``deposit_file`` defaults to ``NEVER``, so nothing
    already in a store starts refreshing on upgrade — an operator opts in per
    entry, which is the same operator promise mutability keys on.

    Ordered oldest-entry-first so a capped run makes round-robin progress rather
    than re-checking the same head every night.
    """
    result = await session.execute(
        select(CorpusEntryRow.entry_id, CorpusEntryRow.uri_r)
        .where(
            CorpusEntryRow.fetch_policy == FetchPolicy.LAZY.value,
            CorpusEntryRow.uri_r.is_not(None),
            CorpusEntryRow.uri_r.startswith("file://"),
        )
        .order_by(CorpusEntryRow.created_at.asc())
    )
    return [(str(row[0]), str(row[1])) for row in result.all()]


async def list_mutable_entries_with_multiple_snapshots(
    session: AsyncSession,
) -> list[tuple[str, str | None]]:
    """``(entry_id, uri_r)`` for MUTABLE entries carrying more than one snapshot.

    The backfill's work list: exactly the entries whose document
    generation has already moved at least once, and which therefore may hold
    ACTIVE particles anchored to a superseded snapshot.
    """
    snapshot_count = (
        select(SnapshotRow.entry_id.label("entry_id"), func.count().label("n"))
        .group_by(SnapshotRow.entry_id)
        .having(func.count() > 1)
        .subquery()
    )
    result = await session.execute(
        select(CorpusEntryRow.entry_id, CorpusEntryRow.uri_r)
        .join(snapshot_count, snapshot_count.c.entry_id == CorpusEntryRow.entry_id)
        .where(CorpusEntryRow.mutability == Mutability.MUTABLE.value)
        .order_by(CorpusEntryRow.created_at.asc())
    )
    return [(str(row[0]), row[1]) for row in result.all()]


async def list_entry_status_pairs_with_extraction_status(
    session: AsyncSession, statuses: list[ExtractionStatus]
) -> list[tuple[str, str]]:
    """Return distinct (entry_id, extraction_status) pairs for snapshots in given statuses."""
    result = await session.execute(
        select(SnapshotRow.entry_id, SnapshotRow.extraction_status)
        .where(SnapshotRow.extraction_status.in_([s.value for s in statuses]))
        .distinct()
    )
    return [(str(row[0]), str(row[1])) for row in result.all()]


async def list_replaced_mutable_snapshot_ids(session: AsyncSession) -> set[str]:
    """COMPLETE RESPONSE snapshots of MUTABLE entries whose replacement is in the store.

    "Replaced" means a newer RESPONSE sibling is COMPLETE and not collapsed —
    an *extracted* later generation. Used by the empty-COMPLETE lint:
    re-extracting such a snapshot could only retire the generation that
    replaced it. A newer sibling that is FAILED, PENDING or collapsed does not
    count, so an empty snapshot that may be the entry's best surviving
    generation stays visible.
    """
    newer = aliased(SnapshotRow)
    result = await session.execute(
        select(SnapshotRow.snapshot_id)
        .join(CorpusEntryRow, CorpusEntryRow.entry_id == SnapshotRow.entry_id)
        .join(newer, newer.entry_id == SnapshotRow.entry_id)
        .where(
            CorpusEntryRow.mutability == Mutability.MUTABLE.value,
            SnapshotRow.extraction_status == ExtractionStatus.COMPLETE.value,
            SnapshotRow.warc_record_type == WarcRecordType.RESPONSE.value,
            newer.warc_record_type == WarcRecordType.RESPONSE.value,
            newer.extraction_status == ExtractionStatus.COMPLETE.value,
            newer.superseded_by_snapshot_id.is_(None),
            newer.captured_at > SnapshotRow.captured_at,
        )
        .distinct()
    )
    return {str(sid) for sid in result.scalars()}
