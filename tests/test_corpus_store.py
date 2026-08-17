"""Tests for repository helpers in particles/corpus/store.py.

Covers query-shape helpers added to back the Obsidian / Wiki exporters'
shared ``{entry_id: uri_r}`` lookup. See tests/AGENTS.md § What
requires tests — "Repository helpers in store/ that introduce a new
query shape".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from particles.core.schema import (
    CorpusEntry,
    ExtractionStatus,
    FetchPolicy,
    Mutability,
    Snapshot,
    WarcRecordType,
)
from particles.corpus.store import (
    CorpusEntryRow,
    SnapshotRow,
    get_entry_uri_map,
    resolve_snapshot_for_blob,
)


def _entry(entry_id: str, uri: str | None) -> CorpusEntry:
    return CorpusEntry(
        entry_id=entry_id,
        uri_r=uri,
        source_type="WEB_PAGE",
        mutability=Mutability.STABLE,
        fetch_policy=FetchPolicy.NEVER,
        deposited_by="test",
        tags=[],
    )


class TestGetEntryUriMap:
    @pytest.mark.asyncio
    async def test_none_filter_returns_every_entry(self, db_session: object) -> None:
        """Obsidian-style call (no filter) returns one row per CorpusEntry."""
        session = db_session  # type: ignore[assignment]
        for eid, uri in [
            ("e-1", "https://example.com/a"),
            ("e-2", "https://example.com/b"),
            ("e-3", None),  # NULL uri_r — synthetic / local-only entry
        ]:
            session.add(CorpusEntryRow.from_model(_entry(eid, uri)))  # type: ignore[union-attr]
        await session.commit()  # type: ignore[union-attr]

        result = await get_entry_uri_map(session)  # type: ignore[arg-type]
        assert result == {
            "e-1": "https://example.com/a",
            "e-2": "https://example.com/b",
            "e-3": None,
        }

    @pytest.mark.asyncio
    async def test_filter_restricts_to_requested_ids(self, db_session: object) -> None:
        """Wiki-style call (entry_ids set) returns only the requested rows."""
        session = db_session  # type: ignore[assignment]
        for eid in ["e-1", "e-2", "e-3"]:
            session.add(CorpusEntryRow.from_model(_entry(eid, f"https://x/{eid}")))  # type: ignore[union-attr]
        await session.commit()  # type: ignore[union-attr]

        result = await get_entry_uri_map(session, {"e-1", "e-3"})  # type: ignore[arg-type]
        assert result == {"e-1": "https://x/e-1", "e-3": "https://x/e-3"}

    @pytest.mark.asyncio
    async def test_empty_filter_skips_query(self, db_session: object) -> None:
        """Empty filter set short-circuits to ``{}`` without hitting the DB.

        Avoids emitting a malformed ``WHERE entry_id IN ()`` and is the
        cheap path the wiki exporter relies on when the corpus has no
        cited entries (every particle was self-asserted).
        """
        session = db_session  # type: ignore[assignment]
        session.add(CorpusEntryRow.from_model(_entry("e-1", "https://x/e-1")))  # type: ignore[union-attr]
        await session.commit()  # type: ignore[union-attr]

        result = await get_entry_uri_map(session, set())  # type: ignore[arg-type]
        assert result == {}


def _snap(snapshot_id: str, content_hash: str, *, captured: datetime) -> Snapshot:
    return Snapshot(
        snapshot_id=snapshot_id,
        captured_at=captured,
        content_hash=content_hash,
        extraction_status=ExtractionStatus.COMPLETE,
        warc_record_type=WarcRecordType.RESPONSE,
    )


class TestResolveSnapshotForBlob:
    """``resolve_snapshot_for_blob`` — the ``corpus cat`` selector resolver
    (snapshot-or-entry prefix → the snapshot whose blob to read)."""

    async def _seed(self, session: object) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        session.add(CorpusEntryRow.from_model(_entry("entry-aaaa-1111", "https://x/a")))  # type: ignore[union-attr]
        session.add(CorpusEntryRow.from_model(_entry("entry-bbbb-2222", "https://x/b")))  # type: ignore[union-attr]
        # entry-a has two snapshots; the later one is the "latest".
        session.add(  # type: ignore[union-attr]
            SnapshotRow.from_model(
                _snap("snap-old-0001", "a" * 64, captured=base), "entry-aaaa-1111"
            )
        )
        session.add(  # type: ignore[union-attr]
            SnapshotRow.from_model(
                _snap("snap-new-0002", "b" * 64, captured=base + timedelta(days=1)),
                "entry-aaaa-1111",
            )
        )
        session.add(  # type: ignore[union-attr]
            SnapshotRow.from_model(
                _snap("snap-cccc-0003", "c" * 64, captured=base), "entry-bbbb-2222"
            )
        )
        await session.commit()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_exact_snapshot_id(self, db_session: object) -> None:
        await self._seed(db_session)
        snap = await resolve_snapshot_for_blob(db_session, "snap-old-0001")  # type: ignore[arg-type]
        assert snap is not None
        assert snap.snapshot_id == "snap-old-0001"
        assert snap.content_hash == "a" * 64

    @pytest.mark.asyncio
    async def test_unique_snapshot_prefix(self, db_session: object) -> None:
        await self._seed(db_session)
        snap = await resolve_snapshot_for_blob(db_session, "snap-cccc")  # type: ignore[arg-type]
        assert snap is not None
        assert snap.snapshot_id == "snap-cccc-0003"

    @pytest.mark.asyncio
    async def test_entry_resolves_to_latest_snapshot(self, db_session: object) -> None:
        """An entry selector picks that entry's most-recently-captured snapshot."""
        await self._seed(db_session)
        snap = await resolve_snapshot_for_blob(db_session, "entry-aaaa-1111")  # type: ignore[arg-type]
        assert snap is not None
        assert snap.snapshot_id == "snap-new-0002"

    @pytest.mark.asyncio
    async def test_unique_entry_prefix(self, db_session: object) -> None:
        await self._seed(db_session)
        snap = await resolve_snapshot_for_blob(db_session, "entry-bbbb")  # type: ignore[arg-type]
        assert snap is not None
        assert snap.snapshot_id == "snap-cccc-0003"

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, db_session: object) -> None:
        await self._seed(db_session)
        assert await resolve_snapshot_for_blob(db_session, "nope-nope") is None  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_ambiguous_snapshot_prefix_raises(self, db_session: object) -> None:
        await self._seed(db_session)
        with pytest.raises(ValueError, match="Ambiguous snapshot prefix"):
            await resolve_snapshot_for_blob(db_session, "snap-")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_ambiguous_entry_prefix_raises(self, db_session: object) -> None:
        """A prefix that matches no snapshot but multiple entries is ambiguous."""
        await self._seed(db_session)
        with pytest.raises(ValueError, match="Ambiguous entry prefix"):
            await resolve_snapshot_for_blob(db_session, "entry-")  # type: ignore[arg-type]
