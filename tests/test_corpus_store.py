# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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


class TestCatchupOrdering:
    """The consolidation catch-up queue: least-tried first, then oldest.

    Measured on the owner's store 2026-09-25: the same handful of transcript
    snapshots failed every night, were handed back PENDING, and led the queue
    again the next night, while 337 snapshots waited behind them.
    """

    @staticmethod
    async def _seed(session: object, rows: list[tuple[str, int]]) -> None:
        """One entry per ``(snapshot_id, age_days)``, every snapshot PENDING."""
        now = datetime.now(UTC)
        for snapshot_id, age in rows:
            entry_id = f"e-{snapshot_id}"
            session.add(CorpusEntryRow.from_model(_entry(entry_id, None)))  # type: ignore[attr-defined]
            snap = Snapshot(
                snapshot_id=snapshot_id,
                captured_at=now - timedelta(days=age),
                content_hash="b" * 64,
                archive_path="/x",
                extraction_status=ExtractionStatus.PENDING,
                warc_record_type=WarcRecordType.RESPONSE,
            )
            session.add(SnapshotRow.from_model(snap, entry_id))  # type: ignore[attr-defined]
        await session.commit()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_claim_counts_attempts(self, db_session: object) -> None:
        from particles.corpus.store import claim_snapshot_for_extraction, update_extraction_status

        session = db_session
        await self._seed(session, [("s-1", 1)])
        for _ in range(2):
            await claim_snapshot_for_extraction(session, "s-1", started_at=datetime.now(UTC))  # type: ignore[arg-type]
            # The pipeline's hand-back after every LLM call failed.
            await update_extraction_status(session, "s-1", ExtractionStatus.PENDING)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[attr-defined]

        row = await session.get(SnapshotRow, "s-1")  # type: ignore[attr-defined]
        assert row.extraction_attempts == 2
        assert row.extraction_status == ExtractionStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_a_tried_snapshot_waits_behind_untried_ones(self, db_session: object) -> None:
        from particles.corpus.store import (
            claim_snapshot_for_extraction,
            list_pending_snapshots_for_catchup,
            update_extraction_status,
        )

        session = db_session
        await self._seed(session, [("old-poison", 60), ("mid", 30), ("new", 1)])
        listed = await list_pending_snapshots_for_catchup(session)  # type: ignore[arg-type]
        assert [p.snapshot_id for p in listed] == ["old-poison", "mid", "new"]

        await claim_snapshot_for_extraction(  # type: ignore[arg-type]
            session, "old-poison", started_at=datetime.now(UTC)
        )
        await update_extraction_status(session, "old-poison", ExtractionStatus.PENDING)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[attr-defined]

        listed = await list_pending_snapshots_for_catchup(session)  # type: ignore[arg-type]
        assert [p.snapshot_id for p in listed] == ["mid", "new", "old-poison"]
        assert [p.attempts for p in listed] == [0, 0, 1]


class TestOwedSinceOrdering:
    """the queue key is how long an entry has owed extraction.

    Measured on the owner's store 2026-10-04: the collapse left each
    rule file one pending snapshot dated at its latest edit, so every edit sent
    the file to the back of a three-to-seven-week queue, and no rule-file
    generation captured after 2026-08-04 was ever extracted.
    """

    @staticmethod
    def _add(
        session: object,
        entry_id: str,
        snapshot_id: str,
        age_days: float,
        status: ExtractionStatus,
        *,
        superseded_by: str | None = None,
    ) -> None:
        snap = Snapshot(
            snapshot_id=snapshot_id,
            captured_at=datetime.now(UTC) - timedelta(days=age_days),
            content_hash="b" * 64,
            archive_path="/x",
            extraction_status=status,
            warc_record_type=WarcRecordType.RESPONSE,
        )
        row = SnapshotRow.from_model(snap, entry_id)
        row.superseded_by_snapshot_id = superseded_by
        session.add(row)  # type: ignore[attr-defined]

    @staticmethod
    def _entry(
        session: object, entry_id: str, mutability: Mutability, tags: list[str] | None = None
    ) -> None:
        entry = CorpusEntry(
            entry_id=entry_id,
            uri_r=None,
            source_type="LOCAL_MARKDOWN",
            mutability=mutability,
            fetch_policy=FetchPolicy.NEVER,
            deposited_by="test",
            tags=tags or [],
        )
        session.add(CorpusEntryRow.from_model(entry))  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_a_collapsed_rule_file_keeps_its_place_in_line(self, db_session: object) -> None:
        """Acceptance 1: the entry owed since day 40 leads, though its snapshot is newest."""
        from particles.corpus.store import list_pending_snapshots_for_catchup

        session = db_session
        self._entry(session, "rules", Mutability.MUTABLE, ["rule-file"])
        self._add(session, "rules", "r-extracted", 60, ExtractionStatus.COMPLETE)
        self._add(session, "rules", "r-gen2", 40, ExtractionStatus.COMPLETE, superseded_by="r-gen4")
        self._add(session, "rules", "r-gen3", 20, ExtractionStatus.COMPLETE, superseded_by="r-gen4")
        self._add(session, "rules", "r-gen4", 0.1, ExtractionStatus.PENDING)
        self._entry(session, "chat", Mutability.APPEND_ONLY)
        self._add(session, "chat", "c-1", 30, ExtractionStatus.PENDING)
        await session.commit()  # type: ignore[attr-defined]

        listed = await list_pending_snapshots_for_catchup(session)  # type: ignore[arg-type]

        assert [p.snapshot_id for p in listed] == ["r-gen4", "c-1"]
        rules = listed[0]
        assert rules.tags == frozenset({"rule-file"})
        assert rules.owed_since is not None
        assert rules.waiting_since < listed[1].waiting_since

    @pytest.mark.asyncio
    async def test_a_collapse_before_the_last_extraction_does_not_count(
        self, db_session: object
    ) -> None:
        """Work collapsed into a generation that was then extracted is no longer owed."""
        from particles.corpus.store import list_pending_snapshots_for_catchup

        session = db_session
        self._entry(session, "rules", Mutability.MUTABLE)
        self._add(session, "rules", "r-old", 50, ExtractionStatus.COMPLETE, superseded_by="r-x")
        self._add(session, "rules", "r-x", 45, ExtractionStatus.COMPLETE)
        self._add(session, "rules", "r-new", 2, ExtractionStatus.PENDING)
        self._entry(session, "chat", Mutability.APPEND_ONLY)
        self._add(session, "chat", "c-1", 10, ExtractionStatus.PENDING)
        await session.commit()  # type: ignore[attr-defined]

        listed = await list_pending_snapshots_for_catchup(session)  # type: ignore[arg-type]

        assert [p.snapshot_id for p in listed] == ["c-1", "r-new"]

    @pytest.mark.asyncio
    async def test_one_transcript_is_read_oldest_snapshot_first(self, db_session: object) -> None:
        """Acceptance 2: shared owed-since must not fall through to the random snapshot id.

        Ids are chosen so their sort order is the reverse of capture order;
        delta reads and rule 4 need the oldest first.
        """
        from particles.corpus.store import list_pending_snapshots_for_catchup
        from particles.operations.consolidation import _one_per_entry

        session = db_session
        self._entry(session, "chat", Mutability.APPEND_ONLY)
        self._add(session, "chat", "z-oldest", 9, ExtractionStatus.PENDING)
        self._add(session, "chat", "m-middle", 5, ExtractionStatus.PENDING)
        self._add(session, "chat", "a-newest", 1, ExtractionStatus.PENDING)
        await session.commit()  # type: ignore[attr-defined]

        listed = await list_pending_snapshots_for_catchup(session)  # type: ignore[arg-type]

        assert [p.snapshot_id for p in listed] == ["z-oldest", "m-middle", "a-newest"]
        assert len({p.waiting_since for p in listed}) == 1
        assert _one_per_entry(listed, 20) == [("chat", "z-oldest")]

    @pytest.mark.asyncio
    async def test_attempts_still_sort_ahead_of_owed_since(self, db_session: object) -> None:
        """Acceptance 3: the v1.151.26 correction survives a long-owed entry."""
        from particles.corpus.store import list_pending_snapshots_for_catchup

        session = db_session
        self._entry(session, "rules", Mutability.MUTABLE)
        self._add(session, "rules", "r-gen1", 90, ExtractionStatus.COMPLETE, superseded_by="r-2")
        self._add(session, "rules", "r-2", 1, ExtractionStatus.PENDING)
        self._entry(session, "chat", Mutability.APPEND_ONLY)
        self._add(session, "chat", "c-1", 5, ExtractionStatus.PENDING)
        await session.commit()  # type: ignore[attr-defined]
        row = await session.get(SnapshotRow, "r-2")  # type: ignore[attr-defined]
        row.extraction_attempts = 1
        await session.commit()  # type: ignore[attr-defined]

        listed = await list_pending_snapshots_for_catchup(session)  # type: ignore[arg-type]

        assert [p.snapshot_id for p in listed] == ["c-1", "r-2"]


class TestOwedSincePure:
    """``owed_since`` is a pure decision over capture times (D2)."""

    T0 = datetime(2026, 8, 1, tzinfo=UTC)

    def _t(self, days: int) -> datetime:
        return self.T0 + timedelta(days=days)

    def test_no_collapse_is_the_oldest_pending_capture(self) -> None:
        from particles.corpus.store import owed_since

        assert owed_since([self._t(5), self._t(2)], [], None) == self._t(2)

    def test_collapsed_after_the_last_extraction_moves_it_earlier(self) -> None:
        from particles.corpus.store import owed_since

        got = owed_since([self._t(60)], [self._t(3), self._t(30)], self._t(1))
        assert got == self._t(3)

    def test_collapsed_before_the_last_extraction_is_ignored(self) -> None:
        from particles.corpus.store import owed_since

        assert owed_since([self._t(60)], [self._t(3)], self._t(10)) == self._t(60)

    def test_never_extracted_counts_every_collapse(self) -> None:
        from particles.corpus.store import owed_since

        assert owed_since([self._t(60)], [self._t(0)], None) == self._t(0)


@pytest.mark.asyncio
async def test_release_extraction_claim_only_releases_in_progress(
    db_session: object, tmp_path: object
) -> None:
    """The extraction failure path hands back its own IN_PROGRESS claim, and
    never overwrites a status the snapshot already moved to (FAILED on a
    missing blob, PENDING on a transient reset, COMPLETE at the write)."""
    from pathlib import Path

    from particles.corpus.deposit import deposit_file
    from particles.corpus.store import (
        claim_snapshot_for_extraction,
        release_extraction_claim,
        update_extraction_status,
    )

    session = db_session  # type: ignore[assignment]
    base = Path(str(tmp_path))
    ids = []
    for name in ("claimed", "done", "failed"):
        doc = base / f"{name}.txt"
        doc.write_text(f"{name} content")
        ids.append((await deposit_file(session, doc, deposited_by="test"))[1])  # type: ignore[arg-type]
    claimed, done, failed = ids
    now = datetime.now(UTC)
    for snapshot_id in ids:
        await claim_snapshot_for_extraction(session, snapshot_id, started_at=now)  # type: ignore[arg-type]
    await update_extraction_status(session, done, ExtractionStatus.COMPLETE)  # type: ignore[arg-type]
    await update_extraction_status(session, failed, ExtractionStatus.FAILED)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[attr-defined]

    released = [
        await release_extraction_claim(session, snapshot_id)  # type: ignore[arg-type]
        for snapshot_id in ids
    ]
    await session.commit()  # type: ignore[attr-defined]

    assert released == [True, False, False]
    rows = {sid: await session.get(SnapshotRow, sid) for sid in ids}  # type: ignore[attr-defined]
    for row in rows.values():
        await session.refresh(row)  # type: ignore[attr-defined]
    assert rows[claimed].extraction_status == ExtractionStatus.PENDING.value
    assert rows[claimed].extraction_started_at is None
    assert rows[done].extraction_status == ExtractionStatus.COMPLETE.value
    assert rows[failed].extraction_status == ExtractionStatus.FAILED.value
