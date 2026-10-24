# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""An unchanged re-deposit writes no snapshot.

``write_entry_and_snapshot`` used to append a ``PENDING`` snapshot on every
re-deposit, so re-running ``particles import vault`` queued every unchanged
note for extraction again. These tests pin the seam: a first deposit and a
changed deposit snapshot as before, and a deposit whose bytes equal the
entry's latest snapshot returns the existing IDs and adds nothing.

No test here calls a model, runs a hook, or runs consolidation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    ExtractionStatus,
    Snapshot,
    SourceType,
    WarcRecordType,
)
from particles.corpus.deposit import deposit_file, deposit_text, sha256
from particles.corpus.store import (
    CorpusEntryRow,
    SnapshotRow,
    get_latest_snapshot,
    update_extraction_status,
)


async def _snapshots(session: AsyncSession, entry_id: str) -> list[SnapshotRow]:
    rows = await session.execute(
        select(SnapshotRow)
        .where(SnapshotRow.entry_id == entry_id)
        .order_by(SnapshotRow.captured_at)
    )
    return list(rows.scalars())


async def _count(session: AsyncSession, model: type[CorpusEntryRow] | type[SnapshotRow]) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


@pytest.mark.asyncio
class TestDepositFileUnchanged:
    async def test_first_deposit_snapshots_pending(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        note = tmp_path / "note.md"
        note.write_text("# Note\n\nThe first version.\n")

        unchanged: list[bool] = []
        entry_id, snapshot_id = await deposit_file(db_session, note, out_unchanged=unchanged)

        assert unchanged == [False]
        snaps = await _snapshots(db_session, entry_id)
        assert [s.snapshot_id for s in snaps] == [snapshot_id]
        assert snaps[0].extraction_status == ExtractionStatus.PENDING.value

    async def test_unchanged_redeposit_writes_nothing(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        note = tmp_path / "note.md"
        note.write_text("# Note\n\nSame bytes every time.\n")
        first = await deposit_file(db_session, note)
        await db_session.commit()

        unchanged: list[bool] = []
        second = await deposit_file(db_session, note, out_unchanged=unchanged)
        await db_session.commit()

        assert unchanged == [True]
        assert second == first
        assert len(await _snapshots(db_session, first[0])) == 1

    async def test_unchanged_after_extraction_queues_nothing(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        """The common re-import case: the note was extracted, then re-deposited."""
        note = tmp_path / "note.md"
        note.write_text("# Note\n\nAlready extracted.\n")
        entry_id, snapshot_id = await deposit_file(db_session, note)
        await update_extraction_status(db_session, snapshot_id, ExtractionStatus.COMPLETE)
        await db_session.commit()

        assert await deposit_file(db_session, note) == (entry_id, snapshot_id)
        snaps = await _snapshots(db_session, entry_id)
        assert len(snaps) == 1
        assert snaps[0].extraction_status == ExtractionStatus.COMPLETE.value

    async def test_changed_content_snapshots_pending(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        note = tmp_path / "note.md"
        note.write_text("# Note\n\nVersion one.\n")
        entry_id, first_snap = await deposit_file(db_session, note)
        await db_session.commit()

        note.write_text("# Note\n\nVersion two.\n")
        unchanged: list[bool] = []
        same_entry, second_snap = await deposit_file(db_session, note, out_unchanged=unchanged)

        assert unchanged == [False]
        assert same_entry == entry_id
        assert second_snap != first_snap
        snaps = await _snapshots(db_session, entry_id)
        assert [s.snapshot_id for s in snaps] == [first_snap, second_snap]
        assert snaps[-1].extraction_status == ExtractionStatus.PENDING.value

    async def test_revert_to_an_older_version_is_a_change(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        """Only the latest snapshot counts: bytes matching an older one still snapshot."""
        note = tmp_path / "note.md"
        note.write_text("A\n")
        entry_id, _ = await deposit_file(db_session, note)
        note.write_text("B\n")
        await deposit_file(db_session, note)

        note.write_text("A\n")
        unchanged: list[bool] = []
        await deposit_file(db_session, note, out_unchanged=unchanged)

        assert unchanged == [False]
        assert len(await _snapshots(db_session, entry_id)) == 3

    async def test_changed_source_type_still_snapshots(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        """Re-depositing as a more specific type is a request for re-extraction."""
        note = tmp_path / "note.txt"
        note.write_text("Plain text that is really Markdown.\n")
        entry_id, _ = await deposit_file(db_session, note)
        await db_session.commit()

        unchanged: list[bool] = []
        await deposit_file(
            db_session, note, source_type=SourceType.LOCAL_MARKDOWN, out_unchanged=unchanged
        )

        assert unchanged == [False]
        assert len(await _snapshots(db_session, entry_id)) == 2
        row = await db_session.get(CorpusEntryRow, entry_id)
        assert row is not None
        assert row.source_type == SourceType.LOCAL_MARKDOWN.value

    async def test_a_new_explicit_date_is_a_re_dating(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        """Re-deposit with ``--date`` then reindex re-dates an archival entry."""
        doc = tmp_path / "concept.md"
        doc.write_text("An old design note.\n")
        entry_id, _ = await deposit_file(db_session, doc)

        unchanged: list[bool] = []
        dated = datetime(2012, 1, 2, tzinfo=UTC)
        await deposit_file(db_session, doc, content_date=dated, out_unchanged=unchanged)
        await deposit_file(db_session, doc, content_date=dated, out_unchanged=unchanged)

        assert unchanged == [False, True]  # the same date again is unchanged
        snaps = await _snapshots(db_session, entry_id)
        assert len(snaps) == 2
        assert snaps[-1].content_published_at is not None
        assert snaps[-1].content_published_at.date() == dated.date()

    async def test_moved_file_with_same_bytes_is_unchanged(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        """Stage-2 content-hash dedup lands on the same entry and writes nothing."""
        original = tmp_path / "a" / "note.md"
        original.parent.mkdir()
        original.write_text("# Moved\n")
        first = await deposit_file(db_session, original)
        await db_session.commit()

        moved = tmp_path / "b" / "note.md"
        moved.parent.mkdir()
        moved.write_bytes(original.read_bytes())
        unchanged: list[bool] = []
        second = await deposit_file(db_session, moved, out_unchanged=unchanged)

        assert unchanged == [True]
        assert second == first
        assert await _count(db_session, CorpusEntryRow) == 1
        assert await _count(db_session, SnapshotRow) == 1

    async def test_latest_revisit_returns_the_snapshot_holding_the_bytes(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        """A REVISIT chain from the refresh ladder resolves to its RESPONSE."""
        note = tmp_path / "note.md"
        note.write_text("# Revisited\n")
        entry_id, response_id = await deposit_file(db_session, note)

        # Two chained REVISITs, the shape corpus/fetch.py writes when each
        # refresh compares against the then-latest snapshot.
        prior_id = response_id
        base = datetime.now(UTC)
        for hours in (1, 2):
            revisit = Snapshot(
                captured_at=base + timedelta(hours=hours),
                content_hash=sha256(note.read_bytes()),
                warc_record_type=WarcRecordType.REVISIT,
                archive_path=None,
                refers_to=prior_id,
                extraction_status=ExtractionStatus.COMPLETE,
            )
            db_session.add(SnapshotRow.from_model(revisit, entry_id=entry_id))
            prior_id = revisit.snapshot_id
        await db_session.flush()
        latest = await get_latest_snapshot(db_session, entry_id)
        assert latest is not None and latest.snapshot_id == prior_id

        unchanged: list[bool] = []
        assert await deposit_file(db_session, note, out_unchanged=unchanged) == (
            entry_id,
            response_id,
        )
        assert unchanged == [True]
        assert len(await _snapshots(db_session, entry_id)) == 3


@pytest.mark.asyncio
class TestDepositTextUnchanged:
    async def test_same_text_twice_is_one_snapshot(self, db_session: AsyncSession) -> None:
        first = await deposit_text(db_session, "One fact, deposited twice.")
        unchanged: list[bool] = []
        second = await deposit_text(
            db_session, "One fact, deposited twice.", out_unchanged=unchanged
        )

        assert unchanged == [True]
        assert second == first
        assert await _count(db_session, SnapshotRow) == 1

    async def test_different_text_is_a_new_entry(self, db_session: AsyncSession) -> None:
        unchanged: list[bool] = []
        await deposit_text(db_session, "First text.", out_unchanged=unchanged)
        await deposit_text(db_session, "Second text.", out_unchanged=unchanged)

        assert unchanged == [False, False]
        assert await _count(db_session, CorpusEntryRow) == 2
