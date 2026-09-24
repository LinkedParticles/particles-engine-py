# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the collapse of superseded, unextracted snapshots.

The rule under test: a PENDING or FAILED RESPONSE snapshot of a MUTABLE entry
is collapsed (COMPLETE + ``superseded_by_snapshot_id``) when a newer RESPONSE
sibling is PENDING / IN_PROGRESS / COMPLETE, not itself collapsed, and has its
blob on disk. Everything else here is a boundary of that sentence: the other
mutability classes, the siblings that must *not* supersede, the conditional
write, and the two readers that must tell a collapsed snapshot from an
extracted one.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from particles.config import get_config
from particles.core.schema import (
    CorpusEntry,
    ExtractionStatus,
    FetchPolicy,
    Mutability,
    Snapshot,
    WarcRecordType,
)
from particles.corpus.deposit import save_blob, sha256
from particles.corpus.store import (
    CorpusEntryRow,
    SnapshotRow,
    claim_snapshot_for_extraction,
    get_latest_completed_snapshot_id,
    get_snapshot_superseded_by,
    list_collapsed_snapshot_ids,
    list_generations_with_unextracted_snapshots,
    list_replaced_mutable_snapshot_ids,
    mark_snapshot_superseded,
)
from particles.ingest.pending_collapse import collapse_superseded_pending
from particles.ingest.pipeline import extract_snapshot

P = ExtractionStatus.PENDING
F = ExtractionStatus.FAILED
C = ExtractionStatus.COMPLETE
IP = ExtractionStatus.IN_PROGRESS


async def _entry(session: Any, mutability: Mutability = Mutability.MUTABLE) -> CorpusEntry:
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="LOCAL_MARKDOWN",
        uri_r=f"file:///tmp/{uuid.uuid4().hex}.md",
        fetch_policy=FetchPolicy.NEVER,
        mutability=mutability,
        deposited_by="test",
    )
    session.add(CorpusEntryRow.from_model(entry))
    await session.flush()
    return entry


async def _snap(
    session: Any,
    entry: CorpusEntry,
    *,
    age: int,
    status: ExtractionStatus = P,
    blob: bool = True,
    revisit_of: Snapshot | None = None,
) -> Snapshot:
    """One snapshot, ``age`` days old. ``blob=False`` leaves the blob off disk."""
    content = f"generation {uuid.uuid4().hex}".encode()
    content_hash = revisit_of.content_hash if revisit_of else sha256(content)
    archive_path = None
    if revisit_of is None and blob:
        archive_path = save_blob(content, content_hash)
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC) - timedelta(days=age),
        content_hash=content_hash,
        warc_record_type=WarcRecordType.REVISIT if revisit_of else WarcRecordType.RESPONSE,
        archive_path=archive_path,
        refers_to=revisit_of.snapshot_id if revisit_of else None,
        extraction_status=status,
    )
    session.add(SnapshotRow.from_model(snap, entry.entry_id))
    await session.flush()
    return snap


async def _state(session: Any, snap: Snapshot) -> tuple[str, str | None]:
    row = await session.get(SnapshotRow, snap.snapshot_id)
    await session.refresh(row)
    return row.extraction_status, row.superseded_by_snapshot_id


class TestTheRule:
    @pytest.mark.asyncio
    async def test_older_pending_generations_collapse_onto_the_newest(
        self, db_session: Any
    ) -> None:
        entry = await _entry(db_session)
        s1 = await _snap(db_session, entry, age=3)
        s2 = await _snap(db_session, entry, age=2)
        s3 = await _snap(db_session, entry, age=1)
        await db_session.commit()

        report = await collapse_superseded_pending(db_session)

        assert report.snapshots == 2 and report.entries == 1
        # Both name the generation that will actually be extracted, not s2.
        assert await _state(db_session, s1) == (C.value, s3.snapshot_id)
        assert await _state(db_session, s2) == (C.value, s3.snapshot_id)
        assert await _state(db_session, s3) == (P.value, None)
        assert "2 superseded snapshot(s) across 1 MUTABLE entry" in (report.summary() or "")

    @pytest.mark.asyncio
    async def test_a_lone_pending_snapshot_is_left_alone(self, db_session: Any) -> None:
        entry = await _entry(db_session)
        only = await _snap(db_session, entry, age=1)
        await db_session.commit()

        report = await collapse_superseded_pending(db_session)

        assert report.snapshots == 0 and report.summary() is None
        assert await _state(db_session, only) == (P.value, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mutability", [Mutability.APPEND_ONLY, Mutability.STABLE])
    async def test_only_mutable_entries_collapse(
        self, db_session: Any, mutability: Mutability
    ) -> None:
        """APPEND_ONLY is additive and STABLE versions are deliberate: both keep history."""
        entry = await _entry(db_session, mutability)
        old = await _snap(db_session, entry, age=2)
        await _snap(db_session, entry, age=1)
        await db_session.commit()

        report = await collapse_superseded_pending(db_session)

        assert report.snapshots == 0
        assert await _state(db_session, old) == (P.value, None)

    @pytest.mark.asyncio
    async def test_a_newer_complete_generation_supersedes(self, db_session: Any) -> None:
        """The ordering hazard: extracting `old` now would retire the current generation."""
        entry = await _entry(db_session)
        old = await _snap(db_session, entry, age=2, status=P)
        current = await _snap(db_session, entry, age=1, status=C)
        await db_session.commit()

        await collapse_superseded_pending(db_session)

        assert await _state(db_session, old) == (C.value, current.snapshot_id)

    @pytest.mark.asyncio
    async def test_a_newer_in_progress_generation_supersedes(self, db_session: Any) -> None:
        entry = await _entry(db_session)
        old = await _snap(db_session, entry, age=2)
        running = await _snap(db_session, entry, age=1, status=IP)
        await db_session.commit()

        await collapse_superseded_pending(db_session)

        assert await _state(db_session, old) == (C.value, running.snapshot_id)
        assert await _state(db_session, running) == (IP.value, None)

    @pytest.mark.asyncio
    async def test_a_failed_older_generation_collapses_too(self, db_session: Any) -> None:
        """`reindex` retries FAILED snapshots; an old one must not run after a newer one."""
        entry = await _entry(db_session)
        failed = await _snap(db_session, entry, age=2, status=F, blob=False)
        newest = await _snap(db_session, entry, age=1)
        await db_session.commit()

        await collapse_superseded_pending(db_session)

        assert await _state(db_session, failed) == (C.value, newest.snapshot_id)


class TestWhatNeverSupersedes:
    @pytest.mark.asyncio
    async def test_a_revisit_does_not(self, db_session: Any) -> None:
        """A REVISIT says the content did not change: `pending` is still current."""
        entry = await _entry(db_session)
        pending = await _snap(db_session, entry, age=2)
        await _snap(db_session, entry, age=1, status=C, revisit_of=pending)
        await db_session.commit()

        report = await collapse_superseded_pending(db_session)

        assert report.snapshots == 0
        assert await _state(db_session, pending) == (P.value, None)

    @pytest.mark.asyncio
    async def test_a_failed_newer_snapshot_does_not(self, db_session: Any) -> None:
        entry = await _entry(db_session)
        pending = await _snap(db_session, entry, age=2)
        await _snap(db_session, entry, age=1, status=F)
        await db_session.commit()

        assert (await collapse_superseded_pending(db_session)).snapshots == 0
        assert await _state(db_session, pending) == (P.value, None)

    @pytest.mark.asyncio
    async def test_a_newer_snapshot_with_no_blob_on_disk_does_not(self, db_session: Any) -> None:
        """Collapsing onto an unextractable snapshot would leave the entry with nothing."""
        entry = await _entry(db_session)
        pending = await _snap(db_session, entry, age=2)
        await _snap(db_session, entry, age=1, blob=False)
        await db_session.commit()

        assert (await collapse_superseded_pending(db_session)).snapshots == 0
        assert await _state(db_session, pending) == (P.value, None)

    @pytest.mark.asyncio
    async def test_the_superseder_is_the_newest_extractable_generation(
        self, db_session: Any
    ) -> None:
        entry = await _entry(db_session)
        s1 = await _snap(db_session, entry, age=3)
        s2 = await _snap(db_session, entry, age=2)
        s3 = await _snap(db_session, entry, age=1, blob=False)
        await db_session.commit()

        await collapse_superseded_pending(db_session)

        assert await _state(db_session, s1) == (C.value, s2.snapshot_id)
        assert await _state(db_session, s2) == (P.value, None)
        assert await _state(db_session, s3) == (P.value, None)

    @pytest.mark.asyncio
    async def test_a_collapsed_snapshot_does_not(self, db_session: Any) -> None:
        """COMPLETE-because-collapsed is not an extracted generation."""
        entry = await _entry(db_session)
        s1 = await _snap(db_session, entry, age=3)
        s2 = await _snap(db_session, entry, age=2)
        s3 = await _snap(db_session, entry, age=1)
        await db_session.commit()
        await collapse_superseded_pending(db_session)  # s1, s2 -> s3
        # s3 then fails for good, and an operator resets s1 by hand.
        row3 = await db_session.get(SnapshotRow, s3.snapshot_id)
        row3.extraction_status = F.value
        row1 = await db_session.get(SnapshotRow, s1.snapshot_id)
        row1.extraction_status = P.value
        row1.superseded_by_snapshot_id = None
        await db_session.commit()

        report = await collapse_superseded_pending(db_session)

        # s2 is COMPLETE and newer than s1, but collapsed: it supersedes nothing.
        assert report.snapshots == 0
        assert await _state(db_session, s1) == (P.value, None)
        assert (await _state(db_session, s2))[1] == s3.snapshot_id


class TestScopeAndSwitches:
    @pytest.mark.asyncio
    async def test_knob_off_is_a_no_op(self, db_session: Any) -> None:
        get_config().extraction.collapse_superseded_pending = False
        entry = await _entry(db_session)
        old = await _snap(db_session, entry, age=2)
        await _snap(db_session, entry, age=1)
        await db_session.commit()

        assert (await collapse_superseded_pending(db_session)).snapshots == 0
        assert await _state(db_session, old) == (P.value, None)

    @pytest.mark.asyncio
    async def test_entry_ids_scope_the_pass(self, db_session: Any) -> None:
        inside = await _entry(db_session)
        outside = await _entry(db_session)
        old_in = await _snap(db_session, inside, age=2)
        await _snap(db_session, inside, age=1)
        old_out = await _snap(db_session, outside, age=2)
        await _snap(db_session, outside, age=1)
        await db_session.commit()

        report = await collapse_superseded_pending(db_session, entry_ids=[inside.entry_id])

        assert report.snapshots == 1
        assert (await _state(db_session, old_in))[0] == C.value
        assert await _state(db_session, old_out) == (P.value, None)
        assert (await collapse_superseded_pending(db_session, entry_ids=[])).snapshots == 0

    @pytest.mark.asyncio
    async def test_dry_run_reports_without_writing(self, db_session: Any) -> None:
        entry = await _entry(db_session)
        old = await _snap(db_session, entry, age=2)
        newest = await _snap(db_session, entry, age=1)
        await db_session.commit()

        report = await collapse_superseded_pending(db_session, dry_run=True)

        assert report.collapsed == [(entry.entry_id, old.snapshot_id, newest.snapshot_id)]
        assert await _state(db_session, old) == (P.value, None)

    @pytest.mark.asyncio
    async def test_second_pass_is_idempotent(self, db_session: Any) -> None:
        entry = await _entry(db_session)
        await _snap(db_session, entry, age=2)
        await _snap(db_session, entry, age=1)
        await db_session.commit()

        assert (await collapse_superseded_pending(db_session)).snapshots == 1
        assert (await collapse_superseded_pending(db_session)).snapshots == 0


class TestStoreSeams:
    @pytest.mark.asyncio
    async def test_candidate_read_returns_every_response_generation_oldest_first(
        self, db_session: Any
    ) -> None:
        entry = await _entry(db_session)
        done = await _snap(db_session, entry, age=3, status=C)
        pending = await _snap(db_session, entry, age=1)
        await _snap(db_session, entry, age=2, status=C, revisit_of=done)
        settled = await _entry(db_session)  # nothing owed: not a candidate
        await _snap(db_session, settled, age=1, status=C)
        await db_session.commit()

        grouped = await list_generations_with_unextracted_snapshots(db_session)

        assert list(grouped) == [entry.entry_id]
        assert [g.snapshot_id for g in grouped[entry.entry_id]] == [
            done.snapshot_id,
            pending.snapshot_id,
        ]

    @pytest.mark.asyncio
    async def test_the_mark_is_conditional_on_work_still_being_owed(self, db_session: Any) -> None:
        """A snapshot another runner claimed between the read and the write is left alone."""
        entry = await _entry(db_session)
        claimed = await _snap(db_session, entry, age=2, status=IP)
        newest = await _snap(db_session, entry, age=1)
        await db_session.commit()

        marked = await mark_snapshot_superseded(
            db_session, claimed.snapshot_id, superseded_by=newest.snapshot_id
        )

        assert marked is False
        assert await _state(db_session, claimed) == (IP.value, None)

    @pytest.mark.asyncio
    async def test_a_claim_un_collapses(self, db_session: Any) -> None:
        """Only an explicit extraction claims a collapsed snapshot, and it then owns it."""
        entry = await _entry(db_session)
        old = await _snap(db_session, entry, age=2)
        await _snap(db_session, entry, age=1)
        await db_session.commit()
        await collapse_superseded_pending(db_session)
        assert await get_snapshot_superseded_by(db_session, old.snapshot_id) is not None

        await claim_snapshot_for_extraction(
            db_session, old.snapshot_id, started_at=datetime.now(UTC)
        )
        await db_session.commit()

        assert await _state(db_session, old) == (IP.value, None)

    @pytest.mark.asyncio
    async def test_latest_completed_skips_a_collapsed_generation(self, db_session: Any) -> None:
        entry = await _entry(db_session)
        extracted = await _snap(db_session, entry, age=3, status=C)
        await _snap(db_session, entry, age=2)
        await _snap(db_session, entry, age=1)
        await db_session.commit()
        await collapse_superseded_pending(db_session)  # age=2 is now COMPLETE + marked

        assert (
            await get_latest_completed_snapshot_id(db_session, entry.entry_id)
            == extracted.snapshot_id
        )

    @pytest.mark.asyncio
    async def test_lint_sets(self, db_session: Any) -> None:
        entry = await _entry(db_session)
        replaced = await _snap(db_session, entry, age=4, status=C)
        collapsed = await _snap(db_session, entry, age=3)
        current = await _snap(db_session, entry, age=2, status=C)
        await db_session.commit()
        await collapse_superseded_pending(db_session)

        assert await list_collapsed_snapshot_ids(db_session) == {collapsed.snapshot_id}
        # Both older generations have an extracted newer sibling (the sets may
        # overlap: a collapsed snapshot is COMPLETE too); `current` has none.
        assert await list_replaced_mutable_snapshot_ids(db_session) == {
            replaced.snapshot_id,
            collapsed.snapshot_id,
        }
        assert current.snapshot_id not in await list_replaced_mutable_snapshot_ids(db_session)


class _ExplodingExtractor:
    """Fails the test if the pipeline reaches the extractor at all."""

    async def extract(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a collapsed snapshot must not reach the extractor")


class TestPipelineGuard:
    @pytest.mark.asyncio
    async def test_bulk_callers_skip_a_snapshot_collapsed_after_they_listed_it(
        self, db_session: Any
    ) -> None:
        entry = await _entry(db_session)
        old = await _snap(db_session, entry, age=2)
        await _snap(db_session, entry, age=1)
        await db_session.commit()
        await collapse_superseded_pending(db_session)

        written = await extract_snapshot(
            db_session,
            entry.entry_id,
            old.snapshot_id,
            extractor=_ExplodingExtractor(),  # type: ignore[arg-type]
            skip_if_superseded=True,
        )

        assert written == []
        # No claim was taken: still collapsed, still COMPLETE.
        status, superseded_by = await _state(db_session, old)
        assert status == C.value and superseded_by is not None
