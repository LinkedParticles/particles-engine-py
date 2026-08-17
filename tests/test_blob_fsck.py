"""Exhaustive blob audit + re-home — the repair half."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import reset_config
from particles.corpus.blob_fsck import audit_blobs, rehome_strays
from particles.corpus.store import CorpusEntryRow, SnapshotRow

_next_snapshot_id = 0


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_blob(root: Path, content_hash: str, payload: bytes, *, flat: bool = False) -> Path:
    """Write ``payload`` at the content-addressed location for ``content_hash``.

    ``content_hash`` is passed separately from the payload on purpose: the
    digest-rejection tests need a file sitting under a name its bytes do not
    hash to.
    """
    target = root / content_hash if flat else root / content_hash[:2] / content_hash
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


async def _add_snapshot(
    session: AsyncSession,
    content_hash: str,
    *,
    entry_id: str = "entry-1",
    uri_r: str = "file:///sources/one.md",
    record_type: str = "RESPONSE",
    archive_path: str | None = "/somewhere/blob",
) -> None:
    global _next_snapshot_id
    _next_snapshot_id += 1
    existing = await session.get(CorpusEntryRow, entry_id)
    if existing is None:
        session.add(
            CorpusEntryRow(
                entry_id=entry_id,
                uri_r=uri_r,
                source_type="LOCAL_MARKDOWN",
                mutability="STABLE",
                fetch_policy="NEVER",
                created_at=datetime(2026, 7, 18, tzinfo=UTC),
                deposited_by="test",
            )
        )
    session.add(
        SnapshotRow(
            snapshot_id=f"fsck-snap-{_next_snapshot_id}",
            entry_id=entry_id,
            captured_at=datetime(2026, 7, 18, tzinfo=UTC),
            content_hash=content_hash,
            warc_record_type=record_type,
            archive_path=archive_path,
            extraction_status="PENDING",
        )
    )
    await session.flush()


@pytest.fixture
def blob_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The resolved blob dir — where extraction will look for content."""
    target = tmp_path / "corpus_blobs"
    target.mkdir()
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(target))
    reset_config()
    return target


@pytest.fixture
def stray_dir(tmp_path: Path) -> Path:
    """A second blob tree, standing in for a sibling worktree's ``corpus_blobs/``."""
    target = tmp_path / "worktree" / "corpus_blobs"
    target.mkdir(parents=True)
    return target


class TestAudit:
    """The read-only default: three disjoint counts over every referenced blob."""

    @pytest.mark.asyncio
    async def test_empty_store_is_healthy(self, blob_dir: Path, db_session: AsyncSession) -> None:
        report = await audit_blobs(db_session)

        assert report.total == 0
        assert report.healthy

    @pytest.mark.asyncio
    async def test_present_blobs(self, blob_dir: Path, db_session: AsyncSession) -> None:
        for n in range(3):
            payload = f"blob {n}".encode()
            await _add_snapshot(db_session, _digest(payload))
            _write_blob(blob_dir, _digest(payload), payload)

        report = await audit_blobs(db_session)

        assert (len(report.present), len(report.elsewhere), len(report.missing)) == (3, 0, 0)
        assert report.healthy

    @pytest.mark.asyncio
    async def test_audit_is_exhaustive_not_sampled(
        self, blob_dir: Path, db_session: AsyncSession
    ) -> None:
        """The sampled probe caps at ``blob_health_sample``; fsck must not."""
        for n in range(25):
            await _add_snapshot(db_session, _digest(f"blob {n}".encode()))

        report = await audit_blobs(db_session)

        assert report.total == 25
        assert len(report.missing) == 25

    @pytest.mark.asyncio
    async def test_missing_without_search_dirs(
        self, blob_dir: Path, stray_dir: Path, db_session: AsyncSession
    ) -> None:
        """Nothing is inferred: a stray the operator did not point at stays missing."""
        payload = b"scattered"
        await _add_snapshot(db_session, _digest(payload))
        _write_blob(stray_dir, _digest(payload), payload)

        report = await audit_blobs(db_session)

        assert len(report.missing) == 1
        assert report.elsewhere == ()

    @pytest.mark.asyncio
    async def test_stray_found_under_search_dir(
        self, blob_dir: Path, stray_dir: Path, db_session: AsyncSession
    ) -> None:
        payload = b"scattered"
        await _add_snapshot(db_session, _digest(payload))
        source = _write_blob(stray_dir, _digest(payload), payload)

        report = await audit_blobs(db_session, search_dirs=[stray_dir])

        assert (len(report.present), len(report.elsewhere), len(report.missing)) == (0, 1, 0)
        assert report.elsewhere[0].source == source
        assert not report.healthy

    @pytest.mark.asyncio
    async def test_flat_layout_is_also_a_candidate(
        self, blob_dir: Path, stray_dir: Path, db_session: AsyncSession
    ) -> None:
        """A rescue directory someone flattened by hand still resolves."""
        payload = b"flattened"
        await _add_snapshot(db_session, _digest(payload))
        _write_blob(stray_dir, _digest(payload), payload, flat=True)

        report = await audit_blobs(db_session, search_dirs=[stray_dir])

        assert len(report.elsewhere) == 1

    @pytest.mark.asyncio
    async def test_blob_dir_as_search_dir_is_ignored(
        self, blob_dir: Path, db_session: AsyncSession
    ) -> None:
        """``--search $(blob_dir)`` must be harmless, not report self-strays."""
        payload = b"present"
        await _add_snapshot(db_session, _digest(payload))
        _write_blob(blob_dir, _digest(payload), payload)

        report = await audit_blobs(db_session, search_dirs=[blob_dir])

        assert len(report.present) == 1
        assert report.search_dirs == ()

    @pytest.mark.asyncio
    async def test_revisit_and_unarchived_snapshots_are_not_misses(
        self, blob_dir: Path, db_session: AsyncSession
    ) -> None:
        """REVISIT rows inherit content; EPHEMERAL entries are never archived."""
        await _add_snapshot(db_session, _digest(b"a"), record_type="REVISIT", archive_path=None)
        await _add_snapshot(db_session, _digest(b"b"), archive_path=None)

        report = await audit_blobs(db_session)

        assert report.total == 0
        assert report.healthy

    @pytest.mark.asyncio
    async def test_missing_ref_carries_entry_ids_and_uris(
        self, blob_dir: Path, db_session: AsyncSession
    ) -> None:
        """The operator has to know *which source* to re-deposit or retract."""
        payload = b"gone"
        await _add_snapshot(
            db_session, _digest(payload), entry_id="entry-9", uri_r="file:///sources/gone.md"
        )

        report = await audit_blobs(db_session)

        (ref,) = report.missing
        assert ref.entry_ids == ("entry-9",)
        assert ref.uris == ("file:///sources/gone.md",)
        assert ref.label == "/sources/gone.md"

    @pytest.mark.asyncio
    async def test_shared_blob_is_audited_once_across_entries(
        self, blob_dir: Path, db_session: AsyncSession
    ) -> None:
        payload = b"shared"
        await _add_snapshot(db_session, _digest(payload), entry_id="entry-a")
        await _add_snapshot(db_session, _digest(payload), entry_id="entry-b")

        report = await audit_blobs(db_session)

        assert report.total == 1
        assert report.missing[0].entry_ids == ("entry-a", "entry-b")


class TestDigestVerification:
    """Content addressing is what makes the repair checkable rather than hopeful."""

    @pytest.mark.asyncio
    async def test_mismatched_candidate_is_rejected_and_stays_missing(
        self, blob_dir: Path, stray_dir: Path, db_session: AsyncSession
    ) -> None:
        wanted = _digest(b"the real content")
        await _add_snapshot(db_session, wanted)
        # Right filename, wrong bytes — the silent false positive the
        # content-addressed layout would otherwise hand us.
        _write_blob(stray_dir, wanted, b"something else entirely")

        report = await audit_blobs(db_session, search_dirs=[stray_dir])

        assert report.elsewhere == ()
        assert len(report.missing) == 1
        assert len(report.rejected) == 1
        assert report.rejected[0].actual_digest == _digest(b"something else entirely")

    @pytest.mark.asyncio
    async def test_rejected_candidate_is_never_copied(
        self, blob_dir: Path, stray_dir: Path, db_session: AsyncSession
    ) -> None:
        wanted = _digest(b"the real content")
        await _add_snapshot(db_session, wanted)
        _write_blob(stray_dir, wanted, b"something else entirely")

        report = await audit_blobs(db_session, search_dirs=[stray_dir])
        outcome = rehome_strays(report)

        assert outcome.copied == ()
        assert not (blob_dir / wanted[:2] / wanted).exists()


class TestReHome:
    """Copy, never move; verify; never touch the database."""

    @pytest.mark.asyncio
    async def test_copies_stray_into_the_blob_dir(
        self, blob_dir: Path, stray_dir: Path, db_session: AsyncSession
    ) -> None:
        payload = b"scattered"
        content_hash = _digest(payload)
        await _add_snapshot(db_session, content_hash)
        source = _write_blob(stray_dir, content_hash, payload)

        report = await audit_blobs(db_session, search_dirs=[stray_dir])
        outcome = rehome_strays(report)

        assert len(outcome.copied) == 1
        assert outcome.failed == ()
        home = blob_dir / content_hash[:2] / content_hash
        assert home.read_bytes() == payload
        # Copy, never move: a wrong --search costs disk, not data.
        assert source.read_bytes() == payload

    @pytest.mark.asyncio
    async def test_is_idempotent(
        self, blob_dir: Path, stray_dir: Path, db_session: AsyncSession
    ) -> None:
        payload = b"scattered"
        await _add_snapshot(db_session, _digest(payload))
        _write_blob(stray_dir, _digest(payload), payload)

        rehome_strays(await audit_blobs(db_session, search_dirs=[stray_dir]))
        second = await audit_blobs(db_session, search_dirs=[stray_dir])

        assert len(second.present) == 1
        assert second.healthy
        assert rehome_strays(second).copied == ()

    @pytest.mark.asyncio
    async def test_unreadable_stray_is_collected_not_raised(
        self, blob_dir: Path, stray_dir: Path, db_session: AsyncSession
    ) -> None:
        """One bad stray must not abandon the rest of the repair."""
        good = b"recoverable"
        bad = b"vanishing"
        await _add_snapshot(db_session, _digest(good))
        await _add_snapshot(db_session, _digest(bad))
        _write_blob(stray_dir, _digest(good), good)
        doomed = _write_blob(stray_dir, _digest(bad), bad)

        report = await audit_blobs(db_session, search_dirs=[stray_dir])
        doomed.unlink()  # disappears between audit and copy
        outcome = rehome_strays(report)

        assert len(outcome.copied) == 1
        assert len(outcome.failed) == 1
        assert outcome.failed[0][0].ref.content_hash == _digest(bad)

    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing(
        self, blob_dir: Path, stray_dir: Path, db_session: AsyncSession
    ) -> None:
        payload = b"scattered"
        content_hash = _digest(payload)
        await _add_snapshot(db_session, content_hash)
        _write_blob(stray_dir, content_hash, payload)

        report = await audit_blobs(db_session, search_dirs=[stray_dir])
        outcome = rehome_strays(report, dry_run=True)

        assert outcome.dry_run
        assert len(outcome.copied) == 1
        assert not (blob_dir / content_hash[:2] / content_hash).exists()

    @pytest.mark.asyncio
    async def test_database_is_never_written(
        self, blob_dir: Path, stray_dir: Path, db_session: AsyncSession
    ) -> None:
        """fsck moves bytes; retraction and re-deposit stay operator verbs."""
        present = b"here"
        gone = b"unrecoverable"
        await _add_snapshot(db_session, _digest(present))
        await _add_snapshot(db_session, _digest(gone), entry_id="entry-2")
        _write_blob(blob_dir, _digest(present), present)
        before = sorted(
            (row.snapshot_id, row.archive_path, row.extraction_status)
            for row in (await db_session.execute(select(SnapshotRow))).scalars()
        )

        report = await audit_blobs(db_session, search_dirs=[stray_dir])
        rehome_strays(report)

        assert len(report.missing) == 1
        db_session.expire_all()
        after = sorted(
            (row.snapshot_id, row.archive_path, row.extraction_status)
            for row in (await db_session.execute(select(SnapshotRow))).scalars()
        )
        assert after == before
