"""Blob-reachability probe — the detection half of the story."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import reset_config
from particles.corpus.blob_health import check_blob_reachability, store_file_missing
from particles.corpus.store import SnapshotRow


def _hash(n: int) -> str:
    """A syntactically valid SHA-256 hex digest — `blob_path` rejects anything else."""
    return f"{n:064x}"


_next_snapshot_id = 0


async def _add_snapshot(
    session: AsyncSession,
    content_hash: str,
    *,
    record_type: str = "RESPONSE",
    archive_path: str | None = "/somewhere/blob",
    captured_at: datetime | None = None,
) -> None:
    global _next_snapshot_id
    _next_snapshot_id += 1
    session.add(
        SnapshotRow(
            snapshot_id=f"snap-{_next_snapshot_id}",
            entry_id="entry-1",
            captured_at=captured_at or datetime(2026, 7, 18, tzinfo=UTC),
            content_hash=content_hash,
            warc_record_type=record_type,
            archive_path=archive_path,
            extraction_status="PENDING",
        )
    )
    await session.flush()


@pytest.fixture
def blob_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point `storage.blob_dir` at an absolute tmp dir for the duration of a test."""
    target = tmp_path / "corpus_blobs"
    target.mkdir()
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(target))
    reset_config()
    return target


def _write_blob(blob_dir: Path, content_hash: str) -> None:
    shard = blob_dir / content_hash[:2]
    shard.mkdir(parents=True, exist_ok=True)
    (shard / content_hash).write_bytes(b"content")


@pytest.mark.asyncio
async def test_empty_store_is_healthy(blob_dir: Path, db_session: AsyncSession) -> None:
    """A first-run store with no snapshots must not be warned about."""
    report = await check_blob_reachability(db_session)

    assert report.snapshots == 0
    assert report.sampled == 0
    assert report.healthy
    assert report.warning_lines() == []


@pytest.mark.asyncio
async def test_present_blobs_report_healthy(blob_dir: Path, db_session: AsyncSession) -> None:
    for n in range(3):
        await _add_snapshot(db_session, _hash(n))
        _write_blob(blob_dir, _hash(n))

    report = await check_blob_reachability(db_session)

    assert (report.snapshots, report.sampled, report.missing) == (3, 3, 0)
    assert report.healthy
    assert report.warning_lines() == []


@pytest.mark.asyncio
async def test_scattered_store_reports_total_loss(blob_dir: Path, db_session: AsyncSession) -> None:
    """Rows present, blob dir empty — the 2026-07-18 sharding signature."""
    for n in range(3):
        await _add_snapshot(db_session, _hash(n))

    report = await check_blob_reachability(db_session)

    assert (report.snapshots, report.sampled, report.missing) == (3, 3, 3)
    assert not report.healthy
    assert report.total_loss
    lines = "\n".join(report.warning_lines())
    assert "none of the 3 sampled" in lines
    assert str(blob_dir) in lines
    assert "likely still on disk" in lines  # the remediation line


@pytest.mark.asyncio
async def test_partial_miss_is_not_total_loss(blob_dir: Path, db_session: AsyncSession) -> None:
    for n in range(3):
        await _add_snapshot(db_session, _hash(n))
    _write_blob(blob_dir, _hash(0))

    report = await check_blob_reachability(db_session)

    assert (report.sampled, report.missing) == (3, 2)
    assert not report.healthy
    assert not report.total_loss
    assert "2 of the 3 sampled" in "\n".join(report.warning_lines())


@pytest.mark.asyncio
async def test_revisit_and_unarchived_snapshots_are_not_misses(
    blob_dir: Path, db_session: AsyncSession
) -> None:
    """REVISIT rows inherit content via `refers_to`; EPHEMERAL rows are never archived.

    Counting either as a miss would warn every store that has ever re-fetched a
    MUTABLE source.
    """
    await _add_snapshot(db_session, _hash(0))
    _write_blob(blob_dir, _hash(0))
    await _add_snapshot(db_session, _hash(1), record_type="REVISIT", archive_path=None)
    await _add_snapshot(db_session, _hash(2), archive_path=None)

    report = await check_blob_reachability(db_session)

    assert (report.snapshots, report.sampled, report.missing) == (1, 1, 0)
    assert report.healthy


@pytest.mark.asyncio
async def test_missing_blob_dir_is_reported(blob_dir: Path, db_session: AsyncSession) -> None:
    blob_dir.rmdir()
    await _add_snapshot(db_session, _hash(0))

    report = await check_blob_reachability(db_session)

    assert not report.dir_exists
    assert report.total_loss
    assert "directory does not exist" in "\n".join(report.warning_lines())


@pytest.mark.asyncio
async def test_sample_is_bounded_and_newest_first(blob_dir: Path, db_session: AsyncSession) -> None:
    """The probe stats at most `sample` hashes, preferring the most recent snapshots."""
    for n in range(5):
        await _add_snapshot(db_session, _hash(n), captured_at=datetime(2026, 7, 10 + n, tzinfo=UTC))
    # Only the two newest have blobs on disk.
    _write_blob(blob_dir, _hash(4))
    _write_blob(blob_dir, _hash(3))

    report = await check_blob_reachability(db_session, sample=2)

    assert report.snapshots == 5
    assert (report.sampled, report.missing) == (2, 0)


class TestStoreFileMissing:
    """Probing must never *create* the database it is probing."""

    def test_absent_sqlite_file_is_missing(self, tmp_path: Path) -> None:
        absent = tmp_path / "never-created.db"
        assert store_file_missing(f"sqlite+aiosqlite:///{absent}")
        # The check itself must not bring the file into existence.
        assert not absent.exists()

    def test_present_sqlite_file_is_not_missing(self, tmp_path: Path) -> None:
        present = tmp_path / "store.db"
        present.write_bytes(b"")
        assert not store_file_missing(f"sqlite+aiosqlite:///{present}")

    @pytest.mark.parametrize(
        "dsn",
        [
            "sqlite+aiosqlite:///:memory:",
            "postgresql+asyncpg://user@host/db",
        ],
    )
    def test_non_file_dsns_are_not_missing(self, dsn: str) -> None:
        """No file to miss — connecting is the only way to learn anything."""
        assert not store_file_missing(dsn)


@pytest.mark.asyncio
async def test_duplicate_hashes_sample_once(blob_dir: Path, db_session: AsyncSession) -> None:
    """Several snapshots can share one blob; the probe must not stat it repeatedly."""
    await _add_snapshot(db_session, _hash(0))
    await _add_snapshot(db_session, _hash(0), captured_at=datetime(2026, 7, 20, tzinfo=UTC))

    report = await check_blob_reachability(db_session)

    assert report.snapshots == 2
    assert report.sampled == 1
