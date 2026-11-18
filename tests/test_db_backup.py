# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles db backup``: the consistent online copy of a store and its blobs."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import shutil
import sqlite3
import stat
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from filelock import FileLock
from typer.testing import CliRunner

from particles.api.cli import app
from particles.db import session_scope
from particles.operations.store_backup import BackupRefused, backup_store, plan_backup

runner = CliRunner()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_blob(root: Path, payload: bytes, *, name: str | None = None) -> str:
    content_hash = name or _digest(payload)
    target = root / content_hash[:2] / content_hash
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return content_hash


def _seed(content_hash: str, *, entry_id: str = "entry-1") -> None:
    """One blob-bearing snapshot on a corpus entry, named by ``content_hash``."""

    async def _impl() -> None:
        from particles.corpus.store import CorpusEntryRow, SnapshotRow

        async with session_scope() as session:
            if await session.get(CorpusEntryRow, entry_id) is None:
                session.add(
                    CorpusEntryRow(
                        entry_id=entry_id,
                        uri_r=f"file:///sources/{entry_id}.md",
                        source_type="LOCAL_MARKDOWN",
                        mutability="STABLE",
                        fetch_policy="NEVER",
                        created_at=datetime(2026, 7, 18, tzinfo=UTC),
                        deposited_by="test",
                    )
                )
            session.add(
                SnapshotRow(
                    snapshot_id=f"snap-{content_hash[:8]}",
                    entry_id=entry_id,
                    captured_at=datetime(2026, 7, 18, tzinfo=UTC),
                    content_hash=content_hash,
                    warc_record_type="RESPONSE",
                    # An old host's absolute path: inert, never read for a location.
                    archive_path="/old/host/corpus_blobs/blob",
                    extraction_status="PENDING",
                )
            )
            await session.commit()

    asyncio.run(_impl())


def _count(db: Path, table: str) -> int:
    with contextlib.closing(sqlite3.connect(db)) as conn:
        return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])  # noqa: S608


@pytest.fixture
def blob_dir(cli_db: Path, tmp_path: Path) -> Path:
    """The live blob dir ``cli_db`` points ``PARTICLES_BLOB_DIR`` at."""
    target = tmp_path / "blobs"
    target.mkdir(exist_ok=True)
    return target


@pytest.fixture
def target(tmp_path: Path) -> Path:
    return tmp_path / "backup" / "2026-11-07"


# ---------------------------------------------------------------------------
# The copy
# ---------------------------------------------------------------------------


def test_backup_copies_the_store_and_its_blobs(cli_db: Path, blob_dir: Path, target: Path) -> None:
    referenced = _write_blob(blob_dir, b"a deposited note")
    _seed(referenced)
    orphan = _write_blob(blob_dir, b"a blob no row names")
    (blob_dir / ".DS_Store").write_bytes(b"junk")
    (blob_dir / referenced[:2] / "not-a-hash").write_bytes(b"junk")

    result = asyncio.run(backup_store(target))

    assert result.verified
    assert result.integrity == "ok"
    assert result.blobs.copied == 2
    assert result.blobs.failed == ()
    assert result.fsck.total == 1
    db = target / "particles.db"
    assert _count(db, "snapshots") == 1
    assert _count(db, "corpus_entries") == 1
    for content_hash in (referenced, orphan):
        copied = target / "corpus_blobs" / content_hash[:2] / content_hash
        assert _digest(copied.read_bytes()) == content_hash
    # Only the content-addressed layout is copied.
    assert not (target / "corpus_blobs" / ".DS_Store").exists()
    assert not (target / "corpus_blobs" / referenced[:2] / "not-a-hash").exists()
    # The copy is the whole store: owner-only.
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    # The archive paths are copied as they were.
    with contextlib.closing(sqlite3.connect(db)) as conn:
        (archive_path,) = conn.execute("SELECT archive_path FROM snapshots").fetchone()
    assert archive_path == "/old/host/corpus_blobs/blob"


def test_backup_leaves_the_live_store_as_it_was(cli_db: Path, blob_dir: Path, target: Path) -> None:
    content_hash = _write_blob(blob_dir, b"live")
    _seed(content_hash)
    before = sorted(p.relative_to(blob_dir) for p in blob_dir.rglob("*"))

    asyncio.run(backup_store(target))

    assert _count(cli_db, "snapshots") == 1
    assert sorted(p.relative_to(blob_dir) for p in blob_dir.rglob("*")) == before


def test_an_existing_empty_directory_is_accepted(
    cli_db: Path, blob_dir: Path, target: Path
) -> None:
    target.mkdir(parents=True)

    result = asyncio.run(backup_store(target))

    assert result.verified
    assert (target / "particles.db").is_file()


def test_the_copy_opens_as_a_store_with_only_its_database_url(
    cli_db: Path, blob_dir: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the default relative ``blob_dir``, the copy's blobs resolve beside its database."""
    content_hash = _write_blob(blob_dir, b"moves hosts")
    _seed(content_hash)
    asyncio.run(backup_store(target))

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{target / 'particles.db'}")
    monkeypatch.delenv("PARTICLES_BLOB_DIR")
    from particles.config import reset_config

    reset_config()
    result = runner.invoke(app, ["corpus", "fsck"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "present:         1" in result.output
    assert str(target / "corpus_blobs") in result.output


# ---------------------------------------------------------------------------
# Consistency without stopping a writer
# ---------------------------------------------------------------------------


def test_backup_does_not_wait_on_a_held_writer_lock(
    cli_db: Path, blob_dir: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backup is a reader: another process's writer lock does not stall it."""
    monkeypatch.setenv("PARTICLES_WRITE_LOCK_TIMEOUT_SECONDS", "0.1")
    holder = FileLock(f"{cli_db}.writelock", thread_local=False)
    holder.acquire()
    try:
        result = asyncio.run(backup_store(target))
    finally:
        holder.release()

    assert result.verified


def test_an_uncommitted_write_is_not_in_the_copy(
    cli_db: Path, blob_dir: Path, target: Path
) -> None:
    """The copy is one read transaction: a writer mid-transaction neither blocks it nor leaks in."""
    with contextlib.closing(sqlite3.connect(cli_db, isolation_level=None)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("CREATE TABLE half_written (x INTEGER)")
        result = asyncio.run(backup_store(target))
        writer.execute("ROLLBACK")

    assert result.verified
    with contextlib.closing(sqlite3.connect(target / "particles.db")) as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert "half_written" not in names


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_a_blob_missing_from_the_live_store_too_is_reported_as_lost_before(
    cli_db: Path, blob_dir: Path, target: Path
) -> None:
    """A loss the live store already has does not fail a copy that holds all it has.

    The owner's store had 12 such blobs at the 2026-11-07 cutover rehearsal, so
    every backup of it failed and none could be moved.
    """
    kept = _write_blob(blob_dir, b"still here")
    _seed(kept)
    gone = _digest(b"never archived")
    _seed(gone, entry_id="entry-9")

    result = asyncio.run(backup_store(target))

    assert result.verified
    assert [ref.content_hash for ref in result.fsck.missing] == [gone]
    assert [ref.content_hash for ref in result.lost_before] == [gone]
    assert result.not_copied == ()


def test_a_blob_the_live_store_has_and_the_copy_lacks_fails_the_backup(
    cli_db: Path, blob_dir: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content_hash = _write_blob(blob_dir, b"unreadable now")
    _seed(content_hash)

    def _refuse(root: Path, name: str) -> bytes:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("particles.operations.store_backup.read_blob", _refuse)
    result = asyncio.run(backup_store(target))

    assert not result.verified
    assert result.lost_before == ()
    assert [ref.content_hash for ref in result.not_copied] == [content_hash]


def test_a_missing_live_blob_dir_fails_the_backup(
    cli_db: Path, tmp_path: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong blob path copies nothing; that is a failure, not a store that lost everything."""
    _seed(_digest(b"one"))
    _seed(_digest(b"two"), entry_id="entry-2")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "no-such-dir"))
    from particles.config import reset_config

    reset_config()
    result = asyncio.run(backup_store(target))

    assert not result.verified
    assert result.lost_before == ()
    assert len(result.not_copied) == 2


def test_a_live_blob_dir_holding_none_of_the_stores_blobs_fails_the_backup(
    cli_db: Path, blob_dir: Path, target: Path
) -> None:
    _seed(_digest(b"elsewhere"))

    result = asyncio.run(backup_store(target))

    assert not result.verified
    assert result.lost_before == ()


def test_a_blob_whose_bytes_miss_their_name_is_not_copied(
    cli_db: Path, blob_dir: Path, target: Path
) -> None:
    name = _digest(b"what it should hold")
    _write_blob(blob_dir, b"something else", name=name)
    _seed(name)

    result = asyncio.run(backup_store(target))

    assert [failed for failed, _ in result.blobs.failed] == [name]
    assert not (target / "corpus_blobs" / name[:2] / name).exists()
    assert not result.verified


def test_a_symlinked_blob_is_not_followed(
    cli_db: Path, blob_dir: Path, target: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "host-file"
    outside.write_bytes(b"a host file")
    name = _digest(b"a host file")
    (blob_dir / name[:2]).mkdir(parents=True)
    (blob_dir / name[:2] / name).symlink_to(outside)

    result = asyncio.run(backup_store(target))

    assert [failed for failed, _ in result.blobs.failed] == [name]
    assert not (target / "corpus_blobs" / name[:2] / name).exists()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_the_live_store_directory_is_refused(cli_db: Path, blob_dir: Path) -> None:
    """Writing the copy into the store's own directory would overwrite ``particles.db``."""
    with pytest.raises(BackupRefused, match="would overwrite the live store"):
        plan_backup(cli_db.parent)


def test_a_target_inside_the_live_blob_dir_is_refused(cli_db: Path, blob_dir: Path) -> None:
    with pytest.raises(BackupRefused, match="live blob directory"):
        plan_backup(blob_dir / "nightly")


def test_the_live_blob_dir_as_the_copy_blob_dir_is_refused(
    cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_blobs = tmp_path / "copy" / "corpus_blobs"
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(live_blobs))
    from particles.config import reset_config

    reset_config()
    with pytest.raises(BackupRefused, match="live blob directory"):
        plan_backup(tmp_path / "copy")


def test_a_non_empty_target_is_refused(cli_db: Path, blob_dir: Path, target: Path) -> None:
    target.mkdir(parents=True)
    (target / "particles.db").write_bytes(b"yesterday's backup")

    with pytest.raises(BackupRefused, match="is not empty"):
        plan_backup(target)
    assert (target / "particles.db").read_bytes() == b"yesterday's backup"


def test_a_file_target_is_refused(cli_db: Path, blob_dir: Path, tmp_path: Path) -> None:
    file_target = tmp_path / "a-file"
    file_target.write_text("x")

    with pytest.raises(BackupRefused, match="not a directory"):
        plan_backup(file_target)


@pytest.fixture
def postgres_store(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:secret@db.example/particles")
    from particles.config import reset_config

    reset_config()
    yield
    reset_config()


def test_a_postgres_store_is_refused(postgres_store: None, target: Path) -> None:
    with pytest.raises(BackupRefused, match="SQLite store only") as excinfo:
        plan_backup(target)
    assert "secret" not in str(excinfo.value)
    assert not target.exists()


def test_an_in_memory_store_is_refused(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    from particles.config import reset_config

    reset_config()
    with pytest.raises(BackupRefused, match="in memory"):
        plan_backup(target)


def test_an_absent_store_is_refused_and_not_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    absent = tmp_path / "nowhere" / "particles.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{absent}")
    from particles.config import reset_config

    reset_config()
    with pytest.raises(BackupRefused, match="nothing to back up"):
        plan_backup(target)
    assert not absent.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_backup_reports_a_verified_copy(cli_db: Path, blob_dir: Path, target: Path) -> None:
    _seed(_write_blob(blob_dir, b"cli"))

    result = runner.invoke(app, ["db", "backup", str(target)], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "quick_check: ok" in result.output
    assert "Blobs copied: 1" in result.output
    assert "Backup verified." in result.output


def test_cli_backup_exits_non_zero_when_the_copy_does_not_verify(
    cli_db: Path, blob_dir: Path, target: Path
) -> None:
    name = _digest(b"what it should hold")
    _write_blob(blob_dir, b"something else", name=name)
    _seed(name, entry_id="entry-9")

    result = runner.invoke(app, ["db", "backup", str(target)], catch_exceptions=False)

    assert result.exit_code == 1
    assert "/sources/entry-9.md" in result.output
    assert "did not verify" in result.output


def test_cli_backup_names_a_failed_quick_check(
    cli_db: Path, blob_dir: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(_write_blob(blob_dir, b"fine"))
    monkeypatch.setattr(
        "particles.operations.store_backup.quick_check", lambda _db: "page 3 is never used"
    )

    result = runner.invoke(app, ["db", "backup", str(target)], catch_exceptions=False)

    assert result.exit_code == 1
    assert "its database failed quick_check" in result.output
    assert "has a file for each blob listed" not in result.output


def test_cli_backup_verifies_and_names_blobs_lost_before_it(
    cli_db: Path, blob_dir: Path, target: Path
) -> None:
    _seed(_write_blob(blob_dir, b"kept"))
    _seed(_digest(b"gone"), entry_id="entry-9")

    result = runner.invoke(app, ["db", "backup", str(target)], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "1 of them are missing from the live store too" in result.output
    assert "/sources/entry-9.md" in result.output
    assert "Backup verified." in result.output
    assert "lost before this backup" in result.output


def test_cli_backup_refusal_is_one_line_and_exit_one(cli_db: Path, blob_dir: Path) -> None:
    result = runner.invoke(app, ["db", "backup", str(cli_db.parent)], catch_exceptions=False)

    assert result.exit_code == 1
    assert "would overwrite the live store" in result.output


def test_cli_backup_needs_a_directory(cli_db: Path) -> None:
    result = runner.invoke(app, ["db", "backup"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "needs the directory" in result.output


def test_cli_init_takes_no_directory(cli_db: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["db", "init", str(tmp_path / "x")], catch_exceptions=False)

    assert result.exit_code == 1
    assert "takes no directory argument" in result.output


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_the_copy_is_owner_only_in_an_existing_directory(
    cli_db: Path, blob_dir: Path, target: Path
) -> None:
    target.mkdir(parents=True, mode=0o755)

    asyncio.run(backup_store(target))

    assert stat.S_IMODE((target / "particles.db").stat().st_mode) == 0o600


def test_a_target_path_with_url_characters_is_audited_in_place(
    cli_db: Path, blob_dir: Path, tmp_path: Path
) -> None:
    content_hash = _write_blob(blob_dir, b"odd path")
    _seed(content_hash)
    odd = tmp_path / "what?#now"

    result = asyncio.run(backup_store(odd))

    assert result.verified
    assert result.fsck.total == 1


def test_a_symlinked_live_blob_root_is_not_followed(
    cli_db: Path, tmp_path: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy reads the live blob dir as every reader names it, symlink checks included."""
    real = tmp_path / "elsewhere"
    content_hash = _write_blob(real, b"behind a link")
    _seed(content_hash)
    link = tmp_path / "linked-blobs"
    link.symlink_to(real)
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(link))
    from particles.config import reset_config

    reset_config()
    result = asyncio.run(backup_store(target))

    assert [failed for failed, _ in result.blobs.failed] == [content_hash]
    assert not result.verified


def test_a_store_that_vanishes_before_the_copy_is_not_recreated(tmp_path: Path) -> None:
    from particles.operations.store_backup import copy_database

    gone = tmp_path / "gone.db"
    with pytest.raises(sqlite3.Error):
        copy_database(gone, tmp_path / "copy.db")
    assert not gone.exists()
    assert not (tmp_path / "copy.db").exists()


def test_the_copy_never_checkpoints_the_live_wal(tmp_path: Path) -> None:
    """The last connection to close does not fold the live WAL into the store file."""
    from particles.operations.store_backup import copy_database

    live = tmp_path / "live.db"
    with contextlib.closing(sqlite3.connect(live, isolation_level=None)) as conn:
        conn.execute("PRAGMA journal_mode=wal")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE t (x)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(500)])
        # Keep the WAL frames: snapshot the three files while this connection
        # is open, since closing it would checkpoint them itself.
        saved = tmp_path / "saved"
        saved.mkdir()
        for suffix in ("", "-wal", "-shm"):
            shutil.copy(f"{live}{suffix}", saved / f"live.db{suffix}")
    for suffix in ("", "-wal", "-shm"):
        shutil.copy(saved / f"live.db{suffix}", f"{live}{suffix}")
    db_before = live.read_bytes()
    wal_before = Path(f"{live}-wal").read_bytes()

    copy_database(live, tmp_path / "copy.db")

    assert live.read_bytes() == db_before
    assert Path(f"{live}-wal").read_bytes() == wal_before
    with contextlib.closing(sqlite3.connect(tmp_path / "copy.db")) as copy:
        assert copy.execute("SELECT count(*) FROM t").fetchone() == (500,)


def test_cli_unwritable_target_is_one_line_and_exit_one(
    cli_db: Path, blob_dir: Path, tmp_path: Path
) -> None:
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        result = runner.invoke(app, ["db", "backup", str(locked / "copy")], catch_exceptions=False)
    finally:
        locked.chmod(0o700)

    assert result.exit_code == 1
    assert "could not be written" in result.output
    assert "Traceback" not in result.output


def test_cli_names_unreferenced_files_it_could_not_copy(
    cli_db: Path, blob_dir: Path, target: Path
) -> None:
    name = _digest(b"expected bytes")
    _write_blob(blob_dir, b"other bytes", name=name)

    result = runner.invoke(app, ["db", "backup", str(target)], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "1 blob file(s) no row in the copy names could not be copied" in result.output
