"""CLI tests for ``particles corpus fsck``."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.db import session_scope


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def blob_dir(cli_db: Path, tmp_path: Path) -> Path:
    """The blob dir ``cli_db`` points ``PARTICLES_BLOB_DIR`` at."""
    target = tmp_path / "blobs"
    target.mkdir(exist_ok=True)
    return target


@pytest.fixture
def stray_dir(tmp_path: Path) -> Path:
    target = tmp_path / "worktree" / "corpus_blobs"
    target.mkdir(parents=True)
    return target


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_blob(root: Path, content_hash: str, payload: bytes) -> Path:
    target = root / content_hash[:2] / content_hash
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def _seed(content_hash: str, *, entry_id: str = "entry-1") -> None:
    """Seed one blob-bearing snapshot on a corpus entry."""

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
                    archive_path="/somewhere/blob",
                    extraction_status="PENDING",
                )
            )
            await session.commit()

    asyncio.run(_impl())


def test_healthy_store_exits_zero(runner: CliRunner, blob_dir: Path) -> None:
    payload = b"present"
    _seed(_digest(payload))
    _write_blob(blob_dir, _digest(payload), payload)

    result = runner.invoke(app, ["corpus", "fsck"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "present:         1" in result.output
    assert "where extraction will look" in result.output


def test_empty_store_is_healthy(runner: CliRunner, blob_dir: Path) -> None:
    result = runner.invoke(app, ["corpus", "fsck"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "Blobs referenced by the store: 0" in result.output


def test_missing_blob_is_reported_with_its_entry(runner: CliRunner, blob_dir: Path) -> None:
    _seed(_digest(b"gone"), entry_id="entry-9")

    result = runner.invoke(app, ["corpus", "fsck"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "missing:         1" in result.output
    assert "/sources/entry-9.md" in result.output
    assert "Nothing was written to the database" in result.output


def test_stray_is_reported_and_points_at_re_home(
    runner: CliRunner, blob_dir: Path, stray_dir: Path
) -> None:
    payload = b"scattered"
    _seed(_digest(payload))
    _write_blob(stray_dir, _digest(payload), payload)

    result = runner.invoke(
        app, ["corpus", "fsck", "--search", str(stray_dir)], catch_exceptions=False
    )

    assert result.exit_code == 1
    assert "found elsewhere: 1" in result.output
    assert "--re-home" in result.output
    # The audit alone must not copy anything.
    assert not (blob_dir / _digest(payload)[:2] / _digest(payload)).exists()


def test_re_home_copies_the_stray(runner: CliRunner, blob_dir: Path, stray_dir: Path) -> None:
    payload = b"scattered"
    content_hash = _digest(payload)
    _seed(content_hash)
    source = _write_blob(stray_dir, content_hash, payload)

    result = runner.invoke(
        app,
        ["corpus", "fsck", "--search", str(stray_dir), "--re-home"],
        catch_exceptions=False,
    )

    assert "Copied 1 blob(s)" in result.output
    assert (blob_dir / content_hash[:2] / content_hash).read_bytes() == payload
    assert source.exists()
    # Re-running now finds it present.
    again = runner.invoke(app, ["corpus", "fsck"], catch_exceptions=False)
    assert again.exit_code == 0, again.output


def test_dry_run_reports_without_copying(
    runner: CliRunner, blob_dir: Path, stray_dir: Path
) -> None:
    payload = b"scattered"
    content_hash = _digest(payload)
    _seed(content_hash)
    _write_blob(stray_dir, content_hash, payload)

    result = runner.invoke(
        app,
        ["corpus", "fsck", "--search", str(stray_dir), "--dry-run"],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "would copy 1 blob(s)" in result.output
    assert "no files were written" in result.output
    assert not (blob_dir / content_hash[:2] / content_hash).exists()


def test_digest_mismatch_is_rejected(runner: CliRunner, blob_dir: Path, stray_dir: Path) -> None:
    content_hash = _digest(b"the real content")
    _seed(content_hash)
    _write_blob(stray_dir, content_hash, b"something else entirely")

    result = runner.invoke(
        app,
        ["corpus", "fsck", "--search", str(stray_dir), "--re-home"],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "Rejected 1 candidate(s)" in result.output
    assert "missing:         1" in result.output
    assert not (blob_dir / content_hash[:2] / content_hash).exists()


def test_multiple_search_dirs_are_repeatable(
    runner: CliRunner, blob_dir: Path, stray_dir: Path, tmp_path: Path
) -> None:
    second = tmp_path / "other" / "corpus_blobs"
    second.mkdir(parents=True)
    first_payload = b"in the first tree"
    second_payload = b"in the second tree"
    _seed(_digest(first_payload), entry_id="entry-a")
    _seed(_digest(second_payload), entry_id="entry-b")
    _write_blob(stray_dir, _digest(first_payload), first_payload)
    _write_blob(second, _digest(second_payload), second_payload)

    result = runner.invoke(
        app,
        ["corpus", "fsck", "--search", str(stray_dir), "--search", str(second), "--re-home"],
        catch_exceptions=False,
    )

    assert "Copied 2 blob(s)" in result.output


def test_absent_store_is_refused_without_creating_one(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostic must not litter the store it is diagnosing.

    ``session_scope()`` on an absent SQLite path creates an empty database, so
    an unguarded run from the wrong directory would leave a stray store *and*
    report a healthy 0-blob audit — a false all-clear on the exact question the
    operator is asking.
    """
    from particles.config import reset_config

    absent = tmp_path / "never-created.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{absent}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    reset_config()

    result = runner.invoke(app, ["corpus", "fsck"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "nothing to audit" in result.output
    assert not absent.exists()
