"""Tests for ``particles corpus cat`` — dump a snapshot's stored content.

Same harness as ``tests/test_cli.py``: a file-based SQLite DB (``cli_db``) so
state survives the fresh ``asyncio.run`` each CLI invocation spins up, real
blobs written to the fixture blob dir via ``save_blob``, and
``typer.testing.CliRunner``. Exercises the local backend end-to-end: selector
resolution → ``load_blob`` → text-preview / ``--raw`` rendering, plus the
empty-content hint that explains a zero-particle extraction.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    CorpusEntry,
    ExtractionStatus,
    Snapshot,
    WarcRecordType,
)

# A minimal HTML page that html2text renders to visible prose.
_HTML = b"<!doctype html><html><body><p>Hello world claim.</p></body></html>"
# A bot-wall / JS-only shell: valid HTML, but no extractable text.
_EMPTY_HTML = b"<!doctype html><html><body></body></html>"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, args: list[str]) -> Any:
    return runner.invoke(app, args, catch_exceptions=False)


async def _add_entry_with_blob(
    content: bytes, *, captured: datetime, entry_id: str | None = None
) -> tuple[str, str]:
    """Insert a corpus entry + one snapshot whose blob holds ``content``."""
    from particles.corpus.store import CorpusEntryRow
    from particles.db import session_scope

    eid = entry_id or str(uuid.uuid4())
    entry = CorpusEntry(
        entry_id=eid,
        source_type="WEB_PAGE",
        uri_r=f"https://example.com/{eid[:8]}",
        deposited_by="test",
    )
    async with session_scope() as session:
        session.add(CorpusEntryRow.from_model(entry))
        await session.commit()
    snap_id = await _add_snapshot_with_blob(eid, content, captured=captured)
    return eid, snap_id


async def _add_snapshot_with_blob(entry_id: str, content: bytes, *, captured: datetime) -> str:
    """Append a snapshot (with its blob) to an existing corpus entry."""
    from particles.corpus.deposit import save_blob, sha256
    from particles.corpus.store import SnapshotRow
    from particles.db import session_scope

    content_hash = sha256(content)
    save_blob(content, content_hash)
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=captured,
        content_hash=content_hash,
        extraction_status=ExtractionStatus.COMPLETE,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    async with session_scope() as session:
        session.add(SnapshotRow.from_model(snap, entry_id))
        await session.commit()
    return snap.snapshot_id


def test_cat_text_preview_by_snapshot(runner: CliRunner, cli_db: Path) -> None:
    """Default output is the extractor's text view — tags stripped, prose kept."""
    _, snap_id = asyncio.run(_add_entry_with_blob(_HTML, captured=datetime.now(UTC)))
    result = _invoke(runner, ["corpus", "cat", snap_id[:8]])
    assert result.exit_code == 0
    assert "Hello world claim." in result.output
    # The text preview strips HTML tags.
    assert "<body>" not in result.output
    # Identifying metadata is surfaced (on stderr, merged into output here).
    assert snap_id in result.output


def test_cat_raw_emits_original_bytes(runner: CliRunner, cli_db: Path) -> None:
    """``--raw`` streams the stored bytes, tags and all."""
    _, snap_id = asyncio.run(_add_entry_with_blob(_HTML, captured=datetime.now(UTC)))
    result = _invoke(runner, ["corpus", "cat", snap_id, "--raw"])
    assert result.exit_code == 0
    assert "<body>" in result.output


def test_cat_by_entry_uses_latest_snapshot(runner: CliRunner, cli_db: Path) -> None:
    """An entry selector resolves to that entry's most-recent snapshot's blob."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    # Older snapshot: empty; newer snapshot: real prose. Both on one entry.
    entry_id, _ = asyncio.run(
        _add_entry_with_blob(_EMPTY_HTML, captured=base, entry_id=str(uuid.uuid4()))
    )
    asyncio.run(_add_snapshot_with_blob(entry_id, _HTML, captured=base + timedelta(days=1)))
    result = _invoke(runner, ["corpus", "cat", entry_id])
    assert result.exit_code == 0
    assert "Hello world claim." in result.output


def test_cat_empty_content_hint(runner: CliRunner, cli_db: Path) -> None:
    """A blob that renders to no text prints the zero-particle explanation."""
    _, snap_id = asyncio.run(_add_entry_with_blob(_EMPTY_HTML, captured=datetime.now(UTC)))
    result = _invoke(runner, ["corpus", "cat", snap_id])
    assert result.exit_code == 0
    assert "empty text preview" in result.output
    assert "0 particles" in result.output


def test_cat_unknown_selector_errors(runner: CliRunner, cli_db: Path) -> None:
    result = _invoke(runner, ["corpus", "cat", "deadbeef"])
    assert result.exit_code == 1
    assert "No blob found" in result.output
