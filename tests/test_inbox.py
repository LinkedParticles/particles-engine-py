"""Tests for the inbox processor (iOS Shortcut → iCloud → Mac)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from particles.operations.inbox import (
    format_failed_marker,
    format_processed_marker,
    parse_inbox,
    process_inbox,
    resolve_inbox_path,
)

# ---------------------------------------------------------------------------
# parse_inbox — pure function, no I/O
# ---------------------------------------------------------------------------


class TestParseInbox:
    def test_empty_text_returns_empty(self) -> None:
        assert parse_inbox("") == []

    def test_single_url(self) -> None:
        assert parse_inbox("https://example.com/a\n") == [(0, "https://example.com/a")]

    def test_skips_blank_lines(self) -> None:
        text = "\n\nhttps://example.com/a\n\nhttps://example.com/b\n\n"
        assert parse_inbox(text) == [
            (2, "https://example.com/a"),
            (4, "https://example.com/b"),
        ]

    def test_skips_comment_and_processed_markers(self) -> None:
        text = (
            "# Processed 2026-05-24T15:30:00+00:00 (entry_id: abc) https://done.example.com\n"
            "# Failed 2026-05-24T15:31:00+00:00 (HTTPError: 404) https://dead.example.com\n"
            "# bare comment\n"
            "https://pending.example.com\n"
        )
        result = parse_inbox(text)
        assert result == [(3, "https://pending.example.com")]

    def test_preserves_line_indices_for_in_place_rewrite(self) -> None:
        """The caller uses line_index to rewrite that specific line in
        the original buffer — indices must be 0-based positions in the
        original split."""
        text = "# done\n\nhttps://a.example.com\n# done\nhttps://b.example.com\n"
        result = parse_inbox(text)
        assert result == [(2, "https://a.example.com"), (4, "https://b.example.com")]

    def test_strips_surrounding_whitespace(self) -> None:
        # iOS Shortcuts sometimes append trailing whitespace; tolerate.
        assert parse_inbox("  https://example.com/a  \n") == [(0, "https://example.com/a")]


# ---------------------------------------------------------------------------
# format_* helpers
# ---------------------------------------------------------------------------


class TestFormatMarkers:
    def test_processed_marker_includes_timestamp_entry_id_and_url(self) -> None:
        when = datetime(2026, 5, 24, 15, 30, 0, tzinfo=UTC)
        marker = format_processed_marker("https://example.com/foo", "abc12345", now=when)
        assert marker.startswith("# Processed 2026-05-24T15:30:00+00:00")
        assert "entry_id: abc12345" in marker
        assert "https://example.com/foo" in marker

    def test_failed_marker_collapses_newlines_in_error(self) -> None:
        when = datetime(2026, 5, 24, 15, 30, 0, tzinfo=UTC)
        marker = format_failed_marker(
            "https://example.com/foo",
            "ValueError: something\nmulti-line\nproblem",
            now=when,
        )
        # Must be one line so the file stays parseable.
        assert "\n" not in marker
        assert "multi-line" in marker
        assert "https://example.com/foo" in marker

    def test_processed_marker_is_skipped_by_parse_inbox(self) -> None:
        """The marker the processor writes must be recognised as
        processed on the next run — otherwise we'd re-deposit forever."""
        marker = format_processed_marker(
            "https://example.com/foo", "abc12345", now=datetime.now(UTC)
        )
        assert parse_inbox(marker + "\n") == []


# ---------------------------------------------------------------------------
# process_inbox — exercises the orchestration with a mocked deposit_url
# ---------------------------------------------------------------------------


class TestProcessInbox:
    @pytest.mark.asyncio
    async def test_missing_file_returns_empty_summary(self, tmp_path: Path) -> None:
        path = tmp_path / "no-such-inbox.txt"
        # Session is unused on this code path; pass a dummy.
        result = await process_inbox(session=AsyncMock(), inbox_path=path)
        assert result == {"processed": [], "failed": [], "skipped": []}

    @pytest.mark.asyncio
    async def test_empty_file_returns_empty_summary(self, tmp_path: Path) -> None:
        path = tmp_path / "inbox.txt"
        path.write_text("")
        result = await process_inbox(session=AsyncMock(), inbox_path=path)
        assert result == {"processed": [], "failed": [], "skipped": []}

    @pytest.mark.asyncio
    async def test_processes_pending_url_and_rewrites_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "inbox.txt"
        path.write_text("https://example.com/foo\n")
        session = AsyncMock()
        session.commit = AsyncMock(return_value=None)
        with patch(
            "particles.corpus.deposit.deposit_url",
            new=AsyncMock(return_value=("entry-1", "snap-1")),
        ):
            result = await process_inbox(session, path)
        assert result["processed"] == ["https://example.com/foo"]
        assert result["failed"] == []
        # Inbox is rewritten with the processed marker; next parse sees nothing.
        rewritten = path.read_text()
        assert "# Processed" in rewritten
        assert "entry_id: entry-1" in rewritten
        assert "https://example.com/foo" in rewritten
        assert parse_inbox(rewritten) == []

    @pytest.mark.asyncio
    async def test_failed_deposit_marks_failed_and_continues(self, tmp_path: Path) -> None:
        path = tmp_path / "inbox.txt"
        path.write_text("https://example.com/bad\nhttps://example.com/good\n")
        session = AsyncMock()
        session.commit = AsyncMock(return_value=None)
        session.rollback = AsyncMock(return_value=None)

        calls = {"n": 0}

        async def _fake_deposit(*args: object, **kwargs: object) -> tuple[str, str]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("network unreachable")
            return ("entry-good", "snap-good")

        with patch("particles.corpus.deposit.deposit_url", new=_fake_deposit):
            result = await process_inbox(session, path)

        # Both URLs were attempted; one succeeded, one failed.
        assert result["processed"] == ["https://example.com/good"]
        assert result["failed"] == ["https://example.com/bad"]
        rewritten = path.read_text()
        assert "# Failed" in rewritten
        assert "network unreachable" in rewritten
        assert "# Processed" in rewritten

    @pytest.mark.asyncio
    async def test_does_not_redeposit_already_processed_urls(self, tmp_path: Path) -> None:
        """A second run against a fully-processed inbox must do nothing
        — the processed markers are the cache."""
        path = tmp_path / "inbox.txt"
        path.write_text(format_processed_marker("https://example.com/done", "abc12345") + "\n")
        deposit_mock = AsyncMock(return_value=("nope", "nope"))
        with patch("particles.corpus.deposit.deposit_url", new=deposit_mock):
            result = await process_inbox(session=AsyncMock(), inbox_path=path)
        assert result == {"processed": [], "failed": [], "skipped": []}
        deposit_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_preserves_unrelated_lines(self, tmp_path: Path) -> None:
        """Comments and existing markers in the file survive the rewrite."""
        path = tmp_path / "inbox.txt"
        original = (
            "# operator note: my reading list\n"
            "# Processed 2026-05-01T00:00:00+00:00 (entry_id: old) https://old.example.com\n"
            "https://new.example.com\n"
        )
        path.write_text(original)
        with patch(
            "particles.corpus.deposit.deposit_url",
            new=AsyncMock(return_value=("new-entry", "new-snap")),
        ):
            await process_inbox(session=AsyncMock(), inbox_path=path)
        rewritten = path.read_text()
        assert "# operator note: my reading list" in rewritten
        assert "https://old.example.com" in rewritten
        assert "entry_id: new-entry" in rewritten


# ---------------------------------------------------------------------------
# resolve_inbox_path — config wiring
# ---------------------------------------------------------------------------


class TestResolveInboxPath:
    def test_returns_none_when_unset(self) -> None:
        from particles.config import reset_config

        reset_config()  # defensive — autouse fixture also does this
        assert resolve_inbox_path() is None

    def test_expands_tilde(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("INBOX_FILE_PATH", "~/inbox.txt")
        reset_config()
        path = resolve_inbox_path()
        assert path is not None
        assert path == tmp_path / "inbox.txt"
