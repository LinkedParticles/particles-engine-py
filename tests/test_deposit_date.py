"""Tests for deposit-time content-date capture (particles/corpus/deposit.py).

Covers the pure leading-date detector, the precedence resolver
(`--date` > leading date line > file mtime), the `deposit_file` wiring that
stamps `content_published_at` on the snapshot, and the CLI `--date` validation.

Also covers the date-line splitter (`split_file_by_date` + the
`_parse_date_line` / `_segment_by_date_lines` primitives it shares with the leading-date detector) and the `deposit --split-by-date` CLI flag.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from particles.config import DepositDateConfig, ParticlesConfig
from particles.corpus.deposit import (
    _detect_leading_date,
    _parse_date_line,
    _resolve_content_published_at,
    _segment_by_date_lines,
    deposit_file,
    split_file_by_date,
)
from particles.corpus.store import get_entry, get_snapshot

# ---------------------------------------------------------------------------
# _detect_leading_date — pure, defaults on
# ---------------------------------------------------------------------------


class TestDetectLeadingDate:
    def test_bare_iso_date_first_line(self) -> None:
        got = _detect_leading_date(b"2026-03-15\nFirst journal entry of the day.")
        assert got == datetime(2026, 3, 15, tzinfo=UTC)

    def test_markdown_heading_date(self) -> None:
        assert _detect_leading_date(b"# 2026-03-15\n\nNotes follow") == datetime(
            2026, 3, 15, tzinfo=UTC
        )

    def test_list_marker_date(self) -> None:
        assert _detect_leading_date(b"- 2026-03-15\n") == datetime(2026, 3, 15, tzinfo=UTC)

    def test_date_label_prefix(self) -> None:
        assert _detect_leading_date(b"Date: 2026-03-15\nbody") == datetime(2026, 3, 15, tzinfo=UTC)

    def test_slash_format(self) -> None:
        assert _detect_leading_date(b"2026/03/15\n") == datetime(2026, 3, 15, tzinfo=UTC)

    def test_date_inside_prose_not_matched(self) -> None:
        # Strict whole-line match: a date embedded in a sentence is ignored.
        assert _detect_leading_date(b"Met the team on 2026-03-15 to plan.\n") is None

    def test_blank_lines_skipped_then_date(self) -> None:
        assert _detect_leading_date(b"\n\n   \n2026-03-15\nbody") == datetime(
            2026, 3, 15, tzinfo=UTC
        )

    def test_date_beyond_scan_window_not_matched(self) -> None:
        # Default scan window is 5 non-blank lines; the date sits on line 6.
        body = b"a\nb\nc\nd\ne\n2026-03-15\n"
        assert _detect_leading_date(body) is None

    def test_no_date(self) -> None:
        assert _detect_leading_date(b"Just some prose with no date at the top.\n") is None

    def test_non_utf8_bytes_returns_none(self) -> None:
        assert _detect_leading_date(b"\xff\xfe\x00\x01 binary blob") is None

    def test_disabled_via_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = ParticlesConfig(deposit_date=DepositDateConfig(detect_leading_date=False))
        monkeypatch.setattr("particles.corpus.deposit.get_config", lambda: cfg)
        assert _detect_leading_date(b"2026-03-15\n") is None


# ---------------------------------------------------------------------------
# _resolve_content_published_at — precedence
# ---------------------------------------------------------------------------


def _set_mtime(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


def _as_utc(dt: datetime) -> datetime:
    # SQLite drops tzinfo on round-trip, so a stored UTC datetime reads back
    # naive; normalize before comparing (decay.py does the same).
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


class TestResolvePrecedence:
    def test_override_wins_over_leading_and_mtime(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("2026-03-15\nbody")
        _set_mtime(f, datetime(2020, 1, 1, tzinfo=UTC))
        override = datetime(2012, 1, 2, tzinfo=UTC)
        assert _resolve_content_published_at(f, f.read_bytes(), override) == override

    def test_leading_wins_over_mtime(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("2026-03-15\nbody")
        _set_mtime(f, datetime(2020, 1, 1, tzinfo=UTC))
        assert _resolve_content_published_at(f, f.read_bytes(), None) == datetime(
            2026, 3, 15, tzinfo=UTC
        )

    def test_mtime_fallback_when_no_override_no_leading(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("no date here\nbody")
        _set_mtime(f, datetime(2019, 6, 1, tzinfo=UTC))
        got = _resolve_content_published_at(f, f.read_bytes(), None)
        assert got is not None
        assert got.astimezone(UTC).date() == datetime(2019, 6, 1, tzinfo=UTC).date()

    def test_mtime_fallback_disabled_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = ParticlesConfig(deposit_date=DepositDateConfig(mtime_fallback=False))
        monkeypatch.setattr("particles.corpus.deposit.get_config", lambda: cfg)
        f = tmp_path / "doc.md"
        f.write_text("no date here\nbody")
        assert _resolve_content_published_at(f, f.read_bytes(), None) is None


# ---------------------------------------------------------------------------
# deposit_file — stamps content_published_at on the snapshot
# ---------------------------------------------------------------------------


class TestDepositFileStampsDate:
    @pytest.mark.asyncio
    async def test_leading_date_captured_on_snapshot(
        self, db_session: object, tmp_path: Path
    ) -> None:
        from particles.corpus.store import get_snapshot

        f = tmp_path / "journal.md"
        f.write_text("2026-03-15\nWoke up early, shipped the feature.")
        _, snapshot_id = await deposit_file(db_session, f)  # type: ignore[arg-type]
        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        assert snap.content_published_at is not None
        assert _as_utc(snap.content_published_at) == datetime(2026, 3, 15, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_explicit_content_date_overrides(
        self, db_session: object, tmp_path: Path
    ) -> None:
        from particles.corpus.store import get_snapshot

        f = tmp_path / "concept.md"
        f.write_text("2026-03-15\nbody")  # leading date present...
        override = datetime(2012, 1, 2, tzinfo=UTC)
        _, snapshot_id = await deposit_file(
            db_session,  # type: ignore[arg-type]
            f,
            content_date=override,  # ...but the explicit date wins
        )
        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        assert snap.content_published_at is not None
        assert _as_utc(snap.content_published_at) == override


# ---------------------------------------------------------------------------
# CLI --date validation
# ---------------------------------------------------------------------------


class TestCliDateValidation:
    def test_invalid_date_errors(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["deposit", "some-file.md", "--date", "not-a-date"])
        assert result.exit_code == 1
        assert "ISO date" in result.output


# ---------------------------------------------------------------------------
# _parse_date_line (pure boundary primitive)
# ---------------------------------------------------------------------------


class TestParseDateLine:
    def test_bare_date(self) -> None:
        assert _parse_date_line("2026-03-15") == datetime(2026, 3, 15, tzinfo=UTC)

    def test_heading_marker(self) -> None:
        assert _parse_date_line("# 2026-03-15") == datetime(2026, 3, 15, tzinfo=UTC)

    def test_list_marker(self) -> None:
        assert _parse_date_line("- 2026-03-15") == datetime(2026, 3, 15, tzinfo=UTC)

    def test_date_label_prefix(self) -> None:
        assert _parse_date_line("Date: 2026-03-15") == datetime(2026, 3, 15, tzinfo=UTC)

    def test_slash_format(self) -> None:
        assert _parse_date_line("2026/03/15") == datetime(2026, 3, 15, tzinfo=UTC)

    def test_prose_date_rejected(self) -> None:
        # Strict whole-line match: a date inside a sentence does not match.
        assert _parse_date_line("Met the team on 2026-03-15 to plan.") is None

    def test_empty_rejected(self) -> None:
        assert _parse_date_line("   ") is None
        assert _parse_date_line("---") is None  # markers strip to empty

    def test_ungated_by_detect_leading_date(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The opt-in splitter's primitive must not be silenced by the
        # leading-date auto-detect config flag (that governs the normal path).
        cfg = ParticlesConfig(deposit_date=DepositDateConfig(detect_leading_date=False))
        monkeypatch.setattr("particles.corpus.deposit.get_config", lambda: cfg)
        assert _parse_date_line("2026-03-15") == datetime(2026, 3, 15, tzinfo=UTC)
        # ...while _detect_leading_date stays gated.
        assert _detect_leading_date(b"2026-03-15\n") is None


# ---------------------------------------------------------------------------
# _segment_by_date_lines (generic, genre-neutral segmentation)
# ---------------------------------------------------------------------------


class TestSegmentByDateLines:
    def test_no_date_single_section(self) -> None:
        secs = _segment_by_date_lines("just prose\nmore prose\n")
        assert len(secs) == 1
        assert secs[0][0] is None

    def test_n_dated_sections(self) -> None:
        secs = _segment_by_date_lines("2026-03-15\na\n2026-03-16\nb\n")
        assert [d for d, _ in secs] == [
            datetime(2026, 3, 15, tzinfo=UTC),
            datetime(2026, 3, 16, tzinfo=UTC),
        ]
        # The date line is kept inside its section; content up to the next date
        # line belongs to that section.
        assert secs[0][1] == "2026-03-15\na\n"
        assert secs[1][1] == "2026-03-16\nb\n"

    def test_preamble_preserved_as_dateless_section(self) -> None:
        secs = _segment_by_date_lines("# My Journal\nintro\n\n2026-03-15\nbody\n")
        assert secs[0][0] is None
        assert "My Journal" in secs[0][1]
        assert secs[1][0] == datetime(2026, 3, 15, tzinfo=UTC)

    def test_whitespace_only_preamble_dropped(self) -> None:
        secs = _segment_by_date_lines("\n\n2026-03-15\nbody\n")
        assert len(secs) == 1
        assert secs[0][0] == datetime(2026, 3, 15, tzinfo=UTC)

    def test_prose_date_does_not_split(self) -> None:
        secs = _segment_by_date_lines("We shipped on 2026-03-15 finally.\nmore\n")
        assert len(secs) == 1
        assert secs[0][0] is None


# ---------------------------------------------------------------------------
# split_file_by_date (per-section corpus write)
# ---------------------------------------------------------------------------


class TestSplitFileByDate:
    @pytest.mark.asyncio
    async def test_multi_entry_splits_into_n_dated_entries(
        self, db_session: object, tmp_path: Path
    ) -> None:
        f = tmp_path / "journal.md"
        f.write_text(
            "2026-03-15\nFirst day.\n\n2026-03-16\nSecond day.\n\n2026-03-17\nThird day.\n"
        )
        rows = await split_file_by_date(db_session, f)  # type: ignore[arg-type]
        assert len(rows) == 3

        dates: list[datetime] = []
        for _, snapshot_id in rows:
            snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
            assert snap is not None and snap.content_published_at is not None
            dates.append(_as_utc(snap.content_published_at))
        assert dates == [
            datetime(2026, 3, 15, tzinfo=UTC),
            datetime(2026, 3, 16, tzinfo=UTC),
            datetime(2026, 3, 17, tzinfo=UTC),
        ]

        # Distinct entries, each with a synthetic per-section #entry-<n> fragment.
        entry_ids = [e for e, _ in rows]
        assert len(set(entry_ids)) == 3
        for ordinal, entry_id in enumerate(entry_ids, start=1):
            entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
            assert entry is not None and entry.uri_r is not None
            assert entry.uri_r.endswith(f"#entry-{ordinal}")

    @pytest.mark.asyncio
    async def test_preamble_emitted_with_mtime_fallback(
        self, db_session: object, tmp_path: Path
    ) -> None:
        f = tmp_path / "journal.md"
        f.write_text("# My Journal\nintro paragraph\n\n2026-03-15\nbody\n")
        _set_mtime(f, datetime(2019, 6, 1, tzinfo=UTC))
        rows = await split_file_by_date(db_session, f)  # type: ignore[arg-type]
        assert len(rows) == 2

        preamble_snap = await get_snapshot(db_session, rows[0][1])  # type: ignore[arg-type]
        dated_snap = await get_snapshot(db_session, rows[1][1])  # type: ignore[arg-type]
        assert preamble_snap is not None and preamble_snap.content_published_at is not None
        assert dated_snap is not None and dated_snap.content_published_at is not None
        # Preamble has no date line → mtime fallback.
        assert _as_utc(preamble_snap.content_published_at).date() == datetime(2019, 6, 1).date()
        # Dated section gets its own date.
        assert _as_utc(dated_snap.content_published_at) == datetime(2026, 3, 15, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_single_dated_entry_falls_back_to_one_entry(
        self, db_session: object, tmp_path: Path
    ) -> None:
        # One date line, no preamble → < 2 sections → normal single-entry deposit
        # (no synthetic #entry- fragment), unchanged from today's behaviour.
        f = tmp_path / "note.md"
        f.write_text("2026-03-15\nonly one dated entry here\n")
        rows = await split_file_by_date(db_session, f)  # type: ignore[arg-type]
        assert len(rows) == 1
        entry = await get_entry(db_session, rows[0][0])  # type: ignore[arg-type]
        assert entry is not None and entry.uri_r is not None
        assert "#entry-" not in entry.uri_r
        snap = await get_snapshot(db_session, rows[0][1])  # type: ignore[arg-type]
        assert snap is not None and snap.content_published_at is not None
        assert _as_utc(snap.content_published_at) == datetime(2026, 3, 15, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_no_date_file_falls_back_to_one_entry(
        self, db_session: object, tmp_path: Path
    ) -> None:
        f = tmp_path / "note.md"
        f.write_text("just prose\nwith no dates at all\n")
        rows = await split_file_by_date(db_session, f)  # type: ignore[arg-type]
        assert len(rows) == 1
        entry = await get_entry(db_session, rows[0][0])  # type: ignore[arg-type]
        assert entry is not None and entry.uri_r is not None
        assert "#entry-" not in entry.uri_r

    @pytest.mark.asyncio
    async def test_source_type_inherited_by_every_section(
        self, db_session: object, tmp_path: Path
    ) -> None:
        f = tmp_path / "journal.md"
        f.write_text("2026-03-15\nfirst\n\n2026-03-16\nsecond\n")
        rows = await split_file_by_date(db_session, f, source_type="JOURNAL")  # type: ignore[arg-type]
        assert len(rows) == 2
        for entry_id, _ in rows:
            entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
            assert entry is not None and entry.source_type == "JOURNAL"

    @pytest.mark.asyncio
    async def test_idempotent_on_unchanged_redeposit(
        self, db_session: object, tmp_path: Path
    ) -> None:
        f = tmp_path / "journal.md"
        f.write_text("2026-03-15\nfirst\n\n2026-03-16\nsecond\n")
        first = await split_file_by_date(db_session, f)  # type: ignore[arg-type]
        second = await split_file_by_date(db_session, f)  # type: ignore[arg-type]
        # Same per-section URIs → same entries, no duplicate corpus entries.
        assert [e for e, _ in first] == [e for e, _ in second]

    @pytest.mark.asyncio
    async def test_non_utf8_raises(self, db_session: object, tmp_path: Path) -> None:
        f = tmp_path / "blob.bin"
        f.write_bytes(b"\xff\xfe\x00\x01 binary payload")
        with pytest.raises(ValueError, match="UTF-8"):
            await split_file_by_date(db_session, f)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI --split-by-date validation
# ---------------------------------------------------------------------------


class TestCliSplitByDateValidation:
    def test_split_and_date_mutually_exclusive(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(
            app, ["deposit", "f.md", "--split-by-date", "--date", "2026-03-15"]
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_split_by_date_rejects_url(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["deposit", "https://example.com/x", "--split-by-date"])
        assert result.exit_code == 1
        assert "local files only" in result.output
