"""Tests for the Obsidian Web Clipper deposit pathway.

Covers:
  - ``_split_frontmatter`` — leading YAML header split + graceful degradation
  - ``deposit_web_clipper`` frontmatter → deposit-field mapping:
    * ``source`` / ``url`` → ``uri_r`` (fragment-stripped, not fetched)
    * ``published`` → ``content_published_at``
    * frontmatter ``tags`` ∪ run-wide ``--tags`` → entry tags
    * source type stamped ``WEB_PAGE``; deposited bytes = frontmatter-stripped body
  - content-hash dedup (re-run is idempotent; body-hash, not file-hash)
  - the ``--date`` > frontmatter ``published`` precedence
  - graceful fallback to ``LOCAL_MARKDOWN`` when frontmatter is absent / malformed
  - the ``particles import web-clipper`` Typer command
  - the ``operations.deposit`` re-export

No network: ``deposit_web_clipper`` deposits already-local bytes and never
fetches, so there is no HTTP seam to mock.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app


def _as_utc(dt: datetime) -> datetime:
    # SQLite drops tzinfo on round-trip, so a stored UTC datetime reads back
    # naive; normalize before comparing (mirrors tests/test_deposit_date.py).
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


# A representative Obsidian Web Clipper capture.
_CAPTURE = """\
---
title: "The Title of the Article"
source: https://example.com/the-article
author:
  - "[[Jane Author]]"
published: 2026-05-12
created: 2026-06-20T14:03:11
description: "A one-line summary."
tags:
  - clippings
  - economics
---
The captured article body, already converted to Markdown.

Second paragraph.
"""


# ---------------------------------------------------------------------------
# _split_frontmatter — the parser
# ---------------------------------------------------------------------------


class TestSplitFrontmatter:
    """The leading ``--- … ---`` YAML block is split from the body; bad headers
    degrade to ``({}, text)`` so a scan never aborts."""

    def test_splits_header_and_body(self) -> None:
        from particles.corpus.deposit import _split_frontmatter

        frontmatter, body = _split_frontmatter(_CAPTURE)
        assert frontmatter["source"] == "https://example.com/the-article"
        assert frontmatter["tags"] == ["clippings", "economics"]
        # The body excludes the YAML header.
        assert body.startswith("The captured article body")
        assert "source:" not in body
        assert "Second paragraph." in body

    def test_no_frontmatter_returns_text_verbatim(self) -> None:
        from particles.corpus.deposit import _split_frontmatter

        text = "# A plain note\n\nNo header here.\n"
        frontmatter, body = _split_frontmatter(text)
        assert frontmatter == {}
        assert body == text

    def test_unclosed_fence_degrades(self) -> None:
        from particles.corpus.deposit import _split_frontmatter

        text = "---\ntitle: x\nstill in the header, no closing fence\n"
        frontmatter, body = _split_frontmatter(text)
        assert frontmatter == {}
        assert body == text

    def test_malformed_yaml_degrades(self) -> None:
        from particles.corpus.deposit import _split_frontmatter

        # A tab-indented mapping value is invalid YAML.
        text = "---\nfoo: [unbalanced\n---\nbody\n"
        frontmatter, body = _split_frontmatter(text)
        assert frontmatter == {}
        assert body == text

    def test_horizontal_rule_not_treated_as_fence(self) -> None:
        from particles.corpus.deposit import _split_frontmatter

        text = "***\nnot a fence\n"
        frontmatter, body = _split_frontmatter(text)
        assert frontmatter == {}
        assert body == text


# ---------------------------------------------------------------------------
# deposit_web_clipper — field mapping
# ---------------------------------------------------------------------------


def _write_capture(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text)
    return path


class TestDepositWebClipperMapping:
    """Frontmatter maps onto uri_r / date / tags / source_type; body is deposited."""

    @pytest.mark.asyncio
    async def test_maps_url_date_tags_and_source_type(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_web_clipper, load_blob
        from particles.corpus.store import CorpusEntryRow, SnapshotRow

        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "article.md", _CAPTURE)

        results = await deposit_web_clipper(db_session, root, deposited_by="tester")  # type: ignore[arg-type]
        assert len(results) == 1
        entry_id, snapshot_id = results[0]

        row = (
            await db_session.execute(  # type: ignore[union-attr]
                select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
            )
        ).scalar_one()
        # uri_r is the clipped page URL, not a file:// path.
        assert row.uri_r == "https://example.com/the-article"
        assert row.source_type == SourceType.WEB_PAGE.value
        # frontmatter tags preserved.
        assert set(json.loads(row.tags_json)) == {"clippings", "economics"}

        # content_published_at = the frontmatter `published` date.
        snap = (
            await db_session.execute(  # type: ignore[union-attr]
                select(SnapshotRow).where(SnapshotRow.snapshot_id == snapshot_id)
            )
        ).scalar_one()
        assert snap.content_published_at is not None
        assert _as_utc(snap.content_published_at) == datetime(2026, 5, 12, tzinfo=UTC)

        # Deposited bytes are the frontmatter-stripped body.
        blob = load_blob(snap.content_hash).decode("utf-8")
        assert blob.startswith("The captured article body")
        assert "source:" not in blob

    @pytest.mark.asyncio
    async def test_url_fragment_stripped(self, tmp_path: Path, db_session: object) -> None:
        from sqlalchemy import select

        from particles.corpus.deposit import deposit_web_clipper
        from particles.corpus.store import CorpusEntryRow

        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(
            root,
            "frag.md",
            "---\nsource: https://example.com/page#section-2\n---\nBody.\n",
        )
        results = await deposit_web_clipper(db_session, root)  # type: ignore[arg-type]
        row = (
            await db_session.execute(  # type: ignore[union-attr]
                select(CorpusEntryRow).where(CorpusEntryRow.entry_id == results[0][0])
            )
        ).scalar_one()
        assert row.uri_r == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_url_fallback_key(self, tmp_path: Path, db_session: object) -> None:
        """``url`` is used when ``source`` is absent (url_keys order)."""
        from sqlalchemy import select

        from particles.corpus.deposit import deposit_web_clipper
        from particles.corpus.store import CorpusEntryRow

        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "u.md", "---\nurl: https://example.org/x\n---\nBody.\n")
        results = await deposit_web_clipper(db_session, root)  # type: ignore[arg-type]
        row = (
            await db_session.execute(  # type: ignore[union-attr]
                select(CorpusEntryRow).where(CorpusEntryRow.entry_id == results[0][0])
            )
        ).scalar_one()
        assert row.uri_r == "https://example.org/x"

    @pytest.mark.asyncio
    async def test_run_wide_tags_union_with_frontmatter(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from sqlalchemy import select

        from particles.corpus.deposit import deposit_web_clipper
        from particles.corpus.store import CorpusEntryRow

        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "article.md", _CAPTURE)

        results = await deposit_web_clipper(
            db_session,  # type: ignore[arg-type]
            root,
            tags=["economics", "research"],  # "economics" duplicates a frontmatter tag
        )
        row = (
            await db_session.execute(  # type: ignore[union-attr]
                select(CorpusEntryRow).where(CorpusEntryRow.entry_id == results[0][0])
            )
        ).scalar_one()
        # Per-file ∪ run-wide, de-duplicated.
        assert set(json.loads(row.tags_json)) == {"clippings", "economics", "research"}

    @pytest.mark.asyncio
    async def test_explicit_date_overrides_published(
        self, tmp_path: Path, db_session: object
    ) -> None:
        """precedence: an explicit content_date > the frontmatter date."""
        from sqlalchemy import select

        from particles.corpus.deposit import deposit_web_clipper
        from particles.corpus.store import SnapshotRow

        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "article.md", _CAPTURE)

        override = datetime(2020, 1, 1, tzinfo=UTC)
        results = await deposit_web_clipper(
            db_session,  # type: ignore[arg-type]
            root,
            content_date=override,
        )
        snap = (
            await db_session.execute(  # type: ignore[union-attr]
                select(SnapshotRow).where(SnapshotRow.snapshot_id == results[0][1])
            )
        ).scalar_one()
        assert snap.content_published_at is not None
        assert _as_utc(snap.content_published_at) == override


class TestDepositWebClipperDedup:
    """Body-hash dedup makes a re-run idempotent."""

    @pytest.mark.asyncio
    async def test_idempotent_redeposit(self, tmp_path: Path, db_session: object) -> None:
        from particles.corpus.deposit import deposit_web_clipper

        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "article.md", _CAPTURE)

        first = await deposit_web_clipper(db_session, root)  # type: ignore[arg-type]
        await db_session.commit()  # type: ignore[union-attr]
        second = await deposit_web_clipper(db_session, root)  # type: ignore[arg-type]
        await db_session.commit()  # type: ignore[union-attr]

        assert len(first) == len(second) == 1
        # Same uri_r → dedup hit, one entry.
        assert first[0][0] == second[0][0]

    @pytest.mark.asyncio
    async def test_clip_time_metadata_difference_dedups_on_body(
        self, tmp_path: Path, db_session: object
    ) -> None:
        """Two captures of the same article differing only in clip-time metadata
        collapse to one entry (body is hashed, frontmatter stripped)."""
        from sqlalchemy import func, select

        from particles.corpus.deposit import deposit_web_clipper
        from particles.corpus.store import CorpusEntryRow

        body = "Same article body, identical bytes.\n"
        cap_a = f"---\nsource: https://example.com/a\ncreated: 2026-06-20T10:00:00\n---\n{body}"
        cap_b = f"---\nsource: https://example.com/a\ncreated: 2026-06-21T18:30:00\n---\n{body}"
        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "a1.md", cap_a)
        _write_capture(root, "a2.md", cap_b)

        await deposit_web_clipper(db_session, root)  # type: ignore[arg-type]
        await db_session.commit()  # type: ignore[union-attr]

        count = (
            await db_session.execute(select(func.count()).select_from(CorpusEntryRow))  # type: ignore[union-attr]
        ).scalar_one()
        assert count == 1  # same uri_r → one entry, two snapshots


class TestDepositWebClipperFallback:
    """A capture without parseable frontmatter falls back to LOCAL_MARKDOWN."""

    @pytest.mark.asyncio
    async def test_no_frontmatter_falls_back_to_local_markdown(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_web_clipper
        from particles.corpus.store import CorpusEntryRow

        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "plain.md", "# A hand-written note\n\nNo frontmatter.\n")

        results = await deposit_web_clipper(db_session, root)  # type: ignore[arg-type]
        assert len(results) == 1
        row = (
            await db_session.execute(  # type: ignore[union-attr]
                select(CorpusEntryRow).where(CorpusEntryRow.entry_id == results[0][0])
            )
        ).scalar_one()
        assert row.source_type == SourceType.LOCAL_MARKDOWN.value
        # No source URL → file:// uri_r, not a web URL.
        assert row.uri_r.startswith("file://")

    @pytest.mark.asyncio
    async def test_frontmatter_without_url_falls_back(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_web_clipper
        from particles.corpus.store import CorpusEntryRow

        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "no-url.md", "---\ntitle: No URL here\n---\nBody.\n")

        results = await deposit_web_clipper(db_session, root)  # type: ignore[arg-type]
        row = (
            await db_session.execute(  # type: ignore[union-attr]
                select(CorpusEntryRow).where(CorpusEntryRow.entry_id == results[0][0])
            )
        ).scalar_one()
        assert row.source_type == SourceType.LOCAL_MARKDOWN.value


class TestDepositWebClipperWalk:
    """Walk behaviour: vault ignore policy, empty folder, missing directory."""

    @pytest.mark.asyncio
    async def test_skips_underscore_and_dot_components(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from particles.corpus.deposit import deposit_web_clipper

        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "keep.md", "---\nsource: https://example.com/k\n---\nB.\n")
        attach = root / "_attachments"
        attach.mkdir()
        _write_capture(attach, "scaffold.md", "---\nsource: https://x/y\n---\nB.\n")
        dotdir = root / ".obsidian"
        dotdir.mkdir()
        _write_capture(dotdir, "settings.md", "---\nsource: https://x/z\n---\nB.\n")

        results = await deposit_web_clipper(db_session, root)  # type: ignore[arg-type]
        assert len(results) == 1  # only keep.md

    @pytest.mark.asyncio
    async def test_empty_folder_returns_empty_no_error(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from particles.corpus.deposit import deposit_web_clipper

        root = tmp_path / "empty"
        root.mkdir()
        messages: list[str] = []
        results = await deposit_web_clipper(
            db_session,  # type: ignore[arg-type]
            root,
            progress=messages.append,
        )
        assert results == []
        assert any("No .md captures" in m for m in messages)

    @pytest.mark.asyncio
    async def test_missing_directory_raises(self, tmp_path: Path, db_session: object) -> None:
        from particles.corpus.deposit import deposit_web_clipper

        with pytest.raises(ValueError, match="Captures directory not found"):
            await deposit_web_clipper(db_session, tmp_path / "nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI: `particles import web-clipper`
# ---------------------------------------------------------------------------


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


class TestImportWebClipperCli:
    """``particles import web-clipper`` end-to-end shape (output + exit code)."""

    def test_help_lists_web_clipper_subcommand(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["import", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert "web-clipper" in clean

    def test_import_web_clipper_deposits_captures(self, tmp_path: Path, cli_db: Path) -> None:
        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "one.md", "---\nsource: https://example.com/1\n---\nB.\n")
        _write_capture(root, "two.md", "---\nsource: https://example.com/2\n---\nB.\n")

        runner = CliRunner()
        result = runner.invoke(app, ["import", "web-clipper", str(root)], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Deposited 2 capture(s)" in result.output

    def test_import_web_clipper_nonexistent_dir_exits_nonzero(
        self, tmp_path: Path, cli_db: Path
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["import", "web-clipper", str(tmp_path / "no-such-dir")])
        assert result.exit_code != 0

    def test_import_web_clipper_refuses_in_remote_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In remote mode the local-only verb must fail fast, not write the local DB."""
        from particles.config import reset_config

        root = tmp_path / "clippings"
        root.mkdir()
        _write_capture(root, "one.md", "---\nsource: https://example.com/1\n---\nB.\n")

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://127.0.0.1:8099")
        reset_config()

        runner = CliRunner()
        result = runner.invoke(app, ["import", "web-clipper", str(root)])
        assert result.exit_code != 0
        clean = _strip_ansi(result.output)
        assert "local-only" in clean
        assert "particles deposit" in clean


# ---------------------------------------------------------------------------
# operations.deposit re-export
# ---------------------------------------------------------------------------


def test_deposit_web_clipper_exported_via_operations_shim() -> None:
    """Per particles/api/AGENTS.md the §9.1 surface is the operations shim."""
    from particles.corpus.deposit import deposit_web_clipper as corpus_fn
    from particles.operations.deposit import deposit_web_clipper as op_fn

    assert op_fn is corpus_fn
