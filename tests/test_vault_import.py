# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the retrospective vault importer.

Covers:
  - ``deposit_vault()`` directory walk + ``.md`` filter + skip rules
  - ``LOCAL_MARKDOWN`` source-type stamping
  - idempotency (re-running on the same vault is a no-op for unchanged files:
    no new entry and no new PENDING snapshot)
  - empty-vault behavior
  - progress-callback invocation shape
  - ``_strip_obsidian_frontmatter()`` parsing edge cases
  - ``GeneralExtractor`` routes ``LOCAL_MARKDOWN`` through frontmatter stripping
  - the ``particles import vault`` Typer command
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import unquote, urlparse

import pytest
from typer.testing import CliRunner

from particles.api.cli import app

# ---------------------------------------------------------------------------
# _strip_obsidian_frontmatter
# ---------------------------------------------------------------------------


class TestStripObsidianFrontmatter:
    """The frontmatter helper handles real-world Obsidian patterns + edge cases."""

    def test_standard_obsidian_frontmatter_stripped(self) -> None:
        from particles.extraction.general import _strip_obsidian_frontmatter

        text = (
            "---\n"
            "title: Test note\n"
            "tags: [foo, bar]\n"
            "aliases: [TN]\n"
            "---\n"
            "# My Note\n\n"
            "The capital of France is Paris.\n"
        )
        meta, body = _strip_obsidian_frontmatter(text)
        assert meta == {
            "title": "Test note",
            "tags": ["foo", "bar"],
            "aliases": ["TN"],
        }
        assert body == "# My Note\n\nThe capital of France is Paris.\n"

    def test_no_frontmatter_returns_text_unchanged(self) -> None:
        from particles.extraction.general import _strip_obsidian_frontmatter

        text = "# Just a heading\n\nNo metadata here.\n"
        meta, body = _strip_obsidian_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_unclosed_frontmatter_returns_text_unchanged(self) -> None:
        """A `---\\n` opener with no closer is NOT frontmatter — preserve body."""
        from particles.extraction.general import _strip_obsidian_frontmatter

        text = "---\nthis looks like frontmatter but never closes\n# heading\n"
        meta, body = _strip_obsidian_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_malformed_yaml_strips_but_returns_empty_meta(self) -> None:
        """Bad YAML inside a valid frontmatter block: drop the block, recover nothing."""
        from particles.extraction.general import _strip_obsidian_frontmatter

        text = "---\nkey: : :\n---\n# body\n"
        meta, body = _strip_obsidian_frontmatter(text)
        assert meta == {}
        # Body still cleaned — operator-visible behavior.
        assert body == "# body\n"

    def test_empty_frontmatter_block_returns_empty_meta(self) -> None:
        from particles.extraction.general import _strip_obsidian_frontmatter

        text = "---\n---\n# body only\n"
        meta, body = _strip_obsidian_frontmatter(text)
        assert meta == {}
        assert body == "# body only\n"

    def test_crlf_line_endings_handled(self) -> None:
        """Notes copied between platforms may have CRLF endings."""
        from particles.extraction.general import _strip_obsidian_frontmatter

        text = "---\r\ntitle: x\r\n---\r\n# body\r\n"
        meta, body = _strip_obsidian_frontmatter(text)
        assert meta == {"title": "x"}
        assert "body" in body


# ---------------------------------------------------------------------------
# deposit_vault
# ---------------------------------------------------------------------------


class TestDepositVault:
    """``deposit_vault`` walks the directory and stamps ``LOCAL_MARKDOWN``."""

    @pytest.mark.asyncio
    async def test_walks_md_files_and_stamps_local_markdown(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_vault
        from particles.corpus.store import CorpusEntryRow

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "alpha.md").write_text("# Alpha\n\nFirst note.\n")
        sub = vault / "topic"
        sub.mkdir()
        (sub / "beta.md").write_text("# Beta\n\nSecond note.\n")
        (sub / "gamma.markdown").write_text("# Gamma\n\nThird.\n")

        results = await deposit_vault(db_session, vault, deposited_by="tester")  # type: ignore[arg-type]
        assert len(results) == 3

        # Every entry is stamped LOCAL_MARKDOWN.
        for entry_id, _ in results:
            row = (
                await db_session.execute(  # type: ignore[union-attr]
                    select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
                )
            ).scalar_one()
            assert row.source_type == SourceType.LOCAL_MARKDOWN.value

    @pytest.mark.asyncio
    async def test_skips_non_markdown_files(self, tmp_path: Path, db_session: object) -> None:
        from particles.corpus.deposit import deposit_vault

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "note.md").write_text("# Note\n")
        (vault / "image.png").write_bytes(b"\x89PNG")
        (vault / "draft.txt").write_text("plain")
        (vault / "data.json").write_text("{}")

        results = await deposit_vault(db_session, vault)  # type: ignore[arg-type]
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_skips_dotted_and_underscored_directories(
        self, tmp_path: Path, db_session: object
    ) -> None:
        """Obsidian's ``.obsidian/`` settings + ``_attachments/`` scaffolds are dropped."""
        from particles.corpus.deposit import deposit_vault

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "keep.md").write_text("# keep\n")

        obsidian = vault / ".obsidian"
        obsidian.mkdir()
        (obsidian / "config.md").write_text("# settings junk\n")

        attachments = vault / "_attachments"
        attachments.mkdir()
        (attachments / "scaffold.md").write_text("# scaffold\n")

        # Dotted file at the root is also skipped.
        (vault / ".hidden.md").write_text("# hidden\n")
        (vault / "_template.md").write_text("# template\n")

        results = await deposit_vault(db_session, vault)  # type: ignore[arg-type]
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_empty_vault_returns_empty_list_no_error(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from particles.corpus.deposit import deposit_vault

        vault = tmp_path / "empty"
        vault.mkdir()
        messages: list[str] = []
        results = await deposit_vault(
            db_session,  # type: ignore[arg-type]
            vault,
            progress=messages.append,
        )
        assert results == []
        assert any("No .md files" in m for m in messages)

    @pytest.mark.asyncio
    async def test_missing_directory_raises(self, tmp_path: Path, db_session: object) -> None:
        from particles.corpus.deposit import deposit_vault

        with pytest.raises(ValueError, match="Vault directory not found"):
            await deposit_vault(db_session, tmp_path / "nope")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_idempotent_redeposit(self, tmp_path: Path, db_session: object) -> None:
        """Re-running on the same vault must not create duplicate entries.

        Uses the existing ``content_hash`` deduplication in ``write_entry_and_snapshot``;
        ``(entry_id)`` from the second pass MUST match the first pass.
        """
        from particles.corpus.deposit import deposit_vault

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "stable.md").write_text("# Same content every run.\n")

        first = await deposit_vault(db_session, vault)  # type: ignore[arg-type]
        await db_session.commit()  # type: ignore[union-attr]
        second = await deposit_vault(db_session, vault)  # type: ignore[arg-type]
        await db_session.commit()  # type: ignore[union-attr]

        assert len(first) == len(second) == 1
        assert first[0][0] == second[0][0]  # same entry_id — dedup hit
        assert first[0][1] == second[0][1]  # same snapshot_id — nothing written

    @pytest.mark.asyncio
    async def test_reimport_queues_no_pending_snapshot_for_unchanged_notes(
        self, tmp_path: Path, db_session: object
    ) -> None:
        """The repro: three passes over a vault, one note edited before the third.

        Passes two and three must add a PENDING snapshot only for the edited note,
        so ``extract --all-pending`` pays for exactly what changed.
        """
        from sqlalchemy import func, select

        from particles.core.schema import ExtractionStatus
        from particles.corpus.deposit import deposit_vault
        from particles.corpus.store import SnapshotRow

        async def pending() -> int:
            stmt = (
                select(func.count())
                .select_from(SnapshotRow)
                .where(SnapshotRow.extraction_status == ExtractionStatus.PENDING.value)
            )
            return int((await db_session.execute(stmt)).scalar_one())  # type: ignore[union-attr]

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "a.md").write_text("# A\n\nAlpha.\n")
        (vault / "b.md").write_text("# B\n\nBeta.\n")
        (vault / "c.md").write_text("# C\n\nGamma.\n")

        await deposit_vault(db_session, vault)  # type: ignore[arg-type]
        await db_session.commit()  # type: ignore[union-attr]
        assert await pending() == 3

        unchanged: list[bool] = []
        await deposit_vault(db_session, vault, out_unchanged=unchanged)  # type: ignore[arg-type]
        await db_session.commit()  # type: ignore[union-attr]
        assert unchanged == [True, True, True]
        assert await pending() == 3  # zero new PENDING snapshots

        (vault / "b.md").write_text("# B\n\nBeta, edited.\n")
        unchanged.clear()
        await deposit_vault(db_session, vault, out_unchanged=unchanged)  # type: ignore[arg-type]
        await db_session.commit()  # type: ignore[union-attr]
        assert unchanged == [True, False, True]
        assert await pending() == 4  # only the edited note

    @pytest.mark.asyncio
    async def test_progress_callback_invoked_in_order(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from particles.corpus.deposit import deposit_vault

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "a.md").write_text("# a\n")
        (vault / "b.md").write_text("# b\n")
        (vault / "c.md").write_text("# c\n")

        messages: list[str] = []
        await deposit_vault(
            db_session,  # type: ignore[arg-type]
            vault,
            progress=messages.append,
        )
        assert len(messages) == 3
        # Stable alphabetical order + "[i/N]" prefix.
        assert messages[0].startswith("[1/3] depositing")
        assert "a.md" in messages[0]
        assert messages[1].startswith("[2/3] depositing")
        assert "b.md" in messages[1]
        assert messages[2].startswith("[3/3] depositing")
        assert "c.md" in messages[2]


# ---------------------------------------------------------------------------
# Single-file deposit_file routing to LOCAL_MARKDOWN
# ---------------------------------------------------------------------------


class TestDepositFileMarkdownDetection:
    """``deposit_file`` on a stray ``.md`` should stamp LOCAL_MARKDOWN."""

    @pytest.mark.asyncio
    async def test_md_extension_routes_to_local_markdown(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_file
        from particles.corpus.store import CorpusEntryRow

        note = tmp_path / "stray.md"
        note.write_text("# Stray note.\n")
        entry_id, _ = await deposit_file(db_session, note)  # type: ignore[arg-type]
        row = (
            await db_session.execute(  # type: ignore[union-attr]
                select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
            )
        ).scalar_one()
        assert row.source_type == SourceType.LOCAL_MARKDOWN.value


# ---------------------------------------------------------------------------
# GeneralExtractor routing for LOCAL_MARKDOWN
# ---------------------------------------------------------------------------


class TestGeneralExtractorMarkdownRouting:
    """``GeneralExtractor`` must strip frontmatter when source_type=LOCAL_MARKDOWN."""

    def test_extract_single_pass_strips_frontmatter_when_markdown(self) -> None:
        from particles.core.schema import (
            ExtractionStatus,
            Snapshot,
            UncertaintyNature,
            WarcRecordType,
        )
        from particles.extraction.general import (
            CandidateParticle,
            ExtractionResult,
            GeneralExtractor,
        )

        # Capture what the LLM seam actually receives.
        captured: list[str] = []

        async def _fake_llm(
            text: str, **_kwargs: object
        ) -> tuple[list[CandidateParticle], list[str], bool]:
            captured.append(text)
            return (
                [
                    CandidateParticle(
                        content="The capital of France is Paris.",
                        confidence_value=0.95,
                        uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    )
                ],
                [],
                False,
            )

        # Reach inside the extractor: monkeypatch the module-level _call_llm
        # used by _extract_single_pass.
        import particles.extraction.general as gen

        original = gen._call_llm
        gen._call_llm = _fake_llm  # type: ignore[assignment]
        try:
            snap = Snapshot(
                content_hash="a" * 64,
                extraction_status=ExtractionStatus.PENDING,
                warc_record_type=WarcRecordType.RESPONSE,
            )
            content = (
                b"---\ntitle: France\ntags: [geo, capital]\n---\n"
                b"# France\n\nThe capital of France is Paris.\n"
            )
            result = asyncio.run(
                GeneralExtractor().extract(snap, content, source_type="LOCAL_MARKDOWN")
            )
        finally:
            gen._call_llm = original  # type: ignore[assignment]

        assert len(captured) == 1
        prompt_text = captured[0]
        # Frontmatter is gone …
        assert "title: France" not in prompt_text
        assert "tags:" not in prompt_text
        # … but body remains.
        assert "Paris" in prompt_text
        assert isinstance(result, ExtractionResult)

    def test_extract_single_pass_preserves_frontmatter_when_not_markdown(self) -> None:
        """For non-markdown source types, the helper is NOT applied — preserves backwards compat."""
        from particles.core.schema import (
            ExtractionStatus,
            Snapshot,
            UncertaintyNature,
            WarcRecordType,
        )
        from particles.extraction.general import CandidateParticle, GeneralExtractor

        captured: list[str] = []

        async def _fake_llm(
            text: str, **_kwargs: object
        ) -> tuple[list[CandidateParticle], list[str], bool]:
            captured.append(text)
            return (
                [
                    CandidateParticle(
                        content="x",
                        confidence_value=0.5,
                        uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    )
                ],
                [],
                False,
            )

        import particles.extraction.general as gen

        original = gen._call_llm
        gen._call_llm = _fake_llm  # type: ignore[assignment]
        try:
            snap = Snapshot(
                content_hash="a" * 64,
                extraction_status=ExtractionStatus.PENDING,
                warc_record_type=WarcRecordType.RESPONSE,
            )
            # Source content is markdown-frontmatter-shaped but the
            # extractor is told this is a different source type.
            content = b"---\ntitle: Plain\n---\n# Body\n"
            asyncio.run(GeneralExtractor().extract(snap, content, source_type="WEB_PAGE"))
        finally:
            gen._call_llm = original  # type: ignore[assignment]

        assert len(captured) == 1
        # Frontmatter survives — the WEB_PAGE path makes no markdown-specific assumptions.
        assert "title: Plain" in captured[0]


# ---------------------------------------------------------------------------
# CLI: `particles import vault`
# ---------------------------------------------------------------------------


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


class TestImportVaultCli:
    """``particles import vault`` end-to-end shape (output + exit code)."""

    def test_help_lists_vault_subcommand(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["import", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert "vault" in clean

    def test_import_vault_deposits_files(self, tmp_path: Path, cli_db: Path) -> None:
        # Build a small vault.
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "one.md").write_text("# one\n")
        (vault / "two.md").write_text("# two\n")

        runner = CliRunner()
        result = runner.invoke(app, ["import", "vault", str(vault)], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Deposited 2 note(s)" in result.output
        assert "2 new or changed; changes take effect at the next extract." in result.output

    def test_import_vault_rerun_reports_unchanged(self, tmp_path: Path, cli_db: Path) -> None:
        """A re-run says how many notes it skipped, and writes no snapshot for them."""
        import asyncio

        from sqlalchemy import func, select

        from particles.corpus.store import SnapshotRow
        from particles.db import session_scope

        async def snapshot_count() -> int:
            async with session_scope() as session:
                stmt = select(func.count()).select_from(SnapshotRow)
                return int((await session.execute(stmt)).scalar_one())

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "one.md").write_text("# one\n")
        (vault / "two.md").write_text("# two\n")
        runner = CliRunner()

        first = runner.invoke(app, ["import", "vault", str(vault)], catch_exceptions=False)
        assert first.exit_code == 0
        assert "unchanged" not in first.output
        assert asyncio.run(snapshot_count()) == 2

        (vault / "two.md").write_text("# two, edited\n")
        second = runner.invoke(app, ["import", "vault", str(vault)], catch_exceptions=False)
        assert second.exit_code == 0
        out = _strip_ansi(second.output)
        assert "Deposited 2 note(s)" in out
        assert "1 new or changed, 1 unchanged; changes take effect at the next extract." in out
        assert asyncio.run(snapshot_count()) == 3

    def test_import_vault_progress_logged_when_verbose(self, tmp_path: Path, cli_db: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "note.md").write_text("# n\n")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["import", "vault", str(vault), "--verbose"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        # Progress is emitted on stderr (typer.echo err=True) — CliRunner
        # merges it into result.output by default.
        assert "[1/1] depositing" in clean

    def test_import_vault_nonexistent_dir_exits_nonzero(self, tmp_path: Path, cli_db: Path) -> None:
        runner = CliRunner()
        # Typer's `exists=True` argument validation rejects before our impl runs.
        result = runner.invoke(app, ["import", "vault", str(tmp_path / "no-such-dir")])
        assert result.exit_code != 0

    def test_import_vault_routes_to_engine_in_remote_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In remote mode the verb routes each walked file to the engine."""
        from particles.api.client.base import DepositOutcome
        from particles.api.client.http import HttpBackend
        from particles.config import reset_config
        from particles.core.schema import SourceType

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "one.md").write_text("# one\n")
        (vault / "two.md").write_text("# two\n")
        (vault / "_skip.md").write_text("ignored\n")  # underscore ⇒ ignored by the walk

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://127.0.0.1:8099")
        reset_config()

        calls: list[tuple[str, Any]] = []

        async def _fake_deposit_file(self: Any, path: Path, **kwargs: Any) -> DepositOutcome:
            calls.append((path.name, kwargs["source_type"]))
            # The engine reports one file unchanged.
            return DepositOutcome(
                entry_id=f"e{len(calls)}", snapshot_id="s", unchanged=path.name == "one.md"
            )

        monkeypatch.setattr(HttpBackend, "deposit_file", _fake_deposit_file)

        runner = CliRunner()
        result = runner.invoke(app, ["import", "vault", str(vault)], catch_exceptions=False)
        assert result.exit_code == 0, _strip_ansi(result.output)
        assert "Deposited 2 file(s)" in _strip_ansi(result.output)
        assert "1 unchanged (skipped)" in _strip_ansi(result.output)
        assert {n for n, _ in calls} == {"one.md", "two.md"}  # _skip.md excluded
        assert all(st == SourceType.LOCAL_MARKDOWN for _, st in calls)

    def test_remote_mode_says_the_living_note_flags_do_not_apply(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deferred: the upload path carries no path or mutability."""
        from particles.api.client.base import DepositOutcome
        from particles.api.client.http import HttpBackend
        from particles.config import reset_config

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "one.md").write_text("# one\n")
        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://127.0.0.1:8099")
        reset_config()

        async def _fake(self: Any, path: Path, **kwargs: Any) -> DepositOutcome:
            return DepositOutcome(entry_id="e", snapshot_id="s", unchanged=False)

        monkeypatch.setattr(HttpBackend, "deposit_file", _fake)
        result = CliRunner().invoke(
            app, ["import", "vault", str(vault), "--confirm-absent"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "do not apply" in _strip_ansi(result.output)


class TestNewestFirst:
    """A first vault pass reads the notes a person is working in first."""

    def test_order_is_descending_mtime_ties_by_path(self) -> None:
        from particles.operations.vault_import import newest_first

        files = [Path("c.md"), Path("a.md"), Path("b.md")]
        assert newest_first(files, [10.0, 30.0, 10.0]) == [
            Path("a.md"),
            Path("b.md"),
            Path("c.md"),
        ]

    def test_a_flat_vault_is_detected(self) -> None:
        from particles.operations.vault_import import mtimes_look_rewritten

        assert mtimes_look_rewritten([1000.0 + i for i in range(30)])
        assert not mtimes_look_rewritten([1000.0 + i * 86_400 for i in range(30)])
        # Too few notes to judge.
        assert not mtimes_look_rewritten([1000.0] * 19)

    def test_import_deposits_newest_first_and_prices_the_queue(
        self, tmp_path: Path, cli_db: Path
    ) -> None:
        import os

        from sqlalchemy import select

        from particles.corpus.store import CorpusEntryRow
        from particles.db import session_scope

        vault = tmp_path / "vault"
        vault.mkdir()
        for name, day in (("a.md", 1), ("b.md", 3), ("c.md", 2)):
            path = vault / name
            path.write_text(f"# {name}\n\nThe note {name} says something.\n")
            os.utime(path, (day * 86_400.0, day * 86_400.0))

        result = CliRunner().invoke(app, ["import", "vault", str(vault)], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        out = _strip_ansi(result.output)
        assert "The next extract reads 3 note(s): about US$" in out
        assert "On claude-haiku-5-5: about US$" in out
        assert "particles extract --all-pending --budget-usd 3" in out
        assert "share one modification time" not in out

        async def deposit_order() -> list[str]:
            async with session_scope() as session:
                rows = await session.execute(
                    select(CorpusEntryRow.uri_r).order_by(CorpusEntryRow.created_at)
                )
                return [uri.rsplit("/", 1)[-1] for uri in rows.scalars()]

        assert asyncio.run(deposit_order()) == ["b.md", "c.md", "a.md"]

    def test_a_flat_vault_says_so(self, tmp_path: Path, cli_db: Path) -> None:
        import os

        vault = tmp_path / "vault"
        vault.mkdir()
        for i in range(20):
            path = vault / f"n{i:02d}.md"
            path.write_text(f"# n{i}\n\nNote {i}.\n")
            os.utime(path, (5_000.0, 5_000.0))
        result = CliRunner().invoke(app, ["import", "vault", str(vault)], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert "share one modification time" in _strip_ansi(result.output)

    def test_an_unchanged_reimport_prices_nothing(self, tmp_path: Path, cli_db: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "one.md").write_text("# one\n")
        runner = CliRunner()
        runner.invoke(app, ["import", "vault", str(vault)], catch_exceptions=False)
        again = runner.invoke(app, ["import", "vault", str(vault)], catch_exceptions=False)
        assert "The next extract reads" not in _strip_ansi(again.output)


class TestLivingNotesCli:
    """``import vault`` and ``extract --all-pending`` as the operator sees them."""

    @staticmethod
    def _vault(tmp_path: Path) -> Path:
        vault = tmp_path / "vault"
        vault.mkdir()
        for name in ("Tasks", "Reading", "Garden"):
            (vault / f"{name}.md").write_text(f"{name} note.\n")
        return vault

    def test_the_first_deletion_is_listed_and_exits_1(self, tmp_path: Path, cli_db: Path) -> None:
        vault = self._vault(tmp_path)
        runner = CliRunner()
        assert runner.invoke(app, ["import", "vault", str(vault)]).exit_code == 0
        (vault / "Tasks.md").write_text("Tasks note, edited.\n")
        (vault / "Reading.md").unlink()

        refused = runner.invoke(app, ["import", "vault", str(vault)])
        out = _strip_ansi(refused.output)
        assert refused.exit_code == 1
        assert "first deletion under this folder" in out
        assert "  Reading.md" in out and "--confirm-absent" in out

        confirmed = runner.invoke(app, ["import", "vault", str(vault), "--confirm-absent"])
        out = _strip_ansi(confirmed.output)
        assert confirmed.exit_code == 0, out
        assert "0 new or changed, 2 unchanged, 1 no longer found" in out
        assert "changes take effect at the next extract" in out
        assert "particles corpus delete <entry>" in out

    def test_confirm_and_keep_cannot_be_combined(self, tmp_path: Path, cli_db: Path) -> None:
        vault = self._vault(tmp_path)
        result = CliRunner().invoke(
            app, ["import", "vault", str(vault), "--confirm-absent", "--keep-missing"]
        )
        assert result.exit_code == 2

    def test_mutability_accepts_only_the_two_promises(self, tmp_path: Path, cli_db: Path) -> None:
        vault = self._vault(tmp_path)
        runner = CliRunner()
        bad = runner.invoke(app, ["import", "vault", str(vault), "--mutability", "EPHEMERAL"])
        assert bad.exit_code == 2
        ok = runner.invoke(app, ["import", "vault", str(vault), "--mutability", "stable"])
        assert ok.exit_code == 0

        async def mutabilities() -> set[str]:
            from sqlalchemy import select

            from particles.corpus.store import CorpusEntryRow
            from particles.db import session_scope

            async with session_scope() as session:
                rows = await session.execute(select(CorpusEntryRow.mutability))
                return {str(m) for (m,) in rows.all()}

        assert asyncio.run(mutabilities()) == {"STABLE"}

    def test_extract_all_pending_processes_an_absence_with_no_model_call(
        self, tmp_path: Path, cli_db: Path
    ) -> None:
        from particles.llm import set_client

        vault = self._vault(tmp_path)
        runner = CliRunner()
        assert runner.invoke(app, ["import", "vault", str(vault)]).exit_code == 0

        async def mark_extracted() -> None:
            from sqlalchemy import update

            from particles.core.schema import ExtractionStatus
            from particles.corpus.store import SnapshotRow
            from particles.db import session_scope

            async with session_scope() as session:
                await session.execute(
                    update(SnapshotRow).values(extraction_status=ExtractionStatus.COMPLETE.value)
                )
                await session.commit()

        asyncio.run(mark_extracted())
        (vault / "Reading.md").unlink()
        assert (
            runner.invoke(app, ["import", "vault", str(vault), "--confirm-absent"]).exit_code == 0
        )

        client = MagicMock()
        set_client(client)
        try:
            result = runner.invoke(app, ["extract", "--all-pending"], catch_exceptions=False)
        finally:
            set_client(None)
        out = _strip_ansi(result.output)
        assert result.exit_code == 0, out
        assert "source no longer found: 0 claim(s) retired, no model call" in out
        assert client.messages.mock_calls == []


# ---------------------------------------------------------------------------
# Bundled sample vault (tests/fixtures/llm_wiki_vault)
# ---------------------------------------------------------------------------

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "llm_wiki_vault"

# The 11 content notes the fixture's _GROUND_TRUTH.md declares. Everything
# else in the directory (.obsidian/, _templates/, _GROUND_TRUTH.md itself,
# inventory.csv) exists to exercise the skip rules and must never deposit.
FIXTURE_NOTES = {
    "1933 Double Eagle.md",
    "Coin Silver Standards.md",
    "Flowing Hair Dollar.md",
    "Grading and Condition.md",
    "Home.md",
    "Mint Records.md",
    "Morgan Dollar.md",
    "Most Expensive Coins.md",
    "Philadelphia Mint.md",
    "Seated Liberty Dollar.md",
    "US Mint History.md",
}


class TestFixtureVault:
    """The bundled planted-contradiction vault deposits exactly its content notes.

    The fixture's lint-level ground truth (contradictions C1–C3, stale claim
    S1) is documented in its ``_GROUND_TRUTH.md`` and exercised manually with
    a real API key; these tests pin the deterministic deposit-level contract
    so the fixture cannot silently rot.
    """

    @pytest.mark.asyncio
    async def test_deposits_exactly_the_declared_content_notes(self, db_session: object) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_vault
        from particles.corpus.store import CorpusEntryRow

        results = await deposit_vault(db_session, FIXTURE_VAULT)  # type: ignore[arg-type]
        assert len(results) == len(FIXTURE_NOTES)

        rows = (
            (await db_session.execute(select(CorpusEntryRow))).scalars().all()  # type: ignore[union-attr]
        )
        deposited = {Path(unquote(urlparse(row.uri_r).path)).name for row in rows}
        assert deposited == FIXTURE_NOTES
        for row in rows:
            assert row.source_type == SourceType.LOCAL_MARKDOWN.value

    @pytest.mark.asyncio
    async def test_rerun_is_idempotent(self, db_session: object) -> None:
        from sqlalchemy import func, select

        from particles.corpus.deposit import deposit_vault
        from particles.corpus.store import CorpusEntryRow

        first = await deposit_vault(db_session, FIXTURE_VAULT)  # type: ignore[arg-type]
        second = await deposit_vault(db_session, FIXTURE_VAULT)  # type: ignore[arg-type]
        assert sorted(e for e, _ in first) == sorted(e for e, _ in second)

        count = (
            await db_session.execute(select(func.count()).select_from(CorpusEntryRow))  # type: ignore[union-attr]
        ).scalar_one()
        assert count == len(FIXTURE_NOTES)

    def test_malformed_frontmatter_note_keeps_body(self) -> None:
        """``Seated Liberty Dollar.md`` plants malformed YAML on purpose."""
        from particles.extraction.general import _strip_obsidian_frontmatter

        text = (FIXTURE_VAULT / "Seated Liberty Dollar.md").read_text(encoding="utf-8")
        meta, body = _strip_obsidian_frontmatter(text)
        assert meta == {}
        assert body.lstrip().startswith("# Seated Liberty Dollar")


# ---------------------------------------------------------------------------
# operations.deposit re-export
# ---------------------------------------------------------------------------


def test_deposit_vault_exported_via_operations_shim() -> None:
    """Per particles/api/AGENTS.md the §9.1 surface is the operations shim."""
    from particles.corpus.deposit import deposit_vault as corpus_deposit_vault
    from particles.operations.deposit import deposit_vault as op_deposit_vault

    assert op_deposit_vault is corpus_deposit_vault


# Silence unused-import flake on optional helpers above.
_ = (Any, AsyncMock, MagicMock)
