"""Tests for the retrospective vault importer.

Covers:
  - ``deposit_vault()`` directory walk + ``.md`` filter + skip rules
  - ``LOCAL_MARKDOWN`` source-type stamping
  - idempotency (re-running on the same vault is a no-op for unchanged files)
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
        assert "Deposited 2 file(s)" in result.output

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
            return DepositOutcome(entry_id=f"e{len(calls)}", snapshot_id="s")

        monkeypatch.setattr(HttpBackend, "deposit_file", _fake_deposit_file)

        runner = CliRunner()
        result = runner.invoke(app, ["import", "vault", str(vault)], catch_exceptions=False)
        assert result.exit_code == 0, _strip_ansi(result.output)
        assert "Deposited 2 file(s)" in _strip_ansi(result.output)
        assert {n for n, _ in calls} == {"one.md", "two.md"}  # _skip.md excluded
        assert all(st == SourceType.LOCAL_MARKDOWN for _, st in calls)


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
