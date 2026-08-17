"""Tests for multi-file structured-source deposit.

Covers:
  - the shared ``_iter_tree`` walker with an injected ignore predicate
  - ``deposit_project()`` directory walk + ``.py`` filter + source-tree ignore
    policy (dot-dirs + build/cache dirs pruned; underscore module files KEPT)
  - ``PYTHON_SOURCE`` source-type stamping
  - the ``--ext`` per-invocation override
  - idempotency (re-running on the same tree is a no-op for unchanged files)
  - empty-tree behavior + missing-directory error
  - single-file ``deposit_file`` routing of ``.py`` → ``PYTHON_SOURCE``
  - the ``particles import project`` Typer command
  - the ``operations.deposit`` re-export
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app

# ---------------------------------------------------------------------------
# _iter_tree — the shared walker (injected ignore predicate)
# ---------------------------------------------------------------------------


class TestIterTree:
    """The factored walker recurses, suffix-filters, and applies skip_part."""

    def test_recurses_and_filters_by_suffix_case_insensitive(self, tmp_path: Path) -> None:
        from particles.corpus.deposit import _iter_tree

        (tmp_path / "a.py").write_text("x = 1\n")
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "b.PY").write_text("y = 2\n")  # uppercase suffix still matches
        (sub / "c.txt").write_text("nope")

        found = _iter_tree(tmp_path, {".py"}, lambda _p: False)
        names = [p.name for p in found]
        assert names == ["a.py", "b.PY"]  # sorted, txt excluded

    def test_skip_predicate_keyed_on_relative_components(self, tmp_path: Path) -> None:
        """A skipped component anywhere in the relative path drops the file —
        but components above ``root`` (e.g. a dotted home dir) never do."""
        from particles.corpus.deposit import _iter_tree

        # Put root itself under a dotted parent to prove only *relative* parts
        # are tested.
        dotted_parent = tmp_path / ".config"
        root = dotted_parent / "proj"
        root.mkdir(parents=True)
        (root / "keep.py").write_text("k = 1\n")
        skipme = root / "skip"
        skipme.mkdir()
        (skipme / "dropped.py").write_text("d = 1\n")

        found = _iter_tree(root, {".py"}, lambda p: p == "skip")
        assert [p.name for p in found] == ["keep.py"]

    def test_results_sorted_deterministically(self, tmp_path: Path) -> None:
        from particles.corpus.deposit import _iter_tree

        for name in ("zeta.py", "alpha.py", "mu.py"):
            (tmp_path / name).write_text("v = 0\n")
        found = _iter_tree(tmp_path, {".py"}, lambda _p: False)
        assert [p.name for p in found] == ["alpha.py", "mu.py", "zeta.py"]


# ---------------------------------------------------------------------------
# _skip_source_part — the source-tree ignore policy
# ---------------------------------------------------------------------------


class TestSkipSourcePart:
    """Dot-dirs and configured build/cache dirs are pruned; underscore files kept."""

    def test_dot_prefixed_components_skipped(self) -> None:
        from particles.corpus.deposit import _skip_source_part

        assert _skip_source_part(".git", set())
        assert _skip_source_part(".venv", set())
        assert _skip_source_part(".mypy_cache", set())

    def test_configured_ignore_dirs_skipped(self) -> None:
        from particles.corpus.deposit import _skip_source_part

        ignore = {"__pycache__", "node_modules", "build"}
        assert _skip_source_part("__pycache__", ignore)
        assert _skip_source_part("node_modules", ignore)
        assert not _skip_source_part("src", ignore)

    def test_underscore_module_files_kept(self) -> None:
        """The substantive asymmetry: underscore *files* are NOT skipped."""
        from particles.corpus.deposit import _skip_source_part

        assert not _skip_source_part("__init__.py", {"__pycache__"})
        assert not _skip_source_part("_shared.py", {"__pycache__"})
        assert not _skip_source_part("_logging.py", {"__pycache__"})


# ---------------------------------------------------------------------------
# deposit_project
# ---------------------------------------------------------------------------


def _build_project(root: Path) -> None:
    """A small package tree exercising every ignore-policy branch."""
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""Package."""\n')
    (pkg / "_shared.py").write_text('"""Shared helpers."""\n')
    (pkg / "main.py").write_text("def f():\n    return 1\n")
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "mod.py").write_text("g = 2\n")

    # Noise that must be pruned.
    cache = pkg / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-311.pyc").write_bytes(b"\x00")
    (cache / "ghost.py").write_text("should_not_deposit = True\n")
    venv = root / ".venv"
    venv.mkdir()
    (venv / "lib.py").write_text("vendored = True\n")
    git = root / ".git"
    git.mkdir()
    (git / "hook.py").write_text("hook = True\n")

    # Non-source file ignored by the suffix filter.
    (root / "README.md").write_text("# readme\n")


class TestDepositProject:
    """``deposit_project`` walks the tree and stamps ``PYTHON_SOURCE``."""

    @pytest.mark.asyncio
    async def test_walks_py_files_keeps_underscore_prunes_noise(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_project
        from particles.corpus.store import CorpusEntryRow

        root = tmp_path / "proj"
        root.mkdir()
        _build_project(root)

        results = await deposit_project(db_session, root, deposited_by="tester")  # type: ignore[arg-type]

        # __init__.py, _shared.py, main.py, sub/mod.py — underscore files kept,
        # __pycache__ / .venv / .git pruned, README.md filtered by suffix.
        assert len(results) == 4

        rows = (
            (await db_session.execute(select(CorpusEntryRow))).scalars().all()  # type: ignore[union-attr]
        )
        from urllib.parse import unquote, urlparse

        names = {Path(unquote(urlparse(r.uri_r).path)).name for r in rows}
        assert names == {"__init__.py", "_shared.py", "main.py", "mod.py"}
        for row in rows:
            assert row.source_type == SourceType.PYTHON_SOURCE.value

    @pytest.mark.asyncio
    async def test_ext_override_changes_glob(self, tmp_path: Path, db_session: object) -> None:
        from particles.corpus.deposit import deposit_project

        root = tmp_path / "proj"
        root.mkdir()
        (root / "a.py").write_text("a = 1\n")
        (root / "b.pyi").write_text("b: int\n")
        (root / "c.txt").write_text("nope")

        # Default config = [".py"] → only a.py.
        default = await deposit_project(db_session, root)  # type: ignore[arg-type]
        assert len(default) == 1

        # Override picks up .pyi too; ".py" passed without a leading dot is
        # normalised. a.py dedups by content hash so the count of *files seen*
        # is what we assert via a fresh tree.
        root2 = tmp_path / "proj2"
        root2.mkdir()
        (root2 / "a.py").write_text("a = 1\n")
        (root2 / "b.pyi").write_text("b: int\n")
        override = await deposit_project(
            db_session,  # type: ignore[arg-type]
            root2,
            extensions={"py", ".pyi"},
        )
        assert len(override) == 2

    @pytest.mark.asyncio
    async def test_empty_tree_returns_empty_list_no_error(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from particles.corpus.deposit import deposit_project

        root = tmp_path / "empty"
        root.mkdir()
        messages: list[str] = []
        results = await deposit_project(
            db_session,  # type: ignore[arg-type]
            root,
            progress=messages.append,
        )
        assert results == []
        assert any("No matching source files" in m for m in messages)

    @pytest.mark.asyncio
    async def test_missing_directory_raises(self, tmp_path: Path, db_session: object) -> None:
        from particles.corpus.deposit import deposit_project

        with pytest.raises(ValueError, match="Project directory not found"):
            await deposit_project(db_session, tmp_path / "nope")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_idempotent_redeposit(self, tmp_path: Path, db_session: object) -> None:
        from particles.corpus.deposit import deposit_project

        root = tmp_path / "proj"
        root.mkdir()
        (root / "stable.py").write_text("x = 1  # unchanging\n")

        first = await deposit_project(db_session, root)  # type: ignore[arg-type]
        await db_session.commit()  # type: ignore[union-attr]
        second = await deposit_project(db_session, root)  # type: ignore[arg-type]
        await db_session.commit()  # type: ignore[union-attr]

        assert len(first) == len(second) == 1
        assert first[0][0] == second[0][0]  # same entry_id — content-hash dedup hit

    @pytest.mark.asyncio
    async def test_progress_callback_invoked_in_order(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from particles.corpus.deposit import deposit_project

        root = tmp_path / "proj"
        root.mkdir()
        (root / "a.py").write_text("a = 1\n")
        (root / "b.py").write_text("b = 1\n")

        messages: list[str] = []
        await deposit_project(
            db_session,  # type: ignore[arg-type]
            root,
            progress=messages.append,
        )
        assert len(messages) == 2
        assert messages[0].startswith("[1/2] depositing")
        assert "a.py" in messages[0]
        assert messages[1].startswith("[2/2] depositing")
        assert "b.py" in messages[1]


# ---------------------------------------------------------------------------
# Single-file deposit_file routing to PYTHON_SOURCE
# ---------------------------------------------------------------------------


class TestDepositFilePythonDetection:
    """``deposit_file`` on a stray ``.py`` stamps PYTHON_SOURCE."""

    @pytest.mark.asyncio
    async def test_py_extension_routes_to_python_source(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_file
        from particles.corpus.store import CorpusEntryRow

        mod = tmp_path / "stray.py"
        mod.write_text('"""A module."""\n')
        entry_id, _ = await deposit_file(db_session, mod)  # type: ignore[arg-type]
        row = (
            await db_session.execute(  # type: ignore[union-attr]
                select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
            )
        ).scalar_one()
        assert row.source_type == SourceType.PYTHON_SOURCE.value


# ---------------------------------------------------------------------------
# _iter_vault_markdown still honours the vault `_`/`.` skip after refactor
# ---------------------------------------------------------------------------


class TestVaultWalkerUnchanged:
    """The vault walker delegates to ``_iter_tree`` but keeps its `_`/`.` rule."""

    def test_vault_skips_underscore_and_dot_components(self, tmp_path: Path) -> None:
        from particles.corpus.deposit import _iter_vault_markdown

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "keep.md").write_text("# keep\n")
        (vault / "_template.md").write_text("# template\n")
        attach = vault / "_attachments"
        attach.mkdir()
        (attach / "scaffold.md").write_text("# scaffold\n")
        dotdir = vault / ".obsidian"
        dotdir.mkdir()
        (dotdir / "settings.md").write_text("# settings\n")

        found = _iter_vault_markdown(vault)
        assert [p.name for p in found] == ["keep.md"]


# ---------------------------------------------------------------------------
# CLI: `particles import project`
# ---------------------------------------------------------------------------


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


class TestImportProjectCli:
    """``particles import project`` end-to-end shape (output + exit code)."""

    def test_help_lists_project_subcommand(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["import", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert "project" in clean

    def test_import_project_deposits_files(self, tmp_path: Path, cli_db: Path) -> None:
        root = tmp_path / "proj"
        root.mkdir()
        (root / "one.py").write_text("one = 1\n")
        (root / "two.py").write_text("two = 2\n")

        runner = CliRunner()
        result = runner.invoke(app, ["import", "project", str(root)], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Deposited 2 file(s)" in result.output

    def test_import_project_progress_logged_when_verbose(
        self, tmp_path: Path, cli_db: Path
    ) -> None:
        root = tmp_path / "proj"
        root.mkdir()
        (root / "mod.py").write_text("m = 1\n")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["import", "project", str(root), "--verbose"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert "[1/1] depositing" in clean

    def test_import_project_nonexistent_dir_exits_nonzero(
        self, tmp_path: Path, cli_db: Path
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["import", "project", str(tmp_path / "no-such-dir")])
        assert result.exit_code != 0

    def test_import_project_routes_to_engine_in_remote_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In remote mode the verb routes each walked source file to the engine."""
        from particles.api.client.base import DepositOutcome
        from particles.api.client.http import HttpBackend
        from particles.config import reset_config
        from particles.core.schema import SourceType

        root = tmp_path / "proj"
        root.mkdir()
        (root / "one.py").write_text("one = 1\n")
        (root / "two.py").write_text("two = 2\n")
        (root / "notes.txt").write_text("ignored\n")  # not a .py ⇒ excluded by the walk

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://127.0.0.1:8099")
        reset_config()

        calls: list[tuple[str, object]] = []

        async def _fake_deposit_file(self: object, path: Path, **kwargs: object) -> DepositOutcome:
            calls.append((path.name, kwargs["source_type"]))
            return DepositOutcome(entry_id=f"e{len(calls)}", snapshot_id="s")

        monkeypatch.setattr(HttpBackend, "deposit_file", _fake_deposit_file)

        runner = CliRunner()
        result = runner.invoke(app, ["import", "project", str(root)], catch_exceptions=False)
        assert result.exit_code == 0, _strip_ansi(result.output)
        assert "Deposited 2 file(s)" in _strip_ansi(result.output)
        assert {n for n, _ in calls} == {"one.py", "two.py"}  # notes.txt excluded
        assert all(st == SourceType.PYTHON_SOURCE for _, st in calls)


# ---------------------------------------------------------------------------
# operations.deposit re-export
# ---------------------------------------------------------------------------


def test_deposit_project_exported_via_operations_shim() -> None:
    """Per particles/api/AGENTS.md the §9.1 surface is the operations shim."""
    from particles.corpus.deposit import deposit_project as corpus_deposit_project
    from particles.operations.deposit import deposit_project as op_deposit_project

    assert op_deposit_project is corpus_deposit_project
