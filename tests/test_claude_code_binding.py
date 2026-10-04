# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the harness binding in particles/api/cli/_claude_code.py.

A Claude Code memory directory is harvested and rendered only by the store the
installed Particles hooks name. These tests pin the pure half: how an installed
hook command reduces to a store, how two database URLs compare, and the refusal
reasons. The end-to-end cases (hook, consolidation, the cycle itself) are in
``tests/test_memory_projection.py`` § Store binding.
"""

from __future__ import annotations

import os
import pwd
from pathlib import Path

import pytest

from particles.api.cli._claude_code import (
    parse_hook_binding,
    projection_refusal,
    store_location,
)
from tests._claude_projects import bind_hooks_to_store

_SETTINGS = Path("/home/u/.claude/settings.json")


class TestParseHookBinding:
    def test_default_store_is_pinned_by_database_url(self, tmp_path: Path) -> None:
        db = tmp_path / "particles.db"
        dsn = f"sqlite+aiosqlite:///{db}"
        command = f"env DATABASE_URL={dsn} /x/particles hook session-end --store default"

        binding = parse_hook_binding(command, _SETTINGS)

        assert binding.handle == "default"
        assert binding.location == store_location(dsn)

    def test_named_store_is_pinned_by_the_pinned_config(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(f"storage:\n  stores:\n    memory: sqlite+aiosqlite:///{tmp_path}/m.db\n")
        command = f"env PARTICLES_CONFIG={config} /x/particles hook session-end --store memory"

        binding = parse_hook_binding(command, _SETTINGS)

        assert binding.handle == "memory"
        assert binding.location == store_location(f"sqlite+aiosqlite:///{tmp_path}/m.db")

    def test_a_quoted_path_with_spaces_survives(self, tmp_path: Path) -> None:
        db = tmp_path / "my store" / "p.db"
        dsn = f"'sqlite+aiosqlite:///{db}'"
        command = f"env DATABASE_URL={dsn} particles hook session-start --store default"

        assert parse_hook_binding(command, _SETTINGS).location == f"sqlite:{os.path.realpath(db)}"

    @pytest.mark.parametrize(
        "command",
        [
            "/x/particles hook session-end --store default",  # a pre-v1.70.2 install
            "env PARTICLES_CONFIG=/nonexistent.yaml particles hook session-end --store memory",
        ],
    )
    def test_an_unpinned_command_names_no_store(self, command: str) -> None:
        assert parse_hook_binding(command, _SETTINGS).location is None

    def test_a_relative_dsn_in_the_pinned_config_names_no_store(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("storage:\n  database_url: sqlite+aiosqlite:///./particles.db\n")
        command = f"env PARTICLES_CONFIG={config} particles hook session-end --store default"

        assert parse_hook_binding(command, _SETTINGS).location is None


class TestStoreLocation:
    def test_relative_and_absolute_forms_of_one_file_compare_equal(self, tmp_path: Path) -> None:
        relative = store_location("sqlite+aiosqlite:///./particles.db", tmp_path)
        absolute = store_location(f"sqlite+aiosqlite:///{tmp_path}/particles.db")
        assert relative == absolute is not None

    def test_a_symlinked_directory_resolves_to_the_same_file(self, tmp_path: Path) -> None:
        (tmp_path / "real").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "real")
        assert store_location(f"sqlite:///{tmp_path}/link/p.db") == store_location(
            f"sqlite+aiosqlite:///{tmp_path}/real/p.db"
        )

    @pytest.mark.parametrize(
        "dsn", ["sqlite+aiosqlite:///:memory:", "sqlite://", "sqlite+aiosqlite:///rel.db"]
    )
    def test_in_memory_or_unanchored_urls_have_no_location(self, dsn: str) -> None:
        assert store_location(dsn) is None

    def test_a_network_url_compares_verbatim(self) -> None:
        assert store_location("postgresql+asyncpg://h/db") == "postgresql+asyncpg://h/db"


class TestProjectionRefusal:
    @pytest.fixture
    def memory_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/scratch.db")
        path = tmp_path / ".claude" / "projects" / "-proj" / "memory"
        path.mkdir(parents=True)
        return path

    def test_the_bound_store_is_served(self, memory_dir: Path, tmp_path: Path) -> None:
        bind_hooks_to_store(tmp_path, f"sqlite+aiosqlite:///{tmp_path}/scratch.db")
        assert projection_refusal("default", memory_dir) is None

    def test_another_store_is_refused(self, memory_dir: Path, tmp_path: Path) -> None:
        bind_hooks_to_store(tmp_path, f"sqlite+aiosqlite:///{tmp_path}/owner.db")
        reason = projection_refusal("default", memory_dir)
        assert reason is not None
        assert "not the store the installed Claude Code hooks name" in reason

    def test_no_installed_hooks_is_refused(self, memory_dir: Path) -> None:
        reason = projection_refusal("default", memory_dir)
        assert reason is not None
        assert "no Particles hook is installed" in reason

    def test_unpinned_hooks_are_refused_with_the_repair(
        self, memory_dir: Path, tmp_path: Path
    ) -> None:
        settings = tmp_path / ".claude" / "settings.json"
        settings.write_text(
            '{"hooks": {"SessionEnd": [{"hooks": [{"type": "command", '
            '"command": "particles hook session-end --store default"}]}]}}'
        )
        reason = projection_refusal("default", memory_dir)
        assert reason is not None
        assert "re-run `particles init claude-code`" in reason

    def test_an_in_memory_store_is_refused(
        self, memory_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bind_hooks_to_store(tmp_path, f"sqlite+aiosqlite:///{tmp_path}/scratch.db")
        monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
        reason = projection_refusal("default", memory_dir)
        assert reason is not None
        assert "no database location" in reason

    def test_a_directory_outside_a_claude_projects_tree_is_not_gated(self, tmp_path: Path) -> None:
        notes = tmp_path / "notes" / "memory"
        notes.mkdir(parents=True)
        assert projection_refusal("default", notes) is None


class TestSuiteGuard:
    """The autouse ``no_real_claude_projects_writes`` guard in tests/conftest.py."""

    def test_a_write_under_the_real_projects_root_is_refused(self) -> None:
        from tests import conftest

        target = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".claude" / "projects" / "-no-such" / "x"
        try:
            with pytest.raises(PermissionError):
                target.write_text("refused before the open")
            assert conftest._claude_projects_writes == [f"open {target}"]
        finally:
            # This test wrote on purpose; keep the teardown check from failing it.
            if conftest._claude_projects_writes is not None:
                conftest._claude_projects_writes.clear()
        assert not target.exists()

    def test_a_fake_home_is_not_guarded(self, tmp_path: Path) -> None:
        target = tmp_path / ".claude" / "projects" / "-p" / "memory" / "MEMORY.md"
        target.parent.mkdir(parents=True)
        target.write_text("fine\n")
        assert target.read_text() == "fine\n"
