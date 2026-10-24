# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""`particles memory rescope` / `widen` — the adapter's key resolver and the verbs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli._claude_code import (
    canonical_project_key,
    claude_project_slug,
    entry_project_key,
    is_live_project_key,
)
from tests._claude_projects import make_repo, make_worktree, write_transcript


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _projects(home: Path) -> Path:
    return home / ".claude" / "projects"


class TestEntryProjectKey:
    def test_a_worktrees_slug_stands_for_its_repository(self, home: Path) -> None:
        repo = make_repo(home / "src" / "repo")
        worktree = make_worktree(repo, home / "elsewhere" / "wt")
        slug = claude_project_slug(worktree)
        write_transcript(_projects(home) / slug / "s.jsonl", str(worktree))

        assert canonical_project_key(slug, _projects(home)) == claude_project_slug(repo)

    def test_pruned_transcripts_fall_back_to_the_worktree_name(self, home: Path) -> None:
        assert canonical_project_key("-repo--claude-worktrees-x", _projects(home)) == "-repo"
        assert canonical_project_key("-repo", _projects(home)) == "-repo"

    def test_a_memory_file_is_attributed_by_the_directory_in_its_uri(self, home: Path) -> None:
        uri = f"file://{_projects(home)}/-repo--claude-worktrees-x/memory/MEMORY.md"
        assert entry_project_key(uri, ["claude-code", "memory-file"], _projects(home)) == "-repo"

    def test_a_transcript_is_attributed_by_where_its_session_was_launched(self, home: Path) -> None:
        repo = make_repo(home / "src" / "repo")
        write_transcript(_projects(home) / claude_project_slug(repo) / "sess-1.jsonl", str(repo))

        key = entry_project_key(
            "claude-code://session/sess-1", ["claude-code", "audit"], _projects(home)
        )

        assert key == claude_project_slug(repo)

    def test_a_transcript_whose_session_file_is_gone_stays_unattributed(self, home: Path) -> None:
        key = entry_project_key("claude-code://session/gone", ["claude-code"], _projects(home))
        assert key is None

    def test_a_legacy_tag_is_canonicalised_when_nothing_else_is_known(self, home: Path) -> None:
        tags = ["claude-code", "project:-repo--claude-worktrees-x"]
        assert entry_project_key("claude-code://session/gone", tags, _projects(home)) == "-repo"

    def test_a_rule_file_is_attributed_by_its_repository(self, home: Path) -> None:
        repo = make_repo(home / "src" / "repo")
        key = entry_project_key(f"file://{repo}/AGENTS.md", ["rule-file"], _projects(home))
        assert key == claude_project_slug(repo)

    def test_a_hand_deposit_is_left_alone(self, home: Path) -> None:
        assert entry_project_key("https://example.org/", ["web"], _projects(home)) is None

    def test_a_key_is_live_only_if_its_directory_exists_and_is_canonical(self, home: Path) -> None:
        (_projects(home) / "-repo").mkdir(parents=True)
        (_projects(home) / "-repo--claude-worktrees-x").mkdir()
        assert is_live_project_key("-repo", _projects(home))
        assert not is_live_project_key("-repo--claude-worktrees-x", _projects(home))
        assert not is_live_project_key("-deleted", _projects(home))


async def _deposit(uri: str, tags: list[str]) -> str:
    from particles.core.schema import Mutability
    from particles.corpus.deposit import deposit_text_versioned
    from particles.db import session_scope

    async with session_scope(write=True) as session:
        entry_id, _, _ = await deposit_text_versioned(
            session,
            text=f"text of {uri}",
            uri_r=uri,
            source_type="CONVERSATION",
            mutability=Mutability.APPEND_ONLY,
            tags=tags,
        )
        await session.commit()
    return entry_id


async def _tags(entry_id: str) -> list[str]:
    from particles.corpus.store import get_entry
    from particles.db import session_scope

    async with session_scope() as session:
        entry = await get_entry(session, entry_id)
    assert entry is not None
    return entry.tags


async def _event_types() -> list[str]:
    from particles.db import session_scope
    from particles.store.event_store import list_events

    async with session_scope() as session:
        return [e.event_type.value for e in await list_events(session, limit=50)]


class TestRescopeVerb:
    def test_dry_run_then_real_run_then_a_no_op(
        self, runner: CliRunner, cli_db: Path, home: Path
    ) -> None:
        (_projects(home) / "-repo").mkdir(parents=True)
        legacy = asyncio.run(
            _deposit(
                "claude-code://session/a", ["claude-code", "project:-repo--claude-worktrees-x"]
            )
        )
        unknown = asyncio.run(_deposit("claude-code://session/b", ["claude-code", "audit"]))

        dry = runner.invoke(app, ["memory", "rescope", "--dry-run"])
        assert dry.exit_code == 0, dry.output
        assert "would add 1 project key" in dry.output
        assert "Dry run: nothing was written." in dry.output
        assert asyncio.run(_tags(legacy)) == ["claude-code", "project:-repo--claude-worktrees-x"]
        assert "OBSERVER_SCOPE_RESCOPED" not in asyncio.run(_event_types())

        real = runner.invoke(app, ["memory", "rescope"])
        assert real.exit_code == 0, real.output
        assert "added 1 project key" in real.output
        assert "1 harvested entry is unattributed" in real.output and unknown in real.output
        assert asyncio.run(_tags(legacy))[-1] == "project:-repo"
        assert "OBSERVER_SCOPE_RESCOPED" in asyncio.run(_event_types())

        again = runner.invoke(app, ["memory", "rescope", "--default-key", "-repo"])
        assert "added 1 project key" in again.output and "unattributed" not in again.output
        assert asyncio.run(_tags(unknown)) == ["claude-code", "audit", "project:-repo"]
        assert "added 0 project key" in runner.invoke(app, ["memory", "rescope"]).output

    def test_entries_of_a_project_that_no_longer_exists_are_reported(
        self, runner: CliRunner, cli_db: Path, home: Path
    ) -> None:
        orphan = asyncio.run(
            _deposit("claude-code://session/o", ["claude-code", "project:-deleted"])
        )

        result = runner.invoke(app, ["memory", "rescope"])

        assert "carry only keys of projects that no longer exist" in result.output
        assert orphan in result.output

    def test_assign_refuses_a_global_entry(
        self, runner: CliRunner, cli_db: Path, home: Path
    ) -> None:
        page = asyncio.run(_deposit("https://example.org/", ["web"]))

        result = runner.invoke(app, ["memory", "rescope", "--assign", page, "-repo"])

        assert result.exit_code == 2
        assert "is global" in result.output
        assert asyncio.run(_tags(page)) == ["web"]


async def _agent_deposit(text: str, who: str) -> str:
    from particles.corpus.deposit import deposit_text
    from particles.db import session_scope

    async with session_scope(write=True) as session:
        entry_id, _ = await deposit_text(session, text, who, "CONVERSATION", [], author_id=who)
        await session.commit()
    return entry_id


class TestMarkAgentWrites:
    def test_preview_then_mark_then_unmark(
        self, runner: CliRunner, cli_db: Path, home: Path
    ) -> None:
        agent = asyncio.run(_agent_deposit("an agent's note", "mcp:claude-code"))
        asyncio.run(_agent_deposit("a renamed agent's note", "mcp:old-name"))

        dry = runner.invoke(app, ["memory", "rescope", "--mark-agent-writes", "--dry-run"])
        assert dry.exit_code == 0, dry.output
        assert "would mark 1 keyless entry" in dry.output
        assert "by mcp:old-name (not matched)" in dry.output
        assert "--identity NAME" in dry.output
        assert asyncio.run(_tags(agent)) == []

        real = runner.invoke(
            app, ["memory", "rescope", "--mark-agent-writes", "--identity", "mcp:old-name"]
        )
        assert real.exit_code == 0, real.output
        assert "marked 2 keyless entries" in real.output
        assert asyncio.run(_tags(agent)) == ["scope:unattributed"]

        undone = runner.invoke(app, ["memory", "rescope", "--unmark", agent])
        assert undone.exit_code == 0, undone.output
        assert asyncio.run(_tags(agent)) == []

    def test_the_marking_flags_are_separate_steps(
        self, runner: CliRunner, cli_db: Path, home: Path
    ) -> None:
        both = runner.invoke(app, ["memory", "rescope", "--mark-agent-writes", "--unmark", "e"])
        assert both.exit_code == 2 and "separate steps" in both.output
        stray = runner.invoke(app, ["memory", "rescope", "--identity", "x"])
        assert stray.exit_code == 2 and "only with --mark-agent-writes" in stray.output


class TestWithdrawVerb:
    def test_the_last_key_is_refused_then_left_unattributed(
        self, runner: CliRunner, cli_db: Path, home: Path
    ) -> None:
        entry = asyncio.run(_deposit("claude-code://session/w", ["claude-code", "project:-a"]))

        refused = runner.invoke(app, ["memory", "rescope", "--withdraw", entry, "-a"])
        assert refused.exit_code == 2 and "last project key" in refused.output
        assert asyncio.run(_tags(entry)) == ["claude-code", "project:-a"]

        done = runner.invoke(
            app, ["memory", "rescope", "--withdraw", entry, "-a", "--leave-unattributed"]
        )
        assert done.exit_code == 0, done.output
        assert asyncio.run(_tags(entry)) == ["claude-code"]

    def test_leave_unattributed_needs_withdraw(
        self, runner: CliRunner, cli_db: Path, home: Path
    ) -> None:
        result = runner.invoke(app, ["memory", "rescope", "--leave-unattributed"])
        assert result.exit_code == 2 and "only with --withdraw" in result.output


class TestWidenVerb:
    def test_widen_and_revoke_a_source(self, runner: CliRunner, cli_db: Path, home: Path) -> None:
        entry_id = asyncio.run(_deposit("claude-code://session/w", ["claude-code", "project:-b"]))

        widened = runner.invoke(app, ["memory", "widen", "--entry", entry_id])
        assert widened.exit_code == 0, widened.output
        assert "now in view for every project" in widened.output
        assert (
            "already widened" in runner.invoke(app, ["memory", "widen", "--entry", entry_id]).output
        )

        revoked = runner.invoke(app, ["memory", "widen", "--entry", entry_id, "--revoke"])
        assert "no longer widened" in revoked.output
        events: list[Any] = asyncio.run(_event_types())
        assert {"OBSERVER_SCOPE_WIDENED", "OBSERVER_SCOPE_WIDEN_REVOKED"} <= set(events)

    def test_an_unknown_entry_is_an_error(
        self, runner: CliRunner, cli_db: Path, home: Path
    ) -> None:
        result = runner.invoke(app, ["memory", "widen", "--entry", "nope"])
        assert result.exit_code == 2 and "No corpus entry" in result.output
