# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Project identity for the Claude Code integration.

Claude Code keys transcripts per working directory but auto-memory per
repository, shared across linked worktrees. These tests pin the resolver that
bridges the two — pure path work, no store, no ``git`` subprocess — and the
conservative rule that a session is only re-keyed when its launch directory
has been positively identified.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.api.cli._claude_code import (
    claude_project_slug,
    launch_directory,
    repository_root,
    resolve_session_project,
    stray_memory_dirs,
)
from tests._claude_projects import make_repo, make_worktree, write_transcript


class TestSlug:
    def test_every_non_alphanumeric_becomes_a_dash(self) -> None:
        assert (
            claude_project_slug("/Users/me/src/my-repo/.claude/worktrees/wt-1")
            == "-Users-me-src-my-repo--claude-worktrees-wt-1"
        )

    def test_accepts_a_path(self) -> None:
        assert claude_project_slug(Path("/a/b.c")) == "-a-b-c"


class TestRepositoryRoot:
    def test_main_checkout_is_its_own_root(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        assert repository_root(repo) == repo

    def test_subdirectory_resolves_to_the_checkout(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        (repo / "pkg" / "a").mkdir(parents=True)
        assert repository_root(repo / "pkg" / "a") == repo

    def test_linked_worktree_resolves_to_the_main_checkout(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        worktree = make_worktree(repo, tmp_path / "elsewhere" / "wt")
        assert repository_root(worktree) == repo

    def test_worktree_with_a_relative_gitdir(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        worktree = make_worktree(repo, tmp_path / "wt")
        (worktree / ".git").write_text("gitdir: ../repo/.git/worktrees/wt\n")
        assert repository_root(worktree) == repo

    def test_submodule_is_its_own_root(self, tmp_path: Path) -> None:
        # A submodule's .git is also a file, but its git dir has no commondir.
        parent = make_repo(tmp_path / "parent")
        module_git = parent / ".git" / "modules" / "sub"
        module_git.mkdir(parents=True)
        sub = parent / "sub"
        sub.mkdir()
        (sub / ".git").write_text(f"gitdir: {module_git}\n")
        assert repository_root(sub) == sub

    def test_bare_repository_worktree_is_its_own_root(self, tmp_path: Path) -> None:
        bare = tmp_path / "repo.git"
        git_dir = bare / "worktrees" / "wt"
        git_dir.mkdir(parents=True)
        (git_dir / "commondir").write_text("../..\n")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {git_dir}\n")
        assert repository_root(worktree) == worktree

    def test_malformed_dot_git_file_is_its_own_root(self, tmp_path: Path) -> None:
        checkout = tmp_path / "odd"
        checkout.mkdir()
        (checkout / ".git").write_text("")
        assert repository_root(checkout) == checkout

    def test_no_git_anywhere_is_the_directory_itself(self, tmp_path: Path) -> None:
        plain = tmp_path / "notes"
        plain.mkdir()
        assert repository_root(plain) == plain

    def test_removed_worktree_under_its_repository_still_resolves(self, tmp_path: Path) -> None:
        # The state a catch-up harvest finds a Claude Code worktree in.
        repo = make_repo(tmp_path / "repo")
        gone = repo / ".claude" / "worktrees" / "gone"
        assert not gone.exists()
        assert repository_root(gone) == repo


class TestLaunchDirectory:
    def test_picks_the_candidate_that_reproduces_the_directory_name(self) -> None:
        assert launch_directory("-a-repo", ["/somewhere/else", "/a/repo"]) == Path("/a/repo")

    def test_no_candidate_matches(self) -> None:
        # Also the shape of a platform slug-rule change: nothing reproduces the
        # name, so nothing is re-keyed.
        assert launch_directory("a_repo_v2", ["/a/repo"]) is None


class TestResolveSessionProject:
    def test_main_checkout_session(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        projects = tmp_path / "home" / ".claude" / "projects"
        transcript = write_transcript(projects / claude_project_slug(repo) / "s.jsonl")

        resolved = resolve_session_project({"transcript_path": str(transcript), "cwd": str(repo)})

        assert resolved.key == claude_project_slug(repo)
        assert resolved.memory_dir == projects / claude_project_slug(repo) / "memory"
        assert resolved.resolved_from == "repository"

    def test_worktree_session_is_keyed_on_its_repository(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        worktree = make_worktree(repo, repo / ".claude" / "worktrees" / "wt")
        projects = tmp_path / "home" / ".claude" / "projects"
        transcript = write_transcript(projects / claude_project_slug(worktree) / "s.jsonl")

        resolved = resolve_session_project(
            {"transcript_path": str(transcript), "cwd": str(worktree)}
        )

        assert resolved.key == claude_project_slug(repo)
        assert resolved.memory_dir == projects / claude_project_slug(repo) / "memory"

    def test_session_that_changed_directory_uses_its_launch_directory(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        other = make_repo(tmp_path / "other")
        projects = tmp_path / "home" / ".claude" / "projects"
        transcript = write_transcript(
            projects / claude_project_slug(repo) / "s.jsonl", str(other), str(repo)
        )

        # The payload reports where the session ended up, not where it started.
        resolved = resolve_session_project({"transcript_path": str(transcript), "cwd": str(other)})

        assert resolved.key == claude_project_slug(repo)

    def test_launch_directory_found_in_the_transcript_when_the_payload_has_none(
        self, tmp_path: Path
    ) -> None:
        repo = make_repo(tmp_path / "repo")
        worktree = make_worktree(repo, tmp_path / "wt")
        projects = tmp_path / "home" / ".claude" / "projects"
        transcript = write_transcript(
            projects / claude_project_slug(worktree) / "s.jsonl", str(worktree)
        )

        resolved = resolve_session_project({"transcript_path": str(transcript)})

        assert resolved.key == claude_project_slug(repo)

    def test_unidentified_launch_directory_keeps_the_transcript_directory(
        self, tmp_path: Path
    ) -> None:
        projects = tmp_path / "home" / ".claude" / "projects"
        transcript = write_transcript(projects / "-my-project" / "s.jsonl", "/unrelated")

        resolved = resolve_session_project(
            {"transcript_path": str(transcript), "cwd": "/some/project"}
        )

        assert resolved.key == "-my-project"
        assert resolved.memory_dir == projects / "-my-project" / "memory"
        assert resolved.resolved_from == "transcript-dir"

    def test_a_session_that_moved_between_worktrees_is_still_its_repositorys(
        self, tmp_path: Path
    ) -> None:
        """Measured on a real store: 17 transcripts recorded only *other* worktrees."""
        projects = tmp_path / "home" / ".claude" / "projects"
        (projects / "-repo").mkdir(parents=True)
        transcript = write_transcript(
            projects / "-repo--claude-worktrees-one" / "s.jsonl", "/repo/.claude/worktrees/two"
        )

        resolved = resolve_session_project({"transcript_path": str(transcript)})

        assert resolved.key == "-repo"
        assert resolved.memory_dir == projects / "-repo" / "memory"
        assert resolved.resolved_from == "worktree-name"

    def test_a_worktree_name_is_not_adopted_when_its_repository_has_no_project_directory(
        self, tmp_path: Path
    ) -> None:
        projects = tmp_path / "home" / ".claude" / "projects"
        transcript = write_transcript(projects / "-gone--claude-worktrees-one" / "s.jsonl")

        resolved = resolve_session_project({"transcript_path": str(transcript)})

        # The hook must not point at a memory directory that does not exist…
        assert resolved.key == "-gone--claude-worktrees-one"
        assert resolved.resolved_from == "transcript-dir"
        # …but a tag is only a name, so an audited transcript still gets the repository's.
        from particles.api.cli._claude_code import transcript_project_key

        assert transcript_project_key(transcript) == "-gone"

    def test_no_transcript_path(self) -> None:
        resolved = resolve_session_project({"cwd": "/a/repo"})
        assert resolved.key == ""
        assert resolved.memory_dir is None


class TestStrayMemoryDirs:
    def test_only_a_linked_worktrees_memory_directory_is_stray(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        worktree = make_worktree(repo, tmp_path / "wt")
        projects = tmp_path / "projects"
        for checkout in (repo, worktree):
            project_dir = projects / claude_project_slug(checkout)
            write_transcript(project_dir / "s.jsonl", str(checkout))
            (project_dir / "memory").mkdir()

        assert stray_memory_dirs(projects) == [projects / claude_project_slug(worktree) / "memory"]

    def test_a_directory_nothing_vouches_for_is_never_stray(self, tmp_path: Path) -> None:
        projects = tmp_path / "projects"
        (projects / "-no-transcripts" / "memory").mkdir(parents=True)
        write_transcript(projects / "-unmatched" / "s.jsonl", "/somewhere/else")
        (projects / "-unmatched" / "memory").mkdir()

        assert stray_memory_dirs(projects) == []

    def test_pruned_transcripts_fall_back_to_region_only_content(self, tmp_path: Path) -> None:
        from particles.render.markdown import insert_projected_region_at_top

        projects = tmp_path / "projects"
        region_only = insert_projected_region_at_top("", "memory-index", "memory.yaml")
        ours = projects / "-repo--claude-worktrees-pruned" / "memory"
        ours.mkdir(parents=True)
        (ours / "MEMORY.md").write_text(region_only)
        authored = projects / "-repo--claude-worktrees-authored" / "memory"
        authored.mkdir(parents=True)
        (authored / "MEMORY.md").write_text(region_only + "\n- a note the user wrote\n")
        with_topic = projects / "-repo--claude-worktrees-topic" / "memory"
        with_topic.mkdir(parents=True)
        (with_topic / "MEMORY.md").write_text(region_only)
        (with_topic / "topic.md").write_text("# Topic\n")

        # A real project with no memory of its own: same content, but its region is read.
        real = projects / "-Users-me" / "memory"
        real.mkdir(parents=True)
        (real / "MEMORY.md").write_text(region_only)

        assert stray_memory_dirs(projects) == [ours]

    def test_missing_projects_root(self, tmp_path: Path) -> None:
        assert stray_memory_dirs(tmp_path / "absent") == []


class TestRuleFileProjectKey:
    def test_a_repositorys_rule_file_belongs_to_that_project(self, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import rule_file_project_key

        repo = make_repo(tmp_path / "repo")
        (repo / "docs").mkdir()
        assert rule_file_project_key(repo / "docs" / "AGENTS.md") == claude_project_slug(repo)

    def test_a_worktrees_rule_file_belongs_to_the_repository(self, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import rule_file_project_key

        repo = make_repo(tmp_path / "repo")
        worktree = make_worktree(repo, tmp_path / "wt")
        assert rule_file_project_key(worktree / "CLAUDE.md") == claude_project_slug(repo)

    def test_the_user_level_rules_are_global(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli._claude_code import rule_file_project_key

        monkeypatch.setenv("HOME", str(tmp_path))
        make_repo(tmp_path)  # even when the home directory is itself a repository
        assert rule_file_project_key(tmp_path / ".claude" / "CLAUDE.md") is None

    def test_a_file_outside_any_repository_is_a_hand_deposit(self, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import rule_file_project_key

        assert rule_file_project_key(tmp_path / "loose" / "AGENTS.md") is None


def test_transcript_project_key_matches_what_the_hook_stamps(tmp_path: Path) -> None:
    from particles.api.cli._claude_code import transcript_project_key

    repo = make_repo(tmp_path / "repo")
    worktree = make_worktree(repo, tmp_path / "wt")
    projects = tmp_path / "home" / ".claude" / "projects"
    transcript = write_transcript(
        projects / claude_project_slug(worktree) / "s.jsonl", str(worktree)
    )
    assert transcript_project_key(transcript) == claude_project_slug(repo)
