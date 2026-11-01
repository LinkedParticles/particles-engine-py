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


class TestPlantedGitFile:
    """Security review 2026-10-24-2, P22: a ``.git`` file is a claim, checked against git."""

    def _plant(self, tmp_path: Path) -> tuple[Path, Path]:
        # The audit's proof layout: a pointer directory the planter owns whose
        # commondir names another repository.
        victim = make_repo(tmp_path / "victim")
        planted = tmp_path / "unpacked"
        pointer = planted / "meta"
        pointer.mkdir(parents=True)
        (planted / ".git").write_text("gitdir: meta\n", encoding="utf-8")
        (pointer / "commondir").write_text("../../victim/.git\n", encoding="utf-8")
        return victim, planted

    def test_planted_gitfile_does_not_resolve_to_another_repository(self, tmp_path: Path) -> None:
        victim, planted = self._plant(tmp_path)
        assert repository_root(planted) == planted != victim

    def test_planted_gitfile_does_not_bind_the_server_to_another_project(
        self, tmp_path: Path
    ) -> None:
        from particles.api.cli._claude_code import server_project_key

        victim, planted = self._plant(tmp_path)
        assert server_project_key(planted) != claude_project_slug(victim)

    def test_a_worktree_entry_without_its_back_pointer_is_refused(self, tmp_path: Path) -> None:
        # The gitdir names a real worktree entry of the victim, but that entry's
        # back-pointer names the victim's own worktree, not this checkout.
        victim = make_repo(tmp_path / "victim")
        make_worktree(victim, tmp_path / "real-wt")
        planted = tmp_path / "planted"
        planted.mkdir()
        (planted / ".git").write_text(f"gitdir: {victim / '.git' / 'worktrees' / 'wt'}\n")
        assert repository_root(planted) == planted

    def test_a_gitdir_outside_the_worktrees_directory_is_refused(self, tmp_path: Path) -> None:
        victim = make_repo(tmp_path / "victim")
        git_dir = tmp_path / "fake" / "worktrees" / "wt"
        git_dir.mkdir(parents=True)
        (git_dir / "commondir").write_text(f"{victim / '.git'}\n")
        planted = tmp_path / "planted"
        planted.mkdir()
        (planted / ".git").write_text(f"gitdir: {git_dir}\n")
        (git_dir / "gitdir").write_text(f"{planted / '.git'}\n")
        assert repository_root(planted) == planted

    def test_a_relative_back_pointer_is_accepted(self, tmp_path: Path) -> None:
        # ``worktree.useRelativePaths`` writes the back-pointer relative to the entry.
        repo = make_repo(tmp_path / "repo")
        worktree = make_worktree(repo, tmp_path / "wt")
        (repo / ".git" / "worktrees" / "wt" / "gitdir").write_text("../../../../wt/.git\n")
        assert repository_root(worktree) == repo

    def test_a_real_git_worktree_resolves(self, tmp_path: Path) -> None:
        import os
        import shutil
        import subprocess

        if shutil.which("git") is None:
            pytest.skip("needs git")
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        repo = tmp_path / "repo"
        repo.mkdir()

        def git(*args: str) -> None:
            subprocess.run(
                ["git", "-C", str(repo), *args], check=True, capture_output=True, env=env
            )

        git("init", "-q")
        git(
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.test",
            "commit",
            "-q",
            "--allow-empty",
            "--no-gpg-sign",
            "-m",
            "init",
        )
        git("worktree", "add", "-q", str(tmp_path / "wt"))
        assert repository_root(tmp_path / "wt") == repo


class TestLinkedGitFile:
    """Security review 2026-10-25-1, Q26: a ``.git`` file that is a link is not believed.

    The audit's proofs: a directory whose ``.git`` is a symbolic or hard link to
    a live worktree's ``.git`` file passed every layout check, since the
    back-pointer names that same file.
    """

    def test_a_symlinked_dot_git_file_does_not_adopt_another_repository(
        self, tmp_path: Path
    ) -> None:
        victim = make_repo(tmp_path / "victim")
        worktree = make_worktree(victim, tmp_path / "victim-wt")
        planted = tmp_path / "planted-symlink"
        planted.mkdir()
        (planted / ".git").symlink_to(worktree / ".git")
        assert repository_root(planted) == planted

    def test_a_hardlinked_dot_git_file_does_not_adopt_another_repository(
        self, tmp_path: Path
    ) -> None:
        import os

        victim = make_repo(tmp_path / "victim")
        worktree = make_worktree(victim, tmp_path / "victim-wt")
        planted = tmp_path / "planted-hardlink"
        planted.mkdir()
        os.link(worktree / ".git", planted / ".git")
        assert repository_root(planted) == planted
        # Nor does the worktree itself while its .git has a second name: the
        # two entries cannot be told apart, so neither is believed.
        assert repository_root(worktree) == worktree

    def test_a_linked_dot_git_file_does_not_bind_the_server(self, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import server_project_key

        victim = make_repo(tmp_path / "victim")
        worktree = make_worktree(victim, tmp_path / "victim-wt")
        planted = tmp_path / "planted"
        planted.mkdir()
        (planted / ".git").symlink_to(worktree / ".git")
        assert server_project_key(planted) != claude_project_slug(victim)

    def test_the_verified_reading_refuses_a_linked_dot_git_file(self, tmp_path: Path) -> None:
        import os

        from particles.corpus.worktree import verified_main_checkout

        victim = make_repo(tmp_path / "victim")
        worktree = make_worktree(victim, tmp_path / "victim-wt")
        assert verified_main_checkout(worktree) == victim.resolve()
        planted = tmp_path / "planted"
        planted.mkdir()
        os.link(worktree / ".git", planted / ".git")
        assert verified_main_checkout(planted) is None
        assert verified_main_checkout(worktree) is None

    def test_a_worktree_reached_through_a_symlinked_ancestor_still_resolves(
        self, tmp_path: Path
    ) -> None:
        repo = make_repo(tmp_path / "repo")
        make_worktree(repo, tmp_path / "real" / "wt")
        (tmp_path / "alias").symlink_to(tmp_path / "real")
        assert repository_root(tmp_path / "alias" / "wt") == repo


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
        assert resolved.root == repo  # the path the key came from

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


class TestServerProjectKey:
    """What a bound MCP server keys on."""

    def test_a_repository_and_its_linked_worktree_share_the_key(self, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import claude_project_slug, server_project_key

        repo = make_repo(tmp_path / "src" / "repo")
        worktree = make_worktree(repo, tmp_path / "elsewhere" / "wt")

        assert server_project_key(repo) == claude_project_slug(repo)
        assert server_project_key(worktree) == claude_project_slug(repo)

    def test_a_directory_outside_any_repository_is_its_own_project(self, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import claude_project_slug, server_project_key

        home = tmp_path / "home"
        home.mkdir()
        assert server_project_key(home) == claude_project_slug(home)

    def test_the_filesystem_root_is_no_project(self) -> None:
        from particles.api.cli._claude_code import server_project_key

        assert server_project_key(Path("/")) is None


class TestObserverBindingLines:
    """`hook doctor` shows where a digest's key and a server's can part."""

    def test_a_fallback_and_a_mismatch_are_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        from particles.api.cli._claude_code import append_hook_log, claude_project_slug
        from particles.api.cli.hook import _observer_binding_lines
        from particles.config import get_config

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        get_config().claude_code.state_dir = str(tmp_path / "state")
        here = make_repo(tmp_path / "src" / "repo")
        (home / ".claude.json").write_text(
            json.dumps(
                {"mcpServers": {"particles": {"command": "/p/particles", "args": ["mcp", "serve"]}}}
            )
        )
        append_hook_log(
            {
                "event": "session-start",
                "cwd": str(here),
                "project_key": "-moved-elsewhere",
                "project_resolved_from": "transcript-dir",
            }
        )

        lines = "\n".join(_observer_binding_lines(here))

        assert "MCP server is not bound" in lines
        assert f"a bound server takes {claude_project_slug(here)}" in lines
        assert "the hooks last resolved -moved-elsewhere" in lines
        assert "read different projects here" in lines
        assert "-moved-elsewhere (transcript-dir)" in lines

    def test_agreement_raises_no_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        from particles.api.cli._claude_code import append_hook_log, claude_project_slug
        from particles.api.cli.hook import _observer_binding_lines
        from particles.config import get_config

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        get_config().claude_code.state_dir = str(tmp_path / "state")
        here = make_repo(tmp_path / "src" / "repo")
        (home / ".claude.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "particles": {
                            "command": "/p/particles",
                            "args": ["mcp", "serve", "--project-observer", "cwd"],
                        }
                    }
                }
            )
        )
        append_hook_log(
            {
                "event": "session-start",
                "cwd": str(here),
                "project_key": claude_project_slug(here),
                "project_resolved_from": "repository",
            }
        )

        lines = _observer_binding_lines(here)

        assert not any("⚠" in line for line in lines)


class TestProjectKeyCollisions:
    """a key two repositories share is recorded and disclosed, never re-keyed."""

    @pytest.fixture
    def state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from particles.config import get_config

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        state = tmp_path / "state"
        get_config().claude_code.state_dir = str(state)
        return state

    def test_two_punctuation_variants_share_one_slug(self, tmp_path: Path) -> None:
        assert claude_project_slug(tmp_path / "a-b") == claude_project_slug(tmp_path / "a" / "b")

    def test_roots_are_a_set_union(self, state: Path, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import record_project_root, recorded_project_roots

        one, two = tmp_path / "a-b", tmp_path / "a" / "b"
        key = claude_project_slug(one)
        assert record_project_root(key, one) == [str(one)]
        assert record_project_root(key, one) == [str(one)]
        assert record_project_root(key, two) == sorted([str(one), str(two)])
        assert record_project_root(key, one) == sorted([str(one), str(two)])
        assert recorded_project_roots(key) == sorted([str(one), str(two)])

    def test_the_record_is_owner_only(self, state: Path, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import project_roots_path, record_project_root

        key = claude_project_slug(tmp_path / "repo")
        record_project_root(key, tmp_path / "repo")
        assert project_roots_path(key).stat().st_mode & 0o777 == 0o600

    @pytest.mark.parametrize("key", ["", "..", "../escape", "a/b"])
    def test_a_key_no_slug_can_be_is_never_written(
        self, state: Path, tmp_path: Path, key: str
    ) -> None:
        from particles.api.cli._claude_code import record_project_root, recorded_project_roots

        assert record_project_root(key, tmp_path / "repo") == []
        assert recorded_project_roots(key) == []
        assert not state.exists()

    def test_an_unreadable_record_reads_as_none(self, state: Path) -> None:
        from particles.api.cli._claude_code import project_roots_path, recorded_project_roots

        path = project_roots_path("-x")
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        assert recorded_project_roots("-x") == []

    def test_the_doctor_discloses_a_shared_key(self, state: Path, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import record_project_root
        from particles.api.cli.hook import _observer_binding_lines

        here = make_repo(tmp_path / "src" / "a-b")
        other = make_repo(tmp_path / "src" / "a" / "b")
        key = claude_project_slug(here)
        record_project_root(key, here)
        record_project_root(key, other)

        lines = "\n".join(_observer_binding_lines(here))

        assert f"project key {key} is shared by 2 repositories" in lines
        assert str(here) in lines and str(other) in lines

    def test_one_root_raises_no_warning(self, state: Path, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import record_project_root
        from particles.api.cli.hook import _observer_binding_lines

        here = make_repo(tmp_path / "src" / "repo")
        record_project_root(claude_project_slug(here), here)

        assert "is shared by" not in "\n".join(_observer_binding_lines(here))


class TestSessionTranscriptLookup:
    """Security review 2026-10-25-2, R28: a session URI names only a transcript under the root."""

    def _planted(self, tmp_path: Path) -> Path:
        import json

        projects = tmp_path / "projects"
        (projects / "proj").mkdir(parents=True)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "planted.jsonl").write_text(
            json.dumps({"timestamp": "2099-01-01T00:00:00Z", "cwd": str(outside)}) + "\n",
            encoding="utf-8",
        )
        return projects

    @pytest.mark.parametrize(
        "session_id", ["../../elsewhere/planted", "../elsewhere/planted", "*", "pl?nted", ""]
    )
    def test_an_id_that_is_not_a_session_id_names_no_transcript(
        self, tmp_path: Path, session_id: str
    ) -> None:
        from particles.api.cli._claude_code import (
            entry_project_key,
            session_started_at,
            session_transcripts,
        )

        projects = self._planted(tmp_path)
        uri = f"claude-code://session/{session_id}"
        assert session_transcripts(uri, projects) == []
        assert session_started_at(uri, projects) is None
        assert entry_project_key(uri, ["claude-code"], projects) is None

    def test_a_symlinked_transcript_outside_the_root_is_not_read(self, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import session_started_at, session_transcripts

        projects = self._planted(tmp_path)
        (projects / "proj" / "abc123.jsonl").symlink_to(tmp_path / "elsewhere" / "planted.jsonl")
        uri = "claude-code://session/abc123"
        assert session_transcripts(uri, projects) == []
        assert session_started_at(uri, projects) is None

    def test_a_session_id_finds_its_transcript(self, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import session_started_at, session_transcripts

        projects = self._planted(tmp_path)
        transcript = projects / "proj" / "3f2a9c1e-0b7d-4e8a-9c2f-1d5e6a7b8c9d.jsonl"
        transcript.write_text('{"timestamp": "2026-10-09T05:12:00Z"}\n', encoding="utf-8")
        uri = f"claude-code://session/{transcript.stem}"
        assert session_transcripts(uri, projects) == [transcript]
        started = session_started_at(uri, projects)
        assert started is not None and started.year == 2026

    def test_flat_finds_a_transcript_in_the_root_itself(self, tmp_path: Path) -> None:
        # A folder of transcripts copied out of the projects layout, the shape
        # `audit --transcripts DIR` reads; only `flat` looks there.
        from particles.api.cli._claude_code import session_started_at, session_transcripts

        projects = self._planted(tmp_path)
        transcript = projects / "abc123.jsonl"
        transcript.write_text('{"timestamp": "2026-08-22T09:15:00Z"}\n', encoding="utf-8")
        uri = "claude-code://session/abc123"
        assert session_transcripts(uri, projects) == []
        assert session_transcripts(uri, projects, flat=True) == [transcript]
        started = session_started_at(uri, projects, flat=True)
        assert started is not None and started.date().isoformat() == "2026-08-22"

    def test_flat_does_not_follow_a_link_out_of_the_root(self, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import session_started_at, session_transcripts

        projects = self._planted(tmp_path)
        (projects / "abc123.jsonl").symlink_to(tmp_path / "elsewhere" / "planted.jsonl")
        uri = "claude-code://session/abc123"
        assert session_transcripts(uri, projects, flat=True) == []
        assert session_started_at(uri, projects, flat=True) is None

    @pytest.mark.parametrize("session_id", ["../elsewhere/planted", "*", ""])
    def test_flat_keeps_the_session_id_shape_check(self, tmp_path: Path, session_id: str) -> None:
        from particles.api.cli._claude_code import session_transcripts

        projects = self._planted(tmp_path)
        uri = f"claude-code://session/{session_id}"
        assert session_transcripts(uri, projects, flat=True) == []
