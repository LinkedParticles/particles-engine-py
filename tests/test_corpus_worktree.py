# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The verified main checkout of a git worktree (security review K6, P12).

A ``.git`` file's ``gitdir:`` line is believed only when the worktree
directory it names exists where git puts it and links back to the file.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from particles.corpus.worktree import verified_main_checkout


def _fake_worktree(main: Path, worktree: Path, name: str = "wt") -> Path:
    gitdir = main / ".git" / "worktrees" / name
    gitdir.mkdir(parents=True)
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n")
    (gitdir / "gitdir").write_text(f"{worktree / '.git'}\n")
    (gitdir / "commondir").write_text("../..\n")
    return gitdir


def test_a_consistent_worktree_resolves_to_its_main_checkout(tmp_path: Path) -> None:
    main = (tmp_path / "main").resolve()
    worktree = (tmp_path / "wt").resolve()
    _fake_worktree(main, worktree)
    assert verified_main_checkout(worktree) == main


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_real_git_worktree_resolves_to_its_main_checkout(tmp_path: Path) -> None:
    """What git itself writes passes the check, relative or absolute."""
    main = (tmp_path / "main").resolve()
    main.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(main), *args], check=True, capture_output=True)

    git("init", "-q")
    git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.org",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "x",
    )
    git("worktree", "add", "-q", str(tmp_path / "linked"))
    assert verified_main_checkout(tmp_path / "linked") == main
    assert verified_main_checkout(main) is None


def test_a_gitdir_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    (victim / ".git").mkdir(parents=True)
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / ".git").write_text(f"gitdir: {victim / '.git' / 'worktrees' / 'x'}\n")
    assert verified_main_checkout(planted) is None


def test_a_worktree_that_does_not_link_back_is_refused(tmp_path: Path) -> None:
    """Another checkout's real worktree cannot be claimed by a planted ``.git`` file."""
    main = (tmp_path / "main").resolve()
    _fake_worktree(main, (tmp_path / "theirs").resolve())
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    assert verified_main_checkout(planted) is None


def test_a_wrong_commondir_is_refused(tmp_path: Path) -> None:
    main = (tmp_path / "main").resolve()
    worktree = (tmp_path / "wt").resolve()
    gitdir = _fake_worktree(main, worktree)
    (gitdir / "commondir").write_text("/\n")
    assert verified_main_checkout(worktree) is None


@pytest.mark.parametrize("target", ["/.git/worktrees/x", "{file}/.git/worktrees/x"])
def test_a_gitdir_naming_the_filesystem_root_or_a_file_is_refused(
    tmp_path: Path, target: str
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / ".git").write_text(f"gitdir: {target.format(file=secret)}\n")
    assert verified_main_checkout(planted) is None


def test_a_symlinked_dot_git_file_is_refused(tmp_path: Path) -> None:
    main = (tmp_path / "main").resolve()
    worktree = (tmp_path / "wt").resolve()
    _fake_worktree(main, worktree)
    other = tmp_path / "other"
    other.mkdir()
    (other / ".git").symlink_to(worktree / ".git")
    assert verified_main_checkout(other) is None


def test_a_submodule_or_unreadable_file_is_not_a_worktree(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".git").write_text("gitdir: ../.git/modules/sub\n")
    assert verified_main_checkout(sub) is None
    (sub / ".git").write_bytes(b"")
    assert verified_main_checkout(sub) is None
    assert verified_main_checkout(tmp_path / "missing") is None
