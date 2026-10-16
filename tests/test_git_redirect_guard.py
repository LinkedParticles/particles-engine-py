# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The suite's guard against git hook variables (``tests/_client_fixtures.py``).

A commit from a worktree runs pre-commit, which runs this suite, with git's hook
environment: ``GIT_DIR`` names the worktree's admin directory under the shared
``.git`` and ``GIT_INDEX_FILE`` its index. A test's ``git -C <tmp_path> init``
then re-initialises *that* repository, and with ``extensions.worktreeConfig``
on, writes ``core.bare = true`` into the shared ``.git/config``: every checkout
of the repository stops working at once. It happened twice before the guard
was made suite-wide (most recently 2026-10-14).

The sentinel here is a small repository of exactly that shape. The first test
shows the hazard is real against it; the second runs a pytest session with the
hook variables set and checks the sentinel comes through untouched.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(*args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.test", *args],
        check=True,
        capture_output=True,
        text=True,
        env=env if env is not None else _clean_env(),
    ).stdout.strip()


def _worktree_sentinel(root: Path) -> tuple[Path, dict[str, str]]:
    """A repository with one worktree; returns it and that worktree's hook env."""
    main = root / "main"
    _git("init", "-q", str(main))
    _git("-C", str(main), "commit", "-q", "--allow-empty", "-m", "seed")
    _git("-C", str(main), "config", "extensions.worktreeConfig", "true")
    _git("-C", str(main), "worktree", "add", "-q", str(root / "wt"))
    admin = main / ".git" / "worktrees" / "wt"
    hook_env = {"GIT_DIR": str(admin), "GIT_INDEX_FILE": str(admin / "index")}
    return main, hook_env


def test_a_hook_env_init_turns_the_outer_repository_bare(tmp_path: Path) -> None:
    # Control: without the guard, the sentinel shows the reported failure.
    main, hook_env = _worktree_sentinel(tmp_path)
    target = tmp_path / "new"
    target.mkdir()

    _git("-C", str(target), "init", "-q", env={**_clean_env(), **hook_env})

    assert not (target / ".git").exists()
    assert _git("-C", str(main), "config", "core.bare") == "true"


@pytest.mark.parametrize("guard", ["no_git_redirect", "pytest_configure"])
def test_a_suite_run_with_hook_env_leaves_the_outer_repository_alone(
    tmp_path: Path, guard: str
) -> None:
    # Each guard on its own: the per-test fixture, and the process-wide scrub
    # that covers collection and session-scoped fixtures.
    main, hook_env = _worktree_sentinel(tmp_path)
    config = main / ".git" / "config"
    before = config.read_bytes()

    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "pytest.ini").write_text("[pytest]\n")
    (suite / "conftest.py").write_text(
        f"from tests._client_fixtures import {guard}  # noqa: F401\n"
    )
    (suite / "test_inner.py").write_text(
        textwrap.dedent(
            """
            import subprocess


            def test_git_init_lands_in_tmp_path(tmp_path):
                subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
                assert (tmp_path / ".git").is_dir()
            """
        )
    )

    env = {**_clean_env(), **hook_env, "PYTHONPATH": str(_REPO_ROOT)}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(suite)],
        cwd=suite,
        env=env,
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert config.read_bytes() == before
    assert _git("-C", str(main), "config", "core.bare") == "false"
