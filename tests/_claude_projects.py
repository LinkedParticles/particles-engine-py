# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""On-disk fixtures for Claude Code project identity.

Builds the three things the resolver reads — a main checkout, a linked
worktree laid out as ``git worktree add`` writes it, and a transcript that
records its working directory — without running ``git``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def make_repo(root: Path) -> Path:
    """A main checkout: a directory with a ``.git`` *directory*."""
    (root / ".git").mkdir(parents=True)
    return root


def make_worktree(repo: Path, worktree: Path, name: str = "wt") -> Path:
    """A linked worktree of ``repo``, laid out exactly as ``git worktree add`` does."""
    git_dir = repo / ".git" / "worktrees" / name
    git_dir.mkdir(parents=True)
    (git_dir / "commondir").write_text("../..\n")
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n")
    return worktree


def write_transcript(path: Path, *cwds: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "summary"})]  # a record with no cwd comes first
    lines += [json.dumps({"type": "user", "cwd": cwd, "message": {}}) for cwd in cwds]
    path.write_text("\n".join(lines) + "\n")
    return path


def bind_hooks_to_store(home: Path, dsn: str, *, store: str = "default") -> Path:
    """Install fake Particles hooks in ``<home>/.claude/settings.json`` naming ``dsn``.

    A Claude Code memory directory is harvested and rendered only by the store
    its installed hooks name (``_claude_code.projection_refusal``). A test that
    drives the projection against a fake ``~/.claude/projects`` binds its
    scratch store here, the way ``particles init claude-code`` would.
    """
    command = f"env DATABASE_URL={dsn} particles hook session-end --store {store}"
    settings = {"hooks": {"SessionEnd": [{"hooks": [{"type": "command", "command": command}]}]}}
    path = home / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n")
    return path


def isolate_claude_code(mp: pytest.MonkeyPatch, work: Path, extra_config: str = "") -> Path:
    """Run the Claude Code harness paths inside ``work``, never the real account.

    For a test that runs consolidation, the hooks, ``audit`` or anything else
    that reaches ``~/.claude/projects``: ``HOME`` moves to ``work/home`` (an
    empty projects root), the state directory follows it, and the projection
    is switched off. ``extra_config`` is prepended to the config file this
    writes and points ``PARTICLES_CONFIG`` at. Returns the fake home.
    """
    home = work / "home"
    (home / ".claude" / "projects").mkdir(parents=True, exist_ok=True)
    # The embedding model's cache lives under the real home; keep it, or the
    # fake one triggers a fresh model download.
    mp.setenv("HF_HOME", os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface"))
    mp.setenv("HOME", str(home))
    config = work / "config.yaml"
    config.write_text(
        extra_config
        + f"claude_code:\n  state_dir: {home / '.particles' / 'claude-code'}\n"
        + "agent_memory:\n  projection:\n    enabled: false\n"
    )
    mp.setenv("PARTICLES_CONFIG", str(config))
    return home
