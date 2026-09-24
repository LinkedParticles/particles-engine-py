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
from pathlib import Path


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
