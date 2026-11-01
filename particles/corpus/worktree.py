# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The main checkout of a git worktree, verified on disk (security review K6).

A worktree's ``.git`` is a file reading ``gitdir: <main>/.git/worktrees/<name>``,
and git writes the matching back-link, ``<main>/.git/worktrees/<name>/gitdir``,
naming the worktree's ``.git`` file. A ``.git`` file is the one thing in a
checkout that git itself cannot check out, but an archive or a tarball can
ship one, so its ``gitdir:`` line is a claim, not evidence: it is believed only
when the directory it names exists where git would put it and links back.

This is the one verified reading; a caller that admits files because a
directory is "the main checkout of" another calls :func:`verified_main_checkout`
rather than parsing the line itself.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def is_unlinked_git_file(dot_git: Path) -> bool:
    """Whether ``dot_git`` is a regular file that is neither a symlink nor hard-linked.

    git writes a worktree's ``.git`` file itself and never links it, so a
    ``.git`` entry that is a symbolic link, or a file with a second name, is a
    copy of another checkout's claim: believing it would key this directory on
    that checkout's repository (security review 2026-10-25-1, Q26).
    """
    try:
        st = os.lstat(dot_git)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_nlink == 1


def _realpath(path: Path) -> Path:
    return Path(os.path.realpath(path))


def verified_main_checkout(root: Path) -> Path | None:
    """The main checkout of the git worktree rooted at ``root``, else ``None``.

    ``None`` unless every link git itself writes is present and consistent:

    * ``<root>/.git`` is a regular file, not a link of either kind
      (:func:`is_unlinked_git_file`), whose first line is ``gitdir: <dir>``;
    * ``<dir>`` exists, is a directory named ``worktrees/<name>`` inside a
      ``.git`` directory, and its parent's parent is that main checkout;
    * ``<dir>/gitdir`` names ``<root>/.git`` (the back-link), and
      ``<dir>/commondir``, when present, names that ``.git`` directory.

    A submodule's ``.git`` file points under ``.git/modules`` and is not a
    worktree, so it yields ``None``. Every comparison is on real paths, and
    no failure raises: an unreadable link is an unverified one.
    """
    dot_git = root / ".git"
    try:
        if not is_unlinked_git_file(dot_git):
            return None
        lines = dot_git.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines or not lines[0].startswith("gitdir:"):
            return None
        named = Path(lines[0][len("gitdir:") :].strip())
        gitdir = _realpath(named if named.is_absolute() else root / named)
        common = gitdir.parent.parent
        if gitdir.parent.name != "worktrees" or common.name != ".git":
            return None
        if not gitdir.is_dir() or not common.is_dir():
            return None
        if _link(gitdir, "gitdir") != _realpath(dot_git):
            return None
        if (gitdir / "commondir").exists() and _link(gitdir, "commondir") != common:
            return None
    except (OSError, ValueError):
        return None
    return common.parent


def _link(gitdir: Path, name: str) -> Path | None:
    """The real path a one-line link file inside ``gitdir`` names, relative to ``gitdir``."""
    text = (gitdir / name).read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None
    return _realpath(gitdir / text)
