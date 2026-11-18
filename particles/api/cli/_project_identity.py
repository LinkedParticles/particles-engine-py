# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Surface repository identity; preserves Claude's existing keys."""

from __future__ import annotations

import os
import re
from pathlib import Path

from particles.config import is_broad_root
from particles.corpus.harvest_marks import claude_project_slug as claude_project_slug
from particles.corpus.worktree import is_unlinked_git_file


def repository_root(cwd: Path) -> Path:
    """The root Claude Code keys auto-memory on for a session launched in ``cwd``.

    Walks up to the nearest ``.git`` entry, reading files only — no ``git``
    subprocess, because this runs inside the hook deadline. A ``.git``
    *directory* marks the root. A ``.git`` *file* is a linked worktree or a
    submodule: its ``gitdir:`` names the per-checkout git directory, and a
    ``commondir`` file there (worktrees have one, submodules do not) names the
    shared ``.git``, whose parent is the main checkout. With no ``.git``
    anywhere, ``cwd`` is its own root.

    The walk does not require ``cwd`` to exist: a worktree created under its
    own repository (Claude Code's ``.claude/worktrees/``) still resolves after
    it has been removed, which is the state a catch-up harvest finds it in.
    Paths are normalised, never symlink-resolved, so the slug still matches
    the string the platform saw.
    """
    return find_repository_root(cwd) or cwd


def find_repository_root(cwd: Path) -> Path | None:
    """:func:`repository_root`, or ``None`` when no ``.git`` entry is found at all."""
    for candidate in (cwd, *cwd.parents):
        dot_git = candidate / ".git"
        if dot_git.is_dir():
            return candidate
        if dot_git.is_file():
            # A linked ``.git`` file repeats another checkout's claim (Q26).
            if not is_unlinked_git_file(dot_git):
                return candidate
            return _main_checkout_of(candidate, dot_git) or candidate
    return None


def enclosing_work_tree(path: Path) -> Path | None:
    """The nearest directory above ``path``'s resolved file that holds a ``.git`` entry, if any.

    Read as the local refresh reads a store: the resolved path, so a name that
    links into a clone counts, and no further up than the home directory or
    another broad root, so a dotfiles repository there does not claim every
    file under home (security review 2026-10-27-1, W16).
    """
    for parent in Path(os.path.realpath(path)).parents:
        if is_broad_root(parent):
            break
        if os.path.lexists(parent / ".git"):
            return parent
    return None


def _main_checkout_of(checkout: Path, dot_git_file: Path) -> Path | None:
    """The main checkout a linked worktree belongs to, or ``None`` if it is not one.

    A ``.git`` file is a claim, so it is checked against git's own layout before
    the checkout is keyed on another directory (security review 2026-10-24-2,
    P22): its ``gitdir:`` must name ``<common>/worktrees/<name>``, that
    directory's ``commondir`` must lead back to ``<common>``, ``<common>`` must
    be a real git directory, and the worktree's ``gitdir`` back-pointer, which
    ``git worktree add`` writes, must name this very ``.git`` file. A planted
    chain (an unpacked archive, a co-tenant's directory) fails one of these and
    the checkout is its own root. A ``.git`` file that is a symbolic or hard
    link is refused before any of this (security review 2026-10-25-1, Q26).
    """
    try:
        first_line = dot_git_file.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        if not first_line.startswith("gitdir:"):
            return None
        git_dir = Path(os.path.normpath(checkout / first_line[len("gitdir:") :].strip()))
        common_file = git_dir / "commondir"
        if not common_file.is_file():
            return None  # a submodule: its own root
        common = Path(os.path.normpath(git_dir / common_file.read_text(encoding="utf-8").strip()))
        if not _is_registered_worktree(git_dir, common, dot_git_file):
            return None
    except (OSError, IndexError):
        return None
    # A bare-repository worktree has no main checkout to key on.
    return common.parent if common.name == ".git" else None


def _is_registered_worktree(git_dir: Path, common: Path, dot_git_file: Path) -> bool:
    """Whether ``git_dir`` is a worktree ``common`` registered for ``dot_git_file``."""
    if git_dir.parent.name != "worktrees" or git_dir.parent.parent != common:
        return False
    if not (common / "HEAD").is_file() or not (common / "objects").is_dir():
        return False
    back_file = git_dir / "gitdir"
    if not back_file.is_file():
        return False
    raw = back_file.read_text(encoding="utf-8", errors="replace").strip()
    back = Path(os.path.normpath(git_dir / raw))  # an absolute raw ignores git_dir
    own = Path(os.path.normpath(dot_git_file.absolute()))
    if back == own:
        return True
    # The same inode under another spelling of its path (case, a symlinked
    # ancestor). Sound only because ``own`` is neither a symlink nor
    # hard-linked, so that inode has no other name a planter could hold (Q26).
    try:
        return is_unlinked_git_file(own) and os.path.samefile(back, own)
    except OSError:
        return False


def observer_for_policy(project_key: str, policy: str) -> str | None:
    """A caller's explicit harness policy, independent of key/tag derivation."""
    return project_key if project_key and policy == "project" else None


def diagnostic_root_paths(key: str) -> tuple[Path, Path]:
    """Non-authoritative legacy/new paths; never read by refresh enrolment."""
    from particles.config import get_config

    config = get_config()
    return (
        Path(config.claude_code.state_dir).expanduser() / "projects" / key / "roots.json",
        Path(config.codex.state_dir).expanduser() / "projects" / key / "diagnostic-roots.json",
    )


def recorded_identity_roots(key: str) -> list[str]:
    """Read the diagnostic union without making an identity or policy decision."""
    import json
    import stat

    if re.fullmatch(r"[A-Za-z0-9-]+", key) is None:
        return []
    roots: set[str] = set()
    for path in diagnostic_root_paths(key):
        try:
            fd = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            )
            with os.fdopen(fd, "r") as stream:
                st = os.fstat(stream.fileno())
                if not stat.S_ISREG(st.st_mode) or st.st_size > 4 * 1024 * 1024:
                    continue
                data = json.load(stream)
            if isinstance(data, list):
                roots.update(item for item in data if isinstance(item, str))
        except (OSError, ValueError):
            continue
    return sorted(roots)


def record_identity_root(key: str, root: Path, *, codex: bool = False) -> list[str]:
    """Atomically union diagnostics under the one nonblocking cross-harness lock."""
    import logging

    from particles.api.cli._hook_state import advisory_lock, write_json
    from particles.config import get_config

    if re.fullmatch(r"[A-Za-z0-9-]+", key) is None:
        return []
    lock = Path(get_config().codex.identity_lock_path).expanduser()
    try:
        with advisory_lock(lock) as acquired:
            known = recorded_identity_roots(key)
            if not acquired:
                logging.getLogger(__name__).debug("project diagnostic lock busy")
                return known
            roots = sorted({*known, str(root)})
            write_json(diagnostic_root_paths(key)[int(codex)], roots)
            return roots
    except (OSError, ValueError):
        logging.getLogger(__name__).debug("project diagnostic recording unavailable")
        return recorded_identity_roots(key)
