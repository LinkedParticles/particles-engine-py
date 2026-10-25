# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Claude Code integration.

Used by ``particles init claude-code`` (``cli/init.py``) and the
``particles hook …`` verbs (``cli/hook.py``). Everything here is deliberately
**pure / store-free** except the hook-log writers and the project resolver
(local file I/O): the
settings-merge, transcript-distillation, redaction, and config.yaml-surgeon
functions take data in and hand data back, so the ADR's determinism and
surgicality guarantees are unit-testable without a store.

The leading underscore marks the module as internal to the ``cli`` package
(see ``cli/AGENTS.md`` — same convention as ``_logging.py`` / ``_remote.py``).
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from particles.config import StorageConfig, get_config, sqlite_file_path, user_config_path
from particles.render.markdown import find_projected_regions

log = logging.getLogger(__name__)

#: Sentinel substring identifying Particles-owned hook entries in Claude Code
#: settings. ``init`` replaces/removes exactly the entries whose
#: command contains it — the surgicality contract.
HOOK_SENTINEL = "particles hook"

#: The per-hook ``timeout`` (seconds) the installer writes on every
#: Particles-owned entry. Without one, Claude Code gives all SessionEnd hooks a
#: shared 1.5 s budget, and interpreter start-up plus imports alone take longer
#: than that, so the harvest is cancelled before its body runs. A per-hook
#: timeout raises that budget to match, up to 60 s, Claude Code's ceiling. On
#: SessionStart, which otherwise defaults to 600 s, it bounds a wedged digest
#: push to a minute. Either way it sits well above the hook's own internal
#: deadline (``claude_code.hook_deadline_seconds``).
HOOK_TIMEOUT_SECONDS = 60

#: Claude Code's default shared SessionEnd budget, in seconds. A Particles
#: SessionEnd hook whose ``timeout`` is absent or no larger than this is cut
#: off by it (``particles hook doctor`` reports one).
SESSION_END_DEFAULT_BUDGET_SECONDS = 1.5

#: Truncate-rotation cap for the hook log: when the JSONL file
#: exceeds this size, it is truncated and restarted (simple, no side-car state).
_HOOK_LOG_MAX_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Project identity — which memory directory a session writes to
# ---------------------------------------------------------------------------

#: How many transcript lines are read looking for the launch directory. The
#: first ``cwd``-bearing record is normally within the first handful.
_LAUNCH_CWD_SCAN_LINES = 60

#: What ``<repo>/.claude/worktrees/<name>`` becomes under the slug rule.
_CLAUDE_WORKTREE_INFIX = "--claude-worktrees-"


@dataclass(frozen=True)
class SessionProject:
    """The project a Claude Code session belongs to, as the platform keys it.

    ``key`` names the ``~/.claude/projects/<key>/`` directory that holds the
    session's **auto-memory**, which is not always the directory holding its
    transcript: Claude Code keys transcripts per working directory but
    auto-memory per repository, shared across worktrees.
    """

    key: str
    memory_dir: Path | None
    #: ``"repository"`` when the key was resolved from the session's launch
    #: directory; ``"worktree-name"`` when that could not be established but the
    #: transcript directory is named like a Claude Code worktree of a project
    #: that exists; ``"transcript-dir"`` when the transcript's own directory was
    #: used, which is the older behaviour.
    resolved_from: str
    #: The repository root the key was derived from, when it was resolved from
    #: the launch directory (``resolved_from == "repository"``); ``None`` for a
    #: key read off a directory name, whose path is not known.
    root: Path | None = None


def claude_project_slug(path: Path | str) -> str:
    """The ``~/.claude/projects/`` directory name Claude Code derives from a path.

    Every character outside ``[A-Za-z0-9]`` becomes ``-``. The platform does
    not document the rule; :func:`resolve_session_project` therefore never
    trusts it blindly — it only re-keys a session whose transcript directory
    this function reproduces.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


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
            return _main_checkout_of(candidate, dot_git) or candidate
    return None


def _main_checkout_of(checkout: Path, dot_git_file: Path) -> Path | None:
    """The main checkout a linked worktree belongs to, or ``None`` if it is not one."""
    try:
        first_line = dot_git_file.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        if not first_line.startswith("gitdir:"):
            return None
        git_dir = Path(os.path.normpath(checkout / first_line[len("gitdir:") :].strip()))
        common_file = git_dir / "commondir"
        if not common_file.is_file():
            return None  # a submodule: its own root
        common = Path(os.path.normpath(git_dir / common_file.read_text(encoding="utf-8").strip()))
    except (OSError, IndexError):
        return None
    # A bare-repository worktree has no main checkout to key on.
    return common.parent if common.name == ".git" else None


def _recorded_cwds(transcript_path: Path) -> list[str]:
    """Distinct ``cwd`` values from the head of a transcript, in order."""
    found: list[str] = []
    try:
        with transcript_path.open(encoding="utf-8", errors="replace") as fh:
            for index, line in enumerate(fh):
                if index >= _LAUNCH_CWD_SCAN_LINES:
                    break
                try:
                    cwd = json.loads(line).get("cwd")
                except (ValueError, AttributeError):
                    continue
                if isinstance(cwd, str) and cwd and cwd not in found:
                    found.append(cwd)
    except OSError:
        pass
    return found


def launch_directory(project_dir_name: str, candidates: list[str]) -> Path | None:
    """The candidate directory whose slug is ``project_dir_name``, if any.

    A session can change directory, so neither the hook payload's ``cwd`` nor
    a transcript's first record is reliably where it was launched. The
    transcript's own directory name *is* the launch directory's slug, which
    makes it the test: the first candidate that reproduces it is the launch
    directory. No match means no re-keying — including when the platform's
    slug rule has moved on, which therefore degrades to the old behaviour
    rather than to a directory that does not exist.
    """
    for candidate in candidates:
        if claude_project_slug(candidate) == project_dir_name:
            return Path(candidate)
    return None


def resolve_session_project(payload: Mapping[str, Any]) -> SessionProject:
    """Resolve a hook payload to the project whose memory the session writes to."""
    raw_transcript = str(payload.get("transcript_path") or "")
    transcript_path = Path(raw_transcript) if raw_transcript else None
    if transcript_path is None or not transcript_path.name:
        return SessionProject(key="", memory_dir=None, resolved_from="transcript-dir")

    project_dir = transcript_path.parent
    candidates = [str(payload.get("cwd") or ""), *_recorded_cwds(transcript_path)]
    launch = launch_directory(project_dir.name, [c for c in candidates if c])
    if launch is None:
        # The launch directory could not be identified (a session that moved
        # between checkouts records other directories). One thing is still
        # knowable from the name alone: a Claude Code worktree lives under its
        # repository, so its slug is the repository's plus a fixed infix. It is
        # adopted only when that repository's project directory really exists.
        repository_key = _strip_worktree_infix(project_dir.name)
        if repository_key != project_dir.name and (project_dir.parent / repository_key).is_dir():
            return SessionProject(
                key=repository_key,
                memory_dir=project_dir.parent / repository_key / "memory",
                resolved_from="worktree-name",
            )
        return SessionProject(
            key=project_dir.name,
            memory_dir=project_dir / "memory",
            resolved_from="transcript-dir",
        )
    root = repository_root(launch)
    key = claude_project_slug(root)
    return SessionProject(
        key=key,
        memory_dir=project_dir.parent / key / "memory",
        resolved_from="repository",
        root=root,
    )


def server_project_key(cwd: Path) -> str | None:
    """The project a ``mcp serve --project-observer cwd`` started in ``cwd`` binds.

    The key the hooks stamp for a session launched there: its repository's,
    or outside a repository the directory's own, so a home-directory session
    is its own observer. ``None`` at the filesystem root, which is no
    session's project; that server runs unbound.
    """
    root = repository_root(cwd)
    if root == Path(root.anchor):
        return None
    return claude_project_slug(root)


def observer_project_for(project_key: str) -> str | None:
    """The observer a session in ``project_key`` reads through, or ``None`` for the whole store.

    ``claude_code.observer_scope`` is the one switch: ``store`` (the default)
    leaves every session surface store-wide; ``project`` reads the digest, the
    ``MEMORY.md`` region and the freshness check through the session's project.
    An empty key — a session whose project could not be
    resolved — is never an observer.
    """
    if get_config().claude_code.observer_scope != "project" or not project_key:
        return None
    return project_key


def transcript_project_key(transcript: Path) -> str:
    """The project key of a transcript on disk — what the SessionEnd hook would stamp.

    Used where there is no hook payload (the first-run audit, ``rescope``), so
    a transcript harvested either way carries the same ``project:`` tag.
    """
    resolved = resolve_session_project({"transcript_path": str(transcript)})
    if resolved.resolved_from != "transcript-dir":
        return resolved.key
    # No hook is writing into a memory directory here, so the name alone is
    # enough: a worktree's slug stands for its repository whether or not that
    # repository's project directory still exists.
    return _strip_worktree_infix(resolved.key)


def rule_file_project_key(path: Path) -> str | None:
    """The project a rule document belongs to, or ``None`` for a global one.

    A rule file under the user-level ``~/.claude`` is how the user works
    everywhere — the *global rules*. One inside a repository is that project's.
    One in neither (a bare path the operator registered) is a hand deposit,
    and stays global.
    """
    resolved = path.expanduser()
    user_level = Path.home() / ".claude"
    if resolved == user_level or user_level in resolved.parents:
        return None
    root = find_repository_root(resolved.parent)
    return claude_project_slug(root) if root is not None else None


def canonical_project_key(slug: str, projects_root: Path) -> str:
    """The project key a ``~/.claude/projects/`` directory name stands for.

    A directory's own transcripts say where it was launched, which resolves a
    linked worktree anywhere on disk. When they are gone (the platform prunes
    them), a Claude Code worktree is still recognisable by name: it lives under
    its repository, so its slug is the repository's plus a fixed infix. Any
    other slug is its own key.
    """
    project_dir = projects_root / slug
    if project_dir.is_dir():
        for transcript in sorted(project_dir.glob("*.jsonl")):
            launch = launch_directory(slug, _recorded_cwds(transcript))
            if launch is not None:
                return claude_project_slug(repository_root(launch))
    return _strip_worktree_infix(slug)


def _strip_worktree_infix(slug: str) -> str:
    return slug.split(_CLAUDE_WORKTREE_INFIX, 1)[0] if _CLAUDE_WORKTREE_INFIX in slug else slug


_MEMORY_URI_RE = re.compile(r"/projects/(?P<slug>[^/]+)/memory/")
_SESSION_URI_PREFIX = "claude-code://session/"


def entry_project_key(uri_r: str | None, tags: list[str], projects_root: Path) -> str | None:
    """The key ``rescope`` should add to an already-deposited entry, or ``None``.

    Only this adapter's own deposits are attributed: a harvested memory file by
    the directory in its URI, a transcript by where its session was launched
    (while the session file still exists), a rule document by the repository it
    sits in, and anything carrying an older, per-worktree ``project:`` tag by
    what that slug stands for. Everything else — a hand deposit, a web page —
    is left alone, and stays global.
    """
    uri = uri_r or ""
    if "rule-file" in tags and uri.startswith("file://"):
        return rule_file_project_key(Path(uri[len("file://") :]))
    if "claude-code" not in tags:
        return None
    match = _MEMORY_URI_RE.search(uri) if uri.startswith("file://") else None
    if match is not None:
        return canonical_project_key(match.group("slug"), projects_root)
    if uri.startswith(_SESSION_URI_PREFIX):
        session_id = uri[len(_SESSION_URI_PREFIX) :]
        for transcript in sorted(projects_root.glob(f"*/{session_id}.jsonl")):
            return transcript_project_key(transcript)
    for tag in tags:
        if tag.startswith("project:") and len(tag) > len("project:"):
            return canonical_project_key(tag[len("project:") :], projects_root)
    return None


def artifact_namespace_key(
    tags: Sequence[str], uri: str | None = None, projects_root: Path | None = None
) -> str | None:
    """The project a qualified subject name from this entry is scoped by.

    Entry tags are additive, so an entry deposited before
    worktree folding carries its per-worktree tag beside the canonical one.
    Both fold to one repository, and that repository's key is the namespace.
    An entry whose tags fold to more than one project, or that carries none,
    gets ``None``, and the gate suppresses the name.
    """
    from particles.core.observer_scope import project_keys

    root = projects_root if projects_root is not None else Path.home() / ".claude" / "projects"
    folded = {_canonical_key_cached(key, str(root)) for key in project_keys(tags)}
    return next(iter(folded)) if len(folded) == 1 else None


@functools.lru_cache(maxsize=1024)
def _canonical_key_cached(slug: str, projects_root: str) -> str:
    # canonical_project_key reads a directory's transcripts, and one extraction
    # pass asks about the same few projects once per candidate.
    return canonical_project_key(slug, Path(projects_root))


def artifact_relative_path(namespace_key: str, path: str) -> str | None:
    """An absolute path's form from its repository root, by this adapter's key rule.

    The repository root is the shortest leading run of the path's directories
    whose :func:`claude_project_slug` is ``namespace_key``. A remainder under
    ``.claude/worktrees/<name>/`` is in one of that repository's worktrees, so
    those three segments go too. A path under no such directory, a home-relative
    ``~/`` path included, gets ``None`` and is kept as spelled.

    Matching on the slug, never on a folded worktree slug, is deliberate: the
    fold strips everything after the worktree infix, so it also matches every
    directory below a worktree root and would turn
    ``…/worktrees/x/particles/secrets.py`` into ``secrets.py``.
    """
    if not path.startswith("/"):
        return None
    parts = path.split("/")
    for i in range(2, len(parts)):
        if claude_project_slug("/".join(parts[:i])) != namespace_key:
            continue
        rest = parts[i:]
        if len(rest) > 3 and rest[0] == ".claude" and rest[1] == "worktrees":
            rest = rest[3:]
        return "/".join(rest) or None
    return None


def register_artifact_namespace_hook() -> None:
    """Make this adapter's project fold the Engine's artifact namespace.

    Also registers its root rule, which maps an absolute path to its form from
    the repository root. Called from the CLI's root callback,
    which every Surface process passes through (`engine serve`, `mcp serve` and
    the hooks are CLI verbs). A raw ``uvicorn particles.api.app:app`` launch
    skips it and gets the Engine's fail-closed defaults. Idempotent.
    """
    from particles.ingest.artifact_namespace import (
        register_artifact_namespace,
        register_artifact_relative_path,
    )

    register_artifact_namespace(artifact_namespace_key)
    register_artifact_relative_path(artifact_relative_path)


def is_live_project_key(key: str, projects_root: Path) -> bool:
    """Whether a session could ever observe as ``key``: its directory exists and is canonical."""
    return (projects_root / key).is_dir() and canonical_project_key(key, projects_root) == key


def is_claude_code_memory_dir(path: Path, projects_root: Path | None = None) -> bool:
    """Whether ``path`` is a memory directory Claude Code itself reads.

    That is ``<projects_root>/<key>/memory`` for a project directory Claude
    Code created, and not one of the stray directories this SDK's own
    projection left behind (:func:`stray_memory_dirs`). ``projects_root``
    defaults to ``~/.claude/projects``. Symlinks resolve first, so a link to a
    real memory directory counts and a copy of one elsewhere does not.
    """
    root = projects_root if projects_root is not None else Path.home() / ".claude" / "projects"
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    if resolved.name != "memory" or resolved.parent.parent != root_resolved:
        return False
    if not resolved.is_dir():
        return False
    return resolved not in {p.resolve() for p in stray_memory_dirs(root_resolved)}


def stray_memory_dirs(projects_root: Path) -> list[Path]:
    """Memory directories under ``projects_root`` that Claude Code never reads.

    A directory is stray when its project directory is a linked worktree's:
    the sessions launched there key their memory on the main checkout, so a
    ``memory/`` beside their transcripts can only have been created by this
    SDK's own projection. Two tests, both conservative:

    * a transcript records the launch directory, and it resolves to a
      different repository key; or
    * no transcript is left to ask (the platform prunes them; the stray
      outlives them), the project directory is named like one of Claude
      Code's own worktrees, and it holds nothing but a ``MEMORY.md`` that is
      purely the projected region — output only this SDK writes. A real
      project with no memory of its own looks the same but for the name, and
      its region *is* read, so the name is part of the test.
    """
    strays: list[Path] = []
    if not projects_root.is_dir():
        return strays
    for memory_dir in sorted(p for p in projects_root.glob("*/memory") if p.is_dir()):
        project_dir = memory_dir.parent
        transcripts = sorted(project_dir.glob("*.jsonl"))
        if not transcripts:
            if _CLAUDE_WORKTREE_INFIX in project_dir.name and _holds_only_the_projected_region(
                memory_dir
            ):
                strays.append(memory_dir)
            continue
        for transcript in transcripts:
            launch = launch_directory(project_dir.name, _recorded_cwds(transcript))
            if launch is None:
                continue
            if claude_project_slug(repository_root(launch)) != project_dir.name:
                strays.append(memory_dir)
            break
    return strays


def _holds_only_the_projected_region(memory_dir: Path) -> bool:
    try:
        if [p.name for p in memory_dir.iterdir()] != ["MEMORY.md"]:
            return False
        text = (memory_dir / "MEMORY.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    regions = find_projected_regions(text)
    if len(regions) != 1:
        return False
    region = regions[0]
    return not (text[: region.start] + text[region.end :]).strip()


# ---------------------------------------------------------------------------
# State directory + hook log (§1 / §6)
# ---------------------------------------------------------------------------


def state_dir() -> Path:
    """The integration's per-machine state directory (``claude_code.state_dir``)."""
    return Path(get_config().claude_code.state_dir).expanduser()


def make_private_dir(path: Path) -> None:
    """Create ``path`` and any missing parents owner-only (``0o700``).

    Only directories this call creates get the mode, so an existing directory
    the owner chose (their home, a configured log folder) is never changed.
    The state directory and everything below it hold harvested memory, render
    backups and hook telemetry, which another account on the machine must not
    read (security review D46).
    """
    missing: list[Path] = []
    for candidate in (path, *path.parents):
        if candidate.exists():
            break
        missing.append(candidate)
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            continue
        os.chmod(directory, 0o700)  # mkdir's mode is filtered by the umask


def _open_private(path: Path, flags: int) -> int:
    make_private_dir(path.parent)
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)  # an existing file written by an older version
    return fd


def write_private_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` owner-only (``0o600``), never through a symlink."""
    fd = _open_private(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def append_private_text(path: Path, text: str) -> None:
    """Append ``text`` to ``path`` owner-only (``0o600``), never through a symlink."""
    fd = _open_private(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(text)


def hook_log_path() -> Path:
    """Resolved hook-log path: ``claude_code.hook_log_path`` or ``<state_dir>/hooks.jsonl``."""
    configured = get_config().claude_code.hook_log_path
    if configured:
        return Path(configured).expanduser()
    return state_dir() / "hooks.jsonl"


def append_hook_log(record: dict[str, Any]) -> None:
    """Append one JSONL line to the hook log; never raises.

    The log's most important entries are written when the store is unreachable,
    so the writer itself must not become a failure surface — any I/O error is
    swallowed (logged at debug). Truncate-rotation: an over-cap file is
    restarted rather than rolled. Transcript *content* is never logged — only
    counts, identifiers, and error strings belong in ``record``.
    """
    try:
        path = hook_log_path()
        if path.exists() and path.stat().st_size > _HOOK_LOG_MAX_BYTES:
            path.unlink()
        append_private_text(path, json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — the log must never break a hook
        log.debug("hook log write failed", exc_info=True)


def read_hook_log_tail(n: int) -> list[str]:
    """Return the last ``n`` raw JSONL lines of the hook log ([] when absent)."""
    path = hook_log_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-n:] if n > 0 else []


# ---------------------------------------------------------------------------
# Settings merge — marker-owned, idempotent, surgical
# ---------------------------------------------------------------------------


def build_hook_commands(
    command_base: str,
    store: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The hook command strings for a given base + store handle.

    Claude Code runs the hook from the *session's* working directory, which is
    almost never the store's directory (it is routinely a git worktree). Any
    ``env`` pins — resolved to absolute paths at install time — are rendered as
    a leading ``env KEY=VALUE …`` prefix so the store the hook resolves is
    CWD-independent (see :func:`render_env_prefix`). An empty /
    absent ``env`` reproduces the original bare command exactly.
    """
    prefix = render_env_prefix(env)
    commands = {
        "SessionStart": f"{prefix}{command_base} hook session-start --store {store}",
        "SessionEnd": f"{prefix}{command_base} hook session-end --store {store}",
    }
    return commands


def render_env_prefix(env: Mapping[str, str] | None) -> str:
    """Render an ``env KEY=VALUE … `` shell prefix (empty string when no vars).

    Values are shell-quoted so paths with spaces survive; keys are assumed to
    be well-formed env-var names (the caller controls them). The trailing space
    makes the prefix concatenate cleanly in front of the command base.
    """
    if not env:
        return ""
    parts = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    return f"env {parts} " if parts else ""


def absolutize_sqlite_dsn(dsn: str, *, base_dir: Path | None = None) -> str:
    """Resolve a CWD-relative SQLite DSN to an absolute one.

    ``sqlite+aiosqlite:///./particles.db`` (three slashes, relative path) is
    resolved against ``base_dir`` (default: the current working directory,
    matching how SQLAlchemy resolves it at runtime) and returned as an absolute
    four-slash DSN. An already-absolute SQLite DSN and any non-SQLite DSN
    (Postgres, …) are returned unchanged — a network DSN is already
    location-independent.
    """
    scheme, sep, path = dsn.partition(":///")
    if not sep or not scheme.startswith("sqlite"):
        return dsn
    file_path = Path(path)
    if file_path.is_absolute():
        return dsn
    base = base_dir if base_dir is not None else Path.cwd()
    abs_path = (base / file_path).resolve()
    return f"{scheme}:///{abs_path}"


def _is_particles_hook(hook: Any) -> bool:
    return isinstance(hook, dict) and HOOK_SENTINEL in str(hook.get("command", ""))


def particles_hook_commands(settings: dict[str, Any]) -> list[str]:
    """Every Particles-owned hook command string currently in ``settings``."""
    found: list[str] = []
    hooks_cfg = settings.get("hooks")
    if not isinstance(hooks_cfg, dict):
        return found
    for groups in hooks_cfg.values():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                continue
            for hook in group["hooks"]:
                if _is_particles_hook(hook):
                    found.append(str(hook["command"]))
    return found


def untimed_session_end_hooks(settings: dict[str, Any]) -> list[str]:
    """Particles-owned SessionEnd commands whose ``timeout`` leaves the 1.5 s budget in force.

    An entry written before the installer set :data:`HOOK_TIMEOUT_SECONDS`, or
    edited by hand, inherits Claude Code's shared SessionEnd budget, which the
    harvest cannot finish inside. A ``timeout`` that is not a number counts as
    absent.
    """
    hooks_cfg = settings.get("hooks")
    if not isinstance(hooks_cfg, dict):
        return []
    groups = hooks_cfg.get("SessionEnd")
    if not isinstance(groups, list):
        return []
    found: list[str] = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
            continue
        for hook in group["hooks"]:
            if not _is_particles_hook(hook):
                continue
            timeout = hook.get("timeout")
            numeric = isinstance(timeout, int | float) and not isinstance(timeout, bool)
            if not numeric or timeout <= SESSION_END_DEFAULT_BUDGET_SECONDS:
                found.append(str(hook["command"]))
    return found


def strip_particles_hook_entries(settings: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``settings`` with every Particles-owned hook removed.

    Surgical by construction: only hooks whose command string contains
    :data:`HOOK_SENTINEL` are dropped. Matcher groups / event arrays / the
    ``hooks`` key are removed only when *our* removal emptied them — a
    structure that was already empty (or is foreign) is preserved.
    """
    result = copy.deepcopy(settings)
    hooks_cfg = result.get("hooks")
    if not isinstance(hooks_cfg, dict):
        return result

    for event in list(hooks_cfg):
        groups = hooks_cfg[event]
        if not isinstance(groups, list):
            continue
        new_groups: list[Any] = []
        removed_any = False
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                new_groups.append(group)
                continue
            kept = [h for h in group["hooks"] if not _is_particles_hook(h)]
            if len(kept) == len(group["hooks"]):
                new_groups.append(group)
                continue
            removed_any = True
            if kept:
                group = dict(group)
                group["hooks"] = kept
                new_groups.append(group)
            # else: the group held only our hooks — drop it entirely.
        if removed_any and not new_groups:
            del hooks_cfg[event]
        else:
            hooks_cfg[event] = new_groups

    if not hooks_cfg and isinstance(settings.get("hooks"), dict) and settings["hooks"]:
        # Our removal emptied the hooks mapping; a pre-existing empty mapping
        # (settings["hooks"] == {}) is left alone by the condition above.
        del result["hooks"]
    return result


def merge_particles_hook_entries(
    settings: dict[str, Any], commands: dict[str, str]
) -> dict[str, Any]:
    """Merge the Particles-owned hook entries into ``settings`` (replace semantics).

    Strips any existing Particles-owned entries first (re-run = repair/upgrade),
    then appends one matcher-group per event, each carrying an explicit
    :data:`HOOK_TIMEOUT_SECONDS`. Everything else is preserved.
    """
    result = strip_particles_hook_entries(settings)
    hooks_cfg = result.setdefault("hooks", {})
    if not isinstance(hooks_cfg, dict):
        raise ValueError(
            "Claude Code settings 'hooks' key is not an object — refusing to rewrite it."
        )
    for event, command in commands.items():
        groups = hooks_cfg.setdefault(event, [])
        if not isinstance(groups, list):
            raise ValueError(
                f"Claude Code settings 'hooks.{event}' is not an array — refusing to rewrite it."
            )
        groups.append(
            {"hooks": [{"type": "command", "command": command, "timeout": HOOK_TIMEOUT_SECONDS}]}
        )
    return result


def render_settings_json(settings: dict[str, Any]) -> str:
    """Stable 2-space serialization for settings files."""
    return json.dumps(settings, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Transcript distillation — deterministic, LLM-free
# ---------------------------------------------------------------------------

#: Preference order for the one-line tool summary: the first of these keys
#: present in a tool_use input is shown (``[tool: Bash — git status]``).
_TOOL_SUMMARY_KEYS = (
    "command",
    "file_path",
    "path",
    "url",
    "pattern",
    "query",
    "description",
    "prompt",
    "skill",
    "title",
)

_MAX_TOOL_SUMMARY_CHARS = 120


def _tool_line(block: dict[str, Any]) -> str:
    name = str(block.get("name", "unknown"))
    tool_input = block.get("input")
    detail = ""
    if isinstance(tool_input, dict):
        for key in _TOOL_SUMMARY_KEYS:
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                detail = " ".join(value.split())
                if len(detail) > _MAX_TOOL_SUMMARY_CHARS:
                    detail = detail[: _MAX_TOOL_SUMMARY_CHARS - 1] + "…"
                break
    return f"[tool: {name} — {detail}]" if detail else f"[tool: {name}]"


def _turn_parts(entry_type: str, content: Any) -> list[str]:
    """Extract the distilled parts of one transcript turn.

    Speaker text blocks are kept verbatim; assistant ``tool_use`` blocks become
    one-line summaries; ``tool_result`` and ``thinking`` blocks are dropped
    (tool results are where payloads and secrets concentrate).
    """
    if isinstance(content, str):
        return [content] if content.strip() else []
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        elif block_type == "tool_use" and entry_type == "assistant":
            parts.append(_tool_line(block))
    return parts


def distill_transcript(
    jsonl_text: str,
    session_id: str | None = None,
) -> str:
    """Distill a Claude Code transcript JSONL into speaker-turn Markdown.

    Deterministic and LLM-free: user and assistant text turns
    verbatim under ``## User`` / ``## Assistant`` headings, tool calls elided
    to one-line summaries, tool results and thinking blocks dropped, malformed
    lines skipped. The output format is **stable input** for a future
    conversation-aware extractor (§ Composition point) — change it
    only with a distiller version story.

    A ``user`` record Claude Code marks ``isMeta`` is left out too: the
    harness injected it (a loaded skill's body, a local-command caveat, an
    auto-continue prompt, a message from another session, an expanded slash
    command), so it is not the person speaking and must not be distilled as
    a ``## User`` turn whose claims would be attributed to them.
    """
    out: list[str] = []
    if session_id:
        out.append(f"# Claude Code session {session_id}")
    for raw_line in jsonl_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            entry = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get("type")
        if entry_type not in ("user", "assistant"):
            continue
        if entry_type == "user" and entry.get("isMeta"):
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        parts = _turn_parts(str(entry_type), message.get("content"))
        if not parts:
            continue
        speaker = "User" if entry_type == "user" else "Assistant"
        out.append(f"## {speaker}")
        out.append("\n\n".join(parts))
    return ("\n\n".join(out) + "\n") if out else ""


def _is_particles_server(server: Any) -> bool:
    """An MCP server entry that runs ``particles mcp serve``."""
    if not isinstance(server, dict):
        return False
    argv = [str(server.get("command", ""))] + [str(a) for a in server.get("args") or []]
    words = " ".join(argv).split()
    return any(w.endswith("particles") for w in words) and "mcp" in words and "serve" in words


def _servers_in(config: Any) -> list[Any]:
    servers = config.get("mcpServers") if isinstance(config, dict) else None
    return list(servers.values()) if isinstance(servers, dict) else []


def particles_executable(argv: str | Sequence[str]) -> str | None:
    """The ``particles`` executable a hook command or MCP server entry runs, if it names one.

    ``argv`` is a hook command string (``env K=V … /path/particles hook session-end``)
    or a server's argv. ``None`` when no word is a ``particles`` executable,
    e.g. a server launched through ``uv run --project … particles``, whose
    install depends on that project's environment.
    """
    if isinstance(argv, str):
        try:
            words = shlex.split(argv)
        except ValueError:
            return None
    else:
        words = list(argv)
    for word in words:
        if Path(word).name == "particles" and os.sep in word:
            return word
    return None


#: The launch flag that binds a Particles MCP server to its session's project.
PROJECT_OBSERVER_ARGS = ("--project-observer", "cwd")


@dataclass(frozen=True)
class McpRegistration:
    """One user-scope Particles MCP server entry, as ``claude mcp add`` takes it."""

    name: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str]

    @property
    def bound(self) -> bool:
        """Whether it starts with ``--project-observer cwd``."""
        words = list(self.args)
        return any(words[i : i + 2] == list(PROJECT_OBSERVER_ARGS) for i in range(len(words) - 1))

    def add_argv(self) -> list[str]:
        """``claude mcp add`` arguments that (re)create this entry at user scope."""
        argv = ["mcp", "add", "--scope", "user"]
        for key, value in sorted(self.env.items()):
            argv += ["-e", f"{key}={value}"]
        return [*argv, self.name, "--", self.command, *self.args]


def particles_mcp_registrations(home: Path | None = None) -> list[McpRegistration]:
    """Every user-scope MCP server in ``~/.claude.json`` that runs ``particles mcp serve``."""
    home = home or Path.home()
    try:
        user = json.loads((home / ".claude.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    servers = user.get("mcpServers") if isinstance(user, dict) else None
    found: list[McpRegistration] = []
    for name, server in (servers or {}).items() if isinstance(servers, dict) else ():
        if not _is_particles_server(server):
            continue
        env = server.get("env") if isinstance(server.get("env"), dict) else {}
        found.append(
            McpRegistration(
                name=str(name),
                command=str(server.get("command", "")),
                args=tuple(str(a) for a in server.get("args") or []),
                env={str(k): str(v) for k, v in env.items()},
            )
        )
    return found


def plan_mcp_registration(
    existing: Sequence[McpRegistration],
    *,
    executable: str,
    env: Mapping[str, str],
    replace_config_pin: bool = False,
) -> tuple[McpRegistration, McpRegistration | None] | None:
    """What ``init claude-code`` registers, as ``(wanted, replaced)``; ``None`` if unchanged.

    With no Particles server, a new ``particles`` entry runs this install's
    executable with the hooks' env pins. An existing one keeps its name and
    executable, gains ``--project-observer cwd`` when it lacks it,
    gains the ``PARTICLES_CONFIG`` pin when it has none,
    and gains the wanted ``DATABASE_URL`` when it has none and pins the wanted
    config, whether or not that pin is new. A server with no config pin read
    whatever config its starting directory offered (finding D8), and one with
    no ``DATABASE_URL`` serves the ``default`` store from ``./particles.db`` in
    each session's directory unless its config names an absolute one
    (security review N3, N17). A ``DATABASE_URL`` already present is kept. A
    config pin naming a different file is kept with its env as found, since
    the installer cannot tell a stale pin from a deliberate one, unless
    ``replace_config_pin`` (an explicit ``--config``) says to replace it; see
    :func:`foreign_config_pin`.
    """
    if not existing:
        return (
            McpRegistration(
                name="particles",
                command=executable,
                args=("mcp", "serve", *PROJECT_OBSERVER_ARGS),
                env=dict(env),
            ),
            None,
        )
    current = existing[0]
    new_env = dict(current.env)
    wanted_config = env.get("PARTICLES_CONFIG")
    have_config = new_env.get("PARTICLES_CONFIG")
    if wanted_config is not None and (
        have_config is None or (replace_config_pin and have_config != wanted_config)
    ):
        new_env["PARTICLES_CONFIG"] = wanted_config
    if (
        "DATABASE_URL" in env
        and "DATABASE_URL" not in new_env
        and new_env.get("PARTICLES_CONFIG") == wanted_config
    ):
        new_env["DATABASE_URL"] = env["DATABASE_URL"]
    args = current.args if current.bound else (*current.args, *PROJECT_OBSERVER_ARGS)
    if args == current.args and new_env == current.env:
        return None
    return (
        McpRegistration(name=current.name, command=current.command, args=args, env=new_env),
        current,
    )


def foreign_config_pin(existing: Sequence[McpRegistration], env: Mapping[str, str]) -> str | None:
    """The ``PARTICLES_CONFIG`` an existing registration pins, when it is not the one wanted.

    :func:`plan_mcp_registration` leaves such a pin in place; the installer
    reports it so the operator can replace it with ``--config``.
    """
    if not existing:
        return None
    have = existing[0].env.get("PARTICLES_CONFIG")
    wanted = env.get("PARTICLES_CONFIG")
    return have if have is not None and wanted is not None and have != wanted else None


def particles_mcp_executables(home: Path | None = None) -> list[str]:
    """The ``particles`` executables the user-scope Particles MCP servers run.

    Only servers that name the executable by path are counted (see
    :func:`particles_executable`).
    """
    home = home or Path.home()
    try:
        user = json.loads((home / ".claude.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    found: list[str] = []
    for server in _servers_in(user):
        if not _is_particles_server(server):
            continue
        argv = [str(server.get("command", ""))] + [str(a) for a in server.get("args") or []]
        exe = particles_executable(argv)
        if exe is not None:
            found.append(exe)
    return found


@dataclass(frozen=True)
class InstallPin:
    """Where a source-checkout install sits relative to its repository's release tags."""

    branch: str | None
    """The checked-out branch, or ``None`` when HEAD is detached."""
    tag: str | None
    """The release tag HEAD is exactly at, if any."""
    newest_tag: str | None
    """The newest ``v*`` tag the checkout has fetched."""


def install_report(
    *,
    root: Path,
    version: str,
    pin: InstallPin | None,
    hook_commands: Sequence[str],
    mcp_exes: Sequence[str],
    running_exe: str | None,
) -> list[str]:
    """The doctor's ``install:`` lines: which install sessions run, and whether it is stale.

    A Claude Code session keeps its MCP server for its whole life, and the
    server imports some modules only on first use. A server running from a
    checkout that is then pulled mixes old and new code and fails with an
    ``ImportError``. A pinned release tag avoids that, but only while someone
    remembers to advance it. Every line is advisory.
    """
    pad = " " * 16
    lines = [f"install:        Particles {version} at {root}"]
    hook_exes = [e for e in map(particles_executable, hook_commands) if e is not None]
    hooks = sorted({str(Path(e).resolve()) for e in hook_exes})
    servers = sorted({str(Path(e).resolve()) for e in mcp_exes})
    if len(hooks) > 1:
        lines.append(f"{pad}⚠ the installed hooks run different executables: {', '.join(hooks)}")
    if hooks and servers and set(servers) != set(hooks):
        lines.append(
            f"{pad}⚠ the MCP server runs {', '.join(servers)} but the hooks run "
            f"{', '.join(hooks)}. Point both at one install."
        )
    running = str(Path(running_exe).resolve()) if running_exe else None
    if running and hooks and running not in hooks:
        lines.append(f"{pad}⚠ this doctor runs {running}, not the install the hooks run.")
        command = next(c for c in hook_commands if particles_executable(c))
        store = re.search(r" --store \S+", command)
        doctor = command.split(" hook ")[0] + " hook doctor" + (store.group(0) if store else "")
        lines.append(f"{pad}To check what your sessions run: `{doctor}`")
        return lines
    if pin is None:
        return lines
    if pin.branch is not None:
        lines.append(
            f"{pad}⚠ runs from a working checkout (branch {pin.branch}). Pulling it changes"
        )
        lines.append(f"{pad}code under running sessions. Run the hooks and the MCP server from a")
        lines.append(
            f"{pad}release tag in a separate worktree (the Claude Code guide, "
            "Running from a source checkout)."
        )
    elif pin.tag is None:
        lines.append(f"{pad}⚠ detached at a commit that is not a release tag.")
    elif pin.newest_tag is not None and pin.newest_tag != pin.tag:
        lines.append(
            f"{pad}⚠ pinned at {pin.tag}; {pin.newest_tag} is the newest tag fetched. To advance:"
        )
        lines.append(f"{pad}  git -C {root} checkout --detach {pin.newest_tag}")
        lines.append(f"{pad}  uv sync --project {root} --frozen --no-dev")
        lines.append(f"{pad}then re-run this doctor: its schema: line prints the `db init` that")
        lines.append(f"{pad}migrates the store when the new release needs one. Then reopen")
        lines.append(f"{pad}running sessions.")
    else:
        lines.append(
            f"{pad}pinned at {pin.tag}, the newest tag fetched (`git fetch --tags` to check)"
        )
    return lines


def session_started_at(uri_r: str | None, projects_root: Path) -> datetime | None:
    """The start of the session a ``claude-code://session/<id>`` URI names, or ``None``.

    Read from the transcript's first ``timestamp`` while the file still exists
    under ``projects_root``; ``particles memory redate`` injects it
    into the Engine op, which never learns where transcripts live.
    """
    uri = uri_r or ""
    if not uri.startswith(_SESSION_URI_PREFIX):
        return None
    session_id = uri[len(_SESSION_URI_PREFIX) :]
    for transcript in sorted(projects_root.glob(f"*/{session_id}.jsonl")):
        try:
            return transcript_started_at(transcript.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return None


def transcript_started_at(jsonl_text: str) -> datetime | None:
    """The first ``timestamp`` a Claude Code transcript records, or ``None``.

    The session's start, which use mining reads exposure as of.
    The distilled transcript keeps no timestamps, so this reads the raw JSONL.
    Malformed lines and unparseable values are skipped.
    """
    for raw_line in jsonl_text.splitlines():
        try:
            entry = json.loads(raw_line)
        except ValueError:
            continue
        value = entry.get("timestamp") if isinstance(entry, dict) else None
        if not isinstance(value, str):
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


# ---------------------------------------------------------------------------
# Redaction — best-effort defence in depth, not a guarantee
# ---------------------------------------------------------------------------

_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    # PEM blocks first (they may contain substrings the later patterns match).
    (
        re.compile(r"-----BEGIN [A-Z0-9 ]+-----.*?-----END [A-Z0-9 ]+-----", re.DOTALL),
        "[REDACTED PEM BLOCK]",
    ),
    # Bearer headers before generic keys so the header keeps its label.
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    # Anthropic / OpenAI-style secret keys.
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "[REDACTED API KEY]"),
    # AWS access key IDs.
    (re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b"), "[REDACTED AWS KEY]"),
    # GitHub tokens: classic (ghp_, gho_, ghu_, ghs_, ghr_) and fine-grained.
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"), "[REDACTED GITHUB TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"), "[REDACTED GITHUB TOKEN]"),
    # Slack tokens (xoxb-, xoxp-, xoxa-, xoxr-, xoxs-, xoxe-).
    (re.compile(r"\bxox[abeprs]-[A-Za-z0-9-]{10,}"), "[REDACTED SLACK TOKEN]"),
    # Google API keys.
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}"), "[REDACTED GOOGLE KEY]"),
    # JSON Web Tokens: three base64url segments, the first two JSON objects.
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        "[REDACTED JWT]",
    ),
    # Secret-named assignments (password=…, api_key: …, "token": "…"), value only.
    (
        re.compile(
            r"(?i)(?P<key>\b(?:password|passwd|pwd|secret|client_secret|api[_-]?key|"
            r"access[_-]?token|auth[_-]?token|refresh[_-]?token|token)\b[\"']?\s*[:=]\s*)"
            r"(?P<quote>[\"']?)(?P<value>[^\s\"',;]{6,})(?P=quote)"
        ),
        r"\g<key>\g<quote>[REDACTED]\g<quote>",
    ),
]


def redact_secrets(text: str) -> str:
    """Mask common credential shapes.

    Deterministic pattern pass over harvested text before it reaches the
    corpus (distilled transcripts, memory files, rule files): ``sk-…`` keys,
    AWS access key IDs, GitHub, Slack and Google tokens, JWTs, ``Bearer``
    headers, PEM blocks, and the value of a secret-named ``key=value`` or
    ``key: value`` assignment. Best-effort defence in depth — stated as such
    in the docs.
    """
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# MEMORY.md projection paths + manifest (§3/§6/§7)
# ---------------------------------------------------------------------------

#: The one projected region name in MEMORY.md. Also the manifest
#: ``name:`` and the render-snapshot filename stem, per the
#: ``snapshot_path_for`` convention relocated into the state directory.
MEMORY_REGION = "memory-index"

#: Filename of the append-only fold-and-archive target in the state directory.
MEMORY_ARCHIVE_NAME = "MEMORY.archive.md"

#: Every fold-and-archive pointer line in MEMORY.md starts with this. It is
#: machine-generated (like the sentinels), so the pre-deposit filter drops it
#: — which is what keeps the §2 fixed point exact once fold-and-archive has
#: converged the file to region + pointer.
ARCHIVE_POINTER_PREFIX = "*Older agent memories were consolidated"


def memory_manifest_path() -> Path:
    """The projection manifest path: ``agent_memory.projection.manifest`` or
    ``<state_dir>/memory.yaml``."""
    configured = get_config().agent_memory.projection.manifest
    if configured:
        return Path(configured).expanduser()
    return state_dir() / "memory.yaml"


def project_state_dir(memory_dir: Path) -> Path:
    """Per-project state: ``<state_dir>/projects/<project>/``.

    ``<project>`` is the name of the ``~/.claude/projects/`` directory that
    holds ``memory_dir``. Whatever records *what was last written into one
    project's* ``MEMORY.md`` lives here, because one machine-wide copy is
    overwritten by every other project's cycle.
    """
    return state_dir() / "projects" / memory_dir.parent.name


# ---------------------------------------------------------------------------
# Project-key collisions
# ---------------------------------------------------------------------------

#: A key :func:`claude_project_slug` can produce, so it is safe as a directory name.
_PROJECT_KEY_RE = re.compile(r"[A-Za-z0-9-]+")


def project_roots_path(key: str) -> Path:
    """Where the repository roots seen for one project key are recorded."""
    return state_dir() / "projects" / key / "roots.json"


def recorded_project_roots(key: str) -> list[str]:
    """The repository roots recorded for ``key``, sorted; ``[]`` when none or unreadable."""
    if not _PROJECT_KEY_RE.fullmatch(key):
        return []
    try:
        data = json.loads(project_roots_path(key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return sorted({item for item in data if isinstance(item, str)})


def record_project_root(key: str, root: Path) -> list[str]:
    """Add ``root`` to the roots recorded for ``key``; return the whole set, sorted.

    A set union, never a removal: a repository that has since
    moved stays listed, because the beliefs harvested from it still carry the
    key. Two or more roots mean two repositories whose paths differ only in
    punctuation share one key, and so one observer scope. The file is written
    owner-only: it names the operator's repositories.
    """
    if not _PROJECT_KEY_RE.fullmatch(key):
        return []
    known = recorded_project_roots(key)
    roots = sorted({*known, str(root)})
    if roots != known:
        write_private_text(project_roots_path(key), json.dumps(roots, indent=2) + "\n")
    return roots


def project_key_collision_note(key: str, roots: Sequence[str]) -> str:
    """The one-sentence disclosure for a key seen from more than one root."""
    return (
        f"project key {key} is shared by {len(roots)} repositories ({', '.join(roots)}): "
        "their beliefs share one project scope, because Claude Code keeps one memory "
        "directory for paths that differ only in punctuation"
    )


def memory_snapshot_path(memory_dir: Path | None = None) -> Path:
    """The render snapshot of one project's region.

    Because the bullets renderer is deterministic, snapshot ≡ last-spliced
    region body — the pristine test at harvest and the drift check before a
    re-splice both compare against this file. It is kept **per project**: each
    cycle splices only the ending session's ``MEMORY.md``, so a single
    snapshot describes one file and makes every other project's region look
    hand-edited, which deposits the store's own bullets as authored input.

    With no ``memory_dir`` this is the machine-wide path older versions wrote,
    ``<state_dir>/memory-index.snapshot.md``. It is still read as a fallback
    for a project that has not been rendered since the upgrade, and never
    written again.
    """
    name = f"{MEMORY_REGION}.snapshot.md"
    return (project_state_dir(memory_dir) if memory_dir is not None else state_dir()) / name


def memory_archive_path(memory_dir: Path | None = None) -> Path:
    """One project's fold-and-archive target.

    Per project, like the snapshot: lines folded out of one project's
    ``MEMORY.md`` are that project's memory, and a single shared archive is
    harvested under whichever project deposited it first. With
    no ``memory_dir`` this is the machine-wide file older versions appended to,
    which is left in place and no longer written or harvested.
    """
    base = project_state_dir(memory_dir) if memory_dir is not None else state_dir()
    return base / MEMORY_ARCHIVE_NAME


def memory_backup_path(memory_dir: Path) -> Path:
    """One project's one-deep pre-splice backup.

    Per project for the same reason as the snapshot: a single machine-wide
    backup holds whichever project's file was spliced last, so it is the wrong
    file to restore for every other project.
    """
    return project_state_dir(memory_dir) / "MEMORY.md.pre-render"


def projection_enabled() -> bool:
    """Whether the MEMORY.md projection is active: enabled in config *and* the
    manifest exists (``particles init claude-code`` writes it)."""
    return get_config().agent_memory.projection.enabled and memory_manifest_path().is_file()


# ---------------------------------------------------------------------------
# Harness binding — which store may write into a memory directory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookBinding:
    """One installed Particles hook command, reduced to the store it names.

    ``location`` is the comparable identity of the pinned database (see
    :func:`store_location`), or ``None`` when the command pins nothing that
    resolves independently of the working directory: a pre-v1.70.2 install, or
    a pinned config whose DSN is itself relative. ``engine_url`` is the
    engine a thin-client hook names: ``PARTICLES_ENGINE_BASE_URL``
    when the command pins it, otherwise the pinned config's ``engine.base_url``,
    normalised by :func:`engine_location`.
    """

    settings_path: Path
    handle: str
    location: str | None
    engine_url: str | None = None


def engine_location(base_url: str | None) -> str | None:
    """A comparable identity for an engine URL: stripped, no trailing slash."""
    if not base_url or not base_url.strip():
        return None
    return base_url.strip().rstrip("/")


def store_location(dsn: str, base_dir: Path | None = None) -> str | None:
    """A comparable identity for a database URL, or ``None`` when it has none.

    A file-backed SQLite URL is identified by the real path of its file, so
    ``sqlite:///./particles.db`` run from the store's directory and the
    four-slash absolute pin ``init claude-code`` bakes into the hooks compare
    equal. A relative path is resolved against ``base_dir`` (SQLAlchemy resolves
    it against the process working directory) and is unidentifiable without
    one. An in-memory SQLite URL identifies nothing. Any other URL is compared
    verbatim.
    """
    if dsn.startswith("sqlite"):
        raw = sqlite_file_path(dsn)
        if raw is None:
            return None
        path = Path(raw)
        if not path.is_absolute():
            if base_dir is None:
                return None
            path = base_dir / path
        return f"sqlite:{os.path.realpath(path)}"
    return dsn.strip() or None


def _pinned_config_tree(config_path: Path) -> dict[str, Any] | None:
    try:
        tree = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    return tree if isinstance(tree, dict) else None


def _pinned_config_engine_url(config_path: Path) -> str | None:
    """The ``engine.base_url`` of the config file a hook command pins."""
    tree = _pinned_config_tree(config_path)
    engine = tree.get("engine") if tree is not None else None
    value = engine.get("base_url") if isinstance(engine, dict) else None
    return value if isinstance(value, str) else None


def _pinned_config_dsn(config_path: Path, handle: str) -> str | None:
    """The DSN ``handle`` resolves to in the config file a hook command pins."""
    tree = _pinned_config_tree(config_path)
    if tree is None:
        return None
    storage = tree.get("storage") if isinstance(tree, dict) else None
    storage = storage if isinstance(storage, dict) else {}
    if handle == "default":
        value = storage.get("database_url", StorageConfig().database_url)
    else:
        stores = storage.get("stores")
        value = stores.get(handle) if isinstance(stores, dict) else None
    return value if isinstance(value, str) else None


def pinned_default_dsn(env: Mapping[str, str]) -> str:
    """The DSN the ``default`` store resolves to in a process started with ``env``'s pins.

    ``DATABASE_URL`` wins; otherwise the ``storage.database_url`` of the config
    the process reads (its ``PARTICLES_CONFIG`` pin, else the user file),
    falling back to the compiled ``./particles.db``. ``hook doctor`` uses it to
    flag an entry whose ``default`` store depends on its working directory
    (security review N17).
    """
    if "DATABASE_URL" in env:
        return env["DATABASE_URL"]
    pin = env.get("PARTICLES_CONFIG")
    dsn = _pinned_config_dsn(Path(pin) if pin else user_config_path(), "default")
    return dsn if dsn is not None else StorageConfig().database_url


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def command_env_pins(command: str) -> dict[str, str]:
    """The ``env KEY=VALUE …`` pins leading an installed hook command."""
    tokens = _command_tokens(command)
    env: dict[str, str] = {}
    if tokens and tokens[0] == "env":
        for token in tokens[1:]:
            key, sep, value = token.partition("=")
            if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                break
            env[key] = value
    return env


def parse_hook_binding(command: str, settings_path: Path) -> HookBinding:
    """Reduce one installed hook command to the store it names.

    The command is ``[env KEY=VALUE …] <particles> hook <verb> --store <handle>``.
    The ``default`` store is pinned by ``DATABASE_URL``; a named store by the
    ``storage.stores`` entry of the pinned ``PARTICLES_CONFIG``.
    """
    tokens = _command_tokens(command)
    env = command_env_pins(command)
    handle = "default"
    for i, token in enumerate(tokens[:-1]):
        if token == "--store":
            handle = tokens[i + 1]
    dsn: str | None = None
    if handle == "default" and "DATABASE_URL" in env:
        dsn = env["DATABASE_URL"]
    elif "PARTICLES_CONFIG" in env:
        dsn = _pinned_config_dsn(Path(env["PARTICLES_CONFIG"]), handle)
    location = store_location(dsn) if dsn is not None else None
    engine_url = env.get("PARTICLES_ENGINE_BASE_URL")
    if engine_url is None and "PARTICLES_CONFIG" in env:
        engine_url = _pinned_config_engine_url(Path(env["PARTICLES_CONFIG"]))
    return HookBinding(
        settings_path=settings_path,
        handle=handle,
        location=location,
        engine_url=engine_location(engine_url),
    )


#: Files Claude Code keeps in its configuration directory beside ``projects/``.
#: Any one of them marks a directory as a harness root whatever it is named.
_CLAUDE_CONFIG_MARKERS = ("settings.json", "settings.local.json", "history.jsonl", ".claude.json")


def _claude_projects_root_of(memory_dir: Path) -> Path | None:
    """The ``<config>/projects`` root holding ``memory_dir``, or ``None``.

    Decided by shape, not by ``$HOME``: a test or script that swapped ``HOME``
    and still reached the real directory is caught the same way. The
    configuration directory is recognised by more than its name, since Claude
    Code can keep it elsewhere (``CLAUDE_CONFIG_DIR``): a ``.claude`` directory,
    the directory that variable names, or one holding a Claude Code
    configuration file. A guard that recognised only the literal ``.claude``
    name failed open for every other location (security review D28).
    """
    try:
        resolved = memory_dir.resolve()
    except OSError:
        return None
    projects = resolved.parent.parent
    if resolved.name != "memory" or projects.name != "projects":
        return None
    config_dir = projects.parent
    if config_dir.name == ".claude" or _is_configured_claude_dir(config_dir):
        return projects
    if any((config_dir / marker).is_file() for marker in _CLAUDE_CONFIG_MARKERS):
        return projects
    return None


def _is_configured_claude_dir(path: Path) -> bool:
    """Whether ``path`` is the directory ``CLAUDE_CONFIG_DIR`` names.

    The variable is Claude Code's, not a Particles setting: it says where the
    harness this module guards keeps its files, the way ``HOME`` does.
    """
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    if not configured:
        return False
    try:
        return Path(configured).expanduser().resolve() == path
    except OSError:
        return False


def _settings_bindings(path: Path) -> list[HookBinding]:
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(settings, dict):
        return []
    return [parse_hook_binding(cmd, path) for cmd in particles_hook_commands(settings)]


def _project_roots_of(memory_dir: Path) -> list[Path]:
    """Repositories whose sessions write to ``memory_dir``, from their transcripts."""
    project_dir = memory_dir.parent
    roots: list[Path] = []
    for transcript in sorted(project_dir.glob("*.jsonl")):
        launch = launch_directory(project_dir.name, _recorded_cwds(transcript))
        if launch is not None:
            root = repository_root(launch)
            if root not in roots:
                roots.append(root)
            break
    return roots


def hook_bindings_for(memory_dir: Path, project_roots: tuple[Path, ...] = ()) -> list[HookBinding]:
    """Every installed Particles hook whose sessions render into ``memory_dir``.

    The user-level ``settings.json`` beside the projects root covers every
    project; a project-scope install (``.claude/settings.local.json``, or a
    hand-written ``.claude/settings.json``) covers its own repository.
    ``project_roots`` adds repositories the caller already knows (the SessionEnd
    hook knows the session's working directory); otherwise they are recovered
    from the project's transcripts.
    """
    projects_root = _claude_projects_root_of(memory_dir)
    if projects_root is None:
        return []
    bindings = _settings_bindings(projects_root.parent / "settings.json")
    roots = list(project_roots) or _project_roots_of(memory_dir)
    for root in roots:
        for name in ("settings.local.json", "settings.json"):
            bindings.extend(_settings_bindings(root / ".claude" / name))
    return bindings


def projection_refusal(
    store: str, memory_dir: Path, project_roots: tuple[Path, ...] = ()
) -> str | None:
    """Why ``store`` may not harvest or render ``memory_dir``, or ``None`` when it may.

    A Claude Code memory directory is served by the store the installed
    Particles hooks name, and only that store: every session in the project
    loads what is rendered there as facts about the user. A scratch store (a
    test, a scenario script, a ``DATABASE_URL`` pointed elsewhere) is refused,
    so it can neither write its beliefs into the file nor ingest the file's
    content. A directory outside any ``.claude/projects`` tree is not a harness
    directory and is never refused here.
    """
    if _claude_projects_root_of(memory_dir) is None:
        return None
    engine_url = engine_location(get_config().engine.base_url)
    if engine_url is not None:
        return _remote_projection_refusal(store, engine_url, memory_dir, project_roots)
    storage = get_config().storage
    dsn = storage.database_url if store == "default" else storage.stores.get(store)
    current = store_location(dsn, Path.cwd()) if dsn is not None else None
    if current is None:
        return f"store {store!r} has no database location a Claude Code hook could name"
    bindings = hook_bindings_for(memory_dir, project_roots)
    if not bindings:
        return (
            f"no Particles hook is installed for {memory_dir.parent.name}; "
            "`particles init claude-code` installs one"
        )
    if any(b.location == current for b in bindings):
        return None
    if all(b.location is None for b in bindings):
        return (
            "the installed Particles hooks do not pin a database; "
            "re-run `particles init claude-code` to pin one"
        )
    return (
        f"store {store!r} is not the store the installed Claude Code hooks name "
        f"({', '.join(sorted({str(b.settings_path) for b in bindings}))})"
    )


def _remote_projection_refusal(
    store: str, engine_url: str, memory_dir: Path, project_roots: tuple[Path, ...]
) -> str | None:
    """:func:`projection_refusal` for a thin client.

    A remote store has no local database file, so the binding is the engine and
    the handle instead: an installed hook must name this store and pin this
    same ``engine.base_url``, through ``PARTICLES_ENGINE_BASE_URL`` or its
    pinned ``PARTICLES_CONFIG``. A scratch config pointed at another engine, or
    at this engine's other store, is refused as a scratch database is locally.
    """
    bindings = hook_bindings_for(memory_dir, project_roots)
    if not bindings:
        return (
            f"no Particles hook is installed for {memory_dir.parent.name}; "
            "`particles init claude-code` installs one"
        )
    if any(b.handle == store and b.engine_url == engine_url for b in bindings):
        return None
    if all(b.engine_url is None for b in bindings):
        return (
            "the installed Particles hooks do not pin an engine; pin "
            "PARTICLES_ENGINE_BASE_URL or a PARTICLES_CONFIG naming engine.base_url "
            "in each hook command"
        )
    return (
        f"store {store!r} on {engine_url} is not the store the installed Claude Code "
        f"hooks name ({', '.join(sorted({str(b.settings_path) for b in bindings}))})"
    )


def default_memory_manifest_text() -> str:
    """The zero-config default ``memory.yaml`` — one ranked list.

    A standard manifest: the operator may edit it, add sections, pin
    with ``select.allow/deny``, and register it in
    ``curation.projection_manifests``. The numeric defaults mirror
    ``agent_memory.projection`` at generation time; afterwards the manifest is
    the single tuning surface (it wins where both speak).
    """
    proj = get_config().agent_memory.projection
    return (
        "# MEMORY.md projection manifest — written by\n"
        "# `particles init claude-code`; yours to edit. Add sections, pin claims\n"
        "# with select.allow/deny, or tighten the floor. Docs:\n"
        "# docs/user-guide/claude-code.md § The MEMORY.md projection.\n"
        f"name: {MEMORY_REGION}\n"
        "sections:\n"
        '  - title: "Memory index"\n'
        "    query: null            # no semantic refinement — rank purely by eff. conf.\n"
        "    top_k: 60\n"
        f"    min_confidence: {proj.min_confidence:.2f}   # noise floor (the R1.8 lesson)\n"
        "    render: bullets        # deterministic ranked bullets — never LLM prose\n"
        f"max_lines: {proj.max_lines}             "
        "# document budget — headroom under the 200-line load cap\n"
        "max_bytes: 16384\n"
    )


def load_projection_snapshots(memory_dir: Path | None = None) -> dict[str, str]:
    """Region-name → last-rendered-body map from the state directory's snapshots.

    Reads every ``*.snapshot.md`` beside the manifest state (the
    ``<name>.snapshot.md`` convention, relocated); the stem
    before ``.snapshot`` is the region name. Unreadable files are skipped —
    the strip then treats their regions as dirtied, which routes the content
    through the harvest ladder rather than risk dropping an edit.

    With ``memory_dir``, that project's own snapshots are layered over the
    machine-wide ones and win: a file under ``memory_dir`` is compared with
    what was last spliced into *it*.
    """
    directories = [state_dir()]
    if memory_dir is not None:
        directories.append(project_state_dir(memory_dir))
    snapshots: dict[str, str] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.snapshot.md")):
            try:
                snapshots[path.name[: -len(".snapshot.md")]] = path.read_text(encoding="utf-8")
            except OSError:
                log.debug("could not read snapshot %s", path, exc_info=True)
    return snapshots


# ---------------------------------------------------------------------------
# Memory-file pre-deposit filter — the sentinel strip
# ---------------------------------------------------------------------------


#: One Claude Code memory-index entry: a list item that is only a link to a
#: sibling Markdown file, optionally followed by a dash or colon and a short
#: description of that file (``- [Title](topic.md) — what it covers``).
_INDEX_ENTRY_RE = re.compile(
    r"^\s*[-*+]\s+\[[^\]]+\]\((?P<target>[^()\s/\\]+\.md)\)\s*(?:[—–:-].*)?$"
)


def index_entry_target(line: str) -> str | None:
    """The file a memory-index line links to, or ``None`` when ``line`` is not one.

    The shape is Claude Code's ``- [Title](name.md) — hook`` index line, a
    bare filename with no directory part. Whether the target exists is the
    caller's question.
    """
    m = _INDEX_ENTRY_RE.match(line)
    return m.group("target") if m is not None else None


def memory_file_type(path: Path) -> str | None:
    """A memory topic file's type from its frontmatter, lower-cased, or ``None``.

    Claude Code writes it as ``metadata.type`` (``feedback``, ``user``,
    ``project``, ``reference``); a top-level ``type`` is the older spelling and
    is read when ``metadata`` carries none. An unreadable file, a missing or
    malformed frontmatter block, or a non-string type is ``None``: no type,
    never an error.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    lines = text.splitlines()
    if not lines or lines[0].rstrip() != "---":
        return None
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].rstrip() == "---")
        header = yaml.safe_load("\n".join(lines[1:end]))
    except (StopIteration, yaml.YAMLError):
        return None
    if not isinstance(header, dict):
        return None
    metadata = header.get("metadata")
    value = metadata.get("type") if isinstance(metadata, dict) else None
    if value is None:
        value = header.get("type")
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


#: Claude Code loads the first 200 lines / 25 KB of ``MEMORY.md`` at session
#: start; anything below is never in context.
MEMORY_LOAD_MAX_LINES = 200
MEMORY_LOAD_MAX_BYTES = 25_000
#: The share of either limit at which ``hook doctor`` starts warning.
MEMORY_LOAD_WARN_FRACTION = 0.9


def memory_load_window_warning(text: str) -> str | None:
    """A one-line warning when ``text`` nears the ``MEMORY.md`` load window, else ``None``.

    The index pointers sit below the projected region, so a file that grows
    past the window loses its tail without any error.
    """
    lines = len(text.splitlines())
    size = len(text.encode("utf-8"))
    if (
        lines < MEMORY_LOAD_MAX_LINES * MEMORY_LOAD_WARN_FRACTION
        and size < MEMORY_LOAD_MAX_BYTES * MEMORY_LOAD_WARN_FRACTION
    ):
        return None
    return (
        f"⚠ MEMORY.md is {lines} lines / {size:,} bytes; Claude Code loads only the "
        f"first {MEMORY_LOAD_MAX_LINES} lines / {MEMORY_LOAD_MAX_BYTES:,} bytes, "
        "so the end of the index may not reach a session."
    )


def drop_index_entries(text: str, source_dir: Path) -> str:
    """Remove memory-index lines whose target file sits beside the indexed file.

    A Claude Code ``MEMORY.md`` is an index: one link per topic file with a
    few words on what the file covers. Those words are a summary of a file the
    harvest deposits in full, so extracting them only produces a second,
    vaguer claim ("the project has open work") that can then contradict the
    topic file it summarises. A line whose target does not exist is
    kept: its description is then the only record of what it says. Pure apart
    from the existence check.
    """
    kept: list[str] = []
    dropped = False
    for line in text.splitlines():
        target = index_entry_target(line)
        if target is not None and (source_dir / target).is_file():
            dropped = True
            continue
        kept.append(line)
    if not dropped:
        return text
    return "\n".join(kept) + ("\n" if text.endswith("\n") else "")


def filter_memory_file_for_deposit(
    text: str,
    snapshot_bodies: Mapping[str, str] | None = None,
    *,
    memory_dir: Path | None = None,
    source_dir: Path | None = None,
) -> str:
    """Pre-deposit filter for harvested memory files.

    Strips every **pristine** ``PROJECTED`` sentinel region — one whose body
    still matches its render snapshot — before deposit: the corpus must never
    contain the store's own rendered output (belt 1 of the round-trip
    contract; the fixed-point test rests on it). A **dirtied** region (body ≠
    snapshot, or no snapshot known) is human/agent signal: its body is kept
    and deposited as ordinarily-authored input for the §6.6 ladder;
    only the sentinel comment lines are dropped.

    The fold-and-archive pointer line is dropped too — it is
    machine-generated, like the sentinels, and the archive file it points at
    is harvested in full anyway. Credential shapes are masked last
    (:func:`redact_secrets`), as for transcripts; a file with one masked is no
    longer its own bytes, so the caller's ``refresh_policy_for`` keeps it out
    of the byte-level refresh loop, which would deposit it raw.

    ``source_dir`` is the folder the file itself sits in. When given, its
    memory-index entries are dropped as well (:func:`drop_index_entries`):
    each summarises a sibling file that is harvested in full.

    ``snapshot_bodies`` (region name → body) defaults to the snapshots of the
    project that owns ``memory_dir`` — pass it for every file harvested from a
    memory directory, or a region last rendered before some *other* project's
    cycle reads as dirtied. The parse regexes are shared with
    the renderer (``particles.render.markdown``), so strip and splice can
    never disagree about where a region begins.
    """
    from particles.render.markdown import strip_projected_regions_for_deposit

    if snapshot_bodies is None:
        snapshot_bodies = load_projection_snapshots(memory_dir)
    stripped = strip_projected_regions_for_deposit(text, snapshot_bodies)
    if ARCHIVE_POINTER_PREFIX in stripped:
        kept = [
            line for line in stripped.splitlines() if not line.startswith(ARCHIVE_POINTER_PREFIX)
        ]
        stripped = "\n".join(kept) + ("\n" if stripped.endswith("\n") else "")
    if source_dir is not None:
        stripped = drop_index_entries(stripped, source_dir)
    return redact_secrets(stripped)


# ---------------------------------------------------------------------------
# Digest byte budget
# ---------------------------------------------------------------------------


def truncate_on_line_boundary(text: str, max_bytes: int) -> str:
    """Enforce the ``claude_code.digest_max_bytes`` budget (0 = no cap).

    Truncates on a line boundary and appends a disclosed footer; the footer is
    budgeted for, so the result (footer included) never exceeds ``max_bytes``
    unless the budget is too small to hold the footer alone.
    """
    kept = kept_prefix_length(text, max_bytes)
    if kept == len(text):
        return text
    return text[:kept].rstrip("\n") + _truncation_footer(max_bytes)


def _truncation_footer(max_bytes: int) -> str:
    return f"\n\n*[digest truncated at {max_bytes} bytes — see `particles query` for the rest]*\n"


def kept_prefix_length(text: str, max_bytes: int) -> int:
    """How many characters of ``text`` :func:`truncate_on_line_boundary` keeps.

    ``len(text)`` when the text fits the budget. Otherwise the length of the
    whole lines that fit beside the footer. A line of ``text`` survived the cut
    exactly when it starts before this offset.
    """
    if max_bytes <= 0 or len(text.encode("utf-8")) <= max_bytes:
        return len(text)
    budget = max_bytes - len(_truncation_footer(max_bytes).encode("utf-8"))
    kept = 0
    used = 0
    for line in text.splitlines(keepends=True):
        line_bytes = len(line.encode("utf-8"))
        if used + line_bytes > budget:
            break
        kept += len(line)
        used += line_bytes
    return kept


# ---------------------------------------------------------------------------
# config.yaml surgeon (fresh-install store auto-create)
# ---------------------------------------------------------------------------
#
# The store-enable edit must parse-preserve-append: parse, preserve everything
# else, append the two keys, never clobber. PyYAML cannot round-trip comments
# or formatting, so the edit is *textual* (block-scoped line surgery) and then
# **verified**: the edited text is re-parsed and compared against the expected
# tree (the old tree plus exactly the intended modifications). Any structure
# the surgeon does not confidently understand raises ConfigEditError instead
# of guessing — an unparseable or unusual config is an error with
# instructions, never a rewrite.

_INIT_MARKER_COMMENT = "# Added by `particles init claude-code`"


class ConfigEditError(ValueError):
    """The config.yaml edit could not be performed safely (never-clobber rule)."""


def _parse_yaml_mapping(yaml_text: str) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(yaml_text) if yaml_text.strip() else None
    except yaml.YAMLError as exc:
        raise ConfigEditError(f"config.yaml is not parseable YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigEditError("config.yaml top level is not a mapping")
    return loaded


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _top_level_key_line(lines: list[str], key: str) -> int | None:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(#.*)?$")
    for i, line in enumerate(lines):
        if pattern.match(line):
            return i
    return None


def _block_end(lines: list[str], key_idx: int) -> int:
    """Index one past the last line belonging to the block opened at ``key_idx``.

    A comment line indented at or left of the key terminates the block (it may
    describe whatever follows, so it must never be swept up by a block
    deletion); a deeper-indented comment belongs to the block. Trailing blank
    lines are excluded either way.
    """
    key_indent = _indent_of(lines[key_idx])
    end = key_idx + 1
    for i in range(key_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        if _indent_of(line) <= key_indent:
            break
        end = i + 1
    return end


def _child_key_line(lines: list[str], parent_idx: int, key: str) -> int | None:
    """Find ``key:`` as a block-mapping child of the key opened at ``parent_idx``."""
    pattern = re.compile(rf"^(\s+){re.escape(key)}:\s*(#.*)?$")
    for i in range(parent_idx + 1, _block_end(lines, parent_idx)):
        if pattern.match(lines[i]):
            return i
    return None


def _child_indent(lines: list[str], key_idx: int, default: int) -> int:
    """Indent of the first effective child of the block at ``key_idx`` (or default)."""
    for i in range(key_idx + 1, _block_end(lines, key_idx)):
        line = lines[i]
        if line.strip() and not line.lstrip().startswith("#"):
            return _indent_of(line)
    return default


def _verify(new_text: str, expected: dict[str, Any]) -> str:
    reparsed = _parse_yaml_mapping(new_text)
    if reparsed != expected:
        raise ConfigEditError(
            "the edited config.yaml did not verify against the expected result — "
            "the file's layout is unusual; make the change by hand instead"
        )
    return new_text


def _insert_after(lines: list[str], idx: int, new_lines: list[str]) -> list[str]:
    return lines[: idx + 1] + new_lines + lines[idx + 1 :]


def _expected_after_enable(tree: dict[str, Any], handle: str, dsn: str) -> dict[str, Any]:
    expected = copy.deepcopy(tree)
    expected.setdefault("storage", {}).setdefault("stores", {})[handle] = dsn
    write = expected.setdefault("mcp", {}).setdefault("write", {})
    enabled = write.setdefault("enabled_stores", [])
    if handle not in enabled:
        enabled.append(handle)
    return expected


def enable_memory_store_text(yaml_text: str, handle: str, dsn: str) -> str:
    """Append ``storage.stores.<handle>`` + ``mcp.write.enabled_stores`` to config.yaml text.

    Parse-preserve-append: existing content — including comments
    and formatting — is untouched; only the two keys are added, and the result
    is verified by re-parse against the expected tree. Raises
    :class:`ConfigEditError` when the file is unparseable or shaped in a way
    the surgeon cannot edit safely (flow-style sections, aliases, …); the
    caller turns that into an actionable message, never a rewrite.
    """
    tree = _parse_yaml_mapping(yaml_text)

    storage = tree.get("storage")
    stores = storage.get("stores") if isinstance(storage, dict) else None
    if isinstance(stores, dict) and handle in stores and stores[handle] != dsn:
        raise ConfigEditError(
            f"store {handle!r} is already configured with a different DSN "
            f"({stores[handle]!r}); refusing to overwrite it"
        )

    expected = _expected_after_enable(tree, handle, dsn)
    if tree == expected:
        return yaml_text  # both keys already present — idempotent no-op

    lines = yaml_text.splitlines()

    # -- storage.stores.<handle> ------------------------------------------
    if not (isinstance(stores, dict) and stores.get(handle) == dsn):
        if "storage" not in tree:
            lines += [
                "",
                _INIT_MARKER_COMMENT,
                "storage:",
                "  stores:",
                f"    {handle}: {dsn}",
            ]
        else:
            storage_idx = _top_level_key_line(lines, "storage")
            if storage_idx is None:
                raise ConfigEditError("could not locate the 'storage:' block")
            stores_idx = _child_key_line(lines, storage_idx, "stores")
            if stores_idx is None:
                if isinstance(stores, dict) and stores:
                    raise ConfigEditError("could not locate the 'storage.stores:' block")
                child = _child_indent(lines, storage_idx, 2)
                lines = _insert_after(
                    lines,
                    storage_idx,
                    [
                        f"{' ' * child}stores:  {_INIT_MARKER_COMMENT}",
                        f"{' ' * (child + 2)}{handle}: {dsn}",
                    ],
                )
            else:
                item = _child_indent(lines, stores_idx, _indent_of(lines[stores_idx]) + 2)
                lines = _insert_after(
                    lines,
                    stores_idx,
                    [f"{' ' * item}{handle}: {dsn}  {_INIT_MARKER_COMMENT}"],
                )

    # -- mcp.write.enabled_stores ------------------------------------------
    mcp = tree.get("mcp")
    write = mcp.get("write") if isinstance(mcp, dict) else None
    enabled = write.get("enabled_stores") if isinstance(write, dict) else None
    if not (isinstance(enabled, list) and handle in enabled):
        if "mcp" not in tree:
            lines += [
                "",
                _INIT_MARKER_COMMENT,
                "mcp:",
                "  write:",
                "    enabled_stores:",
                f"      - {handle}",
            ]
        else:
            mcp_idx = _top_level_key_line(lines, "mcp")
            if mcp_idx is None:
                raise ConfigEditError("could not locate the 'mcp:' block")
            write_idx = _child_key_line(lines, mcp_idx, "write")
            if write_idx is None:
                if isinstance(write, dict) and write:
                    raise ConfigEditError("could not locate the 'mcp.write:' block")
                child = _child_indent(lines, mcp_idx, 2)
                lines = _insert_after(
                    lines,
                    mcp_idx,
                    [
                        f"{' ' * child}write:  {_INIT_MARKER_COMMENT}",
                        f"{' ' * (child + 2)}enabled_stores:",
                        f"{' ' * (child + 4)}- {handle}",
                    ],
                )
            else:
                lines = _add_enabled_store_item(lines, write_idx, handle)

    return _verify("\n".join(lines) + "\n", expected)


def _enabled_stores_line(lines: list[str], write_idx: int) -> int | None:
    """Locate the ``enabled_stores`` key line inside the ``write:`` block."""
    pattern = re.compile(r"^\s+enabled_stores:")
    for i in range(write_idx + 1, _block_end(lines, write_idx)):
        if pattern.match(lines[i]):
            return i
    return None


def _add_enabled_store_item(lines: list[str], write_idx: int, handle: str) -> list[str]:
    """Add ``handle`` to ``mcp.write.enabled_stores`` under the ``write:`` block."""
    key_idx = _enabled_stores_line(lines, write_idx)
    if key_idx is None:
        # No enabled_stores key inside write: — add one.
        child = _child_indent(lines, write_idx, _indent_of(lines[write_idx]) + 2)
        return _insert_after(
            lines,
            write_idx,
            [
                f"{' ' * child}enabled_stores:  {_INIT_MARKER_COMMENT}",
                f"{' ' * (child + 2)}- {handle}",
            ],
        )
    m = re.match(r"^(\s+)enabled_stores:\s*(.*?)\s*(#.*)?$", lines[key_idx])
    assert m is not None  # _enabled_stores_line matched the prefix
    indent, value, comment = m.group(1), m.group(2), m.group(3) or ""
    if value:  # flow style: enabled_stores: [] or [a, b]
        try:
            flow = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ConfigEditError("could not parse the 'enabled_stores' value") from exc
        if not isinstance(flow, list):
            raise ConfigEditError("'enabled_stores' is not a list")
        flow.append(handle)
        rendered = "[" + ", ".join(str(x) for x in flow) + "]"
        suffix = f"  {comment}" if comment else f"  {_INIT_MARKER_COMMENT}"
        lines[key_idx] = f"{indent}enabled_stores: {rendered}{suffix}"
        return lines
    # Block style: append an item at the end of the block (matching the
    # expected tree's list-append), at the existing items' indent if any.
    item = _child_indent(lines, key_idx, len(indent) + 2)
    return _insert_after(
        lines, _block_end(lines, key_idx) - 1, [f"{' ' * item}- {handle}  {_INIT_MARKER_COMMENT}"]
    )


def minimal_config_text() -> str:
    """The config file ``init claude-code`` writes when there is none to pin.

    A comment header and no settings: the installer pins a file that exists, so
    the hooks name their config rather than relying on resolution, and an
    operator who later adds a setting edits the file the hooks already read.
    """
    return (
        "# Created by `particles init claude-code`.\n"
        "# Particles reads this file when PARTICLES_CONFIG is unset, so a setting here\n"
        "# applies to every Particles process run without its own PARTICLES_CONFIG.\n"
    )


def fresh_config_text(handle: str, dsn: str) -> str:
    """A brand-new config.yaml enabling the auto-created memory store."""
    return (
        "# Created by `particles init claude-code`.\n"
        f"# The `{handle}` store holds agent memory harvested from Claude Code sessions.\n"
        "storage:\n"
        "  stores:\n"
        f"    {handle}: {dsn}\n"
        "\n"
        "mcp:\n"
        "  write:\n"
        "    enabled_stores:\n"
        f"      - {handle}\n"
    )


def _expected_after_disable(tree: dict[str, Any], handle: str) -> dict[str, Any]:
    expected = copy.deepcopy(tree)
    storage = expected.get("storage")
    if isinstance(storage, dict) and isinstance(storage.get("stores"), dict):
        storage["stores"].pop(handle, None)
        if not storage["stores"]:
            del storage["stores"]
        if not storage:
            del expected["storage"]
    mcp = expected.get("mcp")
    write = mcp.get("write") if isinstance(mcp, dict) else None
    if isinstance(mcp, dict) and isinstance(write, dict):
        enabled = write.get("enabled_stores")
        if isinstance(enabled, list) and handle in enabled:
            remaining = [x for x in enabled if x != handle]
            if remaining:
                write["enabled_stores"] = remaining
            else:
                del write["enabled_stores"]
            if not write:
                del mcp["write"]
            if not mcp:
                del expected["mcp"]
    return expected


def _block_is_effectively_empty(lines: list[str], key_idx: int) -> bool:
    """True when the block opened at ``key_idx`` holds only blanks / comments."""
    for i in range(key_idx + 1, _block_end(lines, key_idx)):
        line = lines[i]
        if line.strip() and not line.lstrip().startswith("#"):
            return False
    return True


def _delete_block_if_empty(lines: list[str], key_idx: int) -> bool:
    """Delete the key line (plus any orphaned marker comments in its block) if empty."""
    if not _block_is_effectively_empty(lines, key_idx):
        return False
    del lines[key_idx : _block_end(lines, key_idx)]
    return True


def disable_memory_store_text(yaml_text: str, handle: str) -> str:
    """Reverse :func:`enable_memory_store_text` — remove the two keys, keep the rest.

    Removes ``storage.stores.<handle>`` and the ``<handle>`` item of
    ``mcp.write.enabled_stores``, pruning containers *that the removal
    emptied* and any marker comment lines the installer added. The surgery is
    block-scoped (never a whole-file pattern sweep) and verified by re-parse
    against the expected tree; raises :class:`ConfigEditError` when it cannot
    be performed safely — the caller prints instructions, never rewrites.
    """
    tree = _parse_yaml_mapping(yaml_text)
    expected = _expected_after_disable(tree, handle)
    if expected == tree:
        return yaml_text  # nothing to remove

    lines = yaml_text.splitlines()

    # -- storage.stores.<handle> ------------------------------------------
    storage = tree.get("storage")
    stores = storage.get("stores") if isinstance(storage, dict) else None
    if isinstance(stores, dict) and handle in stores:
        storage_idx = _top_level_key_line(lines, "storage")
        if storage_idx is None:
            raise ConfigEditError("could not locate the 'storage:' block")
        stores_idx = _child_key_line(lines, storage_idx, "stores")
        if stores_idx is None:
            raise ConfigEditError("could not locate the 'storage.stores:' block")
        entry_re = re.compile(rf"^\s+{re.escape(handle)}:\s+\S")
        entry_idx = next(
            (
                i
                for i in range(stores_idx + 1, _block_end(lines, stores_idx))
                if entry_re.match(lines[i])
            ),
            None,
        )
        if entry_idx is None:
            raise ConfigEditError(f"could not locate the {handle!r} store entry")
        del lines[entry_idx]
        if _delete_block_if_empty(lines, stores_idx):
            storage_idx = _top_level_key_line(lines, "storage")
            if storage_idx is not None:
                _delete_block_if_empty(lines, storage_idx)

    # -- mcp.write.enabled_stores ------------------------------------------
    mcp = tree.get("mcp")
    write = mcp.get("write") if isinstance(mcp, dict) else None
    enabled = write.get("enabled_stores") if isinstance(write, dict) else None
    if isinstance(enabled, list) and handle in enabled:
        mcp_idx = _top_level_key_line(lines, "mcp")
        if mcp_idx is None:
            raise ConfigEditError("could not locate the 'mcp:' block")
        write_idx = _child_key_line(lines, mcp_idx, "write")
        if write_idx is None:
            raise ConfigEditError("could not locate the 'mcp.write:' block")
        key_idx = _enabled_stores_line(lines, write_idx)
        if key_idx is None:
            raise ConfigEditError("could not locate 'mcp.write.enabled_stores'")
        m = re.match(r"^(\s+)enabled_stores:\s*(.*?)\s*(#.*)?$", lines[key_idx])
        assert m is not None
        indent, value = m.group(1), m.group(2)
        if value:  # flow style
            try:
                flow = yaml.safe_load(value)
            except yaml.YAMLError as exc:
                raise ConfigEditError("could not parse the 'enabled_stores' value") from exc
            if not isinstance(flow, list):
                raise ConfigEditError("'enabled_stores' is not a list")
            remaining = [x for x in flow if x != handle]
            if remaining:
                rendered = "[" + ", ".join(str(x) for x in remaining) + "]"
                lines[key_idx] = f"{indent}enabled_stores: {rendered}"
            else:
                del lines[key_idx]
        else:  # block style: delete the "- handle" item within this block
            item_re = re.compile(rf"^\s+-\s+{re.escape(handle)}\s*(#.*)?$")
            item_idx = next(
                (
                    i
                    for i in range(key_idx + 1, _block_end(lines, key_idx))
                    if item_re.match(lines[i])
                ),
                None,
            )
            if item_idx is None:
                raise ConfigEditError(f"could not locate the {handle!r} enabled_stores item")
            del lines[item_idx]
            _delete_block_if_empty(lines, key_idx)
        # Prune write:/mcp: if the removal emptied them (re-locate: indices moved).
        mcp_idx = _top_level_key_line(lines, "mcp")
        if mcp_idx is not None:
            write_idx = _child_key_line(lines, mcp_idx, "write")
            if write_idx is not None and _delete_block_if_empty(lines, write_idx):
                mcp_idx = _top_level_key_line(lines, "mcp")
            if mcp_idx is not None:
                _delete_block_if_empty(lines, mcp_idx)

    # Drop full-line marker comments the installer added, and inline marker
    # suffixes on lines that survived (e.g. a `stores:` opener the operator
    # later populated with their own entries).
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == _INIT_MARKER_COMMENT or stripped.startswith(
            "# Created by `particles init claude-code`"
        ):
            continue
        # The fresh-file header's second line describes the removed store.
        if stripped.startswith(f"# The `{handle}` store holds agent memory"):
            continue
        cleaned.append(line.replace(f"  {_INIT_MARKER_COMMENT}", ""))

    # Collapse any doubled blank lines the pruning left behind.
    pruned: list[str] = []
    for line in cleaned:
        if not line.strip() and pruned and not pruned[-1].strip():
            continue
        pruned.append(line)
    while pruned and not pruned[-1].strip():
        pruned.pop()

    return _verify(("\n".join(pruned) + "\n") if pruned else "", expected)
