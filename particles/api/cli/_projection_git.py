# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Optional git-versioned history of the projected ``MEMORY.md`` view.

The terminal, best-effort step of ``run_projection_cycle``: when
``agent_memory.projection.git.enabled`` **and** the memory directory is inside a
git repo, commit the render with a structured message (run id + ranking-delta
summary), giving operators a diffable, rollback-able history of the *view* while
the store stays truth. The store is the source of truth; this history
is a bonus.

Every git failure degrades **silently** — not a repo, detached HEAD, nothing to
commit, missing identity, unsigned, permission denied — logged at ``debug`` and
returned as a telemetry string; :func:`commit_projection` never raises. The
SDK repo's GPG-signing requirement is never imposed on the operator's repo:
signing defaults off so an unattended SessionEnd-hook commit can
never block on a signing agent.

Surface-tier (imports config only). Git is invoked via ``asyncio`` subprocess so
the async cycle never blocks on a child process.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from pathlib import Path

from particles.config import get_config
from particles.url_safety import GIT_LOCAL_ARGS, pinned_subprocess_env

log = logging.getLogger(__name__)

#: Ambient git-context env vars that would override ``git -C <memory_dir>``
#: repository discovery and make the commit target the wrong repo. The cycle
#: runs from a Claude Code SessionEnd hook (not a git hook), so these are never
#: set in production — but stripping them keeps the step correct if it ever runs
#: with a git environment inherited from a parent process (e.g. a test suite
#: driven from a pre-commit hook), so ``-C`` alone decides the repo.
_GIT_CONTEXT_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
)

#: Inherited variables that would add configuration or name a program git
#: runs, outside every config file the repository-scope check reads: a filter
#: or diff driver set through ``GIT_CONFIG_*`` would run while ``git add``
#: stages ``MEMORY.md`` (security review 2026-10-28-2, AA60). A file named by
#: ``GIT_CONFIG_GLOBAL`` or ``GIT_CONFIG_SYSTEM`` reports the global or system
#: scope the check honours, and ``GIT_ATTR_SOURCE`` picks the attributes that
#: name a driver, so those go too: the global and system config git reads is
#: then the operator's own ``~/.gitconfig`` and ``/etc/gitconfig``
#: (2026-10-30-1). Dropped from every call along with :data:`_GIT_CONTEXT_VARS`.
_GIT_PROGRAM_VARS = (
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_ATTR_SOURCE",
    "GIT_EXTERNAL_DIFF",
    "GIT_EXEC_PATH",
)
_GIT_PROGRAM_VAR_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")

#: Settings passed on every invocation so the repository's own configuration
#: cannot make this unattended step run a program: no hooks, no fsmonitor
#: daemon, no pager. The memory directory may sit inside a repository the owner
#: did not create (security review D27).
_HARDENING_CONFIG = (
    ("core.hooksPath", os.devnull),
    ("core.fsmonitor", "false"),
    ("core.pager", "cat"),
)

#: Repository settings that name a program git would run while staging or
#: diffing the memory directory. Their presence in the repository's own config
#: skips the commit: a filter or textconv driver cannot be disabled from the
#: command line without knowing its name, and the history is a bonus, never
#: worth running a stranger's command. Global and system config are the
#: operator's own (``git lfs install`` writes its filter there) and are honoured.
_COMMAND_CONFIG_RE = r"^(filter|diff|merge)\..*\.(clean|smudge|process|textconv|command|driver)$"

#: Checked only when ``agent_memory.projection.git.sign`` lets the commit sign
#: (the completeness critic's note on D27): any ``gpg.*`` setting in the
#: repository's own config skips the commit. Several of them name a program
#: (``gpg.program``, ``gpg.<format>.program``, ``gpg.ssh.defaultKeyCommand``)
#: and ``gpg.format`` picks which one runs, so signing follows the operator's
#: global choice or does not happen (security review 2026-10-25-1, Q31). The
#: repository's ``user.signingKey`` (which key signs) and ``commit.gpgSign`` /
#: ``tag.gpgSign`` count too: git reports names lower-cased (W70).
_SIGNING_CONFIG_RE = r"^(gpg\.|user\.signingkey$|commit\.gpgsign$|tag\.gpgsign$)"

#: The config scopes a repository controls. A file the repository's config
#: pulls in through ``include.path`` or ``includeIf`` reports the scope of the
#: file that included it, so it lands here too (security review 2026-10-24-2,
#: the D27 include bypass). The ``command`` scope is what the environment and
#: ``-c`` set; the environment's share is dropped before git starts, and this
#: module's own ``-c`` settings name no program (2026-10-28-2, AA60).
_REPOSITORY_SCOPES = frozenset({"local", "worktree", "command"})

#: The trailing ``p-<shortid>`` handle on a rendered memory-index bullet
#: (§4 / ``render.markdown.format_memory_bullet``). The last such
#: token on a ``- `` line is the bullet's particle handle.
_BULLET_HANDLE_RE = re.compile(r"`p-([0-9a-fA-F]+)`")


def new_run_id() -> str:
    """Mint a short (8-hex) identifier for one projection cycle.

    Also returned in the cycle's telemetry, so a commit in the operator's
    history correlates with the hook-log entry for that run. A single seam so
    tests can pin the run id and assert the commit-message shape.
    """
    return uuid.uuid4().hex[:8]


def _ordered_bullet_ids(text: str | None) -> list[str]:
    """The ranked ``p-<shortid>`` handles of a rendered region body, in order.

    One entry per ``- `` bullet line (the ranking is line order); the handle is
    the last backticked ``p-<hex>`` token on the line. Non-bullet lines (the
    sources trailer, blanks) are ignored. ``None`` / empty ⇒ ``[]``.
    """
    if not text:
        return []
    ids: list[str] = []
    for line in text.splitlines():
        if not line.lstrip().startswith("- "):
            continue
        handles = _BULLET_HANDLE_RE.findall(line)
        if handles:
            ids.append(handles[-1].lower())
    return ids


def _excerpt_for(text: str, short_id: str, *, limit: int = 80) -> str:
    """A trimmed one-line content excerpt for the bullet carrying ``short_id``."""
    for line in text.splitlines():
        handles = _BULLET_HANDLE_RE.findall(line)
        if handles and handles[-1].lower() == short_id:
            body = _BULLET_HANDLE_RE.sub("", line).lstrip("- ").strip()
            return body[:limit].rstrip() + ("…" if len(body) > limit else "")
    return ""


def format_delta(body: str, snapshot: str | None, *, max_excerpts: int) -> str:
    """The ranking-delta block for the commit message.

    Compares the ordered bullet-handle lists of the new render (``body``) against
    the previous snapshot: a count line (added / removed, plus a top-of-ranking
    change when the first belief moved) followed by up to ``max_excerpts``
    added/removed excerpt lines. The count line always states the true totals,
    so a large delta is never silently truncated. First render (no snapshot)
    reads ``initial render (N beliefs)``.
    """
    new_ids = _ordered_bullet_ids(body)
    old_ids = _ordered_bullet_ids(snapshot)
    if snapshot is None:
        return f"delta: initial render ({len(new_ids)} belief{'s' if len(new_ids) != 1 else ''})"

    old_set, new_set = set(old_ids), set(new_ids)
    added = [i for i in new_ids if i not in old_set]
    removed = [i for i in old_ids if i not in new_set]

    parts = [f"+{len(added)} −{len(removed)} beliefs"]
    old_top = old_ids[0] if old_ids else None
    new_top = new_ids[0] if new_ids else None
    if old_top != new_top:
        parts.append(f"top: p-{old_top or '∅'} → p-{new_top or '∅'}")
    if not added and not removed and old_top == new_top:
        parts = ["no ranking change"]

    lines = [f"delta: {'; '.join(parts)}"]
    excerpts: list[str] = []
    for i in added:
        excerpts.append(f"  + {_excerpt_for(body, i)} `p-{i}`")
    for i in removed:
        excerpts.append(f"  - {_excerpt_for(snapshot, i)} `p-{i}`")
    if len(excerpts) > max_excerpts:
        excerpts = excerpts[:max_excerpts]
        excerpts.append(f"  … ({len(added) + len(removed) - max_excerpts} more)")
    return "\n".join(lines + excerpts)


def build_commit_message(
    *, store: str, outcome: str, run_id: str, body: str, snapshot: str | None, max_excerpts: int
) -> str:
    """The structured projection-commit message.

    Deterministic given ``(store, outcome, run_id, body, snapshot)`` — so the
    shape is unit-testable against a tmp git repo.
    """
    delta = format_delta(body, snapshot, max_excerpts=max_excerpts)
    return (
        f"memory-projection: {outcome} memory-index ({store})\n"
        f"\n"
        f"run-id: {run_id}\n"
        f"{delta}\n"
        f"\n"
        f"Projected view of the particle store (source of truth: {store}).\n"
    )


async def _git(
    memory_dir: Path, *args: str, config: list[tuple[str, str]] | None = None
) -> tuple[int, str, str]:
    """Run ``git -C <memory_dir> [-c k=v …] <args>`` async; return (rc, stdout, stderr).

    The hardening settings (:data:`_HARDENING_CONFIG`) ride every call, ahead of
    the caller's own. The environment loses the variables that pick the
    repository or add configuration, and ``PATH`` keeps only the entries no
    other user may write, as a pinned ``git`` fetch's does.
    """
    cmd = ["git", *GIT_LOCAL_ARGS, "-C", str(memory_dir)]
    for key, value in (*_HARDENING_CONFIG, *(config or [])):
        cmd += ["-c", f"{key}={value}"]
    cmd += list(args)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _GIT_CONTEXT_VARS
        and k not in _GIT_PROGRAM_VARS
        and not k.startswith(_GIT_PROGRAM_VAR_PREFIXES)
    }
    env["PATH"] = pinned_subprocess_env({"PATH": os.environ.get("PATH", os.defpath)})["PATH"]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
    )
    out, err = await proc.communicate()
    return (
        proc.returncode or 0,
        out.decode("utf-8", "replace").strip(),
        err.decode("utf-8", "replace").strip(),
    )


async def _repository_program_refusal(memory_dir: Path, pattern: str) -> str | None:
    """A skip reason when the repository's config, includes followed, names a program.

    ``git config --show-scope --get-regexp`` reads every scope and follows
    ``include.path`` / ``includeIf`` (an included file reports the scope of the
    file that included it). Any match from a repository scope refuses; so does a
    read that fails for any reason other than "no match", since the step then
    cannot tell what ``git add`` would run.
    """
    rc, out, _err = await _git(
        memory_dir, "config", "--null", "--show-scope", "--get-regexp", pattern
    )
    if rc == 1 and not out:
        return None
    if rc != 0:
        return "skipped: config-unreadable"
    # ``--null`` output alternates ``scope NUL key LF value NUL``; a config
    # value never holds a NUL, so the even fields are the scopes.
    scopes = out.split("\0")[0::2]
    if any(scope in _REPOSITORY_SCOPES for scope in scopes):
        return "skipped: command-driver-configured"
    return None


async def commit_projection(
    memory_dir: Path,
    *,
    store: str,
    outcome: str,
    body: str,
    snapshot: str | None,
    run_id: str,
) -> str:
    """Commit a memory-directory render to git — best-effort, never raises.

    Called as the terminal step of :func:`run_projection_cycle` after the
    projection has already written its files. Returns a short telemetry string
    (``committed <sha>`` / ``skipped: <reason>``); every git failure degrades
    silently, so the projection outcome is never altered by this step.

    Stages only the ``memory_dir`` pathspec — never ``git add -A`` — so a
    Particles-authored commit can never sweep in an operator's unrelated
    changes. The state-dir backup/snapshot/archive live outside ``memory_dir``
    and are never committed.
    """
    try:
        git_cfg = get_config().agent_memory.projection.git

        # Not a git repo ⇒ the feature is simply inert (headline case).
        rc, _out, _err = await _git(memory_dir, "rev-parse", "--show-toplevel")
        if rc != 0:
            return "skipped: not-a-git-repo"

        # A filter / textconv / merge driver the repository configures is a
        # program `add` or `diff` would run; skip rather than run it (security
        # review D27). Read the effective config, which follows include.path
        # and includeIf exactly as `git add` will, not one file of it.
        patterns = [_COMMAND_CONFIG_RE]
        if git_cfg.sign:
            patterns.append(_SIGNING_CONFIG_RE)
        for pattern in patterns:
            refusal = await _repository_program_refusal(memory_dir, pattern)
            if refusal is not None:
                return refusal

        # Stage only what is under memory_dir (`.` is relative to `-C`). Never -A.
        rc, _out, err = await _git(memory_dir, "add", "--", ".")
        if rc != 0:
            log.debug("projection git add failed: %s", err)
            return "skipped: add-failed"

        # Nothing staged ⇒ no empty commit.
        rc, _out, _err = await _git(
            memory_dir, "diff", "--cached", "--quiet", "--no-ext-diff", "--no-textconv"
        )
        if rc == 0:
            return "skipped: nothing-to-commit"

        message = build_commit_message(
            store=store,
            outcome=outcome,
            run_id=run_id,
            body=body,
            snapshot=snapshot,
            max_excerpts=git_cfg.max_delta_excerpts,
        )
        commit_cfg: list[tuple[str, str]] = []
        if git_cfg.author_name:
            commit_cfg.append(("user.name", git_cfg.author_name))
        if git_cfg.author_email:
            commit_cfg.append(("user.email", git_cfg.author_email))
        commit_args = ["commit", "--no-verify", "-m", message]
        if not git_cfg.sign:
            # Never impose this SDK's signing requirement; keep an unattended
            # hook commit from blocking on a signing agent.
            commit_args.append("--no-gpg-sign")

        rc, _out, err = await _git(memory_dir, *commit_args, config=commit_cfg or None)
        if rc != 0:
            log.debug("projection git commit failed: %s", err)
            return "skipped: commit-failed"

        rc, sha, _err = await _git(memory_dir, "rev-parse", "--short", "HEAD")
        return f"committed {sha}" if rc == 0 and sha else "committed"
    except Exception as exc:  # noqa: BLE001 — best-effort bonus, never a failure surface
        log.debug("projection git step failed: %s", exc, exc_info=True)
        return f"skipped: {type(exc).__name__}"
