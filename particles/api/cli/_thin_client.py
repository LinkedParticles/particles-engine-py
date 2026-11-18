# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The thin-client half of ``particles init claude-code``.

A thin client pins an engine URL and a token file per client (the hooks and
the MCP server), provisions no local store, and reports what it did not do.
This module holds the pieces that need no settings or store access: the URL
rule, the token-file plan and enrolment, the key-role check, the engine
probe, the wrapper detection and the install state. ``init.py`` composes
them.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from particles.api.cli._claude_code import (
    HOOK_SENTINEL,
    make_private_dir,
    particles_executable,
    state_dir,
    write_private_text,
)
from particles.secrets import EngineTokenFileError, check_engine_token_file
from particles.url_safety import pinned_subprocess_env

#: The two clients a thin install gives a key each.
HOOKS = "hooks"
MCP = "mcp"
TOKEN_CLIENTS = (HOOKS, MCP)

#: The principal the hooks' key must carry (rule 5).
HOOK_PRINCIPAL = "claude-code-hook"

#: The install's own record of what it did, beside ``mcp-registration.json``.
STATE_FILE = "thin-client.json"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ThinClientError(ValueError):
    """A thin-client install cannot proceed; the message names the fix."""


def token_dir() -> Path:
    """Where the install keeps the engine keys it enrols."""
    return state_dir().resolve() / "engine-tokens"


def default_token_path(client: str) -> Path:
    """The token file the install pins for ``client`` when no flag names one."""
    return token_dir() / client


def validate_engine_url(url: str, *, allow_plaintext: bool) -> str:
    """The engine URL a thin install writes, or :class:`ThinClientError`.

    No userinfo (a credential in the config file), no query or fragment, an
    ``http`` or ``https`` scheme, and plain ``http`` to a host that is not
    loopback only when ``engine.allow_plaintext_token`` lets the bearer ride
    it, since otherwise every hook would refuse to send it.
    """
    value = url.strip()
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ThinClientError(f"--engine {value!r} is not an http:// or https:// URL with a host.")
    if parts.username is not None or parts.password is not None:
        raise ThinClientError(
            "--engine must not carry a user or password: it is written to the config "
            "file, and the key belongs in a token file."
        )
    if parts.query or parts.fragment:
        raise ThinClientError("--engine must not carry a query or a fragment.")
    host = parts.hostname.lower()
    if parts.scheme == "http" and host not in _LOOPBACK_HOSTS and not allow_plaintext:
        raise ThinClientError(
            f"--engine {value} sends the engine key over plain http to a host that is not "
            "loopback, which every hook would refuse. Use an https:// URL, reach the engine "
            "through an SSH forward to a loopback port, or set "
            "engine.allow_plaintext_token: true on a network you trust end to end."
        )
    return value.rstrip("/")


@dataclass
class TokenPlan:
    """One client's token file, as the install found it (§6)."""

    client: str
    path: Path
    given: bool
    state: str
    role: str | None = None
    check: str | None = None
    refusal: str | None = None

    def shown(self) -> dict[str, Any]:
        """The plan as the result and the dry run show it: never the value."""
        shown: dict[str, Any] = {"path": str(self.path), "state": self.state}
        if self.role is not None:
            shown["role"] = self.role
        if self.check is not None:
            shown["check"] = self.check
        if self.refusal is not None:
            shown["refused"] = self.refusal
        return shown


def key_refusal(client: str, token: str) -> str | None:
    """Why ``token`` may not be pinned for ``client``, from its prefix.

    An operator key, or the legacy shared key (no ``ptk_`` prefix), carries
    every operator action into an agent's environment and is refused for
    either file. A reader key cannot write the harvest, so it is refused for
    the hooks.
    """
    from particles.api._keys import Role, role_from_prefix

    role = role_from_prefix(token)
    if role is None:
        return (
            "the key is not a per-client key, so it is the legacy shared key, which carries "
            "the operator's authority; mint an agent key (particles engine keys create)"
        )
    if role is Role.OPERATOR:
        return (
            "the key is an operator key, which carries every operator action into an "
            "agent's environment; mint an agent key (particles engine keys create)"
        )
    if role is Role.READER and client == HOOKS:
        return (
            "the key is a reader key, and the SessionEnd harvest and the session exposure "
            "write; mint the hooks an agent key"
        )
    return None


def plan_token(client: str, flag_path: Path | None) -> TokenPlan:
    """Judge ``client``'s token file: missing, refused (and why), or usable."""
    from particles.api._keys import role_from_prefix

    given = flag_path is not None
    path = flag_path if flag_path is not None else default_token_path(client)
    if not path.is_absolute():
        return TokenPlan(
            client,
            path,
            given,
            "refused",
            check="relative-path",
            refusal="the path must be absolute",
        )
    try:
        token = check_engine_token_file(str(path))
    except EngineTokenFileError as exc:
        if exc.check == "missing":
            return TokenPlan(client, path, given, "missing")
        return TokenPlan(client, path, given, "refused", check=exc.check, refusal=str(exc))
    role = role_from_prefix(token)
    refusal = key_refusal(client, token)
    if refusal is not None:
        return TokenPlan(
            client,
            path,
            given,
            "refused",
            role=role.value if role else "legacy",
            check="role",
            refusal=refusal,
        )
    return TokenPlan(client, path, given, "present", role=role.value if role else None)


def read_token(plan: TokenPlan) -> str:
    """The key a passing plan names (the installer's probe and ``--remove`` only)."""
    return check_engine_token_file(str(plan.path))


def enrol_token(path: Path, value: str) -> None:
    """Write ``value`` owner-only to ``path``, through a temporary name renamed into place.

    The directory is created ``0700``; the file is ``0600`` from its creation
    and is never written through a symbolic link, so no reader sees a partial
    key.
    """
    make_private_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)  # a crashed run's leftover; O_EXCL then guards a race
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(tmp, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (value + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def exclude_from_backups(directory: Path) -> str | None:
    """Keep ``directory`` out of Time Machine backups on macOS (owner ruling, 2026-11-07).

    ``tmutil addexclusion`` (without ``-p``) sets a sticky exclusion attribute
    on the directory itself, which needs no administrator rights and moves
    with it. Returns a note when the exclusion could not be set; ``None`` when
    it was set or the platform has no Time Machine.
    """
    if sys.platform != "darwin":
        return None
    tmutil = shutil.which("tmutil", path="/usr/bin")
    if tmutil is None:
        return f"Could not exclude {directory} from Time Machine: tmutil was not found."
    try:
        done = subprocess.run(
            [tmutil, "addexclusion", str(directory)],
            capture_output=True,
            text=True,
            timeout=20,
            env=pinned_subprocess_env(os.environ),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Could not exclude {directory} from Time Machine ({type(exc).__name__})."
    if done.returncode != 0:
        return (
            f"Could not exclude {directory} from Time Machine (tmutil exit {done.returncode}); "
            f"run `tmutil addexclusion {shlex.quote(str(directory))}` by hand."
        )
    return None


def wrapper_hook_entries(settings: dict[str, Any]) -> list[str]:
    """Hook commands that run the lifecycle verbs through something other than ``particles``.

    The interim wrapper (``particles-client hook session-start``)
    is not an entry the installer owns (its command lacks ``particles hook``),
    so a thin install refuses to add its own beside one.
    """
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
                command = str(hook.get("command", "")) if isinstance(hook, dict) else ""
                if HOOK_SENTINEL in command:
                    continue
                try:
                    words = shlex.split(command)
                except ValueError:
                    continue
                for i, word in enumerate(words[:-1]):
                    if word == "hook" and words[i + 1] in ("session-start", "session-end"):
                        # A Particles hook names its store; an unrelated command
                        # that only mentions the words does not.
                        if "--store" in words[i + 2 :] and particles_executable(words[:i]) is None:
                            found.append(command)
                        break
    return found


def client_rule_roots(cwd: Path | None = None) -> tuple[list[str], list[str]]:
    """``(rule roots, global rule roots)`` this client's rule push uses.

    The project root is the working directory's repository, a worktree resolved
    to its main checkout; the global root is ``~/.claude``. A directory with no
    repository contributes no rule root: a common parent is never suggested.
    """
    from particles.api.cli._claude_code import find_repository_root

    project = find_repository_root(cwd or Path.cwd())
    claude = Path.home() / ".claude"
    return (
        [str(project.resolve())] if project is not None else [],
        [str(claude.resolve())] if claude.is_dir() else [],
    )


def keys_create_commands(
    asserter_identity: str, memory: list[str], archive: list[str]
) -> dict[str, str]:
    """The engine-host commands that mint each client's key.

    The hook key also carries this client's host id, the origin the engine
    stamps on every file the hooks deposit.
    """
    from particles.host_identity import local_host_id

    host = local_host_id()
    rule_roots, global_rule_roots = client_rule_roots()
    roots = " ".join(
        ([f"--host-id {host}"] if host is not None else [])
        + [f"--memory-root {shlex.quote(root)}" for root in memory]
        + [f"--archive-root {shlex.quote(root)}" for root in archive]
        + [f"--rule-root {shlex.quote(root)}" for root in rule_roots]
        + [f"--global-rule-root {shlex.quote(root)}" for root in global_rule_roots]
    )
    return {
        HOOKS: (
            "particles engine keys create --role agent --principal claude-code-hook "
            f"--reserved-principal {roots} --label 'claude-code hooks' "
            "--output <file on the engine host>"
        ),
        MCP: (
            "particles engine keys create --role agent --principal "
            f"{shlex.quote(asserter_identity)} "
            "--label 'claude-code MCP server' --output <file on the engine host>"
        ),
    }


def revoke_command(token: str) -> str | None:
    """``keys revoke <id>`` for a per-client key, read from its prefix; ``None`` otherwise."""
    from particles.api._keys import parse_token

    parsed = parse_token(token)
    return None if parsed is None else f"particles engine keys revoke {parsed[1]}"


@dataclass
class ProbeResult:
    """What the engine said about each pinned key (§5)."""

    verified: bool
    note: str | None = None
    refusals: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def probe_engine(
    tokens: dict[str, str],
    *,
    base_url: str,
    store: str,
    asserter_identity: str,
    harvest_roots: tuple[list[str], list[str]],
) -> ProbeResult:
    """Confirm each key's principal and roots, and the store, with the engine.

    ``GET /auth/whoami`` with each key, then ``GET /digest/{store}`` with the
    hooks' key, all to ``base_url``, the engine being installed, never one the
    environment names. An unreachable engine never fails the install: the
    result says it was not verified.
    """
    from particles.api.client.http import EngineHttpError, EngineUnreachableError, HttpBackend
    from particles.egress import run_armed

    result = ProbeResult(verified=True)
    expected = {HOOKS: HOOK_PRINCIPAL, MCP: asserter_identity}
    for client, token in tokens.items():
        try:
            who = run_armed(HttpBackend(token=token, base_url=base_url).whoami())
        except EngineUnreachableError as exc:
            return ProbeResult(verified=False, note=f"the engine was not reached ({exc})")
        except EngineHttpError as exc:
            result.refusals[client] = f"the engine refused the key ({exc})"
            continue
        principal = str(who.get("principal") or "")
        if principal != expected[client]:
            result.refusals[client] = (
                f"the key's principal is {principal or 'anonymous'}, not {expected[client]}"
            )
            continue
        if client == HOOKS:
            memory, archive = harvest_roots
            have_memory = set(who.get("memory_roots") or [])
            have_archive = set(who.get("archive_roots") or [])
            missing = [r for r in memory if r not in have_memory] + [
                r for r in archive if r not in have_archive
            ]
            if missing:
                result.warnings.append(
                    "the hooks' key does not record "
                    + ", ".join(missing)
                    + "; the engine refuses memory and archive files there until the key is "
                    "minted again with this client's roots"
                )
    hooks_token = tokens.get(HOOKS)
    if hooks_token is not None and HOOKS not in result.refusals:
        try:
            run_armed(HttpBackend(token=hooks_token, base_url=base_url).digest(store))
        except EngineUnreachableError as exc:
            return ProbeResult(verified=False, note=f"the engine was not reached ({exc})")
        except EngineHttpError as exc:
            result.refusals["store"] = f"the engine has no usable store {store!r} ({exc})"
    return result


def written_files(state: dict[str, Any]) -> list[dict[str, str]]:
    """The token files the state records, each with the settings file that pinned it."""
    entries = state.get("token_files_written")
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict) and isinstance(e.get("path"), str)]


def read_state() -> dict[str, Any]:
    """The thin install's record (token files it wrote); empty when there is none."""
    path = state_dir() / STATE_FILE
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_state(state: dict[str, Any]) -> None:
    """Record what this install did, owner-only, for ``--remove``."""
    write_private_text(state_dir() / STATE_FILE, json.dumps(state, indent=2, sort_keys=True))


def forget_state() -> None:
    """Drop the thin install's record (after ``--remove``)."""
    (state_dir() / STATE_FILE).unlink(missing_ok=True)


def not_done_entries(*, rule_sources: bool) -> list[dict[str, str]]:
    """The steps a thin install skips, each with its consequence."""
    entries = [
        {
            "step": "project-key rescope",
            "reason": "the rescope maps entries with this machine's ~/.claude/projects and "
            "writes the store, which lives on the engine",
            "consequence": "on an engine store that was never rescoped, beliefs harvested "
            "before this install are in view for no project, so the project observer shows "
            "only what this harness harvests from now on",
            "where_it_runs": "not available from a thin client yet",
        },
        {
            "step": "first-run memory audit",
            "reason": "the audit extracts in-process and needs the operator's key on the engine",
            "consequence": "no audit report; the memory files are harvested at SessionEnd "
            "once claude_code.harvest.allow_remote is on, and extracted by the engine's "
            "consolidation",
            "where_it_runs": "not available from a thin client yet",
        },
    ]
    if rule_sources:
        entries.append(
            {
                "step": "rule-source registration",
                "reason": "a rule source is a file on this machine, which the engine cannot "
                "refresh from its own disk",
                "consequence": "this machine's AGENTS.md and CLAUDE.md files are not tracked "
                "as sources",
                "where_it_runs": "not available from a thin client yet; the client will push them",
            }
        )
    return entries
