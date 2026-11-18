# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only schedule discovery by physical store; explicit launchd adapter.

The runner and agents directory are injectable. Tests never bootstrap the
operator's jobs. Unknown wrappers/pins fail closed instead of duplicating work.
"""

from __future__ import annotations

import os
import plistlib
import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from particles.api.cli._codex_runtime import physical_store
from particles.api.cli._harness_pins import config_mapping
from particles.url_safety import pinned_subprocess_env

Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def run_command(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Bounded, non-shell scheduler inspection/registration.

    The program, and any it starts, is looked up on the trusted ``PATH`` only
    (security review 2026-11-17-1, RA2).
    """
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env=pinned_subprocess_env(os.environ),
    )


@dataclass(frozen=True)
class Schedule:
    """An inventoried job, whose ownership is never inferred from its name."""

    identity: str
    store_id: str
    enabled: bool
    loaded: bool
    path: Path | None
    env: dict[str, str]


def destination(
    argv: list[str], env: dict[str, str], configs: dict[str, str] | None = None
) -> str | None:
    """Resolve a direct particles command or explicit env wrapper without running it."""
    if not argv:
        return None
    words = list(argv)
    pins = dict(env)
    if Path(words[0]).name == "env":
        words.pop(0)
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
            key, value = words.pop(0).split("=", 1)
            pins[key] = value
    if not words:
        return None
    if Path(words[0]).name != "particles":
        if any("particles" in w or "consolidat" in w for w in words):
            raise ValueError("unresolved consolidation wrapper; inspect existing schedule")
        return None
    if words[1:3] != ["memory", "consolidate"]:
        return None
    config = pins.get("PARTICLES_CONFIG")
    tree: dict[str, Any] = {}
    if config:
        path = Path(config)
        if configs is not None and config in configs:
            tree = config_mapping(configs[config]) or {}
        elif not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("schedule has unresolved config pin")
        else:
            if path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("schedule config exceeds inspection limit")
            tree = config_mapping(path.read_text()) or {}
        if not isinstance(tree, dict):
            raise ValueError("schedule config is not a mapping")
    else:
        # No inheritance from the inspecting process: a launchd job doesn't
        # inherit this shell's config. An unpinned default could be any store.
        raise ValueError("consolidation schedule lacks an absolute config pin")
    store = "default"
    if "--store" in words:
        index = words.index("--store") + 1
        if index >= len(words):
            raise ValueError("schedule missing store argument")
        store = words[index]
    else:
        for word in words:
            if word.startswith("--store="):
                store = word.split("=", 1)[1]
    storage = tree.get("storage", {})
    if not isinstance(storage, dict):
        raise ValueError("schedule storage cannot be resolved")
    dsn = (
        (pins.get("DATABASE_URL") or storage.get("database_url"))
        if store == "default"
        else (storage.get("stores", {}).get(store))
    )
    if not isinstance(dsn, str):
        raise ValueError("schedule destination cannot be resolved")
    if tree.get("engine", {}).get("base_url"):
        raise ValueError("remote consolidation schedule cannot be resolved locally")
    return physical_store(dsn)


def _plist_schedule(
    path: Path, loaded: set[str], configs: dict[str, str] | None = None
) -> Schedule | None:
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("schedule definition is linked or oversized")
    try:
        value = plistlib.loads(path.read_bytes())
    except (ValueError, plistlib.InvalidFileException):
        raise ValueError("unreadable launch agent definition") from None
    if not isinstance(value, dict):
        raise ValueError("invalid launch agent definition")
    argv = value.get("ProgramArguments", [])
    env = value.get("EnvironmentVariables", {})
    if not isinstance(argv, list) or not all(isinstance(w, str) for w in argv):
        raise ValueError("unrecognized launch agent arguments")
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise ValueError("unrecognized launch agent environment")
    store_id = destination(argv, env, configs)
    if store_id is None:
        return None
    label = value.get("Label")
    if not isinstance(label, str):
        raise ValueError("consolidation agent has no label")
    return Schedule(label, store_id, not value.get("Disabled", False), label in loaded, path, env)


def loaded_arguments(text: str) -> tuple[list[str], dict[str, str]]:
    """Decode the measured launchctl print argument/environment blocks.

    Values occupy whole lines, including embedded spaces. No shell executes
    these strings. Unknown/missing fields cannot establish owned-job identity.
    """
    block = re.search(r"^\s*arguments = \{\n(.*?)^\s*\}", text, re.MULTILINE | re.DOTALL)
    if block:
        argv = [line.strip() for line in block.group(1).splitlines()]
    else:
        program = re.search(r"^\s*program = (.+)$", text, re.MULTILINE)
        argv = [program.group(1)] if program else []
    block = re.search(r"^\s*environment = \{\n(.*?)^\s*\}", text, re.MULTILINE | re.DOTALL)
    env: dict[str, str] = {}
    if block:
        for line in block.group(1).splitlines():
            key, separator, value = line.partition("=>")
            if not separator:
                raise ValueError("unrecognized loaded-job environment")
            env[key.strip()] = value.strip()
    return argv, env


def discover(
    agents: Path,
    store_id: str,
    runner: Runner = run_command,
    *,
    configs: dict[str, str] | None = None,
) -> list[Schedule]:
    """Inventory definitions, loaded jobs AND cron; never infer absence on error."""
    listing = runner(["/bin/launchctl", "list"])
    if listing.returncode:
        raise ValueError("cannot inventory loaded user jobs")
    loaded = {line.split()[-1] for line in listing.stdout.splitlines()[1:] if line.split()}
    disabled = runner(["/bin/launchctl", "print-disabled", f"gui/{os.getuid()}"])
    if disabled.returncode:
        raise ValueError("cannot inspect disabled user jobs")
    disabled_labels = set(re.findall(r'"([^"]+)"\s*=>\s*true', disabled.stdout))
    results: dict[str, Schedule] = {}
    definitions: dict[str, Schedule] = {}
    if agents.exists():
        if not agents.is_dir() or agents.is_symlink():
            raise ValueError("cannot inventory user LaunchAgents")
        for path in sorted(agents.glob("*.plist")):
            job = _plist_schedule(path, loaded, configs)
            if job:
                if job.identity in definitions and definitions[job.identity].path != job.path:
                    raise ValueError("duplicate consolidation definitions share one label")
                definitions[job.identity] = job
                if job.store_id == store_id:
                    results[job.identity] = Schedule(
                        job.identity,
                        job.store_id,
                        job.enabled and job.identity not in disabled_labels,
                        job.loaded,
                        job.path,
                        job.env,
                    )
    # Inspect loaded jobs even if their definition was removed or lives elsewhere.
    # Each record includes its source path or direct program/argument vector.
    for label in sorted(loaded):
        info = runner(["/bin/launchctl", "print", f"gui/{os.getuid()}/{label}"])
        if info.returncode:
            raise ValueError("loaded user job disappeared during discovery; retry")
        argv, env = loaded_arguments(info.stdout)
        actual = destination(argv, env, configs)
        path_match = re.search(r"^\s*path = (.+)$", info.stdout, re.MULTILINE)
        if path_match and Path(path_match.group(1)).exists():
            path = Path(path_match.group(1))
            job = _plist_schedule(path, loaded, configs)
            if job is not None or actual is not None:
                if job is None or job.identity != label or actual != job.store_id:
                    raise ValueError("loaded consolidation destination differs from its definition")
                value = plistlib.loads(path.read_bytes())
                if argv != value.get("ProgramArguments") or any(
                    env.get(key) != val
                    for key, val in value.get("EnvironmentVariables", {}).items()
                ):
                    raise ValueError(
                        "loaded consolidation arguments/pins differ from its definition"
                    )
                known = definitions.get(label)
                if known is not None and known.path != path:
                    raise ValueError("loaded consolidation label points to another definition")
                if job.store_id == store_id:
                    results[job.identity] = Schedule(
                        job.identity,
                        job.store_id,
                        job.enabled and label not in disabled_labels,
                        True,
                        path,
                        env,
                    )
            continue
        if actual == store_id:
            results[label] = Schedule(label, actual, label not in disabled_labels, True, None, env)
        elif label in definitions or label.startswith("dev.particles.consolidate"):
            raise ValueError("loaded Particles job lacks a resolvable definition")
    cron = runner(["/usr/bin/crontab", "-l"])
    if cron.returncode and not (cron.returncode == 1 and "no crontab" in cron.stderr.lower()):
        raise ValueError("cannot inventory user crontab")
    cron_env: dict[str, str] = {}
    for index, line in enumerate(cron.stdout.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", line):
            key, value = line.split("=", 1)
            cron_env[key] = value.strip("\"'")
            continue
        words = shlex.split(line)
        argv = words[1:] if line.startswith("@") else words[5:]
        # Shell syntax/unknown wrappers are never executed to find their target.
        if not argv or any(w in {"|", ";", "&&", ">", "2>"} for w in argv):
            raise ValueError("unresolved crontab wrapper")
        target = destination(argv, cron_env, configs)
        if target is None:
            raise ValueError("crontab contains an unresolved job; inspect before installing")
        if target == store_id:
            identity = f"crontab:{index + 1}"
            results[identity] = Schedule(identity, target, True, True, None, dict(cron_env))
    if len(results) > 1:
        raise ValueError(
            "multiple consolidation schedules target this store; operator cleanup required"
        )
    return list(results.values())


def job_definition(
    store_id: str, command: str, store: str, env: dict[str, str], logs: Path
) -> bytes:
    """Daily plus login catch-up, no shell, secret, cwd or version gate."""
    value = {
        "Label": f"dev.particles.consolidate.{store_id}",
        "ParticlesOwner": "codex/1",
        "ParticlesStore": store_id,
        "ProgramArguments": [command, "memory", "consolidate", "--if-due", "--store", store],
        "EnvironmentVariables": env,
        "RunAtLoad": True,
        "StartCalendarInterval": {"Hour": 3, "Minute": 30},
        "StandardOutPath": str(logs / f"{store_id}.out"),
        "StandardErrorPath": str(logs / f"{store_id}.err"),
    }
    return plistlib.dumps(value, sort_keys=True)


def validate_plist(content: bytes) -> bool:
    """Validate generated launchd syntax without creating a file or job."""
    return (
        subprocess.run(
            ["/usr/bin/plutil", "-lint", "-s", "-"],
            input=content,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        ).returncode
        == 0
    )
