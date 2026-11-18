# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Transactional Codex installation plans; injected scheduler, exact ownership.

Planning is read-only. Apply rechecks discovery and input bytes under a shared
installer lock; a journal records before/after hashes before configuration
writes. It never rolls back a database or adopts a foreign schedule.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
import tomllib
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from particles.api.cli._codex_runtime import physical_store
from particles.api.cli._codex_schedule import (
    Runner,
    Schedule,
    discover,
    job_definition,
    run_command,
    validate_plist,
)
from particles.api.cli._harness_pins import config_mapping
from particles.api.cli._hook_state import advisory_lock, private_dir, read_json, write_json
from particles.config import ParticlesConfig

MCP_BEGIN = "# particles codex MCP begin (owned)"
MCP_END = "# particles codex MCP end (owned)"


def digest(content: bytes | None) -> str | None:
    """Hash a planned before-image; missing and empty differ."""
    return hashlib.sha256(content).hexdigest() if content is not None else None


def read_file(path: Path) -> bytes | None:
    """Bounded no-follow config inspection; directories/links are conflicts."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    import stat

    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4 * 1024 * 1024:
            raise ValueError("installer input is not a bounded regular file")
        return stream.read()


def write_file(path: Path, content: bytes | None) -> None:
    """Atomic owner-only config replacement; preserve unrelated bytes in the plan."""
    if content is None:
        path.unlink(missing_ok=True)
        return
    # User Codex/LaunchAgents directories may already have normal 0755 mode.
    # Refuse linked parents; create only missing ones privately.
    missing = []
    for parent in (path.parent, *path.parent.parents):
        if parent.is_symlink():
            raise ValueError("linked installer destination")
        if not parent.exists():
            missing.append(parent)
    for parent in reversed(missing):
        parent.mkdir(mode=0o700, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def set_profile(text: str, profile: str) -> str:
    """Preserve YAML comments; refuse flow/duplicate-key shapes rather than rewrite."""
    tree = config_mapping(text) or {}
    section = tree.get("consolidation", {})
    if not isinstance(section, dict):
        raise ValueError("consolidation config must be a block mapping")
    if section.get("credential_profile"):
        return text
    lines = text.splitlines(keepends=True)
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(r"^consolidation:\s*(#.*)?$", line.rstrip())
        ),
        None,
    )
    if "consolidation" in tree and start is None:
        raise ValueError("use a block-style consolidation config before installing")
    value = json.dumps(profile)
    if start is None:
        result = text.rstrip() + f"\n\nconsolidation:\n  credential_profile: {value}\n"
    else:
        end = next(
            (i for i in range(start + 1, len(lines)) if re.match(r"^[A-Za-z_]", lines[i])),
            len(lines),
        )
        found = [i for i in range(start + 1, end) if re.match(r"^  credential_profile:", lines[i])]
        if len(found) > 1:
            raise ValueError("duplicate credential_profile")
        if found:
            lines[found[0]] = f"  credential_profile: {value}\n"
        else:
            lines.insert(start + 1, f"  credential_profile: {value}\n")
        result = "".join(lines)
    expected = dict(tree)
    expected["consolidation"] = {**section, "credential_profile": profile}
    if config_mapping(result) != expected:
        raise ValueError(
            "cannot preserve consolidation YAML safely; edit credential_profile explicitly"
        )
    return result


@dataclass
class FileChange:
    """A compared-and-swapped file, with separate confidential rollback image."""

    path: Path
    before: bytes | None = field(repr=False)
    after: bytes | None = field(repr=False)
    sensitive: bool = False


@dataclass
class InstallPlan:
    """Concrete read-only plan plus scheduler inventory identity."""

    home: Path
    state: Path
    agents: Path
    store_id: str
    store: str
    config: Path
    files: list[FileChange]
    schedule: Schedule | None
    owned_job: Path | None
    manifest: dict[str, Any]
    remove: bool = False
    notes: list[str] = field(default_factory=list)
    engine: str | None = None
    schedule_store_id: str | None = None
    schedule_configs: dict[str, str] | None = field(default=None, repr=False)
    previous_scope: str | None = None
    config_source: bytes | None = field(default=None, repr=False)

    def report(self) -> dict[str, Any]:
        """No file contents, foreign env values or credentials in machine output."""
        return {
            "adapter": "codex",
            "action": "remove" if self.remove else "install",
            "store": self.store,
            "store_id": self.store_id,
            "files": [
                {"path": str(f.path), "before": digest(f.before), "after": digest(f.after)}
                for f in self.files
                if not f.sensitive and f.before != f.after
            ],
            "schedule": self.schedule.identity if self.schedule else str(self.owned_job),
            "schedule_reused": bool(self.schedule and not self.owned_job),
            "mode": "thin-client" if self.engine else "local",
            "engine": self.engine,
            "schedule_action": (
                "retire owned local job"
                if self.engine and self.owned_job
                else "preserve existing local job"
                if self.engine and self.schedule
                else "engine-managed"
                if self.engine
                else None
            ),
            "hooks": self.manifest.get("hooks", {}),
            "pins": self.manifest.get("env", {}),
            "mcp": self.manifest.get("mcp"),
            "status": "removal planned" if self.remove else "configured, learning blocked",
            "next_steps": self.notes,
        }


def _hook_commands(
    command: str,
    home: Path,
    state: Path,
    store_id: str,
    store: str,
    env: dict[str, str],
    timeout: float,
    tokens: dict[str, str] | None = None,
) -> dict[str, str]:
    from particles.api.cli import _codex_launcher

    launcher = str(Path(_codex_launcher.__file__).resolve())
    base = ["/usr/bin/env", *[f"{k}={v}" for k, v in sorted(env.items())], sys.executable, launcher]
    tail = [
        "--state",
        str(state),
        "--home",
        str(home),
        "--store-id",
        store_id,
        "--store",
        store,
        "--command",
        command,
        "--deadline",
        str(timeout),
    ]
    if tokens is not None:
        tail.extend(["--thin", "--recall-token-file", tokens["recall"]])
        if "capture" in tokens:
            tail.extend(["--capture-token-file", tokens["capture"]])
    return {
        event: shlex.join([*base, "--event", event, *tail])
        for event in ("SessionStart", "SessionEnd")
    }


def _merge_hooks(raw: bytes | None, old: dict[str, str], new: dict[str, str]) -> bytes | None:
    value = json.loads(raw or b"{}")
    if not isinstance(value, dict) or not isinstance(value.get("hooks", {}), dict):
        raise ValueError("hooks.json must contain a hooks mapping")
    hooks = value.setdefault("hooks", {})
    for event in ("SessionStart", "SessionEnd"):
        groups = hooks.get(event, [])
        if not isinstance(groups, list):
            raise ValueError("invalid hook groups")
        kept = []
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                raise ValueError("invalid hook group")
            entries = []
            for entry in group["hooks"]:
                if not isinstance(entry, dict):
                    raise ValueError("invalid hook entry")
                cmd = entry.get("command", "")
                if cmd == old.get(event):
                    continue
                if isinstance(cmd, str) and (
                    "_codex_launcher.py" in cmd or ("particles" in cmd and "codex-session" in cmd)
                ):
                    raise ValueError("unowned/changed Particles hook; resolve duplicate explicitly")
                entries.append(entry)
            if entries:
                kept.append({**group, "hooks": entries})
        if event in new:
            kept.append(
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": new[event],
                            "timeout": 15 if event == "SessionStart" else 3,
                            **({"additionalContextLimit": 0} if event == "SessionStart" else {}),
                        }
                    ]
                }
            )
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event, None)
    if not old and not new:
        return raw
    return (json.dumps(value, indent=2) + "\n").encode()


def _merge_mcp(raw: bytes | None, old: str | None, new: str | None) -> bytes | None:
    text = (raw or b"").decode()
    parsed = tomllib.loads(text)
    # Inline hooks compose with hooks.json. Do not modify that surface.
    if (
        "particles"
        in json.dumps({k: v for k, v in parsed.get("hooks", {}).items() if k != "state"}).lower()
    ):
        raise ValueError("inline TOML Particles hooks conflict with user hooks.json installation")
    if MCP_BEGIN in text or MCP_END in text:
        if old is None or text.count(old) != 1:
            raise ValueError("changed/unowned Particles MCP block")
        text = text.replace(old, "", 1)
        parsed = tomllib.loads(text)
    if old is None and new is None:
        return raw
    if "particles" in parsed.get("mcp_servers", {}):
        raise ValueError("foreign mcp_servers.particles conflicts with installation")
    if new:
        text = text.rstrip() + "\n\n" + new
    tomllib.loads(text)
    return text.encode()


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2) + "\n").encode()


def _fingerprint_copies(state: Path, previous: str, destination: str) -> list[FileChange]:
    from particles.api.cli._codex_runtime import fingerprints
    from particles.extraction.codex_transcript import session_uuid

    old = state / "stores" / previous
    new = state / "stores" / destination
    result = []
    for source in sorted((old / "emitted").glob("*.json")):
        identity = session_uuid(source.stem)
        path = new / "emitted" / source.name
        copied = sorted(fingerprints(old, identity) | fingerprints(new, identity))
        result.append(FileChange(path, read_file(path), (json.dumps(copied) + "\n").encode()))
    return result


def plan_install(
    *,
    home: Path,
    state: Path,
    config: Path,
    config_text: str,
    store: str,
    dsn: str,
    command: str,
    env: dict[str, str],
    mcp_env: dict[str, str],
    agents: Path,
    runner: Runner = run_command,
    remove: bool = False,
    platform: str | None = None,
    validator: Callable[[bytes], bool] = validate_plist,
    engine: str | None = None,
    tokens: dict[str, str] | None = None,
) -> InstallPlan:
    """Parse and validate the whole installation without writing anything."""
    if (platform or sys.platform) != "darwin":
        raise ValueError(
            "Codex integration currently supports local stores on macOS; "
            "Linux lifecycle and scheduling have not been measured"
        )
    if not Path(command).is_absolute() or not Path(command).is_file():
        raise ValueError("--command must name an existing absolute particles executable")
    if not config.is_absolute() or not home.is_absolute() or not state.is_absolute():
        raise ValueError("installation paths must be absolute")
    allowed_pins = {
        "PARTICLES_CONFIG",
        "DATABASE_URL",
        "PARTICLES_ENGINE_BASE_URL",
        "PARTICLES_BLOB_DIR",
        "CODEX_HOME",
        "PARTICLES_CODEX_STATE_DIR",
        "PARTICLES_PROJECT_IDENTITY_LOCK",
        "ANTHROPIC_BASE_URL",
    }
    if engine and not remove and (tokens is None or "recall" not in tokens or "mcp" not in tokens):
        raise ValueError("thin Codex plan requires recall and MCP credential paths")
    if engine:
        allowed_pins.update({"PARTICLES_ENGINE_TOKEN", "PARTICLES_ENGINE_TOKEN_FILE"})
        if env.get("PARTICLES_ENGINE_TOKEN") or mcp_env.get("PARTICLES_ENGINE_TOKEN"):
            raise ValueError("plaintext token values are forbidden in Codex settings")
    if (set(env) | set(mcp_env)) - allowed_pins:
        raise ValueError("only non-secret installer pins may enter job/hook/MCP definitions")
    cfg = ParticlesConfig.model_validate(config_mapping(config_text) or {})
    if bool(cfg.engine.base_url) != bool(engine):
        raise ValueError("Codex plan mode disagrees with the pinned config.")
    if engine and cfg.engine.base_url:
        from particles.api.cli._codex_remote import checked_endpoint

        if checked_endpoint(cfg.engine.base_url) != engine:
            raise ValueError("Codex engine config pin changed during planning.")
    if "codex" not in cfg.observer_scope.harness_tags:
        raise ValueError("add codex to observer_scope.harness_tags before installing")
    if engine:
        from particles.api.cli._codex_remote import destination_id, token_path

        store_id = destination_id(engine, store)
    else:
        from sqlalchemy.engine import make_url

        if make_url(dsn).password:
            raise ValueError(
                "Codex cannot embed database credentials in job definitions; "
                "use a credential-free local DSN"
            )
        store_id = physical_store(dsn)
    manifest_path = state / "installations" / f"{store_id}.json"
    old_path = manifest_path
    previous = read_json(manifest_path, {})
    if engine:
        for other in manifest_path.parent.glob("*.json"):
            if other == manifest_path:
                continue
            owned = read_json(other, {})
            if owned.get("home") == str(home) and owned.get("store") == store:
                if previous or owned.get("config") != str(config):
                    raise ValueError("ambiguous Codex installation ownership; reconcile its pins")
                old_path, previous = other, owned
    if previous and (previous.get("home") != str(home) or previous.get("config") != str(config)):
        raise ValueError("different Codex home/config pin conflicts with existing ownership")
    binding = read_json(state / "home.json", {})
    if binding and binding.get("path") != str(home):
        raise ValueError("codex_home_moved: moved/restored Codex home is a known limit")
    if previous and (not binding or binding.get("id") != previous.get("home_id")):
        raise ValueError(
            "Codex home binding is missing or changed; restore original state; rebind is deferred"
        )
    home_id = binding.get("id") or str(uuid.uuid4())
    previous_scope = previous.get("store_id") if old_path != manifest_path else None
    schedule_store_id = (
        store_id
        if not engine
        else (previous_scope if previous and not previous.get("engine") else None)
    )
    schedule_configs = None
    if engine and schedule_store_id is not None:
        # The operator has already added engine.base_url to the config. Judge
        # the old schedule by its local pins, then confirm the exact definition.
        import yaml

        tree = config_mapping(config_text) or {}
        tree.pop("engine", None)
        schedule_configs = {str(config): yaml.safe_dump(tree)}
    matches = (
        discover(
            agents,
            schedule_store_id,
            runner,
            **({"configs": schedule_configs} if schedule_configs else {}),
        )
        if schedule_store_id is not None
        else []
    )
    schedule = matches[0] if matches else None
    old_commands = previous.get("hooks", {})
    commands = (
        {}
        if remove
        else _hook_commands(
            command,
            home,
            state,
            store_id,
            store,
            env,
            cfg.codex.hook_deadline_seconds,
            tokens if engine else None,
        )
    )
    # Plugin definitions are read-only. Any apparent Particles lifecycle install
    # there requires explicit cleanup, never a second registration.
    for plugin in home.glob("plugins/**/hooks.json"):
        content = read_file(plugin)
        if content and b"particles" in content.lower():
            raise ValueError("plugin Particles hooks conflict; remove duplicate through Codex")
    mcp = None
    if not remove:
        mcp_args = ["mcp", "serve", "--project-observer", "cwd"]
        if engine:
            mcp_args.extend(["--codex-store", store, "--codex-destination", store_id])
        mcp = (
            f"{MCP_BEGIN}\n[mcp_servers.particles]\ncommand = {json.dumps(command)}\n"
            f"args = {json.dumps(mcp_args)}\n"
            "[mcp_servers.particles.env]\n"
            + "".join(f"{json.dumps(k)} = {json.dumps(v)}\n" for k, v in sorted(mcp_env.items()))
            + f"{MCP_END}\n"
        )
    files = []
    for path, content in (
        (home / "hooks.json", _merge_hooks(read_file(home / "hooks.json"), old_commands, commands)),
        (
            home / "config.toml",
            _merge_mcp(read_file(home / "config.toml"), previous.get("mcp"), mcp),
        ),
    ):
        files.append(FileChange(path, read_file(path), content))
    owned_job: Path | None = None
    if previous.get("job"):
        owned_job = Path(previous["job"])
        raw = read_file(owned_job)
        if digest(raw) != previous.get("job_hash"):
            raise ValueError("owned consolidation job changed; refuse removal/replacement")
        import plistlib

        parsed_job = plistlib.loads(raw or b"")
        if (
            parsed_job.get("ParticlesOwner") != "codex/1"
            or parsed_job.get("ParticlesStore") != schedule_store_id
        ):
            raise ValueError("job ownership marker/pins changed")
        if remove or engine:
            shared = any(
                other != old_path and read_json(other, {}).get("schedule") == str(owned_job)
                for other in manifest_path.parent.glob("*.json")
            )
            if shared and remove and not engine:
                raise ValueError("another installation depends on this owned schedule")
            if shared:
                owned_job = None
            else:
                if engine and (schedule is None or schedule.path != owned_job):
                    raise ValueError("owned consolidation schedule could not be verified")
                files.append(FileChange(owned_job, raw, None))
    elif not engine and not schedule and not remove:
        owned_job = agents / f"dev.particles.consolidate.{store_id}.plist"
        if read_file(owned_job) is not None:
            raise ValueError("unowned job already occupies destination")
    if not engine and not remove and owned_job:
        job = job_definition(store_id, command, store, env, state / "logs")
        import plistlib

        from particles.api.cli._codex_schedule import destination

        if (
            destination(plistlib.loads(job)["ProgramArguments"], env, {str(config): config_text})
            != store_id
        ):
            raise ValueError("schedule does not match selected store")
        if not validator(job):
            raise ValueError("generated launch agent failed plist validation")
        files.append(FileChange(owned_job, read_file(owned_job), job))
    from particles.skills import skill_files

    skills: dict[str, str] = {}
    for source in skill_files():
        path = home / "skills" / f"particles-{source.stem}" / "SKILL.md"
        raw = read_file(path)
        old_hash = previous.get("skills", {}).get(str(path))
        if raw is not None and digest(raw) != old_hash:
            raise ValueError("unowned or edited Codex skill; refusing overwrite/removal")
        content = (
            f"---\nname: particles-{source.stem}\ndescription: Particles {source.stem}\n---\n\n"
        ).encode() + source.read_bytes()
        files.append(FileChange(path, raw, None if remove else content))
        skills[str(path)] = digest(content) or ""
    if not remove:
        if not engine:
            files.append(FileChange(config, read_file(config), config_text.encode()))
        home_path = state / "home.json"
        files.append(
            FileChange(
                home_path,
                read_file(home_path),
                (json.dumps({"path": str(home), "id": home_id}, sort_keys=True) + "\n").encode(),
            )
        )
    manifest = {
        "home": str(home),
        "home_id": home_id,
        "config": str(config),
        "store_id": store_id,
        "store": store,
        "hooks": commands,
        "env": env,
        "command": command,
        "mcp": mcp,
        "skills": skills,
        "job": str(owned_job) if owned_job else None,
        "job_hash": next(
            (digest(f.after) for f in files if f.path == owned_job), previous.get("job_hash")
        ),
        "schedule": str(owned_job) if owned_job else schedule.identity if schedule else None,
    }
    if engine:
        manifest.update(engine=engine, tokens=tokens or {})
        # The new launchers can start as soon as settings become visible.
        # Commit their checked pin and fingerprint copies first.
        if not remove:
            files.insert(
                0, FileChange(manifest_path, read_file(manifest_path), _manifest_bytes(manifest))
            )
            if previous_scope is not None:
                files[0:0] = _fingerprint_copies(state, previous_scope, store_id)
        if remove or old_path != manifest_path:
            files.append(FileChange(old_path, read_file(old_path), None))
    else:
        files.append(
            FileChange(
                manifest_path,
                read_file(manifest_path),
                None if remove else (json.dumps(manifest, indent=2) + "\n").encode(),
            )
        )
    notes = [
        "Best-effort capture: verify capture before deleting Codex sessions; "
        "uncaptured turns can be lost without trace."
    ]
    if engine:
        notes.append(
            "Engine daemon owns consolidation. Local jobs and Keychain are not provisioned."
        )
        notes.append(
            "Supported capture destination: same-host loopback. "
            "A forward can send text off-machine."
        )
        if previous_scope:
            notes.append(f"Old pending hints remain under destination {previous_scope}.")
        if not remove and (not tokens or "capture" not in tokens):
            notes.append(
                "Remote capture disabled. Enable codex.harvest.allow_remote and re-run init."
            )
            if os.path.lexists(token_path(state, "capture")):
                notes.append("Capture credential file retained. Revoke it on the engine.")
        if remove:
            notes.append("Credential files retained for explicit revocation on the engine.")
    elif remove:
        notes.append(
            "Enrolled credentials were retained. Delete an unused dev.particles.extract item "
            "in Keychain Access only after checking whether another scheduled job uses it."
        )
    if not remove and not engine:
        if not cfg.codex.harvest.enabled:
            notes.append("learning blocked: codex.harvest.enabled is false")
        notes += [
            "Awaiting hook trust: review SessionStart and SessionEnd in Codex's hook UI.",
            "Scheduled consolidation uses the configured bounded model budget; "
            "it can incur provider costs.",
        ]
        if schedule and (not schedule.enabled or not schedule.loaded):
            notes.append(
                "Existing schedule is disabled/not loaded; repair it before learning is ready."
            )
    return InstallPlan(
        home,
        state,
        agents,
        store_id,
        store,
        config,
        files,
        schedule,
        owned_job,
        manifest,
        remove,
        notes,
        engine,
        schedule_store_id,
        schedule_configs,
        previous_scope,
        config_text.encode() if engine else None,
    )


def plan_needs_lock(plan: InstallPlan) -> bool:
    """Whether applying ``plan`` takes the installer lock: a no-op removal does not."""
    return not (plan.remove and all(item.before == item.after for item in plan.files))


def apply_plan(
    plan: InstallPlan, runner: Runner = run_command, initialize: Callable[[], None] | None = None
) -> None:
    """Apply only an unchanged plan; rollback ours on failure, never corpus data."""
    if not plan_needs_lock(plan):
        return
    journal_path = plan.state / "transaction.json"
    from particles.api.cli._hook_state import installer_lock_path

    lock_path = installer_lock_path()
    with advisory_lock(lock_path, blocking=False) as acquired, ExitStack() as locks:
        if not acquired:
            raise ValueError("another harness installer is running; retry")
        recover(plan.state, runner)
        if plan.engine and read_file(plan.config) != plan.config_source:
            raise ValueError("engine config pin changed since planning; retry")
        if plan.previous_scope is not None and not plan.remove:
            for scope in sorted({plan.previous_scope, plan.store_id}):
                path = plan.state / "stores" / scope / "emitted.lock"
                if not locks.enter_context(advisory_lock(path, blocking=False)):
                    raise ValueError("Codex recall is writing fingerprints; retry")
            expected = [item for item in plan.files if item.path.parent.name == "emitted"]
            if _fingerprint_copies(plan.state, plan.previous_scope, plan.store_id) != expected:
                raise ValueError("Codex fingerprints changed since planning; retry")
        for item in plan.files:
            if read_file(item.path) != item.before:
                raise ValueError("installer input changed since planning; retry")
        current = (
            discover(
                plan.agents,
                plan.schedule_store_id,
                runner,
                **({"configs": plan.schedule_configs} if plan.schedule_configs else {}),
            )
            if plan.schedule_store_id is not None
            else []
        )
        if plan.engine and current != ([plan.schedule] if plan.schedule else []):
            raise ValueError("local schedule changed since planning; retry")
        if [j.identity for j in current] != ([plan.schedule.identity] if plan.schedule else []):
            raise ValueError("consolidation schedules changed since planning; retry")
        private_dir(plan.state / "rollback")
        records: list[dict[str, Any]] = []
        for index, item in enumerate(plan.files):
            backup = plan.state / "rollback" / str(index)
            if item.before is not None:
                write_file(backup, item.before)
            records.append(
                {
                    "path": str(item.path),
                    "before": digest(item.before),
                    "after": digest(item.after),
                    "backup": str(backup),
                }
            )
        label = f"gui/{os.getuid()}/dev.particles.consolidate.{plan.schedule_store_id}"
        write_json(
            journal_path,
            {
                "files": records,
                "job": str(plan.owned_job) if plan.owned_job else None,
                "label": label,
                "remove": plan.remove or bool(plan.engine),
                "was_loaded": bool(plan.schedule and plan.schedule.loaded),
            },
        )
        try:
            private_dir(plan.state / "logs")
            if plan.owned_job and plan.schedule and plan.schedule.loaded:
                result = runner(["/bin/launchctl", "bootout", label])
                if result.returncode:
                    raise ValueError("owned job bootout failed")
            for item in plan.files:
                if item.before != item.after:
                    write_file(item.path, item.after)
            if not plan.remove and initialize is not None:
                initialize()
            if plan.owned_job and not plan.remove and not plan.engine:
                valid = runner(["/usr/bin/plutil", "-lint", str(plan.owned_job)])
                if valid.returncode:
                    raise ValueError("owned job plist validation failed")
                result = runner(
                    ["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(plan.owned_job)]
                )
                if result.returncode:
                    raise ValueError("owned job bootstrap failed")
        except Exception:
            recover(plan.state, runner)
            raise
        journal_path.unlink()
        for record in records:
            Path(record["backup"]).unlink(missing_ok=True)


def recover(state: Path, runner: Runner = run_command) -> None:
    """Recover interrupted owned writes only when every hash still agrees."""
    path = state / "transaction.json"
    journal = read_json(path, {})
    if not journal:
        return
    for item in journal["files"]:
        if digest(read_file(Path(item["path"]))) not in {item["before"], item["after"]}:
            raise ValueError(
                "incomplete install changed by another writer; manual recovery required"
            )
        if item["before"] is not None and digest(read_file(Path(item["backup"]))) != item["before"]:
            raise ValueError("install rollback image missing or changed")
    if journal.get("job") and not journal.get("remove"):
        # A failed bootstrap may have registered before failing. Only the exact
        # owned job validated above is eligible, never a reused job.
        runner(["/bin/launchctl", "bootout", journal["label"]])
    for item in reversed(journal["files"]):
        write_file(Path(item["path"]), read_file(Path(item["backup"])) if item["before"] else None)
    if journal.get("job") and journal.get("was_loaded"):
        result = runner(["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", journal["job"]])
        if result.returncode:
            raise ValueError("rollback restored owned job but could not reload it")
    path.unlink()
    for item in journal["files"]:
        Path(item["backup"]).unlink(missing_ok=True)
