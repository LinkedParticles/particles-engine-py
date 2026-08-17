"""init sub-Typer — one-command integration installers.

``particles init claude-code`` installs the SessionStart digest-push +
SessionEnd harvest hooks into Claude Code's settings (marker-owned merge,
idempotent re-run, surgical ``--remove``), provisions the per-machine state
directory, and — on a fresh install with no write-enabled store — creates and
enables a ``memory`` store via a parse-preserve-append ``config.yaml`` edit.

The ``init`` group leaves room for future adapters (Cursor, …;
§ Deferred) — one command per harness.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import typer

from particles.api.cli import app, run
from particles.api.cli._claude_code import (
    MEMORY_REGION,
    ConfigEditError,
    absolutize_sqlite_dsn,
    build_hook_commands,
    default_memory_manifest_text,
    disable_memory_store_text,
    enable_memory_store_text,
    fresh_config_text,
    memory_manifest_path,
    merge_particles_hook_entries,
    particles_hook_commands,
    render_settings_json,
    state_dir,
    strip_particles_hook_entries,
)
from particles.config import get_config, reset_config

log = logging.getLogger(__name__)

init_app = typer.Typer(
    help="Install (or remove) an agent-harness memory integration.",
    no_args_is_help=True,
)
app.add_typer(init_app, name="init")

#: The store handle auto-created on a fresh install.
_DEFAULT_MEMORY_STORE = "memory"

#: Suppresses human narration so ``--json`` leaves stdout carrying only the
#: result object. Diagnostics keep going to stderr either way.
_QUIET = False


def _say(message: str) -> None:
    """Emit operator-facing narration, unless ``--json`` owns stdout."""
    if not _QUIET:
        typer.echo(message)


@init_app.command("claude-code")
def init_claude_code_cmd(
    store: str | None = typer.Option(
        None,
        "--store",
        help=(
            "Memory store handle baked into the hook commands. Default: the single "
            "mcp.write.enabled_stores entry; a fresh install auto-creates 'memory'."
        ),
    ),
    project: bool = typer.Option(
        False,
        "--project",
        help=(
            "Install into the current repo's .claude/settings.local.json (gitignored) "
            "instead of the user-level ~/.claude/settings.json."
        ),
    ),
    remove: bool = typer.Option(
        False,
        "--remove",
        help=(
            "Remove exactly the Particles-owned hook entries (and revert the store "
            "auto-create while the store is still empty)."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the resulting files without writing anything."
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        help=(
            "Override the hook command base (default: the absolute path of the running "
            "`particles` console script)."
        ),
    ),
    no_audit: bool = typer.Option(
        False, "--no-audit", help="Skip the first-run memory-audit hand-off."
    ),
    install_skills_files: bool = typer.Option(
        True,
        "--skills/--no-skills",
        help=(
            "Also install the shipped agent-onboarding skill files "
            "into the harness's skills directory (a Particles-owned subdirectory; "
            "--remove deletes exactly that). Default on — an agent that has the "
            "tools but not the guidance is the gap these close."
        ),
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help=(
            "Emit a machine-readable result on stdout — what was "
            "created, what was merged, and what is left for the human — so an "
            "agent can run the installer and report the outcome instead of "
            "scraping human-formatted output. Implies --no-audit: the audit "
            "hand-off is interactive, and the result names it under next_steps."
        ),
    ),
) -> None:
    """Install the Claude Code hook integration (SessionStart digest push + SessionEnd harvest)."""
    global _QUIET
    settings_path = _settings_path(project)
    _QUIET = json_output
    try:
        if remove:
            result = _remove(settings_path, dry_run=dry_run)
            if install_skills_files:
                result["skills_removed"] = _remove_skills(project=project, dry_run=dry_run)
        else:
            result = _install(
                settings_path,
                store_flag=store,
                dry_run=dry_run,
                command_override=command,
                # --json is non-interactive by construction; the audit is a prompt.
                no_audit=no_audit or json_output,
                install_skills_files=install_skills_files,
            )
            if json_output and not no_audit and not dry_run:
                result["next_steps"].append(
                    "Run `particles audit <memory-dir>` for the first-run memory audit "
                    "(skipped: --json is non-interactive)."
                )
    except ConfigEditError as exc:
        typer.echo(f"Error editing config.yaml: {exc}", err=True)
        typer.echo(
            "No files were rewritten. Add the store to `storage.stores` and "
            "`mcp.write.enabled_stores` by hand, then re-run `particles init claude-code`.",
            err=True,
        )
        raise typer.Exit(1) from exc
    finally:
        _QUIET = False
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Paths + settings I/O
# ---------------------------------------------------------------------------


def _settings_path(project: bool) -> Path:
    """User-level settings by default; the *local* (gitignored) file with --project.

    ``--project`` deliberately targets ``.claude/settings.local.json``, never the
    committed ``.claude/settings.json``: the hook entries embed an operator's
    store choice and executable path, which must not be imposed on other
    contributors.
    """
    if project:
        return Path.cwd() / ".claude" / "settings.local.json"
    return Path.home() / ".claude" / "settings.json"


def _load_settings(path: Path) -> dict[str, Any]:
    """Parse the settings file; unparseable is an error with a message, never a rewrite."""
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        typer.echo(
            f"Error: {path} is not parseable JSON ({exc}); refusing to rewrite it. "
            f"Fix the file and re-run.",
            err=True,
        )
        raise typer.Exit(1) from exc
    if not isinstance(loaded, dict):
        typer.echo(
            f"Error: {path} does not contain a JSON object; refusing to rewrite it.",
            err=True,
        )
        raise typer.Exit(1)
    return loaded


def _config_yaml_path() -> Path:
    """The config.yaml the store auto-create edits (bootstrap override or ./config.yaml)."""
    explicit = os.environ.get("PARTICLES_CONFIG")
    if explicit:
        return Path(explicit)
    return Path.cwd() / "config.yaml"


def _default_command_base() -> str:
    """Absolute path of the running ``particles`` console script."""
    argv0 = Path(sys.argv[0])
    if argv0.name == "particles" and argv0.exists():
        return str(argv0.resolve())
    sibling = Path(sys.executable).with_name("particles")
    if sibling.exists():
        return str(sibling.resolve())
    which = shutil.which("particles")
    if which:
        return str(Path(which).resolve())
    typer.echo(
        "Error: could not resolve the absolute path of the `particles` console script. "
        "Pass it explicitly with --command.",
        err=True,
    )
    raise typer.Exit(1)


def _hook_command_env(handle: str, config_path: Path, *, will_write_config: bool) -> dict[str, str]:
    """Absolute env pins baked into the installed hook commands.

    Claude Code runs the hook from the session's working directory — routinely a
    git worktree — so a bare ``particles hook … --store <handle>`` resolves its
    config, and therefore the store DSN, *CWD-relatively*: an absent
    ``./config.yaml`` degrades to compiled defaults, the ``default`` store's
    ``./particles.db`` lands in the wrong directory, and a named store is not
    even declared. Both hooks then hit ``no such table`` and, under the
    degrade-to-nothing contract, silently drop the harvest.

    Two pins make the resolution CWD-independent:

    * ``PARTICLES_CONFIG`` — the operator config.yaml (named-store registry plus
      all hook tuning: harvest, projection, utility mining). Pinned whenever a
      config file exists, or is about to be written by this install.
    * ``DATABASE_URL`` — the ``default`` store's absolute DSN. That store
      resolves to ``storage.database_url`` (default ``./particles.db``), which
      the env var overrides regardless of config discovery. Only the ``default``
      handle uses it — the override maps there; a named store carries its
      absolute DSN in the pinned config.
    """
    env: dict[str, str] = {}
    if config_path.exists() or will_write_config:
        env["PARTICLES_CONFIG"] = str(config_path.resolve())
    if handle == "default":
        dsn = _store_dsn(handle)
        if dsn:
            env["DATABASE_URL"] = absolutize_sqlite_dsn(dsn)
    return env


def _warn_if_store_dsn_cwd_relative(handle: str, hook_env: dict[str, str]) -> None:
    """Warn when a *named* store's DSN is CWD-relative (harvest would misfire).

    The ``default`` store is covered by the ``DATABASE_URL`` pin; an
    auto-created store is always absolute. This fires only for a pre-existing
    named store the operator declared with a relative SQLite path — the one
    case the pins can't repair, since ``DATABASE_URL`` overrides ``default``
    only. Editing the operator's DSN is out of scope, so we surface it.
    """
    if handle == "default" or "DATABASE_URL" in hook_env:
        return
    dsn = _store_dsn(handle)
    if dsn and absolutize_sqlite_dsn(dsn) != dsn:
        typer.echo(
            f"Warning: store '{handle}' has a working-directory-relative database path "
            f"({dsn}). Claude Code runs the hook from other directories, so harvest may "
            f"land in the wrong file. Set an absolute path in storage.stores.{handle}.",
            err=True,
        )


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def _install(
    settings_path: Path,
    *,
    store_flag: str | None,
    dry_run: bool,
    command_override: str | None,
    no_audit: bool,
    install_skills_files: bool = True,
) -> dict[str, Any]:
    """Install the integration; return the machine-readable result.

    The result is built as the install proceeds and describes what *actually*
    happened, so an agent reading it can report the outcome rather than
    inferring it: which store, which files were written, and — in
    ``next_steps`` — what is left for the human to decide or approve.
    """
    settings = _load_settings(settings_path)
    handle = _resolve_store_handle(store_flag)
    result: dict[str, Any] = {
        "adapter": "claude-code",
        "action": "install",
        "dry_run": dry_run,
        "scope": "project" if _is_project_scope(settings_path) else "user",
        "store": handle,
        "settings_path": str(settings_path),
        "created": {},
        "next_steps": [],
    }

    # Fresh-install store auto-create: plan the config.yaml edit.
    config_path, new_config_text = _plan_store_enable(handle)

    command_base = command_override or _default_command_base()
    hook_env = _hook_command_env(handle, config_path, will_write_config=new_config_text is not None)
    merged = merge_particles_hook_entries(
        settings, build_hook_commands(command_base, handle, hook_env)
    )
    rendered = render_settings_json(merged)
    _warn_if_store_dsn_cwd_relative(handle, hook_env)

    if dry_run:
        _say(f"--dry-run: would write {settings_path}:")
        _say(rendered)
        if new_config_text is not None:
            _say(f"--dry-run: would write {config_path} (store auto-create):")
            _say(new_config_text)
        _say(f"--dry-run: would create state directory {state_dir()}")
        if get_config().agent_memory.projection.enabled:
            if not memory_manifest_path().exists():
                _say(f"--dry-run: would write {memory_manifest_path()}:")
                _say(default_memory_manifest_text())
            for md in _memory_files_without_region():
                _say(f"--dry-run: would insert the projected region into {md}")
        if get_config().rule_sources.enabled:
            for path in _resolved_rule_sources():
                _say(f"--dry-run: would register rule source {path}")
        if install_skills_files:
            planned_skills = _plan_skills(project=_is_project_scope(settings_path))
            for path in planned_skills:
                _say(f"--dry-run: would write skill file {path}")
            result["created"]["skills"] = [str(p) for p in planned_skills]
        result["created"]["config_store_enabled"] = new_config_text is not None
        result["created"]["state_dir"] = str(state_dir())
        result["hooks"] = _hook_command_map(merged)
        return result

    # 1. Store auto-create: config edit first, then initialize the store DB.
    if new_config_text is not None:
        config_path.write_text(new_config_text, encoding="utf-8")
        reset_config()  # the new store must be visible to get_engine()
        _say(f"Enabled store '{handle}' in {config_path}.")
    run(_ensure_store_db(handle))
    result["config_path"] = str(config_path)
    result["created"]["config_store_enabled"] = new_config_text is not None

    # 2. State directory (hook log; the projection's manifest lives here too).
    state = state_dir()
    state.mkdir(parents=True, exist_ok=True)
    result["created"]["state_dir"] = str(state)

    # 2b. MEMORY.md projection provisioning (§3/§7): write the default
    # manifest (never clobbering an operator-edited one) and seed the sentinel
    # region into any existing per-project MEMORY.md that lacks it.
    if get_config().agent_memory.projection.enabled:
        _provision_projection()

    # 2c. Rule-source registration: the operating documents that
    # govern how the agent works here become tracked MUTABLE + LAZY sources, so
    # the rules themselves reach the store — not just conversational reports
    # about them — and the loop keeps them current. Runs before the
    # audit hand-off, which extracts, so the rules land as beliefs in this same
    # install rather than waiting for a night.
    if get_config().rule_sources.enabled:
        _provision_rule_sources(handle)

    # 3. The marker-owned settings merge.
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(rendered, encoding="utf-8")
    _say(f"Installed Claude Code hooks into {settings_path} (store: {handle}).")
    _say("  SessionStart → digest push; SessionEnd → transcript + memory-file harvest.")
    _say("  Re-run to repair/upgrade; `particles init claude-code --remove` uninstalls.")
    result["hooks"] = _hook_command_map(merged)

    # 3b. Agent-onboarding skill files. The tools alone do not tell
    # an agent which write verb to reach for, that nothing is deleted, or that
    # ruling on a contradiction is the operator's call; these do.
    if install_skills_files:
        result["created"]["skills"] = [
            str(p) for p in _install_skills(project=_is_project_scope(settings_path))
        ]

    # 4. First-run audit hand-off (skippable).
    if not no_audit:
        _offer_first_run_audit(handle)

    return result


def _hook_command_map(settings: dict[str, Any]) -> dict[str, list[str]]:
    """The Particles-owned hook commands per lifecycle event, for the JSON result."""
    out: dict[str, list[str]] = {}
    for event, groups in settings.get("hooks", {}).items():
        commands = particles_hook_commands({"hooks": {event: groups}})
        if commands:
            out[event] = commands
    return out


def _is_project_scope(settings_path: Path) -> bool:
    """Whether this install targeted the repo-local settings file (``--project``).

    Derived from the resolved path rather than threaded through, so the skills
    install lands at the same scope as the hooks without a second flag.
    """
    return settings_path.name == "settings.local.json"


def _plan_skills(*, project: bool) -> list[Path]:
    """The skill-file paths an install would write; writes nothing."""
    from particles.api.cli.skills import default_skills_dir, install_skills

    return install_skills(default_skills_dir(project), dry_run=True)


def _install_skills(*, project: bool) -> list[Path]:
    """Copy the shipped skill files in; return what was written (empty on failure)."""
    from particles.api.cli.skills import default_skills_dir, install_skills

    target = default_skills_dir(project)
    try:
        written = install_skills(target)
    except OSError as exc:
        # Never fail an otherwise-good install over the skills copy: the hooks
        # are the integration, the skills are guidance. Say what did not happen.
        typer.echo(f"Warning: could not install skill files into {target}: {exc}", err=True)
        typer.echo("  Retry with `particles skills install`; the hooks are installed.", err=True)
        return []
    _say(f"Installed {len(written)} agent-onboarding skill file(s) into {target}/particles.")
    return written


def _remove_skills(*, project: bool, dry_run: bool) -> str | None:
    """Delete exactly the Particles-owned skills subdirectory."""
    from particles.api.cli.skills import default_skills_dir, remove_skills

    target = default_skills_dir(project)
    try:
        removed = remove_skills(target, dry_run=dry_run)
    except OSError as exc:
        typer.echo(f"Warning: could not remove skill files from {target}: {exc}", err=True)
        return None
    if removed is None:
        return None
    _say(f"{'Would remove' if dry_run else 'Removed'} {removed}")
    return str(removed)


def _resolve_store_handle(store_flag: str | None) -> str:
    """The §1 store-targeting ladder: flag > single enabled store > auto-create."""
    if store_flag:
        return store_flag
    enabled = get_config().mcp.write.enabled_stores
    if len(enabled) == 1:
        return enabled[0]
    if len(enabled) == 0:
        return _DEFAULT_MEMORY_STORE
    typer.echo(
        "Error: several stores are write-enabled "
        f"({', '.join(enabled)}); pass --store <handle> to choose the memory store.",
        err=True,
    )
    raise typer.Exit(1)


def _plan_store_enable(handle: str) -> tuple[Path, str | None]:
    """Compute the config.yaml edit enabling ``handle`` (None = no edit needed).

    Parse-preserve-append via the surgeon in ``_claude_code.py``; an
    unparseable / unusual config raises :class:`ConfigEditError` (caught by
    the command body → error with instructions, never a rewrite).
    """
    config_path = _config_yaml_path()
    cfg = get_config()
    stores = cfg.storage.stores
    enabled = cfg.mcp.write.enabled_stores
    if handle == "default" or (handle in stores and handle in enabled):
        return config_path, None
    dsn = stores.get(handle) or _default_store_dsn(handle)
    if config_path.exists():
        new_text = enable_memory_store_text(config_path.read_text(encoding="utf-8"), handle, dsn)
        if new_text == config_path.read_text(encoding="utf-8"):
            return config_path, None
        return config_path, new_text
    return config_path, fresh_config_text(handle, dsn)


def _default_store_dsn(handle: str) -> str:
    """DSN for an auto-created store: a SQLite file under ``~/.particles/``."""
    db_path = Path.home() / ".particles" / f"{handle}.db"
    return f"sqlite+aiosqlite:///{db_path}"


async def _ensure_store_db(handle: str) -> None:
    """Initialize the store's tables + extractor records (idempotent)."""
    from particles.db import create_tables, session_scope
    from particles.ingest.importers.registry import ensure_extractor_records

    dsn = _store_dsn(handle)
    if dsn and dsn.startswith("sqlite"):
        # SQLite creates the file but not its parent directory.
        _sqlite_file_path(dsn).parent.mkdir(parents=True, exist_ok=True)
    await create_tables(handle)
    async with session_scope(handle) as session:
        await ensure_extractor_records(session)
        await session.commit()


def _store_dsn(handle: str) -> str | None:
    cfg = get_config()
    if handle == "default":
        return cfg.storage.database_url
    return cfg.storage.stores.get(handle)


def _sqlite_file_path(dsn: str) -> Path:
    """The on-disk file of a SQLite DSN (``sqlite+aiosqlite:////abs/path.db``)."""
    return Path(re.sub(r"^sqlite[^:]*:///", "", dsn))


def _claude_memory_files() -> list[Path]:
    """Every existing per-project ``MEMORY.md`` under ``~/.claude/projects/``."""
    root = Path.home() / ".claude" / "projects"
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*/memory/MEMORY.md") if p.is_file())


def _memory_files_without_region() -> list[Path]:
    """The MEMORY.md files init still needs to seed with the sentinel region."""
    from particles.render.markdown import find_projected_regions

    missing: list[Path] = []
    for md in _claude_memory_files():
        text = md.read_text(encoding="utf-8", errors="replace")
        if not any(r.region == MEMORY_REGION for r in find_projected_regions(text)):
            missing.append(md)
    return missing


def _provision_projection() -> None:
    """Write the default ``memory.yaml`` + seed sentinel regions (§3/§7).

    The manifest is written only when absent — an operator-edited manifest is
    never clobbered. Region seeding inserts an *empty* sentinel pair at the
    top of each existing ``MEMORY.md`` (all content preserved below); the
    first harvest cycle fills it. Files created later are handled by the
    cycle itself (it creates a missing ``MEMORY.md`` with the region).
    """
    from particles.render.markdown import atomic_write_text, insert_projected_region_at_top

    manifest = memory_manifest_path()
    if not manifest.exists():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(default_memory_manifest_text(), encoding="utf-8")
        _say(f"Wrote the MEMORY.md projection manifest to {manifest} (yours to edit).")
    for md in _memory_files_without_region():
        text = md.read_text(encoding="utf-8", errors="replace")
        atomic_write_text(md, insert_projected_region_at_top(text, MEMORY_REGION, str(manifest)))
        _say(f"Inserted the projected memory-index region into {md}.")


def _resolved_rule_sources() -> list[Path]:
    """The rule-source set this install would register."""
    from particles.corpus.rule_sources import resolve_rule_sources

    return resolve_rule_sources().files


def _provision_rule_sources(handle: str) -> None:
    """Register the rule-source set against ``handle``.

    Non-fatal by the same reasoning as :func:`_offer_first_run_audit`: the hook
    install is already complete and correct by this point, and a rule file with
    a permissions problem must not undo it.
    """
    from particles.api.cli.rules import projected_region_filter
    from particles.corpus.rule_sources import sync_rule_sources
    from particles.db import session_scope

    async def _go() -> None:
        async with session_scope(handle, write=True) as session:
            report = await sync_rule_sources(
                session, filter_text=projected_region_filter(), deposited_by="init"
            )
            await session.commit()
        if not report.resolution.files:
            return
        _say(
            f"Registered {report.changed} rule source(s) "
            f"({len(report.unchanged)} already current) — `particles rules` lists them."
        )

    try:
        run(_go())
    except Exception as exc:  # noqa: BLE001 — the install itself already succeeded
        log.debug("rule-source registration failed", exc_info=True)
        _say(f"Rule-source registration failed ({exc}) — run `particles rules sync` later.")


def _offer_first_run_audit(handle: str) -> None:
    """First-run memory-audit hand-off (implemented).

    Runs ``particles audit`` over every ``~/.claude/projects/*/memory/``
    directory against the freshly registered store, estimate/confirm gate
    included, and renders the census as init's closing screen. A refusal
    (no API key), a declined confirm, or any audit failure never fails the
    install — init's own work is already done.
    """
    from particles.api.cli.audit import run_first_run_audit

    try:
        run_first_run_audit(handle)
    except typer.Exit:
        # Refusal / abort inside the audit flow already printed its message.
        _say(
            f"First-run audit skipped — run `particles audit ~/.claude/projects/"
            f"<project>/memory --store {handle}` any time."
        )
    except Exception as exc:  # noqa: BLE001 — the install itself already succeeded
        log.debug("first-run audit failed", exc_info=True)
        _say(f"First-run audit failed ({exc}) — the hook install itself succeeded.")


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


def _remove(settings_path: Path, *, dry_run: bool) -> dict[str, Any]:
    """Uninstall the integration; return the machine-readable result."""
    settings = _load_settings(settings_path)
    installed = particles_hook_commands(settings)
    stripped = strip_particles_hook_entries(settings)

    if not installed:
        _say(f"No Particles-owned hook entries found in {settings_path}.")
    handle = _installed_store_handle(installed)
    result: dict[str, Any] = {
        "adapter": "claude-code",
        "action": "remove",
        "dry_run": dry_run,
        "scope": "project" if _is_project_scope(settings_path) else "user",
        "store": handle,
        "settings_path": str(settings_path),
        "hooks_removed": installed,
        "state_dir_kept": str(state_dir()),
        "next_steps": [],
    }

    # Revert the store auto-create only while the store is still empty
    # (a store holding data is never deleted).
    config_path = _config_yaml_path()
    revert_config_text: str | None = None
    store_is_empty = False
    if handle and handle != "default" and handle in get_config().storage.stores:
        store_is_empty = run(_store_is_empty(handle))
        if store_is_empty and config_path.exists():
            new_text = disable_memory_store_text(config_path.read_text(encoding="utf-8"), handle)
            if new_text != config_path.read_text(encoding="utf-8"):
                revert_config_text = new_text

    if dry_run:
        if installed:
            _say(f"--dry-run: would write {settings_path}:")
            _say(render_settings_json(stripped))
        if revert_config_text is not None:
            _say(f"--dry-run: would write {config_path} (store auto-create revert):")
            _say(revert_config_text if revert_config_text else "(empty file)")
        elif handle and not store_is_empty:
            _say(f"--dry-run: store '{handle}' holds data — its configuration would be kept.")
        result["store_config_reverted"] = revert_config_text is not None
        return result

    if installed:
        settings_path.write_text(render_settings_json(stripped), encoding="utf-8")
        noun = "entry" if len(installed) == 1 else "entries"
        _say(f"Removed {len(installed)} Particles-owned hook {noun} from {settings_path}.")

    if revert_config_text is not None and handle:
        config_path.write_text(revert_config_text, encoding="utf-8")
        _delete_empty_sqlite_store(handle)
        reset_config()
        _say(f"Reverted the '{handle}' store auto-create in {config_path} (store was empty).")
    elif handle and handle in get_config().storage.stores and not store_is_empty:
        _say(
            f"Store '{handle}' holds data — kept its configuration and database "
            f"(a store holding data is never deleted)."
        )
    _say(f"State directory {state_dir()} kept (hook log history); delete it manually.")
    result["store_config_reverted"] = revert_config_text is not None and handle is not None
    if handle and handle in get_config().storage.stores and not store_is_empty:
        result["next_steps"].append(
            f"Store '{handle}' holds data and was kept — delete it by hand if you meant to."
        )
    result["next_steps"].append(
        f"State directory {state_dir()} kept (hook log history); delete it manually."
    )
    return result


def _installed_store_handle(commands: list[str]) -> str | None:
    """Parse the ``--store <handle>`` the installed hook commands target."""
    for cmd in commands:
        m = re.search(r"--store\s+(\S+)", cmd)
        if m:
            return m.group(1)
    return None


async def _store_is_empty(handle: str) -> bool:
    """True when the store holds nothing worth preserving (or has no DB at all).

    "Nothing worth preserving" means no particles and no corpus entry that
    ``init`` did not itself create. Rule-source entries are
    excluded from the entry count deliberately: they are *this command's own*
    provisioning artifacts, they duplicate no information (the rule file is
    still on disk, and re-registering is one ``particles rules sync``), and
    counting them would mean a fresh install could never be cleanly undone —
    the contract this check exists to uphold.

    Anything *extracted* from them is a different matter: those are particles,
    so the store is preserved exactly as before.
    """
    from sqlalchemy import func, select
    from sqlalchemy.exc import OperationalError

    from particles.corpus.rule_sources import RULE_SOURCE_TAG
    from particles.corpus.store import CorpusEntryRow
    from particles.db import session_scope
    from particles.store.particle_store import ParticleRow

    dsn = _store_dsn(handle)
    if dsn and dsn.startswith("sqlite") and not _sqlite_file_path(dsn).exists():
        return True
    try:
        async with session_scope(handle) as session:
            entries = await session.scalar(
                select(func.count())
                .select_from(CorpusEntryRow)
                .where(CorpusEntryRow.tags_json.not_like(f'%"{RULE_SOURCE_TAG}"%'))
            )
            particles = await session.scalar(select(func.count()).select_from(ParticleRow))
    except OperationalError:
        return True  # tables never created — nothing to preserve
    return not entries and not particles


def _delete_empty_sqlite_store(handle: str) -> None:
    """Delete the (verified-empty) auto-created SQLite store file, if any."""
    dsn = _store_dsn(handle)
    if not dsn or not dsn.startswith("sqlite"):
        return
    db_path = _sqlite_file_path(dsn)
    for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.debug("could not delete %s", path, exc_info=True)
