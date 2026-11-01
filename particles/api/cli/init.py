# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

from particles.api.cli import app, run
from particles.api.cli._claude_code import (
    MEMORY_REGION,
    ConfigEditError,
    McpRegistration,
    absolutize_sqlite_dsn,
    build_hook_commands,
    claude_project_slug,
    command_env_pins,
    default_memory_manifest_text,
    disable_memory_store_text,
    enable_memory_store_text,
    enclosing_work_tree,
    foreign_config_pin,
    fresh_config_text,
    kept_database_url,
    make_private_dir,
    memory_manifest_path,
    merge_particles_hook_entries,
    minimal_config_text,
    particles_hook_commands,
    particles_mcp_registrations,
    pinned_default_dsn,
    plan_mcp_registration,
    render_settings_json,
    repository_root,
    rule_file_project_key,
    state_dir,
    strip_particles_hook_entries,
    write_private_text,
)
from particles.config import (
    environment_database_url,
    environment_engine_url,
    get_config,
    relative_config_pin,
    reset_config,
    sqlite_file_path,
    user_config_path,
)
from particles.corpus.fetch import store_file_judgement
from particles.url_safety import pinned_subprocess_env, url_without_userinfo

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
    config: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "The config file to pin into the hooks and the MCP registration. Default: "
            "PARTICLES_CONFIG, else ~/.particles/config.yaml (written when missing). "
            "A config.yaml in the working directory is never pinned unless named here. "
            "Given explicitly, it also replaces an MCP registration's pin to another file."
        ),
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
            "Install the shipped agent-onboarding skill files too, "
            "into the harness's skills directory (a Particles-owned subdirectory; "
            "--remove deletes exactly that). Default on: an agent that has the "
            "tools but not the guidance is the gap these close."
        ),
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help=(
            "Emit a machine-readable result on stdout (what was "
            "created, what was merged, and what is left for the human) so an "
            "agent can run the installer and report the outcome instead of "
            "scraping human-formatted output. Implies --no-audit: the audit "
            "hand-off is interactive, and the result names it under next_steps."
        ),
    ),
) -> None:
    """Install the Claude Code hook integration (SessionStart digest push + SessionEnd harvest)."""
    global _QUIET
    settings_path = _settings_path(project)
    _CHECKED_PROJECT_SETTINGS.clear()
    _CHECKED_SETTINGS_DIRECTORIES.clear()
    if project:
        _refuse_symlinked_project_settings(settings_path)
    _QUIET = json_output
    # `--config` and `--remove` point this process at the file the hooks pin,
    # the launcher-configures-itself bootstrap `engine serve` uses; the entry
    # value is restored on the way out.
    entry_config = os.environ.get("PARTICLES_CONFIG")
    if config is not None:
        os.environ["PARTICLES_CONFIG"] = str(config.expanduser().resolve())
        reset_config()
    try:
        # Refuse a relative pin before anything is written (security review R10).
        _config_yaml_path()
        if remove:
            result = _remove(settings_path, dry_run=dry_run, explicit_config=config is not None)
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
                explicit_config=config is not None,
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
        _CHECKED_PROJECT_SETTINGS.clear()
        _CHECKED_SETTINGS_DIRECTORIES.clear()
        if os.environ.get("PARTICLES_CONFIG") != entry_config:
            if entry_config is None:
                os.environ.pop("PARTICLES_CONFIG", None)
            else:
                os.environ["PARTICLES_CONFIG"] = entry_config
            reset_config()
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


#: The project settings file's bytes as ``_refuse_symlinked_project_settings``
#: read them through the descriptor it checked (``None``: no file), keyed by
#: path. ``_load_settings`` parses these rather than reopening the name, so a
#: link swapped in after the check is never read (security review
#: 2026-10-25-2, R18). Cleared when the command ends.
_CHECKED_PROJECT_SETTINGS: dict[Path, bytes | None] = {}


def _load_settings(path: Path) -> dict[str, Any]:
    """Parse the settings file; unparseable is an error with a message, never a rewrite.

    A project-scope file is never read by name: its bytes come from the
    descriptor the start-of-command check opened, or from a fresh checked open
    when no check ran in this process (R18).
    """
    if _is_project_scope(path):
        data = (
            _CHECKED_PROJECT_SETTINGS[path]
            if path in _CHECKED_PROJECT_SETTINGS
            else _read_checked_project_settings(path)
        )
        if data is None:
            return {}
    elif not path.exists():
        return {}
    else:
        data = path.read_bytes()
    try:
        loaded = json.loads(data.decode("utf-8"))
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


def _refuse_symlinked_project_settings(settings_path: Path) -> None:
    """Refuse a project-scope settings file reached through a link, or not the user's.

    ``.claude/settings.local.json`` sits in the working directory, and git
    carries symlinks, so a repository can commit one that points at the user's
    own ``~/.claude/settings.json``; an install or remove would then rewrite
    that file (security review 2026-10-24-2, P24). The settings file and every
    directory between it and the working directory must be real. A co-tenant
    on the same volume can plant a hard link instead, which no symlink check
    sees, so an existing settings file must also have one name and belong to
    this user (security review 2026-10-25-1, Q27). The file is read here,
    through the descriptor the check judged, and the merge parses those bytes,
    so a name swapped between this check and the read is never followed; the
    ``.claude`` directory must be this user's too (security review
    2026-10-25-2, R18).
    """
    from particles.api.cli.skills import SkillsPathError, _refuse_symlinks

    try:
        _refuse_symlinks(settings_path, Path.cwd())
    except SkillsPathError as exc:
        typer.echo(f"Error: {exc}. Nothing was written.", err=True)
        raise typer.Exit(1) from exc
    _refuse_foreign_settings_directory(settings_path.parent)
    _CHECKED_PROJECT_SETTINGS[settings_path] = _read_checked_project_settings(settings_path)


def _owned_by_another_account(st: os.stat_result) -> bool:
    getuid = getattr(os, "getuid", None)
    return getuid is not None and st.st_uid != getuid()


#: The identity (device, inode) of each project ``.claude`` directory the
#: start-of-command check judged, ``None`` when it did not exist then; the
#: settings write is made only into that same directory (W30). Cleared with
#: ``_CHECKED_PROJECT_SETTINGS``.
_CHECKED_SETTINGS_DIRECTORIES: dict[Path, tuple[int, int] | None] = {}


def _open_settings_directory(directory: Path) -> int:
    """A descriptor on a project ``.claude`` directory, refused if it is a link or unsafe.

    Exits 1 with a message when the directory is reached through a link, is
    another account's, or any account may write in it (R18, W30). Raises
    ``FileNotFoundError`` when there is no such directory.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    try:
        fd = os.open(directory, flags | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        raise
    except OSError as exc:
        typer.echo(f"Error: cannot inspect {directory}: {exc}. Nothing was written.", err=True)
        raise typer.Exit(1) from exc
    st = os.fstat(fd)
    if _owned_by_another_account(st) or st.st_mode & 0o002:
        os.close(fd)
        typer.echo(
            f"Error: {directory} belongs to another account or any account may write in it; "
            "refusing to read or write settings in it. Nothing was written.",
            err=True,
        )
        raise typer.Exit(1)
    return fd


def _refuse_foreign_settings_directory(directory: Path) -> None:
    """Refuse a project ``.claude`` directory that is a link or another account's (R18)."""
    try:
        fd = _open_settings_directory(directory)
    except FileNotFoundError:
        _CHECKED_SETTINGS_DIRECTORIES[directory] = None
        return
    try:
        st = os.fstat(fd)
    finally:
        os.close(fd)
    _CHECKED_SETTINGS_DIRECTORIES[directory] = (st.st_dev, st.st_ino)


def _read_checked_project_settings(settings_path: Path) -> bytes | None:
    """Read the project settings file through one checked descriptor (Q27, R18).

    The ``.claude`` directory is opened without following a link and, when
    :func:`_refuse_foreign_settings_directory` judged it, must be that one; the file is
    opened relative to that descriptor, so a directory swapped by name after
    the check never supplies the bytes (security review 2026-10-28-1, O26).
    The file is opened without following a link, then judged on its own
    descriptor: a regular file with one name that this user owns. Returns
    ``None`` when there is no file. Exits 1 with a message on anything else.
    """
    directory = settings_path.parent
    try:
        dir_fd = _open_settings_directory(directory)
    except FileNotFoundError:
        return None
    try:
        st = os.fstat(dir_fd)
        identity = (st.st_dev, st.st_ino)
        # With no start-of-command check in this process, the descriptor just
        # opened is the one judged (``_open_settings_directory``).
        if _CHECKED_SETTINGS_DIRECTORIES.get(directory, identity) != identity:
            typer.echo(
                f"Error: {directory} changed since the install checked it; refusing to "
                "read settings from it. Nothing was written.",
                err=True,
            )
            raise typer.Exit(1)
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(settings_path.name, flags | getattr(os, "O_CLOEXEC", 0), dir_fd=dir_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            typer.echo(f"Error: cannot open {settings_path}: {exc}. Nothing was written.", err=True)
            raise typer.Exit(1) from exc
    finally:
        os.close(dir_fd)
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1 or _owned_by_another_account(st):
            typer.echo(
                f"Error: {settings_path} is not a regular file, has another name elsewhere "
                "or belongs to another account; refusing to read or rewrite it. "
                "Nothing was written.",
                err=True,
            )
            raise typer.Exit(1)
        return handle.read()


def _make_settings_directory(settings_path: Path) -> None:
    """Create the settings directory if needed; record a project one the install made.

    A project ``.claude`` directory the start-of-command check found absent is
    created here and opened without following a link, and its identity is
    recorded, so the settings write lands only in the directory this install
    created. One that appeared after the check, or is not the empty directory
    just made, is refused (security review 2026-10-30-1, AB67).
    """
    directory = settings_path.parent
    if not _is_project_scope(settings_path) or (
        _CHECKED_SETTINGS_DIRECTORIES.get(directory, ()) is not None
    ):
        directory.mkdir(parents=True, exist_ok=True)
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(directory)
    except FileExistsError as exc:
        typer.echo(
            f"Error: {directory} appeared since the install checked it; refusing to write "
            "settings in it. Nothing was written.",
            err=True,
        )
        raise typer.Exit(1) from exc
    fd = _open_settings_directory(directory)
    try:
        st = os.fstat(fd)
        with os.scandir(fd) as entries:
            occupied = next(entries, None) is not None
    finally:
        os.close(fd)
    if occupied:
        typer.echo(
            f"Error: {directory} changed while the install created it; refusing to write "
            "settings in it. Nothing was written.",
            err=True,
        )
        raise typer.Exit(1)
    _CHECKED_SETTINGS_DIRECTORIES[directory] = (st.st_dev, st.st_ino)


def _write_settings(path: Path, text: str) -> None:
    """Write a settings file; a project-scope one never through a link.

    The user-level ``~/.claude/settings.json`` is the operator's own and may be
    a deliberate link (a dotfiles checkout), so only the project-scope file is
    written by atomic replace, which neither a symlink (P24) nor a hard link
    (Q27) planted at its name can redirect.
    """
    from particles.api.cli.skills import write_replacing_at

    if not _is_project_scope(path):
        path.write_text(text, encoding="utf-8")
        return
    # The ``.claude`` directory is opened without following a link and must be
    # the one the start-of-command check judged; the write is made through the
    # descriptor, so the directory is never resolved by name again (W30).
    # A directory the check found absent must be one the install then created
    # and recorded (:func:`_make_settings_directory`), never one that appeared
    # by name (security review 2026-10-30-1, AB67).
    fd = _open_settings_directory(path.parent)
    try:
        st = os.fstat(fd)
        checked = _CHECKED_SETTINGS_DIRECTORIES.get(path.parent, (st.st_dev, st.st_ino))
        if checked != (st.st_dev, st.st_ino):
            typer.echo(
                f"Error: {path.parent} changed since the install checked it; refusing to "
                "write settings in it.",
                err=True,
            )
            raise typer.Exit(1)
        write_replacing_at(fd, path.name, text.encode("utf-8"))
    finally:
        os.close(fd)


#: The project-scope store creations ``init`` recorded, keyed by project
#: directory: the only stores a project-scope ``--remove`` may revert (P24 / N25).
_PROJECT_STORES_STATE = "project-stores.json"


def _project_stores() -> dict[str, str]:
    try:
        loaded = json.loads((state_dir() / _PROJECT_STORES_STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(k): str(v) for k, v in loaded.items() if isinstance(v, str)}


def _project_created_store(project_dir: Path) -> str | None:
    """The store a project-scope install in ``project_dir`` recorded creating, if any."""
    return _project_stores().get(str(project_dir.resolve()))


def _record_project_created_store(project_dir: Path, handle: str) -> None:
    stores = _project_stores()
    stores[str(project_dir.resolve())] = handle
    make_private_dir(state_dir())
    write_private_text(state_dir() / _PROJECT_STORES_STATE, json.dumps(stores, sort_keys=True))


def _forget_project_created_store(project_dir: Path) -> None:
    stores = _project_stores()
    if stores.pop(str(project_dir.resolve()), None) is not None:
        write_private_text(state_dir() / _PROJECT_STORES_STATE, json.dumps(stores, sort_keys=True))


def _config_yaml_path() -> Path:
    """The config file the install pins and edits.

    ``PARTICLES_CONFIG`` (which ``--config`` sets for this process), else the
    user file, the same two places the loader reads. Never a file in the
    working directory, which may be a clone: a relative ``PARTICLES_CONFIG``
    resolves there, and pinning its resolved path would bake that file into
    every hook and the MCP registration for good, past the hooks' own
    relative-pin refusal (security review 2026-10-25-2, R10). The install
    refuses one; ``--config`` names a file explicitly and is resolved first.
    """
    explicit = relative_config_pin()
    if explicit is not None:
        typer.echo(
            f"Error: PARTICLES_CONFIG names a relative path ({explicit}), which resolves "
            "against the working directory; the install would pin that file into every "
            "hook. Name the config file with --config, or set PARTICLES_CONFIG to an "
            "absolute path (write $HOME/... rather than ~/...).",
            err=True,
        )
        raise typer.Exit(1)
    pinned = os.environ.get("PARTICLES_CONFIG")
    if pinned:
        return Path(pinned)
    return user_config_path()


def _default_command_base() -> str:
    """Absolute path of the running ``particles`` console script."""
    argv0 = Path(sys.argv[0])
    if argv0.name == "particles" and argv0.exists():
        return str(argv0.resolve())
    sibling = Path(sys.executable).with_name("particles")
    if sibling.exists():
        return str(sibling.resolve())
    which = _trusted_which("particles")
    if which:
        return str(Path(which).resolve())
    typer.echo(
        "Error: could not resolve the absolute path of the `particles` console script. "
        "Pass it explicitly with --command.",
        err=True,
    )
    raise typer.Exit(1)


def _hook_command_env(handle: str, config_path: Path) -> dict[str, str]:
    """Absolute env pins baked into the installed hook commands.

    Claude Code runs the hook from the session's working directory — routinely a
    git worktree — so a bare ``particles hook … --store <handle>`` resolves its
    config, and therefore the store DSN, *CWD-relatively*: an absent
    ``./config.yaml`` degrades to compiled defaults, the ``default`` store's
    ``./particles.db`` lands in the wrong directory, and a named store is not
    even declared. Both hooks then hit ``no such table`` and, under the
    degrade-to-nothing contract, silently drop the harvest.

    Two pins make the resolution CWD-independent:

    * ``PARTICLES_CONFIG`` — the operator config file (named-store registry plus
      all hook tuning: harvest, projection, utility mining). Pinned always, even
      when it is the user file, so a hook states its config instead of relying
      on resolution; the install writes the file when missing.
    * ``DATABASE_URL`` — the ``default`` store's absolute DSN. That store
      resolves to ``storage.database_url`` (default ``./particles.db``), which
      the env var overrides regardless of config discovery. Only the ``default``
      handle uses it — the override maps there; a named store carries its
      absolute DSN in the pinned config.

    ``PARTICLES_ENGINE_BASE_URL`` is pinned empty, which the loader reads as
    no override: the engine a hook writes to is the pinned config's
    ``engine.base_url``, and never one the harness's environment supplies
    (security review 2026-10-25-2, R41). Nor is it the one the installer's
    own environment names, which an agent running the install inside a
    session may have inherited from a project's settings; the install reports
    such a value instead of baking it in (security review 2026-10-26-1, W27).
    """
    env: dict[str, str] = {
        "PARTICLES_CONFIG": str(config_path.resolve()),
        "PARTICLES_ENGINE_BASE_URL": "",
    }
    if handle == "default":
        dsn = _store_dsn(handle)
        if dsn:
            env["DATABASE_URL"] = absolutize_sqlite_dsn(dsn)
    return env


def _mcp_registration_env(handle: str, hook_env: dict[str, str]) -> dict[str, str]:
    """The hooks' pins, plus ``DATABASE_URL`` when the pinned config leaves it relative.

    The MCP read tools (``query``, ``particles_list``, ``graph_view`` …) read
    the ``default`` store. A hook names its store with ``--store``, so the
    hooks of a named-store install carry no ``DATABASE_URL``; a registration
    copied from them left ``default`` at the compiled ``./particles.db``,
    which resolves against the directory Claude Code starts the server in:
    the session's project, possibly a clone that ships its own
    ``particles.db`` (security review N3). When the pinned config names no
    absolute ``default`` store, the registration pins it to the installed
    store's absolute DSN, so the agent's read tools read the memory its hooks
    harvest. A config that names one keeps it: that store is the operator's
    choice. The pinned file is read as written, never through this process's
    own environment. The hooks' empty engine pin is kept too, so the server
    talks to the pinned config's engine and never to one the environment
    Claude Code starts it with names (security review E2, 2026-10-26-1 W32).
    """
    env = dict(hook_env)
    if "DATABASE_URL" in env:
        return env
    default = pinned_default_dsn({"PARTICLES_CONFIG": env["PARTICLES_CONFIG"]})
    if absolutize_sqlite_dsn(default) != default:
        dsn = _store_dsn(handle) or _default_store_dsn(handle)
        env["DATABASE_URL"] = absolutize_sqlite_dsn(dsn)
    return env


def _note_inherited_engine_url(result: dict[str, Any], config_path: Path) -> None:
    """Say so when this install runs with a ``PARTICLES_ENGINE_BASE_URL`` it does not pin.

    The hooks and the MCP registration pin the variable empty, so the engine
    they use is the pinned config's. An inherited value is reported, never
    baked into every session's hooks (security review 2026-10-26-1, W27). It is
    named without any userinfo, since the note reaches the terminal, the
    ``--json`` result and an agent's transcript (2026-10-27-1, W58).
    """
    inherited = environment_engine_url()
    if not inherited:
        return
    message = (
        f"PARTICLES_ENGINE_BASE_URL={_shown_url(inherited)} is set in this install's "
        "environment, but the hooks and the MCP server do not use it: they use "
        f"engine.base_url from {config_path}. To make the hooks thin clients of that "
        "engine, set engine.base_url there and re-run `particles init claude-code`."
    )
    result["next_steps"].append(message)
    typer.echo(f"Note: {message}", err=True)


def _note_harvest_key_command(result: dict[str, Any]) -> None:
    """Print the ``keys create`` command that mints this client's hook key.

    It goes to stderr and into the result's next steps, so a run with or
    without ``--json`` shows it.

    With ``engine.base_url`` set the hooks harvest over HTTP, and the engine
    accepts a memory or archive file only under the client directories the
    hook key records. ``keys create`` runs on the engine's host, which cannot
    see this client's directories, so the install names them, resolved as the
    harvest resolves the paths it sends.
    """
    if not get_config().engine.base_url:
        return
    from particles.api.cli._claude_code import local_harvest_roots

    memory, archive = local_harvest_roots()
    flags = " ".join(
        [f"--memory-root {shlex.quote(root)}" for root in memory]
        + [f"--archive-root {shlex.quote(root)}" for root in archive]
    )
    result["next_steps"].append(
        "On the engine's host, mint the hooks their key with `particles engine keys create "
        f"--role agent --principal claude-code-hook --reserved-principal {flags}`, then give "
        "it to the hooks as PARTICLES_ENGINE_TOKEN. The engine refuses this client's memory "
        "and archive files under a hook key that records other directories."
    )
    # A run without --json shows it too (security review 2026-10-30-1, AB21).
    typer.echo(f"Note: {result['next_steps'][-1]}", err=True)


def _note_config_origin(
    result: dict[str, Any], config_path: Path, *, explicit_config: bool
) -> None:
    """Name the config file the pins carry when this install's environment chose it.

    ``PARTICLES_CONFIG`` in the install's environment (a project's env block, a
    direnv file, an agent running the install in a session) is pinned into
    every hook and the MCP registration, so every later session reads that
    file's store and engine. The install says so unless ``--config`` named the
    file or it is the user file the install would pin anyway, and warns when
    the local refresh would not trust the file as the operator's, such as one
    inside a git work tree (security review 2026-10-28-1, O29; the
    ``DATABASE_URL`` note below is the precedent).
    """
    if explicit_config or not os.environ.get("PARTICLES_CONFIG"):
        return
    resolved = Path(os.path.realpath(config_path))
    if resolved == Path(os.path.realpath(user_config_path())):
        return
    # Judged as the refresh judges a store file, with no SQLite sidecars, and
    # each cause worded with its own fix (security review 2026-10-28-1, G14).
    # The name is judged as pinned, as the refresh's config gate and the
    # doctor judge it: a realpath would skip the directory a link on that
    # name lives in (2026-10-28-2, AA14).
    judgement = store_file_judgement(Path(os.path.abspath(config_path)), sidecars=())
    if judgement is None:
        message = (
            f"PARTICLES_CONFIG={resolved} is set in this install's environment, so the hooks "
            "and the MCP server are pinned to that file and every session reads its store "
            "and engine. If that variable was not meant for Particles, unset it and re-run "
            "`particles init claude-code`, or name the file with --config."
        )
        result["next_steps"].append(message)
        typer.echo(f"Note: {message}", err=True)
        return
    repository = enclosing_work_tree(resolved)
    where = f"inside the git work tree {repository}" if repository is not None else judgement.cause
    warning = (
        f"PARTICLES_CONFIG={resolved} is set in this install's environment, {where}; the "
        "hooks and the MCP server are pinned to it and every session reads its store and "
        "engine. If a clone supplied it, unset it and re-run `particles init claude-code`, "
        "or name your own file with --config."
    )
    result["next_steps"].append(warning)
    typer.echo(f"Warning: {warning}", err=True)


def _note_store_origin(result: dict[str, Any], hook_env: dict[str, str], config_path: Path) -> None:
    """Name the ``default`` store the pins carry when the config file did not choose it.

    The ``DATABASE_URL`` pin is this install's ``default`` store made absolute.
    When a ``DATABASE_URL`` in the install's environment chose it (a project's
    env block, a direnv file, an agent running the install in a session), or
    the compiled ``./particles.db`` was resolved against the working directory
    (an unpacked repository archive has no ``.git``), every later session reads
    and harvests into that file, so the install says which and where (security
    review N3; 2026-10-27-1, N3, W27's class). A pin the local refresh would
    not trust as the operator's, such as one inside a git work tree, is warned
    about as ``hook doctor`` reports it (W16).
    """
    pinned = hook_env.get("DATABASE_URL")
    if pinned is None:
        return
    db_file = sqlite_file_path(pinned)
    shown = db_file or _shown_url(pinned)
    inherited = environment_database_url()
    configured = pinned_default_dsn({"PARTICLES_CONFIG": hook_env["PARTICLES_CONFIG"]})
    message: str | None = None
    if inherited and absolutize_sqlite_dsn(inherited) != absolutize_sqlite_dsn(configured):
        message = (
            f"DATABASE_URL={_shown_url(inherited)} is set in this install's "
            f"environment, so the hooks and the MCP server are pinned to the default store "
            f"{shown}, not the one {config_path} names. If that variable was not meant for "
            "Particles, unset it and re-run `particles init claude-code`."
        )
    elif not inherited and absolutize_sqlite_dsn(configured) != configured:
        message = (
            f"The default store is pinned to {shown}, resolved against this install's working "
            f"directory {Path.cwd()}; every session will read that file. To choose another, "
            f"set storage.database_url to an absolute path in {config_path} and re-run "
            "`particles init claude-code`."
        )
    if message is not None:
        result["next_steps"].append(message)
        typer.echo(f"Note: {message}", err=True)
    judgement = None if db_file is None else store_file_judgement(Path(db_file))
    if judgement is None:
        return
    # Worded as the refresh judges it, each cause with its own fix (security
    # review 2026-10-28-1, G14): another owner and an unopenable file are no
    # longer both "not this account's own".
    warning = (
        f"The default store is pinned to {db_file}, {judgement.cause}; every session will "
        "read that file. If a clone supplied it, set an absolute DATABASE_URL outside the "
        "repository and re-run `particles init claude-code`."
    )
    result["next_steps"].append(warning)
    typer.echo(f"Warning: {warning}", err=True)


def _warn_if_store_dsn_cwd_relative(handle: str, hook_env: dict[str, str]) -> None:
    """Warn when a *named* store's DSN is CWD-relative (the hooks refuse it; review Q5).

    The ``default`` store is covered by the ``DATABASE_URL`` pin; an
    auto-created store is always absolute. This fires for a pre-existing
    named store the operator declared with a relative SQLite path — the one
    case the pins can't repair, since ``DATABASE_URL`` overrides ``default``
    only — and for an absolute one whose file the local refresh refuses
    (inside a git work tree, writable by others, another account's).
    Editing the operator's DSN is out of scope, so we surface it.
    """
    if handle == "default" or "DATABASE_URL" in hook_env:
        return
    dsn = _store_dsn(handle)
    if not dsn:
        return
    if absolutize_sqlite_dsn(dsn) != dsn:
        typer.echo(
            f"Warning: store '{handle}' has a working-directory-relative database path "
            f"({dsn}). Claude Code runs the hooks from each session's directory, so they "
            f"refuse this store and do nothing. Set an absolute path in storage.stores.{handle}.",
            err=True,
        )
        return
    # The store file is judged as the local refresh judges it, as the default
    # store's pin is (security review 2026-10-28-1, O30, E23).
    # Each cause other than a work tree is worded as the refresh words it, with
    # its fix (security review 2026-10-28-1, G14, O60).
    db_file = sqlite_file_path(dsn) if dsn else None
    judgement = None if db_file is None else store_file_judgement(Path(db_file))
    if db_file is None or judgement is None:
        return
    repository = enclosing_work_tree(Path(db_file))
    where = f"inside the git work tree {repository}" if repository is not None else judgement.cause
    typer.echo(
        f"Warning: store '{handle}' is {db_file}, {where}; every session will read that "
        "file. If a clone supplied it, set an absolute path outside the repository in "
        f"storage.stores.{handle} and re-run `particles init claude-code`.",
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
    explicit_config: bool = False,
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
    hook_env = _hook_command_env(handle, config_path)
    _note_inherited_engine_url(result, config_path)
    _note_harvest_key_command(result)
    _note_config_origin(result, config_path, explicit_config=explicit_config)
    _note_store_origin(result, hook_env, config_path)
    # Re-running strips every Particles-owned entry first, so this also drops
    # the Stop entry an install made while agent extraction (
    # deprecated) was on.
    merged = merge_particles_hook_entries(
        settings, build_hook_commands(command_base, handle, hook_env)
    )
    _note_explicit_store_scope(result, config_path)
    rendered = render_settings_json(merged)
    _warn_if_store_dsn_cwd_relative(handle, hook_env)

    # a `--project` install is about this project, so its side
    # effects are too. (The read lens is the same in both installs — it comes
    # from the session's working directory, not from which file holds the hook.)
    project_root = repository_root(Path.cwd()) if _is_project_scope(settings_path) else None
    project_memory_dir = (
        Path.home() / ".claude" / "projects" / claude_project_slug(project_root) / "memory"
        if project_root is not None
        else None
    )
    if dry_run:
        _say(f"--dry-run: would write {settings_path} (values it does not write hidden):")
        _say(_shown_settings_json(merged))
        if new_config_text is not None:
            purpose = (
                "store auto-create" if handle != "default" else "the config file the hooks pin"
            )
            _say(f"--dry-run: would write {config_path} ({purpose}):")
            _say(new_config_text)
        _say(f"--dry-run: would create state directory {state_dir()}")
        if not _is_project_scope(settings_path):
            _register_mcp_server(
                result,
                executable=command_base,
                env=_mcp_registration_env(handle, hook_env),
                dry_run=True,
                replace_config_pin=explicit_config,
            )
        if get_config().agent_memory.projection.enabled:
            if not memory_manifest_path().exists():
                _say(f"--dry-run: would write {memory_manifest_path()}:")
                _say(default_memory_manifest_text())
            for md in _memory_files_without_region(project_memory_dir):
                _say(f"--dry-run: would insert the projected region into {md}")
        if get_config().rule_sources.enabled:
            for path in _resolved_rule_sources(project_root):
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
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(new_config_text, encoding="utf-8")
        reset_config()  # the new store must be visible to get_engine()
        if handle in get_config().storage.stores:
            _say(f"Enabled store '{handle}' in {config_path}.")
            if _is_project_scope(settings_path):
                # What a project-scope remove may revert (P24 / N25).
                _record_project_created_store(Path.cwd(), handle)
        else:
            _say(f"Wrote {config_path}, the config file the hooks pin.")
    run(_ensure_store_db(handle))
    result["config_path"] = str(config_path)
    result["created"]["config_store_enabled"] = new_config_text is not None

    # 2. State directory (hook log; the projection's manifest lives here too).
    state = state_dir()
    make_private_dir(state)
    # An older install created it with the umask's mode; it is this
    # integration's own directory, so tighten it (security review D46).
    state.chmod(0o700)
    result["created"]["state_dir"] = str(state)

    # 2b. MEMORY.md projection provisioning (§3/§7): write the default
    # manifest (never clobbering an operator-edited one) and seed the sentinel
    # region into any existing per-project MEMORY.md that lacks it.
    if get_config().agent_memory.projection.enabled:
        _provision_projection(project_memory_dir)

    # 2c. Rule-source registration: the operating documents that
    # govern how the agent works here become tracked MUTABLE + LAZY sources, so
    # the rules themselves reach the store — not just conversational reports
    # about them — and the loop keeps them current. Runs before the
    # audit hand-off, which extracts, so the rules land as beliefs in this same
    # install rather than waiting for a night.
    if get_config().rule_sources.enabled:
        _provision_rule_sources(handle, project_root)

    # 3. The marker-owned settings merge.
    _make_settings_directory(settings_path)
    _write_settings(settings_path, rendered)
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
        _offer_first_run_audit(handle, project_memory_dir)

    # 5. Project keys. Last, so whatever the audit just deposited
    # is covered; it is also what lets `claude_code.observer_scope: project`
    # engage on this store at all.
    _rescope_store(handle, result)

    # 6. The MCP server, bound to each session's project.
    if not _is_project_scope(settings_path):
        _register_mcp_server(
            result,
            executable=command_base,
            env=_mcp_registration_env(handle, hook_env),
            dry_run=False,
            replace_config_pin=explicit_config,
        )

    return result


#: Where `init` records the MCP registration change it made, so `--remove` reverts
#: exactly that and nothing it did not do.
_MCP_STATE = "mcp-registration.json"


def _trusted_which(name: str) -> str | None:
    """``shutil.which(name)`` over the ``PATH`` entries no other user may write.

    An empty or relative entry would find a program in the working directory,
    which may be a clone, and the install runs ``claude`` and pins the
    ``particles`` it finds into every hook (security review 2026-10-28-1, O61).
    The entries kept are the ones a pinned ``curl`` or ``git`` fetch keeps.
    """
    return shutil.which(name, path=_trusted_path())


def _trusted_path() -> str:
    """The inherited ``PATH`` cut to the entries no other user may write."""
    return pinned_subprocess_env(os.environ).get("PATH", "")


def _claude_mcp(argv: list[str]) -> bool:
    """Run ``claude <argv>``; ``False`` when the CLI is missing or the call fails.

    The ``claude`` CLI owns ``~/.claude.json``, which a running Claude Code
    rewrites often, so the installer edits it through the CLI rather than in
    place. It is looked up on the trusted ``PATH`` entries only, and runs with
    ``PATH`` cut to those entries too: an npm install's ``claude`` finds its
    ``node`` through ``PATH``, which an untrusted entry would answer from the
    working directory (security review 2026-10-28-2, AA10).
    """
    exe = _trusted_which("claude")
    if exe is None:
        return False
    env = {**os.environ, "PATH": _trusted_path()}
    try:
        done = subprocess.run([exe, *argv], capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _register_mcp_server(
    result: dict[str, Any],
    *,
    executable: str,
    env: dict[str, str],
    dry_run: bool,
    replace_config_pin: bool = False,
) -> None:
    """Register (or bind) the user-scope Particles MCP server: ``mcp serve --project-observer cwd``.

    A session's MCP reads then see the same project view as its digest. An
    existing registration keeps its executable, gains the launch flag, and
    gains the hooks' ``PARTICLES_CONFIG`` pin when it has none.
    A pin to a different file is kept and reported unless ``--config`` asked
    to replace it. The flag goes into the Claude Code registration, whose
    server starts in the session's directory, never into machine-wide config.
    """
    existing = particles_mcp_registrations()
    foreign = None if replace_config_pin else foreign_config_pin(existing, env)
    if foreign is not None:
        wanted_config = env.get("PARTICLES_CONFIG", "")
        result["next_steps"].append(
            f"The Particles MCP server pins PARTICLES_CONFIG={foreign}, not {wanted_config}; "
            "it was left as it is. To pin the hooks' file instead, re-run with "
            f"`particles init claude-code --config {wanted_config}`."
        )
        _say(f"The MCP server keeps its own config pin ({foreign}).")
    plan = plan_mcp_registration(
        existing, executable=executable, env=env, replace_config_pin=replace_config_pin
    )
    kept = kept_database_url(existing, env, None if plan is None else plan[0])
    if kept is not None:
        have, wanted_url = _shown_url(kept), _shown_url(env["DATABASE_URL"])
        result["next_steps"].append(
            f"The Particles MCP server keeps DATABASE_URL={have}, not {wanted_url}, which "
            "the hooks pin; its tools serve that store, not the one the hooks harvest. To "
            f"serve the hooks' store, remove it with `claude mcp remove --scope user "
            f"{existing[0].name}` and re-run `particles init claude-code`."
        )
        _say(f"The MCP server keeps its own DATABASE_URL ({have}, not {wanted_url}).")
    if plan is None:
        result["mcp_server"] = {"action": "unchanged", "bound": True}
        return
    wanted, replaced = plan
    command = _shown_claude_command(wanted.add_argv())
    action = "add" if replaced is None else "bind"
    result["mcp_server"] = {"action": action, "name": wanted.name, "command": command}
    if dry_run:
        _say(f"--dry-run: would register the MCP server: {command}")
        return
    removed = replaced is not None and _claude_mcp(
        ["mcp", "remove", "--scope", "user", wanted.name]
    )
    ok = (replaced is None or removed) and _claude_mcp(wanted.add_argv())
    if not ok:
        if removed and replaced is not None and not _claude_mcp(replaced.add_argv()):
            # The remove landed and the add did not: the registration as it
            # was could not be put back either (security review 2026-10-27-1,
            # E13), so the next step names it.
            result["next_steps"].append(
                "The Particles MCP server registration was removed and could not be restored; "
                f"re-add it with `{_shown_claude_command(replaced.add_argv())}`, setting each "
                f"{_HIDDEN_VALUE} value to the one the old registration carried."
            )
        result["mcp_server"]["action"] = "manual"
        result["next_steps"].append(
            "Register the Particles MCP server bound to each session's project: "
            + (f"`claude mcp remove --scope user {wanted.name}` then " if replaced else "")
            + f"`{command}`"
            + (
                f", setting each {_HIDDEN_VALUE} value to the one the old registration carried."
                if _HIDDEN_VALUE in command
                else "."
            )
        )
        _say("Could not run `claude mcp`; the registration command is in next_steps.")
        return
    previous = None if replaced is None else replaced.add_argv()
    write_private_text(
        state_dir() / _MCP_STATE,
        json.dumps({"name": wanted.name, "created": replaced is None, "previous": previous}),
    )
    _say(f"Registered the MCP server bound to each session's project ({wanted.name}).")
    if replaced is not None and wanted.env != replaced.env:
        pins = ", ".join(
            f"{key}={_shown_env_value(key, value)}"
            for key, value in sorted(wanted.env.items())
            if replaced.env.get(key) != value
        )
        result["mcp_server"]["pinned"] = pins
        _say(f"  Pinned its config: {pins}.")
        dropped = sorted(set(replaced.env) - set(wanted.env))
        if dropped:
            # A replaced config pin takes the other config's env with it (AA21).
            result["mcp_server"]["dropped"] = dropped
            _say(f"  Dropped the pins the install does not write: {', '.join(dropped)}.")
    _say("  Restart running Claude Code sessions to pick it up.")


#: What a printed ``claude mcp`` command shows in place of an env value the
#: install did not choose (an engine token the owner pinned by hand).
_HIDDEN_VALUE = "<hidden>"

#: The env pins the install itself writes, shown as they are: the config path,
#: and the store and engine URLs without their userinfo.
_SHOWN_ENV_KEYS = frozenset({"PARTICLES_CONFIG", "DATABASE_URL", "PARTICLES_ENGINE_BASE_URL"})


def _shown_env_value(key: str, value: str) -> str:
    """``value`` as a printed command or note may show it (security review 2026-10-28-1, O27)."""
    if key == "PARTICLES_CONFIG" or not value:
        return value
    if key in _SHOWN_ENV_KEYS:
        return _shown_url(value)
    return _HIDDEN_VALUE


def _shown_url(value: str) -> str:
    """A URL or DSN with its userinfo removed and any query or fragment hidden.

    A DSN may carry its password as a query parameter (``?password=…``), which
    removing the userinfo leaves in place (security review 2026-10-28-2, AA23).
    """
    shown = url_without_userinfo(value)
    cut = min((i for i in (shown.find("?"), shown.find("#")) if i >= 0), default=-1)
    return shown if cut < 0 else f"{shown[:cut]}?{_HIDDEN_VALUE}"


def _shown_claude_command(argv: list[str]) -> str:
    """``claude <argv>`` for the terminal, ``--json`` and next steps, every secret env value hidden.

    A registration's env can hold a value the owner pinned by hand, an engine
    token among them, and a re-run that rebinds it would print it into the
    result, the dry-run narration and a session transcript (security review
    2026-10-28-1, O27, E19). Only the subprocess gets the real argv.
    """
    shown: list[str] = []
    options = True
    for word in argv:
        if options and shown and shown[-1] == "-e" and "=" in word:
            key, value = word.split("=", 1)
            word = f"{key}={_shown_env_value(key, value)}"
        elif word == "--":
            options = False
        shown.append(word)
    return "claude " + shlex.join(shown)


def _unregister_mcp_server(result: dict[str, Any], *, dry_run: bool) -> None:
    """Revert the registration change `init` recorded, and only that one."""
    path = state_dir() / _MCP_STATE
    try:
        raw = _read_private_state(path)
    except OSError:
        return
    if raw is None:
        # Another account could have written it: nothing in it is acted on
        # (security review 2026-10-26-1, W69).
        result["next_steps"].append(
            f"{path} is not a private file of this account; the MCP registration it records "
            "was neither removed nor restored. Check `claude mcp list` by hand."
        )
        return
    try:
        recorded = json.loads(raw)
    except ValueError:
        return
    name = str(recorded.get("name") or "particles")
    previous = recorded.get("previous")
    steps = [["mcp", "remove", "--scope", "user", name]]
    if isinstance(previous, list):
        recorded_add = _recorded_mcp_add([str(a) for a in previous], name)
        if recorded_add is not None:
            steps.append(recorded_add.add_argv())
        else:
            # The state file is owner-only, but what it replays is checked
            # anyway: only the user-scope `mcp add` of the recorded name
            # (security review 2026-10-24-2, P23).
            previous = None
            result["next_steps"].append(
                f"{path} records a previous registration that is not a direct `claude mcp add "
                f"--scope user {name}` of a particles executable; it was not replayed. Re-add the "
                "server by hand if you had one before the install."
            )
    result["mcp_server"] = {
        "action": "restore" if previous else "remove",
        "commands": [_shown_claude_command(s) for s in steps],
    }
    if dry_run:
        for step in steps:
            _say(f"--dry-run: would run {_shown_claude_command(step)}")
        return
    if all(_claude_mcp(step) for step in steps):
        path.unlink(missing_ok=True)
        _say(
            "Restored the MCP server registration as it was."
            if previous
            else f"Removed the MCP server registration '{name}' that init added."
        )
        return
    result["next_steps"].extend(f"Run `{_shown_claude_command(s)}`." for s in steps)


def _read_private_state(path: Path) -> str | None:
    """``path``'s text when it is a regular file of this account no other can write.

    ``None`` when it is not; ``OSError`` (``FileNotFoundError`` among them)
    when it cannot be opened. Read through one ``O_NOFOLLOW`` descriptor, so
    the judged file is the one read.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode) or _owned_by_another_account(st) or st.st_mode & 0o077:
            return None
        return handle.read().decode("utf-8", errors="replace")


def _is_installed_executable(command: Path) -> bool:
    """Whether ``command`` is an executable file this account or root owns and only it can write.

    A recorded registration names its program by path; replaying one that
    another account could have placed or changed would run that account's code
    in every session (security review 2026-10-26-1, W69). Every directory above
    the named path and above the file it resolves to is held to the same owners
    and may not be writable by another account unless sticky, since whoever can
    write a directory can swap the name inside it after the replay
    (2026-10-27-1, W69). The same holds for each link between them, so the
    chain is walked one link at a time and every link's directories are held
    to it too (security review 2026-10-28-1, O66). A link is judged by its own
    owner as well: one another account owns can be repointed by that account
    even in a sticky directory, so every link on the way, a hop or a directory
    above one, must be root's or this account's (2026-10-28-2, AA58).
    """
    try:
        st = command.stat()
        real = command.resolve(strict=True)
        hops = _link_chain(command)
    except (OSError, RuntimeError):
        return False
    owners = _trusted_owners()
    if not (
        stat.S_ISREG(st.st_mode)
        and st.st_uid in owners
        and not st.st_mode & 0o022
        and bool(st.st_mode & 0o100)
    ):
        return False
    for path in {p for hop in hops for p in (hop, *hop.parents)}:
        try:
            link = os.lstat(path)
        except OSError:
            return False
        if stat.S_ISLNK(link.st_mode) and link.st_uid not in owners:
            return False
    directories = {d for hop in (*hops, real) for d in hop.parents}
    return all(_is_trusted_directory(d, owners) for d in directories)


#: The longest link chain a replayed executable may sit behind.
_MAX_LINK_HOPS = 40


def _link_chain(command: Path) -> list[Path]:
    """``command`` and every link it passes through, one ``readlink`` at a time.

    Raises ``OSError`` when the chain is longer than :data:`_MAX_LINK_HOPS`.
    """
    hops = [Path(os.path.abspath(command))]
    while hops[-1].is_symlink():
        if len(hops) > _MAX_LINK_HOPS:
            raise OSError(f"{command}: too many levels of symbolic links")
        # Joined, not normalised: a ``..`` in a link is resolved by the system
        # against the directory the link is in, wherever that leads.
        hops.append(hops[-1].parent / os.readlink(hops[-1]))
    return hops


def _trusted_owners() -> set[int]:
    """Root and this account: the owners a replayed program and its directories may have."""
    getuid = getattr(os, "getuid", None)
    return {0} if getuid is None else {0, getuid()}


def _is_trusted_directory(directory: Path, owners: set[int]) -> bool:
    """Whether ``directory`` is owned by ``owners`` and no other account can rename inside it."""
    try:
        st = directory.stat()
    except OSError:
        return False
    if not stat.S_ISDIR(st.st_mode) or st.st_uid not in owners:
        return False
    return not st.st_mode & 0o022 or bool(st.st_mode & stat.S_ISVTX)


#: The ``mcp serve`` arguments a replayed registration may carry after ``mcp serve``.
_REPLAYABLE_SERVE_ARGS = ((), ("--project-observer", "cwd"), ("--project-observer=cwd",))


def _is_replayable_env(pair: str) -> bool:
    """Whether a recorded ``-e KEY=VALUE`` is one the installer writes, pinned absolute.

    Only ``PARTICLES_CONFIG`` (an absolute path), ``DATABASE_URL`` (a SQLite
    DSN with an absolute path) and the empty ``PARTICLES_ENGINE_BASE_URL`` pin
    qualify: any other key or value, such as ``PATH`` or an engine URL, could
    change which program, store or engine the restored server uses (Q53, W32).
    """
    key, sep, value = pair.partition("=")
    if not sep:
        return False
    if key == "PARTICLES_CONFIG":
        return Path(value).is_absolute()
    if key == "DATABASE_URL":
        db_file = sqlite_file_path(value)
        return db_file is not None and Path(db_file).is_absolute()
    if key == "PARTICLES_ENGINE_BASE_URL":
        return value == ""
    return False


def _recorded_mcp_add(argv: list[str], name: str) -> McpRegistration | None:
    """The registration a recorded ``claude mcp add`` of a Particles server named ``name`` makes.

    The shape :meth:`McpRegistration.add_argv` writes: ``mcp add --scope
    user``, the recorded name, any ``-e KEY=VALUE`` pairs, ``--``, then a
    command line that runs ``particles mcp serve``. The name may also follow
    the pairs, the order releases before the fix wrote, which the CLI refuses
    (security review 2026-10-27-1, E13); either is read, and the replay
    re-emits the name-first order. The state file is owner-only, and what it
    replays is held to the narrowest shape anyway (security review
    2026-10-25-1, Q53): the command is an absolute path to a ``particles``
    executable run directly (never a shell or a launcher), its arguments are
    ``mcp serve`` and the one flag that verb takes, and every ``-e`` pair
    passes :func:`_is_replayable_env`. The executable must pass
    :func:`_is_installed_executable` (W69). ``None`` for anything else.
    """
    if argv[:4] != ["mcp", "add", "--scope", "user"]:
        return None
    rest = argv[4:]
    named = rest[:1] == [name]
    if named:
        rest = rest[1:]
    env: dict[str, str] = {}
    while rest[:1] == ["-e"]:
        if len(rest) < 2 or not _is_replayable_env(rest[1]):
            return None
        key, _, value = rest[1].partition("=")
        env[key] = value
        rest = rest[2:]
    if not named:
        if rest[:1] != [name]:
            return None
        rest = rest[1:]
    if rest[:1] != ["--"] or len(rest) < 4:
        return None
    command, *args = rest[1:]
    if not Path(command).is_absolute() or Path(command).name != "particles":
        return None
    if args[:2] != ["mcp", "serve"] or tuple(args[2:]) not in _REPLAYABLE_SERVE_ARGS:
        return None
    if not _is_installed_executable(Path(command)):
        return None
    return McpRegistration(name=name, command=command, args=tuple(args), env=env)


def _is_recorded_mcp_add(argv: list[str], name: str) -> bool:
    """Whether ``argv`` is a replayable recorded ``mcp add`` (see :func:`_recorded_mcp_add`)."""
    return _recorded_mcp_add(argv, name) is not None


def _hook_command_map(settings: dict[str, Any]) -> dict[str, list[str]]:
    """The Particles-owned hook commands per lifecycle event, for the JSON result.

    Each command's ``NAME=VALUE`` pins are shown without a URL's userinfo, as
    the narrated registration command is (security review 2026-10-28-1, O15);
    the settings file itself carries them as written.
    """
    out: dict[str, list[str]] = {}
    for event, groups in settings.get("hooks", {}).items():
        commands = particles_hook_commands({"hooks": {event: groups}})
        if commands:
            out[event] = [_shown_command(command) for command in commands]
    return out


def _shown_command(command: str) -> str:
    """A shell command with each ``NAME=VALUE`` pin shown as :func:`_shown_env_value` shows it.

    A URL keeps neither its userinfo nor its query string, and a value the
    install does not show (an engine token) is hidden (security review
    2026-10-28-2, AA23).
    """
    try:
        words = shlex.split(command)
    except ValueError:
        return _HIDDEN_VALUE
    for word in words:
        name, sep, value = word.partition("=")
        if not (sep and value and _ENV_NAME.fullmatch(name)):
            continue
        shown = _shown_env_value(name, value)
        if shown == value:
            continue
        quoted = f"{name}={shlex.quote(value)}"
        if quoted in command:
            command = command.replace(quoted, f"{name}={shlex.quote(shown)}")
        else:
            command = command.replace(value, shown)
    return command


def _shown_settings(value: Any) -> Any:
    """A settings value with every ``env`` value and every ``command`` shown, not as written.

    Each ``env`` block's values pass through :func:`_shown_env_value` and each
    ``command`` string through :func:`_shown_command`, at any depth, so a token
    another tool keeps in the file is not printed (security review
    2026-10-30-1, AB25).
    """
    if isinstance(value, list):
        return [_shown_settings(item) for item in value]
    if not isinstance(value, dict):
        return value
    shown: dict[str, Any] = {}
    for key, item in value.items():
        if key == "env" and isinstance(item, dict):
            shown[key] = {
                name: _shown_env_value(name, v) if isinstance(v, str) else _HIDDEN_VALUE
                for name, v in item.items()
            }
        elif key == "command" and isinstance(item, str):
            shown[key] = _shown_command(item)
        else:
            shown[key] = _shown_settings(item)
    return shown


def _shown_settings_json(settings: dict[str, Any]) -> str:
    """The settings file a dry run narrates, rendered with :func:`_shown_settings`."""
    return render_settings_json(_shown_settings(settings))


#: An environment variable name, as a hook command's ``env`` prefix writes one.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _note_explicit_store_scope(result: dict[str, Any], config_path: Path) -> None:
    """Tell a config that still says ``observer_scope: store`` the default moved.

    The compiled default is ``project``, but a ``config.yaml`` copied from an
    older sample names ``store`` explicitly. A value the operator may have
    chosen is never rewritten; the next step says what changed and how to
    take it.
    """
    import yaml

    try:
        tree = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return
    section = tree.get("claude_code") if isinstance(tree, dict) else None
    if isinstance(section, dict) and section.get("observer_scope") == "store":
        result["next_steps"].append(
            f"{config_path} sets `claude_code.observer_scope: store`, so every session keeps "
            "reading the whole store. The default is now `project` (global beliefs plus the "
            "session's project): delete that line, or set it to `project`, to take it."
        )


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
        written = install_skills(target, root=Path.cwd() if project else None)
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
        removed = remove_skills(target, dry_run=dry_run, root=Path.cwd() if project else None)
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
        # Nothing to enable, but the hooks pin the file, so it must exist:
        # a header-only one, never any setting.
        return config_path, None if config_path.exists() else minimal_config_text()
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

    # create_tables() creates a missing SQLite parent directory itself.
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


def _memory_files_without_region(only_memory_dir: Path | None = None) -> list[Path]:
    """The MEMORY.md files init still needs to seed with the sentinel region."""
    from particles.render.markdown import find_projected_regions

    missing: list[Path] = []
    for md in _claude_memory_files():
        if only_memory_dir is not None and md.parent != only_memory_dir:
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        if not any(r.region == MEMORY_REGION for r in find_projected_regions(text)):
            missing.append(md)
    return missing


def _provision_projection(only_memory_dir: Path | None = None) -> None:
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
        # Owner-only and never through a symlink, like the rest of the state
        # directory (security review D46).
        write_private_text(manifest, default_memory_manifest_text())
        _say(f"Wrote the MEMORY.md projection manifest to {manifest} (yours to edit).")
    for md in _memory_files_without_region(only_memory_dir):
        text = md.read_text(encoding="utf-8", errors="replace")
        atomic_write_text(md, insert_projected_region_at_top(text, MEMORY_REGION, str(manifest)))
        _say(f"Inserted the projected memory-index region into {md}.")


def _resolved_rule_sources(project_root: Path | None = None) -> list[Path]:
    """The rule-source set this install would register."""
    from particles.corpus.rule_sources import resolve_rule_sources

    return resolve_rule_sources([str(project_root)] if project_root is not None else None).files


def _provision_rule_sources(handle: str, project_root: Path | None = None) -> None:
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
                session,
                # A `--project` install registers this project's rule documents
                # only; the user-level ones belong to the user-level install.
                [str(project_root)] if project_root is not None else None,
                filter_text=projected_region_filter(),
                deposited_by="init",
                project_key_for=rule_file_project_key,
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


def _rescope_store(handle: str, result: dict[str, Any] | None = None) -> None:
    """Bring the store's project keys up to date. Additive and idempotent.

    Then reports, in ``next_steps``, what a project observer would not show:
    unattributed harvests, entries keyed only to projects
    that no longer exist, lapsed beliefs, and the agent writes
    `--mark-agent-writes` would mark, each with its command. It never assigns
    a project itself: that is the operator's decision, not an installer's.

    Non-fatal, like the other post-install steps: the hooks are installed and
    correct whether or not this succeeds, and `particles memory rescope` is
    there to run by hand.
    """
    from particles.api.cli._claude_code import entry_project_key, is_live_project_key
    from particles.core.status import Status
    from particles.db import session_scope
    from particles.operations.observer_scope import agent_identities, mark_agent_writes, rescope
    from particles.store.observer_scope_join import load_scopes
    from particles.store.particle_store import get_particles_by_status

    root = Path.home() / ".claude" / "projects"
    steps: list[str] = result["next_steps"] if result is not None else []

    async def _go() -> None:
        async with session_scope(handle, write=True) as session:
            report = await rescope(
                session,
                key_for=lambda _entry_id, uri_r, tags: entry_project_key(uri_r, tags, root),
                actor="init",
            )
            await session.commit()
            marks = await mark_agent_writes(
                session, identities=agent_identities(), dry_run=True, actor="init"
            )
            actives = await get_particles_by_status(session, Status.ACTIVE)
            lapsed = sum(1 for s in (await load_scopes(session, actives)).values() if s.lapsed)
        if report.added:
            _say(f"Added {len(report.added)} project key(s) to already-harvested sources.")
        orphaned = [
            entry_id
            for entry_id, keys in report.keys_by_entry.items()
            if not any(is_live_project_key(key, root) for key in keys)
        ]
        if report.unattributed:
            _say(
                f"{len(report.unattributed)} harvested source(s) could not be attributed to a "
                "project — `particles memory rescope` lists them."
            )
            steps.append(
                f"{len(report.unattributed)} harvested source(s) are in view for no project. "
                "If they all belong to one, `particles memory rescope --default-key KEY` "
                "attributes them; `--assign ENTRY_ID KEY` attributes one."
            )
        if orphaned:
            steps.append(
                f"{len(orphaned)} source(s) carry only keys of projects that no longer exist: "
                "`particles memory rescope` lists them, `--assign ENTRY_ID KEY` places one."
            )
        if lapsed:
            steps.append(
                f"{lapsed} belief(s) are lapsed (no source states them any more): "
                "`particles corpus refresh --backfill-cascade` retires them."
            )
        if marks.marked:
            steps.append(
                f"{len(marks.marked)} earlier agent write(s) read as global: "
                "`particles memory rescope --mark-agent-writes --dry-run` previews marking them "
                "unattributed."
            )

    try:
        run(_go())
    except Exception as exc:  # noqa: BLE001 — the install itself already succeeded
        log.debug("rescope failed", exc_info=True)
        _say(f"Project-key rescope failed ({exc}) — run `particles memory rescope` later.")


def _offer_first_run_audit(handle: str, only_memory_dir: Path | None = None) -> None:
    """First-run memory-audit hand-off (implemented).

    Runs ``particles audit`` over every ``~/.claude/projects/*/memory/``
    directory against the freshly registered store, estimate/confirm gate
    included, and renders the census as init's closing screen. A refusal
    (no API key), a declined confirm, or any audit failure never fails the
    install — init's own work is already done.
    """
    from particles.api.cli.audit import run_first_run_audit

    try:
        if only_memory_dir is None:
            run_first_run_audit(handle)
        else:
            run_first_run_audit(handle, only_memory_dir=only_memory_dir)
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


def _remove(settings_path: Path, *, dry_run: bool, explicit_config: bool = False) -> dict[str, Any]:
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
        # Shown as the install's own hook map shows them: an engine token a
        # command pins never reaches the printed result (2026-10-30-1, AB26).
        "hooks_removed": [_shown_command(command) for command in installed],
        "state_dir_kept": str(state_dir()),
        "next_steps": [],
    }

    # The hooks name the config they were installed against; revert that one.
    # Since the working directory is never read, so a remove run from
    # anywhere else would otherwise not see the store the install enabled.
    pinned = next(
        (
            pins["PARTICLES_CONFIG"]
            for c in installed
            if "PARTICLES_CONFIG" in (pins := command_env_pins(c))
        ),
        None,
    )
    if pinned and not _hook_config_pin_trusted(
        pinned, settings_path, explicit_config=explicit_config
    ):
        _say(
            f"Ignoring the config pin {pinned} in {settings_path}: a repository can ship "
            "that file. Pass --config to name the config the install pinned."
        )
        pinned = None
    if pinned and os.environ.get("PARTICLES_CONFIG") != pinned:
        os.environ["PARTICLES_CONFIG"] = pinned
        reset_config()

    # Revert the store auto-create only while the store is still empty
    # (a store holding data is never deleted).
    config_path = _config_yaml_path()
    revert_config_text: str | None = None
    store_is_empty = False
    project_scope = _is_project_scope(settings_path)
    # A project-scope settings file is the repository's to ship, so the store
    # its hooks name is reverted only when this project's own install recorded
    # creating it (security review 2026-10-24-2, P24 / N25).
    revertible = not project_scope or (
        handle is not None and _project_created_store(Path.cwd()) == handle
    )
    if handle and project_scope and not revertible and handle in get_config().storage.stores:
        _say(
            f"Keeping store '{handle}': no project-scope install recorded creating it here, "
            "so its configuration and database are left as they are."
        )
    if revertible and handle and handle != "default" and handle in get_config().storage.stores:
        store_is_empty = run(_store_is_empty(handle))
        if store_is_empty and config_path.exists():
            new_text = disable_memory_store_text(config_path.read_text(encoding="utf-8"), handle)
            if new_text != config_path.read_text(encoding="utf-8"):
                revert_config_text = new_text

    if dry_run:
        if installed:
            _say(f"--dry-run: would write {settings_path} (values it does not write hidden):")
            _say(_shown_settings_json(stripped))
        if revert_config_text is not None:
            _say(f"--dry-run: would write {config_path} (store auto-create revert):")
            _say(revert_config_text if revert_config_text else "(empty file)")
        elif revertible and handle and not store_is_empty:
            _say(f"--dry-run: store '{handle}' holds data — its configuration would be kept.")
        result["store_config_reverted"] = revert_config_text is not None
        if not project_scope:
            _unregister_mcp_server(result, dry_run=True)
        return result

    # Only a user-scope install registers the MCP server, so only a user-scope
    # remove reverts it (security review 2026-10-24-2, P23).
    if not project_scope:
        _unregister_mcp_server(result, dry_run=False)
    if installed:
        _write_settings(settings_path, render_settings_json(stripped))
        noun = "entry" if len(installed) == 1 else "entries"
        _say(f"Removed {len(installed)} Particles-owned hook {noun} from {settings_path}.")

    if revert_config_text is not None and handle:
        config_path.write_text(revert_config_text, encoding="utf-8")
        _delete_empty_sqlite_store(handle)
        if project_scope:
            _forget_project_created_store(Path.cwd())
        reset_config()
        _say(f"Reverted the '{handle}' store auto-create in {config_path} (store was empty).")
    elif revertible and handle and handle in get_config().storage.stores and not store_is_empty:
        _say(
            f"Store '{handle}' holds data — kept its configuration and database "
            f"(a store holding data is never deleted)."
        )
    _say(f"State directory {state_dir()} kept (hook log history); delete it manually.")
    result["store_config_reverted"] = revert_config_text is not None and handle is not None
    if revertible and handle and handle in get_config().storage.stores and not store_is_empty:
        result["next_steps"].append(
            f"Store '{handle}' holds data and was kept — delete it by hand if you meant to."
        )
    result["next_steps"].append(
        f"State directory {state_dir()} kept (hook log history); delete it manually."
    )
    return result


def _hook_config_pin_trusted(pinned: str, settings_path: Path, *, explicit_config: bool) -> bool:
    """Whether ``--remove`` may switch this process to a hook command's config pin.

    An explicit ``--config`` always wins. The user-level settings file is the
    operator's own, so its pin is taken as written. A project-scope file is
    ``.claude/settings.local.json`` in the working directory, which a cloned
    repository can ship; its pin is taken only when it names the user config
    file, since anything else would let the clone choose which store the
    remove judges empty and deletes (security review N25).
    """
    if explicit_config:
        return False
    if not _is_project_scope(settings_path):
        return True
    pin = Path(pinned).expanduser()
    if not pin.is_absolute():
        return False
    try:
        return pin.resolve() == user_config_path().expanduser().resolve()
    except OSError:
        return False


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
    from sqlalchemy.exc import DatabaseError

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
    except DatabaseError:
        # No Particles tables. Empty only when the file holds no tables at all
        # (an install that never reached `db init`); a database with someone
        # else's schema is never judged empty, so it is never deleted
        # (security review N25).
        return dsn is not None and dsn.startswith("sqlite") and _sqlite_has_no_tables(dsn)
    return not entries and not particles


def _sqlite_has_no_tables(dsn: str) -> bool:
    """Whether the SQLite file a DSN names opens as a database with no tables."""
    import sqlite3

    path = _sqlite_file_path(dsn)
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return False
    return bool(row) and row[0] == 0


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
