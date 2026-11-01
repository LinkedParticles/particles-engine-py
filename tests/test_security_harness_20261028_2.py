# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the installer and Claude Code harness (security review 2026-10-28-2).

Each test states a safe property. Every test runs under a temporary ``HOME``,
runs no real ``claude`` CLI (the suite's autouse recorder stands in for it, and
the one test that runs the installer's own subprocess call runs a stub) and
makes no model call.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli.init import _claude_mcp as _unrecorded_claude_mcp
from particles.config import reset_config


@pytest.fixture(autouse=True)
def _temporary_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    return home


def _closed_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def _git_free_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


# --------------------------------------------------------------------------- #
# AA10: the installer's claude subprocess
# --------------------------------------------------------------------------- #


def test_the_installer_claude_call_runs_no_interpreter_from_an_untrusted_path_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installer's ``claude`` finds its interpreter only on the trusted ``PATH`` entries."""
    import particles.api.cli.init as init_mod

    trusted = tmp_path / "bin"
    trusted.mkdir(mode=0o755)
    stub = trusted / "claude"
    stub.write_text("#!/usr/bin/env node\nconsole.log('stub')\n")
    stub.chmod(0o755)
    clone = tmp_path / "clone"
    clone.mkdir()
    marker = tmp_path / "planted-node-ran"
    planted = clone / "node"
    planted.write_text(f"#!/bin/sh\n/usr/bin/touch '{marker}'\nexit 0\n")
    planted.chmod(0o755)
    monkeypatch.chdir(clone)
    monkeypatch.setenv("PATH", f"{trusted}{os.pathsep}.")
    assert init_mod._trusted_which("claude") == str(stub)
    _unrecorded_claude_mcp(["mcp", "list"])
    assert not marker.exists(), "a program found through '.' ran under the installer"


# --------------------------------------------------------------------------- #
# AA14 / G55: the install judges an inherited config pin by its name
# --------------------------------------------------------------------------- #


def test_the_install_warns_on_a_config_pin_the_refresh_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A config pin reached through a link in an open directory is warned about at install."""
    import particles.api.cli.init as init_mod
    from particles.api.cli import hook as hook_mod
    from particles.corpus import fetch

    monkeypatch.delenv("DATABASE_URL", raising=False)
    real = _closed_dir(tmp_path / "closed") / "config.yaml"
    real.write_text("", encoding="utf-8")
    os.chmod(real, 0o600)
    opened = tmp_path / "open"
    opened.mkdir()
    link = opened / "config.yaml"
    link.symlink_to(real)
    os.chmod(opened, 0o777)
    monkeypatch.setenv("PARTICLES_CONFIG", str(link))
    reset_config()
    try:
        refresh = fetch.config_file_judgement()
        doctor = hook_mod._config_pin_problem(str(link))
        result: dict[str, Any] = {"next_steps": []}
        init_mod._note_config_origin(result, link, explicit_config=False)
    finally:
        os.chmod(opened, 0o700)
    err = capsys.readouterr().err
    assert refresh is not None
    assert doctor is not None
    assert "Warning:" in err, err


# --------------------------------------------------------------------------- #
# AA20: hook doctor reads a clone's project settings through a checked descriptor
# --------------------------------------------------------------------------- #


def _engine_key(role: str) -> str:
    secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    return f"ptk_{role}_{secrets.token_hex(6)}_{secret}"


def _clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    (clone / ".claude").mkdir()
    monkeypatch.chdir(clone)
    reset_config()
    return clone


def _never_blocks(fifo: Path, check: Any) -> bool:
    """Whether ``check()`` returns within three seconds while ``fifo`` has no writer."""
    done = threading.Event()

    def run() -> None:
        check()
        done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(3)
    if done.is_set():
        return True
    fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)  # release the reader
    os.write(fd, b"{}")
    os.close(fd)
    worker.join(3)
    return False


def test_the_doctor_never_reads_a_project_settings_file_through_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clone's linked ``.claude/settings.json`` is never read as installed hooks."""
    from particles.api.cli import hook as hook_mod
    from particles.api.cli._claude_code import build_hook_commands

    clone = _clone(tmp_path, monkeypatch)
    commands = build_hook_commands(
        "/opt/venv/bin/particles", "default", {"PARTICLES_ENGINE_TOKEN": _engine_key("read")}
    )
    target = tmp_path / "elsewhere.json"
    target.write_text(
        json.dumps(
            {
                "hooks": {
                    ev: [{"hooks": [{"type": "command", "command": c}]}]
                    for ev, c in commands.items()
                }
            }
        )
    )
    (clone / ".claude" / "settings.json").symlink_to(target)
    assert hook_mod._hook_pinned_tokens() == []


def test_the_doctor_config_pin_check_never_blocks_on_a_fifo_a_clone_ships(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FIFO at a clone's ``.claude/settings.json`` never holds the doctor's config-pin check."""
    from particles.api.cli import hook as hook_mod

    fifo = _clone(tmp_path, monkeypatch) / ".claude" / "settings.json"
    os.mkfifo(fifo)
    assert _never_blocks(fifo, hook_mod._check_config_pins)


def test_the_doctor_config_pin_check_never_blocks_on_a_linked_settings_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clone's ``.claude/settings.json`` linked to a blocking file never holds the check."""
    from particles.api.cli import hook as hook_mod

    clone = _clone(tmp_path, monkeypatch)
    fifo = tmp_path / "elsewhere.fifo"
    os.mkfifo(fifo)
    (clone / ".claude" / "settings.json").symlink_to(fifo)
    assert _never_blocks(fifo, hook_mod._check_config_pins)


def test_the_doctor_still_reads_a_regular_project_settings_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: a regular project settings file this account owns is still read."""
    from particles.api.cli import hook as hook_mod
    from particles.api.cli._claude_code import build_hook_commands

    clone = _clone(tmp_path, monkeypatch)
    token = _engine_key("agent")
    commands = build_hook_commands(
        "/opt/venv/bin/particles", "default", {"PARTICLES_ENGINE_TOKEN": token}
    )
    (clone / ".claude" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    ev: [{"hooks": [{"type": "command", "command": c}]}]
                    for ev, c in commands.items()
                }
            }
        )
    )
    assert hook_mod._hook_pinned_tokens() == [token]


# --------------------------------------------------------------------------- #
# AA21: --config replaces the whole registration env
# --------------------------------------------------------------------------- #

_THEIRS = {
    "PARTICLES_CONFIG": "/clone/config.yaml",
    "DATABASE_URL": "sqlite+aiosqlite:////clone/particles.db",
    "PARTICLES_ENGINE_BASE_URL": "",
    "PYTHONPATH": "/clone/lib",
}
_WANTED = {
    "PARTICLES_CONFIG": "/home/me/.particles/config.yaml",
    "DATABASE_URL": "sqlite+aiosqlite:////home/me/.particles/particles.db",
    "PARTICLES_ENGINE_BASE_URL": "",
}


def _plan(replace: bool) -> Any:
    from particles.api.cli._claude_code import (
        PROJECT_OBSERVER_ARGS,
        McpRegistration,
        plan_mcp_registration,
    )

    existing = McpRegistration(
        name="particles",
        command="/opt/bin/particles",
        args=("mcp", "serve", *PROJECT_OBSERVER_ARGS),
        env=dict(_THEIRS),
    )
    return plan_mcp_registration(
        [existing], executable="/opt/bin/particles", env=_WANTED, replace_config_pin=replace
    )


def test_without_config_a_foreign_config_pin_is_kept() -> None:
    """Control: with no ``--config`` a registration pinned to another config is left as found."""
    assert _plan(False) is None


def test_with_config_the_registration_serves_the_wanted_store() -> None:
    """After ``--config`` the registration's store is the one the wanted config's install pins."""
    plan = _plan(True)
    assert plan is not None
    wanted, _replaced = plan
    assert wanted.env["PARTICLES_CONFIG"] == _WANTED["PARTICLES_CONFIG"]
    assert wanted.env["DATABASE_URL"] == _WANTED["DATABASE_URL"]


def test_with_config_the_registration_keeps_no_env_pin_the_install_does_not_write() -> None:
    """After ``--config`` no pin the install never writes survives from the other config."""
    plan = _plan(True)
    assert plan is not None
    wanted, _replaced = plan
    assert set(wanted.env) <= set(_WANTED)


# --------------------------------------------------------------------------- #
# AA23: no printed value keeps a DSN's query-string password
# --------------------------------------------------------------------------- #

_SECRET = "hunter2s3cr3tvalue"
_DSN = f"postgresql+asyncpg://db.example.invalid/particles?user=me&password={_SECRET}"


def test_a_shown_registration_value_never_keeps_a_userinfo_password() -> None:
    """Control: a DSN's userinfo password is not shown."""
    from particles.api.cli.init import _shown_env_value

    shown = _shown_env_value("DATABASE_URL", f"postgresql+asyncpg://me:{_SECRET}@db/particles")
    assert _SECRET not in shown


def test_a_shown_registration_value_never_keeps_a_query_string_password() -> None:
    """A store DSN carrying its password as a query parameter is never shown with it."""
    from particles.api.cli.init import _shown_env_value

    shown = _shown_env_value("DATABASE_URL", _DSN)
    assert _SECRET not in shown
    assert shown.startswith("postgresql+asyncpg://db.example.invalid/particles")


def test_a_printed_hook_command_never_keeps_a_query_string_password() -> None:
    """A hook command printed by ``--json`` never carries a DSN's query-string password."""
    from particles.api.cli._claude_code import build_hook_commands
    from particles.api.cli.init import _shown_command

    command = build_hook_commands("/opt/bin/particles", "default", {"DATABASE_URL": _DSN})[
        "SessionStart"
    ]
    assert _SECRET not in _shown_command(command)


def test_a_printed_hook_command_never_keeps_an_engine_token() -> None:
    """A hook command printed by ``--json`` hides an engine token it pins."""
    from particles.api.cli._claude_code import build_hook_commands
    from particles.api.cli.init import _shown_command

    token = _engine_key("agent")
    command = build_hook_commands(
        "/opt/bin/particles",
        "default",
        {"PARTICLES_CONFIG": "/home/me/config.yaml", "PARTICLES_ENGINE_TOKEN": token},
    )["SessionStart"]
    shown = _shown_command(command)
    assert token not in shown
    assert "PARTICLES_CONFIG=/home/me/config.yaml" in shown


def test_a_dry_run_install_prints_no_query_string_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``init claude-code --dry-run --json`` prints no store password on stdout or stderr."""
    from particles.api.cli import app

    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", _DSN)
    monkeypatch.chdir(tmp_path)
    reset_config()
    out = CliRunner().invoke(
        app,
        ["init", "claude-code", "--dry-run", "--json", "--no-skills", "--store", "default"],
    )
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout)["hooks"]
    assert _SECRET not in out.stdout
    assert _SECRET not in out.stderr


# --------------------------------------------------------------------------- #
# AA58: a replayed program behind another account's link
# --------------------------------------------------------------------------- #

_FOREIGN_UID = 4242


def _exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)


def _foreign_link_owner(monkeypatch: pytest.MonkeyPatch, link: Path) -> None:
    """Report ``link`` as owned by another account, as a second local user's link would be."""
    real_lstat = os.lstat

    def lstat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        st = real_lstat(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(link):
            fields = list(st)
            fields[4] = _FOREIGN_UID  # st_uid
            return os.stat_result(fields)
        return st

    monkeypatch.setattr(os, "lstat", lstat)


@pytest.fixture
def sticky(tmp_path: Path) -> Any:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o1777)
    yield shared
    shared.chmod(0o755)


def test_a_link_this_account_owns_in_a_sticky_directory_is_trusted(
    tmp_path: Path, sticky: Path
) -> None:
    """Control: a chain through this account's link in a sticky directory is accepted."""
    from particles.api.cli.init import _is_installed_executable

    bindir = tmp_path / "bin"
    _exe(bindir / "particles")
    (sticky / "tool").symlink_to(bindir)
    assert _is_installed_executable(sticky / "tool" / "particles")


def test_a_directory_link_another_account_owns_in_a_sticky_directory_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sticky: Path
) -> None:
    """A replayed path through a directory link another account owns is refused."""
    from particles.api.cli.init import _is_installed_executable

    bindir = tmp_path / "bin"
    _exe(bindir / "particles")
    link = sticky / "tool"
    link.symlink_to(bindir)
    _foreign_link_owner(monkeypatch, link)
    assert not _is_installed_executable(link / "particles")


def test_a_final_link_another_account_owns_in_a_sticky_directory_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sticky: Path
) -> None:
    """A replayed executable whose own link another account owns is refused."""
    from particles.api.cli.init import _is_installed_executable

    real = tmp_path / "bin" / "particles"
    _exe(real)
    link = sticky / "particles"
    link.symlink_to(real)
    _foreign_link_owner(monkeypatch, link)
    assert not _is_installed_executable(link)


# --------------------------------------------------------------------------- #
# AA59: the claude CLI recorder honours scope as the CLI does
# --------------------------------------------------------------------------- #

_SERVE = ["--", "/opt/bin/particles", "mcp", "serve"]


def test_the_recorder_refuses_a_remove_under_a_scope_the_add_did_not_use() -> None:
    """A project-scope remove of a server only a user-scope add registered is refused."""
    import particles.api.cli.init as init_mod

    assert init_mod._claude_mcp(["mcp", "add", "--scope", "user", "particles", *_SERVE])
    assert not init_mod._claude_mcp(["mcp", "remove", "--scope", "project", "particles"])
    assert init_mod._claude_mcp(["mcp", "remove", "--scope", "user", "particles"])


def test_the_recorder_refuses_a_scope_the_cli_does_not_declare() -> None:
    """An add whose scope is none of local, user or project is refused."""
    import particles.api.cli.init as init_mod

    assert not init_mod._claude_mcp(["mcp", "add", "--scope", "everywhere", "particles", *_SERVE])
    assert not init_mod._claude_mcp(["mcp", "add", "-t", "carrier-pigeon", "particles", *_SERVE])


def test_the_recorder_refuses_an_unscoped_remove_of_a_name_in_two_scopes() -> None:
    """A remove naming no scope is refused when the name is registered in two scopes."""
    import particles.api.cli.init as init_mod

    assert init_mod._claude_mcp(["mcp", "add", "--scope", "user", "particles", *_SERVE])
    assert init_mod._claude_mcp(["mcp", "add", "particles", *_SERVE])  # local, the default
    assert not init_mod._claude_mcp(["mcp", "remove", "particles"])
    assert init_mod._claude_mcp(["mcp", "remove", "-s", "local", "particles"])
    assert init_mod._claude_mcp(["mcp", "remove", "particles"])


# --------------------------------------------------------------------------- #
# AA60: the MEMORY.md history commit runs no inherited git program
# --------------------------------------------------------------------------- #


def _memory_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    memory = repo / "memory"
    memory.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=_git_free_env())
    (memory / ".gitattributes").write_text("*.md filter=probe\n")
    (memory / "MEMORY.md").write_text("- a belief `p-abcd1234`\n")
    return memory


def _commit(memory: Path) -> str:
    from particles.api.cli._projection_git import commit_projection

    return asyncio.run(
        commit_projection(
            memory, store="default", outcome="ok", body="", snapshot=None, run_id="r1"
        )
    )


def test_a_repository_filter_driver_skips_the_history_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: a filter driver in the repository's own config skips the commit unrun."""
    memory = _memory_repo(tmp_path)
    marker = tmp_path / "ran"
    subprocess.run(
        ["git", "-C", str(memory), "config", "filter.probe.clean", f"touch {marker}; cat"],
        check=True,
        env=_git_free_env(),
    )
    for name in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(name)
    assert _commit(memory) == "skipped: command-driver-configured"
    assert not marker.exists()


def test_a_filter_driver_from_the_inherited_environment_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filter driver named by inherited ``GIT_CONFIG_*`` variables never runs."""
    memory = _memory_repo(tmp_path)
    marker = tmp_path / "ran"
    for name in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "filter.probe.clean")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", f"touch {marker}; cat")
    _commit(memory)
    assert not marker.exists()
