# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the installer and Claude Code harness (security review 2026-10-28-1).

Each test states a safe property. Every test runs under a temporary ``HOME``,
runs no real ``claude`` CLI (the suite's autouse recorder stands in for it)
and makes no model call.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import stat
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.config import reset_config
from particles.corpus.fetch import RefreshRefusal, _store_file_refusal
from particles.corpus.redaction import redact_secrets


def _engine_key(role: str = "agent") -> str:
    secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    return f"ptk_{role}_{secrets.token_hex(6)}_{secret}"


def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_config()
    return home


# ---------------------------------------------------------------------------
# O25 (G11, G28): the harvest mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("before", ["a=3D", "a\\u0026", "x\\n", "a%3D", "token "])
def test_a_key_after_any_escape_is_masked(before: str) -> None:
    """A per-client key directly after an escape ending in a letter or digit is masked."""
    key = _engine_key()
    assert key[-43:] not in redact_secrets(f"{before}{key}")


def test_a_key_after_a_backslash_escape_in_a_tool_command_is_not_harvested() -> None:
    """A key after a backslash escape in a Bash command never reaches the distilled deposit."""
    from particles.api.cli.hook import _distilled_session

    key = _engine_key()
    entry = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {"command": "printf 'x\\n" + key + "' > f"},
                }
            ]
        },
    }
    text = _distilled_session(json.dumps(entry) + "\n", "s1")
    assert "[tool: Bash" in text
    assert key not in text


# ---------------------------------------------------------------------------
# G41: the harvest file URI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["\r\n", "\t", "\n"])
def test_a_harvest_uri_holding_a_raw_control_character_is_refused(raw: str) -> None:
    """A raw tab, CR or LF fails the harvest shape, though ``urlsplit`` strips it."""
    from particles.api.app import _is_harvest_uri

    assert _is_harvest_uri("file:///home/u/.claude/projects/x/memory/a.md")
    assert not _is_harvest_uri(f"file:///home/u/.claude/projects/x/memory/a{raw}b.md")


def test_a_harvester_deposit_never_stores_a_uri_holding_a_raw_control_character(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A harvester key's text deposit whose URI holds a raw control character is refused."""
    from fastapi.testclient import TestClient

    from particles.api import _keys
    from particles.api._keys import Role

    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n", encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(tmp_path / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", "audit8-legacy")
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _keys.reset_key_cache()

    async def _create() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    async def _uris() -> list[str]:
        from particles.corpus.store import list_entries
        from particles.db import get_engine, session_scope

        async with session_scope() as s:
            out = [e.uri_r or "" for e in await list_entries(s)]
        await get_engine().dispose()
        return out

    try:
        asyncio.run(_create())
        token, _ = _keys.mint(Role.AGENT, "claude-code-hook", reserved_ok=True)
        from particles.api.app import app as engine_app

        client = TestClient(
            engine_app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)
        )
        resp = client.post(
            "/corpus/deposit/text",
            json={
                "text": "memory text",
                "uri_r": "file:///home/u/.claude/projects/x/memory/a\r\nb\t.md",
                "source_type": "LOCAL_MARKDOWN",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        stored = asyncio.run(_uris())
        assert resp.status_code == 403, (resp.status_code, resp.text)
        assert not any(ch in u for u in stored for ch in "\r\n\t"), stored
    finally:
        _keys.reset_key_cache()
        _keys.forget_key_files()
        reset_config()


# ---------------------------------------------------------------------------
# O31 (E22, G10): engine URLs the install, the doctor and the MCP server print
# ---------------------------------------------------------------------------

_PASSWORD = "4417/s3cr3t"


@pytest.mark.parametrize(
    "url",
    [
        f"http://user:{_PASSWORD}@engine:8000/",
        f"http://user:{_PASSWORD}@engine/",
        "http://user:4417?s3cr3t@engine.example/",
        "http://user:4417#s3cr3t@engine.example/",
        f"user:{_PASSWORD}@engine.example:8000",
    ],
)
def test_url_without_userinfo_never_keeps_the_password(url: str) -> None:
    """No part of a userinfo password survives ``url_without_userinfo``."""
    from particles.url_safety import url_without_userinfo

    shown = url_without_userinfo(url)
    assert "s3cr3t" not in shown
    assert "engine" in shown


def test_the_install_note_never_echoes_a_password_before_a_single_label_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The install's inherited-engine note names no password for a container-name engine."""
    import particles.api.cli.init as init_mod

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", f"http://user:{_PASSWORD}@engine/")
    result: dict[str, Any] = {"next_steps": []}
    init_mod._note_inherited_engine_url(result, tmp_path / "config.yaml")
    assert result["next_steps"]
    assert "s3cr3t" not in " ".join(result["next_steps"])


def test_the_mcp_server_log_never_echoes_a_password_before_a_single_label_host(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The MCP server's inherited-engine warning names no password."""
    from particles.mcp import server as server_mod

    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", f"http://user:{_PASSWORD}@engine/")
    with caplog.at_level(logging.WARNING):
        server_mod._note_the_engine_origin()
    assert caplog.records
    assert "s3cr3t" not in caplog.text


# ---------------------------------------------------------------------------
# O27 (E19, L22): the registration command the install prints
# ---------------------------------------------------------------------------

_WANTED_ENV = {
    "PARTICLES_CONFIG": "/cfg/config.yaml",
    "DATABASE_URL": "sqlite+aiosqlite:////data/particles.db",
    "PARTICLES_ENGINE_BASE_URL": "",
}


def _home_with_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> Path:
    home = _home(tmp_path, monkeypatch)
    server = {"command": "/opt/venv/bin/particles", "args": ["mcp", "serve"], "env": env}
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"particles": server}}))
    return home


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_rebind_never_prints_the_token_the_registration_carries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_real_claude_cli: list[list[str]],
    dry_run: bool,
) -> None:
    """A re-run that rebinds a registration holding an engine token prints no token."""
    import particles.api.cli.init as init_mod

    key = _engine_key()
    _home_with_registration(
        tmp_path,
        monkeypatch,
        {"PARTICLES_CONFIG": "/cfg/config.yaml", "PARTICLES_ENGINE_TOKEN": key},
    )
    result: dict[str, Any] = {"next_steps": []}
    init_mod._register_mcp_server(
        result, executable="/opt/venv/bin/particles", env=dict(_WANTED_ENV), dry_run=dry_run
    )
    assert result["mcp_server"]["action"] == "bind"
    printed = json.dumps(result) + capsys.readouterr().out
    assert key not in printed
    assert "PARTICLES_ENGINE_TOKEN=<hidden>" in result["mcp_server"]["command"]
    if not dry_run:
        # The subprocess still gets the real value.
        assert f"PARTICLES_ENGINE_TOKEN={key}" in no_real_claude_cli[-1]


def test_a_failed_rebind_never_prints_the_token_in_its_next_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the add and the restore both fail, the next steps name no engine token."""
    import particles.api.cli.init as init_mod

    key = _engine_key()
    _home_with_registration(
        tmp_path,
        monkeypatch,
        {"PARTICLES_CONFIG": "/cfg/config.yaml", "PARTICLES_ENGINE_TOKEN": key},
    )
    monkeypatch.setattr(init_mod, "_claude_mcp", lambda argv: argv[:2] == ["mcp", "remove"])
    result: dict[str, Any] = {"next_steps": []}
    init_mod._register_mcp_server(
        result, executable="/opt/venv/bin/particles", env=dict(_WANTED_ENV), dry_run=False
    )
    assert result["mcp_server"]["action"] == "manual"
    assert key not in json.dumps(result)
    assert any("<hidden>" in step for step in result["next_steps"])


def test_the_printed_command_keeps_the_pins_the_install_chose() -> None:
    """The config path and the store URL (without userinfo) are shown as they are."""
    import particles.api.cli.init as init_mod

    shown = init_mod._shown_claude_command(
        ["mcp", "add", "--scope", "user", "particles",
         "-e", "DATABASE_URL=postgresql://app:pw123456@db.internal/p",
         "-e", "PARTICLES_CONFIG=/cfg/config.yaml",
         "-e", "PARTICLES_ENGINE_BASE_URL=",
         "--", "/opt/venv/bin/particles", "mcp", "serve"]
    )  # fmt: skip
    assert "pw123456" not in shown
    assert "DATABASE_URL=postgresql://db.internal/p" in shown
    assert "PARTICLES_CONFIG=/cfg/config.yaml" in shown
    assert "PARTICLES_ENGINE_BASE_URL=" in shown


# ---------------------------------------------------------------------------
# O68: the suite's claude stand-in refuses what the installed CLI refuses
# ---------------------------------------------------------------------------


def test_the_autouse_recorder_refuses_a_remove_of_a_server_that_is_not_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_real_claude_cli: list[list[str]]
) -> None:
    import particles.api.cli.init as init_mod

    _home(tmp_path, monkeypatch)
    assert not init_mod._claude_mcp(["mcp", "remove", "--scope", "user", "no-such-server"])


def test_the_autouse_recorder_refuses_a_name_the_cli_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_real_claude_cli: list[list[str]]
) -> None:
    import particles.api.cli.init as init_mod

    _home(tmp_path, monkeypatch)
    assert not init_mod._claude_mcp(
        ["mcp", "add", "--scope", "user", "my.server", "-e", "A=1", "--", "/p", "mcp", "serve"]
    )


def test_the_autouse_recorder_refuses_an_unknown_option_before_the_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_real_claude_cli: list[list[str]]
) -> None:
    import particles.api.cli.init as init_mod

    _home(tmp_path, monkeypatch)
    assert not init_mod._claude_mcp(
        ["mcp", "add", "--scope", "user", "particles", "--project-observer", "cwd",
         "--", "/p", "mcp", "serve"]
    )  # fmt: skip


def test_the_autouse_recorder_refuses_a_second_add_and_takes_a_remove_of_what_it_added(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_real_claude_cli: list[list[str]]
) -> None:
    import particles.api.cli.init as init_mod

    _home(tmp_path, monkeypatch)
    add = ["mcp", "add", "--scope", "user", "particles", "--", "/p", "mcp", "serve"]
    assert init_mod._claude_mcp(add)
    assert not init_mod._claude_mcp(add)
    assert init_mod._claude_mcp(["mcp", "remove", "--scope", "user", "particles"])


# ---------------------------------------------------------------------------
# O26 (R18): the project settings read
# ---------------------------------------------------------------------------


def test_the_project_settings_read_comes_from_the_checked_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes the project install merges are read from the .claude directory it judged."""
    import typer

    import particles.api.cli.init as init_mod

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    claude = project / ".claude"
    claude.mkdir(parents=True)
    (claude / "settings.local.json").write_text('{"origin": "checked"}')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "settings.local.json").write_text('{"origin": "elsewhere"}')
    monkeypatch.chdir(project)

    real_refuse = init_mod._refuse_foreign_settings_directory

    def check_then_swap(directory: Path) -> None:
        real_refuse(directory)
        os.rename(claude, project / ".claude-orig")
        os.symlink(elsewhere, claude)

    monkeypatch.setattr(init_mod, "_refuse_foreign_settings_directory", check_then_swap)
    settings_path = claude / "settings.local.json"
    try:
        with pytest.raises(typer.Exit):
            init_mod._refuse_symlinked_project_settings(settings_path)
        cached = init_mod._CHECKED_PROJECT_SETTINGS.get(settings_path)
    finally:
        init_mod._CHECKED_PROJECT_SETTINGS.clear()
        init_mod._CHECKED_SETTINGS_DIRECTORIES.clear()
    assert cached is None or b"elsewhere" not in cached


def test_a_directory_renamed_in_after_the_check_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real directory (not a link) swapped in by name after the check is refused too."""
    import typer

    import particles.api.cli.init as init_mod

    project = tmp_path / "project"
    claude = project / ".claude"
    claude.mkdir(parents=True)
    (claude / "settings.local.json").write_text('{"origin": "checked"}')
    other = tmp_path / "other"
    other.mkdir()
    (other / "settings.local.json").write_text('{"origin": "elsewhere"}')
    monkeypatch.chdir(project)
    try:
        init_mod._refuse_foreign_settings_directory(claude)
        os.rename(claude, project / ".claude-orig")
        os.rename(other, claude)
        with pytest.raises(typer.Exit):
            init_mod._read_checked_project_settings(claude / "settings.local.json")
    finally:
        init_mod._CHECKED_SETTINGS_DIRECTORIES.clear()


# ---------------------------------------------------------------------------
# O29, O30 (G14, E23): the config and store an install pins
# ---------------------------------------------------------------------------


def test_the_install_names_an_inherited_config_inside_a_work_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An install whose environment chose a config file inside a git work tree says so."""
    _home(tmp_path, monkeypatch)
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    cfg = clone / "config.yaml"
    store = tmp_path / "stores" / "m.db"
    store.parent.mkdir()
    cfg.write_text(f"storage:\n  database_url: sqlite+aiosqlite:///{store}\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    reset_config()
    out = CliRunner().invoke(
        app, ["init", "claude-code", "--dry-run", "--json", "--no-skills", "--store", "default"]
    )
    assert out.exit_code == 0, out.output
    payload = json.loads(out.stdout)
    assert str(cfg.resolve()) in json.dumps(payload.get("hooks", {}))
    assert any(
        str(clone.resolve()) in step and "PARTICLES_CONFIG" in step
        for step in payload["next_steps"]
    )


def test_an_explicit_config_is_not_reported_as_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--config`` names the file the operator chose, so no inherited-config note is added."""
    home = _home(tmp_path, monkeypatch)
    cfg = home / "chosen.yaml"
    cfg.write_text("")
    monkeypatch.delenv("PARTICLES_CONFIG", raising=False)
    out = CliRunner().invoke(
        app,
        ["init", "claude-code", "--dry-run", "--json", "--no-skills", "--store", "default",
         "--config", str(cfg)],
    )  # fmt: skip
    assert out.exit_code == 0, out.output
    steps = json.loads(out.stdout)["next_steps"]
    assert not any("set in this install's environment" in step for step in steps)


def _named_store_in_clone(tmp_path: Path) -> tuple[Path, Path]:
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    db = clone / "work.db"
    db.write_bytes(b"")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"storage:\n  stores:\n    work: sqlite+aiosqlite:///{db}\n")
    return cfg, db


def test_the_doctor_reports_a_named_store_inside_a_work_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The doctor's named-store check reports a store file the refresh refuses as a clone's."""
    from particles.api.cli import hook as hook_mod
    from particles.corpus.fetch import _store_file_refusal

    _home(tmp_path, monkeypatch)
    cfg, db = _named_store_in_clone(tmp_path)
    assert _store_file_refusal(db) is not None
    problem = hook_mod._named_store_pin_problem({"PARTICLES_CONFIG": str(cfg)}, "work")
    assert problem is not None
    assert "git work tree" in problem


def test_the_doctor_names_a_named_store_writable_by_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A named store file other users may write is reported as such."""
    from particles.api.cli import hook as hook_mod

    home = _home(tmp_path, monkeypatch)
    db = home / "stores" / "work.db"
    db.parent.mkdir()
    db.write_bytes(b"")
    db.chmod(0o666)
    cfg = home / "config.yaml"
    cfg.write_text(f"storage:\n  stores:\n    work: sqlite+aiosqlite:///{db}\n")
    problem = hook_mod._named_store_pin_problem({"PARTICLES_CONFIG": str(cfg)}, "work")
    assert problem is not None
    assert "other users may write" in problem


def test_the_install_warns_on_a_named_store_inside_a_work_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The install's named-store warning fires for a store file the refresh refuses."""
    import particles.api.cli.init as init_mod

    _home(tmp_path, monkeypatch)
    cfg, _db = _named_store_in_clone(tmp_path)
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    reset_config()
    init_mod._warn_if_store_dsn_cwd_relative("work", {"PARTICLES_CONFIG": str(cfg)})
    err = capsys.readouterr().err
    assert "work" in err
    assert "git work tree" in err


# ---------------------------------------------------------------------------
# O65 (L32): the role hook doctor reports for a pinned token
# ---------------------------------------------------------------------------


def test_the_doctor_reports_a_reader_key_the_installed_hooks_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reader key pinned in the installed hook commands is reported, whatever this shell holds."""
    from particles.api.cli import hook as hook_mod
    from particles.api.cli._claude_code import build_hook_commands

    home = _home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("engine:\n  base_url: https://engine.invalid\n")
    commands = build_hook_commands(
        "/opt/venv/bin/particles",
        "default",
        {
            "PARTICLES_CONFIG": str(config),
            "PARTICLES_ENGINE_BASE_URL": "",
            "PARTICLES_ENGINE_TOKEN": _engine_key("read"),
        },
    )
    settings = {
        "hooks": {
            event: [{"hooks": [{"type": "command", "command": cmd}]}]
            for event, cmd in commands.items()
        }
    }
    (home / ".claude").mkdir()
    (home / ".claude" / "settings.json").write_text(json.dumps(settings))
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", _engine_key("agent"))
    reset_config()
    line = hook_mod._engine_line()
    assert "reader key" in line
    assert "installed hooks pin" in line


# ---------------------------------------------------------------------------
# O66: the executable an `init --remove` replay runs
# ---------------------------------------------------------------------------


def _exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)


def test_a_link_chain_through_safe_directories_is_trusted(tmp_path: Path) -> None:
    from particles.api.cli.init import _is_installed_executable

    real = tmp_path / "c" / "particles"
    _exe(real)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "link").symlink_to(real)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "particles").symlink_to(tmp_path / "b" / "link")
    assert _is_installed_executable(tmp_path / "a" / "particles")


def test_an_intermediate_link_in_a_directory_others_may_write_is_not_trusted(
    tmp_path: Path,
) -> None:
    """A replayed executable is refused when any link in its chain sits where others may write."""
    from particles.api.cli.init import _is_installed_executable

    real = tmp_path / "c" / "particles"
    _exe(real)
    hop = tmp_path / "b"
    hop.mkdir()
    (hop / "link").symlink_to(real)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "particles").symlink_to(hop / "link")
    hop.chmod(0o777)
    try:
        assert not _is_installed_executable(tmp_path / "a" / "particles")
    finally:
        hop.chmod(0o755)


# ---------------------------------------------------------------------------
# O67 (Z16): the projection's contained write with no search-only open
# ---------------------------------------------------------------------------


def _locked_repository(tmp_path: Path) -> tuple[Path, Path]:
    locked = tmp_path / "locked"
    root = locked / "repo"
    (root / ".git").mkdir(parents=True)
    return locked, root


def test_the_root_walk_never_opens_the_root_by_name_without_a_search_only_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no O_PATH / O_SEARCH, an unreadable ancestor never makes the walk open the root by name.

    The walk resolves only the unreadable stretch below its last descriptor.
    """
    from particles.operations.projection import paths

    locked, root = _locked_repository(tmp_path)
    real_root = Path(os.path.realpath(root))
    opened: list[str] = []
    real_open = os.open

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        opened.append(os.fsdecode(path))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(paths, "_SEARCH_ONLY", 0)
    locked.chmod(0o311)
    try:
        monkeypatch.setattr(paths.os, "open", recording_open)
        paths.write_text_no_follow(root / "out.md", "projected\n", root=root)
    finally:
        monkeypatch.undo()
        locked.chmod(0o755)
    assert (root / "out.md").read_text() == "projected\n"
    assert str(real_root) not in opened


def test_the_root_walk_refuses_where_the_descriptor_path_cannot_be_rechecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no search-only open and no kernel-path recheck, the contained write is refused."""
    from particles.corpus.local_read import LocalReadRefused
    from particles.operations.projection import paths

    locked, root = _locked_repository(tmp_path)
    monkeypatch.setattr(paths, "_SEARCH_ONLY", 0)
    monkeypatch.setattr(paths, "_descriptor_path", lambda fd: None)
    locked.chmod(0o311)
    try:
        with pytest.raises(LocalReadRefused):
            paths.write_text_no_follow(root / "out.md", "projected\n", root=root)
    finally:
        locked.chmod(0o755)
    assert not (root / "out.md").exists()


# ---------------------------------------------------------------------------
# Fix chip 5 (O60, G14, G55, O15, O61): store wording, the config pin's owner
# and mode, the hook commands in --json, and the PATH lookup
# ---------------------------------------------------------------------------

_DSN_PASSWORD = "s3cr3t-db-password-7f1c"


def test_the_doctor_names_a_store_other_users_may_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The doctor words a store others may write as such, with the fix, as the install does."""
    from particles.api.cli import hook as hook_mod

    _home(tmp_path, monkeypatch)
    stores = tmp_path / "stores"
    stores.mkdir()
    store = stores / "particles.db"
    store.write_bytes(b"")
    os.chmod(store, 0o664)
    assert _store_file_refusal(store) is RefreshRefusal.STORE_WRITABLE_BY_OTHERS
    env = {"PARTICLES_CONFIG": str(tmp_path / "c.yaml"), "DATABASE_URL": f"sqlite:///{store}"}
    problem = hook_mod._store_pin_problem(env)
    assert problem is not None
    assert "other users may write" in problem
    assert "chmod go-w" in problem


def test_an_unopenable_store_is_not_called_another_users_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store the install cannot open is worded as unjudgeable, not as another user's (G14)."""
    from particles.api.cli import hook as hook_mod

    _home(tmp_path, monkeypatch)
    stores = tmp_path / "stores"
    stores.mkdir()
    store = stores / "particles.db"
    store.write_bytes(b"")
    os.chmod(store, 0o000)
    try:
        if os.access(store, os.R_OK):
            pytest.skip("running as a user every mode admits")
        env = {"PARTICLES_CONFIG": str(tmp_path / "c.yaml"), "DATABASE_URL": f"sqlite:///{store}"}
        problem = hook_mod._store_pin_problem(env)
    finally:
        os.chmod(store, 0o600)
    assert problem is not None
    assert "cannot open" in problem
    assert "another user" not in problem


def test_the_doctor_judges_a_config_pin_other_users_may_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config pin is judged for owner and mode, not only for a work tree (G55)."""
    from particles.api.cli import hook as hook_mod

    _home(tmp_path, monkeypatch)
    folder = tmp_path / "operator"
    folder.mkdir()
    config = folder / "config.yaml"
    config.write_text("# operator\n")
    os.chmod(config, 0o644)
    assert hook_mod._config_pin_problem(str(config)) is None
    os.chmod(config, 0o666)
    problem = hook_mod._config_pin_problem(str(config))
    assert problem is not None
    assert "other users may write" in problem


def test_the_install_prints_no_store_dsn_password_in_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No part of the install's output (narration, next steps, --json) names a store DSN's
    password."""
    _home(tmp_path, monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("# no store named\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv(
        "DATABASE_URL", f"postgresql+asyncpg://particles:{_DSN_PASSWORD}@db.invalid/p"
    )
    reset_config()
    out = CliRunner().invoke(
        app, ["init", "claude-code", "--store", "default", "--dry-run", "--json", "--no-skills"]
    )
    assert out.exit_code == 0, out.output
    payload: dict[str, Any] = json.loads(out.stdout)
    assert not any(_DSN_PASSWORD in step for step in payload["next_steps"])
    assert _DSN_PASSWORD not in json.dumps(payload.get("mcp_server", {}))
    assert _DSN_PASSWORD not in out.stdout
    assert _DSN_PASSWORD not in out.stderr


def test_the_installer_runs_no_claude_found_through_an_empty_path_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installer never runs a ``claude`` that only a relative or empty PATH entry finds."""
    import particles.api.cli.init as init_mod

    _home(tmp_path, monkeypatch)
    # The suite's recorder replaced the call; take back only the original
    # function, so the real lookup runs against a stub on a temporary PATH.
    original = next(
        saved
        for obj, name, saved in monkeypatch._setattr  # type: ignore[attr-defined]
        if obj is init_mod and name == "_claude_mcp"
    )
    trusted = tmp_path / "bin"
    trusted.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    marker = tmp_path / "ran"
    for name in ("claude", "particles"):
        stub = project / name
        stub.write_text(f"#!/bin/sh\necho ran > '{marker}'\n")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", os.pathsep.join([str(trusted), "", ".", "/usr/bin", "/bin"]))
    monkeypatch.chdir(project)
    original(["mcp", "list"])
    assert not marker.exists()
    assert init_mod._trusted_which("particles") is None
