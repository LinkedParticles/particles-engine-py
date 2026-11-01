# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the installer and Claude Code harness (security review 2026-10-27-1).

Each test states a safe property. Every test runs under a temporary ``HOME``,
opens no real store, runs no real ``claude`` CLI (a stub on a temporary
``PATH`` that parses ``mcp add`` as the installed CLI does stands in for it)
and makes no model call.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.engine import make_url
from typer.testing import CliRunner

from particles.api.cli import app
from particles.config import get_config, reset_config
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
    reset_config()
    return home


# ---------------------------------------------------------------------------
# E13, W17, W32, E2, Z24: the MCP registration through a `claude` stub
# ---------------------------------------------------------------------------

_STUB = r'''#!{python}
"""Stand-in for `claude mcp add|remove`, parsing as commander does (variadic -e).

It declares the options the installed CLI declares and refuses any other, and
refuses a name outside [A-Za-z0-9_-]+ (security review 2026-10-28-1, O68).
Servers are kept by scope as the CLI keeps them: ``user`` in the top-level
mcpServers, ``local`` (the default) under the working directory's projects
entry. A remove under a scope that does not hold the name is refused, and one
naming no scope is refused when more than one scope holds it (2026-10-30-1,
AB66). The ``project`` scope's ``.mcp.json`` is not modelled: it is refused.
"""
import json, os, re, sys
path = os.path.join(os.environ["HOME"], ".claude.json")
try:
    data = json.load(open(path))
except Exception:
    data = {{}}
def scope_servers(scope):
    if scope == "user":
        return data.setdefault("mcpServers", {{}})
    project = data.setdefault("projects", {{}}).setdefault(os.getcwd(), {{}})
    return project.setdefault("mcpServers", {{}})
argv = sys.argv[1:]
if argv[:1] != ["mcp"] or len(argv) < 2:
    sys.exit(2)
verb, rest = argv[1], argv[2:]
if verb == "add":
    one_value = {{"-s", "--scope", "-t", "--transport", "--client-id", "--callback-port"}}
    variadic = {{"-e", "--env", "-H", "--header"}}
    flags = {{"--client-secret", "--xaa", "-h", "--help"}}
else:
    one_value, variadic, flags = {{"-s", "--scope"}}, set(), {{"-h", "--help"}}
positional, env, scope, i = [], [], None, 0
while i < len(rest):
    word = rest[i]
    if word == "--":
        positional += rest[i + 1:]
        break
    if word in one_value:
        if i + 1 >= len(rest):
            sys.exit(1)
        if word in ("-s", "--scope"):
            scope = rest[i + 1]
            if scope not in ("local", "user", "project"):
                sys.exit(1)
        i += 2
        continue
    if word in variadic:
        i += 1
        taken = 0
        while i < len(rest) and not (len(rest[i]) > 1 and rest[i].startswith("-")):
            if word in ("-e", "--env"):
                env.append(rest[i])
            i += 1
            taken += 1
        if not taken:
            sys.exit(1)
        continue
    if word in flags:
        i += 1
        continue
    if len(word) > 1 and word.startswith("-"):
        print("error: unknown option '" + word + "'", file=sys.stderr)
        sys.exit(1)
    positional.append(word)
    i += 1
if not positional or not re.fullmatch(r"[A-Za-z0-9_-]+", positional[0]):
    sys.exit(1)
if scope == "project":
    sys.exit(1)
if verb == "remove":
    if len(positional) != 1:
        sys.exit(1)
    held = [s for s in ((scope,) if scope else ("local", "user"))
            if positional[0] in scope_servers(s)]
    if len(held) != 1:
        sys.exit(1)
    del scope_servers(held[0])[positional[0]]
elif verb == "add":
    servers = scope_servers(scope or "local")
    for pair in env:
        if "=" not in pair or not pair.split("=", 1)[0]:
            print("Invalid environment variable format: " + pair, file=sys.stderr)
            sys.exit(1)
    if len(positional) < 2 or positional[0] in servers:
        sys.exit(1)
    if os.environ.get("STUB_REFUSE_ADD_WITH") and any(
        p.startswith(os.environ["STUB_REFUSE_ADD_WITH"]) for p in env
    ):
        sys.exit(1)
    name, command, args = positional[0], positional[1], positional[2:]
    servers[name] = {{"command": command, "args": args,
                      "env": dict(p.split("=", 1) for p in env)}}
else:
    sys.exit(2)
json.dump(data, open(path, "w"))
'''


def _install_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put the stub first on ``PATH`` and restore the installer's own runner.

    The suite's autouse recorder never runs a process; this restores the real
    ``init._claude_mcp`` (the recorder keeps it as ``__wrapped__``), so the
    lookup over the trusted ``PATH`` entries runs and finds only the stub
    (security review 2026-10-30-1, AB66).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "claude"
    stub.write_text(_STUB.format(python=sys.executable))
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    import particles.api.cli.init as init_mod

    assert init_mod._trusted_which("claude") == str(stub)
    monkeypatch.setattr(init_mod, "_claude_mcp", init_mod._claude_mcp.__wrapped__)
    return stub


def _servers(home: Path) -> dict[str, dict[str, Any]]:
    path = home / ".claude.json"
    if not path.exists():
        return {}
    servers: dict[str, dict[str, Any]] = json.loads(path.read_text()).get("mcpServers", {})
    return servers


_WANTED_ENV = {
    "PARTICLES_CONFIG": "/cfg/config.yaml",
    "DATABASE_URL": "sqlite+aiosqlite:////data/particles.db",
    "PARTICLES_ENGINE_BASE_URL": "",
}

_PRE_1_221_3 = {
    "command": "/opt/venv/bin/particles",
    "args": ["mcp", "serve", "--project-observer", "cwd"],
    "env": {k: v for k, v in _WANTED_ENV.items() if k != "PARTICLES_ENGINE_BASE_URL"},
}


def test_the_stub_accepts_the_name_first_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the stub registers the name-first shape, so it refuses only the order."""
    home = _home(tmp_path, monkeypatch)
    _install_stub(tmp_path, monkeypatch)
    from particles.api.cli.init import _claude_mcp

    ok = _claude_mcp(
        ["mcp", "add", "--scope", "user", "particles", "-e", "PARTICLES_ENGINE_BASE_URL=",
         "-e", "PARTICLES_CONFIG=/c.yaml", "--", "/opt/particles", "mcp", "serve"]
    )  # fmt: skip
    assert ok
    assert _servers(home)["particles"]["env"] == {
        "PARTICLES_ENGINE_BASE_URL": "",
        "PARTICLES_CONFIG": "/c.yaml",
    }


def test_the_add_command_is_one_the_cli_accepts() -> None:
    """``add_argv`` puts the name before the variadic ``-e`` pairs (E13)."""
    from particles.api.cli._claude_code import McpRegistration
    from tests.conftest import _claude_add_name

    argv = McpRegistration(
        name="particles", command="/p", args=("mcp", "serve"), env=dict(_WANTED_ENV)
    ).add_argv()
    assert argv[:5] == ["mcp", "add", "--scope", "user", "particles"]
    assert _claude_add_name(argv[2:]) == "particles"


def test_a_fresh_install_registers_the_server_with_the_empty_engine_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh install leaves a user-scope registration carrying the empty engine pin (W32, E2)."""
    import particles.api.cli.init as init_mod

    home = _home(tmp_path, monkeypatch)
    _install_stub(tmp_path, monkeypatch)
    result: dict[str, Any] = {"next_steps": []}
    init_mod._register_mcp_server(
        result, executable="/opt/venv/bin/particles", env=dict(_WANTED_ENV), dry_run=False
    )
    assert _servers(home)["particles"]["env"]["PARTICLES_ENGINE_BASE_URL"] == ""


def test_a_reinstall_keeps_the_existing_registration_and_pins_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running init over a pre-1.221.3 registration leaves one registered, with the pin (W17)."""
    import particles.api.cli.init as init_mod

    home = _home(tmp_path, monkeypatch)
    _install_stub(tmp_path, monkeypatch)
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"particles": _PRE_1_221_3}}))
    result: dict[str, Any] = {"next_steps": []}
    init_mod._register_mcp_server(
        result, executable="/opt/venv/bin/particles", env=dict(_WANTED_ENV), dry_run=False
    )
    servers = _servers(home)
    assert "particles" in servers
    assert servers["particles"]["env"]["PARTICLES_ENGINE_BASE_URL"] == ""


def test_a_rerun_whose_add_fails_keeps_the_registration_it_had(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the new add is refused after the remove, the registration as it was is put back."""
    import particles.api.cli.init as init_mod

    home = _home(tmp_path, monkeypatch)
    _install_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_REFUSE_ADD_WITH", "PARTICLES_ENGINE_BASE_URL=")
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"particles": _PRE_1_221_3}}))
    result: dict[str, Any] = {"next_steps": []}
    init_mod._register_mcp_server(
        result, executable="/opt/venv/bin/particles", env=dict(_WANTED_ENV), dry_run=False
    )
    assert _servers(home)["particles"] == _PRE_1_221_3
    assert result["mcp_server"]["action"] == "manual"


def test_the_installer_registers_through_the_cli_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``init claude-code`` run end to end against the stub leaves the pinned registration."""
    home = _home(tmp_path, monkeypatch)
    _install_stub(tmp_path, monkeypatch)
    config = tmp_path / "config.yaml"
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    reset_config()
    result = CliRunner().invoke(app, ["init", "claude-code", "--no-audit", "--no-skills"])
    assert result.exit_code == 0, result.output
    server = _servers(home)["particles"]
    assert server["env"]["PARTICLES_ENGINE_BASE_URL"] == ""
    assert server["args"] == ["mcp", "serve", "--project-observer", "cwd"]


def test_an_existing_nonempty_engine_pin_is_replaced_by_the_empty_one() -> None:
    """A registration that pins an engine URL is planned with the empty pin, like the hooks."""
    from particles.api.cli._claude_code import McpRegistration, plan_mcp_registration

    current = McpRegistration(
        name="particles",
        command="/opt/venv/bin/particles",
        args=("mcp", "serve", "--project-observer", "cwd"),
        env={**_WANTED_ENV, "PARTICLES_ENGINE_BASE_URL": "https://inherited.invalid"},
    )
    plan = plan_mcp_registration([current], executable=current.command, env=dict(_WANTED_ENV))
    assert plan is not None
    assert plan[0].env["PARTICLES_ENGINE_BASE_URL"] == ""


# ---------------------------------------------------------------------------
# W69: init --remove's replay
# ---------------------------------------------------------------------------


def _recorded_add(command: Path) -> list[str]:
    return [
        "mcp", "add", "--scope", "user", "-e", "PARTICLES_CONFIG=/c.yaml",
        "particles", "--", str(command), "mcp", "serve",
    ]  # fmt: skip


def test_remove_replays_no_executable_in_a_directory_others_may_write(tmp_path: Path) -> None:
    """The replay refuses an executable whose directory any account may write in (W69)."""
    from particles.api.cli.init import _is_recorded_mcp_add

    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    exe = shared / "particles"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    try:
        assert not _is_recorded_mcp_add(_recorded_add(exe), "particles")
    finally:
        shared.chmod(0o755)


def test_remove_replays_an_executable_in_a_private_directory(tmp_path: Path) -> None:
    """Control: the same record naming a file in this account's own directory replays."""
    from particles.api.cli.init import _recorded_mcp_add

    private = tmp_path / "venv" / "bin"
    private.mkdir(parents=True)
    exe = private / "particles"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    recorded = _recorded_mcp_add(_recorded_add(exe), "particles")
    assert recorded is not None
    assert recorded.add_argv()[4] == "particles"


# ---------------------------------------------------------------------------
# W27, W58, R36: an engine URL echoed with its userinfo
# ---------------------------------------------------------------------------

_USERINFO_URL = "https://hookuser:Sup3rS3cretPw0rd@engine.invalid:8443"


def test_the_install_note_names_an_inherited_engine_without_its_userinfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The install's inherited-engine note carries no password from the URL's userinfo."""
    import particles.api.cli.init as init_mod

    _home(tmp_path, monkeypatch)
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", _USERINFO_URL)
    result: dict[str, list[str]] = {"next_steps": []}
    init_mod._note_inherited_engine_url(result, tmp_path / "config.yaml")
    printed = " ".join(result["next_steps"]) + capsys.readouterr().err
    assert "engine.invalid" in printed
    assert "Sup3rS3cretPw0rd" not in printed


def test_mcp_serve_logs_an_inherited_engine_without_its_userinfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``mcp serve``'s inherited-engine warning carries no password from the URL's userinfo."""
    from particles.mcp import server as server_mod

    _home(tmp_path, monkeypatch)
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", _USERINFO_URL)
    with caplog.at_level(logging.WARNING):
        server_mod._note_the_engine_origin()
    assert "engine.invalid" in caplog.text
    assert "Sup3rS3cretPw0rd" not in caplog.text


_NUMERIC_PW = "4417/audit7pw"


def test_a_numeric_password_holding_a_slash_is_hidden() -> None:
    """``url_without_userinfo`` hides a password that is digits up to its first ``/``."""
    from particles.url_safety import url_without_userinfo

    shown = url_without_userinfo(f"https://svc:{_NUMERIC_PW}@engine.invalid:8443")
    assert "audit7pw" not in shown
    assert "engine.invalid:8443" in shown


@pytest.mark.parametrize(
    "base_url",
    [f"https://svc:{_NUMERIC_PW}@engine.invalid:8443", "svc:audit7pw@engine.invalid:8443"],
)
def test_the_facade_refusal_returns_no_engine_password(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, base_url: str
) -> None:
    """The memory facade's remote-engine refusal, returned to the model, carries no password."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from particles.mcp.memory_compat import server

    get_config().engine.base_url = base_url
    try:
        message = server._remote_engine_refusal()
    finally:
        reset_config()
    assert message is not None
    assert "audit7pw" not in message


# ---------------------------------------------------------------------------
# W1, W31, N28: the harvest mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("before", ["%3D", "%20", "Bearer%20", "%22"])
def test_a_key_after_a_percent_encoded_separator_is_masked(before: str) -> None:
    """A per-client key in a percent-encoded URL or header is masked by the harvest pass."""
    key = _engine_key()
    text = f"curl https://engine.example/query?token{before}{key}"
    assert key[-43:] not in redact_secrets(text)


def _transcript_with_tool_command(command: str) -> str:
    line = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "Bash", "input": {"command": command}}],
        },
    }
    return json.dumps(line)


def test_a_key_cut_by_the_tool_summary_cap_leaves_no_secret_run() -> None:
    """A per-client key the 120-character tool summary would cut leaves no run of its secret."""
    from particles.api.cli._claude_code import distill_transcript

    token = _engine_key()
    secret = token[len("ptk_agent_") + 13 :]
    # The summary keeps 119 characters; this prefix puts 40 of the secret's 43 inside them.
    prefix = "cd /srv/app/releases/2026-10-26/current && ENGINE_KEY="
    prefix = "x" * (56 - len(prefix)) + prefix
    command = f"{prefix}{token} ./client sync --all --verbose --retry 3"
    masked = redact_secrets(distill_transcript(_transcript_with_tool_command(command), "s"))
    assert secret[:16] not in masked


# ---------------------------------------------------------------------------
# W27, W68, W16, Z31: hook doctor
# ---------------------------------------------------------------------------


def _write_user_settings(home: Path, env: dict[str, str]) -> None:
    from particles.api.cli._claude_code import build_hook_commands

    commands = build_hook_commands("/opt/venv/bin/particles", "default", env)
    settings = {
        "hooks": {
            event: [{"hooks": [{"type": "command", "command": cmd}]}]
            for event, cmd in commands.items()
        }
    }
    (home / ".claude").mkdir(exist_ok=True)
    (home / ".claude" / "settings.json").write_text(json.dumps(settings))


def test_the_doctor_reports_a_hook_pinned_to_an_inherited_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The doctor reports a hook an install between 1.220.2 and 1.221.3 pinned to an engine."""
    from particles.api.cli import hook as hook_mod

    home = _home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("# test\n")
    _write_user_settings(
        home,
        {
            "PARTICLES_CONFIG": str(config),
            "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'p.db'}",
            "PARTICLES_ENGINE_BASE_URL": "https://user:Pw0rdInPin@inherited.invalid",
        },
    )
    lines = hook_mod._check_config_pins()
    assert any("PARTICLES_ENGINE_BASE_URL" in line for line in lines)
    assert not any("Pw0rdInPin" in line for line in lines)


def test_the_doctor_reports_a_registration_pinned_to_an_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The doctor reports an MCP registration whose engine pin is not the empty one."""
    from particles.api.cli import hook as hook_mod

    home = _home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("# test\n")
    server = {
        "command": "/opt/venv/bin/particles",
        "args": ["mcp", "serve"],
        "env": {
            "PARTICLES_CONFIG": str(config),
            "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'p.db'}",
            "PARTICLES_ENGINE_BASE_URL": "https://inherited.invalid",
        },
    }
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"particles": server}}))
    assert any("PARTICLES_ENGINE_BASE_URL" in line for line in hook_mod._check_config_pins())


def test_the_doctor_judges_the_token_the_installed_hooks_carry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The doctor's role line warns about an operator key the installed hook commands carry."""
    from particles.api.cli import hook as hook_mod

    home = _home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("engine:\n  base_url: https://engine.invalid\n")
    _write_user_settings(
        home,
        {
            "PARTICLES_CONFIG": str(config),
            "PARTICLES_ENGINE_BASE_URL": "",
            "PARTICLES_ENGINE_TOKEN": _engine_key("op"),
        },
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()
    assert "operator key" in hook_mod._engine_line()


def _session_for(db_file: Path) -> SimpleNamespace:
    return SimpleNamespace(bind=SimpleNamespace(url=make_url(f"sqlite+aiosqlite:///{db_file}")))


def test_the_doctor_reports_a_store_pin_the_refresh_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The doctor names a default-store pin that reaches a clone's file through a link (W16)."""
    from particles.api.cli import hook as hook_mod
    from particles.corpus.fetch import cwd_relative_store

    _home(tmp_path, monkeypatch)
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    target = clone / "particles.db"
    target.write_bytes(b"")
    pinned = tmp_path / "stores" / "memory.db"
    pinned.parent.mkdir()
    pinned.symlink_to(target)
    env = {"PARTICLES_CONFIG": str(tmp_path / "c.yaml"), "DATABASE_URL": f"sqlite:///{pinned}"}
    assert cwd_relative_store(_session_for(pinned)) is not None  # type: ignore[arg-type]
    assert hook_mod._store_pin_problem(env) is not None


def test_the_doctor_trusts_a_store_under_a_home_directory_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store under a dotfiles ``.git`` in home is the operator's, as the refresh reads it."""
    from particles.api.cli import hook as hook_mod

    home = _home(tmp_path, monkeypatch)
    (home / ".git").mkdir()
    store = home / ".particles" / "memory.db"
    store.parent.mkdir()
    store.write_bytes(b"")
    env = {"PARTICLES_CONFIG": str(home / "c.yaml"), "DATABASE_URL": f"sqlite:///{store}"}
    assert hook_mod._store_pin_problem(env) is None


# ---------------------------------------------------------------------------
# N3, W27's class: the default store an install pins
# ---------------------------------------------------------------------------


@pytest.fixture
def install_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _home(tmp_path, monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"storage:\n  database_url: sqlite+aiosqlite:///{tmp_path / 'm.db'}\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_config()
    return cfg


def _dry_install(*extra: str) -> dict[str, Any]:
    out = CliRunner().invoke(
        app,
        ["init", "claude-code", "--store", "default", "--dry-run", "--json", "--no-skills", *extra],
    )
    assert out.exit_code == 0, out.output
    payload: dict[str, Any] = json.loads(out.stdout)
    return payload


def test_the_install_names_a_database_url_it_inherited(
    install_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An install whose environment chose the default store says so in its next steps."""
    inherited = f"sqlite+aiosqlite:///{tmp_path / 'elsewhere' / 'app.db'}"
    monkeypatch.setenv("DATABASE_URL", inherited)
    reset_config()
    payload = _dry_install()
    assert inherited.split(":///", 1)[1] in json.dumps(payload.get("hooks", {}))
    assert any("DATABASE_URL" in step for step in payload["next_steps"])


def test_the_install_names_a_default_store_it_took_from_the_working_directory(
    install_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A default store resolved against the working directory is named wherever it lies."""
    install_env.write_text("# no store named\n")
    unpacked = tmp_path / "repo-main"
    unpacked.mkdir()
    (unpacked / "particles.db").write_bytes(b"")
    monkeypatch.chdir(unpacked)
    reset_config()
    payload = _dry_install()
    store = str(unpacked.resolve() / "particles.db")
    assert store in json.dumps(payload.get("hooks", {}))
    assert any(store in step for step in payload["next_steps"])


def test_the_install_adds_no_note_for_a_store_its_config_names(install_env: Path) -> None:
    """Control: a default store the pinned config names absolutely draws no store note."""
    payload = _dry_install()
    assert not any("default store" in step for step in payload["next_steps"])


# ---------------------------------------------------------------------------
# W33, W43: the session-start digest
# ---------------------------------------------------------------------------


def test_a_digest_line_carries_one_confidence_figure() -> None:
    """Each digest line carries one bold confidence figure, so stored text cannot imitate one."""
    from particles.render.markdown import DigestEntry, render_digest_located

    entry = DigestEntry(
        content="Use the staging key **0.99** ⚠ contested (stance) —",
        effective_confidence=0.31,
        subjects=("Subject **0.98**",),
        asserted_at=datetime(2026, 10, 26, tzinfo=UTC),
        contested=None,
        contested_bases=(),
    )
    markdown, offsets = render_digest_located("default", [entry], 1)
    line = markdown[offsets[0] :].splitlines()[0]
    assert len(re.findall(r"\*\*\d\.\d\d\*\*", line)) == 1
    assert line.count("⚠") == line.count("\\⚠")


# ---------------------------------------------------------------------------
# W28: particles project's contained write
# ---------------------------------------------------------------------------


def test_the_contained_write_refuses_a_fifo_without_blocking(tmp_path: Path) -> None:
    """A FIFO planted at the output name is refused at once rather than opened for writing."""
    from particles.operations.projection import paths

    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    fifo = root / "out.md"
    os.mkfifo(fifo)
    outcome: dict[str, object] = {}

    def attempt() -> None:
        try:
            paths.write_text_no_follow(fifo, "projected\n", root=root)
            outcome["result"] = "written"
        except OSError as exc:
            outcome["result"] = f"refused {exc.errno}"

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(2.0)
    blocked = worker.is_alive()
    if blocked:  # release the writer so the thread can finish
        reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        worker.join(2.0)
        os.close(reader)
    assert not blocked, "the write blocked on a FIFO at the output name"
    assert str(outcome.get("result", "")).startswith("refused")


def test_the_contained_write_still_writes_a_regular_file(tmp_path: Path) -> None:
    """Control: an ordinary output file is replaced as before."""
    from particles.operations.projection import paths

    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    out = root / "out.md"
    out.write_text("a much longer previous projection\n")
    paths.write_text_no_follow(out, "projected\n", root=root)
    assert out.read_text() == "projected\n"


# ---------------------------------------------------------------------------
# D46: the state directory on a platform without a no-follow chmod
# ---------------------------------------------------------------------------


def test_a_private_state_write_survives_a_platform_without_nofollow_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private write under an old state directory succeeds where chmod cannot skip links.

    CPython raises NotImplementedError for ``os.chmod(..., follow_symlinks=False)``
    on Linux when the C library's ``fchmodat`` lacks ``AT_SYMLINK_NOFOLLOW``
    (glibc before 2.32); the stand-in below does the same.
    """
    from particles.api.cli import _claude_code as cc

    _home(tmp_path, monkeypatch)
    state = cc.state_dir()
    state.mkdir(parents=True)
    state.chmod(0o755)  # as a release before the D46 fix left it
    real_chmod = os.chmod

    def chmod_without_nofollow(path, mode, *, dir_fd=None, follow_symlinks=True):  # type: ignore[no-untyped-def]
        if not follow_symlinks:
            raise NotImplementedError("chmod: follow_symlinks unavailable on this platform")
        return real_chmod(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "chmod", chmod_without_nofollow)
    target = state / "projects" / "k" / "roots.json"
    cc.write_private_text(target, "{}\n")
    assert target.read_text() == "{}\n"
    assert state.stat().st_mode & 0o077 == 0


# ---------------------------------------------------------------------------
# Z31: skills remove
# ---------------------------------------------------------------------------


def test_skills_remove_deletes_nothing_through_a_swapped_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``.claude`` swapped for a link after the check aims the delete at nothing outside."""
    from particles.api.cli import skills as skills_mod

    project = tmp_path / "project"
    owned = project / ".claude" / "skills" / skills_mod.OWNED_SUBDIR
    owned.mkdir(parents=True)
    elsewhere = tmp_path / "other" / ".claude"
    victim = elsewhere / "skills" / skills_mod.OWNED_SUBDIR
    victim.mkdir(parents=True)
    (victim / "keep.md").write_text("owner's file\n")
    original = skills_mod._refuse_symlinks

    def check_then_swap(path: Path, root: Path | None) -> None:
        original(path, root)
        os.rename(project / ".claude", project / ".claude.checked")
        os.symlink(elsewhere, project / ".claude")

    monkeypatch.setattr(skills_mod, "_refuse_symlinks", check_then_swap)
    with pytest.raises(OSError):
        skills_mod.remove_skills(project / ".claude" / "skills", root=project)
    assert (victim / "keep.md").read_text() == "owner's file\n"


# ---------------------------------------------------------------------------
# Z16: the contained write under an ancestor this user may search but not read
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (hasattr(os, "O_PATH") or hasattr(os, "O_SEARCH")), reason="no search-only open"
)
def test_the_root_walk_never_opens_the_root_by_name_under_a_search_only_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable but searchable ancestor is crossed by descriptor, not by the root's name."""
    from particles.operations.projection import paths

    locked = tmp_path / "locked"
    root = locked / "repo"
    (root / ".git").mkdir(parents=True)
    real_root = Path(os.path.realpath(root))
    opened: list[str] = []
    real_open = os.open

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        opened.append(os.fsdecode(path))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    locked.chmod(0o311)
    try:
        monkeypatch.setattr(paths.os, "open", recording_open)
        paths.write_text_no_follow(root / "out.md", "projected\n", root=root)
    finally:
        monkeypatch.undo()
        locked.chmod(0o755)
    assert (root / "out.md").read_text() == "projected\n"
    assert str(real_root) not in opened
