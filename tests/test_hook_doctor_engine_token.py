# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""``hook doctor``'s ``engine token:`` lines.

One line per consumer of the engine key (the installed hooks, the MCP
registration), read from their pins: the source, the token-file checks, the
role the prefix carries and, when the engine answers, the principal and role
``GET /auth/whoami`` returns. A plaintext key is a warning with its fix, a
wrapper install is named, and no token, hash or length is ever printed. The
engine is a mocked transport behind the engine client's ``egress_client``
seam; no socket is opened.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import particles
from particles.api.cli import app
from particles.api.cli import hook as hook_mod
from particles.api.cli._claude_code import build_hook_commands
from particles.config import reset_config
from particles.secrets import _forget_engine_token_files
from tests._claude_projects import isolate_claude_code

_BASE = "https://engine.test"
_EXE = "/opt/venv/bin/particles"

Handler = Callable[[httpx.Request], httpx.Response]


def _key(role: str = "agent") -> str:
    secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    return f"ptk_{role}_{secrets.token_hex(6)}_{secret}"


def _engine(principals: dict[str, tuple[str, str]], *, refused: int | None = None) -> Handler:
    """A fake engine answering whoami by bearer, plus the health and digest probes."""

    def handle(request: httpx.Request) -> httpx.Response:
        bearer = request.headers.get("Authorization", "").removeprefix("Bearer ")
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "version": particles.__version__})
        if path == "/auth/whoami":
            if refused is not None or bearer not in principals:
                return httpx.Response(refused or 401, json={"detail": "Invalid or missing token."})
            principal, role = principals[bearer]
            return httpx.Response(200, json={"principal": principal, "role": role})
        if path == "/digest/default" and bearer in principals:
            return httpx.Response(200, json={"markdown": "# Memory digest\n"})
        return httpx.Response(401, json={"detail": "Invalid or missing token."})

    return handle


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A thin client whose pinned config names the engine; returns the fake home."""
    home = isolate_claude_code(
        monkeypatch, tmp_path, extra_config=f"engine:\n  base_url: {_BASE}\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'absent.db'}")
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN_FILE", raising=False)
    _forget_engine_token_files()
    reset_config()
    yield home
    _forget_engine_token_files()
    reset_config()


def _serve(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    real = httpx.AsyncClient

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def _client(
        *, base_url: str, timeout: float, headers: dict[str, str], purpose: str = ""
    ) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(record), base_url=base_url, headers=headers)

    monkeypatch.setattr("particles.api.client.http.egress_client", _client)
    return seen


def _token_file(home: Path, name: str, value: str, mode: int = 0o600) -> Path:
    directory = home / ".particles" / "claude-code" / "engine-tokens"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    path = directory / name
    path.write_text(value + "\n")
    path.chmod(mode)
    return path


def _install_hooks(home: Path, env: dict[str, str], *, extra: dict[str, Any] | None = None) -> Path:
    commands = build_hook_commands(_EXE, "default", {"PARTICLES_ENGINE_BASE_URL": "", **env})
    settings: dict[str, Any] = {
        "hooks": {
            event: [{"hooks": [{"type": "command", "command": cmd}]}]
            for event, cmd in commands.items()
        },
        **(extra or {}),
    }
    path = home / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings))
    return path


def _register_mcp(home: Path, env: dict[str, str]) -> None:
    server = {"command": _EXE, "args": ["mcp", "serve"], "env": env}
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"particles": server}}))


def _lines() -> tuple[bool, list[str]]:
    return hook_mod._engine_token_lines(_BASE)


def _assert_no_secret(text: str, *tokens: str) -> None:
    for token in tokens:
        assert token not in text
        assert hashlib.sha256(token.encode()).hexdigest() not in text
        assert token.split("_", 3)[-1] not in text
        assert f"{len(token)} char" not in text and f"{len(token)} byte" not in text


def test_token_files_with_the_right_keys_pass(
    client: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks_key, mcp_key = _key(), _key()
    hooks_file = _token_file(client, "hooks", hooks_key)
    mcp_file = _token_file(client, "mcp", mcp_key)
    _install_hooks(
        client, {"PARTICLES_ENGINE_TOKEN": "", "PARTICLES_ENGINE_TOKEN_FILE": str(hooks_file)}
    )
    _register_mcp(
        client, {"PARTICLES_ENGINE_TOKEN": "", "PARTICLES_ENGINE_TOKEN_FILE": str(mcp_file)}
    )
    _serve(
        monkeypatch,
        _engine({hooks_key: ("claude-code-hook", "agent"), mcp_key: ("mcp:claude-code", "agent")}),
    )
    ok, lines = _lines()
    assert ok, lines
    text = "\n".join(lines)
    assert (
        f"engine token:   hooks: file {hooks_file}, which passes the token-file checks; an "
        "agent key; the engine says principal claude-code-hook, role agent"
    ) in text
    assert (
        f"engine token:   MCP server particles: file {mcp_file}, which passes the token-file "
        "checks; an agent key; the engine says principal mcp:claude-code, role agent"
    ) in text
    assert "⚠" not in text and "✗" not in text
    _assert_no_secret(text, hooks_key, mcp_key)


def test_the_doctor_passes_a_thin_install_and_prints_no_key(
    client: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks_key = _key()
    hooks_file = _token_file(client, "hooks", hooks_key)
    _install_hooks(
        client, {"PARTICLES_ENGINE_TOKEN": "", "PARTICLES_ENGINE_TOKEN_FILE": str(hooks_file)}
    )
    # This shell holds no key: the probes send the one the hooks run with.
    _serve(monkeypatch, _engine({hooks_key: ("claude-code-hook", "agent")}))
    result = CliRunner().invoke(app, ["hook", "doctor", "--store", "default"])
    assert result.exit_code == 0, result.output
    assert "served" in result.stdout and "to the installed hooks' key" in result.stdout
    assert "the engine token: lines judge the key each consumer pins" in result.stdout
    assert "engine token:   MCP server: none pinned" in result.stdout
    _assert_no_secret(result.output, hooks_key)


def test_a_token_file_a_check_refuses_fails_the_doctor(
    client: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks_key = _key()
    hooks_file = _token_file(client, "hooks", hooks_key, mode=0o644)
    _install_hooks(
        client, {"PARTICLES_ENGINE_TOKEN": "", "PARTICLES_ENGINE_TOKEN_FILE": str(hooks_file)}
    )
    seen = _serve(monkeypatch, _engine({hooks_key: ("claude-code-hook", "agent")}))
    ok, lines = _lines()
    assert not ok
    hooks = next(line for line in lines if line.startswith("engine token:   hooks:"))
    assert f"✗ file {hooks_file} fails the mode check" in hooks
    assert not seen, "a refused file sends nothing"
    _assert_no_secret("\n".join(lines), hooks_key)

    monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", hooks_key)
    reset_config()
    result = CliRunner().invoke(app, ["hook", "doctor", "--store", "default"])
    assert result.exit_code == 1, result.output
    assert "✗ The hooks cannot use store 'default'" in result.stdout


def test_a_plaintext_hook_pin_is_a_warning_with_its_fix(
    client: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _key()
    settings = _install_hooks(client, {"PARTICLES_ENGINE_TOKEN": key})
    _serve(monkeypatch, _engine({key: ("claude-code-hook", "agent")}))
    ok, lines = _lines()
    assert ok, "plaintext still works, so it is no failure"
    text = "\n".join(lines)
    shown = "~/.claude/settings.json"
    assert settings == client / ".claude" / "settings.json"
    assert f"engine token:   hooks: plaintext in {shown}; an agent key" in text
    assert (
        f"⚠ the engine key is stored in plaintext in {shown}; re-run particles init "
        "claude-code to move it to a token file"
    ) in text
    _assert_no_secret(text, key)


def test_a_plaintext_key_in_the_settings_env_block_is_a_warning(client: Path) -> None:
    key = _key()
    _install_hooks(client, {}, extra={"env": {"PARTICLES_ENGINE_TOKEN": key}})
    _, lines = _lines()
    text = "\n".join(lines)
    assert "engine token:   hooks: plaintext in ~/.claude/settings.json" in text
    assert "⚠ the engine key is stored in plaintext in ~/.claude/settings.json" in text
    _assert_no_secret(text, key)


def test_a_plaintext_mcp_registration_is_a_warning(
    client: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _key()
    _register_mcp(client, {"PARTICLES_ENGINE_TOKEN": key})
    _serve(monkeypatch, _engine({key: ("mcp:claude-code", "agent")}))
    _, lines = _lines()
    text = "\n".join(lines)
    assert "engine token:   MCP server particles: plaintext in ~/.claude.json; an agent key" in text
    assert "⚠ the engine key is stored in plaintext in ~/.claude.json" in text
    _assert_no_secret(text, key)


def test_the_role_the_prefix_carries_is_judged_for_each_consumer(client: Path) -> None:
    hooks_key, mcp_key = _key("read"), _key("op")
    _install_hooks(client, {"PARTICLES_ENGINE_TOKEN": hooks_key})
    _register_mcp(client, {"PARTICLES_ENGINE_TOKEN": mcp_key})
    _, lines = _lines()
    hooks = next(line for line in lines if line.startswith("engine token:   hooks:"))
    mcp = next(line for line in lines if line.startswith("engine token:   MCP server"))
    assert "⚠" in hooks and "reader key" in hooks
    assert "⚠" in mcp and "operator key" in mcp


def test_a_reader_key_for_the_mcp_server_notes_the_write_tools(
    client: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _key("read")
    mcp_file = _token_file(client, "mcp", key)
    _register_mcp(client, {"PARTICLES_ENGINE_TOKEN_FILE": str(mcp_file)})
    _serve(monkeypatch, _engine({key: ("mcp:claude-code", "reader")}))
    ok, lines = _lines()
    assert ok
    assert any("a reader key (the MCP write tools answer 403)" in line for line in lines)


def test_a_wrong_principal_is_a_warning(client: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key()
    hooks_file = _token_file(client, "hooks", key)
    _install_hooks(client, {"PARTICLES_ENGINE_TOKEN_FILE": str(hooks_file)})
    _serve(monkeypatch, _engine({key: ("mcp:claude-code", "agent")}))
    ok, lines = _lines()
    assert ok
    hooks = next(line for line in lines if line.startswith("engine token:   hooks:"))
    assert (
        "⚠" in hooks
        and "the engine says principal mcp:claude-code, role agent" in hooks
        and "needs principal claude-code-hook" in hooks
    )


def test_a_key_the_engine_refuses_fails(client: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key()
    hooks_file = _token_file(client, "hooks", key)
    _install_hooks(client, {"PARTICLES_ENGINE_TOKEN_FILE": str(hooks_file)})
    _serve(monkeypatch, _engine({}))
    ok, lines = _lines()
    assert not ok
    hooks = next(line for line in lines if line.startswith("engine token:   hooks:"))
    assert "✗" in hooks and "the engine does not accept the key" in hooks
    _assert_no_secret("\n".join(lines), key)


def test_an_engine_this_shell_names_is_never_sent_the_pinned_keys(
    client: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _key()
    hooks_file = _token_file(client, "hooks", key)
    _install_hooks(client, {"PARTICLES_ENGINE_TOKEN_FILE": str(hooks_file)})
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://elsewhere.test")
    reset_config()
    seen = _serve(monkeypatch, _engine({key: ("claude-code-hook", "agent")}))
    ok, lines = hook_mod._engine_token_lines("https://elsewhere.test")
    assert ok
    assert not seen
    assert any("its principal is not checked, since PARTICLES_ENGINE_BASE_URL" in x for x in lines)


def test_nothing_pinned_says_so_for_each_consumer(client: Path) -> None:
    ok, lines = _lines()
    assert ok
    assert lines == [
        "engine token:   hooks: none pinned",
        "engine token:   MCP server: none pinned",
    ]


def test_a_token_file_a_project_settings_file_names_is_not_read(
    client: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".claude").mkdir()
    secret = tmp_path / "elsewhere"
    secret.write_text(_key() + "\n")
    commands = build_hook_commands(_EXE, "default", {"PARTICLES_ENGINE_TOKEN_FILE": str(secret)})
    (repo / ".claude" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    ev: [{"hooks": [{"type": "command", "command": c}]}]
                    for ev, c in commands.items()
                }
            }
        )
    )
    monkeypatch.chdir(repo)
    _, lines = _lines()
    hooks = next(line for line in lines if line.startswith("engine token:   hooks:"))
    assert "named by a project settings file, so the doctor does not read it" in hooks


def test_a_wrapper_install_is_named_without_its_command(client: Path) -> None:
    key = _key()
    wrapper = (
        f"env PARTICLES_ENGINE_TOKEN={key} /usr/local/bin/particles-client hook "
        "session-start --store default"
    )
    settings = {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": wrapper}]}]}}
    (client / ".claude" / "settings.json").write_text(json.dumps(settings))
    _, lines = _lines()
    text = "\n".join(lines)
    assert (
        "engine token:   ⚠ ~/.claude/settings.json runs the lifecycle hooks through "
        "/usr/local/bin/particles-client, an unrecognised install; the wrapper install is "
        f"retired: delete those entries, then run particles init claude-code --engine {_BASE}"
    ) in text
    _assert_no_secret(text, key)


def test_a_pin_with_both_a_key_and_a_token_file_fails(client: Path) -> None:
    key = _key()
    hooks_file = _token_file(client, "hooks", key)
    _register_mcp(
        client, {"PARTICLES_ENGINE_TOKEN": key, "PARTICLES_ENGINE_TOKEN_FILE": str(hooks_file)}
    )
    ok, lines = _lines()
    assert not ok
    mcp = next(line for line in lines if line.startswith("engine token:   MCP server"))
    assert "✗ PARTICLES_ENGINE_TOKEN and PARTICLES_ENGINE_TOKEN_FILE are both pinned" in mcp
    _assert_no_secret("\n".join(lines), key)


def test_an_egress_refusal_is_reported_not_raised(
    client: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.egress import EgressRefused

    key = _key()
    hooks_file = _token_file(client, "hooks", key)
    _install_hooks(client, {"PARTICLES_ENGINE_TOKEN_FILE": str(hooks_file)})

    def _refuse(**_: Any) -> httpx.AsyncClient:
        raise EgressRefused("engine.test", 443, "engine", "none")

    monkeypatch.setattr("particles.api.client.http.egress_client", _refuse)
    ok, lines = _lines()
    assert ok
    hooks = next(line for line in lines if line.startswith("engine token:   hooks:"))
    assert "its principal is not checked (EgressRefused" in hooks
