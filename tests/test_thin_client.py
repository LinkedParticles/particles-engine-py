# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thin-client ``particles init claude-code``.

``particles/api/cli/_thin_client.py`` and the thin branch of ``init.py``: the
engine URL rule, the token files and their role check, what the install writes
and does not write, the dry run, the wrapper refusal, ``--remove``, and the
hook and MCP refusals a broken token file meets.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from particles.api.cli import _thin_client as thin
from particles.api.cli import app
from particles.api.cli._claude_code import (
    McpRegistration,
    plan_mcp_registration,
    set_engine_base_url_text,
)

AGENT_KEY = "ptk_agent_0123456789ab_" + "A" * 43
MCP_KEY = "ptk_agent_ba9876543210_" + "B" * 43
READER_KEY = "ptk_read_00000000000a_" + "C" * 43
OPERATOR_KEY = "ptk_op_00000000000b_" + "D" * 43
LEGACY_KEY = "a-shared-legacy-key-value"
ENGINE = "http://127.0.0.1:8000"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A sandboxed HOME and config, a recorded ``claude mcp``, and no engine calls."""
    from particles.api.cli import init as init_mod
    from particles.config import reset_config

    home = tmp_path / "home"
    home.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("# test config\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    reset_config()
    calls: list[list[str]] = []

    def _claude(argv: list[str]) -> bool:
        calls.append(argv)
        return True

    monkeypatch.setattr(init_mod, "_claude_mcp", _claude)
    monkeypatch.setattr(init_mod, "particles_mcp_registrations", lambda home=None: [])
    probes: list[dict[str, str]] = []

    def _probe(tokens: dict[str, str], **kwargs: Any) -> thin.ProbeResult:
        probes.append(dict(tokens))
        assert kwargs["base_url"] == ENGINE
        return thin.ProbeResult(verified=True)

    monkeypatch.setattr(thin, "probe_engine", _probe)
    exclusions: list[Path] = []
    monkeypatch.setattr(thin, "exclude_from_backups", lambda d: exclusions.append(d) or None)
    keys = tmp_path / "keys"
    keys.mkdir(mode=0o700)
    keys.chmod(0o700)
    return {
        "home": home,
        "config": config,
        "settings": home / ".claude" / "settings.json",
        "claude": calls,
        "probes": probes,
        "exclusions": exclusions,
        "keys": keys,
    }


def _key_file(env: dict[str, Any], name: str, value: str, mode: int = 0o600) -> Path:
    path: Path = env["keys"] / name
    path.write_text(value + "\n")
    path.chmod(mode)
    return path


def _install(runner: CliRunner, *args: str) -> Any:
    return runner.invoke(
        app, ["init", "claude-code", "--no-audit", "--no-skills", *args], catch_exceptions=False
    )


def _hook_env(settings_path: Path) -> dict[str, str]:
    settings = json.loads(settings_path.read_text())
    (command,) = [h["command"] for g in settings["hooks"]["SessionStart"] for h in g["hooks"]]
    words = shlex.split(command)
    pins: dict[str, str] = {}
    for word in words[1:]:
        if "=" not in word:
            break
        key, value = word.split("=", 1)
        pins[key] = value
    return pins


class TestEngineUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "ftp://engine.test",
            "http://user:pw@127.0.0.1:8000",
            "https://engine.test/?token=x",
            "https://engine.test/#x",
            "not a url",
        ],
    )
    def test_refused(self, url: str) -> None:
        with pytest.raises(thin.ThinClientError):
            thin.validate_engine_url(url, allow_plaintext=False)

    def test_plain_http_beyond_loopback_needs_the_plaintext_knob(self) -> None:
        with pytest.raises(thin.ThinClientError, match="allow_plaintext_token"):
            thin.validate_engine_url("http://mini.lan:8000", allow_plaintext=False)
        assert thin.validate_engine_url("http://mini.lan:8000/", allow_plaintext=True) == (
            "http://mini.lan:8000"
        )

    def test_loopback_and_https_pass(self) -> None:
        assert thin.validate_engine_url(ENGINE, allow_plaintext=False) == ENGINE
        assert thin.validate_engine_url("https://e.test", allow_plaintext=False) == (
            "https://e.test"
        )


class TestKeyRefusal:
    def test_operator_and_legacy_keys_are_refused_for_both(self) -> None:
        for client in thin.TOKEN_CLIENTS:
            assert thin.key_refusal(client, OPERATOR_KEY) is not None
            assert thin.key_refusal(client, LEGACY_KEY) is not None

    def test_reader_key_only_for_the_mcp_server(self) -> None:
        assert thin.key_refusal(thin.HOOKS, READER_KEY) is not None
        assert thin.key_refusal(thin.MCP, READER_KEY) is None

    def test_agent_key_passes(self) -> None:
        assert thin.key_refusal(thin.HOOKS, AGENT_KEY) is None

    def test_revoke_command_reads_the_key_id(self) -> None:
        assert thin.revoke_command(AGENT_KEY) == "particles engine keys revoke 0123456789ab"
        assert thin.revoke_command(LEGACY_KEY) is None


class TestConfigSurgeon:
    def test_sets_and_replaces_the_engine(self) -> None:
        once = set_engine_base_url_text("# hdr\n", ENGINE)
        assert yaml.safe_load(once)["engine"]["base_url"] == ENGINE
        twice = set_engine_base_url_text(once, "https://other.test")
        assert yaml.safe_load(twice)["engine"]["base_url"] == "https://other.test"
        kept = set_engine_base_url_text("engine:\n  timeout_seconds: 5\n", ENGINE)
        assert yaml.safe_load(kept)["engine"] == {"timeout_seconds": 5, "base_url": ENGINE}


class TestMcpPlan:
    def test_thin_drops_the_database_url_and_pins_the_token_file(self) -> None:
        existing = McpRegistration(
            name="particles",
            command="/opt/particles",
            args=("mcp", "serve", "--project-observer", "cwd"),
            env={
                "PARTICLES_CONFIG": "/c.yaml",
                "DATABASE_URL": "sqlite+aiosqlite:////x.db",
                "PARTICLES_ENGINE_TOKEN": "plaintext-key-in-the-registration",
            },
        )
        wanted = {
            "PARTICLES_CONFIG": "/c.yaml",
            "PARTICLES_ENGINE_BASE_URL": "",
            "PARTICLES_ENGINE_TOKEN": "",
            "PARTICLES_ENGINE_TOKEN_FILE": "/k/mcp",
        }
        plan = plan_mcp_registration([existing], executable="/opt/particles", env=wanted, thin=True)
        assert plan is not None
        new_env = plan[0].env
        assert "DATABASE_URL" not in new_env
        assert new_env["PARTICLES_ENGINE_TOKEN"] == ""
        assert new_env["PARTICLES_ENGINE_TOKEN_FILE"] == "/k/mcp"


class TestWrapperDetection:
    def test_a_wrapper_hook_is_found_and_an_owned_one_is_not(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "command": "env A=1 /Users/x/bin/particles-client hook "
                                "session-start --store default"
                            },
                            {"command": "/v/bin/particles hook session-start --store default"},
                            {"command": "echo hook session-start"},
                        ]
                    }
                ]
            }
        }
        found = thin.wrapper_hook_entries(settings)
        assert found == ["env A=1 /Users/x/bin/particles-client hook session-start --store default"]


class TestThinInstall:
    def test_writes_no_store_and_no_secret(self, runner: CliRunner, env: dict[str, Any]) -> None:
        hooks = _key_file(env, "hooks", AGENT_KEY)
        mcp = _key_file(env, "mcp", MCP_KEY)
        result = _install(
            runner,
            "--engine",
            ENGINE,
            "--hooks-token-file",
            str(hooks),
            "--mcp-token-file",
            str(mcp),
            "--json",
        )
        assert result.exit_code == 0, result.output
        out = json.loads(result.stdout)
        assert out["mode"] == "thin-client"
        assert out["store"] == "default"
        assert out["blocked"] == []
        assert {e["step"] for e in out["not_done"]} >= {
            "project-key rescope",
            "first-run memory audit",
        }
        pins = _hook_env(env["settings"])
        assert "DATABASE_URL" not in pins
        assert pins["PARTICLES_ENGINE_TOKEN"] == ""
        assert pins["PARTICLES_ENGINE_TOKEN_FILE"] == str(hooks)
        assert pins["PARTICLES_ENGINE_BASE_URL"] == ""
        config = yaml.safe_load(env["config"].read_text())
        assert config["engine"]["base_url"] == ENGINE
        assert config["mcp"]["write"]["enabled_stores"] == ["default"]
        assert "storage" not in config
        assert not (env["home"] / ".particles" / "memory.db").exists()
        (add,) = [argv for argv in env["claude"] if argv[:2] == ["mcp", "add"]]
        assert f"PARTICLES_ENGINE_TOKEN_FILE={mcp}" in add
        assert not any(word.startswith("DATABASE_URL=") for word in add)
        for text in (env["settings"].read_text(), result.output, json.dumps(env["claude"])):
            assert AGENT_KEY not in text and MCP_KEY not in text
        assert env["probes"] == [{"hooks": AGENT_KEY, "mcp": MCP_KEY}]
        assert env["exclusions"] == [thin.token_dir()]

    def test_dry_run_writes_nothing_and_asks_nothing(
        self, runner: CliRunner, env: dict[str, Any]
    ) -> None:
        hooks = _key_file(env, "hooks", AGENT_KEY)
        before = env["config"].read_text()
        result = _install(runner, "--engine", ENGINE, "--hooks-token-file", str(hooks), "--dry-run")
        assert result.exit_code == 0, result.output
        assert "thin client" in result.output and "No local store" in result.output
        assert "not done on a thin client" in result.output
        assert AGENT_KEY not in result.output
        assert env["config"].read_text() == before
        assert not env["settings"].exists()
        assert env["probes"] == [] and env["claude"] == []

    def test_non_interactive_missing_file_is_blocked(
        self, runner: CliRunner, env: dict[str, Any]
    ) -> None:
        result = _install(runner, "--engine", ENGINE, "--json")
        assert result.exit_code == 0, result.output
        out = json.loads(result.stdout)
        assert len(out["blocked"]) == 2
        assert out["token_files"]["hooks"]["state"] == "missing (blocked: non-interactive)"
        assert any("keys create" in step for step in out["next_steps"])

    @pytest.mark.parametrize(
        ("key", "client"),
        [
            (OPERATOR_KEY, "hooks"),
            (LEGACY_KEY, "hooks"),
            (READER_KEY, "hooks"),
            (OPERATOR_KEY, "mcp"),
        ],
    )
    def test_wrong_role_is_refused_before_any_write(
        self, runner: CliRunner, env: dict[str, Any], key: str, client: str
    ) -> None:
        path = _key_file(env, client, key)
        before = env["config"].read_text()
        result = _install(runner, "--engine", ENGINE, f"--{client}-token-file", str(path))
        assert result.exit_code == 1
        assert key not in result.output
        assert env["config"].read_text() == before
        assert not env["settings"].exists()

    def test_a_loose_token_file_is_refused(self, runner: CliRunner, env: dict[str, Any]) -> None:
        path = _key_file(env, "hooks", AGENT_KEY, mode=0o644)
        result = _install(runner, "--engine", ENGINE, "--hooks-token-file", str(path))
        assert result.exit_code == 1
        assert "(mode)" in result.output

    def test_a_wrapper_install_stops_the_thin_install(
        self, runner: CliRunner, env: dict[str, Any]
    ) -> None:
        env["settings"].parent.mkdir(parents=True)
        env["settings"].write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {
                                "hooks": [
                                    {
                                        "command": "/u/bin/particles-client hook "
                                        "session-start --store default"
                                    }
                                ]
                            }
                        ]
                    }
                }
            )
        )
        before = env["settings"].read_text()
        result = _install(runner, "--engine", ENGINE)
        assert result.exit_code == 1
        assert "wrapper" in result.output
        assert env["settings"].read_text() == before

    def test_a_probe_refusal_pins_nothing(
        self, runner: CliRunner, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            thin,
            "probe_engine",
            lambda tokens, **_k: thin.ProbeResult(
                verified=True, refusals={"hooks": "the key's principal is mcp:claude-code"}
            ),
        )
        hooks = _key_file(env, "hooks", AGENT_KEY)
        result = _install(runner, "--engine", ENGINE, "--hooks-token-file", str(hooks))
        assert result.exit_code == 1
        assert "principal" in result.output
        assert not env["settings"].exists()

    def test_prompted_keys_are_enrolled_and_removed(
        self, runner: CliRunner, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import getpass
        import stat

        from particles.api.cli import init as init_mod
        from particles.config import get_config

        answers = iter([AGENT_KEY, MCP_KEY])
        monkeypatch.setattr(getpass, "getpass", lambda _prompt: next(answers))
        out = init_mod._install(
            env["settings"],
            store_flag=None,
            dry_run=False,
            command_override="/opt/venv/bin/particles",
            no_audit=True,
            install_skills_files=False,
            engine_flag=ENGINE,
            token_flags={},
            interactive=True,
        )
        assert get_config().engine.base_url == ENGINE
        for client, key in (("hooks", AGENT_KEY), ("mcp", MCP_KEY)):
            path = thin.default_token_path(client)
            assert path.read_text() == key + "\n"
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert out["token_files"][client]["state"] == "present"
        assert key not in json.dumps(out)
        removed = _install(runner, "--remove", "--json")
        assert removed.exit_code == 0, removed.output
        result = json.loads(removed.stdout)
        assert not thin.default_token_path("hooks").exists()
        assert not thin.default_token_path("mcp").exists()
        assert any(
            "particles engine keys revoke 0123456789ab" in step for step in result["next_steps"]
        )

    def test_remove_keeps_a_file_it_did_not_write(
        self, runner: CliRunner, env: dict[str, Any]
    ) -> None:
        hooks = _key_file(env, "hooks", AGENT_KEY)
        mcp = _key_file(env, "mcp", MCP_KEY)
        assert (
            _install(
                runner,
                "--engine",
                ENGINE,
                "--hooks-token-file",
                str(hooks),
                "--mcp-token-file",
                str(mcp),
            ).exit_code
            == 0
        )
        assert _install(runner, "--remove").exit_code == 0
        assert hooks.exists() and mcp.exists()


class TestEngineSelection:
    def test_an_inherited_engine_url_never_selects_the_thin_install(
        self, runner: CliRunner, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, cli_db: Path
    ) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://elsewhere.test")
        reset_config()
        result = _install(runner, "--json", "--dry-run")
        assert result.exit_code == 0, result.output
        out = json.loads(result.stdout)
        assert out.get("mode") != "thin-client"
        assert env["probes"] == []

    def test_the_probe_goes_to_the_installed_engine_not_the_environment(
        self, runner: CliRunner, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://elsewhere.test")
        reset_config()
        hooks = _key_file(env, "hooks", AGENT_KEY)
        mcp = _key_file(env, "mcp", MCP_KEY)
        result = _install(
            runner,
            "--engine",
            ENGINE,
            "--hooks-token-file",
            str(hooks),
            "--mcp-token-file",
            str(mcp),
        )
        assert result.exit_code == 0, result.output
        assert env["probes"]  # the fixture's probe asserts base_url == ENGINE

    def test_a_probe_refusal_writes_no_config(
        self, runner: CliRunner, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            thin,
            "probe_engine",
            lambda tokens, **_k: thin.ProbeResult(verified=True, refusals={"mcp": "wrong"}),
        )
        before = env["config"].read_text()
        hooks = _key_file(env, "hooks", AGENT_KEY)
        result = _install(runner, "--engine", ENGINE, "--hooks-token-file", str(hooks))
        assert result.exit_code == 1
        assert env["config"].read_text() == before


class TestRefusalsAtRunTime:
    def test_a_hook_logs_a_token_file_refusal_and_sends_nothing(
        self, runner: CliRunner, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli._claude_code import hook_log_path
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", ENGINE)
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN_FILE", str(env["keys"] / "absent"))
        reset_config()
        result = runner.invoke(
            app, ["hook", "session-start", "--store", "default"], input='{"session_id": "s"}'
        )
        assert result.exit_code == 0
        assert result.stdout == ""
        record = json.loads(hook_log_path().read_text().splitlines()[-1])
        assert record["outcome"] == "refused"
        assert record["reason"] == "token-file-refused"
        assert record["check"] == "missing"

    def test_the_mcp_server_refuses_to_start(
        self, runner: CliRunner, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli import mcp as mcp_cli
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", ENGINE)
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN_FILE", "relative/token")
        reset_config()
        monkeypatch.setattr(mcp_cli, "prepare_stdio_server_store", lambda *a, **k: None)
        served: list[bool] = []
        monkeypatch.setattr("particles.mcp.main", lambda: served.append(True))
        result = runner.invoke(app, ["mcp", "serve"])
        assert result.exit_code == 1
        assert "relative-path" in result.output
        assert served == []

    def test_a_401_with_a_file_token_says_to_restart(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from particles.api.client.http import _http_error

        path = _key_file(env, "hooks", AGENT_KEY)
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN_FILE", str(path))
        error = _http_error(httpx.Response(401, json={"detail": "bad key"}), "/particles/assert")
        assert "takes effect when the process restarts" in str(error)
        monkeypatch.delenv("PARTICLES_ENGINE_TOKEN_FILE")
        plain = _http_error(httpx.Response(401, json={"detail": "bad key"}), "/x")
        assert "restarts" not in str(plain)
