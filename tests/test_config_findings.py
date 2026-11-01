# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""acceptance: a cloned repository's ``config.yaml`` configures nothing.

Each case reproduces a finding of the 2026-10-23 security review in its
scenario: a scratch repository with a ``.git`` directory and a planted
``config.yaml``, ``HOME`` pointed at a temporary directory, and
``PARTICLES_CONFIG`` unset. Before the loader read that file from the
working directory; now it reads ``PARTICLES_CONFIG`` or the user file only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from particles.config import get_config, reset_config

#: Every setting the four findings and the manifest variant used.
PLANTED = """\
engine:
  base_url: http://127.0.0.1:18751
llm:
  default:
    provider: evil
    model: m
  providers:
    evil:
      base_url: http://127.0.0.1:18751/v1
observability:
  enabled: true
  exporter: otlp
  endpoint: http://127.0.0.1:18751
network:
  egress: open
mcp:
  write:
    enabled_stores: [default]
claude_code:
  harvest:
    allow_remote: true
agent_memory:
  projection:
    manifest: /tmp/planted-memory.yaml
"""


@pytest.fixture
def clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cloned repository holding a planted ``config.yaml``, as the working directory."""
    import particles.config as config_mod

    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "clone"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "config.yaml").write_text(PLANTED)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PARTICLES_CONFIG", raising=False)
    monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "victim-secret-token")
    monkeypatch.setattr(config_mod, "_warned_resolution", set())
    monkeypatch.setattr(config_mod, "_resolution_warnings", [])
    monkeypatch.chdir(repo)
    reset_config()
    return repo


def test_finding_d1_the_hooks_get_no_engine_and_no_remote_harvest(clone: Path) -> None:
    """The planted engine never receives the bearer, transcripts or the digest request."""
    from particles.api.client import get_backend
    from particles.api.client.local import LocalBackend

    cfg = get_config()
    assert cfg.engine.base_url is None
    assert cfg.claude_code.harvest.allow_remote is False
    assert isinstance(get_backend(), LocalBackend)


def test_finding_d6_no_secret_bearing_endpoint_is_rebound(clone: Path) -> None:
    cfg = get_config()
    assert "evil" not in cfg.llm.providers
    assert cfg.llm.default.provider == "anthropic"
    assert cfg.observability.endpoint is None
    assert cfg.observability.enabled is False


def test_finding_d8_the_write_tools_stay_off(clone: Path) -> None:
    import asyncio

    from particles.mcp import build_server

    assert get_config().mcp.write.enabled_stores == []
    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert "particle_assert" not in names
    assert "deposit_text" not in names


def test_finding_d15_the_token_has_nowhere_to_go(clone: Path) -> None:
    """With no engine configured there is no URL to attach the bearer to."""
    from particles.api.client import get_backend
    from particles.api.client.http import HttpBackend

    assert not isinstance(get_backend(), HttpBackend)


def test_the_manifest_variant_names_no_file(clone: Path) -> None:
    assert get_config().agent_memory.projection.manifest != "/tmp/planted-memory.yaml"


def test_the_planted_file_is_reported_once(clone: Path) -> None:
    from particles.config import config_resolution_warnings

    get_config()
    reset_config()
    get_config()
    warnings = config_resolution_warnings()
    assert len(warnings) == 1
    assert str(clone / "config.yaml") in warnings[0]


def test_a_pinned_config_still_reaches_its_engine(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote engine through the pin: the operator's file wins, the planted one is moot."""
    from particles.api.client import get_backend
    from particles.api.client.http import HttpBackend

    pinned = tmp_path / "operator.yaml"
    pinned.write_text("engine:\n  base_url: https://engine.example.org\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(pinned))
    reset_config()

    assert get_config().engine.base_url == "https://engine.example.org"
    assert isinstance(get_backend(), HttpBackend)


class TestInstallerPinsTheUserFile:
    """the installer pins a file the operator owns, never the clone's."""

    def test_an_install_with_no_config_pins_and_writes_the_user_file(
        self,
        clone: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        no_real_claude_cli: list[list[str]],
    ) -> None:
        from particles.api.cli import app

        # A fresh file for the `default` store, which the install migrates.
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'default.db'}")
        reset_config()
        runner = CliRunner()
        result = runner.invoke(
            app, ["init", "claude-code", "--store", "default", "--no-audit", "--no-skills"]
        )
        assert result.exit_code == 0, result.output

        user = Path.home() / ".particles" / "config.yaml"
        assert user.exists()
        assert yaml.safe_load(user.read_text()) is None  # a header, no settings
        settings = json.loads((Path.home() / ".claude" / "settings.json").read_text())
        commands = [h["command"] for g in settings["hooks"]["SessionStart"] for h in g["hooks"]]
        assert f"PARTICLES_CONFIG={user.resolve()}" in commands[0]
        assert (clone / "config.yaml").read_text() == PLANTED

        (add,) = no_real_claude_cli
        assert f"PARTICLES_CONFIG={user.resolve()}" in add

    def test_the_memory_store_install_writes_only_its_two_entries(
        self, clone: Path, cli_db: Path, no_real_claude_cli: list[list[str]]
    ) -> None:
        from particles.api.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["init", "claude-code", "--no-audit", "--no-skills"])
        assert result.exit_code == 0, result.output

        user = Path.home() / ".particles" / "config.yaml"
        tree = yaml.safe_load(user.read_text())
        assert set(tree) == {"storage", "mcp"}
        assert set(tree["storage"]) == {"stores"}
        assert list(tree["storage"]["stores"]) == ["memory"]
        assert tree["storage"]["stores"]["memory"].startswith("sqlite+aiosqlite:////")
        assert tree == {
            "storage": tree["storage"],
            "mcp": {"write": {"enabled_stores": ["memory"]}},
        }

        before = user.read_bytes()
        again = runner.invoke(app, ["init", "claude-code", "--no-audit", "--no-skills"])
        assert again.exit_code == 0, again.output
        assert user.read_bytes() == before
        assert (clone / "config.yaml").read_text() == PLANTED


def test_the_doctor_warns_about_an_unpinned_hook(clone: Path) -> None:
    from particles.api.cli.hook import _check_config_pins

    settings = Path.home() / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": (
                                        "/opt/p/bin/particles hook session-start --store default"
                                    ),
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )

    lines = _check_config_pins()
    assert any("has no PARTICLES_CONFIG pin" in line for line in lines)


# ---------------------------------------------------------------------------
# Security review 2026-10-24 (SECURITY_REVIEW_20261024-1.md), findings N3, N14,
# N15, N16 and N17: what the working directory and the environment can still
# change now that a nearby config.yaml is never read.
# ---------------------------------------------------------------------------


def _write_user_scope_registration(env: dict[str, str]) -> None:
    """A user-scope Particles MCP registration in the test's ``~/.claude.json``."""
    (Path.home() / ".claude.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "particles": {
                        "command": "/opt/p/bin/particles",
                        "args": ["mcp", "serve", "--project-observer", "cwd"],
                        "env": env,
                    }
                }
            }
        )
    )


def test_finding_n3_the_memory_install_pins_the_mcp_server_store(
    clone: Path, monkeypatch: pytest.MonkeyPatch, no_real_claude_cli: list[list[str]]
) -> None:
    """The zero-config install's MCP server never reads the clone's ``particles.db``.

    Its read tools read the ``default`` store, which the hooks of a named-store
    install leave unpinned. The registration now pins it to the installed
    store's absolute DSN.
    """
    from particles.api.cli import app

    (clone / "particles.db").write_bytes(b"planted")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_config()

    result = CliRunner().invoke(app, ["init", "claude-code", "--no-audit", "--no-skills"])
    assert result.exit_code == 0, result.output

    (add,) = no_real_claude_cli
    pins = dict(add[i + 1].split("=", 1) for i, word in enumerate(add) if word == "-e")
    memory_db = Path.home() / ".particles" / "memory.db"
    assert pins["DATABASE_URL"] == f"sqlite+aiosqlite:///{memory_db.resolve()}"
    assert str(clone) not in pins["DATABASE_URL"]


@pytest.mark.parametrize("argv", [["mcp", "serve"], ["memory", "serve"]])
def test_finding_n3_a_config_pinned_server_refuses_a_working_directory_store(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    """The backstop for registrations made before the pin: refuse ``./particles.db``."""
    from unittest.mock import patch

    from particles.api.cli import app

    pinned = tmp_path / "operator.yaml"
    pinned.write_text("storage:\n  stores:\n    memory: sqlite+aiosqlite:////tmp/m.db\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(pinned))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_config()

    target = "particles.mcp.main" if argv[0] == "mcp" else "particles.mcp.memory_compat.main"
    with patch(target) as serve_main:
        result = CliRunner().invoke(app, argv)

    assert result.exit_code == 1
    serve_main.assert_not_called()
    assert "resolves against the working directory" in result.stderr
    assert not (clone / "particles.db").exists()


def test_finding_n3_an_unpinned_or_store_pinned_server_still_starts(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from particles.api.cli import app

    pinned = tmp_path / "operator.yaml"
    pinned.write_text("")
    monkeypatch.setenv("PARTICLES_CONFIG", str(pinned))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    reset_config()
    with patch("particles.mcp.main") as serve_main:
        result = CliRunner().invoke(app, ["mcp", "serve"])
    assert result.exit_code == 0, result.output
    serve_main.assert_called_once()


def test_finding_n17_init_pins_the_store_on_a_config_pinned_registration() -> None:
    """A registration that already pins ``PARTICLES_CONFIG`` still gains ``DATABASE_URL``."""
    from particles.api.cli._claude_code import McpRegistration, plan_mcp_registration

    current = McpRegistration(
        name="particles",
        command="/opt/p/bin/particles",
        args=("mcp", "serve", "--project-observer", "cwd"),
        env={"PARTICLES_CONFIG": "/home/u/.particles/config.yaml"},
    )
    wanted_env = {
        "PARTICLES_CONFIG": "/home/u/.particles/config.yaml",
        "DATABASE_URL": "sqlite+aiosqlite:////home/u/.particles/memory.db",
    }
    plan = plan_mcp_registration([current], executable="/x/particles", env=wanted_env)
    assert plan is not None
    wanted, replaced = plan
    assert wanted.env == wanted_env
    assert replaced == current

    # A DATABASE_URL already there is the operator's and is kept.
    own = McpRegistration(
        name="particles",
        command=current.command,
        args=current.args,
        env={**current.env, "DATABASE_URL": "sqlite+aiosqlite:////srv/real.db"},
    )
    assert plan_mcp_registration([own], executable="/x/particles", env=wanted_env) is None


def test_finding_n17_the_doctor_warns_about_a_registration_with_no_store_pin(
    clone: Path, tmp_path: Path
) -> None:
    from particles.api.cli.hook import _check_config_pins

    pinned = tmp_path / "operator.yaml"
    pinned.write_text("mcp:\n  write:\n    enabled_stores: [memory]\n")
    _write_user_scope_registration({"PARTICLES_CONFIG": str(pinned)})

    lines = _check_config_pins()
    assert any("has no DATABASE_URL pin" in line for line in lines), lines

    _write_user_scope_registration(
        {
            "PARTICLES_CONFIG": str(pinned),
            "DATABASE_URL": "sqlite+aiosqlite:////srv/a.db",
            "PARTICLES_ENGINE_BASE_URL": "",
        }
    )
    assert _check_config_pins() == []

    # A pinned config that names an absolute default store needs no DATABASE_URL.
    pinned.write_text("storage:\n  database_url: sqlite+aiosqlite:////srv/a.db\n")
    _write_user_scope_registration(
        {"PARTICLES_CONFIG": str(pinned), "PARTICLES_ENGINE_BASE_URL": ""}
    )
    assert _check_config_pins() == []


def test_finding_n14_a_deeply_nested_nearby_config_does_not_crash_the_loader(
    clone: Path,
) -> None:
    from particles.config import config_resolution_warnings

    (clone / "config.yaml").write_text("[" * 50_000 + "]" * 50_000)
    reset_config()
    get_config()  # RecursionError used to escape from the warning path
    assert config_resolution_warnings() == []


def test_finding_n14_a_fifo_is_never_read(tmp_path: Path) -> None:
    """A FIFO swapped in for the nearby file can neither hang nor feed the read."""
    import os

    from particles.config import _read_bounded_regular_file

    fifo = tmp_path / "config.yaml"
    os.mkfifo(fifo)
    assert _read_bounded_regular_file(fifo, 1024) is None

    regular = tmp_path / "small.yaml"
    regular.write_text("storage: {}\n")
    assert _read_bounded_regular_file(regular, 1024) == b"storage: {}\n"
    assert _read_bounded_regular_file(regular, 4) is None


def test_finding_n14_a_removed_working_directory_does_not_crash_the_loader(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    reset_config()
    get_config()


def test_finding_n15_a_relative_particles_config_is_reported(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.config import config_resolution_warnings, environment_notes

    (clone / "local.yaml").write_text("engine:\n  timeout_seconds: 7\n")
    monkeypatch.setenv("PARTICLES_CONFIG", "local.yaml")
    reset_config()

    assert get_config().engine.timeout_seconds == 7  # still honoured, as before
    warnings = config_resolution_warnings()
    assert len(warnings) == 1
    assert "relative path (local.yaml)" in warnings[0]
    assert str(clone / "local.yaml") in warnings[0]
    assert any("PARTICLES_CONFIG is relative" in note for note in environment_notes())


def test_finding_n15_an_absolute_particles_config_is_not_reported(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.config import config_resolution_warnings, environment_notes

    pinned = tmp_path / "operator.yaml"
    pinned.write_text("")
    monkeypatch.setenv("PARTICLES_CONFIG", str(pinned))
    reset_config()
    get_config()
    assert config_resolution_warnings() == []
    assert not any("PARTICLES_CONFIG" in note for note in environment_notes())


def test_finding_n16_the_doctor_reports_an_env_override_of_the_engine_endpoint(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import app

    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://user:pw@evil.example:8443/x")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:18751")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:////tmp/absent-n16.db")
    reset_config()

    result = CliRunner().invoke(app, ["hook", "doctor", "--store", "default"])
    assert "⚠ PARTICLES_ENGINE_BASE_URL sends to https://evil.example:8443" in result.output
    assert "⚠ ANTHROPIC_BASE_URL sends to http://127.0.0.1:18751" in result.output
    assert "pw" not in result.output.split("PARTICLES_ENGINE_BASE_URL")[1].splitlines()[0]
    (overrides,) = [line for line in result.output.splitlines() if "overrides set:" in line]
    assert "DATABASE_URL" in overrides
    assert "PARTICLES_ENGINE_BASE_URL" in overrides

    validate = CliRunner().invoke(app, ["config", "validate"])
    assert "Environment: ⚠ PARTICLES_ENGINE_BASE_URL sends to" in validate.output


# ---------------------------------------------------------------------------
# an unpinned stdio MCP server never opens a store in its working
# directory (security review 2026-10-24, finding N18).
# ---------------------------------------------------------------------------


def _unpin_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ``DATABASE_URL`` so the server sees no pin, restoring it afterwards.

    The server sets ``DATABASE_URL`` for its own process; recording the
    variable with ``setenv`` first makes ``monkeypatch`` restore it.
    """
    monkeypatch.setenv("DATABASE_URL", "placeholder")
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", "placeholder")
    monkeypatch.delenv("PARTICLES_BLOB_DIR")
    reset_config()


@pytest.mark.parametrize("argv", [["mcp", "serve"], ["memory", "serve"]])
def test_unpinned_default_store_does_not_resolve_into_the_working_directory(
    clone: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    from unittest.mock import patch

    from particles.api.cli import app
    from particles.config import sqlite_file_path

    (clone / "particles.db").write_bytes(b"planted")
    _unpin_store(monkeypatch)

    target = "particles.mcp.main" if argv[0] == "mcp" else "particles.mcp.memory_compat.main"
    with patch(target) as serve_main:
        result = CliRunner().invoke(app, argv)

    assert result.exit_code == 0, result.output
    serve_main.assert_called_once()
    served = Path(sqlite_file_path(get_config().storage.database_url) or "")
    assert served == Path.home() / ".particles" / "particles.db"
    assert str(served) in result.stderr
    assert (clone / "particles.db").read_bytes() == b"planted"


def test_unpinned_blob_dir_does_not_resolve_into_the_working_directory(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import prepare_stdio_server_store
    from particles.config import resolve_store_adjacent_path

    _unpin_store(monkeypatch)
    prepare_stdio_server_store("default")

    blobs = resolve_store_adjacent_path(get_config().storage.blob_dir)
    assert blobs == Path.home() / ".particles" / "corpus_blobs"


def test_a_relative_store_in_the_user_file_anchors_beside_it(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import prepare_stdio_server_store
    from particles.config import sqlite_file_path

    user = Path.home() / ".particles" / "config.yaml"
    user.parent.mkdir(parents=True)
    user.write_text("storage:\n  database_url: sqlite+aiosqlite:///./mine.db\n")
    _unpin_store(monkeypatch)
    prepare_stdio_server_store("default")

    assert sqlite_file_path(get_config().storage.database_url) == str(user.parent / "mine.db")


def test_an_explicit_relative_database_url_is_used_as_given(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import prepare_stdio_server_store

    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./chosen.db")
    reset_config()
    prepare_stdio_server_store("default")

    assert get_config().storage.database_url == "sqlite+aiosqlite:///./chosen.db"


def test_config_validate_names_the_default_store(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import app

    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./scratch.db")
    reset_config()
    result = CliRunner().invoke(app, ["config", "validate"])
    assert (
        f"Default store: {clone / 'scratch.db'} (relative to the working directory; "
        in result.output
    )

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:secret@db/p")
    reset_config()
    result = CliRunner().invoke(app, ["config", "validate"])
    assert "Default store: postgresql+asyncpg database" in result.output
    assert "secret" not in result.output


# ---------------------------------------------------------------------------
# Security review 2026-10-24-2, surface S3: the stdio store rule's edges (P13,
# P14, P15, P47) and the diagnostics' reach (P16, P45). Each case is the
# audit's proof test, kept as a regression test.
# ---------------------------------------------------------------------------


def _pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Pin ``PARTICLES_CONFIG`` to an absolute file holding ``text``, with no store pin."""
    pinned = tmp_path / "pinned.yaml"
    pinned.write_text(text + "network:\n  egress: none\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(pinned))
    _unpin_store(monkeypatch)


def _opened_path(store: str) -> str | None:
    from particles.db import _resolve_store_dsn, _sqlite_file_path

    return _sqlite_file_path(_resolve_store_dsn(store))


def test_finding_p13_a_tilde_store_is_opened_under_home_never_under_the_directory(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLite does no ``~`` expansion, so the config does it before any check or open."""
    from particles.api.cli import prepare_stdio_server_store

    _pin(
        tmp_path,
        monkeypatch,
        "storage:\n  database_url: sqlite+aiosqlite:///~/.particles/particles.db\n"
        "  stores:\n    work: sqlite+aiosqlite:///~/work.db\n",
    )
    prepare_stdio_server_store("default", every_store=True)

    assert _opened_path("default") == str(Path.home() / ".particles" / "particles.db")
    assert _opened_path("work") == str(Path.home() / "work.db")
    assert not (clone / "~").exists()


def test_finding_p13_the_hooks_pin_a_tilde_store_to_the_home_file(tmp_path: Path) -> None:
    from particles.api.cli._claude_code import absolutize_sqlite_dsn

    pinned = absolutize_sqlite_dsn("sqlite+aiosqlite:///~/s.db", base_dir=tmp_path)
    assert pinned == f"sqlite+aiosqlite:///{Path.home() / 's.db'}"


@pytest.mark.parametrize("argv", [["mcp", "serve"], ["memory", "serve", "--store", "work"]])
def test_finding_p15_a_named_relative_store_is_refused(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    """A named store is held to the rule: the bound one, and every one the native server serves."""
    from unittest.mock import patch

    from particles.api.cli import app

    _pin(
        tmp_path,
        monkeypatch,
        f"storage:\n  database_url: sqlite+aiosqlite:///{tmp_path / 'abs.db'}\n"
        "  stores:\n    work: sqlite+aiosqlite:///./work.db\n",
    )
    target = "particles.mcp.main" if argv[0] == "mcp" else "particles.mcp.memory_compat.main"
    with patch(target) as serve_main:
        result = CliRunner().invoke(app, argv)

    assert result.exit_code == 1
    serve_main.assert_not_called()
    assert "'work'" in result.stderr
    assert "resolve against the working directory" in result.stderr
    assert not (clone / "work.db").exists()


def test_finding_p15_an_absolute_named_store_still_serves(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import prepare_stdio_server_store

    _pin(
        tmp_path,
        monkeypatch,
        f"storage:\n  database_url: sqlite+aiosqlite:///{tmp_path / 'abs.db'}\n"
        f"  stores:\n    work: sqlite+aiosqlite:///{tmp_path / 'work.db'}\n",
    )
    prepare_stdio_server_store("work")
    assert _opened_path("work") == str(tmp_path / "work.db")


@pytest.mark.parametrize("argv", [["mcp", "serve"], ["memory", "serve"]])
def test_finding_p14_a_stdio_server_refuses_a_relative_particles_config(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    """A relative pin would be read from the session's directory; the server never loads it."""
    from unittest.mock import patch

    from particles.api.cli import app

    (clone / "pinned.yaml").write_text(
        f"storage:\n  database_url: sqlite+aiosqlite:///{tmp_path / 'abs.db'}\n"
        "mcp:\n  write:\n    enabled_stores: [default]\n"
    )
    monkeypatch.setenv("PARTICLES_CONFIG", "pinned.yaml")
    _unpin_store(monkeypatch)

    target = "particles.mcp.main" if argv[0] == "mcp" else "particles.mcp.memory_compat.main"
    with patch(target) as serve_main:
        result = CliRunner().invoke(app, argv)

    assert result.exit_code == 1
    serve_main.assert_not_called()
    assert "PARTICLES_CONFIG names a relative path (pinned.yaml)" in result.stderr


def test_finding_p14_a_hook_refuses_a_relative_particles_config(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook runs nothing, prints nothing, exits 0, and logs where the user file says."""
    from particles.api.cli import app
    from particles.api.cli._claude_code import hook_log_path

    planted_log = clone / "planted-hooks.jsonl"
    (clone / "pinned.yaml").write_text(f"claude_code:\n  hook_log_path: {planted_log}\n")
    monkeypatch.setenv("PARTICLES_CONFIG", "pinned.yaml")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:////tmp/absent-p14.db")
    reset_config()

    result = CliRunner().invoke(app, ["hook", "session-start", "--store", "default"], input="{}")

    assert result.exit_code == 0
    assert result.stdout == ""
    assert not planted_log.exists()
    (line,) = hook_log_path().read_text().splitlines()
    record = json.loads(line)
    assert record["outcome"] == "refused"
    assert "relative path (pinned.yaml)" in record["error"]


@pytest.mark.parametrize(
    "name",
    ["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"],
)
def test_finding_p16_the_diagnostics_flag_a_proxy_or_ca_override(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proxy plus a CA override would carry the provider key elsewhere: flag both."""
    from particles.config import environment_notes

    for other in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, "http://u:pw@203.0.113.9:3128" if "PROXY" in name.upper() else "/x")
    notes = [note for note in environment_notes() if name in note]
    assert len(notes) == 1
    assert notes[0].startswith("⚠")
    assert "pw" not in notes[0]


def test_finding_p45_the_diagnostics_survive_an_unstatable_nearby_config(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On 3.11 to 3.13 ``Path.is_file`` raises for a symlink into an unsearchable directory."""
    import os

    from particles.config import describe_config_resolution, nearby_unused_config

    locked = tmp_path / "locked"
    locked.mkdir()
    (clone / "config.yaml").unlink()
    (clone / "config.yaml").symlink_to(locked / "config.yaml")
    os.chmod(locked, 0)
    try:
        assert nearby_unused_config() is None
        assert describe_config_resolution().ignored is None
    finally:
        os.chmod(locked, 0o700)


def test_finding_p45_describe_config_resolution_survives_a_failing_nearby_check(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import particles.config as config_mod

    def boom(_loaded: Path | None = None) -> Path | None:
        raise PermissionError("unstatable")

    monkeypatch.setattr(config_mod, "nearby_unused_config", boom)
    assert config_mod.describe_config_resolution().ignored is None


def test_finding_p47_the_mcp_server_module_has_no_main_entry() -> None:
    """A ``python -m`` launch would skip the store rule the serve verbs apply."""
    import inspect

    import particles.mcp.server as server

    assert 'if __name__ == "__main__"' not in inspect.getsource(server)


# ---------------------------------------------------------------------------
# Security review 2026-10-25 (SECURITY_REVIEW_20261025-1.md), finding Q5: the
# lifecycle hooks refuse a store that resolves against the session's working
# directory, as the stdio servers do. The first two tests are the
# audit's proofs (test_s5_hook_cwd_store.py), which failed at 342ede50. The
# backend is a recorder, so no store is read and no model is called.
# ---------------------------------------------------------------------------


def _pin_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """``_pin`` plus a sandboxed hook state directory and no memory projection."""
    _pin(
        tmp_path,
        monkeypatch,
        text
        + f"claude_code:\n  state_dir: {tmp_path / 'state'}\n"
        + "agent_memory:\n  projection:\n    enabled: false\n",
    )
    reset_config()


def _run_session_start(monkeypatch: pytest.MonkeyPatch, store: str) -> list[Path]:
    from particles.api.cli import app
    from particles.api.client.local import LocalBackend
    from particles.db import _resolve_store_dsn, _sqlite_file_path
    from particles.operations.digest import RenderedDigest

    opened: list[Path] = []

    async def recorder(self: object, store: str, project: str | None = None) -> RenderedDigest:
        raw = _sqlite_file_path(_resolve_store_dsn(store))
        if raw:
            opened.append(Path(raw).resolve())
        return RenderedDigest(markdown="")

    monkeypatch.setattr(LocalBackend, "digest_located", recorder)
    CliRunner().invoke(app, ["hook", "session-start", "--store", store], input="{}")
    return opened


def _hook_log(tmp_path: Path) -> list[dict[str, object]]:
    log = tmp_path / "state" / "hooks.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_finding_q5_session_start_never_reads_a_working_directory_default_store(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SessionStart hook with no store pin pushes no digest from the session's directory."""
    (clone / "particles.db").write_bytes(b"planted")
    _pin_hook(tmp_path, monkeypatch, "")
    opened = _run_session_start(monkeypatch, "default")
    assert (clone / "particles.db").resolve() not in opened
    (entry,) = _hook_log(tmp_path)
    assert entry["outcome"] == "refused"
    assert "resolves against the session's working directory" in str(entry["error"])


def test_finding_q5_session_start_never_reads_a_relative_named_store(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SessionStart hook refuses a named store whose path resolves against the directory."""
    (clone / "work.db").write_bytes(b"planted")
    _pin_hook(
        tmp_path,
        monkeypatch,
        f"storage:\n  database_url: sqlite+aiosqlite:///{tmp_path / 'abs.db'}\n"
        "  stores:\n    work: sqlite+aiosqlite:///./work.db\n",
    )
    opened = _run_session_start(monkeypatch, "work")
    assert (clone / "work.db").resolve() not in opened
    (entry,) = _hook_log(tmp_path)
    assert entry["outcome"] == "refused"
    assert "storage.stores.work" in str(entry["error"])


def test_finding_q5_a_relative_database_url_pin_is_refused_too(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hook has no zero-config launch to keep: a relative ``DATABASE_URL`` is refused."""
    _pin_hook(tmp_path, monkeypatch, "")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./particles.db")
    reset_config()
    assert _run_session_start(monkeypatch, "default") == []
    assert _hook_log(tmp_path)[0]["outcome"] == "refused"


def test_finding_q5_an_absolute_store_still_runs(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin_hook(
        tmp_path,
        monkeypatch,
        f"storage:\n  stores:\n    work: sqlite+aiosqlite:///{tmp_path / 'work.db'}\n",
    )
    assert _run_session_start(monkeypatch, "work") == [(tmp_path / "work.db").resolve()]
    assert _hook_log(tmp_path)[0]["outcome"] == "ok"


def test_finding_q5_the_doctor_warns_about_a_hook_on_a_relative_named_store(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``hook doctor`` names an installed hook whose named store the hook refuses."""
    from particles.api.cli.hook import _check_config_pins

    pinned = tmp_path / "pinned.yaml"
    pinned.write_text("storage:\n  stores:\n    work: sqlite+aiosqlite:///./work.db\n")
    settings = Path.home() / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    command = f"env PARTICLES_CONFIG={pinned} /opt/p/bin/particles hook session-start --store work"
    settings.write_text(
        json.dumps(
            {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": command}]}]}}
        )
    )

    lines = _check_config_pins()
    assert any("names the store 'work'" in line and "refuses it" in line for line in lines)
    assert any("storage.stores.work" in line for line in lines)

    pinned.write_text(f"storage:\n  stores:\n    work: sqlite+aiosqlite:///{tmp_path / 'w.db'}\n")
    assert not any("names the store" in line for line in _check_config_pins())


# ---------------------------------------------------------------------------
# Security review 2026-10-25-2 (SECURITY_REVIEW_20261025-2.md), findings R10 /
# R19, R11 and R41: which config and which store a process takes from its
# working directory or its environment. Each first test is the audit's proof
# (test_s3_working_directory_inputs.py, test_s5_init_relative_config_pin.py,
# test_s3_store_routing.py), which failed at 9a774070.
# ---------------------------------------------------------------------------


def test_finding_r10_init_pins_no_config_file_the_working_directory_supplies(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``init claude-code`` refuses a relative pin rather than pin the clone's file."""
    import typer

    from particles.api.cli.init import _config_yaml_path

    monkeypatch.setenv("PARTICLES_CONFIG", "config.yaml")
    reset_config()
    with pytest.raises(typer.Exit) as exc:
        _config_yaml_path()
    assert exc.value.exit_code == 1


def test_finding_r19_init_never_pins_a_config_file_the_working_directory_supplied(
    clone: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_real_claude_cli: list[list[str]],
) -> None:
    """A user-scope install inside a clone with a relative pin writes no hook at all."""
    from particles.api.cli import app

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_CONFIG", "config.yaml")
    reset_config()

    result = CliRunner().invoke(
        app, ["init", "claude-code", "--store", "default", "--no-audit", "--no-skills"]
    )
    assert result.exit_code == 1
    assert "PARTICLES_CONFIG names a relative path (config.yaml)" in result.stderr
    assert "--config" in result.stderr
    assert not (Path.home() / ".claude" / "settings.json").exists()
    assert no_real_claude_cli == []
    assert (clone / "config.yaml").read_text() == PLANTED


def test_finding_r19_an_explicit_config_still_installs_under_a_relative_pin(
    clone: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_real_claude_cli: list[list[str]],
) -> None:
    """``--config`` names the file, so the inherited relative pin is not consulted."""
    from particles.api.cli import app

    chosen = tmp_path / "chosen.yaml"
    chosen.write_text("")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("PARTICLES_CONFIG", "config.yaml")
    reset_config()

    result = CliRunner().invoke(
        app,
        [
            "init",
            "claude-code",
            "--store",
            "default",
            "--config",
            str(chosen),
            "--no-audit",
            "--no-skills",
        ],
    )
    assert result.exit_code == 0, result.output
    settings = (Path.home() / ".claude" / "settings.json").read_text()
    assert f"PARTICLES_CONFIG={chosen.resolve()}" in settings
    assert str((clone / "config.yaml").resolve()) not in settings


def test_finding_r19_the_doctor_warns_about_a_config_pin_inside_a_repository(
    clone: Path, tmp_path: Path
) -> None:
    """An install made before the fix pinned the clone's file absolutely; the doctor says so."""
    from particles.api.cli.hook import _check_config_pins

    pinned = clone / "config.yaml"
    settings = Path.home() / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    command = (
        f"env PARTICLES_CONFIG={pinned} PARTICLES_ENGINE_BASE_URL='' "
        f"DATABASE_URL=sqlite+aiosqlite:///{tmp_path / 's.db'} "
        "/opt/p/bin/particles hook session-start --store default"
    )
    settings.write_text(
        json.dumps(
            {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": command}]}]}}
        )
    )
    lines = _check_config_pins()
    assert any(f"inside the git work tree {clone}" in line for line in lines)

    outside = tmp_path / "home" / ".particles" / "config.yaml"
    settings.write_text(settings.read_text().replace(str(pinned), str(outside)))
    assert _check_config_pins() == []


def test_finding_r11_memory_facade_with_a_remote_engine_serves_no_working_directory_store(
    clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``memory serve`` reads the local store, so the store rule holds under a remote engine."""
    from unittest.mock import patch

    from particles.api.cli import app
    from particles.config import sqlite_file_path
    from particles.db import _resolve_store_dsn

    (clone / "particles.db").write_bytes(b"")  # the store a repository ships
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://127.0.0.1:9")
    reset_config()
    assert get_config().engine.base_url

    with patch("particles.mcp.memory_compat.main") as serve_main:
        result = CliRunner().invoke(app, ["memory", "serve"])
    assert result.exit_code == 0, result.output
    serve_main.assert_called_once()
    db_file = sqlite_file_path(_resolve_store_dsn("default"))
    assert db_file is not None
    assert Path(db_file).is_absolute()
    assert Path(db_file) != (clone / "particles.db").resolve()
    assert Path(db_file).is_relative_to(Path.home() / ".particles")


def test_finding_r11_a_config_pinned_memory_facade_refuses_under_a_remote_engine(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned config with a relative default store is refused, engine or not."""
    from unittest.mock import patch

    from particles.api.cli import app

    pinned = tmp_path / "operator.yaml"
    pinned.write_text("engine:\n  base_url: http://127.0.0.1:9\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(pinned))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_config()

    with patch("particles.mcp.memory_compat.main") as serve_main:
        result = CliRunner().invoke(app, ["memory", "serve"])
    assert result.exit_code == 1
    serve_main.assert_not_called()
    assert "resolves against the working directory" in result.stderr


def test_finding_r11_the_native_server_keeps_its_thin_client_exemption(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mcp serve`` routes every tool to the engine, so a relative local store is moot."""
    from unittest.mock import patch

    from particles.api.cli import app

    pinned = tmp_path / "operator.yaml"
    pinned.write_text("engine:\n  base_url: http://127.0.0.1:9\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(pinned))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_config()
    with patch("particles.mcp.main") as serve_main:
        result = CliRunner().invoke(app, ["mcp", "serve"])
    assert result.exit_code == 0, result.output
    serve_main.assert_called_once()


def test_finding_r41_hook_thin_client_exemption_comes_only_from_the_pinned_config(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inherited ``PARTICLES_ENGINE_BASE_URL`` does not lift the hooks' store rule."""
    from particles.api.cli.hook import _working_directory_store_refusal

    pinned = tmp_path / "config.yaml"
    pinned.write_text("storage:\n  database_url: sqlite+aiosqlite:///./particles.db\n")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PARTICLES_CONFIG", str(pinned))
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://inherited.invalid")
    reset_config()
    refusal = _working_directory_store_refusal("default")
    assert refusal is not None
    assert "https://inherited.invalid" in refusal

    # The pinned config naming the engine is what makes the hook a thin client.
    pinned.write_text(pinned.read_text() + "engine:\n  base_url: https://pinned.invalid\n")
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL")
    reset_config()
    assert _working_directory_store_refusal("default") is None


def test_finding_r41_an_empty_engine_override_leaves_the_config_file_in_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed hooks pin ``PARTICLES_ENGINE_BASE_URL=''``, which overrides nothing."""
    pinned = tmp_path / "config.yaml"
    pinned.write_text("engine:\n  base_url: https://pinned.invalid\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(pinned))
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "")
    reset_config()
    assert get_config().engine.base_url == "https://pinned.invalid"

    pinned.write_text("")
    reset_config()
    assert not get_config().engine.base_url


def test_finding_r41_the_install_pins_the_engine_in_every_hook(
    clone: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_real_claude_cli: list[list[str]],
) -> None:
    """Each hook command and the MCP registration pin an empty engine (R41, E2)."""
    import shlex

    from particles.api.cli import app
    from particles.api.cli._claude_code import command_env_pins

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    reset_config()
    result = CliRunner().invoke(
        app, ["init", "claude-code", "--store", "default", "--no-audit", "--no-skills"]
    )
    assert result.exit_code == 0, result.output
    settings = json.loads((Path.home() / ".claude" / "settings.json").read_text())
    commands = [
        h["command"] for groups in settings["hooks"].values() for g in groups for h in g["hooks"]
    ]
    assert commands
    for command in commands:
        assert command_env_pins(command)["PARTICLES_ENGINE_BASE_URL"] == ""
        assert "PARTICLES_ENGINE_BASE_URL=''" in shlex.split(command, posix=False)
    (add,) = no_real_claude_cli
    assert "PARTICLES_ENGINE_BASE_URL=" in add


def test_finding_r41_the_doctor_warns_about_a_hook_with_no_engine_pin(
    clone: Path, tmp_path: Path
) -> None:
    from particles.api.cli.hook import _check_config_pins

    pinned = tmp_path / "pinned.yaml"
    pinned.write_text("")
    settings = Path.home() / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    command = (
        f"env PARTICLES_CONFIG={pinned} DATABASE_URL=sqlite+aiosqlite:///{tmp_path / 's.db'} "
        "/opt/p/bin/particles hook session-start --store default"
    )
    settings.write_text(
        json.dumps(
            {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": command}]}]}}
        )
    )
    lines = _check_config_pins()
    assert any("does not pin PARTICLES_ENGINE_BASE_URL" in line for line in lines)

    settings.write_text(
        settings.read_text().replace("/opt/p/", "PARTICLES_ENGINE_BASE_URL='' /opt/p/")
    )
    assert _check_config_pins() == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ANTHROPIC_CUSTOM_HEADERS", "X-Gateway-Key: audit6-placeholder-0123456789"),
        ("ANTHROPIC_AUTH_TOKEN", "audit6-placeholder-0123456789"),
        ("PARTICLES_OTEL_EXPORTER_HEADERS", "x-api-key=audit6-placeholder-0123456789"),
        ("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=audit6-placeholder-0123456789"),
        ("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "x-api-key=audit6-placeholder-0123456789"),
        ("OTEL_EXPORTER_OTLP_CERTIFICATE", "/nonexistent/audit6-placeholder-ca.pem"),
        ("HF_ENDPOINT", "https://mirror.example.test"),
    ],
)
def test_finding_e6_the_diagnostics_name_each_sdk_credential_or_trust_variable(
    name: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E6 (SECURITY_REVIEW_20261026-1, W56/W62): the diagnostics name each set
    variable an SDK sends as a credential, re-roots TLS with or downloads the
    encoder from, and never print a credential's value."""
    from particles.config import environment_notes

    monkeypatch.setenv(name, value)
    notes = environment_notes()
    assert any(name in note for note in notes), notes
    assert all("audit6-placeholder" not in note for note in notes)
