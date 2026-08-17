"""Tests for ``particles engine serve``.

The command unifies the bind argument with the fail-closed gate: it
derives ``api.bind_host`` from ``HOST:PORT`` and refuses to start a non-loopback
engine without a real ``PARTICLES_API_KEY`` *before* uvicorn binds. ``uvicorn.run``
is mocked throughout so no socket is ever opened.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli.engine import _parse_bind
from particles.api.daemon import set_projection_runner_factory
from particles.config import get_config, reset_config

_BIND_HOST_ENV = "PARTICLES_API_BIND_HOST"
_DAEMON_ENV = "PARTICLES_DAEMON_ENABLED"


@pytest.fixture(autouse=True)
def _restore_bind_host_env() -> Iterator[None]:
    """Undo `engine serve`'s deliberate write to the process environment.

    `engine.py` sets ``PARTICLES_API_BIND_HOST`` itself — the documented
    bootstrap path for a launcher configuring its own process, so the gate sees the interface uvicorn is about to bind. In production that write
    dies with the process; under ``CliRunner`` the command runs **in-process**,
    so it escapes into the pytest session and any later test reading
    ``api.bind_host`` sees ``0.0.0.0``.

    ``monkeypatch.delenv(..., raising=False)`` does not close this: when the
    variable is absent it registers no restore, so there is nothing to undo the
    write that happens afterwards. Hence an explicit snapshot.

    Latent until 2026-08-03, when the CI suite moved to ``-n auto --dist
    worksteal`` and `test_config.py::TestApiBindHost::test_default_is_loopback`
    first landed in the same worker after these tests.

    ``--daemon`` writes ``PARTICLES_DAEMON_ENABLED`` by exactly the
    same bootstrap path, so it is snapshotted here too.
    """
    before = {name: os.environ.get(name) for name in (_BIND_HOST_ENV, _DAEMON_ENV)}
    try:
        yield
    finally:
        for name, original in before.items():
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original
        # The config singleton cached whatever the command set; drop it so the
        # next test reads the restored environment rather than the stale value.
        reset_config()
        # `engine serve --daemon` registers the CLI-side projection factory on
        # the daemon module; clear it so it does not leak between tests.
        set_projection_runner_factory(None)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestParseBind:
    def test_host_and_port(self) -> None:
        assert _parse_bind("0.0.0.0:8000") == ("0.0.0.0", 8000)

    def test_localhost(self) -> None:
        assert _parse_bind("localhost:8000") == ("localhost", 8000)

    def test_ipv6_brackets_stripped(self) -> None:
        assert _parse_bind("[::1]:9000") == ("::1", 9000)

    def test_missing_colon_rejected(self) -> None:
        with pytest.raises(typer.BadParameter):
            _parse_bind("0.0.0.0")

    def test_non_integer_port_rejected(self) -> None:
        with pytest.raises(typer.BadParameter):
            _parse_bind("localhost:http")

    def test_out_of_range_port_rejected(self) -> None:
        with pytest.raises(typer.BadParameter):
            _parse_bind("localhost:70000")


class TestEngineServe:
    def test_non_loopback_without_key_refuses_before_bind(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # dev-key (unset) + non-loopback bind ⇒ refusal, no uvicorn.
        monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
        monkeypatch.delenv("PARTICLES_API_BIND_HOST", raising=False)
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(app, ["engine", "serve", "0.0.0.0:8123"])
        assert result.exit_code == 1
        assert "Refusing to start" in result.output
        mock_run.assert_not_called()

    def test_loopback_with_dev_key_starts(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # dev-key + loopback bind is allowed; uvicorn.run is invoked with the
        # parsed host/port and api.bind_host is set to match the bind.
        monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
        monkeypatch.delenv("PARTICLES_API_BIND_HOST", raising=False)
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(app, ["engine", "serve", "127.0.0.1:8124"])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["host"] == "127.0.0.1"
        assert mock_run.call_args.kwargs["port"] == 8124

    def test_serve_writes_bind_host_into_the_process_env(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the write the autouse fixture exists to undo.

        The write is deliberate (bootstrap), and under ``CliRunner`` it
        lands in the pytest process. Asserting it here means a future reader who
        deletes ``_restore_bind_host_env`` as redundant sees *why* it is not —
        and the conftest leak guard fails this file rather than whichever
        unrelated test is scheduled next, which is how it reached `main` on
        2026-08-03.
        """
        monkeypatch.setenv("PARTICLES_API_KEY", "real-secret")
        with patch("uvicorn.run"):
            runner.invoke(app, ["engine", "serve", "0.0.0.0:8126"])
        assert os.environ[_BIND_HOST_ENV] == "0.0.0.0"

    def test_non_loopback_with_real_key_starts(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A real key clears the gate, so a public bind is allowed to start.
        monkeypatch.setenv("PARTICLES_API_KEY", "real-secret")
        monkeypatch.delenv("PARTICLES_API_BIND_HOST", raising=False)
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(app, ["engine", "serve", "0.0.0.0:8125"])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["host"] == "0.0.0.0"


class TestDaemonFlag:
    """``--daemon`` — opt-in residency."""

    def test_absent_flag_leaves_daemon_mode_off(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the flag, `engine serve` is what it always was."""
        monkeypatch.setenv("PARTICLES_API_KEY", "real-secret")
        with patch("uvicorn.run"):
            result = runner.invoke(app, ["engine", "serve", "127.0.0.1:8127"])
        assert result.exit_code == 0, result.output
        assert _DAEMON_ENV not in os.environ
        assert get_config().daemon.enabled is False
        assert "mode: server only" in result.output

    def test_flag_wins_over_config(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "real-secret")
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(app, ["engine", "serve", "0.0.0.0:8128", "--daemon"])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        assert get_config().daemon.enabled is True
        assert "mode: resident daemon" in result.output

    def test_daemon_registers_the_projection_runner_factory(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pass 6 is Surface-injected; the launcher wires it."""
        from particles.api import daemon as daemon_mod

        monkeypatch.setenv("PARTICLES_API_KEY", "real-secret")
        with patch("uvicorn.run"):
            runner.invoke(app, ["engine", "serve", "0.0.0.0:8129", "--daemon"])
        assert daemon_mod._projection_runner_factory is not None

    def test_daemon_still_refuses_a_keyless_public_bind(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Daemon mode does not widen the gate — the image relies on this."""
        monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(app, ["engine", "serve", "0.0.0.0:8130", "--daemon"])
        assert result.exit_code == 1
        assert "Refusing to start" in result.output
        mock_run.assert_not_called()
