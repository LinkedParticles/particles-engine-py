"""Tests for the CLI output policy (axes, precedence, stream rule)."""

from __future__ import annotations

import io
import time
from types import SimpleNamespace

import pytest

from particles.api.cli import _output, _progress
from particles.api.cli._output import (
    OutputSettings,
    configure_output,
    current_output,
    narrate,
)


class _FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _reset_output() -> None:
    _output._CURRENT.set(OutputSettings())


class TestShowProgressPrecedence:
    def test_quiet_beats_everything(self) -> None:
        # --quiet > --progress: even an explicit --progress is off.
        assert OutputSettings(quiet=True, progress=True).show_progress() is False

    def test_explicit_progress_overrides_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # --progress forces on even when stderr is piped (human watching stderr).
        monkeypatch.setattr("sys.stderr", io.StringIO())  # not a TTY
        assert OutputSettings(progress=True).show_progress() is True
        assert OutputSettings(progress=False).show_progress() is False

    def test_auto_follows_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stderr", _FakeTty())
        assert OutputSettings(progress=None).show_progress() is True
        monkeypatch.setattr("sys.stderr", io.StringIO())
        assert OutputSettings(progress=None).show_progress() is False


class TestConfigureOutput:
    def test_installs_settings_and_debug_implies_verbose(self) -> None:
        s = configure_output(verbose=False, debug=True, quiet=False, progress=False)
        assert s.debug is True and s.verbose is True  # --debug implies --verbose
        assert current_output() is s
        assert current_output().show_progress() is False

    def test_defaults_are_inert(self) -> None:
        s = configure_output()
        assert (s.verbose, s.debug, s.quiet, s.progress) == (False, False, False, None)


class TestNarrate:
    def test_narrate_goes_to_stderr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stream rule: narration is stderr, never the stdout artifact.
        out, err = io.StringIO(), io.StringIO()
        monkeypatch.setattr("sys.stdout", out)
        monkeypatch.setattr("sys.stderr", err)
        configure_output()
        narrate("working…")
        assert "working…" in err.getvalue()
        assert out.getvalue() == ""

    def test_quiet_silences_narration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        err = io.StringIO()
        monkeypatch.setattr("sys.stderr", err)
        configure_output(quiet=True)
        narrate("working…")
        assert err.getvalue() == ""


class TestHeartbeatHonoursSettings:
    def test_quiet_suppresses_heartbeat_even_on_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sink = _FakeTty()
        monkeypatch.setattr("sys.stderr", sink)
        monkeypatch.setattr(_progress, "get_config", lambda: _cfg(0.01))
        configure_output(quiet=True)
        with _progress.heartbeat("rebuild-utility"):
            time.sleep(0.05)
        assert sink.getvalue() == ""

    def test_progress_forces_heartbeat_off_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sink = io.StringIO()  # not a TTY
        monkeypatch.setattr("sys.stderr", sink)
        monkeypatch.setattr(_progress, "get_config", lambda: _cfg(0.01))
        configure_output(progress=True)
        with _progress.heartbeat("extract"):
            time.sleep(0.15)
        assert "extract: working" in sink.getvalue()

    def test_published_status_replaces_working(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A verb that knows its denominator upgrades the heartbeat from
        time-only "working" to a per-item position, elapsed time intact."""
        sink = io.StringIO()
        monkeypatch.setattr("sys.stderr", sink)
        monkeypatch.setattr(_progress, "get_config", lambda: _cfg(0.01))
        configure_output(progress=True)
        with _progress.heartbeat("reindex"):
            _progress.set_heartbeat_status("snapshot 12/89 (entry 0a8fb1a9…)")
            time.sleep(0.15)
        assert "reindex: snapshot 12/89 (entry 0a8fb1a9…) (" in sink.getvalue()
        # Reset on exit so one verb's last status can't bleed into the next.
        assert _progress._status is None


def _cfg(seconds: float) -> SimpleNamespace:
    return SimpleNamespace(cli=SimpleNamespace(heartbeat_seconds=seconds))
