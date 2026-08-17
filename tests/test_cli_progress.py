"""The heartbeat and per-item progress must not collide on one line."""

from __future__ import annotations

import io
import sys

import pytest

from particles.api.cli._output import OutputSettings
from particles.api.cli._progress import _CLEAR_LINE, progress_line


def _capture(monkeypatch: pytest.MonkeyPatch, settings: OutputSettings) -> io.StringIO:
    from particles.api.cli import _output

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    monkeypatch.setattr(_output, "current_output", lambda: settings)
    return buf


def test_progress_line_clears_a_pending_heartbeat_on_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The heartbeat leaves the cursor mid-line on purpose; a progress line
    written straight after it produced

        … structure: working (5m00s elapsed)[140/500] d882de5a… Contribute

    on a real 500-particle run. Clearing first keeps the two writers composable.
    """
    buf = _capture(monkeypatch, OutputSettings(progress=True))

    progress_line("[140/500] d882de5a… Contribute")

    written = buf.getvalue()
    assert written == f"{_CLEAR_LINE}[140/500] d882de5a… Contribute\n"


def test_an_explicit_request_still_prints_when_stderr_is_piped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--verbose is an explicit ask; suppressing it off a TTY would break
    capturing a long run to a log. No heartbeat runs there, so no escape codes
    either — the line is byte-identical to the pre-helper output.
    """
    buf = _capture(monkeypatch, OutputSettings(progress=False))

    progress_line("[1/5] depositing foo.md")

    assert buf.getvalue() == "[1/5] depositing foo.md\n"


def test_quiet_silences_it(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = _capture(monkeypatch, OutputSettings(quiet=True))

    progress_line("should not appear")

    assert buf.getvalue() == ""
