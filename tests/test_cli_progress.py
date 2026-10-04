# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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


def test_log_records_clear_the_heartbeat_line_while_it_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warning logged mid-heartbeat landed on the heartbeat's open line:

        … audit: working (20s elapsed)LLM unavailable (account-level: …)

    While the heartbeat runs, a stderr log handler prefixes each record with the
    erase sequence; afterwards its own formatter is back.
    """
    import logging

    from particles.api.cli import _progress
    from particles.config import get_config

    buf = _capture(monkeypatch, OutputSettings(progress=True))
    monkeypatch.setattr(get_config().cli, "heartbeat_seconds", 3600)
    handler = logging.StreamHandler(buf)  # buf is sys.stderr for this test
    original = logging.Formatter("%(message)s")
    handler.setFormatter(original)
    monkeypatch.setattr(logging.root, "handlers", [handler])
    logger = logging.getLogger("particles.test_heartbeat")

    with _progress.heartbeat("audit"):
        logger.error("LLM unavailable")
    logger.error("after")

    assert f"{_CLEAR_LINE}LLM unavailable\n" in buf.getvalue()
    assert buf.getvalue().endswith("after\n")
    assert handler.formatter is original


def test_heartbeat_paused_erases_the_line_and_holds_the_ticker_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verb's final output must not start on the heartbeat's open line."""
    import time

    from particles.api.cli import _progress
    from particles.config import get_config

    buf = _capture(monkeypatch, OutputSettings(progress=True))
    monkeypatch.setattr(get_config().cli, "heartbeat_seconds", 0.01)

    with _progress.heartbeat("audit"):
        _progress.set_heartbeat_status("contradiction probe 185/200")
        time.sleep(0.1)
        with _progress.heartbeat_paused():
            painted = buf.getvalue()
            time.sleep(0.1)
            assert buf.getvalue() == painted  # no tick while paused
            buf.write("Audited 96 memory files\n")
        assert _progress._status is None  # the stale probe status is gone

    assert "contradiction probe 185/200 (" in painted
    assert painted.endswith(_CLEAR_LINE)


def test_heartbeat_paused_writes_nothing_without_a_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from particles.api.cli import _progress

    buf = _capture(monkeypatch, OutputSettings(progress=False))

    with _progress.heartbeat_paused():
        buf.write("report\n")

    assert buf.getvalue() == "report\n"
