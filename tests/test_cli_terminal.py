# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The CLI prints stored text with no control character reaching the terminal live.

A stored claim is text a source author, an agent key or an imported bundle
chose, so the ``particle`` verbs show every control character in it as a
visible escape (security review 2026-10-28-1, O39). No model call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli._terminal import terminal_safe

#: Text carrying a C0 control sequence, BEL, DEL and a C1 control.
CONTROLLED = "Plain claim \x1b]0;title\x07 \x1b[2J \x7f \x9b tail"

#: Every character terminal_safe escapes.
_ESCAPED = [chr(c) for c in range(0x20) if chr(c) not in "\n\t"] + [
    chr(0x7F),
    *[chr(c) for c in range(0x80, 0xA0)],
]


def _has_live_control(text: str) -> bool:
    return any(ch in text for ch in _ESCAPED)


class TestTerminalSafe:
    def test_plain_text_is_unchanged(self) -> None:
        text = "Café opens at 9 → closes at 5. «ok»"
        assert terminal_safe(text) is text

    def test_newline_and_tab_are_kept(self) -> None:
        assert terminal_safe("a\tb\nc") == "a\tb\nc"

    @pytest.mark.parametrize("ch", _ESCAPED)
    def test_each_control_character_is_shown_as_an_escape(self, ch: str) -> None:
        out = terminal_safe(f"x{ch}y")
        assert out == f"x\\x{ord(ch):02x}y"
        assert not _has_live_control(out)

    def test_escape_is_visible_and_lowercase_hex(self) -> None:
        assert terminal_safe("\x1b[2J") == "\\x1b[2J"
        assert terminal_safe("\x9b") == "\\x9b"

    def test_a_non_string_is_converted(self) -> None:
        assert terminal_safe(42) == "42"


def test_particle_show_writes_no_terminal_control_sequence(cli_db: Path) -> None:
    """`particles particle show` prints a stored claim with every control character escaped."""
    from tests.test_cli_particle_source import _seed

    pid = asyncio.run(_seed(CONTROLLED, with_source=False))
    result = CliRunner().invoke(app, ["particle", "show", pid[:8]], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert not _has_live_control(result.stdout), repr(result.stdout)
    assert "Plain claim \\x1b]0;title\\x07 \\x1b[2J \\x7f \\x9b tail" in result.stdout


def test_particle_retract_preview_writes_no_terminal_control_sequence(cli_db: Path) -> None:
    """The retract preview shows the stored claim with every control character escaped."""
    from tests.test_cli_particle_source import _seed

    pid = asyncio.run(_seed(CONTROLLED, with_source=False))
    result = CliRunner().invoke(
        app,
        ["particle", "retract", pid[:8], "--reason", "wrong", "--dry-run"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert not _has_live_control(result.stdout), repr(result.stdout)
    assert "\\x1b[2J" in result.stdout
