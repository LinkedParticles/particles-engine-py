# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Terminal-safe rendering of stored text for CLI output.

A stored claim, a subject name or a source URL is text that a source author,
an agent key or an imported bundle chose. Echoed raw, a control character in
it reaches the operator's terminal and is interpreted there. Every CLI path
that prints stored text passes it through :func:`terminal_safe` first, which
shows each control character as a visible escape instead (security review
2026-10-28-1, O39), or through :func:`terminal_echo`, which prints one value
that way. A bidirectional override or other invisible format character is
shown as an escape too, so stored text displays in the order it is stored
(2026-10-28-2, AA31).
"""

from __future__ import annotations

import unicodedata

import typer

#: Control characters kept as they are: they lay text out and drive nothing.
_KEPT = frozenset("\n\t")


def _is_control(ch: str) -> bool:
    """C0 controls, DEL, the C1 controls (U+0080 to U+009F) and format characters."""
    code = ord(ch)
    if code < 0x20 or code == 0x7F or 0x80 <= code <= 0x9F:
        return True
    return code > 0x9F and unicodedata.category(ch) == "Cf"


def _escape(ch: str) -> str:
    code = ord(ch)
    if code <= 0xFF:
        return f"\\x{code:02x}"
    return f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"


def terminal_safe(text: object) -> str:
    """``text`` as a string with every control character shown as an escape.

    C0 controls other than newline and tab, DEL and the C1 controls are
    rendered as ``\\xNN`` (for example ``\\x1b``), so the terminal displays
    them rather than acting on them, and an invisible format character (a
    bidirectional override, a zero-width space) as ``\\uNNNN``. Everything
    else, newline and tab included, is returned unchanged.

    Args:
        text: The stored value to print. A non-string is converted with
            ``str`` first.

    Returns:
        The text with no control character the terminal would interpret,
        apart from newline and tab.
    """
    s = text if isinstance(text, str) else str(text)
    if not any(_is_control(ch) and ch not in _KEPT for ch in s):
        return s
    return "".join(_escape(ch) if _is_control(ch) and ch not in _KEPT else ch for ch in s)


def terminal_echo(message: object = "", *, err: bool = False, nl: bool = True) -> None:
    """``typer.echo`` of ``message`` through :func:`terminal_safe`.

    The print every CLI verb uses for a line that carries stored text, so a
    control or format character in a subject name, a cached article or a claim
    is shown as an escape rather than acted on (security review 2026-10-28-2,
    AA31). A JSON payload is printed as it is: its serializer escapes control
    characters itself, and an escape here would make it invalid JSON.
    """
    typer.echo(terminal_safe(message), err=err, nl=nl)


def terminal_field(text: object) -> str:
    """``text`` as one line of a labelled field, every control character an escape.

    :func:`terminal_safe` with the line break shown as ``\\n`` too, for a
    stored value printed after a label (an actor, a reason, a description, a
    URI), so a line it holds cannot start at the margin and read as one of
    the verb's own field lines (security review 2026-10-30-1, AB70).
    """
    s = terminal_safe(text)
    return s.replace("\n", "\\n") if "\n" in s else s


def indented(text: str, indent: str) -> str:
    """``text`` with ``indent`` before every line, so a stored line break stays inside its field.

    A claim printed under a heading keeps each of its lines indented, so a
    line it holds cannot start at the margin and read as one of the verb's
    own field labels (security review 2026-10-28-2, AA62).
    """
    return "\n".join(f"{indent}{line}" for line in text.split("\n"))
