"""Every `particles …` command we *suggest* to an operator must actually run.

Operator-facing output tells people what to do next ("run `particles reindex`
to retry them"). Those strings are hand-written and never executed, so they rot
silently: four separate surfaces shipped `particles reindex --failed`, a flag
that has never existed, and the error only surfaced when an operator typed it.

This test parses every suggested command out of the source and validates it
against the real Typer/Click tree — subcommand path and every long option. It
is deliberately a *static* check: it never invokes anything, so it costs
nothing and cannot have side effects on a store.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest
import typer.main

from particles.api.cli import app

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "particles"

#: Suggested commands, matched ONLY where they are formatted as commands:
#: backtick-quoted (the dominant convention) or after a literal "run: ".
#: A looser pattern is not viable — "particles" is also the domain noun, so
#: bare matching sweeps up prose like "particles linked to a subject".
_SUGGESTIONS = (
    re.compile(r"`particles ([a-z][^`]*)`"),
    re.compile(r"run: particles ([a-z][^)\"'\n]*)"),
)

#: Argument placeholders an operator substitutes; never option names.
_PLACEHOLDER = re.compile(r"^(<.*>|\[.*\]|…|\.\.\.)$")

#: A token that makes the command path indeterminate: a trailing wildcard
#: (`benchmark*` = the benchmark/-modality/-polarity family) or an ellipsis
#: standing in for an unnamed subcommand (`hook … --store`). The command a
#: reader would run cannot be resolved, so options after it are unverifiable.
_INDETERMINATE = re.compile(r"(\*$|^…$|^\.\.\.$)")


def _iter_suggestions() -> list[tuple[Path, int, str]]:
    """Every suggested command string in the package, with its source location."""
    found: list[tuple[Path, int, str]] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # Only look inside string literals — skip comments and code.
            if '"' not in line and "'" not in line and "`" not in line:
                continue
            for pattern in _SUGGESTIONS:
                for match in pattern.finditer(line):
                    found.append((path, lineno, match.group(1).strip()))
    return found


def _resolve(tokens: list[str]) -> tuple[click.Command | None, list[str]]:
    """Walk the click tree; return (command, remaining tokens) or (None, _).

    Duck-types on ``get_command`` rather than ``isinstance(_, click.Group)``:
    Typer's ``TyperGroup`` does not subclass ``click.Group`` in every version
    (it derives from ``typer._click.core.Command``), so an isinstance check
    silently refuses to descend and every suggestion resolves to the root app.
    """
    command: click.Command = typer.main.get_command(app)
    index = 0
    while index < len(tokens) and not tokens[index].startswith("-"):
        if _PLACEHOLDER.match(tokens[index]):
            break  # `<id>` / `[PATH]` — an argument placeholder, not a subcommand
        get_command = getattr(command, "get_command", None)
        if get_command is None:
            break  # a leaf command; the rest are arguments, not subcommands
        nxt = get_command(click.Context(command), tokens[index])
        if nxt is None:
            return None, tokens[index:]
        command = nxt
        index += 1
    return command, tokens[index:]


def _known_options(command: click.Command) -> set[str]:
    names: set[str] = set()
    for param in command.params:
        names.update(opt for opt in param.opts if opt.startswith("--"))
        names.update(opt for opt in param.secondary_opts if opt.startswith("--"))
    return names


def test_suggestions_exist() -> None:
    """Guard the guard: if the regex stops matching, the test is worthless."""
    assert len(_iter_suggestions()) >= 5


@pytest.mark.parametrize("path,lineno,suggestion", _iter_suggestions())
def test_suggested_command_is_valid(path: Path, lineno: int, suggestion: str) -> None:
    """A `particles …` string in operator output resolves to a real command+flags."""
    tokens = suggestion.split()
    if any(_INDETERMINATE.search(t) for t in tokens):
        pytest.skip(f"indeterminate command path: `particles {suggestion}`")
    command, remainder = _resolve(tokens)
    where = f"{path.relative_to(_PACKAGE_ROOT.parent)}:{lineno}"
    assert command is not None, (
        f"{where}: suggested `particles {suggestion}` — no such command "
        f"(failed at {remainder[0]!r})"
    )

    known = _known_options(command)
    for token in remainder:
        if not token.startswith("--") or _PLACEHOLDER.match(token):
            continue
        assert token in known, (
            f"{where}: suggested `particles {suggestion}` — "
            f"`{token}` is not an option of that command. Known: {sorted(known)}"
        )
