"""CLI output policy — the three orthogonal axes and the stream rule.

**stdout is the artifact; stderr is the narration.** The result (a report, JSON,
a table) goes to stdout and nothing else; progress, diagnostics, warnings, and
errors go to stderr. That one rule makes piping and ``--output-format json``
correct by construction and gives ``--quiet`` an unambiguous meaning.

Three independent knobs, not one vague "verbosity":

- ``--verbose`` / ``--debug`` — **diagnostics** (how much internal detail; stderr).
  ``--verbose`` also un-aggregates per-item detail; ``--debug`` implies it.
- ``--progress`` / ``--no-progress`` — **liveness** (stderr). Default *auto*: on
  when stderr is a TTY, off otherwise; an explicit value overrides.
- ``--output-format`` — **payload shape** (stdout); declared per-verb where it applies.

``--quiet`` is not a fourth axis — it means *narration off*: progress and
non-error diagnostics suppressed, the stdout artifact and errors kept.

A verb declares the shared options (``VERBOSE_OPTION`` …) and calls
:func:`configure_output` in one line, mirroring the old ``configure_logging``.
Enforcement — the heartbeat (``_progress``) and :func:`narrate` — reads the
resolved settings from a context var, so a verb that has **not** adopted the
options still gets the universal liveness floor.
"""

from __future__ import annotations

import contextvars
import sys
from dataclasses import dataclass

import typer

from particles.api.cli._logging import configure_logging


@dataclass(frozen=True)
class OutputSettings:
    """Resolved per-invocation output behaviour."""

    verbose: bool = False
    debug: bool = False
    quiet: bool = False
    #: Explicit ``--progress`` / ``--no-progress``; ``None`` ⇒ auto (on iff a TTY).
    progress: bool | None = None

    def show_progress(self) -> bool:
        """Whether the liveness heartbeat should run (precedence: quiet > explicit > auto)."""
        if self.quiet:
            return False
        if self.progress is not None:
            return self.progress
        return sys.stderr.isatty()

    def narrate_ok(self) -> bool:
        """Whether non-error narration (progress / status lines) should print."""
        return not self.quiet


#: Default settings for an invocation that has not called ``configure_output``
#: (a verb that has not adopted the options, or a non-CLI import). The context var
#: itself defaults to ``None`` — not a constructed value — so the linter's
#: mutable-ContextVar-default rule (B039) is satisfied; ``current_output`` maps
#: ``None`` to this shared, immutable instance.
_DEFAULT_OUTPUT = OutputSettings()

_CURRENT: contextvars.ContextVar[OutputSettings | None] = contextvars.ContextVar(
    "particles_output_settings", default=None
)


def current_output() -> OutputSettings:
    """The active invocation's resolved output settings (defaults if unset)."""
    settings = _CURRENT.get()
    return settings if settings is not None else _DEFAULT_OUTPUT


# Shared Typer option declarations — one definition reused by every adopting verb,
# so the three axes stay identical everywhere instead of being re-invented per verb.
VERBOSE_OPTION = typer.Option(
    False, "--verbose", "-v", help="Raise diagnostics to INFO and un-aggregate per-item detail."
)
DEBUG_OPTION = typer.Option(
    False, "--debug", help="Full DEBUG diagnostics and tracebacks (implies --verbose)."
)
QUIET_OPTION = typer.Option(
    False, "--quiet", "-q", help="Narration off: suppress progress and non-error diagnostics."
)
PROGRESS_OPTION = typer.Option(
    None,
    "--progress/--no-progress",
    help="Liveness on stderr. Default: auto (on when stderr is a terminal).",
)


def configure_output(
    verbose: bool = False,
    debug: bool = False,
    quiet: bool = False,
    progress: bool | None = None,
) -> OutputSettings:
    """Resolve the output axes for this invocation and install them.

    Sets the diagnostics log level (``--debug`` > ``--quiet`` > ``--verbose`` >
    default) and records the settings in the context var so the heartbeat and
    :func:`narrate` obey ``--progress`` / ``--no-progress`` / ``--quiet``. Returns
    the resolved settings. Call once at the top of a verb body, exactly where the
    old ``configure_logging`` was called.
    """
    verbose = verbose or debug  # --debug implies --verbose
    settings = OutputSettings(verbose=verbose, debug=debug, quiet=quiet, progress=progress)
    _CURRENT.set(settings)
    configure_logging(verbose=verbose, debug=debug, quiet=quiet)
    return settings


def narrate(message: str) -> None:
    """Emit a narration line (progress / status) to **stderr**, honouring ``--quiet``.

    The stream rule: status text is narration, not the artifact, so
    it must never land on stdout. Use ``typer.echo`` (stdout) only for the result.
    """
    if current_output().narrate_ok():
        typer.echo(message, err=True)
