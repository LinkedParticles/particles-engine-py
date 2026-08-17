"""Terminal liveness heartbeat for long-running CLI verbs (the ``run()`` seam).

A verb that goes quiet for minutes is indistinguishable from a hung one — the
failure mode ``particles memory rebuild-utility`` hit in the field (27 minutes of
silence between log lines). Rather than teach each verb to print progress — the
per-verb whack-a-mole this replaces — the heartbeat rides the one seam every
async CLI body already passes through: :func:`particles.api.cli.run`.

Three deliberate choices:

- **A daemon thread, not an asyncio task.** A step that blocks without awaiting
  (a long synchronous stretch — exactly what a re-mine does) would starve an
  event-loop ticker precisely when liveness matters most.
- **stderr, erased on completion.** stdout stays a clean artifact, so pipes and
  ``--output-format json`` are unaffected and the heartbeat never lands in a
  captured transcript.
- **Off unless interactive.** Suppressed when stderr is not a TTY (CI, pipes) and
  when ``cli.heartbeat_seconds`` is 0, so non-interactive output stays
  byte-identical to before.

Detailed per-item progress (``done/total``) remains an opt-in concern of the
operation that knows its denominator; this module is only the *floor* that
guarantees no verb can silently look hung.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from particles.config import get_config

#: Carriage return + ANSI erase-to-end-of-line.
_CLEAR_LINE = "\r\x1b[K"

#: Compact per-item status published by the running verb for the heartbeat to
#: render (``None`` ⇒ the generic "working"). A plain module global: str-ref
#: assignment is atomic under the GIL, and the ticker thread is the only
#: reader, so no lock is needed.
_status: str | None = None


def set_heartbeat_status(status: str | None) -> None:
    """Publish a compact per-item status for the heartbeat line.

    The heartbeat's time-only ``working (28m20s elapsed)`` proves the verb is
    alive but says nothing about how far along it is — the gap the first real
    ``reindex`` run surfaced. A verb that knows its denominator publishes
    ``snapshot 12/89 (entry 0a8fb1a9…)`` here and the next tick renders it in
    place of ``working``, elapsed time intact.

    Stored, not printed: the ticker stays the only writer of the in-place
    line, so there is nothing to interleave, and when no heartbeat is running
    (non-TTY, ``--quiet``) the value is simply never read. The heartbeat
    resets it on entry and exit so one verb's last status can never bleed
    into the next invocation in the same process.
    """
    global _status
    _status = status


def format_elapsed(seconds: float) -> str:
    """Compact elapsed rendering: ``42s`` / ``1m35s`` / ``2h05m``."""
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def progress_line(message: str) -> None:
    """Write one per-item progress line to stderr, safely alongside the heartbeat.

    The heartbeat deliberately leaves the cursor mid-line (no trailing newline,
    so the next tick can overwrite it with ``\r``). A verb that then writes its
    own progress with a plain ``echo`` lands *on that line*, producing output
    like::

        … structure: working (5m00s elapsed)[140/500] d882de5a… Contribute

    Clearing first is what makes the two writers composable — which the module
    docstring already assumes ("detailed per-item progress remains an opt-in
    concern of the operation") but nothing enforced. Every verb that passes a
    ``progress=`` callback should route it through here.

    Gating is deliberately *not* the heartbeat's: a verb only passes a progress
    callback when the operator asked for one (``--verbose``), and an explicit
    request must still be honoured when stderr is piped — that is how a long run
    gets captured to a log. Only ``--quiet`` silences it.

    What *is* TTY-conditional is the clear sequence: there is no heartbeat to
    erase off a TTY, and writing ``\r\x1b[K`` into a captured log would put
    escape junk in the file. So a piped run stays byte-identical to before this
    helper existed.
    """
    from particles.api.cli._output import current_output

    settings = current_output()
    if not settings.narrate_ok():
        return
    prefix = _CLEAR_LINE if settings.show_progress() else ""
    sys.stderr.write(f"{prefix}{message}\n")
    sys.stderr.flush()


@contextmanager
def heartbeat(label: str) -> Iterator[None]:
    """Emit a periodic liveness line on stderr while the body runs.

    A no-op when the resolved output settings say progress is off (``--no-progress``,
    ``--quiet``, or the auto default of "stderr is not a TTY") or when
    ``cli.heartbeat_seconds`` is 0, so the heartbeat can never corrupt piped or
    machine-read output. The show/hide decision is made here, in the
    caller's thread, before the ticker spawns — the ticker itself reads no context var.
    """
    from particles.api.cli._output import current_output

    interval = get_config().cli.heartbeat_seconds
    if interval <= 0 or not current_output().show_progress():
        yield
        return

    set_heartbeat_status(None)
    done = threading.Event()
    started = time.monotonic()

    def _tick() -> None:
        while not done.wait(interval):
            elapsed = format_elapsed(time.monotonic() - started)
            detail = _status or "working"
            sys.stderr.write(f"{_CLEAR_LINE}  … {label}: {detail} ({elapsed} elapsed)")
            sys.stderr.flush()

    ticker = threading.Thread(target=_tick, name="particles-heartbeat", daemon=True)
    ticker.start()
    try:
        yield
    finally:
        done.set()
        # Join before erasing: otherwise a tick already in flight can repaint the
        # line *after* the clear and strand it in the operator's scrollback.
        ticker.join(timeout=1.0)
        set_heartbeat_status(None)
        sys.stderr.write(_CLEAR_LINE)
        sys.stderr.flush()
