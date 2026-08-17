"""inbox sub-Typer — process URLs queued from iOS via iCloud Drive.

See ``docs/cli.md`` § Inbox for the iOS-Shortcut setup. The bot-of-the-
operator workflow:

  1. iPhone Safari/Reddit Share → run the operator's Shortcut →
     URL appended to a file in iCloud Drive
  2. Mac picks up the file change via ``particles inbox process``
     (one-shot) or ``particles inbox watch`` (continuous loop)
  3. Each pending URL flows through the regular deposit pipeline
     (importer registry → fall back to generic HTTP fetch); the
     line in the inbox is rewritten in place with the resulting
     ``entry_id`` (or the error)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import typer

from particles.api.cli import app, run
from particles.db import session_scope

inbox_app = typer.Typer(
    help="Process URLs queued from an iOS Shortcut via iCloud Drive.",
    no_args_is_help=True,
)
app.add_typer(inbox_app, name="inbox")


log = logging.getLogger(__name__)


def _resolved_inbox_or_exit() -> Any:
    """Resolve the configured inbox path or exit with a guiding error."""
    from particles.operations.inbox import resolve_inbox_path

    path = resolve_inbox_path()
    if path is None:
        typer.echo(
            "No `inbox.file_path` configured. Set it in config.yaml (see "
            "docs/cli.md § Inbox) or via the `INBOX_FILE_PATH` env var.",
            err=True,
        )
        raise typer.Exit(2)
    return path


def _print_summary(summary: dict[str, list[str]]) -> None:
    """Format ``process_inbox`` summary for the operator."""
    total = len(summary["processed"]) + len(summary["failed"])
    if total == 0:
        typer.echo("Inbox empty — nothing to process.")
        return
    typer.echo(
        f"Processed {len(summary['processed'])} / {total} URL(s) ({len(summary['failed'])} failed)."
    )
    for url in summary["processed"]:
        typer.echo(f"  + {url}")
    for url in summary["failed"]:
        typer.echo(f"  x {url}")


@inbox_app.command("process")
def inbox_process_cmd(
    deposited_by: str = typer.Option("inbox", help="Recorded as the depositor on each entry."),
) -> None:
    """Process all pending URLs in the inbox file, then exit.

    Suitable for cron / launchd / a desktop keyboard shortcut. Each
    pending URL is deposited and the inbox line is rewritten in place
    with the resulting ``entry_id``.
    """
    path = _resolved_inbox_or_exit()
    summary = run(_process_once(path, deposited_by))
    _print_summary(summary)


@inbox_app.command("watch")
def inbox_watch_cmd(
    interval: int | None = typer.Option(
        None,
        "--interval",
        help="Seconds between polls. Defaults to inbox.poll_interval_seconds (30).",
    ),
    deposited_by: str = typer.Option("inbox", help="Recorded as the depositor on each entry."),
) -> None:
    """Continuously poll the inbox file. Ctrl-C to stop.

    Uses mtime to skip the file read when nothing has changed since
    the last poll — cheap enough to leave running in a terminal tab.
    """
    from particles.config import get_config

    path = _resolved_inbox_or_exit()
    effective_interval = (
        interval if interval is not None else get_config().inbox.poll_interval_seconds
    )
    typer.echo(f"Watching {path} every {effective_interval}s. Ctrl-C to stop.")
    run(_watch_loop(path, deposited_by, effective_interval))


@inbox_app.command("status")
def inbox_status_cmd() -> None:
    """Show pending vs processed counts in the inbox file."""
    path = _resolved_inbox_or_exit()
    typer.echo(f"Inbox: {path}")
    if not path.exists():
        typer.echo("  (file does not exist yet — will be created on first iOS share)")
        # Common config-side foot-gun: operators copy-paste the path
        # from a shell command where spaces were escaped as `\ ` and
        # then drop it into config.yaml verbatim. YAML treats the
        # backslash as a literal character, so the resolved path looks
        # right to a human but never matches what's actually on disk.
        if "\\ " in str(path):
            typer.echo(
                "  hint: the resolved path contains '\\ ' (backslash-space). "
                "That's shell escaping; YAML doesn't need it. Remove the "
                "backslashes from `inbox.file_path` in config.yaml, or "
                'wrap the whole path in quotes ("…").',
                err=True,
            )
        elif not path.parent.exists():
            typer.echo(
                f"  hint: the parent directory does not exist either: "
                f"{path.parent}. Check the configured path is correct.",
                err=True,
            )
        return

    from particles.operations.inbox import parse_inbox

    text = path.read_text()
    pending = parse_inbox(text)
    total_lines = sum(1 for line in text.split("\n") if line.strip())
    processed = total_lines - len(pending)
    typer.echo(f"  Pending:   {len(pending)}")
    typer.echo(f"  Processed: {processed}")
    if pending:
        typer.echo("  Next URLs to process:")
        for _idx, url in pending[:5]:
            typer.echo(f"    - {url}")
        if len(pending) > 5:
            typer.echo(f"    … and {len(pending) - 5} more")


# ---------------------------------------------------------------------------
# async helpers
# ---------------------------------------------------------------------------


async def _process_once(path: Any, deposited_by: str) -> dict[str, list[str]]:
    from particles.operations.inbox import process_inbox

    async with session_scope() as session:
        return await process_inbox(session, path, deposited_by=deposited_by)


async def _watch_loop(path: Any, deposited_by: str, interval: int) -> None:
    from particles.operations.inbox import process_inbox

    last_mtime: float | None = None
    while True:
        try:
            current_mtime = path.stat().st_mtime if path.exists() else None
        except OSError as exc:
            log.warning("inbox stat failed: %s", exc)
            current_mtime = None

        if current_mtime is not None and current_mtime != last_mtime:
            async with session_scope() as session:
                summary = await process_inbox(session, path, deposited_by=deposited_by)
            _print_summary(summary)
            # Re-stat after rewriting so the rewrite's own mtime change
            # doesn't trigger an immediate second pass.
            if path.exists():
                last_mtime = path.stat().st_mtime
        # asyncio.sleep so Ctrl-C propagates cleanly between iterations.
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break


# Wall-clock helper exposed for tests; unused in production code path
# but kept as a module-local so monkeypatching is straightforward.
_now = time.time
