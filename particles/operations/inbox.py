"""URL inbox processor (Option A — iCloud Drive + Mac watcher).

Operators share URLs from the iOS Share Sheet (Safari, Reddit, etc.)
into an iCloud-Drive text file via an iOS Shortcut. This module reads
that file, deposits each pending URL through the regular corpus
deposit flow, and rewrites the file so each processed line is marked
in place with the resulting ``entry_id`` (or the error). See
``docs/cli.md`` § Inbox for the Shortcut setup.

File format
-----------

One URL per line. Lines starting with ``#`` are comments / processed
markers and are skipped. Blank lines are skipped. After processing, a
line that was ``https://example.com/foo`` becomes::

    # Processed 2026-05-24T15:30:00+00:00 (entry_id: 12345678) https://example.com/foo

Failed deposits are marked similarly with ``# Failed …``. To retry a
failed URL, delete the leading ``# Failed … `` prefix and run
``particles inbox process`` again.

Design notes
------------

* **Atomic rewrite**: the inbox file is rewritten via
  :func:`particles.render.markdown.atomic_write_text`, so a phone
  append that races with a Mac rewrite produces an iCloud conflict
  file rather than a corrupted half-written inbox.
* **No locking**: running two processors against the same inbox file
  could double-process a URL. For the single-operator dev-time use
  case this is documented-not-prevented.
* **No retry-on-transient-failure**: a 5xx or network blip leaves the
  URL marked ``# Failed`` and never auto-retried. The operator can
  delete the prefix and re-run.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from particles.render.markdown import atomic_write_text

log = logging.getLogger(__name__)


def parse_inbox(text: str) -> list[tuple[int, str]]:
    """Return ``(line_index, url)`` pairs for unprocessed URLs.

    Skips blank lines and lines starting with ``#`` (comments and
    processed markers). The ``line_index`` is the 0-based position in
    the original split so the caller can rewrite that line in place.
    """
    pending: list[tuple[int, str]] = []
    for i, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending.append((i, stripped))
    return pending


def format_processed_marker(url: str, entry_id: str, *, now: datetime | None = None) -> str:
    """Render the in-file marker for a successfully-deposited URL."""
    when = (now or datetime.now(UTC)).isoformat()
    return f"# Processed {when} (entry_id: {entry_id}) {url}"


def format_failed_marker(url: str, error: str, *, now: datetime | None = None) -> str:
    """Render the in-file marker for a URL whose deposit raised."""
    when = (now or datetime.now(UTC)).isoformat()
    # Collapse newlines in the error message to keep the marker on one line.
    cleaned = error.replace("\n", " ").strip()
    return f"# Failed {when} ({cleaned}) {url}"


async def process_inbox(
    session: AsyncSession,
    inbox_path: Path,
    *,
    deposited_by: str = "inbox",
) -> dict[str, list[str]]:
    """Read ``inbox_path``, deposit every pending URL, rewrite the file.

    Returns a summary dict with three keys: ``processed`` (URLs that
    successfully deposited), ``failed`` (URLs whose deposit raised),
    and ``skipped`` (always empty in this version — reserved for
    future per-line skip directives).

    Missing inbox file is treated as empty (returns an all-empty
    summary, does not raise). The file is created on the operator's
    first phone share.
    """
    from particles.corpus.deposit import deposit_url

    summary: dict[str, list[str]] = {"processed": [], "failed": [], "skipped": []}

    if not inbox_path.exists():
        log.debug("Inbox file does not exist yet: %s", inbox_path)
        return summary

    text = inbox_path.read_text()
    pending = parse_inbox(text)
    if not pending:
        log.debug("Inbox empty — nothing to process")
        return summary

    lines = text.split("\n")

    for line_index, url in pending:
        try:
            entry_id, _snapshot_id = await deposit_url(session, url, deposited_by=deposited_by)
            await session.commit()
        except Exception as exc:  # pragma: no cover — exercised via integration
            await session.rollback()
            error = f"{type(exc).__name__}: {exc}"
            log.warning("Inbox deposit failed for %s: %s", url, error)
            lines[line_index] = format_failed_marker(url, error)
            summary["failed"].append(url)
            continue
        log.info("Inbox deposited %s → entry_id %s", url, entry_id)
        lines[line_index] = format_processed_marker(url, entry_id)
        summary["processed"].append(url)

    # Atomic write-then-rename so a concurrent phone append doesn't
    # land in the middle of a half-written file. The trailing newline
    # is preserved if the original had one.
    atomic_write_text(inbox_path, "\n".join(lines))
    return summary


def resolve_inbox_path() -> Path | None:
    """Return the configured inbox path with ``~`` expanded, or None.

    Lives in this module rather than the CLI so the FastAPI surface
    (if it ever grows an inbox endpoint) can use the same resolution.
    """
    from particles.config import get_config

    raw = get_config().inbox.file_path
    if raw is None:
        return None
    return Path(raw).expanduser()
