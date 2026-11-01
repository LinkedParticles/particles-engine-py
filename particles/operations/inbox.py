# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
* **Appends survive the pass**: the rewrite re-reads the file and marks
  the processed lines in what it finds, so a URL shared while the pass
  was depositing stays pending for the next pass rather than being
  overwritten (security review 2026-10-28-1, O3).
* **A failed re-read skips the rewrite**: when the file cannot be read
  again after the pass, the markers are left for the next pass rather than
  written over whatever was shared meanwhile (security review 2026-10-28-2,
  AA4). The next pass deposits those URLs again.
* **Read as a bounded regular file, off the event loop**: the inbox is opened
  without following a link and without blocking, anything but a regular file
  is refused, the read is capped at ``inbox.max_bytes``, and the pass reads in
  a worker thread, so a FIFO or a huge file at the path cannot stall the
  engine (security review 2026-10-28-2, AA7).
* **Writes under the store's writer lock**: each deposit's write units
  commit under the lock, which is never held across a fetch (security
  review 2026-10-28-1, O7).
* **No locking**: running two processors against the same inbox file
  could double-process a URL. For the single-operator dev-time use
  case this is documented-not-prevented.
* **No retry-on-transient-failure**: a 5xx or network blip leaves the
  URL marked ``# Failed`` and never auto-retried. The operator can
  delete the prefix and re-run.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from particles.db import write_transaction
from particles.egress import EgressRefused
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


class InboxUnreadable(OSError):
    """The inbox path holds something a pass refuses to read: not a regular file, or too large."""


def read_inbox(inbox_path: Path) -> str:
    """Read the inbox as a bounded regular file, never following a link or blocking on open.

    Raises :class:`FileNotFoundError` when the file is absent and
    :class:`InboxUnreadable` (an :class:`OSError`) for a link, a FIFO, a device,
    a file over ``inbox.max_bytes`` or one that is not UTF-8 text (security
    review 2026-10-28-2, AA7).
    """
    from particles.config import get_config

    max_bytes = get_config().inbox.max_bytes
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(inbox_path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise InboxUnreadable(f"inbox path is a link: {inbox_path}") from exc
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise InboxUnreadable(f"inbox path is not a regular file: {inbox_path}")
        if st.st_size > max_bytes:
            raise InboxUnreadable(
                f"inbox file is {st.st_size} bytes, over inbox.max_bytes ({max_bytes})"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
    finally:
        os.close(fd)
    if len(data) > max_bytes:
        raise InboxUnreadable(f"inbox file grew over inbox.max_bytes ({max_bytes})")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InboxUnreadable(f"inbox file is not UTF-8 text: {inbox_path}") from exc


def inbox_has_pending(inbox_path: Path) -> bool:
    """Whether the inbox holds a URL no pass has marked, such as one shared during a pass.

    The watchers gate on the file's mtime after their own rewrite, so a line
    carried over by that rewrite would otherwise wait for the next share
    (security review 2026-10-28-1, O21).
    """
    try:
        return bool(parse_inbox(read_inbox(inbox_path)))
    except OSError:
        return False


def format_failed_marker(url: str, error: str, *, now: datetime | None = None) -> str:
    """Render the in-file marker for a URL whose deposit raised."""
    when = (now or datetime.now(UTC)).isoformat()
    # Collapse newlines in the error message to keep the marker on one line.
    cleaned = error.replace("\n", " ").strip()
    return f"# Failed {when} ({cleaned}) {url}"


def format_refused_marker(url: str, *, now: datetime | None = None) -> str:
    """Render the in-file marker for a URL the no-egress posture refused.

    Distinct from ``# Failed`` so an operator can tell a policy from an outage.
    """
    when = (now or datetime.now(UTC)).isoformat()
    return f"# Refused (network.egress) {when} {url}"


#: The depositor the URL inbox records on its entries. Reserved as a key
#: principal, since an agent key under it would own every inbox entry
#: (security review 2026-10-27-1, G42).
INBOX_DEPOSITOR = "inbox"


async def process_inbox(
    session: AsyncSession,
    inbox_path: Path,
    *,
    deposited_by: str = INBOX_DEPOSITOR,
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
    from particles.corpus.deposit import deposit_url, writes_committed_under_lock

    summary: dict[str, list[str]] = {
        "processed": [],
        "failed": [],
        "refused": [],
        "skipped": [],
    }

    try:
        text = await asyncio.to_thread(read_inbox, inbox_path)
    except FileNotFoundError:
        log.debug("Inbox file does not exist yet: %s", inbox_path)
        return summary
    except InboxUnreadable as exc:
        log.warning("Inbox not read: %s", exc)
        return summary
    pending = parse_inbox(text)
    if not pending:
        log.debug("Inbox empty — nothing to process")
        return summary

    markers: dict[int, tuple[str, str]] = {}

    for line_index, url in pending:
        try:
            # Each write unit commits under the store's writer lock, and the
            # fetch runs outside it; the closing block commits, under the lock,
            # any write a unit did not cover (security review 2026-10-28-1, O7).
            with writes_committed_under_lock():
                entry_id, _snapshot_id = await deposit_url(session, url, deposited_by=deposited_by)
            async with write_transaction(session):
                pass
        except EgressRefused:
            await session.rollback()
            log.info("Inbox deposit refused by network.egress: %s", url)
            markers[line_index] = (url, format_refused_marker(url))
            summary["refused"].append(url)
            continue
        except Exception as exc:  # pragma: no cover — exercised via integration
            await session.rollback()
            error = f"{type(exc).__name__}: {exc}"
            log.warning("Inbox deposit failed for %s: %s", url, error)
            markers[line_index] = (url, format_failed_marker(url, error))
            summary["failed"].append(url)
            continue
        log.info("Inbox deposited %s → entry_id %s", url, entry_id)
        markers[line_index] = (url, format_processed_marker(url, entry_id))
        summary["processed"].append(url)

    # Atomic write-then-rename so a concurrent phone append doesn't
    # land in the middle of a half-written file. The trailing newline
    # is preserved if the original had one. A file that cannot be read
    # again is left as it is, so nothing shared during the pass is lost
    # (security review 2026-10-28-2, AA4).
    current = await _read_current(inbox_path)
    if current is not None:
        atomic_write_text(inbox_path, apply_markers(text, current, markers))
    return summary


async def _read_current(inbox_path: Path) -> str | None:
    """The inbox as it is now, for the rewrite; None when it cannot be read."""
    try:
        return await asyncio.to_thread(read_inbox, inbox_path)
    except OSError as exc:
        log.warning(
            "Inbox re-read failed (%s); leaving the file as it is, so the next pass "
            "deposits this pass's URLs again",
            exc,
        )
        return None


def apply_markers(first_read: str, current: str, markers: dict[int, tuple[str, str]]) -> str:
    """Mark the processed lines in ``current``, the file as it is at rewrite time.

    ``markers`` maps a line index of ``first_read`` to ``(url, marker)``. When
    ``current`` only extends ``first_read`` (a share appended during the pass)
    the markers replace their lines and the appended text is kept as it is. When
    the file was otherwise edited, each marker replaces the first unmarked line
    still holding its URL, and a URL no longer in the file is left unmarked.
    Either way no line the pass did not process is dropped (security review
    2026-10-28-1, O3).
    """
    if current.startswith(first_read):
        lines = first_read.split("\n")
        for index, (_url, marker) in markers.items():
            lines[index] = marker
        tail = current[len(first_read) :]
        if tail and not first_read.endswith("\n") and not tail.startswith("\n"):
            # The processed last line was complete when read; what follows it
            # is a new line, never part of the marker.
            tail = "\n" + tail
        return "\n".join(lines) + tail
    fresh = current.split("\n")
    marked: set[int] = set()
    for index, (url, marker) in sorted(markers.items()):
        candidates = [index] if index < len(fresh) else []
        candidates += range(len(fresh))
        for at in candidates:
            if at not in marked and fresh[at].strip() == url:
                fresh[at] = marker
                marked.add(at)
                break
    return "\n".join(fresh)


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
