"""§7.5 Lazy re-fetch protocol for LAZY corpus entries.

Three-tier change detection, over two transports (local tier):

  Tier 1 — Source signal, no body:
             HTTP  — ETag / If-Modified-Since → 304 → REVISIT snapshot
             local — ``stat`` mtime vs the prior snapshot's ``last_modified``
                     → equal → no I/O, no row
  Tier 2 — Content hash: read + SHA-256 compare → REVISIT if unchanged,
           RESPONSE if changed
  Tier 3 — Manual override (``force=True``; ``deposit_url`` / ``corpus refresh``)

:func:`maybe_refetch` dispatches on the URI-R scheme so both transports share
one ladder — the policy gate, the per-source-type floor, the force override and
the RESPONSE/REVISIT distinction have a single home, and the future
backoff and decay have one cadence to tune rather than two.

The local tier has **no network**: no DNS, no redirects, no ``particles_client``
and no SSRF guard, because that entire surface is absent rather than bypassed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    ExtractionStatus,
    FetchPolicy,
    Snapshot,
    WarcRecordType,
)
from particles.corpus.deposit import save_blob, sha256
from particles.corpus.store import (
    CorpusEntryRow,
    SnapshotRow,
    list_snapshots_for_entry,
)
from particles.http import get_capped, particles_client

log = logging.getLogger(__name__)


def _floor_seconds(source_type: str) -> int:
    """The per-source-type re-fetch floor, in seconds (``corpus.refetch_floors``).

    Read from config at call time per the AGENTS.md configuration rule. This
    function used to consult a private dict that shadowed the config section —
    a discrepancy nothing noticed because, until, no production code
    called into this module at all.
    """
    return get_config().refetch_floors.get(source_type, 3600)


def path_from_file_uri(uri_r: str) -> Path:
    """The local path a ``file://`` URI-R names.

    ``deposit_file`` records ``path.resolve().as_uri()``, so the round trip is
    exact for anything this SDK deposited. Percent-escapes (spaces, non-ASCII)
    are decoded; a ``file://host/...`` authority other than ``localhost`` is
    rejected rather than silently reinterpreted as a local path.
    """
    parsed = urlparse(uri_r)
    if parsed.netloc not in ("", "localhost"):
        raise ValueError(f"Refusing non-local file URI with authority {parsed.netloc!r}: {uri_r}")
    return Path(unquote(parsed.path))


async def maybe_refetch(
    session: AsyncSession,
    entry_id: str,
    force: bool = False,
) -> Snapshot | None:
    """Check whether entry needs re-fetching; create a new snapshot if content changed.

    Returns the latest snapshot (new RESPONSE/REVISIT or existing) or None if
    the entry has no URI-R, has fetch_policy=NEVER, or names a local file that
    no longer exists.

    Dispatches on the URI-R scheme: a ``file://`` entry takes the
    local tier — ``stat`` then hash, no network — and everything else takes the
    HTTP ladder.
    """
    entry_row = await session.get(CorpusEntryRow, entry_id)
    if entry_row is None:
        raise ValueError(f"Entry {entry_id} not found")

    if entry_row.fetch_policy != FetchPolicy.LAZY.value and not force:
        return None  # nothing to do
    if entry_row.uri_r is None:
        return None

    is_local = entry_row.uri_r.startswith("file://")

    snapshots = await list_snapshots_for_entry(session, entry_id)
    if not snapshots:
        # No prior snapshot — capture a fresh one.
        if is_local:
            return await _do_local_refresh(session, entry_row, prior=None)
        return await _do_fetch(session, entry_row, prior=None)

    latest = max(snapshots, key=lambda s: s.captured_at)

    # Floor check
    floor = _floor_seconds(entry_row.source_type)
    now = datetime.now(UTC)
    age_seconds = (
        now - latest.captured_at.replace(tzinfo=UTC)
        if latest.captured_at.tzinfo is None
        else (now - latest.captured_at)
    ).total_seconds()
    if not force and age_seconds < floor:
        log.debug(
            "Entry %s last fetched %.0fs ago (floor %ds); skipping",
            entry_id,
            age_seconds,
            floor,
        )
        return latest

    if is_local:
        return await _do_local_refresh(session, entry_row, prior=latest)
    return await _do_fetch(session, entry_row, prior=latest)


async def _do_local_refresh(
    session: AsyncSession,
    entry_row: CorpusEntryRow,
    prior: Snapshot | None,
) -> Snapshot | None:
    """The local tier: ``stat`` mtime, then content hash. No network.

    Returns the resulting snapshot, or ``None`` when the file no longer exists
    or cannot be read. A vanished rule file is an *outcome*, not an error: it
    must not crash the nightly cycle, and — deliberately — it must not retire
    its beliefs either. "The file is gone" is not "the claims are false", and a
    stat taken mid-``git``-operation is not evidence of deletion. Retiring on
    absence is decay, not this ladder's job.
    """
    uri_r = entry_row.uri_r
    assert uri_r is not None

    try:
        path = path_from_file_uri(uri_r)
        follow = get_config().local_refresh.follow_symlinks
        if not follow and path.is_symlink():
            raise ValueError(f"symlinked source and local_refresh.follow_symlinks is off: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
    except (OSError, ValueError) as exc:
        log.info("Entry %s: local source unavailable (%s)", entry_row.entry_id[:8], exc)
        return None

    mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

    # Tier 1 — source signal, no read. `last_modified` already means "when the
    # source last changed"; a local mtime is that same quantity over a different
    # transport, which is why this reuses the column rather than adding one.
    #
    # The comparison is INEQUALITY, not recency: a restore from backup or an
    # `rsync -t` can move an mtime backwards while changing content, so "differs"
    # — in either direction — means re-read. Stated plainly: an unchanged mtime
    # is a heuristic, not proof. Where a 304 is the server's assertion, this is
    # an inference; `force=True` is the escape hatch.
    if (
        prior is not None
        and prior.last_modified is not None
        and _as_utc(prior.last_modified) == mtime
    ):
        log.debug("Entry %s: mtime unchanged; no read", entry_row.entry_id[:8])
        return prior

    try:
        content = path.read_bytes()
    except OSError as exc:
        log.info("Entry %s: local source unreadable (%s)", entry_row.entry_id[:8], exc)
        return None

    now = datetime.now(UTC)
    content_hash = sha256(content)

    # Tier 2 — content hash, authoritative. SHA-256 is the canonical snapshot
    # identity (techspec §7.2/§7.3: the blob-store key and the cross-operator
    # shared-archive identity), so change detection reuses it rather than
    # introducing a second, non-cryptographic hash for the
    # measurement behind that call.
    if prior is not None and content_hash == prior.content_hash:
        snap = await _write_revisit(session, entry_row.entry_id, prior, now, last_modified=mtime)
        log.info(
            "Entry %s: local content unchanged (hash match) → REVISIT %s",
            entry_row.entry_id[:8],
            snap.snapshot_id[:8],
        )
        return snap

    archive_path = save_blob(content, content_hash)
    snap = Snapshot(
        captured_at=now,
        content_hash=content_hash,
        last_modified=mtime,
        warc_record_type=WarcRecordType.RESPONSE,
        archive_path=archive_path,
        extraction_status=ExtractionStatus.PENDING,
    )
    session.add(SnapshotRow.from_model(snap, entry_id=entry_row.entry_id))
    await session.flush()

    # No staleness cascade here: the MUTABLE generation cascade runs
    # AFTER the new snapshot is extracted (particles/ingest/generation.py), so
    # carry-forward — which looks up ACTIVE particles — still sees the
    # prior generation and can keep unchanged chunks rather than re-paying for
    # them and minting duplicates.
    log.info(
        "Entry %s: local content changed → RESPONSE snapshot %s (PENDING)",
        entry_row.entry_id[:8],
        snap.snapshot_id[:8],
    )
    return snap


def _as_utc(dt: datetime) -> datetime:
    """Naive datetimes out of SQLite are UTC by construction; stamp them so."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


async def _do_fetch(
    session: AsyncSession,
    entry_row: CorpusEntryRow,
    prior: Snapshot | None,
) -> Snapshot:
    """Execute the actual HTTP fetch and return the resulting snapshot."""
    uri_r = entry_row.uri_r
    assert uri_r is not None

    # Re-validate on every refetch — DNS can move between deposit and refetch,
    # so the URL that was safe months ago may now resolve to something internal.
    from particles.url_safety import validate_fetch_url

    validate_fetch_url(uri_r)

    extra_headers: dict[str, str] = {}
    if prior:
        if prior.etag:
            extra_headers["If-None-Match"] = prior.etag
        if prior.last_modified:
            import calendar
            from email.utils import formatdate

            extra_headers["If-Modified-Since"] = formatdate(
                calendar.timegm(prior.last_modified.timetuple()), usegmt=True
            )

    async with particles_client(extra_headers=extra_headers) as client:
        resp = await get_capped(client, uri_r)

    now = datetime.now(UTC)

    # Tier 1: 304 Not Modified → REVISIT
    if resp.status_code == 304 and prior:
        snap = await _write_revisit(session, entry_row.entry_id, prior, now)
        log.info("Entry %s: 304 Not Modified → REVISIT %s", entry_row.entry_id, snap.snapshot_id)
        return snap

    resp.raise_for_status()
    content = resp.content
    content_hash = sha256(content)

    # Tier 2: compare content hashes
    if prior and content_hash == prior.content_hash:
        snap = await _write_revisit(session, entry_row.entry_id, prior, now)
        log.info(
            "Entry %s: content unchanged (hash match) → REVISIT %s",
            entry_row.entry_id,
            snap.snapshot_id,
        )
        return snap

    # New content: write RESPONSE snapshot
    archive_path = save_blob(content, content_hash)
    snap = Snapshot(
        captured_at=now,
        content_hash=content_hash,
        etag=resp.headers.get("etag"),
        last_modified=_parse_last_modified(resp.headers.get("last-modified")),
        warc_record_type=WarcRecordType.RESPONSE,
        archive_path=archive_path,
        extraction_status=ExtractionStatus.PENDING,
    )
    snap_row = SnapshotRow.from_model(snap, entry_id=entry_row.entry_id)
    session.add(snap_row)
    await session.flush()

    # The MUTABLE staleness cascade used to run here, BEFORE the new snapshot was
    # extracted. This was moved after extraction
    # (particles/ingest/generation.py): demoting first blinds the
    # carry-forward — which looks up ACTIVE particles — so it re-paid the LLM for
    # every unchanged paragraph and minted duplicates of claims that never
    # changed. The bug was latent only because this module had no caller.
    log.info(
        "Entry %s: content changed → RESPONSE snapshot %s",
        entry_row.entry_id,
        snap.snapshot_id,
    )
    return snap


async def _write_revisit(
    session: AsyncSession,
    entry_id: str,
    prior: Snapshot,
    captured_at: datetime,
    last_modified: datetime | None = None,
) -> Snapshot:
    """A zero-byte REVISIT recording "as of ``captured_at``, still snapshot ``prior``".

    ``last_modified`` stamps the source signal observed alongside the match (the
    local tier's mtime), so the *next* run's tier 1 can short-circuit without a
    read. Without it a file whose mtime moved but whose bytes did not would be
    re-read every night forever.
    """
    snap = Snapshot(
        captured_at=captured_at,
        content_hash=prior.content_hash,
        last_modified=last_modified if last_modified is not None else prior.last_modified,
        warc_record_type=WarcRecordType.REVISIT,
        archive_path=None,
        refers_to=prior.snapshot_id,
        extraction_status=ExtractionStatus.COMPLETE,  # no new extraction needed
    )
    session.add(SnapshotRow.from_model(snap, entry_id=entry_id))
    await session.flush()
    return snap


def _parse_last_modified(value: str | None) -> datetime | None:
    if not value:
        return None
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(value)
    except Exception:
        return None
