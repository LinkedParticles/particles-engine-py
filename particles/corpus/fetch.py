# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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

An HTTP entry whose URI-R an importer claims (a Reddit permalink, a Hacker News
item page) is fetched by that importer's ``refetch``, not by a GET of the URI-R,
because the importer stored bytes from an API rather than the page the URI-R
names. That route has no tier 1: the importer returns a body and
tier 2 decides.

The local tier has **no network**: no DNS, no redirects, no ``particles_client``
and no SSRF guard, because that entire surface is absent rather than bypassed.
What it has instead is a root allowlist (:func:`is_refresh_allowed`): the tier
reads only files under the roots its legitimate depositors write from, because
an entry's ``file://`` URI-R is not proof the operator chose that file
(security review D2).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
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
from particles.corpus.local_read import read_local_file
from particles.corpus.rule_roots import list_rule_source_roots
from particles.corpus.rule_sources import expand_path
from particles.corpus.store import (
    CorpusEntryRow,
    SnapshotRow,
    list_refreshable_local_entries,
    list_snapshots_for_entry,
)
from particles.http import get_capped, particles_client

log = logging.getLogger(__name__)


class SourceImporter(Protocol):
    """The part of an importer the refetch ladder reads.

    Every ``ImporterPlugin`` satisfies it. The ladder takes importers as an
    argument rather than importing the registry, so ``corpus`` gains no edge
    into ``ingest``; the refresh operation passes ``get_importers()``.
    An importer that can re-fetch also defines ``refetch(uri_r) -> bytes``,
    found with ``getattr`` the way ``primary_url`` is.
    """

    def accepts_url(self, url: str) -> bool:
        """Return True if this importer handles the given URL."""
        ...


class RefetchUnsupported(RuntimeError):
    """An importer owns the entry's URI-R but cannot re-fetch it.

    Raised instead of a plain GET of the URI-R, which would store a page the
    importer's extractor cannot read.
    """


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


def _main_checkout(root: Path) -> Path | None:
    """The main checkout of a git worktree rooted at ``root``, else ``None``.

    A worktree's ``.git`` is a file reading ``gitdir: <main>/.git/worktrees/<name>``.
    A submodule's ``.git`` file points under ``.git/modules`` instead and is not
    a worktree, so it yields ``None``.
    """
    dot_git = root / ".git"
    if not dot_git.is_file():
        return None
    try:
        text = dot_git.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text[len("gitdir:") :].strip())
    gitdir = (gitdir if gitdir.is_absolute() else root / gitdir).resolve()
    if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
        return gitdir.parent.parent.parent
    return None


#: The depositors that enrol a ``LAZY`` ``file://`` entry from the engine's own
#: disk: ``rules sync`` (``rule-sync``), ``init claude-code`` (``init``), the
#: SessionEnd harvest (``claude-code-hook``) and ``particles deposit FILE``
#: (``operator``). A derived root admits only their entries (security review
#: N21): before 1.210.4 the HTTP route could enrol any engine path as ``LAZY``
#: (its default identity is ``api-user``), and nothing else tells such an entry
#: from a local one. A new local depositor of ``LAZY`` entries joins this set.
LOCAL_LAZY_DEPOSITORS = frozenset({"rule-sync", "init", "claude-code-hook", "operator"})


def rule_file_roots(recorded_roots: Sequence[Path]) -> list[Path]:
    """The roots under which the local tier admits rule files.

    ``rule_sources.paths``, the roots ``rules sync`` and ``init claude-code``
    recorded, the main checkout of each recorded worktree root (read from the
    root's own ``.git`` file), and ``~/.claude``. The refreshing process's
    working directory is never consulted, so a refresh admits the same files
    whichever directory runs it (security review N38).
    """
    roots = [expand_path(p) for p in get_config().rule_sources.paths]
    roots += [Path(r) for r in recorded_roots]
    roots += [main for r in recorded_roots if (main := _main_checkout(Path(r))) is not None]
    roots.append((Path.home() / ".claude").resolve())
    return roots


def is_refresh_allowed(
    path: Path, *, recorded_roots: Sequence[Path] = (), deposited_by: str | None = None
) -> bool:
    """Whether the local tier may read ``path`` (security review D2).

    The resolved path must sit under one of three kinds of root, each the
    reach of one legitimate depositor of a ``LAZY`` ``file://`` entry:

    * ``local_refresh.allowed_roots`` admits any file under it: the operator's
      own ``particles deposit FILE --fetch-policy LAZY`` opt-ins.
    * The rule-file roots (:func:`rule_file_roots`) admit files named in
      ``rule_sources.filenames`` (or a root that is itself a registered file):
      what ``rules sync`` and ``init claude-code`` enrol. ``recorded_roots`` is
      what those verbs recorded (:func:`list_rule_source_roots`); a caller
      with a session passes it, and none means only the configured and
      user-level roots.
    * ``~/.claude/projects`` admits ``*.md``: the Claude Code memory files the
      SessionEnd harvest enrols.

    Every comparison is on resolved paths, so ``..`` and a symlinked component
    cannot step out of a root.

    ``deposited_by``, when given, is the entry's recorded depositor. The two
    derived roots then admit it only when it is one of
    :data:`LOCAL_LAZY_DEPOSITORS`; ``local_refresh.allowed_roots`` admits any
    depositor, since the operator named that directory (security review N21).
    """
    cfg = get_config()
    # ``os.path.realpath`` rather than ``Path.resolve()``: on Python 3.11
    # ``resolve()`` also stats the result, and a refused entry is never
    # touched. Both walk the same symlinks.
    resolved = Path(os.path.realpath(path))
    if any(resolved.is_relative_to(expand_path(r)) for r in cfg.local_refresh.allowed_roots):
        return True
    if deposited_by is not None and deposited_by not in LOCAL_LAZY_DEPOSITORS:
        return False

    projects = (Path.home() / ".claude" / "projects").resolve()
    if resolved.suffix == ".md" and resolved.is_relative_to(projects):
        return True

    roots = rule_file_roots(recorded_roots)
    filenames = set(cfg.rule_sources.filenames)
    return any(
        resolved == root or (resolved.name in filenames and resolved.is_relative_to(root))
        for root in roots
    )


async def maybe_refetch(
    session: AsyncSession,
    entry_id: str,
    force: bool = False,
    *,
    importers: Sequence[SourceImporter] = (),
) -> Snapshot | None:
    """Check whether entry needs re-fetching; create a new snapshot if content changed.

    Returns the latest snapshot (new RESPONSE/REVISIT or existing) or None if
    the entry has no URI-R, has fetch_policy=NEVER, or names a local file that
    no longer exists.

    Dispatches on the URI-R scheme: a ``file://`` entry takes the
    local tier (``stat`` then hash, no network) and everything else takes the
    HTTP ladder. Within the HTTP ladder, the first of ``importers`` that accepts
    the URI-R fetches it with its ``refetch``; a URI-R no importer accepts is
    fetched with a GET.

    Raises:
        RefetchUnsupported: an importer accepts the URI-R but defines no
            ``refetch``.
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
        return await _do_remote_refresh(session, entry_row, importers, prior=None)

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
    return await _do_remote_refresh(session, entry_row, importers, prior=latest)


async def _do_remote_refresh(
    session: AsyncSession,
    entry_row: CorpusEntryRow,
    importers: Sequence[SourceImporter],
    prior: Snapshot | None,
) -> Snapshot:
    """Fetch a non-file entry through its owning importer, or by GET when none owns it."""
    uri_r = entry_row.uri_r
    assert uri_r is not None
    owner = next((imp for imp in importers if imp.accepts_url(uri_r)), None)
    if owner is None:
        return await _do_fetch(session, entry_row, prior)
    refetch: Callable[[str], Awaitable[bytes]] | None = getattr(owner, "refetch", None)
    if refetch is None:
        raise RefetchUnsupported(
            f"{uri_r} belongs to the {type(owner).__name__}, which cannot refresh an "
            f"entry, and fetching the URL directly would store a different document. "
            f"Deposit the URL again to capture a new snapshot."
        )

    # The importer fetches the way its deposit does, so an unchanged source
    # hashes to the prior snapshot's content_hash and becomes a REVISIT. Its own
    # client carries the SSRF guard and the egress policy.
    content = await refetch(uri_r)
    return await _record_body(session, entry_row.entry_id, content, prior, datetime.now(UTC))


async def _do_local_refresh(
    session: AsyncSession,
    entry_row: CorpusEntryRow,
    prior: Snapshot | None,
) -> Snapshot | None:
    """The local tier: ``stat`` mtime, then content hash. No network.

    Returns the resulting snapshot, or ``None`` when the file no longer exists,
    cannot be read, or lies outside the root allowlist (:func:`is_refresh_allowed`;
    a refused path is never opened or stat'ed). A vanished rule file is an
    *outcome*, not an error: it must not crash the nightly cycle, and
    (deliberately) it must not retire its beliefs either. "The file is gone"
    is not "the claims are false", and a stat taken mid-``git``-operation is
    not evidence of deletion. Retiring on absence is decay, not
    this ladder's job.
    """
    uri_r = entry_row.uri_r
    assert uri_r is not None

    try:
        path = path_from_file_uri(uri_r)
    except ValueError as exc:
        log.info("Entry %s: local source unavailable (%s)", entry_row.entry_id[:8], exc)
        return None
    depositor = entry_row.deposited_by
    recorded = await list_rule_source_roots(session)

    def _allowed(candidate: Path) -> bool:
        return is_refresh_allowed(candidate, recorded_roots=recorded, deposited_by=depositor)

    if not _allowed(path):
        # One warning per run, with the count, comes from the refresh loop
        # (operations/corpus_refresh.py); this line is per entry.
        log.info(
            "Entry %s: %s is outside the local refresh roots for depositor %r; not read.",
            entry_row.entry_id[:8],
            path,
            depositor,
        )
        return None

    cfg = get_config()
    try:
        follow = cfg.local_refresh.follow_symlinks
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

    # The bytes come from one descriptor, and that descriptor's own path is
    # judged again: the check above resolved the name, and a parent swapped for
    # a symlink since then would otherwise be followed here (security review N22).
    try:
        read = read_local_file(
            path,
            follow_symlinks=follow,
            max_bytes=cfg.local_files.max_bytes,
            allowed=_allowed,
        )
    except OSError as exc:
        log.info("Entry %s: local source unreadable (%s)", entry_row.entry_id[:8], exc)
        return None
    content = read.content
    mtime = datetime.fromtimestamp(read.stat.st_mtime, tz=UTC)

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
    return await _record_body(
        session,
        entry_row.entry_id,
        resp.content,
        prior,
        now,
        etag=resp.headers.get("etag"),
        last_modified=_parse_last_modified(resp.headers.get("last-modified")),
    )


async def _record_body(
    session: AsyncSession,
    entry_id: str,
    content: bytes,
    prior: Snapshot | None,
    now: datetime,
    *,
    etag: str | None = None,
    last_modified: datetime | None = None,
) -> Snapshot:
    """Tier 2 for a fetched body: REVISIT on a hash match, else a RESPONSE snapshot."""
    content_hash = sha256(content)

    if prior and content_hash == prior.content_hash:
        snap = await _write_revisit(session, entry_id, prior, now)
        log.info(
            "Entry %s: content unchanged (hash match) → REVISIT %s",
            entry_id,
            snap.snapshot_id,
        )
        return snap

    # New content: write RESPONSE snapshot
    archive_path = save_blob(content, content_hash)
    snap = Snapshot(
        captured_at=now,
        content_hash=content_hash,
        etag=etag,
        last_modified=last_modified,
        warc_record_type=WarcRecordType.RESPONSE,
        archive_path=archive_path,
        extraction_status=ExtractionStatus.PENDING,
    )
    snap_row = SnapshotRow.from_model(snap, entry_id=entry_id)
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
        entry_id,
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


def is_entry_refused(row: CorpusEntryRow, recorded_roots: Sequence[Path]) -> bool:
    """Whether the local tier refuses this entry's ``file://`` source.

    The same judgement :func:`_do_local_refresh` makes, over the entry's own
    path and depositor, so the refresh loop can count a refusal before the
    ladder runs. An entry that is not ``file://``, or whose URI-R does not
    parse, is not refused here; the ladder reports it.
    """
    if row.uri_r is None or not row.uri_r.startswith("file://"):
        return False
    try:
        path = path_from_file_uri(row.uri_r)
    except ValueError:
        return False
    return not is_refresh_allowed(
        path, recorded_roots=recorded_roots, deposited_by=row.deposited_by
    )


async def count_refused_local_entries(session: AsyncSession) -> int:
    """How many ``LAZY`` ``file://`` entries the local tier would refuse to read.

    ``particles rules`` and ``hook doctor`` report it, so the window between an
    upgrade and the first ``rules sync`` is never silent.
    """
    recorded = await list_rule_source_roots(session)
    refused = 0
    for entry_id, _uri_r in await list_refreshable_local_entries(session):
        row = await session.get(CorpusEntryRow, entry_id)
        if row is not None and is_entry_refused(row, recorded):
            refused += 1
    return refused
