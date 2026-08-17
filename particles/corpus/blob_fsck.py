"""Exhaustive blob audit and re-home — the repair half of the story.

:mod:`particles.corpus.blob_health` is the cheap **detection** sibling: a
bounded sample, stat-only, run automatically by ``config validate`` and
``hook doctor``. It tells an operator they have a problem. This module is what
they run next — an operator-invoked, exhaustive audit that
resolves *every* blob-bearing snapshot, separates the recoverable strays from
the genuinely lost, and — behind an explicit flag — copies the recoverable ones
home.

Three rules keep the repair safe enough to run on a store the operator cares
about, and each is load-bearing rather than conservative-by-habit:

- **Search scope is explicit, never inferred.** The resolved ``blob_dir`` plus
  the directories named by ``--search``, and nothing else. No walking the
  filesystem for ``corpus_blobs/`` trees, no guessing at sibling worktrees, no
  reading git metadata. The content-addressed layout makes a false positive
  *silent*: any file at ``<dir>/<hh>/<hash>`` looks authoritative, so a repair
  tool that discovers its own inputs is a repair tool that can quietly adopt
  the wrong ones.
- **Copy, never move.** The source tree is left untouched, so a wrong
  ``--search`` costs disk rather than data, and a second run is a no-op.
- **Verify by digest before accepting.** A candidate whose recomputed SHA-256
  does not match the filename it was found under is rejected and reported, not
  copied. Content addressing is what makes this repair checkable rather than
  hopeful, so the check is not optional and runs in the audit too — otherwise
  "found elsewhere" would promise a recovery the copy step might not deliver.

**The database is never written.** ``fsck`` moves bytes; it does not retract
entries, rewrite ``archive_path``, or mark snapshots FAILED. The unrecoverable
remainder is *reported* with entry IDs and URIs, and the choice between
re-depositing from source and retracting stays the operator's — deciding it
automatically would mean a filesystem tool mutating epistemic state on the
strength of a ``stat`` call.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config, resolve_store_adjacent_path
from particles.core.schema import WarcRecordType
from particles.corpus.deposit import blob_path
from particles.corpus.store import CorpusEntryRow, SnapshotRow

# Read the candidate in chunks: a re-homed blob can be an arbitrarily large
# archived PDF, and the audit may digest many of them in one pass.
_DIGEST_CHUNK = 1 << 20


@dataclass(frozen=True)
class BlobRef:
    """One distinct content hash and the corpus entries that point at it.

    ``entry_ids`` / ``uris`` are what make an unrecoverable blob actionable:
    the operator needs to know *which source* to re-deposit or retract, and a
    bare hash does not say.
    """

    content_hash: str
    entry_ids: tuple[str, ...]
    uris: tuple[str, ...]

    @property
    def label(self) -> str:
        """Short operator-facing identification — first URI, else first entry id."""
        if self.uris:
            return self.uris[0].replace("file://", "")
        return self.entry_ids[0] if self.entry_ids else self.content_hash


@dataclass(frozen=True)
class StrayBlob:
    """A blob absent from ``blob_dir`` but found — and digest-verified — elsewhere."""

    ref: BlobRef
    source: Path


@dataclass(frozen=True)
class RejectedBlob:
    """A candidate at the right path whose bytes hash to something else.

    Reported rather than copied. Its hash still counts as *missing*: a file
    that fails verification has not been recovered.
    """

    ref: BlobRef
    source: Path
    actual_digest: str


@dataclass(frozen=True)
class FsckReport:
    """The audit: three disjoint classes over the store's distinct blob hashes."""

    blob_dir: Path
    search_dirs: tuple[Path, ...]
    present: tuple[BlobRef, ...]
    elsewhere: tuple[StrayBlob, ...]
    missing: tuple[BlobRef, ...]
    rejected: tuple[RejectedBlob, ...] = ()

    @property
    def total(self) -> int:
        """Distinct blob-bearing content hashes in the store."""
        return len(self.present) + len(self.elsewhere) + len(self.missing)

    @property
    def healthy(self) -> bool:
        """True when every blob the store references is already in ``blob_dir``."""
        return not self.elsewhere and not self.missing


@dataclass(frozen=True)
class RehomeOutcome:
    """What ``--re-home`` did (or, under ``--dry-run``, would have done)."""

    dry_run: bool
    copied: tuple[StrayBlob, ...] = ()
    failed: tuple[tuple[StrayBlob, str], ...] = field(default=())


def _digest_file(path: Path) -> str:
    """SHA-256 of a file's contents, read incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DIGEST_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_paths(directory: Path, content_hash: str) -> list[Path]:
    """Where a blob for ``content_hash`` could sit under an operator-named dir.

    Two exact locations, both derived from the hash — the sharded layout
    ``save_blob`` writes, and a flat ``<dir>/<hash>`` for a rescue directory
    someone flattened by hand. Deriving the paths rather than scanning is the
    point: it is one ``stat`` per candidate, and it cannot pick up a file whose
    name does not already claim to be this blob.
    """
    return [directory / content_hash[:2] / content_hash, directory / content_hash]


async def _blob_refs(session: AsyncSession) -> list[BlobRef]:
    """Every distinct content hash the store's blob-bearing snapshots reference.

    ``RESPONSE`` + a non-null ``archive_path`` is the "this snapshot owns a
    blob" predicate, matching :mod:`particles.corpus.blob_health`: REVISIT rows
    inherit content via ``refers_to`` and EPHEMERAL entries are never archived,
    so neither is a miss when its bytes are absent.
    """
    has_blob = (SnapshotRow.warc_record_type == WarcRecordType.RESPONSE.value) & (
        SnapshotRow.archive_path.is_not(None)
    )
    rows = await session.execute(
        select(SnapshotRow.content_hash, SnapshotRow.entry_id, CorpusEntryRow.uri_r)
        .outerjoin(CorpusEntryRow, CorpusEntryRow.entry_id == SnapshotRow.entry_id)
        .where(has_blob)
        .order_by(SnapshotRow.content_hash)
    )

    entries: dict[str, list[str]] = {}
    uris: dict[str, list[str]] = {}
    for content_hash, entry_id, uri_r in rows.all():
        seen_entries = entries.setdefault(content_hash, [])
        if entry_id not in seen_entries:
            seen_entries.append(entry_id)
        if uri_r:
            seen_uris = uris.setdefault(content_hash, [])
            if uri_r not in seen_uris:
                seen_uris.append(uri_r)

    return [
        BlobRef(
            content_hash=content_hash,
            entry_ids=tuple(entry_ids),
            uris=tuple(uris.get(content_hash, ())),
        )
        for content_hash, entry_ids in entries.items()
    ]


def _locate(ref: BlobRef, search_dirs: Sequence[Path]) -> tuple[Path, str] | None:
    """First digest-verified candidate for ``ref`` under ``search_dirs``.

    Returns ``(path, digest)`` for the first candidate that exists; the caller
    accepts it when ``digest == ref.content_hash`` and reports it as rejected
    otherwise. Search-dir order is the operator's — the first *existing*
    candidate wins even if it fails verification, so a corrupt file in the
    first-named directory is reported rather than silently passed over.
    """
    for directory in search_dirs:
        for candidate in _candidate_paths(directory, ref.content_hash):
            if candidate.is_file():
                return candidate, _digest_file(candidate)
    return None


async def audit_blobs(
    session: AsyncSession,
    *,
    search_dirs: Sequence[Path] = (),
) -> FsckReport:
    """Stat every blob the store references; classify present / elsewhere / missing.

    Read-only against both the database and the filesystem. Bytes are read only
    for candidates found under ``search_dirs`` — blobs already in ``blob_dir``
    are never digested, so the cost scales with the damage, not the store.

    Args:
        session: Session for the store to audit. Never written.
        search_dirs: Additional directories to look in for strays. Explicit
            operator input only; nothing is inferred or discovered.

    Returns:
        A :class:`FsckReport`. ``present`` / ``elsewhere`` / ``missing`` are
        disjoint and cover every distinct hash; ``rejected`` annotates the
        subset of ``missing`` where a wrongly-named file was found.
    """
    blob_dir = resolve_store_adjacent_path(get_config().storage.blob_dir)
    # A search dir that *is* the blob dir would report every present blob as a
    # stray of itself; dedupe on the resolved path so `--search $(blob_dir)` is
    # harmless rather than confusing.
    resolved_search: list[Path] = []
    for directory in search_dirs:
        candidate = Path(directory).expanduser().resolve()
        if candidate != blob_dir.resolve() and candidate not in resolved_search:
            resolved_search.append(candidate)

    present: list[BlobRef] = []
    elsewhere: list[StrayBlob] = []
    missing: list[BlobRef] = []
    rejected: list[RejectedBlob] = []

    for ref in sorted(await _blob_refs(session), key=lambda r: r.content_hash):
        try:
            home = blob_path(ref.content_hash)
        except ValueError:
            # A malformed hash can never be located or re-homed; it is missing
            # in the only sense that matters to the operator.
            missing.append(ref)
            continue
        if home.is_file():
            present.append(ref)
            continue
        found = _locate(ref, resolved_search)
        if found is None:
            missing.append(ref)
            continue
        path, digest = found
        if digest == ref.content_hash:
            elsewhere.append(StrayBlob(ref=ref, source=path))
        else:
            rejected.append(RejectedBlob(ref=ref, source=path, actual_digest=digest))
            missing.append(ref)

    return FsckReport(
        blob_dir=blob_dir,
        search_dirs=tuple(resolved_search),
        present=tuple(present),
        elsewhere=tuple(elsewhere),
        missing=tuple(missing),
        rejected=tuple(rejected),
    )


def rehome_strays(report: FsckReport, *, dry_run: bool = False) -> RehomeOutcome:
    """Copy each verified stray in ``report`` into the blob dir.

    Copy, never move: the source tree survives, so a wrong ``--search`` costs
    disk rather than data and a re-run finds the hashes present. Every stray
    here was already digest-verified by :func:`audit_blobs`, and the copy is
    re-verified at its destination — a truncated write must not be left
    looking like a recovery.

    Args:
        report: The audit to act on.
        dry_run: Report what would be copied without touching the filesystem.

    Returns:
        A :class:`RehomeOutcome` naming what was copied and what failed. A
        per-blob failure is collected, never raised: one unreadable stray must
        not abandon the rest of the repair.
    """
    if dry_run:
        return RehomeOutcome(dry_run=True, copied=report.elsewhere)

    copied: list[StrayBlob] = []
    failed: list[tuple[StrayBlob, str]] = []
    for stray in report.elsewhere:
        destination = blob_path(stray.ref.content_hash)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stray.source, destination)
            written = _digest_file(destination)
            if written != stray.ref.content_hash:
                destination.unlink(missing_ok=True)
                failed.append((stray, f"copy hashed to {written[:12]}…, expected match"))
                continue
        except OSError as exc:
            failed.append((stray, str(exc)))
            continue
        copied.append(stray)

    return RehomeOutcome(dry_run=False, copied=tuple(copied), failed=tuple(failed))
