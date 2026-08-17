"""Blob-reachability probe — is the store's content where this process looks?

*New* blob sharding was stopped by anchoring a relative ``storage.blob_dir``
to the store's own directory. It does not detect a store whose blobs were
written elsewhere before that fix: the DB rows survive, pointing at content the
resolved blob dir does not hold, and the damage only surfaces much later as
``Blob not found for hash …`` in the middle of an extraction run (the 2026-07-18
dogfood incident: 313 / 24 / 11 / 8 / 4 / 2 blobs across five directories, 17
extraction failures and ~48 skipped entries in one consolidation run).

This module is the cheap **detection** half of that recovery story:
count the store's blob-bearing snapshots, stat a bounded sample of them under
the *currently resolved* blob dir, and report the miss rate. It is a diagnostic
— it never raises on a miss and never writes. A legitimately empty first-run
store reports healthy (there is nothing to find), so a warning here always means
real rows point at absent content.

The probe deliberately reads the **globally resolved** blob dir rather than one
derived per store handle: `blob_path` resolves `storage.blob_dir` once for the
whole process, so every store's blobs land in the same tree. Probing anything
else would report a directory extraction will never consult.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config, resolve_store_adjacent_path, sqlite_file_path
from particles.core.schema import WarcRecordType
from particles.corpus.deposit import blob_exists
from particles.corpus.store import SnapshotRow


def store_file_missing(database_url: str) -> bool:
    """True when ``database_url`` names a SQLite file that does not exist yet.

    Opening a session against a missing SQLite path *creates* an empty database
    as a side effect. A caller that probes an uninitialized store would
    therefore leave a stray `particles.db` in whatever directory the operator
    happened to run from — the exact cwd-relative litter this whole area exists
    to stop. Check before opening, not after failing.

    Non-SQLite and in-memory URLs report ``False``: there is no file to miss,
    and connecting is the only way to find out whether they are reachable.
    """
    path = sqlite_file_path(database_url)
    return path is not None and not Path(path).expanduser().exists()


@dataclass(frozen=True)
class BlobReachability:
    """Result of probing a store's snapshots against the resolved blob dir.

    Attributes:
        blob_dir: The blob directory this process resolves.
        dir_exists: Whether that directory exists on disk at all.
        snapshots: Count of blob-bearing (RESPONSE) snapshots in the store.
        sampled: How many distinct content hashes were probed.
        missing: How many of the sampled hashes have no blob on disk.
    """

    blob_dir: Path
    dir_exists: bool
    snapshots: int
    sampled: int
    missing: int

    @property
    def healthy(self) -> bool:
        """True when nothing sampled was missing (including the nothing-to-check case)."""
        return self.missing == 0

    @property
    def total_loss(self) -> bool:
        """True when every probed hash was absent — the scattering signature."""
        return self.sampled > 0 and self.missing == self.sampled

    def warning_lines(self) -> list[str]:
        """Operator-facing warning, or an empty list when healthy.

        Shared by every surface that reports this check so the wording — and the
        remediation pointer — has one home rather than one per call site.
        """
        if self.healthy:
            return []
        scope = "none of" if self.total_loss else f"{self.missing} of"
        lines = [
            f"WARNING: {scope} the {self.sampled} sampled blob(s) were found under "
            f"the resolved blob directory.",
            f"  blob_dir:  {self.blob_dir}"
            f"{'' if self.dir_exists else '  (directory does not exist)'}",
            f"  store:     {self.snapshots} snapshot(s) reference stored content.",
        ]
        if self.total_loss:
            lines.append(
                "  This is the signature of blobs written under a different working "
                "directory. The content is likely still on disk in another "
                "`corpus_blobs/` tree — extraction of these snapshots will fail with "
                "`Blob not found for hash …` until they are moved beside the store."
            )
        else:
            lines.append(
                "  Some content is unreachable; extraction of those snapshots will fail "
                "with `Blob not found for hash …`."
            )
        lines.append(
            "  Set `storage.blob_dir` to an absolute path, or move the missing blobs "
            "into the directory above."
        )
        return lines


async def check_blob_reachability(
    session: AsyncSession,
    *,
    sample: int | None = None,
) -> BlobReachability:
    """Probe a bounded sample of the store's snapshots against the blob dir.

    Read-only and cheap: one COUNT, one indexed SELECT, and at most ``sample``
    ``stat`` calls. Never raises on a missing blob — the caller decides how loud
    to be.

    Args:
        session: Session for the store to probe. Any handle works; the blob tree
            is process-global (see the module docstring).
        sample: Maximum number of distinct content hashes to stat. Defaults to
            ``storage.blob_health_sample``.

    Returns:
        A :class:`BlobReachability` report. A store with no blob-bearing
        snapshots reports ``sampled=0``, which is ``healthy``.
    """
    if sample is None:
        sample = get_config().storage.blob_health_sample

    blob_dir = resolve_store_adjacent_path(get_config().storage.blob_dir)

    # RESPONSE + a non-null archive_path is the "this snapshot owns a blob"
    # predicate: REVISIT rows inherit content from `refers_to` and EPHEMERAL
    # entries are never archived, so neither should count as a miss.
    has_blob = (SnapshotRow.warc_record_type == WarcRecordType.RESPONSE.value) & (
        SnapshotRow.archive_path.is_not(None)
    )

    snapshots = (
        await session.scalar(select(func.count()).select_from(SnapshotRow).where(has_blob)) or 0
    )

    # Newest first: a store sharded recently shows it immediately, and the sample
    # stays deterministic for tests. Grouped rather than DISTINCT + ORDER BY,
    # which is invalid when the sort key is not in the select list.
    rows = await session.execute(
        select(SnapshotRow.content_hash, func.max(SnapshotRow.captured_at).label("latest"))
        .where(has_blob)
        .group_by(SnapshotRow.content_hash)
        .order_by(desc("latest"))
        .limit(sample)
    )
    hashes = [content_hash for content_hash, _latest in rows.all()]

    missing = sum(1 for h in hashes if not blob_exists(h))
    return BlobReachability(
        blob_dir=blob_dir,
        dir_exists=blob_dir.is_dir(),
        snapshots=snapshots,
        sampled=len(hashes),
        missing=missing,
    )
