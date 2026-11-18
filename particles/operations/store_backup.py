# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles db backup``: a consistent online copy of the default store and its blobs.

The copy is one directory laid out the way the container image expects its
data volume to be: ``<dir>/particles.db`` beside ``<dir>/corpus_blobs/``.
It is taken while the store stays in use, and it ends verified.

Three steps, in an order that makes the copy consistent without stopping a
writer:

1. **The database** is copied with SQLite's ``VACUUM INTO``, which reads the
   whole store inside one read transaction. On a WAL store (every store the
   engine opens is one) that transaction sees exactly the commits made before
   it began and blocks no writer, so the copy is a state the store really was
   in. The writer lock is not taken: the backup writes nothing to the
   store, and holding the lock for the length of a copy would stall every
   writer behind a slow read, which is what the lock's own rule forbids.
2. **The blobs** are copied after the database. A deposit writes its blob
   before the snapshot row that names it commits, so every blob a row in the
   copy names was already on disk when the copy began. Every valid blob under
   the live ``blob_dir`` is copied, each read through
   :func:`~particles.corpus.blob_io.read_blob` (which refuses a symlink and
   bytes that miss their hash) and written through
   :func:`~particles.corpus.blob_io.write_blob`. A blob written after the
   database copy is copied too and is harmless: no row in the copy names it.
3. **The verification** runs ``PRAGMA quick_check`` on the copy, then the
   ``corpus fsck`` audit (:func:`~particles.corpus.blob_fsck.audit_blobs`)
   over the copy's own database and blob directory. A blob the copy names
   that is not in the copy fails the backup when the live blob directory
   still has a file of that name (it could not be copied: a symlink, bytes
   that miss their name, a read error). A blob the live store has no file for
   either is reported as already lost, not as a failure of the copy: the copy
   holds everything the live store holds, which is what a backup can promise,
   and the remedy (re-deposit or retract the entries that name it) belongs to
   the live store, so a store with old losses can still be backed up and moved.
   When the live blob directory is absent, or the copy holds none of the
   store's blobs, the path is taken to be wrong and every missing blob fails
   the copy.

Snapshot rows keep their ``archive_path`` strings, which name the old host's
paths. Nothing reads them for a location: every reader resolves a blob by its
content hash under the configured ``blob_dir``, so the copy works wherever
``blob_dir`` points at its ``corpus_blobs``.

Only a file-backed SQLite store can be copied this way; a PostgreSQL store is
refused, and so is an in-memory one.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from particles.config import (
    STORE_BLOB_DIR_NAME,
    STORE_DB_FILE_NAME,
    get_config,
    resolve_store_adjacent_path,
    sqlite_file_path,
)
from particles.corpus.blob_fsck import BlobRef, FsckReport, audit_blobs
from particles.corpus.blob_io import read_blob, write_blob
from particles.corpus.deposit import blob_path

_SHARD_RE = re.compile(r"^[0-9a-f]{2}$")


class BackupRefused(Exception):
    """The backup cannot start; the message is one operator-readable line."""


@dataclass(frozen=True)
class BackupPlan:
    """Where the backup reads and writes, resolved and checked.

    ``source_blobs`` is the blob directory exactly as every reader of the live
    store names it (not resolved), so the copy reads it through the same
    symlink checks (:func:`~particles.corpus.blob_io.check_blob_root`).
    """

    source_db: Path
    source_blobs: Path
    target_dir: Path

    @property
    def target_db(self) -> Path:
        """The copy's database file."""
        return self.target_dir / STORE_DB_FILE_NAME

    @property
    def target_blobs(self) -> Path:
        """The copy's blob directory."""
        return self.target_dir / STORE_BLOB_DIR_NAME


@dataclass(frozen=True)
class BlobCopyOutcome:
    """What the blob step copied, and what it could not."""

    copied: int
    failed: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BackupResult:
    """The finished backup and its verification."""

    plan: BackupPlan
    blobs: BlobCopyOutcome
    integrity: str
    fsck: FsckReport
    lost_before: tuple[BlobRef, ...] = ()

    @property
    def not_copied(self) -> tuple[BlobRef, ...]:
        """Blobs the copy names and lacks although the live store has a file for each."""
        lost = {ref.content_hash for ref in self.lost_before}
        return tuple(ref for ref in self.fsck.missing if ref.content_hash not in lost)

    @property
    def verified(self) -> bool:
        """True when the copy's database checks clean and holds every blob the live store holds.

        A blob in :attr:`lost_before` (named by the store, and absent from the
        live blob directory as well as from the copy) does not fail the copy.
        """
        return self.integrity == "ok" and not self.fsck.elsewhere and not self.not_copied


def plan_backup(target: Path) -> BackupPlan:
    """Resolve the default store and ``target``, and refuse an unsafe backup.

    Refused: a store that is not a file-backed SQLite database, a store file
    that does not exist (opening it would create an empty one), the live
    store's own directory, a target whose database or blob directory would be
    the live store's own or would sit inside the live blob directory, and a
    target that exists and is not an empty directory.

    Raises:
        BackupRefused: For each case above.
    """
    url = get_config().storage.database_url
    db_file = sqlite_file_path(url)
    if db_file is None:
        if url.startswith("sqlite"):
            raise BackupRefused("db backup copies a file-backed store; this store is in memory.")
        scheme = url.split(":", 1)[0]
        raise BackupRefused(
            f"db backup copies a SQLite store only; this store is a {scheme} database. "
            "Back it up with that database's own tools (pg_dump for PostgreSQL), and "
            "copy the blob directory beside it."
        )
    source_db = Path(db_file).expanduser().resolve()
    if not source_db.is_file():
        raise BackupRefused(f"No store at {source_db}: there is nothing to back up.")
    source_blobs = resolve_store_adjacent_path(get_config().storage.blob_dir)
    live_blobs = source_blobs.resolve()

    target_dir = target.expanduser().resolve()
    plan = BackupPlan(source_db=source_db, source_blobs=source_blobs, target_dir=target_dir)
    if target_dir in (source_db, source_db.parent):
        raise BackupRefused(
            f"{target_dir} is the live store's own directory: the copy would overwrite "
            f"the live store {source_db}."
        )
    if plan.target_blobs == live_blobs or target_dir.is_relative_to(live_blobs):
        raise BackupRefused(f"{target_dir} would write into the live blob directory {live_blobs}.")
    if target_dir.exists():
        if not target_dir.is_dir():
            raise BackupRefused(f"{target_dir} exists and is not a directory.")
        if any(target_dir.iterdir()):
            raise BackupRefused(
                f"{target_dir} is not empty. Name a new or empty directory, so a backup "
                "never overwrites another one."
            )
    return plan


def copy_database(source_db: Path, target_db: Path) -> None:
    """Write a consistent copy of ``source_db`` to ``target_db`` with ``VACUUM INTO``.

    One read transaction, so the copy holds exactly the commits made before it
    began, and on a WAL store no writer waits on it. The store is opened
    ``mode=ro``: a store file that vanished since it was checked is an error
    rather than a new empty database, and closing the connection never
    checkpoints the live store's WAL into its database file, which a
    read-write connection does when it is the last one open. The copy is made
    owner-only, and a partial file left by a failed copy is removed.
    """
    source_uri = f"file:{quote(str(source_db), safe='/')}?mode=ro"
    try:
        with contextlib.closing(
            sqlite3.connect(source_uri, uri=True, isolation_level=None)
        ) as conn:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("VACUUM INTO ?", (str(target_db),))
        os.chmod(target_db, 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            target_db.unlink()
        raise


def copy_blobs(source_blobs: Path, target_blobs: Path) -> BlobCopyOutcome:
    """Copy every valid blob under ``source_blobs`` into ``target_blobs``.

    Only names in the content-addressed layout (``<hh>/<sha256>``, the shard
    matching the hash) are considered; anything else in the tree is skipped.
    Each blob is read through ``read_blob`` and written through ``write_blob``,
    so a symlink is refused and the bytes are checked against their name on
    both sides. A blob that cannot be copied is recorded and the rest go on.
    """
    target_blobs.mkdir(parents=True, exist_ok=True)
    copied = 0
    failed: list[tuple[str, str]] = []
    if not source_blobs.is_dir():
        return BlobCopyOutcome(copied=0, failed=())
    for shard in sorted(source_blobs.iterdir()):
        if not _SHARD_RE.match(shard.name) or shard.is_symlink() or not shard.is_dir():
            continue
        for entry in sorted(shard.iterdir()):
            name = entry.name
            try:
                blob_path(name, blob_dir=source_blobs)
            except ValueError:
                continue
            if not name.startswith(shard.name):
                continue
            try:
                content = read_blob(source_blobs, name)
                write_blob(target_blobs, name, content)
            except OSError as exc:
                failed.append((name, exc.strerror or str(exc)))
                continue
            copied += 1
    return BlobCopyOutcome(copied=copied, failed=tuple(failed))


def lost_from_live(missing: tuple[BlobRef, ...], source_blobs: Path) -> tuple[BlobRef, ...]:
    """The refs in ``missing`` the live blob directory has no file for either.

    Any directory entry under the blob's name counts as a file, a dangling
    symlink included, so a blob the copy refused to follow still fails the
    backup. A malformed hash cannot name a file and counts as lost.
    """
    lost: list[BlobRef] = []
    for ref in missing:
        try:
            path = blob_path(ref.content_hash, blob_dir=source_blobs)
        except ValueError:
            lost.append(ref)
            continue
        if not os.path.lexists(path):
            lost.append(ref)
    return tuple(lost)


def quick_check(db: Path) -> str:
    """``PRAGMA quick_check`` on ``db``: ``"ok"``, or SQLite's first complaint."""
    with contextlib.closing(sqlite3.connect(db, isolation_level=None)) as conn:
        rows = conn.execute("PRAGMA quick_check").fetchall()
    if not rows:
        return "no result"
    return "ok" if rows[0][0] == "ok" else "; ".join(str(row[0]) for row in rows[:5])


async def audit_copy(plan: BackupPlan) -> FsckReport:
    """Run the ``corpus fsck`` audit over the copy's database and blob directory.

    The copy is opened on an engine of its own, never one the registry caches,
    so the live store's engine and writer lock are not involved.
    """
    url = URL.create("sqlite+aiosqlite", database=str(plan.target_db))
    engine = create_async_engine(url, poolclass=NullPool, hide_parameters=True)
    try:
        async with AsyncSession(engine) as session:
            return await audit_blobs(session, blob_dir=plan.target_blobs)
    finally:
        await engine.dispose()


async def backup_store(target: Path) -> BackupResult:
    """Back up the default store and its blobs into ``target``, then verify the copy.

    Raises:
        BackupRefused: When :func:`plan_backup` refuses.
        sqlite3.Error: When the database copy fails (its partial file is removed).
        OSError: When the target directory or the copy cannot be written.
    """
    plan = plan_backup(target)
    if not plan.target_dir.exists():
        # The copy is the whole store: readable by its owner only.
        plan.target_dir.mkdir(mode=0o700, parents=True)
    await asyncio.to_thread(copy_database, plan.source_db, plan.target_db)
    blobs = await asyncio.to_thread(copy_blobs, plan.source_blobs, plan.target_blobs)
    integrity = await asyncio.to_thread(quick_check, plan.target_db)
    fsck = await audit_copy(plan)
    # A live blob directory that is absent or holds none of the store's blobs is
    # a wrong path, not a store that lost them all: every missing blob then
    # fails the copy, as before.
    lost_before: tuple[BlobRef, ...] = ()
    if fsck.present and plan.source_blobs.is_dir():
        lost_before = await asyncio.to_thread(lost_from_live, fsck.missing, plan.source_blobs)
    return BackupResult(
        plan=plan, blobs=blobs, integrity=integrity, fsck=fsck, lost_before=lost_before
    )
