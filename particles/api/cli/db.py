# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""db verb — initialise the database, or back it up."""

from __future__ import annotations

from pathlib import Path

import typer

from particles.api.cli import app, run
from particles.api.cli._progress import confirm
from particles.corpus.origin import StampPlan
from particles.db import create_tables, session_scope
from particles.ingest.importers.registry import ensure_extractor_records
from particles.store.codex_capture_store import PRESERVED_TABLES as CODEX_CAPTURE_TABLES
from particles.store.session_exposure_store import PRESERVED_TABLES

# Tables the SCHEMA_VERSION 0.3.x → 1.0.0 scrap-and-re-extract upgrade
# must NOT touch. The blob store on disk is preserved likewise —
# it's content-addressed by SHA-256 so re-extraction picks it up by hash.
_CORPUS_TABLES: frozenset[str] = frozenset({"corpus_entries", "snapshots"})
# Primary records that cannot be re-derived from the corpus survive the rebuild
# too: what each session was shown at session start.
_KEPT_TABLES: frozenset[str] = _CORPUS_TABLES | PRESERVED_TABLES | CODEX_CAPTURE_TABLES


@app.command("db")
def db_cmd(
    action: str = typer.Argument(..., help="Action: init or backup"),
    target: Path | None = typer.Argument(
        None,
        help=(
            "backup: the new or empty directory to write the copy into. It receives "
            "particles.db and corpus_blobs/, the layout of the container's /data volume."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "init: drop every particle-store table (preserving the corpus, the blob "
            "store and the session-exposure record) and rebuild from scratch. "
            " this is the upgrade path across a SCHEMA_VERSION major "
            "bump. Confirms before dropping."
        ),
    ),
    origin_preview: bool = typer.Option(
        False,
        "--origin-preview",
        help=(
            "init: print how many file:// entries the one-time origin stamp would mark as "
            "this host's files, by class, and change nothing (no migration either). Run "
            "it before the first init that stamps, on the machine whose files the store "
            "tracks."
        ),
    ),
) -> None:
    """Manage the database.

    ``init`` migrates the store to the current schema. Once per store it also
    marks each ``file://`` entry that records no host as this host's file, since
    the local refresh reads only its own host's files; run it on the machine
    whose files the store tracks.

    ``backup <dir>`` copies the default store and its blobs into ``<dir>`` while
    the store stays in use, then checks the copy and runs the ``corpus fsck``
    audit over it. It exits non-zero unless the copy verifies.
    """
    if action == "backup":
        if target is None:
            typer.echo("db backup needs the directory to write the copy into.", err=True)
            raise typer.Exit(1)
        run(_db_backup(target))
        return
    if target is not None:
        typer.echo(f"db {action} takes no directory argument.", err=True)
        raise typer.Exit(1)
    if action == "init" and origin_preview:
        run(_origin_preview())
        return
    if action == "init":
        if force:
            confirm(
                "This drops the entire particle store (subjects, particles, "
                "trust, taxonomies, caches). The corpus and blob store are "
                "preserved; snapshots become PENDING for re-extraction. "
                "Continue?",
                default=False,
                abort=True,
            )
            run(_db_init_force())
            typer.echo("Particle store rebuilt. Run `particles extract --all-pending` next.")
        else:
            run(_db_init())
            typer.echo("Database tables created.")
    else:
        typer.echo(f"Unknown action: {action}", err=True)
        raise typer.Exit(1)


async def _db_backup(target: Path) -> None:
    import sqlite3

    from particles.api.cli._remote import ensure_local
    from particles.operations.store_backup import BackupRefused, backup_store

    ensure_local("db backup")
    try:
        result = await backup_store(target)
    except BackupRefused as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except sqlite3.Error as exc:
        typer.echo(f"The database copy failed: {exc}. Nothing was changed in the store.", err=True)
        raise typer.Exit(1) from exc
    except OSError as exc:
        typer.echo(
            f"The backup could not be written: {exc}. Nothing was changed in the store.",
            err=True,
        )
        raise typer.Exit(1) from exc

    plan = result.plan
    typer.echo(f"Store:     {plan.source_db}")
    typer.echo(f"Blobs:     {plan.source_blobs}")
    typer.echo(f"Copy:      {plan.target_dir}")
    typer.echo(f"\nDatabase copied (quick_check: {result.integrity}).")
    typer.echo(f"Blobs copied: {result.blobs.copied}")
    if result.blobs.failed:
        typer.echo(f"  {len(result.blobs.failed)} blob file(s) could not be copied:", err=True)
        for name, reason in result.blobs.failed[:10]:
            typer.echo(f"    {name[:16]}…  {reason}", err=True)
    fsck = result.fsck
    typer.echo(
        f"Blobs the copy references: {fsck.total} ({len(fsck.missing)} missing from the copy)"
    )
    if not result.verified:
        not_copied = result.not_copied
        for ref in not_copied[:10]:
            typer.echo(f"  missing {ref.content_hash[:16]}…  {ref.label}", err=True)
        if len(not_copied) > 10:
            typer.echo(f"  … and {len(not_copied) - 10} more", err=True)
        if result.integrity != "ok":
            typer.echo("\nThe copy did not verify: its database failed quick_check.", err=True)
        if not_copied:
            typer.echo(
                "\nThe copy did not verify: the live store has a file for each blob listed, "
                "and it was not copied. Run `particles corpus fsck` on the live store to "
                "check them.",
                err=True,
            )
        raise typer.Exit(1)
    if result.lost_before:
        typer.echo(
            f"  {len(result.lost_before)} of them are missing from the live store too, so no "
            "copy can hold them:",
            err=True,
        )
        for ref in result.lost_before[:10]:
            typer.echo(f"    {ref.content_hash[:16]}…  {ref.label}", err=True)
        if len(result.lost_before) > 10:
            typer.echo(f"    … and {len(result.lost_before) - 10} more", err=True)
    if result.blobs.failed:
        typer.echo(
            f"\nBackup verified. {len(result.blobs.failed)} blob file(s) no row in the copy "
            "names could not be copied (listed above)."
        )
    else:
        typer.echo("\nBackup verified.")
    if result.lost_before:
        typer.echo(
            "The copy holds every blob the live store holds. The blobs listed above were "
            "lost before this backup: `particles corpus fsck` on the live store names the "
            "entries to re-deposit or retract."
        )
    typer.echo(
        f"To use it, point DATABASE_URL at {plan.target_db}; the default "
        f"storage.blob_dir then resolves to {plan.target_blobs} beside it."
    )


async def _db_init() -> None:
    from particles.operations.corpus_origin import origin_stamp

    await create_tables()
    async with session_scope() as session:
        wrote = await ensure_extractor_records(session)
        await session.commit()
    if wrote:
        typer.echo(f"  Registered {wrote} extractor record(s).")
    # Once per store, and only here: the engine's startup migration never
    # stamps, so a moved store migrated in a container is adopted by its
    # client's pushes instead.
    async with session_scope(write=True) as session:
        outcome = await origin_stamp(session, apply=True)
        await session.commit()
    if outcome.written:
        typer.echo(f"  {_stamp_line(outcome.plan, done=True)}")
        typer.echo("  Run db init on the machine whose files this store tracks.")
    elif outcome.plan.host_id is None and outcome.plan.total:
        typer.echo(
            "  This host has no usable host id (host.id_file), so no file:// entry was "
            "stamped and the local refresh reads none of them.",
            err=True,
        )


async def _origin_preview() -> None:
    from particles.operations.corpus_origin import origin_stamp

    async with session_scope() as session:
        outcome = await origin_stamp(session, apply=False)
    if outcome.already_stamped:
        typer.echo(
            "This store was stamped before; db init stamps nothing more. An entry that "
            "records no host is adopted by the next deposit from the host holding it."
        )
        return
    if outcome.plan.host_id is None:
        typer.echo("This host has no usable host id (host.id_file); db init would stamp nothing.")
        return
    typer.echo(_stamp_line(outcome.plan, done=False))


def _stamp_line(plan: StampPlan, *, done: bool) -> str:
    verb = "Stamped" if done else "Would stamp"
    return (
        f"{verb} {plan.total} file:// entr(ies) as this host's files (host {plan.host_id}): "
        f"{plan.rule_files} rule file(s), {plan.memory_files} memory file(s), "
        f"{plan.other} other deposit(s)."
    )


async def _db_init_force() -> None:
    """Scrap-and-re-extract reset.

    Clears every table but the corpus and the session-exposure record,
    and resets ``snapshots.extraction_status``
    to PENDING so the operator's next ``particles extract --all-pending`` run
    rebuilds the particle store from the preserved corpus. The corpus rows
    themselves (and the content-addressed blob files on disk) are untouched.

    Assumes tables already exist — ``--force`` is the SCHEMA_VERSION-bump
    upgrade path, so by construction the operator already has a populated
    store. Run ``particles db init`` (no flag) first if you don't.
    """
    from sqlalchemy import delete, update

    # Importing _orm_modules registers every table on Base.metadata so the
    # sorted_tables iteration below sees the full set. Without this, only
    # tables transitively imported by the CLI module appear (corpus_entries,
    # snapshots) and every particle-store table silently survives the
    # rebuild — the bug v0.34.2 shipped with.
    import particles._orm_modules  # noqa: F401
    from particles.core.schema import ExtractionStatus
    from particles.corpus.store import SnapshotRow
    from particles.db import Base

    async with session_scope() as session:
        # Iterate in reverse dependency order so FK constraints (where they
        # exist) don't reject a child-after-parent delete.
        for table in reversed(Base.metadata.sorted_tables):
            if table.name in _KEPT_TABLES:
                continue
            await session.execute(delete(table))
        # Snapshots survive but their extraction_status was COMPLETE / FAILED /
        # IN_PROGRESS before — reset to PENDING so --all-pending picks them up.
        # The collapse mark goes with them: it records a decision
        # made against the store being wiped, and a row left PENDING *and*
        # marked would be skipped by every bulk pass forever. The next pass
        # re-derives which generations are superseded.
        await session.execute(
            update(SnapshotRow).values(
                extraction_status=ExtractionStatus.PENDING.value,
                superseded_by_snapshot_id=None,
            )
        )
        await session.commit()

    # Re-register extractor records after the wipe so the next run knows
    # which extractors exist and their applicability metadata.
    async with session_scope() as session:
        wrote = await ensure_extractor_records(session)
        await session.commit()
    if wrote:
        typer.echo(f"  Registered {wrote} extractor record(s).")
