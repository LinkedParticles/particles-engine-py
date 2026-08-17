"""db verb — initialise the database."""

from __future__ import annotations

import typer

from particles.api.cli import app, run
from particles.db import create_tables, session_scope
from particles.ingest.importers.registry import ensure_extractor_records

# Tables the SCHEMA_VERSION 0.3.x → 1.0.0 scrap-and-re-extract upgrade
# must NOT touch. The blob store on disk is preserved likewise —
# it's content-addressed by SHA-256 so re-extraction picks it up by hash.
_CORPUS_TABLES: frozenset[str] = frozenset({"corpus_entries", "snapshots"})


@app.command("db")
def db_cmd(
    action: str = typer.Argument(..., help="Action: init"),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "init: drop every particle-store table (preserving the corpus + blob "
            "store) and rebuild from scratch. this is the upgrade "
            "path across a SCHEMA_VERSION major bump. Confirms before dropping."
        ),
    ),
) -> None:
    """Manage the database."""
    if action == "init":
        if force:
            typer.confirm(
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


async def _db_init() -> None:
    await create_tables()
    async with session_scope() as session:
        wrote = await ensure_extractor_records(session)
        await session.commit()
    if wrote:
        typer.echo(f"  Registered {wrote} extractor record(s).")


async def _db_init_force() -> None:
    """Scrap-and-re-extract reset.

    Clears every non-corpus table and resets ``snapshots.extraction_status``
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
            if table.name in _CORPUS_TABLES:
                continue
            await session.execute(delete(table))
        # Snapshots survive but their extraction_status was COMPLETE / FAILED /
        # IN_PROGRESS before — reset to PENDING so --all-pending picks them up.
        await session.execute(
            update(SnapshotRow).values(extraction_status=ExtractionStatus.PENDING.value)
        )
        await session.commit()

    # Re-register extractor records after the wipe so the next run knows
    # which extractors exist and their applicability metadata.
    async with session_scope() as session:
        wrote = await ensure_extractor_records(session)
        await session.commit()
    if wrote:
        typer.echo(f"  Registered {wrote} extractor record(s).")
