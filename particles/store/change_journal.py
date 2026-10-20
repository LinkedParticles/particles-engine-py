# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The particle change journal: which ``particles`` rows changed, and since when.

A read-side cache of decoded rows (:mod:`particles.store.candidate_cache`) has
to know, cheaply and for writes made by *any* process, whether a row it holds
is still what the store holds. The store is written by more than one process
at once (the MCP server, the session hooks, the CLI, an older pinned install),
so an in-process write hook cannot see every write. The database can.

On SQLite, four triggers maintain ``particle_changes``:

* every ``INSERT``, ``UPDATE`` or ``DELETE`` on ``particles`` appends one entry
  naming the row's id (an ``UPDATE`` that changes the id names both), with a
  fresh random ``token``;
* a fourth trigger keeps the newest :data:`JOURNAL_RETENTION` entries and
  deletes older ones, so the table stays bounded.

The newest entry, ``(seq, token)``, is the store's change marker. A reader that
remembers the marker it last saw can tell three cases apart with two indexed
lookups: nothing changed (same marker); rows changed and the reader can catch
up (its entry is still there, with its token, so every later entry is a change
it has not seen); or it cannot tell (its entry is gone or carries another
token) and must start over.

The token is what makes the third case sound. ``seq`` is a rowid, and a
rolled-back transaction gives its rowids back: a reader that looked inside a
write transaction which then rolled back holds a ``seq`` that a later,
committed write can reuse for a different change. The reused entry carries a
different random token, so the reader sees the mismatch and rebuilds instead
of trusting rows that never committed.

The journal records *that* a row changed, never what it holds, and nothing
reads it to decide anything about a belief: it is change tracking for caches,
and deleting every entry only costs the next reader a rebuild. It exists on
SQLite only. On another backend the table is created but stays empty and
:func:`journal_installed` reports ``False``, so every reader takes its uncached
path.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Integer, LargeBinary, String, bindparam, event, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.db import Base

#: How many journal entries are kept. A reader more than this many changes
#: behind rebuilds; a cache refresh that far behind would rebuild anyway.
JOURNAL_RETENTION = 100_000

#: The trigger names, which :func:`journal_installed` checks for.
TRIGGER_NAMES = (
    "trg_particles_journal_insert",
    "trg_particles_journal_update",
    "trg_particles_journal_delete",
    "trg_particle_changes_prune",
)

#: The DDL that installs the journal on SQLite, in order. Idempotent, so it
#: also brings an existing store's journal up (``IF NOT EXISTS``). Migration
#: 054 carries its own frozen copy; a test checks the two agree.
JOURNAL_DDL = (
    "CREATE TRIGGER IF NOT EXISTS trg_particles_journal_insert "
    "AFTER INSERT ON particles BEGIN "
    "INSERT INTO particle_changes (particle_id, token) VALUES (NEW.id, randomblob(8)); "
    "END",
    "CREATE TRIGGER IF NOT EXISTS trg_particles_journal_update "
    "AFTER UPDATE ON particles BEGIN "
    "INSERT INTO particle_changes (particle_id, token) "
    "SELECT OLD.id, randomblob(8) WHERE OLD.id IS NOT NEW.id; "
    "INSERT INTO particle_changes (particle_id, token) VALUES (NEW.id, randomblob(8)); "
    "END",
    "CREATE TRIGGER IF NOT EXISTS trg_particles_journal_delete "
    "AFTER DELETE ON particles BEGIN "
    "INSERT INTO particle_changes (particle_id, token) VALUES (OLD.id, randomblob(8)); "
    "END",
    "CREATE TRIGGER IF NOT EXISTS trg_particle_changes_prune "
    "AFTER INSERT ON particle_changes BEGIN "
    f"DELETE FROM particle_changes WHERE seq <= NEW.seq - {JOURNAL_RETENTION}; "
    "END",
    # The first entry: a store whose particles have not changed since the
    # journal was installed still has a marker, with a token of its own.
    "INSERT INTO particle_changes (particle_id, token) "
    "SELECT '', randomblob(8) WHERE NOT EXISTS (SELECT 1 FROM particle_changes)",
)


class ParticleChangeRow(Base):
    """One journal entry: a ``particles`` row changed (see the module docstring).

    ``seq`` is an ``INTEGER PRIMARY KEY`` and so SQLite's rowid. The entry
    written when the journal is installed names the empty id.
    """

    __tablename__ = "particle_changes"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    particle_id: Mapped[str] = mapped_column(String, nullable=False)
    token: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


def install_journal(sync_connection: Any) -> None:
    """Install the triggers and the first entry on a SQLite connection.

    Idempotent. Expects ``particles`` and ``particle_changes`` to exist.
    """
    for statement in JOURNAL_DDL:
        sync_connection.exec_driver_sql(statement)


def _install_after_create(target: Any, connection: Any, **kw: Any) -> None:
    # ``create_all`` (the unit suite, ``db init`` before Alembic stamps, the
    # benchmark generator) gets the same journal a migrated store has.
    if connection.dialect.name != "sqlite":
        return
    names = {t.name for t in kw.get("tables") or target.sorted_tables}
    if {"particles", "particle_changes"} <= names:
        install_journal(connection)


event.listen(Base.metadata, "after_create", _install_after_create)


async def journal_installed(session: AsyncSession) -> bool:
    """True when this store is SQLite and every journal trigger exists."""
    if session.get_bind().dialect.name != "sqlite":
        return False
    query = text(
        "SELECT count(*) FROM sqlite_master WHERE type = 'trigger' AND name IN :names"
    ).bindparams(bindparam("names", expanding=True))
    result = await session.execute(query, {"names": list(TRIGGER_NAMES)})
    installed: int = result.scalar_one()
    return installed == len(TRIGGER_NAMES)


async def read_head(session: AsyncSession) -> tuple[int, bytes] | None:
    """The newest entry's ``(seq, token)``, or ``None`` when the journal is empty."""
    row = (
        await session.execute(
            select(ParticleChangeRow.seq, ParticleChangeRow.token)
            .order_by(ParticleChangeRow.seq.desc())
            .limit(1)
        )
    ).first()
    return None if row is None else (row[0], row[1])


async def token_at(session: AsyncSession, seq: int) -> bytes | None:
    """The token of entry ``seq``, or ``None`` when that entry is gone."""
    return (
        await session.execute(select(ParticleChangeRow.token).where(ParticleChangeRow.seq == seq))
    ).scalar_one_or_none()


async def changed_since(session: AsyncSession, seq: int, *, limit: int) -> set[str] | None:
    """The distinct ids changed after entry ``seq``, or ``None`` past ``limit``.

    ``None`` tells the caller that catching up would cost more than starting
    over. The empty id of the first entry is never returned.
    """
    rows = (
        await session.execute(
            select(ParticleChangeRow.particle_id)
            .where(ParticleChangeRow.seq > seq, ParticleChangeRow.particle_id != "")
            .distinct()
            .limit(limit + 1)
        )
    ).scalars()
    ids = set(rows)
    return None if len(ids) > limit else ids
