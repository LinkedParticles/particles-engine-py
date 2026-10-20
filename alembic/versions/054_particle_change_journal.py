# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The particle change journal, and an index for the schema-version guard.

Two additive changes, both for the query path's cost at today's store sizes
(``docs/benchmarks/query-latency.md``):

1. **``particle_changes`` and its triggers.** On SQLite, every insert, update
   and delete on ``particles`` appends the row's id to ``particle_changes``
   with a random token, and a fourth trigger keeps the newest 100,000 entries.
   The candidate cache (``particles/store/candidate_cache.py``) reads the
   newest entry before each query to learn whether any process changed a row
   since it last looked, and which. The journal records that a row changed,
   never what it holds, and nothing decides anything about a belief from it.
   One entry is written here so a store has a change marker before its first
   write. On another backend the table is created and stays empty, and the
   cache stays off.
2. **``ix_particles_status_schema_version``.** The schema-version guard runs
   before every query, extraction, review and reindex; with this index it asks
   "is any ACTIVE row on another version?" with two range seeks instead of a
   GROUP BY over the whole table.

**No data is rewritten**, and an older install keeps working against the
migrated store at the SQL level: it neither reads nor writes the new table,
and the triggers fire on its writes too, which is what keeps every reader's
cache correct while two installs share a store.

The trigger DDL is frozen here, as every migration's DDL is.
``particles.store.change_journal.JOURNAL_DDL`` is the copy ``create_all`` uses;
``tests/test_migration_054_change_journal.py`` checks the two agree.

``downgrade()`` drops the triggers, the table and the index.

Revision ID: 054
Revises: 053
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "054"
down_revision = "053"
branch_labels = None
depends_on = None

_TRIGGERS = (
    "trg_particles_journal_insert",
    "trg_particles_journal_update",
    "trg_particles_journal_delete",
    "trg_particle_changes_prune",
)

_JOURNAL_DDL = (
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
    "DELETE FROM particle_changes WHERE seq <= NEW.seq - 100000; "
    "END",
    "INSERT INTO particle_changes (particle_id, token) "
    "SELECT '', randomblob(8) WHERE NOT EXISTS (SELECT 1 FROM particle_changes)",
)


def upgrade() -> None:
    op.create_table(
        "particle_changes",
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("particle_id", sa.String(), nullable=False),
        sa.Column("token", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint("seq"),
    )
    if op.get_bind().dialect.name == "sqlite":
        for statement in _JOURNAL_DDL:
            op.execute(statement)
    op.create_index("ix_particles_status_schema_version", "particles", ["status", "schema_version"])


def downgrade() -> None:
    op.drop_index("ix_particles_status_schema_version", table_name="particles")
    if op.get_bind().dialect.name == "sqlite":
        for name in _TRIGGERS:
            op.execute(f"DROP TRIGGER IF EXISTS {name}")
    op.drop_table("particle_changes")
