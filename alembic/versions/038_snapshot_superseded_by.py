# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Record a collapsed snapshot on the row.

One nullable column on ``snapshots``:

* ``superseded_by_snapshot_id`` — set when the bulk extraction paths skip an
  unextracted snapshot of a ``MUTABLE`` entry because a newer extractable
  snapshot of the same entry exists; it names that newer snapshot. A snapshot
  is collapsed if and only if the column is non-null.

The collapsed snapshot's ``extraction_status`` is ``COMPLETE`` ("no extraction
work is owed"), which is a closed enum in the standard and so gains no new
value. ``COMPLETE`` alone cannot tell a skipped generation from an extracted
one, and two consumers (the generation backfill; ``reindex <entry>``) need
exactly that distinction — hence a column rather than an inference from
provenance edges, which a fully carried-forward or fully suppressed generation
also lacks.

**No backfill and no index.** Every existing row is NULL, which is correct: no
snapshot was collapsed before this revision. The column is only ever read
beside ``entry_id`` / ``extraction_status`` predicates that the existing
``ix_snapshots_entry_extraction`` index already serves.

Revision ID: 038
Revises: 037
"""

import sqlalchemy as sa

from alembic import op

revision = "038"
down_revision = "037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("snapshots") as batch:
        batch.add_column(sa.Column("superseded_by_snapshot_id", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("snapshots") as batch:
        batch.drop_column("superseded_by_snapshot_id")
