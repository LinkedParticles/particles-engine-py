# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Mark a snapshot that holds a partial whole read.

One boolean column on ``snapshots``:

* ``resume_whole`` — true while the snapshot holds the claims of a whole read
  that failed partway, and its next read is the same whole read. Set in the
  transaction that writes those claims with the snapshot ``PENDING``; cleared
  when the snapshot is marked ``COMPLETE`` and by the reindex that retires its
  claims.

Why a column. A partial whole read keeps every answered chunk and relies on
carry-forward to skip them on the retry, which works only while the retry is
the same whole read of the same content. Without a marker the retry turns into
a delta as soon as any snapshot of the entry completes, and a later snapshot
read first chunks the same text differently at its end. The marker holds the
retry to a whole read, and the entry's later snapshots wait while it is set.

**No backfill.** Every existing row is false: no partial read exists before
this record.

Revision ID: 047
Revises: 046
"""

import sqlalchemy as sa

from alembic import op

revision = "047"
down_revision = "046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("snapshots") as batch:
        batch.add_column(
            sa.Column(
                "resume_whole",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("snapshots") as batch:
        batch.drop_column("resume_whole")
