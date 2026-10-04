# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Record how far extraction read each snapshot.

One nullable integer column on ``snapshots``:

* ``extracted_through`` — the offset, in bytes of the raw content, up to which
  extraction read the snapshot. The content's length when the read reached the
  end; the largest paragraph boundary whose decoded prefix was read in full
  when the read was cut at ``extraction.max_llm_calls_per_source``. Written in
  the transaction that marks the snapshot COMPLETE after an extraction.

Why a column. An APPEND_ONLY entry's snapshot is extracted as a delta from the
last snapshot the store extracted, and the delta starts where that read
stopped. Recording the stop is what lets text a capped read skipped be read on
the next pass instead of never.

**No backfill.** Every existing row stays NULL. The pipeline derives the offset
of such a base from the chunk hashes its entry's claims carry the first time it
is used as one, and otherwise counts it as fully extracted.

Revision ID: 043
Revises: 042
"""

import sqlalchemy as sa

from alembic import op

revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("snapshots") as batch:
        batch.add_column(sa.Column("extracted_through", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("snapshots") as batch:
        batch.drop_column("extracted_through")
