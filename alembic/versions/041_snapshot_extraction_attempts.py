# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Count extraction claims per snapshot.

One non-null integer column on ``snapshots``:

* ``extraction_attempts`` — incremented each time extraction claims the
  snapshot (the IN_PROGRESS write), never reset. The consolidation extract
  catch-up orders its PENDING queue on it, least-tried first,
  then by capture time.

Why a column. The pipeline hands a snapshot back to PENDING when every LLM
call for it failed, which is right for a transient outage and wrong for a
failure that recurs: a reply cut at the output budget, or a pooled batch that
outlives ``llm.batch.max_wait_seconds``, fails the same way the next night.
Ordered on capture time alone, those snapshots were first in line every night.
Measured on the owner's store 2026-09-25: the same six transcript snapshots,
captured 2026-07-19 to 2026-08-08, took most of the per-run cap on every run
since mid-September while 337 PENDING snapshots, every memory file written in
September among them, waited behind them. Nothing on the row recorded that a
snapshot had been tried, so no ordering could route around it.

**No backfill.** Every existing row starts at 0. A snapshot that was already
failing repeatedly is tried once more on the first run after this revision,
and from then on sorts behind every snapshot not yet tried.

Revision ID: 041
Revises: 040
"""

import sqlalchemy as sa

from alembic import op

revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("snapshots") as batch:
        batch.add_column(
            sa.Column(
                "extraction_attempts",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("snapshots") as batch:
        batch.drop_column("extraction_attempts")
