# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The probe-verdict ledger records every rung, not only model probes.

Three nullable or defaulted columns on ``probe_verdicts``, additive:

* ``rung`` — the stamp of the rung that produced the row, ``name@version``
  (a model probe's version is its prompt hash);
* ``rung_verdict`` — the verdict in the ladder's own vocabulary
  (``yes`` / ``no``, ``changes`` / ``fixed`` / ``different``), where the
  existing ``verdict`` boolean cannot carry it;
* ``shadow`` — the rung has not met its bar: its verdict is recorded and was
  never returned.

Existing rows keep ``rung`` and ``rung_verdict`` null and ``shadow`` false.
Nothing is backfilled: the rows recorded before the ladder answer exactly as
they did.

Revision ID: 052
Revises: 051
"""

import sqlalchemy as sa

from alembic import op

revision = "052"
down_revision = "051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("probe_verdicts") as batch:
        batch.add_column(sa.Column("rung", sa.String(length=96), nullable=True))
        batch.add_column(sa.Column("rung_verdict", sa.String(length=16), nullable=True))
        batch.add_column(
            sa.Column("shadow", sa.Boolean(), nullable=False, server_default=sa.false())
        )


def downgrade() -> None:
    with op.batch_alter_table("probe_verdicts") as batch:
        batch.drop_column("shadow")
        batch.drop_column("rung_verdict")
        batch.drop_column("rung")
