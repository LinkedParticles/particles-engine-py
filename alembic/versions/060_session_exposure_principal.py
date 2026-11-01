# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Who wrote each session-start exposure row.

One nullable string column, ``recorded_by``, on ``session_exposures``: the
engine key principal of a row written over ``POST /session-exposures``.
An in-process write (the hook writing its own store) leaves it ``NULL``.

**No backfill.** ``NULL`` on an older row means only that it predates the
column; the writer of an earlier HTTP row was never recorded.

Revision ID: 060
Revises: 059
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "060"
down_revision = "059"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("session_exposures") as batch:
        batch.add_column(sa.Column("recorded_by", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("session_exposures") as batch:
        batch.drop_column("recorded_by")
