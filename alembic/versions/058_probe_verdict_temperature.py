# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What each probe-ledger answer was sampled under.

One nullable string column, ``temperature``, naming what the answering call
sent: ``sent`` (a fixed temperature was asked for and received), ``dropped``
(asked for and not received: the model refuses the parameter), ``unset`` (none
was asked for), or ``none`` (a rung that makes no model call wrote the row).

**No backfill.** ``NULL`` means only that the row predates the column. The
history cannot be reconstructed: the adapter learns which models refuse the
parameter per process and never persists it, and inferring it from the model
id would need the compiled model list the adapter deliberately does not keep.

Revision ID: 058
Revises: 057
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "058"
down_revision = "057"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("probe_verdicts") as batch:
        batch.add_column(sa.Column("temperature", sa.String(length=16), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("probe_verdicts") as batch:
        batch.drop_column("temperature")
