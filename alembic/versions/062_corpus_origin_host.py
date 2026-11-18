# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The host a file:// corpus entry's path names.

Adds the nullable ``corpus_entries.origin_host`` column and writes no data.
``particles db init`` stamps the existing ``file://`` entries once, with the id
of the host it runs on; the engine's own startup migration never does, so a
container that migrates a moved store leaves them unknown.

Revision ID: 062
Revises: 061
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "062"
down_revision = "061"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("corpus_entries") as batch:
        batch.add_column(sa.Column("origin_host", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("corpus_entries") as batch:
        batch.drop_column("origin_host")
