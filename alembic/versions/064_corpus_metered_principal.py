# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The metered agent principal a corpus entry's deposits are counted against.

Adds the nullable, indexed ``corpus_entries.metered_principal`` column and
writes no data. An entry deposited before it reads as unmetered until a deposit
made with a metered agent key lands on it, which marks it from then on.

Revision ID: 064
Revises: 063
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "064"
down_revision = "063"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("corpus_entries") as batch:
        batch.add_column(sa.Column("metered_principal", sa.String(), nullable=True))
        batch.create_index("ix_corpus_entries_metered_principal", ["metered_principal"])


def downgrade() -> None:
    with op.batch_alter_table("corpus_entries") as batch:
        batch.drop_index("ix_corpus_entries_metered_principal")
        batch.drop_column("metered_principal")
