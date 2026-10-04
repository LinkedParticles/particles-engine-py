# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""An answer column on the probe-verdict ledger.

The subject-link judge picks one of a name's Wikidata candidates, or none, so
its record is more than the yes or no ``verdict`` holds. One nullable text
column, ``answer``, carries it as JSON: the chosen QID (or null) and every
candidate the judge was offered. Rows of the pairwise kinds leave it null.

**No backfill.** Every existing row is a pairwise verdict, whose answer is its
``verdict``.

Revision ID: 046
Revises: 045
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "046"
down_revision = "045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("probe_verdicts") as batch:
        batch.add_column(sa.Column("answer", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("probe_verdicts") as batch:
        batch.drop_column("answer")
