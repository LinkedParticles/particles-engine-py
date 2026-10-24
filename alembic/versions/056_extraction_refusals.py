# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Sources the extraction model refused.

One additive table, ``extraction_refusals``: one row per
``(content_hash, purpose, provider_model)`` the extraction model declined, with
the snapshot and entry that last met it, the refusal category, the first and
last refusal time and a count. The bulk extraction paths skip a ``PENDING``
snapshot a current row holds (``particles/corpus/refusals.py``).

**No data is rewritten**, and an older install keeps working against the
migrated store: it neither reads nor writes the table, and simply keeps
re-sending refused sources as before.

``downgrade()`` drops the table.

Numbered 056 because migration 055 was claimed by an open sibling branch; the
``down_revision`` names whichever migration is head on ``main`` when this
merges.

Revision ID: 056
Revises: 055
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "056"
down_revision = "055"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "extraction_refusals",
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("purpose", sa.String(), nullable=False),
        sa.Column("provider_model", sa.String(), nullable=False),
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("entry_id", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=True),
        sa.Column("first_refused_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_refused_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("refusals", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("content_hash", "purpose", "provider_model"),
    )


def downgrade() -> None:
    op.drop_table("extraction_refusals")
