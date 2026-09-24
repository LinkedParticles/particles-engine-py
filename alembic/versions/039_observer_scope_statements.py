# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer-scope statements.

One new table, ``observer_scope_statements``: an operator's standing judgement
that a particle, or a whole corpus entry, is in view for every project
observer. It is lens-side policy — the claim and its provenance are untouched —
so it is its own table rather than a column on either.

The primary key is ``(target_kind, target_id)``: a target is widened or it is
not, and the read path looks statements up by exactly that pair, so no further
index is needed.

**No backfill.** A store with no statements reads exactly as before.

Revision ID: 039
Revises: 038
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "039"
down_revision = "038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "observer_scope_statements",
        sa.Column("target_kind", sa.String(), primary_key=True),
        sa.Column("target_id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("observer_scope_statements")
