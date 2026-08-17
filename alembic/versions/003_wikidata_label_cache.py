"""Wikidata label cache table.

Revision ID: 003
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "wikidata_label_cache",
        sa.Column("qid", sa.String, primary_key=True),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("cached_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("wikidata_label_cache")
