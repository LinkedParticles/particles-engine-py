"""Extractor registry and trust model (Extension A).

Adds extractor_records table for per-extractor trust_weight and applicability
clause storage.

Revision ID: 006
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "extractor_records",
        sa.Column("extractor_id", sa.String(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("applicability_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("trust_weight", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column("registered_by", sa.Text(), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("extractor_records")
