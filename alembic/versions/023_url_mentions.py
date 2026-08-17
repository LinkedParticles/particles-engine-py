"""Add URL-mention citation-signal tables.

``url_mentions`` + ``url_suggestion_state`` — track external URLs mentioned
across the corpus (including undeposited ones) so frequently-cited primary
sources can be ranked as deposit suggestions. Storage-layer metadata, not
particle schema: does not touch ``SCHEMA_VERSION``.

Revision ID: 023
Revises: 022
"""

import sqlalchemy as sa

from alembic import op

revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "url_mentions",
        sa.Column("source_entry_id", sa.String(), primary_key=True),
        sa.Column("canonical_url", sa.String(), primary_key=True),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_entry_id", sa.String(), nullable=True),
    )
    op.create_index("ix_url_mentions_canonical", "url_mentions", ["canonical_url"])
    op.create_index("ix_url_mentions_target", "url_mentions", ["target_entry_id"])

    op.create_table(
        "url_suggestion_state",
        sa.Column("canonical_url", sa.String(), primary_key=True),
        sa.Column("suppressed_until", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("url_suggestion_state")
    op.drop_index("ix_url_mentions_target", table_name="url_mentions")
    op.drop_index("ix_url_mentions_canonical", table_name="url_mentions")
    op.drop_table("url_mentions")
