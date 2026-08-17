"""Add chunk_hash column + index to particle_provenance_edges.

Revision ID: 009
Revises: 008
"""

import sqlalchemy as sa

from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "particle_provenance_edges",
        sa.Column("chunk_hash", sa.String, nullable=True),
    )
    op.create_index(
        "ix_prov_edges_chunk",
        "particle_provenance_edges",
        ["corpus_entry_id", "chunk_hash"],
    )


def downgrade() -> None:
    op.drop_index("ix_prov_edges_chunk", table_name="particle_provenance_edges")
    op.drop_column("particle_provenance_edges", "chunk_hash")
