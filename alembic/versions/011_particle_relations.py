"""Add particle_relations table (co-evidential links).

Revision ID: 011
Revises: 010
"""

import sqlalchemy as sa

from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "particle_relations",
        sa.Column("particle_a", sa.String, primary_key=True),
        sa.Column("particle_b", sa.String, primary_key=True),
        sa.Column("relation_type", sa.String, primary_key=True),
        sa.Column("created_by", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.UniqueConstraint(
            "particle_a",
            "particle_b",
            "relation_type",
            name="uq_particle_relations_pair_type",
        ),
    )
    op.create_index(
        "ix_particle_relations_a",
        "particle_relations",
        ["particle_a", "relation_type"],
    )
    op.create_index(
        "ix_particle_relations_b",
        "particle_relations",
        ["particle_b", "relation_type"],
    )


def downgrade() -> None:
    op.drop_index("ix_particle_relations_b", table_name="particle_relations")
    op.drop_index("ix_particle_relations_a", table_name="particle_relations")
    op.drop_table("particle_relations")
