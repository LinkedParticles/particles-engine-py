"""Subject layer.

Revision ID: 002
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subjects",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("canonical_name", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("aliases_json", sa.Text, nullable=False, default="[]"),
        sa.Column("external_ids_json", sa.Text, nullable=False, default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("asserted_by", sa.String, nullable=False),
    )
    op.create_index("ix_subjects_canonical_name", "subjects", ["canonical_name"])

    op.create_table(
        "particle_subjects",
        sa.Column("particle_id", sa.String, primary_key=True),
        sa.Column("subject_id", sa.String, primary_key=True),
    )
    op.create_index("ix_particle_subjects_subject_id", "particle_subjects", ["subject_id"])

    op.add_column(
        "particles",
        sa.Column("subject_ids_json", sa.Text, nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("particles", "subject_ids_json")
    op.drop_table("particle_subjects")
    op.drop_table("subjects")
