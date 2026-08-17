"""Add taxonomy + tag_nodes + particle_tag_edges tables and tags_json column.

Revision ID: 012
Revises: 011
"""

import sqlalchemy as sa

from alembic import op

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "taxonomies",
        sa.Column("taxonomy_id", sa.String, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("version", sa.String, nullable=False),
        sa.Column("author", sa.String, nullable=False),
        sa.Column("domain", sa.String, nullable=True),
        sa.Column("corpus_entry_id", sa.String, nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "tag_nodes",
        sa.Column("taxonomy_id", sa.String, primary_key=True),
        sa.Column("tag", sa.Text, primary_key=True),
        sa.Column("parent", sa.Text, nullable=True),
        sa.Column("aliases_json", sa.Text, nullable=False, server_default="[]"),
        sa.Column("description", sa.Text, nullable=True),
    )
    op.create_index(
        "ix_tag_nodes_taxonomy_parent",
        "tag_nodes",
        ["taxonomy_id", "parent"],
    )
    op.create_table(
        "particle_tag_edges",
        sa.Column("particle_id", sa.String, primary_key=True),
        sa.Column("tag", sa.Text, primary_key=True),
    )
    op.create_index(
        "ix_particle_tag_edges_tag",
        "particle_tag_edges",
        ["tag"],
    )
    op.add_column(
        "particles",
        sa.Column("tags_json", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("particles", "tags_json")
    op.drop_index("ix_particle_tag_edges_tag", table_name="particle_tag_edges")
    op.drop_table("particle_tag_edges")
    op.drop_index("ix_tag_nodes_taxonomy_parent", table_name="tag_nodes")
    op.drop_table("tag_nodes")
    op.drop_table("taxonomies")
