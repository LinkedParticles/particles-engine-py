"""Add corpus_follow_edges table.

Records the depth-1 follow relationship when ``deposit_url`` follows
a link-shaped post's primary URL and deposits the target as a
separate corpus entry. Join table shape because the same article URL
is commonly reached from many sources (press release, viral link)
and a single column on ``corpus_entries`` couldn't represent the
fan-in.

Revision ID: 014
Revises: 013
"""

import sqlalchemy as sa

from alembic import op

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "corpus_follow_edges",
        sa.Column("via_entry_id", sa.String(), nullable=False),
        sa.Column("target_entry_id", sa.String(), nullable=False),
        sa.Column("link_type", sa.String(), nullable=False),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("via_entry_id", "target_entry_id", "link_type"),
    )
    op.create_index(
        "ix_corpus_follow_edges_target",
        "corpus_follow_edges",
        ["target_entry_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_corpus_follow_edges_target", table_name="corpus_follow_edges")
    op.drop_table("corpus_follow_edges")
