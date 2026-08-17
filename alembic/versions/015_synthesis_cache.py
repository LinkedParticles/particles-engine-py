"""Add synthesis_cache table.

Shared cache for LLM-synthesised per-Subject prose articles. Keyed on
``(subject_id, input_hash, prompt_version)``; each prose exporter
(wiki, obsidian, future logseq) consults the table before invoking
the LLM so the same input under the same prompt version pays LLM
cost only once across exporters.

Revision ID: 015
Revises: 014
"""

import sqlalchemy as sa

from alembic import op

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "synthesis_cache",
        sa.Column("subject_id", sa.String(), nullable=False),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("prompt_version", sa.String(), nullable=False),
        sa.Column("article_body", sa.Text(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("layer_b_verdict", sa.String(), nullable=True),
        sa.Column("quality_notes", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("subject_id", "input_hash", "prompt_version"),
    )


def downgrade() -> None:
    op.drop_table("synthesis_cache")
