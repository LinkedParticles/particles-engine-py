"""Trust lenses (0.60.0).

Three tables: ``trust_lenses`` (one row per materialised lens, latest version
per name), ``trust_lens_entries`` (polymorphic policy entries — statements,
URL rules, extractor weights), and ``trust_lens_adoptions`` (the store's
viewpoint: one row per adopted lens name). Lens definitions arrive as
deposited corpus artefacts (``source_type = TRUST_LENS_DEFINITION``) and are
materialised by the TrustLensExtractor; adoption composes them into the query-time TrustPolicy, local-wins / most-skeptical-across-lenses.

Revision ID: 020
Revises: 019
"""

import sqlalchemy as sa

from alembic import op

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trust_lenses",
        sa.Column("lens_id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("publisher", sa.String(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("corpus_entry_id", sa.String(), nullable=True),
        sa.Column("materialised_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_trust_lenses_name", "trust_lenses", ["name"], unique=True)

    op.create_table(
        "trust_lens_entries",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("lens_id", sa.String(), nullable=False),
        sa.Column("entry_kind", sa.String(), nullable=False),
        sa.Column("domain", sa.String(), nullable=True),
        sa.Column("source_type", sa.String(), nullable=True),
        sa.Column("trust_rank", sa.Float(), nullable=True),
        sa.Column("basis", sa.Text(), nullable=True),
        sa.Column("pattern", sa.String(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("modifier", sa.Float(), nullable=True),
        sa.Column("extractor_id", sa.String(), nullable=True),
        sa.Column("weight", sa.Float(), nullable=True),
    )
    op.create_index("ix_trust_lens_entries_lens_id", "trust_lens_entries", ["lens_id"])

    op.create_table(
        "trust_lens_adoptions",
        sa.Column("lens_name", sa.String(), primary_key=True),
        sa.Column("adopted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("adopted_by", sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("trust_lens_adoptions")
    op.drop_index("ix_trust_lens_entries_lens_id", table_name="trust_lens_entries")
    op.drop_table("trust_lens_entries")
    op.drop_index("ix_trust_lenses_name", table_name="trust_lenses")
    op.drop_table("trust_lenses")
