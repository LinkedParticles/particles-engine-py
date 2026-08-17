"""Add assertion_modality to particles (1.2.0).

The truth-aptness axis: the engine applies truth-semantics (§6.6 contradiction
resolution, L-SEM-01, L-IDX-01) only to FALSIFIABLE particles. NOT NULL with a
``server_default`` of ``FALSIFIABLE`` so existing rows backfill without a data
migration — old data is, by definition, falsifiable claims. No SCHEMA_VERSION
change: this is an additive optional field on the particle interchange schema
(the schema freeze holds).

Revision ID: 022
Revises: 021
"""

import sqlalchemy as sa

from alembic import op

revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "particles",
        sa.Column(
            "assertion_modality",
            sa.String(),
            nullable=False,
            server_default="FALSIFIABLE",
        ),
    )


def downgrade() -> None:
    op.drop_column("particles", "assertion_modality")
