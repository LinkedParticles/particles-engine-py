"""Add nullable calibration_json column to extractor_records.

Revision ID: 013
Revises: 012
"""

import sqlalchemy as sa

from alembic import op

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "extractor_records",
        sa.Column("calibration_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("extractor_records", "calibration_json")
