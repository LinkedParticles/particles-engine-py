"""Add snapshots.extraction_started_at column (0.42.2).

Records when an extraction run claims a snapshot (writes IN_PROGRESS).
``extract --all-pending`` uses this to detect rows stranded mid-extraction
when the try/finally cleanup couldn't run (SIGKILL, segfault, oom). The
column is nullable so historical rows don't need a backfill — a NULL
``extraction_started_at`` paired with IN_PROGRESS is treated as orphaned
on the assumption that any genuine claim would have stamped the timestamp.

Revision ID: 016
Revises: 015
"""

import sqlalchemy as sa

from alembic import op

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "snapshots",
        sa.Column(
            "extraction_started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("snapshots", "extraction_started_at")
