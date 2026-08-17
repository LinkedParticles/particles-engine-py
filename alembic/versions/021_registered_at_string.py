"""Align extractor_records.registered_at column type with the ORM (String).

The ORM (particles/store/extractor_store.py) has always declared the column
as String and written ISO-8601 text (``registered_at.isoformat()``), reading
it back with ``datetime.fromisoformat``. Migration 006 declared the column
as DATETIME, so the stored values were already ISO text under SQLite's type
affinity — this migration only changes the declared type so autogenerate
stops proposing the alter. No data conversion occurs in either direction.

Revision ID: 021
Revises: 020
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("extractor_records") as batch_op:
        batch_op.alter_column(
            "registered_at",
            existing_type=sa.DateTime(timezone=True),
            type_=sa.String(),
            existing_nullable=False,
        )


def downgrade() -> None:
    # Hand-rolled recreate instead of batch_op.alter_column: batch mode CASTs
    # a column whose type affinity changes, and on SQLite
    # CAST('2026-06-10T12:34:56+00:00' AS DATETIME) numeric-truncates the ISO
    # string to 2026. Copying the values verbatim preserves them (they were
    # ISO text under the DATETIME declaration too, via SQLite type affinity).
    op.rename_table("extractor_records", "_extractor_records_old")
    op.create_table(
        "extractor_records",
        sa.Column("extractor_id", sa.String(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("applicability_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("trust_weight", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column("registered_by", sa.Text(), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("calibration_json", sa.Text(), nullable=True),
    )
    op.execute(
        "INSERT INTO extractor_records "
        "(extractor_id, name, version, applicability_json, trust_weight, "
        "registered_by, registered_at, calibration_json) "
        "SELECT extractor_id, name, version, applicability_json, trust_weight, "
        "registered_by, registered_at, calibration_json "
        "FROM _extractor_records_old"
    )
    op.drop_table("_extractor_records_old")
