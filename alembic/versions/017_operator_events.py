"""Add operator event log tables.

``operator_events`` + ``operator_event_refs`` — an append-only audit of
operator decisions (retract / split / merge / alias / confirm / unlink /
trust / review / links / tags). This is storage-layer metadata, not particle
schema: it does not touch ``SCHEMA_VERSION``.

Revision ID: 017
Revises: 016
"""

import sqlalchemy as sa

from alembic import op

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "operator_events",
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
    )
    op.create_index("ix_operator_events_type", "operator_events", ["event_type"])
    op.create_index("ix_operator_events_occurred", "operator_events", ["occurred_at"])

    op.create_table(
        "operator_event_refs",
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("ref_kind", sa.String(), primary_key=True),
        sa.Column("ref_id", sa.String(), primary_key=True),
    )
    op.create_index("ix_event_refs_record", "operator_event_refs", ["ref_kind", "ref_id"])


def downgrade() -> None:
    op.drop_index("ix_event_refs_record", table_name="operator_event_refs")
    op.drop_table("operator_event_refs")
    op.drop_index("ix_operator_events_occurred", table_name="operator_events")
    op.drop_index("ix_operator_events_type", table_name="operator_events")
    op.drop_table("operator_events")
