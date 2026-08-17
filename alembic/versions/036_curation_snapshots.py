"""Persisted curation-card collections.

Creates ``curation_snapshots``, the cache `GET /curation` and `particles curate`
read instead of running every finder per request. queue build was
measured at 172 s end-to-end on the 2026-08-02 dogfood store (13,424 cards
collected to return 7); the collection half moves here and only the cheap
session half stays live.

**Pure cache — no backfill, no data migration.** Every row is re-derivable by
running the finders again, so ``upgrade()`` creates an empty table and the
first read on any store builds and persists a collection (cold
start). ``downgrade()`` drops it outright: nothing else references these rows,
and losing them costs one rebuild.

Revision ID: 036
Revises: 035
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "curation_snapshots",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("built_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("semantic", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("semantic_degraded", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("per_kind_scope_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("cards_json", sa.Text(), nullable=False),
        sa.Column("card_count", sa.Integer(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("snapshot_id"),
    )
    # The only read pattern is "newest collection" plus the retention prune,
    # both ordered by built_at.
    op.create_index(
        "ix_curation_snapshots_built_at", "curation_snapshots", ["built_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_curation_snapshots_built_at", table_name="curation_snapshots")
    op.drop_table("curation_snapshots")
