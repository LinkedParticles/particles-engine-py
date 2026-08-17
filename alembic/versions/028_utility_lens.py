"""Usefulness (outcome-learning) lens — utility events + lens utility-rule columns.

Two additive changes for the fourth-and-a-half lens layer:

1. A new ``utility_events`` table — the per-belief, store-local *utility
   evidence*: one row per ``(particle_id, session_id)`` the transcript-mining
   pass credits (the agent demonstrably acted on the belief in that session),
   with ``match_basis`` (``literal`` / ``behavioural``) and ``observed_at``.

2. Nullable ``half_life_uses_days`` and ``cap`` columns on
   ``trust_lens_entries`` for the portable ``utility_rules`` lens layer — a
   ``utility_default`` / ``utility_source_type`` / ``utility_url_pattern`` entry
   reuses the existing ``pattern`` / ``weight`` / ``floor`` columns and carries
   these two here.

Additive and nullable throughout: existing lens entries leave both new columns
NULL and are unaffected; a store with no utility evidence and no utility-bearing
lens ranks byte-for-byte as before (cold-start). SCHEMA_VERSION is
unchanged — the lens artifact is Extension B; the particle interchange is
untouched (the ``TrustLensDefinition`` JSON Schema grows a ``utility_rules``
layer, synced in the activation commit).

Revision ID: 028
Revises: 027
"""

import sqlalchemy as sa

from alembic import op

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "utility_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("particle_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("match_basis", sa.String(), nullable=False),
    )
    op.create_index("ix_utility_events_particle_id", "utility_events", ["particle_id"])
    op.create_index("ix_utility_events_session_id", "utility_events", ["session_id"])

    with op.batch_alter_table("trust_lens_entries") as batch:
        batch.add_column(sa.Column("half_life_uses_days", sa.Float(), nullable=True))
        batch.add_column(sa.Column("cap", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("trust_lens_entries") as batch:
        batch.drop_column("cap")
        batch.drop_column("half_life_uses_days")

    op.drop_index("ix_utility_events_session_id", table_name="utility_events")
    op.drop_index("ix_utility_events_particle_id", table_name="utility_events")
    op.drop_table("utility_events")
