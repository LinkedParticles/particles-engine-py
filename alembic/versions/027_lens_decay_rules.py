"""Trust-lens decay-rule columns.

Add nullable ``half_life_days`` and ``floor`` columns to ``trust_lens_entries``
for the fourth lens layer — per-observer content-age decay. A
``decay_source_type`` / ``decay_url_pattern`` entry reuses the existing
``pattern`` column for the source_type string or URL regex and carries the
absolute ``(half_life_days, floor)`` pair here. Additive and nullable: existing
lens entries (statements / URL rules / extractor weights) leave both NULL and
are unaffected; a store with no decay-bearing lens behaves byte-for-byte as
before. SCHEMA_VERSION is unchanged — the lens artifact is Extension B, and the
particle interchange is untouched (the JSON Schema artifact for
``TrustLensDefinition`` does grow a ``decay_rules`` layer, synced in the
activation commit).

Revision ID: 027
Revises: 026
"""

import sqlalchemy as sa

from alembic import op

revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("trust_lens_entries") as batch:
        batch.add_column(sa.Column("half_life_days", sa.Float(), nullable=True))
        batch.add_column(sa.Column("floor", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("trust_lens_entries") as batch:
        batch.drop_column("floor")
        batch.drop_column("half_life_days")
