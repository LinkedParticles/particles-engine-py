"""As-of bitemporal read lens — write-once ``retired_at`` on particles.

One nullable column: the transaction-time end of a belief — the instant the
particle first left ``ACTIVE`` — stamped write-once at the
``update_particle_status`` choke point from this migration forward. Historical
rows stay NULL (recorded history cannot be backfilled); the reconstruction ladder dates them from successor pointers, the operator event
log, and ``valid_until``, fail-closed where unknown.

Additive and nullable: existing rows are unaffected, the core ``Particle``
model and SCHEMA_VERSION are untouched (``retired_at`` is storage metadata,
like the embedding — never interchanged, never in the schema artifacts).

Revision ID: 029
Revises: 028
"""

import sqlalchemy as sa

from alembic import op

revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("particles") as batch:
        batch.add_column(sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("particles") as batch:
        batch.drop_column("retired_at")
