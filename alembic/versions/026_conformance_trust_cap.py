"""Conformance trust-cap status column.

Add a nullable ``conformance_required_failure`` boolean to ``extractor_records``.
It records the last ``extractor conform`` run's verdict for the read-side trust
cap: ``NULL`` = never evaluated / no fixture (unknown), ``False`` = evaluated and
passed, ``True`` = an *evaluable* REQUIRED failure (fixtures produced particles
and a REQUIRED field fell short). The opt-in cap clamps the extractor's
*effective* trust weight when this is ``True``; the stored ``trust_weight`` is
never touched. Additive and nullable, so existing rows default to "unknown" and
behaviour is unchanged until an operator both runs conformance and enables the
cap. No SCHEMA_VERSION change — this is Extension A store state, not part of the
particle interchange.

Revision ID: 026
Revises: 025
"""

import sqlalchemy as sa

from alembic import op

revision = "026"
down_revision = "025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("extractor_records") as batch:
        batch.add_column(sa.Column("conformance_required_failure", sa.Boolean(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("extractor_records") as batch:
        batch.drop_column("conformance_required_failure")
