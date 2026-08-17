"""Add domain_hint column to particles table (Extension B).

Revision ID: 007
Revises: 006
"""

import sqlalchemy as sa

from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("particles", sa.Column("domain_hint", sa.Text(), nullable=True))
    op.create_index("ix_particles_status_domain_hint", "particles", ["status", "domain_hint"])


def downgrade() -> None:
    op.drop_index("ix_particles_status_domain_hint", table_name="particles")
    op.drop_column("particles", "domain_hint")
