"""Add embedding_model_id to particles (0.56.0).

Tags each stored vector with the id of the embedding model that produced it so
the cosine query path can refuse to compare vectors across embedding spaces —
the same shape as the schema_version guard. Nullable so existing
rows need no backfill: a NULL is a legacy row, grandfathered by the query path
as the historical default model (all-MiniLM-L6-v2). No SCHEMA_VERSION change
(the marker is store-internal, not part of the particle interchange schema).

Revision ID: 019
Revises: 018
"""

import sqlalchemy as sa

from alembic import op

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("particles", sa.Column("embedding_model_id", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("particles", "embedding_model_id")
