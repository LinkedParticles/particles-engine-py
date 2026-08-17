"""Add contributors_json to particles, subjects, corpus_entries (0.55.0).

Extension D/E contributor attribution: a nullable JSON list of
``{id, role, at}`` ``ContributorRef`` objects on the three attributable models.
All columns are nullable so existing rows need no backfill — a NULL
``contributors_json`` means "no contributor attribution recorded" (≡ None ≡ []).
No SCHEMA_VERSION change (the field is additive and Optional; the version stays
1.0.0 until the guard is made major-aware).

Revision ID: 018
Revises: 017
"""

import sqlalchemy as sa

from alembic import op

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None

_TABLES = ("particles", "subjects", "corpus_entries")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(table, sa.Column("contributors_json", sa.Text(), nullable=True))


def downgrade() -> None:
    for table in _TABLES:
        op.drop_column(table, "contributors_json")
