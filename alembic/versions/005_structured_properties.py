"""Structured particle properties and subject classification.

Adds:
  - particles.properties_json  — JSON dict of ontology-keyed structured data
  - subjects.subject_class     — Nomisma ontology class name for exporter template selection

Revision ID: 005
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "particles",
        sa.Column("properties_json", sa.Text(), nullable=True),
    )
    op.add_column(
        "subjects",
        sa.Column("subject_class", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("particles", "properties_json")
    op.drop_column("subjects", "subject_class")
