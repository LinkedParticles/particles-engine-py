"""Add document_supersession_json to corpus_entries (cap. 2, 1.29.0).

Engine-internal capture of a document's supersession relation (the ADR genre
adapter reads `supersedes:` / `superseded_by:` frontmatter). A nullable JSON
object — ``{"key", "supersedes", "superseded_by"}`` — keyed by document
identity so the §6.6 rung-1.5 prior can follow it transitively. Nullable, so
existing rows need no backfill (NULL ≡ no genre relation). No SCHEMA_VERSION
change: the column is on the corpus entry, not the particle, and is never
serialized into the particle interchange.

Revision ID: 024
Revises: 023
"""

import sqlalchemy as sa

from alembic import op

revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "corpus_entries",
        sa.Column("document_supersession_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("corpus_entries", "document_supersession_json")
