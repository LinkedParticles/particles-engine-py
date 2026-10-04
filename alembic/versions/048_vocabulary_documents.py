# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary documents: the predicate-profile carrier.

Two new tables. ``vocabulary_versions`` holds one row per materialised version
of a vocabulary document; rows are only appended, as the corpus entries behind
them are, and the current version of a name is its highest. ``vocabulary_adoptions``
holds one row per adoption: a document name, store-wide (an empty
``lens_name``) or riding one trust lens.

**No backfill.** An existing store starts with no document and no adoption, so
its profile book is empty and every route behaves exactly as before.

Revision ID: 048
Revises: 047
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "048"
down_revision = "047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "vocabulary_versions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("prefix", sa.String(), nullable=False),
        sa.Column("namespace", sa.String(), nullable=False),
        sa.Column("publisher", sa.String(), nullable=True),
        sa.Column("term_count", sa.Integer(), nullable=False),
        sa.Column("document_json", sa.Text(), nullable=False),
        sa.Column("corpus_entry_id", sa.String(), nullable=True),
        sa.Column("materialised_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", "version", name="uq_vocabulary_name_version"),
    )
    op.create_index("ix_vocabulary_versions_name", "vocabulary_versions", ["name"])
    op.create_table(
        "vocabulary_adoptions",
        sa.Column("vocabulary_name", sa.String(), primary_key=True),
        sa.Column("lens_name", sa.String(), primary_key=True),
        sa.Column("adopted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("adopted_by", sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("vocabulary_adoptions")
    op.drop_index("ix_vocabulary_versions_name", table_name="vocabulary_versions")
    op.drop_table("vocabulary_versions")
