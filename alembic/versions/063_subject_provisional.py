# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Provisional Subjects and the agent-deposit stamp (§1, §7).

Creates ``subject_provisional`` (one row per provisional Subject, deleted on
acceptance) and adds the nullable ``corpus_entries.deposit_role`` column. It
writes no rows: an upgraded store behaves as before until an agent principal
mints a Subject, and the operator's ``particles subjects
provisional-backfill`` is the only thing that marks an existing Subject.

Revision ID: 063
Revises: 062
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "063"
down_revision = "062"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subject_provisional",
        sa.Column("subject_id", sa.String(), primary_key=True),
        sa.Column("principal", sa.String(), nullable=False),
        sa.Column("minted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("shared_by", sa.String(), nullable=True),
        sa.Column("shared_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_subject_provisional_principal", "subject_provisional", ["principal"])
    with op.batch_alter_table("corpus_entries") as batch:
        batch.add_column(sa.Column("deposit_role", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("corpus_entries") as batch:
        batch.drop_column("deposit_role")
    op.drop_index("ix_subject_provisional_principal", table_name="subject_provisional")
    op.drop_table("subject_provisional")
