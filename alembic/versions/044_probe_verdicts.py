# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The probe-verdict ledger.

One new table, ``probe_verdicts``: every answer a pairwise LLM probe gave, so a
pair a probe already cleared is not asked again. A row is a record:
the verdict a model gave about two claim contents at one prompt version. It is
keyed by ``probe_kind``, ``prompt_hash`` and both claims' content hashes, and
carries the verdict, the model and the time. Rows are only appended; the newest
row for a key is the one read.

``id`` is an integer surrogate key because one key can hold more than one row
(a later run that answered differently). The read path looks rows up by the
full key, which the one composite index covers.

**No backfill.** An existing store starts with an empty ledger, and its first
run after the upgrade probes exactly as before.

Revision ID: 044
Revises: 043
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "probe_verdicts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("probe_kind", sa.String(length=32), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("hash_a", sa.String(length=64), nullable=False),
        sa.Column("hash_b", sa.String(length=64), nullable=False),
        sa.Column("verdict", sa.Boolean(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_probe_verdicts_key",
        "probe_verdicts",
        ["probe_kind", "prompt_hash", "hash_a", "hash_b"],
    )


def downgrade() -> None:
    op.drop_index("ix_probe_verdicts_key", table_name="probe_verdicts")
    op.drop_table("probe_verdicts")
