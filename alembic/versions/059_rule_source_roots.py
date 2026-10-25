# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The enrolled rule-source roots.

Creates ``rule_source_roots``: one row per root ``rules sync`` or
``init claude-code`` resolved, keyed by its resolved path. The local refresh
tier's allowlist reads it instead of the refreshing process's working
directory.

There is no backfill: an enrolled corpus entry is not evidence the operator
chose its root, so the table starts empty and the first ``rules sync`` after
the upgrade fills it.

``downgrade()`` drops the table; nothing else reads it.

Revision ID: 059
Revises: 058
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "059"
down_revision = "058"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rule_source_roots",
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_by", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("path"),
    )


def downgrade() -> None:
    op.drop_table("rule_source_roots")
