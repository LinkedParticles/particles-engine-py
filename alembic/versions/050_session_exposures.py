# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Record what each session was shown at session start.

One new table, ``session_exposures``: one append-only row per SessionStart
that delivered context, holding the session id and time, the source, the
hook's action, the resolved project key, the observer-scope flag, and the
shown beliefs in order as a JSON array. It is a primary record and no rebuild
path clears it. No backfill: no past session can be reconstructed.

Revision ID: 050
Revises: 049
"""

import sqlalchemy as sa

from alembic import op

revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "session_exposures",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("action", sa.String(length=8), nullable=False),
        sa.Column("project_key", sa.String(), nullable=False),
        sa.Column("observer_scope_applied", sa.Boolean(), nullable=False),
        sa.Column("beliefs", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_session_exposures_session",
        "session_exposures",
        ["session_id", "recorded_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_session_exposures_session", table_name="session_exposures")
    op.drop_table("session_exposures")
