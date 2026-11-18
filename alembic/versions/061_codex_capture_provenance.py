# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Codex source continuity and verified fork boundaries.

Revision ID: 061
Revises: 060
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "061"
down_revision = "060"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "codex_capture_provenance",
        sa.Column("capture_id", sa.String(), primary_key=True),
        sa.Column(
            "snapshot_id",
            sa.String(),
            sa.ForeignKey("snapshots.snapshot_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "codex_capture_heads",
        sa.Column("session_id", sa.String(), primary_key=True),
        sa.Column(
            "capture_id",
            sa.String(),
            sa.ForeignKey("codex_capture_provenance.capture_id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    op.create_table(
        "codex_prefix_proofs",
        sa.Column("session_id", sa.String(), primary_key=True),
        sa.Column("byte_offset", sa.Integer(), primary_key=True),
        sa.Column("ordinal", sa.Integer(), primary_key=True),
        sa.Column(
            "capture_id",
            sa.String(),
            sa.ForeignKey("codex_capture_provenance.capture_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("digest", sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("codex_prefix_proofs")
    op.drop_table("codex_capture_heads")
    op.drop_table("codex_capture_provenance")
