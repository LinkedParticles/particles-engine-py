# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The subject alias index.

Creates ``subject_aliases`` (one row per distinct folded alias per Subject,
keyed ``(subject_id, alias_key)``, with an index on ``alias_key``) and fills it
from ``subjects.aliases_json``, which stays the system of record. The index is
derived: ``particles subjects rebuild-aliases`` re-derives it the same way, so
an older install that writes aliases without it leaves drift that
``rebuild-aliases --check`` reports and a rebuild repairs.

The backfill runs in Python, so it folds exactly as the store does
(``str.lower()``; ``subject_store.alias_key``) on any dialect. The fold is
frozen here, as every migration's logic is.

``downgrade()`` drops the table; nothing else reads it.

Revision ID: 057
Revises: 056
"""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa

from alembic import op

revision = "057"
down_revision = "056"
branch_labels = None
depends_on = None

_CHUNK = 10_000


def backfill_alias_index(conn: Any) -> int:
    """Insert one index row per distinct folded alias of every Subject.

    Takes the connection rather than calling ``op.get_bind()`` so the test can
    drive it directly. Returns the rows written.
    """
    rows: list[dict[str, str]] = []
    for sid, aliases_json in conn.execute(sa.text("SELECT id, aliases_json FROM subjects")):
        seen: set[str] = set()
        for alias in json.loads(aliases_json or "[]"):
            key = alias.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"subject_id": sid, "alias_key": key, "alias": alias})
    insert = sa.text(
        "INSERT INTO subject_aliases (subject_id, alias_key, alias) "
        "VALUES (:subject_id, :alias_key, :alias)"
    )
    for i in range(0, len(rows), _CHUNK):
        conn.execute(insert, rows[i : i + _CHUNK])
    return len(rows)


def upgrade() -> None:
    op.create_table(
        "subject_aliases",
        sa.Column("subject_id", sa.String(), nullable=False),
        sa.Column("alias_key", sa.Text(), nullable=False),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("subject_id", "alias_key"),
    )
    op.create_index("ix_subject_aliases_alias_key", "subject_aliases", ["alias_key"])
    backfill_alias_index(op.get_bind())


def downgrade() -> None:
    op.drop_index("ix_subject_aliases_alias_key", table_name="subject_aliases")
    op.drop_table("subject_aliases")
