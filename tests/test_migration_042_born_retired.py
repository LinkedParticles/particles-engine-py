# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Alembic 042: ``born_retired`` backfilled on rows stored before the column.

The unit suite runs on ``create_all`` and never exercises a migration. This
one decides which stored rows the as-of lens treats as never believed,
so its backfill is driven directly against a plain connection, the
way the 040 test does.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

_MIGRATION = Path("alembic/versions/042_born_retired.py")


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("_m042", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conn() -> Any:
    """An in-memory DB holding only the columns the migration touches."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE particles (id TEXT PRIMARY KEY, content TEXT NOT NULL, "
                "status TEXT NOT NULL, status_reason TEXT, retired_at DATETIME, "
                "provenance_json TEXT NOT NULL DEFAULT '[]', "
                "born_retired BOOLEAN NOT NULL DEFAULT 0)"
            )
        )
        yield connection


def _row(
    conn: Any,
    pid: str,
    *,
    status: str,
    reason: str | None = None,
    retired_at: str | None = None,
    content: str = "a claim",
    refs: tuple[str, ...] = (),
) -> None:
    provenance = [
        {"type": "PARTICLE", "corpus_entry_id": ref, "snapshot_id": ref} for ref in refs
    ] + [{"type": "SOURCE", "corpus_entry_id": "entry-1", "snapshot_id": "snap-1"}]
    conn.execute(
        sa.text(
            "INSERT INTO particles (id, content, status, status_reason, retired_at, "
            "provenance_json) VALUES (:id, :content, :status, :reason, :retired_at, :prov)"
        ),
        {
            "id": pid,
            "content": content,
            "status": status,
            "reason": reason,
            "retired_at": retired_at,
            "prov": json.dumps(provenance),
        },
    )


def _marked(conn: Any) -> set[str]:
    rows = conn.execute(sa.text("SELECT id FROM particles WHERE born_retired")).fetchall()
    return {pid for (pid,) in rows}


def test_marks_open_and_resolved_born_retired_rows(conn: Any) -> None:
    # An open conflict: the record and its CONFLICT_PENDING loser.
    _row(conn, "a1", status="ACTIVE")
    _row(conn, "b1", status="PROVENANCE_STALE", reason="CONFLICT_PENDING")
    _row(conn, "r1", status="INCONSISTENCY", content="INCONSISTENCY: x", refs=("a1", "b1"))
    # A PREFER_A review: loser flipped, record retracted, neither stamped.
    _row(conn, "a2", status="ACTIVE")
    _row(conn, "b2", status="PROVENANCE_STALE", reason="CONFLICT_RESOLVED")
    _row(
        conn,
        "r2",
        status="RETRACTED",
        reason="CONFLICT_RESOLVED",
        content="INCONSISTENCY: y",
        refs=("a2", "b2"),
    )
    # A DISCARD: the believed A is stamped, the loser B is not.
    _row(conn, "a3", status="RETRACTED", reason="CONFLICT_RESOLVED", retired_at="2026-09-01")
    _row(conn, "b3", status="RETRACTED", reason="CONFLICT_RESOLVED")
    _row(
        conn,
        "r3",
        status="PROVENANCE_STALE",
        reason="CONFLICT_RESOLVED",
        content="INCONSISTENCY: z",
        refs=("a3", "b3"),
    )

    marked = _load_migration().backfill_born_retired(conn)

    assert _marked(conn) == {"b1", "r1", "b2", "r2", "b3", "r3"}
    assert marked == 6


def test_leaves_believed_rows_unmarked(conn: Any) -> None:
    # A pre-029 retirement: unstamped, but named by no record as its B claim.
    _row(conn, "legacy", status="SUPERSEDED", reason="EXPLICIT_SUPERSESSION")
    # A record's A side, retired without a stamp: A was believed.
    _row(conn, "a", status="PROVENANCE_STALE", reason="CONFLICT_RESOLVED")
    # A B claim stamped on leaving ACTIVE was believed too.
    _row(conn, "b", status="RETRACTED", reason="CONFLICT_RESOLVED", retired_at="2026-09-01")
    _row(
        conn,
        "r",
        status="RETRACTED",
        reason="CONFLICT_RESOLVED",
        content="INCONSISTENCY: w",
        refs=("a", "b"),
    )
    # A pre-ADR-0117 record whose B ref dangles.
    _row(conn, "r-old", status="INCONSISTENCY", content="INCONSISTENCY: v", refs=("a", "gone"))

    _load_migration().backfill_born_retired(conn)

    assert _marked(conn) == {"r", "r-old"}


def test_idempotent(conn: Any) -> None:
    _row(conn, "b", status="PROVENANCE_STALE", reason="CONFLICT_PENDING")
    migration = _load_migration()
    migration.backfill_born_retired(conn)
    migration.backfill_born_retired(conn)
    assert _marked(conn) == {"b"}
