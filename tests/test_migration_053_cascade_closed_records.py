# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Alembic 053: records the trust cascade closed as PROVENANCE_STALE are retracted.

The unit suite runs on ``create_all`` and never exercises a migration. This one
rewrites stored statuses, and the status pair it reads is shared with demoted
losers, so its selection is driven directly against a plain connection, the
way the 042 test does.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

_MIGRATION = Path("alembic/versions/053_cascade_closed_records_retracted.py")


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("_m053", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conn() -> Any:
    """An in-memory DB holding only the columns the migration reads or must keep."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE particles (id TEXT PRIMARY KEY, content TEXT NOT NULL, "
                "status TEXT NOT NULL, status_reason TEXT, retired_at DATETIME, "
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
    born_retired: bool = False,
    content: str = "a claim",
) -> None:
    conn.execute(
        sa.text(
            "INSERT INTO particles (id, content, status, status_reason, retired_at, "
            "born_retired) VALUES (:id, :content, :status, :reason, :retired_at, :born)"
        ),
        {
            "id": pid,
            "content": content,
            "status": status,
            "reason": reason,
            "retired_at": retired_at,
            "born": born_retired,
        },
    )


def _state(conn: Any) -> dict[str, tuple[str, str | None, str | None, bool]]:
    rows = conn.execute(
        sa.text("SELECT id, status, status_reason, retired_at, born_retired FROM particles")
    ).fetchall()
    return {
        pid: (status, reason, retired, bool(born)) for pid, status, reason, retired, born in rows
    }


def _seed(conn: Any) -> None:
    # Two records the cascade closed: a §6.6 record and a census record.
    for pid, text in (("r1", "INCONSISTENCY: conflict"), ("r2", "INCONSISTENCY: census")):
        _row(
            conn,
            pid,
            status="PROVENANCE_STALE",
            reason="CONFLICT_RESOLVED",
            born_retired=True,
            content=text,
        )
    # The same cascade's losers share the status pair. A quarantined loser
    # (reason flip) is born retired; a believed one was stamped leaving ACTIVE.
    _row(conn, "q1", status="PROVENANCE_STALE", reason="CONFLICT_RESOLVED", born_retired=True)
    _row(
        conn,
        "l1",
        status="PROVENANCE_STALE",
        reason="CONFLICT_RESOLVED",
        retired_at="2026-09-01 00:00:00",
    )
    # A record review already closed, and one still open.
    _row(
        conn,
        "r3",
        status="RETRACTED",
        reason="CONFLICT_RESOLVED",
        born_retired=True,
        content="INCONSISTENCY: reviewed",
    )
    _row(conn, "r4", status="INCONSISTENCY", born_retired=True, content="INCONSISTENCY: open")
    # A believed claim whose text happens to open with the prefix.
    _row(
        conn,
        "c1",
        status="PROVENANCE_STALE",
        reason="CONFLICT_RESOLVED",
        retired_at="2026-09-02 00:00:00",
        content="INCONSISTENCY: a quoted heading",
    )


def test_moves_only_cascade_closed_records(conn: Any) -> None:
    _seed(conn)
    before = _state(conn)

    moved = _load_migration().retract_cascade_closed_records(conn)

    after = _state(conn)
    assert moved == 2
    for pid in ("r1", "r2"):
        # Only the status moves: the reason, the NULL write-once retired_at and
        # born_retired (what the as-of lens reads) are kept.
        assert after[pid] == ("RETRACTED", "CONFLICT_RESOLVED", None, True)
    for pid in ("q1", "l1", "r3", "r4", "c1"):
        assert after[pid] == before[pid], pid


def test_idempotent(conn: Any) -> None:
    _seed(conn)
    migration = _load_migration()
    assert migration.retract_cascade_closed_records(conn) == 2
    once = _state(conn)
    assert migration.retract_cascade_closed_records(conn) == 0
    assert _state(conn) == once


def test_empty_store(conn: Any) -> None:
    assert _load_migration().retract_cascade_closed_records(conn) == 0
