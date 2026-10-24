# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Alembic 055: stored content keys are re-keyed under the typographic fold.

The unit suite runs on ``create_all`` and never exercises a migration, so the
re-key is driven directly against a plain connection, as in the 053 test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from particles.core.duplicate_key import content_hash

_MIGRATION = Path("alembic/versions/055_typographic_content_key.py")


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("_m055", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conn() -> Any:
    """An in-memory DB holding only the columns the migration reads or writes."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE particles (id TEXT PRIMARY KEY, content TEXT NOT NULL, "
                "content_norm_hash TEXT)"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TABLE probe_verdicts (id INTEGER PRIMARY KEY, probe_kind TEXT NOT NULL, "
                "hash_a TEXT NOT NULL, hash_b TEXT NOT NULL)"
            )
        )
        yield connection


QUOTED = '"Quoted title" airs twice a month.'
PLAIN = "Another claim is here."
DASHED = "Options 1–3 apply."


def _seed(conn: Any, migration: Any) -> None:
    for pid, text in (("q", QUOTED), ("p", PLAIN), ("d", DASHED)):
        conn.execute(
            sa.text("INSERT INTO particles VALUES (:id, :c, :h)"),
            {"id": pid, "c": text, "h": migration._hash_v1(text)},
        )


def _keys(conn: Any) -> dict[str, str]:
    return dict(conn.execute(sa.text("SELECT id, content_norm_hash FROM particles")).fetchall())


def _ledger(conn: Any) -> dict[int, tuple[str, str]]:
    rows = conn.execute(sa.text("SELECT id, hash_a, hash_b FROM probe_verdicts")).fetchall()
    return {rid: (a, b) for rid, a, b in rows}


def test_rekeys_changed_rows_to_the_live_function(conn: Any) -> None:
    migration = _load_migration()
    _seed(conn, migration)

    rewritten, _ = migration.rekey(conn)

    assert rewritten == 2  # the plain claim's key does not move
    assert _keys(conn) == {
        pid: content_hash(t) for pid, t in (("q", QUOTED), ("p", PLAIN), ("d", DASHED))
    }


def test_ledger_follows_the_claims_and_symmetric_keys_stay_sorted(conn: Any) -> None:
    migration = _load_migration()
    _seed(conn, migration)
    old_q, old_p, old_d = (migration._hash_v1(t) for t in (QUOTED, PLAIN, DASHED))
    rows = [
        (1, "contradiction", *sorted((old_q, old_p))),
        (2, "update_slot", old_p, old_d),  # directional: order is meaning
        (3, "rung:duplicate_key", *sorted((old_d, old_p))),
        (4, "subject_link", "name-and-claim-hash", "candidates-hash"),
    ]
    for rid, kind, a, b in rows:
        conn.execute(
            sa.text("INSERT INTO probe_verdicts VALUES (:id, :k, :a, :b)"),
            {"id": rid, "k": kind, "a": a, "b": b},
        )

    _, verdicts = migration.rekey(conn)

    new_q, new_p, new_d = (content_hash(t) for t in (QUOTED, PLAIN, DASHED))
    ledger = _ledger(conn)
    assert verdicts == 3
    assert ledger[1] == tuple(sorted((new_q, new_p)))
    assert ledger[2] == (new_p, new_d)
    assert ledger[3] == tuple(sorted((new_d, new_p)))
    assert ledger[4] == ("name-and-claim-hash", "candidates-hash")


def test_idempotent(conn: Any) -> None:
    migration = _load_migration()
    _seed(conn, migration)
    migration.rekey(conn)
    once = (_keys(conn), _ledger(conn))
    assert migration.rekey(conn) == (0, 0)
    assert (_keys(conn), _ledger(conn)) == once


def test_downgrade_restores_the_old_keys(conn: Any) -> None:
    migration = _load_migration()
    _seed(conn, migration)
    before = _keys(conn)
    migration.rekey(conn)
    migration._rekey_particles(conn, migration._hash_v1)
    assert _keys(conn) == before
