# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Alembic 040: tag edges backfilled from ``particles.tags_json``.

The unit suite runs on ``create_all`` and never exercises a migration. This
one writes the index every tag-filtered read joins on, so its backfill is
driven directly against a plain connection, the way the 037 test does.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

_MIGRATION = Path("alembic/versions/040_backfill_particle_tag_edges.py")


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("_m040", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conn() -> Any:
    """An in-memory DB holding only the columns the migration touches."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(sa.text("CREATE TABLE particles (id TEXT PRIMARY KEY, tags_json TEXT)"))
        connection.execute(
            sa.text(
                "CREATE TABLE particle_tag_edges (particle_id TEXT, tag TEXT, "
                "PRIMARY KEY (particle_id, tag))"
            )
        )
        yield connection


def _particle(conn: Any, pid: str, tags_json: str | None) -> None:
    conn.execute(
        sa.text("INSERT INTO particles (id, tags_json) VALUES (:id, :tags)"),
        {"id": pid, "tags": tags_json},
    )


def _edges(conn: Any) -> set[tuple[str, str]]:
    rows = conn.execute(sa.text("SELECT particle_id, tag FROM particle_tag_edges")).fetchall()
    return {(pid, tag) for pid, tag in rows}


def test_backfills_missing_edges(conn: Any) -> None:
    m = _load_migration()
    _particle(conn, "p1", json.dumps(["coins", "ml", "coins"]))
    _particle(conn, "p2", None)
    assert m.backfill_tag_edges(conn) == 2
    assert _edges(conn) == {("p1", "coins"), ("p1", "ml")}


def test_keeps_existing_edges_and_is_idempotent(conn: Any) -> None:
    m = _load_migration()
    _particle(conn, "p1", json.dumps(["coins", "ml"]))
    conn.execute(sa.text("INSERT INTO particle_tag_edges VALUES ('p1', 'coins')"))
    assert m.backfill_tag_edges(conn) == 1
    assert m.backfill_tag_edges(conn) == 0
    assert _edges(conn) == {("p1", "coins"), ("p1", "ml")}


def test_skips_malformed_tags_json(conn: Any) -> None:
    m = _load_migration()
    _particle(conn, "p1", "not json")
    _particle(conn, "p2", json.dumps({"coins": True}))
    assert m.backfill_tag_edges(conn) == 0
    assert _edges(conn) == set()
