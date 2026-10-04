# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Alembic 047 — ``snapshots.resume_whole``.

The unit suite runs on ``create_all`` and never exercises a migration, so the
upgrade and downgrade are driven directly against a plain connection, with an
existing row that must survive both and read as unmarked.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

_MIGRATION = Path("alembic/versions/047_snapshot_resume_whole.py")


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("_m047", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conn() -> Iterator[Any]:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text("CREATE TABLE snapshots (snapshot_id TEXT PRIMARY KEY, content_hash TEXT)")
        )
        connection.execute(sa.text("INSERT INTO snapshots VALUES ('s1', 'h1')"))
        yield connection


def _columns(conn: Any) -> set[str]:
    return {c["name"] for c in sa.inspect(conn).get_columns("snapshots")}


def test_upgrade_adds_an_unmarked_column_and_downgrade_drops_it(conn: Any) -> None:
    migration = _load_migration()
    assert (migration.revision, migration.down_revision) == ("047", "046")
    with Operations.context(MigrationContext.configure(conn)):
        migration.upgrade()
        assert "resume_whole" in _columns(conn)
        row = conn.execute(sa.text("SELECT snapshot_id, resume_whole FROM snapshots")).one()
        assert row[0] == "s1"
        assert not row[1]
        conn.execute(sa.text("UPDATE snapshots SET resume_whole = 1"))

        migration.downgrade()

    assert "resume_whole" not in _columns(conn)
    assert conn.execute(sa.text("SELECT snapshot_id FROM snapshots")).scalar_one() == "s1"
