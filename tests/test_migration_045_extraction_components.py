# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Alembic 045 — ``snapshots.extraction_components_json``.

The unit suite runs on ``create_all`` and never exercises a migration, so the
upgrade and downgrade are driven directly against a plain connection, with an
existing row that must survive both.
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

_MIGRATION = Path("alembic/versions/045_snapshot_extraction_components.py")


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("_m045", _MIGRATION)
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


def test_upgrade_adds_a_nullable_column_and_downgrade_drops_it(conn: Any) -> None:
    migration = _load_migration()
    assert (migration.revision, migration.down_revision) == ("045", "044")
    with Operations.context(MigrationContext.configure(conn)):
        migration.upgrade()
        assert "extraction_components_json" in _columns(conn)
        row = conn.execute(
            sa.text("SELECT snapshot_id, extraction_components_json FROM snapshots")
        ).one()
        assert tuple(row) == ("s1", None)
        conn.execute(sa.text("UPDATE snapshots SET extraction_components_json = '{}'"))

        migration.downgrade()

    assert "extraction_components_json" not in _columns(conn)
    assert conn.execute(sa.text("SELECT snapshot_id FROM snapshots")).scalar_one() == "s1"
