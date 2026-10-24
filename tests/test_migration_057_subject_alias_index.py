# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Alembic 057: the subject alias index is backfilled from ``aliases_json``.

The unit suite runs on ``create_all`` and never exercises a migration, so the
backfill is driven directly against a plain connection, the way the 053 test
does. Its output must equal what the store's own rebuild derives.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from particles.store.subject_store import _alias_index_rows

_MIGRATION = Path("alembic/versions/057_subject_alias_index.py")


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("_m057", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conn() -> Any:
    """An in-memory DB holding the columns the backfill reads and the new table."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text("CREATE TABLE subjects (id TEXT PRIMARY KEY, aliases_json TEXT)")
        )
        connection.execute(
            sa.text(
                "CREATE TABLE subject_aliases (subject_id TEXT NOT NULL, alias_key TEXT NOT NULL,"
                " alias TEXT NOT NULL, PRIMARY KEY (subject_id, alias_key))"
            )
        )
        yield connection


_SUBJECTS = {
    "s1": ["user", "User", "the speaker"],
    "s2": [],
    "s3": ["Straße", "ÉCOLE"],
    "s4": ["Mercury"],
    "s5": ["mercury"],
}


def test_backfill_matches_the_store_rebuild(conn: Any) -> None:
    for sid, aliases in _SUBJECTS.items():
        conn.execute(
            sa.text("INSERT INTO subjects (id, aliases_json) VALUES (:id, :a)"),
            {"id": sid, "a": json.dumps(aliases)},
        )

    written = _load_migration().backfill_alias_index(conn)

    rows = set(conn.execute(sa.text("SELECT subject_id, alias_key, alias FROM subject_aliases")))
    expected = {
        (r["subject_id"], r["alias_key"], r["alias"])
        for sid, aliases in _SUBJECTS.items()
        for r in _alias_index_rows(sid, aliases)
    }
    assert rows == expected
    assert written == len(expected) == 6
    # The Python fold, not SQLite's ASCII-only lower().
    assert ("s3", "école", "ÉCOLE") in rows
