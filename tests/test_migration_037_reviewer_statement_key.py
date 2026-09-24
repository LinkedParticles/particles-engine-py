# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Alembic 037 — reviewer trust statements re-keyed onto corpus entries.

The unit suite runs on ``create_all`` and never exercises a migration; this
one rewrites operator data (a trust-statement key that decides which source
wins a conflict), so its rewrite is driven directly against a plain
connection, the way ``test_migration_035_properties_keys.py`` does.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

_MIGRATION = Path("alembic/versions/037_rekey_reviewer_trust_statements.py")


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("_m037", _MIGRATION)
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
            sa.text("CREATE TABLE particles (id TEXT PRIMARY KEY, provenance_json TEXT)")
        )
        connection.execute(
            sa.text(
                "CREATE TABLE trust_statements (statement_id TEXT PRIMARY KEY, "
                "policy_provenance TEXT, source_ref_type TEXT, source_ref_value TEXT, basis TEXT)"
            )
        )
        yield connection


def _particle(conn: Any, pid: str, refs: list[dict[str, str]]) -> None:
    conn.execute(
        sa.text("INSERT INTO particles (id, provenance_json) VALUES (:id, :prov)"),
        {"id": pid, "prov": json.dumps(refs)},
    )


def _statement(
    conn: Any,
    sid: str,
    value: str,
    *,
    provenance: str = "REVIEWER_DERIVED",
    ref_type: str = "CORPUS_ENTRY",
    basis: str | None = "preferred",
) -> None:
    conn.execute(
        sa.text(
            "INSERT INTO trust_statements (statement_id, policy_provenance, source_ref_type, "
            "source_ref_value, basis) VALUES (:sid, :prov, :rtype, :value, :basis)"
        ),
        {"sid": sid, "prov": provenance, "rtype": ref_type, "value": value, "basis": basis},
    )


def _row(conn: Any, sid: str) -> tuple[str, str | None]:
    row = conn.execute(
        sa.text("SELECT source_ref_value, basis FROM trust_statements WHERE statement_id = :sid"),
        {"sid": sid},
    ).fetchone()
    assert row is not None
    return row[0], row[1]


def test_particle_keyed_statement_is_rekeyed_to_its_source_entry(conn: Any) -> None:
    m = _load_migration()
    _particle(
        conn,
        "p-a",
        [
            {"type": "PARTICLE", "corpus_entry_id": "p-other"},
            {"type": "SOURCE", "corpus_entry_id": "entry-1", "snapshot_id": "snap-1"},
        ],
    )
    _statement(conn, "s1", "p-a")

    assert m.rekey_reviewer_statements(conn) == 1
    value, basis = _row(conn, "s1")
    assert value == "entry-1"
    assert basis == "preferred [migration 037: re-keyed from particle p-a]"


def test_entry_keyed_and_non_reviewer_statements_are_untouched(conn: Any) -> None:
    m = _load_migration()
    _particle(conn, "p-a", [{"type": "SOURCE", "corpus_entry_id": "entry-1"}])
    # Already a corpus-entry key: no particle row matches, nothing to do.
    _statement(conn, "s-entry", "entry-1")
    # Operator-direct statements are keyed by the operator and never touched,
    # even when the value happens to be a particle id.
    _statement(conn, "s-op", "p-a", provenance="OPERATOR_DIRECT")
    # AUTHOR / SOURCE_TYPE tiers are not particle-keyed.
    _statement(conn, "s-author", "p-a", ref_type="AUTHOR")

    assert m.rekey_reviewer_statements(conn) == 0
    assert _row(conn, "s-entry") == ("entry-1", "preferred")
    assert _row(conn, "s-op") == ("p-a", "preferred")
    assert _row(conn, "s-author") == ("p-a", "preferred")


def test_particle_without_source_provenance_is_left_inert(conn: Any) -> None:
    """An agent-asserted or derived preferred claim names no corpus entry."""
    m = _load_migration()
    _particle(conn, "p-derived", [{"type": "PARTICLE", "corpus_entry_id": "p-premise"}])
    _statement(conn, "s1", "p-derived")

    assert m.rekey_reviewer_statements(conn) == 0
    assert _row(conn, "s1") == ("p-derived", "preferred")


def test_missing_particle_and_null_basis_are_tolerated(conn: Any) -> None:
    m = _load_migration()
    _particle(conn, "p-a", [{"type": "SOURCE", "corpus_entry_id": "entry-1"}])
    _statement(conn, "s-gone", "p-purged")
    _statement(conn, "s-null", "p-a", basis=None)

    assert m.rekey_reviewer_statements(conn) == 1
    assert _row(conn, "s-gone") == ("p-purged", "preferred")
    assert _row(conn, "s-null") == ("entry-1", " [migration 037: re-keyed from particle p-a]")


def test_downgrade_restores_the_particle_key_exactly(conn: Any) -> None:
    m = _load_migration()
    _particle(conn, "p-a", [{"type": "SOURCE", "corpus_entry_id": "entry-1"}])
    _statement(conn, "s1", "p-a")
    _statement(conn, "s-untouched", "entry-9")

    assert m.rekey_reviewer_statements(conn) == 1
    assert m.restore_particle_keys(conn) == 1
    assert _row(conn, "s1") == ("p-a", "preferred")
    assert _row(conn, "s-untouched") == ("entry-9", "preferred")
    # Idempotent: a second upgrade re-keys the same one row again.
    assert m.rekey_reviewer_statements(conn) == 1
