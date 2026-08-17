"""Alembic 035 — the ``properties`` key rename, round-tripped.

The unit suite runs on ``create_all`` against an in-memory DB, so it never
exercises a migration (``alembic/AGENTS.md`` § Migration workflow). That is
tolerable for a migration that adds a column and fatal for one that **rewrites
operator data**, which this one does: it renames three keys inside every
``particles.properties_json``, and two of them are visibility levers. A bug
here does not raise — it silently changes which particles a store shows.

The migration exposes ``rewrite_properties_keys(bind, mapping)`` so the rewrite
runs against a plain connection. Loaded by path because ``alembic/versions`` is
not an importable package.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

_MIGRATION = Path("alembic/versions/035_prefix_extraction_properties_keys.py")


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("_m035", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conn() -> Any:
    """An in-memory DB holding only the column the migration touches."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text("CREATE TABLE particles (id TEXT PRIMARY KEY, properties_json TEXT)")
        )
        yield connection


def _insert(conn: Any, rows: dict[str, Any]) -> None:
    for pid, props in rows.items():
        conn.execute(
            sa.text("INSERT INTO particles (id, properties_json) VALUES (:id, :props)"),
            {
                "id": pid,
                "props": props if isinstance(props, str) or props is None else json.dumps(props),
            },
        )


def _read(conn: Any) -> dict[str, Any]:
    out = {}
    for pid, raw in conn.execute(sa.text("SELECT id, properties_json FROM particles")):
        out[pid] = raw
    return out


def test_renames_the_three_keys_and_leaves_everything_else(conn: Any) -> None:
    m = _load_migration()
    _insert(
        conn,
        {
            "p1": {"polarity": "DECLINED", "nmo:hasWeight": 0.75},
            "p2": {"scope": "DOCUMENT_META", "scope_action": "observe"},
            "p3": {"nmo:hasIssuer": "GDR"},
            "p4": None,
            "p5": "",
        },
    )

    m.rewrite_properties_keys(conn, m._RENAMES)
    after = _read(conn)

    assert json.loads(after["p1"]) == {"extraction:polarity": "DECLINED", "nmo:hasWeight": 0.75}
    assert json.loads(after["p2"]) == {
        "extraction:scope": "DOCUMENT_META",
        "extraction:scope_action": "observe",
    }
    # Untouched rows are not even rewritten — byte-identical, not just equal.
    assert after["p3"] == '{"nmo:hasIssuer": "GDR"}'
    assert after["p4"] is None
    assert after["p5"] == ""


def test_round_trips(conn: Any) -> None:
    """upgrade -> downgrade restores the original JSON exactly."""
    m = _load_migration()
    original = {
        "p1": '{"polarity": "HYPOTHETICAL", "scope": "DOCUMENT_META"}',
        "p2": '{"content:hasLanguage": "en"}',
    }
    _insert(conn, original)

    m.rewrite_properties_keys(conn, m._RENAMES)
    m.rewrite_properties_keys(conn, {new: old for old, new in m._RENAMES.items()})

    after = _read(conn)
    assert json.loads(after["p1"]) == json.loads(original["p1"])
    assert after["p2"] == original["p2"]


def test_survives_unparseable_and_non_dict_json(conn: Any) -> None:
    """Renaming keys is the job; repairing malformed rows is not."""
    m = _load_migration()
    _insert(conn, {"bad": "not json at all", "arr": "[1, 2, 3]"})

    m.rewrite_properties_keys(conn, m._RENAMES)

    after = _read(conn)
    assert after["bad"] == "not json at all"
    assert after["arr"] == "[1, 2, 3]"


def test_an_existing_prefixed_key_is_not_clobbered(conn: Any) -> None:
    """A row holding both spellings is malformed; the prefixed value is the one meant."""
    m = _load_migration()
    _insert(conn, {"p1": {"polarity": "DECLINED", "extraction:polarity": "HYPOTHETICAL"}})

    m.rewrite_properties_keys(conn, m._RENAMES)

    assert json.loads(_read(conn)["p1"]) == {"extraction:polarity": "HYPOTHETICAL"}


def test_the_migration_matches_the_application_constants(conn: Any) -> None:
    """The migration inlines its key names (frozen record); this pins them to the code.

    If a later rename moves the constants again, this fails and the author must
    decide deliberately whether a *new* migration is needed — rather than the
    two drifting silently.
    """
    from particles.extraction.polarity import POLARITY_KEY
    from particles.extraction.scope import SCOPE_ACTION_KEY, SCOPE_KEY

    m = _load_migration()
    assert m._RENAMES == {
        "polarity": POLARITY_KEY,
        "scope": SCOPE_KEY,
        "scope_action": SCOPE_ACTION_KEY,
    }
