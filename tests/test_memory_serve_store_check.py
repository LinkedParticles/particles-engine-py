# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""``memory serve`` refuses a store it cannot serve, before the transport starts.

A fresh store answered every façade tool with ``no such table: subjects``, as a
tool error the agent sees and the operator does not. The server never creates
or migrates its store, so it checks before serving: no file, no schema, or a
store behind this install exits 2 with the command that fixes it. No model
call, no outside connection.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from particles.api.cli import app


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The default store pinned to a path under ``tmp_path``, not created."""
    from particles.config import reset_config

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PARTICLES_CONFIG", raising=False)
    db = tmp_path / "store" / "memory.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    reset_config()
    return db


def _serve() -> tuple[int, str, bool]:
    with patch("particles.mcp.memory_compat.main") as serve_main:
        result = CliRunner().invoke(app, ["memory", "serve"])
    return result.exit_code, result.stderr, serve_main.called


def test_a_missing_store_is_refused_and_not_created(store: Path) -> None:
    code, err, served = _serve()
    assert (code, served) == (2, False), err
    assert f"there is no store at {store}" in err
    assert "db init" in err
    assert not store.exists()


def test_a_store_with_no_schema_is_refused(store: Path) -> None:
    store.parent.mkdir(parents=True)
    sqlite3.connect(store).close()
    code, err, served = _serve()
    assert (code, served) == (2, False), err
    assert "the store has no Particles schema" in err
    assert "db init" in err
    assert "Traceback" not in err


def test_an_initialized_store_is_served(store: Path) -> None:
    init = CliRunner().invoke(app, ["db", "init"])
    assert init.exit_code == 0, init.output
    code, err, served = _serve()
    assert (code, served) == (0, True), err


def test_a_store_behind_this_install_is_refused(store: Path) -> None:
    from particles.db import SchemaState

    store.parent.mkdir(parents=True)
    sqlite3.connect(store).close()
    behind = SchemaState(stored=("0001",), unknown=(), pending=("0002",))

    async def _behind(handle: str) -> SchemaState:
        return behind

    with patch("particles.db.schema_state", _behind):
        code, err, served = _serve()
    assert (code, served) == (2, False), err
    assert "1 migration behind this install" in err
    assert "db init" in err
