# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The memory façade writes only a store the stdio store rule checked.

Security review 2026-10-30-1, AB31.

With no ``--store``, ``memory serve`` applies the rule to the sole store
``mcp.write.enabled_stores`` names, and the façade writes only the store it
reads. Configuration only through the loader (``PARTICLES_CONFIG`` naming a
file under ``tmp_path``) and a temporary ``HOME``. No model call, no outside
connection.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    from particles.config import reset_config

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    cfg = tmp_path / "cfgdir" / "config.yaml"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text("network:\n  egress: none\n" + body, encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    reset_config()
    return cfg


# The stdio servers' store rule
# --------------------------------------------------------------------------- #


def _run_cli(args: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[int, list[Any]]:
    from typer.testing import CliRunner

    from particles.api.cli import app

    served: list[Any] = []
    monkeypatch.setattr("particles.mcp.memory_compat.main", lambda store=None: served.append(store))
    monkeypatch.setattr("particles.mcp.main", lambda *a, **k: served.append("native"))
    result = CliRunner().invoke(app, args)
    return result.exit_code, served


def test_memory_serve_refuses_a_relative_store_it_would_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``memory serve`` refuses to start when the store it writes is a relative SQLite path."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'default.db'}")
    _config(
        tmp_path,
        monkeypatch,
        "storage:\n  stores:\n    team: sqlite+aiosqlite:///team.db\n"
        "mcp:\n  write:\n    enabled_stores: [team]\n",
    )
    code, served = _run_cli(["memory", "serve"], monkeypatch)
    assert code != 0 and not served, (code, served)


def test_control_mcp_serve_refuses_the_same_relative_named_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the native server's start refuses the relative named store (passes)."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'default.db'}")
    _config(
        tmp_path,
        monkeypatch,
        "storage:\n  stores:\n    team: sqlite+aiosqlite:///team.db\n"
        "mcp:\n  write:\n    enabled_stores: [team]\n",
    )
    code, served = _run_cli(["mcp", "serve"], monkeypatch)
    assert code != 0 and not served, (code, served)


@pytest.fixture
def stub_embeddings() -> Generator[None, None, None]:
    from particles import embeddings as ep

    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


def test_a_facade_write_never_lands_in_a_store_the_working_directory_ships(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_embeddings: None
) -> None:
    """With no ``--store``, a façade write never lands in a store file under the session's cwd."""
    import mcp.types as types

    from particles.config import reset_config
    from particles.db import reset_engine

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'default.db'}")
    _config(
        tmp_path,
        monkeypatch,
        "storage:\n  stores:\n    team: sqlite+aiosqlite:///team.db\n"
        "mcp:\n  write:\n    enabled_stores: [team]\n",
    )
    clone = tmp_path / "clone"
    clone.mkdir()
    monkeypatch.chdir(clone)

    async def _tables(store: str) -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine(store).begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _go() -> Any:
        from particles.mcp.memory_compat import server

        await _tables("default")
        await _tables("team")  # the store a cloned repository ships in its root
        srv = server.build_server()
        call = srv.request_handlers[types.CallToolRequest]
        return await call(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="create_entities",
                    arguments={
                        "entities": [
                            {
                                "name": "Pluto",
                                "entityType": "planet",
                                "observations": ["Pluto is a dwarf planet"],
                            }
                        ]
                    },
                ),
            )
        )

    before = (clone / "team.db").stat().st_mtime_ns if (clone / "team.db").exists() else None
    asyncio.run(_tables("team"))
    shipped = (clone / "team.db").stat().st_size
    result = asyncio.run(_go())
    reset_engine()
    reset_config()
    import sqlite3

    rows = (
        sqlite3.connect(clone / "team.db").execute("select count(*) from particles").fetchone()[0]
    )
    assert before is None and shipped > 0
    assert not (not result.root.isError and rows > 0), (result.root, rows)
