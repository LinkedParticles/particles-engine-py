# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""No route holds SQLite's write lock while the store writer lock is free.

Security review 2026-10-30-1, AB4.

The agent ``POST /particles/assert`` route writes its subjects, provenance
deposit, author-trust seed and the reconcile ladder's insert inside the store
writer lock, so a cooperating writer queues on that lock rather than
waiting out ``busy_timeout``. Since security review 2026-10-30-2 (AC1) the
ladder's model-calling half runs before the lock, and holds no SQLite write
lock either. Scratch
store and key file under tmp_path; configuration through the loader; no model
call and no outside connection.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

LEGACY = "z69-legacy-operator-key"


def _create_tables() -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    asyncio.run(_go())


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles import embeddings as ep
    from particles.api import _keys
    from particles.config import reset_config

    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    db = tmp_path / "z69.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", LEGACY)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\nmcp:\n  write:\n    enabled_stores: [default]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    yield db
    ep.set_embedding_model(original)
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


def _other_writer_can_write(db: Path) -> bool:
    conn = sqlite3.connect(str(db), timeout=0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def test_agent_assert_holds_no_sqlite_write_lock_outside_the_writer_lock(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Safe property: while an agent assert runs its reconcile ladder (the
    step that calls the model), either the store writer lock is held or no
    other connection is kept from writing."""
    import particles.db as dbmod
    import particles.ingest.pipeline as pipeline
    from particles.api import _keys
    from particles.api._keys import Role

    token = _keys.mint(Role.AGENT, "agent:z69")[0]
    seen: dict[str, Any] = {}
    # The ladder's model-calling half (security review 2026-10-30-2, AC1).
    real = pipeline.gather_reconcile

    async def _probe(session: Any, particle: Any, **kwargs: Any) -> Any:
        lock = dbmod._async_write_locks.get("default")
        seen["advisory_held"] = bool(lock is not None and lock.locked())
        seen["other_can_write"] = _other_writer_can_write(store)
        return await real(session, particle, **kwargs)

    monkeypatch.setattr(pipeline, "gather_reconcile", _probe)
    resp = _client().post(
        "/particles/assert",
        json={
            "content": "The nightly job runs at three.",
            "subject_names": ["nightly job"],
            "confidence": 0.8,
            "source_excerpt": "The nightly job runs at three in the morning.",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    assert "other_can_write" in seen, "the reconcile step did not run"
    assert seen["advisory_held"] or seen["other_can_write"], (
        "SQLite's write lock was held across the reconcile ladder while the "
        "store writer lock was free"
    )


def test_control_the_probe_sees_the_writer_lock_when_it_is_held(store: Path) -> None:
    """Control: the advisory-lock probe the test above reads is live in this setup."""
    import particles.db as dbmod

    async def _go() -> bool:
        async with dbmod.write_lock():
            lock = dbmod._async_write_locks.get("default")
            return bool(lock is not None and lock.locked())

    assert asyncio.run(_go())
    assert _other_writer_can_write(store)
