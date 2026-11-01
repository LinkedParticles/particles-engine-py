# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""A mutating route commits under the store's writer lock.

Security review 2026-10-30-1, AB4 and AB68.

The store writer lock serialises the engine's writes with the host's CLI
writers. Each test records, at every ``AsyncSession.commit`` a request makes,
whether the store's cross-process writer lock (the file lock ``write_lock``
takes) is held. A scratch store and key file under ``tmp_path``; configuration
only through the loader; model and embedding seams stubbed; no outside call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

LEGACY = "z69-legacy-operator-key-audit10"


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
    from particles.api import _keys
    from particles.config import reset_config

    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\nmcp:\n  write:\n    enabled_stores: [default]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'z69.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", LEGACY)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    yield keydir
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


@pytest.fixture
def commits(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """One entry per commit: whether the default store's file writer lock was held."""
    from sqlalchemy.ext.asyncio import AsyncSession

    import particles.db as db

    seen: list[bool] = []
    real = AsyncSession.commit

    async def _commit(self: AsyncSession) -> None:
        lock = db._file_write_locks.get("default")
        seen.append(bool(lock is not None and lock.is_locked))
        await real(self)

    monkeypatch.setattr(AsyncSession, "commit", _commit)
    return seen


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    import particles.ingest.subject_resolver as sr
    from particles import embeddings as ep

    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


def _ok(resp: Any) -> None:
    assert resp.status_code < 300, (resp.status_code, resp.text)


def test_operator_subject_create_commits_under_the_writer_lock(
    store: Path, commits: list[bool]
) -> None:
    """``POST /subjects`` commits its new Subject while holding the store's writer lock."""
    _ok(
        _client().post(
            "/subjects",
            json={"canonical_name": "Acme Widget"},
            headers={"Authorization": f"Bearer {LEGACY}"},
        )
    )
    assert commits and all(commits), commits


def test_link_dismiss_commits_under_the_writer_lock(store: Path, commits: list[bool]) -> None:
    """``POST /corpus/links/dismiss`` commits its suppression while holding the writer lock."""
    _ok(
        _client().post(
            "/corpus/links/dismiss",
            json={"url": "https://example.org/a"},
            headers={"Authorization": f"Bearer {LEGACY}"},
        )
    )
    assert commits and all(commits), commits


@pytest.mark.parametrize("caller", ["agent:ci-bot", "mcp:claude-code", "claude-code-hook"])
def test_agent_assert_commits_under_the_writer_lock(
    store: Path, commits: list[bool], stubs: None, tmp_path: Path, caller: str
) -> None:
    """``POST /particles/assert`` under an agent key commits its belief under the writer lock.

    Driven for a plain agent key, the shared ``mcp:claude-code`` principal the
    HTTP MCP server presents, and a harvester key with recorded roots (security
    review 2026-10-30-1, Z72).
    """
    from particles.api import _keys
    from particles.api._keys import Role

    if caller == "claude-code-hook":
        memory, archive = tmp_path / "memory-root", tmp_path / "archive-root"
        memory.mkdir()
        archive.mkdir()
        token = _keys.mint(
            Role.AGENT,
            caller,
            reserved_ok=True,
            memory_roots=[str(memory.resolve())],
            archive_roots=[str(archive.resolve())],
        )[0]
    else:
        token = _keys.mint(Role.AGENT, caller, reserved_ok=caller != "agent:ci-bot")[0]
    _ok(
        _client().post(
            "/particles/assert",
            json={
                "content": "The widget ships in blue.",
                "subject_names": ["Widget"],
                "confidence": 0.7,
                "source_excerpt": "it ships in blue",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
    )
    assert commits and all(commits), commits


def test_control_the_text_deposit_commits_under_the_writer_lock(
    store: Path, commits: list[bool]
) -> None:
    """Control for the recorder: ``/corpus/deposit/text`` commits under the lock (passes)."""
    _ok(
        _client().post(
            "/corpus/deposit/text",
            json={"text": "an operator note"},
            headers={"Authorization": f"Bearer {LEGACY}"},
        )
    )
    assert commits and all(commits), commits
