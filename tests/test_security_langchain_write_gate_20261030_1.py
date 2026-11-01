# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The LangChain adapter's deposit writes only to a write-enabled store.

Security review 2026-10-30-1, the completeness critic's item.

``particles.integrations.langchain.ParticlesDepositTool`` calls
``get_backend().deposit_text(..., store=DEFAULT_STORE)``, and with no engine
configured that is ``LocalBackend.deposit_text``. The agent path there refuses
a store outside ``mcp.write.enabled_stores``, as the MCP ``deposit_text`` tool
does. This drives the backend call the adapter makes, on a scratch store whose
config (loaded from a file) leaves ``enabled_stores`` empty. ``langchain_core``
is not installed in the dev venv, so the backend call stands for the tool's
coroutine.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Generator
from pathlib import Path

import pytest


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
    from particles.config import reset_config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\nmcp:\n  write:\n    enabled_stores: []\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'lc.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    reset_config()
    _create_tables()
    yield tmp_path
    reset_config()


async def _entry_count() -> int:
    from sqlalchemy import func, select

    from particles.corpus.store import CorpusEntryRow
    from particles.db import get_engine, session_scope

    async with session_scope() as session:
        n = (await session.execute(select(func.count()).select_from(CorpusEntryRow))).scalar_one()
    await get_engine().dispose()
    return int(n)


def test_the_adapters_in_process_deposit_writes_nothing_to_a_store_writes_are_not_enabled_for(
    store: Path,
) -> None:
    """With mcp.write.enabled_stores empty, the deposit the LangChain tool makes is refused."""
    from particles.api.client import get_backend
    from particles.config import get_config
    from particles.db import DEFAULT_STORE

    assert get_config().mcp.write.enabled_stores == []
    backend = get_backend()

    async def _deposit() -> None:
        await backend.deposit_text(text="agent-chosen text", tags=None, store=DEFAULT_STORE)

    with contextlib.suppress(Exception):  # any refusal is the safe outcome
        asyncio.run(_deposit())
    assert asyncio.run(_entry_count()) == 0
