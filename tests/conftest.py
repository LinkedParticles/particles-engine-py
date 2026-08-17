"""Shared pytest fixtures — the Engine (private monorepo) half.

The Client-pure fixtures live in ``tests/_client_fixtures.py`` and are
re-exported below; both exported trees receive that module verbatim so the two
suites cannot drift (D4). Everything defined *here* touches Engine
modules (``particles.db``, the stores, ``particles.ingest``,
``particles.api.cli``) and therefore rides the Engine repo only.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

# Re-exported, not redefined: pytest picks the `pytest_configure` hook and the
# autouse fixtures up off this module's namespace. Keep the list exhaustive —
# a name dropped here silently disables that fixture for the whole suite.
from tests._client_fixtures import (  # noqa: F401
    no_embedding_model,
    no_env_leak,
    pytest_configure,
    reset_client_state,
    restore_logger_levels,
)


@pytest.fixture(autouse=True)
def clear_subject_cache(reset_client_state: None) -> None:  # noqa: F811
    """Engine half of the per-test global reset.

    The Client half — ``reset_config()`` + ``set_client(None)`` — is
    ``reset_client_state`` in ``tests/_client_fixtures.py``; it runs for both
    trees. This fixture adds the Engine-only globals on top, and takes the
    Client half as a parameter so the two halves keep the ordering the single
    pre-split fixture had (config reset first, then the Engine caches). pytest
    resolves fixtures by parameter name, so that parameter shadowing the
    re-exported name above is the idiom rather than a redefinition — hence the
    ``F811`` suppression.
    """
    from particles.ingest.subject_resolver import clear_cache

    # Importing the seam registers its reset hook: the
    # reset_config() in reset_client_state then clears the circuit
    # breaker and the probe-failure counter uniformly — no per-global conftest
    # poking needed.
    from particles.operations import _llm  # noqa: F401

    clear_cache()
    # Reset the CLI output settings context var: it is set by
    # configure_output() and, being a process-global context var, would otherwise
    # leak an explicit --progress/--quiet from one test into the next (e.g. into a
    # heartbeat no-op-off-a-TTY assertion).
    from particles.api.cli._output import _CURRENT, OutputSettings

    _CURRENT.set(OutputSettings())


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    # Maintained in one place — see particles/_orm_modules.py. Manually
    # listing modules here drifts whenever a new ORM module lands
    # (e.g. synthesis_cache); the central registry is the
    # single source of truth.
    import particles._orm_modules  # noqa: F401
    from particles.db import Base, get_engine, session_scope

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_scope() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    # Dispose the engine while the event loop is still alive. The autouse
    # ``reset_client_state`` fixture calls ``reset_config()`` →
    # ``reset_engine()`` between tests, which only nulls the cached engine —
    # it does not close the aiosqlite Connection. Without this dispose, the
    # Connection becomes unreachable after pytest-asyncio tears the loop
    # down, and aiosqlite's ``__del__`` raises
    # ``RuntimeError: Event loop is closed`` when it tries to schedule
    # cleanup work on the dead loop.
    await engine.dispose()


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """File-based SQLite for CLI tests.

    CLI commands wrap their async impl in ``asyncio.run(...)``, which spins a
    fresh event loop and opens its own session via ``session_scope()``. With
    ``:memory:`` SQLite, each connection gets its own database — state would
    not survive between CLI invocations within a single test. A file-based
    DB shares state across asyncio.run boundaries.

    ``PARTICLES_CONFIG`` is already overridden at session level by
    ``pytest_configure`` so the dev's local ``./config.yaml`` does not leak
    into any test (CLI or otherwise).
    """
    db_path = tmp_path / "cli.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))

    from particles.config import reset_config

    reset_config()

    async def _create_tables() -> None:
        # Maintained in one place — see particles/_orm_modules.py.
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create_tables())
    yield db_path

    # Dispose the engine before reset_config() drops the cached pointer.
    # See the matching note on the db_session fixture above — without this
    # the aiosqlite Connection outlives every loop it was bound to and its
    # __del__ raises RuntimeError: Event loop is closed.
    async def _dispose() -> None:
        from particles.db import get_engine

        await get_engine().dispose()

    asyncio.run(_dispose())
    reset_config()
