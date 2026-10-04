# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for particles/db.py — the lazy engine + reset_config integration.

These tests prove that ``storage.database_url`` changes after ``reset_config()``
actually take effect, which was the M2 architecture-review finding: the engine
was previously bound at module-import time and could not be retargeted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles import db
from particles.config import reset_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.store.particle_store import get_particles_by_ids, insert_particle
from tests._write_probe import store_accepts_a_writer


@pytest.fixture(autouse=True)
def _reset_engine_around_each_test() -> Generator[None, None, None]:
    """Reset engine/config state around every test in this module."""
    db.reset_engine()
    yield
    db.reset_engine()
    reset_config()


def test_get_engine_is_lazy_and_cached() -> None:
    db.reset_engine()
    first = db.get_engine()
    second = db.get_engine()
    assert first is second, "get_engine() must cache the engine within a config generation"


def test_reset_engine_discards_cached_engine_and_factory() -> None:
    factory_before = db.get_session_factory()
    engine_before = db.get_engine()
    db.reset_engine()
    factory_after = db.get_session_factory()
    engine_after = db.get_engine()
    assert engine_before is not engine_after
    assert factory_before is not factory_after


def test_engine_url_reflects_database_url_after_reset_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting DATABASE_URL then calling reset_config() retargets the engine."""
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./url-one.db")
    reset_config()
    assert str(db.get_engine().url) == "sqlite+aiosqlite:///./url-one.db"

    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./url-two.db")
    reset_config()
    assert str(db.get_engine().url) == "sqlite+aiosqlite:///./url-two.db"


def test_unknown_store_error_is_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    """F29: the unknown-store error must not enumerate configured store handles.

    The MCP ``particles://digest/{store}`` resource surfaces this error to
    clients; listing the known handles is a minor info disclosure. The message
    is generic and the exception type stays ``KeyError`` for callers that match
    on it.
    """
    from particles.config import ParticlesConfig, StorageConfig

    cfg = ParticlesConfig(
        storage=StorageConfig(
            database_url="sqlite+aiosqlite:///:memory:",
            stores={
                "team-secret-alpha": "sqlite+aiosqlite:///./a.db",
                "team-secret-beta": "sqlite+aiosqlite:///./b.db",
            },
        )
    )
    # db.py binds get_config at module top, so patch the module's binding.
    monkeypatch.setattr(db, "get_config", lambda: cfg)

    with pytest.raises(KeyError) as exc_info:
        db._resolve_store_dsn("does-not-exist")

    # str(KeyError) wraps the message in quotes; assert the configured handles
    # are absent regardless of wrapping.
    message = str(exc_info.value)
    assert "does-not-exist" in message
    assert "team-secret-alpha" not in message
    assert "team-secret-beta" not in message
    assert "known stores" not in message.lower()


def test_reset_config_also_resets_engine() -> None:
    """reset_config() must invalidate the cached engine.

    Regression guard: before M2, reset_config() only reset the config singleton;
    the engine kept its original URL forever.
    """
    engine_before = db.get_engine()
    reset_config()
    engine_after = db.get_engine()
    assert engine_before is not engine_after


def test_create_tables_preserves_existing_loggers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """create_tables() runs Alembic, whose fileConfig() must not disable
    already-created loggers.

    Regression guard: ``create_tables()`` runs ``alembic upgrade head`` inside
    the engine's FastAPI lifespan, and ``alembic/env.py`` calls ``fileConfig``.
    With the default ``disable_existing_loggers=True``, every engine start
    silenced the uvicorn + ``particles.*`` loggers (no "request start"/"request
    end" lines, no SQLITE_BUSY WARNING) — defeating log-based observability.
    The fix passes ``disable_existing_loggers=False``.
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'logtest.db'}")
    reset_config()

    # A logger that already exists before create_tables() runs — exactly the
    # lifespan situation, where uvicorn + particles.* loggers are already set up.
    app_logger = logging.getLogger("particles.api.app")
    app_logger.disabled = False

    asyncio.run(db.create_tables())

    assert not app_logger.disabled, (
        "create_tables() (Alembic fileConfig) disabled an existing logger; "
        "alembic/env.py must pass disable_existing_loggers=False"
    )


def test_session_scope_yields_session_from_current_factory() -> None:
    """session_scope() must use whatever factory get_session_factory() returns now."""
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession

    async def _open_one() -> AsyncSession:
        async with db.session_scope() as session:
            return session

    session = asyncio.run(_open_one())
    assert isinstance(session, AsyncSession)


# --- SQLite-busy counter (Phase 2) -----------------------


def test_record_sqlite_busy_increments_on_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3
    from unittest.mock import MagicMock

    counter = MagicMock()
    monkeypatch.setattr(db, "_sqlite_busy_counter", counter)

    ctx = MagicMock()
    ctx.original_exception = sqlite3.OperationalError("database is locked")
    db._record_sqlite_busy(ctx)
    counter.add.assert_called_once_with(1)


def test_record_sqlite_busy_ignores_non_lock_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3
    from unittest.mock import MagicMock

    counter = MagicMock()
    monkeypatch.setattr(db, "_sqlite_busy_counter", counter)

    ctx = MagicMock()
    ctx.original_exception = sqlite3.OperationalError("no such table: particles")
    db._record_sqlite_busy(ctx)
    counter.add.assert_not_called()


def test_record_sqlite_busy_tolerates_missing_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    counter = MagicMock()
    monkeypatch.setattr(db, "_sqlite_busy_counter", counter)

    ctx = MagicMock()
    ctx.original_exception = None
    db._record_sqlite_busy(ctx)
    counter.add.assert_not_called()


def test_get_engine_registers_busy_handler_for_sqlite() -> None:
    from sqlalchemy import event

    db.reset_engine()
    engine = db.get_engine()
    assert event.contains(engine.sync_engine, "handle_error", db._record_sqlite_busy)


def test_handle_error_hook_fires_on_async_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handle_error listener must actually fire on the async SQLite path.

    Proves the wiring (the SQLAlchemy ``handle_error`` event propagates through
    the AsyncEngine's sync_engine, so a real ``database is locked`` would reach
    :func:`db._record_sqlite_busy`). We trigger a deterministic, non-locking
    OperationalError (a missing table) and assert the hook ran; the lock-vs-other
    branch itself is covered by the unit tests above.
    """
    import asyncio

    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    db.reset_engine()
    calls: list[object] = []
    original = db._record_sqlite_busy

    def _spy(context: object) -> None:
        calls.append(context)
        original(context)

    # Patch before get_engine() so the listener binds the spy (get_engine reads
    # the module global at registration time).
    monkeypatch.setattr(db, "_record_sqlite_busy", _spy)

    async def _trigger() -> None:
        engine = db.get_engine()
        async with engine.connect() as conn:
            with pytest.raises(OperationalError):
                await conn.execute(text("SELECT * FROM does_not_exist"))

    asyncio.run(_trigger())
    assert calls, "handle_error hook did not fire on the async SQLite engine"


# --- Cross-process write lock -------------------------------------


def test_sqlite_file_path_classifies_memory_file_and_postgres() -> None:
    assert db._sqlite_file_path("sqlite+aiosqlite:///./particles.db") == "./particles.db"
    assert db._sqlite_file_path("sqlite+aiosqlite:////tmp/x.db") == "/tmp/x.db"
    assert db._sqlite_file_path("sqlite+aiosqlite:///:memory:") is None
    assert db._sqlite_file_path("sqlite://") is None
    assert db._sqlite_file_path("postgresql+asyncpg://h/db") is None


def test_write_lock_is_noop_for_memory_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    reset_config()
    assert not db._store_is_file_sqlite("default")

    async def _use() -> None:
        async with db.write_lock():
            pass

    asyncio.run(_use())  # must not raise / create a lockfile
    assert not db._file_write_locks  # no FileLock instantiated for the no-op path


def test_write_lock_acquires_and_releases_for_file_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    reset_config()
    assert db._store_is_file_sqlite("default")

    async def _use() -> None:
        async with db.write_lock():
            assert db._file_write_locks["default"].is_locked

    asyncio.run(_use())
    assert not db._file_write_locks["default"].is_locked  # released on exit


def test_write_lock_disabled_is_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import asyncio

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("PARTICLES_WRITE_LOCK_ENABLED", "false")
    reset_config()

    async def _use() -> None:
        async with db.write_lock():
            pass

    asyncio.run(_use())
    assert not db._file_write_locks  # disabled ⇒ no FileLock created


def test_write_lock_serializes_concurrent_writers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two concurrent writers must not interleave their critical sections."""
    import asyncio

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    reset_config()
    order: list[tuple[str, int]] = []

    async def worker(n: int) -> None:
        async with db.write_lock():
            order.append(("enter", n))
            await asyncio.sleep(0.02)
            order.append(("exit", n))

    async def main() -> None:
        await asyncio.gather(worker(1), worker(2))

    asyncio.run(main())
    assert order in (
        [("enter", 1), ("exit", 1), ("enter", 2), ("exit", 2)],
        [("enter", 2), ("exit", 2), ("enter", 1), ("exit", 1)],
    ), order


def test_write_lock_translates_filelock_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    from filelock import Timeout as FileLockTimeout

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    reset_config()

    def _raise(*_a: object, **_k: object) -> None:
        raise FileLockTimeout("held by another process")

    monkeypatch.setattr(db.FileLock, "acquire", _raise)

    async def _use() -> None:
        async with db.write_lock():
            pass

    with pytest.raises(db.WriteLockTimeout):
        asyncio.run(_use())


def test_session_scope_write_acquires_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    reset_config()

    async def _open() -> tuple[AsyncSession, bool]:
        async with db.session_scope(write=True) as session:
            return session, db._file_write_locks["default"].is_locked

    session, locked_during = asyncio.run(_open())
    assert isinstance(session, AsyncSession)
    assert locked_during  # the lock was held inside the write session
    assert not db._file_write_locks["default"].is_locked  # released after


def test_get_write_session_dep_holds_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The engine-side write dependency holds the lock for the request, then releases."""
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    reset_config()

    async def _drive() -> tuple[AsyncSession, bool]:
        agen = db.get_write_session()
        session = await agen.__anext__()
        locked = db._file_write_locks["default"].is_locked
        await agen.aclose()  # runs the finally → releases the lock
        return session, locked

    session, locked_during = asyncio.run(_drive())
    assert isinstance(session, AsyncSession)
    assert locked_during
    assert not db._file_write_locks["default"].is_locked


# ---------------------------------------------------------------------------
# Missing SQLite parent directory: created on first use, or one clear error
# ---------------------------------------------------------------------------


def test_create_tables_creates_missing_sqlite_parent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLite never creates a database file's directory; ``create_tables`` must.

    Regression: ``particles db init`` against a fresh ``DATABASE_URL`` path used
    to die inside Alembic with ``unable to open database file``.
    """
    db_file = tmp_path / "missing" / "nested" / "particles.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    reset_config()

    asyncio.run(db.create_tables())

    assert db_file.parent.is_dir()
    assert db_file.is_file()


def test_get_engine_creates_missing_sqlite_parent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every entry point benefits, not just ``db init``: the engine factory creates it too."""
    db_file = tmp_path / "missing" / "nested" / "particles.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    reset_config()

    db.get_engine()

    assert db_file.parent.is_dir()


def test_ensure_sqlite_parent_dir_ignores_memory_and_postgres(tmp_path: Path) -> None:
    for url in (
        "sqlite+aiosqlite:///:memory:",
        "sqlite://",
        "postgresql+asyncpg://user@host/db",
    ):
        db._ensure_sqlite_parent_dir(url)  # must not raise or touch the filesystem


def test_ensure_sqlite_parent_dir_raises_one_line_error_when_blocked(tmp_path: Path) -> None:
    """A file where the directory should be yields a clean, path-naming error."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")

    with pytest.raises(db.StoreDirectoryError) as excinfo:
        db._ensure_sqlite_parent_dir(f"sqlite+aiosqlite:///{blocker / 'particles.db'}")
    message = str(excinfo.value)
    assert str(blocker) in message
    assert "a file already exists" in message
    assert "\n" not in message

    with pytest.raises(db.StoreDirectoryError, match=str(blocker / "sub")):
        db._ensure_sqlite_parent_dir(f"sqlite+aiosqlite:///{blocker / 'sub' / 'particles.db'}")


# ---------------------------------------------------------------------------
# write_transaction
# ---------------------------------------------------------------------------


def _claim(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
        asserted_by="general-extractor",
    )


@pytest.mark.asyncio
async def test_write_transaction_commits_before_releasing_the_lock(
    file_db_session: AsyncSession,
) -> None:
    p = _claim("a claim")
    async with db.write_transaction(file_db_session):
        await insert_particle(file_db_session, p)
        assert not store_accepts_a_writer()  # the write is in flight
        assert db._file_write_locks["default"].is_locked
    assert store_accepts_a_writer()
    assert not db._file_write_locks["default"].is_locked
    assert p.id in await get_particles_by_ids(file_db_session, [p.id])


@pytest.mark.asyncio
async def test_write_transaction_rolls_back_a_failure_before_releasing_the_lock(
    file_db_session: AsyncSession,
) -> None:
    p = _claim("a claim that fails")
    with pytest.raises(RuntimeError):
        async with db.write_transaction(file_db_session):
            await insert_particle(file_db_session, p)
            raise RuntimeError("boom")
    assert store_accepts_a_writer()
    assert p.id not in await get_particles_by_ids(file_db_session, [p.id])
