"""Async SQLAlchemy engine registry, session factories, and ORM base.

A **store** is one database (a SQLite file, or a Postgres database/schema). The
engine + session factory for each store are constructed lazily on first use and
cached per :data:`StoreHandle`, so changes to ``storage.database_url`` /
``storage.stores`` take effect after ``reset_config()``. Tests rely on this: the
autouse fixture in ``tests/conftest.py`` calls ``reset_config()`` before every
test, which discards the cached engines so the next ``get_engine()`` rebuilds.

Multi-store is a store-parameterization capability: the engine is
*store-parameterized*, every entry point selects a store, and the default
argument ``store=DEFAULT_STORE`` keeps single-store call sites byte-for-byte
unchanged (invariant §1). Which stores a caller may reach (auth / ACL /
provisioning) is a separate access-control layer above this one; this
layer only resolves a handle to a DSN and hands back a session.

``storage.database_url`` defaults to SQLite (development). Set to a
``postgresql+asyncpg://...`` URL for production.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from opentelemetry import metrics
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from particles.config import get_config, register_reset_hook, sqlite_file_path

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


#: A store handle names a database. ``DEFAULT_STORE`` is the implicit store that
#: resolves to ``storage.database_url``; named stores live in ``storage.stores``.
StoreHandle = str
DEFAULT_STORE: StoreHandle = "default"


# Observability (Phase 2): a single OTel counter for SQLite
# write-lock contention. Emitted through the no-op-by-default OTel **API** (a
# base dependency), so it costs nothing until ``setup_observability()`` installs
# a provider. Incremented from the engine ``handle_error`` hook below, so it
# counts every ``database is locked`` at the DB boundary — regardless of which
# writer process (the always-on engine vs. a direct-I/O CLI verb on the host)
# or path (HTTP / CLI / ``extract --all-pending``) lost the lock. This is the
# metric that measures the single-writer contention directly.
_meter = metrics.get_meter("particles.db")
_sqlite_busy_counter = _meter.create_counter(
    "particles.sqlite.busy",
    unit="1",
    description=(
        "SQLite 'database is locked' / busy events — cross-process write-lock "
        "contention"
    ),
)


def _record_sqlite_busy(context: Any) -> None:
    """SQLAlchemy ``handle_error`` hook: count SQLite write-lock contention.

    Fires for every DBAPI error surfacing through a SQLite engine. We increment
    the ``particles.sqlite.busy`` counter only for ``database is locked`` (the
    busy_timeout expired with a writer still holding the single SQLite write
    lock) and leave the error to propagate untouched — this hook is purely
    observational (no ``return``, no ``context`` mutation). Registered on the
    SQLite engines only; Postgres has genuine concurrent writers and no such
    condition.
    """
    exc = context.original_exception
    if exc is not None and "database is locked" in str(exc).lower():
        _sqlite_busy_counter.add(1)


# Cross-process single-writer discipline. SQLite has exactly one
# writer, but the always-on engine host invites a second writer process (a
# direct-I/O CLI verb vs. the engine). ``write_lock`` serializes writers both
# *within* a process (the per-store ``asyncio.Lock``) and *across* processes
# (the per-store ``filelock`` on a lockfile beside the DB), so they queue fairly
# instead of racing the SQLite ``busy_timeout``. Both are keyed per store handle
# and cleared by ``reset_engine`` so tests / reloads start fresh.
_async_write_locks: dict[StoreHandle, asyncio.Lock] = {}
_file_write_locks: dict[StoreHandle, FileLock] = {}


class WriteLockTimeout(Exception):
    """A writer could not acquire the cross-process write lock within the timeout.

    Raised by :func:`write_lock` when ``storage.write_lock.timeout_seconds``
    elapses with another process still holding the store's writer lock. The CLI
    ``run()`` helper and the engine request path translate it into a clean
    "another writer holds the store — retry" message.
    """


#: DSN → on-disk path. Lives in the Client-layer ``particles.config``
#: so the store-adjacent path resolver and this write-lock derivation share one
#: rule; the write lock is a no-op for the ``None`` cases (in-memory stores are
#: not shared across processes, and Postgres has genuine concurrent writers).
_sqlite_file_path = sqlite_file_path


def _store_is_file_sqlite(store: StoreHandle) -> bool:
    return _sqlite_file_path(_resolve_store_dsn(store)) is not None


def _write_lock_path(store: StoreHandle) -> str:
    """Lockfile path for ``store`` — config override, else ``<db_file>.writelock``."""
    configured = get_config().storage.write_lock.path
    if configured:
        return configured
    db_path = _sqlite_file_path(_resolve_store_dsn(store))
    return f"{db_path}.writelock"


def _get_async_write_lock(store: StoreHandle) -> asyncio.Lock:
    lock = _async_write_locks.get(store)
    if lock is None:
        lock = asyncio.Lock()
        _async_write_locks[store] = lock
    return lock


@asynccontextmanager
async def write_lock(store: StoreHandle = DEFAULT_STORE) -> AsyncGenerator[None, None]:
    """Hold the per-store writer lock for the duration of the block.

    Wrap a **write transaction** — never the LLM / fetch / embed phase — so a
    concurrent writer (the engine vs. a direct-I/O CLI verb on the host) queues
    fairly for the lock instead of racing SQLite's ``busy_timeout`` until it
    times out. Acquires the in-process ``asyncio.Lock`` first (intra-process
    serialization), then the cross-process ``filelock`` via ``asyncio.to_thread``
    so the event loop stays responsive while waiting; raises
    :class:`WriteLockTimeout` if ``timeout_seconds`` elapses. The OS file lock
    releases automatically if the holding process dies, so a crash never wedges
    the store.

    A **no-op** when ``storage.write_lock.enabled`` is false or the store is not
    a file-based SQLite database (in-memory SQLite / PostgreSQL), so the
    single-machine and test paths are byte-for-byte unchanged.
    """
    cfg = get_config().storage.write_lock
    if not cfg.enabled or not _store_is_file_sqlite(store):
        yield
        return

    async_lock = _get_async_write_lock(store)
    await async_lock.acquire()
    try:
        flock = _file_write_locks.get(store)
        if flock is None:
            # ``thread_local=False`` is load-bearing: we acquire and release via
            # ``asyncio.to_thread``, which may run them on *different* pool
            # threads. filelock's default thread-local lock state would then make
            # the release a no-op (and ``is_locked`` lie) — sharing the state
            # across threads keeps the acquire/release on the event loop correct.
            flock = FileLock(_write_lock_path(store), thread_local=False)
            _file_write_locks[store] = flock
        try:
            await asyncio.to_thread(flock.acquire, timeout=cfg.timeout_seconds)
        except FileLockTimeout as exc:
            raise WriteLockTimeout(
                "another particles writer is holding the store write lock; "
                "it will free shortly — retry."
            ) from exc
        try:
            yield
        finally:
            await asyncio.to_thread(flock.release)
    finally:
        async_lock.release()


_engines: dict[StoreHandle, AsyncEngine] = {}
_session_factories: dict[StoreHandle, async_sessionmaker[AsyncSession]] = {}


def _resolve_store_dsn(store: StoreHandle) -> str:
    """Map a store handle to its database URL via config.

    The ``default`` handle always resolves to ``storage.database_url``; any other
    handle must be declared in ``storage.stores``. Read at call time so
    ``reset_config()`` is honoured.
    """
    storage = get_config().storage
    if store == DEFAULT_STORE:
        return storage.database_url
    try:
        return storage.stores[store]
    except KeyError:
        # Do not enumerate the configured store handles in the message: the MCP
        # ``particles://digest/{store}`` resource surfaces this error to clients,
        # and listing the known stores is a minor info disclosure (security
        # review F29). The exception type stays ``KeyError`` for callers that
        # match on it.
        raise KeyError(f"Unknown store {store!r}") from None


def get_engine(store: StoreHandle = DEFAULT_STORE) -> AsyncEngine:
    """Return the async engine for ``store``, creating it on first use.

    For SQLite URLs we attach a connection event listener that sets WAL journal
    mode and a 30-second busy_timeout. WAL allows concurrent readers + one writer
    with brief locks — without it, a parallel ``particles deposit`` while
    ``particles extract`` is running fails immediately with ``database is
    locked``. The 30-second busy_timeout is sized to absorb the post-LLM phase of
    an ``extract_snapshot`` run on a fat snapshot (hundreds of particles +
    subject resolution). PostgreSQL URLs are untouched.

    SQLite engines also get a ``handle_error`` listener that increments the
    ``particles.sqlite.busy`` OTel counter on ``database is locked`` (
    Phase 2) — the metric that measures the write-lock
    contention.
    """
    engine = _engines.get(store)
    if engine is None:
        url = _resolve_store_dsn(store)
        engine = create_async_engine(url, echo=False)
        if url.startswith("sqlite"):
            event.listen(engine.sync_engine, "connect", _sqlite_set_pragmas)
            # Count write-lock contention at the DB boundary.
            event.listen(engine.sync_engine, "handle_error", _record_sqlite_busy)
        _engines[store] = engine
    return engine


def _sqlite_set_pragmas(dbapi_connection: object, _conn_record: object) -> None:
    """Apply SQLite-specific pragmas on each new connection.

    Called by SQLAlchemy's ``connect`` event hook. ``WAL`` is the single biggest
    concurrency improvement available to a SQLite backend; ``busy_timeout`` makes
    the brief writer-contention window block instead of raising
    ``OperationalError: database is locked``.
    """
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
    finally:
        cursor.close()


def get_session_factory(store: StoreHandle = DEFAULT_STORE) -> async_sessionmaker[AsyncSession]:
    """Return the session factory for ``store``, creating it on first use."""
    factory = _session_factories.get(store)
    if factory is None:
        factory = async_sessionmaker(get_engine(store), class_=AsyncSession, expire_on_commit=False)
        _session_factories[store] = factory
    return factory


def _dispose_engine(engine: AsyncEngine) -> None:
    """Best-effort dispose of one engine across whatever loop is available.

    Without this, the aiosqlite Connection an engine owns becomes unreachable
    while still bound to a (possibly already-dead) event loop, and its
    ``__del__`` raises ``RuntimeError: Event loop is closed`` when GC fires. We
    try the async dispose on whichever loop is available; if neither path works
    we fall back to the sync pool dispose, which at least releases the file
    handle. Any failure is swallowed and logged at DEBUG — disposal is
    best-effort by design.
    """
    try:
        running_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    try:
        if running_loop is not None:
            # Schedule the dispose without awaiting — caller is sync. The task
            # runs before the loop closes if the caller drains pending callbacks.
            running_loop.create_task(engine.dispose())
        else:
            asyncio.run(engine.dispose())
    except Exception as exc:
        log.debug("Best-effort engine dispose failed: %s", exc)
        try:
            engine.sync_engine.dispose()
        except Exception as inner:
            log.debug("Sync engine fallback dispose also failed: %s", inner)


def reset_engine() -> None:
    """Discard all cached engines and session factories (every store).

    Called automatically by ``particles.config.reset_config()`` so test setups
    and runtime reloads pick up new ``storage.database_url`` / ``storage.stores``
    values without a process restart. Each prior engine is best-effort disposed
    before its reference is dropped (see :func:`_dispose_engine`).
    """
    global _engines, _session_factories, _async_write_locks, _file_write_locks
    prior = list(_engines.values())
    _engines = {}
    _session_factories = {}
    # Drop the per-store write locks too: an ``asyncio.Lock`` binds to
    # the loop that first uses it, so reusing one across a test's fresh loop would
    # raise; ``reset_config()`` clears them alongside the engines.
    _async_write_locks = {}
    _file_write_locks = {}
    for engine in prior:
        _dispose_engine(engine)


# Inverted config↔db coupling: the Engine registers its
# engine-cache reset with the Client-layer config module at import time, so
# ``config.reset_config()`` discards cached engines without ``config`` ever
# importing ``db``. ``db`` importing ``config`` is Engine→Client — allowed.
register_reset_hook(reset_engine)


@asynccontextmanager
async def session_scope(
    store: StoreHandle = DEFAULT_STORE, *, write: bool = False
) -> AsyncGenerator[AsyncSession, None]:
    """Async context manager yielding a fresh session bound to ``store``.

    Use from CLI and scripts::

        async with session_scope() as session:             # default store
            ...
        async with session_scope("team-acme") as session:  # a named store
            ...
        async with session_scope(write=True) as session:   # holds the writer lock

    The session *is* the store binding — operations that take a ``session`` stay
    store-agnostic; only entry points choose the store.

    Pass ``write=True`` to hold the cross-process writer lock for the
    session's duration — correct for a **short, pure write** whose whole session
    body is the write transaction (operator verbs, a local-file deposit). Do
    **not** use it for a session that spans slow I/O before the write (an LLM
    extraction, a URL fetch); those acquire :func:`write_lock` around the write
    phase only, so the lock is never held across the slow call.
    """
    if write:
        async with write_lock(store), get_session_factory(store)() as session:
            yield session
    else:
        async with get_session_factory(store)() as session:
            yield session


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a fresh session from the default store.

    Store selection for HTTP requests (which store a request targets) is a
    separate routing layer's concern; this dependency
    serves the default store.
    """
    async with get_session_factory()() as session:
        yield session


async def get_write_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: a default-store session held under the writer lock.

    For a write endpoint whose handler is a **short, pure write** (the handler
    commits before returning, with no slow I/O before the write). The
    cross-process write lock is held for the whole request so the engine
    serializes against a direct-I/O CLI writer on the host. Endpoints whose work
    spans an LLM/extract (``/extract``, ``/reindex``) use the read
    :func:`get_session` dep instead — the extract pipeline acquires
    :func:`write_lock` around its own write phase, never across the LLM.
    """
    async with write_lock(), get_session_factory()() as session:
        yield session


def alembic_paths() -> tuple[Path, Path]:
    """Return ``(alembic.ini, migrations dir)`` for this installation.

    The packaged-resource pattern, applied to the migrations (the
    ``particles.core._resources.schemas_dir`` precedent): prefer the
    wheel-packaged ``particles/_alembic`` copy, fall back to the source-tree
    ``<repo-root>/alembic`` for editable / checkout use.

    The fallback used to be the *only* path, computed as
    ``particles/../alembic``. For a source checkout that resolves correctly; for
    an installed wheel it resolves to ``<site-packages>/alembic`` — which is the
    **alembic library itself**, not this project's migrations — so
    :func:`create_tables` could never work outside a checkout. The container
    image is the first deployment that has no checkout at all, which
    is where the latent breakage surfaced.

    The two copies never collide on ``import alembic``: ``_alembic`` is not an
    importable name, and the source-tree ``alembic/`` has no ``__init__.py``, so
    the import system records it only as a namespace-package *portion* and the
    real regular package in site-packages still wins.
    """
    packaged = Path(__file__).parent / "_alembic"
    if (packaged / "versions").is_dir():
        return packaged / "alembic.ini", packaged
    repo_root = Path(__file__).parent.parent
    return repo_root / "alembic.ini", repo_root / "alembic"


async def create_tables(store: StoreHandle = DEFAULT_STORE) -> None:
    """Create all tables in ``store`` via Alembic upgrade head.

    Running Alembic (rather than ``metadata.create_all``) stamps the
    ``alembic_version`` table, so subsequent ``alembic upgrade head`` calls work
    without a manual ``alembic stamp``. The target store's DSN is passed to the
    Alembic environment via ``config.attributes['store_url']`` (see
    ``alembic/env.py``), which takes precedence over the ``DATABASE_URL`` env var.
    """
    import asyncio

    from alembic.config import Config

    from alembic import command

    ini_path, script_location = alembic_paths()
    alembic_cfg = Config(str(ini_path))
    # alembic.ini's ``script_location = alembic`` is cwd-relative; pin it to the
    # resolved migrations directory so table creation works from any working
    # directory (e.g. ``particles init claude-code`` run outside the repo root)
    # and from an installed wheel with no checkout at all.
    alembic_cfg.set_main_option("script_location", str(script_location))
    alembic_cfg.attributes["store_url"] = _resolve_store_dsn(store)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, command.upgrade, alembic_cfg, "head")
