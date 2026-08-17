"""Store-parameterized engine registry (2).

Covers DSN resolution (default vs named vs unknown) and that two named stores
are genuinely isolated databases, while single-store call sites (no ``store``
arg) keep working unchanged (invariant §1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.config import get_config
from particles.db import (
    DEFAULT_STORE,
    _resolve_store_dsn,
    get_engine,
    reset_engine,
    session_scope,
)


def test_resolve_store_dsn_default_named_and_unknown() -> None:
    cfg = get_config()
    cfg.storage.database_url = "sqlite+aiosqlite:///:memory:"
    cfg.storage.stores = {"team": "sqlite+aiosqlite:///team.db"}

    assert _resolve_store_dsn(DEFAULT_STORE) == "sqlite+aiosqlite:///:memory:"
    assert _resolve_store_dsn("team") == "sqlite+aiosqlite:///team.db"
    with pytest.raises(KeyError):
        _resolve_store_dsn("nope")


async def test_named_stores_are_isolated(tmp_path: Path) -> None:
    import particles._orm_modules  # noqa: F401
    from particles.core.schema import Subject
    from particles.db import Base
    from particles.store.subject_store import find_by_name, insert_subject

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/default.db"
    cfg.storage.stores = {"other": f"sqlite+aiosqlite:///{tmp_path}/other.db"}

    for handle in (DEFAULT_STORE, "other"):
        engine = get_engine(handle)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Same handle returns the cached engine; distinct handles are distinct engines.
    assert get_engine(DEFAULT_STORE) is get_engine(DEFAULT_STORE)
    assert get_engine(DEFAULT_STORE) is not get_engine("other")

    # A no-arg call site writes to the default store (BC, invariant §1).
    async with session_scope() as s:
        await insert_subject(s, Subject(canonical_name="Berlin", asserted_by="t"))
        await s.commit()

    async with session_scope(DEFAULT_STORE) as s:
        assert await find_by_name(s, "Berlin") is not None
    async with session_scope("other") as s:
        assert await find_by_name(s, "Berlin") is None

    # Dispose on the live loop before the autouse reset_config tears it down.
    for handle in (DEFAULT_STORE, "other"):
        await get_engine(handle).dispose()
    reset_engine()
