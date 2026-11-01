# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""An exposure row whose writer was never recorded is never trusted (security review 2026-10-28-2).

AA44: before migration 060 an agent key's HTTP row looked exactly like the hook's
own, and a downgrade of 060 forgets every writer, so the session frame trusts
only rows that record the in-process writer or a trusted principal. The real
Alembic migrations run against a file store under tmp_path; no model call,
no outside connection.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from alembic.config import Config

from alembic import command
from particles.db import alembic_paths


def _alembic(db: Path) -> Config:
    ini_path, script_location = alembic_paths()
    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(script_location))
    cfg.attributes["store_url"] = f"sqlite+aiosqlite:///{db}"
    return cfg


def _frame_session_ids(db: Path) -> list[str]:
    """The session ids whose rows the session frame's recorder filter admits."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from particles.operations.utility_exposure import exposure_recorders
    from particles.store.session_exposure_store import session_exposures_for

    async def run() -> list[str]:
        engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
        out: list[str] = []
        async with AsyncSession(engine) as session:
            for sid in ("agent-written", "hook-written"):
                rows = await session_exposures_for(session, sid, recorders=exposure_recorders())
                out.extend(r.session_id for r in rows)
        await engine.dispose()
        return out

    return asyncio.run(run())


def _insert(db: Path, session_id: str, recorded_by: str | None = None) -> None:
    cols = "session_id, recorded_at, source, action, project_key, observer_scope_applied, beliefs"
    vals = [session_id, "2026-10-26 12:00:00", "startup", "full", "acme", 1, "[]"]
    if recorded_by is not None:
        cols += ", recorded_by"
        vals.append(recorded_by)
    with sqlite3.connect(db) as conn:
        conn.execute(
            f"INSERT INTO session_exposures ({cols}) VALUES ({', '.join('?' * len(vals))})", vals
        )


@pytest.fixture
def store(tmp_path: Path) -> Path:
    db = tmp_path / "store.db"
    command.upgrade(_alembic(db), "059")
    return db


def test_an_exposure_row_whose_writer_was_never_recorded_is_not_read_as_in_process(
    store: Path,
) -> None:
    """After 060, a row written before the column existed is not trusted as an in-process row.

    Before migration 060 the exposure route recorded no principal, so a row an
    agent key wrote over HTTP is indistinguishable from the hook's own.
    """
    _insert(store, "agent-written")  # an agent key's HTTP row, before 060
    command.upgrade(_alembic(store), "060")
    with sqlite3.connect(store) as conn:
        (recorded_by,) = conn.execute(
            "SELECT recorded_by FROM session_exposures WHERE session_id = 'agent-written'"
        ).fetchone()
    assert recorded_by is None  # no backfill, as the migration says
    assert "agent-written" not in _frame_session_ids(store), (
        "a pre-060 exposure row of unknown writer enters the session frame as an in-process row"
    )


def test_a_downgrade_and_upgrade_never_makes_an_agent_keys_row_trusted(store: Path) -> None:
    """Round-tripping 060 keeps an agent key's exposure row out of the session frame."""
    command.upgrade(_alembic(store), "060")
    _insert(store, "agent-written", recorded_by="agent:ci-bot")
    assert "agent-written" not in _frame_session_ids(store)
    command.downgrade(_alembic(store), "059")
    command.upgrade(_alembic(store), "060")
    assert "agent-written" not in _frame_session_ids(store), (
        "after 060's downgrade and upgrade an agent key's exposure row reads as in-process"
    )


def test_an_in_process_exposure_is_recorded_as_such_and_trusted(store: Path) -> None:
    """The hook's own row records the in-process writer and enters the session frame."""
    from datetime import UTC, datetime

    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from particles.store.session_exposure_store import (
        IN_PROCESS_RECORDER,
        SessionExposure,
        record_session_exposure,
    )

    command.upgrade(_alembic(store), "head")

    async def run() -> None:
        engine = create_async_engine(f"sqlite+aiosqlite:///{store}")
        async with AsyncSession(engine) as session:
            await record_session_exposure(
                session,
                SessionExposure(
                    session_id="hook-written",
                    recorded_at=datetime.now(UTC),
                    source="startup",
                    action="full",
                    project_key="acme",
                    observer_scope_applied=True,
                    beliefs=[],
                ),
            )
            await session.commit()
        await engine.dispose()

    asyncio.run(run())
    with sqlite3.connect(store) as conn:
        (recorded_by,) = conn.execute(
            "SELECT recorded_by FROM session_exposures WHERE session_id = 'hook-written'"
        ).fetchone()
    assert recorded_by == IN_PROCESS_RECORDER
    assert _frame_session_ids(store) == ["hook-written"]
