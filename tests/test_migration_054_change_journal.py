# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Alembic 054: the particle change journal and the schema-version index.

The unit suite builds stores with ``create_all``, which installs the journal
through ``particles.store.change_journal``. This drives the real migration
against a populated file store, checks it installs the same triggers, and
round-trips it.
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


def _objects(db: Path) -> dict[str, str]:
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE name LIKE 'trg_%' "
            "OR name IN ('particle_changes', 'ix_particles_status_schema_version')"
        ).fetchall()
    return dict(rows)


def _insert_particle(db: Path) -> str:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from particles.core.schema import Confidence, Particle, UncertaintyNature
    from particles.core.scoring.confidence import CalibrationSource
    from particles.store.particle_store import insert_particle

    particle = Particle(
        content="A claim stored before the journal existed.",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
    )

    async def run() -> None:
        engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
        async with AsyncSession(engine) as session:
            await insert_particle(session, particle, [0.6, 0.8])
            await session.commit()
        await engine.dispose()

    asyncio.run(run())
    return particle.id


@pytest.fixture
def store(tmp_path: Path) -> Path:
    db = tmp_path / "store.db"
    command.upgrade(_alembic(db), "053")
    return db


def test_upgrade_installs_the_journal_and_keeps_rows(store: Path) -> None:
    pid = _insert_particle(store)
    with sqlite3.connect(store) as conn:
        before = conn.execute("SELECT * FROM particles").fetchall()

    command.upgrade(_alembic(store), "054")

    assert set(_objects(store)) == {
        "particle_changes",
        "ix_particles_status_schema_version",
        "trg_particles_journal_insert",
        "trg_particles_journal_update",
        "trg_particles_journal_delete",
        "trg_particle_changes_prune",
    }
    with sqlite3.connect(store) as conn:
        assert conn.execute("SELECT * FROM particles").fetchall() == before
        # The first entry: a marker before any write.
        assert conn.execute("SELECT particle_id FROM particle_changes").fetchall() == [("",)]
        conn.execute("UPDATE particles SET confidence_value = 0.7 WHERE id = ?", (pid,))
        assert conn.execute("SELECT particle_id FROM particle_changes ORDER BY seq").fetchall() == [
            ("",),
            (pid,),
        ]


def test_migration_and_create_all_install_the_same_journal(store: Path, tmp_path: Path) -> None:
    import sqlalchemy as sa

    import particles._orm_modules  # noqa: F401
    from particles.db import Base

    command.upgrade(_alembic(store), "054")
    fresh = tmp_path / "fresh.db"
    engine = sa.create_engine(f"sqlite:///{fresh}")
    Base.metadata.create_all(engine)
    engine.dispose()
    migrated, created = _objects(store), _objects(fresh)
    assert set(migrated) == set(created)
    for name, sql in migrated.items():
        if name.startswith("trg_"):
            assert sql == created[name], name


def test_downgrade_removes_it_and_keeps_rows(store: Path) -> None:
    command.upgrade(_alembic(store), "054")
    _insert_particle(store)
    with sqlite3.connect(store) as conn:
        before = conn.execute("SELECT * FROM particles").fetchall()

    command.downgrade(_alembic(store), "053")

    assert _objects(store) == {}
    with sqlite3.connect(store) as conn:
        assert conn.execute("SELECT * FROM particles").fetchall() == before
    command.upgrade(_alembic(store), "054")
    assert len(_objects(store)) == 6
