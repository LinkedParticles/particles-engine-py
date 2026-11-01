# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Interchange import, driven in process, acts on the store as it stands under the lock.

Security review 2026-10-31-2, AE4: an import's apply reconciles against a claim
another writer committed since its gather, and an apply that is not fail-closed
never corroborates a claim it finds only under the lock. Scratch file-backed
store under ``tmp_path``; configuration only through the loader
(``network.egress: none``); embedding and ladder probes stubbed at the pipeline
seams; no outside call. The concurrent writer runs from a hook installed
between the gather and the apply (a patched ``gather_reconcile``).
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

E_TEXT = "The Acme widget ships only in red."
D_TEXT = "Acme sells its widget in a blue finish."
C_TEXT = "The Acme widget ships in blue."
_V1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_VE = np.array([0.9, float(np.sqrt(1 - 0.81)), 0.0, 0.0], dtype=np.float32)
_VX = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _vector(text: str) -> np.ndarray[Any, Any]:
    if text == E_TEXT:
        return _VE
    if text in (C_TEXT, D_TEXT):
        return _V1
    return _VX


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
    home = tmp_path / "home"
    home.mkdir()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\nmcp:\n  write:\n    enabled_stores: [default]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    db = tmp_path / "s6.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    yield db
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch, store: Path) -> dict[str, list[Any]]:
    """Stub embedder and probes; each records the advisory and SQLite lock state."""
    import particles.ingest.pipeline as pipeline
    from particles.db import write_lock_held

    seen: dict[str, list[Any]] = {"probe": [], "embed": [], "pairs": []}

    def _state() -> tuple[bool, bool]:
        return write_lock_held("default"), _sqlite_write_locked(store)

    async def _embed(texts: list[str]) -> list[Any]:
        seen["embed"].append(_state())
        return [_vector(t) for t in texts]

    async def _contra(a: str, b: str) -> bool | None:
        seen["probe"].append(_state())
        seen["pairs"].append((a, b))
        return {a, b} == {C_TEXT, E_TEXT}

    async def _update(*_a: Any, **_k: Any) -> None:
        seen["probe"].append(_state())
        return None

    monkeypatch.setattr(pipeline, "_embed_batch_async", _embed)
    monkeypatch.setattr(pipeline, "_has_contradiction_signal", _contra)
    monkeypatch.setattr(pipeline, "_has_update_signal", _update)
    return seen


def _sqlite_write_locked(path: Path) -> bool:
    con = sqlite3.connect(path, timeout=0)
    try:
        con.execute("BEGIN IMMEDIATE")
        con.execute("ROLLBACK")
        return False
    except sqlite3.OperationalError:
        return True
    finally:
        con.close()


def _run(coro: Any) -> Any:
    from particles.db import reset_engine

    try:
        return asyncio.run(coro)
    finally:
        reset_engine()


def _rows(db: Path) -> list[tuple[str, str, str | None]]:
    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT content, status, status_reason FROM particles").fetchall()
    finally:
        con.close()


def _status_of(db: Path, content: str) -> list[str]:
    return [s for c, s, _r in _rows(db) if c == content]


def _inconsistencies(db: Path) -> int:
    return sum(1 for _c, s, _r in _rows(db) if s == "INCONSISTENCY")


def _extracted(content: str, subject_ids: list[str] | None = None) -> Any:
    from particles.core.schema import Confidence, Particle, UncertaintyNature
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status

    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        status=Status.ACTIVE,
        subject_ids=subject_ids or [],
    )


async def _insert_elsewhere(content: str) -> str:
    """Another writer: its own session, under the writer lock, committed."""
    from particles.db import session_scope, write_transaction
    from particles.ingest.pipeline import reconcile_and_insert

    async with session_scope() as other, write_transaction(other):
        landed = await reconcile_and_insert(other, _extracted(content), fail_closed=True)
    assert landed is not None
    return landed.id


def _hook_after_gather(monkeypatch: pytest.MonkeyPatch, trigger: str, commit: str) -> None:
    """Commit ``commit`` from another writer after the gather of ``trigger`` returns."""
    import particles.ingest.pipeline as pipeline

    real = pipeline.gather_reconcile

    async def _hook(session: Any, particle: Any, *a: Any, **k: Any) -> Any:
        gathered = await real(session, particle, *a, **k)
        if particle.content == trigger:
            await _insert_elsewhere(commit)
        return gathered

    monkeypatch.setattr(pipeline, "gather_reconcile", _hook)


def _unit(content: str) -> dict[str, Any]:
    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status
    from particles.interchange.codec import to_unit

    particle = Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="origin-entry-0001",
                snapshot_id="origin-snap-0001",
            )
        ],
    )
    return to_unit(particle, {})


async def _import(units: list[dict[str, Any]]) -> Any:
    from particles.db import session_scope
    from particles.interchange.store import import_units

    async with session_scope("default") as session:
        return await import_units(session, units, store="default")


# --------------------------------------------------------------------------- #
# Interchange import (AE4)
# --------------------------------------------------------------------------- #


def test_control_import_reconciles_against_a_contradiction_stored_before_it(
    store: Path, stubs: dict[str, list[Any]]
) -> None:
    """Control: an imported claim meeting a contradiction already stored is not left
    ACTIVE beside it with no INCONSISTENCY (the stubs drive the conflict path)."""
    _run(_insert_elsewhere(E_TEXT))
    _run(_import([_unit(C_TEXT)]))
    assert _status_of(store, E_TEXT) == ["ACTIVE"], _rows(store)
    assert _status_of(store, C_TEXT) != ["ACTIVE"] or _inconsistencies(store) > 0, _rows(store)


def test_import_reconciles_against_a_contradiction_committed_after_its_gather(
    store: Path, stubs: dict[str, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An import's apply reconciles against a claim another writer committed since
    its gather, so an imported contradiction is never written ACTIVE beside it with
    no INCONSISTENCY."""
    _hook_after_gather(monkeypatch, C_TEXT, E_TEXT)
    _run(_import([_unit(C_TEXT)]))
    assert _status_of(store, E_TEXT) == ["ACTIVE"], _rows(store)
    assert not (_status_of(store, C_TEXT) == ["ACTIVE"] and _inconsistencies(store) == 0), (
        "the imported claim landed ACTIVE beside a contradiction another writer committed "
        f"between its gather and its apply, with no INCONSISTENCY: {_rows(store)}; "
        f"pairs probed: {stubs['pairs']}"
    )


# --------------------------------------------------------------------------- #
# LocalBackend, the facade, curate supersede (AD1, AD3, AD25)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The abstraction accept gesture (Z112)
# --------------------------------------------------------------------------- #


def test_a_non_fail_closed_apply_never_corroborates_an_unprobed_claim_found_under_the_lock(
    store: Path, stubs: dict[str, list[Any]]
) -> None:
    """The import's half of the defect without the cache: an apply that is not
    fail-closed (import passes no ``fail_closed``) and finds, under the lock, a claim
    its gather never probed, does not write the candidate ACTIVE beside it with no
    probe and no INCONSISTENCY, even when the search reads the store fresh."""
    from particles.db import session_scope, write_transaction
    from particles.ingest.pipeline import apply_reconcile, gather_reconcile

    async def _go() -> None:
        async with session_scope() as session:
            particle = _extracted(C_TEXT)
            gathered = await gather_reconcile(session, particle)
            await _insert_elsewhere(E_TEXT)
            async with write_transaction(session):
                await apply_reconcile(session, particle, gathered, append_provenance=False)

    _run(_go())
    assert _status_of(store, E_TEXT) == ["ACTIVE"], _rows(store)
    assert not (_status_of(store, C_TEXT) == ["ACTIVE"] and _inconsistencies(store) == 0), (
        "a claim found only under the lock was corroborated unprobed: "
        f"{_rows(store)}; pairs probed: {stubs['pairs']}"
    )
