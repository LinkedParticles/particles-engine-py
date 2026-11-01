# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""An engine belief write searches the store as it stands, under a bounded lock hold.

Security review 2026-10-31-2, AE1 and AE5. Each test drives one writer on a
scratch file-backed store (configured through the loader, ``network.egress:
none``), with the embedder and the ladder's probes stubbed. The import tests
commit a contradicting claim from another session between the writer's gather
and its apply (a hook installed on the gather, never a timing race): the
writer's claim is not left ACTIVE beside it with no INCONSISTENCY record. The
last test checks the writer lock is never held across a store-wide load.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

LEGACY = "sec13-writers-legacy-operator-key"

E_TEXT = "The Acme widget ships only in red."
C_TEXT = "The Acme widget ships in blue."
P_TEXT = "Acme publishes a catalogue every spring."
_V1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_VE = np.array([0.9, float(np.sqrt(1 - 0.81)), 0.0, 0.0], dtype=np.float32)
_VX = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _vector(text: str) -> np.ndarray[Any, Any]:
    if E_TEXT in text:
        return _VE
    if C_TEXT in text:
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


def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    db = tmp_path / "s1blockj.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", LEGACY)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    return db


@pytest.fixture
def cleanup() -> Generator[None, None, None]:
    yield
    from particles.api import _keys
    from particles.config import reset_config
    from particles.db import reset_engine

    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()
    reset_engine()


@pytest.fixture
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    import particles.ingest.pipeline as pipeline

    async def _embed(texts: list[str]) -> list[Any]:
        return [_vector(t) for t in texts]

    monkeypatch.setattr(pipeline, "_embed_batch_async", _embed)


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Contradiction only between C and E; no update answer."""
    import particles.ingest.pipeline as pipeline

    asked: list[tuple[str, str]] = []

    async def _contra(a: str, b: str) -> bool | None:
        asked.append((a, b))
        return (C_TEXT in a and E_TEXT in b) or (C_TEXT in b and E_TEXT in a)

    async def _update(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(pipeline, "_has_contradiction_signal", _contra)
    monkeypatch.setattr(pipeline, "_has_update_signal", _update)
    return asked


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


def _mint(role: str, principal: str) -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    return _keys.mint(Role(role), principal)[0]


def _extracted(content: str) -> Any:
    from particles.core.schema import Confidence, Particle, UncertaintyNature
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status

    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        status=Status.ACTIVE,
    )


async def _insert_elsewhere(content: str) -> str:
    """Another writer: its own session, under the writer lock, committed."""
    from particles.db import session_scope, write_transaction
    from particles.ingest.pipeline import reconcile_and_insert

    async with session_scope() as other, write_transaction(other):
        landed = await reconcile_and_insert(other, _extracted(content), fail_closed=True)
    assert landed is not None
    return landed.id


def _bundled(content: str) -> Any:
    """A unit as a bundle carries it: one SOURCE ref naming the origin store's entry."""
    from particles.core.schema import ProvenanceRef, ProvenanceRefType

    particle = _extracted(content).model_copy(
        update={
            "provenance": [
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id="origin-entry-0001",
                    snapshot_id="origin-snapshot-0001",
                )
            ]
        }
    )
    return particle


def _seed_extracted(content: str) -> str:
    async def _go() -> str:
        from particles.db import get_engine

        pid = await _insert_elsewhere(content)
        await get_engine().dispose()
        return pid

    return asyncio.run(_go())


def _rows(db: Path) -> list[tuple[str, str, str | None]]:
    con = sqlite3.connect(db)
    try:
        return list(con.execute("SELECT content, status, status_reason FROM particles").fetchall())
    finally:
        con.close()


def _assert_not_beside(db: Path, needle: str) -> None:
    rows = _rows(db)
    active = {c for c, s, _ in rows if s == "ACTIVE"}
    inconsistency = [c for c, s, _ in rows if s == "INCONSISTENCY"]
    beside = any(needle in c for c in active) and E_TEXT in active
    assert not (beside and not inconsistency), (
        f"the writer's claim landed ACTIVE beside a contradiction committed between its "
        f"gather and apply, with no INCONSISTENCY: {rows}"
    )


def _hook_gather(monkeypatch: pytest.MonkeyPatch, needle: str) -> list[str]:
    import particles.ingest.pipeline as pipeline

    real = pipeline.gather_reconcile
    fired: list[str] = []

    async def _hook(session: Any, particle: Any, *a: Any, **k: Any) -> Any:
        gathered = await real(session, particle, *a, **k)
        if needle in particle.content and not fired:
            fired.append(await _insert_elsewhere(E_TEXT))
        return gathered

    monkeypatch.setattr(pipeline, "gather_reconcile", _hook)
    return fired


def _belief(content: str, **extra: Any) -> dict[str, Any]:
    return {
        "content": content,
        "subject_names": ["Acme widget"],
        "confidence": 0.7,
        "source_excerpt": content,
        **extra,
    }


# --------------------------------------------------------------------------- #
# interchange import (AE1)
# --------------------------------------------------------------------------- #


def test_an_import_is_reconciled_against_a_contradiction_committed_after_its_gather(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: None,
    stub_embed: None,
    probes: list[tuple[str, str]],
) -> None:
    """An imported claim is reconciled against a claim another writer committed while the
    import ran (between its gather and its apply), not only against the candidate set the
    import loaded before its first commit, so it is not left ACTIVE beside a contradiction."""
    from particles.interchange.codec import to_unit
    from particles.interchange.store import import_units

    db = _store(tmp_path, monkeypatch)
    fired = _hook_gather(monkeypatch, C_TEXT)
    unit = to_unit(_bundled(C_TEXT), {})

    async def _go() -> None:
        from particles.db import reset_engine, session_scope

        try:
            async with session_scope() as session:
                await import_units(session, [unit])
        finally:
            reset_engine()

    asyncio.run(_go())
    assert fired, "the hook never ran"
    _assert_not_beside(db, C_TEXT)


def test_control_an_import_meeting_a_stored_contradiction_is_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: None,
    stub_embed: None,
    probes: list[tuple[str, str]],
) -> None:
    """Control: with the contradiction stored before the import starts, the same unit
    meets it (the stubs reach the ladder), so the window test above isolates the cache."""
    from particles.interchange.codec import to_unit
    from particles.interchange.store import import_units

    db = _store(tmp_path, monkeypatch)
    _seed_extracted(E_TEXT)
    unit = to_unit(_bundled(C_TEXT), {})

    async def _go() -> None:
        from particles.db import reset_engine, session_scope

        try:
            async with session_scope() as session:
                await import_units(session, [unit])
        finally:
            reset_engine()

    asyncio.run(_go())
    _assert_not_beside(db, C_TEXT)


# --------------------------------------------------------------------------- #
# AE5: what the writer lock is held for
# --------------------------------------------------------------------------- #


def test_a_belief_write_loads_no_store_wide_candidate_set_under_the_writer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: None,
    stub_embed: None,
    probes: list[tuple[str, str]],
) -> None:
    """A belief write holds the store's writer lock only for its writes: it never loads
    the store's whole ACTIVE candidate set (every claim and its embedding) while it holds
    the lock, so an in-process writer queued behind it, which waits with no timeout,
    does not wait for a scan that grows with the store."""
    import particles.ingest.pipeline as pipeline
    from particles.db import write_lock_held

    _store(tmp_path, monkeypatch)
    for i in range(20):
        _seed_extracted(f"Acme filler claim number {i} about shipping.")
    real = pipeline.load_active_conflict_candidates
    loads: list[tuple[bool, int]] = []

    async def _recording(session: Any) -> Any:
        pairs = await real(session)
        loads.append((write_lock_held("default"), len(pairs)))
        return pairs

    monkeypatch.setattr(pipeline, "load_active_conflict_candidates", _recording)
    token = _mint("agent", "agent:sec13-writers")
    resp = _client().post(
        "/particles/assert", json=_belief(C_TEXT), headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code < 300, resp.text
    assert loads, "the conflict search never ran"
    under_lock = [n for held, n in loads if held]
    assert not under_lock, (
        f"the belief write loaded the store-wide candidate set ({under_lock} claims) while "
        f"holding the writer lock; every load: {loads}"
    )
