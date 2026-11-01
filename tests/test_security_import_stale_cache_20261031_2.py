# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Interchange import reconciles each unit against the store as it stands.

Security review 2026-10-31-2, AE3 and L10: the import commits per unit, so other
writers commit between its units. A later unit meets a claim committed during
the import, is not reconciled against a claim retired during it, and never
treats one of the bundle's own non-ACTIVE units as a conflict partner.

Scratch file-backed stores under tmp_path, configuration through the loader
(network.egress: none), embedding and probes stubbed, no outside call. The
concurrent writer runs from a hook installed after one unit's gather, never a
timing race.
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
C_TEXT = "The Acme widget ships in blue."
X_TEXT = "The Acme factory sits beside a river."
_V1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_VE = np.array([0.9, float(np.sqrt(1 - 0.81)), 0.0, 0.0], dtype=np.float32)
_VX = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _vector(text: str) -> np.ndarray[Any, Any]:
    if text == E_TEXT:
        return _VE
    if text == C_TEXT:
        return _V1
    return _VX


def _create_tables(store: str) -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine(store).begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine(store).dispose()

    asyncio.run(_go())


@pytest.fixture
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    import particles.ingest.pipeline as pipeline
    from particles.config import reset_config

    home = tmp_path / "home"
    home.mkdir()
    src = tmp_path / "src.db"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"network:\n  egress: none\nstorage:\n  stores:\n    src: sqlite+aiosqlite:///{src}\n",
        encoding="utf-8",
    )
    target = tmp_path / "target.db"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{target}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _create_tables("default")
    _create_tables("src")

    async def _embed(texts: list[str]) -> list[Any]:
        return [_vector(t) for t in texts]

    async def _contra(a: str, b: str) -> bool | None:
        return {a, b} == {C_TEXT, E_TEXT}

    async def _update(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(pipeline, "_embed_batch_async", _embed)
    monkeypatch.setattr(pipeline, "_has_contradiction_signal", _contra)
    monkeypatch.setattr(pipeline, "_has_update_signal", _update)
    yield target
    reset_config()


def _particle(content: str, who: str) -> Any:
    import uuid

    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status

    return Particle(
        content=content,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=str(uuid.uuid4()),
                snapshot_id=str(uuid.uuid4()),
            )
        ],
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=who,
        status=Status.ACTIVE,
    )


def _bundle(tmp_path: Path) -> Path:
    """A bundle of two units, X then C, exported from the src store."""
    from particles.db import reset_engine, session_scope
    from particles.interchange.store import export_store_bundle
    from particles.store.particle_store import insert_particle

    async def _go() -> dict[str, str]:
        async with session_scope("src") as s:
            await insert_particle(s, _particle(X_TEXT, "general-extractor"), _VX.tolist())
            await insert_particle(s, _particle(C_TEXT, "general-extractor"), _V1.tolist())
            await s.commit()
            return await export_store_bundle(s)

    try:
        files = asyncio.run(_go())
    finally:
        reset_engine()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name, content in files.items():
        (bundle / name).write_text(content, encoding="utf-8")
    return bundle


async def _insert_into_target(content: str) -> str:
    """Another writer on the target store: its own session, under the writer lock."""
    from particles.db import session_scope, write_transaction
    from particles.ingest.pipeline import reconcile_and_insert

    async with session_scope() as other, write_transaction(other):
        landed = await reconcile_and_insert(other, _particle(content, "general-extractor"))
    assert landed is not None
    return landed.id


async def _retract_in_target(particle_id: str) -> None:
    from particles.core.status import Status, StatusReason
    from particles.db import session_scope, write_transaction
    from particles.store.particle_store import update_particle_status

    async with session_scope() as other, write_transaction(other):
        await update_particle_status(
            other, particle_id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )


def _rows(db: Path) -> list[tuple[str, str]]:
    con = sqlite3.connect(db)
    try:
        return list(con.execute("SELECT content, status FROM particles").fetchall())
    finally:
        con.close()


def _import(bundle: Path) -> None:
    from particles.api.cli import interchange as cli_interchange
    from particles.db import reset_engine

    try:
        asyncio.run(cli_interchange._import(bundle, "default"))
    finally:
        reset_engine()


def test_control_an_imported_claim_meeting_a_stored_contradiction_is_not_left_active_beside_it(
    tmp_path: Path, stores: Path
) -> None:
    """Control: with E stored before the import, the imported C is reconciled against it."""
    from particles.db import reset_engine

    bundle = _bundle(tmp_path)
    try:
        asyncio.run(_insert_into_target(E_TEXT))
    finally:
        reset_engine()
    _import(bundle)
    rows = _rows(stores)
    statuses = {c: s for c, s in rows}
    assert statuses[E_TEXT] == "ACTIVE", rows
    assert "INCONSISTENCY" in {s for _, s in rows} or statuses.get(C_TEXT) != "ACTIVE", rows


def test_an_import_reconciles_a_unit_against_a_claim_another_writer_committed_during_the_import(
    tmp_path: Path, stores: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An import's later unit is reconciled against a contradicting claim another writer
    committed after the import began (between two of its unit commits), so the imported
    claim is never left ACTIVE beside it with no INCONSISTENCY."""
    import particles.ingest.pipeline as pipeline

    bundle = _bundle(tmp_path)
    real = pipeline.gather_reconcile

    async def _hook(session: Any, particle: Any, *a: Any, **k: Any) -> Any:
        gathered = await real(session, particle, *a, **k)
        if particle.content == X_TEXT:
            await _insert_into_target(E_TEXT)
        return gathered

    monkeypatch.setattr(pipeline, "gather_reconcile", _hook)
    _import(bundle)
    rows = _rows(stores)
    statuses = {c: s for c, s in rows}
    assert statuses.get(E_TEXT) == "ACTIVE", rows
    assert "INCONSISTENCY" in {s for _, s in rows} or statuses.get(C_TEXT) != "ACTIVE", (
        "the imported claim landed ACTIVE beside a contradiction another writer committed "
        f"during the import, with no INCONSISTENCY: {rows}"
    )


def test_an_import_does_not_reconcile_a_unit_against_a_claim_retired_during_the_import(
    tmp_path: Path, stores: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An import's later unit is not reconciled against a claim another writer retracted
    after the import began: the imported claim is written ACTIVE, with no INCONSISTENCY
    naming the retracted claim and no failed import, as the baseline's re-read of the
    gathered conflict under the lock did."""
    import particles.ingest.pipeline as pipeline
    from particles.db import reset_engine

    bundle = _bundle(tmp_path)
    try:
        e_id = asyncio.run(_insert_into_target(E_TEXT))
    finally:
        reset_engine()
    real = pipeline.gather_reconcile

    async def _hook(session: Any, particle: Any, *a: Any, **k: Any) -> Any:
        gathered = await real(session, particle, *a, **k)
        if particle.content == X_TEXT:
            await _retract_in_target(e_id)
        return gathered

    monkeypatch.setattr(pipeline, "gather_reconcile", _hook)
    raised: BaseException | None = None
    try:
        _import(bundle)
    except BaseException as exc:  # typer.Exit on a refused import
        raised = exc
    rows = _rows(stores)
    statuses = {c: s for c, s in rows}
    assert statuses.get(E_TEXT) == "RETRACTED", rows
    assert raised is None, f"the import failed midway: {raised!r}; rows {rows}"
    assert statuses.get(C_TEXT) == "ACTIVE" and "INCONSISTENCY" not in {s for _, s in rows}, (
        "the imported claim was reconciled against a claim retracted during the import "
        f"(a stale candidate-cache entry): {rows}"
    )


def test_an_imported_retracted_unit_is_not_a_conflict_partner_for_a_later_unit(
    tmp_path: Path, stores: Path
) -> None:
    """L10: a unit the import keeps RETRACTED (as the bundle states) is not
    treated as an ACTIVE claim by a later unit of the same import: the later unit lands
    ACTIVE, with no INCONSISTENCY naming the retracted unit and no failed import."""
    import json

    from particles.db import reset_engine, session_scope
    from particles.interchange.store import export_store_bundle
    from particles.store.particle_store import insert_particle

    async def _go() -> dict[str, str]:
        async with session_scope("src") as s:
            await insert_particle(s, _particle(E_TEXT, "general-extractor"), _VE.tolist())
            await insert_particle(s, _particle(C_TEXT, "general-extractor"), _V1.tolist())
            await s.commit()
            return await export_store_bundle(s)

    try:
        files = asyncio.run(_go())
    finally:
        reset_engine()
    units = [json.loads(line) for line in files["particles.jsonl"].splitlines() if line]
    units.sort(key=lambda u: 0 if u["content"] == E_TEXT else 1)
    units[0]["status"] = "RETRACTED"
    units[0]["statusReason"] = "EXPLICIT_RETRACTION"
    files["particles.jsonl"] = "".join(json.dumps(u) + "\n" for u in units)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name, content in files.items():
        (bundle / name).write_text(content, encoding="utf-8")
    raised: BaseException | None = None
    try:
        _import(bundle)
    except BaseException as exc:
        raised = exc
    rows = _rows(stores)
    statuses = {c: s for c, s in rows}
    assert statuses.get(E_TEXT) == "RETRACTED", rows
    assert raised is None, f"the import failed midway: {raised!r}; rows {rows}"
    assert statuses.get(C_TEXT) == "ACTIVE" and "INCONSISTENCY" not in {s for _, s in rows}, (
        f"a later unit was reconciled against the import's own RETRACTED unit: {rows}"
    )
