# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The extraction write phase searches the store again under the writer lock.

Security review 2026-10-31-2, AE2: a claim committed after a pass's probes and
before its write phase is reconciled, not left ACTIVE beside the extracted
claim with no INCONSISTENCY. Scratch file-backed store under ``tmp_path``,
configuration through the loader (``network.egress: none``), the embedder and
the ladder's probes stubbed.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

E_TEXT = "The Zorblax compressor shipped only in 2021."
C_TEXT = "The Zorblax compressor shipped in 2019."
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


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    import particles.ingest.pipeline as pipeline
    from particles.config import reset_config
    from particles.db import reset_engine

    home = tmp_path / "home"
    home.mkdir()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\nsubjects:\n  skip_live_authorities_source_types: []\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    db = tmp_path / "extract.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _create_tables()

    async def _embed(texts: list[str]) -> list[Any]:
        return [_vector(t) for t in texts]

    async def _contra(a: str, b: str) -> bool | None:
        return (C_TEXT in a and E_TEXT in b) or (C_TEXT in b and E_TEXT in a)

    async def _update(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(pipeline, "_embed_batch_async", _embed)
    monkeypatch.setattr(pipeline, "_has_contradiction_signal", _contra)
    monkeypatch.setattr(pipeline, "_has_update_signal", _update)
    yield db
    reset_config()
    reset_engine()


def _extractor(claims: list[tuple[str, list[str]]]) -> Any:
    from particles.core.schema import UncertaintyNature
    from particles.extraction.general import CandidateParticle, ExtractionResult

    class _Extractor:
        EXTRACTOR_ID = "sec13-writers-stub"
        EXTRACTOR_VERSION = "0.0.1"

        def accepts(self, source_type: str) -> bool:
            return True

        async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> Any:
            return ExtractionResult(
                candidates=[
                    CandidateParticle(
                        content=text,
                        confidence_value=0.9,
                        uncertainty_nature=UncertaintyNature.EPISTEMIC,
                        subjects=names,
                    )
                    for text, names in claims
                ]
            )

    return _Extractor()


def _extract(text: str, extractor: Any) -> None:
    async def _go() -> None:
        from particles.core.schema import Mutability
        from particles.corpus.deposit import deposit_text_versioned
        from particles.db import reset_engine, session_scope
        from particles.ingest.pipeline import extract_snapshot

        try:
            async with session_scope() as session:
                entry_id, snapshot_id, _same = await deposit_text_versioned(
                    session,
                    text=text,
                    uri_r="https://example.invalid/zorblax",
                    source_type="LOCAL_MARKDOWN",
                    mutability=Mutability.APPEND_ONLY,
                    tags=[],
                )
                await session.commit()
                await extract_snapshot(session, entry_id, snapshot_id, extractor=extractor)
                await session.commit()
        finally:
            reset_engine()

    asyncio.run(_go())


def test_extraction_write_phase_reconciles_against_a_contradiction_committed_before_its_lock(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extracted claim is reconciled against a contradicting claim another writer
    committed after the pass's probes and before its write phase, so it is not written
    ACTIVE beside that claim with no INCONSISTENCY (B21: a belief write searches again
    under the lock and acts on the store as it stands)."""
    import particles.ingest.pipeline as pipeline
    import particles.ingest.subject_resolver as sr

    monkeypatch.setattr(sr, "get_authorities", lambda: [])
    real = pipeline.prefetch_subjects
    fired: list[str] = []

    async def _other_writer() -> None:
        from particles.core.schema import Confidence, Particle, UncertaintyNature
        from particles.core.scoring.confidence import CalibrationSource
        from particles.core.status import Status
        from particles.db import session_scope, write_transaction
        from particles.ingest.pipeline import reconcile_and_insert

        particle = Particle(
            content=E_TEXT,
            confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="general-extractor",
            status=Status.ACTIVE,
        )
        async with session_scope() as other, write_transaction(other):
            landed = await reconcile_and_insert(other, particle, fail_closed=True)
        assert landed is not None
        fired.append(landed.id)

    async def _hook(*a: Any, **k: Any) -> Any:
        out = await real(*a, **k)
        if not fired and C_TEXT in str(k.get("particle_content")):
            await _other_writer()
        return out

    monkeypatch.setattr(pipeline, "prefetch_subjects", _hook)
    _extract(
        "Quuxco released the Zorblax compressor in 2019.",
        _extractor([(C_TEXT, ["Zorblax compressor"])]),
    )
    assert fired, "the hook never ran"
    con = sqlite3.connect(store)
    try:
        rows = con.execute("SELECT content, status FROM particles").fetchall()
    finally:
        con.close()
    active = {c for c, s in rows if s == "ACTIVE"}
    inconsistency = [c for c, s in rows if s == "INCONSISTENCY"]
    assert not ({C_TEXT, E_TEXT} <= active and not inconsistency), (
        f"the extracted claim landed ACTIVE beside a contradiction committed before its "
        f"write phase, with no INCONSISTENCY: {rows}"
    )


def test_control_extraction_meeting_a_stored_contradiction_is_reconciled(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: with the contradiction stored before the pass, the same claim is
    reconciled (an INCONSISTENCY, or not both ACTIVE), so the window test isolates the
    missing search under the lock."""
    import particles.ingest.subject_resolver as sr

    monkeypatch.setattr(sr, "get_authorities", lambda: [])
    _extract("The Zorblax compressor shipped only in 2021.", _extractor([(E_TEXT, [])]))
    _extract(
        "Quuxco released the Zorblax compressor in 2019.",
        _extractor([(C_TEXT, ["Zorblax compressor"])]),
    )
    con = sqlite3.connect(store)
    try:
        rows = con.execute("SELECT content, status FROM particles").fetchall()
    finally:
        con.close()
    active = {c for c, s in rows if s == "ACTIVE"}
    inconsistency = [c for c, s in rows if s == "INCONSISTENCY"]
    assert not ({C_TEXT, E_TEXT} <= active and not inconsistency), rows
