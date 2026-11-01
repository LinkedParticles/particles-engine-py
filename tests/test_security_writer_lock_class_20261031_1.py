# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The store writer lock, the whole class: every belief writer gathers, then applies.

Security review 2026-10-31-1, AD1, AD2, AD3, AD4, AD5, AD6, AD15 and AD25.
Ported from the audit's proofs; each test states the safe property. A belief
write gathers before the store's writer lock (its subject lookups, the
embedding and the reconcile ladder's model probes) and applies under it, and
the apply runs the conflict search again against the store as it stands, so a
claim another writer committed or retired meanwhile is still reconciled and a
fail-closed caller quarantines it. ``tests/test_writer_lock_guard.py`` is the
table of writers; this file holds the window and SQLite-lock properties.

Scratch file-backed stores and key file under ``tmp_path``; configuration only
through the loader (``network.egress: none``); embedding, model and authority
stubbed; no outside call. A concurrent writer is driven from a hook installed
between the gather and the apply (a patched ``gather_reconcile`` or authority),
never from a timing race.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

LEGACY = "writer-lock-class-legacy-operator-key"

# Contents and their stub embeddings: C and D embed alike; E sits at cosine 0.9.
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


def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: str = "") -> Path:
    from particles.api import _keys
    from particles.config import reset_config

    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    home = tmp_path / "home"
    home.mkdir()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\nmcp:\n  write:\n    enabled_stores: [default]\n" + extra,
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    db = tmp_path / "blockw.db"
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

    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


@pytest.fixture
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    import particles.ingest.pipeline as pipeline

    async def _embed(texts: list[str]) -> list[Any]:
        return [_vector(t) for t in texts]

    monkeypatch.setattr(pipeline, "_embed_batch_async", _embed)


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Contradiction only between C and E; no update answer. Records every pair asked."""
    import particles.ingest.pipeline as pipeline

    asked: list[tuple[str, str]] = []

    async def _contra(a: str, b: str) -> bool | None:
        asked.append((a, b))
        return {a, b} == {C_TEXT, E_TEXT}

    async def _update(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(pipeline, "_has_contradiction_signal", _contra)
    monkeypatch.setattr(pipeline, "_has_update_signal", _update)
    return asked


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


def _agent_token(principal: str = "agent:audit12-s1") -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    return _keys.mint(Role.AGENT, principal)[0]


def _assert(client: TestClient, token: str, content: str, **extra: Any) -> Any:
    body = {
        "content": content,
        "subject_names": ["Acme widget"],
        "confidence": 0.7,
        "source_excerpt": content,
        **extra,
    }
    return client.post("/particles/assert", json=body, headers={"Authorization": f"Bearer {token}"})


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


def _statuses(db: Path) -> dict[str, str]:
    con = sqlite3.connect(db)
    try:
        return dict(con.execute("SELECT content, status FROM particles").fetchall())
    finally:
        con.close()


def test_an_agent_assertion_is_reconciled_against_a_contradiction_committed_after_its_gather(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: None,
    stub_embed: None,
    probes: list[tuple[str, str]],
) -> None:
    """An agent assertion is reconciled against a contradicting claim another writer
    committed after its gather and before its apply, so it is never written ACTIVE
    beside that claim with no INCONSISTENCY (section 6, fail-closed)."""
    import particles.ingest.pipeline as pipeline

    db = _store(tmp_path, monkeypatch)
    real = pipeline.gather_reconcile

    async def _hook(session: Any, particle: Any, *a: Any, **k: Any) -> Any:
        gathered = await real(session, particle, *a, **k)
        if particle.content == C_TEXT:
            await _insert_elsewhere(E_TEXT)
        return gathered

    monkeypatch.setattr(pipeline, "gather_reconcile", _hook)
    resp = _assert(_client(), _agent_token(), C_TEXT)
    assert resp.status_code < 300, resp.text
    rows = _statuses(db)
    assert rows.get(E_TEXT) == "ACTIVE", rows
    assert resp.json()["verdict"] == "INCONSISTENCY_RAISED", (
        f"the agent's claim landed {resp.json()['verdict']} / {rows.get(C_TEXT)} beside a "
        "contradiction committed between its gather and apply; the apply never searched again"
    )


def test_an_agent_assertion_whose_gathered_conflict_was_retired_is_reconciled_against_the_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: None,
    stub_embed: None,
    probes: list[tuple[str, str]],
) -> None:
    """An agent assertion whose gathered conflict another writer retired is reconciled
    against the store as it stands under the lock, so a second, contradicting ACTIVE
    claim the gather passed over is still probed and the write is quarantined."""
    import particles.ingest.pipeline as pipeline
    from particles.db import session_scope, write_transaction
    from particles.operations.agent_write import retract_belief

    db = _store(tmp_path, monkeypatch)
    principal = "agent:audit12-s1"
    token = _agent_token(principal)

    async def _seed() -> None:
        from particles.db import get_engine

        await _insert_elsewhere(E_TEXT)
        await get_engine().dispose()

    asyncio.run(_seed())
    client = _client()
    first = _assert(client, token, D_TEXT)
    assert first.status_code < 300, first.text
    d_id = first.json()["asserted_particle_id"]
    assert _statuses(db).get(D_TEXT) == "ACTIVE", _statuses(db)

    real = pipeline.gather_reconcile
    gathered_ids: list[str | None] = []

    async def _hook(session: Any, particle: Any, *a: Any, **k: Any) -> Any:
        gathered = await real(session, particle, *a, **k)
        if particle.content == C_TEXT:
            gathered_ids.append(gathered.conflict.id if gathered.conflict else None)
            # The same agent (or any writer) retracts the gathered pair meanwhile.
            async with session_scope() as other, write_transaction(other):
                await retract_belief(
                    other, store="default", particle_id=d_id, reason="x", identity=principal
                )
        return gathered

    monkeypatch.setattr(pipeline, "gather_reconcile", _hook)
    probes.clear()
    resp = _assert(client, token, C_TEXT)
    assert resp.status_code < 300, resp.text
    assert gathered_ids == [d_id], gathered_ids
    rows = _statuses(db)
    assert rows.get(E_TEXT) == "ACTIVE" and rows.get(D_TEXT) == "RETRACTED", rows
    assert resp.json()["verdict"] == "INCONSISTENCY_RAISED", (
        f"the agent's claim landed {resp.json()['verdict']} / {rows.get(C_TEXT)} beside the "
        f"contradicting ACTIVE claim; pairs probed: {probes}"
    )


class _JudgingAuthority:
    """A live authority that asks the real Wikidata judge on the session it is given.

    ``WikidataAuthority.resolve`` calls ``judge_candidates(session, ...)``, which
    records its answer in the probe ledger on that session and flushes
    (``wikidata_judge.py:303-312``). The first name is judged; at the second
    name the stub checks whether another connection can open a write
    transaction on the store while the writer lock is free.
    """

    NAMESPACE = "slow"
    PRIORITY = 0
    LIVE = True
    APPLICABILITY: list[Any] = []

    def __init__(self, db: Path) -> None:
        self.db = db
        self.calls = 0
        self.observed: dict[str, Any] = {}

    def uri_for(self, external_id: str) -> str | None:
        return None

    def recognize(self, name: str) -> Any:
        return None

    async def resolve(
        self, session: Any, name: str, *, particle_content: str | None, domain: str | None
    ) -> Any:
        from particles import db as dbmod
        from particles.ingest.authorities.wikidata_judge import judge_candidates

        self.calls += 1
        if self.calls == 1:
            await judge_candidates(
                session,
                name,
                particle_content or "",
                [{"id": "Q4242", "label": name, "description": "a thing"}],
            )
            return None
        lock = dbmod._file_write_locks.get("default")
        self.observed["writer_lock_held"] = bool(lock is not None and lock.is_locked)
        con = sqlite3.connect(self.db, timeout=0.5)
        try:
            con.execute("BEGIN IMMEDIATE")
            con.execute("ROLLBACK")
            self.observed["other_can_write"] = True
        except sqlite3.OperationalError as exc:
            self.observed["other_can_write"] = False
            self.observed["error"] = str(exc)
        finally:
            con.close()
        return None

    async def canonical_name_for(self, session: Any, external_id: str) -> str | None:
        return None


def test_an_agent_assertions_gather_holds_no_sqlite_write_lock_with_live_authorities_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: None, stub_embed: None
) -> None:
    """While an agent assertion's gather runs its live subject lookups, no SQLite write
    transaction is open on the store, so another writer that holds the writer lock can
    begin one (the AB68 shape), also when an operator turns live authorities on for
    conversation sources and the Wikidata judge records its answer."""
    import particles.ingest.authorities.wikidata_judge as judge
    import particles.ingest.subject_resolver as sr

    db = _store(tmp_path, monkeypatch, "subjects:\n  skip_live_authorities_source_types: []\n")

    async def _llm(*_a: Any, **_k: Any) -> str:
        return '{"qid": "Q4242"}'

    monkeypatch.setattr(judge, "_llm_call", _llm)
    authority = _JudgingAuthority(db)
    monkeypatch.setattr(sr, "get_authorities", lambda: [authority])
    resp = _assert(
        _client(),
        _agent_token(),
        "The gadget ships in blue.",
        subject_names=["Audit Twelve Gadget", "Audit Twelve Maker"],
    )
    assert resp.status_code < 300, resp.text
    assert authority.calls == 2, authority.calls
    assert authority.observed.get("writer_lock_held") is False, authority.observed
    assert authority.observed.get("other_can_write") is True, (
        "the gather opened SQLite's write transaction (the judge's ledger row) before the "
        f"writer lock and held it across the next live lookup: {authority.observed}"
    )


@pytest.fixture
def model_calls(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """One entry per ladder probe: whether the store's writer lock was held."""
    import particles.ingest.pipeline as pipeline
    from particles import db as dbmod

    seen: list[bool] = []

    async def _contra(a: str, b: str) -> bool | None:
        lock = dbmod._file_write_locks.get("default")
        seen.append(bool(lock is not None and lock.is_locked))
        return False

    async def _update(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(pipeline, "_has_contradiction_signal", _contra)
    monkeypatch.setattr(pipeline, "_has_update_signal", _update)
    return seen


def test_the_memory_facade_runs_no_ladder_probe_under_the_writer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: None,
    model_calls: list[bool],
) -> None:
    """The memory-server facade's assertions run the reconcile ladder's model probes
    before the store's writer lock, as the engine routes and LocalBackend do."""
    import mcp.types as types

    import particles.ingest.pipeline as pipeline
    from particles.db import reset_engine

    _store(tmp_path, monkeypatch)

    async def _embed(texts: list[str]) -> list[Any]:
        return [_V1 for _ in texts]

    monkeypatch.setattr(pipeline, "_embed_batch_async", _embed)

    async def _go() -> Any:
        from particles.mcp.memory_compat import server

        srv = server.build_server("default")
        call = srv.request_handlers[types.CallToolRequest]
        return await call(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="create_entities",
                    arguments={
                        "entities": [
                            {
                                "name": "Pluto",
                                "entityType": "planet",
                                "observations": [
                                    "Pluto orbits beyond Neptune",
                                    "Pluto has five known moons",
                                ],
                            }
                        ]
                    },
                ),
            )
        )

    result = asyncio.run(_go())
    reset_engine()
    assert not getattr(result.root, "isError", False), result
    assert model_calls, "no facade assertion reached the ladder's probe"
    assert not any(model_calls), (
        f"the facade held the store's writer lock across {sum(model_calls)} ladder probe(s)"
    )


def test_localbackend_runs_no_ladder_probe_under_the_writer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: None,
    model_calls: list[bool],
) -> None:
    """The stdio MCP write path (LocalBackend) probes before the store's writer lock."""
    import particles.ingest.pipeline as pipeline
    from particles.api.client.local import LocalBackend
    from particles.db import reset_engine

    _store(tmp_path, monkeypatch)

    async def _embed(texts: list[str]) -> list[Any]:
        return [_V1 for _ in texts]

    monkeypatch.setattr(pipeline, "_embed_batch_async", _embed)

    async def _go() -> None:
        backend = LocalBackend()
        for content in ("The widget ships in blue.", "The widget ships in red."):
            await backend.particle_assert(
                content=content,
                subject_names=["Widget"],
                confidence=0.7,
                source_excerpt=content,
                corpus_entry_id=None,
                uncertainty_nature="EPISTEMIC",
                tags=None,
                store="default",
            )

    asyncio.run(_go())
    reset_engine()
    assert model_calls, "the ladder made no probe"
    assert not any(model_calls), model_calls


def test_control_an_assertion_meeting_a_stored_contradiction_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: None,
    stub_embed: None,
    probes: list[tuple[str, str]],
) -> None:
    """Control for the two window tests: with the contradiction committed before the
    gather, the same agent assertion is quarantined, so the stubs reach the ladder."""

    _store(tmp_path, monkeypatch)

    async def _seed() -> None:
        from particles.db import get_engine

        await _insert_elsewhere(E_TEXT)
        await get_engine().dispose()

    asyncio.run(_seed())
    resp = _assert(_client(), _agent_token(), C_TEXT)
    assert resp.status_code < 300, resp.text
    assert resp.json()["verdict"] == "INCONSISTENCY_RAISED", resp.json()


class _LockSeeingAuthority:
    """A live authority that records whether the store's writer lock is held when asked."""

    NAMESPACE = "slow"
    PRIORITY = 0
    LIVE = True
    APPLICABILITY: list[Any] = []

    def __init__(self) -> None:
        self.held: list[bool] = []

    def uri_for(self, external_id: str) -> str | None:
        return None

    def recognize(self, name: str) -> Any:
        return None

    async def resolve(
        self, session: Any, name: str, *, particle_content: str | None, domain: str | None
    ) -> Any:
        from particles import db as dbmod

        lock = dbmod._file_write_locks.get("default")
        self.held.append(bool(lock is not None and lock.is_locked))
        return None

    async def canonical_name_for(self, session: Any, external_id: str) -> str | None:
        return None


def test_extraction_asks_no_live_subject_authority_under_the_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: None
) -> None:
    """The extraction write phase asks no live subject authority (a network lookup and,
    for Wikidata, the judge's model call) while it holds the store's writer lock."""
    import particles.ingest.pipeline as pipeline
    import particles.ingest.subject_resolver as sr
    from particles.core.schema import Mutability
    from particles.db import reset_engine

    _store(tmp_path, monkeypatch)

    async def _embed(texts: list[str]) -> list[Any]:
        return [_VX for _ in texts]

    monkeypatch.setattr(pipeline, "_embed_batch_async", _embed)
    authority = _LockSeeingAuthority()
    monkeypatch.setattr(sr, "get_authorities", lambda: [authority])
    from particles.core.schema import UncertaintyNature
    from particles.extraction.general import CandidateParticle, ExtractionResult

    class _Extractor:
        EXTRACTOR_ID = "audit12-stub"
        EXTRACTOR_VERSION = "0.0.1"

        def accepts(self, source_type: str) -> bool:
            return True

        async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> Any:
            return ExtractionResult(
                candidates=[
                    CandidateParticle(
                        content="The Zorblax compressor was released by Quuxco in 2019.",
                        confidence_value=0.9,
                        uncertainty_nature=UncertaintyNature.EPISTEMIC,
                        subjects=["Zorblax compressor", "Quuxco"],
                    )
                ]
            )

    async def _go() -> None:
        from particles.corpus.deposit import deposit_text_versioned
        from particles.db import session_scope
        from particles.ingest.pipeline import extract_snapshot

        async with session_scope() as session:
            entry_id, snapshot_id, _same = await deposit_text_versioned(
                session,
                text="Quuxco released the Zorblax compressor in 2019.",
                uri_r="https://example.invalid/zorblax",
                source_type="LOCAL_MARKDOWN",
                mutability=Mutability.APPEND_ONLY,
                tags=[],
            )
            await session.commit()
            await extract_snapshot(session, entry_id, snapshot_id, extractor=_Extractor())
            await session.commit()

    try:
        asyncio.run(_go())
    finally:
        reset_engine()
    assert authority.held, "extraction asked no live authority; the path is not exercised"
    assert not any(authority.held), (
        f"{sum(authority.held)} of {len(authority.held)} live subject lookup(s) ran while "
        "the extraction held the store's writer lock"
    )


# S6: the in-process callers

VECTOR = [0.1, 0.2, 0.3, 0.4]


def _create_tables_inproc() -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    asyncio.run(_go())


@pytest.fixture
def inproc_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles import embeddings as ep
    from particles.config import reset_config

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\nmcp:\n  write:\n    enabled_stores: [default]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'inproc.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _create_tables_inproc()
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, *a, **k: np.array(
            [VECTOR] * (len(texts) if isinstance(texts, list) else 1), dtype=np.float32
        )
    )
    original = ep._embedding_model
    ep.set_embedding_model(model)
    yield tmp_path
    ep.set_embedding_model(original)
    reset_config()


def _lock_held() -> bool:
    import particles.db as db

    lock = db._file_write_locks.get("default")
    return bool(lock is not None and lock.is_locked)


@pytest.fixture
def llm_calls(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """One entry per model call: whether the store's writer lock was held."""
    import particles.llm as llm_pkg

    seen: list[bool] = []

    async def _complete(*_a: Any, **_k: Any) -> str:
        seen.append(_lock_held())
        return "VERDICT: NO"

    monkeypatch.setattr(llm_pkg, "complete", _complete)
    return seen


def _backend() -> Any:
    from particles.api.client.local import LocalBackend

    return LocalBackend()


async def _local_assert(content: str, names: list[str] | None = None) -> Any:
    return await _backend().particle_assert(
        content=content,
        subject_names=names if names is not None else ["Widget"],
        confidence=0.7,
        source_excerpt=content,
        corpus_entry_id=None,
        uncertainty_nature="EPISTEMIC",
        tags=None,
        store="default",
    )


# --------------------------------------------------------------------------- #
# The native stdio MCP server, through LocalBackend
# --------------------------------------------------------------------------- #


def test_local_backend_runs_no_model_call_under_the_writer_lock(
    inproc_store: Path, llm_calls: list[bool]
) -> None:
    """The stdio MCP server's in-process assert and supersede probe before the writer lock."""

    async def _go() -> None:
        first = await _local_assert("The widget ships in blue.")
        await _local_assert("The widget ships in red.")
        await _backend().particle_supersede(
            supersedes_id=first.asserted_particle_id,
            content="The widget ships in green.",
            subject_names=["Widget"],
            confidence=0.7,
            source_excerpt="it ships in green",
            corpus_entry_id=None,
            uncertainty_nature="EPISTEMIC",
            tags=None,
            store="default",
            reason="repainted",
        )

    asyncio.run(_go())
    assert llm_calls, "no in-process write reached the ladder's model probe"
    assert not any(llm_calls), llm_calls


def test_local_backend_supersede_refuses_a_predecessor_retired_since_the_gather(
    inproc_store: Path, llm_calls: list[bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-process supersede whose predecessor was retired after the gather is refused."""
    import particles.ingest.pipeline as pipeline
    from particles.core.status import Status, StatusReason
    from particles.db import session_scope
    from particles.store.particle_store import get_particle, update_particle_status

    first = asyncio.run(_local_assert("The widget ships in blue."))
    pid = first.asserted_particle_id
    real = pipeline.gather_reconcile

    async def _hooked(session: Any, particle: Any, *a: Any, **k: Any) -> Any:
        gathered = await real(session, particle, *a, **k)
        async with session_scope("default") as other:
            await update_particle_status(
                other, pid, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
            )
            await other.commit()
        return gathered

    monkeypatch.setattr(pipeline, "gather_reconcile", _hooked)

    async def _go() -> Any:
        return await _backend().particle_supersede(
            supersedes_id=pid,
            content="The widget ships in green.",
            subject_names=["Widget"],
            confidence=0.7,
            source_excerpt="it ships in green",
            corpus_entry_id=None,
            uncertainty_nature="EPISTEMIC",
            tags=None,
            store="default",
            reason="repainted",
        )

    with pytest.raises(ValueError, match="not ACTIVE"):
        asyncio.run(_go())

    async def _status() -> Any:
        async with session_scope("default") as s:
            return await get_particle(s, pid)

    stored = asyncio.run(_status())
    assert stored is not None and stored.status is Status.RETRACTED


def _conflict_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every contradiction probe says YES; the update probe has no answer."""
    import particles.ingest.pipeline as pipeline

    async def _yes(*_a: Any, **_k: Any) -> bool:
        return True

    async def _none(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(pipeline, "_has_contradiction_signal", _yes)
    monkeypatch.setattr(pipeline, "_has_update_signal", _none)


async def _other_writer_claim(content: str) -> tuple[str, str]:
    """Another principal mints Subject "Widget" and writes an ACTIVE claim about it; commits."""
    from particles.core.schema import Confidence, Particle, SourceType, UncertaintyNature
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status
    from particles.db import session_scope
    from particles.ingest.subject_resolver import resolve_subjects
    from particles.store.particle_store import insert_particle

    async with session_scope("default") as other:
        ids = await resolve_subjects(
            other, ["Widget"], "agent:other", source_type=SourceType.CONVERSATION
        )
        claim = Particle(
            content=content,
            confidence=Confidence(value=0.7, calibration_source=CalibrationSource.AGENT_ASSERTED),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="agent:other",
            status=Status.ACTIVE,
            subject_ids=ids,
        )
        await insert_particle(other, claim, VECTOR)
        await other.commit()
        return claim.id, ids[0]


def test_control_local_backend_quarantines_a_contradiction_already_stored(
    inproc_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: with the contradicting claim stored before the call, the write is quarantined."""
    _conflict_stubs(monkeypatch)
    asyncio.run(_other_writer_claim("The widget ships in red."))
    result = asyncio.run(_local_assert("The widget ships in blue."))
    assert result.verdict == "INCONSISTENCY_RAISED", result


def test_local_backend_assert_reconciles_against_a_subject_and_conflict_written_since_the_gather(
    inproc_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-process agent assertion never lands ACTIVE beside a contradiction on its own subject.

    The gather resolves "Widget" to nothing (no such Subject yet) and finds no
    conflict. Before the apply, another writer mints "Widget" and writes a
    contradicting claim about it. The apply links the belief to that Subject,
    a subject set the gather's search never ran on, and must still reconcile
    against the claim it now sits beside.
    """
    import particles.ingest.pipeline as pipeline
    from particles.core.status import Status
    from particles.db import session_scope
    from particles.store.particle_store import get_particle

    _conflict_stubs(monkeypatch)
    real = pipeline.gather_reconcile
    seen: dict[str, Any] = {}

    async def _hooked(session: Any, particle: Any, *a: Any, **k: Any) -> Any:
        gathered = await real(session, particle, *a, **k)
        seen["draft_subjects"] = list(particle.subject_ids)
        seen["claim"], seen["subject"] = await _other_writer_claim("The widget ships in red.")
        return gathered

    monkeypatch.setattr(pipeline, "gather_reconcile", _hooked)
    result = asyncio.run(_local_assert("The widget ships in blue."))

    async def _read() -> Any:
        async with session_scope("default") as s:
            return await get_particle(s, result.asserted_particle_id)

    written = asyncio.run(_read())
    assert seen["draft_subjects"] == [], seen
    assert written is not None and seen["subject"] in written.subject_ids
    assert not (result.verdict == "ASSERTED" and written.status is Status.ACTIVE), (
        "an in-process agent belief landed ACTIVE beside a contradicting claim on the "
        "subject it was linked to under the lock, with no INCONSISTENCY",
        result,
    )


# --------------------------------------------------------------------------- #
# The memory-server facade
# --------------------------------------------------------------------------- #


def _facade_call(name: str, arguments: dict[str, Any]) -> Any:
    import mcp.types as types

    from particles.mcp.memory_compat import server

    async def _go() -> Any:
        srv = server.build_server()
        call = srv.request_handlers[types.CallToolRequest]
        return await call(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(name=name, arguments=arguments),
            )
        )

    return asyncio.run(_go())


def test_facade_assertions_run_no_model_call_under_the_writer_lock(
    inproc_store: Path, llm_calls: list[bool]
) -> None:
    """A memory-server facade write runs each assertion's model probe outside the writer lock."""
    first = _facade_call(
        "create_entities",
        {
            "entities": [
                {
                    "name": "Widget",
                    "entityType": "product",
                    "observations": ["The widget ships in blue."],
                }
            ]
        },
    )
    assert not first.root.isError, first.root
    second = _facade_call(
        "add_observations",
        {
            "observations": [
                {
                    "entityName": "Widget",
                    "contents": [
                        "The widget ships in red.",
                        "The widget ships in green.",
                        "The widget ships in grey.",
                    ],
                }
            ]
        },
    )
    assert not second.root.isError, second.root
    assert llm_calls, "no facade assertion reached the ladder's model probe"
    held = sum(llm_calls)
    assert held == 0, (
        f"{held} of {len(llm_calls)} model calls of one facade write ran while the "
        "store's writer lock was held"
    )


# --------------------------------------------------------------------------- #
# particles curate apply supersede
# --------------------------------------------------------------------------- #


@pytest.fixture
def commits(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """One entry per ``AsyncSession.commit``: whether the store's writer lock was held."""
    from sqlalchemy.ext.asyncio import AsyncSession

    seen: list[bool] = []
    real = AsyncSession.commit

    async def _commit(self: AsyncSession) -> None:
        seen.append(_lock_held())
        await real(self)

    monkeypatch.setattr(AsyncSession, "commit", _commit)
    return seen


def test_curate_apply_supersede_commits_its_belief_under_the_writer_lock(
    inproc_store: Path, llm_calls: list[bool], commits: list[bool]
) -> None:
    """``curate apply supersede`` commits the operator's successor belief under the writer lock."""
    from particles.api.cli import curate
    from particles.operations.curation.session import BeliefRevision

    first = asyncio.run(_local_assert("The widget ships in blue."))
    commits.clear()
    asyncio.run(
        curate._apply(
            gesture="supersede",
            card_key=f"stale:{first.asserted_particle_id}",
            reason="the catalogue changed",
            days=None,
            revision=BeliefRevision(content="The widget ships in green.", confidence=0.8),
        )
    )
    assert commits, "the gesture committed nothing"
    assert all(commits), (
        "curate apply supersede committed an operator belief with the store's writer "
        f"lock free: {commits}"
    )


# S6: interchange import


def _create_tables_for(store: str = "default") -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine(store).begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine(store).dispose()

    asyncio.run(_go())


@pytest.fixture
def embedding() -> Generator[None, None, None]:
    from particles import embeddings as ep

    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, *a, **k: np.array(
            [VECTOR] * (len(texts) if isinstance(texts, list) else 1), dtype=np.float32
        )
    )
    original = ep._embedding_model
    ep.set_embedding_model(model)
    yield
    ep.set_embedding_model(original)


def _configure_callers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: str = "") -> None:
    from particles.api import _keys
    from particles.config import reset_config

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\nmcp:\n  write:\n    enabled_stores: [default]\n" + extra,
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'default.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()


@pytest.fixture
def callers_cleanup() -> Generator[None, None, None]:
    yield
    from particles.api import _keys
    from particles.config import reset_config

    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


def _sqlite_write_locked(path: Path) -> bool:
    """Whether another connection is refused SQLite's write lock on ``path`` right now."""
    conn = sqlite3.connect(path, timeout=0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
        return False
    except sqlite3.OperationalError:
        return True
    finally:
        conn.close()


def test_interchange_import_holds_no_sqlite_write_lock_across_a_model_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedding: None, callers_cleanup: None
) -> None:
    """``particles interchange import`` holds no SQLite write lock across a unit's model probe."""
    import particles.llm as llm_pkg
    from particles.api.cli import interchange as cli_interchange
    from particles.core.schema import Confidence, Particle, UncertaintyNature
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status
    from particles.db import session_scope
    from particles.interchange.store import export_store_bundle
    from particles.store.particle_store import insert_particle

    dest = tmp_path / "dest.db"
    _configure_callers(
        tmp_path,
        monkeypatch,
        f"storage:\n  stores:\n    dest: sqlite+aiosqlite:///{dest}\n",
    )
    _create_tables_for("default")
    _create_tables_for("dest")

    async def _seed_and_export() -> dict[str, str]:
        async with session_scope("default") as s:
            for who, content in (
                ("agent:a", "The widget ships in blue."),
                ("agent:b", "The widget ships in red."),
            ):
                await insert_particle(
                    s,
                    Particle(
                        content=content,
                        confidence=Confidence(
                            value=0.7, calibration_source=CalibrationSource.AGENT_ASSERTED
                        ),
                        uncertainty_nature=UncertaintyNature.EPISTEMIC,
                        asserted_by=who,
                        status=Status.ACTIVE,
                    ),
                    VECTOR,
                )
            await s.commit()
            return await export_store_bundle(s)

    files = asyncio.run(_seed_and_export())
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name, content in files.items():
        (bundle / name).write_text(content, encoding="utf-8")

    seen: list[bool] = []

    async def _complete(*_a: Any, **_k: Any) -> str:
        seen.append(_sqlite_write_locked(dest))
        return "VERDICT: NO"

    monkeypatch.setattr(llm_pkg, "complete", _complete)
    asyncio.run(cli_interchange._import(bundle, "dest"))
    rows = sqlite3.connect(dest).execute("select count(*) from particles").fetchone()[0]
    assert rows == 2, rows
    assert seen, "the import made no model call"
    held = sum(seen)
    assert held == 0, (
        f"{held} of {len(seen)} model calls of one import ran while the import held "
        "SQLite's write lock on the target store, outside the writer lock"
    )
