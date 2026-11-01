# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""A belief write holds the store's writer lock for its writes alone.

Security review 2026-10-30-2, AC1, AC3 and AC7.

The writer lock wraps a write transaction, never the model, fetch or
embedding phase. An agent assert or supersede, and the operator supersede, ask
the live subject authorities and run the reconcile ladder's model probes before
taking it, then write under it; a body refused on its own terms takes no lock
at all. Scratch store and key file under ``tmp_path``; configuration only
through the loader; embedding, model and authority stubbed; no outside call.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from particles.core.schema import ExternalRef
from particles.ingest.authorities import AuthorityResolution

LEGACY = "sec11-writer-lock-legacy-operator-key"


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
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\nmcp:\n  write:\n    enabled_stores: [default]\n" + extra,
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'lock.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", LEGACY)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    return keydir


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles.api import _keys
    from particles.config import reset_config

    yield _store(tmp_path, monkeypatch)
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


@pytest.fixture
def same_embedding() -> Generator[None, None, None]:
    """Every embedding is the same vector, so a second claim meets the first in the ladder."""
    from particles import embeddings as ep

    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, *a, **k: np.array(
            [[0.1, 0.2, 0.3, 0.4]] * (len(texts) if isinstance(texts, list) else 1),
            dtype=np.float32,
        )
    )
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


@pytest.fixture
def model_calls(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """One entry per model call: whether the store's writer lock was held."""
    import particles.llm as llm_pkg

    seen: list[bool] = []

    async def _complete(*_a: Any, **_k: Any) -> str:
        seen.append(_lock_held())
        return "VERDICT: NO"

    monkeypatch.setattr(llm_pkg, "complete", _complete)
    return seen


@pytest.fixture
def lock_entries(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Each store whose writer lock a request entered, read off the real write_lock."""
    import particles.db as db

    entered: list[str] = []
    real = db.write_lock

    def _recording(store: Any = db.DEFAULT_STORE) -> Any:
        entered.append(str(store))
        return real(store)

    monkeypatch.setattr(db, "write_lock", _recording)
    return entered


def _lock_held() -> bool:
    import particles.db as db

    lock = db._file_write_locks.get("default")
    return bool(lock is not None and lock.is_locked)


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


def _agent_token(principal: str = "agent:sec11-writer-lock") -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    return _keys.mint(Role.AGENT, principal)[0]


def _assert(client: TestClient, token: str, content: str, **extra: Any) -> Any:
    body = {
        "content": content,
        "subject_names": ["Widget"],
        "confidence": 0.7,
        "source_excerpt": content,
        **extra,
    }
    return client.post("/particles/assert", json=body, headers={"Authorization": f"Bearer {token}"})


class _SlowAuthority:
    """A live authority whose every lookup takes ``delay`` seconds and finds nothing."""

    NAMESPACE = "slow"
    PRIORITY = 0
    LIVE = True
    APPLICABILITY: list[Any] = []

    def __init__(self, delay: float, answer: AuthorityResolution | None = None) -> None:
        self.delay = delay
        self.answer = answer
        self.calls = 0
        self.entered = threading.Event()

    def uri_for(self, external_id: str) -> str | None:
        return None

    def recognize(self, name: str) -> ExternalRef | None:
        return None

    async def resolve(
        self, session: Any, name: str, *, particle_content: str | None, domain: str | None
    ) -> AuthorityResolution | None:
        self.calls += 1
        self.entered.set()
        await asyncio.sleep(self.delay)
        return self.answer

    async def canonical_name_for(self, session: Any, external_id: str) -> str | None:
        return None


def test_no_model_call_of_an_agent_assert_runs_under_the_writer_lock(
    store: Path, same_embedding: None, model_calls: list[bool]
) -> None:
    """The reconcile ladder's model probes of ``/particles/assert`` run before the lock."""
    token = _agent_token()
    client = _client()
    for content in ("The widget ships in blue.", "The widget ships in red."):
        resp = _assert(client, token, content)
        assert resp.status_code < 300, (resp.status_code, resp.text)
    assert model_calls, "the ladder made no model call; the probe is not exercised"
    assert not any(model_calls), model_calls


def test_no_model_call_of_a_supersede_runs_under_the_writer_lock(
    store: Path, same_embedding: None, model_calls: list[bool]
) -> None:
    """The agent and the operator supersede probe their successor before the lock."""
    token = _agent_token()
    client = _client()
    ids = []
    for content in ("The widget ships in blue.", "The widget ships in red."):
        resp = _assert(client, token, content)
        assert resp.status_code < 300, (resp.status_code, resp.text)
        ids.append(resp.json()["asserted_particle_id"])
    model_calls.clear()

    resp = client.post(
        "/particles/supersede",
        json={
            "supersedes_id": ids[0],
            "content": "The widget ships in green.",
            "subject_names": ["Widget"],
            "confidence": 0.7,
            "source_excerpt": "it ships in green",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code < 300, (resp.status_code, resp.text)
    resp = client.post(
        f"/particles/{ids[1]}/supersede",
        json={
            "content": "The widget ships in grey.",
            "subject_names": ["Widget"],
            "confidence": 0.7,
            "source_excerpt": "it ships in grey",
            "reason": "the catalogue changed",
        },
        headers={"Authorization": f"Bearer {LEGACY}"},
    )
    assert resp.status_code < 300, (resp.status_code, resp.text)
    assert model_calls, "no supersede reached the ladder's model probe"
    assert not any(model_calls), model_calls


def test_a_second_writer_gets_the_lock_within_its_timeout_during_one_agent_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_embedding: None
) -> None:
    """While one assertion's subject lookups run, another process's writer gets the lock in time.

    A live authority taking 0.1 s per name stands in for the network lookup;
    the assertion names as many subjects as ``mcp.write.max_subject_names``
    allows, so its lookups outlast the one-second lock timeout. Another
    process's writer is stood in by a second ``FileLock`` on the lock file.
    """
    from filelock import FileLock, Timeout

    import particles.ingest.subject_resolver as sr
    from particles import db
    from particles.api import _keys
    from particles.config import get_config, reset_config

    _store(
        tmp_path,
        monkeypatch,
        "storage:\n  write_lock:\n    timeout_seconds: 1.0\n"
        "subjects:\n  skip_live_authorities_source_types: []\n",
    )
    try:
        authority = _SlowAuthority(0.1)
        monkeypatch.setattr(sr, "get_authorities", lambda: [authority])
        token = _agent_token("agent:ci-bot")
        limit = get_config().mcp.write.max_subject_names
        names = [f"Unknown Thing {i}" for i in range(limit)]
        outcome: dict[str, Any] = {}

        def _run() -> None:
            outcome["resp"] = _assert(
                _client(), token, "The widget ships in blue.", subject_names=names
            )

        worker = threading.Thread(target=_run)
        worker.start()
        try:
            assert authority.entered.wait(20), "the assertion never reached a live lookup"
            time.sleep(0.2)
            other = FileLock(db._write_lock_path("default"))
            timeout = get_config().storage.write_lock.timeout_seconds
            try:
                other.acquire(timeout=timeout)
                got = True
                other.release()
            except Timeout:
                got = False
        finally:
            worker.join(60)
        assert outcome["resp"].status_code < 300, outcome["resp"].text
        assert authority.calls == len(names), authority.calls
        assert got, (
            f"another writer waited {timeout}s for the store's writer lock while one agent "
            f"assertion naming {len(names)} subjects held it"
        )
    finally:
        _keys.reset_key_cache()
        _keys.forget_key_files()
        reset_config()


def test_an_assertion_naming_too_many_subjects_is_refused_without_the_lock(
    store: Path, same_embedding: None, lock_entries: list[str]
) -> None:
    """``mcp.write.max_subject_names`` bounds the lookups one request can make."""
    from particles.config import get_config

    names = [f"Thing {i}" for i in range(get_config().mcp.write.max_subject_names + 1)]
    resp = _assert(_client(), _agent_token(), "The widget ships in blue.", subject_names=names)
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert "max_subject_names" in resp.json()["detail"]
    assert lock_entries == [], lock_entries


def test_a_body_refused_by_the_granularity_gate_takes_no_writer_lock(
    store: Path, same_embedding: None, lock_entries: list[str]
) -> None:
    """A ``/particles/assert`` body refused for its size alone takes no writer lock."""
    resp = _assert(
        _client(),
        _agent_token(),
        "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. " * 40,
        source_excerpt="x",
    )
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert lock_entries == [], lock_entries


def test_an_operator_supersede_with_an_empty_reason_takes_no_writer_lock(
    store: Path, same_embedding: None, lock_entries: list[str]
) -> None:
    """The operator supersede's empty-reason 400 (a body-only refusal) takes no writer lock."""
    resp = _client().post(
        "/particles/p-0000000000000000/supersede",
        json={
            "content": "The widget ships in green.",
            "subject_names": ["Widget"],
            "confidence": 0.7,
            "reason": "   ",
        },
        headers={"Authorization": f"Bearer {LEGACY}"},
    )
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert lock_entries == [], lock_entries


async def test_the_replay_asks_no_authority_and_keeps_its_answer(db_session: Any) -> None:
    """Resolution under the lock replays the prefetched answer and calls no authority."""
    from particles.ingest.subject_resolver import prefetch_subjects, resolve_subjects
    from particles.store.subject_store import get_subject

    ref = ExternalRef(namespace="slow", id="S1", uri=None, confidence=0.95)
    authority = _SlowAuthority(
        0.0, AuthorityResolution(external_ref=ref, canonical_name="Acme Widget")
    )
    with patch("particles.ingest.subject_resolver.get_authorities", return_value=[authority]):
        prefetch = await prefetch_subjects(db_session, ["acme widget"])
        assert authority.calls == 1
        assert prefetch.known_ids == []
        assert not db_session.new, "the prefetch wrote"
        ids = await resolve_subjects(db_session, ["acme widget"], answers=prefetch.answers)
    assert authority.calls == 1, "the replay asked the authority again"
    subject = await get_subject(db_session, ids[0])
    assert subject is not None
    assert subject.canonical_name == "Acme Widget"
    assert [e.id for e in subject.external_ids] == ["S1"]


async def test_a_conflict_retired_since_the_gather_is_no_conflict(
    db_session: Any, same_embedding: None
) -> None:
    """The apply searches again under the lock, so a gathered conflict another writer
    retired meanwhile is gone; with no other claim near, the particle is inserted.
    The window's other half (another near claim left ACTIVE) is quarantined:
    tests/test_security_writer_lock_class_20261031_1.py (security review
    2026-10-31-1, AD1)."""
    from particles.core.schema import Confidence, Particle, UncertaintyNature
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status, StatusReason
    from particles.ingest.pipeline import (
        apply_reconcile,
        gather_reconcile,
        reconcile_and_insert,
    )
    from particles.store.particle_store import get_particle, update_particle_status

    def _claim(content: str) -> Particle:
        return Particle(
            content=content,
            confidence=Confidence(value=0.7, calibration_source=CalibrationSource.AGENT_ASSERTED),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="agent:a",
            status=Status.ACTIVE,
        )

    async def _yes(*_a: Any, **_k: Any) -> bool:
        return True

    first = await reconcile_and_insert(db_session, _claim("The widget ships in blue."))
    assert first is not None
    second = _claim("The widget ships in red.")
    with patch("particles.ingest.pipeline._has_contradiction_signal", _yes):
        gathered = await gather_reconcile(db_session, second)
    assert gathered.conflict is not None and gathered.conflict.id == first.id
    await update_particle_status(
        db_session, first.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    result = await apply_reconcile(db_session, second, gathered, fail_closed=True)
    assert result is not None and result.id == second.id
    stored = await get_particle(db_session, second.id)
    assert stored is not None and stored.status is Status.ACTIVE
