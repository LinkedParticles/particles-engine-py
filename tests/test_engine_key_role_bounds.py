# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What a reader or agent key's request may write, lock and read.

Security review 2026-10-27-1: G7 / G32 / G38 (a reader key's narrative read
writes no synthesis-cache row), G36 (a refused belief write takes no writer
lock) and G47 (a non-operator key is served no other key's id from the event
log).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from particles.api import _keys
from particles.api._keys import Role
from tests.test_engine_keys import _bearer, _mint, engine, keys_file  # noqa: F401

AGENT = "agent:ci-bot"


async def _seed_narrative() -> tuple[str, list[str]]:
    from particles.core.schema import (
        Confidence,
        Particle,
        ParticleType,
        RelationCreatedBy,
        RelationType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation

    def mk(content: str, kind: ParticleType | None = None) -> Particle:
        extra: dict[str, Any] = {} if kind is None else {"particle_type": kind}
        return Particle(
            content=content,
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            **extra,
        )

    narrative = mk("A day.", ParticleType.NARRATIVE)
    c1, c2 = mk("Woke up."), mk("Went out.")
    async with session_scope() as session:
        for p in (narrative, c1, c2):
            await insert_particle(session, p)
        for c in (c1, c2):
            await create_relation(
                session, c.id, narrative.id, RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
            )
        await create_relation(
            session, c1.id, c2.id, RelationType.SEQUENCE_IN, RelationCreatedBy.MANUAL_CLI
        )
        await session.commit()
    return narrative.id, [c1.id, c2.id]


async def _cache_rows() -> int:
    from sqlalchemy import func, select

    from particles.db import get_engine, session_scope
    from particles.store.synthesis_cache_store import SynthesisCacheRow

    try:
        async with session_scope() as session:
            count = await session.execute(select(func.count()).select_from(SynthesisCacheRow))
            return int(count.scalar_one())
    finally:
        await get_engine().dispose()


def _stub_model(text: str) -> MagicMock:
    import anthropic

    content = MagicMock()
    content.text = text
    resp = MagicMock()
    resp.content = [content]
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(return_value=resp)
    return client


class TestNarrativeCacheWrites:
    """G7, G32 and G38: one defect, the reader key's narrative cache write."""

    @pytest.mark.parametrize(("role", "persists"), [("reader", False), ("agent", True)])
    def test_only_a_key_that_can_write_persists_the_narrative_cache(
        self,
        engine: TestClient,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
        role: str,
        persists: bool,
    ) -> None:
        from particles.config import get_config
        from particles.llm import set_client

        monkeypatch.delenv("CLAUDECODE", raising=False)
        nid, claims = asyncio.run(_seed_narrative())
        monkeypatch.setattr(get_config().wiki, "layer_b_enabled", False)
        client = _stub_model(f"# A day.\n\nWoke up.[^p-{claims[0][:8]}]\n")
        set_client(client)
        try:
            token = _mint(role, "reader:web-ui" if role == "reader" else AGENT)
            before = asyncio.run(_cache_rows())
            r = engine.get(f"/particles/{nid}/narrative/synthesis", headers=_bearer(token))
            after = asyncio.run(_cache_rows())
        finally:
            set_client(None)
        assert r.status_code == 200, r.text
        assert client.messages.create.call_count >= 1
        assert (after - before == 1) is persists, (role, before, after)

    def test_a_reader_key_never_asks_the_synthesis_to_persist(
        self,
        engine: TestClient,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import particles.operations.narrative_synthesis as ns

        seen: list[bool] = []

        async def _spy(*args: object, **kwargs: object) -> object:
            from fastapi import HTTPException

            seen.append(bool(kwargs.get("persist_cache")))
            raise HTTPException(status_code=404, detail="stop here")

        monkeypatch.setattr(ns, "synthesize_narrative", _spy)
        token = _mint("reader", "viewer")
        engine.get("/particles/p-0000/narrative/synthesis", headers=_bearer(token))
        assert seen == [False]


def _exposure() -> dict[str, Any]:
    return {
        "session_id": "s",
        "recorded_at": "2026-10-26T00:00:00+00:00",
        "source": "startup",
        "action": "full",
        "project_key": "a",
        "observer_scope_applied": True,
        "beliefs": [],
    }


class TestBeliefWriteRefusalTakesNoLock:
    """G36: the write-enable 403 comes before the store's writer lock."""

    @pytest.mark.parametrize(
        ("path", "body"),
        [
            ("/session-exposures", _exposure()),
            ("/particles/link", {"particle_a": "p-a", "particle_b": "p-b"}),
            ("/particles/unlink", {"particle_a": "p-a", "particle_b": "p-b"}),
            ("/particles/tag", {"particle_id": "p-a", "tags": ["x"]}),
            ("/particles/untag", {"particle_id": "p-a", "tags": ["x"]}),
            ("/particles/deposit", {"text": "A note.", "title": "t"}),
        ],
    )
    def test_a_store_without_belief_writes_refuses_before_the_writer_lock(
        self,
        engine: TestClient,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
        path: str,
        body: dict[str, Any],
    ) -> None:
        import particles.db as db
        from particles.config import get_config

        taken: list[str] = []
        real = db.write_lock

        @asynccontextmanager
        async def _recording(store: Any = db.DEFAULT_STORE) -> AsyncIterator[None]:
            taken.append(str(store))
            async with real(store):
                yield

        monkeypatch.setattr(db, "write_lock", _recording)
        get_config().mcp.write.enabled_stores = []
        token = _mint("agent", AGENT)
        resp = engine.post(path, json=body, headers=_bearer(token))
        assert resp.status_code == 403, resp.text
        assert taken == []


class TestEventLogKeyIds:
    """G47: a reader or agent key reads no other key's id from the event log."""

    def _dismiss_as_operator(self, engine: TestClient) -> tuple[str, str]:  # noqa: F811
        from particles.config import get_config

        token, record = _keys.mint(Role.OPERATOR, get_config().curation.operator_identity)
        done = engine.post(
            "/corpus/links/dismiss", json={"url": "https://example.org/x"}, headers=_bearer(token)
        )
        assert done.status_code == 200, done.text
        return token, record.key_id

    @pytest.mark.parametrize("role", ["reader", "agent"])
    def test_a_non_operator_key_reads_no_other_keys_id_from_the_event_log(
        self,
        engine: TestClient,  # noqa: F811
        role: str,
    ) -> None:
        _, op_key = self._dismiss_as_operator(engine)
        token = _mint(role, "viewer" if role == "reader" else AGENT)
        events = engine.get("/events", headers=_bearer(token))
        assert events.status_code == 200
        assert events.json(), "the operator's event is still listed"
        assert op_key not in events.text
        for event in events.json():
            one = engine.get(f"/events/{event['event_id']}", headers=_bearer(token))
            assert one.status_code == 200 and op_key not in one.text
        by_key = engine.get("/events", params={"key": op_key}, headers=_bearer(token))
        assert by_key.status_code == 403

    def test_an_operator_key_still_reads_every_key_id(
        self,
        engine: TestClient,  # noqa: F811
    ) -> None:
        token, op_key = self._dismiss_as_operator(engine)
        events = engine.get("/events", params={"key": op_key}, headers=_bearer(token))
        assert events.status_code == 200
        assert events.json() and op_key in events.text
