# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811 (the engine and keys_file fixtures are imported, then named as parameters)

"""``POST /subjects/{id}/accept`` and the ``provisional`` key on subject reads (§5)."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi.testclient import TestClient

from tests.test_engine_keys import (  # noqa: F401
    _bearer,
    _mint,
    engine,
    keys_file,
)


def _mint_provisional_subject() -> str:
    from particles.core.schema import Subject
    from particles.db import get_engine, session_scope
    from particles.store.subject_provisional import mark_provisional
    from particles.store.subject_store import insert_subject

    async def _go() -> str:
        try:
            async with session_scope() as s:
                subject = Subject(canonical_name="the server", asserted_by="agent:ci-bot")
                await insert_subject(s, subject)
                await mark_provisional(s, subject.id, "agent:ci-bot")
                await s.commit()
                return subject.id
        finally:
            await get_engine().dispose()

    return asyncio.run(_go())


def test_subject_reads_carry_the_mark_and_accept_removes_it(engine: TestClient) -> None:
    subject_id = _mint_provisional_subject()

    shown = engine.get(f"/subjects/{subject_id}")
    assert shown.status_code == 200, shown.text
    assert shown.json()["provisional"]["principal"] == "agent:ci-bot"
    listed = engine.get("/subjects").json()
    assert [s["provisional"]["principal"] for s in listed if s["id"] == subject_id] == [
        "agent:ci-bot"
    ]

    accepted = engine.post(f"/subjects/{subject_id}/accept")
    assert accepted.status_code == 200, accepted.text
    body: dict[str, Any] = accepted.json()
    assert body["accepted"] is True and "provisional" not in body["subject"]
    assert "provisional" not in engine.get(f"/subjects/{subject_id}").json()
    assert engine.post(f"/subjects/{subject_id}/accept").json()["accepted"] is False
    assert engine.post("/subjects/no-such-subject/accept").status_code == 404


def test_an_agent_key_may_not_accept(engine: TestClient) -> None:
    subject_id = _mint_provisional_subject()
    token = _mint("agent", "agent:ci-bot")
    resp = engine.post(f"/subjects/{subject_id}/accept", headers=_bearer(token))
    assert resp.status_code == 403


def test_an_ambiguous_prefix_is_refused_not_reported_missing(engine: TestClient) -> None:
    from particles.core.schema import Subject
    from particles.db import get_engine, session_scope
    from particles.store.subject_store import insert_subject

    async def _go() -> None:
        try:
            async with session_scope() as s:
                for suffix in ("1", "2"):
                    await insert_subject(
                        s, Subject(id=f"abcd000{suffix}", canonical_name=suffix, asserted_by="x")
                    )
                await s.commit()
        finally:
            await get_engine().dispose()

    asyncio.run(_go())
    resp = engine.post("/subjects/abcd/accept")
    assert resp.status_code == 400
    assert "Ambiguous" in resp.json()["detail"]
