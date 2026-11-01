# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The bound-project guard on the agent HTTP verbs (security review 2026-10-24-2, P27).

The engine holds no project binding: the agent routes take ``project_key`` from
the request, sent by the project-bound MCP server. Every project guard ran only
when the key was present, so leaving it out skipped them all (K3). On a
rescoped store every agent route now refuses a request that names no project,
with or without a bearer; before the rescope an omitted key still writes.
Adopted from the review's proof tests, and the operator mirrors' N5 check.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests._engine_client import keyed_engine_client

_KEY = "test-bearer-key-for-binding"
_REFUSAL = "must name the project"


@pytest.fixture
def engine(cli_db: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """A dev-key engine (no PARTICLES_API_KEY) with belief writes on the default store."""
    import particles.ingest.subject_resolver as sr
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.db import DEFAULT_STORE

    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield keyed_engine_client(monkeypatch)
    finally:
        ep.set_embedding_model(original)


def _rescope() -> None:
    async def _go() -> None:
        from particles.db import get_engine, session_scope
        from tests._observer_scope import rescoped

        async with session_scope(write=True) as s:
            await rescoped(s)
            await s.commit()
        await get_engine().dispose()

    asyncio.run(_go())


def _assert(engine: TestClient, content: str, **extra: Any) -> Any:
    return engine.post(
        "/particles/assert",
        json={
            "content": content,
            "subject_names": ["project b"],
            "confidence": 0.8,
            "source_excerpt": content,
            **extra,
        },
    )


def _project_b_belief(engine: TestClient, content: str = "Project B ships on Fridays.") -> str:
    resp = _assert(engine, content, project_key="proj-b")
    assert resp.status_code == 200, resp.text
    return str(resp.json()["asserted_particle_id"])


def _requests(a: str, b: str) -> dict[str, tuple[str, dict[str, Any]]]:
    """One request per agent verb, none naming a project."""
    claim = {
        "content": "Project B ships on Mondays.",
        "subject_names": ["project b"],
        "confidence": 0.8,
        "source_excerpt": "B ships on Mondays",
    }
    return {
        "assert": ("/particles/assert", claim),
        "supersede": ("/particles/supersede", {**claim, "supersedes_id": a}),
        "retract": ("/particles/retract", {"particle_id": a, "reason": "cleanup"}),
        "link": ("/particles/link", {"particle_a": a, "particle_b": b}),
        "unlink": ("/particles/unlink", {"particle_a": a, "particle_b": b}),
        "tag": ("/particles/tag", {"particle_id": a, "tags": ["x"]}),
        "untag": ("/particles/untag", {"particle_id": a, "tags": ["x"]}),
        "deposit": ("/particles/deposit", {"text": "worth archiving"}),
    }


_VERBS = ["assert", "supersede", "retract", "link", "unlink", "tag", "untag", "deposit"]


@pytest.mark.parametrize("bearer", [False, True], ids=["dev-key", "bearer"])
@pytest.mark.parametrize("verb", _VERBS)
def test_an_agent_verb_naming_no_project_is_refused_on_a_rescoped_store(
    engine: TestClient, monkeypatch: pytest.MonkeyPatch, verb: str, bearer: bool
) -> None:
    """P27: omitting ``project_key`` no longer skips the bound-project guard."""
    a = _project_b_belief(engine)
    b = _project_b_belief(engine, "Project B has two maintainers.")
    _rescope()
    headers: dict[str, str] = {}
    if bearer:
        monkeypatch.setenv("PARTICLES_API_KEY", _KEY)
        headers = {"Authorization": f"Bearer {_KEY}"}
    path, body = _requests(a, b)[verb]
    resp = engine.post(path, json=body, headers=headers)
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert _REFUSAL in resp.json()["detail"]


def test_the_bound_project_guard_is_not_skipped_by_omitting_the_project_key(
    engine: TestClient,
) -> None:
    """K3 proof: a request with no binding cannot retract a belief only project B sees."""
    pid = _project_b_belief(engine)
    _rescope()
    resp = engine.post("/particles/retract", json={"particle_id": pid, "reason": "cleanup"})
    assert resp.status_code != 200, "retracted a project-B-only belief by omitting project_key"


def test_control_a_retract_naming_another_project_is_refused(engine: TestClient) -> None:
    pid = _project_b_belief(engine)
    _rescope()
    resp = engine.post(
        "/particles/retract",
        json={"particle_id": pid, "reason": "cleanup", "project_key": "proj-a"},
    )
    assert resp.status_code == 400
    assert "not in view for project 'proj-a'" in resp.json()["detail"]


def test_control_the_beliefs_own_project_can_still_write(engine: TestClient) -> None:
    pid = _project_b_belief(engine)
    _rescope()
    tagged = engine.post(
        "/particles/tag", json={"particle_id": pid, "tags": ["x"], "project_key": "proj-b"}
    )
    assert tagged.status_code == 200, tagged.text
    retracted = engine.post(
        "/particles/retract",
        json={"particle_id": pid, "reason": "cleanup", "project_key": "proj-b"},
    )
    assert retracted.status_code == 200, retracted.text


def test_control_before_the_rescope_an_omitted_key_still_writes(engine: TestClient) -> None:
    """No project guard engages on a store that was never rescoped, so nothing changes there."""
    resp = _assert(engine, "Unscoped stores are unaffected.")
    assert resp.status_code == 200, resp.text
    pid = resp.json()["asserted_particle_id"]
    retracted = engine.post("/particles/retract", json={"particle_id": pid, "reason": "cleanup"})
    assert retracted.status_code == 200, retracted.text


def test_n5_operator_tag_and_link_mirrors_refuse_the_dev_key_skip(
    engine: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N5 (fixed): the four operator mirrors refuse a keyless loopback caller."""
    pid = _project_b_belief(engine)
    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    tagged = engine.post(f"/particles/{pid}/tags", json={"tags": ["x"]})
    untagged = engine.request("DELETE", f"/particles/{pid}/tags", json={"tags": ["x"]})
    linked = engine.post("/links", json={"particle_a": pid, "particle_b": pid})
    unlinked = engine.request("DELETE", "/links", json={"particle_a": pid, "particle_b": pid})
    for resp in (tagged, untagged, linked, unlinked):
        assert resp.status_code in (401, 403, 503), resp.status_code


# ---------------------------------------------------------------------------
# An empty key names no project (security review 2026-10-25-1, Q54)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
@pytest.mark.parametrize("verb", _VERBS)
def test_a_blank_project_key_is_refused_like_an_omitted_one(
    engine: TestClient, verb: str, blank: str
) -> None:
    """Q54: on a rescoped store a ``project_key`` naming no project is refused on every verb."""
    a = _project_b_belief(engine)
    b = _project_b_belief(engine, "Project B has two maintainers.")
    _rescope()
    path, body = _requests(a, b)[verb]
    resp = engine.post(path, json={**body, "project_key": blank})
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert _REFUSAL in resp.json()["detail"]


@pytest.mark.parametrize("bearer", [False, True], ids=["dev-key", "bearer"])
def test_p27_a_rescoped_store_refuses_an_agent_call_whose_project_key_is_empty(
    engine: TestClient, monkeypatch: pytest.MonkeyPatch, bearer: bool
) -> None:
    """Q54 proof: an empty key does not land an unattributed agent belief past the check."""
    _rescope()
    headers: dict[str, str] = {}
    if bearer:
        monkeypatch.setenv("PARTICLES_API_KEY", _KEY)
        headers = {"Authorization": f"Bearer {_KEY}"}
    claim = "An empty key names no project."
    resp = engine.post(
        "/particles/assert",
        json={
            "content": claim,
            "subject_names": ["thing"],
            "confidence": 0.8,
            "source_excerpt": claim,
            "project_key": "",
        },
        headers=headers,
    )
    assert resp.status_code == 400, (resp.status_code, resp.text)


def test_control_before_the_rescope_an_empty_key_writes_as_an_omitted_one(
    engine: TestClient,
) -> None:
    """An empty key on a store never rescoped is the omitted key: the write lands unattributed."""
    resp = _assert(engine, "Blank keys read as no key.", project_key="")
    assert resp.status_code == 200, resp.text
    pid = resp.json()["asserted_particle_id"]

    async def _go() -> Any:
        from particles.corpus.store import get_entry
        from particles.db import get_engine, session_scope
        from particles.store.particle_store import get_particle

        try:
            async with session_scope() as s:
                particle = await get_particle(s, pid)
                assert particle is not None
                return await get_entry(s, particle.provenance[0].corpus_entry_id)
        finally:
            await get_engine().dispose()

    entry = asyncio.run(_go())
    assert entry is not None
    assert "scope:unattributed" in (entry.tags or [])
    assert not [t for t in entry.tags or [] if t.startswith("project:")]


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
async def test_require_project_binding_refuses_a_blank_key(db_session: Any, blank: str) -> None:
    """The operation refuses a blank key itself, whatever reaches it besides the HTTP models."""
    from particles.operations.agent_write import require_project_binding
    from tests._observer_scope import rescoped

    await require_project_binding(db_session, blank)  # no rescope yet: nothing engages
    await rescoped(db_session)
    with pytest.raises(ValueError, match=_REFUSAL):
        await require_project_binding(db_session, blank)
    await require_project_binding(db_session, "proj-b")
