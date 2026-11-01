# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Agent provenance over the HTTP surface (security review 2026-10-24-2, P28, P29).

The agent path cites only its own deposits (N4) and deduplicates only onto its
own entries (N29). These pin the residuals the third delta review found: a
``deposited_by`` label any HTTP depositor supplies is not enough to cite an
entry (P29), and the remote MCP ``deposit_text`` never reuses or rescopes
another principal's entry (P28). Adopted from the review's proof tests.

In-process only (TestClient / ASGITransport against a scratch store), with the
suite's autouse fixtures refusing live LLM and network calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests._engine_client import TEST_ENGINE_KEY, keyed_engine_client

AGENT = "mcp:claude-code"  # the compiled-in mcp.write.asserter_identity


@pytest.fixture
def engine(cli_db: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """A keyed engine with belief writes on the default store."""
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


def _source_rank(particle_id: str) -> tuple[float | None, Any]:
    """The asserted source-trust rank the read path applies, and the source entry."""

    async def _go() -> tuple[float | None, Any]:
        from particles.corpus.store import get_entry
        from particles.db import get_engine, session_scope
        from particles.operations.query.source_trust import load_source_trust_ranks
        from particles.store.particle_store import get_particle

        try:
            async with session_scope() as s:
                particle = await get_particle(s, particle_id)
                assert particle is not None
                ranks = await load_source_trust_ranks(s, [particle])
                entry = await get_entry(s, particle.provenance[0].corpus_entry_id)
                return ranks.get(particle_id), entry
        finally:
            await get_engine().dispose()

    return asyncio.run(_go())


def test_an_agent_belief_citing_an_entry_labelled_with_the_agent_identity_keeps_the_agent_trust(
    engine: TestClient,
) -> None:
    """P29: the agent-only citation rule does not key on a caller-supplied ``deposited_by``.

    Safe property: an agent assertion over HTTP that cites a corpus entry
    ranks at ``mcp.write.agent_trust_rank`` (or is refused), whatever source
    type and author the depositing HTTP caller chose for that entry.
    """
    from particles.config import get_config

    deposit = engine.post(
        "/corpus/deposit/text",
        json={
            "text": "The release train leaves every second Tuesday.",
            "source_type": "WEB_PAGE",
            "deposited_by": AGENT,
        },
    )
    assert deposit.status_code == 200, deposit.text
    asserted = engine.post(
        "/particles/assert",
        json={
            "content": "The release train is biweekly.",
            "subject_names": ["release train"],
            "confidence": 0.9,
            "corpus_entry_id": deposit.json()["entry_id"],
        },
    )
    if asserted.status_code == 400:
        return  # refused: the safe outcome
    assert asserted.status_code == 200, asserted.text
    rank, _entry = _source_rank(asserted.json()["asserted_particle_id"])
    assert rank == get_config().mcp.write.agent_trust_rank, (
        f"agent belief ranks at source trust {rank!r} (absent = neutral 1.0), "
        "not the seeded agent rank"
    )


def test_citing_an_operator_entry_over_http_is_refused(engine: TestClient) -> None:
    """N4 as aimed (fixed): an entry another principal deposited cannot be cited by id."""
    deposit = engine.post(
        "/corpus/deposit/text",
        json={"text": "A harvested note.", "deposited_by": "another-principal"},
    )
    assert deposit.status_code == 200
    asserted = engine.post(
        "/particles/assert",
        json={
            "content": "An operator's note holds.",
            "subject_names": ["note"],
            "confidence": 0.9,
            "corpus_entry_id": deposit.json()["entry_id"],
        },
    )
    assert asserted.status_code == 400
    assert "may cite only its own deposits" in asserted.json()["detail"]


def test_an_http_agent_excerpt_equal_to_an_operator_deposit_stays_the_agents(
    engine: TestClient,
) -> None:
    """N29 as aimed (fixed for the assert route): the excerpt never lands on that entry."""
    text = "The deploy key rotates monthly."
    operator = engine.post(
        "/corpus/deposit/text", json={"text": text, "deposited_by": "another-principal"}
    ).json()["entry_id"]
    asserted = engine.post(
        "/particles/assert",
        json={
            "content": "The deploy key rotates monthly.",
            "subject_names": ["deploy key"],
            "confidence": 0.8,
            "source_excerpt": text,
        },
    )
    assert asserted.status_code == 200, asserted.text
    _rank, entry = _source_rank(asserted.json()["asserted_particle_id"])
    assert entry.entry_id != operator
    assert entry.deposited_by == AGENT


@pytest.fixture
def http_backend(cli_db: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Any, None, None]:
    """``HttpBackend`` wired in-process to the app, as tests/test_api_client.py does."""
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://engine.test")
    # A keyless engine serves reads only: key the engine and send it.
    monkeypatch.setenv("PARTICLES_API_KEY", TEST_ENGINE_KEY)
    monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", TEST_ENGINE_KEY)
    from particles.api.app import app as fastapi_app
    from particles.api.client.http import HttpBackend
    from particles.config import get_config, reset_config
    from particles.db import DEFAULT_STORE

    reset_config()
    get_config().api.allowed_hosts = ["engine.test"]
    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    real_async_client = httpx.AsyncClient

    def _asgi_client(
        *, base_url: str, timeout: float, headers: dict[str, str], purpose: str = ""
    ) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.ASGITransport(app=fastapi_app), base_url=base_url, headers=headers
        )

    monkeypatch.setattr("particles.api.client.http.egress_client", _asgi_client)
    yield HttpBackend()


@pytest.mark.asyncio
async def test_mcp_deposit_text_through_the_http_backend_never_keys_another_projects_entry(
    http_backend: Any,
) -> None:
    """P28: the remote MCP ``deposit_text`` path deduplicates onto the agent's own entries only.

    Safe property: a project-bound agent's ``deposit_text`` through a remote
    engine gets an entry of its own and adds no project key to an entry another
    project's harvest deposited.
    """
    from particles.corpus.deposit import deposit_text
    from particles.corpus.store import get_entry
    from particles.db import get_engine, session_scope

    text = "Project B's private design note."
    async with session_scope(write=True) as s:
        b_entry, _ = await deposit_text(
            s, text, deposited_by="claude-code-hook", tags=["project:proj-b"]
        )
        await s.commit()

    entry_id, _snap = await http_backend.deposit_text(
        text=text, tags=None, store="default", project_key="proj-a"
    )

    async with session_scope() as s:
        b = await get_entry(s, b_entry)
        mine = await get_entry(s, entry_id)
    await get_engine().dispose()
    assert b is not None and mine is not None
    assert b.tags == ["project:proj-b"], f"project B's entry now carries {b.tags}"
    assert mine.deposited_by == AGENT, f"the agent's deposit landed on {mine.deposited_by}'s entry"


def test_control_an_agent_excerpt_belief_ranks_at_the_agent_trust(engine: TestClient) -> None:
    """Control: the seeded agent rank binds on the agent's own excerpt deposit."""
    from particles.config import get_config

    asserted = engine.post(
        "/particles/assert",
        json={
            "content": "The release train is biweekly.",
            "subject_names": ["release train"],
            "confidence": 0.9,
            "source_excerpt": "we agreed the train is biweekly",
        },
    )
    assert asserted.status_code == 200, asserted.text
    rank, _entry = _source_rank(asserted.json()["asserted_particle_id"])
    assert rank == get_config().mcp.write.agent_trust_rank


def test_an_http_labelled_conversation_entry_cites_at_the_agent_trust(engine: TestClient) -> None:
    """P29: an entry labelled exactly as an agent deposit reads exactly as one.

    The HTTP depositor names the agent as depositor and author and the source
    type as CONVERSATION. Citing it is admitted, and the belief ranks at the
    agent trust with the agent's attribution, no better than its own excerpt.
    """
    from particles.config import get_config

    deposit = engine.post(
        "/corpus/deposit/text",
        json={
            "text": "Standups are at nine.",
            "source_type": "CONVERSATION",
            "deposited_by": AGENT,
            "author_id": AGENT,
            "tags": ["scope:unattributed"],
        },
    )
    assert deposit.status_code == 200, deposit.text
    asserted = engine.post(
        "/particles/assert",
        json={
            "content": "Standups are at nine.",
            "subject_names": ["standup"],
            "confidence": 0.9,
            "corpus_entry_id": deposit.json()["entry_id"],
        },
    )
    if asserted.status_code == 400:
        return  # refused (an HTTP deposit route that stamps its own depositor): also safe
    assert asserted.status_code == 200, asserted.text
    rank, _entry = _source_rank(asserted.json()["asserted_particle_id"])
    assert rank == get_config().mcp.write.agent_trust_rank


@pytest.mark.parametrize(
    "override",
    [
        {"author_id": "operator"},
        {"source_type": "WEB_PAGE"},
        {"tags": []},
    ],
    ids=["another-author", "another-source-type", "global-scope"],
)
def test_an_entry_differing_from_an_agent_deposit_is_refused(
    engine: TestClient, override: dict[str, Any]
) -> None:
    """P29: each of author, source type and scope marking is checked, not just the depositor."""
    body: dict[str, Any] = {
        "text": f"A note for {sorted(override)}.",
        "source_type": "CONVERSATION",
        "deposited_by": AGENT,
        "author_id": AGENT,
        "tags": ["scope:unattributed"],
    }
    body.update(override)
    # The engine is keyed, and its one bearer carries operator authority (P42),
    # so even the deposit naming another author is admitted; the property under
    # test is that the agent's assert then refuses to cite it.
    deposit = engine.post("/corpus/deposit/text", json=body)
    assert deposit.status_code == 200, deposit.text
    entry_id = deposit.json()["entry_id"]
    asserted = engine.post(
        "/particles/assert",
        json={
            "content": "A note holds.",
            "subject_names": ["note"],
            "confidence": 0.9,
            "corpus_entry_id": entry_id,
        },
    )
    assert asserted.status_code == 400, asserted.text
    assert "may cite only its own deposits" in asserted.json()["detail"]


def test_the_agent_deposit_route_attributes_server_side(engine: TestClient) -> None:
    """P28: ``POST /particles/deposit`` stamps depositor, author, type and scope itself."""

    async def _entry(entry_id: str) -> Any:
        from particles.corpus.store import get_entry
        from particles.db import get_engine, session_scope

        try:
            async with session_scope() as s:
                return await get_entry(s, entry_id)
        finally:
            await get_engine().dispose()

    resp = engine.post(
        "/particles/deposit",
        json={"text": "Worth archiving.", "tags": ["project:proj-x", "note"]},
    )
    assert resp.status_code == 200, resp.text
    entry = asyncio.run(_entry(resp.json()["entry_id"]))
    assert entry.deposited_by == AGENT
    assert entry.source_type == "CONVERSATION"
    assert entry.snapshots[-1].author_id == AGENT
    assert sorted(entry.tags) == ["note", "scope:unattributed"]


def test_the_agent_deposit_route_refuses_a_store_not_write_enabled(engine: TestClient) -> None:
    from particles.config import get_config

    get_config().mcp.write.enabled_stores = []
    resp = engine.post("/particles/deposit", json={"text": "x"})
    assert resp.status_code == 403


def test_u14_the_agent_deposit_route_is_rate_limited_like_assert(engine: TestClient) -> None:
    """POST /particles/deposit queues a snapshot the nightly consolidation extracts
    through a model, so it is bounded by api.rate_limit_per_minute like
    /particles/assert (security review 2026-10-25-2, U14 / R25)."""
    from particles.config import get_config

    get_config().api.rate_limit_per_minute = 2
    codes = [
        engine.post("/particles/deposit", json={"text": f"archive item {i}"}).status_code
        for i in range(4)
    ]
    assert codes == [200, 200, 429, 429], codes
