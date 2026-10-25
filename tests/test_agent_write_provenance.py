# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""An agent's SOURCE is always its own deposit (security review 2026-10-24, N4 and N29).

Trust and attribution are read from a belief's provenance entry. An agent that
could cite an operator's entry, by id or by reproducing its bytes as an
excerpt, would rank at the operator's trust instead of the seeded agent rank
and read as the operator's document in the digest.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from particles.config import get_config
from particles.core.schema import SourceType
from particles.corpus.deposit import deposit_text
from particles.corpus.store import get_entry
from particles.db import DEFAULT_STORE, session_scope
from particles.store.particle_store import get_particle

AGENT = "mcp:test-agent"
OPERATOR_TEXT = "The release train leaves every second Tuesday."


@pytest.fixture(autouse=True)
def _writes(monkeypatch: pytest.MonkeyPatch) -> None:
    import particles.ingest.subject_resolver as sr

    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))
    w = get_config().mcp.write
    w.enabled_stores = [DEFAULT_STORE]
    w.asserter_identity = AGENT


async def _operator_entry(db_session: Any, text: str = OPERATOR_TEXT) -> str:
    entry_id, _ = await deposit_text(
        db_session, text, deposited_by="operator", source_type=SourceType.CONVERSATION
    )
    await db_session.commit()
    return entry_id


async def _source_entry(particle_id: str) -> Any:
    async with session_scope(DEFAULT_STORE) as s:
        particle = await get_particle(s, particle_id)
        assert particle is not None
        return await get_entry(s, particle.provenance[0].corpus_entry_id)


@pytest.mark.asyncio
async def test_citing_an_operator_entry_by_id_is_refused(db_session: Any) -> None:
    from particles.mcp.tools.write import particle_assert

    entry_id = await _operator_entry(db_session)
    with pytest.raises(ValueError, match="may cite only its own deposits"):
        await particle_assert("The train is biweekly.", ["train"], 0.8, corpus_entry_id=entry_id)


@pytest.mark.asyncio
async def test_an_agent_may_cite_its_own_deposit(db_session: Any) -> None:
    from particles.mcp.tools.write import deposit_text as mcp_deposit_text
    from particles.mcp.tools.write import particle_assert

    out = await mcp_deposit_text("We settled on a biweekly release train.")
    res = await particle_assert(
        "The train is biweekly.", ["train"], 0.8, corpus_entry_id=out["corpus_entry_id"]
    )
    assert res["verdict"] == "ASSERTED"
    entry = await _source_entry(res["asserted_particle_id"])
    assert entry.entry_id == out["corpus_entry_id"]


@pytest.mark.asyncio
async def test_an_excerpt_equal_to_an_operator_deposit_stays_the_agents(db_session: Any) -> None:
    from particles.mcp.tools.write import particle_assert

    operator_entry = await _operator_entry(db_session)
    res = await particle_assert(
        "The train leaves every second Tuesday.", ["train"], 0.8, source_excerpt=OPERATOR_TEXT
    )
    entry = await _source_entry(res["asserted_particle_id"])
    assert entry.entry_id != operator_entry
    assert entry.deposited_by == AGENT
    assert entry.snapshots[-1].author_id == AGENT
    assert entry.source_type == SourceType.CONVERSATION
    assert "scope:unattributed" in entry.tags


@pytest.mark.asyncio
async def test_an_agents_repeated_excerpt_still_deduplicates(db_session: Any) -> None:
    from particles.mcp.tools.write import deposit_text as mcp_deposit_text

    first = await mcp_deposit_text("A note worth keeping.")
    second = await mcp_deposit_text("A note worth keeping.")
    assert first["corpus_entry_id"] == second["corpus_entry_id"]


@pytest.mark.asyncio
async def test_a_bound_agents_excerpt_never_keys_another_projects_entry(
    db_session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.mcp.tools import write

    monkeypatch.setattr(write, "bound_project", lambda: "proj-b")
    b_entry = (await write.deposit_text("Project B's private note."))["corpus_entry_id"]
    monkeypatch.setattr(write, "bound_project", lambda: "proj-a")
    a_entry = (await write.deposit_text("Project B's private note."))["corpus_entry_id"]

    assert a_entry != b_entry
    async with session_scope(DEFAULT_STORE) as s:
        b = await get_entry(s, b_entry)
        a = await get_entry(s, a_entry)
    assert b is not None and a is not None
    assert b.tags == ["project:proj-b"]
    assert a.tags == ["project:proj-a"]
