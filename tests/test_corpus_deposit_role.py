# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811 (the engine and keys_file fixtures are imported, then named as parameters)

"""An agent principal's route stamps its corpus entries.

Extracting a stamped entry runs its subject resolutions as the depositing
agent's, so a Subject the extraction mints for an unknown name is provisional
to that agent. The stamp is set from the authenticated caller at deposit, never
from a request body, and a harvester's deposit is never stamped.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.observer_scope import project_tag
from particles.core.provisional import ResolutionKind
from particles.core.schema import SourceType
from particles.corpus.deposit import deposit_text
from particles.corpus.deposit_role import AGENT_ROLE, agent_deposit, agent_depositor
from particles.corpus.store import CorpusEntryRow
from tests.test_deposit_confinement import HOOK_ROOTS
from tests.test_engine_keys import (  # noqa: F401
    HOOK,
    _bearer,
    _mint,
    engine,
    keys_file,
)


def _role(entry_id: str) -> str | None:
    from particles.db import get_engine, session_scope

    async def _go() -> str | None:
        try:
            async with session_scope() as s:
                row = await s.get(CorpusEntryRow, entry_id)
                assert row is not None
                return row.deposit_role
        finally:
            await get_engine().dispose()

    return asyncio.run(_go())


# --- in process ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_deposit_inside_the_agent_scope_is_stamped(db_session: AsyncSession) -> None:
    with agent_deposit():
        stamped, _ = await deposit_text(
            db_session, "An agent's note.", "agent-p", SourceType.CONVERSATION
        )
    plain, _ = await deposit_text(db_session, "An operator's note.", "operator", "LOCAL_MARKDOWN")

    assert await agent_depositor(db_session, stamped) == "agent-p"
    assert await agent_depositor(db_session, plain) is None
    row = await db_session.get(CorpusEntryRow, stamped)
    assert row is not None and row.deposit_role == AGENT_ROLE


@pytest.mark.asyncio
async def test_an_agent_snapshot_on_an_unstamped_entry_stamps_it(
    db_session: AsyncSession,
) -> None:
    from particles.core.schema import Mutability
    from particles.corpus.deposit import deposit_text_versioned

    uri = "https://example.org/notes"
    entry_id, _, _ = await deposit_text_versioned(
        db_session,
        text="First version.",
        uri_r=uri,
        source_type="WEB_PAGE",
        mutability=Mutability.MUTABLE,
        tags=[],
        deposited_by="agent-p",
        content_published_at=None,
    )
    assert await agent_depositor(db_session, entry_id) is None
    with agent_deposit():
        again, _, _ = await deposit_text_versioned(
            db_session,
            text="Second version.",
            uri_r=uri,
            source_type="WEB_PAGE",
            mutability=Mutability.MUTABLE,
            tags=[],
            deposited_by="agent-p",
            content_published_at=None,
        )
    assert again == entry_id
    assert await agent_depositor(db_session, entry_id) == "agent-p"


@pytest.mark.asyncio
async def test_the_agent_write_paths_stamp_their_deposits(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.config import get_config
    from particles.operations.agent_write import deposit_conversation_text

    get_config().mcp.write.asserter_identity = "mcp:agent"
    entry_id, _ = await deposit_conversation_text(db_session, text="Agent material.")
    assert await agent_depositor(db_session, entry_id) == "mcp:agent"


@pytest.mark.asyncio
async def test_extraction_of_a_stamped_entry_is_its_depositors_minting_resolution(
    db_session: AsyncSession,
) -> None:
    from particles.corpus.store import get_entry
    from particles.ingest.pipeline import _entry_resolution

    with agent_deposit():
        stamped, _ = await deposit_text(db_session, "Agent text.", "agent-p", "LOCAL_MARKDOWN")
    plain, _ = await deposit_text(db_session, "Operator text.", "operator", "LOCAL_MARKDOWN")

    stamped_entry = await get_entry(db_session, stamped)
    plain_entry = await get_entry(db_session, plain)
    assert stamped_entry is not None and plain_entry is not None
    assert await _entry_resolution(db_session, stamped_entry) == {
        "principal": "agent-p",
        "kind": ResolutionKind.MINTING,
    }
    assert await _entry_resolution(db_session, plain_entry) == {
        "principal": None,
        "kind": ResolutionKind.NON_MINTING,
    }


# --- over HTTP ------------------------------------------------------------------


def test_an_agent_keys_deposit_is_stamped_and_an_operators_is_not(engine: TestClient) -> None:
    token = _mint("agent", "agent:ci-bot")
    agents = engine.post(
        "/corpus/deposit/text", json={"text": "Agent material."}, headers=_bearer(token)
    )
    assert agents.status_code == 200, agents.text
    operators = engine.post("/corpus/deposit/text", json={"text": "Operator material."})
    assert operators.status_code == 200, operators.text

    assert _role(agents.json()["entry_id"]) == AGENT_ROLE
    assert _role(operators.json()["entry_id"]) is None


def test_a_harvesters_deposit_is_not_stamped(engine: TestClient) -> None:
    token = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)
    body = {
        "text": "A session.",
        "uri_r": "claude-code://session/stamp-session",
        "mutability": "APPEND_ONLY",
        "tags": ["claude-code", project_tag("alpha")],
    }
    resp = engine.post("/corpus/deposit/text", json=body, headers=_bearer(token))
    assert resp.status_code == 200, resp.text
    assert _role(resp.json()["entry_id"]) is None
