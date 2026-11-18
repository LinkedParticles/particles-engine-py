# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Extracting an entry an agent deposited mints provisionally.

End to end through ``extract_snapshot`` with the model's reply mocked: a stamped
entry's new Subject is provisional to the depositing agent, an unstamped
entry's is confirmed, and an extractor-emitted external reference is not
attached to the agent's Subject, since a ref read from text the agent chose
is no independent reading.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles import embeddings as ep
from particles.core.schema import ExternalRef, Mutability
from particles.corpus.deposit import deposit_text_versioned
from particles.corpus.deposit_role import agent_deposit
from particles.extraction.general import GeneralExtractor
from particles.ingest.pipeline import extract_snapshot
from particles.store.particle_store import get_active_particle_ids_for_snapshot, get_particle
from particles.store.subject_provisional import mark_of
from particles.store.subject_store import get_subject

_NAME = "Quokka Station"
_REF = ExternalRef(namespace="wikidata", id="Q999999", uri=None, confidence=1.0)


def _vector(text: str) -> list[float]:
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest() * 4, dtype=np.uint8)
    vec = raw.astype(np.float32) - 127.5
    return list(vec / np.linalg.norm(vec))


@pytest.fixture
def embeddings() -> Iterator[None]:
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, **_: np.array([_vector(t) for t in texts], dtype=np.float32)
    )
    original = ep._embedding_model
    ep.set_embedding_model(model)
    yield
    ep.set_embedding_model(original)


@pytest.fixture
def extractor_with_a_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """The general extractor, answering one claim and naming an external ref for it."""
    reply = json.dumps(
        [
            {
                "content": f"{_NAME} opened in 2031.",
                "subjects": [_NAME],
                "confidence_value": 0.9,
                "uncertainty_nature": "EPISTEMIC",
            }
        ]
    )

    async def fake_complete(*args: Any, **kwargs: Any) -> tuple[str, str]:
        return reply, "anthropic:test-model"

    monkeypatch.setattr("particles.llm.complete_with_provider_model", fake_complete)
    original = GeneralExtractor.extract

    async def with_ref(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original(self, *args, **kwargs)
        for candidate in result.candidates:
            candidate.external_refs = {_NAME: _REF}
        return result

    monkeypatch.setattr(GeneralExtractor, "extract", with_ref)


async def _extract(session: AsyncSession, *, agent: bool, uri: str) -> str:
    """Deposit (as an agent route when ``agent``), extract, and return the claim's Subject id."""

    async def deposit() -> tuple[str, str, bool]:
        return await deposit_text_versioned(
            session,
            text=f"{_NAME} opened in 2031.",
            uri_r=uri,
            source_type="WEB_PAGE",
            mutability=Mutability.MUTABLE,
            deposited_by="agent-p" if agent else "operator",
        )

    if agent:
        with agent_deposit():
            entry_id, snapshot_id, _ = await deposit()
    else:
        entry_id, snapshot_id, _ = await deposit()
    await session.commit()
    await extract_snapshot(session, entry_id, snapshot_id)
    await session.commit()
    [particle_id] = await get_active_particle_ids_for_snapshot(session, entry_id, snapshot_id)
    particle = await get_particle(session, particle_id)
    assert particle is not None
    [subject_id] = particle.subject_ids
    return subject_id


@pytest.mark.asyncio
async def test_an_agent_deposit_mints_provisional_and_attaches_no_ref(
    db_session: AsyncSession, embeddings: None, extractor_with_a_ref: None
) -> None:
    subject_id = await _extract(db_session, agent=True, uri="https://example.com/agent")

    mark = await mark_of(db_session, subject_id)
    assert mark is not None and mark.principal == "agent-p"
    subject = await get_subject(db_session, subject_id)
    assert subject is not None and subject.external_ids == []


@pytest.mark.asyncio
async def test_an_operator_deposit_mints_confirmed_and_attaches_the_ref(
    db_session: AsyncSession, embeddings: None, extractor_with_a_ref: None
) -> None:
    subject_id = await _extract(db_session, agent=False, uri="https://example.com/operator")

    assert await mark_of(db_session, subject_id) is None
    subject = await get_subject(db_session, subject_id)
    assert subject is not None and [r.id for r in subject.external_ids] == [_REF.id]
