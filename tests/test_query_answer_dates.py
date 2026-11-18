# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The answer step is shown each claim's date.

Shown no dates, the composer sided with the value three older claims repeated
over the one newer claim that replaced it. The pure tier covers which date a
particle carries; the op tier drives the real query operation over an
in-memory store with the Anthropic client mocked at the shared ``set_client``
seam (tests/AGENTS.md § Mocking strategy), so the prompt is inspected exactly
as it would be sent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from particles.config import get_config
from particles.core.schema import (
    AudienceHint,
    CalibrationSource,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    Status,
    UncertaintyNature,
    ValidKind,
)
from particles.operations.query.respond import _DATED_RULE, _generate_response, answer_dates

_OLD = datetime(2026, 10, 10, tzinfo=UTC)
_NEW = datetime(2026, 10, 24, tzinfo=UTC)


def _particle(content: str, valid_from: datetime | None = None) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id="s")
        ],
        valid_from=valid_from,
        valid_kind=ValidKind.STATE if valid_from is not None else None,
    )


class TestAnswerDates:
    def test_valid_from_wins_over_the_source_date(self) -> None:
        p = _particle("The quota is 40.", _NEW)
        assert answer_dates([p], {p.id: _OLD}) == {p.id: _NEW}

    def test_the_source_date_stands_in_for_a_missing_valid_from(self) -> None:
        p = _particle("The quota is 40.")
        assert answer_dates([p], {p.id: _OLD}) == {p.id: _OLD}

    def test_an_undated_particle_carries_no_date(self) -> None:
        p = _particle("The quota is 40.")
        assert answer_dates([p], {p.id: None}) == {}


async def _sent(particles: list[Particle], dates: dict[str, datetime] | None) -> tuple[str, str]:
    call = AsyncMock(return_value="An answer.")
    # A fixed fence nonce, so two prompts compare byte for byte.
    with patch("particles.llm.complete", call), patch("particles.llm.make_nonce", lambda: "n0"):
        await _generate_response(
            "What is the quota?",
            particles,
            [0.9] * len(particles),
            AudienceHint.GENERAL,
            "",
            "",
            dates=dates,
        )
    assert call.await_args is not None
    return call.await_args.kwargs["system"], call.await_args.args[1]


class TestPrompt:
    async def test_a_dated_line_carries_its_day_and_the_rule_joins(self) -> None:
        old = [_particle("The quota is 40.", _OLD) for _ in range(3)]
        new = _particle("The quota is 60.", _NEW)
        system, user = await _sent([*old, new], answer_dates([*old, new], {}))
        assert "The quota is 60. <FALSIFIABLE> (dated 2026-10-24)" in user
        assert user.count("(dated 2026-10-10)") == 3
        assert _DATED_RULE in system

    async def test_an_undated_prompt_is_unchanged(self) -> None:
        particles = [_particle("The quota is 40.")]
        system_none, user_none = await _sent(particles, None)
        system_empty, user_empty = await _sent(particles, {})
        assert (system_none, user_none) == (system_empty, user_empty)
        assert _DATED_RULE not in system_none and "(dated" not in user_none


class _Encoder:
    """Every text embeds to the same unit vector: every hit clears the floor."""

    def encode(self, texts: list[str], **_: Any) -> Any:
        return [np.ones(4, dtype=np.float32) / 2.0 for _ in texts]


def _client() -> MagicMock:
    import anthropic

    content = MagicMock()
    content.text = "The quota is 60."
    content.type = "text"
    reply = MagicMock()
    reply.content = [content]
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(return_value=reply)
    return client


async def _ask_user_turn(session: Any) -> str:
    from particles import embeddings as ep
    from particles.llm import set_client
    from particles.operations.query import query

    client = _client()
    original = ep._embedding_model
    ep.set_embedding_model(_Encoder())  # type: ignore[arg-type]
    set_client(client)
    try:
        await query(session, QueryRequest(question="What is the quota?"))
    finally:
        set_client(None)
        ep.set_embedding_model(original)
    user = client.messages.create.call_args.kwargs["messages"][0]["content"]
    return user if isinstance(user, str) else " ".join(b["text"] for b in user)


class TestQueryOp:
    async def test_the_query_dates_each_claim(self, db_session: Any) -> None:
        from particles.store.particle_store import insert_particle

        await insert_particle(db_session, _particle("The quota is 60.", _NEW), [0.5] * 4)
        await db_session.commit()
        assert "(dated 2026-10-24)" in await _ask_user_turn(db_session)

    async def test_the_setting_turns_it_off(self, db_session: Any) -> None:
        from particles.store.particle_store import insert_particle

        await insert_particle(db_session, _particle("The quota is 60.", _NEW), [0.5] * 4)
        await db_session.commit()
        get_config().query.answer_dates = False
        assert "(dated" not in await _ask_user_turn(db_session)
