"""Tests for particles/operations/narrative_synthesis.py — render path."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ParticleType,
    RelationCreatedBy,
    RelationType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.operations.narrative_synthesis import synthesize_narrative
from particles.store.particle_store import insert_particle
from particles.store.relation_store import create_relation

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _claim(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
    )


def _narrative(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        particle_type=ParticleType.NARRATIVE,
    )


async def _build_narrative(session: AsyncSession) -> tuple[Particle, list[Particle]]:
    """A NARRATIVE over three ordered constituents (c1 → c2 → c3)."""
    narrative = _narrative("A hard day the author got through anyway.")
    c1 = _claim("The author woke up tired.")
    c2 = _claim("The author is not good at Balatro.")
    c3 = _claim("The author felt better by evening.")
    for p in (narrative, c1, c2, c3):
        await insert_particle(session, p)
    for c in (c1, c2, c3):
        await create_relation(
            session, c.id, narrative.id, RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
        )
    await create_relation(
        session, c1.id, c2.id, RelationType.SEQUENCE_IN, RelationCreatedBy.MANUAL_CLI
    )
    await create_relation(
        session, c2.id, c3.id, RelationType.SEQUENCE_IN, RelationCreatedBy.MANUAL_CLI
    )
    await session.flush()
    return narrative, [c1, c2, c3]


@pytest.mark.asyncio
async def test_not_a_narrative_returns_none(db_session: AsyncSession) -> None:
    claim = _claim("Just a claim.")
    await insert_particle(db_session, claim)
    assert await synthesize_narrative(db_session, claim.id) is None


@pytest.mark.asyncio
async def test_missing_particle_returns_none(db_session: AsyncSession) -> None:
    assert await synthesize_narrative(db_session, "does-not-exist") is None


@pytest.mark.asyncio
async def test_empty_narrative_returns_empty_body(db_session: AsyncSession) -> None:
    narrative = _narrative("Lonely label, no constituents.")
    await insert_particle(db_session, narrative)
    await db_session.flush()
    article = await synthesize_narrative(db_session, narrative.id)
    assert article is not None
    assert article.constituents == []
    assert article.body == ""
    assert article.used_synthesis is False


@pytest.mark.asyncio
async def test_without_synthesis_renders_cited_listing(db_session: AsyncSession) -> None:
    """The deterministic, key-free path: a cited structured listing of the
    constituents (with particle references), no LLM call."""
    narrative, claims = await _build_narrative(db_session)
    article = await synthesize_narrative(db_session, narrative.id, without_synthesis=True)
    assert article is not None
    assert article.used_synthesis is False
    assert [c.id for c in article.constituents] == [c.id for c in claims]  # SEQUENCE_IN order
    assert article.body  # non-empty
    # Particle references are present (the ADR's "complete with particle references").
    assert f"p-{claims[0].id[:8]}" in article.body


@pytest.mark.asyncio
async def test_synthesis_happy_path_uses_llm(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a (mocked) LLM and Layer B disabled, the narrative renders as the
    cited prose body the model returned (used_synthesis=True)."""
    import anthropic

    from particles.config import get_config
    from particles.llm import set_client

    narrative, claims = await _build_narrative(db_session)

    # Layer B is a second LLM call; disable it so this test pins the render path
    # without mocking the judge (tests/AGENTS.md § Mocking strategy).
    monkeypatch.setattr(get_config().wiki, "layer_b_enabled", False)

    llm_body = (
        f"# {narrative.content}\n\n"
        f"The author woke up tired.[^p-{claims[0].id[:8]}] "
        f"The author is not good at Balatro.[^p-{claims[1].id[:8]}]\n"
    )
    mock_content = MagicMock()
    mock_content.text = llm_body
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    set_client(mock_client)
    try:
        article = await synthesize_narrative(db_session, narrative.id)
    finally:
        set_client(None)

    assert article is not None
    assert article.used_synthesis is True
    assert f"p-{claims[0].id[:8]}" in article.body
    assert article.narrative.id == narrative.id
    assert len(article.constituents) == 3
