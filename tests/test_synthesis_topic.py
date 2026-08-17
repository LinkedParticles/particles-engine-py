"""Tests for the section-topic synthesis generalization.

``render_article`` was subject-scoped; its entry point was generalized to
any :class:`SynthesisTopic`. These tests pin that a non-Subject topic
(:class:`SectionTopic`) drives the engine — both the deterministic listing and
the LLM path — and that ``Subject`` still satisfies the protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from particles.core.schema import Confidence, Particle, Subject, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.render.article_synthesis import (
    SectionTopic,
    SynthesisTopic,
    compute_input_hash,
    render_article,
)

if TYPE_CHECKING:
    pass


def _claim(content: str, conf: float = 0.9) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=conf, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
    )


def test_subject_satisfies_synthesis_topic() -> None:
    """The thin shim is structural: a real Subject *is* a SynthesisTopic."""
    subject = Subject(canonical_name="Particles standard", asserted_by="test")
    topic: SynthesisTopic = subject  # must type-check and hold at runtime
    assert topic.id == subject.id
    assert topic.canonical_name == "Particles standard"


def test_section_topic_is_a_synthesis_topic() -> None:
    topic: SynthesisTopic = SectionTopic(id="readme:overview", canonical_name="Overview")
    assert topic.id == "readme:overview"
    assert topic.description is None


@pytest.mark.asyncio
async def test_render_article_with_section_topic_deterministic() -> None:
    """A SectionTopic drives the key-free structured-listing path (no cache, no LLM)."""
    particles = [_claim("Particles is a Python SDK."), _claim("It tracks provenance.")]
    eff = {p.id: p.confidence.value for p in particles}
    topic = SectionTopic(id="readme:what", canonical_name="What is Particles")

    body, used = await render_article(
        subject=topic,
        particles=particles,
        eff=eff,
        input_hash=compute_input_hash(particles),
        corpus_uris={},
        max_tokens=512,
        layer_b_enabled=False,
        session=None,
        without_synthesis=True,
    )
    assert used is False
    # The topic title is the article H1; the claims are cited by short-id.
    assert "# What is Particles" in body
    assert f"p-{particles[0].id[:8]}" in body


@pytest.mark.asyncio
async def test_render_article_with_section_topic_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SectionTopic also drives the LLM path; Layer A accepts a cited body."""
    import anthropic

    from particles.config import get_config
    from particles.llm import set_client

    monkeypatch.setattr(get_config().wiki, "layer_b_enabled", False)
    particles = [_claim("Particles is a Python SDK.")]
    eff = {p.id: p.confidence.value for p in particles}
    topic = SectionTopic(id="readme:what", canonical_name="What is Particles")

    llm_body = f"# What is Particles\n\nParticles is a Python SDK.[^p-{particles[0].id[:8]}]\n"
    mock_content = MagicMock()
    mock_content.text = llm_body
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    set_client(mock_client)
    try:
        body, used = await render_article(
            subject=topic,
            particles=particles,
            eff=eff,
            input_hash=compute_input_hash(particles),
            corpus_uris={},
            max_tokens=512,
            layer_b_enabled=False,
            session=None,
        )
    finally:
        set_client(None)

    assert used is True
    assert f"p-{particles[0].id[:8]}" in body
