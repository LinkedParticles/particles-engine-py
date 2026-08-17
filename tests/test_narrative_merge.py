"""Tests for particles/ingest/narrative_merge.py — the narrative merge."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from particles.core.schema import ParticleType, UncertaintyNature
from particles.extraction.general import CandidateParticle
from particles.ingest.narrative_merge import collapse_chunk_narratives


def _claim(content: str, narrative_index: int) -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        narrative_index=narrative_index,
    )


def _narrative(label: str) -> CandidateParticle:
    return CandidateParticle(
        content=label,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        particle_type=ParticleType.NARRATIVE,
    )


def _two_chunk_candidates() -> list[CandidateParticle]:
    """Flat candidate list as extract_with_carry_forward returns it for a
    2-chunk journal: each chunk's claims (chunk-local index) then its NARRATIVE."""
    return [
        _claim("a", 0),
        _claim("b", 1),
        _narrative("Label A"),
        _claim("c", 0),
        _claim("d", 1),
        _narrative("Label B"),
    ]


@pytest.mark.asyncio
async def test_single_narrative_is_noop() -> None:
    """A single-pass journal (one NARRATIVE candidate) is returned untouched."""
    candidates = [_claim("a", 0), _claim("b", 1), _narrative("Label A")]
    merged, notes = await collapse_chunk_narratives(candidates)
    assert merged is candidates  # same object — no-op
    assert notes == []


@pytest.mark.asyncio
async def test_no_narrative_is_noop() -> None:
    """A non-journal extraction (zero NARRATIVE candidates) is returned untouched."""
    candidates = [_claim("a", 0)]
    # A bare CLAIM carrying no narrative_index (the general-extractor shape) must
    # also be left alone.
    candidates[0].narrative_index = None
    merged, notes = await collapse_chunk_narratives(candidates)
    assert merged is candidates
    assert notes == []


@pytest.mark.asyncio
async def test_collapses_and_reindexes_globally(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two per-chunk NARRATIVE fragments collapse to one; constituents get a
    single global narrative_index in document order. Synthesis disabled → the
    first chunk's label is kept deterministically (no LLM needed)."""
    from particles.config import get_config

    monkeypatch.setattr(get_config().journal_extractor, "synthesize_merged_narrative", False)

    merged, notes = await collapse_chunk_narratives(_two_chunk_candidates())

    narratives = [c for c in merged if c.particle_type == ParticleType.NARRATIVE]
    claims = [c for c in merged if c.particle_type != ParticleType.NARRATIVE]
    assert len(narratives) == 1
    assert narratives[0].content == "Label A"  # first chunk's label (fallback)
    assert [(c.content, c.narrative_index) for c in claims] == [
        ("a", 0),
        ("b", 1),
        ("c", 2),
        ("d", 3),
    ]
    assert any("NARRATIVE_MERGE" in n for n in notes)


@pytest.mark.asyncio
async def test_synthesizes_whole_entry_label() -> None:
    """With synthesis enabled (default), the merged NARRATIVE's label is the
    LLM-synthesized whole-entry sentence, not a per-chunk fragment."""
    import anthropic

    from particles.llm import set_client

    mock_content = MagicMock()
    mock_content.text = "A hard day the author got through anyway."
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    set_client(mock_client)
    try:
        merged, _ = await collapse_chunk_narratives(_two_chunk_candidates())
    finally:
        set_client(None)

    narratives = [c for c in merged if c.particle_type == ParticleType.NARRATIVE]
    assert len(narratives) == 1
    assert narratives[0].content == "A hard day the author got through anyway."
    assert mock_client.messages.create.call_count == 1  # one synthesis call


@pytest.mark.asyncio
async def test_synthesis_failure_falls_back_to_first_label() -> None:
    """When the synthesis LLM call raises, the merge falls back to the first
    chunk's label + a note — it never fails extraction."""
    import anthropic

    from particles.llm import set_client

    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(side_effect=RuntimeError("boom"))

    set_client(mock_client)
    try:
        merged, notes = await collapse_chunk_narratives(_two_chunk_candidates())
    finally:
        set_client(None)

    narratives = [c for c in merged if c.particle_type == ParticleType.NARRATIVE]
    assert narratives[0].content == "Label A"
    assert any("synthesis failed" in n for n in notes)
