# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for particles/ingest/narrative_merge.py — the narrative merge.

The collapse is pure (D2): :func:`collapse_narratives` takes the
whole-entry label as a value, so its cases need no LLM patch. Only
:func:`_merge_label`, the one model call, is tested against a mocked client.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from particles.core.schema import ParticleType, UncertaintyNature
from particles.extraction.general import CandidateParticle
from particles.ingest.narrative_merge import (
    _merge_label,
    collapse_narratives,
    narrative_labels,
)


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


# ---------------------------------------------------------------------------
# narrative_labels — pure
# ---------------------------------------------------------------------------


def test_narrative_labels_in_candidate_order() -> None:
    assert narrative_labels(_two_chunk_candidates()) == ["Label A", "Label B"]


def test_narrative_labels_empty_without_narratives() -> None:
    assert narrative_labels([_claim("a", 0)]) == []


# ---------------------------------------------------------------------------
# collapse_narratives — pure, no LLM patch
# ---------------------------------------------------------------------------


def test_single_narrative_is_noop() -> None:
    """A single-pass journal (one NARRATIVE candidate) is returned untouched,
    and the label argument is ignored."""
    candidates = [_claim("a", 0), _claim("b", 1), _narrative("Label A")]
    merged, notes = collapse_narratives(candidates, "ignored")
    assert merged is candidates  # same object — no-op
    assert notes == []
    assert candidates[2].content == "Label A"


def test_no_narrative_is_noop() -> None:
    """A non-journal extraction (zero NARRATIVE candidates) is returned untouched."""
    candidates = [_claim("a", 0)]
    # A bare CLAIM carrying no narrative_index (the general-extractor shape) must
    # also be left alone.
    candidates[0].narrative_index = None
    merged, notes = collapse_narratives(candidates, "ignored")
    assert merged is candidates
    assert notes == []


def test_empty_is_noop() -> None:
    candidates: list[CandidateParticle] = []
    merged, notes = collapse_narratives(candidates, "ignored")
    assert merged is candidates
    assert notes == []


def test_collapses_and_reindexes_globally() -> None:
    """Two per-chunk NARRATIVE fragments collapse to one carrying the given
    label; constituents get a single global narrative_index in document order."""
    merged, notes = collapse_narratives(_two_chunk_candidates(), "Whole entry.")

    narratives = [c for c in merged if c.particle_type == ParticleType.NARRATIVE]
    claims = [c for c in merged if c.particle_type != ParticleType.NARRATIVE]
    assert len(narratives) == 1
    assert narratives[0].content == "Whole entry."
    assert [(c.content, c.narrative_index) for c in claims] == [
        ("a", 0),
        ("b", 1),
        ("c", 2),
        ("d", 3),
    ]
    assert notes == [
        "NARRATIVE_MERGE: collapsed 2 per-chunk narrative fragments into one "
        "whole-entry NARRATIVE over 4 constituents"
    ]


def test_surviving_narrative_is_the_first_in_place() -> None:
    """The first NARRATIVE survives at its own position; later ones are dropped."""
    candidates = _two_chunk_candidates()
    first = candidates[2]
    merged, _ = collapse_narratives(candidates, "Whole entry.")
    assert merged[2] is first
    assert [c.content for c in merged] == ["a", "b", "Whole entry.", "c", "d"]


def test_unindexed_candidate_is_not_conscripted() -> None:
    """A candidate with no narrative_index keeps None and does not consume a slot."""
    candidates = _two_chunk_candidates()
    candidates.insert(3, _claim("stray", 0))
    candidates[3].narrative_index = None
    merged, notes = collapse_narratives(candidates, "L")
    assert [(c.content, c.narrative_index) for c in merged if c.content != "L"] == [
        ("a", 0),
        ("b", 1),
        ("stray", None),
        ("c", 2),
        ("d", 3),
    ]
    assert "over 4 constituents" in notes[0]


def test_three_fragments_collapse_to_one() -> None:
    candidates = [*_two_chunk_candidates(), _claim("e", 0), _narrative("Label C")]
    merged, notes = collapse_narratives(candidates, "L")
    assert [c.content for c in merged if c.particle_type == ParticleType.NARRATIVE] == ["L"]
    assert [c.narrative_index for c in merged if c.content in "abcde"] == [0, 1, 2, 3, 4]
    assert "collapsed 3 per-chunk" in notes[0]


# ---------------------------------------------------------------------------
# _merge_label — the one model call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_label_disabled_keeps_first_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthesis disabled → the first chunk's label, deterministically, with no
    LLM call and no note."""
    from particles.config import get_config

    monkeypatch.setattr(get_config().journal_extractor, "synthesize_merged_narrative", False)

    notes: list[str] = []
    assert await _merge_label(["Label A", "Label B"], notes) == "Label A"
    assert notes == []


def _mock_client(**create_kwargs: object) -> MagicMock:
    import anthropic

    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(**create_kwargs)
    return mock_client


def _text_response(text: str) -> MagicMock:
    mock_content = MagicMock()
    mock_content.text = text
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    return mock_resp


@pytest.mark.asyncio
async def test_merge_label_synthesizes_whole_entry_label() -> None:
    """With synthesis enabled (default), the label is the LLM-synthesized
    whole-entry sentence, from exactly one call."""
    from particles.llm import set_client

    mock_client = _mock_client(
        return_value=_text_response('"A hard day the author got through anyway."')
    )
    notes: list[str] = []
    set_client(mock_client)
    try:
        label = await _merge_label(["Label A", "Label B"], notes)
    finally:
        set_client(None)

    assert label == "A hard day the author got through anyway."
    assert notes == []
    assert mock_client.messages.create.call_count == 1  # one synthesis call


@pytest.mark.asyncio
async def test_merge_label_failure_falls_back_to_first_label() -> None:
    """When the synthesis call raises, the first chunk's label is used and a
    note is recorded; extraction never fails."""
    from particles.llm import set_client

    mock_client = _mock_client(side_effect=RuntimeError("boom"))
    notes: list[str] = []
    set_client(mock_client)
    try:
        label = await _merge_label(["Label A", "Label B"], notes)
    finally:
        set_client(None)

    assert label == "Label A"
    assert any("synthesis failed" in n for n in notes)


@pytest.mark.asyncio
async def test_merge_label_empty_response_falls_back_to_first_label() -> None:
    from particles.llm import set_client

    mock_client = _mock_client(return_value=_text_response("  "))
    notes: list[str] = []
    set_client(mock_client)
    try:
        label = await _merge_label(["Label A", "Label B"], notes)
    finally:
        set_client(None)

    assert label == "Label A"
    assert any("returned empty" in n for n in notes)
