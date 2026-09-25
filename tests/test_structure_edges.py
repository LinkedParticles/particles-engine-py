# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for particles/ingest/structure_edges.py — the pure stance + narrative edge plan.

No DB and no mocks: :func:`plan_structure_edges` is a function of the
candidates, the candidate-index → stored-id map, and the stance specs
(D2). The end-to-end paths stay in ``test_extract.py``,
``test_journal.py`` and ``test_narrative.py``.
"""

from __future__ import annotations

from particles.core.schema import ParticleType, RelationType, UncertaintyNature
from particles.extraction.general import CandidateParticle
from particles.ingest.structure_edges import plan_structure_edges

PART_OF = RelationType.PART_OF
SEQUENCE_IN = RelationType.SEQUENCE_IN
ENDORSES = RelationType.ENDORSES
DISPUTES = RelationType.DISPUTES


def _claim(content: str, narrative_index: int | None = None) -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        narrative_index=narrative_index,
    )


def _narrative(label: str = "Narrative") -> CandidateParticle:
    return CandidateParticle(
        content=label,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        particle_type=ParticleType.NARRATIVE,
    )


# ── Stance edges ────────────────────────────────────────────


def test_stance_binds_to_landed_target() -> None:
    candidates = [_claim("t"), _claim("s1"), _claim("s2")]
    index_to_id = {0: "T", 1: "S1", 2: "S2"}
    specs = [(1, 0, ENDORSES), (2, 0, DISPUTES)]
    assert plan_structure_edges(candidates, index_to_id, specs) == [
        ("S1", "T", ENDORSES),
        ("S2", "T", DISPUTES),
    ]


def test_stance_binds_through_suppression_to_the_twin_id() -> None:
    """A target suppressed into an existing twin binds to the twin's stored id."""
    candidates = [_claim("t"), _claim("s")]
    assert plan_structure_edges(candidates, {0: "TWIN", 1: "S"}, [(1, 0, ENDORSES)]) == [
        ("S", "TWIN", ENDORSES)
    ]


def test_stance_with_dropped_endpoint_is_skipped() -> None:
    candidates = [_claim("t"), _claim("s1"), _claim("s2")]
    # Target 0 dropped for the first stance; stance 2 itself dropped.
    index_to_id = {1: "S1"}
    specs = [(1, 0, ENDORSES), (2, 1, DISPUTES)]
    assert plan_structure_edges(candidates, index_to_id, specs) == []


def test_stance_resolving_to_itself_is_skipped() -> None:
    """Stance and target folded into the same stored particle: no self-edge."""
    candidates = [_claim("t"), _claim("s")]
    assert plan_structure_edges(candidates, {0: "X", 1: "X"}, [(1, 0, ENDORSES)]) == []


# ── Narrative graph ──────────────────────────────────────────────


def test_narrative_graph_part_of_and_sequence() -> None:
    candidates = [_claim("a", 0), _claim("b", 1), _claim("c", 2), _narrative()]
    index_to_id = {0: "A", 1: "B", 2: "C", 3: "N"}
    assert plan_structure_edges(candidates, index_to_id, []) == [
        ("A", "N", PART_OF),
        ("B", "N", PART_OF),
        ("A", "B", SEQUENCE_IN),
        ("C", "N", PART_OF),
        ("B", "C", SEQUENCE_IN),
    ]


def test_constituents_ordered_by_narrative_index_not_position() -> None:
    candidates = [_claim("c", 2), _narrative(), _claim("a", 0), _claim("b", 1)]
    index_to_id = {0: "C", 1: "N", 2: "A", 3: "B"}
    assert plan_structure_edges(candidates, index_to_id, []) == [
        ("A", "N", PART_OF),
        ("B", "N", PART_OF),
        ("A", "B", SEQUENCE_IN),
        ("C", "N", PART_OF),
        ("B", "C", SEQUENCE_IN),
    ]


def test_no_narrative_writes_no_narrative_edges() -> None:
    candidates = [_claim("a", 0), _claim("b", 1)]
    assert plan_structure_edges(candidates, {0: "A", 1: "B"}, []) == []


def test_several_narratives_write_no_narrative_edges() -> None:
    candidates = [_claim("a", 0), _narrative("N1"), _claim("b", 1), _narrative("N2")]
    index_to_id = {0: "A", 1: "N1", 2: "B", 3: "N2"}
    assert plan_structure_edges(candidates, index_to_id, []) == []


def test_only_landed_narratives_count() -> None:
    """Two NARRATIVE candidates, one dropped: exactly one landed, so the graph is written."""
    candidates = [_claim("a", 0), _narrative("N1"), _narrative("N2")]
    assert plan_structure_edges(candidates, {0: "A", 1: "N1"}, []) == [("A", "N1", PART_OF)]


def test_dropped_narrative_writes_no_narrative_edges() -> None:
    candidates = [_claim("a", 0), _claim("b", 1), _narrative()]
    assert plan_structure_edges(candidates, {0: "A", 1: "B"}, []) == []


def test_dropped_constituent_is_absent_from_the_chain() -> None:
    candidates = [_claim("a", 0), _claim("b", 1), _claim("c", 2), _narrative()]
    index_to_id = {0: "A", 2: "C", 3: "N"}
    assert plan_structure_edges(candidates, index_to_id, []) == [
        ("A", "N", PART_OF),
        ("C", "N", PART_OF),
        ("A", "C", SEQUENCE_IN),
    ]


def test_candidates_without_narrative_index_are_not_constituents() -> None:
    candidates = [_claim("a", 0), _claim("loose"), _narrative()]
    assert plan_structure_edges(candidates, {0: "A", 1: "L", 2: "N"}, []) == [("A", "N", PART_OF)]


def test_constituent_suppressed_into_the_container_is_skipped() -> None:
    """a constituent folded into the NARRATIVE gets no PART_OF self-edge
    and does not break the chain between its neighbours."""
    candidates = [_claim("a", 0), _claim("b", 1), _claim("c", 2), _narrative()]
    index_to_id = {0: "A", 1: "N", 2: "C", 3: "N"}
    assert plan_structure_edges(candidates, index_to_id, []) == [
        ("A", "N", PART_OF),
        ("C", "N", PART_OF),
        ("A", "C", SEQUENCE_IN),
    ]


def test_constituent_suppressed_into_its_predecessor_gets_no_sequence_self_edge() -> None:
    """a constituent folded into its predecessor writes no SEQUENCE_IN
    self-edge, and the chain resumes from the shared id.

    The plan keeps the pipeline's behaviour from before the cut, which re-emits the
    shared id's PART_OF edge; this test pins that so a change to it is deliberate.
    """
    candidates = [_claim("a", 0), _claim("b", 1), _claim("c", 2), _narrative()]
    index_to_id = {0: "A", 1: "A", 2: "C", 3: "N"}
    assert plan_structure_edges(candidates, index_to_id, []) == [
        ("A", "N", PART_OF),
        ("A", "N", PART_OF),
        ("C", "N", PART_OF),
        ("A", "C", SEQUENCE_IN),
    ]


def test_stance_edges_precede_narrative_edges() -> None:
    candidates = [_claim("a", 0), _claim("s"), _narrative()]
    index_to_id = {0: "A", 1: "S", 2: "N"}
    assert plan_structure_edges(candidates, index_to_id, [(1, 0, ENDORSES)]) == [
        ("S", "A", ENDORSES),
        ("A", "N", PART_OF),
    ]
