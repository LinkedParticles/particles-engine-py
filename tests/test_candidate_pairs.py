# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure unit tests for the shared candidate-pair decide step.

``enumerate_candidate_pairs`` and ``co_evidential_components`` take plain
values, so every candidacy rule is exercised here with no session, no store
and no mocks (D2). The DB-backed behaviour of the two callers stays
covered by ``test_lint.py`` and ``test_links_suggest.py``.
"""

from __future__ import annotations

import numpy as np

from particles.core.equivalence import co_evidential_components
from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    ParticleRelation,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.stance import STANCE_HOLDER_KEY
from particles.extraction.polarity import POLARITY_DECLINED, POLARITY_KEY
from particles.operations.candidate_pairs import (
    CandidatePair,
    enumerate_candidate_pairs,
    is_pair_eligible,
)


def _p(
    pid: str,
    *,
    properties: dict[str, object] | None = None,
    modality: AssertionModality = AssertionModality.FALSIFIABLE,
) -> Particle:
    return Particle(
        id=pid,
        content=f"claim {pid}",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce")],
        assertion_modality=modality,
        properties=properties or {},
    )


def _emb(angle: float) -> np.ndarray:
    """A unit vector in the plane; cosine between two is cos(Δangle)."""
    return np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float32)


def _edge(a: str, b: str, confidence: float = 1.0) -> ParticleRelation:
    return ParticleRelation(
        particle_a=a,
        particle_b=b,
        relation_type=RelationType.CO_EVIDENTIAL,
        created_by=RelationCreatedBy.HUMAN_REVIEW,
        confidence=confidence,
    )


def _ids(pairs: list[CandidatePair]) -> list[tuple[str, str]]:
    return [(c.a.id, c.b.id) for c in pairs]


# ---------------------------------------------------------------------------
# co_evidential_components
# ---------------------------------------------------------------------------


class TestCoEvidentialComponents:
    def test_empty(self) -> None:
        assert co_evidential_components([], 0.0) == {}

    def test_transitive_closure_maps_every_member(self) -> None:
        comps = co_evidential_components([_edge("a", "b"), _edge("c", "b"), _edge("x", "y")], 0.0)
        abc = frozenset({"a", "b", "c"})
        assert comps == {
            "a": abc,
            "b": abc,
            "c": abc,
            "x": frozenset({"x", "y"}),
            "y": frozenset({"x", "y"}),
        }

    def test_edges_below_min_confidence_are_not_traversed(self) -> None:
        comps = co_evidential_components([_edge("a", "b", 0.9), _edge("b", "c", 0.4)], 0.5)
        assert comps["a"] == frozenset({"a", "b"})
        assert "c" not in comps

    def test_accepts_any_iterable(self) -> None:
        comps = co_evidential_components(iter([_edge("a", "b")]), 0.0)
        assert comps["b"] == frozenset({"a", "b"})


# ---------------------------------------------------------------------------
# is_pair_eligible
# ---------------------------------------------------------------------------


class TestIsPairEligible:
    def test_falsifiable_asserted_is_eligible(self) -> None:
        assert is_pair_eligible(_p("a"))

    def test_non_truth_apt_is_not(self) -> None:
        assert not is_pair_eligible(_p("a", modality=AssertionModality.EVALUATIVE))

    def test_non_asserted_is_not(self) -> None:
        assert not is_pair_eligible(_p("a", properties={POLARITY_KEY: POLARITY_DECLINED}))


# ---------------------------------------------------------------------------
# enumerate_candidate_pairs
# ---------------------------------------------------------------------------


class TestEnumerateCandidatePairs:
    def test_fewer_than_two_candidates(self) -> None:
        assert enumerate_candidate_pairs([], threshold=0.5, linked={}) == []
        assert enumerate_candidate_pairs([(_p("a"), _emb(0.0))], threshold=0.5, linked={}) == []

    def test_threshold_is_inclusive_and_similarity_on_unit_scale(self) -> None:
        cands = [(_p("a"), _emb(0.0)), (_p("b"), _emb(0.0)), (_p("c"), _emb(np.pi / 2))]
        pairs = enumerate_candidate_pairs(cands, threshold=0.5, linked={})
        assert _ids(pairs) == [("a", "b")]
        assert pairs[0].similarity == np.float32(1.0)
        assert pairs[0].tier == 0

    def test_anti_correlated_pair_clamps_to_zero(self) -> None:
        cands = [(_p("a"), _emb(0.0)), (_p("b"), _emb(np.pi))]
        pairs = enumerate_candidate_pairs(cands, threshold=0.0, linked={})
        assert [c.similarity for c in pairs] == [0.0]

    def test_unnormalised_vectors_are_normalised(self) -> None:
        cands = [(_p("a"), 3 * _emb(0.0)), (_p("b"), 0.5 * _emb(0.0))]
        pairs = enumerate_candidate_pairs(cands, threshold=0.99, linked={})
        assert _ids(pairs) == [("a", "b")]

    def test_orientation_follows_candidate_order(self) -> None:
        cands = [(_p("z"), _emb(0.0)), (_p("a"), _emb(0.0))]
        assert _ids(enumerate_candidate_pairs(cands, threshold=0.5, linked={})) == [("z", "a")]

    def test_ordered_by_similarity_then_ids(self) -> None:
        cands = [
            (_p("a"), _emb(0.0)),
            (_p("b"), _emb(0.3)),
            (_p("c"), _emb(0.1)),
            (_p("d"), _emb(0.1)),
        ]
        pairs = enumerate_candidate_pairs(cands, threshold=0.9, linked={})
        sims = [c.similarity for c in pairs]
        assert sims == sorted(sims, reverse=True)
        # c and d are identical, so (c, d) leads; a is equidistant-ish to c/d
        # and the id tie-break orders (a, c) before (a, d).
        assert _ids(pairs)[0] == ("c", "d")
        assert _ids(pairs).index(("a", "c")) < _ids(pairs).index(("a", "d"))

    def test_already_linked_pairs_are_skipped_in_either_direction(self) -> None:
        cands = [(_p("a"), _emb(0.0)), (_p("b"), _emb(0.0)), (_p("c"), _emb(0.0))]
        linked = co_evidential_components([_edge("b", "a")], 0.0)
        pairs = enumerate_candidate_pairs(cands, threshold=0.5, linked=linked)
        assert sorted(_ids(pairs)) == [("a", "c"), ("b", "c")]

    def test_transitively_linked_pair_is_skipped(self) -> None:
        cands = [(_p("a"), _emb(0.0)), (_p("c"), _emb(0.0))]
        linked = co_evidential_components([_edge("a", "b"), _edge("b", "c")], 0.0)
        assert enumerate_candidate_pairs(cands, threshold=0.5, linked=linked) == []

    def test_self_pair_is_never_a_candidate(self) -> None:
        # A particle's co-evidential group always includes itself.
        cands = [(_p("a"), _emb(0.0)), (_p("a"), _emb(0.0))]
        assert enumerate_candidate_pairs(cands, threshold=0.5, linked={}) == []

    def test_excluded_pair_is_skipped_unordered(self) -> None:
        cands = [(_p("a"), _emb(0.0)), (_p("b"), _emb(0.0)), (_p("c"), _emb(0.0))]
        pairs = enumerate_candidate_pairs(
            cands, threshold=0.5, linked={}, exclude={frozenset({"b", "a"})}
        )
        assert ("a", "b") not in _ids(pairs)
        assert len(pairs) == 2

    def test_stance_pairs_only_with_same_holder(self) -> None:
        cands = [
            (_p("plain"), _emb(0.0)),
            (_p("alice1", properties={STANCE_HOLDER_KEY: "alice"}), _emb(0.0)),
            (_p("alice2", properties={STANCE_HOLDER_KEY: "alice"}), _emb(0.0)),
            (_p("bob", properties={STANCE_HOLDER_KEY: "bob"}), _emb(0.0)),
            (_p("plain2"), _emb(0.0)),
        ]
        pairs = enumerate_candidate_pairs(cands, threshold=0.5, linked={})
        assert sorted(_ids(pairs)) == [("alice1", "alice2"), ("plain", "plain2")]

    def test_scope_drops_out_of_scope_pairs_and_tiers_the_rest(self) -> None:
        # (h1, s) is the most similar pair but mixed; (h1, h2) is intra-scope.
        cands = [(_p("h1"), _emb(0.0)), (_p("h2"), _emb(0.2)), (_p("s"), _emb(0.01))]
        cands.append((_p("t"), _emb(0.02)))
        pairs = enumerate_candidate_pairs(
            cands, threshold=0.9, linked={}, scope=frozenset({"h1", "h2"})
        )
        tiers = [c.tier for c in pairs]
        assert tiers == sorted(tiers)
        assert _ids(pairs)[0] == ("h1", "h2")
        assert pairs[0].tier == 0
        assert ("s", "t") not in _ids(pairs)  # neither side in scope
        assert all(c.tier == 1 for c in pairs[1:])

    def test_empty_scope_keeps_nothing(self) -> None:
        cands = [(_p("a"), _emb(0.0)), (_p("b"), _emb(0.0))]
        assert enumerate_candidate_pairs(cands, threshold=0.5, linked={}, scope=frozenset()) == []

    def test_no_scope_is_all_tier_zero(self) -> None:
        cands = [(_p("a"), _emb(0.0)), (_p("b"), _emb(0.1)), (_p("c"), _emb(0.2))]
        pairs = enumerate_candidate_pairs(cands, threshold=0.9, linked={})
        assert pairs and all(c.tier == 0 for c in pairs)
