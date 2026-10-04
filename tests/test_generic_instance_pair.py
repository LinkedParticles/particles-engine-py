# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A generic claim and an instance claim are not an adjudicable pair.

"Most mammals bear live young" and "the platypus lays eggs" never meet on the
§6.6 ladder: an exception does not falsify "most". Pure tests, no session, for
each place the rule lives: the detector in ``core/`` (moved there from the
abstraction pass, behaviour unchanged), the guard beside the truth-apt gate in
:func:`~particles.core.conflict_resolution.resolve_conflict`, the overrides in
:func:`~particles.core.conflict_resolution.decide_ladder`, the pipeline's
nearest-claim selection, the subject pool, and the shared lint candidate-pair
site. The DB-backed write paths are pinned in ``test_extract.py`` and
``test_update_supersession.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from particles.core import generics
from particles.core.conflict_resolution import (
    ConflictVerdict,
    RungInputs,
    decide_ladder,
    is_generic_instance_pair,
    resolve_conflict,
)
from particles.core.observer_scope import PairPrecondition
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.ingest.pipeline import _find_conflict
from particles.ingest.update_supersession import SubjectIndex
from particles.operations import abstraction
from particles.operations.candidate_pairs import enumerate_candidate_pairs

GENERIC = "Most mammals bear live young."
GENERIC_NEGATED = "Most mammals do not bear live young."
INSTANCE = "The platypus lays eggs."
INSTANCE_OTHER = "The platypus bears live young."


def _p(content: str, pid: str | None = None) -> Particle:
    kwargs: dict[str, object] = {} if pid is None else {"id": pid}
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce", snapshot_id="s")
        ],
        asserted_by="test",
        subject_ids=["mammal"],
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# The detector, moved to core
# ---------------------------------------------------------------------------


class TestDetector:
    def test_abstraction_reads_the_core_detector(self) -> None:
        """One detector: the abstraction pass imports it from ``core``."""
        assert abstraction.population_quantifier is generics.population_quantifier

    @pytest.mark.parametrize(
        "claim",
        [
            GENERIC,
            GENERIC_NEGATED,
            "All mammals bear live young.",
            "Mammals bear live young.",
            "Engineers tend to prefer dark mode.",
            "Rust users typically prefer vim keybindings.",
        ],
    )
    def test_generic_claims(self, claim: str) -> None:
        assert generics.is_generic_claim(claim)

    @pytest.mark.parametrize(
        "claim",
        [
            INSTANCE,
            INSTANCE_OTHER,
            "The tower is 300 metres tall.",
            "Alice typically uses vim.",
            "Alice, Bob, and Carol all prefer vim.",
            "Every backgrounded commit in this store failed at GPG signing.",
        ],
    )
    def test_instance_claims(self, claim: str) -> None:
        assert not generics.is_generic_claim(claim)

    def test_an_evidential_clause_does_not_make_an_instance_claim_generic(self) -> None:
        """The abstraction pass's fourth shape is its own: citing is not ranging."""
        claim = "Alice uses vim, as the recorded sessions show."
        assert generics.population_quantifier(claim) == "as the recorded sessions show"
        assert generics.generic_quantifier(claim) is None
        claim = "According to the 2020 census, Paris has 2.1 million residents."
        assert generics.population_quantifier(claim) is not None
        assert not generics.is_generic_claim(claim)

    def test_the_two_readings_agree_on_every_kind_shape(self) -> None:
        for claim in (GENERIC, "Mammals bear live young.", "Christians generally attend church."):
            assert generics.generic_quantifier(claim) == generics.population_quantifier(claim)


# ---------------------------------------------------------------------------
# The guard beside the truth-apt gate
# ---------------------------------------------------------------------------


class TestGuard:
    def test_exactly_one_generic_side_is_excluded(self) -> None:
        assert is_generic_instance_pair(_p(GENERIC), _p(INSTANCE))
        assert is_generic_instance_pair(_p(INSTANCE), _p(GENERIC))
        assert not is_generic_instance_pair(_p(GENERIC), _p(GENERIC_NEGATED))
        assert not is_generic_instance_pair(_p(INSTANCE), _p(INSTANCE_OTHER))

    @pytest.mark.parametrize(
        ("existing", "new"), [(GENERIC, INSTANCE), (INSTANCE, GENERIC)], ids=["g-i", "i-g"]
    )
    def test_generic_against_instance_corroborates(self, existing: str, new: str) -> None:
        """A confirmed signal, a winning trust gap, or a newer value retires nothing."""
        for kwargs in (
            {},
            {"trust_score_existing": 0.1, "trust_score_new": 0.9},
            {"trust_score_existing": 0.9, "trust_score_new": 0.1},
            {"update_order": 1},
            {"update_order": -1},
            {"single_trust_order": False},
        ):
            verdict = resolve_conflict(
                _p(existing), _p(new), has_contradiction_signal=True, **kwargs
            )  # type: ignore[arg-type]
            assert verdict is ConflictVerdict.CORROBORATES, kwargs

    def test_two_generics_still_reach_the_ladder(self) -> None:
        a, b = _p(GENERIC), _p(GENERIC_NEGATED)
        assert resolve_conflict(a, b) is ConflictVerdict.INCONSISTENT
        assert (
            resolve_conflict(a, b, trust_score_existing=0.1, trust_score_new=0.9)
            is ConflictVerdict.SUPERSEDES
        )

    def test_two_instances_are_unchanged(self) -> None:
        a, b = _p(INSTANCE), _p(INSTANCE_OTHER)
        assert resolve_conflict(a, b) is ConflictVerdict.INCONSISTENT
        assert resolve_conflict(a, b, update_order=1) is ConflictVerdict.UPDATE_SUPERSEDES
        assert (
            resolve_conflict(a, b, has_contradiction_signal=False) is ConflictVerdict.CORROBORATES
        )

    def test_the_editorial_supersession_prior_still_runs_above_it(self) -> None:
        """Rung 1.5 is not adjudication: like the truth-apt gate, the guard sits below it."""
        verdict = resolve_conflict(_p(GENERIC), _p(INSTANCE), new_supersedes_existing=True)
        assert verdict is ConflictVerdict.DOCUMENT_SUPERSEDES


class TestDecideLadder:
    @staticmethod
    def _decide(
        existing: str, new: str, *, probe: bool | None, precondition: PairPrecondition
    ) -> tuple[ConflictVerdict | None, bool]:
        outcome = decide_ladder(
            _p(existing),
            _p(new),
            probe=probe,
            fail_closed=True,
            precondition=precondition,
            rung_inputs=RungInputs(new_supersedes_existing=True),
            single_trust_order=True,
            trust_differential_threshold=0.15,
        )
        return outcome.verdict, outcome.record_divergence

    def test_a_failed_probe_never_forces_an_inconsistency(self) -> None:
        """Nor feeds the supersession prior: the forced pair corroborates outright."""
        assert self._decide(
            GENERIC, INSTANCE, probe=None, precondition=PairPrecondition.RECONCILE
        ) == (ConflictVerdict.CORROBORATES, False)
        assert self._decide(
            INSTANCE, INSTANCE_OTHER, probe=None, precondition=PairPrecondition.RECONCILE
        ) == (ConflictVerdict.INCONSISTENT, False)

    def test_a_contested_global_claim_is_not_sent_to_review(self) -> None:
        assert self._decide(
            GENERIC, INSTANCE, probe=True, precondition=PairPrecondition.REVIEW
        ) == (ConflictVerdict.CORROBORATES, False)
        assert self._decide(
            GENERIC, GENERIC_NEGATED, probe=True, precondition=PairPrecondition.REVIEW
        ) == (ConflictVerdict.INCONSISTENT, False)

    def test_a_declined_pair_records_no_divergence(self) -> None:
        assert self._decide(
            GENERIC, INSTANCE, probe=True, precondition=PairPrecondition.DECLINE
        ) == (None, False)
        assert self._decide(
            INSTANCE, INSTANCE_OTHER, probe=True, precondition=PairPrecondition.DECLINE
        ) == (None, True)


# ---------------------------------------------------------------------------
# Pair selection: the pipeline, the subject pool, the lint
# ---------------------------------------------------------------------------


def _emb(*xs: float) -> np.ndarray:  # type: ignore[type-arg]
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestPipelineSelection:
    def test_the_nearest_claim_of_the_other_kind_is_passed_over(self) -> None:
        generic, instance = _p(GENERIC), _p(INSTANCE_OTHER)
        existing = [generic, instance]
        # The generic is the closer neighbour of the candidate.
        embs = [_emb(1, 0.01), _emb(1, 0.05)]
        q = _emb(1, 0)
        assert _find_conflict(q, existing, embs) is generic
        assert _find_conflict(q, existing, embs, candidate_generic=False) is instance
        assert _find_conflict(q, existing, embs, candidate_generic=True) is generic

    def test_no_pair_when_only_the_other_kind_is_near(self) -> None:
        q = _emb(1, 0)
        assert _find_conflict(q, [_p(GENERIC)], [_emb(1, 0)], candidate_generic=False) is None

    def test_the_subject_pool_keeps_one_kind(self) -> None:
        generic, instance = _p(GENERIC), _p(INSTANCE_OTHER)
        index = SubjectIndex.build([(generic, _emb(1, 0.01)), (instance, _emb(1, 0.05))])
        q = _emb(1, 0)
        assert index.candidates(["mammal"], q, floor=0.5, limit=3) == [generic, instance]
        assert index.candidates(["mammal"], q, floor=0.5, limit=3, generic=False) == [instance]
        assert index.candidates(["mammal"], q, floor=0.5, limit=3, generic=True) == [generic]


class TestLintCandidatePairs:
    """One site for L-SEM-01, L-IDX-01 and the audit / census probe."""

    def test_only_the_generic_instance_pairing_is_dropped(self) -> None:
        g1, g2 = _p(GENERIC, "g1"), _p(GENERIC_NEGATED, "g2")
        i1, i2 = _p(INSTANCE, "i1"), _p(INSTANCE_OTHER, "i2")
        same = _emb(1, 0)
        pairs = enumerate_candidate_pairs(
            [(g1, same), (g2, same), (i1, same), (i2, same)], threshold=0.9, linked={}
        )
        assert {frozenset((c.a.id, c.b.id)) for c in pairs} == {
            frozenset(("g1", "g2")),
            frozenset(("i1", "i2")),
        }
