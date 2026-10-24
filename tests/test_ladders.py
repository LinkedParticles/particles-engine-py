# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/ladders.py: the engine's ladders and their deterministic rungs."""

from __future__ import annotations

import pytest

from particles.core.duplicate_key import normalize_content
from particles.core.ladders import (
    ADJUDICABILITY,
    DUPLICATE_KEY,
    GENERIC_GUARD,
    POPULATION_QUANTIFIER,
    NormalizedForms,
    candidate_ladder,
    contradiction_ladder,
    duplicate_key_rung,
    update_ladder,
)
from particles.core.rung_ladder import (
    CandidateVerdict,
    ContradictionVerdict,
    CostClass,
    RungStamp,
    UpdateVerdict,
    settle,
)
from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource

V = ContradictionVerdict


def _p(content: str, *, subjects: list[str] | None = None, apt: bool = True) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        subject_ids=subjects or [],
        assertion_modality=AssertionModality.FALSIFIABLE if apt else AssertionModality.EVALUATIVE,
    )


class TestDuplicateKeyRung:
    def test_the_same_sentence_is_not_a_contradiction(self) -> None:
        assert DUPLICATE_KEY.decide is not None
        pair = (_p("The PR was created at /pull/12."), _p("The PR was created at  /pull/12"))
        assert DUPLICATE_KEY.decide(pair) is V.NO

    def test_the_text_test_ignores_subject_sets_the_resolver_split(self) -> None:
        assert DUPLICATE_KEY.decide is not None
        pair = (_p("Main is at 1.2.", subjects=["s1"]), _p("Main is at 1.2.", subjects=["s2"]))
        assert DUPLICATE_KEY.decide(pair) is V.NO

    def test_two_sentences_abstain(self) -> None:
        assert DUPLICATE_KEY.decide is not None
        pair = (_p("Main is at 1.2."), _p("Main is at 1.3."))
        assert DUPLICATE_KEY.decide(pair) is V.ABSTAIN

    def test_it_is_local_and_keyed_on_content(self) -> None:
        assert DUPLICATE_KEY.cost is CostClass.LOCAL
        assert DUPLICATE_KEY.content_keyed

    def test_a_memo_normalises_each_claim_once_across_every_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from itertools import combinations

        from particles.core import ladders as ladders_mod

        calls: list[str] = []

        def counting(text: str) -> str:
            calls.append(text)
            return normalize_content(text)

        monkeypatch.setattr(ladders_mod, "normalize_content", counting)
        forms = NormalizedForms()
        rung = duplicate_key_rung(forms)
        claims = [_p(f"Main is at 1.{i}.") for i in range(4)] + [_p("Main is at 1.0.")]
        assert rung.decide is not None
        verdicts = [rung.decide(pair) for pair in combinations(claims, 2)]
        # Ten pairs, five claims: five normalisations, and the same verdicts
        # the memo-less rung gives.
        assert len(calls) == 5 and len(forms) == 5
        assert verdicts.count(V.NO) == 1
        assert DUPLICATE_KEY.decide is not None
        assert verdicts == [DUPLICATE_KEY.decide(pair) for pair in combinations(claims, 2)]
        assert rung.stamp == DUPLICATE_KEY.stamp


class TestPureRungs:
    def test_adjudicability_answers_no_for_a_non_truth_apt_side(self) -> None:
        assert ADJUDICABILITY.decide is not None
        assert ADJUDICABILITY.decide((_p("x"), _p("y", apt=False))) is V.NO
        assert ADJUDICABILITY.decide((_p("x"), _p("y"))) is V.ABSTAIN
        # It reads the stored modality, which two records of one text need not share.
        assert not ADJUDICABILITY.content_keyed

    def test_the_generic_guard_answers_no_for_a_generic_and_an_instance(self) -> None:
        assert GENERIC_GUARD.decide is not None
        generic = _p("Most mammals bear live young.")
        instance = _p("The platypus lays eggs.")
        assert GENERIC_GUARD.decide((generic, instance)) is V.NO
        assert GENERIC_GUARD.decide((instance, _p("The echidna lays eggs."))) is V.ABSTAIN

    def test_the_population_quantifier_rejects_a_kind_ranging_claim(self) -> None:
        assert POPULATION_QUANTIFIER.decide is not None
        verdict = POPULATION_QUANTIFIER.decide(("Most operators deploy on Fridays.", ("a",)))
        assert verdict is CandidateVerdict.NOT_ENTAILED


class TestLadders:
    def test_the_contradiction_ladder_runs_cheap_rungs_before_the_probe(self) -> None:
        ladder = contradiction_ladder("p1", "r1")
        assert [r.name for r in ladder.rungs] == [
            "adjudicability",
            "generic_guard",
            "duplicate_key",
            "contradiction_probe",
            "second_reading",
        ]
        assert ladder.rung("second_reading").confirms == frozenset({"yes"})

    def test_a_duplicate_pair_never_reaches_the_probe(self) -> None:
        ladder = contradiction_ladder("p1", "r1")
        run = ladder.start((_p("Main is at 1.2."), _p("Main is at 1.2.")))
        offered = []
        while (rung := run.next_rung()) is not None:
            offered.append(rung.name)
            assert rung.decide is not None, "the probe was offered"
            run.record(rung, rung.decide(run.item))
        assert offered == ["adjudicability", "generic_guard", "duplicate_key"]
        assert run.settled().decided_by == RungStamp("duplicate_key", "2")

    def test_the_update_ladder_confirms_changes_only(self) -> None:
        ladder = update_ladder("h")
        second = ladder.rung("update_slot_second")
        assert second.confirms == frozenset({UpdateVerdict.CHANGES.value})
        assert second.cost is CostClass.MODEL

    def test_the_candidate_ladder_settles_on_the_pre_check_alone(self) -> None:
        ladder = candidate_ladder("j")
        pure = settle(
            ladder.__class__(ladder.kind, ladder.abstain, (POPULATION_QUANTIFIER,)),
            ("Every operator deploys on Fridays.", ("a",)),
        )
        assert pure.decided_by == POPULATION_QUANTIFIER.stamp
