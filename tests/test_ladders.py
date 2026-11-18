# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/ladders.py: the engine's ladders and their deterministic rungs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from particles.core.duplicate_key import normalize_content
from particles.core.ladders import (
    ADJUDICABILITY,
    DUPLICATE_KEY,
    GENERIC_GUARD,
    POPULATION_QUANTIFIER,
    NormalizedForms,
    UpdatePair,
    candidate_ladder,
    contradiction_ladder,
    duplicate_key_rung,
    restatement_ladder,
    update_ladder,
)
from particles.core.rung_ladder import (
    CandidateVerdict,
    ContradictionVerdict,
    CostClass,
    QuestionKind,
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

    # Each case passes a factory, so the ladder is built inside the test rather
    # than at collection.
    @pytest.mark.parametrize(
        ("build", "kind", "models"),
        [
            (
                lambda: contradiction_ladder("p", "s"),
                QuestionKind.CONTRADICTION,
                {"contradiction_probe": "p", "second_reading": "s"},
            ),
            (
                lambda: update_ladder("h"),
                QuestionKind.UPDATE,
                {"update_slot": "h", "update_slot_passage": "h", "update_slot_second": "h"},
            ),
            (lambda: candidate_ladder("j"), QuestionKind.ENTAILMENT, {"entailment_judge": "j"}),
            (lambda: restatement_ladder("v"), QuestionKind.DUPLICATE, {"duplicate_judge": "v"}),
        ],
    )
    def test_each_ladder_stamps_its_model_rungs_with_the_version_given(
        self, build: Callable[[], Any], kind: QuestionKind, models: dict[str, str]
    ) -> None:
        """A model rung's stamp is what a verdict's ``decided_by`` records."""
        ladder = build()
        assert ladder.kind is kind
        assert {r.name: r.version for r in ladder.rungs if r.cost is CostClass.MODEL} == models

    def test_the_update_ladder_confirms_changes_only(self) -> None:
        ladder = update_ladder("h")
        for name in ("update_slot_passage", "update_slot_second"):
            rung = ladder.rung(name)
            assert rung.confirms == frozenset({UpdateVerdict.CHANGES.value})
            assert rung.cost is CostClass.MODEL

    def test_only_the_passage_reading_is_not_content_keyed(self) -> None:
        """The passage rung reads the pair's source passages as well as its two texts.

        Two pairs with one text and different passages need not share its
        verdict, so it may not be keyed on the texts alone. The second reading
        sees the texts only.
        """
        ladder = update_ladder("h")
        assert not ladder.rung("update_slot_passage").content_keyed
        assert ladder.rung("update_slot_second").content_keyed

    def test_a_sourced_pair_is_confirmed_on_its_passages(self) -> None:
        """the claim-text confirmation is the assertion door's alone."""
        ladder = update_ladder("h")
        for sourced, name in ((True, "update_slot_passage"), (False, "update_slot_second")):
            run = ladder.start(UpdatePair("A.", "B.", sourced=sourced))
            first = run.next_rung()
            assert first is not None and first.name == "update_slot"
            run.record(first, UpdateVerdict.CHANGES)
            confirm = run.next_rung()
            assert confirm is not None and confirm.name == name
            run.record(confirm, UpdateVerdict.CHANGES)
            assert run.next_rung() is None

    def test_the_candidate_ladder_settles_on_the_pre_check_alone(self) -> None:
        ladder = candidate_ladder("j")
        pure = settle(
            ladder.__class__(ladder.kind, ladder.abstain, (POPULATION_QUANTIFIER,)),
            ("Every operator deploys on Fridays.", ("a",)),
        )
        assert pure.decided_by == POPULATION_QUANTIFIER.stamp
