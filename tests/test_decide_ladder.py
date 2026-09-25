# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for the overrides around the §6.6 verdict (D2).

``decide_ladder`` folds the tri-state probe, the observer
precondition and the rung inputs into one outcome. Before it was
lifted out of ``pipeline._resolve_conflict``, the override order was reachable
only through a session; these tests pin it with no DB.
"""

from __future__ import annotations

import pytest

from particles.core.conflict_resolution import (
    ConflictVerdict,
    LadderOutcome,
    RungInputs,
    UpdateOrderSource,
    decide_ladder,
    effective_single_trust_order,
    forces_inconsistent,
    ladder_signal,
    needs_rung_inputs,
    update_order_source,
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

RECONCILE = PairPrecondition.RECONCILE
DECLINE = PairPrecondition.DECLINE
REVIEW = PairPrecondition.REVIEW


def _particle(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
        asserted_by="test",
    )


def _decide(
    *,
    probe: bool | None,
    fail_closed: bool = False,
    precondition: PairPrecondition = RECONCILE,
    rung_inputs: RungInputs | None = None,
    single_trust_order: bool = True,
) -> LadderOutcome:
    return decide_ladder(
        _particle("The tower is 300 metres tall."),
        _particle("The tower is 324 metres tall."),
        probe=probe,
        fail_closed=fail_closed,
        precondition=precondition,
        rung_inputs=rung_inputs,
        single_trust_order=single_trust_order,
        trust_differential_threshold=0.15,
    )


class TestLadderSignal:
    @pytest.mark.parametrize(
        ("probe", "fail_closed", "expected"),
        [
            (True, False, True),
            (True, True, True),
            (False, False, False),
            (False, True, False),
            (None, False, False),  # extraction: fail open
            (None, True, True),  # assertion pathway: fail closed
        ],
    )
    def test_tri_state(self, probe: bool | None, fail_closed: bool, expected: bool) -> None:
        assert ladder_signal(probe, fail_closed=fail_closed) is expected


class TestOverrideOrder:
    """DECLINE outranks a forced INCONSISTENT; both outrank the rungs."""

    def test_decline_outranks_fail_closed(self) -> None:
        outcome = _decide(probe=None, fail_closed=True, precondition=DECLINE)
        assert outcome == LadderOutcome(verdict=None, record_divergence=True)

    def test_decline_outranks_review_force(self) -> None:
        # REVIEW and DECLINE are exclusive preconditions; DECLINE with a
        # confirmed signal still declines and records.
        outcome = _decide(probe=True, precondition=DECLINE)
        assert outcome == LadderOutcome(verdict=None, record_divergence=True)

    def test_decline_outranks_winning_rung(self) -> None:
        # A trust differential that would supersede never runs on a declined pair.
        inputs = RungInputs(trust_score_new=0.9, trust_score_existing=0.1)
        outcome = _decide(probe=True, precondition=DECLINE, rung_inputs=inputs)
        assert outcome.verdict is None

    @pytest.mark.parametrize("probe", [False, None])
    def test_decline_without_signal_records_nothing(self, probe: bool | None) -> None:
        outcome = _decide(probe=probe, fail_closed=False, precondition=DECLINE)
        assert outcome == LadderOutcome(verdict=None, record_divergence=False)

    def test_fail_closed_forces_inconsistent_over_trust(self) -> None:
        inputs = RungInputs(trust_score_new=0.9, trust_score_existing=0.1)
        outcome = _decide(probe=None, fail_closed=True, rung_inputs=inputs)
        assert outcome == LadderOutcome(ConflictVerdict.INCONSISTENT, record_divergence=False)

    def test_fail_closed_forces_inconsistent_over_update_order(self) -> None:
        outcome = _decide(probe=None, fail_closed=True, rung_inputs=RungInputs(update_order=1))
        assert outcome.verdict is ConflictVerdict.INCONSISTENT

    def test_review_with_signal_forces_inconsistent_over_rungs(self) -> None:
        inputs = RungInputs(trust_score_new=0.9, trust_score_existing=0.1, update_order=1)
        outcome = _decide(probe=True, precondition=REVIEW, rung_inputs=inputs)
        assert outcome.verdict is ConflictVerdict.INCONSISTENT

    def test_review_without_signal_corroborates(self) -> None:
        assert _decide(probe=False, precondition=REVIEW).verdict is ConflictVerdict.CORROBORATES

    def test_review_fail_open_probe_corroborates(self) -> None:
        # An incomplete probe on extraction is no signal, so REVIEW forces nothing.
        outcome = _decide(probe=None, fail_closed=False, precondition=REVIEW)
        assert outcome.verdict is ConflictVerdict.CORROBORATES


class TestLadderFallsThrough:
    """With no override, the outcome is ``resolve_conflict``'s verdict."""

    def test_fail_open_incomplete_probe_corroborates(self) -> None:
        assert _decide(probe=None).verdict is ConflictVerdict.CORROBORATES

    def test_no_signal_corroborates(self) -> None:
        assert _decide(probe=False).verdict is ConflictVerdict.CORROBORATES

    def test_signal_without_inputs_is_inconsistent(self) -> None:
        assert _decide(probe=True).verdict is ConflictVerdict.INCONSISTENT

    def test_trust_rung(self) -> None:
        up = RungInputs(trust_score_new=0.9, trust_score_existing=0.1)
        down = RungInputs(trust_score_new=0.1, trust_score_existing=0.9)
        assert _decide(probe=True, rung_inputs=up).verdict is ConflictVerdict.SUPERSEDES
        assert (
            _decide(probe=True, rung_inputs=down).verdict is ConflictVerdict.SUPERSEDED_BY_EXISTING
        )

    def test_trust_rung_skipped_in_multi_store(self) -> None:
        up = RungInputs(trust_score_new=0.9, trust_score_existing=0.1)
        outcome = _decide(probe=True, rung_inputs=up, single_trust_order=False)
        assert outcome.verdict is ConflictVerdict.INCONSISTENT

    def test_document_rung(self) -> None:
        inputs = RungInputs(new_supersedes_existing=True)
        assert _decide(probe=True, rung_inputs=inputs).verdict is (
            ConflictVerdict.DOCUMENT_SUPERSEDES
        )

    @pytest.mark.parametrize(
        ("order", "verdict"),
        [
            (1, ConflictVerdict.UPDATE_SUPERSEDES),
            (-1, ConflictVerdict.UPDATE_SUPERSEDED_BY_EXISTING),
        ],
    )
    def test_update_rung(self, order: int, verdict: ConflictVerdict) -> None:
        outcome = _decide(probe=True, rung_inputs=RungInputs(update_order=order))
        assert outcome == LadderOutcome(verdict=verdict, record_divergence=False)


class TestNeedsRungInputs:
    @pytest.mark.parametrize(
        ("probe", "fail_closed", "precondition", "expected"),
        [
            (True, False, RECONCILE, True),
            (False, False, RECONCILE, False),
            (None, False, RECONCILE, False),
            (None, True, RECONCILE, True),  # forced, but the domain hint is read
            (True, False, REVIEW, True),  # forced, but the domain hint is read
            (True, False, DECLINE, False),
            (None, True, DECLINE, False),
        ],
    )
    def test_gather_only_when_the_ladder_runs_on_a_signal(
        self,
        probe: bool | None,
        fail_closed: bool,
        precondition: PairPrecondition,
        expected: bool,
    ) -> None:
        assert (
            needs_rung_inputs(probe, fail_closed=fail_closed, precondition=precondition) is expected
        )


class TestForcesInconsistent:
    @pytest.mark.parametrize(
        ("probe", "fail_closed", "precondition", "expected"),
        [
            (None, True, RECONCILE, True),
            (None, False, RECONCILE, False),
            (True, False, REVIEW, True),
            (False, False, REVIEW, False),
            (None, False, REVIEW, False),
            (True, False, RECONCILE, False),
        ],
    )
    def test_the_two_overrides(
        self,
        probe: bool | None,
        fail_closed: bool,
        precondition: PairPrecondition,
        expected: bool,
    ) -> None:
        assert (
            forces_inconsistent(probe, fail_closed=fail_closed, precondition=precondition)
            is expected
        )


class TestUpdateOrderSource:
    def test_forced_consults_neither(self) -> None:
        assert update_order_source(allow_update=True, allow_own_assertion=True, forced=True) is None

    def test_extraction_door(self) -> None:
        assert (
            update_order_source(allow_update=True, allow_own_assertion=False, forced=False)
            is UpdateOrderSource.UPDATE
        )

    def test_update_door_wins_when_both_open(self) -> None:
        assert (
            update_order_source(allow_update=True, allow_own_assertion=True, forced=False)
            is UpdateOrderSource.UPDATE
        )

    def test_assertion_door(self) -> None:
        assert (
            update_order_source(allow_update=False, allow_own_assertion=True, forced=False)
            is UpdateOrderSource.OWN_ASSERTION
        )

    def test_no_door(self) -> None:
        assert (
            update_order_source(allow_update=False, allow_own_assertion=False, forced=False) is None
        )


class TestEffectiveSingleTrustOrder:
    @pytest.mark.parametrize(
        ("override", "store_mode", "expected"),
        [
            (None, "single", True),
            (None, "multi", False),
            (True, "multi", True),
            (False, "single", False),
        ],
    )
    def test_override_wins_else_store_mode(
        self, override: bool | None, store_mode: str, expected: bool
    ) -> None:
        assert effective_single_trust_order(override, store_mode) is expected
