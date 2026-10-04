# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for the §6.6 pair selection (``ingest/pair_selection.py``).

``plan_probes``, ``plan_update_probes`` and ``select_pairs`` decide, with no DB
and no model, which existing claims a candidate is paired with and which the
update probe is asked about; ``plan_update_extras`` decides which
rung 2.5 extras are demoted. The DB-level behaviour stays covered by
``tests/test_extract.py``, ``tests/test_update_supersession.py`` and
``tests/test_observer_gate.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from particles.core.conflict_resolution import SlotVerdict
from particles.core.observer_scope import PairPrecondition
from particles.core.schema import (
    Confidence,
    CorpusEntry,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.ingest.pair_selection import (
    CandidatePairs,
    PairRole,
    PairSelection,
    plan_probes,
    plan_update_extras,
    plan_update_probes,
    select_pairs,
)

T0 = datetime(2026, 6, 1, tzinfo=UTC)
DECLINE = PairPrecondition.DECLINE
REVIEW = PairPrecondition.REVIEW


def _p(content: str, *, entry: CorpusEntry | None = None) -> Particle:
    provenance = (
        [
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry.entry_id,
                snapshot_id=entry.snapshots[0].snapshot_id,
            )
        ]
        if entry is not None
        else []
    )
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        asserted_at=T0,
        subject_ids=["u"],
        provenance=provenance,
    )


NEAREST, A, B, C = _p("nearest"), _p("a"), _p("b"), _p("c")


def _probed(pairs: CandidatePairs) -> list[tuple[str, PairRole]]:
    return [(p.content, role) for p, role in plan_probes(pairs)]


# ---------------------------------------------------------------------------
# plan_probes
# ---------------------------------------------------------------------------


class TestPlanProbes:
    def test_nothing_to_pair_probes_nothing(self) -> None:
        assert plan_probes(CandidatePairs()) == []

    def test_nearest_first_then_pool_in_order(self) -> None:
        pairs = CandidatePairs(nearest=NEAREST, subject_pool=(A, B))
        assert _probed(pairs) == [
            ("nearest", PairRole.NEAREST),
            ("a", PairRole.SUBJECT),
            ("b", PairRole.SUBJECT),
        ]

    def test_declined_nearest_is_still_probed(self) -> None:
        # Recorded as a divergence only if the probe confirms it.
        pairs = CandidatePairs(nearest=NEAREST, precondition={NEAREST.id: DECLINE})
        assert _probed(pairs) == [("nearest", PairRole.NEAREST)]

    def test_subject_pool_roles(self) -> None:
        pairs = CandidatePairs(
            subject_pool=(A, B, C),
            precondition={A.id: DECLINE, C.id: REVIEW},
        )
        # REVIEW is not a pre-probe decline: the ladder forces INCONSISTENT later.
        assert _probed(pairs) == [
            ("a", PairRole.DECLINED),
            ("b", PairRole.SUBJECT),
            ("c", PairRole.SUBJECT),
        ]

    def test_multi_regime_skips_unupdatable_members_without_probing(self) -> None:
        # a pair rung 2.5 cannot act on is never probed.
        pairs = CandidatePairs(
            subject_pool=(A, B, C),
            precondition={C.id: DECLINE},
            can_update={A.id: False, B.id: True},
        )
        assert _probed(pairs) == [("b", PairRole.SUBJECT), ("c", PairRole.DECLINED)]

    def test_multi_regime_member_missing_from_the_facts_is_not_offered(self) -> None:
        pairs = CandidatePairs(subject_pool=(A,), can_update={})
        assert plan_probes(pairs) == []

    def test_single_regime_offers_every_member(self) -> None:
        pairs = CandidatePairs(subject_pool=(A, B), can_update=None)
        assert [role for _, role in plan_probes(pairs)] == [PairRole.SUBJECT, PairRole.SUBJECT]


# ---------------------------------------------------------------------------
# select_pairs
# ---------------------------------------------------------------------------


def _ids(particles: tuple[Particle, ...]) -> list[str]:
    return [p.content for p in particles]


class TestSelectPairs:
    def test_nothing_to_pair_selects_nothing(self) -> None:
        assert select_pairs(CandidatePairs(), {}) == PairSelection()

    def test_confirmed_nearest_is_primary(self) -> None:
        pairs = CandidatePairs(nearest=NEAREST, subject_pool=(A,))
        sel = select_pairs(pairs, {NEAREST.id: True, A.id: True})
        assert sel.primary is NEAREST and sel.signal
        assert _ids(sel.extras) == ["a"]
        assert sel.declined == ()

    def test_unconfirmed_nearest_stays_primary_without_a_confirmed_pool(self) -> None:
        # The ladder corroborates it.
        pairs = CandidatePairs(nearest=NEAREST, subject_pool=(A,))
        sel = select_pairs(pairs, {NEAREST.id: False, A.id: False})
        assert sel == PairSelection(primary=NEAREST, signal=False)

    def test_declined_nearest_confirmed_is_recorded_not_offered(self) -> None:
        pairs = CandidatePairs(nearest=NEAREST, precondition={NEAREST.id: DECLINE})
        sel = select_pairs(pairs, {NEAREST.id: True})
        assert sel.primary is None and not sel.signal
        assert _ids(sel.declined) == ["nearest"]

    def test_declined_nearest_unconfirmed_stays_primary(self) -> None:
        # The precondition is asked only of a confirmed pair; the ladder's own
        # DECLINE then writes the candidate ACTIVE beside it.
        pairs = CandidatePairs(nearest=NEAREST, precondition={NEAREST.id: DECLINE})
        sel = select_pairs(pairs, {NEAREST.id: False})
        assert sel == PairSelection(primary=NEAREST, signal=False)

    def test_best_confirmed_pool_member_is_promoted_when_nearest_unconfirmed(self) -> None:
        pairs = CandidatePairs(nearest=NEAREST, subject_pool=(A, B, C))
        sel = select_pairs(pairs, {NEAREST.id: False, A.id: False, B.id: True, C.id: True})
        assert sel.primary is B and sel.signal
        assert _ids(sel.extras) == ["c"]

    def test_promotion_when_nearest_absent(self) -> None:
        pairs = CandidatePairs(subject_pool=(A, B))
        sel = select_pairs(pairs, {A.id: True, B.id: True})
        assert sel.primary is A and sel.signal
        assert _ids(sel.extras) == ["b"]

    def test_promotion_when_nearest_declined(self) -> None:
        pairs = CandidatePairs(
            nearest=NEAREST, subject_pool=(A,), precondition={NEAREST.id: DECLINE}
        )
        sel = select_pairs(pairs, {NEAREST.id: True, A.id: True})
        assert sel.primary is A and sel.signal
        assert _ids(sel.declined) == ["nearest"]
        assert sel.extras == ()

    def test_extras_keep_pool_order(self) -> None:
        pairs = CandidatePairs(nearest=NEAREST, subject_pool=(C, A, B))
        sel = select_pairs(pairs, {NEAREST.id: True, C.id: True, A.id: True, B.id: True})
        assert _ids(sel.extras) == ["c", "a", "b"]

    def test_declined_pool_members_recorded_only_when_confirmed(self) -> None:
        pairs = CandidatePairs(subject_pool=(A, B), precondition={A.id: DECLINE, B.id: DECLINE})
        sel = select_pairs(pairs, {A.id: True, B.id: False})
        assert sel.primary is None
        assert _ids(sel.declined) == ["a"]

    def test_declined_order_is_nearest_then_pool(self) -> None:
        pairs = CandidatePairs(
            nearest=NEAREST,
            subject_pool=(A,),
            precondition={NEAREST.id: DECLINE, A.id: DECLINE},
        )
        sel = select_pairs(pairs, {NEAREST.id: True, A.id: True})
        assert _ids(sel.declined) == ["nearest", "a"]

    def test_multi_regime_unupdatable_member_is_never_selected(self) -> None:
        pairs = CandidatePairs(subject_pool=(A, B), can_update={A.id: False, B.id: True})
        # Even a stray probe result for A cannot offer it.
        sel = select_pairs(pairs, {A.id: True, B.id: True})
        assert sel.primary is B and sel.extras == ()

    def test_review_member_is_offered(self) -> None:
        pairs = CandidatePairs(subject_pool=(A,), precondition={A.id: REVIEW})
        sel = select_pairs(pairs, {A.id: True})
        assert sel.primary is A and sel.signal

    def test_reconcile_and_insert_policy_one_pair_always_primary(self) -> None:
        # The assertion path: the store-wide nearest claim, empty pool, no
        # precondition pre-screen, ``can_update=None``.
        pairs = CandidatePairs(nearest=NEAREST)
        assert plan_probes(pairs) == [(NEAREST, PairRole.NEAREST)]
        for confirmed in (True, False):
            sel = select_pairs(pairs, {NEAREST.id: confirmed})
            assert sel == PairSelection(primary=NEAREST, signal=confirmed)


# ---------------------------------------------------------------------------
# The slot rule: plan_update_probes, and select_pairs with slot verdicts
# ---------------------------------------------------------------------------


CHANGES, FIXED, DIFFERENT = SlotVerdict.CHANGES, SlotVerdict.FIXED, SlotVerdict.DIFFERENT


class TestSlotRule:
    def test_only_confirmed_actionable_pairs_are_asked(self) -> None:
        pairs = CandidatePairs(
            nearest=NEAREST,
            subject_pool=(A, B, C),
            precondition={B.id: DECLINE},
        )
        probes = {NEAREST.id: True, A.id: False, B.id: True, C.id: True}
        # A is unconfirmed, B is declined (recorded, never retired): neither is asked.
        assert plan_update_probes(pairs, probes) == [NEAREST, C]

    def test_declined_nearest_is_not_asked(self) -> None:
        pairs = CandidatePairs(nearest=NEAREST, precondition={NEAREST.id: DECLINE})
        assert plan_update_probes(pairs, {NEAREST.id: True}) == []

    def test_multi_regime_unupdatable_member_is_not_asked(self) -> None:
        pairs = CandidatePairs(subject_pool=(A, B), can_update={A.id: False, B.id: True})
        assert plan_update_probes(pairs, {A.id: True, B.id: True}) == [B]

    def test_a_different_slot_pool_member_is_not_offered(self) -> None:
        # The Delhi scenario: the contradiction probe confirmed the pair, the
        # update probe says the claims fill different slots. Both stay ACTIVE.
        pairs = CandidatePairs(subject_pool=(A,))
        sel = select_pairs(pairs, {A.id: True}, {A.id: DIFFERENT})
        assert sel.primary is None and not sel.signal and sel.extras == ()

    def test_the_next_same_slot_member_is_promoted(self) -> None:
        pairs = CandidatePairs(subject_pool=(A, B, C))
        sel = select_pairs(
            pairs,
            {A.id: True, B.id: True, C.id: True},
            {A.id: DIFFERENT, B.id: CHANGES, C.id: CHANGES},
        )
        assert sel.primary is B and sel.signal
        assert _ids(sel.extras) == ["c"]
        assert sel.updatable == frozenset({B.id, C.id})

    def test_a_different_slot_nearest_stays_primary_without_the_rung(self) -> None:
        # The same-entry pair reached the ladder before; the verdict
        # only keeps rung 2.5 off it, so it falls through to rung 3.
        pairs = CandidatePairs(nearest=NEAREST)
        sel = select_pairs(pairs, {NEAREST.id: True}, {NEAREST.id: DIFFERENT})
        assert sel.primary is NEAREST and sel.signal
        assert NEAREST.id not in sel.updatable

    def test_an_unasked_member_is_offered_as_before(self) -> None:
        # No verdict means rung 2.5 could not order the pair, so the ladder
        # treats it as it always has.
        pairs = CandidatePairs(subject_pool=(A,))
        sel = select_pairs(pairs, {A.id: True}, {})
        assert sel.primary is A and sel.signal and sel.updatable == frozenset()
        assert sel.slot_of(A.id) is None

    def test_a_fixed_slot_pool_member_is_offered_for_review(self) -> None:
        # "The Poison Tree was written by Erin Kelly" against "…by
        # Barbara Hambly". The pair fills one slot, so unlike a different-slot
        # pair it reaches the ladder, but the slot is fixed, so it is not
        # updatable: the ladder falls through to rung 3 (review).
        pairs = CandidatePairs(subject_pool=(A,))
        sel = select_pairs(pairs, {A.id: True}, {A.id: FIXED})
        assert sel.primary is A and sel.signal
        assert sel.slot_of(A.id) is FIXED
        assert sel.updatable == frozenset()

    def test_a_fixed_slot_extra_is_not_updatable(self) -> None:
        pairs = CandidatePairs(subject_pool=(A, B))
        sel = select_pairs(pairs, {A.id: True, B.id: True}, {A.id: CHANGES, B.id: FIXED})
        assert sel.primary is A and _ids(sel.extras) == ["b"]
        assert sel.updatable == frozenset({A.id})


# ---------------------------------------------------------------------------
# plan_update_extras
# ---------------------------------------------------------------------------


def _entry(published: datetime | None, uri: str = "claude-code://session/a") -> CorpusEntry:
    return CorpusEntry(
        uri_r=uri,
        source_type="CONVERSATION",
        snapshots=[Snapshot(content_hash="0" * 64, content_published_at=published)],
        deposited_by="test",
    )


WINNER_ENTRY = _entry(T0 + timedelta(days=10), uri="claude-code://session/w")


def _winner() -> Particle:
    return _p("winner", entry=WINNER_ENTRY)


def _plan(
    winner: Particle,
    others: list[Particle],
    entries: list[CorpusEntry],
    precondition: dict[str, PairPrecondition] | None = None,
    updatable: set[str] | None = None,
) -> list[str]:
    return plan_update_extras(
        winner,
        WINNER_ENTRY,
        WINNER_ENTRY.snapshots[0].snapshot_id,
        others,
        {e.entry_id: e for e in entries},
        precondition or {},
        {o.id for o in others} if updatable is None else updatable,
    )


class TestPlanUpdateExtras:
    def test_an_extra_without_a_same_slot_verdict_is_left_alone(self) -> None:
        e1, e2 = _entry(T0), _entry(T0 + timedelta(days=1))
        winner, o1, o2 = _winner(), _p("o1", entry=e1), _p("o2", entry=e2)
        assert _plan(winner, [o1, o2], [e1, e2], updatable={o2.id}) == [o2.id]

    def test_older_extras_are_demoted_in_order(self) -> None:
        e1, e2 = _entry(T0), _entry(T0 + timedelta(days=1))
        winner, o1, o2 = _winner(), _p("o1", entry=e1), _p("o2", entry=e2)
        assert _plan(winner, [o1, o2], [e1, e2]) == [o1.id, o2.id]

    def test_run_stops_at_the_winner(self) -> None:
        older, newer, after = (
            _entry(T0),
            _entry(T0 + timedelta(days=20)),
            _entry(T0 + timedelta(days=1)),
        )
        winner = _winner()
        o1, o2, o3 = _p("o1", entry=older), _p("o2", entry=newer), _p("o3", entry=after)
        # o2 is newer than the winner: the winner is retired and o3 is never reached.
        assert _plan(winner, [o1, o2, o3], [older, newer, after]) == [o1.id, winner.id]

    def test_unqualified_pairs_are_left_alone(self) -> None:
        dated, undated, other_lineage = (
            _entry(T0),
            _entry(None),
            _entry(T0, uri="https://example.org/page"),
        )
        winner = _winner()
        no_ref = _p("no ref")
        missing_entry = _p("missing", entry=_entry(T0))
        undated_p = _p("undated", entry=undated)
        foreign = _p("foreign", entry=other_lineage)
        ok = _p("ok", entry=dated)
        others = [no_ref, missing_entry, undated_p, foreign, ok]
        assert _plan(winner, others, [dated, undated, other_lineage]) == [ok.id]

    def test_only_reconcile_precondition_qualifies(self) -> None:
        e1, e2, e3 = _entry(T0), _entry(T0), _entry(T0)
        winner = _winner()
        declined, review, ok = _p("d", entry=e1), _p("r", entry=e2), _p("ok", entry=e3)
        precondition = {declined.id: DECLINE, review.id: REVIEW}
        assert _plan(winner, [declined, review, ok], [e1, e2, e3], precondition) == [ok.id]

    def test_no_extras_demotes_nothing(self) -> None:
        assert _plan(_winner(), [], []) == []
