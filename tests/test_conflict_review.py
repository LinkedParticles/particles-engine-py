# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure INCONSISTENCY-resolution decisions shared by Review and the trust cascade.

Lifted from ``particles/operations/review.py`` and
``particles/operations/cascade.py`` so each rule can be tested without a
database (D2). The I/O half, meaning the reads, the status writes and
their order, stays covered by ``tests/test_review.py`` and
``tests/test_cascade.py``.
"""

from __future__ import annotations

import pytest

from particles.core.conflict_resolution import RETIRED_VALUE_KEY
from particles.core.conflict_review import (
    AleatoryMark,
    Demotion,
    Retirement,
    TrustJudgment,
    cascade_pair,
    conflict_pair_ids,
    decide_cascade,
    decide_demotion,
    decide_resolution,
    decide_retirement,
    trust_statement_source,
)
from particles.core.schema import (
    SCHEMA_VERSION,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    ResolutionAction,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason

_CONF = Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT)


def _claim(
    status: Status = Status.ACTIVE,
    reason: StatusReason | None = None,
    *,
    entry: str | None = "entry-1",
) -> Particle:
    provenance = (
        [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry, snapshot_id="s")]
        if entry is not None
        else []
    )
    return Particle(
        content="a claim",
        confidence=_CONF,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=provenance,
        asserted_by="test",
        status=status,
        status_reason=reason,
        schema_version=SCHEMA_VERSION,
    )


def _quarantined(entry: str | None = "entry-b") -> Particle:
    return _claim(Status.PROVENANCE_STALE, StatusReason.CONFLICT_PENDING, entry=entry)


def _wrapper(a_id: str | None, b_id: str | None, *, retired_value: bool = False) -> Particle:
    refs = [
        ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=pid, snapshot_id=pid)
        for pid in (a_id, b_id)
        if pid is not None
    ]
    return Particle(
        content="INCONSISTENCY",
        confidence=_CONF,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=refs,
        asserted_by="extract-pipeline",
        status=Status.INCONSISTENCY,
        properties={RETIRED_VALUE_KEY: "retired"} if retired_value else None,
        schema_version=SCHEMA_VERSION,
    )


# -- conflict_pair_ids ------------------------------------------------------


def test_pair_ids_read_the_first_two_particle_refs_in_order() -> None:
    assert conflict_pair_ids(_wrapper("a", "b")) == ("a", "b")


def test_pair_ids_ignore_source_refs_and_read_a_missing_ref_as_none() -> None:
    inc = _wrapper("a", None)
    inc = inc.model_copy(
        update={
            "provenance": [
                ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e"),
                *inc.provenance,
            ]
        }
    )
    assert conflict_pair_ids(inc) == ("a", None)
    assert conflict_pair_ids(_wrapper(None, None)) == (None, None)


# -- decide_demotion --------------------------------------------------------


def test_demotion_of_a_live_claim_is_a_transition() -> None:
    assert decide_demotion(_claim()) is Demotion.TRANSITION


def test_demotion_of_a_quarantined_claim_flips_its_reason() -> None:
    """stale stays stale; only the reason moves."""
    assert decide_demotion(_quarantined()) is Demotion.REASON_FLIP


def test_demotion_of_an_already_stale_claim_writes_nothing() -> None:
    stale = _claim(Status.PROVENANCE_STALE, StatusReason.CONFLICT_RESOLVED)
    assert decide_demotion(stale) is Demotion.NONE


@pytest.mark.parametrize("status", [Status.RETRACTED, Status.SUPERSEDED])
def test_demotion_of_a_terminal_claim_writes_nothing(status: Status) -> None:
    """there is no legal transition out of a terminal state."""
    assert decide_demotion(_claim(status)) is Demotion.NONE


def test_demotion_of_a_dangling_ref_writes_nothing() -> None:
    assert decide_demotion(None) is Demotion.NONE


# -- trust_statement_source --------------------------------------


def test_trust_statement_is_keyed_on_the_corpus_entry_not_the_particle() -> None:
    """The bug: keying on the particle id made a statement no
    consumer could look up."""
    preferred = _claim(entry="entry-42")
    source = trust_statement_source(preferred)
    assert source == "entry-42"
    assert source != preferred.id


def test_trust_statement_uses_the_first_source_ref() -> None:
    preferred = _claim(entry="first")
    preferred = preferred.model_copy(
        update={
            "provenance": [
                ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id="p"),
                *preferred.provenance,
                ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="second"),
            ]
        }
    )
    assert trust_statement_source(preferred) == "first"


def test_no_trust_statement_for_an_unknown_preferred_claim() -> None:
    assert trust_statement_source(None) is None


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (Status.PROVENANCE_STALE, StatusReason.CONFLICT_PENDING),
        (Status.RETRACTED, None),
        (Status.SUPERSEDED, None),
    ],
)
def test_no_trust_statement_for_a_claim_that_is_not_active(
    status: Status, reason: StatusReason | None
) -> None:
    assert trust_statement_source(_claim(status, reason)) is None


def test_no_trust_statement_without_source_provenance() -> None:
    """An agent-asserted or derived claim names no corpus entry."""
    assert trust_statement_source(_claim(entry=None)) is None


# -- decide_resolution ------------------------------------------------------


def test_prefer_a_demotes_b_and_keys_the_statement_on_a() -> None:
    a, b = _claim(entry="entry-a"), _quarantined()
    plan = decide_resolution(ResolutionAction.PREFER_A, _wrapper(a.id, b.id), a, b)
    assert plan.demote == (b, Demotion.REASON_FLIP)
    assert plan.promote is None
    assert plan.aleatory == ()
    assert plan.trust == TrustJudgment("entry-a", a.id, b.id)
    assert plan.close_wrapper is True


def test_prefer_a_over_a_dangling_b_still_closes_the_wrapper() -> None:
    a = _claim(entry="entry-a")
    plan = decide_resolution(ResolutionAction.PREFER_A, _wrapper(a.id, "gone"), a, None)
    assert plan.demote is None
    assert plan.trust == TrustJudgment("entry-a", a.id, "gone")
    assert plan.close_wrapper is True


def test_prefer_b_promotes_a_quarantined_b_and_keys_on_its_source() -> None:
    """B is stale as stored, but the minted particle is ACTIVE with B's
    provenance, so the statement is still written."""
    a, b = _claim(entry="entry-a"), _quarantined(entry="entry-b")
    plan = decide_resolution(ResolutionAction.PREFER_B, _wrapper(a.id, b.id), a, b)
    assert plan.demote == (a, Demotion.TRANSITION)
    assert plan.promote is b
    assert plan.trust == TrustJudgment("entry-b", b.id, a.id)
    assert plan.close_wrapper is True


def test_prefer_b_over_an_active_b_promotes_nothing() -> None:
    a, b = _claim(entry="entry-a"), _claim(entry="entry-b")
    plan = decide_resolution(ResolutionAction.PREFER_B, _wrapper(a.id, b.id), a, b)
    assert plan.promote is None
    assert plan.trust == TrustJudgment("entry-b", b.id, a.id)


def test_prefer_b_over_a_dangling_b_writes_no_statement() -> None:
    a = _claim()
    plan = decide_resolution(ResolutionAction.PREFER_B, _wrapper(a.id, "gone"), a, None)
    assert plan.demote == (a, Demotion.TRANSITION)
    assert plan.promote is None
    assert plan.trust is None
    assert plan.close_wrapper is True


def test_prefer_b_over_an_unsourced_quarantined_b_writes_no_statement() -> None:
    a, b = _claim(), _quarantined(entry=None)
    plan = decide_resolution(ResolutionAction.PREFER_B, _wrapper(a.id, b.id), a, b)
    assert plan.promote is b
    assert plan.trust is None


@pytest.mark.parametrize("action", [ResolutionAction.PREFER_A, ResolutionAction.PREFER_B])
def test_a_retired_value_record_writes_no_trust_statement(action: ResolutionAction) -> None:
    """the ruling is on an earlier judgment, not on a source."""
    a, b = _claim(Status.RETRACTED), _quarantined()
    plan = decide_resolution(action, _wrapper(a.id, b.id, retired_value=True), a, b)
    assert plan.trust is None
    assert plan.close_wrapper is True
    if action is ResolutionAction.PREFER_A:
        assert plan.demote == (b, Demotion.REASON_FLIP)
    else:
        # Lifting the retirement: A is terminal, so nothing to demote, and B
        # is promoted.
        assert plan.demote == (a, Demotion.NONE)
        assert plan.promote is b


def test_both_valid_marks_each_known_claim_in_a_b_order() -> None:
    a, b = _claim(), _quarantined()
    plan = decide_resolution(ResolutionAction.BOTH_VALID, _wrapper(a.id, b.id), a, b)
    assert plan.aleatory == (AleatoryMark(a, mint=False), AleatoryMark(b, mint=True))
    assert plan.demote is None
    assert plan.promote is None
    assert plan.trust is None
    assert plan.close_wrapper is True


def test_both_valid_skips_a_dangling_ref() -> None:
    a = _claim()
    plan = decide_resolution(ResolutionAction.BOTH_VALID, _wrapper(a.id, "gone"), a, None)
    assert plan.aleatory == (AleatoryMark(a, mint=False),)


def test_defer_writes_nothing_and_leaves_the_wrapper_open() -> None:
    a, b = _claim(), _quarantined()
    plan = decide_resolution(ResolutionAction.DEFER, _wrapper(a.id, b.id), a, b)
    assert plan.demote is None
    assert plan.promote is None
    assert plan.aleatory == ()
    assert plan.trust is None
    assert plan.close_wrapper is False


# -- DISCARD ------------------------------------------------------


def test_retirement_of_a_live_claim_is_a_believed_retraction() -> None:
    a = _claim()
    assert decide_retirement(a) == Retirement(a, was_believed=True)


def test_retirement_of_a_quarantined_claim_was_never_believed() -> None:
    b = _quarantined()
    assert decide_retirement(b) == Retirement(b, was_believed=False)


@pytest.mark.parametrize(
    "reason",
    [StatusReason.CORPUS_ENTRY_MISSING, StatusReason.LOWER_TRUST_SOURCE, None],
)
def test_retirement_of_a_stale_claim_retracts_it(reason: StatusReason | None) -> None:
    """A stale side is not terminal, so DISCARD closes it with the rest."""
    stale = _claim(Status.PROVENANCE_STALE, reason)
    assert decide_retirement(stale) == Retirement(stale, was_believed=True)


@pytest.mark.parametrize("status", [Status.RETRACTED, Status.SUPERSEDED])
def test_retirement_of_a_terminal_claim_writes_nothing(status: Status) -> None:
    assert decide_retirement(_claim(status)) is None


def test_retirement_of_a_dangling_ref_writes_nothing() -> None:
    assert decide_retirement(None) is None


def test_discard_retires_both_sides_with_no_verdict() -> None:
    a, b = _claim(), _quarantined()
    plan = decide_resolution(ResolutionAction.DISCARD, _wrapper(a.id, b.id), a, b)
    assert plan.retire == (Retirement(a, was_believed=True), Retirement(b, was_believed=False))
    assert plan.demote is None
    assert plan.promote is None
    assert plan.aleatory == ()
    assert plan.trust is None
    assert plan.close_wrapper is True


def test_discard_over_a_dangling_b_retires_a_alone() -> None:
    a = _claim()
    plan = decide_resolution(ResolutionAction.DISCARD, _wrapper(a.id, "gone"), a, None)
    assert plan.retire == (Retirement(a, was_believed=True),)
    assert plan.close_wrapper is True


def test_discard_of_a_retired_value_record_leaves_the_judged_original() -> None:
    """the terminal original keeps its judgment reason; only the
    held re-assertion goes."""
    a, b = _claim(Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION), _quarantined()
    plan = decide_resolution(
        ResolutionAction.DISCARD, _wrapper(a.id, b.id, retired_value=True), a, b
    )
    assert plan.retire == (Retirement(b, was_believed=False),)
    assert plan.trust is None


@pytest.mark.parametrize(
    "action", [a for a in ResolutionAction if a is not ResolutionAction.DISCARD]
)
def test_only_discard_retires(action: ResolutionAction) -> None:
    a, b = _claim(), _quarantined()
    assert decide_resolution(action, _wrapper(a.id, b.id), a, b).retire == ()


# -- cascade_pair / decide_cascade -------------------------------------------


def test_cascade_pair_leaves_a_dangling_ref_for_a_person() -> None:
    a = _claim()
    assert cascade_pair(_wrapper(a.id, "gone"), a, None) is None
    assert cascade_pair(_wrapper("gone", a.id), None, a) is None


def test_cascade_pair_leaves_a_retired_value_record_for_a_person() -> None:
    a, b = _claim(), _quarantined()
    assert cascade_pair(_wrapper(a.id, b.id, retired_value=True), a, b) is None


@pytest.mark.parametrize("status", [Status.RETRACTED, Status.SUPERSEDED])
def test_cascade_pair_leaves_a_terminal_a_for_a_person(status: Status) -> None:
    a, b = _claim(status), _quarantined()
    assert cascade_pair(_wrapper(a.id, b.id), a, b) is None


def test_cascade_pair_returns_the_claims_to_rank() -> None:
    a, b = _claim(), _quarantined()
    assert cascade_pair(_wrapper(a.id, b.id), a, b) == (a, b)


def test_cascade_a_outranks_b_demotes_quarantined_b() -> None:
    a, b = _claim(), _quarantined()
    verdict = decide_cascade(_wrapper(a.id, b.id), a, b, 0.9, 0.2, differential_threshold=0.3)
    assert verdict is not None
    assert verdict.winner is a
    assert verdict.loser is b
    assert verdict.loser_demotion is Demotion.REASON_FLIP
    assert verdict.promote_winner is False


def test_cascade_b_outranks_a_promotes_quarantined_b() -> None:
    """The cascade resolves for the quarantined claim as PREFER_B would."""
    a, b = _claim(), _quarantined()
    verdict = decide_cascade(_wrapper(a.id, b.id), a, b, 0.2, 0.9, differential_threshold=0.3)
    assert verdict is not None
    assert verdict.winner is b
    assert verdict.loser is a
    assert verdict.loser_demotion is Demotion.TRANSITION
    assert verdict.promote_winner is True


def test_cascade_below_the_differential_threshold_stays_open() -> None:
    a, b = _claim(), _quarantined()
    assert decide_cascade(_wrapper(a.id, b.id), a, b, 0.6, 0.4, differential_threshold=0.3) is None


def test_cascade_threshold_boundary_is_inclusive() -> None:
    a, b = _claim(), _quarantined()
    verdict = decide_cascade(_wrapper(a.id, b.id), a, b, 0.75, 0.25, differential_threshold=0.5)
    assert verdict is not None
    assert verdict.winner is a


def test_cascade_exact_tie_at_zero_threshold_goes_to_b() -> None:
    a, b = _claim(), _claim()
    verdict = decide_cascade(_wrapper(a.id, b.id), a, b, 0.5, 0.5, differential_threshold=0.0)
    assert verdict is not None
    assert verdict.winner is b


@pytest.mark.parametrize(("rank_a", "rank_b"), [(None, 0.9), (0.9, None), (None, None)])
def test_cascade_with_an_unknown_rank_stays_open(
    rank_a: float | None, rank_b: float | None
) -> None:
    a, b = _claim(), _quarantined()
    verdict = decide_cascade(_wrapper(a.id, b.id), a, b, rank_a, rank_b, differential_threshold=0.3)
    assert verdict is None


def test_decide_cascade_applies_the_structural_check_itself() -> None:
    """A caller that skips `cascade_pair` still gets the full decision."""
    a, b = _claim(Status.RETRACTED), _quarantined()
    assert decide_cascade(_wrapper(a.id, b.id), a, b, 0.9, 0.1, differential_threshold=0.3) is None


def test_cascade_uses_the_shared_demotion_rule_for_a_terminal_loser() -> None:
    """A terminal B that loses needs no write, as in review."""
    a, b = _claim(), _claim(Status.RETRACTED)
    verdict = decide_cascade(_wrapper(a.id, b.id), a, b, 0.9, 0.1, differential_threshold=0.3)
    assert verdict is not None
    assert verdict.loser_demotion is Demotion.NONE


def _census(a: Particle, b: Particle, *extra_a: Particle) -> Particle:
    """A census record with ``a`` and any ``extra_a`` on side a, ``b`` on side b."""
    from particles.core.contradiction_disclosure import (
        ConfirmedPair,
        NoteLabel,
        Sides,
        build_census_record,
    )

    members = {p.id: p for p in (a, b, *extra_a)}
    return build_census_record(
        sides=Sides(a=(a.id, *(p.id for p in extra_a)), b=(b.id,)),
        pairs=[ConfirmedPair(a=p.id, b=b.id, same_source=False, reason="r") for p in (a, *extra_a)],
        members=members,
        labels={pid: NoteLabel("n", "d") for pid in members},
        sources={pid: ["e"] for pid in members},
        trigger_entry_id="e",
        trigger_snapshot_id=None,
    )


def test_cascade_leaves_a_census_record_of_more_than_two_claims_for_a_person() -> None:
    """a side drawn from several notes has no one trust rank to compare."""
    a, a2, b = _claim(entry="entry-a"), _claim(entry="entry-c"), _claim(entry="entry-b")
    assert cascade_pair(_census(a, b), a, b) == (a, b)
    assert cascade_pair(_census(a, b, a2), a, b) is None


def test_prefer_a_on_a_census_record_demotes_the_whole_other_side() -> None:
    from particles.core.conflict_review import FurtherMembers

    a, a2, b, b2 = (
        _claim(entry="entry-a"),
        _claim(entry="entry-c"),
        _claim(entry="entry-b"),
        _claim(entry="entry-d"),
    )
    record = _census(a, b, a2)
    plan = decide_resolution(
        ResolutionAction.PREFER_A, record, a, b, FurtherMembers(a=(a2,), b=(b2,))
    )
    assert plan.demote == (b, Demotion.TRANSITION)
    assert plan.also_demote == ((b2, Demotion.TRANSITION),)
    # A two-claim plan is unchanged: no further demotion.
    assert decide_resolution(ResolutionAction.PREFER_A, record, a, b).also_demote == ()
