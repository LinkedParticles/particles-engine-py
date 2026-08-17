"""Tests for ``particles/core/conflict_resolution.py`` — pure §6.6 ladder.

These tests exercise the decision logic and the INCONSISTENCY-particle
constructor without touching a DB or the LLM. The pipeline glue layer that
turns each verdict into DB writes (``_resolve_conflict`` in
``particles/ingest/pipeline.py``) is tested end-to-end through
``extract_snapshot`` by ``TestConflictWritePath`` in ``tests/test_extract.py``
— demote-existing, INCONSISTENCY insertion + ``domain_hint``, the silent drop,
and consensus-mode suppression.
"""

from __future__ import annotations

import pytest

from particles.core.conflict_resolution import (
    ConflictVerdict,
    build_inconsistency_particle,
    resolve_conflict,
)
from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status


def _particle(
    *,
    content: str = "claim",
    confidence: float = 0.8,
    uncertainty_nature: UncertaintyNature = UncertaintyNature.EPISTEMIC,
    subject_ids: list[str] | None = None,
    asserted_by: str = "test",
    assertion_modality: AssertionModality = AssertionModality.FALSIFIABLE,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        ),
        uncertainty_nature=uncertainty_nature,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="entry-1",
                snapshot_id="snap-1",
            ),
        ],
        asserted_by=asserted_by,
        subject_ids=subject_ids or [],
        assertion_modality=assertion_modality,
    )


# ---------------------------------------------------------------------------
# resolve_conflict() — the pure ladder
# ---------------------------------------------------------------------------


class TestResolveConflictTruthAptGate:
    """Step -1 — non-truth-apt particles never contradict."""

    def test_non_falsifiable_new_corroborates(self) -> None:
        existing = _particle()
        new = _particle(assertion_modality=AssertionModality.EVALUATIVE)
        # Even with a positive contradiction signal and a trust gap that would
        # otherwise auto-supersede, an opinion cannot contradict a fact.
        assert (
            resolve_conflict(
                existing,
                new,
                has_contradiction_signal=True,
                trust_score_existing=0.1,
                trust_score_new=0.99,
            )
            is ConflictVerdict.CORROBORATES
        )

    def test_non_falsifiable_existing_corroborates(self) -> None:
        existing = _particle(assertion_modality=AssertionModality.EXPERIENTIAL)
        new = _particle()
        assert (
            resolve_conflict(existing, new, has_contradiction_signal=True)
            is ConflictVerdict.CORROBORATES
        )

    def test_two_non_falsifiable_corroborate(self) -> None:
        existing = _particle(assertion_modality=AssertionModality.CONSTITUTIVE)
        new = _particle(assertion_modality=AssertionModality.CONSTITUTIVE)
        assert (
            resolve_conflict(existing, new, has_contradiction_signal=True)
            is ConflictVerdict.CORROBORATES
        )

    def test_two_falsifiable_still_reach_inconsistent(self) -> None:
        # Control: the gate does not perturb the normal falsifiable path.
        existing = _particle()
        new = _particle()
        assert (
            resolve_conflict(existing, new, has_contradiction_signal=True)
            is ConflictVerdict.INCONSISTENT
        )


class TestResolveConflictGate:
    """Step 0 — caller's contradiction-signal probe takes precedence."""

    def test_no_signal_returns_corroborates(self) -> None:
        existing = _particle(content="A says X")
        new = _particle(content="B quotes the claim that X")
        assert (
            resolve_conflict(existing, new, has_contradiction_signal=False)
            is ConflictVerdict.CORROBORATES
        )

    def test_no_signal_overrides_trust_and_aleatory(self) -> None:
        existing = _particle(uncertainty_nature=UncertaintyNature.ALEATORY)
        new = _particle()
        assert (
            resolve_conflict(
                existing,
                new,
                has_contradiction_signal=False,
                trust_score_existing=0.1,
                trust_score_new=0.9,
            )
            is ConflictVerdict.CORROBORATES
        )


class TestResolveConflictAleatory:
    """Step 1 — ALEATORY on either side forces INCONSISTENT."""

    def test_aleatory_existing_skips_trust(self) -> None:
        existing = _particle(uncertainty_nature=UncertaintyNature.ALEATORY)
        new = _particle()
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            trust_score_existing=0.1,
            trust_score_new=0.99,  # large enough gap to otherwise SUPERSEDES
        )
        assert verdict is ConflictVerdict.INCONSISTENT

    def test_aleatory_new_skips_trust(self) -> None:
        existing = _particle()
        new = _particle(uncertainty_nature=UncertaintyNature.ALEATORY)
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            trust_score_existing=0.99,
            trust_score_new=0.1,
        )
        assert verdict is ConflictVerdict.INCONSISTENT

    def test_both_aleatory_inconsistent(self) -> None:
        existing = _particle(uncertainty_nature=UncertaintyNature.ALEATORY)
        new = _particle(uncertainty_nature=UncertaintyNature.ALEATORY)
        assert (
            resolve_conflict(existing, new, has_contradiction_signal=True)
            is ConflictVerdict.INCONSISTENT
        )


class TestResolveConflictDocumentSupersession:
    """Step 1.5 — document-supersession prior (cap. 2)."""

    def test_new_supersedes_existing(self) -> None:
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=True,
            new_supersedes_existing=True,
        )
        assert verdict is ConflictVerdict.DOCUMENT_SUPERSEDES

    def test_existing_supersedes_new(self) -> None:
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=True,
            existing_supersedes_new=True,
        )
        assert verdict is ConflictVerdict.DOCUMENT_SUPERSEDED_BY_EXISTING

    def test_supersession_outranks_trust(self) -> None:
        # Rung 1.5 sits ABOVE the trust rung: an authored "this replaces that"
        # wins even when the trust differential would prefer the other side.
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=True,
            new_supersedes_existing=True,
            trust_score_existing=0.99,  # trust would otherwise drop `new`
            trust_score_new=0.1,
            trust_differential_threshold=0.15,
        )
        assert verdict is ConflictVerdict.DOCUMENT_SUPERSEDES

    def test_aleatory_outranks_supersession(self) -> None:
        # Rung 1 (ALEATORY) is above rung 1.5: irreducible disagreement is never
        # resolved by an editorial relation.
        verdict = resolve_conflict(
            _particle(uncertainty_nature=UncertaintyNature.ALEATORY),
            _particle(),
            has_contradiction_signal=True,
            new_supersedes_existing=True,
        )
        assert verdict is ConflictVerdict.INCONSISTENT

    def test_no_signal_overrides_supersession(self) -> None:
        # Step 0 gate still wins: no contradiction → CORROBORATES, prior unused.
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=False,
            new_supersedes_existing=True,
        )
        assert verdict is ConflictVerdict.CORROBORATES

    def test_mutual_supersession_falls_through_to_trust(self) -> None:
        # A supersession cycle fires neither branch; the pair falls through to
        # the trust rung (here a gap resolves it, proving the fall-through).
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=True,
            new_supersedes_existing=True,
            existing_supersedes_new=True,
            trust_score_existing=0.5,
            trust_score_new=0.8,
            trust_differential_threshold=0.15,
        )
        assert verdict is ConflictVerdict.SUPERSEDES

    def test_gated_by_single_trust_order(self) -> None:
        # v1 gates rung 1.5 to single-trust-order stores (like rung 2). In a
        # consensus store the prior is suppressed and the pair falls to
        # INCONSISTENT — disagreement is surfaced, never resolved away.
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=True,
            new_supersedes_existing=True,
            single_trust_order=False,
        )
        assert verdict is ConflictVerdict.INCONSISTENT

    def test_no_relation_unaffected(self) -> None:
        # Control: with neither direction set, rung 1.5 is a no-op and the
        # trust rung resolves the pair exactly as before.
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=True,
            trust_score_existing=0.5,
            trust_score_new=0.8,
            trust_differential_threshold=0.15,
        )
        assert verdict is ConflictVerdict.SUPERSEDES


class TestResolveConflictModalityIndependentSupersession:
    """the document-supersession prior runs ABOVE the truth-apt gate.

    The binding defect fixed here: under cap. 2 the step -1
    truth-apt gate returned CORROBORATES for any non-truth-apt pair *before* the
    supersession branch could fire, so a superseded ``CONSTITUTIVE`` definition
    (exactly what supersession should retire) was invisible to rung 1.5. The fix
    lifts the branch above the gate and makes it modality-independent.
    """

    def test_constitutive_superseded_is_demoted(self) -> None:
        # The flagship: a superseded CONSTITUTIVE definition vs the current one.
        # this returned CORROBORATES (truth-apt gate first); under
        # the authored supersession edge + replacement signal retires it.
        existing = _particle(assertion_modality=AssertionModality.CONSTITUTIVE)
        new = _particle(assertion_modality=AssertionModality.FALSIFIABLE)
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            new_supersedes_existing=True,
        )
        assert verdict is ConflictVerdict.DOCUMENT_SUPERSEDES

    def test_both_constitutive_superseded_is_demoted(self) -> None:
        existing = _particle(assertion_modality=AssertionModality.CONSTITUTIVE)
        new = _particle(assertion_modality=AssertionModality.CONSTITUTIVE)
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            new_supersedes_existing=True,
        )
        assert verdict is ConflictVerdict.DOCUMENT_SUPERSEDES

    def test_existing_supersedes_constitutive_new(self) -> None:
        existing = _particle(assertion_modality=AssertionModality.FALSIFIABLE)
        new = _particle(assertion_modality=AssertionModality.CONSTITUTIVE)
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            existing_supersedes_new=True,
        )
        assert verdict is ConflictVerdict.DOCUMENT_SUPERSEDED_BY_EXISTING

    def test_non_truthapt_no_edge_still_corroborates(self) -> None:
        # Regression guard: the change narrows the truth-apt gate's scope but does
        # not remove it. Without a supersession edge a non-truth-apt pair still
        # corroborates (the gate fires below the inert supersession branch).
        existing = _particle(assertion_modality=AssertionModality.CONSTITUTIVE)
        new = _particle(assertion_modality=AssertionModality.CONSTITUTIVE)
        verdict = resolve_conflict(existing, new, has_contradiction_signal=True)
        assert verdict is ConflictVerdict.CORROBORATES

    def test_non_truthapt_edge_no_signal_keeps_both(self) -> None:
        # Default-safe direction: no replacement signal → keep both,
        # preserving the cap. 2(c) never-blanket-demote invariant.
        existing = _particle(assertion_modality=AssertionModality.CONSTITUTIVE)
        new = _particle(assertion_modality=AssertionModality.FALSIFIABLE)
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=False,
            new_supersedes_existing=True,
        )
        assert verdict is ConflictVerdict.CORROBORATES

    def test_aleatory_constitutive_not_demoted_by_supersession(self) -> None:
        # ALEATORY (uncertainty_nature) still outranks supersession even when the
        # claim is non-truth-apt: an irreducible disagreement is never retired by
        # an editorial relation. Here the truth-apt gate then corroborates (no
        # silent demotion via the prior).
        existing = _particle(
            uncertainty_nature=UncertaintyNature.ALEATORY,
            assertion_modality=AssertionModality.CONSTITUTIVE,
        )
        new = _particle(assertion_modality=AssertionModality.FALSIFIABLE)
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            new_supersedes_existing=True,
        )
        assert verdict is not ConflictVerdict.DOCUMENT_SUPERSEDES

    def test_non_truthapt_supersession_gated_by_single_trust_order(self) -> None:
        # The single-trust-order v1 gate still applies to the modality-independent
        # path: in a consensus store the prior is suppressed.
        existing = _particle(assertion_modality=AssertionModality.CONSTITUTIVE)
        new = _particle(assertion_modality=AssertionModality.FALSIFIABLE)
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            new_supersedes_existing=True,
            single_trust_order=False,
        )
        assert verdict is not ConflictVerdict.DOCUMENT_SUPERSEDES


class TestResolveConflictTrust:
    """Step 3 — trust-differential resolution."""

    def test_new_higher_trust_supersedes(self) -> None:
        existing = _particle()
        new = _particle()
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            trust_score_existing=0.5,
            trust_score_new=0.8,
            trust_differential_threshold=0.15,
        )
        assert verdict is ConflictVerdict.SUPERSEDES

    def test_existing_higher_trust_drops_new(self) -> None:
        existing = _particle()
        new = _particle()
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            trust_score_existing=0.9,
            trust_score_new=0.5,
            trust_differential_threshold=0.15,
        )
        assert verdict is ConflictVerdict.SUPERSEDED_BY_EXISTING

    def test_threshold_inclusive_at_exact_gap(self) -> None:
        # diff == threshold is auto-resolution (>= comparison).
        existing = _particle()
        new = _particle()
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            trust_score_existing=0.5,
            trust_score_new=0.65,
            trust_differential_threshold=0.15,
        )
        assert verdict is ConflictVerdict.SUPERSEDES

    def test_subthreshold_falls_to_inconsistent(self) -> None:
        existing = _particle()
        new = _particle()
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            trust_score_existing=0.5,
            trust_score_new=0.6,
            trust_differential_threshold=0.15,
        )
        assert verdict is ConflictVerdict.INCONSISTENT

    def test_missing_trust_scores_falls_to_inconsistent(self) -> None:
        # Caller did not run trust lookup → rung 2 is skipped entirely.
        existing = _particle()
        new = _particle()
        verdict = resolve_conflict(
            existing,
            new,
            has_contradiction_signal=True,
            trust_score_existing=None,
            trust_score_new=None,
        )
        assert verdict is ConflictVerdict.INCONSISTENT

    def test_one_missing_trust_score_falls_to_inconsistent(self) -> None:
        # Both must be present to compute the gap.
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=True,
            trust_score_existing=0.5,
            trust_score_new=None,
        )
        assert verdict is ConflictVerdict.INCONSISTENT


class TestResolveConflictDefault:
    """Step 3 — no other rung fires."""

    def test_default_inconsistent(self) -> None:
        assert (
            resolve_conflict(_particle(), _particle(), has_contradiction_signal=True)
            is ConflictVerdict.INCONSISTENT
        )


class TestResolveConflictConsensusMode:
    """``single_trust_order=False`` suppresses rung 2.

    In a multi-contributor / consensus store there is no global trust order
    , so auto-supersede must never fire: a confirmed contradiction
    surfaces as INCONSISTENCY (both claims stay ACTIVE), and a contributor's
    claim is never dropped by another contributor's trust.
    """

    def test_new_higher_trust_does_not_supersede(self) -> None:
        # Same inputs that yield SUPERSEDES in single mode → INCONSISTENT here.
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=True,
            trust_score_existing=0.5,
            trust_score_new=0.9,
            trust_differential_threshold=0.15,
            single_trust_order=False,
        )
        assert verdict is ConflictVerdict.INCONSISTENT

    def test_existing_higher_trust_does_not_drop_new(self) -> None:
        # Same inputs that yield SUPERSEDED_BY_EXISTING in single mode.
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=True,
            trust_score_existing=0.9,
            trust_score_new=0.5,
            trust_differential_threshold=0.15,
            single_trust_order=False,
        )
        assert verdict is ConflictVerdict.INCONSISTENT

    def test_single_mode_default_still_supersedes(self) -> None:
        # Regression guard: the default (single_trust_order=True) is unchanged.
        verdict = resolve_conflict(
            _particle(),
            _particle(),
            has_contradiction_signal=True,
            trust_score_existing=0.5,
            trust_score_new=0.9,
            trust_differential_threshold=0.15,
        )
        assert verdict is ConflictVerdict.SUPERSEDES

    def test_no_signal_still_corroborates_in_consensus_mode(self) -> None:
        # Corroboration is not a dropped claim — the gate still exonerates.
        verdict = resolve_conflict(
            _particle(content="A says X"),
            _particle(content="B quotes the claim that X"),
            has_contradiction_signal=False,
            trust_score_existing=0.5,
            trust_score_new=0.9,
            single_trust_order=False,
        )
        assert verdict is ConflictVerdict.CORROBORATES

    def test_aleatory_still_inconsistent_in_consensus_mode(self) -> None:
        verdict = resolve_conflict(
            _particle(uncertainty_nature=UncertaintyNature.ALEATORY),
            _particle(),
            has_contradiction_signal=True,
            trust_score_existing=0.1,
            trust_score_new=0.99,
            single_trust_order=False,
        )
        assert verdict is ConflictVerdict.INCONSISTENT


# ---------------------------------------------------------------------------
# build_inconsistency_particle() — pure constructor
# ---------------------------------------------------------------------------


class TestBuildInconsistencyParticle:
    def test_status_and_uncertainty_are_normative(self) -> None:
        inc = build_inconsistency_particle(
            _particle(content="A"),
            _particle(content="B"),
            corpus_entry_id="entry-99",
            snapshot_id="snap-99",
        )
        assert inc.status is Status.INCONSISTENCY
        assert inc.uncertainty_nature is UncertaintyNature.EPISTEMIC
        assert inc.confidence.calibration_source is CalibrationSource.EXTRACTOR_DIRECT

    def test_confidence_is_min_of_inputs(self) -> None:
        inc = build_inconsistency_particle(
            _particle(content="A", confidence=0.9),
            _particle(content="B", confidence=0.4),
            corpus_entry_id="entry-99",
            snapshot_id="snap-99",
        )
        assert inc.confidence.value == pytest.approx(0.4)

    def test_provenance_ordering_A_B_source(self) -> None:
        existing = _particle(content="A")
        new = _particle(content="B")
        inc = build_inconsistency_particle(
            existing,
            new,
            corpus_entry_id="entry-99",
            snapshot_id="snap-99",
        )
        assert len(inc.provenance) == 3
        # First PARTICLE ref points to existing (A)
        assert inc.provenance[0].type is ProvenanceRefType.PARTICLE
        assert inc.provenance[0].corpus_entry_id == existing.id
        # Second PARTICLE ref points to new (B)
        assert inc.provenance[1].type is ProvenanceRefType.PARTICLE
        assert inc.provenance[1].corpus_entry_id == new.id
        # SOURCE ref points to the triggering corpus entry
        assert inc.provenance[2].type is ProvenanceRefType.SOURCE
        assert inc.provenance[2].corpus_entry_id == "entry-99"
        assert inc.provenance[2].snapshot_id == "snap-99"

    def test_content_quotes_existing_id_and_excerpts(self) -> None:
        existing = _particle(content="The capital of France is Paris.")
        new = _particle(content="The capital of France is Lyon.")
        inc = build_inconsistency_particle(
            existing,
            new,
            corpus_entry_id="entry-99",
            snapshot_id="snap-99",
        )
        assert existing.id in inc.content
        assert "Paris" in inc.content
        assert "Lyon" in inc.content

    def test_asserted_by_defaults_and_overrides(self) -> None:
        inc_default = build_inconsistency_particle(
            _particle(),
            _particle(),
            corpus_entry_id="e",
            snapshot_id="s",
        )
        assert inc_default.asserted_by == "extract-pipeline"

        inc_custom = build_inconsistency_particle(
            _particle(),
            _particle(),
            corpus_entry_id="e",
            snapshot_id="s",
            asserted_by="reindex-pipeline",
        )
        assert inc_custom.asserted_by == "reindex-pipeline"


class TestSubjectIdsPropagation:
    """The bug this refactor fixes — INCONSISTENCY must inherit subject_ids.

    Before the fix, ``_resolve_conflict`` built the INCONSISTENCY ``Particle``
    inline without passing ``subject_ids``, so the resulting row had an
    empty list. That broke subject-filtered queries that should have
    surfaced the INCONSISTENCY (e.g. ``query --subject Karpathy`` skipped
    the wrappers entirely).
    """

    def test_inherits_subject_ids_from_existing(self) -> None:
        existing = _particle(content="A", subject_ids=["subj-1", "subj-2"])
        new = _particle(content="B", subject_ids=["subj-3"])
        inc = build_inconsistency_particle(
            existing,
            new,
            corpus_entry_id="entry-99",
            snapshot_id="snap-99",
        )
        # The bug: this used to be []. The fix: take from existing (A).
        assert inc.subject_ids == ["subj-1", "subj-2"]

    def test_falls_back_to_new_when_existing_empty(self) -> None:
        # Older rows / pre-Subject-store particles may have no subject_ids;
        # use the new candidate's set as fallback rather than emitting [].
        existing = _particle(content="A", subject_ids=[])
        new = _particle(content="B", subject_ids=["subj-3"])
        inc = build_inconsistency_particle(
            existing,
            new,
            corpus_entry_id="entry-99",
            snapshot_id="snap-99",
        )
        assert inc.subject_ids == ["subj-3"]

    def test_empty_when_both_empty(self) -> None:
        # Genuine corner case: neither side resolved any subjects. Empty list
        # is acceptable here because there's nothing to inherit.
        existing = _particle(content="A", subject_ids=[])
        new = _particle(content="B", subject_ids=[])
        inc = build_inconsistency_particle(
            existing,
            new,
            corpus_entry_id="entry-99",
            snapshot_id="snap-99",
        )
        assert inc.subject_ids == []

    def test_returned_list_is_independent_of_existing(self) -> None:
        # Mutating the returned subject_ids must not corrupt the input
        # particle's list — confirms we copy rather than alias.
        existing = _particle(content="A", subject_ids=["subj-1"])
        new = _particle(content="B")
        inc = build_inconsistency_particle(
            existing,
            new,
            corpus_entry_id="entry-99",
            snapshot_id="snap-99",
        )
        inc.subject_ids.append("mutated")
        assert existing.subject_ids == ["subj-1"]
