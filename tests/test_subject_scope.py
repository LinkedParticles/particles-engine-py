"""Subject scope — which claims owe a subject, and who agrees about it.

The defect this closes was a disagreement, not a bug in either surface:
techspec §9 enumerates legitimate zero-subject cases and `L-STR-09` honoured
most of them, while §14.5 measured a 100 % `subject_ids` floor over every
particle a run produced. Journal prose was the first genre to emit enough
exempt claims for the two to visibly contradict each other.

So the tests that matter here are the ones that pin *one* answer: the shared
predicate, both callers using it, and the key that cannot be set to buy a pass.
"""

from __future__ import annotations

import json

import pytest

from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    ParticleType,
    UncertaintyNature,
)
from particles.extraction.journal import _parse_journal_response
from particles.extraction.subject_scope import (
    SUBJECT_SCOPE_KEY,
    SUBJECT_SCOPE_SELF,
    is_self_scoped,
    subject_expected,
)

_SELF = {SUBJECT_SCOPE_KEY: SUBJECT_SCOPE_SELF}


def _particle(**overrides: object) -> Particle:
    fields: dict[str, object] = {
        "content": "The author woke before the alarm.",
        "confidence": Confidence(value=0.8),
        "uncertainty_nature": UncertaintyNature.EPISTEMIC,
        "asserted_by": "journal-extractor",
    }
    fields.update(overrides)
    return Particle(**fields)  # type: ignore[arg-type]


class TestIsSelfScoped:
    @pytest.mark.parametrize(
        "properties,expected",
        [
            (None, False),
            ({}, False),
            (_SELF, True),
            ({SUBJECT_SCOPE_KEY: "self"}, True),  # case-insensitive
            ({SUBJECT_SCOPE_KEY: "WORLD"}, False),
            ({SUBJECT_SCOPE_KEY: "nonsense"}, False),  # unknown ⇒ ordinary case
            ({"extraction:scope": "DOCUMENT_META"}, False),
        ],
    )
    def test_only_an_explicit_self_marks_the_claim(
        self, properties: dict[str, object] | None, expected: bool
    ) -> None:
        assert is_self_scoped(properties) is expected


class TestSubjectExpected:
    def test_an_ordinary_claim_owes_a_subject(self) -> None:
        assert subject_expected(ParticleType.CLAIM, None) is True

    @pytest.mark.parametrize("ptype", [ParticleType.REVIEW, ParticleType.NARRATIVE])
    def test_non_claim_types_do_not(self, ptype: ParticleType) -> None:
        # The journal extractor's NARRATIVE label lands here — it names an
        # entry, not an entity.
        assert subject_expected(ptype, None) is False

    @pytest.mark.parametrize(
        "properties",
        [
            {"extraction:scope": "DOCUMENT_META"},
            {"extraction:polarity": "DECLINED"},
            {"extraction:polarity": "HYPOTHETICAL"},
            _SELF,
        ],
    )
    def test_the_four_exempt_populations(self, properties: dict[str, object]) -> None:
        assert subject_expected(ParticleType.CLAIM, properties) is False

    def test_a_failed_resolution_is_not_exempt(self) -> None:
        # The whole point: "subject resolution produced nothing" must stay
        # visible. Only a *marked* claim is excused.
        assert subject_expected(ParticleType.CLAIM, {"nmo:hasIssuer": "x"}) is True


class TestJournalParserSetsTheKey:
    @staticmethod
    def _reply(**claim: object) -> str:
        base: dict[str, object] = {
            "content": "The author woke before the alarm.",
            "confidence_value": 0.9,
            "uncertainty_nature": "EPISTEMIC",
            "assertion_modality": "FALSIFIABLE",
            "subjects": [],
        }
        base.update(claim)
        return json.dumps({"narrative_label": "A hard travel day.", "claims": [base]})

    def test_self_with_no_subjects_records_the_key(self) -> None:
        candidates, _ = _parse_journal_response(self._reply(subject_scope="SELF"))
        assert candidates[0].properties == _SELF

    def test_world_records_nothing(self) -> None:
        candidates, _ = _parse_journal_response(self._reply(subject_scope="WORLD"))
        assert candidates[0].properties is None

    def test_a_reply_without_the_field_records_nothing(self) -> None:
        # A prompt predating this axis (or a model that dropped the field)
        # behaves exactly as extractor 0.3.0 did.
        candidates, _ = _parse_journal_response(self._reply())
        assert candidates[0].properties is None

    def test_self_is_ignored_when_the_claim_named_a_subject(self) -> None:
        # The key asserts "the author is the only available subject", which is
        # false here — and this is what stops it being a way to buy a pass on
        # the floor, since a claim with subjects already passes.
        candidates, _ = _parse_journal_response(
            self._reply(subject_scope="SELF", subjects=["Balatro"])
        )
        assert candidates[0].properties is None
        assert candidates[0].subjects == ["Balatro"]

    def test_the_narrative_candidate_is_untouched(self) -> None:
        candidates, _ = _parse_journal_response(self._reply(subject_scope="SELF"))
        narrative = candidates[-1]
        assert narrative.particle_type is ParticleType.NARRATIVE
        assert narrative.properties is None
        # …and is exempt anyway, by type.
        assert subject_expected(narrative.particle_type, narrative.properties) is False


class TestLintHonoursTheExclusion:
    @pytest.mark.asyncio
    async def test_a_self_scoped_claim_is_not_flagged(self, db_session) -> None:  # type: ignore[no-untyped-def]
        from particles.operations.lint.coverage import _check_no_subject_claims
        from particles.store.particle_store import insert_particle

        await insert_particle(db_session, _particle(properties=_SELF))

        assert await _check_no_subject_claims(db_session) == []

    @pytest.mark.asyncio
    async def test_an_unmarked_subjectless_claim_still_is(self, db_session) -> None:  # type: ignore[no-untyped-def]
        from particles.operations.lint.coverage import _check_no_subject_claims
        from particles.store.particle_store import insert_particle

        await insert_particle(db_session, _particle())

        findings = await _check_no_subject_claims(db_session)
        assert [f.finding_type for f in findings] == ["NO_SUBJECT"]


class TestConformanceMeasuresTheRightPopulation:
    @staticmethod
    def _stat(particles: list[Particle]):  # type: ignore[no-untyped-def]
        from particles.conformance.contract import CONTRACT
        from particles.conformance.validator import _compute_field_stat

        entry = next(c for c in CONTRACT if c.field == "subject_ids")
        return _compute_field_stat(entry, particles, {}, 0.8)

    def test_exempt_particles_leave_the_denominator(self) -> None:
        stat = self._stat(
            [
                _particle(subject_ids=["s1"]),
                _particle(properties=_SELF),
                _particle(particle_type=ParticleType.NARRATIVE),
            ]
        )
        assert stat.total_count == 1
        assert stat.excluded_count == 2
        assert stat.rate == 1.0
        assert stat.passes_threshold is True

    def test_an_unmarked_subjectless_claim_still_fails(self) -> None:
        stat = self._stat([_particle(subject_ids=["s1"]), _particle()])
        assert (stat.total_count, stat.excluded_count) == (2, 0)
        assert stat.passes_threshold is False

    def test_an_all_exempt_run_is_unevaluated_not_perfect(self) -> None:
        # A collapsed denominator must not read as 100 % — nothing about the
        # extractor was measured, which is the guard's whole premise.
        stat = self._stat([_particle(properties=_SELF)])
        assert (stat.total_count, stat.excluded_count, stat.rate) == (0, 1, 0.0)
        assert stat.passes_threshold is False
        assert "exempt" in (stat.failure_reason or "")

    def test_an_empty_run_keeps_its_own_message(self) -> None:
        stat = self._stat([])
        assert stat.excluded_count == 0
        assert stat.failure_reason == "No particles produced; cannot evaluate"

    def test_other_fields_measure_over_the_whole_run(self) -> None:
        from particles.conformance.contract import CONTRACT
        from particles.conformance.validator import _compute_field_stat

        entry = next(c for c in CONTRACT if c.field == "content")
        stat = _compute_field_stat(entry, [_particle(properties=_SELF), _particle()], {}, 0.8)
        assert (stat.total_count, stat.excluded_count) == (2, 0)

    def test_the_exemption_is_not_a_free_pass_for_a_marked_world_claim(self) -> None:
        # Marking a claim SELF while it carries a world subject is the abuse the
        # parser refuses to write; if one reached the store anyway it is exempt,
        # so the record must stay honest at the point it is written, not here.
        stat = self._stat([_particle(properties={**_SELF, "extraction:polarity": "ASSERTED"})])
        assert stat.excluded_count == 1


class TestBothSurfacesShareOnePredicate:
    def test_lint_and_conformance_agree_on_every_shape(self) -> None:
        from particles.conformance.contract import FIELD_EXEMPTIONS

        exempt = FIELD_EXEMPTIONS["subject_ids"]
        for p in (
            _particle(),
            _particle(properties=_SELF),
            _particle(properties={"extraction:scope": "DOCUMENT_META"}),
            _particle(properties={"extraction:polarity": "DECLINED"}),
            _particle(particle_type=ParticleType.NARRATIVE),
            _particle(assertion_modality=AssertionModality.EXPERIENTIAL),
        ):
            assert exempt(p) is not subject_expected(p.particle_type, p.properties)
