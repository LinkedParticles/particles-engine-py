"""Tests for the derived structured-claim annotation.

Covers the four surfaces the ADR adds, in the order a reader meets them:

1. Core model validators (``tests/AGENTS.md``: schema + Pydantic model changes
   are always in scope).
2. Storage round-trip through ``ParticleRow`` — including the payload/stamp
   split, which exists so the backfill scope query and coverage report stay SQL.
3. The Client-layer parser and subject binder shared by both producers.
4. The backfill pass and the L-STR-11 lint check.

The LLM call itself is out of scope per ``tests/AGENTS.md`` § Mocking strategy
(the seam is tested in ``test_llm.py``); what is tested here is the parser
against representative replies, which the same section requires.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from particles.core.schema import (
    CanonicalForm,
    ClaimTerm,
    Confidence,
    Particle,
    StructuredClaim,
    TermKind,
    UncertaintyNature,
)
from particles.extraction.structure import (
    STRUCTURIZER_ID,
    STRUCTURIZER_VERSION,
    bind_subject_id,
    parse_structured_claim_payload,
)
from particles.store.particle_store import (
    ParticleRow,
    count_particles_needing_structured_claim,
    count_structured_claim_coverage,
    get_particles_needing_structured_claim,
    insert_particle,
    record_structured_claim_declined,
    set_structured_claim,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _term(value: str = "Sirius", kind: TermKind = TermKind.TOKEN) -> ClaimTerm:
    return ClaimTerm(kind=kind, value=value)


def _claim(**overrides: object) -> StructuredClaim:
    fields: dict[str, object] = {
        "subject": _term(),
        "predicate": _term("has spectral type"),
        "object": ClaimTerm(kind=TermKind.LITERAL, value="A1V"),
        "structurizer_id": "general-extractor",
        "structurizer_version": "0.13.0",
        "generated_at": datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
    }
    fields.update(overrides)
    return StructuredClaim(**fields)  # type: ignore[arg-type]


def _particle(**overrides: object) -> Particle:
    fields: dict[str, object] = {
        "content": "Sirius has spectral type A1V.",
        "confidence": Confidence(value=0.8),
        "uncertainty_nature": UncertaintyNature.EPISTEMIC,
        "asserted_by": "general-extractor",
    }
    fields.update(overrides)
    return Particle(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Core model validators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", ["subject", "predicate"])
def test_literal_is_rejected_in_subject_and_predicate_position(position: str) -> None:
    """A literal cannot be a subject or a predicate in any exportable triple."""
    with pytest.raises(ValueError, match="cannot occupy the"):
        _claim(**{position: ClaimTerm(kind=TermKind.LITERAL, value="x")})


def test_literal_is_accepted_in_object_position() -> None:
    assert _claim().object.kind is TermKind.LITERAL


@pytest.mark.parametrize("kind", [TermKind.URI, TermKind.TOKEN])
def test_datatype_and_language_are_literal_only(kind: TermKind) -> None:
    with pytest.raises(ValueError, match="only on a LITERAL term"):
        ClaimTerm(kind=kind, value="x", language="en")


def test_a_literal_carries_a_datatype_or_a_language_never_both() -> None:
    with pytest.raises(ValueError, match="never both"):
        ClaimTerm(kind=TermKind.LITERAL, value="x", datatype="xsd:string", language="en")


def test_structured_canonical_form_requires_a_structured_claim() -> None:
    with pytest.raises(ValueError, match="requires a structured_claim"):
        _particle(canonical_form=CanonicalForm.STRUCTURED)


def test_prose_canonical_form_may_still_carry_an_annotation() -> None:
    """The normal case: prose is the assertion, the triple is the annotation."""
    p = _particle(structured_claim=_claim())
    assert p.canonical_form is CanonicalForm.PROSE
    assert p.structured_claim is not None


def test_a_particle_defaults_to_prose_canonical_and_unannotated() -> None:
    """Absence is the default and a legal permanent state."""
    p = _particle()
    assert p.structured_claim is None
    assert p.canonical_form is CanonicalForm.PROSE


# ---------------------------------------------------------------------------
# 2. Storage
# ---------------------------------------------------------------------------


def test_row_round_trip_preserves_the_annotation_and_its_stamp() -> None:
    claim = _claim(subject_id="11111111-1111-1111-1111-111111111111")
    row = ParticleRow.from_model(_particle(structured_claim=claim))
    restored = row.to_model().structured_claim
    assert restored == claim


def test_stamp_is_stored_in_columns_not_duplicated_in_the_payload() -> None:
    """One storage location per fact, so payload and stamp cannot drift."""
    row = ParticleRow.from_model(_particle(structured_claim=_claim()))
    payload = json.loads(row.structured_claim_json or "{}")
    assert set(payload) == {"subject", "predicate", "object"}
    assert row.structurizer_id == "general-extractor"
    assert row.structurizer_version == "0.13.0"
    assert row.structured_claim_generated_at is not None


def test_payload_and_stamp_are_written_together_or_not_at_all() -> None:
    """There is no legacy tier: a payload with a NULL stamp is unreachable."""
    row = ParticleRow.from_model(_particle())
    assert row.structured_claim_json is None
    assert row.structurizer_id is None
    assert row.structurizer_version is None
    assert row.structured_claim_generated_at is None
    assert row.canonical_form == "PROSE"


def test_a_payload_without_a_stamp_is_reported_as_corrupt() -> None:
    """No grandfathered default — the unreachable state says so rather than
    inventing a LEGACY_* sentinel (the one divergence)."""
    row = ParticleRow.from_model(_particle(structured_claim=_claim()))
    row.structurizer_id = None
    with pytest.raises(ValueError, match="without a complete structurizer stamp"):
        row.to_model()


@pytest.mark.asyncio
async def test_set_structured_claim_touches_only_the_annotation(db_session) -> None:  # type: ignore[no-untyped-def]
    """annotating may not disturb the belief."""
    particle = _particle()
    await insert_particle(db_session, particle)

    await set_structured_claim(db_session, particle.id, _claim())

    row = await db_session.get(ParticleRow, particle.id)
    assert row.structurizer_id == "general-extractor"
    # The asserted quantities are byte-identical.
    assert row.content == particle.content
    assert row.confidence_value == particle.confidence.value
    assert row.status == particle.status.value
    assert row.provenance_json == json.dumps([])


@pytest.mark.asyncio
async def test_scope_query_finds_unannotated_then_stale_versions(db_session) -> None:  # type: ignore[no-untyped-def]
    bare = _particle(content="Unannotated claim.")
    stamped = _particle(content="Annotated claim.", structured_claim=_claim())
    await insert_particle(db_session, bare)
    await insert_particle(db_session, stamped)

    unannotated = await get_particles_needing_structured_claim(db_session)
    assert [p.id for p in unannotated] == [bare.id]

    # The regeneration scope is the mirror of `reindex --extractor-version`:
    # annotated particles stamped with some OTHER version.
    stale = await get_particles_needing_structured_claim(db_session, structurizer_version="9.9.9")
    assert [p.id for p in stale] == [stamped.id]

    current = await get_particles_needing_structured_claim(
        db_session, structurizer_version="0.13.0"
    )
    assert current == []


@pytest.mark.asyncio
async def test_structure_canonical_particles_are_never_in_the_backfill_scope(db_session) -> None:  # type: ignore[no-untyped-def]
    """The backfill regenerates *derived* annotations. On a
    STRUCTURED particle the triple is the assertion a parser read from the
    source, so regenerating it would replace an assertion with
    the content-structurizer's guess at it. Redoing one is an
    ``EXTRACTOR_VERSION`` bump plus ``reindex``, never this pass — and since no
    structure-native extractor stamps the standalone structurizer's version,
    every one of them would otherwise land in a routine regeneration sweep.
    """
    from particles.core.schema import CanonicalForm

    structured = _particle(
        content="Douglas Adams place of birth: Cambridge",
        structured_claim=_claim(structurizer_id="wikidata-extractor"),
        canonical_form=CanonicalForm.STRUCTURED,
    )
    prose = _particle(content="Annotated prose claim.", structured_claim=_claim())
    await insert_particle(db_session, structured)
    await insert_particle(db_session, prose)

    stale = await get_particles_needing_structured_claim(db_session, structurizer_version="9.9.9")
    assert [p.id for p in stale] == [prose.id]
    assert (
        await count_particles_needing_structured_claim(db_session, structurizer_version="9.9.9")
        == 1
    )


@pytest.mark.asyncio
async def test_coverage_counts_are_a_census_not_a_finding(db_session) -> None:  # type: ignore[no-untyped-def]
    await insert_particle(db_session, _particle(content="Bare."))
    await insert_particle(db_session, _particle(content="Rich.", structured_claim=_claim()))

    coverage = await count_structured_claim_coverage(db_session)
    assert coverage["active"] == 2
    assert coverage["annotated"] == 1
    assert coverage["by_structurizer"] == {"general-extractor@0.13.0": 1}


# ---------------------------------------------------------------------------
# 3. The shared parser + subject binder
# ---------------------------------------------------------------------------


def test_parser_infers_term_kinds_from_a_bare_string_reply() -> None:
    """The shape an LLM actually returns: three plain strings."""
    claim = parse_structured_claim_payload(
        {"subject": "Sirius", "predicate": "has spectral type", "object": "A1V"},
        structurizer_id=STRUCTURIZER_ID,
        structurizer_version=STRUCTURIZER_VERSION,
    )
    assert claim is not None
    assert claim.subject.kind is TermKind.TOKEN
    assert claim.predicate.kind is TermKind.TOKEN
    assert claim.object.kind is TermKind.LITERAL
    assert claim.structurizer_id == STRUCTURIZER_ID


def test_parser_recognises_a_uri_term() -> None:
    claim = parse_structured_claim_payload(
        {"subject": "wd:Q3132", "predicate": "wdt:P215", "object": "A1V"},
        structurizer_id="x",
        structurizer_version="1",
    )
    assert claim is not None
    assert claim.subject.kind is TermKind.URI
    assert claim.predicate.kind is TermKind.URI


def test_parser_demotes_a_literal_out_of_predicate_position() -> None:
    """A model slip coerces rather than losing the whole triple."""
    claim = parse_structured_claim_payload(
        {
            "subject": {"kind": "LITERAL", "value": "Sirius"},
            "predicate": "has spectral type",
            "object": "A1V",
        },
        structurizer_id="x",
        structurizer_version="1",
    )
    assert claim is not None
    assert claim.subject.kind is TermKind.TOKEN


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "null",
        {},
        {"subject": "Sirius"},
        {"subject": "Sirius", "predicate": "", "object": "A1V"},
        {"subject": " ", "predicate": "p", "object": "o"},
        [1, 2, 3],
    ],
)
def test_parser_returns_none_rather_than_a_partial_triple(payload: object) -> None:
    """The claim is never lost to a structurizing failure — it just goes
    unannotated, which is legal and permanent."""
    assert (
        parse_structured_claim_payload(payload, structurizer_id="x", structurizer_version="1")
        is None
    )


def test_bind_subject_id_matches_case_insensitively() -> None:
    claim = bind_subject_id(_claim(subject=_term("sirius")), ["Sirius"], ["uuid-1"])
    assert claim.subject_id == "uuid-1"


def test_bind_subject_id_leaves_none_when_nothing_matches() -> None:
    """An unresolved subject term is honest, not an error (lint agrees)."""
    claim = bind_subject_id(_claim(subject=_term("Betelgeuse")), ["Sirius"], ["uuid-1"])
    assert claim.subject_id is None


# ---------------------------------------------------------------------------
# 4. Lint L-STR-11
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lint_flags_a_subject_the_particle_is_not_about(db_session) -> None:  # type: ignore[no-untyped-def]
    from particles.operations.lint.coverage import _check_structured_claim_subjects

    mine = "11111111-1111-1111-1111-111111111111"
    theirs = "22222222-2222-2222-2222-222222222222"
    await insert_particle(
        db_session,
        _particle(subject_ids=[mine], structured_claim=_claim(subject_id=theirs)),
    )

    findings = await _check_structured_claim_subjects(db_session)
    assert [f.finding_type for f in findings] == ["STRUCTURED_CLAIM_SUBJECT_MISMATCH"]
    assert findings[0].severity == "WARNING"
    # The remedy is regeneration of the annotation, never a change to the claim.
    assert "particles structure" in (findings[0].recommended_action or "")


@pytest.mark.asyncio
async def test_lint_does_not_flag_a_matching_or_unresolved_subject(db_session) -> None:  # type: ignore[no-untyped-def]
    from particles.operations.lint.coverage import _check_structured_claim_subjects

    mine = "11111111-1111-1111-1111-111111111111"
    await insert_particle(
        db_session,
        _particle(
            content="Matching subject.",
            subject_ids=[mine],
            structured_claim=_claim(subject_id=mine),
        ),
    )
    # subject_id=None records "the term resolved to no Subject" — a coverage
    # gap, not an error, so flagging it would create a recurring false alarm.
    await insert_particle(
        db_session,
        _particle(content="Unresolved subject.", subject_ids=[mine], structured_claim=_claim()),
    )
    # And an un-annotated particle is never flagged at all.
    await insert_particle(db_session, _particle(content="No annotation."))

    assert await _check_structured_claim_subjects(db_session) == []


# ---------------------------------------------------------------------------
# 5. Extraction-time population (the free path)
# ---------------------------------------------------------------------------


def _patch_structure_config(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    from particles.config import ParticlesConfig, StructuredClaimConfig
    from particles.extraction import general

    cfg = ParticlesConfig(structured_claim=StructuredClaimConfig(enabled=enabled))
    monkeypatch.setattr(general, "get_config", lambda: cfg)


def _parse_extraction_response_for(raw: str):  # type: ignore[no-untyped-def]
    from particles.extraction.general import _parse_extraction_response

    return _parse_extraction_response(raw)


def _reply(item: dict[str, object]) -> str:
    base: dict[str, object] = {
        "content": "Sirius has spectral type A1V.",
        "subjects": ["Sirius"],
        "confidence_value": 0.9,
        "uncertainty_nature": "EPISTEMIC",
    }
    base.update(item)
    return json.dumps([base])


class TestExtractionTimePopulation:
    def test_triple_is_parsed_and_stamped_with_the_extractor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.extraction.general import EXTRACTOR_ID, EXTRACTOR_VERSION

        _patch_structure_config(monkeypatch, enabled=True)
        raw = _reply(
            {
                "structured_claim": {
                    "subject": "Sirius",
                    "predicate": "has spectral type",
                    "object": "A1V",
                }
            }
        )
        candidates, _notes = _parse_extraction_response_for(raw)
        claim = candidates[0].structured_claim
        assert claim is not None
        # The stamp records what ACTUALLY produced the triple, which is what
        # keeps the extraction-time and backfill populations tellable apart.
        assert claim.structurizer_id == EXTRACTOR_ID
        assert claim.structurizer_version == EXTRACTOR_VERSION

    def test_null_triple_leaves_the_claim_unannotated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The model declining is the designed outcome, not a failure."""
        _patch_structure_config(monkeypatch, enabled=True)
        candidates, _notes = _parse_extraction_response_for(_reply({"structured_claim": None}))
        assert candidates[0].structured_claim is None
        assert candidates[0].content  # the claim itself survives intact

    def test_malformed_triple_is_absorbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_structure_config(monkeypatch, enabled=True)
        candidates, _notes = _parse_extraction_response_for(
            _reply({"structured_claim": {"subject": "Sirius"}})
        )
        assert candidates[0].structured_claim is None
        assert candidates[0].content == "Sirius has spectral type A1V."

    def test_disabled_config_never_annotates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_structure_config(monkeypatch, enabled=False)
        candidates, _notes = _parse_extraction_response_for(
            _reply(
                {
                    "structured_claim": {
                        "subject": "Sirius",
                        "predicate": "p",
                        "object": "o",
                    }
                }
            )
        )
        assert candidates[0].structured_claim is None


class TestPromptGating:
    def test_disabled_prompt_is_byte_identical(self) -> None:
        """Disabling the feature is fully inert, as for every other gated field."""
        from particles.extraction.general import (
            _build_extract_prompt,
            _extraction_response_schema,
        )

        off = _build_extract_prompt(scope_enabled=False, modality_enabled=False)
        on = _build_extract_prompt(
            scope_enabled=False, modality_enabled=False, structure_enabled=True
        )
        assert on != off
        assert "structured_claim" in on
        assert "structured_claim" not in off

        schema_off = _extraction_response_schema(scope_enabled=False, modality_enabled=False)
        schema_on = _extraction_response_schema(
            scope_enabled=False, modality_enabled=False, structure_enabled=True
        )
        assert "structured_claim" not in schema_off["items"]["properties"]
        assert "structured_claim" in schema_on["items"]["properties"]


class TestCandidateToParticle:
    def test_subject_term_binds_to_the_resolved_uuid(self) -> None:
        from particles.extraction.general import CandidateParticle, candidate_to_particle

        cand = CandidateParticle(
            content="Sirius has spectral type A1V.",
            confidence_value=0.9,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            subjects=["Sirius"],
            structured_claim=_claim(),
        )
        particle = candidate_to_particle(
            cand, corpus_entry_id="e1", snapshot_id="s1", subject_ids=["uuid-1"]
        )
        assert particle.structured_claim is not None
        assert particle.structured_claim.subject_id == "uuid-1"
        # Prose stays the assertion for everything this extractor produces.
        assert particle.canonical_form is CanonicalForm.PROSE

    def test_unannotated_candidate_stays_unannotated(self) -> None:
        from particles.extraction.general import CandidateParticle, candidate_to_particle

        cand = CandidateParticle(
            content="A claim with no triple.",
            confidence_value=0.9,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
        )
        particle = candidate_to_particle(
            cand, corpus_entry_id="e1", snapshot_id="s1", subject_ids=[]
        )
        assert particle.structured_claim is None


# ---------------------------------------------------------------------------
# 6. Interchange round-trip
# ---------------------------------------------------------------------------


class TestInterchange:
    def test_annotation_survives_a_round_trip_with_its_stamp(self) -> None:
        """A stamped derived annotation travels; dropping it would lose an
        LLM call's worth of work on every store copy."""
        from particles.interchange.codec import from_unit, to_unit

        original = _particle(structured_claim=_claim())
        restored = from_unit(to_unit(original, {})).particle
        assert restored.structured_claim is not None
        assert restored.structured_claim.subject == original.structured_claim.subject
        assert restored.structured_claim.object == original.structured_claim.object
        assert restored.structured_claim.structurizer_id == "general-extractor"
        assert restored.structured_claim.structurizer_version == "0.13.0"
        assert restored.canonical_form is CanonicalForm.PROSE

    def test_store_local_subject_uuid_does_not_cross_the_boundary(self) -> None:
        """Cross-store identity travels by external reference (§3), so a
        foreign UUID would be meaningless in the importing store."""
        from particles.interchange.codec import from_unit, to_unit

        original = _particle(structured_claim=_claim(subject_id="uuid-1"))
        unit = to_unit(original, {})
        assert "subjectId" not in unit["structuredClaim"]
        assert from_unit(unit).particle.structured_claim.subject_id is None

    def test_unannotated_particle_emits_no_structured_claim(self) -> None:
        from particles.interchange.codec import to_unit

        unit = to_unit(_particle(), {})
        assert "structuredClaim" not in unit
        assert unit["canonicalForm"] == "PROSE"


# ---------------------------------------------------------------------------
# 7. Backfill
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_reports_backlog_and_writes_nothing(db_session) -> None:  # type: ignore[no-untyped-def]
    from particles.operations import structure as structure_op

    await insert_particle(db_session, _particle(content="Unannotated."))

    summary = await structure_op.backfill_structured_claims(db_session, dry_run=True)

    assert summary["dry_run"] is True
    assert summary["backlog"] == 1
    assert summary["annotated"] == 0
    assert (await count_structured_claim_coverage(db_session))["annotated"] == 0


@pytest.mark.asyncio
async def test_dry_run_reports_the_whole_backlog_not_the_batch_cap(db_session) -> None:  # type: ignore[no-untyped-def]
    """A binding cap must be disclosed, never reported as the size of the job.

    The bug this pins: the probe used to count the *capped* list, so a
    21k-particle store's dry run said "scope: 200" — the number an operator
    would size the job from.
    """
    from particles.operations import structure as structure_op

    for i in range(5):
        await insert_particle(db_session, _particle(content=f"Unannotated {i}."))

    summary = await structure_op.backfill_structured_claims(db_session, limit=2, dry_run=True)

    assert summary["backlog"] == 5
    assert summary["batch_limit"] == 2
    assert summary["runs_needed"] == 3  # ceil(5 / 2)


@pytest.mark.asyncio
async def test_a_capped_run_reports_what_is_left(db_session) -> None:  # type: ignore[no-untyped-def]
    """So a capped run never reads as "done"."""
    from particles.operations import structure as structure_op

    for i in range(5):
        await insert_particle(db_session, _particle(content=f"Unannotated {i}."))

    async def _annotates(content: str, subject_names: list[str]) -> StructuredClaim:
        return _claim()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(structure_op, "structure_content", _annotates)
    try:
        summary = await structure_op.backfill_structured_claims(
            db_session, limit=2, rate_limit_per_minute=0
        )
    finally:
        monkeypatch.undo()

    assert summary["scope"] == 2
    assert summary["annotated"] == 2
    assert summary["remaining"] == 3


@pytest.mark.asyncio
async def test_backfill_annotates_and_leaves_the_belief_untouched(  # type: ignore[no-untyped-def]
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.operations import structure as structure_op

    particle = _particle(content="Sirius has spectral type A1V.")
    await insert_particle(db_session, particle)

    async def _fake_structure(content: str, subject_names: list[str]) -> StructuredClaim:
        return _claim(structurizer_id=STRUCTURIZER_ID, structurizer_version=STRUCTURIZER_VERSION)

    monkeypatch.setattr(structure_op, "structure_content", _fake_structure)

    summary = await structure_op.backfill_structured_claims(db_session, rate_limit_per_minute=0)

    assert summary["annotated"] == 1
    assert summary["failed"] == 0
    row = await db_session.get(ParticleRow, particle.id)
    assert row.structurizer_id == STRUCTURIZER_ID
    # the asserted quantities are byte-identical after the pass.
    assert row.content == particle.content
    assert row.confidence_value == particle.confidence.value
    assert row.status == particle.status.value


@pytest.mark.asyncio
async def test_declined_particles_are_skipped_not_failed(  # type: ignore[no-untyped-def]
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "This prose has no honest triple" is a valid permanent answer, so it
    must not read as an error in the operator's summary."""
    from particles.operations import structure as structure_op

    await insert_particle(db_session, _particle(content="The migration was harder than expected."))

    async def _declines(content: str, subject_names: list[str]) -> None:
        return None

    monkeypatch.setattr(structure_op, "structure_content", _declines)

    summary = await structure_op.backfill_structured_claims(db_session, rate_limit_per_minute=0)
    assert summary["skipped"] == 1
    assert summary["annotated"] == 0
    assert summary["failed"] == 0


@pytest.mark.asyncio
async def test_one_failure_does_not_end_the_run(  # type: ignore[no-untyped-def]
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.operations import structure as structure_op

    first = _particle(content="First claim.")
    second = _particle(content="Second claim.")
    await insert_particle(db_session, first)
    await insert_particle(db_session, second)

    calls: list[str] = []

    async def _fails_once(content: str, subject_names: list[str]) -> StructuredClaim:
        calls.append(content)
        if len(calls) == 1:
            raise RuntimeError("provider exploded")
        return _claim()

    monkeypatch.setattr(structure_op, "structure_content", _fails_once)

    summary = await structure_op.backfill_structured_claims(db_session, rate_limit_per_minute=0)
    assert summary["failed"] == 1
    assert summary["annotated"] == 1


@pytest.mark.asyncio
async def test_limit_bounds_the_run(db_session, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from particles.operations import structure as structure_op

    for i in range(3):
        await insert_particle(db_session, _particle(content=f"Claim {i}."))

    async def _annotates(content: str, subject_names: list[str]) -> StructuredClaim:
        return _claim()

    monkeypatch.setattr(structure_op, "structure_content", _annotates)

    summary = await structure_op.backfill_structured_claims(
        db_session, limit=2, rate_limit_per_minute=0
    )
    assert summary["scope"] == 2
    assert summary["annotated"] == 2


@pytest.mark.asyncio
async def test_quarantine_promotion_carries_the_annotation_verbatim(db_session) -> None:  # type: ignore[no-untyped-def]
    """A mint-from-verbatim keeps the stamp it was given.

    Re-stamping with the *current* structurizer version would misdate a triple
    nobody regenerated. The embedding needed ``copy_particle_embedding`` for
    this because it is storage metadata; the annotation is a model field, so
    ``model_copy`` already carries it — this test is what keeps that true.
    """
    from particles.core.status import Status, StatusReason
    from particles.operations._quarantine import promote_quarantined

    claim = _claim(structurizer_version="0.0.1-ancient")
    quarantined = _particle(
        content="A quarantined conflict loser.",
        status=Status.PROVENANCE_STALE,
        status_reason=StatusReason.CONFLICT_PENDING,
        structured_claim=claim,
    )
    await insert_particle(db_session, quarantined)

    minted = await promote_quarantined(db_session, quarantined)

    assert minted.structured_claim == claim
    row = await db_session.get(ParticleRow, minted.id)
    assert row.structurizer_version == "0.0.1-ancient"


# ---------------------------------------------------------------------------
# 8. A recorded decline is what makes the pass terminable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_decline_is_recorded_as_a_stamp_without_a_payload(db_session) -> None:  # type: ignore[no-untyped-def]
    """The fourth column state: attempted, declined, still un-annotated."""
    particle = _particle(content="The migration was harder than expected.")
    await insert_particle(db_session, particle)

    await record_structured_claim_declined(
        db_session,
        particle.id,
        structurizer_id="content-structurizer",
        structurizer_version="1.0.0",
    )

    row = await db_session.get(ParticleRow, particle.id)
    assert row.structured_claim_json is None  # still no annotation …
    assert row.structurizer_version == "1.0.0"  # … but we recorded that we asked
    # It reads back as un-annotated: absence stays absence.
    assert row.to_model().structured_claim is None
    # And the belief is untouched.
    assert row.content == particle.content
    assert row.confidence_value == particle.confidence.value


@pytest.mark.asyncio
async def test_a_declined_particle_leaves_the_backlog(db_session) -> None:  # type: ignore[no-untyped-def]
    """The bug this pins: without it the backfill cannot finish.

    A declined particle used to stay in scope forever, so every future run
    re-paid for it. Observed on the dogfood store: a 500-particle run declined
    207, and those 207 led the next run's scan — progress asymptotes to zero
    while spend continues.
    """
    particle = _particle(content="No honest triple here.")
    await insert_particle(db_session, particle)
    assert await count_particles_needing_structured_claim(db_session) == 1

    await record_structured_claim_declined(
        db_session,
        particle.id,
        structurizer_id="content-structurizer",
        structurizer_version="1.0.0",
    )

    assert await count_particles_needing_structured_claim(db_session) == 0
    assert await get_particles_needing_structured_claim(db_session) == []


@pytest.mark.asyncio
async def test_a_better_structurizer_re_asks_the_declined(db_session) -> None:  # type: ignore[no-untyped-def]
    """Recording the decline must not make it permanent — that is what the
    version stamp buys."""
    particle = _particle(content="No honest triple here.")
    await insert_particle(db_session, particle)
    await record_structured_claim_declined(
        db_session,
        particle.id,
        structurizer_id="content-structurizer",
        structurizer_version="1.0.0",
    )

    stale = await get_particles_needing_structured_claim(db_session, structurizer_version="2.0.0")
    assert [p.id for p in stale] == [particle.id]


@pytest.mark.asyncio
async def test_the_backfill_drains_a_store_of_undeclinable_prose(  # type: ignore[no-untyped-def]
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end termination: a store the structurizer declines *entirely*
    must still reach a zero backlog, in one pass, and never be re-charged."""
    from particles.operations import structure as structure_op

    for i in range(5):
        await insert_particle(db_session, _particle(content=f"Untriplable {i}."))

    calls: list[str] = []

    async def _always_declines(content: str, subject_names: list[str]) -> None:
        calls.append(content)
        return None

    monkeypatch.setattr(structure_op, "structure_content", _always_declines)

    first = await structure_op.backfill_structured_claims(db_session, rate_limit_per_minute=0)
    assert first["skipped"] == 5
    assert first["remaining"] == 0  # ← the whole point

    second = await structure_op.backfill_structured_claims(db_session, rate_limit_per_minute=0)
    assert second["scope"] == 0
    assert len(calls) == 5  # the second run cost nothing


@pytest.mark.asyncio
async def test_limit_zero_means_the_whole_backlog(db_session) -> None:  # type: ignore[no-untyped-def]
    from particles.operations import structure as structure_op

    for i in range(5):
        await insert_particle(db_session, _particle(content=f"Claim {i}."))

    summary = await structure_op.backfill_structured_claims(db_session, limit=0, dry_run=True)
    assert summary["backlog"] == 5
    assert summary["batch_limit"] == 0
    assert summary["runs_needed"] == 1


@pytest.mark.asyncio
async def test_work_is_committed_before_the_run_ends(  # type: ignore[no-untyped-def]
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long run must not hold every annotation until the final commit — an
    interrupt would discard every call already paid for."""
    from particles.operations import structure as structure_op

    for i in range(4):
        await insert_particle(db_session, _particle(content=f"Claim {i}."))

    commits: list[int] = []
    seen = 0

    async def _annotates(content: str, subject_names: list[str]) -> StructuredClaim:
        nonlocal seen
        seen += 1
        return _claim()

    original_commit = db_session.commit

    async def _tracking_commit() -> None:
        commits.append(seen)
        await original_commit()

    monkeypatch.setattr(structure_op, "structure_content", _annotates)
    monkeypatch.setattr(db_session, "commit", _tracking_commit)
    monkeypatch.setattr(structure_op.get_config().structured_claim, "backfill_commit_interval", 2)

    await structure_op.backfill_structured_claims(db_session, rate_limit_per_minute=0)

    # Committed mid-run (after 2 and 4), not only once at the end.
    assert commits[0] < 4
    assert len(commits) >= 2
