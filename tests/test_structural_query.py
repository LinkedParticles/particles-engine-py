"""Tests for operations/query/structural.py — structural claim filters.

Covers the ADR's test table: mode dispatch (question / filters / both /
aggregates, question+aggregate rejection), the deterministic path making zero
LLM/embedding calls, the prefilter leaving semantic ranking unchanged, the
coverage footer, the "claims" output wording, the ``--predicates`` listing,
and the ``--group-by subject`` fallback join.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from particles.core.schema import (
    ClaimTerm,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    StructuralGroupBy,
    StructuredClaim,
    Subject,
    TermKind,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.query.structural import coverage_line, disclosure_lines

Session = Any  # the db_session fixture is untyped


def _claim(
    predicate: str = "nmo:hasWeight",
    obj_value: str = "3.9",
    obj_datatype: str | None = "xsd:decimal",
    subject_id: str | None = None,
) -> StructuredClaim:
    return StructuredClaim(
        subject=ClaimTerm(kind=TermKind.TOKEN, value="1 Pfennig"),
        predicate=ClaimTerm(kind=TermKind.URI, value=predicate),
        object=ClaimTerm(kind=TermKind.LITERAL, value=obj_value, datatype=obj_datatype),
        subject_id=subject_id,
        structurizer_id="test-structurizer",
        structurizer_version="1.0.0",
    )


def _particle(
    content: str,
    claim: StructuredClaim | None,
    confidence: float = 0.8,
    subject_ids: list[str] | None = None,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        subject_ids=subject_ids or [],
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
        structured_claim=claim,
    )


async def _seed(session: Session) -> None:
    """Three claim particles + one un-annotated particle.

    Weights 3.5 / 3.9 / "about four grams" (untyped) under ``nmo:hasWeight``,
    one ``nmo:hasMaterial`` claim, one particle with no claim at all.
    """
    from particles.store.particle_store import insert_particle

    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    await insert_particle(session, _particle("Weighs 3.5 g.", _claim(obj_value="3.5"), 0.9), emb)
    await insert_particle(session, _particle("Weighs 3.9 g.", _claim(obj_value="3.9"), 0.8), emb)
    await insert_particle(
        session,
        _particle(
            "Weighs about four grams.", _claim(obj_value="about four grams", obj_datatype=None), 0.7
        ),
        emb,
    )
    await insert_particle(
        session,
        _particle("Made of copper.", _claim(predicate="nmo:hasMaterial", obj_value="copper"), 0.6),
        emb,
    )
    await insert_particle(session, _particle("No claim here.", None, 0.5), emb)
    await session.commit()


class _Spies:
    def __init__(self) -> None:
        self.embed = MagicMock(name="embedding-model")
        self.embed.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
        import anthropic

        self.llm = MagicMock(spec=anthropic.Anthropic)
        self.llm.messages = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "answer text"
        mock_resp = MagicMock()
        mock_resp.content = [mock_content]
        self.llm.messages.create = MagicMock(return_value=mock_resp)


@pytest.fixture
def spies() -> Any:
    """Spy embedding model + LLM client installed for the test's duration."""
    from particles import embeddings as ep
    from particles.llm import set_client

    s = _Spies()
    original = ep._embedding_model
    ep.set_embedding_model(s.embed)
    set_client(s.llm)
    try:
        yield s
    finally:
        ep.set_embedding_model(original)
        set_client(None)


# ---------------------------------------------------------------------------
# Mode dispatch + request validation
# ---------------------------------------------------------------------------


class TestModeValidation:
    def test_no_question_no_filters_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs a question"):
            QueryRequest()

    def test_aggregate_rejects_simultaneous_question(self) -> None:
        with pytest.raises(ValueError, match="adds nothing but risk"):
            QueryRequest(question="how many?", count=True)

    def test_group_by_rejects_simultaneous_question(self) -> None:
        with pytest.raises(ValueError, match="adds nothing but risk"):
            QueryRequest(question="how many?", group_by=StructuralGroupBy.SUBJECT)

    def test_list_predicates_is_standalone(self) -> None:
        with pytest.raises(ValueError, match="standalone"):
            QueryRequest(list_predicates=True, predicate="nmo:hasWeight")

    def test_min_effective_confidence_is_aggregate_only(self) -> None:
        with pytest.raises(ValueError, match="aggregate modes"):
            QueryRequest(predicate="nmo:hasWeight", min_effective_confidence=0.5)

    def test_non_comparable_bound_rejected(self) -> None:
        with pytest.raises(ValueError, match="neither a number"):
            QueryRequest(object_gt="heavy")

    def test_structural_mode_flags(self) -> None:
        assert QueryRequest(predicate="p").is_structural_mode
        assert QueryRequest(count=True).is_structural_mode
        assert QueryRequest(list_predicates=True).is_structural_mode
        assert not QueryRequest(question="q", predicate="p").is_structural_mode
        assert not QueryRequest(question="q").is_structural_mode


# ---------------------------------------------------------------------------
# Deterministic listing mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_listing_orders_by_effective_confidence(
    db_session: Session, spies: _Spies
) -> None:
    await _seed(db_session)
    from particles.operations.query import query

    result = await query(db_session, QueryRequest(predicate="nmo:hasWeight"))
    contents = [p.content for p in result.particles]
    assert contents == ["Weighs 3.5 g.", "Weighs 3.9 g.", "Weighs about four grams."]
    assert result.effective_confidences == sorted(result.effective_confidences, reverse=True)
    assert "claims" in result.answer


@pytest.mark.asyncio
async def test_deterministic_path_makes_zero_llm_and_embedding_calls(
    db_session: Session, spies: _Spies
) -> None:
    await _seed(db_session)
    from particles.operations.query import query

    result = await query(db_session, QueryRequest(predicate="nmo:hasWeight", object_gt="3"))
    assert len(result.particles) == 2
    assert spies.embed.encode.call_count == 0
    assert spies.llm.messages.create.call_count == 0

    await query(db_session, QueryRequest(count=True))
    await query(db_session, QueryRequest(list_predicates=True))
    assert spies.embed.encode.call_count == 0
    assert spies.llm.messages.create.call_count == 0


@pytest.mark.asyncio
async def test_listing_finds_unembedded_claim_particles(db_session: Session, spies: _Spies) -> None:
    """The structural fetch must not require an embedding."""
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    await insert_particle(
        db_session, _particle("Unembedded claim.", _claim(obj_value="5.0"), 0.8), None
    )
    await db_session.commit()
    result = await query(db_session, QueryRequest(predicate="nmo:hasWeight"))
    assert [p.content for p in result.particles] == ["Unembedded claim."]


@pytest.mark.asyncio
async def test_gt_excludes_and_discloses_non_normalizable_objects(
    db_session: Session, spies: _Spies
) -> None:
    await _seed(db_session)
    from particles.operations.query import query

    result = await query(db_session, QueryRequest(predicate="nmo:hasWeight", object_gt="3.6"))
    assert [p.content for p in result.particles] == ["Weighs 3.9 g."]
    assert result.claim_coverage is not None
    assert result.claim_coverage.not_normalizable_excluded == 1
    lines = disclosure_lines(result.claim_coverage)
    assert any("would not normalize" in line for line in lines)


# ---------------------------------------------------------------------------
# Coverage footer (§2.6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coverage_footer_counts_and_wording(db_session: Session, spies: _Spies) -> None:
    await _seed(db_session)
    from particles.operations.query import query

    result = await query(db_session, QueryRequest(predicate="nmo:hasWeight"))
    cov = result.claim_coverage
    assert cov is not None
    assert cov.active_total == 5
    assert cov.with_claims == 4
    assert cov.matched == 3
    line = coverage_line(cov)
    assert line == (
        "Matched against the 4 of 5 ACTIVE particles carrying a structured "
        "claim (store coverage 80%)."
    )


# ---------------------------------------------------------------------------
# Aggregates (§2.5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_counts_claims_and_discloses_distribution(
    db_session: Session, spies: _Spies
) -> None:
    await _seed(db_session)
    from particles.operations.query import query

    result = await query(db_session, QueryRequest(predicate="nmo:hasWeight", count=True))
    agg = result.structural_aggregate
    assert agg is not None
    assert agg.claim_count == 3
    assert agg.min_effective_confidence is not None
    assert agg.median_effective_confidence is not None
    assert agg.max_effective_confidence is not None
    assert agg.min_effective_confidence <= agg.median_effective_confidence
    assert agg.median_effective_confidence <= agg.max_effective_confidence
    # Epistemics commitment (§2.5): counts speak of claims, never "facts".
    assert "claims" in result.answer
    assert "fact" not in result.answer.lower()


@pytest.mark.asyncio
async def test_count_has_no_default_confidence_floor(db_session: Session, spies: _Spies) -> None:
    """A low-confidence claim is counted unless the caller sets the floor."""
    await _seed(db_session)
    from particles.operations.query import query

    everything = await query(db_session, QueryRequest(count=True))
    assert everything.structural_aggregate is not None
    assert everything.structural_aggregate.claim_count == 4

    floored = await query(db_session, QueryRequest(count=True, min_effective_confidence=0.65))
    agg = floored.structural_aggregate
    assert agg is not None
    assert agg.claim_count < 4
    assert floored.claim_coverage is not None
    assert floored.claim_coverage.below_min_effective_confidence == 4 - agg.claim_count


@pytest.mark.asyncio
async def test_group_by_predicate(db_session: Session, spies: _Spies) -> None:
    await _seed(db_session)
    from particles.operations.query import query

    result = await query(db_session, QueryRequest(group_by=StructuralGroupBy.PREDICATE))
    agg = result.structural_aggregate
    assert agg is not None
    by_key = {b.key: b.claim_count for b in agg.buckets}
    assert by_key == {"nmo:hasWeight": 3, "nmo:hasMaterial": 1}


@pytest.mark.asyncio
async def test_group_by_subject_falls_back_to_particle_subjects_link(
    db_session: Session, spies: _Spies
) -> None:
    """§2.5: claim.subject_id when present, else the particle_subjects link."""
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject

    subject_a = Subject(canonical_name="Coin A", asserted_by="test-agent")
    subject_b = Subject(canonical_name="Coin B", asserted_by="test-agent")
    await insert_subject(db_session, subject_a)
    await insert_subject(db_session, subject_b)
    # Claim-level resolution present:
    await insert_particle(
        db_session,
        _particle("A weighs 3.5 g.", _claim(obj_value="3.5", subject_id=subject_a.id)),
    )
    # Claim-level None, particle linked to subject B — the fallback join:
    await insert_particle(
        db_session,
        _particle("B weighs 3.9 g.", _claim(obj_value="3.9"), subject_ids=[subject_b.id]),
    )
    # Neither: buckets under "(no subject)".
    await insert_particle(db_session, _particle("Orphan weighs 1 g.", _claim(obj_value="1.0")))
    await db_session.commit()

    result = await query(db_session, QueryRequest(group_by=StructuralGroupBy.SUBJECT))
    agg = result.structural_aggregate
    assert agg is not None
    by_key = {b.key: b for b in agg.buckets}
    assert by_key[subject_a.id].claim_count == 1
    assert by_key[subject_a.id].label == "Coin A"
    assert by_key[subject_b.id].claim_count == 1
    assert by_key[subject_b.id].label == "Coin B"
    assert by_key["(no subject)"].claim_count == 1


# ---------------------------------------------------------------------------
# Predicate vocabulary listing (§2.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predicates_listing(db_session: Session, spies: _Spies) -> None:
    await _seed(db_session)
    from particles.operations.query import query

    result = await query(db_session, QueryRequest(list_predicates=True))
    vocab = {(v.value, v.kind.value): v.claim_count for v in result.predicate_vocabulary}
    assert vocab == {("nmo:hasWeight", "URI"): 3, ("nmo:hasMaterial", "URI"): 1}
    assert "claims" in result.answer
    assert result.claim_coverage is not None
    assert spies.embed.encode.call_count == 0
    assert spies.llm.messages.create.call_count == 0


# ---------------------------------------------------------------------------
# Prefilter mode: question + flags (§2.1 mode two, §2.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefilter_intersection_leaves_semantic_ranking_unchanged(
    db_session: Session, spies: _Spies
) -> None:
    """The flags narrow the candidate set; scores and relative order of the
    surviving particles are byte-identical to the unfiltered run (§2.4)."""
    await _seed(db_session)
    from particles.operations.query import query

    unfiltered = await query(db_session, QueryRequest(question="how heavy is the coin?"))
    filtered = await query(
        db_session,
        QueryRequest(question="how heavy is the coin?", predicate="nmo:hasWeight"),
    )

    filtered_ids = [p.id for p in filtered.particles]
    unfiltered_scores = dict(
        zip((p.id for p in unfiltered.particles), unfiltered.effective_confidences, strict=True)
    )
    # Every filtered hit carries the matching claim.
    for p in filtered.particles:
        assert p.structured_claim is not None
        assert p.structured_claim.predicate.value == "nmo:hasWeight"
    # The surviving particles keep the unfiltered run's relative order …
    unfiltered_order = [p.id for p in unfiltered.particles if p.id in set(filtered_ids)]
    assert filtered_ids == unfiltered_order
    # … and their effective confidences are untouched.
    for pid, eff in zip(filtered_ids, filtered.effective_confidences, strict=True):
        assert eff == pytest.approx(unfiltered_scores[pid])
    # The LLM answered (semantic mode), and the coverage footer rides along.
    assert spies.llm.messages.create.call_count == 2
    assert filtered.claim_coverage is not None
    assert filtered.claim_coverage.matched == 3
    assert unfiltered.claim_coverage is None
