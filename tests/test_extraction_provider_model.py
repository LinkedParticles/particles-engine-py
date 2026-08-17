"""Tests for per-particle extraction-model provenance.

Covers the surfaces the ADR adds, in the order a reader meets them:

1. The Core model field and its ``None``-means-UNRECORDED semantics.
2. Storage round-trip, including that ``NULL`` is never grandfathered to a
   default pairing (the one place it diverges from the embedding-marker
   precedent it otherwise follows).
3. The exact-equality selector — specifically that a *nesting* pairing does
   not false-hit, which is the reason the stamp got its own column instead
   of joining ``extractor_ref``'s JSON blob (§2.1).
4. The call-site stamp, and the corollary that a deterministic extractor is
   unstamped by construction (§2.5).
5. Interchange round-trip (§2.8) and the reindex scope (§2.7).

The LLM call itself stays out of scope per ``tests/AGENTS.md`` § Mocking
strategy; what is exercised here is that the seam propagates the pairing it
is handed.
"""

from __future__ import annotations

import pytest

from particles.core.schema import (
    Confidence,
    ExtractorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    UncertaintyNature,
)
from particles.extraction.general import CandidateParticle, candidate_to_particle
from particles.store.particle_store import (
    ParticleRow,
    get_active_particles_with_provider_model,
    insert_particle,
)

_LUNA = "openai:gpt-5.6-luna"
_CLAUDE = "anthropic:claude-sonnet-4-6"


def _particle(
    content: str = "A claim.",
    provider_model: str | None = None,
    snapshot_id: str | None = None,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.8),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        extraction_provider_model=provider_model,
        provenance=(
            [
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id="entry-1",
                    snapshot_id=snapshot_id,
                )
            ]
            if snapshot_id
            else []
        ),
    )


# ---------------------------------------------------------------------------
# 1. The Core model
# ---------------------------------------------------------------------------


def test_field_defaults_to_none() -> None:
    """Absence is the default, so every pre-0229 particle deserializes clean."""
    assert _particle().extraction_provider_model is None


def test_field_round_trips_through_pydantic() -> None:
    dumped = _particle(provider_model=_LUNA).model_dump()
    assert dumped["extraction_provider_model"] == _LUNA
    assert Particle.model_validate(dumped).extraction_provider_model == _LUNA


def test_stamp_is_a_sibling_of_extractor_ref_not_a_member() -> None:
    """§2.1: a model swap must not read as an extractor upgrade.

    The pairing lives beside ``extractor_ref``, never inside it — otherwise
    ``reindex --extractor-version`` and techspec §14.3 would see a changed
    extractor reference every time the model changed.
    """
    p = _particle(provider_model=_LUNA)
    p.extractor_ref = ExtractorRef(name="general-extractor", version="0.11.0")
    assert not hasattr(p.extractor_ref, "provider_model")
    assert set(type(p.extractor_ref).model_fields) == {"name", "version"}
    assert p.extraction_provider_model == _LUNA


# ---------------------------------------------------------------------------
# 2. Storage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_storage_round_trip_preserves_the_stamp(db_session) -> None:  # type: ignore[no-untyped-def]
    particle = _particle(provider_model=_LUNA)
    await insert_particle(db_session, particle)

    row = await db_session.get(ParticleRow, particle.id)
    assert row.extraction_provider_model == _LUNA
    assert row.to_model().extraction_provider_model == _LUNA


@pytest.mark.asyncio
async def test_null_stamp_is_never_grandfathered(db_session) -> None:  # type: ignore[no-untyped-def]
    """§2.6: unlike ``embedding_model_id``, NULL gets no legacy default.

    That precedent could coalesce because one known model had embedded every
    pre-existing row. Here the pre-stamp population is already model-mixed, so
    any default would be wrong for roughly half of it.
    """
    particle = _particle()
    await insert_particle(db_session, particle)

    row = await db_session.get(ParticleRow, particle.id)
    assert row.extraction_provider_model is None
    assert row.to_model().extraction_provider_model is None


# ---------------------------------------------------------------------------
# 3. The selector — exact equality, and why
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selector_matches_only_the_exact_pairing(db_session) -> None:  # type: ignore[no-untyped-def]
    luna = _particle(content="Luna claim.", provider_model=_LUNA)
    claude = _particle(content="Claude claim.", provider_model=_CLAUDE)
    await insert_particle(db_session, luna)
    await insert_particle(db_session, claude)

    found = await get_active_particles_with_provider_model(db_session, _LUNA)
    assert [p.id for p in found] == [luna.id]


@pytest.mark.asyncio
async def test_selector_does_not_false_hit_on_a_nesting_pairing(db_session) -> None:  # type: ignore[no-untyped-def]
    """The regression this design exists to prevent (§2.1).

    ``openai:gpt-5.6`` is a *prefix* of ``openai:gpt-5.6-luna``. A substring
    scope — which is what the two ``extractor_ref``-based reindex helpers do
     — would sweep in the sibling model the operator is trying to
    keep, undoing good particles along with the bad.
    """
    luna = _particle(content="Luna claim.", provider_model=_LUNA)
    await insert_particle(db_session, luna)

    assert await get_active_particles_with_provider_model(db_session, "openai:gpt-5.6") == []


@pytest.mark.asyncio
async def test_unstamped_particles_never_match(db_session) -> None:  # type: ignore[no-untyped-def]
    """Unrecorded is not a pairing; a model scope may not guess at it."""
    await insert_particle(db_session, _particle(content="Unstamped."))
    assert await get_active_particles_with_provider_model(db_session, _LUNA) == []


# ---------------------------------------------------------------------------
# 4. The call-site stamp
# ---------------------------------------------------------------------------


def test_candidate_to_particle_threads_the_stamp() -> None:
    candidate = CandidateParticle(
        content="A claim.",
        confidence_value=0.8,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=[],
        provider_model=_LUNA,
    )
    particle = candidate_to_particle(candidate, "entry-1", "snap-1")
    assert particle.extraction_provider_model == _LUNA


def test_candidate_without_a_pairing_yields_an_unstamped_particle() -> None:
    candidate = CandidateParticle(
        content="A deterministic claim.",
        confidence_value=0.8,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=[],
    )
    assert candidate.provider_model is None
    assert candidate_to_particle(candidate, "e", "s").extraction_provider_model is None


@pytest.mark.asyncio
async def test_call_llm_stamps_candidates_with_the_serving_pairing() -> None:
    """§2.4: the seam stamps from the provider that actually served the call."""
    from unittest.mock import AsyncMock, patch

    from particles.extraction.general import _call_llm

    reply = '[{"content": "a claim", "subjects": [], '
    reply += '"confidence_value": 0.9, "uncertainty_nature": "EPISTEMIC"}]'
    with patch(
        "particles.llm.complete_with_provider_model",
        AsyncMock(return_value=(reply, _LUNA)),
    ):
        candidates, _notes, transient = await _call_llm("some source text")

    assert transient is False
    assert [c.provider_model for c in candidates] == [_LUNA]


@pytest.mark.asyncio
async def test_journal_seam_stamps_too() -> None:
    """The journal extractor has its own seam; it must not be forgotten."""
    from unittest.mock import AsyncMock, patch

    from particles.extraction.journal import _call_journal_llm

    reply = (
        '{"narrative_label": "A day.", "claims": [{"content": "The author rested.", '
        '"subjects": [], "confidence_value": 0.9, "uncertainty_nature": "EPISTEMIC", '
        '"assertion_modality": "EXPERIENTIAL"}]}'
    )
    with patch(
        "particles.llm.complete_with_provider_model",
        AsyncMock(return_value=(reply, _CLAUDE)),
    ):
        candidates, _notes, transient = await _call_journal_llm("Rested today.")

    assert transient is False
    assert candidates
    assert all(c.provider_model == _CLAUDE for c in candidates)


@pytest.mark.asyncio
async def test_deterministic_extractor_is_unstamped_by_construction() -> None:
    """§2.5: no capability flag, no allowlist — it simply never calls a model.

    Stamping the ambient pairing here would be a false record, and would make
    ``reindex --provider-model`` sweep in particles that model never touched.
    """
    from particles.extraction.rdf import RdfExtractor

    turtle = b'<http://example.org/a> <http://example.org/p> "v" .'
    result = await RdfExtractor().extract(Snapshot(content_hash="a" * 64), turtle)

    assert result.candidates
    assert all(c.provider_model is None for c in result.candidates)


# ---------------------------------------------------------------------------
# 5. Interchange and the reindex scope
# ---------------------------------------------------------------------------


def test_interchange_round_trips_the_stamp() -> None:
    """§2.8: the pairing is immutable substrate, so it crosses the boundary."""
    from particles.interchange.codec import from_unit, to_unit

    unit = to_unit(_particle(provider_model=_LUNA), {})
    assert unit["extractionProviderModel"] == _LUNA
    assert from_unit(unit).particle.extraction_provider_model == _LUNA


def test_interchange_omits_an_absent_stamp() -> None:
    from particles.interchange.codec import from_unit, to_unit

    unit = to_unit(_particle(), {})
    assert "extractionProviderModel" not in unit
    assert from_unit(unit).particle.extraction_provider_model is None


@pytest.mark.asyncio
async def test_reindex_scope_selects_the_pairings_snapshots(db_session) -> None:  # type: ignore[no-untyped-def]
    """§2.7: matched particles project to their SOURCE snapshots."""
    from particles.operations.reindex import _identify_scope

    luna = _particle(content="Luna claim.", provider_model=_LUNA, snapshot_id="snap-luna")
    claude = _particle(content="Claude claim.", provider_model=_CLAUDE, snapshot_id="snap-claude")
    await insert_particle(db_session, luna)
    await insert_particle(db_session, claude)

    scope = await _identify_scope(
        db_session, None, None, None, include_failed=False, provider_model=_LUNA
    )
    assert ("entry-1", "snap-luna") in scope
    assert ("entry-1", "snap-claude") not in scope
