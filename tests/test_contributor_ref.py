"""Tests for ContributorRef and the contributors field.

Covers the model invariants, persistence round-trip on all three attributable
models, interchange round-trip, JSON-Schema acceptance, and the conformance
contract tier. The field is additive-Optional Extension D/E — Core never
branches on it, so these tests pin shape + round-trip, not behavior.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from particles.conformance.contract import CONTRACT
from particles.conformance.jsonschema import validate_particle
from particles.conformance.types import FieldTier
from particles.core.schema import (
    CONTRIBUTOR_ROLES,
    Confidence,
    ContributorRef,
    CorpusEntry,
    FetchPolicy,
    Mutability,
    Particle,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource


def _ref(id: str = "github:torvalds", role: str = "author") -> ContributorRef:
    return ContributorRef(id=id, role=role)


def _particle(contributors: list[ContributorRef] | None = None) -> Particle:
    return Particle(
        content="claim",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="t",
        contributors=contributors,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TestContributorRefModel:
    def test_fields_and_default_timestamp(self) -> None:
        r = _ref()
        assert r.id == "github:torvalds"
        assert r.role == "author"
        assert isinstance(r.at, datetime)

    def test_frozen(self) -> None:
        r = _ref()
        with pytest.raises(ValidationError):
            r.id = "other"  # type: ignore[misc]

    def test_id_and_role_min_length(self) -> None:
        with pytest.raises(ValidationError):
            ContributorRef(id="", role="author")
        with pytest.raises(ValidationError):
            ContributorRef(id="x", role="")

    def test_recommended_roles_constant(self) -> None:
        assert {"author", "extractor", "curator", "reviewer", "importer", "agent"} == set(
            CONTRIBUTOR_ROLES
        )

    def test_open_vocabulary_accepts_unknown_role(self) -> None:
        # Open set: an unrecognized role is valid (no enum / validator).
        assert ContributorRef(id="did:web:x", role="notary").role == "notary"

    def test_default_is_none_on_all_three_models(self) -> None:
        assert _particle().contributors is None
        assert Subject(canonical_name="S", asserted_by="t").contributors is None
        assert (
            CorpusEntry(
                source_type="WEB_PAGE",
                mutability=Mutability.STABLE,
                fetch_policy=FetchPolicy.NEVER,
                deposited_by="t",
            ).contributors
            is None
        )


# ---------------------------------------------------------------------------
# Persistence round-trip (from_model → to_model)
# ---------------------------------------------------------------------------


class TestPersistenceRoundTrip:
    def test_particle_row_round_trip(self) -> None:
        from particles.store.particle_store import ParticleRow

        p = _particle([_ref(), _ref("reddit:u/jeff", "curator")])
        back = ParticleRow.from_model(p).to_model()
        assert back.contributors is not None
        assert [(c.id, c.role) for c in back.contributors] == [
            ("github:torvalds", "author"),
            ("reddit:u/jeff", "curator"),
        ]
        # at survives the ISO round-trip as a tz-aware datetime
        assert back.contributors[0].at == p.contributors[0].at

    def test_particle_row_none_round_trip(self) -> None:
        from particles.store.particle_store import ParticleRow

        row = ParticleRow.from_model(_particle(None))
        assert row.contributors_json is None
        assert row.to_model().contributors is None

    def test_subject_row_round_trip(self) -> None:
        from particles.store.subject_store import SubjectRow

        s = Subject(canonical_name="S", asserted_by="t", contributors=[_ref(role="curator")])
        back = SubjectRow.from_model(s).to_model()
        assert back.contributors is not None
        assert back.contributors[0].role == "curator"

    def test_corpus_entry_row_round_trip(self) -> None:
        from particles.corpus.store import CorpusEntryRow

        e = CorpusEntry(
            source_type="WEB_PAGE",
            mutability=Mutability.STABLE,
            fetch_policy=FetchPolicy.NEVER,
            deposited_by="t",
            contributors=[_ref(role="importer")],
        )
        back = CorpusEntryRow.from_model(e).to_model()
        assert back.contributors is not None
        assert back.contributors[0].role == "importer"


@pytest.mark.asyncio
async def test_particle_db_round_trip(db_session: object) -> None:
    from particles.store.particle_store import get_particle, insert_particle

    session = db_session  # type: ignore[assignment]
    p = _particle([_ref()])
    await insert_particle(session, p)  # type: ignore[arg-type]
    loaded = await get_particle(session, p.id)  # type: ignore[arg-type]
    assert loaded is not None
    assert loaded.contributors is not None
    assert loaded.contributors[0].id == "github:torvalds"


# ---------------------------------------------------------------------------
# Interchange round-trip
# ---------------------------------------------------------------------------


class TestInterchangeRoundTrip:
    def test_to_unit_from_unit_preserves_contributors(self) -> None:
        from particles.interchange.codec import from_unit, to_unit

        p = _particle([_ref(), _ref("particles-store:acme", "importer")])
        unit = to_unit(p, subjects={})
        assert "contributors" in unit
        parsed = from_unit(unit)
        assert parsed.particle.contributors is not None
        assert [(c.id, c.role) for c in parsed.particle.contributors] == [
            ("github:torvalds", "author"),
            ("particles-store:acme", "importer"),
        ]

    def test_none_is_omitted_from_unit(self) -> None:
        from particles.interchange.codec import from_unit, to_unit

        unit = to_unit(_particle(None), subjects={})
        assert "contributors" not in unit
        assert from_unit(unit).particle.contributors is None


# ---------------------------------------------------------------------------
# Conformance + JSON Schema
# ---------------------------------------------------------------------------


class TestConformance:
    def test_jsonschema_accepts_contributors(self) -> None:
        assert validate_particle(_particle([_ref()])) == []

    def test_jsonschema_accepts_none(self) -> None:
        assert validate_particle(_particle(None)) == []

    def test_contract_lists_contributors_as_optional(self) -> None:
        entry = next((c for c in CONTRACT if c.field == "contributors"), None)
        assert entry is not None
        assert entry.tier is FieldTier.OPTIONAL
