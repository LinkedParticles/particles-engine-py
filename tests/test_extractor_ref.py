"""Tests for the modelled ExtractorRef.

`extractor_ref` was an untyped ``dict[str, Any]`` through 1.109.x, with its
``{name, version}`` key set stated only in a JSON Schema ``description``
string — so nothing checked it at any of the three validation layers. These
cover the four surfaces the ADR pins it at, plus the two strictness decisions
the owner made at sign-off.

1. The Core model: both keys required, both non-empty.
2. Storage: a malformed stored payload **coerces to None with a warning**
   rather than raising — the read path may not wedge on one legacy row.
3. Interchange: the ``extractorName`` / ``extractorVersion`` spelling, and the
   opposite strictness — a malformed *unit* raises, because an import is
   something the operator can reject and re-request.
4. The artifacts: JSON Schema ``$defs/ExtractorRef`` and the SHACL
   ``ExtractorRefShape``, including that an absent ref stays legal (§9.1a).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from particles.core.schema import (
    Confidence,
    ExtractorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.store.particle_store import ParticleRow, insert_particle


def _particle(ref: ExtractorRef | None = None) -> Particle:
    return Particle(
        content="A claim.",
        confidence=Confidence(value=0.8),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        extractor_ref=ref,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="entry-1",
                snapshot_id="snap-1",
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. The Core model
# ---------------------------------------------------------------------------


class TestExtractorRefModel:
    def test_plain_mapping_still_validates_into_the_model(self) -> None:
        """The 1.109.x call shape — a bare dict — keeps working."""
        p = _particle().model_copy(
            update={"extractor_ref": ExtractorRef(name="general-extractor", version="0.13.0")}
        )
        assert p.extractor_ref == ExtractorRef(name="general-extractor", version="0.13.0")

        via_dict = Particle(
            content="A claim.",
            confidence=Confidence(value=0.8),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="x",
            extractor_ref={"name": "general-extractor", "version": "0.13.0"},  # type: ignore[arg-type]
        )
        assert via_dict.extractor_ref is not None
        assert via_dict.extractor_ref.name == "general-extractor"

    @pytest.mark.parametrize(
        "payload",
        [
            {"name": "general-extractor"},  # no version
            {"version": "0.13.0"},  # no name
            {"id": "general-extractor", "version": "0.13.0"},  # wrong key
            {"name": "", "version": "0.13.0"},  # empty name
            {"name": "general-extractor", "version": ""},  # empty version
            {},
        ],
    )
    def test_half_refs_and_wrong_keys_are_rejected(self, payload: dict[str, str]) -> None:
        """A half-ref serves neither operation the ref exists for.

        ``name`` alone cannot scope re-extraction (§9.5); ``version`` alone
        cannot join the extractor registry (§14.3). The wrong-key case is not
        hypothetical — two test fixtures in this repo carried ``{"id": ...}``
        and validated fine until the field was modelled.
        """
        with pytest.raises(ValidationError):
            ExtractorRef(**payload)

    def test_ref_is_frozen(self) -> None:
        ref = ExtractorRef(name="general-extractor", version="0.13.0")
        with pytest.raises(ValidationError):
            ref.name = "other"  # type: ignore[misc]

    def test_absent_ref_is_legal(self) -> None:
        """A direct operator/agent assertion has no extractor (§9.1a)."""
        assert _particle().extractor_ref is None


# ---------------------------------------------------------------------------
# 2. Storage — the coerce decision
# ---------------------------------------------------------------------------


class TestStorageRoundTrip:
    @pytest.mark.asyncio
    async def test_round_trip(self, db_session: Any) -> None:
        p = _particle(ExtractorRef(name="github-gist-extractor", version="0.6.1"))
        await insert_particle(db_session, p, embedding=None)
        await db_session.commit()

        row = await db_session.get(ParticleRow, p.id)
        assert json.loads(row.extractor_ref_json) == {
            "name": "github-gist-extractor",
            "version": "0.6.1",
        }
        assert row.to_model().extractor_ref == ExtractorRef(
            name="github-gist-extractor", version="0.6.1"
        )

    def test_malformed_stored_ref_coerces_to_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """Owner decision at sign-off: coerce, do not raise.

        A malformed stored row is history the operator cannot re-request, and
        a read path that raised would take out query, export and lint alike
        over a field whose absence every consumer already handles. The warning
        names the particle so the row is findable and re-extractable.
        """
        row = ParticleRow.from_model(_particle(ExtractorRef(name="general", version="1.0.0")))
        row.extractor_ref_json = json.dumps({"id": "general", "ver": "1.0"})

        with caplog.at_level("WARNING"):
            model = row.to_model()

        assert model.extractor_ref is None
        assert "malformed extractor_ref" in caplog.text
        assert row.id in caplog.text

    def test_malformed_stored_ref_does_not_wedge_the_read(self) -> None:
        """The whole point of coercing: everything else on the row survives."""
        p = _particle(ExtractorRef(name="general", version="1.0.0"))
        row = ParticleRow.from_model(p)
        row.extractor_ref_json = "not even json"

        model = row.to_model()
        assert model.extractor_ref is None
        assert model.content == p.content
        assert model.confidence.value == p.confidence.value


# ---------------------------------------------------------------------------
# 3. Interchange — the straight rename, and the opposite strictness
# ---------------------------------------------------------------------------


class TestInterchange:
    def test_unit_uses_the_spelled_out_sub_terms(self) -> None:
        """bare ``name`` / ``version`` would collide in a shared context."""
        from particles.interchange.codec import to_unit

        unit = to_unit(_particle(ExtractorRef(name="general-extractor", version="0.13.0")), {})
        assert unit["extractorRef"] == {
            "extractorName": "general-extractor",
            "extractorVersion": "0.13.0",
        }

    def test_round_trip(self) -> None:
        from particles.interchange.codec import from_unit, to_unit

        p = _particle(ExtractorRef(name="general-extractor", version="0.13.0"))
        assert from_unit(to_unit(p, {})).particle.extractor_ref == p.extractor_ref

    def test_absent_ref_emits_no_key(self) -> None:
        from particles.interchange.codec import from_unit, to_unit

        unit = to_unit(_particle(), {})
        assert "extractorRef" not in unit
        assert from_unit(unit).particle.extractor_ref is None

    def test_malformed_unit_raises(self) -> None:
        """Unlike the store's read path — a bad import is re-requestable."""
        from particles.interchange.codec import from_unit, to_unit

        unit = to_unit(_particle(ExtractorRef(name="general", version="1.0.0")), {})
        unit["extractorRef"] = {"name": "general", "version": "1.0.0"}  # old spelling
        with pytest.raises(KeyError):
            from_unit(unit)


# ---------------------------------------------------------------------------
# 4. The normative artifacts
# ---------------------------------------------------------------------------


class TestJsonSchema:
    def test_well_formed_ref_validates(self) -> None:
        from particles.conformance.jsonschema import validate_particle

        assert validate_particle(_particle(ExtractorRef(name="general", version="1.0.0"))) == []

    def test_defs_entry_pins_both_keys(self) -> None:
        """The key names now live where a validator reads them, not in prose."""
        from particles.conformance._resources import schemas_dir

        schema = json.loads((schemas_dir() / "particle.schema.json").read_text())
        defs = schema["$defs"]["ExtractorRef"]
        assert defs["required"] == ["name", "version"]
        assert defs["properties"]["name"]["minLength"] == 1
        assert defs["properties"]["version"]["minLength"] == 1
        # Open by design, as for ContributorRef / StructuredClaim.
        assert "additionalProperties" not in defs

    def test_malformed_ref_dict_is_rejected(self) -> None:
        from particles.conformance.jsonschema import validate_particle_dict

        d = _particle(ExtractorRef(name="general", version="1.0.0")).model_dump(mode="json")
        d["extractor_ref"] = {"id": "general"}
        assert validate_particle_dict(d) != []


class TestShaclShape:
    """The first shape ever placed on ``extractorRef``.

    The tests that go through ``particle_to_jsonld`` assert on
    ``result.conforms``; the serializer's untyped ``assertedAt``
    / ``confidenceValue`` literals were fixed, so a real document conforms whole and no
    longer needs the message-level workaround these carried.
    The ``_doc``-built cases stay message-scoped on purpose: they are
    deliberately partial documents that omit required Particle fields.
    """

    @staticmethod
    def _skip_without_pyshacl() -> None:
        pytest.importorskip("pyshacl")

    @staticmethod
    def _ref_violations(result: Any) -> list[str]:
        return [v for v in result.violations if "ExtractorRef" in v]

    def _doc(self, name: object, version: object) -> dict[str, Any]:
        from particles.conformance.shacl import _load_context

        ref: dict[str, Any] = {}
        if name is not None:
            ref["particles:extractorName"] = name
        if version is not None:
            ref["particles:extractorVersion"] = version
        return {
            "@context": _load_context(),
            "@type": "particles:Particle",
            "particles:extractorRef": ref,
        }

    def test_well_formed_ref_draws_no_violation(self) -> None:
        self._skip_without_pyshacl()
        from particles.conformance.shacl import particle_to_jsonld, validate_particle

        doc = particle_to_jsonld(_particle(ExtractorRef(name="general", version="0.13.0")))
        result = validate_particle(doc)
        assert result.conforms is True, result.violations

    def test_absent_ref_draws_no_violation(self) -> None:
        """No sh:minCount — an operator assertion has no ref and is valid."""
        self._skip_without_pyshacl()
        from particles.conformance.shacl import particle_to_jsonld, validate_particle

        result = validate_particle(particle_to_jsonld(_particle()))
        assert result.conforms is True, result.violations

    def test_missing_name_is_a_violation(self) -> None:
        self._skip_without_pyshacl()
        from particles.conformance.shacl import validate_particle

        result = validate_particle(self._doc(None, "0.13.0"))
        assert any("ExtractorRef.name" in v for v in result.violations), result.violations

    def test_missing_version_is_a_violation(self) -> None:
        self._skip_without_pyshacl()
        from particles.conformance.shacl import validate_particle

        result = validate_particle(self._doc("general", None))
        assert any("ExtractorRef.version" in v for v in result.violations), result.violations

    def test_non_semver_version_is_a_violation(self) -> None:
        """Owner decision at sign-off: Violation, not Warning.

        §14.3 builds re-extraction eligibility on *ordering* extractor
        versions; one that cannot be ordered breaks that contract rather than
        degrading it. The accepted cost is that a date-versioned third-party
        extractor is non-conformant on this field.
        """
        self._skip_without_pyshacl()
        from particles.conformance.shacl import validate_particle

        result = validate_particle(self._doc("general", "2026-08-01"))
        assert any("ExtractorRef.version" in v for v in result.violations), result.violations

    def test_semver_with_prerelease_suffix_draws_no_violation(self) -> None:
        self._skip_without_pyshacl()
        from particles.conformance.shacl import validate_particle

        result = validate_particle(self._doc("general", "1.0.0-rc1"))
        assert self._ref_violations(result) == []
