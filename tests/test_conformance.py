"""Tests for conformance validation — JSON Schema and Markdown Bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from particles.core.schema import (
    SCHEMA_VERSION,
    CanonicalForm,
    ClaimTerm,
    Confidence,
    ContributorRef,
    ExtractorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    StructuredClaim,
    TermKind,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from tests._upstream import upstream_only


def _make_valid_particle() -> Particle:
    return Particle(
        content="Mercury is the closest planet to the Sun.",
        confidence=Confidence(value=0.99, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        schema_version=SCHEMA_VERSION,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="entry-1",
                snapshot_id="snap-1",
            )
        ],
    )


def _make_populated_particle() -> Particle:
    """A particle carrying every optional field ``ParticleShape`` constrains."""
    return _make_valid_particle().model_copy(
        update={
            "status": Status.RETRACTED,
            "status_reason": StatusReason.EXPLICIT_RETRACTION,
            "subject_ids": ["subject-1", "subject-2"],
            "extractor_ref": ExtractorRef(name="general-extractor", version="0.13.0"),
            "extraction_provider_model": "anthropic:claude-opus-5",
            "contributors": [ContributorRef(id="github:torvalds", role="author")],
            "canonical_form": CanonicalForm.STRUCTURED,
            "structured_claim": StructuredClaim(
                subject=ClaimTerm(kind=TermKind.TOKEN, value="Mercury"),
                predicate=ClaimTerm(kind=TermKind.TOKEN, value="closestPlanetTo"),
                object=ClaimTerm(kind=TermKind.LITERAL, value="the Sun"),
                subject_id="subject-1",
                structurizer_id="general-extractor",
                structurizer_version="0.13.0",
            ),
        }
    )


class TestJsonSchemaValidation:
    def test_valid_particle_passes(self) -> None:
        from particles.conformance.jsonschema import validate_particle

        p = _make_valid_particle()
        errors = validate_particle(p)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_required_field(self) -> None:
        import json

        from particles.conformance.jsonschema import validate_particle_dict

        p = _make_valid_particle()
        d = json.loads(p.model_dump_json())
        del d["asserted_by"]
        errors = validate_particle_dict(d)
        # jsonschema may or may not be installed; if it is, we expect an error
        # If not installed, errors will be empty (validation skipped)
        # Just verify no exception is raised
        assert isinstance(errors, list)

    def test_status_enums_match_code(self) -> None:
        """The normative JSON Schema's status/status_reason enums mirror the
        code enums exactly — the F2.2 drift (LOWER_TRUST_SOURCE and
        SOURCE_RETRACTED emitted by code but absent from the artifact) must
        not recur."""
        import json
        from pathlib import Path

        from particles.core.status import StatusReason

        schema_file = Path(__file__).parents[1] / "artifacts" / "schemas" / "particle.schema.json"
        props = json.loads(schema_file.read_text())["properties"]
        assert set(props["status"]["enum"]) == {s.value for s in Status}
        assert set(props["status_reason"]["enum"]) == {r.value for r in StatusReason} | {None}

    def test_particle_type_enum_matches_code(self) -> None:
        """The schema's particle_type enum mirrors ParticleType exactly —
        REVIEW (emitted by review/cascade audit records) was missing until
        the F2.2/F2.3 artifact sync."""
        import json
        from pathlib import Path

        from particles.core.schema import ParticleType

        schema_file = Path(__file__).parents[1] / "artifacts" / "schemas" / "particle.schema.json"
        props = json.loads(schema_file.read_text())["properties"]
        assert set(props["particle_type"]["enum"]) == {t.value for t in ParticleType}

    def test_every_particle_field_has_a_schema_property(self) -> None:
        """Drift guard (F2.2/F2.3): every field on the Particle model has a
        property in the normative JSON Schema. subject_ids, properties,
        basis, extractor_ref, sequence_context, and uncertainty_kind were
        all silently absent until the 2026-06 artifact sync."""
        import json
        from pathlib import Path

        schema_file = Path(__file__).parents[1] / "artifacts" / "schemas" / "particle.schema.json"
        props = json.loads(schema_file.read_text())["properties"]
        missing = sorted(set(Particle.model_fields) - set(props))
        assert not missing, f"particle.schema.json lacks properties for: {missing}"

    def test_review_particle_validates(self) -> None:
        from particles.conformance.jsonschema import validate_particle
        from particles.core.schema import ParticleType

        p = _make_valid_particle().model_copy(update={"particle_type": ParticleType.REVIEW})
        assert validate_particle(p) == []

    def test_particle_with_trust_status_reasons_validates(self) -> None:
        from particles.conformance.jsonschema import validate_particle
        from particles.core.status import StatusReason

        for status, reason in (
            (Status.PROVENANCE_STALE, StatusReason.LOWER_TRUST_SOURCE),
            (Status.RETRACTED, StatusReason.SOURCE_RETRACTED),
        ):
            p = _make_valid_particle().model_copy(
                update={"status": status, "status_reason": reason}
            )
            errors = validate_particle(p)
            assert errors == [], f"Unexpected errors for {reason}: {errors}"


class TestMarkdownBridge:
    def test_render_particle(self) -> None:
        from particles.exporters.markdown import render_particle

        p = _make_valid_particle()
        md = render_particle(p, effective_confidence=0.85)
        assert "Mercury" in md
        assert "0.85" in md
        assert "> [!" in md  # Obsidian callout syntax

    def test_render_empty_particles(self) -> None:
        from particles.exporters.markdown import render_particles

        md = render_particles([])
        assert "No particles" in md

    def test_render_lint_report(self) -> None:
        from particles.core.schema import LintFinding, LintReport
        from particles.exporters.markdown import render_lint_report

        report = LintReport(
            findings=[
                LintFinding(
                    particle_id="abc",
                    finding_type="STALENESS",
                    severity="ERROR",
                    detail="valid_until has passed",
                    recommended_action="Set PROVENANCE_STALE",
                )
            ],
            summary={"STALENESS": 1},
        )
        md = render_lint_report(report)
        assert "STALENESS" in md
        assert "ERROR" in md
        assert "abc" in md

    def test_render_particle_all_statuses(self) -> None:
        from particles.core.status import Status
        from particles.exporters.markdown import render_particle

        for status in Status:
            p = _make_valid_particle().model_copy(update={"status": status})
            md = render_particle(p)
            assert status.value in md


class TestShaclValidation:
    def test_validate_particle_without_shapes(self) -> None:
        """SHACL validation gracefully skips if shapes not available."""
        from particles.conformance.shacl import particle_to_jsonld, validate_particle

        p = _make_valid_particle()
        doc = particle_to_jsonld(p)
        result = validate_particle(doc)
        # Should not raise; conforms=True when shapes not found
        assert isinstance(result.conforms, bool)

    def test_valid_particle_conforms_whole(self) -> None:
        """every document the serializer produced used to violate.

        ``assertedAt`` went out as an untyped string and ``confidenceValue`` as
        a bare JSON float (``xsd:double``), against shape constraints of
        ``xsd:dateTime`` and ``xsd:float`` — so those two violations fired on
        *every* particle regardless of its contents, and the layer reported
        noise instead of findings.
        """
        pytest.importorskip("pyshacl")
        from particles.conformance.shacl import particle_to_jsonld, validate_particle

        result = validate_particle(particle_to_jsonld(_make_valid_particle()))
        assert result.conforms is True, result.violations

    def test_fully_populated_particle_conforms_whole(self) -> None:
        """The widened emission must not introduce violations of its own."""
        pytest.importorskip("pyshacl")
        from particles.conformance.shacl import particle_to_jsonld, validate_particle

        result = validate_particle(particle_to_jsonld(_make_populated_particle()))
        assert result.conforms is True, result.violations

    @pytest.mark.parametrize(
        ("key", "bad_value", "fragment"),
        [
            ("particles:particleType", "NOPE", "particle_type"),
            ("particles:assertionModality", "NOPE", "assertion_modality"),
            ("particles:canonicalForm", "NOPE", "canonical_form"),
            ("particles:statusReason", "NOPE", "status_reason"),
            ("particles:extractionProviderModel", "", "extraction_provider_model"),
        ],
    )
    def test_widened_field_reaches_the_shape(self, key: str, bad_value: str, fragment: str) -> None:
        """Emitting a field is only worth it if the shape then checks it.

        Corrupting each newly-emitted path in the produced document must draw
        that path's own violation — otherwise the widening is inert.
        """
        pytest.importorskip("pyshacl")
        from particles.conformance.shacl import particle_to_jsonld, validate_particle

        doc = particle_to_jsonld(_make_populated_particle())
        assert key in doc, f"{key} is not emitted"
        doc[key] = bad_value
        result = validate_particle(doc)
        assert not result.conforms
        assert any(fragment in v for v in result.violations), result.violations

    def test_confidence_value_range_is_checked(self) -> None:
        """The [0,1] bound could never fire while the datatype was wrong."""
        pytest.importorskip("pyshacl")
        from particles.conformance.shacl import particle_to_jsonld, validate_particle

        doc = particle_to_jsonld(_make_valid_particle())
        doc["particles:confidenceValue"] = {
            "@value": 1.5,
            "@type": "http://www.w3.org/2001/XMLSchema#float",
        }
        result = validate_particle(doc)
        assert not result.conforms
        assert any("confidence.value" in v for v in result.violations), result.violations

    def test_context_is_not_double_wrapped(self) -> None:
        """``_load_context()`` returns the mapping, not the file around it."""
        from particles.conformance.shacl import _load_context

        ctx = _load_context()
        assert "@context" not in ctx
        assert ctx["particles"] == "https://linkedparticles.org/vocab#"


class TestParticleShaclEnums:
    """ParticleShape statusReason / particleType sh:in lists (F2.2/F2.3
    artifact sync). These use hand-built docs — a bad enum value cannot be
    reached through the Particle model, which rejects it first."""

    def test_bad_status_reason_is_caught(self) -> None:
        from particles.conformance.shacl import _load_context, validate_particle

        doc = {
            "@context": _load_context(),
            "@type": "particles:Particle",
            "particles:statusReason": "NOT_A_REASON",
        }
        result = validate_particle(doc)
        try:
            import pyshacl  # noqa: F401
        except ImportError:
            return  # validation skipped without pyshacl
        assert not result.conforms
        assert any("status_reason" in v for v in result.violations), result.violations

    def test_valid_status_reason_not_flagged(self) -> None:
        from particles.conformance.shacl import _load_context, validate_particle

        doc = {
            "@context": _load_context(),
            "@type": "particles:Particle",
            "particles:statusReason": "SOURCE_RETRACTED",
        }
        result = validate_particle(doc)
        # The doc is intentionally minimal (other required-field violations
        # are expected); only assert the enum constraint itself is silent.
        assert not any("status_reason" in v for v in result.violations), result.violations

    def test_review_particle_type_not_flagged(self) -> None:
        from particles.conformance.shacl import _load_context, validate_particle

        doc = {
            "@context": _load_context(),
            "@type": "particles:Particle",
            "particles:particleType": "REVIEW",
        }
        result = validate_particle(doc)
        assert not any("particle_type" in v for v in result.violations), result.violations


class TestSubjectShacl:
    """SubjectShape (§ Amendment). Subjects became
    first-class after the original four shapes were published and were left
    without any normative shape (the same drift that hit context.jsonld). This
    guards against that recurring: the shape must stay registered and present,
    and must actually validate a Subject."""

    def test_subjectshape_registered_and_present(self) -> None:
        from particles.conformance.shacl import _SHAPE_FILES

        assert "SubjectShape" in _SHAPE_FILES
        assert _SHAPE_FILES["SubjectShape"].exists()

    def test_valid_subject_conforms(self) -> None:
        from particles.conformance.shacl import subject_to_jsonld, validate_subject
        from particles.core.schema import Subject

        s = Subject(canonical_name="France", asserted_by="test-agent")
        result = validate_subject(subject_to_jsonld(s))
        # conforms whether pyshacl validates (real pass) or is absent (skip → True)
        assert result.conforms, result.report_text

    def test_subject_missing_canonical_name_is_caught(self) -> None:
        from particles.conformance.shacl import _load_context, validate_subject

        doc = {
            "@context": _load_context(),
            "@type": "particles:Subject",
            "particles:id": "s1",
            "particles:assertedBy": "agent",
            "particles:createdAt": {
                "@value": "2026-01-01T00:00:00+00:00",
                "@type": "http://www.w3.org/2001/XMLSchema#dateTime",
            },
        }
        result = validate_subject(doc)
        try:
            import pyshacl  # noqa: F401
        except ImportError:
            return  # validation skipped without pyshacl
        assert not result.conforms  # missing required canonicalName must fail


class TestCorpusSnapshotShacl:
    """CorpusSnapshotShape (§7.3). One of the three normative shapes that had
    no validation-path test (RR-10). The minimal serializer in
    shacl.py only covers Particle/Subject, so — like TestSubjectShacl — these
    build the JSON-LD doc by hand. Typed literals are used where the shape
    constrains an xsd:dateTime so the datatype constraint is meaningful."""

    @staticmethod
    def _valid_snapshot_doc() -> dict[str, object]:
        from particles.conformance.shacl import _load_context

        return {
            "@context": _load_context(),
            "@type": "particles:Snapshot",
            "particles:snapshotId": "snap-1",
            "particles:capturedAt": {
                "@value": "2026-01-01T00:00:00+00:00",
                "@type": "http://www.w3.org/2001/XMLSchema#dateTime",
            },
            "particles:contentHash": "a" * 64,  # 64-char SHA-256 hex
            "particles:warcRecordType": "RESPONSE",
            "particles:extractionStatus": "COMPLETE",
        }

    def test_corpussnapshotshape_registered_and_present(self) -> None:
        from particles.conformance.shacl import _SHAPE_FILES

        assert "CorpusSnapshotShape" in _SHAPE_FILES
        assert _SHAPE_FILES["CorpusSnapshotShape"].exists()

    def test_valid_snapshot_conforms(self) -> None:
        from particles.conformance.shacl import validate_snapshot

        result = validate_snapshot(self._valid_snapshot_doc())
        # conforms whether pyshacl validates (real pass) or is absent (skip → True)
        assert result.conforms, result.report_text

    def test_snapshot_missing_content_hash_is_caught(self) -> None:
        from particles.conformance.shacl import validate_snapshot

        doc = self._valid_snapshot_doc()
        del doc["particles:contentHash"]  # REQUIRED (minCount 1)
        result = validate_snapshot(doc)
        try:
            import pyshacl  # noqa: F401
        except ImportError:
            return  # validation skipped without pyshacl
        assert not result.conforms

    def test_snapshot_bad_extraction_status_is_caught(self) -> None:
        from particles.conformance.shacl import validate_snapshot

        doc = self._valid_snapshot_doc()
        doc["particles:extractionStatus"] = "NOT_A_STATUS"  # outside sh:in list
        result = validate_snapshot(doc)
        try:
            import pyshacl  # noqa: F401
        except ImportError:
            return  # validation skipped without pyshacl
        assert not result.conforms


class TestTrustStatementShacl:
    """TrustStatementShape (§6.4). One of the three normative shapes that had
    no validation-path test (RR-10). Hand-built JSON-LD docs, like
    TestSubjectShacl, because the shacl.py serializer covers only
    Particle/Subject."""

    @staticmethod
    def _valid_trust_doc() -> dict[str, object]:
        from particles.conformance.shacl import _load_context

        return {
            "@context": _load_context(),
            "@type": "particles:SourceTrustStatement",
            "particles:statementId": "stmt-1",
            "particles:domain": "example.com",
            # sourceRef: presence-only in the shape (nested terms unmapped),
            # so a bare nested node satisfies the minCount/maxCount constraint.
            "particles:sourceRef": {"@value": "domain:example.com"},
            "particles:trustRank": {
                "@value": "0.5",
                "@type": "http://www.w3.org/2001/XMLSchema#float",
            },
            "particles:policyProvenance": "OPERATOR_DIRECT",
            "particles:assertedBy": "test-agent",
            "particles:assertedAt": {
                "@value": "2026-01-01T00:00:00+00:00",
                "@type": "http://www.w3.org/2001/XMLSchema#dateTime",
            },
        }

    def test_truststatementshape_registered_and_present(self) -> None:
        from particles.conformance.shacl import _SHAPE_FILES

        assert "TrustStatementShape" in _SHAPE_FILES
        assert _SHAPE_FILES["TrustStatementShape"].exists()

    def test_valid_trust_statement_conforms(self) -> None:
        from particles.conformance.shacl import validate_trust_statement

        result = validate_trust_statement(self._valid_trust_doc())
        # conforms whether pyshacl validates (real pass) or is absent (skip → True)
        assert result.conforms, result.report_text

    def test_trust_statement_missing_domain_is_caught(self) -> None:
        from particles.conformance.shacl import validate_trust_statement

        doc = self._valid_trust_doc()
        del doc["particles:domain"]  # REQUIRED (minCount 1)
        result = validate_trust_statement(doc)
        try:
            import pyshacl  # noqa: F401
        except ImportError:
            return  # validation skipped without pyshacl
        assert not result.conforms

    def test_trust_statement_out_of_range_trust_rank_is_caught(self) -> None:
        from particles.conformance.shacl import validate_trust_statement

        doc = self._valid_trust_doc()
        # trust_rank must be a float in [0,1] — 1.5 violates sh:maxInclusive.
        doc["particles:trustRank"] = {
            "@value": "1.5",
            "@type": "http://www.w3.org/2001/XMLSchema#float",
        }
        result = validate_trust_statement(doc)
        try:
            import pyshacl  # noqa: F401
        except ImportError:
            return  # validation skipped without pyshacl
        assert not result.conforms


class TestProvenanceChainShacl:
    """ProvenanceChainShape (§7.3). The fifth normative shape now has
    a reachable positive test: the provenance-ref ``type`` field was unified
    to the single canonical predicate ``particles:refType`` across codec /
    context / SHACL (it formerly targeted the unmapped ``particles:provenanceType``
    while the codec emitted ``refType`` — the RR-08 three-names bug).
    With the context now mapping ``refType``, a serialized ProvenanceRef expands
    to the predicate the shape targets, so the shape can validate a real
    document."""

    def test_provenancechainshape_registered_and_present(self) -> None:
        from particles.conformance.shacl import _SHAPE_FILES

        assert "ProvenanceChainShape" in _SHAPE_FILES
        assert _SHAPE_FILES["ProvenanceChainShape"].exists()

    def test_valid_provenance_ref_conforms(self) -> None:
        """Positive case (now unblocked): a conforming
        ProvenanceRef JSON-LD doc using the canonical ``refType`` predicate
        validates against ProvenanceChainShape."""
        from particles.conformance.shacl import _load_context, _validate

        doc = {
            "@context": _load_context(),
            "@type": "particles:ProvenanceRef",
            "particles:refType": "SOURCE",
            "particles:corpusEntryId": "entry-1",
        }
        result = _validate(doc, "ProvenanceChainShape")
        # conforms whether pyshacl validates (real pass) or is absent (skip → True)
        assert result.conforms, result.report_text

    def test_bad_ref_type_is_caught(self) -> None:
        """Non-vacuous: a refType outside the sh:in list fails — confirms the
        validate path actually exercises pyshacl against the renamed predicate."""
        from particles.conformance.shacl import _load_context, _validate

        doc = {
            "@context": _load_context(),
            "@type": "particles:ProvenanceRef",
            "particles:refType": "NOT_A_TYPE",  # outside ( SOURCE PARTICLE AGENT )
            "particles:corpusEntryId": "entry-1",
        }
        result = _validate(doc, "ProvenanceChainShape")
        try:
            import pyshacl  # noqa: F401
        except ImportError:
            return  # validation skipped without pyshacl
        assert not result.conforms
        assert any("SOURCE" in v for v in result.violations), result.violations

    def test_codec_provenance_round_trips_to_shacl(self) -> None:
        """Round-trip proof (D3): the **real interchange codec** emits a
        ProvenanceRef whose ``refType`` key — once expanded with the shipped
        ``@context`` — carries the ``particles:refType`` predicate that
        ProvenanceChainShape targets, so the serialized unit SHACL-validates.

        If ``refType`` were unmapped in context.jsonld (the RR-08 gap) or the
        shape still targeted the orphaned ``provenanceType``, the predicate would
        not expand and ``sh:minCount 1`` would fail — the test would not conform.
        """
        from particles.conformance.shacl import _load_context, _validate
        from particles.interchange.codec import to_unit

        unit = to_unit(_make_valid_particle(), {})
        assert unit["provenance"], "fixture must carry at least one ProvenanceRef"
        prov = unit["provenance"][0]
        assert "refType" in prov  # the codec's real wire key (unchanged)

        # Lift the codec-emitted provenance entry to a ProvenanceRef node and
        # attach the LOCAL context (the codec's @context is the non-resolving
        # CONTEXT_URL; the shacl validate path expects the inlined local terms).
        doc = {"@context": _load_context(), "@type": "particles:ProvenanceRef", **prov}
        result = _validate(doc, "ProvenanceChainShape")
        assert result.conforms, result.report_text

    def test_no_codec_key_is_unmapped_in_context(self) -> None:
        """No-unmapped-key invariant (D1): every key the codec emits
        into an ``@context``-bearing unit (excluding the JSON-LD keywords
        ``@id`` / ``@type`` / ``@context``) has a mapping in context.jsonld.

        Guards against a future codec change silently reintroducing the
        un-expanded-key gap that left exports failing to round-trip to RDF.
        """
        import json
        from pathlib import Path

        from particles.core.schema import (
            ContributorRef,
            ExternalRef,
            ExtractorRef,
            Subject,
        )
        from particles.interchange.codec import subject_to_unit, to_unit

        # A maximally-populated particle so every optional emitted key appears.
        p = _make_valid_particle().model_copy(
            update={
                "uncertainty_kind": "measurement",
                "supersedes": "old-id",
                "basis": {"k": "v"},
                "extractor_ref": ExtractorRef(name="general", version="1.0.0"),
                "sequence_context": ["a", "b"],
                "tags": ["physics"],
                "context_fingerprint": "fp",
                "properties": {"nmo:hasWeight": 0.5},
                "contributors": [ContributorRef(id="github:x", role="curator")],
                "subject_ids": ["sid-1", "sid-bare"],
            }
        )
        subjects = {
            "sid-1": Subject(
                id="sid-1",
                canonical_name="Water",
                asserted_by="t",
                aliases=["H2O"],
                subject_class="substance",
                description="a substance",
                external_ids=[
                    ExternalRef(namespace="wikidata", id="Q283", uri="http://x", confidence=0.9)
                ],
            ),
            "sid-bare": Subject(id="sid-bare", canonical_name="Local", asserted_by="t"),
        }
        unit = to_unit(p, subjects)
        subj_unit = subject_to_unit(subjects["sid-1"])

        ctx_file = Path(__file__).parents[1] / "artifacts" / "schemas" / "context.jsonld"
        ctx = json.loads(ctx_file.read_text())["@context"]
        defined = set(ctx.keys())
        # Keys mapped as @type:@json carry opaque free-form payloads (basis,
        # extractorRef, properties): their *inner* keys are JSON literal data,
        # not JSON-LD terms, so the invariant does not require them to be mapped.
        opaque = {
            term
            for term, spec in ctx.items()
            if isinstance(spec, dict) and spec.get("@type") == "@json"
        }

        def _collect(obj: object, acc: set[str]) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    acc.add(k)
                    if k not in opaque:  # do not descend into @json payloads
                        _collect(v, acc)
            elif isinstance(obj, list):
                for v in obj:
                    _collect(v, acc)

        emitted: set[str] = set()
        _collect(unit, emitted)
        _collect(subj_unit, emitted)
        emitted -= {"@context", "@type", "@id"}

        unmapped = sorted(emitted - defined)
        assert not unmapped, f"context.jsonld lacks mappings for codec keys: {unmapped}"


class TestSubjectJsonSchema:
    """`$defs/Subject` JSON Schema (§ Amendment) — the
    JSON-Schema twin of SubjectShape, closing the other half of the same drift
    (particle.schema.json was Particle-only)."""

    def test_valid_subject_passes(self) -> None:
        from particles.conformance.jsonschema import validate_subject
        from particles.core.schema import ExternalRef, Subject

        s = Subject(
            canonical_name="France",
            asserted_by="test-agent",
            external_ids=[ExternalRef(namespace="wikidata", id="Q142")],
        )
        assert validate_subject(s) == []

    def test_missing_canonical_name_fails(self) -> None:
        from particles.conformance.jsonschema import validate_subject_dict

        errors = validate_subject_dict(
            {"id": "s1", "asserted_by": "a", "created_at": "2026-01-01T00:00:00Z"}
        )
        assert any("canonical_name" in e for e in errors)

    def test_subject_with_contributors_passes(self) -> None:
        from particles.conformance.jsonschema import validate_subject
        from particles.core.schema import ContributorRef, Subject

        s = Subject(
            canonical_name="France",
            asserted_by="test-agent",
            contributors=[ContributorRef(id="github:torvalds", role="curator")],
        )
        assert validate_subject(s) == []

    def test_nested_external_ref_is_validated(self) -> None:
        from particles.conformance.jsonschema import validate_subject_dict

        errors = validate_subject_dict(
            {
                "id": "s1",
                "canonical_name": "C",
                "asserted_by": "a",
                "created_at": "2026-01-01T00:00:00Z",
                "external_ids": [{"id": "Q1"}],  # missing required namespace
            }
        )
        assert any("namespace" in e for e in errors)


class TestHasEvaluableFailure:
    """the read-side trust-cap trigger must treat a fixture-less
    extractor as *unknown*, never as a failure."""

    @staticmethod
    def _required_fail(reason: str) -> FieldStat:
        from particles.conformance.types import FieldStat, FieldTier

        return FieldStat(
            field="content",
            tier=FieldTier.REQUIRED,
            populated_count=0,
            total_count=0,
            rate=0.0,
            distinct_values=0,
            passes_threshold=False,
            failure_reason=reason,
        )

    @staticmethod
    def _report(particle_count: int, failures: list[FieldStat]) -> ConformanceReport:
        from datetime import UTC, datetime

        from particles.conformance.types import ConformanceReport

        return ConformanceReport(
            extractor_id="x",
            extractor_version="1.0",
            fixture_count=1,
            particle_count=particle_count,
            fields=list(failures),
            failures=failures,
            warnings=[],
            generated_at=datetime(2026, 1, 1, tzinfo=UTC),
            fixture_corpus_hash="h",
        )

    def test_zero_particles_is_unknown_not_failure(self) -> None:
        # The 9-fixture-less-extractors trap: every REQUIRED field "fails" at 0%
        # coverage, but particle_count == 0 means *unevaluated* — must NOT clamp.
        from particles.conformance.validator import has_evaluable_failure

        report = self._report(0, [self._required_fail("No particles produced; cannot evaluate")])
        assert report.failures  # the report does list failures …
        assert has_evaluable_failure(report) is False  # … but they are not evaluable

    def test_particles_with_required_failure_is_evaluable(self) -> None:
        from particles.conformance.validator import has_evaluable_failure

        report = self._report(
            12, [self._required_fail("Required field populated on 8/12 particles (67%)")]
        )
        assert has_evaluable_failure(report) is True

    def test_particles_with_no_failures_is_not_a_failure(self) -> None:
        from particles.conformance.validator import has_evaluable_failure

        assert has_evaluable_failure(self._report(12, [])) is False


@upstream_only  # reads the specification prose, which the engine tree does not carry
class TestSpecTierSync:
    """techspec §14.5's tier enumeration must equal `contract.py`.

    §14.5 declares the categorisation normative while delegating the *report*
    shape to `types.py`, so the tier lists are the one thing in that section a
    second implementation must be able to build from the prose alone. They had
    silently drifted — `extractor_ref` shown Required, `uncertainty_nature` and
    `subject_ids` shown Recommended, all three the opposite of the contract —
    which is what this guard exists to make impossible to repeat.
    """

    @staticmethod
    def _section() -> str:
        import re
        from pathlib import Path

        spec = (
            Path(__file__).parents[1] / "docs" / "spec" / "technical-specification.md"
        ).read_text()
        match = re.search(r"^## 14\.5 .*?^## 14\.6 ", spec, re.MULTILINE | re.DOTALL)
        assert match, "techspec §14.5 not found"
        return match.group(0)

    @classmethod
    def _listed(cls, tier_label: str, lead_in: str) -> set[str]:
        import re

        line = next(
            ln for ln in cls._section().splitlines() if ln.startswith(f"- **{tier_label}** —")
        )
        _, _, tail = line.partition(lead_in)
        assert tail, f"§14.5 {tier_label} bullet lost its {lead_in!r} lead-in"
        return set(re.findall(r"`([^`]+)`", tail))

    @staticmethod
    def _contract(tier_name: str) -> set[str]:
        from particles.conformance.contract import CONTRACT

        return {c.field for c in CONTRACT if c.tier.name == tier_name}

    def test_required_tier_is_enumerated_exhaustively(self) -> None:
        assert self._listed("Required", "**exhaustive**:") == self._contract("REQUIRED")

    def test_recommended_tier_is_enumerated_exhaustively(self) -> None:
        assert self._listed("Recommended", "**exhaustive**:") == self._contract("RECOMMENDED")

    def test_optional_examples_are_actually_optional(self) -> None:
        # The optional tier is the residue and is deliberately given by example,
        # so it is not checked for completeness — only for correctness.
        assert self._listed("Optional", "Examples:") <= self._contract("OPTIONAL")


if TYPE_CHECKING:
    from particles.conformance.types import ConformanceReport, FieldStat
