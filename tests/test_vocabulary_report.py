# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The read-time vocabulary report.

The fold is pure (``core/vocabulary_report.py``) and is tested over plain rows; the
Engine path is tested once end to end on a seeded store, through the observer,
and with an as-of cut; the CLI table / JSON and the ``GET /vocabulary`` block
are tested at their surfaces.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from particles.core.schema import (
    ClaimTerm,
    Confidence,
    ExternalRef,
    LinkBand,
    ObjectShape,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    QueryResponse,
    StructuredClaim,
    Subject,
    TermKind,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.core.vocabulary_report import (
    NO_SUBJECT,
    UNCLASSED,
    alignments_from_documents,
    build_vocabulary_report,
    canonical_form,
    class_namespace,
    link_band,
    object_shape,
    predicate_rows,
)

Session = Any  # the db_session fixture is untyped


def _term(value: str, kind: TermKind = TermKind.TOKEN, **qualifiers: str) -> ClaimTerm:
    return ClaimTerm(kind=kind, value=value, **qualifiers)


def _lit(value: str, **qualifiers: str) -> ClaimTerm:
    return ClaimTerm(kind=TermKind.LITERAL, value=value, **qualifiers)


def _claim(
    predicate: str,
    obj: ClaimTerm | None = None,
    *,
    kind: TermKind = TermKind.TOKEN,
    subject_id: str | None = None,
) -> StructuredClaim:
    return StructuredClaim(
        subject=_term("thing"),
        predicate=_term(predicate, kind),
        object=obj or _lit("value"),
        subject_id=subject_id,
        structurizer_id="test",
        structurizer_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# The pure fold
# ---------------------------------------------------------------------------


class TestObjectShape:
    @pytest.mark.parametrize(
        ("term", "shape"),
        [
            (_lit("3.9", datatype="xsd:decimal"), ObjectShape.NUMERIC),
            (_lit("2026-10-01", datatype="xsd:date"), ObjectShape.DATED),
            (_lit("1987"), ObjectShape.NUMERIC),
            (_lit("2026-10-01"), ObjectShape.DATED),
            (_lit("about four grams"), ObjectShape.TEXT),
            (_lit("3 grams"), ObjectShape.TEXT),
            (_lit("not a number", datatype="xsd:decimal"), ObjectShape.TEXT),
            (_lit("1987", language="en"), ObjectShape.TEXT),
            (_lit("true", datatype="xsd:boolean"), ObjectShape.TEXT),
            (_term("wd:Q42", TermKind.URI), ObjectShape.URI),
            (_term("Douglas Adams"), ObjectShape.TOKEN),
        ],
    )
    def test_shape_is_read_from_form(self, term: ClaimTerm, shape: ObjectShape) -> None:
        assert object_shape(term) is shape


class TestLinkBand:
    @pytest.mark.parametrize(
        ("confidence", "band"),
        [
            (1.0, LinkBand.ASSERTED),
            (0.5, LinkBand.UNSCORED),
            (0.8, LinkBand.SCORED),
            (0.25, LinkBand.SCORED),
            (0.24, LinkBand.SUPPRESSED),
            (0.0, LinkBand.SUPPRESSED),
        ],
    )
    def test_bands_follow_adr_0054(self, confidence: float, band: LinkBand) -> None:
        assert link_band(confidence, 0.25) is band


class TestCanonicalForm:
    def test_inflections_share_one_form_and_prepositions_stay_apart(self) -> None:
        forms = {canonical_form(_term(p)) for p in ("moved to", "moves to", "was moved to")}
        assert len(forms) == 1
        assert canonical_form(_term("moved from")) not in forms

    def test_a_uri_predicate_is_its_own_form(self) -> None:
        assert canonical_form(_term("schema:birthDate", TermKind.URI)) == "schema:birthDate"

    @pytest.mark.parametrize(
        ("cls", "namespace"),
        [
            ("artifact:file", "artifact"),
            ("nmo:Coin", "nmo"),
            ("http://ex.org/onto#Coin", "http://ex.org/onto#"),
            ("http://ex.org/Coin", "http://ex.org/"),
            ("Coin", "(none)"),
        ],
    )
    def test_class_namespace(self, cls: str, namespace: str) -> None:
        assert class_namespace(cls) == namespace


class TestPredicateRows:
    def _rows(self) -> list[tuple[StructuredClaim, str]]:
        return [
            (_claim("has status", _lit("done")), "artifact:record"),
            (_claim("has status", _lit("open")), "artifact:record"),
            (_claim("had status", _lit("2026-01-02")), UNCLASSED),
            (_claim("contains", _lit("3")), "artifact:file"),
            (_claim("contains", _term("wd:Q1", TermKind.URI)), NO_SUBJECT),
        ]

    def test_groups_surface_forms_under_the_canonical_predicate(self) -> None:
        rows = predicate_rows(self._rows())

        assert [r.label for r in rows] == ["has status", "contains"]
        status = rows[0]
        assert status.claim_count == 3
        assert [(f.value, f.claim_count) for f in status.surface_forms] == [
            ("has status", 2),
            ("had status", 1),
        ]
        assert status.subject_classes[0].subject_class == "artifact:record"
        assert status.object_shapes[ObjectShape.TEXT] == 2
        assert status.object_shapes[ObjectShape.DATED] == 1
        assert status.alignments == []

    def test_every_column_sums_to_the_claim_count(self) -> None:
        for row in predicate_rows(self._rows()):
            assert sum(f.claim_count for f in row.surface_forms) == row.claim_count
            assert sum(row.object_shapes.values()) == row.claim_count
            assert sum(c.count for c in row.subject_classes) == row.claim_count
            assert set(row.object_shapes) == set(ObjectShape)

    def test_alignments_are_read_by_canonical_form(self) -> None:
        form = canonical_form(_term("contains"))
        rows = predicate_rows(self._rows(), {form: ["schema:hasPart"]})
        assert {r.label: r.alignments for r in rows} == {
            "has status": [],
            "contains": ["schema:hasPart"],
        }


def _document() -> Any:
    """A document with one term on files, an alias, and one outward alignment."""
    from particles.core.predicate_profile import normalise_predicate
    from particles.core.vocabulary import (
        Alignment,
        MatchStrength,
        Ruling,
        add_alias,
        add_alignment,
        new_document,
    )

    ruling = Ruling(confirmed_by="reviewer")
    doc = new_document(name="acme", prefix="acme", namespace="https://acme.example.org/vocab/")
    doc = add_alias(
        doc,
        form=normalise_predicate("contains"),
        subject_class="artifact:file",
        aliases=[normalise_predicate("includes")],
        ruling=ruling,
    )
    local = doc.terms[0].local_name
    return add_alignment(
        doc,
        term_ref=f"acme:{local}",
        alignment=Alignment(
            target="schema:hasPart", match=MatchStrength.EXACT, confidence=0.9, ruling=ruling
        ),
    )


class TestAlignmentsFromDocuments:
    def test_no_document_is_no_alignment_and_no_source(self) -> None:
        assert alignments_from_documents([]) == ({}, None)

    def test_a_term_its_alias_and_its_outward_link(self) -> None:
        doc = _document()
        local = doc.terms[0].local_name

        alignments, source = alignments_from_documents([doc])

        expected = [
            f"acme:{local} on artifact:file",
            "schema:hasPart (exact) on artifact:file",
        ]
        assert alignments[canonical_form(_term("contains"))] == expected
        assert alignments[canonical_form(_term("includes"))] == expected
        assert source == f"acme v{doc.version}"

    def test_rows_carry_the_alignment_of_their_form(self) -> None:
        alignments, _ = alignments_from_documents([_document()])
        rows = predicate_rows(
            [(_claim("contained"), "artifact:file"), (_claim("covers"), UNCLASSED)], alignments
        )
        by_label = {r.label: r.alignments for r in rows}
        assert len(by_label["contained"]) == 2 and by_label["covers"] == []


class TestBuildReport:
    def test_header_counts(self) -> None:
        subjects: list[tuple[str | None, list[ExternalRef]]] = [
            (None, [ExternalRef(namespace="wikidata", id="Q1", confidence=0.8)]),
            # Two links in one namespace: the subject counts once, in its best band.
            (
                None,
                [
                    ExternalRef(namespace="wikidata", id="Q2", confidence=0.1),
                    ExternalRef(namespace="wikidata", id="Q3", confidence=1.0),
                ],
            ),
            ("artifact:file", [ExternalRef(namespace="artifact", id="p/a.py")]),
            ("artifact:record", []),
            (None, []),
        ]
        claims = [
            (_claim("has status"), "artifact:record"),
            (_claim("had status"), "artifact:record"),
            (_claim("schema:url", _term("https://x", TermKind.URI), kind=TermKind.URI), UNCLASSED),
        ]

        report = build_vocabulary_report(
            claims,
            subjects,
            structured_claims_total=10,
            event_counts={"DUPLICATES_MERGED": 4, "PARTICLE_TAGGED": 9},
            decision_types=["DUPLICATES_MERGED", "SUBJECTS_MERGED"],
            suppress_threshold=0.25,
        )

        assert (report.subjects_total, report.subjects_aligned, report.subjects_classed) == (
            5,
            3,
            2,
        )
        wikidata = next(n for n in report.aligned_by_namespace if n.namespace == "wikidata")
        assert wikidata.subjects == 2
        assert wikidata.bands[LinkBand.ASSERTED] == 1 and wikidata.bands[LinkBand.SCORED] == 1
        assert [(c.subject_class, c.count) for c in report.classed_by_namespace] == [
            ("artifact", 2)
        ]
        assert report.structured_claims_total == 10 and report.claims_in_view == 3
        assert report.object_kinds[TermKind.URI] == 1 and report.object_kinds[TermKind.LITERAL] == 2
        assert (report.predicates_distinct, report.predicates_canonical) == (3, 2)
        # Only decision types are reported, and an absent one reads as zero.
        assert report.modelling_decisions == {"DUPLICATES_MERGED": 4, "SUBJECTS_MERGED": 0}
        assert report.alignment_source is None


# ---------------------------------------------------------------------------
# The Engine path
# ---------------------------------------------------------------------------


def _particle(claim: StructuredClaim, *, subject_ids: list[str] | None = None) -> Particle:
    return Particle(
        content=f"{claim.predicate.value} {claim.object.value}",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        subject_ids=subject_ids or [],
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e-1", snapshot_id="s-1")
        ],
        structured_claim=claim,
    )


async def _seed(session: Session) -> dict[str, str]:
    from particles.store.event_store import OperatorEventType, record_event
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject

    record = Subject(
        canonical_name="RFC-2119",
        asserted_by="test",
        subject_class="artifact:record",
        external_ids=[ExternalRef(namespace="artifact", id="llm/rfc-2119")],
    )
    person = Subject(
        canonical_name="Ada Lovelace",
        asserted_by="test",
        external_ids=[ExternalRef(namespace="wikidata", id="Q7259", confidence=0.6)],
    )
    late = Subject(
        canonical_name="Later subject",
        asserted_by="test",
        created_at=datetime.now(UTC) + timedelta(days=30),
    )
    for subject in (record, person, late):
        await insert_subject(session, subject)
    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    for claim in (
        _claim("has status", _lit("open"), subject_id=record.id),
        _claim("had status", _lit("done"), subject_id=record.id),
        _claim("was born on", _lit("1815-12-10"), subject_id=person.id),
        _claim("covers", _lit("the report")),
    ):
        await insert_particle(session, _particle(claim), emb)
    await record_event(session, actor="test", event_type=OperatorEventType.DUPLICATES_MERGED)
    await record_event(session, actor="test", event_type=OperatorEventType.PARTICLE_TAGGED)
    await session.commit()
    return {"record": record.id, "person": person.id}


class _NoModels:
    def __init__(self) -> None:
        import anthropic

        self.embed = MagicMock(name="embedding-model")
        self.llm = MagicMock(spec=anthropic.Anthropic)
        self.llm.messages = MagicMock()


@pytest.fixture
def no_models() -> Generator[_NoModels, None, None]:
    from particles import embeddings as ep
    from particles.llm import set_client

    spies = _NoModels()
    original = ep._embedding_model
    ep.set_embedding_model(spies.embed)
    set_client(spies.llm)
    try:
        yield spies
    finally:
        ep.set_embedding_model(original)
        set_client(None)


async def test_the_report_over_a_seeded_store(db_session: Session, no_models: _NoModels) -> None:
    from particles.operations.query import query

    await _seed(db_session)

    result = await query(db_session, QueryRequest(list_vocabulary=True))
    report = result.vocabulary_report

    assert report is not None
    assert report.subjects_total == 3 and report.subjects_aligned == 2
    assert [(c.subject_class, c.count) for c in report.classed_by_class] == [("artifact:record", 1)]
    wikidata = next(n for n in report.aligned_by_namespace if n.namespace == "wikidata")
    assert wikidata.bands[LinkBand.SCORED] == 1
    # A tie between surface forms labels the row with the alphabetically first.
    status = next(p for p in report.predicates if p.label == "had status")
    assert status.claim_count == 2
    assert [(c.subject_class, c.count) for c in status.subject_classes] == [("artifact:record", 2)]
    born = next(p for p in report.predicates if p.label == "was born on")
    assert born.subject_classes[0].subject_class == UNCLASSED
    assert born.object_shapes[ObjectShape.DATED] == 1
    covers = next(p for p in report.predicates if p.label == "covers")
    assert covers.subject_classes[0].subject_class == NO_SUBJECT
    assert report.modelling_decisions["DUPLICATES_MERGED"] == 1
    assert "PARTICLE_TAGGED" not in report.modelling_decisions
    assert report.alignment_source is None
    assert all(p.alignments == [] for p in report.predicates)
    # Deterministic: no embedding, no LLM call.
    assert no_models.embed.encode.call_count == 0
    assert no_models.llm.messages.create.call_count == 0


async def test_the_listing_it_extends_is_unchanged(
    db_session: Session, no_models: _NoModels
) -> None:
    """``--predicates`` still lists terms as stored; the report counts the same terms."""
    from particles.operations.query import query

    await _seed(db_session)

    listing = await query(db_session, QueryRequest(list_predicates=True))
    report = (await query(db_session, QueryRequest(list_vocabulary=True))).vocabulary_report

    assert listing.vocabulary_report is None
    assert {(v.value, v.claim_count) for v in listing.predicate_vocabulary} == {
        ("has status", 1),
        ("had status", 1),
        ("was born on", 1),
        ("covers", 1),
    }
    assert report is not None
    assert report.predicates_distinct == len(listing.predicate_vocabulary)
    assert report.predicates_canonical == 3


async def test_as_of_cuts_the_header_at_the_instant(
    db_session: Session, no_models: _NoModels
) -> None:
    from particles.operations.query import query

    await _seed(db_session)
    now = datetime.now(UTC)
    past = now - timedelta(days=1)

    now_view = (
        await query(db_session, QueryRequest(list_vocabulary=True, as_of=now))
    ).vocabulary_report
    then_view = (
        await query(db_session, QueryRequest(list_vocabulary=True, as_of=past))
    ).vocabulary_report

    assert now_view is not None and then_view is not None
    # The subject stamped next month is not counted as of now.
    assert now_view.subjects_total == 2 and now_view.as_of == now
    assert now_view.modelling_decisions["DUPLICATES_MERGED"] == 1
    assert then_view.subjects_total == 0 and then_view.claims_in_view == 0
    assert then_view.modelling_decisions["DUPLICATES_MERGED"] == 0


async def test_the_report_reads_through_the_observer(db_session: Session) -> None:
    """a selection by something other than id goes through the observer."""
    from particles.operations.query.structural import structural_query
    from tests._observer_scope import belief, project_source_tags, rescoped, source

    a, b = "-proj-a", "-proj-b"
    in_a = await source(db_session, "a", project_source_tags(a))
    in_b = await source(db_session, "b", project_source_tags(b))
    await belief(db_session, "A deploys Fridays.", in_a, structured_claim=_claim("deploys on"))
    await belief(db_session, "B ships Mondays.", in_b, structured_claim=_claim("ships on"))
    await rescoped(db_session)
    await db_session.commit()

    scoped = await structural_query(
        db_session, QueryRequest(list_vocabulary=True, observer_project=a)
    )
    everything = await structural_query(db_session, QueryRequest(list_vocabulary=True))

    assert scoped.vocabulary_report is not None and everything.vocabulary_report is not None
    assert [p.label for p in scoped.vocabulary_report.predicates] == ["deploys on"]
    assert scoped.vocabulary_report.claims_in_view == 1
    note = scoped.vocabulary_report.observer_scope
    assert note is not None and note.engaged and note.project == a
    assert scoped.observer_scope == note
    assert {p.label for p in everything.vocabulary_report.predicates} == {
        "deploys on",
        "ships on",
    }
    assert everything.vocabulary_report.observer_scope is None


async def test_an_adopted_document_aligns_the_report(
    db_session: Session, no_models: _NoModels
) -> None:
    """the alignment column reads the documents the store adopts."""
    from particles.core.vocabulary_jsonld import dumps
    from particles.operations.query import query
    from particles.operations.vocabulary import adopt, import_document
    from particles.store.particle_store import insert_particle

    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    await insert_particle(db_session, _particle(_claim("includes", _lit("x"))), emb)
    await db_session.commit()
    before = (await query(db_session, QueryRequest(list_vocabulary=True))).vocabulary_report

    doc = _document()
    await import_document(db_session, dumps(doc))
    await adopt(db_session, "acme")
    await db_session.commit()
    after = (await query(db_session, QueryRequest(list_vocabulary=True))).vocabulary_report

    assert before is not None and before.alignment_source is None
    assert before.predicates[0].alignments == []
    assert after is not None and after.alignment_source == f"acme v{doc.version}"
    assert "schema:hasPart (exact) on artifact:file" in after.predicates[0].alignments


class TestRequest:
    def test_list_vocabulary_is_a_structural_mode(self) -> None:
        assert QueryRequest(list_vocabulary=True).is_structural_mode

    @pytest.mark.parametrize(
        "extra",
        [
            {"question": "what?"},
            {"predicate": "covers"},
            {"count": True},
            {"list_predicates": True},
        ],
    )
    def test_list_vocabulary_is_standalone(self, extra: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="list_vocabulary is a standalone report"):
            QueryRequest(list_vocabulary=True, **extra)


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

runner = CliRunner()


def _report_response() -> QueryResponse:
    claims = [
        (_claim("has status", _lit("open")), "artifact:record"),
        (_claim("has status", _lit("done")), "artifact:record"),
        (_claim("had status", _lit("2026-01-02")), UNCLASSED),
    ]
    report = build_vocabulary_report(
        claims,
        [("artifact:record", [ExternalRef(namespace="wikidata", id="Q1", confidence=0.3)])],
        structured_claims_total=2,
        event_counts={"DUPLICATES_MERGED": 7},
        decision_types=["DUPLICATES_MERGED"],
        suppress_threshold=0.25,
    )
    return QueryResponse(
        answer="1 canonical predicate(s).",
        particles=[],
        effective_confidences=[],
        vocabulary_report=report,
    )


@pytest.fixture
def backend() -> Generator[MagicMock, None, None]:
    from unittest.mock import AsyncMock

    fake = MagicMock()
    fake.remote = False
    fake.query = AsyncMock(return_value=_report_response())
    with patch("particles.api.cli.query.get_backend", return_value=fake):
        yield fake


class TestCli:
    def test_table(self, backend: MagicMock) -> None:
        result = runner.invoke(app_(), ["query", "--vocabulary"], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        out = result.output
        assert "Subjects: 1 · aligned 1 · classed 1" in out
        assert "wikidata" in out and "0 / 1 / 0 / 0" in out
        assert "no vocabulary document adopted" in out
        assert "DUPLICATES_MERGED 7" in out
        assert "CLAIMS" in out and "PREDICATE" in out
        assert "has status  [hav status]" in out
        assert "forms: has status 2 · had status 1" in out
        assert "classes: artifact:record 2 · (unclassed) 1" in out
        req = backend.query.call_args.args[0]
        assert req.list_vocabulary and not req.list_predicates

    def test_json(self, backend: MagicMock) -> None:
        result = runner.invoke(
            app_(), ["query", "--vocabulary", "--format", "json"], catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        body = json.loads(result.output)
        assert body["subjects_total"] == 1
        assert body["predicates"][0]["label"] == "has status"
        assert body["predicates"][0]["object_shapes"]["dated"] == 1
        assert body["predicates"][0]["alignments"] == []
        assert body["modelling_decisions"] == {"DUPLICATES_MERGED": 7}

    def test_json_needs_the_report(self, backend: MagicMock) -> None:
        result = runner.invoke(app_(), ["query", "--predicates", "--format", "json"])
        assert result.exit_code == 1
        assert "--format json applies to --vocabulary only" in result.output
        backend.query.assert_not_called()

    def test_unknown_format(self, backend: MagicMock) -> None:
        result = runner.invoke(app_(), ["query", "--vocabulary", "--format", "yaml"])
        assert result.exit_code == 1
        assert "Unknown --format" in result.output


def app_() -> Any:
    from particles.api.cli import app

    return app
