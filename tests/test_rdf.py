"""RDF parsing extractor tests.

Every test here is a deterministic parse: a fixed blob in, a fixed candidate
list out, no LLM and no network. That is the whole point of the extractor, so
the suite pins it rather than mocking around it.
"""

from __future__ import annotations

import json

import pytest

from particles.core.schema import CanonicalForm, Snapshot, SourceType, TermKind
from particles.extraction.rdf import (
    EXTRACTOR_ID,
    RDF_CONTENT_TYPES,
    RDF_SUFFIXES,
    RdfExtractor,
    RemoteRetrievalRefused,
    _format_candidates,
    external_ref_for,
    no_remote_retrieval,
    prettify_local_name,
    strip_remote_contexts,
)

TURTLE = b"""
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix wd: <http://www.wikidata.org/entity/> .

ex:coin rdfs:label "5 Pfennigs" .
ex:mint rdfs:label "Berlin Mint" .
wd:Q64 rdfs:label "Berlin" .

ex:coin ex:wasMintedAt ex:mint .
ex:coin ex:hasWeight "1.1" .
ex:mint ex:locatedIn wd:Q64 .
"""

REIFIED = b"""
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix p: <https://linkedparticles.org/vocab#> .

_:r a rdf:Statement ;
    rdf:subject ex:coin ; rdf:predicate ex:wasDemonetized ; rdf:object "1990-12-31" ;
    p:confidenceValue 0.72 .
"""

TRIG = b"""
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix p: <https://linkedparticles.org/vocab#> .

ex:g1 { ex:a rdfs:label "Alpha" . ex:b rdfs:label "Beta" . ex:a ex:knows ex:b . }
ex:g1 p:confidenceValue 0.55 .
"""


def _snapshot() -> Snapshot:
    return Snapshot(content_hash="a" * 64)


async def _extract(content: bytes, uri: str | None = None) -> object:
    return await RdfExtractor().extract(_snapshot(), content, entry_uri_r=uri)


# ---------------------------------------------------------------------------
# Plugin identity and routing
# ---------------------------------------------------------------------------


def test_accepts_only_rdf_graph() -> None:
    extractor = RdfExtractor()
    assert extractor.accepts(SourceType.RDF_GRAPH)
    assert not extractor.accepts(SourceType.LOCAL_FILE)
    assert not extractor.accepts(SourceType.WEB_PAGE)


def test_registered_before_the_general_fallback() -> None:
    from particles.extraction.registry import get_extractors

    ids = [type(e).__name__ for e in get_extractors()]
    assert "RdfExtractor" in ids
    assert ids.index("RdfExtractor") < ids.index("GeneralExtractor")


@pytest.mark.parametrize(
    ("uri", "expected_first"),
    [
        ("/tmp/a.ttl", "turtle"),
        ("/tmp/a.nt", "nt"),
        ("/tmp/a.trig", "trig"),
        ("/tmp/a.nq", "nquads"),
        ("https://example.org/data.jsonld?v=2", "json-ld"),
        ("https://example.org/o.owl", "xml"),
    ],
)
def test_format_hint_from_uri(uri: str, expected_first: str) -> None:
    assert _format_candidates(uri, b"")[0] == expected_first


def test_format_candidates_are_deduped_and_exhaustive() -> None:
    candidates = _format_candidates("/tmp/a.ttl", b"@prefix ex: <http://e/> .")
    assert len(candidates) == len(set(candidates))
    # Every syntax stays reachable even when the hint is wrong.
    assert {"trig", "nquads", "json-ld", "xml"} <= set(candidates)


def test_json_and_xml_content_sniffs_promote_their_format() -> None:
    assert _format_candidates(None, b'  {"@id": "x"}')[0] == "json-ld"
    assert _format_candidates(None, b'<?xml version="1.0"?><rdf:RDF/>')[0] == "xml"


def test_deposit_detects_rdf_suffixes_and_content_types() -> None:
    from pathlib import Path

    from particles.corpus.deposit import _detect_source_type

    for suffix in RDF_SUFFIXES:
        assert _detect_source_type(None, None, Path(f"/tmp/x{suffix}")) == SourceType.RDF_GRAPH
    for content_type in RDF_CONTENT_TYPES:
        assert _detect_source_type(None, f"{content_type}; charset=utf-8", None) is (
            SourceType.RDF_GRAPH
        )


def test_bare_json_is_not_routed_to_rdf() -> None:
    """the .json slot is contended; an @context sniff would
    hijack this SDK's own interchange bundles."""
    from pathlib import Path

    from particles.corpus.deposit import _detect_source_type

    assert _detect_source_type(None, None, Path("/tmp/bundle.json")) != SourceType.RDF_GRAPH


# ---------------------------------------------------------------------------
# Verbalization (§3.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("local", "expected"),
    [
        ("wasMintedAt", "was minted at"),
        ("BerlinMint", "Berlin mint"),
        ("has_weight", "has weight"),
        ("located-in", "located in"),
        ("P571", "P571"),
        ("label", "label"),
    ],
)
def test_prettify_local_name(local: str, expected: str) -> None:
    assert prettify_local_name(local) == expected


async def test_labels_come_from_the_document() -> None:
    result = await _extract(TURTLE, "/tmp/coins.ttl")
    contents = {c.content for c in result.candidates}
    assert "5 Pfennigs was minted at: Berlin Mint" in contents
    assert "Berlin Mint located in: Berlin" in contents


async def test_content_falls_back_to_the_iri_and_is_never_empty() -> None:
    """The ladder terminates at the full IRI, which is what makes
    ``Particle.content``'s min_length=1 hold by construction (§3.4)."""
    unlabelled = b"<http://example.org/a> <http://example.org/b> <http://example.org/c> ."
    result = await _extract(unlabelled, "/tmp/x.nt")
    assert len(result.candidates) == 1
    assert result.candidates[0].content
    assert result.candidates[0].content == "a b: c"


async def test_non_english_literal_is_flagged_with_its_language() -> None:
    data = '<http://example.org/a> <http://example.org/name> "Köln"@de .'.encode()
    result = await _extract(data, "/tmp/x.nt")
    assert result.candidates[0].content.endswith("Köln (de)")


# ---------------------------------------------------------------------------
# What becomes a particle (§3.3)
# ---------------------------------------------------------------------------


async def test_label_triples_are_not_emitted_as_claims() -> None:
    result = await _extract(TURTLE, "/tmp/coins.ttl")
    assert len(result.candidates) == 3
    assert not any("label" in c.content for c in result.candidates)


async def test_blank_node_subjects_are_skipped_by_default() -> None:
    data = b"""
    @prefix ex: <http://example.org/> .
    [] ex:p ex:o .
    ex:s ex:p ex:o .
    """
    result = await _extract(data, "/tmp/x.ttl")
    assert len(result.candidates) == 1
    assert result.candidates[0].structured_claim is not None
    assert result.candidates[0].structured_claim.subject.value == "http://example.org/s"


async def test_reification_folds_to_one_particle_and_consumes_the_plumbing() -> None:
    result = await _extract(REIFIED, "/tmp/x.ttl")
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.content == "coin was demonetized: 1990-12-31"
    # The bundle's own confidence, not the parser default.
    assert candidate.confidence_value == pytest.approx(0.72)
    assert not any("rdf:subject" in c.content for c in result.candidates)


async def test_reification_does_not_double_emit_an_also_asserted_triple() -> None:
    data = b"""
    @prefix ex: <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix p: <https://linkedparticles.org/vocab#> .
    ex:s ex:p "v" .
    _:r a rdf:Statement ; rdf:subject ex:s ; rdf:predicate ex:p ; rdf:object "v" ;
        p:confidenceValue 0.4 .
    """
    result = await _extract(data, "/tmp/x.ttl")
    assert len(result.candidates) == 1
    assert result.candidates[0].confidence_value == pytest.approx(0.4)


async def test_named_graph_confidence_applies_and_is_recorded() -> None:
    result = await _extract(TRIG, "/tmp/x.trig")
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.content == "Alpha knows: Beta"
    assert candidate.confidence_value == pytest.approx(0.55)
    assert candidate.properties == {"rdf:graph": "http://example.org/g1"}


async def test_malformed_confidence_annotation_falls_back_to_the_default() -> None:
    data = b"""
    @prefix ex: <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix p: <https://linkedparticles.org/vocab#> .
    _:r a rdf:Statement ; rdf:subject ex:s ; rdf:predicate ex:p ; rdf:object "v" ;
        p:confidenceValue "not a number" .
    """
    result = await _extract(data, "/tmp/x.ttl")
    assert result.candidates[0].confidence_value == pytest.approx(0.95)


async def test_out_of_range_confidence_is_ignored() -> None:
    data = b"""
    @prefix ex: <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix p: <https://linkedparticles.org/vocab#> .
    _:r a rdf:Statement ; rdf:subject ex:s ; rdf:predicate ex:p ; rdf:object "v" ;
        p:confidenceValue 4.2 .
    """
    result = await _extract(data, "/tmp/x.ttl")
    assert result.candidates[0].confidence_value == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Structure-canonical output (§3.5, §3.6)
# ---------------------------------------------------------------------------


async def test_every_candidate_is_structure_canonical_and_stamped() -> None:
    result = await _extract(TURTLE, "/tmp/coins.ttl")
    for candidate in result.candidates:
        assert candidate.canonical_form is CanonicalForm.STRUCTURED
        claim = candidate.structured_claim
        assert claim is not None
        # For STRUCTURED the stamp records what *read* the triple (§3.6).
        assert claim.structurizer_id == EXTRACTOR_ID
        assert claim.structurizer_version


async def test_term_kinds_follow_the_rdf_term_types() -> None:
    result = await _extract(TURTLE, "/tmp/coins.ttl")
    by_content = {c.content: c for c in result.candidates}
    weight = by_content["5 Pfennigs has weight: 1.1"].structured_claim
    assert weight is not None
    assert weight.subject.kind is TermKind.URI
    assert weight.predicate.kind is TermKind.URI
    assert weight.object.kind is TermKind.LITERAL

    minted = by_content["5 Pfennigs was minted at: Berlin Mint"].structured_claim
    assert minted is not None
    assert minted.object.kind is TermKind.URI


async def test_typed_literal_carries_its_datatype_and_no_language() -> None:
    data = (
        b"<http://example.org/a> <http://example.org/w> "
        b'"1.1"^^<http://www.w3.org/2001/XMLSchema#decimal> .'
    )
    result = await _extract(data, "/tmp/x.nt")
    claim = result.candidates[0].structured_claim
    assert claim is not None
    assert claim.object.datatype == "http://www.w3.org/2001/XMLSchema#decimal"
    assert claim.object.language is None


# ---------------------------------------------------------------------------
# Subject binding (§3.7)
# ---------------------------------------------------------------------------


def test_external_ref_for_maps_known_prefixes() -> None:
    namespaces = {
        "http://www.wikidata.org/entity/": "wikidata",
        "http://nomisma.org/id/": "nomisma",
    }
    ref = external_ref_for("http://www.wikidata.org/entity/Q64", namespaces)
    assert ref is not None
    assert (ref.namespace, ref.id) == ("wikidata", "Q64")
    assert ref.uri == "http://www.wikidata.org/entity/Q64"
    assert external_ref_for("http://example.org/thing", namespaces) is None
    # A bare prefix with no local part is not an entity.
    assert external_ref_for("http://www.wikidata.org/entity/", namespaces) is None


def test_external_ref_for_prefers_the_longest_matching_prefix() -> None:
    namespaces = {"http://x.org/": "broad", "http://x.org/deep/": "narrow"}
    ref = external_ref_for("http://x.org/deep/42", namespaces)
    assert ref is not None
    assert (ref.namespace, ref.id) == ("narrow", "42")


async def test_uri_object_becomes_a_second_subject_with_its_external_ref() -> None:
    result = await _extract(TURTLE, "/tmp/coins.ttl")
    located = next(c for c in result.candidates if c.content.startswith("Berlin Mint located"))
    assert located.subjects == ["Berlin Mint", "Berlin"]
    assert located.external_refs["Berlin"].namespace == "wikidata"
    assert located.external_refs["Berlin"].id == "Q64"


def test_bind_subject_id_matches_a_uri_term_via_external_refs() -> None:
    """The rung added: a structure-native triple's subject term is
    an IRI while the candidate's names are labels, so name matching alone would
    never bind for the population with the best keys."""
    from particles.core.schema import ClaimTerm, ExternalRef, StructuredClaim
    from particles.extraction.structure import bind_subject_id

    claim = StructuredClaim(
        subject=ClaimTerm(kind=TermKind.URI, value="http://www.wikidata.org/entity/Q64"),
        predicate=ClaimTerm(kind=TermKind.URI, value="http://example.org/p"),
        object=ClaimTerm(kind=TermKind.LITERAL, value="v"),
        structurizer_id="rdf-extractor",
        structurizer_version="0.1.0",
    )
    refs = {
        "Berlin": ExternalRef(
            namespace="wikidata", id="Q64", uri="http://www.wikidata.org/entity/Q64"
        )
    }
    bound = bind_subject_id(claim, ["Berlin"], ["subject-uuid"], external_refs=refs)
    assert bound.subject_id == "subject-uuid"


async def test_uri_binding_survives_the_whole_extractor_to_particle_path() -> None:
    """End-to-end wiring, not just the rung in isolation.

    The extractor keys ``external_refs`` by the human-readable *label* while the
    triple's subject term is an IRI, so the connection between the two runs
    through ``candidate_to_particle`` → ``bind_subject_id``. It is exactly the
    kind of seam that breaks silently if someone re-keys the map, and the
    payoff it protects is the invariant below: a bound
    ``subject_id`` must be one of the particle's own subjects, or ``L-STR-11``
    reports the annotation as hallucinated.
    """
    from particles.extraction.general import candidate_to_particle

    data = b"""
    @prefix ex: <http://example.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix wd: <http://www.wikidata.org/entity/> .
    wd:Q64 rdfs:label "Berlin" .
    wd:Q64 ex:population "3600000" .
    """
    result = await _extract(data, "/tmp/x.ttl")
    candidate = result.candidates[0]
    assert candidate.subjects == ["Berlin"]

    particle = candidate_to_particle(
        candidate, "entry-1", "snapshot-1", "rdf-extractor", subject_ids=["uuid-berlin"]
    )
    assert particle.structured_claim is not None
    assert particle.structured_claim.subject_id == "uuid-berlin"
    assert particle.structured_claim.subject_id in particle.subject_ids


async def test_structure_canonical_particle_survives_a_store_round_trip() -> None:
    """`canonical_form`, the triple and the stamp all persist."""
    from particles.extraction.general import candidate_to_particle
    from particles.store.particle_store import ParticleRow

    result = await _extract(TURTLE, "/tmp/coins.ttl")
    particle = candidate_to_particle(
        result.candidates[0], "entry-1", "snapshot-1", "rdf-extractor", subject_ids=["uuid-1"]
    )
    restored = ParticleRow.from_model(particle).to_model()

    assert restored.canonical_form is CanonicalForm.STRUCTURED
    assert restored.content == particle.content
    assert restored.structured_claim is not None
    assert restored.structured_claim.structurizer_id == EXTRACTOR_ID
    assert restored.structured_claim.subject.value == (
        particle.structured_claim.subject.value if particle.structured_claim else None
    )


def test_candidate_claiming_structured_without_a_triple_is_demoted_not_raised() -> None:
    """raising would lose every candidate in the pass to one
    malformed sibling. No shipped extractor can produce this state."""
    from particles.core.schema import UncertaintyNature
    from particles.extraction.general import CandidateParticle, candidate_to_particle

    candidate = CandidateParticle(
        content="a claim with no triple",
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        canonical_form=CanonicalForm.STRUCTURED,
        structured_claim=None,
    )
    particle = candidate_to_particle(candidate, "entry-1", "snapshot-1", "some-extractor")
    assert particle.canonical_form is CanonicalForm.PROSE
    assert particle.content == "a claim with no triple"


def test_bind_subject_id_name_rung_still_wins_and_uri_rung_is_optional() -> None:
    from particles.core.schema import ClaimTerm, StructuredClaim
    from particles.extraction.structure import bind_subject_id

    claim = StructuredClaim(
        subject=ClaimTerm(kind=TermKind.TOKEN, value="Berlin"),
        predicate=ClaimTerm(kind=TermKind.TOKEN, value="p"),
        object=ClaimTerm(kind=TermKind.LITERAL, value="v"),
        structurizer_id="content-structurizer",
        structurizer_version="1.0.0",
    )
    assert bind_subject_id(claim, ["berlin"], ["uuid-1"]).subject_id == "uuid-1"
    # No match, no refs: subject_id stays None, which lint deliberately
    # does not flag.
    assert bind_subject_id(claim, ["Paris"], ["uuid-2"]).subject_id is None


# ---------------------------------------------------------------------------
# Hostile input (§3.10)
# ---------------------------------------------------------------------------


def test_strip_remote_contexts_drops_remote_urls_and_keeps_inline() -> None:
    doc = json.dumps(
        {"@context": ["https://schema.org/", {"ex": "http://example.org/"}], "@id": "ex:a"}
    ).encode()
    cleaned, dropped = strip_remote_contexts(doc)
    assert dropped == ["https://schema.org/"]
    assert json.loads(cleaned)["@context"] == [{"ex": "http://example.org/"}]


def test_strip_remote_contexts_walks_nested_nodes() -> None:
    doc = json.dumps(
        {"@id": "x", "child": {"@context": "http://evil.internal/ctx", "@id": "y"}}
    ).encode()
    _cleaned, dropped = strip_remote_contexts(doc)
    assert dropped == ["http://evil.internal/ctx"]


def test_strip_remote_contexts_passes_non_json_through_untouched() -> None:
    content = b"@prefix ex: <http://example.org/> ."
    cleaned, dropped = strip_remote_contexts(content)
    assert cleaned == content
    assert dropped == []


@pytest.mark.parametrize(
    "context_value",
    [
        "file:///etc/hostname",
        "ftp://evil.internal/ctx.jsonld",
        "//evil.internal/ctx.jsonld",
        "context.jsonld",
        "HTTPS://Schema.Org/",
    ],
)
def test_every_string_context_is_dropped_whatever_the_scheme(context_value: str) -> None:
    """A string ``@context`` is a retrieval instruction, so the *type* decides.

    ``_is_remote``'s ``http(s)://`` prefix test was the whole bypass: a
    ``file://`` context is a local file read, ``ftp://`` is still a fetch, and
    a relative reference resolves against the document base into either.
    """
    doc = json.dumps({"@context": context_value, "@id": "http://example.org/a"}).encode()
    _cleaned, dropped = strip_remote_contexts(doc)
    assert dropped == [context_value]


def test_import_inside_a_context_object_is_dropped() -> None:
    """``@import`` retrieves from *inside* the context, where a pass over
    ``@context`` values never looks."""
    doc = json.dumps(
        {
            "@context": {"@version": 1.1, "@import": "file:///etc/hostname", "ex": "http://ex/"},
            "@id": "http://example.org/a",
        }
    ).encode()
    cleaned, dropped = strip_remote_contexts(doc)
    assert dropped == ["file:///etc/hostname"]
    context = json.loads(cleaned)["@context"]
    assert "@import" not in context
    # The rest of the inline context is untouched.
    assert context == {"@version": 1.1, "ex": "http://ex/"}


def test_scoped_context_inside_a_term_definition_is_dropped() -> None:
    """JSON-LD 1.1 lets a term definition carry its own ``@context``."""
    doc = json.dumps(
        {
            "@context": {
                "ex": "http://ex/",
                "child": {"@id": "http://ex/child", "@context": "file:///etc/hostname"},
            },
            "@id": "http://example.org/a",
        }
    ).encode()
    cleaned, dropped = strip_remote_contexts(doc)
    assert dropped == ["file:///etc/hostname"]
    context = json.loads(cleaned)["@context"]
    assert context["child"] == {"@id": "http://ex/child"}


def test_inline_context_terms_including_relative_iris_survive() -> None:
    """The scrub decides on the ``@context`` *value*, never on term definitions.

    A relative IRI inside an inline context is ordinary vocabulary, not a
    retrieval — dropping it would break every document that uses ``@base``.
    """
    doc = json.dumps(
        {
            "@context": {"@base": "http://example.org/", "ex": "http://ex/", "rel": "path/to#t"},
            "@id": "a",
        }
    ).encode()
    cleaned, dropped = strip_remote_contexts(doc)
    assert dropped == []
    assert cleaned == doc  # byte-identical: no re-serialization when nothing is dropped
    assert json.loads(cleaned)["@context"]["rel"] == "path/to#t"


async def test_remote_context_drop_is_disclosed_in_quality_notes() -> None:
    doc = json.dumps(
        {
            "@context": "https://schema.org/",
            "@id": "http://example.org/a",
            "http://example.org/p": "v",
        }
    ).encode()
    result = await _extract(doc, "/tmp/x.jsonld")
    assert any("Dropped non-inline JSON-LD @context" in n for n in result.quality_notes)
    assert any("schema.org" in n for n in result.quality_notes)


# --- the structural guard: no non-inline source is retrieved, ever -----------


def test_guard_refuses_a_context_fetch_the_scrub_never_saw() -> None:
    """The guard, not the scrub, is what makes no-network structural.

    Driving rdflib directly is the point: it proves the seam is closed for any
    document shape the scrub does not recognise, rather than re-testing the
    scrub through the extractor.
    """
    import rdflib

    doc = json.dumps({"@context": "file:///etc/hostname", "@id": "http://example.org/a"})
    with no_remote_retrieval() as refused, pytest.raises(RemoteRetrievalRefused):
        rdflib.Graph().parse(data=doc, format="json-ld")
    assert refused == ["file:///etc/hostname"]


def test_guard_leaves_a_legitimate_json_ld_parse_alone() -> None:
    """Only the *context* fetch seam is closed — the parser's own read of the
    deposited bytes goes through the same function name and must still work."""
    import rdflib

    doc = json.dumps(
        {
            "@context": {"name": "http://example.org/name"},
            "@id": "http://example.org/a",
            "name": "x",
        }
    )
    with no_remote_retrieval() as refused:
        graph = rdflib.Graph()
        graph.parse(data=doc, format="json-ld")
    assert refused == []
    assert len(graph) == 1


def test_guard_restores_module_globals_and_nests() -> None:
    from rdflib.plugins.parsers import rdfxml
    from rdflib.plugins.shared.jsonld import context as jsonld_context

    before = (jsonld_context.source_to_json, rdfxml.create_parser)
    with no_remote_retrieval():
        assert jsonld_context.source_to_json is not before[0]
        with no_remote_retrieval():
            assert rdfxml.create_parser is not before[1]
        # The inner exit restores what *it* replaced, not the original.
        assert jsonld_context.source_to_json is not before[0]
    assert (jsonld_context.source_to_json, rdfxml.create_parser) == before


def test_guard_restores_module_globals_after_an_exception() -> None:
    from rdflib.plugins.shared.jsonld import context as jsonld_context

    before = jsonld_context.source_to_json
    with pytest.raises(ValueError), no_remote_retrieval():
        raise ValueError("boom")
    assert jsonld_context.source_to_json is before


async def test_xxe_entity_is_not_expanded_and_no_error_escapes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """RDF/XML is a second parser the JSON scrub cannot see (§3.10).

    Asserts the file is not read *and* that the parse still returns the
    established ``ExtractionResult`` shape rather than raising.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("SUPERSECRETVALUE")
    doc = f"""<?xml version="1.0"?>
<!DOCTYPE rdf:RDF [ <!ENTITY xxe SYSTEM "file://{secret}"> ]>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:ex="http://example.org/">
  <rdf:Description rdf:about="http://example.org/a">
    <ex:leaked>&xxe;</ex:leaked>
    <ex:benign>still here</ex:benign>
  </rdf:Description>
</rdf:RDF>""".encode()

    result = await _extract(doc, "/tmp/x.rdf")

    rendered = " ".join(c.content for c in result.candidates)
    assert "SUPERSECRETVALUE" not in rendered
    assert "SUPERSECRETVALUE" not in " ".join(result.quality_notes)
    # The benign half of the same document still extracts.
    assert "a benign: still here" in [c.content for c in result.candidates]


async def test_a_hostile_document_opens_no_socket(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The no-network assertion, made positively: any connection attempt fails
    the test, rather than the test trusting that the refusal happened."""
    import socket

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("the RDF extractor attempted a network connection")

    monkeypatch.setattr(socket, "getaddrinfo", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr(socket.socket, "connect", _forbidden)

    secret = tmp_path / "secret.txt"
    secret.write_text("SUPERSECRETVALUE")
    doc = json.dumps(
        {
            "@context": [
                "https://schema.org/",
                f"file://{secret}",
                {"@version": 1.1, "@import": "http://evil.internal/ctx.jsonld"},
            ],
            "@id": "http://example.org/a",
            "http://example.org/p": "benign value",
        }
    ).encode()

    result = await _extract(doc, "/tmp/x.jsonld")

    # Every hostile reference is disclosed, and the benign triple survives.
    notes = " ".join(result.quality_notes)
    assert "schema.org" in notes
    assert str(secret) in notes
    assert "evil.internal" in notes
    assert [c.content for c in result.candidates] == ["a p: benign value"]
    assert "SUPERSECRETVALUE" not in " ".join(c.content for c in result.candidates)


async def test_oversize_document_is_rejected_before_the_parser(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("PARTICLES_CONFIG", "/nonexistent-test-config.yaml")
    from particles.config import get_config, reset_config

    reset_config()
    cfg = get_config()
    object.__setattr__(cfg.rdf, "max_bytes", 10)
    try:
        result = await _extract(TURTLE, "/tmp/x.ttl")
        assert result.candidates == []
        assert any("max_bytes" in n for n in result.quality_notes)
    finally:
        reset_config()


async def test_triple_cap_truncates_and_discloses(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from particles.config import get_config, reset_config

    reset_config()
    cfg = get_config()
    object.__setattr__(cfg.rdf, "max_triples", 2)
    try:
        result = await _extract(TURTLE, "/tmp/coins.ttl")
        assert len(result.candidates) == 2
        assert any("max_triples" in n for n in result.quality_notes)
    finally:
        reset_config()


async def test_unparseable_content_returns_notes_not_an_exception() -> None:
    result = await _extract(b"this is not RDF at all {{{ ><", "/tmp/x.ttl")
    assert result.candidates == []
    assert result.quality_notes


async def test_empty_graph_reports_no_emittable_triples() -> None:
    result = await _extract(b"@prefix ex: <http://example.org/> .", "/tmp/x.ttl")
    assert result.candidates == []
    assert result.quality_notes


# ---------------------------------------------------------------------------
# Determinism — the property the whole design rests on
# ---------------------------------------------------------------------------


async def test_two_extractions_of_the_same_bytes_agree_exactly() -> None:
    first = await _extract(TURTLE, "/tmp/coins.ttl")
    second = await _extract(TURTLE, "/tmp/coins.ttl")
    assert [c.content for c in first.candidates] == [c.content for c in second.candidates]
    assert [c.confidence_value for c in first.candidates] == [
        c.confidence_value for c in second.candidates
    ]
