"""Tests for the Nomisma extractor and importer (naming)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.core.schema import ExternalRef, Snapshot, UncertaintyNature
from particles.extraction.nomisma import (
    SOURCE_TYPE,
    NomismaExtractor,
    _concept_id_from_uri,
    _expand_id,
    _find_main_node,
    _get_en,
    _get_ids,
    _get_typed_decimal,
    _parse_graph,
)
from particles.ingest.importers.nomisma import NomismaImporter
from tests._capped_http import set_capped_responses

# ---------------------------------------------------------------------------
# Fixtures — match Nomisma's real JSON-LD format
# ---------------------------------------------------------------------------

_ALUMINIUM_DOC: dict = {
    "@context": {
        "nm": "http://nomisma.org/id/",
        "nmo": "http://nomisma.org/ontology#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
    },
    "@graph": [
        {
            "@id": "nm:al",
            "@type": ["nmo:Material", "skos:Concept"],
            "skos:prefLabel": [
                {"@language": "en", "@value": "Aluminum"},
                {"@language": "de", "@value": "Aluminium"},
            ],
            "skos:definition": {
                "@language": "en",
                "@value": "Aluminum is a chemical element in the boron group.",
            },
            "skos:exactMatch": [
                {"@id": "http://www.wikidata.org/entity/Q663"},
                {"@id": "http://vocab.getty.edu/aat/300011015"},
            ],
        }
    ],
}

_BERLIN_DOC: dict = {
    "@context": {
        "nm": "http://nomisma.org/id/",
        "nmo": "http://nomisma.org/ontology#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
    },
    "@graph": [
        {
            "@id": "nm:berlin",
            "@type": ["nmo:Mint", "skos:Concept"],
            "skos:prefLabel": [{"@language": "en", "@value": "Berlin"}],
            "skos:definition": {
                "@language": "en",
                "@value": "The mint(s) at the City of Berlin.",
            },
            "geo:location": {"@id": "http://nomisma.org/id/berlin#this"},
            "skos:closeMatch": [
                {"@id": "http://www.wikidata.org/entity/Q64"},
                {"@id": "http://dbpedia.org/resource/Berlin"},
            ],
        },
        {
            "@id": "http://nomisma.org/id/berlin#this",
            "@type": "geo:SpatialThing",
            "geo:lat": {"@type": "xsd:decimal", "@value": "52.52437"},
            "geo:long": {"@type": "xsd:decimal", "@value": "13.41053"},
        },
    ],
}


def _snapshot() -> Snapshot:
    return Snapshot(snapshot_id="snap-1", content_hash="abc", archive_path="/tmp/abc")


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_get_en_from_list(self) -> None:
        values = [
            {"@language": "de", "@value": "Aluminium"},
            {"@language": "en", "@value": "Aluminum"},
        ]
        assert _get_en(values) == "Aluminum"

    def test_get_en_from_single_dict(self) -> None:
        assert _get_en({"@language": "en", "@value": "Aluminum"}) == "Aluminum"

    def test_get_en_wrong_language(self) -> None:
        assert _get_en({"@language": "de", "@value": "Aluminium"}) is None

    def test_get_ids_list(self) -> None:
        values = [{"@id": "http://example.org/a"}, {"@id": "http://example.org/b"}]
        assert _get_ids(values) == ["http://example.org/a", "http://example.org/b"]

    def test_get_ids_single(self) -> None:
        assert _get_ids({"@id": "http://example.org/a"}) == ["http://example.org/a"]

    def test_get_typed_decimal(self) -> None:
        assert _get_typed_decimal({"@type": "xsd:decimal", "@value": "52.52437"}) == pytest.approx(
            52.52437
        )

    def test_get_typed_decimal_bare_float(self) -> None:
        assert _get_typed_decimal(52.5) == 52.5

    def test_parse_graph_extracts_nodes_and_prefix_map(self) -> None:
        content = json.dumps(_ALUMINIUM_DOC).encode()
        nodes, prefix_map = _parse_graph(content)
        assert len(nodes) == 1
        assert prefix_map["nm"] == "http://nomisma.org/id/"

    def test_expand_id_nm_prefix(self) -> None:
        prefix_map = {"nm": "http://nomisma.org/id/"}
        assert _expand_id("nm:al", prefix_map) == "http://nomisma.org/id/al"

    def test_expand_id_already_absolute(self) -> None:
        assert _expand_id("http://nomisma.org/id/al", {}) == "http://nomisma.org/id/al"

    def test_find_main_node_picks_nm_prefixed(self) -> None:
        content = json.dumps(_ALUMINIUM_DOC).encode()
        nodes, prefix_map = _parse_graph(content)
        node = _find_main_node(nodes, prefix_map)
        assert node is not None
        assert node["@id"] == "nm:al"

    def test_concept_id_from_nm_prefix(self) -> None:
        assert _concept_id_from_uri("nm:al") == "al"
        assert _concept_id_from_uri("http://nomisma.org/id/berlin") == "berlin"


# ---------------------------------------------------------------------------
# NomismaExtractor
# ---------------------------------------------------------------------------


class TestNomismaExtractor:
    def test_accepts_only_nomisma_source_type(self) -> None:
        e = NomismaExtractor()
        assert e.accepts(SOURCE_TYPE)
        assert not e.accepts("WIKIDATA_API")
        assert not e.accepts("WEB_PAGE")

    @pytest.mark.asyncio
    async def test_extracts_material_particle(self) -> None:
        e = NomismaExtractor()
        content = json.dumps(_ALUMINIUM_DOC).encode()
        result = await e.extract(_snapshot(), content)

        assert len(result.candidates) == 1
        c = result.candidates[0]
        assert "Aluminum" in c.content
        assert c.uncertainty_nature == UncertaintyNature.EPISTEMIC
        assert c.confidence_value == 0.99
        assert c.subjects == ["Aluminum"]
        assert c.subject_classes == {"Aluminum": "nmo:Material"}

    # -- the structured claim on the concept particle -------------

    @pytest.mark.asyncio
    async def test_structured_claim_subject_is_the_entity_iri(self) -> None:
        """§2.2: the URI subject term is what lets the §3.7 bind rung work."""
        from particles.core.schema import CanonicalForm, TermKind
        from particles.extraction.structure import bind_subject_id

        e = NomismaExtractor()
        result = await e.extract(_snapshot(), json.dumps(_ALUMINIUM_DOC).encode())
        c = result.candidates[0]

        claim = c.structured_claim
        assert claim is not None
        assert claim.subject.kind is TermKind.URI
        assert claim.subject.value.startswith("http://nomisma.org/id/")
        assert claim.structurizer_id == "nomisma-extractor"
        # content renders the class AND the definition — two facts, one
        # triple — so it fails the §2.1 test and stays PROSE-canonical.
        assert c.canonical_form is CanonicalForm.PROSE

        # The label never matches the IRI, so rung 1 cannot bind; rung 2 does,
        # via the external ref this extractor attaches.
        bound = bind_subject_id(claim, c.subjects, ["subject-uuid-1"], c.external_refs)
        assert bound.subject_id == "subject-uuid-1"

        unbound = bind_subject_id(claim, c.subjects, ["subject-uuid-1"], None)
        assert unbound.subject_id is None

    @pytest.mark.asyncio
    async def test_definition_absent_falls_back_to_the_type_triple(self) -> None:
        from particles.core.schema import TermKind

        doc = json.loads(json.dumps(_ALUMINIUM_DOC))
        for node in doc["@graph"]:
            node.pop("skos:definition", None)
        result = await NomismaExtractor().extract(_snapshot(), json.dumps(doc).encode())

        claim = result.candidates[0].structured_claim
        assert claim is not None
        assert claim.predicate.value == "rdf:type"
        assert claim.object.value == "nmo:Material"
        assert claim.object.kind is TermKind.URI

    @pytest.mark.asyncio
    async def test_definition_predicate_is_an_absolute_iri(self) -> None:
        """`skos:` is not published in context.jsonld, so the CURIE is not used."""
        from particles.core.schema import TermKind

        result = await NomismaExtractor().extract(_snapshot(), json.dumps(_ALUMINIUM_DOC).encode())
        claim = result.candidates[0].structured_claim
        assert claim is not None
        assert claim.predicate.value == "http://www.w3.org/2004/02/skos/core#definition"
        assert claim.predicate.kind is TermKind.URI
        assert claim.object.language == "en"

    @pytest.mark.asyncio
    async def test_material_properties(self) -> None:
        e = NomismaExtractor()
        content = json.dumps(_ALUMINIUM_DOC).encode()
        result = await e.extract(_snapshot(), content)
        props = result.candidates[0].properties
        assert props is not None
        assert "Aluminum is a chemical element" in str(props["skos:definition"])
        assert props["skos:exactMatch"] == "http://www.wikidata.org/entity/Q663"

    @pytest.mark.asyncio
    async def test_external_ref_attached(self) -> None:
        e = NomismaExtractor()
        content = json.dumps(_ALUMINIUM_DOC).encode()
        result = await e.extract(_snapshot(), content)
        ref = result.candidates[0].external_refs.get("Aluminum")
        assert isinstance(ref, ExternalRef)
        assert ref.namespace == "nomisma"
        assert ref.id == "al"
        assert ref.uri == "http://nomisma.org/id/al"

    @pytest.mark.asyncio
    async def test_mint_with_geo_coordinates(self) -> None:
        e = NomismaExtractor()
        content = json.dumps(_BERLIN_DOC).encode()
        result = await e.extract(_snapshot(), content)
        assert len(result.candidates) == 1
        c = result.candidates[0]
        assert c.subject_classes == {"Berlin": "nmo:Mint"}
        props = c.properties
        assert props is not None
        assert props["geo:lat"] == pytest.approx(52.52437)
        assert props["geo:long"] == pytest.approx(13.41053)

    @pytest.mark.asyncio
    async def test_empty_graph_returns_quality_note(self) -> None:
        e = NomismaExtractor()
        content = json.dumps({"@context": {}, "@graph": []}).encode()
        result = await e.extract(_snapshot(), content)
        assert len(result.candidates) == 0
        assert result.quality_notes


# ---------------------------------------------------------------------------
# NomismaImporter
# ---------------------------------------------------------------------------


class TestNomismaImporter:
    def test_accepts_nomisma_urls(self) -> None:
        d = NomismaImporter()
        assert d.accepts_url("http://nomisma.org/id/al")
        assert d.accepts_url("https://nomisma.org/id/berlin")
        assert not d.accepts_url("https://www.wikidata.org/wiki/Q663")
        assert not d.accepts_url("https://en.numista.com/catalogue/pieces1234.html")

    @pytest.mark.asyncio
    async def test_deposit_calls_write_entry(self) -> None:
        content = json.dumps(_ALUMINIUM_DOC).encode()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = content
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, return_value=mock_resp)

        with (
            patch("particles.http.particles_client", return_value=mock_client),
            patch(
                "particles.corpus.deposit.write_entry_and_snapshot",
                new_callable=AsyncMock,
                return_value=("entry-1", "snap-1"),
            ) as mock_write,
            patch("particles.corpus.deposit.sha256", return_value="hash123"),
            patch("particles.corpus.deposit.save_blob", return_value="/tmp/hash123"),
        ):
            d = NomismaImporter()
            entry_id, snap_id = await d.deposit(AsyncMock(), "http://nomisma.org/id/al", "test", [])

        assert entry_id == "entry-1"
        call_kwargs = mock_write.call_args.kwargs
        assert call_kwargs["source_type"] == SOURCE_TYPE
        assert call_kwargs["uri_r"] == "http://nomisma.org/id/al"

    @pytest.mark.asyncio
    async def test_deposit_404_raises(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.content = b""

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, return_value=mock_resp)

        with patch("particles.http.particles_client", return_value=mock_client):
            d = NomismaImporter()
            with pytest.raises(ValueError, match="not found"):
                await d.deposit(AsyncMock(), "http://nomisma.org/id/nonexistent", "test", [])


# ---------------------------------------------------------------------------
# add_external_ref integration
# ---------------------------------------------------------------------------


class TestAddExternalRef:
    @pytest.mark.asyncio
    async def test_external_ref_attached_to_subject(self, db_session: object) -> None:
        from particles.core.schema import Subject
        from particles.store.subject_store import add_external_ref, insert_subject

        s = Subject(canonical_name="Aluminum", asserted_by="test")
        await insert_subject(db_session, s)  # type: ignore[arg-type]

        ref = ExternalRef(namespace="nomisma", id="al", uri="http://nomisma.org/id/al")
        assert await add_external_ref(db_session, s.id, ref) is True  # type: ignore[arg-type]
        assert await add_external_ref(db_session, s.id, ref) is False  # idempotent

    @pytest.mark.asyncio
    async def test_external_ref_missing_subject_returns_false(self, db_session: object) -> None:
        from particles.store.subject_store import add_external_ref

        ref = ExternalRef(namespace="nomisma", id="al", uri=None)
        assert await add_external_ref(db_session, "nonexistent-id", ref) is False  # type: ignore[arg-type]
