"""Tests for Numista-specific extractors."""

from __future__ import annotations

import json

import pytest

from particles.core.schema import ExtractionStatus, Snapshot
from particles.extraction.numista import SOURCE_TYPE_COIN, SOURCE_TYPE_ISSUER, SOURCE_TYPE_LISTING


def _snapshot() -> Snapshot:
    return Snapshot(content_hash="a" * 64, extraction_status=ExtractionStatus.PENDING)


# ---------------------------------------------------------------------------
# Fixtures — minimal but realistic blobs
# ---------------------------------------------------------------------------

_COIN_JSON: dict = {
    "id": 3022,
    "title": "5 Pfennigs",
    "min_year": 1948,
    "max_year": 1950,
    "issuer": {"code": "ddr", "name": "German Democratic Republic"},
    "value": {
        "text": "5 Pfennigs",
        "currency": {"id": 229, "name": "Mark", "full_name": "Mark (1948-1990)"},
    },
    "type": "Standard circulation coins",
    "category": "coin",
    "composition": {"text": "Aluminium"},
    "weight": 1.1,
    "size": 19.0,
    "thickness": 1.5,
    "shape": {"text": "Round"},
    "technique": {"text": "Milled"},
    "orientation": "medal",
    "edge": {"description": "Plain"},
    "demonetization": {"is_demonetized": True, "demonetization_date": "1990-12-31"},
    "references": [
        {"catalogue": {"code": "KM"}, "number": "2"},
        {"catalogue": {"code": "Schön"}, "number": "2"},
    ],
    "obverse": {"description": "Ear of rye"},
    "reverse": {"description": "Face value"},
    "mints": [{"id": 5, "name": "Berlin"}],
    "comments": "",
}

# HTML for a two-coin listing page (description_piece divs)
# HTML matches real Numista structure: title and years are on separate text lines
# around the <br />, not on the same text line (lxml text_content() drops <br> tags)
_LISTING_HTML: bytes = """
<html><body>
<div class="description_piece">
  <img src="https://en.numista.com/design/pays/ddr.gif" alt="" title="German Democratic Republic" />
  <strong><a href="/3022">
    5 Pfennigs                    <br />
    1948-1950
  </a></strong><br />
  <em>Coins › Standard circulation coins</em><br />
  Aluminium &bull; 1.1&nbsp;g &bull; &#8960;&nbsp;19&nbsp;mm<br />
  KM#&#8239;2, Sch&ouml;n#&#8239;2, N#&#8239;3022
</div>
<div class="description_piece">
  <img src="https://en.numista.com/design/pays/ddr.gif" alt="" title="German Democratic Republic" />
  <strong><a href="/8562">
    1 Pfennig                    <br />
    1948-1950
  </a></strong><br />
  <em>Coins › Standard circulation coins</em><br />
  Aluminium &bull; 0.75&nbsp;g &bull; &#8960;&nbsp;17&nbsp;mm<br />
  KM#&#8239;1, Sch&ouml;n#&#8239;1, N#&#8239;8562
</div>
</body></html>
""".encode()

_ISSUER_ENVELOPE: dict = {
    "issuer_code": "ddr",
    "issuer_name": "German Democratic Republic",
    "total_count": 1,
    "pages_fetched": 1,
    "object_type_ids": [],
    "types": [
        {
            "id": 3022,
            "title": "5 Pfennigs",
            "min_year": 1948,
            "max_year": 1950,
            "object_type": {"id": 1, "name": "Standard circulation coins"},
            "issuer": {"code": "ddr", "name": "German Democratic Republic"},
            "category": "coin",
            "composition": {"text": "Aluminium"},
            "weight": 1.1,
            "size": 19.0,
            "references": [{"catalogue": {"code": "KM"}, "number": "2"}],
        }
    ],
}


# ---------------------------------------------------------------------------
# NumistaListingExtractor
# ---------------------------------------------------------------------------


class TestNumistaListingExtractor:
    def test_accepts_listing_source_type(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        ext = NumistaListingExtractor()
        assert ext.accepts(SOURCE_TYPE_LISTING)
        assert not ext.accepts(SOURCE_TYPE_COIN)

    @pytest.mark.asyncio
    async def test_extracts_two_coins(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        result = await NumistaListingExtractor().extract(_snapshot(), _LISTING_HTML)
        assert len(result.candidates) == 2

    @pytest.mark.asyncio
    async def test_coin_label_built_correctly(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        result = await NumistaListingExtractor().extract(_snapshot(), _LISTING_HTML)
        labels = [c.content.split(":")[0] for c in result.candidates]
        assert "5 Pfennigs (1948-1950) GDR" in labels
        assert "1 Pfennig (1948-1950) GDR" in labels

    @pytest.mark.asyncio
    async def test_physical_properties_parsed(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        result = await NumistaListingExtractor().extract(_snapshot(), _LISTING_HTML)
        pfennig5 = next(c for c in result.candidates if "5 Pfennig" in c.content)
        assert pfennig5.properties is not None
        assert pfennig5.properties["nmo:hasMaterial"] == "Aluminium"
        assert pfennig5.properties["nmo:hasWeight"] == 1.1
        assert pfennig5.properties["nmo:hasDiameter"] == 19.0

    @pytest.mark.asyncio
    async def test_catalog_refs_parsed(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        result = await NumistaListingExtractor().extract(_snapshot(), _LISTING_HTML)
        pfennig5 = next(c for c in result.candidates if "5 Pfennig" in c.content)
        refs = pfennig5.properties["nuds:references"]
        assert isinstance(refs, list)
        assert any("KM" in r for r in refs)

    @pytest.mark.asyncio
    async def test_type_and_issuer_present(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        result = await NumistaListingExtractor().extract(_snapshot(), _LISTING_HTML)
        pfennig5 = next(c for c in result.candidates if "5 Pfennig" in c.content)
        assert pfennig5.properties["nmo:hasObjectType"] == "Standard circulation coins"
        assert pfennig5.properties["nmo:hasIssuer"] == "German Democratic Republic"

    @pytest.mark.asyncio
    async def test_numista_url_constructed(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        result = await NumistaListingExtractor().extract(_snapshot(), _LISTING_HTML)
        pfennig5 = next(c for c in result.candidates if "5 Pfennig" in c.content)
        assert (
            pfennig5.properties["numista:url"] == "https://en.numista.com/catalogue/pieces3022.html"
        )
        assert pfennig5.properties["numista:id"] == "3022"

    @pytest.mark.asyncio
    async def test_subject_class_coin(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        result = await NumistaListingExtractor().extract(_snapshot(), _LISTING_HTML)
        pfennig5 = next(c for c in result.candidates if "5 Pfennig" in c.content)
        assert pfennig5.subject_classes.get("5 Pfennigs (1948-1950) GDR") == "nmo:NumismaticObject"

    @pytest.mark.asyncio
    async def test_subject_class_material(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        result = await NumistaListingExtractor().extract(_snapshot(), _LISTING_HTML)
        pfennig5 = next(c for c in result.candidates if "5 Pfennig" in c.content)
        assert pfennig5.subject_classes.get("Aluminium") == "nmo:Material"

    @pytest.mark.asyncio
    async def test_issuer_code_from_flag_img(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        # Issuer code "ddr" maps to suffix "GDR" via _ISSUER_SUFFIX
        result = await NumistaListingExtractor().extract(_snapshot(), _LISTING_HTML)
        assert any("GDR" in c.content for c in result.candidates)

    @pytest.mark.asyncio
    async def test_empty_html_returns_quality_note(self) -> None:
        from particles.extraction.numista import NumistaListingExtractor

        result = await NumistaListingExtractor().extract(_snapshot(), b"<html><body></body></html>")
        assert result.candidates == []
        assert result.quality_notes


# ---------------------------------------------------------------------------
# NumistaCoinExtractor
# ---------------------------------------------------------------------------


class TestNumistaCoinExtractor:
    def test_accepts_coin_source_type(self) -> None:
        from particles.extraction.numista import NumistaCoinExtractor

        ext = NumistaCoinExtractor()
        assert ext.accepts(SOURCE_TYPE_COIN)
        assert not ext.accepts(SOURCE_TYPE_ISSUER)

    @pytest.mark.asyncio
    async def test_structured_particle_produced(self) -> None:
        from particles.extraction.numista import NumistaCoinExtractor

        content = json.dumps(_COIN_JSON).encode()
        result = await NumistaCoinExtractor().extract(_snapshot(), content)
        structured = [c for c in result.candidates if c.properties]
        assert len(structured) == 1

    @pytest.mark.asyncio
    async def test_physical_properties(self) -> None:
        from particles.extraction.numista import NumistaCoinExtractor

        content = json.dumps(_COIN_JSON).encode()
        result = await NumistaCoinExtractor().extract(_snapshot(), content)
        p = next(c for c in result.candidates if c.properties)
        assert p.properties["nmo:hasWeight"] == 1.1
        assert p.properties["nmo:hasDiameter"] == 19.0
        assert p.properties["nmo:hasMaterial"] == "Aluminium"

    @pytest.mark.asyncio
    async def test_demonetization_dict_parsed(self) -> None:
        from particles.extraction.numista import NumistaCoinExtractor

        content = json.dumps(_COIN_JSON).encode()
        result = await NumistaCoinExtractor().extract(_snapshot(), content)
        p = next(c for c in result.candidates if c.properties)
        assert p.properties["nuds:demonetizationDate"] == "1990-12-31"

    @pytest.mark.asyncio
    async def test_currency_from_nested_value(self) -> None:
        from particles.extraction.numista import NumistaCoinExtractor

        content = json.dumps(_COIN_JSON).encode()
        result = await NumistaCoinExtractor().extract(_snapshot(), content)
        p = next(c for c in result.candidates if c.properties)
        assert p.properties["nmo:hasDenomination"] == "Mark (1948-1990)"

    @pytest.mark.asyncio
    async def test_type_field_over_category(self) -> None:
        from particles.extraction.numista import NumistaCoinExtractor

        content = json.dumps(_COIN_JSON).encode()
        result = await NumistaCoinExtractor().extract(_snapshot(), content)
        p = next(c for c in result.candidates if c.properties)
        assert p.properties["nmo:hasObjectType"] == "Standard circulation coins"

    @pytest.mark.asyncio
    async def test_subject_class_numismatic_object(self) -> None:
        from particles.extraction.numista import NumistaCoinExtractor

        content = json.dumps(_COIN_JSON).encode()
        result = await NumistaCoinExtractor().extract(_snapshot(), content)
        p = next(c for c in result.candidates if c.properties)
        assert p.subject_classes.get("5 Pfennigs (1948-1950) GDR") == "nmo:NumismaticObject"
        assert p.subject_classes.get("Aluminium") == "nmo:Material"

    @pytest.mark.asyncio
    async def test_descriptive_particles_have_no_properties(self) -> None:
        from particles.extraction.numista import NumistaCoinExtractor

        content = json.dumps(_COIN_JSON).encode()
        result = await NumistaCoinExtractor().extract(_snapshot(), content)
        descriptive = [c for c in result.candidates if not c.properties]
        assert len(descriptive) > 0
        assert any("obverse" in c.content.lower() for c in descriptive)

    # -- the per-candidate structure-canonical rule ---------------

    @pytest.mark.asyncio
    async def test_infobox_is_prose_canonical_with_a_type_triple(self) -> None:
        """The infobox states many facts, so it may not be STRUCTURED (§2.3)."""
        from particles.core.schema import CanonicalForm, TermKind
        from particles.extraction.numista import NumistaCoinExtractor

        content = json.dumps(_COIN_JSON).encode()
        result = await NumistaCoinExtractor().extract(_snapshot(), content)
        infobox = next(c for c in result.candidates if c.properties)

        assert infobox.canonical_form is CanonicalForm.PROSE
        claim = infobox.structured_claim
        assert claim is not None
        # Derived from subject_classes, not from the prose — content never
        # says "is a numismatic object". Legal because PROSE-canonical
        # particles may carry an annotation.
        assert claim.predicate.value == "rdf:type"
        assert claim.object.value == "nmo:NumismaticObject"
        assert claim.subject.kind is TermKind.TOKEN

    @pytest.mark.asyncio
    async def test_templated_candidates_are_structure_canonical(self) -> None:
        from particles.core.schema import CanonicalForm
        from particles.extraction.numista import NumistaCoinExtractor
        from particles.extraction.numista._shared import (
            EXTRACTOR_ID_COIN,
            EXTRACTOR_VERSION_COIN,
        )

        content = json.dumps(_COIN_JSON).encode()
        result = await NumistaCoinExtractor().extract(_snapshot(), content)
        structured = [c for c in result.candidates if c.canonical_form is CanonicalForm.STRUCTURED]
        assert structured, "the per-field templated candidates must be STRUCTURED"

        for cand in structured:
            claim = cand.structured_claim
            assert claim is not None, "STRUCTURED requires a triple"
            assert claim.structurizer_id == EXTRACTOR_ID_COIN
            assert claim.structurizer_version == EXTRACTOR_VERSION_COIN

        by_predicate = {c.structured_claim.predicate.value: c for c in structured}  # type: ignore[union-attr]
        assert "nuds:references" in by_predicate
        assert "has obverse description" in by_predicate

    @pytest.mark.asyncio
    async def test_published_curies_are_uri_terms_and_verb_phrases_are_tokens(self) -> None:
        """§2.2: no ontology term is guessed — unaligned predicates stay TOKEN."""
        from particles.core.schema import TermKind
        from particles.extraction.numista import NumistaCoinExtractor

        content = json.dumps(_COIN_JSON).encode()
        result = await NumistaCoinExtractor().extract(_snapshot(), content)
        kinds = {
            c.structured_claim.predicate.value: c.structured_claim.predicate.kind
            for c in result.candidates
            if c.structured_claim is not None
        }
        # The alignment table supplies these, and their prefixes are
        # published in context.jsonld.
        assert kinds["nuds:references"] is TermKind.URI
        assert kinds["rdf:type"] is TermKind.URI
        # It supplies no mint or engraver predicate, so neither is invented.
        assert kinds["was struck at"] is TermKind.TOKEN

    @pytest.mark.asyncio
    async def test_mintmark_sentence_states_a_second_fact_so_stays_prose(self) -> None:
        """A mintmark suffix makes content say more than its triple (§2.1)."""
        from particles.core.schema import CanonicalForm
        from particles.extraction.numista import NumistaCoinExtractor

        marked = json.loads(json.dumps(_COIN_JSON))
        marked["mints"] = [{"name": "Berlin", "mark": "A"}]
        result = await NumistaCoinExtractor().extract(_snapshot(), json.dumps(marked).encode())
        mint = next(c for c in result.candidates if "was struck at" in c.content)
        assert "mintmark" in mint.content
        assert mint.canonical_form is CanonicalForm.PROSE
        assert mint.structured_claim is not None

        unmarked = json.loads(json.dumps(_COIN_JSON))
        unmarked["mints"] = [{"name": "Berlin"}]
        result = await NumistaCoinExtractor().extract(_snapshot(), json.dumps(unmarked).encode())
        mint = next(c for c in result.candidates if "was struck at" in c.content)
        assert mint.canonical_form is CanonicalForm.STRUCTURED

    @pytest.mark.asyncio
    async def test_verbatim_source_prose_carries_no_annotation(self) -> None:
        """`comments` is the source's own words — asserted, not derived."""
        from particles.core.schema import CanonicalForm
        from particles.extraction.numista import NumistaCoinExtractor

        data = json.loads(json.dumps(_COIN_JSON))
        data["comments"] = "Struck during the currency reform; see the 1948 decree."
        result = await NumistaCoinExtractor().extract(_snapshot(), json.dumps(data).encode())
        comment = next(c for c in result.candidates if c.content == data["comments"])
        assert comment.canonical_form is CanonicalForm.PROSE
        assert comment.structured_claim is None

    @pytest.mark.asyncio
    async def test_demonetization_string_also_handled(self) -> None:
        from particles.extraction.numista import NumistaCoinExtractor

        data = {**_COIN_JSON, "demonetization": "1990-12-31"}
        result = await NumistaCoinExtractor().extract(_snapshot(), json.dumps(data).encode())
        p = next(c for c in result.candidates if c.properties)
        assert p.properties["nuds:demonetizationDate"] == "1990-12-31"

    @pytest.mark.asyncio
    async def test_generic_category_coin_not_emitted_as_type(self) -> None:
        from particles.extraction.numista import NumistaCoinExtractor

        # When only category="coin" and no type field, hasObjectType should not be set
        data = {**_COIN_JSON}
        del data["type"]
        result = await NumistaCoinExtractor().extract(_snapshot(), json.dumps(data).encode())
        p = next(c for c in result.candidates if c.properties)
        assert "nmo:hasObjectType" not in p.properties


# ---------------------------------------------------------------------------
# NumistaIssuerExtractor
# ---------------------------------------------------------------------------


class TestNumistaIssuerExtractor:
    def test_accepts_issuer_source_type(self) -> None:
        from particles.extraction.numista import NumistaIssuerExtractor

        ext = NumistaIssuerExtractor()
        assert ext.accepts(SOURCE_TYPE_ISSUER)
        assert not ext.accepts(SOURCE_TYPE_COIN)

    @pytest.mark.asyncio
    async def test_structured_particles_produced(self) -> None:
        from particles.extraction.numista import NumistaIssuerExtractor

        content = json.dumps(_ISSUER_ENVELOPE).encode()
        result = await NumistaIssuerExtractor().extract(_snapshot(), content)
        structured = [c for c in result.candidates if c.properties]
        assert len(structured) == 1

    @pytest.mark.asyncio
    async def test_physical_properties_from_envelope(self) -> None:
        from particles.extraction.numista import NumistaIssuerExtractor

        content = json.dumps(_ISSUER_ENVELOPE).encode()
        result = await NumistaIssuerExtractor().extract(_snapshot(), content)
        p = next(c for c in result.candidates if c.properties)
        assert p.properties["nmo:hasMaterial"] == "Aluminium"
        assert p.properties["nmo:hasWeight"] == 1.1
        assert p.properties["nmo:hasDiameter"] == 19.0

    @pytest.mark.asyncio
    async def test_subject_class_set(self) -> None:
        from particles.extraction.numista import NumistaIssuerExtractor

        content = json.dumps(_ISSUER_ENVELOPE).encode()
        result = await NumistaIssuerExtractor().extract(_snapshot(), content)
        p = next(c for c in result.candidates if c.properties)
        assert p.subject_classes.get("5 Pfennigs (1948-1950) GDR") == "nmo:NumismaticObject"

    @pytest.mark.asyncio
    async def test_empty_types_returns_quality_note(self) -> None:
        from particles.extraction.numista import NumistaIssuerExtractor

        envelope = {**_ISSUER_ENVELOPE, "types": "not-a-list"}
        result = await NumistaIssuerExtractor().extract(_snapshot(), json.dumps(envelope).encode())
        assert result.candidates == []
        assert result.quality_notes
