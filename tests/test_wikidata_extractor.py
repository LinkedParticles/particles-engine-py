"""Wikidata extractor tests — structure-canonical emission.

The label lookups the extractor performs for ``content`` are pre-seeded into its
in-process caches, so every test here is a deterministic parse of a fixed blob
with no network. That is not just test hygiene: the *triple* half is built from
the deposited identifiers alone, and pinning it against a seeded-label run is
what demonstrates the two halves are independent.
"""

from __future__ import annotations

from typing import Any

import pytest

from particles.core.schema import CanonicalForm, Snapshot, TermKind, WarcRecordType
from particles.extraction.wikidata import (
    EXTRACTOR_ID,
    EXTRACTOR_VERSION,
    SOURCE_TYPE,
    WD_ENTITY_PREFIX,
    WDT_PROP_PREFIX,
    WikidataExtractor,
    _object_term,
    _time_literal,
    entity_uri,
    external_ref_for,
)

_XSD = "http://www.w3.org/2001/XMLSchema#"

# A Douglas Adams blob trimmed to one statement per interesting value type.
ENTITY: dict[str, Any] = {
    "id": "Q42",
    "type": "item",
    "labels": {"en": "Douglas Adams"},
    "statements": {
        "P19": [
            {
                "rank": "normal",
                "property": {"id": "P19", "data_type": "wikibase-item"},
                "value": {"type": "value", "content": "Q350"},
            }
        ],
        "P569": [
            {
                "rank": "preferred",
                "property": {"id": "P569", "data_type": "time"},
                "value": {
                    "type": "value",
                    "content": {"time": "+1952-03-11T00:00:00Z", "precision": 11},
                },
            }
        ],
        "P2048": [
            {
                "rank": "normal",
                "property": {"id": "P2048", "data_type": "quantity"},
                "value": {"type": "value", "content": {"amount": "+1.96", "unit": "1"}},
            }
        ],
        "P1477": [
            {
                "rank": "normal",
                "property": {"id": "P1477", "data_type": "monolingualtext"},
                "value": {
                    "type": "value",
                    "content": {"text": "Douglas Noël Adams", "language": "en"},
                },
            }
        ],
        "P856": [
            {
                "rank": "normal",
                "property": {"id": "P856", "data_type": "url"},
                "value": {"type": "value", "content": "https://douglasadams.com/"},
            }
        ],
        "P214": [
            {
                "rank": "normal",
                "property": {"id": "P214", "data_type": "external-id"},
                "value": {"type": "value", "content": "113230702"},
            }
        ],
        "P1038": [
            {
                "rank": "deprecated",
                "property": {"id": "P1038", "data_type": "wikibase-item"},
                "value": {"type": "value", "content": "Q999999"},
            }
        ],
    },
}

LABELS = {
    "P19": "place of birth",
    "P569": "date of birth",
    "P2048": "height",
    "P1477": "birth name",
    "P856": "official website",
    "P214": "VIAF ID",
    # Seeded even though its only statement is deprecated: the property label is
    # fetched per property group, before the per-statement rank check.
    "P1038": "relative",
    "Q350": "Cambridge",
}


@pytest.fixture(autouse=True)
def _seeded_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-seed L1 so no test reaches the REST API."""
    from particles.extraction import wikidata as wd

    monkeypatch.setattr(
        wd, "_property_label_cache", {k: v for k, v in LABELS.items() if k.startswith("P")}
    )
    monkeypatch.setattr(
        wd, "_item_label_cache", {k: v for k, v in LABELS.items() if k.startswith("Q")}
    )


def _snapshot() -> Snapshot:
    return Snapshot(
        snapshot_id="00000000-0000-0000-0000-0000000000ff",
        content_hash="0" * 64,
        warc_record_type=WarcRecordType.RESPONSE,
    )


async def _extract(entity: dict[str, Any] | None = None) -> Any:
    import json

    payload = json.dumps(entity if entity is not None else ENTITY).encode()
    return await WikidataExtractor().extract(_snapshot(), payload)


def _by_property(result: Any) -> dict[str, Any]:
    """Index candidates by the P-id in their triple's predicate."""
    return {
        c.structured_claim.predicate.value.rsplit("/", 1)[1]: c
        for c in result.candidates
        if c.structured_claim is not None
    }


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_accepts_only_the_wikidata_source_type() -> None:
    extractor = WikidataExtractor()
    assert extractor.accepts(SOURCE_TYPE)
    assert not extractor.accepts("RDF_GRAPH")


def test_extractor_version_is_the_structure_canonical_one() -> None:
    """bumps the minor so ``reindex --extractor-version 0.1.0``
    discovers the prose-only particles the prior version minted."""
    assert EXTRACTOR_VERSION == "0.2.0"


# ---------------------------------------------------------------------------
# The triple is the assertion
# ---------------------------------------------------------------------------


async def test_every_candidate_is_structure_canonical_and_stamped() -> None:
    result = await _extract()
    assert result.candidates
    for candidate in result.candidates:
        assert candidate.canonical_form is CanonicalForm.STRUCTURED
        assert candidate.structured_claim is not None
        assert candidate.structured_claim.structurizer_id == EXTRACTOR_ID
        assert candidate.structured_claim.structurizer_version == EXTRACTOR_VERSION


async def test_triple_uris_are_the_ones_wikidata_published() -> None:
    """The export round-trip's precondition: the stored triple must be
    the source statement's own ``(Q-id, P-id, value)``, reachable back to
    Wikidata without a label round-trip. Pinned on the item-valued statement,
    where all three positions are IRIs."""
    result = await _extract()
    claim = _by_property(result)["P19"].structured_claim

    assert claim.subject.kind is TermKind.URI
    assert claim.subject.value == f"{WD_ENTITY_PREFIX}Q42"
    assert claim.predicate.kind is TermKind.URI
    assert claim.predicate.value == f"{WDT_PROP_PREFIX}P19"
    assert claim.object.kind is TermKind.URI
    assert claim.object.value == f"{WD_ENTITY_PREFIX}Q350"


async def test_content_is_the_unchanged_prose_rendering() -> None:
    """Labels keep serving ``content`` exactly as at 0.1.0 — the structured
    claim rides alongside rather than replacing the verbalization."""
    result = await _extract()
    contents = {c.content for c in result.candidates}
    assert "Douglas Adams place of birth: Cambridge" in contents
    assert "Douglas Adams date of birth: 11 March 1952" in contents


async def test_a_deprecated_statement_gets_no_triple_because_it_gets_no_particle() -> None:
    """Matching Wikidata's own rule that deprecated rank has no truthy form."""
    result = await _extract()
    assert "P1038" not in _by_property(result)


# ---------------------------------------------------------------------------
# Object terms follow Wikidata's published RDF mapping
# ---------------------------------------------------------------------------


async def test_typed_literals_carry_their_xsd_datatype() -> None:
    by_prop = _by_property(await _extract())

    date = by_prop["P569"].structured_claim.object
    assert date.kind is TermKind.LITERAL
    assert date.value == "1952-03-11T00:00:00Z"
    assert date.datatype == f"{_XSD}dateTime"
    assert date.language is None

    height = by_prop["P2048"].structured_claim.object
    assert height.value == "1.96"
    assert height.datatype == f"{_XSD}decimal"


async def test_monolingual_text_becomes_a_language_tagged_literal() -> None:
    name = _by_property(await _extract())["P1477"].structured_claim.object
    assert name.kind is TermKind.LITERAL
    assert name.value == "Douglas Noël Adams"
    assert name.language == "en"
    assert name.datatype is None


async def test_url_becomes_an_iri_node_and_external_id_a_plain_literal() -> None:
    by_prop = _by_property(await _extract())

    site = by_prop["P856"].structured_claim.object
    assert site.kind is TermKind.URI
    assert site.value == "https://douglasadams.com/"

    viaf = by_prop["P214"].structured_claim.object
    assert viaf.kind is TermKind.LITERAL
    assert viaf.value == "113230702"
    assert viaf.datatype is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+1952-03-11T00:00:00Z", "1952-03-11T00:00:00Z"),
        # Year / month precision writes 00 for the unknown components; Wikidata
        # normalises those to 01 so the literal stays a valid xsd:dateTime.
        ("+1952-00-00T00:00:00Z", "1952-01-01T00:00:00Z"),
        ("+1952-03-00T00:00:00Z", "1952-03-01T00:00:00Z"),
        ("-0044-03-15T00:00:00Z", "-0044-03-15T00:00:00Z"),
        ("", None),
        ("not-a-date", None),
    ],
)
def test_time_literal_normalisation(raw: str, expected: str | None) -> None:
    assert _time_literal(raw) == expected


def test_an_unmapped_datatype_yields_no_term() -> None:
    """The tolerant backstop: a value type with no honest term
    produces no triple, which leaves the particle prose-canonical rather than
    losing the claim."""
    assert _object_term("wikibase-lexeme", "L123") is None
    assert _object_term("time", "not-a-dict") is None


# ---------------------------------------------------------------------------
# Subject binding — the URI rung
# ---------------------------------------------------------------------------


def test_entity_uri_and_external_ref_agree() -> None:
    ref = external_ref_for("Q42")
    assert ref.namespace == "wikidata"
    assert ref.id == "Q42"
    assert ref.uri == entity_uri("Q42") == f"{WD_ENTITY_PREFIX}Q42"


async def test_every_subject_carries_its_external_ref() -> None:
    """Keyed by subject *name* (a Q-id here), which is how the pipeline zips
    refs onto resolved Subjects — and how the URI rung finds its key."""
    result = await _extract()
    edge = _by_property(result)["P19"]
    assert edge.subjects == ["Q42", "Q350"]
    assert set(edge.external_refs) == {"Q42", "Q350"}
    assert edge.external_refs["Q350"].uri == f"{WD_ENTITY_PREFIX}Q350"


async def test_subject_term_binds_to_the_resolved_subject_uuid() -> None:
    """End-to-end: the triple's subject is an IRI while the candidate's names
    are Q-ids, so binding runs through ``candidate_to_particle`` →
    ``bind_subject_id``'s URI rung. Without it ``subject_id`` would always be
    ``None`` for exactly the population with the best keys."""
    from particles.extraction.general import candidate_to_particle

    candidate = _by_property(await _extract())["P19"]
    particle = candidate_to_particle(
        candidate,
        "entry-1",
        "snapshot-1",
        EXTRACTOR_ID,
        subject_ids=["uuid-q42", "uuid-q350"],
    )
    assert particle.structured_claim is not None
    assert particle.structured_claim.subject_id == "uuid-q42"
    # The invariant L-STR-11 checks.
    assert particle.structured_claim.subject_id in particle.subject_ids


async def test_structure_canonical_particle_survives_a_store_round_trip() -> None:
    """`canonical_form`, the triple and the stamp all persist."""
    from particles.extraction.general import candidate_to_particle
    from particles.store.particle_store import ParticleRow

    candidate = _by_property(await _extract())["P19"]
    particle = candidate_to_particle(
        candidate, "entry-1", "snapshot-1", EXTRACTOR_ID, subject_ids=["uuid-q42", "uuid-q350"]
    )
    restored = ParticleRow.from_model(particle).to_model()

    assert restored.canonical_form is CanonicalForm.STRUCTURED
    assert restored.content == particle.content
    assert restored.structured_claim is not None
    assert restored.structured_claim.object.value == f"{WD_ENTITY_PREFIX}Q350"
    assert restored.structured_claim.structurizer_version == EXTRACTOR_VERSION


async def test_two_extractions_of_the_same_bytes_produce_identical_triples() -> None:
    """The triple half is derived from the deposited identifiers alone, so it is
    reproducible even though the prose half consults the label API."""

    def triples(result: dict[str, Any]) -> dict[str, Any]:
        # ``generated_at`` is the one field that legitimately differs: it stamps
        # when the annotation was produced, not what it says.
        return {
            k: v.structured_claim.model_dump(exclude={"generated_at"}) for k, v in result.items()
        }

    assert triples(_by_property(await _extract())) == triples(_by_property(await _extract()))
