"""Tests for particles/core/claims.py — the normalizer + matcher."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from particles.core.claims import (
    ClaimFilters,
    ClaimMatch,
    match_claim,
    normalize_term,
    parse_bound,
    predicate_vocabulary,
)
from particles.core.schema import ClaimTerm, StructuredClaim, TermKind


def _literal(value: str, datatype: str | None = None) -> ClaimTerm:
    return ClaimTerm(kind=TermKind.LITERAL, value=value, datatype=datatype)


def _claim(
    predicate: str = "nmo:hasWeight",
    obj: ClaimTerm | None = None,
    predicate_kind: TermKind = TermKind.URI,
) -> StructuredClaim:
    return StructuredClaim(
        subject=ClaimTerm(kind=TermKind.TOKEN, value="1 Pfennig (1948-1950)"),
        predicate=ClaimTerm(kind=predicate_kind, value=predicate),
        object=obj if obj is not None else _literal("3.9", "xsd:decimal"),
        structurizer_id="test-structurizer",
        structurizer_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# normalize_term
# ---------------------------------------------------------------------------


class TestNormalizeTerm:
    def test_curie_decimal(self) -> None:
        assert normalize_term(_literal("3.9", "xsd:decimal")) == Decimal("3.9")

    def test_full_iri_decimal(self) -> None:
        # The Wikidata extractor stamps the expanded IRI spelling.
        term = _literal("3.9", "http://www.w3.org/2001/XMLSchema#decimal")
        assert normalize_term(term) == Decimal("3.9")

    @pytest.mark.parametrize("local", ["integer", "float", "double"])
    def test_other_numeric_datatypes(self, local: str) -> None:
        assert normalize_term(_literal("42", f"xsd:{local}")) == Decimal("42")

    def test_date(self) -> None:
        assert normalize_term(_literal("1948-06-20", "xsd:date")) == datetime(
            1948, 6, 20, tzinfo=UTC
        )

    def test_datetime_aware_preserved(self) -> None:
        term = _literal("1948-06-20T12:00:00+02:00", "xsd:dateTime")
        value = normalize_term(term)
        assert isinstance(value, datetime)
        assert value.utcoffset() is not None

    def test_untyped_literal_is_lexical(self) -> None:
        assert normalize_term(_literal("3 grams")) == "3 grams"

    def test_unparseable_typed_literal_falls_back_to_lexical(self) -> None:
        # "3 grams" stamped xsd:decimal does not parse — the honest fallback
        # is the lexical string (which gt/lt then disclose), never a guess.
        assert normalize_term(_literal("3 grams", "xsd:decimal")) == "3 grams"

    def test_nan_and_infinity_are_not_comparable(self) -> None:
        assert normalize_term(_literal("NaN", "xsd:double")) == "NaN"
        assert normalize_term(_literal("Infinity", "xsd:double")) == "Infinity"

    def test_non_xsd_datatype_is_lexical(self) -> None:
        term = _literal("Q1234", "http://wikiba.se/ontology#WikibaseItem")
        assert normalize_term(term) == "Q1234"

    def test_uri_term_is_lexical(self) -> None:
        term = ClaimTerm(kind=TermKind.URI, value="nm:berlin")
        assert normalize_term(term) == "nm:berlin"

    @given(st.decimals(allow_nan=False, allow_infinity=False, places=6))
    def test_decimal_round_trip(self, value: Decimal) -> None:
        assert normalize_term(_literal(str(value), "xsd:decimal")) == value

    @given(st.datetimes(timezones=st.just(UTC)))
    def test_datetime_round_trip(self, value: datetime) -> None:
        assert normalize_term(_literal(value.isoformat(), "xsd:dateTime")) == value

    @given(st.text(min_size=1))
    def test_untyped_text_always_lexical(self, value: str) -> None:
        assert normalize_term(_literal(value)) == value


# ---------------------------------------------------------------------------
# parse_bound
# ---------------------------------------------------------------------------


class TestParseBound:
    def test_number(self) -> None:
        assert parse_bound("3") == Decimal("3")

    def test_date(self) -> None:
        assert parse_bound("1950-01-01") == datetime(1950, 1, 1, tzinfo=UTC)

    def test_junk_is_none(self) -> None:
        assert parse_bound("three grams") is None

    def test_nan_is_none(self) -> None:
        assert parse_bound("NaN") is None


# ---------------------------------------------------------------------------
# match_claim
# ---------------------------------------------------------------------------


class TestMatchClaim:
    def test_predicate_exact_case_insensitive(self) -> None:
        filters = ClaimFilters(predicate="NMO:hasweight")
        assert match_claim(_claim(), filters) is ClaimMatch.MATCHED

    def test_predicate_curie_and_iri_are_different_strings(self) -> None:
        # §2.2: no prefix expansion — the expanded IRI does not match the CURIE.
        filters = ClaimFilters(predicate="http://nomisma.org/ontology#hasWeight")
        assert match_claim(_claim(predicate="nmo:hasWeight"), filters) is ClaimMatch.UNMATCHED

    def test_object_eq_numeric_across_lexical_forms(self) -> None:
        # 3.90 == 3.9 when both sides normalize numerically.
        filters = ClaimFilters(object_eq="3.90")
        assert match_claim(_claim(), filters) is ClaimMatch.MATCHED

    def test_object_eq_lexical_fallback_case_insensitive(self) -> None:
        filters = ClaimFilters(object_eq="COPPER")
        claim = _claim(obj=_literal("copper"))
        assert match_claim(claim, filters) is ClaimMatch.MATCHED

    def test_object_contains(self) -> None:
        filters = ClaimFilters(object_contains="gram")
        claim = _claim(obj=_literal("3 Grams"))
        assert match_claim(claim, filters) is ClaimMatch.MATCHED

    def test_object_gt_matched_and_unmatched(self) -> None:
        assert match_claim(_claim(), ClaimFilters(object_gt="3")) is ClaimMatch.MATCHED
        assert match_claim(_claim(), ClaimFilters(object_gt="4")) is ClaimMatch.UNMATCHED

    def test_object_lt_dates(self) -> None:
        claim = _claim(obj=_literal("1948-06-20", "xsd:date"))
        assert match_claim(claim, ClaimFilters(object_lt="1950-01-01")) is ClaimMatch.MATCHED
        assert match_claim(claim, ClaimFilters(object_lt="1940-01-01")) is ClaimMatch.UNMATCHED

    def test_gt_non_normalizable_object_is_disclosed_not_dropped(self) -> None:
        claim = _claim(obj=_literal("3 grams"))
        assert match_claim(claim, ClaimFilters(object_gt="3")) is ClaimMatch.NOT_COMPARABLE

    def test_gt_type_mismatch_is_not_comparable(self) -> None:
        # A date object against a numeric bound: normalized, but not to a type
        # comparable with the bound.
        claim = _claim(obj=_literal("1948-06-20", "xsd:date"))
        assert match_claim(claim, ClaimFilters(object_gt="3")) is ClaimMatch.NOT_COMPARABLE

    def test_predicate_mismatch_wins_over_not_comparable(self) -> None:
        # A claim outside the predicate slice is UNMATCHED, not a disclosure row.
        claim = _claim(predicate="nmo:hasMaterial", obj=_literal("copper"))
        filters = ClaimFilters(predicate="nmo:hasWeight", object_gt="3")
        assert match_claim(claim, filters) is ClaimMatch.UNMATCHED

    def test_filters_intersect(self) -> None:
        filters = ClaimFilters(predicate="nmo:hasWeight", object_gt="3", object_lt="4")
        assert match_claim(_claim(), filters) is ClaimMatch.MATCHED
        assert (
            match_claim(_claim(), ClaimFilters(predicate="nmo:hasWeight", object_gt="3.95"))
            is ClaimMatch.UNMATCHED
        )

    def test_unparseable_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="neither a number nor an ISO-8601"):
            match_claim(_claim(), ClaimFilters(object_gt="heavy"))

    def test_empty_filters_falsy(self) -> None:
        assert not ClaimFilters()
        assert ClaimFilters(predicate="x")

    @given(st.decimals(allow_nan=False, allow_infinity=False, places=4))
    def test_gt_lt_partition(self, bound: Decimal) -> None:
        # Against a fixed numeric object, exactly one of gt / lt / eq holds.
        claim = _claim(obj=_literal("3.9", "xsd:decimal"))
        outcomes = [
            match_claim(claim, ClaimFilters(object_gt=str(bound))) is ClaimMatch.MATCHED,
            match_claim(claim, ClaimFilters(object_lt=str(bound))) is ClaimMatch.MATCHED,
            match_claim(claim, ClaimFilters(object_eq=str(bound))) is ClaimMatch.MATCHED,
        ]
        assert outcomes.count(True) == 1


# ---------------------------------------------------------------------------
# predicate_vocabulary
# ---------------------------------------------------------------------------


class TestPredicateVocabulary:
    def test_counts_and_order(self) -> None:
        claims = [
            _claim(predicate="nmo:hasWeight"),
            _claim(predicate="nmo:hasWeight"),
            _claim(predicate="was struck at", predicate_kind=TermKind.TOKEN),
        ]
        vocab = predicate_vocabulary(claims)
        assert vocab == [
            ("nmo:hasWeight", "URI", 2),
            ("was struck at", "TOKEN", 1),
        ]

    def test_distinct_as_stored(self) -> None:
        # Case variants are distinct terms in the listing — the vocabulary is
        # shown as it is, never consolidated at the query layer.
        claims = [_claim(predicate="nmo:hasWeight"), _claim(predicate="nmo:hasweight")]
        assert len(predicate_vocabulary(claims)) == 2
