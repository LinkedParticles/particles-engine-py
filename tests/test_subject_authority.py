"""Tests for the Subject Authority registry.

Covers the Protocol/registry surface, `PatternAuthority` parity with the old
`_NAMESPACE_PATTERNS`, priority arbitration, `uri_for` templates,
domain applicability, and per-authority config. The Wikidata live path and the
end-to-end resolver cascade are exercised in `tests/test_subjects.py`.
"""

from __future__ import annotations

import re
from unittest.mock import patch

from particles.config import AuthorityConfig, ParticlesConfig
from particles.core.schema import ApplicabilityClause
from particles.ingest.authorities import (
    PatternAuthority,
    SubjectAuthority,
    WikidataAuthority,
    get_authorities,
    is_applicable,
)
from particles.ingest.authorities.registry import _make_authorities

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_builtin_authorities_satisfy_protocol() -> None:
    assert isinstance(WikidataAuthority(), SubjectAuthority)
    assert isinstance(
        PatternAuthority(namespace="numista", pattern=re.compile(r"N#(\d+)"), priority=10),
        SubjectAuthority,
    )


# ---------------------------------------------------------------------------
# PatternAuthority parity with the old _NAMESPACE_PATTERNS
# ---------------------------------------------------------------------------


class TestPatternParity:
    """The four migrated recognize-only namespaces match exactly as before."""

    def _authorities_by_ns(self) -> dict[str, SubjectAuthority]:
        return {a.NAMESPACE: a for a in get_authorities()}

    def test_numista(self) -> None:
        ref = self._authorities_by_ns()["numista"].recognize("coin N#123 of France")
        assert ref is not None and ref.namespace == "numista" and ref.id == "123"
        # Recognized ref carries no uri (parity with _detect_namespace_pattern).
        assert ref.uri is None

    def test_km_catalog(self) -> None:
        ref = self._authorities_by_ns()["km_catalog"].recognize("KM#A-12")
        assert ref is not None and ref.namespace == "km_catalog" and ref.id == "A-12"

    def test_isbn(self) -> None:
        ref = self._authorities_by_ns()["isbn"].recognize("ISBN 0306406152")
        assert ref is not None and ref.namespace == "isbn" and ref.id == "0306406152"

    def test_doi(self) -> None:
        ref = self._authorities_by_ns()["doi"].recognize("see DOI: 10.1/abc.def")
        assert ref is not None and ref.namespace == "doi" and ref.id == "10.1/abc.def"

    def test_no_match_returns_none(self) -> None:
        assert self._authorities_by_ns()["numista"].recognize("plain name") is None


class TestWikidataRecognize:
    def test_four_plus_digits_recognized(self) -> None:
        ref = WikidataAuthority().recognize("entity Q123456 here")
        # id is digits-only (the pre-existing recognize/resolve asymmetry).
        assert ref is not None and ref.namespace == "wikidata" and ref.id == "123456"

    def test_short_qid_not_recognized_in_name(self) -> None:
        # The \d{4,} floor: a bare Q42 in prose is not auto-detected (preserved).
        assert WikidataAuthority().recognize("the Q42 question") is None


# ---------------------------------------------------------------------------
# uri_for templates (the IRI-template capability)
# ---------------------------------------------------------------------------


class TestUriFor:
    def test_wikidata_uri(self) -> None:
        assert WikidataAuthority().uri_for("Q42") == "https://www.wikidata.org/wiki/Q42"

    def test_numista_uri_template(self) -> None:
        ns = {a.NAMESPACE: a for a in get_authorities()}
        assert ns["numista"].uri_for("123") == "https://en.numista.com/catalogue/pieces123.html"
        assert ns["doi"].uri_for("10.1/x") == "https://doi.org/10.1/x"

    def test_template_absent_returns_none(self) -> None:
        ns = {a.NAMESPACE: a for a in get_authorities()}
        assert ns["isbn"].uri_for("0306406152") is None
        assert ns["km_catalog"].uri_for("8") is None


# ---------------------------------------------------------------------------
# Registry ordering + priority arbitration
# ---------------------------------------------------------------------------


class TestRegistryOrder:
    def test_default_priority_order(self) -> None:
        assert [a.NAMESPACE for a in get_authorities()] == [
            "numista",
            "km_catalog",
            "wikidata",
            "isbn",
            "doi",
        ]

    def test_lower_priority_int_wins_when_both_recognize(self) -> None:
        # when >1 authority recognizes the same name, the lowest
        # PRIORITY (first in the ordered registry) is chosen.
        pat = re.compile(r"X(\d+)")
        winner = PatternAuthority(namespace="winner", pattern=pat, priority=5)
        loser = PatternAuthority(namespace="loser", pattern=pat, priority=50)
        ordered = sorted([loser, winner], key=lambda a: a.PRIORITY)
        chosen = next(a for a in ordered if a.recognize("X9") is not None)
        assert chosen.NAMESPACE == "winner"


# ---------------------------------------------------------------------------
# Domain applicability
# ---------------------------------------------------------------------------


class TestApplicability:
    def test_broad_authority_always_applies(self) -> None:
        assert is_applicable(WikidataAuthority(), domain="numismatics") is True
        assert is_applicable(WikidataAuthority(), domain=None) is True

    def test_must_clause_gates_to_its_domain(self) -> None:
        auth = PatternAuthority(
            namespace="coins",
            pattern=re.compile(r"C(\d+)"),
            priority=10,
            applicability=[
                ApplicabilityClause(
                    keyword="MUST",
                    domain_uri="http://www.wikidata.org/entity/Q1",
                    domain_label="numismatics",
                    source_types=["NUMISTA_COIN"],
                )
            ],
        )
        assert is_applicable(auth, domain="numismatics") is True
        assert is_applicable(auth, domain="physics") is False
        assert is_applicable(auth, domain=None) is True  # unknown ⇒ applicable

    def test_must_not_excludes(self) -> None:
        auth = PatternAuthority(
            namespace="x",
            pattern=re.compile(r"x"),
            priority=10,
            applicability=[
                ApplicabilityClause(
                    keyword="MUST_NOT",
                    domain_uri="http://www.wikidata.org/entity/Q2",
                    domain_label="physics",
                    source_types=[],
                )
            ],
        )
        assert is_applicable(auth, domain="physics") is False
        assert is_applicable(auth, domain="numismatics") is True


# ---------------------------------------------------------------------------
# Per-authority config
# ---------------------------------------------------------------------------


class TestAuthorityConfig:
    def _build_with_config(self, authorities: dict[str, AuthorityConfig]) -> list[SubjectAuthority]:
        cfg = ParticlesConfig(authorities=authorities)
        with patch("particles.ingest.authorities.registry.get_config", return_value=cfg):
            return _make_authorities()

    def test_disable_removes_authority(self) -> None:
        built = self._build_with_config({"numista": AuthorityConfig(enabled=False)})
        assert "numista" not in [a.NAMESPACE for a in built]

    def test_priority_override_reorders(self) -> None:
        # Push doi (default 50) to the front.
        built = self._build_with_config({"doi": AuthorityConfig(priority=1)})
        assert built[0].NAMESPACE == "doi"

    def test_default_config_is_full_set(self) -> None:
        built = self._build_with_config({})
        assert len(built) == 5
