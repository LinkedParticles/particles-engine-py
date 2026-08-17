"""The published JSON-LD context as the single CURIE-prefix authority.

Three surfaces used to answer "is this prefix published?" from three hand-lists
that all disagreed with `artifacts/schemas/context.jsonld`. These tests pin the
reader and the two drift guards that keep the artifact, the registry, and the code from separating again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from particles.core.jsonld_context import is_published_curie, published_prefixes
from tests._upstream import upstream_only

_REPO = Path(__file__).parents[1]
_ADR_0072 = _REPO / "docs" / "ADR" / "active" / "0072-particle-properties-convention.md"


class TestPublishedPrefixes:
    def test_reads_prefixes_from_the_artifact(self) -> None:
        ctx = json.loads((_REPO / "artifacts" / "schemas" / "context.jsonld").read_text())
        expected = {
            f"{term}:"
            for term, value in ctx["@context"].items()
            if isinstance(value, str) and value[-1] in "#/:?[]@"
        }
        assert set(published_prefixes()) == expected

    def test_returns_a_sorted_tuple(self) -> None:
        prefixes = published_prefixes()
        assert isinstance(prefixes, tuple)
        assert list(prefixes) == sorted(prefixes)

    def test_class_and_value_terms_are_not_prefixes(self) -> None:
        # "Particle": "particles:Particle" and "ALEATORY": "psum:ALEATORY" are
        # plain strings but do not end in a gen-delim, so they map no namespace.
        assert "Particle:" not in published_prefixes()
        assert "ALEATORY:" not in published_prefixes()

    @pytest.mark.parametrize(
        "value,published",
        [
            ("skos:definition", True),
            ("geo:lat", True),
            ("wdt:P31", True),
            ("numista:id", True),
            ("content:hasUrl", False),  # deliberately unpublishable — see below
            ("was struck at", False),
            ("http://nomisma.org/id/rome", False),  # an absolute IRI is not a CURIE
        ],
    )
    def test_is_published_curie(self, value: str, published: bool) -> None:
        assert is_published_curie(value) is published

    def test_missing_artifact_publishes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A fork shipping no artifacts gets an empty set, so every CURIE degrades
        # to TOKEN — the honest reading, not a silent fallback
        # to some hand-list that would reintroduce the drift.
        published_prefixes.cache_clear()
        monkeypatch.setattr(
            "particles.core.jsonld_context.schemas_dir", lambda: Path("/nonexistent")
        )
        try:
            assert published_prefixes() == ()
            assert is_published_curie("nmo:hasIssuer") is False
        finally:
            published_prefixes.cache_clear()


@upstream_only  # reads the decision record that holds the registry table
class TestRegistryContextSync:
    """§5's registry and the published context must not separate again."""

    @staticmethod
    def _registered() -> set[str]:
        body = _ADR_0072.read_text()
        table = body.split("### 5. Prefix Registry (normative)")[1].split("\n\n> ")[0]
        return {
            m.group(1) for line in table.splitlines() if (m := re.match(r"\|\s*`([a-z]+:)`", line))
        }

    def test_registry_is_not_empty(self) -> None:
        # Guards the parser itself: a table-format change must fail loudly here
        # rather than silently making the next assertion vacuous.
        assert len(self._registered()) >= 10

    def test_every_registered_prefix_is_published_except_content(self) -> None:
        # `content:` cannot be published: the term `content` is already bound to
        # `particles:content`, and a JSON-LD term has exactly one definition.
        unpublished = self._registered() - set(published_prefixes())
        assert unpublished == {"content:"}
