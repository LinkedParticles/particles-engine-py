"""Schema validation for TagNode / TaxonomyDefinition."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from particles.core.schema import SourceType, TagNode, TaxonomyDefinition


class TestTagNode:
    def test_root_tag_has_no_parent(self) -> None:
        node = TagNode(tag="coins")
        assert node.parent is None
        assert node.aliases == []

    def test_aliases_default_empty(self) -> None:
        node = TagNode(tag="ml", aliases=[])
        assert node.aliases == []

    def test_tag_required_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            TagNode(tag="")


class TestTaxonomyDefinition:
    def _valid_payload(self) -> dict[str, object]:
        return {
            "name": "Coins",
            "version": "1.0.0",
            "author": "jeff",
            "tags": [
                {"tag": "coins"},
                {"tag": "coins/by-region", "parent": "coins"},
                {"tag": "coins/by-region/germany", "parent": "coins/by-region"},
            ],
        }

    def test_round_trip(self) -> None:
        td = TaxonomyDefinition(**self._valid_payload())  # type: ignore[arg-type]
        assert td.name == "Coins"
        assert td.version == "1.0.0"
        assert len(td.tags) == 3
        assert td.tags[2].tag == "coins/by-region/germany"
        assert td.tags[2].parent == "coins/by-region"

    def test_root_with_explicit_parent_rejected(self) -> None:
        payload = self._valid_payload()
        payload["tags"] = [{"tag": "coins", "parent": "something"}]
        with pytest.raises(ValidationError):
            TaxonomyDefinition(**payload)  # type: ignore[arg-type]

    def test_child_with_wrong_parent_rejected(self) -> None:
        payload = self._valid_payload()
        payload["tags"] = [
            {"tag": "coins"},
            {"tag": "coins/by-region/germany", "parent": "coins"},  # skips middle
        ]
        with pytest.raises(ValidationError):
            TaxonomyDefinition(**payload)  # type: ignore[arg-type]

    def test_taxonomy_id_auto_assigned(self) -> None:
        td = TaxonomyDefinition(**self._valid_payload())  # type: ignore[arg-type]
        assert td.taxonomy_id  # non-empty UUID
        assert len(td.taxonomy_id) == 36

    def test_empty_tags_list_allowed(self) -> None:
        payload = self._valid_payload()
        payload["tags"] = []
        td = TaxonomyDefinition(**payload)  # type: ignore[arg-type]
        assert td.tags == []

    def test_validates_from_json(self) -> None:
        td = TaxonomyDefinition.model_validate_json(
            '{"name":"X","version":"1.0","author":"a","tags":[{"tag":"r"}]}'
        )
        assert td.name == "X"


class TestSourceType:
    def test_taxonomy_definition_enum_member(self) -> None:
        assert SourceType.TAXONOMY_DEFINITION.value == "TAXONOMY_DEFINITION"
