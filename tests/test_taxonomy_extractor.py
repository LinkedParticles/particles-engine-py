"""TaxonomyExtractor parsing + materialisation tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import ExtractionStatus, Snapshot, WarcRecordType
from particles.extraction.taxonomy import TaxonomyExtractor
from particles.store.taxonomy_store import (
    expand_tags,
    get_taxonomy,
    list_taxonomies,
)


def _snapshot() -> Snapshot:
    return Snapshot(
        snapshot_id="00000000-0000-0000-0000-000000000001",
        captured_at=datetime.now(UTC),
        content_hash="abc",
        extraction_status=ExtractionStatus.PENDING,
        warc_record_type=WarcRecordType.RESPONSE,
    )


def _valid_taxonomy_json() -> bytes:
    return json.dumps(
        {
            "name": "Coins",
            "version": "1.0.0",
            "author": "tester",
            "tags": [
                {"tag": "coins"},
                {"tag": "coins/by-region", "parent": "coins"},
                {"tag": "coins/by-region/germany", "parent": "coins/by-region"},
            ],
        }
    ).encode("utf-8")


class TestAccepts:
    def test_accepts_taxonomy_source_type(self) -> None:
        ex = TaxonomyExtractor()
        assert ex.accepts("TAXONOMY_DEFINITION") is True
        assert ex.accepts("WEB_PAGE") is False


class TestExtract:
    @pytest.mark.asyncio
    async def test_materialises_rows(self, db_session: AsyncSession) -> None:
        ex = TaxonomyExtractor()
        result = await ex.extract(
            _snapshot(),
            _valid_taxonomy_json(),
            session=db_session,
            corpus_entry_id="entry-abc",
        )
        assert result.candidates == []
        listed = await list_taxonomies(db_session)
        assert len(listed) == 1
        td = listed[0]
        assert td.name == "Coins"
        assert td.corpus_entry_id == "entry-abc"
        assert {n.tag for n in td.tags} == {
            "coins",
            "coins/by-region",
            "coins/by-region/germany",
        }

    @pytest.mark.asyncio
    async def test_subtree_walkable_after_extract(self, db_session: AsyncSession) -> None:
        ex = TaxonomyExtractor()
        await ex.extract(
            _snapshot(),
            _valid_taxonomy_json(),
            session=db_session,
            corpus_entry_id=None,
        )
        expanded = await expand_tags(db_session, ["coins"])
        assert "coins/by-region/germany" in expanded

    @pytest.mark.asyncio
    async def test_invalid_json_returns_quality_note(self, db_session: AsyncSession) -> None:
        ex = TaxonomyExtractor()
        result = await ex.extract(
            _snapshot(),
            b"not valid json",
            session=db_session,
            corpus_entry_id=None,
        )
        assert result.candidates == []
        assert any("Invalid" in note for note in result.quality_notes)

    @pytest.mark.asyncio
    async def test_bad_parent_path_returns_quality_note(self, db_session: AsyncSession) -> None:
        ex = TaxonomyExtractor()
        bad = json.dumps(
            {
                "name": "X",
                "version": "1.0",
                "author": "a",
                "tags": [{"tag": "root", "parent": "nope"}],  # invalid: root has parent
            }
        ).encode()
        result = await ex.extract(_snapshot(), bad, session=db_session, corpus_entry_id=None)
        assert result.candidates == []
        assert result.quality_notes

    @pytest.mark.asyncio
    async def test_empty_tags_list_ok(self, db_session: AsyncSession) -> None:
        ex = TaxonomyExtractor()
        empty = json.dumps({"name": "Empty", "version": "0.1", "author": "a", "tags": []}).encode()
        result = await ex.extract(_snapshot(), empty, session=db_session, corpus_entry_id=None)
        assert result.candidates == []
        listed = await list_taxonomies(db_session)
        assert any(t.name == "Empty" for t in listed)


class TestRegistered:
    def test_appears_in_registry_before_general(self) -> None:
        from particles.extraction.registry import get_extractors

        extractors = get_extractors()
        ids = [type(p).__name__ for p in extractors]
        assert "TaxonomyExtractor" in ids
        assert ids.index("TaxonomyExtractor") < ids.index("GeneralExtractor")

    def test_registry_finds_extractor_for_source_type(self) -> None:
        from particles.extraction.registry import get_extractors

        for plugin in get_extractors():
            if plugin.accepts("TAXONOMY_DEFINITION"):
                assert type(plugin).__name__ == "TaxonomyExtractor"
                return
        raise AssertionError("No extractor accepts TAXONOMY_DEFINITION")


class TestGetTaxonomyNotFound:
    @pytest.mark.asyncio
    async def test_returns_none(self, db_session: AsyncSession) -> None:
        result = await get_taxonomy(db_session, "no-such-id")
        assert result is None
