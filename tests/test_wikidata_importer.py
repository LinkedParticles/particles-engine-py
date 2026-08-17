"""Tests for the Wikidata importer (particles/ingest/importers/wikidata.py).

Covers what tests/AGENTS.md calls for on an importer: the URL patterns
``accepts_url`` claims, the registry routing that follows from them, and the
error paths of ``deposit`` — the REST URL it derives from the QID, the 404
translation, and a non-404 HTTP failure. One happy-path deposit pins the
corpus-entry classification (source type / mutability / fetch policy) that the
lazy re-fetch ladder later reads.

``deposit`` imports ``particles_client`` inside the function body, so the patch
targets the source module (tests/AGENTS.md § Mocking strategy).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import FetchPolicy, Mutability
from particles.extraction.wikidata import SOURCE_TYPE
from particles.ingest.importers.wikidata import WikidataImporter
from tests._capped_http import set_capped_responses

_ENTITY_BODY = json.dumps({"id": "Q42", "labels": {"en": "Douglas Adams"}}).encode()


def _response(status_code: int = 200, content: bytes = _ENTITY_BODY) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.headers = {"content-type": "application/json"}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# accepts_url — the routing predicate
# ---------------------------------------------------------------------------


class TestAcceptsUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.wikidata.org/wiki/Q42",
            "https://www.wikidata.org/entity/Q42",
            "https://www.wikidata.org/wiki/Q1",
            "https://www.wikidata.org/wiki/Q42#sitelinks",
        ],
    )
    def test_entity_urls_are_accepted(self, url: str) -> None:
        assert WikidataImporter().accepts_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.wikidata.org/wiki/Property:P31",  # properties, not items
            "https://www.wikidata.org/wiki/Special:Search",
            "https://wikidata.org/wiki/Q42",  # bare domain — not the canonical host
            "http://www.wikidata.org/wiki/Q42",  # plaintext scheme
            "https://en.wikipedia.org/wiki/Q42",
            "https://example.com/Q42",
        ],
    )
    def test_other_urls_are_declined(self, url: str) -> None:
        assert WikidataImporter().accepts_url(url) is False

    def test_registry_routes_an_entity_url_here(self) -> None:
        from particles.ingest.importers.registry import get_importers

        url = "https://www.wikidata.org/wiki/Q42"
        accepting = [i for i in get_importers() if i.accepts_url(url)]
        assert [type(i).__name__ for i in accepting] == ["WikidataImporter"]


# ---------------------------------------------------------------------------
# deposit — URL derivation and error paths
# ---------------------------------------------------------------------------


class TestDepositErrors:
    @pytest.mark.asyncio
    async def test_missing_entity_is_translated_to_a_value_error(
        self, db_session: AsyncSession
    ) -> None:
        client = MagicMock()
        set_capped_responses(client, return_value=_response(status_code=404, content=b""))
        with patch("particles.http.particles_client") as ctx:
            ctx.return_value.__aenter__.return_value = client
            ctx.return_value.__aexit__.return_value = False
            with pytest.raises(ValueError, match="Wikidata entity Q42 not found"):
                await WikidataImporter().deposit(
                    db_session, "https://www.wikidata.org/wiki/Q42", "tester", []
                )

    @pytest.mark.asyncio
    async def test_other_http_errors_propagate(self, db_session: AsyncSession) -> None:
        resp = _response(status_code=500, content=b"")
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        client = MagicMock()
        set_capped_responses(client, return_value=resp)
        with patch("particles.http.particles_client") as ctx:
            ctx.return_value.__aenter__.return_value = client
            ctx.return_value.__aexit__.return_value = False
            with pytest.raises(httpx.HTTPStatusError):
                await WikidataImporter().deposit(
                    db_session, "https://www.wikidata.org/wiki/Q42", "tester", []
                )


class TestDepositSuccess:
    @pytest.mark.asyncio
    async def test_fetches_the_rest_endpoint_and_classifies_the_entry(
        self, db_session: AsyncSession
    ) -> None:
        client = MagicMock()
        set_capped_responses(client, return_value=_response())
        with patch("particles.http.particles_client") as ctx:
            ctx.return_value.__aenter__.return_value = client
            ctx.return_value.__aexit__.return_value = False
            entry_id, snapshot_id = await WikidataImporter().deposit(
                db_session,
                "https://www.wikidata.org/entity/Q42",
                "tester",
                ["people"],
            )

        # The QID drives the Wikibase REST URL — never the page URL itself.
        assert client.stream.call_args.args[1] == (
            "https://www.wikidata.org/w/rest.php/wikibase/v1/entities/items/Q42"
        )

        from particles.corpus.store import get_entry

        entry = await get_entry(db_session, entry_id)
        assert entry is not None
        assert snapshot_id
        assert entry.uri_r == "https://www.wikidata.org/entity/Q42"
        assert entry.source_type == SOURCE_TYPE
        # An entity page changes under its own URL, and the API body is cheap to
        # re-fetch — MUTABLE + LAZY is what the ladder expects.
        assert entry.mutability == Mutability.MUTABLE
        assert entry.fetch_policy == FetchPolicy.LAZY
        assert entry.tags == ["people"]
