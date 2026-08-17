"""Wikidata importer (Engine layer; moved from particles.extraction.wikidata
).

Imports Wikidata entity URLs by fetching the Wikibase REST API.
The Client-safe URL regex, REST base, and source-type constant stay in
``particles.extraction.wikidata`` (alongside the extractor); this module
imports them back across the Client/Engine boundary.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from particles.extraction.wikidata import (
    _WIKIDATA_REST,
    _WIKIDATA_URL_RE,
    SOURCE_TYPE,
)

log = logging.getLogger(__name__)


class WikidataImporter:
    """Imports Wikidata entity URLs by fetching the Wikibase REST API."""

    def accepts_url(self, url: str) -> bool:
        return bool(_WIKIDATA_URL_RE.match(url))

    async def deposit(
        self,
        session: AsyncSession,
        url: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        m = _WIKIDATA_URL_RE.match(url)
        assert m is not None
        qid = m.group(1)
        api_url = f"{_WIKIDATA_REST}/entities/items/{qid}"

        from particles.http import get_capped, particles_client

        async with particles_client() as client:
            resp = await get_capped(client, api_url)
        if resp.status_code == 404:
            raise ValueError(f"Wikidata entity {qid} not found (404)")
        resp.raise_for_status()

        content = resp.content
        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot

        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)
        log.info("Deposited Wikidata entity %s via REST API (%d bytes)", qid, len(content))
        return await write_entry_and_snapshot(
            session=session,
            uri_r=url,
            source_type=SOURCE_TYPE,
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
            content=content,
            content_hash=content_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
        )
