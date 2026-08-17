"""Numista importer (Engine layer; moved from particles.extraction.numista.importer
).

``NumistaImporter`` dispatches on URL pattern: individual coin URLs go through
the Numista API (requires ``NUMISTA_API_KEY``), catalogue listing URLs are
fetched as raw HTML (no API key required) and stored as
``NUMISTA_LISTING_HTML`` for the listing extractor to parse. The Client-safe
URL regexes and constants stay in ``particles.extraction.numista._shared``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from particles.extraction.numista._shared import (
    NUMISTA_API_BASE,
    NUMISTA_COIN_RE,
    NUMISTA_ISSUER_RE,
    SOURCE_TYPE_COIN,
    SOURCE_TYPE_LISTING,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


class NumistaImporter:
    """Imports Numista coin and catalogue-listing URLs."""

    def accepts_url(self, url: str) -> bool:
        return bool(NUMISTA_COIN_RE.search(url) or NUMISTA_ISSUER_RE.search(url))

    async def deposit(
        self,
        session: AsyncSession,
        url: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        m = NUMISTA_COIN_RE.search(url)
        if m:
            coin_id = m.group(1) or m.group(2)
            return await self._deposit_coin(session, url, coin_id, deposited_by, tags)
        return await self._deposit_listing_html(session, url, deposited_by, tags)

    async def _deposit_coin(
        self,
        session: AsyncSession,
        url: str,
        coin_id: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        from particles.http import get_capped, particles_client
        from particles.secrets import get_numista_api_key

        api_key = get_numista_api_key()
        api_url = f"{NUMISTA_API_BASE}/types/{coin_id}"
        async with particles_client() as client:
            resp = await get_capped(client, api_url, headers={"Numista-API-Key": api_key})
        if resp.status_code == 429:
            raise ValueError("Numista API quota exhausted (HTTP 429). Monthly limit reached.")
        if resp.status_code == 404:
            raise ValueError(f"Numista coin type {coin_id} not found (404)")
        resp.raise_for_status()

        content = resp.content
        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot

        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)
        log.info("Deposited Numista coin %s via API (%d bytes)", coin_id, len(content))
        return await write_entry_and_snapshot(
            session=session,
            uri_r=url,
            source_type=SOURCE_TYPE_COIN,
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
            content=content,
            content_hash=content_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
        )

    async def _deposit_listing_html(
        self,
        session: AsyncSession,
        url: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        from particles.http import get_capped, particles_client

        async with particles_client() as client:
            resp = await get_capped(client, url)
        if resp.status_code == 403:
            raise ValueError(
                f"Numista listing page blocked (403) for {url!r}. "
                "The page may require authentication or the URL is not a public listing."
            )
        resp.raise_for_status()

        content = resp.content
        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot

        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)
        log.info("Deposited Numista listing HTML %s (%d bytes)", url, len(content))
        return await write_entry_and_snapshot(
            session=session,
            uri_r=url,
            source_type=SOURCE_TYPE_LISTING,
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
            content=content,
            content_hash=content_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
        )
