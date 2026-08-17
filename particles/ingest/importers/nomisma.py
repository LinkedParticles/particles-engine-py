"""Nomisma importer (Engine layer; moved from particles.extraction.nomisma
).

Imports Nomisma entity URIs by fetching the ``.jsonld`` document URI. The Client-safe URL regex and constants stay in
``particles.extraction.nomisma``.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from particles.extraction.nomisma import _NOMISMA_URL_RE, SOURCE_TYPE

log = logging.getLogger(__name__)


class NomismaImporter:
    """Imports Nomisma entity URIs by fetching the .jsonld document URI."""

    def accepts_url(self, url: str) -> bool:
        return bool(_NOMISMA_URL_RE.match(url))

    async def deposit(
        self,
        session: AsyncSession,
        url: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        m = _NOMISMA_URL_RE.match(url)
        assert m is not None
        concept_id = m.group(1)
        # Canonical concept IRI (what we store as uri_r)
        canonical_uri = f"http://nomisma.org/id/{concept_id}"
        # Nomisma publishes a stable JSON-LD document URI distinct from the
        # concept IRI. Fetch over HTTPS (nomisma.org serves it) so the
        # document is not retrievable in cleartext — the one MITM-reachable
        # leg flagged by security finding F14. The concept IRI stored as
        # ``uri_r`` keeps its canonical ``http://`` form (it is an identifier,
        # never dereferenced here).
        jsonld_uri = f"https://nomisma.org/id/{concept_id}.jsonld"

        from particles.http import get_capped, particles_client

        async with particles_client() as client:
            resp = await get_capped(client, jsonld_uri, follow_redirects=True)
        if resp.status_code == 404:
            raise ValueError(f"Nomisma concept {concept_id!r} not found (404)")
        resp.raise_for_status()

        content = resp.content
        if not content:
            raise ValueError(
                f"Nomisma returned empty body for {concept_id!r} "
                f"(status {resp.status_code}, URL {resp.url})"
            )

        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot

        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)
        log.info("Deposited Nomisma concept %s via JSON-LD (%d bytes)", concept_id, len(content))
        return await write_entry_and_snapshot(
            session=session,
            uri_r=canonical_uri,
            source_type=SOURCE_TYPE,
            mutability=Mutability.STABLE,
            fetch_policy=FetchPolicy.LAZY,
            content=content,
            content_hash=content_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
        )
