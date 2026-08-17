"""Mastodon importer (Engine layer; moved from particles.extraction.mastodon
).

Accepts the three Mastodon URL shapes (local status, federated status, raw
API URL), fetches the status + its context tree (``ancestors`` +
``descendants``) via two unauthenticated public API calls, and stores the
assembled JSON blob as a single ``MASTODON_THREAD`` corpus entry. The
Client-safe URL parsing and fetch helpers stay in
``particles.extraction.mastodon``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.extraction.mastodon import (
    SOURCE_TYPE,
    _fetch_thread,
    _parse_status_route,
    _status_api_url,
)

log = logging.getLogger(__name__)


class MastodonImporter:
    """Fetch a Mastodon status + context via the public API and store as MASTODON_THREAD."""

    # a Mastodon status with a link card carries the external
    # URL in ``status.card.url``. The status text is meta-discussion;
    # the substance lives at the card's target. Statuses without a
    # card (pure text posts) return None and don't trigger a follow.
    DEFAULT_FOLLOW_POST_LINKS: bool = True
    DEFAULT_FOLLOW_COMMENT_LINKS: bool = False

    def accepts_url(self, url: str) -> bool:
        return _parse_status_route(url) is not None

    def primary_url(self, content: bytes) -> str | None:
        """Return the status's link-card URL, or None for text-only posts.

        Mastodon represents external links via the ``card`` field on the
        status (the same field the extractor reads to populate
        ``content:hasUrl``). If no card is present (text-only
        statuses, statuses linking only to other Mastodon statuses) the
        method returns None.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        status = data.get("status")
        if not isinstance(status, dict):
            return None
        card = status.get("card")
        if not isinstance(card, dict):
            return None
        url = card.get("url")
        if not isinstance(url, str) or not url:
            return None
        return url

    def derive_fetch_instance(self, url: str) -> str | None:
        """Return the host the importer would fetch ``url`` from (or ``None``).

        Exposed as a public helper so tests — and operators reasoning
        about federation routing — can confirm the importer fetches from
        the *home* instance for the ``@user@remote`` shape rather than
        the viewing instance.
        """
        parsed = _parse_status_route(url)
        return parsed[0] if parsed else None

    async def deposit(
        self,
        session: AsyncSession,
        url: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot
        from particles.http import particles_client

        parsed = _parse_status_route(url)
        if parsed is None:
            raise ValueError(f"Not a recognised Mastodon URL: {url}")
        home_instance, status_id, _handle_hint = parsed

        # Canonical URL is the home-instance API URL — it is the addressable
        # identity of the bytes we just fetched. Two operators pasting the
        # same status from different viewing instances thereby end up
        # collapsing onto the same corpus entry (provided the home instance
        # serves the same bytes both times — which it always does for a
        # public unedited status).
        canonical = _status_api_url(home_instance, status_id)
        max_replies = get_config().mastodon.max_replies

        async with particles_client() as client:
            payload = await _fetch_thread(client, home_instance, status_id, max_replies)

        content = json.dumps(payload, sort_keys=True).encode("utf-8")
        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)

        # Author + published timestamp for Snapshot stamping.
        status = payload["status"]
        account = status.get("account") if isinstance(status, dict) else None
        acct = ""
        if isinstance(account, dict):
            acct_raw = account.get("acct")
            if isinstance(acct_raw, str):
                acct = acct_raw
        author_id: str | None = f"mastodon:{acct}" if acct else None

        published: datetime | None = None
        created_raw = status.get("created_at") if isinstance(status, dict) else None
        if isinstance(created_raw, str):
            try:
                # Mastodon emits RFC3339 with explicit timezone (``...+00:00``);
                # ``fromisoformat`` handles the form natively on 3.11+.
                published = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            except ValueError:
                log.warning(
                    "Unparseable created_at on Mastodon status %s: %r",
                    status_id,
                    created_raw,
                )

        log.info(
            "Deposited Mastodon thread %s (%d bytes, author=%s, published=%s, "
            "ancestors=%d, descendants=%d)",
            canonical,
            len(content),
            author_id,
            published,
            len(payload["context"]["ancestors"]),
            len(payload["context"]["descendants"]),
        )
        return await write_entry_and_snapshot(
            session=session,
            uri_r=canonical,
            source_type=SOURCE_TYPE,
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
            content=content,
            content_hash=content_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
            author_id=author_id,
            content_published_at=published,
        )
