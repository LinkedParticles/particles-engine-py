"""Hacker News importer (Engine layer; moved from particles.extraction.hackernews
).

Accepts ``news.ycombinator.com/item?id=N`` and the raw Firebase API URL
``hacker-news.firebaseio.com/v0/item/N.json``. Walks the comment tree via the
public Firebase API and stores the assembled JSON blob (story + comments) as a
single corpus entry of source type ``HACKERNEWS_THREAD``. The Client-safe
parsing helpers and constants stay in ``particles.extraction.hackernews``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.extraction.hackernews import (
    SOURCE_TYPE,
    _fetch_thread,
    _get_story,
    _parse_item_id,
)

log = logging.getLogger(__name__)


class HackerNewsImporter:
    """Fetch an HN thread via the Firebase API and store it as HACKERNEWS_THREAD."""

    # HN stories with a ``url`` field are link-shaped — the
    # primary URL is the substance, the HN page is title + score +
    # comments. Ask-HN / text-only / Tell-HN have no ``url`` and
    # ``primary_url`` returns None for them.
    DEFAULT_FOLLOW_POST_LINKS: bool = True
    DEFAULT_FOLLOW_COMMENT_LINKS: bool = False

    def accepts_url(self, url: str) -> bool:
        return _parse_item_id(url) is not None

    def primary_url(self, content: bytes) -> str | None:
        """Return the HN story's external URL, or None for Ask-HN / text posts.

        HN's Firebase API gives stories an optional ``url`` field; we
        cache that under ``story.url`` in the importer-shaped JSON
        blob. Ask-HN and Tell-HN (and any text-only submission) omit
        the field entirely.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        story = _get_story(data)
        if not story:
            return None
        url = story.get("url")
        if not isinstance(url, str) or not url:
            return None
        return url

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

        item_id = _parse_item_id(url)
        if item_id is None:
            raise ValueError(f"Not a recognised Hacker News URL: {url}")

        # Canonical URL is the human-facing item URL — operators searching
        # the corpus by URL paste from their browser, not from the API.
        canonical = f"https://news.ycombinator.com/item?id={item_id}"
        max_comments = get_config().hackernews.max_comments

        async with particles_client() as client:
            payload = await _fetch_thread(client, item_id, max_comments)

        content = json.dumps(payload, sort_keys=True).encode("utf-8")
        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)

        # Extract author + published timestamp from the story object for
        # Snapshot stamping.
        story = payload["story"]
        raw_author = story.get("by") or ""
        author_id: str | None = f"hn:{raw_author}" if raw_author else None
        published: datetime | None = None
        ts = story.get("time")
        if isinstance(ts, (int, float)):
            published = datetime.fromtimestamp(float(ts), tz=UTC)

        log.info(
            "Deposited Hacker News thread %s (%d bytes, author=%s, published=%s, comments=%d)",
            canonical,
            len(content),
            author_id,
            published,
            len(payload["comments"]),
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
