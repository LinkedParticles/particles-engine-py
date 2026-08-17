"""Reddit importer (Engine layer; moved from particles.extraction.reddit).

Fetches the Reddit public JSON API ({url}.json?limit=200) and stores the raw
JSON blob. The Client-safe parsing helpers and constants stay in
``particles.extraction.reddit``; this module imports them back across the
Client/Engine boundary (Engine → Client is allowed).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.extraction.reddit import (
    _REDDIT_SHARE_URL_RE,
    _REDDIT_URL_RE,
    SOURCE_TYPE,
    _fetch_with_curl,
    _get_post,
    _resolve_reddit_redirect,
)

log = logging.getLogger(__name__)


class RedditImporter:
    """Fetch a Reddit thread via the public JSON API and store it as REDDIT_POST."""

    # this importer participates in deposit-time follow of the
    # post's primary URL. Reddit link-posts are the canonical case —
    # the post envelope is meta-discussion; the substance lives at the
    # linked URL.
    DEFAULT_FOLLOW_POST_LINKS: bool = True
    DEFAULT_FOLLOW_COMMENT_LINKS: bool = False

    def accepts_url(self, url: str) -> bool:
        return bool(_REDDIT_URL_RE.match(url) or _REDDIT_SHARE_URL_RE.match(url))

    def primary_url(self, content: bytes) -> str | None:
        """Return the link-post's external URL, or None for self-posts.

        Reddit's link posts carry the external URL in
        ``data.children[0].data.url``. Self-posts (``is_self=True``)
        and posts whose ``url`` field points back at the Reddit thread
        itself have no primary URL to follow — return None so the
        deposit machinery skips the follow.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        post = _get_post(data)
        if not post:
            return None
        if post.get("is_self"):
            return None
        url = post.get("url")
        if not isinstance(url, str) or not url:
            return None
        # Reddit's URL field defaults to the thread's own permalink for
        # self-posts that lack a link target — guard against that even
        # when is_self isn't explicitly True.
        if "reddit.com/r/" in url and "/comments/" in url:
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

        # A `/s/` share link carries no post id — resolve its redirect to the
        # canonical `/comments/` permalink (validated against the reddit-host
        # allowlist) before building the `.json` API URL.
        if _REDDIT_SHARE_URL_RE.match(url):
            url = await _resolve_reddit_redirect(url)

        # Normalise to www.reddit.com canonical form
        canonical = re.sub(r"^https?://(?:www\.|old\.)?reddit\.com", "https://www.reddit.com", url)
        if not canonical.endswith("/"):
            canonical = canonical + "/"
        api_url = canonical + ".json?limit=200"

        # Reddit's CDN (Cloudflare) blocks Python's TLS fingerprint but allows curl.
        # Use subprocess curl so the OS TLS stack is used instead.
        content = await _fetch_with_curl(api_url)
        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)

        # Extract post author and publication timestamp
        author_id: str | None = None
        content_published_at: datetime | None = None
        try:
            data = json.loads(content)
            post = _get_post(data)
            if post:
                raw_author = post.get("author", "")
                if raw_author and raw_author not in ("[deleted]", "[removed]"):
                    author_id = f"reddit:u/{raw_author}"
                created_utc = post.get("created_utc")
                if created_utc is not None:
                    content_published_at = datetime.fromtimestamp(float(created_utc), tz=UTC)
        except Exception:
            pass

        log.info(
            "Deposited Reddit thread %s (%d bytes, author=%s, published=%s)",
            canonical,
            len(content),
            author_id,
            content_published_at,
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
            content_published_at=content_published_at,
        )
