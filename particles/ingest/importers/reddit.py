# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reddit importer (Engine layer; moved from particles.extraction.reddit).

Fetches a thread's post and comment tree and stores them as one JSON blob in
Reddit's ``{permalink}.json`` shape (a post Listing, then a comment Listing),
whichever backend served the bytes. ``reddit.fetch_backend`` picks the backend:

* ``arctic_shift`` (the default) reads the Arctic Shift archive, a free
  third-party mirror of Reddit that serves Reddit's own field names, through
  ``particles_client()``. Reddit answers its own ``.json`` endpoint
  with a 403 bot-wall for unauthenticated clients and no longer grants API
  credentials to scripts. The importer appends a provenance record
  naming the archive host and its capture times as a third blob element.
* ``reddit`` fetches ``{permalink}.json`` from Reddit with curl, for an operator
  whose network can still reach it.

A ``/s/`` share link is resolved through reddit.com first in both cases, and the
corpus entry's URI-R is always the canonical reddit.com permalink, so a thread
deduplicates to one entry whichever backend fetched it.

The Client-safe parsing helpers and constants stay in
``particles.extraction.reddit``; this module imports them back across the
Client/Engine boundary (Engine → Client is allowed).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.extraction.reddit import (
    _REDDIT_SHARE_URL_RE,
    _REDDIT_URL_RE,
    ARCTIC_SHIFT_BACKEND,
    SOURCE_TYPE,
    _assemble_reddit_listing,
    _count_unsettled_below_floor,
    _fetch_with_curl,
    _get_post,
    _is_score_settled,
    _iter_comments,
    _resolve_reddit_redirect,
)
from particles.http import ResponseTooLarge, SourceFetchError, get_capped, particles_client

log = logging.getLogger(__name__)

# Reddit post ids are short base-36 strings. Checked before the id is sent to
# the archive, so a malformed permalink fails here rather than as an odd query.
_POST_ID_RE = re.compile(r"[0-9a-z]{1,16}", re.IGNORECASE)

# The rate-limit wait. A module name rather than ``asyncio.sleep`` at the call
# site so a test can stub this one wait without patching asyncio for the whole
# process (tests/AGENTS.md § Mocking strategy).
_sleep = asyncio.sleep

# Wait used on a 429 that carries no usable ``X-RateLimit-Reset`` header,
# before the ``reddit.arctic_shift_max_wait_seconds`` cap is applied.
_DEFAULT_RATE_LIMIT_WAIT_SECONDS = 5.0


class RedditImporter:
    """Fetch a Reddit thread (Arctic Shift archive by default) and store it as REDDIT_POST."""

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

    async def refetch(self, uri_r: str) -> bytes:
        """Fetch the thread again for ``particles corpus refresh``.

        The URI-R is the reddit.com permalink, an HTML page, while the stored
        blob is the thread JSON from the configured backend, so a refresh has
        to fetch the way ``deposit`` does. Archive scores settle about 36 hours
        after posting, and this is how a deposit made before then picks them up.
        """
        return await _fetch_thread(await _canonical_permalink(uri_r))

    async def deposit(
        self,
        session: AsyncSession,
        url: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot

        canonical = await _canonical_permalink(url)
        content = await _fetch_thread(canonical)
        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)

        # Extract post author and publication timestamp
        author_id: str | None = None
        content_published_at: datetime | None = None
        data: Any = None
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

        # The archive reports a score of 0 or 1 until its second pass, about
        # 36 hours after a comment is made, and the extractor drops comments
        # below ``reddit.min_comment_score``. This warns rather than lifting the
        # floor for the deposit. With the floor lifted, comments would be
        # ranked by a placeholder score, so ``top_comment_count`` would cut an
        # arbitrary sample of a busy thread and the LLM would be paid to read
        # it. A filtered deposit is cheap and says what it left out, and a
        # refresh (or re-deposit) after the archive settles appends a snapshot
        # with real scores to the same entry, where carry-forward skips the unchanged
        # post body. The extractor repeats this as a REDDIT_SCORES_UNSETTLED
        # quality note, since a deposit-time log line is easy to miss.
        dropped, total = _count_unsettled_below_floor(data)
        if dropped:
            log.warning(
                "Reddit thread %s: %d of %d comments have scores the Arctic Shift "
                "archive has not settled yet (they read 0 or 1 until its second pass, "
                "about 36 hours after posting) and fall below reddit.min_comment_score "
                "%d, so extraction will leave them out. Once the scores settle, run "
                "`particles corpus refresh` on the entry to include them.",
                canonical,
                dropped,
                total,
                get_config().reddit.min_comment_score,
            )

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


async def _canonical_permalink(url: str) -> str:
    """The ``https://www.reddit.com/.../`` permalink a deposit stores as its URI-R."""
    # A `/s/` share link carries no post id — resolve its redirect to the
    # canonical `/comments/` permalink (validated against the reddit-host
    # allowlist) before fetching the thread.
    if _REDDIT_SHARE_URL_RE.match(url):
        url = await _resolve_reddit_redirect(url)

    # Normalise to www.reddit.com canonical form
    canonical = re.sub(r"^https?://(?:www\.|old\.)?reddit\.com", "https://www.reddit.com", url)
    if not canonical.endswith("/"):
        canonical = canonical + "/"
    return canonical


async def _fetch_thread(canonical: str) -> bytes:
    """Fetch the thread at a canonical permalink from the configured backend."""
    if get_config().reddit.fetch_backend == ARCTIC_SHIFT_BACKEND:
        match = _REDDIT_URL_RE.match(canonical)
        post_id = match.group("id") if match else ""
        if not _POST_ID_RE.fullmatch(post_id):
            raise SourceFetchError(
                f"Could not read a Reddit post id from {canonical}", url=canonical
            )
        return await _fetch_from_arctic_shift(post_id, canonical)
    # Reddit's CDN (Cloudflare) blocks Python's TLS fingerprint but allows curl.
    # Use subprocess curl so the OS TLS stack is used instead.
    return await _fetch_with_curl(canonical + ".json?limit=200")


async def _fetch_from_arctic_shift(post_id: str, canonical: str) -> bytes:
    """Fetch a post and its comment tree from Arctic Shift as a Reddit-shaped blob.

    Two requests: ``/api/posts/ids`` for the post and ``/api/comments/tree`` for
    every comment, nested the way Reddit nests them. Keys are sorted on
    serialisation so an unchanged archive record re-deposits to the same bytes.

    Raises:
        SourceFetchError: the archive does not hold the post, refused or failed
            the request, or answered with something other than its JSON envelope.
    """
    cfg = get_config().reddit
    base_url = cfg.arctic_shift_base_url.rstrip("/")
    log.info("Fetching Reddit post %s from the Arctic Shift archive (%s)", post_id, base_url)
    async with particles_client(extra_headers={"Accept": "application/json"}) as client:
        posts = await _arctic_shift_get(client, f"{base_url}/api/posts/ids", {"ids": post_id})
        if not posts:
            raise SourceFetchError(
                f"The Arctic Shift archive has not archived Reddit post {post_id} "
                f"({canonical}), so there is nothing to deposit. A post made in the "
                "last few minutes may simply not be captured yet.",
                url=canonical,
                status_code=404,
            )
        post = posts[0]
        if not isinstance(post, dict):
            raise SourceFetchError(
                f"The Arctic Shift archive returned a malformed record for Reddit post {post_id}",
                url=canonical,
            )
        nodes = await _arctic_shift_get(
            client,
            f"{base_url}/api/comments/tree",
            {"link_id": post_id, "limit": str(cfg.arctic_shift_comment_limit)},
        )
    listing = _assemble_reddit_listing(post, nodes, {})
    listing[2]["data"] = _arctic_shift_provenance(base_url, post_id, post, listing)
    return json.dumps(listing, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _arctic_shift_provenance(
    base_url: str, post_id: str, post: dict[str, Any], listing: list[Any]
) -> dict[str, Any]:
    """The record saying these bytes came from the archive, and when it captured them.

    Only archive-side values go in, never this fetch's own clock, so an
    unchanged archive record still re-deposits to identical bytes (the corpus
    deduplicates on the content hash).
    """
    meta = post.get("_meta")
    captured: list[float] = []
    unsettled = 0
    count = 0
    for c in _iter_comments(listing):
        count += 1
        retrieved_on = c.get("retrieved_on")
        if isinstance(retrieved_on, int | float):
            captured.append(retrieved_on)
        if not _is_score_settled(c):
            unsettled += 1
    return {
        "backend": ARCTIC_SHIFT_BACKEND,
        "archive": "Arctic Shift",
        "archive_host": urlsplit(base_url).hostname,
        "post_id": post_id,
        "post_retrieved_on": post.get("retrieved_on"),
        "post_retrieved_2nd_on": (meta.get("retrieved_2nd_on") if isinstance(meta, dict) else None),
        "post_score_settled": _is_score_settled(post),
        "comment_count": count,
        "comments_retrieved_on_earliest": min(captured) if captured else None,
        "comments_retrieved_on_latest": max(captured) if captured else None,
        "comments_score_unsettled": unsettled,
    }


async def _arctic_shift_get(
    client: httpx.AsyncClient, url: str, params: dict[str, str]
) -> list[Any]:
    """GET one archive endpoint and return its ``data`` list.

    A 429 gets one retry after the ``X-RateLimit-Reset`` wait, when that wait is
    within ``reddit.arctic_shift_max_wait_seconds``. The archive reports a
    failure it could answer (a timeout under load, a bad parameter) as HTTP 200
    with ``{"data": null, "error": "..."}``; that body is a fetch failure too.

    Raises:
        SourceFetchError: for every failure above, and for a network error, an
            oversized body, or a body that is not the archive's JSON envelope.
    """
    resp = await _arctic_shift_request(client, url, params)
    if resp.status_code == 429:
        max_wait = get_config().reddit.arctic_shift_max_wait_seconds
        wait = _rate_limit_wait(resp, max_wait)
        if wait > max_wait:
            raise SourceFetchError(
                f"The Arctic Shift archive is rate-limiting requests and resets in "
                f"{wait:.0f}s, longer than reddit.arctic_shift_max_wait_seconds "
                f"({max_wait:.0f}s). Try the deposit again later.",
                url=url,
                status_code=429,
            )
        log.warning("Arctic Shift rate limit hit for %s; retrying once in %.1fs", url, wait)
        await _sleep(wait)
        resp = await _arctic_shift_request(client, url, params)
        if resp.status_code == 429:
            raise SourceFetchError(
                "The Arctic Shift archive is still rate-limiting requests after one "
                "retry. Try the deposit again later.",
                url=url,
                status_code=429,
            )
    if resp.status_code >= 400:
        raise SourceFetchError(
            f"The Arctic Shift archive returned HTTP {resp.status_code} for {url}",
            url=url,
            status_code=resp.status_code,
        )
    try:
        body = resp.json()
    except ValueError as exc:
        raise SourceFetchError(
            f"The Arctic Shift archive returned a body that is not JSON for {url}", url=url
        ) from exc
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        error = body.get("error") if isinstance(body, dict) else None
        raise SourceFetchError(
            f"The Arctic Shift archive could not answer {url}: {error or 'no data in response'}",
            url=url,
            status_code=resp.status_code,
        )
    return data


async def _arctic_shift_request(
    client: httpx.AsyncClient, url: str, params: dict[str, str]
) -> httpx.Response:
    """One size-capped GET, with transport failures mapped to ``SourceFetchError``.

    A policy refusal (``EgressRefused``) and an SSRF block (``UnsafeUrlError``)
    are neither ``httpx`` errors nor caught here, so each keeps its own meaning.
    """
    try:
        return await get_capped(client, url, params=params)
    except ResponseTooLarge as exc:
        raise SourceFetchError(
            f"The Arctic Shift archive response was too large: {exc}", url=url
        ) from exc
    except httpx.HTTPError as exc:
        raise SourceFetchError(
            f"Could not reach the Arctic Shift archive at {url}: {exc}", url=url
        ) from exc


def _rate_limit_wait(resp: httpx.Response, max_wait: float) -> float:
    """Seconds to wait before retrying a 429, from ``X-RateLimit-Reset``."""
    raw = resp.headers.get("x-ratelimit-reset")
    try:
        return (
            max(0.0, float(raw))
            if raw is not None
            else min(_DEFAULT_RATE_LIMIT_WAIT_SECONDS, max_wait)
        )
    except ValueError:
        return min(_DEFAULT_RATE_LIMIT_WAIT_SECONDS, max_wait)
