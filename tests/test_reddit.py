# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Reddit extractor and importer (naming)."""

from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from particles.core.schema import Snapshot, UncertaintyNature
from particles.extraction.reddit import (
    APPLICABILITY,
    DEFAULT_TRUST_WEIGHT,
    EXTRACTOR_ID,
    SOURCE_TYPE,
    RedditExtractor,
    _build_reddit_chunks,
    _count_unsettled_below_floor,
    _get_all_qualifying_comments,
    _get_fetch_provenance,
    _get_post,
    _infer_ticker,
    _is_score_settled,
    _render_reddit_body,
    _render_reddit_comment_lines,
)
from particles.ingest.importers.reddit import RedditImporter
from particles.url_safety import IPAddress, Resolver, UnsafeUrlError

# ---------------------------------------------------------------------------
# Stub DNS — the curl paths pre-resolve and pin their connection, so
# every test that reaches one injects a resolver and touches no network.
# ---------------------------------------------------------------------------

# A public Reddit-edge address; the value is arbitrary, only its publicness
# matters (a blocked address would be rejected before curl is spawned).
_PUBLIC_ADDR = "151.101.65.140"


def _stub_resolver(*addrs: str) -> Resolver:
    """A ``Resolver`` returning fixed addresses for any hostname."""
    ips: list[IPAddress] = [ipaddress.ip_address(a) for a in addrs]
    return lambda _host: list(ips)


# ---------------------------------------------------------------------------
# Fixtures — realistic Reddit JSON API shape
# ---------------------------------------------------------------------------

_POST_DATA = {
    "subreddit": "POETTechnologiesInc",
    "title": "POET -30% premarket after Marvell cancels partnership",
    "author": "throwaway_investor",
    "score": 1594,
    "selftext": "Marvell just announced they are walking away from the deal.",
}

_COMMENT_DATA = [
    {
        "kind": "t1",
        "data": {"author": "bullish_guy", "score": 45, "body": "This is FUD.", "replies": ""},
    },
    {
        "kind": "t1",
        "data": {
            "author": "bearish_gal",
            "score": 120,
            "body": "Management lied about the timeline.",
            "replies": "",
        },
    },
    {
        "kind": "t1",
        "data": {"author": "deleted_user", "score": 5, "body": "[deleted]", "replies": ""},
    },
    {"kind": "t1", "data": {"author": "low_score", "score": 1, "body": "meh", "replies": ""}},
]

_REDDIT_BLOB: list = [
    {"data": {"children": [{"kind": "t3", "data": _POST_DATA}]}},
    {"data": {"children": _COMMENT_DATA}},
]


def _make_snapshot() -> Snapshot:
    from datetime import UTC, datetime

    from particles.core.schema import ExtractionStatus, WarcRecordType

    return Snapshot(
        snapshot_id="test-snap",
        captured_at=datetime.now(UTC),
        content_hash="abc",
        extraction_status=ExtractionStatus.PENDING,
        warc_record_type=WarcRecordType.RESPONSE,
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_source_type(self) -> None:
        assert SOURCE_TYPE == "REDDIT_POST"

    def test_extractor_id(self) -> None:
        assert EXTRACTOR_ID == "reddit-extractor"

    def test_default_trust_weight(self) -> None:
        assert pytest.approx(0.40) == DEFAULT_TRUST_WEIGHT

    def test_applicability_must_social_media(self) -> None:
        assert len(APPLICABILITY) == 1
        clause = APPLICABILITY[0]
        assert clause.keyword == "MUST"
        assert clause.domain_label == "social media"
        assert "REDDIT_POST" in clause.source_types

    def test_infer_domain_returns_social_media(self) -> None:
        from particles.extraction.registry import infer_domain

        assert infer_domain("REDDIT_POST") == "social media"


# ---------------------------------------------------------------------------
# Importer URL matching
# ---------------------------------------------------------------------------


class TestRedditImporter:
    def test_accepts_www_reddit_url(self) -> None:
        d = RedditImporter()
        assert d.accepts_url("https://www.reddit.com/r/POETTechnologiesInc/comments/abc123/title/")

    def test_accepts_old_reddit_url(self) -> None:
        d = RedditImporter()
        assert d.accepts_url("https://old.reddit.com/r/wallstreetbets/comments/xyz789/post/")

    def test_rejects_non_reddit(self) -> None:
        d = RedditImporter()
        assert not d.accepts_url("https://twitter.com/someone")
        assert not d.accepts_url("https://numista.com/coin/1")

    def test_rejects_reddit_non_thread(self) -> None:
        d = RedditImporter()
        # subreddit listing, not a thread
        assert not d.accepts_url("https://www.reddit.com/r/wallstreetbets/")

    def test_accepts_share_url(self) -> None:
        """`/s/` share links (the iOS share-sheet form) are accepted;
        the redirect to the `/comments/` permalink is resolved in `deposit`."""
        d = RedditImporter()
        assert d.accepts_url("https://www.reddit.com/r/AIEval/s/2KHTCY0Bgq")
        assert d.accepts_url("https://old.reddit.com/r/AIEval/s/2KHTCY0Bgq/")

    def test_rejects_share_url_on_non_reddit_host(self) -> None:
        """A `/s/`-shaped path on a non-reddit host must not route to Reddit."""
        d = RedditImporter()
        assert not d.accepts_url("https://evil.com/r/AIEval/s/2KHTCY0Bgq")

    # ----- primary_url -----

    def test_primary_url_link_post(self) -> None:
        """A Reddit link post carries ``url`` pointing at the external
        article. ``primary_url`` returns it for the follow."""
        blob = [
            {
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "url": "https://www.theatlantic.com/article-x",
                                "is_self": False,
                                "selftext": "",
                            },
                        }
                    ]
                }
            },
            {"data": {"children": []}},
        ]
        d = RedditImporter()
        assert d.primary_url(json.dumps(blob).encode()) == "https://www.theatlantic.com/article-x"

    def test_primary_url_self_post_returns_none(self) -> None:
        """Self-posts (``is_self=True``) have no external URL to follow."""
        blob = [
            {
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "url": "https://www.reddit.com/r/foo/comments/abc/title/",
                                "is_self": True,
                                "selftext": "the body lives here",
                            },
                        }
                    ]
                }
            },
            {"data": {"children": []}},
        ]
        d = RedditImporter()
        assert d.primary_url(json.dumps(blob).encode()) is None

    def test_primary_url_thread_permalink_returns_none(self) -> None:
        """A Reddit post whose ``url`` field points back at the thread
        itself (the legacy self-post case where ``is_self`` is missing)
        has no external URL to follow."""
        blob = [
            {
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "url": "https://www.reddit.com/r/foo/comments/abc/title/",
                                # is_self deliberately absent
                            },
                        }
                    ]
                }
            },
            {"data": {"children": []}},
        ]
        d = RedditImporter()
        assert d.primary_url(json.dumps(blob).encode()) is None

    def test_primary_url_malformed_returns_none(self) -> None:
        d = RedditImporter()
        assert d.primary_url(b"not json") is None
        assert d.primary_url(b"[]") is None  # well-formed but empty
        assert d.primary_url(b"{}") is None  # wrong shape

    def test_default_follow_post_links_true(self) -> None:
        """Reddit ships at follow_post_links=True."""
        assert RedditImporter.DEFAULT_FOLLOW_POST_LINKS is True
        assert RedditImporter.DEFAULT_FOLLOW_COMMENT_LINKS is False

    @pytest.mark.asyncio
    async def test_deposit_stores_author_id(self, db_session: object) -> None:
        content = json.dumps(_REDDIT_BLOB).encode()

        with patch(
            "particles.ingest.importers.reddit._fetch_thread",
            AsyncMock(return_value=content),
        ):
            d = RedditImporter()
            entry_id, snapshot_id = await d.deposit(
                db_session,  # type: ignore[arg-type]
                "https://www.reddit.com/r/POETTechnologiesInc/comments/abc123/title/",
                "test-operator",
                [],
            )

        from particles.corpus.store import get_snapshot

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        assert snap.author_id == "reddit:u/throwaway_investor"

    @pytest.mark.asyncio
    async def test_deposit_source_type(self, db_session: object) -> None:
        content = json.dumps(_REDDIT_BLOB).encode()

        with patch(
            "particles.ingest.importers.reddit._fetch_thread",
            AsyncMock(return_value=content),
        ):
            d = RedditImporter()
            entry_id, _ = await d.deposit(
                db_session,  # type: ignore[arg-type]
                "https://www.reddit.com/r/POETTechnologiesInc/comments/abc123/title/",
                "test-operator",
                [],
            )

        from particles.corpus.store import get_entry

        entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
        assert entry is not None
        assert entry.source_type == "REDDIT_POST"

    @pytest.mark.asyncio
    async def test_deposit_share_link_resolves_and_canonicalizes(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a `/s/` share link deposits under its resolved `/comments/`
        permalink, and on the direct backend the `.json` fetch targets that
        canonical URL."""
        from particles.config import get_config

        monkeypatch.setattr(get_config().reddit, "fetch_backend", "reddit")
        content = json.dumps(_REDDIT_BLOB).encode()
        canonical = "https://www.reddit.com/r/AIEval/comments/1uf0fkd/slug/"
        fetched: list[str] = []

        async def _fake_fetch(url: str) -> bytes:
            fetched.append(url)
            return content

        with (
            patch(
                "particles.ingest.importers.reddit._resolve_reddit_redirect",
                AsyncMock(return_value=canonical),
            ),
            patch("particles.ingest.importers.reddit._fetch_with_curl", _fake_fetch),
        ):
            d = RedditImporter()
            entry_id, _ = await d.deposit(
                db_session,  # type: ignore[arg-type]
                "https://www.reddit.com/r/AIEval/s/2KHTCY0Bgq",
                "test-operator",
                [],
            )

        from particles.corpus.store import get_entry

        entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
        assert entry is not None
        assert entry.uri_r == canonical
        assert entry.source_type == "REDDIT_POST"
        # The JSON API URL is built from the canonical permalink, not the /s/ link.
        assert fetched == [canonical + ".json?limit=200"]

    @pytest.mark.asyncio
    async def test_deposit_share_link_dedups_with_direct(self, db_session: object) -> None:
        """A `/s/` link and the direct `/comments/` link for the same post
        collapse to one corpus entry (uri_r dedup)."""
        content = json.dumps(_REDDIT_BLOB).encode()
        canonical = "https://www.reddit.com/r/AIEval/comments/1uf0fkd/slug/"

        with patch(
            "particles.ingest.importers.reddit._fetch_thread",
            AsyncMock(return_value=content),
        ):
            d = RedditImporter()
            direct_id, _ = await d.deposit(
                db_session,
                canonical,
                "test-operator",
                [],  # type: ignore[arg-type]
            )
            with patch(
                "particles.ingest.importers.reddit._resolve_reddit_redirect",
                AsyncMock(return_value=canonical),
            ):
                share_id, _ = await d.deposit(
                    db_session,  # type: ignore[arg-type]
                    "https://www.reddit.com/r/AIEval/s/2KHTCY0Bgq",
                    "test-operator",
                    [],
                )

        assert share_id == direct_id


# ---------------------------------------------------------------------------
# Arctic Shift backend, the default fetch source
# ---------------------------------------------------------------------------

_PERMALINK = "https://www.reddit.com/r/AIEval/comments/1uf0fkd/slug/"
_NOW = 1_792_000_000  # arbitrary epoch seconds


def _archived_post(*, settled: bool) -> dict[str, Any]:
    """An archive post record: Reddit's fields plus ``retrieved_on`` and ``_meta``."""
    return {
        **_POST_DATA,
        "id": "1uf0fkd",
        "is_self": True,
        "created_utc": _NOW - 3600,
        "retrieved_on": _NOW - 3500,
        "_meta": {"retrieved_2nd_on": _NOW} if settled else {},
    }


def _archived_comment(
    author: str,
    score: int,
    body: str,
    *,
    settled: bool,
    replies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "t1",
        "data": {
            "author": author,
            "score": score,
            "body": body,
            "created_utc": _NOW - 3000,
            "retrieved_on": _NOW - 2900,
            "_meta": {"retrieved_2nd_on": _NOW} if settled else {},
            "replies": {"kind": "Listing", "data": {"children": replies}} if replies else "",
        },
    }


def _archive_tree(*, settled: bool) -> list[dict[str, Any]]:
    """Two top-level comments and one nested reply, in the archive's tree shape.

    Unsettled records carry the archive's placeholder score of 1.
    """
    reply = _archived_comment(
        "deep_reply", 30 if settled else 1, "Nested insight.", settled=settled
    )
    return [
        _archived_comment(
            "bearish_gal",
            120 if settled else 1,
            "Management lied.",
            settled=settled,
            replies=[reply],
        ),
        _archived_comment("low_score", 1, "meh", settled=settled),
    ]


class _ArchiveStub:
    """Serves the archive's two endpoints and records every request it gets.

    ``queued`` responses are returned first, in order, whatever the path.
    """

    def __init__(
        self,
        *,
        posts: list[Any] | None = None,
        tree: list[Any] | None = None,
        queued: list[httpx.Response] | None = None,
    ) -> None:
        self.posts = posts if posts is not None else [_archived_post(settled=True)]
        self.tree = tree if tree is not None else _archive_tree(settled=True)
        self.queued = list(queued or [])
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.queued:
            return self.queued.pop(0)
        if request.url.path == "/api/posts/ids":
            return httpx.Response(200, json={"data": self.posts})
        if request.url.path == "/api/comments/tree":
            return httpx.Response(200, json={"data": self.tree})
        return httpx.Response(404)


@contextlib.contextmanager
def _serve_archive(stub: _ArchiveStub) -> Iterator[None]:
    """Route the importer's ``particles_client()`` to ``stub`` (no network)."""

    @contextlib.asynccontextmanager
    async def _client(**_kwargs: object) -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(stub)) as client:
            yield client

    with patch("particles.ingest.importers.reddit.particles_client", _client):
        yield


async def _deposited_blob(db_session: object, snapshot_id: str) -> Any:
    from particles.corpus.deposit import load_blob
    from particles.corpus.store import get_snapshot

    snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
    assert snap is not None
    return json.loads(load_blob(snap.content_hash))


class TestArcticShiftFetch:
    """The importer fetches thread content from Arctic Shift and stores Reddit's shape."""

    @pytest.mark.asyncio
    async def test_reassembles_reddit_envelope(self, db_session: object) -> None:
        """The archive's ``{"data": [...]}`` bodies become Reddit's two listings,
        which the unchanged parsers read, under the reddit.com URI-R."""
        from particles.corpus.store import get_entry

        stub = _ArchiveStub()
        with _serve_archive(stub):
            entry_id, snapshot_id = await RedditImporter().deposit(
                db_session,  # type: ignore[arg-type]
                _PERMALINK,
                "test-operator",
                [],
            )

        entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
        assert entry is not None
        assert entry.uri_r == _PERMALINK
        assert entry.source_type == SOURCE_TYPE

        data = await _deposited_blob(db_session, snapshot_id)
        post = _get_post(data)
        assert post is not None and post["id"] == "1uf0fkd"
        comments = _get_all_qualifying_comments(data)
        # The nested reply survives reassembly; the score-1 comment is filtered.
        assert [c["author"] for c in comments] == ["bearish_gal", "deep_reply"]
        assert RedditImporter().primary_url(json.dumps(data).encode()) is None  # self-post

        paths = [(r.url.path, dict(r.url.params)) for r in stub.requests]
        assert paths == [
            ("/api/posts/ids", {"ids": "1uf0fkd"}),
            ("/api/comments/tree", {"link_id": "1uf0fkd", "limit": "25000"}),
        ]

    @pytest.mark.asyncio
    async def test_records_archive_provenance(self, db_session: object) -> None:
        """The blob names the archive host and its capture times, and nothing
        from this fetch's own clock, so an unchanged record re-deposits to the
        same bytes."""
        from particles.corpus.store import get_snapshot

        with _serve_archive(_ArchiveStub()):
            importer = RedditImporter()
            _, first = await importer.deposit(db_session, _PERMALINK, "op", [])  # type: ignore[arg-type]
            _, second = await importer.deposit(db_session, _PERMALINK, "op", [])  # type: ignore[arg-type]

        provenance = _get_fetch_provenance(await _deposited_blob(db_session, first))
        assert provenance == {
            "backend": "arctic_shift",
            "archive": "Arctic Shift",
            "archive_host": "arctic-shift.photon-reddit.com",
            "post_id": "1uf0fkd",
            "post_retrieved_on": _NOW - 3500,
            "post_retrieved_2nd_on": _NOW,
            "post_score_settled": True,
            "comment_count": 3,
            "comments_retrieved_on_earliest": _NOW - 2900,
            "comments_retrieved_on_latest": _NOW - 2900,
            "comments_score_unsettled": 0,
        }
        snap_a = await get_snapshot(db_session, first)  # type: ignore[arg-type]
        snap_b = await get_snapshot(db_session, second)  # type: ignore[arg-type]
        assert snap_a is not None and snap_b is not None
        assert snap_a.content_hash == snap_b.content_hash

    @pytest.mark.asyncio
    async def test_share_link_fetches_resolved_post_from_archive(self, db_session: object) -> None:
        """The share-link redirect still resolves through Reddit; the content
        comes from the archive under the resolved post id."""
        stub = _ArchiveStub()
        with (
            _serve_archive(stub),
            patch(
                "particles.ingest.importers.reddit._resolve_reddit_redirect",
                AsyncMock(return_value=_PERMALINK),
            ),
        ):
            await RedditImporter().deposit(
                db_session,  # type: ignore[arg-type]
                "https://www.reddit.com/r/AIEval/s/2KHTCY0Bgq",
                "test-operator",
                [],
            )
        assert stub.requests[0].url.params["ids"] == "1uf0fkd"

    @pytest.mark.asyncio
    async def test_not_archived_is_one_clear_failure(self) -> None:
        """An empty ``data`` list means the archive lacks the post: a
        SourceFetchError naming Arctic Shift, with no OAuth advice."""
        from particles.http import SourceFetchError
        from particles.ingest.importers.reddit import _fetch_thread

        stub = _ArchiveStub(posts=[])
        with _serve_archive(stub), pytest.raises(SourceFetchError) as ei:
            await _fetch_thread(_PERMALINK)
        message = str(ei.value)
        assert "Arctic Shift" in message and "has not archived" in message
        assert "OAuth" not in message
        assert ei.value.status_code == 404
        assert len(stub.requests) == 1  # no comment-tree request for a missing post

    @pytest.mark.asyncio
    async def test_error_body_is_a_fetch_failure(self) -> None:
        """``{"data": null, "error": ...}`` (the archive's timeout answer) is a
        SourceFetchError carrying the archive's own words, not a traceback."""
        from particles.http import SourceFetchError
        from particles.ingest.importers.reddit import _fetch_thread

        busy = httpx.Response(200, json={"data": None, "error": "Timeout. Maybe slow down a bit"})
        with _serve_archive(_ArchiveStub(queued=[busy])), pytest.raises(SourceFetchError) as ei:
            await _fetch_thread(_PERMALINK)
        assert "Timeout. Maybe slow down a bit" in str(ei.value)

    @pytest.mark.asyncio
    async def test_server_error_is_a_fetch_failure(self) -> None:
        from particles.http import SourceFetchError
        from particles.ingest.importers.reddit import _fetch_thread

        with (
            _serve_archive(_ArchiveStub(queued=[httpx.Response(503)])),
            pytest.raises(SourceFetchError) as ei,
        ):
            await _fetch_thread(_PERMALINK)
        assert ei.value.status_code == 503

    @pytest.mark.asyncio
    async def test_429_waits_the_reset_and_retries_once(self) -> None:
        from particles.ingest.importers.reddit import _fetch_thread

        limited = httpx.Response(429, headers={"X-RateLimit-Reset": "2"})
        stub = _ArchiveStub(queued=[limited])
        sleep = AsyncMock()
        with _serve_archive(stub), patch("particles.ingest.importers.reddit._sleep", sleep):
            content = await _fetch_thread(_PERMALINK)
        sleep.assert_awaited_once_with(2.0)
        assert _get_post(json.loads(content)) is not None
        assert len(stub.requests) == 3  # the limited post request, its retry, the tree

    @pytest.mark.asyncio
    async def test_second_429_fails_without_another_wait(self) -> None:
        from particles.http import SourceFetchError
        from particles.ingest.importers.reddit import _fetch_thread

        limited = [httpx.Response(429, headers={"X-RateLimit-Reset": "1"}) for _ in range(2)]
        stub = _ArchiveStub(queued=limited)
        sleep = AsyncMock()
        with (
            _serve_archive(stub),
            patch("particles.ingest.importers.reddit._sleep", sleep),
            pytest.raises(SourceFetchError) as ei,
        ):
            await _fetch_thread(_PERMALINK)
        assert ei.value.status_code == 429
        assert sleep.await_count == 1
        assert len(stub.requests) == 2

    @pytest.mark.asyncio
    async def test_429_reset_beyond_the_cap_fails_at_once(self) -> None:
        """A reset further away than ``arctic_shift_max_wait_seconds`` does not
        stall the deposit."""
        from particles.http import SourceFetchError
        from particles.ingest.importers.reddit import _fetch_thread

        limited = httpx.Response(429, headers={"X-RateLimit-Reset": "600"})
        stub = _ArchiveStub(queued=[limited])
        sleep = AsyncMock()
        with (
            _serve_archive(stub),
            patch("particles.ingest.importers.reddit._sleep", sleep),
            pytest.raises(SourceFetchError, match="arctic_shift_max_wait_seconds"),
        ):
            await _fetch_thread(_PERMALINK)
        sleep.assert_not_awaited()
        assert len(stub.requests) == 1

    @pytest.mark.asyncio
    async def test_reddit_backend_routes_to_curl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``fetch_backend: reddit`` keeps the direct ``.json`` fetch and never
        touches the archive."""
        from particles.config import get_config
        from particles.ingest.importers.reddit import _fetch_thread

        monkeypatch.setattr(get_config().reddit, "fetch_backend", "reddit")
        curl = AsyncMock(return_value=b"[]")
        stub = _ArchiveStub()
        with (
            _serve_archive(stub),
            patch("particles.ingest.importers.reddit._fetch_with_curl", curl),
        ):
            assert await _fetch_thread(_PERMALINK) == b"[]"
        curl.assert_awaited_once_with(_PERMALINK + ".json?limit=200")
        assert stub.requests == []


class TestUnsettledArchiveScores:
    """Archive scores read 0 or 1 until the second pass; the deposit says so."""

    @pytest.mark.asyncio
    async def test_unsettled_deposit_warns_and_extractor_notes_it(
        self, db_session: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        stub = _ArchiveStub(
            posts=[_archived_post(settled=False)], tree=_archive_tree(settled=False)
        )
        with (
            _serve_archive(stub),
            caplog.at_level(logging.WARNING, logger="particles.ingest.importers.reddit"),
        ):
            _, snapshot_id = await RedditImporter().deposit(
                db_session,  # type: ignore[arg-type]
                _PERMALINK,
                "test-operator",
                [],
            )
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "3 of 3 comments" in warnings[0] and "corpus refresh" in warnings[0]

        data = await _deposited_blob(db_session, snapshot_id)
        assert _get_all_qualifying_comments(data) == []  # the floor still applies
        provenance = _get_fetch_provenance(data)
        assert provenance is not None
        assert provenance["post_score_settled"] is False
        assert provenance["comments_score_unsettled"] == 3

        doc = RedditExtractor()._normalise(json.dumps(data).encode(), _make_snapshot())
        notes = [n for n in doc.quality_notes if n.startswith("REDDIT_SCORES_UNSETTLED")]
        assert len(notes) == 1 and "3 of 3 comments" in notes[0]

    @pytest.mark.asyncio
    async def test_settled_deposit_is_quiet(
        self, db_session: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            _serve_archive(_ArchiveStub()),
            caplog.at_level(logging.WARNING, logger="particles.ingest.importers.reddit"),
        ):
            _, snapshot_id = await RedditImporter().deposit(
                db_session,  # type: ignore[arg-type]
                _PERMALINK,
                "test-operator",
                [],
            )
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]
        data = await _deposited_blob(db_session, snapshot_id)
        doc = RedditExtractor()._normalise(json.dumps(data).encode(), _make_snapshot())
        assert not any(n.startswith("REDDIT_SCORES_UNSETTLED") for n in doc.quality_notes)

    def test_reddit_served_blob_is_never_unsettled(self) -> None:
        """A blob without the provenance record came from Reddit, whose scores
        are live: the low-score comment there is a real low score."""
        assert _count_unsettled_below_floor(_REDDIT_BLOB) == (0, 0)

    def test_late_first_capture_counts_as_settled(self) -> None:
        """A record the archive first captured long after it was made (a
        historical import) carries a real score with no second pass."""
        late = {"created_utc": _NOW, "retrieved_on": _NOW + 40 * 3600, "_meta": {}}
        fresh = {"created_utc": _NOW, "retrieved_on": _NOW + 60, "_meta": {}}
        second_pass = {
            "created_utc": _NOW,
            "retrieved_on": _NOW + 60,
            "_meta": {"retrieved_2nd_on": 1},
        }
        assert _is_score_settled(late) is True
        assert _is_score_settled(fresh) is False
        assert _is_score_settled(second_pass) is True
        assert _is_score_settled({}) is False


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


class TestRefresh:
    """``corpus refresh`` re-runs the importer, not a GET of the permalink."""

    @pytest.mark.asyncio
    async def test_refetch_reproduces_the_deposited_bytes(self, db_session: object) -> None:
        from particles.corpus.store import get_snapshot

        with _serve_archive(_ArchiveStub()):
            _, snapshot_id = await RedditImporter().deposit(
                db_session,  # type: ignore[arg-type]
                _PERMALINK,
                "test-operator",
                [],
            )
            refetched = await RedditImporter().refetch(_PERMALINK)

        from particles.corpus.deposit import sha256

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None and sha256(refetched) == snap.content_hash

    @pytest.mark.asyncio
    async def test_refresh_picks_up_settled_scores(self, db_session: object) -> None:
        from particles.corpus.refresh_outcome import RefreshOutcome
        from particles.operations.corpus_refresh import refresh_entries

        unsettled = _ArchiveStub(
            posts=[_archived_post(settled=False)], tree=_archive_tree(settled=False)
        )
        with _serve_archive(unsettled):
            entry_id, first_id = await RedditImporter().deposit(
                db_session,  # type: ignore[arg-type]
                _PERMALINK,
                "test-operator",
                [],
            )
        await db_session.commit()  # type: ignore[attr-defined]

        settled = _ArchiveStub()
        # A GET of the reddit.com permalink would go through the ladder's own
        # client; fail loudly if it is ever reached.
        with (
            _serve_archive(settled),
            patch("particles.corpus.fetch.particles_client", side_effect=AssertionError),
        ):
            results = [r async for r in refresh_entries(db_session, [entry_id], force=True)]  # type: ignore[arg-type]

        assert len(results) == 1 and results[0].error is None
        assert results[0].outcome == RefreshOutcome.CHANGED
        assert {r.url.host for r in settled.requests} == {"arctic-shift.photon-reddit.com"}

        data = await _deposited_blob(db_session, results[0].snapshot_id or "")
        assert results[0].snapshot_id != first_id
        assert [c["author"] for c in _get_all_qualifying_comments(data)] == [
            "bearish_gal",
            "deep_reply",
        ]
        provenance = _get_fetch_provenance(data)
        assert provenance is not None and provenance["comments_score_unsettled"] == 0

    @pytest.mark.asyncio
    async def test_refresh_of_an_unchanged_thread_is_a_revisit(self, db_session: object) -> None:
        from particles.corpus.refresh_outcome import RefreshOutcome
        from particles.operations.corpus_refresh import refresh_entries

        with _serve_archive(_ArchiveStub()):
            entry_id, _ = await RedditImporter().deposit(
                db_session,  # type: ignore[arg-type]
                _PERMALINK,
                "test-operator",
                [],
            )
            await db_session.commit()  # type: ignore[attr-defined]
            results = [r async for r in refresh_entries(db_session, [entry_id], force=True)]  # type: ignore[arg-type]

        assert results[0].error is None
        assert results[0].outcome == RefreshOutcome.UNCHANGED_HASH


class TestFetchWithCurl:
    """The curl subprocess must carry the security limits (F-5 hardening)."""

    @pytest.mark.asyncio
    async def test_curl_argv_includes_size_and_time_limits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config
        from particles.extraction.reddit import _fetch_with_curl

        captured_argv: list[str] = []

        async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            captured_argv.extend(argv)
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"{}", b""))
            return proc

        monkeypatch.setattr(
            "particles.extraction.reddit.asyncio.create_subprocess_exec", _fake_exec
        )

        out = await _fetch_with_curl(
            "https://www.reddit.com/r/x/comments/abc/t/.json",
            resolve=_stub_resolver(_PUBLIC_ADDR),
        )
        assert out == b"{}"

        cfg = get_config().http
        # --max-filesize bounds the body; value is config.http.max_bytes.
        assert "--max-filesize" in captured_argv
        assert captured_argv[captured_argv.index("--max-filesize") + 1] == str(cfg.max_bytes)
        # --max-time bounds wall-clock; value is config.http.timeout_seconds.
        assert "--max-time" in captured_argv
        assert captured_argv[captured_argv.index("--max-time") + 1] == str(cfg.timeout_seconds)

    @pytest.mark.asyncio
    async def test_403_raises_source_fetch_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 403 (Reddit's bot-wall) is a typed SourceFetchError with status 403
        that points at the archive backend — not a bare RuntimeError/traceback."""
        from particles.extraction.reddit import _fetch_with_curl
        from particles.http import SourceFetchError

        async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            proc = MagicMock()
            proc.returncode = 22
            proc.communicate = AsyncMock(
                return_value=(b"", b"curl: (22) The requested URL returned error: 403")
            )
            return proc

        monkeypatch.setattr(
            "particles.extraction.reddit.asyncio.create_subprocess_exec", _fake_exec
        )
        with pytest.raises(SourceFetchError) as ei:
            await _fetch_with_curl(
                "https://www.reddit.com/r/x/comments/abc/t/.json",
                resolve=_stub_resolver(_PUBLIC_ADDR),
            )
        assert ei.value.status_code == 403
        # Name the archive backend, never a closed OAuth route.
        assert "403" in str(ei.value) and "arctic_shift" in str(ei.value)
        assert "OAuth" not in str(ei.value)

    @pytest.mark.asyncio
    async def test_non_http_failure_has_no_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A transport failure (no HTTP status in stderr) still raises
        SourceFetchError, with status_code None."""
        from particles.extraction.reddit import _fetch_with_curl
        from particles.http import SourceFetchError

        async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            proc = MagicMock()
            proc.returncode = 28
            proc.communicate = AsyncMock(
                return_value=(b"", b"curl: (28) Operation timed out after 30000 ms")
            )
            return proc

        monkeypatch.setattr(
            "particles.extraction.reddit.asyncio.create_subprocess_exec", _fake_exec
        )
        with pytest.raises(SourceFetchError) as ei:
            await _fetch_with_curl(
                "https://www.reddit.com/r/x/comments/abc/t/.json",
                resolve=_stub_resolver(_PUBLIC_ADDR),
            )
        assert ei.value.status_code is None

    def test_parse_curl_http_status(self) -> None:
        from particles.extraction.reddit import _parse_curl_http_status

        assert _parse_curl_http_status("curl: (22) The requested URL returned error: 403") == 403
        assert _parse_curl_http_status("curl: (28) Operation timed out") is None


def _patch_curl(
    monkeypatch: pytest.MonkeyPatch, *, stdout: bytes, returncode: int = 0
) -> list[str]:
    """Patch the redirect-resolver's curl subprocess to return fixed headers.

    Returns a list that captures the argv the resolver invoked curl with.
    """
    captured_argv: list[str] = []

    async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
        captured_argv.extend(argv)
        proc = MagicMock()
        proc.returncode = returncode
        proc.communicate = AsyncMock(return_value=(stdout, b""))
        return proc

    monkeypatch.setattr("particles.extraction.reddit.asyncio.create_subprocess_exec", _fake_exec)
    return captured_argv


class TestResolveRedditRedirect:
    """`/s/` share-link resolution + the open-redirect guard."""

    @pytest.mark.asyncio
    async def test_resolves_post_share_and_strips_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.extraction.reddit import _resolve_reddit_redirect

        headers = (
            b"HTTP/2 302 \r\n"
            b"location: https://www.reddit.com/r/AIEval/comments/1uf0fkd/slug/"
            b"?share_id=x&utm_source=share\r\n\r\n"
        )
        argv = _patch_curl(monkeypatch, stdout=headers)
        out = await _resolve_reddit_redirect(
            "https://www.reddit.com/r/AIEval/s/2KHTCY0Bgq", resolve=_stub_resolver(_PUBLIC_ADDR)
        )
        assert out == "https://www.reddit.com/r/AIEval/comments/1uf0fkd/slug/"
        # Never follows the redirect itself (guard validates the target instead).
        assert "--max-redirs" in argv and argv[argv.index("--max-redirs") + 1] == "0"
        assert "-L" not in argv

    @pytest.mark.asyncio
    async def test_comment_share_reduces_to_post_permalink(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shared *comment* resolves to a 5-segment permalink; it is reduced to
        the post permalink so the whole thread is fetched."""
        from particles.extraction.reddit import _resolve_reddit_redirect

        headers = (
            b"HTTP/1.1 301 Moved Permanently\r\n"
            b"Location: https://www.reddit.com/r/AIEval/comments/1uf0fkd/slug/abc123/"
            b"?context=3\r\n\r\n"
        )
        _patch_curl(monkeypatch, stdout=headers)
        out = await _resolve_reddit_redirect(
            "https://www.reddit.com/r/AIEval/s/xyz", resolve=_stub_resolver(_PUBLIC_ADDR)
        )
        assert out == "https://www.reddit.com/r/AIEval/comments/1uf0fkd/slug/"

    @pytest.mark.asyncio
    async def test_rejects_non_reddit_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.extraction.reddit import _resolve_reddit_redirect

        headers = b"HTTP/2 302 \r\nlocation: https://evil.com/r/AIEval/comments/x/y/\r\n\r\n"
        _patch_curl(monkeypatch, stdout=headers)
        with pytest.raises(RuntimeError, match="unexpected target"):
            await _resolve_reddit_redirect(
                "https://www.reddit.com/r/AIEval/s/xyz", resolve=_stub_resolver(_PUBLIC_ADDR)
            )

    @pytest.mark.asyncio
    async def test_rejects_userinfo_host_trick(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`www.reddit.com@evil.com` — the real host is evil.com; must reject."""
        from particles.extraction.reddit import _resolve_reddit_redirect

        headers = (
            b"HTTP/2 302 \r\n"
            b"location: https://www.reddit.com@evil.com/r/AIEval/comments/x/y/\r\n\r\n"
        )
        _patch_curl(monkeypatch, stdout=headers)
        with pytest.raises(RuntimeError, match="unexpected target"):
            await _resolve_reddit_redirect(
                "https://www.reddit.com/r/AIEval/s/xyz", resolve=_stub_resolver(_PUBLIC_ADDR)
            )

    @pytest.mark.asyncio
    async def test_rejects_suffix_host_trick(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`reddit.com.evil.com` must not be treated as reddit.com."""
        from particles.extraction.reddit import _resolve_reddit_redirect

        headers = (
            b"HTTP/2 302 \r\nlocation: https://reddit.com.evil.com/r/AIEval/comments/x/y/\r\n\r\n"
        )
        _patch_curl(monkeypatch, stdout=headers)
        with pytest.raises(RuntimeError, match="unexpected target"):
            await _resolve_reddit_redirect(
                "https://www.reddit.com/r/AIEval/s/xyz", resolve=_stub_resolver(_PUBLIC_ADDR)
            )

    @pytest.mark.asyncio
    async def test_raises_when_no_location_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.extraction.reddit import _resolve_reddit_redirect

        headers = b"HTTP/2 200 \r\ncontent-type: text/html\r\n\r\n"
        _patch_curl(monkeypatch, stdout=headers)
        with pytest.raises(RuntimeError, match="did not redirect"):
            await _resolve_reddit_redirect(
                "https://www.reddit.com/r/AIEval/s/xyz", resolve=_stub_resolver(_PUBLIC_ADDR)
            )

    @pytest.mark.asyncio
    async def test_raises_on_curl_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A share link that 404s makes curl --fail exit non-zero → raise."""
        from particles.extraction.reddit import _resolve_reddit_redirect

        _patch_curl(monkeypatch, stdout=b"", returncode=22)
        with pytest.raises(RuntimeError, match="curl failed"):
            await _resolve_reddit_redirect(
                "https://www.reddit.com/r/AIEval/s/xyz", resolve=_stub_resolver(_PUBLIC_ADDR)
            )


class TestShareLinkRefusal:
    """The share-link path fails the same clean way as the fetch."""

    @pytest.mark.asyncio
    async def test_403_is_a_source_fetch_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.extraction.reddit import _resolve_reddit_redirect
        from particles.http import SourceFetchError

        async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            proc = MagicMock()
            proc.returncode = 22
            proc.communicate = AsyncMock(
                return_value=(b"", b"curl: (22) The requested URL returned error: 403")
            )
            return proc

        monkeypatch.setattr(
            "particles.extraction.reddit.asyncio.create_subprocess_exec", _fake_exec
        )
        with pytest.raises(SourceFetchError) as ei:
            await _resolve_reddit_redirect(
                "https://www.reddit.com/r/AIEval/s/xyz", resolve=_stub_resolver(_PUBLIC_ADDR)
            )
        assert ei.value.status_code == 403
        assert "permalink" in str(ei.value) and "OAuth" not in str(ei.value)


class TestCurlConnectPin:
    """the curl subprocess connects to an address this process vetted.

    curl never enters httpx, so ValidatingTransport cannot see it.
    ``--resolve`` reaches the same guarantee by a second mechanism: the address
    is resolved and blocklist-checked in-process, then curl is pinned to it, so
    the validated address *is* the connected address.
    """

    @pytest.mark.asyncio
    async def test_fetch_argv_carries_the_pin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.extraction.reddit import _fetch_with_curl

        captured: list[str] = []

        async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            captured.extend(argv)
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"{}", b""))
            return proc

        monkeypatch.setattr(
            "particles.extraction.reddit.asyncio.create_subprocess_exec", _fake_exec
        )
        await _fetch_with_curl(
            "https://www.reddit.com/r/x/comments/abc/t/.json",
            resolve=_stub_resolver(_PUBLIC_ADDR),
        )

        assert "--resolve" in captured
        assert captured[captured.index("--resolve") + 1] == f"www.reddit.com:443:{_PUBLIC_ADDR}"

    @pytest.mark.asyncio
    async def test_multi_address_pin_passes_all_vetted_addresses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multi-A-record host pins the whole vetted set, preserving failover.

        Every address has already passed ``_is_blocked_ip``, so widening from
        one to all cannot admit a blocked one — it only keeps libcurl's own
        failover across nodes this process approved.
        """
        from particles.extraction.reddit import _fetch_with_curl

        captured: list[str] = []

        async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            captured.extend(argv)
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"{}", b""))
            return proc

        monkeypatch.setattr(
            "particles.extraction.reddit.asyncio.create_subprocess_exec", _fake_exec
        )
        await _fetch_with_curl(
            "https://www.reddit.com/r/x/comments/abc/t/.json",
            resolve=_stub_resolver("151.101.1.140", "151.101.65.140", "2a04:4e42::396"),
        )

        pin = captured[captured.index("--resolve") + 1]
        # IPv6 is bracketed so its own colons can't be read as field separators.
        assert pin == "www.reddit.com:443:151.101.1.140,151.101.65.140,[2a04:4e42::396]"

    @pytest.mark.asyncio
    async def test_blocked_address_short_circuits_before_curl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rebinding-equivalent case: no subprocess is spawned at all."""
        from particles.extraction.reddit import _fetch_with_curl

        spawned = False

        async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            nonlocal spawned
            spawned = True
            return MagicMock()

        monkeypatch.setattr(
            "particles.extraction.reddit.asyncio.create_subprocess_exec", _fake_exec
        )
        with pytest.raises(UnsafeUrlError, match="169.254.169.254"):
            await _fetch_with_curl(
                "https://www.reddit.com/r/x/comments/abc/t/.json",
                resolve=_stub_resolver("169.254.169.254"),  # cloud metadata endpoint
            )
        assert not spawned

    @pytest.mark.asyncio
    async def test_split_answer_is_all_or_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One public + one private address is rejected, not filtered down."""
        from particles.extraction.reddit import _fetch_with_curl

        spawned = False

        async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            nonlocal spawned
            spawned = True
            return MagicMock()

        monkeypatch.setattr(
            "particles.extraction.reddit.asyncio.create_subprocess_exec", _fake_exec
        )
        with pytest.raises(UnsafeUrlError):
            await _fetch_with_curl(
                "https://www.reddit.com/r/x/comments/abc/t/.json",
                resolve=_stub_resolver(_PUBLIC_ADDR, "10.0.0.5"),
            )
        assert not spawned

    @pytest.mark.asyncio
    async def test_unresolvable_host_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.extraction.reddit import _fetch_with_curl

        spawned = False

        async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            nonlocal spawned
            spawned = True
            return MagicMock()

        monkeypatch.setattr(
            "particles.extraction.reddit.asyncio.create_subprocess_exec", _fake_exec
        )
        with pytest.raises(UnsafeUrlError, match="did not resolve"):
            await _fetch_with_curl(
                "https://www.reddit.com/r/x/comments/abc/t/.json",
                resolve=_stub_resolver(),  # empty answer
            )
        assert not spawned

    @pytest.mark.asyncio
    async def test_share_link_hop_is_pinned_independently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the one multi-hop flow pins each hop separately.

        ``--max-redirs 0`` makes "every hop" exactly one hop per fetch, and the
        share-link walk is two independent fetches, so per-hop parity with
        transport comes from composition rather than a transport.
        """
        from particles.extraction.reddit import _resolve_reddit_redirect

        headers = (
            b"HTTP/2 302 \r\n"
            b"location: https://www.reddit.com/r/AIEval/comments/1uf0fkd/slug/\r\n\r\n"
        )
        argv = _patch_curl(monkeypatch, stdout=headers)
        await _resolve_reddit_redirect(
            "https://old.reddit.com/r/AIEval/s/xyz", resolve=_stub_resolver(_PUBLIC_ADDR)
        )

        # Hop 1 pins the share link's own host...
        assert argv[argv.index("--resolve") + 1] == f"old.reddit.com:443:{_PUBLIC_ADDR}"
        # ...and the pin composes with the pre-existing no-redirect flags
        # rather than replacing them (stay asserted).
        assert argv[argv.index("--max-redirs") + 1] == "0"
        assert argv[argv.index("--proto") + 1] == "=https"

    @pytest.mark.asyncio
    async def test_share_link_blocked_address_never_spawns_curl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.extraction.reddit import _resolve_reddit_redirect

        argv = _patch_curl(monkeypatch, stdout=b"")
        with pytest.raises(UnsafeUrlError):
            await _resolve_reddit_redirect(
                "https://www.reddit.com/r/AIEval/s/xyz", resolve=_stub_resolver("127.0.0.1")
            )
        assert argv == []


class TestParsing:
    def test_get_post_returns_dict(self) -> None:
        post = _get_post(_REDDIT_BLOB)
        assert post is not None
        assert post["title"] == _POST_DATA["title"]
        assert post["author"] == "throwaway_investor"

    def test_get_post_bad_input_returns_none(self) -> None:
        assert _get_post([]) is None
        assert _get_post(None) is None
        assert _get_post({}) is None

    def test_qualifying_comments_filter_deleted(self) -> None:
        comments = _get_all_qualifying_comments(_REDDIT_BLOB)
        bodies = [c["body"] for c in comments]
        assert "[deleted]" not in bodies

    def test_qualifying_comments_filter_low_score(self) -> None:
        comments = _get_all_qualifying_comments(_REDDIT_BLOB)
        assert all(c["score"] >= 2 for c in comments)

    def test_qualifying_comments_sorted_by_score_desc(self) -> None:
        comments = _get_all_qualifying_comments(_REDDIT_BLOB)
        scores = [c["score"] for c in comments]
        assert scores == sorted(scores, reverse=True)

    def test_qualifying_comments_walks_full_tree(self) -> None:
        """BFS through `replies` captures comments at arbitrary depth — not
        just top-level + first-replies. Even when a parent has a low score
        (and is dropped), its replies are still considered.
        """
        nested = [
            {"data": {"children": [{"kind": "t3", "data": _POST_DATA}]}},
            {
                "data": {
                    "children": [
                        {
                            "kind": "t1",
                            "data": {
                                "author": "shallow",
                                "score": 5,
                                "body": "Top-level take.",
                                "replies": {
                                    "data": {
                                        "children": [
                                            {
                                                "kind": "t1",
                                                "data": {
                                                    "author": "deep_1",
                                                    "score": 15,
                                                    "body": "Depth-1 follow-up.",
                                                    "replies": {
                                                        "data": {
                                                            "children": [
                                                                {
                                                                    "kind": "t1",
                                                                    "data": {
                                                                        "author": "deep_2",
                                                                        "score": 30,
                                                                        "body": (
                                                                            "Depth-2 killer reply."
                                                                        ),
                                                                        "replies": "",
                                                                    },
                                                                }
                                                            ]
                                                        }
                                                    },
                                                },
                                            }
                                        ]
                                    }
                                },
                            },
                        },
                        {
                            "kind": "t1",
                            "data": {
                                "author": "low_parent",
                                "score": 1,  # below min_comment_score → dropped
                                "body": "meh",
                                "replies": {
                                    "data": {
                                        "children": [
                                            {
                                                "kind": "t1",
                                                "data": {
                                                    "author": "high_child",
                                                    "score": 50,
                                                    "body": "But the reply was insightful.",
                                                    "replies": "",
                                                },
                                            }
                                        ]
                                    }
                                },
                            },
                        },
                    ]
                }
            },
        ]
        comments = _get_all_qualifying_comments(nested)
        authors = [c["author"] for c in comments]
        # depth-2 reply was captured; low-scoring parent was dropped but
        # its high-scoring child still surfaced
        assert "shallow" in authors
        assert "deep_1" in authors
        assert "deep_2" in authors
        assert "low_parent" not in authors
        assert "high_child" in authors
        # Sort order is by score desc across all depths
        scores = [c["score"] for c in comments]
        assert scores == sorted(scores, reverse=True)

    def test_qualifying_comments_ignores_more_kind(self) -> None:
        """Reddit's `kind: "more"` continuation pointers must be skipped."""
        blob = [
            {"data": {"children": [{"kind": "t3", "data": _POST_DATA}]}},
            {
                "data": {
                    "children": [
                        {"kind": "more", "data": {"count": 100}},
                        {
                            "kind": "t1",
                            "data": {
                                "author": "real_user",
                                "score": 10,
                                "body": "actual comment",
                                "replies": "",
                            },
                        },
                    ]
                }
            },
        ]
        comments = _get_all_qualifying_comments(blob)
        assert len(comments) == 1
        assert comments[0]["author"] == "real_user"

    def test_render_body_includes_title_and_subreddit(self) -> None:
        post = _get_post(_REDDIT_BLOB)
        assert post is not None
        body_text = _render_reddit_body(post)
        assert "REDDIT THREAD: r/POETTechnologiesInc" in body_text
        assert "POET -30%" in body_text
        # The body chunk excludes comments — that's part of the carry-forward
        # design: an unchanged post body produces an unchanged body chunk
        # hash even when comments change.
        assert "TOP COMMENTS:" not in body_text

    def test_comment_lines_truncate_to_body_limit(self) -> None:
        long_comment = [{"author": "user", "score": 10, "body": "x" * 1500}]
        lines = _render_reddit_comment_lines(long_comment, body_limit=1000)
        assert len(lines) == 1
        assert "x" * 1000 in lines[0]
        assert "x" * 1001 not in lines[0]

    def test_build_chunks_splits_body_and_comments(self) -> None:
        post = _get_post(_REDDIT_BLOB)
        assert post is not None
        comments = _get_all_qualifying_comments(_REDDIT_BLOB)
        chunks = _build_reddit_chunks(
            post=post,
            comments=comments,
            body_limit=1000,
            single_call_threshold=30000,
            chunk_chars=10000,
        )
        # Body always its own chunk, then one or more comment chunks
        assert chunks[0].chunk_id == "body"
        assert any(c.chunk_id.startswith("comments") for c in chunks[1:])
        # Body chunk contains the title; comment chunks contain user mentions
        assert "POET -30%" in chunks[0].chunk_text
        assert all("TOP COMMENTS" not in chunks[0].chunk_text for _ in [chunks[0]])
        assert any("u/" in c.chunk_text for c in chunks[1:])


# ---------------------------------------------------------------------------
# Ticker inference
# ---------------------------------------------------------------------------


class TestInferTicker:
    def test_infers_ticker_from_title(self) -> None:
        assert _infer_ticker("POET -30% premarket after Marvell cancels partnership") == "POET"

    def test_skips_stop_words(self) -> None:
        # CEO and SEC are in the stop list
        assert _infer_ticker("CEO of SEC files against AFRM") == "AFRM"

    def test_returns_none_when_no_ticker(self) -> None:
        assert _infer_ticker("a thread with no uppercase ticker") is None

    def test_ignores_long_tokens(self) -> None:
        # 6-char tokens should not match the 2–5 letter range
        assert _infer_ticker("TOOLONG is not a ticker") is None


# ---------------------------------------------------------------------------
# Extractor: accepts()
# ---------------------------------------------------------------------------


class TestRedditExtractor:
    def test_accepts_reddit_post(self) -> None:
        assert RedditExtractor().accepts("REDDIT_POST")

    def test_rejects_other_source_types(self) -> None:
        assert not RedditExtractor().accepts("WEB_PAGE")
        assert not RedditExtractor().accepts("NUMISTA_API_COIN")

    @pytest.mark.asyncio
    async def test_extract_injects_subjects(self) -> None:
        """Both subreddit and inferred ticker are injected on every candidate."""
        content = json.dumps(_REDDIT_BLOB).encode()
        snapshot = _make_snapshot()

        # Use side_effect so each chunk call returns a fresh candidate;
        # the extractor produces 2 chunks (body + comments) and we want to
        # verify subject injection lands on both.
        def make_candidate() -> MagicMock:
            c = MagicMock()
            c.subjects = []
            c.content = "Marvell cancelled the deal."
            c.confidence_value = 0.8
            c.uncertainty_nature = UncertaintyNature.EPISTEMIC
            return c

        async def fake_llm(_text: str) -> tuple[list[MagicMock], list[str], bool]:
            return ([make_candidate()], [], False)

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=fake_llm),
        ):
            result = await RedditExtractor().extract(snapshot, content)

        assert len(result.candidates) >= 1
        # Every candidate has the subreddit and ticker subjects appended
        for c in result.candidates:
            assert "r/POETTechnologiesInc" in c.subjects
            assert "POET" in c.subjects

    @pytest.mark.asyncio
    async def test_extract_returns_quality_note_on_bad_json(self) -> None:
        result = await RedditExtractor().extract(_make_snapshot(), b"not json")
        assert result.candidates == []
        assert any("JSON parse error" in n for n in result.quality_notes)

    @pytest.mark.asyncio
    async def test_extract_returns_quality_note_on_missing_post(self) -> None:
        result = await RedditExtractor().extract(_make_snapshot(), b"[]")
        assert result.candidates == []
        assert any("post data" in n for n in result.quality_notes)


# ---------------------------------------------------------------------------
# Subject canonicalisation
# ---------------------------------------------------------------------------


class TestRewriteRedditSubjects:
    """The LLM is shown `u/{author}: <body>` and usually emits `u/{author}`
    as a subject. When it strips the prefix, the bare name ends up at the
    Obsidian vault root instead of `reddit.com/u/{author}.md`. The
    `_rewrite_reddit_subjects` post-step recovers the prefix by matching
    bare names against `u/`/`r/` tokens seen in the source chunks."""

    def _candidate(self, subjects: list[str]) -> MagicMock:
        c = MagicMock()
        c.subjects = list(subjects)
        return c

    def _chunk(self, text: str) -> object:
        from particles.extraction.incremental import ChunkUnit

        return ChunkUnit(chunk_id="c0", chunk_text=text)

    def test_bare_username_in_chunks_gets_prefixed(self) -> None:
        from particles.extraction.reddit import _rewrite_reddit_subjects

        chunks = [self._chunk("[score:5] u/ExtremeAddict: nice trade")]
        candidates = [self._candidate(["ExtremeAddict", "POET"])]
        _rewrite_reddit_subjects(candidates, chunks)
        assert "u/ExtremeAddict" in candidates[0].subjects
        assert "ExtremeAddict" not in candidates[0].subjects
        # Other (non-user) subjects pass through.
        assert "POET" in candidates[0].subjects

    def test_already_prefixed_passes_through(self) -> None:
        from particles.extraction.reddit import _rewrite_reddit_subjects

        chunks = [self._chunk("u/Alice posted")]
        candidates = [self._candidate(["u/Alice"])]
        _rewrite_reddit_subjects(candidates, chunks)
        assert candidates[0].subjects == ["u/Alice"]

    def test_bare_subreddit_in_chunks_gets_prefixed(self) -> None:
        from particles.extraction.reddit import _rewrite_reddit_subjects

        chunks = [self._chunk("posted in r/scifi about books")]
        candidates = [self._candidate(["scifi"])]
        _rewrite_reddit_subjects(candidates, chunks)
        assert candidates[0].subjects == ["r/scifi"]

    def test_bare_name_not_in_chunks_passes_through(self) -> None:
        """If the LLM emits a bare name that isn't a known username, leave it
        alone — it might be a real-world entity mentioned in a comment."""
        from particles.extraction.reddit import _rewrite_reddit_subjects

        chunks = [self._chunk("u/Alice talked about Einstein")]
        candidates = [self._candidate(["Einstein"])]
        _rewrite_reddit_subjects(candidates, chunks)
        # Einstein wasn't seen as u/Einstein, so no prefix added.
        assert candidates[0].subjects == ["Einstein"]

    def test_deduplicates_after_rewriting(self) -> None:
        """If the LLM emits both `ExtremeAddict` and `u/ExtremeAddict`, the
        rewrite collapses them to a single canonical entry."""
        from particles.extraction.reddit import _rewrite_reddit_subjects

        chunks = [self._chunk("u/ExtremeAddict: hello")]
        candidates = [self._candidate(["ExtremeAddict", "u/ExtremeAddict"])]
        _rewrite_reddit_subjects(candidates, chunks)
        assert candidates[0].subjects == ["u/ExtremeAddict"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_reddit_extractor_in_registry(self) -> None:
        from particles.extraction.registry import get_extractors

        ids = [p.EXTRACTOR_ID for p in get_extractors()]
        assert "reddit-extractor" in ids

    def test_reddit_importer_in_registry(self) -> None:
        from particles.ingest.importers.registry import get_importers

        importers = get_importers()
        assert any(isinstance(d, RedditImporter) for d in importers)

    def test_reddit_extractor_before_general(self) -> None:
        from particles.extraction.general import GeneralExtractor
        from particles.extraction.registry import get_extractors

        plugins = get_extractors()
        reddit_idx = next(i for i, p in enumerate(plugins) if p.EXTRACTOR_ID == "reddit-extractor")
        general_idx = next(i for i, p in enumerate(plugins) if isinstance(p, GeneralExtractor))
        assert reddit_idx < general_idx

    def test_ensure_extractor_records_includes_reddit(self) -> None:
        # Just verify it can be called without error via the in-process registry
        plugins = __import__(
            "particles.extraction.registry", fromlist=["get_extractors"]
        ).get_extractors()
        ids = [p.EXTRACTOR_ID for p in plugins]
        assert "reddit-extractor" in ids
