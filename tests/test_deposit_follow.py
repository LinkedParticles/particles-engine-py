"""Integration tests — deposit-time primary-URL follow.

The per-importer parser tests (each importer's ``primary_url`` method)
live in ``tests/test_reddit.py``, ``tests/test_hackernews_extractor.py``,
and ``tests/test_mastodon_extractor.py``. This file covers the
*orchestration*: the follow algorithm in
``particles/corpus/deposit.py::_maybe_follow_primary_url``, the
``--follow-post-links`` / ``--follow-comment-links`` flag plumbing, and
the ``corpus_follow_edges`` join-table helpers.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.corpus.follow_edges import (
    LINK_TYPE_COMMENT,
    LINK_TYPE_POST,
    add_follow_edge,
    get_follow_sources,
    get_follow_targets,
)

# ---------------------------------------------------------------------------
# Helper: a Reddit blob whose post URL is an external article
# ---------------------------------------------------------------------------


def _reddit_link_post_blob(*, url: str) -> bytes:
    """A minimal Reddit-API-shape blob with a link-post pointing at ``url``."""
    return json.dumps(
        [
            {
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "title": "external article",
                                "url": url,
                                "is_self": False,
                                "selftext": "",
                                "author": "someone",
                            },
                        }
                    ]
                }
            },
            {"data": {"children": []}},
        ]
    ).encode("utf-8")


async def _aiter_bytes(chunks: list[bytes]) -> Any:
    for chunk in chunks:
        yield chunk


def _make_http_response(content: bytes, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.headers = {"content-type": "text/html"}
    resp.raise_for_status = MagicMock()
    # The generic (no-importer) deposit path streams via get_capped →
    # client.stream(...).aiter_bytes(); expose a one-chunk stream.
    resp.aiter_bytes = lambda: _aiter_bytes([content])
    return resp


def _stream_cm(resp: MagicMock) -> AsyncMock:
    """An async-context-manager mock yielding ``resp``, for client.stream()."""
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# corpus_follow_edges ORM helpers
# ---------------------------------------------------------------------------


class TestCorpusFollowEdges:
    @pytest.mark.asyncio
    async def test_add_and_retrieve_edge(self, db_session: Any) -> None:
        await add_follow_edge(
            db_session,
            via_entry_id="reddit-1",
            target_entry_id="article-A",
        )
        await db_session.commit()
        targets = await get_follow_targets(db_session, "reddit-1")
        assert len(targets) == 1
        assert targets[0].target_entry_id == "article-A"
        assert targets[0].link_type == LINK_TYPE_POST

    @pytest.mark.asyncio
    async def test_idempotent_on_duplicate(self, db_session: Any) -> None:
        """Re-depositing a Reddit URL must not produce two edge rows."""
        await add_follow_edge(db_session, via_entry_id="reddit-1", target_entry_id="article-A")
        await add_follow_edge(db_session, via_entry_id="reddit-1", target_entry_id="article-A")
        await db_session.commit()
        targets = await get_follow_targets(db_session, "reddit-1")
        assert len(targets) == 1

    @pytest.mark.asyncio
    async def test_fan_in_many_sources_one_target(self, db_session: Any) -> None:
        """The press-release / viral-link case: one article reached from
        many sources writes N rows in the join table."""
        for via in ("reddit-1", "hn-1", "mastodon-1"):
            await add_follow_edge(db_session, via_entry_id=via, target_entry_id="press-release-X")
        await db_session.commit()
        sources = await get_follow_sources(db_session, "press-release-X")
        assert len(sources) == 3
        via_ids = {row.via_entry_id for row in sources}
        assert via_ids == {"reddit-1", "hn-1", "mastodon-1"}

    @pytest.mark.asyncio
    async def test_link_type_dimension(self, db_session: Any) -> None:
        """``(via, target, link_type)`` is the PK — POST_LINK and the
        deferred COMMENT_LINK can coexist for the same edge."""
        await add_follow_edge(
            db_session,
            via_entry_id="reddit-1",
            target_entry_id="article-A",
            link_type=LINK_TYPE_POST,
        )
        await add_follow_edge(
            db_session,
            via_entry_id="reddit-1",
            target_entry_id="article-A",
            link_type=LINK_TYPE_COMMENT,
        )
        await db_session.commit()
        targets = await get_follow_targets(db_session, "reddit-1")
        assert len(targets) == 2
        kinds = {row.link_type for row in targets}
        assert kinds == {LINK_TYPE_POST, LINK_TYPE_COMMENT}


# ---------------------------------------------------------------------------
# deposit_url → follow → recursive deposit → edge written
# ---------------------------------------------------------------------------


class TestDepositFollowIntegration:
    """End-to-end: deposit a Reddit link-post and verify the follow
    machinery deposits the article URL and writes the edge."""

    @pytest.mark.asyncio
    async def test_reddit_link_post_triggers_follow(self, db_session: Any) -> None:
        from particles.corpus.deposit import deposit_url

        article_url = "https://www.theatlantic.com/article-x"
        reddit_url = "https://www.reddit.com/r/news/comments/abc123/title/"
        reddit_blob = _reddit_link_post_blob(url=article_url)
        article_html = b"<html><body><p>article body</p></body></html>"
        follow_targets: list[tuple[str, str, str]] = []

        with (
            patch(
                "particles.ingest.importers.reddit._fetch_with_curl",
                AsyncMock(return_value=reddit_blob),
            ),
            patch("particles.corpus.deposit.particles_client") as mock_client_ctx,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            article_resp = _make_http_response(article_html)
            mock_client.get = AsyncMock(return_value=article_resp)
            mock_client.stream = MagicMock(return_value=_stream_cm(article_resp))
            mock_client_ctx.return_value = mock_client

            primary_entry_id, _ = await deposit_url(
                db_session,
                reddit_url,
                deposited_by="test",
                follow_post_links=True,
                out_follow_targets=follow_targets,
            )
            await db_session.commit()

        # The follow recursively deposited the article URL — a second
        # corpus entry should now exist for the article.
        from particles.corpus.store import get_entry_by_uri

        article_entry = await get_entry_by_uri(db_session, article_url)
        assert article_entry is not None, "follow must have deposited the article"

        # And the join-table edge records the relationship.
        targets = await get_follow_targets(db_session, primary_entry_id)
        assert len(targets) == 1
        assert targets[0].target_entry_id == article_entry.entry_id
        assert targets[0].link_type == LINK_TYPE_POST

        # The out-parameter (0.36.1 ergonomics fix) surfaces the
        # follow target so the deposit CLI can print it without an
        # extra query.
        assert len(follow_targets) == 1
        target_entry_id, _target_snap_id, target_uri = follow_targets[0]
        assert target_entry_id == article_entry.entry_id
        assert target_uri == article_url

    @pytest.mark.asyncio
    async def test_explicit_no_follow_overrides_default(self, db_session: Any) -> None:
        """``--no-follow-post-links`` wins over Reddit's
        ``DEFAULT_FOLLOW_POST_LINKS=True``."""
        from particles.corpus.deposit import deposit_url

        reddit_url = "https://www.reddit.com/r/news/comments/abc123/title/"
        reddit_blob = _reddit_link_post_blob(url="https://example.com/article")

        with (
            patch(
                "particles.ingest.importers.reddit._fetch_with_curl",
                AsyncMock(return_value=reddit_blob),
            ),
            patch("particles.corpus.deposit.particles_client") as mock_client_ctx,
        ):
            # The HTTP client should NEVER be invoked — the follow is
            # disabled, so no secondary fetch.
            mock_client_ctx.side_effect = AssertionError("follow should not fire")
            primary_entry_id, _ = await deposit_url(
                db_session,
                reddit_url,
                deposited_by="test",
                follow_post_links=False,
            )
            await db_session.commit()

        # No edge written.
        targets = await get_follow_targets(db_session, primary_entry_id)
        assert targets == []

    @pytest.mark.asyncio
    async def test_self_post_no_follow(self, db_session: Any) -> None:
        """Reddit self-post → ``primary_url`` returns None → no follow,
        no edge. The primary deposit still succeeds."""
        from particles.corpus.deposit import deposit_url

        self_post_blob = json.dumps(
            [
                {
                    "data": {
                        "children": [
                            {
                                "kind": "t3",
                                "data": {
                                    "title": "discussion thread",
                                    "is_self": True,
                                    "selftext": "What do you think about X?",
                                    "url": "https://www.reddit.com/r/foo/comments/abc/title/",
                                    "author": "someone",
                                },
                            }
                        ]
                    }
                },
                {"data": {"children": []}},
            ]
        ).encode("utf-8")

        with patch(
            "particles.ingest.importers.reddit._fetch_with_curl",
            AsyncMock(return_value=self_post_blob),
        ):
            primary_entry_id, _ = await deposit_url(
                db_session,
                "https://www.reddit.com/r/foo/comments/abc/title/",
                deposited_by="test",
                follow_post_links=True,
            )
            await db_session.commit()

        assert await get_follow_targets(db_session, primary_entry_id) == []

    @pytest.mark.asyncio
    async def test_paywall_on_follow_logs_warning_does_not_fail_primary(
        self, db_session: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If the article URL returns 403, the primary Reddit deposit
        still succeeds, no edge is written, and a WARNING is logged."""
        import logging

        from particles.corpus.deposit import deposit_url

        reddit_url = "https://www.reddit.com/r/news/comments/abc123/title/"
        reddit_blob = _reddit_link_post_blob(url="https://paywalled-news.com/x")

        with (
            patch(
                "particles.ingest.importers.reddit._fetch_with_curl",
                AsyncMock(return_value=reddit_blob),
            ),
            patch("particles.corpus.deposit.particles_client") as mock_client_ctx,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            forbidden_resp = _make_http_response(b"forbidden", status_code=403)
            mock_client.get = AsyncMock(return_value=forbidden_resp)
            mock_client.stream = MagicMock(return_value=_stream_cm(forbidden_resp))
            mock_client_ctx.return_value = mock_client

            caplog.set_level(logging.WARNING, logger="particles.corpus.deposit")
            primary_entry_id, _ = await deposit_url(
                db_session,
                reddit_url,
                deposited_by="test",
                follow_post_links=True,
            )
            await db_session.commit()

        # Primary deposit survived.
        assert primary_entry_id is not None
        # No edge — the follow failed.
        assert await get_follow_targets(db_session, primary_entry_id) == []
        # Operator-facing warning fired.
        assert any("Follow failed" in rec.getMessage() for rec in caplog.records), (
            "expected Follow-failed warning on paywall"
        )

    @pytest.mark.asyncio
    async def test_follow_comment_links_true_warns_and_noops(
        self, db_session: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``--follow-comment-links`` is reserved-but-deferred: passing True emits a warning and proceeds as if False."""
        import logging

        from particles.corpus.deposit import deposit_url

        reddit_url = "https://www.reddit.com/r/news/comments/abc123/title/"
        reddit_blob = _reddit_link_post_blob(url="https://example.com/a")

        with (
            patch(
                "particles.ingest.importers.reddit._fetch_with_curl",
                AsyncMock(return_value=reddit_blob),
            ),
            patch("particles.corpus.deposit.particles_client") as mock_client_ctx,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            x_resp = _make_http_response(b"<html>x</html>")
            mock_client.get = AsyncMock(return_value=x_resp)
            mock_client.stream = MagicMock(return_value=_stream_cm(x_resp))
            mock_client_ctx.return_value = mock_client

            caplog.set_level(logging.WARNING, logger="particles.corpus.deposit")
            await deposit_url(
                db_session,
                reddit_url,
                deposited_by="test",
                # Post-link follow OFF — we're testing the comment-link
                # warning in isolation.
                follow_post_links=False,
                follow_comment_links=True,
            )
            await db_session.commit()

        assert any(
            "comment-link following is deferred" in rec.getMessage() for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_unparticipating_importer_no_follow(self, db_session: Any) -> None:
        """An importer without ``primary_url`` (or with the method
        absent) doesn't participate in following — even with
        ``--follow-post-links``. We exercise this via the generic
        fallback path: a URL no registered importer accepts."""
        from particles.corpus.deposit import deposit_url

        with patch("particles.corpus.deposit.particles_client") as mock_client_ctx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            generic_resp = _make_http_response(b"<html>generic</html>")
            mock_client.get = AsyncMock(return_value=generic_resp)
            mock_client.stream = MagicMock(return_value=_stream_cm(generic_resp))
            mock_client_ctx.return_value = mock_client

            entry_id, _ = await deposit_url(
                db_session,
                # example.com is IANA's documentation domain — resolves
                # to a real IP so the SSRF guard accepts it. The fetch
                # itself is mocked.
                "https://example.com/page",
                deposited_by="test",
                follow_post_links=True,  # explicit, but irrelevant
            )
            await db_session.commit()

        # Generic fallback path — no importer matched, so no follow.
        assert await get_follow_targets(db_session, entry_id) == []


class TestMentionReconciliation:
    """depositing a previously-cited URL binds its mentions and
    writes COMMENT_LINK follow edges from each citing source."""

    async def test_reconcile_binds_and_writes_comment_edges(self, db_session: Any) -> None:
        from particles.corpus.deposit import _reconcile_mentions_for_deposit
        from particles.store.url_mention_store import (
            list_undeposited_mentions,
            record_url_mentions,
        )

        url = "https://press.example/release"
        await record_url_mentions(db_session, source_entry_id="reddit-1", canonical_urls=[url])
        await record_url_mentions(db_session, source_entry_id="hn-2", canonical_urls=[url])

        await _reconcile_mentions_for_deposit(db_session, url=url, entry_id="press-entry")
        await db_session.commit()

        # Mentions are now bound → no longer undeposited.
        assert await list_undeposited_mentions(db_session) == []
        # Each citing source gained a COMMENT_LINK edge into the new entry.
        sources = await get_follow_sources(db_session, "press-entry")
        assert {s.via_entry_id for s in sources} == {"reddit-1", "hn-2"}
        assert all(s.link_type == LINK_TYPE_COMMENT for s in sources)

    async def test_reconcile_is_idempotent(self, db_session: Any) -> None:
        from particles.corpus.deposit import _reconcile_mentions_for_deposit
        from particles.store.url_mention_store import record_url_mentions

        url = "https://press.example/release"
        await record_url_mentions(db_session, source_entry_id="reddit-1", canonical_urls=[url])
        await _reconcile_mentions_for_deposit(db_session, url=url, entry_id="press-entry")
        await _reconcile_mentions_for_deposit(db_session, url=url, entry_id="press-entry")
        await db_session.commit()
        sources = await get_follow_sources(db_session, "press-entry")
        assert len(sources) == 1  # no duplicate edge

    async def test_reconcile_skips_self_citation(self, db_session: Any) -> None:
        from particles.corpus.deposit import _reconcile_mentions_for_deposit
        from particles.store.url_mention_store import record_url_mentions

        url = "https://press.example/release"
        # The entry being deposited had itself cited the URL (self-reference).
        await record_url_mentions(db_session, source_entry_id="press-entry", canonical_urls=[url])
        await _reconcile_mentions_for_deposit(db_session, url=url, entry_id="press-entry")
        await db_session.commit()
        assert await get_follow_sources(db_session, "press-entry") == []
