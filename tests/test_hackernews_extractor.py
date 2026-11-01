# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hacker News extractor and importer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from particles.core.schema import Snapshot, UncertaintyNature
from particles.extraction.hackernews import (
    APPLICABILITY,
    DEFAULT_TRUST_WEIGHT,
    EXTRACTOR_ID,
    SOURCE_TYPE,
    HackerNewsExtractor,
    _build_hn_chunks,
    _build_story_meta_candidate,
    _fetch_item,
    _get_comments_map,
    _get_story,
    _parse_item_id,
    _render_hn_story_body,
    _rewrite_hn_subjects,
    _walk_comments_dfs,
)
from particles.extraction.hackernews import _fetch_thread as hn_fetch_thread
from particles.ingest.importers.hackernews import HackerNewsImporter
from tests._capped_http import set_capped_responses

# ---------------------------------------------------------------------------
# Fixtures — realistic HN Firebase API blob shape
# ---------------------------------------------------------------------------

_STORY = {
    "id": 12345678,
    "by": "patio11",
    "title": "Show HN: A new tool for structured knowledge",
    "score": 256,
    "descendants": 5,
    "time": 1735689600,  # 2025-01-01 00:00 UTC
    "url": "https://example.com/tool",
    "type": "story",
    "kids": [12345679, 12345680],
    "text": "I built this because I needed structured belief management.",
}

_COMMENTS = {
    "12345679": {
        "id": 12345679,
        "by": "alice",
        "parent": 12345678,
        "text": "Nice work. Reminds me of doxastic logic.",
        "kids": [12345681],
        "time": 1735689700,
    },
    "12345680": {
        "id": 12345680,
        "by": "bob",
        "parent": 12345678,
        "text": "How does this compare to <i>existing knowledge graphs</i>?",
        "kids": [],
        "time": 1735689800,
    },
    "12345681": {
        "id": 12345681,
        "by": "alice",  # alice replies to her own comment via a fork
        "parent": 12345679,
        "text": "The carry-forward optimisation is the interesting bit.",
        "kids": [],
        "time": 1735689900,
    },
    "12345682": {
        "id": 12345682,
        "by": "deleted_one",
        "parent": 12345678,
        "deleted": True,
        "text": "",
        "kids": [],
    },
    "12345683": {
        "id": 12345683,
        "by": "dead_one",
        "parent": 12345678,
        "dead": True,
        "text": "spam content",
        "kids": [],
    },
}

_HN_BLOB = {"story": _STORY, "comments": _COMMENTS}


def _make_snapshot() -> Snapshot:
    from datetime import UTC, datetime

    from particles.core.schema import ExtractionStatus, WarcRecordType

    return Snapshot(
        snapshot_id="test-snap",
        captured_at=datetime.now(UTC),
        content_hash="abc",
        extraction_status=ExtractionStatus.PENDING,
        warc_record_type=WarcRecordType.RESPONSE,
        author_id="hn:patio11",
        content_published_at=datetime.fromtimestamp(1735689600, tz=UTC),
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_source_type(self) -> None:
        assert SOURCE_TYPE == "HACKERNEWS_THREAD"

    def test_extractor_id(self) -> None:
        assert EXTRACTOR_ID == "hackernews-extractor"

    def test_default_trust_weight(self) -> None:
        assert pytest.approx(0.50) == DEFAULT_TRUST_WEIGHT

    def test_applicability_must_hacker_news(self) -> None:
        assert len(APPLICABILITY) == 1
        clause = APPLICABILITY[0]
        assert clause.keyword == "MUST"
        assert clause.domain_label == "Hacker News"
        assert "HACKERNEWS_THREAD" in clause.source_types

    def test_infer_domain_returns_hacker_news(self) -> None:
        from particles.extraction.registry import infer_domain

        assert infer_domain("HACKERNEWS_THREAD") == "Hacker News"


# ---------------------------------------------------------------------------
# Importer URL matching
# ---------------------------------------------------------------------------


class TestHackerNewsImporter:
    def test_accepts_item_url(self) -> None:
        d = HackerNewsImporter()
        assert d.accepts_url("https://news.ycombinator.com/item?id=12345")

    def test_accepts_item_url_with_extra_params(self) -> None:
        d = HackerNewsImporter()
        assert d.accepts_url("https://news.ycombinator.com/item?id=12345&p=2")
        assert d.accepts_url("https://news.ycombinator.com/item?p=2&id=12345")

    def test_accepts_raw_api_url(self) -> None:
        d = HackerNewsImporter()
        assert d.accepts_url("https://hacker-news.firebaseio.com/v0/item/12345.json")

    def test_rejects_bare_hn_root(self) -> None:
        """Bare host without an item must NOT match — there's no thread to fetch."""
        d = HackerNewsImporter()
        assert not d.accepts_url("https://news.ycombinator.com")
        assert not d.accepts_url("https://news.ycombinator.com/")
        assert not d.accepts_url("https://news.ycombinator.com/newest")

    def test_rejects_unrelated_url(self) -> None:
        d = HackerNewsImporter()
        assert not d.accepts_url("https://reddit.com/r/programming/comments/abc/x/")
        assert not d.accepts_url("https://example.com/item?id=12345")

    def test_parse_item_id_returns_int(self) -> None:
        assert _parse_item_id("https://news.ycombinator.com/item?id=999") == 999
        assert _parse_item_id("https://hacker-news.firebaseio.com/v0/item/42.json") == 42
        assert _parse_item_id("https://reddit.com/r/x/") is None

    # ----- primary_url -----

    def test_primary_url_story_returns_url(self) -> None:
        """HN story with a ``url`` field — the link-shaped case."""
        blob = {
            "story": {
                "id": 48278374,
                "title": "Some interesting article",
                "url": "https://example.com/article-x",
                "by": "dang",
            },
            "comments": {},
        }
        assert (
            HackerNewsImporter().primary_url(json.dumps(blob).encode())
            == "https://example.com/article-x"
        )

    def test_primary_url_ask_hn_returns_none(self) -> None:
        """Ask-HN posts have no ``url`` field — text-only submission."""
        blob = {
            "story": {
                "id": 12345,
                "title": "Ask HN: how do you handle X?",
                "text": "I'm wondering how others approach X...",
                "by": "curious_dev",
            },
            "comments": {},
        }
        assert HackerNewsImporter().primary_url(json.dumps(blob).encode()) is None

    def test_primary_url_malformed_returns_none(self) -> None:
        d = HackerNewsImporter()
        assert d.primary_url(b"not json") is None
        assert d.primary_url(b"{}") is None  # no story key
        assert d.primary_url(b'{"story": null}') is None

    def test_default_follow_post_links_true(self) -> None:
        assert HackerNewsImporter.DEFAULT_FOLLOW_POST_LINKS is True
        assert HackerNewsImporter.DEFAULT_FOLLOW_COMMENT_LINKS is False

    @pytest.mark.asyncio
    async def test_deposit_fetches_via_firebase_and_stores_blob(self, db_session: object) -> None:
        """End-to-end deposit: importer walks ``kids`` and writes a single blob.

        Patches ``particles.http.particles_client`` per ``tests/AGENTS.md``
        because the importer does ``from particles.http import particles_client``
        inside its ``deposit()`` body (deferred import).
        """

        # The Firebase API returns one item per HTTP call. Set up a router
        # mock that responds based on the requested URL.
        def make_response(payload: object) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(return_value=payload)
            resp.raise_for_status = MagicMock()
            return resp

        def fake_get(url: str) -> MagicMock:
            if url.endswith("/item/12345678.json"):
                return make_response(_STORY)
            for cid, comment in _COMMENTS.items():
                if url.endswith(f"/item/{cid}.json"):
                    return make_response(comment)
            return make_response(None)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, router=fake_get)

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            entry_id, snapshot_id = await HackerNewsImporter().deposit(
                db_session,  # type: ignore[arg-type]
                "https://news.ycombinator.com/item?id=12345678",
                "test-operator",
                [],
            )

        from particles.corpus.store import get_entry, get_snapshot

        entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
        assert entry is not None
        assert entry.source_type == "HACKERNEWS_THREAD"
        # The deposited URL canonicalises to the human item form.
        assert entry.uri_r == "https://news.ycombinator.com/item?id=12345678"

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        assert snap.author_id == "hn:patio11"
        # content_published_at threaded through from the story time.
        assert snap.content_published_at is not None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


class TestParsing:
    def test_get_story_returns_dict(self) -> None:
        story = _get_story(_HN_BLOB)
        assert story is not None
        assert story["id"] == 12345678
        assert story["by"] == "patio11"

    def test_get_story_bad_input_returns_none(self) -> None:
        assert _get_story([]) is None
        assert _get_story(None) is None
        assert _get_story({}) is None
        assert _get_story({"story": "not-a-dict"}) is None

    def test_get_comments_map_returns_dict(self) -> None:
        m = _get_comments_map(_HN_BLOB)
        assert "12345679" in m
        assert m["12345679"]["by"] == "alice"

    def test_walk_comments_dfs_filters_deleted_and_dead(self) -> None:
        comments = _walk_comments_dfs(_STORY, _get_comments_map(_HN_BLOB), min_score=1)
        authors = [c.get("by") for _, c in comments]
        # alice and bob are surfaced; deleted_one and dead_one are dropped.
        assert "alice" in authors
        assert "bob" in authors
        assert "deleted_one" not in authors
        assert "dead_one" not in authors

    def test_walk_comments_dfs_depth_accounting(self) -> None:
        """Reply to a reply gets depth 1; top-level kids are depth 0."""
        comments = _walk_comments_dfs(_STORY, _get_comments_map(_HN_BLOB), min_score=1)
        # alice (top-level) → alice's reply (depth 1)
        depths_by_id = {c["id"]: d for d, c in comments}
        assert depths_by_id[12345679] == 0
        assert depths_by_id[12345681] == 1

    def test_render_story_body_includes_title_and_author(self) -> None:
        body = _render_hn_story_body(_STORY)
        assert "HACKER NEWS THREAD:" in body
        assert "Show HN" in body
        assert "hn/patio11" in body
        assert "[score: 256]" in body
        assert "External URL: https://example.com/tool" in body
        # Body chunk excludes comments — carry-forward design.
        assert "COMMENTS:" not in body

    def test_render_story_body_omits_url_when_missing(self) -> None:
        body = _render_hn_story_body({"title": "T", "by": "u", "score": 1, "text": ""})
        assert "External URL:" not in body

    def test_build_chunks_splits_story_and_comments(self) -> None:
        chunks = _build_hn_chunks(
            story=_STORY,
            comments_map=_get_comments_map(_HN_BLOB),
            indent_per_level=2,
            min_score=1,
            single_call_threshold=30000,
            chunk_chars=10000,
        )
        assert chunks[0].chunk_id == "story"
        assert any(c.chunk_id.startswith("comments") for c in chunks[1:])
        assert "Show HN" in chunks[0].chunk_text
        # Comment chunks include the indented hn/{author} lines.
        assert any("hn/alice" in c.chunk_text for c in chunks[1:])


# ---------------------------------------------------------------------------
# Story-meta particle (dual-emission)
# ---------------------------------------------------------------------------


class TestStoryMetaParticle:
    def test_dual_emission_keys_present(self) -> None:
        """The story-meta particle MUST emit both hn:* and social:* keys
        with the same underlying value."""
        particle = _build_story_meta_candidate(_STORY, injected_subjects=["hn/patio11"])
        assert particle is not None
        props = particle.properties
        assert props is not None
        # Hacker News-specific keys.
        assert props["hn:hasPoints"] == 256
        assert props["hn:hasAuthor"] == "patio11"
        assert props["hn:hasItemId"] == 12345678
        assert props["hn:hasCommentCount"] == 5
        # Cross-platform UGC keys — same values via the dual emission.
        assert props["social:hasScore"] == 256
        assert props["social:hasScore"] == props["hn:hasPoints"]
        assert props["social:hasReplyCount"] == 5
        assert props["social:hasReplyCount"] == props["hn:hasCommentCount"]
        assert props["social:hasAuthorHandle"] == "patio11"
        assert props["social:hasAuthorHandle"] == props["hn:hasAuthor"]
        # Generic content + thread structure.
        assert props["content:hasUrl"] == "https://example.com/tool"
        assert props["thread:hasRootId"] == 12345678
        assert props["thread:hasRootId"] == props["hn:hasItemId"]

    def test_property_keys_all_use_prefix_form(self) -> None:
        """every key MUST be ``prefix:LocalName``."""
        particle = _build_story_meta_candidate(_STORY, injected_subjects=[])
        assert particle is not None
        assert particle.properties is not None
        for key in particle.properties:
            assert ":" in key, f"key {key!r} violates the prefix:LocalName rule"

    def test_returns_none_for_malformed_story(self) -> None:
        # Missing id → not enough information for a meaningful meta record.
        assert _build_story_meta_candidate({}, []) is None
        assert _build_story_meta_candidate({"by": "u"}, []) is None
        assert _build_story_meta_candidate({"id": 1}, []) is None  # no author

    def test_summary_content_is_human_readable(self) -> None:
        particle = _build_story_meta_candidate(_STORY, injected_subjects=["hn/patio11"])
        assert particle is not None
        assert "Hacker News thread" in particle.content
        assert "256 points" in particle.content
        assert "5 comments" in particle.content
        assert particle.uncertainty_nature == UncertaintyNature.EPISTEMIC

    def test_injected_subjects_propagate(self) -> None:
        particle = _build_story_meta_candidate(_STORY, injected_subjects=["hn/patio11"])
        assert particle is not None
        assert "hn/patio11" in particle.subjects

    def test_handles_missing_url_as_none(self) -> None:
        no_url = {**_STORY, "url": None}
        particle = _build_story_meta_candidate(no_url, [])
        assert particle is not None
        assert particle.properties is not None
        assert particle.properties["content:hasUrl"] is None


# ---------------------------------------------------------------------------
# Extractor: accepts() + the synthesised meta particle
# ---------------------------------------------------------------------------


class TestHackerNewsExtractor:
    def test_accepts_hn_thread(self) -> None:
        assert HackerNewsExtractor().accepts("HACKERNEWS_THREAD")

    def test_rejects_other_source_types(self) -> None:
        assert not HackerNewsExtractor().accepts("REDDIT_POST")
        assert not HackerNewsExtractor().accepts("WEB_PAGE")
        assert not HackerNewsExtractor().accepts("NUMISTA_API_COIN")

    @pytest.mark.asyncio
    async def test_extract_prepends_story_meta_particle(self) -> None:
        """The first candidate is the synthesised story-meta particle —
        carrying ``properties``, ahead of any LLM-derived claims."""
        content = json.dumps(_HN_BLOB).encode()
        snapshot = _make_snapshot()

        def make_candidate() -> MagicMock:
            c = MagicMock()
            c.subjects = []
            c.content = "Alice praises the carry-forward design."
            c.confidence_value = 0.75
            c.uncertainty_nature = UncertaintyNature.EPISTEMIC
            c.properties = None  # explicit — distinguishes LLM from synth meta
            return c

        async def fake_llm(_text: str) -> tuple[list[MagicMock], list[str], bool]:
            return ([make_candidate()], [], False)

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=fake_llm),
        ):
            result = await HackerNewsExtractor().extract(snapshot, content)

        # At least 2 candidates: story-meta + ≥1 LLM. story-meta is first.
        assert len(result.candidates) >= 2
        first = result.candidates[0]
        assert first.properties is not None
        assert first.properties.get("hn:hasItemId") == 12345678
        assert first.properties.get("social:hasScore") == 256

    @pytest.mark.asyncio
    async def test_extract_injects_author_subject_on_llm_candidates(self) -> None:
        """LLM-derived candidates get the story author injected as a subject."""
        content = json.dumps(_HN_BLOB).encode()
        snapshot = _make_snapshot()

        def make_candidate() -> MagicMock:
            c = MagicMock()
            c.subjects = []
            c.content = "Alice praises the carry-forward design."
            c.confidence_value = 0.75
            c.uncertainty_nature = UncertaintyNature.EPISTEMIC
            c.properties = None  # explicit — distinguishes LLM from synth meta
            return c

        async def fake_llm(_text: str) -> tuple[list[MagicMock], list[str], bool]:
            return ([make_candidate()], [], False)

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=fake_llm),
        ):
            result = await HackerNewsExtractor().extract(snapshot, content)

        # Skip the story-meta particle (index 0) and inspect LLM-derived ones.
        llm_candidates = [c for c in result.candidates if c.properties is None]
        assert len(llm_candidates) >= 1
        for c in llm_candidates:
            assert "hn/patio11" in c.subjects

    @pytest.mark.asyncio
    async def test_extract_quality_note_on_bad_json(self) -> None:
        result = await HackerNewsExtractor().extract(_make_snapshot(), b"not json")
        assert result.candidates == []
        assert any("JSON parse error" in n for n in result.quality_notes)

    @pytest.mark.asyncio
    async def test_extract_quality_note_on_missing_story(self) -> None:
        result = await HackerNewsExtractor().extract(_make_snapshot(), b"{}")
        assert result.candidates == []
        assert any("story data" in n for n in result.quality_notes)


# ---------------------------------------------------------------------------
# Subject canonicalisation
# ---------------------------------------------------------------------------


class TestRewriteHnSubjects:
    def _candidate(self, subjects: list[str]) -> MagicMock:
        c = MagicMock()
        c.subjects = list(subjects)
        return c

    def _chunk(self, text: str) -> object:
        from particles.extraction.incremental import ChunkUnit

        return ChunkUnit(chunk_id="c0", chunk_text=text)

    def test_bare_handle_in_chunks_gets_prefixed(self) -> None:
        chunks = [self._chunk("  hn/alice: nice take")]
        candidates = [self._candidate(["alice", "doxastic logic"])]
        _rewrite_hn_subjects(candidates, chunks)  # type: ignore[arg-type]
        assert "hn/alice" in candidates[0].subjects
        assert "alice" not in candidates[0].subjects
        # Non-user subjects pass through.
        assert "doxastic logic" in candidates[0].subjects

    def test_already_prefixed_passes_through(self) -> None:
        chunks = [self._chunk("hn/alice said hi")]
        candidates = [self._candidate(["hn/alice"])]
        _rewrite_hn_subjects(candidates, chunks)  # type: ignore[arg-type]
        assert candidates[0].subjects == ["hn/alice"]

    def test_bare_name_not_in_chunks_passes_through(self) -> None:
        chunks = [self._chunk("hn/alice talked about Einstein")]
        candidates = [self._candidate(["Einstein"])]
        _rewrite_hn_subjects(candidates, chunks)  # type: ignore[arg-type]
        assert candidates[0].subjects == ["Einstein"]

    def test_deduplicates_after_rewriting(self) -> None:
        chunks = [self._chunk("hn/alice: hi")]
        candidates = [self._candidate(["alice", "hn/alice"])]
        _rewrite_hn_subjects(candidates, chunks)  # type: ignore[arg-type]
        assert candidates[0].subjects == ["hn/alice"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_hn_extractor_in_registry(self) -> None:
        from particles.extraction.registry import get_extractors

        ids = [p.EXTRACTOR_ID for p in get_extractors()]
        assert "hackernews-extractor" in ids

    def test_hn_importer_in_registry(self) -> None:
        from particles.ingest.importers.registry import get_importers

        importers = get_importers()
        assert any(isinstance(d, HackerNewsImporter) for d in importers)

    def test_hn_extractor_before_general(self) -> None:
        from particles.extraction.general import GeneralExtractor
        from particles.extraction.registry import get_extractors

        plugins = get_extractors()
        hn_idx = next(i for i, p in enumerate(plugins) if p.EXTRACTOR_ID == "hackernews-extractor")
        general_idx = next(i for i, p in enumerate(plugins) if isinstance(p, GeneralExtractor))
        assert hn_idx < general_idx


# ---------------------------------------------------------------------------
# Bounds on what the API sends (security review 2026-10-28-2: AA18, AA57),
# through a real ``httpx.AsyncClient`` on an in-process ``MockTransport``.
# ---------------------------------------------------------------------------


def _mock_client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_hacker_news_body_nested_past_the_decoder_limit_reads_as_no_item() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[" * 200_000)

    async with _mock_client(handler) as client:
        assert await _fetch_item(client, 1) is None


# --- AA57: the Hacker News walks visit each item once -----------------------


@pytest.mark.asyncio
async def test_a_hacker_news_kids_cycle_fetches_no_more_items_than_max_comments() -> None:
    fetches = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal fetches
        fetches += 1
        item_id = int(request.url.path.rsplit("/", 1)[-1].split(".")[0])
        body = {"id": item_id, "type": "story" if item_id == 1 else "comment", "kids": [2]}
        return httpx.Response(200, content=json.dumps(body).encode())

    max_comments = 5
    async with _mock_client(handler) as client:
        await hn_fetch_thread(client, 1, max_comments)
    assert fetches <= max_comments + 1


@pytest.mark.asyncio
async def test_a_hacker_news_walk_of_distinct_ids_stops_at_max_comments_fetches() -> None:
    fetches = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal fetches
        fetches += 1
        item_id = int(request.url.path.rsplit("/", 1)[-1].split(".")[0])
        body = {"id": item_id, "text": f"c{item_id}", "kids": [item_id * 2, item_id * 2 + 1]}
        return httpx.Response(200, content=json.dumps(body).encode())

    async with _mock_client(handler) as client:
        payload = await hn_fetch_thread(client, 1, 7)
    assert fetches == 8
    # Depth first: each reply sequence stays contiguous.
    assert list(payload["comments"]) == ["2", "4", "8", "16", "32", "64", "128"]


def test_a_comment_map_whose_kids_revisit_a_comment_renders_each_once() -> None:
    story = {"id": 1, "kids": [2]}
    comments = {
        "2": {"id": 2, "text": "two", "kids": [3]},
        "3": {"id": 3, "text": "three", "kids": [2, 3]},
    }
    walked = _walk_comments_dfs(story, comments, min_score=0)
    assert [(depth, c["id"]) for depth, c in walked] == [(0, 2), (1, 3)]


# --- Chunk budget and kid ids (security review 2026-10-30, AB18, AB63) ------


def _hn_budget() -> tuple[int, int]:
    from particles.config import get_config

    ext = get_config().extraction
    return ext.single_call_threshold_chars, ext.comment_chunk_chars


def test_hackernews_one_long_comment_stays_within_the_chunk_budget() -> None:
    """A single comment past the chunk budget is clipped to it, not sent as one call."""
    from particles.extraction.hackernews import _build_hn_chunks

    threshold, chunk_chars = _hn_budget()
    story = {"id": 1, "by": "a", "title": "t", "kids": [2]}
    comments = {"2": {"id": 2, "by": "b", "text": "x" * (threshold + chunk_chars)}}
    chunks = _build_hn_chunks(story, comments, 2, 0, threshold, chunk_chars)
    assert max(len(c.chunk_text) for c in chunks) <= threshold + 400


def test_hackernews_story_text_stays_within_the_chunk_budget() -> None:
    """A story text past the threshold is split into parts, none past the budget."""
    from particles.extraction.hackernews import _build_hn_chunks

    threshold, chunk_chars = _hn_budget()
    story = {"id": 1, "by": "a", "title": "t", "text": "q" * (threshold + chunk_chars)}
    chunks = _build_hn_chunks(story, {}, 2, 0, threshold, chunk_chars)
    assert max(len(c.chunk_text) for c in chunks) <= threshold + 400
    assert "".join(c.chunk_text for c in chunks).count("q") == threshold + chunk_chars


def test_hackernews_short_story_keeps_its_single_story_chunk() -> None:
    from particles.extraction.hackernews import _build_hn_chunks

    chunks = _build_hn_chunks(_STORY, {}, 2, 0, 30_000, 10_000)
    assert [c.chunk_id for c in chunks] == ["story"]
    assert chunks[0].chunk_text == _render_hn_story_body(_STORY)


async def test_hackernews_walk_fetches_only_integer_item_ids() -> None:
    """Every URL the thread walk requests has the path /v0/item/<digits>.json."""
    import json
    import re

    import httpx

    from particles.extraction.hackernews import _fetch_thread

    item_path = re.compile(r"^/v0/item/\d+\.json$")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.raw_path.decode())
        if request.url.path.endswith("/item/1.json"):
            body = {"id": 1, "type": "story", "kids": ["../../user/x", {"k": 1}, 2.5, True, 3]}
        else:
            body = {"id": 3, "type": "comment", "text": "t"}
        return httpx.Response(200, content=json.dumps(body).encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        thread = await _fetch_thread(client, 1, 50)

    assert requested
    assert all(item_path.match(p) for p in requested), requested
    assert set(thread["comments"]) == {"3"}


def test_hackernews_stored_walk_skips_kids_that_are_not_item_ids() -> None:
    from particles.extraction.hackernews import _walk_comments_dfs

    story = {"kids": ["2", True, 2.0, 3]}
    comments = {"2": {"text": "string id"}, "True": {"text": "bool"}, "3": {"text": "ok"}}
    walked = _walk_comments_dfs(story, comments, 0)
    assert [c["text"] for _, c in walked] == ["ok"]


def test_hackernews_api_url_refuses_a_non_integer_id() -> None:
    from particles.extraction.hackernews import _api_url

    for bad in ("../x", True, -1, 2.5):
        with pytest.raises(ValueError):
            _api_url(bad)  # type: ignore[arg-type]
