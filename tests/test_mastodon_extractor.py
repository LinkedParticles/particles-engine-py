"""Tests for the Mastodon extractor and importer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.core.schema import Snapshot, UncertaintyNature
from particles.extraction.incremental import ChunkUnit
from particles.extraction.mastodon import (
    APPLICABILITY,
    DEFAULT_TRUST_WEIGHT,
    EXTRACTOR_ID,
    SOURCE_TYPE,
    MastodonExtractor,
    _build_mastodon_chunks,
    _build_status_meta_candidate,
    _html_to_text,
    _parse_status_route,
    _render_status_headline,
    _rewrite_mastodon_subjects,
    _root_id_from_ancestors,
)
from particles.ingest.importers.mastodon import MastodonImporter
from tests._capped_http import set_capped_responses

# ---------------------------------------------------------------------------
# Fixtures — realistic Mastodon Status entity shape
# ---------------------------------------------------------------------------

_ACCOUNT = {
    "id": "11111",
    "username": "patio11",
    "acct": "patio11",  # local — no @host suffix
    "display_name": "Patrick McKenzie",
    "url": "https://mastodon.social/@patio11",
}

_STATUS = {
    "id": "112345678901234567",
    "created_at": "2025-01-01T12:00:00.000Z",
    "in_reply_to_id": None,
    "in_reply_to_account_id": None,
    "sensitive": False,
    "spoiler_text": "",
    "language": "en",
    "url": "https://mastodon.social/@patio11/112345678901234567",
    "content": (
        "<p>Show HN: a new tool for structured knowledge. "
        "Built this because I needed structured belief management.</p>"
    ),
    "favourites_count": 42,
    "reblogs_count": 7,
    "replies_count": 3,
    "reblog": None,
    "account": _ACCOUNT,
    "card": None,
}

# Five-reply context tree:
#   _STATUS (root)
#   ├── reply A (alice; top-level)
#   │   └── reply A1 (alice replies to self)
#   ├── reply B (bob; top-level)
#   └── reply C (charlie; top-level, with CW)
_REPLY_A = {
    "id": "112345678901234568",
    "created_at": "2025-01-01T12:05:00.000Z",
    "in_reply_to_id": "112345678901234567",
    "spoiler_text": "",
    "language": "en",
    "content": "<p>Nice work. Reminds me of doxastic logic.</p>",
    "favourites_count": 3,
    "reblogs_count": 0,
    "replies_count": 1,
    "reblog": None,
    "account": {"acct": "alice@othersite.org"},
    "card": None,
}
_REPLY_A1 = {
    "id": "112345678901234569",
    "created_at": "2025-01-01T12:10:00.000Z",
    "in_reply_to_id": "112345678901234568",
    "spoiler_text": "",
    "content": "<p>The carry-forward optimisation is the interesting bit.</p>",
    "favourites_count": 1,
    "reblogs_count": 0,
    "replies_count": 0,
    "reblog": None,
    "account": {"acct": "alice@othersite.org"},
    "card": None,
}
_REPLY_B = {
    "id": "112345678901234570",
    "created_at": "2025-01-01T12:15:00.000Z",
    "in_reply_to_id": "112345678901234567",
    "spoiler_text": "",
    "content": "<p>How does this compare to <i>existing knowledge graphs</i>?</p>",
    "favourites_count": 0,
    "reblogs_count": 0,
    "replies_count": 0,
    "reblog": None,
    "account": {"acct": "bob"},
    "card": None,
}
_REPLY_C = {
    "id": "112345678901234571",
    "created_at": "2025-01-01T12:20:00.000Z",
    "in_reply_to_id": "112345678901234567",
    "spoiler_text": "long take on knowledge graphs",
    "content": "<p>Lots of thoughts here&#8230;</p>",
    "favourites_count": 5,
    "reblogs_count": 0,
    "replies_count": 0,
    "reblog": None,
    "account": {"acct": "charlie"},
    "card": None,
}

_MASTODON_BLOB = {
    "status": _STATUS,
    "context": {
        "ancestors": [],
        "descendants": [_REPLY_A, _REPLY_A1, _REPLY_B, _REPLY_C],
    },
    "instance": "mastodon.social",
}


def _make_snapshot() -> Snapshot:
    from datetime import UTC, datetime

    from particles.core.schema import ExtractionStatus, WarcRecordType

    return Snapshot(
        snapshot_id="test-snap",
        captured_at=datetime.now(UTC),
        content_hash="abc",
        extraction_status=ExtractionStatus.PENDING,
        warc_record_type=WarcRecordType.RESPONSE,
        author_id="mastodon:patio11",
        content_published_at=datetime.fromisoformat("2025-01-01T12:00:00+00:00"),
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_source_type(self) -> None:
        assert SOURCE_TYPE == "MASTODON_THREAD"

    def test_extractor_id(self) -> None:
        assert EXTRACTOR_ID == "mastodon-extractor"

    def test_default_trust_weight(self) -> None:
        assert pytest.approx(0.50) == DEFAULT_TRUST_WEIGHT

    def test_applicability_must_mastodon(self) -> None:
        assert len(APPLICABILITY) == 1
        clause = APPLICABILITY[0]
        assert clause.keyword == "MUST"
        assert clause.domain_label == "Mastodon"
        assert "MASTODON_THREAD" in clause.source_types

    def test_infer_domain_returns_mastodon(self) -> None:
        from particles.extraction.registry import infer_domain

        assert infer_domain("MASTODON_THREAD") == "Mastodon"


# ---------------------------------------------------------------------------
# URL parsing + federation routing
# ---------------------------------------------------------------------------


class TestMastodonImporterUrls:
    def test_accepts_local_status_url(self) -> None:
        d = MastodonImporter()
        assert d.accepts_url("https://mastodon.social/@patio11/112345678901234567")

    def test_accepts_cross_instance_status_url(self) -> None:
        d = MastodonImporter()
        assert d.accepts_url("https://fosstodon.org/@alice@mastodon.social/112345678901234567")

    def test_accepts_raw_api_url(self) -> None:
        d = MastodonImporter()
        assert d.accepts_url("https://mastodon.social/api/v1/statuses/112345678901234567")

    def test_rejects_bare_root(self) -> None:
        d = MastodonImporter()
        assert not d.accepts_url("https://mastodon.social")
        assert not d.accepts_url("https://mastodon.social/")
        assert not d.accepts_url("https://mastodon.social/explore")

    def test_rejects_unrelated_url(self) -> None:
        d = MastodonImporter()
        assert not d.accepts_url("https://news.ycombinator.com/item?id=12345")
        assert not d.accepts_url("https://reddit.com/r/x/")

    def test_parse_local_returns_viewing_instance(self) -> None:
        parsed = _parse_status_route("https://mastodon.social/@patio11/112345678901234567")
        assert parsed is not None
        home, sid, acct = parsed
        assert home == "mastodon.social"
        assert sid == "112345678901234567"
        assert acct == "patio11"

    def test_parse_cross_instance_returns_HOME_not_viewing(self) -> None:
        """Federation routing: the importer fetches from the user's home
        instance, NOT the viewing instance. This avoids two corpus entries
        for the same canonical status (per v1 design — see module docstring).
        """
        parsed = _parse_status_route(
            "https://fosstodon.org/@alice@mastodon.social/112345678901234567"
        )
        assert parsed is not None
        home, sid, acct = parsed
        # CRITICAL: fetch from mastodon.social (alice's home), not fosstodon.org.
        assert home == "mastodon.social"
        assert sid == "112345678901234567"
        # The visible account handle preserves the cross-instance suffix.
        assert acct == "alice@mastodon.social"

    def test_parse_api_url_returns_host(self) -> None:
        parsed = _parse_status_route("https://mastodon.social/api/v1/statuses/112345678901234567")
        assert parsed is not None
        home, sid, acct = parsed
        assert home == "mastodon.social"
        assert sid == "112345678901234567"
        # No handle until we fetch — None is the explicit signal.
        assert acct is None

    def test_derive_fetch_instance_federation_safe(self) -> None:
        """Exposes the routing decision as a public helper so operators can
        confirm the importer is federation-aware before depositing."""
        d = MastodonImporter()
        assert (
            d.derive_fetch_instance(
                "https://fosstodon.org/@alice@mastodon.social/112345678901234567"
            )
            == "mastodon.social"
        )
        # Local view: viewing instance == home instance, so the fetch
        # stays on the viewing host.
        assert (
            d.derive_fetch_instance("https://mastodon.social/@patio11/112345678901234567")
            == "mastodon.social"
        )

    def test_derive_fetch_instance_returns_none_for_garbage(self) -> None:
        d = MastodonImporter()
        assert d.derive_fetch_instance("https://example.com/foo") is None

    # ----- primary_url -----

    def test_primary_url_with_card_returns_card_url(self) -> None:
        """A status with a link card carries the external URL in ``card.url``."""
        blob = {
            "status": {
                "id": "112345678901234567",
                "url": "https://mastodon.social/@alice/112345678901234567",
                "content": "<p>Interesting read</p>",
                "card": {
                    "url": "https://example.com/article-x",
                    "title": "Article X",
                    "type": "link",
                },
            },
            "context": {"ancestors": [], "descendants": []},
            "instance": "mastodon.social",
        }
        assert (
            MastodonImporter().primary_url(json.dumps(blob).encode())
            == "https://example.com/article-x"
        )

    def test_primary_url_no_card_returns_none(self) -> None:
        """Text-only Mastodon statuses have no link card."""
        blob = {
            "status": {
                "id": "112345678901234567",
                "url": "https://mastodon.social/@alice/112345678901234567",
                "content": "<p>Just my thoughts.</p>",
                "card": None,
            },
            "context": {"ancestors": [], "descendants": []},
            "instance": "mastodon.social",
        }
        assert MastodonImporter().primary_url(json.dumps(blob).encode()) is None

    def test_primary_url_card_without_url_returns_none(self) -> None:
        """Defensive: a malformed card without ``url`` returns None."""
        blob = {
            "status": {
                "id": "112345678901234567",
                "card": {"title": "Article without url field"},
            },
            "context": {"ancestors": [], "descendants": []},
        }
        assert MastodonImporter().primary_url(json.dumps(blob).encode()) is None

    def test_primary_url_malformed_returns_none(self) -> None:
        d = MastodonImporter()
        assert d.primary_url(b"not json") is None
        assert d.primary_url(b"{}") is None
        assert d.primary_url(b'{"status": null}') is None

    def test_default_follow_post_links_true(self) -> None:
        assert MastodonImporter.DEFAULT_FOLLOW_POST_LINKS is True
        assert MastodonImporter.DEFAULT_FOLLOW_COMMENT_LINKS is False


# ---------------------------------------------------------------------------
# Importer deposit (end-to-end with mocked HTTP)
# ---------------------------------------------------------------------------


class TestMastodonImporterDeposit:
    @pytest.mark.asyncio
    async def test_deposit_fetches_status_and_context(self, db_session: object) -> None:
        """End-to-end deposit: importer hits both status + context endpoints."""

        def make_response(payload: object) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(return_value=payload)
            resp.raise_for_status = MagicMock()
            return resp

        def fake_get(url: str) -> MagicMock:
            if url.endswith("/api/v1/statuses/112345678901234567"):
                return make_response(_STATUS)
            if url.endswith("/api/v1/statuses/112345678901234567/context"):
                return make_response(
                    {
                        "ancestors": [],
                        "descendants": [_REPLY_A, _REPLY_A1, _REPLY_B, _REPLY_C],
                    }
                )
            return make_response(None)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, router=fake_get)

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            entry_id, snapshot_id = await MastodonImporter().deposit(
                db_session,  # type: ignore[arg-type]
                "https://mastodon.social/@patio11/112345678901234567",
                "test-operator",
                [],
            )

        from particles.corpus.store import get_entry, get_snapshot

        entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
        assert entry is not None
        assert entry.source_type == "MASTODON_THREAD"
        # Canonical URL is the API form on the home instance.
        assert entry.uri_r == ("https://mastodon.social/api/v1/statuses/112345678901234567")

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        assert snap.author_id == "mastodon:patio11"
        assert snap.content_published_at is not None

    @pytest.mark.asyncio
    async def test_deposit_federation_routes_to_home_instance(self, db_session: object) -> None:
        """When given an @user@remote URL, the importer fetches from <remote>.

        Concretely: pasting the same status via fosstodon.org's mirror MUST
        result in an HTTP call to mastodon.social (the home instance), not
        to fosstodon.org. Otherwise content_hash dedup cannot
        collapse the two operator-visible URLs onto one corpus entry.
        """
        seen_hosts: list[str] = []

        def make_response(payload: object) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(return_value=payload)
            resp.raise_for_status = MagicMock()
            return resp

        def fake_get(url: str) -> MagicMock:
            # Record the host so the assertion below can verify routing.
            from urllib.parse import urlparse as _up

            seen_hosts.append(_up(url).netloc)
            if url.endswith("/api/v1/statuses/112345678901234567"):
                return make_response(_STATUS)
            return make_response({"ancestors": [], "descendants": []})

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, router=fake_get)

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            await MastodonImporter().deposit(
                db_session,  # type: ignore[arg-type]
                "https://fosstodon.org/@patio11@mastodon.social/112345678901234567",
                "test-operator",
                [],
            )

        # Every outbound GET targeted the home instance, not the viewer.
        assert seen_hosts, "importer made no HTTP calls"
        assert all(host == "mastodon.social" for host in seen_hosts), (
            f"federation routing broken: saw hosts {seen_hosts}"
        )


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------


class TestHtmlToText:
    def test_strips_paragraph_tags_to_blank_lines(self) -> None:
        assert _html_to_text("<p>One.</p><p>Two.</p>") == "One.\n\nTwo."

    def test_converts_br_to_newline(self) -> None:
        assert _html_to_text("Line A<br>Line B") == "Line A\nLine B"
        assert _html_to_text("Line A<br/>Line B") == "Line A\nLine B"

    def test_strips_inline_anchor(self) -> None:
        out = _html_to_text('<p>Hi <a href="https://x">there</a> world.</p>')
        assert out == "Hi there world."

    def test_unescapes_html_entities(self) -> None:
        # mdash (&mdash; → —), apostrophe (&#x27; → '), nbsp (&nbsp; → \xa0)
        out = _html_to_text("<p>Lots of thoughts here&#8230;</p>")
        assert "…" in out

    def test_strips_to_empty_on_pure_markup(self) -> None:
        assert _html_to_text("<p></p>") == ""


# ---------------------------------------------------------------------------
# Prose rendering + chunking
# ---------------------------------------------------------------------------


class TestRenderHeadline:
    def test_includes_handle_and_engagement_counts(self) -> None:
        head = _render_status_headline(_STATUS, "mastodon.social")
        assert "mastodon/patio11" in head
        assert "favourites: 42" in head
        assert "boosts: 7" in head
        assert "replies: 3" in head
        # No CW on this status — header should not include "CW:".
        assert "CW:" not in head
        assert "mastodon.social" in head

    def test_includes_cw_when_spoiler_text_present(self) -> None:
        cw_status = {**_STATUS, "spoiler_text": "long take on knowledge graphs"}
        head = _render_status_headline(cw_status, "mastodon.social")
        assert "CW: long take on knowledge graphs" in head

    def test_marks_boost(self) -> None:
        boost = {
            **_STATUS,
            "reblog": {
                "id": "9999",
                "content": "<p>Original poster's content.</p>",
                "account": {"acct": "originalauthor"},
                "spoiler_text": "",
                "favourites_count": 100,
                "reblogs_count": 50,
                "replies_count": 10,
            },
        }
        head = _render_status_headline(boost, "mastodon.social")
        assert "[BOOST]" in head
        # The boosted content is what the LLM sees, not the wrapper's empty content.
        assert "Original poster" in head


class TestBuildChunks:
    def test_status_chunk_plus_replies_chunk(self) -> None:
        chunks = _build_mastodon_chunks(
            status=_STATUS,
            ancestors=[],
            descendants=[_REPLY_A, _REPLY_A1, _REPLY_B, _REPLY_C],
            instance="mastodon.social",
            indent_per_level=2,
            min_favourites=0,
            single_call_threshold=30000,
            chunk_chars=10000,
        )
        ids = [c.chunk_id for c in chunks]
        assert "status" in ids
        assert any(c.startswith("replies") for c in ids)
        # Reply prose includes the indented handle lines.
        joined = "\n".join(c.chunk_text for c in chunks)
        assert "mastodon/alice@othersite.org" in joined
        assert "mastodon/bob" in joined

    def test_nested_reply_is_indented_deeper(self) -> None:
        chunks = _build_mastodon_chunks(
            status=_STATUS,
            ancestors=[],
            descendants=[_REPLY_A, _REPLY_A1],
            instance="mastodon.social",
            indent_per_level=2,
            min_favourites=0,
            single_call_threshold=30000,
            chunk_chars=10000,
        )
        text = next(c.chunk_text for c in chunks if c.chunk_id.startswith("replies"))
        lines = [ln for ln in text.split("\n") if "mastodon/alice" in ln]
        assert len(lines) >= 2
        # The reply-to-the-reply has more leading whitespace than the top-level one.
        top_level = next(ln for ln in lines if "Nice work" in ln)
        nested = next(ln for ln in lines if "carry-forward" in ln)
        assert len(nested) - len(nested.lstrip()) > len(top_level) - len(top_level.lstrip())

    def test_min_favourites_filters_replies(self) -> None:
        """Setting min_reply_favourites>0 drops low-engagement replies."""
        chunks = _build_mastodon_chunks(
            status=_STATUS,
            ancestors=[],
            descendants=[_REPLY_A, _REPLY_B],  # A has 3 favs, B has 0
            instance="mastodon.social",
            indent_per_level=2,
            min_favourites=2,
            single_call_threshold=30000,
            chunk_chars=10000,
        )
        joined = "\n".join(c.chunk_text for c in chunks)
        assert "mastodon/alice@othersite.org" in joined
        assert "mastodon/bob" not in joined

    def test_ancestor_chain_appears_above_focal_status(self) -> None:
        ancestor = {
            **_REPLY_A,
            "id": "112345678901234500",
            "in_reply_to_id": None,
            "content": "<p>I started this thread.</p>",
        }
        chunks = _build_mastodon_chunks(
            status=_STATUS,
            ancestors=[ancestor],
            descendants=[],
            instance="mastodon.social",
            indent_per_level=2,
            min_favourites=0,
            single_call_threshold=30000,
            chunk_chars=10000,
        )
        status_text = next(c.chunk_text for c in chunks if c.chunk_id == "status")
        # Ancestor appears under a CONTEXT header, ahead of the headline.
        assert "CONTEXT (ancestors" in status_text
        assert "I started this thread" in status_text
        # And the headline still follows.
        assert "MASTODON THREAD" in status_text


# ---------------------------------------------------------------------------
# Story-meta particle (dual-emission)
# ---------------------------------------------------------------------------


class TestStatusMetaParticle:
    def test_dual_emission_keys_align(self) -> None:
        """``mastodon:*`` and ``social:*`` keys MUST carry the same values
        §3."""
        particle = _build_status_meta_candidate(
            _STATUS, "mastodon.social", "112345678901234567", ["mastodon/patio11"]
        )
        assert particle is not None
        props = particle.properties
        assert props is not None
        # Mastodon-specific.
        assert props["mastodon:hasFavouritesCount"] == 42
        assert props["mastodon:hasReblogsCount"] == 7
        assert props["mastodon:hasRepliesCount"] == 3
        assert props["mastodon:hasAccountHandle"] == "patio11"
        assert props["mastodon:hasStatusId"] == "112345678901234567"
        assert props["mastodon:hasInstance"] == "mastodon.social"
        assert props["mastodon:isReblog"] is False
        # Cross-platform dual emission — same values, different keys.
        assert props["social:hasScore"] == props["mastodon:hasFavouritesCount"]
        assert props["social:hasReplyCount"] == props["mastodon:hasRepliesCount"]
        assert props["social:hasReactionCount"] == props["mastodon:hasReblogsCount"]
        assert props["social:hasAuthorHandle"] == props["mastodon:hasAccountHandle"]
        # Thread structure: root with no parent → root id = self.
        assert props["thread:hasRootId"] == "112345678901234567"
        # Language present.
        assert props["content:hasLanguage"] == "en"

    def test_property_keys_use_prefix_form(self) -> None:
        """every key MUST be ``prefix:LocalName``."""
        particle = _build_status_meta_candidate(
            _STATUS, "mastodon.social", "112345678901234567", []
        )
        assert particle is not None
        assert particle.properties is not None
        for key in particle.properties:
            assert ":" in key, f"key {key!r} violates the prefix:LocalName rule"

    def test_cw_present_when_spoiler_text_nonempty(self) -> None:
        cw_status = {**_STATUS, "spoiler_text": "discussion of mental health"}
        particle = _build_status_meta_candidate(cw_status, "mastodon.social", cw_status["id"], [])
        assert particle is not None
        assert particle.properties is not None
        assert particle.properties["mastodon:hasSpoilerText"] == "discussion of mental health"

    def test_cw_key_absent_when_no_spoiler_text(self) -> None:
        """Per task spec: omit ``mastodon:hasSpoilerText`` entirely when
        the field is empty. The key's absence is the signal, not None."""
        particle = _build_status_meta_candidate(_STATUS, "mastodon.social", _STATUS["id"], [])
        assert particle is not None
        assert particle.properties is not None
        assert "mastodon:hasSpoilerText" not in particle.properties

    def test_cw_key_absent_when_spoiler_text_whitespace_only(self) -> None:
        """An all-whitespace spoiler_text is treated as absent."""
        ws_status = {**_STATUS, "spoiler_text": "   "}
        particle = _build_status_meta_candidate(ws_status, "mastodon.social", ws_status["id"], [])
        assert particle is not None
        assert particle.properties is not None
        assert "mastodon:hasSpoilerText" not in particle.properties

    # ----- reblog-target identity capture -----

    def test_reblog_target_identity_present_when_status_is_boost(self) -> None:
        """when ``status.reblog`` is non-null, the meta
        particle captures the boosted status's id / account / uri so a
        future ``BOOSTS``-relation activation can backfill the edge."""
        boost = {
            **_STATUS,
            "reblog": {
                "id": "111111111111111111",
                "uri": "https://othersite.org/users/alice/statuses/111111111111111111",
                "content": "<p>The post being boosted.</p>",
                "account": {
                    "acct": "alice@othersite.org",
                    "url": "https://othersite.org/@alice",
                },
                "spoiler_text": "",
                "favourites_count": 100,
                "reblogs_count": 50,
                "replies_count": 10,
            },
        }
        particle = _build_status_meta_candidate(boost, "mastodon.social", boost["id"], [])
        assert particle is not None
        assert particle.properties is not None
        props = particle.properties
        # Boolean still present for the fast-path predicate.
        assert props["mastodon:isReblog"] is True
        # New keys (this ADR): three pieces of boosted-status identity.
        assert props["mastodon:reblogOfStatusId"] == "111111111111111111"
        assert props["mastodon:reblogOfAccountAcct"] == "alice@othersite.org"
        assert (
            props["mastodon:reblogOfStatusUri"]
            == "https://othersite.org/users/alice/statuses/111111111111111111"
        )

    def test_reblog_target_keys_absent_when_not_a_boost(self) -> None:
        """A regular (non-boost) status MUST NOT carry the reblog-of
        keys. Their absence is the signal."""
        particle = _build_status_meta_candidate(_STATUS, "mastodon.social", _STATUS["id"], [])
        assert particle is not None
        assert particle.properties is not None
        props = particle.properties
        assert props["mastodon:isReblog"] is False
        assert "mastodon:reblogOfStatusId" not in props
        assert "mastodon:reblogOfAccountAcct" not in props
        assert "mastodon:reblogOfStatusUri" not in props

    def test_reblog_target_partial_data_still_captures_what_it_can(self) -> None:
        """Defensive: if the boosted status is missing one of the three
        fields (malformed Mastodon API response), capture the ones that
        ARE present and omit the missing ones."""
        boost = {
            **_STATUS,
            "reblog": {
                "id": "222222222222222222",
                # No "uri" field.
                "content": "<p>partial boost record.</p>",
                "account": {"acct": "bob@example.org"},
                "spoiler_text": "",
                "favourites_count": 0,
                "reblogs_count": 0,
                "replies_count": 0,
            },
        }
        particle = _build_status_meta_candidate(boost, "mastodon.social", boost["id"], [])
        assert particle is not None
        assert particle.properties is not None
        props = particle.properties
        assert props["mastodon:reblogOfStatusId"] == "222222222222222222"
        assert props["mastodon:reblogOfAccountAcct"] == "bob@example.org"
        assert "mastodon:reblogOfStatusUri" not in props

    def test_is_reblog_true_when_reblog_field_set(self) -> None:
        boost = {
            **_STATUS,
            "reblog": {
                "id": "9999",
                "content": "<p>Original.</p>",
                "account": {"acct": "originalauthor"},
                "spoiler_text": "",
                "favourites_count": 100,
                "reblogs_count": 50,
                "replies_count": 10,
            },
        }
        particle = _build_status_meta_candidate(boost, "mastodon.social", boost["id"], [])
        assert particle is not None
        assert particle.properties is not None
        assert particle.properties["mastodon:isReblog"] is True

    def test_is_reblog_false_on_ordinary_status(self) -> None:
        particle = _build_status_meta_candidate(_STATUS, "mastodon.social", _STATUS["id"], [])
        assert particle is not None
        assert particle.properties is not None
        assert particle.properties["mastodon:isReblog"] is False

    def test_content_url_only_emitted_when_card_present(self) -> None:
        # No card on the base fixture → key absent.
        p_base = _build_status_meta_candidate(_STATUS, "mastodon.social", _STATUS["id"], [])
        assert p_base is not None and p_base.properties is not None
        assert "content:hasUrl" not in p_base.properties

        # With card → key carries the card's URL.
        carded = {**_STATUS, "card": {"url": "https://example.com/article"}}
        p_card = _build_status_meta_candidate(carded, "mastodon.social", carded["id"], [])
        assert p_card is not None and p_card.properties is not None
        assert p_card.properties["content:hasUrl"] == "https://example.com/article"

    def test_thread_parent_id_only_emitted_when_in_reply(self) -> None:
        # Root status has no parent → key absent.
        p_root = _build_status_meta_candidate(_STATUS, "mastodon.social", _STATUS["id"], [])
        assert p_root is not None and p_root.properties is not None
        assert "thread:hasParentId" not in p_root.properties

        # A reply carries the parent id.
        p_reply = _build_status_meta_candidate(_REPLY_A, "mastodon.social", _STATUS["id"], [])
        assert p_reply is not None and p_reply.properties is not None
        assert p_reply.properties["thread:hasParentId"] == "112345678901234567"

    def test_returns_none_for_malformed_status(self) -> None:
        assert _build_status_meta_candidate({}, "mastodon.social", "", []) is None
        # Missing id.
        assert _build_status_meta_candidate({"account": {"acct": "u"}}, "x", "", []) is None
        # Missing account.acct.
        assert _build_status_meta_candidate({"id": "1"}, "x", "", []) is None

    def test_summary_content_is_human_readable(self) -> None:
        particle = _build_status_meta_candidate(
            _STATUS, "mastodon.social", _STATUS["id"], ["mastodon/patio11"]
        )
        assert particle is not None
        assert "Mastodon status" in particle.content
        assert "42 favourites" in particle.content
        assert "7 boosts" in particle.content
        assert particle.uncertainty_nature == UncertaintyNature.EPISTEMIC

    def test_injected_subjects_propagate(self) -> None:
        particle = _build_status_meta_candidate(
            _STATUS, "mastodon.social", _STATUS["id"], ["mastodon/patio11"]
        )
        assert particle is not None
        assert "mastodon/patio11" in particle.subjects


# ---------------------------------------------------------------------------
# root_id walking
# ---------------------------------------------------------------------------


class TestRootIdFromAncestors:
    def test_root_is_first_ancestor_when_chain_exists(self) -> None:
        ancestors = [
            {"id": "root-1"},
            {"id": "mid-2"},
        ]
        assert _root_id_from_ancestors(_STATUS, ancestors) == "root-1"

    def test_root_is_self_when_no_ancestors(self) -> None:
        assert _root_id_from_ancestors(_STATUS, []) == "112345678901234567"


# ---------------------------------------------------------------------------
# Extractor: accepts() + integration with the meta particle
# ---------------------------------------------------------------------------


class TestMastodonExtractor:
    def test_accepts_mastodon_thread(self) -> None:
        assert MastodonExtractor().accepts("MASTODON_THREAD")

    def test_rejects_other_source_types(self) -> None:
        assert not MastodonExtractor().accepts("HACKERNEWS_THREAD")
        assert not MastodonExtractor().accepts("REDDIT_POST")
        assert not MastodonExtractor().accepts("WEB_PAGE")

    @pytest.mark.asyncio
    async def test_extract_prepends_status_meta_particle(self) -> None:
        """The first candidate is the synthesised status-meta particle."""
        content = json.dumps(_MASTODON_BLOB).encode()
        snapshot = _make_snapshot()

        def make_candidate() -> MagicMock:
            c = MagicMock()
            c.subjects = []
            c.content = "Alice praises the carry-forward design."
            c.confidence_value = 0.75
            c.uncertainty_nature = UncertaintyNature.EPISTEMIC
            c.properties = None
            return c

        async def fake_llm(_text: str) -> tuple[list[MagicMock], list[str], bool]:
            return ([make_candidate()], [], False)

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=fake_llm),
        ):
            result = await MastodonExtractor().extract(snapshot, content)

        assert len(result.candidates) >= 2
        first = result.candidates[0]
        assert first.properties is not None
        assert first.properties.get("mastodon:hasStatusId") == "112345678901234567"
        assert first.properties.get("social:hasScore") == 42
        assert first.properties.get("mastodon:hasInstance") == "mastodon.social"

    @pytest.mark.asyncio
    async def test_extract_injects_author_subject_on_llm_candidates(self) -> None:
        content = json.dumps(_MASTODON_BLOB).encode()
        snapshot = _make_snapshot()

        def make_candidate() -> MagicMock:
            c = MagicMock()
            c.subjects = []
            c.content = "Alice praises the carry-forward design."
            c.confidence_value = 0.75
            c.uncertainty_nature = UncertaintyNature.EPISTEMIC
            c.properties = None
            return c

        async def fake_llm(_text: str) -> tuple[list[MagicMock], list[str], bool]:
            return ([make_candidate()], [], False)

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=fake_llm),
        ):
            result = await MastodonExtractor().extract(snapshot, content)

        llm_candidates = [c for c in result.candidates if c.properties is None]
        assert len(llm_candidates) >= 1
        for c in llm_candidates:
            assert "mastodon/patio11" in c.subjects

    @pytest.mark.asyncio
    async def test_extract_quality_note_on_bad_json(self) -> None:
        result = await MastodonExtractor().extract(_make_snapshot(), b"not json")
        assert result.candidates == []
        assert any("JSON parse error" in n for n in result.quality_notes)

    @pytest.mark.asyncio
    async def test_extract_quality_note_on_missing_status(self) -> None:
        result = await MastodonExtractor().extract(_make_snapshot(), b"{}")
        assert result.candidates == []
        assert any("status data" in n for n in result.quality_notes)


# ---------------------------------------------------------------------------
# Subject canonicalisation
# ---------------------------------------------------------------------------


class TestRewriteMastodonSubjects:
    def _candidate(self, subjects: list[str]) -> MagicMock:
        c = MagicMock()
        c.subjects = list(subjects)
        return c

    def _chunk(self, text: str) -> ChunkUnit:
        return ChunkUnit(chunk_id="c0", chunk_text=text)

    def test_bare_handle_in_chunks_gets_prefixed(self) -> None:
        chunks = [self._chunk("  mastodon/alice@othersite.org: nice take")]
        candidates = [self._candidate(["alice@othersite.org", "doxastic logic"])]
        _rewrite_mastodon_subjects(candidates, chunks)  # type: ignore[arg-type]
        assert "mastodon/alice@othersite.org" in candidates[0].subjects
        assert "alice@othersite.org" not in candidates[0].subjects
        assert "doxastic logic" in candidates[0].subjects

    def test_already_prefixed_passes_through(self) -> None:
        chunks = [self._chunk("mastodon/patio11 said hi")]
        candidates = [self._candidate(["mastodon/patio11"])]
        _rewrite_mastodon_subjects(candidates, chunks)  # type: ignore[arg-type]
        assert candidates[0].subjects == ["mastodon/patio11"]

    def test_bare_name_not_in_chunks_passes_through(self) -> None:
        chunks = [self._chunk("mastodon/patio11 talked about Einstein")]
        candidates = [self._candidate(["Einstein"])]
        _rewrite_mastodon_subjects(candidates, chunks)  # type: ignore[arg-type]
        assert candidates[0].subjects == ["Einstein"]

    def test_deduplicates_after_rewriting(self) -> None:
        chunks = [self._chunk("mastodon/patio11: hi")]
        candidates = [self._candidate(["patio11", "mastodon/patio11"])]
        _rewrite_mastodon_subjects(candidates, chunks)  # type: ignore[arg-type]
        assert candidates[0].subjects == ["mastodon/patio11"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_mastodon_extractor_in_registry(self) -> None:
        from particles.extraction.registry import get_extractors

        ids = [p.EXTRACTOR_ID for p in get_extractors()]
        assert "mastodon-extractor" in ids

    def test_mastodon_importer_in_registry(self) -> None:
        from particles.ingest.importers.registry import get_importers

        importers = get_importers()
        assert any(isinstance(d, MastodonImporter) for d in importers)

    def test_mastodon_extractor_before_general(self) -> None:
        from particles.extraction.general import GeneralExtractor
        from particles.extraction.registry import get_extractors

        plugins = get_extractors()
        m_idx = next(i for i, p in enumerate(plugins) if p.EXTRACTOR_ID == "mastodon-extractor")
        general_idx = next(i for i, p in enumerate(plugins) if isinstance(p, GeneralExtractor))
        assert m_idx < general_idx
