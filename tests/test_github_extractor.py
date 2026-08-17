"""Tests for the GitHub extractor and importer (naming)."""

from __future__ import annotations

import base64
import ipaddress
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.core.schema import Snapshot, UncertaintyNature
from particles.extraction.github import (
    APPLICABILITY_GIST,
    APPLICABILITY_PAGES,
    APPLICABILITY_REPO,
    DEFAULT_TRUST_WEIGHT_GIST,
    DEFAULT_TRUST_WEIGHT_PAGES,
    DEFAULT_TRUST_WEIGHT_REPO,
    EXTRACTOR_ID_GIST,
    EXTRACTOR_ID_PAGES,
    EXTRACTOR_ID_REPO,
    SOURCE_TYPE_GIST,
    SOURCE_TYPE_PAGES,
    SOURCE_TYPE_REPO,
    GitHubGistExtractor,
    GitHubPagesExtractor,
    GitHubRepoExtractor,
    _api_headers,
    _date_from_path,
    _normalize_raw_url,
    _parse_gist_url,
    _parse_iso_utc,
    _parse_pages_url,
    _parse_repo_url,
)
from particles.ingest.importers.github import GitHubImporter
from particles.url_safety import IPAddress, Resolver, UnsafeUrlError
from tests._capped_http import set_capped_responses

# ---------------------------------------------------------------------------
# Stub DNS — the gist git-clone fallback pre-resolves and pins its
# connection, so every test that reaches it injects a resolver, not the network.
# ---------------------------------------------------------------------------

# A public github.com edge address; only its publicness matters (a blocked
# address would be rejected before git is spawned).
_PUBLIC_ADDR = "140.82.112.4"


def _stub_resolver(*addrs: str) -> Resolver:
    """A ``Resolver`` returning fixed addresses for any hostname."""
    ips: list[IPAddress] = [ipaddress.ip_address(a) for a in addrs]
    return lambda _host: list(ips)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_snapshot(author_id: str | None = None) -> Snapshot:
    from particles.core.schema import ExtractionStatus, WarcRecordType

    return Snapshot(
        snapshot_id="test-snap",
        captured_at=datetime.now(UTC),
        content_hash="abc",
        extraction_status=ExtractionStatus.PENDING,
        warc_record_type=WarcRecordType.RESPONSE,
        author_id=author_id,
    )


_GIST_BLOB = {
    "id": "abc123",
    "description": "Quick notes on PyTorch attention",
    "owner": {"login": "karpathy"},
    "files": {
        "notes.md": {"content": "# Attention\n\nIs all you need."},
        "snippet.py": {"content": "def f():\n    return 1"},
    },
    "created_at": "2026-01-15T08:00:00Z",
    "updated_at": "2026-02-20T10:30:00Z",
}


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_source_types(self) -> None:
        assert SOURCE_TYPE_REPO == "GITHUB_REPO"
        assert SOURCE_TYPE_GIST == "GITHUB_GIST"
        assert SOURCE_TYPE_PAGES == "GITHUB_PAGES"

    def test_extractor_ids(self) -> None:
        assert EXTRACTOR_ID_REPO == "github-repo-extractor"
        assert EXTRACTOR_ID_GIST == "github-gist-extractor"
        assert EXTRACTOR_ID_PAGES == "github-pages-extractor"

    def test_default_trust_weights(self) -> None:
        assert pytest.approx(0.75) == DEFAULT_TRUST_WEIGHT_REPO
        assert pytest.approx(0.65) == DEFAULT_TRUST_WEIGHT_GIST
        assert pytest.approx(0.70) == DEFAULT_TRUST_WEIGHT_PAGES

    def test_applicability_clauses(self) -> None:
        for clauses, source_type in (
            (APPLICABILITY_REPO, "GITHUB_REPO"),
            (APPLICABILITY_GIST, "GITHUB_GIST"),
            (APPLICABILITY_PAGES, "GITHUB_PAGES"),
        ):
            assert len(clauses) == 1
            assert clauses[0].keyword == "MUST"
            assert clauses[0].domain_label == "software"
            assert source_type in clauses[0].source_types

    def test_infer_domain_resolves_to_software(self) -> None:
        from particles.extraction.registry import infer_domain

        assert infer_domain("GITHUB_REPO") == "software"
        assert infer_domain("GITHUB_GIST") == "software"
        assert infer_domain("GITHUB_PAGES") == "software"


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


class TestUrlParsing:
    def test_parse_repo_root(self) -> None:
        parsed = _parse_repo_url("https://github.com/karpathy/nanoGPT")
        assert parsed == ("karpathy", "nanoGPT", None, None)

    def test_parse_repo_root_trailing_slash(self) -> None:
        parsed = _parse_repo_url("https://github.com/karpathy/nanoGPT/")
        assert parsed == ("karpathy", "nanoGPT", None, None)

    def test_parse_repo_blob(self) -> None:
        parsed = _parse_repo_url("https://github.com/karpathy/nanoGPT/blob/master/README.md")
        assert parsed == ("karpathy", "nanoGPT", "master", "README.md")

    def test_parse_repo_blob_nested_path(self) -> None:
        parsed = _parse_repo_url("https://github.com/owner/repo/blob/main/docs/guides/intro.md")
        assert parsed == ("owner", "repo", "main", "docs/guides/intro.md")

    def test_parse_repo_rejects_non_github(self) -> None:
        assert _parse_repo_url("https://gitlab.com/owner/repo") is None

    def test_normalize_raw_url(self) -> None:
        result = _normalize_raw_url(
            "https://raw.githubusercontent.com/owner/repo/main/path/to/file.md"
        )
        assert result == "https://github.com/owner/repo/blob/main/path/to/file.md"

    def test_normalize_passthrough_when_not_raw(self) -> None:
        url = "https://github.com/owner/repo/blob/main/README.md"
        assert _normalize_raw_url(url) == url

    def test_parse_gist_url(self) -> None:
        assert _parse_gist_url("https://gist.github.com/karpathy/442a6bf5f6dbd61cb0e4") == (
            "karpathy",
            "442a6bf5f6dbd61cb0e4",
        )

    def test_parse_gist_rejects_non_gist(self) -> None:
        assert _parse_gist_url("https://github.com/karpathy/nanoGPT") is None

    def test_parse_pages_url_with_path(self) -> None:
        assert _parse_pages_url("https://karpathy.github.io/2026/02/12/microgpt/") == (
            "karpathy",
            "/2026/02/12/microgpt/",
        )

    def test_parse_pages_url_root(self) -> None:
        assert _parse_pages_url("https://karpathy.github.io") == ("karpathy", "")

    def test_parse_pages_url_doc_site(self) -> None:
        result = _parse_pages_url("https://recharts.github.io/en-US/examples/SimpleLineChart/")
        assert result == ("recharts", "/en-US/examples/SimpleLineChart/")

    def test_parse_pages_rejects_non_pages(self) -> None:
        assert _parse_pages_url("https://github.io/foo") is None
        assert _parse_pages_url("https://example.com") is None


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


class TestDateHelpers:
    def test_date_from_path_with_blog_pattern(self) -> None:
        d = _date_from_path("/2026/02/12/microgpt/")
        assert d == datetime(2026, 2, 12, tzinfo=UTC)

    def test_date_from_path_none_when_no_date(self) -> None:
        assert _date_from_path("/en-US/examples/SimpleLineChart/") is None

    def test_date_from_path_rejects_invalid(self) -> None:
        # February 30 — not a valid date
        assert _date_from_path("/2026/02/30/post/") is None

    def test_parse_iso_utc_z_suffix(self) -> None:
        assert _parse_iso_utc("2026-01-15T08:00:00Z") == datetime(2026, 1, 15, 8, 0, tzinfo=UTC)

    def test_parse_iso_utc_explicit_offset(self) -> None:
        result = _parse_iso_utc("2026-01-15T08:00:00+00:00")
        assert result == datetime(2026, 1, 15, 8, 0, tzinfo=UTC)

    def test_parse_iso_utc_none(self) -> None:
        assert _parse_iso_utc(None) is None
        assert _parse_iso_utc("") is None
        assert _parse_iso_utc("not a date") is None


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


class TestApiHeaders:
    def test_no_key_omits_authorization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_API_KEY", raising=False)
        headers = _api_headers()
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github+json"

    def test_with_key_adds_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_API_KEY", "ghp_secret")
        headers = _api_headers()
        assert headers["Authorization"] == "Bearer ghp_secret"


# ---------------------------------------------------------------------------
# Importer URL acceptance
# ---------------------------------------------------------------------------


class TestGitHubImporterAccepts:
    def setup_method(self) -> None:
        self.d = GitHubImporter()

    def test_accepts_repo_root(self) -> None:
        assert self.d.accepts_url("https://github.com/karpathy/nanoGPT")

    def test_accepts_repo_blob(self) -> None:
        assert self.d.accepts_url("https://github.com/karpathy/nanoGPT/blob/master/README.md")

    def test_accepts_raw_content(self) -> None:
        assert self.d.accepts_url("https://raw.githubusercontent.com/owner/repo/main/file.md")

    def test_accepts_gist(self) -> None:
        assert self.d.accepts_url("https://gist.github.com/karpathy/442a6bf5f6dbd61cb0e4")

    def test_accepts_pages_blog(self) -> None:
        assert self.d.accepts_url("https://karpathy.github.io/2026/02/12/microgpt/")

    def test_accepts_pages_root(self) -> None:
        assert self.d.accepts_url("https://karpathy.github.io")

    def test_rejects_non_github(self) -> None:
        assert not self.d.accepts_url("https://gitlab.com/owner/repo")
        assert not self.d.accepts_url("https://example.com")
        assert not self.d.accepts_url("https://www.reddit.com/r/foo/comments/x/y/")


# ---------------------------------------------------------------------------
# Importer — pages flow (no GitHub API, just HTTP)
# ---------------------------------------------------------------------------


class TestGitHubImporterPages:
    @pytest.mark.asyncio
    async def test_pages_sets_author_and_url_date(self, db_session: object) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body><p>Hello</p></body></html>"
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, return_value=mock_response)

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            d = GitHubImporter()
            entry_id, snapshot_id = await d.deposit(
                db_session,  # type: ignore[arg-type]
                "https://karpathy.github.io/2026/02/12/microgpt/",
                "test-operator",
                [],
            )

        from particles.corpus.store import get_entry, get_snapshot

        entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert entry is not None and snap is not None
        assert entry.source_type == "GITHUB_PAGES"
        assert snap.author_id == "github:karpathy"
        # SQLite strips tzinfo on round-trip; compare naive values.
        published = snap.content_published_at
        assert published is not None
        assert published.replace(tzinfo=None) == datetime(2026, 2, 12)

    @pytest.mark.asyncio
    async def test_pages_falls_back_to_last_modified_when_no_url_date(
        self, db_session: object
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body>x</body></html>"
        mock_response.headers = {"Last-Modified": "Wed, 21 Oct 2026 07:28:00 GMT"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, return_value=mock_response)

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            d = GitHubImporter()
            _, snapshot_id = await d.deposit(
                db_session,  # type: ignore[arg-type]
                "https://recharts.github.io/en-US/examples/SimpleLineChart/",
                "test-operator",
                [],
            )

        from particles.corpus.store import get_snapshot

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        assert snap.content_published_at is not None
        assert snap.content_published_at.year == 2026
        assert snap.content_published_at.month == 10


# ---------------------------------------------------------------------------
# Importer — gist flow
# ---------------------------------------------------------------------------


class TestGitHubImporterGist:
    @pytest.mark.asyncio
    async def test_gist_stores_author_and_updated_at(self, db_session: object) -> None:
        gist_bytes = json.dumps(_GIST_BLOB).encode()

        mock_response = MagicMock()
        mock_response.content = gist_bytes
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value=_GIST_BLOB)
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, return_value=mock_response)

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            d = GitHubImporter()
            entry_id, snapshot_id = await d.deposit(
                db_session,  # type: ignore[arg-type]
                "https://gist.github.com/karpathy/abc123",
                "test-operator",
                [],
            )

        from particles.corpus.store import get_entry, get_snapshot

        entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert entry is not None and snap is not None
        assert entry.source_type == "GITHUB_GIST"
        assert snap.author_id == "github:karpathy"
        published = snap.content_published_at
        assert published is not None
        assert published.replace(tzinfo=None) == datetime(2026, 2, 20, 10, 30)

    @pytest.mark.asyncio
    async def test_gist_embeds_fetched_comments_in_envelope(self, db_session: object) -> None:
        """REST deposit must fetch /gists/{id}/comments and store it in the envelope."""
        gist_resp = MagicMock()
        gist_resp.status_code = 200
        gist_resp.json = MagicMock(return_value=_GIST_BLOB)
        gist_resp.raise_for_status = MagicMock()
        gist_resp.headers = {}

        comments_resp = MagicMock()
        comments_resp.status_code = 200
        comments_resp.headers = {}  # no Link → single page, stop after this fetch
        comments_resp.json = MagicMock(
            return_value=[
                {
                    "user": {"login": "alice"},
                    "body": "Nice!",
                    "created_at": "2026-02-21T00:00:00Z",
                    "updated_at": "2026-02-21T00:00:00Z",
                },
                {
                    "user": {"login": "bob"},
                    "body": "Same here.",
                    "created_at": "2026-02-22T00:00:00Z",
                    "updated_at": "2026-02-22T00:00:00Z",
                },
            ]
        )
        comments_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        # First call: /gists/{id}. Second call: /gists/{id}/comments.
        set_capped_responses(mock_client, side_effect=[gist_resp, comments_resp])

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            _, snapshot_id = await GitHubImporter().deposit(
                db_session,  # type: ignore[arg-type]
                "https://gist.github.com/karpathy/abc123",
                "test-operator",
                [],
            )

        from particles.corpus.deposit import load_blob
        from particles.corpus.store import get_snapshot

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        envelope = json.loads(load_blob(snap.content_hash))
        assert isinstance(envelope.get("comments"), list)
        logins = [c["user"]["login"] for c in envelope["comments"]]
        assert logins == ["alice", "bob"]
        assert envelope["comments"][0]["body"] == "Nice!"
        # Comments fetch failure must not block the deposit (covered separately,
        # but assert the call sequence here).
        assert mock_client.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_gist_paginates_comments(self, db_session: object) -> None:
        """When a page has a `Link: …; rel="next"` header, the importer
        follows it to fetch additional pages."""
        gist_resp = MagicMock()
        gist_resp.status_code = 200
        gist_resp.json = MagicMock(return_value=_GIST_BLOB)
        gist_resp.raise_for_status = MagicMock()
        gist_resp.headers = {}

        def _page(start: int, count: int, link: str | None) -> MagicMock:
            r = MagicMock()
            r.status_code = 200
            r.headers = {"Link": link} if link else {}
            r.json = MagicMock(
                return_value=[
                    {
                        "user": {"login": f"user{start + i}"},
                        "body": f"comment {start + i}",
                        "created_at": "2026-02-21T00:00:00Z",
                        "updated_at": "2026-02-21T00:00:00Z",
                    }
                    for i in range(count)
                ]
            )
            return r

        next_link = '<https://api.github.com/gists/abc123/comments?per_page=100&page=2>; rel="next"'
        page1 = _page(1, 100, link=next_link)
        page2 = _page(101, 23, link=None)  # no next → loop stops
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, side_effect=[gist_resp, page1, page2])

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            _, snapshot_id = await GitHubImporter().deposit(
                db_session,  # type: ignore[arg-type]
                "https://gist.github.com/karpathy/abc123",
                "test-operator",
                [],
            )

        from particles.corpus.deposit import load_blob
        from particles.corpus.store import get_snapshot

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        envelope = json.loads(load_blob(snap.content_hash))  # type: ignore[union-attr]
        assert len(envelope["comments"]) == 123
        # 1 body + 2 comment pages = 3 GETs total
        assert mock_client.stream.call_count == 3
        # Page 1 URL is the initial one; page 2 URL came from the Link header.
        # get_capped opens client.stream("GET", url) — the URL is the 2nd arg.
        urls = [str(call.args[1]) for call in mock_client.stream.call_args_list]
        assert any("per_page=100" in u for u in urls)
        assert any("page=2" in u for u in urls)

    @pytest.mark.asyncio
    async def test_gist_comments_failure_does_not_break_deposit(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))
        gist_resp = MagicMock()
        gist_resp.status_code = 200
        gist_resp.json = MagicMock(return_value=_GIST_BLOB)
        gist_resp.raise_for_status = MagicMock()

        comments_resp = MagicMock()
        comments_resp.status_code = 500  # API hiccup — must not block deposit

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        # gist body OK; then comments returns 500 on every retry.
        set_capped_responses(
            mock_client, side_effect=[gist_resp, comments_resp, comments_resp, comments_resp]
        )

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            _, snapshot_id = await GitHubImporter().deposit(
                db_session,  # type: ignore[arg-type]
                "https://gist.github.com/karpathy/abc123",
                "test-operator",
                [],
            )

        from particles.corpus.deposit import load_blob
        from particles.corpus.store import get_snapshot

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        envelope = json.loads(load_blob(snap.content_hash))
        assert envelope.get("comments") == []  # graceful degradation


# ---------------------------------------------------------------------------
# Importer — repo flow
# ---------------------------------------------------------------------------


class TestGitHubImporterRepo:
    @pytest.mark.asyncio
    async def test_repo_blob_fetch_decodes_base64(self, db_session: object) -> None:
        readme_bytes = b"# nanoGPT\n\nA tiny GPT."
        contents_resp = MagicMock()
        contents_resp.status_code = 200
        contents_resp.json = MagicMock(
            return_value={
                "name": "README.md",
                "path": "README.md",
                "sha": "deadbeef",
                "encoding": "base64",
                "content": base64.b64encode(readme_bytes).decode(),
            }
        )
        contents_resp.raise_for_status = MagicMock()

        commits_resp = MagicMock()
        commits_resp.status_code = 200
        commits_resp.json = MagicMock(
            return_value=[
                {
                    "author": {"login": "karpathy"},
                    "commit": {
                        "committer": {"date": "2026-03-05T12:00:00Z"},
                    },
                }
            ]
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        # Two calls per deposit: contents, then commits
        set_capped_responses(mock_client, side_effect=[contents_resp, commits_resp])

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            d = GitHubImporter()
            entry_id, snapshot_id = await d.deposit(
                db_session,  # type: ignore[arg-type]
                "https://github.com/karpathy/nanoGPT/blob/master/README.md",
                "test-operator",
                [],
            )

        from particles.corpus.deposit import load_blob
        from particles.corpus.store import get_entry, get_snapshot

        entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert entry is not None and snap is not None
        assert entry.source_type == "GITHUB_REPO"
        assert snap.author_id == "github:karpathy"
        published = snap.content_published_at
        assert published is not None
        assert published.replace(tzinfo=None) == datetime(2026, 3, 5, 12, 0)
        assert load_blob(snap.content_hash) == readme_bytes

    @pytest.mark.asyncio
    async def test_repo_falls_back_to_owner_when_commit_author_missing(
        self, db_session: object
    ) -> None:
        readme_bytes = b"# repo"
        contents_resp = MagicMock()
        contents_resp.status_code = 200
        contents_resp.json = MagicMock(
            return_value={
                "name": "README.md",
                "path": "README.md",
                "encoding": "base64",
                "content": base64.b64encode(readme_bytes).decode(),
            }
        )
        contents_resp.raise_for_status = MagicMock()

        commits_resp = MagicMock()
        commits_resp.status_code = 200
        commits_resp.json = MagicMock(return_value=[])  # no commits returned

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, side_effect=[contents_resp, commits_resp])

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            d = GitHubImporter()
            _, snapshot_id = await d.deposit(
                db_session,  # type: ignore[arg-type]
                "https://github.com/anonymous-owner/somerepo",
                "test-operator",
                [],
            )

        from particles.corpus.store import get_snapshot

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        # Owner-fallback when API doesn't return an author
        assert snap.author_id == "github:anonymous-owner"


# ---------------------------------------------------------------------------
# Extractor accepts() routing
# ---------------------------------------------------------------------------


class TestExtractorAccepts:
    def test_repo_extractor_routes_correctly(self) -> None:
        e = GitHubRepoExtractor()
        assert e.accepts("GITHUB_REPO")
        assert not e.accepts("GITHUB_GIST")
        assert not e.accepts("GITHUB_PAGES")
        assert not e.accepts("WEB_PAGE")

    def test_gist_extractor_routes_correctly(self) -> None:
        e = GitHubGistExtractor()
        assert e.accepts("GITHUB_GIST")
        assert not e.accepts("GITHUB_REPO")

    def test_pages_extractor_routes_correctly(self) -> None:
        e = GitHubPagesExtractor()
        assert e.accepts("GITHUB_PAGES")
        assert not e.accepts("GITHUB_REPO")


# ---------------------------------------------------------------------------
# Extractor extract() — subject injection
# ---------------------------------------------------------------------------


def _mock_candidate() -> MagicMock:
    c = MagicMock()
    c.subjects = []
    c.content = "Some claim."
    c.confidence_value = 0.8
    c.uncertainty_nature = UncertaintyNature.EPISTEMIC
    return c


class TestGistExtractor:
    @pytest.mark.asyncio
    async def test_extract_injects_author_and_description(self) -> None:
        content = json.dumps(_GIST_BLOB).encode()
        candidate = _mock_candidate()
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        assert len(result.candidates) == 1
        subjects = result.candidates[0].subjects
        assert "github:karpathy" in subjects
        assert "Quick notes on PyTorch attention" in subjects

    @pytest.mark.asyncio
    async def test_extract_skips_long_description_as_subject(self) -> None:
        blob = dict(_GIST_BLOB)
        blob["description"] = "x" * 200  # exceeds 120 char cap
        content = json.dumps(blob).encode()
        candidate = _mock_candidate()
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        subjects = result.candidates[0].subjects
        assert "github:karpathy" in subjects
        assert not any(s.startswith("x" * 50) for s in subjects)

    @pytest.mark.asyncio
    async def test_extract_bad_json_returns_quality_note(self) -> None:
        result = await GitHubGistExtractor().extract(_make_snapshot(), b"not json")
        assert result.candidates == []
        assert any("JSON parse error" in n for n in result.quality_notes)

    @pytest.mark.asyncio
    async def test_extract_renders_comments_in_llm_text(self) -> None:
        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {"user": {"login": "alice"}, "body": "I tried this with PyTorch 2.5."},
            {"user": {"login": "bob"}, "body": "Same here."},
        ]
        content = json.dumps(blob).encode()
        seen: dict[str, str] = {}

        async def capture(text: str) -> tuple[list[object], list[str], bool]:
            seen["text"] = text
            return ([], [], False)

        with patch("particles.extraction.incremental._call_llm", AsyncMock(side_effect=capture)):
            await GitHubGistExtractor().extract(_make_snapshot(), content)

        # Reddit-style inline rendering: gh/{login}: <body>
        assert "gh/alice: I tried this with PyTorch 2.5." in seen["text"]
        assert "gh/bob: Same here." in seen["text"]
        # Owner is also rendered with the gh/ prefix so the LLM picks it up
        # when the LLM doesn't otherwise reference it.
        assert "gh/karpathy" in seen["text"]

    @pytest.mark.asyncio
    async def test_extract_rewrites_gh_subjects_to_github(self) -> None:
        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {"user": {"login": "alice"}, "body": "I tried this."},
        ]
        content = json.dumps(blob).encode()
        candidate = _mock_candidate()
        candidate.subjects = ["gh/alice", "gh/bob"]  # what the LLM would emit
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        subjects = result.candidates[0].subjects
        assert "github:alice" in subjects
        assert "github:bob" in subjects
        # gh/-prefixed entries must not leak through to storage
        assert not any(s.startswith("gh/") for s in subjects)

    @pytest.mark.asyncio
    async def test_extract_rewrites_bare_login_to_github(self) -> None:
        """The LLM is shown `gh/{login}: <body>` but occasionally strips
        the `gh/` prefix and emits a bare login as a subject. The bare
        login must still be canonicalised to `github:{login}` so the
        Obsidian exporter routes them under `github.com/{login}.md`."""
        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {"user": {"login": "alice"}, "body": "I tried this."},
        ]
        content = json.dumps(blob).encode()
        candidate = _mock_candidate()
        candidate.subjects = ["alice", "bob"]  # bare logins (no gh/)
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        subjects = result.candidates[0].subjects
        # `alice` was rendered as gh/alice in the comments chunk → rewritten.
        assert "github:alice" in subjects
        # `bob` was never seen as gh/bob → left alone.
        assert "bob" in subjects
        assert "github:bob" not in subjects

    @pytest.mark.asyncio
    async def test_extract_handles_no_comments_field(self) -> None:
        # Old-style envelope written before EXTRACTOR_VERSION_GIST=0.2.0 has
        # no "comments" key; the extractor must still work on those.
        blob = dict(_GIST_BLOB)
        blob.pop("comments", None)
        content = json.dumps(blob).encode()
        candidate = _mock_candidate()
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        assert "github:karpathy" in result.candidates[0].subjects

    @pytest.mark.asyncio
    async def test_content_normalizes_gh_login_to_at_mention(self) -> None:
        """LLM-emitted `gh/{login}` in particle content is rewritten to
        `@{login}` so the rendered claim reads naturally."""
        blob = dict(_GIST_BLOB)
        blob["comments"] = [{"user": {"login": "alice"}, "body": "I tried this."}]
        content = json.dumps(blob).encode()
        candidate = _mock_candidate()
        candidate.content = "gh/7TIN's project is available at https://example.com/7TIN/x"
        candidate.subjects = ["gh/7TIN"]
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        # Content: gh/7TIN rewritten to @7TIN; the URL path "7TIN/x" is left
        # alone because it doesn't carry the gh/ prefix.
        rewritten = result.candidates[0].content
        assert "gh/7TIN" not in rewritten
        assert "@7TIN" in rewritten
        assert "https://example.com/7TIN/x" in rewritten
        # Subject is still routed via github:{login} (handled by a separate pass).
        assert "github:7TIN" in result.candidates[0].subjects

    @pytest.mark.asyncio
    async def test_synthesized_particle_uses_at_mention_in_content(self) -> None:
        """Synthesized fallback particles use @{login} in their content for
        consistency with the LLM-content normalization."""
        from particles.config import get_config, reset_config

        reset_config()
        get_config().github.gist_synthesize_commenter_particles = True

        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {
                "user": {"login": "carol"},
                "body": (
                    "Tried this approach with mixed precision training and saw 40% speedup on A100."
                ),
            }
        ]
        content = json.dumps(blob).encode()
        candidate = _mock_candidate()
        candidate.content = "Attention is all you need."
        candidate.subjects = ["Attention"]
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        synthesized = [c for c in result.candidates if c is not candidate]
        assert len(synthesized) == 1
        assert synthesized[0].content.startswith("@carol commented")
        # The subject still uses the github: prefix for Obsidian routing.
        assert synthesized[0].subjects == ["github:carol"]

        reset_config()

    @pytest.mark.asyncio
    async def test_synthesis_off_by_default_skips_uncovered_commenter(self) -> None:
        """The default config has synthesis off; a substantive commenter who
        isn't covered by the LLM/attribution gets no fallback particle."""
        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {
                "user": {"login": "carol"},
                "body": (
                    "Tried this approach with mixed precision training "
                    "and observed roughly 40% speedup on my A100 GPU."
                ),
            }
        ]
        content = json.dumps(blob).encode()
        candidate = _mock_candidate()
        candidate.content = "Attention is all you need."
        candidate.subjects = ["Attention"]
        # Body chunk returns the candidate; comments chunk returns nothing.
        # That way we get a single LLM-derived candidate and can assert that
        # no synthesized fallback was appended.
        responses: list[tuple[list[object], list[str], bool]] = [
            ([candidate], [], False),
            ([], [], False),
        ]

        async def fake_llm(_text: str) -> tuple[list[object], list[str], bool]:
            return responses.pop(0) if responses else ([], [], False)

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=fake_llm),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        all_subjects = {s for c in result.candidates for s in c.subjects}
        assert "github:carol" not in all_subjects
        assert len(result.candidates) == 1

    @pytest.mark.asyncio
    async def test_attribution_tags_paraphrased_claim(self) -> None:
        """LLM-paraphrased claim that still shares content tokens with a comment
        gets the commenter as an auxiliary subject — even when the LLM didn't
        tag gh/{login}."""
        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {
                "user": {"login": "alice"},
                "body": (
                    "I tried AdamW with cosine decay schedule on the OpenWebText "
                    "corpus and it converged faster than vanilla Adam."
                ),
            },
            {
                "user": {"login": "bob"},
                "body": "Great post!",  # pleasantry — no content tokens
            },
        ]
        content = json.dumps(blob).encode()
        # LLM paraphrases alice's claim; doesn't tag her.
        candidate = _mock_candidate()
        candidate.content = "AdamW with cosine decay converges faster than Adam on OpenWebText."
        candidate.subjects = ["AdamW"]  # topical subject only
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        subjects = result.candidates[0].subjects
        assert "github:alice" in subjects  # attributed by overlap
        assert "github:bob" not in subjects  # pleasantry → no attribution
        assert "AdamW" in subjects  # topical subject preserved

    @pytest.mark.asyncio
    async def test_attribution_skips_when_candidate_lacks_content_tokens(self) -> None:
        """Very short / generic candidates don't get spurious attribution."""
        blob = dict(_GIST_BLOB)
        blob["comments"] = [{"user": {"login": "alice"}, "body": "I tried this with PyTorch 2.5."}]
        content = json.dumps(blob).encode()
        candidate = _mock_candidate()
        candidate.content = "It works."  # no content tokens after stopword filter
        candidate.subjects = []
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        assert "github:alice" not in result.candidates[0].subjects

    @pytest.mark.asyncio
    async def test_synthesis_creates_particle_for_uncovered_commenter(self) -> None:
        """A substantive commenter who isn't tagged by the LLM or by overlap
        attribution still gets a synthesized fallback particle — when the
        opt-in synthesis flag is enabled."""
        from particles.config import get_config, reset_config

        reset_config()
        get_config().github.gist_synthesize_commenter_particles = True

        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {
                "user": {"login": "carol"},
                "body": (
                    "Tried this approach with mixed precision training "
                    "and observed roughly 40% speedup on my A100 GPU."
                ),
            },
            {"user": {"login": "dave"}, "body": "Thanks!"},  # below substantive threshold
        ]
        content = json.dumps(blob).encode()
        # LLM returns one candidate that does NOT overlap carol's body — so
        # neither LLM-tagging nor attribution catches her.
        candidate = _mock_candidate()
        candidate.content = "Attention is all you need."
        candidate.subjects = ["Attention"]
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        all_subjects = {s for c in result.candidates for s in c.subjects}
        assert "github:carol" in all_subjects  # synthesized fallback
        assert "github:dave" not in all_subjects  # pleasantry — no subject
        # The synthesis adds a particle whose content quotes the comment.
        synthesized = [c for c in result.candidates if c is not candidate]
        assert len(synthesized) == 1
        assert "carol" in synthesized[0].content
        assert "mixed precision training" in synthesized[0].content

        reset_config()

    @pytest.mark.asyncio
    async def test_synthesis_skips_commenter_already_covered_by_attribution(self) -> None:
        """If LLM/attribution already produced a subject for a commenter, no
        duplicate synthesized particle is added — even when synthesis is on."""
        from particles.config import get_config, reset_config

        reset_config()
        get_config().github.gist_synthesize_commenter_particles = True

        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {
                "user": {"login": "alice"},
                "body": "I tried AdamW with cosine decay on OpenWebText.",
            }
        ]
        content = json.dumps(blob).encode()
        # LLM-emitted candidate that overlap will attribute to alice.
        candidate = _mock_candidate()
        candidate.content = "AdamW with cosine decay performs well on OpenWebText."
        candidate.subjects = ["AdamW"]
        # Body chunk returns the candidate; comments chunk returns nothing.
        responses: list[tuple[list[object], list[str], bool]] = [
            ([candidate], [], False),
            ([], [], False),
        ]

        async def fake_llm(_text: str) -> tuple[list[object], list[str], bool]:
            return responses.pop(0) if responses else ([], [], False)

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=fake_llm),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        # Exactly one particle (the LLM's); no synthesized fallback for alice.
        assert len(result.candidates) == 1
        assert "github:alice" in result.candidates[0].subjects

        reset_config()

    @pytest.mark.asyncio
    async def test_synthesis_disabled_via_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the synthesis flag is off, uncovered commenters get no particle."""
        from particles.config import get_config, reset_config

        reset_config()
        get_config().github.gist_synthesize_commenter_particles = False

        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {
                "user": {"login": "carol"},
                "body": "Tried this with mixed precision and saw 40% speedup on A100 GPU.",
            }
        ]
        content = json.dumps(blob).encode()
        candidate = _mock_candidate()
        candidate.content = "Attention is all you need."
        candidate.subjects = ["Attention"]
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        all_subjects = {s for c in result.candidates for s in c.subjects}
        assert "github:carol" not in all_subjects
        reset_config()

    @pytest.mark.asyncio
    async def test_attribution_does_not_duplicate_existing_subject(self) -> None:
        """If the LLM already tagged gh/alice → github:alice via _rewrite_gh_subjects,
        the overlap pass must not append a duplicate."""
        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {
                "user": {"login": "alice"},
                "body": "I tried gradient checkpointing with batch size 32 and it worked.",
            }
        ]
        content = json.dumps(blob).encode()
        candidate = _mock_candidate()
        candidate.content = "gradient checkpointing works with batch size 32"
        candidate.subjects = ["gh/alice"]  # LLM tagged it
        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)
        subjects = result.candidates[0].subjects
        assert subjects.count("github:alice") == 1
        assert "gh/alice" not in subjects  # rewritten

    @pytest.mark.asyncio
    async def test_chunked_extraction_when_comments_exceed_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When rendered comments exceed the single-call threshold, the
        comments are split into multiple LLM calls in addition to the body."""
        from particles.config import get_config, reset_config

        reset_config()
        ext = get_config().extraction
        ext.single_call_threshold_chars = 200
        ext.comment_chunk_chars = 200
        ext.max_llm_calls_per_source = 8
        get_config().github.gist_comment_body_limit = 100

        # 6 comments × ~80 chars each → > 200 threshold; chunks of 2 each.
        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {
                "user": {"login": f"user{i}"},
                "body": f"This is a substantive comment number {i} with several words.",
            }
            for i in range(6)
        ]
        content = json.dumps(blob).encode()

        call_log: list[str] = []

        async def fake_llm(text: str) -> tuple[list[object], list[str], bool]:
            call_log.append(text)
            return ([], [], False)

        with patch("particles.extraction.incremental._call_llm", AsyncMock(side_effect=fake_llm)):
            await GitHubGistExtractor().extract(_make_snapshot(), content)

        # Body call + at least 2 comment-chunk calls
        assert len(call_log) >= 3
        # Body chunk has gist body but no "## Comments" header
        body_calls = [c for c in call_log if "Gist by gh/karpathy" in c]
        assert len(body_calls) == 1
        assert "user0" not in body_calls[0]
        # Comment chunks include the context header
        comment_calls = [c for c in call_log if c not in body_calls]
        assert all("Context: comments on a gist" in c for c in comment_calls)
        all_chunk_text = "\n".join(comment_calls)
        assert "gh/user0:" in all_chunk_text
        assert "gh/user5:" in all_chunk_text

        reset_config()

    @pytest.mark.asyncio
    async def test_body_and_comments_are_always_separate_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even a small gist gets body and comments as separate chunks so
        that a future re-deposit with unchanged body but new comments can
        carry forward the body LLM call."""
        from particles.config import get_config, reset_config

        reset_config()
        get_config().extraction.single_call_threshold_chars = 100000

        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {"user": {"login": "alice"}, "body": "Short comment."},
            {"user": {"login": "bob"}, "body": "Another short one."},
        ]
        content = json.dumps(blob).encode()

        call_log: list[str] = []

        async def fake_llm(text: str) -> tuple[list[object], list[str], bool]:
            call_log.append(text)
            return ([], [], False)

        with patch("particles.extraction.incremental._call_llm", AsyncMock(side_effect=fake_llm)):
            await GitHubGistExtractor().extract(_make_snapshot(), content)

        # Two LLM calls: body alone + comments alone
        assert len(call_log) == 2
        body = next(c for c in call_log if "Gist by gh/karpathy" in c)
        comments = next(c for c in call_log if "gh/alice:" in c)
        assert "gh/alice:" not in body
        assert "Gist by gh/karpathy" not in comments
        assert "gh/bob:" in comments

        reset_config()

    @pytest.mark.asyncio
    async def test_chunked_extraction_respects_max_llm_calls_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hard ceiling on LLM calls bounds worst-case cost."""
        from particles.config import get_config, reset_config

        reset_config()
        ext = get_config().extraction
        ext.single_call_threshold_chars = 100
        ext.comment_chunk_chars = 50
        ext.max_llm_calls_per_source = 3  # body + 2 chunks max
        get_config().github.gist_comment_body_limit = 100

        blob = dict(_GIST_BLOB)
        blob["comments"] = [
            {"user": {"login": f"user{i}"}, "body": f"Substantive comment number {i} here."}
            for i in range(10)
        ]
        content = json.dumps(blob).encode()

        call_log: list[str] = []

        async def fake_llm(text: str) -> tuple[list[object], list[str], bool]:
            call_log.append(text)
            return ([], [], False)

        with patch("particles.extraction.incremental._call_llm", AsyncMock(side_effect=fake_llm)):
            result = await GitHubGistExtractor().extract(_make_snapshot(), content)

        # 3 LLM calls (the cap), with at least one CHUNK_TRUNCATION note
        assert len(call_log) == 3
        assert any("CHUNK_TRUNCATION" in n for n in result.quality_notes)

        reset_config()

    @pytest.mark.asyncio
    async def test_extract_truncates_long_comment_bodies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARTICLES_CONFIG", "")  # no override
        long_body = "x" * 5000
        blob = dict(_GIST_BLOB)
        blob["comments"] = [{"user": {"login": "alice"}, "body": long_body}]
        content = json.dumps(blob).encode()
        seen: dict[str, str] = {}

        async def capture(text: str) -> tuple[list[object], list[str], bool]:
            seen["text"] = text
            return ([], [], False)

        with patch("particles.extraction.incremental._call_llm", AsyncMock(side_effect=capture)):
            await GitHubGistExtractor().extract(_make_snapshot(), content)

        # Default gist_comment_body_limit is 1000; LLM text must not carry
        # the full 5000-char body.
        assert "x" * 1000 in seen["text"]
        assert "x" * 1500 not in seen["text"]


class TestPagesExtractor:
    @pytest.mark.asyncio
    async def test_extract_injects_author_from_snapshot(self) -> None:
        html = b"<html><body><p>Important claim about transformers.</p></body></html>"
        candidate = _mock_candidate()
        snapshot = _make_snapshot(author_id="github:karpathy")
        # Pages extractor calls _call_llm via _llm_extract_with_subjects in
        # the github._shared module (short-body path) — that's where the
        # binding lives, so that's where we patch.
        with patch(
            "particles.extraction.github._shared._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubPagesExtractor().extract(snapshot, html)
        assert "github:karpathy" in result.candidates[0].subjects

    @pytest.mark.asyncio
    async def test_extract_handles_empty_content(self) -> None:
        result = await GitHubPagesExtractor().extract(_make_snapshot(), b"")
        assert result.candidates == []
        assert any("Empty content" in n for n in result.quality_notes)


class TestRepoExtractor:
    @pytest.mark.asyncio
    async def test_extract_injects_repo_and_owner(self) -> None:
        # the Engine pipeline passes the entry URL via
        # the ``entry_uri_r`` kwarg; the extractor parses owner/repo from it
        # without reading the store (no DB session needed).
        content = b"# nanoGPT\n\nA tiny GPT."
        candidate = _mock_candidate()
        # Repo extractor calls _call_llm via _llm_extract_with_subjects, which
        # lives in github._shared — that's where the binding lives.
        with patch(
            "particles.extraction.github._shared._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubRepoExtractor().extract(
                _make_snapshot(),
                content,
                entry_uri_r="https://github.com/karpathy/nanoGPT/blob/master/README.md",
            )

        subjects = result.candidates[0].subjects
        assert "nanoGPT" in subjects
        assert "github:karpathy" in subjects

    @pytest.mark.asyncio
    async def test_extract_handles_missing_uri_gracefully(self) -> None:
        # No entry_uri_r supplied (pure-Client / unknown source) — extractor
        # still calls the LLM but the injected repo/owner extras stay empty.
        candidate = _mock_candidate()
        with patch(
            "particles.extraction.github._shared._call_llm",
            AsyncMock(return_value=([candidate], [], False)),
        ):
            result = await GitHubRepoExtractor().extract(_make_snapshot(), b"some content")
        assert result.candidates[0].subjects == []


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestTransientRetry:
    """5xx retry + ValueError mapping on exhaustion."""

    def _failing_resp(self, status: int = 502) -> MagicMock:
        r = MagicMock()
        r.status_code = status
        r.headers = {}
        return r

    def _gist_success_resp(self) -> MagicMock:
        r = MagicMock()
        r.status_code = 200
        r.content = json.dumps(_GIST_BLOB).encode()
        r.json = MagicMock(return_value=_GIST_BLOB)
        r.headers = {}
        return r

    def _empty_comments_resp(self) -> MagicMock:
        r = MagicMock()
        r.status_code = 200
        r.json = MagicMock(return_value=[])
        r.headers = {}
        return r

    @pytest.mark.asyncio
    async def test_gist_retries_then_succeeds(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(
            mock_client,
            side_effect=[
                self._failing_resp(502),
                self._failing_resp(503),
                self._gist_success_resp(),
                self._empty_comments_resp(),
            ],
        )

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            _, snapshot_id = await GitHubImporter().deposit(
                db_session,  # type: ignore[arg-type]
                "https://gist.github.com/karpathy/abc123",
                "test-operator",
                [],
            )

        # 3 body attempts (2 retries) + 1 comments fetch
        assert mock_client.stream.call_count == 4
        from particles.corpus.store import get_snapshot

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        assert snap.author_id == "github:karpathy"

    @pytest.mark.asyncio
    async def test_repo_raises_value_error_after_retries_exhausted(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Gist 502-exhaustion has its own fallback (see TestGistGitFallback);
        # repo deposits have no fallback, so retries-exhausted surfaces as ValueError.
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, return_value=self._failing_resp(502))

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            with pytest.raises(ValueError, match="GitHub API unavailable"):
                await GitHubImporter().deposit(
                    db_session,  # type: ignore[arg-type]
                    "https://github.com/karpathy/nanoGPT",
                    "test-operator",
                    [],
                )

        assert mock_client.stream.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_404_is_not_retried(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, return_value=self._failing_resp(404))

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            with pytest.raises(ValueError, match="not found \\(404\\)"):
                await GitHubImporter().deposit(
                    db_session,  # type: ignore[arg-type]
                    "https://gist.github.com/karpathy/abc123",
                    "test-operator",
                    [],
                )

        assert mock_client.stream.call_count == 1

    @pytest.mark.asyncio
    async def test_403_gives_helpful_message(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        set_capped_responses(mock_client, return_value=self._failing_resp(403))

        with patch("particles.http.particles_client") as mock_ctx:
            mock_ctx.return_value = mock_client
            with pytest.raises(ValueError, match="forbidden \\(403\\).*rate limit"):
                await GitHubImporter().deposit(
                    db_session,  # type: ignore[arg-type]
                    "https://gist.github.com/karpathy/abc123",
                    "test-operator",
                    [],
                )


class TestGistGitFallback:
    """Git-clone fallback path when the REST API returns a stable 5xx."""

    @pytest.mark.asyncio
    async def test_fallback_triggered_after_502_exhaustion(
        self,
        db_session: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        fail_resp = MagicMock()
        fail_resp.status_code = 502
        set_capped_responses(mock_client, return_value=fail_resp)

        async def fake_clone(
            gist_id: str,
        ) -> tuple[dict[str, dict[str, str]], str | None]:
            return (
                {"big.md": {"content": "Lots of content."}},
                "2026-03-01T12:00:00+00:00",
            )

        with (
            patch("particles.http.particles_client") as mock_ctx,
            patch(
                "particles.ingest.importers.github._clone_gist_files",
                side_effect=fake_clone,
            ),
        ):
            mock_ctx.return_value = mock_client
            entry_id, snapshot_id = await GitHubImporter().deposit(
                db_session,  # type: ignore[arg-type]
                "https://gist.github.com/karpathy/abcdef",
                "test-operator",
                [],
            )

        from particles.corpus.deposit import load_blob
        from particles.corpus.store import get_entry, get_snapshot

        entry = await get_entry(db_session, entry_id)  # type: ignore[arg-type]
        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert entry is not None and snap is not None
        assert entry.source_type == "GITHUB_GIST"
        assert snap.author_id == "github:karpathy"

        envelope = json.loads(load_blob(snap.content_hash))
        assert envelope["_fallback"] == "git-clone"
        assert envelope["files"] == {"big.md": {"content": "Lots of content."}}
        assert envelope["owner"]["login"] == "karpathy"
        # Comments still fetched via REST in the fallback path; an empty list
        # is acceptable when the API stays unreachable.
        assert envelope["comments"] == []
        # 3 body attempts (1 initial + 2 retries), then fallback +
        # 3 comments attempts (also 502 in this test) = 6 total.
        assert mock_client.stream.call_count == 6

    @pytest.mark.asyncio
    async def test_clone_failure_surfaces_value_error(
        self,
        db_session: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        fail_resp = MagicMock()
        fail_resp.status_code = 502
        set_capped_responses(mock_client, return_value=fail_resp)

        async def boom(gist_id: str) -> tuple[dict[str, dict[str, str]], str | None]:
            raise ValueError("git clone failed for gist abcdef: repo not found")

        with (
            patch("particles.http.particles_client") as mock_ctx,
            patch(
                "particles.ingest.importers.github._clone_gist_files",
                side_effect=boom,
            ),
        ):
            mock_ctx.return_value = mock_client
            with pytest.raises(ValueError, match="git clone failed"):
                await GitHubImporter().deposit(
                    db_session,  # type: ignore[arg-type]
                    "https://gist.github.com/karpathy/abcdef",
                    "test-operator",
                    [],
                )

    @pytest.mark.asyncio
    async def test_clone_helper_surfaces_filenotfound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.ingest.importers.github import _clone_gist_files

        async def fake_exec(*args: object, **kwargs: object) -> object:
            raise FileNotFoundError("git")

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

        with pytest.raises(ValueError, match="git binary not found"):
            await _clone_gist_files("abc123", resolve=_stub_resolver(_PUBLIC_ADDR))


class TestGistCloneConnectPin:
    """the gist clone connects to an address this process vetted.

    A ``git clone`` never enters httpx, so ValidatingTransport cannot
    see it. ``http.curloptResolve`` is the git-side equivalent of
    ``curl --resolve``: the addresses are resolved and blocklist-checked
    in-process, then git is pinned to them, and git still verifies the
    certificate against the real hostname.
    """

    @pytest.mark.asyncio
    async def test_clone_argv_carries_the_pin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.ingest.importers.github import _clone_gist_files

        captured: list[list[str]] = []

        async def fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            captured.append(list(argv))
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        await _clone_gist_files("abc123", resolve=_stub_resolver(_PUBLIC_ADDR))

        clone_argv = captured[0]
        assert "-c" in clone_argv
        assert (
            clone_argv[clone_argv.index("-c") + 1]
            == f"http.curloptResolve=gist.github.com:443:{_PUBLIC_ADDR}"
        )
        # The pin is added beneath the pre-existing protocol pin, not instead
        # of it — the second invocation is the local `git log`, which is not
        # egress and carries no pin (class 3).
        assert clone_argv[clone_argv.index("-c") + 2 :][:1] == ["clone"]
        assert "-c" not in captured[1]

    @pytest.mark.asyncio
    async def test_multi_address_pin_is_comma_joined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.ingest.importers.github import _clone_gist_files

        captured: list[list[str]] = []

        async def fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            captured.append(list(argv))
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        await _clone_gist_files(
            "abc123", resolve=_stub_resolver("140.82.112.4", "140.82.113.4", "2606:50c0::153")
        )

        clone_argv = captured[0]
        assert clone_argv[clone_argv.index("-c") + 1] == (
            "http.curloptResolve=gist.github.com:443:140.82.112.4,140.82.113.4,[2606:50c0::153]"
        )

    @pytest.mark.asyncio
    async def test_blocked_address_short_circuits_before_git(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rebinding-equivalent case: no subprocess is spawned at all."""
        from particles.ingest.importers.github import _clone_gist_files

        spawned = False

        async def fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            nonlocal spawned
            spawned = True
            return MagicMock()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        with pytest.raises(UnsafeUrlError, match="169.254.169.254"):
            await _clone_gist_files("abc123", resolve=_stub_resolver("169.254.169.254"))
        assert not spawned

    @pytest.mark.asyncio
    async def test_split_answer_is_all_or_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One public + one private address is rejected, not filtered down."""
        from particles.ingest.importers.github import _clone_gist_files

        spawned = False

        async def fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            nonlocal spawned
            spawned = True
            return MagicMock()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        with pytest.raises(UnsafeUrlError):
            await _clone_gist_files("abc123", resolve=_stub_resolver(_PUBLIC_ADDR, "192.168.1.7"))
        assert not spawned

    @pytest.mark.asyncio
    async def test_unresolvable_host_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.ingest.importers.github import _clone_gist_files

        spawned = False

        async def fake_exec(*argv: str, **kwargs: object) -> MagicMock:
            nonlocal spawned
            spawned = True
            return MagicMock()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        with pytest.raises(UnsafeUrlError, match="did not resolve"):
            await _clone_gist_files("abc123", resolve=_stub_resolver())
        assert not spawned


class TestRegistryIntegration:
    def test_three_github_extractors_in_registry(self) -> None:
        from particles.extraction.registry import get_extractors

        ids = [p.EXTRACTOR_ID for p in get_extractors()]
        assert "github-repo-extractor" in ids
        assert "github-gist-extractor" in ids
        assert "github-pages-extractor" in ids

    def test_github_extractors_before_general(self) -> None:
        from particles.extraction.registry import get_extractors

        ids = [p.EXTRACTOR_ID for p in get_extractors()]
        general_idx = ids.index("general-extractor")
        for github_id in (
            "github-repo-extractor",
            "github-gist-extractor",
            "github-pages-extractor",
        ):
            assert ids.index(github_id) < general_idx

    def test_github_importer_in_registry(self) -> None:
        from particles.ingest.importers.registry import get_importers

        types = {type(d).__name__ for d in get_importers()}
        assert "GitHubImporter" in types
