"""Tests for deposit URL routing (particles/corpus/deposit.py).

These verify that URLs are routed to the correct source type without
making real network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.extraction.numista import SOURCE_TYPE_COIN, SOURCE_TYPE_LISTING
from tests._capped_http import set_capped_responses

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _aiter_bytes(chunks: list[bytes]) -> object:
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
# Routing tests
# ---------------------------------------------------------------------------


class TestDepositUrlRouting:
    """deposit_url() should route to the correct source type for each URL pattern."""

    @pytest.mark.asyncio
    async def test_numista_coin_url_routes_to_coin(
        self, db_session: object, tmp_path: Path
    ) -> None:
        from sqlalchemy import select

        from particles.corpus.deposit import deposit_url
        from particles.corpus.store import CorpusEntryRow

        coin_json = json.dumps({"id": 8562, "title": "1 Pfennig", "min_year": 1948}).encode()
        mock_resp = _make_http_response(coin_json)

        with patch("particles.http.particles_client") as mock_client_ctx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            set_capped_responses(mock_client, return_value=mock_resp)
            mock_client_ctx.return_value = mock_client

            with patch.dict("os.environ", {"NUMISTA_API_KEY": "test-key"}):
                entry_id, _ = await deposit_url(
                    db_session,  # type: ignore[arg-type]
                    "https://en.numista.com/catalogue/pieces8562.html",
                    deposited_by="test",
                )

        session = db_session  # type: ignore[assignment]
        result = await session.execute(  # type: ignore[union-attr]
            select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
        )
        entry = result.scalar_one()
        assert entry.source_type == SOURCE_TYPE_COIN

    @pytest.mark.asyncio
    async def test_numista_listing_url_routes_to_listing_html(
        self, db_session: object, tmp_path: Path
    ) -> None:
        from sqlalchemy import select

        from particles.corpus.deposit import deposit_url
        from particles.corpus.store import CorpusEntryRow

        listing_html = b"<html><body><div class='description_piece'></div></body></html>"
        mock_resp = _make_http_response(listing_html)

        with patch("particles.http.particles_client") as mock_client_ctx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            set_capped_responses(mock_client, return_value=mock_resp)
            mock_client_ctx.return_value = mock_client

            entry_id, _ = await deposit_url(
                db_session,  # type: ignore[arg-type]
                "https://en.numista.com/catalogue/index.php?e=ddr&st=1-2&cat=y",
                deposited_by="test",
            )

        session = db_session  # type: ignore[assignment]
        result = await session.execute(  # type: ignore[union-attr]
            select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
        )
        entry = result.scalar_one()
        assert entry.source_type == SOURCE_TYPE_LISTING

    @pytest.mark.asyncio
    async def test_short_numista_coin_url_routes_to_coin(self, db_session: object) -> None:
        from sqlalchemy import select

        from particles.corpus.deposit import deposit_url
        from particles.corpus.store import CorpusEntryRow

        coin_json = json.dumps({"id": 8562, "title": "1 Pfennig"}).encode()
        mock_resp = _make_http_response(coin_json)

        with patch("particles.http.particles_client") as mock_client_ctx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            set_capped_responses(mock_client, return_value=mock_resp)
            mock_client_ctx.return_value = mock_client

            with patch.dict("os.environ", {"NUMISTA_API_KEY": "test-key"}):
                entry_id, _ = await deposit_url(
                    db_session,  # type: ignore[arg-type]
                    "https://en.numista.com/8562",
                    deposited_by="test",
                )

        session = db_session  # type: ignore[assignment]
        result = await session.execute(  # type: ignore[union-attr]
            select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
        )
        entry = result.scalar_one()
        assert entry.source_type == SOURCE_TYPE_COIN


# ---------------------------------------------------------------------------
# URL fragment stripping (#anchor)
# ---------------------------------------------------------------------------


class TestDepositUrlFragmentStripping:
    """``deposit_url`` strips ``#anchor`` from the URL before any importer
    lookup, fetch, or storage. HTTP servers never receive the fragment,
    and keeping it in ``uri_r`` would create spurious one-entry-per-
    anchor noise for operators who paste section links.
    """

    @pytest.mark.asyncio
    async def test_fragment_stripped_from_uri_r(
        self, db_session: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from sqlalchemy import select

        from particles.corpus.deposit import deposit_url
        from particles.corpus.store import CorpusEntryRow

        mock_resp = _make_http_response(b"<html><body>x</body></html>")

        with patch("particles.http.particles_client") as mock_client_ctx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            set_capped_responses(mock_client, return_value=mock_resp)
            mock_client.stream = MagicMock(return_value=_stream_cm(mock_resp))
            mock_client_ctx.return_value = mock_client

            caplog.set_level(logging.INFO, logger="particles.corpus.deposit")
            entry_id, _ = await deposit_url(
                db_session,  # type: ignore[arg-type]
                "https://en.wikipedia.org/wiki/Foo#Section",
                deposited_by="test",
            )

        session = db_session  # type: ignore[assignment]
        result = await session.execute(  # type: ignore[union-attr]
            select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
        )
        entry = result.scalar_one()
        # The fragment is gone from the stored URI.
        assert entry.uri_r == "https://en.wikipedia.org/wiki/Foo"
        # And the operator sees a log line naming the stripped fragment.
        assert any("Stripped fragment '#Section'" in rec.getMessage() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_url_without_fragment_unchanged(
        self, db_session: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from sqlalchemy import select

        from particles.corpus.deposit import deposit_url
        from particles.corpus.store import CorpusEntryRow

        mock_resp = _make_http_response(b"<html><body>y</body></html>")

        with patch("particles.http.particles_client") as mock_client_ctx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            set_capped_responses(mock_client, return_value=mock_resp)
            mock_client.stream = MagicMock(return_value=_stream_cm(mock_resp))
            mock_client_ctx.return_value = mock_client

            caplog.set_level(logging.INFO, logger="particles.corpus.deposit")
            entry_id, _ = await deposit_url(
                db_session,  # type: ignore[arg-type]
                "https://en.wikipedia.org/wiki/Foo",
                deposited_by="test",
            )

        session = db_session  # type: ignore[assignment]
        result = await session.execute(  # type: ignore[union-attr]
            select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
        )
        entry = result.scalar_one()
        assert entry.uri_r == "https://en.wikipedia.org/wiki/Foo"
        # No strip-message log line — there was nothing to strip.
        assert not any("Stripped fragment" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# DB round-trip for properties and subject_class
# ---------------------------------------------------------------------------


class TestStructuredPropertiesStorage:
    """properties and subject_class survive the ORM round-trip."""

    @pytest.mark.asyncio
    async def test_particle_properties_persisted(self, db_session: object) -> None:
        from particles.core.schema import Confidence, Particle, UncertaintyNature
        from particles.core.scoring.confidence import CalibrationSource
        from particles.core.status import Status
        from particles.store.particle_store import get_particle, insert_particle

        props = {"nmo:hasWeight": 0.75, "nuds:references": ["KM# 1"]}
        p = Particle(
            content="Test particle with properties.",
            confidence=Confidence(
                value=0.95, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
            ),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            status=Status.ACTIVE,
            properties=props,
        )

        session = db_session  # type: ignore[assignment]
        await insert_particle(session, p)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        retrieved = await get_particle(session, p.id)  # type: ignore[arg-type]
        assert retrieved is not None
        assert retrieved.properties is not None
        assert retrieved.properties["nmo:hasWeight"] == 0.75
        assert retrieved.properties["nuds:references"] == ["KM# 1"]

    @pytest.mark.asyncio
    async def test_particle_none_properties_persisted(self, db_session: object) -> None:
        from particles.core.schema import Confidence, Particle, UncertaintyNature
        from particles.core.scoring.confidence import CalibrationSource
        from particles.core.status import Status
        from particles.store.particle_store import get_particle, insert_particle

        p = Particle(
            content="Particle without properties.",
            confidence=Confidence(
                value=0.90, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
            ),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            status=Status.ACTIVE,
        )

        session = db_session  # type: ignore[assignment]
        await insert_particle(session, p)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        retrieved = await get_particle(session, p.id)  # type: ignore[arg-type]
        assert retrieved is not None
        assert retrieved.properties is None

    @pytest.mark.asyncio
    async def test_subject_class_persisted(self, db_session: object) -> None:
        from particles.core.schema import Subject
        from particles.store.subject_store import get_subject, insert_subject

        s = Subject(
            canonical_name="Aluminium",
            asserted_by="test",
            subject_class="nmo:Material",
        )
        session = db_session  # type: ignore[assignment]
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        retrieved = await get_subject(session, s.id)  # type: ignore[arg-type]
        assert retrieved is not None
        assert retrieved.subject_class == "nmo:Material"

    @pytest.mark.asyncio
    async def test_set_subject_class_updates(self, db_session: object) -> None:
        from particles.core.schema import Subject
        from particles.store.subject_store import get_subject, insert_subject, set_subject_class

        s = Subject(canonical_name="Berlin", asserted_by="test")
        session = db_session  # type: ignore[assignment]
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        await set_subject_class(session, s.id, "nmo:Mint")  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        retrieved = await get_subject(session, s.id)  # type: ignore[arg-type]
        assert retrieved is not None
        assert retrieved.subject_class == "nmo:Mint"


# ---------------------------------------------------------------------------
# Taxonomy definition file routing
# ---------------------------------------------------------------------------


class TestTaxonomyDefinitionFileRouting:
    """deposit_file() must stamp source_type=TAXONOMY_DEFINITION for taxonomy JSON."""

    @pytest.mark.asyncio
    async def test_taxonomy_json_routes_to_taxonomy_definition(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_file
        from particles.corpus.store import CorpusEntryRow

        taxonomy_path = tmp_path / "coins.json"
        taxonomy_path.write_text(
            json.dumps(
                {
                    "name": "Coins",
                    "version": "1.0.0",
                    "author": "test",
                    "tags": [{"tag": "coins"}],
                }
            )
        )
        entry_id, _ = await deposit_file(
            db_session,  # type: ignore[arg-type]
            taxonomy_path,
            deposited_by="test",
        )
        result = await db_session.execute(  # type: ignore[union-attr]
            select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
        )
        entry = result.scalar_one()
        assert entry.source_type == SourceType.TAXONOMY_DEFINITION.value

    @pytest.mark.asyncio
    async def test_non_taxonomy_json_falls_through(
        self, tmp_path: Path, db_session: object
    ) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_file
        from particles.corpus.store import CorpusEntryRow

        unrelated = tmp_path / "data.json"
        unrelated.write_text(json.dumps({"some": "blob", "without": "tags"}))
        entry_id, _ = await deposit_file(
            db_session,  # type: ignore[arg-type]
            unrelated,
            deposited_by="test",
        )
        result = await db_session.execute(  # type: ignore[union-attr]
            select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
        )
        entry = result.scalar_one()
        assert entry.source_type != SourceType.TAXONOMY_DEFINITION.value

    @pytest.mark.asyncio
    async def test_non_json_suffix_falls_through(self, tmp_path: Path, db_session: object) -> None:
        from sqlalchemy import select

        from particles.core.schema import SourceType
        from particles.corpus.deposit import deposit_file
        from particles.corpus.store import CorpusEntryRow

        # Same content but a .txt extension — shape-detection only fires on .json
        text_path = tmp_path / "coins.txt"
        text_path.write_text(
            json.dumps(
                {
                    "name": "Coins",
                    "version": "1.0.0",
                    "author": "test",
                    "tags": [{"tag": "coins"}],
                }
            )
        )
        entry_id, _ = await deposit_file(
            db_session,  # type: ignore[arg-type]
            text_path,
            deposited_by="test",
        )
        result = await db_session.execute(  # type: ignore[union-attr]
            select(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id)
        )
        entry = result.scalar_one()
        assert entry.source_type != SourceType.TAXONOMY_DEFINITION.value


class TestImageSourceTypeDetection:
    """standalone images route to the IMAGE source type."""

    def test_png_suffix_detected(self) -> None:
        from pathlib import Path

        from particles.core.schema import SourceType
        from particles.corpus.deposit import _detect_source_type

        assert _detect_source_type(None, None, Path("diagram.png")) == SourceType.IMAGE

    def test_image_content_type_detected(self) -> None:
        from particles.core.schema import SourceType
        from particles.corpus.deposit import _detect_source_type

        assert _detect_source_type("https://e/x", "image/jpeg", None) == SourceType.IMAGE

    def test_magic_bytes_win_over_extension(self) -> None:
        from pathlib import Path

        from particles.core.schema import SourceType
        from particles.corpus.deposit import _resolve_source_type

        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20  # PNG bytes, misleading .bin name
        assert _resolve_source_type(Path("mystery.bin"), png, None) == SourceType.IMAGE

    def test_non_image_falls_through(self) -> None:
        from pathlib import Path

        from particles.core.schema import SourceType
        from particles.corpus.deposit import _resolve_source_type

        assert _resolve_source_type(Path("notes.txt"), b"hello world", None) != SourceType.IMAGE


class TestBlobPathHexGuard:
    """F21: ``blob_path`` interpolates ``content_hash`` straight into the
    on-disk path, so it must reject anything that is not a 64-char lowercase
    SHA-256 hex digest (defence-in-depth against a future fetch-by-hash caller).
    ``save_blob`` / ``load_blob`` both route through ``blob_path`` and inherit
    the guard."""

    def test_valid_sha256_returns_sharded_path(self) -> None:
        from particles.corpus.deposit import blob_path

        digest = "a" * 64
        path = blob_path(digest)
        # hash[:2] shard prefix, then the full digest as the leaf.
        assert path.parent.name == "aa"
        assert path.name == digest

    @pytest.mark.parametrize(
        "bad",
        [
            "../../etc/passwd",  # path traversal
            "a" * 63,  # too short
            "a" * 65,  # too long
            "A" * 64,  # uppercase (SHA-256 hex is lowercase)
            "g" * 64,  # non-hex char
            "",  # empty
            "a" * 32 + "/" + "a" * 31,  # embedded separator
        ],
    )
    def test_non_hex_raises(self, bad: str) -> None:
        from particles.corpus.deposit import blob_path

        with pytest.raises(ValueError):
            blob_path(bad)

    def test_load_blob_inherits_guard(self) -> None:
        from particles.corpus.deposit import load_blob

        with pytest.raises(ValueError):
            load_blob("../../etc/passwd")


class TestRedditShareLinkRouting:
    """a Reddit `/s/` share link routes to the RedditImporter, not the
    generic web fallback (which would hit Reddit's Cloudflare bot-wall)."""

    def test_share_link_routes_to_reddit_importer(self) -> None:
        from particles.ingest.importers.reddit import RedditImporter
        from particles.ingest.importers.registry import get_importers

        url = "https://www.reddit.com/r/AIEval/s/2KHTCY0Bgq"
        accepting = [imp for imp in get_importers() if imp.accepts_url(url)]
        assert len(accepting) == 1
        assert isinstance(accepting[0], RedditImporter)
