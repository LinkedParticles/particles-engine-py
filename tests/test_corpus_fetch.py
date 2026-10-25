# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for §7.5 lazy re-fetch (particles/corpus/fetch.py).

The three-tier change-detection protocol:
  Tier 1: ETag / If-Modified-Since → 304 → REVISIT
  Tier 2: SHA-256 of fetched body → match → REVISIT, mismatch → RESPONSE
  Tier 3: force=True bypasses everything

An entry whose URI-R an importer accepts is fetched by that importer's
``refetch`` instead; TestImporterOwnedEntries covers that route.

The local tier (``file://`` URI-R) is covered by TestLocalTier at the
bottom: stat mtime as tier 1, SHA-256 as tier 2, and no network at all.

httpx is mocked via the documented ``patch("particles.http.particles_client")``
recipe from tests/AGENTS.md. No network IO; no real blob writes are needed
beyond what save_blob does into PARTICLES_BLOB_DIR (conftest points it at
/tmp/particles_test_blobs).
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.config import get_config
from particles.core.schema import (
    CorpusEntry,
    ExtractionStatus,
    FetchPolicy,
    Mutability,
    Snapshot,
    WarcRecordType,
)
from particles.core.status import Status
from particles.corpus.deposit import sha256
from particles.corpus.fetch import (
    RefetchUnsupported,
    _floor_seconds,
    _parse_last_modified,
    is_refresh_allowed,
    maybe_refetch,
    path_from_file_uri,
)
from particles.corpus.store import CorpusEntryRow, SnapshotRow, list_snapshots_for_entry

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _aiter_bytes(chunks: list[bytes]) -> Any:
    for chunk in chunks:
        yield chunk


def _make_resp(
    *,
    status_code: int = 200,
    content: bytes = b"<html>body</html>",
    etag: str | None = None,
    last_modified: str | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.headers = {}
    if etag:
        resp.headers["etag"] = etag
    if last_modified:
        resp.headers["last-modified"] = last_modified
    resp.raise_for_status = MagicMock()
    # fetch.py routes through particles.http.get_capped, which streams the
    # body via client.stream(...).aiter_bytes(); expose a one-chunk stream.
    resp.aiter_bytes = lambda: _aiter_bytes([content])
    return resp


def _patch_client(resp: MagicMock) -> Any:
    """Return a context manager patching particles_client to yield a client
    whose .get() returns the given response."""
    # fetch.py imports particles_client at module top, so patch its local
    # binding rather than the source in particles.http.
    mock_ctx = patch("particles.corpus.fetch.particles_client")
    return _ClientPatcher(mock_ctx, resp)


class _ClientPatcher:
    def __init__(self, ctx_patch: Any, resp: MagicMock) -> None:
        self._ctx_patch = ctx_patch
        self._resp = resp
        self.captured_headers: dict[str, str] | None = None

    def __enter__(self) -> _ClientPatcher:
        self._mock_ctx = self._ctx_patch.__enter__()
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(return_value=self._resp)

        # get_capped uses client.stream("GET", url) as an async CM yielding
        # the response. .stream is a sync call returning that CM.
        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(return_value=self._resp)
        stream_cm.__aexit__ = AsyncMock(return_value=False)
        client.stream = MagicMock(return_value=stream_cm)

        # Record the extra_headers passed to particles_client() — fetch.py
        # uses these to send If-None-Match / If-Modified-Since.
        def _capture(extra_headers: dict[str, str] | None = None) -> Any:
            self.captured_headers = extra_headers
            return client

        self._mock_ctx.side_effect = _capture
        return self

    def __exit__(self, *exc: Any) -> None:
        self._ctx_patch.__exit__(*exc)


async def _add_entry(
    session: Any,
    *,
    source_type: str = "WEB_PAGE",
    uri_r: str | None = "https://example.com/x",
    fetch_policy: FetchPolicy = FetchPolicy.LAZY,
    mutability: Mutability = Mutability.MUTABLE,
    deposited_by: str = "test",
) -> CorpusEntry:
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type=source_type,
        uri_r=uri_r,
        fetch_policy=fetch_policy,
        mutability=mutability,
        deposited_by=deposited_by,
    )
    session.add(CorpusEntryRow.from_model(entry))
    await session.flush()
    return entry


async def _add_snapshot(
    session: Any,
    entry: CorpusEntry,
    *,
    captured_at: datetime | None = None,
    content_hash: str = "deadbeef" * 8,
    etag: str | None = None,
    last_modified: datetime | None = None,
    extraction_status: ExtractionStatus = ExtractionStatus.COMPLETE,
) -> Snapshot:
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=captured_at or datetime.now(UTC),
        content_hash=content_hash,
        etag=etag,
        last_modified=last_modified,
        warc_record_type=WarcRecordType.RESPONSE,
        extraction_status=extraction_status,
    )
    session.add(SnapshotRow.from_model(snap, entry.entry_id))
    await session.flush()
    return snap


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


class TestFloorSeconds:
    def test_known_source_type(self) -> None:
        assert _floor_seconds("WEB_PAGE") == 3600
        assert _floor_seconds("PDF") == 7 * 86400
        assert _floor_seconds("CONVERSATION") == 0

    def test_unknown_source_type_defaults_to_one_hour(self) -> None:
        assert _floor_seconds("MY_NEW_TYPE") == 3600


class TestParseLastModified:
    def test_none_returns_none(self) -> None:
        assert _parse_last_modified(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_last_modified("") is None

    def test_valid_http_date_returns_datetime(self) -> None:
        result = _parse_last_modified("Wed, 21 Oct 2026 07:28:00 GMT")
        assert result is not None
        assert result.year == 2026
        assert result.month == 10
        assert result.day == 21

    def test_garbage_string_returns_none(self) -> None:
        assert _parse_last_modified("not a date") is None


# ---------------------------------------------------------------------------
# maybe_refetch — guards and short-circuits
# ---------------------------------------------------------------------------


class TestMaybeRefetchGuards:
    @pytest.mark.asyncio
    async def test_unknown_entry_raises(self, db_session: Any) -> None:
        with pytest.raises(ValueError, match="not found"):
            await maybe_refetch(db_session, "missing-entry-id")

    @pytest.mark.asyncio
    async def test_non_lazy_policy_returns_none_without_force(self, db_session: Any) -> None:
        entry = await _add_entry(db_session, fetch_policy=FetchPolicy.NEVER)
        await db_session.commit()
        assert await maybe_refetch(db_session, entry.entry_id) is None

    @pytest.mark.asyncio
    async def test_missing_uri_returns_none(self, db_session: Any) -> None:
        entry = await _add_entry(db_session, uri_r=None)
        await db_session.commit()
        assert await maybe_refetch(db_session, entry.entry_id) is None

    @pytest.mark.asyncio
    async def test_within_floor_returns_latest_without_fetch(self, db_session: Any) -> None:
        entry = await _add_entry(db_session, source_type="WEB_PAGE")
        # Snapshot taken just now — well within the 3600s floor for WEB_PAGE
        snap = await _add_snapshot(db_session, entry, captured_at=datetime.now(UTC))
        await db_session.commit()

        with _patch_client(_make_resp()) as p:
            result = await maybe_refetch(db_session, entry.entry_id)

        assert result is not None
        assert result.snapshot_id == snap.snapshot_id
        # The HTTP client should never have been touched.
        assert p.captured_headers is None


# ---------------------------------------------------------------------------
# Tier 1 — 304 Not Modified → REVISIT
# ---------------------------------------------------------------------------


class TestTier1NotModified:
    @pytest.mark.asyncio
    async def test_304_writes_revisit_snapshot(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        # Snapshot old enough to be eligible for re-fetch
        old_time = datetime.now(UTC) - timedelta(hours=2)
        prior = await _add_snapshot(
            db_session, entry, captured_at=old_time, etag='"abc123"', content_hash="original"
        )
        await db_session.commit()

        with _patch_client(_make_resp(status_code=304)) as p:
            new_snap = await maybe_refetch(db_session, entry.entry_id)

        assert new_snap is not None
        assert new_snap.warc_record_type == WarcRecordType.REVISIT
        assert new_snap.refers_to == prior.snapshot_id
        assert new_snap.content_hash == "original"  # inherited from prior
        # If-None-Match header was sent with the prior etag
        assert p.captured_headers == {"If-None-Match": '"abc123"'}


# ---------------------------------------------------------------------------
# Tier 2 — content hash compare
# ---------------------------------------------------------------------------


class TestTier2ContentHash:
    @pytest.mark.asyncio
    async def test_unchanged_content_writes_revisit(self, db_session: Any) -> None:
        from particles.corpus.deposit import sha256

        body = b"<html>same body</html>"
        body_hash = sha256(body)

        entry = await _add_entry(db_session)
        prior = await _add_snapshot(
            db_session,
            entry,
            captured_at=datetime.now(UTC) - timedelta(hours=2),
            content_hash=body_hash,
        )
        await db_session.commit()

        with _patch_client(_make_resp(content=body)):
            new_snap = await maybe_refetch(db_session, entry.entry_id)

        assert new_snap is not None
        assert new_snap.warc_record_type == WarcRecordType.REVISIT
        assert new_snap.refers_to == prior.snapshot_id

    @pytest.mark.asyncio
    async def test_changed_content_writes_response(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        await _add_snapshot(
            db_session,
            entry,
            captured_at=datetime.now(UTC) - timedelta(hours=2),
            content_hash="old-hash-different-from-new-body",
        )
        await db_session.commit()

        with _patch_client(_make_resp(content=b"completely new content", etag='"v2"')):
            new_snap = await maybe_refetch(db_session, entry.entry_id)

        assert new_snap is not None
        assert new_snap.warc_record_type == WarcRecordType.RESPONSE
        assert new_snap.etag == '"v2"'
        assert new_snap.extraction_status == ExtractionStatus.PENDING


# ---------------------------------------------------------------------------
# Importer-owned entries — the owning importer re-fetches
# ---------------------------------------------------------------------------

_THREAD_URI = "https://threads.example/item/7"


class _FakeImporter:
    """Accepts ``_THREAD_URI``'s host; ``refetch`` returns ``body`` and records calls."""

    def __init__(self, body: bytes = b'{"thread": 1}') -> None:
        self.body = body
        self.calls: list[str] = []

    def accepts_url(self, url: str) -> bool:
        return url.startswith("https://threads.example/")

    async def refetch(self, uri_r: str) -> bytes:
        self.calls.append(uri_r)
        return self.body


class _NoRefetchImporter:
    """An importer that owns the URL but predates ``refetch``."""

    def accepts_url(self, url: str) -> bool:
        return url.startswith("https://threads.example/")


class TestImporterOwnedEntries:
    async def _stale_entry(self, session: Any, content_hash: str) -> tuple[CorpusEntry, Snapshot]:
        entry = await _add_entry(session, source_type="THREAD", uri_r=_THREAD_URI)
        prior = await _add_snapshot(
            session,
            entry,
            captured_at=datetime.now(UTC) - timedelta(hours=2),
            content_hash=content_hash,
        )
        await session.commit()
        return entry, prior

    @pytest.mark.asyncio
    async def test_unchanged_importer_bytes_write_revisit_without_a_get(
        self, db_session: Any
    ) -> None:
        importer = _FakeImporter()
        entry, prior = await self._stale_entry(db_session, sha256(importer.body))

        with _patch_client(_make_resp(content=b"<html>the permalink page</html>")) as p:
            snap = await maybe_refetch(db_session, entry.entry_id, importers=[importer])

        assert importer.calls == [_THREAD_URI]
        assert p.captured_headers is None  # the URI-R was never fetched directly
        assert snap is not None
        assert snap.warc_record_type == WarcRecordType.REVISIT
        assert snap.refers_to == prior.snapshot_id

    @pytest.mark.asyncio
    async def test_changed_importer_bytes_write_a_pending_response(self, db_session: Any) -> None:
        importer = _FakeImporter(body=b'{"thread": 2}')
        entry, _prior = await self._stale_entry(db_session, sha256(b'{"thread": 1}'))

        snap = await maybe_refetch(db_session, entry.entry_id, importers=[importer])

        assert snap is not None
        assert snap.warc_record_type == WarcRecordType.RESPONSE
        assert snap.content_hash == sha256(importer.body)
        assert snap.extraction_status == ExtractionStatus.PENDING
        assert snap.etag is None  # an importer body carries no HTTP validators

    @pytest.mark.asyncio
    async def test_first_capture_goes_through_the_importer(self, db_session: Any) -> None:
        importer = _FakeImporter()
        entry = await _add_entry(db_session, source_type="THREAD", uri_r=_THREAD_URI)
        await db_session.commit()

        snap = await maybe_refetch(db_session, entry.entry_id, importers=[importer])

        assert importer.calls == [_THREAD_URI]
        assert snap is not None and snap.warc_record_type == WarcRecordType.RESPONSE

    @pytest.mark.asyncio
    async def test_owner_without_refetch_refuses_rather_than_get(self, db_session: Any) -> None:
        entry, _prior = await self._stale_entry(db_session, "old")

        with (
            _patch_client(_make_resp()) as p,
            pytest.raises(RefetchUnsupported, match="_NoRefetchImporter.*Deposit the URL again"),
        ):
            await maybe_refetch(db_session, entry.entry_id, importers=[_NoRefetchImporter()])

        assert p.captured_headers is None
        assert len(await list_snapshots_for_entry(db_session, entry.entry_id)) == 1

    @pytest.mark.asyncio
    async def test_owner_without_refetch_inside_the_floor_is_not_an_error(
        self, db_session: Any
    ) -> None:
        entry = await _add_entry(db_session, source_type="THREAD", uri_r=_THREAD_URI)
        snap = await _add_snapshot(db_session, entry, captured_at=datetime.now(UTC))
        await db_session.commit()

        result = await maybe_refetch(db_session, entry.entry_id, importers=[_NoRefetchImporter()])

        assert result is not None and result.snapshot_id == snap.snapshot_id

    @pytest.mark.asyncio
    async def test_url_no_importer_accepts_still_takes_the_get(self, db_session: Any) -> None:
        importer = _FakeImporter()
        entry = await _add_entry(db_session)  # https://example.com/x
        await _add_snapshot(db_session, entry, captured_at=datetime.now(UTC) - timedelta(hours=2))
        await db_session.commit()

        with _patch_client(_make_resp(content=b"<html>new</html>")) as p:
            snap = await maybe_refetch(db_session, entry.entry_id, importers=[importer])

        assert importer.calls == []
        assert p.captured_headers is not None
        assert snap is not None and snap.warc_record_type == WarcRecordType.RESPONSE


# ---------------------------------------------------------------------------
# MUTABLE re-fetch side effect — the ladder does NOT demote
# ---------------------------------------------------------------------------


class TestMutableSideEffect:
    @pytest.mark.asyncio
    async def test_changed_mutable_leaves_prior_particles_active(self, db_session: Any) -> None:
        """The fetch ladder writes the snapshot; it does not retire the generation.

        The MUTABLE staleness cascade was moved out of this module into
        ``particles.ingest.generation``, where it runs *after* the new snapshot
        has been extracted. Demoting here — as this code used to — blinds
        carry-forward, which looks up ACTIVE particles: the
        re-extraction would then re-pay the LLM for every unchanged paragraph
        and mint duplicates of claims that never changed. So the prior
        generation must still be ACTIVE when the ladder returns.
        """
        from particles.core.schema import (
            Confidence,
            Particle,
            ProvenanceRef,
            ProvenanceRefType,
            UncertaintyNature,
        )
        from particles.core.scoring.confidence import CalibrationSource
        from particles.store.particle_store import get_particle, insert_particle

        entry = await _add_entry(db_session, mutability=Mutability.MUTABLE)
        prior_snap = await _add_snapshot(
            db_session,
            entry,
            captured_at=datetime.now(UTC) - timedelta(hours=2),
            content_hash="old",
        )
        # Attach an ACTIVE particle to the prior snapshot
        p = Particle(
            id=str(uuid.uuid4()),
            content="A claim about a mutable page.",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            asserted_at=datetime.now(UTC),
            status=Status.ACTIVE,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=entry.entry_id,
                    snapshot_id=prior_snap.snapshot_id,
                )
            ],
        )
        await insert_particle(db_session, p)
        await db_session.commit()

        with _patch_client(_make_resp(content=b"NEW content not matching old hash")):
            new_snap = await maybe_refetch(db_session, entry.entry_id)

        assert new_snap is not None
        assert new_snap.warc_record_type == WarcRecordType.RESPONSE

        # The prior generation is untouched — retiring it is the post-extraction
        # cascade's job, so that carry-forward still sees it.
        after = await get_particle(db_session, p.id)
        assert after is not None
        assert after.status == Status.ACTIVE


# ---------------------------------------------------------------------------
# No prior snapshot → fresh fetch
# ---------------------------------------------------------------------------


class TestNoPriorSnapshot:
    @pytest.mark.asyncio
    async def test_first_fetch_writes_response_without_conditional_headers(
        self, db_session: Any
    ) -> None:
        entry = await _add_entry(db_session)
        await db_session.commit()

        with _patch_client(_make_resp(content=b"first fetch")) as p:
            snap = await maybe_refetch(db_session, entry.entry_id)

        assert snap is not None
        assert snap.warc_record_type == WarcRecordType.RESPONSE
        # No prior → no If-None-Match / If-Modified-Since
        assert p.captured_headers == {}


# ---------------------------------------------------------------------------
# The local tier — stat mtime, then content hash. No network.
# ---------------------------------------------------------------------------


def _local_entry_kwargs(path: Path) -> dict[str, Any]:
    return {
        "source_type": "LOCAL_MARKDOWN",
        "uri_r": path.resolve().as_uri(),
        "fetch_policy": FetchPolicy.LAZY,
        "mutability": Mutability.MUTABLE,
    }


class TestPathFromFileUri:
    def test_round_trips_a_deposited_path(self, tmp_path: Path) -> None:
        f = tmp_path / "AGENTS.md"
        f.write_text("x")
        assert path_from_file_uri(f.resolve().as_uri()) == f.resolve()

    def test_decodes_percent_escapes(self, tmp_path: Path) -> None:
        f = tmp_path / "my rules.md"
        f.write_text("x")
        assert path_from_file_uri(f.resolve().as_uri()) == f.resolve()

    def test_rejects_a_remote_authority(self) -> None:
        # file://evil.example/etc/passwd must not silently become /etc/passwd.
        with pytest.raises(ValueError, match="non-local file URI"):
            path_from_file_uri("file://evil.example/etc/passwd")

    def test_allows_the_localhost_authority(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text("x")
        assert path_from_file_uri(f"file://localhost{f.resolve()}") == f.resolve()


@pytest.fixture
def allow_tmp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Admit ``tmp_path`` to the local refresh roots (security review D2)."""
    monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(tmp_path)])
    return tmp_path


@pytest.mark.usefixtures("allow_tmp_root")
class TestLocalTier:
    @pytest.mark.asyncio
    async def test_unchanged_mtime_short_circuits_without_reading(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        """Tier 1: an equal mtime returns the prior snapshot and touches no bytes."""
        f = tmp_path / "AGENTS.md"
        f.write_text("rule one")
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)

        entry = await _add_entry(db_session, **_local_entry_kwargs(f))
        prior = await _add_snapshot(
            db_session,
            entry,
            captured_at=datetime.now(UTC) - timedelta(days=1),
            content_hash="whatever-the-hash-was",
            last_modified=mtime,
        )
        await db_session.commit()

        # read_bytes must not be called — if tier 1 fell through, the stub raises.
        with patch.object(Path, "read_bytes", side_effect=AssertionError("read the file")):
            snap = await maybe_refetch(db_session, entry.entry_id)

        assert snap is not None
        assert snap.snapshot_id == prior.snapshot_id
        assert len(await list_snapshots_for_entry(db_session, entry.entry_id)) == 1

    @pytest.mark.asyncio
    async def test_changed_mtime_same_bytes_writes_revisit_and_restamps(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        """Tier 2 match → REVISIT, stamped with the new mtime so tier 1 catches next time."""
        f = tmp_path / "AGENTS.md"
        f.write_text("rule one")
        content_hash = sha256(f.read_bytes())

        entry = await _add_entry(db_session, **_local_entry_kwargs(f))
        await _add_snapshot(
            db_session,
            entry,
            captured_at=datetime.now(UTC) - timedelta(days=1),
            content_hash=content_hash,
            last_modified=datetime.now(UTC) - timedelta(days=2),  # a stale mtime stamp
        )
        await db_session.commit()

        snap = await maybe_refetch(db_session, entry.entry_id)

        assert snap is not None
        assert snap.warc_record_type == WarcRecordType.REVISIT
        assert snap.content_hash == content_hash
        assert snap.extraction_status == ExtractionStatus.COMPLETE
        # Restamped: the next run's tier 1 short-circuits instead of re-reading.
        assert snap.last_modified is not None
        assert snap.last_modified.replace(tzinfo=UTC) == datetime.fromtimestamp(
            f.stat().st_mtime, tz=UTC
        )

    @pytest.mark.asyncio
    async def test_changed_content_writes_pending_response(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        """The loop's first link: an edited rule file becomes a PENDING snapshot."""
        f = tmp_path / "AGENTS.md"
        f.write_text("old rule")
        entry = await _add_entry(db_session, **_local_entry_kwargs(f))
        await _add_snapshot(
            db_session,
            entry,
            captured_at=datetime.now(UTC) - timedelta(days=1),
            content_hash=sha256(f.read_bytes()),
            last_modified=datetime.fromtimestamp(f.stat().st_mtime, tz=UTC),
        )
        await db_session.commit()

        f.write_text("NEW rule — the old one is forbidden now")
        snap = await maybe_refetch(db_session, entry.entry_id)

        assert snap is not None
        assert snap.warc_record_type == WarcRecordType.RESPONSE
        assert snap.content_hash == sha256(f.read_bytes())
        assert snap.extraction_status == ExtractionStatus.PENDING
        assert snap.archive_path is not None

    @pytest.mark.asyncio
    async def test_mtime_moving_backwards_still_triggers_a_read(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        """Inequality, not recency: a restore that rewinds mtime must not be missed."""
        f = tmp_path / "AGENTS.md"
        f.write_text("restored older content")
        entry = await _add_entry(db_session, **_local_entry_kwargs(f))
        await _add_snapshot(
            db_session,
            entry,
            captured_at=datetime.now(UTC) - timedelta(days=1),
            content_hash="a-different-hash",
            # Recorded mtime is in the FUTURE relative to the file on disk.
            last_modified=datetime.now(UTC) + timedelta(days=7),
        )
        await db_session.commit()

        snap = await maybe_refetch(db_session, entry.entry_id)

        assert snap is not None
        assert snap.warc_record_type == WarcRecordType.RESPONSE

    @pytest.mark.asyncio
    async def test_null_recorded_mtime_bootstraps_via_tier_2(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        """Every pre-ADR-0206 snapshot has last_modified NULL; it must stamp itself."""
        f = tmp_path / "AGENTS.md"
        f.write_text("same bytes")
        entry = await _add_entry(db_session, **_local_entry_kwargs(f))
        await _add_snapshot(
            db_session,
            entry,
            captured_at=datetime.now(UTC) - timedelta(days=1),
            content_hash=sha256(f.read_bytes()),
            last_modified=None,
        )
        await db_session.commit()

        snap = await maybe_refetch(db_session, entry.entry_id)

        assert snap is not None
        assert snap.warc_record_type == WarcRecordType.REVISIT
        assert snap.last_modified is not None  # bootstrapped

    @pytest.mark.asyncio
    async def test_missing_file_returns_none_without_raising(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        """A deleted rule file is an outcome, not a crash — and not a retraction."""
        f = tmp_path / "gone.md"
        f.write_text("here for now")
        entry = await _add_entry(db_session, **_local_entry_kwargs(f))
        await _add_snapshot(db_session, entry, content_hash=sha256(f.read_bytes()))
        await db_session.commit()
        f.unlink()

        assert await maybe_refetch(db_session, entry.entry_id) is None

    @pytest.mark.asyncio
    async def test_never_policy_is_not_refreshed(self, db_session: Any, tmp_path: Path) -> None:
        """The opt-in gate: a default local deposit (NEVER) stays frozen."""
        f = tmp_path / "AGENTS.md"
        f.write_text("x")
        kwargs = _local_entry_kwargs(f) | {"fetch_policy": FetchPolicy.NEVER}
        entry = await _add_entry(db_session, **kwargs)
        await db_session.commit()

        assert await maybe_refetch(db_session, entry.entry_id) is None

    @pytest.mark.asyncio
    async def test_symlink_refused_unless_configured(self, db_session: Any, tmp_path: Path) -> None:
        """A path that became a symlink *after* deposit is refused, not followed.

        ``deposit_file`` records ``path.resolve().as_uri()``, so a path that was
        already a symlink at deposit time is stored as its target and never
        reaches this guard. What the guard covers is the path swapped for a
        symlink afterwards: that is a change of *identity*, not of content, and
        silently ingesting the new target is not a call this pass should make.
        """
        target = tmp_path / "real.md"
        target.write_text("content")
        link = tmp_path / "link.md"
        link.symlink_to(target)

        kwargs = _local_entry_kwargs(target) | {"uri_r": link.absolute().as_uri()}
        entry = await _add_entry(db_session, **kwargs)
        await _add_snapshot(db_session, entry, content_hash="old")
        await db_session.commit()

        assert await maybe_refetch(db_session, entry.entry_id) is None

    @pytest.mark.asyncio
    async def test_local_tier_makes_no_http_call(self, db_session: Any, tmp_path: Path) -> None:
        """No network, no SSRF surface: the HTTP client is never constructed."""
        f = tmp_path / "AGENTS.md"
        f.write_text("v1")
        entry = await _add_entry(db_session, **_local_entry_kwargs(f))
        await _add_snapshot(db_session, entry, content_hash="old-hash")
        await db_session.commit()

        with patch("particles.corpus.fetch.particles_client") as client:
            snap = await maybe_refetch(db_session, entry.entry_id)

        assert snap is not None
        client.assert_not_called()


# ---------------------------------------------------------------------------
# The local tier's root allowlist (security review D2)
# ---------------------------------------------------------------------------


class TestRefreshAllowlist:
    """The local tier reads only under the roots its legitimate depositors use.

    An entry's ``file://`` URI-R is not proof the operator chose that file: an
    HTTP caller could once enrol any engine-host path as ``LAZY`` and read it
    back from ``/corpus/blob``.
    """

    @pytest.fixture(autouse=True)
    def _isolated_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        # Discovery adds ~/.claude; keep the real one out of every decision.
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        return home

    @pytest.mark.asyncio
    async def test_outside_root_is_refused_without_touching_the_file(
        self, db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        secret = tmp_path / "elsewhere" / "id_ed25519"
        secret.parent.mkdir()
        secret.write_text("PRIVATE KEY")
        monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(allowed)])

        entry = await _add_entry(db_session, **_local_entry_kwargs(secret))
        await db_session.commit()

        # Only the refused file is off limits: resolving the roots may stat
        # them (``Path.resolve()`` does on Python 3.11).
        real_stat = Path.stat

        def _stat(self: Path, *args: Any, **kwargs: Any) -> Any:
            if os.path.realpath(self) == os.path.realpath(secret):
                raise AssertionError("stat the file")
            return real_stat(self, *args, **kwargs)

        with (
            patch.object(Path, "read_bytes", side_effect=AssertionError("read the file")),
            patch.object(Path, "stat", _stat),
        ):
            assert await maybe_refetch(db_session, entry.entry_id, force=True) is None
        assert await list_snapshots_for_entry(db_session, entry.entry_id) == []

    @pytest.mark.asyncio
    async def test_inside_root_is_refreshed(
        self, db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notes = tmp_path / "notes"
        notes.mkdir()
        f = notes / "plan.md"
        f.write_text("v2")
        monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(notes)])

        entry = await _add_entry(db_session, **_local_entry_kwargs(f))
        await _add_snapshot(db_session, entry, content_hash="old-hash")
        await db_session.commit()

        snap = await maybe_refetch(db_session, entry.entry_id)
        assert snap is not None
        assert snap.content_hash == sha256(b"v2")
        assert snap.warc_record_type is WarcRecordType.RESPONSE

    def test_dotdot_cannot_step_out_of_a_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(tmp_path / "a")])
        assert not is_refresh_allowed(tmp_path / "a" / ".." / "b" / "x.md")

    def test_rule_source_root_admits_rule_filenames_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        (repo / "docs").mkdir(parents=True)
        monkeypatch.setattr(get_config().rule_sources, "paths", [str(repo)])

        assert is_refresh_allowed(repo / "docs" / "AGENTS.md")
        assert not is_refresh_allowed(repo / ".env")

    def test_registered_rule_file_is_admitted_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rules = tmp_path / "house-rules.txt"
        rules.write_text("x")
        monkeypatch.setattr(get_config().rule_sources, "paths", [str(rules)])
        assert is_refresh_allowed(rules)
        assert not is_refresh_allowed(tmp_path / "other.txt")

    def test_recorded_root_and_user_claude(self, tmp_path: Path, _isolated_home: Path) -> None:
        repo = (tmp_path / "repo").resolve()
        (repo / ".git").mkdir(parents=True)
        (repo / "sub").mkdir()
        (_isolated_home / ".claude").mkdir()

        assert is_refresh_allowed(repo / "sub" / "CLAUDE.md", recorded_roots=[repo])
        assert is_refresh_allowed(_isolated_home / ".claude" / "CLAUDE.md")
        assert not is_refresh_allowed(_isolated_home / ".claude" / ".credentials.json")
        # A root nobody recorded is not derived, whatever the working directory.
        assert not is_refresh_allowed(repo / "AGENTS.md")

    def test_refresh_allowlist_does_not_depend_on_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Security review N38: the same entry, the same answer, anywhere."""
        project = (tmp_path / "project").resolve()
        (project / ".git").mkdir(parents=True)
        clone = (tmp_path / "clone").resolve()
        (clone / ".git").mkdir(parents=True)
        neutral = tmp_path / "neutral"
        neutral.mkdir()

        for cwd in (project, clone, neutral):
            monkeypatch.chdir(cwd)
            assert is_refresh_allowed(project / "AGENTS.md", recorded_roots=[project])
            assert not is_refresh_allowed(clone / "AGENTS.md", recorded_roots=[project])

    def test_a_recorded_worktree_root_also_admits_the_main_checkout(self, tmp_path: Path) -> None:
        main = (tmp_path / "main").resolve()
        worktree = main / ".claude" / "worktrees" / "wt"
        (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
        worktree.mkdir(parents=True)
        (worktree / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")

        assert is_refresh_allowed(main / "AGENTS.md", recorded_roots=[worktree])
        assert is_refresh_allowed(worktree / "AGENTS.md", recorded_roots=[worktree])

    def test_claude_code_memory_files_are_admitted(
        self, tmp_path: Path, _isolated_home: Path
    ) -> None:
        memory = _isolated_home / ".claude" / "projects" / "-repo" / "memory"
        memory.mkdir(parents=True)
        assert is_refresh_allowed(memory / "feedback.md")
        assert not is_refresh_allowed(memory.parent / "session.jsonl")


class TestLocalRefreshProvenanceAndSwap:
    """The refresh reads only what a local depositor enrolled, through one checked descriptor.

    Security review 2026-10-24: N21 (a LAZY ``file://`` entry the HTTP route
    enrolled before 1.210.4 still refreshed inside the derived roots) and N22
    (the root check resolved the path, then the read re-walked the name, so a
    parent swapped in between was followed).
    """

    @pytest.fixture(autouse=True)
    def _isolated_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        return home

    def _memory_file(self, home: Path) -> Path:
        memory = home / ".claude" / "projects" / "-other-project" / "memory"
        memory.mkdir(parents=True)
        note = memory / "note.md"
        note.write_text("another project's memory")
        return note

    @pytest.mark.asyncio
    async def test_an_http_enrolled_entry_is_not_refreshed(
        self, db_session: Any, _isolated_home: Path
    ) -> None:
        note = self._memory_file(_isolated_home)
        entry = await _add_entry(db_session, **_local_entry_kwargs(note), deposited_by="api-user")
        await db_session.commit()

        assert await maybe_refetch(db_session, entry.entry_id, force=True) is None
        assert await list_snapshots_for_entry(db_session, entry.entry_id) == []

    @pytest.mark.asyncio
    async def test_the_harvest_hooks_entry_still_refreshes(
        self, db_session: Any, _isolated_home: Path
    ) -> None:
        note = self._memory_file(_isolated_home)
        entry = await _add_entry(
            db_session, **_local_entry_kwargs(note), deposited_by="claude-code-hook"
        )
        await db_session.commit()

        snap = await maybe_refetch(db_session, entry.entry_id, force=True)
        assert snap is not None
        assert snap.content_hash == sha256(b"another project's memory")

    @pytest.mark.asyncio
    async def test_an_allowed_root_admits_any_depositor(
        self, db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notes = tmp_path / "notes"
        notes.mkdir()
        f = notes / "plan.md"
        f.write_text("mine")
        monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(notes)])
        entry = await _add_entry(db_session, **_local_entry_kwargs(f), deposited_by="api-user")
        await db_session.commit()

        assert await maybe_refetch(db_session, entry.entry_id, force=True) is not None

    def test_every_local_lazy_depositor_is_admitted(self) -> None:
        """The ids the local enrolment paths record are the ones the derived roots admit."""
        import inspect

        from particles.corpus.fetch import LOCAL_LAZY_DEPOSITORS
        from particles.corpus.rule_sources import sync_rule_sources

        default = inspect.signature(sync_rule_sources).parameters["deposited_by"].default
        assert {default, "init", "claude-code-hook", "operator"} <= LOCAL_LAZY_DEPOSITORS
        hook = Path(inspect.getfile(sync_rule_sources)).parents[1] / "api" / "cli" / "hook.py"
        assert 'deposited_by="claude-code-hook"' in hook.read_text(encoding="utf-8")
        init = hook.with_name("init.py").read_text(encoding="utf-8")
        assert 'deposited_by="init"' in init

    @pytest.mark.asyncio
    async def test_a_parent_swapped_after_the_root_check_is_not_read(
        self, db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil

        from particles.corpus import fetch as fetch_mod

        notes = tmp_path / "notes"
        (notes / "sub").mkdir(parents=True)
        f = notes / "sub" / "plan.md"
        f.write_text("inside")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "plan.md").write_text("OUTSIDE THE ROOT")
        monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(notes)])

        entry = await _add_entry(db_session, **_local_entry_kwargs(f), deposited_by="operator")
        await db_session.commit()

        real_check = fetch_mod.is_refresh_allowed
        calls: list[Path] = []

        def _check_then_swap(path: Path, **kwargs: Any) -> bool:
            verdict = real_check(path, **kwargs)
            calls.append(path)
            if len(calls) == 1:
                shutil.rmtree(notes / "sub")
                (notes / "sub").symlink_to(outside, target_is_directory=True)
            return verdict

        monkeypatch.setattr(fetch_mod, "is_refresh_allowed", _check_then_swap)
        assert await maybe_refetch(db_session, entry.entry_id, force=True) is None
        assert await list_snapshots_for_entry(db_session, entry.entry_id) == []
