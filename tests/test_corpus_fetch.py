"""Tests for §7.5 lazy re-fetch (particles/corpus/fetch.py).

The three-tier change-detection protocol:
  Tier 1: ETag / If-Modified-Since → 304 → REVISIT
  Tier 2: SHA-256 of fetched body → match → REVISIT, mismatch → RESPONSE
  Tier 3: force=True bypasses everything

The local tier (``file://`` URI-R) is covered by TestLocalTier at the
bottom: stat mtime as tier 1, SHA-256 as tier 2, and no network at all.

httpx is mocked via the documented ``patch("particles.http.particles_client")``
recipe from tests/AGENTS.md. No network IO; no real blob writes are needed
beyond what save_blob does into PARTICLES_BLOB_DIR (conftest points it at
/tmp/particles_test_blobs).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    _floor_seconds,
    _parse_last_modified,
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
) -> CorpusEntry:
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type=source_type,
        uri_r=uri_r,
        fetch_policy=fetch_policy,
        mutability=mutability,
        deposited_by="test",
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
