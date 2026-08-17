"""Tests for the FastAPI app (particles/api/app.py).

The HTTP layer was 0% covered in the architecture-review baseline. These
tests exercise each endpoint via FastAPI's TestClient against a file-based
SQLite (the lazy engine in particles.db is shared across requests within
a process, so this works as a stand-in for the real server's lifecycle).

PARTICLES_API_KEY is left at its default ("dev-key"), which disables the
bearer-token check — see particles/api/auth.py. The auth seam itself is
already covered by tests/test_auth.py; here we exercise endpoint behaviour.
The TestClient presents a loopback peer (127.0.0.1) so the dev-key
non-loopback gate does not refuse the authenticated write
endpoints — an in-process test IS a local caller.

LLM-driven endpoints (POST /extract, POST /query) and external-HTTP
endpoints (POST /corpus/deposit/url) are out of scope — covered elsewhere
or would require non-trivial mocking. We test their bad-input error paths.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from particles.api.app import app
from particles.core.schema import (
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    Subject,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def api_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """File-based SQLite for FastAPI tests; tables pre-created via Base.metadata."""
    db_path = tmp_path / "api.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))

    from particles.config import reset_config

    reset_config()

    async def _create_tables() -> None:
        import particles.corpus.store  # noqa: F401
        import particles.store.extractor_store  # noqa: F401
        import particles.store.particle_store  # noqa: F401
        import particles.store.relation_store  # noqa: F401
        import particles.store.subject_store  # noqa: F401
        import particles.store.trust_store  # noqa: F401
        import particles.store.wikidata_cache  # noqa: F401
        from particles.db import Base, get_engine

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create_tables())
    yield db_path

    # Dispose the engine before reset_config() drops the cached pointer —
    # otherwise aiosqlite's __del__ raises RuntimeError on a dead loop.
    # See tests/conftest.py::db_session for the full rationale.
    async def _dispose() -> None:
        from particles.db import get_engine

        await get_engine().dispose()

    asyncio.run(_dispose())
    reset_config()


@pytest.fixture
def client(api_db: Path) -> TestClient:
    # Constructing without the context-manager form skips FastAPI startup
    # events; we don't want create_tables() (alembic) to run.
    # client=("127.0.0.1", …) presents a loopback peer so the dev-key
    # non-loopback gate treats these in-process calls as local.
    return TestClient(app, client=("127.0.0.1", 50000))


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


async def _add_corpus_entry(
    *, source_type: str = "WEB_PAGE", uri_r: str = "https://example.com/x"
) -> tuple[str, str]:
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.db import session_scope

    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()), source_type=source_type, uri_r=uri_r, deposited_by="test"
    )
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash="b" * 64,
        extraction_status=ExtractionStatus.COMPLETE,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    async with session_scope() as session:
        session.add(CorpusEntryRow.from_model(entry))
        session.add(SnapshotRow.from_model(snap, entry.entry_id))
        await session.commit()
    return entry.entry_id, snap.snapshot_id


async def _add_corpus_entry_with_blob(content: bytes) -> tuple[str, str]:
    """Insert an entry + snapshot whose content_hash blob is written to disk."""
    from particles.corpus.deposit import save_blob, sha256
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.db import session_scope

    content_hash = sha256(content)
    save_blob(content, content_hash)
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="WEB_PAGE",
        uri_r=f"https://example.com/{content_hash[:8]}",
        deposited_by="test",
    )
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash=content_hash,
        extraction_status=ExtractionStatus.COMPLETE,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    async with session_scope() as session:
        session.add(CorpusEntryRow.from_model(entry))
        session.add(SnapshotRow.from_model(snap, entry.entry_id))
        await session.commit()
    return entry.entry_id, snap.snapshot_id


async def _add_subject(name: str = "Test") -> str:
    from particles.db import session_scope
    from particles.store.subject_store import insert_subject

    subj = Subject(id=str(uuid.uuid4()), canonical_name=name, asserted_by="test")
    async with session_scope() as session:
        await insert_subject(session, subj)
        await session.commit()
    return subj.id


async def _add_active_particle(*, content: str = "A claim.") -> str:
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
    )
    async with session_scope() as session:
        await insert_particle(session, p)
        await session.commit()
    return p.id


async def _add_conflict_trio() -> tuple[str, str, str]:
    """One §6.6 conflict wired for the graph evidence scope: returns
    (inconsistency_id, winner_id, loser_id)."""
    from particles.core.status import StatusReason
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject

    subj = Subject(id=str(uuid.uuid4()), canonical_name="Conflict Subject", asserted_by="test")

    def _p(content: str, status: Status, reason: StatusReason | None = None) -> Particle:
        return Particle(
            id=str(uuid.uuid4()),
            content=content,
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            asserted_at=datetime.now(UTC),
            status=status,
            status_reason=reason,
            subject_ids=[subj.id],
        )

    winner = _p("The claim that won.", Status.ACTIVE)
    loser = _p("The claim that lost.", Status.PROVENANCE_STALE, StatusReason.CONFLICT_PENDING)
    inc = _p("conflict record", Status.INCONSISTENCY)
    inc.provenance = [
        ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=winner.id, snapshot_id="-"),
        ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=loser.id, snapshot_id="-"),
    ]
    async with session_scope() as session:
        await insert_subject(session, subj)
        for p in (winner, loser, inc):
            await insert_particle(session, p)
        await session.commit()
    return inc.id, winner.id, loser.id


async def _add_particle_bound_to(content: str, subject_id: str) -> str:
    """Insert an ACTIVE CLAIM particle bound to a subject; return its id.

    ``insert_particle`` writes the ``particle_subjects`` join rows from
    ``subject_ids``, so the split endpoint sees the binding.
    """
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        subject_ids=[subject_id],
    )
    async with session_scope() as session:
        await insert_particle(session, p)
        await session.commit()
    return p.id


async def _add_particle_from_entry(
    entry_id: str, snapshot_id: str, content: str = "A sourced claim."
) -> str:
    """Insert an ACTIVE particle whose provenance traces to a corpus entry.

    The SOURCE provenance ref writes the edge ``corpus retract`` partitions on.
    """
    from particles.core.schema import ProvenanceRef, ProvenanceRefType
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_id,
                snapshot_id=snapshot_id,
            )
        ],
    )
    async with session_scope() as session:
        await insert_particle(session, p)
        await session.commit()
    return p.id


async def _particle_status(particle_id: str) -> Status:
    from particles.db import session_scope
    from particles.store.particle_store import get_particle

    async with session_scope() as session:
        p = await get_particle(session, particle_id)
        assert p is not None
        return p.status


async def _add_inconsistency() -> str:
    """Seed two ACTIVE particles + an INCONSISTENCY wrapper; return the wrapper id."""
    from particles.core.schema import ProvenanceRef, ProvenanceRefType
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    a = await _add_active_particle(content="claim A")
    b = await _add_active_particle(content="claim B")
    inc = Particle(
        id=str(uuid.uuid4()),
        content="INCONSISTENCY between A and B",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.INCONSISTENCY,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=a),
            ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=b),
        ],
    )
    async with session_scope() as session:
        await insert_particle(session, inc)
        await session.commit()
    return inc.id


# ---------------------------------------------------------------------------
# Health & quality
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_built_at_is_null_from_a_source_checkout(self, client: TestClient) -> None:
        """Nothing stamped it, so /health says so rather than guessing."""
        assert client.get("/health").json()["built_at"] is None

    def test_built_at_discloses_the_image_build_date(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The container stamp reaches /health, which is how the UI dates the engine."""
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_BUILD_DATE", "2026-08-08T18:38:08Z")
        reset_config()
        assert client.get("/health").json()["built_at"] == "2026-08-08T18:38:08Z"

    def test_an_unstamped_image_exports_empty_and_reads_as_absent(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`docker build` with no --build-arg exports "" — an empty stamp is no stamp."""
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_BUILD_DATE", "")
        reset_config()
        assert client.get("/health").json()["built_at"] is None


class TestQualityDashboard:
    def test_empty_db(self, client: TestClient) -> None:
        resp = client.get("/quality")
        assert resp.status_code == 200
        body = resp.json()
        assert body["active_particles"] == 0
        assert body["total_entries"] == 0

    def test_with_data(self, client: TestClient) -> None:
        _run_async(_add_corpus_entry())
        _run_async(_add_active_particle())
        resp = client.get("/quality")
        assert resp.status_code == 200
        body = resp.json()
        assert body["active_particles"] == 1
        assert body["total_entries"] == 1


# ---------------------------------------------------------------------------
# Curation — GET /curation, the bus-stop-editing queue over HTTP
# ---------------------------------------------------------------------------


class TestCuration:
    def test_empty_db_returns_empty_queue(self, client: TestClient) -> None:
        # semantic=False keeps the LLM-assisted finders out of the request.
        resp = client.get("/curation", params={"semantic": False})
        assert resp.status_code == 200
        body = resp.json()
        assert body["cards"] == []
        assert body["count"] == 0

    def test_cold_store_is_served_live_and_says_so(self, client: TestClient) -> None:
        # No collection has been built, so the read collects live rather than
        # writing a cache from a GET. Slow but correct, and honestly labelled.
        body = client.get("/curation", params={"semantic": False}).json()
        assert body["source"] == "live"
        assert body["snapshot_id"] is None
        assert body["built_at"] is None

    def test_response_carries_the_adr_0238_staleness_stamp(self, client: TestClient) -> None:
        # Once a collection exists, the response must say how old the detection
        # is rather than implying it is live.
        client.post("/curation/rebuild", params={"semantic": False})
        body = client.get("/curation", params={"semantic": False}).json()
        assert body["source"] == "snapshot"
        assert body["snapshot_id"]
        assert body["built_at"]
        assert body["stale"] is False
        assert body["scope"] == "store"
        # Every card kind is disclosed as store-wide on an explicit rebuild.
        assert set(body["per_kind_scope"].values()) == {"store"}

    def test_no_snapshot_param_bypasses_the_cache(self, client: TestClient) -> None:
        resp = client.get("/curation", params={"semantic": False, "no_snapshot": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "live"
        assert body["snapshot_id"] is None

    def test_rebuild_writes_a_new_collection(self, client: TestClient) -> None:
        first = client.post("/curation/rebuild", params={"semantic": False}).json()
        assert first["snapshot_id"]
        second = client.post("/curation/rebuild", params={"semantic": False}).json()
        assert second["snapshot_id"] != first["snapshot_id"]
        assert second["scope"] == "store"

    def test_contested_belief_surfaces_a_card(self, client: TestClient) -> None:
        # An open INCONSISTENCY referencing two ACTIVE beliefs yields contested
        # cards via get_inconsistency_backrefs (no LLM, no fix mutation).
        _run_async(_add_inconsistency())
        resp = client.get("/curation", params={"semantic": False})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == len(body["cards"])
        assert body["count"] >= 1
        kinds = {c["kind"] for c in body["cards"]}
        assert "contested" in kinds
        # The card shape carries the fields the client renders.
        contested = next(c for c in body["cards"] if c["kind"] == "contested")
        assert contested["particle_ids"]
        assert contested["suggested_gestures"]
        assert "leverage" in contested

    def test_kind_filter_restricts_card_kinds(self, client: TestClient) -> None:
        _run_async(_add_inconsistency())
        resp = client.get("/curation", params={"semantic": False, "kind": "contested"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1
        assert all(c["kind"] == "contested" for c in body["cards"])

    def test_limit_caps_the_queue(self, client: TestClient) -> None:
        # _add_inconsistency seeds two contested beliefs ⇒ two cards; limit=1 caps.
        _run_async(_add_inconsistency())
        resp = client.get("/curation", params={"semantic": False, "limit": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert len(body["cards"]) == 1

    def test_unknown_kind_returns_400(self, client: TestClient) -> None:
        resp = client.get("/curation", params={"semantic": False, "kind": "bogus"})
        assert resp.status_code == 400
        assert "Unknown kind" in resp.json()["detail"]

    def test_auth_required_when_key_set(
        self, api_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With a real API key set, the bearer-protected GET /curation rejects a
        # request that presents no credentials.
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        reset_config()
        client = TestClient(app, client=("127.0.0.1", 50000))
        resp = client.get("/curation", params={"semantic": False})
        assert resp.status_code == 401
        # The same key as a bearer token is accepted.
        ok = client.get(
            "/curation",
            params={"semantic": False},
            headers={"Authorization": "Bearer prod-secret"},
        )
        assert ok.status_code == 200


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


class TestCorpus:
    def test_deposit_text(self, client: TestClient) -> None:
        resp = client.post(
            "/corpus/deposit/text",
            json={"text": "A short test document.", "source_type": "CONVERSATION"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "entry_id" in body
        assert "snapshot_id" in body

    def test_deposit_url_upstream_fetch_failure_returns_502(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An expected upstream fetch failure (e.g. Reddit's 403 bot-wall) maps to
        502 with the origin detail — not an opaque 400."""
        from unittest.mock import AsyncMock

        from particles.http import SourceFetchError

        monkeypatch.setattr(
            "particles.api.app.deposit_url",
            AsyncMock(
                side_effect=SourceFetchError(
                    "Reddit returned HTTP 403 ... OAuth access is required.",
                    url="https://www.reddit.com/r/x/s/abc",
                    status_code=403,
                )
            ),
        )
        resp = client.post("/corpus/deposit/url", json={"url": "https://www.reddit.com/r/x/s/abc"})
        assert resp.status_code == 502
        assert "403" in resp.json()["detail"]

    def test_deposit_url_unexpected_error_returns_400(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-fetch failure still returns the generic 400 (regression)."""
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "particles.api.app.deposit_url",
            AsyncMock(side_effect=ValueError("bad url")),
        )
        resp = client.post("/corpus/deposit/url", json={"url": "https://example.com/x"})
        assert resp.status_code == 400

    def test_get_existing_entry(self, client: TestClient) -> None:
        entry_id, _ = _run_async(_add_corpus_entry(source_type="PDF"))
        resp = client.get(f"/corpus/{entry_id}")
        assert resp.status_code == 200
        assert resp.json()["entry_id"] == entry_id
        assert resp.json()["source_type"] == "PDF"

    def test_get_unknown_entry_404(self, client: TestClient) -> None:
        resp = client.get(f"/corpus/{uuid.uuid4()}")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_blob_returns_bytes_and_headers(self, client: TestClient) -> None:
        """GET /corpus/blob/{selector} streams the stored bytes + id headers."""
        content = b"<html><body><p>hi</p></body></html>"
        entry_id, snap_id = _run_async(_add_corpus_entry_with_blob(content))
        resp = client.get(f"/corpus/blob/{snap_id}")
        assert resp.status_code == 200
        assert resp.content == content
        assert resp.headers["X-Snapshot-Id"] == snap_id
        # Entry selector resolves to the same (only) snapshot's blob.
        resp2 = client.get(f"/corpus/blob/{entry_id}")
        assert resp2.status_code == 200
        assert resp2.content == content

    def test_blob_unknown_selector_404(self, client: TestClient) -> None:
        resp = client.get("/corpus/blob/deadbeef")
        assert resp.status_code == 404

    def test_blob_missing_on_disk_404(self, client: TestClient) -> None:
        """A snapshot row whose blob was never written yields 404, not a 500."""
        _, snap_id = _run_async(_add_corpus_entry())  # content_hash set, no blob saved
        resp = client.get(f"/corpus/blob/{snap_id}")
        assert resp.status_code == 404
        assert "missing" in resp.json()["detail"].lower()

    def test_deposit_file_does_not_leak_temp_on_crash(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F20: a crash mid-deposit must not leave the delete=False temp behind.

        ``corpus_deposit_file`` writes to a ``NamedTemporaryFile(delete=False)``;
        the fix moved both the create and the write inside the ``try`` whose
        ``finally`` unlinks the path, so a failure during the deposit call still
        cleans up the temp file instead of accreting one per failed upload.
        """
        import tempfile

        from particles.api import app as app_mod

        tmp_dir = Path(tempfile.gettempdir())
        before = set(tmp_dir.glob("*.txt"))

        # Make the deposit raise after the temp file has been written, simulating
        # a crash mid-request.
        async def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("simulated crash during deposit")

        monkeypatch.setattr(app_mod, "deposit_file", _boom)

        # TestClient re-raises unhandled server exceptions; the point is the
        # ``finally`` cleanup runs regardless of how the request fails.
        with pytest.raises(RuntimeError, match="simulated crash"):
            client.post(
                "/corpus/deposit/file",
                files={"file": ("doc.txt", b"some bytes", "text/plain")},
            )

        after = set(tmp_dir.glob("*.txt"))
        # No new temp file should survive the failed request.
        assert after == before


# (sibling): POST /corpus/{id}/retract.
class TestCorpusRetract:
    def test_retract_marks_live_particle_retracted(self, client: TestClient) -> None:
        entry_id, snap = _run_async(_add_corpus_entry())
        pid = _run_async(_add_particle_from_entry(entry_id, snap))
        resp = client.post(f"/corpus/{entry_id}/retract", json={"reason": "publisher correction"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["dry_run"] is False
        assert body["retracted_ids"] == [pid]
        assert _run_async(_particle_status(pid)) == Status.RETRACTED

    def test_dry_run_lists_without_writing(self, client: TestClient) -> None:
        entry_id, snap = _run_async(_add_corpus_entry())
        pid = _run_async(_add_particle_from_entry(entry_id, snap))
        resp = client.post(f"/corpus/{entry_id}/retract", json={"dry_run": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["dry_run"] is True
        assert body["retracted_ids"] == [pid]
        # Nothing written — the particle is still ACTIVE.
        assert _run_async(_particle_status(pid)) == Status.ACTIVE

    def test_unknown_entry_404(self, client: TestClient) -> None:
        resp = client.post(f"/corpus/{uuid.uuid4()}/retract", json={})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Citation-signal deposit suggestions
# ---------------------------------------------------------------------------


async def _seed_two_source_citation(url: str) -> None:
    from datetime import UTC, datetime

    from particles.corpus.store import CorpusEntryRow
    from particles.db import session_scope
    from particles.store.url_mention_store import record_url_mentions

    async with session_scope() as session:
        for i, host in enumerate(("a.example", "b.example")):
            session.add(
                CorpusEntryRow(
                    entry_id=f"src{i}",
                    uri_r=f"https://{host}/{i}",
                    source_type="WEB_PAGE",
                    mutability="MUTABLE",
                    fetch_policy="LAZY",
                    created_at=datetime.now(UTC),
                    deposited_by="test",
                )
            )
        await session.flush()
        for i in range(2):
            await record_url_mentions(session, source_entry_id=f"src{i}", canonical_urls=[url])
        await session.commit()


class TestCorpusLinksSuggest:
    def test_suggest_empty(self, client: TestClient) -> None:
        resp = client.post("/corpus/links/suggest", json={})
        assert resp.status_code == 200
        assert resp.json()["suggestions"] == []

    def test_suggest_returns_cited_url(self, client: TestClient) -> None:
        _run_async(_seed_two_source_citation("https://press.example/release"))
        resp = client.post("/corpus/links/suggest", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["suggestions"][0]["canonical_url"] == "https://press.example/release"
        assert body["suggestions"][0]["distinct_sources"] == 2

    def test_dismiss_then_suggest_excludes(self, client: TestClient) -> None:
        url = "https://press.example/release"
        _run_async(_seed_two_source_citation(url))
        dismissed = client.post("/corpus/links/dismiss", json={"url": url})
        assert dismissed.status_code == 200
        assert dismissed.json()["canonical_url"] == url
        after = client.post("/corpus/links/suggest", json={})
        assert after.json()["suggestions"] == []

    def test_dismiss_bad_url_400(self, client: TestClient) -> None:
        resp = client.post("/corpus/links/dismiss", json={"url": "not-a-url"})
        assert resp.status_code == 400
        assert "not a usable" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Extract — error paths only (LLM calls out of scope here)
# ---------------------------------------------------------------------------


class TestExtract:
    def test_unknown_entry_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/extract",
            json={"entry_id": str(uuid.uuid4()), "snapshot_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------


class TestLint:
    def test_lint_report_empty(self, client: TestClient) -> None:
        resp = client.get("/lint/report")
        assert resp.status_code == 200
        body = resp.json()
        assert "findings" in body
        assert "summary" in body
        assert body["findings"] == []

    def test_run_lint_no_semantic(self, client: TestClient) -> None:
        # semantic=False to avoid LLM calls
        resp = client.post("/lint", json={"fix": False, "semantic": False})
        assert resp.status_code == 200
        body = resp.json()
        assert "findings" in body


# ---------------------------------------------------------------------------
# Request body-size limit (F-7) + generic error messages
# ---------------------------------------------------------------------------


class TestBodySizeLimit:
    """The BodySizeLimitMiddleware caps the request body (config-driven)."""

    def test_oversized_body_rejected_413(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_MAX_REQUEST_BODY_BYTES", "100")
        reset_config()
        resp = client.post(
            "/corpus/deposit/text",
            json={"text": "x" * 500, "source_type": "CONVERSATION"},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_within_limit_passes(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_MAX_REQUEST_BODY_BYTES", "100000")
        reset_config()
        resp = client.post(
            "/corpus/deposit/text",
            json={"text": "small enough", "source_type": "CONVERSATION"},
        )
        assert resp.status_code == 200

    def test_zero_disables_the_check(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_MAX_REQUEST_BODY_BYTES", "0")
        reset_config()
        resp = client.post(
            "/corpus/deposit/text",
            json={"text": "x" * 5000, "source_type": "CONVERSATION"},
        )
        assert resp.status_code == 200


class TestDevKeyLoopbackGate:
    """End-to-end wiring of the per-request gate through FastAPI.

    Under dev-key (the test default), a non-loopback peer must be refused on
    the authenticated write endpoints (503), while unauthenticated read
    endpoints are unaffected (the gate lives in the AuthDep dependency)."""

    def test_write_endpoint_refused_from_non_loopback(self, api_db: Path) -> None:
        remote = TestClient(app, client=("203.0.113.7", 5000))
        resp = remote.post(
            "/corpus/deposit/text",
            json={"text": "from afar", "source_type": "CONVERSATION"},
        )
        assert resp.status_code == 503
        assert "non-loopback" in resp.json()["detail"]

    def test_read_endpoint_unaffected_from_non_loopback(self, api_db: Path) -> None:
        remote = TestClient(app, client=("203.0.113.7", 5000))
        assert remote.get("/health").status_code == 200


class TestErrorMessagesAreGeneric:
    """Handlers no longer echo str(exc) to clients (F-7); messages are owned."""

    def test_review_unknown_particle_message(self, client: TestClient) -> None:
        resp = client.post(
            f"/review/{uuid.uuid4()}",
            json={"action": "PREFER_A", "reviewer_id": "tester"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "No INCONSISTENCY particle with that id"

    def test_events_conflicting_refs_message(self, client: TestClient) -> None:
        resp = client.get("/events", params={"subject": "s", "particle": "p"})
        assert resp.status_code == 400
        assert "at most one" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


class TestReview:
    def test_review_queue_empty(self, client: TestClient) -> None:
        resp = client.get("/review")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_review_unknown_particle_404(self, client: TestClient) -> None:
        resp = client.post(
            f"/review/{uuid.uuid4()}",
            json={"action": "PREFER_A", "reviewer_id": "tester"},
        )
        assert resp.status_code == 404

    def test_review_over_http_logs_event(self, client: TestClient) -> None:
        # Cross-interface guarantee: resolving over POST /review/{id}
        # writes the same REVIEW_RESOLVED event the CLI verb would, because
        # record_event lives in the shared resolve() operation, not the CLI body.
        inc_id = _run_async(_add_inconsistency())
        resp = client.post(
            f"/review/{inc_id}",
            json={"action": "BOTH_VALID", "reviewer_id": "tester"},
        )
        assert resp.status_code == 200
        events = client.get("/events", params={"type": "REVIEW_RESOLVED"}).json()
        assert len(events) == 1
        assert any(r["ref_id"] == inc_id for r in events[0]["refs"])


# ---------------------------------------------------------------------------
# Reindex
# ---------------------------------------------------------------------------


class TestReindex:
    def test_reindex_empty_returns_zero(self, client: TestClient) -> None:
        resp = client.post("/reindex", json={"include_failed": False})
        assert resp.status_code == 200
        body = resp.json()
        assert body["scope"] == 0
        assert body["succeeded"] == 0
        assert body["failed"] == 0


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------


class TestSubjects:
    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get("/subjects")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_then_get(self, client: TestClient) -> None:
        create_resp = client.post(
            "/subjects",
            json={"canonical_name": "Marie Curie", "aliases": ["Skłodowska"]},
        )
        assert create_resp.status_code == 200
        sid = create_resp.json()["id"]

        get_resp = client.get(f"/subjects/{sid}")
        assert get_resp.status_code == 200
        assert get_resp.json()["canonical_name"] == "Marie Curie"

    def test_get_unknown_subject_404(self, client: TestClient) -> None:
        resp = client.get(f"/subjects/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_search_finds_match(self, client: TestClient) -> None:
        _run_async(_add_subject("Albert Einstein"))
        _run_async(_add_subject("Marie Curie"))
        resp = client.get("/subjects/search", params={"q": "einstein"})
        assert resp.status_code == 200
        names = [s["canonical_name"] for s in resp.json()]
        assert "Albert Einstein" in names
        assert "Marie Curie" not in names

    def test_particles_for_subject_empty(self, client: TestClient) -> None:
        sid = _run_async(_add_subject("Lonely Subject"))
        resp = client.get(f"/subjects/{sid}/particles")
        assert resp.status_code == 200
        assert resp.json() == []

    #: POST /subjects/{id}/split — HTTP mirror of `subjects split`.
    def test_split_relinks_particle_via_external_id(self, client: TestClient) -> None:
        source = _run_async(_add_subject("Conflated Source"))
        pid = _run_async(
            _add_particle_bound_to("Applied Optoelectronics ships from Texas.", source)
        )
        # Non-Wikidata external id → bare local construct, no network.
        resp = client.post(
            f"/subjects/{source}/split",
            json={
                "particle_ids": [pid],
                "new_name": "Applied Optoelectronics",
                "new_external_id": "numista:99999",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["relinked_particle_ids"] == [pid]
        assert body["new_subject"]["canonical_name"] == "Applied Optoelectronics"
        # The particle now points at the new subject, not the source.
        new_sid = body["new_subject"]["id"]
        moved = client.get(f"/subjects/{new_sid}/particles").json()
        assert [p["id"] for p in moved] == [pid]
        assert client.get(f"/subjects/{source}/particles").json() == []

    def test_split_unknown_source_404(self, client: TestClient) -> None:
        resp = client.post(
            f"/subjects/{uuid.uuid4()}/split",
            json={"particle_ids": [str(uuid.uuid4())], "new_external_id": "numista:1"},
        )
        assert resp.status_code == 404

    def test_split_empty_particles_400(self, client: TestClient) -> None:
        source = _run_async(_add_subject("Source"))
        resp = client.post(
            f"/subjects/{source}/split",
            json={"particle_ids": [], "new_external_id": "numista:1"},
        )
        assert resp.status_code == 400

    def test_split_missing_identity_400(self, client: TestClient) -> None:
        source = _run_async(_add_subject("Source"))
        pid = _run_async(_add_particle_bound_to("a claim", source))
        resp = client.post(
            f"/subjects/{source}/split",
            json={"particle_ids": [pid]},  # neither new_name nor new_external_id
        )
        assert resp.status_code == 400


# /db/init is not unit-tested: the endpoint runs the full alembic upgrade,
# which conflicts with the Base.metadata.create_all our fixture uses to
# avoid alembic overhead in the rest of the suite. Worth covering as an
# integration test against a clean tmp DB if/when that infra is added.


# ---------------------------------------------------------------------------
# Operator event log
# ---------------------------------------------------------------------------


async def _add_event(
    *, event_type: str = "SOURCE_RETRACTED", ref: tuple[str, str] | None = None
) -> str:
    from particles.db import session_scope
    from particles.store.event_store import EventRefKind, OperatorEventType, record_event

    refs = [(EventRefKind(ref[0]), ref[1])] if ref else []
    async with session_scope() as session:
        event = await record_event(
            session,
            actor="test",
            event_type=OperatorEventType(event_type),
            reason="r",
            refs=refs,
        )
        await session.commit()
    return event.event_id


class TestEvents:
    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get("/events")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_seeded(self, client: TestClient) -> None:
        _run_async(_add_event())
        resp = client.get("/events")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["event_type"] == "SOURCE_RETRACTED"

    def test_list_filter_by_ref(self, client: TestClient) -> None:
        _run_async(_add_event(event_type="SUBJECTS_MERGED", ref=("subject", "s-1")))
        _run_async(_add_event(event_type="SUBJECT_ALIASED", ref=("subject", "s-2")))
        resp = client.get("/events", params={"subject": "s-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["event_type"] == "SUBJECTS_MERGED"

    def test_list_conflicting_refs_400(self, client: TestClient) -> None:
        resp = client.get("/events", params={"subject": "s", "particle": "p"})
        assert resp.status_code == 400

    def test_list_bad_type_400(self, client: TestClient) -> None:
        resp = client.get("/events", params={"type": "NOPE"})
        assert resp.status_code == 400

    def test_get_existing(self, client: TestClient) -> None:
        eid = _run_async(_add_event())
        resp = client.get(f"/events/{eid}")
        assert resp.status_code == 200
        assert resp.json()["event_id"] == eid

    def test_get_unknown_404(self, client: TestClient) -> None:
        resp = client.get("/events/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Narrative endpoints
# ---------------------------------------------------------------------------


async def _add_narrative_with_constituents() -> tuple[str, list[str]]:
    """Seed a NARRATIVE over two ordered claims; return (narrative_id, [c1, c2])."""
    from particles.core.schema import ParticleType, RelationCreatedBy, RelationType
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation

    c1_id = await _add_active_particle(content="First, then.")
    c2_id = await _add_active_particle(content="Then, second.")
    narrative = Particle(
        id=str(uuid.uuid4()),
        content="A two-step narrative",
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        particle_type=ParticleType.NARRATIVE,
    )
    async with session_scope() as session:
        await insert_particle(session, narrative)
        for cid in (c1_id, c2_id):
            await create_relation(
                session, cid, narrative.id, RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
            )
        await create_relation(
            session, c1_id, c2_id, RelationType.SEQUENCE_IN, RelationCreatedBy.MANUAL_CLI
        )
        await session.commit()
    return narrative.id, [c1_id, c2_id]


class TestNarrativeEndpoints:
    def test_get_narrative_returns_sequence(self, client: TestClient) -> None:
        narrative_id, [c1_id, c2_id] = _run_async(_add_narrative_with_constituents())
        resp = client.get(f"/particles/{narrative_id}/narrative")
        assert resp.status_code == 200
        body = resp.json()
        assert body["narrative"]["id"] == narrative_id
        assert [p["id"] for p in body["constituents"]] == [c1_id, c2_id]

    def test_get_narrative_on_non_narrative_404(self, client: TestClient) -> None:
        claim_id = _run_async(_add_active_particle(content="Just a claim."))
        resp = client.get(f"/particles/{claim_id}/narrative")
        assert resp.status_code == 404

    def test_get_narrative_unknown_404(self, client: TestClient) -> None:
        resp = client.get("/particles/does-not-exist/narrative")
        assert resp.status_code == 404

    def test_get_containing_narratives(self, client: TestClient) -> None:
        narrative_id, [c1_id, _c2_id] = _run_async(_add_narrative_with_constituents())
        resp = client.get(f"/particles/{c1_id}/narratives")
        assert resp.status_code == 200
        assert [p["id"] for p in resp.json()] == [narrative_id]

    def test_containing_narratives_empty_for_unlinked(self, client: TestClient) -> None:
        claim_id = _run_async(_add_active_particle(content="Unlinked claim."))
        resp = client.get(f"/particles/{claim_id}/narratives")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_containing_narratives_unknown_404(self, client: TestClient) -> None:
        resp = client.get("/particles/does-not-exist/narratives")
        assert resp.status_code == 404

    def test_narrative_synthesis_unknown_404(self, client: TestClient) -> None:
        """synthesis endpoint 404s on a missing particle."""
        resp = client.get("/particles/does-not-exist/narrative/synthesis")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Particle not found"

    def test_narrative_synthesis_on_non_narrative_404(self, client: TestClient) -> None:
        """synthesis endpoint 404s when the particle is not a NARRATIVE."""
        claim_id = _run_async(_add_active_particle(content="Just a claim."))
        resp = client.get(f"/particles/{claim_id}/narrative/synthesis")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Particle is not a NARRATIVE"


# ---------------------------------------------------------------------------
# Operator-verb HTTP parity
# ---------------------------------------------------------------------------


class TestParticleRead:
    def test_get_particle_returns_seeded(self, client: TestClient) -> None:
        pid = _run_async(_add_active_particle(content="Parity claim."))
        resp = client.get(f"/particles/{pid}")
        assert resp.status_code == 200, resp.text
        assert resp.json()["content"] == "Parity claim."

    def test_get_particle_missing_404(self, client: TestClient) -> None:
        resp = client.get("/particles/does-not-exist")
        assert resp.status_code == 404


class TestParticleTags:
    def test_add_then_remove_tags(self, client: TestClient) -> None:
        pid = _run_async(_add_active_particle())
        add = client.post(f"/particles/{pid}/tags", json={"tags": ["alpha", "beta"]})
        assert add.status_code == 200, add.text
        assert set(add.json()["tags"]) == {"alpha", "beta"}
        rm = client.request("DELETE", f"/particles/{pid}/tags", json={"tags": ["alpha"]})
        assert rm.status_code == 200, rm.text
        assert rm.json()["tags"] == ["alpha"]

    def test_add_tags_idempotent(self, client: TestClient) -> None:
        pid = _run_async(_add_active_particle())
        client.post(f"/particles/{pid}/tags", json={"tags": ["x"]})
        again = client.post(f"/particles/{pid}/tags", json={"tags": ["x"]})
        assert again.status_code == 200
        assert again.json()["tags"] == []  # already present ⇒ nothing added


class TestLinks:
    def test_create_then_delete(self, client: TestClient) -> None:
        a = _run_async(_add_active_particle(content="link A"))
        b = _run_async(_add_active_particle(content="link B"))
        created = client.post(
            "/links",
            json={"particle_a": a, "particle_b": b, "relation_type": "CO_EVIDENTIAL"},
        )
        assert created.status_code == 200, created.text
        deleted = client.request(
            "DELETE",
            "/links",
            json={"particle_a": a, "particle_b": b, "relation_type": "CO_EVIDENTIAL"},
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True

    def test_unknown_relation_type_400(self, client: TestClient) -> None:
        a = _run_async(_add_active_particle())
        b = _run_async(_add_active_particle())
        resp = client.post(
            "/links", json={"particle_a": a, "particle_b": b, "relation_type": "BOGUS"}
        )
        assert resp.status_code == 400

    def test_self_relation_400(self, client: TestClient) -> None:
        a = _run_async(_add_active_particle())
        resp = client.post("/links", json={"particle_a": a, "particle_b": a})
        assert resp.status_code == 400


class TestSubjectAliasMerge:
    def test_add_aliases(self, client: TestClient) -> None:
        sid = _run_async(_add_subject(name="Bitcoin"))
        resp = client.post(f"/subjects/{sid}/aliases", json={"aliases": ["BTC", "XBT"]})
        assert resp.status_code == 200, resp.text
        assert set(resp.json()["added"]) == {"BTC", "XBT"}

    def test_add_aliases_missing_subject_404(self, client: TestClient) -> None:
        resp = client.post("/subjects/nope/aliases", json={"aliases": ["X"]})
        assert resp.status_code == 404

    def test_merge(self, client: TestClient) -> None:
        src = _run_async(_add_subject(name="BTC"))
        tgt = _run_async(_add_subject(name="Bitcoin"))
        resp = client.post(f"/subjects/{src}/merge", json={"target_id": tgt})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["subject"]["id"] == tgt

    def test_merge_missing_source_404(self, client: TestClient) -> None:
        tgt = _run_async(_add_subject(name="Target"))
        resp = client.post("/subjects/nope/merge", json={"target_id": tgt})
        assert resp.status_code == 404


class TestTrustWrites:
    def test_set_domain_rule(self, client: TestClient) -> None:
        resp = client.post(
            "/trust/rules",
            json={"scope": "domain", "pattern": "en.wikipedia.org", "score": 0.8},
        )
        assert resp.status_code == 200, resp.text

    def test_bad_scope_400(self, client: TestClient) -> None:
        resp = client.post("/trust/rules", json={"scope": "nonsense", "pattern": "x"})
        assert resp.status_code == 400

    def test_set_statement(self, client: TestClient) -> None:
        resp = client.post(
            "/trust/statements",
            json={
                "domain": "numismatics",
                "source_ref_type": "SOURCE_TYPE",
                "source_ref_value": "NUMISTA_COIN",
                "trust_rank": 0.9,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["cascade_resolved"] == 0  # empty store ⇒ no cascade

    def test_statement_bad_rank_400(self, client: TestClient) -> None:
        resp = client.post(
            "/trust/statements",
            json={
                "domain": "d",
                "source_ref_type": "AUTHOR",
                "source_ref_value": "x",
                "trust_rank": 1.5,
            },
        )
        assert resp.status_code == 400

    def test_statement_bad_ref_type_400(self, client: TestClient) -> None:
        resp = client.post(
            "/trust/statements",
            json={
                "domain": "d",
                "source_ref_type": "BOGUS",
                "source_ref_value": "x",
                "trust_rank": 0.5,
            },
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# MCP-read parity endpoints + belief-write endpoints.
# ---------------------------------------------------------------------------


class TestMcpReadEndpoints:
    """The thin GETs the routed MCP read tools rely on."""

    def test_list_particles(self, client: TestClient) -> None:
        pid = _run_async(_add_active_particle(content="Listed."))
        resp = client.get("/particles", params={"status": "ACTIVE"})
        assert resp.status_code == 200
        assert any(p["id"] == pid for p in resp.json())

    def test_list_particles_bad_status_400(self, client: TestClient) -> None:
        assert client.get("/particles", params={"status": "NOPE"}).status_code == 400

    def test_search_particles_by_fingerprint(self, client: TestClient) -> None:
        fp = "d" * 64
        p = Particle(
            id=str(uuid.uuid4()),
            content="Reproducible.",
            confidence=Confidence(value=0.8, calibration_source=CalibrationSource.AGENT_ASSERTED),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            asserted_at=datetime.now(UTC),
            status=Status.ACTIVE,
            context_fingerprint=fp,
        )
        _run_async(_insert(p))
        resp = client.get("/particles/search", params={"fingerprint": fp[:12]})
        assert resp.status_code == 200
        assert any(row["id"] == p.id for row in resp.json())

    def test_search_particles_bad_fingerprint_400(self, client: TestClient) -> None:
        assert client.get("/particles/search", params={"fingerprint": "xyz"}).status_code == 400

    def test_contested_backrefs_empty(self, client: TestClient) -> None:
        resp = client.get("/particles/contested")
        assert resp.status_code == 200 and resp.json() == {}

    def test_list_taxonomies_empty(self, client: TestClient) -> None:
        resp = client.get("/taxonomies")
        assert resp.status_code == 200 and resp.json() == []

    def test_list_corpus(self, client: TestClient) -> None:
        eid, _sid = _run_async(_add_corpus_entry())
        resp = client.get("/corpus")
        assert resp.status_code == 200
        assert any(e["entry_id"] == eid for e in resp.json())

    def test_subject_particle_ids(self, client: TestClient) -> None:
        sid = _run_async(_add_subject("Groschen"))
        pid = _run_async(_add_particle_bound_to("A claim about Groschen.", sid))
        resp = client.get(f"/subjects/{sid}/particle-ids")
        assert resp.status_code == 200
        body = resp.json()
        assert pid in body["particle_ids"] and body["particle_count"] == 1

    def test_digest_renders(self, client: TestClient) -> None:
        from particles.db import DEFAULT_STORE

        resp = client.get(f"/digest/{DEFAULT_STORE}")
        assert resp.status_code == 200 and resp.json()["markdown"]

    def test_digest_unknown_store_404(self, client: TestClient) -> None:
        assert client.get("/digest/no-such-store").status_code == 404


async def _insert(p: Particle) -> None:
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    async with session_scope() as session:
        await insert_particle(session, p)
        await session.commit()


@pytest.fixture
def belief_writes_enabled(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    from particles.config import get_config
    from particles.db import DEFAULT_STORE

    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]

    from unittest.mock import AsyncMock, MagicMock

    import numpy as np

    import particles.ingest.subject_resolver as sr
    from particles import embeddings as ep

    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


class TestBeliefWriteEndpoints:
    """The belief-write endpoints: the engine write gate + happy path."""

    def test_assert_disabled_403(self, client: TestClient) -> None:
        # No belief_writes_enabled fixture → default-deny → 403.
        resp = client.post(
            "/particles/assert",
            json={
                "content": "X.",
                "subject_names": ["X"],
                "confidence": 0.5,
                "source_excerpt": "x",
            },
        )
        assert resp.status_code == 403

    def test_assert_happy_path(self, client: TestClient, belief_writes_enabled: None) -> None:
        resp = client.post(
            "/particles/assert",
            json={
                "content": "The deploy key rotates monthly.",
                "subject_names": ["deploy key"],
                "confidence": 0.99,
                "source_excerpt": "we agreed it rotates monthly",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["verdict"] == "ASSERTED" and body["asserted_particle_id"]

    def test_assert_unprovenanced_400(
        self, client: TestClient, belief_writes_enabled: None
    ) -> None:
        resp = client.post(
            "/particles/assert",
            json={"content": "X.", "subject_names": ["X"], "confidence": 0.5},
        )
        assert resp.status_code == 400

    def test_retract_round_trip(self, client: TestClient, belief_writes_enabled: None) -> None:
        asserted = client.post(
            "/particles/assert",
            json={
                "content": "X holds.",
                "subject_names": ["X"],
                "confidence": 0.8,
                "source_excerpt": "x holds",
            },
        ).json()
        resp = client.post(
            "/particles/retract",
            json={"particle_id": asserted["asserted_particle_id"], "reason": "no longer"},
        )
        assert resp.status_code == 200 and resp.json()["verdict"] == "RETRACTED"

    def test_retract_disabled_403(self, client: TestClient) -> None:
        resp = client.post("/particles/retract", json={"particle_id": "x", "reason": "y"})
        assert resp.status_code == 403


def _extracted_particle(
    content: str = "An extracted claim.", *, subject_ids: list[str] | None = None
) -> Particle:
    """An extracted (general-extractor) particle — not owned by the agent identity."""
    from particles.core.schema import ProvenanceRef, ProvenanceRefType

    return Particle(
        content=content,
        confidence=Confidence(
            value=0.7,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
            calibration_method="temperature_scaling",
            calibration_ref="cref-1",
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
        asserted_by="general-extractor",
        extractor_ref={"name": "general", "version": "1.0"},
        subject_ids=subject_ids or [],
    )


class TestOperatorCurationWrites:
    """Operator-scoped curation writes + subject-assign."""

    def test_operator_retract_extracted_belief(
        self, client: TestClient, belief_writes_enabled: None
    ) -> None:
        target = _extracted_particle()
        _run_async(_insert(target))
        resp = client.post(f"/particles/{target.id}/retract", json={"reason": "spurious"})
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "RETRACTED"
        fetched = client.get(f"/particles/{target.id}").json()
        assert fetched["status"] == Status.RETRACTED.value

    def test_operator_retract_disabled_403(self, client: TestClient) -> None:
        resp = client.post("/particles/some-id/retract", json={"reason": "x"})
        assert resp.status_code == 403

    def test_operator_supersede_extracted_belief(
        self, client: TestClient, belief_writes_enabled: None
    ) -> None:
        target = _extracted_particle("X is five.")
        _run_async(_insert(target))
        resp = client.post(
            f"/particles/{target.id}/supersede",
            json={
                "content": "X is six.",
                "subject_names": ["X"],
                "confidence": 0.6,
                "source_excerpt": "actually x is six",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["verdict"] == "ASSERTED"
        old = client.get(f"/particles/{target.id}").json()
        assert old["status"] == Status.SUPERSEDED.value

    def test_operator_supersede_disabled_403(self, client: TestClient) -> None:
        resp = client.post(
            "/particles/some-id/supersede",
            json={"content": "x", "subject_names": [], "confidence": 0.5},
        )
        assert resp.status_code == 403

    def test_assign_subject_by_id_carries_provenance(
        self, client: TestClient, belief_writes_enabled: None
    ) -> None:
        # Create a Subject the operator picks, plus a NO_SUBJECT orphan.
        subj = client.post("/subjects", json={"canonical_name": "Picked Subject"}).json()
        target = _extracted_particle("An orphaned claim.")
        _run_async(_insert(target))
        resp = client.post(f"/particles/{target.id}/subjects", json={"subject_id": subj["id"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["verdict"] == "ASSERTED"
        successor = client.get(f"/particles/{body['asserted_particle_id']}").json()
        # Provenance carry-over: same content + confidence + extractor.
        assert successor["content"] == "An orphaned claim."
        assert successor["confidence"]["value"] == 0.7
        assert successor["confidence"]["calibration_source"] == "EXTRACTOR_DIRECT"
        assert successor["asserted_by"] == "general-extractor"
        assert subj["id"] in successor["subject_ids"]
        old = client.get(f"/particles/{target.id}").json()
        assert old["status"] == Status.SUPERSEDED.value

    def test_assign_subject_bad_request_400(
        self, client: TestClient, belief_writes_enabled: None
    ) -> None:
        target = _extracted_particle()
        _run_async(_insert(target))
        # Neither id nor name → 400.
        resp = client.post(f"/particles/{target.id}/subjects", json={})
        assert resp.status_code == 400

    def test_assign_subject_disabled_403(self, client: TestClient) -> None:
        resp = client.post("/particles/x/subjects", json={"subject_name": "Y"})
        assert resp.status_code == 403

    def test_affirm_records_event(self, client: TestClient, belief_writes_enabled: None) -> None:
        target = _extracted_particle()
        _run_async(_insert(target))
        resp = client.post(
            "/curation/affirm",
            json={"particle_id": target.id, "card_key": f"contested:{target.id}"},
        )
        assert resp.status_code == 200
        assert resp.json()["event_id"]
        events = client.get("/events", params={"type": "BELIEF_AFFIRMED"}).json()
        assert any(
            (e.get("payload") or {}).get("card_key") == f"contested:{target.id}" for e in events
        )

    def test_affirm_disabled_403(self, client: TestClient) -> None:
        resp = client.post("/curation/affirm", json={"particle_id": "x"})
        assert resp.status_code == 403

    def test_mark_useful_records_event_and_credit(
        self, client: TestClient, belief_writes_enabled: None
    ) -> None:
        target = _extracted_particle()
        _run_async(_insert(target))
        resp = client.post(
            "/memory/useful", json={"particle_id": target.id, "reason": "load-bearing"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["counted"] is True
        assert body["credit_key"].startswith("explicit:http:/memory/useful:")
        events = client.get("/events", params={"type": "BELIEF_MARKED_USEFUL"}).json()
        assert any((e.get("payload") or {}).get("counted") is True for e in events)

    def test_mark_useful_is_rate_bounded_per_day(
        self, client: TestClient, belief_writes_enabled: None
    ) -> None:
        target = _extracted_particle()
        _run_async(_insert(target))
        first = client.post("/memory/useful", json={"particle_id": target.id})
        second = client.post("/memory/useful", json={"particle_id": target.id})
        assert first.json()["counted"] is True
        # Recorded, but not double-counted.
        assert second.status_code == 200
        assert second.json()["counted"] is False

    def test_mark_useful_unknown_belief_400(
        self, client: TestClient, belief_writes_enabled: None
    ) -> None:
        resp = client.post("/memory/useful", json={"particle_id": "p-nope"})
        assert resp.status_code == 400

    def test_mark_useful_disabled_403(self, client: TestClient) -> None:
        resp = client.post("/memory/useful", json={"particle_id": "x"})
        assert resp.status_code == 403

    def test_snooze_records_event(self, client: TestClient, belief_writes_enabled: None) -> None:
        resp = client.post(
            "/curation/snooze",
            json={"card_key": "stale:abc", "particle_ids": ["abc"], "snooze_days": 7},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["card_key"] == "stale:abc"
        assert body["snoozed_until"] is not None

    def test_snooze_permanent_dismiss(
        self, client: TestClient, belief_writes_enabled: None
    ) -> None:
        resp = client.post(
            "/curation/snooze", json={"card_key": "stale:def", "particle_ids": ["def"]}
        )
        assert resp.status_code == 200
        assert resp.json()["snoozed_until"] is None

    def test_snooze_disabled_403(self, client: TestClient) -> None:
        resp = client.post("/curation/snooze", json={"card_key": "k"})
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Pagination caps (security review F5) — every list endpoint clamps `limit`
# to storage.max_page_size and rejects non-positive limit / negative offset.
# ---------------------------------------------------------------------------


class TestPaginationCaps:
    def _set_max_page(self, monkeypatch: pytest.MonkeyPatch, value: int) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_MAX_PAGE_SIZE", str(value))
        reset_config()

    def test_huge_limit_is_clamped_to_max_page_size(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("Aa", "Bb", "Cc", "Dd", "Ee"):
            _run_async(_add_subject(name))
        self._set_max_page(monkeypatch, 2)
        resp = client.get("/subjects", params={"limit": 2_000_000_000})
        assert resp.status_code == 200
        # Clamped to max_page_size (2), NOT the full 5-row table.
        assert len(resp.json()) == 2

    def test_omitted_limit_is_bounded_by_max_page_size(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("Aa", "Bb", "Cc", "Dd"):
            _run_async(_add_subject(name))
        self._set_max_page(monkeypatch, 3)
        # No `limit` param at all — formerly returned every subject (F5).
        resp = client.get("/subjects")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_order_degree_sorts_most_connected_first(self, client: TestClient) -> None:
        """`order=degree` returns hub subjects first (ACTIVE links desc);
        `limit=1` on top of it is the web UI's Browse-seed call."""
        lone = _run_async(_add_subject("Aardvark Lone"))  # alphabetically first, 0 links
        hub = _run_async(_add_subject("Zebra Hub"))
        for i in range(2):
            _run_async(_add_particle_bound_to(f"Hub claim {i}.", hub))

        resp = client.get("/subjects", params={"order": "degree"})
        assert resp.status_code == 200
        assert [s["id"] for s in resp.json()] == [hub, lone]
        seed = client.get("/subjects", params={"order": "degree", "limit": 1})
        assert [s["id"] for s in seed.json()] == [hub]
        # Default stays alphabetical.
        assert [s["id"] for s in client.get("/subjects").json()] == [lone, hub]

    def test_order_invalid_value_rejected(self, client: TestClient) -> None:
        assert client.get("/subjects", params={"order": "sideways"}).status_code == 422

    def test_negative_limit_rejected(self, client: TestClient) -> None:
        assert client.get("/subjects", params={"limit": -5}).status_code == 422
        assert client.get("/particles", params={"limit": -5}).status_code == 422

    def test_zero_limit_rejected(self, client: TestClient) -> None:
        assert client.get("/subjects", params={"limit": 0}).status_code == 422

    def test_negative_offset_rejected(self, client: TestClient) -> None:
        assert client.get("/subjects", params={"offset": -1}).status_code == 422
        assert client.get("/particles", params={"offset": -1}).status_code == 422

    def test_particles_huge_limit_clamped(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sid = _run_async(_add_subject("Holder"))
        for i in range(4):
            _run_async(_add_particle_bound_to(f"claim {i}", sid))
        self._set_max_page(monkeypatch, 2)
        resp = client.get("/particles", params={"limit": 2_000_000_000})
        assert resp.status_code == 200
        assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# /subjects/search hardening (security review F7) — LIKE wildcards escaped,
# `q` bounded, limit capped.
# ---------------------------------------------------------------------------


class TestSubjectSearchHardening:
    def test_percent_wildcard_is_escaped_not_match_all(self, client: TestClient) -> None:
        # Three rows with NO literal '%' and one that does. `q=%` must match only
        # the literal-'%' row — not the whole table (the F7 bug).
        for name in ("Alpha", "Beta", "Gamma"):
            _run_async(_add_subject(name))
        _run_async(_add_subject("50% Off"))
        resp = client.get("/subjects/search", params={"q": "%"})
        assert resp.status_code == 200
        names = [s["canonical_name"] for s in resp.json()]
        assert names == ["50% Off"]

    def test_underscore_wildcard_is_escaped(self, client: TestClient) -> None:
        _run_async(_add_subject("ab"))  # '_' would match any single char if unescaped
        _run_async(_add_subject("a_b"))
        resp = client.get("/subjects/search", params={"q": "a_b"})
        assert resp.status_code == 200
        names = [s["canonical_name"] for s in resp.json()]
        assert names == ["a_b"]

    def test_empty_q_rejected(self, client: TestClient) -> None:
        assert client.get("/subjects/search", params={"q": ""}).status_code == 422

    def test_overlong_q_rejected(self, client: TestClient) -> None:
        assert client.get("/subjects/search", params={"q": "x" * 201}).status_code == 422

    def test_search_limit_capped(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import reset_config

        for name in ("match one", "match two", "match three"):
            _run_async(_add_subject(name))
        monkeypatch.setenv("PARTICLES_MAX_PAGE_SIZE", "1")
        reset_config()
        resp = client.get("/subjects/search", params={"q": "match", "limit": 999})
        assert resp.status_code == 200
        assert len(resp.json()) == 1


# ---------------------------------------------------------------------------
# /query question bounds (security review F6) — schema-level rejection.
# ---------------------------------------------------------------------------


class TestQueryQuestionValidation:
    def test_overlong_question_rejected(self, client: TestClient) -> None:
        # 10000 chars > the 8192 max_length — rejected before any LLM/embedding.
        resp = client.post("/query", json={"question": "x" * 10000})
        assert resp.status_code == 422

    def test_empty_question_rejected(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_future_as_of_rejected(self, client: TestClient) -> None:
        # as-of is a historical lens; a future instant is a caller
        # bug — the QueryRequest model validator turns it into a 422.
        future = (datetime.now(UTC) + timedelta(days=30)).isoformat()
        resp = client.post("/query", json={"question": "anything?", "as_of": future})
        assert resp.status_code == 422
        assert "future" in resp.text


# ---------------------------------------------------------------------------
# GET /graph — the served GraphData contract.
# Subgraph-assembly behaviour (hops, caps, encodings, history, as-of) is
# covered at the operation level in tests/test_graph_exporter.py; here we pin
# the HTTP contract: scope validation, error mapping, and the happy path.
# ---------------------------------------------------------------------------


async def _seed_graph_pair() -> tuple[str, str, str]:
    """Two subjects joined by one multi-subject particle; returns (a, b, pid)."""
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject

    a = Subject(id=str(uuid.uuid4()), canonical_name="Alpha", asserted_by="test")
    b = Subject(id=str(uuid.uuid4()), canonical_name="Beta", asserted_by="test")
    p = Particle(
        id=str(uuid.uuid4()),
        content="Alpha borders Beta.",
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        subject_ids=[a.id, b.id],
    )
    async with session_scope() as session:
        await insert_subject(session, a)
        await insert_subject(session, b)
        await insert_particle(session, p)
        await session.commit()
    return a.id, b.id, p.id


class TestGraph:
    def test_missing_scope_is_422(self, client: TestClient) -> None:
        # Unscoped requests are rejected (the anti-hairball invariant): scope
        # is a required query param, so FastAPI validation 422s its absence.
        assert client.get("/graph").status_code == 422

    def test_subject_scope_without_selector_is_422(self, client: TestClient) -> None:
        resp = client.get("/graph", params={"scope": "subject"})
        assert resp.status_code == 422
        assert "subject_id" in resp.text

    def test_query_scope_without_selector_is_422(self, client: TestClient) -> None:
        resp = client.get("/graph", params={"scope": "query"})
        assert resp.status_code == 422
        assert "requires q" in resp.text

    def test_cross_scope_selector_is_422(self, client: TestClient) -> None:
        resp = client.get("/graph", params={"scope": "subject", "subject_id": "s", "q": "extra"})
        assert resp.status_code == 422

    def test_scope_requires_its_selector(self, client: TestClient) -> None:
        # Every scope 422s without its own selector (closed the two
        # formerly-deferred ones — they now validate like the v0 pair).
        for scope in ("subject", "query", "inconsistency", "projection"):
            resp = client.get("/graph", params={"scope": scope})
            assert resp.status_code == 422, scope
            assert "requires" in resp.text

    def test_foreign_selector_rejected(self, client: TestClient) -> None:
        resp = client.get(
            "/graph",
            params={"scope": "inconsistency", "inconsistency_id": "x", "q": "y"},
        )
        assert resp.status_code == 422
        assert "does not take" in resp.text

    def test_inconsistency_scope_renders_evidence(self, client: TestClient) -> None:
        inc_id, winner_id, loser_id = _run_async(_add_conflict_trio())
        resp = client.get("/graph", params={"scope": "inconsistency", "inconsistency_id": inc_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["scope_type"] == "inconsistency"
        assert data["scope_ref"] == inc_id
        # Anchor + both disputants render, highlighted, with true statuses —
        # including the quarantined loser an ordinary render never shows.
        for pid in (inc_id, winner_id, loser_id):
            assert data["particles"][pid]["retrieval_hit"] is True
        assert data["particles"][loser_id]["status"] == "PROVENANCE_STALE"

    def test_unknown_inconsistency_is_404(self, client: TestClient) -> None:
        resp = client.get(
            "/graph",
            params={"scope": "inconsistency", "inconsistency_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 404
        assert "unknown inconsistency id" in resp.text

    def test_unknown_manifest_is_404(self, client: TestClient) -> None:
        resp = client.get(
            "/graph",
            params={"scope": "projection", "manifest": "/nonexistent/m.yaml", "section": "x"},
        )
        assert resp.status_code == 404
        assert "unknown manifest" in resp.text

    def test_unknown_store_is_404(self, client: TestClient) -> None:
        resp = client.get("/graph", params={"scope": "subject", "subject_id": "s", "store": "nope"})
        assert resp.status_code == 404
        assert "nope" in resp.text

    def test_unknown_subject_is_404(self, client: TestClient) -> None:
        resp = client.get("/graph", params={"scope": "subject", "subject_id": str(uuid.uuid4())})
        assert resp.status_code == 404
        assert "unknown subject" in resp.text

    def test_subject_scope_renders_graphdata(self, client: TestClient) -> None:
        a_id, b_id, pid = _run_async(_seed_graph_pair())
        resp = client.get("/graph", params={"scope": "subject", "subject_id": a_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["scope_type"] == "subject"
        assert data["scope_ref"] == a_id
        node_ids = {n["subject_id"] for n in data["nodes"]}
        assert {a_id, b_id} <= node_ids
        assert any(e["particle_id"] == pid for e in data["edges"])
        # Epistemics are server-computed and ride the payload (
        # the client only renders).
        assert 0.0 < data["particles"][pid]["effective_confidence"] <= 1.0
        assert data["census"]["rendered_subjects"] == len(data["nodes"])

    def test_graph_shares_the_rate_limit_budget(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_API_RATE_LIMIT_PER_MINUTE", "1")
        reset_config()
        # Spend the host's single token on /reindex (no LLM); /graph is then
        # throttled before any embedding work (same posture as /query).
        assert client.post("/reindex", json={"include_failed": False}).status_code == 200
        resp = client.get("/graph", params={"scope": "subject", "subject_id": "s"})
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# In-app rate limiter (security review F6) — token bucket on the
# LLM/embedding-driving endpoints, keyed on the real client host. Driven via
# /reindex on an empty store, which returns 200 WITHOUT any LLM call.
# ---------------------------------------------------------------------------


class TestRateLimit:
    def _set_rate(self, monkeypatch: pytest.MonkeyPatch, value: int) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_API_RATE_LIMIT_PER_MINUTE", str(value))
        reset_config()  # reset hook clears the bucket state too

    def test_throttled_past_threshold(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_rate(monkeypatch, 1)
        first = client.post("/reindex", json={"include_failed": False})
        assert first.status_code == 200  # consumes the one token
        second = client.post("/reindex", json={"include_failed": False})
        assert second.status_code == 429
        assert second.headers.get("Retry-After") == "60"

    def test_disabled_when_zero(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_rate(monkeypatch, 0)
        for _ in range(5):
            resp = client.post("/reindex", json={"include_failed": False})
            assert resp.status_code == 200

    def test_query_shares_the_per_host_budget(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_rate(monkeypatch, 1)
        # Spend the host's single token on /reindex (no LLM), then /query is
        # throttled before its paid embedding+completion ever runs.
        assert client.post("/reindex", json={"include_failed": False}).status_code == 200
        resp = client.post("/query", json={"question": "anything"})
        assert resp.status_code == 429

    def test_structural_lint_is_not_throttled(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_rate(monkeypatch, 1)
        # Exhaust the bucket, then a structural-only lint (semantic=False, no LLM)
        # must still pass — only the semantic path is rate-limited.
        assert client.post("/reindex", json={"include_failed": False}).status_code == 200
        resp = client.post("/lint", json={"fix": False, "semantic": False})
        assert resp.status_code == 200
        # The semantic path, by contrast, is gated before run_lint's LLM call.
        assert client.post("/lint", json={"fix": False, "semantic": True}).status_code == 429


class TestReadAuthForReads:
    """Security review F2 — ``api.require_auth_for_reads`` gates the read surface.

    The bearer gates the write verbs; the bulk read routes are open
    even when a real ``PARTICLES_API_KEY`` is set. ``require_auth_for_reads=true``
    extends the bearer to those reads. ``/query``, ``/events`` and ``/digest`` are
    gated regardless of the flag. All cases below use a *real* key (the loopback
    dev-key skip is exercised in ``tests/test_auth.py``).
    """

    @staticmethod
    def _real_key(monkeypatch: pytest.MonkeyPatch, *, require_reads: bool) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        monkeypatch.setenv(
            "PARTICLES_API_REQUIRE_AUTH_FOR_READS", "true" if require_reads else "false"
        )
        reset_config()

    def test_bulk_read_open_by_default_with_real_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Flag false (the default): a bulk read route stays open with a real key
        # and no bearer — the historical posture is preserved unchanged.
        self._real_key(monkeypatch, require_reads=False)
        assert client.get("/particles").status_code == 200
        assert client.get("/quality").status_code == 200

    def test_bulk_read_rejects_without_bearer_when_required(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._real_key(monkeypatch, require_reads=True)
        assert client.get("/particles").status_code == 401
        assert client.get("/quality").status_code == 401
        assert client.get("/subjects").status_code == 401

    def test_bulk_read_rejects_invalid_bearer_when_required(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._real_key(monkeypatch, require_reads=True)
        resp = client.get("/particles", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_bulk_read_accepts_valid_bearer_when_required(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._real_key(monkeypatch, require_reads=True)
        resp = client.get("/particles", headers={"Authorization": "Bearer prod-secret"})
        assert resp.status_code == 200

    def test_query_events_digest_gated_regardless_of_flag(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.db import DEFAULT_STORE

        # Crown jewels: the bearer is required even with the flag false. The auth
        # dependency rejects before the handler body, so /query never reaches the
        # paid LLM completion.
        self._real_key(monkeypatch, require_reads=False)
        assert client.get("/events").status_code == 401
        assert client.get("/events/anything").status_code == 401
        assert client.get(f"/digest/{DEFAULT_STORE}").status_code == 401
        assert client.post("/query", json={"question": "x"}).status_code == 401

    def test_crown_jewels_accept_valid_bearer(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.db import DEFAULT_STORE

        self._real_key(monkeypatch, require_reads=False)
        h = {"Authorization": "Bearer prod-secret"}
        assert client.get("/events", headers=h).status_code == 200
        assert client.get(f"/digest/{DEFAULT_STORE}", headers=h).status_code == 200

    def test_dev_key_loopback_keeps_reads_open_even_when_required(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Local dev (dev-key + loopback peer): the flag is moot — the loopback
        # skip applies to the bulk reads and the crown jewels alike.
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        monkeypatch.setenv("PARTICLES_API_REQUIRE_AUTH_FOR_READS", "true")
        reset_config()
        assert client.get("/particles").status_code == 200
        assert client.get("/events").status_code == 200
