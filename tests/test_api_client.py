"""Tests for the thin-client backend seam (``particles/api/client``).

Covers the three things the seam must get right: the factory picks the backend
from config; ``LocalBackend`` runs in-process against the store; ``HttpBackend``
round-trips through the real FastAPI app (via an in-process ASGI transport) and
parses the JSON back into the shared Pydantic models.

The LLM/embedding-dependent verbs (query, semantic lint) are exercised
elsewhere; here we use the no-LLM endpoints (quality, structural lint, the empty
review list) so the seam is tested without mocking the model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest

from particles.api.client import get_backend
from particles.api.client.http import HttpBackend
from particles.api.client.local import LocalBackend
from particles.core.schema import LintReport, QualityReport


class TestGetBackend:
    """The factory selects the backend from ``engine.base_url``."""

    def test_defaults_to_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
        from particles.config import reset_config

        reset_config()
        backend = get_backend()
        assert isinstance(backend, LocalBackend)
        assert backend.remote is False

    def test_http_when_base_url_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://engine.test")
        from particles.config import reset_config

        reset_config()
        backend = get_backend()
        assert isinstance(backend, HttpBackend)
        assert backend.remote is True


class TestLocalBackend:
    """LocalBackend runs the operations in-process against the local store."""

    def test_quality(self, cli_db: Path) -> None:
        report = asyncio.run(LocalBackend().quality())
        assert isinstance(report, QualityReport)
        assert report.active_particles == 0

    def test_review_list_empty(self, cli_db: Path) -> None:
        assert asyncio.run(LocalBackend().review_list()) == []

    def test_lint_structural(self, cli_db: Path) -> None:
        report = asyncio.run(
            LocalBackend().lint(fix=False, semantic=False, low_coverage_threshold=3)
        )
        assert isinstance(report, LintReport)


@pytest.fixture
def http_backend(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[HttpBackend, None, None]:
    """An ``HttpBackend`` whose httpx client is wired to the in-process app.

    The engine's endpoints already run against the ``cli_db`` store; routing the
    backend's requests through ``ASGITransport`` exercises the real
    request → handler → operations → JSON → model round-trip without a socket.
    """
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://engine.test")
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
    from particles.config import reset_config

    reset_config()

    from particles.api.app import app as fastapi_app

    real_async_client = httpx.AsyncClient

    def _asgi_client(
        *, base_url: str, timeout: float, headers: dict[str, str]
    ) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.ASGITransport(app=fastapi_app),
            base_url=base_url,
            headers=headers,
        )

    monkeypatch.setattr(httpx, "AsyncClient", _asgi_client)
    yield HttpBackend()


class TestHttpBackendRoundTrip:
    """HttpBackend → FastAPI app → operations → JSON → shared model."""

    def test_quality(self, http_backend: HttpBackend) -> None:
        report = asyncio.run(http_backend.quality())
        assert isinstance(report, QualityReport)
        assert report.active_particles == 0

    def test_review_list_empty(self, http_backend: HttpBackend) -> None:
        assert asyncio.run(http_backend.review_list()) == []

    def test_lint_structural(self, http_backend: HttpBackend) -> None:
        report = asyncio.run(http_backend.lint(fix=False, semantic=False, low_coverage_threshold=3))
        assert isinstance(report, LintReport)

    def test_extract_404_surfaces_engine_detail(self, http_backend: HttpBackend) -> None:
        from particles.api.client.http import EngineHttpError

        with pytest.raises(EngineHttpError) as excinfo:
            asyncio.run(http_backend.extract("nope", "nope", agent_id="t"))
        # The engine's 404 detail is carried through, not swallowed.
        assert "404" in str(excinfo.value)


class TestHttpBackendRequiresBaseUrl:
    def test_raises_without_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
        from particles.config import reset_config

        reset_config()
        with pytest.raises(RuntimeError, match="base_url"):
            asyncio.run(HttpBackend().quality())


def _wire_unreachable_engine(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    """Point the HttpBackend at ``base_url`` and make every request connection-refused.

    Mirrors the operator who configured a remote engine but whose SSH tunnel is
    closed: the socket connect fails, so httpx raises ``ConnectError`` before any
    response. A ``MockTransport`` whose handler raises reproduces that without a
    real socket (no flaky network dependency).
    """
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", base_url)
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
    from particles.config import reset_config

    reset_config()

    real_async_client = httpx.AsyncClient

    def _refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    def _refusing_client(
        *, base_url: str, timeout: float, headers: dict[str, str]
    ) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.MockTransport(_refuse), base_url=base_url, headers=headers
        )

    monkeypatch.setattr(httpx, "AsyncClient", _refusing_client)


class TestHttpBackendUnreachable:
    """A refused connection to the engine becomes a clean, actionable error.

    Regression: an operator who forgot to open the SSH tunnel to the engine saw
    the raw ``httpx.ConnectError`` traceback from ``particles deposit`` instead
    of a message naming the unreachable engine.
    """

    def test_transport_error_translated_to_engine_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.client.http import EngineHttpError, EngineUnreachableError

        _wire_unreachable_engine(monkeypatch, "http://localhost:8000")

        with pytest.raises(EngineUnreachableError) as excinfo:
            asyncio.run(HttpBackend().quality())

        msg = str(excinfo.value)
        assert "http://localhost:8000" in msg  # names the base_url the operator set
        assert "engine.base_url" in msg  # points at the fix
        # Subclasses EngineHttpError so run()/links catch both cases uniformly.
        assert isinstance(excinfo.value, EngineHttpError)

    def test_deposit_url_translated_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The exact verb the operator hit (deposit goes through _post too).
        from particles.api.client.http import EngineUnreachableError

        _wire_unreachable_engine(monkeypatch, "http://localhost:8000")

        with pytest.raises(EngineUnreachableError):
            asyncio.run(
                HttpBackend().deposit_url(
                    "https://en.wikipedia.org/wiki/Andrej_Karpathy",
                    deposited_by="operator",
                    source_type=None,
                    tags=[],
                    follow_post_links=None,
                    follow_comment_links=None,
                )
            )

    def test_cli_deposit_exits_cleanly_without_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: `particles deposit <url>` against a down engine → exit 1.

        The verb → ``run()`` → ``HttpBackend`` path must surface the friendly
        message on stderr and exit 1, never dump a ``Traceback``.
        """
        from typer.testing import CliRunner

        from particles.api.cli import app

        _wire_unreachable_engine(monkeypatch, "http://localhost:8000")

        result = CliRunner().invoke(
            app, ["deposit", "https://en.wikipedia.org/wiki/Andrej_Karpathy"]
        )

        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert "Could not reach the Particles engine" in result.output
        assert "http://localhost:8000" in result.output


# ---------------------------------------------------------------------------
# operator-verb round-trips through the FastAPI engine.
#
# Seed via the local store (the same ``cli_db`` the in-process app serves),
# then drive the verb's method on ``HttpBackend`` so the request → handler →
# operations → JSON → shared-model round-trip is exercised end-to-end.
# ---------------------------------------------------------------------------


async def _seed_subject(name: str) -> str:
    from datetime import UTC, datetime

    from particles.core.schema import Subject
    from particles.db import session_scope
    from particles.store.subject_store import insert_subject

    subj = Subject(canonical_name=name, created_at=datetime.now(UTC), asserted_by="test")
    async with session_scope() as session:
        await insert_subject(session, subj)
        await session.commit()
    return subj.id


async def _seed_particle(content: str, *, entry_id: str | None = None) -> str:
    import uuid
    from datetime import UTC, datetime

    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    pid = str(uuid.uuid4())
    provenance = []
    if entry_id is not None:
        provenance = [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)]
    async with session_scope() as session:
        await insert_particle(
            session,
            Particle(
                id=pid,
                content=content,
                confidence=Confidence(
                    value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="test",
                asserted_at=datetime.now(UTC),
                status=Status.ACTIVE,
                provenance=provenance,
            ),
        )
        await session.commit()
    return pid


async def _seed_entry_with_particle() -> tuple[str, str]:
    import uuid
    from datetime import UTC, datetime

    from particles.core.schema import CorpusEntry, Snapshot
    from particles.corpus.store import (
        CorpusEntryRow,
        ExtractionStatus,
        SnapshotRow,
        WarcRecordType,
    )
    from particles.db import session_scope

    eid = str(uuid.uuid4())
    sid = str(uuid.uuid4())
    async with session_scope() as session:
        entry = CorpusEntry(
            entry_id=eid, source_type="WEB_PAGE", uri_r="https://example.com/x", deposited_by="test"
        )
        snap = Snapshot(
            snapshot_id=sid,
            captured_at=datetime.now(UTC),
            content_hash="c" * 64,
            extraction_status=ExtractionStatus.COMPLETE,
            warc_record_type=WarcRecordType.RESPONSE,
        )
        session.add(CorpusEntryRow.from_model(entry))
        session.add(SnapshotRow.from_model(snap, eid))
        await session.commit()
    pid = await _seed_particle("A retractable claim.", entry_id=eid)
    return eid, pid


class TestHttpBackendOperatorVerbs:
    """(a) — endpoint-backed operator verbs route through HttpBackend."""

    def test_subjects_list_and_show(self, http_backend: HttpBackend) -> None:
        from particles.core.schema import Subject

        sid = asyncio.run(_seed_subject("Pfennig"))
        subjects = asyncio.run(http_backend.subjects_list())
        assert all(isinstance(s, Subject) for s in subjects)
        assert any(s.id == sid for s in subjects)
        shown = asyncio.run(http_backend.subject_show(sid))
        assert shown is not None and shown.canonical_name == "Pfennig"
        assert asyncio.run(http_backend.subject_show("does-not-exist")) is None

    def test_subjects_search(self, http_backend: HttpBackend) -> None:
        asyncio.run(_seed_subject("Reichsmark"))
        found = asyncio.run(http_backend.subjects_search("Reichsmark"))
        assert any(s.canonical_name == "Reichsmark" for s in found)

    def test_subject_alias(self, http_backend: HttpBackend) -> None:
        sid = asyncio.run(_seed_subject("Mark"))
        outcome = asyncio.run(http_backend.subject_alias(sid, ["Deutsche Mark"]))
        assert outcome.subject.id == sid
        assert "Deutsche Mark" in outcome.added

    def test_subject_merge(self, http_backend: HttpBackend) -> None:
        source = asyncio.run(_seed_subject("Source subj"))
        target = asyncio.run(_seed_subject("Target subj"))
        outcome = asyncio.run(http_backend.subject_merge(source, target))
        assert outcome.subject.id == target
        assert isinstance(outcome.particles_relinked, int)

    def test_subject_split_dry_run_refused_remotely(self, http_backend: HttpBackend) -> None:
        from particles.api.client.base import NotYetRemoteError

        with pytest.raises(NotYetRemoteError):
            asyncio.run(
                http_backend.subject_split(
                    source_id="x",
                    particle_ids=["y"],
                    new_name="Z",
                    new_external_id=None,
                    dry_run=True,
                )
            )

    def test_particle_show(self, http_backend: HttpBackend) -> None:
        pid = asyncio.run(_seed_particle("A claim."))
        p = asyncio.run(http_backend.particle_show(pid))
        assert p is not None and p.id == pid
        assert asyncio.run(http_backend.particle_show("does-not-exist")) is None

    def test_particle_tag_untag(self, http_backend: HttpBackend) -> None:
        pid = asyncio.run(_seed_particle("Taggable claim."))
        added = asyncio.run(http_backend.particle_tag(pid, ["coins/germany"]))
        assert "coins/germany" in added
        removed = asyncio.run(http_backend.particle_untag(pid, ["coins/germany"]))
        assert "coins/germany" in removed

    def test_links_add_remove(self, http_backend: HttpBackend) -> None:
        a = asyncio.run(_seed_particle("Claim A."))
        b = asyncio.run(_seed_particle("Claim B."))
        rel = asyncio.run(
            http_backend.links_add(a, b, relation_type="CO_EVIDENTIAL", confidence=1.0)
        )
        assert {rel.particle_a, rel.particle_b} == {a, b}
        assert asyncio.run(http_backend.links_remove(a, b, relation_type="CO_EVIDENTIAL")) is True

    def test_trust_set_and_statement_set(self, http_backend: HttpBackend) -> None:
        from particles.core.schema import (
            PolicyProvenance,
            SourceRef,
            SourceRefType,
            SourceTrustStatement,
        )

        # trust_set has no read endpoint; assert the write round-trips without error.
        asyncio.run(
            http_backend.trust_set(
                scope="domain",
                pattern="en.wikipedia.org",
                score=0.9,
                modifier=None,
                rationale="trusted",
            )
        )
        stmt = SourceTrustStatement(
            domain="general",
            source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="WEB_PAGE"),
            trust_rank=0.5,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
        )
        resolved = asyncio.run(http_backend.trust_statement_set(stmt))
        assert isinstance(resolved, int)

    def test_corpus_show_and_retract(self, http_backend: HttpBackend) -> None:
        eid, pid = asyncio.run(_seed_entry_with_particle())
        entry = asyncio.run(http_backend.corpus_show(eid))
        assert entry is not None and entry.entry_id == eid
        assert asyncio.run(http_backend.corpus_show("does-not-exist")) is None
        plan = asyncio.run(http_backend.corpus_retract(eid, reason=None, dry_run=True))
        assert plan.dry_run is True and pid in plan.retracted_ids
        result = asyncio.run(http_backend.corpus_retract(eid, reason="pulled", dry_run=False))
        assert result.dry_run is False and pid in result.retracted_ids

    def test_events_list_and_show(self, http_backend: HttpBackend) -> None:
        eid, _pid = asyncio.run(_seed_entry_with_particle())
        # The retraction logs a SOURCE_RETRACTED operator event.
        asyncio.run(http_backend.corpus_retract(eid, reason="x", dry_run=False))
        events = asyncio.run(
            http_backend.events_list(
                particle=None, subject=None, entry=eid, event_type=None, limit=50
            )
        )
        assert events
        one = asyncio.run(http_backend.event_show(events[0].event_id))
        assert one is not None and one.event_id == events[0].event_id
        assert asyncio.run(http_backend.event_show("does-not-exist")) is None

    def test_corpus_links_suggest_and_dismiss(self, http_backend: HttpBackend) -> None:
        report = asyncio.run(http_backend.corpus_links_suggest(limit=None, min_sources=None))
        assert hasattr(report, "suggestions")
        outcome = asyncio.run(
            http_backend.corpus_links_dismiss(url="https://example.com/cited", snooze_days=None)
        )
        assert outcome.canonical_url

    def test_links_suggest(self, http_backend: HttpBackend) -> None:
        from particles.core.schema import SuggestMode, SuggestReport

        report = asyncio.run(
            http_backend.links_suggest(
                subject_id=None, threshold=None, mode=SuggestMode.REPORT, confirmed=False
            )
        )
        assert isinstance(report, SuggestReport)


# ---------------------------------------------------------------------------
# MCP-read round-trips through the FastAPI engine. These exercise the
# thin GETs the routed MCP read tools need (the engine endpoints + the
# HttpBackend parse). Store-only enrichment (subject names, provenance URIs)
# degrades gracefully over HTTP — asserted below.
# ---------------------------------------------------------------------------


async def _seed_particle_with_fingerprint(content: str, fingerprint: str) -> str:
    import uuid
    from datetime import UTC, datetime

    from particles.core.schema import Confidence, Particle, UncertaintyNature
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    pid = str(uuid.uuid4())
    async with session_scope() as session:
        await insert_particle(
            session,
            Particle(
                id=pid,
                content=content,
                confidence=Confidence(
                    value=0.8, calibration_source=CalibrationSource.AGENT_ASSERTED
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="test",
                asserted_at=datetime.now(UTC),
                status=Status.ACTIVE,
                context_fingerprint=fingerprint,
            ),
        )
        await session.commit()
    return pid


class TestHttpBackendMcpReads:
    """the routed MCP read tools' backend methods over HTTP."""

    def test_particle_detail_degrades_enrichment(self, http_backend: HttpBackend) -> None:
        eid, pid = asyncio.run(_seed_entry_with_particle())
        detail = asyncio.run(http_backend.particle_detail(pid))
        assert detail.particle.id == pid
        # Subject-name / URI enrichment is store-only → degraded over HTTP.
        assert detail.subjects == []
        assert detail.provenance and detail.provenance[0]["corpus_entry_id"] == eid
        assert detail.provenance[0]["uri_r"] is None
        with pytest.raises(ValueError, match="not found"):
            asyncio.run(http_backend.particle_detail("does-not-exist"))

    def test_particles_list(self, http_backend: HttpBackend) -> None:
        pid = asyncio.run(_seed_particle("A listed claim."))
        particles = asyncio.run(
            http_backend.particles_list(status="ACTIVE", subject_id=None, limit=50, offset=0)
        )
        assert any(p.id == pid for p in particles)

    def test_particles_by_fingerprint(self, http_backend: HttpBackend) -> None:
        fp = "a" * 64
        pid = asyncio.run(_seed_particle_with_fingerprint("Reproducible claim.", fp))
        rows = asyncio.run(http_backend.particles_by_fingerprint(fp[:12], limit=50))
        assert any(p.id == pid for p in rows)

    def test_subject_detail(self, http_backend: HttpBackend) -> None:
        sid = asyncio.run(_seed_subject("Thaler"))
        detail = asyncio.run(http_backend.subject_detail(sid, particle_id_limit=100))
        assert detail.subject.id == sid
        assert detail.particle_count == 0
        with pytest.raises(ValueError, match="not found"):
            asyncio.run(http_backend.subject_detail("does-not-exist", particle_id_limit=100))

    def test_subjects_list_pagination(self, http_backend: HttpBackend) -> None:
        asyncio.run(_seed_subject("Aaa subj"))
        asyncio.run(_seed_subject("Bbb subj"))
        page = asyncio.run(http_backend.subjects_list(limit=1, offset=0))
        assert len(page) == 1

    def test_list_taxonomies_empty(self, http_backend: HttpBackend) -> None:
        assert asyncio.run(http_backend.list_taxonomies()) == []

    def test_list_corpus_entries(self, http_backend: HttpBackend) -> None:
        eid, _pid = asyncio.run(_seed_entry_with_particle())
        entries = asyncio.run(http_backend.list_corpus_entries(limit=50, source_type=None))
        assert any(e.entry_id == eid for e in entries)

    def test_inconsistency_backrefs_empty(self, http_backend: HttpBackend) -> None:
        assert asyncio.run(http_backend.inconsistency_backrefs()) == {}

    def test_digest_round_trips(self, http_backend: HttpBackend) -> None:
        from particles.db import DEFAULT_STORE

        md = asyncio.run(http_backend.digest(DEFAULT_STORE))
        assert isinstance(md, str) and md  # the empty-store digest is non-empty text


# ---------------------------------------------------------------------------
# belief-write round-trips through the FastAPI engine. The §6.6
# reconciliation, server-side field construction, and event logging all run
# engine-side; the HttpBackend is the thin POST. Subject resolution + embeddings
# are stubbed so the round-trip stays offline (mirrors tests/test_mcp_write.py).
# ---------------------------------------------------------------------------


@pytest.fixture
def writes_enabled() -> Generator[None, None, None]:
    from particles.config import get_config
    from particles.db import DEFAULT_STORE

    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    yield


@pytest.fixture
def stub_subjects(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    import particles.ingest.subject_resolver as sr

    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))


@pytest.fixture
def stub_embeddings() -> Generator[None, None, None]:
    from unittest.mock import MagicMock

    import numpy as np

    from particles import embeddings as ep

    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


class TestHttpBackendBeliefWrites:
    """assert / supersede / retract / deposit_text over HTTP."""

    def test_assert_round_trips_with_reconciliation(
        self,
        http_backend: HttpBackend,
        writes_enabled: None,
        stub_subjects: None,
        stub_embeddings: None,
    ) -> None:
        from particles.db import DEFAULT_STORE

        result = asyncio.run(
            http_backend.particle_assert(
                content="The deploy key rotates monthly.",
                subject_names=["deploy key"],
                confidence=0.99,  # above the 0.90 ceiling — clamped server-side
                source_excerpt="we agreed the deploy key rotates monthly",
                corpus_entry_id=None,
                uncertainty_nature="EPISTEMIC",
                tags=None,
                store=DEFAULT_STORE,
            )
        )
        assert result.verdict == "ASSERTED"
        assert result.asserted_particle_id is not None
        # The §6.6 construction ran ENGINE-side: clamp + AGENT_ASSERTED applied.
        p = asyncio.run(_get_particle_via_store(result.asserted_particle_id))
        assert p is not None
        assert p.confidence.value == pytest.approx(0.90)
        from particles.core.scoring.confidence import CalibrationSource

        assert p.confidence.calibration_source == CalibrationSource.AGENT_ASSERTED

    def test_assert_disabled_503_or_403(self, http_backend: HttpBackend) -> None:
        from particles.api.client.http import EngineHttpError
        from particles.db import DEFAULT_STORE

        # No writes_enabled fixture → the engine's write gate refuses.
        with pytest.raises(EngineHttpError) as excinfo:
            asyncio.run(
                http_backend.particle_assert(
                    content="X.",
                    subject_names=["X"],
                    confidence=0.5,
                    source_excerpt="x",
                    corpus_entry_id=None,
                    uncertainty_nature="EPISTEMIC",
                    tags=None,
                    store=DEFAULT_STORE,
                )
            )
        assert "403" in str(excinfo.value)

    def test_supersede_and_retract_round_trip(
        self,
        http_backend: HttpBackend,
        writes_enabled: None,
        stub_subjects: None,
        stub_embeddings: None,
    ) -> None:
        from particles.core.status import Status
        from particles.db import DEFAULT_STORE

        first = asyncio.run(
            http_backend.particle_assert(
                content="X is 5.",
                subject_names=["X"],
                confidence=0.8,
                source_excerpt="x is 5",
                corpus_entry_id=None,
                uncertainty_nature="EPISTEMIC",
                tags=None,
                store=DEFAULT_STORE,
            )
        )
        old_id = first.asserted_particle_id
        assert old_id is not None
        out = asyncio.run(
            http_backend.particle_supersede(
                supersedes_id=old_id,
                content="X is 6.",
                subject_names=["X"],
                confidence=0.85,
                source_excerpt="actually x is 6",
                corpus_entry_id=None,
                uncertainty_nature="EPISTEMIC",
                tags=None,
                store=DEFAULT_STORE,
            )
        )
        assert out.verdict == "ASSERTED"
        old = asyncio.run(_get_particle_via_store(old_id))
        assert old is not None and old.status == Status.SUPERSEDED

        new_id = out.asserted_particle_id
        assert new_id is not None
        asyncio.run(
            http_backend.particle_retract(
                particle_id=new_id, reason="no longer holds", store=DEFAULT_STORE
            )
        )
        retracted = asyncio.run(_get_particle_via_store(new_id))
        assert retracted is not None and retracted.status == Status.RETRACTED

    def test_deposit_text_round_trips(
        self, http_backend: HttpBackend, writes_enabled: None
    ) -> None:
        from particles.db import DEFAULT_STORE

        entry_id, snapshot_id = asyncio.run(
            http_backend.deposit_text(text="a worth-archiving note", tags=None, store=DEFAULT_STORE)
        )
        assert entry_id and snapshot_id


async def _get_particle_via_store(pid: str) -> object:
    from particles.db import session_scope
    from particles.store.particle_store import get_particle

    async with session_scope() as session:
        return await get_particle(session, pid)


class TestPlaintextTokenWarning:
    """Security review F18 — warn when a bearer token rides plaintext non-loopback http://.

    The documented remote-engine setup tunnels http:// over Tailscale / SSH, so
    loopback http:// must stay silent; only a *non-loopback* http:// with a
    token present is exposed and warns (once). https:// never warns.
    """

    @staticmethod
    def _reset_warn_cache() -> None:
        # _warn_plaintext_token_once is lru_cache'd to fire once per base_url for
        # the process; clear it so each case starts from a clean slate.
        from particles.api.client.http import _warn_plaintext_token_once

        _warn_plaintext_token_once.cache_clear()

    def test_non_loopback_http_with_token_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from particles.api.client.http import _maybe_warn_plaintext_token

        self._reset_warn_cache()
        with caplog.at_level(logging.WARNING, logger="particles.api.client.http"):
            _maybe_warn_plaintext_token("http://mac-mini:8000", has_token=True)
        assert any("unencrypted" in r.message for r in caplog.records)
        assert any("mac-mini" in r.message for r in caplog.records)

    def test_loopback_http_with_token_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from particles.api.client.http import _maybe_warn_plaintext_token

        for base_url in ("http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000"):
            self._reset_warn_cache()
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="particles.api.client.http"):
                _maybe_warn_plaintext_token(base_url, has_token=True)
            assert not caplog.records, f"{base_url} should not warn"

    def test_https_with_token_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from particles.api.client.http import _maybe_warn_plaintext_token

        self._reset_warn_cache()
        with caplog.at_level(logging.WARNING, logger="particles.api.client.http"):
            _maybe_warn_plaintext_token("https://mac-mini:8000", has_token=True)
        assert not caplog.records

    def test_no_token_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from particles.api.client.http import _maybe_warn_plaintext_token

        self._reset_warn_cache()
        with caplog.at_level(logging.WARNING, logger="particles.api.client.http"):
            _maybe_warn_plaintext_token("http://mac-mini:8000", has_token=False)
        assert not caplog.records

    def test_warns_at_most_once_per_base_url(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from particles.api.client.http import _maybe_warn_plaintext_token

        self._reset_warn_cache()
        with caplog.at_level(logging.WARNING, logger="particles.api.client.http"):
            _maybe_warn_plaintext_token("http://mac-mini:8000", has_token=True)
            _maybe_warn_plaintext_token("http://mac-mini:8000", has_token=True)
            _maybe_warn_plaintext_token("http://mac-mini:8000", has_token=True)
        warnings = [r for r in caplog.records if "unencrypted" in r.message]
        assert len(warnings) == 1
