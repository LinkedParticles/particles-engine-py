"""the MCP server's tools route through the engine backend seam.

With ``engine.base_url`` set, the local stdio MCP tools must operate on the
**canonical engine** over HTTP rather than the laptop's local store (closing the
split-brain). These tests wire ``get_backend()``'s ``HttpBackend`` through the
in-process FastAPI app (the engine) via ``ASGITransport``, record the HTTP paths
the engine receives, and call the MCP tool functions directly — proving each tool
reached the engine endpoint (reads) and that belief writes run §6.6 reconciliation
server-side (writes). With no engine configured the tools stay in-process
(covered by tests/test_mcp_server.py / test_mcp_write.py).
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest

# Import the MCP package (and the `mcp` SDK it pulls in) eagerly, at collection
# time, BEFORE the `engine` fixture monkeypatches httpx.AsyncClient. The mcp SDK's
# streamable_http module evaluates an `httpx.AsyncClient | None` annotation at
# import time on Python 3.11 (the CI floor); if AsyncClient has been patched to a
# function first, that becomes `function | None` → TypeError. Eager import loads
# it with the real class; the deferred tool imports inside the tests are then
# cache hits.
import particles.mcp  # noqa: F401
from particles.db import DEFAULT_STORE, session_scope


@pytest.fixture
def engine(cli_db: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[list[str], None, None]:
    """Route ``HttpBackend`` through the in-process engine; yield the recorded paths.

    The engine serves the same ``cli_db`` default store the seed helpers write, so
    a routed read/write is observable; the recorded path list proves the tool took
    the HTTP path rather than the in-process one. Writes are enabled on the
    default store so the tools are offered locally and the engine gate passes.
    """
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://engine.test")
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
    from particles.config import get_config, reset_config

    reset_config()
    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]

    from particles.api.app import app as fastapi_app

    recorded: list[str] = []

    async def _recorder(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            recorded.append(scope["path"])
        await fastapi_app(scope, receive, send)

    real_async_client = httpx.AsyncClient

    def _asgi_client(
        *, base_url: str, timeout: float, headers: dict[str, str]
    ) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.ASGITransport(app=_recorder), base_url=base_url, headers=headers
        )

    monkeypatch.setattr(httpx, "AsyncClient", _asgi_client)
    yield recorded


@pytest.fixture
def stub_subjects(monkeypatch: pytest.MonkeyPatch) -> None:
    import particles.ingest.subject_resolver as sr

    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))


@pytest.fixture
def similar_embeddings() -> Generator[None, None, None]:
    from particles import embeddings as ep

    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


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
    from particles.store.particle_store import insert_particle

    pid = str(uuid.uuid4())
    provenance = (
        [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)] if entry_id else []
    )
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
                provenance=provenance,
            ),
        )
        await session.commit()
    return pid


class TestMcpReadRouting:
    """Every routed MCP read tool hits its engine endpoint when an engine is set."""

    def test_particles_list_routes_to_engine(self, engine: list[str]) -> None:
        from particles.mcp.tools.particles import particles_list

        pid = asyncio.run(_seed_particle("A routed claim."))
        result = asyncio.run(particles_list(status="ACTIVE"))
        assert any(p["id"] == pid for p in result["particles"])
        assert "/particles" in engine
        assert "/particles/contested" in engine  # the §7 contested marker is routed too
        # the badge is composed engine-side by the one
        # composer, not hand-rolled in the tool — so it is a routed call.
        assert "/particles/contested-badges" in engine

    def test_particle_show_routes_and_degrades(self, engine: list[str]) -> None:
        from particles.mcp.tools.particles import particle_show

        pid = asyncio.run(_seed_particle("Shown over HTTP."))
        result = asyncio.run(particle_show(pid))
        assert result["particle"]["id"] == pid
        # Subject-name enrichment is store-only → degraded over the engine.
        assert result["subjects"] == []
        assert f"/particles/{pid}" in engine

    def test_subjects_list_routes_to_engine(self, engine: list[str]) -> None:
        from particles.mcp.tools.subjects import subjects_list

        asyncio.run(subjects_list())
        assert "/subjects" in engine

    def test_quality_report_routes_to_engine(self, engine: list[str]) -> None:
        from particles.mcp.tools.diagnostics import quality_report

        report = asyncio.run(quality_report())
        assert "active_particles" in report
        assert "/quality" in engine

    def test_lint_routes_to_engine(self, engine: list[str]) -> None:
        from particles.mcp.tools.lint import lint

        asyncio.run(lint())
        assert "/lint" in engine

    def test_corpus_listing_routes_to_engine(self, engine: list[str]) -> None:
        from particles.mcp.tools.corpus import list_corpus_entries

        asyncio.run(list_corpus_entries())
        assert "/corpus" in engine

    def test_list_taxonomies_routes_to_engine(self, engine: list[str]) -> None:
        from particles.mcp.tools.taxonomy import list_taxonomies

        asyncio.run(list_taxonomies())
        assert "/taxonomies" in engine

    def test_digest_resource_routes_to_engine(self, engine: list[str]) -> None:
        from particles.mcp.resources import _render_digest

        md = asyncio.run(_render_digest(DEFAULT_STORE))
        assert isinstance(md, str) and md
        assert f"/digest/{DEFAULT_STORE}" in engine

    def test_graph_view_routes_to_engine(self, engine: list[str]) -> None:
        """graph_view hits GET /graph and carries the /app#/browse deep link
        (an engine is configured, so the url field is present)."""
        import uuid

        from particles.core.schema import Subject
        from particles.mcp.tools.graph import graph_view
        from particles.store.subject_store import insert_subject

        async def _seed_subject() -> str:
            subj = Subject(id=str(uuid.uuid4()), canonical_name="Routed", asserted_by="test")
            async with session_scope() as session:
                await insert_subject(session, subj)
                await session.commit()
            return subj.id

        sid = asyncio.run(_seed_subject())
        result = asyncio.run(graph_view(subject_id=sid))
        assert result["scope_type"] == "subject"
        assert any(n["subject_id"] == sid for n in result["nodes"])
        assert "/graph" in engine
        assert result["url"].startswith("http://engine.test/app#/browse?scope=subject")


class TestMcpWriteRouting:
    """Belief writes run §6.6 reconciliation engine-side when routed."""

    def test_assert_routes_and_constructs_server_side(
        self, engine: list[str], stub_subjects: None, similar_embeddings: None
    ) -> None:
        from particles.core.scoring.confidence import CalibrationSource
        from particles.mcp.tools.write import particle_assert
        from particles.store.particle_store import get_particle

        out = asyncio.run(
            particle_assert(
                "The deploy key rotates monthly.",
                ["deploy key"],
                0.99,  # above the 0.90 ceiling
                source_excerpt="we agreed it rotates monthly",
            )
        )
        assert out["verdict"] == "ASSERTED"
        assert "/particles/assert" in engine

        async def _read() -> Any:
            async with session_scope(DEFAULT_STORE) as s:
                return await get_particle(s, out["asserted_particle_id"])

        p = asyncio.run(_read())
        assert p is not None
        # Construction ran ENGINE-side: clamp + AGENT_ASSERTED applied there.
        assert p.confidence.value == pytest.approx(0.90)
        assert p.confidence.calibration_source == CalibrationSource.AGENT_ASSERTED

    def test_contradiction_reconciles_server_side(
        self,
        engine: list[str],
        stub_subjects: None,
        similar_embeddings: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A confirmed contradiction surfaces as INCONSISTENCY over HTTP — proving
        the §6.6 consensus ladder ran on the engine, not the client."""
        import particles.llm as llm
        from particles.mcp.tools.write import particle_assert

        monkeypatch.setattr(llm, "complete", AsyncMock(return_value="YES: they disagree"))

        a = asyncio.run(
            particle_assert(
                "Deploy key rotates monthly.", ["deploy key"], 0.8, source_excerpt="rotates monthly"
            )
        )
        assert a["verdict"] == "ASSERTED"
        b = asyncio.run(
            particle_assert(
                "Deploy key never rotates.", ["deploy key"], 0.9, source_excerpt="never rotates"
            )
        )
        assert b["verdict"] == "INCONSISTENCY_RAISED"
        assert b["inconsistency_id"] and b["asserted_particle_id"] != b["inconsistency_id"]

    def test_retract_routes_to_engine(
        self, engine: list[str], stub_subjects: None, similar_embeddings: None
    ) -> None:
        from particles.core.status import Status
        from particles.mcp.tools.write import particle_assert, particle_retract
        from particles.store.particle_store import get_particle

        asserted = asyncio.run(particle_assert("X holds.", ["X"], 0.8, source_excerpt="x holds"))
        pid = asserted["asserted_particle_id"]
        out = asyncio.run(particle_retract(pid, reason="no longer holds"))
        assert out["verdict"] == "RETRACTED"
        assert "/particles/retract" in engine

        async def _read() -> Any:
            async with session_scope(DEFAULT_STORE) as s:
                return await get_particle(s, pid)

        p = asyncio.run(_read())
        assert p is not None and p.status == Status.RETRACTED

    def test_deposit_text_routes_to_engine(self, engine: list[str]) -> None:
        from particles.mcp.tools.write import deposit_text

        out = asyncio.run(deposit_text("worth archiving"))
        assert out["corpus_entry_id"]
        assert "/corpus/deposit/text" in engine
