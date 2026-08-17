"""Unit tests for the read-only MCP server.

Each tool handler is an async callable registered on FastMCP; we invoke
each one directly through ``server.call_tool(name, args)`` with an
in-memory DB session — same shape as ``tests/test_app.py``'s FastAPI
handler tests. The transport layer (stdio) is the SDK's concern; we
trust that and test our handlers.

A golden tool-schema file (``tests/mcp/tool-schema.json``) pins the
contract — if an ``operations/`` signature changes and the MCP surface
needs an update, regenerate via ``particles mcp tools > tests/mcp/tool-schema.json``
in the same PR.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    Subject,
    TagNode,
    TaxonomyDefinition,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from tests._upstream import upstream_only

_GOLDEN_TOOL_SCHEMA = Path(__file__).parent / "mcp" / "tool-schema.json"
_GOLDEN_RESOURCE_SCHEMA = Path(__file__).parent / "mcp" / "resource-schema.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call_tool(server: Any, name: str, args: dict[str, Any]) -> Any:
    """Invoke a registered tool by name and return the parsed-JSON result.

    FastMCP's ``call_tool`` returns a ``(ContentList, StructuredResult)``
    tuple; the structured result is the dict our handlers return.
    """
    result = await server.call_tool(name, args)
    # FastMCP shape: (content_blocks, structured_result_dict)
    if isinstance(result, tuple) and len(result) == 2:
        _, structured = result
        return structured
    return result


def _make_particle(
    *,
    content: str = "A test claim.",
    tags: list[str] | None = None,
    subject_ids: list[str] | None = None,
    fingerprint: str | None = None,
    status: Status = Status.ACTIVE,
) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=status,
        tags=tags,
        subject_ids=subject_ids or [],
        context_fingerprint=fingerprint,
    )


# ---------------------------------------------------------------------------
# Tool-handler unit tests
# ---------------------------------------------------------------------------


class TestServerRegistration:
    @pytest.mark.asyncio
    async def test_all_tools_registered(self) -> None:
        from particles.mcp import build_server

        server = build_server()
        tools = await server.list_tools()
        names = sorted(t.name for t in tools)
        assert names == sorted(
            [
                "query",
                "particle_show",
                "particle_search",
                "particles_list",
                "subjects_list",
                "subjects_search",
                "subjects_show",
                "list_taxonomies",
                "list_corpus_entries",
                "lint",
                "quality_report",
                "links_suggest",
                "corpus_links_suggest",
                "events_list",
                "event_show",
                "graph_view",
            ]
        )

    @pytest.mark.asyncio
    async def test_every_tool_has_a_docstring_description(self) -> None:
        from particles.mcp import build_server

        server = build_server()
        for t in await server.list_tools():
            assert t.description, f"Tool {t.name} has no description"

    def test_every_tool_parameter_is_documented(self) -> None:
        """Drift-guard: every MCP tool parameter is mentioned by name
        in the tool's description.

        The MCP description IS the full tool docstring (including its ``Args:``
        block), and the generated input-schema parameters carry no descriptions
        of their own — so the docstring is the only place per-parameter agent
        documentation can live. This guards the exact silent drift flagged
        : adding or renaming a parameter without updating the prose now
        fails CI, where the golden ``tool-schema.json`` snapshot cannot tell a
        stale description from a current one. Covers the read and write surface
        (the latter is registered only when a store is write-enabled).
        """
        import inspect

        from particles.mcp.server import (
            _TOOL_REGISTRATION_ORDER,
            _WRITE_TOOL_REGISTRATION_ORDER,
        )

        for fn in (*_TOOL_REGISTRATION_ORDER, *_WRITE_TOOL_REGISTRATION_ORDER):
            doc = inspect.getdoc(fn) or ""
            assert doc.strip(), f"Tool {fn.__name__} has no description"
            assert len(doc.split()) >= 5, f"Tool {fn.__name__} description is too thin"
            params = [
                name
                for name, p in inspect.signature(fn).parameters.items()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                and name not in ("self", "ctx")
            ]
            for name in params:
                assert name in doc, (
                    f"Tool {fn.__name__} parameter {name!r} is not documented in its "
                    "description (drift-guard) — update the docstring's Args block."
                )


class TestSubjectsTools:
    @pytest.mark.asyncio
    async def test_subjects_list_returns_inserted_subjects(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.subject_store import insert_subject

        await insert_subject(db_session, Subject(canonical_name="AdamW", asserted_by="t"))
        await insert_subject(db_session, Subject(canonical_name="Adam", asserted_by="t"))
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "subjects_list", {})
        names = {s["canonical_name"] for s in result["result"]}
        assert {"AdamW", "Adam"} <= names

    @pytest.mark.asyncio
    async def test_subjects_list_paginates_with_limit_and_offset(
        self, db_session: AsyncSession
    ) -> None:
        from particles.mcp import build_server
        from particles.store.subject_store import insert_subject

        # Insert in non-alphabetical order; the tool returns by canonical_name.
        names = ["Charlie", "Alpha", "Echo", "Bravo", "Delta"]
        for n in names:
            await insert_subject(db_session, Subject(canonical_name=n, asserted_by="t"))
        await db_session.commit()

        server = build_server()

        # limit caps the page size, ordering is by canonical_name asc.
        first = await _call_tool(server, "subjects_list", {"limit": 2})
        first_names = [s["canonical_name"] for s in first["result"]]
        assert first_names == ["Alpha", "Bravo"]

        # offset skips the first N; limit then caps what follows.
        second = await _call_tool(server, "subjects_list", {"limit": 2, "offset": 2})
        second_names = [s["canonical_name"] for s in second["result"]]
        assert second_names == ["Charlie", "Delta"]

        # offset past the end yields an empty page (not an error).
        empty = await _call_tool(server, "subjects_list", {"limit": 2, "offset": 100})
        assert empty["result"] == []

    @pytest.mark.asyncio
    async def test_subjects_search_substring(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.subject_store import insert_subject

        await insert_subject(db_session, Subject(canonical_name="AdamW Optimizer", asserted_by="t"))
        await insert_subject(db_session, Subject(canonical_name="Stochastic GD", asserted_by="t"))
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "subjects_search", {"query": "adam"})
        names = {s["canonical_name"] for s in result["result"]}
        assert names == {"AdamW Optimizer"}

    @pytest.mark.asyncio
    async def test_subjects_search_paginates_with_limit_and_offset(
        self, db_session: AsyncSession
    ) -> None:
        from particles.mcp import build_server
        from particles.store.subject_store import insert_subject

        # All match the substring "Foo"; ordering by canonical_name is stable.
        names = ["FooDelta", "FooAlpha", "FooCharlie", "FooBravo", "FooEcho"]
        for n in names:
            await insert_subject(db_session, Subject(canonical_name=n, asserted_by="t"))
        await db_session.commit()

        server = build_server()
        first = await _call_tool(server, "subjects_search", {"query": "foo", "limit": 2})
        assert [s["canonical_name"] for s in first["result"]] == ["FooAlpha", "FooBravo"]

        second = await _call_tool(
            server, "subjects_search", {"query": "foo", "limit": 2, "offset": 2}
        )
        assert [s["canonical_name"] for s in second["result"]] == ["FooCharlie", "FooDelta"]

        # offset past the end yields an empty page (not an error).
        empty = await _call_tool(
            server, "subjects_search", {"query": "foo", "limit": 2, "offset": 100}
        )
        assert empty["result"] == []

    @pytest.mark.asyncio
    async def test_subjects_show_returns_linked_particle_ids(
        self, db_session: AsyncSession
    ) -> None:
        from particles.mcp import build_server
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        subj = Subject(canonical_name="X", asserted_by="t")
        await insert_subject(db_session, subj)
        p1 = _make_particle(subject_ids=[subj.id])
        p2 = _make_particle(subject_ids=[subj.id])
        await insert_particle(db_session, p1)
        await insert_particle(db_session, p2)
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "subjects_show", {"subject_id": subj.id})
        assert result["subject"]["canonical_name"] == "X"
        assert set(result["particle_ids"]) == {p1.id, p2.id}
        assert result["particle_count"] == 2

    @pytest.mark.asyncio
    async def test_subjects_show_caps_particle_ids_via_particle_id_limit(
        self, db_session: AsyncSession
    ) -> None:
        from particles.mcp import build_server
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        subj = Subject(canonical_name="HotSubject", asserted_by="t")
        await insert_subject(db_session, subj)
        # 10 particles linked to the same subject
        linked = [_make_particle(subject_ids=[subj.id]) for _ in range(10)]
        for p in linked:
            await insert_particle(db_session, p)
        await db_session.commit()

        server = build_server()
        result = await _call_tool(
            server,
            "subjects_show",
            {"subject_id": subj.id, "particle_id_limit": 3},
        )
        # particle_ids is capped at the limit; particle_count gives the true total.
        assert len(result["particle_ids"]) == 3
        assert result["particle_count"] == 10
        # All returned IDs belong to the linked particles set (a valid prefix).
        assert set(result["particle_ids"]).issubset({p.id for p in linked})

    @pytest.mark.asyncio
    async def test_subjects_show_missing_raises(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server

        server = build_server()
        # FastMCP wraps handler exceptions; just verify the call surfaces an error.
        with pytest.raises(Exception):  # noqa: B017 — error type varies by SDK version
            await _call_tool(server, "subjects_show", {"subject_id": "no-such"})


class TestParticleTools:
    @pytest.mark.asyncio
    async def test_particle_show_returns_subjects_and_provenance(
        self, db_session: AsyncSession
    ) -> None:
        from particles.corpus.store import CorpusEntryRow, SnapshotRow
        from particles.mcp import build_server
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        entry = CorpusEntry(uri_r="https://example.com/x", source_type="WEB_PAGE", deposited_by="t")
        snap = Snapshot(
            captured_at=datetime.now(UTC),
            content_hash="a" * 64,
            extraction_status=ExtractionStatus.COMPLETE,
            warc_record_type=WarcRecordType.RESPONSE,
        )
        db_session.add(CorpusEntryRow.from_model(entry))
        db_session.add(SnapshotRow.from_model(snap, entry.entry_id))

        subj = Subject(canonical_name="Y", asserted_by="t")
        await insert_subject(db_session, subj)

        p = _make_particle(subject_ids=[subj.id])
        p.provenance = [
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry.entry_id,
                snapshot_id=snap.snapshot_id,
            )
        ]
        await insert_particle(db_session, p)
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "particle_show", {"particle_id": p.id})
        assert result["particle"]["content"] == p.content
        assert result["subjects"] == [{"id": subj.id, "canonical_name": "Y"}]
        assert result["provenance"][0]["uri_r"] == "https://example.com/x"
        assert result["provenance"][0]["source_type"] == "WEB_PAGE"

    @pytest.mark.asyncio
    async def test_particle_show_accepts_prefix(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.particle_store import insert_particle

        p = _make_particle()
        await insert_particle(db_session, p)
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "particle_show", {"particle_id": p.id[:8]})
        assert result["particle"]["id"] == p.id

    @pytest.mark.asyncio
    async def test_particle_search_by_fingerprint(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.particle_store import insert_particle

        fp = "a" * 64
        p = _make_particle(fingerprint=fp)
        await insert_particle(db_session, p)
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "particle_search", {"fingerprint": fp})
        assert result["fingerprint"] == fp
        assert any(item["id"] == p.id for item in result["particles"])

    @pytest.mark.asyncio
    async def test_particle_search_rejects_short_prefix(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server

        server = build_server()
        with pytest.raises(Exception):  # noqa: B017
            await _call_tool(server, "particle_search", {"fingerprint": "abc"})

    @pytest.mark.asyncio
    async def test_particle_search_rejects_oversized_limit(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server

        server = build_server()
        # limit > 200 caps to keep response bounded — fingerprint matches
        # share the full content string per row, so the worst case scales
        # with (matches × content_length).
        with pytest.raises(Exception, match="limit must be 200 or less"):
            await _call_tool(
                server,
                "particle_search",
                {"fingerprint": "0" * 64, "limit": 201},
            )
        with pytest.raises(Exception, match="limit must be a positive integer"):
            await _call_tool(
                server,
                "particle_search",
                {"fingerprint": "0" * 64, "limit": 0},
            )


class TestParticlesList:
    @pytest.mark.asyncio
    async def test_status_filter_returns_only_matching(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.particle_store import insert_particle, update_particle_status

        active = _make_particle(content="active claim")
        inconsist = _make_particle(content="inconsistency claim", status=Status.INCONSISTENCY)
        await insert_particle(db_session, active)
        await insert_particle(db_session, inconsist)
        # The other particle stays INCONSISTENCY (born that way via _make_particle).
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "particles_list", {"status": "INCONSISTENCY"})
        ids = {p["id"] for p in result["particles"]}
        assert ids == {inconsist.id}
        assert all(p["status"] == "INCONSISTENCY" for p in result["particles"])
        # Embedding bytes must never leak into the response.
        assert all("embedding" not in p and "embedding_json" not in p for p in result["particles"])

        # Verify a different status filter also works (RETRACTED needs a transition).
        await update_particle_status(db_session, active.id, Status.RETRACTED)
        await db_session.commit()
        result = await _call_tool(server, "particles_list", {"status": "RETRACTED"})
        ids = {p["id"] for p in result["particles"]}
        assert ids == {active.id}

    @pytest.mark.asyncio
    async def test_subject_filter_returns_only_linked(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        subj_a = Subject(canonical_name="SubjectA", asserted_by="t")
        subj_b = Subject(canonical_name="SubjectB", asserted_by="t")
        await insert_subject(db_session, subj_a)
        await insert_subject(db_session, subj_b)

        linked = _make_particle(content="about A", subject_ids=[subj_a.id])
        unlinked = _make_particle(content="about B", subject_ids=[subj_b.id])
        await insert_particle(db_session, linked)
        await insert_particle(db_session, unlinked)
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "particles_list", {"subject_id": subj_a.id})
        ids = {p["id"] for p in result["particles"]}
        assert ids == {linked.id}
        assert subj_a.id in result["particles"][0]["subject_ids"]

    @pytest.mark.asyncio
    async def test_pagination_via_limit_and_offset(self, db_session: AsyncSession) -> None:
        from datetime import UTC, datetime, timedelta

        from particles.mcp import build_server
        from particles.store.particle_store import insert_particle

        # Insert five particles with distinct asserted_at so the ordering
        # (desc by asserted_at) is deterministic across pages.
        base = datetime(2026, 1, 1, tzinfo=UTC)
        particles = []
        for i in range(5):
            p = _make_particle(content=f"claim {i}")
            p.asserted_at = base + timedelta(hours=i)
            await insert_particle(db_session, p)
            particles.append(p)
        await db_session.commit()

        server = build_server()

        first = await _call_tool(server, "particles_list", {"limit": 2, "offset": 0})
        assert len(first["particles"]) == 2
        assert first["limit"] == 2
        assert first["offset"] == 0
        # asserted_at desc → most recent (claim 4, claim 3) first.
        assert [p["content"] for p in first["particles"]] == ["claim 4", "claim 3"]

        second = await _call_tool(server, "particles_list", {"limit": 2, "offset": 2})
        assert [p["content"] for p in second["particles"]] == ["claim 2", "claim 1"]

        third = await _call_tool(server, "particles_list", {"limit": 2, "offset": 4})
        assert [p["content"] for p in third["particles"]] == ["claim 0"]

        # offset past the end returns empty, not an error.
        empty = await _call_tool(server, "particles_list", {"limit": 2, "offset": 100})
        assert empty["particles"] == []

    @pytest.mark.asyncio
    async def test_invalid_status_raises(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server

        server = build_server()
        with pytest.raises(Exception):  # noqa: B017 — error type varies by SDK version
            await _call_tool(server, "particles_list", {"status": "NOT_A_STATUS"})

    @pytest.mark.asyncio
    async def test_invalid_limit_raises(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server

        server = build_server()
        with pytest.raises(Exception):  # noqa: B017
            await _call_tool(server, "particles_list", {"limit": 0})


class TestTaxonomyTools:
    @pytest.mark.asyncio
    async def test_list_taxonomies(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.taxonomy_store import insert_taxonomy

        td = TaxonomyDefinition(
            name="Coins",
            version="1.0.0",
            author="t",
            tags=[TagNode(tag="coins")],
        )
        await insert_taxonomy(db_session, td)
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "list_taxonomies", {})
        names = {t["name"] for t in result["result"]}
        assert "Coins" in names

    @pytest.mark.asyncio
    async def test_list_taxonomies_paginates_with_limit_and_offset(
        self, db_session: AsyncSession
    ) -> None:
        from particles.mcp import build_server
        from particles.store.taxonomy_store import insert_taxonomy

        # Insert in non-alphabetical order; results are by name asc.
        for name in ["Charlie Taxonomy", "Alpha Taxonomy", "Bravo Taxonomy"]:
            await insert_taxonomy(
                db_session,
                TaxonomyDefinition(
                    name=name,
                    version="1.0.0",
                    author="t",
                    tags=[TagNode(tag="root")],
                ),
            )
        await db_session.commit()

        server = build_server()
        first = await _call_tool(server, "list_taxonomies", {"limit": 2})
        first_names = [t["name"] for t in first["result"]]
        assert first_names == ["Alpha Taxonomy", "Bravo Taxonomy"]

        second = await _call_tool(server, "list_taxonomies", {"limit": 2, "offset": 2})
        second_names = [t["name"] for t in second["result"]]
        assert second_names == ["Charlie Taxonomy"]


class TestCorpusListing:
    @pytest.mark.asyncio
    async def test_list_corpus_entries_orders_most_recent_first(
        self, db_session: AsyncSession
    ) -> None:
        from particles.corpus.store import CorpusEntryRow
        from particles.mcp import build_server

        old = CorpusEntry(
            uri_r="https://a/",
            source_type="WEB_PAGE",
            deposited_by="t",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        new = CorpusEntry(
            uri_r="https://b/",
            source_type="WEB_PAGE",
            deposited_by="t",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        db_session.add(CorpusEntryRow.from_model(old))
        db_session.add(CorpusEntryRow.from_model(new))
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "list_corpus_entries", {})
        entries = result["result"]
        assert entries[0]["uri_r"] == "https://b/"
        assert entries[1]["uri_r"] == "https://a/"

    @pytest.mark.asyncio
    async def test_list_corpus_entries_source_type_filter(self, db_session: AsyncSession) -> None:
        from particles.corpus.store import CorpusEntryRow
        from particles.mcp import build_server

        db_session.add(
            CorpusEntryRow.from_model(
                CorpusEntry(uri_r="https://wd/", source_type="WIKIDATA_API", deposited_by="t")
            )
        )
        db_session.add(
            CorpusEntryRow.from_model(
                CorpusEntry(uri_r="https://wp/", source_type="WEB_PAGE", deposited_by="t")
            )
        )
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "list_corpus_entries", {"source_type": "WIKIDATA_API"})
        entries = result["result"]
        assert len(entries) == 1
        assert entries[0]["source_type"] == "WIKIDATA_API"


class TestDiagnostics:
    @pytest.mark.asyncio
    async def test_lint_on_empty_store_returns_report_shape(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server

        server = build_server()
        result = await _call_tool(server, "lint", {})
        assert "findings" in result
        assert isinstance(result["findings"], list)

    @pytest.mark.asyncio
    async def test_lint_summary_only_drops_findings_list(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        # Phantom subject (zero ACTIVE CLAIM particles) → at least one
        # PHANTOM_SUBJECT finding so the summary dict is non-empty.
        subj = Subject(canonical_name="Phantom", asserted_by="t")
        await insert_subject(db_session, subj)
        # And one ACTIVE particle without a subject → not phantom-flagged
        # but populates the active set so other checks have something to
        # work with.
        await insert_particle(db_session, _make_particle())
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "lint", {"summary_only": True})
        # summary_only drops the findings list but keeps the counts dict.
        assert "findings" not in result
        assert "summary" in result and isinstance(result["summary"], dict)
        assert result["summary"].get("PHANTOM_SUBJECT", 0) >= 1

    @pytest.mark.asyncio
    async def test_lint_category_filter_returns_only_matching_kind(
        self, db_session: AsyncSession
    ) -> None:
        from particles.mcp import build_server
        from particles.store.subject_store import insert_subject

        # Two phantom subjects → two PHANTOM_SUBJECT findings.
        for name in ("PhantomA", "PhantomB"):
            await insert_subject(db_session, Subject(canonical_name=name, asserted_by="t"))
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "lint", {"category": "PHANTOM_SUBJECT"})
        assert result["filtered_category"] == "PHANTOM_SUBJECT"
        # Every finding in the filtered response matches the requested kind.
        assert all(f["finding_type"] == "PHANTOM_SUBJECT" for f in result["findings"])
        assert len(result["findings"]) >= 2

        # An unrelated category should yield zero findings (the run still
        # happened, but nothing of that kind was emitted).
        none_result = await _call_tool(server, "lint", {"category": "DOES_NOT_EXIST_FINDING_TYPE"})
        assert none_result["findings"] == []

    @pytest.mark.asyncio
    async def test_lint_truncates_findings_at_limit(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.subject_store import insert_subject

        # Five phantom subjects, ask for at most 2.
        for i in range(5):
            await insert_subject(db_session, Subject(canonical_name=f"Phantom{i}", asserted_by="t"))
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "lint", {"category": "PHANTOM_SUBJECT", "limit": 2})
        assert result["truncated"] is True
        assert result["total_findings_before_truncation"] == 5
        assert len(result["findings"]) == 2

    @pytest.mark.asyncio
    async def test_quality_report_returns_expected_keys(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server

        server = build_server()
        result = await _call_tool(server, "quality_report", {})
        # QualityReport schema includes at least these top-level fields
        for key in ("active_particles", "calibration", "generated_at"):
            assert key in result


class TestQuerySummary:
    @pytest.mark.asyncio
    async def test_query_summary_returns_slim_hits(self, db_session: AsyncSession) -> None:
        from unittest.mock import AsyncMock, patch

        from particles.core.schema import QueryResponse
        from particles.mcp import build_server

        # Build a QueryResponse with one full Particle hit; the slim
        # transformation should strip provenance/tags/properties and keep
        # just the small fields documented in the tool docstring.
        p = _make_particle(content="atomic claim", subject_ids=["s-1"])
        fake_resp = QueryResponse(
            answer="The answer.",
            particles=[p],
            effective_confidences=[0.42],
        )

        with patch("particles.operations.query.query", new=AsyncMock(return_value=fake_resp)):
            server = build_server()
            result = await _call_tool(
                server,
                "query",
                {"question": "anything?", "summary": True},
            )

        # NL answer is preserved verbatim.
        assert result["answer"] == "The answer."
        # particles is a list of slim dicts, NOT full Particle Pydantic JSON.
        assert len(result["particles"]) == 1
        hit = result["particles"][0]
        assert set(hit.keys()) == {
            "id",
            "content",
            "confidence",
            "effective_confidence",
            "subject_ids",
            "contested",
        }
        assert hit["id"] == p.id
        assert hit["content"] == "atomic claim"
        assert hit["confidence"] == p.confidence.value
        assert hit["effective_confidence"] == 0.42
        assert hit["subject_ids"] == ["s-1"]
        # not contested (no INCONSISTENCY references it).
        assert hit["contested"] is None
        # Provenance/tags/properties — present on the full Particle dump
        # — must be absent in the slim representation.
        assert "provenance" not in hit
        assert "tags" not in hit
        assert "properties" not in hit

    @pytest.mark.asyncio
    async def test_query_default_returns_full_particles(self, db_session: AsyncSession) -> None:
        """Backward-compat: omitting ``summary`` returns full Particle dumps."""
        from unittest.mock import AsyncMock, patch

        from particles.core.schema import QueryResponse
        from particles.mcp import build_server

        p = _make_particle(content="atomic claim", subject_ids=["s-1"])
        fake_resp = QueryResponse(
            answer="The answer.",
            particles=[p],
            effective_confidences=[0.42],
        )

        with patch("particles.operations.query.query", new=AsyncMock(return_value=fake_resp)):
            server = build_server()
            result = await _call_tool(
                server,
                "query",
                {"question": "anything?"},  # summary defaults to False
            )

        # Full Particle Pydantic dump includes these fields the slim
        # variant strips out.
        hit = result["particles"][0]
        assert "provenance" in hit
        assert "uncertainty_nature" in hit
        assert "schema_version" in hit


class TestGraphView:
    """The scoped-subgraph read tool."""

    @pytest.mark.asyncio
    async def test_graph_view_requires_scope(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server

        server = build_server()
        with pytest.raises(Exception):  # noqa: B017
            await _call_tool(server, "graph_view", {})
        with pytest.raises(Exception):  # noqa: B017
            await _call_tool(server, "graph_view", {"subject_id": "s", "query": "q"})

    @pytest.mark.asyncio
    async def test_graph_view_subject_scope_renders(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        subj = Subject(id=str(uuid.uuid4()), canonical_name="Pluto", asserted_by="test")
        await insert_subject(db_session, subj)
        p = _make_particle(content="Pluto is a dwarf planet.", subject_ids=[subj.id])
        await insert_particle(db_session, p)
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "graph_view", {"subject_id": subj.id})
        assert result["scope_type"] == "subject"
        assert any(n["subject_id"] == subj.id for n in result["nodes"])
        assert p.id in result["particles"]
        # Epistemics are engine-computed and ride the payload.
        assert 0.0 < result["particles"][p.id]["effective_confidence"] <= 1.0
        # No engine configured → no deep link (the stdio install has no HTTP
        # server; the url field is additive, never assumed).
        assert "url" not in result

    @pytest.mark.asyncio
    async def test_graph_view_url_when_engine_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No db_session: the backend is mocked, so no store is touched — and
        # the reset_config() below must not race the fixture's engine teardown.
        from unittest.mock import AsyncMock, MagicMock, patch

        from particles.core.schema import GraphCensus, GraphData
        from particles.mcp import build_server

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://engine.test")
        from particles.config import reset_config

        reset_config()

        fake = GraphData(
            scope_type="subject",
            scope_ref="sid-1",
            nodes=[],
            edges=[],
            census=GraphCensus(
                scope="subject sid-1 neighbourhood (1 hop)",
                candidate_subjects=0,
                rendered_subjects=0,
                candidate_particles=0,
                rendered_particles=0,
            ),
        )
        backend = MagicMock()
        backend.graph = AsyncMock(return_value=fake)
        with patch("particles.api.client.get_backend", return_value=backend):
            server = build_server()
            result = await _call_tool(
                server, "graph_view", {"subject_id": "sid-1", "history": True}
            )
        # The deep link targets the unified web UI's #/browse route (
        # §2) with the same scope, engine-side.
        assert result["url"].startswith("http://engine.test/app#/browse?")
        assert "scope=subject" in result["url"]
        assert "subject_id=sid-1" in result["url"]
        assert "history=true" in result["url"]


# ---------------------------------------------------------------------------
# Golden tool-schema file
# ---------------------------------------------------------------------------


@upstream_only  # golden pinned to docstring prose, which a published copy carries edited
class TestToolSchemaGolden:
    """The committed ``tests/mcp/tool-schema.json`` pins the surface.

    Regenerate via ``uv run particles mcp tools > tests/mcp/tool-schema.json``
    when an ``operations/`` signature changes intentionally; the diff
    surfaces breaking-change candidates in PR review.
    """

    def test_golden_file_matches_live_surface(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["mcp", "tools"], catch_exceptions=False)
        assert result.exit_code == 0
        live = json.loads(result.output)
        golden = json.loads(_GOLDEN_TOOL_SCHEMA.read_text())
        assert live == golden, (
            "MCP tool surface drifted from the golden file. "
            "If this is intentional, run:\n"
            "  uv run particles mcp tools > tests/mcp/tool-schema.json\n"
            "and commit the update."
        )


def _confident(content: str, value: float) -> Particle:
    """An ACTIVE agent-asserted particle at a given confidence (no source → eff == value)."""
    return _make_particle(content=content).model_copy(
        update={
            "confidence": Confidence(
                value=value, calibration_source=CalibrationSource.AGENT_ASSERTED
            )
        }
    )


class TestDigestResource:
    """session-start memory digest — ``particles://digest/<store>``."""

    @pytest.mark.asyncio
    async def test_template_registered_no_concrete_when_unlisted(self) -> None:
        from particles.mcp import build_server

        server = build_server()
        templates = await server.list_resource_templates()
        # The template is always available (a read-only view of any store).
        assert [t.uriTemplate for t in templates] == ["particles://digest/{store}"]
        # Default config lists no stores → no concrete digest resource enumerated.
        assert [str(r.uri) for r in await server.list_resources()] == []

    @pytest.mark.asyncio
    async def test_write_enabled_store_is_listed(self) -> None:
        from particles.config import get_config
        from particles.db import DEFAULT_STORE
        from particles.mcp import build_server

        get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
        server = build_server()
        uris = {str(r.uri) for r in await server.list_resources()}
        assert f"particles://digest/{DEFAULT_STORE}" in uris

    @pytest.mark.asyncio
    async def test_digest_stores_lists_without_write_enable(self) -> None:
        from particles.config import get_config
        from particles.db import DEFAULT_STORE
        from particles.mcp import build_server

        # A read-only store can be opted into the listing via mcp.recall.digest_stores.
        get_config().mcp.recall.digest_stores = [DEFAULT_STORE]
        server = build_server()
        uris = {str(r.uri) for r in await server.list_resources()}
        assert f"particles://digest/{DEFAULT_STORE}" in uris

    @pytest.mark.asyncio
    async def test_digest_orders_by_effective_confidence(self, db_session: AsyncSession) -> None:
        from particles.db import DEFAULT_STORE
        from particles.mcp.resources import build_digest
        from particles.store.particle_store import insert_particle

        for content, conf in [("high belief", 0.9), ("low belief", 0.4), ("mid belief", 0.7)]:
            await insert_particle(db_session, _confident(content, conf))
        await db_session.commit()

        md = await build_digest(DEFAULT_STORE)
        # No trust statements → effective confidence == value, so order tracks it.
        assert md.index("high belief") < md.index("mid belief") < md.index("low belief")
        assert "**0.90**" in md and "**0.40**" in md

    @pytest.mark.asyncio
    async def test_digest_marks_contested(self, db_session: AsyncSession) -> None:
        from particles.db import DEFAULT_STORE
        from particles.mcp.resources import build_digest
        from particles.store.particle_store import insert_particle

        a = _confident("Deploy key rotates monthly.", 0.8)
        await insert_particle(db_session, a)
        inc = Particle(
            id=str(uuid.uuid4()),
            content=f"INCONSISTENCY: conflict.\nParticle A: {a.id}",
            confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE, corpus_entry_id=a.id, snapshot_id=a.id
                )
            ],
            asserted_by="extract-pipeline",
            status=Status.INCONSISTENCY,
        )
        await insert_particle(db_session, inc)
        await db_session.commit()

        md = await build_digest(DEFAULT_STORE)
        # MUST: the contested belief carries its INCONSISTENCY id;
        # the composed badge (default-on) names the fired basis.
        assert f"contested (inconsistency) by `{inc.id}`" in md
        # The INCONSISTENCY meta-particle itself is not ACTIVE → not a digest line.
        assert "INCONSISTENCY: conflict" not in md

    @pytest.mark.asyncio
    async def test_digest_truncates_with_disclosed_footer(self, db_session: AsyncSession) -> None:
        from particles.config import get_config
        from particles.db import DEFAULT_STORE
        from particles.mcp.resources import build_digest
        from particles.store.particle_store import insert_particle

        get_config().mcp.recall.digest_max_beliefs = 1
        await insert_particle(db_session, _confident("top belief", 0.9))
        await insert_particle(db_session, _confident("dropped belief", 0.3))
        await db_session.commit()

        md = await build_digest(DEFAULT_STORE)
        assert "top belief" in md
        assert "dropped belief" not in md  # capped out
        assert "1 not shown" in md  # no silent truncation

    @pytest.mark.asyncio
    async def test_digest_empty_store(self, db_session: AsyncSession) -> None:
        from particles.db import DEFAULT_STORE
        from particles.mcp.resources import build_digest

        md = await build_digest(DEFAULT_STORE)
        assert "No ACTIVE beliefs" in md


class TestResourceSchemaGolden:
    """The committed ``tests/mcp/resource-schema.json`` pins the resource surface.

    Regenerate via ``uv run particles mcp resources > tests/mcp/resource-schema.json``
    when the digest resource contract changes intentionally.
    """

    def test_golden_file_matches_live_surface(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["mcp", "resources"], catch_exceptions=False)
        assert result.exit_code == 0
        live = json.loads(result.output)
        golden = json.loads(_GOLDEN_RESOURCE_SCHEMA.read_text())
        assert live == golden, (
            "MCP resource surface drifted from the golden file. "
            "If this is intentional, run:\n"
            "  uv run particles mcp resources > tests/mcp/resource-schema.json\n"
            "and commit the update."
        )


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


class TestMcpCli:
    def test_mcp_tools_json_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["mcp", "tools"], catch_exceptions=False)
        assert result.exit_code == 0
        # Output is JSON-parseable
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 16

    def test_mcp_tools_text_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["mcp", "tools", "--format", "text"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "query" in result.output
        assert "lint" in result.output

    def test_mcp_resources_json_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["mcp", "resources"], catch_exceptions=False)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert set(parsed) == {"resources", "templates"}
        assert any(t["uriTemplate"] == "particles://digest/{store}" for t in parsed["templates"])

    def test_mcp_resources_text_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            app, ["mcp", "resources", "--format", "text"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "particles://digest/{store}" in result.output

    def test_mcp_help_lists_subcommands(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["mcp", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "serve" in result.output
        assert "tools" in result.output
        assert "resources" in result.output


# Avoid the linter complaining about asyncio.run import — keep it accessible.
_ = asyncio


class TestEventsTools:
    @pytest.mark.asyncio
    async def test_events_list_returns_recorded(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.event_store import (
            EventRefKind,
            OperatorEventType,
            record_event,
        )

        await record_event(
            db_session,
            actor="subjects-merge",
            event_type=OperatorEventType.SUBJECTS_MERGED,
            refs=[(EventRefKind.SUBJECT, "s-1")],
        )
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "events_list", {})
        assert result["result"][0]["event_type"] == "SUBJECTS_MERGED"

    @pytest.mark.asyncio
    async def test_events_list_filter_by_subject(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.event_store import (
            EventRefKind,
            OperatorEventType,
            record_event,
        )

        await record_event(
            db_session,
            actor="t",
            event_type=OperatorEventType.SUBJECTS_MERGED,
            refs=[(EventRefKind.SUBJECT, "keep")],
        )
        await record_event(
            db_session,
            actor="t",
            event_type=OperatorEventType.SUBJECT_ALIASED,
            refs=[(EventRefKind.SUBJECT, "other")],
        )
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "events_list", {"subject": "keep"})
        assert len(result["result"]) == 1
        assert result["result"][0]["event_type"] == "SUBJECTS_MERGED"

    @pytest.mark.asyncio
    async def test_event_show_returns_full(self, db_session: AsyncSession) -> None:
        from particles.mcp import build_server
        from particles.store.event_store import OperatorEventType, record_event

        event = await record_event(
            db_session,
            actor="trust-set",
            event_type=OperatorEventType.TRUST_CHANGED,
            reason="manual",
            payload={"old_rank": 0.3, "new_rank": 0.7},
        )
        await db_session.commit()

        server = build_server()
        result = await _call_tool(server, "event_show", {"event_id": event.event_id})
        assert result["event_id"] == event.event_id
        assert result["payload"] == {"old_rank": 0.3, "new_rank": 0.7}
