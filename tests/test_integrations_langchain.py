"""Tests for the LangChain adapter.

The unit gate runs **without** the optional ``langchain`` extra installed (like
``otel``), so the dependency-bearing assertions ("the real ``StructuredTool`` /
``BaseRetriever`` / ``Document`` are constructed") are deferred to the
integration-tier test (skipped when ``langchain_core`` is unimportable). The
adapter's *logic* — the ``QueryResponse`` → answer-string / Document mapping and
the backend-routing — is factored into pure helpers in
``particles.integrations._mapping`` and the ``_run_query`` seam, so it is fully
unit-tested here with the ``Backend`` mocked and no ``langchain_core`` present.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import AsyncMock

import pytest

from particles.core.schema import (
    CalibrationSource,
    Confidence,
    CoverageGapKind,
    Particle,
    QueryRequest,
    QueryResponse,
    Status,
    SubjectCoverageGap,
    UncertaintyNature,
)
from particles.integrations._mapping import (
    deposit_confirmation_text,
    query_answer_text,
    query_documents,
)

_LANGCHAIN_PRESENT = importlib.util.find_spec("langchain_core") is not None


def _particle(content: str, *, confidence: float, subject_ids: list[str]) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.AGENT_ASSERTED
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=Status.ACTIVE,
        subject_ids=subject_ids,
    )


def _response(
    *,
    answer: str = "The synthesised answer. [p-1]",
    truncation_warning: str | None = None,
    subject_gaps: list[SubjectCoverageGap] | None = None,
) -> QueryResponse:
    particles = [
        _particle("Claim one.", confidence=0.8, subject_ids=["sid-a"]),
        _particle("Claim two.", confidence=0.6, subject_ids=["sid-a", "sid-b"]),
    ]
    return QueryResponse(
        answer=answer,
        particles=particles,
        effective_confidences=[0.72, 0.55],
        truncation_warning=truncation_warning,
        subject_coverage_gaps=subject_gaps or [],
    )


# ---------------------------------------------------------------------------
# Pure mapping helpers (no langchain_core) — the QueryResponse → shape mapping.
# ---------------------------------------------------------------------------


class TestQueryAnswerText:
    def test_answer_flows_through_unchanged_when_no_notes(self) -> None:
        resp = _response(answer="Just the answer. [p-1]")
        assert query_answer_text(resp) == "Just the answer. [p-1]"

    def test_truncation_warning_appended_as_note(self) -> None:
        resp = _response(answer="Answer.", truncation_warning="Results truncated at top_k.")
        out = query_answer_text(resp)
        assert out.startswith("Answer.")
        assert "Note:" in out
        assert "Results truncated at top_k." in out

    def test_coverage_gap_detail_appended_as_note(self) -> None:
        gap = SubjectCoverageGap(
            subject_id=None,
            subject_name="Atlantis",
            kind=CoverageGapKind.NO_SUBJECT_MATCH,
            detail="No subject matched 'Atlantis'.",
        )
        resp = _response(answer="Answer.", subject_gaps=[gap])
        out = query_answer_text(resp)
        assert "Note:" in out
        assert "No subject matched 'Atlantis'." in out


class TestQueryDocuments:
    def test_one_pair_per_ranked_particle(self) -> None:
        docs = query_documents(_response())
        assert len(docs) == 2

    def test_page_content_is_claim_text(self) -> None:
        docs = query_documents(_response())
        assert docs[0][0] == "Claim one."
        assert docs[1][0] == "Claim two."

    def test_metadata_carries_provenance_and_ranking(self) -> None:
        resp = _response()
        docs = query_documents(resp)
        meta0 = docs[0][1]
        assert meta0["particle_id"] == resp.particles[0].id
        assert meta0["effective_confidence"] == 0.72
        assert meta0["confidence"] == 0.8  # the stored confidence.value
        assert meta0["subject_ids"] == ["sid-a"]
        assert meta0["status"] == str(Status.ACTIVE)
        # No invented relevance-score field.
        assert "score" not in meta0
        assert "relevance" not in meta0

    def test_parallel_effective_confidence_aligns_with_particles(self) -> None:
        docs = query_documents(_response())
        assert docs[1][1]["effective_confidence"] == 0.55
        assert docs[1][1]["subject_ids"] == ["sid-a", "sid-b"]


class TestDepositConfirmation:
    def test_confirmation_names_entry_and_snapshot(self) -> None:
        out = deposit_confirmation_text("entry-123", "snap-456")
        assert "entry-123" in out
        assert "snap-456" in out


# ---------------------------------------------------------------------------
# Backend routing — the tool callables build the QueryRequest and route through
# the backend seam. Mockable without langchain_core: the StructuredTool wrapping
# is the only langchain-dependent step, exercised in the integration test below.
# ---------------------------------------------------------------------------


class TestBackendRouting:
    def test_run_query_builds_request_and_routes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        from particles.integrations import langchain as adapter

        captured: dict[str, QueryRequest] = {}
        backend = AsyncMock()

        async def _query(req: QueryRequest) -> QueryResponse:
            captured["req"] = req
            return _response()

        backend.query = _query
        monkeypatch.setattr(adapter, "get_backend", lambda: backend)

        resp = asyncio.run(
            adapter._run_query(
                "what is X?",
                tags=["t1"],
                subject_id="sid-a",
                min_confidence=0.3,
                top_k=12,
            )
        )
        req = captured["req"]
        assert req.question == "what is X?"
        assert req.tags == ["t1"]
        assert req.subject_id == "sid-a"
        assert req.min_confidence == 0.3
        assert req.top_k == 12
        assert query_answer_text(resp) == _response().answer


# ---------------------------------------------------------------------------
# Missing-extra path — with langchain_core absent (the CI condition), the guard
# and every public factory raise the actionable ImportError.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _LANGCHAIN_PRESENT, reason="langchain_core is installed; missing-extra path not exercisable"
)
class TestMissingExtra:
    def test_require_langchain_raises_actionable_importerror(self) -> None:
        from particles.integrations.langchain import _require_langchain

        with pytest.raises(ImportError, match=r"pip install particles\[langchain\]"):
            _require_langchain()

    def test_get_langchain_tools_factory_raises(self) -> None:
        from particles.integrations import get_langchain_tools

        with pytest.raises(ImportError, match=r"pip install particles\[langchain\]"):
            get_langchain_tools()

    def test_query_tool_factory_raises(self) -> None:
        from particles.integrations.langchain import ParticlesQueryTool

        with pytest.raises(ImportError, match=r"pip install particles\[langchain\]"):
            ParticlesQueryTool()

    def test_deposit_tool_factory_raises(self) -> None:
        from particles.integrations.langchain import ParticlesDepositTool

        with pytest.raises(ImportError, match=r"pip install particles\[langchain\]"):
            ParticlesDepositTool()

    def test_retriever_access_raises_without_extra(self) -> None:
        # ParticlesRetriever subclasses the *real* BaseRetriever, which cannot be
        # imported without the extra — so *accessing* the symbol (which builds the
        # class via the cached factory) raises the actionable ImportError.
        with pytest.raises(ImportError, match=r"pip install particles\[langchain\]"):
            from particles.integrations.langchain import (  # noqa: F401
                ParticlesRetriever,
            )

    def test_retriever_package_access_raises_without_extra(self) -> None:
        # PEP 562 __getattr__ on the package delegates to the submodule factory,
        # so package-level access raises the same actionable ImportError.
        import particles.integrations as pkg

        with pytest.raises(ImportError, match=r"pip install particles\[langchain\]"):
            _ = pkg.ParticlesRetriever


# ---------------------------------------------------------------------------
# Integration tier — the real langchain_core primitives. Marked `integration`
# (deselected by `-m "not integration"`) AND skipped when the extra is absent,
# so it runs only when `uv sync --extra langchain` has installed langchain_core.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _LANGCHAIN_PRESENT, reason="requires the optional `langchain` extra")
class TestRealLangChainPrimitives:
    def test_query_tool_is_structuredtool_returning_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        from langchain_core.tools import StructuredTool

        from particles.integrations import langchain as adapter
        from particles.integrations.langchain import ParticlesQueryTool

        backend = AsyncMock()
        backend.query = AsyncMock(return_value=_response(answer="Cited answer. [p-1]"))
        monkeypatch.setattr(adapter, "get_backend", lambda: backend)

        tool = ParticlesQueryTool()
        assert isinstance(tool, StructuredTool)
        assert tool.name == "particles_query"
        result = asyncio.run(tool.ainvoke({"question": "what is X?"}))
        assert result == "Cited answer. [p-1]"

    def test_deposit_tool_is_structuredtool_returning_confirmation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        from langchain_core.tools import StructuredTool

        from particles.integrations import langchain as adapter
        from particles.integrations.langchain import ParticlesDepositTool

        backend = AsyncMock()
        backend.deposit_text = AsyncMock(return_value=("entry-9", "snap-9"))
        monkeypatch.setattr(adapter, "get_backend", lambda: backend)

        tool = ParticlesDepositTool()
        assert isinstance(tool, StructuredTool)
        assert tool.name == "particles_deposit"
        result = asyncio.run(tool.ainvoke({"text": "archive me"}))
        assert "entry-9" in result and "snap-9" in result

    def test_get_langchain_tools_returns_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.integrations import get_langchain_tools

        tools = get_langchain_tools()
        names = {t.name for t in tools}
        assert names == {"particles_query", "particles_deposit"}

    def test_retriever_maps_each_particle_to_one_document(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever

        from particles.integrations import langchain as adapter
        from particles.integrations.langchain import ParticlesRetriever

        resp = _response()
        backend = AsyncMock()
        backend.query = AsyncMock(return_value=resp)
        monkeypatch.setattr(adapter, "get_backend", lambda: backend)

        retriever = ParticlesRetriever()
        assert isinstance(retriever, BaseRetriever)
        docs = asyncio.run(retriever.ainvoke("what is X?"))
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].page_content == "Claim one."
        assert docs[0].metadata["particle_id"] == resp.particles[0].id
        assert docs[0].metadata["effective_confidence"] == 0.72
