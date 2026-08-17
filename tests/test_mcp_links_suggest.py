"""Tests for the ``links_suggest`` MCP tool (particles/mcp/tools/links_suggest.py).

The candidate-finding operation is covered by ``tests/test_links_suggest.py``;
this file pins the tool wrapper — the ``limit`` validation, the read-only mode
the MCP surface is allowed to request, and the flat cap that keeps
one enormous Subject from blowing the per-tool-result size budget.

The tool defers ``from particles.api.client import get_backend``, so the patch
below reaches it (tests/AGENTS.md § Mocking strategy).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.core.schema import (
    CandidateCluster,
    CoEvidentialCandidate,
    SuggestMode,
    SuggestReport,
)
from particles.mcp.tools.links_suggest import links_suggest


def _cluster(subject_id: str, n: int) -> CandidateCluster:
    return CandidateCluster(
        subject_id=subject_id,
        subject_name=f"Subject {subject_id}",
        candidates=[
            CoEvidentialCandidate(
                particle_a=f"{subject_id}-a{i}",
                particle_b=f"{subject_id}-b{i}",
                similarity=0.95,
            )
            for i in range(n)
        ],
    )


def _report(*sizes: int, warnings: list[str] | None = None) -> SuggestReport:
    clusters = [_cluster(f"s-{i}", n) for i, n in enumerate(sizes)]
    return SuggestReport(
        mode=SuggestMode.REPORT,
        clusters=clusters,
        total_candidates=sum(sizes),
        warnings=warnings or [],
    )


def _patched_backend(report: SuggestReport) -> Any:
    backend = MagicMock()
    backend.links_suggest = AsyncMock(return_value=report)
    return patch("particles.api.client.get_backend", return_value=backend)


# ---------------------------------------------------------------------------
# limit validation
# ---------------------------------------------------------------------------


class TestLimitValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("limit", [0, -1])
    async def test_non_positive_limit_is_rejected(self, limit: int) -> None:
        with (
            _patched_backend(_report(1)),
            pytest.raises(ValueError, match="limit must be a positive integer"),
        ):
            await links_suggest(limit=limit)

    @pytest.mark.asyncio
    async def test_validation_happens_before_the_backend_call(self) -> None:
        backend = MagicMock()
        backend.links_suggest = AsyncMock()
        with (
            patch("particles.api.client.get_backend", return_value=backend),
            pytest.raises(ValueError),
        ):
            await links_suggest(limit=0)
        backend.links_suggest.assert_not_awaited()


# ---------------------------------------------------------------------------
# The read-only contract
# ---------------------------------------------------------------------------


class TestReadOnlyMode:
    @pytest.mark.asyncio
    async def test_always_requests_report_mode_unconfirmed(self) -> None:
        backend = MagicMock()
        backend.links_suggest = AsyncMock(return_value=_report(1))
        with patch("particles.api.client.get_backend", return_value=backend):
            await links_suggest(subject_id="s-0", threshold=0.88)
        kwargs = backend.links_suggest.await_args.kwargs
        assert kwargs["mode"] is SuggestMode.REPORT
        assert kwargs["confirmed"] is False
        assert kwargs["subject_id"] == "s-0"
        assert kwargs["threshold"] == 0.88

    @pytest.mark.asyncio
    async def test_omitted_filters_pass_through_as_none(self) -> None:
        backend = MagicMock()
        backend.links_suggest = AsyncMock(return_value=_report(1))
        with patch("particles.api.client.get_backend", return_value=backend):
            await links_suggest()
        kwargs = backend.links_suggest.await_args.kwargs
        assert kwargs["subject_id"] is None
        assert kwargs["threshold"] is None


# ---------------------------------------------------------------------------
# The flat candidate cap
# ---------------------------------------------------------------------------


class TestTruncation:
    @pytest.mark.asyncio
    async def test_under_the_cap_returns_the_report_untouched(self) -> None:
        with _patched_backend(_report(2, 1, warnings=["no embeddings for 3 particles"])):
            out = await links_suggest(limit=100)
        assert "truncated" not in out
        assert out["total_candidates"] == 3
        assert out["mode"] == SuggestMode.REPORT.value
        assert out["warnings"] == ["no embeddings for 3 particles"]
        assert [len(c["candidates"]) for c in out["clusters"]] == [2, 1]

    @pytest.mark.asyncio
    async def test_at_the_cap_is_not_truncated(self) -> None:
        with _patched_backend(_report(3)):
            out = await links_suggest(limit=3)
        assert "truncated" not in out

    @pytest.mark.asyncio
    async def test_cap_spans_clusters_and_is_disclosed(self) -> None:
        with _patched_backend(_report(2, 2, 2)):
            out = await links_suggest(limit=3)
        assert out["truncated"] is True
        assert out["total_candidates_before_truncation"] == 6
        # The cap is flat across clusters: 2 from the first, 1 from the second,
        # and the third is dropped entirely rather than emptied.
        assert [len(c["candidates"]) for c in out["clusters"]] == [2, 1]
        # ``total_candidates`` keeps the pre-truncation count so a client cannot
        # mistake the capped page for the whole finding.
        assert out["total_candidates"] == 6

    @pytest.mark.asyncio
    async def test_truncation_does_not_mutate_the_kept_cluster_metadata(self) -> None:
        with _patched_backend(_report(5, 5)):
            out = await links_suggest(limit=4)
        assert out["clusters"][0]["subject_id"] == "s-0"
        assert out["clusters"][0]["subject_name"] == "Subject s-0"
        assert len(out["clusters"][0]["candidates"]) == 4
        assert len(out["clusters"]) == 1
