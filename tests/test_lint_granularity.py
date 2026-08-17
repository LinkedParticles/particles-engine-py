"""Tests for the lint granularity detectors (particles/operations/lint/granularity.py).

Distinct from tests/test_granularity.py, which covers the *predicate* in
particles/core/granularity.py. These cover the two lint detectors:

  - ``_check_granularity_length`` — structural length-outlier flag (no LLM).
  - ``_check_granularity_violations`` — LLM-assisted multi-claim detection
    (the ``_llm_call`` seam is patched so the tests stay offline).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.lint.granularity import (
    _check_granularity_length,
    _check_granularity_violations,
)


async def _add_particle(session: Any, content: str) -> str:
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
    await insert_particle(session, p)
    await session.commit()
    return p.id


class TestGranularityLength:
    @pytest.mark.asyncio
    async def test_empty_store_no_findings(self, db_session: Any) -> None:
        assert await _check_granularity_length(db_session) == []

    @pytest.mark.asyncio
    async def test_outlier_flagged(self, db_session: Any) -> None:
        # Several short particles set a small median; one very long particle
        # exceeds 3× it and must be flagged.
        for _ in range(4):
            await _add_particle(db_session, "Short.")
        outlier = await _add_particle(db_session, "x" * 500)
        findings = await _check_granularity_length(db_session)
        assert [f.particle_id for f in findings] == [outlier]
        assert findings[0].finding_type == "GRANULARITY_VIOLATION_CANDIDATE"
        assert findings[0].severity == "WARNING"

    @pytest.mark.asyncio
    async def test_uniform_lengths_no_findings(self, db_session: Any) -> None:
        for _ in range(5):
            await _add_particle(db_session, "A claim of similar length here.")
        assert await _check_granularity_length(db_session) == []


class TestGranularityViolations:
    @pytest.mark.asyncio
    async def test_short_content_skipped(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import particles.operations.lint.granularity as g

        called = AsyncMock(return_value="YES: a; b")
        monkeypatch.setattr(g, "_llm_call", called)
        await _add_particle(db_session, "Under 100 chars.")  # too short → skipped
        findings = await _check_granularity_violations(db_session)
        assert findings == []
        called.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multi_claim_flagged(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import particles.operations.lint.granularity as g

        monkeypatch.setattr(g, "_llm_call", AsyncMock(return_value="YES: claim one; claim two"))
        pid = await _add_particle(db_session, "A long particle. " * 10)  # > 100 chars
        findings = await _check_granularity_violations(db_session)
        assert [f.particle_id for f in findings] == [pid]
        assert findings[0].finding_type == "GRANULARITY_VIOLATION"
        assert "claim one; claim two" in findings[0].detail

    @pytest.mark.asyncio
    async def test_atomic_long_content_not_flagged(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import particles.operations.lint.granularity as g

        monkeypatch.setattr(g, "_llm_call", AsyncMock(return_value="NO"))
        await _add_particle(db_session, "A single long atomic claim. " * 6)  # > 100 chars
        assert await _check_granularity_violations(db_session) == []

    @pytest.mark.asyncio
    async def test_llm_failure_returns_no_finding(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import particles.operations.lint.granularity as g

        # _llm_call swallows errors and returns None on failure → no finding.
        monkeypatch.setattr(g, "_llm_call", AsyncMock(return_value=None))
        await _add_particle(db_session, "A long particle whose probe failed. " * 4)
        assert await _check_granularity_violations(db_session) == []
