"""Tests for the lint contestedness distribution (operations/lint/contestedness.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    TrustLensDefinition,
    TrustLensUrlRule,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.operations.lint.contestedness import (
    _histogram,
    _report_contestedness_distribution,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_P0 = "00000000-0000-0000-0000-0000000000e0"


def _claim(content: str, pid: str, entry_id: str, confidence: float = 0.9) -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)],
    )


async def _adopt(session: AsyncSession) -> None:
    from particles.store.lens_store import adopt_lens, materialise_lens

    lens = TrustLensDefinition(
        name="acme-numismatics",
        version=1,
        url_rules=[TrustLensUrlRule(scope="domain", pattern="sketchy.example", score=0.2)],
        extractor_weights={},
    )
    await materialise_lens(session, lens)
    await adopt_lens(session, lens.name)


async def _add_entry(session: AsyncSession, entry_id: str, uri_r: str) -> None:
    from particles.corpus.store import CorpusEntryRow

    session.add(
        CorpusEntryRow(
            entry_id=entry_id,
            uri_r=uri_r,
            source_type="WEB_PAGE",
            mutability="MUTABLE",
            fetch_policy="LAZY",
            created_at=datetime.now(UTC),
            deposited_by="test",
        )
    )
    await session.flush()


def test_histogram_buckets_partition_spreads() -> None:
    counts = _histogram([0.0, 0.04, 0.3, 1.0])
    assert sum(counts.values()) == 4
    assert counts["[0.00,0.05)"] == 2  # 0.0 and 0.04
    assert counts["[0.20,0.40)"] == 1  # 0.3
    assert counts["[0.80,1.00]"] == 1  # 1.0 lands in the closed top bucket


@pytest.mark.asyncio
async def test_no_finding_below_two_policies(db_session: AsyncSession) -> None:
    """§3 degeneracy: a one-policy store emits no contestedness finding."""
    await _add_entry(db_session, "e1", "https://sketchy.example/p")
    from particles.store.particle_store import insert_particle

    await insert_particle(db_session, _claim("A claim.", _P0, "e1"))
    await db_session.commit()

    assert await _report_contestedness_distribution(db_session) == []


@pytest.mark.asyncio
async def test_finding_reports_distribution_with_lens(db_session: AsyncSession) -> None:
    from particles.store.particle_store import insert_particle

    await _adopt(db_session)
    await _add_entry(db_session, "e1", "https://sketchy.example/p")
    await insert_particle(db_session, _claim("Sketchy claim.", _P0, "e1"))
    await db_session.commit()

    findings = await _report_contestedness_distribution(db_session)
    assert len(findings) == 1
    f = findings[0]
    assert f.finding_type == "CONTESTEDNESS_DISTRIBUTION"
    assert f.severity == "INFO"
    assert "acme-numismatics" in f.detail
    assert "Spread histogram" in f.detail
    # The sketchy claim diverges past the default 0.2 threshold → named as contested.
    assert "Most contested" in f.detail


@pytest.mark.asyncio
async def test_finding_absent_when_no_claims(db_session: AsyncSession) -> None:
    await _adopt(db_session)
    await db_session.commit()
    assert await _report_contestedness_distribution(db_session) == []
