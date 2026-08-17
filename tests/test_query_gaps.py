"""Tests for the §9.3 coverage-gap detectors (particles/operations/query/gaps.py).

Two flavours: corpus-level (entries with PENDING/FAILED snapshots) and
subject-level (NO_SUBJECT_MATCH / SUBJECT_HAS_NO_PARTICLES /
SUBJECT_HAS_LOW_COVERAGE), the latter only when an explicit subject_id is set.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from particles.core.schema import (
    Confidence,
    CorpusEntry,
    CoverageGapKind,
    ExtractionStatus,
    Particle,
    ParticleType,
    Snapshot,
    Subject,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.query.gaps import _find_coverage_gaps, _find_subject_coverage_gaps


async def _add_subject(session: Any, name: str = "Groschen") -> str:
    from particles.store.subject_store import insert_subject

    subj = Subject(id=str(uuid.uuid4()), canonical_name=name, asserted_by="test")
    await insert_subject(session, subj)
    await session.commit()
    return subj.id


async def _add_claim(session: Any, subject_id: str, content: str = "A claim.") -> str:
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        particle_type=ParticleType.CLAIM,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        subject_ids=[subject_id],
    )
    await insert_particle(session, p)
    await session.commit()
    return p.id


async def _add_entry_with_status(session: Any, status: ExtractionStatus) -> str:
    from particles.corpus.store import CorpusEntryRow, SnapshotRow

    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="WEB_PAGE",
        uri_r=f"https://example.com/{uuid.uuid4()}",
        deposited_by="test",
    )
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash="e" * 64,
        extraction_status=status,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    session.add(CorpusEntryRow.from_model(entry))
    session.add(SnapshotRow.from_model(snap, entry.entry_id))
    await session.commit()
    return entry.entry_id


class TestCorpusCoverageGaps:
    @pytest.mark.asyncio
    async def test_empty_store_no_gaps(self, db_session: Any) -> None:
        assert await _find_coverage_gaps(db_session) == []

    @pytest.mark.asyncio
    async def test_pending_and_failed_entries_surface(self, db_session: Any) -> None:
        pending = await _add_entry_with_status(db_session, ExtractionStatus.PENDING)
        failed = await _add_entry_with_status(db_session, ExtractionStatus.FAILED)
        await _add_entry_with_status(db_session, ExtractionStatus.COMPLETE)
        gaps = await _find_coverage_gaps(db_session)
        assert pending in gaps
        assert failed in gaps
        assert len(gaps) == 2  # the COMPLETE entry is not a gap


class TestSubjectCoverageGaps:
    @pytest.mark.asyncio
    async def test_no_subject_filter_returns_empty(self, db_session: Any) -> None:
        assert await _find_subject_coverage_gaps(db_session, None, retrieved_count=0) == []

    @pytest.mark.asyncio
    async def test_unknown_subject_id(self, db_session: Any) -> None:
        gaps = await _find_subject_coverage_gaps(db_session, "does-not-exist", retrieved_count=0)
        assert len(gaps) == 1
        assert gaps[0].kind == CoverageGapKind.NO_SUBJECT_MATCH

    @pytest.mark.asyncio
    async def test_subject_with_no_particles(self, db_session: Any) -> None:
        sid = await _add_subject(db_session)
        gaps = await _find_subject_coverage_gaps(db_session, sid, retrieved_count=0)
        assert len(gaps) == 1
        assert gaps[0].kind == CoverageGapKind.SUBJECT_HAS_NO_PARTICLES
        assert gaps[0].particle_count == 0

    @pytest.mark.asyncio
    async def test_subject_with_low_coverage(self, db_session: Any) -> None:
        sid = await _add_subject(db_session)
        await _add_claim(db_session, sid)
        # count (1) < threshold (3) → low coverage, even with retrieved_count > 0.
        gaps = await _find_subject_coverage_gaps(
            db_session, sid, retrieved_count=1, low_coverage_threshold=3
        )
        assert len(gaps) == 1
        assert gaps[0].kind == CoverageGapKind.SUBJECT_HAS_LOW_COVERAGE
        assert gaps[0].particle_count == 1

    @pytest.mark.asyncio
    async def test_zero_retrieved_is_low_coverage_even_when_well_populated(
        self, db_session: Any
    ) -> None:
        sid = await _add_subject(db_session)
        for i in range(5):
            await _add_claim(db_session, sid, content=f"Claim {i}.")
        # count (5) >= threshold but nothing retrieved → still a low-coverage gap.
        gaps = await _find_subject_coverage_gaps(
            db_session, sid, retrieved_count=0, low_coverage_threshold=3
        )
        assert len(gaps) == 1
        assert gaps[0].kind == CoverageGapKind.SUBJECT_HAS_LOW_COVERAGE

    @pytest.mark.asyncio
    async def test_well_covered_subject_no_gap(self, db_session: Any) -> None:
        sid = await _add_subject(db_session)
        for i in range(5):
            await _add_claim(db_session, sid, content=f"Claim {i}.")
        gaps = await _find_subject_coverage_gaps(
            db_session, sid, retrieved_count=5, low_coverage_threshold=3
        )
        assert gaps == []
