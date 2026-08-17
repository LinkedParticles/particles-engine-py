"""Tests for the extraction quality dashboard (particles/operations/quality.py)."""

from __future__ import annotations

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.quality import get_quality_report


async def _insert(session: object, **kwargs: object) -> Particle:
    from particles.store.particle_store import insert_particle

    defaults: dict[str, object] = {
        "content": "Test particle.",
        "confidence": Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        "uncertainty_nature": UncertaintyNature.EPISTEMIC,
        "asserted_by": "test",
        "status": Status.ACTIVE,
    }
    defaults.update(kwargs)
    p = Particle(**defaults)  # type: ignore[arg-type]
    await insert_particle(session, p)  # type: ignore[arg-type]
    return p


class TestQualityReportEmpty:
    @pytest.mark.asyncio
    async def test_empty_db_returns_zeros(self, db_session: object) -> None:
        report = await get_quality_report(db_session)  # type: ignore[arg-type]
        assert report.active_particles == 0
        assert report.inconsistency_particles == 0
        assert report.calibration == []
        assert report.extractor_direct_fraction == 0.0
        assert report.total_entries == 0
        assert report.snapshots_pending == 0
        assert report.snapshots_failed == 0
        assert report.snapshots_complete == 0
        assert report.snapshots_in_progress == 0
        assert report.total_subjects == 0
        assert report.subjects_without_particles == 0


class TestQualityReportParticles:
    @pytest.mark.asyncio
    async def test_active_count(self, db_session: object) -> None:
        await _insert(db_session, content="A.")
        await _insert(db_session, content="B.")
        report = await get_quality_report(db_session)  # type: ignore[arg-type]
        assert report.active_particles == 2

    @pytest.mark.asyncio
    async def test_inconsistency_count(self, db_session: object) -> None:
        await _insert(db_session, content="A.")
        await _insert(db_session, content="B.", status=Status.INCONSISTENCY)
        report = await get_quality_report(db_session)  # type: ignore[arg-type]
        assert report.active_particles == 1
        assert report.inconsistency_particles == 1

    @pytest.mark.asyncio
    async def test_calibration_distribution(self, db_session: object) -> None:
        await _insert(
            db_session,
            content="A.",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        )
        await _insert(
            db_session,
            content="B.",
            confidence=Confidence(
                value=0.8, calibration_source=CalibrationSource.CALIBRATED_BENCHMARK
            ),
        )
        report = await get_quality_report(db_session)  # type: ignore[arg-type]
        sources = {b.source: b for b in report.calibration}
        assert "EXTRACTOR_DIRECT" in sources
        assert "CALIBRATED_BENCHMARK" in sources
        assert sources["EXTRACTOR_DIRECT"].count == 1
        assert sources["CALIBRATED_BENCHMARK"].count == 1

    @pytest.mark.asyncio
    async def test_calibration_fractions_sum_to_one(self, db_session: object) -> None:
        for i in range(3):
            await _insert(db_session, content=f"P{i}.")
        report = await get_quality_report(db_session)  # type: ignore[arg-type]
        total = sum(b.fraction for b in report.calibration)
        assert abs(total - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_extractor_direct_fraction(self, db_session: object) -> None:
        # 2 EXTRACTOR_DIRECT, 1 CALIBRATED_BENCHMARK → fraction = 0.667
        for i in range(2):
            await _insert(db_session, content=f"D{i}.")
        await _insert(
            db_session,
            content="Cal.",
            confidence=Confidence(
                value=0.8, calibration_source=CalibrationSource.CALIBRATED_BENCHMARK
            ),
        )
        report = await get_quality_report(db_session)  # type: ignore[arg-type]
        assert abs(report.extractor_direct_fraction - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_only_active_particles_in_calibration(self, db_session: object) -> None:
        await _insert(db_session, content="Active.")
        # A PROVENANCE_STALE birth is only legal as a quarantined
        # conflict loser (insert_particle enforces the reason condition).
        await _insert(
            db_session,
            content="Quarantined.",
            status=Status.PROVENANCE_STALE,
            status_reason=StatusReason.CONFLICT_PENDING,
        )
        report = await get_quality_report(db_session)  # type: ignore[arg-type]
        assert report.active_particles == 1
        total_cal = sum(b.count for b in report.calibration)
        assert total_cal == 1


class TestQualityReportCorpus:
    @pytest.mark.asyncio
    async def test_corpus_snapshot_counts(self, db_session: object) -> None:
        import tempfile
        from pathlib import Path

        from particles.corpus.deposit import deposit_file

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Hello world.")
            tmp = Path(f.name)
        try:
            await deposit_file(db_session, tmp, deposited_by="test")  # type: ignore[arg-type]
        finally:
            tmp.unlink(missing_ok=True)

        report = await get_quality_report(db_session)  # type: ignore[arg-type]
        assert report.total_entries == 1
        assert report.snapshots_pending == 1


class TestQualityReportSubjects:
    @pytest.mark.asyncio
    async def test_subjects_without_particles(self, db_session: object) -> None:
        from particles.core.schema import Subject
        from particles.store.subject_store import insert_subject

        s = Subject(canonical_name="Orphan Subject", asserted_by="test")
        await insert_subject(db_session, s)  # type: ignore[arg-type]

        report = await get_quality_report(db_session)  # type: ignore[arg-type]
        assert report.total_subjects == 1
        assert report.subjects_without_particles == 1

    @pytest.mark.asyncio
    async def test_covered_subject_not_counted(self, db_session: object) -> None:
        from particles.core.schema import Subject
        from particles.store.subject_store import insert_subject, link_particle_to_subjects

        s = Subject(canonical_name="Covered Subject", asserted_by="test")
        await insert_subject(db_session, s)  # type: ignore[arg-type]
        p = await _insert(db_session, content="Claim about subject.")
        await link_particle_to_subjects(db_session, p.id, [s.id])  # type: ignore[arg-type]

        report = await get_quality_report(db_session)  # type: ignore[arg-type]
        assert report.total_subjects == 1
        assert report.subjects_without_particles == 0
