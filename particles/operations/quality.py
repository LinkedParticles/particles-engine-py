"""Extraction quality dashboard (Appendix B §8).

Five aggregate queries delegated to the store layer; no LLM calls, no
mutations. Returns a QualityReport with particle calibration distribution,
corpus snapshot status counts, and subject coverage metrics.

For full structural and semantic diagnostics use the Lint operation.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import CalibrationBucket, QualityReport
from particles.core.status import Status
from particles.corpus.store import count_entries, count_snapshots_by_extraction_status
from particles.observability import traced
from particles.store.particle_store import (
    count_active_particles_by_calibration_source,
    count_particles_by_status,
    count_structured_claim_coverage,
)
from particles.store.subject_store import (
    count_subjects,
    count_subjects_without_active_particles,
)

log = logging.getLogger(__name__)


@traced("quality")
async def get_quality_report(session: AsyncSession) -> QualityReport:
    """Compute and return the extraction quality dashboard metrics."""

    status_counts = await count_particles_by_status(session)
    active_particles = status_counts.get(Status.ACTIVE.value, 0)
    inconsistency_particles = status_counts.get(Status.INCONSISTENCY.value, 0)

    cal_counts = await count_active_particles_by_calibration_source(session)
    total_active = active_particles or 1  # avoid division by zero
    calibration = [
        CalibrationBucket(
            source=src,
            count=cnt,
            fraction=round(cnt / total_active, 4),
        )
        for src, cnt in sorted(cal_counts.items(), key=lambda x: -x[1])
    ]
    # Extractor-only by design: this fraction tracks raw extractor output for the
    # metadata-theater alert (Risk #11). Uncalibrated agent self-reports
    # (AGENT_ASSERTED) appear as their own `calibration` bucket above and
    # are surfaced by the COMPOUND_ASSERTION lint — not folded in here, since
    # recalibration benchmarks apply to extractors, not self-reports.
    direct_count = cal_counts.get("EXTRACTOR_DIRECT", 0)
    extractor_direct_fraction = round(direct_count / total_active, 4) if active_particles else 0.0

    snap_counts = await count_snapshots_by_extraction_status(session)
    total_entries = await count_entries(session)
    total_subjects = await count_subjects(session)
    subjects_without_particles = await count_subjects_without_active_particles(session)
    # exporters and reports surface structured-claim coverage;
    # nothing generates the annotation outside extraction and `particles structure`.
    coverage = await count_structured_claim_coverage(session)

    return QualityReport(
        active_particles=active_particles,
        inconsistency_particles=inconsistency_particles,
        calibration=calibration,
        extractor_direct_fraction=extractor_direct_fraction,
        total_entries=total_entries,
        snapshots_pending=snap_counts.get("PENDING", 0),
        snapshots_in_progress=snap_counts.get("IN_PROGRESS", 0),
        snapshots_complete=snap_counts.get("COMPLETE", 0),
        snapshots_failed=snap_counts.get("FAILED", 0),
        total_subjects=total_subjects,
        subjects_without_particles=subjects_without_particles,
        structured_claims=int(coverage["annotated"]),
        structured_claims_by_structurizer=dict(coverage["by_structurizer"]),
    )
