"""Coverage-gap detectors for the §9.3 Query response.

Two flavours of gap are surfaced to the caller alongside the answer:

  - ``_find_coverage_gaps`` — corpus-level: entries with PENDING/FAILED
    snapshots that haven't been extracted yet.
  - ``_find_subject_coverage_gaps`` — subject-level: only populated when
    the request carried an explicit ``subject_id`` filter. Reports
    NO_SUBJECT_MATCH, SUBJECT_HAS_NO_PARTICLES, or SUBJECT_HAS_LOW_COVERAGE.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import CoverageGapKind, SubjectCoverageGap


async def _find_coverage_gaps(session: AsyncSession) -> list[str]:
    """Return corpus entry_ids with PENDING or FAILED extraction status."""
    from particles.core.schema import ExtractionStatus
    from particles.corpus.store import list_entry_ids_with_extraction_status

    return await list_entry_ids_with_extraction_status(
        session, [ExtractionStatus.PENDING, ExtractionStatus.FAILED]
    )


async def _find_subject_coverage_gaps(
    session: AsyncSession,
    subject_id: str | None,
    retrieved_count: int,
    low_coverage_threshold: int = 3,
) -> list[SubjectCoverageGap]:
    """Return subject-level coverage gaps for the queried subject (if any).

    Only populated when an explicit subject_id filter was used on the request.
    """
    if subject_id is None:
        return []

    from particles.store.subject_store import get_particle_count_for_subject, get_subject

    subject = await get_subject(session, subject_id)
    if subject is None:
        return [
            SubjectCoverageGap(
                kind=CoverageGapKind.NO_SUBJECT_MATCH,
                detail=f"Subject ID {subject_id!r} not found in the registry",
            )
        ]

    count = await get_particle_count_for_subject(session, subject.id)

    if count == 0:
        return [
            SubjectCoverageGap(
                subject_id=subject.id,
                subject_name=subject.canonical_name,
                kind=CoverageGapKind.SUBJECT_HAS_NO_PARTICLES,
                particle_count=0,
                detail=(
                    f"'{subject.canonical_name}' exists in the registry"
                    " but has no ACTIVE CLAIM particles"
                ),
            )
        ]

    if retrieved_count == 0 or count < low_coverage_threshold:
        return [
            SubjectCoverageGap(
                subject_id=subject.id,
                subject_name=subject.canonical_name,
                kind=CoverageGapKind.SUBJECT_HAS_LOW_COVERAGE,
                particle_count=count,
                detail=(
                    f"'{subject.canonical_name}' has only {count} ACTIVE CLAIM particle(s); "
                    f"answer may be incomplete"
                ),
            )
        ]

    return []
