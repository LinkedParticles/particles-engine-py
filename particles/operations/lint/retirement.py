"""Undated-retirement detector.

One aggregate, read-only structural finding: how many once-believed retired
particles carry no stored retirement instant (``NULL retired_at``). Shares the
born-retired exclusion helper with the query disclosure count
(``operations/query/as_of.py``), so quarantine losers and INCONSISTENCY
records — never believed, correctly unstamped — are not counted.

On a store born after migration 029 the count is zero and any hit is an
anomaly; on an older store it names the fixed legacy population. **Growth over
time is the alarm** — it means an out-of-band writer or a path bypassing the
``update_particle_status`` choke point is retiring particles without stamping.
Read-only by design: there is no ``--fix`` action, because a lost retirement
instant cannot be manufactured (the fail-closed rule).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import LintFinding
from particles.core.status import Status, StatusReason
from particles.operations.query.as_of import ensure_utc, is_once_believed_retirement
from particles.store.particle_store import ParticleRow


async def _check_undated_retirements(session: AsyncSession) -> list[LintFinding]:
    """Aggregate count of once-believed retirements with ``NULL retired_at``."""
    result = await session.execute(
        select(
            ParticleRow.status,
            ParticleRow.status_reason,
            ParticleRow.retired_at,
            ParticleRow.asserted_at,
        ).where(ParticleRow.status != Status.ACTIVE.value)
    )
    count = 0
    latest_asserted: datetime | None = None
    for status_value, reason_value, retired_at, asserted_at in result.all():
        status = Status(status_value)
        reason = StatusReason(reason_value) if reason_value else None
        if not is_once_believed_retirement(status, reason):
            continue
        if retired_at is not None:
            continue
        count += 1
        asserted = ensure_utc(asserted_at)
        if latest_asserted is None or asserted > latest_asserted:
            latest_asserted = asserted
    if count == 0:
        return []
    latest = latest_asserted.isoformat() if latest_asserted is not None else "unknown"
    return [
        LintFinding(
            finding_type="UNDATED_RETIREMENT",
            severity="WARNING",
            detail=(
                f"{count} once-believed retired particle(s) carry no stored "
                f"retirement instant (NULL retired_at); most recent asserted_at "
                f"in that set: {latest}. Where the reconstruction "
                f"ladder cannot date them either, they are excluded fail-closed "
                f"from --as-of views (with a disclosure count)."
            ),
            recommended_action=(
                "Expected only for pre-migration history. Growth on a "
                "post-migration store means an out-of-band writer or a code "
                "path bypassing update_particle_status is retiring particles "
                "without stamping — find and fix that writer."
            ),
        )
    ]
