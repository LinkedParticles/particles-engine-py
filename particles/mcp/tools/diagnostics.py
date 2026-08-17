"""``quality_report`` — extraction-quality dashboard snapshot.

Routed through the ``Backend`` seam.
"""

from __future__ import annotations

from typing import Any


async def quality_report() -> dict[str, Any]:
    """Return the extraction-quality dashboard snapshot.

    Particle counts by status / calibration_source / schema_version,
    plus the structural-mix percentages. Computed entirely from live
    DB queries — no LLM involvement.
    """
    from particles.api.client import get_backend

    report = await get_backend().quality()
    return report.model_dump(mode="json")
