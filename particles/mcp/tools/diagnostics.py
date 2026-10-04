# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``quality_report`` — extraction-quality dashboard snapshot.

Routed through the ``Backend`` seam.
"""

from __future__ import annotations

from typing import Any


async def quality_report() -> dict[str, Any]:
    """Return the extraction-quality dashboard snapshot.

    Particle counts by status / calibration_source / schema_version,
    plus the structural-mix percentages, the store's recorded LLM spend
    (``llm_spend``), and the curation queue's precision over the default
    window (``curation_precision``: per card kind, the cards acted on,
    dismissed, snoozed and untouched, with the denominator). Computed
    entirely from live DB queries — no LLM involvement.
    """
    from particles.api.client import get_backend

    report = await get_backend().quality()
    return report.model_dump(mode="json")
