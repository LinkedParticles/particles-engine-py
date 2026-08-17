"""JSON Lines exporter.

A single ``.jsonl`` file with one ACTIVE particle per line — a flat,
analysis-friendly dump (jq / pandas / DuckDB) of the full particle model.

This is **distinct from the interchange format** (``particles interchange
export``): interchange writes a round-trippable bundle of JSON-LD
units (substrate-only, subjects by external reference) for store-to-store
transfer; this exporter writes the full Pydantic particle dump for downstream
consumption, not re-import. Reach for interchange when you want to *move a
store*; reach for ``export jsonl`` when you want your particles *as data*.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from particles.core.scoring.confidence import compute_effective_confidence
from particles.core.status import Status
from particles.exporters.summaries import JsonlSummary

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


class JsonlExporter:
    """One-particle-per-line JSON Lines exporter."""

    FORMAT = "jsonl"

    async def export(
        self,
        session: AsyncSession,
        output: Path | None,
        **options: object,
    ) -> JsonlSummary:
        """Write every ACTIVE particle as one JSON object per line.

        Options:
            min_particle_confidence (float, default 0.0): drop particles whose
                ``effective_confidence`` (trust-weighted) is
                below this threshold before writing (cross-exporter contract).
            include_non_asserted (bool, default False): keep non-asserted
                particles — a document's rejected / superseded / deferred /
                counterfactual prose (polarity DECLINED / HYPOTHETICAL). Excluded from the default surface; set True to keep.
        """
        if output is None:
            raise ValueError("JsonlExporter requires an output file path")

        raw_mpc = options.get("min_particle_confidence")
        min_particle_confidence = float(raw_mpc) if raw_mpc is not None else 0.0  # type: ignore[arg-type]

        from particles.operations.query.source_trust import load_source_trust_ranks
        from particles.render.markdown import exclude_non_asserted
        from particles.store.extractor_store import (
            get_cached_trust_weight,
            get_trust_weight_map,
            populate_trust_cache,
        )
        from particles.store.particle_store import get_particles_by_status

        particles_all = exclude_non_asserted(
            await get_particles_by_status(session, Status.ACTIVE), options
        )
        populate_trust_cache(await get_trust_weight_map(session))
        source_ranks = await load_source_trust_ranks(session, particles_all)

        lines: list[str] = []
        dropped = 0
        for p in sorted(particles_all, key=lambda x: x.id):
            extractor_id = p.extractor_ref.name if p.extractor_ref else ""
            trust = get_cached_trust_weight(extractor_id) if extractor_id else 1.0
            eff = compute_effective_confidence(
                p.confidence.value,
                extractor_trust_weight=trust,
                source_trust_rank=source_ranks.get(p.id, 1.0),
                calibration_source=p.confidence.calibration_source,
            )
            if eff < min_particle_confidence:
                dropped += 1
                continue
            lines.append(json.dumps(p.model_dump(mode="json"), ensure_ascii=False))

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        log.info("JSONL export: %d particles → %s", len(lines), output)

        return JsonlSummary(
            particles_written=len(lines),
            particles_dropped_below_threshold=dropped,
        )
