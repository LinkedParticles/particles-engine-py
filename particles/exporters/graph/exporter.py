"""GraphExporter — the scoped epistemic graph view plugin.

A single-file exporter: one self-contained HTML artifact rendering one scoped
subgraph. Scope is mandatory (``subject`` or ``query``) — there
is deliberately no whole-store default. The subgraph assembly lives in
:func:`particles.operations.graph_view.build_graph_data`; this plugin is the
thin presentation shell that writes the HTML.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from particles.exporters.summaries import GraphSummary

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


class GraphExporter:
    """Scoped self-contained-HTML graph exporter."""

    FORMAT = "graph"

    async def export(
        self,
        session: AsyncSession,
        output: Path | None,
        **options: object,
    ) -> GraphSummary:
        """Render one scoped subgraph to a self-contained HTML file.

        Options:
            subject (str | None): anchor subject id — neighbourhood scope.
            query (str | None): question text — retrieval-set scope.
            inconsistency (str | None): INCONSISTENCY particle id (or unique
                prefix) — a contradiction's evidence scope.
            manifest (str | None) + section (str | None): a
                manifest section's deterministic selection.
            hops (int, default 1): neighbourhood radius for subject scope,
                clamped to ``graph.max_hops``.
            history (bool, default False): include retired supersession-chain
                ancestors as ghosts.
            as_of (datetime | None): render the graph as believed at T
                (single-instant lens).
            max_nodes (int | None): per-run node cap, clamped to
                ``graph.max_nodes``.
            min_particle_confidence (float, default 0.0): the
                cross-exporter floor on effective confidence.
            include_non_asserted (bool, default False): keep DECLINED /
                HYPOTHETICAL particles.
        """
        if output is None:
            raise ValueError("GraphExporter requires an output file path (out.html)")

        from particles.operations.graph_view import build_graph_data

        from .render import render_html

        raw_mpc = options.get("min_particle_confidence")
        min_particle_confidence = float(raw_mpc) if raw_mpc is not None else 0.0  # type: ignore[arg-type]
        raw_subject = options.get("subject")
        raw_query = options.get("query")
        raw_inconsistency = options.get("inconsistency")
        raw_manifest = options.get("manifest")
        raw_section = options.get("section")
        raw_hops = options.get("hops")
        raw_max_nodes = options.get("max_nodes")
        raw_as_of = options.get("as_of")

        def _opt(v: object) -> str | None:
            return v if isinstance(v, str) and v else None

        data = await build_graph_data(
            session,
            subject_id=_opt(raw_subject),
            query=_opt(raw_query),
            inconsistency_id=_opt(raw_inconsistency),
            manifest=_opt(raw_manifest),
            section=_opt(raw_section),
            hops=raw_hops if isinstance(raw_hops, int) else 1,
            history=bool(options.get("history", False)),
            as_of=raw_as_of if isinstance(raw_as_of, datetime) else None,
            max_nodes=raw_max_nodes if isinstance(raw_max_nodes, int) else None,
            min_particle_confidence=min_particle_confidence,
            include_non_asserted=bool(options.get("include_non_asserted", False)),
        )

        html = render_html(data, generated_at=datetime.now(UTC))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        log.info(
            "graph export: %s → %s (%d nodes, %d edges)",
            data.census.scope,
            output,
            len(data.nodes),
            len(data.edges),
        )

        return GraphSummary(
            subjects=len(data.nodes),
            candidate_subjects=data.census.candidate_subjects,
            particles=len(data.particles),
            edges=len(data.edges),
            files_written=1,
            min_particle_confidence=min_particle_confidence,
            particles_dropped_below_threshold=data.census.dropped_below_threshold,
        )
