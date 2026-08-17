"""Logseq exporter plugin — registered surface."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from particles.exporters.logseq.vault import export_vault
from particles.exporters.summaries import LogseqSummary


class LogseqExporter:
    """Logseq vault exporter plugin.

    Writes a Logseq graph: one ``pages/<subject_slug>.md`` per
    qualifying Subject, in Logseq's native bullet-outline format
    with particle IDs as block UUIDs (``id:: <particle_id>``) for
    cross-page citation via ``((<particle_id>))``.
    """

    FORMAT = "logseq"

    async def export(
        self,
        session: AsyncSession,
        output: Path | None,
        **options: object,
    ) -> LogseqSummary:
        """Export to a Logseq graph directory.

        Options:
            min_particles (int, default 0): minimum particle count per subject.
            min_links (int, default 1): minimum graph link count per subject.
            min_particle_confidence (float, default 0.0): drop particles
                with ``effective_confidence`` below this threshold from
                every rendered page.
            with_synthesis (bool, default False): splice LLM-synthesised
                prose articles into per-subject pages. Requires
                ANTHROPIC_API_KEY. Shares the cross-exporter synthesis
                cache so the LLM cost is paid once across
                wiki / obsidian / logseq.
            invalidate_stale_links (bool, default False): drop the
                article-cache hash from any page whose ``[[X]]``
                wikilinks reference a renamed subject.
        """
        if output is None:
            raise ValueError("LogseqExporter requires an output directory path")
        from particles.config import get_config

        common_cfg = get_config().exporter_common
        min_particles = int(options.get("min_particles", 0))  # type: ignore[call-overload]
        min_links = int(options.get("min_links", 1))  # type: ignore[call-overload]
        with_synthesis = bool(options.get("with_synthesis", False))
        invalidate_stale_links = bool(options.get("invalidate_stale_links", False))
        min_particle_confidence = float(
            options.get(  # type: ignore[arg-type]
                "min_particle_confidence", common_cfg.min_particle_confidence
            )
        )
        include_non_asserted = bool(options.get("include_non_asserted", False))
        return await export_vault(
            session,
            output,
            min_particles=min_particles,
            min_links=min_links,
            with_synthesis=with_synthesis,
            invalidate_stale_links=invalidate_stale_links,
            min_particle_confidence=min_particle_confidence,
            include_non_asserted=include_non_asserted,
        )
