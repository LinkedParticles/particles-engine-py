"""Typed exporter summary models.

Every :class:`particles.exporters.registry.ExporterPlugin` returns a
concrete subclass of :class:`BaseExporterSummary`. The
``format: Literal[...]`` discriminator lets downstream consumers
(MCP tools, the CLI, scripts) switch on the value without
``isinstance`` checks, and Pydantic validation locks the field shape
in for the 1.0.0 strict-SemVer regime.

The CLI consumes these via ``summary.model_dump(exclude_none=True)``
so conditional fields (e.g. ``synthesis_used`` on the wiki exporter
only when not ``dry_run``) stay quiet when the step did not run.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class BaseExporterSummary(BaseModel):
    """Common shape every exporter summary satisfies.

    ``format`` is the discriminator. ``particles_dropped_below_threshold``
    is the universal audit field — every exporter reports it,
    even when the threshold is 0.0 and the value is 0.
    """

    format: Literal["wiki", "obsidian", "anki", "logseq", "jsonl", "notion", "graph"]
    particles_dropped_below_threshold: int = 0


class WikiSummary(BaseExporterSummary):
    """Summary returned by :class:`particles.exporters.wiki.WikiExporter`."""

    format: Literal["wiki"] = "wiki"
    dry_run: bool
    qualifying_subjects: int
    cache_hits: int
    articles_regenerated: int
    estimated_prompt_tokens: int
    min_particle_confidence: float
    # Present only when ``dry_run=False``.
    articles_written: int | None = None
    synthesis_used: int | None = None
    synthesis_failed: int | None = None
    # Present only when ``without_synthesis=True``: articles written as
    # deterministic structured listings with no LLM call.
    synthesis_skipped: int | None = None
    # Count of per-NARRATIVE articles written under ``Narratives/`` (
    # the mechanism); present only when ``wiki.emit_narrative_notes``
    # is on and the run was not narrowed with ``--subjects``.
    narrative_articles: int | None = None
    # Present only when ``invalidate_stale_links=True``.
    stale_link_articles_invalidated: int | None = None


class ObsidianSummary(BaseExporterSummary):
    """Summary returned by :class:`particles.exporters.obsidian.ObsidianExporter`."""

    format: Literal["obsidian"] = "obsidian"
    subjects: int
    particles: int
    phantoms: int
    suppressed: int
    files_written: int
    # Present only when ``with_synthesis=True``.
    synthesis_used: int | None = None
    synthesis_failed: int | None = None
    synthesis_cache_hits: int | None = None
    synthesis_skipped: int | None = None
    # Count of per-NARRATIVE notes written under ``Narratives/``;
    # present only when ``with_synthesis=True`` and ``obsidian.emit_narrative_notes``.
    narrative_notes: int | None = None
    # Present only when ``invalidate_stale_links=True``.
    stale_link_articles_invalidated: int | None = None


class AnkiSummary(BaseExporterSummary):
    """Summary returned by :class:`particles.exporters.anki.AnkiExporter`."""

    format: Literal["anki"] = "anki"
    cards_written: int
    decks: int


class JsonlSummary(BaseExporterSummary):
    """Summary returned by :class:`particles.exporters.jsonl.JsonlExporter`."""

    format: Literal["jsonl"] = "jsonl"
    particles_written: int


class LogseqSummary(BaseExporterSummary):
    """Summary returned by :class:`particles.exporters.logseq.LogseqExporter`.

    Field set mirrors :class:`ObsidianSummary` — the two exporters do
    equivalent work, only the on-disk format differs (bullet-outline
    pages vs free-form Markdown notes)."""

    format: Literal["logseq"] = "logseq"
    subjects: int
    particles: int
    phantoms: int
    suppressed: int
    files_written: int
    # Present only when ``with_synthesis=True``.
    synthesis_used: int | None = None
    synthesis_failed: int | None = None
    synthesis_cache_hits: int | None = None
    synthesis_skipped: int | None = None
    # Count of per-NARRATIVE pages written into the ``Narratives/`` page
    # namespace (mechanism); present only when
    # ``with_synthesis=True`` and ``logseq.emit_narrative_notes``.
    narrative_notes: int | None = None
    # Present only when ``invalidate_stale_links=True``.
    stale_link_articles_invalidated: int | None = None


class GraphSummary(BaseExporterSummary):
    """Summary returned by :class:`particles.exporters.graph.GraphExporter`.

    ``subjects`` vs ``candidate_subjects`` is the anti-hairball disclosure in
    summary form: when they differ, the ``graph.max_nodes`` cap bound and the
    rendered page carries the bounded-view-style banner — the render is a
    disclosed lower bound, not a census.
    """

    format: Literal["graph"] = "graph"
    # Rendered Subject nodes (post-cap).
    subjects: int
    # What an uncapped render would have shown.
    candidate_subjects: int
    # Rendered particles (edges + node cargo, incl. history ghosts).
    particles: int
    # Rendered subject-pair edge segments.
    edges: int
    files_written: int
    min_particle_confidence: float


class NotionSummary(BaseExporterSummary):
    """Summary returned by :class:`particles.exporters.notion.NotionExporter`.

    The Notion exporter is the first **API-target** exporter: it writes to one
    operator-provided Notion database rather than the local filesystem. The
    dry-run path (``dry_run=True``) reads the store and computes the full plan
    but makes **zero** Notion API writes and skips the existence probe, so the
    created-vs-updated split (``pages_to_create`` / ``pages_to_update``) is
    ``None`` on a dry run — only the planned totals are known.

    The run-only fields (``pages_created`` / ``pages_updated`` / ``api_calls``)
    follow the wiki exporter's ``field present only when not dry_run``
    convention: the CLI consumes ``model_dump(exclude_none=True)`` so they stay
    quiet on a dry run.
    """

    format: Literal["notion"] = "notion"
    dry_run: bool
    # Subjects that qualify after the min_particles count check.
    subjects_planned: int
    # Total particles synced across all qualifying pages (multi-subject edges
    # de-duplicated by particle id within a page).
    particles_synced: int
    # universal field is inherited from BaseExporterSummary
    # (particles_dropped_below_threshold).
    min_particle_confidence: float
    # The managed Notion database id this run synced into.
    database_id: str | None = None
    # When False (``--no-update-blocks``), the exporter only
    # creates new pages' blocks and never rewrites an existing page's managed
    # block range — hand-edits inside the range survive.
    update_blocks: bool = True
    # Present only when ``dry_run=False`` — the existence-probe split.
    pages_created: int | None = None
    pages_updated: int | None = None
    api_calls: int | None = None
