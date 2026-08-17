"""Obsidian vault exporter.

Renders the subject+particle graph as a folder of interlinked Markdown
files, one per Subject, suitable for Obsidian's graph view.

This package was split out of a single ``obsidian.py`` once the
file passed ~1300 lines. The submodules are:

* :mod:`particles.exporters.obsidian.exporter` — the registered
  :class:`ObsidianExporter` plugin class.
* :mod:`particles.exporters.obsidian.vault` — note dispatch
  (coin / pivot / generic templates), :func:`export_vault`,
  index rendering, audit-trail rendering.
* :mod:`particles.exporters.obsidian.format` — Obsidian-flavoured
  Markdown shaping (wikilinks, callouts, block-references, frontmatter
  splice).
* :mod:`particles.exporters.obsidian.synthesis` — the LLM-synthesis
  splice for per-subject notes.

The exporter registry imports :class:`ObsidianExporter` from
``particles.exporters.obsidian`` directly; tests reach for several
private helpers by name. Both surfaces are preserved here as
re-exports.
"""

from __future__ import annotations

from particles.exporters.obsidian.exporter import ObsidianExporter
from particles.exporters.obsidian.format import (
    _annotate_obsidian_frontmatter,
    _claim_for_heading,
    _insert_synthesised_prose,
    _is_category,
    _note_has_particle_audit_trail,
    _provenance_label,
    _render_obsidian_reference_entry,
    _render_particle_audit_callouts,
    _source_link_line,
    _strip_per_particle_callouts,
    _to_obsidian_block_refs,
    _to_superscript,
    _wiki,
)
from particles.exporters.obsidian.synthesis import (
    _should_skip_synthesis_for_low_coverage,
    _splice_synthesised_article,
)
from particles.exporters.obsidian.vault import (
    _PIVOT_CLASSES,
    _PIVOT_TAG,
    _render_coin_note,
    _render_index,
    _render_pivot_note,
    _render_subject_note,
    _sanitize,
    _subject_slug,
    export_vault,
)

__all__ = [
    # Public surface — used by the exporter registry and external callers.
    "ObsidianExporter",
    "export_vault",
    # Internal helpers — re-exported because the test suite imports them
    # directly and ADR docs reference them by name. New consumers should
    # import from the submodule that owns the symbol.
    "_PIVOT_CLASSES",
    "_PIVOT_TAG",
    "_annotate_obsidian_frontmatter",
    "_claim_for_heading",
    "_insert_synthesised_prose",
    "_is_category",
    "_note_has_particle_audit_trail",
    "_provenance_label",
    "_render_coin_note",
    "_render_index",
    "_render_obsidian_reference_entry",
    "_render_particle_audit_callouts",
    "_render_pivot_note",
    "_render_subject_note",
    "_sanitize",
    "_should_skip_synthesis_for_low_coverage",
    "_source_link_line",
    "_splice_synthesised_article",
    "_strip_per_particle_callouts",
    "_subject_slug",
    "_to_obsidian_block_refs",
    "_to_superscript",
    "_wiki",
]
