"""Back-compat shim — the Markdown rendering utilities moved to
:mod:`particles.render.markdown`.

These are a Client-layer rendering utility (pure functions over Pydantic models
plus filesystem-safe write/slug helpers), not an exporter. Housing them under
``particles.exporters`` made the Engine *reasoning* layer (``operations`` —
digest, inbox, lint) import the Engine *output* layer (``exporters``), an
upward dependency / package cycle. They now live in :mod:`particles.render`,
below both layers.

Importing from ``particles.exporters.markdown`` still works — the exporters and
the existing test suite rely on it. New code should import from
``particles.render.markdown``.
"""

from __future__ import annotations

from particles.render.markdown import (
    DigestEntry,
    DisambiguationGroup,
    SubjectNaming,
    atomic_write_text,
    build_narrative_naming,
    build_subject_naming,
    disambiguation_name,
    exclude_non_asserted,
    is_within_directory,
    prune_obsolete_markdown,
    render_contested_callout,
    render_digest,
    render_lint_finding,
    render_lint_report,
    render_particle,
    render_particles,
    render_stance_callout,
    sanitize_filename,
    subject_slug,
)

__all__ = [
    "DigestEntry",
    "DisambiguationGroup",
    "SubjectNaming",
    "atomic_write_text",
    "build_narrative_naming",
    "build_subject_naming",
    "disambiguation_name",
    "exclude_non_asserted",
    "is_within_directory",
    "prune_obsolete_markdown",
    "render_contested_callout",
    "render_digest",
    "render_lint_finding",
    "render_lint_report",
    "render_particle",
    "render_particles",
    "render_stance_callout",
    "sanitize_filename",
    "subject_slug",
]
