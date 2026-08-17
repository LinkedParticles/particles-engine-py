"""Back-compat shim — the article-synthesis engine moved to
:mod:`particles.render.article_synthesis`.

The cross-exporter article-rendering machinery used to live here, but
housing it under ``particles.exporters`` made the Engine *reasoning* layer
(``operations.narrative_synthesis``) import the Engine *output* layer
(``exporters``) — the last ``operations → exporters`` back-edge after the
Markdown renderer moved to :mod:`particles.render`. It now lives at
:mod:`particles.render.article_synthesis`, alongside that renderer, so both
``operations`` and ``exporters`` depend on it downward.

Importing the public surface from ``particles.exporters.article_synthesis`` still
works — this shim re-exports it. **New code should import from
:mod:`particles.render.article_synthesis`** (and its submodules directly for the
internal helpers, e.g. ``particles.render.article_synthesis.cache``).
"""

from __future__ import annotations

from particles.render.article_synthesis import (
    _ARTICLE_TAG,
    _CITATION_RE,
    _FRONTMATTER_RE,
    _INLINE_FOOTNOTE_DEF_RE,
    _LAYER_B_INSTRUCTIONS,
    _LAYER_B_WINDOW,
    _LINT_CALLOUT_TYPES,
    _PROMPT_VERSION,
    _SYNTHESIS_PROMPT_FLOWING,
    _SYNTHESIS_PROMPT_NARRATIVE,
    _SYNTHESIS_PROMPT_STANDARD,
    _SYNTHESIS_PROMPT_STRICT,
    _SYNTHESIS_PROMPT_STRICT_LAYER_B,
    _VERDICT_ALIASES,
    _WIKILINK_RE,
    LayerBResult,
    SynthesisUnavailable,
    _build_layer_b_prompt,
    _build_synthesis_prompt,
    _call_synthesis_llm,
    _count_contradictions,
    _decide_layer_b_pass,
    _extract_wikilink_targets,
    _format_misalignments_for_prompt,
    _format_particles_for_prompt,
    _normalise_verdict,
    _parse_frontmatter,
    _parse_layer_b_verdicts,
    _render_frontmatter,
    _render_lint_callouts,
    _short_id,
    _strip_inline_footnote_defs,
    _strip_input_hash_from_frontmatter,
    apply_lint_callouts,
    compute_input_hash,
    count_uncited_paragraphs,
    extract_cited_segments,
    invalidate_stale_link_articles,
    layer_b_check,
    render_article,
    render_structured_listing,
    render_synthesised_article,
    split_rendered_article,
    strip_lint_callouts,
    validate_citations,
)

__all__ = [
    "compute_input_hash",
    "invalidate_stale_link_articles",
    "validate_citations",
    "layer_b_check",
    "render_article",
    "SynthesisUnavailable",
    "render_synthesised_article",
    "render_structured_listing",
    "apply_lint_callouts",
    "strip_lint_callouts",
    "split_rendered_article",
    "extract_cited_segments",
    "count_uncited_paragraphs",
    "LayerBResult",
    "_ARTICLE_TAG",
    "_CITATION_RE",
    "_FRONTMATTER_RE",
    "_INLINE_FOOTNOTE_DEF_RE",
    "_LAYER_B_INSTRUCTIONS",
    "_LAYER_B_WINDOW",
    "_LINT_CALLOUT_TYPES",
    "_PROMPT_VERSION",
    "_SYNTHESIS_PROMPT_FLOWING",
    "_SYNTHESIS_PROMPT_NARRATIVE",
    "_SYNTHESIS_PROMPT_STANDARD",
    "_SYNTHESIS_PROMPT_STRICT",
    "_SYNTHESIS_PROMPT_STRICT_LAYER_B",
    "_VERDICT_ALIASES",
    "_WIKILINK_RE",
    "_build_layer_b_prompt",
    "_build_synthesis_prompt",
    "_call_synthesis_llm",
    "_count_contradictions",
    "_decide_layer_b_pass",
    "_extract_wikilink_targets",
    "_format_misalignments_for_prompt",
    "_format_particles_for_prompt",
    "_normalise_verdict",
    "_parse_frontmatter",
    "_parse_layer_b_verdicts",
    "_render_frontmatter",
    "_render_lint_callouts",
    "_short_id",
    "_strip_inline_footnote_defs",
    "_strip_input_hash_from_frontmatter",
]
