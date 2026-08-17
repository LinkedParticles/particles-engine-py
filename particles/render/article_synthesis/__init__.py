"""Article synthesis helper — shared rendering layer for cited prose articles.

This package is the **content** half of the wiki/article-synthesis design
originally introduced and lifted out of ``wiki.py``. Every exporter that wants per-Subject prose articles
(WikiExporter, Obsidian's ``--with-synthesis`` mode, the Logseq
exporter, …) goes through these helpers — and so does the narrative-synthesis path in ``operations``.

**Location / layer (follow-up).** This package used to live at
``particles/exporters/article_synthesis/``, which forced Engine *reasoning*
(``operations.narrative_synthesis``) to import the Engine *output* layer
(``exporters``) — the last ``operations → exporters`` back-edge after the
Markdown renderer moved to :mod:`particles.render`. It now lives alongside
that renderer, so both ``operations`` and ``exporters`` depend on it
*downward*. Unlike :mod:`particles.render.markdown` (a store-free Client
utility), this package is the **Engine** half of the ``render`` straddle: it
reaches the synthesis cache in ``store`` (deferred import in
:mod:`~particles.render.article_synthesis.orchestrate`) and calls the LLM. The
import-linter contracts in ``pyproject.toml`` reflect the split — ``render.markdown``
is pinned Client, ``render.article_synthesis`` is listed in the Engine tier (the
same per-module straddle treatment as ``interchange``). A thin back-compat shim
remains at ``particles/exporters/article_synthesis`` re-exporting this surface.

The exporter retains responsibility for:

* loading qualifying subjects from the DB,
* computing per-particle effective confidence,
* deciding where the rendered article lives on disk,
* lint pre-pass coordination, trust-cache warming, and
* per-export progress reporting.

This package is responsible for:

* the synthesis prompts and validation layers,
* Layer A regex citation-ID-membership validation,
* Layer B semantic-alignment LLM-judge,
* retry-then-fallback orchestration,
* the structured-listing fallback render,
* the synthesised-article render (frontmatter + body + References),
* the cache-key (input-hash) computation.

The helper is **filesystem-blind** — it returns Markdown bodies; the
exporter writes them. This makes the same article body cacheable across
exporters (the same ``input_hash`` validates a body whether the exporter
stores it under ``./wiki-export/{slug}.md`` or
``./obsidian-vault/{slug}.md``).

The 1300-line single-file ``article_synthesis.py`` was split into a
sub-package once it crossed the maintenance threshold (same pattern as
the M1 github/numista and M3 obsidian splits). Submodules:

* :mod:`particles.render.article_synthesis.cache` — cache key
  (``compute_input_hash``) and rendered-article decomposition
  (``split_rendered_article``, ``_parse_frontmatter``).
* :mod:`particles.render.article_synthesis.layer_a` — deterministic
  citation validation (``validate_citations``,
  ``count_uncited_paragraphs``, ``_strip_inline_footnote_defs``) plus
  the canonical ``_short_id`` and ``_CITATION_RE`` constants.
* :mod:`particles.render.article_synthesis.layer_b` — semantic-
  alignment LLM-judge (``layer_b_check``, ``LayerBResult``, the
  trichotomy verdict normaliser) and the shared LLM-call seam
  ``_call_synthesis_llm``.
* :mod:`particles.render.article_synthesis.render` — the YAML
  frontmatter renderer, the synthesis prompt constants, and the two
  body renderers (``render_structured_listing``,
  ``render_synthesised_article``).
* :mod:`particles.render.article_synthesis.orchestrate` —
  ``render_article``, the public entry point that chains the LLM call
  with Layer-A retry + Layer-B retry and the structured-listing
  fallback.

The dependency graph is acyclic:
``orchestrate → {cache, layer_a, layer_b, render}``;
``render → layer_a`` (for ``_short_id``);
``layer_b → layer_a`` (for ``_short_id`` + ``_CITATION_RE``);
``cache`` and ``layer_a`` have no internal deps.

Consumers that pin this package's public surface as load-bearing:

* ``particles/exporters/wiki.py`` — re-exports
  ``compute_input_hash``, ``validate_citations``, ``layer_b_check``,
  ``render_article``, ``render_synthesised_article``,
  ``render_structured_listing``, ``extract_cited_segments``,
  ``_parse_frontmatter``, ``_parse_layer_b_verdicts``.
* ``particles/exporters/obsidian/synthesis.py`` — uses
  ``render_article`` and ``split_rendered_article``.
* ``particles/exporters/obsidian/vault.py`` — uses
  ``compute_input_hash`` and ``_parse_frontmatter``.
* ``tests/test_wiki_exporter.py`` — also imports
  ``_strip_inline_footnote_defs``, ``split_rendered_article``,
  ``count_uncited_paragraphs``, ``_normalise_verdict``,
  ``_decide_layer_b_pass``, ``LayerBResult``, ``layer_b_check``,
  ``_build_synthesis_prompt``, plus ``article_synthesis as syn`` for
  attribute-style access to ``_PROMPT_VERSION`` and
  ``compute_input_hash``.
* ``tests/test_dogfood_corpus.py`` — uses ``_parse_frontmatter``.

Anything that appears in those lists must remain reachable through
this ``__init__.py``.
"""

from __future__ import annotations

from particles.render.article_synthesis.cache import (
    _FRONTMATTER_RE,
    _PROMPT_VERSION,
    _WIKILINK_RE,
    _extract_wikilink_targets,
    _parse_frontmatter,
    _strip_input_hash_from_frontmatter,
    compute_input_hash,
    find_unresolved_wikilinks,
    invalidate_stale_link_articles,
    split_rendered_article,
)
from particles.render.article_synthesis.layer_a import (
    _CITATION_RE,
    _INLINE_FOOTNOTE_DEF_RE,
    _short_id,
    _strip_inline_footnote_defs,
    count_uncited_paragraphs,
    validate_citations,
)
from particles.render.article_synthesis.layer_b import (
    _LAYER_B_INSTRUCTIONS,
    _LAYER_B_WINDOW,
    _VERDICT_ALIASES,
    LayerBResult,
    _build_layer_b_prompt,
    _call_synthesis_llm,
    _decide_layer_b_pass,
    _normalise_verdict,
    _parse_layer_b_verdicts,
    extract_cited_segments,
    layer_b_check,
)
from particles.render.article_synthesis.orchestrate import (
    SynthesisUnavailable,
    render_article,
)
from particles.render.article_synthesis.render import (
    _ARTICLE_TAG,
    _LINT_CALLOUT_TYPES,
    _SYNTHESIS_PROMPT_FLOWING,
    _SYNTHESIS_PROMPT_NARRATIVE,
    _SYNTHESIS_PROMPT_STANDARD,
    _SYNTHESIS_PROMPT_STRICT,
    _SYNTHESIS_PROMPT_STRICT_LAYER_B,
    _build_synthesis_prompt,
    _count_contradictions,
    _format_misalignments_for_prompt,
    _format_particles_for_prompt,
    _render_frontmatter,
    _render_lint_callouts,
    apply_lint_callouts,
    render_structured_listing,
    render_synthesised_article,
    strip_lint_callouts,
)
from particles.render.article_synthesis.topic import (
    SectionTopic,
    SynthesisTopic,
)

__all__ = [
    # Public surface — imported by exporters and external callers.
    "compute_input_hash",
    "find_unresolved_wikilinks",
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
    "SynthesisTopic",
    "SectionTopic",
    # Internal helpers — re-exported because the test suite imports them
    # by name (or attribute-accesses them via ``article_synthesis as
    # syn``). New consumers should import from the owning submodule.
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
