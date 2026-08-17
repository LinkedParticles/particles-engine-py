"""synthesis splice for the Logseq exporter.

The orchestrator hands us the rendered structural-outline page;
this helper:

1. Calls :func:`render_article` (with the cross-exporter session
   so caching kicks in).
2. On cache hit / successful synthesis, splits the rendered
   article into prose + references and wraps both in a
   ``- ## Synthesis`` parent block whose children are the prose
   (one block) and a ``- ### References`` block containing the
   reference list.
3. Emits ``article_input_hash:: <hash>`` on the page's H1 block
   as rendered-artefact metadata so an operator inspecting the
   file can see what input it was built from. (The DB cache is
   the source of truth for cache lookups; the on-disk hash is for
   human inspection.)
4. Falls back to the existing structural-outline page if synthesis
   fails — the Logseq outline already lists every particle, so
   doubling them up via the structured-listing renderer would be
   pure noise.
"""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING

from particles.exporters.logseq.format import _INDENT  # noqa: PLC2701 — module-internal
from particles.render.article_synthesis import (
    render_article,
    split_rendered_article,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from particles.core.schema import LintFinding, Particle, Subject

log = logging.getLogger(__name__)


class _SynthesisOutcome(enum.Enum):
    """Per-subject synthesis result, mirrored in the exporter summary counts."""

    SYNTHESISED = "synthesised"
    CACHE_HIT = "cache_hit"
    FALLBACK = "fallback"
    SKIPPED = "skipped"


async def _splice_synthesised_article(
    *,
    page: str,
    subject: Subject,
    particles: list[Particle],
    eff_conf: dict[str, float],
    entry_uri_map: dict[str, str | None],
    article_hash: str,
    regenerate: bool,
    lint_findings: list[LintFinding],
    progress_prefix: str,
    session: AsyncSession,
) -> tuple[str, _SynthesisOutcome]:
    """Splice an LLM-synthesised article into a Logseq outline page.

    Returns ``(updated_page, outcome)``. The orchestrator tallies
    outcomes into the :class:`LogseqSummary` counters.
    """
    from particles.config import get_config

    cfg = get_config().wiki

    # Low-coverage short-circuit (matches Obsidian's behaviour). Subjects
    # with too few particles burn an LLM call to paraphrase a single
    # claim — net negative. Threshold lives in
    # ``exporter_common.synthesis_min_particles`` (default 3); the
    # Obsidian + Logseq exporters share it. The check runs *before* the
    # on-disk cache-hit short-circuit so lowering the threshold takes
    # effect on the next export.
    min_particles = get_config().exporter_common.synthesis_min_particles
    if len(particles) < min_particles:
        suffix = "" if len(particles) == 1 else "s"
        log.info(
            "%s — skipping synthesis (%d particle%s, threshold %d)",
            progress_prefix,
            len(particles),
            suffix,
            min_particles,
        )
        return _stamp_article_synthesis_state(
            page, "skipped-low-coverage"
        ), _SynthesisOutcome.SKIPPED

    if not regenerate:
        log.info("%s — article cache hit (on-disk frontmatter)", progress_prefix)
        # The on-disk hash matched — re-emit the existing
        # rendered-artefact stamp on the H1 block. The DB cache
        # is still consulted by render_article on the next render
        # if something else triggers it.
        return _stamp_article_input_hash(page, article_hash), _SynthesisOutcome.CACHE_HIT

    log.info("%s — synthesising article (%d particles)…", progress_prefix, len(particles))

    body, used_synthesis = await render_article(
        subject=subject,
        particles=particles,
        eff=eff_conf,
        input_hash=article_hash,
        corpus_uris=entry_uri_map,
        max_tokens=cfg.max_tokens,
        layer_b_enabled=cfg.layer_b_enabled,
        lint_findings=lint_findings or None,
        session=session,
    )

    if not used_synthesis:
        # Fallback: the existing outline already lists every particle.
        # Don't double up; just stamp the hash and emit a marker so the
        # operator knows synthesis didn't run.
        log.info("%s — article fell back to structured-listing", progress_prefix)
        page = _stamp_article_input_hash(page, article_hash)
        page = _stamp_article_synthesis_state(page, "structured-listing")
        return page, _SynthesisOutcome.FALLBACK

    # Successful synthesis (or cache hit at the DB layer — render_article
    # returns the cached body verbatim in both cases). Pull the prose +
    # references out and wrap them as a Logseq ``- ## Synthesis`` block.
    _frontmatter, _h1, prose, references = split_rendered_article(body)
    synthesis_block = _format_synthesis_block(prose, references)

    page = _stamp_article_input_hash(page, article_hash)
    page = _stamp_article_synthesis_state(page, "llm")
    # Splice the synthesis block right after the H1 block, before any
    # ``## Properties`` / ``## Description`` sections.
    page = _insert_after_h1(page, synthesis_block)
    return page, _SynthesisOutcome.SYNTHESISED


# ---------------------------------------------------------------------------
# Block-wrapping helpers
# ---------------------------------------------------------------------------


def _format_synthesis_block(
    prose: str, references: str, *, heading: str | None = "## Synthesis"
) -> str:
    """Wrap LLM prose + References into Logseq blocks.

    The prose body becomes ONE block (multi-line content within a
    Logseq block is fine as long as continuation lines are indented
    one level past the bullet). The References section becomes a
    nested ``- ### References`` parent block whose children are one
    block per reference line.

    ``heading`` is the parent block the prose nests under — the default
    ``## Synthesis`` marks LLM prose spliced *into* a structural subject page.
    Pass ``None`` for a standalone page whose whole body is the article (the narrative pages), which emits the same blocks one level shallower.
    """
    depth = 1 if heading is not None else 0
    indent = _INDENT * depth
    lines: list[str] = []
    if heading is not None:
        lines.append(f"- {heading}")
    if prose:
        # Logseq supports multi-line block content — continuation
        # lines indent one level past the bullet (depth 1 + one
        # space). Use four spaces (two for depth-1 indent + two for
        # continuation) on each prose-body line after the first.
        prose_lines = prose.splitlines()
        if prose_lines:
            lines.append(f"{indent}- {prose_lines[0]}")
            for cont in prose_lines[1:]:
                lines.append(f"{indent}  {cont}")

    if references:
        lines.append(f"{indent}- ### References")
        for ref_line in references.splitlines():
            line = ref_line.strip()
            if not line or line.startswith("## References") or line.startswith("##References"):
                continue
            lines.append(f"{_INDENT * (depth + 1)}- {line}")

    return "\n".join(lines) + "\n"


def _stamp_article_input_hash(page: str, article_hash: str) -> str:
    """Add or replace the ``article_input_hash::`` property on the H1 block.

    Idempotent: a prior stamp gets replaced, no stamp gets added on a
    new continuation line one indent past the H1.
    """
    return _stamp_property(page, "article_input_hash", article_hash)


def _stamp_article_synthesis_state(page: str, state: str) -> str:
    return _stamp_property(page, "article_synthesis", state)


def _stamp_property(page: str, key: str, value: str) -> str:
    """Idempotently set ``key:: value`` on the H1 block of a Logseq page.

    The H1 is the first line of the page (``- # <title>``). Properties
    live on continuation lines indented one level past it (the
    ``_INDENT`` two-space convention). We scan the lines after the H1
    until either:

    * we find an existing ``<key>:: …`` line (replace in place); or
    * we hit a non-property line (insert immediately before it).
    """
    lines = page.splitlines(keepends=True)
    if not lines:
        return page

    prefix = f"{_INDENT}{key}:: "
    h1_idx = next((i for i, ln in enumerate(lines) if ln.startswith("- # ")), None)
    if h1_idx is None:
        return page

    # Look at continuation lines (those starting with two-space indent).
    insert_idx = h1_idx + 1
    while insert_idx < len(lines):
        candidate = lines[insert_idx]
        if not candidate.startswith(_INDENT):
            break
        if candidate.startswith(prefix):
            lines[insert_idx] = f"{prefix}{value}\n"
            return "".join(lines)
        insert_idx += 1

    lines.insert(insert_idx, f"{prefix}{value}\n")
    return "".join(lines)


def _insert_after_h1(page: str, block: str) -> str:
    """Insert ``block`` after the H1 + its continuation properties.

    Finds the H1 line, walks past its continuation property lines,
    then injects ``block`` (which must already end with a newline).
    """
    lines = page.splitlines(keepends=True)
    h1_idx = next((i for i, ln in enumerate(lines) if ln.startswith("- # ")), None)
    if h1_idx is None:
        return block + page

    insert_idx = h1_idx + 1
    while insert_idx < len(lines) and lines[insert_idx].startswith(_INDENT):
        insert_idx += 1
    lines.insert(insert_idx, block)
    return "".join(lines)
