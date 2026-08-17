"""Per-NARRATIVE Logseq page rendering (mechanism).

A NARRATIVE has no subject page to splice into (per-subject synthesis is in
:mod:`particles.exporters.logseq.synthesis`), so its rendered prose becomes a
*standalone* page in Logseq's ``Narratives/`` page namespace — on disk
``pages/Narratives___<slug>.md``, since ``logseq_slug`` maps ``/`` to Logseq's
``___`` hierarchy separator. That is the same ``Narratives/`` namespace ADR
0130 §3 reserved for the Obsidian vault, expressed in Logseq's flat-file
encoding of a hierarchical page name.

The render path (``sequence_mode``) and the synthesis cache
(keyed by narrative id + ordered constituents) are shared with the Obsidian and
Wiki narrative notes, so a narrative already rendered by another exporter — or
by ``particle narrative --synthesize`` — is a cache hit here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from particles.exporters.logseq.format import _INDENT  # noqa: PLC2701 — module-internal
from particles.exporters.logseq.synthesis import (  # noqa: PLC2701 — module-internal
    _format_synthesis_block,
    _stamp_article_input_hash,
    _stamp_article_synthesis_state,
)
from particles.render.article_synthesis import render_article, split_rendered_article
from particles.render.markdown import narrative_as_subject

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from particles.core.schema import Particle

log = logging.getLogger(__name__)

# Logseq's native tag property, mirroring the Obsidian note's
# ``tags: particles/narrative`` frontmatter so both graphs expose the same
# "this page is a narrative" facet.
_NARRATIVE_TAG = "particles/narrative"


async def render_narrative_page(
    *,
    narrative: Particle,
    constituents: list[Particle],
    entry_uri_map: dict[str, str | None],
    eff_conf: dict[str, float],
    article_hash: str,
    session: AsyncSession,
) -> tuple[str, str]:
    """Render one NARRATIVE as a standalone Logseq page.

    Returns ``(page, state)`` where ``state`` is ``"synthesised"`` (LLM body
    passed validation) or ``"fallback"`` (deterministic structured listing — no
    key, LLM error, or a validation failure). Only the page assembly is new;
    the prose comes from the shared render path.
    """
    from particles.config import get_config

    cfg = get_config().wiki
    synthetic = narrative_as_subject(narrative)
    body, used_synthesis = await render_article(
        subject=synthetic,
        particles=constituents,
        eff=eff_conf,
        input_hash=article_hash,
        corpus_uris=entry_uri_map,
        max_tokens=cfg.max_tokens,
        layer_b_enabled=cfg.layer_b_enabled,
        session=session,
        sequence_mode=True,
    )
    # Discard the engine's Markdown frontmatter + H1 — a Logseq page carries
    # its title as the first block and its metadata as block properties.
    _fm, _h1, prose, references = split_rendered_article(body)

    page = f"- # {narrative.content or 'Narrative'}\n{_INDENT}tags:: {_NARRATIVE_TAG}\n"
    page = _stamp_article_input_hash(page, article_hash)
    page = _stamp_article_synthesis_state(page, "llm" if used_synthesis else "structured-listing")
    page += _format_synthesis_block(prose, references, heading=None)
    return page, ("synthesised" if used_synthesis else "fallback")
