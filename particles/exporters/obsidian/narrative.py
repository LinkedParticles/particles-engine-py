"""per-NARRATIVE Obsidian note rendering.

A NARRATIVE has no subject note to splice into (per-subject synthesis is in
:mod:`particles.exporters.obsidian.synthesis`), so its rendered prose becomes a
*standalone* note under ``Narratives/``. This module builds that note, reusing
the render path (``render_article(sequence_mode=True)``) and the
Obsidian block-ref conversion (:func:`_to_obsidian_block_refs`).

The exporter passes its own trust-weighted ``eff_conf`` / ``subject_map`` /
``naming`` (the same dicts the per-subject notes use), so a narrative note's
confidence numbers and ``[[Subject]]`` wikilinks match the rest of the vault.
The synthesis cache (keyed by narrative id + ordered constituents) is consulted via ``session``, so re-exports and prior
``particle narrative --synthesize`` runs avoid re-paying the LLM.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from particles.core.schema import Particle, Subject
from particles.exporters.obsidian.format import (
    _annotate_obsidian_frontmatter,
    _to_obsidian_block_refs,
)
from particles.render.article_synthesis import render_article, split_rendered_article
from particles.render.markdown import SubjectNaming, narrative_as_subject

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

# ``narrative_as_subject`` moved to ``particles.render.markdown`` alongside
# ``build_narrative_naming`` when a change gave the Logseq and Wiki exporters
# per-narrative notes too. Re-exported here so existing call sites keep working.
__all__ = ["narrative_as_subject", "render_narrative_note"]


def _build_narrative_note(*, label: str, prose: str, references: str) -> str:
    """Assemble a standalone narrative note: frontmatter + H1 + prose + refs."""
    parts = ["---", "tags:", "  - particles/narrative", "---", "", f"# {label}", "", prose.rstrip()]
    if references.strip():
        parts += ["", references.rstrip()]
    return "\n".join(parts) + "\n"


async def render_narrative_note(
    *,
    narrative: Particle,
    constituents: list[Particle],
    subject_map: dict[str, Subject],
    eligible_ids: set[str],
    eff_conf: dict[str, float],
    entry_uri_map: dict[str, str | None],
    naming: SubjectNaming,
    article_hash: str,
    session: AsyncSession,
) -> tuple[str, str]:
    """Render one NARRATIVE as a standalone Obsidian note.

    Returns ``(note, state)`` where ``state`` is ``"synthesised"`` (LLM body
    passed validation) or ``"fallback"`` (deterministic structured listing — no
    key, LLM error, or a validation failure). The render path and cache are
    shared; only the standalone-note assembly + Obsidian block-refs are new.
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
    # Discard the engine's subject-flavoured frontmatter + footnote references;
    # keep the prose and re-render Obsidian-native block refs from the
    # constituents (same conversion the per-subject splice uses).
    _fm, _h1, prose, _refs = split_rendered_article(body)
    prose, references = _to_obsidian_block_refs(
        prose,
        particles=constituents,
        parent_subject=synthetic,
        subject_map=subject_map,
        eligible_ids=eligible_ids,
        eff_conf=eff_conf,
        entry_uri_map=entry_uri_map,
        naming=naming,
    )
    note = _build_narrative_note(
        label=narrative.content or "Narrative", prose=prose, references=references
    )
    note = _annotate_obsidian_frontmatter(
        note,
        article_input_hash=article_hash,
        article_synthesis="llm" if used_synthesis else "structured-listing",
    )
    return note, ("synthesised" if used_synthesis else "fallback")
