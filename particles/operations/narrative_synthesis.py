"""Narrative-aware synthesis rendering.

The first *consumer* of the narrative graph: given a NARRATIVE
particle, render its ``SEQUENCE_IN`` constituents as one cited prose narrative,
reusing the ``article_synthesis`` engine (Layer A citation check, Layer B
semantic-alignment judge, structured-listing fallback, shared cache).

The adapter is thin: the ordered constituents become the engine's
particle set, the NARRATIVE's label becomes the article title (carried on a
synthetic :class:`Subject` whose ``id`` is the narrative id, so narrative
articles cache under a key that never collides with a per-subject one), and the
engine is asked for the sequence-aware prompt. Citations stay by particle id, so
Layers A/B are unchanged.

Engine layer: traverses the relation graph, reads the store, and
calls the LLM — the same footing as per-subject synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import Particle, ParticleType, Subject
from particles.core.scoring.confidence import compute_effective_confidence
from particles.operations.narrative import get_narrative_sequence
from particles.render.article_synthesis import compute_input_hash, render_article


@dataclass
class NarrativeArticle:
    """The rendered narrative plus the inputs it was built from.

    ``constituents`` is the ``SEQUENCE_IN`` order; an empty list means the
    NARRATIVE has no ``PART_OF`` children yet (``body`` is then ``""``).
    ``used_synthesis`` is True when the LLM produced a body that passed Layer A
    (and, if enabled, Layer B); False means the deterministic structured-listing
    fallback was used (no key, LLM error, or a validation failure).
    """

    narrative: Particle
    constituents: list[Particle]
    body: str
    used_synthesis: bool


def _narrative_as_subject(narrative: Particle) -> Subject:
    """Adapt a NARRATIVE particle to the ``Subject`` shape the engine expects.

    The label becomes the title; the narrative id becomes the synthesis-cache
    key (``subject.id``) — a distinct id space from real Subjects, so a narrative
    article never collides with a per-subject one.
    """
    return Subject(
        id=narrative.id,
        canonical_name=narrative.content or "Narrative",
        asserted_by=narrative.asserted_by,
    )


async def synthesize_narrative(
    session: AsyncSession,
    narrative_id: str,
    *,
    without_synthesis: bool = False,
) -> NarrativeArticle | None:
    """Render a NARRATIVE's constituents as one cited prose narrative.

    Returns ``None`` when ``narrative_id`` is absent or not a NARRATIVE. When the
    narrative has no constituents yet, returns a :class:`NarrativeArticle` with an
    empty ``constituents`` list and empty ``body`` so the caller can message it.
    ``without_synthesis=True`` forces the deterministic, key-free structured
    listing (no LLM, no cache).
    """
    # Deferred import (cycle-free but keeps the store dep out of module import
    # for the test-mock seam, tests/AGENTS.md § Mocking strategy).
    from particles.corpus.store import get_entry_uri_map
    from particles.store.particle_store import get_particle

    narrative = await get_particle(session, narrative_id)
    if narrative is None or narrative.particle_type != ParticleType.NARRATIVE:
        return None

    constituents = await get_narrative_sequence(session, narrative_id)
    if not constituents:
        return NarrativeArticle(narrative=narrative, constituents=[], body="", used_synthesis=False)

    # Effective confidence with neutral trust weighting — the per-source trust /
    # recency refinements the wiki exporter applies are deferred for this first
    # slice; confidence.value is already calibrated at creation.
    eff = {
        p.id: compute_effective_confidence(
            p.confidence.value, calibration_source=p.confidence.calibration_source
        )
        for p in constituents
    }
    needed_entry_ids = {
        ref.corpus_entry_id for p in constituents for ref in p.provenance if ref.corpus_entry_id
    }
    corpus_uris = await get_entry_uri_map(session, needed_entry_ids)

    synthetic = _narrative_as_subject(narrative)
    input_hash = compute_input_hash(constituents, synthetic, ordered=True)
    wiki_cfg = get_config().wiki
    body, used_synthesis = await render_article(
        subject=synthetic,
        particles=constituents,
        eff=eff,
        input_hash=input_hash,
        corpus_uris=corpus_uris,
        max_tokens=wiki_cfg.max_tokens,
        layer_b_enabled=wiki_cfg.layer_b_enabled,
        session=session,
        sequence_mode=True,
        without_synthesis=without_synthesis,
    )
    return NarrativeArticle(
        narrative=narrative,
        constituents=constituents,
        body=body,
        used_synthesis=used_synthesis,
    )
