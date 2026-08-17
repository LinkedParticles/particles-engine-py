"""Semantic ranking, effective-confidence math, and co-evidential collapse.

Helpers used by ``main.query``:

  - ``_embed`` — encode the question to a single embedding vector (or ``None``
    when no embedding model is loaded; the caller falls back to confidence
    ranking).
  - ``_collapse_co_evidential_top_k`` — within the top-k retrieval result,
    group CO_EVIDENTIAL particles (§6.10), keep only the
    highest-scored representative per cluster, and replace its
    effective_confidence with the noisy-OR merge (§6.9).
  - ``_first_source_key`` — corpus_entry_id of a particle's first SOURCE
    provenance ref; used as the ``source_independence`` key for noisy-OR.

The combined-score weighting (``config.query.similarity_weight`` × cosine_sim
+ ``config.query.confidence_weight`` × effective_confidence) and the
truncation-warning heuristic live inline in ``main.query`` because they only
fire once per request.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle, ProvenanceRefType
from particles.core.scoring.confidence import merge_co_evidential_confidence

log = logging.getLogger(__name__)

#: The disclosure for a query that ran with no embedding model.
#:
#: Without a query vector ``main._gather_scored`` aliases similarity to
#: effective confidence, so the top-k is the store's most-confident beliefs
#: rather than the ones about the question — and ``_relevance_note`` returns
#: ``None``, which switches off the very floor that exists to catch
#: "the nearest belief is not about this question". Both halves are named here
#: because the second is the reason the first cannot be caught downstream.
RANKING_DEGRADED_NO_ENCODER = (
    "Semantic ranking unavailable: no embedding model could be loaded. These "
    "beliefs are the store's most confident ones, NOT necessarily ones about "
    "your question, and the relevance floor that would normally refuse an "
    "off-topic answer is inactive. Install the embedding model "
    "(sentence-transformers) and re-run to get a real answer."
)


def _embed(text: str) -> np.ndarray[Any, np.dtype[np.float32]] | None:
    """Encode the question, or ``None`` when no encoder is available.

    ``None`` is not a neutral outcome — see :data:`RANKING_DEGRADED_NO_ENCODER`
    for what the caller silently becomes without it — so it is logged at
    WARNING rather than left to the one-time load-failure line in
    ``particles.embeddings``, which a long-running process emits once and never
    repeats.
    """
    from particles.embeddings import get_embedding_model

    model = get_embedding_model()
    if model is None:
        log.warning(
            "Query ran without an embedding model: ranking falls back to effective "
            "confidence and the relevance floor is inactive."
        )
        return None
    result = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    arr: np.ndarray[Any, np.dtype[np.float32]] = result[0]
    return arr


async def _collapse_co_evidential_top_k(
    session: AsyncSession,
    top: list[tuple[Particle, float, float]],
) -> list[tuple[Particle, float, float]]:
    """Collapse CO_EVIDENTIAL groups within the top-k retrieval result.

    Within each CO_EVIDENTIAL cluster, the highest-scored particle (first in
    rank order, since ``top`` arrives sorted) is the representative. Other
    members of the cluster are dropped from the rendered result. The
    representative's effective_confidence is replaced with the noisy-OR
    merge (§6.9) over all top-k cluster members, using the corpus_entry_id
    of the first SOURCE provenance ref as the source_independence key so
    multiple particles from the same corpus entry are discounted.

    Particles with no co-evidential links pass through unchanged.

    Implementation note: only top-k members of a cluster participate in the
    merge. A full-group merge (including members outside top-k) is the
    wiki exporter's job where the cost of loading non-retrieved
    particles is amortised across a per-Subject article render.
    """
    from particles.config import get_config
    from particles.store.relation_store import get_co_evidential_group

    if not top:
        return top

    # gate collapse on effective_equivalence >= threshold (default 0.0
    # reproduces pre-0106 behaviour). Read at call time per the config rule.
    min_equivalence = get_config().query.equivalence_threshold

    top_id_to_index = {p.id: i for i, (p, _, _) in enumerate(top)}
    cluster_for: dict[str, frozenset[str]] = {}
    clusters_seen: set[frozenset[str]] = set()
    collapsed: list[tuple[Particle, float, float]] = []

    for particle, sim, eff_conf in top:
        if particle.id in cluster_for:
            # This particle is already represented by an earlier cluster's representative.
            continue

        # Compute the CO_EVIDENTIAL closure for this particle and intersect with top-k.
        full_group = await get_co_evidential_group(
            session, particle.id, min_confidence=min_equivalence
        )
        in_top_k = frozenset(full_group & set(top_id_to_index))

        if len(in_top_k) <= 1:
            # No collapse needed — either singleton group or only this particle is in top-k.
            collapsed.append((particle, sim, eff_conf))
            for pid in in_top_k:
                cluster_for[pid] = in_top_k
            continue

        # Multiple top-k members: this particle (the highest-ranked, since we iterate
        # top in sort order) is the representative. Mark all members as represented.
        for pid in in_top_k:
            cluster_for[pid] = in_top_k

        if in_top_k in clusters_seen:
            continue
        clusters_seen.add(in_top_k)

        # Build noisy-OR merge inputs from every top-k member of the cluster,
        # iterating in rank order (frozenset iteration order varies across
        # processes; the merge itself is order-independent — see
        # merge_co_evidential_confidence — but a deterministic input keeps the
        # whole path reproducible).
        merge_entries: list[tuple[float, str]] = []
        for pid in sorted(in_top_k, key=lambda i: top_id_to_index[i]):
            member_p, _member_sim, member_ec = top[top_id_to_index[pid]]
            source_key = _first_source_key(member_p)
            merge_entries.append((member_ec, source_key))

        merged_conf = merge_co_evidential_confidence(merge_entries)
        collapsed.append((particle, sim, merged_conf))

    return collapsed


def _first_source_key(p: Particle) -> str:
    """Return the corpus_entry_id of the first SOURCE provenance ref, or particle.id.

    Used as the source_independence key for co-evidential merge so multiple
    particles from the same source are discounted by 1/k.
    """
    for ref in p.provenance:
        if ref.type == ProvenanceRefType.SOURCE and ref.corpus_entry_id:
            return ref.corpus_entry_id
    return p.id
