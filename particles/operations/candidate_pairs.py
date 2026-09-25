# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Similar-particle candidate pairs: the pure decide step two passes share.

The contradiction lint (``lint/contradictions.py``, L-SEM-01) and ``links
suggest`` (``links_suggest.py``) both look for near-neighbour pairs
under the same candidacy rules, then do different things with them: the lint
probes each pair for a contradiction, the suggester proposes it as a
co-evidential link. This module holds those rules once, as a function over
plain values, so the caller gathers (candidates, co-evidential components,
recorded pairs) and this decides (D2).

A pair is a candidate when:

* its cosine similarity is at or above the threshold (scale);
* at least one side is in scope, when a scope is given;
* it is not already linked CO_EVIDENTIAL, transitively (§6.10);
* it is not in the caller's ``exclude`` set;
* both sides have the same stance holder.

Per-particle eligibility (truth-apt, asserted) is
:func:`is_pair_eligible`, which each caller applies while gathering, because
the lint narrows it further (DOCUMENT_META) and ``links suggest``
groups the survivors by Subject first. The similarity pass needs numpy, which
``core/`` does not import, so this lives beside the operations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any

import numpy as np

from particles.core.schema import Particle, is_truth_apt
from particles.core.stance import stance_holder
from particles.extraction.polarity import is_non_asserted
from particles.operations._scope import pair_scope_tier


@dataclass(frozen=True)
class CandidatePair:
    """One similar-particle pair that passed every candidacy rule.

    ``a`` precedes ``b`` in the caller's candidate sequence.
    """

    #: Scope tier: 0 both sides in scope (or no scope), 1 one side.
    tier: int
    #: Cosine similarity on the ``[0, 1]`` scale.
    similarity: float
    a: Particle
    b: Particle


def is_pair_eligible(p: Particle) -> bool:
    """True when ``p`` may appear in a candidate pair at all.

    Only truth-apt particles share a truth that can be corroborated or
    contradicted, and non-asserted prose (a rejected, deferred or
    counterfactual claim) is off the factual surface.
    """
    return is_truth_apt(p) and not is_non_asserted(p.properties)


def enumerate_candidate_pairs(
    candidates: Sequence[tuple[Particle, np.ndarray[Any, np.dtype[np.float32]]]],
    *,
    threshold: float,
    linked: Mapping[str, AbstractSet[str]],
    exclude: AbstractSet[frozenset[str]] = frozenset(),
    scope: frozenset[str] | None = None,
) -> list[CandidatePair]:
    """Every candidate pair among ``candidates``, in probe order.

    Args:
        candidates: ``(particle, embedding)`` pairs, already filtered for
            per-particle eligibility.
        threshold: Similarity floor, inclusive.
        linked: Particle id → its CO_EVIDENTIAL component, as
            :func:`~particles.core.equivalence.co_evidential_components`
            returns. An id that is absent is a singleton.
        exclude: Unordered id pairs to drop, e.g. contradictions already
            recorded.
        scope: Harvest scope. When set, a pair needs at least one
            side in it, and the tier records how many.

    Returns:
        The pairs ordered by tier, then similarity (highest first), then the two
        ids as a deterministic tie-break. With no scope every tier is
        0, so the order is pure similarity.
    """
    if len(candidates) < 2:
        return []

    # Row-normalise once so cosine reduces to a dot product. The stored vectors
    # are already unit-norm; normalise defensively, mirroring ``_find_conflict``.
    emb_matrix: np.ndarray[Any, np.dtype[np.float32]] = np.asarray(
        [emb for _, emb in candidates], dtype=np.float32
    )
    emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10)

    pairs: list[CandidatePair] = []
    n = len(candidates)
    for i in range(n):
        p_a = candidates[i][0]
        component = linked.get(p_a.id, frozenset())
        # Cosine of i against every later particle in one vectorized product;
        # the upper triangle (j > i) visits each unordered pair exactly once.
        # Negatives clamp to 0 so this stays on the normative [0, 1] scale
        # (the vectorized analogue of embeddings.cosine_similarity).
        sims = np.clip(emb_matrix[i + 1 :] @ emb_matrix[i], 0.0, 1.0)
        for offset in np.flatnonzero(sims >= threshold).tolist():
            p_b = candidates[i + 1 + offset][0]

            tier = pair_scope_tier(scope, p_a.id, p_b.id)
            if tier > 1:
                continue
            # A particle's co-evidential group always includes itself.
            if p_b.id == p_a.id or p_b.id in component:
                continue
            if frozenset((p_a.id, p_b.id)) in exclude:
                continue
            # A stance pairs only with a same-holder stance, never with its
            # target or another holder's stance. Two non-stances
            # both read None and pass through.
            if stance_holder(p_a) != stance_holder(p_b):
                continue

            pairs.append(CandidatePair(tier, float(sims[offset]), p_a, p_b))

    pairs.sort(key=lambda c: (c.tier, -c.similarity, c.a.id, c.b.id))
    return pairs
