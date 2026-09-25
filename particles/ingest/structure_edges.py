# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plan the extract pass's stance and narrative edges (D2).

Once the write loop has landed every candidate, the pipeline knows which stored
particle id represents each candidate index, and two sets of edges follow from
that map alone:

* **Stance edges**. Each stance binds to the sibling claim it
  endorses or disputes. If either endpoint was dropped (conflict-superseded),
  or the target resolves to the stance itself, the edge is skipped: an unbound
  stance has nothing to aggregate.
* **The narrative graph**. With exactly one landed NARRATIVE, every
  landed constituent links to it by PART_OF (constituent → narrative), and
  consecutive constituents by SEQUENCE_IN (predecessor → successor), in
  ``narrative_index`` order. A conflict-dropped constituent is simply absent
  from the chain. A constituent suppressed into the container is skipped, and
  one suppressed into its predecessor gets no SEQUENCE_IN self-edge.

:func:`plan_structure_edges` is the pure decide step over plain values; the
pipeline applies its result as one loop of ``create_relation`` calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from particles.core.schema import ParticleType, RelationType
from particles.extraction.general import CandidateParticle


def plan_structure_edges(
    candidates: Sequence[CandidateParticle],
    index_to_id: Mapping[int, str],
    stance_specs: Sequence[tuple[int, int, RelationType]],
) -> list[tuple[str, str, RelationType]]:
    """Return the stance and narrative edges to write, in write order.

    Args:
        candidates: The extract pass's candidates, in extraction order.
        index_to_id: Candidate index → the stored particle id that represents
            it (minted, suppressed into a twin, or resolved). A missing index
            is a candidate that did not land.
        stance_specs: ``(stance_index, target_index, kind)`` per stance
            candidate, in extraction order.

    Returns:
        ``(particle_a, particle_b, relation_type)`` triples: every stance edge
        first, in ``stance_specs`` order, then the narrative graph.
    """
    edges: list[tuple[str, str, RelationType]] = []

    for stance_idx, target_idx, kind in stance_specs:
        s_id = index_to_id.get(stance_idx)
        t_id = index_to_id.get(target_idx)
        if s_id is not None and t_id is not None and s_id != t_id:
            edges.append((s_id, t_id, kind))

    narrative_ids = [
        index_to_id[i]
        for i, c in enumerate(candidates)
        if c.particle_type == ParticleType.NARRATIVE and i in index_to_id
    ]
    if len(narrative_ids) != 1:
        return edges
    narrative_id = narrative_ids[0]

    constituents: list[tuple[int, int]] = []
    for i, c in enumerate(candidates):
        ni = c.narrative_index
        if ni is None or c.particle_type == ParticleType.NARRATIVE or i not in index_to_id:
            continue
        constituents.append((ni, i))
    constituents.sort()

    prev_id: str | None = None
    for _ni, i in constituents:
        cid = index_to_id[i]
        if cid == narrative_id:
            continue
        edges.append((cid, narrative_id, RelationType.PART_OF))
        if prev_id is not None and prev_id != cid:
            edges.append((prev_id, cid, RelationType.SEQUENCE_IN))
        prev_id = cid
    return edges
