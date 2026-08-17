"""Narrative traversal operations.

A NARRATIVE particle is prose-level structural connective tissue: its
``content`` is a one-sentence label, and the narrative it names exists only
as the subgraph induced by ``PART_OF`` / ``SEQUENCE_IN`` edges in the
relation store. No "chain" object is persisted — these three read helpers
*are* the narrative surface, reconstructing constituents and their order from
the edges on demand.

Both kinds are asymmetric: ``PART_OF`` runs constituent →
narrative, ``SEQUENCE_IN`` runs predecessor → successor. For v1 the sequence
is **linear** — every constituent has at most one predecessor and one
successor (DAG semantics are deferred). Lives in the operation layer so every
front-end (CLI / API / MCP) gets identical traversal. Caller owns the session.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle, RelationType
from particles.store.particle_store import get_particle
from particles.store.relation_store import get_incoming, get_outgoing


async def _fetch_particles(session: AsyncSession, ids: list[str]) -> list[Particle]:
    """Resolve ids to Particles, skipping any that no longer exist.

    The relation store holds no FK to particles (relation_store §docstring), so
    an edge can outlive its endpoint; a dangling constituent is dropped here
    rather than surfaced — lint owns that diagnostic, not traversal.
    """
    particles: list[Particle] = []
    for pid in ids:
        particle = await get_particle(session, pid)
        if particle is not None:
            particles.append(particle)
    return particles


async def get_narrative_constituents(session: AsyncSession, narrative_id: str) -> list[Particle]:
    """Return the particles that are ``PART_OF`` the narrative.

    These are the narrative's members in no particular order (use
    :func:`get_narrative_sequence` for the SEQUENCE_IN ordering). Ordered by
    particle id for deterministic output.
    """
    constituent_ids = await get_incoming(session, narrative_id, RelationType.PART_OF)
    return await _fetch_particles(session, constituent_ids)


async def get_narrative_sequence(session: AsyncSession, narrative_id: str) -> list[Particle]:
    """Return the narrative's constituents in ``SEQUENCE_IN`` order.

    Walks the linear ``SEQUENCE_IN`` chain over the constituent set: the head
    (a constituent with no in-set predecessor) followed by successors. v1
    assumes a linear chain — at most one predecessor / successor each.
    Constituents not wired into the chain (isolated nodes) are appended after
    it, ordered by id. Cycles — which the linear assumption forbids — are
    broken defensively by a visited guard so this never loops.
    """
    constituent_ids = await get_incoming(session, narrative_id, RelationType.PART_OF)
    in_set = set(constituent_ids)
    if not in_set:
        return []

    # Build the predecessor → successor map, restricted to edges whose *both*
    # endpoints are constituents of this narrative (a SEQUENCE_IN edge to some
    # particle outside the narrative is not part of this narrative's order).
    successor: dict[str, str] = {}
    has_predecessor: set[str] = set()
    for cid in constituent_ids:
        for succ in await get_outgoing(session, cid, RelationType.SEQUENCE_IN):
            if succ in in_set:
                successor[cid] = succ
                has_predecessor.add(succ)

    # A chain head has no in-set predecessor but does have a successor — it
    # starts an ordered run. Walk each (sorted for determinism; the
    # well-formed linear case has exactly one). Isolated constituents (no
    # predecessor *and* no successor) are deliberately excluded here so they
    # land after the ordered runs, not interleaved by id among them.
    chain_heads = sorted(
        cid for cid in constituent_ids if cid not in has_predecessor and cid in successor
    )
    ordered_ids: list[str] = []
    visited: set[str] = set()
    for head in chain_heads:
        current: str | None = head
        while current is not None and current not in visited:
            visited.add(current)
            ordered_ids.append(current)
            current = successor.get(current)

    # Anything still unvisited is appended in id order: isolated constituents
    # (the common case) plus, defensively, any node only reachable via a cycle
    # the linear assumption forbids — so none is ever dropped.
    ordered_ids.extend(sorted(cid for cid in constituent_ids if cid not in visited))

    return await _fetch_particles(session, ordered_ids)


async def get_narratives_containing(session: AsyncSession, particle_id: str) -> list[Particle]:
    """Return the NARRATIVE particles this particle is ``PART_OF``.

    A particle may belong to several narratives, so this returns
    a list. Ordered by particle id for deterministic output.
    """
    narrative_ids = await get_outgoing(session, particle_id, RelationType.PART_OF)
    return await _fetch_particles(session, narrative_ids)
