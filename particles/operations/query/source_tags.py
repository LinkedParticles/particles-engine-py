# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Which beliefs only sources of an excluded kind hold.

The agent-memory region drops a candidate held only by ``rule-file`` sources,
the operating documents the harness serves from the files themselves. "Held"
is read through the observer-scope join (``store.observer_scope_join``),
so the region and the observer lens agree on which entries
state a belief:

- a belief whose **current** attesting entries are all excluded is dropped;
- a belief no entry currently states is judged by the entries that **ever**
  did, so one a rule file stopped stating is dropped too, on a store the
  observer lens has never engaged on;
- a belief with no attesting entry at all is kept;
- a derived belief is dropped only when every premise is.

Gather / decide: :func:`excluded_by_source_tags` does the one read,
:func:`held_only_by` and :func:`excluded_ids` decide over plain values.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle
from particles.store.observer_scope_join import (
    Attestation,
    AttestationGraph,
    load_attestations,
)

#: How deep a chain of derived-from-derived beliefs is followed; past this a
#: belief is kept, which is the side this filter errs to.
_MAX_DERIVATION_DEPTH = 8


def held_only_by(attestation: Attestation, excluded_tags: Collection[str]) -> bool:
    """Whether every entry that states this belief carries one of ``excluded_tags``.

    The current entries decide when there are any; otherwise the entries that
    ever stated it do. No entry at all is never "held only by" anything.
    """
    entries = attestation.now or attestation.ever
    if not entries:
        return False
    excluded = set(excluded_tags)
    return all(excluded.intersection(attestation.tags.get(e, ())) for e in entries)


def excluded_ids(
    graph: AttestationGraph, ids: Sequence[str], excluded_tags: Collection[str]
) -> frozenset[str]:
    """The ids, of ``ids``, held only by sources tagged with one of ``excluded_tags``."""
    if not excluded_tags:
        return frozenset()
    memo: dict[str, bool] = {}

    def excluded(pid: str, trail: frozenset[str]) -> bool:
        if pid in memo:
            return memo[pid]
        if pid in graph.sourced:
            result = held_only_by(graph.sourced[pid], excluded_tags)
        else:
            premises = graph.premises.get(pid)
            if not premises or pid in trail or len(trail) >= _MAX_DERIVATION_DEPTH:
                return False
            result = all(excluded(q, trail | {pid}) for q in premises)
        memo[pid] = result
        return result

    return frozenset(pid for pid in ids if excluded(pid, frozenset()))


async def excluded_by_source_tags(
    session: AsyncSession, particles: Sequence[Particle], excluded_tags: Collection[str]
) -> frozenset[str]:
    """Read the attesting entries of ``particles`` and return the ids to drop."""
    if not excluded_tags or not particles:
        return frozenset()
    graph = await load_attestations(session, particles)
    return excluded_ids(graph, [p.id for p in particles], excluded_tags)
