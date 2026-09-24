# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The observer-scope join — which projects currently observe a belief.

The pure algebra lives in :mod:`particles.core.observer_scope`. This module is
its store half: it joins a set of beliefs to the corpus entries that attest
them and reads those entries' ``project:`` tags. It sits in ``store`` because
two layers above it must agree on it exactly: the read path's lens
(``operations.query.observer_scope``) and the write path's reconciliation
precondition (``ingest.observer_gate``). One implementation, so the two can
never disagree about who observes a claim.

A belief's attesting entries are those named by its own ``SOURCE`` provenance
**and** by every particle it is ``CO_EVIDENTIAL`` with — the same group the
confidence merge already collapses, and the only path to a merged duplicate's
sources. A **derived** belief (any ``PARTICLE`` ref: an abstraction,
an ``INCONSISTENCY``) takes the meet of its premises instead.

**Observed means currently stated**. A ``MUTABLE`` entry attests
a belief only while the belief's provenance refs name the entry's latest
extracted snapshot — every re-observation (minted, suppressed, carried forward)
appends such a ref, so a claim the file stopped stating drops out
of that project's scope with nothing written. Other mutabilities have no
generations and every snapshot counts. The refs are read from the particle's
own provenance, never from the edge index, whose one row per entry names the
*first* snapshot and is never re-pointed. A belief that some entry once
attested and none now does is **lapsed** — in view for no project, and
disclosed under that name, distinct from an unattributed stamping gap.

Everything here is a read-time derivation over the substrate: nothing is
stored.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.observer_scope import (
    GLOBAL_SCOPE,
    LAPSED_SCOPE,
    UNATTRIBUTED_SCOPE,
    BeliefScope,
    meet,
    scope_of_entries,
)
from particles.core.schema import Particle, ProvenanceRefType, RelationType
from particles.corpus.store import EntryCurrency, get_entry_currency
from particles.store.event_store import OperatorEventRow, OperatorEventType
from particles.store.observer_scope_store import ScopeTarget, any_widening, widened_ids
from particles.store.particle_store import ParticleRow
from particles.store.relation_store import ParticleRelationRow

_IN_CHUNK = 500

#: How deep a chain of derived-from-derived beliefs is followed. Abstractions
#: of abstractions are rare and shallow; past this the belief is unattributed
#: (in view for no project), which is the fail-closed answer.
_MAX_DERIVATION_DEPTH = 8


async def lens_may_engage(session: AsyncSession) -> bool:
    """Whether this store has been rescoped — the marker a project observer waits for.

    Before ``particles memory rescope`` has run, a store's entries carry the
    project tags of whatever version harvested them (or none), and reading
    through them would silently empty a digest on upgrade. The write-path
    precondition waits for the same marker. One indexed lookup.
    """
    row = await session.execute(
        select(OperatorEventRow.event_id)
        .where(OperatorEventRow.event_type == OperatorEventType.OBSERVER_SCOPE_RESCOPED.value)
        .limit(1)
    )
    return row.first() is not None


@dataclass(frozen=True)
class SourceRef:
    """One non-derivation provenance ref: the entry it names and the snapshot, if any."""

    entry_id: str
    snapshot_id: str | None


async def load_scopes(
    session: AsyncSession,
    particles: Sequence[Particle],
    *,
    as_of: datetime | None = None,
    current_only: bool = True,
) -> dict[str, BeliefScope]:
    """Each belief's observer scope, keyed by particle id.

    ``as_of`` evaluates "latest extracted snapshot" at that instant,
    so an as-of read sees the scopes the store would have given then.
    ``current_only=False`` counts every snapshot an entry ever attested — the
    reading for a claim being written right now, whose own snapshot is not
    extracted yet.
    """
    known = {p.id: p for p in particles}
    premises: dict[str, list[str]] = {}
    sourced: set[str] = set()
    for p in particles:
        refs = [r.corpus_entry_id for r in p.provenance if r.type is ProvenanceRefType.PARTICLE]
        if refs:
            premises[p.id] = [r for r in refs if r]
        else:
            sourced.add(p.id)

    # Premises are particles too, and may themselves be derived: resolve the
    # whole derivation graph first, so leaves are scoped in one batch.
    frontier = {pid for refs in premises.values() for pid in refs} - premises.keys() - sourced
    for _ in range(_MAX_DERIVATION_DEPTH):
        if not frontier:
            break
        loaded = await _load_premise_refs(session, frontier)
        next_frontier: set[str] = set()
        for pid in frontier:
            premise_ids = loaded.get(pid)
            if premise_ids:
                premises[pid] = premise_ids
                next_frontier.update(premise_ids)
            else:
                sourced.add(pid)
        frontier = next_frontier - premises.keys() - sourced

    resolved: dict[str, BeliefScope] = await _scopes_of_sourced(
        session, sourced, known, as_of=as_of, current_only=current_only
    )

    def scope_of(pid: str, trail: frozenset[str]) -> BeliefScope:
        if pid in resolved:
            return resolved[pid]
        if pid not in premises or pid in trail or len(trail) >= _MAX_DERIVATION_DEPTH:
            return UNATTRIBUTED_SCOPE
        scope = meet(scope_of(q, trail | {pid}) for q in premises[pid])
        resolved[pid] = scope
        return scope

    return {p.id: scope_of(p.id, frozenset()) for p in particles}


def attests(ref: SourceRef, currency: EntryCurrency | None) -> bool:
    """Whether one provenance ref is a *current* statement by its entry.

    A ``MUTABLE`` entry attests only through its latest extracted snapshot; an
    entry of any other mutability, a ref with no snapshot, and an entry with no
    extracted snapshot yet (nothing newer can have replaced it) attest
    through every ref.
    """
    if currency is None or not currency.mutable or currency.latest_snapshot_id is None:
        return True
    return ref.snapshot_id is None or ref.snapshot_id == currency.latest_snapshot_id


async def _scopes_of_sourced(
    session: AsyncSession,
    ids: Collection[str],
    known: Mapping[str, Particle],
    *,
    as_of: datetime | None,
    current_only: bool,
) -> dict[str, BeliefScope]:
    if not ids:
        return {}
    groups = await _co_evidential_groups(session, ids)
    members = {m for group in groups.values() for m in group}
    refs_of = await _source_refs(session, members, known)
    all_entries = {r.entry_id for refs in refs_of.values() for r in refs}
    currency = await get_entry_currency(session, all_entries, as_of=as_of)
    widened_entries = (
        await widened_ids(session, ScopeTarget.CORPUS_ENTRY, all_entries)
        if await any_widening(session)
        else set()
    )
    harness_tags = get_config().observer_scope.harness_tags

    scopes: dict[str, BeliefScope] = {}
    for pid in ids:
        # An "entry id" with no corpus entry behind it is not a source at all.
        refs = [r for m in groups[pid] for r in refs_of.get(m, ()) if r.entry_id in currency]
        ever = {r.entry_id for r in refs}
        if ever & widened_entries:
            scopes[pid] = GLOBAL_SCOPE
            continue
        now = (
            {r.entry_id for r in refs if attests(r, currency[r.entry_id])} if current_only else ever
        )
        if ever and not now:
            scopes[pid] = LAPSED_SCOPE
            continue
        scopes[pid] = scope_of_entries((currency[e].tags for e in now), harness_tags)
    return scopes


async def _source_refs(
    session: AsyncSession, ids: Collection[str], known: Mapping[str, Particle]
) -> dict[str, list[SourceRef]]:
    """Every non-``PARTICLE`` provenance ref of each particle, from its own provenance.

    Particles already in hand are read directly; the rest (co-evidential
    partners) are loaded in batches.
    """
    refs: dict[str, list[SourceRef]] = {}
    missing: list[str] = []
    for pid in ids:
        particle = known.get(pid)
        if particle is None:
            missing.append(pid)
            continue
        refs[pid] = [
            SourceRef(r.corpus_entry_id, r.snapshot_id)
            for r in particle.provenance
            if r.type is not ProvenanceRefType.PARTICLE
        ]
    for start in range(0, len(missing), _IN_CHUNK):
        rows = await session.execute(
            select(ParticleRow.id, ParticleRow.provenance_json).where(
                ParticleRow.id.in_(missing[start : start + _IN_CHUNK])
            )
        )
        for particle_id, provenance_json in rows.all():
            refs[particle_id] = [
                SourceRef(ref["corpus_entry_id"], ref.get("snapshot_id"))
                for ref in json.loads(provenance_json or "[]")
                if ref.get("type") != ProvenanceRefType.PARTICLE.value
                and ref.get("corpus_entry_id")
            ]
    return refs


async def _co_evidential_groups(
    session: AsyncSession, ids: Collection[str]
) -> dict[str, frozenset[str]]:
    """Each id's ``CO_EVIDENTIAL`` component, expanded breadth-first in batches."""
    neighbours: dict[str, set[str]] = {}
    expanded: set[str] = set()
    frontier = set(ids)
    while frontier:
        batch = list(frontier)
        expanded |= frontier
        found: set[str] = set()
        for start in range(0, len(batch), _IN_CHUNK):
            chunk = batch[start : start + _IN_CHUNK]
            rows = await session.execute(
                select(ParticleRelationRow.particle_a, ParticleRelationRow.particle_b).where(
                    ParticleRelationRow.relation_type == RelationType.CO_EVIDENTIAL.value,
                    or_(
                        ParticleRelationRow.particle_a.in_(chunk),
                        ParticleRelationRow.particle_b.in_(chunk),
                    ),
                )
            )
            for a, b in rows.all():
                neighbours.setdefault(a, set()).add(b)
                neighbours.setdefault(b, set()).add(a)
                found.update((a, b))
        frontier = found - expanded

    groups: dict[str, frozenset[str]] = {}
    for pid in ids:
        if pid in groups:
            continue
        component = {pid}
        stack = [pid]
        while stack:
            for n in neighbours.get(stack.pop(), ()):
                if n not in component:
                    component.add(n)
                    stack.append(n)
        frozen = frozenset(component)
        for member in component:
            groups[member] = frozen
    return groups


async def _load_premise_refs(session: AsyncSession, ids: Collection[str]) -> dict[str, list[str]]:
    """``PARTICLE``-ref premises of particles not already in hand; ``[]`` for a sourced one."""
    batch = list(ids)
    refs: dict[str, list[str]] = {}
    for start in range(0, len(batch), _IN_CHUNK):
        rows = await session.execute(
            select(ParticleRow.id, ParticleRow.provenance_json).where(
                ParticleRow.id.in_(batch[start : start + _IN_CHUNK])
            )
        )
        for particle_id, provenance_json in rows.all():
            refs[particle_id] = [
                ref["corpus_entry_id"]
                for ref in json.loads(provenance_json or "[]")
                if ref.get("type") == ProvenanceRefType.PARTICLE.value
                and ref.get("corpus_entry_id")
            ]
    return refs
