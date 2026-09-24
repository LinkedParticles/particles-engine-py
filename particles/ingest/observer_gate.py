# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The observer precondition on the §6.6 ladder — one gate every route consults.

A claim is reconciled against another only when every project that currently
observes the existing claim also observes the update:
``scope(existing) ⊆ scope(candidate)``. The pair may have been found by the
same-entry 0.80 search, the subject-keyed cross-entry search, the
store-wide search of the assertion path, or the backlog sweep—
the verdict is a property of the pair, so each route asks this
gate and gets the same answer. The scopes come from the same join the read
path's lens uses (:mod:`particles.store.observer_scope_join`), so the lens and
the ladder can never disagree about who observes a claim.

**Engaged only on a rescoped store.** Before ``particles memory rescope`` has
run, entry keys may be stale per-worktree slugs, and the precondition would
read one project as many observers and decline its own updates. On such a
store the gate is open and reconciliation is exactly what it was.

A declined pair that the probe confirmed is recorded as a ``CONTRADICTS``
relation (``created_by = OBSERVER_DIVERGENCE``) and counted in the extraction's
quality notes — the disclosure that replaces a silent retirement.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.observer_scope import (
    BeliefScope,
    PairPrecondition,
    pair_precondition,
    scope_of_entries,
)
from particles.core.schema import Particle
from particles.corpus.store import get_tags_for_entries
from particles.store.observer_scope_join import lens_may_engage, load_scopes
from particles.store.relation_store import record_observer_divergence


def divergence_note(count: int) -> str:
    """The quality note an extraction carries when it left pairs standing."""
    return (
        f"OBSERVER_DIVERGENCE: {count} pair(s) left standing because another project "
        "observes the existing claim"
    )


@dataclass
class DivergenceTally:
    """What one or more passes did with pairs the precondition declined."""

    declined: int = 0
    """Probe-confirmed pairs the precondition declined to reconcile."""
    recorded: int = 0
    """…of those, the pairs joined by a ``CONTRADICTS`` relation afterwards. A
    declined pair goes unrecorded only when an end is no longer ``ACTIVE`` by
    the time the pass writes — the candidate itself was quarantined against a
    different claim, or a sibling retired the other."""


@dataclass
class ObserverGate:
    """The precondition for one write pass, with the scopes it has read memoised.

    Scopes are read before the pass writes anything and are not refreshed
    within it, as the §6.6 candidate set is not.
    """

    engaged: bool
    _scopes: dict[str, BeliefScope] = field(default_factory=dict)

    @classmethod
    async def open(cls, session: AsyncSession) -> ObserverGate:
        return cls(engaged=await lens_may_engage(session))

    async def entry_scope(self, session: AsyncSession, entry_id: str) -> BeliefScope:
        """The scope of a claim being extracted from ``entry_id`` right now."""
        tags = await get_tags_for_entries(session, [entry_id])
        return scope_of_entries(
            [tags[entry_id]] if entry_id in tags else [],
            get_config().observer_scope.harness_tags,
        )

    async def candidate_scope(self, session: AsyncSession, candidate: Particle) -> BeliefScope:
        """The scope of a claim being written now, from every source it names.

        Every ref counts, since the claim's own snapshot is being written and is
        not an extracted generation yet.
        """
        return (await load_scopes(session, [candidate], current_only=False))[candidate.id]

    async def prime(self, session: AsyncSession, particles: Iterable[Particle]) -> None:
        """Read the current scopes of ``particles`` in one batch."""
        missing = [p for p in particles if p.id not in self._scopes]
        if self.engaged and missing:
            self._scopes.update(await load_scopes(session, missing))

    async def scope_of(self, session: AsyncSession, existing: Particle) -> BeliefScope:
        await self.prime(session, [existing])
        return self._scopes[existing.id]

    async def verdict(
        self, session: AsyncSession, candidate: BeliefScope, existing: Particle
    ) -> PairPrecondition:
        """The precondition for (candidate, existing); ``RECONCILE`` when not engaged."""
        if not self.engaged:
            return PairPrecondition.RECONCILE
        return pair_precondition(candidate, await self.scope_of(session, existing))

    async def record(self, session: AsyncSession, new_id: str, others: Sequence[Particle]) -> int:
        """Join ``new_id`` to each declined ``other`` by ``CONTRADICTS``; returns how many are.

        A pair already joined (by an earlier pass) counts: the disclosure exists.
        """
        for other in others:
            await record_observer_divergence(session, new_id, other.id)
        return len([o for o in others if o.id != new_id])
