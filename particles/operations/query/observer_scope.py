# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer scope on the read path — the one filter every read calls.

The pure algebra lives in :mod:`particles.core.observer_scope`; the join that
says which projects currently observe a belief lives in
:mod:`particles.store.observer_scope_join`, shared with the write path's
reconciliation precondition so the two can never disagree. This
module answers which beliefs are in view for a project observer and discloses
what it left out.

With no observer project :func:`filter_visible` does no work at all, so every
unscoped read is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.observer_scope import visible
from particles.core.schema import ObserverScopeNote, Particle
from particles.core.status import Status
from particles.store.observer_scope_join import lens_may_engage, load_scopes
from particles.store.particle_store import list_particles_filtered

__all__ = [
    "ScopeFilter",
    "filter_visible",
    "in_view",
    "lens_may_engage",
    "list_particles_in_view",
    "load_scopes",
    "merged_scope_note",
]


@dataclass(frozen=True)
class ScopeFilter:
    """The outcome of reading a set of beliefs through a project observer."""

    visible_ids: frozenset[str]
    """Ids in view. Every id, when the lens is not engaged."""
    engaged: bool
    """Whether the observer was honoured. ``False`` with an observer project
    means the store has not been rescoped yet, so the read stayed store-wide."""
    total: int
    unattributed: int
    """Beliefs in view for no project because no source was ever keyed —
    disclosed, never silently dropped."""
    lapsed: int = 0
    """Beliefs in view for no project because no source still states them—
    disclosed apart from ``unattributed``."""
    as_of: datetime | None = None
    """The instant of an as-of read, whose projects are today's attribution."""

    @property
    def in_scope(self) -> int:
        return len(self.visible_ids)

    def note(self, observer_project: str) -> ObserverScopeNote:
        """The disclosure a response carries."""
        return ObserverScopeNote(
            project=observer_project,
            engaged=self.engaged,
            total=self.total,
            in_scope=self.in_scope,
            unattributed=self.unattributed,
            lapsed=self.lapsed,
            attribution=as_of_attribution(self.as_of) if self.engaged else None,
        )


def as_of_attribution(as_of: datetime | None) -> str | None:
    """The as-of read's attribution sentence, or ``None`` for a read of now."""
    if as_of is None:
        return None
    return (
        f"Projects are today's attribution, applied to what the store believed at "
        f"{as_of.isoformat()}; they are not what a session in that project saw then."
    )


def merged_scope_note(notes: list[ObserverScopeNote]) -> ObserverScopeNote | None:
    """One disclosure for a read that spanned several stores: counts summed.

    ``engaged`` only if every store honoured the observer — a federated read
    that stayed store-wide anywhere must not claim to be scoped.
    """
    if not notes:
        return None
    return ObserverScopeNote(
        project=notes[0].project,
        engaged=all(n.engaged for n in notes),
        total=sum(n.total for n in notes),
        in_scope=sum(n.in_scope for n in notes),
        unattributed=sum(n.unattributed for n in notes),
        lapsed=sum(n.lapsed for n in notes),
        attribution=next((n.attribution for n in notes if n.attribution), None),
    )


async def filter_visible(
    session: AsyncSession,
    particles: Sequence[Particle],
    observer_project: str | None,
    *,
    as_of: datetime | None = None,
) -> ScopeFilter:
    """Read ``particles`` through the project observer.

    Call this on the candidate set **before** scoring, ``top_k``, a relevance
    floor or any cap: filtering afterwards would starve a project's results.
    Any read that selects beliefs by something other than their id goes
    through here; a read *by id* is addressing, not retrieval, and does not.

    ``as_of`` is the instant of an as-of read: a project observes a
    claim its sources stated *then*, so "latest extracted snapshot" is taken at
    that instant.
    """
    ids = frozenset(p.id for p in particles)
    if observer_project is None:
        return ScopeFilter(ids, engaged=False, total=len(ids), unattributed=0)
    if not await lens_may_engage(session):
        return ScopeFilter(ids, engaged=False, total=len(ids), unattributed=0)

    scopes = await load_scopes(session, particles, as_of=as_of)
    in_view = frozenset(pid for pid, scope in scopes.items() if visible(scope, observer_project))
    unattributed = sum(1 for s in scopes.values() if s.unattributed)
    lapsed = sum(1 for s in scopes.values() if s.lapsed and not s.widened)
    return ScopeFilter(
        in_view,
        engaged=True,
        total=len(ids),
        unattributed=unattributed,
        lapsed=lapsed,
        as_of=as_of,
    )


async def in_view(
    session: AsyncSession, particles: Sequence[Particle], observer_project: str | None
) -> list[Particle]:
    """``particles`` that are in view for the observer, order preserved."""
    if observer_project is None:
        return list(particles)
    scope = await filter_visible(session, particles, observer_project)
    return [p for p in particles if p.id in scope.visible_ids]


async def list_particles_in_view(
    session: AsyncSession,
    *,
    status: Status | None,
    subject_id: str | None,
    limit: int,
    offset: int,
    observer_project: str | None,
) -> list[Particle]:
    """The browse listing, read through the observer.

    The predicate has to run before the page is cut, or a project's first page
    could be empty while later pages are not: under an observer the whole
    filtered listing is loaded, reduced to what is in view, and then paged. With
    no observer this is exactly the store's own paged query.
    """
    if observer_project is None:
        return await list_particles_filtered(
            session, status=status, subject_id=subject_id, limit=limit, offset=offset
        )
    everything = await list_particles_filtered(
        session, status=status, subject_id=subject_id, limit=None, offset=0
    )
    return (await in_view(session, everything, observer_project))[offset : offset + limit]
