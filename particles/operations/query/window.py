# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The window route — a question about a period, answered by world time.

The Engine half of the valid-time axis. The pure predicate lives in
:mod:`particles.core.validity`; this module loads one store's dated
candidates across statuses, closes a retired claim's interval by the
earliest-of rule (§4), lets a restatement stand in for the retired original
it supersedes (§4), and composes the deterministic listing (§3). The semantic
and structural paths call :func:`load_window_view` and
:meth:`WindowView.evaluate` where they load their candidates, so the window
cuts the candidate set before scoring, ``top_k`` and any cap, exactly where
observer scope cuts.

Nothing here is stored, and nothing here touches a score: a window is a
filter, and the three read-time axes are
untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    AsOfNote,
    AsOfSuccessor,
    ObserverScopeNote,
    Particle,
    QueryRequest,
    QueryResponse,
    TimeWindow,
    ValidKind,
    WindowCoverage,
    WindowHit,
)
from particles.core.stance import has_stance_marker
from particles.core.status import Status
from particles.core.validity import (
    VALID_FROM_BASIS_KEY,
    VALID_FROM_GRANULARITY_KEY,
    ValidFromBasis,
    ValidityFacts,
    ensure_utc,
    read_window,
    resolve_close,
)
from particles.db import DEFAULT_STORE, StoreHandle, session_scope
from particles.extraction.polarity import is_non_asserted
from particles.extraction.scope import is_excluded_document_meta
from particles.operations.version_guard import assert_store_schema_current
from particles.store.extractor_store import get_trust_weight_map, populate_trust_cache
from particles.store.particle_store import (
    count_active_undated,
    get_particles_in_window_scope,
    load_valid_from_by_ids,
)

from .as_of import RetirementIndex, is_never_believed, load_as_of_view, load_retirement_index
from .decay_policy import DecayPolicy, load_decay_policy
from .effective_confidence import score_effective_confidence
from .observer_scope import filter_visible, merged_scope_note
from .source_trust import TrustPolicy, load_trust_policy

#: Bases whose ``valid_from`` may close a predecessor: a stated,
#: prose or operator start, and an ``as-of`` anchor, which says the successor
#: held on that day and so bounds the predecessor's end from above. A
#: ``coarse`` anchor (a month, a year) is too wide to close anything.
CLOSING_BASES: frozenset[str] = frozenset(
    {
        ValidFromBasis.STATED,
        ValidFromBasis.PROSE,
        ValidFromBasis.OPERATOR,
        ValidFromBasis.AS_OF,
        ValidFromBasis.SESSION,
    }
)


@dataclass(frozen=True)
class WindowEvaluation:
    """The verdict for one candidate against the window."""

    hit: WindowHit | None
    """The reading and crumbs when admitted; None when not."""
    undated: bool = False
    unkinded: bool = False
    """Carried a date and no kind: admitted under ``BEGAN`` only."""
    excluded_undatable: bool = False
    """A retired claim whose close or retirement instant could not be
    resolved (rung 4): excluded fail-closed, disclosed."""


@dataclass(frozen=True)
class WindowView:
    """One store's window lens: the window, the retirement index, and the
    successors' starts the close rule reads (§4)."""

    window: TimeWindow
    index: RetirementIndex
    successor_valid_from: dict[str, datetime]

    def evaluate(self, particle: Particle, stored_retired_at: datetime | None) -> WindowEvaluation:
        """Admit or exclude one candidate, with the reading and the close."""
        if particle.valid_from is None:
            return WindowEvaluation(hit=None, undated=True)
        if is_never_believed(
            particle.status,
            particle.status_reason,
            born_retired=particle.id in self.index.never_believed,
        ):
            return WindowEvaluation(hit=None)
        props = particle.properties or {}
        basis_raw = props.get(VALID_FROM_BASIS_KEY)
        basis = str(basis_raw) if basis_raw is not None else None
        granularity_raw = props.get(VALID_FROM_GRANULARITY_KEY)
        granularity: Any = str(granularity_raw) if granularity_raw is not None else None
        unkinded = particle.valid_kind is None

        closed_at: datetime | None = None
        close_basis: Any = None
        retired: AsOfNote | None = None
        if particle.status is not Status.ACTIVE:
            resolution = self.index.resolve(
                particle.id,
                status_reason=particle.status_reason,
                valid_until=particle.valid_until,
                stored_retired_at=stored_retired_at,
            )
            if resolution is None:
                return WindowEvaluation(hit=None, excluded_undatable=True)
            successor = self.index.successor_by_predecessor.get(particle.id)
            successor_payload = None
            if successor is not None:
                successor_payload = AsOfSuccessor(
                    id=successor.id,
                    content=successor.content,
                    asserted_at=ensure_utc(successor.asserted_at),
                )
            retired = AsOfNote(
                status=particle.status,
                status_reason=particle.status_reason,
                retired_at=resolution.retired_at,
                basis=resolution.basis,
                successor=successor_payload,
            )
            if particle.valid_kind is ValidKind.STATE:
                close = resolve_close(
                    valid_until=particle.valid_until,
                    successor_valid_from=(
                        self.successor_valid_from.get(successor.id) if successor else None
                    ),
                    retired_at=resolution.retired_at,
                )
                if close is None:  # unreachable: retired_at is always set here
                    return WindowEvaluation(hit=None, excluded_undatable=True)
                closed_at, close_basis = close

        facts = ValidityFacts(
            valid_from=particle.valid_from,
            valid_kind=particle.valid_kind,
            valid_until=particle.valid_until,
            basis=basis,
            granularity=granularity,
            closed_at=closed_at,
        )
        reading = read_window(facts, self.window)
        if reading is None:
            return WindowEvaluation(hit=None, unkinded=unkinded)
        hit = WindowHit(
            reading=reading,
            valid_from=ensure_utc(particle.valid_from),
            valid_kind=particle.valid_kind,
            closed_at=closed_at,
            close_basis=close_basis,
            retired=retired,
        )
        return WindowEvaluation(hit=hit, unkinded=unkinded)


async def load_window_view(session: AsyncSession, window: TimeWindow) -> WindowView:
    """Snapshot one store's window lens.

    Reuses the retirement index (the successor map, the event map,
    the born-retired ids) and adds the successors' ``valid_from`` under a
    closing basis, loaded in one batch.
    """
    index = await load_retirement_index(session)
    successor_ids = {ref.id for ref in index.successor_by_predecessor.values()}
    starts = await load_valid_from_by_ids(session, successor_ids) if successor_ids else {}
    successor_valid_from = {
        pid: ensure_utc(vf)
        for pid, (vf, basis) in starts.items()
        if vf is not None and basis in CLOSING_BASES
    }
    return WindowView(window=window, index=index, successor_valid_from=successor_valid_from)


@dataclass
class WindowGather:
    """What a window read collected beside its candidates: the per-hit
    readings and the coverage counts. Mutable so the semantic and structural
    gathers can fill it in place, the ``scope_notes`` pattern."""

    hits: dict[str, WindowHit] = field(default_factory=dict)
    candidates: int = 0
    admitted: int = 0
    unkinded: int = 0
    excluded_undatable: int = 0
    retired_admitted: int = 0
    restated: int = 0

    async def coverage(self, session: AsyncSession) -> WindowCoverage:
        return WindowCoverage(
            candidates=self.candidates,
            admitted=self.admitted,
            undated=await count_active_undated(session),
            unkinded=self.unkinded,
            retired_admitted=self.retired_admitted,
            restated=self.restated,
        )

    def merge(self, other: WindowCoverage | None, mine: WindowCoverage) -> WindowCoverage:
        """Sum two stores' coverage (federation)."""
        if other is None:
            return mine
        return WindowCoverage(
            candidates=other.candidates + mine.candidates,
            admitted=other.admitted + mine.admitted,
            undated=other.undated + mine.undated,
            unkinded=other.unkinded + mine.unkinded,
            retired_admitted=other.retired_admitted + mine.retired_admitted,
            restated=other.restated + mine.restated,
        )


def apply_stand_in(gather: WindowGather, index: RetirementIndex) -> set[str]:
    """A restatement stands in for the retired original it supersedes.

    When an admitted retired claim's ``supersedes`` successor is itself in
    the admitted set, the original is dropped and the successor's hit carries
    a ``restates`` note naming it. Returns the dropped ids; the caller removes
    them from its candidates.
    """
    dropped: set[str] = set()
    for pid, hit in list(gather.hits.items()):
        if hit.retired is None:
            continue
        successor = index.successor_by_predecessor.get(pid)
        if successor is None or successor.id not in gather.hits:
            continue
        dropped.add(pid)
        gather.hits[successor.id] = gather.hits[successor.id].model_copy(update={"restates": pid})
    for pid in dropped:
        del gather.hits[pid]
    gather.restated += len(dropped)
    gather.retired_admitted -= len(dropped)
    return dropped


async def gather_window_candidates(
    session: AsyncSession,
    request: QueryRequest,
    gather: WindowGather,
    *,
    require_embedding: bool = False,
    require_claim: bool = False,
    as_of_notes: dict[str, AsOfNote] | None = None,
) -> tuple[list[tuple[Particle, Any, datetime | None]], int]:
    """Load and window-filter one store's dated candidates.

    Shared by the semantic path (``require_embedding``), the structural path
    (``require_claim``) and the listing. With ``request.as_of`` also set, the
    as-of visibility applies first: world time selects, believed-at-T
    filters. Returns the admitted ``(particle, embedding, retired_at)``
    triples and the as-of undatable count.
    """
    assert request.window is not None
    view = await load_window_view(session, request.window)
    as_of_view = await load_as_of_view(session, request.as_of) if request.as_of else None
    loaded = await get_particles_in_window_scope(
        session,
        request.min_confidence,
        subject_id=request.subject_id,
        require_embedding=require_embedding,
        require_claim=require_claim,
    )
    gather.candidates += len(loaded)
    as_of_excluded = 0
    admitted: list[tuple[Particle, Any, datetime | None]] = []
    for p, emb, retired_at in loaded:
        if as_of_view is not None:
            evaluation = as_of_view.evaluate(p, retired_at)
            if evaluation.excluded_undatable:
                as_of_excluded += 1
                continue
            if not evaluation.visible:
                continue
            if evaluation.note is not None and as_of_notes is not None:
                as_of_notes[p.id] = evaluation.note
        verdict = view.evaluate(p, retired_at)
        if verdict.excluded_undatable:
            gather.excluded_undatable += 1
        if verdict.unkinded:
            gather.unkinded += 1
        if verdict.hit is None:
            continue
        gather.hits[p.id] = verdict.hit
        gather.admitted += 1
        if verdict.hit.retired is not None:
            gather.retired_admitted += 1
        admitted.append((p, emb, retired_at))
    dropped = apply_stand_in(gather, view.index)
    if dropped:
        admitted = [t for t in admitted if t[0].id not in dropped]
    return admitted, as_of_excluded


async def _default_filters(
    session: AsyncSession,
    request: QueryRequest,
    particles: list[Particle],
    *,
    scope_notes: list[ObserverScopeNote] | None,
) -> list[Particle]:
    """The default exclusions every read applies, for the listing path.

    Mirrors the chain in ``main._gather_scored`` and
    ``structural._candidate_claims`` (DOCUMENT_META, non-asserted, stance
    markers, the nature and modality filters, tags, the observer), so the
    listing sees the same candidate universe.
    """
    if not request.include_document_meta:
        particles = [p for p in particles if not is_excluded_document_meta(p.properties)]
    if not request.include_non_asserted:
        particles = [p for p in particles if not is_non_asserted(p.properties)]
    particles = [p for p in particles if not has_stance_marker(p)]
    if request.uncertainty_nature:
        particles = [p for p in particles if p.uncertainty_nature == request.uncertainty_nature]
    if request.assertion_modality is not None:
        particles = [p for p in particles if p.assertion_modality == request.assertion_modality]
    if request.tags:
        from particles.store.taxonomy_store import expand_tags, get_particle_ids_for_tags

        expanded = await expand_tags(
            session, request.tags, include_ancestors=request.include_ancestors
        )
        tagged_ids = await get_particle_ids_for_tags(session, expanded)
        particles = [p for p in particles if p.id in tagged_ids]
    if request.observer_project is not None:
        scope = await filter_visible(
            session, particles, request.observer_project, as_of=request.as_of
        )
        particles = [p for p in particles if p.id in scope.visible_ids]
        if scope_notes is not None:
            scope_notes.append(scope.note(request.observer_project))
    return particles


def _hit_sort_key(hit: WindowHit) -> tuple[datetime, int]:
    #: BEGAN before the standing context at the same instant, so a day's
    #: events read in order under the states that frame them.
    order = {"BEGAN": 0, "IN_FORCE": 1, "HELD_AT": 2, "WITHIN": 3}
    return hit.valid_from, order.get(hit.reading.value, 9)


async def _listing_store(
    session: AsyncSession,
    request: QueryRequest,
    *,
    trust_policy: TrustPolicy | None,
    decay_policy: DecayPolicy | None,
    populate_cache: bool,
    scope_notes: list[ObserverScopeNote],
) -> tuple[list[tuple[Particle, float]], WindowGather, dict[str, AsOfNote], int]:
    gather = WindowGather()
    as_of_notes: dict[str, AsOfNote] = {}
    admitted, as_of_excluded = await gather_window_candidates(
        session, request, gather, as_of_notes=as_of_notes
    )
    particles = await _default_filters(
        session, request, [p for p, _, _ in admitted], scope_notes=scope_notes
    )
    eff = await score_effective_confidence(
        session,
        particles,
        trust_policy,
        decay_policy=decay_policy,
        populate_cache=populate_cache,
        now=request.as_of,
    )
    rows = [(p, eff[p.id]) for p in particles if eff[p.id] >= request.min_confidence]
    return rows, gather, as_of_notes, as_of_excluded


def _listing_response(
    request: QueryRequest,
    rows: list[tuple[Particle, float]],
    gather: WindowGather,
    coverage: WindowCoverage,
    as_of_notes: dict[str, AsOfNote],
    as_of_excluded: int,
    scope_notes: list[ObserverScopeNote],
) -> QueryResponse:
    ordered = sorted(rows, key=lambda row: _hit_sort_key(gather.hits[row[0].id]))
    top = ordered[: request.top_k]
    began = sum(1 for p, _ in ordered if gather.hits[p.id].reading.value == "BEGAN")
    answer = f"{len(ordered)} claims sit in the window: {began} began in it, "
    answer += f"{len(ordered) - began} were in force."
    if len(ordered) > len(top):
        answer += f" Showing the first {len(top)} by valid_from."
    return QueryResponse(
        answer=answer,
        particles=[p for p, _ in top],
        effective_confidences=[e for _, e in top],
        window=request.window,
        window_hits=[gather.hits.get(p.id) for p, _ in top],
        window_coverage=coverage,
        as_of=request.as_of,
        as_of_notes=([as_of_notes.get(p.id) for p, _ in top] if request.as_of is not None else []),
        as_of_excluded_undatable=as_of_excluded,
        observer_scope=merged_scope_note(scope_notes),
    )


async def window_listing(session: AsyncSession, request: QueryRequest) -> QueryResponse:
    """The deterministic window listing: no embedding, no LLM.

    Dispatched from :func:`particles.operations.query.main.query` when the
    request carries a window and no question, claim filter or aggregate. The
    claims in the window ordered by ``valid_from``, each with its reading.
    """
    await assert_store_schema_current(session)
    scope_notes: list[ObserverScopeNote] = []
    rows, gather, as_of_notes, as_of_excluded = await _listing_store(
        session,
        request,
        trust_policy=None,
        decay_policy=None,
        populate_cache=True,
        scope_notes=scope_notes,
    )
    coverage = await gather.coverage(session)
    return _listing_response(
        request, rows, gather, coverage, as_of_notes, as_of_excluded, scope_notes
    )


async def window_listing_federated(
    stores: list[StoreHandle],
    request: QueryRequest,
    viewer_store: StoreHandle | None = None,
) -> QueryResponse:
    """The window listing across several stores under one viewer's lens."""
    if not stores:
        stores = [DEFAULT_STORE]
    viewer = viewer_store or stores[0]
    async with session_scope(viewer) as viewer_session:
        await assert_store_schema_current(viewer_session)
        populate_trust_cache(await get_trust_weight_map(viewer_session))
        trust_policy = await load_trust_policy(viewer_session)
        decay_policy = await load_decay_policy(viewer_session)

    scope_notes: list[ObserverScopeNote] = []
    merged_rows: list[tuple[Particle, float]] = []
    merged = WindowGather()
    merged_coverage: WindowCoverage | None = None
    as_of_notes: dict[str, AsOfNote] = {}
    as_of_excluded = 0
    for store in stores:
        async with session_scope(store) as s:
            await assert_store_schema_current(s)
            rows, gather, notes, excluded = await _listing_store(
                s,
                request,
                trust_policy=trust_policy,
                decay_policy=decay_policy,
                populate_cache=False,
                scope_notes=scope_notes,
            )
            merged_rows.extend(rows)
            merged.hits.update(gather.hits)
            merged_coverage = merged.merge(merged_coverage, await gather.coverage(s))
            as_of_notes.update(notes)
            as_of_excluded += excluded
    assert merged_coverage is not None
    return _listing_response(
        request, merged_rows, merged, merged_coverage, as_of_notes, as_of_excluded, scope_notes
    )


__all__ = [
    "CLOSING_BASES",
    "WindowEvaluation",
    "WindowGather",
    "WindowView",
    "apply_stand_in",
    "gather_window_candidates",
    "load_window_view",
    "window_listing",
    "window_listing_federated",
]
