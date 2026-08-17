"""Scoped epistemic subgraph assembly.

The one build behind both graph presentation surfaces: the static HTML
``graph`` exporter embeds the resulting :class:`~particles.core.schema.GraphData`
as JSON, and the deferred ``GET /graph`` endpoint will return the same model on
the wire. Scope resolution → store reads → read-time epistemics annotation
(effective confidence, as-of visibility, contested
badges, utility evidence)
→ anti-hairball caps with bounded-view-style
disclosures.

The anti-hairball invariant is enforced here, not at the surfaces: scope is
mandatory (exactly one of ``subject_id`` / ``query``), and the
``graph.max_nodes`` / ``graph.max_particles_per_subject`` / ``graph.max_hops``
caps bound every render. A capped render carries both a human-readable
disclosure line naming the knob and the machine-readable census — a disclosed
lower bound, never a silent truncation.

Every epistemic quantity annotated here is computed at read time and never
stored (the spine); the build never writes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    AsOfNote,
    ContestedBadge,
    GraphCensus,
    GraphData,
    GraphEdge,
    GraphNode,
    GraphParticleInfo,
    GraphSupersession,
    Particle,
    ProvenanceRefType,
    QueryRequest,
)
from particles.core.status import Status, StatusReason
from particles.corpus.store import get_entry_uri_map
from particles.operations.query.as_of import AsOfView, ensure_utc, load_as_of_view
from particles.operations.query.contested import compute_contested_badges
from particles.operations.query.effective_confidence import score_effective_confidence
from particles.operations.query.main import retrieve_ranked
from particles.operations.query.owner_policy import load_owner_policy
from particles.render.markdown import build_subject_naming, exclude_non_asserted
from particles.store.particle_store import (
    ParticleRow,
    get_particle,
    get_particles_by_ids,
    get_retired_at,
    get_superseding_particle,
)
from particles.store.subject_store import (
    find_by_name,
    get_particles_for_subject,
    get_subject,
    list_all_subjects,
    search_subjects,
)
from particles.store.utility_store import get_reinforcement_scores

log = logging.getLogger(__name__)


async def build_graph_data(
    session: AsyncSession,
    *,
    subject_id: str | None = None,
    query: str | None = None,
    inconsistency_id: str | None = None,
    manifest: str | None = None,
    section: str | None = None,
    hops: int = 1,
    history: bool = False,
    as_of: datetime | None = None,
    max_nodes: int | None = None,
    min_particle_confidence: float = 0.0,
    include_non_asserted: bool = False,
) -> GraphData:
    """Assemble one scoped subgraph.

    Exactly one scope must be given — an unscoped render is a ``ValueError``,
    not a default: ``subject_id`` (neighbourhood), ``query`` (retrieval set),
    ``inconsistency_id`` (a contradiction's evidence), or
    ``manifest`` + ``section`` together (a projection section's selection
). ``subject_id`` accepts a subject id or an exact
    (case-insensitive) canonical name / alias; ``inconsistency_id`` accepts a
    full particle id or a unique prefix (the badge prose truncates to 8
    chars). The returned ``scope_ref`` is always the resolved anchor address.
    ``hops`` applies to subject scope only and is clamped to
    ``graph.max_hops``. ``history`` additionally includes the retired
    supersession-chain ancestors of in-scope particles (rendered as ghosts;
    they never expand the neighbourhood). ``as_of`` renders the graph as
    believed at T (visibility + decay move to T; trust / utility /
    contested stay current). ``max_nodes`` is clamped to ``graph.max_nodes``.
    """
    cfg = get_config().graph
    if (manifest is None) != (section is None):
        raise ValueError(
            "projection scope needs both selectors: pass manifest (the manifest "
            "path) and section (a section's region or title) together"
        )
    selectors = [s for s in (subject_id, query, inconsistency_id, manifest) if s is not None]
    if len(selectors) != 1:
        raise ValueError(
            "graph scope is mandatory: pass exactly one of subject_id, query, "
            "inconsistency_id, or manifest+section (a whole-store render does "
            "not exist)"
        )
    node_cap = cfg.max_nodes if max_nodes is None else min(max_nodes, cfg.max_nodes)
    hops = min(hops, cfg.max_hops)

    view = await load_as_of_view(session, as_of) if as_of is not None else None

    scope_type: Literal["subject", "query", "inconsistency", "projection"]
    hit_ids: set[str] = set()
    hit_rank: dict[str, int] = {}
    missing_disputants = 0
    if subject_id is not None:
        scope_type = "subject"
        particles, hop_by_subject, excluded_undatable, anchor_id = await _gather_subject_scope(
            session, subject_id, hops, view, include_non_asserted
        )
        # scope_ref is always the RESOLVED anchor id (the model documents it as
        # "the anchor subject id"), even when the caller passed a name — so
        # deep links and re-anchoring stay canonical.
        scope_ref = anchor_id
        scope_desc = f"subject {anchor_id} neighbourhood ({hops} hop{'s' if hops != 1 else ''})"
    elif inconsistency_id is not None:
        scope_type = "inconsistency"
        (
            particles,
            hop_by_subject,
            hit_ids,
            hit_rank,
            excluded_undatable,
            anchor_pid,
            missing_disputants,
        ) = await _gather_inconsistency_scope(session, inconsistency_id, view, include_non_asserted)
        scope_ref = anchor_pid
        scope_desc = f"INCONSISTENCY {anchor_pid[:8]} evidence"
    elif manifest is not None:
        assert section is not None
        scope_type = "projection"
        (
            particles,
            hop_by_subject,
            hit_ids,
            hit_rank,
            excluded_undatable,
            resolved_section,
        ) = await _gather_projection_scope(session, manifest, section, view, include_non_asserted)
        scope_ref = f"{manifest}#{resolved_section}"
        scope_desc = f"projection section {resolved_section!r} selection"
    else:
        assert query is not None
        scope_type = "query"
        scope_ref = query
        scope_desc = f"query retrieval set (top {cfg.query_top_k})"
        (
            particles,
            hop_by_subject,
            hit_ids,
            hit_rank,
            excluded_undatable,
        ) = await _gather_query_scope(session, query, view, as_of, include_non_asserted)

    # History: pull the retired supersession-chain ancestors of what's in
    # scope (plus successors of retired in-scope particles), as ghosts.
    ghosts: set[str] = set()
    if history:
        ghosts = await _extend_history(session, particles, view)

    # Epistemics annotation — all read-time, never stored.
    # `apply_utility_factor` stays False on purpose: the rule
    # that utility never touches truth/query surfaces holds in the visual
    # channel too, so utility reaches the graph only as node *diameter*. The
    # owner lens inherits that stance exactly — it is applied to the
    # node *selection* key below, never to `eff`, so it can decide which nodes
    # survive `max_nodes` but can never alter a rendered confidence.
    eff = await score_effective_confidence(
        session, list(particles.values()), populate_cache=True, now=as_of
    )

    # cross-exporter contract: the effective-confidence floor applies
    # before any downstream step. Ghosts are epistemic history, not the
    # current surface — the floor applies to them identically.
    dropped_below_threshold = 0
    if min_particle_confidence > 0.0:
        surviving: dict[str, Particle] = {}
        for pid, p in particles.items():
            if eff.get(pid, 0.0) < min_particle_confidence:
                dropped_below_threshold += 1
            else:
                surviving[pid] = p
        particles = surviving
        ghosts &= set(particles)

    # As-of notes: annotate visible-but-since-retired particles.
    as_of_notes: dict[str, AsOfNote] = {}
    if view is not None:
        retired_at_map = await _load_retired_at(session, list(particles))
        for pid, p in particles.items():
            if pid in ghosts:
                continue
            evaluation = view.evaluate(p, retired_at_map.get(pid))
            if evaluation.visible and evaluation.note is not None:
                as_of_notes[pid] = evaluation.note

    # Contested badges for the current (non-ghost) surface, gated by
    # the §7 kill switch like every other call site — off leaves every node's
    # ``contested`` flag False and every ``GraphParticleInfo.contested`` None,
    # the pre-badge render exactly. The gate was missing here until.
    badges: dict[str, ContestedBadge | None] = {}
    if get_config().contestedness.badge_enabled:
        current = [p for pid, p in sorted(particles.items()) if pid not in ghosts]
        badges = dict(
            zip(
                [p.id for p in current],
                await compute_contested_badges(session, current),
                strict=True,
            )
        )

    # Utility evidence: raw reinforcement scores, display-only —
    # node size, never opacity; absent evidence is the neutral 0.0.
    utility_cfg = get_config().utility
    utility: dict[str, float] = {}
    if utility_cfg.enabled and particles:
        utility = await get_reinforcement_scores(
            session, list(particles), utility_cfg.default.half_life_uses_days
        )

    source_uris = await _load_source_uris(session, list(particles.values()))

    # disambiguated node labels, computed on the FULL subject set.
    subjects_by_id = {s.id: s for s in await list_all_subjects(session)}
    naming = build_subject_naming(subjects_by_id.values())

    def _node_support(sid: str) -> float:
        """Max effective confidence over a subject's non-ghost particles."""
        return max(
            (
                eff.get(pid, 0.0)
                for pid, p in particles.items()
                if pid not in ghosts and sid in p.subject_ids
            ),
            default=0.0,
        )

    def _node_hit_rank(sid: str) -> int:
        """Best (lowest) retrieval rank among a subject's hit particles."""
        return min(
            (
                hit_rank[pid]
                for pid, p in particles.items()
                if pid in hit_rank and sid in p.subject_ids
            ),
            default=len(hit_rank),
        )

    candidate_subject_ids = {
        sid for p in particles.values() for sid in p.subject_ids if sid in hop_by_subject
    }
    if subject_id is not None:
        candidate_subject_ids.add(subject_id)  # the anchor always renders
    candidate_particles = len(particles)

    # on the graph: this surface's unit is a Subject, not a belief, so
    # the viewer is not a promoted claim here — the viewer *is a node*, and what
    # the lens does is keep their neighbourhood alive through the max_nodes cut.
    # A subject is viewer-adjacent if it is the viewer or shares an **in-scope**
    # particle with them. In-scope only, deliberately: the lens re-orders what
    # the traversal already loaded and never widens the scope, which the spec
    # makes mandatory and explicit. So at the hop boundary — where nodes render
    # without their own cargo fetch — adjacency degrades to "is the viewer",
    # which is the correct conservative answer rather than a missing one. Inert
    # (a constant key element, so the order is byte-identical) when no viewer
    # resolves.
    owner_policy = await load_owner_policy(session)
    viewer_adjacent: set[str] = set()
    if owner_policy.viewer_subject_ids:
        viewer_adjacent |= owner_policy.viewer_subject_ids & candidate_subject_ids
        for p in particles.values():
            if not owner_policy.viewer_subject_ids.isdisjoint(p.subject_ids):
                viewer_adjacent.update(p.subject_ids)
        viewer_adjacent &= candidate_subject_ids

    # Node ranking for the max_nodes cap (§2/§3): subject scope ranks
    # by hop distance then support; query scope by retrieval rank then support.
    # Viewer-adjacency sits directly under hop distance — scope
    # structure still wins, but within a hop level the viewer's neighbourhood
    # survives truncation ahead of an unrelated subject. Promotion-only, like
    # every other owner-lens term.
    ranked_subjects = sorted(
        candidate_subject_ids,
        key=lambda sid: (
            hop_by_subject.get(sid, 0),
            0 if sid in viewer_adjacent else 1,
            _node_hit_rank(sid),
            -_node_support(sid),
            sid,
        ),
    )
    rendered_subject_ids = ranked_subjects[:node_cap]
    rendered_set = set(rendered_subject_ids)

    # A particle renders as an edge iff ≥2 of its subjects survived the cap
    # (a 3+-subject particle renders as a pairwise clique sharing one id);
    # with exactly 1 it renders as that node's cargo; with 0 it drops.
    edges: list[GraphEdge] = []
    cargo_by_subject: dict[str, list[str]] = defaultdict(list)
    rendered_particles: dict[str, Particle] = {}
    unanchored_foreground = 0
    for pid in sorted(particles):
        p = particles[pid]
        in_scope = sorted(set(p.subject_ids) & rendered_set)
        if not in_scope:
            # A FOREGROUND particle (query hit, inconsistency evidence,
            # projection selection) with no rendered subject still reaches the
            # payload — real stores hold pre-subject-binding records (e.g. an
            # old INCONSISTENCY whose disputants resolved no subjects), and
            # dropping the very particles the scope exists to show would be a
            # silent lie. They render in the detail panel, not on the canvas,
            # and the gap is disclosed below. Incidental cargo still drops.
            if pid in hit_ids:
                rendered_particles[pid] = p
                unanchored_foreground += 1
            continue
        rendered_particles[pid] = p
        if len(in_scope) >= 2:
            for a, b in combinations(in_scope, 2):
                edges.append(GraphEdge(particle_id=pid, source=a, target=b))
        else:
            cargo_by_subject[in_scope[0]].append(pid)

    # Per-subject cargo cap (descending effective confidence, id tiebreak).
    # The scope's foreground set (query hits, inconsistency evidence,
    # projection selection) sorts ahead of incidental cargo so the cap can
    # never truncate the very particles the scope exists to show.
    edge_pids = {e.particle_id for e in edges}
    cargo_truncated_total = 0
    cargo_truncated_by_subject: dict[str, int] = {}
    for sid in list(cargo_by_subject):
        pids = cargo_by_subject[sid]
        pids.sort(key=lambda pid: (0 if pid in hit_ids else 1, -eff.get(pid, 0.0), pid))
        if len(pids) > cfg.max_particles_per_subject:
            overflow = pids[cfg.max_particles_per_subject :]
            cargo_by_subject[sid] = pids[: cfg.max_particles_per_subject]
            cargo_truncated_by_subject[sid] = len(overflow)
            cargo_truncated_total += len(overflow)
            for pid in overflow:
                if pid not in edge_pids:
                    rendered_particles.pop(pid, None)

    nodes = [
        GraphNode(
            subject_id=sid,
            label=(naming.display_name(subjects_by_id[sid]) if sid in subjects_by_id else sid),
            subject_class=(subjects_by_id[sid].subject_class if sid in subjects_by_id else None),
            hop=hop_by_subject.get(sid, 0),
            max_effective_confidence=_node_support(sid),
            utility_score=sum(
                utility.get(pid, 0.0)
                for pid, p in rendered_particles.items()
                if sid in p.subject_ids
            ),
            contested=any(
                badges.get(pid) is not None
                for pid, p in rendered_particles.items()
                if sid in p.subject_ids
            ),
            cargo=cargo_by_subject.get(sid, []),
            cargo_truncated=cargo_truncated_by_subject.get(sid, 0),
        )
        for sid in rendered_subject_ids
    ]

    supersessions = [
        GraphSupersession(predecessor_id=p.supersedes, successor_id=pid)
        for pid, p in sorted(rendered_particles.items())
        if p.supersedes is not None and p.supersedes in rendered_particles
    ]

    infos = {
        pid: GraphParticleInfo(
            id=pid,
            content=p.content,
            status=p.status,
            status_reason=p.status_reason,
            confidence=p.confidence.value,
            effective_confidence=eff.get(pid, 0.0),
            subject_ids=sorted(p.subject_ids),
            asserted_at=p.asserted_at,
            valid_until=p.valid_until,
            supersedes=p.supersedes,
            contested=badges.get(pid),
            utility_score=utility.get(pid, 0.0),
            source_uri=source_uris.get(pid),
            retrieval_hit=pid in hit_ids,
            ghost=pid in ghosts,
            as_of_note=as_of_notes.get(pid),
        )
        for pid, p in sorted(rendered_particles.items())
    }

    disclosures: list[str] = []
    if len(ranked_subjects) > len(nodes):
        disclosures.append(
            f"showing {len(nodes)} of {len(ranked_subjects)} subjects "
            f"(graph.max_nodes = {node_cap}) — this view is a disclosed lower "
            f"bound, not a census"
        )
    if cargo_truncated_total:
        disclosures.append(
            f"{cargo_truncated_total} particle(s) beyond the per-subject panel cap "
            f"omitted (graph.max_particles_per_subject = {cfg.max_particles_per_subject})"
        )
    if dropped_below_threshold:
        disclosures.append(
            f"{dropped_below_threshold} particle(s) below "
            f"min_particle_confidence = {min_particle_confidence} dropped"
        )
    if excluded_undatable:
        disclosures.append(
            f"{excluded_undatable} retired particle(s) excluded fail-closed: "
            f"retirement instant not reconstructible"
        )
    if missing_disputants:
        disclosures.append(
            f"{missing_disputants} disputant particle(s) referenced by this "
            f"INCONSISTENCY no longer exist in the store — the evidence shown "
            f"is incomplete"
        )
    if unanchored_foreground:
        disclosures.append(
            f"{unanchored_foreground} foreground particle(s) have no linked "
            f"subject — they appear in the detail panel, not on the canvas"
        )

    log.info(
        "graph build: %s → %d nodes, %d edges, %d particles",
        scope_desc,
        len(nodes),
        len(edges),
        len(infos),
    )
    return GraphData(
        scope_type=scope_type,
        scope_ref=scope_ref,
        as_of=as_of,
        history=history,
        nodes=nodes,
        edges=sorted(edges, key=lambda e: (e.source, e.target, e.particle_id)),
        supersessions=supersessions,
        particles=infos,
        census=GraphCensus(
            scope=scope_desc,
            candidate_subjects=len(candidate_subject_ids),
            rendered_subjects=len(nodes),
            candidate_particles=candidate_particles,
            rendered_particles=len(infos),
            dropped_below_threshold=dropped_below_threshold,
            excluded_undatable=excluded_undatable,
        ),
        disclosures=disclosures,
    )


def _currently_visible(
    p: Particle,
    retired_at: datetime | None,
    view: AsOfView | None,
) -> tuple[bool, bool]:
    """(visible-on-the-current-lens, excluded-undatable) for one particle.

    Present-time lens: visible = ACTIVE. As-of lens: the predicate. INCONSISTENCY records are never rendered as ordinary elements
    on either lens (they surface only through the contested
    badge's basis).
    """
    if p.status is Status.INCONSISTENCY:
        return False, False
    if view is None:
        return p.status is Status.ACTIVE, False
    evaluation = view.evaluate(p, retired_at)
    return evaluation.visible, evaluation.excluded_undatable


def _history_eligible(p: Particle) -> bool:
    """A retired particle a --history render may show as a ghost.

    Once-believed retirements only: INCONSISTENCY records and born-retired
    quarantine losers (``CONFLICT_PENDING``) were never believed and never
    render (the exclusion-set rule).
    """
    return (
        p.status is not Status.INCONSISTENCY
        and p.status_reason is not StatusReason.CONFLICT_PENDING
    )


async def _load_retired_at(
    session: AsyncSession, particle_ids: list[str]
) -> dict[str, datetime | None]:
    """Batch-load the ``retired_at`` storage column for the id set."""
    out: dict[str, datetime | None] = {}
    chunk = 500
    for i in range(0, len(particle_ids), chunk):
        result = await session.execute(
            select(ParticleRow.id, ParticleRow.retired_at).where(
                ParticleRow.id.in_(particle_ids[i : i + chunk])
            )
        )
        for pid, retired_at in result.all():
            out[pid] = retired_at
    return out


async def _load_source_uris(session: AsyncSession, particles: list[Particle]) -> dict[str, str]:
    """Map particle id → source URI via each particle's SOURCE provenance ref."""
    entry_by_pid: dict[str, str] = {}
    for p in particles:
        src = next((r for r in p.provenance if r.type == ProvenanceRefType.SOURCE), None)
        if src is not None and src.corpus_entry_id:
            entry_by_pid[p.id] = src.corpus_entry_id
    if not entry_by_pid:
        return {}
    uri_map = await get_entry_uri_map(session, set(entry_by_pid.values()))
    return {pid: uri for pid, eid in entry_by_pid.items() if (uri := uri_map.get(eid)) is not None}


async def _fetch_subject_particles(
    session: AsyncSession,
    sid: str,
    view: AsOfView | None,
    include_non_asserted: bool,
) -> tuple[dict[str, Particle], int]:
    """One subject's visible particles on the current lens (+ undatable count)."""
    ids = await get_particles_for_subject(session, sid)
    by_id = await get_particles_by_ids(session, ids)
    candidates = exclude_non_asserted(
        list(by_id.values()), {"include_non_asserted": include_non_asserted}
    )
    retired_at = await _load_retired_at(session, [p.id for p in candidates])
    visible: dict[str, Particle] = {}
    undatable = 0
    for p in candidates:
        ok, excl = _currently_visible(p, retired_at.get(p.id), view)
        if excl:
            undatable += 1
        elif ok:
            visible[p.id] = p
    return visible, undatable


async def _gather_subject_scope(
    session: AsyncSession,
    subject_id: str,
    hops: int,
    view: AsOfView | None,
    include_non_asserted: bool,
) -> tuple[dict[str, Particle], dict[str, int], int, str]:
    """BFS one subject's neighbourhood: visible particles + hop distances.

    ``subject_id`` may be a subject id or an exact (case-insensitive)
    canonical name / alias — the same :func:`find_by_name` lookup extraction
    resolves against, so a pasted "Donkey Kong" just works. Substring /
    near-miss inputs stay an error with suggestions (a render must never
    guess its anchor). The last tuple element is the RESOLVED anchor id.

    Expansion crosses only lens-visible multi-subject particles — history
    ghosts never widen the neighbourhood. This is the *anchor's*
    neighbourhood, not the neighbours': boundary-hop subjects
    render as nodes with their connecting edges, but their own particle sets
    are not fetched — that is what re-anchoring the render on them is for.
    """
    anchor = await get_subject(session, subject_id)
    if anchor is None:
        # Not an id — try the exact (case-insensitive) name/alias lookup the
        # ingest resolver uses. Duplicate canonical names resolve
        # deterministically to the earliest subject (find_by_name logs the
        # collision), exactly as extraction does.
        anchor = await find_by_name(session, subject_id)
    if anchor is None:
        # A pasted id that misses is usually a *different store's* id (renders
        # are store-scoped); a near-miss name can still be rescued by suggestion.
        matches = await search_subjects(session, subject_id, limit=3)
        if matches:
            names = "; ".join(f"{s.canonical_name} ({s.id})" for s in matches)
            hint = f"closest name matches in this store: {names}"
        else:
            hint = (
                "subject ids are store-scoped — a render's ids resolve only "
                "against the store it was exported from (check DATABASE_URL)"
            )
        raise ValueError(f"unknown subject id: {subject_id!r} — {hint}")

    anchor_id = anchor.id
    hop_by_subject: dict[str, int] = {anchor_id: 0}
    particles: dict[str, Particle] = {}
    excluded_undatable = 0
    frontier = [anchor_id]
    for hop in range(hops + 1):
        if hop == hops and hop != 0:
            break  # boundary nodes render without their own cargo fetch
        next_frontier: list[str] = []
        for sid in frontier:
            visible, undatable = await _fetch_subject_particles(
                session, sid, view, include_non_asserted
            )
            excluded_undatable += undatable
            particles.update(visible)
            if hop < hops:
                for p in visible.values():
                    for neighbour in sorted(p.subject_ids):
                        if neighbour not in hop_by_subject:
                            hop_by_subject[neighbour] = hop + 1
                            next_frontier.append(neighbour)
        frontier = next_frontier
    return particles, hop_by_subject, excluded_undatable, anchor_id


async def _gather_query_scope(
    session: AsyncSession,
    query: str,
    view: AsOfView | None,
    as_of: datetime | None,
    include_non_asserted: bool,
) -> tuple[dict[str, Particle], dict[str, int], set[str], dict[str, int], int]:
    """One query's retrieval set + its subjects' incidental cargo.

    The retrieval hits come from the shared ranked-selection pipeline
    (:func:`retrieve_ranked` — no NL answer generated); every hit subject
    renders at hop 0, topped up with its own lens-visible particles as
    incidental cargo so the picture shows what surrounds the consulted
    beliefs.
    """
    cfg = get_config().graph
    request = QueryRequest(
        question=query,
        top_k=cfg.query_top_k,
        as_of=as_of,
        include_non_asserted=include_non_asserted,
    )
    scored = await retrieve_ranked(session, request)

    particles: dict[str, Particle] = {}
    hit_ids: set[str] = set()
    hit_rank: dict[str, int] = {}
    hop_by_subject: dict[str, int] = {}
    for rank, (p, _sim, _eff) in enumerate(scored):
        particles[p.id] = p
        hit_ids.add(p.id)
        hit_rank[p.id] = rank
        for sid in p.subject_ids:
            hop_by_subject.setdefault(sid, 0)

    excluded_undatable = 0
    for sid in sorted(hop_by_subject):
        visible, undatable = await _fetch_subject_particles(
            session, sid, view, include_non_asserted
        )
        excluded_undatable += undatable
        for pid, p in visible.items():
            particles.setdefault(pid, p)
    return particles, hop_by_subject, hit_ids, hit_rank, excluded_undatable


async def _inconsistency_by_prefix(session: AsyncSession, prefix: str) -> Particle | None:
    """Resolve a unique id prefix against INCONSISTENCY rows (badge prose is 8 chars).

    Exact-miss fallback only; ambiguity raises rather than guessing an anchor.
    """
    from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern

    pattern = f"{escape_like_pattern(prefix)}%"
    result = await session.execute(
        select(ParticleRow)
        .where(
            ParticleRow.id.like(pattern, escape=LIKE_ESCAPE),
            ParticleRow.status == Status.INCONSISTENCY.value,
        )
        .limit(2)
    )
    rows = result.scalars().all()
    if len(rows) > 1:
        raise ValueError(
            f"ambiguous inconsistency id prefix {prefix!r} matches multiple "
            "INCONSISTENCY particles; use more characters"
        )
    return rows[0].to_model() if rows else None


async def _gather_inconsistency_scope(
    session: AsyncSession,
    inconsistency_id: str,
    view: AsOfView | None,
    include_non_asserted: bool,
) -> tuple[dict[str, Particle], dict[str, int], set[str], dict[str, int], int, str, int]:
    """A contradiction's evidence: the INCONSISTENCY anchor, its disputants,
    their subjects and sources.

    The anchor renders here by design — the one exception to the §5.2 rule
    that INCONSISTENCY records never render as ordinary elements ("and, in
    the deferred evidence scope, as the anchor"). Anchor + disputants form
    the scope's foreground set (``retrieval_hit``-highlighted) and are
    included with their **true statuses regardless of the current lens**: the
    quarantined loser of a §6.6 conflict is PROVENANCE_STALE and would never
    surface on an ordinary render, but it is half the evidence. Their
    subjects render at hop 0, topped up with lens-visible incidental cargo
    exactly like query scope. Accepts a full particle id or a unique prefix
    (the contested-badge prose truncates to 8 chars).

    Returns ``(particles, hop_by_subject, hit_ids, hit_rank,
    excluded_undatable, anchor_id, missing_disputants)``.
    """
    anchor = await get_particle(session, inconsistency_id)
    if anchor is None:
        anchor = await _inconsistency_by_prefix(session, inconsistency_id)
    if anchor is None or anchor.status is not Status.INCONSISTENCY:
        raise ValueError(
            f"unknown inconsistency id: {inconsistency_id!r} — pass the id of an "
            "open INCONSISTENCY particle (a contested badge's inconsistency_id)"
        )
    # The §6.6 constructor writes the conflicting pair as the FIRST TWO
    # PARTICLE-type provenance refs (the cascade resolver reads them as
    # particle A and B); a later PARTICLE-typed ref can be the trigger ref of
    # a derived-particle conflict and is not a disputant.
    disputant_ids = [
        r.corpus_entry_id for r in anchor.provenance if r.type == ProvenanceRefType.PARTICLE
    ][:2]
    disputants = await get_particles_by_ids(session, disputant_ids)
    missing_disputants = len(disputant_ids) - len(disputants)

    particles: dict[str, Particle] = {anchor.id: anchor}
    hit_ids: set[str] = {anchor.id}
    hit_rank: dict[str, int] = {anchor.id: 0}
    # Disputants rank in constructor order (A then B), after the anchor.
    for rank, pid in enumerate(disputant_ids, start=1):
        p = disputants.get(pid)
        if p is None:
            continue
        particles[p.id] = p
        hit_ids.add(p.id)
        hit_rank[p.id] = rank

    hop_by_subject: dict[str, int] = {}
    for p in particles.values():
        for sid in sorted(p.subject_ids):
            hop_by_subject.setdefault(sid, 0)

    excluded_undatable = 0
    for sid in sorted(hop_by_subject):
        visible, undatable = await _fetch_subject_particles(
            session, sid, view, include_non_asserted
        )
        excluded_undatable += undatable
        for pid, p in visible.items():
            particles.setdefault(pid, p)
    return (
        particles,
        hop_by_subject,
        hit_ids,
        hit_rank,
        excluded_undatable,
        anchor.id,
        missing_disputants,
    )


async def _gather_projection_scope(
    session: AsyncSession,
    manifest_path: str,
    section_name: str,
    view: AsOfView | None,
    include_non_asserted: bool,
) -> tuple[dict[str, Particle], dict[str, int], set[str], dict[str, int], int, str]:
    """A projection section's selection rendered as a graph.

    ``section_name`` addresses a derived section by its ``region`` id first,
    then by exact (case-insensitive) title. The selection is the section's
    **deterministic** drift-gate selection (``use_embeddings=False``, the same
    key-free ranking ``project --check`` snapshots) so the render is
    reproducible without an API key. Selected particles are the foreground
    set; their subjects render at hop 0 with lens-visible incidental cargo,
    exactly like query scope.

    Returns ``(particles, hop_by_subject, hit_ids, hit_rank,
    excluded_undatable, resolved_section_name)``.
    """
    from pathlib import Path

    from particles.operations.projection.manifest import DerivedSection, load_manifest
    from particles.operations.projection.project import select_section_particles

    try:
        doc = load_manifest(Path(manifest_path))
    except FileNotFoundError:
        raise ValueError(f"unknown manifest: {manifest_path!r} (no such file)") from None
    except ValueError as exc:
        raise ValueError(f"invalid manifest {manifest_path!r}: {exc}") from exc

    derived = [s for s in doc.sections if isinstance(s, DerivedSection)]
    target = next((s for s in derived if s.region == section_name), None)
    if target is None:
        target = next((s for s in derived if s.title.lower() == section_name.lower()), None)
    if target is None:
        names = "; ".join(s.region or s.title for s in derived) or "(no derived sections)"
        raise ValueError(
            f"unknown section {section_name!r} in {manifest_path!r} — derived sections: {names}"
        )

    selected = await select_section_particles(session, target)

    particles: dict[str, Particle] = {}
    hit_ids: set[str] = set()
    hit_rank: dict[str, int] = {}
    hop_by_subject: dict[str, int] = {}
    for rank, (p, _eff) in enumerate(selected):
        particles[p.id] = p
        hit_ids.add(p.id)
        hit_rank[p.id] = rank
        for sid in p.subject_ids:
            hop_by_subject.setdefault(sid, 0)

    excluded_undatable = 0
    for sid in sorted(hop_by_subject):
        visible, undatable = await _fetch_subject_particles(
            session, sid, view, include_non_asserted
        )
        excluded_undatable += undatable
        for pid, p in visible.items():
            particles.setdefault(pid, p)
    return (
        particles,
        hop_by_subject,
        hit_ids,
        hit_rank,
        excluded_undatable,
        target.region or target.title,
    )


async def _extend_history(
    session: AsyncSession,
    particles: dict[str, Particle],
    view: AsOfView | None,
) -> set[str]:
    """Pull supersession-chain ghosts into ``particles``; return their ids.

    Walks each in-scope particle's ``supersedes`` ancestor chain (and the
    successor of each retired in-scope particle) via the forward pointer.
    Ghosts render but never expand the neighbourhood. On an as-of lens this
    is "history as known at T": anything asserted after T stays out, and a
    chain member that was *believed* at T joins the current surface rather
    than the ghosts.
    """
    ghosts: set[str] = set()
    queue = list(particles.values())
    seen = set(particles)
    while queue:
        p = queue.pop()
        for related in await _chain_neighbours(session, p):
            if related.id in seen:
                continue
            seen.add(related.id)
            if view is not None and ensure_utc(related.asserted_at) > view.as_of:
                continue
            retired_at = await get_retired_at(session, related.id)
            visible, _excl = _currently_visible(related, retired_at, view)
            if visible:
                particles[related.id] = related
                queue.append(related)
                continue
            if not _history_eligible(related):
                continue
            particles[related.id] = related
            ghosts.add(related.id)
            queue.append(related)
    return ghosts


async def _chain_neighbours(session: AsyncSession, p: Particle) -> list[Particle]:
    """The supersession-chain steps from one particle (predecessor + successor)."""
    out: list[Particle] = []
    if p.supersedes is not None:
        predecessor = await get_particle(session, p.supersedes)
        if predecessor is not None:
            out.append(predecessor)
    if p.status is not Status.ACTIVE:
        successor = await get_superseding_particle(session, p.id)
        if successor is not None:
            out.append(successor)
    return out
