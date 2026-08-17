"""structural claim filters — the Engine composition.

The deterministic modes of the one query surface (§2.1): the flags-only
listing, the ``count`` / ``group_by`` aggregates, and the predicate-vocabulary
listing. No embedding, no LLM call, on any path in this module — ordering is
``effective_confidence`` alone (tie: ``asserted_at`` descending), computed
through the same scoring kernel as every other read surface
(:mod:`.effective_confidence` — no new formula).

The pure term normalizer and matcher live in the Client layer
(``particles/core/claims.py``); this module owns what needs the store: scan
composition over the claim-carrying candidate set, aggregate assembly, and
the §2.6 coverage footer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import median

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.claims import ClaimFilters, ClaimMatch, match_claim, predicate_vocabulary
from particles.core.schema import (
    AggregateBucket,
    AsOfNote,
    ClaimCoverage,
    Particle,
    PredicateInfo,
    QueryRequest,
    QueryResponse,
    StructuralAggregate,
    StructuralGroupBy,
    StructuredClaim,
    TermKind,
)
from particles.core.stance import has_stance_marker
from particles.db import DEFAULT_STORE, StoreHandle, session_scope
from particles.extraction.polarity import is_non_asserted
from particles.extraction.scope import is_excluded_document_meta
from particles.operations.version_guard import assert_store_schema_current
from particles.store.extractor_store import get_trust_weight_map, populate_trust_cache
from particles.store.particle_store import (
    count_structured_claim_coverage,
    get_active_particles_with_claims,
    get_particles_with_claims_as_of,
)

from .as_of import load_as_of_view
from .decay_policy import DecayPolicy, load_decay_policy
from .effective_confidence import score_effective_confidence
from .source_trust import TrustPolicy, load_trust_policy

log = logging.getLogger(__name__)

#: One matched row: the particle, its claim, and its effective confidence.
ClaimRow = tuple[Particle, StructuredClaim, float]

#: Bucket key for a claim whose subject resolves to no Subject at all.
_NO_SUBJECT_KEY = "(no subject)"


def claim_filters_from_request(request: QueryRequest) -> ClaimFilters:
    """The request's §2.2 filter flags as one immutable Client-layer filter set."""
    return ClaimFilters(
        predicate=request.predicate,
        object_eq=request.object_eq,
        object_gt=request.object_gt,
        object_lt=request.object_lt,
        object_contains=request.object_contains,
    )


def coverage_line(coverage: ClaimCoverage) -> str:
    """Render the §2.6 coverage footer.

    Surfaces print this on every structural-filter result: absence of a hit
    must never be mistaken for absence of a belief — the filter only ever saw
    the annotated fraction of the store.
    """
    pct = 100.0 * coverage.with_claims / coverage.active_total if coverage.active_total else 0.0
    return (
        f"Matched against the {coverage.with_claims} of {coverage.active_total} "
        f"ACTIVE particles carrying a structured claim (store coverage {pct:.0f}%)."
    )


def disclosure_lines(coverage: ClaimCoverage) -> list[str]:
    """The §2.2 / §2.5 explicit-exclusion disclosures, when any row was excluded."""
    lines: list[str] = []
    if coverage.not_normalizable_excluded:
        lines.append(
            f"{coverage.not_normalizable_excluded} claim(s) excluded from the "
            "object comparison because their object would not normalize to a "
            "comparable value."
        )
    if coverage.below_min_effective_confidence:
        lines.append(
            f"{coverage.below_min_effective_confidence} claim(s) excluded by "
            "min_effective_confidence."
        )
    return lines


def _tz_aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


@dataclass
class _Gathered:
    """One store's filtered, scored structural candidate set."""

    rows: list[ClaimRow] = field(default_factory=list)
    not_comparable: int = 0
    as_of_notes: dict[str, AsOfNote] = field(default_factory=dict)
    excluded_undatable: int = 0
    active_total: int = 0
    with_claims: int = 0


async def _candidate_claims(
    session: AsyncSession, request: QueryRequest
) -> tuple[list[tuple[Particle, StructuredClaim]], dict[str, AsOfNote], int]:
    """Load and default-filter one store's claim-carrying candidates.

    Applies the same default exclusions and filters as the semantic path's
    ``_gather_scored`` (DOCUMENT_META, non-asserted, stance markers,
    uncertainty/modality filters, tags, recency window) so the structural
    modes see the same candidate universe minus the embedding requirement.
    """
    as_of_notes: dict[str, AsOfNote] = {}
    excluded_undatable = 0
    if request.as_of is None:
        particles = await get_active_particles_with_claims(
            session, request.min_confidence, subject_id=request.subject_id
        )
    else:
        view = await load_as_of_view(session, request.as_of)
        widened = await get_particles_with_claims_as_of(
            session, request.as_of, request.min_confidence, subject_id=request.subject_id
        )
        particles = []
        for p, stored_retired_at in widened:
            evaluation = view.evaluate(p, stored_retired_at)
            if evaluation.excluded_undatable:
                excluded_undatable += 1
                continue
            if not evaluation.visible:
                continue
            particles.append(p)
            if evaluation.note is not None:
                as_of_notes[p.id] = evaluation.note

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
    if request.recency_window_days:
        reference_now = request.as_of if request.as_of is not None else datetime.now(UTC)
        cutoff = reference_now - timedelta(days=request.recency_window_days)
        particles = [p for p in particles if _tz_aware(p.asserted_at) >= cutoff]

    pairs: list[tuple[Particle, StructuredClaim]] = []
    for p in particles:
        if p.structured_claim is not None:  # guaranteed by the fetch; narrows for mypy
            pairs.append((p, p.structured_claim))
    return pairs, as_of_notes, excluded_undatable


async def _gather_store(
    session: AsyncSession,
    request: QueryRequest,
    *,
    trust_policy: TrustPolicy | None = None,
    decay_policy: DecayPolicy | None = None,
    populate_cache: bool = True,
) -> _Gathered:
    """Filter and score one store's claims; used by both the single-store and
    federated paths (federation passes the viewer's policies)."""
    pairs, as_of_notes, excluded_undatable = await _candidate_claims(session, request)
    filters = claim_filters_from_request(request)
    matched: list[tuple[Particle, StructuredClaim]] = []
    not_comparable = 0
    if filters:
        for p, claim in pairs:
            outcome = match_claim(claim, filters)
            if outcome is ClaimMatch.MATCHED:
                matched.append((p, claim))
            elif outcome is ClaimMatch.NOT_COMPARABLE:
                not_comparable += 1
    else:
        matched = pairs

    eff = await score_effective_confidence(
        session,
        [p for p, _ in matched],
        trust_policy,
        decay_policy=decay_policy,
        populate_cache=populate_cache,
        now=request.as_of,
    )
    # §9.3 step 5 parity with the semantic path: min_confidence filters on
    # *effective* confidence; the SQL raw-value filter in the fetch
    # was the superset prefilter.
    rows = [(p, claim, eff[p.id]) for p, claim in matched if eff[p.id] >= request.min_confidence]
    counts = await count_structured_claim_coverage(session)
    return _Gathered(
        rows=rows,
        not_comparable=not_comparable,
        as_of_notes=as_of_notes,
        excluded_undatable=excluded_undatable,
        active_total=counts["active"],
        with_claims=counts["annotated"],
    )


def _coverage(gathered: _Gathered, *, below_min: int = 0) -> ClaimCoverage:
    return ClaimCoverage(
        active_total=gathered.active_total,
        with_claims=gathered.with_claims,
        matched=len(gathered.rows),
        not_normalizable_excluded=gathered.not_comparable,
        below_min_effective_confidence=below_min,
    )


def _distribution(effs: list[float]) -> tuple[float | None, float | None, float | None]:
    if not effs:
        return None, None, None
    return min(effs), median(effs), max(effs)


def _subject_bucket_keys(particle: Particle, claim: StructuredClaim) -> list[str]:
    """§2.5 ``--group-by subject`` resolution: ``claim.subject_id`` when
    present, else the particle's ``particle_subjects`` link (a
    claim-level ``None`` is weaker, not wrong). A multi-subject particle's
    claim counts in each linked subject's bucket."""
    if claim.subject_id is not None:
        return [claim.subject_id]
    if particle.subject_ids:
        return list(particle.subject_ids)
    return [_NO_SUBJECT_KEY]


async def _build_buckets(
    rows: list[ClaimRow],
    group_by: StructuralGroupBy,
    session: AsyncSession | None,
) -> list[AggregateBucket]:
    grouped: dict[str, list[float]] = {}
    for particle, claim, eff in rows:
        if group_by is StructuralGroupBy.SUBJECT:
            keys = _subject_bucket_keys(particle, claim)
        elif group_by is StructuralGroupBy.PREDICATE:
            keys = [claim.predicate.value]
        else:
            keys = [claim.object.value]
        for key in keys:
            grouped.setdefault(key, []).append(eff)

    labels: dict[str, str] = {}
    if group_by is StructuralGroupBy.SUBJECT and session is not None:
        from particles.store.subject_store import get_subject

        for key in grouped:
            if key == _NO_SUBJECT_KEY:
                continue
            subject = await get_subject(session, key)
            if subject is not None:
                labels[key] = subject.canonical_name

    buckets: list[AggregateBucket] = []
    for key, effs in grouped.items():
        buckets.append(
            AggregateBucket(
                key=key,
                label=labels.get(key),
                claim_count=len(effs),
                min_effective_confidence=min(effs),
                median_effective_confidence=float(median(effs)),
                max_effective_confidence=max(effs),
            )
        )
    buckets.sort(key=lambda b: (-b.claim_count, b.key))
    return buckets


async def _assemble(
    request: QueryRequest,
    gathered: _Gathered,
    session: AsyncSession | None,
) -> QueryResponse:
    """Build the deterministic listing or aggregate response from scored rows.

    ``session`` resolves subject labels for ``--group-by subject``; the
    federated path passes ``None`` (subject ids are store-local)
    and buckets keep their raw keys.
    """
    if request.is_aggregate:
        below_min = 0
        rows = gathered.rows
        if request.min_effective_confidence is not None:
            floor = request.min_effective_confidence
            kept = [row for row in rows if row[2] >= floor]
            below_min = len(rows) - len(kept)
            rows = kept
        low, mid, high = _distribution([eff for _, _, eff in rows])
        buckets: list[AggregateBucket] = []
        if request.group_by is not None:
            buckets = await _build_buckets(rows, request.group_by, session)
        aggregate = StructuralAggregate(
            claim_count=len(rows),
            min_effective_confidence=low,
            median_effective_confidence=mid,
            max_effective_confidence=high,
            group_by=request.group_by,
            buckets=buckets,
        )
        if request.group_by is not None:
            answer = (
                f"{len(rows)} claims match, in {len(buckets)} {request.group_by.value} bucket(s)."
            )
        else:
            answer = f"{len(rows)} claims match."
        return QueryResponse(
            answer=answer,
            particles=[],
            effective_confidences=[],
            structural_aggregate=aggregate,
            claim_coverage=_coverage(gathered, below_min=below_min),
            as_of=request.as_of,
            as_of_excluded_undatable=gathered.excluded_undatable,
        )

    # Deterministic listing (§2.1 mode three): effective_confidence descending,
    # tie asserted_at descending — no similarity term exists on this path.
    ordered = sorted(
        gathered.rows,
        key=lambda row: (row[2], _tz_aware(row[0].asserted_at)),
        reverse=True,
    )
    top = ordered[: request.top_k]
    matched_total = len(ordered)
    answer = f"{matched_total} claims matched the structural filter."
    if matched_total > len(top):
        answer += f" Showing the top {len(top)} by effective confidence."
    as_of_notes: list[AsOfNote | None] = (
        [gathered.as_of_notes.get(p.id) for p, _, _ in top] if request.as_of is not None else []
    )
    return QueryResponse(
        answer=answer,
        particles=[p for p, _, _ in top],
        effective_confidences=[eff for _, _, eff in top],
        claim_coverage=_coverage(gathered),
        as_of=request.as_of,
        as_of_notes=as_of_notes,
        as_of_excluded_undatable=gathered.excluded_undatable,
    )


async def _predicates_listing(session: AsyncSession, request: QueryRequest) -> QueryResponse:
    """§2.2 ``--predicates``: the distinct predicate terms with kind and count."""
    pairs, _notes, excluded_undatable = await _candidate_claims(session, request)
    vocabulary = [
        PredicateInfo(value=value, kind=TermKind(kind), claim_count=n)
        for value, kind, n in predicate_vocabulary(claim for _, claim in pairs)
    ]
    counts = await count_structured_claim_coverage(session)
    coverage = ClaimCoverage(
        active_total=counts["active"],
        with_claims=counts["annotated"],
        matched=len(pairs),
    )
    return QueryResponse(
        answer=(
            f"{len(vocabulary)} distinct predicate term(s) across {len(pairs)} structured claims."
        ),
        particles=[],
        effective_confidences=[],
        predicate_vocabulary=vocabulary,
        claim_coverage=coverage,
        as_of=request.as_of,
        as_of_excluded_undatable=excluded_undatable,
    )


async def structural_query(session: AsyncSession, request: QueryRequest) -> QueryResponse:
    """Run one store's deterministic structural mode.

    Dispatched from :func:`particles.operations.query.main.query` when the
    request is purely structural. Zero embedding and zero LLM calls on every
    path through here.
    """
    await assert_store_schema_current(session)
    if request.list_predicates:
        return await _predicates_listing(session, request)
    gathered = await _gather_store(session, request)
    return await _assemble(request, gathered, session)


async def structural_query_federated(
    stores: list[StoreHandle],
    request: QueryRequest,
    viewer_store: StoreHandle | None = None,
) -> QueryResponse:
    """The structural modes across several stores under one viewer's lens.

    The composition rule unchanged: the viewer's trust and decay
    policies score every store's candidates; coverage counts are summed.
    Subject labels are omitted (subject ids are store-local).
    """
    if not stores:
        stores = [DEFAULT_STORE]
    viewer = viewer_store or stores[0]

    async with session_scope(viewer) as viewer_session:
        await assert_store_schema_current(viewer_session)
        populate_trust_cache(await get_trust_weight_map(viewer_session))
        trust_policy = await load_trust_policy(viewer_session)
        decay_policy = await load_decay_policy(viewer_session)

    if request.list_predicates:
        merged_pairs: list[tuple[Particle, StructuredClaim]] = []
        active_total = with_claims = 0
        for store in stores:
            async with session_scope(store) as s:
                await assert_store_schema_current(s)
                pairs, _notes, _undatable = await _candidate_claims(s, request)
                merged_pairs.extend(pairs)
                counts = await count_structured_claim_coverage(s)
                active_total += counts["active"]
                with_claims += counts["annotated"]
        vocabulary = [
            PredicateInfo(value=value, kind=TermKind(kind), claim_count=n)
            for value, kind, n in predicate_vocabulary(claim for _, claim in merged_pairs)
        ]
        return QueryResponse(
            answer=(
                f"{len(vocabulary)} distinct predicate term(s) across "
                f"{len(merged_pairs)} structured claims."
            ),
            particles=[],
            effective_confidences=[],
            predicate_vocabulary=vocabulary,
            claim_coverage=ClaimCoverage(
                active_total=active_total, with_claims=with_claims, matched=len(merged_pairs)
            ),
            as_of=request.as_of,
        )

    merged = _Gathered()
    for store in stores:
        async with session_scope(store) as s:
            await assert_store_schema_current(s)
            gathered = await _gather_store(
                s,
                request,
                trust_policy=trust_policy,
                decay_policy=decay_policy,
                populate_cache=False,
            )
            merged.rows.extend(gathered.rows)
            merged.not_comparable += gathered.not_comparable
            merged.as_of_notes.update(gathered.as_of_notes)
            merged.excluded_undatable += gathered.excluded_undatable
            merged.active_total += gathered.active_total
            merged.with_claims += gathered.with_claims
    return await _assemble(request, merged, None)
