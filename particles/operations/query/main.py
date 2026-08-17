"""§9.3 Query operation — orchestrator.

Retrieves relevant particles and generates a natural language response.
Does NOT block on extraction — coverage gaps are disclosed instead.

``query`` runs against one store. ``query_federated`` fans the
same retrieval out across several stores, merges the candidates, reranks them
under the **viewer's** trust policy, and generates one answer. Both share the
``_gather_scored`` (retrieve + score) and ``_build_response`` (respond) stages,
so the two paths can never drift in how they score or render.

The heavy lifting lives in the sibling submodules:

  - ``.rank``         semantic embedding, co-evidential collapse, source-key helper.
  - ``.source_info``  batch provenance lookup for recency decay + trust inputs.
  - ``.source_trust`` per-query TrustPolicy snapshot — source_trust_rank.
  - ``.gaps``         corpus-level and subject-level coverage-gap detection.
  - ``.respond``      audience-aware NL response generation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.claims import ClaimMatch, match_claim
from particles.core.schema import (
    SCHEMA_VERSION,
    AsOfNote,
    ClaimCoverage,
    ContestedBadge,
    ContestednessReading,
    CoverageGapKind,
    Particle,
    ParticleType,
    QueryRequest,
    QueryResponse,
    RelevanceNote,
    StancePosition,
    SubjectCoverageGap,
)
from particles.core.scoring.confidence import compute_effective_confidence
from particles.core.stance import has_stance_marker
from particles.db import DEFAULT_STORE, StoreHandle, session_scope
from particles.embeddings import cosine_similarity
from particles.extraction.polarity import is_non_asserted
from particles.extraction.registry import CODE_SYMBOL_EXTRACTOR_IDS
from particles.extraction.scope import is_excluded_document_meta
from particles.observability import traced
from particles.operations.abstraction import stale_support_discounts
from particles.operations.version_guard import assert_store_schema_current
from particles.store.extractor_store import (
    get_cached_trust_weight,
    get_trust_weight_map,
    populate_trust_cache,
)
from particles.store.particle_store import (
    count_structured_claim_coverage,
    get_active_particles_with_embeddings,
    get_particles_with_embeddings_as_of,
)

from .as_of import load_as_of_view
from .decay_policy import DecayPolicy, load_decay_policy
from .gaps import _find_coverage_gaps, _find_subject_coverage_gaps
from .rank import RANKING_DEGRADED_NO_ENCODER, _collapse_co_evidential_top_k, _embed
from .respond import (
    _generate_response,
    fallback_listing,
    generation_error_reason,
    strip_refusal_marker,
)
from .source_info import load_source_rows
from .source_trust import TrustPolicy, load_trust_policy
from .structural import claim_filters_from_request, structural_query, structural_query_federated

log = logging.getLogger(__name__)

_SCHEMA_VERSION_CURRENT = SCHEMA_VERSION

#: (particle, cosine_sim, effective_confidence)
Scored = tuple[Particle, float, float]


def _combined(sim: float, eff_conf: float) -> float:
    """Combined ranking score: similarity_weight × sim + confidence_weight × eff_conf.

    Weights come from ``config.query`` (defaults 0.6 / 0.4); read at call time
    per the config rule (``get_config()`` is a cached singleton).
    """
    cfg = get_config().query
    return cfg.similarity_weight * sim + cfg.confidence_weight * eff_conf


def _is_code_symbol(particle: Particle) -> bool:
    """True if the particle is a structured code-symbol reading,
    keyed on the extractor id already stamped on ``extractor_ref``."""
    name = particle.extractor_ref.name if particle.extractor_ref else None
    return name in CODE_SYMBOL_EXTRACTOR_IDS


def _rank_score(
    s: Scored,
    code_symbol_weight: float = 1.0,
    precedence_demoted: frozenset[str] = frozenset(),
) -> float:
    """Combined score with the per-type, code-symbol, and
    document-precedence rank weights applied.

    A NARRATIVE's combined score is multiplied by ``query.narrative_rank_weight``
    so its broad, high-confidence label doesn't dominate top-k (§Harder).
    A code-symbol (docstring) particle's score is multiplied by
    ``code_symbol_weight`` (default ``1.0`` = inert) so a projection conceptual
    section can lean away from code trivia. A particle whose id is in
    ``precedence_demoted`` — the recency-loser of a *detected* conflict
    — is multiplied by ``document_precedence.rank_penalty`` so the later authored
    decision surfaces first. All three are rank-time only — the reported
    ``effective_confidence`` (``s[2]``) is untouched, so confidence stays truthful
    and ``min_confidence`` is not perturbed.
    """
    particle, sim, eff_conf = s
    score = _combined(sim, eff_conf)
    if particle.particle_type == ParticleType.NARRATIVE:
        score *= get_config().query.narrative_rank_weight
    if code_symbol_weight != 1.0 and _is_code_symbol(particle):
        score *= code_symbol_weight
    if particle.id in precedence_demoted:
        score *= get_config().document_precedence.rank_penalty
    return score


async def _expand_narratives(
    session: AsyncSession, particles: list[Particle]
) -> dict[str, list[Particle]]:
    """Map each NARRATIVE hit to its ``SEQUENCE_IN`` constituents.

    One ``get_narrative_sequence`` fetch per narrative hit; non-narrative hits and
    narratives with no constituents are absent from the result.
    """
    from particles.operations.narrative import get_narrative_sequence

    out: dict[str, list[Particle]] = {}
    for p in particles:
        if p.particle_type != ParticleType.NARRATIVE:
            continue
        seq = await get_narrative_sequence(session, p.id)
        if seq:
            out[p.id] = seq
    return out


async def _gather_scored(
    session: AsyncSession,
    request: QueryRequest,
    query_emb: np.ndarray | None,
    trust_policy: TrustPolicy,
    decay_policy: DecayPolicy,
) -> tuple[
    list[Scored], dict[str, datetime | None], dict[str, AsOfNote], int, tuple[int, int] | None
]:
    """Load, filter, and score one store's candidate particles.

    Does **not** sort, truncate, collapse, populate the trust cache, or load
    the trust / decay policies — the caller owns those (the extractor trust
    cache, the ``trust_policy``, and the ``decay_policy`` are
    all loaded once from the viewer's store in federation, so the viewer's
    policies apply to every store's candidates). Returns the scored 3-tuples,
    a ``particle_id -> pub_at`` map for recency display, the as-of surfaces — a ``particle_id -> AsOfNote`` map for hits retired after
    the reference instant, plus the fail-closed undatable-exclusion count
    (both empty/zero when ``request.as_of`` is unset) — and the claim-prefilter stats ``(matched, not_comparable)``, ``None`` when the
    request carries no structural claim filter.

    with ``as_of`` set, the **temporal** quantities move to T —
    the candidate set becomes "believed at T" (the per-store ``AsOfView`` is
    loaded here, once per store, so federation gets each store's own
    successor/event maps), the recency window cuts relative to T, and decay is
    evaluated with ``now=T``. **Judgment** quantities — the trust policy and
    the extractor trust cache — stay current: trust is the viewer's present
    judgment applied to historical beliefs.
    """
    as_of_notes: dict[str, AsOfNote] = {}
    excluded_undatable = 0
    if request.as_of is None:
        # Fast path — byte-for-byte today's read behaviour (ACTIVE-only plan).
        candidates = await get_active_particles_with_embeddings(
            session, request.min_confidence, subject_id=request.subject_id
        )
    else:
        view = await load_as_of_view(session, request.as_of)
        widened = await get_particles_with_embeddings_as_of(
            session, request.as_of, request.min_confidence, subject_id=request.subject_id
        )
        candidates = []
        for p, emb, stored_retired_at in widened:
            evaluation = view.evaluate(p, stored_retired_at)
            if evaluation.excluded_undatable:
                excluded_undatable += 1
                continue
            if not evaluation.visible:
                continue
            candidates.append((p, emb))
            if evaluation.note is not None:
                as_of_notes[p.id] = evaluation.note

    # Exclude DOCUMENT_META particles from the default factual
    # surface unless the caller explicitly opts in.
    if not request.include_document_meta:
        candidates = [
            (p, emb) for p, emb in candidates if not is_excluded_document_meta(p.properties)
        ]

    # Exclude non-asserted particles — polarity DECLINED / HYPOTHETICAL
    # (cap. 1) — from the default factual surface unless the caller
    # explicitly opts in. A document's rejected / deferred / counterfactual
    # prose is real (so confidence is untouched) but is not current truth.
    if not request.include_non_asserted:
        candidates = [(p, emb) for p, emb in candidates if not is_non_asserted(p.properties)]

    # keep stance particles out of the factual top-k. A stance is a
    # FALSIFIABLE claim about an agent's attitude (role-marked by its outbound
    # ENDORSES/DISPUTES edge, cheaply detected by its stance:holder marker); it
    # is surfaced via the query-time agreement distribution, not as a factual hit.
    candidates = [(p, emb) for p, emb in candidates if not has_stance_marker(p)]

    # Filter by uncertainty_nature if requested
    if request.uncertainty_nature:
        candidates = [
            (p, emb) for p, emb in candidates if p.uncertainty_nature == request.uncertainty_nature
        ]

    # Filter by assertion_modality if requested. Unset returns every
    # modality — non-FALSIFIABLE particles stay queryable (the journal
    # feelings-recall case); they are only kept out of truth-arbitration.
    if request.assertion_modality is not None:
        candidates = [
            (p, emb) for p, emb in candidates if p.assertion_modality == request.assertion_modality
        ]

    # Filter by tag (subtree-expanded across all active taxonomies;
    # also up-expanded over the parent chain when include_ancestors)
    if request.tags:
        from particles.store.taxonomy_store import (
            expand_tags,
            get_particle_ids_for_tags,
        )

        expanded = await expand_tags(
            session, request.tags, include_ancestors=request.include_ancestors
        )
        tagged_ids = await get_particle_ids_for_tags(session, expanded)
        candidates = [(p, emb) for p, emb in candidates if p.id in tagged_ids]

    # structural claim filters **prefilter** the candidate set,
    # exactly as the subject / tag filters above do. Cosine ranking, the
    # effective-confidence composition, and the LLM response below are
    # untouched (§2.4 — filters intersect; scores never fuse). Rows whose
    # object would not normalize against a gt/lt bound are counted for the
    # §2.2 disclosure, never silently dropped.
    claim_stats: tuple[int, int] | None = None
    if request.has_claim_filters:
        filters = claim_filters_from_request(request)
        kept: list[tuple[Particle, np.ndarray[Any, np.dtype[np.float32]]]] = []
        claim_not_comparable = 0
        for p, emb in candidates:
            if p.structured_claim is None:
                continue
            outcome = match_claim(p.structured_claim, filters)
            if outcome is ClaimMatch.MATCHED:
                kept.append((p, emb))
            elif outcome is ClaimMatch.NOT_COMPARABLE:
                claim_not_comparable += 1
        candidates = kept
        claim_stats = (len(kept), claim_not_comparable)

    # Filter by recency window if requested: the window cuts
    # relative to the reference instant — as of T, "the last N days" means the
    # N days before T.
    if request.recency_window_days:
        reference_now = request.as_of if request.as_of is not None else datetime.now(UTC)
        cutoff = reference_now - timedelta(days=request.recency_window_days)
        candidates = [
            (p, emb)
            for p, emb in candidates
            if (
                p.asserted_at.replace(tzinfo=UTC) if p.asserted_at.tzinfo is None else p.asserted_at
            )
            >= cutoff
        ]

    # Batch-load (content_published_at, source_type, entry_id, uri_r,
    # author_id) for all candidate particles — recency decay + the trust-policy
    # inputs, including the §6.4 AUTHOR tier.
    source_info = await load_source_rows(session, [p for p, _ in candidates])

    # keep-ACTIVE-and-discount for derived particles whose
    # premise set changed since the last revalidation. Applied before the
    # min_confidence filter so a discounted abstraction ranks (and filters)
    # at its honest score.
    support_discounts = await stale_support_discounts(session, [p for p, _ in candidates])

    scored: list[Scored] = []
    pub_at_by_id: dict[str, datetime | None] = {}
    for p, emb in candidates:
        extractor_id = p.extractor_ref.name if p.extractor_ref else "general-extractor"
        trust_weight = get_cached_trust_weight(extractor_id)
        pub_at, source_type, entry_id, uri_r, author_id = source_info.get(
            p.id, (None, "", None, None, None)
        )
        # decay resolves through the viewer's composed decay policy
        # (local config + adopted-lens decay_rules); identical to the global
        # config when no decay-bearing lens is adopted:
        # evaluated at the reference instant — content that was fresh at T
        # scores fresh (``now=None`` is the kernel's wall-clock default).
        decay = decay_policy.recency_factor(pub_at, source_type, uri_r, now=request.as_of)
        # absence of trust policy is strictly neutral (None → 1.0).
        rank = trust_policy.evaluate(entry_id, source_type, uri_r, author_id)
        eff_conf = compute_effective_confidence(
            p.confidence.value,
            extractor_trust_weight=trust_weight,
            source_trust_rank=1.0 if rank is None else rank,
            recency_factor=decay,
            calibration_source=p.confidence.calibration_source,
        ) * support_discounts.get(p.id, 1.0)

        # §9.3 step 5: min_confidence filters on *effective* confidence
        #. The SQL raw-value filter in
        # get_active_particles_with_embeddings stays as a superset prefilter —
        # every factor above is ≤ 1.0 (extractor trust is demotion-only per
        # rank and decay clamp at 1.0), so effective ≤ raw and the
        # prefilter can never drop a particle this check would keep.
        if eff_conf < request.min_confidence:
            continue

        # the single normative similarity primitive — cosine over
        # L2-normalized vectors, clamped to [0, 1]. §9.3 ranks by sim × eff_conf.
        # Fall back to confidence ranking when there is no query embedding.
        sim = cosine_similarity(query_emb, emb) if query_emb is not None else eff_conf

        scored.append((p, sim, eff_conf))
        pub_at_by_id[p.id] = pub_at

    return scored, pub_at_by_id, as_of_notes, excluded_undatable, claim_stats


def _truncation_warning(scored: list[Scored], top: list[Scored], top_k: int) -> str | None:
    """Warn when the top-k cutoff likely excluded relevant particles.

    ``scored`` is the full sorted candidate list; ``top`` is the rendered
    (post-collapse) result. The thresholds (minimum score gap, near-cutoff
    margin and count) come from ``config.query``.
    """
    if not (top and len(scored) > top_k):
        return None
    cfg = get_config().query
    # Use the rank score (with the NARRATIVE demotion) so the warning
    # heuristic matches the order the result was actually sorted by.
    last_score = _rank_score(top[-1])
    next_score = _rank_score(scored[top_k])
    gap = last_score - next_score
    near_cutoff = sum(
        1 for s in scored[top_k:] if _rank_score(s) >= last_score - cfg.truncation_near_margin
    )
    if gap < cfg.truncation_min_gap or near_cutoff > cfg.truncation_near_count:
        # 200 is the QueryRequest.top_k schema ceiling, not a tuneable.
        suggested_k = min(top_k * 3, 200)
        return (
            f"top_k={top_k} may be too low: {near_cutoff} additional particles "
            f"scored within {cfg.truncation_near_margin} of the cutoff "
            f"(gap to next: {gap:.3f}). "
            f"Try --top-k {suggested_k} for a more complete answer."
        )
    return None


def _confidence_note(top_particles: list[Particle], top_eff_confs: list[float]) -> str:
    """Provisional-confidence disclosure for a ranked result (OVERCONFIDENCE GUARD, §6.3).

    Fires when the mean effective confidence is low, or when any top hit is
    uncalibrated — raw extractor output (``EXTRACTOR_DIRECT``) or a direct agent
    self-report (``AGENT_ASSERTED``), both the lowest, uncalibrated
    trust tier. Returns the empty string when neither applies.
    """
    from particles.core.scoring.confidence import CalibrationSource

    if not top_particles:
        return ""
    mean_eff = sum(top_eff_confs) / len(top_eff_confs)
    if mean_eff < 0.6:
        return (
            "⚠ Note: The mean confidence of retrieved particles is below 0.6. "
            "This knowledge base has not been fully validated for this topic."
        )
    uncalibrated = (CalibrationSource.EXTRACTOR_DIRECT, CalibrationSource.AGENT_ASSERTED)
    if any(p.confidence.calibration_source in uncalibrated for p in top_particles):
        return (
            "⚠ Note: Some retrieved particles are uncalibrated — raw extractor "
            "output (EXTRACTOR_DIRECT) or direct agent self-reports (AGENT_ASSERTED). "
            "Confidence values are provisional."
        )
    return ""


def _relevance_note(top: list[Scored], embeddings_used: bool) -> RelevanceNote | None:
    """The question-level relevance verdict for a rendered top-k.

    Reads the **raw** cosine similarity column of the ``Scored`` tuples — never
    the combined score, whose confidence term is the pollution being detected.
    ``None`` when no query embedding was in play (the sim column is then the
    ``eff_conf`` ranking fallback, not a relevance signal) or the result is
    empty. Computed after ranking, from the rendered top-k only, and feeds
    rendering + disclosure — never scores or order (discipline; the
    max is an aggregate but only over the rendered result, feeding a
    disclosure, the category permits).
    """
    if not embeddings_used or not top:
        return None
    floor = get_config().query.relevance_floor
    max_sim = max(sim for _, sim, _ in top)
    return RelevanceNote(max_similarity=max_sim, floor=floor, below_floor=max_sim < floor)


def _below_floor_answer(request: QueryRequest, note: RelevanceNote, hit_count: int) -> str:
    """The deterministic no-LLM refusal for a below-floor query.

    Server-built prose: cannot be argued out of by a crafted question, costs
    zero tokens, and is reproducible in tests. With ``as_of`` set the sentence
    keeps the tense.
    """
    detail = (
        f"(nearest belief similarity {note.max_similarity:.2f}, "
        f"below the relevance floor {note.floor:.2f})"
    )
    tail = (
        f" The {hit_count} nearest belief(s) are listed for transparency; "
        f"they are likely unrelated to the question."
    )
    if request.as_of is not None:
        return (
            f"As of {request.as_of.isoformat()}, the store held no beliefs "
            f"relevant to this question {detail}.{tail}"
        )
    return f"The store holds no beliefs relevant to this question {detail}.{tail}"


async def _build_response(
    request: QueryRequest,
    top: list[Scored],
    pub_at_by_id: dict[str, datetime | None],
    coverage_gaps: list[str],
    subject_coverage_gaps: list[SubjectCoverageGap],
    truncation_warning: str | None,
    agreement_distributions: list[list[StancePosition]] | None = None,
    agreement_caveat: str | None = None,
    contestedness: list[ContestednessReading] | None = None,
    contested: list[ContestedBadge | None] | None = None,
    narrative_constituents: dict[str, list[Particle]] | None = None,
    as_of_notes_by_id: dict[str, AsOfNote] | None = None,
    as_of_excluded_undatable: int = 0,
    claim_coverage: ClaimCoverage | None = None,
    embeddings_used: bool = True,
) -> QueryResponse:
    """Render the final NL answer + envelope from a ranked, collapsed result.

    Shared by ``query`` and ``query_federated`` so the response shape and the
    confidence / coverage notes are identical on both paths. ``narrative_constituents``
     maps each NARRATIVE hit to its ordered constituents; empty on the
    federated path (per-store expansion is out of scope). ``as_of_notes_by_id``
    / ``as_of_excluded_undatable`` carry the per-hit supersession
    crossings and the fail-closed disclosure count when the request set
    ``as_of``. ``embeddings_used`` says whether a query embedding
    was in play — the relevance verdict is inert without one (the sim column
    is then a ranking fallback, not a relevance signal).
    """
    # The structural modes never reach this renderer — a
    # question-less request was dispatched to ``structural_query`` upstream.
    if request.question is None:
        raise ValueError("_build_response requires a question (structural modes bypass it)")
    narrative_constituents = narrative_constituents or {}
    as_of_notes_by_id = as_of_notes_by_id or {}
    top_particles = [p for p, _, _ in top]
    top_eff_confs = [ec for _, _, ec in top]
    top_pub_ats = [pub_at_by_id.get(p.id) for p in top_particles]
    # the parallel-list pattern of effective_confidences — entry i
    # annotates particles[i]; None for a hit still ACTIVE today.
    as_of_notes: list[AsOfNote | None] = (
        [as_of_notes_by_id.get(p.id) for p in top_particles] if request.as_of is not None else []
    )

    # Overconfidence guard (§6.3): low mean confidence, or any uncalibrated hit.
    conf_note = _confidence_note(top_particles, top_eff_confs)

    coverage_note = ""
    if subject_coverage_gaps:
        scg = subject_coverage_gaps[0]
        if scg.kind == CoverageGapKind.SUBJECT_HAS_NO_PARTICLES:
            coverage_note = (
                f"⚠ Coverage note: Subject '{scg.subject_name}' exists in the registry "
                f"but has no extracted claim particles. This answer is based on zero evidence."
            )
        elif scg.kind == CoverageGapKind.SUBJECT_HAS_LOW_COVERAGE:
            coverage_note = (
                f"⚠ Coverage note: Only {scg.particle_count} particle(s) about "
                f"'{scg.subject_name}' were found. This answer may be incomplete."
            )

    # the question-level relevance verdict. Below the floor the
    # answer is deterministic — no LLM call — and the hits remain in the
    # envelope as nearest-but-likely-unrelated transparency.
    relevance = _relevance_note(top, embeddings_used)
    # with no encoder there is no semantic retrieval to disclose a
    # floor for — `sim` was aliased to effective confidence, so these are the
    # store's most-confident beliefs, not the ones about this question. Say so;
    # `relevance is None` alone is indistinguishable from the inert case.
    ranking_degraded = None if embeddings_used else RANKING_DEGRADED_NO_ENCODER
    answer_generation_error: str | None = None
    answer_refused = False
    if relevance is not None and relevance.below_floor:
        answer = _below_floor_answer(request, relevance, len(top_particles))
        answer_refused = True
    else:
        try:
            answer = await _generate_response(
                request.question,
                top_particles,
                top_eff_confs,
                request.audience,
                conf_note,
                coverage_note,
                narrative_constituents=narrative_constituents,
                as_of=request.as_of,
            )
            # the responder leads a non-bearing refusal with the
            # NO_RELEVANT_KNOWLEDGE marker — strip it, keep the prose, record
            # the machine-readable flag.
            answer, answer_refused = strip_refusal_marker(answer)
        except Exception as exc:
            # Disclosed degradation, never a quiet one (the honesty
            # posture): a billing/network/provider failure used to be silently
            # papered over with the raw particle listing presented as the
            # "answer". Keep the useful deterministic listing, but say what it
            # is — in the answer text itself (for plain-text consumers) and in
            # the machine-readable field (for UI banners).
            log.error("Response generation failed: %s", exc)
            answer_generation_error = generation_error_reason(exc)
            answer = (
                f"[Answer generation unavailable: {answer_generation_error}]\n"
                "Showing the retrieved beliefs instead — this is a listing, "
                "not an answer:\n" + fallback_listing(top_particles)
            )

    # the machine-readable field above serves UI banners, but a
    # plain-text consumer (a piped CLI run, an agent reading the answer string)
    # sees only `answer` — and generated prose over the store's most-confident
    # beliefs reads exactly like a good answer. Prefix it, the same way an
    # answer-generation failure is prefixed rather than quietly swapped.
    if ranking_degraded is not None:
        answer = f"[{ranking_degraded}]\n\n{answer}"

    # Under a refusal (either flavour) the truncation warning is incoherent —
    # it advises a larger top_k "for a more complete answer" to a question the
    # store cannot answer. Suppressed here so every consumer agrees.
    if answer_refused:
        truncation_warning = None

    return QueryResponse(
        answer=answer,
        answer_generation_error=answer_generation_error,
        ranking_degraded=ranking_degraded,
        answer_refused=answer_refused,
        particles=top_particles,
        effective_confidences=top_eff_confs,
        content_published_ats=top_pub_ats,
        coverage_gaps=coverage_gaps,
        subject_coverage_gaps=subject_coverage_gaps,
        truncation_warning=truncation_warning,
        agreement_distributions=agreement_distributions or [],
        agreement_caveat=agreement_caveat,
        contestedness=contestedness or [],
        contested=contested or [],
        narrative_constituents=narrative_constituents,
        as_of=request.as_of,
        as_of_notes=as_of_notes,
        as_of_excluded_undatable=as_of_excluded_undatable,
        claim_coverage=claim_coverage,
        relevance=relevance,
    )


async def _claim_coverage_for_stats(
    session: AsyncSession, claim_stats: tuple[int, int] | None
) -> ClaimCoverage | None:
    """Compose the coverage footer for a prefiltered semantic query."""
    if claim_stats is None:
        return None
    counts = await count_structured_claim_coverage(session)
    matched, not_comparable = claim_stats
    return ClaimCoverage(
        active_total=counts["active"],
        with_claims=counts["annotated"],
        matched=matched,
        not_normalizable_excluded=not_comparable,
    )


@traced("query")
async def query(
    session: AsyncSession,
    request: QueryRequest,
) -> QueryResponse:
    """Execute a semantic search over one store and generate a NL answer.

    A purely structural request (modes three and four — filters
    without a question, aggregates, or the predicate listing) dispatches to
    :func:`.structural.structural_query` instead: deterministic, no embedding,
    no LLM call.
    """
    if request.is_structural_mode:
        return await structural_query(session, request)
    # refuse to operate on a store with mismatched schema_version.
    await assert_store_schema_current(session)

    # Seed trust weight cache for this query run (no-op if already loaded)
    populate_trust_cache(await get_trust_weight_map(session))
    # snapshot this store's source-trust policy for the run
    trust_policy = await load_trust_policy(session)
    # snapshot this store's composed content-age decay policy
    decay_policy = await load_decay_policy(session)

    if request.question is None:  # unreachable: the structural dispatch above caught None
        raise ValueError("semantic query requires a question")
    query_emb = _embed(request.question)
    scored, pub_at_by_id, as_of_notes_by_id, as_of_excluded, claim_stats = await _gather_scored(
        session, request, query_emb, trust_policy, decay_policy
    )
    claim_coverage = await _claim_coverage_for_stats(session, claim_stats)

    # Rank by combined score, then collapse CO_EVIDENTIAL groups in the top-k
    # (§6.10) — see _collapse_co_evidential_top_k for the noisy-OR merge.
    scored.sort(key=_rank_score, reverse=True)
    top = scored[: request.top_k]
    top = await _collapse_co_evidential_top_k(session, top)

    truncation_warning = _truncation_warning(scored, top, request.top_k)
    coverage_gaps = await _find_coverage_gaps(session)
    subject_coverage_gaps = await _find_subject_coverage_gaps(session, request.subject_id, len(top))

    # expand any NARRATIVE hit to its SEQUENCE_IN constituents so the
    # answer (and structured response) reflect the memory's claims, not the label.
    narrative_constituents = await _expand_narratives(session, [p for p, _, _ in top])

    # optionally attach the per-result agreement distribution.
    agreement_distributions: list[list[StancePosition]] | None = None
    agreement_caveat: str | None = None
    if request.include_agreement:
        from .stance import AGREEMENT_CAVEAT, compute_stance_distribution

        agreement_distributions, has_any = await compute_stance_distribution(
            session, [p for p, _, _ in top], trust_policy
        )
        if has_any:
            agreement_caveat = AGREEMENT_CAVEAT

    # optionally attach the per-result contestedness reading — the
    # spread of effective confidence across the viewer's policy set. Absent
    # (empty) when fewer than two policies are configured (§3 degeneracy).
    contestedness: list[ContestednessReading] | None = None
    if request.include_contestedness:
        from .contestedness import compute_contestedness, load_member_policies

        members = await load_member_policies(session)
        contestedness = await compute_contestedness(session, [p for p, _, _ in top], members)

    # compose the default-on contested badge over the three bases
    # (stance / divergence / inconsistency), gated by the §7 kill switch. Read
    # the config at call time. Computed after ranking, from the
    # rendered top-k only — it can never feed scores or order (§4). When the
    # readings were already computed above they are reused ([] means
    # the divergence basis is absent, §3); otherwise the composer evaluates
    # divergence itself, free on the common zero-lens store.
    contested_badges: list[ContestedBadge | None] | None = None
    if get_config().contestedness.badge_enabled:
        from .contested import compute_contested_badges

        contested_badges = await compute_contested_badges(
            session, [p for p, _, _ in top], readings=contestedness
        )

    return await _build_response(
        request,
        top,
        pub_at_by_id,
        coverage_gaps,
        subject_coverage_gaps,
        truncation_warning,
        agreement_distributions,
        agreement_caveat,
        contestedness,
        contested_badges,
        narrative_constituents,
        as_of_notes_by_id=as_of_notes_by_id,
        as_of_excluded_undatable=as_of_excluded,
        claim_coverage=claim_coverage,
        embeddings_used=query_emb is not None,
    )


async def _precedence_demotions(
    session: AsyncSession,
    scored: list[Scored],
    pub_at_by_id: dict[str, datetime | None],
    conflict_pairs: set[frozenset[str]] | None,
) -> frozenset[str]:
    """Resolve the recency-loser set for a scored candidate list.

    Conflict-gated and config-gated: returns the empty set when
    ``document_precedence.enabled`` is false, when no ``conflict_pairs`` are
    supplied (the key-free deterministic projection path detects no conflicts —
    so the tie-break is inert and the order stays byte-stable), or when no pair
    has a comparable precedence key on both sides. Otherwise it reads each
    candidate's authored precedence key (ADR id ordinal via the genre-adapter seam + ``content_published_at`` fallback) and returns the
    recency-losers. Rank-time only — never mutates the store.
    """
    if not conflict_pairs or not get_config().document_precedence.enabled:
        return frozenset()

    from particles.corpus.store import get_document_supersession_map

    from .precedence import _source_entry_id, build_precedence_keys, precedence_demotions

    particles = [p for p, _, _ in scored]
    entry_ids = {eid for p in particles if (eid := _source_entry_id(p)) is not None}
    supersession_by_entry = await get_document_supersession_map(session, entry_ids)
    keys = build_precedence_keys(
        particles,
        pub_at_by_id=pub_at_by_id,
        supersession_by_entry=supersession_by_entry,
    )
    return frozenset(precedence_demotions(particles, conflict_pairs, keys))


async def retrieve_ranked(
    session: AsyncSession,
    request: QueryRequest,
    *,
    use_embeddings: bool = True,
    code_symbol_rank_weight: float = 1.0,
    conflict_pairs: set[frozenset[str]] | None = None,
    apply_utility: bool = False,
    apply_owner: bool = False,
) -> list[Scored]:
    """Select + rank one store's candidate particles — no NL answer generated.

    The selection half of :func:`query`: it runs the identical filter + scoring +
    co-evidential-collapse pipeline (``_gather_scored`` → ``_rank_score`` →
    ``_collapse_co_evidential_top_k``) and returns the ranked, collapsed top-k as
    ``Scored`` 3-tuples, but skips ``_generate_response``. Documentation
    projection needs the candidate set, not a prose answer, and must
    run **without an API key** in CI — so it passes ``use_embeddings=False``,
    which drops the semantic-similarity term and ranks purely by the §6.6
    recency-weighted effective confidence (deterministic, model-free). Sharing
    ``_gather_scored`` / ``_rank_score`` with :func:`query` means the two paths
    can never drift in *how* they score.

    ``code_symbol_rank_weight`` (default ``1.0`` = inert) demotes
    code-symbol (docstring) particles at rank time for a projection conceptual
    section, mirroring the NARRATIVE demotion.

    ``conflict_pairs`` (default ``None`` = inert) is the set of
    detected-conflict id pairs (each a 2-element ``frozenset``) from the
    The L-SEM-01 cross-source contradiction detector. When supplied (and
    ``document_precedence.enabled``), the recency-loser of each conflict — the
    earlier authored decision, by the ADR ``date`` + id ordinal — is demoted at
    rank time so the later decision surfaces first. It is **conflict-gated**:
    co-active complementary decisions are never reordered. The key-free
    deterministic path (``use_embeddings=False``, no API key) cannot detect
    conflicts, so it passes ``None`` and stays byte-stable.

    ``apply_utility`` and ``apply_owner`` (default
    ``False`` = inert) are the two **recall-path** addends — usefulness and
    owner-aboutness — folded into the effective-confidence component *before*
    the ``top_k`` cut so either can promote a belief **into** the rendered set.
    Both are promotion-only and orthogonal; the semantic-search ``query`` path
    passes neither.

    All these signals are rank-time only — stored confidence and the returned
    ``effective_confidence`` are untouched.
    """
    await assert_store_schema_current(session)
    populate_trust_cache(await get_trust_weight_map(session))
    trust_policy = await load_trust_policy(session)
    decay_policy = await load_decay_policy(session)

    query_emb = (
        _embed(request.question) if use_embeddings and request.question is not None else None
    )
    # The as-of note map / disclosure count are query-response surfaces; the
    # projection / digest path (as_of unset there) has no use for them — and
    # neither are the claim-prefilter stats (this path sets no
    # structural filter).
    scored, pub_at_by_id, _as_of_notes, _as_of_excluded, _claim_stats = await _gather_scored(
        session, request, query_emb, trust_policy, decay_policy
    )

    # on the projection / digest path (never `query`, which scores
    # inline and never calls this), add the usefulness rank-lift to each
    # candidate's effective confidence **before** the top-k truncation below —
    # so an acted-upon belief tied at the confidence ceiling is promoted *into*
    # the rendered set, not merely reordered within an already-cut top-k. This
    # is the case the additive form exists to serve: on a store where
    # the cap ties thousands of beliefs at one confidence value, a
    # multiplier could not separate them at all.
    if apply_utility:
        scored = await _apply_utility_to_scored(session, scored)

    # the aboutness addend, applied immediately after utility and on
    # the same side of the top_k cut, for the same reason — a belief about the
    # viewer tied at the confidence ceiling must be promoted *into* the rendered
    # set, not merely reordered inside an already-cut top_k. Orthogonal to
    # utility: separate addend, separate coefficient, neither reads the other.
    if apply_owner:
        scored = await _apply_owner_to_scored(session, scored)

    demoted = await _precedence_demotions(session, scored, pub_at_by_id, conflict_pairs)
    scored.sort(
        key=lambda s: _rank_score(s, code_symbol_rank_weight, demoted),
        reverse=True,
    )
    top = scored[: request.top_k]
    return await _collapse_co_evidential_top_k(session, top)


async def _apply_utility_to_scored(session: AsyncSession, scored: list[Scored]) -> list[Scored]:
    """Add the usefulness rank-lift to each candidate's effective confidence.

    Reshapes only the effective-confidence component of the ``Scored`` tuple
    (the similarity term is untouched); the whole candidate set is adjusted
    before ranking, so utility governs which beliefs survive the ``top_k`` cut.
    A no-op when utility is disabled or nothing has utility evidence.

    The reshaped component is a *ranking* score, not a confidence — the additive
    bonus is unbounded above. Only the projection path passes
    ``apply_utility=True``, and it consumes the value as a sort key.
    """
    from .utility_policy import apply_utility

    if not scored:
        return scored
    particles = [p for p, _, _ in scored]
    rows = await load_source_rows(session, particles)
    source_info: dict[str, tuple[str, str | None]] = {}
    for p in particles:
        _pub, st, _eid, uri, _auth = rows.get(p.id, (None, "", None, None, None))
        source_info[p.id] = (st, uri)
    eff_by_id = {p.id: eff for p, _, eff in scored}
    adjusted = await apply_utility(session, eff_by_id, source_info)
    return [(p, sim, adjusted.get(p.id, eff)) for p, sim, eff in scored]


async def _apply_owner_to_scored(session: AsyncSession, scored: list[Scored]) -> list[Scored]:
    """Add the owner-relevance rank-lift to each candidate's score.

    The aboutness sibling of :func:`_apply_utility_to_scored`, and deliberately
    the same shape: it reshapes only the effective-confidence component of the
    ``Scored`` tuple (the similarity term is untouched), and the whole candidate
    set is adjusted before ranking so aboutness governs which beliefs survive
    the ``top_k`` cut. A no-op when the lens is disabled or no viewer resolves.

    The reshaped component is a *ranking* score, not a confidence. Only the
    recall path passes ``apply_owner=True``; the semantic-search ``query`` path
    never does — it already has ``QueryRequest.subject_id`` for a caller who
    wants the viewer's beliefs specifically.
    """
    from .owner_policy import apply_owner as apply_owner_policy

    if not scored:
        return scored
    eff_by_id = {p.id: eff for p, _, eff in scored}
    subject_ids_by_id = {p.id: p.subject_ids for p, _, _ in scored}
    adjusted = await apply_owner_policy(session, eff_by_id, subject_ids_by_id)
    return [(p, sim, adjusted.get(p.id, eff)) for p, sim, eff in scored]


async def query_federated(
    stores: list[StoreHandle],
    request: QueryRequest,
    viewer_store: StoreHandle | None = None,
) -> QueryResponse:
    """Run a query across several stores under one viewer's trust lens.

    Fans out the retrieval to each store, merges the candidates, reranks the
    union by combined score, collapses CO_EVIDENTIAL groups **within each origin
    store** (links are store-local), and generates a single answer.

    The *viewer's* trust policy (``viewer_store``, default = the first handle) is
    applied to every store's candidates — "consensus is per-viewer trust at query
    time". Writes never federate; this is read-only.

    Known limitation: v1 merges *subjects* across stores but not
    *claims* — the same fact from two stores surfaces as two results until graded
    claim-equivalence / cross-entry conflict mode land.
    Subject-coverage gaps are omitted for federated queries (the subject filter
    is a store-local id; cross-store subject identity is future work). The viewer's
    ``CORPUS_ENTRY``-scoped trust statements key on store-local entry ids, so
    they only match candidates from the viewer's own store;
    SOURCE_TYPE- and URL-scoped policy applies uniformly across stores.
    """
    # the purely structural modes federate through their own
    # deterministic composition (same viewer-lens rule, no embedding, no LLM).
    if request.is_structural_mode:
        return await structural_query_federated(stores, request, viewer_store)
    if not stores:
        stores = [DEFAULT_STORE]
    viewer = viewer_store or stores[0]

    # Per-viewer trust lens: populate the extractor trust cache and snapshot
    # the source-trust policy and the decay policy once
    # from the viewer's store; every store's candidates are then scored against
    # them.
    async with session_scope(viewer) as viewer_session:
        await assert_store_schema_current(viewer_session)
        populate_trust_cache(await get_trust_weight_map(viewer_session))
        trust_policy = await load_trust_policy(viewer_session)
        decay_policy = await load_decay_policy(viewer_session)

    if request.question is None:  # unreachable: the structural dispatch above caught None
        raise ValueError("semantic query requires a question")
    query_emb = _embed(request.question)

    merged: list[Scored] = []
    pub_at_by_id: dict[str, datetime | None] = {}
    store_by_id: dict[str, StoreHandle] = {}
    coverage_gaps: list[str] = []
    # the viewer's single as_of (on the shared request) applies to
    # every store's candidates — the same viewer-owns-the-lens rule as trust
    # and decay. Each store contributes its own supersession crossings and
    # undatable exclusions; the disclosure count is summed across stores.
    as_of_notes_by_id: dict[str, AsOfNote] = {}
    as_of_excluded = 0
    # claim-prefilter stats and coverage counts summed across stores.
    claim_coverage: ClaimCoverage | None = None
    for store in stores:
        async with session_scope(store) as s:
            await assert_store_schema_current(s)
            scored, pubs, notes, excluded, claim_stats = await _gather_scored(
                s, request, query_emb, trust_policy, decay_policy
            )
            merged.extend(scored)
            pub_at_by_id.update(pubs)
            as_of_notes_by_id.update(notes)
            as_of_excluded += excluded
            if claim_stats is not None:
                store_coverage = await _claim_coverage_for_stats(s, claim_stats)
                if claim_coverage is None:
                    claim_coverage = store_coverage
                elif store_coverage is not None:
                    claim_coverage = ClaimCoverage(
                        active_total=claim_coverage.active_total + store_coverage.active_total,
                        with_claims=claim_coverage.with_claims + store_coverage.with_claims,
                        matched=claim_coverage.matched + store_coverage.matched,
                        not_normalizable_excluded=claim_coverage.not_normalizable_excluded
                        + store_coverage.not_normalizable_excluded,
                    )
            for p, _, _ in scored:
                store_by_id[p.id] = store
            coverage_gaps.extend(await _find_coverage_gaps(s))

    merged.sort(key=_rank_score, reverse=True)
    top = merged[: request.top_k]
    top = await _collapse_federated(top, store_by_id)
    top.sort(key=_rank_score, reverse=True)

    truncation_warning = _truncation_warning(merged, top, request.top_k)
    return await _build_response(
        request,
        top,
        pub_at_by_id,
        coverage_gaps,
        [],
        truncation_warning,
        as_of_notes_by_id=as_of_notes_by_id,
        as_of_excluded_undatable=as_of_excluded,
        claim_coverage=claim_coverage,
        embeddings_used=query_emb is not None,
    )


async def _collapse_federated(
    top: list[Scored],
    store_by_id: dict[str, StoreHandle],
) -> list[Scored]:
    """Collapse CO_EVIDENTIAL groups per origin store (links are store-local).

    Cross-store equivalents are *not* collapsed in v1 (known
    limitation) — that needs graded claim-equivalence.
    """
    by_store: dict[StoreHandle, list[Scored]] = defaultdict(list)
    for item in top:
        by_store[store_by_id[item[0].id]].append(item)

    collapsed: list[Scored] = []
    for store, items in by_store.items():
        async with session_scope(store) as s:
            collapsed.extend(await _collapse_co_evidential_top_k(s, items))
    return collapsed
