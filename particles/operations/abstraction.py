"""Abstraction-promotion consolidation.

The dream-cycle pass that distills clusters of settled, subject-scoped
episodic particles into single semantic beliefs carrying **premise links** —
one ``ProvenanceRef(type=PARTICLE)`` per source particle (the particle id
travels in ``corpus_entry_id``, the §3 field-reuse convention). Two halves,
sharing one promotion-shaped LLM budget (``max_promotions_per_run``):

1. **Revalidation** (§5) — for each ACTIVE derived particle with a
   non-ACTIVE premise, run the cheapest-first ladder: structural no-op /
   entailment re-check / re-synthesize + paraphrase-compare / retire.
   Repairs run before new promotions so a changed store is reconciled
   before it is further abstracted.
2. **Promotion** (§2) — cluster eligible specifics per subject (pairwise
   cosine ≥ ``cluster_similarity_threshold``, connected components, size ≥
   ``min_cluster_size``), synthesize one candidate claim per cluster, gate
   it (entailment + dedup), then either assert it through the normal §6.6
   ingest path (``mode: auto``) or record an ``ABSTRACTION_CANDIDATE``
   operator event for the curation queue (``mode: propose``, the default).

Every LLM call routes through the shared semantic seam
(:func:`particles.operations._llm._llm_call`) with ``purpose="abstraction"``
(per-purpose routing; the circuit breaker is inherited),
and every prompt F3-fences the particle contents — claims are
LLM-extracted from untrusted sources and must not be able to coerce a
verdict.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    ContributorRef,
    JudgeVerdictKind,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
    is_truth_apt,
)
from particles.core.scoring.confidence import CalibrationSource, derive_abstraction_confidence
from particles.core.stance import stance_holder
from particles.core.status import Status, StatusReason
from particles.embeddings import cosine_similarity, get_embedding_model
from particles.extraction.polarity import is_non_asserted
from particles.operations._llm import _llm_call
from particles.store.event_store import EventRefKind, OperatorEventType, record_event
from particles.store.particle_store import (
    get_active_derived_particles,
    get_active_particles_with_embeddings,
    get_inconsistency_backrefs,
    get_particles_by_ids,
    get_superseding_particle,
    update_particle_provenance,
    update_particle_status,
)
from particles.store.subject_store import list_all_subjects

log = logging.getLogger(__name__)

#: ``asserted_by`` / event ``actor`` identity for everything this pass writes.
ABSTRACTION_ACTOR = "consolidation:abstraction"

#: Revision-chain hops followed when replacing a SUPERSEDED premise with its
#: successor (§5). A chain longer than this is pathological; the premise is
#: treated as dropped.
_MAX_SUPERSESSION_HOPS = 10

_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["claim"],
    "additionalProperties": False,
}

_ENTAILMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entailed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["entailed"],
    "additionalProperties": False,
}

_PARAPHRASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PARAPHRASE", "DISTINCT", "UNSURE"]},
    },
    "required": ["verdict"],
    "additionalProperties": False,
}


class RevalidationCounts(BaseModel):
    """§5 ladder outcomes for one pass run."""

    checked: int = 0  # derived particles with a non-ACTIVE premise
    refreshed_structural: int = 0  # rung 1: content-preserving change
    refreshed_entailed: int = 0  # rung 2: entailment re-confirmed
    refreshed_paraphrase: int = 0  # rung 3: D′ ≈ D, refs refreshed
    superseded: int = 0  # rung 3: D′ DISTINCT → D superseded by D′
    retired: int = 0  # rung 4: PROVENANCE_STALE / RETRACTED_DEPENDENCY
    deferred: int = 0  # budget exhausted or LLM unavailable; next cycle


class AbstractionReport(BaseModel):
    """What one abstraction-promotion pass did."""

    enabled: bool = True
    mode: str = "propose"
    clusters_found: int = 0
    candidates_synthesized: int = 0
    promoted_particle_ids: list[str] = Field(default_factory=list)
    proposed_event_ids: list[str] = Field(default_factory=list)
    rejected_entailment: int = 0
    rejected_duplicate: int = 0
    skipped_budget: int = 0  # eligible clusters left for the next cycle
    revalidation: RevalidationCounts = Field(default_factory=RevalidationCounts)
    llm_calls: int = 0
    warnings: list[str] = Field(default_factory=list)


class _Budget:
    """Shared promotion-shaped spend counter (§9 ``max_promotions_per_run``)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.spent = 0

    def take(self) -> bool:
        if self.spent >= self.limit:
            return False
        self.spent += 1
        return True


# ---------------------------------------------------------------------------
# LLM helpers — synthesis + the two judges (all F3-fenced, purpose-routed)
# ---------------------------------------------------------------------------


def _parse_json_object(response: str | None) -> dict[str, Any] | None:
    """Tolerantly isolate and parse one JSON object from an LLM reply."""
    if not response:
        return None
    text = response.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        raw: Any = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


async def _synthesize_claim(premise_contents: list[str]) -> tuple[str, str] | None:
    """One generalized claim sentence from a premise cluster, or ``None``.

    Returns ``(claim, rationale)``; ``None`` on LLM failure or an unparseable
    reply (the caller discloses and skips — never a crashed pass).
    """
    from particles.llm import data_fence_instruction, fence, make_nonce

    nonce = make_nonce()
    system = (
        "You are consolidating an epistemic knowledge base. The user message "
        "contains several specific, related claims. Write ONE general claim "
        "sentence that captures the durable knowledge they jointly support.\n"
        "Rules:\n"
        "- The general claim must be fully supported by the specific claims — "
        "never broader than the evidence.\n"
        "- Preserve discriminative detail: a vague summary that loses the "
        "actionable specifics is worse than no summary. Prefer 'X fails at "
        "the signing step; do Y' over 'there are issues with X'.\n"
        "- Carry every date, quantity, and version identifier through "
        "VERBATIM. Never replace one with a vaguer stand-in: not 'recently' "
        "for '2026-07-18', not 'several' for 'three', not 'a recent release' "
        "for 'v1.77.4'. If the specifics disagree on such a value, keep each "
        "one explicitly rather than averaging or dropping them.\n"
        "- One declarative sentence; no meta-commentary.\n"
        'Return a JSON object: {"claim": "...", "rationale": "one '
        'sentence on why this generalization is supported"}.\n\n' + data_fence_instruction(nonce)
    )
    user = "\n\n".join(
        f"Claim {i + 1}:\n{fence(content, nonce, label=f'claim_{i + 1}')}"
        for i, content in enumerate(premise_contents)
    )
    response = await _llm_call(
        user,
        max_tokens=400,
        system=system,
        response_schema=_SYNTHESIS_SCHEMA,
        purpose="abstraction",
    )
    data = _parse_json_object(response)
    if data is None:
        return None
    claim = str(data.get("claim") or "").strip()
    if not claim:
        return None
    return claim, str(data.get("rationale") or "").strip()


async def _check_entailment(claim: str, premise_contents: list[str]) -> bool | None:
    """Is ``claim`` entailed by the conjunction of the premises? ``None`` = LLM failure."""
    from particles.llm import data_fence_instruction, fence, make_nonce

    nonce = make_nonce()
    system = (
        "You are auditing a knowledge base. Decide whether the GENERAL claim "
        "in the user message is fully entailed by the SPECIFIC claims taken "
        "together — i.e. it asserts nothing beyond what they jointly support. "
        "Over-generalization (a broader scope, a stronger quantifier, an "
        "added causal link) is NOT entailed.\n"
        "Detail loss also fails this gate: if a specific claim carries a date, "
        "a quantity, or a version identifier and the general claim drops it or "
        "replaces it with a vaguer stand-in ('recently' for a date, 'several' "
        "for a count, 'a recent version' for a version), answer false — a "
        "reader of the general claim alone could no longer recover what the "
        "specifics said.\n"
        'Return a JSON object: {"entailed": true|false, "reason": "..."}.\n\n'
        + data_fence_instruction(nonce)
    )
    user = f"General claim:\n{fence(claim, nonce, label='general')}\n\n" + "\n\n".join(
        f"Specific claim {i + 1}:\n{fence(content, nonce, label=f'specific_{i + 1}')}"
        for i, content in enumerate(premise_contents)
    )
    response = await _llm_call(
        user,
        max_tokens=200,
        system=system,
        response_schema=_ENTAILMENT_SCHEMA,
        purpose="abstraction",
    )
    data = _parse_json_object(response)
    if data is None or not isinstance(data.get("entailed"), bool):
        return None
    return bool(data["entailed"])


async def _paraphrase_verdict(content_a: str, content_b: str) -> JudgeVerdictKind:
    """Single-pair PARAPHRASE / DISTINCT / UNSURE judge (LLM failure → UNSURE)."""
    from particles.llm import data_fence_instruction, fence, make_nonce

    nonce = make_nonce()
    system = (
        "Decide whether the two claims in the user message assert the SAME "
        "underlying fact (a paraphrase) or DIFFERENT facts.\n"
        'Return a JSON object: {"verdict": "PARAPHRASE"|"DISTINCT"|"UNSURE"}.\n\n'
        + data_fence_instruction(nonce)
    )
    user = (
        f"Claim A:\n{fence(content_a, nonce, label='claim_a')}\n\n"
        f"Claim B:\n{fence(content_b, nonce, label='claim_b')}"
    )
    response = await _llm_call(
        user,
        max_tokens=100,
        system=system,
        response_schema=_PARAPHRASE_SCHEMA,
        purpose="abstraction",
    )
    data = _parse_json_object(response)
    if data is None:
        return JudgeVerdictKind.UNSURE
    try:
        return JudgeVerdictKind(str(data.get("verdict", "")).strip().upper())
    except ValueError:
        return JudgeVerdictKind.UNSURE


# ---------------------------------------------------------------------------
# Premise-link helpers
# ---------------------------------------------------------------------------


def premise_ids_of(particle: Particle) -> list[str]:
    """Premise particle ids of a derived particle (§3 field-reuse convention)."""
    return [
        ref.corpus_entry_id for ref in particle.provenance if ref.type is ProvenanceRefType.PARTICLE
    ]


def _premise_refs(premise_ids: list[str]) -> list[ProvenanceRef]:
    """PARTICLE-typed premise refs, mirroring ``build_inconsistency_particle``."""
    return [
        ProvenanceRef(
            type=ProvenanceRefType.PARTICLE,
            corpus_entry_id=pid,
            snapshot_id=pid,
        )
        for pid in premise_ids
    ]


def is_derived(particle: Particle) -> bool:
    """True for a machine-derived particle."""
    return particle.confidence.calibration_source is CalibrationSource.DERIVED


async def stale_support_discounts(
    session: AsyncSession, particles: list[Particle]
) -> dict[str, float]:
    """Read-time discount factors for derived particles with stale support (§5).

    A derived particle whose premise set contains a non-ACTIVE (or missing)
    particle stays ACTIVE and visible between the premise change and its next
    revalidation, but its effective confidence is multiplied by
    ``stale_support_discount``. Computed live from premise status — no stored
    flag exists to go stale (everything reactive is computed at
    read). Applies regardless of ``consolidation.abstraction.enabled``: the
    derived population may predate a config flip, and honest scoring of it is
    read-surface correctness, not pass behaviour.

    Returns:
        ``{particle_id: factor}`` for exactly the derived particles that are
        currently discounted; absent ids are undiscounted (factor 1.0).
    """
    derived = [p for p in particles if is_derived(p)]
    if not derived:
        return {}
    discount = get_config().consolidation.abstraction.stale_support_discount
    premise_ids = sorted({pid for d in derived for pid in premise_ids_of(d)})
    by_id = await get_particles_by_ids(session, premise_ids)
    out: dict[str, float] = {}
    for d in derived:
        for pid in premise_ids_of(d):
            premise = by_id.get(pid)
            if premise is None or premise.status is not Status.ACTIVE:
                out[d.id] = discount
                break
    return out


async def projection_suppressed_premise_ids(session: AsyncSession) -> frozenset[str]:
    """Premises of ACTIVE derived particles, for projection-ranker suppression (§10).

    Ranking-side only — no status change: the projection budget is never spent
    on both an abstraction and its sources. Premises reappear automatically if
    the derived particle is ever retired (they simply stop being returned
    here). Empty when ``source_demotion`` is ``none``.
    """
    if get_config().consolidation.abstraction.source_demotion != "suppress_in_projection":
        return frozenset()
    derived = await get_active_derived_particles(session)
    return frozenset(pid for d in derived for pid in premise_ids_of(d))


def _derived_depth(
    particle: Particle,
    derived_by_id: dict[str, Particle],
    _memo: dict[str, int] | None = None,
) -> int:
    """Abstraction depth: 0 for non-derived; 1 + max premise depth otherwise.

    Walks only within the loaded derived population; a premise outside it is
    depth 0 by construction (non-derived, or not ACTIVE and thus not eligible
    anyway). Cycles cannot occur (a particle's premises exist before it does),
    but the provisional memo entry bounds a corrupted graph defensively.
    """
    if not is_derived(particle):
        return 0
    memo = {} if _memo is None else _memo
    if particle.id in memo:
        return memo[particle.id]
    memo[particle.id] = 1  # provisional: breaks a (theoretically impossible) cycle
    deepest_premise = 0
    for pid in premise_ids_of(particle):
        premise = derived_by_id.get(pid)
        if premise is not None:
            deepest_premise = max(deepest_premise, _derived_depth(premise, derived_by_id, memo))
    memo[particle.id] = 1 + deepest_premise
    return memo[particle.id]


def _build_derived_particle(
    *,
    claim: str,
    premises: list[Particle],
    subject_ids: list[str],
    supersedes: str | None = None,
) -> Particle:
    """Construct the derived Particle per the §3 lineage contract."""
    now = datetime.now(UTC)
    return Particle(
        content=claim,
        confidence=Confidence(
            value=derive_abstraction_confidence([p.confidence.value for p in premises]),
            calibration_source=CalibrationSource.DERIVED,
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=_premise_refs([p.id for p in premises]),
        asserted_by=ABSTRACTION_ACTOR,
        subject_ids=subject_ids,
        supersedes=supersedes,
        extractor_ref=None,
        contributors=[ContributorRef(id=ABSTRACTION_ACTOR, role="agent", at=now)],
    )


def _shared_subject_ids(premises: list[Particle], fallback_subject_id: str) -> list[str]:
    """Subjects shared by every premise; never empty (cluster is subject-scoped)."""
    shared: set[str] | None = None
    for p in premises:
        ids = set(p.subject_ids)
        shared = ids if shared is None else shared & ids
    if not shared:
        return [fallback_subject_id]
    return sorted(shared)


# ---------------------------------------------------------------------------
# Cluster discovery (§2)
# ---------------------------------------------------------------------------


class _Cluster:
    def __init__(self, subject_id: str, members: list[Particle]) -> None:
        self.subject_id = subject_id
        self.members = members

    @property
    def member_ids(self) -> frozenset[str]:
        return frozenset(p.id for p in self.members)


def _components(pairs: list[tuple[str, str]]) -> list[set[str]]:
    """Transitive components over similarity edges (BFS, order-stable)."""
    adjacency: dict[str, set[str]] = {}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    components: list[set[str]] = []
    unvisited = set(adjacency)
    while unvisited:
        start = min(unvisited)  # deterministic traversal order
        component: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            queue.extend(adjacency[node] - component)
        components.append(component)
        unvisited -= component
    return components


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


#: Cheap content heuristic for a date-anchored claim. Matches an absolute
#: anchor (a 19xx/20xx year — which also covers the year of any ISO date, a
#: month name, a weekday, a clock time, a quarter) or a relative one
#: ("tomorrow", "next week", "two days ago"). Deliberately over-inclusive per
#: ``exclude_time_anchored``: a false exclusion costs one un-promoted cluster,
#: a false inclusion is the temporal regression this guard exists to stop.
_TIME_ANCHOR_PATTERN = re.compile(
    r"""
      \b(?:19|20)\d{2}\b                          # year, incl. the year of an ISO date
    | \b\d{1,2}:\d{2}\b                           # clock time
    | \b(?:january|february|march|april|june|july|august
         |september|october|november|december)\b                # unambiguous full names
    | \b(?:jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)\.?\s+\d{1,4}\b
    | \b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)\b
    | \b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b
    | \bq[1-4]\b                                  # fiscal quarter
    | \b(?:today|tonight|tomorrow|yesterday|currently|recently)\b
    | \b(?:next|last|past|this|coming|previous)\s+
        (?:week|month|year|quarter|day|decade)\b
    | \b(?:ago|as\s+of)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_time_anchored(particle: Particle) -> bool:
    """True when a particle's truth is pinned to a moment.

    Two signals: an explicit ``valid_until`` (the expiry bearer — the
    claim is known to have a shelf life) and a date / relative-time mention in
    the content. Such particles are excluded from cluster formation
    eligibility, because a faithful generalization over them is exactly the §7
    *vague-but-true* failure: "the operator deployed several times" entails its
    premises while dropping the dates a temporal-reasoning question asks for.
    """
    if particle.valid_until is not None:
        return True
    return _TIME_ANCHOR_PATTERN.search(particle.content) is not None


async def _find_clusters(
    session: AsyncSession,
    *,
    scope_ids: frozenset[str] | None,
) -> list[_Cluster]:
    """Eligible premise clusters, largest first (§2 eligibility + clustering)."""
    cfg = get_config().consolidation.abstraction
    now = datetime.now(UTC)
    min_age = timedelta(days=cfg.min_source_age_days)

    contested = await get_inconsistency_backrefs(session)
    derived = await get_active_derived_particles(session)
    derived_by_id = {p.id: p for p in derived}
    already_premise: set[str] = set()
    for d in derived:
        already_premise.update(premise_ids_of(d))

    def eligible(p: Particle) -> bool:
        if p.id in contested or p.id in already_premise:
            return False
        # Depth gate (§9 max_depth): a premise must sit strictly below the
        # cap so the abstraction built on it stays within it. Default 1 ⇒
        # premises are never themselves derived.
        if _derived_depth(p, derived_by_id) >= cfg.max_depth:
            return False
        if now - _aware(p.asserted_at) < min_age:
            return False
        # Temporal guard: a date-anchored premise generalizes into
        # a claim that has lost the date, which is what the first §8 A/B
        # measured as the whole QA-at-budget regression.
        if cfg.exclude_time_anchored and _is_time_anchored(p):
            return False
        # Mirrors links_suggest: only truth-apt, asserted, non-stance claims.
        # Abstracting stances would collapse per-holder attribution.
        if not is_truth_apt(p) or is_non_asserted(p.properties):
            return False
        return stance_holder(p) is None

    clusters: list[_Cluster] = []
    seen_components: set[frozenset[str]] = set()
    subjects = sorted(await list_all_subjects(session), key=lambda s: s.id)
    for subject in subjects:
        with_embs = await get_active_particles_with_embeddings(session, subject_id=subject.id)
        candidates = [(p, e) for p, e in with_embs if eligible(p)]
        if len(candidates) < cfg.min_cluster_size:
            continue
        pairs: list[tuple[str, str]] = []
        for i, (p_a, emb_a) in enumerate(candidates):
            for p_b, emb_b in candidates[i + 1 :]:
                if cosine_similarity(emb_a, emb_b) >= cfg.cluster_similarity_threshold:
                    pairs.append((p_a.id, p_b.id))
        if not pairs:
            continue
        by_id = {p.id: p for p, _ in candidates}
        for component in _components(pairs):
            if len(component) < cfg.min_cluster_size:
                continue
            key = frozenset(component)
            if key in seen_components:  # same cluster reachable via a shared subject
                continue
            seen_components.add(key)
            if scope_ids is not None and not (component & scope_ids):
                continue  # delta scope: at least one member changed since last run
            members = sorted((by_id[pid] for pid in component), key=lambda p: p.id)
            clusters.append(_Cluster(subject.id, members))

    clusters.sort(key=lambda c: (-len(c.members), min(c.member_ids)))
    return clusters


# ---------------------------------------------------------------------------
# Gates (§2)
# ---------------------------------------------------------------------------


async def _duplicate_of(
    session: AsyncSession,
    claim: str,
    subject_id: str,
    exclude_ids: frozenset[str],
) -> str | None:
    """Existing ACTIVE belief this claim paraphrases, or ``None``.

    Embedding pre-filter at ``links_suggest.candidate_threshold`` (the
    near-duplicate scale), then the single-pair paraphrase judge.

    ``None`` means *not a duplicate*, so the encoder-free early return is
    ambiguous at this layer — the caller pre-checks the model and discards the
    candidate before reaching here, the same guarded shape
    ``find_duplicate_subjects`` uses.
    """
    model = get_embedding_model()
    if model is None:
        return None
    threshold = get_config().links_suggest.candidate_threshold
    encoded = await asyncio.to_thread(
        model.encode, [claim], convert_to_numpy=True, normalize_embeddings=True
    )
    claim_emb = encoded[0]
    with_embs = await get_active_particles_with_embeddings(session, subject_id=subject_id)
    near = [
        (p, cosine_similarity(claim_emb, emb)) for p, emb in with_embs if p.id not in exclude_ids
    ]
    near = [(p, sim) for p, sim in near if sim >= threshold]
    near.sort(key=lambda t: -t[1])
    for p, _sim in near[:3]:  # judge at most the top few near-duplicates
        if await _paraphrase_verdict(claim, p.content) is JudgeVerdictKind.PARAPHRASE:
            return p.id
    return None


# ---------------------------------------------------------------------------
# Revalidation ladder (§5)
# ---------------------------------------------------------------------------


async def _updated_premises(
    session: AsyncSession, premise_ids: list[str]
) -> tuple[list[Particle], bool, bool]:
    """Resolve the updated premise set S′ for a derived particle.

    Returns ``(premises, changed, content_preserving)``:
    - ``premises``: ACTIVE premises, with SUPERSEDED ones replaced by their
      ACTIVE successors and retracted/stale/missing ones dropped;
    - ``changed``: True when any premise is non-ACTIVE (revalidation due);
    - ``content_preserving``: True when nothing was dropped and every
      replacement's content is byte-identical (§5 rung 1).
    """
    by_id = await get_particles_by_ids(session, premise_ids)
    premises: list[Particle] = []
    changed = False
    content_preserving = True
    for pid in premise_ids:
        p = by_id.get(pid)
        if p is None:
            changed = True
            content_preserving = False
            continue
        if p.status is Status.ACTIVE:
            premises.append(p)
            continue
        changed = True
        if p.status is Status.SUPERSEDED:
            successor: Particle | None = p
            for _ in range(_MAX_SUPERSESSION_HOPS):
                if successor is None:
                    break
                successor = await get_superseding_particle(session, successor.id)
                if successor is None or successor.status is Status.ACTIVE:
                    break
                if successor.status is not Status.SUPERSEDED:
                    successor = None
                    break
            if successor is not None and successor.status is Status.ACTIVE:
                premises.append(successor)
                if successor.content != p.content:
                    content_preserving = False
                continue
        content_preserving = False  # dropped (retracted / stale / dead chain)
    return premises, changed, content_preserving


async def _revalidate(
    session: AsyncSession,
    report: AbstractionReport,
    budget: _Budget,
) -> None:
    """Run the §5 ladder over every ACTIVE derived particle with stale support."""
    cfg = get_config().consolidation.abstraction
    counts = report.revalidation
    for d in sorted(await get_active_derived_particles(session), key=lambda p: p.id):
        premise_ids = premise_ids_of(d)
        premises, changed, content_preserving = await _updated_premises(session, premise_ids)
        if not changed:
            continue
        counts.checked += 1
        new_ids = [p.id for p in premises]

        # Rung 4 (no LLM): support collapsed below the floor → retire.
        if len(premises) < cfg.min_cluster_size:
            update = await get_particles_by_ids(session, [d.id])  # re-check still ACTIVE
            if update.get(d.id) is not None:
                await update_particle_status(
                    session, d.id, Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY
                )
            counts.retired += 1
            continue

        # Rung 1 (no LLM): content-preserving change → refresh refs.
        if content_preserving:
            await update_particle_provenance(session, d.id, _premise_refs(new_ids))
            counts.refreshed_structural += 1
            continue

        if not budget.take():
            counts.deferred += 1
            continue

        # Rung 2: one judge call — is D still entailed by S′?
        contents = [p.content for p in premises]
        report.llm_calls += 1
        entailed = await _check_entailment(d.content, contents)
        if entailed is None:
            counts.deferred += 1  # LLM unavailable; leave discounted for next cycle
            continue
        if entailed:
            await update_particle_provenance(session, d.id, _premise_refs(new_ids))
            counts.refreshed_entailed += 1
            continue

        # Rung 3: re-synthesize and compare.
        report.llm_calls += 1
        synthesized = await _synthesize_claim(contents)
        if synthesized is None:
            counts.deferred += 1
            continue
        claim, _rationale = synthesized
        report.llm_calls += 1
        verdict = await _paraphrase_verdict(claim, d.content)
        if verdict is JudgeVerdictKind.PARAPHRASE:
            await update_particle_provenance(session, d.id, _premise_refs(new_ids))
            counts.refreshed_paraphrase += 1
            continue
        if verdict is JudgeVerdictKind.UNSURE:
            counts.deferred += 1  # keep the read-time discount; retry next cycle
            continue

        # DISTINCT → supersede D with D′ through the normal path. The status
        # flip is what lets the one-hop lint flag D's own dependents next
        # cycle — propagation is this ladder recursing one level per run.
        from particles.ingest.pipeline import reconcile_and_insert

        await update_particle_status(
            session, d.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
        )
        successor = _build_derived_particle(
            claim=claim,
            premises=premises,
            subject_ids=d.subject_ids or _shared_subject_ids(premises, ""),
            supersedes=d.id,
        )
        await reconcile_and_insert(session, successor, fail_closed=True)
        counts.superseded += 1


# ---------------------------------------------------------------------------
# Promotion (§2 + §6)
# ---------------------------------------------------------------------------


async def _promote_cluster(
    session: AsyncSession,
    cluster: _Cluster,
    report: AbstractionReport,
) -> None:
    """Synthesize, gate, and promote (or propose) one cluster's abstraction."""
    cfg = get_config().consolidation.abstraction
    contents = [p.content for p in cluster.members]

    report.llm_calls += 1
    synthesized = await _synthesize_claim(contents)
    if synthesized is None:
        report.warnings.append(
            f"synthesis unavailable for a {len(cluster.members)}-member cluster "
            f"(subject {cluster.subject_id}); retried next cycle"
        )
        return
    claim, rationale = synthesized
    report.candidates_synthesized += 1

    if cfg.require_entailment:
        report.llm_calls += 1
        entailed = await _check_entailment(claim, contents)
        if entailed is None:
            report.warnings.append("entailment judge unavailable; candidate discarded")
            return
        if not entailed:
            report.rejected_entailment += 1
            return

    # the duplicate gate needs an encoder for its pre-filter, and
    # without one `_duplicate_of` returns None — which this caller would read as
    # "not a duplicate" and promote a paraphrase of an existing ACTIVE belief,
    # leaving `rejected_duplicate` at a healthy-looking 0. Discard instead, the
    # same way the entailment judge above handles its own unavailability: an
    # unrunnable check is not a passed check.
    if get_embedding_model() is None:
        report.warnings.append(
            "duplicate check unavailable (no embedding model); candidate discarded"
        )
        return

    report.llm_calls += 1  # embedding is local; the paraphrase judge is the LLM spend
    duplicate = await _duplicate_of(session, claim, cluster.subject_id, cluster.member_ids)
    if duplicate is not None:
        report.rejected_duplicate += 1
        return

    subject_ids = _shared_subject_ids(cluster.members, cluster.subject_id)
    if cfg.mode == "auto":
        from particles.ingest.pipeline import reconcile_and_insert

        particle = _build_derived_particle(
            claim=claim, premises=cluster.members, subject_ids=subject_ids
        )
        inserted = await reconcile_and_insert(session, particle, fail_closed=True)
        if inserted is not None:
            report.promoted_particle_ids.append(inserted.id)
    else:
        event = await record_event(
            session,
            actor=ABSTRACTION_ACTOR,
            event_type=OperatorEventType.ABSTRACTION_CANDIDATE,
            reason=rationale or None,
            refs=[(EventRefKind.PARTICLE, p.id) for p in cluster.members],
            payload={
                "claim": claim,
                "rationale": rationale,
                "premise_ids": [p.id for p in cluster.members],
                "subject_ids": subject_ids,
                "confidence_value": derive_abstraction_confidence(
                    [p.confidence.value for p in cluster.members]
                ),
            },
        )
        report.proposed_event_ids.append(event.event_id)


async def run_abstraction_pass(
    session: AsyncSession,
    *,
    scope_ids: frozenset[str] | None = None,
) -> AbstractionReport:
    """Run one abstraction-promotion pass.

    Revalidation first (repair the DAG before extending it), then promotion
    of new clusters, both under the shared ``max_promotions_per_run`` budget.
    The caller (the dream cycle's pass wrapper, or a test) owns the session
    transaction; this function flushes through the store helpers but does not
    commit.

    Args:
        session: Store session.
        scope_ids: delta scope — when given, only clusters with
            at least one member in the scope are considered for promotion
            (revalidation always runs store-wide: a premise change outside
            the delta window still needs repair).

    Returns:
        The pass report; ``llm_calls`` feeds the ConsolidationPass entry.
    """
    cfg = get_config().consolidation.abstraction
    report = AbstractionReport(enabled=cfg.enabled, mode=cfg.mode)
    if not cfg.enabled:
        return report

    budget = _Budget(cfg.max_promotions_per_run)
    await _revalidate(session, report, budget)

    clusters = await _find_clusters(session, scope_ids=scope_ids)
    report.clusters_found = len(clusters)
    for cluster in clusters:
        if not budget.take():
            report.skipped_budget += 1
            continue
        await _promote_cluster(session, cluster, report)

    if report.skipped_budget:
        report.warnings.append(
            f"{report.skipped_budget} eligible cluster(s) beyond "
            f"max_promotions_per_run={cfg.max_promotions_per_run}; next cycle continues"
        )
    return report


# ---------------------------------------------------------------------------
# Propose-mode resolution (§6) — the curation accept / reject gestures
# ---------------------------------------------------------------------------


class CandidateStaleError(ValueError):
    """The candidate's premises changed since it was proposed.

    Accepting it would assert a claim whose recorded evidence no longer
    stands; the dream cycle re-clusters the surviving premises and proposes a
    fresh candidate on its next run.
    """


async def _load_pending_candidate(session: AsyncSession, candidate_event_id: str) -> dict[str, Any]:
    """The candidate event's payload, after verifying it exists and is unresolved."""
    from particles.store.event_store import get_event, list_events

    event = await get_event(session, candidate_event_id)
    if event is None or event.event_type is not OperatorEventType.ABSTRACTION_CANDIDATE:
        raise ValueError(f"No abstraction candidate event {candidate_event_id!r}.")
    resolutions = await list_events(
        session, event_type=OperatorEventType.ABSTRACTION_RESOLVED, limit=1000
    )
    for r in resolutions:
        if (r.payload or {}).get("candidate_event_id") == candidate_event_id:
            raise ValueError(
                f"Candidate {candidate_event_id!r} was already "
                f"{(r.payload or {}).get('resolution', 'resolved')}."
            )
    payload = event.payload or {}
    if not payload.get("claim") or not payload.get("premise_ids"):
        raise ValueError(f"Candidate event {candidate_event_id!r} has a malformed payload.")
    return payload


async def accept_candidate(
    session: AsyncSession, candidate_event_id: str, *, actor: str = "curate"
) -> Particle:
    """Assert a proposed abstraction (the curation ``accept`` gesture, §6).

    Re-reads the candidate from its ``ABSTRACTION_CANDIDATE`` event, requires
    every premise to still be ACTIVE (a changed premise set makes the
    candidate stale — :class:`CandidateStaleError`; the next cycle re-proposes
    from the surviving premises), asserts the derived particle through the
    normal §6.6 ingest path, and records the ``ABSTRACTION_RESOLVED`` verdict.
    Does not commit — the caller owns the transaction.
    """
    from particles.ingest.pipeline import reconcile_and_insert

    payload = await _load_pending_candidate(session, candidate_event_id)
    premise_ids = [str(pid) for pid in payload["premise_ids"]]
    by_id = await get_particles_by_ids(session, premise_ids)
    stale = [
        pid
        for pid in premise_ids
        if by_id.get(pid) is None or by_id[pid].status is not Status.ACTIVE
    ]
    if stale:
        raise CandidateStaleError(
            f"{len(stale)} premise(s) of candidate {candidate_event_id!r} are no longer "
            "ACTIVE; the candidate is stale. The next consolidation run re-clusters the "
            "surviving premises."
        )
    premises = [by_id[pid] for pid in premise_ids]
    particle = _build_derived_particle(
        claim=str(payload["claim"]),
        premises=premises,
        subject_ids=[str(s) for s in payload.get("subject_ids") or []]
        or _shared_subject_ids(premises, ""),
    )
    inserted = await reconcile_and_insert(session, particle, fail_closed=True)
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.ABSTRACTION_RESOLVED,
        refs=([(EventRefKind.PARTICLE, inserted.id)] if inserted is not None else []),
        payload={
            "resolution": "accepted",
            "candidate_event_id": candidate_event_id,
            "particle_id": inserted.id if inserted is not None else None,
        },
    )
    # ``reconcile_and_insert`` returns None only when trust resolution kept an
    # existing particle; for an operator-accepted abstraction that means the
    # claim already stood — surface the acceptance either way.
    return inserted if inserted is not None else particle


async def reject_candidate(
    session: AsyncSession,
    candidate_event_id: str,
    *,
    actor: str = "curate",
    reason: str | None = None,
) -> None:
    """Record a rejected candidate (the curation ``reject`` gesture, §6).

    No store mutation beyond the ``ABSTRACTION_RESOLVED`` event — the verdict
    is the payload, and every rejection is a labelled datapoint for the §8
    faithfulness evaluation. Does not commit.
    """
    await _load_pending_candidate(session, candidate_event_id)
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.ABSTRACTION_RESOLVED,
        reason=reason,
        payload={
            "resolution": "rejected",
            "candidate_event_id": candidate_event_id,
            "particle_id": None,
        },
    )


async def pending_candidate_events(session: AsyncSession) -> list[Any]:
    """Unresolved ABSTRACTION_CANDIDATE events, oldest first (the card finder's feed)."""
    from particles.store.event_store import list_events

    candidates = await list_events(
        session, event_type=OperatorEventType.ABSTRACTION_CANDIDATE, limit=1000
    )
    resolutions = await list_events(
        session, event_type=OperatorEventType.ABSTRACTION_RESOLVED, limit=1000
    )
    resolved_ids = {(r.payload or {}).get("candidate_event_id") for r in resolutions}
    pending = [c for c in candidates if c.event_id not in resolved_ids]
    pending.reverse()  # list_events is newest-first; the queue reads oldest-first
    return pending
