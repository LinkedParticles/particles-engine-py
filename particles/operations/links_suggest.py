"""`links suggest` — co-evidential candidate proposal and resolution.

Within each Subject, compute pairwise cosine similarity between ACTIVE
particle embeddings and propose pairs at or above
``links_suggest.candidate_threshold`` as candidate co-evidential links. Pairs
already linked CO_EVIDENTIAL (transitively) are skipped.

This is the curation workflow that used to live as the L-IDX-01 lint check
(``CO_EVIDENTIAL_CANDIDATE`` findings). Lint is a pure diagnostic again;
suggesting and resolving co-evidential links is a curation operation with its
own three modes:

* ``REPORT`` (default) — list candidate pairs; no LLM, no mutation.
* ``LLM_JUDGE`` — batch each Subject's candidate cluster to the LLM for a
  per-pair PARAPHRASE / DISTINCT / UNSURE verdict; no mutation.
* ``APPLY`` — implies ``LLM_JUDGE``; auto-links PARAPHRASE pairs via
  ``create_relation(..., CO_EVIDENTIAL)``. Guarded by
  ``apply_confirm_threshold`` so a stray invocation can't link thousands.

The AUTO_CLUSTER_V1 path (§ Deferred) is *not* activated here: this
operation is pair-wise, not cluster-wise. AUTO_CLUSTER_V1, if it ships, layers
on top by consuming this operation's output.

**Exact-duplicate auto-merge** lives in the second half of this
module — :func:`find_exact_duplicate_groups` / :func:`auto_merge_exact_duplicates`,
the `particles links dedup` verb. It is a *different mechanism* on the same
duplicate path: Tier A is identical content under the §6.10 normalized key
(:mod:`particles.core.duplicate_key` — the same key extract-time
suppression rung uses), decided by SHA-256 with no embedding, no
threshold, and no LLM call; everything else (Tier B, including cosine `0.9989`)
stays advisory above. Auto-merge is the only store-mutating curation path, and
it is OFF by default.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.duplicate_key import content_hash as norm_content_hash
from particles.core.schema import (
    CandidateCluster,
    CoEvidentialCandidate,
    DedupReport,
    DuplicateGroup,
    JudgeVerdictKind,
    Particle,
    RelationCreatedBy,
    RelationType,
    SuggestMode,
    SuggestReport,
    UnmergeGroup,
    UnmergeReport,
    UnmergeSkip,
    UnmergeSkipReason,
    is_truth_apt,
)
from particles.core.stance import stance_holder
from particles.core.status import Status, StatusReason
from particles.extraction.polarity import is_non_asserted
from particles.operations._scope import pair_scope_tier

if TYPE_CHECKING:
    from particles.store.event_store import OperatorEvent

log = logging.getLogger(__name__)


class ApplyConfirmationRequired(Exception):
    """Raised when ``APPLY`` would link more pairs than the confirm threshold.

    The caller (CLI ``--yes`` / API ``confirmed: true``) must re-invoke with
    ``confirmed=True`` to proceed. Carries ``pair_count`` and ``threshold`` so
    the operator-facing message can be specific.
    """

    def __init__(self, pair_count: int, threshold: int) -> None:
        self.pair_count = pair_count
        self.threshold = threshold
        super().__init__(
            f"--apply would link {pair_count} pairs, more than the "
            f"apply_confirm_threshold of {threshold}. Re-run with --yes to confirm."
        )


def _pair_key(a: str, b: str) -> str:
    """Short id-pair key used in the batch prompt and verdict mapping."""
    return f"{a[:8]}+{b[:8]}"


def _normalise_key(key: str) -> frozenset[str]:
    """Order- and format-tolerant lookup key from an ``a8+b8`` string."""
    return frozenset(part.strip() for part in key.split("+") if part.strip())


async def suggest_co_evidential(
    session: AsyncSession,
    *,
    subject_id: str | None = None,
    threshold: float | None = None,
    mode: SuggestMode = SuggestMode.REPORT,
    confirmed: bool = False,
    scope_particle_ids: frozenset[str] | None = None,
) -> SuggestReport:
    """Propose (and optionally resolve) co-evidential candidate pairs.

    Args:
        session: async DB session.
        subject_id: restrict to one Subject; ``None`` scans every Subject.
        threshold: cosine-similarity floor; falls back to
            ``links_suggest.candidate_threshold`` when ``None``.
        mode: ``REPORT`` / ``LLM_JUDGE`` / ``APPLY`` (see module docstring).
        confirmed: required ``True`` for ``APPLY`` to link more than
            ``links_suggest.apply_confirm_threshold`` pairs.
        scope_particle_ids: harvest scope. Candidate *enumeration* is
            always store-wide (the returned ``clusters`` / ``total_candidates``
            are the full set, so ``REPORT`` callers and the store-wide total are
            unchanged). When set, only pairs with **at least one side** in this
            set are judged in ``LLM_JUDGE`` / linked in ``APPLY`` — bounding the
            audit's ``--judge`` LLM cost to the harvest, symmetric with the contradiction probe — and the judge pass runs in two
            tiers: pairs with **both** sides in scope are judged
            before mixed pairs, highest similarity first within each tier, so
            a judge pass cut short (circuit breaker, transient LLM
            failures) has verdicted the intra-harvest pairs before spending on
            coincidental cross-pairs. ``None`` = judge/apply store-wide.

    Returns:
        A ``SuggestReport`` with one ``CandidateCluster`` per Subject that has
        candidates, plus summary counts and any fan-out / parse warnings.

    Raises:
        ApplyConfirmationRequired: ``APPLY`` mode, PARAPHRASE count exceeds the
            confirm threshold, and ``confirmed`` is ``False``. No links are
            written in that case.
    """
    from particles.config import get_config
    from particles.embeddings import cosine_similarity
    from particles.store.particle_store import get_active_particles_with_embeddings
    from particles.store.relation_store import get_co_evidential_group
    from particles.store.subject_store import (
        get_subject,
        list_all_subjects,
        list_particle_subject_pairs,
    )

    cfg = get_config().links_suggest
    if threshold is None:
        threshold = cfg.candidate_threshold

    warnings: list[str] = []

    if subject_id is not None:
        subj = await get_subject(session, subject_id)
        subjects = [subj] if subj is not None else []
        if subj is None:
            warnings.append(f"Subject {subject_id!r} not found; no candidates produced.")
    else:
        subjects = await list_all_subjects(session)

    # Cluster cache shared across subjects so a multi-subject particle BFSes once.
    cluster_for: dict[str, set[str]] = {}

    async def _cluster(pid: str) -> set[str]:
        if pid not in cluster_for:
            cluster_for[pid] = await get_co_evidential_group(session, pid)
        return cluster_for[pid]

    # fetch the embedding set **once**, store-wide, and group it by
    # Subject in process. This loop used to issue one
    # ``get_active_particles_with_embeddings(subject_id=…)`` per Subject — an
    # N+1 that dominated every curation-queue build: on the 2026-08-02 dogfood
    # store (27,008 ACTIVE particles / 4,009 subjects) those 4,009 queries cost
    # 301.9 s of a 324 s call, while the same rows fetched once cost 3.3 s.
    # Output is unchanged (byte-identical candidate set, measured); only the
    # number of round-trips is. Grouping keys off the ``particle_subjects`` join
    # rows, which is exactly the predicate the per-subject query joined on.
    #
    # co-evidence asserts shared *truth* between two sources, so only
    # truth-apt (FALSIFIABLE) particles are candidates. Opinions / feelings /
    # constitutive rules are never co-evidentially clustered:
    # non-asserted (rejected / deferred / counterfactual) prose is likewise
    # excluded — it is off the factual surface. Both filters are applied once
    # here rather than once per Subject.
    # A single-Subject scan keeps the one targeted join — it is already one
    # query, and a store-wide fetch would make `links suggest --subject X`
    # strictly slower. The batch path is for the store-wide scan, where the
    # per-Subject query is issued once per Subject.
    by_subject: dict[str, list[tuple[Particle, Any]]] = {}

    def _keep(p: Particle) -> bool:
        return is_truth_apt(p) and not is_non_asserted(p.properties)

    if subject_id is not None:
        for subject in subjects:
            by_subject[subject.id] = [
                (p, e)
                for p, e in await get_active_particles_with_embeddings(
                    session, subject_id=subject.id
                )
                if _keep(p)
            ]
    else:
        eligible = {
            p.id: (p, e) for p, e in await get_active_particles_with_embeddings(session) if _keep(p)
        }
        wanted = {s.id for s in subjects}
        for pid, sid in await list_particle_subject_pairs(session):
            if sid in wanted and (entry := eligible.get(pid)) is not None:
                by_subject.setdefault(sid, []).append(entry)

    clusters: list[CandidateCluster] = []
    seen_pairs: set[frozenset[str]] = set()

    for subject in subjects:
        particles_with_embs = by_subject.get(subject.id, [])
        if len(particles_with_embs) < 2:
            continue

        candidates: list[CoEvidentialCandidate] = []
        for i, (p_a, emb_a) in enumerate(particles_with_embs):
            for p_b, emb_b in particles_with_embs[i + 1 :]:
                pair_key = frozenset({p_a.id, p_b.id})
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Skip already-linked pairs (in any direction, transitively).
                if p_b.id in await _cluster(p_a.id):
                    continue

                # M2: a stance is the same claim only as another stance
                # by the SAME holder — never co-evidential with a non-stance (its
                # target) or with a different holder's stance. Merging those would
                # collapse distinct holders' positions, destroying the §4
                # per-holder distribution. Cheap structural skip (holders differ
                # ⇒ at least one is a stance the other is not equal to); two
                # non-stances both read None and pass through unchanged.
                if stance_holder(p_a) != stance_holder(p_b):
                    continue

                # normalized cosine clamped to [0, 1]; the candidate
                # threshold lives on this scale.
                sim = cosine_similarity(emb_a, emb_b)
                if sim < threshold:
                    continue

                candidates.append(
                    CoEvidentialCandidate(particle_a=p_a.id, particle_b=p_b.id, similarity=sim)
                )

        if candidates:
            clusters.append(
                CandidateCluster(
                    subject_id=subject.id,
                    subject_name=subject.canonical_name,
                    candidates=candidates,
                )
            )

    report = SuggestReport(
        mode=mode,
        clusters=clusters,
        total_candidates=sum(len(c.candidates) for c in clusters),
        warnings=warnings,
    )

    if mode is SuggestMode.REPORT:
        return report

    # --- LLM_JUDGE / APPLY: batch-judge each Subject's candidate cluster ---
    # when a harvest scope is given, judge only pairs that touch it
    # (at least one side in scope). The judged candidates are the same object
    # references, so their verdicts propagate into ``clusters``; out-of-scope
    # candidates keep ``verdict=None`` and remain in the store-wide total.
    # the in-scope pairs are judged in two tiers — both-sides-in-scope
    # (intra-harvest) clusters first, mixed (one-side) clusters second, highest
    # similarity first within each — so an interrupted judge pass (
    # circuit breaker, per-call failures) has spent its budget on the pairs a
    # memory audit is about before any coincidental cross-store pair.
    if scope_particle_ids is not None:
        tiered: dict[int, list[CandidateCluster]] = {0: [], 1: []}
        for cl in clusters:
            by_tier: dict[int, list[CoEvidentialCandidate]] = {0: [], 1: []}
            for c in cl.candidates:
                tier = pair_scope_tier(scope_particle_ids, c.particle_a, c.particle_b)
                if tier <= 1:
                    by_tier[tier].append(c)
            for tier, kept in by_tier.items():
                if kept:
                    kept.sort(key=lambda c: (-c.similarity, c.particle_a, c.particle_b))
                    tiered[tier].append(
                        CandidateCluster(
                            subject_id=cl.subject_id,
                            subject_name=cl.subject_name,
                            candidates=kept,
                        )
                    )
        judge_clusters = tiered[0] + tiered[1]
    else:
        judge_clusters = clusters

    content_for = await _content_lookup(session, judge_clusters)

    judged = 0
    for cluster in judge_clusters:
        judged += await _judge_cluster(cluster, content_for, cfg.max_cluster_size, warnings)
    report.judged_pairs = judged

    if mode is SuggestMode.LLM_JUDGE:
        return report

    # --- APPLY: link the PARAPHRASE pairs ---
    paraphrases = [
        c
        for cluster in clusters
        for c in cluster.candidates
        if c.verdict is JudgeVerdictKind.PARAPHRASE
    ]
    if len(paraphrases) > cfg.apply_confirm_threshold and not confirmed:
        raise ApplyConfirmationRequired(len(paraphrases), cfg.apply_confirm_threshold)

    from particles.store.relation_store import create_relation

    applied = 0
    for candidate in paraphrases:
        await create_relation(
            session,
            candidate.particle_a,
            candidate.particle_b,
            RelationType.CO_EVIDENTIAL,
            RelationCreatedBy.LLM_JUDGE,
            confidence=candidate.similarity,
        )
        candidate.applied = True
        applied += 1
    if applied:
        await session.commit()
    report.applied_pairs = applied
    return report


async def _content_lookup(
    session: AsyncSession, clusters: list[CandidateCluster]
) -> dict[str, str]:
    """Map every particle id referenced by the candidates to its content."""
    from particles.store.particle_store import get_particle

    ids: set[str] = set()
    for cluster in clusters:
        for c in cluster.candidates:
            ids.add(c.particle_a)
            ids.add(c.particle_b)

    out: dict[str, str] = {}
    for pid in ids:
        particle = await get_particle(session, pid)
        if particle is not None:
            out[pid] = particle.content
    return out


def _connected_components(candidates: list[CoEvidentialCandidate]) -> list[set[str]]:
    """Group the candidate pairs of one Subject into transitive components.

    Each component is the set of particle ids reachable through candidate
    edges. The fan-out batcher never splits a component across LLM calls.
    """
    adjacency: dict[str, set[str]] = {}
    for c in candidates:
        adjacency.setdefault(c.particle_a, set()).add(c.particle_b)
        adjacency.setdefault(c.particle_b, set()).add(c.particle_a)

    components: list[set[str]] = []
    unvisited = set(adjacency)
    while unvisited:
        start = next(iter(unvisited))
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


def _batch_components(components: list[set[str]], max_cluster_size: int) -> list[set[str]]:
    """Pack components into batches of ≤ ``max_cluster_size`` particles.

    A single component larger than the cap goes in its own (oversized) batch;
    the caller emits a fan-out WARNING for it. Never splits a component.
    """
    batches: list[set[str]] = []
    current: set[str] = set()
    for component in sorted(components, key=len, reverse=True):
        if len(component) > max_cluster_size:
            batches.append(set(component))
            continue
        if current and len(current) + len(component) > max_cluster_size:
            batches.append(current)
            current = set()
        current |= component
    if current:
        batches.append(current)
    return batches


async def _judge_cluster(
    cluster: CandidateCluster,
    content_for: dict[str, str],
    max_cluster_size: int,
    warnings: list[str],
) -> int:
    """Assign an LLM verdict to every candidate in one Subject's cluster.

    Fans out across multiple LLM calls when the candidate graph spans more
    particles than ``max_cluster_size``. Returns the number of pairs judged.
    """
    components = _connected_components(cluster.candidates)
    batches = _batch_components(components, max_cluster_size)
    if len(batches) > 1 or any(len(b) > max_cluster_size for b in batches):
        warnings.append(
            f"Subject '{cluster.subject_name or cluster.subject_id}': candidate "
            f"graph fanned out across {len(batches)} LLM call(s) "
            f"(max_cluster_size={max_cluster_size})."
        )

    # Index candidates by their order-insensitive prefix key for verdict mapping.
    by_key: dict[frozenset[str], CoEvidentialCandidate] = {
        _normalise_key(_pair_key(c.particle_a, c.particle_b)): c for c in cluster.candidates
    }

    judged = 0
    for batch in batches:
        batch_candidates = [
            c for c in cluster.candidates if c.particle_a in batch and c.particle_b in batch
        ]
        if not batch_candidates:
            continue
        verdicts = await _judge_batch(batch_candidates, content_for)
        for raw_key, verdict in verdicts.items():
            candidate = by_key.get(_normalise_key(raw_key))
            if candidate is not None and candidate.verdict is None:
                candidate.verdict = verdict
        # Any candidate the LLM didn't return a verdict for defaults to UNSURE.
        for c in batch_candidates:
            if c.verdict is None:
                c.verdict = JudgeVerdictKind.UNSURE
            judged += 1
    return judged


async def _judge_batch(
    candidates: list[CoEvidentialCandidate], content_for: dict[str, str]
) -> dict[str, JudgeVerdictKind]:
    """Send one batch of candidate pairs to the LLM and parse the verdicts.

    Returns a mapping of ``a8+b8`` pair key → verdict. Pairs the LLM omits or
    returns unparseably are left out of the map (the caller defaults them to
    UNSURE).
    """
    from particles.operations._llm import _llm_call

    lines: list[str] = []
    for c in candidates:
        key = _pair_key(c.particle_a, c.particle_b)
        a = content_for.get(c.particle_a, "(content unavailable)")
        b = content_for.get(c.particle_b, "(content unavailable)")
        lines.append(f"[{key}]\nA: {a}\nB: {b}")

    prompt = (
        "You are curating an epistemic knowledge base. Below are candidate "
        "pairs of claims that may assert the SAME underlying fact (a "
        "paraphrase) or DIFFERENT facts. For each pair, choose exactly one "
        "verdict:\n"
        "- PARAPHRASE: the two claims assert the same underlying fact\n"
        "- DISTINCT: the two claims assert different facts\n"
        "- UNSURE: you cannot tell\n\n"
        "Return ONLY a JSON object mapping each bracketed pair key to its "
        'verdict, e.g. {"c42e15ba+cdcaa152": "PARAPHRASE"}. No prose.\n\n' + "\n\n".join(lines)
    )
    # ~24 tokens per verdict entry is generous; scale with the batch size.
    max_tokens = max(200, 40 * len(candidates))
    response = await _llm_call(prompt, max_tokens=max_tokens)
    return _parse_verdicts(response)


def _parse_verdicts(response: str | None) -> dict[str, JudgeVerdictKind]:
    """Parse the LLM's JSON verdict object, tolerating code fences / prose."""
    if not response:
        return {}
    text = response.strip()
    # Strip a ```json … ``` fence if present, then isolate the JSON object.
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        raw: Any = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, JudgeVerdictKind] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = JudgeVerdictKind(str(value).strip().upper())
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# exact-duplicate auto-merge (Tier A)
# ---------------------------------------------------------------------------


class AutoMergeDisabled(Exception):
    """Raised when ``--apply`` is requested but ``auto_merge.enabled`` is false.

    Default OFF is permanent (§ Constrained): a stock install never
    auto-mutates a store, so the operator must opt in in ``config.yaml`` before
    a merge can write anything. Detection (``--dry-run``) is always available.
    """

    def __init__(self) -> None:
        super().__init__(
            "Exact-duplicate auto-merge is disabled. Set "
            "`links_suggest.auto_merge.enabled: true` in config.yaml to permit "
            "it (default OFF; a stock install never auto-mutates a "
            "store). `--dry-run` needs no flag and mutates nothing."
        )


def _content_hash(content: str) -> str:
    """The Tier-A predicate itself — SHA-256 over the §6.10 *normalized* content.

    Keying the tier on a hash rather than on cosine keeps the predicate
    decidable, model-independent, and stable across the embedding-model swaps
    This is what the § Deferred note anticipates. The measured `1.000000` / `0.998920`
    separation is evidence for the tier, not the mechanism of it.

    **Normalized, not raw bytes**. This was keyed on
    ``sha256(content)`` while extract-time suppression rung keyed on
    :func:`particles.core.duplicate_key.content_hash` (whitespace runs
    collapsed, sentence-final punctuation trimmed, case and wording preserved).
    That left prevention strictly wider than cleanup: a pair differing only by
    a trailing period was never minted twice going forward, yet an existing
    such pair was permanently unreachable by the mop, so an operator reading
    ``--apply``'s "0 groups" as clean was reading it wrong. Both rungs now share
    the one Core key, which is what that module has always claimed to be.
    Measured on the live store (27,008 ACTIVE, 2026-08-02): 206 groups / 225
    redundant copies against raw bytes' 203 / 222 — a strict superset, +1.5 %,
    every added copy a trailing-period twin.
    """
    return norm_content_hash(content)


def _election_key(particle: Particle) -> tuple[bool, datetime, str]:
    """Deterministic survivor ordering: subject-linked, then earliest, then id.

    Determinism is load-bearing — it is what makes a re-run a
    no-op and the revert scriptable. A timezone-naive stored stamp is assumed
    UTC, matching the read-path convention in ``particle_store``.

    The leading term is ``False`` (has subjects), sorting before
    ``True``, so a subject-linked copy always outranks a subject-less one and
    earliest/id ordering decides everything below that. Without it
    the election is Subject-blind, and a group whose earliest copy happens to
    be an orphan would supersede its subject-linked copies underneath it —
    dropping those claims out of §6.7 subject-filtered query entirely. Age is a
    proxy for "the original"; subject-linkage is the thing the store is for, so
    where they disagree indexing wins. The superseded copy's ``asserted_at``
    survives in the merge event either way.
    """
    from datetime import UTC

    asserted = particle.asserted_at
    if asserted.tzinfo is None:
        asserted = asserted.replace(tzinfo=UTC)
    return (not particle.subject_ids, asserted, particle.id)


def _duplicate_components(members: list[Particle]) -> list[list[Particle]]:
    """Split one content-hash bucket into merge groups.

    The subject-linked members form same-Subject connected components.
    Membership is transitive through shared Subjects for the same reason
    `CO_EVIDENTIAL` itself is (``get_co_evidential_group`` BFSes): if A~B and
    B~C are each legitimate same-Subject co-evidential edges over
    *identical* content, A~C is already implied. Computing components up
    front — instead of merging each (Subject, hash) pair independently — is
    what keeps every particle in exactly one group, so no copy can be elected
    survivor in one group and superseded in another.

    The placement rule covers the bucket's **subject-less** members,
    which the earlier rule excluded outright:

    * exactly one linked component → the orphans **join it**. The linked copy
      supplies the referent the orphan is missing and the content is
      identical, so there is one claim and one home.
    * two or more linked components → the orphans form **their own** component.
      Absorbing them into one of several candidate referents would be a guess,
      and this pass does not guess.
    * no linked component → the orphans are the single component.
    """
    by_subject: dict[str, list[Particle]] = {}
    for p in members:
        for sid in p.subject_ids:
            by_subject.setdefault(sid, []).append(p)

    adjacency: dict[str, set[str]] = {p.id: set() for p in members}
    for sharers in by_subject.values():
        ids = [p.id for p in sharers]
        for pid in ids:
            adjacency[pid].update(i for i in ids if i != pid)

    by_id = {p.id: p for p in members}
    components: list[list[Particle]] = []
    unvisited = {p.id for p in members if p.subject_ids}
    while unvisited:
        start = min(unvisited)
        component: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            queue.extend(adjacency[node] - component)
        components.append([by_id[i] for i in component])
        unvisited -= component

    orphans = [p for p in members if not p.subject_ids]
    if orphans:
        if len(components) == 1:
            components[0].extend(orphans)
        else:
            components.append(orphans)
    return components


async def find_exact_duplicate_groups(
    session: AsyncSession,
    *,
    subject_id: str | None = None,
) -> list[DuplicateGroup]:
    """Detect every Tier-A identical-content ACTIVE group.

    Read-only. No embedding scan, no similarity threshold, no LLM call — the
    predicate is exact content equality under the §6.10 normalized key
    (:func:`_content_hash`), so the whole pass is a hash bucket.

    Members must be ACTIVE, truth-apt (co-evidence asserts shared
    *truth*), and asserted (rejected / hypothetical prose is off the
    factual surface). Buckets are additionally keyed on ``stance:holder``
    : two identical sentences held by *different* principals are
    different claims about who believes what, and merging them would collapse
    the per-holder distribution.

    Subject-less duplicates **are** reached. The earlier rule gated
    membership on carrying a Subject and left them behind; that restriction was
    reachability, not safety — at exact identity the safety comes from the
    content key — so the Subject moved to the survivor election
    (``_election_key``) and the orphan-placement rule (``_duplicate_components``)
    instead. Passing ``subject_id`` necessarily excludes subject-less copies,
    since an orphan is in no Subject; a whole-store run is the superset.

    Args:
        session: async DB session.
        subject_id: restrict detection to one Subject; ``None`` scans the store.

    Returns:
        Groups of ≥ 2 copies, ordered deterministically by
        ``(content_hash, survivor_id)``.
    """
    from particles.store.particle_store import get_active_particles

    candidates = [
        p
        for p in await get_active_particles(session)
        if is_truth_apt(p) and not is_non_asserted(p.properties)
    ]
    if subject_id is not None:
        candidates = [p for p in candidates if subject_id in p.subject_ids]

    buckets: dict[tuple[str, str | None], list[Particle]] = {}
    for p in candidates:
        buckets.setdefault((_content_hash(p.content), stance_holder(p)), []).append(p)

    groups: list[DuplicateGroup] = []
    for (digest, _holder), members in buckets.items():
        if len(members) < 2:
            continue
        for component in _duplicate_components(members):
            if len(component) < 2:
                continue
            ordered = sorted(component, key=_election_key)
            survivor, *redundant = ordered
            subjects: list[str] = []
            for p in ordered:
                subjects.extend(s for s in p.subject_ids if s not in subjects)
            linked = sum(1 for p in ordered if p.subject_ids)
            groups.append(
                DuplicateGroup(
                    content_hash=digest,
                    content_excerpt=survivor.content[:160],
                    subject_ids=subjects,
                    survivor_id=survivor.id,
                    redundant_ids=[p.id for p in redundant],
                    subject_class=(
                        "linked"
                        if linked == len(ordered)
                        else ("orphan" if linked == 0 else "mixed")
                    ),
                )
            )

    groups.sort(key=lambda g: (g.content_hash, g.survivor_id))
    return groups


async def auto_merge_exact_duplicates(
    session: AsyncSession,
    *,
    subject_id: str | None = None,
    dry_run: bool = True,
    actor: str = "links-dedup",
) -> DedupReport:
    """Report — and, when explicitly permitted, merge — Tier-A duplicates.

    ``dry_run=True`` (the default) is a pure census: it opens no write, so the
    caller can inspect the blast radius before choosing to act.

    A merge does exactly three things per group, and never a fourth:

    1. Links the survivor to each redundant copy with `CO_EVIDENTIAL`
       (``created_by = EXACT_DUPLICATE``) — this is what preserves the
       corroboration structure the superseded copies' provenance carried
       .
    2. Transitions each redundant copy ``ACTIVE → SUPERSEDED`` with
       ``status_reason = DUPLICATE_MERGED``, through the §6.6 validator.
    3. Records one ``DUPLICATES_MERGED`` event carrying the survivor, every
       superseded id, the content hash, and the config in force.

    It never hard-deletes (supersession is a ledger transition — every copy
    stays readable and recoverable, the append-only discipline applied
    to curation), and never mutates the survivor. Bounding the write set to
    losers-only is what makes the blast radius statable and the revert exact.

    Idempotent: a merged copy is no longer ACTIVE, so a second run finds
    nothing.

    Args:
        session: async DB session.
        subject_id: restrict to one Subject; ``None`` scans the store.
        dry_run: ``True`` reports only and mutates nothing.
        actor: interface entry-point recorded on each event.

    Raises:
        AutoMergeDisabled: ``dry_run=False`` while
            ``links_suggest.auto_merge.enabled`` is false. Nothing is written.
    """
    from particles.config import get_config
    from particles.store.event_store import EventRefKind, OperatorEventType, record_event
    from particles.store.particle_store import update_particle_status
    from particles.store.relation_store import create_relation, get_co_evidential_group

    cfg = get_config().links_suggest.auto_merge
    if not dry_run and not cfg.enabled:
        raise AutoMergeDisabled

    # one id per invocation, stamped on every event this call
    # writes. Without it a run's N events share no key, and "undo the run I
    # just did" is not expressible — which is exactly the state the 181 live
    # events from the 2026-07-25 merge are in.
    run_id = str(uuid.uuid4())

    groups = await find_exact_duplicate_groups(session, subject_id=subject_id)
    report = DedupReport(
        dry_run=dry_run,
        groups=groups,
        total_groups=len(groups),
        total_redundant=sum(len(g.redundant_ids) for g in groups),
    )
    if dry_run or not groups:
        return report

    merging = groups[: cfg.max_per_run]
    deferred = groups[cfg.max_per_run :]
    report.deferred_groups = len(deferred)
    report.deferred_redundant = sum(len(g.redundant_ids) for g in deferred)
    if deferred:
        report.warnings.append(
            f"max_per_run={cfg.max_per_run} bound: merged {len(merging)} group(s); "
            f"{report.deferred_groups} group(s) / {report.deferred_redundant} redundant "
            "copy/copies remain. Re-run to continue."
        )

    for group in merging:
        # Computed once, before any write: adding survivor↔loser edges grows the
        # group, but pre-existing membership is all that can collide.
        already_linked = await get_co_evidential_group(session, group.survivor_id)
        for loser_id in group.redundant_ids:
            if loser_id not in already_linked:
                await create_relation(
                    session,
                    group.survivor_id,
                    loser_id,
                    RelationType.CO_EVIDENTIAL,
                    RelationCreatedBy.EXACT_DUPLICATE,
                    confidence=1.0,
                )
                report.links_created += 1
            await update_particle_status(
                session, loser_id, Status.SUPERSEDED, StatusReason.DUPLICATE_MERGED
            )
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.DUPLICATES_MERGED,
            refs=[
                (EventRefKind.PARTICLE, group.survivor_id),
                *((EventRefKind.PARTICLE, pid) for pid in group.redundant_ids),
            ],
            payload={
                "run_id": run_id,
                "content_hash": group.content_hash,
                "survivor": group.survivor_id,
                "superseded": list(group.redundant_ids),
                "subject_ids": list(group.subject_ids),
                "config": {"enabled": cfg.enabled, "max_per_run": cfg.max_per_run},
            },
        )
        group.merged = True
        report.merged_groups += 1
        report.merged_particles += len(group.redundant_ids)

    await session.commit()
    return report


# ---------------------------------------------------------------------------
# Unmerge — reverting an exact-duplicate auto-merge
# ---------------------------------------------------------------------------


class UnmergeSelectorError(ValueError):
    """Exactly one of ``event_id`` / ``run_id`` / ``since`` must select events."""


async def _select_merge_events(
    session: AsyncSession,
    *,
    event_id: str | None,
    run_id: str | None,
    since: datetime | None,
    until: datetime | None,
) -> tuple[list[OperatorEvent], str]:
    """Resolve one selector to the merge events it names, plus its description."""
    from particles.store.event_store import (
        OperatorEventType,
        get_event,
        list_events_in_range,
    )

    chosen = [n for n, v in (("event", event_id), ("run", run_id), ("since", since)) if v]
    if len(chosen) != 1:
        raise UnmergeSelectorError(
            "Pass exactly one of <event-id>, --run, or --since "
            f"(got {len(chosen)}: {', '.join(chosen) or 'none'})."
        )

    if event_id is not None:
        event = await get_event(session, event_id)
        if event is None:
            raise UnmergeSelectorError(f"No operator event with id {event_id}.")
        if event.event_type is not OperatorEventType.DUPLICATES_MERGED:
            raise UnmergeSelectorError(
                f"Event {event_id} is {event.event_type.value}, not DUPLICATES_MERGED — "
                "unmerge reverts auto-merges only."
            )
        return [event], f"event {event_id}"

    events = await list_events_in_range(
        session,
        event_type=OperatorEventType.DUPLICATES_MERGED,
        since=since,
        until=until,
    )
    if run_id is not None:
        # Filtered in Python rather than in SQL: ``run_id`` lives inside the
        # free-form JSON payload, and JSON extraction is not portable across
        # the backends allowed here. The event log is small enough that the
        # cost is irrelevant.
        events = [e for e in events if (e.payload or {}).get("run_id") == run_id]
        return events, f"run {run_id}"

    window = f"since {since:%Y-%m-%d %H:%M} UTC"
    if until is not None:
        window += f" until {until:%Y-%m-%d %H:%M} UTC"
    return events, window


async def unmerge_exact_duplicates(
    session: AsyncSession,
    *,
    event_id: str | None = None,
    run_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    dry_run: bool = True,
    actor: str = "cli:links-unmerge",
) -> UnmergeReport:
    """Revert one or more auto-merges, restoring the superseded copies.

    The exact inverse of :func:`auto_merge_exact_duplicates`, and it is an
    inverse rather than a re-assertion: the retained rows are restored in
    place, keeping their ids, so ``merge ∘ unmerge`` is the identity. That is
    what *"the pre-merge state is reconstructible exactly"* claim
    rests on, and minting fresh particles would forfeit it.

    Per group, the exact inverse of the merge's three writes:

    1. Transitions each listed copy ``SUPERSEDED → ACTIVE`` and **clears** its
       ``status_reason`` — restoring the row means restoring it, not tagging it
       with a scar. The §6.6 reason gate at the persistence seam is what makes
       this legal; the ``retired_at`` stamp is cleared
       with it (§4).
    2. Deletes the ``CO_EVIDENTIAL`` edge to the survivor **only** when it
       carries ``created_by = EXACT_DUPLICATE`` — a link a human or the judge made for the same pair is never withdrawn (§5).
    3. Records one ``DUPLICATES_UNMERGED`` event. The merge event it reverts is
       never deleted or edited; the pair is the audit trail (§9).

    The survivor is never touched, even when the survivor itself has drifted.
    Drifted copies are skipped and named rather than restored, and never abort
    the rest of the group (§8) — all-or-nothing would let one drifted row block
    recovery of every other copy beside it. Idempotent: a second run finds
    everything ACTIVE and skips it.

    Args:
        event_id: Revert exactly this ``DUPLICATES_MERGED`` event.
        run_id: Revert every event stamped with this merge run.
        since: Revert every merge event at or after this instant. The migration
            path for events written before ``run_id`` existed (§7b) — not a
            general time-travel surface over the ledger.
        until: Optional exclusive upper bound, only with ``since``.
        dry_run: ``True`` plans and reports without opening a write.
        actor: Recorded on each event this call writes.

    Raises:
        UnmergeSelectorError: No selector, more than one, or one that resolves
            to a missing or non-merge event. Nothing is written.
    """
    from particles.store.event_store import EventRefKind, OperatorEventType, record_event
    from particles.store.particle_store import get_particle, update_particle_status
    from particles.store.relation_store import delete_relation

    events, selector = await _select_merge_events(
        session, event_id=event_id, run_id=run_id, since=since, until=until
    )
    report = UnmergeReport(dry_run=dry_run, selector=selector, total_events=len(events))
    if not events:
        report.warnings.append(f"No DUPLICATES_MERGED events matched {selector}.")
        return report

    for event in events:
        payload = event.payload or {}
        survivor_id = payload.get("survivor")
        listed = payload.get("superseded") or []
        if not isinstance(survivor_id, str) or not isinstance(listed, list):
            report.warnings.append(
                f"Event {event.event_id} has no readable survivor/superseded payload; skipped."
            )
            continue

        survivor = await get_particle(session, survivor_id)
        group = UnmergeGroup(
            merge_event_id=event.event_id,
            survivor_id=survivor_id,
            survivor_status=survivor.status.value if survivor else None,
            content_hash=payload.get("content_hash"),
        )

        restorable: list[str] = []
        for loser_id in listed:
            loser = await get_particle(session, loser_id)
            if loser is None:
                group.skipped.append(
                    UnmergeSkip(particle_id=loser_id, reason=UnmergeSkipReason.MISSING)
                )
                continue
            found = UnmergeSkip(
                particle_id=loser_id,
                reason=UnmergeSkipReason.NOT_SUPERSEDED,
                found_status=loser.status.value,
                found_status_reason=(loser.status_reason.value if loser.status_reason else None),
            )
            if loser.status is Status.ACTIVE:
                found.reason = UnmergeSkipReason.ALREADY_ACTIVE
                group.skipped.append(found)
            elif loser.status is not Status.SUPERSEDED:
                group.skipped.append(found)
            elif loser.status_reason is not StatusReason.DUPLICATE_MERGED:
                found.reason = UnmergeSkipReason.NOT_MERGE_SUPERSEDED
                group.skipped.append(found)
            else:
                restorable.append(loser_id)

        if dry_run:
            group.restored_ids = restorable
            # The edge is only withdrawn alongside the copy it belongs to, so
            # the planned count tracks the restorable set, not the listed one.
            group.relations_deleted = len(restorable)
        else:
            for loser_id in restorable:
                await update_particle_status(session, loser_id, Status.ACTIVE, None)
                if await delete_relation(
                    session,
                    survivor_id,
                    loser_id,
                    RelationType.CO_EVIDENTIAL,
                    created_by=RelationCreatedBy.EXACT_DUPLICATE,
                    actor=actor,
                ):
                    group.relations_deleted += 1
                group.restored_ids.append(loser_id)
            if group.restored_ids:
                await record_event(
                    session,
                    actor=actor,
                    event_type=OperatorEventType.DUPLICATES_UNMERGED,
                    refs=[
                        (EventRefKind.PARTICLE, survivor_id),
                        *((EventRefKind.PARTICLE, pid) for pid in group.restored_ids),
                    ],
                    payload={
                        "merge_event_id": event.event_id,
                        "survivor": survivor_id,
                        "restored": list(group.restored_ids),
                        "skipped": [s.model_dump(mode="json") for s in group.skipped],
                        "relations_deleted": group.relations_deleted,
                        "content_hash": group.content_hash,
                    },
                )
                group.reverted = True

        report.groups.append(group)
        report.restored_particles += len(group.restored_ids)
        report.skipped_particles += len(group.skipped)
        report.relations_deleted += group.relations_deleted

    if not dry_run:
        await session.commit()
    return report
