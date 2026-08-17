"""LLM-assisted contradiction detector (L-SEM-01).

Generate candidate pairs across **all** ACTIVE truth-apt particles store-wide
(across corpus entries) by embedding cosine similarity, and ask the
LLM whether each above-threshold pair contradicts. The similarity gate
(``lint.contradiction_candidate_threshold``) is what bounds the candidate set:
only the cheap cosine comparison is O(n²); the expensive LLM probe runs only on
near-neighbour pairs (review F4.3).

This is the contradiction pole of the candidate-pair machinery the
corroboration check ``suggest_co_evidential`` (``L-IDX-01``) already
uses. Unlike co-evidence — which is inherently within a single Subject — a
contradiction can straddle two Subjects the resolver split apart, so this pass
is store-wide, not subject-scoped.

Pairs already linked CO_EVIDENTIAL (§6.10) are skipped — those
particles have been judged paraphrases of the same claim, not contradictions,
so running the LLM check on them would produce a false positive.

The probe is the store's single largest LLM consumer by call count (the nightly
cycle caps it at ``audit.max_contradiction_probes``, currently 1000), and every
probe is independent of every other. A caller that sets
``ContradictionProbeControl.latency_tolerant`` therefore gets the whole planned
prefix submitted as one asynchronous batch at half the token price;
an interactive caller keeps the sequential loop. Same prompt, same parser, same
verdicts either way.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import LintFinding, Particle, is_truth_apt
from particles.core.stance import stance_holder
from particles.extraction.polarity import is_non_asserted
from particles.extraction.scope import is_excluded_document_meta
from particles.operations._llm import _llm_call, _llm_call_many
from particles.operations._scope import pair_scope_tier
from particles.store.particle_store import get_active_particles_with_embeddings

if TYPE_CHECKING:
    from particles.llm import CompletionRequest


@dataclass
class ContradictionProbeControl:
    """Caller-supplied bounding for the LLM probe, plus the probe census.

    The candidate-set seam for the store-wide probe: a
    caller that must bound the probe's LLM cost — the audit's §4 cost
    envelope — passes one of these down
    ``run_memory_audit → collect_cards → run_lint → _check_contradictions``.
    The *in* fields cap, scope, and observe the probe loop; the *out* fields
    are filled by the detector so the caller can disclose "probed X of Y
    candidate pairs" (honesty stance) without any return-shape
    change along the chain. ``particles lint`` itself passes nothing and stays
    uncapped, store-wide.
    """

    #: Probe at most this many candidate pairs (highest similarity first).
    #: ``None`` = unbounded (the ``particles lint`` behaviour).
    max_probes: int | None = None
    #: When set, a candidate pair is kept only if at least one side's particle
    #: id is in this set (the audit's ``--scope harvested`` mode). ``None`` =
    #: store-wide. An empty set keeps no pairs. Surviving pairs are probed in
    #: two tiers: pairs with **both** sides in scope first, mixed
    #: pairs second — so a binding ``max_probes`` never starves the
    #: intra-harvest pairs behind higher-similarity coincidental cross-pairs.
    scope_particle_ids: frozenset[str] | None = None
    #: Called after each LLM probe with ``(done, total_planned)`` where
    #: ``total_planned = min(candidate_pairs, max_probes)``.
    on_progress: Callable[[int, int], None] | None = None
    #: The caller asserts nobody is waiting on this probe, so the planned pairs
    #: may be submitted as one asynchronous half-price batch instead of N
    #: sequential calls. Set by the nightly consolidation cycle;
    #: left ``False`` by ``particles lint`` and the interactive memory audit,
    #: which must not trade an answer-in-seconds for an answer-in-hours. Under
    #: a batch the probes are all in flight at once, so ``on_progress`` reports
    #: the whole planned prefix when the batch lands rather than pair by pair.
    latency_tolerant: bool = False

    #: Out: pairs that survived every gate (similarity, scope, co-evidential,
    #: stance) — the pairs an unbounded probe would have checked.
    candidate_pairs: int = 0
    #: Out: the intra-scope subset of ``candidate_pairs`` — pairs with **both**
    #: sides in ``scope_particle_ids`` (0 when no scope is set). The audit
    #: renders the tier split from this.
    intra_scope_pairs: int = 0
    #: Out: pairs actually sent to the LLM probe.
    probes_run: int = 0

    @property
    def capped(self) -> bool:
        """True when ``max_probes`` left candidate pairs unprobed."""
        return self.probes_run < self.candidate_pairs


async def _check_contradictions(
    session: AsyncSession,
    fix: bool,
    control: ContradictionProbeControl | None = None,
) -> list[LintFinding]:
    """Detect semantic contradictions among ACTIVE particles using LLM (L-SEM-01).

    Candidate pairs are generated store-wide (across corpus entries)
    by embedding cosine similarity at or above
    ``lint.contradiction_candidate_threshold``; only those survivors reach the
    LLM probe. DOCUMENT_META particles and non-truth-apt particles
     never participate — the former are claims about a document's own
    apparatus, the latter (opinions / feelings / constitutive rules) have no
    shared truth to contradict. Pairs already linked CO_EVIDENTIAL (§6.10) are skipped — paraphrases, not contradictions. A stance pairs for
    contradiction only with a *same-holder* stance.

    ``control`` caps / scopes the probe loop and receives
    the candidate-pair census; surviving pairs are probed **highest similarity
    first**, so a cap spends its budget on the closest — most
    contradiction-likely — pairs. ``None`` keeps the unbounded store-wide
    behaviour. Under a harvested scope the similarity order applies **within
    two tiers**: pairs with both sides in scope are probed before
    mixed pairs, so a binding cap goes to the intra-harvest pairs a memory
    audit is about instead of coincidental cross-store neighbours.
    """
    from particles.store.relation_store import get_co_evidential_group

    if control is None:
        control = ContradictionProbeControl()
    threshold = get_config().lint.contradiction_candidate_threshold

    # ACTIVE particles carrying a current-model embedding (the similarity gate
    # needs a vector), minus DOCUMENT_META, non-asserted,
    # and non-truth-apt particles. A rejected / deferred / counterfactual claim
    # must not manufacture an INCONSISTENCY against the chosen decision.
    candidates = [
        (p, emb)
        for p, emb in await get_active_particles_with_embeddings(session)
        if not is_excluded_document_meta(p.properties)
        and not is_non_asserted(p.properties)
        and is_truth_apt(p)
    ]
    if len(candidates) < 2:
        return []

    # Row-normalise once so cosine reduces to a dot product. The stored vectors
    # are already unit-norm; normalise defensively, mirroring ``_find_conflict``.
    emb_matrix: np.ndarray[Any, np.dtype[np.float32]] = np.asarray(
        [emb for _, emb in candidates], dtype=np.float32
    )
    emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10)

    # Per-particle co-evidential cluster cache to avoid repeated BFS.
    cluster_for: dict[str, set[str]] = {}

    async def _cluster(pid: str) -> set[str]:
        if pid not in cluster_for:
            cluster_for[pid] = await get_co_evidential_group(session, pid)
        return cluster_for[pid]

    # Phase 1 — enumerate the candidate-pair set (no LLM). Knowing the full
    # set before the first probe is what makes the cap, the "probed X of Y"
    # disclosure, and per-pair done/total progress possible.
    pairs: list[tuple[int, float, Particle, Particle]] = []
    n = len(candidates)
    for i in range(n):
        p_a = candidates[i][0]
        # Cosine of i against every later particle in one vectorized product;
        # the upper triangle (j > i) visits each unordered pair exactly once.
        # clamp negatives to 0 so this stays on the normative [0, 1]
        # similarity scale (vectorized analogue of embeddings.cosine_similarity;
        # inert for the positive ``threshold`` gate, kept for contract uniformity).
        sims = np.clip(emb_matrix[i + 1 :] @ emb_matrix[i], 0.0, 1.0)
        for offset in np.flatnonzero(sims >= threshold).tolist():
            j = i + 1 + offset
            p_b = candidates[j][0]

            # Scope gate: at least one side must be in scope. The
            # tier — 0 both sides, 1 one side — also drives the probe order
            # below; store-wide (no scope) reads tier 0 throughout.
            tier = pair_scope_tier(control.scope_particle_ids, p_a.id, p_b.id)
            if tier > 1:
                continue

            # Skip pairs linked CO_EVIDENTIAL — paraphrases, not contradictions.
            if p_b.id in await _cluster(p_a.id):
                continue

            # a stance only contradicts a same-holder stance; a
            # stance never contradicts its target, and different-holder stances
            # never contradict. Two non-stances both read None and pass through.
            if stance_holder(p_a) != stance_holder(p_b):
                continue

            pairs.append((tier, float(sims[offset]), p_a, p_b))

    # Intra-scope pairs first, then mixed; highest similarity first
    # within each tier (ids as a deterministic tie-break). Under a cap the LLM
    # budget goes to the harvest's own pairs before coincidental cross-pairs;
    # with no scope every tier is 0 and this is the pure similarity order.
    pairs.sort(key=lambda t: (t[0], -t[1], t[2].id, t[3].id))
    control.candidate_pairs = len(pairs)
    if control.scope_particle_ids is not None:
        control.intra_scope_pairs = sum(1 for t in pairs if t[0] == 0)
    planned = len(pairs) if control.max_probes is None else min(len(pairs), control.max_probes)

    # Phase 2 — probe the planned prefix. Two shapes, one verdict list: N
    # sequential calls when someone is waiting, or one asynchronous half-price
    # batch when the caller has declared itself latency-tolerant.
    # The probes are independent by construction — each asks about one pair and
    # nothing carries between them — which is exactly what makes the set
    # batchable without changing a single verdict.
    probe_pairs = pairs[:planned]
    verdicts: list[str | None] = []
    if control.latency_tolerant:
        verdicts = await _batch_check_contradictions(
            [(p_a.content, p_b.content) for _tier, _sim, p_a, p_b in probe_pairs]
        )
        control.probes_run = len(probe_pairs)
        if control.on_progress is not None and probe_pairs:
            # One report for the whole batch: under batching there is no
            # per-pair completion moment to report.
            control.on_progress(len(probe_pairs), planned)
    else:
        for done, (_tier, _sim, p_a, p_b) in enumerate(probe_pairs, start=1):
            verdicts.append(await _llm_check_contradiction(p_a.content, p_b.content))
            control.probes_run = done
            if control.on_progress is not None:
                control.on_progress(done, planned)

    findings: list[LintFinding] = []
    for (_tier, _sim, p_a, p_b), contradiction in zip(probe_pairs, verdicts, strict=True):
        if contradiction:
            findings.append(
                LintFinding(
                    particle_id=p_a.id,
                    particle_content=p_a.content,
                    finding_type="CONTRADICTION",
                    severity="ERROR",
                    detail=f"Semantic contradiction with particle {p_b.id}: {contradiction}",
                    recommended_action=(
                        f"Create INCONSISTENCY particle or run Review for {p_a.id} ↔ {p_b.id}"
                    ),
                )
            )
    return findings


# The probe's verdict shape: a schema-enforcing adapter (the
# OpenAICompatProvider when the entry's ``structured_output`` enforces) constrains the
# reply to this object; the Anthropic adapter ignores it and keeps answering
# in the YES/NO text protocol. The parser below accepts both.
_PROBE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contradicts": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["contradicts"],
    "additionalProperties": False,
}


def _parse_probe_reply(response: str) -> str | None:
    """Contradiction description from a probe reply, or None for no contradiction.

    Accepts both reply dialects: the enforced JSON verdict object
    (``{"contradicts": true, "description": …}``) and the
    original ``'YES: …'`` / ``'NO'`` text protocol the Anthropic path keeps.
    """
    stripped = response.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "contradicts" in data:
            if data.get("contradicts"):
                return str(data.get("description") or "contradiction detected")
            return None
    if stripped.upper().startswith("YES"):
        return stripped[4:].strip() if len(stripped) > 4 else "contradiction detected"
    return None


def _probe_request(content_a: str, content_b: str) -> CompletionRequest:
    """Build one contradiction probe as a self-contained completion request.

    F3 hardening: the trusted YES/NO instruction goes in the ``system`` turn and
    the two particle contents (LLM-extracted from untrusted sources) go in the
    user turn behind a per-call nonce fence, so a crafted claim cannot coerce a
    verdict. The nonce is minted **per request**, which is why
    :class:`~particles.llm.CompletionRequest` carries its own ``system`` rather
    than sharing one across a batch — a claim that learned the nonce
    from its own probe must not be able to close the fence in a sibling's.
    """
    from particles.llm import CompletionRequest, data_fence_instruction, fence, make_nonce

    nonce = make_nonce()
    system = (
        "Decide whether the two claims in the user message contradict each other. "
        "Answer with exactly one of:\n"
        "- 'YES: <brief description>' if they contradict\n"
        "- 'NO' if they do not contradict or are about different things\n\n"
        + data_fence_instruction(nonce)
    )
    user = (
        f"Claim A:\n{fence(content_a, nonce, label='claim_a')}\n\n"
        f"Claim B:\n{fence(content_b, nonce, label='claim_b')}"
    )
    return CompletionRequest(prompt=user, system=system)


async def _llm_check_contradiction(content_a: str, content_b: str) -> str | None:
    """Return a description of the contradiction or None if no contradiction."""
    request = _probe_request(content_a, content_b)
    response = await _llm_call(
        request.prompt,
        max_tokens=100,
        system=request.system,
        response_schema=_PROBE_RESPONSE_SCHEMA,
    )
    if response is None:
        return None
    return _parse_probe_reply(response)


async def _batch_check_contradictions(pairs: Sequence[tuple[str, str]]) -> list[str | None]:
    """Probe every pair as one batch; verdicts align positionally with ``pairs``.

    The batched twin of :func:`_llm_check_contradiction` — same
    prompt, same parser, same ``None``-means-unavailable contract, one job
    instead of N calls at half the token price.
    """
    replies = await _llm_call_many(
        [_probe_request(content_a, content_b) for content_a, content_b in pairs],
        max_tokens=100,
        response_schema=_PROBE_RESPONSE_SCHEMA,
        latency_tolerant=True,
    )
    return [None if reply is None else _parse_probe_reply(reply) for reply in replies]
