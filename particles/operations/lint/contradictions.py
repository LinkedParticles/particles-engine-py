# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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

Every answer the probe and its second reading give is recorded in the
probe-verdict ledger, keyed by both claims' content hashes and the
prompt's version. A pair the ledger already cleared is not asked again while
both claims and the prompt are unchanged; it is counted in
``ContradictionProbeControl.previously_cleared`` and never takes a probe from
the cap.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Final, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.contradiction_disclosure import (
    READING_OBSERVED,
    READING_STANDING,
    ConfirmedPair,
)
from particles.core.equivalence import co_evidential_components
from particles.core.probe_verdict import (
    ProbeKind,
    VerdictKey,
    prompt_hash,
    remembered_clear,
    verdict_key,
)
from particles.core.schema import (
    LintFinding,
    Particle,
    RelationCreatedBy,
    RelationType,
)
from particles.core.status import Status
from particles.extraction.scope import is_excluded_document_meta

# The second reading moved to ``ingest`` so extraction can call it;
# its names are re-exported here for the census and for callers that import
# them from this module.
from particles.ingest.second_reading import (
    _PROBE_RESPONSE_SCHEMA,
    _VERIFY_INSTRUCTION,
    _VERIFY_OBSERVED_INSTRUCTION,
    _VERIFY_PASSAGE_CHARS,
    ClaimContext,
    ProbeVerdict,
    SourceKind,
    _claim_context_for,
    _claim_kind,
    _claim_source,
    _ClaimSource,
    _llm_verify_contradiction,
    _parse_probe_verdict,
    _verify_request,
    claim_context,
    one_line,
    reading_for,
    second_reading_prompt_hash,
    source_kind,
)
from particles.operations._llm import (
    _llm_call,
    _llm_call_many,
    llm_circuit_open,
    record_unusable_reply,
)
from particles.operations.candidate_pairs import (
    CandidatePair,
    enumerate_candidate_pairs,
    is_pair_eligible,
)
from particles.operations.probe_ledger import recall, remember
from particles.store.particle_store import (
    get_active_particles_with_embeddings,
    get_particles_by_ids,
)

if TYPE_CHECKING:
    from particles.llm import CompletionRequest

__all__ = [
    "READING_OBSERVED",
    "READING_STANDING",
    "_PROBE_RESPONSE_SCHEMA",
    "_VERIFY_INSTRUCTION",
    "_VERIFY_OBSERVED_INSTRUCTION",
    "_VERIFY_PASSAGE_CHARS",
    "ClaimContext",
    "ContradictionProbeControl",
    "Disagreements",
    "UNANSWERED",
    "ProbeAnswer",
    "ProbeVerdict",
    "RereadOutcome",
    "SourceKind",
    "_ClaimSource",
    "_claim_context_for",
    "_claim_kind",
    "_claim_source",
    "_llm_verify_contradiction",
    "_parse_probe_verdict",
    "_verify_request",
    "claim_context",
    "contradiction_partner",
    "count_disagreements",
    "one_line",
    "probe_prompt_hash",
    "reading_for",
    "reread_stale_pairs",
    "source_kind",
]

log = logging.getLogger(__name__)


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
    #: Read each pair the probe flagged a second time before it becomes a
    #: finding: on ``llm.verification``, with each claim's source
    #: passage, note name and note date. Only confirmed pairs are reported.
    #: Set by the memory audit; ``particles lint`` and the nightly cycle leave
    #: it off and report every flag.
    verify: bool = False
    #: Second readings at most, spent in probe order. ``None`` = every flag.
    #: A flag past the cap, or one whose second reading failed, is counted in
    #: ``unverified`` and is not reported.
    max_verifications: int | None = None
    #: Called after each second reading with ``(done, total_planned)``, so the
    #: phase is never silent. ``total_planned`` is the flag count,
    #: bounded by ``max_verifications``.
    on_verify_progress: Callable[[int, int], None] | None = None
    #: Pairs never probed, as unordered id pairs: the pairs a census record
    #: already discloses. ``None`` = none. ``particles lint``
    #: passes nothing and stays exhaustive.
    exclude_pairs: frozenset[frozenset[str]] | None = None

    #: Out: pairs that survived every gate (similarity, scope, co-evidential,
    #: stance) — the pairs an unbounded probe would have checked.
    candidate_pairs: int = 0
    #: Out: the intra-scope subset of ``candidate_pairs`` — pairs with **both**
    #: sides in ``scope_particle_ids`` (0 when no scope is set). The audit
    #: renders the tier split from this.
    intra_scope_pairs: int = 0
    #: Out: pairs that passed every gate but were not probed because the
    #: probe-verdict ledger already holds a NO for them under the current
    #: prompt (or, under ``verify``, a NO from the second reading): the claims
    #: are unchanged since they were cleared. Not counted in
    #: ``candidate_pairs``, so they never consume ``max_probes``.
    previously_cleared: int = 0
    #: Out: pairs actually sent to the LLM probe.
    probes_run: int = 0
    #: Out: pairs the probe answered YES for.
    flagged: int = 0
    #: Out, under ``verify``: second readings actually sent (for call accounting).
    verifications_run: int = 0
    #: Out, under ``verify``: flagged pairs the second reading confirmed.
    confirmed: int = 0
    #: Out, under ``verify``: flagged pairs never read a second time (past
    #: ``max_verifications``, or the call failed). Not reported.
    unverified: int = 0
    #: Out: reported pairs whose two claims came from one note, so
    #: a caller can say how many of its findings cross notes.
    same_source_findings: int = 0
    #: Out: every reported pair as ``(particle_a, particle_b, same_source)``,
    #: in report order, for :meth:`disagreements`.
    finding_pairs: list[tuple[str, str, bool]] = field(default_factory=list)
    #: Out, under ``verify``: every confirmed pair with the second reading's
    #: reason, in probe order. What the nightly disclosure pass reads
    #:.
    confirmed_pairs: list[ConfirmedPair] = field(default_factory=list)

    @property
    def capped(self) -> bool:
        """True when ``max_probes`` left candidate pairs unprobed."""
        return self.probes_run < self.candidate_pairs

    def disagreements(self) -> Disagreements:
        """The reported pairs grouped into disagreements."""
        return count_disagreements(self.finding_pairs)


@dataclass(frozen=True)
class Disagreements:
    """Reported contradiction pairs grouped by the claims they share."""

    #: Groups of pairs connected through a shared claim; each counts once.
    groups: int
    #: Groups whose every pair came from one note.
    within_one_note: int
    #: Pairs grouped.
    pairs: int


def count_disagreements(pairs: Sequence[tuple[str, str, bool]]) -> Disagreements:
    """Count disagreements: pairs connected through a shared claim count once.

    Two notes often disagree through more than one claim pair (one claim
    against two phrasings of the other), and one claim can disagree with
    claims in two other notes; either way the reader has one thing to resolve.
    Pure: ``pairs`` are ``(particle_a, particle_b, same_source)``. A group is
    within one note when every pair in it is.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        parent[find(a)] = find(b)
    cross_note_roots = {find(a) for a, _, same in pairs if not same}
    roots = {find(a) for a, _, _ in pairs}
    return Disagreements(
        groups=len(roots),
        within_one_note=len(roots - cross_note_roots),
        pairs=len(pairs),
    )


#: The partner claim named in a CONTRADICTION finding's detail. Both the probe's
#: and the recorded-edge finding's details name it this way.
_PARTNER_RE = re.compile(r"contradiction with particle ([0-9A-Za-z-]+)")


def contradiction_partner(detail: str) -> str | None:
    """The second claim of a CONTRADICTION finding (or card), read from its detail.

    A finding carries one ``particle_id``; the pair's other claim is named in
    the detail. The nightly disclosure uses this to find the card of a pair it
    has disclosed.
    """
    match = _PARTNER_RE.search(detail)
    return match.group(1) if match else None


async def _check_recorded_contradictions(session: AsyncSession) -> list[LintFinding]:
    """Report every recorded ``CONTRADICTS`` edge between two ACTIVE particles — no probe.

    The write path records a contradiction it confirmed but declined to settle,
    because another project observes the existing claim. The
    verdict was already paid for, so this is a structural read, available
    without the semantic pass and outside any probe cap. A pair either side of
    which has since been retired is no longer a live disagreement and is not
    reported.
    """
    from particles.store.relation_store import get_all_relations

    edges = await get_all_relations(session, RelationType.CONTRADICTS)
    if not edges:
        return []
    ids = list({e.particle_a for e in edges} | {e.particle_b for e in edges})
    particles = await get_particles_by_ids(session, ids)
    findings: list[LintFinding] = []
    for edge in edges:
        a, b = particles.get(edge.particle_a), particles.get(edge.particle_b)
        if a is None or b is None or a.status is not Status.ACTIVE or b.status is not Status.ACTIVE:
            continue
        why = (
            "two projects state different values and each keeps its own"
            if edge.created_by is RelationCreatedBy.OBSERVER_DIVERGENCE
            else f"recorded by {edge.created_by.value}"
        )
        findings.append(
            LintFinding(
                particle_id=a.id,
                particle_content=a.content,
                finding_type="CONTRADICTION",
                severity="ERROR",
                detail=f"Recorded contradiction with particle {b.id} ({why}); no probe run",
                recommended_action=(
                    f"Widen, retract or reconcile {a.id} ↔ {b.id} "
                    "(`particles memory widen`, `particles particle retract`)"
                ),
            )
        )
    return findings


async def _recorded_pairs(session: AsyncSession) -> set[frozenset[str]]:
    from particles.store.relation_store import get_all_relations

    return {
        frozenset((e.particle_a, e.particle_b))
        for e in await get_all_relations(session, RelationType.CONTRADICTS)
    }


#: Fresh verdicts are committed to the ledger after this many sequential probes.
_RECORD_EVERY = 50


def _pair_key(c: CandidatePair) -> VerdictKey:
    """The ledger key of one candidate pair. Both census questions are symmetric."""
    return verdict_key(ProbeKind.CONTRADICTION, c.a.content, c.b.content)


async def _drop_cleared(
    session: AsyncSession, pairs: list[CandidatePair], *, verify: bool
) -> tuple[list[CandidatePair], int]:
    """Split off the pairs the ledger already cleared; order is kept.

    A pair is cleared by a recorded NO from the probe under its current prompt.
    Under ``verify`` a recorded NO from the second reading clears it too: a
    flag is reported only once a second reading confirms it, so that NO
    settles the pair whatever the first probe says now.
    """
    if not pairs:
        return pairs, 0
    keys = [_pair_key(c) for c in pairs]
    probed = await recall(session, ProbeKind.CONTRADICTION, probe_prompt_hash(), keys)
    read: dict[VerdictKey, bool] = {}
    if verify:
        read = await recall(session, ProbeKind.SECOND_READING, second_reading_prompt_hash(), keys)
    kept = [
        c
        for c, key in zip(pairs, keys, strict=True)
        if not remembered_clear(probed.get(key), read.get(key))
    ]
    return kept, len(pairs) - len(kept)


async def _record_answers(
    session: AsyncSession, pairs: Sequence[CandidatePair], answers: Sequence[ProbeAnswer]
) -> None:
    """Record each answered probe in the ledger, YES and NO alike."""
    fresh = {
        _pair_key(c): isinstance(answer, str)
        for c, answer in zip(pairs, answers, strict=True)
        if answer is not UNANSWERED
    }
    await remember(
        session, ProbeKind.CONTRADICTION, probe_prompt_hash(), fresh, purpose="semantic_lint"
    )


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
    shared truth to contradict. Pairs already linked CO_EVIDENTIAL (§6.10)
    are skipped — paraphrases, not contradictions. A stance pairs for
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
    from particles.store.relation_store import get_all_relations

    if control is None:
        control = ContradictionProbeControl()
    threshold = get_config().lint.contradiction_candidate_threshold

    # Gather. ACTIVE particles carrying a current-model embedding (the
    # similarity gate needs a vector), minus DOCUMENT_META,
    # non-asserted, and non-truth-apt particles. A rejected /
    # deferred / counterfactual claim must not manufacture an INCONSISTENCY
    # against the chosen decision.
    candidates = [
        (p, emb)
        for p, emb in await get_active_particles_with_embeddings(session)
        if not is_excluded_document_meta(p.properties) and is_pair_eligible(p)
    ]
    if len(candidates) < 2:
        return []
    # Pairs linked CO_EVIDENTIAL are paraphrases, not contradictions: the
    # components are read once, not walked per particle.
    linked = co_evidential_components(
        await get_all_relations(session, RelationType.CO_EVIDENTIAL), 0.0
    )
    # A recorded contradiction is reported by _check_recorded_contradictions
    # without a probe; probing it again would pay twice and
    # report it twice.
    recorded = await _recorded_pairs(session)

    # Decide — enumerate the candidate-pair set (no LLM). Knowing the full set
    # before the first probe is what makes the cap, the "probed X of Y"
    # disclosure, and per-pair done/total progress possible.
    # Intra-scope pairs come first, then mixed; highest similarity
    # first within each tier. Under a cap the LLM budget goes to the harvest's
    # own pairs before coincidental cross-pairs; with no scope every tier is 0
    # and this is the pure similarity order.
    pairs = enumerate_candidate_pairs(
        candidates,
        threshold=threshold,
        linked=linked,
        exclude=recorded | (control.exclude_pairs or frozenset()),
        scope=control.scope_particle_ids,
        # Disagreement between sources is the lens: two claims cut
        # from one note are mostly siblings of one sentence or entries of one
        # series, so under a cap they wait behind cross-note pairs.
        cross_source_first=True,
    )
    # A pair the ledger already cleared under the current prompt is not asked
    # again. It is dropped before the cap is applied, so a cleared
    # pair never takes a probe from one that has not been asked: a capped
    # census reaches further down the similarity order on every run.
    pairs, control.previously_cleared = await _drop_cleared(session, pairs, verify=control.verify)
    if control.previously_cleared:
        log.info(
            "contradiction probe: %d pair(s) skipped as previously cleared",
            control.previously_cleared,
        )
    control.candidate_pairs = len(pairs)
    if control.scope_particle_ids is not None:
        control.intra_scope_pairs = sum(1 for c in pairs if c.tier == 0)
    planned = len(pairs) if control.max_probes is None else min(len(pairs), control.max_probes)

    # Phase 2 — probe the planned prefix. Two shapes, one verdict list: N
    # sequential calls when someone is waiting, or one asynchronous half-price
    # batch when the caller has declared itself latency-tolerant.
    # The probes are independent by construction — each asks about one pair and
    # nothing carries between them — which is exactly what makes the set
    # batchable without changing a single verdict.
    probe_pairs = pairs[:planned]
    verdicts: list[ProbeAnswer] = []
    if control.latency_tolerant:
        verdicts = await _batch_check_contradictions(
            [(c.a.content, c.b.content) for c in probe_pairs]
        )
        control.probes_run = len(probe_pairs)
        if control.on_progress is not None and probe_pairs:
            # One report for the whole batch: under batching there is no
            # per-pair completion moment to report.
            control.on_progress(len(probe_pairs), planned)
        await _record_answers(session, probe_pairs, verdicts)
    else:
        recorded_through = 0
        for done, c in enumerate(probe_pairs, start=1):
            if llm_circuit_open():
                # Account-level failure: every remaining probe would
                # return None without touching the API, so stop rather than
                # walk the rest as instant no-ops. ``probes_run`` stays at the
                # probes that were actually sent.
                log.info(
                    "contradiction probe stopped after %d of %d pairs: LLM unavailable",
                    done - 1,
                    planned,
                )
                break
            verdicts.append(await _llm_check_contradiction(c.a.content, c.b.content))
            control.probes_run = done
            if control.on_progress is not None:
                control.on_progress(done, planned)
            if done - recorded_through >= _RECORD_EVERY:
                # Recorded as the loop goes, so a run that stops part-way keeps
                # what it paid for.
                await _record_answers(
                    session, probe_pairs[recorded_through:done], verdicts[recorded_through:done]
                )
                recorded_through = done
        await _record_answers(
            session, probe_pairs[recorded_through : len(verdicts)], verdicts[recorded_through:]
        )

    flagged = [
        (c, why)
        for c, why in zip(probe_pairs[: len(verdicts)], verdicts, strict=True)
        if isinstance(why, str)
    ]
    control.flagged = len(flagged)
    if control.verify:
        flagged = await _verify_flagged(session, flagged, control)

    findings: list[LintFinding] = []
    for c, contradiction in flagged:
        p_a, p_b = c.a, c.b
        control.same_source_findings += c.same_source
        control.finding_pairs.append((p_a.id, p_b.id, c.same_source))
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


async def _verify_flagged(
    session: AsyncSession,
    flagged: list[tuple[CandidatePair, str]],
    control: ContradictionProbeControl,
) -> list[tuple[CandidatePair, str]]:
    """Keep the flagged pairs a second, context-rich reading confirms.

    Gather each claim's note name, note date and source passage (a store and
    blob read), then ask ``llm.verification`` once per pair, in probe order, up
    to ``control.max_verifications``. The confirmed pairs come back with the
    second reading's reason; the rest are counted, never reported.
    """
    budget = len(flagged) if control.max_verifications is None else control.max_verifications
    confirmed: list[tuple[CandidatePair, str]] = []
    # Each note is read once; the passage window is chosen per pair.
    sources: dict[str, _ClaimSource] = {}
    planned = flagged[:budget]
    read = 0
    readings: dict[VerdictKey, bool] = {}
    for c, _first_reason in planned:
        if llm_circuit_open():
            break
        for p in (c.a, c.b):
            if p.id not in sources:
                sources[p.id] = await _claim_source(session, p)
        ctx_a = _claim_context_for(c.a, sources[c.a.id], c.b)
        ctx_b = _claim_context_for(c.b, sources[c.b.id], c.a)
        verdict = await _llm_verify_contradiction(ctx_a, ctx_b)
        read += 1
        control.verifications_run = read
        if control.on_verify_progress is not None:
            control.on_verify_progress(read, len(planned))
        if verdict is not None:
            readings[_pair_key(c)] = verdict.contradicts
        if verdict is None:
            control.unverified += 1
        elif verdict.contradicts:
            control.confirmed += 1
            confirmed.append((c, verdict.description))
            control.confirmed_pairs.append(
                ConfirmedPair(
                    a=c.a.id,
                    b=c.b.id,
                    same_source=c.same_source,
                    reason=verdict.description,
                    reading=reading_for(ctx_a, ctx_b),
                )
            )
    # Flags past the cap, or left when the breaker opened, were never read.
    control.unverified += len(flagged) - read
    # The census's own readings are recorded; a NO clears the pair
    # for every later verifying run while both claims are unchanged.
    await remember(
        session,
        ProbeKind.SECOND_READING,
        second_reading_prompt_hash(),
        readings,
        purpose="verification",
    )
    return confirmed


#: The probe's output budget. The reply is a one-sentence reason and a verdict
#: line (~60 tokens); 100 cut replies from claude-haiku-4-5 mid-sentence more
#: than 40 times in one audit because the model wrote paragraphs. The verdict
#: comes last, so a reply cut at this budget has none and is never counted.
_PROBE_MAX_TOKENS = 250

#: The probe samples at temperature 0 so one store gives one verdict set: at
#: the provider default the same 200 pairs flagged 13 to 18 contradictions
#: across four runs of one store. A model that rejects the
#: parameter is retried without it by the adapter.
_PROBE_TEMPERATURE = 0.0


class _Unanswered(Enum):
    TOKEN = "unanswered"


#: A probe that produced no verdict: the call failed, the breaker was open, or
#: the reply carried no usable verdict line. Never a contradiction, and never
#: recorded in the probe-verdict ledger.
UNANSWERED: Final = _Unanswered.TOKEN

#: One probe's answer: the contradiction's description (YES), ``None`` (NO), or
#: :data:`UNANSWERED`.
ProbeAnswer = str | None | Literal[_Unanswered.TOKEN]


def _parse_probe_reply(response: str) -> ProbeAnswer:
    """Contradiction description from a probe reply, ``None`` for no contradiction.

    A reply with no usable verdict (cut at the budget, ambiguous, off
    protocol) is counted as a failed probe and returns :data:`UNANSWERED`:
    never a contradiction, and never remembered as a NO.
    """
    verdict = _parse_probe_verdict(response)
    if not verdict.usable:
        record_unusable_reply("contradiction probe", response)
        return UNANSWERED
    return verdict.description if verdict.contradicts else None


def probe_prompt_hash() -> str:
    """The census probe's prompt version, rendered by :func:`_probe_request`.

    The nonce and both claims are fixed placeholders, so the hash changes
    exactly when the trusted instruction or the user-turn template does.
    """
    request = _probe_request("{claim_a}", "{claim_b}", nonce="{nonce}")
    return prompt_hash(request.system or "", request.prompt)


def _probe_request(
    content_a: str, content_b: str, *, nonce: str | None = None
) -> CompletionRequest:
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

    # ``nonce`` is fixed only by :func:`probe_prompt_hash`; a real probe mints its own.
    nonce = nonce or make_nonce()
    system = (
        "Decide whether the two claims in the user message contradict each other: "
        "whether both cannot be true at once. Claims that differ in wording or "
        "detail but can both hold (one narrower, one a restatement or summary of "
        "the other) do not contradict, and neither do claims about different "
        "things.\n\n"
        "Reply with exactly two lines and nothing else:\n"
        "REASON: <one sentence of at most 25 words naming the conflicting detail, "
        "or why there is none>\n"
        "VERDICT: YES or VERDICT: NO\n\n" + data_fence_instruction(nonce)
    )
    user = (
        f"Claim A:\n{fence(content_a, nonce, label='claim_a')}\n\n"
        f"Claim B:\n{fence(content_b, nonce, label='claim_b')}"
    )
    return CompletionRequest(prompt=user, system=system)


async def _llm_check_contradiction(content_a: str, content_b: str) -> ProbeAnswer:
    """A description of the contradiction, ``None`` for none, or :data:`UNANSWERED`."""
    request = _probe_request(content_a, content_b)
    response = await _llm_call(
        request.prompt,
        max_tokens=_PROBE_MAX_TOKENS,
        system=request.system,
        response_schema=_PROBE_RESPONSE_SCHEMA,
        temperature=_PROBE_TEMPERATURE,
    )
    if response is None:
        return UNANSWERED
    return _parse_probe_reply(response)


async def _batch_check_contradictions(pairs: Sequence[tuple[str, str]]) -> list[ProbeAnswer]:
    """Probe every pair as one batch; verdicts align positionally with ``pairs``.

    The batched twin of :func:`_llm_check_contradiction` — same
    prompt, same parser, same :data:`UNANSWERED`-means-unavailable contract,
    one job instead of N calls at half the token price.
    """
    replies = await _llm_call_many(
        [_probe_request(content_a, content_b) for content_a, content_b in pairs],
        max_tokens=_PROBE_MAX_TOKENS,
        response_schema=_PROBE_RESPONSE_SCHEMA,
        latency_tolerant=True,
        temperature=_PROBE_TEMPERATURE,
    )
    return [UNANSWERED if reply is None else _parse_probe_reply(reply) for reply in replies]


# ---------------------------------------------------------------------------
# Re-reading a confirmed pair after its instruction changed
# ---------------------------------------------------------------------------


@dataclass
class RereadOutcome:
    """What re-reading the stale pairs of open census records found."""

    #: By pair key: the pair as the new reading confirmed it, or ``None`` when
    #: the new reading rejected it. A pair not read (current, failed, deferred)
    #: is absent.
    verdicts: dict[frozenset[str], ConfirmedPair | None] = field(default_factory=dict)
    #: The new reading's reason for each withdrawn pair.
    withdrawn_reasons: dict[frozenset[str], str] = field(default_factory=dict)
    read: int = 0
    confirmed: int = 0
    withdrawn: int = 0
    #: Readings sent that returned no usable verdict; re-read on a later night.
    failed: int = 0
    #: Stale pairs past the cap, or left when the breaker opened.
    deferred: int = 0

    def payload(self) -> dict[str, int]:
        return {
            "read": self.read,
            "confirmed": self.confirmed,
            "withdrawn": self.withdrawn,
            "failed": self.failed,
            "deferred": self.deferred,
        }


async def reread_stale_pairs(
    session: AsyncSession,
    pairs: Sequence[ConfirmedPair],
    particles: Mapping[str, Particle],
    *,
    budget: int,
) -> RereadOutcome:
    """Read again each pair confirmed under an instruction it would not get now.

    A pair records the instruction its confirming reading ran under
    (:attr:`ConfirmedPair.reading`). When the instruction a pair would get
    today differs, because the instruction changed or because the pair was
    confirmed before the source kind was read, it is read again on
    ``llm.verification``, in the order given, at most ``budget`` times. A pair
    with a member missing from ``particles`` is skipped: the lapse sweep owns
    it. Read-only: the caller decides what the verdicts do to the records.
    """
    out = RereadOutcome()
    kinds: dict[str, SourceKind] = {}
    sources: dict[str, _ClaimSource] = {}
    for pair in pairs:
        pa, pb = particles.get(pair.a), particles.get(pair.b)
        if pa is None or pb is None:
            continue
        for p in (pa, pb):
            if p.id not in kinds:
                kinds[p.id] = await _claim_kind(session, p)
        observed = SourceKind.RECORD in (kinds[pa.id], kinds[pb.id])
        if (READING_OBSERVED if observed else READING_STANDING) == pair.reading:
            continue
        if out.read >= budget or llm_circuit_open():
            out.deferred += 1
            continue
        for p in (pa, pb):
            if p.id not in sources:
                sources[p.id] = await _claim_source(session, p)
        ctx_a = _claim_context_for(pa, sources[pa.id], pb)
        ctx_b = _claim_context_for(pb, sources[pb.id], pa)
        verdict = await _llm_verify_contradiction(ctx_a, ctx_b)
        out.read += 1
        if verdict is None:
            out.failed += 1
        elif verdict.contradicts:
            out.confirmed += 1
            out.verdicts[pair.key] = replace(
                pair, reason=verdict.description, reading=reading_for(ctx_a, ctx_b)
            )
        else:
            out.withdrawn += 1
            out.verdicts[pair.key] = None
            out.withdrawn_reasons[pair.key] = verdict.description
    return out
