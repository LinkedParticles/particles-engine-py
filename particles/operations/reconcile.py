# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-entry document-supersession reconcile sweep.

Activates the §6.6 rung-1.5 document-supersession prior (cap. 2) on a
realistic deposit → extract → lint store. ``extract_snapshot`` reconciles
**intra-entry** only, so cross-ADR supersession never fires there; this batch
pass runs §6.6 **cross-entry** over already-extracted ACTIVE particles, scoped
to the corpus-entry pairs that stand in an authored document-supersession
relation, and demotes the superseded claim to ``PROVENANCE_STALE`` /
``DOCUMENT_SUPERSEDED``.

the candidacy deliberately includes **non-truth-apt** particles —
the truth-apt pre-filter is lifted *here*, in the sweep, not in the intra-entry
hot path — so a superseded ``CONSTITUTIVE`` definition (the case rung 1.5 was
built for, but which the adjudicability gate hid) is reachable. The pure §6.6 verdict
comes from :func:`particles.core.conflict_resolution.resolve_conflict` (amended
to run the supersession branch above the adjudicability gate); this
module applies only the demotion side effect.

Conflict-gated and idempotent: a pair demotes **only** on a confirmed
replacement signal (the reframed contradiction probe), so a still-true
/ agreeing superseded claim is kept (cap. 2(c)); a re-run re-demotes
nothing already off the ACTIVE surface. Demotion-only — the loser
stays in the store, auditable. v1 is single-trust-order only (matching the trust
rung); the multi-contributor extension is deferred.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.conflict_resolution import (
    ConflictVerdict,
    SlotVerdict,
    SweepAction,
    admits_update,
    build_inconsistency_particle,
    is_generic_instance_pair,
    resolve_conflict,
    sweep_action,
)
from particles.core.contradiction_disclosure import ORIGIN_KEY
from particles.core.observer_scope import PairPrecondition
from particles.core.probe_verdict import ProbeKind, VerdictKey, remembered_clear, verdict_key
from particles.core.schema import CorpusEntry, Particle, ProvenanceRefType
from particles.core.status import Status, StatusReason, validate_transition
from particles.corpus.store import get_entry
from particles.corpus.supersession import iter_supersession_entry_pairs
from particles.db import write_transaction
from particles.embeddings import cosine_similarity
from particles.extraction.registry import infer_domain
from particles.ingest.observer_gate import ObserverGate
from particles.ingest.pipeline import (
    _CONTRADICTION_PROBE_MAX_TOKENS,
    _contradiction_prompt,
    _contradiction_verdict,
    _is_attribution_paraphrase,
    _slot_verdict,
    _trigger_ref_for,
    _update_prompt,
    contradiction_prompt_hash,
    update_prompt_hash,
)
from particles.operations._llm import _llm_call, record_unusable_reply
from particles.operations.probe_ledger import recall, remember
from particles.store.particle_store import (
    get_active_particles_with_embeddings,
    get_inconsistency_particles,
    insert_particle,
    update_particle_status,
)

log = logging.getLogger(__name__)

EmbeddingPair = tuple[Particle, np.ndarray[Any, np.dtype[np.float32]]]

#: One probe-ready candidate: (similarity, superseded, superseding, sup_entry,
#: sub_entry). Collected in full, then probed highest-similarity-first under
#: ``consolidation.max_reconcile_probes``.
_Candidate = tuple[float, Particle, Particle, str, str]


async def _has_contradiction_signal(content_a: str, content_b: str) -> bool | None:
    """The replacement-signal probe, routed through the breaker seam.

    Same semantics as :func:`particles.ingest.pipeline._has_contradiction_signal`
    (attribution pre-filter, then the shared L-SEM-01 prompt), but the LLM leg
    goes through :func:`particles.operations._llm._llm_call` so an open circuit
    breaker (bad key, no credit) short-circuits every probe to ``None`` instead
    of hammering a dead API once per candidate pair. ``None`` stays fail-open
    (keep both) at the call site — the default-safe direction.
    """
    if _is_attribution_paraphrase(content_a, content_b):
        return False
    response = await _llm_call(
        _contradiction_prompt(content_a, content_b), max_tokens=_CONTRADICTION_PROBE_MAX_TOKENS
    )
    if response is None:
        return None
    verdict = _contradiction_verdict(response)
    if verdict is None:
        record_unusable_reply("reconcile contradiction probe", response)
    return verdict


#: ``asserted_by`` of the review record the sweep opens for a fixed slot.
UPDATE_SWEEP_ACTOR = "update-sweep"
#: ``conflict:origin`` of that record, so review and the run report can tell it
#: from an extraction-time record and a census record.
FIXED_SLOT_ORIGIN = "update_sweep_fixed_slot"


async def _has_update_signal(earlier: str, later: str) -> SlotVerdict | None:
    """The rung 2.5 update probe, routed through the breaker seam.

    Same prompt and parser as :func:`particles.ingest.pipeline._has_update_signal`.
    ``None`` when the probe could not complete; the sweep reads that as no
    update and keeps both claims, but does not record it as a verdict.
    """
    response = await _llm_call(
        _update_prompt(earlier, later), max_tokens=_CONTRADICTION_PROBE_MAX_TOKENS
    )
    if response is None:
        return None
    verdict = _slot_verdict(response)
    if verdict is None:
        record_unusable_reply("reconcile update probe", response)
    return verdict


async def _open_record_pairs(session: AsyncSession) -> set[frozenset[str]]:
    """Every pair of claims one open INCONSISTENCY record already names."""
    pairs: set[frozenset[str]] = set()
    for record in await get_inconsistency_particles(session):
        ids = [r.corpus_entry_id for r in record.provenance if r.type is ProvenanceRefType.PARTICLE]
        pairs |= {frozenset((x, y)) for i, x in enumerate(ids) for y in ids[i + 1 :] if x != y}
    return pairs


async def _open_fixed_slot_review(session: AsyncSession, older: Particle, newer: Particle) -> str:
    """Open the review record for a fixed slot given two values; return its id.

    Rung 3 as a maintenance pass applies it: disclosure, not quarantine. Both
    claims are already in recall, so neither status changes, as at the nightly
    census. The record is the §6.6 one rung 3 writes at
    extraction, A the older claim and B the newer, marked with its origin. It
    carries no second-reading stamp, so the nightly re-reading checks it like
    an extraction-time record.
    """
    entry_id, snapshot_id, ref_type = _trigger_ref_for(newer)
    entry = await get_entry(session, entry_id) if ref_type is ProvenanceRefType.SOURCE else None
    record = build_inconsistency_particle(
        older,
        newer,
        corpus_entry_id=entry_id,
        snapshot_id=snapshot_id,
        asserted_by=UPDATE_SWEEP_ACTOR,
        trigger_ref_type=ref_type,
    )
    record = record.model_copy(
        update={"properties": {**(record.properties or {}), ORIGIN_KEY: FIXED_SLOT_ORIGIN}}
    )
    validate_transition(None, Status.INCONSISTENCY)
    await insert_particle(
        session, record, domain_hint=infer_domain(entry.source_type) if entry else None
    )
    return record.id


async def update_checks(older: str, newer: str) -> tuple[bool | None, bool | None]:
    """The two checks the update sweep asks of a pair: conflict, then same slot.

    Returns ``(contradiction, same_slot)``. ``same_slot`` is asked only when the
    contradiction check says YES, as in the sweep, and is ``None`` otherwise or
    when the call could not complete. It is True only when the slot check
    allows a retirement: one slot that changes over time. A fixed slot given
    two values reads False, since the sweep keeps both claims and opens a
    review instead. The sweep retires ``older`` only when both are
    True. The ledger is neither read nor written: the memory-rot
    benchmark scores these answers against operator rulings, so it
    asks them fresh.
    """
    contradiction = await _has_contradiction_signal(newer, older)
    if contradiction is not True:
        return contradiction, None
    slot = await _has_update_signal(older, newer)
    return True, None if slot is None else admits_update(slot)


def _source_entry_id(particle: Particle) -> str | None:
    """The corpus entry id of a particle's first SOURCE provenance ref, if any."""
    ref = next((r for r in particle.provenance if r.type == ProvenanceRefType.SOURCE), None)
    return ref.corpus_entry_id if ref is not None else None


def _cosine(
    a: np.ndarray[Any, np.dtype[np.float32]], b: np.ndarray[Any, np.dtype[np.float32]]
) -> float:
    # the normative similarity primitive — normalized cosine clamped to
    # [0, 1]. The threshold this feeds (extraction.similarity_threshold) is on
    # that scale.
    return cosine_similarity(a, b)


async def _supersession_candidates(
    session: AsyncSession, entry_pairs: list[tuple[str, str]], threshold: float
) -> list[_Candidate]:
    """Phase 1 of the document sweep: every probe-worthy pair, no LLM spend.

    Loads ACTIVE particles and their stored embeddings once, groups them by
    source corpus entry (only entries in a supersession pair), and pairs each
    superseded particle with its most similar superseding one above
    ``threshold``. Embeddings are already stored, so there is no re-embedding.
    """
    relevant = {eid for pair in entry_pairs for eid in pair}
    by_entry: dict[str, list[EmbeddingPair]] = {}
    for particle, emb in await get_active_particles_with_embeddings(session):
        eid = _source_entry_id(particle)
        if eid is not None and eid in relevant:
            by_entry.setdefault(eid, []).append((particle, emb))

    candidates: list[_Candidate] = []
    for sup_entry, sub_entry in entry_pairs:
        sup_particles = by_entry.get(sup_entry, [])
        sub_particles = by_entry.get(sub_entry, [])
        if not sup_particles or not sub_particles:
            continue
        for sub_p, sub_emb in sub_particles:
            # Pair the superseded particle with its most-similar superseding one.
            best_sim = 0.0
            best_sup: Particle | None = None
            for sup_p, sup_emb in sup_particles:
                sim = _cosine(sup_emb, sub_emb)
                if sim > best_sim:
                    best_sim, best_sup = sim, sup_p
            if best_sup is None or best_sim < threshold:
                continue
            candidates.append((best_sim, sub_p, best_sup, sup_entry, sub_entry))
    return candidates


async def count_supersession_candidates(session: AsyncSession) -> int:
    """Candidate pairs the document sweep would probe from, uncapped.

    The sweep's gather alone: no probe, no write. ``0`` whenever the sweep
    itself would be a no-op (disabled, multi-trust-order, no entry pairs).
    """
    cfg = get_config()
    if not cfg.document_supersession.enabled or cfg.reconciliation.store_mode != "single":
        return 0
    entry_pairs = await iter_supersession_entry_pairs(session)
    if not entry_pairs:
        return 0
    return len(
        await _supersession_candidates(session, entry_pairs, cfg.extraction.similarity_threshold)
    )


async def reconcile_supersession(
    session: AsyncSession,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run the cross-entry document-supersession reconcile sweep.

    Args:
        dry_run: report what *would* be demoted without mutating the store.
        progress: optional human-readable progress callback (CLI ``--verbose``).

    Returns:
        A summary dict: ``enabled`` and ``single_trust_order`` (the v1 gates),
        ``dry_run``, ``scope_pairs`` (corpus-entry pairs in a supersession
        relation), ``candidate_pairs`` (particle pairs above the similarity
        floor), ``probed`` (replacement-signal probes run — capped at
        ``consolidation.max_reconcile_probes`` per run, spent
        highest-similarity-first; a truncated run is disclosed, never silent),
        ``probe_cap`` (the cap in force), ``demoted`` (count), and
        ``demotions`` (per-demotion winner/loser/entry/similarity records —
        the audit trail; no-silent-truncation).
    """
    # a sweep that demotes ACTIVE particles assumes the surrounding
    # store is schema-current, exactly like Reindex.
    from particles.operations.version_guard import assert_store_schema_current

    await assert_store_schema_current(session)

    cfg = get_config()
    enabled = cfg.document_supersession.enabled
    single_trust_order = cfg.reconciliation.store_mode == "single"
    threshold = cfg.extraction.similarity_threshold

    probe_cap = cfg.consolidation.max_reconcile_probes

    summary: dict[str, object] = {
        "enabled": enabled,
        "single_trust_order": single_trust_order,
        "dry_run": dry_run,
        "scope_pairs": 0,
        "candidate_pairs": 0,
        "probed": 0,
        "probe_cap": probe_cap,
        "demoted": 0,
        "demotions": [],
    }

    if not enabled:
        log.info(
            "Document-supersession disabled (document_supersession.enabled=false); "
            "reconcile sweep is a no-op."
        )
        return summary
    if not single_trust_order:
        # rung 1.5 is single-trust-order only in v1.
        log.info(
            "Store is multi-trust-order; the supersession prior is gated to "
            "single-trust-order stores in v1. Reconcile sweep is a no-op."
        )
        return summary

    entry_pairs = await iter_supersession_entry_pairs(session)
    summary["scope_pairs"] = len(entry_pairs)
    if not entry_pairs:
        return summary
    if progress is not None:
        progress(f"Supersession entry pairs in scope: {len(entry_pairs)}")

    # Phase 1 — collect every probe-worthy candidate pair (no LLM spend).
    candidates = await _supersession_candidates(session, entry_pairs, threshold)

    # Phase 2 — probe highest-similarity-first under the per-run cap
    # (``consolidation.max_reconcile_probes`` correction v1.74.1):
    # each replacement-signal probe is one LLM call, and an unattended sweep
    # must not spend unboundedly. Truncation is disclosed via the summary
    # ("probed X of Y candidate pairs"), in the spirit of the census
    # cap; the highest-similarity pairs are the likeliest true supersessions,
    # so the capped budget goes where the leverage is.
    candidates.sort(key=lambda c: c[0], reverse=True)
    demoted_ids: set[str] = set()
    probed = 0
    demotions: list[dict[str, object]] = []
    for best_sim, sub_p, best_sup, sup_entry, sub_entry in candidates:
        if probed >= probe_cap:
            break
        if sub_p.id in demoted_ids:
            continue

        # Replacement signal — the reframed contradiction probe:
        # "does the superseding claim replace, not merely restate, the
        # superseded one?" None (probe unavailable / breaker open) is treated
        # as False (fail-open / keep both), the default-safe direction.
        probe = await _has_contradiction_signal(best_sup.content, sub_p.content)
        probed += 1
        verdict = resolve_conflict(
            sub_p,  # existing = the superseded-document claim
            best_sup,  # new = the superseding-document claim
            has_contradiction_signal=(probe is True),
            new_supersedes_existing=True,
            existing_supersedes_new=False,
            single_trust_order=single_trust_order,
        )
        # The sweep only ACTS on a document-supersession verdict; any other
        # outcome (CORROBORATES / INCONSISTENT / ALEATORY fallthrough) leaves
        # both claims ACTIVE — the sweep never manufactures an INCONSISTENCY.
        if verdict is not ConflictVerdict.DOCUMENT_SUPERSEDES:
            continue

        demotions.append(
            {
                "superseded_particle_id": sub_p.id,
                "winning_particle_id": best_sup.id,
                "superseded_entry_id": sub_entry,
                "superseding_entry_id": sup_entry,
                "similarity": round(best_sim, 4),
            }
        )
        demoted_ids.add(sub_p.id)
        if not dry_run:
            # Committed before the next probe, under the writer lock: an open
            # write held across the probes is what failed every concurrent
            # writer with ``database is locked``.
            async with write_transaction(session):
                await update_particle_status(
                    session,
                    sub_p.id,
                    Status.PROVENANCE_STALE,
                    StatusReason.DOCUMENT_SUPERSEDED,
                )
        if progress is not None:
            verb = "would demote" if dry_run else "demoted"
            progress(f"{verb} {sub_p.id[:8]} (superseded by {best_sup.id[:8]}, sim {best_sim:.3f})")

    summary["candidate_pairs"] = len(candidates)
    summary["probed"] = probed
    summary["demoted"] = len(demoted_ids)
    summary["demotions"] = demotions
    if probed < len(candidates):
        msg = (
            f"Reconcile probe capped: probed {probed} of {len(candidates)} candidate "
            f"pair(s) (consolidation.max_reconcile_probes = {probe_cap})."
        )
        log.info(msg)
        if progress is not None:
            progress(msg)
    log.info(
        "Document-supersession sweep: %d entry pair(s), %d candidate pair(s), %d probed, "
        "%d demoted%s",
        len(entry_pairs),
        len(candidates),
        probed,
        len(demoted_ids),
        " (dry run)" if dry_run else "",
    )
    return summary


def _source_ref(particle: Particle) -> tuple[str, str | None] | None:
    """``(corpus_entry_id, snapshot_id)`` of a particle's SOURCE ref, if any."""
    ref = next((r for r in particle.provenance if r.type is ProvenanceRefType.SOURCE), None)
    return (ref.corpus_entry_id, ref.snapshot_id) if ref is not None else None


async def _entry_of(
    session: AsyncSession, cache: dict[str, CorpusEntry | None], entry_id: str
) -> CorpusEntry | None:
    """``get_entry`` memoised for one sweep — a subject's claims share entries."""
    if entry_id not in cache:
        from particles.corpus.store import get_entry

        cache[entry_id] = await get_entry(session, entry_id)
    return cache[entry_id]


async def _pair_update_order(
    session: AsyncSession,
    cache: dict[str, CorpusEntry | None],
    a: Particle,
    b: Particle,
    *,
    require_attribution: bool,
) -> int | None:
    """``+1`` when ``a`` is the newer of a rung-2.5-qualifying pair, ``-1`` when ``b`` is."""
    from particles.ingest.update_supersession import latest_source_date, update_order

    a_ref, b_ref = _source_ref(a), _source_ref(b)
    if a_ref is None or b_ref is None:
        return None
    a_entry = await _entry_of(session, cache, a_ref[0])
    b_entry = await _entry_of(session, cache, b_ref[0])
    # The latest observation of each claim, not the first: a value restated
    # after an intervening change is current again, and dating it by its first
    # ref retired exactly those reverted values.
    return update_order(
        a,
        a_entry,
        a_ref[1],
        b,
        b_entry,
        b_ref[1],
        require_attribution=require_attribution,
        new_date=await latest_source_date(session, a, cache),
        existing_date=await latest_source_date(session, b, cache),
    )


async def _set_supersedes_if_unset(session: AsyncSession, winner_id: str, loser_id: str) -> None:
    """Point the winner's ``supersedes`` at the claim it replaced, if it is free.

    The as-of lens dates a retirement from this edge, so the sweep
    leaves the same trail the write-time rung does. An already-set pointer is
    left alone: it records an earlier, equally real supersession.
    """
    from particles.store.particle_store import ParticleRow

    row = await session.get(ParticleRow, winner_id)
    if row is not None and row.supersedes is None:
        row.supersedes = loser_id
        await session.flush()


@dataclass
class _UpdateGather:
    """Phase 1 of the update sweep: its qualifying pairs and what it dropped."""

    candidates: list[tuple[float, Particle, Particle]] = field(default_factory=list)
    subject_groups: int = 0
    declined: int = 0


async def _update_candidates(
    session: AsyncSession,
    *,
    scope_ids: frozenset[str] | None,
    subject_floor: float,
    require_attribution: bool,
) -> _UpdateGather:
    """Same-subject pairs rung 2.5 could act on, ``(similarity, newer, older)``; no LLM spend.

    the observer precondition covers the backlog too. Here the
    would-be winner is itself a stored claim, so its current scope is the
    candidate's. A declined pair is dropped before any probe is spent, and so
    is a generic paired with an instance claim.
    """
    from particles.ingest.update_supersession import SubjectIndex

    index = SubjectIndex.build(await get_active_particles_with_embeddings(session))
    out = _UpdateGather(subject_groups=len(index.by_subject))
    entries: dict[str, CorpusEntry | None] = {}
    gate = await ObserverGate.open(session)
    seen: set[tuple[str, str]] = set()
    for members in index.by_subject.values():
        for i, (a, a_emb) in enumerate(members):
            for b, b_emb in members[i + 1 :]:
                key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                if key in seen:
                    continue
                seen.add(key)
                if scope_ids is not None and a.id not in scope_ids and b.id not in scope_ids:
                    continue
                sim = _cosine(a_emb, b_emb)
                if sim < subject_floor:
                    continue
                # A generic and an instance claim are not an adjudicable pair:
                # never retired as an update, never opened for review.
                if is_generic_instance_pair(a, b):
                    continue
                order = await _pair_update_order(
                    session, entries, a, b, require_attribution=require_attribution
                )
                if order is None:
                    continue
                newer, older = (a, b) if order > 0 else (b, a)
                if gate.engaged and (
                    await gate.verdict(session, await gate.scope_of(session, newer), older)
                    is not PairPrecondition.RECONCILE
                ):
                    out.declined += 1
                    continue
                out.candidates.append((sim, newer, older))
    return out


async def count_update_candidates(session: AsyncSession, scope_ids: frozenset[str] | None) -> int:
    """Qualifying pairs the update sweep would probe from, uncapped.

    The sweep's gather alone: no probe, no write. ``0`` when the sweep is off.
    """
    cfg = get_config()
    update_cfg = cfg.reconciliation.update_supersession
    if not update_cfg.enabled:
        return 0
    gathered = await _update_candidates(
        session,
        scope_ids=scope_ids,
        subject_floor=update_cfg.subject_floor,
        require_attribution=cfg.reconciliation.store_mode != "single",
    )
    return len(gathered.candidates)


# Known deviation: decision logic is interleaved with I/O in this function. Extract it with the
# next substantive change here (D2).
async def reconcile_updates(  # noqa: PLR0912 — one linear two-phase sweep
    session: AsyncSession,
    *,
    dry_run: bool = False,
    scope_ids: frozenset[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Apply same-subject update supersession to the backlog.

    An update is reconciled against the value it replaces **at write time**.
    Everything written before that — and every store upgraded —
    still holds both values ACTIVE, and will until some *new* claim about the
    same subject happens to pair with them. This sweep is that reconciliation,
    run over already-extracted particles.

    Two phases, mirroring :func:`reconcile_supersession`:

    1. **Collect, free.** Group ACTIVE reconcilable particles by subject
       (every subject a particle names), pair each group above
       ``update_supersession.subject_floor``, and keep only pairs
       :func:`~particles.ingest.update_supersession.update_order` already
       qualifies — same lineage, both extractor-asserted, strictly dated.
       Rung 2.5 cannot fire on any other pair, so probing one would be pure
       spend. Most same-subject pairs are about different attributes and cost
       nothing here.
    2. **Probe under a cap.** Highest-cosine first, up to
       ``consolidation.max_update_probes``. A pair is confirmed when the
       contradiction probe says the claims conflict **and** the update probe
       says the newer one gives a new value for the older one's slot;
       the cap counts pairs, and ``update_probed`` counts the second
       calls. A confirmed pair demotes its older
       member ``PROVENANCE_STALE`` / ``SUPERSEDED_BY_UPDATE`` and points the
       newer one's ``supersedes`` at it. A particle demoted earlier in the run
       is never probed again, so a subject carrying three generations
       converges on the newest in one pass. The update probe also names the
       slot's kind, and only a slot that changes over time is retired:
       a fixed slot given two values (an author, a page count)
       opens an INCONSISTENCY record for review with both claims left
       ACTIVE, counted in ``fixed_slot`` and listed in ``reviews``; a pair an
       open record already names is not opened twice.

    ``scope_ids`` (the delta scope) keeps a pair only when at least
    one member is in scope. Idempotent: a re-run demotes nothing and opens no
    second record.

    On a rescoped store a pair the observer precondition does not reconcile —
    another project observes the older claim, or the older claim is global — is
    dropped before any probe and counted in ``observer_declined``.
    """
    from particles.operations.version_guard import assert_store_schema_current

    await assert_store_schema_current(session)

    cfg = get_config()
    update_cfg = cfg.reconciliation.update_supersession
    probe_cap = cfg.consolidation.max_update_probes
    require_attribution = cfg.reconciliation.store_mode != "single"
    summary: dict[str, object] = {
        "enabled": update_cfg.enabled,
        "dry_run": dry_run,
        "subject_groups": 0,
        "candidate_pairs": 0,
        "probed": 0,
        "update_probed": 0,
        "different_slot": 0,
        "fixed_slot": 0,
        "reviews": [],
        "previously_cleared": 0,
        "probe_cap": probe_cap,
        "demoted": 0,
        "demotions": [],
        "observer_declined": 0,
    }
    if not update_cfg.enabled:
        log.info(
            "Update supersession disabled "
            "(reconciliation.update_supersession.enabled=false); sweep is a no-op."
        )
        return summary

    # Phase 1 — collect qualifying pairs. No LLM spend, and the update_order
    # pre-filter is what keeps it affordable.
    gathered = await _update_candidates(
        session,
        scope_ids=scope_ids,
        subject_floor=update_cfg.subject_floor,
        require_attribution=require_attribution,
    )
    summary["subject_groups"] = gathered.subject_groups
    candidates = gathered.candidates
    declined = gathered.declined

    # A pair the ledger already cleared is dropped before the cap:
    # a recorded NO from either probe means the pair cannot confirm while both
    # claims are unchanged, and a cleared pair never takes a probe from one
    # that has not been asked.
    candidates, cleared = await _drop_cleared_updates(session, candidates)

    # Phase 2 — probe highest-cosine first, under the cap.
    candidates.sort(key=lambda c: c[0], reverse=True)
    demoted_ids: set[str] = set()
    probed = 0
    update_probed = 0
    different_slot = 0
    reviews: list[dict[str, object]] = []
    open_pairs: set[frozenset[str]] | None = None
    demotions: list[dict[str, object]] = []
    for sim, newer, older in candidates:
        if probed >= probe_cap:
            break
        if newer.id in demoted_ids or older.id in demoted_ids:
            continue
        probe = await _has_contradiction_signal(newer.content, older.content)
        probed += 1
        if probe is not None and not _is_attribution_paraphrase(newer.content, older.content):
            # The attribution pre-filter answers without a call; only a model's
            # answer is a verdict worth keeping.
            await remember(
                session,
                ProbeKind.RECONCILE_CONTRADICTION,
                contradiction_prompt_hash(),
                {_contradiction_key(newer, older): probe},
                purpose="semantic_lint",
            )
        if probe is not True:
            continue
        # A contradiction is an update only when both claims fill one slot and
        # the newer one gives it a new value, and only when that slot
        # changes over time. The ledger keeps whether the pair may
        # retire.
        update_probed += 1
        slot = await _has_update_signal(older.content, newer.content)
        if slot is not None:
            await remember(
                session,
                ProbeKind.UPDATE_SLOT,
                update_prompt_hash(),
                {_slot_key(older, newer): admits_update(slot)},
                purpose="semantic_lint",
            )
        action = sweep_action(slot)
        if action is SweepAction.KEEP:
            different_slot += 1
            continue
        if action is SweepAction.REVIEW:
            if open_pairs is None:
                open_pairs = await _open_record_pairs(session)
            key = frozenset((older.id, newer.id))
            review: dict[str, object] = {
                "a_particle_id": older.id,
                "b_particle_id": newer.id,
                "similarity": round(sim, 4),
            }
            if key in open_pairs:
                review["already_open"] = True
            elif not dry_run:
                async with write_transaction(session):
                    review["record_id"] = await _open_fixed_slot_review(session, older, newer)
            open_pairs.add(key)
            reviews.append(review)
            if progress is not None:
                verb = "would open" if dry_run else "opened"
                progress(
                    f"{verb} a review of {older.id[:8]} against {newer.id[:8]} "
                    f"(a fixed slot given two values, sim {sim:.3f})"
                )
            continue
        verdict = resolve_conflict(
            older,
            newer,
            has_contradiction_signal=True,
            update_order=1,
            single_trust_order=cfg.reconciliation.store_mode == "single",
        )
        if verdict is not ConflictVerdict.UPDATE_SUPERSEDES:
            continue
        demotions.append(
            {
                "superseded_particle_id": older.id,
                "winning_particle_id": newer.id,
                "similarity": round(sim, 4),
            }
        )
        demoted_ids.add(older.id)
        if not dry_run:
            # One commit per demotion, as in the document sweep.
            async with write_transaction(session):
                await update_particle_status(
                    session,
                    older.id,
                    Status.PROVENANCE_STALE,
                    StatusReason.SUPERSEDED_BY_UPDATE,
                )
                await _set_supersedes_if_unset(session, newer.id, older.id)
        if progress is not None:
            verb = "would demote" if dry_run else "demoted"
            progress(f"{verb} {older.id[:8]} (superseded by {newer.id[:8]}, sim {sim:.3f})")

    summary["candidate_pairs"] = len(candidates)
    summary["observer_declined"] = declined
    summary["probed"] = probed
    summary["update_probed"] = update_probed
    summary["different_slot"] = different_slot
    summary["fixed_slot"] = len(reviews)
    summary["reviews"] = reviews
    summary["previously_cleared"] = cleared
    if cleared:
        msg = f"Update sweep: {cleared} pair(s) skipped as previously cleared."
        log.info(msg)
        if progress is not None:
            progress(msg)
    summary["demoted"] = len(demoted_ids)
    summary["demotions"] = demotions
    if probed < len(candidates):
        msg = (
            f"Update sweep probe capped: probed {probed} of {len(candidates)} qualifying "
            f"pair(s) (consolidation.max_update_probes = {probe_cap})."
        )
        log.info(msg)
        if progress is not None:
            progress(msg)
    log.info(
        "Update-supersession sweep: %d subject group(s), %d qualifying pair(s), "
        "%d previously cleared, %d probed, %d demoted%s",
        summary["subject_groups"],
        len(candidates),
        cleared,
        probed,
        len(demoted_ids),
        " (dry run)" if dry_run else "",
    )
    return summary


def _contradiction_key(newer: Particle, older: Particle) -> VerdictKey:
    return verdict_key(ProbeKind.RECONCILE_CONTRADICTION, newer.content, older.content)


def _slot_key(older: Particle, newer: Particle) -> VerdictKey:
    # Directional: the slot probe is asked (earlier, later).
    return verdict_key(ProbeKind.UPDATE_SLOT, older.content, newer.content)


async def _drop_cleared_updates(
    session: AsyncSession, candidates: list[tuple[float, Particle, Particle]]
) -> tuple[list[tuple[float, Particle, Particle]], int]:
    """Split off the qualifying pairs the ledger already cleared.

    A pair demotes only when the contradiction probe and the slot probe both
    answer YES, so a recorded NO from either, under that probe's
    current prompt, clears it.
    """
    if not candidates:
        return candidates, 0
    contradiction = await recall(
        session,
        ProbeKind.RECONCILE_CONTRADICTION,
        contradiction_prompt_hash(),
        [_contradiction_key(newer, older) for _, newer, older in candidates],
    )
    slot = await recall(
        session,
        ProbeKind.UPDATE_SLOT,
        update_prompt_hash(),
        [_slot_key(older, newer) for _, newer, older in candidates],
    )
    kept = [
        (sim, newer, older)
        for sim, newer, older in candidates
        if not remembered_clear(
            contradiction.get(_contradiction_key(newer, older)),
            slot.get(_slot_key(older, newer)),
        )
    ]
    return kept, len(candidates) - len(kept)
