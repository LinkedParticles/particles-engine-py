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
built for, but which the truth-apt gate hid) is reachable. The pure §6.6 verdict
comes from :func:`particles.core.conflict_resolution.resolve_conflict` (amended
to run the supersession branch above the truth-apt gate); this
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
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.conflict_resolution import ConflictVerdict, resolve_conflict
from particles.core.observer_scope import PairPrecondition
from particles.core.schema import CorpusEntry, Particle, ProvenanceRefType
from particles.core.status import Status, StatusReason
from particles.corpus.supersession import iter_supersession_entry_pairs
from particles.embeddings import cosine_similarity
from particles.ingest.observer_gate import ObserverGate
from particles.ingest.pipeline import _contradiction_prompt, _is_attribution_paraphrase
from particles.operations._llm import _llm_call
from particles.store.particle_store import (
    get_active_particles_with_embeddings,
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
    response = await _llm_call(_contradiction_prompt(content_a, content_b), max_tokens=120)
    if response is None:
        return None
    return response.strip().upper().startswith("YES")


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

    # Load ACTIVE particles + their stored embeddings once and group them by
    # source corpus entry, keeping only entries that appear in a supersession
    # pair. Embeddings are already stored — no re-embedding cost.
    relevant = {eid for pair in entry_pairs for eid in pair}
    by_entry: dict[str, list[EmbeddingPair]] = {}
    for particle, emb in await get_active_particles_with_embeddings(session):
        eid = _source_entry_id(particle)
        if eid is not None and eid in relevant:
            by_entry.setdefault(eid, []).append((particle, emb))

    # Phase 1 — collect every probe-worthy candidate pair (no LLM spend).
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
            await update_particle_status(
                session,
                sub_p.id,
                Status.PROVENANCE_STALE,
                StatusReason.DOCUMENT_SUPERSEDED,
            )
        if progress is not None:
            verb = "would demote" if dry_run else "demoted"
            progress(f"{verb} {sub_p.id[:8]} (superseded by {best_sup.id[:8]}, sim {best_sim:.3f})")

    if not dry_run and demoted_ids:
        await session.commit()

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
       ``consolidation.max_update_probes``. A confirmed pair demotes its older
       member ``PROVENANCE_STALE`` / ``SUPERSEDED_BY_UPDATE`` and points the
       newer one's ``supersedes`` at it. A particle demoted earlier in the run
       is never probed again, so a subject carrying three generations
       converges on the newest in one pass.

    ``scope_ids`` (the delta scope) keeps a pair only when at least
    one member is in scope. Idempotent: a re-run demotes nothing.

    On a rescoped store a pair the observer precondition does not reconcile —
    another project observes the older claim, or the older claim is global — is
    dropped before any probe and counted in ``observer_declined``.
    """
    from particles.ingest.update_supersession import SubjectIndex
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

    pairs = await get_active_particles_with_embeddings(session)
    index = SubjectIndex.build(pairs)
    summary["subject_groups"] = len(index.by_subject)
    entries: dict[str, CorpusEntry | None] = {}
    # the observer precondition covers the backlog too. Here the
    # would-be winner is itself a stored claim, so its current scope is the
    # candidate's. A declined pair is skipped before any probe is spent.
    gate = await ObserverGate.open(session)
    declined = 0

    # Phase 1 — collect qualifying pairs. No LLM spend, and the update_order
    # pre-filter is what keeps it affordable.
    seen: set[tuple[str, str]] = set()
    candidates: list[tuple[float, Particle, Particle]] = []
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
                if sim < update_cfg.subject_floor:
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
                    declined += 1
                    continue
                candidates.append((sim, newer, older))

    # Phase 2 — probe highest-cosine first, under the cap.
    candidates.sort(key=lambda c: c[0], reverse=True)
    demoted_ids: set[str] = set()
    probed = 0
    demotions: list[dict[str, object]] = []
    for sim, newer, older in candidates:
        if probed >= probe_cap:
            break
        if newer.id in demoted_ids or older.id in demoted_ids:
            continue
        probe = await _has_contradiction_signal(newer.content, older.content)
        probed += 1
        verdict = resolve_conflict(
            older,
            newer,
            has_contradiction_signal=(probe is True),
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

    if not dry_run and demoted_ids:
        await session.commit()

    summary["candidate_pairs"] = len(candidates)
    summary["observer_declined"] = declined
    summary["probed"] = probed
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
        "Update-supersession sweep: %d subject group(s), %d qualifying pair(s), %d probed, "
        "%d demoted%s",
        summary["subject_groups"],
        len(candidates),
        probed,
        len(demoted_ids),
        " (dry run)" if dry_run else "",
    )
    return summary
