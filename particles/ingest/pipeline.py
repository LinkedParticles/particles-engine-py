# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""§9.2 Extract operation and §6.6 conflict resolution ladder.

Extract derives particles from a corpus snapshot. It is asynchronous and re-runnable.

Conflict resolution ladder (normative, applied in order):
  1. ALEATORY check — if either conflicting particle has ALEATORY uncertainty → INCONSISTENCY
  2. Source trust check; auto-resolves when |score_A - score_B| >= threshold
  3. Default → persist the losing candidate quarantined (PROVENANCE_STALE /
     CONFLICT_PENDING) and create an INCONSISTENCY particle
     referencing both persisted rows; queue for Review

The pure decision logic and INCONSISTENCY-particle constructor live in
:mod:`particles.core.conflict_resolution`; this module owns the DB-touching
half (trust-rank lookups, embedding probe, LLM contradiction-signal call,
status writes).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from opentelemetry import metrics, trace
from sqlalchemy.ext.asyncio import AsyncSession

import particles.store.taxonomy_store  # noqa: F401  # register taxonomy sink
import particles.store.wikidata_cache  # noqa: F401  # register wikidata label cache
from particles.config import get_config
from particles.core.conflict_resolution import (
    RungInputs,
    UpdateOrderSource,
    decide_ladder,
    effective_single_trust_order,
    forces_inconsistent,
    needs_rung_inputs,
    update_order_source,
)
from particles.core.observer_scope import BeliefScope, PairPrecondition
from particles.core.provenance import reobservation_ref
from particles.core.schema import (
    AssertionModality,
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    ExtractorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    is_truth_apt,
)
from particles.core.stance import (
    STANCE_HOLDER_KEY,
    STANCE_MAGNITUDE_KEY,
    holder_from_properties,
    stance_holder,
)
from particles.core.status import Status, StatusReason, validate_transition
from particles.corpus.deposit import load_blob
from particles.corpus.store import (
    claim_snapshot_for_extraction,
    get_entry,
    get_snapshot_superseded_by,
    update_extraction_status,
)
from particles.db import write_lock
from particles.embeddings import cosine_similarity, get_embedding_model
from particles.extraction.calibration import scaler_for_record
from particles.extraction.general import CandidateParticle, PageStat, candidate_to_particle
from particles.extraction.registry import ExtractorPlugin, infer_domain, select_extractor
from particles.extraction.scope import (
    apply_source_exemption,
    is_scope_exempt_source,
)
from particles.extraction.subject_gate import gate_candidate_subjects
from particles.ingest.candidate_dedup import dedupe_exact_candidates
from particles.ingest.conflict_plan import (
    ConflictWritePlan,
    plan_conflict_writes,
    plan_retired_hold,
)
from particles.ingest.duplicate_suppression import (
    DuplicateIndex,
    build_duplicate_index,
    build_retired_value_index,
    is_pending_hold,
    retired_value_note,
    suppression_note,
)
from particles.ingest.generation import cascade_superseded_generation
from particles.ingest.narrative_merge import (
    _merge_label,
    collapse_narratives,
    narrative_labels,
)
from particles.ingest.observer_gate import DivergenceTally, ObserverGate, divergence_note
from particles.ingest.pair_selection import (
    CandidatePairs,
    PairRole,
    PairSelection,
    first_source_ref_entry_id,
    plan_probes,
    plan_update_extras,
    select_pairs,
)
from particles.ingest.routing import Route, route_particle
from particles.ingest.routing import skips_conflict_resolution as _skips_conflict_resolution
from particles.ingest.structure_edges import plan_structure_edges
from particles.ingest.subject_resolver import (
    _persona_canonical,
    find_existing_subject,
    resolve_subjects,
)
from particles.ingest.update_supersession import (
    SubjectIndex,
    candidate_subject_names,
    latest_source_date,
    own_assertion_order,
    particle_subject_ids,
    update_order,
)
from particles.llm import CompletionPool
from particles.store.event_store import record_event
from particles.store.extractor_store import get_calibration
from particles.store.particle_store import (
    append_provenance_ref,
    compute_context_fingerprint,
    get_active_particles_for_entry,
    get_active_particles_with_embeddings,
    get_particles_by_ids,
    insert_particle,
    update_particle_status,
)
from particles.store.relation_store import create_relation
from particles.store.trust_store import get_layered_trust_rank, resolve_trust_score
from particles.store.url_mention_store import build_deposited_url_map, record_url_mentions
from particles.url_canonical import harvest_urls

log = logging.getLogger(__name__)

# Hand-rolled telemetry (Phase 2). These use the OTel **API** directly
# (no-op until setup_observability installs a provider), so the Engine pipeline
# stays decoupled from the SDK. The spans localize where an extract's time goes —
# the embedding offload (the event-loop concern) gets its own child span.
_tracer = trace.get_tracer("particles.ingest")
_meter = metrics.get_meter("particles.ingest")
_embed_duration = _meter.create_histogram(
    "particles.embed.duration", unit="s", description="Embedding batch wall time"
)
_extracted_counter = _meter.create_counter(
    "particles.extracted", unit="1", description="Particles written by an extract pass"
)


# A particle paired with its embedding vector — the unit of the §6.6
# conflict-candidate set.
EmbeddingPair = tuple[Particle, "np.ndarray[Any, np.dtype[np.float32]]"]

# Default agent ID used when no asserted_by is provided
_DEFAULT_AGENT = "general-extractor"


def _current_extraction_provider_model() -> str:
    """``"<provider>:<model>"`` the extraction purpose resolves to right now."""
    sel = get_config().llm.for_purpose("extraction")
    return f"{sel.provider}:{sel.model}"


async def extract_snapshot(
    session: AsyncSession,
    entry_id: str,
    snapshot_id: str,
    extractor: ExtractorPlugin | None = None,
    agent_id: str = _DEFAULT_AGENT,
    page_stats_out: list[PageStat] | None = None,
    supersede_ids: frozenset[str] = frozenset(),
    carry_forward_ids_out: list[str] | None = None,
    suppressed_ids_out: list[str] | None = None,
    completion_pool: CompletionPool | None = None,
    skip_if_superseded: bool = False,
    divergences_out: DivergenceTally | None = None,
) -> list[Particle]:
    """Run extraction for a single corpus snapshot.

    Returns the list of Particle objects written to the store (ACTIVE or INCONSISTENCY).
    Extractor is selected from the plugin registry by source_type.
    If page_stats_out is provided, page-level stats from PDF extraction are appended to it.
    If carry_forward_ids_out is provided, it is extended with the IDs of
    existing ACTIVE particles that the extractor's chunk-hash carry-forward
    matched; reindex callers should exclude these from supersession.
    If suppressed_ids_out is provided, it is extended with the ID of the
    existing ACTIVE particle each suppressed duplicate candidate was folded into,
    one entry per suppressed candidate, so ``len()`` is the suppression count.
    Like carry-forward, these particles must be excluded from supersession.
    If divergences_out is provided, it is incremented by the pairs
    the observer precondition declined and by how many of them this pass joined
    with a ``CONTRADICTS`` relation — the count the quality note carries.
    ``completion_pool`` is the latency-tolerance assertion, threaded
    as a parameter and never sniffed: only the consolidation extract pass
    passes one, and pool-aware extractors then merge their LLM requests into
    the pooled half-price batch. Interactive callers leave it ``None`` and get
    today's sequential calls unchanged.
    ``skip_if_superseded`` is passed ``True`` by the bulk paths: a
    snapshot collapsed since the caller listed its work — by this process or
    another — is skipped with no claim and no LLM call. The default ``False``
    is the explicit ``extract <entry> --snapshot-id`` contract: naming a
    generation extracts it, and the claim un-collapses it.

    Span-wrapped: the work runs under an ``extract.snapshot`` span so a
    request's time localizes across the embed / LLM / DB child spans, and the
    written-particle count feeds the ``particles.extracted`` throughput metric.
    """
    with _tracer.start_as_current_span("extract.snapshot") as span:
        span.set_attribute("particles.entry_id", entry_id)
        span.set_attribute("particles.snapshot_id", snapshot_id)
        written = await _extract_snapshot_impl(
            session,
            entry_id,
            snapshot_id,
            extractor=extractor,
            agent_id=agent_id,
            page_stats_out=page_stats_out,
            supersede_ids=supersede_ids,
            carry_forward_ids_out=carry_forward_ids_out,
            suppressed_ids_out=suppressed_ids_out,
            completion_pool=completion_pool,
            skip_if_superseded=skip_if_superseded,
            divergences_out=divergences_out,
        )
        span.set_attribute("particles.extracted_count", len(written))
        _extracted_counter.add(len(written))
        return written


async def _extract_snapshot_impl(
    session: AsyncSession,
    entry_id: str,
    snapshot_id: str,
    extractor: ExtractorPlugin | None = None,
    agent_id: str = _DEFAULT_AGENT,
    page_stats_out: list[PageStat] | None = None,
    supersede_ids: frozenset[str] = frozenset(),
    carry_forward_ids_out: list[str] | None = None,
    suppressed_ids_out: list[str] | None = None,
    completion_pool: CompletionPool | None = None,
    skip_if_superseded: bool = False,
    divergences_out: DivergenceTally | None = None,
) -> list[Particle]:
    """Body of :func:`extract_snapshot` (span-wrapped by the public function)."""
    # refuse to extract into a store with mismatched-schema
    # particles. The §6.6 conflict resolver would otherwise compare new
    # ACTIVE particles against legacy particles and emit nonsense.
    from particles.operations.version_guard import assert_store_schema_current

    await assert_store_schema_current(session)

    entry = await get_entry(session, entry_id)
    if entry is None:
        raise ValueError(f"Corpus entry {entry_id} not found")

    # Select extractor by registry priority — the routing every
    # other consumer reads back through select_extractor.
    if extractor is None:
        extractor = select_extractor(entry.source_type)

    snapshot = next((s for s in entry.snapshots if s.snapshot_id == snapshot_id), None)
    if snapshot is None:
        raise ValueError(f"Snapshot {snapshot_id} not found in entry {entry_id}")

    if snapshot.archive_path is None:
        log.warning("Snapshot %s has no archive_path; skipping extraction", snapshot_id)
        return []

    # a bulk caller's work list can predate a collapse (another
    # runner, or this one's own pass). Checked before the claim, because the
    # claim un-collapses — which is right for an operator naming a generation
    # and wrong for a loop that merely had it on a stale list.
    if skip_if_superseded:
        superseded_by = await get_snapshot_superseded_by(session, snapshot_id)
        if superseded_by is not None:
            log.info(
                "Snapshot %s: superseded by %s; skipping extraction",
                snapshot_id[:8],
                superseded_by[:8],
            )
            return []

    # Claim the snapshot by committing IN_PROGRESS BEFORE the long LLM
    # call below. Without this commit, the SQLite WAL-WRITE lock is held
    # for the entire extraction (minutes), and a parallel ``particles
    # deposit`` (or any other writer) hits its busy_timeout and fails
    # with ``database is locked``. Committing here also makes the
    # IN_PROGRESS state visible to other ``--all-pending`` runners so
    # they skip this snapshot (the filter at the top of the CLI query is
    # ``extraction_status == PENDING``).
    #
    # The claim helper stamps ``extraction_started_at`` (0.42.2) so the
    # stale-IN_PROGRESS reset in ``extract --all-pending`` can detect
    # rows stranded by a SIGKILL whose try/except cleanup didn't run.
    from datetime import UTC, datetime

    await claim_snapshot_for_extraction(session, snapshot_id, started_at=datetime.now(UTC))
    await session.commit()

    try:
        content = load_blob(snapshot.content_hash)
    except FileNotFoundError:
        await update_extraction_status(session, snapshot_id, ExtractionStatus.FAILED)
        await session.commit()
        raise

    # Snapshot the context fingerprint once before extraction begins.
    # Every candidate produced in this run shares this fingerprint — they were
    # all asserted "in the same context". Carry-forward particles keep their
    # original fingerprint (handled below by leaving theirs in place).
    run_fingerprint = await compute_context_fingerprint(session)

    # On Ctrl+C, programming error, or any other interrupt during the
    # long LLM extraction below, release the IN_PROGRESS claim so the
    # next ``extract --all-pending`` picks the snapshot up. Without this,
    # an interrupted run leaves the snapshot stranded — invisible to the
    # PENDING-filtered scan and recoverable only by direct SQL or
    # ``db init --force`` (which wipes the entire particle store).
    #
    # The cleanup uses a fresh session because ``session`` may be in an
    # indeterminate state after a cancelled await (open transaction,
    # connection-level error). BaseException catches both the synchronous
    # KeyboardInterrupt path and asyncio's CancelledError.
    try:
        result = await extractor.extract(
            snapshot,
            content,
            session=session,
            corpus_entry_id=entry_id,
            source_type=entry.source_type,
            # pass the entry URL so the GitHub repo
            # extractor resolves owner/repo/path without reading the store.
            entry_uri_r=entry.uri_r,
            # the entry's depositor is the actor a migration
            # extractor attributes each imported record to (the ``importer``
            # ContributorRef). Every other extractor ignores the kwarg.
            deposited_by=entry.deposited_by,
            # pool-aware extractors merge their LLM requests into
            # the caller's pooled batch; everyone else ignores the kwarg.
            completion_pool=completion_pool,
            # Reindex marks these particles for replacement; the chunk-hash
            # carry-forward must not count them as cache hits, or
            # a same-version scope like ``reindex --provider-model`` (ADR
            # 0229) never reaches the LLM it exists to re-run.
            supersede_ids=supersede_ids,
        )
    except BaseException:
        from particles.db import session_scope

        try:
            async with session_scope() as cleanup_session:
                await update_extraction_status(
                    cleanup_session, snapshot_id, ExtractionStatus.PENDING
                )
                await cleanup_session.commit()
            log.warning(
                "Snapshot %s: extraction interrupted; reset IN_PROGRESS → PENDING",
                snapshot_id,
            )
        except Exception as cleanup_exc:
            # Best-effort. If the cleanup itself fails (DB gone, disk
            # full), the stale-IN_PROGRESS reset at the next
            # ``--all-pending`` start is the safety net.
            log.warning(
                "Snapshot %s: cleanup after interrupted extraction failed: %s",
                snapshot_id,
                cleanup_exc,
            )
        raise

    # Stamp the run fingerprint on every newly-emitted candidate. Carry-forward
    # candidates are short-circuited by the extractor in this same call,
    # so any candidate the extractor returns here is a fresh
    # extraction belonging to this run.
    for candidate in result.candidates:
        candidate.context_fingerprint = run_fingerprint

    if page_stats_out is not None:
        page_stats_out.extend(result.page_stats)
    if carry_forward_ids_out is not None:
        carry_forward_ids_out.extend(result.carry_forward_ids)

    # API errors are transient (rate limit, billing, server 5xx, network) —
    # the snapshot's *content* is fine, so keep it PENDING so the next
    # `extract --all-pending` retries once the API issue is resolved. Marking
    # FAILED here would hide the snapshot from --all-pending and require the
    # user to know about `reindex`.
    #
    # Detection is structural (``transient_error_count``), never a string-match
    # on the notes: the chunked/PDF paths prefix the "API error: …" note with
    # chunk/page labels, which silently defeated the old ``startswith`` check
    # and stamped fully-failed multi-call extractions COMPLETE with zero
    # particles (F4.1). Any transient failure — even a partial one where some
    # chunks succeeded — resets the whole snapshot: the partial candidates are
    # discarded here (before the insert loop) and carry-forward dedupes the
    # already-succeeded chunks cheaply on the retry, so no claim is silently lost.
    if result.transient_error_count:
        await update_extraction_status(session, snapshot_id, ExtractionStatus.PENDING)
        # Commit the PENDING reset eagerly so concurrent writers see the
        # released-back state. Same rationale as the IN_PROGRESS commit
        # above — paired writes that must be visible without holding the
        # WAL-WRITE lock through the rest of the function.
        await session.commit()
        log.warning(
            "Snapshot %s: %d extraction API call(s) failed; resetting to PENDING for retry "
            "(%d partial candidate(s) discarded)",
            snapshot_id,
            result.transient_error_count,
            len(result.candidates),
        )
        return []

    # collapse per-chunk NARRATIVE fragments (from chunked multi-pass
    # journal extraction) into one whole-entry NARRATIVE with a global
    # SEQUENCE_IN order, before embedding / §6.6 / the write loop see the
    # candidate list. No-op for ≤1 NARRATIVE candidate (single-pass journal,
    # every other extractor), so behaviour is unchanged outside the multi-chunk
    # journal path. The label is the one model call, made here only when a merge
    # is needed; the collapse itself is pure (D2).
    merge_notes: list[str] = []
    fragment_labels = narrative_labels(result.candidates)
    if len(fragment_labels) > 1:
        merge_label = await _merge_label(fragment_labels, merge_notes)
        result.candidates, collapse_notes = collapse_narratives(result.candidates, merge_label)
        merge_notes.extend(collapse_notes)
    if merge_notes:
        log.info("Narrative-merge notes for %s: %s", snapshot_id, merge_notes)

    # fold away intra-pass exact-content duplicate candidates before
    # embedding / §6.6 / the write loop see them. Repetitive multi-section
    # sources (Wikipedia timeline/response sections) restate a sentence verbatim
    # across chunks, and the chunked path mints one candidate per
    # occurrence — so two NEW candidates with identical content would otherwise
    # become two ACTIVE particles in one pass. No-op when nothing duplicates;
    # intra-pass and exact-content only, so carry-forward and cross-source
    # co-evidence (both cross-pass) are untouched.
    result.candidates, dedup_notes = dedupe_exact_candidates(result.candidates)
    if dedup_notes:
        log.info("Intra-pass dedup notes for %s: %s", snapshot_id, dedup_notes)

    # Embed candidates for conflict detection (off the event loop)
    embeddings = await _embed_batch_async([c.content for c in result.candidates])
    # an encoder-free pass is not a slightly-worse pass. Two things
    # break at once, and neither was visible before: §6.6 conflict resolution
    # cannot run (``_find_conflict`` short-circuits on a null candidate vector,
    # so every candidate reads as non-conflicting), and the particles are
    # written with ``embedding_json = NULL``, which
    # ``get_active_particles_with_embeddings`` filters out — so they stay
    # invisible to every later semantic query, dedup and lint until something
    # re-extracts them. That second half outlives the outage, which is why this
    # names the remedy rather than only the symptom.
    if result.candidates and all(e is None for e in embeddings):
        result.quality_notes.append(
            f"No embedding model: §6.6 conflict resolution was skipped for this pass, and "
            f"{len(result.candidates)} particle(s) were stored without embeddings — they "
            f"will not appear in semantic query, dedup or lint results until re-extracted "
            f"(`particles reindex --entry-ids {entry_id}`) with the model available."
        )
        log.warning(
            "Extraction ran without an embedding model for snapshot %s: conflict "
            "resolution skipped and %d particle(s) stored unsearchable.",
            snapshot_id,
            len(result.candidates),
        )

    # Emitted here rather than before the embed step: the note above is a
    # quality note like any other, and logging the list earlier would have
    # silently dropped it.
    if result.quality_notes:
        log.info("Extraction quality notes for %s: %s", snapshot_id, result.quality_notes)

    # Load existing ACTIVE particles for this entry (for conflict detection).
    # Exclude particles that are about to be superseded by this reindex pass so
    # that within-entry re-extraction conflicts are not incorrectly flagged.
    all_existing: list[Particle] = await get_active_particles_for_entry(session, entry_id)
    # DOCUMENT_META and non-asserted particles never
    # participate in §6.6 conflict resolution — see _skips_conflict_resolution.
    existing: list[Particle] = [
        p
        for p in all_existing
        if p.id not in supersede_ids and not _skips_conflict_resolution(p.properties)
    ]
    existing_embs: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    update_cfg = get_config().reconciliation.update_supersession
    all_with_embs: list[tuple[Particle, Any]] = []
    if existing or update_cfg.enabled:
        all_with_embs = await get_active_particles_with_embeddings(session, min_confidence=0.0)
    if existing:
        existing_emb_map = {p.id: emb for p, emb in all_with_embs}
        existing_embs = [existing_emb_map.get(p.id) for p in existing]  # type: ignore[misc]
    # the subject-keyed, cross-entry §6.6 candidate pool — every
    # ACTIVE reconcilable particle grouped by the entity it is about. Built from
    # the same load the same-entry search above already paid for.
    subject_index: SubjectIndex | None = (
        SubjectIndex.build(all_with_embs, exclude_ids=supersede_ids) if update_cfg.enabled else None
    )
    # with no global trust order (a ``multi`` store) rung 2 is off,
    # so a cross-entry pair the update rung cannot act on has exactly one
    # possible outcome — an INCONSISTENCY, and a quarantined *newer* claim.
    # That is worse than the behaviour it was meant to preserve
    # (the pair was never found at all, and both claims stayed ACTIVE), so the
    # subject-scoped search declines such a pair rather than manufacturing a
    # review item. The same-entry 0.80 search is untouched.
    updates_only = get_config().reconciliation.store_mode != "single"

    written: list[Particle] = []

    # Determine extractor identity for provenance (use class-level attrs)
    ext_ref = ExtractorRef(name=extractor.EXTRACTOR_ID, version=extractor.EXTRACTOR_VERSION)
    ext_agent = extractor.EXTRACTOR_ID

    # select the calibration fitted under the model now producing these
    # confidences, keyed by provider_model. Calibration is provider-sensitive,
    # so a record exists for this pairing only if it was benchmarked
    # under it; when present, candidate_to_particle scales raw confidences via
    # TemperatureScaler and stamps CALIBRATED_BENCHMARK. None ⇒ this pairing is
    # uncalibrated (no record, or a swap to an un-benchmarked model), so particles
    # carry calibration_source=EXTRACTOR_DIRECT — the unchanged fallback. Storing
    # per pairing means a swap back to a previously-calibrated model restores its
    # calibration without re-benchmarking.
    current_model = _current_extraction_provider_model()
    calibration = await get_calibration(session, ext_agent, current_model)
    # a record this SDK will not apply is dropped here rather than
    # inside candidate_to_particle, so the operator gets one warning per run
    # instead of one per particle, and everything downstream sees the single
    # uncalibrated state. Every pre-0238 fit is such a record — its temperature
    # both parameterises the retired linear form and was fitted against
    # all-False labels.
    if calibration is not None and scaler_for_record(calibration) is None:
        log.warning(
            "Ignoring the stored calibration for %s under %s: it was fitted before "
            "a format change (transform=%r, T=%.4f) and cannot be applied. Particles carry "
            "EXTRACTOR_DIRECT until you re-fit with `particles extractor calibrate "
            "%s --regenerate`.",
            ext_agent,
            current_model,
            calibration.transform,
            calibration.temperature,
            ext_agent,
        )
        calibration = None
    if calibration is None:
        log.debug(
            "No calibration for %s under %s; particles carry EXTRACTOR_DIRECT "
            "(run `particles extractor calibrate` to fit this pairing).",
            ext_agent,
            current_model,
        )
    else:
        # name the suite set the applied record was fitted over.
        # Whether that set is *stale* cannot be decided here — the predicate
        # needs the benchmark suites, and `tests/` ships in neither the wheel
        # nor the sdist, so a deployment has nothing to compare against
        # (`extractor calibrations` is where the check runs). What this line
        # buys is that a record fitted against a suite set the extractor no
        # longer has is at least *visible* in the extract log at the moment it
        # is applied, instead of silently scaling every confidence.
        log.info(
            "Applying calibration for %s under %s: T=%.4f, fitted over %s "
            "(`particles extractor calibrations %s` reports suite-set staleness).",
            ext_agent,
            current_model,
            calibration.temperature,
            calibration.benchmark_suite_id,
            ext_agent,
        )

    # a rule-source entry exempts its claims from the
    # DOCUMENT_META exclusion. Stamped here — before the §6.6 precompute
    # consults _skips_conflict_resolution — because this is the first point at
    # which the entry's tags (Engine) and the candidate's scope label (Client)
    # are both in hand. The extractor keeps classifying scope in ignorance of
    # the source's genre; the Engine decides what the classification *does*.
    # Under ``mode: suppress`` a DOCUMENT_META candidate is dropped
    # inside the extractor and never reaches here — documented, not overlooked.
    if is_scope_exempt_source(entry.tags):
        exempted = 0
        for candidate in result.candidates:
            stamped = apply_source_exemption(candidate.properties)
            if stamped is not candidate.properties:
                candidate.properties = stamped
                exempted += 1
        if exempted:
            # No silent behaviour change (the disclosure rule).
            result.quality_notes.append(
                f"scope: {exempted} DOCUMENT_META candidate(s) exempted — this source is "
                f"tagged as a rule source"
            )
            log.info(
                "Snapshot %s: %d DOCUMENT_META candidate(s) exempted by source tag",
                snapshot_id,
                exempted,
            )

    # stamp stance:holder (the source author) onto stance
    # candidates BEFORE the §6.6 precompute, so they are treated stance-aware (a
    # stance only conflicts with a same-holder stance). A stance with no
    # derivable holder degrades to a plain claim — aggregation groups by holder,
    # so an unattributable stance has nothing to aggregate. ``stance_specs``
    # records (stance_index, target_index, kind) for post-write edge creation.
    stance_specs: list[tuple[int, int, RelationType]] = []
    if get_config().extraction_stance.enabled and snapshot.author_id:
        for i, candidate in enumerate(result.candidates):
            if candidate.stance_kind is None or candidate.stance_target_index is None:
                continue
            props = dict(candidate.properties or {})
            props[STANCE_HOLDER_KEY] = snapshot.author_id
            if candidate.stance_magnitude is not None:
                props[STANCE_MAGNITUDE_KEY] = candidate.stance_magnitude
            candidate.properties = props
            stance_specs.append((i, candidate.stance_target_index, candidate.stance_kind))

    # F4.3: run the §6.6 contradiction-signal (LLM) checks up front — before the
    # write loop below opens its transaction — so the SQLite WAL write lock is
    # never held across these network round-trips. Commit first to flush any
    # extractor-side writes (subject/label caches) and release the lock. The
    # candidate set is the pre-loop ``existing`` snapshot, which is exactly what
    # the write loop uses (the loop does not reconcile a candidate against its
    # same-run siblings), so precomputing here is behaviour-preserving.
    # ``selections[i]`` is the pair selection for ``result.candidates[i]``;
    # DOCUMENT_META / non-asserted candidates skip §6.6 and pair with nothing
    # (mirrors the write-loop guard below).
    await session.commit()
    # the observer precondition, one gate for every route that can
    # pair a candidate with an existing claim. Open (and free) until the store
    # has been rescoped; every candidate of this pass shares the entry's scope.
    gate = await ObserverGate.open(session)
    candidate_scope = await gate.entry_scope(session, entry_id) if gate.engaged else None
    # The §6.6 pair selection, in ``ingest.pair_selection`` (D2).
    # Gather what every candidate could pair with: the same-entry nearest
    # claim, the subject pool, one batched observer-scope read, and
    # the lineage facts in a ``multi`` store. Then decide which
    # pairs to probe, probe exactly those, and decide the primary pair, the
    # rung 2.5 extras and the declined pairs. ``entries`` caches the corpus
    # entries the lineage read and the extras' apply share.
    entries: dict[str, CorpusEntry | None] = {}
    candidate_pairs = await _gather_candidate_pairs(
        session,
        result.candidates,
        embeddings,
        existing,
        existing_embs,
        subject_index=subject_index,
        gate=gate,
        candidate_scope=candidate_scope,
        entry=entry,
        snapshot_id=snapshot_id,
        updates_only=updates_only,
        entries=entries,
    )
    selections: list[PairSelection] = []
    for candidate, pairs in zip(result.candidates, candidate_pairs, strict=True):
        probes = await _probe_pairs(candidate.content, plan_probes(pairs))
        selections.append(select_pairs(pairs, probes))
    # Per candidate: existing claims the precondition declined and the probe
    # confirmed, recorded as CONTRADICTS once the candidate has an id.
    divergent = [list(selection.declined) for selection in selections]
    # The rung 2.5 extras' entries, read here rather than inside the write lock.
    # In a ``multi`` store the lineage gather has already read every one.
    for pairs, selection in zip(candidate_pairs, selections, strict=True):
        for other in selection.extras:
            key = first_source_ref_entry_id(other)
            if (
                key is not None
                and key not in entries
                and pairs.precondition_of(other) is PairPrecondition.RECONCILE
            ):
                entries[key] = await get_entry(session, key)

    # non-entity subject gate. Drop doc-ID / enum / filename / CLI /
    # snake_case tokens from each candidate's subjects (and the parallel
    # subject_classes / external_refs maps) before resolution, so they are never
    # promoted to Subjects. Lexical + general (covers every extractor at this one
    # choke point); the MCP deliberate-assertion path is intentionally not gated.
    # binding constraint: a code-domain extractor legitimately mints
    # snake_case / dotted code-symbol subjects (e.g. the docstring extractor's
    # ``particles.core.scoring.confidence.effective_confidence``); keying the exemption on
    # the source type keeps the blanket lexical gate from stripping them.
    gate_cfg = get_config().subject_gate
    if gate_cfg.enabled and entry.source_type not in gate_cfg.exempt_source_types:
        for candidate in result.candidates:
            suppressed = gate_candidate_subjects(
                candidate,
                cli_binaries=gate_cfg.cli_binaries,
                allowlist=gate_cfg.allowlist,
            )
            if suppressed:
                log.info(
                    "Subject gate suppressed %d non-entity name(s): %s",
                    len(suppressed),
                    ", ".join(f"{name!r} [{cls}]" for name, cls in suppressed),
                )

    # load the exact-duplicate suppression index for this pass — one
    # indexed probe over ACTIVE particles whose normalized-content hash matches
    # some candidate. Built BEFORE the write loop (so it is a single query, not
    # one per candidate) and maintained in place inside it. ``supersede_ids`` is
    # excluded for a correctness reason, not a performance one: a reindex is
    # about to retire those particles, and suppressing a fresh candidate into
    # one of them would leave the claim with no ACTIVE copy at all.
    dup_index = (
        await build_duplicate_index(
            session,
            [c.content for c in result.candidates],
            exclude_ids=supersede_ids,
        )
        if get_config().extraction.duplicate_suppression.enabled
        else DuplicateIndex()
    )
    suppressed_into: list[str] = []
    # the retired-value index for this pass — judgment-retired twins
    # and open holds of some candidate, one probe, consulted after the ACTIVE
    # index above. ``supersede_ids`` are ACTIVE and so never in it; passed for
    # symmetry with the exact-duplicate index.
    retired_index = (
        await build_retired_value_index(
            session,
            [c.content for c in result.candidates],
            exclude_ids=supersede_ids,
        )
        if get_config().extraction.retired_value_quarantine.enabled
        else DuplicateIndex()
    )
    retired_held: list[str] = []
    retired_absorbed: list[str] = []

    # hold the cross-process writer lock around the write phase
    # only. The LLM extraction and the §6.6 contradiction probes above ran
    # OUTSIDE it (the IN_PROGRESS-claim commit and the line-420 precheck
    # commit both released the SQLite write lock), so the lock is never held
    # across a network round-trip. We commit the particle writes here, under
    # the lock — releasing SQLite's write lock before the advisory lock — so
    # the caller's later commit is a no-op.
    divergences = 0
    async with write_lock():
        # candidate index → the stored particle id that represents it,
        # so post-write edge creation can bind a stance to its (co-extracted) target.
        index_to_id: dict[int, str] = {}
        for i, candidate in enumerate(result.candidates):
            # Resolve subject names/QIDs to Subject UUIDs
            subject_ids = await resolve_subjects(
                session,
                candidate.subjects,
                ext_agent,
                particle_content=candidate.content,
                source_type=entry.source_type,
            )

            # Apply subject classification for structured extractors
            if candidate.subject_classes:
                from particles.store.subject_store import set_subject_class

                for name, subject_id in zip(candidate.subjects, subject_ids, strict=False):
                    cls = candidate.subject_classes.get(name)
                    if cls:
                        await set_subject_class(session, subject_id, cls)

            # Attach external refs for LOD extractors
            if candidate.external_refs:
                from particles.store.subject_store import add_external_ref

                for name, subject_id in zip(candidate.subjects, subject_ids, strict=False):
                    ref = candidate.external_refs.get(name)
                    if ref:
                        await add_external_ref(session, subject_id, ref)

            particle = candidate_to_particle(
                candidate,
                entry_id,
                snapshot_id,
                ext_agent,
                subject_ids=subject_ids,
                extractor_ref=ext_ref,
                calibration=calibration,
            )
            emb = embeddings[i] if i < len(embeddings) else None
            if emb is None:
                emb_list = None
            elif hasattr(emb, "tolist"):
                emb_list = emb.tolist()
            else:
                emb_list = list(emb)

            # The rungs above §6.6 (D2): one pure router shared with
            # ``reconcile_and_insert``, so "duplicate" means one thing on both
            # write paths. The §6.6 conflict candidate and its
            # contradiction signal were precomputed before the write loop
            # (F4.3), keeping the LLM call out of the write transaction. A
            # candidate earlier in this pass may already have demoted that
            # conflict (many claims about one subject per session);
            # it is then no longer ACTIVE and routes to a plain insert.
            selection = selections[i]
            retired = subject_index.retired if subject_index is not None else set()
            route, target = route_particle(
                particle,
                duplicates=dup_index,
                retired_values=retired_index,
                conflict=selection.primary,
                retired_ids=retired,
            )

            if route is Route.UNCHECKED:
                # DOCUMENT_META and non-asserted candidates
                # skip §6.6 entirely — they would only manufacture spurious
                # INCONSISTENCY. Write straight to ACTIVE (stored, label-not-delete).
                validate_transition(None, Status.ACTIVE)
                await insert_particle(session, particle, emb_list)
                written.append(particle)
                index_to_id[i] = particle.id
            elif route is Route.SUPPRESS:
                # this claim is already held verbatim by an ACTIVE
                # particle with the same subjects and stance holder, so don't
                # mint a second one — record this source on the existing
                # particle and move on. Deterministic (content identity, not
                # similarity), so nothing distinguishable leaves the ACTIVE
                # surface. The stance/narrative edge-writers below bind through
                # ``index_to_id``, which maps this candidate's position onto the
                # surviving particle, so an edge whose endpoint was suppressed
                # still lands correctly.
                assert target is not None
                for prov_ref in particle.provenance:
                    await append_provenance_ref(session, target.id, prov_ref)
                suppressed_into.append(target.id)
                index_to_id[i] = target.id
                log.info(
                    "Duplicate suppressed: candidate %d already ACTIVE as %s",
                    i,
                    target.id[:8],
                )
            elif route is Route.RETIRED_TWIN:
                # this exact claim was retired by an operator's or
                # reviewer's judgment, so it does not walk back onto the ACTIVE
                # surface because a source still says it: the candidate is
                # stored quarantined behind an INCONSISTENCY record for review,
                # or — when a hold for this claim is already open — recorded on
                # that hold.
                assert target is not None
                held = await _hold_retired_reassertion(
                    session,
                    particle,
                    emb_list,
                    target,
                    corpus_entry_id=entry_id,
                    snapshot_id=snapshot_id,
                    domain=infer_domain(entry.source_type),
                )
                if held.status is Status.INCONSISTENCY:
                    written.append(held)
                    retired_held.append(held.id)
                else:
                    retired_absorbed.append(held.id)
                index_to_id[i] = held.id
            elif route is Route.INSERT:
                # No conflict — write as ACTIVE
                validate_transition(None, Status.ACTIVE)
                await insert_particle(session, particle, emb_list)
                written.append(particle)
                index_to_id[i] = particle.id
                # Same-pass siblings suppress against it (first writer wins).
                dup_index.add(particle)
            else:
                # Conflict found — apply resolution ladder
                # Returns None when trust resolution kept existing particle (new is dropped)
                assert target is not None
                resolved = await _resolve_conflict(
                    session,
                    particle,
                    target,
                    entry_id,
                    snapshot_id,
                    probe=selection.signal,
                    new_embedding=emb,
                    allow_update_supersession=True,
                    retired_out=retired,
                    observer=gate,
                    candidate_scope=candidate_scope,
                )
                if resolved is not None:
                    written.append(resolved)
                    index_to_id[i] = resolved.id
                    dup_index.add(resolved)
                    if resolved.id == particle.id and resolved.status == Status.ACTIVE:
                        await _apply_update_extras(
                            session,
                            resolved,
                            entry,
                            snapshot_id,
                            [p for p in selection.extras if p.id not in retired],
                            retired,
                            entries=entries,
                            precondition=candidate_pairs[i].precondition,
                        )

        # record every declined, probe-confirmed pair once the claim
        # its candidate landed as is known — minted, suppressed into a twin, or
        # resolved — and only while both ends are still ACTIVE.
        declined_total = sum(len(others) for others in divergent)
        divergences += await _record_divergences(session, gate, divergent, index_to_id)

        # now that every candidate's stored id is known,
        # write the stance edges and the entry-level NARRATIVE graph. The plan is
        # a pure function of the candidates and ``index_to_id``
        # (``ingest.structure_edges``); this is its apply step.
        for a_id, b_id, kind in plan_structure_edges(result.candidates, index_to_id, stance_specs):
            await create_relation(session, a_id, b_id, kind, RelationCreatedBy.EXTRACTOR_DIRECT)

        # capture every URL mentioned in this snapshot as a citation
        # signal (best-effort; never fails extraction). Runs once extraction has
        # succeeded, in the same transaction as the COMPLETE status below.
        await _capture_url_mentions(session, entry_id, content)

        # A carried-forward particle is a claim this snapshot states, so its
        # re-observation is recorded exactly as a suppressed-into one's is: one
        # appended ref naming this snapshot, nothing re-pointed.
        # That is what lets "a source currently states a claim" be read off the
        # claim's own provenance.
        await _record_carry_forward(session, result.carry_forward_ids, entry_id, snapshot_id)

        # a MUTABLE source's new snapshot retires the generation it
        # replaced. Runs HERE — after extraction, in the same transaction as the
        # COMPLETE status below — and not at deposit time, so
        # carry-forward has already kept the particles whose chunks are
        # unchanged. Those are excluded explicitly: a carried-forward particle's
        # provenance edge still names the snapshot it was first extracted from
        # (the edge is never re-pointed), which would otherwise read as a stale
        # generation. A no-op for every non-MUTABLE entry, and zero LLM calls.
        # particles are excluded for the same reason as carry-forward:
        # a suppressed-into particle is still current (this snapshot re-observed
        # its claim), but its provenance edge still names the snapshot it was
        # first extracted from, so the cascade would otherwise read it as a
        # stale generation and demote the very particle the suppression kept.
        # The cascade also leaves alone anything another entry's current
        # snapshot still attests; these exclusions are the belt.
        await cascade_superseded_generation(
            session,
            entry_id=entry_id,
            current_snapshot_id=snapshot_id,
            exclude_ids=(
                frozenset(result.carry_forward_ids) | supersede_ids | frozenset(suppressed_into)
            ),
        )

        # suppression is never silent. The count reaches the
        # operator through the extraction quality notes and, via
        # ``suppressed_ids_out``, the `particles extract` summary line.
        if suppressed_into:
            note = suppression_note(len(suppressed_into))
            result.quality_notes.append(note)
            log.info("Snapshot %s: %s", snapshot_id, note)
        # a hold is never silent either.
        if retired_held or retired_absorbed:
            note = retired_value_note(len(retired_held), len(retired_absorbed))
            result.quality_notes.append(note)
            log.info("Snapshot %s: %s", snapshot_id, note)
        if suppressed_ids_out is not None:
            suppressed_ids_out.extend(suppressed_into)
        # a left-standing divergence is never silent either.
        if divergences:
            note = divergence_note(divergences)
            result.quality_notes.append(note)
            log.info("Snapshot %s: %s", snapshot_id, note)
        if divergences_out is not None:
            divergences_out.declined += declined_total
            divergences_out.recorded += divergences

        await update_extraction_status(session, snapshot_id, ExtractionStatus.COMPLETE)
        log.info(
            "Extracted %d particles from snapshot %s (entry %s; %d duplicate(s) suppressed)",
            len(written),
            snapshot_id,
            entry_id,
            len(suppressed_into),
        )
        await session.commit()
    return written


async def _record_divergences(
    session: AsyncSession,
    gate: ObserverGate,
    divergent: Sequence[Sequence[Particle]],
    index_to_id: dict[int, str],
) -> int:
    """Write the ``CONTRADICTS`` edges for this pass's declined pairs."""
    pairs = [
        (index_to_id[i], others)
        for i, others in enumerate(divergent)
        if others and i in index_to_id
    ]
    if not pairs:
        return 0
    ids = {landed for landed, _ in pairs} | {o.id for _, others in pairs for o in others}
    status = {pid: p.status for pid, p in (await get_particles_by_ids(session, list(ids))).items()}
    written = 0
    for landed, others in pairs:
        if status.get(landed) is not Status.ACTIVE:
            continue
        standing = [o for o in others if o.id != landed and status.get(o.id) is Status.ACTIVE]
        written += await gate.record(session, landed, standing)
    return written


async def _record_carry_forward(
    session: AsyncSession, particle_ids: Sequence[str], entry_id: str, snapshot_id: str
) -> int:
    """Append a ref naming ``snapshot_id`` to each carried-forward particle.

    The appended ref copies the particle's latest ref to ``entry_id`` — its
    location and chunk hash are the chunk's, which is why it was carried — and
    names the re-observing snapshot. ``append_provenance_ref`` keeps it
    additive and idempotent: the earliest ref stays the decay anchor, the edge
    index gains no row, and ``confidence.value`` / ``asserted_at`` are untouched.
    Returns how many refs were added.
    """
    ids = list(dict.fromkeys(particle_ids))
    particles_by_id = await get_particles_by_ids(session, ids)
    added = 0
    for particle_id in ids:
        particle = particles_by_id.get(particle_id)
        if particle is None:
            continue
        ref = reobservation_ref(particle, entry_id, snapshot_id)
        if await append_provenance_ref(session, particle_id, ref):
            added += 1
    return added


async def _capture_url_mentions(session: AsyncSession, entry_id: str, content: bytes) -> None:
    """Harvest + persist every URL mentioned in a snapshot's content.

    The citation-signal capture path: all URLs mentioned anywhere in the source
    — comments and prose included, not just a post's primary link —
    are canonicalized and recorded against this source entry, idempotently. URLs
    already deposited are born bound to their entry, so only undeposited ones
    surface as suggestions. Gated by ``citation_signal.capture_enabled``.

    Best-effort and isolated: any failure (decode, parse, store) is logged and
    swallowed so a citation-signal hiccup never fails extraction itself.
    """
    if not get_config().citation_signal.capture_enabled:
        return
    try:
        urls = harvest_urls(content.decode("utf-8", errors="ignore"))
        if not urls:
            return
        deposited_map = await build_deposited_url_map(session)
        await record_url_mentions(
            session,
            source_entry_id=entry_id,
            canonical_urls=urls,
            deposited_map=deposited_map,
        )
    except Exception as exc:
        log.warning("URL-mention capture failed for entry %s: %s", entry_id, exc)


async def load_active_conflict_candidates(session: AsyncSession) -> list[EmbeddingPair]:
    """Load the §6.6 conflict-candidate set once for a batch of reconcile calls.

    The candidate set is every ACTIVE particle with an embedding, minus the
    DOCUMENT_META particles §6.6 ignores — exactly what
    :func:`reconcile_and_insert` would load per call. Pass the returned list as
    that function's ``candidate_cache`` for a whole batch (e.g. an interchange
    import): the function maintains it in place as particles are written, so a
    batch of N imports runs **one** full-store scan instead of N (F4.3).
    """
    return [
        pair
        for pair in await get_active_particles_with_embeddings(session, min_confidence=0.0)
        if not _skips_conflict_resolution(pair[0].properties)
    ]


async def reconcile_and_insert(
    session: AsyncSession,
    particle: Particle,
    embedding: list[float] | None = None,
    *,
    candidate_cache: list[EmbeddingPair] | None = None,
    single_trust_order: bool | None = None,
    fail_closed: bool = False,
) -> Particle | None:
    """Insert one particle into the current store, running the §6.6 ladder.

    Public seam for **non-snapshot** inserts — the interchange importer
    uses it so an imported claim reconciles against the target store
    exactly as a freshly extracted one would (import is a
    single-store write that runs §6.6). Re-embeds when no embedding is supplied
    (the embedding is derived, never carried in interchange). The INCONSISTENCY
    particle, when the ladder falls through, takes its provenance from the
    incoming particle's first ref.

    This is the **cross-entry write-time reconciliation** path (§1/§4):
    unlike :func:`extract_snapshot`, which scopes to the same corpus entry, the
    candidate set here is every ACTIVE particle in the store (similarity-gated
    in :func:`_find_conflict`), so contributions / imports reconcile against
    claims from *other* sources. Subject-scoped candidate retrieval for scale is
    deferred. The consensus-safe trust regime is resolved per store by
    the caller and passed as ``single_trust_order``;
    ``None`` falls back to the process-global ``reconciliation.store_mode``.
    ``fail_closed`` makes an unverifiable high-similarity pair
    quarantine instead of corroborate — the assertion pathway passes ``True``.

    ``candidate_cache`` lets a batch caller (e.g. interchange import) avoid the
    per-call full-store embedding scan that is O(N²) over a batch (F4.3): pass
    the list returned by :func:`load_active_conflict_candidates` and it is read
    *and maintained in place* — newly written ACTIVE particles are appended and
    trust-superseded ones removed — so later items in the same batch still
    reconcile against earlier ones. When ``None`` (the default, single-call
    path) the candidate set is loaded fresh from the store.

    Returns the inserted / winning particle, or ``None`` when trust resolution
    kept the existing particle and dropped this one. When suppression
    fires — this exact claim is already ACTIVE with the same subjects and stance
    holder — the **existing** particle is returned with this call's provenance
    ref appended to it, which is what makes a repeated assertion idempotent
    rather than a second copy.
    """
    if embedding is not None:
        emb: np.ndarray[Any, np.dtype[np.float32]] | None = np.asarray(embedding, dtype=np.float32)
        emb_list: list[float] | None = embedding
    else:
        batch = await _embed_batch_async([particle.content])
        emb = batch[0] if batch else None
        emb_list = emb.tolist() if emb is not None else None
        if emb is None:
            # same two failures as the extract path, on the write-time
            # reconciliation route (agent asserts, interchange import). There is
            # no ExtractResult to hang a quality note on here, so the log line is
            # the whole disclosure — make it name both halves.
            log.warning(
                "Reconciling %r without an embedding model: §6.6 conflict resolution "
                "skipped and the particle stored without an embedding, so it will not "
                "appear in semantic query, dedup or lint until re-extracted.",
                particle.content[:80],
            )

    # The rungs above §6.6 route through the one pure router the extract loop
    # uses (D2), so "duplicate" means one thing on both write paths.
    # The conflict search is the expensive gather here, so it runs
    # only when the cheap rungs miss: the router is called once without a
    # conflict, then again with the search's result.
    duplicates = (
        await build_duplicate_index(session, [particle.content])
        if get_config().extraction.duplicate_suppression.enabled
        else DuplicateIndex()
    )
    retired_values = (
        await build_retired_value_index(session, [particle.content])
        if get_config().extraction.retired_value_quarantine.enabled
        else DuplicateIndex()
    )
    route, target = route_particle(
        particle,
        duplicates=duplicates,
        retired_values=retired_values,
        conflict=None,
        retired_ids=frozenset(),
    )

    if route is Route.UNCHECKED:
        # DOCUMENT_META and non-asserted never participate
        # in §6.6 (mirrors extract_snapshot). They are also excluded from the
        # conflict-candidate set, so neither is ever appended to ``candidate_cache``.
        validate_transition(None, Status.ACTIVE)
        await insert_particle(session, particle, emb_list)
        return particle

    if route is Route.SUPPRESS:
        # without this rung an exact duplicate reaches the ladder,
        # finds no contradiction signal, returns CORROBORATES, and is written as
        # a second ACTIVE particle — which is why ``particle_assert`` was not
        # idempotent before this ADR.
        assert target is not None
        for ref in particle.provenance:
            await append_provenance_ref(session, target.id, ref)
        log.info(
            "Duplicate suppressed: assertion already ACTIVE as %s",
            target.id[:8],
        )
        return target

    if route is Route.RETIRED_TWIN:
        # a re-asserted retired claim is held on both write paths.
        assert target is not None
        corpus_entry_id, snapshot_id, trigger_ref_type = _trigger_ref_for(particle)
        entry = (
            await get_entry(session, corpus_entry_id)
            if trigger_ref_type is ProvenanceRefType.SOURCE
            else None
        )
        return await _hold_retired_reassertion(
            session,
            particle,
            emb_list,
            target,
            corpus_entry_id=corpus_entry_id,
            snapshot_id=snapshot_id,
            trigger_ref_type=trigger_ref_type,
            domain=infer_domain(entry.source_type) if entry else None,
        )

    if candidate_cache is None:
        pairs = await load_active_conflict_candidates(session)
    else:
        pairs = candidate_cache
    existing = [p for p, _ in pairs]
    existing_embs = [e for _, e in pairs]

    conflict = _find_conflict(emb, existing, existing_embs)
    if conflict is None and emb is not None:
        # the subject-keyed search reaches this path too — an
        # agent's "the user lives in Denver" scores ~0.70 against its own
        # earlier "…Boston" and never cleared the 0.80 gate either.
        update_cfg = get_config().reconciliation.update_supersession
        if update_cfg.enabled and particle_subject_ids(particle):
            index = SubjectIndex.build(pairs)
            nearest = index.candidates(
                particle_subject_ids(particle),
                emb,
                floor=update_cfg.subject_floor,
                limit=1,
            )
            conflict = nearest[0] if nearest else None
    # This path's candidacy policy, stated as arguments (``ingest.pair_selection``):
    # one pair, always primary. No subject pool, no precondition pre-screen (the
    # ladder applies it) and no lineage filter. The probe keeps its tri-state for
    # the ladder, where ``fail_closed`` reads an incomplete probe as a signal.
    candidacy = CandidatePairs(nearest=conflict)
    probe_results: dict[str, bool | None] = {}
    for other, _role in plan_probes(candidacy):
        probe_results[other.id] = await _has_contradiction_signal(particle.content, other.content)
    selection = select_pairs(candidacy, {pid: bool(p) for pid, p in probe_results.items()})
    route, target = route_particle(
        particle,
        duplicates=duplicates,
        retired_values=retired_values,
        conflict=selection.primary,
        retired_ids=frozenset(),
    )
    if route is Route.INSERT:
        validate_transition(None, Status.ACTIVE)
        await insert_particle(session, particle, emb_list)
        if candidate_cache is not None and emb is not None:
            candidate_cache.append((particle, emb))
        return particle

    assert route is Route.LADDER and target is not None
    corpus_entry_id, snapshot_id, trigger_ref_type = _trigger_ref_for(particle)
    return await _resolve_conflict(
        session,
        particle,
        target,
        corpus_entry_id,
        snapshot_id,
        probe=probe_results[target.id],
        candidate_cache=candidate_cache,
        new_embedding=emb,
        single_trust_order=single_trust_order,
        fail_closed=fail_closed,
        trigger_ref_type=trigger_ref_type,
        allow_own_assertion_supersession=True,
        observer=await ObserverGate.open(session),
    )


def _find_conflict(
    candidate_emb: np.ndarray[Any, np.dtype[np.float32]] | None,
    existing: Sequence[Particle],
    existing_embs: Sequence[np.ndarray[Any, np.dtype[np.float32]] | None],
    candidate_stance_holder: str | None = None,
) -> Particle | None:
    """Return the most similar existing ACTIVE particle if similarity > threshold, else None.

    Pure / in-memory: takes the candidate's embedding (not the candidate
    particle) so it can be called during the pre-write precompute phase in
    :func:`extract_snapshot` before a particle object exists.

    ``candidate_stance_holder`` is the candidate's ``stance:holder`` (None when
    the candidate is not a stance). Stance §6.6 semantics: a stance
    never contradicts its target, and opposing stances by *different* holders
    never contradict — so a stance pairs for contradiction **only** with another
    stance of the *same* holder (the same-holder reversal, the one case §6.6
    catches when both are co-located in this entry). Different-holder and
    stance-vs-claim pairs are skipped before the signal probe.
    """
    if candidate_emb is None or not existing:
        return None

    best_sim = 0.0
    best_particle: Particle | None = None
    candidate_is_stance = candidate_stance_holder is not None

    for p, emb in zip(existing, existing_embs, strict=False):
        if emb is None:
            continue
        # non-truth-apt existing particles (opinions / feelings /
        # constitutive rules) are never §6.6 comparison pairs — there is no
        # shared truth to contradict, so don't even pay the LLM signal probe.
        if not is_truth_apt(p):
            continue
        # stance-aware candidacy. A stance only conflicts with a
        # same-holder stance; a stance never contradicts its target, and
        # different-holder stances never contradict each other.
        p_holder = stance_holder(p)
        p_is_stance = p_holder is not None
        if candidate_is_stance != p_is_stance:
            continue
        if candidate_is_stance and candidate_stance_holder != p_holder:
            continue
        # normalized cosine clamped to [0, 1] — the threshold below
        # (extraction.similarity_threshold) lives on this scale.
        sim = cosine_similarity(candidate_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_particle = p

    if best_sim >= get_config().extraction.similarity_threshold:
        return best_particle
    return None


# Attribution / quoting surface patterns. A claim that introduces another
# claim ("X quotes the claim that …", "according to X, …", "X says that …")
# sits one level above the underlying claim semantically but embeds near it
# textually — high cosine similarity with no actual disagreement. These
# patterns are checked case-insensitively against both sides of the pair.
_ATTRIBUTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(quotes?|cites?|references?|repeats?|paraphrases?)\b.*\bclaim\b", re.IGNORECASE),
    re.compile(r"\baccording to\b", re.IGNORECASE),
    re.compile(
        r"\b(says|argues|claims|states|writes|notes|observes|asserts|reports)\s+that\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bas (noted|stated|written|argued|observed) by\b", re.IGNORECASE),
)


def _is_attribution_paraphrase(content_a: str, content_b: str) -> bool:
    """Return True if either side reads as an attribution wrapping a quoted claim.

    Catches the canonical false-positive: "@X quotes the claim that …",
    "according to X, …", "X says that …". Such sentences embed the original
    claim near-verbatim, so cosine similarity is high, but they corroborate
    rather than contradict.
    """
    return any(pat.search(content_a) or pat.search(content_b) for pat in _ATTRIBUTION_PATTERNS)


def _contradiction_prompt(content_a: str, content_b: str) -> str:
    """The L-SEM-01-style contradiction-probe prompt (one source of truth).

    Shared with the reconcile sweep
    (:mod:`particles.operations.reconcile`), whose breaker-routed probe uses
    the same prompt through the ``operations._llm`` seam.
    """
    return (
        "Do these two claims contradict each other? Answer with exactly one of:\n"
        "- 'YES: <brief description>' if they directly disagree\n"
        "- 'NO' if they agree, are about different things, are paraphrases,"
        " or if one merely attributes / quotes / restates the other\n\n"
        f"Claim A: {content_a}\n\nClaim B: {content_b}"
    )


async def _llm_confirms_contradiction(content_a: str, content_b: str) -> bool | None:
    """Return True/False from the LLM, or None when the probe cannot complete.

    Mirrors the L-SEM-01 prompt in :mod:`particles.operations.lint`. A bool is a
    positive / negative judgement; ``None`` means the probe could not run (no
    client, API error, unparseable response). Callers choose how to treat
    ``None``: extraction maps it to "no contradiction" (fail-open, unchanged);
    the assertion pathway fails closed (quarantine + INCONSISTENCY).
    """
    prompt = _contradiction_prompt(content_a, content_b)
    try:
        from particles.llm import complete

        # The reconcile-ladder contradiction check is part of the semantic-lint
        # purpose: it mirrors the L-SEM-01 lint prompt.
        response = await complete("semantic_lint", prompt, max_tokens=120)
        return response.upper().startswith("YES")
    except Exception as exc:
        log.warning(
            "Contradiction confirmation LLM probe could not complete; "
            "caller decides fail-open vs fail-closed: %s",
            exc,
        )
        return None


async def _has_contradiction_signal(content_a: str, content_b: str) -> bool | None:
    """Return True/False, or None when the LLM probe cannot complete.

    High embedding similarity is *not* a contradiction signal on its own —
    paraphrases and attribution/quoting wrappers also embed near each other.
    The §6.6 ladder must not fire without positive confirmation.

    Order of checks (cheapest first):
      1. Attribution / quoting surface patterns → no contradiction (``False``).
      2. LLM contradiction check (L-SEM-01-style prompt) → ``True`` only on YES,
         ``False`` on NO, ``None`` if the probe could not run.
    """
    if _is_attribution_paraphrase(content_a, content_b):
        return False
    return await _llm_confirms_contradiction(content_a, content_b)


def _trigger_ref_for(particle: Particle) -> tuple[str, str, ProvenanceRefType]:
    """``(corpus_entry_id, snapshot_id, trigger_ref_type)`` for a wrapper's trigger ref.

    The INCONSISTENCY wrapper's trigger ref points at the corpus entry that
    produced the conflicting candidate — the first SOURCE-typed ref. A derived
    particle has only PARTICLE-typed premise refs; its trigger is
    the particle itself, and the wrapper's trigger ref stays PARTICLE-typed so
    a particle id is never mislabelled as a corpus entry.
    """
    prov = next((r for r in particle.provenance if r.type is ProvenanceRefType.SOURCE), None)
    if prov is not None:
        return prov.corpus_entry_id, prov.snapshot_id or "", ProvenanceRefType.SOURCE
    return particle.id, "", ProvenanceRefType.PARTICLE


async def _hold_retired_reassertion(
    session: AsyncSession,
    particle: Particle,
    emb_list: list[float] | None,
    twin: Particle,
    *,
    corpus_entry_id: str,
    snapshot_id: str,
    domain: str | None,
    trigger_ref_type: ProvenanceRefType = ProvenanceRefType.SOURCE,
) -> Particle:
    """Hold a candidate that re-asserts a judgment-retired claim.

    Two cases, by what ``twin`` is:

    * an **open hold** (``PROVENANCE_STALE`` / ``CONFLICT_PENDING`` — an
      ordinary §6.6 loser, or an earlier hold under this rung): the new observation is
      recorded on it (a provenance append, as for a suppression) and the hold is
      returned. No second record — a repeated reindex of an unchanged source is
      idempotent, and one review lifts every observation at once.
    * a **judgment-retired twin**: the candidate is stored in full, born
      quarantined, and an INCONSISTENCY record pairing the retired
      twin (A) with the held candidate (B) is written for Review, with the
      retired-value marker so PREFER_A reads as "the retirement stands" and
      PREFER_B as "lift it". The wrapper is returned.

    Nothing here touches the twin's status, its ``retired_at``, or the stored
    ``confidence.value`` of anything.
    """
    if is_pending_hold(twin):
        for ref in particle.provenance:
            await append_provenance_ref(session, twin.id, ref)
        log.info(
            "Retired-value hold: candidate %r already held for review as %s; observation recorded",
            particle.content[:60],
            twin.id[:8],
        )
        return twin

    plan = plan_retired_hold(
        twin,
        particle,
        corpus_entry_id=corpus_entry_id,
        snapshot_id=snapshot_id,
        trigger_ref_type=trigger_ref_type,
        domain=domain,
    )
    held = await _apply_conflict_plan(session, plan, existing=twin, embedding=emb_list)
    assert held is not None
    return held


async def _apply_conflict_plan(
    session: AsyncSession,
    plan: ConflictWritePlan,
    *,
    existing: Particle,
    embedding: Any,
    candidate_cache: list[EmbeddingPair] | None = None,
    retired_out: set[str] | None = None,
    observer: ObserverGate | None = None,
) -> Particle | None:
    """Perform a :class:`~particles.ingest.conflict_plan.ConflictWritePlan`, in its order.

    Insert, then demote (the mirror verdicts demote the row just inserted),
    then the INCONSISTENCY record, the dropped-candidate event and the observer
    divergence. ``embedding`` is stored with the inserted row, and joins a batch
    caller's ``candidate_cache`` beside it (F4.3), so a later item in the batch
    reconciles against this verdict.
    """
    if plan.insert is not None:
        validate_transition(None, plan.insert.status)
        await insert_particle(session, plan.insert, _storable(embedding))
    if plan.demote is not None:
        demote_id, reason = plan.demote
        await update_particle_status(session, demote_id, Status.PROVENANCE_STALE, reason)
    if plan.wrapper is not None:
        validate_transition(None, Status.INCONSISTENCY)
        await insert_particle(session, plan.wrapper, domain_hint=plan.wrapper_domain_hint)
    if plan.dropped_event is not None:
        event = plan.dropped_event
        await record_event(
            session,
            actor=event.actor,
            event_type=event.event_type,
            reason=event.reason,
            refs=list(event.refs),
            payload=dict(event.payload),
        )
    if plan.record_divergence and observer is not None and plan.insert is not None:
        await observer.record(session, plan.insert.id, [existing])
    if plan.cache_drop_existing:
        if candidate_cache is not None:
            candidate_cache[:] = [pair for pair in candidate_cache if pair[0].id != existing.id]
        if retired_out is not None:
            retired_out.add(existing.id)
    if (
        plan.cache_add_new
        and plan.insert is not None
        and candidate_cache is not None
        and embedding is not None
    ):
        candidate_cache.append((plan.insert, embedding))
    if plan.log_message is not None:
        log.info(plan.log_message, *plan.log_args)
    return plan.result


async def _resolve_conflict(
    session: AsyncSession,
    new_particle: Particle,
    existing: Particle,
    corpus_entry_id: str,
    snapshot_id: str,
    *,
    probe: bool | None,
    candidate_cache: list[EmbeddingPair] | None = None,
    new_embedding: np.ndarray[Any, np.dtype[np.float32]] | None = None,
    single_trust_order: bool | None = None,
    fail_closed: bool = False,
    trigger_ref_type: ProvenanceRefType = ProvenanceRefType.SOURCE,
    allow_update_supersession: bool = False,
    allow_own_assertion_supersession: bool = False,
    retired_out: set[str] | None = None,
    observer: ObserverGate | None = None,
    candidate_scope: BeliefScope | None = None,
) -> Particle | None:
    """Apply the §6.6 conflict-resolution ladder for one (existing, new) pair.

    Gather, decide, apply (D2):

      - **Gather** (here): the observer precondition, and, when
        :func:`~particles.core.conflict_resolution.needs_rung_inputs`, the
        rung inputs (the supersession prior, the Extension B trust scores with
        the URL baseline, the rung 2.5 order).
      - **Decide** (pure):
        :func:`~particles.core.conflict_resolution.decide_ladder` applies the
        overrides and the §6.4 verdict, and
        :func:`~particles.ingest.conflict_plan.plan_conflict_writes` maps the
        outcome to writes.
      - **Apply**: :func:`_apply_conflict_plan` performs them in plan order.

    ``probe`` is the contradiction probe (attribution pattern and LLM gate)
    the caller ran when it selected the pair (``ingest.pair_selection``),
    outside the write transaction (F4.3). It is tri-state: ``None`` means the
    probe could not complete, which only the ``fail_closed`` assertion pathway
    distinguishes from ``False``; extraction passes a bool.

    ``new_embedding`` is the candidate's embedding —
    stored alongside whichever row this verdict persists, and used with
    ``candidate_cache`` (when given) to mirror the write in place: the new
    particle is appended when it lands ACTIVE and a trust-superseded existing
    one removed, so a batch caller's later items reconcile against this
    verdict.

    ``allow_update_supersession`` lets rung 2.5 fire; only the
    extraction path passes it (never the assertion pathway), and it
    is further gated on ``reconciliation.update_supersession.enabled``.
    ``allow_own_assertion_supersession`` is the assertion pathway's own,
    narrower door: an agent's newer assertion supersedes its own
    earlier one, and nothing else.
    ``retired_out`` (when given) collects the id of an existing particle this
    verdict demoted, so a caller reconciling several claims in one pass never
    offers — or demotes — it twice.

    ``observer`` is the precondition, checked here before any rung,
    so it covers every route that reaches the ladder.
    A pair another project observes is **declined**: the candidate is written
    ``ACTIVE`` beside the existing claim, and a confirmed contradiction is
    recorded as ``CONTRADICTS``. A global existing claim contested by a
    project's candidate skips rung 2.5 and falls through to ``INCONSISTENT``.
    ``candidate_scope`` is the candidate's scope when the caller already has it
    (extraction: the entry's); otherwise it is read from the candidate's refs.

    Returns:
      - The winning Particle (inserted as ACTIVE) if trust resolution prefers
        the new candidate.
      - ``None`` if trust resolution prefers the existing particle (the
        new candidate is dropped without insertion).
      - The new candidate Particle (inserted as ACTIVE) if the
        contradiction-signal gate exonerated the pair.
      - The INCONSISTENCY particle (inserted) if the conflict fell through
        to rung 3. The losing candidate is additionally persisted as a real
        particle born quarantined — ``PROVENANCE_STALE`` with
        ``status_reason = CONFLICT_PENDING`` — so Review can
        recover it and the wrapper's B ref resolves.
    """
    # the observer precondition, a property of the pair (it does
    # not depend on the probe), checked once here before any rung.
    precondition = PairPrecondition.RECONCILE
    if observer is not None and observer.engaged:
        scope = (
            candidate_scope
            if candidate_scope is not None
            else await observer.candidate_scope(session, new_particle)
        )
        precondition = await observer.verdict(session, scope, existing)

    # The effective trust regime, resolved before
    # the rung inputs: the attribution requirement keys on it —
    # in a ``multi`` store an anonymous lineage is not one principal.
    effective_single = effective_single_trust_order(
        single_trust_order, get_config().reconciliation.store_mode
    )

    # Gather the rung inputs only when the ladder can consult them. The domain
    # read here is also the INCONSISTENCY record's ``domain_hint`` (the
    # Extension B cascade).
    domain: str | None = None
    rung_inputs: RungInputs | None = None
    if needs_rung_inputs(probe, fail_closed=fail_closed, precondition=precondition):
        rung_inputs, domain = await _gather_rung_inputs(
            session,
            new_particle,
            existing,
            corpus_entry_id,
            snapshot_id,
            order_source=update_order_source(
                allow_update=allow_update_supersession
                and get_config().reconciliation.update_supersession.enabled,
                allow_own_assertion=allow_own_assertion_supersession,
                forced=forces_inconsistent(
                    probe, fail_closed=fail_closed, precondition=precondition
                ),
            ),
            effective_single=effective_single,
        )

    outcome = decide_ladder(
        existing,
        new_particle,
        probe=probe,
        fail_closed=fail_closed,
        precondition=precondition,
        rung_inputs=rung_inputs,
        single_trust_order=effective_single,
        trust_differential_threshold=get_config().trust.differential_threshold,
    )
    plan = plan_conflict_writes(
        outcome,
        new_particle,
        existing,
        corpus_entry_id=corpus_entry_id,
        snapshot_id=snapshot_id,
        trigger_ref_type=trigger_ref_type,
        scores=(
            (rung_inputs.trust_score_new, rung_inputs.trust_score_existing)
            if rung_inputs is not None
            else (None, None)
        ),
        domain=domain,
    )
    return await _apply_conflict_plan(
        session,
        plan,
        existing=existing,
        embedding=new_embedding,
        candidate_cache=candidate_cache,
        retired_out=retired_out,
        observer=observer,
    )


async def _gather_rung_inputs(
    session: AsyncSession,
    new_particle: Particle,
    existing: Particle,
    corpus_entry_id: str,
    snapshot_id: str,
    *,
    order_source: UpdateOrderSource | None,
    effective_single: bool,
) -> tuple[RungInputs, str | None]:
    """Read the §6.4 rung inputs for a signalled pair, and the candidate's domain.

    A staged gather: the supersession prior only when enabled, the trust
    scores from the Extension B layered lookup (the URL baseline as
    fallback), and the rung 2.5 order from whichever source applies.
    """
    existing_source_ref = next(
        (ref for ref in existing.provenance if ref.type == ProvenanceRefType.SOURCE),
        None,
    )
    existing_entry_id = existing_source_ref.corpus_entry_id if existing_source_ref else None
    existing_snapshot_id = existing_source_ref.snapshot_id if existing_source_ref else None

    # cap. 2: document-supersession prior. Resolve whether either
    # side's provenance corpus entry (transitively) supersedes the other's,
    # so rung 1.5 can prefer the superseding document's claim. Gated by
    # config; needs the existing particle's source entry to compare against.
    new_supersedes_existing = False
    existing_supersedes_new = False
    if get_config().document_supersession.enabled and existing_entry_id:
        from particles.corpus.supersession import entry_supersedes

        new_supersedes_existing = await entry_supersedes(
            session,
            superseding_entry_id=corpus_entry_id,
            superseded_entry_id=existing_entry_id,
        )
        existing_supersedes_new = await entry_supersedes(
            session,
            superseding_entry_id=existing_entry_id,
            superseded_entry_id=corpus_entry_id,
        )
    new_entry = await get_entry(session, corpus_entry_id)
    existing_entry = await get_entry(session, existing_entry_id) if existing_entry_id else None
    domain = infer_domain(new_entry.source_type) if new_entry else None

    # §6.4 AUTHOR tier inputs — each side's author_id from its SOURCE
    # snapshot, read off the entries already fetched above.
    new_author_id = _snapshot_author_id(new_entry, snapshot_id)
    existing_author_id = _snapshot_author_id(existing_entry, existing_snapshot_id)

    # Trust scores are only consulted if neither particle is ALEATORY,
    # but resolving them unconditionally keeps the call shape simple and
    # the result is just dropped on the ALEATORY path inside
    # resolve_conflict().
    score_new, score_existing = await _resolve_trust_scores(
        session,
        domain,
        new_entry_id=corpus_entry_id,
        new_source_type=new_entry.source_type if new_entry else "",
        new_uri=new_entry.uri_r if new_entry else None,
        new_author_id=new_author_id,
        existing_entry_id=existing_entry_id,
        existing_source_type=existing_entry.source_type if existing_entry else "",
        existing_uri=existing_entry.uri_r if existing_entry else None,
        existing_author_id=existing_author_id,
    )

    # rung 2.5 input; ``None`` unless the pair qualifies (see
    # :func:`particles.ingest.update_supersession.update_order`).
    order: int | None = None
    if order_source is UpdateOrderSource.UPDATE:
        # A claim re-observed since its first extraction carries the date
        # of that latest observation (folds the
        # repeat into this same particle, in a later entry).
        existing_latest = await latest_source_date(session, existing)
        order = update_order(
            new_particle,
            new_entry,
            snapshot_id,
            existing,
            existing_entry,
            existing_snapshot_id,
            require_attribution=not effective_single,
            existing_date=existing_latest,
        )
    elif order_source is UpdateOrderSource.OWN_ASSERTION:
        # the assertion pathway's one carve-out — an agent
        # revising its own earlier assertion.
        order = own_assertion_order(new_particle, existing)

    return (
        RungInputs(
            new_supersedes_existing=new_supersedes_existing,
            existing_supersedes_new=existing_supersedes_new,
            trust_score_new=score_new,
            trust_score_existing=score_existing,
            update_order=order,
        ),
        domain,
    )


def _storable(embedding: Any) -> list[float] | None:
    """An embedding in storable form; a mocked model may yield plain lists."""
    if embedding is None:
        return None
    if hasattr(embedding, "tolist"):
        return list(embedding.tolist())
    return list(embedding)


async def _gather_candidate_pairs(
    session: AsyncSession,
    candidates: Sequence[CandidateParticle],
    embeddings: Sequence[Any],
    existing: Sequence[Particle],
    existing_embs: Sequence[np.ndarray[Any, np.dtype[np.float32]] | None],
    *,
    subject_index: SubjectIndex | None,
    gate: ObserverGate,
    candidate_scope: BeliefScope | None,
    entry: CorpusEntry,
    snapshot_id: str,
    updates_only: bool,
    entries: dict[str, CorpusEntry | None],
) -> list[CandidatePairs]:
    """Gather 1 of the §6.6 pair selection: every existing claim each candidate could pair with.

    Three stages, each a batch over the whole pass:

      1. per candidate, the same-entry nearest claim (pure over the loaded
         embeddings) and, when the candidate is eligible for the
         subject-keyed search, its read-only subject ids and the pool (the
         nearest claim excluded);
      2. one ``gate.prime`` over every pairable claim, then the
         precondition of each (a no-op until the store is rescoped);
      3. in a ``multi`` store only, whether rung 2.5 could act on each pool
         member the precondition does not decline.

    ``entries`` collects the corpus entries stage 3 reads.
    """
    update_cfg = get_config().reconciliation.update_supersession
    about_ids: dict[str, str | None] = {}
    found: list[tuple[Particle | None, list[Particle]]] = []
    for i, candidate in enumerate(candidates):
        if _skips_conflict_resolution(candidate.properties):
            found.append((None, []))
            continue
        cand_emb = embeddings[i] if i < len(embeddings) else None
        holder = holder_from_properties(candidate.properties)
        nearest = _find_conflict(
            cand_emb,
            existing,
            existing_embs,
            candidate_stance_holder=holder,
        )
        pool: list[Particle] = []
        if (
            subject_index is not None
            and cand_emb is not None
            and holder is None
            and candidate.assertion_modality == AssertionModality.FALSIFIABLE
        ):
            subject_ids = await _candidate_subject_ids_readonly(
                session, candidate, about_ids, source_type=entry.source_type
            )
            if subject_ids:
                pool = subject_index.candidates(
                    subject_ids,
                    cand_emb,
                    floor=update_cfg.subject_floor,
                    limit=update_cfg.max_candidates,
                    skip_ids=() if nearest is None else (nearest.id,),
                )
        found.append((nearest, pool))

    precondition: dict[str, PairPrecondition] = {}
    if candidate_scope is not None:
        pairable = [p for nearest, pool in found for p in ([nearest] if nearest else []) + pool]
        await gate.prime(session, pairable)
        for other in pairable:
            precondition[other.id] = await gate.verdict(session, candidate_scope, other)

    out: list[CandidatePairs] = []
    for candidate, (nearest, pool) in zip(candidates, found, strict=True):
        can_update: dict[str, bool] | None = None
        if updates_only:
            can_update = {
                other.id: await _pair_can_update(
                    session, entry, snapshot_id, candidate, other, entries
                )
                for other in pool
                if precondition.get(other.id) is not PairPrecondition.DECLINE
            }
        out.append(
            CandidatePairs(
                nearest=nearest,
                subject_pool=tuple(pool),
                precondition=precondition,
                can_update=can_update,
            )
        )
    return out


async def _probe_pairs(
    content: str, planned: Sequence[tuple[Particle, PairRole]]
) -> dict[str, bool]:
    """Gather 2 of the §6.6 pair selection: exactly the probes ``plan_probes`` named.

    Extraction is fail-open: a probe that could not complete (``None``) reads
    as no contradiction. The assertion pathway fails closed instead,
    and keeps the tri-state (``reconcile_and_insert``).
    """
    results: dict[str, bool] = {}
    for other, _role in planned:
        results[other.id] = bool(await _has_contradiction_signal(content, other.content))
    return results


async def _pair_can_update(
    session: AsyncSession,
    entry: CorpusEntry,
    snapshot_id: str,
    candidate: CandidateParticle,
    other: Particle,
    entries: dict[str, CorpusEntry | None],
) -> bool:
    """Whether rung 2.5 could act on this cross-entry pair.

    Asked only where the trust rung is off (a ``multi`` store), so a pair the
    update rung cannot settle has no outcome but a review item. The candidate
    is not a ``Particle`` yet, and only lineage and dates matter here, so the
    check runs against a stand-in carrying the candidate's provenance.
    ``entries`` caches the other side's entry, which the rung 2.5 extras reuse.
    """
    stand_in = Particle(
        content=candidate.content,
        confidence=Confidence(value=candidate.confidence_value),
        uncertainty_nature=candidate.uncertainty_nature,
        asserted_by="extract-pipeline",
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry.entry_id,
                snapshot_id=snapshot_id,
            )
        ],
    )
    other_ref = next((r for r in other.provenance if r.type is ProvenanceRefType.SOURCE), None)
    other_entry = None
    if other_ref is not None:
        key = other_ref.corpus_entry_id
        if key not in entries:
            entries[key] = await get_entry(session, key)
        other_entry = entries[key]
    order = update_order(
        stand_in,
        entry,
        snapshot_id,
        other,
        other_entry,
        other_ref.snapshot_id if other_ref is not None else None,
        require_attribution=True,
        existing_date=await latest_source_date(session, other),
    )
    return order is not None


async def _candidate_subject_ids_readonly(
    session: AsyncSession,
    candidate: CandidateParticle,
    memo: dict[str, str | None],
    *,
    source_type: str | None = None,
) -> list[str]:
    """Resolve a candidate's subject names to *existing* Subject ids, read-only.

    The §6.6 precompute runs before the write loop resolves (and may mint)
    subjects, so it cannot use that resolution. It does not need to: a subject
    not yet in the store has no stored claims about it to pair with. A local
    name/alias lookup (no live authority, no write) answers exactly the
    question the subject-keyed search asks, memoised
    per pass.

    ``source_type`` applies the same persona fold the write path applies.
    Without it the two disagreed about the *name*, and the
    disagreement silently disabled the whole rung: the write path stores a
    conversational persona under one canonical Subject ("the user"), while this
    lookup asked for the extractor's raw surface form ("user", the commonest
    one), found nothing, and returned no key — so every claim about the speaker
    skipped the subject-keyed search. Measured on three fixed live extraction
    samples: 1 / 3 / 0 update supersessions with folding on, 13 / 10 / 13 with
    it off. The memo is keyed by the *folded* name for the same reason.

    The lookup itself is :func:`find_existing_subject` — the resolver's own
    read-only half — so the persona-alias scope cannot drift from
    the write path either: a web page naming "I" finds no key here, exactly as
    the write loop will mint a Subject for it rather than bind it to the
    persona.
    """
    out: list[str] = []
    for raw in candidate_subject_names(candidate):
        name = _persona_canonical(raw, source_type)
        key = name.casefold()
        if key not in memo:
            subject = await find_existing_subject(session, raw, source_type=source_type)
            memo[key] = subject.id if subject is not None else None
        resolved = memo[key]
        if resolved is not None and resolved not in out:
            out.append(resolved)
    return out


async def _apply_update_extras(
    session: AsyncSession,
    winner: Particle,
    winner_entry: CorpusEntry,
    winner_snapshot_id: str,
    others: list[Particle],
    retired: set[str],
    *,
    entries: Mapping[str, CorpusEntry | None],
    precondition: Mapping[str, PairPrecondition],
) -> None:
    """Resolve further confirmed same-subject pairs by rung 2.5 alone.

    The second apply after the ladder: the demotions are planned by the pure
    :func:`~particles.ingest.pair_selection.plan_update_extras` over the
    entries and preconditions read before the write lock. A slot that already
    carries several stale values (every store written before)
    converges here; a pair that does not qualify is left as it was.
    """
    for loser_id in plan_update_extras(
        winner, winner_entry, winner_snapshot_id, others, entries, precondition
    ):
        await update_particle_status(
            session, loser_id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
        )
        retired.add(loser_id)
        if loser_id == winner.id:
            log.info(
                "Update supersession (rung 2.5, extra pair): %s superseded by a newer extra pair",
                loser_id[:8],
            )
        else:
            log.info(
                "Update supersession (rung 2.5, extra pair): %s superseded by %s",
                loser_id[:8],
                winner.id[:8],
            )


def _snapshot_author_id(entry: CorpusEntry | None, snapshot_id: str | None) -> str | None:
    """Return the ``author_id`` of one snapshot on an already-loaded entry."""
    if entry is None or not snapshot_id:
        return None
    snap = next((s for s in entry.snapshots if s.snapshot_id == snapshot_id), None)
    return snap.author_id if snap else None


async def _resolve_trust_scores(
    session: AsyncSession,
    domain: str | None,
    new_entry_id: str,
    new_source_type: str,
    new_uri: str | None,
    new_author_id: str | None,
    existing_entry_id: str | None,
    existing_source_type: str,
    existing_uri: str | None,
    existing_author_id: str | None,
) -> tuple[float, float]:
    """Return (score_new, score_existing) using the Extension B layered lookup.

    If domain is known, walks the §6.4 cascade — CORPUS_ENTRY → AUTHOR →
    SOURCE_TYPE SourceTrustStatements — then falls back
    resolve_trust_score().
    """
    if domain:
        score_new = await get_layered_trust_rank(
            session, domain, new_entry_id, new_source_type, new_uri, new_author_id
        )
        score_existing = await get_layered_trust_rank(
            session,
            domain,
            existing_entry_id or "",
            existing_source_type,
            existing_uri,
            existing_author_id,
        )
        return score_new, score_existing

    # No domain inferred — fall through to the URL baseline only
    score_new = await resolve_trust_score(session, new_uri)
    score_existing = await resolve_trust_score(session, existing_uri)
    return score_new, score_existing


def _embed_batch(texts: list[str]) -> list[Any]:
    """Compute embeddings using sentence-transformers. Lazy-loaded singleton."""
    if not texts:
        return []
    model = get_embedding_model()
    if model is None:
        return [None] * len(texts)
    return list(model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))


async def _embed_batch_async(texts: list[str]) -> list[Any]:
    """Offload :func:`_embed_batch` to a worker thread.

    ``sentence-transformers`` ``model.encode`` is a synchronous, CPU-bound call.
    Run on the event loop directly it freezes every other coroutine for the
    duration — which on the client-server path means a routed write
    or extract starves the engine's loop, the exact event-loop-starvation half
    (the LLM call already offloads via ``asyncio.to_thread`` in the
    Anthropic adapter; the embedding did not). Delegating to a thread keeps the
    loop responsive. ``_embed_batch`` is resolved from the module global at call
    time, so the ``set_embedding_model`` / patch test seams still reach it.

    Span-wrapped: the span wraps the ``await`` call site (not the
    worker body), so the active OTel context parents it for free across the
    ``asyncio.to_thread`` boundary (which copies the context).
    """
    with _tracer.start_as_current_span("embed.batch") as span:
        span.set_attribute("particles.embed_count", len(texts))
        start = time.perf_counter()
        try:
            return await asyncio.to_thread(_embed_batch, texts)
        finally:
            _embed_duration.record(time.perf_counter() - start)
