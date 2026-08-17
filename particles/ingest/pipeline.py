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
from collections.abc import Sequence
from typing import Any

import numpy as np
from opentelemetry import metrics, trace
from sqlalchemy.ext.asyncio import AsyncSession

import particles.store.taxonomy_store  # noqa: F401  # register taxonomy sink
import particles.store.wikidata_cache  # noqa: F401  # register wikidata label cache
from particles.config import get_config
from particles.core.conflict_resolution import (
    ConflictVerdict,
    build_inconsistency_particle,
    resolve_conflict,
)
from particles.core.schema import (
    CorpusEntry,
    ExtractionStatus,
    ExtractorRef,
    Particle,
    ParticleType,
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
    update_extraction_status,
)
from particles.db import write_lock
from particles.embeddings import cosine_similarity, get_embedding_model
from particles.extraction.calibration import scaler_for_record
from particles.extraction.general import PageStat, candidate_to_particle
from particles.extraction.polarity import is_non_asserted
from particles.extraction.registry import ExtractorPlugin, infer_domain, select_extractor
from particles.extraction.scope import (
    apply_source_exemption,
    is_excluded_document_meta,
    is_scope_exempt_source,
)
from particles.extraction.subject_gate import gate_candidate_subjects
from particles.ingest.candidate_dedup import dedupe_exact_candidates
from particles.ingest.duplicate_suppression import (
    DuplicateIndex,
    build_duplicate_index,
    suppression_note,
)
from particles.ingest.generation import cascade_superseded_generation
from particles.ingest.narrative_merge import collapse_chunk_narratives
from particles.ingest.subject_resolver import resolve_subjects
from particles.llm import CompletionPool
from particles.store.event_store import EventRefKind, OperatorEventType, record_event
from particles.store.extractor_store import get_calibration
from particles.store.particle_store import (
    append_provenance_ref,
    compute_context_fingerprint,
    get_active_particles_for_entry,
    get_active_particles_with_embeddings,
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


def _skips_conflict_resolution(properties: dict[str, object] | None) -> bool:
    """True for particles §6.6 conflict resolution must ignore.

    Two off-the-conflict-surface classes share the same treatment — kept out of
    the candidate set and written straight to ACTIVE without conflict-checking:

    * **DOCUMENT_META** — claims about a document's own apparatus,
      not about the world.
    * **non-asserted** (cap. 1) — a document's rejected / superseded /
      deferred / counterfactual prose (``polarity`` DECLINED / HYPOTHETICAL). A
      rejected alternative must not manufacture an ``INCONSISTENCY`` against the
      chosen decision.

    Both stay stored and ACTIVE (label, never delete); the query / lint / export
    layers apply the visibility exclusion.
    """
    return is_excluded_document_meta(properties) or is_non_asserted(properties)


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
) -> list[Particle]:
    """Run extraction for a single corpus snapshot.

    Returns the list of Particle objects written to the store (ACTIVE or INCONSISTENCY).
    Extractor is selected from the plugin registry by source_type.
    If page_stats_out is provided, page-level stats from PDF extraction are appended to it.
    If carry_forward_ids_out is provided, it is extended with the IDs of
    existing ACTIVE particles that the extractor's chunk-hash carry-forward
    matched — reindex callers should exclude these from supersession.
    If suppressed_ids_out is provided, it is extended with the ID of the
    existing ACTIVE particle each suppressed duplicate candidate was folded into —
    one entry per suppressed candidate, so ``len()`` is the suppression count.
    Like carry-forward, these particles must be excluded from supersession.
    ``completion_pool`` is the latency-tolerance assertion, threaded
    as a parameter and never sniffed: only the consolidation extract pass
    passes one, and pool-aware extractors then merge their LLM requests into
    the pooled half-price batch. Interactive callers leave it ``None`` and get
    today's sequential calls unchanged.

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
    # candidates are short-circuited by the extractor in this same call
    #, so any candidate the extractor returns here is a fresh
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
    # journal path.
    result.candidates, merge_notes = await collapse_chunk_narratives(result.candidates)
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
    if existing:
        all_with_embs = await get_active_particles_with_embeddings(session, min_confidence=0.0)
        existing_emb_map = {p.id: emb for p, emb in all_with_embs}
        existing_embs = [existing_emb_map.get(p.id) for p in existing]  # type: ignore[misc]

    written: list[Particle] = []

    # Determine extractor identity for provenance (use class-level attrs)
    ext_ref = ExtractorRef(name=extractor.EXTRACTOR_ID, version=extractor.EXTRACTOR_VERSION)
    ext_agent = extractor.EXTRACTOR_ID

    # select the calibration fitted under the model now producing these
    # confidences, keyed by provider_model. Calibration is provider-sensitive
    #, so a record exists for this pairing only if it was benchmarked
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
    # the source's genre; the Engine decides what the classification *does*
    #. Under ``mode: suppress`` a DOCUMENT_META candidate is dropped
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
    # ``prechecks[i]`` is the (conflict, has_contradiction_signal) pair for
    # ``result.candidates[i]``; DOCUMENT_META / non-asserted candidates skip
    # §6.6 and need no signal (mirrors the write-loop guard below).
    await session.commit()
    prechecks: list[tuple[Particle | None, bool]] = []
    for i, candidate in enumerate(result.candidates):
        if _skips_conflict_resolution(candidate.properties):
            prechecks.append((None, False))
            continue
        cand_emb = embeddings[i] if i < len(embeddings) else None
        conflict = _find_conflict(
            cand_emb,
            existing,
            existing_embs,
            candidate_stance_holder=holder_from_properties(candidate.properties),
        )
        # Extraction is fail-open: a probe that could not complete (None) maps to
        # "no contradiction" (today's behaviour). The assertion pathway
        # fails closed instead, but it does not precompute the signal here.
        signal = (
            bool(await _has_contradiction_signal(candidate.content, conflict.content))
            if conflict is not None
            else False
        )
        prechecks.append((conflict, signal))

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

    # hold the cross-process writer lock around the write phase
    # only. The LLM extraction and the §6.6 contradiction probes above ran
    # OUTSIDE it (the IN_PROGRESS-claim commit and the line-420 precheck
    # commit both released the SQLite write lock), so the lock is never held
    # across a network round-trip. We commit the particle writes here, under
    # the lock — releasing SQLite's write lock before the advisory lock — so
    # the caller's later commit is a no-op.
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

            # DOCUMENT_META and non-asserted candidates skip
            # §6.6 entirely — they would only manufacture spurious INCONSISTENCY.
            # Write straight to ACTIVE (stored, label-not-delete).
            if _skips_conflict_resolution(particle.properties):
                validate_transition(None, Status.ACTIVE)
                await insert_particle(session, particle, emb_list)
                written.append(particle)
                index_to_id[i] = particle.id
                continue

            # the exact-duplicate rung, ABOVE §6.6. If this claim is
            # already held verbatim by an ACTIVE particle with the same subjects
            # and stance holder, don't mint a second one — record this source on
            # the existing particle and move on. Deterministic (content identity,
            # not similarity), so nothing distinguishable leaves the ACTIVE
            # surface: the claim is on it character for character. Note the
            # stance/narrative edge-writers below bind through ``index_to_id``,
            # which maps this candidate's position onto the surviving particle,
            # so an edge whose endpoint was suppressed still lands correctly.
            duplicate_of = dup_index.find(particle)
            if duplicate_of is not None:
                for prov_ref in particle.provenance:
                    await append_provenance_ref(session, duplicate_of.id, prov_ref)
                suppressed_into.append(duplicate_of.id)
                index_to_id[i] = duplicate_of.id
                log.info(
                    "Duplicate suppressed: candidate %d already ACTIVE as %s",
                    i,
                    duplicate_of.id[:8],
                )
                continue

            # §6.6 conflict candidate + contradiction signal were precomputed before
            # the write loop (F4.3), keeping the LLM call out of the write transaction.
            conflict, has_signal = prechecks[i]

            if conflict is None:
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
                resolved = await _resolve_conflict(
                    session,
                    particle,
                    conflict,
                    entry_id,
                    snapshot_id,
                    has_signal=has_signal,
                    new_embedding=emb,
                )
                if resolved is not None:
                    written.append(resolved)
                    index_to_id[i] = resolved.id
                    dup_index.add(resolved)

        # create the ENDORSES/DISPUTES edges now that every candidate's
        # stored id is known. A stance binds to the sibling claim it endorses /
        # disputes; if either endpoint was dropped (conflict-superseded) or the
        # target resolves to the stance itself, the edge is skipped (no identifiable
        # target — an unbound stance has nothing to aggregate).
        for stance_idx, target_idx, kind in stance_specs:
            s_id = index_to_id.get(stance_idx)
            t_id = index_to_id.get(target_idx)
            if s_id is not None and t_id is not None and s_id != t_id:
                await create_relation(session, s_id, t_id, kind, RelationCreatedBy.EXTRACTOR_DIRECT)

        # write the entry-level NARRATIVE graph. The journal extractor
        # emits exactly one NARRATIVE candidate (particle_type == NARRATIVE) plus a
        # narrative_index on each constituent claim. Link every constituent to the
        # narrative via PART_OF (constituent → narrative) and consecutive
        # constituents via SEQUENCE_IN (predecessor → successor), in narrative_index
        # order — the same index_to_id → create_relation pattern the stance edges
        # use above. Edges are written only when both endpoints landed; a
        # conflict-dropped constituent is simply absent from the chain.
        narrative_ids = [
            index_to_id[i]
            for i, c in enumerate(result.candidates)
            if c.particle_type == ParticleType.NARRATIVE and i in index_to_id
        ]
        if len(narrative_ids) == 1:
            narrative_id = narrative_ids[0]
            constituents: list[tuple[int, int]] = []
            for i, c in enumerate(result.candidates):
                ni = c.narrative_index
                if ni is None or c.particle_type == ParticleType.NARRATIVE or i not in index_to_id:
                    continue
                constituents.append((ni, i))
            constituents.sort()
            prev_id: str | None = None
            for _ni, i in constituents:
                cid = index_to_id[i]
                if cid == narrative_id:
                    continue
                await create_relation(
                    session,
                    cid,
                    narrative_id,
                    RelationType.PART_OF,
                    RelationCreatedBy.EXTRACTOR_DIRECT,
                )
                if prev_id is not None and prev_id != cid:
                    await create_relation(
                        session,
                        prev_id,
                        cid,
                        RelationType.SEQUENCE_IN,
                        RelationCreatedBy.EXTRACTOR_DIRECT,
                    )
                prev_id = cid

        # capture every URL mentioned in this snapshot as a citation
        # signal (best-effort; never fails extraction). Runs once extraction has
        # succeeded, in the same transaction as the COMPLETE status below.
        await _capture_url_mentions(session, entry_id, content)

        # a MUTABLE source's new snapshot retires the generation it
        # replaced. Runs HERE — after extraction, in the same transaction as the
        # COMPLETE status below — and not at deposit time, so
        # carry-forward has already kept the particles whose chunks are
        # unchanged. Those are excluded explicitly: a carried-forward particle
        # still points at the snapshot it was first extracted from (provenance is
        # deliberately not mutated), which would otherwise read as a stale
        # generation. A no-op for every non-MUTABLE entry, and zero LLM calls.
        # particles are excluded for the same reason as carry-forward:
        # a suppressed-into particle is still current (this snapshot re-observed
        # its claim), but its provenance edge still names the snapshot it was
        # first extracted from, so the cascade would otherwise read it as a
        # stale generation and demote the very particle the suppression kept.
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
        if suppressed_ids_out is not None:
            suppressed_ids_out.extend(suppressed_into)

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

    # DOCUMENT_META and non-asserted never participate in
    # §6.6 (mirrors extract_snapshot). They are also excluded from the
    # conflict-candidate set, so neither is ever appended to ``candidate_cache``.
    if _skips_conflict_resolution(particle.properties):
        validate_transition(None, Status.ACTIVE)
        await insert_particle(session, particle, emb_list)
        return particle

    # the exact-duplicate rung runs here too, above §6.6, so
    # "duplicate" means one thing on both write paths. Without it an exact
    # duplicate reaches the ladder, finds no contradiction signal, returns
    # CORROBORATES, and is written as a second ACTIVE particle — which is why
    # ``particle_assert`` was not idempotent before this ADR.
    if get_config().extraction.duplicate_suppression.enabled:
        dup_index = await build_duplicate_index(session, [particle.content])
        duplicate_of = dup_index.find(particle)
        if duplicate_of is not None:
            for ref in particle.provenance:
                await append_provenance_ref(session, duplicate_of.id, ref)
            log.info(
                "Duplicate suppressed: assertion already ACTIVE as %s",
                duplicate_of.id[:8],
            )
            return duplicate_of

    if candidate_cache is None:
        pairs = await load_active_conflict_candidates(session)
    else:
        pairs = candidate_cache
    existing = [p for p, _ in pairs]
    existing_embs = [e for _, e in pairs]

    conflict = _find_conflict(emb, existing, existing_embs)
    if conflict is None:
        validate_transition(None, Status.ACTIVE)
        await insert_particle(session, particle, emb_list)
        if candidate_cache is not None and emb is not None:
            candidate_cache.append((particle, emb))
        return particle

    # The INCONSISTENCY wrapper's trigger ref points at the corpus entry that
    # produced the conflicting candidate — the first SOURCE-typed ref. A
    # derived particle has only PARTICLE-typed premise refs; its
    # trigger is the particle itself, and the wrapper's trigger ref stays
    # PARTICLE-typed so a particle id is never mislabelled as a corpus entry.
    prov = next((r for r in particle.provenance if r.type is ProvenanceRefType.SOURCE), None)
    if prov is not None:
        corpus_entry_id = prov.corpus_entry_id
        snapshot_id = prov.snapshot_id or ""
        trigger_ref_type = ProvenanceRefType.SOURCE
    else:
        corpus_entry_id = particle.id
        snapshot_id = ""
        trigger_ref_type = ProvenanceRefType.PARTICLE
    return await _resolve_conflict(
        session,
        particle,
        conflict,
        corpus_entry_id,
        snapshot_id,
        candidate_cache=candidate_cache,
        new_embedding=emb,
        single_trust_order=single_trust_order,
        fail_closed=fail_closed,
        trigger_ref_type=trigger_ref_type,
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
    paraphrases and attribution/quoting wrappers also embed near each other
    . The §6.6 ladder must not fire without positive confirmation.

    Order of checks (cheapest first):
      1. Attribution / quoting surface patterns → no contradiction (``False``).
      2. LLM contradiction check (L-SEM-01-style prompt) → ``True`` only on YES,
         ``False`` on NO, ``None`` if the probe could not run.
    """
    if _is_attribution_paraphrase(content_a, content_b):
        return False
    return await _llm_confirms_contradiction(content_a, content_b)


async def _resolve_conflict(
    session: AsyncSession,
    new_particle: Particle,
    existing: Particle,
    corpus_entry_id: str,
    snapshot_id: str,
    *,
    has_signal: bool | None = None,
    candidate_cache: list[EmbeddingPair] | None = None,
    new_embedding: np.ndarray[Any, np.dtype[np.float32]] | None = None,
    single_trust_order: bool | None = None,
    fail_closed: bool = False,
    trigger_ref_type: ProvenanceRefType = ProvenanceRefType.SOURCE,
) -> Particle | None:
    """Apply the §6.6 conflict-resolution ladder for one (existing, new) pair.

    Splits cleanly between the pure decision (delegated to
    :func:`particles.core.conflict_resolution.resolve_conflict`) and the
    side-effecting parts that touch the DB / LLM (handled here):

      - Resolve the contradiction signal via the attribution-pattern and LLM
        gate.
      - Look up trust scores via the Extension B layered lookup, falling
        back to the URL baseline.
      - Persist whatever the verdict says: insert the new particle as
        ACTIVE, demote the existing particle to PROVENANCE_STALE, or
        build & insert the INCONSISTENCY particle (with ``domain_hint``).

    ``has_signal`` may be supplied by a caller that precomputed the
    contradiction signal outside the write transaction (F4.3); when ``None``
    the LLM gate runs here. ``new_embedding`` is the candidate's embedding —
    stored alongside whichever row this verdict persists, and used with
    ``candidate_cache`` (when given) to mirror the write in place: the new
    particle is appended when it lands ACTIVE and a trust-superseded existing
    one removed, so a batch caller's later items reconcile against this
    verdict.

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
    # Tri-state contradiction signal. A precomputed ``has_signal``
    # is always a bool (extraction coerces None→False before passing). When not
    # precomputed, run the probe here: True/False is a verdict; None means the
    # probe could not complete, and ``fail_closed`` decides — extraction stays
    # fail-open (None→False), the assertion pathway quarantines (force INCONSISTENT).
    force_inconsistent = False
    if has_signal is None:
        probe = await _has_contradiction_signal(new_particle.content, existing.content)
        if probe is None:
            if fail_closed:
                force_inconsistent = True
                has_signal = True  # run the trust/domain lookup; verdict overridden below
            else:
                has_signal = False
        else:
            has_signal = probe

    # Resolve trust inputs up front when the ladder might consult them.
    # Skip the lookup entirely for the gate-corroborates case — it ignores
    # trust scores. Also resolve domain for the INCONSISTENCY domain_hint
    # (used by the Extension B cascade).
    domain: str | None = None
    score_new: float | None = None
    score_existing: float | None = None
    # cap. 2 rung-1.5 inputs (default off → no prior). Only resolved
    # when there is a confirmed conflict to adjudicate (inside ``if has_signal``).
    new_supersedes_existing = False
    existing_supersedes_new = False
    if has_signal:
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

    # in a multi-contributor / consensus store there is no global
    # trust order, so auto-supersede (rung 2) is suppressed and a confirmed
    # contradiction surfaces as an INCONSISTENCY instead. The effective regime is
    # resolved per-store by the caller and passed as ``single_trust_order``
    #; when not supplied it falls back to the
    # process-global default, so extraction / interchange are byte-for-byte unchanged.
    effective_single = (
        single_trust_order
        if single_trust_order is not None
        else (get_config().reconciliation.store_mode == "single")
    )
    if force_inconsistent:
        # Probe could not complete and the assertion pathway fails closed: skip
        # trust resolution and quarantine the candidate.
        verdict = ConflictVerdict.INCONSISTENT
    else:
        verdict = resolve_conflict(
            existing,
            new_particle,
            has_contradiction_signal=has_signal,
            new_supersedes_existing=new_supersedes_existing,
            existing_supersedes_new=existing_supersedes_new,
            trust_score_existing=score_existing,
            trust_score_new=score_new,
            trust_differential_threshold=get_config().trust.differential_threshold,
            single_trust_order=effective_single,
        )

    # The candidate's embedding, in storable form — persisted with whichever
    # row this verdict writes so conflict-path particles stay searchable.
    # Mirrors the tolist/list dance in extract_snapshot: a mocked embedding
    # model may yield plain lists instead of numpy arrays.
    new_emb_list: list[float] | None
    if new_embedding is None:
        new_emb_list = None
    elif hasattr(new_embedding, "tolist"):
        new_emb_list = new_embedding.tolist()
    else:
        new_emb_list = list(new_embedding)

    # Keep a batch caller's ``candidate_cache`` consistent with the writes below
    # (F4.3): a newly-ACTIVE particle joins the candidate set; a trust-superseded
    # one (no longer ACTIVE) leaves it. No-ops when no cache was passed.
    def _cache_add_new() -> None:
        if candidate_cache is not None and new_embedding is not None:
            candidate_cache.append((new_particle, new_embedding))

    def _cache_drop_existing() -> None:
        if candidate_cache is not None:
            candidate_cache[:] = [pair for pair in candidate_cache if pair[0].id != existing.id]

    if verdict is ConflictVerdict.CORROBORATES:
        validate_transition(None, Status.ACTIVE)
        await insert_particle(session, new_particle, new_emb_list)
        _cache_add_new()
        log.info(
            "High-similarity pair without contradiction signal — new %s written as"
            " ACTIVE alongside existing %s (no §6.6 conflict)",
            new_particle.id[:8],
            existing.id[:8],
        )
        return new_particle

    if verdict is ConflictVerdict.SUPERSEDES:
        validate_transition(None, Status.ACTIVE)
        await insert_particle(session, new_particle, new_emb_list)
        await update_particle_status(
            session,
            existing.id,
            Status.PROVENANCE_STALE,
            StatusReason.LOWER_TRUST_SOURCE,
        )
        _cache_drop_existing()
        _cache_add_new()
        log.info(
            "Trust resolution: new %s (%.2f) preferred over existing %s (%.2f)",
            new_particle.id[:8],
            score_new if score_new is not None else float("nan"),
            existing.id[:8],
            score_existing if score_existing is not None else float("nan"),
        )
        return new_particle

    if verdict is ConflictVerdict.SUPERSEDED_BY_EXISTING:
        # The candidate stays a drop — it is redundant with a strictly
        # better existing claim — but an audited one: the event
        # log keeps the excerpt, verdict, and winning particle id. The
        # candidate is never persisted, so it appears in the payload only,
        # not as a record ref.
        await record_event(
            session,
            actor="extract-pipeline",
            event_type=OperatorEventType.CONFLICT_CANDIDATE_DROPPED,
            reason="§6.6 trust resolution preferred the existing particle",
            refs=[(EventRefKind.PARTICLE, existing.id)],
            payload={
                "verdict": verdict.value,
                "candidate_id": new_particle.id,
                "candidate_excerpt": new_particle.content[:240],
                "winning_particle_id": existing.id,
                "trust_score_new": score_new,
                "trust_score_existing": score_existing,
            },
        )
        log.info(
            "Trust resolution: existing %s (%.2f) preferred over new %s (%.2f);"
            " new particle dropped (event logged)",
            existing.id[:8],
            score_existing if score_existing is not None else float("nan"),
            new_particle.id[:8],
            score_new if score_new is not None else float("nan"),
        )
        return None

    if verdict is ConflictVerdict.DOCUMENT_SUPERSEDES:
        # Rung 1.5 (cap. 2): new's provenance document (transitively)
        # supersedes existing's. Insert new ACTIVE; demote the existing claim
        # ACTIVE → PROVENANCE_STALE / DOCUMENT_SUPERSEDED — reusing the trust
        # rung's demotion machinery and demotion-only invariant, so
        # the retired decision stays in the store, auditable and off the default
        # surface. No INCONSISTENCY is queued.
        validate_transition(None, Status.ACTIVE)
        await insert_particle(session, new_particle, new_emb_list)
        await update_particle_status(
            session,
            existing.id,
            Status.PROVENANCE_STALE,
            StatusReason.DOCUMENT_SUPERSEDED,
        )
        _cache_drop_existing()
        _cache_add_new()
        log.info(
            "Document-supersession (rung 1.5): new %s supersedes existing %s"
            " → existing demoted DOCUMENT_SUPERSEDED",
            new_particle.id[:8],
            existing.id[:8],
        )
        return new_particle

    if verdict is ConflictVerdict.DOCUMENT_SUPERSEDED_BY_EXISTING:
        # Rung 1.5 mirror: existing's document supersedes new's. The new
        # candidate is a retired-document decision; existing stays ACTIVE. Store
        # the loser but demote it — insert ACTIVE then transition (the insert
        # seam forbids a born-PROVENANCE_STALE row except for the CONFLICT_PENDING
        # quarantine birth), landing it PROVENANCE_STALE / DOCUMENT_SUPERSEDED
        # (auditable, off the default surface, never an ACTIVE candidate).
        validate_transition(None, Status.ACTIVE)
        await insert_particle(session, new_particle, new_emb_list)
        await update_particle_status(
            session,
            new_particle.id,
            Status.PROVENANCE_STALE,
            StatusReason.DOCUMENT_SUPERSEDED,
        )
        log.info(
            "Document-supersession (rung 1.5): existing %s supersedes new %s"
            " → new stored DOCUMENT_SUPERSEDED",
            existing.id[:8],
            new_particle.id[:8],
        )
        return new_particle.model_copy(
            update={
                "status": Status.PROVENANCE_STALE,
                "status_reason": StatusReason.DOCUMENT_SUPERSEDED,
            }
        )

    if verdict is ConflictVerdict.INCONSISTENT:
        # persist the losing candidate as a real particle born
        # quarantined — full content, provenance (incl. chunk_hash),
        # confidence, subjects, and embedding intact. PROVENANCE_STALE keeps
        # it out of query/lint by the existing status filters; the
        # CONFLICT_PENDING reason carries the real semantics and is what the
        # insert seam requires for this birth. Review recovers it on
        # PREFER_B / BOTH_VALID; the wrapper's B ref below points at this
        # persisted row instead of a dangling UUID (P4-2).
        quarantined = new_particle.model_copy(
            update={
                "status": Status.PROVENANCE_STALE,
                "status_reason": StatusReason.CONFLICT_PENDING,
            }
        )
        validate_transition(None, Status.PROVENANCE_STALE)
        await insert_particle(session, quarantined, new_emb_list)

        inc_particle = build_inconsistency_particle(
            existing,
            quarantined,
            corpus_entry_id=corpus_entry_id,
            snapshot_id=snapshot_id,
            asserted_by="extract-pipeline",
            trigger_ref_type=trigger_ref_type,
        )
        validate_transition(None, Status.INCONSISTENCY)
        await insert_particle(session, inc_particle, domain_hint=domain)
        log.info(
            "INCONSISTENCY particle %s created (conflicts: %s ↔ quarantined %s,"
            " domain=%s, subject_ids=%d inherited)",
            inc_particle.id,
            existing.id,
            quarantined.id,
            domain,
            len(inc_particle.subject_ids),
        )
        return inc_particle

    # ConflictVerdict.NO_CONFLICT — reachable only if a caller starts using
    # resolve_conflict for below-threshold pairs. The pipeline gates on
    # similarity in _find_conflict, so we never get here today — but treat
    # the candidate as ACTIVE for safety rather than dropping it.
    validate_transition(None, Status.ACTIVE)
    await insert_particle(session, new_particle, new_emb_list)
    _cache_add_new()
    return new_particle


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
