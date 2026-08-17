"""Agent belief-write orchestration.

The §6.6-reconciling write verbs an MCP agent (or any HTTP client) drives —
``assert_belief`` (the flagship), ``supersede_belief``, ``retract_belief``, and
the standalone cheap ``deposit_conversation_text``. This module is the **single
convergence point** for them: the local MCP write tools reach it
through ``LocalBackend``; the remote engine reaches it through the belief-write
HTTP endpoints. Before this orchestration lived inside
``particles/mcp/tools/write.py`` and was reachable only locally; lifting it here
is what makes belief-writes remoteable, closing circular deferral.

Each function:

* constructs the ``Particle`` **server-side** (§4a) — the caller never supplies a
  trust-, status-, or identity-bearing field. ``extractor_ref`` is omitted,
  ``calibration_source`` is forced ``AGENT_ASSERTED`` (the honest label for an
  uncalibrated agent self-report), ``confidence.value`` is clamped to
  ``mcp.write.max_asserted_confidence``, ``asserted_by`` is the server-bound
  ``mcp.write.asserter_identity``, and ``status`` is owned by the verb;
* reconciles through the §6.6 ladder in the store's resolved mode (``multi`` /
  consensus for a write store) and **fail-closed** (b/§6) so
  a confirmed (or unverifiable) contradiction is quarantined + surfaced as an
  INCONSISTENCY, never an auto-supersede;
* records the audit event;
* returns an :class:`AgentWriteResult` carrying the §6.6 verdict — a conflict is
  a first-class result, not an error.

The functions take an open session and **do not** open it, commit it, or gate
the store: the *caller* (``LocalBackend``, which opens ``session_scope(store)``
and commits; or the FastAPI endpoint, which uses ``SessionDep`` and commits)
owns the transaction boundary, and the **store write-enablement gate is a
surface concern** — ``mcp.write.enabled_stores`` is checked by the MCP tool's
``_resolve_store`` locally and by the engine endpoint server-side.
``store`` is threaded through only for the per-store §6.6 mode lookup.

Mutating another principal's beliefs (supersede/retract) is own-beliefs-only
unless ``mcp.write.allow_cross_asserter``; operator-asserted (``HUMAN_REVIEW``)
particles are never agent-mutable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from particles.config import get_config
from particles.core.granularity import granularity_violation
from particles.core.schema import (
    Confidence,
    Particle,
    PolicyProvenance,
    ProvenanceRef,
    ProvenanceRefType,
    SourceRef,
    SourceRefType,
    SourceTrustStatement,
    SourceType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.corpus.deposit import deposit_text as _deposit_text_seam
from particles.extraction.registry import infer_domain
from particles.store.event_store import EventRefKind, OperatorEventType, record_event


class AgentWriteResult(BaseModel):
    """The §6.6 verdict of a belief write — a first-class result, not an error.

    ``asserted_particle_id`` is always the *agent's* belief id (the constructed
    candidate), never the INCONSISTENCY meta-particle's id; on a conflict that
    belief is the quarantined ``CONFLICT_PENDING`` particle and
    ``inconsistency_id`` names the separate INCONSISTENCY record. It is ``None``
    only on a lower-trust drop (unreachable on a consensus write store).
    """

    asserted_particle_id: str | None
    verdict: str
    status: str | None = None
    inconsistency_id: str | None = None


def _clamp_confidence(value: float) -> float:
    ceiling = get_config().mcp.write.max_asserted_confidence
    return max(0.0, min(value, ceiling))


async def _seed_author_trust(session: Any, identity: str) -> None:
    """Idempotently seed the AUTHOR-scoped trust statement for an asserter (§6/§6a).

    Seeded under the same domain ``infer_domain`` resolves for a CONVERSATION
    excerpt, so the §6.4 AUTHOR tier actually binds at query/conflict time. Never
    overwrites an existing statement — an operator demotion via the ``trust`` CLI
    must win (demotion-only).
    """
    from particles.store.trust_store import get_trust_rank, insert_trust_statement

    domain = infer_domain(SourceType.CONVERSATION)
    if domain is None:
        return  # no trust domain configured; the seed would be unreachable anyway
    existing = await get_trust_rank(session, domain, SourceRefType.AUTHOR.value, identity)
    if existing is not None:
        return
    await insert_trust_statement(
        session,
        SourceTrustStatement(
            domain=domain,
            source_ref=SourceRef(type=SourceRefType.AUTHOR, value=identity),
            trust_rank=get_config().mcp.write.agent_trust_rank,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="mcp.write.agent_trust_rank",
        ),
    )


async def _resolve_provenance(
    session: Any,
    identity: str,
    *,
    source_excerpt: str | None,
    corpus_entry_id: str | None,
) -> tuple[str, str]:
    """Return (corpus_entry_id, snapshot_id) for the assertion's SOURCE ref (§3).

    Deposits ``source_excerpt`` as a CONVERSATION entry attributed to ``identity``
    (the deposit-excerpt pattern). If an existing ``corpus_entry_id`` is given
    instead, reuses its latest snapshot. An assertion with neither is rejected —
    an unprovenanced belief is the failure mode this design exists to end (§4).
    """
    if source_excerpt is not None:
        return await _deposit_text_seam(
            session,
            source_excerpt,
            deposited_by=identity,
            source_type=SourceType.CONVERSATION,
            author_id=identity,
        )
    if corpus_entry_id is not None:
        from particles.corpus.store import get_entry

        entry = await get_entry(session, corpus_entry_id)
        if entry is None or not entry.snapshots:
            raise ValueError(f"corpus_entry_id {corpus_entry_id!r} not found or has no snapshot.")
        return corpus_entry_id, entry.snapshots[-1].snapshot_id
    raise ValueError(
        "An assertion requires provenance: pass `source_excerpt` (deposited as the "
        "belief's source) or an existing `corpus_entry_id`."
    )


async def _construct_and_insert(
    session: Any,
    store: str,
    identity: str,
    *,
    content: str,
    subject_names: list[str],
    confidence_value: float,
    uncertainty_nature: str,
    source_excerpt: str | None,
    corpus_entry_id: str | None,
    tags: list[str] | None,
    supersedes: str | None,
    carry_over: Particle | None = None,
    subject_ids: list[str] | None = None,
    granularity: tuple[int, int] | None = None,
) -> tuple[str, Particle | None]:
    """Server-side particle construction + §6.6 consensus/fail-closed insert (§4a/§6b).

    Returns ``(candidate_id, reconcile_result)``. ``candidate_id`` is the
    constructed particle's own id — the agent's belief — which is persisted under
    that id whether it lands ACTIVE or is quarantined as the loser of a conflict.
    The reconcile result is the same particle (ACTIVE), the INCONSISTENCY particle
    (conflict — a *different* id), or ``None`` (dropped); callers must not confuse
    the result id with the belief id (that was the M6 review finding).

    Provenance carry-over: when ``carry_over`` is a predecessor
    particle, the successor copies its **full** ``confidence`` record (value +
    calibration_source / calibration_method / calibration_ref), its
    ``extractor_ref``, and its SOURCE provenance verbatim — it is the *same
    extracted claim* with a corrected linkage, not a re-statement, so the
    operator must not reset the calibrated confidence or re-attribute the
    extractor's claim to themselves. ``confidence_value`` and the deposit /
    ``corpus_entry_id`` provenance args are then ignored. ``subject_ids``, when
    given, is used directly (already-resolved Subject ids) instead of resolving
    ``subject_names``.
    """
    from particles.ingest.pipeline import reconcile_and_insert
    from particles.ingest.subject_resolver import resolve_subjects

    try:
        nature = UncertaintyNature(uncertainty_nature)
    except ValueError as exc:
        allowed = ", ".join(n.value for n in UncertaintyNature)
        raise ValueError(
            f"Unknown uncertainty_nature {uncertainty_nature!r}. Allowed: {allowed}."
        ) from exc

    # Claim-granularity soft-gate (§3.3): reject a compound/multi-claim
    # assertion before any DB work. Deterministic size proxy, interim;
    # the COMPOUND_ASSERTION lint reads the same knobs so the two cannot drift.
    write_cfg = get_config().mcp.write
    max_chars, max_sentences = granularity or (
        write_cfg.max_assertion_chars,
        write_cfg.max_assertion_sentences,
    )
    granularity_reason = granularity_violation(
        content,
        max_chars=max_chars,
        max_sentences=max_sentences,
    )
    if granularity_reason is not None:
        raise ValueError(
            f"Assertion rejected by the claim-granularity gate: {granularity_reason}. "
            "Split it into separate particle_assert calls (interim soft-gate; "
            "a compatibility interim)."
        )

    if subject_ids is None:
        subject_ids = await resolve_subjects(
            session,
            list(subject_names),
            identity,
            particle_content=content,
            source_type=SourceType.CONVERSATION,
        )

    if carry_over is not None:
        # Linkage-correction successor: keep the predecessor's
        # calibrated confidence, extractor attribution, and source provenance.
        # No new deposit — the claim's origin is unchanged.
        confidence = carry_over.confidence
        provenance = list(carry_over.provenance)
        extractor_ref = carry_over.extractor_ref
        # the production history travels with the attribution —
        # it is the same claim, produced once, by whatever produced it.
        provider_model = carry_over.extraction_provider_model
        asserted_by = carry_over.asserted_by
    else:
        entry_id, snapshot_id = await _resolve_provenance(
            session, identity, source_excerpt=source_excerpt, corpus_entry_id=corpus_entry_id
        )
        confidence = Confidence(
            value=_clamp_confidence(confidence_value),
            calibration_source=CalibrationSource.AGENT_ASSERTED,
        )
        provenance = [
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_id,
                snapshot_id=snapshot_id,
            )
        ]
        extractor_ref = None  # omitted (§4) — this is a direct assertion.
        # Likewise omitted: no model produced this claim, the
        # agent asserted it; ``calibration_source=AGENT_ASSERTED`` says so.
        provider_model = None
        asserted_by = identity

    particle = Particle(
        content=content,
        confidence=confidence,
        uncertainty_nature=nature,
        provenance=provenance,
        asserted_by=asserted_by,
        status=Status.ACTIVE,
        subject_ids=subject_ids,
        tags=list(tags) if tags else None,
        supersedes=supersedes,
        extractor_ref=extractor_ref,
        extraction_provider_model=provider_model,
    )
    await _seed_author_trust(session, identity)
    single = get_config().reconciliation_mode_for(store) == "single"
    result = await reconcile_and_insert(
        session, particle, single_trust_order=single, fail_closed=True
    )
    return particle.id, result


def _map_result(candidate_id: str, returned: Particle | None) -> AgentWriteResult:
    """Map a reconcile result to the §6.6 verdict — a conflict is first-class.

    ``asserted_particle_id`` is always the agent's belief id (``candidate_id``),
    never the INCONSISTENCY meta-particle's id; on a conflict that belief is the
    quarantined ``CONFLICT_PENDING`` particle and ``inconsistency_id`` names the
    separate INCONSISTENCY record.
    """
    if returned is None:
        # Lower-trust drop — unreachable on a consensus write store (rung 2 is
        # suppressed); the candidate was not persisted, so no belief id to return.
        return AgentWriteResult(asserted_particle_id=None, verdict="DROPPED_LOWER_TRUST")
    if returned.status == Status.INCONSISTENCY:
        return AgentWriteResult(
            asserted_particle_id=candidate_id,  # the quarantined belief (CONFLICT_PENDING)
            verdict="INCONSISTENCY_RAISED",
            status=StatusReason.CONFLICT_PENDING.value,
            inconsistency_id=returned.id,
        )
    return AgentWriteResult(
        asserted_particle_id=candidate_id,  # == returned.id when it lands ACTIVE
        verdict="ASSERTED",
        status=returned.status.value,
    )


async def _load_mutable_target(
    session: Any, particle_id: str, identity: str, *, operator: bool = False
) -> Particle:
    """Fetch a particle and enforce the §6 mutation guards (ownership + HUMAN_REVIEW).

    The default (agent) path is own-beliefs-only: a target asserted by another
    principal is rejected unless ``mcp.write.allow_cross_asserter``. The
    ``operator`` path skips **only** the ownership check, so an
    operator may supersede / retract an *extracted* belief (asserted by the
    extractor, not the agent) — the case that fills a curation queue. It does
    **not** relax the other guards: the HUMAN_REVIEW guard and the ACTIVE-status
    check still apply. The operator path is reached only behind the
    ``mcp.write.enabled_stores`` + bearer gate (the surface concern), never the
    dev-key loopback skip.
    """
    from particles.store.particle_store import get_particle

    target = await get_particle(session, particle_id)
    if target is None:
        raise ValueError(f"Particle {particle_id!r} not found.")
    if target.confidence.calibration_source == CalibrationSource.HUMAN_REVIEW:
        raise ValueError(
            f"Particle {particle_id!r} is operator-asserted (HUMAN_REVIEW) and is not "
            "agent-mutable — revising it is Review's job."
        )
    if (
        not operator
        and target.asserted_by != identity
        and not get_config().mcp.write.allow_cross_asserter
    ):
        raise ValueError(
            f"Particle {particle_id!r} was asserted by {target.asserted_by!r}, not "
            f"{identity!r}; cross-asserter mutation requires "
            "mcp.write.allow_cross_asserter=true."
        )
    if target.status != Status.ACTIVE:
        raise ValueError(f"Particle {particle_id!r} is {target.status.value}, not ACTIVE.")
    return target


async def assert_belief(
    session: Any,
    *,
    store: str,
    content: str,
    subject_names: list[str],
    confidence: float,
    source_excerpt: str | None = None,
    corpus_entry_id: str | None = None,
    uncertainty_nature: str = "EPISTEMIC",
    tags: list[str] | None = None,
    identity: str | None = None,
    subject_ids: list[str] | None = None,
    granularity: tuple[int, int] | None = None,
) -> AgentWriteResult:
    """Assert one belief through the §6.6 ladder (the flagship).

    Deposits ``source_excerpt`` as the belief's provenance and asserts one
    particle. Trust-/status-/identity-bearing fields are constructed server-side
    (§4a). A confirmed contradiction surfaces as an INCONSISTENCY (consensus
    mode) rather than replacing the existing belief. Does not commit — the caller
    owns the transaction.

    Args:
        identity: Overrides the server-bound asserting principal. Still
            server-side — a *surface* selects it, never a client call. The façade passes its own principal so façade-origin claims
            stay separately attributable from the native surface's.
        subject_ids: Pre-resolved Subject ids, used instead of resolving
            ``subject_names`` through the authority ladder. The façade
            resolves bare-locally because a live authority can rewrite a
            canonical name and break its exact round-trip contract.
        granularity: ``(max_chars, max_sentences)`` overriding the
            ``mcp.write`` claim-granularity soft-gate, for a surface whose
            external contract sets its own limits.
    """
    identity = identity or get_config().mcp.write.asserter_identity
    candidate_id, result = await _construct_and_insert(
        session,
        store,
        identity,
        content=content,
        subject_names=subject_names,
        confidence_value=confidence,
        uncertainty_nature=uncertainty_nature,
        source_excerpt=source_excerpt,
        corpus_entry_id=corpus_entry_id,
        tags=tags,
        supersedes=None,
        subject_ids=subject_ids,
        granularity=granularity,
    )
    await record_event(
        session,
        actor=identity,
        event_type=OperatorEventType.PARTICLE_ASSERTED,
        # The belief is persisted (ACTIVE or quarantined) iff reconcile didn't drop it.
        refs=[(EventRefKind.PARTICLE, candidate_id)] if result is not None else [],
        payload={"store": store, "subjects": list(subject_names)},
    )
    return _map_result(candidate_id, result)


async def supersede_belief(
    session: Any,
    *,
    store: str,
    supersedes_id: str,
    content: str,
    subject_names: list[str],
    confidence: float,
    source_excerpt: str | None = None,
    corpus_entry_id: str | None = None,
    uncertainty_nature: str = "EPISTEMIC",
    tags: list[str] | None = None,
    operator: bool = False,
    actor: str | None = None,
) -> AgentWriteResult:
    """Revise a belief: retire the predecessor to SUPERSEDED, then assert a successor.

    The prior particle transitions ACTIVE → SUPERSEDED with reason
    ``EXPLICIT_SUPERSESSION`` first (so the successor does not spuriously conflict
    with the claim it replaces), then the successor is asserted with
    ``supersedes`` set. By default own-beliefs-only; operator (HUMAN_REVIEW)
    particles are never agent-mutable (§6).

    ``operator=True`` takes the operator-scoped path: it may supersede
    a belief the agent does **not** own (incl. an extracted belief) and records
    the event under ``actor`` instead of the agent identity. The HUMAN_REVIEW and
    ACTIVE guards still apply, and the surface keeps the
    ``mcp.write.enabled_stores`` + bearer gate. Does not commit — the caller owns
    the transaction.
    """
    from particles.store.particle_store import update_particle_status

    identity = get_config().mcp.write.asserter_identity
    event_actor = actor or identity
    await _load_mutable_target(session, supersedes_id, identity, operator=operator)
    # Retire the prior belief first so the successor reconciles against the rest
    # of the store, not against the claim it is replacing.
    await update_particle_status(
        session, supersedes_id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
    )
    candidate_id, result = await _construct_and_insert(
        session,
        store,
        identity,
        content=content,
        subject_names=subject_names,
        confidence_value=confidence,
        uncertainty_nature=uncertainty_nature,
        source_excerpt=source_excerpt,
        corpus_entry_id=corpus_entry_id,
        tags=tags,
        supersedes=supersedes_id,
    )
    await record_event(
        session,
        actor=event_actor,
        event_type=OperatorEventType.PARTICLE_SUPERSEDED,
        refs=[
            (EventRefKind.PARTICLE, supersedes_id),
            *([(EventRefKind.PARTICLE, candidate_id)] if result is not None else []),
        ],
        payload={"store": store, "supersedes": supersedes_id, "operator": operator},
    )
    return _map_result(candidate_id, result)


async def assign_subject_belief(
    session: Any,
    *,
    store: str,
    particle_id: str,
    subject_id: str | None = None,
    subject_name: str | None = None,
    actor: str | None = None,
) -> AgentWriteResult:
    """Attach a subject to an orphan via a provenance-preserving operator-supersede.

    The subject-assign: supersede the ``NO_SUBJECT`` particle with a
    successor carrying the **same content** plus the resolved subject(s), in
    *provenance-carry-over* mode — the successor copies the predecessor's full
    ``confidence`` record, ``extractor_ref``, and source provenance (it is the
    same extracted claim, only the subject linkage is corrected), so the
    calibrated confidence and the extractor's authorship are preserved. The
    operator's act is the recorded ``PARTICLE_SUPERSEDED`` event.

    The subject is resolved from an explicit ``subject_id`` (linked directly) or
    a ``subject_name`` run through the standard resolver, so
    identity stays canonical (no ad-hoc duplicate Subjects). Operator-scoped:
    the orphan is extracted, so the own-beliefs-only agent path cannot touch it.
    Does not commit — the caller owns the transaction.
    """
    from particles.ingest.subject_resolver import resolve_subject
    from particles.store.particle_store import update_particle_status

    if (subject_id is None) == (subject_name is None):
        raise ValueError("Provide exactly one of subject_id or subject_name.")

    identity = get_config().mcp.write.asserter_identity
    event_actor = actor or identity
    target = await _load_mutable_target(session, particle_id, identity, operator=True)

    if subject_id is not None:
        from particles.store.subject_store import get_subject

        subject = await get_subject(session, subject_id)
        if subject is None:
            raise ValueError(f"Subject {subject_id!r} not found.")
        resolved_id = subject.id
    else:
        assert subject_name is not None
        if not subject_name.strip():
            raise ValueError("subject_name must be non-empty.")
        subject = await resolve_subject(
            session,
            subject_name.strip(),
            asserted_by=event_actor,
            particle_content=target.content,
        )
        resolved_id = subject.id

    # Carry the existing subjects forward too — assign *adds* the resolved subject
    # to whatever the orphan already had (which for a NO_SUBJECT card is none).
    new_subject_ids = list(dict.fromkeys([*target.subject_ids, resolved_id]))

    await update_particle_status(
        session, particle_id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
    )
    candidate_id, result = await _construct_and_insert(
        session,
        store,
        identity,
        content=target.content,
        subject_names=[],
        confidence_value=target.confidence.value,
        uncertainty_nature=target.uncertainty_nature.value,
        source_excerpt=None,
        corpus_entry_id=None,
        tags=target.tags,
        supersedes=particle_id,
        carry_over=target,
        subject_ids=new_subject_ids,
    )
    await record_event(
        session,
        actor=event_actor,
        event_type=OperatorEventType.PARTICLE_SUPERSEDED,
        refs=[
            (EventRefKind.PARTICLE, particle_id),
            *([(EventRefKind.PARTICLE, candidate_id)] if result is not None else []),
            (EventRefKind.SUBJECT, resolved_id),
        ],
        payload={
            "store": store,
            "supersedes": particle_id,
            "operator": True,
            "subject_assign": resolved_id,
        },
    )
    return _map_result(candidate_id, result)


async def retract_belief(
    session: Any,
    *,
    store: str,
    particle_id: str,
    reason: str,
    operator: bool = False,
    actor: str | None = None,
    identity: str | None = None,
) -> None:
    """Retract a belief: ACTIVE → RETRACTED with reason ``EXPLICIT_RETRACTION``.

    By default own-beliefs-only; operator (HUMAN_REVIEW) particles are never
    agent-mutable (§6). ``operator=True`` retracts a belief the agent
    does **not** own (incl. an extracted belief) and records the event under
    ``actor``; the HUMAN_REVIEW and ACTIVE guards still apply, and the surface
    keeps the ``mcp.write.enabled_stores`` + bearer gate. The free-text ``reason``
    is recorded on the audit event. Does not commit — the caller owns the
    transaction. Raises ``ValueError`` on an empty reason or a guard violation.

    Args:
        identity: Overrides the server-bound principal used for the
            own-beliefs-only ownership check, so a surface that asserts under
            its own principal (the façade) can retract what it wrote.
    """
    from particles.store.particle_store import update_particle_status

    if not reason.strip():
        raise ValueError("particle_retract requires a non-empty reason.")
    identity = identity or get_config().mcp.write.asserter_identity
    event_actor = actor or identity
    await _load_mutable_target(session, particle_id, identity, operator=operator)
    await update_particle_status(
        session, particle_id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await record_event(
        session,
        actor=event_actor,
        event_type=OperatorEventType.PARTICLE_RETRACTED,
        reason=reason,
        refs=[(EventRefKind.PARTICLE, particle_id)],
        payload={"store": store, "operator": operator},
    )


async def deposit_conversation_text(
    session: Any,
    *,
    text: str,
    tags: list[str] | None = None,
    identity: str | None = None,
) -> tuple[str, str]:
    """Deposit conversational material as a CONVERSATION corpus entry.

    The standalone cheap deposit (zero extraction) for material worth archiving
    that does not yet warrant a belief. Attributed to the server-bound asserter
    identity (both ``deposited_by`` and ``author_id``). Returns
    ``(corpus_entry_id, snapshot_id)``; does not commit — the caller owns the
    transaction.

    Args:
        identity: Overrides the server-bound principal stamped as
            ``deposited_by`` / ``author_id``, so a surface asserting under its
            own principal (the façade) attributes its deposits to it.
    """
    identity = identity or get_config().mcp.write.asserter_identity
    return await _deposit_text_seam(
        session,
        text,
        deposited_by=identity,
        source_type=SourceType.CONVERSATION,
        tags=tags,
        author_id=identity,
    )
