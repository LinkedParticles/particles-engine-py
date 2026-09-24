# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""§9.6 Review operation (Extension B: cascade enabled).

Presents INCONSISTENCY particles for human review; supports four resolution actions:
  PREFER_A  → loser (B) demoted — a quarantined B flips its reason to
              CONFLICT_RESOLVED in place; write SourceTrustStatement + REVIEW
              particle; wrapper RETRACTED (CONFLICT_RESOLVED); trigger trust
              cascade if policy gate passes
  PREFER_B  → loser (A) → PROVENANCE_STALE; a quarantined B is promoted to a new
              ACTIVE particle (Reindex pattern — fresh id, supersedes); trust
              statement + REVIEW particle; wrapper RETRACTED; cascade as above
  BOTH_VALID → both claims get uncertainty_nature=ALEATORY (a quarantined B is
              promoted with ALEATORY); INCONSISTENCY particle retracted
  DEFER      → no status change; add reviewer note; re-queue — the only action
              that leaves the wrapper open

Every non-DEFER resolution terminates its wrapper, so resolved conflicts leave
the ``list_inconsistencies`` queue (review P4-3). Cascade runs in the same
transaction as the PREFER action, after the wrapper is closed.

Pre-ADR-0117 wrappers may carry a dangling B ref (the candidate was never
persisted); they still resolve — B's demotion/promotion is skipped and the
wrapper's 120-char excerpt remains the only record of claim B.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.conflict_resolution import RETIRED_VALUE_KEY
from particles.core.schema import (
    SCHEMA_VERSION,
    Confidence,
    Particle,
    ParticleType,
    PolicyProvenance,
    ProvenanceRefType,
    ResolutionAction,
    ReviewParticle,
    SourceRef,
    SourceRefType,
    SourceTrustStatement,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.observability import traced
from particles.operations._quarantine import is_quarantined, promote_quarantined
from particles.operations.cascade import run_trust_cascade
from particles.store.event_store import EventRefKind, OperatorEventType, record_event
from particles.store.particle_store import (
    get_inconsistency_particles,
    get_particle,
    insert_particle,
    update_particle_status,
    update_status_reason,
    update_uncertainty_nature,
)
from particles.store.trust_store import insert_trust_statement

log = logging.getLogger(__name__)


async def list_inconsistencies(session: AsyncSession) -> list[Particle]:
    """Return all INCONSISTENCY particles pending review."""
    return await get_inconsistency_particles(session)


@traced("review")
async def resolve(
    session: AsyncSession,
    inconsistency_particle_id: str,
    action: ResolutionAction,
    reviewer_id: str,
    domain: str = "general",
    note: str | None = None,
    actor: str = "review",
) -> ReviewParticle:
    """Apply a resolution action to an INCONSISTENCY particle.

    Returns the REVIEW particle written as an audit record.
    The demotion-only rule is enforced: PREFER resolutions set the lower-trust
    particle to PROVENANCE_STALE rather than silently suppressing it.
    """
    # refuse to resolve in a store with mismatched-schema
    # particles. The resolution writes new ACTIVE / PROVENANCE_STALE rows
    # whose interpretation depends on the surrounding store being current.
    from particles.operations.version_guard import assert_store_schema_current

    await assert_store_schema_current(session)

    inc = await get_particle(session, inconsistency_particle_id)
    if inc is None:
        raise ValueError(f"Particle {inconsistency_particle_id} not found")
    if inc.status != Status.INCONSISTENCY:
        raise ValueError(
            f"Particle {inconsistency_particle_id} has status {inc.status!r};"
            " expected INCONSISTENCY"
        )

    # Extract the two conflicting particle IDs from the INCONSISTENCY provenance
    # Convention: first PARTICLE provenance ref = particle A, second = particle B
    particle_refs = [p for p in inc.provenance if p.type.value == "PARTICLE"]
    particle_a_id = particle_refs[0].corpus_entry_id if len(particle_refs) > 0 else None
    particle_b_id = particle_refs[1].corpus_entry_id if len(particle_refs) > 1 else None

    trust_statement_id: str | None = None
    trust_stmt: SourceTrustStatement | None = None
    # a retired-value record pairs a judgment-retired twin (A) with
    # a held re-assertion (B). PREFER_A = the retirement stands; PREFER_B =
    # lift it. Neither is a verdict on a *source*, so no trust statement is
    # written and no cascade runs — the REVIEW particle and the event are the
    # whole record.
    retired_value = bool(inc.properties and inc.properties.get(RETIRED_VALUE_KEY))

    cascade_count = 0
    promoted_ids: list[str] = []
    if action == ResolutionAction.PREFER_A:
        await _demote_loser(session, particle_b_id)
        preferred = await get_particle(session, particle_a_id) if particle_a_id else None
        if not retired_value:
            trust_statement_id, trust_stmt = await _write_trust_statement(
                session, domain, preferred, particle_b_id, reviewer_id
            )
        # Close the wrapper BEFORE the cascade runs: the cascade
        # scans open INCONSISTENCY particles in the domain, and must not
        # re-process — possibly contradicting — the resolution just made.
        await update_particle_status(
            session, inconsistency_particle_id, Status.RETRACTED, StatusReason.CONFLICT_RESOLVED
        )
        if trust_stmt is not None:
            cascade_count = await run_trust_cascade(session, trust_stmt)

    elif action == ResolutionAction.PREFER_B:
        await _demote_loser(session, particle_a_id)
        promoted = await _recover_claim_b(session, particle_b_id)
        if promoted is not None:
            promoted_ids.append(promoted.id)
        # The preferred claim is the minted ACTIVE particle when B was
        # quarantined, else B as stored (an already-ACTIVE lint-built pair).
        preferred = promoted
        if preferred is None and particle_b_id:
            preferred = await get_particle(session, particle_b_id)
        if not retired_value:
            trust_statement_id, trust_stmt = await _write_trust_statement(
                session, domain, preferred, particle_a_id, reviewer_id
            )
        await update_particle_status(
            session, inconsistency_particle_id, Status.RETRACTED, StatusReason.CONFLICT_RESOLVED
        )
        if trust_stmt is not None:
            cascade_count = await run_trust_cascade(session, trust_stmt)

    elif action == ResolutionAction.BOTH_VALID:
        # Both claims stay queryable with uncertainty_nature=ALEATORY. A
        # quarantined B is promoted to a new ACTIVE particle with
        # ALEATORY nature; a pre-0117 dangling B ref is unrecoverable (the
        # candidate's content was never stored) — update best-effort and
        # always retract the INCONSISTENCY wrapper.
        for pid in filter(None, [particle_a_id, particle_b_id]):
            p = await get_particle(session, pid)
            if p is None:
                log.debug("BOTH_VALID: particle %s not in DB (never stored); skipping", pid)
            elif is_quarantined(p):
                minted = await promote_quarantined(
                    session, p, uncertainty_nature=UncertaintyNature.ALEATORY
                )
                promoted_ids.append(minted.id)
            else:
                await update_uncertainty_nature(session, pid, UncertaintyNature.ALEATORY)
        await update_particle_status(
            session, inconsistency_particle_id, Status.RETRACTED, StatusReason.CONFLICT_RESOLVED
        )

    elif action == ResolutionAction.DEFER:
        # No status change; re-set same status (allowed by transition table).
        # DEFER is the only action that leaves the wrapper open.
        await update_particle_status(session, inconsistency_particle_id, Status.INCONSISTENCY)
        log.info("Review deferred for INCONSISTENCY particle %s", inconsistency_particle_id)

    # Write REVIEW particle as audit record
    review = ReviewParticle(
        inconsistency_particle_id=inconsistency_particle_id,
        resolution=action,
        reviewer_id=reviewer_id,
        reviewed_at=datetime.now(UTC),
        trust_statement_id=trust_statement_id,
        note=note,
    )
    await _persist_review_particle(session, review)

    refs: list[tuple[EventRefKind, str]] = [(EventRefKind.PARTICLE, inconsistency_particle_id)]
    for ref_pid in (particle_a_id, particle_b_id, *promoted_ids):
        if ref_pid:
            refs.append((EventRefKind.PARTICLE, ref_pid))
    if trust_statement_id:
        refs.append((EventRefKind.TRUST_STATEMENT, trust_statement_id))
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.REVIEW_RESOLVED,
        reason=note,
        refs=refs,
        payload={
            "action": action.value,
            "reviewer_id": reviewer_id,
            "cascade_resolved": cascade_count,
            "trust_statement_id": trust_statement_id,
            "promoted_particle_ids": promoted_ids,
        },
    )
    await session.commit()

    log.info(
        "Review %s: action=%s for INCONSISTENCY %s (cascade resolved %d)",
        review.review_id,
        action.value,
        inconsistency_particle_id,
        cascade_count,
    )
    return review


async def _demote_loser(session: AsyncSession, loser_id: str | None) -> None:
    """Demote the non-preferred particle; tolerate a legacy dangling ref.

    A quarantined loser is already PROVENANCE_STALE — its reason
    flips ``CONFLICT_PENDING → CONFLICT_RESOLVED`` with no status transition.
    A pre-0117 wrapper's B ref may point at a never-persisted UUID; the
    resolution proceeds anyway (the ADR's Consequences keep legacy wrappers
    resolvable with excerpt-only display for B).
    """
    if loser_id is None:
        return
    loser = await get_particle(session, loser_id)
    if loser is None:
        log.debug("Loser %s not in DB (pre-ADR-0117 wrapper); skipping demotion", loser_id)
        return
    if loser.status in (Status.RETRACTED, Status.SUPERSEDED):
        # the "loser" is a judgment-retired twin — already off the
        # surface by an earlier judgment. There is no legal transition out of
        # a terminal state and nothing further to demote.
        log.debug("Loser %s is already %s; nothing to demote", loser_id, loser.status.value)
        return
    if loser.status is Status.PROVENANCE_STALE:
        if loser.status_reason is StatusReason.CONFLICT_PENDING:
            await update_status_reason(session, loser_id, StatusReason.CONFLICT_RESOLVED)
        return
    await update_particle_status(
        session, loser_id, Status.PROVENANCE_STALE, StatusReason.CONFLICT_RESOLVED
    )


async def _recover_claim_b(session: AsyncSession, particle_b_id: str | None) -> Particle | None:
    """PREFER_B: recover claim B as a queryable ACTIVE particle.

    A quarantined B is promoted to a new ACTIVE particle (Reindex pattern); an
    already-ACTIVE B (e.g. a lint-built wrapper between two live particles)
    needs nothing; a pre-0117 dangling B is unrecoverable — its content was
    never persisted — so only the wrapper excerpt survives.

    Returns the minted particle, or ``None`` when no promotion happened.
    """
    if particle_b_id is None:
        return None
    pb = await get_particle(session, particle_b_id)
    if pb is None:
        log.warning(
            "PREFER_B: particle %s not in DB (pre-ADR-0117 wrapper); claim B is"
            " unrecoverable beyond the wrapper excerpt",
            particle_b_id,
        )
        return None
    if is_quarantined(pb):
        return await promote_quarantined(session, pb)
    return None


async def _write_trust_statement(
    session: AsyncSession,
    domain: str,
    preferred: Particle | None,
    demoted_id: str | None,
    reviewer_id: str,
) -> tuple[str | None, SourceTrustStatement | None]:
    """Write the SourceTrustStatement encoding a PREFER judgment.

    The statement is keyed on the **corpus entry of the preferred claim's
    SOURCE provenance** — ``SourceRef(CORPUS_ENTRY, <entry_id>)`` — because
    that is the key every consumer looks up: the §6.4 layered cascade
    (``get_layered_trust_rank``, consulted by the ingest ladder and the
    query path) and the reviewer-confirmation gate
    (``count_reviewer_confirmations``) both match this column
    against a corpus-entry id. Keying it on the particle id instead — which is
    what this function did until 1.141.14 — produced a statement nothing could
    ever find.

    Returns ``(None, None)``, writing nothing, when there is no source to
    prefer: the preferred particle is unknown (a pre-ADR-0117 dangling ref),
    is not ACTIVE (a retired claim is not a source recommendation), or carries
    no SOURCE provenance (an agent-asserted or derived claim has no corpus
    entry a trust statement could name). A review without a statement is still
    a complete resolution — the wrapper closes and the REVIEW particle records
    the judgment — it simply drives no cascade.
    """
    if preferred is None:
        log.debug("Review: preferred particle unknown; no trust statement written")
        return None, None
    if preferred.status is not Status.ACTIVE:
        log.debug(
            "Review: preferred particle %s is %s, not ACTIVE; no trust statement written",
            preferred.id[:8],
            preferred.status.value,
        )
        return None, None
    source_ref = next(
        (ref for ref in preferred.provenance if ref.type is ProvenanceRefType.SOURCE),
        None,
    )
    if source_ref is None:
        log.debug(
            "Review: preferred particle %s has no SOURCE provenance; no trust statement written",
            preferred.id[:8],
        )
        return None, None
    stmt = SourceTrustStatement(
        domain=domain,
        source_ref=SourceRef(
            type=SourceRefType.CORPUS_ENTRY,
            value=source_ref.corpus_entry_id,
        ),
        trust_rank=get_config().trust.reviewer_trust_rank,  # reviewer-derived preference
        policy_provenance=PolicyProvenance.REVIEWER_DERIVED,
        asserted_by=reviewer_id,
        basis=(
            f"source of particle {preferred.id} preferred over particle "
            f"{demoted_id or 'unknown'} in conflict resolution"
        ),
    )
    await insert_trust_statement(session, stmt)
    return stmt.statement_id, stmt


async def _persist_review_particle(session: AsyncSession, review: ReviewParticle) -> None:
    """Store the REVIEW particle as a Particle record for audit trail."""
    content = (
        f"REVIEW: {review.resolution.value} on INCONSISTENCY {review.inconsistency_particle_id}. "
        f"Reviewer: {review.reviewer_id}." + (f" Note: {review.note}" if review.note else "")
    )
    particle = Particle(
        id=review.review_id,
        content=content,
        particle_type=ParticleType.REVIEW,
        confidence=Confidence(value=1.0, calibration_source=CalibrationSource.HUMAN_REVIEW),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[],
        asserted_by=review.reviewer_id,
        asserted_at=review.reviewed_at,
        status=Status.ACTIVE,
        schema_version=SCHEMA_VERSION,
    )
    from particles.core.status import validate_transition

    validate_transition(None, Status.ACTIVE)
    await insert_particle(session, particle)
