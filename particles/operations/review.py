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
from particles.core.conflict_review import TrustJudgment, conflict_pair_ids, decide_resolution
from particles.core.schema import (
    SCHEMA_VERSION,
    Confidence,
    Particle,
    ParticleType,
    PolicyProvenance,
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
from particles.operations._quarantine import apply_demotion, promote_quarantined
from particles.operations.cascade import run_trust_cascade
from particles.store.event_store import EventRefKind, OperatorEventType, record_event
from particles.store.particle_store import (
    get_inconsistency_particles,
    get_particle,
    insert_particle,
    update_particle_status,
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

    # Gather: the two claims the wrapper names. The first PARTICLE provenance
    # ref is A and the second is B; either may dangle on a pre-ADR-0117 wrapper.
    particle_a_id, particle_b_id = conflict_pair_ids(inc)
    particle_a = await get_particle(session, particle_a_id) if particle_a_id else None
    particle_b = await get_particle(session, particle_b_id) if particle_b_id else None

    # Decide (D2): every write below is chosen here, store-free.
    plan = decide_resolution(action, inc, particle_a, particle_b)

    # Apply, in the order the plan documents.
    if plan.demote is not None:
        loser, demotion = plan.demote
        await apply_demotion(session, loser.id, demotion)
    promoted_ids: list[str] = []
    minted: Particle | None = None
    if plan.promote is not None:
        minted = await promote_quarantined(session, plan.promote)
        promoted_ids.append(minted.id)
    elif action == ResolutionAction.PREFER_B and particle_b_id and particle_b is None:
        log.warning(
            "PREFER_B: particle %s not in DB (pre-ADR-0117 wrapper); claim B is"
            " unrecoverable beyond the wrapper excerpt",
            particle_b_id,
        )
    for mark in plan.aleatory:
        if mark.mint:
            aleatory = await promote_quarantined(
                session, mark.particle, uncertainty_nature=UncertaintyNature.ALEATORY
            )
            promoted_ids.append(aleatory.id)
        else:
            await update_uncertainty_nature(session, mark.particle.id, UncertaintyNature.ALEATORY)

    trust_stmt: SourceTrustStatement | None = None
    if plan.trust is not None:
        # A promoted claim is preferred as the minted ACTIVE particle.
        preferred_id = minted.id if minted is not None else plan.trust.preferred_id
        trust_stmt = await _write_trust_statement(
            session, domain, plan.trust, preferred_id, reviewer_id
        )
    trust_statement_id = trust_stmt.statement_id if trust_stmt is not None else None

    cascade_count = 0
    if plan.close_wrapper:
        # Close the wrapper BEFORE the cascade runs: the cascade
        # scans open INCONSISTENCY particles in the domain, and must not
        # re-process — possibly contradicting — the resolution just made.
        await update_particle_status(
            session, inconsistency_particle_id, Status.RETRACTED, StatusReason.CONFLICT_RESOLVED
        )
        if trust_stmt is not None:
            cascade_count = await run_trust_cascade(session, trust_stmt)
    else:
        # DEFER: no status change; re-set same status (allowed by transition
        # table). The only action that leaves the wrapper open.
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


async def _write_trust_statement(
    session: AsyncSession,
    domain: str,
    trust: TrustJudgment,
    preferred_id: str,
    reviewer_id: str,
) -> SourceTrustStatement:
    """Write the SourceTrustStatement encoding a PREFER judgment.

    The statement is keyed on ``trust.source_entry_id``, the corpus entry of
    the preferred claim's SOURCE provenance, which
    :func:`particles.core.conflict_review.trust_statement_source` chose.
    When the plan carries no judgment nothing is written: a review
    without a statement is still a complete resolution, it simply drives no
    cascade.
    """
    stmt = SourceTrustStatement(
        domain=domain,
        source_ref=SourceRef(
            type=SourceRefType.CORPUS_ENTRY,
            value=trust.source_entry_id,
        ),
        trust_rank=get_config().trust.reviewer_trust_rank,  # reviewer-derived preference
        policy_provenance=PolicyProvenance.REVIEWER_DERIVED,
        asserted_by=reviewer_id,
        basis=(
            f"source of particle {preferred_id} preferred over particle "
            f"{trust.demoted_id or 'unknown'} in conflict resolution"
        ),
    )
    await insert_trust_statement(session, stmt)
    return stmt


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
