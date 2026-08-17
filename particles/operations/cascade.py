"""Extension B: source trust cascade.

When a SourceTrustStatement is written, run_trust_cascade() auto-resolves open
INCONSISTENCY particles whose domain_hint matches the statement's domain and whose
constituent particles' sources fall under the statement's authority.

Policy gate:
  OPERATOR_DIRECT / REGISTRY_ENDORSED  → always cascade
  REVIEWER_DERIVED                     → only when N >= TRUST_CASCADE_MIN_REVIEWER_CONFIRMATIONS
                                         distinct statements exist for the same source_ref in domain
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.cascade_gate import apply_cascade_cap, cascade_gate_passes
from particles.core.schema import (
    SCHEMA_VERSION,
    Confidence,
    Particle,
    ParticleType,
    PolicyProvenance,
    ProvenanceRefType,
    SourceTrustStatement,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason, validate_transition
from particles.operations._quarantine import is_quarantined, promote_quarantined
from particles.store.particle_store import (
    get_inconsistency_particles_by_domain,
    get_particle,
    insert_particle,
    update_particle_status,
    update_status_reason,
)
from particles.store.trust_store import (
    count_reviewer_confirmations,
    get_layered_trust_rank,
)

log = logging.getLogger(__name__)


async def run_trust_cascade(
    session: AsyncSession,
    statement: SourceTrustStatement,
    max_cascade: int | None = None,
) -> int:
    """Auto-resolve open INCONSISTENCY particles matching a new SourceTrustStatement.

    Returns the count of INCONSISTENCY particles resolved (loser demoted).
    Called from review.py and cli.py within the same DB transaction as the trigger.
    """
    cfg = get_config().trust
    if max_cascade is None:
        max_cascade = cfg.cascade_max_per_run

    if not await _gate_passes(session, statement, cfg.cascade_min_reviewer_confirmations):
        log.debug(
            "Cascade gate blocked for statement %s (policy=%s)",
            statement.statement_id,
            statement.policy_provenance,
        )
        return 0

    particles = await get_inconsistency_particles_by_domain(
        session, statement.domain, limit=max_cascade + 1
    )
    processed, capped = apply_cascade_cap(len(particles), max_cascade)
    if capped:
        particles = particles[:processed]
        log.warning(
            "Cascade for statement %s capped at %d (TRUST_CASCADE_MAX_PER_RUN); "
            "remaining INCONSISTENCY particles left for manual review",
            statement.statement_id,
            max_cascade,
        )

    resolved = 0
    for inconsistency in particles:
        did_resolve = await _try_resolve_inconsistency(session, inconsistency, statement)
        if did_resolve:
            resolved += 1

    if resolved:
        log.info(
            "Trust cascade for statement %s resolved %d INCONSISTENCY particle(s) in domain %r",
            statement.statement_id,
            resolved,
            statement.domain,
        )
    return resolved


async def _gate_passes(
    session: AsyncSession,
    statement: SourceTrustStatement,
    min_reviewer_confirmations: int | None = None,
) -> bool:
    """Return True if this statement's policy_provenance allows auto-cascade.

    The I/O half: resolve the reviewer-confirmation count, then delegate the
    §15.1 decision to :func:`particles.core.cascade_gate.cascade_gate_passes`
    (the pure predicate the L2 vectors pin).
    """
    if min_reviewer_confirmations is None:
        min_reviewer_confirmations = get_config().trust.cascade_min_reviewer_confirmations
    confirmations = 0
    if statement.policy_provenance == PolicyProvenance.REVIEWER_DERIVED:
        confirmations = await count_reviewer_confirmations(
            session,
            statement.domain,
            statement.source_ref.type.value,
            statement.source_ref.value,
        )
    return cascade_gate_passes(
        statement.policy_provenance,
        reviewer_confirmations=confirmations,
        min_reviewer_confirmations=min_reviewer_confirmations,
    )


async def _try_resolve_inconsistency(
    session: AsyncSession,
    inconsistency: Particle,
    statement: SourceTrustStatement,
) -> bool:
    """Attempt to auto-resolve one INCONSISTENCY particle. Returns True if resolved."""
    particle_refs = [r for r in inconsistency.provenance if r.type == ProvenanceRefType.PARTICLE]
    if len(particle_refs) < 2:
        return False

    particle_a_id = particle_refs[0].corpus_entry_id
    particle_b_id = particle_refs[1].corpus_entry_id

    particle_a = await get_particle(session, particle_a_id)
    particle_b = await get_particle(session, particle_b_id)

    # Genuinely exceptional since (the INCONSISTENT-verdict loser is
    # persisted quarantined, so both constituents normally exist): only
    # pre-0117 wrappers with a dangling B ref land here. They stay open for
    # manual review (PREFER_A / DEFER still work over them).
    if particle_a is None or particle_b is None:
        return False

    # Resolve trust ranks for both constituent particles
    rank_a = await _particle_trust_rank(session, particle_a, statement.domain)
    rank_b = await _particle_trust_rank(session, particle_b, statement.domain)

    if rank_a is None or rank_b is None:
        return False

    diff = rank_a - rank_b
    if abs(diff) < get_config().trust.differential_threshold:
        return False

    if diff > 0:
        winner_id, loser_id = particle_a_id, particle_b_id
        winner, loser = particle_a, particle_b
    else:
        winner_id, loser_id = particle_b_id, particle_a_id
        winner, loser = particle_b, particle_a

    # Demote loser. A quarantined loser is already
    # PROVENANCE_STALE — flip its reason in place instead of re-transitioning.
    if loser.status is Status.PROVENANCE_STALE:
        if loser.status_reason is StatusReason.CONFLICT_PENDING:
            await update_status_reason(session, loser_id, StatusReason.CONFLICT_RESOLVED)
    else:
        await update_particle_status(
            session, loser_id, Status.PROVENANCE_STALE, StatusReason.CONFLICT_RESOLVED
        )
    # Promote a quarantined winner: a cascade resolving in favour
    # of the quarantined candidate mints the new ACTIVE particle exactly as a
    # PREFER_B review would.
    if is_quarantined(winner):
        await promote_quarantined(session, winner)
    # Mark INCONSISTENCY resolved
    await update_particle_status(
        session, inconsistency.id, Status.PROVENANCE_STALE, StatusReason.CONFLICT_RESOLVED
    )
    # Write system REVIEW particle
    await _write_cascade_review(
        session,
        inconsistency_id=inconsistency.id,
        winner_id=winner_id,
        loser_id=loser_id,
        statement_id=statement.statement_id,
    )
    log.debug(
        "Cascade resolved INCONSISTENCY %s: winner=%s loser=%s (ranks %.2f vs %.2f)",
        inconsistency.id[:8],
        winner_id[:8],
        loser_id[:8],
        rank_a,
        rank_b,
    )
    return True


async def _particle_trust_rank(
    session: AsyncSession, particle: Particle, domain: str
) -> float | None:
    """Return the trust rank for a particle's source within a domain.

    Resolves the source corpus entry and calls the layered lookup.
    Returns None if the source entry cannot be determined.
    """
    from particles.corpus.store import get_entry

    source_ref = next(
        (ref for ref in particle.provenance if ref.type == ProvenanceRefType.SOURCE),
        None,
    )
    if source_ref is None:
        return None

    entry = await get_entry(session, source_ref.corpus_entry_id)
    if entry is None:
        return None

    # §6.4 AUTHOR tier input — the particle's author_id from its SOURCE
    # snapshot, read off the entry already fetched above.
    author_id: str | None = None
    if source_ref.snapshot_id:
        snap = next((s for s in entry.snapshots if s.snapshot_id == source_ref.snapshot_id), None)
        author_id = snap.author_id if snap else None

    return await get_layered_trust_rank(
        session,
        domain,
        entry.entry_id,
        entry.source_type,
        entry.uri_r,
        author_id,
    )


async def _write_cascade_review(
    session: AsyncSession,
    inconsistency_id: str,
    winner_id: str,
    loser_id: str,
    statement_id: str,
) -> None:
    """Write a system-generated REVIEW particle recording the cascade auto-resolution."""
    content = (
        f"CASCADE REVIEW: auto-resolved INCONSISTENCY {inconsistency_id}. "
        f"Winner: {winner_id}. Loser: {loser_id}. "
        f"Triggered by trust statement {statement_id}."
    )
    review_particle = Particle(
        id=str(uuid.uuid4()),
        content=content,
        particle_type=ParticleType.REVIEW,
        confidence=Confidence(value=1.0, calibration_source=CalibrationSource.HUMAN_REVIEW),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[],
        asserted_by="trust-cascade",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        schema_version=SCHEMA_VERSION,
    )
    validate_transition(None, Status.ACTIVE)
    await insert_particle(session, review_particle)
