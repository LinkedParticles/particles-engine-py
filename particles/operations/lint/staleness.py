"""Staleness / provenance-stale detectors and confidence-decay reporter.

Covers the structural checks that flip ACTIVE particles to PROVENANCE_STALE
when a temporal or provenance condition has expired:

  - ``_check_staleness`` — ``valid_until`` has passed.
  - ``_check_retraction_propagation`` — a particle the provenance DAG depends
    on has been RETRACTED or SUPERSEDED.
  - ``_check_corpus_link_integrity`` — referenced ``snapshot_id`` no longer
    exists in the corpus.
  - ``_check_confidence_decay`` — EPISTEMIC particle whose ``confidence.variance``
    has grown past the alert threshold (read-only; no status change).
  - ``_check_recency_staleness`` — ACTIVE particle whose ``effective_confidence``
    is materially discounted by content age (read-only).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import LintFinding, ProvenanceRefType
from particles.core.status import Status, StatusReason
from particles.corpus.store import CorpusEntryRow, get_snapshot
from particles.operations.abstraction import is_derived
from particles.operations.query.decay_policy import load_decay_policy
from particles.store.particle_store import (
    get_active_epistemic_particles_with_variance,
    get_active_particles,
    get_active_particles_with_valid_until,
    get_particle,
    update_particle_status,
)


async def _check_staleness(session: AsyncSession, fix: bool) -> list[LintFinding]:
    """Flag ACTIVE particles whose valid_until has passed."""
    now = datetime.now(UTC)
    findings: list[LintFinding] = []
    for p in await get_active_particles_with_valid_until(session):
        vu = p.valid_until
        if vu is None:
            continue
        if vu.tzinfo is None:
            vu = vu.replace(tzinfo=UTC)
        if vu < now:
            if fix:
                await update_particle_status(
                    session, p.id, Status.PROVENANCE_STALE, StatusReason.VALIDITY_EXPIRED
                )
            findings.append(
                LintFinding(
                    particle_id=p.id,
                    particle_content=p.content,
                    finding_type="STALENESS",
                    severity="ERROR",
                    detail=f"valid_until {p.valid_until} has passed",
                    recommended_action="Set PROVENANCE_STALE",
                )
            )
    return findings


async def _check_retraction_propagation(session: AsyncSession, fix: bool) -> list[LintFinding]:
    """Find ACTIVE particles whose provenance chain includes RETRACTED/SUPERSEDED particles.

    Deliberately **one-hop** (each particle's own refs only) — for the consolidation DAG this is the memoization boundary: when a premise changes,
    only its direct dependents are flagged; further propagation happens only if
    revalidation actually supersedes a level (§5).

    Derived particles (``calibration_source == DERIVED``) get a
    ``DERIVED_REVALIDATION`` finding instead of ``RETRACTION_CASCADE``, and the
    ``--fix`` transition to PROVENANCE_STALE is **not** applied: the keep-ACTIVE-and-discount contract keeps a still-plausible abstraction
    visible (its effective confidence discounted at read time) until the dream
    cycle's revalidation ladder repairs or retires it.
    """
    findings: list[LintFinding] = []
    for p in await get_active_particles(session):
        for ref in p.provenance:
            if ref.type != ProvenanceRefType.PARTICLE:
                continue
            dep = await get_particle(session, ref.corpus_entry_id)
            if dep is None:
                continue
            if dep.status in (Status.RETRACTED, Status.SUPERSEDED):
                if is_derived(p):
                    findings.append(
                        LintFinding(
                            particle_id=p.id,
                            particle_content=p.content,
                            finding_type="DERIVED_REVALIDATION",
                            severity="WARNING",
                            detail=(
                                f"Premise {ref.corpus_entry_id} has status "
                                f"{dep.status.value}; effective confidence is "
                                "discounted until revalidation"
                            ),
                            recommended_action=(
                                "Run `particles memory consolidate` (the "
                                "revalidation ladder repairs or retires it)"
                            ),
                        )
                    )
                    break
                if fix:
                    await update_particle_status(
                        session, p.id, Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY
                    )
                findings.append(
                    LintFinding(
                        particle_id=p.id,
                        particle_content=p.content,
                        finding_type="RETRACTION_CASCADE",
                        severity="ERROR",
                        detail=(
                            f"Provenance dependency {ref.corpus_entry_id} "
                            f"has status {dep.status.value}"
                        ),
                        recommended_action="Set PROVENANCE_STALE (RETRACTED_DEPENDENCY)",
                    )
                )
                break
    return findings


async def _check_corpus_link_integrity(session: AsyncSession, fix: bool) -> list[LintFinding]:
    """Find ACTIVE particles pointing to snapshots that no longer exist."""
    findings: list[LintFinding] = []
    for p in await get_active_particles(session):
        for ref in p.provenance:
            if not ref.snapshot_id:
                continue
            if await get_snapshot(session, ref.snapshot_id) is None:
                if fix:
                    await update_particle_status(
                        session, p.id, Status.PROVENANCE_STALE, StatusReason.CORPUS_ENTRY_MISSING
                    )
                findings.append(
                    LintFinding(
                        particle_id=p.id,
                        particle_content=p.content,
                        finding_type="CORPUS_LINK_INTEGRITY",
                        severity="ERROR",
                        detail=f"Referenced snapshot {ref.snapshot_id} does not exist",
                        recommended_action="Set PROVENANCE_STALE (CORPUS_ENTRY_MISSING)",
                    )
                )
                break
    return findings


async def _check_recency_staleness(session: AsyncSession) -> list[LintFinding]:
    """Flag ACTIVE particles whose effective_confidence is materially reduced by
    content age alone (decay; surfaced in lint).

    Read-only (no status change), like ``_check_confidence_decay``. Age decay is
    a continuous, recoverable discount — re-fetching a newer snapshot restores
    it — not a provenance break, so a decay threshold must not flip status the
    way ``valid_until`` expiry does.

    For each ACTIVE particle, resolve its source provenance — the snapshot
    carries ``content_published_at`` and the corpus entry carries the
    ``source_type`` the decay curve is keyed on — compute the ``recency_factor``, and flag when ``1 - recency_factor`` reaches
    ``lint.recency_decay_threshold``. Particles whose source type has no decay
    config (``recency_factor == 1.0``) or that carry no publication date never
    fire.
    """
    threshold = get_config().lint.recency_decay_threshold
    # resolve decay through the store's composed policy (local config +
    # adopted-lens decay_rules), so a lens that speeds decay also surfaces the
    # corresponding staleness findings. Identical to the global config when no
    # decay-bearing lens is adopted.
    decay_policy = await load_decay_policy(session)
    findings: list[LintFinding] = []
    for p in await get_active_particles(session):
        for ref in p.provenance:
            if not ref.snapshot_id or not ref.corpus_entry_id:
                continue
            snapshot = await get_snapshot(session, ref.snapshot_id)
            if snapshot is None or snapshot.content_published_at is None:
                continue
            entry = await session.get(CorpusEntryRow, ref.corpus_entry_id)
            if entry is None:
                continue
            rf = decay_policy.recency_factor(
                snapshot.content_published_at, entry.source_type, entry.uri_r
            )
            discount = 1.0 - rf
            if discount >= threshold:
                findings.append(
                    LintFinding(
                        particle_id=p.id,
                        particle_content=p.content,
                        finding_type="RECENCY_DECAY",
                        severity="WARNING",
                        detail=(
                            f"{entry.source_type} content from "
                            f"{snapshot.content_published_at.date().isoformat()} — "
                            f"recency_factor={rf:.2f} discounts effective_confidence by "
                            f"{discount:.0%} due to age alone"
                        ),
                        recommended_action=(
                            "Re-fetch / re-index if a fresher source version exists"
                        ),
                    )
                )
            break  # one finding per particle, from its first source ref
    return findings


async def _check_confidence_decay(session: AsyncSession) -> list[LintFinding]:
    """Flag EPISTEMIC particles with high confidence variance."""
    threshold = get_config().lint.variance_threshold
    findings: list[LintFinding] = []
    for p in await get_active_epistemic_particles_with_variance(session):
        variance = p.confidence.variance
        if variance is not None and variance > threshold:
            findings.append(
                LintFinding(
                    particle_id=p.id,
                    particle_content=p.content,
                    finding_type="CONFIDENCE_DECAY",
                    severity="WARNING",
                    detail=(
                        f"EPISTEMIC particle has confidence.variance={variance:.3f} > {threshold}"
                    ),
                    recommended_action="Re-evaluate confidence; consider re-extraction",
                )
            )
    return findings
