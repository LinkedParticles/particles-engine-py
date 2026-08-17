"""Coverage and extraction-quality reporters.

Surface gaps in the particle/subject graph that don't fit the staleness or
contradiction model:

  - ``_check_orphans`` — ACTIVE CLAIM particles with zero provenance.
  - ``_check_no_subject_claims`` — ACTIVE CLAIM particles with zero subjects
    (§6.7: a claim SHOULD be about at least one subject).
  - ``_check_phantom_subjects`` — Subjects with zero (PHANTOM) or below-threshold
    (LOW_COVERAGE) ACTIVE CLAIM particles.
  - ``_report_extraction_quality`` — distribution of ``calibration_source``;
    warns when EXTRACTOR_DIRECT dominates (>50%).
  - ``_report_pending_extractions`` — corpus entries stuck in PENDING/FAILED.
  - ``_report_schema_versions`` — ACTIVE particles on a non-current schema.
  - ``_check_structured_claim_subjects`` — a structured claim whose subject is
    not one of the particle's subjects (L-STR-11).
  - ``_check_bare_properties_keys`` — a persisted particle carrying a
    ``properties`` key with no ``prefix:`` (L-STR-12).
"""

from __future__ import annotations

import json

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    SCHEMA_VERSION,
    ExtractionStatus,
    LintFinding,
    ParticleType,
)
from particles.corpus.store import (
    list_complete_response_snapshots,
    list_entry_status_pairs_with_extraction_status,
)
from particles.extraction.property_keys import bare_properties_keys
from particles.extraction.subject_scope import subject_expected
from particles.store.particle_store import (
    count_active_particles_by_calibration_source,
    count_active_particles_by_schema_version,
    get_active_particles,
    get_snapshot_ids_with_particles,
)
from particles.store.subject_store import (
    get_low_coverage_subjects,
    get_phantom_subjects,
)


async def _check_orphans(session: AsyncSession) -> list[LintFinding]:
    """Flag ACTIVE particles with no provenance at all (completely disconnected).

    Normal extracted claims are leaves in the provenance DAG: they point outward
    to corpus entries (SOURCE provenance) but nothing points inward to them.
    That is expected. Only flag particles with zero provenance entries, which
    indicates they were created without any corpus anchor and are genuinely adrift.
    """
    findings: list[LintFinding] = []
    for p in await get_active_particles(session):
        if not p.provenance and p.particle_type == ParticleType.CLAIM:
            findings.append(
                LintFinding(
                    particle_id=p.id,
                    particle_content=p.content,
                    finding_type="ORPHAN",
                    severity="WARNING",
                    detail="ACTIVE particle has no provenance — not anchored to any corpus entry",
                    recommended_action="Verify particle origin or retract if spurious",
                )
            )
    return findings


async def _check_no_subject_claims(session: AsyncSession) -> list[LintFinding]:
    """Flag ACTIVE CLAIM particles whose ``subject_ids`` is empty (L-STR-09).

    §6.7: a claim SHOULD be about at least one subject — zero-subject claims
    are unreachable by subject-filtered query and subject-graph traversal.
    Legitimate zero-subject records are excluded by ``subject_expected`` — the
    §9 table made executable, and since shared with the conformance
    ``subject_ids`` floor so the two surfaces cannot answer differently:
    non-CLAIM particle types (REVIEW audit records carry no subjects by design),
    DOCUMENT_META claims (scoped to the document, not a subject),
    non-asserted claims, and author-scoped journal claims whose only
    subject is one the extractor withholds. Flagging that last class
    would turn a state the system deliberately chose into a recurring error
    report — the same argument by which ``L-STR-11`` leaves a ``null``
    ``subject_id`` alone.

    An import or extraction that could not resolve any subject is **not**
    excluded: it lands here rather than being rejected at the persistence seam,
    and it is the gap this rule exists to surface.
    """
    findings: list[LintFinding] = []
    for p in await get_active_particles(session):
        if not p.subject_ids and subject_expected(p.particle_type, p.properties):
            findings.append(
                LintFinding(
                    particle_id=p.id,
                    particle_content=p.content,
                    finding_type="NO_SUBJECT",
                    severity="WARNING",
                    detail=(
                        "ACTIVE CLAIM particle has no subjects — unreachable by "
                        "subject-filtered query (§6.7: SHOULD have ≥ 1 subject)"
                    ),
                    recommended_action=(
                        "Link the particle to a subject (re-extract, or resolve its "
                        "subject manually) or retract if spurious"
                    ),
                )
            )
    return findings


async def _check_structured_claim_subjects(session: AsyncSession) -> list[LintFinding]:
    """Flag a structured claim whose subject is not one of the particle's (L-STR-11).

    The cheapest available signal that the structurizer hallucinated a subject:
    the triple makes a statement about an entity the particle is not about
    . Structural, no LLM.

    A ``None`` ``subject_id`` is deliberately **not** flagged — it records "the
    subject term resolved to no Subject", which is the honest state for a claim
    about something the store has no Subject for. Flagging it would turn a
    coverage gap into a recurring error report.

    The remedy is always regeneration, never a change to the claim: the
    annotation is derived, the claim is asserted.
    """
    findings: list[LintFinding] = []
    for p in await get_active_particles(session):
        claim = p.structured_claim
        if claim is None or claim.subject_id is None:
            continue
        if claim.subject_id in p.subject_ids:
            continue
        findings.append(
            LintFinding(
                particle_id=p.id,
                particle_content=p.content,
                finding_type="STRUCTURED_CLAIM_SUBJECT_MISMATCH",
                severity="WARNING",
                detail=(
                    f"Structured claim's subject {claim.subject_id[:8]}… "
                    f"({claim.subject.value!r}) is not among the particle's subjects — "
                    f"the triple is about an entity the claim is not "
                    f"(structurizer {claim.structurizer_id}@{claim.structurizer_version})"
                ),
                recommended_action=(
                    "Regenerate the annotation (`particles structure "
                    "--structurizer-version <stamped version>`); the claim itself "
                    "needs no change"
                ),
            )
        )
    return findings


async def _check_bare_properties_keys(session: AsyncSession) -> list[LintFinding]:
    """Flag ACTIVE particles carrying a ``properties`` key with no prefix (L-STR-12).

    The spec requires every key to be ``prefix:LocalName`` so a consumer can
    determine its provenance and applicability from the key alone. Until
    the rule was only ever asked of *fresh extractor output*, inside a
    ``particles extractor conform`` run over a fixture — so nothing asked it of a
    store. That gap is why the bare ``polarity`` / ``scope`` keys survived from
    their introduction to 1.109.0 unnoticed
    , and it stays
    open for the three routes that reach a store without a conform run: an
    interchange import, a third-party extractor, and a store predating a
    convention change.

    **Advisory, and read-only.** Severity WARNING, no ``fix`` arm: the remedy is
    never a status transition, and it is not uniform either. For a key this SDK
    once emitted it is a data migration (Alembic 035 rewrote ``polarity`` /
    ``scope`` in place); for a third-party key it is a decision for whoever owns
    the producer. Neither is lint's to make, and the claim itself is unaffected
    — a mis-keyed property does not make a belief false.
    """
    findings: list[LintFinding] = []
    for p in await get_active_particles(session):
        bare = bare_properties_keys(p.properties)
        if not bare:
            continue
        findings.append(
            LintFinding(
                particle_id=p.id,
                particle_content=p.content,
                finding_type="BARE_PROPERTIES_KEY",
                severity="WARNING",
                detail=(
                    f"`properties` key(s) {', '.join(repr(k) for k in bare)} carry no "
                    f"prefix — the spec requires `prefix:LocalName`, so a consumer "
                    f"cannot attribute them to any namespace"
                ),
                recommended_action=(
                    "Re-extract the particle (`particles reindex --extractor-version "
                    "<old>`) if a shipped extractor produced it, or migrate the key "
                    "to its registered prefix; check the registry for "
                    "the prefix that applies"
                ),
            )
        )
    return findings


async def _check_phantom_subjects(
    session: AsyncSession, low_coverage_threshold: int
) -> list[LintFinding]:
    """Flag subjects with zero or few ACTIVE CLAIM particles.

    PHANTOM_SUBJECT (WARNING): zero ACTIVE CLAIM particles — subject was created
        by the resolver but no claims about it were ever extracted.
    LOW_COVERAGE_SUBJECT (INFO): 1 to low_coverage_threshold-1 ACTIVE CLAIM particles —
        subject exists but has sparse coverage.
    """
    findings: list[LintFinding] = []
    for subject in await get_phantom_subjects(session):
        findings.append(
            LintFinding(
                subject_id=subject.id,
                finding_type="PHANTOM_SUBJECT",
                severity="WARNING",
                detail=(
                    f"Subject {subject.canonical_name!r} ({subject.id[:8]}…) "
                    "has no ACTIVE CLAIM particles"
                ),
                recommended_action="Extract claims about this subject or merge/delete if spurious",
            )
        )

    # Restrict LOW_COVERAGE_SUBJECT to canonical (externally-identified) subjects.
    # Author-name / handle subjects (u/foo, github:bar) are expected to be sparse
    # and produced ~22k noise findings before this filter.
    for subject, cnt in await get_low_coverage_subjects(
        session, low_coverage_threshold, canonical_only=True
    ):
        findings.append(
            LintFinding(
                subject_id=subject.id,
                finding_type="LOW_COVERAGE_SUBJECT",
                severity="INFO",
                detail=(
                    f"Subject {subject.canonical_name!r} ({subject.id[:8]}…) has only {cnt} "
                    f"ACTIVE CLAIM particle(s) (threshold: {low_coverage_threshold}); "
                    "subject is canonical (has external_ids)"
                ),
                recommended_action="Deposit and extract additional sources about this subject",
            )
        )

    return findings


async def _report_extraction_quality(session: AsyncSession) -> list[LintFinding]:
    """Report calibration_source distribution; alert if EXTRACTOR_DIRECT fraction > 50%.

    Extractor-only by design: the >50% alert and its "run calibration benchmarks"
    remediation target extractor output (metadata theater, Risk #11). Uncalibrated
    agent self-reports (AGENT_ASSERTED) are surfaced via their own
    distribution bucket and the COMPOUND_ASSERTION lint, not this alert —
    benchmarks recalibrate extractors, not self-reports.
    """
    dist = await count_active_particles_by_calibration_source(session)
    total = sum(dist.values())
    findings: list[LintFinding] = []
    if total == 0:
        return findings
    direct_count = dist.get("EXTRACTOR_DIRECT", 0)
    direct_fraction = direct_count / total
    findings.append(
        LintFinding(
            finding_type="EXTRACTION_QUALITY_REPORT",
            severity="INFO" if direct_fraction <= 0.5 else "WARNING",
            detail=(
                f"calibration_source distribution: {json.dumps(dist)}. "
                f"EXTRACTOR_DIRECT fraction: {direct_fraction:.1%}"
            ),
            recommended_action=(
                "Run calibration benchmarks to produce CALIBRATED_BENCHMARK particles"
                if direct_fraction > 0.5
                else None
            ),
        )
    )
    return findings


async def _report_pending_extractions(session: AsyncSession) -> list[LintFinding]:
    """Report corpus entries with PENDING or FAILED extraction status."""
    findings: list[LintFinding] = []
    pairs = await list_entry_status_pairs_with_extraction_status(
        session, [ExtractionStatus.PENDING, ExtractionStatus.FAILED]
    )
    for entry_id, status in pairs:
        findings.append(
            LintFinding(
                corpus_entry_id=entry_id,
                finding_type="PENDING_EXTRACTION",
                severity="WARNING",
                detail=f"Corpus entry {entry_id} has snapshot with extraction_status={status}",
                recommended_action="Run extraction for this entry",
            )
        )
    return findings


async def _report_empty_complete_snapshots(session: AsyncSession) -> list[LintFinding]:
    """Flag COMPLETE snapshots that produced zero particles (report-only).

    This is the audit surface for the F4.1 silent-loss bug: a fully-failed
    chunked/PDF extraction used to be stamped COMPLETE with zero particles and
    silently leave the retry queue. The pipeline now resets such snapshots to
    PENDING, but snapshots lost *before* the fix landed stay COMPLETE — this
    check re-surfaces them so the operator can re-extract.

    Report-only by design: a COMPLETE-but-empty snapshot can also be a
    genuinely claim-free source, so the finding recommends re-extraction to
    confirm rather than auto-resetting status. REVISIT snapshots (empty by
    design) are excluded upstream in ``list_complete_response_snapshots``.
    """
    findings: list[LintFinding] = []
    produced = await get_snapshot_ids_with_particles(session)
    for entry_id, snapshot_id in await list_complete_response_snapshots(session):
        if snapshot_id not in produced:
            findings.append(
                LintFinding(
                    corpus_entry_id=entry_id,
                    finding_type="EMPTY_COMPLETE_SNAPSHOT",
                    severity="WARNING",
                    detail=(
                        f"Snapshot {snapshot_id} of entry {entry_id} is COMPLETE but "
                        "produced zero particles — likely a silently-failed extraction "
                        "(F4.1), or a genuinely claim-free source"
                    ),
                    recommended_action=(
                        "Re-extract this entry (`reindex`, or reset the snapshot to "
                        "PENDING and run `extract --all-pending`) to confirm; if the "
                        "source is genuinely claim-free, no action is needed"
                    ),
                )
            )
    return findings


async def _report_schema_versions(session: AsyncSession) -> list[LintFinding]:
    """Report schema_version distribution; flag particles with mismatched versions."""
    findings: list[LintFinding] = []
    for version, count in (await count_active_particles_by_schema_version(session)).items():
        if version != SCHEMA_VERSION:
            findings.append(
                LintFinding(
                    finding_type="SCHEMA_VERSION_MISMATCH",
                    severity="WARNING",
                    detail=(
                        f"{count} ACTIVE particles have schema_version={version!r}"
                        f" (current: {SCHEMA_VERSION!r})"
                    ),
                    recommended_action="Migrate or reindex particles to current schema version",
                )
            )
    return findings
