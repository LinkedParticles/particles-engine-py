"""§9.4 Lint operation — orchestrator.

Gathers ACTIVE particles, runs each structural and (optionally) semantic
detector in turn, and assembles a ``LintReport``.

Structural checks (no LLM):
  - Staleness, retraction propagation, corpus-link integrity, confidence
    decay (``.staleness``).
  - Orphans, no-subject claims, phantom/low-coverage subjects, extraction-quality report,
    pending extractions, empty-COMPLETE-snapshot audit, schema-version audit,
    structured-claim subject mismatch, bare ``properties`` keys (``.coverage``).
  - Granularity length pre-check (``.granularity``).
  - Undated-retirement census (``.retirement``) — once-believed retired
    particles with no stored retirement instant.
  - Wikidata link confidence (``.wikidata_links``).
  - Contestedness distribution (``.contestedness``) — store-level lens-divergence
    histogram under the store's own adopted policy set (absent when
    fewer than two policies are configured).

Semantic checks (LLM-assisted, when ``semantic=True``):
  - Contradictions (``.contradictions``).
  - Granularity violations (``.granularity``).

Co-evidential candidate proposal moved out of lint into the
``particles links suggest`` curation operation; lint no longer
emits ``CO_EVIDENTIAL_CANDIDATE`` findings.

Output: ``LintReport`` with JSON-LD-serialisable findings + Markdown Bridge
rendering.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import FIX_CAPABLE_CATEGORIES, LintFinding, LintReport
from particles.observability import traced
from particles.operations._llm import llm_circuit_open

from .assertion_quality import _check_compound_assertions
from .citation_signal import _check_undeposited_cited_sources
from .contestedness import _check_contested
from .contradictions import ContradictionProbeControl, _check_contradictions
from .coverage import (
    _check_bare_properties_keys,
    _check_no_subject_claims,
    _check_orphans,
    _check_phantom_subjects,
    _check_structured_claim_subjects,
    _report_empty_complete_snapshots,
    _report_extraction_quality,
    _report_pending_extractions,
    _report_schema_versions,
)
from .granularity import (
    _check_granularity_length,
    _check_granularity_violations,
)
from .retirement import _check_undated_retirements
from .staleness import (
    _check_confidence_decay,
    _check_corpus_link_integrity,
    _check_recency_staleness,
    _check_retraction_propagation,
    _check_staleness,
)
from .wikidata_links import _check_wikidata_link_confidence


@traced("lint")
async def run_lint(
    session: AsyncSession,
    fix: bool = False,
    semantic: bool = True,
    low_coverage_threshold: int = 3,
    *,
    contradiction_probe: ContradictionProbeControl | None = None,
    granularity_probe: bool = True,
) -> LintReport:
    """Run all lint checks and return a LintReport.

    Args:
        fix: if True, apply status changes immediately (staleness → PROVENANCE_STALE etc.).
            Defaults to False — lint is read-only unless the caller opts in.
        semantic: if True, run LLM-assisted contradiction and granularity checks
        low_coverage_threshold: subjects with fewer than this many ACTIVE CLAIM particles
            are flagged as PHANTOM_SUBJECT (0) or LOW_COVERAGE_SUBJECT (< threshold).
        contradiction_probe: optional cap/scope/progress control for the contradiction probe; also carries back the
            candidate-pair census. ``None`` keeps the probe unbounded.
        granularity_probe: set False to skip the per-particle LLM granularity
            check within the semantic pass. ``collect_cards`` does —
            ``GRANULARITY_VIOLATION`` has no card kind, so for the curation
            queue and the audit those LLM calls would be pure discard.
    """
    findings: list[LintFinding] = []

    # --- Structural checks ---
    findings += await _check_staleness(session, fix)
    findings += await _check_retraction_propagation(session, fix)
    findings += await _check_corpus_link_integrity(session, fix)
    findings += await _check_confidence_decay(session)
    findings += await _check_recency_staleness(session)
    findings += await _check_orphans(session)
    findings += await _check_no_subject_claims(session)
    findings += await _check_phantom_subjects(session, low_coverage_threshold)
    findings += await _report_extraction_quality(session)
    findings += await _report_pending_extractions(session)
    findings += await _report_empty_complete_snapshots(session)
    findings += await _report_schema_versions(session)
    findings += await _check_structured_claim_subjects(session)
    findings += await _check_bare_properties_keys(session)
    findings += await _check_granularity_length(session)
    findings += await _check_compound_assertions(session)
    findings += await _check_undated_retirements(session)

    findings += await _check_wikidata_link_confidence(session)
    findings += await _check_undeposited_cited_sources(session)
    findings += await _check_contested(session)

    # --- Semantic checks (LLM-assisted) ---
    if semantic:
        findings += await _check_contradictions(session, fix, control=contradiction_probe)
        if granularity_probe:
            findings += await _check_granularity_violations(session)

    summary: dict[str, int] = {}
    for f in findings:
        summary[f.finding_type] = summary.get(f.finding_type, 0) + 1

    fixed_counts: dict[str, int] = {}
    if fix:
        # Every finding in a fix-capable category was paired with a status
        # transition by its detector (1:1). Reflect that here.
        for category in FIX_CAPABLE_CATEGORIES:
            fixed_counts[category] = summary.get(category, 0)
        await session.commit()

    return LintReport(
        findings=findings,
        summary=summary,
        fixed_counts=fixed_counts,
        # Account-level LLM failure during the semantic pass tripped the breaker
        #: report that semantic checks were skipped, not clean.
        semantic_skipped=semantic and llm_circuit_open(),
    )
