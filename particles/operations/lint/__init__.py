"""§9.4 Lint operation package — structural and semantic checks.

Public surface (stable):
    run_lint, LintReport, LintFinding, ContradictionProbeControl

Internal detector helpers re-exported for tests and tooling that still
reference the pre-split module paths (``particles.operations.lint.<name>``):
    _llm_call                          (../_llm.py — shared seam)
    _check_contradictions,             (./contradictions.py)
        _llm_check_contradiction
    _check_granularity_length,         (./granularity.py)
        _check_granularity_violations,
        _llm_check_granularity
    _check_compound_assertions         (./assertion_quality.py — COMPOUND_ASSERTION)
    _check_staleness,                  (./staleness.py)
        _check_retraction_propagation,
        _check_corpus_link_integrity,
        _check_confidence_decay,
        _check_recency_staleness
    _check_orphans,                    (./coverage.py)
        _check_no_subject_claims,
        _check_phantom_subjects,
        _report_extraction_quality,
        _report_pending_extractions,
        _report_schema_versions,
        _check_structured_claim_subjects
    _check_wikidata_link_confidence    (./wikidata_links.py)

Co-evidential candidate proposal moved to ``particles.operations.links_suggest``
; lint no longer emits ``CO_EVIDENTIAL_CANDIDATE`` findings.

See ``orchestrator.run_lint`` for the entry point ordering.
"""

from __future__ import annotations

from particles.core.schema import LintFinding, LintReport
from particles.operations._llm import _llm_call

from .assertion_quality import _check_compound_assertions
from .contradictions import (
    ContradictionProbeControl,
    _check_contradictions,
    _llm_check_contradiction,
)
from .coverage import (
    _check_no_subject_claims,
    _check_orphans,
    _check_phantom_subjects,
    _check_structured_claim_subjects,
    _report_extraction_quality,
    _report_pending_extractions,
    _report_schema_versions,
)
from .granularity import (
    _check_granularity_length,
    _check_granularity_violations,
    _llm_check_granularity,
)
from .orchestrator import run_lint
from .staleness import (
    _check_confidence_decay,
    _check_corpus_link_integrity,
    _check_recency_staleness,
    _check_retraction_propagation,
    _check_staleness,
)
from .wikidata_links import _check_wikidata_link_confidence

__all__ = [
    # Public
    "run_lint",
    "LintReport",
    "LintFinding",
    "ContradictionProbeControl",
    # Internal detectors (re-exported for tests / external callers that pin
    # the pre-split import paths). New code should import from the submodule
    # directly.
    "_llm_call",
    "_check_staleness",
    "_check_retraction_propagation",
    "_check_corpus_link_integrity",
    "_check_confidence_decay",
    "_check_recency_staleness",
    "_check_orphans",
    "_check_no_subject_claims",
    "_check_phantom_subjects",
    "_report_extraction_quality",
    "_report_pending_extractions",
    "_report_schema_versions",
    "_check_structured_claim_subjects",
    "_check_granularity_length",
    "_check_granularity_violations",
    "_llm_check_granularity",
    "_check_compound_assertions",
    "_check_contradictions",
    "_llm_check_contradiction",
    "_check_wikidata_link_confidence",
]
