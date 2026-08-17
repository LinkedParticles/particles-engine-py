"""Modality-benchmark runner.

``run_modality_benchmark(suite, extractor)`` is the one public entry point.
Report-only, like the content runner: it never writes the store, never
calibrates, never gates registration. Per case it

1. feeds the entry prose to ``extractor.extract`` (passing the suite's
   ``source_type`` so genre-gated extractors accept it),
2. converts candidates to ``Particle`` via the shared
   :func:`particles.extraction.general.candidate_to_particle` (so emitted
   ``assertion_modality`` / ``particle_type`` mirror what the real pipeline
   would persist),
3. splits off the whole-entry ``NARRATIVE`` (for the narrative-emission rate),
4. aligns the remaining claim particles to the case's gold labels with the
   **shared embedding equivalence judge** (:mod:`particles.benchmark.equivalence`),
   reusing its greedy-assignment logic rather than re-deriving cosine, and
5. records each aligned pair's ``(expected, emitted)`` modality into the
   confusion accounting.

A case the extractor declines, or whose ``extract`` raises, degrades to a
quality note + zero aligned pairs rather than aborting the run — same
robustness contract as the content runner.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from particles.benchmark.equivalence import EquivalenceJudge, match_emitted_to_expected
from particles.benchmark.modality.metrics import (
    ModalityPair,
    confusion_counts,
    false_non_falsifiable_rate,
    modality_precision,
    modality_recall,
    narrative_emission_rate,
)
from particles.benchmark.modality.schema import ModalitySuite
from particles.benchmark.schema import ExpectedParticle
from particles.core.schema import (
    AssertionModality,
    ExtractorRef,
    ParticleType,
    Snapshot,
    UncertaintyNature,
)
from particles.extraction.general import candidate_to_particle
from particles.extraction.registry import ExtractorPlugin

log = logging.getLogger(__name__)

# The modalities reported, in a stable display order (FALSIFIABLE first — it
# anchors the false-non-FALSIFIABLE headline).
_REPORTED_MODALITIES: tuple[AssertionModality, ...] = (
    AssertionModality.FALSIFIABLE,
    AssertionModality.EVALUATIVE,
    AssertionModality.EXPERIENTIAL,
    AssertionModality.CONSTITUTIVE,
)


@dataclass
class ModalityConfusionCell:
    """One ``(expected, emitted)`` modality bucket and its count (JSON-friendly)."""

    expected: str
    emitted: str
    count: int


@dataclass
class ModalityCaseResult:
    """Per-case detail surfaced in the table / JSON renderers."""

    case_id: str
    claims_emitted: int
    claims_aligned: int
    claims_unaligned: int
    narrative_expected: bool
    narrative_emitted: bool
    pairs: list[tuple[str, str]]  # aligned (expected, emitted) modality names


@dataclass
class ModalityReport:
    """Output of one modality-benchmark run — extractor × suite."""

    suite_id: str
    suite_version: str
    extractor_id: str
    extractor_version: str
    cases_run: int
    cases_total: int
    claims_aligned: int
    claims_unaligned: int
    confusion: list[ModalityConfusionCell]
    precision: dict[str, float]
    recall: dict[str, float]
    false_non_falsifiable_rate: float
    narrative_cases_expected: int
    narrative_cases_emitted: int
    narrative_emission_rate: float
    per_case: list[ModalityCaseResult]
    judge: str
    alignment_threshold: float
    generated_at: datetime
    quality_notes: list[str] = field(default_factory=list)


async def run_modality_benchmark(
    suite: ModalitySuite,
    extractor: ExtractorPlugin,
    *,
    judge: EquivalenceJudge = EquivalenceJudge.EMBEDDING,
    threshold: float = 0.65,
) -> ModalityReport:
    """Run one modality suite against one extractor and return the report.

    ``threshold`` is the cosine floor for aligning an emitted claim to a gold
    label — deliberately looser than the content harness's 0.80 because the
    journal extractor *reifies* and rephrases ("I felt anxious" → "The author
    felt anxious"), so the gold text and emitted text are paraphrases, and the
    metric we care about is the *modality* of the aligned claim, not exact
    content recall (that is the content harness's job).
    """
    aligned_pairs: list[ModalityPair] = []
    per_case: list[ModalityCaseResult] = []
    quality_notes: list[str] = []
    cases_run = 0
    total_aligned = 0
    total_unaligned = 0
    narrative_expected_count = 0
    narrative_emitted_count = 0

    ext_ref = ExtractorRef(name=extractor.EXTRACTOR_ID, version=extractor.EXTRACTOR_VERSION)

    for case in suite.cases:
        if not extractor.accepts(suite.source_type):
            quality_notes.append(
                f"Case {case.case_id}: extractor declines source_type {suite.source_type!r}"
            )
            continue

        cases_run += 1
        if case.narrative_expected:
            narrative_expected_count += 1

        content = case.entry.encode("utf-8")
        snapshot = Snapshot(content_hash=hashlib.sha256(content).hexdigest())

        try:
            result = await extractor.extract(snapshot, content, source_type=suite.source_type)
        except Exception as exc:  # noqa: BLE001 — one bad case must not abort the suite
            quality_notes.append(f"Case {case.case_id}: extract() raised {exc!r}")
            total_unaligned += len(case.labels)
            per_case.append(
                ModalityCaseResult(
                    case_id=case.case_id,
                    claims_emitted=0,
                    claims_aligned=0,
                    claims_unaligned=len(case.labels),
                    narrative_expected=case.narrative_expected,
                    narrative_emitted=False,
                    pairs=[],
                )
            )
            continue

        emitted = [
            candidate_to_particle(
                candidate,
                corpus_entry_id="benchmark-modality-entry",
                snapshot_id=snapshot.snapshot_id,
                asserted_by=extractor.EXTRACTOR_ID,
                extractor_ref=ext_ref,
                subject_ids=list(candidate.subjects),
            )
            for candidate in result.candidates
        ]
        quality_notes.extend(f"Case {case.case_id}: {n}" for n in result.quality_notes)

        narrative_emitted = any(p.particle_type == ParticleType.NARRATIVE for p in emitted)
        if case.narrative_expected and narrative_emitted:
            narrative_emitted_count += 1

        claim_particles = [p for p in emitted if p.particle_type != ParticleType.NARRATIVE]

        # Reuse the content harness's greedy embedding judge to align emitted
        # claims to gold labels. confidence_min=0.0 disables the under-confidence
        # demotion — modality alignment is content-only.
        throwaway_expected = [
            ExpectedParticle(
                content=label.content,
                confidence_min=0.0,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=True,
            )
            for label in case.labels
        ]
        match = await match_emitted_to_expected(
            claim_particles, throwaway_expected, judge=judge, threshold=threshold
        )
        modality_by_content = {label.content: label.modality for label in case.labels}

        case_pairs: list[ModalityPair] = [
            (modality_by_content[expected_p.content], emitted_p.assertion_modality)
            for expected_p, emitted_p in match.matched
        ]
        aligned_pairs.extend(case_pairs)
        aligned = len(case_pairs)
        unaligned = len(case.labels) - aligned
        total_aligned += aligned
        total_unaligned += unaligned

        per_case.append(
            ModalityCaseResult(
                case_id=case.case_id,
                claims_emitted=len(claim_particles),
                claims_aligned=aligned,
                claims_unaligned=unaligned,
                narrative_expected=case.narrative_expected,
                narrative_emitted=narrative_emitted,
                pairs=[(exp.value, emi.value) for exp, emi in case_pairs],
            )
        )

    confusion = [
        ModalityConfusionCell(expected=exp.value, emitted=emi.value, count=count)
        for (exp, emi), count in sorted(
            confusion_counts(aligned_pairs).items(),
            key=lambda kv: (kv[0][0].value, kv[0][1].value),
        )
    ]
    precision = {m.value: modality_precision(aligned_pairs, m) for m in _REPORTED_MODALITIES}
    recall = {m.value: modality_recall(aligned_pairs, m) for m in _REPORTED_MODALITIES}

    return ModalityReport(
        suite_id=suite.suite_id,
        suite_version=suite.version,
        extractor_id=extractor.EXTRACTOR_ID,
        extractor_version=extractor.EXTRACTOR_VERSION,
        cases_run=cases_run,
        cases_total=len(suite.cases),
        claims_aligned=total_aligned,
        claims_unaligned=total_unaligned,
        confusion=confusion,
        precision=precision,
        recall=recall,
        false_non_falsifiable_rate=false_non_falsifiable_rate(aligned_pairs),
        narrative_cases_expected=narrative_expected_count,
        narrative_cases_emitted=narrative_emitted_count,
        narrative_emission_rate=narrative_emission_rate(
            narrative_expected_count, narrative_emitted_count
        ),
        per_case=per_case,
        judge=judge.value,
        alignment_threshold=threshold,
        generated_at=datetime.now(UTC),
        quality_notes=quality_notes,
    )
