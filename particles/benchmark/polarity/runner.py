"""Polarity-benchmark runner.

``run_polarity_benchmark(suite, extractor)`` is the one public entry point.
Report-only, like its content and modality siblings: it never writes the store,
never calibrates, never gates registration. Per case it

1. feeds the document prose to ``extractor.extract`` (passing the suite's
   ``source_type``; the general extractor accepts any type, cap. 1),
2. converts candidates to ``Particle`` via the shared
   :func:`particles.extraction.general.candidate_to_particle` (so emitted
   ``properties["extraction:polarity"]`` mirrors what the real pipeline would
   persist),
3. aligns the emitted claims to the case's gold labels with the **shared
   embedding equivalence judge** (:mod:`particles.benchmark.equivalence`),
   reusing its greedy-assignment logic rather than re-deriving cosine, and
4. records each aligned pair's ``(expected, emitted)`` polarity into the
   confusion accounting.

A case the extractor declines, or whose ``extract`` raises, degrades to a
quality note + zero aligned pairs rather than aborting the run — same
robustness contract as the content and modality runners.

The polarity classifier is **config-gated** in the general extractor
(``extraction_polarity.enabled``, default on; cap. 1). When it is off
every emitted claim is ``ASSERTED`` and the danger rates are vacuously ``0.0`` —
flattering, not real — so the runner reads the flag once, records it on the
report (``polarity_classifier_enabled``), and prepends a prominent quality note
when it is disabled.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from particles.benchmark.equivalence import EquivalenceJudge, match_emitted_to_expected
from particles.benchmark.polarity.metrics import (
    PolarityPair,
    confusion_counts,
    polarity_precision,
    polarity_recall,
    wrong_declined_rate,
    wrong_hidden_rate,
)
from particles.benchmark.polarity.schema import ClaimPolarity, PolaritySuite
from particles.benchmark.schema import ExpectedParticle
from particles.config import get_config
from particles.core.schema import ExtractorRef, Particle, Snapshot, UncertaintyNature
from particles.extraction.general import candidate_to_particle
from particles.extraction.polarity import POLARITY_ASSERTED, POLARITY_KEY
from particles.extraction.registry import ExtractorPlugin

log = logging.getLogger(__name__)

# The polarities reported, in a stable display order (ASSERTED first — it
# anchors the wrong-DECLINED headline, which is about ASSERTED claims).
_REPORTED_POLARITIES: tuple[ClaimPolarity, ...] = (
    ClaimPolarity.ASSERTED,
    ClaimPolarity.DECLINED,
    ClaimPolarity.HYPOTHETICAL,
)


def polarity_of(particle: Particle) -> ClaimPolarity:
    """Read a particle's claim-polarity off its Extension-side ``properties``.

    Mirrors :func:`particles.extraction.polarity.is_non_asserted`'s contract —
    the absence of the ``extraction:polarity`` key (or any unknown value) means
    ``ASSERTED``, so a particle minted before the axis existed reads as
    ``ASSERTED``. Core never branches on this; only the operation /
    benchmark layers do (cap. 1).
    """
    props = particle.properties or {}
    raw = str(props.get(POLARITY_KEY, POLARITY_ASSERTED)).strip().upper()
    try:
        return ClaimPolarity(raw)
    except ValueError:
        return ClaimPolarity.ASSERTED


@dataclass
class PolarityConfusionCell:
    """One ``(expected, emitted)`` polarity bucket and its count (JSON-friendly)."""

    expected: str
    emitted: str
    count: int


@dataclass
class PolarityCaseResult:
    """Per-case detail surfaced in the table / JSON renderers."""

    case_id: str
    claims_emitted: int
    claims_aligned: int
    claims_unaligned: int
    pairs: list[tuple[str, str]]  # aligned (expected, emitted) polarity names


@dataclass
class PolarityReport:
    """Output of one polarity-benchmark run — extractor × suite."""

    suite_id: str
    suite_version: str
    extractor_id: str
    extractor_version: str
    cases_run: int
    cases_total: int
    claims_aligned: int
    claims_unaligned: int
    confusion: list[PolarityConfusionCell]
    precision: dict[str, float]
    recall: dict[str, float]
    wrong_declined_rate: float
    wrong_hidden_rate: float
    polarity_classifier_enabled: bool
    per_case: list[PolarityCaseResult]
    judge: str
    alignment_threshold: float
    generated_at: datetime
    quality_notes: list[str] = field(default_factory=list)


async def run_polarity_benchmark(
    suite: PolaritySuite,
    extractor: ExtractorPlugin,
    *,
    judge: EquivalenceJudge = EquivalenceJudge.EMBEDDING,
    threshold: float = 0.65,
) -> PolarityReport:
    """Run one polarity suite against one extractor and return the report.

    ``threshold`` is the cosine floor for aligning an emitted claim to a gold
    label — deliberately looser than the content harness's 0.80 (matching the
    modality sibling) because the general extractor paraphrases the source, so
    the gold text and emitted text are near-paraphrases, and the metric we care
    about is the *polarity* of the aligned claim, not exact content recall (that
    is the content harness's job).
    """
    aligned_pairs: list[PolarityPair] = []
    per_case: list[PolarityCaseResult] = []
    quality_notes: list[str] = []
    cases_run = 0
    total_aligned = 0
    total_unaligned = 0

    classifier_enabled = get_config().extraction_polarity.enabled
    if not classifier_enabled:
        quality_notes.append(
            "extraction_polarity.enabled is False — the classifier is OFF, so every "
            "emitted claim is ASSERTED and the danger rates are vacuously 0.0 "
            "(flattering, not real). Enable it to measure the classifier."
        )

    ext_ref = ExtractorRef(name=extractor.EXTRACTOR_ID, version=extractor.EXTRACTOR_VERSION)

    for case in suite.cases:
        if not extractor.accepts(suite.source_type):
            quality_notes.append(
                f"Case {case.case_id}: extractor declines source_type {suite.source_type!r}"
            )
            continue

        cases_run += 1

        content = case.document.encode("utf-8")
        snapshot = Snapshot(content_hash=hashlib.sha256(content).hexdigest())

        try:
            result = await extractor.extract(snapshot, content, source_type=suite.source_type)
        except Exception as exc:  # noqa: BLE001 — one bad case must not abort the suite
            quality_notes.append(f"Case {case.case_id}: extract() raised {exc!r}")
            total_unaligned += len(case.labels)
            per_case.append(
                PolarityCaseResult(
                    case_id=case.case_id,
                    claims_emitted=0,
                    claims_aligned=0,
                    claims_unaligned=len(case.labels),
                    pairs=[],
                )
            )
            continue

        emitted = [
            candidate_to_particle(
                candidate,
                corpus_entry_id="benchmark-polarity-entry",
                snapshot_id=snapshot.snapshot_id,
                asserted_by=extractor.EXTRACTOR_ID,
                extractor_ref=ext_ref,
                subject_ids=list(candidate.subjects),
            )
            for candidate in result.candidates
        ]
        quality_notes.extend(f"Case {case.case_id}: {n}" for n in result.quality_notes)

        # Reuse the content harness's greedy embedding judge to align emitted
        # claims to gold labels. confidence_min=0.0 disables the under-confidence
        # demotion — polarity alignment is content-only.
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
            emitted, throwaway_expected, judge=judge, threshold=threshold
        )
        polarity_by_content = {label.content: label.polarity for label in case.labels}

        case_pairs: list[PolarityPair] = [
            (polarity_by_content[expected_p.content], polarity_of(emitted_p))
            for expected_p, emitted_p in match.matched
        ]
        aligned_pairs.extend(case_pairs)
        aligned = len(case_pairs)
        unaligned = len(case.labels) - aligned
        total_aligned += aligned
        total_unaligned += unaligned

        per_case.append(
            PolarityCaseResult(
                case_id=case.case_id,
                claims_emitted=len(emitted),
                claims_aligned=aligned,
                claims_unaligned=unaligned,
                pairs=[(exp.value, emi.value) for exp, emi in case_pairs],
            )
        )

    confusion = [
        PolarityConfusionCell(expected=exp.value, emitted=emi.value, count=count)
        for (exp, emi), count in sorted(
            confusion_counts(aligned_pairs).items(),
            key=lambda kv: (kv[0][0].value, kv[0][1].value),
        )
    ]
    precision = {p.value: polarity_precision(aligned_pairs, p) for p in _REPORTED_POLARITIES}
    recall = {p.value: polarity_recall(aligned_pairs, p) for p in _REPORTED_POLARITIES}

    return PolarityReport(
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
        wrong_declined_rate=wrong_declined_rate(aligned_pairs),
        wrong_hidden_rate=wrong_hidden_rate(aligned_pairs),
        polarity_classifier_enabled=classifier_enabled,
        per_case=per_case,
        judge=judge.value,
        alignment_threshold=threshold,
        generated_at=datetime.now(UTC),
        quality_notes=quality_notes,
    )
