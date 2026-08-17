"""Validity-benchmark runner.

``run_validity_benchmark(suite, extractor)`` is the one public entry point.
Report-only, like its content / modality / polarity siblings: it never writes the
store, never calibrates, never gates registration. Per case it

1. stamps the case's ``reference_date`` onto the snapshot's
   ``content_published_at`` so the extractor resolves relative validity
   boundaries against a fixed, reproducible instant,
2. feeds the document prose to ``extractor.extract`` (passing the suite's
   ``source_type``; the general extractor accepts any type),
3. converts candidates to ``Particle`` via the shared
   :func:`particles.extraction.general.candidate_to_particle` (so the emitted
   ``valid_until`` mirrors what the real pipeline would persist),
4. aligns emitted claims to the case's gold labels with the **shared embedding
   equivalence judge** (:mod:`particles.benchmark.equivalence`) at the looser
   0.65 threshold, and
5. records each aligned pair's ``(expected_boundary, emitted_boundary)`` into the
   accounting.

A case the extractor declines, or whose ``extract`` raises, degrades to a
quality note + zero aligned pairs rather than aborting the run — same robustness
contract as the sibling runners.

The validity extractor is **config-gated** (``extraction_validity.enabled``,
default on). When it is off no emitted claim carries a ``valid_until``
and the danger rate is vacuously ``0.0`` — flattering, not real — so the runner
reads the flag once, records it on the report (``validity_extractor_enabled``),
and prepends a prominent quality note when it is disabled.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from particles.benchmark.equivalence import EquivalenceJudge, match_emitted_to_expected
from particles.benchmark.schema import ExpectedParticle
from particles.benchmark.validity.metrics import (
    ValidityPair,
    date_accuracy,
    expiry_precision,
    expiry_recall,
    support_counts,
    wrong_expiry_rate,
)
from particles.benchmark.validity.schema import ValiditySuite
from particles.config import get_config
from particles.core.schema import ExtractorRef, Particle, Snapshot, UncertaintyNature
from particles.extraction.general import candidate_to_particle
from particles.extraction.registry import ExtractorPlugin

log = logging.getLogger(__name__)


def valid_until_date(particle: Particle) -> date | None:
    """The emitted validity boundary as a ``date``, or ``None`` if unbounded.

    ``valid_until`` is a first-class ``Particle`` field (unlike polarity, which
    rides on ``properties``), so this simply projects it to day granularity for
    comparison against the gold ``date`` labels.
    """
    return particle.valid_until.date() if particle.valid_until is not None else None


@dataclass
class ValidityCaseResult:
    """Per-case detail surfaced in the table / JSON renderers."""

    case_id: str
    claims_emitted: int
    claims_aligned: int
    claims_unaligned: int
    # aligned (expected ISO or "-", emitted ISO or "-") boundary pairs
    pairs: list[tuple[str, str]]


@dataclass
class ValidityReport:
    """Output of one validity-benchmark run — extractor × suite."""

    suite_id: str
    suite_version: str
    extractor_id: str
    extractor_version: str
    cases_run: int
    cases_total: int
    claims_aligned: int
    claims_unaligned: int
    wrong_expiry_rate: float
    expiry_precision: float
    expiry_recall: float
    date_accuracy: float
    date_tolerance_days: int
    support: dict[str, int]
    validity_extractor_enabled: bool
    per_case: list[ValidityCaseResult]
    judge: str
    alignment_threshold: float
    generated_at: datetime
    quality_notes: list[str] = field(default_factory=list)


def _iso_or_dash(d: date | None) -> str:
    return d.isoformat() if d is not None else "-"


async def run_validity_benchmark(
    suite: ValiditySuite,
    extractor: ExtractorPlugin,
    *,
    judge: EquivalenceJudge = EquivalenceJudge.EMBEDDING,
    threshold: float = 0.65,
) -> ValidityReport:
    """Run one validity suite against one extractor and return the report.

    ``threshold`` is the cosine floor for aligning an emitted claim to a gold
    label — deliberately looser than the content harness's 0.80 (matching the
    modality / polarity siblings) because the general extractor paraphrases the
    source, so the gold text and emitted text are near-paraphrases, and the
    metric we care about is the *validity boundary* of the aligned claim, not
    exact content recall.
    """
    aligned_pairs: list[ValidityPair] = []
    per_case: list[ValidityCaseResult] = []
    quality_notes: list[str] = []
    cases_run = 0
    total_aligned = 0
    total_unaligned = 0

    classifier_enabled = get_config().extraction_validity.enabled
    if not classifier_enabled:
        quality_notes.append(
            "extraction_validity.enabled is False — the validity extractor is OFF, so no "
            "emitted claim carries a valid_until and wrong_expiry_rate is vacuously 0.0 "
            "(flattering, not real). Enable it to measure the extractor."
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
        # stamp the case's reference date as the snapshot's publication
        # instant so the extractor anchors relative boundaries reproducibly.
        snapshot = Snapshot(
            content_hash=hashlib.sha256(content).hexdigest(),
            content_published_at=case.reference_datetime(),
        )

        try:
            result = await extractor.extract(snapshot, content, source_type=suite.source_type)
        except Exception as exc:  # noqa: BLE001 — one bad case must not abort the suite
            quality_notes.append(f"Case {case.case_id}: extract() raised {exc!r}")
            total_unaligned += len(case.labels)
            per_case.append(
                ValidityCaseResult(
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
                corpus_entry_id="benchmark-validity-entry",
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
        # demotion — validity alignment is content-only.
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
        boundary_by_content = {label.content: label.expected_valid_until for label in case.labels}

        case_pairs: list[ValidityPair] = [
            (boundary_by_content[expected_p.content], valid_until_date(emitted_p))
            for expected_p, emitted_p in match.matched
        ]
        aligned_pairs.extend(case_pairs)
        aligned = len(case_pairs)
        unaligned = len(case.labels) - aligned
        total_aligned += aligned
        total_unaligned += unaligned

        per_case.append(
            ValidityCaseResult(
                case_id=case.case_id,
                claims_emitted=len(emitted),
                claims_aligned=aligned,
                claims_unaligned=unaligned,
                pairs=[(_iso_or_dash(exp), _iso_or_dash(emi)) for exp, emi in case_pairs],
            )
        )

    return ValidityReport(
        suite_id=suite.suite_id,
        suite_version=suite.version,
        extractor_id=extractor.EXTRACTOR_ID,
        extractor_version=extractor.EXTRACTOR_VERSION,
        cases_run=cases_run,
        cases_total=len(suite.cases),
        claims_aligned=total_aligned,
        claims_unaligned=total_unaligned,
        wrong_expiry_rate=wrong_expiry_rate(aligned_pairs),
        expiry_precision=expiry_precision(aligned_pairs),
        expiry_recall=expiry_recall(aligned_pairs),
        date_accuracy=date_accuracy(aligned_pairs, suite.date_tolerance_days),
        date_tolerance_days=suite.date_tolerance_days,
        support=support_counts(aligned_pairs),
        validity_extractor_enabled=classifier_enabled,
        per_case=per_case,
        judge=judge.value,
        alignment_threshold=threshold,
        generated_at=datetime.now(UTC),
        quality_notes=quality_notes,
    )
