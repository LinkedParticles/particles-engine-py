# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark runner — ties loader + extractor + equivalence + metrics.

``run_benchmark(suite, extractor)`` is the one public entry point.
It is *report-only*: never writes calibration_history, never modifies
the particle store, never registers anything. The caller — typically
the CLI — gets a :class:`BenchmarkReport` and decides what to do
with it.

Per-case orchestration mirrors the conformance validator
(:mod:`particles.conformance.validator`):

1. Resolve the case's source content + snapshot (via
   :func:`particles.benchmark.loader.resolve_case_content`).
2. Skip cases the extractor's ``accepts(source_type)`` rejects;
   record a quality note so the operator can see what was skipped.
3. Invoke ``extractor.extract(snapshot, content)`` and convert each
   ``CandidateParticle`` to a ``Particle`` via the extractor-agnostic
   ``candidate_to_particle`` helper so what's matched mirrors what
   the real pipeline would persist.
4. Run the equivalence match between emitted and expected.
5. Accumulate matched-IDs and emitted-particles across cases for the
   global metrics rollup, and record each emitted particle's
   ``(raw confidence, matched?)`` pair on its ``CaseResult``, plus one
   :class:`EmittedClaim` per emission carrying that claim's text, stated
   confidence and judged outcome.

Step 5's pairs are what :func:`graded_pairs` hands to ``particles extractor
calibrate``. They exist because a caller cannot reconstruct them from the
report's summaries: re-running the extractor to recover the confidences mints
*new* particle ids, which match nothing in this run's id set — the
bug, which labelled every particle incorrect and drove six months of fits to
the optimizer bound.

:func:`run_benchmark_repeated` is the repeat-runs wrapper: it
calls :func:`run_benchmark` N times over the same suite and returns an
:class:`AggregateBenchmarkReport` — the N unmodified reports plus a
per-metric distribution. :class:`BenchmarkReport` gains no *run-level*
field; the aggregate is a separate object that *contains* reports, the same
way the memory harness keeps its own report model beside it. (Only
run-level aggregates are excluded. Per-case detail with a default is a
different thing and is fine — ``graded`` and ``emitted_claims`` below. What
techspec §13.3 actually freezes is the suite *input* schema in
:mod:`particles.benchmark.schema`; it specifies no report shape at all.)

Step 5's :class:`EmittedClaim` records are why a *saved* report can be audited.
The id lists (``matched`` / ``spurious`` / ``under_confidence``) name emitted
particles by uuid, and the harness never persists to the store — so after the
process exits those uuids resolve to nothing, and a report saying precision
0.73 gave no way to read the 27 %. A hallucinated claim and a correct claim
the gold set happens not to list were the same three lines of JSON. The
records are strictly additive: every id list keeps its shape, so the format-1
envelope needs no bump and existing readers are untouched.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from particles.benchmark.equivalence import (
    EquivalenceJudge,
    MatchResult,
    match_emitted_to_expected,
)
from particles.benchmark.loader import (
    SuiteLoadError,
    resolve_case_content,
)
from particles.benchmark.metrics import (
    compute_calibration_error,
    compute_precision,
    compute_recall,
)
from particles.benchmark.schema import BenchmarkSuite
from particles.config import get_config
from particles.core.schema import ExtractorRef, Particle
from particles.extraction.general import candidate_to_particle
from particles.extraction.registry import ExtractorPlugin

log = logging.getLogger(__name__)

#: A Wikidata QID, which ``CandidateParticle.subjects`` may carry alongside
#: names (``numista/coin.py`` appends one to issuer subjects). It is an
#: identifier, not a name, so it is dropped before subject-aware matching for
#: the same reason a UUID is (D4): "Q42: the coin was struck in 1961"
#: is not a rendering of the claim any gold string will be written as. Under
#: the max it could only ever lose, so this costs nothing but the embedding
#: pass it avoids.
_QID_RE = re.compile(r"^Q\d+$")


class ClaimOutcome(StrEnum):
    """How the equivalence judge classified one emitted claim.

    The three outcomes partition a case's emitted set: every emitted
    particle is matched, demoted to an under-confidence partial match, or
    spurious. A :class:`StrEnum` so ``asdict()`` + ``json.dumps`` render it
    as its bare string in a persisted run file.
    """

    MATCHED = "matched"
    UNDER_CONFIDENCE = "under_confidence"
    SPURIOUS = "spurious"


@dataclass(frozen=True)
class EmittedClaim:
    """One emitted claim, recorded with enough text to audit it later.

    The sibling id lists (``matched`` / ``spurious`` / ``under_confidence``)
    identify emitted particles by uuid, and a benchmark run never persists to
    the store — so once the process exits, those uuids resolve to nothing. A
    saved report could report precision 0.73 and give an operator no way to
    read the 27 % that matched nothing: a hallucination and a
    correct-but-not-in-the-gold-set claim were indistinguishable, and the only
    recourse was re-running the suite and paying for the inference again. This
    record is that missing half; the id lists are unchanged beside it.

    ``content`` is the emitted claim text, or ``None`` when
    ``benchmark.record_claim_text`` is off — the harness can be pointed at a
    private corpus, and the run files outlive the run. ``confidence`` is the
    **raw stated** value (the runner converts candidates with
    ``calibration=None``, so nothing has been scaled), which is what makes the
    reported ``calibration_error`` auditable per claim rather than only in
    aggregate. ``matched_expected`` is the gold text this claim was assigned
    to — ``None`` for a spurious claim — so *what matched what* is readable
    from one record instead of joined across two lists.
    """

    particle_id: str
    outcome: ClaimOutcome
    confidence: float
    content: str | None = None
    matched_expected: str | None = None
    # The subject names this claim is about. Recorded because they
    # are an *input to the match* under subject-aware matching, so a report
    # that omitted them could not explain why a pair scored as it did — the
    # same after-the-fact-auditability argument the text itself rests on.
    # Suppressed with ``content`` when ``benchmark.record_claim_text`` is off:
    # a subject name is extracted from the corpus too.
    subjects: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    """Per-case detail surfaced in the table and JSON renderers."""

    case_id: str
    emitted_count: int
    matched: list[tuple[str, str]]  # (expected_content, emitted_particle_id)
    matched_required_count: int  # subset of ``matched`` whose expected was required
    missed_required: list[str]
    spurious: list[str]
    under_confidence: list[tuple[str, float, float]]  # (expected, stated, required_min)
    # (raw stated confidence, matched?) for every particle this case emitted —
    # the labelled population `extractor calibrate` fits its temperature on.
    # Raw because the runner converts candidates with
    # ``calibration=None``, so nothing has been scaled. Defaulted so a case that
    # crashed mid-extract contributes an empty list rather than a wrong label.
    graded: list[tuple[float, bool]] = field(default_factory=list)
    # One :class:`EmittedClaim` per emitted particle — the same population as
    # ``graded``, carrying the claim *text* and its judged outcome so a saved
    # report can be audited without re-running the suite. Additive beside the
    # id lists above, which keep their shapes; defaulted for the same reason
    # ``graded`` is, and empty when the extractor emitted nothing.
    emitted_claims: list[EmittedClaim] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Output of one benchmark run — extractor × suite."""

    suite_id: str
    suite_version: str
    extractor_id: str
    extractor_version: str
    cases_run: int
    cases_total: int
    particles_emitted: int
    particles_required_total: int
    metrics: dict[str, float]
    per_case: list[CaseResult]
    generated_at: datetime
    judge: str
    equivalence_threshold: float
    quality_notes: list[str] = field(default_factory=list)


def graded_pairs(report: BenchmarkReport) -> tuple[list[float], list[bool]]:
    """Flatten a report's per-case labelled pairs into ``(raw_values, correct)``.

    The population ``particles extractor calibrate`` fits on. Order
    is case order then emission order, and the two lists are index-aligned —
    :meth:`particles.extraction.calibration.TemperatureScaler.fit` requires
    equal lengths and raises otherwise.

    The label is **semantic match**, so it includes under-confidence partial
    matches — deliberately a wider set than the ``matched_ids``
    precision and recall are computed from. ``fit`` then drops the saturated
    pairs from what it returns here; this function is the whole population.
    """
    raw_values: list[float] = []
    correct: list[bool] = []
    for case in report.per_case:
        for raw, matched in case.graded:
            raw_values.append(raw)
            correct.append(matched)
    return raw_values, correct


@dataclass(frozen=True)
class MetricStat:
    """Distribution of one metric across N repeat runs.

    ``spread`` is the plain range (``maximum - minimum``) — the number an
    operator comparing two providers needs beside the mean to know whether
    the gap between them is bigger than the noise. ``stdev`` is the sample
    standard deviation, ``0.0`` at ``n == 1`` (undefined, reported as no
    observed spread rather than raising).
    """

    name: str
    runs: int
    values: list[float]
    mean: float
    minimum: float
    maximum: float
    spread: float
    stdev: float


def summarise_metric(name: str, values: list[float]) -> MetricStat:
    """Reduce one metric's per-run values to a :class:`MetricStat`."""
    if not values:
        raise ValueError(f"metric {name!r} has no values to summarise")
    return MetricStat(
        name=name,
        runs=len(values),
        values=list(values),
        mean=statistics.fmean(values),
        minimum=min(values),
        maximum=max(values),
        spread=max(values) - min(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


@dataclass
class AggregateBenchmarkReport:
    """N repeat runs of one suite × extractor, plus per-metric spread.

    This is a **wrapper**, not a variant of :class:`BenchmarkReport`: the
    §13.3 report model is frozen, so the aggregate holds the N reports
    verbatim in ``reports`` and adds the distribution beside them. Anything
    reading a single run keeps working on ``reports[i]`` unchanged.
    """

    suite_id: str
    suite_version: str
    extractor_id: str
    extractor_version: str
    runs: int
    metric_stats: dict[str, MetricStat]
    reports: list[BenchmarkReport]
    generated_at: datetime
    judge: str
    equivalence_threshold: float

    @property
    def mean_metrics(self) -> dict[str, float]:
        """Per-metric means — what ``--fail-on`` is evaluated against."""
        return {name: stat.mean for name, stat in self.metric_stats.items()}


@dataclass(frozen=True)
class BenchmarkRunEstimate:
    """Projected LLM cost of a (possibly repeated) benchmark run.

    Computed before any call is made, from the resolved case bytes and the
    same chunk math the memory harness's estimate uses.
    ``estimated_extraction_calls`` is a floor: an
    extractor may issue extra calls per source (subject resolution,
    classifier passes) that this cannot see.
    """

    suites: int
    cases: int
    runs: int
    total_chars: int
    estimated_extraction_calls: int
    estimated_tokens: int


def estimate_benchmark_run(
    suites: list[BenchmarkSuite],
    extractor: ExtractorPlugin,
    *,
    fixture_dir: Path,
    runs: int,
) -> BenchmarkRunEstimate:
    """Project extraction-call count + token volume for ``runs`` passes.

    Cases the extractor declines, or whose content will not resolve, are
    excluded — they cost nothing and the runner skips them with a quality
    note. Resolution reads local fixture bytes only; no network, no LLM.
    """
    cfg = get_config().extraction
    case_count = 0
    calls_per_pass = 0
    total_chars = 0
    for suite in suites:
        for case in suite.cases:
            try:
                _snapshot, content, source_type = resolve_case_content(case, fixture_dir)
            except SuiteLoadError:
                continue
            if source_type and not extractor.accepts(source_type):
                continue
            case_count += 1
            n = len(content)
            total_chars += n
            if n <= 0:
                continue
            if n <= cfg.html_chunk_size:
                calls_per_pass += 1
            else:
                calls_per_pass += min(-(-n // cfg.html_chunk_size), cfg.max_llm_calls_per_source)
    return BenchmarkRunEstimate(
        suites=len(suites),
        cases=case_count,
        runs=runs,
        total_chars=total_chars * runs,
        estimated_extraction_calls=calls_per_pass * runs,
        estimated_tokens=(total_chars * runs) // 4,
    )


def render_benchmark_estimate(estimate: BenchmarkRunEstimate) -> str:
    """Human rendering of the estimate — printed before any LLM call."""
    return (
        f"Estimate: {estimate.cases} case(s) across {estimate.suites} suite(s) × "
        f"{estimate.runs} run(s) → ≥{estimate.estimated_extraction_calls} extraction "
        f"call(s), ~{estimate.estimated_tokens:,} input tokens. Repeat runs are "
        f"deliberately uncached — measuring sampling variance is the point — so "
        f"cost scales linearly with --runs."
    )


async def run_benchmark_repeated(
    suite: BenchmarkSuite,
    extractor: ExtractorPlugin,
    *,
    runs: int,
    fixture_dir: Path,
    judge: EquivalenceJudge = EquivalenceJudge.EMBEDDING,
    threshold: float = 0.80,
    on_report: Callable[[int, BenchmarkReport], None] | None = None,
) -> AggregateBenchmarkReport:
    """Run one suite ``runs`` times and summarise each metric's spread.

    Every pass is an independent :func:`run_benchmark` call — no caching
    between them, because run-to-run sampling variance is exactly what the
    aggregate reports. ``on_report(index, report)`` fires as each
    pass lands, so a caller can stream progress or persist per-run artifacts
    without waiting for the whole series.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")

    reports: list[BenchmarkReport] = []
    for i in range(runs):
        report = await run_benchmark(
            suite,
            extractor,
            fixture_dir=fixture_dir,
            judge=judge,
            threshold=threshold,
        )
        reports.append(report)
        if on_report is not None:
            on_report(i, report)

    # Union of metric names, in first-seen order; a metric absent from some
    # run contributes only the runs that reported it (the reference runner
    # always emits the same three, but a fork's runner need not).
    names: list[str] = []
    for report in reports:
        for name in report.metrics:
            if name not in names:
                names.append(name)
    metric_stats = {
        name: summarise_metric(name, [r.metrics[name] for r in reports if name in r.metrics])
        for name in names
    }

    first = reports[0]
    return AggregateBenchmarkReport(
        suite_id=first.suite_id,
        suite_version=first.suite_version,
        extractor_id=first.extractor_id,
        extractor_version=first.extractor_version,
        runs=runs,
        metric_stats=metric_stats,
        reports=reports,
        generated_at=datetime.now(UTC),
        judge=judge.value,
        equivalence_threshold=threshold,
    )


def _record_emitted_claims(
    emitted: list[Particle],
    match: MatchResult,
    subject_names: dict[str, list[str]],
) -> list[EmittedClaim]:
    """Build one :class:`EmittedClaim` per emitted particle, in emission order.

    The judged outcome comes from ``match``, whose three buckets partition the
    emitted set — so a particle in none of them would be a bug in the
    assignment, not a fourth state; it is recorded ``SPURIOUS`` (the
    conservative reading: it matched no gold claim) rather than dropped, so
    the record stays one-per-emission and the count reconciles with
    ``emitted_count``.

    ``benchmark.record_claim_text`` is read here rather than at persist time
    on purpose. The text is suppressed at *production*, so an operator who
    turns it off is not relying on every downstream sink — the persisted run
    file, ``--format json`` on stdout, a test that pickles a report — to
    remember to scrub it. Nothing ever holds the text.

    It suppresses ``content`` only. ``matched_expected`` is *gold* text, read
    from the suite YAML, and the report already carries every gold string
    unconditionally in ``matched`` and ``missed_required`` — a flag that
    redacted it here while leaving those intact would suggest a protection it
    does not provide. An operator whose gold claims quote private material is
    already exposed by the pre-existing fields; the fix there is the suite
    file, not this switch.
    """
    record_text = get_config().benchmark.record_claim_text
    matched_gold: dict[str, str] = {p.id: e.content for e, p in match.matched}
    under_gold: dict[str, str] = {p.id: e.content for e, p in match.under_confidence}

    claims: list[EmittedClaim] = []
    for particle in emitted:
        if particle.id in matched_gold:
            outcome, gold = ClaimOutcome.MATCHED, matched_gold[particle.id]
        elif particle.id in under_gold:
            outcome, gold = ClaimOutcome.UNDER_CONFIDENCE, under_gold[particle.id]
        else:
            outcome, gold = ClaimOutcome.SPURIOUS, None
        claims.append(
            EmittedClaim(
                particle_id=particle.id,
                outcome=outcome,
                confidence=particle.confidence.value,
                content=particle.content if record_text else None,
                matched_expected=gold,
                subjects=list(subject_names.get(particle.id, ())) if record_text else [],
            )
        )
    return claims


async def run_benchmark(
    suite: BenchmarkSuite,
    extractor: ExtractorPlugin,
    *,
    fixture_dir: Path,
    judge: EquivalenceJudge = EquivalenceJudge.EMBEDDING,
    threshold: float = 0.80,
) -> BenchmarkReport:
    """Run one suite against one extractor and return the report.

    ``fixture_dir`` is the directory ``fixture:`` references resolve
    against — usually ``tests/conformance/fixtures/`` so the
    benchmark corpus reuses the conformance fixture corpus. Inline
    ``source_snapshot`` cases ignore this argument.

    Raises nothing on extractor errors — a case that raises is logged
    as a quality note and contributes zero matched + zero emitted
    particles to the rollup. This makes the report robust to one
    bad case mid-suite.
    """
    all_matched_ids: set[str] = set()
    all_emitted: list[Particle] = []
    per_case: list[CaseResult] = []
    quality_notes: list[str] = []
    cases_run = 0
    total_required = 0

    ext_ref = ExtractorRef(name=extractor.EXTRACTOR_ID, version=extractor.EXTRACTOR_VERSION)

    for case in suite.cases:
        # Resolve case → (snapshot, bytes, source_type)
        try:
            snapshot, content, source_type = resolve_case_content(case, fixture_dir)
        except SuiteLoadError as exc:
            quality_notes.append(f"Case {case.case_id}: {exc}")
            continue

        # Skip if extractor declines this source_type (still counts toward
        # cases_total — operator sees the skipped count in the report)
        if source_type and not extractor.accepts(source_type):
            quality_notes.append(
                f"Case {case.case_id}: extractor declines source_type {source_type!r}"
            )
            continue

        cases_run += 1
        total_required += sum(1 for e in case.expected if e.required)

        # Invoke the extractor; convert candidates to Particle instances.
        try:
            result = await extractor.extract(snapshot, content)
        except Exception as exc:
            quality_notes.append(f"Case {case.case_id}: extract() raised {exc!r}")
            per_case.append(
                CaseResult(
                    case_id=case.case_id,
                    emitted_count=0,
                    matched=[],
                    matched_required_count=0,
                    missed_required=[e.content for e in case.expected if e.required],
                    spurious=[],
                    under_confidence=[],
                )
            )
            continue

        emitted: list[Particle] = []
        # particle id -> the subject *names* the candidate declared. Built here
        # rather than read back off ``particle.subject_ids`` because that field
        # holds resolved UUIDs in any store-backed particle; see
        # ``match_emitted_to_expected``'s note on why the caller states names.
        subject_names: dict[str, list[str]] = {}
        for candidate in result.candidates:
            particle = candidate_to_particle(
                candidate,
                corpus_entry_id="benchmark-fixture-entry",
                snapshot_id=snapshot.snapshot_id,
                asserted_by=extractor.EXTRACTOR_ID,
                extractor_ref=ext_ref,
                subject_ids=list(candidate.subjects),
            )
            emitted.append(particle)
            subject_names[particle.id] = [
                name
                for name in (str(x).strip() for x in candidate.subjects)
                if name and not _QID_RE.match(name)
            ]
        quality_notes.extend(f"Case {case.case_id}: {n}" for n in result.quality_notes)

        # the emitted side is embedded as the particle asserts it —
        # subject prepended when ``content`` does not already name it. Off
        # restores the pre-0262 comparison, which is what every run file and
        # provider-survey page written before 1.140.0 was measured under.
        match: MatchResult = await match_emitted_to_expected(
            emitted,
            case.expected,
            judge=judge,
            threshold=threshold,
            subject_names=subject_names if get_config().benchmark.subject_aware_matching else None,
        )

        # Per-case detail
        case_matched_required = sum(1 for expected_p, _ in match.matched if expected_p.required)
        # the calibration label is *semantic match*, so it counts
        # under-confidence partial matches — which `matched_ids` deliberately
        # excludes. That exclusion is right for precision/recall (it is the
        # §13.3 under-trusting signal, given neither credit) and inverts the
        # question for calibration: it would label a correct claim incorrect
        # for the sole reason that it was stated timidly, teaching the
        # temperature that low confidence predicts error — the exact bias the
        # fit exists to remove. Only `graded` uses this set; the metrics below
        # keep `match.matched_ids` untouched.
        case_matched_ids = match.matched_ids
        case_semantic_ids = case_matched_ids | {p.id for _, p in match.under_confidence}
        emitted_claims = _record_emitted_claims(emitted, match, subject_names)
        per_case.append(
            CaseResult(
                case_id=case.case_id,
                emitted_count=len(emitted),
                matched=[
                    (expected_p.content, emitted_p.id) for expected_p, emitted_p in match.matched
                ],
                matched_required_count=case_matched_required,
                missed_required=[m.content for m in match.missed_required],
                spurious=[p.id for p in match.spurious],
                under_confidence=[
                    (expected_p.content, emitted_p.confidence.value, expected_p.confidence_min)
                    for expected_p, emitted_p in match.under_confidence
                ],
                graded=[(p.confidence.value, p.id in case_semantic_ids) for p in emitted],
                emitted_claims=emitted_claims,
            )
        )

        # Roll up for global metrics
        all_matched_ids |= match.matched_ids
        all_emitted.extend(emitted)

    matched_required = sum(c.matched_required_count for c in per_case)
    metrics: dict[str, float] = {
        "precision": compute_precision(len(all_matched_ids), len(all_emitted)),
        "recall": compute_recall(matched_required, total_required),
        "calibration_error": compute_calibration_error(all_matched_ids, all_emitted),
    }

    return BenchmarkReport(
        suite_id=suite.suite_id,
        suite_version=suite.version,
        extractor_id=extractor.EXTRACTOR_ID,
        extractor_version=extractor.EXTRACTOR_VERSION,
        cases_run=cases_run,
        cases_total=len(suite.cases),
        particles_emitted=len(all_emitted),
        particles_required_total=total_required,
        metrics=metrics,
        per_case=per_case,
        generated_at=datetime.now(UTC),
        judge=judge.value,
        equivalence_threshold=threshold,
        quality_notes=quality_notes,
    )
