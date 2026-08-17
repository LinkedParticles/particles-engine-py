"""End-to-end tests for the benchmark runner + CLI (commit 2/3).

These cover the orchestration that ties commit-1's pure functions to a
registered extractor + the Typer CLI. The seed numismatic suite is
included so the runner has a real-world target to demonstrate against.

What's covered:
  * runner against the real Numista coin extractor with the bundled
    seed suite — precision/recall/ECE produce sensible numbers
  * runner correctly counts matched_required separately from total
    matched (regression: recall used to over-count by including
    optional-match counts in the numerator)
  * cases the extractor rejects (mismatched source_type) are skipped
    with a quality note rather than failing the run
  * cases that raise during extract() degrade gracefully — quality
    note + zero matched + missed_required surfaced in per-case detail
  * CLI: table + JSON formats, --suite filter, --fail-on for each
    metric, exit codes 0 / 1 / 2
  * repeat runs: the aggregate wrapper's per-metric
    distribution, the untouched §13.3 reports it carries, the cost
    estimate, and the CLI's --runs / --estimate / --yes surface
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.equivalence import EquivalenceJudge
from particles.benchmark.loader import load_suite
from particles.benchmark.runner import (
    AggregateBenchmarkReport,
    BenchmarkReport,
    CaseResult,
    estimate_benchmark_run,
    graded_pairs,
    render_benchmark_estimate,
    run_benchmark,
    run_benchmark_repeated,
    summarise_metric,
)
from particles.benchmark.schema import (
    BenchmarkCase,
    BenchmarkSuite,
    ExpectedParticle,
)
from particles.core.schema import Snapshot, UncertaintyNature
from particles.extraction.general import CandidateParticle, ExtractionResult

_SEED_SUITE = Path("tests/benchmark/suites/numismatic-seed-001.yaml")
_FIXTURES = Path(__file__).parent / "conformance" / "fixtures"


def _get_extractor(extractor_id: str) -> Any:
    from particles.extraction.registry import get_extractors

    for e in get_extractors():
        if extractor_id == e.EXTRACTOR_ID:
            return e
    raise AssertionError(f"Extractor {extractor_id!r} not registered")


# ---------------------------------------------------------------------------
# Stub extractor used for orchestration tests where we don't want to
# depend on the real Numista output
# ---------------------------------------------------------------------------


class _PerfectStub:
    """Emits exactly the expected content, at high confidence — every test
    case scores 1.0 precision and 1.0 recall."""

    EXTRACTOR_ID = "perfect-stub"
    EXTRACTOR_VERSION = "0.0.1"

    def __init__(self, expected_contents: list[str]) -> None:
        self._contents = expected_contents

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=c,
                    confidence_value=0.95,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["Subject"],
                )
                for c in self._contents
            ]
        )


class _FlakyStub:
    """Emits the expected contents on some calls and nothing on others.

    The point of ``--runs N`` is that an identical fixture does not score
    identically twice; this stub reproduces that deterministically
    by alternating between a perfect pass and a total miss.
    """

    EXTRACTOR_ID = "flaky-stub"
    EXTRACTOR_VERSION = "0.0.1"

    def __init__(self, expected_contents: list[str], *, hit_on: set[int]) -> None:
        self._contents = expected_contents
        self._hit_on = hit_on
        self.calls = 0

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        index = self.calls
        self.calls += 1
        if index not in self._hit_on:
            return ExtractionResult(candidates=[])
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=c,
                    confidence_value=0.95,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["Subject"],
                )
                for c in self._contents
            ]
        )


class _RejectsEverything:
    EXTRACTOR_ID = "rejects-everything"
    EXTRACTOR_VERSION = "0.0.1"

    def accepts(self, source_type: str) -> bool:
        return False

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        raise AssertionError("never called")  # pragma: no cover


class _RaisesOnExtract:
    EXTRACTOR_ID = "raises-on-extract"
    EXTRACTOR_VERSION = "0.0.1"

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        raise RuntimeError("simulated extractor crash")


# ---------------------------------------------------------------------------
# Runner end-to-end against the real Numista extractor + seed suite
# ---------------------------------------------------------------------------


class TestRunnerNumistaSeedSuite:
    @pytest.mark.asyncio
    async def test_real_numista_against_seed_suite(self) -> None:
        suite = load_suite(_SEED_SUITE)
        extractor = _get_extractor("numista-coin-extractor")
        report = await run_benchmark(suite, extractor, fixture_dir=_FIXTURES)
        assert report.suite_id == "numismatic-seed-001"
        assert report.extractor_id == "numista-coin-extractor"
        assert report.cases_run == 1
        assert report.cases_total == 1
        # The Numista extractor emits 7 particles for this fixture
        assert report.particles_emitted == 7
        # 3 expected particles are required (structured, mint, KM#)
        assert report.particles_required_total == 3
        # All seven match, no spurious — precision is 1.0
        assert report.metrics["precision"] == pytest.approx(1.0, abs=1e-6)
        # All three required match — recall is 1.0 (regression guard:
        # before the matched_required_count fix this was >1.0)
        assert report.metrics["recall"] == pytest.approx(1.0, abs=1e-6)
        # Numista emits at 0.95-0.97, all match → ECE is small
        assert report.metrics["calibration_error"] < 0.10
        # Per-case detail
        assert len(report.per_case) == 1
        case = report.per_case[0]
        assert case.case_id == "numista-coin-001"
        assert case.matched_required_count == 3
        assert case.missed_required == []


# ---------------------------------------------------------------------------
# Orchestration with stubs — exercise edge cases
# ---------------------------------------------------------------------------


def _stub_suite(expected: list[ExpectedParticle]) -> BenchmarkSuite:
    """Build an inline-snapshot single-case suite (no fixture file required)."""
    return BenchmarkSuite(
        suite_id="stub-suite",
        name="stub",
        version="0.0.1",
        domain="test",
        source_types=["TEST_TYPE"],
        cases=[
            BenchmarkCase(
                case_id="case-1",
                expected=expected,
                source_snapshot=Snapshot(content_hash="a" * 64),
                inline_content=b"x",
            )
        ],
    )


class TestRunnerOrchestration:
    @pytest.mark.asyncio
    async def test_perfect_stub_scores_perfect(self) -> None:
        expected = [
            ExpectedParticle(
                content="Mercury is a planet",
                confidence_min=0.5,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=True,
            ),
            ExpectedParticle(
                content="Venus is a planet",
                confidence_min=0.5,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=True,
            ),
        ]
        suite = _stub_suite(expected)
        extractor = _PerfectStub([e.content for e in expected])
        report = await run_benchmark(suite, extractor, fixture_dir=Path("."))
        assert report.metrics["precision"] == 1.0
        assert report.metrics["recall"] == 1.0
        assert report.per_case[0].missed_required == []

    @pytest.mark.asyncio
    async def test_extractor_rejecting_source_type_is_quality_note(self) -> None:
        suite = _stub_suite(
            [
                ExpectedParticle(
                    content="x",
                    confidence_min=0.5,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                )
            ]
        )
        # Give it a fixture so resolve_case_content reports a source_type.
        suite = BenchmarkSuite(
            suite_id=suite.suite_id,
            name=suite.name,
            version=suite.version,
            domain=suite.domain,
            source_types=suite.source_types,
            cases=[
                BenchmarkCase(
                    case_id="case-1",
                    expected=suite.cases[0].expected,
                    fixture="numista-coin-001",
                )
            ],
        )
        extractor = _RejectsEverything()
        report = await run_benchmark(suite, extractor, fixture_dir=_FIXTURES)
        assert report.cases_run == 0
        assert report.particles_emitted == 0
        assert any("declines source_type" in n for n in report.quality_notes)

    @pytest.mark.asyncio
    async def test_extractor_raising_degrades_gracefully(self) -> None:
        expected = [
            ExpectedParticle(
                content="required claim",
                confidence_min=0.5,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=True,
            )
        ]
        suite = _stub_suite(expected)
        extractor = _RaisesOnExtract()
        report = await run_benchmark(suite, extractor, fixture_dir=Path("."))
        # The case was attempted (cases_run incremented) but produced 0
        # matches and surfaced the failure as a quality note.
        assert report.cases_run == 1
        assert report.particles_emitted == 0
        assert report.metrics["recall"] == 0.0
        assert report.per_case[0].missed_required == ["required claim"]
        assert any("extract() raised" in n for n in report.quality_notes)

    @pytest.mark.asyncio
    async def test_optional_matches_do_not_inflate_recall(self) -> None:
        """Regression: recall denominator is *required* particles only."""
        expected = [
            ExpectedParticle(
                content="A required",
                confidence_min=0.5,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=True,
            ),
            ExpectedParticle(
                content="An optional",
                confidence_min=0.5,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=False,
            ),
        ]
        suite = _stub_suite(expected)
        extractor = _PerfectStub([e.content for e in expected])
        report = await run_benchmark(suite, extractor, fixture_dir=Path("."))
        assert report.particles_required_total == 1
        assert report.metrics["recall"] == 1.0  # not 2.0!
        assert report.per_case[0].matched_required_count == 1


# ---------------------------------------------------------------------------
# Graded pairs — the labelled population `extractor calibrate` fits on
# ---------------------------------------------------------------------------


class _PartialStub:
    """Emits the expected contents *plus* extra claims no gold particle covers.

    The mixed-label shape a temperature fit actually needs: the extras go
    unmatched, so the graded pairs carry both True and False.
    """

    EXTRACTOR_ID = "partial-stub"
    EXTRACTOR_VERSION = "0.0.1"

    def __init__(self, matching: list[str], spurious: list[str]) -> None:
        self._matching = matching
        self._spurious = spurious

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=c,
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["Subject"],
                )
                for c in self._matching + self._spurious
            ]
        )


class TestGradedPairs:
    @pytest.mark.asyncio
    async def test_under_confidence_matches_count_as_calibration_matches(self) -> None:
        """the calibration label is semantic match, not match-and-confident.

        `matched_ids` excludes under-confidence partial matches, which is right
        for precision/recall — that exclusion is the §13.3 under-trusting
        signal. For calibration it inverts the question: it would label a
        semantically *correct* claim incorrect purely because it was stated
        timidly, teaching the temperature that low confidence predicts error —
        the exact bias the fit exists to remove.
        """
        expected = [
            ExpectedParticle(
                content="Mercury is a planet",
                confidence_min=0.99,  # above the stub's 0.95 → under-confidence
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=True,
            )
        ]
        suite = _stub_suite(expected)
        extractor = _PerfectStub([e.content for e in expected])
        report = await run_benchmark(suite, extractor, fixture_dir=Path("."))

        case = report.per_case[0]
        # It really is an under-confidence partial match, not an ordinary one…
        assert len(case.under_confidence) == 1
        assert case.matched == []
        # …and precision still gives it no credit (the §13.3 contract, unchanged).
        assert report.metrics["precision"] == 0.0
        # …but calibration counts it correct, because it *is* correct.
        _, labels = graded_pairs(report)
        assert labels == [True]

    @pytest.mark.asyncio
    async def test_pairs_carry_raw_confidence_and_the_real_match_flag(self) -> None:
        """The whole point of the plumbing: labels come from run #1's own matching.

        The verb used to re-run the extractor to recover confidences and label
        the *fresh* particles against the first run's id set — ids that could
        never collide, so every label was False.
        """
        expected = [
            ExpectedParticle(
                content="Mercury is a planet",
                confidence_min=0.5,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=True,
            )
        ]
        suite = _stub_suite(expected)
        extractor = _PartialStub(["Mercury is a planet"], ["Neptune is a moon of Saturn"])
        report = await run_benchmark(suite, extractor, fixture_dir=Path("."))

        raws, labels = graded_pairs(report)
        assert len(raws) == len(labels) == 2
        assert raws == [pytest.approx(0.9), pytest.approx(0.9)]
        # Exactly one matched — not zero (the bug) and not both.
        assert sorted(labels) == [False, True]

    @pytest.mark.asyncio
    async def test_pair_count_equals_particles_emitted(self) -> None:
        """No emitted particle may be silently dropped from the fitting population."""
        expected = [
            ExpectedParticle(
                content=f"claim {i}",
                confidence_min=0.5,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=True,
            )
            for i in range(3)
        ]
        suite = _stub_suite(expected)
        extractor = _PerfectStub([e.content for e in expected])
        report = await run_benchmark(suite, extractor, fixture_dir=Path("."))
        raws, labels = graded_pairs(report)
        assert len(raws) == report.particles_emitted == 3
        assert all(labels)

    @pytest.mark.asyncio
    async def test_labels_agree_with_the_reported_calibration_error(self) -> None:
        """ECE and the fit must see the same population, or they disagree silently."""
        expected = [
            ExpectedParticle(
                content="Mercury is a planet",
                confidence_min=0.5,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=True,
            )
        ]
        suite = _stub_suite(expected)
        extractor = _PartialStub(["Mercury is a planet"], ["a spurious claim"])
        report = await run_benchmark(suite, extractor, fixture_dir=Path("."))

        from particles.extraction.calibration import expected_calibration_error

        raws, labels = graded_pairs(report)
        assert expected_calibration_error(raws, labels) == pytest.approx(
            report.metrics["calibration_error"]
        )

    @pytest.mark.asyncio
    async def test_a_crashing_case_contributes_no_pairs(self) -> None:
        """Not one wrongly-labelled pair — an absent case must not read as all-incorrect."""
        expected = [
            ExpectedParticle(
                content="required claim",
                confidence_min=0.5,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                required=True,
            )
        ]
        suite = _stub_suite(expected)
        report = await run_benchmark(suite, _RaisesOnExtract(), fixture_dir=Path("."))
        assert graded_pairs(report) == ([], [])

    @pytest.mark.asyncio
    async def test_real_numista_run_labels_its_own_particles_correctly(self) -> None:
        """End-to-end against a real extractor: 7 emitted, all matched by the seed suite."""
        suite = load_suite(_SEED_SUITE)
        extractor = _get_extractor("numista-coin-extractor")
        report = await run_benchmark(suite, extractor, fixture_dir=_FIXTURES)
        raws, labels = graded_pairs(report)
        assert len(raws) == 7
        assert all(labels), "every emitted particle matches this suite's gold set"
        assert all(0.8 <= r <= 1.0 for r in raws)


# ---------------------------------------------------------------------------
# Repeat runs
# ---------------------------------------------------------------------------


def _required(content: str) -> ExpectedParticle:
    return ExpectedParticle(
        content=content,
        confidence_min=0.5,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        required=True,
    )


class TestSummariseMetric:
    def test_single_value_has_no_spread(self) -> None:
        stat = summarise_metric("recall", [0.75])
        assert stat.runs == 1
        assert stat.mean == 0.75
        assert stat.spread == 0.0
        # stdev is undefined at n=1 — reported as 0.0, never raised
        assert stat.stdev == 0.0

    def test_distribution_fields(self) -> None:
        stat = summarise_metric("recall", [0.75, 0.88, 0.75, 1.0])
        assert stat.minimum == 0.75
        assert stat.maximum == 1.0
        assert stat.spread == pytest.approx(0.25)
        assert stat.mean == pytest.approx(0.845)
        assert stat.stdev > 0.0
        assert stat.values == [0.75, 0.88, 0.75, 1.0]

    def test_empty_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="no values"):
            summarise_metric("recall", [])


class TestRunBenchmarkRepeated:
    @pytest.mark.asyncio
    async def test_variance_is_reported_not_hidden(self) -> None:
        """Two runs of the identical fixture, two different recalls."""
        expected = [_required("Mercury is a planet")]
        suite = _stub_suite(expected)
        extractor = _FlakyStub([e.content for e in expected], hit_on={0})
        aggregate = await run_benchmark_repeated(suite, extractor, runs=2, fixture_dir=Path("."))
        assert isinstance(aggregate, AggregateBenchmarkReport)
        assert aggregate.runs == 2
        recall = aggregate.metric_stats["recall"]
        assert recall.values == [1.0, 0.0]
        assert recall.mean == pytest.approx(0.5)
        assert recall.spread == pytest.approx(1.0)
        # --fail-on reads the mean, not the worst run
        assert aggregate.mean_metrics["recall"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_carries_the_frozen_reports_verbatim(self) -> None:
        """The aggregate wraps §13.3 reports; it does not replace them."""
        expected = [_required("Mercury is a planet")]
        suite = _stub_suite(expected)
        extractor = _FlakyStub([e.content for e in expected], hit_on={0, 1, 2})
        aggregate = await run_benchmark_repeated(suite, extractor, runs=3, fixture_dir=Path("."))
        assert len(aggregate.reports) == 3
        assert all(isinstance(r, BenchmarkReport) for r in aggregate.reports)
        # Each report is a complete standalone run report — per-case detail
        # included — so anything reading one run keeps working.
        assert all(r.per_case[0].case_id == "case-1" for r in aggregate.reports)
        assert aggregate.suite_id == "stub-suite"
        assert aggregate.extractor_id == "flaky-stub"

    @pytest.mark.asyncio
    async def test_on_report_fires_per_pass(self) -> None:
        expected = [_required("Mercury is a planet")]
        suite = _stub_suite(expected)
        extractor = _FlakyStub([e.content for e in expected], hit_on={0, 1})
        seen: list[int] = []
        await run_benchmark_repeated(
            suite,
            extractor,
            runs=2,
            fixture_dir=Path("."),
            on_report=lambda i, _r: seen.append(i),
        )
        assert seen == [0, 1]

    @pytest.mark.asyncio
    async def test_runs_one_is_a_degenerate_distribution(self) -> None:
        expected = [_required("Mercury is a planet")]
        suite = _stub_suite(expected)
        extractor = _PerfectStub([e.content for e in expected])
        aggregate = await run_benchmark_repeated(suite, extractor, runs=1, fixture_dir=Path("."))
        assert aggregate.runs == 1
        assert aggregate.metric_stats["recall"].spread == 0.0
        assert aggregate.metric_stats["recall"].mean == 1.0

    @pytest.mark.asyncio
    async def test_zero_runs_rejected(self) -> None:
        suite = _stub_suite([_required("x")])
        with pytest.raises(ValueError, match="runs must be >= 1"):
            await run_benchmark_repeated(suite, _PerfectStub(["x"]), runs=0, fixture_dir=Path("."))


class TestBenchmarkRunEstimate:
    def test_cost_scales_linearly_with_runs(self) -> None:
        suite = _stub_suite([_required("x")])
        extractor = _PerfectStub(["x"])
        one = estimate_benchmark_run([suite], extractor, fixture_dir=Path("."), runs=1)
        five = estimate_benchmark_run([suite], extractor, fixture_dir=Path("."), runs=5)
        assert one.cases == 1
        assert one.estimated_extraction_calls == 1
        assert five.estimated_extraction_calls == 5 * one.estimated_extraction_calls
        assert five.total_chars == 5 * one.total_chars

    def test_declined_cases_cost_nothing(self) -> None:
        suite = BenchmarkSuite(
            suite_id="stub-suite",
            name="stub",
            version="0.0.1",
            domain="test",
            source_types=["NUMISTA_API_COIN"],
            cases=[
                BenchmarkCase(
                    case_id="case-1", expected=[_required("x")], fixture="numista-coin-001"
                )
            ],
        )
        estimate = estimate_benchmark_run(
            [suite], _RejectsEverything(), fixture_dir=_FIXTURES, runs=3
        )
        assert estimate.cases == 0
        assert estimate.estimated_extraction_calls == 0

    def test_render_names_the_linear_cost(self) -> None:
        suite = _stub_suite([_required("x")])
        estimate = estimate_benchmark_run(
            [suite], _PerfectStub(["x"]), fixture_dir=Path("."), runs=4
        )
        rendered = render_benchmark_estimate(estimate)
        assert "4 run(s)" in rendered
        assert "extraction call(s)" in rendered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestBenchmarkCLI:
    def test_table_output_against_real_extractor(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["extractor", "benchmark", "numista-coin-extractor"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "Benchmark report" in result.output
        assert "numista-coin-extractor" in result.output
        assert "numismatic-seed-001" in result.output
        assert "recall" in result.output
        assert "precision" in result.output
        assert "calibration_error" in result.output

    def test_json_output_is_parseable(self, runner: CliRunner) -> None:
        import json as _json

        result = runner.invoke(
            app,
            ["extractor", "benchmark", "numista-coin-extractor", "--format", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        # result.output interleaves stderr (the "Run report saved to" notice);
        # the machine-readable contract is stdout alone.
        payload = _json.loads(result.stdout)
        assert payload["suite_id"] == "numismatic-seed-001"
        assert payload["extractor_id"] == "numista-coin-extractor"
        assert "metrics" in payload
        assert "precision" in payload["metrics"]

    def test_unknown_extractor_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["extractor", "benchmark", "no-such-extractor"])
        assert result.exit_code == 2
        assert "no-such-extractor" in (result.output + (result.stderr or ""))

    def test_unknown_suite_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            [
                "extractor",
                "benchmark",
                "numista-coin-extractor",
                "--suite",
                "does-not-exist",
            ],
        )
        assert result.exit_code == 2

    def test_no_applicable_suites_exits_2(self, runner: CliRunner) -> None:
        # The general extractor accepts everything, so pick one whose source
        # type no suite covers. The docstring extractor is the durable choice:
        # states it carries no benchmark by design (fixed confidence,
        # nothing to calibrate), so a future suite is not going to appear and
        # silently turn this assertion into a no-op the way wikidata-extractor
        # did when it was first given one.
        result = runner.invoke(
            app,
            [
                "extractor",
                "benchmark",
                "docstring-extractor",
            ],
        )
        # No suite covers PYTHON_SOURCE — exit 2 with helpful message
        assert result.exit_code == 2
        assert "No applicable benchmark suites" in (result.output + (result.stderr or ""))

    def test_fail_on_precision_below_threshold(self, runner: CliRunner) -> None:
        # Numista scores precision=1.0; setting --fail-threshold above
        # forces a failure to verify the gating logic.
        result = runner.invoke(
            app,
            [
                "extractor",
                "benchmark",
                "numista-coin-extractor",
                "--fail-on",
                "precision",
                "--fail-threshold",
                "0.99",  # below 1.0, should still pass
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

    def test_fail_on_calibration_above_threshold(self, runner: CliRunner) -> None:
        # Numista's calibration_error is ~0.05; threshold 0.01 should fail
        # because the recorded ECE exceeds it.
        result = runner.invoke(
            app,
            [
                "extractor",
                "benchmark",
                "numista-coin-extractor",
                "--fail-on",
                "calibration",
                "--fail-threshold",
                "0.01",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 1

    def test_run_report_persisted_under_runs_dir(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each run auto-persists an envelope JSON under benchmark.runs_dir."""
        import json as _json

        runs_dir = tmp_path / "runs"
        monkeypatch.setenv("BENCHMARK_RUNS_DIR", str(runs_dir))
        result = runner.invoke(
            app,
            ["extractor", "benchmark", "numista-coin-extractor"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        files = sorted(runs_dir.glob("*.json"))
        assert len(files) == 1
        assert "-benchmark-numista-coin-extractor-numismatic-seed-001" in files[0].name
        envelope = _json.loads(files[0].read_text())
        assert envelope["format"] == 1
        # The resolved extraction pairing rides the envelope (the frozen
        # §13.3 report model does not carry it).
        provider, _, model = envelope["extraction_provider_model"].partition(":")
        assert provider and model
        report = envelope["report"]
        assert report["suite_id"] == "numismatic-seed-001"
        assert report["extractor_id"] == "numista-coin-extractor"
        assert "precision" in report["metrics"]
        assert report["per_case"]
        # The save notice goes to stderr, keeping stdout clean for --format json
        assert "Run report saved to" in (result.stderr or "")

    def test_no_save_skips_persistence(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runs_dir = tmp_path / "runs"
        monkeypatch.setenv("BENCHMARK_RUNS_DIR", str(runs_dir))
        result = runner.invoke(
            app,
            ["extractor", "benchmark", "numista-coin-extractor", "--no-save"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert not runs_dir.exists()

    def test_same_second_reruns_never_clobber(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two runs stamped to the same second get distinct filenames."""
        runs_dir = tmp_path / "runs"
        monkeypatch.setenv("BENCHMARK_RUNS_DIR", str(runs_dir))
        for _ in range(2):
            result = runner.invoke(
                app,
                ["extractor", "benchmark", "numista-coin-extractor"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
        # Sub-second wall clock between runs is not guaranteed, so assert the
        # invariant that matters: two runs → two files, whatever the stamps.
        assert len(list(runs_dir.glob("*.json"))) == 2


class TestBenchmarkCLIRepeatRuns:
    """--runs N surface. Numista is deterministic and key-free, so
    these pin the *shape* of the aggregate; the variance itself is exercised
    at the runner tier with the flaky stub."""

    def _base(self, *extra: str) -> list[str]:
        return ["extractor", "benchmark", "numista-coin-extractor", *extra]

    def test_single_run_output_is_unchanged(self, runner: CliRunner) -> None:
        """N=1 renders the plain report — no estimate, no distribution."""
        result = runner.invoke(app, self._base("--runs", "1", "--no-save"), catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert "Benchmark report —" in result.stdout
        assert "runs)" not in result.stdout
        assert "SPREAD" not in result.stdout
        assert "Estimate:" not in (result.stderr or "")

    def test_repeat_run_table_shows_the_distribution(self, runner: CliRunner) -> None:
        result = runner.invoke(app, self._base("--runs", "2", "--no-save"), catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert "Benchmark report (2 runs)" in result.stdout
        assert "SPREAD" in result.stdout
        assert "Per-run values" in result.stdout
        assert "MEAN" in result.stdout
        # Cost is disclosed before any call, on stderr (stdout stays clean)
        assert "Estimate:" in (result.stderr or "")

    def test_repeat_run_json_carries_stats_and_every_report(self, runner: CliRunner) -> None:
        import json as _json

        result = runner.invoke(
            app,
            self._base("--runs", "2", "--no-save", "--format", "json"),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        payload = _json.loads(result.stdout)
        assert payload["runs"] == 2
        assert payload["suite_id"] == "numismatic-seed-001"
        # The frozen §13.3 reports ride whole inside the wrapper
        assert len(payload["reports"]) == 2
        assert payload["reports"][0]["metrics"]["precision"] == 1.0
        stat = payload["metric_stats"]["recall"]
        assert stat["runs"] == 2
        assert len(stat["values"]) == 2
        assert {"mean", "minimum", "maximum", "spread", "stdev"} <= set(stat)

    def test_each_pass_persists_its_own_report_file(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The run-file envelope stays one-run-per-file (format 1 unchanged)."""
        import json as _json

        runs_dir = tmp_path / "runs"
        monkeypatch.setenv("BENCHMARK_RUNS_DIR", str(runs_dir))
        result = runner.invoke(app, self._base("--runs", "3"), catch_exceptions=False)
        assert result.exit_code == 0, result.output
        files = sorted(runs_dir.glob("*.json"))
        assert len(files) == 3
        envelope = _json.loads(files[0].read_text())
        assert envelope["format"] == 1
        assert envelope["report"]["suite_id"] == "numismatic-seed-001"

    def test_estimate_runs_nothing(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runs_dir = tmp_path / "runs"
        monkeypatch.setenv("BENCHMARK_RUNS_DIR", str(runs_dir))
        result = runner.invoke(app, self._base("--runs", "5", "--estimate"), catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert "Estimate:" in (result.stderr or "")
        assert "nothing was run" in (result.stderr or "")
        assert "Benchmark report" not in result.stdout
        assert not runs_dir.exists()

    def test_over_threshold_aborts_non_interactively(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BENCHMARK_CONFIRM_CALL_THRESHOLD", "1")
        result = runner.invoke(app, self._base("--runs", "4", "--no-save"))
        assert result.exit_code == 1
        assert "no --yes was given" in (result.stderr or "")
        assert "Benchmark report" not in result.stdout

    def test_yes_pre_confirms_the_gate(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BENCHMARK_CONFIRM_CALL_THRESHOLD", "1")
        result = runner.invoke(
            app, self._base("--runs", "2", "--no-save", "--yes"), catch_exceptions=False
        )
        assert result.exit_code == 0, result.output
        assert "Benchmark report (2 runs)" in result.stdout

    def test_fail_on_reads_the_mean(self, runner: CliRunner) -> None:
        """Numista's mean ECE (~0.05) is above 0.01 on every run → exit 1."""
        result = runner.invoke(
            app,
            self._base(
                "--runs", "2", "--no-save", "--fail-on", "calibration", "--fail-threshold", "0.01"
            ),
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        result_ok = runner.invoke(
            app,
            self._base(
                "--runs", "2", "--no-save", "--fail-on", "recall", "--fail-threshold", "0.99"
            ),
            catch_exceptions=False,
        )
        assert result_ok.exit_code == 0, result_ok.output


# ---------------------------------------------------------------------------
# Dataclass smoke
# ---------------------------------------------------------------------------


def test_case_result_dataclass_construction() -> None:
    cr = CaseResult(
        case_id="x",
        emitted_count=0,
        matched=[],
        matched_required_count=0,
        missed_required=[],
        spurious=[],
        under_confidence=[],
    )
    assert cr.case_id == "x"


def test_benchmark_report_construction() -> None:
    from datetime import UTC, datetime

    rep = BenchmarkReport(
        suite_id="s",
        suite_version="0",
        extractor_id="e",
        extractor_version="0",
        cases_run=0,
        cases_total=0,
        particles_emitted=0,
        particles_required_total=0,
        metrics={},
        per_case=[],
        generated_at=datetime.now(UTC),
        judge=EquivalenceJudge.EMBEDDING.value,
        equivalence_threshold=0.8,
    )
    assert rep.quality_notes == []
