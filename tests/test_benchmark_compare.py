"""Tests for the multi-extractor benchmark comparison helper.

The library helper (``particles.benchmark.compare.compare_benchmarks``)
is the test-accessible seam: this file exercises it directly with stub
extractors so we can assert matrix shape, column-order preservation,
and the ``None`` semantics for declined source_types without spinning
up the real Numista LLM pipeline.

One CliRunner-based smoke test confirms the ``benchmark-compare``
subcommand wires the helper into the CLI and emits valid JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.compare import (
    BenchmarkComparison,
    compare_benchmarks,
)
from particles.benchmark.equivalence import EquivalenceJudge
from particles.benchmark.schema import (
    BenchmarkCase,
    BenchmarkSuite,
    ExpectedParticle,
)
from particles.core.schema import Snapshot, UncertaintyNature
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.registry import ExtractorPlugin

_SEED_SUITE = Path("tests/benchmark/suites/numismatic-seed-001.yaml")
_FIXTURES = Path(__file__).parent / "conformance" / "fixtures"


# ---------------------------------------------------------------------------
# Stub extractors — small enough that the comparison matrix is human-verifiable
# ---------------------------------------------------------------------------


class _PerfectStub:
    """Emits exactly the expected content at high confidence."""

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


class _WeakerStub:
    """Emits only the first expected particle — partial recall."""

    EXTRACTOR_ID = "weaker-stub"
    EXTRACTOR_VERSION = "0.0.1"

    def __init__(self, expected_contents: list[str]) -> None:
        self._contents = expected_contents[:1]

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


class _DeclinesEverything:
    """Refuses every source_type — should produce None cells."""

    EXTRACTOR_ID = "declines-everything"
    EXTRACTOR_VERSION = "0.0.1"

    def accepts(self, source_type: str) -> bool:
        return False

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        raise AssertionError("never called")  # pragma: no cover


def _make_inline_suite() -> BenchmarkSuite:
    """Build a tiny inline suite with two expected particles."""
    expected = [
        ExpectedParticle(
            content="The capital of France is Paris.",
            confidence_min=0.80,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            required=True,
        ),
        ExpectedParticle(
            content="The Eiffel Tower opened in 1889.",
            confidence_min=0.80,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            required=True,
        ),
    ]
    case = BenchmarkCase(
        case_id="france-001",
        expected=expected,
        source_snapshot=Snapshot(content_hash="0" * 64),
        inline_content=b"Dummy content; the stub extractor ignores it.",
    )
    return BenchmarkSuite(
        suite_id="france-test-001",
        name="France inline test",
        version="0.1.0",
        domain="general-knowledge",
        source_types=["TEST"],
        cases=[case],
    )


# ---------------------------------------------------------------------------
# compare_benchmarks() — happy path + ordering + None semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compare_benchmarks_returns_matrix_for_two_extractors() -> None:
    suite = _make_inline_suite()
    expected_contents = [e.content for e in suite.cases[0].expected]
    extractors: list[ExtractorPlugin] = [
        _PerfectStub(expected_contents),
        _WeakerStub(expected_contents),
    ]

    comparison = await compare_benchmarks(
        suite, extractors, fixture_dir=_FIXTURES, judge=EquivalenceJudge.EMBEDDING
    )

    assert isinstance(comparison, BenchmarkComparison)
    assert comparison.extractor_ids == ["perfect-stub", "weaker-stub"]
    assert len(comparison.suites) == 1
    suite_comp = comparison.suites[0]
    assert suite_comp.suite_id == "france-test-001"
    # Perfect stub should have full recall; weaker stub should have lower recall.
    perfect_recall = suite_comp.metrics["recall"]["perfect-stub"]
    weaker_recall = suite_comp.metrics["recall"]["weaker-stub"]
    assert perfect_recall is not None and weaker_recall is not None
    assert perfect_recall == pytest.approx(1.0)
    assert weaker_recall < perfect_recall
    # All three normative metrics are reported for both extractors.
    for metric in ("recall", "precision", "calibration_error"):
        for eid in ("perfect-stub", "weaker-stub"):
            assert eid in suite_comp.metrics[metric]


@pytest.mark.asyncio
async def test_compare_benchmarks_preserves_caller_column_order() -> None:
    """The order of `extractors` is preserved verbatim in `extractor_ids`
    and in every metric inner dict — operators rely on this for choosing
    the baseline (leftmost) column."""
    suite = _make_inline_suite()
    expected_contents = [e.content for e in suite.cases[0].expected]
    # Pass them in reverse-alphabetical order to confirm we don't sort.
    extractors: list[ExtractorPlugin] = [
        _WeakerStub(expected_contents),
        _PerfectStub(expected_contents),
    ]

    comparison = await compare_benchmarks(suite, extractors, fixture_dir=_FIXTURES)

    assert comparison.extractor_ids == ["weaker-stub", "perfect-stub"]
    for metric in ("recall", "precision", "calibration_error"):
        # The dict literal order matches caller input order.
        assert list(comparison.suites[0].metrics[metric].keys()) == [
            "weaker-stub",
            "perfect-stub",
        ]


@pytest.mark.asyncio
async def test_compare_benchmarks_emits_none_when_extractor_declines_source_type() -> None:
    """An extractor whose `accepts()` returns False for every case still
    appears as a column — every metric cell is None so operators see
    'did not run' rather than a silently-dropped extractor.

    Uses the seed numismatic suite because the decline-path check
    requires a non-empty source_type, which is only set when the
    BenchmarkCase resolves via the `fixture:` form (inline cases return
    source_type="" and bypass the accepts() filter)."""
    from particles.benchmark.loader import load_suite

    suite = load_suite(_SEED_SUITE)
    second_stub = _DeclinesEverything()
    # Rename the second instance so the dict-keying works.
    second_stub.EXTRACTOR_ID = "declines-everything-2"
    extractors: list[ExtractorPlugin] = [_DeclinesEverything(), second_stub]

    comparison = await compare_benchmarks(suite, extractors, fixture_dir=_FIXTURES)

    assert comparison.extractor_ids == ["declines-everything", "declines-everything-2"]
    # Every cell is None because every extractor declined every case.
    for metric in ("recall", "precision", "calibration_error"):
        for eid in comparison.extractor_ids:
            assert comparison.suites[0].metrics[metric][eid] is None


@pytest.mark.asyncio
async def test_compare_benchmarks_model_dump_json_serialises_embedded_reports() -> None:
    """`reports` field holds BenchmarkReport dataclasses; the
    field_serializer must convert them to dicts so model_dump_json
    emits valid JSON without TypeErrors."""
    suite = _make_inline_suite()
    expected_contents = [e.content for e in suite.cases[0].expected]
    extractors: list[ExtractorPlugin] = [
        _PerfectStub(expected_contents),
        _WeakerStub(expected_contents),
    ]

    comparison = await compare_benchmarks(suite, extractors, fixture_dir=_FIXTURES)

    blob = comparison.model_dump_json()
    parsed = json.loads(blob)
    assert parsed["extractor_ids"] == ["perfect-stub", "weaker-stub"]
    # Each suite carries a `reports` dict keyed by extractor_id with
    # nested case detail — proof the field_serializer ran.
    reports = parsed["suites"][0]["reports"]
    assert set(reports) == {"perfect-stub", "weaker-stub"}
    assert reports["perfect-stub"]["cases_run"] == 1


# ---------------------------------------------------------------------------
# CLI smoke — wire the helper into `extractor benchmark-compare`
# ---------------------------------------------------------------------------


def test_benchmark_compare_cli_rejects_single_extractor_id() -> None:
    """One --extractor-id is a usage error directing the operator at the
    single-extractor verb."""
    runner = CliRunner()
    result = runner.invoke(
        app, ["extractor", "benchmark-compare", "--extractor-id", "numista-coin-extractor"]
    )
    assert result.exit_code == 2
    assert "at least two --extractor-id" in result.stderr or "at least two --extractor-id" in (
        result.output
    )


def test_benchmark_compare_cli_rejects_unknown_extractor_id() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "extractor",
            "benchmark-compare",
            "--extractor-id",
            "numista-coin-extractor",
            "--extractor-id",
            "does-not-exist",
        ],
    )
    assert result.exit_code == 2
    assert "Unknown extractor_id" in result.stderr or "Unknown extractor_id" in result.output


def test_benchmark_compare_cli_emits_json_for_two_real_extractors() -> None:
    """End-to-end: run two registered extractors against the seed suite
    via the CLI, parse the JSON output, confirm the matrix shape."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "extractor",
            "benchmark-compare",
            "--extractor-id",
            "numista-coin-extractor",
            "--extractor-id",
            "general-extractor",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    # The JSON output starts after any log lines the runner emitted.
    # Find the first '{' that begins a JSON object.
    json_start = result.output.find("{")
    assert json_start >= 0, f"No JSON object found in:\n{result.output}"
    parsed = json.loads(result.output[json_start:])
    assert parsed["extractor_ids"] == ["numista-coin-extractor", "general-extractor"]
    # We don't pin specific numbers (they depend on the real Numista
    # output) — only the shape.
    assert len(parsed["suites"]) >= 1
    metrics = parsed["suites"][0]["metrics"]
    assert set(metrics) == {"recall", "precision", "calibration_error"}
    for metric_name in metrics:
        assert set(metrics[metric_name]) == {
            "numista-coin-extractor",
            "general-extractor",
        }
