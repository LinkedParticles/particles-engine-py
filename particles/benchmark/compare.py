"""Multi-extractor benchmark comparison.

The :func:`compare_benchmarks` helper aggregates per-extractor
:class:`particles.benchmark.runner.BenchmarkReport` results into a
:class:`BenchmarkComparison` matrix. Pure function — no I/O beyond what
:func:`particles.benchmark.runner.run_benchmark` already performs.

The helper is the library-surface answer to the deferred
"single-extractor at a time" harness limitation. The CLI verb
``particles extractor benchmark-compare`` is a thin renderer over this
helper; the project's own pytest suite uses the helper directly for
cross-extractor regression pins.

Cells where an extractor declined a suite's ``source_type`` are
represented as ``None`` — distinguishing "did not run" from a real zero
metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from particles.benchmark.equivalence import EquivalenceJudge
from particles.benchmark.runner import BenchmarkReport, run_benchmark
from particles.benchmark.schema import BenchmarkSuite
from particles.extraction.registry import ExtractorPlugin

_METRIC_NAMES: tuple[str, ...] = ("recall", "precision", "calibration_error")


class SuiteComparison(BaseModel):
    """Per-suite comparison: one row per metric, one column per extractor.

    ``metrics`` is a nested mapping ``{metric_name: {extractor_id: value}}``.
    ``value`` is ``None`` when the extractor declined the suite's
    ``source_type`` (recorded by :func:`run_benchmark` as a quality note
    + zero cases run).

    ``reports`` carries the raw per-extractor :class:`BenchmarkReport`
    so test suites can drill into per-case detail (matched / spurious /
    missed_required) without re-running the suite.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    suite_id: str
    suite_version: str
    metrics: dict[str, dict[str, float | None]]
    reports: dict[str, BenchmarkReport] = Field(default_factory=dict)

    @field_serializer("reports")
    def _serialize_reports(self, value: dict[str, BenchmarkReport]) -> dict[str, dict[str, Any]]:
        # BenchmarkReport is a stdlib dataclass; asdict() turns it into a
        # nested dict so Pydantic can emit valid JSON via model_dump_json.
        return {extractor_id: asdict(report) for extractor_id, report in value.items()}


class BenchmarkComparison(BaseModel):
    """Top-level result of :func:`compare_benchmarks`.

    ``extractor_ids`` preserves the caller-supplied column order so
    operators control which extractor lands in the leftmost (baseline)
    column.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    judge: str
    threshold: float
    extractor_ids: list[str]
    suites: list[SuiteComparison]
    generated_at: datetime


async def compare_benchmarks(
    suite: BenchmarkSuite,
    extractors: Sequence[ExtractorPlugin],
    *,
    fixture_dir: Path,
    judge: EquivalenceJudge = EquivalenceJudge.EMBEDDING,
    threshold: float = 0.80,
) -> BenchmarkComparison:
    """Run one suite against multiple extractors and aggregate the results.

    Iterates ``extractors`` in caller order — that ordering is preserved
    in :attr:`BenchmarkComparison.extractor_ids` and in every
    ``metrics`` inner dict so consumers can rely on column stability.

    Extractors whose ``accepts(source_type)`` rejects every case in the
    suite still appear in the output — every metric cell is ``None``,
    making "extractor X declines suite Y" visible to operators rather
    than silently dropped.

    The function does not catch exceptions raised by
    :func:`run_benchmark` itself — that function is already designed to
    swallow per-case failures into quality notes. A raised exception
    here would mean the harness itself is broken (e.g. fixture path
    invalid) and should bubble up.
    """
    extractor_ids = [e.EXTRACTOR_ID for e in extractors]
    reports_by_extractor: dict[str, BenchmarkReport] = {}
    for extractor in extractors:
        reports_by_extractor[extractor.EXTRACTOR_ID] = await run_benchmark(
            suite,
            extractor,
            fixture_dir=fixture_dir,
            judge=judge,
            threshold=threshold,
        )

    metrics: dict[str, dict[str, float | None]] = {
        name: {extractor_id: None for extractor_id in extractor_ids} for name in _METRIC_NAMES
    }
    for extractor_id, report in reports_by_extractor.items():
        # cases_run == 0 means every case was declined — surface that
        # as None cells so an operator can spot "extractor didn't run
        # this suite" without parsing the quality notes.
        if report.cases_run == 0:
            continue
        for name in _METRIC_NAMES:
            value = report.metrics.get(name)
            if value is not None:
                metrics[name][extractor_id] = float(value)

    suite_comparison = SuiteComparison(
        suite_id=suite.suite_id,
        suite_version=suite.version,
        metrics=metrics,
        reports=reports_by_extractor,
    )

    return BenchmarkComparison(
        judge=judge.value,
        threshold=threshold,
        extractor_ids=extractor_ids,
        suites=[suite_comparison],
        generated_at=datetime.now(UTC),
    )
