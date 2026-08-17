"""Tests for the event-anchored-validity benchmark.

Covers the unit tier — no LLM, no real embeddings:
  * pure metrics: the headline wrong-expiry rate, existence precision/recall,
    date accuracy, support counts, and their vacuous-denominator conventions
  * loader: the bundled seed suite round-trips; error paths (missing fields,
    the durable⇔no-boundary consistency invariant, unknown keys, empty labels,
    bad dates) raise; discovery skips malformed files but yields the good ones
  * runner orchestration: a stub extractor + a monkeypatched similarity matrix
    drive the full accounting deterministically (mirrors the polarity/modality
    harness tests so no embedding model is needed)
  * CLI: argument-validation / error exit codes that never reach the LLM

The live end-to-end run against the real general extractor lives in
tests/test_integration_validity_benchmark.py (integration tier).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.equivalence import EquivalenceJudge
from particles.benchmark.validity import (
    ValiditySuiteLoadError,
    date_accuracy,
    discover_validity_suites,
    expiry_precision,
    expiry_recall,
    load_validity_suite,
    run_validity_benchmark,
    support_counts,
    wrong_expiry_rate,
)
from particles.benchmark.validity.schema import ValidityCase, ValidityLabel, ValiditySuite
from particles.core.schema import UncertaintyNature
from particles.extraction.general import CandidateParticle, ExtractionResult

_SEED = Path("tests/benchmark/validity/durability-seed-001.yaml")

D1 = date(2026, 12, 31)
D2 = date(2026, 2, 28)


# ---------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------


class TestValidityMetrics:
    def test_wrong_expiry_rate_headline(self) -> None:
        # Three durable golds; the extractor wrongly bounded two of them.
        pairs = [(None, D1), (None, D2), (None, None), (D1, D1)]
        assert wrong_expiry_rate(pairs) == pytest.approx(2 / 3)

    def test_wrong_expiry_rate_vacuous_is_zero(self) -> None:
        # No durable gold → no such error possible.
        assert wrong_expiry_rate([(D1, D1)]) == 0.0
        assert wrong_expiry_rate([]) == 0.0

    def test_wrong_expiry_rate_perfect_when_durable_kept_bare(self) -> None:
        assert wrong_expiry_rate([(None, None), (None, None)]) == 0.0

    def test_expiry_precision_counts_false_positives(self) -> None:
        # Two emitted boundaries; one lands on a durable fact (a false positive).
        pairs = [(D1, D1), (None, D2)]
        assert expiry_precision(pairs) == pytest.approx(0.5)

    def test_expiry_precision_vacuous_is_one(self) -> None:
        # Nothing emitted a boundary → no false positives possible.
        assert expiry_precision([(D1, None), (None, None)]) == 1.0
        assert expiry_precision([]) == 1.0

    def test_expiry_recall_counts_false_negatives(self) -> None:
        # Two bounded golds; the extractor caught one.
        pairs = [(D1, D1), (D2, None)]
        assert expiry_recall(pairs) == pytest.approx(0.5)

    def test_expiry_recall_vacuous_is_one(self) -> None:
        assert expiry_recall([(None, None)]) == 1.0
        assert expiry_recall([]) == 1.0

    def test_date_accuracy_within_tolerance(self) -> None:
        # Emitted two days early — inside a 3-day tolerance, outside a 1-day one.
        pairs = [(date(2026, 6, 30), date(2026, 6, 28))]
        assert date_accuracy(pairs, tolerance_days=3) == 1.0
        assert date_accuracy(pairs, tolerance_days=1) == 0.0

    def test_date_accuracy_only_scores_both_bounded_pairs(self) -> None:
        # A durable-wrongly-bounded pair and a missed-boundary pair don't count —
        # date accuracy is only defined where both sides carry a boundary.
        pairs = [(None, D1), (D1, None), (D1, D1)]
        assert date_accuracy(pairs, tolerance_days=0) == 1.0

    def test_date_accuracy_vacuous_is_one(self) -> None:
        assert date_accuracy([(None, None)], tolerance_days=3) == 1.0
        assert date_accuracy([], tolerance_days=3) == 1.0

    def test_support_counts(self) -> None:
        pairs = [(None, D1), (None, None), (D1, D1), (D2, None)]
        s = support_counts(pairs)
        assert s == {"durable": 2, "bounded": 2, "emitted": 2, "both": 1}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestValidityLoader:
    def test_seed_suite_round_trips(self) -> None:
        suite = load_validity_suite(_SEED)
        assert suite.suite_id == "durability-seed-001"
        assert suite.source_type == "TEXT"
        assert len(suite.cases) == 3
        # Every case carries at least one durable decoy — the wrong-expiry rate
        # is only measurable when durable facts sit beside genuine boundaries.
        for case in suite.cases:
            assert any(lbl.is_durable for lbl in case.labels)
        # Both kinds of gold appear across the suite.
        durables = [lbl for c in suite.cases for lbl in c.labels if lbl.is_durable]
        bounded = [lbl for c in suite.cases for lbl in c.labels if not lbl.is_durable]
        assert durables and bounded
        # A durable label carries no boundary; a bounded one carries a date.
        assert all(lbl.expected_valid_until is None for lbl in durables)
        assert all(lbl.expected_valid_until is not None for lbl in bounded)

    def test_reference_datetime_is_utc_midnight(self) -> None:
        suite = load_validity_suite(_SEED)
        ref = suite.cases[0].reference_datetime()
        assert ref.tzinfo is UTC
        assert (ref.hour, ref.minute, ref.second) == (0, 0, 0)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(ValiditySuiteLoadError, match="not found"):
            load_validity_suite(Path("tests/benchmark/validity/does-not-exist.yaml"))

    def test_durable_with_boundary_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: TEXT\n"
            "cases:\n  - case_id: c1\n    document: hello\n    reference_date: 2026-01-01\n"
            "    labels:\n      - content: a claim\n        is_durable: true\n"
            "        expected_valid_until: 2026-06-01\n"
        )
        with pytest.raises(ValiditySuiteLoadError, match="durable label carries no boundary"):
            load_validity_suite(bad)

    def test_bounded_without_date_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: TEXT\n"
            "cases:\n  - case_id: c1\n    document: hello\n    reference_date: 2026-01-01\n"
            "    labels:\n      - content: a claim\n        is_durable: false\n"
        )
        with pytest.raises(ValiditySuiteLoadError, match="bounded label needs a date"):
            load_validity_suite(bad)

    def test_bad_date_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: TEXT\n"
            "cases:\n  - case_id: c1\n    document: hello\n    reference_date: not-a-date\n"
            "    labels:\n      - content: a claim\n        is_durable: true\n"
        )
        with pytest.raises(ValiditySuiteLoadError, match="not an ISO-8601 date"):
            load_validity_suite(bad)

    def test_unknown_case_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: TEXT\n"
            "cases:\n  - case_id: c1\n    document: hello\n    reference_date: 2026-01-01\n"
            "    surprise: 1\n    labels:\n      - content: a\n        is_durable: true\n"
        )
        with pytest.raises(ValiditySuiteLoadError, match="unknown ValidityCase field"):
            load_validity_suite(bad)

    def test_empty_labels_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: TEXT\n"
            "cases:\n  - case_id: c1\n    document: hello\n    reference_date: 2026-01-01\n"
            "    labels: []\n"
        )
        with pytest.raises(ValiditySuiteLoadError, match="non-empty list"):
            load_validity_suite(bad)

    def test_discover_skips_malformed_yields_good(self, tmp_path: Path) -> None:
        good = tmp_path / "good.yaml"
        good.write_text(
            "suite_id: good\nname: g\nversion: '0'\ndomain: d\nsource_type: TEXT\n"
            "cases:\n  - case_id: c1\n    document: hello\n    reference_date: 2026-01-01\n"
            "    labels:\n      - content: a claim\n        is_durable: true\n"
        )
        broken = tmp_path / "broken.yaml"
        broken.write_text("suite_id: broken\ncases: not-a-list\n")
        found = list(discover_validity_suites(tmp_path))
        assert [s.suite_id for s in found] == ["good"]


# ---------------------------------------------------------------------------
# Runner orchestration (stub extractor + patched similarity matrix)
# ---------------------------------------------------------------------------


class _StubGeneralExtractor:
    """Emits a fixed claim set; accepts only the TEXT source type used in tests."""

    EXTRACTOR_ID = "stub-general"
    EXTRACTOR_VERSION = "0.0.1"

    def __init__(self, candidates: list[CandidateParticle]) -> None:
        self._candidates = candidates

    def accepts(self, source_type: str) -> bool:
        return source_type == "TEXT"

    async def extract(self, snapshot: object, content: bytes, **kwargs: object) -> ExtractionResult:
        return ExtractionResult(candidates=list(self._candidates))


def _claim(content: str, valid_until: datetime | None) -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=[],
        valid_until=valid_until,
    )


def _identity_matrix(emitted: list[str], expected: list[str]) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(len(expected))] for i in range(len(emitted))]


def _dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


class TestValidityRunner:
    @pytest.mark.asyncio
    async def test_runner_scores_wrong_expiry_and_precision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Gold labels (order matters — the identity matrix aligns by index):
        # L1 bounded, L2 durable (decoy), L3 bounded, L4 durable (decoy).
        suite = ValiditySuite(
            suite_id="stub-suite",
            name="stub",
            version="0.0.1",
            domain="test",
            source_type="TEXT",
            date_tolerance_days=3,
            cases=[
                ValidityCase(
                    case_id="c1",
                    document="some prose",
                    reference_date=date(2026, 1, 1),
                    labels=[
                        ValidityLabel("contract through 2026", D1, is_durable=False),
                        ValidityLabel("founded in 1998", None, is_durable=True),
                        ValidityLabel("offer expires feb", D2, is_durable=False),
                        ValidityLabel("met her in 2019", None, is_durable=True),
                    ],
                )
            ],
        )
        # Emitted (same order): E1 correct boundary, E2 wrongly bounds a durable
        # fact (the over-eager-expiry miss), E3 misses a real boundary, E4 correct
        # (durable kept bare).
        extractor = _StubGeneralExtractor(
            [
                _claim("contract emitted", _dt(D1)),
                _claim("founded emitted", _dt(date(2019, 1, 1))),
                _claim("offer emitted", None),
                _claim("met emitted", None),
            ]
        )
        monkeypatch.setattr("particles.benchmark.equivalence._similarity_matrix", _identity_matrix)

        report = await run_validity_benchmark(
            suite, extractor, judge=EquivalenceJudge.EMBEDDING, threshold=0.65
        )

        assert report.cases_run == 1
        assert report.claims_aligned == 4
        # Two durable golds; one wrongly bounded → 0.5 headline danger.
        assert report.wrong_expiry_rate == pytest.approx(0.5)
        # Two emitted boundaries (E1, E2); one lands on a durable fact → 0.5.
        assert report.expiry_precision == pytest.approx(0.5)
        # Two bounded golds (L1, L3); one caught (E1) → 0.5.
        assert report.expiry_recall == pytest.approx(0.5)
        # The one both-bounded pair (L1↔E1) is exactly right → 1.0.
        assert report.date_accuracy == 1.0
        assert report.validity_extractor_enabled is True
        assert report.support == {"durable": 2, "bounded": 2, "emitted": 2, "both": 1}

    @pytest.mark.asyncio
    async def test_runner_extractor_disabled_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.benchmark.validity import runner as validity_runner
        from particles.config import ExtractionValidityConfig, ParticlesConfig

        disabled = ParticlesConfig(extraction_validity=ExtractionValidityConfig(enabled=False))
        monkeypatch.setattr(validity_runner, "get_config", lambda: disabled)

        suite = ValiditySuite(
            suite_id="s",
            name="s",
            version="0",
            domain="t",
            source_type="TEXT",
            cases=[
                ValidityCase(
                    "c1",
                    "prose",
                    date(2026, 1, 1),
                    [ValidityLabel("durable", None, is_durable=True)],
                )
            ],
        )
        report = await run_validity_benchmark(suite, _StubGeneralExtractor([]))
        assert report.validity_extractor_enabled is False
        assert any("extractor is OFF" in n for n in report.quality_notes)

    @pytest.mark.asyncio
    async def test_runner_skips_declined_source_type(self) -> None:
        class _RejectsAll(_StubGeneralExtractor):
            def accepts(self, source_type: str) -> bool:
                return False

        suite = ValiditySuite(
            suite_id="s",
            name="s",
            version="0",
            domain="t",
            source_type="TEXT",
            cases=[
                ValidityCase(
                    "c1",
                    "prose",
                    date(2026, 1, 1),
                    [ValidityLabel("durable", None, is_durable=True)],
                )
            ],
        )
        report = await run_validity_benchmark(suite, _RejectsAll([]))
        assert report.cases_run == 0
        assert report.claims_aligned == 0
        assert any("declines source_type" in n for n in report.quality_notes)


# ---------------------------------------------------------------------------
# CLI argument validation (no LLM reached)
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestValidityCLI:
    def test_unknown_extractor_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["extractor", "benchmark-validity", "no-such-extractor"])
        assert result.exit_code == 2
        assert "no-such-extractor" in (result.output + (result.stderr or ""))

    def test_unknown_suite_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["extractor", "benchmark-validity", "general-extractor", "--suite", "does-not-exist"],
        )
        assert result.exit_code == 2

    def test_no_applicable_suites_exits_2(self, runner: CliRunner) -> None:
        # wikidata-extractor accepts WIKIDATA_ENTITY, never TEXT — so no validity
        # suite applies and the command exits 2 before any LLM call.
        result = runner.invoke(app, ["extractor", "benchmark-validity", "wikidata-extractor"])
        assert result.exit_code == 2
        assert "No applicable validity suites" in (result.output + (result.stderr or ""))
