"""Tests for the claim-polarity benchmark.

Covers the unit tier — no LLM, no real embeddings:
  * pure metrics: per-polarity precision/recall, the headline wrong-DECLINED
    rate, the wrong-hidden superset, and their vacuous-denominator conventions
  * loader: the bundled seed suite round-trips; error paths (missing fields,
    bad polarity, unknown keys, empty labels) raise; discovery skips malformed
    files but yields the good ones
  * runner orchestration: a stub extractor + a monkeypatched similarity matrix
    drive the full accounting deterministically (mirrors test_benchmark_modality's
    `_identity_matrix` patch so the tests don't need the embedding model)
  * CLI: argument-validation / error exit codes that never reach the LLM

The live end-to-end run against the real general extractor lives in
tests/test_integration_polarity.py (integration tier).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.equivalence import EquivalenceJudge
from particles.benchmark.polarity import (
    ClaimPolarity,
    PolaritySuiteLoadError,
    confusion_counts,
    discover_polarity_suites,
    load_polarity_suite,
    polarity_precision,
    polarity_recall,
    run_polarity_benchmark,
    wrong_declined_rate,
    wrong_hidden_rate,
)
from particles.benchmark.polarity.schema import (
    PolarityCase,
    PolarityLabel,
    PolaritySuite,
)
from particles.core.schema import UncertaintyNature
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.polarity import POLARITY_KEY

ASST = ClaimPolarity.ASSERTED
DECL = ClaimPolarity.DECLINED
HYPO = ClaimPolarity.HYPOTHETICAL

_SEED = Path("tests/benchmark/polarity/adr-polarity-seed-001.yaml")


# ---------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------


class TestPolarityMetrics:
    def test_precision_perfect(self) -> None:
        pairs = [(ASST, ASST), (DECL, DECL), (HYPO, HYPO)]
        assert polarity_precision(pairs, ASST) == 1.0
        assert polarity_recall(pairs, ASST) == 1.0

    def test_precision_counts_false_positives(self) -> None:
        # Two claims emitted as DECLINED; one of them should have been ASSERTED.
        pairs = [(DECL, DECL), (ASST, DECL)]
        assert polarity_precision(pairs, DECL) == pytest.approx(0.5)
        # Recall of DECLINED: one expected, one caught → 1.0
        assert polarity_recall(pairs, DECL) == 1.0

    def test_recall_counts_false_negatives(self) -> None:
        # Two claims expected ASSERTED; one demoted to DECLINED.
        pairs = [(ASST, ASST), (ASST, DECL)]
        assert polarity_recall(pairs, ASST) == pytest.approx(0.5)
        # Precision of ASSERTED: one emitted as ASST, it was right → 1.0
        assert polarity_precision(pairs, ASST) == 1.0

    def test_vacuous_precision_and_recall_are_one(self) -> None:
        pairs = [(ASST, ASST)]
        # No claim is expected or emitted as HYPOTHETICAL → vacuously perfect.
        assert polarity_precision(pairs, HYPO) == 1.0
        assert polarity_recall(pairs, HYPO) == 1.0
        assert polarity_precision([], ASST) == 1.0
        assert polarity_recall([], ASST) == 1.0

    def test_wrong_declined_rate(self) -> None:
        # Three expected ASSERTED; one wrongly DECLINED, one wrongly HYPOTHETICAL.
        pairs = [(ASST, ASST), (ASST, DECL), (ASST, HYPO), (DECL, DECL)]
        # Only the DECLINED demotion counts toward the headline → 1/3.
        assert wrong_declined_rate(pairs) == pytest.approx(1 / 3)

    def test_wrong_declined_rate_vacuous_is_zero(self) -> None:
        # No ASSERTED expected → no such miss is possible.
        assert wrong_declined_rate([(DECL, DECL)]) == 0.0
        assert wrong_declined_rate([]) == 0.0

    def test_wrong_hidden_rate_superset_of_declined(self) -> None:
        # Both the DECLINED and the HYPOTHETICAL demotion hide the decision → 2/3.
        pairs = [(ASST, ASST), (ASST, DECL), (ASST, HYPO), (DECL, DECL)]
        assert wrong_hidden_rate(pairs) == pytest.approx(2 / 3)
        # The hidden rate is always ≥ the DECLINED-only rate.
        assert wrong_hidden_rate(pairs) >= wrong_declined_rate(pairs)

    def test_wrong_hidden_rate_is_complement_of_asserted_recall(self) -> None:
        pairs = [(ASST, ASST), (ASST, DECL), (HYPO, HYPO)]
        assert wrong_hidden_rate(pairs) == pytest.approx(1.0 - polarity_recall(pairs, ASST))

    def test_wrong_hidden_rate_vacuous_is_zero(self) -> None:
        assert wrong_hidden_rate([(DECL, DECL)]) == 0.0
        assert wrong_hidden_rate([]) == 0.0

    def test_confusion_counts(self) -> None:
        pairs = [(ASST, ASST), (ASST, ASST), (ASST, DECL)]
        counts = confusion_counts(pairs)
        assert counts[(ASST, ASST)] == 2
        assert counts[(ASST, DECL)] == 1


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestPolarityLoader:
    def test_seed_suite_round_trips(self) -> None:
        suite = load_polarity_suite(_SEED)
        assert suite.suite_id == "adr-polarity-seed-001"
        assert suite.source_type == "ADR"
        assert len(suite.cases) == 3
        first = suite.cases[0]
        assert first.case_id == "adr-decision-rejected-001"
        assert len(first.labels) == 4
        # Every label carries a valid polarity; all three classes appear.
        pols = {lbl.polarity for case in suite.cases for lbl in case.labels}
        assert pols == {ASST, DECL, HYPO}
        # The seed plants at least one ASSERTED trap per case (the wrong-DECLINED
        # risk is only measurable when ASSERTED decisions sit by rejection prose).
        for case in suite.cases:
            assert any(lbl.polarity == ASST for lbl in case.labels)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(PolaritySuiteLoadError, match="not found"):
            load_polarity_suite(Path("tests/benchmark/polarity/does-not-exist.yaml"))

    def test_bad_polarity_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: ADR\n"
            "cases:\n  - case_id: c1\n    document: hello\n    labels:\n"
            "      - content: a claim\n        polarity: NOT_A_POLARITY\n"
        )
        with pytest.raises(PolaritySuiteLoadError, match="not a valid ClaimPolarity"):
            load_polarity_suite(bad)

    def test_unknown_case_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: ADR\n"
            "cases:\n  - case_id: c1\n    document: hello\n    surprise: 1\n    labels:\n"
            "      - content: a claim\n        polarity: ASSERTED\n"
        )
        with pytest.raises(PolaritySuiteLoadError, match="unknown PolarityCase field"):
            load_polarity_suite(bad)

    def test_empty_labels_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: ADR\n"
            "cases:\n  - case_id: c1\n    document: hello\n    labels: []\n"
        )
        with pytest.raises(PolaritySuiteLoadError, match="non-empty list"):
            load_polarity_suite(bad)

    def test_discover_skips_malformed_yields_good(self, tmp_path: Path) -> None:
        good = tmp_path / "good.yaml"
        good.write_text(
            "suite_id: good\nname: g\nversion: '0'\ndomain: d\nsource_type: ADR\n"
            "cases:\n  - case_id: c1\n    document: hello\n    labels:\n"
            "      - content: a claim\n        polarity: ASSERTED\n"
        )
        broken = tmp_path / "broken.yaml"
        broken.write_text("suite_id: broken\ncases: not-a-list\n")
        found = list(discover_polarity_suites(tmp_path))
        assert [s.suite_id for s in found] == ["good"]


# ---------------------------------------------------------------------------
# Runner orchestration (stub extractor + patched similarity matrix)
# ---------------------------------------------------------------------------


class _StubGeneralExtractor:
    """Emits a fixed claim set; accepts only the ADR source type used in tests."""

    EXTRACTOR_ID = "stub-general"
    EXTRACTOR_VERSION = "0.0.1"

    def __init__(self, candidates: list[CandidateParticle]) -> None:
        self._candidates = candidates

    def accepts(self, source_type: str) -> bool:
        return source_type == "ADR"

    async def extract(self, snapshot: object, content: bytes, **kwargs: object) -> ExtractionResult:
        return ExtractionResult(candidates=list(self._candidates))


def _claim(content: str, polarity: ClaimPolarity) -> CandidateParticle:
    # ASSERTED is the absence of the key (matching the producer in general.py);
    # the two non-asserted values ride on properties["extraction:polarity"].
    props = None if polarity is ASST else {POLARITY_KEY: polarity.value}
    return CandidateParticle(
        content=content,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=[],
        properties=props,
    )


def _identity_matrix(emitted: list[str], expected: list[str]) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(len(expected))] for i in range(len(emitted))]


class TestPolarityRunner:
    @pytest.mark.asyncio
    async def test_runner_scores_polarity_and_wrong_declined(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Gold labels (order matters — the identity matrix aligns by index).
        suite = PolaritySuite(
            suite_id="stub-suite",
            name="stub",
            version="0.0.1",
            domain="test",
            source_type="ADR",
            cases=[
                PolarityCase(
                    case_id="c1",
                    document="some ADR prose",
                    labels=[
                        PolarityLabel("decision one", ASST),
                        PolarityLabel("rejected one", DECL),
                        PolarityLabel("decision two", ASST),
                        PolarityLabel("supposition one", HYPO),
                    ],
                )
            ],
        )
        # Emitted claims (same order) — E2 demotes a real decision to DECLINED
        # (the dangerous wrong-DECLINED miss the suite exists to bound).
        extractor = _StubGeneralExtractor(
            [
                _claim("decision one emitted", ASST),
                _claim("rejected one emitted", DECL),
                _claim("decision two emitted", DECL),
                _claim("supposition one emitted", HYPO),
            ]
        )
        monkeypatch.setattr("particles.benchmark.equivalence._similarity_matrix", _identity_matrix)

        report = await run_polarity_benchmark(
            suite, extractor, judge=EquivalenceJudge.EMBEDDING, threshold=0.65
        )

        assert report.cases_run == 1
        assert report.cases_total == 1
        assert report.claims_aligned == 4
        assert report.claims_unaligned == 0
        # Two expected ASSERTED (decision one/two); one demoted DECLINED → 0.5.
        assert report.wrong_declined_rate == pytest.approx(0.5)
        # No HYPOTHETICAL demotion here, so the hidden rate equals it.
        assert report.wrong_hidden_rate == pytest.approx(0.5)
        # Classifier is on by the config default.
        assert report.polarity_classifier_enabled is True
        # Confusion carries every aligned pair.
        assert sum(c.count for c in report.confusion) == 4
        # ASSERTED recall: 1 of 2 caught.
        assert report.recall["ASSERTED"] == pytest.approx(0.5)
        # DECLINED precision: emitted twice (E1 right, E2 wrong) → 0.5.
        assert report.precision["DECLINED"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_runner_classifier_disabled_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No env override exists for extraction_polarity, so patch the runner's
        # get_config to return a config with the classifier off (the runner
        # imports get_config at module top).
        from particles.benchmark.polarity import runner as polarity_runner
        from particles.config import ExtractionPolarityConfig, ParticlesConfig

        disabled = ParticlesConfig(extraction_polarity=ExtractionPolarityConfig(enabled=False))
        monkeypatch.setattr(polarity_runner, "get_config", lambda: disabled)

        suite = PolaritySuite(
            suite_id="s",
            name="s",
            version="0",
            domain="t",
            source_type="ADR",
            cases=[PolarityCase("c1", "prose", [PolarityLabel("decision", ASST)])],
        )
        report = await run_polarity_benchmark(suite, _StubGeneralExtractor([]))
        assert report.polarity_classifier_enabled is False
        assert any("classifier is OFF" in n for n in report.quality_notes)

    @pytest.mark.asyncio
    async def test_runner_skips_declined_source_type(self) -> None:
        class _RejectsAll(_StubGeneralExtractor):
            def accepts(self, source_type: str) -> bool:
                return False

        suite = PolaritySuite(
            suite_id="s",
            name="s",
            version="0",
            domain="t",
            source_type="ADR",
            cases=[PolarityCase("c1", "prose", [PolarityLabel("decision", ASST)])],
        )
        report = await run_polarity_benchmark(suite, _RejectsAll([]))
        assert report.cases_run == 0
        assert report.claims_aligned == 0
        assert any("declines source_type" in n for n in report.quality_notes)


# ---------------------------------------------------------------------------
# CLI argument validation (no LLM reached)
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestPolarityCLI:
    def test_unknown_extractor_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["extractor", "benchmark-polarity", "no-such-extractor"])
        assert result.exit_code == 2
        assert "no-such-extractor" in (result.output + (result.stderr or ""))

    def test_unknown_suite_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["extractor", "benchmark-polarity", "general-extractor", "--suite", "does-not-exist"],
        )
        assert result.exit_code == 2

    def test_no_applicable_suites_exits_2(self, runner: CliRunner) -> None:
        # wikidata-extractor accepts WIKIDATA_ENTITY, never ADR — so no polarity
        # suite applies and the command exits 2 before any LLM call.
        result = runner.invoke(app, ["extractor", "benchmark-polarity", "wikidata-extractor"])
        assert result.exit_code == 2
        assert "No applicable polarity suites" in (result.output + (result.stderr or ""))
