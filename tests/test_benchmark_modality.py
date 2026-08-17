"""Tests for the journal modality benchmark.

Covers the unit tier — no LLM, no real embeddings:
  * pure metrics: per-modality precision/recall, the false-non-FALSIFIABLE
    rate, narrative-emission rate, and their vacuous-denominator conventions
  * loader: the bundled seed suite round-trips; error paths (missing fields,
    bad modality, unknown keys, empty labels) raise; discovery skips malformed
    files but yields the good ones
  * runner orchestration: a stub extractor + a monkeypatched similarity matrix
    drive the full accounting deterministically (mirrors test_benchmark_schema's
    `_similarity_matrix` patch so the tests don't need the embedding model)
  * CLI: argument-validation / error exit codes that never reach the LLM

The live end-to-end run against the real journal extractor lives in
tests/test_integration_journal_modality.py (integration tier).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.equivalence import EquivalenceJudge
from particles.benchmark.modality import (
    ModalitySuiteLoadError,
    confusion_counts,
    discover_modality_suites,
    false_non_falsifiable_rate,
    load_modality_suite,
    modality_precision,
    modality_recall,
    narrative_emission_rate,
    run_modality_benchmark,
)
from particles.benchmark.modality.schema import (
    ModalityCase,
    ModalityLabel,
    ModalitySuite,
)
from particles.core.schema import AssertionModality, UncertaintyNature
from particles.extraction.general import CandidateParticle, ExtractionResult

FALS = AssertionModality.FALSIFIABLE
EVAL = AssertionModality.EVALUATIVE
EXP = AssertionModality.EXPERIENTIAL
CONS = AssertionModality.CONSTITUTIVE

_SEED = Path("tests/benchmark/modality/journal-modality-seed-001.yaml")


# ---------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------


class TestModalityMetrics:
    def test_precision_perfect(self) -> None:
        pairs = [(FALS, FALS), (EVAL, EVAL), (EXP, EXP)]
        assert modality_precision(pairs, FALS) == 1.0
        assert modality_recall(pairs, FALS) == 1.0

    def test_precision_counts_false_positives(self) -> None:
        # Two claims emitted as FALSIFIABLE; one of them should have been EVAL.
        pairs = [(FALS, FALS), (EVAL, FALS)]
        assert modality_precision(pairs, FALS) == pytest.approx(0.5)
        # Recall of FALSIFIABLE: one expected, one caught → 1.0
        assert modality_recall(pairs, FALS) == 1.0

    def test_recall_counts_false_negatives(self) -> None:
        # Two claims expected FALSIFIABLE; one demoted to EVAL.
        pairs = [(FALS, FALS), (FALS, EVAL)]
        assert modality_recall(pairs, FALS) == pytest.approx(0.5)
        # Precision of FALSIFIABLE: one emitted as FALS, it was right → 1.0
        assert modality_precision(pairs, FALS) == 1.0

    def test_vacuous_precision_and_recall_are_one(self) -> None:
        pairs = [(FALS, FALS)]
        # No claim is expected or emitted as CONSTITUTIVE → vacuously perfect.
        assert modality_precision(pairs, CONS) == 1.0
        assert modality_recall(pairs, CONS) == 1.0
        assert modality_precision([], FALS) == 1.0
        assert modality_recall([], FALS) == 1.0

    def test_false_non_falsifiable_rate(self) -> None:
        # Three expected FALSIFIABLE; two wrongly demoted (one EVAL, one EXP).
        pairs = [(FALS, FALS), (FALS, EVAL), (FALS, EXP), (EVAL, EVAL)]
        assert false_non_falsifiable_rate(pairs) == pytest.approx(2 / 3)

    def test_false_non_falsifiable_rate_is_complement_of_recall(self) -> None:
        pairs = [(FALS, FALS), (FALS, EVAL), (EXP, EXP)]
        assert false_non_falsifiable_rate(pairs) == pytest.approx(
            1.0 - modality_recall(pairs, FALS)
        )

    def test_false_non_falsifiable_rate_vacuous_is_zero(self) -> None:
        # No FALSIFIABLE expected → no such miss is possible.
        assert false_non_falsifiable_rate([(EVAL, EVAL)]) == 0.0
        assert false_non_falsifiable_rate([]) == 0.0

    def test_narrative_emission_rate(self) -> None:
        assert narrative_emission_rate(3, 2) == pytest.approx(2 / 3)
        assert narrative_emission_rate(0, 0) == 1.0  # vacuous

    def test_confusion_counts(self) -> None:
        pairs = [(FALS, FALS), (FALS, FALS), (FALS, EVAL)]
        counts = confusion_counts(pairs)
        assert counts[(FALS, FALS)] == 2
        assert counts[(FALS, EVAL)] == 1


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestModalityLoader:
    def test_seed_suite_round_trips(self) -> None:
        suite = load_modality_suite(_SEED)
        assert suite.suite_id == "journal-modality-seed-001"
        assert suite.source_type == "JOURNAL"
        assert len(suite.cases) == 3
        first = suite.cases[0]
        assert first.case_id == "journal-travel-001"
        assert len(first.labels) == 6
        assert first.narrative_expected is True
        # Every label carries a valid modality.
        mods = {lbl.modality for case in suite.cases for lbl in case.labels}
        assert FALS in mods and EVAL in mods and EXP in mods
        # The seed embeds at least one FALSIFIABLE trap per case.
        for case in suite.cases:
            assert any(lbl.modality == FALS for lbl in case.labels)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(ModalitySuiteLoadError, match="not found"):
            load_modality_suite(Path("tests/benchmark/modality/does-not-exist.yaml"))

    def test_bad_modality_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: JOURNAL\n"
            "cases:\n  - case_id: c1\n    entry: hello\n    labels:\n"
            "      - content: a fact\n        modality: NOT_A_MODALITY\n"
        )
        with pytest.raises(ModalitySuiteLoadError, match="not a valid AssertionModality"):
            load_modality_suite(bad)

    def test_unknown_case_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: JOURNAL\n"
            "cases:\n  - case_id: c1\n    entry: hello\n    surprise: 1\n    labels:\n"
            "      - content: a fact\n        modality: FALSIFIABLE\n"
        )
        with pytest.raises(ModalitySuiteLoadError, match="unknown ModalityCase field"):
            load_modality_suite(bad)

    def test_empty_labels_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "suite_id: x\nname: x\nversion: '0'\ndomain: d\nsource_type: JOURNAL\n"
            "cases:\n  - case_id: c1\n    entry: hello\n    labels: []\n"
        )
        with pytest.raises(ModalitySuiteLoadError, match="non-empty list"):
            load_modality_suite(bad)

    def test_discover_skips_malformed_yields_good(self, tmp_path: Path) -> None:
        good = tmp_path / "good.yaml"
        good.write_text(
            "suite_id: good\nname: g\nversion: '0'\ndomain: d\nsource_type: JOURNAL\n"
            "cases:\n  - case_id: c1\n    entry: hello\n    labels:\n"
            "      - content: a fact\n        modality: FALSIFIABLE\n"
        )
        broken = tmp_path / "broken.yaml"
        broken.write_text("suite_id: broken\ncases: not-a-list\n")
        found = list(discover_modality_suites(tmp_path))
        assert [s.suite_id for s in found] == ["good"]


# ---------------------------------------------------------------------------
# Runner orchestration (stub extractor + patched similarity matrix)
# ---------------------------------------------------------------------------


class _StubJournalExtractor:
    """Emits a fixed claim set + one NARRATIVE; accepts only JOURNAL."""

    EXTRACTOR_ID = "stub-journal"
    EXTRACTOR_VERSION = "0.0.1"

    def __init__(self, candidates: list[CandidateParticle]) -> None:
        self._candidates = candidates

    def accepts(self, source_type: str) -> bool:
        return source_type == "JOURNAL"

    async def extract(self, snapshot: object, content: bytes, **kwargs: object) -> ExtractionResult:
        return ExtractionResult(candidates=list(self._candidates))


def _claim(content: str, modality: AssertionModality) -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=[],
        assertion_modality=modality,
    )


def _narrative(content: str) -> CandidateParticle:
    from particles.core.schema import ParticleType

    return CandidateParticle(
        content=content,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=[],
        particle_type=ParticleType.NARRATIVE,
    )


def _identity_matrix(emitted: list[str], expected: list[str]) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(len(expected))] for i in range(len(emitted))]


class TestModalityRunner:
    @pytest.mark.asyncio
    async def test_runner_scores_modality_and_narrative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Gold labels (order matters — the identity matrix aligns by index).
        suite = ModalitySuite(
            suite_id="stub-suite",
            name="stub",
            version="0.0.1",
            domain="test",
            source_type="JOURNAL",
            cases=[
                ModalityCase(
                    case_id="c1",
                    entry="some journal prose",
                    narrative_expected=True,
                    labels=[
                        ModalityLabel("fact one", FALS),
                        ModalityLabel("feeling one", EXP),
                        ModalityLabel("opinion one", EVAL),
                        ModalityLabel("fact two", FALS),
                    ],
                )
            ],
        )
        # Emitted claims (same order) — C2 over-tags an opinion FALSIFIABLE,
        # C3 demotes a real fact to EVALUATIVE (a false-non-FALSIFIABLE miss).
        extractor = _StubJournalExtractor(
            [
                _claim("fact one emitted", FALS),
                _claim("feeling one emitted", EXP),
                _claim("opinion one emitted", FALS),
                _claim("fact two emitted", EVAL),
                _narrative("Journal entry — a representative day"),
            ]
        )
        monkeypatch.setattr("particles.benchmark.equivalence._similarity_matrix", _identity_matrix)

        report = await run_modality_benchmark(
            suite, extractor, judge=EquivalenceJudge.EMBEDDING, threshold=0.65
        )

        assert report.cases_run == 1
        assert report.cases_total == 1
        assert report.claims_aligned == 4
        assert report.claims_unaligned == 0
        # Two expected FALSIFIABLE (fact one, fact two); one demoted → 0.5.
        assert report.false_non_falsifiable_rate == pytest.approx(0.5)
        # NARRATIVE emitted for the one entry that expects it.
        assert report.narrative_cases_expected == 1
        assert report.narrative_cases_emitted == 1
        assert report.narrative_emission_rate == 1.0
        # Confusion carries every aligned pair.
        assert sum(c.count for c in report.confusion) == 4
        # FALSIFIABLE recall: 1 of 2 caught.
        assert report.recall["FALSIFIABLE"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_runner_skips_declined_source_type(self) -> None:
        class _RejectsAll(_StubJournalExtractor):
            def accepts(self, source_type: str) -> bool:
                return False

        suite = ModalitySuite(
            suite_id="s",
            name="s",
            version="0",
            domain="t",
            source_type="JOURNAL",
            cases=[
                ModalityCase("c1", "prose", [ModalityLabel("fact", FALS)], narrative_expected=True)
            ],
        )
        report = await run_modality_benchmark(suite, _RejectsAll([]))
        assert report.cases_run == 0
        assert report.claims_aligned == 0
        assert any("declines source_type" in n for n in report.quality_notes)


# ---------------------------------------------------------------------------
# CLI argument validation (no LLM reached)
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestModalityCLI:
    def test_unknown_extractor_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["extractor", "benchmark-modality", "no-such-extractor"])
        assert result.exit_code == 2
        assert "no-such-extractor" in (result.output + (result.stderr or ""))

    def test_unknown_suite_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["extractor", "benchmark-modality", "journal-extractor", "--suite", "does-not-exist"],
        )
        assert result.exit_code == 2

    def test_no_applicable_suites_exits_2(self, runner: CliRunner) -> None:
        # wikidata-extractor accepts WIKIDATA_ENTITY, never JOURNAL — so no
        # modality suite applies and the command exits 2 before any LLM call.
        result = runner.invoke(app, ["extractor", "benchmark-modality", "wikidata-extractor"])
        assert result.exit_code == 2
        assert "No applicable modality suites" in (result.output + (result.stderr or ""))
