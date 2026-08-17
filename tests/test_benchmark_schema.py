"""Tests for the benchmark schema + loader + metrics + equivalence.

Covers commit 1/3 (pure functions). The runner orchestration and CLI
land in commit 2/3 with end-to-end tests against a registered extractor.

What's covered here:
  * schema dataclasses (immutability, defaults)
  * loader YAML round-trip, error paths (missing fields, unknown keys,
    fixture/snapshot mutual exclusion, bad uncertainty_nature)
  * loader skips malformed files but yields good ones
  * resolve_case_content against the existing conformance fixture corpus
  * metrics: precision / recall edge cases + ECE on a hand-built case
  * equivalence: greedy assignment, under-confidence demotion, spurious +
    missed-required counts (embedding judge mocked to a fixed similarity
    matrix so tests do not depend on the embedding model being installed)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from particles.benchmark.equivalence import (
    EquivalenceJudge,
    MatchResult,
    match_emitted_to_expected,
)
from particles.benchmark.loader import (
    SuiteLoadError,
    discover_suites,
    load_suite,
    resolve_case_content,
)
from particles.benchmark.metrics import (
    compute_calibration_error,
    compute_precision,
    compute_recall,
)
from particles.benchmark.schema import (
    BenchmarkCase,
    BenchmarkSuite,
    ExpectedParticle,
    RequiredMetric,
)
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _make_emitted(content: str, confidence: float = 0.9) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(
            value=confidence,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="entry-1",
                snapshot_id="snap-1",
            )
        ],
        asserted_by="stub-extractor",
        asserted_at=datetime.now(UTC),
        extractor_ref={"name": "stub-extractor", "version": "0.1.0"},
        subject_ids=[],
    )


def _make_expected(
    content: str, confidence_min: float = 0.5, required: bool = True
) -> ExpectedParticle:
    return ExpectedParticle(
        content=content,
        confidence_min=confidence_min,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        required=required,
    )


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------


class TestSchema:
    def test_expected_particle_defaults(self) -> None:
        ep = ExpectedParticle(
            content="x",
            confidence_min=0.5,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
        )
        assert ep.required is True  # default

    def test_expected_particle_is_frozen(self) -> None:
        ep = _make_expected("x")
        with pytest.raises((AttributeError, TypeError)):
            ep.content = "y"  # type: ignore[misc]

    def test_benchmark_suite_defaults(self) -> None:
        suite = BenchmarkSuite(
            suite_id="s",
            name="n",
            version="0.1.0",
            domain="d",
            source_types=["X"],
            cases=[],
        )
        assert suite.metrics == []
        assert suite.published_by == ""
        assert suite.published_at is None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


_MINIMAL_SUITE = """\
suite_id: minimal-001
name: Minimal Test Suite
version: 0.1.0
domain: test
source_types: [TEST_BLOB]
cases:
  - case_id: case-1
    fixture: some-fixture
    expected:
      - content: "Hello"
        confidence_min: 0.5
        uncertainty_nature: EPISTEMIC
        required: true
"""


class TestLoader:
    def test_load_minimal_suite(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "s.yaml", _MINIMAL_SUITE)
        suite = load_suite(path)
        assert suite.suite_id == "minimal-001"
        assert suite.source_types == ["TEST_BLOB"]
        assert len(suite.cases) == 1
        case = suite.cases[0]
        assert case.case_id == "case-1"
        assert case.fixture == "some-fixture"
        assert case.source_snapshot is None
        assert len(case.expected) == 1
        assert case.expected[0].uncertainty_nature is UncertaintyNature.EPISTEMIC

    def test_load_suite_missing_field(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "s.yaml",
            "suite_id: x\nname: x\nversion: 1\ndomain: x\nsource_types: []\n",
        )
        with pytest.raises(SuiteLoadError, match="cases"):
            load_suite(path)

    def test_load_suite_invalid_uncertainty_nature(self, tmp_path: Path) -> None:
        body = _MINIMAL_SUITE.replace("EPISTEMIC", "NOT_AN_ENUM")
        path = _write(tmp_path, "s.yaml", body)
        with pytest.raises(SuiteLoadError, match="uncertainty_nature"):
            load_suite(path)

    def test_load_suite_rejects_unknown_expected_field(self, tmp_path: Path) -> None:
        body = _MINIMAL_SUITE.replace(
            "        required: true",
            "        required: true\n        bogus: 1",
        )
        path = _write(tmp_path, "s.yaml", body)
        with pytest.raises(SuiteLoadError, match="unknown"):
            load_suite(path)

    def test_load_suite_fixture_and_snapshot_mutually_exclusive(self, tmp_path: Path) -> None:
        body = _MINIMAL_SUITE.replace(
            "    fixture: some-fixture",
            "    fixture: some-fixture\n    source_snapshot:\n      content_hash: " + ("a" * 64),
        )
        path = _write(tmp_path, "s.yaml", body)
        with pytest.raises(SuiteLoadError, match="mutually exclusive"):
            load_suite(path)

    def test_load_suite_requires_fixture_or_snapshot(self, tmp_path: Path) -> None:
        body = _MINIMAL_SUITE.replace("    fixture: some-fixture\n", "")
        path = _write(tmp_path, "s.yaml", body)
        with pytest.raises(SuiteLoadError, match="fixture' or 'source_snapshot"):
            load_suite(path)

    def test_unknown_root_key_is_warning_not_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        body = _MINIMAL_SUITE + "future_field: 42\n"
        path = _write(tmp_path, "s.yaml", body)
        suite = load_suite(path)  # does NOT raise
        assert suite.suite_id == "minimal-001"
        assert any("unrecognised root-level field" in r.message for r in caplog.records)

    def test_load_suite_parses_metrics(self, tmp_path: Path) -> None:
        body = _MINIMAL_SUITE + "metrics:\n  - name: recall\n    definition: matched/required\n"
        suite = load_suite(_write(tmp_path, "s.yaml", body))
        assert suite.metrics == [RequiredMetric(name="recall", definition="matched/required")]

    def test_load_suite_parses_published_at(self, tmp_path: Path) -> None:
        body = _MINIMAL_SUITE + "published_at: 2026-05-23T00:00:00+00:00\n"
        suite = load_suite(_write(tmp_path, "s.yaml", body))
        assert suite.published_at is not None
        assert suite.published_at.tzinfo is not None

    def test_load_suite_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteLoadError, match="not found"):
            load_suite(tmp_path / "missing.yaml")

    def test_discover_suites_skips_malformed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write(tmp_path, "good.yaml", _MINIMAL_SUITE)
        _write(tmp_path, "bad.yaml", "{{ not yaml")
        _write(tmp_path, "ignored.txt", "not yaml")
        loaded = list(discover_suites(tmp_path))
        assert [s.suite_id for s in loaded] == ["minimal-001"]
        assert any("Skipping benchmark suite bad.yaml" in r.message for r in caplog.records)

    def test_discover_suites_missing_dir_empty(self, tmp_path: Path) -> None:
        assert list(discover_suites(tmp_path / "nope")) == []

    def test_resolve_case_content_against_conformance_fixture(self) -> None:
        # Uses the real numista-coin-001 fixture shipped with the extractor.
        fixture_dir = Path(__file__).parent / "conformance" / "fixtures"
        case = BenchmarkCase(
            case_id="x",
            expected=[],
            fixture="numista-coin-001",
        )
        snapshot, content, source_type = resolve_case_content(case, fixture_dir)
        assert snapshot is not None
        assert content.startswith(b"{")  # JSON
        assert source_type == "NUMISTA_API_COIN"

    def test_resolve_case_content_missing_fixture_raises(self) -> None:
        fixture_dir = Path(__file__).parent / "conformance" / "fixtures"
        case = BenchmarkCase(case_id="x", expected=[], fixture="does-not-exist")
        with pytest.raises(SuiteLoadError, match="not found"):
            resolve_case_content(case, fixture_dir)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_precision_vacuous_on_empty_emitted(self) -> None:
        assert compute_precision(0, 0) == 1.0

    def test_precision_normal(self) -> None:
        assert compute_precision(3, 5) == 0.6

    def test_recall_vacuous_on_no_required(self) -> None:
        assert compute_recall(0, 0) == 1.0

    def test_recall_normal(self) -> None:
        assert compute_recall(2, 4) == 0.5

    def test_ece_empty_is_zero(self) -> None:
        assert compute_calibration_error(set(), [], bins=10) == 0.0

    def test_ece_perfect_calibration(self) -> None:
        # All particles at confidence 0.5; exactly half match. ECE = 0.
        ps = [_make_emitted(f"c{i}", confidence=0.5) for i in range(10)]
        matched = {ps[i].id for i in range(5)}
        assert compute_calibration_error(matched, ps, bins=10) == pytest.approx(0.0, abs=1e-9)

    def test_ece_all_wrong_overconfident(self) -> None:
        # All particles at confidence 1.0; none match → ECE = 1.0
        ps = [_make_emitted(f"c{i}", confidence=1.0) for i in range(10)]
        ece = compute_calibration_error(set(), ps, bins=10)
        assert ece == pytest.approx(1.0, abs=1e-9)

    def test_ece_all_right_overconfident_is_low(self) -> None:
        # All at 0.95 confidence, all match → small ECE (|0.95 - 1.0| = 0.05)
        ps = [_make_emitted(f"c{i}", confidence=0.95) for i in range(10)]
        matched = {p.id for p in ps}
        ece = compute_calibration_error(matched, ps, bins=10)
        assert ece == pytest.approx(0.05, abs=1e-9)

    def test_ece_handles_confidence_of_exactly_one(self) -> None:
        # Bin [0.9, 1.0) excludes 1.0 by default; the function must
        # include conf==1.0 in the last bin so it isn't silently dropped.
        ps = [_make_emitted("c", confidence=1.0)]
        ece = compute_calibration_error(set(), ps, bins=10)
        # All in last bin, none matched, conf=1.0 → |1.0 - 0.0| = 1.0
        assert ece == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Equivalence — mock the similarity matrix to keep tests deterministic
# ---------------------------------------------------------------------------


class TestEquivalence:
    async def test_empty_inputs_return_empty_result(self) -> None:
        result = await match_emitted_to_expected([], [])
        assert result.matched == []
        assert result.spurious == []
        assert result.missed_required == []
        assert result.under_confidence == []

    async def test_greedy_picks_highest_similarity_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        e1 = _make_emitted("Mercury is a planet.", confidence=0.9)
        e2 = _make_emitted("Mars is a planet.", confidence=0.9)
        x1 = _make_expected("Mercury is the smallest planet.")

        # e1 is very similar to x1, e2 only weakly so. e1 should match,
        # e2 should be spurious, and x1 should NOT be in missed_required.
        sim = [[0.95], [0.50]]
        monkeypatch.setattr(
            "particles.benchmark.equivalence._similarity_matrix",
            lambda emitted_texts, expected_texts: sim,
        )

        result = await match_emitted_to_expected([e1, e2], [x1])
        assert len(result.matched) == 1
        matched_expected, matched_emitted = result.matched[0]
        assert matched_emitted.id == e1.id
        assert matched_expected.content == x1.content
        assert result.spurious == [e2]
        assert result.missed_required == []

    async def test_under_confidence_demotion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        e = _make_emitted("Mercury", confidence=0.4)
        x = _make_expected("Mercury", confidence_min=0.9)  # require ≥0.9
        monkeypatch.setattr(
            "particles.benchmark.equivalence._similarity_matrix",
            lambda emitted_texts, expected_texts: [[0.99]],
        )
        result = await match_emitted_to_expected([e], [x])
        assert result.matched == []
        assert len(result.under_confidence) == 1
        assert result.spurious == []
        # The under-confidence particle does NOT count toward
        # missed_required either (it semantically matched).
        assert result.missed_required == []

    async def test_missed_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        e = _make_emitted("Mars", confidence=0.9)
        x_required = _make_expected("Pluto", required=True)
        x_optional = _make_expected("Charon", required=False)
        monkeypatch.setattr(
            "particles.benchmark.equivalence._similarity_matrix",
            lambda emitted_texts, expected_texts: [[0.1, 0.1]],  # below threshold
        )
        result = await match_emitted_to_expected([e], [x_required, x_optional])
        assert result.matched == []
        assert result.spurious == [e]
        # Only the required expected appears in missed_required.
        assert len(result.missed_required) == 1
        assert result.missed_required[0].content == "Pluto"

    async def test_matched_ids_excludes_under_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        e1 = _make_emitted("a", confidence=0.95)
        e2 = _make_emitted("b", confidence=0.30)
        x1 = _make_expected("a", confidence_min=0.5)
        x2 = _make_expected("b", confidence_min=0.5)
        monkeypatch.setattr(
            "particles.benchmark.equivalence._similarity_matrix",
            lambda emitted_texts, expected_texts: [[0.95, 0.10], [0.10, 0.95]],
        )
        result = await match_emitted_to_expected([e1, e2], [x1, x2])
        assert e1.id in result.matched_ids
        # e2 matched semantically but under-confidence — excluded
        assert e2.id not in result.matched_ids

    async def test_threshold_filters_low_similarity_pairs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        e = _make_emitted("a", confidence=0.9)
        x = _make_expected("a")
        monkeypatch.setattr(
            "particles.benchmark.equivalence._similarity_matrix",
            lambda emitted_texts, expected_texts: [[0.79]],  # just below 0.80
        )
        result = await match_emitted_to_expected([e], [x], threshold=0.80)
        assert result.matched == []
        assert result.spurious == [e]
        assert result.missed_required == [x]

    async def test_llm_judge_called_only_in_uncertain_band(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        e1 = _make_emitted("clearly matches", confidence=0.9)
        e2 = _make_emitted("ambiguous", confidence=0.9)
        e3 = _make_emitted("clearly does not", confidence=0.9)
        x1 = _make_expected("target a")
        x2 = _make_expected("target b")
        x3 = _make_expected("target c")
        sim = [
            [0.90, 0.10, 0.10],  # e1 clearly matches x1 — no LLM call
            [0.10, 0.70, 0.10],  # e2 in uncertain band against x2 — LLM call
            [0.10, 0.10, 0.10],  # e3 too low everywhere — no LLM call
        ]
        monkeypatch.setattr(
            "particles.benchmark.equivalence._similarity_matrix",
            lambda et, xt: sim,
        )
        calls: list[tuple[str, str]] = []

        async def fake_judge(emitted: Particle, expected: ExpectedParticle) -> bool:
            calls.append((emitted.content, expected.content))
            return True

        monkeypatch.setattr("particles.benchmark.equivalence._llm_pair_aligned", fake_judge)
        await match_emitted_to_expected(
            [e1, e2, e3], [x1, x2, x3], judge=EquivalenceJudge.LLM, threshold=0.80
        )
        # Exactly one LLM call — for the ambiguous pair (e2, x2)
        assert calls == [("ambiguous", "target b")]


# ---------------------------------------------------------------------------
# MatchResult sanity
# ---------------------------------------------------------------------------


def test_match_result_matched_ids_empty() -> None:
    assert MatchResult().matched_ids == set()
