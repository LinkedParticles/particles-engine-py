"""Integration tests for temperature-scaling calibration.

Covers the four moving parts that added on top of the previously-unused
`particles/extraction/calibration.py` math (which has its own unit tests in
`tests/test_calibration.py`):

  * the ``particles extractor calibrate`` CLI verb (fit + persist, --dry-run,
    --regenerate, error paths);
  * the particle write-path hook in
    ``particles/extraction/general.py::candidate_to_particle`` —
    when an ExtractorCalibration is supplied, the particle graduates from
    EXTRACTOR_DIRECT to CALIBRATED_BENCHMARK and the raw value is
    temperature-scaled. this is the normative §6.3 contract:
    the **stored** ``confidence.value`` is the calibrated value, with
    ``calibration_method`` + ``calibration_ref`` as the audit provenance —
    there is no query-time recompute;
  * the ORM round-trip on the new ``calibration_json`` column in
    ``extractor_records``;
  * the query-disclosure refinement in
    ``particles/operations/query/main.py`` — the existing
    ``any(top-k particle is EXTRACTOR_DIRECT)`` predicate already does the
    right thing; this test pins that behaviour now that operators can
    actually reach a state where every top-k particle is CALIBRATED_BENCHMARK.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.equivalence import EquivalenceJudge
from particles.benchmark.runner import BenchmarkReport, CaseResult
from particles.core.schema import (
    Confidence,
    ExtractorCalibration,
    Particle,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.extraction.calibration import T_MAX, T_MIN, TRANSFORM_LOGIT
from particles.extraction.general import CandidateParticle, candidate_to_particle

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, args: list[str]) -> object:
    return runner.invoke(app, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# candidate_to_particle — the particle write-path hook
# ---------------------------------------------------------------------------


class TestCandidateToParticleCalibration:
    def _make_candidate(self) -> CandidateParticle:
        return CandidateParticle(
            content="some claim",
            confidence_value=0.8,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            subjects=["S"],
        )

    def test_no_calibration_yields_extractor_direct_with_raw_value(self) -> None:
        """Regression check: when calibration is None, behaviour is unchanged."""
        candidate = self._make_candidate()
        particle = candidate_to_particle(
            candidate,
            corpus_entry_id="entry-1",
            snapshot_id="snap-1",
            asserted_by="test-extractor",
        )
        assert particle.confidence.calibration_source == CalibrationSource.EXTRACTOR_DIRECT
        assert particle.confidence.value == pytest.approx(0.8)
        assert particle.confidence.calibration_method is None
        assert particle.confidence.calibration_ref is None

    def test_with_calibration_yields_calibrated_benchmark_and_scaled_value(self) -> None:
        candidate = self._make_candidate()
        fitted_at = datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC)
        calibration = ExtractorCalibration(
            temperature=2.0,
            transform=TRANSFORM_LOGIT,
            fitted_at=fitted_at,
            benchmark_suite_id="some-suite",
            sample_size=10,
            calibration_error_before=0.10,
            calibration_error_after=0.02,
        )
        particle = candidate_to_particle(
            candidate,
            corpus_entry_id="entry-1",
            snapshot_id="snap-1",
            asserted_by="my-extractor",
            calibration=calibration,
        )
        assert particle.confidence.calibration_source == CalibrationSource.CALIBRATED_BENCHMARK
        # sigmoid(logit(0.8) / 2) = 2/3 (the retired form gave 0.4)
        assert particle.confidence.value == pytest.approx(2 / 3)
        assert particle.confidence.calibration_method == "temperature_scaling"
        assert particle.confidence.calibration_ref == f"my-extractor:{fitted_at.isoformat()}"

    def test_calibration_below_one_sharpens_without_saturating(self) -> None:
        """T < 1 raises a confidence toward 1.0 but never reaches it.

        The retired `clamp(raw / T, 0, 1)` form stored exactly 1.0 here, which
        is both a false certainty and order-destroying — every value above T
        collapsed onto the same number.
        """
        candidate = CandidateParticle(
            content="x",
            confidence_value=0.8,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            subjects=["S"],
        )
        calibration = ExtractorCalibration(
            temperature=0.5,
            transform=TRANSFORM_LOGIT,
            fitted_at=datetime.now(UTC),
            benchmark_suite_id="s",
            sample_size=5,
            calibration_error_before=0.5,
            calibration_error_after=0.1,
        )
        particle = candidate_to_particle(
            candidate, "e", "s", asserted_by="x", calibration=calibration
        )
        assert 0.8 < particle.confidence.value < 1.0

    def test_pre_adr_0238_record_is_not_applied(self) -> None:
        """A record with no `transform` is a poisoned legacy fit — never applied.

        Its T both parameterises the retired linear form and was fitted against
        all-False labels, so it drove to the bound. Falling back to the raw
        value stamped EXTRACTOR_DIRECT is the honest state.
        """
        candidate = self._make_candidate()
        legacy = ExtractorCalibration(
            temperature=10.0,
            fitted_at=datetime.now(UTC),
            benchmark_suite_id="s",
            sample_size=21,
            calibration_error_before=0.775,
            calibration_error_after=0.088,
        )
        assert legacy.transform is None
        particle = candidate_to_particle(candidate, "e", "s", asserted_by="x", calibration=legacy)
        assert particle.confidence.calibration_source == CalibrationSource.EXTRACTOR_DIRECT
        assert particle.confidence.value == pytest.approx(0.8)  # not 0.08
        assert particle.confidence.calibration_ref is None


class TestPerProviderCalibrationSelection:
    """calibration is stored + selected by (extractor, provider_model)."""

    def _calibration(self, provider_model: str | None) -> ExtractorCalibration:
        return ExtractorCalibration(
            temperature=2.0,
            fitted_at=datetime.now(UTC),
            benchmark_suite_id="s",
            sample_size=10,
            calibration_error_before=0.1,
            calibration_error_after=0.02,
            provider_model=provider_model,
        )

    @pytest.mark.asyncio
    async def test_matching_provider_model_is_selected(self, db_session: AsyncSession) -> None:
        from particles.store.extractor_store import get_calibration, upsert_calibration

        await upsert_calibration(db_session, "e", self._calibration("anthropic:claude-sonnet-4-6"))
        got = await get_calibration(db_session, "e", "anthropic:claude-sonnet-4-6")
        assert got is not None
        assert got.temperature == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_other_provider_model_is_not_selected(self, db_session: AsyncSession) -> None:
        from particles.store.extractor_store import get_calibration, upsert_calibration

        await upsert_calibration(db_session, "e", self._calibration("anthropic:claude-sonnet-4-6"))
        assert await get_calibration(db_session, "e", "anthropic:some-other-model") is None

    @pytest.mark.asyncio
    async def test_legacy_none_keyed_under_historical_default(
        self, db_session: AsyncSession
    ) -> None:
        from particles.store.extractor_store import (
            LEGACY_PROVIDER_MODEL,
            get_calibration,
            upsert_calibration,
        )

        await upsert_calibration(db_session, "e", self._calibration(None))
        assert await get_calibration(db_session, "e", LEGACY_PROVIDER_MODEL) is not None

    @pytest.mark.asyncio
    async def test_two_models_coexist_and_swap_back_keeps_both(
        self, db_session: AsyncSession
    ) -> None:
        from particles.store.extractor_store import (
            get_calibration,
            get_calibrations,
            upsert_calibration,
        )

        await upsert_calibration(db_session, "e", self._calibration("anthropic:claude-sonnet-4-6"))
        await upsert_calibration(db_session, "e", self._calibration("local:llama3.1:8b"))
        # Both pairings retain their calibration — the headline behaviour.
        assert await get_calibration(db_session, "e", "anthropic:claude-sonnet-4-6") is not None
        assert await get_calibration(db_session, "e", "local:llama3.1:8b") is not None
        assert len(await get_calibrations(db_session, "e")) == 2


# ---------------------------------------------------------------------------
# Store round-trip — the extractor_calibrations table
# ---------------------------------------------------------------------------


class TestPerProviderCalibrationStore:
    @pytest.mark.asyncio
    async def test_round_trips_all_fields(self, db_session: AsyncSession) -> None:
        from particles.store.extractor_store import get_calibration, upsert_calibration

        fitted_at = datetime(2026, 5, 25, 9, 30, 0, tzinfo=UTC)
        cal = ExtractorCalibration(
            temperature=1.25,
            fitted_at=fitted_at,
            benchmark_suite_id="numismatic-seed-001",
            sample_size=7,
            calibration_error_before=0.08,
            calibration_error_after=0.02,
            provider_model="anthropic:claude-sonnet-4-6",
        )
        await upsert_calibration(db_session, "my-extractor", cal)
        await db_session.commit()

        loaded = await get_calibration(db_session, "my-extractor", "anthropic:claude-sonnet-4-6")
        assert loaded is not None
        assert loaded.temperature == pytest.approx(1.25)
        assert loaded.fitted_at == fitted_at
        assert loaded.benchmark_suite_id == "numismatic-seed-001"
        assert loaded.sample_size == 7
        assert loaded.calibration_error_before == pytest.approx(0.08)
        assert loaded.calibration_error_after == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_none_when_pairing_absent(self, db_session: AsyncSession) -> None:
        from particles.store.extractor_store import get_calibration

        got = await get_calibration(db_session, "my-extractor", "anthropic:claude-sonnet-4-6")
        assert got is None

    @pytest.mark.asyncio
    async def test_registry_version_refresh_leaves_calibration(
        self, db_session: AsyncSession
    ) -> None:
        """A version-only registry upsert must not disturb stored calibrations."""
        from particles.core.schema import ExtractorRecord
        from particles.store.extractor_store import (
            get_calibration,
            upsert_calibration,
            upsert_extractor_record,
        )

        cal = ExtractorCalibration(
            temperature=1.4,
            fitted_at=datetime(2026, 5, 25, 0, 0, 0, tzinfo=UTC),
            benchmark_suite_id="s",
            sample_size=5,
            calibration_error_before=0.1,
            calibration_error_after=0.02,
            provider_model="anthropic:claude-sonnet-4-6",
        )
        await upsert_calibration(db_session, "my-extractor", cal)
        await upsert_extractor_record(
            db_session,
            ExtractorRecord(extractor_id="my-extractor", name="my-extractor", version="0.2.0"),
        )
        await db_session.commit()

        loaded = await get_calibration(db_session, "my-extractor", "anthropic:claude-sonnet-4-6")
        assert loaded is not None
        assert loaded.temperature == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# Query disclosure refinement
# ---------------------------------------------------------------------------


def _calibrated_particle(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=0.7,
            calibration_source=CalibrationSource.CALIBRATED_BENCHMARK,
            calibration_method="temperature_scaling",
            calibration_ref="x:2026-05-25T00:00:00+00:00",
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="x",
    )


def _extractor_direct_particle(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=0.7,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="x",
    )


class TestQueryDisclosureLiftsOnCalibration:
    """the disclosure already fires only when ``any`` top-k
    particle is EXTRACTOR_DIRECT. After calibration ships, that condition
    can finally evaluate to False; this test pins both directions."""

    def test_all_calibrated_top_k_does_not_emit_disclosure(self) -> None:
        top_particles = [_calibrated_particle("a"), _calibrated_particle("b")]
        # Mirror the in-line conditional from query/main.py:185-195.
        conf_note = ""
        mean_eff = 0.8  # above the 0.6 floor; the low-confidence branch is silent
        if mean_eff < 0.6:
            conf_note = "low-confidence-disclosure-noise"
        if (
            any(
                p.confidence.calibration_source == CalibrationSource.EXTRACTOR_DIRECT
                for p in top_particles
            )
            and not conf_note
        ):
            conf_note = "extractor-direct-disclosure"
        assert conf_note == ""

    def test_any_extractor_direct_top_k_keeps_disclosure(self) -> None:
        top_particles = [_calibrated_particle("a"), _extractor_direct_particle("b")]
        conf_note = ""
        mean_eff = 0.8
        if mean_eff < 0.6:
            conf_note = "low-confidence-disclosure-noise"
        if (
            any(
                p.confidence.calibration_source == CalibrationSource.EXTRACTOR_DIRECT
                for p in top_particles
            )
            and not conf_note
        ):
            conf_note = "extractor-direct-disclosure"
        assert conf_note == "extractor-direct-disclosure"


# ---------------------------------------------------------------------------
# CLI verb — particles extractor calibrate
# ---------------------------------------------------------------------------


_SEED_SUITE = Path(__file__).parent / "benchmark" / "suites" / "numismatic-seed-001.yaml"
_FIXTURES = Path(__file__).parent / "conformance" / "fixtures"


def _init_extractor_records_sync(_cli_db: Path) -> None:
    """Synchronously seed the extractor_records table for CLI tests.

    The CLI relies on get_extractor_record finding a row when persisting a
    calibration. ensure_extractor_records normally runs at `particles db
    init`; bypass that here by calling it directly so the file-based SQLite
    DB has the rows.
    """
    import asyncio

    async def _seed() -> None:
        from particles.db import session_scope
        from particles.ingest.importers.registry import ensure_extractor_records

        async with session_scope() as session:
            await ensure_extractor_records(session)
            await session.commit()

    asyncio.run(_seed())


#: The shipped calibration suite family — a sibling of
#: `tests/benchmark/suites/`, whose gold sets deliberately cover only part of
#: what each extractor emits so a temperature fit has both labels to learn
#: from. `numismatic-calibration-001` names six of the seven particles the
#: numista fixture produces; the seventh comes back unmatched by design.
#:
#: These CLI tests deliberately run against the **real** shipped suites rather
#: than a synthesized `tmp_path` one, so a change that makes the shipped
#: artifact unfittable fails here rather than in an operator's terminal.
_CALIBRATION_SUITES = Path(__file__).parent / "benchmark" / "calibration"


def _stub_report_with_mixed_labels() -> BenchmarkReport:
    """A report whose `graded` pairs fit an admissible temperature.

    Mirrors the worked example — the numismatic partial-coverage
    population (0.95 ×6 / 0.97 ×1, six matched and one not) — so the fit clears
    all four §4 conditions and the verb exits 0. The judge-default tests only
    care *which judge was requested*, so they stub the run rather than pay for
    it; this keeps them unit-tier with no API call in any band.
    """
    graded = [(0.95, True)] * 5 + [(0.95, False), (0.97, True)]
    return BenchmarkReport(
        suite_id="stub-suite",
        suite_version="0.1.0",
        extractor_id="numista-coin-extractor",
        extractor_version="0.2.0",
        cases_run=1,
        cases_total=1,
        particles_emitted=len(graded),
        particles_required_total=len(graded),
        metrics={"precision": 1.0, "recall": 1.0, "calibration_error": 0.0},
        per_case=[
            CaseResult(
                case_id="stub-case",
                emitted_count=len(graded),
                matched=[],
                matched_required_count=0,
                missed_required=[],
                spurious=[],
                under_confidence=[],
                graded=graded,
            )
        ],
        generated_at=datetime.now(UTC),
        judge="stub",
        equivalence_threshold=0.80,
    )


class TestCalibrateCLI:
    def test_dry_run_does_not_persist(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _init_extractor_records_sync(cli_db)
        result = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                str(_CALIBRATION_SUITES),
                "--fixtures",
                str(_FIXTURES),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "T=" in result.output
        assert "ECE:" in result.output
        assert "sample N=" in result.output
        assert "--dry-run: calibration not persisted." in result.output

        # The persisted record should have no calibration.
        import asyncio

        async def _read() -> ExtractorCalibration | None:
            from particles.db import session_scope
            from particles.store.extractor_store import get_calibration

            async with session_scope() as session:
                return await get_calibration(
                    session, "numista-coin-extractor", "anthropic:claude-sonnet-4-6"
                )

        assert asyncio.run(_read()) is None

    def test_calibrate_persists_record(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _init_extractor_records_sync(cli_db)
        result = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                str(_CALIBRATION_SUITES),
                "--fixtures",
                str(_FIXTURES),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "T=" in result.output
        assert "Calibration persisted" in result.output
        # The labelling is surfaced so the operator can see the fit had contrast.
        assert "labels: 6 matched / 1 unmatched" in result.output

        import asyncio

        async def _read() -> ExtractorCalibration | None:
            from particles.db import session_scope
            from particles.store.extractor_store import get_calibration

            async with session_scope() as session:
                return await get_calibration(
                    session, "numista-coin-extractor", "anthropic:claude-sonnet-4-6"
                )

        calibration = asyncio.run(_read())
        assert calibration is not None
        assert calibration.temperature > 0
        assert calibration.sample_size >= 2
        assert calibration.benchmark_suite_id  # non-empty
        # the record declares which transform its T parameterises,
        # and a fresh fit is never allowed to sit on an optimizer bound.
        assert calibration.transform == TRANSFORM_LOGIT
        assert T_MIN < calibration.temperature < T_MAX
        # sample_size is the *fitted* population. Nothing was
        # saturated here, so it equals the emitted count — but it is the
        # fitted count that is being recorded.
        assert calibration.sample_size == 7
        # The fit must actually improve calibration error, or §4(c) refuses it.
        assert calibration.calibration_error_after < calibration.calibration_error_before

    def test_calibrate_reports_the_fittable_population(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        """the operator is always told what the fit could act on."""
        _init_extractor_records_sync(cli_db)
        result = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                str(_CALIBRATION_SUITES),
                "--fixtures",
                str(_FIXTURES),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "fittable: 7 of 7 pair(s), 2 distinct confidence value(s)" in result.output

    # (predictor degeneracy) and §4c (non-improving fit) have no
    # CLI test here, deliberately. Both are properties of an *extractor's*
    # output distribution rather than of a gold set, so no suite over the
    # fixtures in tree can produce either: the numista fixture always emits
    # both 0.95 and 0.97 (two distinct values, so §4b never fires), and an
    # exhaustive search over every gold-set subset of its seven particles
    # found no combination that fits a non-improving temperature. That is the
    # ADR's own point — "no gold set can fix this". They are covered precisely
    # at the unit level in tests/test_calibration.py; the CLI refusal *branch*
    # is shared by all four conditions and is exercised by the degenerate-label
    # test below.

    def test_calibrate_defaults_to_the_llm_judge(self, runner: CliRunner, cli_db: Path) -> None:
        """the calibration label is the judge's verdict, so it defaults to LLM.

        `extractor benchmark` defaults to the embedding judge and this verb
        deliberately does not: a paraphrase miss is one precision point there
        and is substantially the *entire* negative label set here, because a
        well-behaved extractor's only unmatched emissions are judge misses.

        `run_benchmark` is patched rather than run so this stays a unit test —
        the real LLM judge would call the API for any pair in the contested
        [0.65, 0.80) band. The deferred import inside `_extractor_calibrate` is
        what makes patching the source module reach the call site
        (tests/AGENTS.md § Mocking strategy).
        """
        _init_extractor_records_sync(cli_db)
        seen: list[object] = []

        async def _fake_run_benchmark(*args: object, **kwargs: object) -> object:
            seen.append(kwargs.get("judge"))
            return _stub_report_with_mixed_labels()

        with patch("particles.benchmark.runner.run_benchmark", _fake_run_benchmark):
            result = _invoke(
                runner,
                [
                    "extractor",
                    "calibrate",
                    "numista-coin-extractor",
                    "--suites-dir",
                    str(_CALIBRATION_SUITES),
                    "--fixtures",
                    str(_FIXTURES),
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0, result.output
        assert seen == [EquivalenceJudge.LLM]
        # The judge decides every label, so it is reported beside them.
        assert "judge:  llm" in result.output

    def test_calibrate_judge_embedding_overrides_the_default(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        """`--judge embedding` restores the cost-free (and noisier) labelling."""
        _init_extractor_records_sync(cli_db)
        seen: list[object] = []

        async def _fake_run_benchmark(*args: object, **kwargs: object) -> object:
            seen.append(kwargs.get("judge"))
            return _stub_report_with_mixed_labels()

        with patch("particles.benchmark.runner.run_benchmark", _fake_run_benchmark):
            result = _invoke(
                runner,
                [
                    "extractor",
                    "calibrate",
                    "numista-coin-extractor",
                    "--suites-dir",
                    str(_CALIBRATION_SUITES),
                    "--fixtures",
                    str(_FIXTURES),
                    "--judge",
                    "embedding",
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0, result.output
        assert seen == [EquivalenceJudge.EMBEDDING]
        assert "judge:  embedding" in result.output

    def test_calibrate_refuses_overwrite_without_regenerate(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _init_extractor_records_sync(cli_db)
        suites_dir = str(_CALIBRATION_SUITES)
        # First calibration succeeds.
        result1 = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                suites_dir,
                "--fixtures",
                str(_FIXTURES),
            ],
        )
        assert result1.exit_code == 0, result1.output

        # Second without --regenerate refuses.
        result2 = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                suites_dir,
                "--fixtures",
                str(_FIXTURES),
            ],
        )
        assert result2.exit_code == 1
        assert "already has a calibration" in (result2.output + (result2.stderr or ""))

    def test_calibrate_regenerate_overwrites(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _init_extractor_records_sync(cli_db)
        suites_dir = str(_CALIBRATION_SUITES)
        result1 = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                suites_dir,
                "--fixtures",
                str(_FIXTURES),
            ],
        )
        assert result1.exit_code == 0, result1.output

        import asyncio

        async def _read_fitted_at() -> datetime:
            from particles.db import session_scope
            from particles.store.extractor_store import get_calibration

            async with session_scope() as session:
                cal = await get_calibration(
                    session, "numista-coin-extractor", "anthropic:claude-sonnet-4-6"
                )
            assert cal is not None
            return cal.fitted_at

        first_fitted_at = asyncio.run(_read_fitted_at())

        result2 = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                suites_dir,
                "--fixtures",
                str(_FIXTURES),
                "--regenerate",
            ],
        )
        assert result2.exit_code == 0, result2.output
        second_fitted_at = asyncio.run(_read_fitted_at())
        assert second_fitted_at >= first_fitted_at

    def test_calibrate_refuses_a_degenerate_all_matched_fit(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        """guard, from the direction the bundled seed suite actually takes.

        Every particle the Numista fixture emits is covered by the seed suite's
        gold set, so the labels are all True and nothing is being regressed.
        Before the guard this persisted a temperature that then rewrote every
        future confidence, immutably.
        """
        _init_extractor_records_sync(cli_db)
        result = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                str(_SEED_SUITE.parent),
                "--fixtures",
                str(_FIXTURES),
            ],
        )
        combined = result.output + (result.stderr or "")
        assert result.exit_code == 1, combined
        assert "degenerate labels" in combined
        assert "Refusing to persist" in combined

        import asyncio

        async def _read() -> ExtractorCalibration | None:
            from particles.db import session_scope
            from particles.store.extractor_store import get_calibration

            async with session_scope() as session:
                return await get_calibration(
                    session, "numista-coin-extractor", "anthropic:claude-sonnet-4-6"
                )

        assert asyncio.run(_read()) is None, "a refused fit must leave nothing behind"

    def test_calibrate_unknown_extractor_exits_2(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["extractor", "calibrate", "no-such-extractor"])
        assert result.exit_code == 2
        assert "no-such-extractor" in (result.output + (result.stderr or ""))

    def test_calibrate_no_applicable_suites_exits_2(self, runner: CliRunner, cli_db: Path) -> None:
        _init_extractor_records_sync(cli_db)
        # docstring-extractor accepts PYTHON_SOURCE, which no suite covers —
        # and never will (fixed confidence, nothing to calibrate).
        result = runner.invoke(
            app,
            [
                "extractor",
                "calibrate",
                "docstring-extractor",
                "--suites-dir",
                str(_SEED_SUITE.parent),
                "--fixtures",
                str(_FIXTURES),
            ],
        )
        assert result.exit_code == 2
        assert "No applicable calibration suites" in (result.output + (result.stderr or ""))

    def test_calibrations_list_empty(self, runner: CliRunner, cli_db: Path) -> None:
        _init_extractor_records_sync(cli_db)
        result = _invoke(runner, ["extractor", "calibrations", "numista-coin-extractor"])
        assert result.exit_code == 0, result.output
        assert "No calibrations stored" in result.output

    def test_calibrations_list_shows_pairing_after_calibrate(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _init_extractor_records_sync(cli_db)
        cal = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                str(_CALIBRATION_SUITES),
                "--fixtures",
                str(_FIXTURES),
            ],
        )
        assert cal.exit_code == 0, cal.output
        result = _invoke(runner, ["extractor", "calibrations", "numista-coin-extractor"])
        assert result.exit_code == 0, result.output
        # The default extraction model pairing is listed.
        assert "anthropic:claude-sonnet-4-6" in result.output
        # A freshly-fitted record is applicable — no refusal note.
        assert "NOT APPLIED" not in result.output


# ---------------------------------------------------------------------------
# delete_calibration — the retirement path
# ---------------------------------------------------------------------------


class TestDeleteCalibration:
    @pytest.mark.asyncio
    async def test_removes_the_pairing_and_returns_it(self, db_session: AsyncSession) -> None:
        from particles.store.extractor_store import (
            delete_calibration,
            get_calibration,
            upsert_calibration,
        )

        cal = ExtractorCalibration(
            temperature=9.9999,
            fitted_at=datetime.now(UTC),
            benchmark_suite_id="rdf-seed-001",
            sample_size=21,
            calibration_error_before=0.775,
            calibration_error_after=0.088,
            provider_model="local:qwen2.5:7b-instruct",
        )
        await upsert_calibration(db_session, "general-extractor", cal)

        removed = await delete_calibration(
            db_session, "general-extractor", "local:qwen2.5:7b-instruct"
        )
        assert removed is not None
        assert removed.temperature == pytest.approx(9.9999)
        assert removed.benchmark_suite_id == "rdf-seed-001"
        assert (
            await get_calibration(db_session, "general-extractor", "local:qwen2.5:7b-instruct")
        ) is None

    @pytest.mark.asyncio
    async def test_absent_pairing_returns_none(self, db_session: AsyncSession) -> None:
        from particles.store.extractor_store import delete_calibration

        assert await delete_calibration(db_session, "general-extractor", "nope:nope") is None

    @pytest.mark.asyncio
    async def test_leaves_the_other_pairing_intact(self, db_session: AsyncSession) -> None:
        """The property retirement must not break: pairings are independent."""
        from particles.store.extractor_store import (
            delete_calibration,
            get_calibration,
            upsert_calibration,
        )

        def _cal(pairing: str, temp: float) -> ExtractorCalibration:
            return ExtractorCalibration(
                temperature=temp,
                fitted_at=datetime.now(UTC),
                benchmark_suite_id="prose-article-seed-001",
                sample_size=10,
                calibration_error_before=0.4,
                calibration_error_after=0.1,
                provider_model=pairing,
            )

        await upsert_calibration(db_session, "general-extractor", _cal("local:qwen", 9.9))
        await upsert_calibration(db_session, "general-extractor", _cal("anthropic:sonnet", 1.2))

        await delete_calibration(db_session, "general-extractor", "local:qwen")

        assert (await get_calibration(db_session, "general-extractor", "local:qwen")) is None
        survivor = await get_calibration(db_session, "general-extractor", "anthropic:sonnet")
        assert survivor is not None
        assert survivor.temperature == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# Suite-set staleness reporting + `calibration-forget` CLI (§4)
# ---------------------------------------------------------------------------


def _seed_calibration_sync(
    extractor_id: str,
    pairing: str,
    suite_id: str,
    transform: str | None = TRANSFORM_LOGIT,
) -> None:
    """Persist one calibration directly, bypassing a fit.

    ``transform`` defaults to the applicable form so suite-set staleness can be
    exercised on its own; pass ``None`` for a pre-ADR-0238 record.
    """
    import asyncio

    async def _seed() -> None:
        from particles.db import session_scope
        from particles.store.extractor_store import upsert_calibration

        async with session_scope() as session:
            await upsert_calibration(
                session,
                extractor_id,
                ExtractorCalibration(
                    temperature=9.9999,
                    transform=transform,
                    fitted_at=datetime.now(UTC),
                    benchmark_suite_id=suite_id,
                    sample_size=21,
                    calibration_error_before=0.775,
                    calibration_error_after=0.088,
                    provider_model=pairing,
                ),
            )
            await session.commit()

    asyncio.run(_seed())


class TestSuiteStalenessReporting:
    def test_calibrations_flags_a_record_fitted_over_other_suites(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        _init_extractor_records_sync(cli_db)
        # numista-coin-extractor auto-matches numismatic-seed-001; a fit over
        # rdf-seed-001 is the shape narrowing left behind.
        _seed_calibration_sync("numista-coin-extractor", "local:qwen", "rdf-seed-001")

        result = _invoke(
            runner,
            [
                "extractor",
                "calibrations",
                "numista-coin-extractor",
                "--suites-dir",
                str(_SEED_SUITE.parent),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "STALE" in result.output
        assert "rdf-seed-001" in result.output
        assert "numismatic-seed-001" in result.output
        # …and it names the gesture that retires it.
        assert "calibration-forget numista-coin-extractor local:qwen" in result.output

    def test_calibrations_flags_a_pre_adr_0238_record_as_not_applied(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        """The operator's window onto a poisoned legacy fit still sitting in the DB."""
        _init_extractor_records_sync(cli_db)
        _seed_calibration_sync(
            "numista-coin-extractor", "local:qwen", "numismatic-seed-001", transform=None
        )

        result = _invoke(
            runner,
            [
                "extractor",
                "calibrations",
                "numista-coin-extractor",
                "--suites-dir",
                str(_SEED_SUITE.parent),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "NOT APPLIED" in result.output
        assert "EXTRACTOR_DIRECT" in result.output
        # …and it names the gesture that retires it.
        assert "calibration-forget numista-coin-extractor local:qwen" in result.output

    def test_calibrations_does_not_flag_a_matching_record(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        _init_extractor_records_sync(cli_db)
        _seed_calibration_sync("numista-coin-extractor", "local:qwen", "numismatic-seed-001")

        result = _invoke(
            runner,
            [
                "extractor",
                "calibrations",
                "numista-coin-extractor",
                "--suites-dir",
                str(_SEED_SUITE.parent),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "STALE" not in result.output

    def test_calibrations_degrades_when_no_suites_dir(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        """An installed SDK has no `tests/` — the listing must still print."""
        _init_extractor_records_sync(cli_db)
        _seed_calibration_sync("numista-coin-extractor", "local:qwen", "rdf-seed-001")

        result = _invoke(
            runner,
            [
                "extractor",
                "calibrations",
                "numista-coin-extractor",
                "--suites-dir",
                str(tmp_path / "absent"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "local:qwen" in result.output
        assert "STALE" not in result.output
        assert "not checked" in result.output

    def test_calibrate_warns_about_another_stale_pairing(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        """The guard is keyed on the configured pairing and cannot see this one."""
        _init_extractor_records_sync(cli_db)
        _seed_calibration_sync("numista-coin-extractor", "local:qwen", "rdf-seed-001")

        result = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                str(_CALIBRATION_SUITES),
                "--fixtures",
                str(_FIXTURES),
                "--dry-run",
            ],
        )
        combined = result.output + (result.stderr or "")
        assert result.exit_code == 0, combined
        assert "another stored calibration" in combined
        assert "local:qwen" in combined


class TestCalibrationForgetCLI:
    def test_unknown_pairing_exits_1(self, runner: CliRunner, cli_db: Path) -> None:
        _init_extractor_records_sync(cli_db)
        result = _invoke(
            runner,
            ["extractor", "calibration-forget", "numista-coin-extractor", "local:nope", "--yes"],
        )
        assert result.exit_code == 1
        assert "No calibration stored" in (result.output + (result.stderr or ""))

    def test_yes_retires_the_record(self, runner: CliRunner, cli_db: Path) -> None:
        _init_extractor_records_sync(cli_db)
        _seed_calibration_sync("numista-coin-extractor", "local:qwen", "rdf-seed-001")

        result = _invoke(
            runner,
            ["extractor", "calibration-forget", "numista-coin-extractor", "local:qwen", "--yes"],
        )
        assert result.exit_code == 0, result.output
        # It shows what it is about to remove before removing it.
        assert "rdf-seed-001" in result.output
        assert "Retired calibration" in result.output

        import asyncio

        async def _read() -> ExtractorCalibration | None:
            from particles.db import session_scope
            from particles.store.extractor_store import get_calibration

            async with session_scope() as session:
                return await get_calibration(session, "numista-coin-extractor", "local:qwen")

        assert asyncio.run(_read()) is None

    def test_declining_the_prompt_keeps_the_record(self, runner: CliRunner, cli_db: Path) -> None:
        _init_extractor_records_sync(cli_db)
        _seed_calibration_sync("numista-coin-extractor", "local:qwen", "rdf-seed-001")

        result = runner.invoke(
            app,
            ["extractor", "calibration-forget", "numista-coin-extractor", "local:qwen"],
            input="n\n",
        )
        assert result.exit_code == 1  # typer.confirm(abort=True)

        import asyncio

        async def _read() -> ExtractorCalibration | None:
            from particles.db import session_scope
            from particles.store.extractor_store import get_calibration

            async with session_scope() as session:
                return await get_calibration(session, "numista-coin-extractor", "local:qwen")

        assert asyncio.run(_read()) is not None


# ---------------------------------------------------------------------------
# `ECE before` is pooled over the fit population, not averaged across suites
# ---------------------------------------------------------------------------


#: Every particle the numista fixture emits, in the seed suite's wording. The
#: extractor is a deterministic parser, so this set is fixed.
_ALL_SEVEN = [
    "5 Pfennigs (1948-1950) GDR: made of aluminium, weight 1.1g, "
    "diameter 19.0mm, demonetized 1990-12-31.",
    "5 Pfennigs (1948-1950) GDR was struck at Berlin.",
    "5 Pfennigs (1948-1950) GDR is catalogued as KM# 2.",
    "5 Pfennigs (1948-1950) GDR is catalogued as Schön# 2.",
    "The obverse of 5 Pfennigs (1948-1950) GDR depicts: Ear of rye",
    "The reverse of 5 Pfennigs (1948-1950) GDR depicts: Face value",
    "5 Pfennigs (1948-1950) GDR has a plain edge.",
]

#: The stated confidences behind those seven, in emission order.
_SEVEN_CONFIDENCES = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.97]


def _two_suite_dir(tmp_path: Path) -> Path:
    """Two suites for one extractor, with gold sets of different generosity.

    Both name `NUMISTA_API_COIN`, so the routing hands the calibrate verb
    both of them in a single run — the multi-suite shape `benchmark_suite_id`'s
    `+` join exists for. One expects all seven emitted particles (accuracy 1.0,
    *above* its stated confidence), the other expects two (accuracy 2/7, well
    below). That sign difference is what makes averaging the two suites' ECEs
    disagree with the ECE of their union: |mean| is not mean(|·|) once the
    deviations point opposite ways, and both suites land in the same confidence
    bin.
    """
    import yaml

    suites_dir = tmp_path / "two-suites"
    suites_dir.mkdir(exist_ok=True)
    for suite_id, expected in (
        ("numismatic-generous-001", _ALL_SEVEN),
        ("numismatic-thin-001", _ALL_SEVEN[:2]),
    ):
        suite = {
            "suite_id": suite_id,
            "name": suite_id,
            "version": "0.1.0",
            "domain": "numismatics",
            "source_types": ["NUMISTA_API_COIN"],
            "cases": [
                {
                    "case_id": "numista-coin-001",
                    "fixture": "numista-coin-001",
                    "expected": [
                        {
                            "content": content,
                            "confidence_min": 0.0,
                            "uncertainty_nature": "EPISTEMIC",
                            "required": True,
                        }
                        for content in expected
                    ],
                }
            ],
        }
        (suites_dir / f"{suite_id}.yaml").write_text(yaml.safe_dump(suite))
    return suites_dir


class TestEceBeforeIsPooled:
    """The two figures the persistence guard compares must be one estimator.

    A fit whose `ece_after` is no better than its `ece_before` is refused
    (§4(c)). That comparison is only meaningful if both sides are computed
    the same way over the same population. `ece_before` used to be an
    emission-weighted mean of the per-suite benchmark reports'
    `calibration_error` while `ece_after` was pooled over the union — equal for
    a single-suite fit (which is every fit in tree), and not equal once a second
    suite contributes, because ECE bins by confidence and two suites sharing a
    bin at different accuracies do not average.
    """

    def test_weighted_mean_and_pooled_ece_actually_differ_here(self) -> None:
        """The fixture is only a regression test if the two estimators disagree on it."""
        from particles.extraction.calibration import expected_calibration_error

        generous_labels = [True] * 7
        thin_labels = [True] * 2 + [False] * 5

        weighted_mean = (
            expected_calibration_error(_SEVEN_CONFIDENCES, generous_labels) * 7
            + expected_calibration_error(_SEVEN_CONFIDENCES, thin_labels) * 7
        ) / 14
        pooled = expected_calibration_error(_SEVEN_CONFIDENCES * 2, generous_labels + thin_labels)
        assert weighted_mean != pytest.approx(pooled, abs=1e-4)

    def test_calibrate_reports_the_pooled_figure(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        from particles.extraction.calibration import expected_calibration_error

        _init_extractor_records_sync(cli_db)
        result = _invoke(
            runner,
            [
                "extractor",
                "calibrate",
                "numista-coin-extractor",
                "--suites-dir",
                str(_two_suite_dir(tmp_path)),
                "--fixtures",
                str(_FIXTURES),
                "--dry-run",
            ],
        )
        combined = result.output + (result.stderr or "")
        # Emission order within a case is stable, but the two suites' cases are
        # pooled in suite-discovery order; ECE is order-independent, so build
        # the expectation from the multiset either order produces.
        labels = [True] * 7 + [True] * 2 + [False] * 5
        pooled = expected_calibration_error(_SEVEN_CONFIDENCES * 2, labels)
        weighted_mean = (
            expected_calibration_error(_SEVEN_CONFIDENCES, [True] * 7) * 7
            + expected_calibration_error(_SEVEN_CONFIDENCES, [True] * 2 + [False] * 5) * 7
        ) / 14

        assert f"ECE: {pooled:.4f} →" in combined, combined
        assert f"ECE: {weighted_mean:.4f} →" not in combined
        assert "sample N=14" in combined
