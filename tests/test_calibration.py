"""Tests for the temperature-scaling calibrator (particles/extraction/calibration.py).

calibration.py was 0% covered in the architecture-review baseline. The module
is a pair of pure functions (one class, one helper) with no I/O — ideal for
focused unit tests.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from particles.core.schema import ExtractorCalibration
from particles.extraction.calibration import (
    T_MAX,
    T_MIN,
    TRANSFORM_LOGIT,
    FitDiagnostics,
    TemperatureScaler,
    calibration_error,
    expected_calibration_error,
    fitted_suite_ids,
    is_saturated,
    is_suite_stale,
    scaler_for_record,
)


def _calibrated_pairs() -> tuple[list[float], list[bool]]:
    """300 realistic pairs whose empirical accuracy matches stated confidence.

    Three confidence levels, each correct at exactly its stated rate — the
    definition of a well-calibrated extractor. Overall positive rate 85%.
    A correct fit must return T ≈ 1 on this, since no rescaling is warranted.
    """
    raws = [0.95] * 100 + [0.85] * 100 + [0.75] * 100
    labels = [True] * 95 + [False] * 5 + [True] * 85 + [False] * 15 + [True] * 75 + [False] * 25
    return raws, labels


# ---------------------------------------------------------------------------
# TemperatureScaler.fit
# ---------------------------------------------------------------------------


class TestFit:
    def test_calibrated_data_fits_temperature_near_one(self) -> None:
        """When stated confidence already matches accuracy, T ≈ 1 (no scaling needed).

        The regression guard. Before the labelling fix this exact
        shape of input never reached the fitter — every particle was labelled
        incorrect — and T came back pinned to the bound. A landed-on-bound
        result here means the labels are being dropped again.
        """
        raws, labels = _calibrated_pairs()
        scaler = TemperatureScaler().fit(raws, labels)
        assert scaler.temperature == pytest.approx(1.0, abs=0.05)
        assert scaler.diagnostics is not None
        assert scaler.diagnostics.hit_bound is False
        assert scaler.diagnostics.is_trustworthy is True

    def test_overconfident_data_fits_temperature_above_one(self) -> None:
        """Overconfident predictions (high conf, often wrong) → T > 1 to scale down."""
        raws = [0.95] * 100
        labels = [True] * 70 + [False] * 30  # 70% correct at 95% stated
        scaler = TemperatureScaler().fit(raws, labels)
        assert scaler.temperature > 1.0
        # …and the fitted T actually maps the stated value onto the observed rate.
        assert scaler.calibrate(0.95) == pytest.approx(0.70, abs=0.01)

    def test_underconfident_data_fits_temperature_below_one(self) -> None:
        raws = [0.6] * 100
        labels = [True] * 90 + [False] * 10  # 90% correct at 60% stated
        scaler = TemperatureScaler().fit(raws, labels)
        assert scaler.temperature < 1.0
        assert scaler.calibrate(0.6) == pytest.approx(0.90, abs=0.01)

    def test_fit_marks_fitted(self) -> None:
        scaler = TemperatureScaler()
        assert scaler._fitted is False
        scaler.fit([0.5, 0.6], [True, False])
        assert scaler._fitted is True

    def test_fit_returns_self_for_chaining(self) -> None:
        scaler = TemperatureScaler()
        result = scaler.fit([0.5, 0.6], [True, False])
        assert result is scaler

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            TemperatureScaler().fit([0.5, 0.6], [True])


# ---------------------------------------------------------------------------
# Fit diagnostics — the guard
# ---------------------------------------------------------------------------


class TestFitDiagnostics:
    def test_all_false_labels_are_degenerate_and_hit_the_upper_bound(self) -> None:
        """The live failure: two independent fits both returned exactly T=10.0000.

        Every emitted particle labelled incorrect, so NLL is monotone in T and
        the optimizer walks to T_MAX. Both flags fire; either alone must be
        enough to refuse the fit.
        """
        scaler = TemperatureScaler().fit([0.9, 0.8, 0.95, 0.7], [False] * 4)
        d = scaler.diagnostics
        assert d is not None
        assert d.temperature == pytest.approx(T_MAX, abs=1e-3)
        assert d.degenerate_labels is True
        assert d.hit_bound is True
        assert d.is_trustworthy is False

    def test_all_true_labels_are_degenerate(self) -> None:
        """The mirror case — a suite whose gold set covers everything emitted.

        This one does *not* reach the bound (the NLL flattens out first), which
        is precisely why the degenerate-label check cannot be folded into the
        bound check.
        """
        scaler = TemperatureScaler().fit([0.9, 0.8, 0.95, 0.7], [True] * 4)
        d = scaler.diagnostics
        assert d is not None
        assert d.degenerate_labels is True
        assert d.is_trustworthy is False

    def test_mixed_labels_are_not_degenerate(self) -> None:
        raws, labels = _calibrated_pairs()
        d = TemperatureScaler().fit(raws, labels).diagnostics
        assert d is not None
        assert d.degenerate_labels is False
        assert d.positive_rate == pytest.approx(0.85)

    def test_bound_hit_is_flagged_without_degenerate_labels(self) -> None:
        """Uniform 0.95 confidence at 30% accuracy is unreachable by any T > 0.

        In logit space sigmoid(positive / T) never drops below 0.5, so the
        optimizer runs out of room at T_MAX with a perfectly ordinary,
        non-degenerate label set. A bound-hit is its own diagnosis.
        """
        d = TemperatureScaler().fit([0.95] * 10, [True] * 3 + [False] * 7).diagnostics
        assert d is not None
        assert d.degenerate_labels is False
        assert d.hit_bound is True
        assert d.is_trustworthy is False

    def test_reasons_are_empty_for_a_trustworthy_fit(self) -> None:
        raws, labels = _calibrated_pairs()
        d = TemperatureScaler().fit(raws, labels).diagnostics
        assert d is not None
        assert d.reasons() == []

    def test_reasons_name_both_conditions_when_both_fire(self) -> None:
        # Spread the confidences so only the two label/bound conditions fire —
        # a uniform population would also be predictor-degenerate
        # and report a third reason.
        d = TemperatureScaler().fit([0.9, 0.8, 0.7, 0.6, 0.5], [False] * 5).diagnostics
        assert d is not None
        reasons = d.reasons()
        assert len(reasons) == 2
        assert any("degenerate labels" in r for r in reasons)
        assert any("optimizer bound" in r for r in reasons)

    def test_lower_bound_is_flagged(self) -> None:
        d = FitDiagnostics(
            n=10, n_fitted=10, n_saturated=0, distinct_raw=3, positive_rate=0.5, temperature=T_MIN
        )
        assert d.hit_bound is True
        assert d.is_trustworthy is False

    def test_diagnostics_is_none_before_a_fit(self) -> None:
        assert TemperatureScaler().diagnostics is None


# ---------------------------------------------------------------------------
# Saturation exclusion + the two admissibility conditions
# ---------------------------------------------------------------------------


class TestSaturationExclusion:
    """the fit must not learn from points the apply cannot move."""

    def test_is_saturated_matches_the_apply_fixed_points(self) -> None:
        assert is_saturated(0.0) is True
        assert is_saturated(1.0) is True
        assert is_saturated(0.5) is False
        assert is_saturated(0.99) is False
        # …and the predicate agrees with what `calibrate` actually does.
        scaler = TemperatureScaler(temperature=4.0)
        for v in (0.0, 1.0):
            assert scaler.calibrate(v) == v

    def test_saturated_pairs_are_dropped_from_the_fit(self) -> None:
        raws = [1.0] * 6 + [0.8, 0.6]
        labels = [True] * 6 + [True, False]
        d = TemperatureScaler().fit(raws, labels).diagnostics
        assert d is not None
        assert d.n == 8
        assert d.n_fitted == 2
        assert d.n_saturated == 6
        assert d.distinct_raw == 2

    def test_the_endpoint_asymmetry_no_longer_sets_the_temperature(self) -> None:
        """The defect the fix was written to address, in its measured shape.

        94 of 101 particles state exactly 1.0. Under the old fit those were
        clipped to 1-1e-7, contributed a logit of ~16.1 each, and dominated
        every other term — deriving T=6.29 from points the transform is
        structurally incapable of moving, then applying it to the 7 it barely
        saw. Excluded, the temperature answers only to the movable pairs.
        """
        raws = [1.0] * 94 + [0.90] * 6 + [0.95]
        labels = [True] * 86 + [False] * 8 + [True] * 6 + [True]
        d = TemperatureScaler().fit(raws, labels).diagnostics
        assert d is not None
        assert d.n_saturated == 94
        assert d.n_fitted == 7
        # All seven movable pairs are correct, so the honest verdict is
        # "degenerate", not a confident temperature derived from immovables.
        assert d.degenerate_labels is True
        assert d.is_trustworthy is False

    def test_an_all_saturated_population_refuses_without_optimising(self) -> None:
        d = TemperatureScaler().fit([1.0, 1.0, 0.0], [True, False, False]).diagnostics
        assert d is not None
        assert d.n_fitted == 0
        assert d.predictor_degenerate is True
        assert d.is_trustworthy is False
        assert any("no fittable pairs" in r for r in d.reasons())

    def test_positive_rate_is_computed_over_the_fitted_pairs_only(self) -> None:
        # 4 saturated (all correct) would drag the rate to 0.83 if counted.
        raws = [1.0] * 4 + [0.7, 0.6]
        labels = [True] * 4 + [True, False]
        d = TemperatureScaler().fit(raws, labels).diagnostics
        assert d is not None
        assert d.positive_rate == pytest.approx(0.5)


class TestPredictorDegeneracy:
    """the mirror of degenerate labels, in the other variable."""

    def test_a_single_confidence_level_identifies_no_temperature(self) -> None:
        d = TemperatureScaler().fit([0.9] * 10, [True] * 7 + [False] * 3).diagnostics
        assert d is not None
        assert d.distinct_raw == 1
        assert d.predictor_degenerate is True
        assert d.degenerate_labels is False  # labels are fine; the predictor is not
        assert d.is_trustworthy is False
        assert any("predictor degeneracy" in r for r in d.reasons())

    def test_two_distinct_values_clear_the_check(self) -> None:
        d = TemperatureScaler().fit([0.95] * 6 + [0.97], [True] * 5 + [False, True]).diagnostics
        assert d is not None
        assert d.distinct_raw == 2
        assert d.predictor_degenerate is False

    def test_the_reason_says_no_gold_set_can_fix_it(self) -> None:
        """The operator-facing distinction that matters: suite vs extractor."""
        d = TemperatureScaler().fit([0.9] * 10, [True] * 7 + [False] * 3).diagnostics
        assert d is not None
        reason = next(r for r in d.reasons() if "predictor degeneracy" in r)
        assert "property of the extractor" in reason


class TestNonImprovingFit:
    """the only condition about fit quality rather than shape."""

    def _clean(self) -> FitDiagnostics:
        raws, labels = _calibrated_pairs()
        d = TemperatureScaler().fit(raws, labels).diagnostics
        assert d is not None
        return d

    def test_absent_ece_pair_does_not_trip_the_check(self) -> None:
        d = self._clean()
        assert d.ece_before is None
        assert d.non_improving is False
        assert d.is_trustworthy is True

    def test_worse_ece_is_refused(self) -> None:
        d = self._clean().with_ece(0.0728, 0.0878)
        assert d.non_improving is True
        assert d.is_trustworthy is False
        assert any("non-improving fit" in r for r in d.reasons())

    def test_equal_ece_is_refused(self) -> None:
        """No improvement is not an improvement — the comparison is >=, not >."""
        d = self._clean().with_ece(0.05, 0.05)
        assert d.non_improving is True

    def test_better_ece_passes(self) -> None:
        d = self._clean().with_ece(0.0957, 0.0028)
        assert d.non_improving is False
        assert d.is_trustworthy is True

    def test_with_ece_does_not_mutate_the_original(self) -> None:
        d = self._clean()
        d.with_ece(0.9, 0.1)
        assert d.ece_before is None


# ---------------------------------------------------------------------------
# TemperatureScaler.calibrate
# ---------------------------------------------------------------------------


class TestCalibrate:
    """The logit-space transform, ``sigmoid(logit(p) / T)``."""

    def test_identity_when_temperature_is_one(self) -> None:
        scaler = TemperatureScaler(temperature=1.0)
        assert scaler.calibrate(0.7) == pytest.approx(0.7)

    def test_matches_the_closed_form(self) -> None:
        import math

        scaler = TemperatureScaler(temperature=2.0)
        logit = math.log(0.8 / 0.2)
        assert scaler.calibrate(0.8) == pytest.approx(1 / (1 + math.exp(-logit / 2)))

    def test_pulls_toward_a_half_when_temperature_above_one(self) -> None:
        scaler = TemperatureScaler(temperature=2.0)
        assert 0.5 < scaler.calibrate(0.8) < 0.8
        # …symmetrically from below.
        assert 0.2 < scaler.calibrate(0.2) < 0.5

    def test_pushes_toward_the_ends_when_temperature_below_one(self) -> None:
        scaler = TemperatureScaler(temperature=0.5)
        assert scaler.calibrate(0.8) > 0.8
        assert scaler.calibrate(0.2) < 0.2

    def test_never_saturates_to_one(self) -> None:
        """The defect the retired linear form had: `clamp(0.8 / 0.5)` was exactly 1.0.

        Saturation is not merely inaccurate — it is order-destroying: every
        value above T collapsed onto the same 1.0, so 0.90 and 0.99 became
        indistinguishable to every downstream ranking.
        """
        scaler = TemperatureScaler(temperature=0.5)
        assert scaler.calibrate(0.8) < 1.0
        assert scaler.calibrate(0.9) < 1.0
        assert scaler.calibrate(0.8) < scaler.calibrate(0.9)

    def test_extreme_temperature_squashes_toward_a_half_not_toward_zero(self) -> None:
        """The live incident's blast radius, bounded.

        The retired form at the fitted T=10 divided every confidence by ten —
        a raw 0.9 stored as 0.09, immutably. The logit form's worst
        case at the same T is a squash toward 0.5.
        """
        scaler = TemperatureScaler(temperature=10.0)
        assert scaler.calibrate(0.9) == pytest.approx(0.55, abs=0.02)

    def test_is_order_preserving(self) -> None:
        scaler = TemperatureScaler(temperature=3.7)
        values = [0.01, 0.2, 0.5, 0.8, 0.95, 0.999]
        out = scaler.calibrate_batch(values)
        assert out == sorted(out)
        assert len(set(out)) == len(out)

    def test_zero_and_one_are_exact_fixed_points(self) -> None:
        for t in (0.1, 1.0, 9.9):
            scaler = TemperatureScaler(temperature=t)
            assert scaler.calibrate(0.0) == 0.0
            assert scaler.calibrate(1.0) == 1.0

    def test_clamps_to_zero_for_negative_input(self) -> None:
        scaler = TemperatureScaler(temperature=1.0)
        assert scaler.calibrate(-0.1) == 0.0

    def test_zero_temperature_raises(self) -> None:
        scaler = TemperatureScaler(temperature=0.0)
        with pytest.raises(ValueError, match="positive"):
            scaler.calibrate(0.5)

    def test_negative_temperature_raises(self) -> None:
        scaler = TemperatureScaler(temperature=-1.0)
        with pytest.raises(ValueError, match="positive"):
            scaler.calibrate(0.5)


class TestCalibrateBatch:
    def test_applies_to_each_value(self) -> None:
        scaler = TemperatureScaler(temperature=2.0)
        assert scaler.calibrate_batch([0.4, 0.8, 1.0]) == pytest.approx(
            [scaler.calibrate(0.4), scaler.calibrate(0.8), 1.0]
        )

    def test_empty_input_returns_empty_list(self) -> None:
        assert TemperatureScaler().calibrate_batch([]) == []


# ---------------------------------------------------------------------------
# scaler_for_record — the refusal to apply a pre-ADR-0238 fit
# ---------------------------------------------------------------------------


class TestScalerForRecord:
    @staticmethod
    def _record(transform: str | None) -> ExtractorCalibration:
        return ExtractorCalibration(
            temperature=10.0,
            transform=transform,
            fitted_at=datetime.now(UTC),
            benchmark_suite_id="some-suite",
            sample_size=7,
            calibration_error_before=0.2,
            calibration_error_after=0.09,
            provider_model="anthropic:claude-sonnet-4-6",
        )

    def test_logit_record_yields_a_scaler_at_its_temperature(self) -> None:
        scaler = scaler_for_record(self._record(TRANSFORM_LOGIT))
        assert scaler is not None
        assert scaler.temperature == 10.0

    def test_unlabelled_legacy_record_is_refused(self) -> None:
        """Every pre-ADR-0238 record: fitted on all-False labels, linear transform."""
        assert scaler_for_record(self._record(None)) is None

    def test_unknown_transform_is_refused(self) -> None:
        assert scaler_for_record(self._record("isotonic")) is None

    def test_transform_defaults_to_none_so_old_json_deserialises_as_legacy(self) -> None:
        """Records are stored as JSON blobs, so an old row simply lacks the key."""
        cal = ExtractorCalibration.model_validate(
            {
                "temperature": 10.0,
                "fitted_at": datetime.now(UTC).isoformat(),
                "benchmark_suite_id": "s",
                "sample_size": 7,
                "calibration_error_before": 0.2,
                "calibration_error_after": 0.09,
            }
        )
        assert cal.transform is None
        assert scaler_for_record(cal) is None


# ---------------------------------------------------------------------------
# calibration_error (Expected Calibration Error / ECE)
# ---------------------------------------------------------------------------


class TestCalibrationError:
    def test_perfectly_calibrated_returns_zero(self) -> None:
        # 10 samples at 0.9 confidence, 9 correct → bin accuracy = bin conf
        probs = [0.9] * 10
        labels = [True] * 9 + [False] * 1
        # ECE bins at [0.9, 1.0): accuracy 0.9, conf 0.9 → 0 error.
        assert calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)

    def test_overconfident_returns_positive_error(self) -> None:
        probs = [0.9] * 10
        labels = [True] * 5 + [False] * 5  # only 50% correct at 90% confidence
        ece = calibration_error(probs, labels, n_bins=10)
        assert ece == pytest.approx(0.4)  # |0.9 - 0.5| = 0.4

    def test_empty_input_returns_zero(self) -> None:
        assert calibration_error([], [], n_bins=10) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            calibration_error([0.5, 0.6], [True], n_bins=10)

    def test_skips_empty_bins(self) -> None:
        """A bin with no samples must not poison the average."""
        # All probs in [0.0, 0.1) and [0.9, 1.0); middle bins empty.
        probs = [0.05] * 5 + [0.95] * 5
        labels = [False] * 5 + [True] * 5  # perfectly calibrated
        assert calibration_error(probs, labels, n_bins=10) == pytest.approx(0.05, abs=0.01)


class TestExpectedCalibrationError:
    """The canonical ECE shared with the benchmark harness."""

    def test_confidence_exactly_one_is_counted(self) -> None:
        # The reconciliation bug: the old right-open final bin dropped a 1.0
        # confidence from every bin while still dividing by the full count,
        # biasing ECE low. The canonical closed-right last bin counts it.
        confidences = [1.0] * 4
        correctness = [False] * 4  # 100% confident, 0% correct → ECE = 1.0
        assert expected_calibration_error(confidences, correctness, bins=10) == pytest.approx(1.0)

    def test_calibration_error_adapter_counts_one(self) -> None:
        # The extraction adapter now inherits the fixed behaviour: a 1.0 that
        # used to vanish now contributes its full error.
        assert calibration_error([1.0, 1.0], [False, False], n_bins=10) == pytest.approx(1.0)

    def test_empty_and_nonpositive_bins_return_zero(self) -> None:
        assert expected_calibration_error([], [], bins=10) == 0.0
        assert expected_calibration_error([0.5], [True], bins=0) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            expected_calibration_error([0.5, 0.6], [True], bins=10)

    def test_agrees_with_benchmark_adapter(self) -> None:
        # The benchmark adapter maps Particles → (confidence, matched?) and
        # delegates to the same function, so the two must agree on equivalent
        # input. Guards against the implementations re-diverging.
        from datetime import UTC, datetime

        from particles.benchmark.metrics import compute_calibration_error
        from particles.core.schema import (
            Confidence,
            Particle,
            ProvenanceRef,
            ProvenanceRefType,
            UncertaintyNature,
        )
        from particles.core.scoring.confidence import CalibrationSource

        def _particle(pid: str, conf: float) -> Particle:
            return Particle(
                id=pid,
                content=f"claim {pid}",
                confidence=Confidence(
                    value=conf,
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

        particles = [_particle("a", 0.9), _particle("b", 0.6), _particle("c", 1.0)]
        matched = {"a", "c"}  # b is a miss
        via_benchmark = compute_calibration_error(matched, particles, bins=10)
        via_canonical = expected_calibration_error([0.9, 0.6, 1.0], [True, False, True], bins=10)
        assert via_benchmark == pytest.approx(via_canonical)


# ---------------------------------------------------------------------------
# Suite-set staleness
# ---------------------------------------------------------------------------


class TestFittedSuiteIds:
    def test_single_suite(self) -> None:
        assert fitted_suite_ids("reddit-seed-001") == {"reddit-seed-001"}

    def test_joined_suites_split_on_the_separator(self) -> None:
        """The shape `_extractor_calibrate` actually writes when several suites contribute."""
        assert fitted_suite_ids("hackernews-seed-001+numismatic-seed-001+reddit-seed-001") == {
            "hackernews-seed-001",
            "numismatic-seed-001",
            "reddit-seed-001",
        }

    def test_order_is_not_significant(self) -> None:
        assert fitted_suite_ids("b+a") == fitted_suite_ids("a+b")

    def test_empty_value_yields_empty_set_not_a_blank_id(self) -> None:
        assert fitted_suite_ids("") == set()

    def test_empty_segments_and_padding_are_dropped(self) -> None:
        assert fitted_suite_ids("a+ +b+") == {"a", "b"}


class TestIsSuiteStale:
    def test_same_set_is_clean(self) -> None:
        assert is_suite_stale("reddit-seed-001", {"reddit-seed-001"}) is False

    def test_same_set_in_a_different_order_is_clean(self) -> None:
        """Set comparison, not string comparison — join order must not decide staleness."""
        assert is_suite_stale("b+a", ["a", "b"]) is False

    def test_disjoint_set_is_stale(self) -> None:
        """The live pre-ADR-0230 general-extractor record, verbatim."""
        assert (
            is_suite_stale(
                "hackernews-seed-001+numismatic-seed-001+reddit-seed-001",
                {"prose-article-seed-001"},
            )
            is True
        )

    def test_narrower_fit_is_stale(self) -> None:
        """A `calibrate --suite X` record: accurate to report, and it never blocks."""
        assert is_suite_stale("a", {"a", "b"}) is True

    def test_wider_fit_is_stale(self) -> None:
        assert is_suite_stale("a+b", {"a"}) is True

    def test_any_fit_is_stale_when_the_extractor_now_matches_no_suite(self) -> None:
        """11 of 17 registered extractors auto-match nothing."""
        assert is_suite_stale("a", set()) is True

    def test_empty_fit_against_no_suites_is_clean(self) -> None:
        assert is_suite_stale("", set()) is False
