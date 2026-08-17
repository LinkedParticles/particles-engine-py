"""Tests for the usefulness math + lens rule schema (core, pure)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from particles.core.schema import TrustLensUtilityRule
from particles.core.scoring.utility import (
    SweepRow,
    content_dedup_key,
    rank_lift_grid,
    reinforcement_score,
    sweep_rank_lift,
    utility_rank_bonus,
)

_NOW = datetime(2026, 7, 1, tzinfo=UTC)


def _mined(*times: datetime) -> list[tuple[datetime, float]]:
    """Utility events on the mined channel (weight 1.0 each)."""
    return [(t, 1.0) for t in times]


class TestReinforcementScore:
    def test_no_events_is_zero(self) -> None:
        assert reinforcement_score([], 30.0, now=_NOW) == 0.0

    def test_event_now_counts_full(self) -> None:
        assert math.isclose(reinforcement_score(_mined(_NOW), 30.0, now=_NOW), 1.0, abs_tol=1e-9)

    def test_one_half_life_halves(self) -> None:
        old = _NOW - timedelta(days=30)
        assert math.isclose(reinforcement_score(_mined(old), 30.0, now=_NOW), 0.5, abs_tol=1e-9)

    def test_events_accumulate(self) -> None:
        # Two fresh events count ~2; recency-weighted so a stale one adds less.
        fresh_pair = reinforcement_score(_mined(_NOW, _NOW), 30.0, now=_NOW)
        assert math.isclose(fresh_pair, 2.0, abs_tol=1e-9)
        mixed = reinforcement_score(_mined(_NOW, _NOW - timedelta(days=30)), 30.0, now=_NOW)
        assert math.isclose(mixed, 1.5, abs_tol=1e-9)

    def test_future_event_clamped_not_amplified(self) -> None:
        future = _NOW + timedelta(days=10)
        assert math.isclose(reinforcement_score(_mined(future), 30.0, now=_NOW), 1.0, abs_tol=1e-9)

    def test_naive_datetime_treated_as_utc(self) -> None:
        naive = datetime(2026, 7, 1)  # noqa: DTZ001 — deliberately naive for the test
        assert math.isclose(reinforcement_score(_mined(naive), 30.0, now=_NOW), 1.0, abs_tol=1e-6)


class TestChannelWeighting:
    """per-event weights, so the two channels are commensurable."""

    def test_weight_scales_the_event(self) -> None:
        assert math.isclose(reinforcement_score([(_NOW, 25.0)], 30.0, now=_NOW), 25.0, abs_tol=1e-9)

    def test_weight_decays_with_age_like_any_event(self) -> None:
        old = _NOW - timedelta(days=30)
        assert math.isclose(reinforcement_score([(old, 25.0)], 30.0, now=_NOW), 12.5, abs_tol=1e-9)

    def test_channels_sum(self) -> None:
        mixed = reinforcement_score([(_NOW, 1.0), (_NOW, 25.0)], 30.0, now=_NOW)
        assert math.isclose(mixed, 26.0, abs_tol=1e-9)

    def test_non_positive_weight_contributes_nothing(self) -> None:
        assert reinforcement_score([(_NOW, 0.0), (_NOW, -5.0)], 30.0, now=_NOW) == 0.0

    def test_one_gesture_is_commensurable_with_a_well_used_belief(self) -> None:
        """The whole point of the weight.

        At weight 1 a single explicit press cannot compete with a belief the
        miner has credited across ~29 sessions — it buys a fifth of the lift.
        At the calibrated weight it lands in the same range, which is what makes
        the explicit channel able to move a belief at all.
        """
        well_used = utility_rank_bonus(
            reinforcement_score(_mined(*[_NOW] * 29), 30.0, now=_NOW), 0.011
        )
        unweighted_press = utility_rank_bonus(
            reinforcement_score([(_NOW, 1.0)], 30.0, now=_NOW), 0.011
        )
        weighted_press = utility_rank_bonus(
            reinforcement_score([(_NOW, 25.0)], 30.0, now=_NOW), 0.011
        )
        assert unweighted_press < well_used / 4
        assert 0.8 < weighted_press / well_used < 1.0


class TestUtilityRankBonus:
    """The additive rank-lift, λ·ln(1+R), replacing multiplier."""

    def test_zero_reinforcement_is_neutral(self) -> None:
        # Cold-start: no evidence → +0 (never a penalty), preserved.
        assert utility_rank_bonus(0.0, 0.02) == 0.0

    def test_zero_lambda_disables_the_lift(self) -> None:
        assert utility_rank_bonus(100.0, 0.0) == 0.0

    def test_matches_the_closed_form(self) -> None:
        assert math.isclose(utility_rank_bonus(24.0, 0.02), 0.02 * math.log(25.0), abs_tol=1e-12)

    def test_promotion_only(self) -> None:
        # λ ≥ 0 and R ≥ 0 ⇒ bonus ≥ 0: utility can lift a rank, never lower it.
        for r in (0.0, 0.1, 1.0, 10.0, 1000.0):
            assert utility_rank_bonus(r, 0.02) >= 0.0

    def test_monotone_and_non_saturating(self) -> None:
        """The defect this fixes: count magnitude must survive.

        The superseded ``clamp(1 + (cap-1)(1-e^{-weight·R}), floor, cap)`` was
        ~0.95 of the way to the cap by R≈6, so 6×, 24× and 200× use were
        indistinguishable and the head reverted to base-confidence order.
        """
        b6 = utility_rank_bonus(6.0, 0.02)
        b24 = utility_rank_bonus(24.0, 0.02)
        b200 = utility_rank_bonus(200.0, 0.02)
        assert 0.0 < b6 < b24 < b200
        # Still meaningfully separated way past where the old form flat-lined.
        assert b200 - b24 > 1e-3

    def test_growth_is_logarithmically_bounded(self) -> None:
        """Bounded influence without a hard cap: doubling R adds at most λ·ln2.

        ``ln(1+2R) − ln(1+R) = ln((1+2R)/(1+R))``, which rises to λ·ln2 from
        below — so the "no single belief can run away" bound holds
        at every R, tightest for small R.
        """
        ceiling = 0.02 * math.log(2)
        for r in (1.0, 10.0, 100.0, 10_000.0):
            delta = utility_rank_bonus(2 * r, 0.02) - utility_rank_bonus(r, 0.02)
            assert 0.0 < delta <= ceiling + 1e-12
        # …and it does converge to that ceiling for large R.
        far = utility_rank_bonus(2e9, 0.02) - utility_rank_bonus(1e9, 0.02)
        assert math.isclose(far, ceiling, rel_tol=1e-6)

    def test_lambda_scales_the_lift(self) -> None:
        low = utility_rank_bonus(10.0, 0.02)
        high = utility_rank_bonus(10.0, 0.2)
        assert math.isclose(high, 10 * low, rel_tol=1e-9)


class TestAdditiveComposition:
    """`rank_score = effective_confidence + λ·ln(1+R)` — the key."""

    def test_equal_absolute_lift_regardless_of_base_confidence(self) -> None:
        """The core fix: the multiplier gave *less* lift to lower-confidence beliefs.

        Under ``eff × 1.4`` a 0.90 belief gained +0.36 while a 0.66 belief gained
        only +0.26 — structurally disadvantaging exactly the low-confidence,
        high-use guidelines the lens exists to promote (§Context).
        """
        bonus = utility_rank_bonus(24.0, 0.02)
        assert math.isclose((0.66 + bonus) - 0.66, (0.90 + bonus) - 0.90, abs_tol=1e-12)

    def test_high_use_low_confidence_outranks_low_use_high_confidence(self) -> None:
        """The exact defect fixed here, at the calibrated λ = 0.02.

        Under the superseded multiplier a base-0.66 belief capped at 0.66×1.4 =
        0.924 and could never overtake a ~0.95 head at *any* reinforcement.
        """
        lam = 0.02
        load_bearing = 0.66 + utility_rank_bonus(24.0, lam)  # acted on 24×
        trivia = 0.70 + utility_rank_bonus(0.0, lam)  # never acted on
        assert load_bearing > trivia

    def test_rank_score_may_exceed_one(self) -> None:
        """It is an ordering key, not a probability (§Consequences)."""
        assert 0.99 + utility_rank_bonus(500.0, 0.2) > 1.0

    def test_unused_belief_keeps_its_base_position(self) -> None:
        lam = 0.02
        a, b = 0.80, 0.70
        assert (a + utility_rank_bonus(0.0, lam)) > (b + utility_rank_bonus(0.0, lam))


class TestUtilityRuleSchema:
    def test_default_scope_forbids_pattern(self) -> None:
        with pytest.raises(ValidationError):
            TrustLensUtilityRule(
                scope="default", pattern="x", half_life_uses_days=30, rank_lift=0.02
            )

    def test_source_type_scope_requires_pattern(self) -> None:
        with pytest.raises(ValidationError):
            TrustLensUtilityRule(scope="source_type", half_life_uses_days=30, rank_lift=0.02)

    def test_negative_rank_lift_rejected(self) -> None:
        # λ ≥ 0 is what makes the lens promotion-only.
        with pytest.raises(ValidationError):
            TrustLensUtilityRule(half_life_uses_days=30, rank_lift=-0.01)

    def test_old_multiplier_vocabulary_no_longer_validates(self) -> None:
        """config-schema change: a weight/floor/cap rule is not a valid rule.

        ``rank_lift`` is required, so a lens definition still written in the
        superseded vocabulary fails validation outright rather than silently
        resolving to some default λ — the loud half of the compat story (the
        quiet half is `lens_store` skipping already-materialised rows).
        """
        with pytest.raises(ValidationError):
            TrustLensUtilityRule(  # type: ignore[call-arg]
                half_life_uses_days=30, weight=0.5, floor=1.0, cap=1.4
            )

    def test_retired_keys_are_not_model_fields(self) -> None:
        assert set(TrustLensUtilityRule.model_fields) == {
            "scope",
            "pattern",
            "half_life_uses_days",
            "rank_lift",
        }

    def test_valid_default_rule(self) -> None:
        rule = TrustLensUtilityRule(half_life_uses_days=30, rank_lift=0.02)
        assert rule.scope == "default"
        assert rule.rank_lift == 0.02
        assert rule.pattern is None


# ---------------------------------------------------------------------------
# the calibration sweep (pure)
# ---------------------------------------------------------------------------


def _row(pid: str, eff: float, r: float, content: str | None = None) -> SweepRow:
    return SweepRow(
        particle_id=pid,
        effective_confidence=eff,
        reinforcement=r,
        content_key=content_dedup_key(content if content is not None else pid),
    )


#: A miniature of the real store's shape: a confidence plateau (``a``/``d``/``e``
#: with no utility evidence), one low-confidence high-use target (``b`` — the
#: `git commit -s` analogue), and one heavily-reinforced duplicate of ``a``
#: (``c`` — the over-extraction analogue that sets the ceiling).
_BAND_FIXTURE = [
    _row("a", 0.70, 0.0, "alpha"),
    _row("d", 0.65, 0.0, "gamma"),
    _row("e", 0.60, 0.0, "delta"),
    _row("b", 0.30, 24.0, "beta"),
    _row("c", 0.20, 30.0, "alpha"),
]


class TestContentDedupKey:
    def test_punctuation_and_case_collapse(self) -> None:
        assert content_dedup_key("The `git mv` trap!") == content_dedup_key("the git mv trap")

    def test_whitespace_collapses(self) -> None:
        assert content_dedup_key("a   b\n\tc") == "a b c"

    def test_distinct_claims_stay_distinct(self) -> None:
        assert content_dedup_key("uv parses pyproject") != content_dedup_key("uv writes uv.lock")

    def test_empty_content(self) -> None:
        assert content_dedup_key("!!!") == ""


class TestRankLiftGrid:
    def test_always_includes_the_zero_baseline(self) -> None:
        # λ=0 is the pre-utility head the whole sweep is read against.
        assert rank_lift_grid(0.12, 120)[0] == 0.0

    def test_step_count_and_endpoint(self) -> None:
        grid = rank_lift_grid(0.1, 10)
        assert len(grid) == 11
        assert grid[-1] == 0.1
        assert grid[1] == 0.01

    def test_degenerate_inputs_return_baseline_only(self) -> None:
        assert rank_lift_grid(0.0, 10) == (0.0,)
        assert rank_lift_grid(0.1, 0) == (0.0,)


class TestSweepRankLift:
    def test_no_rows_yields_empty_bands(self) -> None:
        sweep = sweep_rank_lift([], grid=(0.0, 0.01), head_sizes=[10])
        assert sweep.points == ()
        assert sweep.intersection.empty
        assert sweep.bands[0][1].empty

    def test_ranking_matches_the_adr_0204_key(self) -> None:
        """rank_score = eff + λ·ln(1+R), with the digest's (-score, id) tiebreak."""
        rows = [_row("a", 0.70, 0.0), _row("b", 0.60, 24.0)]
        # λ=0: confidence order (a first). The lift is +0 for the unreinforced.
        base = sweep_rank_lift(rows, grid=(0.0,), head_sizes=[1], target_ids=["b"])
        assert dict(base.points[0].heads[0].target_ranks)["b"] == 2
        # λ large enough that 0.02·ln(25) ≈ 0.064 > the 0.10 gap? No — pick one that is.
        lifted = sweep_rank_lift(rows, grid=(0.05,), head_sizes=[1], target_ids=["b"])
        assert dict(lifted.points[0].heads[0].target_ranks)["b"] == 1

    def test_ties_break_by_id_so_the_order_is_deterministic(self) -> None:
        rows = [_row("z", 0.7, 0.0), _row("a", 0.7, 0.0)]
        sweep = sweep_rank_lift(rows, grid=(0.0,), head_sizes=[2], target_ids=["a", "z"])
        ranks = dict(sweep.points[0].heads[0].target_ranks)
        assert (ranks["a"], ranks["z"]) == (1, 2)

    def test_absent_target_reported_as_rank_zero_and_never_admissible(self) -> None:
        sweep = sweep_rank_lift(
            [_row("a", 0.7, 0.0)], grid=(0.0,), head_sizes=[1], target_ids=["ghost"]
        )
        assert dict(sweep.points[0].heads[0].target_ranks)["ghost"] == 0
        assert not sweep.points[0].heads[0].targets_in_head

    def test_no_targets_leaves_only_the_diversity_criterion(self) -> None:
        rows = [_row("a", 0.7, 0.0), _row("b", 0.6, 0.0)]
        head = sweep_rank_lift(rows, grid=(0.0,), head_sizes=[2]).points[0].heads[0]
        assert head.targets_in_head  # vacuously
        assert head.admissible

    def test_duplicate_cluster_is_counted(self) -> None:
        rows = [
            _row("a", 0.70, 0.0, "same claim"),
            _row("b", 0.69, 0.0, "same claim!"),
            _row("c", 0.68, 0.0, "other claim"),
        ]
        head = sweep_rank_lift(rows, grid=(0.0,), head_sizes=[3]).points[0].heads[0]
        assert head.distinct_contents == 2
        assert head.largest_duplicate_cluster == 2

    def test_distinct_ratio_sets_the_diversity_bar(self) -> None:
        rows = [
            _row("a", 0.70, 0.0, "same claim"),
            _row("b", 0.69, 0.0, "same claim"),
            _row("c", 0.68, 0.0, "other"),
            _row("d", 0.67, 0.0, "another"),
        ]
        strict = sweep_rank_lift(rows, grid=(0.0,), head_sizes=[4], distinct_ratio=1.0)
        assert not strict.points[0].heads[0].diverse  # needs 4 distinct, has 3
        loose = sweep_rank_lift(rows, grid=(0.0,), head_sizes=[4], distinct_ratio=0.75)
        assert loose.points[0].heads[0].diverse  # needs 3, has 3

    def test_head_larger_than_population_scales_the_requirement(self) -> None:
        # required_distinct is computed against the *available* rows, so a small
        # store is not failed for having fewer beliefs than the surface renders.
        rows = [_row("a", 0.7, 0.0), _row("b", 0.6, 0.0)]
        head = sweep_rank_lift(rows, grid=(0.0,), head_sizes=[60]).points[0].heads[0]
        assert head.required_distinct == 2
        assert head.admissible

    def test_band_has_both_edges_and_they_are_real(self) -> None:
        """The two-sided shape measured, in miniature.

        Below the band the target ``b`` cannot clear the confidence gap to
        ``e``; above it, the heavily-reinforced duplicate ``c`` overtakes ``d``
        and lands in the head next to its twin ``a``, breaking diversity. So the
        band is a genuine interval — the lower edge set by the target, the upper
        edge set by over-extraction, exactly as on the real store.
        """
        rows = _BAND_FIXTURE
        sweep = sweep_rank_lift(
            rows,
            grid=rank_lift_grid(0.5, 50),
            head_sizes=[3],
            target_ids=["b"],
            distinct_ratio=1.0,
        )
        band = sweep.bands[0][1]
        assert (band.low, band.high) == (0.1, 0.13)
        assert band.contiguous
        by_lift = {p.rank_lift: p.heads[0] for p in sweep.points}
        assert not by_lift[0.0].admissible  # baseline: target outside the head
        assert not by_lift[0.09].targets_in_head  # just below the lower edge
        assert by_lift[0.1].admissible  # the lower edge itself
        assert by_lift[0.13].admissible  # the upper edge itself
        assert not by_lift[0.14].diverse  # just above: the duplicate is in

    def test_intersection_empty_when_a_larger_surface_cannot_be_satisfied(self) -> None:
        """the band belongs to the surface, not the store.

        The population holds one duplicate pair, so a strict all-distinct
        criterion is unsatisfiable at ``N = 5`` for *any* ``λ`` — the head
        contains every row. The ``N = 3`` band is unaffected, and the
        intersection is empty: no single ``λ`` serves both surfaces.
        """
        sweep = sweep_rank_lift(
            _BAND_FIXTURE,
            grid=rank_lift_grid(0.5, 50),
            head_sizes=[3, 5],
            target_ids=["b"],
            distinct_ratio=1.0,
        )
        bands = dict(sweep.bands)
        assert not bands[3].empty
        assert bands[5].empty
        assert sweep.intersection.empty

    def test_relaxing_the_ratio_recovers_an_intersection(self) -> None:
        # The same store at ≥80% distinct: N=5 becomes satisfiable (4 of 5), and
        # the surfaces overlap again. This is the knob leaned on.
        sweep = sweep_rank_lift(
            _BAND_FIXTURE,
            grid=rank_lift_grid(0.5, 50),
            head_sizes=[3, 5],
            target_ids=["b"],
            distinct_ratio=0.8,
        )
        assert not dict(sweep.bands)[5].empty
        assert not sweep.intersection.empty

    def test_configured_value_flag(self) -> None:
        rows = [_row("a", 0.70, 0.0, "alpha"), _row("b", 0.60, 24.0, "beta")]
        sweep = sweep_rank_lift(
            rows,
            grid=rank_lift_grid(0.2, 20),
            head_sizes=[1],
            target_ids=["b"],
            configured_rank_lift=0.0,
        )
        # λ=0 can't lift 'b' into a top-1, so the configured value is out of band.
        assert not sweep.configured_admissible

    def test_scored_count_is_reported(self) -> None:
        rows = [_row(f"p{i}", 0.5, 0.0) for i in range(7)]
        assert sweep_rank_lift(rows, grid=(0.0,), head_sizes=[3]).scored == 7

    def test_zero_lift_head_is_the_pre_utility_order(self) -> None:
        """cold-start: at λ=0 reinforcement cannot move anything."""
        rows = [_row("a", 0.70, 0.0), _row("b", 0.69, 500.0)]
        sweep = sweep_rank_lift(rows, grid=(0.0,), head_sizes=[2], target_ids=["a", "b"])
        assert dict(sweep.points[0].heads[0].target_ranks) == {"a": 1, "b": 2}
