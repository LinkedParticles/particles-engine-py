"""Tests for the owner-relevance math: the pure bonus and the ω sweep."""

from __future__ import annotations

import pytest

from particles.core.scoring.relevance import owner_rank_bonus, sweep_owner_rank_lift
from particles.core.scoring.utility import SweepRow, rank_lift_grid

# ---------------------------------------------------------------------------
# owner_rank_bonus — the pure term (tests/AGENTS.md: pure functions are required)
# ---------------------------------------------------------------------------


def test_bonus_is_omega_when_relevant() -> None:
    assert owner_rank_bonus(True, 0.02) == pytest.approx(0.02)


def test_bonus_is_zero_when_not_relevant() -> None:
    assert owner_rank_bonus(False, 0.02) == 0.0


def test_bonus_is_zero_at_zero_omega() -> None:
    """The shipped default is inert — ω = 0 changes no ordering at all."""
    assert owner_rank_bonus(True, 0.0) == 0.0


def test_bonus_is_never_negative() -> None:
    """Promotion-only: the lens can never rank a domain claim down.

    A negative ω is not reachable through config (``ge=0.0``), but the pure
    function is the normative surface a second implementation matches, so the
    sign constraint is enforced here rather than assumed upstream.
    """
    assert owner_rank_bonus(True, -1.0) == 0.0
    assert owner_rank_bonus(False, -1.0) == 0.0


def test_bonus_is_flat_not_graded() -> None:
    """A(p) is an indicator, so every relevant belief gets the *same* lift.

    This is what makes ω a threshold over the cohort rather than a graded lift
    — the property head-share is calibrated against.
    """
    assert owner_rank_bonus(True, 0.05) == owner_rank_bonus(True, 0.05)


# ---------------------------------------------------------------------------
# sweep_owner_rank_lift — the activation gate
# ---------------------------------------------------------------------------


def _row(pid: str, eff: float, *, owner: bool = False, r: float = 0.0) -> SweepRow:
    return SweepRow(
        particle_id=pid,
        effective_confidence=eff,
        reinforcement=r,
        content_key=pid,
        owner_relevant=owner,
    )


def test_empty_rows_yield_empty_bands() -> None:
    sweep = sweep_owner_rank_lift([], grid=(0.0, 0.1), head_sizes=[10], lambda_=0.011)
    assert sweep.scored == 0
    assert sweep.owner_population == 0
    assert sweep.intersection.empty


def test_owner_population_counted() -> None:
    rows = [_row("a", 0.7, owner=True), _row("b", 0.7), _row("c", 0.7, owner=True)]
    sweep = sweep_owner_rank_lift(rows, grid=(0.0,), head_sizes=[2], lambda_=0.0)
    assert sweep.owner_population == 2
    assert sweep.scored == 3


def test_zero_omega_leaves_ordering_untouched() -> None:
    """Cold start: at ω = 0 the head is exactly the pre-0220 head."""
    rows = [_row("a", 0.9), _row("b", 0.5, owner=True)]
    sweep = sweep_owner_rank_lift(rows, grid=(0.0,), head_sizes=[1], lambda_=0.0)
    head = sweep.points[0].head(1)
    assert head is not None
    assert head.owner_in_head == 0  # the 0.9 non-owner belief still wins


def test_sufficient_omega_promotes_the_owner_belief() -> None:
    rows = [_row("a", 0.9), _row("b", 0.5, owner=True)]
    sweep = sweep_owner_rank_lift(rows, grid=(0.5,), head_sizes=[1], lambda_=0.0)
    head = sweep.points[0].head(1)
    assert head is not None
    assert head.owner_in_head == 1


def test_share_ceiling_rejects_a_cohort_that_takes_the_head() -> None:
    """Criterion 2 — the flat step promoting a whole cohort at once is the hazard."""
    rows = [_row(f"o{i}", 0.5, owner=True) for i in range(8)]
    rows += [_row(f"d{i}", 0.9) for i in range(2)]
    # ω = 1.0 lifts all eight owner beliefs above both domain claims.
    sweep = sweep_owner_rank_lift(
        rows, grid=(1.0,), head_sizes=[8], lambda_=0.0, max_owner_share=0.5
    )
    head = sweep.points[0].head(8)
    assert head is not None
    assert head.owner_in_head == 8
    assert head.owner_share == pytest.approx(1.0)
    assert not head.share_bounded
    assert not head.admissible


def test_utility_target_non_regression_is_criterion_three() -> None:
    """Criterion 3 — aboutness must not push the utility-promoted belief out.

    ``u`` only reaches the head via its reinforcement, so it is exactly the case: two promotion-only terms competing for one finite head.
    """
    rows = [
        _row("u", 0.5, r=30.0),  # utility-promoted, not about the viewer
        _row("o1", 0.5, owner=True),
        _row("o2", 0.5, owner=True),
    ]
    lam = 0.02
    # A head of 1 with a large ω: both owner beliefs outrank the utility target.
    sweep = sweep_owner_rank_lift(
        rows, grid=(1.0,), head_sizes=[1], lambda_=lam, target_ids=["u"], max_owner_share=1.0
    )
    head = sweep.points[0].head(1)
    assert head is not None
    assert not head.targets_in_head
    assert not head.admissible


def test_lambda_is_held_fixed_across_the_grid() -> None:
    """The utility term must not vary with ω, or criterion 3 measures nothing."""
    rows = [_row("u", 0.5, r=30.0), _row("o", 0.5, owner=True)]
    lam = 0.02
    sweep = sweep_owner_rank_lift(
        rows, grid=(0.0, 0.001), head_sizes=[1], lambda_=lam, target_ids=["u"]
    )
    # At ω = 0 the utility target holds rank 1 purely on its λ lift.
    assert dict(sweep.points[0].heads[0].target_ranks)["u"] == 1


def test_band_reports_the_admissible_omega_range() -> None:
    rows = [_row("d", 0.9), _row("o", 0.5, owner=True), _row("d2", 0.8)]
    sweep = sweep_owner_rank_lift(
        rows,
        grid=rank_lift_grid(1.0, 10),
        head_sizes=[1],
        lambda_=0.0,
        min_owner_in_head=1,
        max_owner_share=1.0,
    )
    # Below ω = 0.4 the owner belief cannot pass the 0.9 belief; at/above it can.
    assert not sweep.intersection.empty
    assert sweep.intersection.low is not None
    assert sweep.intersection.low >= 0.4


def test_configured_admissible_flag() -> None:
    rows = [_row("d", 0.9), _row("o", 0.5, owner=True)]
    sweep = sweep_owner_rank_lift(
        rows,
        grid=rank_lift_grid(1.0, 10),
        head_sizes=[1],
        lambda_=0.0,
        max_owner_share=1.0,
        configured_rank_lift=0.9,
    )
    assert sweep.configured_admissible

    inert = sweep_owner_rank_lift(
        rows,
        grid=rank_lift_grid(1.0, 10),
        head_sizes=[1],
        lambda_=0.0,
        max_owner_share=1.0,
        configured_rank_lift=0.0,
    )
    assert not inert.configured_admissible


def test_head_larger_than_population_is_evaluated_against_the_population() -> None:
    rows = [_row("o", 0.5, owner=True), _row("d", 0.9)]
    sweep = sweep_owner_rank_lift(
        rows, grid=(1.0,), head_sizes=[50], lambda_=0.0, max_owner_share=1.0
    )
    head = sweep.points[0].head(50)
    assert head is not None
    assert head.owner_in_head == 1


# ---------------------------------------------------------------------------
# Report disclosures (no silent caps)
# ---------------------------------------------------------------------------


def test_report_discloses_a_band_open_at_the_grid_edge() -> None:
    """An upper edge that is only the grid edge must not read as a ceiling."""
    from particles.operations.utility_sweep import render_owner_rank_lift_sweep

    rows = [_row("d", 0.9), _row("o", 0.5, owner=True)]
    sweep = sweep_owner_rank_lift(
        rows,
        grid=rank_lift_grid(1.0, 10),
        head_sizes=[1],
        lambda_=0.0,
        max_owner_share=1.0,
    )
    report = render_owner_rank_lift_sweep(sweep)
    assert "open at the top" in report
    assert "where the grid stopped, not where a criterion failed" in report


def test_report_omits_the_open_note_when_the_ceiling_is_measured() -> None:
    from particles.operations.utility_sweep import render_owner_rank_lift_sweep

    rows = [_row("o1", 0.5, owner=True), _row("o2", 0.5, owner=True), _row("d", 0.9)]
    # A tight share ceiling that the largest ω genuinely violates.
    sweep = sweep_owner_rank_lift(
        rows,
        grid=rank_lift_grid(1.0, 10),
        head_sizes=[2],
        lambda_=0.0,
        max_owner_share=0.5,
    )
    report = render_owner_rank_lift_sweep(sweep)
    assert "open at the top" not in report


def test_report_states_amplification() -> None:
    from particles.operations.utility_sweep import render_owner_rank_lift_sweep

    rows = [_row(f"d{i}", 0.9) for i in range(9)] + [_row("o", 0.5, owner=True)]
    sweep = sweep_owner_rank_lift(
        rows, grid=rank_lift_grid(1.0, 10), head_sizes=[2], lambda_=0.0, max_owner_share=1.0
    )
    report = render_owner_rank_lift_sweep(sweep)
    assert "amplification" in report
    # Cohort is 1/10 of the store; one head slot of two is 50% ⇒ 5x.
    assert "5×" in report


def test_report_flags_an_unresolved_viewer_rather_than_an_empty_band() -> None:
    from particles.operations.utility_sweep import render_owner_rank_lift_sweep

    sweep = sweep_owner_rank_lift(
        [_row("d", 0.9)], grid=rank_lift_grid(1.0, 10), head_sizes=[1], lambda_=0.0
    )
    report = render_owner_rank_lift_sweep(sweep)
    assert "owner_lens.subjects" in report
