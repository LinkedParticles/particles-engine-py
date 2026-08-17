"""Tests for the noisy-OR confidence merge (§6.9)."""

from __future__ import annotations

import itertools
import math

from particles.core.scoring.confidence import (
    compute_effective_confidence,
    merge_co_evidential_confidence,
)


def test_empty_group_returns_zero() -> None:
    assert merge_co_evidential_confidence([]) == 0.0


def test_singleton_returns_input_unchanged() -> None:
    assert merge_co_evidential_confidence([(0.87, "src-a")]) == 0.87


def test_two_independent_sources_increase_confidence() -> None:
    """Two distinct-source corroborations exceed either alone (noisy-OR property)."""
    merged = merge_co_evidential_confidence([(0.7, "src-a"), (0.7, "src-b")])
    # 1 - (1 - 0.7)(1 - 0.7) = 1 - 0.09 = 0.91
    assert math.isclose(merged, 0.91, abs_tol=1e-6)
    assert merged > 0.7


def test_weak_corroboration_lifts_strong_evidence() -> None:
    """Adding a low-confidence corroborator from a new source should lift confidence."""
    base = merge_co_evidential_confidence([(0.95, "src-a")])
    lifted = merge_co_evidential_confidence([(0.95, "src-a"), (0.40, "src-b")])
    assert lifted > base
    # 1 - (1 - 0.95)(1 - 0.40) = 1 - 0.05 * 0.60 = 1 - 0.03 = 0.97
    assert math.isclose(lifted, 0.97, abs_tol=1e-6)


def test_same_source_repeats_are_throttled() -> None:
    """Second/third particles from the same source contribute less, by 1/k decay."""
    # Two particles, both from src-a:
    # first contribution: 0.7 × 1.0 = 0.7
    # second contribution: 0.7 × 0.5 = 0.35
    # 1 - (1 - 0.7)(1 - 0.35) = 1 - 0.3 × 0.65 = 1 - 0.195 = 0.805
    merged = merge_co_evidential_confidence([(0.7, "src-a"), (0.7, "src-a")])
    assert math.isclose(merged, 0.805, abs_tol=1e-6)


def test_spam_attack_is_neutered() -> None:
    """Many particles from the same source cannot saturate confidence."""
    # 10 particles all from src-a, each 0.5 effective confidence.
    # Weights: 0.5/1, 0.5/2, 0.5/3, ..., 0.5/10
    entries = [(0.5, "src-a")] * 10
    merged = merge_co_evidential_confidence(entries)

    # Compare to genuine consensus: 10 particles each 0.5 from distinct sources.
    distinct = [(0.5, f"src-{i}") for i in range(10)]
    merged_distinct = merge_co_evidential_confidence(distinct)

    # Spam version is well below distinct version.
    assert merged < merged_distinct
    # And below the asymptote 1.0 — does not reach near-certainty.
    assert merged < 0.85


def test_distinct_sources_can_approach_one() -> None:
    """Many independent moderate sources push merged confidence near 1.0."""
    entries = [(0.5, f"src-{i}") for i in range(20)]
    merged = merge_co_evidential_confidence(entries)
    # 1 - (1 - 0.5)^20 ≈ 0.999999
    assert merged > 0.999


def test_clamps_above_one_and_below_zero() -> None:
    """Out-of-range inputs are clamped before merge."""
    # Should clamp 1.5 to 1.0; one input at 1.0 makes the product term 0 and result 1.0
    assert merge_co_evidential_confidence([(1.5, "x")]) == 1.0
    assert merge_co_evidential_confidence([(-0.3, "x")]) == 0.0


def test_three_way_distinct_sources_matches_formula() -> None:
    merged = merge_co_evidential_confidence([(0.6, "a"), (0.8, "b"), (0.5, "c")])
    # 1 - (1 - 0.6)(1 - 0.8)(1 - 0.5) = 1 - 0.4 * 0.2 * 0.5 = 1 - 0.04 = 0.96
    assert math.isclose(merged, 0.96, abs_tol=1e-6)


def test_merge_is_insertion_order_independent() -> None:
    """The merged value is identical for every permutation of the same group.

    Regression for P4-5 (2026-06-11 review): the query path builds the entry
    list by iterating a ``frozenset``, whose order varies across processes.
    With ≥2 same-source members of unequal confidence the 1/k discount used
    to land on a process-dependent member.
    """
    entries = [(0.8, "src-a"), (0.4, "src-a"), (0.6, "src-b"), (0.3, "src-b"), (0.5, "src-c")]
    merged_values = {
        merge_co_evidential_confidence(list(perm)) for perm in itertools.permutations(entries)
    }
    assert len(merged_values) == 1


def test_same_source_strongest_member_carries_full_weight() -> None:
    """Pins the descending-confidence merge order within a source.

    The 1/k throttle exists to stop repetition from saturating the merge, not
    to penalize a source's best evidence — so the strongest same-source member
    contributes at full weight and the weaker repeat absorbs the discount:
      1 - (1 - 0.8 × 1.0)(1 - 0.4 × 0.5) = 1 - 0.2 × 0.8 = 0.84
    (ascending order would give 1 - 0.6 × 0.6 = 0.64), in either input order.
    """
    for entries in ([(0.8, "src-a"), (0.4, "src-a")], [(0.4, "src-a"), (0.8, "src-a")]):
        assert math.isclose(merge_co_evidential_confidence(entries), 0.84, abs_tol=1e-9)


def test_mixed_independence_and_corroboration() -> None:
    """Two particles from src-a + one from src-b: only the first src-a entry uses full weight."""
    # src-a #1: 0.8 × 1.0 = 0.8
    # src-a #2: 0.8 × 0.5 = 0.4
    # src-b #1: 0.6 × 1.0 = 0.6
    # 1 - (1 - 0.8)(1 - 0.4)(1 - 0.6) = 1 - 0.2 * 0.6 * 0.4 = 1 - 0.048 = 0.952
    merged = merge_co_evidential_confidence([(0.8, "src-a"), (0.8, "src-a"), (0.6, "src-b")])
    assert math.isclose(merged, 0.952, abs_tol=1e-6)


# ---------------------------------------------------------------------------
#: validate the merge's interaction with content-age decay
# and the source-trust cascade. The deferred
# question left open was whether the merge composes correctly once its
# inputs are the *fully-modulated* effective confidences. It does: the query
# path (`_gather_scored` / `score_effective_confidence`) applies
# value × extractor_trust × source_trust × recency **per particle** to produce
# each ``effective_confidence``, and `merge_co_evidential_confidence` then runs
# the §6.9 noisy-OR over those — decay and trust are folded *before* the merge,
# never after. These tests pin that composition using the real production
# functions, so a future refactor cannot reorder the layering unnoticed.
# ---------------------------------------------------------------------------


def test_decay_and_trust_compose_into_merge() -> None:
    """Per-particle decay + trust fold into eff_conf, then §6.9 merges those.

    Mirrors the query path: each member is scored with the real
    ``compute_effective_confidence`` (the same call `_gather_scored` makes),
    and the group is merged with the real noisy-OR. The merged value must equal
    the §6.9 closed form over the *modulated* confidences, not the raw ones.
    """
    # Member A — fresh, trusted extractor, neutral source trust:
    eff_a = compute_effective_confidence(
        0.8, extractor_trust_weight=0.9, source_trust_rank=1.0, recency_factor=1.0
    )
    # Member B — same raw confidence, but a distrusted source AND aged content:
    eff_b = compute_effective_confidence(
        0.8, extractor_trust_weight=0.9, source_trust_rank=0.5, recency_factor=0.5
    )
    assert math.isclose(eff_a, 0.72, abs_tol=1e-9)
    assert math.isclose(eff_b, 0.18, abs_tol=1e-9)

    merged = merge_co_evidential_confidence([(eff_a, "ce-a"), (eff_b, "ce-b")])
    # §6.9: 1 - (1 - 0.72)(1 - 0.18) = 1 - 0.28 × 0.82 = 0.7704
    assert math.isclose(merged, 0.7704, abs_tol=1e-9)
    # The merge of the modulated inputs sits below the merge of the raw ones —
    # decay/distrust on B genuinely lowered the group's rendered confidence.
    raw_merged = merge_co_evidential_confidence([(0.8, "ce-a"), (0.8, "ce-b")])
    assert merged < raw_merged


def test_decay_reduces_a_members_contribution_to_the_merge() -> None:
    """An aged (more-decayed) member lifts the merged confidence strictly less.

    The "interaction with content age decay" case: holding the other
    member fixed, a shorter recency_factor on one member must produce a lower
    merged confidence — corroboration from stale content counts for less.
    """
    fresh = compute_effective_confidence(0.7, recency_factor=1.0)
    decayed = compute_effective_confidence(0.7, recency_factor=0.3)
    other = compute_effective_confidence(0.6, recency_factor=1.0)

    merged_fresh = merge_co_evidential_confidence([(fresh, "ce-a"), (other, "ce-b")])
    merged_decayed = merge_co_evidential_confidence([(decayed, "ce-a"), (other, "ce-b")])
    assert merged_decayed < merged_fresh
    # And a fully-decayed-to-floor member still corroborates (noisy-OR never
    # *lowers* below the surviving member alone): merged ≥ the other member.
    assert merged_decayed >= other
