"""Tests for ``particles/operations/_scope.py`` — the shared harvest-scope tier.

``pair_scope_tier`` is the two-tier priority both budgeted candidate
scans order by: the contradiction probe and the duplicate-judge pass consume intra-scope pairs (both sides harvested) before
mixed pairs (one side), highest similarity within each tier. Pure function —
no store, no LLM.
"""

from __future__ import annotations

from particles.operations._scope import pair_scope_tier


class TestPairScopeTier:
    def test_no_scope_everything_ties_at_tier_zero(self) -> None:
        """Store-wide runs (scope None) have no tiers: ordering stays pure similarity."""
        assert pair_scope_tier(None, "a", "b") == 0

    def test_both_sides_in_scope_is_tier_zero(self) -> None:
        assert pair_scope_tier(frozenset({"a", "b"}), "a", "b") == 0

    def test_one_side_in_scope_is_tier_one_symmetrically(self) -> None:
        scope = frozenset({"a"})
        assert pair_scope_tier(scope, "a", "b") == 1
        assert pair_scope_tier(scope, "b", "a") == 1

    def test_neither_side_in_scope_is_tier_two(self) -> None:
        """Tier 2 pairs fail the admission rule; callers drop them."""
        assert pair_scope_tier(frozenset({"x"}), "a", "b") == 2

    def test_empty_scope_keeps_no_pairs(self) -> None:
        assert pair_scope_tier(frozenset(), "a", "b") == 2

    def test_tier_orders_before_similarity(self) -> None:
        """Sorting by (tier, -similarity) puts a low-similarity intra-scope pair
        ahead of a higher-similarity mixed pair — the starvation fix."""
        scope = frozenset({"h1", "h2", "h3"})
        pairs = [("h1", "s1", 0.99), ("h1", "h2", 0.90), ("h3", "s2", 0.95)]
        ordered = sorted(pairs, key=lambda p: (pair_scope_tier(scope, p[0], p[1]), -p[2]))
        # Intra tier first (despite lowest similarity), then the mixed tier in
        # its own highest-similarity-first order.
        assert ordered == [("h1", "h2", 0.90), ("h1", "s1", 0.99), ("h3", "s2", 0.95)]
