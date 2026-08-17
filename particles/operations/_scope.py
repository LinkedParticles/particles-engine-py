"""Harvest-scope pair test shared by the audit's two candidate scans.

The contradiction probe and the duplicate scan both admit a
candidate pair when **at least one side** traces to a harvested entry. On a
populated store that admission rule lets coincidental (harvested ↔ unrelated
store particle) cross-pairs out-compete the (harvested ↔ harvested) pairs a
memory audit is actually about whenever a budget binds — the
``audit.max_contradiction_probes`` cap, or an LLM-judge pass cut short by the circuit breaker (owner dogfood 2026-07-11).

``pair_scope_tier`` is the shared fix: it classifies a pair into the two-tier
priority both consumers order their candidates by — intra-scope pairs first,
mixed pairs second, highest similarity within each tier — so a bounded scan
spends its budget on the harvest before the coincidental cross-pairs. With no
scope (store-wide runs) every pair ties at tier 0 and ordering stays pure
similarity, exactly as before.
"""

from __future__ import annotations


def pair_scope_tier(scope: frozenset[str] | None, id_a: str, id_b: str) -> int:
    """Two-tier scope priority of a candidate pair.

    Returns:
        * ``0`` — intra-scope: both sides in ``scope`` (or ``scope is None``,
          the store-wide case where every pair ties and similarity alone
          orders the candidates);
        * ``1`` — mixed: exactly one side in ``scope``;
        * ``2`` — out of scope: neither side in ``scope`` (callers drop these
          pairs entirely — the admission rule).
    """
    if scope is None:
        return 0
    return 2 - (id_a in scope) - (id_b in scope)
