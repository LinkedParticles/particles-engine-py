"""Pure §15.1 cascade-gating decisions (Extension B).

The policy gate and the per-run cap, extracted from
``particles/operations/cascade.py`` so the L2 vectors can recompute
them. The I/O half (the confirmation-count query, the status writes) stays
covered by ``tests/test_cascade.py``.
"""

from __future__ import annotations

import pytest

from particles.core.cascade_gate import apply_cascade_cap, cascade_gate_passes
from particles.core.schema import PolicyProvenance


@pytest.mark.parametrize(
    "provenance",
    [PolicyProvenance.OPERATOR_DIRECT, PolicyProvenance.REGISTRY_ENDORSED],
)
def test_direct_and_endorsed_always_cascade(provenance: PolicyProvenance) -> None:
    """Neither branch consults the confirmation count."""
    assert cascade_gate_passes(provenance, reviewer_confirmations=0) is True


def test_reviewer_derived_blocked_below_n() -> None:
    assert cascade_gate_passes(PolicyProvenance.REVIEWER_DERIVED, 2, 3) is False


def test_reviewer_derived_passes_at_n() -> None:
    """N >= threshold — the boundary is inclusive."""
    assert cascade_gate_passes(PolicyProvenance.REVIEWER_DERIVED, 3, 3) is True


def test_reviewer_derived_passes_above_n() -> None:
    assert cascade_gate_passes(PolicyProvenance.REVIEWER_DERIVED, 9, 3) is True


def test_reviewer_derived_default_threshold_is_three() -> None:
    """The §15.1 default, so a caller that omits it gets the normative gate."""
    assert cascade_gate_passes(PolicyProvenance.REVIEWER_DERIVED, 2) is False
    assert cascade_gate_passes(PolicyProvenance.REVIEWER_DERIVED, 3) is True


def test_cap_leaves_a_short_batch_untouched() -> None:
    assert apply_cascade_cap(120, 500) == (120, False)


def test_cap_boundary_at_exactly_the_limit_is_not_capped() -> None:
    assert apply_cascade_cap(500, 500) == (500, False)


def test_cap_truncates_and_discloses() -> None:
    """One over the limit truncates; `capped` is the disclosure that the
    remainder was left for manual review rather than silently dropped."""
    assert apply_cascade_cap(501, 500) == (500, True)
    assert apply_cascade_cap(5000, 500) == (500, True)


def test_cap_handles_an_empty_batch() -> None:
    assert apply_cascade_cap(0, 500) == (0, False)
