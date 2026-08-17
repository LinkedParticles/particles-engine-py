"""Tests for particles/core/granularity.py — the claim-granularity soft-gate predicate."""

from __future__ import annotations

from particles.core.granularity import count_sentences, granularity_violation


def test_count_sentences() -> None:
    assert count_sentences("One atomic claim.") == 1
    assert count_sentences("One. Two. Three.") == 3
    assert count_sentences("No terminator") == 1
    assert count_sentences("Q? A! B.") == 3


def test_under_thresholds_passes() -> None:
    assert granularity_violation("Short atomic claim.", max_chars=320, max_sentences=3) is None


def test_char_limit_breached() -> None:
    reason = granularity_violation("x" * 50, max_chars=40, max_sentences=0)
    assert reason is not None
    assert "chars" in reason


def test_sentence_limit_breached() -> None:
    reason = granularity_violation("A. B. C. D.", max_chars=0, max_sentences=3)
    assert reason is not None
    assert "sentences" in reason


def test_zero_threshold_disables_each_check() -> None:
    assert granularity_violation("x" * 9999, max_chars=0, max_sentences=0) is None
