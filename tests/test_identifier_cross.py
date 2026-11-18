# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``particles/core/identifier_cross.py`` and its rung.

The reading is pinned case by case: identifiers that cross, formatting that
never does, a one-sided restatement kept, and the floor applied after
normalising. The rung built on it answers ``DIFFERENT`` or ``ABSTAIN`` only.
"""

from __future__ import annotations

import pytest

from particles.core.identifier_cross import (
    MIN_LENGTH,
    Relation,
    crossed,
    identifier_set,
    pair_relation,
)
from particles.core.ladders import IDENTIFIER_CROSS
from particles.core.rung_ladder import CostClass, DuplicateVerdict


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "The version in pyproject.toml was changed to 1.114.1.",
            "The version in pyproject.toml was changed to 1.114.2.",
        ),
        ("PR #284 was opened.", "PR #288 was opened."),
        ("Ticket T-4436 collides with main.", "Ticket T-4437 collides with main."),
        ("The run measured a recall of 0.75.", "The run measured a recall of 0.88."),
    ],
)
def test_identifiers_that_cross(a: str, b: str) -> None:
    assert crossed(a, b)
    assert pair_relation(a, b) is Relation.CROSSED


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Fixed in PR #447.", "Fixed in https://github.com/o/r/pull/447."),
        ("Released as v1.70.0.", "Released as version 1.70.0."),
        ("Capped at 1,024 tokens.", "Capped at 1024 tokens."),
        ("Issue 0042 adds the addend.", "Issue 42 adds the addend."),
        ("It costs $5.00 per 1M tokens.", "It costs $5 per 1 million tokens."),
        ("Commit 35dc4e3 merged it.", "Commit 35dc4e376a1b merged it."),
        ("Mem0 v3 removed belief revision.", "Mem0 version 3 removed belief revision."),
    ],
)
def test_formatting_alone_never_crosses(a: str, b: str) -> None:
    assert not crossed(a, b)
    assert not crossed(a, b, 1)


def test_a_restatement_that_drops_an_identifier_is_one_sided() -> None:
    a = "The fix landed in PR #447."
    b = "The fix landed in PR #447, released as 1.70.0."
    assert pair_relation(a, b) is Relation.ONE_SIDED
    assert not crossed(a, b)


def test_claims_with_no_identifiers_are_equal() -> None:
    assert pair_relation("The sky is blue.", "Sky is blue.") is Relation.EQUAL


def test_single_digits_fall_under_the_default_floor() -> None:
    a = "Use claude-opus-4-6 for the extractor."
    b = "Use claude-opus-4-5 for the extractor."
    assert MIN_LENGTH == 3
    assert not crossed(a, b)
    assert crossed(a, b, 1)


def test_the_floor_applies_after_normalising() -> None:
    # $1.00 and 1M both normalise to 1, which a floor of 3 drops on both sides.
    a = "Claude Haiku 4.5 costs $1.00 per 1M input tokens."
    b = "Claude Haiku 4.5 costs $5.00 per 1M output tokens."
    assert identifier_set(a) == identifier_set(b) == frozenset({"4.5"})
    assert pair_relation(a, b, 1) is Relation.ONE_SIDED


def test_a_fraction_with_another_digit_keeps_its_zeros() -> None:
    assert crossed("Requires Python 3.10.", "Requires Python 3.1.")


def test_a_three_part_version_keeps_its_zeros() -> None:
    assert crossed("Bumped to 1.10.0.", "Bumped to 1.1.0.")


def test_a_long_plain_number_is_not_a_commit_id() -> None:
    assert crossed("Comment 1234567 was posted.", "Comment 12345678 was posted.")


def test_the_rung_answers_different_or_abstain_only() -> None:
    assert IDENTIFIER_CROSS.decide is not None
    assert IDENTIFIER_CROSS.decide(("PR #284 opened.", "PR #288 opened.")) is (
        DuplicateVerdict.DIFFERENT
    )
    for pair in [
        ("Released as v1.70.0.", "Released as version 1.70.0."),
        ("Acme acquired Widget.", "Acme Corp bought Widget."),
        ("Acme acquired Widget.", "Acme acquired Widget."),
    ]:
        assert IDENTIFIER_CROSS.decide(pair) is DuplicateVerdict.ABSTAIN


def test_the_rung_is_admitted_pure_and_versioned() -> None:
    assert IDENTIFIER_CROSS.cost is CostClass.PURE
    assert not IDENTIFIER_CROSS.shadow
    assert str(IDENTIFIER_CROSS.stamp) == "identifier_cross@1"
