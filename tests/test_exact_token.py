# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The exact identifier-token tokeniser and share (§2)."""

from __future__ import annotations

import pytest

from particles.core.scoring.exact_token import identifier_tokens, token_share


class TestIdentifierTokens:
    @pytest.mark.parametrize(
        "text", ["Is REQ 0246 done?", "REQ-0246", "see #0246", "req0246 is not it"]
    )
    def test_record_forms_yield_the_bare_number(self, text: str) -> None:
        tokens = identifier_tokens(text, 3)
        if "req0246" in text:
            # A letter run joined to the digits is one token, kept whole.
            assert tokens == {"req0246"}
        else:
            assert tokens == {"0246"}

    def test_dot_joins_only_between_digits(self) -> None:
        assert identifier_tokens("ship 1.234.2 today", 3) == {"1.234.2"}
        assert identifier_tokens("ends with PR 480.", 3) == {"480"}
        # A prefixed version is one token, so it never matches the bare one.
        assert identifier_tokens("v1.231.0", 3) == {"v1.231.0"}

    def test_case_folded(self) -> None:
        assert identifier_tokens("Build S22 and s22", 3) == {"s22"}

    def test_length_minimum_drops_small_counts(self) -> None:
        assert identifier_tokens("top 40 for run #3", 3) == frozenset()
        assert identifier_tokens("top 40 for run #3", 2) == {"40"}

    def test_needs_a_digit(self) -> None:
        assert identifier_tokens("what about the split-package ADR?", 3) == frozenset()

    def test_unicode_digit_rule(self) -> None:
        # Numeric_Type Digit counts (superscript two); a vulgar fraction and a
        # Roman numeral are token characters but not digits.
        assert identifier_tokens("x²yz", 3) == {"x²yz"}
        assert identifier_tokens("½abc", 3) == frozenset()
        assert identifier_tokens("chapter Ⅷabc", 3) == frozenset()

    def test_underscore_splits(self) -> None:
        assert identifier_tokens("run_9110_x", 3) == {"9110"}


class TestTokenShare:
    def test_empty_tokens_is_zero(self) -> None:
        assert token_share("RFC 9110 is implementable.", frozenset()) == 0.0

    def test_full_and_partial_share(self) -> None:
        tokens = frozenset({"9110", "2026"})
        assert token_share("RFC 9110 was decided in 2026.", tokens) == 1.0
        assert token_share("Something happened in 2026.", tokens) == 0.5
        assert token_share("RFC 9112 is unrelated.", tokens) == 0.0

    def test_substring_without_token_is_zero(self) -> None:
        # "9110" occurs inside "91100" but is not one of its tokens.
        assert token_share("record 91100 only", frozenset({"9110"})) == 0.0

    def test_leading_zeros_are_significant(self) -> None:
        assert token_share("PR 696 merged", frozenset({"0696"})) == 0.0

    def test_content_is_case_folded(self) -> None:
        assert token_share("Build S22 shipped", frozenset({"s22"})) == 1.0
