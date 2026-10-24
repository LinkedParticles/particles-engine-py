# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The typographic fold in the exact-duplicate key.

The fold widens "the same sentence" by typography only. The tests in
``test_duplicate_suppression.py`` pin the older half (whitespace and a
sentence-final mark) and the one-token differences that must never collapse;
these pin what the fold joins and, as importantly, what it must keep apart.
"""

from __future__ import annotations

import unicodedata

import pytest

from particles.core.duplicate_key import content_hash, normalize_content


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # The measured case: a vault note's claim re-minted without its quotes.
        (
            '"No hay negros en el Tibet" publica episodios un par de veces al mes.',
            "No hay negros en el Tibet publica episodios un par de veces al mes.",
        ),
        # Curly against straight double quotes, and guillemets.
        ("The header is “skills-2025-10-02”.", 'The header is "skills-2025-10-02".'),
        ("Le titre est «La Peste».", "Le titre est La Peste."),
        # A curly apostrophe against a straight one.
        ("The store’s path is fixed.", "The store's path is fixed."),
        # Dashes and the minus sign.
        ("Options 1–3 are supported.", "Options 1-3 are supported."),
        ("It failed — twice.", "It failed - twice."),
        ("The floor is −5.", "The floor is -5."),
        # The ellipsis character against three dots.
        ("It waits… then retries.", "It waits... then retries."),
        # Invisible marks.
        ("Zero​width.", "Zerowidth."),
        # Composed against decomposed.
        ("Café opens at nine.", unicodedata.normalize("NFD", "Café opens at nine.")),
    ],
)
def test_typographic_variants_are_one_claim(left: str, right: str) -> None:
    assert normalize_content(left) == normalize_content(right)
    assert content_hash(left) == content_hash(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # A single quote is also an apostrophe: dropping it would join these.
        ("The dog's bowl is empty.", "The dogs bowl is empty."),
        ("The dog's bowl is empty.", "The dogs' bowl is empty."),
        # Compatibility folding is not applied: a superscript is not a digit.
        ("The area is 4 m².", "The area is 4 m2."),
        ("Half is ½.", "Half is 1/2."),
        # A prime is a mark, not an apostrophe: folded to ' it would be trimmed
        # as a trailing mark and the derivative would key equal to the function.
        ("The derivative is f\u2032.", "The derivative is f."),
        # Case and wording are still preserved.
        ('The "Live" store.', "The live store."),
    ],
)
def test_meaningful_differences_survive_the_fold(left: str, right: str) -> None:
    assert normalize_content(left) != normalize_content(right)


def test_fold_keeps_wording_and_order() -> None:
    assert normalize_content("  “A” — b…  ") == "A - b"


def test_fold_is_idempotent() -> None:
    once = normalize_content("It’s “done” – mostly…")
    assert normalize_content(once) == once


def test_fold_matches_migration_055() -> None:
    """The migration vendors the fold; the two must key every text alike."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_m055", Path("alembic/versions/055_typographic_content_key.py")
    )
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    for text in (
        '"Quoted" claim.',
        "It’s 1–3…",
        "Café​ «x» − ′",
        "Plain claim",
    ):
        assert migration._hash_v2(text) == content_hash(text)
