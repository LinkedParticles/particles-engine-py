# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The predicate vocabulary's pure pieces: the canonical form and a slot's kind.

Letting reviewed profiles steer the update rung was proposed and declined,
and that code removed. What remains, and is tested here, serves
the vocabulary documents: the normaliser a canonical predicate is
keyed by, the kind read from a published constraint, and the subject-class
lookup the vocabulary step uses.
"""

from __future__ import annotations

from typing import Any

import pytest

from particles.core.predicate_profile import (
    MULTI_VALUE,
    SINGLE_BEST_VALUE,
    SINGLE_VALUE,
    SlotKind,
    SourceConstraints,
    kind_from_source,
    normalise_predicate,
)

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestNormalise:
    @pytest.mark.parametrize(
        "forms",
        [
            ["moved to", "was moved to", "moves to", "moving to", "move to"],
            ["lives in", "lived in", "live in", "is living in"],
            ["has status", "had status", "have status"],
            ["uses", "used", "use", "is using"],
            ["passes", "passed", "pass"],
            ["contains", "contained", "contain"],
            ["added", "adds", "add"],
            ["stopped", "stops", "stop"],
            ["covered", "covers", "cover"],
            ["renamed to", "renames to", "rename to"],
            ["agreed", "agrees", "agree"],
            ["applies to", "applied to", "apply to"],
            ["ran", "runs", "running", "run"],
            ["was written by", "is written by", "written by"],
        ],
    )
    def test_inflections_of_one_verb_share_a_form(self, forms: list[str]) -> None:
        assert len({normalise_predicate(f) for f in forms}) == 1

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("moved to", "moved from"),
            ("renamed to", "renamed from"),
            ("supports", "does not support"),
            ("includes", "does not include"),
            ("lives in", "is located at"),
        ],
    )
    def test_direction_polarity_and_synonyms_stay_apart(self, a: str, b: str) -> None:
        # The measured failure of embedding clusters (finding 1): a cardinality
        # rule over a merged `to` / `from` cluster would compare an old value
        # with a new one.
        assert normalise_predicate(a) != normalise_predicate(b)

    def test_has_is_the_main_verb_before_a_noun(self) -> None:
        assert normalise_predicate("has status") != normalise_predicate("status")
        assert normalise_predicate("has implemented") == normalise_predicate("implements")

    def test_articles_case_and_spacing_do_not_matter(self) -> None:
        assert normalise_predicate("Is The Author Of") == normalise_predicate("is author of")


# ---------------------------------------------------------------------------
# The kind read from the source
# ---------------------------------------------------------------------------

TIME = frozenset({"P580"})


class TestKindFromSource:
    @pytest.mark.parametrize(
        ("constraints", "expected"),
        [
            pytest.param(SourceConstraints(), None, id="nothing-said"),
            pytest.param(
                SourceConstraints(owl_functional=True), SlotKind.TIMELESS_SINGLE, id="owl"
            ),
            pytest.param(SourceConstraints(sh_max_count=1), SlotKind.TIMELESS_SINGLE, id="shacl"),
            pytest.param(SourceConstraints(sh_max_count=3), None, id="shacl-max-3"),
            pytest.param(
                SourceConstraints(wikidata=((SINGLE_VALUE, frozenset()),)),
                SlotKind.TIMELESS_SINGLE,
                id="single-value",
            ),
            pytest.param(
                SourceConstraints(wikidata=((SINGLE_VALUE, frozenset({"P585"})),)),
                SlotKind.ONE_AT_A_TIME,
                id="single-value-time-separator",
            ),
            pytest.param(
                SourceConstraints(wikidata=((SINGLE_VALUE, frozenset({"P1545"})),)),
                None,
                id="single-value-other-separator",
            ),
            pytest.param(
                # P569 date of birth, checked live: single-best-value whose
                # separators are object-has-role and sourcing circumstances.
                SourceConstraints(wikidata=((SINGLE_BEST_VALUE, frozenset({"P3831", "P1480"})),)),
                SlotKind.TIMELESS_SINGLE,
                id="date-of-birth",
            ),
            pytest.param(
                SourceConstraints(wikidata=((SINGLE_BEST_VALUE, TIME),)),
                SlotKind.ONE_AT_A_TIME,
                id="single-best-time-separator",
            ),
            pytest.param(
                SourceConstraints(wikidata=((MULTI_VALUE, frozenset()),)),
                SlotKind.MANY_AT_ONCE,
                id="multi-value",
            ),
            pytest.param(
                # P551 residence, P108 employer, P50 author, checked live: no
                # single-value rule. Allowed time qualifiers are not a
                # constraint item here at all, and say nothing on their own.
                SourceConstraints(wikidata=(("Q21510851", TIME),)),
                None,
                id="allowed-time-qualifiers-alone",
            ),
        ],
    )
    def test_reading(self, constraints: SourceConstraints, expected: SlotKind | None) -> None:
        assert kind_from_source(constraints) is expected

    def test_disagreeing_sources_keep_more(self) -> None:
        both = SourceConstraints(
            sh_max_count=1, wikidata=((SINGLE_BEST_VALUE, TIME), (MULTI_VALUE, frozenset()))
        )
        assert kind_from_source(both) is SlotKind.TIMELESS_SINGLE
        timed_and_many = SourceConstraints(
            wikidata=((SINGLE_VALUE, TIME), (MULTI_VALUE, frozenset()))
        )
        assert kind_from_source(timed_and_many) is SlotKind.MANY_AT_ONCE


@pytest.mark.asyncio
async def test_get_subject_classes_reads_only_the_named_subjects(db_session: Any) -> None:
    from particles.core.schema import Subject
    from particles.store.subject_store import get_subject_classes, insert_subject

    classed = Subject(canonical_name="the user", subject_class="person", asserted_by="test")
    bare = Subject(canonical_name="Boston", asserted_by="test")
    for s in (classed, bare):
        await insert_subject(db_session, s)
    await db_session.commit()

    got = await get_subject_classes(db_session, [classed.id, bare.id, "no-such-id", ""])
    assert got == {classed.id: "person", bare.id: None}
    assert await get_subject_classes(db_session, []) == {}


@pytest.mark.asyncio
async def test_get_subject_classes_chunks_past_the_in_clause_limit(
    db_session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.core.schema import Subject
    from particles.store import subject_store

    subjects = [
        Subject(canonical_name=f"s{i}", subject_class=f"c{i}", asserted_by="test") for i in range(5)
    ]
    for s in subjects:
        await subject_store.insert_subject(db_session, s)
    await db_session.commit()

    monkeypatch.setattr(subject_store, "_IN_CHUNK", 2)
    got = await subject_store.get_subject_classes(db_session, [s.id for s in subjects])
    assert got == {s.id: s.subject_class for s in subjects}
