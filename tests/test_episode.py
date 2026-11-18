# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The episode question's pure half: the reading, the ladder, the fold.

* A stored reading parses only with a verdict and a well-formed stamp, and
  ``ABSTAIN`` is never a stored verdict.
* The ladder: a credit or a direct assertion settles ``STANDING`` before any
  reading; the anchor and shape rungs are shadow, recorded and never returned;
  the reading decides under its writer's stamp.
* The fold: a card is withheld only when every side has one source entry and
  reads session-bound; the stratum is one entry or the rest; the entry key is
  the unordered stamp combination; the shadow counts session cards by
  distinct entry and nets them off the admitted withheld cards.
"""

from __future__ import annotations

from particles.core.episode import (
    EPISODE_KEY,
    EPISODE_MODEL_KEY,
    EPISODE_RULE_KEY,
    OPERATOR_RULE,
    SHAPES_DIGEST,
    CardSides,
    EpisodeClaim,
    EpisodeInput,
    EpisodeReading,
    Stratum,
    entry_key,
    episode_shadow,
    fold_entry,
    fold_stamp,
    parse_reading,
    reading_of,
    reading_verdict,
    shapes,
    stamp_for,
)
from particles.core.ladders import CREDIT, SESSION_ANCHOR, SHAPE, episode_ladder
from particles.core.rung_ladder import CostClass, EpisodeVerdict, QuestionKind, RungStamp, settle

REGEN = "regen.episode@aaa"
EXTRACT = "prompt.episode@bbb"


def _input(
    content: str = "The design is settled.",
    *,
    credited: bool = False,
    asserted: bool = False,
    basis: str | None = None,
    reading: EpisodeReading | None = None,
) -> EpisodeInput:
    return EpisodeInput(content, credited, asserted, basis, reading)


def _claim(
    pid: str, entries: set[str], reading: EpisodeReading | None, **kw: object
) -> EpisodeClaim:
    item = _input(reading=reading, **kw)  # type: ignore[arg-type]
    return EpisodeClaim(pid, frozenset(entries), settle(episode_ladder(reading), item))


SB = EpisodeReading(EpisodeVerdict.SESSION_BOUND, REGEN)
ST = EpisodeReading(EpisodeVerdict.STANDING, REGEN)


class TestReading:
    def test_a_reading_needs_a_verdict_and_a_stamp(self) -> None:
        assert parse_reading({EPISODE_KEY: "session_bound", EPISODE_RULE_KEY: REGEN}) == SB
        assert parse_reading({EPISODE_KEY: "session_bound"}) is None
        assert parse_reading({EPISODE_KEY: "abstain", EPISODE_RULE_KEY: REGEN}) is None
        assert parse_reading({EPISODE_KEY: "standing", EPISODE_RULE_KEY: "no-at-sign"}) is None
        assert parse_reading(None) is None

    def test_the_serving_model_is_part_of_the_stamp(self) -> None:
        assert str(stamp_for(REGEN, "anthropic:claude-sonnet-5-5")) == (
            f"{REGEN}~anthropic:claude-sonnet-5-5"
        )
        assert str(stamp_for(REGEN, None)) == REGEN
        assert stamp_for(OPERATOR_RULE, "anthropic:x") == RungStamp(OPERATOR_RULE, "1")
        stored = {EPISODE_KEY: "session_bound", EPISODE_RULE_KEY: REGEN, EPISODE_MODEL_KEY: "m:1"}
        reading = parse_reading(stored, fallback_model="ignored:0")
        assert reading is not None and reading.model == "m:1"
        bare = parse_reading({EPISODE_KEY: "standing", EPISODE_RULE_KEY: REGEN})
        assert bare is not None and bare.model is None

    def test_an_extracted_reading_takes_the_claims_extraction_model(self) -> None:
        extracted = {EPISODE_KEY: "standing", EPISODE_RULE_KEY: EXTRACT}
        assert reading_of(extracted, "anthropic:s").model == "anthropic:s"  # type: ignore[union-attr]
        regen = {EPISODE_KEY: "standing", EPISODE_RULE_KEY: REGEN}
        assert reading_of(regen, "anthropic:s").model is None  # type: ignore[union-attr]

    def test_an_operator_reading_has_its_own_stamp(self) -> None:
        reading = parse_reading({EPISODE_KEY: "standing", EPISODE_RULE_KEY: OPERATOR_RULE})
        assert reading is not None and reading.stamp == RungStamp(OPERATOR_RULE, "1")

    def test_reply_spellings_normalise_and_abstain_is_no_answer(self) -> None:
        assert reading_verdict("Session-Bound") is EpisodeVerdict.SESSION_BOUND
        assert reading_verdict(" standing ") is EpisodeVerdict.STANDING
        assert reading_verdict("abstain") is None
        assert reading_verdict(3) is None


class TestShapes:
    def test_session_report_shapes_match(self) -> None:
        assert "pr_number" in shapes("PR #447 is open")
        assert "version_change" in shapes("The version went from 1.114.1 to 1.114.2")
        assert not shapes("Backgrounded commits die at GPG signing")

    def test_the_shape_rung_is_versioned_by_its_patterns(self) -> None:
        assert SHAPE.version == SHAPES_DIGEST and len(SHAPES_DIGEST) == 12


class TestLadder:
    def test_the_ladder_answers_the_episode_kind(self) -> None:
        ladder = episode_ladder(SB)
        assert ladder.kind is QuestionKind.EPISODE
        assert [r.name for r in ladder.rungs] == [
            "credit",
            "session-anchor",
            "shape",
            "regen.episode",
        ]
        assert ladder.rungs[-1].cost is CostClass.MODEL
        assert SESSION_ANCHOR.shadow and SHAPE.shadow and not CREDIT.shadow

    def test_a_credit_or_an_assertion_is_standing_before_any_reading(self) -> None:
        for kw in ({"credited": True}, {"asserted": True}):
            s = settle(episode_ladder(SB), _input("PR #1 is open", reading=SB, **kw))  # type: ignore[arg-type]
            assert s.verdict is EpisodeVerdict.STANDING
            assert s.decided_by == CREDIT.stamp
            assert fold_stamp(s) is None

    def test_the_reading_decides_under_its_writers_stamp(self) -> None:
        s = settle(episode_ladder(SB), _input("PR #1 is open", basis="session", reading=SB))
        assert s.verdict is EpisodeVerdict.SESSION_BOUND
        assert str(s.decided_by) == REGEN
        # Shadow rungs ran and are recorded, never returned.
        assert {str(stamp) for stamp, _ in s.shadow} == {
            str(SESSION_ANCHOR.stamp),
            str(SHAPE.stamp),
        }

    def test_a_standing_reading_overrides_a_shadow_session_bound(self) -> None:
        s = settle(episode_ladder(ST), _input("PR #1 is open", basis="session", reading=ST))
        assert s.verdict is EpisodeVerdict.STANDING
        assert fold_stamp(s) is None

    def test_with_no_reading_the_fold_stamp_falls_back_to_a_shadow_rung(self) -> None:
        s = settle(episode_ladder(None), _input("PR #1 is open", basis="session"))
        assert s.verdict is EpisodeVerdict.ABSTAIN and s.decided_by is None
        assert fold_stamp(s) == SESSION_ANCHOR.stamp
        quiet = settle(episode_ladder(None), _input("Commits die at signing"))
        assert fold_stamp(quiet) is None


class TestFold:
    def test_one_entry_pair_folds_in_the_one_entry_stratum(self) -> None:
        entry = fold_entry([_claim("a", {"e1"}, SB), _claim("b", {"e1"}, SB)])
        assert entry is not None
        assert entry.stratum is Stratum.ONE_ENTRY
        assert entry.key == f"one_entry:{REGEN}"

    def test_a_pair_from_two_entries_is_the_rest_and_the_combination_is_unordered(self) -> None:
        other = EpisodeReading(EpisodeVerdict.SESSION_BOUND, EXTRACT)
        ab = fold_entry([_claim("a", {"e1"}, SB), _claim("b", {"e2"}, other)])
        ba = fold_entry([_claim("b", {"e2"}, other), _claim("a", {"e1"}, SB)])
        assert ab == ba and ab is not None
        assert ab.stratum is Stratum.REST
        assert ab.key == entry_key(Stratum.REST, [REGEN, EXTRACT])

    def test_a_group_card_folds_whole_under_its_distinct_stamps(self) -> None:
        other = EpisodeReading(EpisodeVerdict.SESSION_BOUND, EXTRACT)
        members = [_claim("a", {"e1"}, SB), _claim("b", {"e1"}, SB), _claim("c", {"e2"}, other)]
        entry = fold_entry(members)
        assert entry is not None and entry.stratum is Stratum.REST
        assert entry.stamps == tuple(sorted({REGEN, EXTRACT}))
        shadow = episode_shadow(
            [CardSides("duplicate_group", ("a", "b", "c"))], {m.particle_id: m for m in members}
        )
        assert shadow.withheld == 1 and shadow.session_cards == 2
        standing = _claim("d", {"e1"}, ST)
        assert fold_entry([*members, standing]) is None

    def test_a_multi_entry_or_standing_side_keeps_the_card(self) -> None:
        assert fold_entry([_claim("a", {"e1", "e2"}, SB), _claim("b", {"e1"}, SB)]) is None
        assert fold_entry([_claim("a", {"e1"}, ST), _claim("b", {"e1"}, SB)]) is None
        assert fold_entry([_claim("a", {"e1"}, SB)]) is None

    def test_the_shadow_counts_withheld_cards_and_distinct_sessions(self) -> None:
        claims = {
            "a": _claim("a", {"e1"}, SB),
            "b": _claim("b", {"e1"}, SB),
            "c": _claim("c", {"e2"}, SB),
            "d": _claim("d", {"e2"}, ST),
        }
        cards = [
            CardSides("duplicate_pair", ("a", "b")),
            CardSides("contradiction", ("a", "c")),
            CardSides("duplicate_pair", ("c", "d")),  # a standing side: stays
            CardSides("duplicate_pair", ("a", "zz")),  # an unasked side: stays
            CardSides("subjectless_source", ("a", "b")),  # not a pair question
        ]
        admitted = [f"one_entry:{REGEN}"]
        shadow = episode_shadow(cards, claims, admitted)
        assert shadow.withheld == 2
        assert shadow.session_cards == 2
        assert shadow.admitted_withheld == 1
        assert shadow.admitted_session_cards == 1
        assert shadow.admitted_net == 0
        one = shadow.entries[f"one_entry:{REGEN}"]
        assert one == {"withheld": {"duplicate_pair": 1}, "session_cards": 1, "admitted": True}
        rest = shadow.entries[f"rest:{REGEN}"]
        assert rest["withheld"] == {"contradiction": 1} and rest["admitted"] is False
        assert shadow.as_dict()["admitted_net"] == 0
