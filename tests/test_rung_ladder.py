# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/rung_ladder.py: the rung ladder's dispatch.

Also the update-probe port's acceptance bar: the ported ladder's verdicts on
the trip-versus-move and update-gate suites are identical to the code's before
the port, pair for pair, asserted at ``read_update_slot``.
"""

from __future__ import annotations

import json

import pytest

from particles.core.rung_ladder import (
    Availability,
    ContradictionVerdict,
    CostClass,
    Ladder,
    QuestionKind,
    Rung,
    RungStamp,
    annotation_is_current,
    settle,
    stamp_guarded,
)

V = ContradictionVerdict
Item = tuple[str, str]


def _rung(
    name: str,
    answer: ContradictionVerdict,
    *,
    cost: CostClass = CostClass.PURE,
    shadow: bool = False,
    confirms: frozenset[str] = frozenset(),
    endpoint: tuple[str, int | None] | None = None,
    seen: list[str] | None = None,
) -> Rung[Item, ContradictionVerdict]:
    def decide(_: Item) -> ContradictionVerdict:
        if seen is not None:
            seen.append(name)
        return answer

    return Rung(
        name=name,
        version="1",
        cost=cost,
        decide=decide,
        shadow=shadow,
        confirms=confirms,
        endpoint=endpoint,
    )


def _ladder(*rungs: Rung[Item, ContradictionVerdict]) -> Ladder[Item, ContradictionVerdict]:
    return Ladder(QuestionKind.CONTRADICTION, V.ABSTAIN, rungs)


class TestDispatch:
    def test_the_first_non_abstain_verdict_stands_with_its_stamp(self) -> None:
        seen: list[str] = []
        ladder = _ladder(
            _rung("probe", V.YES, cost=CostClass.MODEL, seen=seen),
            _rung("guard", V.ABSTAIN, seen=seen),
            _rung("key", V.NO, cost=CostClass.LOCAL, seen=seen),
        )
        settled = settle(ladder, ("a", "b"))
        assert settled.verdict is V.NO
        assert settled.decided_by == RungStamp("key", "1")
        # Ascending cost class: the model rung never ran.
        assert seen == ["guard", "key"]
        assert [str(s) for s, _ in settled.trail] == ["guard@1", "key@1"]

    def test_every_rung_abstaining_names_no_rung(self) -> None:
        ladder = _ladder(_rung("a", V.ABSTAIN), _rung("b", V.ABSTAIN, cost=CostClass.LOCAL))
        settled = settle(ladder, ("a", "b"))
        assert settled.verdict is V.ABSTAIN
        assert settled.decided_by is None

    def test_a_rung_that_does_not_apply_abstains(self) -> None:
        skipped = Rung(
            name="never",
            version="1",
            cost=CostClass.PURE,
            decide=lambda _: V.YES,
            applies=lambda _: False,
        )
        settled = settle(_ladder(skipped, _rung("b", V.NO, cost=CostClass.LOCAL)), ("a", "b"))
        assert settled.decided_by == RungStamp("b", "1")
        assert settled.trail == ((RungStamp("b", "1"), V.NO),)

    def test_applies_is_asked_of_the_input(self) -> None:
        only_x: Rung[Item, ContradictionVerdict] = Rung(
            name="only_x",
            version="1",
            cost=CostClass.PURE,
            decide=lambda _: V.YES,
            applies=lambda item: item[0] == "x",
        )
        ladder = _ladder(only_x, _rung("b", V.NO, cost=CostClass.LOCAL))
        assert settle(ladder, ("x", "y")).decided_by == RungStamp("only_x", "1")
        assert settle(ladder, ("a", "y")).decided_by == RungStamp("b", "1")

    def test_an_all_abstaining_run_keeps_its_trail_and_shadow(self) -> None:
        """Both feed the run tally, which counts abstentions too."""
        ladder = _ladder(
            _rung("watch", V.YES, shadow=True),
            _rung("guard", V.ABSTAIN, cost=CostClass.LOCAL),
        )
        settled = settle(ladder, ("a", "b"))
        assert (settled.verdict, settled.decided_by) == (V.ABSTAIN, None)
        assert settled.trail == ((RungStamp("guard", "1"), V.ABSTAIN),)
        assert settled.shadow == ((RungStamp("watch", "1"), V.YES),)

    def test_a_shadow_rung_is_recorded_and_never_returned(self) -> None:
        ladder = _ladder(
            _rung("candidate", V.YES, shadow=True),
            _rung("probe", V.NO, cost=CostClass.LOCAL),
        )
        settled = settle(ladder, ("a", "b"))
        assert settled.verdict is V.NO
        assert settled.decided_by == RungStamp("probe", "1")
        assert settled.shadow == ((RungStamp("candidate", "1"), V.YES),)
        assert RungStamp("candidate", "1") not in [s for s, _ in settled.trail]

    def test_a_model_rung_is_never_shadow(self) -> None:
        with pytest.raises(ValueError, match="never run in shadow"):
            _rung("probe", V.YES, cost=CostClass.MODEL, shadow=True)


class TestConfirmation:
    def _ladder(self, first: V, second: V) -> Ladder[Item, ContradictionVerdict]:
        return _ladder(
            _rung("probe", first, cost=CostClass.MODEL),
            _rung("reading", second, cost=CostClass.MODEL, confirms=frozenset({"yes"})),
        )

    def test_the_confirming_rung_replaces_the_provisional_verdict(self) -> None:
        settled = settle(self._ladder(V.YES, V.NO), ("a", "b"))
        assert (settled.verdict, settled.decided_by) == (V.NO, RungStamp("reading", "1"))

    def test_the_confirmation_joins_the_trail(self) -> None:
        settled = settle(self._ladder(V.YES, V.NO), ("a", "b"))
        assert settled.trail == (
            (RungStamp("probe", "1"), V.YES),
            (RungStamp("reading", "1"), V.NO),
        )

    def test_a_confirmation_of_another_verdict_is_passed_over(self) -> None:
        ladder = _ladder(
            _rung("probe", V.YES, cost=CostClass.MODEL),
            _rung("on_no", V.YES, cost=CostClass.MODEL, confirms=frozenset({"no"})),
            _rung("on_yes", V.NO, cost=CostClass.MODEL, confirms=frozenset({"yes"})),
        )
        settled = settle(ladder, ("a", "b"))
        assert (settled.verdict, settled.decided_by) == (V.NO, RungStamp("on_yes", "1"))

    def test_the_confirming_rung_runs_only_on_its_named_verdict(self) -> None:
        settled = settle(self._ladder(V.NO, V.YES), ("a", "b"))
        assert (settled.verdict, settled.decided_by) == (V.NO, RungStamp("probe", "1"))
        assert len(settled.trail) == 1

    def test_a_confirmation_abstaining_replaces_the_provisional_verdict(self) -> None:
        settled = settle(self._ladder(V.YES, V.ABSTAIN), ("a", "b"))
        assert (settled.verdict, settled.decided_by) == (V.ABSTAIN, RungStamp("reading", "1"))

    def test_a_confirmation_that_cannot_run_leaves_the_verdict_unconfirmed(self) -> None:
        ladder = self._ladder(V.YES, V.YES)
        run = ladder.start(("a", "b"))
        first = run.next_rung(Availability())
        assert first is not None
        run.record(first, V.YES)
        # The breaker opens between the two readings.
        assert run.next_rung(Availability(breaker_open=True)) is None
        settled = run.settled()
        assert (settled.verdict, settled.decided_by) == (V.ABSTAIN, RungStamp("reading", "1"))
        assert len(settled.trail) == 1

    def test_a_confirmation_never_runs_without_a_provisional_verdict(self) -> None:
        settled = settle(self._ladder(V.ABSTAIN, V.YES), ("a", "b"))
        assert (settled.verdict, settled.decided_by) == (V.ABSTAIN, None)

    def test_a_confirmation_that_does_not_apply_leaves_the_verdict_unconfirmed(self) -> None:
        reading = Rung[Item, ContradictionVerdict](
            name="reading",
            version="1",
            cost=CostClass.MODEL,
            decide=lambda _: V.YES,
            applies=lambda _: False,
            confirms=frozenset({"yes"}),
        )
        settled = settle(_ladder(_rung("probe", V.YES, cost=CostClass.MODEL), reading), ("a", "b"))
        assert (settled.verdict, settled.decided_by) == (V.ABSTAIN, RungStamp("reading", "1"))
        assert len(settled.trail) == 1

    @staticmethod
    def _split(name: str, verdict: V, sees: str) -> Rung[Item, ContradictionVerdict]:
        return Rung[Item, ContradictionVerdict](
            name=name,
            version="1",
            cost=CostClass.MODEL,
            decide=lambda _: verdict,
            applies=lambda item: item[0] == sees,
            confirms=frozenset({"yes"}),
        )

    def test_two_confirmations_split_the_inputs_between_them(self) -> None:
        """the passage rung, and the claim-text rung for the assertion door."""
        ladder = _ladder(
            _rung("probe", V.YES, cost=CostClass.MODEL),
            self._split("passage", V.NO, sees="sourced"),
            self._split("claim_text", V.YES, sees="asserted"),
        )
        sourced = settle(ladder, ("sourced", "b"))
        assert (sourced.verdict, sourced.decided_by) == (V.NO, RungStamp("passage", "1"))
        asserted = settle(ladder, ("asserted", "b"))
        assert (asserted.verdict, asserted.decided_by) == (V.YES, RungStamp("claim_text", "1"))
        assert len(asserted.trail) == 2

    def test_only_one_confirmation_answers(self) -> None:
        ladder = _ladder(
            _rung("probe", V.YES, cost=CostClass.MODEL),
            self._split("passage", V.YES, sees="a"),
            self._split("claim_text", V.NO, sees="a"),
        )
        settled = settle(ladder, ("a", "b"))
        assert (settled.verdict, settled.decided_by) == (V.YES, RungStamp("passage", "1"))
        assert len(settled.trail) == 2

    def test_no_confirmation_applying_leaves_the_verdict_unconfirmed(self) -> None:
        ladder = _ladder(
            _rung("probe", V.YES, cost=CostClass.MODEL),
            self._split("passage", V.YES, sees="x"),
            self._split("claim_text", V.YES, sees="y"),
        )
        settled = settle(ladder, ("a", "b"))
        assert (settled.verdict, settled.decided_by) == (V.ABSTAIN, RungStamp("passage", "1"))

    def test_a_confirmation_sorted_before_its_target_is_rejected(self) -> None:
        # A pure confirmer of a model rung's YES would be passed before any
        # provisional verdict exists and never revisited: skipped forever.
        with pytest.raises(ValueError, match="rung check: a confirmation rung sorts after"):
            _ladder(
                _rung("probe", V.YES, cost=CostClass.MODEL),
                _rung("check", V.NO, confirms=frozenset({"yes"})),
            )
        # Declaration order within one class counts too.
        with pytest.raises(ValueError, match="rung check"):
            _ladder(
                _rung("check", V.NO, cost=CostClass.MODEL, confirms=frozenset({"yes"})),
                _rung("probe", V.YES, cost=CostClass.MODEL),
            )


class TestAvailability:
    def test_a_network_rung_abstains_under_an_egress_refusal(self) -> None:
        network = _rung("fetch", V.NO, cost=CostClass.NETWORK, endpoint=("www.wikidata.org", 443))
        refused = Availability(egress_allows=lambda host, port: False)
        assert not refused.admits(network)
        assert Availability().admits(network)
        settled = settle(_ladder(network), ("a", "b"), refused)
        assert settled.decided_by is None

    def test_a_network_rung_names_its_endpoint(self) -> None:
        with pytest.raises(ValueError, match="endpoint"):
            _rung("fetch", V.NO, cost=CostClass.NETWORK)

    def test_a_model_rung_abstains_while_the_breaker_is_open(self) -> None:
        model = _rung("probe", V.YES, cost=CostClass.MODEL)
        assert not Availability(breaker_open=True).admits(model)
        settled = settle(_ladder(model), ("a", "b"), Availability(breaker_open=True))
        assert settled.decided_by is None

    def test_a_pure_rung_runs_whatever_the_world_says(self) -> None:
        pure = _rung("guard", V.NO)
        closed = Availability(egress_allows=lambda h, p: False, breaker_open=True)
        assert closed.admits(pure)


class TestStamps:
    def test_a_stale_stamp_annotation_abstains(self) -> None:
        comparator = _rung("comparator", V.NO)
        stamps = {"fresh": "structurer@2", "stale": "structurer@1", "missing": None}
        guarded = stamp_guarded(
            comparator,
            abstain=V.ABSTAIN,
            stamp_of=lambda item: stamps[item[0]],
            current="structurer@2",
        )
        assert guarded.decide is not None
        assert guarded.decide(("fresh", "x")) is V.NO
        assert guarded.decide(("stale", "x")) is V.ABSTAIN
        assert guarded.decide(("missing", "x")) is V.ABSTAIN
        assert not annotation_is_current(None, "structurer@2")

    def test_a_guarded_rung_keeps_the_rung_it_guards(self) -> None:
        def decide(item: Item) -> ContradictionVerdict:
            return V.NO if item[1] == "x" else V.YES

        for inner in (
            Rung(
                name="comparator",
                version="3",
                cost=CostClass.NETWORK,
                decide=decide,
                applies=lambda item: item[0] != "skip",
                confirms=frozenset({"yes"}),
                endpoint=("example.org", 443),
                content_keyed=False,
            ),
            Rung(name="watcher", version="4", cost=CostClass.LOCAL, decide=decide, shadow=True),
        ):
            guarded = stamp_guarded(
                inner, abstain=V.ABSTAIN, stamp_of=lambda _: "structurer@2", current="structurer@2"
            )
            fields = ("name", "version", "cost", "applies", "confirms", "shadow", "endpoint")
            for name in (*fields, "content_keyed"):
                assert getattr(guarded, name) == getattr(inner, name), name
            assert guarded.decide is not None
            assert guarded.decide(("fresh", "x")) is V.NO
            assert guarded.decide(("fresh", "y")) is V.YES

    def test_a_stamp_round_trips(self) -> None:
        stamp = RungStamp("duplicate_key", "1")
        assert RungStamp.parse(str(stamp)) == stamp
        with pytest.raises(ValueError):
            RungStamp.parse("no-version")

    def test_a_stamp_splits_at_its_last_at_and_needs_both_halves(self) -> None:
        assert RungStamp.parse("judge@v@2") == RungStamp("judge@v", "2")
        for text in ("judge@", "@2"):
            with pytest.raises(ValueError):
                RungStamp.parse(text)


@pytest.mark.asyncio
async def test_the_update_probe_port_reproduces_the_code_before_it() -> None:
    """§ Acceptance item 2: zero differences, pair for pair, at read_update_slot."""
    from tests._update_probe_identity import GOLDEN, collect

    before = json.loads(GOLDEN.read_text())
    after = await collect()
    assert len(before) == len(after) > 700
    differences = {k: (before[k], after.get(k)) for k in before if before[k] != after.get(k)}
    assert differences == {}
