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

    def test_a_stamp_round_trips(self) -> None:
        stamp = RungStamp("duplicate_key", "1")
        assert RungStamp.parse(str(stamp)) == stamp
        with pytest.raises(ValueError):
            RungStamp.parse("no-version")


@pytest.mark.asyncio
async def test_the_update_probe_port_reproduces_the_code_before_it() -> None:
    """§ Acceptance item 2: zero differences, pair for pair, at read_update_slot."""
    from tests._update_probe_identity import GOLDEN, collect

    before = json.loads(GOLDEN.read_text())
    after = await collect()
    assert len(before) == len(after) > 700
    differences = {k: (before[k], after.get(k)) for k in before if before[k] != after.get(k)}
    assert differences == {}
