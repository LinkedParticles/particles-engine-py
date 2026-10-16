# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ingest/ladder_shell.py: availability, the driver and the run tally."""

from __future__ import annotations

import pytest

from particles.config import NetworkConfig, get_config
from particles.core.rung_ladder import (
    Availability,
    ContradictionVerdict,
    CostClass,
    Ladder,
    QuestionKind,
    Rung,
    RungStamp,
)
from particles.ingest import ladder_shell
from particles.ingest.ladder_shell import current_availability, drive, report, rung_tally

V = ContradictionVerdict
Item = tuple[str, str]


def _network_rung() -> Rung[Item, ContradictionVerdict]:
    return Rung(
        name="closure_fill",
        version="1",
        cost=CostClass.NETWORK,
        endpoint=("www.wikidata.org", 443),
    )


def _ladder(*rungs: Rung[Item, ContradictionVerdict]) -> Ladder[Item, ContradictionVerdict]:
    return Ladder(QuestionKind.CONTRADICTION, V.ABSTAIN, rungs)


class TestAvailability:
    def test_a_network_rung_is_unavailable_under_a_no_egress_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config(), "network", NetworkConfig(egress="none"))
        assert not current_availability().admits(_network_rung())

    def test_a_network_rung_is_available_on_an_open_posture(self) -> None:
        assert current_availability().admits(_network_rung())

    def test_a_model_rung_is_unavailable_while_the_breaker_is_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = Rung[Item, ContradictionVerdict](name="probe", version="1", cost=CostClass.MODEL)
        monkeypatch.setattr(ladder_shell, "llm_circuit_open", lambda: True)
        assert not current_availability().admits(model)


@pytest.mark.asyncio
class TestDrive:
    async def test_a_bound_rung_is_asked_only_when_the_cheaper_rungs_abstain(self) -> None:
        asked: list[Item] = []

        async def probe(item: Item) -> ContradictionVerdict:
            asked.append(item)
            return V.YES

        ladder = _ladder(
            Rung(name="key", version="1", cost=CostClass.LOCAL, decide=lambda i: V.ABSTAIN),
            Rung(name="probe", version="p", cost=CostClass.MODEL),
        )
        settled = await drive(
            ladder, ("a", "b"), {"probe": probe}, availability=lambda: Availability()
        )
        assert (settled.verdict, settled.decided_by) == (V.YES, RungStamp("probe", "p"))
        assert asked == [("a", "b")]

    async def test_a_model_rung_is_not_asked_while_the_breaker_is_open(self) -> None:
        async def probe(item: Item) -> ContradictionVerdict:
            raise AssertionError("asked while the breaker was open")

        ladder = _ladder(Rung(name="probe", version="p", cost=CostClass.MODEL))
        settled = await drive(
            ladder,
            ("a", "b"),
            {"probe": probe},
            availability=lambda: Availability(breaker_open=True),
        )
        assert (settled.verdict, settled.decided_by) == (V.ABSTAIN, None)

    async def test_every_ladder_run_reports_to_the_open_tally(self) -> None:
        ladder = _ladder(
            Rung(
                name="shadowed",
                version="1",
                cost=CostClass.PURE,
                decide=lambda i: V.NO,
                shadow=True,
            ),
            Rung(name="key", version="1", cost=CostClass.LOCAL, decide=lambda i: V.NO),
        )
        with rung_tally() as tally:
            for _ in range(3):
                await drive(ladder, ("a", "b"), {}, availability=lambda: Availability())
        assert tally.payload() == {
            "contradiction": {
                "rungs": {"key@1": {"no": 3}},
                "settled": {"key@1": {"no": 3}},
                "shadow": {"shadowed@1": {"no": 3}},
            }
        }

    async def test_reporting_with_no_tally_open_is_a_no_op(self) -> None:
        report(QuestionKind.CONTRADICTION, "rungs", None, "abstain")
