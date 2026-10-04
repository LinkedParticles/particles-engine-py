# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``particles memory rebuild-utility`` (§4).

The operation is pinned in ``tests/test_utility_exposure.py``; these pin what
the verb decides: it prints the re-mine's judge calls and priced cost before
anything is cleared, a declined confirmation clears nothing, ``--yes`` skips
the prompt, and a literal-only configuration is called out as recording
nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.config import get_config
from particles.operations.spend_estimate import PassEstimate
from particles.operations.utility_mining import MiningResult


class _Plan:
    mines: list[object] = [object(), object()]
    calls = 7

    def estimate(self) -> PassEstimate:
        return PassEstimate(calls=7, usd=0.5, basis="")


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import particles.operations.utility_mining as um

    seen: dict[str, Any] = {"rebuilt": False}

    async def plan(_store: str) -> _Plan:
        return _Plan()

    async def rebuild(_store: str, _plan: object, *, latency_tolerant: bool) -> MiningResult:
        seen.update(rebuilt=True, latency_tolerant=latency_tolerant)
        return MiningResult(literal=3, behavioural=1, candidates=10, behavioural_calls=7)

    monkeypatch.setattr(um, "plan_store_utility", plan)
    monkeypatch.setattr(um, "rebuild_store_utility", rebuild)
    return seen


def test_the_cost_is_shown_and_a_decline_clears_nothing(stubbed: dict[str, Any]) -> None:
    result = CliRunner().invoke(app, ["memory", "rebuild-utility"], input="n\n")
    assert result.exit_code == 1
    assert "2 harvested session(s)" in result.output
    assert "7 judge call(s)" in result.output
    assert "$0.50 at list price" in result.output
    assert "batched" in result.output
    assert stubbed["rebuilt"] is False


def test_yes_skips_the_prompt_and_batches(stubbed: dict[str, Any]) -> None:
    result = CliRunner().invoke(app, ["memory", "rebuild-utility", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Proceed?" not in result.output
    assert stubbed == {"rebuilt": True, "latency_tolerant": True}
    assert "3 literal + 1 behavioural events" in result.output


def test_no_batch_sends_the_calls_one_at_a_time(stubbed: dict[str, Any]) -> None:
    result = CliRunner().invoke(app, ["memory", "rebuild-utility", "--yes", "--no-batch"])
    assert result.exit_code == 0, result.output
    assert stubbed["latency_tolerant"] is False
    assert "batched" not in result.output


def test_a_literal_only_configuration_is_said_to_record_nothing(
    stubbed: dict[str, Any],
) -> None:
    get_config().utility.mining.behavioural_matching = False
    result = CliRunner().invoke(app, ["memory", "rebuild-utility", "--yes"])
    assert result.exit_code == 0, result.output
    assert "mined channel will be rebuilt empty" in result.output
