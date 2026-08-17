"""Tests for ``particles memory useful``.

The operation is pinned in ``tests/test_utility_feedback.py``; these pin what
the *verb* decides — the strict id resolution (a mis-resolved target would
silently credit the wrong belief with the store's heaviest single utility
signal), the ACTIVE-only refusal, and that it is an **operator** verb that does
not take its authorization from ``mcp.write.enabled_stores`` (the agent-facing
knob), matching the precedent set by ``particle retract``.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.utility_feedback import CLI_ACTOR


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


async def _insert(
    *,
    content: str = "Never prepend `export PATH` to a command.",
    status: Status = Status.ACTIVE,
    particle_id: str | None = None,
) -> str:
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=particle_id or str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.7, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        asserted_at=datetime.now(UTC),
        status=status,
    )
    async with session_scope() as session:
        await insert_particle(session, p)
        await session.commit()
    return p.id


# The CLI stamps the credit at wall-clock now and this reads it back moments
# later, so `R` has already decayed a little by the time we see it: at a 30-day
# half-life a 156 ms gap on a loaded CI runner costs ~1e-6. Compare with a
# tolerance that swallows tens of seconds of real time (~7e-5 of drift) but is
# still orders of magnitude tighter than any outcome worth catching here — a
# mined-weight credit (1.0), a double count (50.0), or no credit at all.
_DECAY_SLACK = 1e-3


async def _reinforcement(particle_id: str) -> float:
    from particles.db import session_scope
    from particles.store.utility_store import get_reinforcement_scores

    async with session_scope() as session:
        scores = await get_reinforcement_scores(session, [particle_id], 30.0, explicit_weight=25.0)
    return scores.get(particle_id, 0.0)


async def _events(particle_id: str) -> list[object]:
    from particles.db import session_scope
    from particles.store.event_store import EventRefKind, OperatorEventType, list_events

    async with session_scope() as session:
        return list(
            await list_events(
                session,
                ref_kind=EventRefKind.PARTICLE,
                ref_id=particle_id,
                event_type=OperatorEventType.BELIEF_MARKED_USEFUL,
            )
        )


class TestHappyPath:
    def test_credits_the_belief(self, runner: CliRunner, cli_db: Path) -> None:
        pid = asyncio.run(_insert())
        result = runner.invoke(app, ["memory", "useful", pid])
        assert result.exit_code == 0, result.output
        assert "useful" in result.output
        assert asyncio.run(_reinforcement(pid)) == pytest.approx(25.0, abs=_DECAY_SLACK)

    def test_accepts_the_display_prefix_form(self, runner: CliRunner, cli_db: Path) -> None:
        pid = asyncio.run(_insert())
        result = runner.invoke(app, ["memory", "useful", f"p-{pid[:8]}"])
        assert result.exit_code == 0, result.output
        assert asyncio.run(_reinforcement(pid)) > 0.0

    def test_records_the_event_under_the_cli_actor(self, runner: CliRunner, cli_db: Path) -> None:
        pid = asyncio.run(_insert())
        runner.invoke(app, ["memory", "useful", pid, "--reason", "miner cannot see it"])
        events = asyncio.run(_events(pid))
        assert len(events) == 1
        assert events[0].actor == CLI_ACTOR  # type: ignore[attr-defined]
        assert events[0].reason == "miner cannot see it"  # type: ignore[attr-defined]

    def test_works_with_agent_writes_disabled(self, runner: CliRunner, cli_db: Path) -> None:
        """an operator verb does not read the agent policy knob.

        ``mcp.write.enabled_stores`` is empty by default in tests, so this
        passing at all is the assertion — the CLI is the operator's own store.
        """
        from particles.config import get_config

        assert not get_config().mcp.write.enabled_stores
        pid = asyncio.run(_insert())
        assert runner.invoke(app, ["memory", "useful", pid]).exit_code == 0

    def test_second_press_same_day_is_reported_not_double_counted(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        pid = asyncio.run(_insert())
        runner.invoke(app, ["memory", "useful", pid])
        second = runner.invoke(app, ["memory", "useful", pid])
        assert second.exit_code == 0
        assert "Already marked" in second.output
        assert asyncio.run(_reinforcement(pid)) == pytest.approx(25.0, abs=_DECAY_SLACK)
        assert len(asyncio.run(_events(pid))) == 2


class TestResolutionIsStrict:
    def test_unknown_id_exits_2(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "useful", "does-not-exist"])
        assert result.exit_code == 2
        assert "matches no belief" in result.output

    def test_non_active_belief_exits_2(self, runner: CliRunner, cli_db: Path) -> None:
        pid = asyncio.run(_insert(status=Status.RETRACTED))
        result = runner.invoke(app, ["memory", "useful", pid])
        assert result.exit_code == 2
        assert "no ACTIVE belief" in result.output
        assert asyncio.run(_reinforcement(pid)) == 0.0

    def test_ambiguous_prefix_exits_2(self, runner: CliRunner, cli_db: Path) -> None:
        asyncio.run(_insert(particle_id="abc11111-0000-0000-0000-000000000001"))
        asyncio.run(_insert(particle_id="abc11111-0000-0000-0000-000000000002"))
        result = runner.invoke(app, ["memory", "useful", "abc11111"])
        assert result.exit_code == 2
        assert "ambiguous" in result.output

    def test_empty_id_exits_2(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "useful", "p-"])
        assert result.exit_code == 2
        assert "must not be empty" in result.output
