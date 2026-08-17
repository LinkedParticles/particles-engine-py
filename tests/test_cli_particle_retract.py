"""Tests for ``particles particle retract``.

The verb is thin — the operation, both guards, the ``retired_at`` stamp
and the event all predate it — so these tests pin the things
the *verb* decides:

* the authority call — it works with ``mcp.write.enabled_stores``
  **empty** and with ``allow_cross_asserter`` **false**, because an operator CLI
  verb must not take its authorization from an agent-facing policy knob;
* that it retires an *extractor-asserted* belief, the motivating class the
  cross-asserter guardrail correctly blocks agents from touching;
* that the guards it must **not** relax still fire (HUMAN_REVIEW, ACTIVE-only);
* and the write-path effects the row demanded: ``retired_at`` stamped and a
  reason-carrying ``PARTICLE_RETRACTED`` event under a CLI-specific actor.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli.particle import RETRACT_ACTOR
from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


async def _insert(
    *,
    content: str = '`pre-commit` can be found by prefixing `PATH="$PWD/.venv/bin:$PATH"`.',
    asserted_by: str = "general-extractor",
    calibration: CalibrationSource = CalibrationSource.EXTRACTOR_DIRECT,
    status: Status = Status.ACTIVE,
    particle_id: str | None = None,
) -> str:
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=particle_id or str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=1.0, calibration_source=calibration),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=asserted_by,
        asserted_at=datetime.now(UTC),
        status=status,
    )
    async with session_scope() as session:
        await insert_particle(session, p)
        await session.commit()
    return p.id


async def _load(particle_id: str) -> Any:
    from particles.db import session_scope
    from particles.store.particle_store import get_particle

    async with session_scope() as session:
        return await get_particle(session, particle_id)


async def _row(particle_id: str) -> Any:
    """The ORM row, for the ``retired_at`` column the model does not expose."""
    from particles.db import session_scope
    from particles.store.particle_store import ParticleRow

    async with session_scope() as session:
        return await session.get(ParticleRow, particle_id)


async def _events() -> list[Any]:
    from sqlalchemy import select

    from particles.db import session_scope
    from particles.store.event_store import OperatorEventRow

    async with session_scope() as session:
        return list((await session.execute(select(OperatorEventRow))).scalars())


class TestOperatorRetract:
    def test_retracts_an_extractor_asserted_belief(self, cli_db: Path, runner: CliRunner) -> None:
        """The motivating case: the class agents are correctly blocked from touching."""
        pid = asyncio.run(_insert())

        result = runner.invoke(
            app,
            ["particle", "retract", pid[:8], "--reason", "Contradicts the AGENTS.md rule", "--yes"],
        )
        assert result.exit_code == 0, result.output
        assert "Retracted" in result.output
        assert "particles lint" in result.output

        particle = asyncio.run(_load(pid))
        assert particle.status is Status.RETRACTED
        assert particle.status_reason is StatusReason.EXPLICIT_RETRACTION

    def test_works_with_agent_write_policy_fully_closed(
        self, cli_db: Path, runner: CliRunner
    ) -> None:
        """the authority decision, asserted directly.

        No store is MCP-write-enabled and ``allow_cross_asserter`` is false: the
        stock default, and the exact configuration under which the row was
        filed. If this ever starts requiring the agent allowlist, the verb has
        regressed into the workaround the row rejected.
        """
        from particles.config import get_config

        cfg = get_config()
        assert cfg.mcp.write.enabled_stores == []
        assert cfg.mcp.write.allow_cross_asserter is False

        pid = asyncio.run(_insert())
        result = runner.invoke(
            app, ["particle", "retract", pid, "--reason", "operator decision", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert asyncio.run(_load(pid)).status is Status.RETRACTED

    def test_stamps_retired_at(self, cli_db: Path, runner: CliRunner) -> None:
        """via ``update_particle_status``, which the row required."""
        pid = asyncio.run(_insert())
        assert asyncio.run(_row(pid)).retired_at is None

        runner.invoke(app, ["particle", "retract", pid, "--reason", "stale", "--yes"])
        assert asyncio.run(_row(pid)).retired_at is not None

    def test_records_the_reason_on_a_cli_actor_event(self, cli_db: Path, runner: CliRunner) -> None:
        """and the actor is what makes the log worth having."""
        pid = asyncio.run(_insert())
        runner.invoke(
            app, ["particle", "retract", pid, "--reason", "superseded by the live rule", "--yes"]
        )

        events = asyncio.run(_events())
        assert len(events) == 1
        assert events[0].event_type == "PARTICLE_RETRACTED"
        assert events[0].reason == "superseded by the live rule"
        assert events[0].actor == RETRACT_ACTOR
        assert RETRACT_ACTOR.startswith("cli:")


class TestGuardsStillApply:
    def test_human_review_belief_is_refused(self, cli_db: Path, runner: CliRunner) -> None:
        """Deliberately not widened: revising an operator-asserted belief is Review's job."""
        pid = asyncio.run(_insert(calibration=CalibrationSource.HUMAN_REVIEW))

        result = runner.invoke(
            app, ["particle", "retract", pid, "--reason", "changed my mind", "--yes"]
        )
        assert result.exit_code == 1
        assert "HUMAN_REVIEW" in result.output
        assert asyncio.run(_load(pid)).status is Status.ACTIVE

    def test_already_retracted_is_reported_not_re_retracted(
        self, cli_db: Path, runner: CliRunner
    ) -> None:
        """Idempotence is the ACTIVE-only guard's, not a special case in the verb."""
        pid = asyncio.run(_insert())
        assert (
            runner.invoke(app, ["particle", "retract", pid, "--reason", "first", "--yes"]).exit_code
            == 0
        )

        again = runner.invoke(app, ["particle", "retract", pid, "--reason", "second", "--yes"])
        assert again.exit_code == 1
        assert "not ACTIVE" in again.output
        # The first retraction's event is the only one; the refusal wrote nothing.
        assert len(asyncio.run(_events())) == 1


class TestArgumentHandling:
    def test_unknown_id_exits_nonzero(self, cli_db: Path, runner: CliRunner) -> None:
        result = runner.invoke(app, ["particle", "retract", "deadbeef", "--reason", "x", "--yes"])
        assert result.exit_code == 1
        assert "No particle matches" in result.output

    def test_ambiguous_prefix_is_refused(self, cli_db: Path, runner: CliRunner) -> None:
        """Retiring by eight characters demands the same disambiguation as reading by eight."""
        base = uuid.uuid4().hex
        shared = f"{base[:8]}-{base[8:12]}-4{base[13:16]}"
        a = f"{shared}-8{base[17:20]}-{base[20:32]}"
        b = f"{shared}-9{base[17:20]}-{base[20:32]}"
        asyncio.run(_insert(particle_id=a))
        asyncio.run(_insert(particle_id=b, content="A second claim sharing the prefix."))

        result = runner.invoke(app, ["particle", "retract", base[:8], "--reason", "x", "--yes"])
        assert result.exit_code == 1
        assert "Ambiguous prefix" in result.output
        assert asyncio.run(_load(a)).status is Status.ACTIVE
        assert asyncio.run(_load(b)).status is Status.ACTIVE

    def test_blank_reason_is_refused(self, cli_db: Path, runner: CliRunner) -> None:
        pid = asyncio.run(_insert())
        result = runner.invoke(app, ["particle", "retract", pid, "--reason", "   ", "--yes"])
        assert result.exit_code == 1
        assert "non-empty" in result.output
        assert asyncio.run(_load(pid)).status is Status.ACTIVE

    def test_missing_reason_is_a_usage_error(self, cli_db: Path, runner: CliRunner) -> None:
        pid = asyncio.run(_insert())
        assert runner.invoke(app, ["particle", "retract", pid]).exit_code == 2
        assert asyncio.run(_load(pid)).status is Status.ACTIVE

    def test_dry_run_shows_the_target_and_writes_nothing(
        self, cli_db: Path, runner: CliRunner
    ) -> None:
        pid = asyncio.run(_insert())
        result = runner.invoke(
            app, ["particle", "retract", pid, "--reason", "considering it", "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        assert "nothing written" in result.output
        assert "general-extractor" in result.output  # the target is shown before any write
        assert "considering it" in result.output
        assert asyncio.run(_load(pid)).status is Status.ACTIVE
        assert asyncio.run(_events()) == []

    def test_declining_the_prompt_writes_nothing(self, cli_db: Path, runner: CliRunner) -> None:
        pid = asyncio.run(_insert())
        result = runner.invoke(app, ["particle", "retract", pid, "--reason", "maybe"], input="n\n")
        assert result.exit_code != 0  # typer.confirm(abort=True)
        assert asyncio.run(_load(pid)).status is Status.ACTIVE
