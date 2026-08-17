"""CLI tests for narrative construction + inspection.

Exercises the operator path end to end: build a NARRATIVE over claims with
``links add --type part-of`` / ``--type sequence-in``, then read it back with
``particle narrative`` and ``particle show``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    Confidence,
    Particle,
    ParticleType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.db import session_scope
from particles.store.particle_store import insert_particle


def _mk(content: str, particle_id: str, *, kind: ParticleType = ParticleType.CLAIM) -> Particle:
    return Particle(
        id=particle_id,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        particle_type=kind,
    )


def _seed(db_path: Path, particles: list[Particle]) -> None:
    async def _impl() -> None:
        async with session_scope() as session:
            for p in particles:
                await insert_particle(session, p)
            await session.commit()

    asyncio.run(_impl())


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


_NAR = "00000000-0000-0000-0000-0000000000aa"
_C1 = "00000000-0000-0000-0000-0000000000b1"
_C2 = "00000000-0000-0000-0000-0000000000b2"


def test_links_add_part_of_and_sequence_in(runner: CliRunner, cli_db: Path) -> None:
    """The new --type values are accepted and create directional edges."""
    _seed(
        cli_db,
        [
            _mk("Lunch with Sarah", _NAR, kind=ParticleType.NARRATIVE),
            _mk("Had lunch.", _C1),
            _mk("She got a job.", _C2),
        ],
    )
    r1 = runner.invoke(
        app, ["links", "add", _C1, _NAR, "--type", "part-of"], catch_exceptions=False
    )
    assert r1.exit_code == 0, r1.output
    assert "PART_OF" in r1.output
    r2 = runner.invoke(
        app, ["links", "add", _C1, _C2, "--type", "sequence-in"], catch_exceptions=False
    )
    assert r2.exit_code == 0, r2.output
    assert "SEQUENCE_IN" in r2.output


def test_particle_narrative_shows_ordered_constituents(runner: CliRunner, cli_db: Path) -> None:
    _seed(
        cli_db,
        [
            _mk("A two-step memory", _NAR, kind=ParticleType.NARRATIVE),
            _mk("First step.", _C1),
            _mk("Second step.", _C2),
        ],
    )
    runner.invoke(app, ["links", "add", _C1, _NAR, "--type", "part-of"], catch_exceptions=False)
    runner.invoke(app, ["links", "add", _C2, _NAR, "--type", "part-of"], catch_exceptions=False)
    runner.invoke(app, ["links", "add", _C1, _C2, "--type", "sequence-in"], catch_exceptions=False)

    result = runner.invoke(app, ["particle", "narrative", _NAR], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Constituents (2, in sequence):" in result.output
    # First step listed before second step.
    assert result.output.index("First step.") < result.output.index("Second step.")


def test_particle_narrative_rejects_non_narrative(runner: CliRunner, cli_db: Path) -> None:
    _seed(cli_db, [_mk("Just a claim.", _C1)])
    result = runner.invoke(app, ["particle", "narrative", _C1], catch_exceptions=False)
    assert result.exit_code == 1
    assert "not NARRATIVE" in result.output


def test_particle_narrative_empty_hint(runner: CliRunner, cli_db: Path) -> None:
    _seed(cli_db, [_mk("Bare narrative", _NAR, kind=ParticleType.NARRATIVE)])
    result = runner.invoke(app, ["particle", "narrative", _NAR], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "No constituents yet" in result.output


def test_particle_narrative_synthesize_rejects_non_narrative(
    runner: CliRunner, cli_db: Path
) -> None:
    """--synthesize runs the same NARRATIVE-type guard before any synthesis."""
    _seed(cli_db, [_mk("Just a claim.", _C1)])
    result = runner.invoke(
        app, ["particle", "narrative", _C1, "--synthesize"], catch_exceptions=False
    )
    assert result.exit_code == 1
    assert "not NARRATIVE" in result.output


def test_particle_narrative_synthesize_empty(runner: CliRunner, cli_db: Path) -> None:
    """--synthesize on a constituent-less narrative reports nothing to render."""
    _seed(cli_db, [_mk("Bare narrative", _NAR, kind=ParticleType.NARRATIVE)])
    result = runner.invoke(
        app, ["particle", "narrative", _NAR, "--synthesize"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "nothing to synthesize" in result.output


def test_particle_narrative_synthesize_fallback_without_key(
    runner: CliRunner, cli_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--synthesize with no API key falls back to the deterministic cited
    structured listing — prose with particle references, no crash."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # force the no-LLM fallback
    _seed(
        cli_db,
        [
            _mk("A two-step memory", _NAR, kind=ParticleType.NARRATIVE),
            _mk("First step.", _C1),
            _mk("Second step.", _C2),
        ],
    )
    runner.invoke(app, ["links", "add", _C1, _NAR, "--type", "part-of"], catch_exceptions=False)
    runner.invoke(app, ["links", "add", _C2, _NAR, "--type", "part-of"], catch_exceptions=False)
    runner.invoke(app, ["links", "add", _C1, _C2, "--type", "sequence-in"], catch_exceptions=False)

    result = runner.invoke(
        app, ["particle", "narrative", _NAR, "--synthesize"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert f"p-{_C1[:8]}" in result.output  # particle references present
    assert "structured-listing fallback" in result.output


def test_particle_show_surfaces_narrative_membership(runner: CliRunner, cli_db: Path) -> None:
    _seed(
        cli_db,
        [
            _mk("Lunch narrative", _NAR, kind=ParticleType.NARRATIVE),
            _mk("A constituent claim.", _C1),
        ],
    )
    runner.invoke(app, ["links", "add", _C1, _NAR, "--type", "part-of"], catch_exceptions=False)

    # The claim shows the narrative it belongs to.
    claim_show = runner.invoke(app, ["particle", "show", _C1], catch_exceptions=False)
    assert claim_show.exit_code == 0, claim_show.output
    assert "Part of narratives:" in claim_show.output

    # The narrative shows its constituents.
    nar_show = runner.invoke(app, ["particle", "show", _NAR], catch_exceptions=False)
    assert nar_show.exit_code == 0, nar_show.output
    assert "Narrative constituents" in nar_show.output
