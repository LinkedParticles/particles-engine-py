# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``particles particle source``.

The derivation is covered by ``tests/test_source_passage.py``; this pins what
the verb decides: prefix resolution, the stdout/stderr split that keeps the
passage pipeable, and a non-zero exit when there is no passage to print.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)

SOURCE = "# Notes\n\n- The deploy script needs the staging token.\n- Lunch is at noon.\n"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


async def _seed(content: str, *, with_source: bool = True) -> str:
    from particles.corpus.deposit import deposit_text
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    async with session_scope() as session:
        provenance = []
        if with_source:
            entry_id, snap_id = await deposit_text(session, SOURCE, source_type="LOCAL_MARKDOWN")
            provenance = [
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snap_id
                )
            ]
        p = Particle(
            id=str(uuid.uuid4()),
            content=content,
            confidence=Confidence(value=0.8),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="general-extractor",
            asserted_at=datetime.now(UTC),
            provenance=provenance,
        )
        await insert_particle(session, p)
        await session.commit()
    return p.id


def test_prints_the_passage_on_stdout_and_metadata_on_stderr(
    runner: CliRunner, cli_db: Path
) -> None:
    pid = asyncio.run(_seed("The deploy script needs the staging token."))
    result = runner.invoke(app, ["particle", "source", pid[:8]], catch_exceptions=False)
    assert result.exit_code == 0
    assert result.stdout.strip() == "- The deploy script needs the staging token."
    assert "Match:     located: best term overlap with the belief (100%)" in result.stderr
    assert "(LOCAL_MARKDOWN)" in result.stderr


def test_unknown_id_exits_one(runner: CliRunner, cli_db: Path) -> None:
    result = runner.invoke(app, ["particle", "source", "deadbeef"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "No particle matches prefix 'deadbeef'." in result.stderr


def test_no_passage_exits_one_with_the_reason(runner: CliRunner, cli_db: Path) -> None:
    pid = asyncio.run(_seed("A belief with no corpus source.", with_source=False))
    result = runner.invoke(app, ["particle", "source", pid], catch_exceptions=False)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Match:     unavailable" in result.stderr
    assert "no corpus source" in result.stderr
