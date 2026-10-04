# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The verbs: ``particles particle reclassify`` and ``particles modality``.

The record, the regeneration pass and the lens are covered by
``tests/test_modality.py``; this file pins the CLI wrappers: argument
validation, the operator verdict end to end over a file-backed store, the
remote refusal, and the dry-run envelope.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource

runner = CliRunner()


def _seed() -> str:
    async def _insert() -> str:
        from particles.db import session_scope
        from particles.store.particle_store import insert_particle

        particle = Particle(
            content="Tabs are nicer than spaces.",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce-1")],
        )
        async with session_scope() as session:
            await insert_particle(session, particle)
            await session.commit()
        return particle.id

    return asyncio.run(_insert())


def _read(particle_id: str) -> tuple[str, str | None]:
    async def _get() -> tuple[str, str | None]:
        from particles.db import session_scope
        from particles.store.particle_store import ParticleRow

        async with session_scope() as session:
            row = await session.get(ParticleRow, particle_id)
            assert row is not None
            return row.assertion_modality, row.modality_classifier

    return asyncio.run(_get())


def test_reclassify_rejects_an_unknown_modality(cli_db: Path) -> None:
    pid = _seed()
    result = runner.invoke(
        app, ["particle", "reclassify", pid[:8], "--modality", "OPINION", "--reason", "x"]
    )
    assert result.exit_code == 1
    assert "Unknown modality 'OPINION'" in result.output
    assert _read(pid) == ("FALSIFIABLE", None)


def test_reclassify_rejects_a_blank_reason(cli_db: Path) -> None:
    pid = _seed()
    result = runner.invoke(
        app, ["particle", "reclassify", pid[:8], "--modality", "EVALUATIVE", "--reason", "  "]
    )
    assert result.exit_code == 1
    assert "--reason must be non-empty" in result.output


def test_reclassify_dry_run_writes_nothing(cli_db: Path) -> None:
    pid = _seed()
    result = runner.invoke(
        app,
        [
            "particle",
            "reclassify",
            pid[:8],
            "--modality",
            "evaluative",
            "--reason",
            "opinion",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--dry-run: nothing written." in result.output
    assert _read(pid) == ("FALSIFIABLE", None)


def test_reclassify_writes_the_operator_verdict(cli_db: Path) -> None:
    pid = _seed()
    result = runner.invoke(
        app, ["particle", "reclassify", pid[:8], "--modality", "EVALUATIVE", "--reason", "opinion"]
    )
    assert result.exit_code == 0, result.output
    assert "FALSIFIABLE → EVALUATIVE" in result.output
    assert _read(pid) == (AssertionModality.EVALUATIVE.value, "operator")

    events = runner.invoke(app, ["events", "list", "--type", "MODALITY_RECLASSIFIED"])
    assert events.exit_code == 0, events.output
    assert "cli:particle-reclassify" in events.output


def test_modality_refuses_a_remote_engine(cli_db: Path) -> None:
    remote = MagicMock()
    remote.remote = True
    with patch("particles.api.client.get_backend", return_value=remote):
        result = runner.invoke(app, ["modality", "--dry-run"])
    assert result.exit_code == 2
    assert "one local store" in result.output


def test_modality_dry_run_reports_the_census(cli_db: Path) -> None:
    _seed()
    result = runner.invoke(app, ["modality", "--dry-run"])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["dry_run"] is True
    assert summary["backlog"] == 0  # an assertion is unclassified, not stale
    assert summary["census"]["by_state"]["unclassified"] == 1
    assert summary["classifier"].startswith("prompt.modality@")

    with_unclassified = runner.invoke(app, ["modality", "--dry-run", "--include-unclassified"])
    assert json.loads(with_unclassified.output)["backlog"] == 1
