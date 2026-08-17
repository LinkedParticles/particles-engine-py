"""Tests for the JSON Lines exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.exporters.jsonl import JsonlExporter
from particles.store.particle_store import insert_particle


def _p(content: str, conf: float = 0.9) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=conf, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="t",
        status=Status.ACTIVE,
    )


@pytest.mark.asyncio
async def test_one_json_object_per_active_particle(db_session: object, tmp_path: Path) -> None:
    session = db_session  # type: ignore[assignment]
    await insert_particle(session, _p("Claim one."))  # type: ignore[arg-type]
    await insert_particle(session, _p("Claim two."))  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    out = tmp_path / "particles.jsonl"
    summary = await JsonlExporter().export(session, out)  # type: ignore[arg-type]

    lines = out.read_text(encoding="utf-8").splitlines()
    assert summary.particles_written == 2
    assert len(lines) == 2
    objs = [json.loads(line) for line in lines]
    assert {o["content"] for o in objs} == {"Claim one.", "Claim two."}
    # Full Pydantic model dump, not the interchange unit.
    assert all("assertion_modality" in o and "confidence" in o for o in objs)


@pytest.mark.asyncio
async def test_min_confidence_filter_drops_below_threshold(
    db_session: object, tmp_path: Path
) -> None:
    session = db_session  # type: ignore[assignment]
    await insert_particle(session, _p("High.", 0.9))  # type: ignore[arg-type]
    await insert_particle(session, _p("Low.", 0.2))  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    out = tmp_path / "particles.jsonl"
    summary = await JsonlExporter().export(  # type: ignore[arg-type]
        session, out, min_particle_confidence=0.5
    )
    assert summary.particles_written == 1
    assert summary.particles_dropped_below_threshold == 1
    objs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert objs[0]["content"] == "High."


@pytest.mark.asyncio
async def test_requires_output_path(db_session: object) -> None:
    with pytest.raises(ValueError):
        await JsonlExporter().export(db_session, None)  # type: ignore[arg-type]
