# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A stored calibration applies only under the extractor version it was fitted under.

Measured 2026-10-01 on recorded general-extractor runs: the prompt change
between 0.15.0 and 0.16.0 moved raw ECE from 0.030 to 0.100, and a temperature
fitted on one version raised the other's ECE. A fit that inverts is worse than
none, and the confidence it writes is immutable, so a record whose
version differs from the running extractor's, or is unknown, is not applied.

Covers the pure decision, the particle constructor's backstop, the extract
pipeline (one warning per run, not per claim), and the CLI listing. The
calibrate verb's stamp is pinned in ``tests/test_calibration_integration.py``
beside the rest of that verb's tests.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from sqlalchemy import select

from particles.core.conflict_resolution import SlotVerdict
from particles.core.schema import (
    ExtractorCalibration,
    ExtractorRef,
    Mutability,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.extraction.calibration import TRANSFORM_LOGIT, version_mismatch_reason
from particles.extraction.general import (
    CandidateParticle,
    ExtractionResult,
    candidate_to_particle,
)


def _record(
    extractor_version: str | None, *, provider_model: str | None = None
) -> ExtractorCalibration:
    return ExtractorCalibration(
        temperature=2.0,
        transform=TRANSFORM_LOGIT,
        fitted_at=datetime(2026, 9, 17, tzinfo=UTC),
        benchmark_suite_id="prose-calibration-001",
        sample_size=40,
        calibration_error_before=0.144,
        calibration_error_after=0.038,
        provider_model=provider_model,
        extractor_version=extractor_version,
    )


# ---------------------------------------------------------------------------
# The pure decision
# ---------------------------------------------------------------------------


class TestVersionMismatchReason:
    def test_same_version_applies(self) -> None:
        assert version_mismatch_reason(_record("0.16.0"), "0.16.0") is None

    def test_other_version_names_both(self) -> None:
        reason = version_mismatch_reason(_record("0.15.0"), "0.16.0")
        assert reason == "fitted under 0.15.0, running 0.16.0"

    def test_unknown_version_is_refused(self) -> None:
        """A record from before the key existed says nothing about its prompt."""
        reason = version_mismatch_reason(_record(None), "0.16.0")
        assert reason == "fitted under an unknown extractor version, running 0.16.0"

    def test_comparison_is_exact(self) -> None:
        """No version arithmetic: a patch bump can change the prompt too."""
        assert version_mismatch_reason(_record("0.16.0"), "0.16.1") is not None

    def test_existing_record_json_reads_as_unknown(self) -> None:
        """Every record already stored deserializes with no version, and is refused."""
        legacy = _record(None).model_dump(exclude={"extractor_version"})
        loaded = ExtractorCalibration.model_validate(legacy)
        assert loaded.extractor_version is None
        assert version_mismatch_reason(loaded, "0.16.0") is not None


# ---------------------------------------------------------------------------
# candidate_to_particle — the backstop when the caller names the version
# ---------------------------------------------------------------------------


def _candidate() -> CandidateParticle:
    return CandidateParticle(
        content="some claim",
        confidence_value=0.8,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=["S"],
    )


class TestCandidateToParticleVersionGate:
    def test_matching_version_is_applied(self) -> None:
        particle = candidate_to_particle(
            _candidate(),
            "e",
            "s",
            asserted_by="x",
            extractor_ref=ExtractorRef(name="x", version="0.16.0"),
            calibration=_record("0.16.0"),
        )
        assert particle.confidence.calibration_source is CalibrationSource.CALIBRATED_BENCHMARK
        assert particle.confidence.value == pytest.approx(2 / 3)

    @pytest.mark.parametrize("fitted", ["0.15.0", None])
    def test_mismatched_or_unknown_version_mints_extractor_direct(self, fitted: str | None) -> None:
        particle = candidate_to_particle(
            _candidate(),
            "e",
            "s",
            asserted_by="x",
            extractor_ref=ExtractorRef(name="x", version="0.16.0"),
            calibration=_record(fitted),
        )
        assert particle.confidence.calibration_source is CalibrationSource.EXTRACTOR_DIRECT
        assert particle.confidence.value == pytest.approx(0.8)
        assert particle.confidence.calibration_ref is None


# ---------------------------------------------------------------------------
# The extract pipeline
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9']+")

#: Three unrelated claims, so no §6.6 pair forms and every one is written.
CLAIMS = [
    "The lighthouse on Skerry Point was automated in 1987.",
    "Marmalade is traditionally made from Seville oranges.",
    "The Rhine flows into the North Sea.",
]


class _BagOfWords:
    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                out[i, int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16) % 256] += 1.0
            out[i] /= max(float(np.linalg.norm(out[i])), 1e-9)
        return out


class _Scripted:
    """An extractor at version 0.16.0 emitting :data:`CLAIMS` at stated 0.8."""

    EXTRACTOR_ID = "version-test-extractor"
    EXTRACTOR_VERSION = "0.16.0"

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> ExtractionResult:
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=claim,
                    confidence_value=0.8,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=[f"Subject {i}"],
                )
                for i, claim in enumerate(CLAIMS)
            ]
        )


@pytest.fixture
def offline_pipeline() -> Generator[None, None, None]:
    """A deterministic encoder, and write-time probes that find nothing."""
    from particles import embeddings as ep

    original = ep._embedding_model
    ep.set_embedding_model(_BagOfWords())  # type: ignore[arg-type]
    try:
        with (
            patch(
                "particles.ingest.pipeline._has_contradiction_signal",
                AsyncMock(return_value=False),
            ),
            patch(
                "particles.ingest.pipeline._has_update_signal",
                AsyncMock(return_value=SlotVerdict.DIFFERENT),
            ),
        ):
            yield
    finally:
        ep.set_embedding_model(original)


async def _extract_with(session: Any, calibration: ExtractorCalibration | None) -> list[Any]:
    from particles.corpus.deposit import deposit_text_versioned
    from particles.ingest.pipeline import _current_extraction_provider_model, extract_snapshot
    from particles.store.extractor_store import upsert_calibration

    if calibration is not None:
        calibration = calibration.model_copy(
            update={"provider_model": _current_extraction_provider_model()}
        )
        await upsert_calibration(session, _Scripted.EXTRACTOR_ID, calibration)
    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text="\n".join(CLAIMS) + "\n",
        uri_r="file:///notes/calibration-version.md",
        source_type="LOCAL_MARKDOWN",
        mutability=Mutability.STABLE,
        deposited_by="test",
    )
    await session.commit()
    await extract_snapshot(session, entry_id, snapshot_id, extractor=_Scripted())  # type: ignore[arg-type]
    await session.commit()

    from particles.store.particle_store import ParticleRow

    rows = (
        (await session.execute(select(ParticleRow).where(ParticleRow.content.in_(CLAIMS))))
        .scalars()
        .all()
    )
    return [r.to_model() for r in rows]


def _warnings_about_calibration(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "Ignoring the stored calibration" in r.getMessage()
    ]


class TestPipelineVersionGate:
    @pytest.mark.asyncio
    async def test_record_fitted_under_the_running_version_is_applied(
        self, db_session: Any, offline_pipeline: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="particles.ingest.pipeline")
        particles = await _extract_with(db_session, _record("0.16.0"))
        assert len(particles) == 3
        for p in particles:
            assert p.confidence.calibration_source is CalibrationSource.CALIBRATED_BENCHMARK
            assert p.confidence.value == pytest.approx(2 / 3)
        assert _warnings_about_calibration(caplog) == []

    @pytest.mark.asyncio
    async def test_record_fitted_under_another_version_is_not_applied(
        self, db_session: Any, offline_pipeline: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="particles.ingest.pipeline")
        particles = await _extract_with(db_session, _record("0.15.0"))
        assert len(particles) == 3
        for p in particles:
            assert p.confidence.calibration_source is CalibrationSource.EXTRACTOR_DIRECT
            assert p.confidence.value == pytest.approx(0.8)
        # One warning for the run, not one per claim, naming both versions.
        warnings = _warnings_about_calibration(caplog)
        assert len(warnings) == 1
        assert "NOT APPLIED (fitted under 0.15.0, running 0.16.0)" in warnings[0]
        assert "--regenerate" in warnings[0]

    @pytest.mark.asyncio
    async def test_record_with_unknown_version_is_not_applied(
        self, db_session: Any, offline_pipeline: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="particles.ingest.pipeline")
        particles = await _extract_with(db_session, _record(None))
        assert {p.confidence.calibration_source for p in particles} == {
            CalibrationSource.EXTRACTOR_DIRECT
        }
        warnings = _warnings_about_calibration(caplog)
        assert len(warnings) == 1
        assert "unknown extractor version" in warnings[0]

    @pytest.mark.asyncio
    async def test_refused_record_stays_stored(
        self, db_session: Any, offline_pipeline: None
    ) -> None:
        """Not applying is not deleting: the record stays for refit or retirement."""
        from particles.ingest.pipeline import _current_extraction_provider_model
        from particles.store.extractor_store import get_calibration

        await _extract_with(db_session, _record("0.15.0"))
        kept = await get_calibration(
            db_session, _Scripted.EXTRACTOR_ID, _current_extraction_provider_model()
        )
        assert kept is not None
        assert kept.extractor_version == "0.15.0"
