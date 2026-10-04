# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles reindex --estimate`` and ``--only-changed-components``.

The argument and gate paths: the refused flag, the flag combinations
``--estimate`` cannot honour, the fail-closed non-interactive run, and a
``--yes`` run that spends through the reindex meter. The estimate's own
arithmetic is tested in ``test_reindex_estimate.py``.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    ExtractorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.reindex_estimate import (
    EstimatePlan,
    ReindexEstimate,
    SweepProjection,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, args: list[str]) -> Any:
    return runner.invoke(app, args, catch_exceptions=False)


async def _seed_version_scope() -> None:
    """One snapshot with a blob and one 0.15.0 claim: a sampleable version scope."""
    from particles.corpus.deposit import save_blob, sha256
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    raw = b"The bridge opened in 1932."
    content_hash = sha256(raw)
    save_blob(raw, content_hash)
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="WEB_PAGE",
        uri_r="https://example.com/bridge",
        deposited_by="test",
    )
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash=content_hash,
        extraction_status=ExtractionStatus.COMPLETE,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    async with session_scope() as session:
        session.add(CorpusEntryRow.from_model(entry))
        session.add(SnapshotRow.from_model(snap, entry.entry_id))
        await session.flush()
        await insert_particle(
            session,
            Particle(
                id=str(uuid.uuid4()),
                content="The bridge opened in 1932.",
                confidence=Confidence(
                    value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="test",
                asserted_at=datetime.now(UTC),
                status=Status.ACTIVE,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE,
                        corpus_entry_id=entry.entry_id,
                        snapshot_id=snap.snapshot_id,
                    )
                ],
                extractor_ref=ExtractorRef(name="general-extractor", version="0.15.0"),
            ),
        )
        await session.commit()


def _report(plan: EstimatePlan) -> ReindexEstimate:
    return ReindexEstimate(
        plan=plan,
        projection=SweepProjection(
            measured=1,
            changed=0,
            share=0.0,
            share_low=0.0,
            share_high=0.79,
            projected_changed=0,
            claim_change_rate=0.0,
            sample_extraction_usd=0.01,
            projected_sweep_usd=0.01,
        ),
    )


class TestOnlyChangedComponentsFlag:
    def test_refused_with_the_reason(self, runner: CliRunner, cli_db: Path) -> None:
        extract = AsyncMock(return_value=[])
        with patch("particles.operations.reindex.extract_snapshot", new=extract):
            result = _invoke(
                runner, ["reindex", "--extractor-version", "0.15.0", "--only-changed-components"]
            )
        assert result.exit_code == 2
        assert "not enabled yet" in result.output
        extract.assert_not_called()


class TestEstimateArguments:
    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--estimate"], "pass the superseded version with --extractor-version"),
            (
                ["--estimate", "--extractor-version", "0.15.0", "--dry-run"],
                "--estimate and --dry-run are separate reports",
            ),
        ],
    )
    def test_misuse_is_refused_before_any_read(
        self, runner: CliRunner, cli_db: Path, args: list[str], message: str
    ) -> None:
        result = _invoke(runner, ["reindex", *args])
        assert result.exit_code == 2
        assert message in result.output

    def test_a_non_interactive_run_without_yes_fails_closed(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        asyncio.run(_seed_version_scope())
        spend = AsyncMock()
        with patch("particles.operations.reindex_estimate.run_reindex_estimate", new=spend):
            result = _invoke(runner, ["reindex", "--estimate", "--extractor-version", "0.15.0"])
        assert result.exit_code == 2
        # The free plan still prints, so the operator sees what --yes would buy.
        assert "Reindex plan: 1 entries, 1 snapshots, 1 particles" in result.output
        assert "Estimate: re-extract 1 of 1 sampleable snapshot(s)" in result.output
        assert "Nothing was spent" in result.output
        spend.assert_not_called()

    def test_yes_spends_through_the_reindex_meter(self, runner: CliRunner, cli_db: Path) -> None:
        asyncio.run(_seed_version_scope())
        routes: list[str] = []

        from particles.operations import llm_spend

        real_meter = llm_spend.MeteredExtractRun

        def meter(**kwargs: Any) -> Any:
            routes.append(kwargs["route"])
            return real_meter(**kwargs)

        async def spend(session: Any, plan: EstimatePlan) -> ReindexEstimate:
            return _report(plan)

        with (
            patch("particles.operations.llm_spend.MeteredExtractRun", side_effect=meter),
            patch("particles.operations.reindex_estimate.run_reindex_estimate", new=spend),
        ):
            result = _invoke(
                runner,
                ["reindex", "--estimate", "--extractor-version", "0.15.0", "--yes", "--seed", "3"],
            )
        assert result.exit_code == 0, result.output
        assert routes == ["reindex"]
        assert "(seed 3)" in result.output
        assert "0 of 1 sampled snapshot(s) changed" in result.output
        assert "Nothing was written" in result.output

    def test_nothing_to_sample_spends_nothing(self, runner: CliRunner, cli_db: Path) -> None:
        spend = AsyncMock()
        with patch("particles.operations.reindex_estimate.run_reindex_estimate", new=spend):
            result = _invoke(
                runner, ["reindex", "--estimate", "--extractor-version", "0.15.0", "--yes"]
            )
        assert result.exit_code == 0
        assert "Nothing to sample" in result.output
        spend.assert_not_called()
