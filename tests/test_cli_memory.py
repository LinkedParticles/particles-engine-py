"""Tests for the ``particles memory consolidate`` verb (§1/§8).

Pins the CLI contract with the operation mocked: flag validation, the exit
codes (0 success/skip, 1 pass(es) failed, 2 could not start), the kwargs the
thin CLI threads through, the rendered/json/--output surfaces, and the
projection-runner injection seam. The operation itself is pinned in
``tests/test_consolidation.py``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from particles.api.cli import app
from particles.config import get_config
from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.core.scoring.utility import sweep_rank_lift
from particles.core.status import Status
from particles.operations.consolidation import ConsolidationPass, ConsolidationReport

runner = CliRunner()

NOW = datetime.now(UTC)


def _report(**kwargs: object) -> ConsolidationReport:
    defaults: dict[str, object] = {
        "store": "default",
        "completed_at": NOW,
        "passes": [ConsolidationPass(name="census", status="ran")],
    }
    defaults.update(kwargs)
    return ConsolidationReport(**defaults)  # type: ignore[arg-type]


class TestFlagValidation:
    def test_bad_format_exits_2(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "consolidate", "--format", "yaml"])
        assert result.exit_code == 2
        assert "--format" in result.output

    def test_bad_scope_exits_2(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "consolidate", "--scope", "harvested"])
        assert result.exit_code == 2
        assert "--scope" in result.output


class TestExitCodes:
    def test_success_exits_0(self, cli_db: Path) -> None:
        consolidate = AsyncMock(return_value=_report())
        with patch("particles.operations.consolidation.run_consolidation", consolidate):
            result = runner.invoke(app, ["memory", "consolidate"])
        assert result.exit_code == 0, result.output
        assert "Consolidated store 'default'" in result.output

    def test_skip_exits_0_with_one_line(self, cli_db: Path) -> None:
        consolidate = AsyncMock(
            return_value=_report(
                outcome="skipped", skip_reason="consolidation already running — skipped"
            )
        )
        with patch("particles.operations.consolidation.run_consolidation", consolidate):
            result = runner.invoke(app, ["memory", "consolidate", "--if-due"])
        assert result.exit_code == 0, result.output
        assert "already running — skipped" in result.output
        assert "Consolidated store" not in result.output  # one log line, no report

    def test_failed_pass_exits_1(self, cli_db: Path) -> None:
        consolidate = AsyncMock(
            return_value=_report(
                passes=[
                    ConsolidationPass(name="census", status="failed", detail="boom"),
                    ConsolidationPass(name="projection", status="ran"),
                ]
            )
        )
        with patch("particles.operations.consolidation.run_consolidation", consolidate):
            result = runner.invoke(app, ["memory", "consolidate"])
        assert result.exit_code == 1
        assert "pass(es) failed: census" in result.output

    def test_could_not_start_exits_2(self, cli_db: Path) -> None:
        consolidate = AsyncMock(side_effect=RuntimeError("no such table: particles"))
        with patch("particles.operations.consolidation.run_consolidation", consolidate):
            result = runner.invoke(app, ["memory", "consolidate"])
        assert result.exit_code == 2
        assert "could not start" in result.output


class TestThreading:
    def test_flags_thread_through_to_operation(self, cli_db: Path) -> None:
        consolidate = AsyncMock(return_value=_report())
        with patch("particles.operations.consolidation.run_consolidation", consolidate):
            result = runner.invoke(
                app,
                [
                    "memory",
                    "consolidate",
                    "--if-due",
                    "--structural-only",
                    "--scope",
                    "store",
                ],
            )
        assert result.exit_code == 0, result.output
        kwargs = consolidate.call_args.kwargs
        assert kwargs["if_due"] is True
        assert kwargs["structural_only"] is True
        assert kwargs["scope"] == "store"
        assert kwargs["store"] == "default"
        # The Surface injects the projection tail (or its disclosed skip reason).
        assert "projection_runner" in kwargs
        assert "projection_skip_reason" in kwargs

    def test_json_format(self, cli_db: Path) -> None:
        consolidate = AsyncMock(return_value=_report())
        with patch("particles.operations.consolidation.run_consolidation", consolidate):
            result = runner.invoke(app, ["memory", "consolidate", "--format", "json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["store"] == "default"
        assert data["outcome"] == "ran"

    def test_output_writes_markdown(self, cli_db: Path, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        consolidate = AsyncMock(return_value=_report())
        with patch("particles.operations.consolidation.run_consolidation", consolidate):
            result = runner.invoke(app, ["memory", "consolidate", "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert out.is_file()
        assert "Consolidated store 'default'" in out.read_text()


# ---------------------------------------------------------------------------
# sweep-rank-lift
# ---------------------------------------------------------------------------

#: A fixed id so a test can name its 8-char truncation the way an operator
#: reads one out of the digest.
TARGET_UUID = "4fbcc320-c6c0-43ba-bb09-26a5dd97da86"


def _seed_particle(pid: str, *, content: str = "seeded", status: Status = Status.ACTIVE) -> None:
    """Insert one particle into the ``cli_db`` store, synchronously."""

    async def _insert() -> None:
        from particles.db import session_scope
        from particles.store.particle_store import insert_particle

        async with session_scope() as session:
            await insert_particle(
                session,
                Particle(
                    id=pid,
                    content=content,
                    confidence=Confidence(
                        value=0.7, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                    ),
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    asserted_by="test",
                    asserted_at=NOW,
                    status=status,
                ),
            )
            await session.commit()

    asyncio.run(_insert())


class TestSweepRankLiftValidation:
    """The flag contract. Every rejection exits 2 before any store work."""

    def test_bad_format_exits_2(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "sweep-rank-lift", "--format", "yaml"])
        assert result.exit_code == 2
        assert "--format" in result.output

    def test_non_positive_grid_max_exits_2(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "sweep-rank-lift", "--grid-max", "0"])
        assert result.exit_code == 2
        assert "--grid-max" in result.output

    def test_zero_grid_steps_exits_2(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "sweep-rank-lift", "--grid-steps", "0"])
        assert result.exit_code == 2
        assert "--grid-steps" in result.output

    def test_distinct_ratio_out_of_range_exits_2(self, cli_db: Path) -> None:
        for bad in ("0", "1.5", "-0.1"):
            result = runner.invoke(app, ["memory", "sweep-rank-lift", "--distinct-ratio", bad])
            assert result.exit_code == 2, bad
            assert "--distinct-ratio" in result.output

    def test_distinct_ratio_of_one_is_allowed(self, cli_db: Path) -> None:
        # The strict all-distinct criterion is severe, not invalid
        # reports both, so the CLI must not reject it.
        result = runner.invoke(app, ["memory", "sweep-rank-lift", "--distinct-ratio", "1.0"])
        assert result.exit_code == 0

    def test_non_positive_head_exits_2(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "sweep-rank-lift", "--head", "0"])
        assert result.exit_code == 2
        assert "--head" in result.output


class TestSweepRankLiftOutput:
    def test_empty_store_reports_nothing_to_calibrate(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "sweep-rank-lift"])
        assert result.exit_code == 0
        assert "nothing to calibrate" in result.output

    def test_json_format_is_parseable_on_an_empty_store(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "sweep-rank-lift", "--format", "json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["scored"] == 0
        assert payload["points"] == []

    def test_flags_thread_through_to_the_operation(self, cli_db: Path) -> None:
        _seed_particle(TARGET_UUID)
        with patch(
            "particles.operations.utility_sweep.sweep_store_rank_lift", new_callable=AsyncMock
        ) as mock_sweep:
            mock_sweep.return_value = sweep_rank_lift([], grid=(0.0,), head_sizes=[60])
            result = runner.invoke(
                app,
                [
                    "memory",
                    "sweep-rank-lift",
                    "--target",
                    "p-4fbcc320",
                    "--head",
                    "60",
                    "--head",
                    "200",
                    "--grid-max",
                    "0.05",
                    "--grid-steps",
                    "10",
                    "--distinct-ratio",
                    "0.9",
                ],
            )
        assert result.exit_code == 0, result.output
        kwargs = mock_sweep.await_args.kwargs
        assert kwargs["head_sizes"] == [60, 200]
        assert kwargs["grid_max"] == 0.05
        assert kwargs["grid_steps"] == 10
        assert kwargs["distinct_ratio"] == 0.9
        # The display prefix is stripped AND the truncation is expanded to the
        # full store id. Passing ``4fbcc320`` straight through is what made the
        # sweep score the target rank 0 at every λ and report an empty band.
        assert kwargs["target_ids"] == [TARGET_UUID]

    def test_full_uuid_target_resolves_unchanged(self, cli_db: Path) -> None:
        _seed_particle(TARGET_UUID)
        with patch(
            "particles.operations.utility_sweep.sweep_store_rank_lift", new_callable=AsyncMock
        ) as mock_sweep:
            mock_sweep.return_value = sweep_rank_lift([], grid=(0.0,), head_sizes=[60])
            result = runner.invoke(app, ["memory", "sweep-rank-lift", "--target", TARGET_UUID])
        assert result.exit_code == 0, result.output
        assert mock_sweep.await_args.kwargs["target_ids"] == [TARGET_UUID]


class TestSweepRankLiftTargetResolution:
    """An unresolvable ``--target`` is a hard error, never a silent rank-0.

    The regression this class pins: ``HeadOutcome.admissible`` requires
    ``0 < rank <= head_size``, so a target the sweep cannot find fails at every
    ``λ`` on the grid and renders as an empty admissible band on every surface
    — a typo made indistinguishable from a store that cannot be calibrated.
    """

    def test_unresolvable_target_exits_2_without_sweeping(self, cli_db: Path) -> None:
        _seed_particle(TARGET_UUID)
        with patch(
            "particles.operations.utility_sweep.sweep_store_rank_lift", new_callable=AsyncMock
        ) as mock_sweep:
            result = runner.invoke(app, ["memory", "sweep-rank-lift", "--target", "p-deadbeef"])
        assert result.exit_code == 2
        assert "matches no belief" in result.output
        assert "no `λ`" not in result.output  # never the misleading empty-band report
        mock_sweep.assert_not_awaited()

    def test_unresolvable_target_on_empty_store_exits_2(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["memory", "sweep-rank-lift", "--target", "p-4fbcc320"])
        assert result.exit_code == 2
        assert "matches no belief" in result.output
        assert "nothing to calibrate" not in result.output

    def test_ambiguous_prefix_exits_2_and_lists_candidates(self, cli_db: Path) -> None:
        _seed_particle("abcd1234-0000-4000-8000-000000000001", content="one")
        _seed_particle("abcd1234-0000-4000-8000-000000000002", content="two")
        result = runner.invoke(app, ["memory", "sweep-rank-lift", "--target", "abcd1234"])
        assert result.exit_code == 2
        assert "ambiguous" in result.output
        assert "abcd1234-0000-4000-8000-000000000001" in result.output
        assert "abcd1234-0000-4000-8000-000000000002" in result.output

    def test_non_active_target_exits_2_naming_the_status(self, cli_db: Path) -> None:
        # It exists, so "not found" would misdirect — the sweep ranks ACTIVE
        # beliefs only, and a retracted one can never reach the head.
        _seed_particle(TARGET_UUID, status=Status.RETRACTED)
        result = runner.invoke(app, ["memory", "sweep-rank-lift", "--target", "p-4fbcc320"])
        assert result.exit_code == 2
        assert "no ACTIVE belief" in result.output
        assert "RETRACTED" in result.output

    def test_one_bad_target_rejects_the_whole_sweep(self, cli_db: Path) -> None:
        _seed_particle(TARGET_UUID)
        with patch(
            "particles.operations.utility_sweep.sweep_store_rank_lift", new_callable=AsyncMock
        ) as mock_sweep:
            result = runner.invoke(
                app,
                [
                    "memory",
                    "sweep-rank-lift",
                    "--target",
                    "p-4fbcc320",
                    "--target",
                    "p-deadbeef",
                ],
            )
        assert result.exit_code == 2
        assert "p-deadbeef" in result.output
        mock_sweep.assert_not_awaited()

    def test_head_defaults_to_the_digest_cap(self, cli_db: Path) -> None:
        with patch(
            "particles.operations.utility_sweep.sweep_store_rank_lift", new_callable=AsyncMock
        ) as mock_sweep:
            mock_sweep.return_value = sweep_rank_lift([], grid=(0.0,), head_sizes=[200])
            result = runner.invoke(app, ["memory", "sweep-rank-lift"])
        assert result.exit_code == 0
        assert mock_sweep.await_args.kwargs["head_sizes"] == [
            get_config().mcp.recall.digest_max_beliefs
        ]
