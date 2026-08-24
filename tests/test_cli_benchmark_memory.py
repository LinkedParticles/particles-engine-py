# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``particles benchmark memory`` verb.

Pins the CLI contract with mocked seams: flag validation, the ``--estimate``
no-run exit, the confirm gate (non-interactive abort / ``--yes``), the no-key
refusal, and the table/json output paths. The harness itself is unit-tested
in ``tests/test_benchmark_memory.py``; the live end-to-end fixture run is
``tests/test_integration_memory_benchmark.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.memory.schema import MemoryBenchmarkReport, RunSelection

runner = CliRunner()

FIXTURE = (
    Path(__file__).parent
    / "benchmark"
    / "memory"
    / "fixtures"
    / "longmemeval_oracle_synthetic.json"
)


def _fake_report() -> MemoryBenchmarkReport:
    return MemoryBenchmarkReport(
        selection=RunSelection(
            dataset_revision="rev-test",
            variant="oracle",
            sample_seed=13,
            question_limit=2,
            questions_selected=2,
            questions_total=3,
            answer_model_id="anthropic:answer",
            judge_model_id="anthropic:judge",
        )
    )


class TestFlagValidation:
    def test_limit_and_all_are_mutually_exclusive(self) -> None:
        result = runner.invoke(
            app, ["benchmark", "memory", "--limit", "2", "--all", "--dataset-file", str(FIXTURE)]
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    def test_unknown_variant_rejected(self) -> None:
        result = runner.invoke(app, ["benchmark", "memory", "--variant", "xl"])
        assert result.exit_code == 2

    def test_missing_dataset_file_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["benchmark", "memory", "--dataset-file", str(tmp_path / "absent.json"), "--estimate"],
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_types_filter_with_no_match_errors(self) -> None:
        result = runner.invoke(
            app,
            [
                "benchmark",
                "memory",
                "--dataset-file",
                str(FIXTURE),
                "--types",
                "temporal-reasoning",
                "--estimate",
            ],
        )
        assert result.exit_code == 1
        assert "No questions matched" in result.output


class TestEstimateGate:
    def test_estimate_prints_and_runs_nothing(self) -> None:
        with patch(
            "particles.benchmark.memory.run_memory_benchmark", new_callable=AsyncMock
        ) as run_mock:
            result = runner.invoke(
                app,
                ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--estimate"],
            )
        assert result.exit_code == 0, result.output
        assert "Estimate:" in result.output
        assert "nothing was run" in result.output
        run_mock.assert_not_called()

    def test_estimate_states_the_context_window_verdict(self) -> None:
        """``--estimate`` is a complete dry run: cost *and* whether it fits.

        An operator trying a bigger variant learns it will not fit here,
        before the confirm gate — not from the runner's refusal later.
        """
        with patch(
            "particles.benchmark.memory.run_memory_benchmark", new_callable=AsyncMock
        ) as run_mock:
            result = runner.invoke(
                app,
                ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--estimate"],
            )
        assert result.exit_code == 0, result.output
        assert "Context window" in result.output
        assert "fits" in result.output
        run_mock.assert_not_called()

    def test_an_over_window_run_is_refused_with_a_nonzero_exit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The runner's refusal reaches the operator as an error, not a traceback."""
        from particles.benchmark.memory import ContextWindowExceeded

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch(
            "particles.benchmark.memory.run_memory_benchmark",
            new_callable=AsyncMock,
            side_effect=ContextWindowExceeded("does not fit on the 'm' variant"),
        ):
            result = runner.invoke(
                app, ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--yes"]
            )
        assert result.exit_code == 1
        assert "does not fit on the 'm' variant" in result.output

    def test_non_interactive_over_threshold_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("BENCHMARK_MEMORY_CONFIRM_CALL_THRESHOLD", "0")
        reset_config()
        with patch(
            "particles.benchmark.memory.run_memory_benchmark", new_callable=AsyncMock
        ) as run_mock:
            result = runner.invoke(app, ["benchmark", "memory", "--dataset-file", str(FIXTURE)])
        assert result.exit_code == 1
        assert "Estimate:" in result.output  # always printed first
        assert "confirm_call_threshold" in result.output
        assert "--yes" in result.output
        run_mock.assert_not_called()

    def test_yes_pre_confirms_over_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("BENCHMARK_MEMORY_CONFIRM_CALL_THRESHOLD", "0")
        reset_config()
        with patch(
            "particles.benchmark.memory.run_memory_benchmark",
            new_callable=AsyncMock,
            return_value=_fake_report(),
        ) as run_mock:
            result = runner.invoke(
                app, ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--yes"]
            )
        assert result.exit_code == 0, result.output
        run_mock.assert_awaited_once()


class TestBatchQAFlag:
    def test_batch_qa_flag_threads_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch(
            "particles.benchmark.memory.run_memory_benchmark",
            new_callable=AsyncMock,
            return_value=_fake_report(),
        ) as run_mock:
            result = runner.invoke(
                app,
                ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--yes", "--batch-qa"],
            )
        assert result.exit_code == 0, result.output
        assert run_mock.await_args is not None
        assert run_mock.await_args.kwargs["batch_qa"] is True

    def test_batch_qa_defaults_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch(
            "particles.benchmark.memory.run_memory_benchmark",
            new_callable=AsyncMock,
            return_value=_fake_report(),
        ) as run_mock:
            result = runner.invoke(
                app, ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--yes"]
            )
        assert result.exit_code == 0, result.output
        assert run_mock.await_args is not None
        assert run_mock.await_args.kwargs["batch_qa"] is False


class TestNoKeyRefusal:
    def test_refuses_without_key_before_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch(
            "particles.benchmark.memory.run_memory_benchmark", new_callable=AsyncMock
        ) as run_mock:
            result = runner.invoke(
                app, ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--yes"]
            )
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output
        run_mock.assert_not_called()


class TestOutput:
    def _invoke(self, *extra: str) -> tuple[int, str]:
        with patch(
            "particles.benchmark.memory.run_memory_benchmark",
            new_callable=AsyncMock,
            return_value=_fake_report(),
        ):
            result = runner.invoke(
                app,
                ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--yes", *extra],
            )
        return result.exit_code, result.output

    def test_table_output_shows_both_families(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        code, output = self._invoke()
        assert code == 0, output
        assert "== Retrieval stage" in output
        assert "== End-to-end QA" in output
        assert "not run" in output  # baseline rows render even on an empty report

    def test_json_output_parses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        code, output = self._invoke("--format", "json")
        assert code == 0, output
        payload = json.loads(output[output.index("{") :])
        assert payload["selection"]["dataset_revision"] == "rev-test"
        assert payload["qa_full_context"] is None

    def test_output_file_written(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        out = tmp_path / "report.txt"
        code, output = self._invoke("--output", str(out))
        assert code == 0, output
        assert out.exists()
        assert "== Retrieval stage" in out.read_text()


class TestPdr0488Flags:
    """The five memory-benchmark ablations, as CLI knobs."""

    def test_estimate_drops_the_write_side_under_reuse(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The projection an operator confirms must describe the run they get."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        base = ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--estimate"]
        fresh = runner.invoke(app, base)
        reused = runner.invoke(app, [*base, "--reuse-stores", "--store-dir", str(tmp_path)])
        assert fresh.exit_code == 0, fresh.output
        assert reused.exit_code == 0, reused.output
        assert "~0 write-time" in reused.output
        assert "~0 write-time" not in fresh.output

    def test_consolidation_and_abstraction_are_mutually_exclusive(self) -> None:
        """The cycle runs the abstraction pass itself — accepting both would
        run it twice and record a tuple claiming two knobs for one arm."""
        result = runner.invoke(
            app,
            [
                "benchmark",
                "memory",
                "--dataset-file",
                str(FIXTURE),
                "--consolidation",
                "--abstraction",
            ],
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    def test_reuse_stores_requires_store_dir(self) -> None:
        result = runner.invoke(
            app, ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--reuse-stores"]
        )
        assert result.exit_code == 2
        assert "--store-dir" in result.output

    @pytest.mark.parametrize(
        ("flag", "kwarg"),
        [
            ("--consolidation", "consolidation"),
            ("--dedup-judge", "dedup_judge"),
            ("--no-qa", "qa"),
        ],
    )
    def test_flag_reaches_the_runner(
        self, monkeypatch: pytest.MonkeyPatch, flag: str, kwarg: str
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        run_mock = AsyncMock(return_value=_fake_report())
        with patch("particles.benchmark.memory.run_memory_benchmark", run_mock):
            result = runner.invoke(
                app,
                ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--yes", flag],
            )
        assert result.exit_code == 0, result.output
        expected = flag != "--no-qa"
        assert run_mock.await_args.kwargs[kwarg] is expected

    def test_top_k_reaches_the_runner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        run_mock = AsyncMock(return_value=_fake_report())
        with patch("particles.benchmark.memory.run_memory_benchmark", run_mock):
            result = runner.invoke(
                app,
                [
                    "benchmark",
                    "memory",
                    "--dataset-file",
                    str(FIXTURE),
                    "--yes",
                    "--top-k",
                    "25",
                ],
            )
        assert result.exit_code == 0, result.output
        assert run_mock.await_args.kwargs["top_k"] == 25

    def test_unbounded_component_still_gates(self, tmp_path: Path) -> None:
        """A projection of "~0 calls" must not be a way around the confirm gate.

        --dedup-judge over reused stores has zero *boundable* calls, so the
        threshold comparison alone would wave it through while the judge fans
        out per Subject cluster.
        """
        result = runner.invoke(
            app,
            [
                "benchmark",
                "memory",
                "--dataset-file",
                str(FIXTURE),
                "--reuse-stores",
                "--store-dir",
                str(tmp_path),
                "--no-qa",
                "--dedup-judge",
            ],
        )
        assert result.exit_code == 1
        assert "NOT INCLUDED ABOVE" in result.output
        assert "no --yes" in result.output

    def test_consolidation_bound_appears_in_the_estimate(self) -> None:
        """The two per-store probe caps are the arm's ceiling — show them."""
        result = runner.invoke(
            app,
            [
                "benchmark",
                "memory",
                "--dataset-file",
                str(FIXTURE),
                "--estimate",
                "--consolidation",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ablation-pass probe(s), capped" in result.output
