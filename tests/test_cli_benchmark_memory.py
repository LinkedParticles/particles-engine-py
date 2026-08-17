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
