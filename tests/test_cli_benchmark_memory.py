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

    def test_a_replay_draws_from_the_prepared_questions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The stratified sampler does not nest, so a narrower replay drawn from
        the whole variant lands outside the kept set and can never run."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        fixture_ids = [q["question_id"] for q in json.loads(FIXTURE.read_text())]
        assert len(fixture_ids) > 2
        (tmp_path / "stores-manifest.json").write_text(
            json.dumps({"format": 1, "write_side": {}, "question_ids": fixture_ids[:2]})
        )
        base = ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--estimate"]
        reused = runner.invoke(app, [*base, "--reuse-stores", "--store-dir", str(tmp_path)])
        assert reused.exit_code == 0, reused.output
        assert "Estimate: 2 question(s)" in reused.output

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


# ---------------------------------------------------------------------------
# benchmark memory rejudge
# ---------------------------------------------------------------------------


def _answered_report(*, answers: bool = True) -> MemoryBenchmarkReport:
    """A saved report whose qa_particles rows carry (or lack) stored answers."""
    from particles.benchmark.memory.schema import QaConditionMetrics, QaQuestionResult

    rows = [
        QaQuestionResult(
            question_id=f"q{i}",
            question_type="single-session-user",
            correct=False,
            answer="an answer" if answers else None,
            verdict="no" if answers else None,
        )
        for i in range(3)
    ]
    return MemoryBenchmarkReport(
        selection=RunSelection(
            dataset_revision="rev-test",
            variant="oracle",
            sample_seed=13,
            question_limit=3,
            questions_selected=3,
            questions_total=3,
            answer_model_id="anthropic:answer",
            judge_model_id="anthropic:old-judge",
            judge_protocol=1,
        ),
        qa_particles=QaConditionMetrics(
            condition="qa_particles",
            model_id="anthropic:answer",
            questions=3,
            accuracy=0.0,
            per_question=rows,
        ),
    )


class TestRejudge:
    """The verb is thin over ``rejudge_report``: load, gate, render, write."""

    def _write(self, tmp_path: Path, report: MemoryBenchmarkReport) -> Path:
        src = tmp_path / "src.json"
        src.write_text(report.model_dump_json(indent=2))
        return src

    def _invoke(self, tmp_path: Path, *extra: str) -> tuple[int, str, AsyncMock]:
        src = self._write(tmp_path, _answered_report())
        rejudged = _answered_report()
        rejudged.selection.judge_protocol = 2
        rejudged.selection.judge_model_id = "anthropic:new-judge"
        rejudged.quality_notes = ["Re-judged from src.json: …"]
        with patch(
            "particles.benchmark.memory.rejudge_report",
            new_callable=AsyncMock,
            return_value=rejudged,
        ) as mock:
            result = runner.invoke(
                app,
                [
                    "benchmark",
                    "memory",
                    "rejudge",
                    str(src),
                    "--output",
                    str(tmp_path / "out" / "new.json"),
                    "--dataset-file",
                    str(FIXTURE),
                    *extra,
                ],
            )
        return result.exit_code, result.output, mock

    def test_bare_memory_verb_still_runs_the_benchmark(self) -> None:
        """Turning ``memory`` into a group must not break its bare form."""
        with patch(
            "particles.benchmark.memory.run_memory_benchmark", new_callable=AsyncMock
        ) as run_mock:
            result = runner.invoke(
                app, ["benchmark", "memory", "--dataset-file", str(FIXTURE), "--estimate"]
            )
        assert result.exit_code == 0, result.output
        assert "--estimate: nothing was run." in result.output
        run_mock.assert_not_called()

    def test_writes_the_json_report_and_prints_the_table(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        code, output, mock = self._invoke(tmp_path, "--yes")
        assert code == 0, output
        # The source, its dataset, and the path for the provenance note reach the harness.
        source, questions = mock.await_args.args
        assert isinstance(source, MemoryBenchmarkReport)
        assert source.selection.judge_model_id == "anthropic:old-judge"
        assert len(questions) == 3  # the fixture's questions were loaded
        assert mock.await_args.kwargs["source_path"].endswith("src.json")
        # stdout: the estimate line, then the table; the file: always JSON.
        assert "3 judge call(s) over stored answers, no answer call" in output
        assert "== End-to-end QA" in output
        assert "judge protocol v2" in output
        written = json.loads((tmp_path / "out" / "new.json").read_text())
        assert written["selection"]["judge_protocol"] == 2
        assert written["selection"]["judge_model_id"] == "anthropic:new-judge"
        assert written["quality_notes"][0].startswith("Re-judged from")
        assert "Re-judged report written to" in output

    def test_format_json_prints_the_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        code, output, _ = self._invoke(tmp_path, "--yes", "--format", "json")
        assert code == 0, output
        payload = json.loads(output[output.index("{") : output.rindex("}") + 1])
        assert payload["selection"]["judge_protocol"] == 2

    def test_missing_report_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "benchmark",
                "memory",
                "rejudge",
                str(tmp_path / "absent.json"),
                "--output",
                str(tmp_path / "new.json"),
            ],
        )
        assert result.exit_code == 1
        assert "report not found" in result.output

    def test_not_a_report_errors(self, tmp_path: Path) -> None:
        src = tmp_path / "src.json"
        src.write_text('{"hello": "world"}')
        result = runner.invoke(
            app,
            ["benchmark", "memory", "rejudge", str(src), "--output", str(tmp_path / "new.json")],
        )
        assert result.exit_code == 1
        assert "not a memory-benchmark JSON report" in result.output

    def test_no_stored_answers_refuses_before_loading_the_dataset(self, tmp_path: Path) -> None:
        src = self._write(tmp_path, _answered_report(answers=False))
        with patch("particles.benchmark.memory.rejudge_report", new_callable=AsyncMock) as mock:
            result = runner.invoke(
                app,
                [
                    "benchmark",
                    "memory",
                    "rejudge",
                    str(src),
                    "--output",
                    str(tmp_path / "new.json"),
                    "--dataset-file",
                    str(tmp_path / "never-read.json"),
                ],
            )
        assert result.exit_code == 1
        assert "no stored answers" in result.output
        mock.assert_not_called()

    def test_non_interactive_over_threshold_aborts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from particles.config import get_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(get_config().benchmark_memory, "confirm_call_threshold", 1)
        code, output, mock = self._invoke(tmp_path)
        assert code == 1
        assert "exceed benchmark_memory.confirm_call_threshold" in output
        mock.assert_not_called()

    def test_refuses_without_key_before_running(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        code, output, mock = self._invoke(tmp_path, "--yes")
        assert code == 1
        assert "ANTHROPIC_API_KEY" in output
        assert "benchmark" in output  # names the one purpose a re-judge needs
        mock.assert_not_called()

    def test_harness_refusal_is_an_error_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from particles.benchmark.memory import RejudgeError

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        src = self._write(tmp_path, _answered_report())
        with patch(
            "particles.benchmark.memory.rejudge_report",
            new_callable=AsyncMock,
            side_effect=RejudgeError("The dataset does not contain 3 question(s)"),
        ):
            result = runner.invoke(
                app,
                [
                    "benchmark",
                    "memory",
                    "rejudge",
                    str(src),
                    "--output",
                    str(tmp_path / "new.json"),
                    "--dataset-file",
                    str(FIXTURE),
                    "--yes",
                ],
            )
        assert result.exit_code == 1
        assert "does not contain" in result.output
        assert not (tmp_path / "new.json").exists()
