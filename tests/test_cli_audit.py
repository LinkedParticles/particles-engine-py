"""Tests for the ``particles audit`` verb + the init hand-off (§1/§4/§7).

Pins the CLI contract with mocked seams: the no-key refusal (before anything
touches the store), the ``--estimate`` no-deposit exit, the confirm gate
(``--yes`` / non-interactive abort), the harvest plan (sentinel filter,
transcript cap), the re-audit degradation kwargs, the projection-cycle tail,
and the ``init claude-code`` hand-off wiring. The live end-to-end fixture run
is ``tests/test_integration_audit.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import typer
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli import audit as audit_mod
from particles.operations.audit import AuditReport

runner = CliRunner()


@pytest.fixture
def mem_dir(tmp_path: Path) -> Path:
    d = tmp_path / "memory"
    d.mkdir()
    (d / "MEMORY.md").write_text("# Memory index\n\n- The sky is blue.\n")
    (d / "topic.md").write_text("# Topic\n\n- Water is wet.\n")
    return d


def _fake_report(**kwargs: object) -> AuditReport:
    defaults: dict[str, object] = {"files_audited": 2, "beliefs": 2, "subjects": 1}
    defaults.update(kwargs)
    return AuditReport(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# No-key refusal (§7) — before the store is touched
# ---------------------------------------------------------------------------


class TestNoKeyRefusal:
    def test_harvest_without_key_refuses_and_deposits_nothing(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("particles.operations.deposit.deposit_text_versioned") as deposit:
            result = runner.invoke(app, ["audit", str(mem_dir)])
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output
        assert "export ANTHROPIC_API_KEY" in result.output  # the one-line fix
        deposit.assert_not_called()

    def test_reaudit_without_key_degrades_gracefully(
        self, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        audit = AsyncMock(return_value=_fake_report(files_audited=None))
        with patch("particles.operations.audit.run_memory_audit", audit):
            result = runner.invoke(app, ["audit"])
        assert result.exit_code == 0, result.output
        assert audit.call_args.kwargs["semantic"] is False
        assert audit.call_args.kwargs["semantic_skip_reason"] == "no API key"

    def test_reaudit_with_key_runs_semantic(
        self, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        audit = AsyncMock(return_value=_fake_report(files_audited=None))
        with patch("particles.operations.audit.run_memory_audit", audit):
            result = runner.invoke(app, ["audit"])
        assert result.exit_code == 0, result.output
        assert audit.call_args.kwargs["semantic"] is True


# ---------------------------------------------------------------------------
# Estimate + confirm gate (§4)
# ---------------------------------------------------------------------------


class TestEstimateGate:
    def test_estimate_prints_and_deposits_nothing(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch("particles.operations.deposit.deposit_text_versioned") as deposit:
            result = runner.invoke(app, ["audit", str(mem_dir), "--estimate"])
        assert result.exit_code == 0, result.output
        assert "Estimate:" in result.output
        assert "nothing was deposited" in result.output
        deposit.assert_not_called()

    def test_non_interactive_over_threshold_aborts_with_estimate(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("AUDIT_CONFIRM_CALL_THRESHOLD", "0")
        reset_config()  # cli_db already cached a config without the override
        with patch("particles.operations.deposit.deposit_text_versioned") as deposit:
            result = runner.invoke(app, ["audit", str(mem_dir)])
        assert result.exit_code == 1
        assert "Estimate:" in result.output  # always printed before extraction
        assert "confirm_call_threshold" in result.output
        assert "--yes" in result.output
        deposit.assert_not_called()

    def test_yes_pre_confirms_over_threshold(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("AUDIT_CONFIRM_CALL_THRESHOLD", "0")
        reset_config()  # cli_db already cached a config without the override
        deposit = AsyncMock(return_value=("e1", "s1", False))
        audit = AsyncMock(return_value=_fake_report())
        with (
            patch("particles.operations.deposit.deposit_text_versioned", deposit),
            patch("particles.operations.audit.run_memory_audit", audit),
            patch.object(audit_mod, "projection_enabled", lambda: False),
        ):
            result = runner.invoke(app, ["audit", str(mem_dir), "--yes"])
        assert result.exit_code == 0, result.output
        assert deposit.await_count == 2  # both memory files
        assert audit.call_args.kwargs["harvested_entry_ids"] == ["e1", "e1"]
        assert audit.call_args.kwargs["files_audited"] == 2
        assert audit.call_args.kwargs["semantic"] is True
        assert "Audited 2 memory files" in result.output

    def test_estimate_without_path_is_a_noop(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["audit", "--estimate"])
        assert result.exit_code == 0
        assert "deposits and\nextracts nothing" in result.output or (
            "extracts nothing" in result.output
        )


# ---------------------------------------------------------------------------
# Flags + output
# ---------------------------------------------------------------------------


class TestFlags:
    def test_unknown_format_rejected(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["audit", "--format", "yaml"])
        assert result.exit_code == 1
        assert "--format" in result.output

    def test_json_format_dumps_the_model(
        self, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        audit = AsyncMock(return_value=_fake_report(files_audited=None))
        with patch("particles.operations.audit.run_memory_audit", audit):
            result = runner.invoke(app, ["audit", "--format", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["beliefs"] == 2

    def test_output_writes_markdown_report(
        self, cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        out = tmp_path / "report.md"
        audit = AsyncMock(return_value=_fake_report(files_audited=None, store="default"))
        with patch("particles.operations.audit.run_memory_audit", audit):
            result = runner.invoke(app, ["audit", "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert out.is_file()
        assert "Re-audited store 'default'" in out.read_text()

    def test_missing_path_errors(self, cli_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        result = runner.invoke(app, ["audit", "/nonexistent/memory-dir"])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_judge_flag_reaches_the_operation(
        self, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        audit = AsyncMock(return_value=_fake_report(files_audited=None))
        with patch("particles.operations.audit.run_memory_audit", audit):
            result = runner.invoke(app, ["audit", "--judge"])
        assert result.exit_code == 0, result.output
        assert audit.call_args.kwargs["judge"] is True

    def test_unknown_scope_rejected(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["audit", "--scope", "everything"])
        assert result.exit_code == 1
        assert "--scope" in result.output

    def test_harvested_scope_without_harvest_rejected(self, cli_db: Path) -> None:
        # A re-audit has no harvested entries to scope to.
        result = runner.invoke(app, ["audit", "--scope", "harvested"])
        assert result.exit_code == 1
        assert "needs a harvest" in result.output

    def test_reaudit_probe_is_store_wide(
        self, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        audit = AsyncMock(return_value=_fake_report(files_audited=None))
        with patch("particles.operations.audit.run_memory_audit", audit):
            result = runner.invoke(app, ["audit"])
        assert result.exit_code == 0, result.output
        assert audit.call_args.kwargs["contradiction_scope"] == "store"

    def test_harvest_defaults_to_harvested_scope(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        deposit = AsyncMock(return_value=("e1", "s1", False))
        audit = AsyncMock(return_value=_fake_report())
        with (
            patch("particles.operations.deposit.deposit_text_versioned", deposit),
            patch("particles.operations.audit.run_memory_audit", audit),
            patch.object(audit_mod, "projection_enabled", lambda: False),
        ):
            result = runner.invoke(app, ["audit", str(mem_dir)])
        assert result.exit_code == 0, result.output
        assert audit.call_args.kwargs["contradiction_scope"] == "harvested"

    def test_scope_store_opts_into_store_wide_on_harvest(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        deposit = AsyncMock(return_value=("e1", "s1", False))
        audit = AsyncMock(return_value=_fake_report())
        with (
            patch("particles.operations.deposit.deposit_text_versioned", deposit),
            patch("particles.operations.audit.run_memory_audit", audit),
            patch.object(audit_mod, "projection_enabled", lambda: False),
        ):
            result = runner.invoke(app, ["audit", str(mem_dir), "--scope", "store"])
        assert result.exit_code == 0, result.output
        assert audit.call_args.kwargs["contradiction_scope"] == "store"


# ---------------------------------------------------------------------------
# Harvest plan (reuses the helpers)
# ---------------------------------------------------------------------------


class TestHarvestPlan:
    def test_memory_dir_plan_shapes_deposits_like_the_hook(self, mem_dir: Path) -> None:
        plan = audit_mod.build_harvest_plan([mem_dir], None, None)
        assert plan.memory_files == 2
        assert plan.transcripts == 0
        uris = {d.uri_r for d in plan.deposits}
        assert (mem_dir / "MEMORY.md").resolve().as_uri() in uris
        assert all(d.source_type == "LOCAL_MARKDOWN" for d in plan.deposits)
        assert all(d.mutability == "MUTABLE" for d in plan.deposits)
        # The raw MEMORY.md text is captured for the projection cycle's
        # changed-since-harvest refusal.
        assert plan.memory_dirs == [(mem_dir, (mem_dir / "MEMORY.md").read_text())]

    def test_pristine_projected_region_is_stripped(self, tmp_path: Path) -> None:
        from particles.render.markdown import PROJECTED_BEGIN_TMPL, PROJECTED_END_TMPL

        d = tmp_path / "memory"
        d.mkdir()
        begin = PROJECTED_BEGIN_TMPL.format(region="memory-index", manifest="m.yaml")
        end = PROJECTED_END_TMPL.format(region="memory-index")
        (d / "MEMORY.md").write_text(f"{begin}\nrendered line\n{end}\n\n- Authored fact.\n")
        with patch.object(
            audit_mod,
            "filter_memory_file_for_deposit",
            wraps=audit_mod.filter_memory_file_for_deposit,
        ) as filt:
            plan = audit_mod.build_harvest_plan([d], None, None)
        filt.assert_called()
        (deposit,) = plan.deposits
        assert "Authored fact." in deposit.text

    def test_transcripts_capped_newest_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tdir = tmp_path / "project"
        tdir.mkdir()
        line = json.dumps({"type": "user", "message": {"content": "hello there"}})
        import os

        for i, name in enumerate(["old", "mid", "new"]):
            p = tdir / f"{name}.jsonl"
            p.write_text(line + "\n")
            os.utime(p, (1_000_000 + i, 1_000_000 + i))
        plan = audit_mod.build_harvest_plan([], tdir, max_entries=2)
        assert plan.transcripts == 2
        uris = [d.uri_r for d in plan.deposits]
        assert uris == ["claude-code://session/new", "claude-code://session/mid"]
        assert all(d.source_type == "CONVERSATION" for d in plan.deposits)
        assert all(d.mutability == "APPEND_ONLY" for d in plan.deposits)

    def test_leading_date_line_becomes_content_date(self, tmp_path: Path) -> None:
        # The ladder: a leading date line beats the file mtime, so the
        # age-discount lens sees the memory's real age.
        f = tmp_path / "dated.md"
        f.write_text("# Notes\n\n2024-03-02\n\n- An old fact.\n")
        plan = audit_mod.build_harvest_plan([f], None, None)
        published = plan.deposits[0].content_published_at
        assert published is not None
        assert (published.year, published.month, published.day) == (2024, 3, 2)

    def test_max_entries_caps_memory_files(self, mem_dir: Path) -> None:
        plan = audit_mod.build_harvest_plan([mem_dir], None, max_entries=1)
        assert plan.memory_files == 1

    def test_single_jsonl_file_is_a_transcript(self, tmp_path: Path) -> None:
        p = tmp_path / "abc123.jsonl"
        p.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
        plan = audit_mod.build_harvest_plan([p], None, None)
        assert plan.transcripts == 1
        assert plan.deposits[0].uri_r == "claude-code://session/abc123"

    def test_project_tag_derived_from_claude_layout(self, tmp_path: Path) -> None:
        d = tmp_path / "projects" / "-Users-x-repo" / "memory"
        d.mkdir(parents=True)
        (d / "MEMORY.md").write_text("- A fact.\n")
        plan = audit_mod.build_harvest_plan([d], None, None)
        assert "project:-Users-x-repo" in plan.deposits[0].tags


# ---------------------------------------------------------------------------
# The projection tail (render after successful harvest+extract)
# ---------------------------------------------------------------------------


class TestProjectionTail:
    @pytest.mark.asyncio
    async def test_cycle_runs_per_harvested_memory_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mem = tmp_path / "memory"
        plan = audit_mod._HarvestPlan(memory_dirs=[(mem, "raw text")])
        monkeypatch.setattr(audit_mod, "projection_enabled", lambda: True)
        cycle = AsyncMock(return_value={"outcome": "rendered"})
        with patch("particles.api.cli._memory_projection.run_projection_cycle", cycle):
            rendered = await audit_mod._run_projection_cycles("memory", plan)
        assert rendered is True
        cycle.assert_awaited_once_with("memory", mem, "raw text")

    @pytest.mark.asyncio
    async def test_disabled_projection_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = audit_mod._HarvestPlan(memory_dirs=[(tmp_path / "memory", "raw")])
        monkeypatch.setattr(audit_mod, "projection_enabled", lambda: False)
        with patch("particles.api.cli._memory_projection.run_projection_cycle") as cycle:
            assert await audit_mod._run_projection_cycles("memory", plan) is False
        cycle.assert_not_called()

    @pytest.mark.asyncio
    async def test_never_mints_memory_md_into_arbitrary_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = audit_mod._HarvestPlan(memory_dirs=[(tmp_path / "notes", None)])
        monkeypatch.setattr(audit_mod, "projection_enabled", lambda: True)
        with patch("particles.api.cli._memory_projection.run_projection_cycle") as cycle:
            assert await audit_mod._run_projection_cycles("memory", plan) is False
        cycle.assert_not_called()

    def test_full_flow_ends_with_projection_and_closing_line(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        deposit = AsyncMock(return_value=("e1", "s1", False))
        audit = AsyncMock(return_value=_fake_report())
        cycle = AsyncMock(return_value={"outcome": "rendered"})
        monkeypatch.setattr(audit_mod, "projection_enabled", lambda: True)
        with (
            patch("particles.operations.deposit.deposit_text_versioned", deposit),
            patch("particles.operations.audit.run_memory_audit", audit),
            patch("particles.api.cli._memory_projection.run_projection_cycle", cycle),
        ):
            result = runner.invoke(app, ["audit", str(mem_dir), "--yes"])
        assert result.exit_code == 0, result.output
        cycle.assert_awaited_once()  # the render-after-successful-harvest tail
        assert "MEMORY.md was re-projected from the audited store" in result.output


# ---------------------------------------------------------------------------
# init claude-code hand-off (seam, filled)
# ---------------------------------------------------------------------------


class TestInitHandOff:
    def test_offer_first_run_audit_calls_the_flow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.api.cli import init as init_mod

        called: dict[str, str] = {}
        monkeypatch.setattr(
            "particles.api.cli.audit.run_first_run_audit",
            lambda store: called.setdefault("store", store),
        )
        init_mod._offer_first_run_audit("memory")
        assert called["store"] == "memory"

    def test_offer_first_run_audit_survives_refusal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from particles.api.cli import init as init_mod

        def _refuse(store: str) -> None:
            raise typer.Exit(1)

        monkeypatch.setattr("particles.api.cli.audit.run_first_run_audit", _refuse)
        init_mod._offer_first_run_audit("memory")  # must not raise — init succeeded
        assert "First-run audit skipped" in capsys.readouterr().out

    def test_offer_first_run_audit_survives_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from particles.api.cli import init as init_mod

        def _boom(store: str) -> None:
            raise RuntimeError("kaput")

        monkeypatch.setattr("particles.api.cli.audit.run_first_run_audit", _boom)
        init_mod._offer_first_run_audit("memory")
        assert "First-run audit failed" in capsys.readouterr().out

    def test_no_memory_dirs_prints_pointer(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        audit_mod.run_first_run_audit("memory")
        out = capsys.readouterr().out
        assert "no memory directories found" in out

    def test_first_run_audit_audits_discovered_dirs(
        self,
        cli_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        home = tmp_path / "home"
        mem = home / ".claude" / "projects" / "-Users-x-repo" / "memory"
        mem.mkdir(parents=True)
        (mem / "MEMORY.md").write_text("- A fact worth keeping.\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        deposit = AsyncMock(return_value=("e1", "s1", False))
        audit = AsyncMock(return_value=_fake_report(files_audited=1))
        monkeypatch.setattr(audit_mod, "projection_enabled", lambda: False)
        with (
            patch("particles.operations.deposit.deposit_text_versioned", deposit),
            patch("particles.operations.audit.run_memory_audit", audit),
        ):
            audit_mod.run_first_run_audit("default")
        out = capsys.readouterr().out
        assert "First-run memory audit over 1 memory directory" in out
        assert audit.call_args.kwargs["harvested_entry_ids"] == ["e1"]


class TestProgressRenderer:
    """The `_make_progress_renderer` closure (owner-reported UX gap, 2026-07-11)."""

    def test_extract_census_and_failure_lines(self, capsys: pytest.CaptureFixture[str]) -> None:
        from particles.operations.audit import AuditProgress

        render = audit_mod._make_progress_renderer()
        render(AuditProgress(phase="extract", done=3, total=35, label="alpha.md", particles=9))
        render(AuditProgress(phase="extract", done=4, total=35, label="beta.md", particles=1))
        render(AuditProgress(phase="extract", done=5, total=35, label="gamma.md", failed=True))
        render(
            AuditProgress(
                phase="census", done=0, total=1, label="contradiction probe + duplicate scan"
            )
        )
        out = capsys.readouterr().out.splitlines()
        assert out[0].startswith("  [3/35] alpha.md → 9 beliefs (")
        assert out[1].startswith("  [4/35] beta.md → 1 belief (")
        assert "extraction failed" in out[2] and "[5/35] gamma.md" in out[2]
        assert out[3].startswith("  Scanning findings — contradiction probe + duplicate scan…")
        assert all("elapsed" in line for line in out)
