# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from click.testing import Result
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli import audit as audit_mod
from particles.config import get_config
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
        assert result.exit_code == 2
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
        # A priced dollar range at the configured model's list price.
        assert re.search(
            r"Cost: ≈ \$\d+\.\d\d(–\d+\.\d\d)? expected, \$[\d.,]+ ceiling", result.output
        )
        assert "list price" in result.output
        assert "nothing was deposited" in result.output
        deposit.assert_not_called()

    def test_estimate_json_carries_the_priced_range(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch("particles.operations.deposit.deposit_text_versioned") as deposit:
            result = runner.invoke(app, ["audit", str(mem_dir), "--estimate", "--format", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["entries"] == 2
        assert payload["cost"]["ceiling_usd"] >= payload["cost"]["expected_high_usd"]
        extraction = get_config().extraction
        assert payload["output_tokens_ceiling"] == (
            payload["estimated_llm_calls"] * extraction.max_tokens
            + math.ceil(payload["expected_retries"]) * extraction.retry_max_tokens
        )
        assert payload["wall_seconds"] > 0
        deposit.assert_not_called()

    def test_estimate_for_an_unpriced_model_prints_tokens_and_the_fix(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        get_config().llm.price_per_mtok.clear()
        with patch("particles.operations.deposit.deposit_text_versioned") as deposit:
            result = runner.invoke(app, ["audit", str(mem_dir), "--estimate"])
        assert result.exit_code == 0, result.output
        assert "Cost: not priced" in result.output
        assert "llm.price_per_mtok" in result.output
        assert "Tokens: ~" in result.output
        deposit.assert_not_called()

    def test_confirmation_prompt_shows_the_dollar_range(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        from particles.config import reset_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("AUDIT_CONFIRM_CALL_THRESHOLD", "0")
        reset_config()  # cli_db already cached a config without the override
        # An interactive terminal, so the gate asks instead of aborting.
        monkeypatch.setattr(
            audit_mod, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True))
        )
        with patch("particles.operations.deposit.deposit_text_versioned") as deposit:
            result = runner.invoke(app, ["audit", str(mem_dir)], input="n\n")
        assert result.exit_code == 2
        assert re.search(
            r"Proceed with ~2 extraction LLM calls \(≈ \$[^)]*ceiling, about [^)]* per file\)\?",
            result.output,
        )
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
        assert result.exit_code == 2
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
        assert result.exit_code == 2
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
        assert result.exit_code == 2
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
        assert result.exit_code == 2
        assert "--scope" in result.output

    def test_harvested_scope_without_harvest_rejected(self, cli_db: Path) -> None:
        # A re-audit has no harvested entries to scope to.
        result = runner.invoke(app, ["audit", "--scope", "harvested"])
        assert result.exit_code == 2
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

    def test_project_tag_derived_from_claude_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = _claude_code_memory_dir(tmp_path, monkeypatch)
        plan = audit_mod.build_harvest_plan([d], None, None)
        assert all("project:-Users-x-repo" in dep.tags for dep in plan.deposits)

    def test_no_project_tag_outside_claude_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A directory named ``memory`` elsewhere (the rung-1 copy under
        ``/tmp/rung1``) is not tagged with its parent's name: the tag would
        collide across unrelated directories and decide observer scope."""
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        d = tmp_path / "rung1" / "memory"
        d.mkdir(parents=True)
        (d / "MEMORY.md").write_text("- A fact.\n")
        plan = audit_mod.build_harvest_plan([d], None, None)
        assert plan.deposits
        for dep in plan.deposits:
            assert not any(t.startswith("project:") for t in dep.tags)
            assert "memory-file" in dep.tags


# ---------------------------------------------------------------------------
# The projection tail (render after successful harvest+extract)
# ---------------------------------------------------------------------------


def _claude_code_memory_dir(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A memory dir in Claude Code's own layout, under a fake ``$HOME``."""
    monkeypatch.setenv("HOME", str(home))
    mem = home / ".claude" / "projects" / "-Users-x-repo" / "memory"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text("# Memory index\n\n- The sky is blue.\n")
    (mem / "topic.md").write_text("# Topic\n\n- Water is wet.\n")
    return mem


class TestProjectionTail:
    @pytest.mark.asyncio
    async def test_cycle_runs_per_harvested_memory_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mem = _claude_code_memory_dir(tmp_path, monkeypatch)
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

    @pytest.mark.asyncio
    async def test_never_projects_into_a_memory_dir_claude_code_does_not_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A copy named ``memory`` outside ~/.claude/projects (the 2026-09-25
        # rung-1 run audited /tmp/rung1/memory) is not a projection target.
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        copy = tmp_path / "rung1" / "memory"
        copy.mkdir(parents=True)
        plan = audit_mod._HarvestPlan(memory_dirs=[(copy, None), (copy, "- raw\n")])
        monkeypatch.setattr(audit_mod, "projection_enabled", lambda: True)
        with patch("particles.api.cli._memory_projection.run_projection_cycle") as cycle:
            assert await audit_mod._run_projection_cycles("memory", plan) is False
        cycle.assert_not_called()

    def test_audit_of_a_non_claude_code_dir_writes_nothing_into_it(
        self, cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole verb with the projection on and its real cycle unpatched:
        the audited directory's files are byte-identical afterwards, no
        MEMORY.md is minted, no per-project state is written, and the closing
        line never claims a re-projection."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        copy = tmp_path / "rung1" / "memory"
        copy.mkdir(parents=True)
        (copy / "topic.md").write_text("# Topic\n\n- Water is wet.\n")
        before = {p.name: p.read_bytes() for p in copy.iterdir()}
        monkeypatch.setattr(audit_mod, "projection_enabled", lambda: True)
        deposit = AsyncMock(return_value=("e1", "s1", False))
        audit = AsyncMock(return_value=_fake_report())
        # The render is the only store read in the cycle; everything after it
        # (the MEMORY.md write, the per-project snapshot) runs for real.
        body = AsyncMock(return_value="- Water is wet.\n")
        with (
            patch("particles.operations.deposit.deposit_text_versioned", deposit),
            patch("particles.operations.audit.run_memory_audit", audit),
            patch("particles.api.cli._memory_projection._render_region_body", body),
            patch(
                "particles.api.cli._memory_projection._maybe_commit", AsyncMock(return_value=None)
            ),
        ):
            result = runner.invoke(app, ["audit", str(copy), "--yes"])
        assert result.exit_code == 0, result.output
        assert {p.name: p.read_bytes() for p in copy.iterdir()} == before
        assert not (home / ".particles" / "claude-code" / "projects").exists()
        assert "re-projected" not in result.output

    def test_full_flow_ends_with_projection_and_closing_line(
        self, cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mem_dir = _claude_code_memory_dir(tmp_path, monkeypatch)
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
        assert "Cost: ≈ $" in out  # the init hand-off shares the priced estimate
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
        # Progress is narration: stderr, never the stdout report.
        captured = capsys.readouterr()
        assert captured.out == ""
        out = captured.err.splitlines()
        assert out[0].startswith("  [3/35] alpha.md → 9 beliefs (")
        assert out[1].startswith("  [4/35] beta.md → 1 belief (")
        assert "extraction failed" in out[2] and "[5/35] gamma.md" in out[2]
        assert out[3].startswith("  Scanning findings — contradiction probe + duplicate scan…")
        assert all("elapsed" in line for line in out)

    def test_probe_progress_is_bounded_off_a_terminal(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from particles.operations.audit import AuditProgress

        render = audit_mod._make_progress_renderer()
        for done in range(1, 201):
            render(AuditProgress(phase="probe", done=done, total=200, label="contradiction probe"))
        lines = capsys.readouterr().err.splitlines()
        # Every 10% plus the last, never one line per probe (2026-09-25 re-audit).
        assert len(lines) == 10
        assert lines[-1].startswith("  [200/200] contradiction probe…")

    def test_probe_progress_renders_in_place_on_a_terminal(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli._output import OutputSettings
        from particles.operations.audit import AuditProgress

        monkeypatch.setattr(audit_mod, "current_output", lambda: OutputSettings(progress=True))
        statuses: list[str | None] = []
        monkeypatch.setattr(audit_mod, "set_heartbeat_status", statuses.append)
        render = audit_mod._make_progress_renderer()
        for done in range(1, 4):
            render(AuditProgress(phase="probe", done=done, total=3, label="contradiction probe"))
        assert capsys.readouterr().err == ""
        # The last event clears the status so the heartbeat falls back to "working".
        assert statuses == ["contradiction probe 1/3", "contradiction probe 2/3", None]


# ---------------------------------------------------------------------------
# An audit the LLM could not finish (owner-reported, 2026-09-25)
# ---------------------------------------------------------------------------


class _CreditBalanceError(Exception):
    """The Anthropic SDK's out-of-credit 400: a status code plus the billing text."""

    status_code = 400

    def __init__(self) -> None:
        super().__init__(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'Your credit balance is too low to access the Anthropic API.'}}"
        )


def _seed_similar_beliefs(count: int) -> None:
    """``count`` mutually-similar beliefs: count·(count-1)/2 contradiction candidates."""
    import asyncio

    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    async def _seed() -> None:
        async with session_scope(write=True) as session:
            for i in range(count):
                belief = Particle(
                    content=f"The service listens on port {8000 + i}.",
                    confidence=Confidence(
                        value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                    ),
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    asserted_by="test-agent",
                    provenance=[
                        ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=f"ce-{i}")
                    ],
                )
                # One tight cluster: every pair clears the 0.6 similarity gate.
                embedding = [0.6, 0.8 + i * 1e-4] + [0.0] * 382
                await insert_particle(session, belief, embedding=embedding)
            await session.commit()

    asyncio.run(_seed())


class TestIncompleteAudit:
    """An account-level LLM failure is an incomplete audit, not a clean exit 0."""

    @pytest.fixture
    def seeded(self, cli_db: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("AUDIT_MAX_CONTRADICTION_PROBES", "200")
        from particles.config import reset_config

        reset_config()
        _seed_similar_beliefs(21)  # 210 candidate pairs, 200 planned under the cap
        return cli_db

    def test_credit_balance_error_exits_incomplete(self, seeded: Path) -> None:
        from unittest.mock import MagicMock

        from particles.llm import set_client

        client = MagicMock()
        client.messages.create.side_effect = _CreditBalanceError()
        set_client(client)
        try:
            result = runner.invoke(app, ["audit"])
        finally:
            set_client(None)

        assert result.exit_code == audit_mod.EXIT_INCOMPLETE == 1, result.output
        out = result.stdout
        assert (
            "Re-audited store 'default' (INCOMPLETE: contradiction check skipped, "
            "LLM unavailable) → 21 beliefs" in out
        )
        # One skip line naming the cause class, not the raw 400 JSON, and no
        # contradictory "probe failed" line beside it.
        assert out.count("contradiction check skipped:") == 1
        assert "contradiction check skipped: LLM unavailable (credit balance too low)" in out
        assert "semantic probe" not in out
        assert "Error code: 400" not in out
        # The breaker stopped the loop: one probe was sent, not 200 no-ops.
        assert client.messages.create.call_count == 1
        probe_lines = [
            line for line in result.stderr.splitlines() if "contradiction probe…" in line
        ]
        assert len(probe_lines) < 15
        assert "audit incomplete" in result.stderr
        assert "`particles audit`" in result.stderr

    def test_json_carries_complete_false_and_the_reason(self, seeded: Path) -> None:
        from unittest.mock import MagicMock

        from particles.llm import set_client

        client = MagicMock()
        client.messages.create.side_effect = _CreditBalanceError()
        set_client(client)
        try:
            result = runner.invoke(app, ["audit", "--format", "json"])
        finally:
            set_client(None)

        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert payload["complete"] is False
        assert payload["semantic_skip_reason"] == "LLM unavailable (credit balance too low)"
        assert payload["beliefs"] == 21

    def test_normal_run_still_exits_zero(self, seeded: Path) -> None:
        with patch(
            "particles.operations.lint.contradictions._llm_check_contradiction",
            AsyncMock(return_value=None),
        ) as probe:
            result = runner.invoke(app, ["audit"])

        assert result.exit_code == 0, result.output
        assert probe.await_count == 200
        assert "INCOMPLETE" not in result.stdout
        assert "Re-audited store 'default' → 21 beliefs" in result.stdout
        assert "audit incomplete" not in result.stderr

    def test_reaudit_without_key_is_incomplete(
        self, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        audit = AsyncMock(
            return_value=_fake_report(
                files_audited=None, semantic_skipped=True, semantic_skip_reason="no API key"
            )
        )
        with patch("particles.operations.audit.run_memory_audit", audit):
            result = runner.invoke(app, ["audit"])
        assert result.exit_code == 1
        assert "INCOMPLETE: contradiction check skipped, no API key" in result.stdout
        assert "once ANTHROPIC_API_KEY is set" in result.stderr

    def test_first_run_audit_reports_incomplete_without_failing_init(
        self,
        cli_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from particles.api.cli import init as init_mod

        home = tmp_path / "home"
        mem = home / ".claude" / "projects" / "-Users-x-repo" / "memory"
        mem.mkdir(parents=True)
        (mem / "MEMORY.md").write_text("- A fact worth keeping.\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(audit_mod, "projection_enabled", lambda: False)
        report = _fake_report(
            files_audited=1,
            semantic_skipped=True,
            semantic_skip_reason="LLM unavailable (credit balance too low)",
        )
        with (
            patch(
                "particles.operations.deposit.deposit_text_versioned",
                AsyncMock(return_value=("e1", "s1", False)),
            ),
            patch("particles.operations.audit.run_memory_audit", AsyncMock(return_value=report)),
        ):
            init_mod._offer_first_run_audit("default")  # must not raise

        out = capsys.readouterr().out
        assert "INCOMPLETE: contradiction check skipped" in out
        assert "Run `particles audit` once the LLM is available again" in out
        # The audit ran; init must not claim it was skipped.
        assert "First-run audit skipped" not in out

    def test_notice_names_a_non_default_store(self) -> None:
        report = _fake_report(
            store="memory",
            semantic_skipped=True,
            semantic_skip_reason="LLM unavailable (credit balance too low)",
        )
        notice = audit_mod._incomplete_notice(report)
        assert notice == (
            "audit incomplete: contradiction check skipped, LLM unavailable (credit "
            "balance too low). Run `particles audit --store memory` once the LLM is "
            "available again to finish it."
        )

    def test_unextracted_file_exits_incomplete_and_names_the_retry(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A harvested file that produced no beliefs makes the audit incomplete,
        under the same convention as a skipped contradiction check."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        deposit = AsyncMock(return_value=("e1", "s1", False))
        audit = AsyncMock(
            return_value=_fake_report(
                extracted_snapshots=1,
                extraction_failures=1,
                extraction_failed_files=["topic.md"],
            )
        )
        with (
            patch("particles.operations.deposit.deposit_text_versioned", deposit),
            patch("particles.operations.audit.run_memory_audit", audit),
        ):
            result = runner.invoke(app, ["audit", str(mem_dir), "--yes"])
        assert result.exit_code == audit_mod.EXIT_INCOMPLETE == 1, result.output
        assert "(INCOMPLETE: 1 file not extracted)" in result.stdout
        assert "not extracted, so absent from every count below: topic.md" in result.stdout
        assert (
            "audit incomplete: 1 file not extracted, so the census does not cover them. "
            f"Run `particles audit {mem_dir}` to retry them."
        ) in result.stderr

    def test_json_format_carries_the_extraction_outcome(
        self, cli_db: Path, mem_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        deposit = AsyncMock(return_value=("e1", "s1", False))
        audit = AsyncMock(
            return_value=_fake_report(
                extracted_snapshots=2,
                extraction_failures=1,
                extraction_failed_files=["gone.md"],
                extraction_partial_files=["long.md"],
            )
        )
        with (
            patch("particles.operations.deposit.deposit_text_versioned", deposit),
            patch("particles.operations.audit.run_memory_audit", audit),
        ):
            result = runner.invoke(app, ["audit", str(mem_dir), "--yes", "--format", "json"])
        assert result.exit_code == 1
        payload = json.loads(result.stdout[result.stdout.index("{") :])
        assert payload["complete"] is False
        assert payload["extracted_fully"] == 1
        assert payload["extraction_failed_files"] == ["gone.md"]
        assert payload["extraction_partial_files"] == ["long.md"]


class TestMeasuredUsage:
    """The audit ends with the run's measured LLM usage, priced at list price."""

    @pytest.fixture
    def seeded(self, cli_db: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from particles.config import reset_config

        reset_config()
        _seed_similar_beliefs(3)  # 3 candidate pairs, all under the probe cap
        return cli_db

    @staticmethod
    def _client() -> MagicMock:
        reply = SimpleNamespace(
            content=[SimpleNamespace(text='{"contradiction": false}')],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=100_000,
                output_tokens=2_000,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )
        client = MagicMock()
        client.messages.create.return_value = reply
        return client

    def _invoke(self, *args: str) -> Result:
        from particles.llm import set_client

        set_client(self._client())
        try:
            return runner.invoke(app, ["audit", *args])
        finally:
            set_client(None)

    # 3 probes x (100k in, 2k out) on the default claude-sonnet-4-6 ($3/$15 per
    # MTok): 0.90 + 0.09.
    LINE = (
        "LLM usage: 3 semantic lint calls (claude-sonnet-4-6), 300k input, 6k output "
        "tokens; ≈ $0.99 at list price."
    )

    def test_markdown_report_ends_with_the_usage_line(self, seeded: Path) -> None:
        result = self._invoke()
        assert result.exit_code == 0, result.output
        assert result.stdout.rstrip().splitlines()[-1] == self.LINE

    def test_json_carries_the_totals_and_stderr_the_line(self, seeded: Path) -> None:
        result = self._invoke("--format", "json")
        assert result.exit_code == 0, result.output
        usage = json.loads(result.stdout)["llm_usage"]
        assert usage["cost_usd"] == pytest.approx(0.99)
        [row] = usage["rows"]
        assert (row["purpose"], row["model"], row["calls"]) == (
            "semantic_lint",
            "claude-sonnet-4-6",
            3,
        )
        assert (row["input_tokens"], row["output_tokens"]) == (300_000, 6_000)
        assert self.LINE in result.stderr

    def test_the_run_record_carries_the_totals(self, seeded: Path) -> None:
        import asyncio

        from particles.db import session_scope
        from particles.store.event_store import OperatorEventType, list_events

        assert self._invoke().exit_code == 0

        async def _payload() -> dict[str, Any]:
            async with session_scope() as session:
                events = await list_events(session, event_type=OperatorEventType.CONSOLIDATION_RUN)
            return events[0].payload or {}

        usage = asyncio.run(_payload())["llm_usage"]
        assert isinstance(usage, dict)
        assert usage["cost_usd"] == pytest.approx(0.99)
        assert usage["rows"][0]["calls"] == 3


# ---------------------------------------------------------------------------
# The report and the heartbeat share one terminal (owner-reported, 2026-09-25)
# ---------------------------------------------------------------------------


def test_report_starts_on_a_fresh_line_after_an_in_place_heartbeat(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report began on the heartbeat's open line::

        … audit: contradiction probe 185/200 (1h36m elapsed)Audited 96 memory files …

    The heartbeat's line is erased before the report prints, and no tick repaints it.
    """
    import asyncio

    from particles.api.cli import _output
    from particles.api.cli._output import OutputSettings
    from particles.api.cli._progress import _CLEAR_LINE
    from particles.operations.audit import AuditProgress

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(get_config().cli, "heartbeat_seconds", 0.01)
    on_a_terminal = OutputSettings(progress=True)
    monkeypatch.setattr(_output, "current_output", lambda: on_a_terminal)
    monkeypatch.setattr(audit_mod, "current_output", lambda: on_a_terminal)

    async def _audit(*_args: object, **kwargs: object) -> AuditReport:
        on_progress = kwargs["on_progress"]
        assert callable(on_progress)
        on_progress(AuditProgress(phase="probe", done=185, total=200, label="contradiction probe"))
        await asyncio.sleep(0.2)  # several ticks paint the probe status in place
        return _fake_report(files_audited=96)

    with patch("particles.operations.audit.run_memory_audit", _audit):
        result = runner.invoke(app, ["audit"])

    assert result.exit_code == 0, result.output
    terminal = result.output
    assert "contradiction probe 185/200" in terminal  # the heartbeat did paint in place
    report_at = terminal.index("Audited ")
    before = terminal[:report_at]
    assert before.endswith(_CLEAR_LINE) or before.endswith("\n")
    assert "elapsed)Audited" not in terminal
    # Nothing repainted inside the report while it printed.
    assert "… audit:" not in terminal[report_at:].split(_CLEAR_LINE)[0]


def test_harvest_drops_memory_index_entries(tmp_path: Path) -> None:
    """the audit deposits MEMORY.md without its index entries."""
    d = tmp_path / "memory"
    d.mkdir()
    (d / "deploy.md").write_text("The deploy runs nightly.\n")
    (d / "MEMORY.md").write_text(
        "- [Deploy](deploy.md) — status, schedule, open work\n- Authored fact.\n"
    )
    plan = audit_mod.build_harvest_plan([d], None, None)
    memory = next(p for p in plan.deposits if p.uri_r.endswith("/MEMORY.md"))
    assert "open work" not in memory.text
    assert "Authored fact." in memory.text
