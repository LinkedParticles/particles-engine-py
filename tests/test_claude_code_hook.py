# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``particles hook …`` verbs + their pure helpers.

Covers the ADR's test checklist for the hook side: source-gating of the
digest push, distillation determinism, redaction patterns, the byte budget,
APPEND_ONLY re-harvest no-op vs tail-append, the catch-up sweep, and the
degrade-to-nothing contract (every failure path exits 0 and logs).

The CLI tests use the ``cli_db`` file-based-SQLite fixture (see
``tests/test_cli.py`` — each invocation opens its own session in a fresh
``asyncio.run``) and point ``HOME`` at ``tmp_path`` so the hook log
(``~/.particles/claude-code/hooks.jsonl``) lands in the sandbox.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli._claude_code import (
    distill_transcript,
    redact_secrets,
    transcript_started_at,
    truncate_on_line_boundary,
)
from tests._claude_projects import bind_hooks_to_store

# ---------------------------------------------------------------------------
# fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def hook_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HOME at tmp_path so the hook log / state dir land in the sandbox."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # The hooks this fake home "installed" name the ``cli_db`` store, which is
    # what lets the projection render into its fake ``~/.claude/projects``.
    bind_hooks_to_store(tmp_path, f"sqlite+aiosqlite:///{tmp_path / 'cli.db'}")
    return tmp_path


def _hook_log_lines(home: Path) -> list[dict[str, Any]]:
    path = home / ".particles" / "claude-code" / "hooks.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_transcript(path: Path, *turns: tuple[str, str], timestamp: str | None = None) -> None:
    lines = [
        json.dumps(
            {
                "type": speaker,
                "message": {"role": speaker, "content": text},
                **({"timestamp": timestamp} if timestamp else {}),
            }
        )
        for speaker, text in turns
    ]
    path.write_text("\n".join(lines) + "\n")


def _session_end_payload(transcript: Path, session_id: str) -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": str(transcript),
            "cwd": "/some/project",
            "reason": "prompt_input_exit",
        }
    )


async def _corpus_entries() -> list[Any]:
    from particles.corpus.store import list_entries
    from particles.db import session_scope

    async with session_scope() as session:
        return await list_entries(session, limit=100, source_type=None)


async def _snapshot_count(entry_id: str) -> int:
    from particles.corpus.store import list_snapshots_for_entry
    from particles.db import session_scope

    async with session_scope() as session:
        return len(await list_snapshots_for_entry(session, entry_id))


# ---------------------------------------------------------------------------
# Distillation (§3a) — deterministic, LLM-free
# ---------------------------------------------------------------------------


class TestTranscriptStart:
    """use mining reads exposure as of the session's first timestamp."""

    def test_first_parseable_timestamp_wins(self) -> None:
        from datetime import UTC, datetime

        jsonl = "\n".join(
            [
                "not json",
                json.dumps({"type": "summary"}),
                json.dumps({"type": "user", "timestamp": "garbage"}),
                json.dumps({"type": "user", "timestamp": "2026-07-15T12:00:00.000Z"}),
                json.dumps({"type": "assistant", "timestamp": "2026-07-15T12:05:00Z"}),
            ]
        )
        assert transcript_started_at(jsonl) == datetime(2026, 7, 15, 12, 0, tzinfo=UTC)

    def test_none_without_a_timestamp(self) -> None:
        assert transcript_started_at(json.dumps({"type": "user"})) is None


class TestDistillation:
    TRANSCRIPT = "\n".join(
        [
            json.dumps({"type": "user", "message": {"role": "user", "content": "hello there"}}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "private reasoning"},
                            {"type": "text", "text": "hi — checking."},
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "git status"},
                            },
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "content": "SECRET tool payload"}],
                    },
                }
            ),
            "not json at all {{{",
            json.dumps({"type": "summary", "summary": "meta line ignored"}),
        ]
    )

    def test_speaker_turns_verbatim_tools_elided_results_dropped(self) -> None:
        out = distill_transcript(self.TRANSCRIPT, "sess-1")
        assert "# Claude Code session sess-1" in out
        assert "## User\n\nhello there" in out
        assert "hi — checking." in out
        assert "[tool: Bash — git status]" in out
        # Tool results and thinking never reach the corpus (§7).
        assert "SECRET tool payload" not in out
        assert "private reasoning" not in out

    def test_deterministic(self) -> None:
        assert distill_transcript(self.TRANSCRIPT, "s") == distill_transcript(self.TRANSCRIPT, "s")

    def test_malformed_and_empty_lines_skipped(self) -> None:
        assert distill_transcript("not json\n\n[1, 2]\n") == ""
        assert distill_transcript("") == ""

    def test_harness_injected_user_records_are_left_out(self) -> None:
        # A loaded skill's body arrives as a user record marked isMeta; it is
        # not the person speaking, so it never becomes a ## User turn.
        skill = json.dumps(
            {
                "type": "user",
                "isMeta": True,
                "sourceToolUseID": "toolu_1",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Base directory for this skill: /x"}],
                },
            }
        )
        typed = json.dumps({"type": "user", "message": {"role": "user", "content": "run it"}})
        out = distill_transcript(skill + "\n" + typed)
        assert "Base directory for this skill" not in out
        assert out.count("## User") == 1 and "run it" in out

    def test_tool_result_only_turn_produces_no_heading(self) -> None:
        line = json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": [{"type": "tool_result", "content": "x"}]},
            }
        )
        assert distill_transcript(line) == ""


# ---------------------------------------------------------------------------
# Redaction (§7)
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_sk_keys(self) -> None:
        out = redact_secrets("my key is sk-ant-api03-abcdefghijklmnop123456 ok")
        assert "sk-ant" not in out
        assert "[REDACTED API KEY]" in out

    def test_aws_access_key(self) -> None:
        out = redact_secrets("aws_access_key_id = AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in out
        assert "[REDACTED AWS KEY]" in out

    def test_bearer_header(self) -> None:
        out = redact_secrets("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig")
        assert "eyJhbGciOiJIUzI1NiJ9" not in out
        assert "Bearer [REDACTED]" in out

    def test_pem_block(self) -> None:
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"
        out = redact_secrets(f"here:\n{pem}\ndone")
        assert "MIIEowIBAAKCAQEA" not in out
        assert "[REDACTED PEM BLOCK]" in out

    def test_plain_text_untouched(self) -> None:
        text = "nothing secret here, just a skeleton key mention"
        assert redact_secrets(text) == text


# ---------------------------------------------------------------------------
# Digest byte budget (§2)
# ---------------------------------------------------------------------------


class TestTruncateOnLineBoundary:
    def test_under_budget_untouched(self) -> None:
        assert truncate_on_line_boundary("a\nb\n", 100) == "a\nb\n"

    def test_zero_budget_means_no_cap(self) -> None:
        big = "x" * 100_000
        assert truncate_on_line_boundary(big, 0) == big

    def test_truncates_on_line_boundary_with_disclosed_footer(self) -> None:
        text = "\n".join(f"line {i:04d} " + "x" * 40 for i in range(200))
        out = truncate_on_line_boundary(text, 1000)
        assert len(out.encode("utf-8")) <= 1000
        assert "truncated at 1000 bytes" in out
        # Every retained content line is intact (no mid-line cut).
        for line in out.splitlines():
            if line.startswith("line "):
                assert len(line) == len("line 0000 ") + 40


# ---------------------------------------------------------------------------
# hook session-start (§2)
# ---------------------------------------------------------------------------


class TestSessionStart:
    def test_startup_pushes_digest_json(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["hook", "session-start", "--store", "default"],
            input=json.dumps({"session_id": "s1", "source": "startup"}),
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        hso = payload["hookSpecificOutput"]
        assert hso["hookEventName"] == "SessionStart"
        assert "Memory digest" in hso["additionalContext"]
        records = _hook_log_lines(hook_home)
        assert records and records[-1]["outcome"] == "ok"
        assert records[-1]["digest_bytes"] > 0

    def test_resume_is_skipped_with_no_output(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["hook", "session-start", "--store", "default"],
            input=json.dumps({"session_id": "s1", "source": "resume"}),
        )
        assert result.exit_code == 0
        assert result.stdout == ""
        records = _hook_log_lines(hook_home)
        assert records[-1]["skipped"] == "resume"

    def test_failure_degrades_to_exit_zero_and_logs(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        # An unknown store handle raises inside the backend — the contract is
        # exit 0, no output, one error line in the hook log (§2).
        result = runner.invoke(
            app,
            ["hook", "session-start", "--store", "no-such-store"],
            input=json.dumps({"session_id": "s1", "source": "startup"}),
        )
        assert result.exit_code == 0
        assert result.stdout == ""
        records = _hook_log_lines(hook_home)
        assert records[-1]["outcome"] == "error"
        assert records[-1]["error"]

    def test_garbage_stdin_degrades_to_exit_zero(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        result = runner.invoke(
            app, ["hook", "session-start", "--store", "default"], input="not json {{{"
        )
        assert result.exit_code == 0
        assert result.stdout == ""
        assert _hook_log_lines(hook_home)[-1]["outcome"] == "error"


# ---------------------------------------------------------------------------
# hook session-end (§3)
# ---------------------------------------------------------------------------


class TestSessionEnd:
    def test_harvests_transcript_as_conversation_append_only(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / ".claude" / "projects" / "-my-project"
        project_dir.mkdir(parents=True)
        transcript = project_dir / "sess-a.jsonl"
        _write_transcript(transcript, ("user", "remember my key sk-abcdefghijklmnop1234 please"))

        result = runner.invoke(
            app,
            ["hook", "session-end", "--store", "default"],
            input=_session_end_payload(transcript, "sess-a"),
        )
        assert result.exit_code == 0
        assert result.stdout == ""

        entries = asyncio.run(_corpus_entries())
        conv = [e for e in entries if e.uri_r == "claude-code://session/sess-a"]
        assert len(conv) == 1
        assert conv[0].source_type == "CONVERSATION"
        assert conv[0].mutability == "APPEND_ONLY"
        assert "claude-code" in conv[0].tags
        assert "session:sess-a" in conv[0].tags
        assert "project:-my-project" in conv[0].tags

        # The redaction pass ran before the bytes reached the corpus (§7).
        from particles.corpus.deposit import load_blob
        from particles.db import session_scope

        async def _blob_text() -> str:
            from particles.corpus.store import list_snapshots_for_entry

            async with session_scope() as session:
                snap = (await list_snapshots_for_entry(session, conv[0].entry_id))[0]
            return load_blob(snap.content_hash).decode()

        text = asyncio.run(_blob_text())
        assert "sk-abcdefghijklmnop1234" not in text
        assert "[REDACTED API KEY]" in text

        records = _hook_log_lines(hook_home)
        assert records[-1]["outcome"] == "ok"
        assert records[-1]["deposited"] == 1

    def test_transcript_deposit_carries_the_session_start(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        # the session's first timestamp is the reference date its
        # claims resolve against, so an act the session performed is dated to
        # the day it happened rather than to the deferred extract pass.
        project_dir = tmp_path / ".claude" / "projects" / "-dated"
        project_dir.mkdir(parents=True)
        transcript = project_dir / "sess-d.jsonl"
        _write_transcript(transcript, ("user", "run the tests"), timestamp="2026-10-14T05:12:00Z")
        result = runner.invoke(
            app,
            ["hook", "session-end", "--store", "default"],
            input=_session_end_payload(transcript, "sess-d"),
        )
        assert result.exit_code == 0

        async def _published_at() -> Any:
            from particles.corpus.store import list_snapshots_for_entry
            from particles.db import session_scope

            entries = await _corpus_entries()
            conv = [e for e in entries if e.uri_r == "claude-code://session/sess-d"]
            async with session_scope() as session:
                snap = (await list_snapshots_for_entry(session, conv[0].entry_id))[0]
            return snap.content_published_at

        published = asyncio.run(_published_at())
        assert published is not None
        assert published.date().isoformat() == "2026-10-14"

    def test_unchanged_reharvest_noop_vs_tail_append(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / ".claude" / "projects" / "-p"
        project_dir.mkdir(parents=True)
        transcript = project_dir / "sess-b.jsonl"
        _write_transcript(transcript, ("user", "first turn"))
        payload = _session_end_payload(transcript, "sess-b")

        runner.invoke(app, ["hook", "session-end", "--store", "default"], input=payload)
        entries = asyncio.run(_corpus_entries())
        entry = next(e for e in entries if e.uri_r == "claude-code://session/sess-b")
        assert asyncio.run(_snapshot_count(entry.entry_id)) == 1

        # Unchanged transcript → content-hash no-op (idempotence is the
        # existing corpus contract, §3a).
        runner.invoke(app, ["hook", "session-end", "--store", "default"], input=payload)
        assert asyncio.run(_snapshot_count(entry.entry_id)) == 1
        assert _hook_log_lines(hook_home)[-1]["unchanged"] == 1

        # Grown transcript → a second snapshot on the SAME entry.
        with transcript.open("a") as f:
            f.write(
                json.dumps({"type": "user", "message": {"role": "user", "content": "more"}}) + "\n"
            )
        runner.invoke(app, ["hook", "session-end", "--store", "default"], input=payload)
        assert asyncio.run(_snapshot_count(entry.entry_id)) == 2
        sess_b = [e for e in asyncio.run(_corpus_entries()) if e.uri_r and "sess-b" in e.uri_r]
        assert len(sess_b) == 1

    def test_harvests_memory_files_as_mutable_markdown(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / ".claude" / "projects" / "-p2"
        memory_dir = project_dir / "memory"
        memory_dir.mkdir(parents=True)
        transcript = project_dir / "sess-c.jsonl"
        _write_transcript(transcript, ("user", "hi"))
        (memory_dir / "MEMORY.md").write_text("# Memory\n- prefers tabs\n")

        runner.invoke(
            app,
            ["hook", "session-end", "--store", "default"],
            input=_session_end_payload(transcript, "sess-c"),
        )
        entries = asyncio.run(_corpus_entries())
        mem = [e for e in entries if e.source_type == "LOCAL_MARKDOWN"]
        assert len(mem) == 1
        assert mem[0].mutability == "MUTABLE"
        assert "memory-file" in mem[0].tags

    def test_catchup_sweep_harvests_recent_unharvested_transcripts(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / ".claude" / "projects" / "-p3"
        project_dir.mkdir(parents=True)
        current = project_dir / "sess-live.jsonl"
        _write_transcript(current, ("user", "current session"))
        # A crashed session's transcript persists on disk but was never
        # harvested (SessionEnd never fired) — the sweep picks it up (§3c).
        crashed = project_dir / "sess-crashed.jsonl"
        _write_transcript(crashed, ("user", "crashed session content"))

        result = runner.invoke(
            app,
            ["hook", "session-end", "--store", "default"],
            input=_session_end_payload(current, "sess-live"),
        )
        assert result.exit_code == 0
        uris = {e.uri_r for e in asyncio.run(_corpus_entries())}
        assert "claude-code://session/sess-live" in uris
        assert "claude-code://session/sess-crashed" in uris
        assert _hook_log_lines(hook_home)[-1]["swept"] == 1

    def test_transcripts_disabled_still_harvests_memory_files(
        self,
        runner: CliRunner,
        cli_db: Path,
        hook_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # claude_code.harvest.transcripts=false via a config file (§7): the
        # transcript-free-beliefs posture.
        config = tmp_path / "config.yaml"
        config.write_text(
            "claude_code:\n  harvest:\n    transcripts: false\n"
            f"storage:\n  database_url: sqlite+aiosqlite:///{cli_db}\n"
        )
        monkeypatch.setenv("PARTICLES_CONFIG", str(config))
        from particles.config import reset_config

        reset_config()

        project_dir = tmp_path / ".claude" / "projects" / "-p4"
        memory_dir = project_dir / "memory"
        memory_dir.mkdir(parents=True)
        transcript = project_dir / "sess-d.jsonl"
        _write_transcript(transcript, ("user", "should not be harvested"))
        (memory_dir / "MEMORY.md").write_text("# Memory\n- a note\n")

        runner.invoke(
            app,
            ["hook", "session-end", "--store", "default"],
            input=_session_end_payload(transcript, "sess-d"),
        )
        entries = asyncio.run(_corpus_entries())
        assert not [e for e in entries if e.source_type == "CONVERSATION"]
        assert [e for e in entries if e.source_type == "LOCAL_MARKDOWN"]

    def test_remote_harvest_refused_without_opt_in(
        self,
        runner: CliRunner,
        cli_db: Path,
        hook_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # engine.base_url set + allow_remote unset ⇒ the harvest refuses to
        # ship transcripts off-machine, logs the refusal, exits 0 (§7).
        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://engine.example:8000")
        from particles.config import reset_config

        reset_config()

        project_dir = tmp_path / ".claude" / "projects" / "-p5"
        project_dir.mkdir(parents=True)
        transcript = project_dir / "sess-e.jsonl"
        _write_transcript(transcript, ("user", "sensitive"))

        result = runner.invoke(
            app,
            ["hook", "session-end", "--store", "default"],
            input=_session_end_payload(transcript, "sess-e"),
        )
        assert result.exit_code == 0
        records = _hook_log_lines(hook_home)
        assert records[-1]["skipped"] == "remote-harvest-disabled"

        # The store never saw the transcript (the engine would have).
        monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL")
        reset_config()
        assert not [e for e in asyncio.run(_corpus_entries()) if e.source_type == "CONVERSATION"]

    def test_missing_transcript_is_not_an_error(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["hook", "session-end", "--store", "default"],
            input=_session_end_payload(tmp_path / "gone.jsonl", "sess-x"),
        )
        assert result.exit_code == 0
        assert _hook_log_lines(hook_home)[-1]["outcome"] == "ok"

    def test_failure_degrades_to_exit_zero_and_logs(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / ".claude" / "projects" / "-p6"
        project_dir.mkdir(parents=True)
        transcript = project_dir / "sess-f.jsonl"
        _write_transcript(transcript, ("user", "hello"))
        result = runner.invoke(
            app,
            ["hook", "session-end", "--store", "no-such-store"],
            input=_session_end_payload(transcript, "sess-f"),
        )
        assert result.exit_code == 0
        assert result.stdout == ""
        records = _hook_log_lines(hook_home)
        assert records[-1]["outcome"] == "error"


# ---------------------------------------------------------------------------
# hook log (§6)
# ---------------------------------------------------------------------------


class TestHookLog:
    def test_tail_prints_recent_entries(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        for i in range(3):
            runner.invoke(
                app,
                ["hook", "session-start", "--store", "default"],
                input=json.dumps({"session_id": f"s{i}", "source": "resume"}),
            )
        result = runner.invoke(app, ["hook", "log", "--tail", "2"])
        assert result.exit_code == 0
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(lines) == 2
        assert json.loads(lines[-1])["session_id"] == "s2"

    def test_empty_log_message(self, runner: CliRunner, hook_home: Path) -> None:
        result = runner.invoke(app, ["hook", "log", "--tail", "5"])
        assert result.exit_code == 0
        assert "No hook log entries yet" in result.stdout


# ---------------------------------------------------------------------------
# Store-resolution diagnostics
# ---------------------------------------------------------------------------


class TestUninitializedStoreHint:
    def test_session_end_logs_actionable_hint_on_missing_tables(
        self, runner: CliRunner, hook_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point the default store at an existing-but-untabled DB (the exact
        # shape of the CWD-mismatch bug), then harvest a real transcript.
        empty_db = tmp_path / "empty.db"
        empty_db.touch()
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{empty_db}")
        from particles.config import reset_config

        reset_config()
        transcript = tmp_path / "sess-x.jsonl"
        _write_transcript(transcript, ("user", "hello there"))
        result = runner.invoke(
            app,
            ["hook", "session-end", "--store", "default"],
            input=_session_end_payload(transcript, "sess-x"),
        )
        assert result.exit_code == 0  # degrade-to-nothing contract preserved
        record = _hook_log_lines(hook_home)[-1]
        assert record["outcome"] == "error"
        assert "not initialized" in record["hint"]
        assert str(empty_db) in record["hint"]
        assert "hook doctor" in record["hint"]


class TestHookDoctor:
    def test_reports_healthy_store(self, runner: CliRunner, cli_db: Path, hook_home: Path) -> None:
        result = runner.invoke(app, ["hook", "doctor", "--store", "default"])
        assert result.exit_code == 0, result.output
        assert "resolves and is initialized" in result.stdout
        assert str(cli_db) in result.stdout

    def test_names_the_extraction_model_and_its_calibration(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        """The doctor names the extraction model and whether a fit reaches it."""
        from particles.config import ProviderSelection

        compiled = ProviderSelection()
        result = runner.invoke(app, ["hook", "doctor", "--store", "default"])
        assert result.exit_code == 0, result.output
        assert (
            f"model:          extraction {compiled.provider}:{compiled.model} (compiled default)"
            in result.stdout
        )
        assert "general-extractor calibration: none for this pairing" in result.stdout

    def test_undeclared_store_exits_nonzero(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        result = runner.invoke(app, ["hook", "doctor", "--store", "ghost"])
        assert result.exit_code == 1
        assert "not declared" in result.stdout

    def test_flags_cwd_relative_dsn(
        self, runner: CliRunner, hook_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./relative.db")
        from particles.config import reset_config

        reset_config()
        result = runner.invoke(app, ["hook", "doctor", "--store", "default"])
        assert result.exit_code == 1
        assert "working-directory-relative" in result.stdout

    def test_empty_store_reports_no_blobs(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        result = runner.invoke(app, ["hook", "doctor", "--store", "default"])
        assert result.exit_code == 0, result.output
        assert "nothing deposited yet" in result.stdout

    def test_warns_when_memory_md_nears_the_load_window(
        self,
        runner: CliRunner,
        cli_db: Path,
        hook_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """the index sits below the region, so its tail is lost first."""
        from particles.api.cli._claude_code import claude_project_slug
        from tests._claude_projects import make_repo

        repo = make_repo(tmp_path / "src" / "repo")
        memory_dir = tmp_path / ".claude" / "projects" / claude_project_slug(repo) / "memory"
        memory_dir.mkdir(parents=True)
        memory_md = memory_dir / "MEMORY.md"
        monkeypatch.chdir(repo)

        memory_md.write_text("".join(f"- line {i}\n" for i in range(179)))
        quiet = runner.invoke(app, ["hook", "doctor", "--store", "default"])
        assert "memory size:" not in quiet.stdout

        memory_md.write_text("".join(f"- line {i}\n" for i in range(180)))
        loud = runner.invoke(app, ["hook", "doctor", "--store", "default"])
        assert loud.exit_code == 0, loud.output  # reported, never fatal
        assert "memory size:    ⚠ MEMORY.md is 180 lines" in loud.stdout

    def test_reports_the_memory_directory_and_strays(
        self,
        runner: CliRunner,
        cli_db: Path,
        hook_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A linked worktree's stray memory directory is named, never removed."""
        from particles.api.cli._claude_code import claude_project_slug
        from tests._claude_projects import make_repo, make_worktree, write_transcript

        repo = make_repo(tmp_path / "src" / "repo")
        worktree = make_worktree(repo, tmp_path / "src" / "wt")
        projects = tmp_path / ".claude" / "projects"
        stray = projects / claude_project_slug(worktree) / "memory"
        write_transcript(stray.parent / "s.jsonl", str(worktree))
        stray.mkdir()
        monkeypatch.chdir(worktree)

        result = runner.invoke(app, ["hook", "doctor", "--store", "default"])

        assert result.exit_code == 0, result.output
        repo_memory = projects / claude_project_slug(repo) / "memory"
        assert f"memory dir:     {repo_memory} (not created yet)" in result.stdout
        assert "shares its repository's memory directory" in result.stdout
        assert "stray memory:   1 directory" in result.stdout
        assert str(stray) in result.stdout
        assert stray.is_dir()

    def test_reports_a_project_observer_that_is_not_in_effect_yet(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        """The one silent state: `project` set, store never rescoped."""
        from particles.config import get_config

        get_config().claude_code.observer_scope = "store"
        assert "observer scope: store" in runner.invoke(app, ["hook", "doctor"]).stdout

        get_config().claude_code.observer_scope = "project"  # the default
        before = runner.invoke(app, ["hook", "doctor"])
        assert before.exit_code == 0 and "NOT in effect" in before.stdout

        runner.invoke(app, ["memory", "rescope"])
        get_config().claude_code.observer_scope = "project"
        assert "global beliefs plus its project's" in runner.invoke(app, ["hook", "doctor"]).stdout

    def test_warns_about_a_session_end_hook_without_a_timeout(
        self,
        runner: CliRunner,
        cli_db: Path,
        hook_home: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An untimed SessionEnd hook is cancelled at 1.5 s; the doctor names it, non-fatally."""
        from particles.api.cli._claude_code import (
            build_hook_commands,
            merge_particles_hook_entries,
            render_settings_json,
        )

        monkeypatch.chdir(hook_home)  # keep the real repository's settings out of it
        settings_path = hook_home / ".claude" / "settings.json"

        # The fixture's install is the pre-timeout shape.
        result = runner.invoke(app, ["hook", "doctor", "--store", "default"])
        assert result.exit_code == 0, result.output
        assert f"the SessionEnd hook in {settings_path} has no timeout" in result.stdout
        assert "`particles init claude-code` to add one" in result.stdout

        settings = json.loads(settings_path.read_text())
        repaired = merge_particles_hook_entries(
            settings, build_hook_commands("particles", "default")
        )
        settings_path.write_text(render_settings_json(repaired))
        result = runner.invoke(app, ["hook", "doctor", "--store", "default"])
        assert result.exit_code == 0, result.output
        assert "hook timeout" not in result.stdout

    def test_names_project_flag_for_a_project_scope_install(
        self,
        runner: CliRunner,
        cli_db: Path,
        hook_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from tests._claude_projects import make_repo

        (hook_home / ".claude" / "settings.json").unlink()
        repo = make_repo(tmp_path / "src" / "repo")
        bind_hooks_to_store(repo, "sqlite+aiosqlite:////elsewhere.db")
        monkeypatch.chdir(repo)

        result = runner.invoke(app, ["hook", "doctor", "--store", "default"])

        assert result.exit_code == 0, result.output
        assert f"the SessionEnd hook in {repo / '.claude' / 'settings.json'}" in result.stdout
        assert "`particles init claude-code --project` to add one" in result.stdout

    def test_unreachable_blobs_warn_without_failing(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        """A store whose content is elsewhere is diagnosable, not fatal."""
        _add_snapshot_with_blob()

        result = runner.invoke(app, ["hook", "doctor", "--store", "default"])

        # The store still resolves from here — only its content does not.
        assert result.exit_code == 0, result.output
        assert "resolves and is initialized" in result.stdout
        assert "none of the 1 sampled" in result.stdout


def _add_snapshot_with_blob() -> None:
    """Insert one blob-bearing snapshot whose blob was never written to disk."""
    import asyncio
    import uuid
    from datetime import UTC, datetime

    from particles.corpus.store import SnapshotRow
    from particles.db import session_scope

    async def _add() -> None:
        async with session_scope() as session:
            session.add(
                SnapshotRow(
                    snapshot_id=str(uuid.uuid4()),
                    entry_id=str(uuid.uuid4()),
                    captured_at=datetime.now(UTC),
                    content_hash="b" * 64,
                    warc_record_type="RESPONSE",
                    archive_path="/gone/corpus_blobs/bb/" + "b" * 64,
                    extraction_status="PENDING",
                )
            )
            await session.commit()

    asyncio.run(_add())


# ---------------------------------------------------------------------------
# hook doctor: which install sessions run, and whether it has gone stale
# ---------------------------------------------------------------------------

_HOOK = (
    "env PARTICLES_CONFIG=/cfg.yaml DATABASE_URL=sqlite+aiosqlite:////s.db "
    "/pin/.venv/bin/particles hook session-end --store default"
)


class TestParticlesExecutable:
    def test_reads_the_executable_from_a_hook_command(self) -> None:
        from particles.api.cli._claude_code import particles_executable

        assert particles_executable(_HOOK) == "/pin/.venv/bin/particles"

    def test_a_uv_run_server_names_no_executable(self) -> None:
        from particles.api.cli._claude_code import particles_executable

        argv = ["/opt/homebrew/bin/uv", "run", "--project", "/repo", "particles", "mcp", "serve"]
        assert particles_executable(argv) is None
        assert particles_executable(["/pin/.venv/bin/particles", "mcp", "serve"]) == (
            "/pin/.venv/bin/particles"
        )

    def test_reads_user_scope_mcp_servers(self, tmp_path: Path) -> None:
        from particles.api.cli._claude_code import particles_mcp_executables

        (tmp_path / ".claude.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "particles": {
                            "command": "/pin/.venv/bin/particles",
                            "args": ["mcp", "serve"],
                        },
                        "other": {"command": "/usr/bin/other", "args": []},
                    }
                }
            )
        )
        assert particles_mcp_executables(tmp_path) == ["/pin/.venv/bin/particles"]


class TestInstallReport:
    def _report(self, pin: Any = None, **kw: Any) -> str:
        from particles.api.cli._claude_code import install_report

        args: dict[str, Any] = {
            "root": Path("/pin"),
            "version": "1.2.0",
            "pin": pin,
            "hook_commands": [_HOOK],
            "mcp_exes": ["/pin/.venv/bin/particles"],
            "running_exe": "/pin/.venv/bin/particles",
        }
        args.update(kw)
        return "\n".join(install_report(**args))

    @staticmethod
    def _pin(branch: str | None = None, tag: str | None = None, newest: str | None = None) -> Any:
        from particles.api.cli._claude_code import InstallPin

        return InstallPin(branch=branch, tag=tag, newest_tag=newest)

    def test_a_current_pin_is_not_a_warning(self) -> None:
        out = self._report(self._pin(tag="v1.2.0", newest="v1.2.0"))
        assert "pinned at v1.2.0, the newest tag fetched" in out
        assert "⚠" not in out

    def test_a_pin_behind_the_newest_tag_says_how_to_advance(self) -> None:
        out = self._report(self._pin(tag="v1.2.0", newest="v1.3.0"))
        assert "⚠ pinned at v1.2.0; v1.3.0 is the newest tag fetched" in out
        assert "git -C /pin checkout --detach v1.3.0" in out
        assert "uv sync --project /pin --frozen --no-dev" in out

    def test_a_working_checkout_is_a_warning(self) -> None:
        out = self._report(self._pin(branch="main", newest="v1.3.0"))
        assert "⚠ runs from a working checkout (branch main)" in out

    def test_a_detached_commit_that_is_no_tag(self) -> None:
        assert "not a release tag" in self._report(self._pin(newest="v1.3.0"))

    def test_an_installed_wheel_reports_only_its_version(self) -> None:
        assert self._report() == "install:        Particles 1.2.0 at /pin"

    def test_mcp_server_and_hooks_on_different_installs(self) -> None:
        out = self._report(mcp_exes=["/dev/.venv/bin/particles"])
        assert "⚠ the MCP server runs /dev/.venv/bin/particles but the hooks run" in out

    def test_a_doctor_run_from_another_install_names_the_right_command(self) -> None:
        out = self._report(self._pin(branch="main"), running_exe="/dev/.venv/bin/particles")
        assert "not the install the hooks run" in out
        assert (
            "env PARTICLES_CONFIG=/cfg.yaml DATABASE_URL=sqlite+aiosqlite:////s.db "
            "/pin/.venv/bin/particles hook doctor --store default" in out
        )
        assert "working checkout" not in out  # the branch is the doctor's, not the hooks'


def test_install_pin_reads_a_real_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    import subprocess

    from particles.api.cli.hook import _install_pin

    if shutil.which("git") is None:
        pytest.skip("git not installed")
    # A git hook (pre-commit runs this suite) exports GIT_DIR / GIT_INDEX_FILE,
    # which would aim every command below at the real repository instead of
    # tmp_path; a run like that once flipped its core.bare. Clear them all.
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    # The developer's own git config may sign every commit and tag.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
        )

    git("init", "-q", "-b", "main")
    for n in (1, 2):
        git("commit", "-q", "--allow-empty", "-m", str(n))
        git("tag", f"v1.{n}.0")
    assert _install_pin(tmp_path).branch == "main"  # type: ignore[union-attr]
    git("checkout", "-q", "--detach", "v1.1.0")
    pin = _install_pin(tmp_path)
    assert pin is not None
    assert (pin.branch, pin.tag, pin.newest_tag) == (None, "v1.1.0", "v1.2.0")
    assert _install_pin(tmp_path / "not-a-checkout") is None


class TestSchemaCheck:
    def _stamp(self, db: Path, revision: str) -> None:
        import sqlite3

        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS alembic_version (version_num TEXT)")
            conn.execute("DELETE FROM alembic_version")
            conn.execute("INSERT INTO alembic_version VALUES (?)", (revision,))

    def _head_and_parent(self) -> tuple[str, str]:
        from alembic.script import ScriptDirectory

        from particles.db import alembic_paths

        script = ScriptDirectory(str(alembic_paths()[1]))
        head = script.get_revision("head")
        assert head is not None and isinstance(head.down_revision, str)
        return head.revision, head.down_revision

    def test_a_migration_this_install_does_not_know(self, cli_db: Path) -> None:
        from particles.api.cli.hook import _check_schema

        self._stamp(cli_db, "f00dfeedbeef")
        ok, lines = _check_schema("default")
        assert ok  # ahead is advisory
        assert "⚠ the store is at migration f00dfeedbeef" in lines[0]

    def test_an_unstamped_store_says_nothing_and_the_head_says_so(self, cli_db: Path) -> None:
        from particles.api.cli.hook import _check_schema

        assert _check_schema("default") == (True, [])  # create_all leaves no alembic_version
        head, _ = self._head_and_parent()
        self._stamp(cli_db, head)
        assert _check_schema("default") == (
            True,
            [f"schema:         migration {head}, this install's head"],
        )

    def test_a_store_behind_this_install_fails_with_the_migrate_command(self, cli_db: Path) -> None:
        from particles.api.cli.hook import _check_schema

        head, parent = self._head_and_parent()
        self._stamp(cli_db, parent)
        ok, lines = _check_schema("default")
        assert not ok
        assert f"✗ the store is at migration {parent}, 1 migration behind" in lines[0]
        assert f"head {head}" in lines[0]
        assert "Scheduled consolidation, the hooks and the MCP server fail" in lines[1]
        command = lines[-1].strip()
        assert command.startswith(f"env DATABASE_URL=sqlite+aiosqlite:///{cli_db} ")
        assert command.endswith(" db init")

    def test_the_doctor_exits_1_on_a_store_behind_this_install(
        self, runner: CliRunner, cli_db: Path, hook_home: Path
    ) -> None:
        _, parent = self._head_and_parent()
        self._stamp(cli_db, parent)
        result = runner.invoke(app, ["hook", "doctor", "--store", "default"])
        assert result.exit_code == 1, result.output
        assert "behind this install's migrations" in result.stdout
        assert "✓ Store resolves" not in result.stdout


def test_doctor_reports_the_install(runner: CliRunner, cli_db: Path, hook_home: Path) -> None:
    result = runner.invoke(app, ["hook", "doctor", "--store", "default"])
    assert result.exit_code == 0, result.output
    assert "install:        Particles " in result.stdout


# ---------------------------------------------------------------------------
# The retired Stop hook (deprecated)
# ---------------------------------------------------------------------------


class TestRetiredStopHook:
    _STALE = "env DATABASE_URL=x /pin/.venv/bin/particles hook stop --store default"

    def test_the_installer_writes_no_stop_entry(self) -> None:
        from particles.api.cli._claude_code import build_hook_commands

        assert set(build_hook_commands("particles", "memory")) == {"SessionStart", "SessionEnd"}

    def test_a_reinstall_drops_a_stale_stop_entry_and_keeps_a_foreign_one(self) -> None:
        from particles.api.cli._claude_code import (
            build_hook_commands,
            merge_particles_hook_entries,
        )

        foreign = {"type": "command", "command": "say done"}
        settings = {
            "hooks": {
                "Stop": [
                    {"hooks": [{"type": "command", "command": self._STALE}]},
                    {"hooks": [foreign]},
                ]
            }
        }
        merged = merge_particles_hook_entries(settings, build_hook_commands("particles", "m"))
        stop_commands = [h["command"] for g in merged["hooks"]["Stop"] for h in g["hooks"]]
        assert stop_commands == ["say done"]

    def test_the_verb_is_a_silent_no_op(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app, ["hook", "stop", "--store", "default"], input='{"session_id": "s"}'
        )
        assert result.exit_code == 0
        assert result.output == ""

    def test_doctor_warns_while_the_entry_remains(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from particles.api.cli import hook

        monkeypatch.setenv("HOME", str(tmp_path))
        settings = tmp_path / ".claude" / "settings.json"
        settings.parent.mkdir()
        settings.write_text(
            json.dumps({"hooks": {"Stop": [{"hooks": [{"command": self._STALE}]}]}})
        )
        assert any("retired" in line for line in hook._check_retired_stop_hook())
        settings.write_text(json.dumps({"hooks": {}}))
        assert hook._check_retired_stop_hook() == []
