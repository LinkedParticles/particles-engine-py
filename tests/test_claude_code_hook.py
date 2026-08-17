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
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli._claude_code import (
    distill_transcript,
    redact_secrets,
    truncate_on_line_boundary,
)

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
    return tmp_path


def _hook_log_lines(home: Path) -> list[dict[str, Any]]:
    path = home / ".particles" / "claude-code" / "hooks.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_transcript(path: Path, *turns: tuple[str, str]) -> None:
    lines = [
        json.dumps({"type": speaker, "message": {"role": speaker, "content": text}})
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
