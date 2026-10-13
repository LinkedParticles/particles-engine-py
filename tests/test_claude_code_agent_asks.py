# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The Claude Code adapter's agent-extraction helpers (§2).

An ask and the agent's answer must never reach a snapshot: they would put the
previous span's text and the agent's own summary of its claims into the next
span, self-echo the evidence rule would accept. Pinned here: the exclusion
range with and without a recorded continuation, that the ask itself never ends
an open range, that the distiller drops excluded records while keeping
snapshots prefixes of each other, and that the Stop entry is opt-in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.api.cli import hook
from particles.api.cli._claude_code import (
    SessionProject,
    ask_exclusions,
    build_hook_commands,
    distill_transcript,
    transcript_last_record,
)
from particles.config import get_config
from particles.extraction.agent import ASK_MARKER
from particles.ingest.agent_extraction import apply_submission
from particles.ingest.append_base import delta_compatible


def _rec(uuid: str, kind: str, text: str, **extra: object) -> str:
    message: dict[str, object] = {"role": kind, "content": [{"type": "text", "text": text}]}
    if kind == "assistant":
        message["model"] = "claude-opus-5-5"
    return json.dumps({"uuid": uuid, "type": kind, "message": message, **extra})


LINES = [
    _rec("u1", "user", "Move staging to Postgres 17."),
    _rec("a1", "assistant", "Done; staging runs Postgres 17."),
    _rec("h1", "user", f"{ASK_MARKER}, extract this session's durable claims."),
    _rec("a2", "assistant", "Recorded 1 claim."),
    _rec("u2", "user", "Now schedule the nightly deploy."),
    _rec("a3", "assistant", "Scheduled for 02:00 UTC."),
]
JSONL = "\n".join(LINES)


def test_a_continued_ask_excludes_exactly_its_range() -> None:
    assert ask_exclusions(JSONL, [("a1", "a2")], ask_marker=ASK_MARKER) == {"h1", "a2"}


def test_an_open_ask_ends_before_the_next_typed_prompt() -> None:
    assert ask_exclusions(JSONL, [("a1", None)], ask_marker=ASK_MARKER) == {"h1", "a2"}


def test_no_asks_exclude_nothing() -> None:
    assert ask_exclusions(JSONL, [], ask_marker=ASK_MARKER) == frozenset()


def test_distiller_drops_excluded_records_and_keeps_prefixes() -> None:
    exclude = frozenset({"h1", "a2"})
    whole = distill_transcript(JSONL, "s", exclude_uuids=exclude)
    assert ASK_MARKER not in whole and "Recorded 1 claim" not in whole
    assert "Scheduled for 02:00 UTC" in whole
    # The snapshot the ask was made on (through a1) is a prefix of the later one.
    earlier = distill_transcript("\n".join(LINES[:2]), "s", exclude_uuids=exclude)
    assert whole.startswith(earlier.rstrip("\n"))


def test_last_record_and_model() -> None:
    assert transcript_last_record(JSONL) == ("a3", "claude-opus-5-5")


def test_the_stop_entry_is_opt_in() -> None:
    assert "Stop" not in build_hook_commands("particles", "memory")
    commands = build_hook_commands("particles", "memory", include_stop=True)
    assert commands["Stop"] == "particles hook stop --store memory"


def test_delta_compatibility_is_one_way_and_keeps_the_old_rule() -> None:
    assert delta_compatible({"claude-code-agent"}, "general-extractor")
    assert delta_compatible({"general-extractor", "journal-extractor"}, "general-extractor")
    assert not delta_compatible({"general-extractor"}, "claude-code-agent")
    assert not delta_compatible({"journal-extractor"}, "general-extractor")
    assert not delta_compatible({"claude-code-agent", "journal-extractor"}, "general-extractor")


def test_mcp_registration_is_found_where_claude_code_reads_it(tmp_path: object) -> None:
    from particles.api.cli._claude_code import particles_mcp_registered

    root = Path(str(tmp_path))
    home, project = root / "home", root / "work" / "repo"
    home.mkdir()
    project.mkdir(parents=True)
    server = {"command": "uv", "args": ["run", "--project", "/x", "particles", "mcp", "serve"]}
    assert not particles_mcp_registered(project, home=home)
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": {"p": server}}))
    assert particles_mcp_registered(project, home=home)
    (project / ".mcp.json").unlink()
    (home / ".claude.json").write_text(
        json.dumps({"projects": {str(project): {"mcpServers": {"p": server}}}})
    )
    assert particles_mcp_registered(project, home=home)
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    assert not particles_mcp_registered(project, home=home)


# ---------------------------------------------------------------------------
# The Stop hook end to end: a shadow answer is not asked again
# ---------------------------------------------------------------------------


def _long_transcript(turns: int) -> str:
    lines = []
    for i in range(turns):
        lines.append(_rec(f"u{i}", "user", f"Turn {i}: " + "the staging database notes. " * 120))
        lines.append(_rec(f"a{i}", "assistant", f"Reply {i}: " + "recorded that point. " * 120))
    return "\n".join(lines)


async def test_a_submitted_shadow_ask_is_not_asked_again(
    file_db_session: AsyncSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two Stops in shadow with a submission between: the second never re-asks the span.

    The owner's live store asked the same span at the end of every turn,
    because a shadow answer never moves the extraction base and an answered
    ask allowed the catch-up waiver (observed 2026-10-11).
    """
    cfg = get_config().claude_code.agent_extraction
    cfg.mode = "shadow"
    cfg.min_new_chars = 2000
    cfg.max_span_chars = 4000  # the backlog exceeds one span, so catch-up is in play
    cfg.max_asks_per_session = None
    monkeypatch.setattr(
        "particles.api.cli._claude_code.particles_mcp_registered", lambda cwd, **_: True
    )
    monkeypatch.setattr(
        hook,
        "resolve_session_project",
        lambda payload: SessionProject(key="", memory_dir=None, resolved_from="test"),
    )
    transcript = tmp_path / "s.jsonl"
    transcript.write_text(_long_transcript(6))
    payload = {"session_id": "shadow-loop", "transcript_path": str(transcript), "cwd": "/"}

    first: dict[str, object] = {}
    assert await hook._stop("default", payload, first) is not None
    start, end = first["span"]  # type: ignore[misc]
    assert start == 0

    out = await apply_submission(
        file_db_session, ask_id=str(first["asked"]), store="default", candidates=[]
    )
    assert (out.mode, out.written, out.closed_span) == ("shadow", 0, False)

    # The agent's answering turn ends; the continuation Stop records it.
    with transcript.open("a") as f:
        f.write("\n" + _rec("h0", "user", f"{ASK_MARKER}, extract this session's claims."))
        f.write("\n" + _rec("r0", "assistant", "Recorded nothing."))
    cont: dict[str, object] = {}
    assert await hook._stop("default", {**payload, "stop_hook_active": True}, cont) is None
    assert cont["continued"] == first["asked"]

    second: dict[str, object] = {}
    assert await hook._stop("default", payload, second) is not None
    next_start, next_end = second["span"]  # type: ignore[misc]
    assert (next_start, next_end > end) == (end, True)

    # The second ask is unanswered, so the next Stop waits out the interval.
    # This reads the asks' times back from SQLite, which drops their zone.
    third: dict[str, object] = {}
    assert await hook._stop("default", payload, third) is None
    assert third["ask"] == "inside interval"
