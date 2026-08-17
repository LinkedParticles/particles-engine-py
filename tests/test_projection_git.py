"""Tests for particles/api/cli/_projection_git.py — git-versioned history.

Covers the two things the ADR calls out explicitly: the **degrade-when-not-a-
git-repo** path (the feature is simply inert, never an error) and the
**commit-message shape** (run id + ranking-delta summary), exercised against a
real temp git repo. Plus the pure delta/ordering helpers and the
failure-isolation guarantees (nothing-to-commit, memory-dir-only staging).

Git is invoked as a plain subprocess (not the Anthropic integration seam), so
these run under ``pytest -m "not integration"``.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from particles.api.cli._projection_git import (
    _GIT_CONTEXT_VARS,
    _ordered_bullet_ids,
    build_commit_message,
    commit_projection,
    format_delta,
    new_run_id,
)


@pytest.fixture(autouse=True)
def _clean_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip inherited git-context env vars (GIT_DIR, …).

    The suite may run inside a git hook (pre-commit), which exports GIT_DIR /
    GIT_INDEX_FILE pointing at the outer repo. Those would hijack the tmp-repo
    ``git -C`` calls in these tests (both the test helpers and the code under
    test), so clear them for every test here — production runs from a
    SessionEnd hook where they are never set.
    """
    for var in _GIT_CONTEXT_VARS:
        monkeypatch.delenv(var, raising=False)


# A rendered memory-index region body: ranked bullets (order = ranking) each
# ending in a `p-<shortid>` handle, plus the sources trailer.
_BODY_V1 = (
    "- Mac sleep severs subagent streams `p-aaaa11`\n"
    "- uv parses pyproject.toml on every run `p-bbbb22`\n"
    "- The live store is named particles.db `p-cccc33`\n"
    "\n"
    "<!-- sources: p-aaaa11, p-bbbb22, p-cccc33 -->\n"
)
# V2: cccc33 dropped, dddd44 added, and the top belief changed (bbbb22 first).
_BODY_V2 = (
    "- uv parses pyproject.toml on every run `p-bbbb22`\n"
    "- Mac sleep severs subagent streams `p-aaaa11`\n"
    "- A brand new belief worth recalling `p-dddd44`\n"
    "\n"
    "<!-- sources: p-aaaa11, p-bbbb22, p-dddd44 -->\n"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Test Operator")
    _git(path, "config", "user.email", "op@example.test")
    return path


# ---------------------------------------------------------------------------
# Pure helpers — ordering + delta + message shape
# ---------------------------------------------------------------------------


def test_ordered_bullet_ids_is_rank_order() -> None:
    assert _ordered_bullet_ids(_BODY_V1) == ["aaaa11", "bbbb22", "cccc33"]
    assert _ordered_bullet_ids(None) == []
    assert _ordered_bullet_ids("<!-- sources: p-x -->\n") == []


def test_delta_initial_render() -> None:
    # No prior snapshot ⇒ initial render, count only.
    assert format_delta(_BODY_V1, None, max_excerpts=6) == "delta: initial render (3 beliefs)"


def test_delta_reports_added_removed_and_top_change() -> None:
    out = format_delta(_BODY_V2, _BODY_V1, max_excerpts=6)
    head = out.splitlines()[0]
    assert head.startswith("delta: +1 −1 beliefs")
    assert "top: p-aaaa11 → p-bbbb22" in head
    # The added and removed beliefs each get an excerpt line with their handle.
    assert "  + A brand new belief worth recalling `p-dddd44`" in out
    assert "  - The live store is named particles.db `p-cccc33`" in out


def test_delta_no_change() -> None:
    assert format_delta(_BODY_V1, _BODY_V1, max_excerpts=6) == "delta: no ranking change"


def test_delta_excerpts_are_capped_but_totals_are_truthful() -> None:
    body = "".join(f"- belief number {i} `p-{i:06d}`\n" for i in range(10))
    out = format_delta(body, "", max_excerpts=3)
    # Count line states the true total (10 added), excerpts capped at 3 + a
    # "… (N more)" marker — a large delta is never silently truncated.
    assert out.splitlines()[0].startswith("delta: +10 −0 beliefs")
    assert out.count("  + ") == 3
    assert "  … (7 more)" in out


def test_commit_message_shape() -> None:
    msg = build_commit_message(
        store="default",
        outcome="rendered",
        run_id="deadbeef",
        body=_BODY_V2,
        snapshot=_BODY_V1,
        max_excerpts=6,
    )
    lines = msg.splitlines()
    assert lines[0] == "memory-projection: rendered memory-index (default)"
    assert "run-id: deadbeef" in lines
    assert any(line.startswith("delta: +1 −1 beliefs") for line in lines)
    assert lines[-1] == "Projected view of the particle store (source of truth: default)."


def test_new_run_id_is_short_hex() -> None:
    rid = new_run_id()
    assert len(rid) == 8
    assert all(c in "0123456789abcdef" for c in rid)


# ---------------------------------------------------------------------------
# commit_projection — the async git step (real temp repo)
# ---------------------------------------------------------------------------


def _commit(memory_dir: Path, *, body: str, snapshot: str | None, outcome: str = "rendered") -> str:
    return asyncio.run(
        commit_projection(
            memory_dir,
            store="default",
            outcome=outcome,
            body=body,
            snapshot=snapshot,
            run_id="deadbeef",
        )
    )


def test_degrades_silently_when_not_a_git_repo(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text(_BODY_V1, encoding="utf-8")
    # No `git init` anywhere above tmp_path ⇒ inert, never an error.
    assert _commit(memory_dir, body=_BODY_V1, snapshot=None) == "skipped: not-a-git-repo"


def test_commits_render_with_structured_message(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "memrepo")
    memory_dir = repo / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text(_BODY_V1, encoding="utf-8")

    status = _commit(memory_dir, body=_BODY_V1, snapshot=None, outcome="created")
    assert status.startswith("committed ")

    message = _git(repo, "log", "-1", "--format=%B")
    assert message.splitlines()[0] == "memory-projection: created memory-index (default)"
    assert "run-id: deadbeef" in message
    assert "delta: initial render (3 beliefs)" in message
    # The MEMORY.md was actually committed.
    tracked = _git(repo, "ls-files").split()
    assert "memory/MEMORY.md" in tracked


def test_nothing_to_commit_is_skipped(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "memrepo")
    memory_dir = repo / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text(_BODY_V1, encoding="utf-8")
    assert _commit(memory_dir, body=_BODY_V1, snapshot=None).startswith("committed ")
    # Second call with no file change ⇒ no empty commit.
    assert _commit(memory_dir, body=_BODY_V1, snapshot=_BODY_V1) == "skipped: nothing-to-commit"


def test_only_memory_dir_is_staged_never_git_add_all(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "memrepo")
    memory_dir = repo / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text(_BODY_V1, encoding="utf-8")
    # An unrelated operator file sitting elsewhere in the same repo…
    (repo / "operator_notes.txt").write_text("private, not ours\n", encoding="utf-8")

    assert _commit(memory_dir, body=_BODY_V1, snapshot=None).startswith("committed ")
    tracked = _git(repo, "ls-files").split()
    assert "memory/MEMORY.md" in tracked
    assert "operator_notes.txt" not in tracked  # never swept into our commit
