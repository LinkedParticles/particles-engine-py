# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0
"""The memory-file walkers do not follow a symlinked ``*.md`` (security review P20, N24).

Safe property: neither the SessionEnd harvest nor ``particles audit`` deposits
the bytes of a file outside the memory directory through a symlink planted in
it, the rule the folder importers already follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_CANARY = "AUDIT-CANARY-OUTSIDE-MEMORY"


@pytest.fixture
def planted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    memory = home / ".claude" / "projects" / "-repo" / "memory"
    memory.mkdir(parents=True)
    (memory / "real.md").write_text("an ordinary memory\n", encoding="utf-8")
    outside = tmp_path / "private" / "notes.txt"
    outside.parent.mkdir()
    outside.write_text(f"{_CANARY}\n", encoding="utf-8")
    (memory / "linked.md").symlink_to(outside)
    return memory, outside


def test_audit_plan_does_not_follow_a_symlinked_memory_file(planted: tuple[Path, Path]) -> None:
    from particles.api.cli.audit import build_harvest_plan

    memory, outside = planted
    plan = build_harvest_plan([memory], None, None)
    assert all(_CANARY not in d.text for d in plan.deposits)
    assert all(d.uri_r != outside.resolve().as_uri() for d in plan.deposits)


@dataclass
class _Outcome:
    entry_id: str = "e"
    snapshot_id: str = "s"
    unchanged: bool = False


class _RecordingBackend:
    # Not the local store, so the harvest's absence step is skipped
    # and the test stays about what the walk deposits.
    remote = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def deposit_text_at_uri(self, **kwargs: Any) -> _Outcome:
        self.calls.append(kwargs)
        return _Outcome()


async def test_session_end_harvest_does_not_follow_a_symlinked_memory_file(
    planted: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import particles.api.client as client_module
    from particles.api.cli.hook import _harvest_memory_files

    memory, outside = planted
    backend = _RecordingBackend()
    monkeypatch.setattr(client_module, "get_backend", lambda: backend)
    await _harvest_memory_files("default", memory, "-repo")
    assert all(_CANARY not in c["text"] for c in backend.calls)
    assert all(c["uri_r"] != outside.resolve().as_uri() for c in backend.calls)


def test_audit_skips_a_symlinked_transcript(tmp_path: Path) -> None:
    """The audit's transcript walk does not follow a symlinked ``*.jsonl`` (R17)."""
    import json
    import os

    from particles.api.cli import audit as audit_mod

    outside = tmp_path / "elsewhere" / "notes.jsonl"
    outside.parent.mkdir()
    outside.write_text(
        json.dumps({"type": "user", "message": {"content": "outside the dir"}}) + "\n"
    )
    tdir = tmp_path / "project"
    tdir.mkdir()
    os.symlink(outside, tdir / "linked.jsonl")

    plan = audit_mod.build_harvest_plan([], tdir, max_entries=None)
    assert plan.transcripts == 0


def test_audit_still_plans_a_regular_transcript(tmp_path: Path) -> None:
    """Control for R17: an ordinary transcript in the walked directory is planned."""
    import json

    from particles.api.cli import audit as audit_mod

    tdir = tmp_path / "project"
    tdir.mkdir()
    (tdir / "s1.jsonl").write_text(
        json.dumps({"type": "user", "message": {"content": "a real session line"}}) + "\n"
    )

    plan = audit_mod.build_harvest_plan([], tdir, max_entries=None)
    assert plan.transcripts == 1


def test_a_symlinked_transcript_records_no_launch_directory(tmp_path: Path) -> None:
    """A transcript planted as a link never chooses a project key by its ``cwd`` (R17)."""
    import json
    import os

    from particles.api.cli._claude_code import _recorded_cwds

    outside = tmp_path / "elsewhere.jsonl"
    outside.write_text(json.dumps({"cwd": "/somewhere/else"}) + "\n")
    link = tmp_path / "session.jsonl"
    os.symlink(outside, link)

    assert _recorded_cwds(outside) == ["/somewhere/else"]
    assert _recorded_cwds(link) == []
