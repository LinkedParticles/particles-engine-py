# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The detached SessionEnd worker's spool protocol."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from particles.api.cli import _session_end_worker as worker
from particles.api.cli._claude_launcher import LOCK_NAME, write_spool_file


@dataclass(frozen=True)
class _Plan:
    memory_dir: Path
    text: str


class _Recorder:
    def __init__(self) -> None:
        self.log: list[dict[str, Any]] = []
        self.harvested: list[str] = []
        self.rendered: list[tuple[Path, str]] = []

    async def harvest(self, store: str, payload: dict[str, Any], record: dict[str, Any]) -> Any:
        sid = str(payload["session_id"])
        self.harvested.append(sid)
        if sid == "boom":
            raise RuntimeError("deposit failed")
        record["deposited"] = 1
        return _Plan(Path(str(payload["memory_dir"])), sid) if "memory_dir" in payload else None

    async def render(self, store: str, plan: Any, record: dict[str, Any]) -> None:
        self.rendered.append((plan.memory_dir, plan.text))
        record["projection"] = {"outcome": "rendered"}

    def run(self, spool: Path) -> int:
        def cycle(batch: list[Any]) -> None:
            asyncio.run(
                worker.run_cycle(
                    "memory",
                    batch,
                    harvest=self.harvest,
                    render=self.render,
                    write_log=self.log.append,
                    deadline=5.0,
                )
            )

        return worker.run_worker("memory", spool, cycle=cycle, write_log=self.log.append)


@pytest.fixture
def spool(tmp_path: Path) -> Path:
    (tmp_path / "state").mkdir(mode=0o700)
    return tmp_path / "state" / "session-end"


def test_a_wave_is_harvested_once_each_and_each_directory_rendered_once(spool: Path) -> None:
    for sid, mdir in (("a", "/m1"), ("b", "/m2"), ("c", "/m1"), ("boom", "/m2")):
        write_spool_file(spool, {"session_id": sid, "memory_dir": mdir})
    rec = _Recorder()

    assert rec.run(spool) == 4
    assert rec.harvested == ["a", "b", "c", "boom"]
    # One render per directory, with its newest successful harvest's text.
    assert sorted(rec.rendered) == [(Path("/m1"), "c"), (Path("/m2"), "b")]
    by_sid = {r["session_id"]: r for r in rec.log}
    assert {r["mode"] for r in rec.log} == {"worker"}
    assert by_sid["boom"]["outcome"] == "error" and "deposit failed" in by_sid["boom"]["error"]
    assert by_sid["c"]["projection"] == {"outcome": "rendered"}
    assert "projection" not in by_sid["a"]  # its directory was rendered with c's text
    assert not worker.pending(spool) and not worker.taken(spool)


def test_a_lost_worker_is_an_error_and_its_session_is_harvested_again(spool: Path) -> None:
    path = write_spool_file(spool, {"session_id": "lost"})
    path.rename(path.with_suffix(worker.TAKEN_SUFFIX))  # a worker took it, then died
    rec = _Recorder()

    rec.run(spool)
    assert rec.harvested == ["lost"]
    lost, done = rec.log
    assert lost["outcome"] == "error" and lost["error"].startswith("worker lost")
    assert done["outcome"] == "ok"


def test_a_session_lost_too_often_is_dropped(spool: Path) -> None:
    path = write_spool_file(spool, {"session_id": "cursed"})
    envelope = json.loads(path.read_text())
    envelope["attempts"] = worker.MAX_ATTEMPTS - 1
    taken = path.with_suffix(worker.TAKEN_SUFFIX)
    taken.write_text(json.dumps(envelope))
    path.unlink()
    rec = _Recorder()

    rec.run(spool)
    assert rec.harvested == []
    (record,) = rec.log
    assert record["outcome"] == "error" and "dropped" in record["error"]
    assert not worker.taken(spool)


def test_a_held_lock_leaves_the_spool_to_its_holder(spool: Path) -> None:
    from particles.api.cli._hook_state import advisory_lock

    write_spool_file(spool, {"session_id": "a"})
    rec = _Recorder()
    with advisory_lock(spool / LOCK_NAME) as held:
        assert held
        assert rec.run(spool) == 0
    assert rec.harvested == [] and len(worker.pending(spool)) == 1
