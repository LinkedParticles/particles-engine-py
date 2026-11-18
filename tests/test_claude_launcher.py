# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The SessionEnd hand-off launcher, run as the installed command runs it."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from particles.api.cli import _claude_launcher as launcher

LAUNCHER = Path(launcher.__file__)


def _run(tmp_path: Path, payload: bytes, worker: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(LAUNCHER),
            "--spool",
            str(tmp_path / "state" / "session-end"),
            "--log",
            str(tmp_path / "state" / "hooks.jsonl"),
            "--",
            *worker,
        ],
        input=payload,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _log(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "state" / "hooks.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wait_for(path: Path) -> None:
    for _ in range(200):
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def test_it_spools_the_payload_starts_the_worker_and_logs(tmp_path: Path) -> None:
    marker = tmp_path / "worker-ran"
    worker = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()", "--store", "memory"]
    payload = {"session_id": "s-1", "transcript_path": "/t.jsonl", "reason": "other"}
    result = _run(tmp_path, json.dumps(payload).encode(), worker)

    assert result.returncode == 0 and result.stdout == b"" and result.stderr == b""
    _wait_for(marker)
    spool = tmp_path / "state" / "session-end"
    (spooled,) = spool.glob("*.json")
    assert stat.S_IMODE(spooled.stat().st_mode) == 0o600
    assert stat.S_IMODE(spool.stat().st_mode) == 0o700
    envelope = json.loads(spooled.read_text())
    assert envelope["payload"] == payload and envelope["attempts"] == 0
    (record,) = _log(tmp_path)
    assert record["event"] == "session-end-handoff"
    assert record["outcome"] == "handed_off" and record["worker"] == "started"
    assert record["store"] == "memory" and record["session_id"] == "s-1"


def test_a_running_worker_takes_the_file_and_none_is_started(tmp_path: Path) -> None:
    import fcntl

    spool = tmp_path / "state" / "session-end"
    launcher.private_directory(spool)
    fd = os.open(spool / launcher.LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        marker = tmp_path / "worker-ran"
        worker = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
        assert _run(tmp_path, b'{"session_id": "s-2"}', worker).returncode == 0
    finally:
        os.close(fd)
    time.sleep(0.3)
    assert not marker.exists()
    assert len(list(spool.glob("*.json"))) == 1
    (record,) = _log(tmp_path)
    assert record["outcome"] == "handed_off" and record["worker"] == "running"


@pytest.mark.parametrize(
    ("payload", "worker", "logged"),
    [
        (b"x" * (launcher.MAX_PAYLOAD_BYTES + 1), [sys.executable, "-c", "pass"], True),
        (b"not json", [sys.executable, "-c", "pass"], True),
        (b"{}", [], False),  # no worker after --: nothing names the log
    ],
)
def test_every_failure_exits_zero_and_spools_nothing(
    tmp_path: Path, payload: bytes, worker: list[str], logged: bool
) -> None:
    result = _run(tmp_path, payload, worker)
    assert result.returncode == 0 and result.stdout == b""
    assert not list((tmp_path / "state" / "session-end").glob("*.json"))
    if logged:
        (record,) = _log(tmp_path)
        assert record["outcome"] == "error"


def test_a_shared_spool_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir(mode=0o700)
    spool = tmp_path / "state" / "session-end"
    spool.mkdir()
    spool.chmod(0o755)
    _run(tmp_path, b'{"session_id": "s"}', [sys.executable, "-c", "pass"])
    assert not list(spool.glob("*.json"))
    (record,) = _log(tmp_path)
    assert record["outcome"] == "error" and "shared_spool_directory" in str(record["error"])


def test_it_imports_nothing_outside_the_standard_library() -> None:
    import ast

    tree = ast.parse(LAUNCHER.read_text())
    names = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (
            node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")]
        )
    }
    assert names <= set(sys.stdlib_module_names) | {"__future__"}
