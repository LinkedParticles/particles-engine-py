"""Resident daemon mode — the in-process task runner."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from particles.api import daemon as daemon_mod
from particles.api.daemon import (
    DaemonTaskStatus,
    _poll_loop,
    consolidation_tick,
    daemon_status,
    set_projection_runner_factory,
    start_daemon,
    stop_daemon,
)
from particles.config import reset_config

#: An mtime comfortably ahead of anything the test tree already carries, so a
#: rewrite is unambiguously "changed" even on a coarse-grained filesystem.
_FUTURE = time.time() + 3600


@pytest.fixture(autouse=True)
def _clean_daemon_runtime() -> Any:
    """Every test starts and ends with an unstarted daemon and no injected factory."""
    daemon_mod._runtime.reset()
    set_projection_runner_factory(None)
    yield
    daemon_mod._runtime.reset()
    set_projection_runner_factory(None)


# ---------------------------------------------------------------------------
# Opt-in: off by default (plain `engine serve` is unchanged)
# ---------------------------------------------------------------------------


def test_start_daemon_is_a_noop_when_disabled() -> None:
    assert start_daemon() is False
    status = daemon_status()
    assert status.enabled is False
    assert status.tasks == []


@pytest.mark.asyncio
async def test_start_daemon_registers_the_consolidation_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARTICLES_DAEMON_ENABLED", "true")
    reset_config()
    monkeypatch.setattr(daemon_mod, "consolidation_tick", _never)
    try:
        assert start_daemon() is True
        status = daemon_status()
        assert status.enabled is True
        assert [task.name for task in status.tasks] == ["consolidation"]
        assert status.tasks[0].interval_seconds == 3600.0
    finally:
        await stop_daemon()


# ---------------------------------------------------------------------------
# Intake watchers on the shared seam
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchers_activate_from_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Activation is derived from config, not from extra on/off switches."""
    inbox = tmp_path / "inbox.txt"
    inbox.write_text("")
    captures = tmp_path / "clippings"
    captures.mkdir()
    monkeypatch.setenv("PARTICLES_DAEMON_ENABLED", "true")
    monkeypatch.setenv("INBOX_FILE_PATH", str(inbox))
    monkeypatch.setenv("PARTICLES_DAEMON_WEB_CLIPPER_DIR", str(captures))
    reset_config()
    monkeypatch.setattr(daemon_mod, "consolidation_tick", _never)
    monkeypatch.setattr(daemon_mod, "_make_inbox_tick", lambda _path: _never)
    monkeypatch.setattr(daemon_mod, "_make_web_clipper_tick", lambda _dir: _never)
    try:
        assert start_daemon() is True
        tasks = {task.name: task for task in daemon_status().tasks}
        assert set(tasks) == {"consolidation", "inbox", "web-clipper"}
        # The inbox watcher reuses inbox.poll_interval_seconds, not a new knob.
        assert tasks["inbox"].interval_seconds == 30.0
        assert tasks["web-clipper"].interval_seconds == 300.0
    finally:
        await stop_daemon()


@pytest.mark.asyncio
async def test_watchers_stay_inactive_without_their_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARTICLES_DAEMON_ENABLED", "true")
    monkeypatch.delenv("INBOX_FILE_PATH", raising=False)
    reset_config()
    monkeypatch.setattr(daemon_mod, "consolidation_tick", _never)
    try:
        assert start_daemon() is True
        assert [task.name for task in daemon_status().tasks] == ["consolidation"]
    finally:
        await stop_daemon()


@pytest.mark.asyncio
async def test_inbox_tick_skips_the_read_until_the_mtime_moves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """mtime-poll: the expensive half only runs when the file actually changed."""
    inbox = tmp_path / "inbox.txt"
    inbox.write_text("https://example.com/one\n")
    calls: list[Path] = []

    async def _fake_process_inbox(_session: Any, path: Path, **_kw: Any) -> dict[str, list[str]]:
        calls.append(path)
        return {"processed": ["https://example.com/one"], "failed": []}

    monkeypatch.setattr("particles.operations.inbox.process_inbox", _fake_process_inbox)
    monkeypatch.setattr("particles.db.session_scope", _fake_session_scope)

    tick = daemon_mod._make_inbox_tick(inbox)
    assert await tick() == "processed 1, failed 0"
    # Nothing changed since the last pass → no second read.
    assert await tick() == "unchanged"
    assert len(calls) == 1

    # A phone append moves the mtime; the next tick picks it up.
    inbox.write_text("https://example.com/one\nhttps://example.com/two\n")
    os.utime(inbox, (_FUTURE, _FUTURE))
    assert await tick() == "processed 1, failed 0"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_inbox_tick_reports_unchanged_when_the_file_is_absent(tmp_path: Path) -> None:
    tick = daemon_mod._make_inbox_tick(tmp_path / "never-created.txt")
    assert await tick() == "unchanged"


@pytest.mark.asyncio
async def test_web_clipper_tick_rescans_only_on_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captures = tmp_path / "clippings"
    captures.mkdir()
    (captures / "a.md").write_text("---\nsource: https://example.com/a\n---\nbody\n")
    calls: list[Path] = []

    async def _fake_scan(_session: Any, captures_dir: Path, **_kw: Any) -> list[tuple[str, str]]:
        calls.append(captures_dir)
        return [("entry-1", "snap-1")]

    monkeypatch.setattr("particles.operations.deposit.deposit_web_clipper", _fake_scan)
    monkeypatch.setattr("particles.db.session_scope", _fake_session_scope)

    tick = daemon_mod._make_web_clipper_tick(captures)
    assert await tick() == "deposited 1"
    assert await tick() == "unchanged"
    assert len(calls) == 1

    (captures / "b.md").write_text("---\nsource: https://example.com/b\n---\nbody\n")
    os.utime(captures / "b.md", (_FUTURE, _FUTURE))
    assert await tick() == "deposited 1"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_web_clipper_tick_tolerates_a_missing_directory(tmp_path: Path) -> None:
    """A not-yet-created captures dir is a disclosed no-op, not a crash loop."""
    tick = daemon_mod._make_web_clipper_tick(tmp_path / "absent")
    assert await tick() == "captures directory absent"


@pytest.mark.asyncio
async def test_start_daemon_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_DAEMON_ENABLED", "true")
    reset_config()
    monkeypatch.setattr(daemon_mod, "consolidation_tick", _never)
    try:
        assert start_daemon() is True
        assert start_daemon() is True
        assert len(daemon_status().tasks) == 1
    finally:
        await stop_daemon()


# ---------------------------------------------------------------------------
# The tick honours the --if-due due-guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidation_tick_passes_if_due(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tick is a scheduler over the due-guard, never a bypass of it."""
    captured: dict[str, Any] = {}

    async def _fake_run_consolidation(_session: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _Report(outcome="skipped", skip_reason="not due: last successful run …")

    monkeypatch.setattr(
        "particles.operations.consolidation.run_consolidation", _fake_run_consolidation
    )
    monkeypatch.setattr("particles.db.session_scope", _fake_session_scope)

    outcome = await consolidation_tick()

    assert captured["if_due"] is True
    assert captured["store"] == "default"
    assert outcome.startswith("skipped: not due")


@pytest.mark.asyncio
async def test_consolidation_tick_reports_failed_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_run_consolidation(_session: Any, **_kwargs: Any) -> Any:
        return _Report(outcome="completed", failed=["census"])

    monkeypatch.setattr(
        "particles.operations.consolidation.run_consolidation", _fake_run_consolidation
    )
    monkeypatch.setattr("particles.db.session_scope", _fake_session_scope)

    assert await consolidation_tick() == "completed, failed pass(es): census"


@pytest.mark.asyncio
async def test_consolidation_tick_injects_the_projection_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a registered factory, pass 6 is a *disclosed* skip, not a silent one."""
    captured: dict[str, Any] = {}

    async def _fake_run_consolidation(_session: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _Report(outcome="completed")

    monkeypatch.setattr(
        "particles.operations.consolidation.run_consolidation", _fake_run_consolidation
    )
    monkeypatch.setattr("particles.db.session_scope", _fake_session_scope)

    await consolidation_tick()
    assert captured["projection_runner"] is None
    assert captured["projection_skip_reason"] == "no projection runner registered (daemon mode)"

    def _factory(store: str) -> tuple[Any, str | None]:
        return (lambda: None), None

    set_projection_runner_factory(_factory)
    await consolidation_tick()
    assert captured["projection_runner"] is not None
    assert captured["projection_skip_reason"] is None


# ---------------------------------------------------------------------------
# A crashed task is disclosed, never silently dead (Consequences)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failing_iteration_is_logged_recorded_and_survived(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One raising iteration must not kill the task — it keeps its cadence."""
    calls: list[int] = []

    async def _flaky() -> str:
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "ok"

    status = DaemonTaskStatus(name="flaky", interval_seconds=0.0)
    with caplog.at_level(logging.ERROR, logger="particles.api.daemon"):
        task = asyncio.create_task(_poll_loop(status, lambda: 0.0, _flaky))
        while status.runs < 2:
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert status.failures == 1
    assert status.runs >= 2
    # The failure is in the log …
    assert "boom" in caplog.text
    # … and the task recovered on the next iteration.
    assert status.last_outcome == "ok"
    assert status.last_error is None


@pytest.mark.asyncio
async def test_a_dead_task_is_marked_crashed_and_makes_the_daemon_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task that dies past the ``except Exception`` net still surfaces in /health."""
    monkeypatch.setenv("PARTICLES_DAEMON_ENABLED", "true")
    reset_config()

    async def _die() -> str:
        raise _Fatal("out of file descriptors")

    monkeypatch.setattr(daemon_mod, "consolidation_tick", _die)
    assert start_daemon() is True
    # Let the task run and die; the done-callback records the crash.
    await asyncio.wait(daemon_mod._runtime.tasks)
    await asyncio.sleep(0)

    status = daemon_status()
    assert status.healthy is False
    assert status.tasks[0].state == "crashed"
    assert status.tasks[0].last_error == "_Fatal: out of file descriptors"


@pytest.mark.asyncio
async def test_stop_daemon_cancels_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_DAEMON_ENABLED", "true")
    reset_config()
    monkeypatch.setattr(daemon_mod, "consolidation_tick", _never)
    assert start_daemon() is True
    await stop_daemon()
    assert daemon_status().enabled is False
    assert daemon_status().tasks == []


# ---------------------------------------------------------------------------
# /health disclosure (/health serves liveness and readiness)
# ---------------------------------------------------------------------------


def test_health_omits_the_daemon_block_when_off() -> None:
    from particles.api.app import app

    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["daemon"] is None


def test_health_reports_degraded_when_a_task_crashed() -> None:
    from particles.api.app import app

    daemon_mod._runtime.enabled = True
    daemon_mod._runtime.statuses["consolidation"] = DaemonTaskStatus(
        name="consolidation",
        interval_seconds=3600.0,
        state="crashed",
        last_error="RuntimeError: boom",
    )
    with TestClient(app) as client:
        body = client.get("/health").json()
    # Still a 200 — the API must not take itself down because a tick died — but
    # the disclosure is machine-readable.
    assert body["status"] == "degraded"
    assert body["daemon"]["healthy"] is False
    assert body["daemon"]["tasks"][0]["last_error"] == "RuntimeError: boom"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Fatal(BaseException):
    """A failure below ``Exception`` — the class ``_poll_loop`` deliberately lets through."""


async def _never() -> str:
    """A tick that blocks forever — registers the task without doing any work."""
    await asyncio.Event().wait()
    return "unreachable"  # pragma: no cover


class _Report:
    """Minimal stand-in for ``ConsolidationReport``."""

    def __init__(
        self, *, outcome: str, skip_reason: str | None = None, failed: list[str] | None = None
    ) -> None:
        self.outcome = outcome
        self.skip_reason = skip_reason
        self._failed = failed or []

    def failed_passes(self) -> list[str]:
        return self._failed


class _FakeSessionScope:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _fake_session_scope(_store: str = "default") -> _FakeSessionScope:
    return _FakeSessionScope()
