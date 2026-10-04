# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dream-cycle consolidation operation.

Pins the deterministic parts with mocked seams: the §3 pass composition and
ordering, the §4 delta-scope watermark computation, the §6 degradation
disclosure ("not probed this run", never "0"), the §8 lockfile protocol
(the kernel-held lock, the holder message, the
fallbacks and mixed versions) and ``--if-due`` guard, the §7
``CONSOLIDATION_RUN`` payload shape + delta report against a prior event, and
the interactive audit's recording seam. The LLM-priced pass internals are the
composed operations' own tests.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations import consolidation as consolidation_mod
from particles.operations.consolidation import (
    ConsolidationReport,
    CycleLock,
    LockHeld,
    acquire_cycle_lock,
    build_run_payload,
    latest_run_event,
    record_audit_run,
    release_cycle_lock,
    render_consolidation_report,
    run_consolidation,
)
from particles.operations.curation.cards import CardKind, CurationCard, gestures_for
from particles.operations.curation.snapshot import CurationQueueResult
from particles.operations.utility_exposure import Exposure, ExposureReader
from particles.store.event_store import OperatorEventType, list_events, record_event
from particles.store.particle_store import (
    get_particle_ids_changed_since,
    insert_particle,
)

NOW = datetime.now(UTC)


def _card(kind: CardKind, *ids: str, diagnostic: str = "diag") -> CurationCard:
    return CurationCard(
        kind=kind,
        particle_ids=list(ids),
        diagnostic=diagnostic,
        suggested_gestures=gestures_for(kind),
    )


def _particle(content: str, asserted_at: datetime, entry_id: str = "e1") -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id="s1")
        ],
        asserted_by="general-extractor",
        asserted_at=asserted_at,
    )


async def _seed_run_event(
    session: AsyncSession,
    *,
    completed_at: datetime,
    started_at: datetime | None = None,
    actor: str = "memory-consolidate",
    failed: bool = False,
    degraded: bool = False,
    cards: dict[str, int] | None = None,
    duplicates_total: int = 0,
    census_status: str | None = None,
) -> None:
    """Write a prior CONSOLIDATION_RUN event in the §7 payload shape.

    ``started_at`` defaults to five minutes before ``completed_at`` — the §4
    delta watermark is the *started_at* (correction v1.74.1), so tests that
    pin the watermark pass it explicitly.
    """
    if started_at is None:
        started_at = completed_at - timedelta(minutes=5)
    payload = {
        "format": 1,
        "store": "default",
        "actor": actor,
        "scope": "store",
        "watermark": None,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "semantic_degraded": degraded,
        "semantic_degraded_reason": "no API key (structural-only run)" if degraded else None,
        "providers": {},
        "passes": [
            {
                "name": "census",
                "status": census_status or ("failed(boom)" if failed else "ran"),
            },
        ],
        "census": {
            "cards": cards or {},
            "duplicate_candidate_pairs_total": duplicates_total,
            **({"skipped": census_status} if census_status else {}),
        },
    }
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.CONSOLIDATION_RUN,
        payload=payload,
    )
    await session.commit()


@pytest.fixture
def cycle_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the cycle lock at tmp and mock every composed pass seam.

    The composed operations have their own tests; here each seam is replaced
    so pass composition, ordering, disclosure, and the run record can be
    asserted hermetically.
    """
    lock_path = tmp_path / "consolidate.lock"
    monkeypatch.setattr(consolidation_mod, "cycle_lock_path", lambda: lock_path)
    # The census runs on every cycle here (0, the pre-cadence
    # behaviour), so the tests below pin the census's internals regardless of
    # the prior runs they seed. TestCensusCadence restores the weekly default.
    get_config().consolidation.census.interval_hours = 0
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        consolidation_mod,
        "reconcile_supersession",
        AsyncMock(return_value={"demoted": 2, "probed": 3, "candidate_pairs": 3}),
    )
    monkeypatch.setattr(
        consolidation_mod,
        "collect_cards",
        AsyncMock(
            return_value=[
                _card(CardKind.CONTRADICTION, "p1"),
                _card(CardKind.INCONSISTENCY, "p2"),
                _card(CardKind.DUPLICATE_PAIR, "p1", "p3"),
                _card(CardKind.STALE, "p4"),
            ]
        ),
    )
    monkeypatch.setattr(consolidation_mod, "_suppressed_keys", AsyncMock(return_value=set()))
    monkeypatch.setattr(
        consolidation_mod,
        "build_curation_queue",
        AsyncMock(
            return_value=CurationQueueResult(cards=[_card(CardKind.CONTRADICTION, "p1")], count=1)
        ),
    )
    # pass 4 persists the collection. The fixture store has no
    # snapshot table rows and the finders are mocked out, so stub the write and
    # assert on the report's pointer instead.
    monkeypatch.setattr(
        consolidation_mod,
        "collect_and_persist",
        AsyncMock(side_effect=lambda _s, **kw: (list(kw["cards"]), "snap-1")),
    )
    # No PENDING snapshots / CONVERSATION entries exist in the empty fixture
    # store, so passes 1 and 5 run against genuinely empty inputs.
    return lock_path


# ---------------------------------------------------------------------------
# Lockfile protocol (§8)
# ---------------------------------------------------------------------------


class TestCycleLock:
    def test_acquire_and_release(self, tmp_path: Path) -> None:
        path = tmp_path / "consolidate.lock"
        lock = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(lock, CycleLock)
        data = json.loads(path.read_text())
        assert data["pid"] == os.getpid()
        assert data["lock"] == "kernel"
        assert data["host"] == socket.gethostname()
        release_cycle_lock(lock)
        # truncated, never unlinked; an empty file is an unheld lock.
        assert path.read_text() == ""
        release_cycle_lock(lock)  # idempotent
        again = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(again, CycleLock)
        release_cycle_lock(again)

    def test_held_in_process_is_not_reclaimed(self, tmp_path: Path) -> None:
        path = tmp_path / "consolidate.lock"
        first = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(first, CycleLock)
        held = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(held, LockHeld)
        assert f"pid {os.getpid()}" in held.reason
        release_cycle_lock(first)

    def test_live_holder_older_than_timeout_is_not_reclaimed(self, tmp_path: Path) -> None:
        # a live cycle three hours old was reclaimed by age.
        path = tmp_path / "consolidate.lock"
        with _holder(path, age_minutes=180, pass_name="extract") as child:
            before = path.read_text()
            held = acquire_cycle_lock(path, timeout_minutes=120)
            assert isinstance(held, LockHeld)
            assert path.read_text() == before  # untouched
            assert f"pid {child.pid}" in held.reason
            assert "pass extract since" in held.reason
            assert "started " in held.reason
            assert held.warning is not None
            assert "held for 3h" in held.warning
            assert f"if it is hung, stop pid {child.pid}" in held.warning

    def test_young_live_holder_has_no_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "consolidate.lock"
        with _holder(path):
            held = acquire_cycle_lock(path, timeout_minutes=120)
            assert isinstance(held, LockHeld)
            assert held.warning is None

    def test_dead_holder_is_reclaimed(self, tmp_path: Path) -> None:
        # The child takes the lock and exits without releasing it.
        path = tmp_path / "consolidate.lock"
        with _holder(path, age_minutes=5) as child:
            child.stdin.close()  # type: ignore[union-attr]
            child.wait(timeout=30)
        assert json.loads(path.read_text())["pid"] == child.pid  # left behind
        lock = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(lock, CycleLock)
        assert json.loads(path.read_text())["pid"] == os.getpid()
        release_cycle_lock(lock)

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX")
    def test_killed_holder_is_reclaimed(self, tmp_path: Path) -> None:
        # The "stopped heartbeat" case on one host: the kernel releases the
        # lock when the holder dies, whatever its description says.
        path = tmp_path / "consolidate.lock"
        with _holder(path) as child:
            child.send_signal(signal.SIGKILL)
            child.wait(timeout=30)
            lock = acquire_cycle_lock(path, timeout_minutes=120)
            assert isinstance(lock, CycleLock)
            release_cycle_lock(lock)

    def test_description_follows_passes_and_heartbeat(self, tmp_path: Path) -> None:
        path = tmp_path / "consolidate.lock"
        lock = acquire_cycle_lock(path, timeout_minutes=120, heartbeat_seconds=0.05)
        assert isinstance(lock, CycleLock)
        lock.set_pass("census")
        # Read as a contender does: the heartbeat thread rewrites the file
        # every 50 ms, and a raw read can race it.
        first = consolidation_mod._read_description(path)
        assert first is not None
        assert first["pass"] == "census"
        # The heartbeat comes from a thread, so it advances while the caller
        # blocks synchronously.
        time.sleep(0.3)
        later = consolidation_mod._read_description(path)
        assert later is not None
        assert later["heartbeat_at"] > first["heartbeat_at"]
        assert later["pass"] == "census"
        release_cycle_lock(lock)

    @pytest.mark.parametrize("kernel", [True, False], ids=["kernel", "fallback"])
    def test_rewrite_never_shows_a_reader_a_partial_description(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kernel: bool
    ) -> None:
        # A contender reads the description while the holder rewrites it. An
        # unreadable one is "no holder" after taking the kernel lock and
        # "stale" in the fallback, so either could take a live cycle's lock.
        if not kernel:

            def no_flock(_fd: int, _op: int) -> None:
                raise OSError(errno.ENOSYS, "flock not supported")

            monkeypatch.setattr(consolidation_mod.fcntl, "flock", no_flock)
        path = tmp_path / "consolidate.lock"
        lock = acquire_cycle_lock(path, timeout_minutes=120, heartbeat_seconds=3600)
        assert isinstance(lock, CycleLock)
        assert (lock.fd is not None) is kernel
        stop = threading.Event()
        unread: list[str] = []

        def _read() -> None:
            while not stop.is_set():
                if consolidation_mod._read_description(path) is None:
                    unread.append(path.read_text())

        reader = threading.Thread(target=_read)
        reader.start()
        try:
            # Alternate long and short pass names so the file shrinks as well
            # as grows: a shrink is what leaves a stale tail behind.
            for i in range(2000):
                lock.set_pass("x" * 40 if i % 2 else "census")
        finally:
            stop.set()
            reader.join()
        assert unread == []
        assert json.loads(path.read_text())["pass"] == "x" * 40
        release_cycle_lock(lock)

    def test_corrupt_description_is_still_malformed_after_rereads(self, tmp_path: Path) -> None:
        # The torn-read retry must not turn a crashed writer's garbage into a
        # lock nobody can reclaim: malformed is stale.
        path = tmp_path / "consolidate.lock"
        path.write_text('{"pid": 12')
        assert consolidation_mod._read_description(path) is None
        assert consolidation_mod._lock_is_stale(path, timeout_minutes=120)

    def test_descriptions_are_per_lock_path(self, tmp_path: Path) -> None:
        # The benchmark runs concurrent cycles, one lock path each, in one dir.
        a = acquire_cycle_lock(tmp_path / "q1.consolidate.lock", timeout_minutes=120)
        b = acquire_cycle_lock(tmp_path / "q2.consolidate.lock", timeout_minutes=120)
        assert isinstance(a, CycleLock) and isinstance(b, CycleLock)
        a.set_pass("extract")
        b.set_pass("census")
        assert json.loads((tmp_path / "q1.consolidate.lock").read_text())["pass"] == "extract"
        assert json.loads((tmp_path / "q2.consolidate.lock").read_text())["pass"] == "census"
        release_cycle_lock(a)
        release_cycle_lock(b)

    def test_other_host_with_fresh_heartbeat_blocks(self, tmp_path: Path) -> None:
        # a container's kernel lock may not be visible here.
        path = tmp_path / "consolidate.lock"
        path.write_text(json.dumps(_description(host="container-1", beat_minutes_ago=2)))
        held = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(held, LockHeld)
        assert "on container-1" in held.reason
        assert json.loads(path.read_text())["host"] == "container-1"

    def test_other_host_with_stopped_heartbeat_is_reclaimed(self, tmp_path: Path) -> None:
        path = tmp_path / "consolidate.lock"
        path.write_text(json.dumps(_description(host="container-1", beat_minutes_ago=30)))
        lock = acquire_cycle_lock(path, timeout_minutes=120, heartbeat_stale_minutes=10)
        assert isinstance(lock, CycleLock)
        assert json.loads(path.read_text())["host"] == socket.gethostname()
        release_cycle_lock(lock)

    def test_this_host_description_never_blocks(self, tmp_path: Path) -> None:
        # A crash on this host leaves a fresh-looking description; the kernel
        # lock already said nobody holds it.
        path = tmp_path / "consolidate.lock"
        path.write_text(json.dumps(_description(host=socket.gethostname(), beat_minutes_ago=0)))
        lock = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(lock, CycleLock)
        release_cycle_lock(lock)

    def test_pre_change_live_holder_is_honoured(self, tmp_path: Path) -> None:
        # a pre-change binary writes pid + started_at, no "lock".
        path = tmp_path / "consolidate.lock"
        path.write_text(json.dumps({"pid": os.getpid(), "started_at": _ago(5)}))
        held = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(held, LockHeld)
        assert "lock" not in json.loads(path.read_text())

    @pytest.mark.parametrize(
        "data",
        [
            {"pid": 2**30, "started_at": "now"},  # dead pid
            {"pid": os.getpid(), "started_at": "old"},  # older than the timeout
        ],
    )
    def test_pre_change_stale_holder_is_replaced(
        self, tmp_path: Path, data: dict[str, Any]
    ) -> None:
        path = tmp_path / "consolidate.lock"
        started = _ago(5) if data["started_at"] == "now" else _ago(300)
        path.write_text(json.dumps({"pid": data["pid"], "started_at": started}))
        lock = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(lock, CycleLock)
        release_cycle_lock(lock)

    def test_pre_change_contender_honours_a_new_holder(self, tmp_path: Path) -> None:
        # The reverse direction: the pre-change staleness test, run against a
        # new holder's file, sees a live lock; against a released one, stale.
        path = tmp_path / "consolidate.lock"
        lock = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(lock, CycleLock)
        assert not _pre_change_is_stale(path, timeout_minutes=120)
        release_cycle_lock(lock)
        assert _pre_change_is_stale(path, timeout_minutes=120)

    def test_replaced_lock_file_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "consolidate.lock"
        lock = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(lock, CycleLock)
        path.unlink()  # what a pre-change contender does past its timeout
        path.write_text(json.dumps({"pid": 4242, "started_at": _ago(0)}))
        with caplog.at_level("ERROR", logger=consolidation_mod.log.name):
            lock.beat()
        assert "was replaced" in caplog.text
        assert "4242" in caplog.text
        release_cycle_lock(lock)

    def test_without_flock_falls_back_to_pid_and_age(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_flock(_fd: int, _op: int) -> None:
            raise OSError(errno.ENOSYS, "flock not supported")

        monkeypatch.setattr(consolidation_mod.fcntl, "flock", no_flock)
        path = tmp_path / "consolidate.lock"
        # A live pid older than the timeout is reclaimed, as.
        path.write_text(json.dumps({"pid": os.getpid(), "started_at": _ago(300)}))
        lock = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(lock, CycleLock)
        assert lock.fd is None
        assert json.loads(path.read_text())["lock"] == "fallback"
        # Held and young: skipped.
        assert isinstance(acquire_cycle_lock(path, timeout_minutes=120), LockHeld)
        release_cycle_lock(lock)
        assert not path.exists()
        # A dead pid is reclaimed.
        path.write_text(json.dumps({"pid": 2**30, "started_at": _ago(1)}))
        again = acquire_cycle_lock(path, timeout_minutes=120)
        assert isinstance(again, CycleLock)
        release_cycle_lock(again)

    def test_fallback_reclaim_race_never_deletes_a_fresh_rival_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # TOCTOU (correction v1.74.1), still live in the fallback: both
        # contenders judge the old lock stale; the loser must back off.
        def no_flock(_fd: int, _op: int) -> None:
            raise OSError(errno.ENOSYS, "flock not supported")

        monkeypatch.setattr(consolidation_mod.fcntl, "flock", no_flock)
        path = tmp_path / "consolidate.lock"
        path.write_text(json.dumps({"pid": 2**30, "started_at": _ago(0)}))
        rival = json.dumps({"pid": os.getpid(), "started_at": _ago(0)})

        real_is_stale = consolidation_mod._lock_is_stale

        def racing_is_stale(p: Path, timeout_minutes: int) -> bool:
            verdict = real_is_stale(p, timeout_minutes)
            p.unlink()
            p.write_text(rival)
            return verdict

        monkeypatch.setattr(consolidation_mod, "_lock_is_stale", racing_is_stale)
        assert isinstance(acquire_cycle_lock(path, timeout_minutes=120), LockHeld)
        assert json.loads(path.read_text()) == json.loads(rival)

    @pytest.mark.asyncio
    async def test_held_lock_skips_run(self, db_session: AsyncSession, cycle_env: Path) -> None:
        held = acquire_cycle_lock(cycle_env, timeout_minutes=120)
        assert isinstance(held, CycleLock)
        report = await run_consolidation(db_session)
        assert report.outcome == "skipped"
        assert report.skip_reason is not None
        assert "already running" in report.skip_reason
        assert f"pid {os.getpid()}" in report.skip_reason
        assert report.lock_warning is None
        # No run record was written for a lock skip.
        assert await latest_run_event(db_session) is None
        release_cycle_lock(held)

    @pytest.mark.asyncio
    async def test_long_hold_sets_the_warning(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        held = acquire_cycle_lock(cycle_env, timeout_minutes=120)
        assert isinstance(held, CycleLock)
        held.info["started_at"] = _ago(500)
        held.write_description()
        report = await run_consolidation(db_session)
        assert report.outcome == "skipped"
        assert report.lock_warning is not None
        assert "held for 8h" in report.lock_warning
        release_cycle_lock(held)


def _ago(minutes: float) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()


def _description(*, host: str, beat_minutes_ago: float) -> dict[str, Any]:
    return {
        "pid": 4242,
        "started_at": _ago(beat_minutes_ago + 60),
        "host": host,
        "heartbeat_at": _ago(beat_minutes_ago),
        "lock": "kernel",
    }


def _pre_change_is_stale(path: Path, timeout_minutes: int) -> bool:
    """The staleness test from before the kernel lock, verbatim: an old binary."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    pid = data.get("pid")
    if isinstance(pid, int) and not consolidation_mod._pid_alive(pid):
        return True
    try:
        started = datetime.fromisoformat(str(data.get("started_at")))
    except (TypeError, ValueError):
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return datetime.now(UTC) - started > timedelta(minutes=timeout_minutes)


#: A holder in a separate process, so the kernel lock is really another
#: process's. It acquires, backdates ``started_at``, prints "ready", then holds
#: until its stdin closes, exiting WITHOUT releasing.
_HOLDER_SCRIPT = """
import json, os, sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from particles.operations.consolidation import CycleLock, acquire_cycle_lock
path, age, pass_name = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
lock = acquire_cycle_lock(path, timeout_minutes=120, heartbeat_seconds=3600)
assert isinstance(lock, CycleLock), lock
lock.info["started_at"] = (datetime.now(UTC) - timedelta(minutes=age)).isoformat()
if pass_name:
    lock.set_pass(pass_name)
lock.write_description()
print("ready", flush=True)
sys.stdin.read()
os._exit(0)
"""


@contextlib.contextmanager
def _holder(
    path: Path, *, age_minutes: float = 1, pass_name: str = ""
) -> Iterator[subprocess.Popen[str]]:
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, str(path), str(age_minutes), pass_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=30)


# ---------------------------------------------------------------------------
# --if-due (§2)
# ---------------------------------------------------------------------------


class TestIfDue:
    @pytest.mark.asyncio
    async def test_young_successful_run_skips(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=1))
        report = await run_consolidation(db_session, if_due=True)
        assert report.outcome == "skipped"
        assert report.skip_reason is not None
        assert "not due" in report.skip_reason

    @pytest.mark.asyncio
    async def test_old_run_is_due(self, db_session: AsyncSession, cycle_env: Path) -> None:
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=30))
        report = await run_consolidation(db_session, if_due=True)
        assert report.outcome == "ran"

    @pytest.mark.asyncio
    async def test_failed_run_does_not_satisfy_if_due(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        # A young FAILED run does not count as "last successful" — still due.
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=1), failed=True)
        report = await run_consolidation(db_session, if_due=True)
        assert report.outcome == "ran"

    @pytest.mark.asyncio
    async def test_first_run_is_always_due(self, db_session: AsyncSession, cycle_env: Path) -> None:
        report = await run_consolidation(db_session, if_due=True)
        assert report.outcome == "ran"

    @pytest.mark.asyncio
    async def test_audit_event_does_not_satisfy_if_due(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        # Correction v1.74.1: `particles audit` writes the same event type but
        # runs none of the cross-session passes — it must not satisfy cadence.
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=1), actor="audit")
        report = await run_consolidation(db_session, if_due=True)
        assert report.outcome == "ran"

    @pytest.mark.asyncio
    async def test_degraded_run_still_satisfies_if_due(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        # Deliberate (correction v1.74.1): a disclosed structural-only run
        # counts for cadence, so a key-less setup does not hot-loop hourly.
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=1), degraded=True)
        report = await run_consolidation(db_session, if_due=True)
        assert report.outcome == "skipped"
        assert report.skip_reason is not None
        assert "not due" in report.skip_reason


# ---------------------------------------------------------------------------
# Pass composition + ordering (§3)
# ---------------------------------------------------------------------------


class TestPassComposition:
    @pytest.mark.asyncio
    async def test_fixed_pass_order(self, db_session: AsyncSession, cycle_env: Path) -> None:
        report = await run_consolidation(db_session)
        assert [p.name for p in report.passes] == [
            # pass 0.5 — ahead of extract, so a rule file edited today
            # is re-snapshotted, extracted, and reconciled in the SAME run.
            "refresh",
            "extract",
            "reconcile",
            "reconcile_updates",
            "reanchor",  # pass 2c: after both sweeps, before the census
            "census",
            "disclose",  # pass 3b: zero-LLM, runs whatever the census did
            "curation",
            "utility",
            "abstraction",  # pass 5b — skipped (default off) but slotted
            "projection",
            "measure",  # pass 6b: zero-LLM, read-only, every night
        ]
        assert report.outcome == "ran"
        assert report.completed_at is not None
        assert report.reconcile_demoted == 2
        # Reconcile's replacement-signal probes are its LLM spend.
        assert next(p for p in report.passes if p.name == "reconcile").llm_calls == 3

    @pytest.mark.asyncio
    async def test_census_counts_and_queue_reuse(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        report = await run_consolidation(db_session)
        assert report.card_counts == {
            "contradiction": 1,
            "inconsistency": 1,
            "duplicate_pair": 1,
            "stale": 1,
        }
        assert report.headline_contradictions == 2
        assert report.duplicate_candidate_pairs_total == 1
        # Pass 4 ranks the SAME collection pass 3 paid for: collect_cards ran
        # exactly once and build_curation_queue received cards=..., not None.
        assert consolidation_mod.collect_cards.await_count == 1  # type: ignore[attr-defined]
        queue_kwargs = consolidation_mod.build_curation_queue.call_args.kwargs  # type: ignore[attr-defined]
        assert len(queue_kwargs["cards"]) == 4
        assert report.curation_queue_total == 4
        # the run record points at the collection this run persisted.
        assert report.curation_snapshot_id == "snap-1"
        assert len(report.curation_queue) == 1

    @pytest.mark.asyncio
    async def test_projection_runner_injected(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        async def runner() -> dict[str, Any]:
            return {"rendered": 1}

        report = await run_consolidation(db_session, projection_runner=runner)
        assert report.projection == {"rendered": 1}
        assert next(p for p in report.passes if p.name == "projection").status == "ran"

    @pytest.mark.asyncio
    async def test_projection_skip_reason_disclosed(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        report = await run_consolidation(
            db_session, projection_skip_reason="agent_memory.projection.enabled is false"
        )
        projection = next(p for p in report.passes if p.name == "projection")
        assert projection.status == "skipped"
        assert projection.detail == "agent_memory.projection.enabled is false"

    @pytest.mark.asyncio
    async def test_pass_failure_continues_and_reports(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            consolidation_mod, "collect_cards", AsyncMock(side_effect=RuntimeError("boom"))
        )

        async def runner() -> dict[str, Any]:
            return {"rendered": 1}

        report = await run_consolidation(db_session, projection_runner=runner)
        census = next(p for p in report.passes if p.name == "census")
        assert census.status == "failed"
        assert census.detail is not None and "boom" in census.detail
        # Curation cannot rank a collection that never happened — disclosed.
        curation = next(p for p in report.passes if p.name == "curation")
        assert curation.status == "skipped"
        # The zero-LLM tail still ran and the run record was still written (§8).
        assert report.projection == {"rendered": 1}
        assert report.failed_passes() == ["census"]
        event = await latest_run_event(db_session)
        assert event is not None
        passes = (event.payload or {})["passes"]
        assert any(str(p["status"]).startswith("failed(") for p in passes)

    @pytest.mark.asyncio
    async def test_lock_released_after_run(self, db_session: AsyncSession, cycle_env: Path) -> None:
        await run_consolidation(db_session)
        assert cycle_env.read_text() == ""  # released: truncated, not unlinked
        again = acquire_cycle_lock(cycle_env, timeout_minutes=120)
        assert isinstance(again, CycleLock)
        release_cycle_lock(again)


# ---------------------------------------------------------------------------
# Degradation disclosure (§6)
# ---------------------------------------------------------------------------


class TestDegradation:
    @pytest.mark.asyncio
    async def test_no_key_degrades_and_discloses(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        report = await run_consolidation(db_session)
        assert report.semantic_degraded is True
        assert report.semantic_degraded_reason == "no API key (structural-only run)"
        extract = next(p for p in report.passes if p.name == "extract")
        assert extract.status == "skipped"
        assert extract.detail is not None and "LLM-priced" in extract.detail
        # Census still ran — structural finders only, no probe control.
        kwargs = consolidation_mod.collect_cards.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["semantic"] is False
        assert kwargs["contradiction_probe"] is None
        # The run record discloses the degradation.
        event = await latest_run_event(db_session)
        assert event is not None
        assert (event.payload or {})["semantic_degraded"] is True

    @pytest.mark.asyncio
    async def test_structural_only_flag(self, db_session: AsyncSession, cycle_env: Path) -> None:
        report = await run_consolidation(db_session, structural_only=True)
        assert report.semantic_degraded is True
        assert report.semantic_degraded_reason == "--structural-only"

    @pytest.mark.asyncio
    async def test_structural_only_makes_zero_reconcile_probes(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        # Correction v1.74.1: pass 2 is probe-bearing (one semantic_lint call
        # per candidate pair). --structural-only must skip it with a
        # disclosure — never run it probe-less behind a clean bill.
        report = await run_consolidation(db_session, structural_only=True)
        reconcile = next(p for p in report.passes if p.name == "reconcile")
        assert reconcile.status == "skipped"
        assert reconcile.detail is not None and "LLM-priced" in reconcile.detail
        assert reconcile.llm_calls == 0
        assert consolidation_mod.reconcile_supersession.await_count == 0  # type: ignore[attr-defined]
        rendered = render_consolidation_report(report)
        assert "pass skipped: reconcile" in rendered

    @pytest.mark.asyncio
    async def test_no_key_makes_zero_reconcile_probes(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        report = await run_consolidation(db_session)
        reconcile = next(p for p in report.passes if p.name == "reconcile")
        assert reconcile.status == "skipped"
        assert reconcile.llm_calls == 0
        assert consolidation_mod.reconcile_supersession.await_count == 0  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_degraded_render_reads_not_probed(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        report = await run_consolidation(db_session, structural_only=True)
        rendered = render_consolidation_report(report)
        assert "not probed this run" in rendered
        assert "contradictions   0" not in rendered  # §6: never a silent "0"
        assert "semantic passes skipped: --structural-only" in rendered

    @pytest.mark.asyncio
    async def test_semantic_run_passes_probe_control(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        await run_consolidation(db_session)
        kwargs = consolidation_mod.collect_cards.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["semantic"] is True
        control = kwargs["contradiction_probe"]
        assert control is not None
        assert control.max_probes == 50  # audit.max_contradiction_probes default


# ---------------------------------------------------------------------------
# Delta scope (§4)
# ---------------------------------------------------------------------------


class TestDeltaScope:
    @pytest.mark.asyncio
    async def test_changed_since_watermark(self, db_session: AsyncSession) -> None:
        watermark = NOW - timedelta(days=1)
        old = _particle("old", NOW - timedelta(days=5))
        new = _particle("new", NOW - timedelta(hours=2))
        await insert_particle(db_session, old)
        await insert_particle(db_session, new)
        await db_session.commit()
        changed = await get_particle_ids_changed_since(db_session, watermark)
        assert changed == {new.id}

    @pytest.mark.asyncio
    async def test_retired_since_watermark_counts_as_modified(
        self, db_session: AsyncSession
    ) -> None:
        from particles.core.status import Status, StatusReason
        from particles.store.particle_store import update_particle_status

        watermark = NOW - timedelta(days=1)
        old = _particle("old but retracted today", NOW - timedelta(days=5))
        await insert_particle(db_session, old)
        await db_session.commit()
        await update_particle_status(
            db_session, old.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()
        changed = await get_particle_ids_changed_since(db_session, watermark)
        assert old.id in changed

    @pytest.mark.asyncio
    async def test_first_run_is_store_wide(self, db_session: AsyncSession, cycle_env: Path) -> None:
        report = await run_consolidation(db_session, scope="delta")
        assert report.effective_scope == "store"
        assert report.watermark is None
        control = consolidation_mod.collect_cards.call_args.kwargs["contradiction_probe"]  # type: ignore[attr-defined]
        assert control.scope_particle_ids is None

    @pytest.mark.asyncio
    async def test_delta_run_scopes_to_watermark(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        # The watermark is the previous eligible run's STARTED_AT (correction
        # v1.74.1) — the instant the scope is computed from.
        completed = NOW - timedelta(hours=25)
        started = completed - timedelta(minutes=5)
        await _seed_run_event(db_session, completed_at=completed, started_at=started)
        old = _particle("before watermark", NOW - timedelta(days=3))
        new = _particle("after watermark", NOW - timedelta(hours=1))
        await insert_particle(db_session, old)
        await insert_particle(db_session, new)
        await db_session.commit()

        report = await run_consolidation(db_session, scope="delta")
        assert report.effective_scope == "delta"
        assert report.watermark == started
        control = consolidation_mod.collect_cards.call_args.kwargs["contradiction_probe"]  # type: ignore[attr-defined]
        assert control.scope_particle_ids == frozenset({new.id})
        assert report.scope_particle_count == 1

    @pytest.mark.asyncio
    async def test_delta_window_opens_at_prior_started_at(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        # A particle asserted BETWEEN the prior run's started_at and its
        # completed_at (e.g. a SessionEnd harvest landing mid-cycle) must be
        # in the next run's scope — under the old completed_at basis it was
        # in no run's scope, ever (correction v1.74.1).
        completed = NOW - timedelta(hours=25)
        started = completed - timedelta(minutes=50)
        await _seed_run_event(db_session, completed_at=completed, started_at=started)
        mid_cycle = _particle("landed mid-run", completed - timedelta(minutes=20))
        await insert_particle(db_session, mid_cycle)
        await db_session.commit()

        report = await run_consolidation(db_session, scope="delta")
        assert report.effective_scope == "delta"
        assert report.watermark == started
        control = consolidation_mod.collect_cards.call_args.kwargs["contradiction_probe"]  # type: ignore[attr-defined]
        assert control.scope_particle_ids == frozenset({mid_cycle.id})

    @pytest.mark.asyncio
    async def test_pass1_output_lands_in_same_run_scope(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The scope is computed AFTER pass 1 (correction v1.74.1): a particle
        # extraction just minted self-includes via asserted_at > watermark.
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=25))
        minted = _particle("minted by pass 1", NOW)

        async def fake_extract(session: AsyncSession, report: ConsolidationReport) -> int:
            await insert_particle(session, minted)
            await session.commit()
            report.pending_extracted += 1
            return 1

        monkeypatch.setattr(consolidation_mod, "_pass_extract", fake_extract)
        report = await run_consolidation(db_session, scope="delta")
        assert report.effective_scope == "delta"
        assert report.pending_extracted == 1
        control = consolidation_mod.collect_cards.call_args.kwargs["contradiction_probe"]  # type: ignore[attr-defined]
        assert minted.id in control.scope_particle_ids

    @pytest.mark.asyncio
    async def test_degraded_run_is_not_watermark_eligible(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        # A structural-only night must not convert its disclosed "not probed
        # this run" into "never probed" — the watermark stays at the last
        # NON-degraded successful run (correction v1.74.1).
        eligible_completed = NOW - timedelta(hours=50)
        eligible_started = eligible_completed - timedelta(minutes=5)
        await _seed_run_event(db_session, completed_at=eligible_completed)
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=25), degraded=True)

        report = await run_consolidation(db_session, scope="delta")
        assert report.effective_scope == "delta"
        assert report.watermark == eligible_started

    @pytest.mark.asyncio
    async def test_only_degraded_prior_runs_store_wide(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=25), degraded=True)
        report = await run_consolidation(db_session, scope="delta")
        assert report.effective_scope == "store"
        assert report.watermark is None

    @pytest.mark.asyncio
    async def test_audit_event_is_not_watermark_eligible(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        # An interactive audit's event stays in the log (delta report) but
        # never becomes the delta watermark (correction v1.74.1).
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=1), actor="audit")
        report = await run_consolidation(db_session, scope="delta")
        assert report.effective_scope == "store"
        assert report.watermark is None

    @pytest.mark.asyncio
    async def test_scope_store_overrides_delta(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=25))
        report = await run_consolidation(db_session, scope="store")
        assert report.effective_scope == "store"
        control = consolidation_mod.collect_cards.call_args.kwargs["contradiction_probe"]  # type: ignore[attr-defined]
        assert control.scope_particle_ids is None


# ---------------------------------------------------------------------------
# Census cadence
# ---------------------------------------------------------------------------


def _census_pass(report: ConsolidationReport) -> Any:
    return next(p for p in report.passes if p.name == "census")


class TestCensusCadence:
    """Pass 3 runs every ``consolidation.census.interval_hours``, not every night."""

    @pytest.fixture(autouse=True)
    def weekly(self, cycle_env: Path) -> None:
        # cycle_env pins the every-run census; these tests pin the default.
        get_config().consolidation.census.interval_hours = 168

    @pytest.mark.asyncio
    async def test_second_run_inside_the_interval_makes_no_census_call(
        self, db_session: AsyncSession
    ) -> None:
        first = await run_consolidation(db_session)
        second = await run_consolidation(db_session)

        assert _census_pass(first).status == "ran"
        census = _census_pass(second)
        assert census.status == "skipped"
        assert census.llm_calls == 0
        # The finders ran once, for the first run only.
        assert consolidation_mod.collect_cards.await_count == 1  # type: ignore[attr-defined]
        assert second.census_skipped is not None
        assert second.census_skipped.startswith("census skipped: last ran ")
        assert "next due " in second.census_skipped
        assert second.census_last_ran_at == first.started_at
        assert second.census_next_due_at == first.started_at + timedelta(hours=168)

        rendered = render_consolidation_report(second)
        assert second.census_skipped in rendered
        assert "pass skipped: census" not in rendered  # said once, in the headline
        # The last census's counts are shown as that census's, never tonight's.
        assert "as of that census: 2 contradictions, 1 duplicate pair, 1 stale" in rendered
        assert "contradictions   " not in rendered

        event = await latest_run_event(db_session)
        assert event is not None
        payload = event.payload or {}
        assert payload["census"]["skipped"] == second.census_skipped
        census_entry = next(p for p in payload["passes"] if p["name"] == "census")
        assert census_entry["status"] == f"skipped({second.census_skipped})"
        assert census_entry["llm_calls"] == 0

    @pytest.mark.asyncio
    async def test_a_run_after_the_interval_runs_the_census(self, db_session: AsyncSession) -> None:
        await _seed_run_event(db_session, completed_at=NOW - timedelta(days=8))
        report = await run_consolidation(db_session)
        assert _census_pass(report).status == "ran"
        assert report.census_skipped is None
        assert consolidation_mod.collect_cards.await_count == 1  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_first_run_on_a_store_runs_the_census(self, db_session: AsyncSession) -> None:
        # A nightly record exists, but none of its runs ran the census.
        await _seed_run_event(
            db_session,
            completed_at=NOW - timedelta(hours=2),
            census_status="skipped(consolidation.census.enabled is false)",
        )
        report = await run_consolidation(db_session)
        assert _census_pass(report).status == "ran"
        assert report.census_scope == "store"

    @pytest.mark.asyncio
    async def test_census_window_opens_at_the_last_census_not_the_last_run(
        self, db_session: AsyncSession
    ) -> None:
        # Census eight days ago, then a nightly run that skipped it. The census
        # must probe everything since ITS last run, or the skipped nights'
        # beliefs would never be probed by any census.
        census_started = NOW - timedelta(days=8, minutes=5)
        await _seed_run_event(
            db_session, completed_at=NOW - timedelta(days=8), started_at=census_started
        )
        nightly_started = NOW - timedelta(days=1, minutes=5)
        await _seed_run_event(
            db_session,
            completed_at=NOW - timedelta(days=1),
            started_at=nightly_started,
            census_status="skipped(census skipped: not due)",
        )
        skipped_night = _particle("written on a skipped night", NOW - timedelta(days=4))
        tonight = _particle("written today", NOW - timedelta(hours=1))
        before = _particle("before the last census", NOW - timedelta(days=10))
        for particle in (skipped_night, tonight, before):
            await insert_particle(db_session, particle)
        await db_session.commit()

        report = await run_consolidation(db_session, scope="delta")
        assert report.watermark == nightly_started  # the nightly passes' window
        assert report.census_scope == "delta"
        assert report.census_watermark == census_started
        control = consolidation_mod.collect_cards.call_args.kwargs["contradiction_probe"]  # type: ignore[attr-defined]
        assert control.scope_particle_ids == frozenset({skipped_night.id, tonight.id})
        assert report.census_scope_particle_count == 2
        assert "census scope: delta since the last census" in render_consolidation_report(report)

    @pytest.mark.asyncio
    async def test_a_degraded_census_does_not_satisfy_the_cadence(
        self, db_session: AsyncSession
    ) -> None:
        # A structural-only census probed nothing, so it is not the last census.
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=2), degraded=True)
        report = await run_consolidation(db_session)
        assert _census_pass(report).status == "ran"

    @pytest.mark.asyncio
    async def test_an_audit_census_does_not_satisfy_the_cadence(
        self, db_session: AsyncSession
    ) -> None:
        # `particles audit` records a census too; it is not this verb's census.
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=2), actor="audit")
        report = await run_consolidation(db_session)
        assert _census_pass(report).status == "ran"

    @pytest.mark.asyncio
    async def test_scope_store_runs_the_census_inside_the_interval(
        self, db_session: AsyncSession
    ) -> None:
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=2))
        report = await run_consolidation(db_session, scope="store")
        assert _census_pass(report).status == "ran"
        assert report.census_scope == "store"

    @pytest.mark.asyncio
    async def test_disabled_never_runs_the_census(self, db_session: AsyncSession) -> None:
        get_config().consolidation.census.enabled = False
        report = await run_consolidation(db_session, scope="store")
        census = _census_pass(report)
        assert census.status == "skipped"
        assert census.detail == "consolidation.census.enabled is false"
        assert consolidation_mod.collect_cards.await_count == 0  # type: ignore[attr-defined]
        # No collection is stored in this fixture, so pass 4 says so.
        curation = next(p for p in report.passes if p.name == "curation")
        assert curation.status == "skipped"
        assert curation.detail is not None and "no card collection is stored" in curation.detail

    @pytest.mark.asyncio
    async def test_structural_only_keeps_its_meaning_on_a_due_night(
        self, db_session: AsyncSession
    ) -> None:
        report = await run_consolidation(db_session, structural_only=True)
        assert _census_pass(report).status == "ran"
        kwargs = consolidation_mod.collect_cards.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["semantic"] is False
        assert kwargs["contradiction_probe"] is None

    @pytest.mark.asyncio
    async def test_skipped_night_has_no_deltas_and_the_next_census_compares_with_the_last(
        self, db_session: AsyncSession
    ) -> None:
        await _seed_run_event(
            db_session,
            completed_at=NOW - timedelta(days=8),
            cards={"contradiction": 1, "inconsistency": 0, "stale": 3},
            duplicates_total=6,
        )
        await _seed_run_event(
            db_session,
            completed_at=NOW - timedelta(days=1),
            census_status="skipped(census skipped: not due)",
        )
        report = await run_consolidation(db_session)
        assert _census_pass(report).status == "ran"
        # Compared with the census eight days ago, not the empty skipped night.
        assert report.deltas == {"contradictions": 1, "duplicates": -5, "stale": -2}

        get_config().consolidation.census.interval_hours = 168
        skipped = await run_consolidation(db_session)
        assert skipped.census_skipped is not None
        assert skipped.deltas == {}


class TestQueueOnASkippedCensus:
    """Pass 4 serves the last census's stored cards between censuses."""

    @pytest.mark.asyncio
    async def test_the_queue_still_has_cards_on_a_skipped_run(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations.curation import session as session_mod
        from particles.operations.curation import snapshot as snapshot_mod

        get_config().consolidation.census.interval_hours = 168
        # The real snapshot write and queue read, over real ACTIVE beliefs.
        monkeypatch.setattr(
            consolidation_mod, "collect_and_persist", snapshot_mod.collect_and_persist
        )
        monkeypatch.setattr(
            consolidation_mod, "build_curation_queue", session_mod.build_curation_queue
        )
        monkeypatch.setattr(consolidation_mod, "_suppressed_keys", session_mod._suppressed_keys)
        first_belief = _particle("the build uses uv", NOW - timedelta(days=40))
        second_belief = _particle("the build uses uv sync", NOW - timedelta(days=39))
        for particle in (first_belief, second_belief):
            await insert_particle(db_session, particle)
        await db_session.commit()
        monkeypatch.setattr(
            consolidation_mod,
            "collect_cards",
            AsyncMock(
                return_value=[
                    _card(CardKind.STALE, first_belief.id),
                    _card(CardKind.DUPLICATE_PAIR, first_belief.id, second_belief.id),
                ]
            ),
        )

        census_night = await run_consolidation(db_session)
        assert census_night.curation_snapshot_id is not None
        assert census_night.curation_queue

        skipped_night = await run_consolidation(db_session)
        assert _census_pass(skipped_night).status == "skipped"
        assert consolidation_mod.collect_cards.await_count == 1  # type: ignore[attr-defined]
        curation = next(p for p in skipped_night.passes if p.name == "curation")
        assert curation.status == "ran"
        # The same stored collection, ranked live: still the morning's worklist.
        assert skipped_night.curation_snapshot_id == census_night.curation_snapshot_id
        assert skipped_night.curation_snapshot_built_at is not None
        assert skipped_night.curation_queue_total == 2
        assert len(skipped_night.curation_queue) == 2
        rendered = render_consolidation_report(skipped_night)
        assert "Curation queue — top 2 of 2 (from the census of " in rendered

    @pytest.mark.asyncio
    async def test_a_skipped_night_never_collects_live(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        get_config().consolidation.census.interval_hours = 168
        get_config().curation.snapshot_enabled = False
        await _seed_run_event(db_session, completed_at=NOW - timedelta(hours=2))
        report = await run_consolidation(db_session)
        curation = next(p for p in report.passes if p.name == "curation")
        assert curation.status == "skipped"
        assert curation.detail is not None and "snapshot_enabled is false" in curation.detail
        assert consolidation_mod.collect_cards.await_count == 0  # type: ignore[attr-defined]
        assert consolidation_mod.build_curation_queue.await_count == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Pass 2 probe cap (correction v1.74.1 — consolidation.max_reconcile_probes)
# ---------------------------------------------------------------------------


class TestReconcileProbeCap:
    @pytest.mark.asyncio
    async def test_cap_truncates_highest_similarity_first(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        import particles.operations.reconcile as reconcile_mod
        from particles.config import get_config
        from particles.operations.reconcile import reconcile_supersession

        get_config().consolidation.max_reconcile_probes = 2
        get_config().extraction.similarity_threshold = 0.5

        monkeypatch.setattr(
            "particles.operations.version_guard.assert_store_schema_current",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            reconcile_mod,
            "iter_supersession_entry_pairs",
            AsyncMock(return_value=[("sup-e", "sub-e")]),
        )
        sup = _particle("the superseding claim", NOW, entry_id="sup-e")
        subs = [_particle(f"old claim {i}", NOW, entry_id="sub-e") for i in range(4)]
        sims = [0.95, 0.90, 0.85, 0.80]
        with_embeddings = [(sup, np.array([1.0], dtype=np.float32))] + [
            (p, np.array([s], dtype=np.float32)) for p, s in zip(subs, sims, strict=True)
        ]
        monkeypatch.setattr(
            reconcile_mod,
            "get_active_particles_with_embeddings",
            AsyncMock(return_value=with_embeddings),
        )
        monkeypatch.setattr(reconcile_mod, "_cosine", lambda a, b: float(a[0] * b[0]))
        probed_contents: list[str] = []

        async def probe(_a: str, b: str) -> bool | None:
            probed_contents.append(b)
            return False  # "keep both" — the re-probed-every-night shape

        monkeypatch.setattr(reconcile_mod, "_has_contradiction_signal", probe)

        summary = await reconcile_supersession(db_session)
        assert summary["candidate_pairs"] == 4
        assert summary["probed"] == 2  # capped
        assert summary["probe_cap"] == 2
        # The budget went highest-similarity-first (0.95, then 0.90).
        assert probed_contents == ["old claim 0", "old claim 1"]

    def test_capped_reconcile_probe_disclosed_in_report(self) -> None:
        report = ConsolidationReport(
            completed_at=NOW,
            reconcile_candidate_pairs=9,
            reconcile_probes_run=2,
        )
        rendered = render_consolidation_report(report)
        assert "reconcile probe capped: probed 2 of 9 candidate pairs" in rendered
        assert "consolidation.max_reconcile_probes" in rendered


# ---------------------------------------------------------------------------
# Pass 4 commits its snapshot
# ---------------------------------------------------------------------------


class TestCurationPassCommits:
    """The 2026-09-29 chain: pass 4 inserted, and the next commit came hours later."""

    @pytest.mark.asyncio
    async def test_the_snapshot_is_committed_when_the_pass_returns(
        self, file_db_session: AsyncSession
    ) -> None:
        from particles.store.curation_snapshot_store import CollectionScope, get_snapshot
        from tests._write_probe import store_accepts_a_writer

        report = ConsolidationReport()
        await consolidation_mod._pass_curation(
            file_db_session, report, [], scope=CollectionScope.STORE, semantic=False
        )

        # Pass 5 starts here, and its first batch wait can last an hour. The
        # insert used to be uncommitted at this point, so a harvest or an
        # operator verb in another process could not write until pass 5
        # committed at its end, and failed with ``database is locked``.
        assert store_accepts_a_writer()
        assert report.curation_snapshot_id is not None
        await file_db_session.rollback()  # a later failure cannot undo it
        assert await get_snapshot(file_db_session, report.curation_snapshot_id) is not None


# ---------------------------------------------------------------------------
# Pass 5 shared behavioural budget (correction v1.74.1)
# ---------------------------------------------------------------------------


class TestUtilityBudget:
    """Pass 5: one behavioural budget per run, one pooled matcher batch."""

    @staticmethod
    def _sessions(monkeypatch: pytest.MonkeyPatch, n: int, beliefs: int = 45) -> list[Particle]:
        """Stage ``n`` harvested sessions and ``beliefs`` soft ACTIVE beliefs.

        Session ``i``'s transcript names ``run-s{i}``, so a fake matcher can
        answer per session from the prompt alone. ``_BEHAVIOURAL_BATCH`` is 15,
        so 45 beliefs make three full matcher groups per session.
        """
        from types import SimpleNamespace

        get_config().utility.mining.behavioural_matching = True
        get_config().utility.mining.behavioural_candidate_limit = 0  # no pre-filter
        entries = [
            SimpleNamespace(entry_id=f"e{i}", uri_r=f"claude-code://session/s{i}", tags=[])
            for i in range(n)
        ]
        snaps = {
            f"e{i}": [SimpleNamespace(content_hash=f"h{i}", archive_path="a", captured_at=NOW)]
            for i in range(n)
        }

        async def list_snapshots(_session: object, entry_id: str) -> list[object]:
            return snaps[entry_id]

        monkeypatch.setattr("particles.corpus.store.list_entries", AsyncMock(return_value=entries))
        monkeypatch.setattr("particles.corpus.store.list_snapshots_for_entry", list_snapshots)
        monkeypatch.setattr(
            "particles.corpus.deposit.load_blob",
            lambda h: f"[tool: Bash — run-s{h[1:]}]\n".encode(),
        )
        actives = [
            Particle(
                id=f"p-soft-{i}",
                content=f"Prefer general mechanism number {i}",
                confidence=Confidence(
                    value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="test",
                status=Status.ACTIVE,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1"
                    )
                ],
            )
            for i in range(beliefs)
        ]
        monkeypatch.setattr(
            consolidation_mod, "get_particles_by_status", AsyncMock(return_value=actives)
        )

        # every staged belief was shown to every session. Read at
        # call time, so a belief a test appends to ``actives`` is shown too.
        async def shown_all(*_a: object, **_k: object) -> Exposure:
            return Exposure(recorded=frozenset(p.id for p in actives))

        monkeypatch.setattr(ExposureReader, "exposure", shown_all)
        return actives

    @staticmethod
    def _reply(prompt: str) -> str:
        """A deterministic matcher: session ``s{i}`` follows guideline ``i % 3 + 1``."""
        import re

        m = re.search(r"run-s(\d+)", prompt)
        assert m is not None
        return f"[{int(m.group(1)) % 3 + 1}]"

    @staticmethod
    async def _events(session: AsyncSession, session_id: str) -> set[tuple[str, str]]:
        from sqlalchemy import select

        from particles.store.utility_store import UtilityEventRow

        rows = await session.execute(
            select(UtilityEventRow.particle_id, UtilityEventRow.match_basis).where(
                UtilityEventRow.session_id == session_id
            )
        )
        return {(pid, basis) for pid, basis in rows.all()}

    @pytest.mark.asyncio
    async def test_n_sessions_make_one_complete_many_call(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._sessions(monkeypatch, 5)
        get_config().utility.mining.max_behavioural_calls = 200
        submissions: list[int] = []

        async def fake_complete_many(
            purpose: str, requests: list[Any], **kwargs: object
        ) -> list[str | None]:
            assert kwargs["latency_tolerant"] is True
            submissions.append(len(requests))
            return [self._reply(r.prompt) for r in requests]

        monkeypatch.setattr("particles.llm.complete_many", fake_complete_many)
        report = ConsolidationReport()
        calls = await consolidation_mod._pass_utility(
            db_session, report, watermark=None, behavioural=True
        )
        assert submissions == [15]  # five sessions x three groups, one submission
        assert calls == 15
        assert report.utility_sessions_mined == 5
        assert report.utility_behavioural == 15  # one guideline per group
        assert report.utility_behavioural_exhausted_after is None

    @pytest.mark.asyncio
    async def test_pooled_results_match_the_per_session_path(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations.utility_mining import mine_session

        actives = self._sessions(monkeypatch, 4)
        get_config().utility.mining.max_behavioural_calls = 200

        async def fake_complete_many(
            purpose: str, requests: list[Any], **kwargs: object
        ) -> list[str | None]:
            return [self._reply(r.prompt) for r in requests]

        monkeypatch.setattr("particles.llm.complete_many", fake_complete_many)
        await consolidation_mod._pass_utility(
            db_session, ConsolidationReport(), watermark=None, behavioural=True
        )
        for i in range(4):
            await mine_session(
                db_session,
                f"solo-{i}",
                f"[tool: Bash — run-s{i}]\n",
                actives,
                latency_tolerant=True,
            )
            await db_session.commit()
            pooled = await self._events(db_session, f"s{i}")
            assert pooled  # the matcher credited something
            assert pooled == await self._events(db_session, f"solo-{i}")

    @pytest.mark.asyncio
    async def test_behavioural_budget_accumulates_across_sessions(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # utility.mining.max_behavioural_calls is a per-RUN cap: with cap 4
        # and two sessions each wanting 3 calls, the run submits 4 requests in
        # entry order (all of s0's, the first of s1's) and discloses exhaustion.
        self._sessions(monkeypatch, 2)
        get_config().utility.mining.max_behavioural_calls = 4
        submitted: list[str] = []

        async def fake_complete_many(
            purpose: str, requests: list[Any], **kwargs: object
        ) -> list[str | None]:
            submitted.extend(r.prompt for r in requests)
            return [self._reply(r.prompt) for r in requests]

        monkeypatch.setattr("particles.llm.complete_many", fake_complete_many)
        report = ConsolidationReport()
        calls = await consolidation_mod._pass_utility(
            db_session, report, watermark=None, behavioural=True
        )
        assert [("run-s0" in p, "run-s1" in p) for p in submitted] == [
            (True, False),
            (True, False),
            (True, False),
            (False, True),
        ]
        assert "mechanism number 0\n" in submitted[3]  # s1's FIRST group was kept
        assert calls == 4
        assert report.utility_behavioural_calls == 4
        assert report.utility_sessions_mined == 2
        assert report.utility_behavioural_exhausted_after == 2
        rendered = render_consolidation_report(report)
        assert "use-judge budget exhausted after 2 of 2 sessions" in rendered
        assert "utility.mining.max_behavioural_calls" in rendered

    @pytest.mark.asyncio
    async def test_a_failed_batch_credits_nothing_in_any_session(
        self,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        actives = self._sessions(monkeypatch, 3)
        # One belief with a literal token each session's transcript carries.
        actives.append(actives[0].model_copy(update={"id": "p-lit", "content": "Use `run-s`"}))
        get_config().utility.mining.max_behavioural_calls = 200

        async def boom(*a: object, **k: object) -> list[str | None]:
            raise RuntimeError("batch job rejected")

        monkeypatch.setattr("particles.llm.complete_many", boom)
        report = ConsolidationReport()
        with caplog.at_level(logging.INFO, logger="particles.operations.utility_mining"):
            calls = await consolidation_mod._pass_utility(
                db_session, report, watermark=None, behavioural=True
            )
        assert calls == 0
        assert report.utility_behavioural == 0
        assert report.utility_sessions_mined == 3
        # a literal nomination with no ruling records nothing.
        assert report.utility_literal == 0
        for i in range(3):
            assert await self._events(db_session, f"s{i}") == set()
        assert "nothing credited this run for 3 session(s)" in caplog.text

    @pytest.mark.asyncio
    async def test_a_spent_budget_skips_the_judge_and_credits_nothing(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the judge is skipped, never sent; with no ruling nothing is credited."""
        from particles.llm.spend_budget import SpendBudget, spend_budget

        actives = self._sessions(monkeypatch, 3)
        actives.append(actives[0].model_copy(update={"id": "p-lit", "content": "Use `run-s`"}))
        get_config().utility.mining.max_behavioural_calls = 200
        sent = AsyncMock()
        monkeypatch.setattr("particles.llm.complete_many", sent)

        report = ConsolidationReport()
        with spend_budget(SpendBudget(budget_usd=0.0, spent=lambda: 0.0)):
            calls = await consolidation_mod._pass_utility(
                db_session, report, watermark=None, behavioural=True
            )

        sent.assert_not_awaited()
        assert calls == 0
        assert (report.utility_sessions_mined, report.utility_literal) == (3, 0)
        assert report.utility_behavioural_skipped is not None
        assert report.utility_behavioural_skipped.startswith(
            "spent US$0.00 of US$0.00; pass 9's use judge skipped"
        )

    @pytest.mark.asyncio
    async def test_a_partial_batch_degrades_only_the_unanswered_session(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._sessions(monkeypatch, 2)
        get_config().utility.mining.max_behavioural_calls = 200

        async def fake_complete_many(
            purpose: str, requests: list[Any], **kwargs: object
        ) -> list[str | None]:
            return [None if "run-s1" in r.prompt else self._reply(r.prompt) for r in requests]

        monkeypatch.setattr("particles.llm.complete_many", fake_complete_many)
        report = ConsolidationReport()
        calls = await consolidation_mod._pass_utility(
            db_session, report, watermark=None, behavioural=True
        )
        assert calls == 3  # a dead request costs no budget
        assert len(await self._events(db_session, "s0")) == 3
        assert await self._events(db_session, "s1") == set()

    @pytest.mark.asyncio
    async def test_write_lock_is_not_held_across_the_batch(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._sessions(monkeypatch, 2)
        get_config().utility.mining.max_behavioural_calls = 200
        held = {"now": False, "ever": False}

        @contextlib.asynccontextmanager
        async def fake_write_lock(*a: object, **k: object) -> Any:
            held["now"] = held["ever"] = True
            try:
                yield
            finally:
                held["now"] = False

        async def fake_complete_many(
            purpose: str, requests: list[Any], **kwargs: object
        ) -> list[str | None]:
            assert not held["now"], "the matcher batch ran under the write lock"
            return [self._reply(r.prompt) for r in requests]

        monkeypatch.setattr(consolidation_mod, "write_lock", fake_write_lock)
        monkeypatch.setattr("particles.llm.complete_many", fake_complete_many)
        await consolidation_mod._pass_utility(
            db_session, ConsolidationReport(), watermark=None, behavioural=True
        )
        assert held["ever"]  # the writes did take the lock


# ---------------------------------------------------------------------------
# The run record + delta report (§7)
# ---------------------------------------------------------------------------


class TestRunRecord:
    @pytest.mark.asyncio
    async def test_payload_shape(self, db_session: AsyncSession, cycle_env: Path) -> None:
        report = await run_consolidation(db_session)
        event = await latest_run_event(db_session)
        assert event is not None
        assert event.event_id == report.event_id
        assert event.actor == "memory-consolidate"
        payload = event.payload or {}
        assert payload["format"] == 1
        assert payload["store"] == "default"
        assert payload["semantic_degraded"] is False
        assert payload["completed_at"] is not None
        assert set(payload["providers"]) == {
            "extraction",
            "semantic_lint",
            "abstraction",
            "use_judge",
        }
        assert payload["providers"]["extraction"].startswith("anthropic:")
        # Measured usage rides the run record; the cycle's passes are mocked,
        # so nothing was spent.
        assert payload["llm_usage"] == {"rows": [], "cost_usd": 0.0, "unpriced": []}
        assert report.llm_usage is not None
        assert "LLM usage: no LLM calls." in render_consolidation_report(report)
        pass_names = [p["name"] for p in payload["passes"]]
        assert pass_names == [
            "refresh",
            "extract",
            "reconcile",
            "reconcile_updates",
            "reanchor",
            "census",
            "disclose",
            "curation",
            "utility",
            "abstraction",
            "projection",
            "measure",
        ]
        for p in payload["passes"]:
            assert "duration_seconds" in p
            assert "llm_calls" in p
            assert p["status"] == "ran" or "(" in p["status"]
        census = payload["census"]
        assert census["cards"] == report.card_counts
        for key in (
            "contradiction_candidate_pairs",
            "contradiction_probes_run",
            "duplicate_candidate_pairs_total",
            "pending_backlog",
            "reconcile_candidate_pairs",
            "reconcile_probes_run",
            "utility_literal",
            "utility_behavioural_calls",
            "utility_behavioural_exhausted_after",
            "curation_queue_total",
        ):
            assert key in census

    @pytest.mark.asyncio
    async def test_delta_report_against_prior_event(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        prior_completed = NOW - timedelta(hours=26)
        await _seed_run_event(
            db_session,
            completed_at=prior_completed,
            cards={"contradiction": 1, "inconsistency": 0, "stale": 3},
            duplicates_total=6,
        )
        report = await run_consolidation(db_session, scope="store")
        # Current census (mocked cards): 2 contradictions, 1 duplicate, 1 stale.
        assert report.previous_run_at == prior_completed
        assert report.deltas == {"contradictions": 1, "duplicates": -5, "stale": -2}
        rendered = render_consolidation_report(report)
        assert "(+1 since the last census)" in rendered
        assert "previous run:" in rendered

    @pytest.mark.asyncio
    async def test_no_contradictions_delta_against_a_pre_0280_record(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        """a prior record that counted conflicts by claim is not comparable."""
        await _seed_run_event(
            db_session,
            completed_at=NOW - timedelta(hours=26),
            cards={"contradiction": 1, "contested": 5, "stale": 3},
            duplicates_total=6,
        )
        report = await run_consolidation(db_session, scope="store")
        assert report.deltas == {"duplicates": -5, "stale": -2}
        assert report.card_counts["inconsistency"] >= 0

    @pytest.mark.asyncio
    async def test_first_run_has_no_deltas(self, db_session: AsyncSession, cycle_env: Path) -> None:
        report = await run_consolidation(db_session)
        assert report.previous_run_at is None
        assert report.deltas == {}
        assert "first recorded run" in render_consolidation_report(report)

    def test_build_run_payload_is_versioned(self) -> None:
        payload = build_run_payload(
            store="default",
            actor="memory-consolidate",
            scope="delta",
            watermark=None,
            started_at=NOW,
            completed_at=NOW,
            semantic_degraded=False,
            semantic_degraded_reason=None,
            providers={},
            passes=[],
            census={},
        )
        assert payload["format"] == 1


# ---------------------------------------------------------------------------
# The audit's recording (§7 — actor: audit)
# ---------------------------------------------------------------------------


class TestClosureMeasurePass:
    """Pass 6b: the closure measure rides every run record."""

    @pytest.mark.asyncio
    async def test_the_run_record_carries_the_measure(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        report = await run_consolidation(db_session)
        event = await latest_run_event(db_session)
        assert event is not None
        closure = (event.payload or {})["closure"]
        assert closure["window_start"] is None  # the store's first run
        assert {"active", "contested", "contested_fraction", "autonomous_share"} <= set(closure)
        assert next(p for p in report.passes if p.name == "measure").llm_calls == 0
        assert "  contested        " in render_consolidation_report(report)

    @pytest.mark.asyncio
    async def test_consecutive_windows_tile(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        first = await run_consolidation(db_session, scope="store")
        second = await run_consolidation(db_session, scope="store")
        assert first.closure is not None and second.closure is not None
        assert second.closure.window_start == first.closure.window_end
        assert second.closure_previous is not None

    @pytest.mark.asyncio
    async def test_the_measure_runs_on_a_structural_only_night(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        report = await run_consolidation(db_session, structural_only=True)
        assert next(p for p in report.passes if p.name == "measure").status == "ran"
        assert report.closure is not None

    @pytest.mark.asyncio
    async def test_a_failed_measure_still_writes_the_run_record(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            consolidation_mod, "measure_closure", AsyncMock(side_effect=RuntimeError("boom"))
        )
        report = await run_consolidation(db_session)
        assert report.failed_passes() == ["measure"]
        event = await latest_run_event(db_session)
        assert event is not None and "closure" not in (event.payload or {})


class TestAuditRecording:
    @pytest.mark.asyncio
    async def test_record_audit_run_writes_shared_shape(self, db_session: AsyncSession) -> None:
        from particles.operations.audit import AuditBucket, AuditReport

        audit_report = AuditReport(
            store="default",
            files_audited=2,
            extracted_snapshots=3,
            buckets=[AuditBucket(kind=CardKind.CONTRADICTION, count=4)],
            contradiction_probes_run=7,
            contradiction_candidate_pairs=9,
        )
        await record_audit_run(db_session, audit_report, started_at=NOW)
        await db_session.commit()
        events = await list_events(db_session, event_type=OperatorEventType.CONSOLIDATION_RUN)
        assert len(events) == 1
        event = events[0]
        assert event.actor == "audit"
        payload = event.payload or {}
        assert payload["format"] == 1
        assert payload["census"]["cards"] == {"contradiction": 4}
        assert payload["census"]["contradiction_probes_run"] == 7
        names = {p["name"]: p for p in payload["passes"]}
        assert names["extract"]["status"] == "ran"
        assert names["census"]["llm_calls"] == 7

    @pytest.mark.asyncio
    async def test_run_memory_audit_records_measured_usage(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every completion the census pays for lands on the report and the event."""
        from particles.llm import record_usage
        from particles.llm.usage import purpose_scope
        from particles.operations import audit as audit_mod

        async def _paying_census(*args: object, **kwargs: object) -> list[object]:
            with purpose_scope("semantic_lint"):
                for _ in range(2):
                    record_usage(
                        "anthropic:claude-haiku-4-5", input_tokens=500_000, output_tokens=10_000
                    )
            return []

        monkeypatch.setattr(audit_mod, "collect_cards", _paying_census)
        report = await audit_mod.run_memory_audit(db_session, semantic=True)
        await db_session.commit()
        assert report.llm_usage is not None
        # $1/$5 per MTok: 1M in + 20k out.
        assert report.llm_usage.cost_usd == pytest.approx(1.10)
        events = await list_events(db_session, event_type=OperatorEventType.CONSOLIDATION_RUN)
        [row] = (events[0].payload or {})["llm_usage"]["rows"]
        assert (row["purpose"], row["model"], row["calls"]) == (
            "semantic_lint",
            "claude-haiku-4-5",
            2,
        )
        assert (row["input_tokens"], row["output_tokens"]) == (1_000_000, 20_000)

    @pytest.mark.asyncio
    async def test_run_memory_audit_records_event(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The audit operation itself contributes to the delta chain."""
        from particles.operations import audit as audit_mod

        # audit binds collect_cards at module top — patch its namespace.
        monkeypatch.setattr(audit_mod, "collect_cards", AsyncMock(return_value=[]))
        report = await audit_mod.run_memory_audit(db_session, semantic=False)
        await db_session.commit()
        assert report.semantic_skipped is True
        events = await list_events(db_session, event_type=OperatorEventType.CONSOLIDATION_RUN)
        assert len(events) == 1
        assert events[0].actor == "audit"
        assert (events[0].payload or {})["semantic_degraded"] is True


# ---------------------------------------------------------------------------
# Renderer (§7)
# ---------------------------------------------------------------------------


class TestRenderer:
    def test_capped_probe_disclosed(self) -> None:
        report = ConsolidationReport(
            completed_at=NOW,
            contradiction_candidate_pairs=80,
            contradiction_probes_run=50,
        )
        rendered = render_consolidation_report(report)
        assert "probed 50 of 80 candidate pairs" in rendered

    def test_pending_remainder_disclosed(self) -> None:
        report = ConsolidationReport(
            completed_at=NOW,
            pending_total=15,
            pending_extracted=3,
            pending_remaining=12,
        )
        rendered = render_consolidation_report(report)
        assert "12 remain; the next run continues" in rendered

    def test_pending_line_names_how_long_the_backlog_has_waited(self) -> None:
        report = ConsolidationReport(
            completed_at=NOW,
            pending_total=340,
            pending_extracted=0,
            pending_retry=12,
            pending_remaining=340,
            pending_oldest_at=datetime(2026, 7, 19, tzinfo=UTC),
        )
        rendered = render_consolidation_report(report)
        assert (
            "pending          extracted 0, 12 left for retry "
            "(340 remain, oldest waiting since 2026-07-19; the next run continues)"
        ) in rendered
        payload = consolidation_mod._census_payload(report)
        assert payload["pending_oldest_at"] == "2026-07-19T00:00:00+00:00"
        assert payload["pending_retry"] == 12

    def test_queue_footer(self) -> None:
        report = ConsolidationReport(
            completed_at=NOW,
            curation_queue=['[stale] "The sky is green." — expired'],
            curation_queue_total=12,
        )
        rendered = render_consolidation_report(report)
        assert "Curation queue — top 1 of 12:" in rendered
        assert "Run 'particles curate'" in rendered


class TestAbstractionPass:
    """pass 5b — gating, wiring, run-record counts, render line."""

    @pytest.mark.asyncio
    async def test_disabled_by_default_skipped_with_disclosure(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        report = await run_consolidation(db_session)
        entry = next(p for p in report.passes if p.name == "abstraction")
        assert entry.status == "skipped"
        assert "consolidation.abstraction.enabled" in (entry.detail or "")
        rendered = render_consolidation_report(report)
        assert "pass skipped: abstraction" in rendered

    @pytest.mark.asyncio
    async def test_enabled_pass_runs_and_records(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config
        from particles.operations.abstraction import AbstractionReport, RevalidationCounts

        get_config().consolidation.abstraction.enabled = True
        fake = AbstractionReport(
            mode="propose",
            clusters_found=2,
            candidates_synthesized=1,
            proposed_event_ids=["ev-1"],
            revalidation=RevalidationCounts(checked=2, refreshed_entailed=1, deferred_in_review=1),
            llm_calls=3,
        )
        mock_pass = AsyncMock(return_value=fake)
        monkeypatch.setattr(consolidation_mod, "run_abstraction_pass", mock_pass)

        report = await run_consolidation(db_session)
        entry = next(p for p in report.passes if p.name == "abstraction")
        assert entry.status == "ran"
        assert entry.llm_calls == 3
        assert report.abstraction is fake
        # Pass ordering: after utility, before projection.
        names = [p.name for p in report.passes]
        assert names.index("utility") < names.index("abstraction") < names.index("projection")

        event = await latest_run_event(db_session)
        assert event is not None
        census = (event.payload or {})["census"]
        assert census["abstraction_proposed"] == 1
        assert census["abstraction_clusters"] == 2
        assert census["abstraction_revalidated"] == 1
        assert census["abstraction_deferred_in_review"] == 1

        rendered = render_consolidation_report(report)
        assert "abstraction      1 proposed, 1 revalidated, 1 held for review" in rendered

    @pytest.mark.asyncio
    async def test_semantic_degraded_skips_abstraction(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        get_config().consolidation.abstraction.enabled = True
        report = await run_consolidation(db_session, structural_only=True)
        entry = next(p for p in report.passes if p.name == "abstraction")
        assert entry.status == "skipped"
        assert "LLM-priced" in (entry.detail or "")


# ---------------------------------------------------------------------------
# Pass 0.5 — local-source refresh
# ---------------------------------------------------------------------------


class TestRefreshPass:
    @pytest.mark.asyncio
    async def test_runs_before_extract(self, db_session: AsyncSession, cycle_env: Path) -> None:
        """Ordering is the whole point: a file edited today must reach the
        projection tonight, not three nights from now. Refresh writes the
        PENDING snapshot; extract (next) turns it into beliefs; reconcile
        (after that) sweeps cross-entry."""
        report = await run_consolidation(db_session)
        names = [p.name for p in report.passes]
        assert names.index("refresh") < names.index("extract") < names.index("reconcile")

    @pytest.mark.asyncio
    async def test_runs_on_a_degraded_night(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pass is zero-LLM, so unlike every other semantic pass it still
        runs with no key: a structural-only night must still notice that the
        rules changed."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        report = await run_consolidation(db_session, structural_only=True)

        refresh = next(p for p in report.passes if p.name == "refresh")
        assert refresh.status == "ran"
        assert refresh.llm_calls == 0
        # …while the LLM-priced passes disclose their skip.
        assert report.semantic_degraded is True
        assert next(p for p in report.passes if p.name == "reconcile").status == "skipped"

    @pytest.mark.asyncio
    async def test_disabled_by_config_is_disclosed(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        monkeypatch.setattr(get_config().local_refresh, "enabled", False)
        report = await run_consolidation(db_session)

        refresh = next(p for p in report.passes if p.name == "refresh")
        assert refresh.status == "skipped"
        assert refresh.detail == "local_refresh.enabled is false"

    @pytest.mark.asyncio
    async def test_changed_file_becomes_a_pending_snapshot(
        self,
        db_session: AsyncSession,
        cycle_env: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End to end through the pass: edit on disk → new PENDING snapshot."""
        # Pass 1 (extract) runs right after refresh and would pick the new
        # snapshot up. Unstubbed, it sent a live extraction call under the
        # fixture's fake key, and the snapshot stayed PENDING only because that
        # call failed. Refresh is the pass under test, so extract is a seam here.
        monkeypatch.setattr(consolidation_mod, "_pass_extract", AsyncMock(return_value=0))
        from particles.core.schema import (
            CorpusEntry,
            ExtractionStatus,
            FetchPolicy,
            Mutability,
            Snapshot,
            WarcRecordType,
        )
        from particles.corpus.deposit import sha256
        from particles.corpus.store import (
            CorpusEntryRow,
            SnapshotRow,
            list_snapshots_for_entry,
        )

        rules = tmp_path / "AGENTS.md"
        rules.write_text("the old rule")
        entry = CorpusEntry(
            entry_id=str(uuid.uuid4()),
            source_type="LOCAL_MARKDOWN",
            uri_r=rules.resolve().as_uri(),
            fetch_policy=FetchPolicy.LAZY,
            mutability=Mutability.MUTABLE,
            deposited_by="test",
        )
        db_session.add(CorpusEntryRow.from_model(entry))
        snap = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            captured_at=datetime.now(UTC) - timedelta(days=1),
            content_hash=sha256(rules.read_bytes()),
            last_modified=datetime.fromtimestamp(rules.stat().st_mtime, tz=UTC),
            warc_record_type=WarcRecordType.RESPONSE,
            extraction_status=ExtractionStatus.COMPLETE,
        )
        db_session.add(SnapshotRow.from_model(snap, entry.entry_id))
        await db_session.commit()

        rules.write_text("the NEW rule — the old one is forbidden")
        report = await run_consolidation(db_session)

        assert report.refresh_checked == 1
        assert report.refresh_updated == 1
        snapshots = await list_snapshots_for_entry(db_session, entry.entry_id)
        assert len(snapshots) == 2
        newest = max(snapshots, key=lambda s: s.captured_at)
        assert newest.extraction_status == ExtractionStatus.PENDING
        assert "local sources    1 checked, 1 changed" in render_consolidation_report(report)

    @pytest.mark.asyncio
    async def test_never_policy_entries_are_not_swept(
        self, db_session: AsyncSession, cycle_env: Path, tmp_path: Path
    ) -> None:
        """The opt-in gate: a default local deposit is invisible to the sweep."""
        from particles.core.schema import CorpusEntry, FetchPolicy, Mutability
        from particles.corpus.store import CorpusEntryRow

        f = tmp_path / "notes.md"
        f.write_text("x")
        db_session.add(
            CorpusEntryRow.from_model(
                CorpusEntry(
                    entry_id=str(uuid.uuid4()),
                    source_type="LOCAL_MARKDOWN",
                    uri_r=f.resolve().as_uri(),
                    fetch_policy=FetchPolicy.NEVER,
                    mutability=Mutability.MUTABLE,
                    deposited_by="test",
                )
            )
        )
        await db_session.commit()

        report = await run_consolidation(db_session)
        assert report.refresh_checked == 0


class TestPassExtractPooled:
    """The pooled extract pass: dedupe rule, accounting, fallback."""

    @staticmethod
    def _fake_session_scope(monkeypatch: pytest.MonkeyPatch) -> None:
        """Give each task an inert session so no real engine is touched."""
        from contextlib import asynccontextmanager

        import particles.db as db_mod

        @asynccontextmanager
        async def fake_scope(*args: Any, **kwargs: Any) -> Any:
            yield AsyncMock()

        monkeypatch.setattr(db_mod, "session_scope", fake_scope)

    @pytest.mark.asyncio
    async def test_one_snapshot_per_entry_and_pool_threading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.llm import CompletionPool
        from particles.operations import extract as extract_mod

        self._fake_session_scope(monkeypatch)
        seen: list[tuple[str, str]] = []

        async def fake_extract(
            session: Any, entry_id: str, snapshot_id: str, **kwargs: Any
        ) -> list[Any]:
            seen.append((entry_id, snapshot_id))
            assert isinstance(kwargs.get("completion_pool"), CompletionPool)
            return []

        monkeypatch.setattr(extract_mod, "extract_snapshot", fake_extract)

        report = ConsolidationReport()
        report.pending_total = 3
        batch = [("entry-a", "snap-1"), ("entry-a", "snap-2"), ("entry-b", "snap-3")]
        extracted = await consolidation_mod._pass_extract_pooled(batch, report)

        # entry-a's second pending snapshot stays PENDING for the next run
        # (at most one snapshot per corpus entry per pooled pass).
        assert sorted(seen) == [("entry-a", "snap-1"), ("entry-b", "snap-3")]
        assert extracted == 2
        assert report.pending_extracted == 2
        assert report.pending_failed == 0
        assert report.pending_remaining == 1

    @pytest.mark.asyncio
    async def test_account_level_failure_is_disclosed_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.llm.errors import AccountLevelLLMError
        from particles.operations import extract as extract_mod

        self._fake_session_scope(monkeypatch)

        async def fake_extract(*args: Any, **kwargs: Any) -> list[Any]:
            raise AccountLevelLLMError(RuntimeError("credit balance is too low"))

        monkeypatch.setattr(extract_mod, "extract_snapshot", fake_extract)

        report = ConsolidationReport()
        report.pending_total = 2
        batch = [("entry-a", "snap-1"), ("entry-b", "snap-2")]
        extracted = await consolidation_mod._pass_extract_pooled(batch, report)

        # Every parked task fails identically; the serial loop's single break
        # is mirrored as ONE disclosed failure, not one per snapshot.
        assert extracted == 0
        assert report.pending_extracted == 0
        assert report.pending_failed == 1
        assert report.pending_remaining == 2

    @pytest.mark.asyncio
    async def test_per_snapshot_failure_does_not_sink_the_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations import extract as extract_mod

        self._fake_session_scope(monkeypatch)

        async def fake_extract(
            session: Any, entry_id: str, snapshot_id: str, **kwargs: Any
        ) -> list[Any]:
            if entry_id == "entry-bad":
                raise ValueError("malformed blob")
            return []

        monkeypatch.setattr(extract_mod, "extract_snapshot", fake_extract)

        report = ConsolidationReport()
        report.pending_total = 2
        batch = [("entry-bad", "snap-1"), ("entry-good", "snap-2")]
        extracted = await consolidation_mod._pass_extract_pooled(batch, report)

        assert extracted == 1
        assert report.pending_extracted == 1
        assert report.pending_failed == 1

    @pytest.mark.asyncio
    async def test_a_snapshot_handed_back_pending_is_not_counted_extracted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every LLM call failed, so the pipeline left the snapshot PENDING.

        ``extract_snapshot`` returns normally in that case, and the pass used to
        count every normal return as extracted: the owner's store reported
        "extracted 12" on a night that wrote nothing.
        """
        from particles.operations import extract as extract_mod

        self._fake_session_scope(monkeypatch)

        async def fake_extract(
            session: Any, entry_id: str, snapshot_id: str, **kwargs: Any
        ) -> list[Any]:
            if entry_id == "entry-cut":
                kwargs["outcome_out"].failed_calls = 3
            else:
                kwargs["carry_forward_ids_out"].append("p-carried")
            return []

        monkeypatch.setattr(extract_mod, "extract_snapshot", fake_extract)

        report = ConsolidationReport()
        report.pending_total = 2
        batch = [("entry-cut", "snap-1"), ("entry-ok", "snap-2")]
        extracted = await consolidation_mod._pass_extract_pooled(batch, report)

        assert extracted == 1
        assert report.pending_extracted == 1
        assert report.pending_retry == 1
        assert report.pending_empty == 0  # the carried-forward one wrote nothing new, rightly
        assert report.pending_remaining == 1
        rendered = render_consolidation_report(report)
        assert "extracted 1, 1 left for retry" in rendered
        assert "1 snapshot(s) left PENDING after failed LLM calls" in rendered
        assert "kept the chunks that answered" not in rendered

    @pytest.mark.asyncio
    async def test_a_completion_with_no_beliefs_is_disclosed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations import extract as extract_mod

        self._fake_session_scope(monkeypatch)

        async def fake_extract(*args: Any, **kwargs: Any) -> list[Any]:
            return []

        monkeypatch.setattr(extract_mod, "extract_snapshot", fake_extract)

        report = ConsolidationReport()
        report.pending_total = 1
        await consolidation_mod._pass_extract_pooled([("entry-a", "snap-1")], report)

        assert report.pending_extracted == 1
        assert report.pending_empty == 1
        payload = consolidation_mod._census_payload(report)
        assert payload["pending_empty"] == 1
        assert "1 snapshot(s) completed with no beliefs" in render_consolidation_report(report)

    @pytest.mark.asyncio
    async def test_a_snapshot_superseded_after_listing_leaves_the_remainder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations import extract as extract_mod

        self._fake_session_scope(monkeypatch)

        async def fake_extract(*args: Any, **kwargs: Any) -> list[Any]:
            kwargs["outcome_out"].skipped = "superseded"
            return []

        monkeypatch.setattr(extract_mod, "extract_snapshot", fake_extract)

        report = ConsolidationReport()
        report.pending_total = 1
        await consolidation_mod._pass_extract_pooled([("entry-a", "snap-1")], report)

        assert report.pending_extracted == 0
        assert report.pending_empty == 0
        assert report.pending_superseded_late == 1
        assert report.pending_remaining == 0

    def test_the_pooled_cap_counts_distinct_entries(self) -> None:
        """Twenty slots bought thirteen extractions a night on the owner's store."""
        from particles.corpus.store import PendingSnapshot

        pending = [
            PendingSnapshot(entry, f"snap-{i}", NOW, 0)
            for i, entry in enumerate(["a", "a", "a", "b", "b", "c", "d"])
        ]
        assert consolidation_mod._one_per_entry(pending, 3) == [
            ("a", "snap-0"),
            ("b", "snap-3"),
            ("c", "snap-5"),
        ]
        assert len(consolidation_mod._one_per_entry(pending, 20)) == 4

    @pytest.mark.asyncio
    async def test_pass_extract_resets_a_stranded_claim_and_extracts_it(
        self, monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
    ) -> None:
        """A claim left IN_PROGRESS by a killed run was invisible to every later night."""
        from particles.config import get_config
        from particles.core.schema import (
            CorpusEntry,
            ExtractionStatus,
            FetchPolicy,
            Mutability,
            Snapshot,
            WarcRecordType,
        )
        from particles.corpus.store import CorpusEntryRow, SnapshotRow
        from particles.operations import extract as extract_mod

        get_config().consolidation.extract_batching = False
        entry = CorpusEntry(
            entry_id=str(uuid.uuid4()),
            source_type="LOCAL_MARKDOWN",
            uri_r="file:///tmp/memory/note.md",
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.NEVER,
            deposited_by="test",
        )
        db_session.add(CorpusEntryRow.from_model(entry))
        snap = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            captured_at=NOW - timedelta(days=60),
            content_hash="c" * 64,
            archive_path="/x",
            extraction_status=ExtractionStatus.IN_PROGRESS,
            warc_record_type=WarcRecordType.RESPONSE,
        )
        row = SnapshotRow.from_model(snap, entry.entry_id)
        row.extraction_started_at = NOW - timedelta(days=55)
        db_session.add(row)
        await db_session.commit()

        calls: list[str] = []

        async def fake_extract(
            session: Any, entry_id: str, snapshot_id: str, **kwargs: Any
        ) -> list[Any]:
            calls.append(snapshot_id)
            return []

        monkeypatch.setattr(extract_mod, "extract_snapshot", fake_extract)

        report = ConsolidationReport()
        await consolidation_mod._pass_extract(db_session, report)

        assert report.pending_reset_stale == 1
        assert calls == [snap.snapshot_id]
        assert "1 snapshot(s) stranded IN_PROGRESS" in render_consolidation_report(report)

    @pytest.mark.asyncio
    async def test_pass_extract_collapses_before_it_lists(
        self, monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
    ) -> None:
        """three edits of one MUTABLE file are one night's one extraction.

        Without the collapse, oldest-first plus one snapshot per entry per pooled
        run drains such an entry over three nights, each extracting a generation
        the file no longer holds.
        """
        from particles.config import get_config
        from particles.core.schema import (
            CorpusEntry,
            ExtractionStatus,
            FetchPolicy,
            Mutability,
            Snapshot,
            WarcRecordType,
        )
        from particles.corpus.deposit import save_blob, sha256
        from particles.corpus.store import CorpusEntryRow, SnapshotRow
        from particles.operations import extract as extract_mod
        from particles.operations.consolidation import _census_payload

        get_config().consolidation.extract_batching = False
        entry = CorpusEntry(
            entry_id=str(uuid.uuid4()),
            source_type="LOCAL_MARKDOWN",
            uri_r="file:///tmp/MEMORY.md",
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.NEVER,
            deposited_by="test",
        )
        db_session.add(CorpusEntryRow.from_model(entry))
        snapshot_ids: list[str] = []
        for age in (3, 2, 1):
            content = f"edit {age}".encode()
            snap = Snapshot(
                snapshot_id=str(uuid.uuid4()),
                captured_at=datetime.now(UTC) - timedelta(days=age),
                content_hash=sha256(content),
                archive_path=save_blob(content, sha256(content)),
                extraction_status=ExtractionStatus.PENDING,
                warc_record_type=WarcRecordType.RESPONSE,
            )
            db_session.add(SnapshotRow.from_model(snap, entry.entry_id))
            snapshot_ids.append(snap.snapshot_id)
        await db_session.commit()

        calls: list[tuple[str, Any]] = []

        async def fake_extract(
            session: Any, entry_id: str, snapshot_id: str, **kwargs: Any
        ) -> list[Any]:
            calls.append((snapshot_id, kwargs.get("skip_if_superseded")))
            return []

        monkeypatch.setattr(extract_mod, "extract_snapshot", fake_extract)

        report = ConsolidationReport()
        extracted = await consolidation_mod._pass_extract(db_session, report)

        assert calls == [(snapshot_ids[-1], True)]
        assert extracted == 1
        assert report.pending_collapsed == 2
        assert report.pending_total == 1  # the collapsed generations were never owed
        assert _census_payload(report)["pending_collapsed"] == 2

    @pytest.mark.asyncio
    async def test_extract_batching_false_restores_the_serial_loop(
        self, monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
    ) -> None:
        from particles.config import get_config
        from particles.operations import extract as extract_mod

        get_config().consolidation.extract_batching = False
        monkeypatch.setattr(
            consolidation_mod,
            "_pass_extract_pooled",
            AsyncMock(side_effect=AssertionError("pooled path must not run")),
        )
        serial_calls: list[tuple[str, str]] = []

        async def fake_extract(
            session: Any, entry_id: str, snapshot_id: str, **kwargs: Any
        ) -> list[Any]:
            serial_calls.append((entry_id, snapshot_id))
            # The serial loop passes no completion pool.
            assert "completion_pool" not in kwargs
            return []

        monkeypatch.setattr(extract_mod, "extract_snapshot", fake_extract)

        import particles.corpus.store as corpus_store

        async def fake_pending(session: Any) -> list[corpus_store.PendingSnapshot]:
            return [corpus_store.PendingSnapshot("entry-a", "snap-1", NOW, 0)]

        monkeypatch.setattr(corpus_store, "list_pending_snapshots_for_catchup", fake_pending)

        report = ConsolidationReport()
        extracted = await consolidation_mod._pass_extract(db_session, report)

        assert serial_calls == [("entry-a", "snap-1")]
        assert extracted == 1


# ---------------------------------------------------------------------------
# The second reading on the nightly card path
# ---------------------------------------------------------------------------


class TestNightlySecondReading:
    @pytest.mark.asyncio
    async def test_census_verifies_flags_and_records_the_split(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The nightly census counts confirmed flags, as the interactive audit does."""
        seen: dict[str, Any] = {}

        async def fake_collect(*args: object, **kwargs: Any) -> list[CurationCard]:
            control = kwargs["contradiction_probe"]
            seen["control"] = control
            # What the detector would leave behind: 6 flags, 3 read, 1 kept.
            control.probes_run = 50
            control.flagged = 6
            control.verifications_run = 3
            control.confirmed = 1
            control.unverified = 3
            return [_card(CardKind.CONTRADICTION, "p1")]

        monkeypatch.setattr(consolidation_mod, "collect_cards", fake_collect)
        report = await run_consolidation(db_session)

        control = seen["control"]
        assert control.verify is True
        assert control.latency_tolerant is True
        assert control.max_verifications == 25
        assert report.contradiction_verified is True
        assert (report.contradiction_flagged, report.contradiction_confirmed) == (6, 1)
        # Both the batched probes and the second readings are the pass's spend.
        assert next(p for p in report.passes if p.name == "census").llm_calls == 53
        assert "first pass flagged 6 claim pairs, a second reading confirmed 1, 3 not read" in (
            render_consolidation_report(report)
        )

    @pytest.mark.asyncio
    async def test_verify_off_counts_every_flag(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        monkeypatch.setattr(get_config().audit, "verify_contradictions", False)
        seen: dict[str, Any] = {}

        async def fake_collect(*args: object, **kwargs: Any) -> list[CurationCard]:
            seen["control"] = kwargs["contradiction_probe"]
            return []

        monkeypatch.setattr(consolidation_mod, "collect_cards", fake_collect)
        report = await run_consolidation(db_session)
        assert seen["control"].verify is False
        assert report.contradiction_verified is False
        assert "second reading" not in render_consolidation_report(report)


class TestDisagreementHeadline:
    def test_headline_counts_groups(self) -> None:
        report = ConsolidationReport(
            completed_at=NOW,
            card_counts={"contradiction": 4, "inconsistency": 1},
            contradiction_disagreements=2,
            contradiction_grouped_pairs=4,
        )
        assert report.headline_contradictions == 3

    def test_old_run_record_counts_cards(self) -> None:
        payload = {"census": {"cards": {"contradiction": 4, "inconsistency": 1}}}
        headline = consolidation_mod._headline_from_payload(payload)
        assert headline is not None and headline["contradictions"] == 5
        payload["census"]["contradiction_disagreements"] = 2
        payload["census"]["contradiction_grouped_pairs"] = 4
        headline = consolidation_mod._headline_from_payload(payload)
        assert headline is not None and headline["contradictions"] == 3

    def test_a_pre_0280_record_has_no_contradictions_headline(self) -> None:
        payload = {"census": {"cards": {"contradiction": 4, "contested": 1}}}
        headline = consolidation_mod._headline_from_payload(payload)
        assert headline is not None and "contradictions" not in headline


class TestDisclosePass:
    """Pass 3b: the card swap and the headline that does not move."""

    @pytest.mark.asyncio
    async def test_a_disclosed_pair_swaps_its_card_and_keeps_the_headline(
        self, db_session: Any
    ) -> None:
        from particles.core.contradiction_disclosure import ConfirmedPair
        from particles.operations.consolidation import (
            ConsolidationReport,
            _count_cards,
            _pass_disclose,
        )
        from particles.operations.curation.cards import CardKind, CurationCard
        from tests._observer_scope import belief, generation

        n1 = await generation(db_session, "n1.md", "one", [])
        n2 = await generation(db_session, "n2.md", "two", [])
        a = await belief(db_session, "the endpoint exists", n1)
        b = await belief(db_session, "the endpoint returns not found", n2)
        await db_session.commit()

        card = CurationCard(
            kind=CardKind.CONTRADICTION,
            particle_ids=[a.id],
            diagnostic=f"Semantic contradiction with particle {b.id}: exists vs not found",
        )
        report = ConsolidationReport(
            contradiction_verified=True,
            contradiction_flagged=3,
            contradiction_confirmed=1,
            contradiction_disagreements=1,
            contradiction_grouped_pairs=1,
            contradiction_finding_pairs=[(a.id, b.id, False)],
            contradiction_confirmed_pairs=[
                ConfirmedPair(a=a.id, b=b.id, same_source=False, reason="exists vs not found")
            ],
        )
        _count_cards(report, [card], None)
        before = report.headline_contradictions
        assert before == 1

        cards, covered = await _pass_disclose(
            db_session,
            report,
            [card],
            census_ran=True,
            semantic=True,
            degrade_reason=None,
            actor="memory-consolidate",
            scope_ids=None,
        )

        assert frozenset((a.id, b.id)) in covered
        # the record replaces the pair's CONTRADICTION card with
        # its own conflict card, not one CONTESTED card per claim.
        (conflict,) = cards
        assert conflict.kind is CardKind.INCONSISTENCY
        assert set(conflict.particle_ids) == {a.id, b.id}
        assert report.disclosure is not None and report.disclosure.opened_records == 1
        assert report.disclosure.skipped == {"unconfirmed": 2}
        assert (report.disclosure_open_records, report.contested_census_claims) == (1, 0)
        # One disagreement before, one after: now counted through its record.
        assert report.headline_contradictions == before

    @pytest.mark.asyncio
    async def test_no_census_means_no_mint_but_the_line_says_why(self, db_session: Any) -> None:
        from particles.operations.consolidation import (
            ConsolidationReport,
            _pass_disclose,
            render_disclosure_line,
        )

        report = ConsolidationReport()
        cards, _ = await _pass_disclose(
            db_session,
            report,
            [],
            census_ran=False,
            semantic=False,
            degrade_reason="no API key",
            actor="memory-consolidate",
            scope_ids=None,
        )
        assert cards == []
        assert report.disclosure is not None
        assert render_disclosure_line(report.disclosure) == (
            "  contradiction disclosure: opened none (the census did not run this night); 0 waiting"
        )

    def test_report_line_names_what_opened_and_closed(self) -> None:
        from particles.operations.consolidation import render_disclosure_line
        from particles.operations.contradiction_disclosure import DisclosureReport

        line = render_disclosure_line(
            DisclosureReport(
                cap=10,
                opened=[{"record_ids": ["r1"], "members": 3}],
                closed=[{"cause": "lapsed"}],
            )
        )
        assert line == (
            "  contradiction disclosure: opened 1 inconsistency (3 claims) for the agent "
            "(cap 10); closed 1 whose side is no longer stated; 0 waiting"
        )

    def test_report_line_names_the_rereading(self) -> None:
        from particles.operations.consolidation import render_disclosure_line
        from particles.operations.contradiction_disclosure import DisclosureReport

        line = render_disclosure_line(
            DisclosureReport(
                cap=10,
                closed=[{"cause": "withdrawn"}, {"cause": "withdrawn"}, {"cause": "regrouped"}],
                reread={"read": 3, "confirmed": 1, "withdrawn": 2, "failed": 0, "deferred": 4},
            )
        )
        assert line == (
            "  contradiction disclosure: opened none (cap 10); closed 2 that a re-reading "
            "no longer confirms; regrouped 1; 0 waiting; re-read 3 pair(s) confirmed under "
            "an earlier instruction, withdrew 2, 4 left for a later night "
            "(consolidation.contradiction_disclosure.max_rereadings_per_run)"
        )

    def test_headline_counts_a_census_record_once(self) -> None:
        from particles.operations.consolidation import _contradiction_headline

        # Two probe disagreements, one recorded edge card, and two open records
        # (one extract-time, one census): each record counts once.
        assert _contradiction_headline(3, 2, 2, 2) == 2 + 1 + 2
        # A run record written before the grouping counts cards.
        assert _contradiction_headline(3, 2, None, 0) == 3 + 2


_ONE_CLAIM = (
    '[{"content": "The source states one thing.", '
    '"confidence_value": 0.8, "uncertainty_nature": "EPISTEMIC"}]'
)


@pytest.fixture
async def small_pool_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """A file-backed SQLite store whose pool holds two connections, 0.5 s timeout.

    Yields a deposit helper: ``await deposit(n)`` writes ``n`` distinct notes
    and returns their ``(entry_id, snapshot_id)`` pairs. Embeddings are mocked.
    """
    import functools

    import numpy as np

    import particles._orm_modules  # noqa: F401
    import particles.db as db_mod
    from particles import embeddings as ep
    from particles.config import reset_config
    from particles.corpus.deposit import deposit_file

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    reset_config()
    monkeypatch.setattr(
        db_mod,
        "create_async_engine",
        functools.partial(
            db_mod.create_async_engine, pool_size=2, max_overflow=0, pool_timeout=0.5
        ),
    )
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, **_: np.ones((len(texts), 4), dtype=np.float32)
    )
    original_model = ep._embedding_model
    ep.set_embedding_model(model)
    engine = db_mod.get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(db_mod.Base.metadata.create_all)

    async def deposit(n: int) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        async with db_mod.session_scope() as session:
            for i in range(n):
                path = tmp_path / f"note-{i}.md"
                path.write_text(f"# Note {i}\n\nA distinct note number {i}.\n")
                pairs.append(await deposit_file(session, path))
            await session.commit()
        return pairs

    try:
        yield deposit
    finally:
        ep.set_embedding_model(original_model)
        await engine.dispose()


async def _statuses(batch: list[tuple[str, str]]) -> list[str]:
    from particles.corpus.store import get_entry
    from particles.db import session_scope

    out: list[str] = []
    async with session_scope() as session:
        for entry_id, snapshot_id in batch:
            entry = await get_entry(session, entry_id)
            assert entry is not None
            (snap,) = [s for s in entry.snapshots if s.snapshot_id == snapshot_id]
            out.append(snap.extraction_status.value)
    return out


class TestPooledExtractionHoldsNoConnectionAcrossTheBatch:
    """The 2026-09-27 pool exhaustion: many pooled tasks, few connections, a slow batch.

    Each pooled task used to keep its session's read transaction open while it
    waited on the Message Batches job, so the tasks past the SQLAlchemy pool's
    capacity timed out acquiring a connection (``QueuePool limit … reached``)
    and their snapshots were stranded IN_PROGRESS. The store here pools two
    connections with a half-second timeout, and the batch takes twice that.
    """

    @pytest.mark.asyncio
    async def test_more_tasks_than_connections_with_a_slow_batch(
        self, monkeypatch: pytest.MonkeyPatch, small_pool_store: Any
    ) -> None:
        import asyncio

        from particles.config import get_config
        from particles.llm import CompletionRequest
        from particles.llm import pool as pool_mod

        get_config().consolidation.extract_db_concurrency = 1
        dispatches: list[int] = []

        async def slow_batch(
            purpose: str, requests: list[CompletionRequest], **kwargs: Any
        ) -> tuple[list[str | None], str]:
            dispatches.append(len(requests))
            await asyncio.sleep(1.0)  # twice the pool timeout
            return [_ONE_CLAIM] * len(requests), "anthropic:test-model"

        monkeypatch.setattr(pool_mod, "complete_many_with_provider_model", slow_batch)
        batch = await small_pool_store(6)

        report = ConsolidationReport()
        report.pending_total = len(batch)
        extracted = await consolidation_mod._pass_extract_pooled(batch, report)

        # One merged batch for all six, and nothing timed out on the pool.
        assert dispatches == [6]
        assert report.pending_failed == 0
        assert report.pending_retry == 0
        assert extracted == 6
        assert await _statuses(batch) == ["COMPLETE"] * 6

    @pytest.mark.asyncio
    async def test_a_failure_after_the_batch_leaves_the_snapshot_pending(
        self, monkeypatch: pytest.MonkeyPatch, small_pool_store: Any
    ) -> None:
        """A failure in the apply phase used to strand the snapshot IN_PROGRESS.

        The claim release covered only the LLM call, so the six pool timeouts
        of one night were the six stale claims the next night reset.
        """
        from particles.ingest import pipeline as pipeline_mod
        from particles.llm import CompletionRequest
        from particles.llm import pool as pool_mod

        async def batch_reply(
            purpose: str, requests: list[CompletionRequest], **kwargs: Any
        ) -> tuple[list[str | None], str]:
            return [_ONE_CLAIM] * len(requests), "anthropic:test-model"

        monkeypatch.setattr(pool_mod, "complete_many_with_provider_model", batch_reply)
        batch = await small_pool_store(2)
        doomed_entry = batch[0][0]
        real_load = pipeline_mod.get_active_particles_for_entry

        async def failing_load(session: Any, entry_id: str) -> Any:
            if entry_id == doomed_entry:
                raise RuntimeError("the store went away")
            return await real_load(session, entry_id)

        monkeypatch.setattr(pipeline_mod, "get_active_particles_for_entry", failing_load)

        report = ConsolidationReport()
        report.pending_total = len(batch)
        await consolidation_mod._pass_extract_pooled(batch, report)

        assert report.pending_failed == 1
        assert report.pending_extracted == 1
        assert report.pending_remaining == 1
        assert await _statuses(batch) == ["PENDING", "COMPLETE"]


# ---------------------------------------------------------------------------
# The run-level batch-wait budget
# ---------------------------------------------------------------------------


class TestBatchWaitBudget:
    @pytest.mark.asyncio
    async def test_a_spent_budget_is_disclosed_per_pass_and_in_the_run_record(
        self,
        db_session: AsyncSession,
        cycle_env: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An extract batch that eats the budget sends the census set sequential, said twice.

        The census still runs and still computes the same thing; the pass
        reads ``ran`` with a clause, the record carries a ``batch_wait`` object
        naming where the budget ran out, and the night is not degraded.
        """
        from types import SimpleNamespace

        import anthropic

        from particles import llm
        from particles.config import get_config
        from particles.llm import CompletionRequest
        from particles.llm.batch_budget import current_batch_wait_budget

        async def fake_extract(session: AsyncSession, report: ConsolidationReport) -> int:
            budget = current_batch_wait_budget()
            assert budget is not None
            # The pooled extraction batch waited out the whole budget.
            budget.charge(3500, requests=12, cut_short=True)
            # The cut batch ended in its grace with 9 of 12 answered.
            budget.record_cancellation(kept=9, rerun=0, lost=3)
            return 12

        cards = consolidation_mod.collect_cards.return_value  # type: ignore[attr-defined]

        async def census_that_probes(*_a: object, **_kw: object) -> list[CurationCard]:
            probes = [CompletionRequest(prompt=f"pair {i}") for i in range(4)]
            await llm.complete_many("semantic_lint", probes, max_tokens=10, latency_tolerant=True)
            return list(cards)

        client = MagicMock(spec=anthropic.Anthropic)
        client.messages.create.side_effect = [
            SimpleNamespace(content=[SimpleNamespace(text="no")], stop_reason="end_turn")
            for _ in range(4)
        ]
        monkeypatch.setattr(consolidation_mod, "_pass_extract", fake_extract)
        consolidation_mod.collect_cards.side_effect = census_that_probes  # type: ignore[attr-defined]
        get_config().llm.batch.min_requests = 2
        llm.set_client(client)
        try:
            report = await run_consolidation(db_session, scope="store")
        finally:
            llm.set_client(None)

        client.messages.batches.create.assert_not_called()
        assert client.messages.create.call_count == 4
        passes = {p.name: p for p in report.passes}
        assert passes["extract"].batch_wait == (
            "1 batch(es) cut short by the batch-wait budget; "
            "1 batch(es) cancelled (9 kept, 0 re-run, 3 unavailable)"
        )
        assert passes["census"].status == "ran"
        assert passes["census"].batch_wait == (
            "batch-wait budget spent: 1 set(s), 4 request(s) sequential at full price"
        )
        assert report.batch_wait is not None
        assert report.batch_wait.exhausted_in_pass == "extract"
        assert report.batch_wait.sequential_requests == 4
        assert report.semantic_degraded is False
        rendered = render_consolidation_report(report)
        assert "batch-wait budget: waited 3500s of 3600s, spent during extract" in rendered
        assert "1 cancelled (9 request(s) kept, 0 re-run, 3 unavailable)" in rendered

        event = await latest_run_event(db_session)
        assert event is not None and event.payload is not None
        assert event.payload["batch_wait"]["exhausted_in_pass"] == "extract"
        assert event.payload["batch_wait"]["cut_short_requests"] == 12
        assert event.payload["batch_wait"]["cancelled_batches"] == 1
        assert event.payload["batch_wait"]["cancelled_kept"] == 9
        assert event.payload["batch_wait"]["cancelled_lost"] == 3
        recorded = {p["name"]: p for p in event.payload["passes"]}
        assert recorded["census"]["status"] == "ran"
        assert "sequential at full price" in recorded["census"]["batch_wait"]
        assert "batch_wait" not in recorded["curation"]
        assert event.payload["semantic_degraded"] is False

    @pytest.mark.asyncio
    async def test_a_night_inside_the_budget_moves_nothing(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        report = await run_consolidation(db_session, scope="store")
        assert report.batch_wait is not None
        assert not report.batch_wait.moved_work
        assert all(p.batch_wait is None for p in report.passes)
        assert "batch-wait budget" not in render_consolidation_report(report)


# ---------------------------------------------------------------------------
# Progress events
# ---------------------------------------------------------------------------


class TestProgressEvents:
    """The cycle reports each pass's start, counter, batch wait and end; it never prints."""

    @pytest.mark.asyncio
    async def test_every_pass_starts_and_ends_in_order(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        from particles.core.progress import ProgressEvent
        from particles.operations.consolidation import PASS_ORDER, PassEnded

        events: list[ProgressEvent] = []
        report = await run_consolidation(db_session, on_progress=events.append)

        ended = [e for e in events if isinstance(e, PassEnded)]
        assert [e.label for e in ended] == [p.name for p in report.passes]
        assert [e.label for e in ended] == [n for n in PASS_ORDER if n in {e.label for e in ended}]
        for e in ended:
            assert e.total == len(PASS_ORDER)
            assert e.done == PASS_ORDER.index(e.label) + 1
        started = [e for e in events if e.phase == "pass"]
        assert [e.label for e in started] == [
            p.name for p in report.passes if p.status != "skipped"
        ]
        census = next(e for e in ended if e.label == "census")
        assert census.outcome == "ok"
        assert "contradiction" in census.summary
        assert "1 duplicate pair" in census.summary

    @pytest.mark.asyncio
    async def test_the_report_is_the_same_with_or_without_a_callback(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        # Store scope both times, so the second run is not delta-scoped by the first.
        silent = await run_consolidation(db_session, scope="store")
        watched = await run_consolidation(db_session, scope="store", on_progress=lambda _e: None)

        # The second run's last census is the first run.
        volatile = {
            "started_at",
            "completed_at",
            "event_id",
            "previous_run_at",
            "deltas",
            "census_last_ran_at",
            "census_next_due_at",
        }
        a = silent.model_dump(exclude=volatile)
        b = watched.model_dump(exclude=volatile)
        for passes in (a["passes"], b["passes"]):
            for entry in passes:
                entry.pop("duration_seconds")
        # Each run's measure window opens where the previous one closed.
        for dumped in (a, b):
            dumped["closure"].pop("window_start")
            dumped["closure"].pop("window_end")
        assert a == b

    @pytest.mark.asyncio
    async def test_a_raising_callback_never_fails_the_cycle(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        def _boom(_event: object) -> None:
            raise RuntimeError("terminal went away")

        report = await run_consolidation(db_session, on_progress=_boom)
        assert report.outcome == "ran"
        assert report.failed_passes() == []

    @pytest.mark.asyncio
    async def test_the_pooled_extract_counter_advances(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.core.progress import ProgressEvent
        from particles.operations import extract as extract_mod

        TestPassExtractPooled._fake_session_scope(monkeypatch)

        async def fake_extract(*args: Any, **kwargs: Any) -> list[Any]:
            return []

        monkeypatch.setattr(extract_mod, "extract_snapshot", fake_extract)
        events: list[ProgressEvent] = []
        token = consolidation_mod._active_progress.set(
            consolidation_mod._RunProgress(events.append)
        )
        try:
            await consolidation_mod._pass_extract_pooled(
                [("entry-a", "snap-1"), ("entry-b", "snap-2")], ConsolidationReport()
            )
        finally:
            consolidation_mod._active_progress.reset(token)

        counts = [(e.done, e.total, e.label) for e in events if e.phase == "extract"]
        assert counts == [(0, 2, "snapshots"), (1, 2, "snapshots"), (2, 2, "snapshots")]

    @pytest.mark.asyncio
    async def test_a_failed_pass_and_a_degraded_extract_are_marked(self) -> None:
        from particles.core.progress import ProgressEvent
        from particles.operations.consolidation import PassEnded

        events: list[ProgressEvent] = []
        progress = consolidation_mod._RunProgress(events.append)
        token = consolidation_mod._active_progress.set(progress)
        session = AsyncMock()
        report = ConsolidationReport()
        try:

            async def _fails() -> int:
                raise RuntimeError("provider down")

            async def _extract() -> int:
                consolidation_mod._count("extract", 20, 20, "snapshots")
                report.pending_retry = 20
                return 0

            await consolidation_mod._run_pass(session, report, "census", _fails)
            await consolidation_mod._run_pass(session, report, "extract", _extract)
            consolidation_mod._skip(report, "reconcile", "reconciliation is off")
        finally:
            consolidation_mod._active_progress.reset(token)

        ended = [e for e in events if isinstance(e, PassEnded)]
        assert [(e.label, e.outcome) for e in ended] == [
            ("census", "failed"),
            ("extract", "degraded"),
            ("reconcile", "skipped"),
        ]
        assert "provider down" in ended[0].summary
        assert ended[1].summary.startswith("0 of 20 extracted, 20 left for retry")
        assert ended[2].summary == "reconciliation is off"

    def test_a_pending_batch_publishes_its_wait_and_the_budget_left(self) -> None:
        from particles.core.progress import ProgressEvent
        from particles.llm.batch_budget import BatchWaitBudget
        from particles.operations.consolidation import BatchWaiting

        events: list[ProgressEvent] = []
        budget = BatchWaitBudget(budget_seconds=3600, min_remaining_seconds=300)
        budget.pass_name = "extract"
        budget.on_wait = consolidation_mod._batch_wait_publisher(
            consolidation_mod._RunProgress(events.append)
        )
        token = budget.begin_wait(3600)
        budget.end_wait(token)

        waits = [e for e in events if isinstance(e, BatchWaiting)]
        assert waits[0].total == 3600
        assert waits[0].label == "extract"
        assert 3590 < waits[0].budget_left_seconds <= 3600
        assert waits[0].budget_seconds == 3600
        assert waits[-1].total == 0  # closed: nothing pending


# ---------------------------------------------------------------------------
# The dollar budget
# ---------------------------------------------------------------------------


def _pass(report: ConsolidationReport, name: str) -> Any:
    return next(p for p in report.passes if p.name == name)


async def _projection() -> dict[str, Any]:
    return {"rendered": 1}


class TestSpendBudget:
    """``consolidation.budget_usd``: a pass that would pass it is skipped and disclosed."""

    @pytest.fixture
    def haiku_probes(self) -> None:
        # claude-haiku-4-5 lists at $1 / $5 per MTok: one probe at the audit's
        # figures (350 in, 250 out) is US$0.0016.
        from particles.config import ProviderSelection

        llm = get_config().llm
        for purpose in ("semantic_lint", "verification"):
            setattr(llm, purpose, ProviderSelection(provider="anthropic", model="claude-haiku-4-5"))

    @pytest.mark.asyncio
    async def test_unbounded_by_default_and_the_spend_is_still_recorded(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        report = await run_consolidation(db_session)
        assert report.budget is not None
        assert report.budget.budget_usd is None
        assert not report.budget.limited
        assert _pass(report, "reconcile").status == "ran"
        event = await latest_run_event(db_session)
        assert event is not None
        budget = (event.payload or {})["budget"]
        assert budget["budget_usd"] is None
        assert budget["spent_usd"] == 0.0
        assert budget["limited"] is False
        assert "spend budget" not in render_consolidation_report(report)

    @pytest.mark.asyncio
    async def test_a_spent_budget_skips_the_llm_passes_and_runs_the_rest(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        get_config().consolidation.budget_usd = 0.0
        monkeypatch.setattr(
            consolidation_mod, "count_supersession_candidates", AsyncMock(return_value=5)
        )
        report = await run_consolidation(db_session, projection_runner=_projection)

        reconcile = _pass(report, "reconcile")
        assert reconcile.status == "skipped"
        assert reconcile.detail is not None
        assert reconcile.detail.startswith("spent US$0.00 of US$0.00; pass 3 skipped")
        assert reconcile.detail.endswith("(consolidation.budget_usd)")
        assert consolidation_mod.reconcile_supersession.await_count == 0  # type: ignore[attr-defined]

        # The census still ran, structurally: its probe is the skipped part.
        assert _pass(report, "census").status == "ran"
        kwargs = consolidation_mod.collect_cards.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["semantic"] is False
        assert kwargs["contradiction_probe"] is None
        assert report.census_probe_skipped is not None
        assert "pass 6's contradiction probe skipped" in report.census_probe_skipped

        # The zero-LLM passes are never budgeted.
        for name in ("disclose", "curation", "projection"):
            assert _pass(report, name).status == "ran", name

        assert report.budget is not None
        assert report.budget.skipped_passes == ["reconcile", "census (contradiction probe)"]
        assert report.budget.limited
        rendered = render_consolidation_report(report)
        assert (
            "spend budget: spent US$0.00 of US$0.00 (consolidation.budget_usd); "
            "skipped reconcile, census (contradiction probe)"
        ) in rendered
        assert "contradictions   not probed this run (spent US$0.00 of US$0.00" in rendered
        assert "pass skipped: reconcile — spent US$0.00 of US$0.00" in rendered

    @pytest.mark.asyncio
    async def test_the_record_carries_the_budget_and_withholds_the_watermark(
        self, db_session: AsyncSession, cycle_env: Path
    ) -> None:
        """A run that left LLM work undone satisfies cadence, but not the delta watermark."""
        get_config().consolidation.budget_usd = 0.0
        await run_consolidation(db_session)

        event = await latest_run_event(db_session)
        assert event is not None
        budget = (event.payload or {})["budget"]
        assert budget["budget_usd"] == 0.0
        assert budget["limited"] is True
        assert budget["skipped_passes"] == ["census (contradiction probe)"]
        assert await latest_run_event(db_session, successful_only=True) is not None
        assert (
            await latest_run_event(db_session, successful_only=True, exclude_degraded=True) is None
        )

    @pytest.mark.asyncio
    async def test_a_pass_whose_estimate_would_pass_the_budget_is_skipped(
        self,
        db_session: AsyncSession,
        cycle_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        haiku_probes: None,
    ) -> None:
        # 50 candidate pairs at the default cap of 50: US$0.08, over a US$0.05 budget.
        get_config().consolidation.budget_usd = 0.05
        monkeypatch.setattr(
            consolidation_mod, "count_supersession_candidates", AsyncMock(return_value=80)
        )
        report = await run_consolidation(db_session)

        reconcile = _pass(report, "reconcile")
        assert reconcile.status == "skipped"
        assert reconcile.detail == (
            "spent US$0.00 of US$0.05; pass 3 skipped (estimated US$0.08 for 50 LLM calls "
            "at list price) (consolidation.budget_usd)"
        )

    @pytest.mark.asyncio
    async def test_a_budget_that_fits_runs_every_pass_with_the_budget_in_scope(
        self, db_session: AsyncSession, cycle_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.llm.spend_budget import current_spend_budget

        get_config().consolidation.budget_usd = 100.0
        monkeypatch.setattr(
            consolidation_mod, "count_supersession_candidates", AsyncMock(return_value=5)
        )
        seen: list[tuple[float, str | None]] = []

        async def reconcile(_session: AsyncSession) -> dict[str, object]:
            budget = current_spend_budget()
            assert budget is not None
            seen.append((budget.budget_usd, budget.pass_name))
            return {"demoted": 0, "probed": 5, "candidate_pairs": 5}

        monkeypatch.setattr(consolidation_mod, "reconcile_supersession", reconcile)
        report = await run_consolidation(db_session)

        # The batch adapter sees the run's budget, labelled with the pass in flight.
        assert seen == [(100.0, "reconcile")]
        assert _pass(report, "reconcile").status == "ran"
        assert consolidation_mod.collect_cards.call_args.kwargs["semantic"] is True  # type: ignore[attr-defined]
        assert report.budget is not None and not report.budget.limited
        assert "spend budget: spent US$0.00 of US$100" in render_consolidation_report(report)


class TestBudgetAndCensusCadence:
    """A census counts toward its cadence only when its own probe ran in full."""

    @staticmethod
    async def _run(session: AsyncSession, budget: dict[str, Any] | None) -> None:
        payload: dict[str, Any] = {
            "format": 1,
            "actor": "memory-consolidate",
            "started_at": (NOW - timedelta(hours=1)).isoformat(),
            "completed_at": NOW.isoformat(),
            "semantic_degraded": False,
            "passes": [{"name": "census", "status": "ran"}],
            "census": {"cards": {}},
        }
        if budget is not None:
            payload["budget"] = budget
        await record_event(
            session,
            actor="memory-consolidate",
            event_type=OperatorEventType.CONSOLIDATION_RUN,
            payload=payload,
        )
        await session.commit()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("budget", "counts"),
        [
            (None, True),
            ({"limited": True, "skipped_passes": ["reconcile"], "skipped_requests": 0}, True),
            (
                {
                    "limited": True,
                    "skipped_passes": ["census (contradiction probe)"],
                    "skipped_requests": 0,
                },
                False,
            ),
            ({"limited": True, "skipped_passes": [], "skipped_requests": 4}, False),
        ],
    )
    async def test_which_budgeted_runs_satisfy_the_cadence(
        self, db_session: AsyncSession, budget: dict[str, Any] | None, counts: bool
    ) -> None:
        from particles.operations.consolidation import latest_census_event

        await self._run(db_session, budget)
        found = await latest_census_event(db_session, actor="memory-consolidate")
        assert (found is not None) is counts


class TestPartialReadCounts:
    """A kept partial read and a waiting snapshot are counted and named."""

    def test_a_kept_partial_read_counts_as_retry_and_partial(self) -> None:
        from particles.ingest.pipeline import SnapshotOutcome

        report = ConsolidationReport()
        outcome = SnapshotOutcome(failed_calls=1, kept_calls=3, rebilled_calls=1)
        consolidation_mod._record_extraction(
            report, "entry-1", "snap-1", [], consolidation_mod._SnapshotRun(outcome=outcome)
        )
        assert (report.pending_retry, report.pending_partial, report.pending_extracted) == (
            1,
            1,
            0,
        )
        rendered = render_consolidation_report(report)
        assert "1 kept part of their read" in rendered
        assert "1 of them kept the chunks that answered" in rendered

    def test_a_waiting_snapshot_is_counted_and_not_retried(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.ingest.pipeline import SnapshotOutcome

        report = ConsolidationReport()
        outcome = SnapshotOutcome(skipped="waiting", waiting_on="held-snap", waiting_on_failed=True)
        with caplog.at_level("WARNING"):
            consolidation_mod._record_extraction(
                report, "entry-1", "snap-2", [], consolidation_mod._SnapshotRun(outcome=outcome)
            )
        assert (report.pending_waiting, report.pending_retry) == (1, 0)
        assert "FAILED snapshot holding a partial whole read" in caplog.text
        assert "restore the blob" in caplog.text
        assert "1 snapshot(s) wait on an earlier snapshot" in render_consolidation_report(report)
