"""Tests for the dream-cycle consolidation operation.

Pins the deterministic parts with mocked seams: the §3 pass composition and
ordering, the §4 delta-scope watermark computation, the §6 degradation
disclosure ("not probed this run", never "0"), the §8 lockfile protocol
(acquire / held-skip / stale reclaim) and ``--if-due`` guard, the §7
``CONSOLIDATION_RUN`` payload shape + delta report against a prior event, and
the interactive audit's recording seam. The LLM-priced pass internals are the
composed operations' own tests.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.operations import consolidation as consolidation_mod
from particles.operations.consolidation import (
    ConsolidationReport,
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
            {"name": "census", "status": "failed(boom)" if failed else "ran"},
        ],
        "census": {
            "cards": cards or {},
            "duplicate_candidate_pairs_total": duplicates_total,
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
                _card(CardKind.CONTESTED, "p2"),
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
        assert lock is not None
        data = json.loads(path.read_text())
        assert data["pid"] == os.getpid()
        release_cycle_lock(lock)
        assert not path.exists()
        release_cycle_lock(lock)  # idempotent

    def test_held_by_live_pid_is_not_reclaimed(self, tmp_path: Path) -> None:
        path = tmp_path / "consolidate.lock"
        first = acquire_cycle_lock(path, timeout_minutes=120)
        assert first is not None
        assert acquire_cycle_lock(path, timeout_minutes=120) is None

    def test_dead_pid_is_reclaimed(self, tmp_path: Path) -> None:
        path = tmp_path / "consolidate.lock"
        path.write_text(json.dumps({"pid": 2**30, "started_at": datetime.now(UTC).isoformat()}))
        lock = acquire_cycle_lock(path, timeout_minutes=120)
        assert lock is not None
        assert json.loads(path.read_text())["pid"] == os.getpid()

    def test_expired_lock_is_reclaimed(self, tmp_path: Path) -> None:
        path = tmp_path / "consolidate.lock"
        started = datetime.now(UTC) - timedelta(minutes=300)
        path.write_text(json.dumps({"pid": os.getpid(), "started_at": started.isoformat()}))
        assert acquire_cycle_lock(path, timeout_minutes=120) is not None

    def test_malformed_lock_is_stale(self, tmp_path: Path) -> None:
        path = tmp_path / "consolidate.lock"
        path.write_text("not json")
        assert acquire_cycle_lock(path, timeout_minutes=120) is not None

    def test_reclaim_race_never_deletes_a_fresh_rival_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # TOCTOU (correction v1.74.1): both contenders judge the old lock
        # stale; the winner reclaims and writes a FRESH lock while the loser
        # is still judging. The loser must back off, not unlink the winner's
        # live lock. Simulated by swapping the lockfile mid-judgement.
        path = tmp_path / "consolidate.lock"
        path.write_text(json.dumps({"pid": 2**30, "started_at": datetime.now(UTC).isoformat()}))
        rival = json.dumps({"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()})

        real_is_stale = consolidation_mod._lock_is_stale

        def racing_is_stale(p: Path, timeout_minutes: int) -> bool:
            verdict = real_is_stale(p, timeout_minutes)
            # Mid-judgement the rival reclaims the stale lock and re-creates.
            p.unlink()
            p.write_text(rival)
            return verdict

        monkeypatch.setattr(consolidation_mod, "_lock_is_stale", racing_is_stale)
        assert acquire_cycle_lock(path, timeout_minutes=120) is None  # lost the race
        assert json.loads(path.read_text()) == json.loads(rival)  # rival's lock intact

    @pytest.mark.asyncio
    async def test_held_lock_skips_run(self, db_session: AsyncSession, cycle_env: Path) -> None:
        held = acquire_cycle_lock(cycle_env, timeout_minutes=120)
        assert held is not None
        report = await run_consolidation(db_session)
        assert report.outcome == "skipped"
        assert report.skip_reason is not None
        assert "already running" in report.skip_reason
        # No run record was written for a lock skip.
        assert await latest_run_event(db_session) is None


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
            "census",
            "curation",
            "utility",
            "abstraction",  # pass 5b — skipped (default off) but slotted
            "projection",
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
            "contested": 1,
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
        assert not cycle_env.exists()


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
# Pass 5 shared behavioural budget (correction v1.74.1)
# ---------------------------------------------------------------------------


class TestUtilityBudget:
    @pytest.mark.asyncio
    async def test_behavioural_budget_accumulates_across_sessions(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # utility.mining.max_behavioural_calls is a per-RUN cap: with cap 3
        # and two sessions each wanting 2 calls, the run makes 3 calls total
        # (not 4) and discloses the exhaustion.
        from types import SimpleNamespace

        from particles.config import get_config
        from particles.operations.utility_mining import MiningResult

        get_config().utility.mining.max_behavioural_calls = 3

        entries = [
            SimpleNamespace(entry_id=f"e{i}", uri_r=f"claude-code://session/s{i}") for i in (1, 2)
        ]
        snap = SimpleNamespace(content_hash="h", archive_path="a", captured_at=NOW)
        monkeypatch.setattr("particles.corpus.store.list_entries", AsyncMock(return_value=entries))
        monkeypatch.setattr(
            "particles.corpus.store.list_snapshots_for_entry", AsyncMock(return_value=[snap])
        )
        monkeypatch.setattr(
            "particles.corpus.deposit.load_blob", lambda _h: b"[tool: Bash - transcript]"
        )
        monkeypatch.setattr(
            consolidation_mod, "get_particles_by_status", AsyncMock(return_value=[])
        )

        budgets_seen: list[int | None] = []

        async def fake_mine(
            session: AsyncSession,
            sid: str,
            text: str,
            actives: list[Particle],
            *,
            behavioural_matching: bool | None = None,
            max_behavioural_calls: int | None = None,
            latency_tolerant: bool = False,
        ) -> MiningResult:
            budgets_seen.append(max_behavioural_calls)
            budget = 0 if max_behavioural_calls is None else max_behavioural_calls
            calls = min(2, budget)  # each session WANTS 2 behavioural calls
            return MiningResult(
                literal=1,
                behavioural=calls,
                candidates=1,
                behavioural_calls=calls,
                behavioural_truncated=calls < 2,
            )

        monkeypatch.setattr(consolidation_mod, "mine_session", fake_mine)

        report = ConsolidationReport()
        calls = await consolidation_mod._pass_utility(
            db_session, report, watermark=None, behavioural=True
        )
        assert budgets_seen == [3, 1]  # the remainder is threaded, not the cap
        assert calls == 3
        assert report.utility_behavioural_calls == 3
        assert report.utility_sessions_mined == 2
        assert report.utility_behavioural_exhausted_after == 2
        rendered = render_consolidation_report(report)
        assert "behavioural budget exhausted after 2 of 2 sessions" in rendered
        assert "utility.mining.max_behavioural_calls" in rendered


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
        assert set(payload["providers"]) == {"extraction", "semantic_lint", "abstraction"}
        assert payload["providers"]["extraction"].startswith("anthropic:")
        pass_names = [p["name"] for p in payload["passes"]]
        assert pass_names == [
            "refresh",
            "extract",
            "reconcile",
            "census",
            "curation",
            "utility",
            "abstraction",
            "projection",
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
            cards={"contradiction": 1, "contested": 0, "stale": 3},
            duplicates_total=6,
        )
        report = await run_consolidation(db_session, scope="store")
        # Current census (mocked cards): 2 contradictions, 1 duplicate, 1 stale.
        assert report.previous_run_at == prior_completed
        assert report.deltas == {"contradictions": 1, "duplicates": -5, "stale": -2}
        rendered = render_consolidation_report(report)
        assert "(+1 since last run)" in rendered
        assert "previous run:" in rendered

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
        assert "12 remain — next run continues" in rendered

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
            revalidation=RevalidationCounts(checked=1, refreshed_entailed=1),
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

        rendered = render_consolidation_report(report)
        assert "abstraction      1 proposed, 1 revalidated" in rendered

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
        self, db_session: AsyncSession, cycle_env: Path, tmp_path: Path
    ) -> None:
        """End to end through the pass: edit on disk → new PENDING snapshot."""
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

        async def fake_pending(session: Any) -> list[tuple[str, str]]:
            return [("entry-a", "snap-1")]

        import particles.corpus.store as corpus_store

        monkeypatch.setattr(corpus_store, "list_pending_snapshots_oldest_first", fake_pending)

        report = ConsolidationReport()
        extracted = await consolidation_mod._pass_extract(db_session, report)

        assert serial_calls == [("entry-a", "snap-1")]
        assert extracted == 1
