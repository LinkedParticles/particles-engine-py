"""Tests for the as-of bitemporal read lens.

Covers:
  - the write-once ``retired_at`` stamp (first-departure semantics, multi-hop
    chains, born-retired rows never stamped);
  - stamp coverage per retirement path (pipeline demotion, reindex, retract
    cascade, lint ``--fix``, review, agent supersede/retract) — a future
    bypass path fails this suite;
  - the §1 visibility predicate per §2b ladder rung (stored / successor /
    event / valid_until / fail-closed + disclosure);
  - edge Ts (pre-store, future rejected, boundary ``asserted_at == T``);
  - decay and the recency window evaluated at T; federation threading;
  - the Pluto demo chain (§8.1);
  - the CLI / MCP parameter surfaces.

The ``UNDATED_RETIREMENT`` lint finding is tested in tests/test_lint.py.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.query.as_of import (
    AsOfView,
    RetirementIndex,
    SuccessorRef,
    is_once_believed_retirement,
    load_retirement_index,
)

EMB = (np.ones(4, dtype=np.float32) / 2.0).tolist()

T1980 = datetime(1980, 1, 1, tzinfo=UTC)
T1996 = datetime(1996, 3, 1, tzinfo=UTC)
T2000 = datetime(2000, 1, 1, tzinfo=UTC)
T2006 = datetime(2006, 8, 24, tzinfo=UTC)


def _particle(
    content: str,
    *,
    asserted_at: datetime | None = None,
    status: Status = Status.ACTIVE,
    status_reason: StatusReason | None = None,
    valid_until: datetime | None = None,
    supersedes: str | None = None,
    asserted_by: str = "test-agent",
    confidence: float = 0.9,
    entry_id: str = "entry-1",
    snapshot_id: str = "snap-1",
) -> Particle:
    kwargs: dict[str, Any] = {}
    if asserted_at is not None:
        kwargs["asserted_at"] = asserted_at
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=asserted_by,
        status=status,
        status_reason=status_reason,
        valid_until=valid_until,
        supersedes=supersedes,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_id,
                snapshot_id=snapshot_id,
            )
        ],
        **kwargs,
    )


async def _set_retired_at(session: Any, particle_id: str, value: datetime | None) -> None:
    """Backdate / clear the storage stamp directly (simulates pre-migration rows)."""
    from particles.store.particle_store import ParticleRow

    row = await session.get(ParticleRow, particle_id)
    assert row is not None
    row.retired_at = value
    await session.flush()


def _mock_embeddings() -> MagicMock:
    model = MagicMock()
    model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    return model


# ---------------------------------------------------------------------------
# §2a — the write-once retired_at stamp.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRetiredAtStamp:
    async def test_stamp_set_when_leaving_active(self, db_session: Any) -> None:
        from particles.store.particle_store import (
            get_retired_at,
            insert_particle,
            update_particle_status,
        )

        p = _particle("a belief")
        await insert_particle(db_session, p, EMB)
        assert await get_retired_at(db_session, p.id) is None

        before = datetime.now(UTC)
        await update_particle_status(
            db_session, p.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        stamped = await get_retired_at(db_session, p.id)
        assert stamped is not None
        assert before <= stamped.replace(tzinfo=UTC) <= datetime.now(UTC)

    async def test_write_once_multi_hop_keeps_first_instant(self, db_session: Any) -> None:
        """ACTIVE → PROVENANCE_STALE → SUPERSEDED keeps the departure-from-ACTIVE
        instant — a later hop must never overwrite it."""
        from particles.store.particle_store import (
            get_retired_at,
            insert_particle,
            update_particle_status,
        )

        p = _particle("a belief")
        await insert_particle(db_session, p, EMB)
        await update_particle_status(
            db_session, p.id, Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY
        )
        first = await get_retired_at(db_session, p.id)
        assert first is not None

        await update_particle_status(
            db_session, p.id, Status.SUPERSEDED, StatusReason.SUPERSEDED_BY_REINDEX
        )
        assert await get_retired_at(db_session, p.id) == first

    async def test_born_retired_never_stamped(self, db_session: Any) -> None:
        """A quarantine loser, born PROVENANCE_STALE / CONFLICT_PENDING,
        was never believed: no transition away from ACTIVE ever stamps it."""
        from particles.store.particle_store import (
            get_retired_at,
            insert_particle,
            update_particle_status,
        )

        loser = _particle(
            "quarantined claim",
            status=Status.PROVENANCE_STALE,
            status_reason=StatusReason.CONFLICT_PENDING,
        )
        await insert_particle(db_session, loser, EMB)
        assert await get_retired_at(db_session, loser.id) is None

        # Even a later hop (operator cleanup) does not stamp — the row never
        # left ACTIVE because it was never ACTIVE.
        await update_particle_status(
            db_session, loser.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        assert await get_retired_at(db_session, loser.id) is None


# ---------------------------------------------------------------------------
# §2a — stamp coverage: every retirement path leaves retired_at set.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRetirementStampCoverage:
    """Each real retirement path routes through the choke point and stamps.

    A future code path that retires particles without going through
    ``update_particle_status`` will not stamp — and should be caught by the
    ``UNDATED_RETIREMENT`` lint finding plus a failure here when its path is
    added to this class.
    """

    async def test_pipeline_trust_demotion_stamps(self, db_session: Any, tmp_path: Any) -> None:
        """§6.6 trust-rung demotion (LOWER_TRUST_SOURCE) via extract_snapshot."""
        from particles.store.particle_store import get_particle, get_retired_at
        from tests.test_extract import _drive_conflict

        _written, seed_id, _entry = await _drive_conflict(
            db_session, tmp_path, has_signal=True, score_new=0.9, score_existing=0.1
        )
        seed = await get_particle(db_session, seed_id)
        assert seed is not None and seed.status == Status.PROVENANCE_STALE
        assert await get_retired_at(db_session, seed_id) is not None

    async def test_reindex_supersession_stamps(self, db_session: Any) -> None:
        from particles.operations import reindex as reindex_mod
        from particles.store.particle_store import get_retired_at, insert_particle

        p = _particle("old extraction", entry_id="entry-r", snapshot_id="snap-r")
        await insert_particle(db_session, p, EMB)
        await db_session.commit()

        with patch.object(reindex_mod, "extract_snapshot", new=AsyncMock(return_value=[])):
            await reindex_mod._reindex_snapshot(db_session, "entry-r", "snap-r", None)

        assert await get_retired_at(db_session, p.id) is not None

    async def test_retract_cascade_stamps(self, db_session: Any) -> None:
        from particles.operations.retract import retract_entry
        from particles.store.particle_store import get_retired_at, insert_particle

        p = _particle("sourced claim", entry_id="entry-x")
        await insert_particle(db_session, p, EMB)
        await db_session.commit()

        result = await retract_entry(db_session, "entry-x", reason="source withdrawn")
        assert p.id in result.retracted_ids
        assert await get_retired_at(db_session, p.id) is not None

    async def test_lint_fix_stamps(self, db_session: Any) -> None:
        from particles.operations.lint.staleness import _check_staleness
        from particles.store.particle_store import get_retired_at, insert_particle

        p = _particle("expired claim", valid_until=datetime(2001, 1, 1, tzinfo=UTC))
        await insert_particle(db_session, p, EMB)
        await db_session.flush()

        findings = await _check_staleness(db_session, fix=True)
        assert any(f.particle_id == p.id for f in findings)
        assert await get_retired_at(db_session, p.id) is not None

    async def test_review_resolution_stamps(self, db_session: Any) -> None:
        from particles.core.schema import ResolutionAction
        from particles.operations.review import resolve
        from particles.store.particle_store import get_retired_at, insert_particle

        a = _particle("claim A")
        b = _particle("claim B")
        inc = Particle(
            content="A conflicts with B",
            confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="lint",
            status=Status.INCONSISTENCY,
            provenance=[
                ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=a.id),
                ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=b.id),
            ],
        )
        for particle in (a, b, inc):
            await insert_particle(db_session, particle, EMB)
        await db_session.commit()

        await resolve(db_session, inc.id, ResolutionAction.PREFER_A, reviewer_id="rev-1")
        # The demoted loser (B) left ACTIVE and carries the stamp.
        assert await get_retired_at(db_session, b.id) is not None

    async def test_agent_supersede_and_retract_stamp(self, db_session: Any) -> None:
        from particles.config import get_config
        from particles.operations import agent_write
        from particles.store.particle_store import get_retired_at, insert_particle

        identity = get_config().mcp.write.asserter_identity
        pred = _particle("agent belief v1", asserted_by=identity)
        victim = _particle("agent belief to retract", asserted_by=identity)
        for particle in (pred, victim):
            await insert_particle(db_session, particle, EMB)
        await db_session.commit()

        successor = _particle("agent belief v2", asserted_by=identity, supersedes=pred.id)
        with patch.object(
            agent_write,
            "_construct_and_insert",
            new=AsyncMock(return_value=(successor.id, successor)),
        ):
            await agent_write.supersede_belief(
                db_session,
                store="default",
                supersedes_id=pred.id,
                content="agent belief v2",
                subject_names=[],
                confidence=0.9,
            )
        assert await get_retired_at(db_session, pred.id) is not None

        await agent_write.retract_belief(
            db_session, store="default", particle_id=victim.id, reason="wrong"
        )
        assert await get_retired_at(db_session, victim.id) is not None


# ---------------------------------------------------------------------------
# §1 / §2b — the visibility predicate and the reconstruction ladder (pure).
# ---------------------------------------------------------------------------


def _view(
    as_of: datetime,
    successors: dict[str, SuccessorRef] | None = None,
    events: dict[str, datetime] | None = None,
) -> AsOfView:
    return AsOfView(
        as_of=as_of,
        index=RetirementIndex(
            successor_by_predecessor=successors or {},
            event_retired_at=events or {},
        ),
    )


class TestVisibilityPredicate:
    def test_active_visible_between_assertion_and_now(self) -> None:
        p = _particle("held belief", asserted_at=T1996)
        ev = _view(T2000).evaluate(p, None)
        assert ev.visible and ev.note is None and not ev.excluded_undatable

    def test_asserted_after_t_invisible(self) -> None:
        p = _particle("later belief", asserted_at=T2006)
        assert not _view(T2000).evaluate(p, None).visible

    def test_boundary_asserted_at_equals_t_is_visible(self) -> None:
        p = _particle("boundary belief", asserted_at=T2000)
        assert _view(T2000).evaluate(p, None).visible

    def test_naive_asserted_at_assumed_utc(self) -> None:
        p = _particle("naive ts", asserted_at=datetime(1996, 3, 1))  # noqa: DTZ001
        assert _view(T2000).evaluate(p, None).visible

    def test_rung0_stored_visible_until_stamp(self) -> None:
        p = _particle(
            "retired belief",
            asserted_at=T1996,
            status=Status.SUPERSEDED,
            status_reason=StatusReason.EXPLICIT_SUPERSESSION,
        )
        ev = _view(T2000).evaluate(p, T2006)
        assert ev.visible
        assert ev.note is not None
        assert ev.note.basis == "stored"
        assert ev.note.retired_at == T2006
        assert ev.note.status is Status.SUPERSEDED
        # After the stamp, no longer believed.
        assert not _view(datetime(2010, 1, 1, tzinfo=UTC)).evaluate(p, T2006).visible
        # Retired exactly at T: the instant belongs to the successor.
        assert not _view(T2006).evaluate(p, T2006).visible

    def test_rung1_successor_pointer_with_payload(self) -> None:
        p = _particle(
            "planet claim",
            asserted_at=T1996,
            status=Status.SUPERSEDED,
            status_reason=StatusReason.EXPLICIT_SUPERSESSION,
        )
        successors = {p.id: SuccessorRef(id="succ-1", content="dwarf claim", asserted_at=T2006)}
        ev = _view(T2000, successors=successors).evaluate(p, None)
        assert ev.visible
        assert ev.note is not None
        assert ev.note.basis == "successor"
        assert ev.note.retired_at == T2006
        assert ev.note.successor is not None
        assert ev.note.successor.id == "succ-1"
        assert ev.note.successor.content == "dwarf claim"

    def test_rung2_operator_event(self) -> None:
        p = _particle(
            "retracted belief",
            asserted_at=T1996,
            status=Status.RETRACTED,
            status_reason=StatusReason.EXPLICIT_RETRACTION,
        )
        ev = _view(T2000, events={p.id: T2006}).evaluate(p, None)
        assert ev.visible
        assert ev.note is not None
        assert ev.note.basis == "event"
        assert ev.note.retired_at == T2006
        assert ev.note.successor is None

    def test_rung3_valid_until(self) -> None:
        p = _particle(
            "expired belief",
            asserted_at=T1996,
            status=Status.PROVENANCE_STALE,
            status_reason=StatusReason.VALIDITY_EXPIRED,
            valid_until=T2006,
        )
        ev = _view(T2000).evaluate(p, None)
        assert ev.visible
        assert ev.note is not None
        assert ev.note.basis == "valid_until"
        assert ev.note.retired_at == T2006

    def test_rung4_fail_closed_and_disclosed(self) -> None:
        """An automated demotion with no stamp, pointer, or event is undatable:
        never visible at any T, and counted for the disclosure line."""
        p = _particle(
            "trust-demoted belief",
            asserted_at=T1996,
            status=Status.PROVENANCE_STALE,
            status_reason=StatusReason.LOWER_TRUST_SOURCE,
        )
        ev = _view(T2000).evaluate(p, None)
        assert not ev.visible
        assert ev.excluded_undatable

    def test_inconsistency_never_visible(self) -> None:
        p = _particle("conflict record", asserted_at=T1996, status=Status.INCONSISTENCY)
        ev = _view(T2000).evaluate(p, None)
        assert not ev.visible and not ev.excluded_undatable

    def test_born_retired_never_visible_and_not_counted(self) -> None:
        p = _particle(
            "quarantined loser",
            asserted_at=T1996,
            status=Status.PROVENANCE_STALE,
            status_reason=StatusReason.CONFLICT_PENDING,
        )
        ev = _view(T2000).evaluate(p, None)
        assert not ev.visible and not ev.excluded_undatable

    def test_valid_until_evaluated_against_t_not_now(self) -> None:
        """A claim valid until 2007 was in force in 2000 — and out of force at a
        later T even if its status never transitioned."""
        p = _particle(
            "valid until 2007", asserted_at=T1996, valid_until=datetime(2007, 1, 1, tzinfo=UTC)
        )
        assert _view(T2000).evaluate(p, None).visible
        assert not _view(datetime(2008, 1, 1, tzinfo=UTC)).evaluate(p, None).visible

    def test_is_once_believed_retirement_helper(self) -> None:
        assert is_once_believed_retirement(Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION)
        assert is_once_believed_retirement(Status.PROVENANCE_STALE, None)
        assert not is_once_believed_retirement(Status.ACTIVE, None)
        assert not is_once_believed_retirement(Status.INCONSISTENCY, None)
        assert not is_once_believed_retirement(
            Status.PROVENANCE_STALE, StatusReason.CONFLICT_PENDING
        )


@pytest.mark.asyncio
async def test_load_retirement_index_maps(db_session: Any) -> None:
    """The successor map keys on ``supersedes`` and the event map on the four
    retirement-dating operator event types."""
    from particles.store.event_store import EventRefKind, OperatorEventType, record_event
    from particles.store.particle_store import insert_particle

    pred = _particle("old", asserted_at=T1996)
    succ = _particle("new", asserted_at=T2006, supersedes=pred.id)
    retracted = _particle("dropped", asserted_at=T1996)
    for p in (pred, succ, retracted):
        await insert_particle(db_session, p, EMB)
    event = await record_event(
        db_session,
        actor="test",
        event_type=OperatorEventType.PARTICLE_RETRACTED,
        refs=[(EventRefKind.PARTICLE, retracted.id)],
    )
    # A non-retirement event type must not date anything.
    await record_event(
        db_session,
        actor="test",
        event_type=OperatorEventType.PARTICLE_TAGGED,
        refs=[(EventRefKind.PARTICLE, pred.id)],
    )
    await db_session.commit()

    index = await load_retirement_index(db_session)
    assert index.successor_by_predecessor[pred.id].id == succ.id
    assert index.successor_by_predecessor[pred.id].content == "new"
    assert index.event_retired_at[retracted.id] == event.occurred_at
    assert pred.id not in index.event_retired_at


# ---------------------------------------------------------------------------
# QueryRequest validation (§4).
# ---------------------------------------------------------------------------


class TestAsOfRequestValidation:
    def test_future_as_of_rejected(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="future"):
            QueryRequest(question="q", as_of=datetime.now(UTC) + timedelta(days=1))

    def test_naive_as_of_assumed_utc(self) -> None:
        req = QueryRequest(question="q", as_of=datetime(2000, 1, 1))  # noqa: DTZ001
        assert req.as_of == T2000

    def test_unset_as_of_default(self) -> None:
        assert QueryRequest(question="q").as_of is None


# ---------------------------------------------------------------------------
# Query integration (§5) — notes, disclosure, empty-at-T, decay/window at T.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsOfQuery:
    async def _run_query(self, session: Any, monkeypatch: Any, request: QueryRequest) -> Any:
        import particles.operations.query.main as qmain
        from particles import embeddings as ep

        original = ep._embedding_model
        ep.set_embedding_model(_mock_embeddings())
        monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="answer"))
        try:
            return await qmain.query(session, request)
        finally:
            ep.set_embedding_model(original)

    async def test_retired_hit_visible_at_t_with_note(
        self, db_session: Any, monkeypatch: Any
    ) -> None:
        from particles.store.particle_store import insert_particle, update_particle_status

        old = _particle("the old truth", asserted_at=T1996)
        await insert_particle(db_session, old, EMB)
        await update_particle_status(
            db_session, old.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
        )
        new = _particle("the new truth", supersedes=old.id)
        await insert_particle(db_session, new, EMB)
        await db_session.commit()

        # Today: only the successor.
        result_now = await self._run_query(
            db_session, monkeypatch, QueryRequest(question="truth?", top_k=5)
        )
        assert [p.id for p in result_now.particles] == [new.id]
        assert result_now.as_of is None
        assert result_now.as_of_notes == []

        # As of 2000: only the predecessor, annotated with the crossing
        # (rung 0 — the stamp was written by update_particle_status).
        result_then = await self._run_query(
            db_session, monkeypatch, QueryRequest(question="truth?", top_k=5, as_of=T2000)
        )
        assert [p.id for p in result_then.particles] == [old.id]
        assert result_then.as_of == T2000
        note = result_then.as_of_notes[0]
        assert note is not None
        assert note.status is Status.SUPERSEDED
        assert note.basis == "stored"
        assert note.successor is not None and note.successor.id == new.id
        assert result_then.as_of_excluded_undatable == 0

    async def test_active_hit_has_none_note(self, db_session: Any, monkeypatch: Any) -> None:
        from particles.store.particle_store import insert_particle

        p = _particle("still believed", asserted_at=T1996)
        await insert_particle(db_session, p, EMB)
        await db_session.commit()

        result = await self._run_query(
            db_session, monkeypatch, QueryRequest(question="q", top_k=5, as_of=T2000)
        )
        assert [p2.id for p2 in result.particles] == [p.id]
        assert result.as_of_notes == [None]

    async def test_undatable_exclusion_disclosed(self, db_session: Any, monkeypatch: Any) -> None:
        from particles.store.particle_store import insert_particle, update_particle_status

        undatable = _particle("undatable retirement", asserted_at=T1996)
        await insert_particle(db_session, undatable, EMB)
        await update_particle_status(
            db_session, undatable.id, Status.PROVENANCE_STALE, StatusReason.LOWER_TRUST_SOURCE
        )
        # Simulate a pre-migration row: no stamp, no successor, no event.
        await _set_retired_at(db_session, undatable.id, None)
        await db_session.commit()

        result = await self._run_query(
            db_session, monkeypatch, QueryRequest(question="q", top_k=5, as_of=T2000)
        )
        assert result.particles == []
        assert result.as_of_excluded_undatable == 1

    async def test_empty_store_at_t_returns_honest_note(self, db_session: Any) -> None:
        """A T before the store's first assertion is a legitimate question: the
        standard empty surface with an honest as-of message (no LLM call)."""
        import particles.operations.query.main as qmain
        from particles import embeddings as ep
        from particles.store.particle_store import insert_particle

        p = _particle("later belief", asserted_at=T1996)
        await insert_particle(db_session, p, EMB)
        await db_session.commit()

        original = ep._embedding_model
        ep.set_embedding_model(_mock_embeddings())
        try:
            result = await qmain.query(
                db_session, QueryRequest(question="anything?", top_k=5, as_of=T1980)
            )
        finally:
            ep.set_embedding_model(original)
        assert result.particles == []
        assert "held no beliefs" in result.answer
        assert T1980.isoformat() in result.answer

    async def test_recency_window_cuts_relative_to_t(
        self, db_session: Any, monkeypatch: Any
    ) -> None:
        from particles.store.particle_store import insert_particle

        fresh_at_t = _particle("asserted just before T", asserted_at=T2000 - timedelta(days=5))
        stale_at_t = _particle("asserted long before T", asserted_at=T2000 - timedelta(days=90))
        for p in (fresh_at_t, stale_at_t):
            await insert_particle(db_session, p, EMB)
        await db_session.commit()

        result = await self._run_query(
            db_session,
            monkeypatch,
            QueryRequest(question="q", top_k=5, as_of=T2000, recency_window_days=30),
        )
        assert [p.id for p in result.particles] == [fresh_at_t.id]

    async def test_decay_evaluated_at_t(self, db_session: Any) -> None:
        """_gather_scored passes now=T into the decay kernel."""
        from particles.operations.query.main import _gather_scored
        from particles.operations.query.source_trust import load_trust_policy
        from particles.store.particle_store import insert_particle

        p = _particle("dated belief", asserted_at=T1996)
        await insert_particle(db_session, p, EMB)
        await db_session.commit()

        calls: list[datetime | None] = []

        class _RecordingDecay:
            def recency_factor(
                self,
                content_published_at: datetime | None,
                source_type: str,
                uri_r: str | None,
                now: datetime | None = None,
            ) -> float:
                calls.append(now)
                return 1.0

        trust_policy = await load_trust_policy(db_session)
        scored, _pubs, _notes, _excluded, _claim_stats = await _gather_scored(
            db_session,
            QueryRequest(question="q", as_of=T2000),
            None,
            trust_policy,
            _RecordingDecay(),  # type: ignore[arg-type]
        )
        assert [s[0].id for s in scored] == [p.id]
        assert calls == [T2000]

        # Without as_of the kernel keeps its wall-clock default (now=None).
        calls.clear()
        await _gather_scored(
            db_session,
            QueryRequest(question="q"),
            None,
            trust_policy,
            _RecordingDecay(),  # type: ignore[arg-type]
        )
        assert calls == [None]


async def test_query_federated_threads_as_of(tmp_path: Any, monkeypatch: Any) -> None:
    """The viewer's single as_of applies to every store's candidates (§4)."""
    import particles._orm_modules  # noqa: F401
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.db import DEFAULT_STORE, Base, get_engine, reset_engine, session_scope
    from particles.store.particle_store import insert_particle, update_particle_status

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/default.db"
    cfg.storage.stores = {"other": f"sqlite+aiosqlite:///{tmp_path}/other.db"}

    for handle in (DEFAULT_STORE, "other"):
        engine = get_engine(handle)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    still_active = _particle("still active fact", asserted_at=T1996)
    retired = _particle("since-retired fact", asserted_at=T1996)
    successor = _particle("replacement fact", supersedes=retired.id)
    async with session_scope(DEFAULT_STORE) as s:
        await insert_particle(s, still_active, EMB)
        await s.commit()
    async with session_scope("other") as s:
        await insert_particle(s, retired, EMB)
        await update_particle_status(
            s, retired.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
        )
        await insert_particle(s, successor, EMB)
        await s.commit()

    original = ep._embedding_model
    ep.set_embedding_model(_mock_embeddings())
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="answer"))
    try:
        result = await qmain.query_federated(
            [DEFAULT_STORE, "other"], QueryRequest(question="facts?", top_k=10, as_of=T2000)
        )
        ids = {p.id for p in result.particles}
        # At T both original beliefs held; the successor (asserted now) did not.
        assert ids == {still_active.id, retired.id}
        notes_by_id = dict(zip([p.id for p in result.particles], result.as_of_notes, strict=True))
        assert notes_by_id[still_active.id] is None
        retired_note = notes_by_id[retired.id]
        assert retired_note is not None and retired_note.status is Status.SUPERSEDED
    finally:
        ep.set_embedding_model(original)
        for handle in (DEFAULT_STORE, "other"):
            await get_engine(handle).dispose()
        reset_engine()


# ---------------------------------------------------------------------------
# §8.1 — the Pluto demo chain.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pluto_demo_chain(db_session: Any, monkeypatch: Any) -> None:
    """The owner-chosen demo: `--as-of 2000-01-01` answers "planet" with the
    supersession crossing (basis ``successor``, retired 2006-08-24); the same
    query today answers "dwarf planet"; `--as-of 1980-01-01` answers empty."""
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.corpus.deposit import deposit_text
    from particles.store.particle_store import insert_particle, update_particle_status

    pre_entry, pre_snap = await deposit_text(
        db_session,
        "Pluto is the ninth planet of the Solar System.",
        deposited_by="fixture",
        source_type="WEB_PAGE",
    )
    planet = _particle(
        "Pluto is the ninth planet of the Solar System.",
        asserted_at=T1996,
        entry_id=pre_entry,
        snapshot_id=pre_snap,
    )
    await insert_particle(db_session, planet, EMB)

    iau_entry, iau_snap = await deposit_text(
        db_session,
        "IAU 2006: Pluto is a dwarf planet.",
        deposited_by="fixture",
        source_type="WEB_PAGE",
    )
    await update_particle_status(
        db_session, planet.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
    )
    # Simulate pre-migration history: the chain must date through the
    # successor pointer (rung 1), not the stamp.
    await _set_retired_at(db_session, planet.id, None)
    dwarf = _particle(
        "Pluto is a dwarf planet (IAU 2006 reclassification).",
        asserted_at=T2006,
        supersedes=planet.id,
        entry_id=iau_entry,
        snapshot_id=iau_snap,
    )
    await insert_particle(db_session, dwarf, EMB)
    await db_session.commit()

    original = ep._embedding_model
    ep.set_embedding_model(_mock_embeddings())
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="answer"))
    try:
        question = "How many planets are in the Solar System?"

        then = await qmain.query(db_session, QueryRequest(question=question, top_k=5, as_of=T2000))
        assert [p.id for p in then.particles] == [planet.id]
        note = then.as_of_notes[0]
        assert note is not None
        assert note.basis == "successor"
        assert note.retired_at == T2006
        assert note.successor is not None
        assert note.successor.id == dwarf.id
        assert "dwarf planet" in note.successor.content

        now = await qmain.query(db_session, QueryRequest(question=question, top_k=5))
        assert [p.id for p in now.particles] == [dwarf.id]

        before = await qmain.query(
            db_session, QueryRequest(question=question, top_k=5, as_of=T1980)
        )
        assert before.particles == []
    finally:
        ep.set_embedding_model(original)


# ---------------------------------------------------------------------------
# Surfaces — CLI flag parsing and the MCP tool parameter.
# ---------------------------------------------------------------------------


class TestCliAsOf:
    def test_invalid_as_of_rejected(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["query", "anything?", "--as-of", "not-a-date"])
        assert result.exit_code == 1
        assert "Invalid --as-of" in result.output

    def test_future_as_of_rejected(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        future = (datetime.now(UTC) + timedelta(days=30)).date().isoformat()
        result = CliRunner().invoke(app, ["query", "anything?", "--as-of", future])
        assert result.exit_code == 1
        assert "future" in result.output


@pytest.mark.asyncio
class TestMcpAsOf:
    async def test_as_of_parsed_into_request(self, db_session: Any) -> None:
        from particles.core.schema import QueryResponse
        from particles.mcp.tools.query import query as mcp_query

        captured: list[QueryRequest] = []

        async def _fake_query(session: Any, request: QueryRequest) -> QueryResponse:
            captured.append(request)
            return QueryResponse(answer="ok", particles=[], effective_confidences=[])

        with patch("particles.operations.query.query", new=_fake_query):
            out = await mcp_query("q?", as_of="2000-01-01")
        assert captured[0].as_of == T2000
        assert out["answer"] == "ok"
        # The response surface carries the as_of echo fields (set by the real
        # operation; the fake above returns a default-shaped response).
        assert "as_of" in out and "as_of_notes" in out and "as_of_excluded_undatable" in out

    async def test_invalid_as_of_raises(self) -> None:
        from particles.mcp.tools.query import query as mcp_query

        with pytest.raises(ValueError, match="Invalid as_of"):
            await mcp_query("q?", as_of="not-a-date")
