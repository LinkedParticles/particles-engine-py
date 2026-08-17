"""Tests for `particles corpus retract`.

Covers the retract operation (store-level) and the CLI verb. The verb is the
non-destructive sibling of `corpus delete`: it transitions live particles to
RETRACTED while preserving the corpus entry, its snapshots, and the particles.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    Confidence,
    CorpusEntry,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.corpus.store import ExtractionStatus, WarcRecordType


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


async def _seed_entry(
    session: AsyncSession, *, uri: str = "https://example.com/x"
) -> tuple[str, str]:
    from particles.corpus.store import CorpusEntryRow, SnapshotRow

    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()), source_type="WEB_PAGE", uri_r=uri, deposited_by="test"
    )
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash="b" * 64,
        extraction_status=ExtractionStatus.COMPLETE,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    session.add(CorpusEntryRow.from_model(entry))
    session.add(SnapshotRow.from_model(snap, entry.entry_id))
    await session.flush()
    return entry.entry_id, snap.snapshot_id


async def _add_particle(
    session: AsyncSession,
    *,
    entry_id: str,
    snapshot_id: str,
    status: Status,
    content: str = "A claim.",
) -> str:
    from particles.store.particle_store import insert_particle

    pid = str(uuid.uuid4())
    await insert_particle(
        session,
        Particle(
            id=pid,
            content=content,
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            asserted_at=datetime.now(UTC),
            status=status,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=entry_id,
                    snapshot_id=snapshot_id,
                )
            ],
        ),
    )
    return pid


class TestRetractOperation:
    @pytest.mark.asyncio
    async def test_retracts_live_skips_dead(self, db_session: AsyncSession) -> None:
        from particles.operations.retract import retract_entry
        from particles.store.particle_store import get_particle

        eid, sid = await _seed_entry(db_session)
        a = await _add_particle(db_session, entry_id=eid, snapshot_id=sid, status=Status.ACTIVE)
        inc = await _add_particle(
            db_session, entry_id=eid, snapshot_id=sid, status=Status.INCONSISTENCY
        )
        sup = await _add_particle(
            db_session, entry_id=eid, snapshot_id=sid, status=Status.SUPERSEDED
        )
        await db_session.commit()

        result = await retract_entry(db_session, eid, reason="publisher correction")
        await db_session.commit()

        assert set(result.retracted_ids) == {a, inc}
        assert result.skipped == {"SUPERSEDED": 1}
        for pid in (a, inc):
            p = await get_particle(db_session, pid)
            assert p is not None and p.status == Status.RETRACTED
            assert p.status_reason == StatusReason.SOURCE_RETRACTED
        # the SUPERSEDED particle is untouched
        sp = await get_particle(db_session, sup)
        assert sp is not None and sp.status == Status.SUPERSEDED

    @pytest.mark.asyncio
    async def test_corpus_entry_and_snapshots_survive(self, db_session: AsyncSession) -> None:
        from particles.corpus.store import CorpusEntryRow, SnapshotRow
        from particles.operations.retract import retract_entry

        eid, sid = await _seed_entry(db_session)
        await _add_particle(db_session, entry_id=eid, snapshot_id=sid, status=Status.ACTIVE)
        await db_session.commit()

        await retract_entry(db_session, eid)
        await db_session.commit()

        assert await db_session.get(CorpusEntryRow, eid) is not None
        snaps = (
            (await db_session.execute(select(SnapshotRow).where(SnapshotRow.entry_id == eid)))
            .scalars()
            .all()
        )
        assert len(snaps) == 1

    @pytest.mark.asyncio
    async def test_idempotent_rerun(self, db_session: AsyncSession) -> None:
        from particles.operations.retract import retract_entry

        eid, sid = await _seed_entry(db_session)
        await _add_particle(db_session, entry_id=eid, snapshot_id=sid, status=Status.ACTIVE)
        await db_session.commit()

        first = await retract_entry(db_session, eid)
        await db_session.commit()
        assert len(first.retracted_ids) == 1

        second = await retract_entry(db_session, eid)
        await db_session.commit()
        assert second.retracted_ids == []

    @pytest.mark.asyncio
    async def test_later_snapshot_particles_included(self, db_session: AsyncSession) -> None:
        # A particle re-extracted from a *later* snapshot of the same entry is
        # bound by corpus_entry_id, so it is retracted too.
        from particles.operations.retract import retract_entry

        eid, sid = await _seed_entry(db_session)
        later_snapshot = str(uuid.uuid4())
        a = await _add_particle(db_session, entry_id=eid, snapshot_id=sid, status=Status.ACTIVE)
        b = await _add_particle(
            db_session, entry_id=eid, snapshot_id=later_snapshot, status=Status.ACTIVE
        )
        await db_session.commit()

        result = await retract_entry(db_session, eid)
        await db_session.commit()
        assert set(result.retracted_ids) == {a, b}

    @pytest.mark.asyncio
    async def test_logs_source_retracted_event(self, db_session: AsyncSession) -> None:
        from particles.operations.retract import retract_entry
        from particles.store.event_store import (
            EventRefKind,
            OperatorEventType,
            list_events,
        )

        eid, sid = await _seed_entry(db_session)
        a = await _add_particle(db_session, entry_id=eid, snapshot_id=sid, status=Status.ACTIVE)
        await db_session.commit()

        await retract_entry(db_session, eid, reason="NYT correction 2026-05-30")
        await db_session.commit()

        events = await list_events(db_session, ref_kind=EventRefKind.CORPUS_ENTRY, ref_id=eid)
        assert len(events) == 1
        assert events[0].event_type is OperatorEventType.SOURCE_RETRACTED
        assert events[0].reason == "NYT correction 2026-05-30"
        refs = {(r.ref_kind, r.ref_id) for r in events[0].refs}
        assert (EventRefKind.PARTICLE, a) in refs

    @pytest.mark.asyncio
    async def test_co_evidential_relation_removed(self, db_session: AsyncSession) -> None:
        # retracting a particle removes it from its relations.
        from particles.core.schema import RelationCreatedBy, RelationType
        from particles.operations.retract import retract_entry
        from particles.store.relation_store import create_relation, get_relations_for_particle

        eid, sid = await _seed_entry(db_session)
        a = await _add_particle(db_session, entry_id=eid, snapshot_id=sid, status=Status.ACTIVE)
        b = await _add_particle(db_session, entry_id=eid, snapshot_id=sid, status=Status.ACTIVE)
        await create_relation(
            db_session, a, b, RelationType.CO_EVIDENTIAL, RelationCreatedBy.MANUAL_CLI
        )
        await db_session.commit()

        await retract_entry(db_session, eid)
        await db_session.commit()

        assert await get_relations_for_particle(db_session, a) == []


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestRetractCLI:
    def test_dry_run_writes_nothing(self, runner: CliRunner, cli_db: Path) -> None:
        from particles.db import session_scope

        async def seed() -> str:
            async with session_scope() as session:
                eid, sid = await _seed_entry(session)
                await _add_particle(session, entry_id=eid, snapshot_id=sid, status=Status.ACTIVE)
                await session.commit()
                return eid

        eid = _run(seed())
        result = runner.invoke(app, ["corpus", "retract", eid, "--dry-run"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Dry run" in result.stdout

        async def check() -> Status:
            from particles.store.particle_store import get_particles_for_entry

            async with session_scope() as session:
                ps = await get_particles_for_entry(session, eid)
                return ps[0].status

        assert _run(check()) == Status.ACTIVE  # unchanged

    def test_retract_applies_with_yes(self, runner: CliRunner, cli_db: Path) -> None:
        from particles.db import session_scope

        async def seed() -> str:
            async with session_scope() as session:
                eid, sid = await _seed_entry(session)
                await _add_particle(session, entry_id=eid, snapshot_id=sid, status=Status.ACTIVE)
                await session.commit()
                return eid

        eid = _run(seed())
        result = runner.invoke(
            app, ["corpus", "retract", eid, "--yes", "--reason", "pulled"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "Retracted 1 particle" in result.stdout

        async def check() -> Status:
            from particles.store.particle_store import get_particles_for_entry

            async with session_scope() as session:
                ps = await get_particles_for_entry(session, eid)
                return ps[0].status

        assert _run(check()) == Status.RETRACTED

    def test_unknown_entry_exits_1(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["corpus", "retract", "nope12345"])
        assert result.exit_code == 1
