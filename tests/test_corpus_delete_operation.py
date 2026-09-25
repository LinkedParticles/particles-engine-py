# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the corpus delete operation (D1).

``particles corpus delete`` is the one exception to append-only provenance: a
particle the entry solely supports is deleted, a particle other entries also
support is stripped of the entry's ``SOURCE`` refs, and ``PARTICLE`` refs are
left alone. Covers the pure decide function, the apply path against a DB, the
``CORPUS_ENTRY_DELETED`` event (which must hold no deleted content), and the
CLI preview.
"""

from __future__ import annotations

import asyncio
import json
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
    ExtractionStatus,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    Subject,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.corpus_delete import (
    ProvenanceStrip,
    decide_entry_deletion,
    delete_entry,
    plan_entry_deletion,
)
from particles.store.event_store import EventRefKind, OperatorEventType, list_events
from particles.store.particle_store import (
    ParticleRow,
    ProvenanceEdgeRow,
    append_provenance_ref,
    get_particle,
    strip_entry_provenance_refs,
)

_SECRET_URI = "file:///home/op/private/medical-notes.md"
_SECRET_CLAIM = "The operator was diagnosed with a private condition."
_SECRET_HASH = "c" * 64


def _src(entry: str, *, loc: str | None = None, chunk: str | None = None) -> ProvenanceRef:
    return ProvenanceRef(
        type=ProvenanceRefType.SOURCE, corpus_entry_id=entry, location=loc, chunk_hash=chunk
    )


def _prem(particle_id: str) -> ProvenanceRef:
    return ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=particle_id)


# --- decide (pure) ------------------------------------------------------------


class TestDecideEntryDeletion:
    def test_sole_source_is_deleted(self) -> None:
        d = decide_entry_deletion("E", [("p1", [_src("E")])])
        assert d.to_delete == ["p1"]
        assert d.to_strip == []

    def test_two_refs_to_the_same_entry_is_still_sole_source(self) -> None:
        d = decide_entry_deletion("E", [("p1", [_src("E", loc="a"), _src("E", loc="b")])])
        assert d.to_delete == ["p1"]

    def test_multi_source_is_stripped_not_deleted(self) -> None:
        d = decide_entry_deletion("E", [("p1", [_src("F"), _src("E")])])
        assert d.to_delete == []
        assert d.to_strip == [ProvenanceStrip(particle_id="p1", removed_refs=1, anchor_moved=False)]

    def test_stripping_provenance_zero_moves_the_anchor(self) -> None:
        d = decide_entry_deletion("E", [("p1", [_src("E"), _src("F"), _src("E", loc="x")])])
        assert d.to_strip == [ProvenanceStrip(particle_id="p1", removed_refs=2, anchor_moved=True)]

    def test_particle_refs_neither_count_as_source_nor_are_stripped(self) -> None:
        # A premise ref is not a second source: E is still the sole source.
        d = decide_entry_deletion("E", [("p1", [_src("E"), _prem("F")])])
        assert d.to_delete == ["p1"]
        # A premise ref whose id happens to equal the entry id is not stripped.
        d = decide_entry_deletion("E", [("p2", [_src("F"), _prem("E")])])
        assert d.to_delete == [] and d.to_strip == []

    def test_mixed_batch(self) -> None:
        d = decide_entry_deletion(
            "E",
            [
                ("only", [_src("E")]),
                ("shared", [_src("F"), _src("E")]),
                ("elsewhere", [_src("F")]),
            ],
        )
        assert d.to_delete == ["only"]
        assert [s.particle_id for s in d.to_strip] == ["shared"]


# --- apply (DB) ---------------------------------------------------------------


async def _entry(session: AsyncSession, uri: str) -> tuple[str, str]:
    from particles.corpus.store import CorpusEntryRow, SnapshotRow

    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()), source_type="WEB_PAGE", uri_r=uri, deposited_by="test"
    )
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash="a" * 64,
        extraction_status=ExtractionStatus.COMPLETE,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    session.add(CorpusEntryRow.from_model(entry))
    session.add(SnapshotRow.from_model(snap, entry.entry_id))
    await session.flush()
    return entry.entry_id, snap.snapshot_id


async def _particle(
    session: AsyncSession,
    refs: list[ProvenanceRef],
    *,
    content: str = "A claim.",
    subject_ids: list[str] | None = None,
) -> str:
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="extractor",
        asserted_at=datetime(2026, 1, 2, tzinfo=UTC),
        status=Status.ACTIVE,
        provenance=refs,
        subject_ids=subject_ids or [],
    )
    await insert_particle(session, p)
    return p.id


async def _subject(session: AsyncSession, name: str) -> str:
    from particles.store.subject_store import insert_subject

    subj = Subject(id=str(uuid.uuid4()), canonical_name=name, asserted_by="test")
    await insert_subject(session, subj)
    return subj.id


class _Seeded:
    def __init__(self) -> None:
        self.entry = ""
        self.other = ""
        self.sole = ""
        self.shared = ""
        self.derived = ""
        self.subject_only = ""


async def _seed(session: AsyncSession) -> _Seeded:
    s = _Seeded()
    s.entry, snap = await _entry(session, _SECRET_URI)
    s.other, snap_o = await _entry(session, "https://example.com/other")
    s.subject_only = await _subject(session, "Only in the deleted entry")
    s.sole = await _particle(
        session,
        [
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=s.entry,
                snapshot_id=snap,
                location="p3",
                chunk_hash=_SECRET_HASH,
            )
        ],
        content=_SECRET_CLAIM,
        subject_ids=[s.subject_only],
    )
    # The deleted entry is provenance[0], so stripping it moves the anchor.
    s.shared = await _particle(
        session,
        [
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=s.entry,
                snapshot_id=snap,
                chunk_hash=_SECRET_HASH,
            ),
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=s.other, snapshot_id=snap_o
            ),
        ],
    )
    # A later observation from the other entry, appended as carry-forward does.
    await append_provenance_ref(
        session,
        s.shared,
        ProvenanceRef(
            type=ProvenanceRefType.SOURCE,
            corpus_entry_id=s.other,
            snapshot_id=snap_o,
            location="later",
        ),
    )
    # A derived claim over the doomed particle: its premise ref is left alone.
    s.derived = await _particle(
        session,
        [
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=s.other),
            _prem(s.sole),
        ],
    )
    await session.commit()
    return s


class TestDeleteEntry:
    async def test_plan_shows_the_split(self, db_session: AsyncSession) -> None:
        s = await _seed(db_session)
        plan = await plan_entry_deletion(db_session, s.entry)
        assert plan.snapshots == 1
        assert plan.decision.to_delete == [s.sole]
        assert plan.decision.to_strip == [
            ProvenanceStrip(particle_id=s.shared, removed_refs=1, anchor_moved=True)
        ]

    async def test_apply_deletes_sole_strips_shared(self, db_session: AsyncSession) -> None:
        from particles.corpus.store import CorpusEntryRow, SnapshotRow
        from particles.store.subject_store import SubjectRow

        s = await _seed(db_session)
        before = await get_particle(db_session, s.shared)
        assert before is not None

        result = await delete_entry(db_session, s.entry)
        await db_session.commit()

        assert result.particles_deleted == 1
        assert result.particles_stripped == 1
        assert result.anchors_moved == 1
        assert result.snapshots_deleted == 1
        assert result.subjects_orphaned == 1

        assert await db_session.get(ParticleRow, s.sole) is None
        assert await db_session.get(CorpusEntryRow, s.entry) is None
        snaps = (
            await db_session.execute(select(SnapshotRow).where(SnapshotRow.entry_id == s.entry))
        ).all()
        assert snaps == []
        assert await db_session.get(SubjectRow, s.subject_only) is None

        after = await get_particle(db_session, s.shared)
        assert after is not None
        # The deleted entry's ref is gone; the rest keep their order, so the
        # next-earliest ref is now provenance[0].
        assert [r.corpus_entry_id for r in after.provenance] == [s.other, s.other]
        assert [r.location for r in after.provenance] == [None, "later"]
        assert all(r.chunk_hash != _SECRET_HASH for r in after.provenance)
        # The assertion record is untouched.
        assert after.confidence.value == before.confidence.value
        assert after.asserted_at == before.asserted_at
        assert after.asserted_by == before.asserted_by
        assert after.content == before.content

        edges = (
            await db_session.execute(
                select(ProvenanceEdgeRow).where(ProvenanceEdgeRow.corpus_entry_id == s.entry)
            )
        ).all()
        assert edges == []

        # The derived claim keeps its dangling premise ref for the
        # revalidation ladder to find.
        derived = await get_particle(db_session, s.derived)
        assert derived is not None
        assert [(r.type, r.corpus_entry_id) for r in derived.provenance] == [
            (ProvenanceRefType.SOURCE, s.other),
            (ProvenanceRefType.PARTICLE, s.sole),
        ]

    async def test_event_records_counts_and_no_content(self, db_session: AsyncSession) -> None:
        s = await _seed(db_session)
        await delete_entry(db_session, s.entry)
        await db_session.commit()

        events = await list_events(db_session, event_type=OperatorEventType.CORPUS_ENTRY_DELETED)
        assert len(events) == 1
        ev = events[0]
        assert ev.actor == "cli:corpus-delete"
        assert ev.payload == {
            "particles_deleted": 1,
            "particles_stripped": 1,
            "anchors_moved": 1,
            "snapshots_deleted": 1,
            "subjects_orphaned": 1,
            "synthesis_rows_deleted": 0,
        }
        assert {(r.ref_kind, r.ref_id) for r in ev.refs} == {
            (EventRefKind.CORPUS_ENTRY, s.entry),
            (EventRefKind.PARTICLE, s.sole),
            (EventRefKind.PARTICLE, s.shared),
        }
        dumped = ev.model_dump_json()
        for secret in (_SECRET_CLAIM, _SECRET_HASH, "medical-notes", "p3"):
            assert secret not in dumped
        assert ev.reason is None

    async def test_entry_with_no_particles_still_records_event(
        self, db_session: AsyncSession
    ) -> None:
        entry, _ = await _entry(db_session, "https://example.com/empty")
        await db_session.commit()
        result = await delete_entry(db_session, entry)
        await db_session.commit()
        assert result.particles_deleted == 0
        events = await list_events(db_session, event_type=OperatorEventType.CORPUS_ENTRY_DELETED)
        assert len(events) == 1

    async def test_strip_refuses_to_remove_the_last_source(self, db_session: AsyncSession) -> None:
        s = await _seed(db_session)
        with pytest.raises(ValueError, match="deleted, not stripped"):
            await strip_entry_provenance_refs(db_session, s.sole, s.entry)
        row = await db_session.get(ParticleRow, s.sole)
        assert row is not None
        assert json.loads(row.provenance_json)[0]["corpus_entry_id"] == s.entry


# --- CLI preview --------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


async def _seed_cli() -> _Seeded:
    from particles.db import session_scope

    async with session_scope() as session:
        return await _seed(session)


class TestCorpusDeleteCli:
    def test_preview_shows_split_and_abort_writes_nothing(self, cli_db: Path) -> None:
        s = _run_async(_seed_cli())
        result = CliRunner().invoke(app, ["corpus", "delete", s.entry[:8]], input="n\n")
        assert result.exit_code == 1
        assert "1 will be deleted" in result.output
        assert "1 kept" in result.output
        assert "1 with a new earliest source" in result.output

        async def _still_there() -> bool:
            from particles.db import session_scope

            async with session_scope() as session:
                return await session.get(ParticleRow, s.sole) is not None

        assert _run_async(_still_there())

    def test_yes_applies(self, cli_db: Path) -> None:
        s = _run_async(_seed_cli())
        result = CliRunner().invoke(app, ["corpus", "delete", s.entry[:8], "--yes"])
        assert result.exit_code == 0, result.output
        assert "1 particles removed" in result.output
        assert "1 kept with refs stripped" in result.output
