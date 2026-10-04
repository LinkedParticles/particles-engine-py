# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The component record on a snapshot: written with COMPLETE, read back whole.

Covers the store half (``update_extraction_status`` writes the record only
with ``COMPLETE``; ``get_extraction_component_records`` reads it back, and an
unreadable one as none) and the pipeline half (``extract_snapshot`` stamps
what the extraction exercised, and folds in the record of any snapshot a
carried-forward claim came from). The LLM is mocked at the completion port so
the real request builder runs and records the prompt it assembled.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from particles import embeddings as ep
from particles.core.schema import (
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    ExtractorRef,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.corpus.deposit import deposit_text_versioned
from particles.corpus.store import (
    CorpusEntryRow,
    SnapshotRow,
    get_extraction_component_records,
    update_extraction_status,
)
from particles.extraction import general
from particles.extraction.components import ComponentRecord, ComponentTally
from particles.extraction.subject_gate import GATE_COMPONENT
from particles.ingest.pipeline import _component_record, extract_snapshot
from particles.store.particle_store import insert_particle


def _vector(text: str) -> list[float]:
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest() * 4, dtype=np.uint8)
    vec = raw.astype(np.float32) - 127.5
    return list(vec / np.linalg.norm(vec))


@pytest.fixture
def embeddings() -> Iterator[None]:
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, **_: np.array([_vector(t) for t in texts], dtype=np.float32)
    )
    original = ep._embedding_model
    ep.set_embedding_model(model)
    yield
    ep.set_embedding_model(original)


async def _snapshot(session: AsyncSession) -> tuple[CorpusEntry, Snapshot]:
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="WEB_PAGE",
        uri_r="https://example.com/stamp",
        deposited_by="test",
    )
    session.add(CorpusEntryRow.from_model(entry))
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash="0" * 64,
        extraction_status=ExtractionStatus.IN_PROGRESS,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    session.add(SnapshotRow.from_model(snap, entry.entry_id))
    await session.flush()
    return entry, snap


class TestStoreRoundTrip:
    @pytest.mark.asyncio
    async def test_a_complete_write_round_trips(self, db_session: AsyncSession) -> None:
        _, snap = await _snapshot(db_session)
        record = ComponentRecord(
            extractor="general-extractor",
            exercised={"prompt.general.rules": "abc", "path.single_pass": "def"},
            available=["path.chunked", "path.single_pass", "prompt.general.rules"],
        )
        await update_extraction_status(
            db_session, snap.snapshot_id, ExtractionStatus.COMPLETE, components=record
        )
        assert await get_extraction_component_records(db_session, [snap.snapshot_id]) == {
            snap.snapshot_id: record
        }

    @pytest.mark.asyncio
    async def test_failed_does_not_write_the_record(self, db_session: AsyncSession) -> None:
        _, snap = await _snapshot(db_session)
        record = ComponentRecord(exercised={"prompt.a": "1"})
        await update_extraction_status(
            db_session, snap.snapshot_id, ExtractionStatus.FAILED, components=record
        )
        assert await get_extraction_component_records(db_session, [snap.snapshot_id]) == {
            snap.snapshot_id: None
        }

    @pytest.mark.asyncio
    async def test_a_partial_read_writes_the_record_with_pending(
        self, db_session: AsyncSession
    ) -> None:
        """A partly failed read stores its attempt's record with PENDING."""
        _, snap = await _snapshot(db_session)
        record = ComponentRecord(exercised={"prompt.a": "1"})
        await update_extraction_status(
            db_session, snap.snapshot_id, ExtractionStatus.PENDING, components=record
        )
        assert await get_extraction_component_records(db_session, [snap.snapshot_id]) == {
            snap.snapshot_id: record
        }

    @pytest.mark.asyncio
    async def test_an_unreadable_record_reads_as_none(self, db_session: AsyncSession) -> None:
        _, snap = await _snapshot(db_session)
        await db_session.execute(
            update(SnapshotRow)
            .where(SnapshotRow.snapshot_id == snap.snapshot_id)
            .values(extraction_components_json="{not json")
        )
        assert await get_extraction_component_records(db_session, [snap.snapshot_id]) == {
            snap.snapshot_id: None
        }
        assert await get_extraction_component_records(db_session, []) == {}


class TestPipelineStamp:
    @pytest.mark.asyncio
    async def test_an_extraction_stamps_what_it_exercised(
        self,
        db_session: AsyncSession,
        embeddings: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reply = json.dumps(
            [
                {
                    "content": "The bridge opened in 1932.",
                    "subjects": ["Sydney Harbour Bridge"],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                }
            ]
        )

        async def fake_complete(*args: Any, **kwargs: Any) -> tuple[str, str]:
            return reply, "anthropic:test-model"

        monkeypatch.setattr("particles.llm.complete_with_provider_model", fake_complete)
        entry_id, snapshot_id, _ = await deposit_text_versioned(
            db_session,
            text="The Sydney Harbour Bridge opened in 1932.",
            uri_r="https://example.com/bridge",
            source_type="WEB_PAGE",
            mutability=Mutability.MUTABLE,
            deposited_by="test",
        )
        await db_session.commit()

        await extract_snapshot(db_session, entry_id, snapshot_id)
        await db_session.commit()

        record = (await get_extraction_component_records(db_session, [snapshot_id]))[snapshot_id]
        assert record is not None
        assert record.extractor == general.EXTRACTOR_ID
        assert record.complete
        expected_prompt = {c.name: c.digest for c in general._configured_prompt_components()}
        assert expected_prompt.items() <= record.exercised.items()
        assert general.PATH_SINGLE_PASS in record.exercised
        assert GATE_COMPONENT in record.exercised
        assert general.CODE_PARSE in record.exercised
        assert general.ROUTING in record.exercised
        # Everything the extractor could have exercised is declared available.
        assert set(general.general_component_table().digests) <= set(record.available)


async def _particle(session: AsyncSession, entry_id: str, snapshot_ids: list[str]) -> Particle:
    particle = Particle(
        id=str(uuid.uuid4()),
        content="A carried claim.",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=sid)
            for sid in snapshot_ids
        ],
        extractor_ref=ExtractorRef(name="general-extractor", version="0.16.0"),
    )
    await insert_particle(session, particle)
    return particle


class TestCarryForwardMerge:
    @pytest.mark.asyncio
    async def test_a_carried_claim_brings_its_first_extractions_components(
        self, db_session: AsyncSession
    ) -> None:
        entry, first = await _snapshot(db_session)
        second = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            captured_at=datetime.now(UTC),
            content_hash="1" * 64,
            extraction_status=ExtractionStatus.IN_PROGRESS,
            warc_record_type=WarcRecordType.RESPONSE,
        )
        db_session.add(SnapshotRow.from_model(second, entry.entry_id))
        await update_extraction_status(
            db_session,
            first.snapshot_id,
            ExtractionStatus.COMPLETE,
            components=ComponentRecord(
                extractor="general-extractor",
                exercised={"prompt.modality": "old", "path.chunked": "p"},
                available=["path.chunked", "prompt.modality"],
            ),
        )
        # The reobservation ref to ``second`` is excluded by the
        # merge either way; the earlier snapshot is what it reads.
        carried = await _particle(db_session, entry.entry_id, [first.snapshot_id])
        tally = ComponentTally()
        tally.add("prompt.modality", "new")
        tally.add("path.chunked", "p")

        record = await _component_record(
            db_session,
            tally,
            extractor_id="general-extractor",
            entry_id=entry.entry_id,
            snapshot_id=second.snapshot_id,
            carry_forward_ids=[carried.id],
        )
        assert record.complete
        assert record.exercised == {"path.chunked": "p", "prompt.modality": "mixed"}

    @pytest.mark.asyncio
    async def test_a_carried_claim_from_an_unrecorded_snapshot_makes_it_incomplete(
        self, db_session: AsyncSession
    ) -> None:
        entry, first = await _snapshot(db_session)
        carried = await _particle(db_session, entry.entry_id, [first.snapshot_id])
        record = await _component_record(
            db_session,
            ComponentTally(),
            extractor_id="general-extractor",
            entry_id=entry.entry_id,
            snapshot_id="current",
            carry_forward_ids=[carried.id],
        )
        assert not record.complete

    @pytest.mark.asyncio
    async def test_no_carry_forward_is_the_tally_alone(self, db_session: AsyncSession) -> None:
        tally = ComponentTally()
        tally.add("prompt.a", "1")
        record = await _component_record(
            db_session,
            tally,
            extractor_id="x",
            entry_id="e",
            snapshot_id="s",
            carry_forward_ids=[],
        )
        assert record == tally.record("x")
