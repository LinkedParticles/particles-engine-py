"""Operator-scoped belief mutation + subject-assign.

Covers the operator path of ``particles.operations.agent_write``:

* ``operator=True`` supersede / retract may target a belief the agent does NOT
  own (incl. extracted beliefs) — the curation-queue case — while still
  rejecting HUMAN_REVIEW targets and recording the act under the operator actor;
* ``assign_subject_belief`` (the provenance-preserving supersede):
  resolve-by-id and resolve-by-name, and the carry-over invariant — the
  successor keeps the predecessor's full confidence record + extractor_ref +
  source provenance, only the subject linkage is corrected.

Subject resolution is stubbed so the tests stay offline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.db import DEFAULT_STORE
from particles.operations.agent_write import (
    assign_subject_belief,
    retract_belief,
    supersede_belief,
)
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import get_particle, insert_particle
from particles.store.subject_store import insert_subject


def _enable_writes(**overrides: Any) -> None:
    w = get_config().mcp.write
    w.enabled_stores = [DEFAULT_STORE]
    w.asserter_identity = "mcp:test-agent"
    for key, value in overrides.items():
        setattr(w, key, value)


@pytest.fixture
def stub_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the canonical subject resolver (agent_write defers the import)."""
    import particles.ingest.subject_resolver as sr

    resolved = Subject(canonical_name="Resolved Subject", asserted_by="resolver")

    async def _fake_resolve(*_a: Any, **_k: Any) -> Subject:
        return resolved

    monkeypatch.setattr(sr, "resolve_subject", AsyncMock(side_effect=_fake_resolve))
    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))
    # Stash the resolved subject id on the fixture for assertions.
    monkeypatch.setattr(sr, "_TEST_RESOLVED_SUBJECT_ID", resolved.id, raising=False)
    return resolved.id  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def _stub_embeddings() -> Any:
    """A constant embedding so reconcile_and_insert does not call a real model."""
    import numpy as np

    from particles import embeddings as ep

    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


async def _insert_extracted(
    session: Any,
    *,
    content: str = "An extracted claim.",
    subject_ids: list[str] | None = None,
    calib: CalibrationSource = CalibrationSource.EXTRACTOR_DIRECT,
    asserted_by: str = "general-extractor",
) -> Particle:
    p = Particle(
        content=content,
        confidence=Confidence(
            value=0.73,
            calibration_source=calib,
            calibration_method="temperature_scaling",
            calibration_ref="calib-ref-123",
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=asserted_by,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
        extractor_ref={"name": "general", "version": "1.0"},
        subject_ids=subject_ids or [],
    )
    await insert_particle(session, p)
    await session.flush()
    return p


class TestOperatorRetract:
    @pytest.mark.asyncio
    async def test_operator_can_retract_extracted_belief(self, db_session: Any) -> None:
        _enable_writes(allow_cross_asserter=False)
        target = await _insert_extracted(db_session)

        # The own-beliefs-only path refuses it (asserted by the extractor).
        with pytest.raises(ValueError, match="cross-asserter"):
            await retract_belief(db_session, store=DEFAULT_STORE, particle_id=target.id, reason="x")

        # The operator path retracts it.
        await retract_belief(
            db_session,
            store=DEFAULT_STORE,
            particle_id=target.id,
            reason="spurious",
            operator=True,
            actor="http:/particles/{id}/retract",
        )
        p = await get_particle(db_session, target.id)
        assert p is not None
        assert p.status is Status.RETRACTED
        assert p.status_reason is StatusReason.EXPLICIT_RETRACTION

        events = await list_events(db_session, event_type=OperatorEventType.PARTICLE_RETRACTED)
        assert events and events[0].actor == "http:/particles/{id}/retract"
        assert events[0].payload is not None and events[0].payload.get("operator") is True

    @pytest.mark.asyncio
    async def test_operator_still_rejects_human_review(self, db_session: Any) -> None:
        _enable_writes()
        target = await _insert_extracted(
            db_session, asserted_by="operator", calib=CalibrationSource.HUMAN_REVIEW
        )
        with pytest.raises(ValueError, match="HUMAN_REVIEW"):
            await retract_belief(
                db_session,
                store=DEFAULT_STORE,
                particle_id=target.id,
                reason="nope",
                operator=True,
            )


class TestOperatorSupersede:
    @pytest.mark.asyncio
    async def test_operator_supersede_extracted_belief(
        self, db_session: Any, stub_resolver: Any
    ) -> None:
        _enable_writes()
        target = await _insert_extracted(db_session)
        result = await supersede_belief(
            db_session,
            store=DEFAULT_STORE,
            supersedes_id=target.id,
            content="A corrected claim.",
            subject_names=["X"],
            confidence=0.6,
            source_excerpt="the corrected statement",
            operator=True,
            actor="http:/particles/{id}/supersede",
        )
        assert result.verdict == "ASSERTED"
        old = await get_particle(db_session, target.id)
        new = await get_particle(db_session, result.asserted_particle_id or "")
        assert old is not None and old.status is Status.SUPERSEDED
        assert new is not None and new.supersedes == target.id
        # Standard operator supersede is operator-attributed with the confidence
        # the operator set (NOT carried over).
        assert new.asserted_by == "mcp:test-agent"
        assert new.confidence.calibration_source is CalibrationSource.AGENT_ASSERTED


class TestAssignSubject:
    @pytest.mark.asyncio
    async def test_assign_by_id_carries_over_provenance(self, db_session: Any) -> None:
        _enable_writes()
        subject = Subject(canonical_name="Picked Subject", asserted_by="operator")
        await insert_subject(db_session, subject)
        target = await _insert_extracted(db_session)
        await db_session.flush()

        result = await assign_subject_belief(
            db_session,
            store=DEFAULT_STORE,
            particle_id=target.id,
            subject_id=subject.id,
            actor="http:/particles/{id}/subjects",
        )
        old = await get_particle(db_session, target.id)
        new = await get_particle(db_session, result.asserted_particle_id or "")
        assert old is not None and old.status is Status.SUPERSEDED
        assert new is not None
        assert new.supersedes == target.id
        assert subject.id in new.subject_ids
        # Provenance carry-over: same content, same confidence record,
        # same extractor_ref, same source — only the subject linkage changed.
        assert new.content == target.content
        assert new.confidence.value == target.confidence.value
        assert new.confidence.calibration_source is CalibrationSource.EXTRACTOR_DIRECT
        assert new.confidence.calibration_method == "temperature_scaling"
        assert new.confidence.calibration_ref == "calib-ref-123"
        assert new.extractor_ref == target.extractor_ref
        assert new.asserted_by == target.asserted_by  # the extractor, not the operator
        assert [pr.corpus_entry_id for pr in new.provenance] == ["entry-1"]

        events = await list_events(db_session, event_type=OperatorEventType.PARTICLE_SUPERSEDED)
        assert events and events[0].payload is not None
        assert events[0].payload.get("subject_assign") == subject.id

    @pytest.mark.asyncio
    async def test_assign_by_name_uses_resolver(self, db_session: Any, stub_resolver: Any) -> None:
        _enable_writes()
        target = await _insert_extracted(db_session)
        result = await assign_subject_belief(
            db_session,
            store=DEFAULT_STORE,
            particle_id=target.id,
            subject_name="Some Entity",
        )
        new = await get_particle(db_session, result.asserted_particle_id or "")
        assert new is not None
        # The resolver-returned subject id is attached.
        assert stub_resolver in new.subject_ids

    @pytest.mark.asyncio
    async def test_assign_requires_exactly_one_of_id_or_name(self, db_session: Any) -> None:
        _enable_writes()
        target = await _insert_extracted(db_session)
        with pytest.raises(ValueError, match="exactly one"):
            await assign_subject_belief(db_session, store=DEFAULT_STORE, particle_id=target.id)
        with pytest.raises(ValueError, match="exactly one"):
            await assign_subject_belief(
                db_session,
                store=DEFAULT_STORE,
                particle_id=target.id,
                subject_id="a",
                subject_name="b",
            )

    @pytest.mark.asyncio
    async def test_assign_rejects_human_review(self, db_session: Any) -> None:
        _enable_writes()
        subject = Subject(canonical_name="S", asserted_by="operator")
        await insert_subject(db_session, subject)
        target = await _insert_extracted(
            db_session, asserted_by="operator", calib=CalibrationSource.HUMAN_REVIEW
        )
        await db_session.flush()
        with pytest.raises(ValueError, match="HUMAN_REVIEW"):
            await assign_subject_belief(
                db_session,
                store=DEFAULT_STORE,
                particle_id=target.id,
                subject_id=subject.id,
            )
