# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Operator-scoped belief mutation + subject-assign.

Covers the operator path of ``particles.operations.agent_write``:

* ``operator=True`` supersede / retract may target a belief the agent does NOT
  own (incl. extracted beliefs) — the curation-queue case — while still
  rejecting HUMAN_REVIEW targets and recording the act under the operator actor;
* ``assign_subject_belief`` (in place since):
  resolve-by-id and resolve-by-name, and the in-place invariant — the belief
  keeps its id, confidence record, extractor_ref and provenance, and only its
  subject link changes; an already-linked belief is refused.

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
    ParticleType,
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

    async def _fake_resolve(session: Any, *_a: Any, **_k: Any) -> Subject:
        # The in-place assign links a Subject row that exists.
        from particles.store.subject_store import get_subject

        if await get_subject(session, resolved.id) is None:
            await insert_subject(session, resolved)
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
    particle_type: ParticleType = ParticleType.CLAIM,
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
        particle_type=particle_type,
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
    async def test_operator_may_retract_an_operator_claim(self, db_session: Any) -> None:
        """the HUMAN_REVIEW guard is agent policy; the operator is exempt."""
        _enable_writes(allow_cross_asserter=True)
        target = await _insert_extracted(
            db_session, asserted_by="operator:local", calib=CalibrationSource.HUMAN_REVIEW
        )
        # An agent is refused even with the cross-asserter grant: the guard fires first.
        with pytest.raises(ValueError, match="HUMAN_REVIEW"):
            await retract_belief(db_session, store=DEFAULT_STORE, particle_id=target.id, reason="x")
        await retract_belief(
            db_session, store=DEFAULT_STORE, particle_id=target.id, reason="mine", operator=True
        )
        p = await get_particle(db_session, target.id)
        assert p is not None and p.status is Status.RETRACTED

    @pytest.mark.asyncio
    async def test_operator_still_rejects_a_human_review_non_claim(self, db_session: Any) -> None:
        _enable_writes()
        target = await _insert_extracted(
            db_session,
            asserted_by="cli-user",
            calib=CalibrationSource.HUMAN_REVIEW,
            particle_type=ParticleType.REVIEW,
        )
        with pytest.raises(ValueError, match="REVIEW record"):
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
            reason="the source misread the figure",
        )
        assert result.verdict == "ASSERTED"
        old = await get_particle(db_session, target.id)
        new = await get_particle(db_session, result.asserted_particle_id or "")
        assert old is not None and old.status is Status.SUPERSEDED
        assert new is not None and new.supersedes == target.id
        # the successor is the operator's belief, with the confidence
        # the operator set (NOT carried over, NOT the agent's).
        assert new.asserted_by == "operator:local"
        assert new.confidence.calibration_source is CalibrationSource.HUMAN_REVIEW
        assert new.confidence.value == pytest.approx(0.6)
        # the why of the revision is on the audit event, as for a
        # retraction.
        from particles.store.event_store import OperatorEventType, list_events

        events = await list_events(db_session, event_type=OperatorEventType.PARTICLE_SUPERSEDED)
        assert len(events) == 1
        assert events[0].reason == "the source misread the figure"

    @pytest.mark.asyncio
    async def test_operator_successor_is_attributed_to_the_operator(
        self, db_session: Any, stub_resolver: Any
    ) -> None:
        """principal, excerpt author, no agent ceiling, no trust seeded."""
        from particles.corpus.store import get_entry
        from particles.store.trust_store import get_trust_statements_for_domain

        _enable_writes(max_asserted_confidence=0.9)
        get_config().curation.operator_identity = "operator:jeff"
        target = await _insert_extracted(db_session)
        result = await supersede_belief(
            db_session,
            store=DEFAULT_STORE,
            supersedes_id=target.id,
            content="A claim the operator is sure of.",
            subject_names=["X"],
            confidence=0.97,
            source_excerpt="checked against the release notes",
            operator=True,
            actor="curate",
            reason="verified",
        )
        new = await get_particle(db_session, result.asserted_particle_id or "")
        assert new is not None
        assert new.asserted_by == "operator:jeff"
        assert new.confidence.value == pytest.approx(0.97)  # not clamped to 0.9
        entry = await get_entry(db_session, new.provenance[0].corpus_entry_id or "")
        assert entry is not None and entry.deposited_by == "operator:jeff"
        assert entry.snapshots[-1].author_id == "operator:jeff"
        statements = await get_trust_statements_for_domain(db_session, "agent-memory")
        assert not any(s.source_ref.value == "operator:jeff" for s in statements)
        events = await list_events(db_session, event_type=OperatorEventType.PARTICLE_SUPERSEDED)
        assert events[0].actor == "curate"  # the surface, beside the principal

    @pytest.mark.asyncio
    async def test_operator_can_correct_their_own_correction(
        self, db_session: Any, stub_resolver: Any
    ) -> None:
        _enable_writes(allow_cross_asserter=True)
        target = await _insert_extracted(db_session)
        kwargs: dict[str, Any] = {
            "store": DEFAULT_STORE,
            "subject_names": ["X"],
            "source_excerpt": "e",
            "operator": True,
            "actor": "curate",
            "reason": "r",
        }
        first = await supersede_belief(
            db_session, supersedes_id=target.id, content="First fix.", confidence=0.7, **kwargs
        )
        first_id = first.asserted_particle_id or ""
        # An agent may not revise the operator's belief...
        with pytest.raises(ValueError, match="HUMAN_REVIEW"):
            await supersede_belief(
                db_session,
                store=DEFAULT_STORE,
                supersedes_id=first_id,
                content="Agent rewrite.",
                subject_names=["X"],
                confidence=0.5,
                source_excerpt="e",
            )
        # ...but the operator may.
        second = await supersede_belief(
            db_session, supersedes_id=first_id, content="Second fix.", confidence=0.8, **kwargs
        )
        prior = await get_particle(db_session, first_id)
        assert prior is not None and prior.status is Status.SUPERSEDED
        latest = await get_particle(db_session, second.asserted_particle_id or "")
        assert latest is not None and latest.supersedes == first_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    async def test_operator_confidence_must_be_a_probability(
        self, db_session: Any, stub_resolver: Any, bad: float
    ) -> None:
        _enable_writes()
        target = await _insert_extracted(db_session)
        with pytest.raises(ValueError, match="between 0 and 1"):
            await supersede_belief(
                db_session,
                store=DEFAULT_STORE,
                supersedes_id=target.id,
                content="c",
                subject_names=["X"],
                confidence=bad,
                source_excerpt="e",
                operator=True,
                actor="curate",
                reason="r",
            )

    @pytest.mark.asyncio
    async def test_agent_supersede_is_unchanged(self, db_session: Any, stub_resolver: Any) -> None:
        _enable_writes(max_asserted_confidence=0.9)
        own = await _insert_extracted(db_session, asserted_by="mcp:test-agent")
        result = await supersede_belief(
            db_session,
            store=DEFAULT_STORE,
            supersedes_id=own.id,
            content="c",
            subject_names=["X"],
            confidence=0.97,
            source_excerpt="e",
        )
        new = await get_particle(db_session, result.asserted_particle_id or "")
        assert new is not None
        assert new.asserted_by == "mcp:test-agent"
        assert new.confidence.calibration_source is CalibrationSource.AGENT_ASSERTED
        assert new.confidence.value == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_operator_supersede_requires_a_reason(
        self, db_session: Any, stub_resolver: Any
    ) -> None:
        """a supersession is a judgment; the operator path must say why."""
        _enable_writes()
        target = await _insert_extracted(db_session)
        for empty in (None, "", "   "):
            with pytest.raises(ValueError, match="non-empty reason"):
                await supersede_belief(
                    db_session,
                    store=DEFAULT_STORE,
                    supersedes_id=target.id,
                    content="A corrected claim.",
                    subject_names=["X"],
                    confidence=0.6,
                    source_excerpt="the corrected statement",
                    operator=True,
                    actor="http:/particles/{id}/supersede",
                    reason=empty,
                )
        untouched = await get_particle(db_session, target.id)
        assert untouched is not None and untouched.status is Status.ACTIVE


class TestAssignSubject:
    @pytest.mark.asyncio
    async def test_assign_by_id_links_in_place(self, db_session: Any) -> None:
        """the orphan keeps its id; no successor is minted."""
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
        assert result.asserted_particle_id == target.id
        assert result.verdict == "SUBJECT_ASSIGNED"
        same = await get_particle(db_session, target.id)
        assert same is not None and same.status is Status.ACTIVE
        assert same.subject_ids == [subject.id]
        # Nothing about the claim changed but the link.
        assert same.content == target.content
        assert same.confidence.value == target.confidence.value
        assert same.confidence.calibration_ref == "calib-ref-123"
        assert same.extractor_ref == target.extractor_ref
        assert same.supersedes is None

        relinked = await list_events(db_session, event_type=OperatorEventType.SUBJECTS_RELINKED)
        assert len(relinked) == 1
        assert relinked[0].actor == "http:/particles/{id}/subjects"
        assert {r.ref_id for r in relinked[0].refs} == {target.id, subject.id}
        assert not await list_events(db_session, event_type=OperatorEventType.PARTICLE_SUPERSEDED)

    @pytest.mark.asyncio
    async def test_assign_refuses_an_already_linked_belief(self, db_session: Any) -> None:
        _enable_writes()
        first = Subject(canonical_name="First", asserted_by="operator")
        second = Subject(canonical_name="Second", asserted_by="operator")
        await insert_subject(db_session, first)
        await insert_subject(db_session, second)
        target = await _insert_extracted(db_session)
        await db_session.flush()
        await assign_subject_belief(
            db_session, store=DEFAULT_STORE, particle_id=target.id, subject_id=first.id
        )
        with pytest.raises(ValueError, match="already has a subject"):
            await assign_subject_belief(
                db_session, store=DEFAULT_STORE, particle_id=target.id, subject_id=second.id
            )

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
        assert result.asserted_particle_id == target.id
        same = await get_particle(db_session, target.id)
        assert same is not None
        # The resolver-returned subject id is attached, in place.
        assert stub_resolver in same.subject_ids

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
    async def test_assign_rejects_a_human_review_non_claim(self, db_session: Any) -> None:
        _enable_writes()
        subject = Subject(canonical_name="S", asserted_by="operator")
        await insert_subject(db_session, subject)
        target = await _insert_extracted(
            db_session,
            asserted_by="operator",
            calib=CalibrationSource.HUMAN_REVIEW,
            particle_type=ParticleType.REVIEW,
        )
        await db_session.flush()
        with pytest.raises(ValueError, match="HUMAN_REVIEW"):
            await assign_subject_belief(
                db_session,
                store=DEFAULT_STORE,
                particle_id=target.id,
                subject_id=subject.id,
            )
