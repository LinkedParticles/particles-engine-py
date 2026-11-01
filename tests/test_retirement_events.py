# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Every structural retiring path names the claims it retired in the event log.

``update_particle_status`` writes no event itself, so a lint fix, the snapshot
generation cascade and reindex each record one ``CLAIMS_RETIRED`` event per
pass, naming the retired claims, the actor and the reason (security review
2026-10-28-1, O36). No model call.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from particles.core.schema import ProvenanceRef, ProvenanceRefType
from particles.core.status import Status, StatusReason
from particles.store.event_store import (
    EventRefKind,
    OperatorEvent,
    OperatorEventType,
    list_events,
)
from particles.store.particle_store import get_particle, insert_particle


async def _retirement_events(session: Any) -> list[OperatorEvent]:
    return await list_events(session, event_type=OperatorEventType.CLAIMS_RETIRED, limit=500)


def _names(event: OperatorEvent, particle_id: str) -> bool:
    payload = event.payload or {}
    refs = {(r.ref_kind, r.ref_id) for r in event.refs}
    return particle_id in payload.get("retired_ids", []) and (
        (EventRefKind.PARTICLE, particle_id) in refs
    )


@pytest.mark.asyncio
async def test_a_lint_fix_that_retires_an_expired_claim_records_an_event(db_session: Any) -> None:
    """A lint fix that moves an expired claim out of ACTIVE leaves one event naming that claim."""
    from particles.operations.lint.staleness import _check_staleness
    from tests.test_reconcile import _particle

    p = _particle(content="The office opens at nine.", entry_id="e1", snapshot_id="s1")
    p = p.model_copy(update={"valid_until": datetime.now(UTC) - timedelta(days=1)})
    await insert_particle(db_session, p, embedding=[1.0, 0.0, 0.0])
    await db_session.commit()

    await _check_staleness(db_session, fix=True)
    await db_session.commit()

    stored = await get_particle(db_session, p.id)
    assert stored is not None and stored.status is Status.PROVENANCE_STALE
    [event] = await _retirement_events(db_session)
    assert _names(event, p.id)
    assert event.actor == "lint"
    assert event.reason
    assert event.payload is not None
    assert event.payload["route"] == "lint_staleness"
    assert event.payload["status"] == Status.PROVENANCE_STALE.value
    assert event.payload["status_reason"] == StatusReason.VALIDITY_EXPIRED.value


@pytest.mark.asyncio
async def test_a_lint_report_without_fix_records_no_retirement_event(db_session: Any) -> None:
    """A report-only lint retires nothing and writes no retirement event."""
    from particles.operations.lint.staleness import _check_staleness
    from tests.test_reconcile import _particle

    p = _particle(content="The shop closes at five.", entry_id="e1", snapshot_id="s1")
    p = p.model_copy(update={"valid_until": datetime.now(UTC) - timedelta(days=1)})
    await insert_particle(db_session, p, embedding=[1.0, 0.0, 0.0])
    await db_session.commit()

    findings = await _check_staleness(db_session, fix=False)
    await db_session.commit()

    assert [f.particle_id for f in findings] == [p.id]
    assert await _retirement_events(db_session) == []


@pytest.mark.asyncio
async def test_a_lint_fix_names_each_claim_its_missing_snapshot_retired(db_session: Any) -> None:
    """The corpus-link fix names every claim it retires in one event, with the run's actor."""
    from particles.operations.lint.staleness import _check_corpus_link_integrity
    from tests.test_reconcile import _particle

    a = _particle(content="The bridge is 40 m long.", entry_id="e1", snapshot_id="gone-1")
    b = _particle(content="The bridge opened in 1901.", entry_id="e1", snapshot_id="gone-2")
    await insert_particle(db_session, a, embedding=[1.0, 0.0, 0.0])
    await insert_particle(db_session, b, embedding=[0.0, 1.0, 0.0])
    await db_session.commit()

    await _check_corpus_link_integrity(db_session, fix=True, actor="reindex")
    await db_session.commit()

    [event] = await _retirement_events(db_session)
    assert _names(event, a.id) and _names(event, b.id)
    assert event.actor == "reindex"
    assert event.payload is not None
    assert event.payload["route"] == "lint_corpus_link"
    assert event.payload["status_reason"] == StatusReason.CORPUS_ENTRY_MISSING.value


@pytest.mark.asyncio
async def test_a_lint_fix_names_the_claim_a_retracted_premise_retired(db_session: Any) -> None:
    """The retraction-cascade fix names the dependent claim it retires."""
    from particles.operations.lint.staleness import _check_retraction_propagation
    from particles.store.particle_store import update_particle_status
    from tests.test_reconcile import _particle

    premise = _particle(content="The river floods every spring.", entry_id="e1", snapshot_id="s1")
    await insert_particle(db_session, premise, embedding=[1.0, 0.0, 0.0])
    dependent = _particle(content="The valley farms move uphill.", entry_id="e1", snapshot_id="s1")
    dependent = dependent.model_copy(
        update={
            "provenance": [
                *dependent.provenance,
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE,
                    corpus_entry_id=premise.id,
                    snapshot_id=premise.id,
                ),
            ]
        }
    )
    await insert_particle(db_session, dependent, embedding=[0.0, 1.0, 0.0])
    await update_particle_status(db_session, premise.id, Status.RETRACTED)
    await db_session.commit()

    await _check_retraction_propagation(db_session, fix=True)
    await db_session.commit()

    stored = await get_particle(db_session, dependent.id)
    assert stored is not None and stored.status is Status.PROVENANCE_STALE
    [event] = await _retirement_events(db_session)
    assert _names(event, dependent.id)
    assert event.payload is not None
    assert event.payload["route"] == "lint_retraction_cascade"


@pytest.mark.asyncio
async def test_the_generation_cascade_names_each_claim_it_retires(db_session: Any) -> None:
    """A new generation of a MUTABLE entry retires the old one's claims and names them."""
    from particles.ingest.generation import cascade_superseded_generation
    from tests.test_generation_cascade import _entry, _particle, _snapshot

    entry = await _entry(db_session)
    old = await _snapshot(db_session, entry, age_days=8)
    new = await _snapshot(db_session, entry)
    stale = await _particle(db_session, entry, old, "The tool is found on PATH.")
    current = await _particle(db_session, entry, new, "The tool is installed per project.")
    await db_session.commit()

    demoted = await cascade_superseded_generation(
        db_session, entry_id=entry.entry_id, current_snapshot_id=new.snapshot_id
    )
    await db_session.commit()

    assert demoted == [stale.id]
    [event] = await _retirement_events(db_session)
    assert _names(event, stale.id)
    assert not _names(event, current.id)
    assert event.actor == "extract"
    assert event.payload is not None
    assert event.payload["route"] == "generation_cascade"
    assert event.payload["entry_id"] == entry.entry_id
    assert event.payload["snapshot_id"] == new.snapshot_id
    assert (EventRefKind.CORPUS_ENTRY, entry.entry_id) in {
        (r.ref_kind, r.ref_id) for r in event.refs
    }


@pytest.mark.asyncio
async def test_a_generation_cascade_that_retires_nothing_records_no_event(
    db_session: Any,
) -> None:
    """A cascade with nothing to retire leaves the event log unchanged."""
    from particles.ingest.generation import cascade_superseded_generation
    from tests.test_generation_cascade import _entry, _particle, _snapshot

    entry = await _entry(db_session)
    snap = await _snapshot(db_session, entry)
    await _particle(db_session, entry, snap, "The tool is installed per project.")
    await db_session.commit()

    assert (
        await cascade_superseded_generation(
            db_session, entry_id=entry.entry_id, current_snapshot_id=snap.snapshot_id
        )
        == []
    )
    assert await _retirement_events(db_session) == []


@pytest.mark.asyncio
async def test_reindex_names_each_claim_a_re_extraction_retires(
    db_session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reindexed snapshot's replaced claims are named in one event, actor ``reindex``."""
    from particles.operations.reindex import _reindex_snapshot
    from tests.test_reindex import _add_entry, _add_particle, _add_snapshot, _patch_extract

    entry = await _add_entry(db_session)
    snap_this = await _add_snapshot(db_session, entry, content_hash="this")
    snap_other = await _add_snapshot(db_session, entry, content_hash="other")
    p_this = await _add_particle(db_session, entry, snap_this, content="from this snap")
    p_other = await _add_particle(db_session, entry, snap_other, content="from other snap")
    await db_session.commit()

    _patch_extract(monkeypatch, return_value=[])
    await _reindex_snapshot(db_session, entry.entry_id, snap_this.snapshot_id, None)

    after = await get_particle(db_session, p_this.id)
    assert after is not None and after.status is Status.SUPERSEDED
    [event] = await _retirement_events(db_session)
    assert _names(event, p_this.id)
    assert not _names(event, p_other.id)
    assert event.actor == "reindex"
    assert event.payload is not None
    assert event.payload["route"] == "reindex"
    assert event.payload["status_reason"] == StatusReason.SUPERSEDED_BY_REINDEX.value
    assert event.payload["snapshot_id"] == snap_this.snapshot_id


@pytest.mark.asyncio
async def test_reindex_post_lint_fix_names_reindex_as_the_actor(db_session: Any) -> None:
    """The lint fix reindex runs afterwards records its retirements under actor ``reindex``."""
    from particles.operations.lint import run_lint
    from particles.operations.reindex import REINDEX_ACTOR
    from tests.test_reconcile import _particle

    p = _particle(content="The office opens at nine.", entry_id="e1", snapshot_id="s1")
    p = p.model_copy(update={"valid_until": datetime.now(UTC) - timedelta(days=1)})
    await insert_particle(db_session, p, embedding=[1.0, 0.0, 0.0])
    await db_session.commit()

    await run_lint(db_session, fix=True, semantic=False, actor=REINDEX_ACTOR)
    await db_session.commit()

    events = [e for e in await _retirement_events(db_session) if _names(e, p.id)]
    assert events and all(e.actor == "reindex" for e in events)
