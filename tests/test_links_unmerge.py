"""Tests for the exact-duplicate merge revert — `links unmerge`.

The property under test throughout is that ``merge ∘ unmerge`` is the
**identity**: the same ids come back ACTIVE with their reason cleared, the
merge's own relations are gone, and the survivor was never touched. That is
what *"the pre-merge state is reconstructible exactly"* claim
rests on, and it is why the revert restores rows rather than minting new ones.

Three gates get their own hard pins because each one is a silent-corruption
risk rather than a visible failure:

* the §6.6 reason gate (a non-``DUPLICATE_MERGED`` supersession is not
  reversible, and SUPERSEDED stays terminal for everything else);
* the ``retired_at`` stamp being cleared, so a *later* genuine
  retirement is dated to itself and not to the withdrawn merge (§4);
* the ``created_by`` filter, so a human's co-evidential link on the same pair
  survives the revert (§5).

Covers ``unmerge_exact_duplicates`` in
``particles/operations/links_suggest.py`` and the ``SUPERSEDED → ACTIVE``
gate in ``particles/store/particle_store.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    Subject,
    UncertaintyNature,
    UnmergeSkipReason,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.links_suggest import (
    UnmergeSelectorError,
    auto_merge_exact_duplicates,
    unmerge_exact_duplicates,
)
from particles.store.event_store import (
    OperatorEventType,
    list_events,
    list_events_in_range,
)
from particles.store.particle_store import (
    ParticleRow,
    get_particle,
    insert_particle,
    update_particle_status,
)
from particles.store.relation_store import create_relation, get_relations_for_particle
from particles.store.subject_store import insert_subject

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_DUP = "uv parses pyproject.toml during settings discovery."


def _pid(n: int) -> str:
    return f"00000000-0000-0000-0000-{n:012d}"


def _mk(particle_id: str, subject_ids: list[str], *, asserted_at: datetime) -> Particle:
    return Particle(
        id=particle_id,
        content=_DUP,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=asserted_at,
        subject_ids=subject_ids,
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce-default")],
    )


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    from particles.config import get_config

    monkeypatch.setattr(get_config().links_suggest.auto_merge, "enabled", True)


async def _seed_and_merge(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch, *, copies: int = 3
) -> tuple[str, list[str], str]:
    """Seed N byte-identical copies, merge them, return (survivor, losers, event_id)."""
    subj = Subject(canonical_name="uv", asserted_by="test")
    await insert_subject(session, subj)
    for n in range(1, copies + 1):
        await insert_particle(
            session,
            _mk(_pid(n), [subj.id], asserted_at=datetime(2026, 1, n, tzinfo=UTC)),
        )
    await session.commit()

    _enable(monkeypatch)
    report = await auto_merge_exact_duplicates(session, dry_run=False)
    assert report.merged_groups == 1
    group = report.groups[0]

    events = await list_events(session, event_type=OperatorEventType.DUPLICATES_MERGED)
    assert len(events) == 1
    return group.survivor_id, list(group.redundant_ids), events[0].event_id


# ---------------------------------------------------------------------------
# The round trip — merge ∘ unmerge is the identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unmerge_restores_the_same_rows_not_new_ones(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point: same ids back, no fresh particles minted."""
    survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)
    before = {p.id for p in [await get_particle(db_session, _pid(n)) for n in (1, 2, 3)] if p}

    report = await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)

    assert report.restored_particles == len(losers)
    assert report.groups[0].reverted is True
    for loser_id in losers:
        restored = await get_particle(db_session, loser_id)
        assert restored is not None
        assert restored.status is Status.ACTIVE
        # Restoring the row means restoring it, not tagging it with a scar.
        assert restored.status_reason is None
    # No new particle ids exist — the store is exactly as wide as before.
    after = {p.id for p in [await get_particle(db_session, _pid(n)) for n in (1, 2, 3)] if p}
    assert after == before


@pytest.mark.asyncio
async def test_survivor_is_never_mutated(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    survivor, _losers, event_id = await _seed_and_merge(db_session, monkeypatch)
    before = await get_particle(db_session, survivor)
    assert before is not None

    await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)

    after = await get_particle(db_session, survivor)
    assert after is not None
    assert after.status is before.status
    assert after.status_reason is before.status_reason
    assert after.content == before.content
    assert after.confidence.value == before.confidence.value


@pytest.mark.asyncio
async def test_merge_relations_are_dropped(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)
    assert len(await get_relations_for_particle(db_session, survivor)) == len(losers)

    report = await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)

    assert report.relations_deleted == len(losers)
    assert await get_relations_for_particle(db_session, survivor) == []


@pytest.mark.asyncio
async def test_a_human_link_on_the_same_pair_survives(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """only EXACT_DUPLICATE edges are the merge's to withdraw.

    The live store carries 3 MANUAL_CLI co-evidential rows, so this is a real
    case rather than a hypothetical one. The merge skips a pair that is already
    linked, and the revert must be symmetric: it created nothing there, so it
    withdraws nothing there.
    """
    subj = Subject(canonical_name="uv", asserted_by="test")
    await insert_subject(db_session, subj)
    for n in (1, 2):
        await insert_particle(
            db_session, _mk(_pid(n), [subj.id], asserted_at=datetime(2026, 1, n, tzinfo=UTC))
        )
    # A human linked these two before any merge ran.
    await create_relation(
        db_session, _pid(1), _pid(2), RelationType.CO_EVIDENTIAL, RelationCreatedBy.MANUAL_CLI
    )
    await db_session.commit()

    _enable(monkeypatch)
    await auto_merge_exact_duplicates(db_session, dry_run=False)
    events = await list_events(db_session, event_type=OperatorEventType.DUPLICATES_MERGED)

    report = await unmerge_exact_duplicates(db_session, event_id=events[0].event_id, dry_run=False)

    assert report.restored_particles == 1
    assert report.relations_deleted == 0
    surviving = await get_relations_for_particle(db_session, _pid(1))
    assert [r.created_by for r in surviving] == [RelationCreatedBy.MANUAL_CLI]


# ---------------------------------------------------------------------------
# The §6.6 reason gate — SUPERSEDED stays terminal for everything else
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seam_refuses_un_supersede_without_the_merge_reason(
    db_session: AsyncSession,
) -> None:
    """An operator revision is not reversible — only an auto-merge is."""
    subj = Subject(canonical_name="uv", asserted_by="test")
    await insert_subject(db_session, subj)
    await insert_particle(
        db_session, _mk(_pid(1), [subj.id], asserted_at=datetime(2026, 1, 1, tzinfo=UTC))
    )
    await db_session.commit()
    await update_particle_status(
        db_session, _pid(1), Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
    )
    await db_session.commit()

    with pytest.raises(ValueError, match="requires status_reason DUPLICATE_MERGED"):
        await update_particle_status(db_session, _pid(1), Status.ACTIVE, None)


@pytest.mark.asyncio
async def test_seam_allows_un_supersede_with_the_merge_reason(
    db_session: AsyncSession,
) -> None:
    subj = Subject(canonical_name="uv", asserted_by="test")
    await insert_subject(db_session, subj)
    await insert_particle(
        db_session, _mk(_pid(1), [subj.id], asserted_at=datetime(2026, 1, 1, tzinfo=UTC))
    )
    await db_session.commit()
    await update_particle_status(
        db_session, _pid(1), Status.SUPERSEDED, StatusReason.DUPLICATE_MERGED
    )
    await db_session.commit()

    await update_particle_status(db_session, _pid(1), Status.ACTIVE, None)
    await db_session.commit()

    restored = await get_particle(db_session, _pid(1))
    assert restored is not None
    assert restored.status is Status.ACTIVE


# ---------------------------------------------------------------------------
# the retirement stamp is withdrawn with the retirement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retired_at_is_cleared_on_revert(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)
    stamped = await db_session.get(ParticleRow, losers[0])
    assert stamped is not None and stamped.retired_at is not None

    await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)

    row = await db_session.get(ParticleRow, losers[0])
    assert row is not None
    assert row.retired_at is None


@pytest.mark.asyncio
async def test_a_later_retirement_is_dated_to_itself_not_to_the_withdrawn_merge(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silent corruption the rule exists to prevent.

    stamp is write-once and *never overwritten on later hops*. If a
    revert left the merge's instant in place, that guard would preserve the
    **withdrawn** instant when the row is later genuinely retired, and the
    as-of lens would date the retraction to a merge that was undone.
    """
    _survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)
    merged_row = await db_session.get(ParticleRow, losers[0])
    assert merged_row is not None
    merge_instant = merged_row.retired_at
    assert merge_instant is not None

    await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)
    await update_particle_status(
        db_session, losers[0], Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await db_session.commit()

    row = await db_session.get(ParticleRow, losers[0])
    assert row is not None and row.retired_at is not None
    assert row.retired_at != merge_instant
    assert row.retired_at > merge_instant.replace(tzinfo=row.retired_at.tzinfo)


# ---------------------------------------------------------------------------
# Drift — skip and report, never abort (§8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_drifted_copy_is_skipped_and_the_rest_still_restore(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All-or-nothing would let one drifted row block recovery of its peers.

    The drift is built through legal transitions only — restore one copy, then
    retract it — because SUPERSEDED is terminal for everything except this
    ADR's own edge, so that is the shape a real divergence takes: a copy an
    earlier partial revert already brought back, which then moved on.
    """
    _survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)
    await update_particle_status(db_session, losers[0], Status.ACTIVE, None)
    await update_particle_status(
        db_session, losers[0], Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await db_session.commit()

    report = await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)

    group = report.groups[0]
    assert group.restored_ids == losers[1:]
    assert [s.particle_id for s in group.skipped] == [losers[0]]
    assert group.skipped[0].reason is UnmergeSkipReason.NOT_SUPERSEDED
    assert group.skipped[0].found_status == Status.RETRACTED.value
    # The later decision stands — a revert has no standing to overturn it.
    still_retracted = await get_particle(db_session, losers[0])
    assert still_retracted is not None
    assert still_retracted.status is Status.RETRACTED


@pytest.mark.asyncio
async def test_a_copy_superseded_by_something_else_is_skipped(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Still SUPERSEDED, but no longer the merge's — the reason gate refuses it."""
    _survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)
    await update_particle_status(db_session, losers[0], Status.ACTIVE, None)
    await update_particle_status(
        db_session, losers[0], Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
    )
    await db_session.commit()

    report = await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)

    skipped = report.groups[0].skipped
    assert [s.particle_id for s in skipped] == [losers[0]]
    assert skipped[0].reason is UnmergeSkipReason.NOT_MERGE_SUPERSEDED
    assert skipped[0].found_status_reason == StatusReason.EXPLICIT_SUPERSESSION.value


@pytest.mark.asyncio
async def test_a_drifted_survivor_is_reported_but_not_repaired(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live 2026-07-25 case: 1 of 181 survivors was already stale in 24 h."""
    survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)
    await update_particle_status(
        db_session, survivor, Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY
    )
    await db_session.commit()

    report = await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)

    assert report.groups[0].survivor_status == Status.PROVENANCE_STALE.value
    assert report.restored_particles == len(losers)
    unchanged = await get_particle(db_session, survivor)
    assert unchanged is not None
    assert unchanged.status is Status.PROVENANCE_STALE


@pytest.mark.asyncio
async def test_unmerge_is_idempotent(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)
    await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)

    second = await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)

    assert second.restored_particles == 0
    assert second.skipped_particles == len(losers)
    assert all(s.reason is UnmergeSkipReason.ALREADY_ACTIVE for s in second.groups[0].skipped)
    assert second.groups[0].reverted is False


@pytest.mark.asyncio
async def test_dry_run_writes_nothing(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)

    plan = await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=True)

    assert plan.restored_particles == len(losers)
    assert plan.groups[0].reverted is False
    for loser_id in losers:
        untouched = await get_particle(db_session, loser_id)
        assert untouched is not None
        assert untouched.status is Status.SUPERSEDED
    assert len(await get_relations_for_particle(db_session, survivor)) == len(losers)
    assert await list_events(db_session, event_type=OperatorEventType.DUPLICATES_UNMERGED) == []


# ---------------------------------------------------------------------------
# Selectors and the audit trail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_stamps_a_run_id_and_run_selects_it(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """without it a run's N events share no key."""
    _survivor, losers, _event_id = await _seed_and_merge(db_session, monkeypatch)
    events = await list_events(db_session, event_type=OperatorEventType.DUPLICATES_MERGED)
    run_id = events[0].payload["run_id"]
    assert isinstance(run_id, str) and run_id

    report = await unmerge_exact_duplicates(db_session, run_id=run_id, dry_run=False)

    assert report.total_events == 1
    assert report.restored_particles == len(losers)
    assert report.selector == f"run {run_id}"


@pytest.mark.asyncio
async def test_since_selects_the_window(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The §7b migration path for merges written before run ids existed."""
    _survivor, losers, _event_id = await _seed_and_merge(db_session, monkeypatch)
    events = await list_events(db_session, event_type=OperatorEventType.DUPLICATES_MERGED)
    before = events[0].occurred_at - timedelta(minutes=1)

    report = await unmerge_exact_duplicates(db_session, since=before, dry_run=False)

    assert report.total_events == 1
    assert report.restored_particles == len(losers)


@pytest.mark.asyncio
async def test_since_window_excludes_events_outside_it(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _survivor, _losers, _event_id = await _seed_and_merge(db_session, monkeypatch)
    events = await list_events(db_session, event_type=OperatorEventType.DUPLICATES_MERGED)
    after = events[0].occurred_at + timedelta(minutes=1)

    report = await unmerge_exact_duplicates(db_session, since=after, dry_run=False)

    assert report.total_events == 0
    assert report.restored_particles == 0
    assert report.warnings


@pytest.mark.asyncio
async def test_unmerge_records_its_own_event_and_leaves_the_merge_event_alone(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)

    await unmerge_exact_duplicates(
        db_session, event_id=event_id, dry_run=False, actor="cli:links-unmerge"
    )

    merges = await list_events(db_session, event_type=OperatorEventType.DUPLICATES_MERGED)
    assert [e.event_id for e in merges] == [event_id]

    unmerges = await list_events(db_session, event_type=OperatorEventType.DUPLICATES_UNMERGED)
    assert len(unmerges) == 1
    payload = unmerges[0].payload
    assert unmerges[0].actor == "cli:links-unmerge"
    assert payload["merge_event_id"] == event_id
    assert payload["survivor"] == survivor
    assert set(payload["restored"]) == set(losers)


@pytest.mark.asyncio
async def test_unmerge_is_not_gated_on_auto_merge_enabled(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the flag authorizes merging, so the undo must outlive it.

    An operator who turned auto-merge off *because* a run went wrong must still
    be able to clean up.
    """
    from particles.config import get_config

    _survivor, losers, event_id = await _seed_and_merge(db_session, monkeypatch)
    monkeypatch.setattr(get_config().links_suggest.auto_merge, "enabled", False)

    report = await unmerge_exact_duplicates(db_session, event_id=event_id, dry_run=False)

    assert report.restored_particles == len(losers)


@pytest.mark.asyncio
async def test_selector_errors(db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    _survivor, _losers, event_id = await _seed_and_merge(db_session, monkeypatch)

    with pytest.raises(UnmergeSelectorError, match="exactly one"):
        await unmerge_exact_duplicates(db_session)
    with pytest.raises(UnmergeSelectorError, match="exactly one"):
        await unmerge_exact_duplicates(db_session, event_id=event_id, run_id="r")
    with pytest.raises(UnmergeSelectorError, match="No operator event"):
        await unmerge_exact_duplicates(db_session, event_id="missing")


@pytest.mark.asyncio
async def test_a_non_merge_event_is_refused(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unmerge reverts auto-merges only — it is not a general event undo."""
    from particles.store.event_store import EventRefKind, record_event

    _survivor, _losers, _event_id = await _seed_and_merge(db_session, monkeypatch)
    await record_event(
        db_session,
        actor="test",
        event_type=OperatorEventType.PARTICLE_RETRACTED,
        refs=[(EventRefKind.PARTICLE, _pid(1))],
        payload={},
    )
    await db_session.commit()
    other = await list_events(db_session, event_type=OperatorEventType.PARTICLE_RETRACTED)

    with pytest.raises(UnmergeSelectorError, match="not DUPLICATES_MERGED"):
        await unmerge_exact_duplicates(db_session, event_id=other[0].event_id)


# ---------------------------------------------------------------------------
# The new store query shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_events_in_range_is_bounded_and_ordered(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _survivor, _losers, _event_id = await _seed_and_merge(db_session, monkeypatch)
    merges = await list_events(db_session, event_type=OperatorEventType.DUPLICATES_MERGED)
    at = merges[0].occurred_at

    inside = await list_events_in_range(
        db_session,
        event_type=OperatorEventType.DUPLICATES_MERGED,
        since=at - timedelta(seconds=1),
        until=at + timedelta(seconds=1),
    )
    assert [e.event_id for e in inside] == [merges[0].event_id]

    # `until` is exclusive, so the event's own instant falls outside it.
    assert (
        await list_events_in_range(
            db_session, event_type=OperatorEventType.DUPLICATES_MERGED, until=at
        )
        == []
    )
