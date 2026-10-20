# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The release pass: a claim held behind a retired claim is reconsidered.

Built on the Ledgerly case across passes. Session 3 says Ledgerly uses
PostgreSQL 16; a later session says its whole database is one file, which
contradicts the Postgres claim without filling its slot, so rung 3 holds it
quarantined behind an INCONSISTENCY record. A still later session says Ledgerly
uses SQLite, which retires the Postgres claim by update and leaves the record
with no ACTIVE side. The pass re-checks the held claim against what remains
and releases it only when nothing still contradicts it.

The write path runs for real, with its probes scripted at their module seams
(tests/AGENTS.md § Mocking strategy). The release pass's own LLM legs are the
same contradiction probe, resolved at call time from ``particles.ingest.pipeline``,
and the second reading, patched where the pass binds it. The encoder is a bag of
words so similarity is real and deterministic.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from particles.core.conflict_resolution import RETIRED_VALUE_KEY, SlotVerdict
from particles.core.contradiction_disclosure import CENSUS_ORIGIN, ORIGIN_KEY
from particles.core.schema import (
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    SourceType,
)
from particles.core.status import Status, StatusReason
from particles.ingest.second_reading import ClaimContext, ProbeVerdict
from tests.test_reanchor import (
    PG,
    SQLITE,
    T0,
    _BagOfWords,
    _one,
    _Sessions,
)

ONE_FILE = "Ledgerly's entire database is stored in a single file named `ledgerly.db`."
NO_SERVER = "Ledgerly's database needs no separate server process."
JSONB = "Ledgerly's migrations in `migrations/` use JSONB columns for the audit log."
#: A claim that contradicts the held one and stays ACTIVE when the partner goes.
SERVER = "Ledgerly's database runs as a separate server process on port 5432."

_LEDGERLY = "Ledgerly"


@dataclass
class _Probes:
    """The scripted probes, and the pairs they call contradictory (mutable per test)."""

    conflicts: dict[frozenset[str], SlotVerdict] = field(default_factory=dict)
    #: Pairs whose contradiction probe cannot complete (returns ``None``).
    unverifiable: set[frozenset[str]] = field(default_factory=set)
    probe: AsyncMock = field(default_factory=AsyncMock)
    update: AsyncMock = field(default_factory=AsyncMock)
    reading: AsyncMock = field(default_factory=AsyncMock)


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> Generator[_Probes, None, None]:
    """A bag-of-words encoder and every probe scripted for the Ledgerly pairs."""
    from particles import embeddings as ep
    from particles.config import get_config

    state = _Probes(
        conflicts={
            frozenset((PG, SQLITE)): SlotVerdict.CHANGES,
            # "One file" is not a new value of the database: rung 3 holds it.
            frozenset((PG, ONE_FILE)): SlotVerdict.FIXED,
            frozenset((PG, NO_SERVER)): SlotVerdict.FIXED,
            # Rung 2.5 would retire the older of the two: SERVER, dated earlier.
            frozenset((SERVER, ONE_FILE)): SlotVerdict.CHANGES,
        }
    )

    async def contradicts(a: str, b: str) -> bool | None:
        if frozenset((a, b)) in state.unverifiable:
            return None
        return frozenset((a, b)) in state.conflicts

    async def updates(earlier: str, later: str) -> SlotVerdict:
        return state.conflicts.get(frozenset((earlier, later)), SlotVerdict.DIFFERENT)

    async def read(a: ClaimContext, b: ClaimContext) -> ProbeVerdict:
        hit = frozenset((a.content, b.content)) in state.conflicts
        return ProbeVerdict(usable=True, contradicts=hit, description="scripted")

    state.probe.side_effect = contradicts
    state.update.side_effect = updates
    state.reading.side_effect = read
    # Every Ledgerly claim is in the subject pool, so the scripted probes alone
    # decide which pairs contradict.
    update_cfg = get_config().reconciliation.update_supersession
    monkeypatch.setattr(update_cfg, "subject_floor", 0.0)
    monkeypatch.setattr(update_cfg, "max_candidates", 10)
    # The write path's second reading has its own tests; the pass's is scripted.
    monkeypatch.setattr(get_config().extraction, "verify_conflicts", False)
    original = ep._embedding_model
    ep.set_embedding_model(_BagOfWords())  # type: ignore[arg-type]
    try:
        with (
            patch("particles.ingest.pipeline._has_contradiction_signal", state.probe),
            patch("particles.ingest.pipeline._has_update_signal", state.update),
            patch("particles.operations.held_release.read_pair", state.reading),
        ):
            yield state
    finally:
        ep.set_embedding_model(original)


async def _session(session: Any, ex: _Sessions, claims: list[str], *, day: int) -> None:
    from particles.corpus.deposit import deposit_text_versioned
    from particles.ingest.pipeline import extract_snapshot

    text = f"Session on day {day}.\n\n" + "\n\n".join(f"User: {c}" for c in claims)
    ex.claims[text.encode()] = [(c, [_LEDGERLY]) for c in claims]
    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text=text,
        uri_r=f"claude-code://session/{uuid.uuid4()}",
        source_type=SourceType.CONVERSATION,
        mutability=Mutability.STABLE,
        content_published_at=T0 + timedelta(days=day),
    )
    await session.commit()
    await extract_snapshot(session, entry_id, snapshot_id, extractor=ex)
    await session.commit()


async def _open_records(session: Any) -> list[Particle]:
    from particles.store.particle_store import get_particles_by_status

    return await get_particles_by_status(session, Status.INCONSISTENCY)


async def _held(session: Any, ex: _Sessions, *held: str) -> list[Particle]:
    """Session 3, then a session whose claims rung 3 holds behind the Postgres claim."""
    await _session(session, ex, [PG, JSONB], day=4)
    await _session(session, ex, list(held or (ONE_FILE,)), day=12)
    records = await _open_records(session)
    assert len(records) == len(held or (ONE_FILE,))
    for content in held or (ONE_FILE,):
        row = await _one(session, content)
        assert (row.status, row.status_reason) == (
            Status.PROVENANCE_STALE,
            StatusReason.CONFLICT_PENDING,
        )
    return records


async def _update(session: Any, ex: _Sessions) -> Particle:
    """The later session that retires the Postgres claim by update."""
    await _session(session, ex, [SQLITE], day=20)
    pg = await _one(session, PG)
    assert (pg.status, pg.status_reason) == (
        Status.PROVENANCE_STALE,
        StatusReason.SUPERSEDED_BY_UPDATE,
    )
    return await _one(session, SQLITE)


async def _statuses(session: Any) -> dict[str, tuple[Status, StatusReason | None]]:
    from particles.store.particle_store import get_particles_by_status

    out: dict[str, tuple[Status, StatusReason | None]] = {}
    for status in Status:
        for p in await get_particles_by_status(session, status):
            out[p.id] = (p.status, p.status_reason)
    return out


async def _events(session: Any, kind: str) -> list[Any]:
    from particles.store.event_store import OperatorEventType, list_events

    return await list_events(session, event_type=OperatorEventType(kind), limit=100)


async def _release(session: Any) -> Any:
    from particles.operations.held_release import run_release

    report = await run_release(session)
    await session.commit()
    return report


@pytest.mark.asyncio
class TestRelease:
    async def test_a_claim_held_behind_an_update_retired_claim_is_released(
        self, db_session: Any, probes: _Probes
    ) -> None:
        ex = _Sessions()
        [record] = await _held(db_session, ex)
        held = await _one(db_session, ONE_FILE)
        sqlite = await _update(db_session, ex)
        before = await _statuses(db_session)

        report = await _release(db_session)

        assert (report.qualifying, report.examined, len(report.released)) == (1, 1, 1)
        after = await _statuses(db_session)
        assert after[record.id] == (Status.RETRACTED, StatusReason.CONFLICT_RESOLVED)
        assert after[held.id] == (Status.SUPERSEDED, StatusReason.CONFLICT_RESOLVED)
        minted = await _one(db_session, ONE_FILE, Status.ACTIVE)
        assert minted.supersedes == held.id
        # Nothing else in the store changes.
        changed = {pid for pid in before if before[pid] != after.get(pid)}
        assert changed == {record.id, held.id}
        assert set(after) - set(before) == {minted.id}
        [event] = await _events(db_session, "INCONSISTENCY_CLOSED")
        pg = await _one(db_session, PG)
        assert event.payload["cause"] == "partner_retired"
        assert event.payload["origin"] == "extraction"
        assert event.payload["record_id"] == record.id
        assert event.payload["partner"] == pg.id
        assert event.payload["retired_by"] == sqlite.id
        assert event.payload["newcomer"] == held.id
        assert event.payload["successor"] == minted.id
        assert {r.ref_id for r in event.refs} == {record.id, pg.id, held.id, minted.id}

    async def test_a_verbatim_active_twin_is_folded_into_not_minted(
        self, db_session: Any, probes: _Probes
    ) -> None:
        from particles.store.particle_store import get_particle, insert_particle

        ex = _Sessions()
        await _held(db_session, ex)
        held = await _one(db_session, ONE_FILE)
        await _update(db_session, ex)
        other_source = ProvenanceRef(
            type=ProvenanceRefType.SOURCE,
            corpus_entry_id=str(uuid.uuid4()),
            snapshot_id=str(uuid.uuid4()),
        )
        twin = held.model_copy(
            update={
                "id": str(uuid.uuid4()),
                "status": Status.ACTIVE,
                "status_reason": None,
                "provenance": [other_source],
            }
        )
        await insert_particle(db_session, twin)
        await db_session.commit()

        report = await _release(db_session)

        assert report.released[0]["folded_into"] == twin.id
        active = await _one(db_session, ONE_FILE, Status.ACTIVE)
        assert active.id == twin.id  # no successor minted
        reread = await get_particle(db_session, held.id)
        assert reread is not None
        assert (reread.status, reread.status_reason) == (
            Status.SUPERSEDED,
            StatusReason.CONFLICT_RESOLVED,
        )
        folded = await get_particle(db_session, twin.id)
        assert folded is not None
        assert {r.corpus_entry_id for r in folded.provenance} >= {
            other_source.corpus_entry_id,
            *(r.corpus_entry_id for r in held.provenance),
        }
        [event] = await _events(db_session, "INCONSISTENCY_CLOSED")
        assert event.payload["folded_into"] == twin.id

    @pytest.mark.parametrize(
        ("status", "reason"),
        [
            (Status.SUPERSEDED, StatusReason.SUPERSEDED_BY_REANCHOR),
            (Status.PROVENANCE_STALE, StatusReason.DOCUMENT_SUPERSEDED),
        ],
    )
    async def test_the_other_autonomous_retirements_release_too(
        self, db_session: Any, probes: _Probes, status: Status, reason: StatusReason
    ) -> None:
        from particles.store.particle_store import update_particle_status

        ex = _Sessions()
        [record] = await _held(db_session, ex)
        pg = await _one(db_session, PG)
        await update_particle_status(db_session, pg.id, status, reason)
        await db_session.commit()

        report = await _release(db_session)

        assert len(report.released) == 1
        [event] = await _events(db_session, "INCONSISTENCY_CLOSED")
        assert event.payload["record_id"] == record.id
        assert event.payload["partner_reason"] == reason.value
        assert event.payload["retired_by"] is None
        assert (await _one(db_session, ONE_FILE, Status.ACTIVE)).content == ONE_FILE

    @pytest.mark.parametrize(
        ("status", "reason"),
        [
            (Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION),
            (Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION),
        ],
    )
    async def test_a_partner_a_person_retired_is_not_examined(
        self, db_session: Any, probes: _Probes, status: Status, reason: StatusReason
    ) -> None:
        from particles.store.particle_store import update_particle_status

        ex = _Sessions()
        await _held(db_session, ex)
        pg = await _one(db_session, PG)
        await update_particle_status(db_session, pg.id, status, reason)
        await db_session.commit()
        probes.probe.reset_mock()

        report = await _release(db_session)

        assert (report.qualifying, report.examined) == (0, 0)
        assert probes.probe.await_count == 0
        assert len(await _open_records(db_session)) == 1

    async def test_census_records_and_retired_value_holds_are_never_selected(
        self, db_session: Any, probes: _Probes
    ) -> None:
        from particles.store.particle_store import set_particle_property

        ex = _Sessions()
        [record] = await _held(db_session, ex)
        await _update(db_session, ex)
        for key, value in ((ORIGIN_KEY, CENSUS_ORIGIN), (RETIRED_VALUE_KEY, True)):
            await set_particle_property(db_session, record.id, key, value)
            await db_session.commit()
            report = await _release(db_session)
            assert report.qualifying == 0, key
            await set_particle_property(db_session, record.id, key, None)
            await db_session.commit()
        assert len(await _open_records(db_session)) == 1


@pytest.mark.asyncio
class TestBlocked:
    async def _blocked(self, session: Any) -> tuple[Particle, Particle, Particle, Particle]:
        ex = _Sessions()
        [record] = await _held(session, ex)
        # Dated before the held claim, so rung 2.5 would retire SERVER in its favour.
        await _session(session, ex, [SERVER], day=10)
        server = await _one(session, SERVER)
        assert server.status is Status.ACTIVE
        sqlite = await _update(session, ex)
        return record, await _one(session, ONE_FILE), server, sqlite

    async def test_nothing_is_written_to_any_claim_and_the_examination_is_recorded(
        self, db_session: Any, probes: _Probes
    ) -> None:
        from particles.operations.lint.open_inconsistency import open_inconsistency_findings

        record, held, server, sqlite = await self._blocked(db_session)
        before = await _statuses(db_session)
        probes.update.reset_mock()

        report = await _release(db_session)

        assert report.released == [] and len(report.blocked) == 1
        assert await _statuses(db_session) == before
        # The ladder is never consulted: the pass does not retire a claim it would beat.
        assert probes.update.await_count == 0
        assert await _events(db_session, "INCONSISTENCY_CLOSED") == []
        [event] = await _events(db_session, "HELD_CLAIM_EXAMINED")
        pg = await _one(db_session, PG)
        assert event.payload == {
            "record_id": record.id,
            "partner_id": pg.id,
            "partner_reason": "SUPERSEDED_BY_UPDATE",
            "retired_by": sqlite.id,
            "held_id": held.id,
            "blocking_ids": [server.id],
        }
        [finding] = await open_inconsistency_findings(db_session)
        assert "Claim A was retired by update in favour of" in finding.detail
        assert SQLITE in finding.detail
        assert f"the held claim B is still contradicted by “{SERVER}”" in finding.detail

    async def test_re_examined_only_once_a_blocker_leaves_active(
        self, db_session: Any, probes: _Probes
    ) -> None:
        from particles.store.particle_store import update_particle_status

        record, held, server, _ = await self._blocked(db_session)
        await _release(db_session)

        probes.probe.reset_mock()
        again = await _release(db_session)
        assert (again.examined, again.unchanged) == (0, 1)
        assert probes.probe.await_count == 0
        assert len(await _events(db_session, "HELD_CLAIM_EXAMINED")) == 1

        await update_particle_status(
            db_session, server.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()
        freed = await _release(db_session)
        assert len(freed.released) == 1
        assert (await _one(db_session, ONE_FILE, Status.ACTIVE)).supersedes == held.id
        assert await _open_records(db_session) == []


@pytest.mark.asyncio
class TestFailClosedAndCaps:
    async def test_a_failed_probe_keeps_the_record_open(
        self, db_session: Any, probes: _Probes
    ) -> None:
        ex = _Sessions()
        [record] = await _held(db_session, ex)
        await _update(db_session, ex)
        probes.unverifiable.add(frozenset((ONE_FILE, SQLITE)))
        before = await _statuses(db_session)

        report = await _release(db_session)

        assert (report.incomplete, report.released, report.blocked) == (1, [], [])
        assert await _statuses(db_session) == before
        assert await _events(db_session, "HELD_CLAIM_EXAMINED") == []
        # Nothing was recorded, so the next run asks again.
        probes.unverifiable.clear()
        assert len((await _release(db_session)).released) == 1
        assert record.id not in {r.id for r in await _open_records(db_session)}

    async def test_a_failed_second_reading_keeps_it_open(
        self, db_session: Any, probes: _Probes
    ) -> None:
        ex = _Sessions()
        await _held(db_session, ex)
        await _session(db_session, ex, [SERVER], day=10)
        await _update(db_session, ex)
        probes.reading.side_effect = None
        probes.reading.return_value = None

        report = await _release(db_session)

        assert report.incomplete == 1 and report.blocked == []
        assert len(await _open_records(db_session)) == 1

    async def test_the_cap_takes_the_oldest_and_leaves_the_rest(
        self, db_session: Any, probes: _Probes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        monkeypatch.setattr(get_config().consolidation.release, "max_records", 1)
        ex = _Sessions()
        await _held(db_session, ex, ONE_FILE, NO_SERVER)
        await _update(db_session, ex)

        first = await _release(db_session)
        assert (first.qualifying, first.examined, first.waiting) == (2, 1, 1)
        assert len(first.released) == 1
        second = await _release(db_session)
        assert (second.qualifying, len(second.released)) == (1, 1)
        assert await _open_records(db_session) == []

    async def test_a_store_with_nothing_qualifying_pays_nothing(
        self, db_session: Any, probes: _Probes
    ) -> None:
        ex = _Sessions()
        await _held(db_session, ex)
        probes.probe.reset_mock()
        report = await _release(db_session)
        assert (report.qualifying, report.llm_calls) == (0, 0)
        assert probes.probe.await_count == 0

    async def test_the_switch_turns_it_off(
        self, db_session: Any, probes: _Probes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        monkeypatch.setattr(get_config().consolidation.release, "enabled", False)
        ex = _Sessions()
        await _held(db_session, ex)
        await _update(db_session, ex)
        report = await _release(db_session)
        assert report.skipped_reason == "consolidation.release.enabled is false"
        assert len(await _open_records(db_session)) == 1

    async def test_the_plan_estimate_and_the_degraded_skip(
        self, db_session: Any, probes: _Probes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations.consolidation_plan import plan_consolidation

        # A key makes the plan non-degraded; the dry run sends nothing.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        ex = _Sessions()
        await _held(db_session, ex)
        await _update(db_session, ex)
        probes.probe.reset_mock()

        plan = await plan_consolidation(db_session)
        [release] = [p for p in plan.passes if p.name == "release"]
        assert release.action == "run" and release.llm_priced
        # The dry run selects the pairs and prices a probe and a reading on each.
        assert "1 held claim behind a retired claim, would examine 1" in release.detail
        assert release.calls > 0 and release.calls % 2 == 0
        assert probes.probe.await_count == 0
        assert len(await _open_records(db_session)) == 1

        degraded = await plan_consolidation(db_session, structural_only=True)
        [skipped] = [p for p in degraded.passes if p.name == "release"]
        assert skipped.action == "skip" and "LLM-priced" in skipped.detail


@pytest.mark.asyncio
class TestRaces:
    async def test_a_record_the_cascade_closed_after_selection_is_left_alone(
        self, db_session: Any, probes: _Probes
    ) -> None:
        """The re-read before the write (step 1)."""
        from particles.operations import held_release
        from particles.store.particle_store import update_particle_status

        ex = _Sessions()
        [record] = await _held(db_session, ex)
        held = await _one(db_session, ONE_FILE)
        await _update(db_session, ex)
        real_check = held_release._check

        async def check_then_cascade(*args: Any, **kwargs: Any) -> Any:
            out = await real_check(*args, **kwargs)
            # The trust cascade's closure shape: the record retracted, no event.
            await update_particle_status(
                db_session, record.id, Status.RETRACTED, StatusReason.CONFLICT_RESOLVED
            )
            await db_session.commit()
            return out

        with patch.object(held_release, "_check", check_then_cascade):
            report = await _release(db_session)

        assert report.closed_meanwhile == 1 and report.released == []
        assert await _events(db_session, "INCONSISTENCY_CLOSED") == []
        reread = await _one(db_session, ONE_FILE)
        assert reread.id == held.id and reread.status_reason is StatusReason.CONFLICT_PENDING


@pytest.mark.asyncio
class TestAfterRelease:
    async def test_a_re_assertion_is_reconciled_normally(
        self, db_session: Any, probes: _Probes
    ) -> None:
        """The held row ends SUPERSEDED, outside the judgment set.

        Pins the reason the first draft's §4 was withdrawn: a released value
        said again is not held as the re-assertion of a retired one.
        """
        from particles.ingest.duplicate_suppression import (
            build_retired_value_index,
            is_judgment_retired,
            is_pending_hold,
        )
        from particles.store.particle_store import get_particle

        ex = _Sessions()
        await _held(db_session, ex)
        held = await _one(db_session, ONE_FILE)
        await _update(db_session, ex)
        await _release(db_session)
        reread = await get_particle(db_session, held.id)
        assert reread is not None
        assert not is_judgment_retired(reread) and not is_pending_hold(reread)
        index = await build_retired_value_index(db_session, [ONE_FILE])
        assert index.find(reread) is None

        minted = await _one(db_session, ONE_FILE, Status.ACTIVE)
        await _session(db_session, ex, [ONE_FILE], day=30)
        assert await _open_records(db_session) == []
        again = await _one(db_session, ONE_FILE, Status.ACTIVE)
        assert again.id == minted.id  # the repeat folded into the released claim


class TestTheSpeakerKey:
    """the release pool reads the widened key."""

    def test_the_pool_reaches_a_claim_keyed_only_on_the_speaker(self) -> None:
        import numpy as np

        from particles.operations.held_release import select_pairs
        from tests.test_update_supersession import _p, _speaker

        held = _p("The user is not going back to Delhi.", subject_ids=["delhi", "persona"])
        flight = _p("The user flies back home to Delhi on Thursday.", subject_ids=["persona"])
        emb = np.array([1.0, 0.0], dtype=np.float32)
        # Cosine 0.7: below the store-wide 0.80 search, above the 0.45 subject floor.
        near = np.array([0.7, np.sqrt(1 - 0.49)], dtype=np.float32)
        pool = [(flight, near)]
        assert select_pairs(held, emb, pool, exclude=set(), speaker=_speaker()) == [flight]
        assert select_pairs(held, emb, pool, exclude=set(), speaker=None) == []
