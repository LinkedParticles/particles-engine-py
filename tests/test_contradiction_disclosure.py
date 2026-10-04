# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pass 3b, the nightly contradiction disclosure, against a store."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from particles.config import get_config
from particles.core.contradiction_disclosure import (
    DISCLOSURE_ACTOR,
    ConfirmedPair,
    census_replaces,
    census_sides,
)
from particles.core.schema import ProvenanceRef, ProvenanceRefType, ResolutionAction
from particles.core.status import Status, StatusReason
from particles.operations.contradiction_disclosure import covered_pair_set, run_disclosure
from particles.store.event_store import (
    OperatorEventType,
    list_events,
    record_event,
)
from particles.store.particle_store import (
    append_provenance_ref,
    get_census_records,
    get_inconsistency_backrefs,
    get_particle,
    update_particle_status,
)
from tests._observer_scope import belief, generation, project_source_tags, rescoped
from tests.test_update_supersession import bow  # noqa: F401 — a fixture

EARLY = datetime(2026, 9, 18, tzinfo=UTC)
LATE = datetime(2026, 9, 25, tzinfo=UTC)


def _pair(a: str, b: str, reason: str = "they disagree", same: bool = False) -> ConfirmedPair:
    return ConfirmedPair(a=a, b=b, same_source=same, reason=reason)


async def _note(
    session: Any, name: str, text: str, tags: list[str] | None = None, when: Any = None
) -> tuple[str, str]:
    return await generation(session, name, text, tags or [], captured_at=when)


async def _open(session: Any) -> list[Any]:
    return [r for r in await get_census_records(session) if r.status is Status.INCONSISTENCY]


async def _evidence(session: Any) -> tuple[Any, Any, tuple[str, str], tuple[str, str]]:
    """The evidence case: one note says the endpoint exists, a later one that it 404s."""
    n1 = await _note(session, "sept18.md", "stats/total exists", when=EARLY)
    n2 = await _note(session, "sept25.md", "stats/total returns not found", when=LATE)
    a = await belief(session, "GoatCounter's stats/total endpoint exists", n1)
    b = await belief(session, "GoatCounter's stats/total endpoint returns not found", n2)
    await session.commit()
    return a, b, n1, n2


class TestMint:
    @pytest.mark.asyncio
    async def test_confirmed_cross_source_pair_opens_one_record(self, db_session: Any) -> None:
        a, b, _, n2 = await _evidence(db_session)
        # Probe order has the newer claim first; the record puts the older first.
        report = await run_disclosure(db_session, confirmed=[_pair(b.id, a.id)], mint=True)

        assert report.opened_records == 1
        records = await _open(db_session)
        assert len(records) == 1
        record = records[0]
        assert census_sides(record) is not None
        assert census_sides(record).a == (a.id,)  # type: ignore[union-attr]
        assert census_sides(record).b == (b.id,)  # type: ignore[union-attr]
        trigger = record.provenance[2]
        assert (trigger.type, trigger.corpus_entry_id) == (ProvenanceRefType.SOURCE, n2[0])
        # Disclosure only: neither claim changed.
        for claim in (a, b):
            stored = await get_particle(db_session, claim.id)
            assert stored is not None and stored.status is Status.ACTIVE
            assert stored.confidence.value == claim.confidence.value
        # The digest's contested marker reads these backrefs.
        backrefs = await get_inconsistency_backrefs(db_session)
        assert backrefs[a.id] == backrefs[b.id] == record.id
        assert "sept18.md, 2026-09-18" in record.content

    @pytest.mark.asyncio
    async def test_pairs_sharing_a_claim_open_one_record(self, db_session: Any) -> None:
        n1 = await _note(db_session, "n1.md", "one", when=EARLY)
        n2 = await _note(db_session, "n2.md", "two", when=LATE)
        a1 = await belief(db_session, "endpoint exists", n1)
        a2 = await belief(db_session, "the endpoint is live", n1)
        b = await belief(db_session, "endpoint returns not found", n2)
        await db_session.commit()

        report = await run_disclosure(
            db_session, confirmed=[_pair(a1.id, b.id), _pair(a2.id, b.id)], mint=True
        )

        assert report.opened == [{"record_ids": report.opened[0]["record_ids"], "members": 3}]
        records = await _open(db_session)
        assert len(records) == 1
        backrefs = await get_inconsistency_backrefs(db_session)
        assert {backrefs[a1.id], backrefs[a2.id], backrefs[b.id]} == {records[0].id}

    @pytest.mark.asyncio
    async def test_same_source_and_dead_pairs_never_mint(self, db_session: Any) -> None:
        a, b, n1, _ = await _evidence(db_session)
        sibling = await belief(db_session, "a sibling claim", n1)
        await update_particle_status(
            db_session, b.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()

        report = await run_disclosure(
            db_session,
            confirmed=[_pair(a.id, sibling.id, same=True), _pair(a.id, b.id)],
            mint=True,
            unconfirmed=4,
        )

        assert report.opened == []
        assert report.skipped == {"unconfirmed": 4, "same_source": 1, "not_live": 1}
        assert await _open(db_session) == []

    @pytest.mark.asyncio
    async def test_a_second_run_does_not_mint_again(self, db_session: Any) -> None:
        a, b, _, _ = await _evidence(db_session)
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        report = await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)

        assert report.opened == []
        assert report.skipped["covered"] == 1
        assert len(await _open(db_session)) == 1
        assert frozenset((a.id, b.id)) in await covered_pair_set(db_session)

    @pytest.mark.asyncio
    async def test_nothing_mints_when_minting_is_off(self, db_session: Any) -> None:
        a, b, _, _ = await _evidence(db_session)
        report = await run_disclosure(
            db_session,
            confirmed=[_pair(a.id, b.id)],
            mint=False,
            not_minting_reason="the census did not run this night",
        )
        assert report.opened == []
        assert report.not_minting_reason == "the census did not run this night"
        assert await _open(db_session) == []

    @pytest.mark.asyncio
    async def test_switch_off_does_nothing(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().consolidation.contradiction_disclosure, "enabled", False)
        a, b, _, _ = await _evidence(db_session)
        report = await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        assert not report.enabled
        assert await get_census_records(db_session) == []


class TestObserver:
    @pytest.mark.asyncio
    async def test_disjoint_projects_are_not_disclosed(self, db_session: Any) -> None:
        alpha = await _note(db_session, "alpha.md", "x", project_source_tags("alpha"), EARLY)
        beta = await _note(db_session, "beta.md", "y", project_source_tags("beta"), LATE)
        a = await belief(db_session, "the branch is main", alpha)
        b = await belief(db_session, "the branch is trunk", beta)
        await rescoped(db_session)
        await db_session.commit()

        report = await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)

        assert report.skipped == {"observer_disjoint": 1}
        assert await get_census_records(db_session) == []
        from particles.core.schema import RelationType
        from particles.store.relation_store import get_all_relations

        # Nothing is written for a disjoint pair: not even a relation.
        assert await get_all_relations(db_session, RelationType.CONTRADICTS) == []

    @pytest.mark.asyncio
    async def test_a_shared_project_is_disclosed(self, db_session: Any) -> None:
        one = await _note(db_session, "one.md", "x", project_source_tags("alpha"), EARLY)
        two = await _note(db_session, "two.md", "y", project_source_tags("alpha"), LATE)
        a = await belief(db_session, "the branch is main", one)
        b = await belief(db_session, "the branch is trunk", two)
        await rescoped(db_session)
        await db_session.commit()

        report = await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        assert report.opened_records == 1


class TestLapse:
    @pytest.mark.asyncio
    async def test_record_closes_when_a_side_is_retracted(self, db_session: Any) -> None:
        a, b, _, _ = await _evidence(db_session)
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        record = (await _open(db_session))[0]
        await update_particle_status(
            db_session, b.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()

        report = await run_disclosure(db_session, confirmed=[], mint=False)

        closed = await get_particle(db_session, record.id)
        assert closed is not None
        assert (closed.status, closed.status_reason) == (
            Status.RETRACTED,
            StatusReason.CONFLICT_RESOLVED,
        )
        assert [c["cause"] for c in report.closed] == ["lapsed"]
        # Neither claim is touched by the close, and no claim gets a judgment reason.
        kept = await get_particle(db_session, a.id)
        assert kept is not None and kept.status is Status.ACTIVE
        events = await list_events(db_session, event_type=OperatorEventType.INCONSISTENCY_CLOSED)
        assert len(events) == 1
        assert events[0].actor == DISCLOSURE_ACTOR
        assert events[0].payload is not None and events[0].payload["cause"] == "lapsed"
        assert a.id not in await get_inconsistency_backrefs(db_session)

    @pytest.mark.asyncio
    async def test_record_closes_when_the_note_stops_stating_a_side(self, db_session: Any) -> None:
        """The worked example: the Sept 25 note is corrected; its claim is no longer stated."""
        a, b, _, _ = await _evidence(db_session)
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        # The corrected note is a new, extracted generation that does not state b.
        corrected = await _note(db_session, "sept25.md", "stats/total exists after all")
        await db_session.commit()

        report = await run_disclosure(db_session, confirmed=[], mint=False)

        assert [c["cause"] for c in report.closed] == ["lapsed"]
        dropped = report.closed[0]["dropped"]
        assert dropped[0]["particle_id"] == b.id
        assert dropped[0]["state"] == "lapsed"
        assert dropped[0]["stopped_by"] == [corrected[1]]
        # The lapse is not a judgment: b keeps its status.
        stored = await get_particle(db_session, b.id)
        assert stored is not None and stored.status is Status.ACTIVE

    @pytest.mark.asyncio
    async def test_regroup_drops_a_lapsed_member_and_names_the_old_record(
        self, db_session: Any
    ) -> None:
        n1 = await _note(db_session, "n1.md", "one", when=EARLY)
        n3 = await _note(db_session, "n3.md", "three", when=EARLY)
        n2 = await _note(db_session, "n2.md", "two", when=LATE)
        a1 = await belief(db_session, "endpoint exists", n1)
        a2 = await belief(db_session, "the endpoint is live", n3)
        b = await belief(db_session, "endpoint returns not found", n2)
        await db_session.commit()
        await run_disclosure(
            db_session, confirmed=[_pair(a1.id, b.id), _pair(a2.id, b.id)], mint=True
        )
        old = (await _open(db_session))[0]
        await _note(db_session, "n3.md", "three, rewritten without the claim")
        await db_session.commit()

        report = await run_disclosure(db_session, confirmed=[], mint=False)

        assert [c["cause"] for c in report.closed] == ["regrouped"]
        # A replacement is not a new disclosure.
        assert report.opened == []
        records = await _open(db_session)
        assert len(records) == 1
        new = records[0]
        assert census_replaces(new) == old.id
        assert set(census_sides(new).members) == {a1.id, b.id}  # type: ignore[union-attr]
        assert report.closed[0]["replacements"] == [new.id]

    @pytest.mark.asyncio
    async def test_a_gone_member_with_its_partner_intact_keeps_the_id(
        self, db_session: Any
    ) -> None:
        n1 = await _note(db_session, "n1.md", "one", when=EARLY)
        n2 = await _note(db_session, "n2.md", "two", when=LATE)
        a = await belief(db_session, "endpoint exists", n1)
        b1 = await belief(db_session, "endpoint returns not found", n2)
        b2 = await belief(db_session, "the endpoint 404s", n2)
        await db_session.commit()
        await run_disclosure(
            db_session, confirmed=[_pair(a.id, b1.id), _pair(a.id, b2.id)], mint=True
        )
        old = (await _open(db_session))[0]
        await update_particle_status(
            db_session, b2.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()

        report = await run_disclosure(db_session, confirmed=[], mint=False)

        assert report.closed == []
        assert [r.id for r in await _open(db_session)] == [old.id]


class TestCap:
    @pytest.mark.asyncio
    async def test_past_the_cap_waits_for_the_next_run(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().consolidation.contradiction_disclosure, "max_per_run", 1)
        n1 = await _note(db_session, "n1.md", "one", when=EARLY)
        n2 = await _note(db_session, "n2.md", "two", when=LATE)
        pairs = []
        for topic in ("port", "branch"):
            x = await belief(db_session, f"the {topic} is A", n1)
            y = await belief(db_session, f"the {topic} is B", n2)
            pairs.append(_pair(x.id, y.id, f"{topic} disagrees"))
        await db_session.commit()

        first = await run_disclosure(db_session, confirmed=pairs, mint=True)
        assert first.opened_records == 1
        assert [w["reason"] for w in first.waiting] == ["branch disagrees"]
        # The run record carries the waiting list, as the cycle writes it.
        await record_event(
            db_session,
            actor=DISCLOSURE_ACTOR,
            event_type=OperatorEventType.CONSOLIDATION_RUN,
            payload={"census": {"disclosure": first.payload()}},
        )
        await db_session.commit()

        second = await run_disclosure(db_session, confirmed=[], mint=False)
        assert second.opened_records == 1
        assert second.waiting == []
        assert len(await _open(db_session)) == 2


class TestCoverageAfterAClose:
    @pytest.mark.asyncio
    async def test_review_close_covers_until_a_new_source_states_a_side(
        self, db_session: Any
    ) -> None:
        from particles.operations.review import resolve

        a, b, _, _ = await _evidence(db_session)
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        record = (await _open(db_session))[0]
        await resolve(db_session, record.id, ResolutionAction.BOTH_VALID, "owner")

        assert frozenset((a.id, b.id)) in await covered_pair_set(db_session)

        # A third note now states b: the owner's judgment predates it.
        n3 = await _note(db_session, "n3.md", "it 404s")
        await append_provenance_ref(
            db_session,
            b.id,
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=n3[0], snapshot_id=n3[1]),
        )
        await db_session.commit()
        assert frozenset((a.id, b.id)) not in await covered_pair_set(db_session)

    @pytest.mark.asyncio
    async def test_a_lapse_close_covers_nothing(self, db_session: Any) -> None:
        a, b, _, _ = await _evidence(db_session)
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        await update_particle_status(
            db_session, b.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()
        await run_disclosure(db_session, confirmed=[], mint=False)
        assert await covered_pair_set(db_session) == frozenset()


class TestReread:
    """A record confirmed under an instruction that has since changed is read again."""

    @staticmethod
    async def _records(session: Any) -> tuple[Any, Any]:
        """Two session-transcript claims: one count observed at two moments."""
        from particles.core.schema import Mutability

        t1 = await generation(
            session,
            "session-1",
            "the gate passes with 10 rows",
            [],
            mutability=Mutability.APPEND_ONLY,
            captured_at=EARLY,
        )
        t2 = await generation(
            session,
            "session-2",
            "the gate passes with 12 rows",
            [],
            mutability=Mutability.APPEND_ONLY,
            captured_at=LATE,
        )
        a = await belief(session, "The gate passes with 10 rows.", t1)
        b = await belief(session, "The gate passes with 12 rows.", t2)
        await session.commit()
        return a, b

    @staticmethod
    def _reader(*verdicts: bool | None) -> Any:
        from unittest.mock import AsyncMock

        from particles.operations.lint.contradictions import ProbeVerdict

        replies = [
            None
            if v is None
            else ProbeVerdict(usable=True, contradicts=v, description=f"verdict {v}")
            for v in verdicts
        ]
        return AsyncMock(side_effect=replies)

    @pytest.mark.asyncio
    async def test_a_rejected_pair_closes_its_record_as_withdrawn(self, db_session: Any) -> None:
        from unittest.mock import patch

        a, b = await self._records(db_session)
        # Minted before readings were named: the pair carries the standing instruction.
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        [record] = await _open(db_session)
        reader = self._reader(False)

        with patch("particles.operations.lint.contradictions._llm_verify_contradiction", reader):
            report = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)

        assert reader.await_count == 1
        assert report.reread == {
            "read": 1,
            "confirmed": 0,
            "withdrawn": 1,
            "failed": 0,
            "deferred": 0,
        }
        assert await _open(db_session) == []
        closed = await get_particle(db_session, record.id)
        assert closed is not None
        assert (closed.status, closed.status_reason) == (
            Status.RETRACTED,
            StatusReason.CONFLICT_RESOLVED,
        )
        [entry] = report.closed
        assert entry["cause"] == "withdrawn"
        assert entry["withdrawn"][0]["withdrawn_reason"] == "verdict False"
        # No member is touched, and a mechanical close covers nothing.
        for claim in (a, b):
            stored = await get_particle(db_session, claim.id)
            assert stored is not None and stored.status is Status.ACTIVE
        assert frozenset((a.id, b.id)) not in await covered_pair_set(db_session)
        events = await list_events(db_session, event_type=OperatorEventType.INCONSISTENCY_CLOSED)
        assert [e.payload["cause"] for e in events] == ["withdrawn"]

    @pytest.mark.asyncio
    async def test_a_reconfirmed_pair_is_regrouped_once(self, db_session: Any) -> None:
        from unittest.mock import patch

        from particles.core.contradiction_disclosure import READING_OBSERVED, census_pairs

        a, b = await self._records(db_session)
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        [old] = await _open(db_session)
        reader = self._reader(True)

        with patch("particles.operations.lint.contradictions._llm_verify_contradiction", reader):
            report = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)
            again = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)

        [new] = await _open(db_session)
        assert census_replaces(new) == old.id
        assert [c["cause"] for c in report.closed] == ["regrouped"]
        [pair] = census_pairs(new)
        assert (pair.reading, pair.reason) == (READING_OBSERVED, "verdict True")
        assert report.opened == []  # a replacement is not a new disclosure
        # The replacement records the new reading, so the next night reads nothing.
        assert reader.await_count == 1
        assert again.reread.get("read", 0) == 0 and again.closed == []

    @pytest.mark.asyncio
    async def test_note_pairs_are_not_read_again(self, db_session: Any) -> None:
        from unittest.mock import patch

        a, b, _, _ = await _evidence(db_session)
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        reader = self._reader()

        with patch("particles.operations.lint.contradictions._llm_verify_contradiction", reader):
            report = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)

        reader.assert_not_awaited()
        assert report.closed == [] and len(await _open(db_session)) == 1

    @pytest.mark.asyncio
    async def test_no_rereading_without_the_census(self, db_session: Any) -> None:
        from unittest.mock import patch

        a, b = await self._records(db_session)
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        reader = self._reader()

        with patch("particles.operations.lint.contradictions._llm_verify_contradiction", reader):
            report = await run_disclosure(db_session, confirmed=[], mint=False)

        reader.assert_not_awaited()
        assert report.reread == {} and len(await _open(db_session)) == 1

    @pytest.mark.asyncio
    async def test_a_failed_or_capped_reading_keeps_the_record(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import patch

        a, b = await self._records(db_session)
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)

        with patch(
            "particles.operations.lint.contradictions._llm_verify_contradiction",
            self._reader(None),
        ):
            failed = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)
        assert failed.reread["failed"] == 1 and failed.closed == []

        monkeypatch.setattr(
            get_config().consolidation.contradiction_disclosure, "max_rereadings_per_run", 0
        )
        capped = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)
        assert capped.reread == {} and capped.closed == []
        assert len(await _open(db_session)) == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("bow")
class TestExtractTimeRecords:
    """Pass 3b reads the records extraction opened on the probe alone."""

    @staticmethod
    async def _record(session: Any) -> Any:
        """One unstamped record: a transcript re-extracted with the reading off."""
        from tests.test_second_reading import _transcript

        await _transcript(session, verify=False)
        [record] = await _extract_records(session)
        return record

    @staticmethod
    def _reads(monkeypatch: pytest.MonkeyPatch, verdict: bool | None) -> list[Any]:
        from particles.ingest import second_reading
        from particles.ingest.second_reading import ProbeVerdict

        seen: list[Any] = []

        async def read(a: Any, b: Any) -> ProbeVerdict | None:
            seen.append((a, b))
            if verdict is None:
                return None
            return ProbeVerdict(usable=True, contradicts=verdict, description=f"verdict {verdict}")

        monkeypatch.setattr(second_reading, "_llm_verify_contradiction", read)
        return seen

    async def test_not_confirmed_withdraws_and_mints_a_successor(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.test_update_supersession import BOSTON, DENVER, _by_content

        record = await self._record(db_session)
        seen = self._reads(monkeypatch, False)
        report = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)

        [(a, b)] = seen
        assert (a.content, b.content) == (BOSTON, DENVER)
        # The newcomer is read from its own snapshot, dated by it.
        assert b.observed != "unknown" and DENVER in b.passage
        assert report.extract_reread == {
            "read": 1,
            "confirmed": 0,
            "withdrawn": 1,
            "failed": 0,
            "deferred": 0,
        }
        closed = await get_particle(db_session, record.id)
        assert closed is not None
        assert (closed.status, closed.status_reason) == (
            Status.RETRACTED,
            StatusReason.CONFLICT_RESOLVED,
        )
        [boston] = await _by_content(db_session, BOSTON)
        assert boston.status is Status.ACTIVE
        denvers = {p.status: p for p in await _by_content(db_session, DENVER)}
        successor, quarantined = denvers[Status.ACTIVE], denvers[Status.SUPERSEDED]
        assert successor.supersedes == quarantined.id
        events = await list_events(db_session, event_type=OperatorEventType.INCONSISTENCY_CLOSED)
        [event] = events
        assert event.payload is not None
        assert event.payload["cause"] == "withdrawn"
        assert event.payload["origin"] == "extraction"
        assert event.payload["successor"] == successor.id
        assert [c["cause"] for c in report.closed] == ["withdrawn"]

    async def test_a_verbatim_active_duplicate_takes_the_newcomers_provenance(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.store.particle_store import get_particles_by_status, insert_particle
        from tests.test_update_supersession import DENVER, _by_content

        record = await self._record(db_session)
        [quarantined] = await _by_content(db_session, DENVER)
        # Re-extraction has since stated the claim verbatim, ACTIVE.
        twin = quarantined.model_copy(
            update={
                "id": "twin-" + quarantined.id[5:],
                "status": Status.ACTIVE,
                "status_reason": None,
                "provenance": [
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE,
                        corpus_entry_id="another-entry",
                        snapshot_id="another-snapshot",
                    )
                ],
            }
        )
        await insert_particle(db_session, twin)
        await db_session.commit()
        self._reads(monkeypatch, False)

        await run_disclosure(db_session, confirmed=[], mint=True, reread=True)

        active = [p for p in await get_particles_by_status(db_session, Status.ACTIVE)]
        assert [p.id for p in active if p.content == DENVER] == [twin.id]
        stored = await get_particle(db_session, twin.id)
        assert stored is not None
        assert {r.snapshot_id for r in stored.provenance} >= {
            r.snapshot_id for r in quarantined.provenance
        }
        folded = await get_particle(db_session, quarantined.id)
        assert folded is not None and folded.status is Status.SUPERSEDED
        [event] = await list_events(db_session, event_type=OperatorEventType.INCONSISTENCY_CLOSED)
        assert event.payload is not None
        assert event.payload["folded_into"] == twin.id
        assert (await get_particle(db_session, record.id)).status is Status.RETRACTED  # type: ignore[union-attr]

    async def test_confirmed_is_stamped_and_not_read_again(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.core.contradiction_disclosure import READING_KEY, READING_OBSERVED

        record = await self._record(db_session)
        seen = self._reads(monkeypatch, True)
        await run_disclosure(db_session, confirmed=[], mint=True, reread=True)
        again = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)

        assert len(seen) == 1
        assert again.extract_reread["read"] == 0
        stamped = await get_particle(db_session, record.id)
        assert stamped is not None and stamped.status is Status.INCONSISTENCY
        assert (stamped.properties or {})[READING_KEY] == READING_OBSERVED

    async def test_a_failed_reading_leaves_the_record_for_a_later_night(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.core.contradiction_disclosure import READING_KEY

        record = await self._record(db_session)
        self._reads(monkeypatch, None)
        report = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)
        assert report.extract_reread["failed"] == 1
        stored = await get_particle(db_session, record.id)
        assert stored is not None and stored.status is Status.INCONSISTENCY
        assert READING_KEY not in (stored.properties or {})

    async def test_a_retired_value_hold_is_never_read(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.core.conflict_review import is_retired_value
        from particles.core.schema import ProvenanceRefType as RefType
        from particles.ingest.conflict_plan import plan_retired_hold
        from particles.store.particle_store import insert_particle
        from tests.test_update_supersession import BOSTON, DENVER, _by_content

        await self._record(db_session)
        [boston] = await _by_content(db_session, BOSTON)
        retired = boston.model_copy(
            update={
                "id": "retired-" + boston.id[8:],
                "status": Status.RETRACTED,
                "status_reason": StatusReason.EXPLICIT_RETRACTION,
            }
        )
        await insert_particle(db_session, retired)
        [quarantined] = await _by_content(db_session, DENVER)
        plan = plan_retired_hold(
            retired,
            quarantined.model_copy(update={"id": "held-" + quarantined.id[5:]}),
            corpus_entry_id="e",
            snapshot_id="s",
            trigger_ref_type=RefType.SOURCE,
            domain=None,
        )
        assert plan.insert is not None and plan.wrapper is not None
        await insert_particle(db_session, plan.insert)
        await insert_particle(db_session, plan.wrapper)
        await db_session.commit()
        assert is_retired_value(plan.wrapper)
        # Retire claim A of the extract-time record too, so it is left to review.
        await update_particle_status(
            db_session, boston.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()
        seen = self._reads(monkeypatch, False)

        report = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)

        assert seen == [] and report.extract_reread["read"] == 0
        hold = await get_particle(db_session, plan.wrapper.id)
        assert hold is not None and hold.status is Status.INCONSISTENCY

    async def test_it_shares_the_budget_with_the_census_rereading(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import patch

        await self._record(db_session)
        a, b = await TestReread._records(db_session)
        await run_disclosure(db_session, confirmed=[_pair(a.id, b.id)], mint=True)
        monkeypatch.setattr(
            get_config().consolidation.contradiction_disclosure, "max_rereadings_per_run", 1
        )
        seen = self._reads(monkeypatch, False)

        with patch(
            "particles.operations.lint.contradictions._llm_verify_contradiction",
            TestReread._reader(True),
        ):
            report = await run_disclosure(db_session, confirmed=[], mint=True, reread=True)

        assert report.reread["read"] == 1
        assert seen == []
        assert report.extract_reread == {
            "read": 0,
            "confirmed": 0,
            "withdrawn": 0,
            "failed": 0,
            "deferred": 1,
        }


async def _extract_records(session: Any) -> list[Any]:
    """Open INCONSISTENCY records that are not census records."""
    from particles.core.contradiction_disclosure import is_census_record
    from particles.store.particle_store import get_inconsistency_particles

    return [r for r in await get_inconsistency_particles(session) if not is_census_record(r)]
