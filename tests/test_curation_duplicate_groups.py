# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""One curation card per connected group of duplicate candidates.

The pure grouping and regrouping, the partition rulings, the election, and the
three gestures (merge, split, dismiss) with their refusals.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    Confidence,
    JudgeVerdictKind,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.operations.curation import apply_gesture
from particles.operations.curation.cards import (
    CardKind,
    CurationCard,
    DuplicateEdge,
    DuplicateVerdict,
    describe_gesture,
)
from particles.operations.curation.collect import duplicate_group_cards, election_ranks
from particles.operations.curation.duplicate_groups import (
    GroupRulings,
    all_distinct,
    brief_ids,
    election_key,
    group_pairs,
    regroup,
    resolve_members,
    split_card,
)
from particles.operations.curation.precision import classify_event
from particles.operations.curation.suppression import group_rulings, suppressed_keys
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import insert_particle
from particles.store.relation_store import create_relation, get_co_evidential_group

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _edge(
    a: str, b: str, sim: float = 0.95, verdict: JudgeVerdictKind | None = None
) -> DuplicateEdge:
    return DuplicateEdge(
        a=a,
        b=b,
        similarity=sim,
        subject_id="s1",
        verdict=DuplicateVerdict(verdict=verdict) if verdict is not None else None,
    )


def _rank(*ids: str) -> dict[str, tuple[bool, datetime | None, datetime | None, str]]:
    """Election keys in the order given: the first id is elected."""
    return {
        pid: election_key(
            pid, subject_linked=True, first_read=_T0 + timedelta(days=i), asserted_at=_T0
        )
        for i, pid in enumerate(ids)
    }


def _group(edges: list[DuplicateEdge], *, rank: dict | None = None, linked=None, rulings=None):  # type: ignore[no-untyped-def]
    ids = sorted({p for e in edges for p in (e.a, e.b)})
    return group_pairs(
        edges,
        linked=linked or {},
        withheld=(rulings or GroupRulings()).withholds,
        rank=rank or _rank(*ids),
        rulings=rulings or GroupRulings(),
    )


class TestGroupPairs:
    def test_a_chain_is_one_card_keyed_by_its_survivor(self) -> None:
        cards = _group(
            [_edge("a", "b"), _edge("b", "c"), _edge("x", "y")], rank=_rank("c", "a", "b", "x", "y")
        )
        assert [len(c.particle_ids) for c in cards] == [3, 2]
        chain = next(c for c in cards if "c" in c.particle_ids)
        assert chain.kind is CardKind.DUPLICATE_GROUP
        assert chain.survivor_id == "c"
        assert chain.key == "duplicate_group:c"
        assert chain.particle_ids[0] == "c"
        assert len(chain.pairs or []) == 2
        assert chain.suggested_gestures == ["merge", "split", "dismiss", "snooze"]

    def test_members_follow_the_survivor_weakest_linked_first(self) -> None:
        cards = _group([_edge("a", "b", 0.99), _edge("a", "c", 0.93)], rank=_rank("a", "b", "c"))
        assert cards[0].particle_ids == ["a", "c", "b"]

    def test_a_withheld_pair_splits_the_component(self) -> None:
        rulings = GroupRulings(((frozenset({"a"}), frozenset({"b"})),))
        cards = _group([_edge("a", "b"), _edge("b", "c")], rulings=rulings)
        assert [c.particle_ids for c in cards] == [["b", "c"]]

    def test_a_linked_set_is_one_member(self) -> None:
        linked = {"a": frozenset({"a", "b"}), "b": frozenset({"a", "b"})}
        # The a–b pair is inside one node; c pairs with both.
        cards = _group([_edge("a", "b"), _edge("a", "c"), _edge("b", "c")], linked=linked)
        assert len(cards) == 1
        assert cards[0].linked_sets == [["a", "b"]]
        assert set(cards[0].particle_ids) == {"a", "b", "c"}
        # A group whose members are all linked is no question at all.
        assert _group([_edge("a", "b")], linked=linked) == []

    def test_a_bridged_cut_is_shown_on_the_card(self) -> None:
        rulings = GroupRulings(((frozenset({"a"}), frozenset({"b"})),))
        cards = _group([_edge("a", "n"), _edge("n", "b")], rulings=rulings)
        assert cards[0].prior_rulings == [[["a"], ["b"]]]

    def test_a_one_pair_group_carries_its_verdict(self) -> None:
        (card,) = _group([_edge("a", "b", verdict=JudgeVerdictKind.DISTINCT)])
        assert card.verdict is not None and card.verdict.verdict is JudgeVerdictKind.DISTINCT
        assert all_distinct(card)
        (mixed,) = _group([_edge("a", "b", verdict=JudgeVerdictKind.DISTINCT), _edge("b", "c")])
        assert mixed.verdict is None
        assert not all_distinct(mixed)


class TestRulings:
    def test_a_pair_is_withheld_only_across_classes_of_one_ruling(self) -> None:
        rulings = GroupRulings(((frozenset({"a", "b"}), frozenset({"c"})),))
        assert rulings.withholds("a", "c")
        assert rulings.withholds("c", "b")
        assert not rulings.withholds("a", "b")
        assert not rulings.withholds("a", "z")


class TestRegroup:
    def _card(self) -> CurationCard:
        (card,) = _group(
            [_edge("a", "b"), _edge("b", "c"), _edge("c", "d")], rank=_rank("a", "b", "c", "d")
        )
        return card

    def test_nothing_changed_is_the_same_card(self) -> None:
        card = self._card()
        assert regroup(card, active=None, linked={}, rulings=GroupRulings()) is card

    def test_a_retired_member_leaves_and_may_cut_the_chain(self) -> None:
        out = regroup(self._card(), active={"a", "b", "d"}, linked={}, rulings=GroupRulings())
        assert out is not None
        assert out.particle_ids == ["a", "b"]
        assert out.key == "duplicate_group:a"

    def test_a_split_keeps_the_survivors_side(self) -> None:
        rulings = GroupRulings(((frozenset({"c", "d"}), frozenset({"a", "b"})),))
        out = regroup(self._card(), active=None, linked={}, rulings=rulings)
        assert out is not None and out.particle_ids == ["a", "b"]

    def test_a_merge_of_everything_drops_the_card(self) -> None:
        everyone = frozenset({"a", "b", "c", "d"})
        linked = {pid: everyone for pid in everyone}
        assert regroup(self._card(), active=None, linked=linked, rulings=GroupRulings()) is None

    def test_a_gone_survivor_hands_the_key_to_the_election(self) -> None:
        out = regroup(
            self._card(),
            active={"b", "c", "d"},
            linked={},
            rulings=GroupRulings(),
            rank=_rank("d", "c", "b"),
        )
        assert out is not None
        assert out.survivor_id == "d"
        assert out.key == "duplicate_group:d"

    def test_the_input_is_not_mutated(self) -> None:
        card = self._card()
        before = list(card.particle_ids)
        regroup(card, active={"a", "b"}, linked={}, rulings=GroupRulings())
        assert card.particle_ids == before


class TestMembers:
    def _card(self, n: int) -> CurationCard:
        ids = [f"{i:02d}-member" for i in range(n)]
        (card,) = _group([_edge(ids[0], other) for other in ids[1:]], rank=_rank(*ids))
        return card

    def test_prefixes_resolve_and_linked_sets_come_whole(self) -> None:
        linked = {
            "01-member": frozenset({"01-member", "02-member"}),
            "02-member": frozenset({"01-member", "02-member"}),
        }
        ids = ["00-member", "01-member", "02-member", "03-member"]
        (card,) = _group([_edge("00-member", x) for x in ids[1:]], rank=_rank(*ids), linked=linked)
        assert resolve_members(card, ["00", "01"]) == ["00-member", "01-member", "02-member"]

    def test_all_is_limited_to_ten(self) -> None:
        assert len(resolve_members(self._card(10), ["all"])) == 10
        with pytest.raises(ValueError, match="group of 10 or fewer"):
            resolve_members(self._card(11), ["all"])

    def test_a_value_naming_no_single_member_is_refused(self) -> None:
        with pytest.raises(ValueError, match="matches no member"):
            resolve_members(self._card(3), ["zz"])
        with pytest.raises(ValueError, match="more than one member"):
            resolve_members(self._card(3), ["0"])
        with pytest.raises(ValueError, match="Name the members"):
            resolve_members(self._card(3), [])

    def test_the_slice_briefs_a_small_group_whole_and_samples_a_large_one(self) -> None:
        assert len(brief_ids(self._card(10), full=False)) == 10
        assert len(brief_ids(self._card(12), full=False)) == 6
        assert len(brief_ids(self._card(12), full=True)) == 12


def test_the_audit_split_regroups_what_is_left() -> None:
    (card,) = _group([_edge("a", "b"), _edge("b", "c")], rank=_rank("a", "b", "c"))
    parts = split_card(card, lambda e: {e.a, e.b} != {"a", "b"})
    assert [p.particle_ids for p in parts] == [["b", "c"]]


def test_the_merge_help_says_an_answer_shows_one_member() -> None:
    card = CurationCard.from_key("duplicate_group:abc")
    assert card.kind is CardKind.DUPLICATE_GROUP and card.survivor_id == "abc"
    text = describe_gesture(card, "merge", snooze_days=14)
    assert "every belief stays active" in text
    assert "split off one that states less" in text


# --- store-level -------------------------------------------------------------


def _belief(
    content: str, *, subjects: list[str] | None = None, asserted: datetime = _T0
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id=None)
        ],
        asserted_by="general-extractor",
        subject_ids=["sid"] if subjects is None else subjects,
        asserted_at=asserted,
    )


async def _store(session: AsyncSession, *particles: Particle) -> list[Particle]:
    for p in particles:
        await insert_particle(session, p)
    await session.flush()
    return list(particles)


async def _served(session: AsyncSession, edges: list[DuplicateEdge]) -> CurationCard:
    (card,) = await duplicate_group_cards(session, edges)
    return card


class TestElection:
    @pytest.mark.asyncio
    async def test_subject_linked_first_then_the_older(self, db_session: AsyncSession) -> None:
        old_orphan, newer, older = await _store(
            db_session,
            _belief("The repo is at /src.", subjects=[], asserted=_T0),
            _belief("The repository is at /src.", asserted=_T0 + timedelta(days=2)),
            _belief("The repo lives at /src.", asserted=_T0 + timedelta(days=1)),
        )
        ranks = await election_ranks(db_session, [old_orphan.id, newer.id, older.id])
        assert min(ranks, key=lambda pid: ranks[pid][2] or _T0) == old_orphan.id
        card = await _served(
            db_session, [_edge(old_orphan.id, newer.id), _edge(newer.id, older.id)]
        )
        assert card.survivor_id == older.id

    @pytest.mark.asyncio
    async def test_an_old_pair_dismissal_still_withholds_its_pair(
        self, db_session: AsyncSession
    ) -> None:
        a, b, c = await _store(db_session, _belief("A x."), _belief("A y."), _belief("A z."))
        card = CurationCard.from_key(f"duplicate_pair:{'|'.join(sorted([a.id, b.id]))}")
        await apply_gesture(db_session, card, "dismiss")
        cards = await duplicate_group_cards(db_session, [_edge(a.id, b.id), _edge(b.id, c.id)])
        assert [sorted(x.particle_ids) for x in cards] == [sorted([b.id, c.id])]


class TestGestures:
    @pytest.mark.asyncio
    async def test_merge_all_links_every_member_to_the_hub(self, db_session: AsyncSession) -> None:
        a, b, c = await _store(
            db_session,
            _belief("Deploys run Fridays."),
            _belief("Deploys are Fridays."),
            _belief("We deploy Fridays."),
        )
        card = await _served(db_session, [_edge(a.id, b.id), _edge(b.id, c.id)])
        with patch(
            "particles.operations.curation.session.get_curation_card", AsyncMock(return_value=card)
        ):
            line = await apply_gesture(
                db_session, CurationCard.from_key(card.key), "merge", members=["all"]
            )
        assert "every belief stays active" in line
        group = await get_co_evidential_group(db_session, a.id)
        assert group == {a.id, b.id, c.id}
        events = await list_events(db_session, event_type=OperatorEventType.RELATION_ADDED)
        assert len(events) == 2
        # The served card regroups away: every member is now one linked set.
        state = await duplicate_group_cards(db_session, [_edge(a.id, b.id), _edge(b.id, c.id)])
        assert state == []

    @pytest.mark.asyncio
    async def test_merge_with_keep_uses_that_hub_and_leaves_the_rest(
        self, db_session: AsyncSession
    ) -> None:
        a, b, c = await _store(
            db_session, _belief("X one."), _belief("X two."), _belief("X three.")
        )
        card = await _served(db_session, [_edge(a.id, b.id), _edge(b.id, c.id)])
        with patch(
            "particles.operations.curation.session.get_curation_card", AsyncMock(return_value=card)
        ):
            line = await apply_gesture(
                db_session,
                CurationCard.from_key(card.key),
                "merge",
                members=[b.id, c.id],
                keep=c.id,
            )
        assert "1 member(s) stay on the card" in line
        assert await get_co_evidential_group(db_session, c.id) == {b.id, c.id}
        (left,) = await duplicate_group_cards(db_session, [_edge(a.id, b.id), _edge(b.id, c.id)])
        assert left.linked_sets == [sorted([b.id, c.id])]

    @pytest.mark.asyncio
    async def test_merge_refuses_a_pair_a_split_withholds(self, db_session: AsyncSession) -> None:
        a, b, n = await _store(
            db_session, _belief("Y one."), _belief("Y two."), _belief("Y three.")
        )
        first = await _served(db_session, [_edge(a.id, b.id)])
        with patch(
            "particles.operations.curation.session.get_curation_card", AsyncMock(return_value=first)
        ):
            await apply_gesture(
                db_session, CurationCard.from_key(first.key), "split", members=[b.id]
            )
        # A later claim bridges the two sides into one group again.
        bridged = await _served(db_session, [_edge(a.id, n.id), _edge(n.id, b.id)])
        assert bridged.prior_rulings is not None
        with (
            patch(
                "particles.operations.curation.session.get_curation_card",
                AsyncMock(return_value=bridged),
            ),
            pytest.raises(ValueError, match="ruled not the same claim"),
        ):
            await apply_gesture(
                db_session, CurationCard.from_key(bridged.key), "merge", members=["all"]
            )
        assert await get_co_evidential_group(db_session, a.id) == {a.id}

    @pytest.mark.asyncio
    async def test_merge_refuses_a_recorded_contradiction(self, db_session: AsyncSession) -> None:
        a, b = await _store(
            db_session, _belief("The header is required."), _belief("The header is not required.")
        )
        await create_relation(
            db_session, a.id, b.id, RelationType.CONTRADICTS, RelationCreatedBy.MANUAL_CLI
        )
        card = await _served(db_session, [_edge(a.id, b.id)])
        with (
            patch(
                "particles.operations.curation.session.get_curation_card",
                AsyncMock(return_value=card),
            ),
            pytest.raises(ValueError, match="recorded as contradicting"),
        ):
            await apply_gesture(
                db_session, CurationCard.from_key(card.key), "merge", members=["all"]
            )

    @pytest.mark.asyncio
    async def test_a_member_outside_the_group_is_refused(self, db_session: AsyncSession) -> None:
        a, b, other = await _store(
            db_session, _belief("Z one."), _belief("Z two."), _belief("Z three.")
        )
        card = await _served(db_session, [_edge(a.id, b.id)])
        with (
            patch(
                "particles.operations.curation.session.get_curation_card",
                AsyncMock(return_value=card),
            ),
            pytest.raises(ValueError, match="matches no member"),
        ):
            await apply_gesture(
                db_session, CurationCard.from_key(card.key), "merge", members=[a.id, other.id]
            )

    @pytest.mark.asyncio
    async def test_split_and_dismiss_record_partitions(self, db_session: AsyncSession) -> None:
        a, b, c = await _store(
            db_session, _belief("W one."), _belief("W two."), _belief("W three.")
        )
        card = await _served(db_session, [_edge(a.id, b.id), _edge(b.id, c.id), _edge(a.id, c.id)])
        with patch(
            "particles.operations.curation.session.get_curation_card", AsyncMock(return_value=card)
        ):
            await apply_gesture(
                db_session, CurationCard.from_key(card.key), "split", members=[c.id]
            )
        rulings = await group_rulings(db_session)
        assert rulings.withholds(a.id, c.id) and rulings.withholds(b.id, c.id)
        assert not rulings.withholds(a.id, b.id)
        (event,) = await list_events(db_session, event_type=OperatorEventType.CURATION_CARD_SNOOZED)
        assert event.payload["ruling"] == "split"
        assert len(event.payload["pairs"]) == 2
        # A group ruling hides no key: a later claim may still join one side.
        assert card.key not in await suppressed_keys(db_session)
        # The precision ledger counts the card once; the withheld pairs stay in
        # the event payload as labelled data, not as cards.
        labels = classify_event(event, membership={}, deposited_urls=set())
        assert [k for k, _ in labels] == [card.key]

        left = await _served(db_session, [_edge(a.id, b.id), _edge(b.id, c.id), _edge(a.id, c.id)])
        with patch(
            "particles.operations.curation.session.get_curation_card", AsyncMock(return_value=left)
        ):
            await apply_gesture(db_session, CurationCard.from_key(left.key), "dismiss")
        assert (await group_rulings(db_session)).withholds(a.id, b.id)
        assert await duplicate_group_cards(db_session, [_edge(a.id, b.id), _edge(b.id, c.id)]) == []

    @pytest.mark.asyncio
    async def test_members_on_another_kind_are_refused(self, db_session: AsyncSession) -> None:
        card = CurationCard.from_key("stale:x")
        with pytest.raises(ValueError, match="only to a duplicate_group"):
            await apply_gesture(db_session, card, "snooze", members=["x"])


@pytest.mark.asyncio
async def test_a_classless_group_dismissal_hides_the_key(db_session: AsyncSession) -> None:
    """A plain permanent snooze on a group key (an older client) is never a no-op."""
    from particles.store.event_store import EventRefKind, record_event

    await record_event(
        db_session,
        actor="http:/curation/snooze",
        event_type=OperatorEventType.CURATION_CARD_SNOOZED,
        refs=[(EventRefKind.PARTICLE, "p-a")],
        payload={"card_key": "duplicate_group:p-a", "snoozed_until": None, "snooze_days": None},
    )
    assert "duplicate_group:p-a" in await suppressed_keys(db_session)
    assert (await group_rulings(db_session)).rulings == ()


@pytest.mark.asyncio
async def test_an_old_pair_dismissal_after_the_build_applies_to_the_served_group(
    db_session: AsyncSession,
) -> None:
    from particles.operations.curation.duplicate_groups import pair_withheld

    a, b, c = await _store(db_session, _belief("V one."), _belief("V two."), _belief("V three."))
    card = await _served(db_session, [_edge(a.id, b.id), _edge(b.id, c.id)])
    await apply_gesture(
        db_session,
        CurationCard.from_key(f"duplicate_pair:{'|'.join(sorted([b.id, c.id]))}"),
        "dismiss",
    )
    rulings = await group_rulings(db_session)
    out = regroup(
        card,
        active=None,
        linked={},
        rulings=rulings,
        withheld=pair_withheld(await suppressed_keys(db_session), rulings),
    )
    assert out is not None and sorted(out.particle_ids) == sorted([a.id, b.id])
