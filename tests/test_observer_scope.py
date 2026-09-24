# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer scope in the Engine: the scope join, ``rescope``, ``assign``, ``widen``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from particles.core.observer_scope import GLOBAL_SCOPE, LAPSED_SCOPE, BeliefScope
from particles.core.schema import (
    Mutability,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
)
from particles.core.status import Status, StatusReason
from particles.operations.observer_scope import assign_key, rescope, widen
from particles.operations.query.observer_scope import filter_visible, load_scopes
from particles.store.observer_scope_store import ScopeTarget
from particles.store.particle_store import append_provenance_ref, get_particle
from tests._observer_scope import belief, generation, project_source_tags, rescoped, source

A, B = "-proj-a", "-proj-b"


def keyed(*keys: str) -> BeliefScope:
    return BeliefScope(keys=frozenset(keys))


class TestLoadScopes:
    async def test_scope_is_the_union_over_every_source_not_the_first(
        self, db_session: Any
    ) -> None:
        in_a = await source(db_session, "a", project_source_tags(A))
        in_b = await source(db_session, "b", project_source_tags(B))
        both = await belief(db_session, "observed in both", in_a, in_b)
        only_b = await belief(db_session, "observed in b", in_b)

        scopes = await load_scopes(db_session, [both, only_b])

        assert scopes[both.id] == keyed(A, B)
        assert scopes[only_b.id] == keyed(B)

    async def test_a_hand_deposit_is_global_and_an_unstamped_harvest_is_not(
        self, db_session: Any
    ) -> None:
        page = await source(db_session, "page", ["web"])
        audit = await source(db_session, "audit", ["claude-code", "audit"])
        from_page = await belief(db_session, "from a web page", page)
        from_audit = await belief(db_session, "from an unstamped transcript", audit)

        scopes = await load_scopes(db_session, [from_page, from_audit])

        assert scopes[from_page.id] == GLOBAL_SCOPE
        assert scopes[from_audit.id].unattributed

    async def test_a_merged_copys_sources_reach_the_survivor(self, db_session: Any) -> None:
        """leaves the loser's refs on the loser; the walk is the only path to them."""
        from particles.store.particle_store import update_particle_status
        from particles.store.relation_store import create_relation

        survivor = await belief(
            db_session, "same claim", await source(db_session, "a", project_source_tags(A))
        )
        copy = await belief(
            db_session, "same claim", await source(db_session, "b", project_source_tags(B))
        )
        await update_particle_status(
            db_session, copy.id, Status.SUPERSEDED, StatusReason.DUPLICATE_MERGED
        )
        await create_relation(
            db_session,
            survivor.id,
            copy.id,
            RelationType.CO_EVIDENTIAL,
            RelationCreatedBy.EXACT_DUPLICATE,
        )

        scopes = await load_scopes(db_session, [survivor])

        assert scopes[survivor.id] == keyed(A, B)

    async def test_a_derived_belief_takes_the_meet_of_its_premises(self, db_session: Any) -> None:
        in_a = await source(db_session, "a", project_source_tags(A))
        in_b = await source(db_session, "b", project_source_tags(B))
        page = await source(db_session, "page", ["web"])
        a1 = await belief(db_session, "a one", in_a)
        a2 = await belief(db_session, "a two", in_a, in_b)
        b1 = await belief(db_session, "b one", in_b)
        world = await belief(db_session, "a public fact", page)

        within_a = await belief(db_session, "abstraction over a", premises=(a1.id, a2.id, world.id))
        across = await belief(
            db_session, "a contradiction across projects", in_a, premises=(a1.id, b1.id)
        )
        second_order = await belief(
            db_session, "abstraction of an abstraction", premises=(within_a.id,)
        )

        scopes = await load_scopes(db_session, [within_a, across, second_order])

        assert scopes[within_a.id] == keyed(A)
        # PARTICLE refs govern: the SOURCE ref beside them does not put it in A.
        assert scopes[across.id].unattributed
        assert scopes[second_order.id] == keyed(A)

    async def test_a_missing_premise_fails_closed(self, db_session: Any) -> None:
        orphan = await belief(db_session, "derived from nothing findable", premises=("no-such-id",))
        assert (await load_scopes(db_session, [orphan]))[orphan.id].unattributed


class TestCurrentlyStated:
    """A project observes a claim while one of its sources currently states it."""

    async def test_a_claim_the_latest_generation_does_not_name_is_out_of_that_projects_view(
        self, db_session: Any
    ) -> None:
        a1 = await generation(db_session, "a", "v1", project_source_tags(A))
        b1 = await generation(db_session, "b", "v1", project_source_tags(B))
        both = await belief(db_session, "stated by both, then dropped by a", a1, b1)
        await generation(db_session, "a", "v2", project_source_tags(A))  # a moved on

        assert (await load_scopes(db_session, [both]))[both.id] == keyed(B)

    async def test_a_re_observation_keeps_the_claim_current(self, db_session: Any) -> None:
        a1 = await generation(db_session, "a", "v1", project_source_tags(A))
        restated = await belief(db_session, "carried into v2", a1)
        a2 = await generation(db_session, "a", "v2", project_source_tags(A))
        # How carry-forward and suppression record a re-observation.
        await append_provenance_ref(
            db_session,
            restated.id,
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=a2[0], snapshot_id=a2[1]),
        )
        restated = await get_particle(db_session, restated.id)
        assert restated is not None

        assert (await load_scopes(db_session, [restated]))[restated.id] == keyed(A)

    async def test_a_claim_no_source_states_any_more_is_lapsed_not_unattributed(
        self, db_session: Any
    ) -> None:
        a1 = await generation(db_session, "a", "v1", project_source_tags(A))
        gone = await belief(db_session, "dropped by its only source", a1)
        await generation(db_session, "a", "v2", project_source_tags(A))
        await rescoped(db_session)

        scope = (await load_scopes(db_session, [gone]))[gone.id]
        assert scope == LAPSED_SCOPE and not scope.unattributed
        filtered = await filter_visible(db_session, [gone], A)
        assert filtered.visible_ids == frozenset()
        assert (filtered.lapsed, filtered.unattributed) == (1, 0)
        assert filtered.note(A).lapsed == 1

    async def test_every_snapshot_of_a_non_mutable_source_counts(self, db_session: Any) -> None:
        t1 = await generation(
            db_session, "t", "turn 1", project_source_tags(A), mutability=Mutability.APPEND_ONLY
        )
        await generation(
            db_session, "t", "turn 1, 2", project_source_tags(A), mutability=Mutability.APPEND_ONLY
        )
        said = await belief(db_session, "said in turn 1", t1)

        assert (await load_scopes(db_session, [said]))[said.id] == keyed(A)

    async def test_an_as_of_read_takes_the_generation_current_then(self, db_session: Any) -> None:
        then = datetime.now(UTC) - timedelta(days=10)
        a1 = await generation(
            db_session, "a", "v1", project_source_tags(A), captured_at=then - timedelta(days=1)
        )
        dropped = await belief(db_session, "dropped by v2", a1)
        await generation(db_session, "a", "v2", project_source_tags(A))
        await rescoped(db_session)

        assert (await filter_visible(db_session, [dropped], A)).visible_ids == frozenset()
        at_then = await filter_visible(db_session, [dropped], A, as_of=then)
        assert at_then.visible_ids == {dropped.id}

    async def test_a_derived_belief_over_a_lapsed_premise_is_lapsed(self, db_session: Any) -> None:
        a1 = await generation(db_session, "a", "v1", project_source_tags(A))
        premise = await belief(db_session, "premise", a1)
        await generation(db_session, "a", "v2", project_source_tags(A))
        derived = await belief(db_session, "derived", premises=(premise.id,))

        assert (await load_scopes(db_session, [derived]))[derived.id].lapsed


class TestEntryCurrency:
    async def test_latest_is_the_newest_extracted_generation_only(self, db_session: Any) -> None:
        from particles.corpus.store import get_entry_currency

        entry, first = await generation(db_session, "a", "v1", project_source_tags(A))
        await source(db_session, "a", project_source_tags(A))  # a newer, unextracted generation
        stable, _ = await generation(
            db_session, "s", "the handbook", ["handbook"], mutability=Mutability.STABLE
        )

        currency = await get_entry_currency(db_session, [entry, stable, "not-an-entry"])

        assert set(currency) == {entry, stable}
        assert currency[entry].mutable and currency[entry].latest_snapshot_id == first
        assert currency[stable].tags == ["handbook"]
        assert not currency[stable].mutable and currency[stable].latest_snapshot_id is None


class TestFilterVisible:
    async def test_no_observer_does_no_work_and_hides_nothing(self, db_session: Any) -> None:
        b = await belief(
            db_session, "b only", await source(db_session, "b", project_source_tags(B))
        )
        result = await filter_visible(db_session, [b], None)
        assert result.visible_ids == {b.id} and not result.engaged

    async def test_an_observer_is_not_honoured_before_the_store_is_rescoped(
        self, db_session: Any
    ) -> None:
        b = await belief(
            db_session, "b only", await source(db_session, "b", project_source_tags(B))
        )

        before = await filter_visible(db_session, [b], A)
        await rescoped(db_session)
        after = await filter_visible(db_session, [b], A)

        assert before.visible_ids == {b.id} and not before.engaged
        assert after.visible_ids == frozenset() and after.engaged

    async def test_a_project_sees_global_and_its_own_and_counts_the_unattributed(
        self, db_session: Any
    ) -> None:
        mine = await belief(
            db_session, "mine", await source(db_session, "a", project_source_tags(A))
        )
        theirs = await belief(
            db_session, "theirs", await source(db_session, "b", project_source_tags(B))
        )
        world = await belief(db_session, "world", await source(db_session, "page", ["web"]))
        stray = await belief(db_session, "stray", await source(db_session, "t", ["claude-code"]))
        await rescoped(db_session)

        result = await filter_visible(db_session, [mine, theirs, world, stray], A)

        assert result.visible_ids == {mine.id, world.id}
        assert (result.total, result.in_scope, result.unattributed) == (4, 2, 1)


class TestWiden:
    async def test_widening_a_belief_and_taking_it_back(self, db_session: Any) -> None:
        theirs = await belief(
            db_session, "theirs", await source(db_session, "b", project_source_tags(B))
        )
        await rescoped(db_session)

        assert await widen(db_session, ScopeTarget.PARTICLE, theirs.id, actor="op")
        assert not await widen(
            db_session, ScopeTarget.PARTICLE, theirs.id, actor="op"
        )  # idempotent
        assert (await filter_visible(db_session, [theirs], A)).visible_ids == {theirs.id}

        assert await widen(db_session, ScopeTarget.PARTICLE, theirs.id, actor="op", revoke=True)
        assert (await filter_visible(db_session, [theirs], A)).visible_ids == frozenset()

    async def test_widening_a_source_widens_every_belief_from_it(self, db_session: Any) -> None:
        entry = await source(db_session, "b", project_source_tags(B))
        one = await belief(db_session, "one", entry)
        two = await belief(db_session, "two", entry)
        await rescoped(db_session)

        await widen(db_session, ScopeTarget.CORPUS_ENTRY, entry[0], actor="op")

        assert (await filter_visible(db_session, [one, two], A)).visible_ids == {one.id, two.id}

    async def test_an_unknown_target_is_refused(self, db_session: Any) -> None:
        with pytest.raises(ValueError, match="No particle"):
            await widen(db_session, ScopeTarget.PARTICLE, "nope", actor="op")


class TestRescope:
    @staticmethod
    def _by_uri(mapping: dict[str, str]) -> Any:
        return lambda _entry_id, uri, _tags: mapping.get(uri or "")

    async def test_adds_the_canonical_key_beside_the_legacy_one_and_is_idempotent(
        self, db_session: Any
    ) -> None:
        from particles.corpus.store import get_entry

        legacy = await source(
            db_session, "wt", ["claude-code", "project:-repo--claude-worktrees-x"]
        )
        key_for = self._by_uri({"test://wt": "-repo"})

        first = await rescope(db_session, key_for=key_for)
        second = await rescope(db_session, key_for=key_for)

        entry = await get_entry(db_session, legacy[0])
        assert entry is not None
        assert entry.tags == ["claude-code", "project:-repo--claude-worktrees-x", "project:-repo"]
        assert first.added == [(legacy[0], "-repo")] and second.added == []

    async def test_dry_run_writes_nothing_not_even_the_marker(self, db_session: Any) -> None:
        from particles.corpus.store import get_entry
        from particles.operations.query.observer_scope import lens_may_engage

        entry_id, _ = await source(db_session, "t", ["claude-code"])

        report = await rescope(db_session, key_for=self._by_uri({"test://t": A}), dry_run=True)

        assert report.added == [(entry_id, A)]
        entry = await get_entry(db_session, entry_id)
        assert entry is not None and entry.tags == ["claude-code"]
        assert not await lens_may_engage(db_session)

    async def test_reports_what_stays_unattributed_and_a_default_key_resolves_it(
        self, db_session: Any
    ) -> None:
        unknown, _ = await source(db_session, "gone", ["claude-code", "audit"])
        await source(db_session, "page", ["web"])  # global: never touched, never counted

        report = await rescope(db_session, key_for=self._by_uri({}))
        assert report.unattributed == [unknown]

        report = await rescope(db_session, key_for=self._by_uri({}), default_key=A)
        assert report.unattributed == [] and report.added == [(unknown, A)]
        assert report.entries_per_key == {A: 1}

    async def test_assign_refuses_a_global_entry(self, db_session: Any) -> None:
        page, _ = await source(db_session, "page", ["web"])
        harvested, _ = await source(db_session, "t", ["claude-code"])

        assert await assign_key(db_session, harvested, A, actor="op")
        assert not await assign_key(db_session, harvested, A, actor="op")
        # Attributing one source is not a rescope: it must not switch the lens on.
        from particles.operations.query.observer_scope import lens_may_engage

        assert not await lens_may_engage(db_session)
        with pytest.raises(ValueError, match="is global"):
            await assign_key(db_session, page, A, actor="op")
