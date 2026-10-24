# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The observer-scope join's store half, pinned across the refactor.

``TestLoadScopesPinned`` was written and run green against the module before
its attesting-entries computation was factored out, and is kept unchanged
through the refactor: it pins the old behaviour, not the new code.
"""

from __future__ import annotations

from typing import Any

from particles.core.observer_scope import GLOBAL_SCOPE, LAPSED_SCOPE, BeliefScope
from particles.core.schema import Mutability, RelationCreatedBy, RelationType
from particles.core.status import Status, StatusReason
from particles.operations.observer_scope import widen
from particles.store.observer_scope_join import lens_may_engage, load_scopes
from particles.store.observer_scope_store import ScopeTarget
from tests._observer_scope import belief, generation, project_source_tags, rescoped, source

A, B = "-proj-a", "-proj-b"


def keyed(*keys: str) -> BeliefScope:
    return BeliefScope(keys=frozenset(keys))


class TestLoadScopesPinned:
    async def test_mutable_entry_attests_only_through_its_latest_generation(
        self, db_session: Any
    ) -> None:
        a1 = await generation(db_session, "a", "v1", project_source_tags(A))
        b1 = await generation(db_session, "b", "v1", project_source_tags(B))
        named_by_old = await belief(db_session, "named by a's old generation", a1, b1)
        a2 = await generation(db_session, "a", "v2", project_source_tags(A))
        named_by_new = await belief(db_session, "named by a's new generation", a2)
        only_old = await belief(db_session, "only a's old generation", a1)

        scopes = await load_scopes(db_session, [named_by_old, named_by_new, only_old])

        assert scopes[named_by_old.id] == keyed(B)
        assert scopes[named_by_new.id] == keyed(A)
        assert scopes[only_old.id] == LAPSED_SCOPE

    async def test_every_snapshot_reading_counts_the_old_generation(self, db_session: Any) -> None:
        a1 = await generation(db_session, "a", "v1", project_source_tags(A))
        await generation(db_session, "a", "v2", project_source_tags(A))
        only_old = await belief(db_session, "only a's old generation", a1)

        scopes = await load_scopes(db_session, [only_old], current_only=False)

        assert scopes[only_old.id] == keyed(A)

    async def test_append_only_entry_attests_through_an_older_snapshot(
        self, db_session: Any
    ) -> None:
        t1 = await generation(
            db_session, "t", "turn 1", project_source_tags(A), mutability=Mutability.APPEND_ONLY
        )
        await generation(
            db_session, "t", "turn 1, 2", project_source_tags(A), mutability=Mutability.APPEND_ONLY
        )
        said = await belief(db_session, "said in turn 1", t1)

        assert (await load_scopes(db_session, [said]))[said.id] == keyed(A)

    async def test_co_evidential_group_pools_every_members_entries(self, db_session: Any) -> None:
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

    async def test_derived_belief_takes_the_meet_and_missing_premises_fail_closed(
        self, db_session: Any
    ) -> None:
        in_a = await source(db_session, "a", project_source_tags(A))
        in_b = await source(db_session, "b", project_source_tags(B))
        a1 = await belief(db_session, "a one", in_a)
        ab = await belief(db_session, "a and b", in_a, in_b)
        derived = await belief(db_session, "over a", premises=(a1.id, ab.id))
        second = await belief(db_session, "over the abstraction", premises=(derived.id,))
        orphan = await belief(db_session, "over nothing", premises=("no-such-id",))

        scopes = await load_scopes(db_session, [derived, second, orphan])

        assert scopes[derived.id] == keyed(A)
        assert scopes[second.id] == keyed(A)
        assert scopes[orphan.id].unattributed

    async def test_a_widened_source_makes_its_beliefs_global(self, db_session: Any) -> None:
        in_a = await source(db_session, "a", project_source_tags(A))
        held = await belief(db_session, "held in a", in_a)
        await widen(db_session, ScopeTarget.CORPUS_ENTRY, in_a[0], actor="test")

        scopes = await load_scopes(db_session, [held])

        assert scopes[held.id] == GLOBAL_SCOPE

    async def test_the_lens_engages_only_after_a_rescope(self, db_session: Any) -> None:
        assert not await lens_may_engage(db_session)
        await rescoped(db_session)
        assert await lens_may_engage(db_session)
