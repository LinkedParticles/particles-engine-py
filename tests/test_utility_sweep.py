# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The rank sweeps read through a project observer (gate B criterion 3)."""

from __future__ import annotations

from typing import Any

from particles.operations.utility_sweep import (
    head_population,
    sweep_store_owner_rank_lift,
    sweep_store_rank_lift,
)
from tests._observer_scope import belief, project_source_tags, rescoped, source

A, B = "-proj-a", "-proj-b"


async def _two_projects(session: Any) -> None:
    await belief(session, "A deploys Fridays.", await source(session, "a", project_source_tags(A)))
    await belief(session, "B deploys Mondays.", await source(session, "b", project_source_tags(B)))
    await belief(session, "Pluto has five moons.", await source(session, "w", ["web"]))


async def test_a_project_head_ranks_its_own_and_the_global_beliefs(db_session: Any) -> None:
    await _two_projects(db_session)
    await rescoped(db_session)

    population, note = await head_population(db_session, A)

    assert population is not None and note is not None and note.engaged
    assert sorted(p.content for p in population) == ["A deploys Fridays.", "Pluto has five moons."]
    rank = await sweep_store_rank_lift(db_session, head_sizes=[2], particles=population)
    owner = await sweep_store_owner_rank_lift(db_session, head_sizes=[2], particles=population)
    assert rank.scored == 2 and owner.scored == 2
    assert (await sweep_store_rank_lift(db_session, head_sizes=[2])).scored == 3


async def test_no_project_is_the_store_wide_sweep(db_session: Any) -> None:
    assert await head_population(db_session, None) == (None, None)


async def test_an_unrescoped_store_says_the_head_stayed_store_wide(db_session: Any) -> None:
    await _two_projects(db_session)

    population, note = await head_population(db_session, A)

    assert note is not None and not note.engaged
    assert population is not None and len(population) == 3
