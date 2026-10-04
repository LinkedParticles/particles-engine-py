# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reindex of an append-only entry: retire the entry's claims, replay its snapshots.

The per-snapshot reindex does not work on an ``APPEND_ONLY`` entry read as a
delta: a delta of the latest snapshot would replace the entry's claims with its
tail's, and a whole read of it would duplicate them. These tests drive
``operations.reindex`` over a transcript deposited and extracted in three
snapshots, with the extraction LLM mocked as in ``test_append_base.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.status import Status, StatusReason
from particles.operations.reindex import _group_append_only, reindex
from particles.store.particle_store import get_particles_for_entry
from tests.test_append_base import (
    FIRST,
    SECOND,
    THIRD,
    FakeLLM,
    _deposit,
    _extract,
    _extracted_through,
    fake_extraction_llm,
)


@pytest.fixture
def llm() -> Iterator[FakeLLM]:
    with fake_extraction_llm() as fake:
        yield fake


async def _three_snapshots(session: AsyncSession) -> tuple[str, list[str]]:
    entry_id, s1 = await _deposit(session, FIRST)
    await _extract(session, entry_id, s1)
    _, s2 = await _deposit(session, SECOND)
    await _extract(session, entry_id, s2)
    _, s3 = await _deposit(session, THIRD)
    await _extract(session, entry_id, s3)
    return entry_id, [s1, s2, s3]


@pytest.mark.asyncio
async def test_reindex_retires_the_entry_and_replays_in_capture_order(
    db_session: AsyncSession, llm: FakeLLM
) -> None:
    entry_id, snapshots = await _three_snapshots(db_session)
    before = {p.id for p in await get_particles_for_entry(db_session, entry_id)}
    assert len(before) == 3
    llm.reset()

    result = await reindex(
        db_session, entry_ids=[entry_id], run_post_lint=False, rate_limit_per_minute=0
    )

    assert result["failed"] == 0
    # The first snapshot is read whole with no base; each later one as a delta
    # from the one before it.
    assert [context is None for _, context in llm.calls] == [True, False, False]
    assert "Alpha" in llm.calls[0][0] and "Beta" in llm.calls[0][0]
    assert "Gamma" in llm.calls[1][0] and "Alpha" not in llm.calls[1][0]
    assert "Delta" in llm.calls[2][0] and "Gamma" not in llm.calls[2][0]

    after = await get_particles_for_entry(db_session, entry_id)
    retired = [p for p in after if p.id in before]
    assert all(p.status is Status.SUPERSEDED for p in retired)
    assert all(p.status_reason is StatusReason.SUPERSEDED_BY_REINDEX for p in retired)
    fresh = [p for p in after if p.id not in before]
    assert all(p.status is Status.ACTIVE for p in fresh)
    # Each claim cites the snapshot that first contained its passage.
    assert sorted(p.provenance[0].snapshot_id for p in fresh) == sorted(snapshots)
    for snapshot_id, text in zip(snapshots, (FIRST, SECOND, THIRD), strict=True):
        assert await _extracted_through(db_session, snapshot_id) == len(text.encode())


@pytest.mark.asyncio
async def test_a_scope_naming_snapshots_out_of_order_replays_in_order(
    db_session: AsyncSession, llm: FakeLLM
) -> None:
    entry_id, (s1, s2, s3) = await _three_snapshots(db_session)

    replays, ordinary = await _group_append_only(db_session, [(entry_id, s3), (entry_id, s1)])

    assert replays == {entry_id: [s1, s2, s3]}
    assert ordinary == []


@pytest.mark.asyncio
async def test_an_extractor_version_scope_reaches_the_whole_entry(
    db_session: AsyncSession, llm: FakeLLM
) -> None:
    from particles.extraction.general import EXTRACTOR_VERSION

    entry_id, _ = await _three_snapshots(db_session)
    llm.reset()

    await reindex(
        db_session,
        extractor_version=EXTRACTOR_VERSION,
        include_failed=False,
        run_post_lint=False,
        rate_limit_per_minute=0,
    )

    assert len(llm.calls) == 3
    active = [
        p for p in await get_particles_for_entry(db_session, entry_id) if p.status is Status.ACTIVE
    ]
    assert len(active) == 3


@pytest.mark.asyncio
async def test_other_mutability_classes_reindex_per_snapshot(
    db_session: AsyncSession, llm: FakeLLM
) -> None:
    from particles.core.schema import Mutability

    entry_id, s1 = await _deposit(db_session, FIRST, mutability=Mutability.MUTABLE)
    await _extract(db_session, entry_id, s1)

    replays, ordinary = await _group_append_only(db_session, [(entry_id, s1)])

    assert replays == {}
    assert ordinary == [(entry_id, s1)]
