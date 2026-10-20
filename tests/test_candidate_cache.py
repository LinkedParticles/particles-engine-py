# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The candidate cache returns exactly what the uncached loader returns.

Every test compares :func:`load_active_candidates` against
:func:`get_active_particles_with_embeddings` on the same store at the same
moment: same particles, same vectors, same order. The cases are the ways a
cache goes stale: a write in the same session, a write from another
connection, a rolled-back write, a status change, a deletion, and a journal
the cache has fallen behind.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.store import candidate_cache
from particles.store.candidate_cache import cached_rows, load_active_candidates
from particles.store.change_journal import read_head
from particles.store.particle_store import (
    ParticleRow,
    get_active_particles_with_embeddings,
    insert_particle,
    update_particle_status,
)


def _particle(content: str, confidence: float, subjects: list[str] | None = None) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        subject_ids=subjects or [],
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
    )


def _vector(seed: int) -> list[float]:
    v = np.random.default_rng(seed).normal(size=8).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()  # type: ignore[no-any-return]


async def _seed(session: AsyncSession) -> list[Particle]:
    """Ten rows: confidence ties, two subjects, a legacy model id, one without a vector."""
    out = []
    for i in range(10):
        # 0.9 four times, 0.5 three times, 0.7 three times: ties broken by rowid.
        confidence = (0.9, 0.5, 0.7)[i % 3]
        subjects = ["subj-a"] if i % 2 else ["subj-a", "subj-b"] if i % 3 == 0 else []
        p = _particle(f"claim {i}", confidence, subjects)
        await insert_particle(session, p, None if i == 9 else _vector(i))
        out.append(p)
    await session.execute(
        update(ParticleRow)
        .where(ParticleRow.id == out[4].id)
        .values(embedding_model_id="old-model")
    )
    await session.commit()
    return out


async def _assert_same(
    session: AsyncSession, min_confidence: float = 0.0, subject_id: str | None = None
) -> list[tuple[Particle, Any]]:
    cached = await load_active_candidates(session, min_confidence, subject_id)
    direct = await get_active_particles_with_embeddings(session, min_confidence, subject_id)
    assert [p.id for p, _ in cached] == [p.id for p, _ in direct]
    for (cp, ce), (dp, de) in zip(cached, direct, strict=True):
        assert cp.model_dump() == dp.model_dump()
        assert ce.dtype == de.dtype and np.array_equal(ce, de)
    return cached


@contextmanager
def _count_hydrations() -> Iterator[list[int]]:
    saved = ParticleRow.to_model
    calls = [0]

    def counting(self: ParticleRow) -> Particle:
        calls[0] += 1
        return saved(self)

    setattr(ParticleRow, "to_model", counting)  # noqa: B010
    try:
        yield calls
    finally:
        setattr(ParticleRow, "to_model", saved)  # noqa: B010


async def test_same_rows_order_and_vectors_as_the_loader(db_session: AsyncSession) -> None:
    await _seed(db_session)
    rows = await _assert_same(db_session)
    assert cached_rows(db_session) == 9  # every ACTIVE row with a vector, any model
    assert len(rows) == 8  # the old-model row is skipped per query, as the loader does
    confidences = [p.confidence.value for p, _ in rows]
    assert confidences == sorted(confidences)
    await _assert_same(db_session, min_confidence=0.6)
    await _assert_same(db_session, subject_id="subj-a")
    await _assert_same(db_session, subject_id="subj-b")
    await _assert_same(db_session, subject_id="no-such-subject")


async def test_a_warm_query_hydrates_nothing(db_session: AsyncSession) -> None:
    await _seed(db_session)
    await load_active_candidates(db_session)
    with _count_hydrations() as calls:
        await _assert_same(db_session)
    # Every hydration was the loader's 8 rows; the cache hydrated none.
    assert calls[0] == 8


async def test_the_skipped_model_warning_is_kept(
    db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    await _seed(db_session)
    await load_active_candidates(db_session)
    caplog.clear()
    await load_active_candidates(db_session)
    assert any("Skipped 1 ACTIVE particle(s)" in r.getMessage() for r in caplog.records)


async def test_writes_in_the_session_are_seen(db_session: AsyncSession) -> None:
    seeded = await _seed(db_session)
    await load_active_candidates(db_session)

    new = _particle("a new claim", 0.9, ["subj-b"])
    await insert_particle(db_session, new, _vector(99))
    with _count_hydrations() as calls:
        rows = await load_active_candidates(db_session)
        assert calls[0] == 1  # only the new row is read
    assert new.id in {p.id for p, _ in rows}
    await _assert_same(db_session)

    await update_particle_status(
        db_session, seeded[0].id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    rows = await _assert_same(db_session)
    assert seeded[0].id not in {p.id for p, _ in rows}

    await db_session.execute(
        update(ParticleRow).where(ParticleRow.id == seeded[1].id).values(confidence_value=0.95)
    )
    await _assert_same(db_session)

    await db_session.execute(text("DELETE FROM particles WHERE id = :id"), {"id": seeded[2].id})
    rows = await _assert_same(db_session)
    assert seeded[2].id not in {p.id for p, _ in rows}

    # A vector stamped with the current model makes the skipped row searchable.
    await db_session.execute(
        update(ParticleRow).where(ParticleRow.id == seeded[4].id).values(embedding_model_id=None)
    )
    rows = await _assert_same(db_session)
    assert seeded[4].id in {p.id for p, _ in rows}


async def test_a_rolled_back_write_never_survives(db_session: AsyncSession) -> None:
    await _seed(db_session)
    await load_active_candidates(db_session)

    # The cache is refreshed inside a write transaction, which then rolls back.
    phantom = _particle("never committed", 0.9)
    await insert_particle(db_session, phantom, _vector(7))
    seen = await load_active_candidates(db_session)
    assert phantom.id in {p.id for p, _ in seen}
    phantom_head = await read_head(db_session)
    await db_session.rollback()

    # The next write reuses the rolled-back journal seq, with another token, so
    # the cache cannot mistake it for the change it already holds.
    other = _particle("committed instead", 0.9)
    await insert_particle(db_session, other, _vector(8))
    await db_session.commit()
    head = await read_head(db_session)
    assert head is not None and phantom_head is not None
    assert head[0] == phantom_head[0] and head[1] != phantom_head[1]
    rows = await _assert_same(db_session)
    ids = {p.id for p, _ in rows}
    assert other.id in ids and phantom.id not in ids


async def test_a_rollback_seen_before_the_next_write(db_session: AsyncSession) -> None:
    await _seed(db_session)
    phantom = _particle("never committed", 0.9)
    await insert_particle(db_session, phantom, _vector(7))
    await load_active_candidates(db_session)
    await db_session.rollback()
    rows = await _assert_same(db_session)
    assert phantom.id not in {p.id for p, _ in rows}


async def test_a_write_from_another_connection_is_seen(file_db_session: AsyncSession) -> None:
    seeded = await _seed(file_db_session)
    await load_active_candidates(file_db_session)
    db_path = file_db_session.get_bind().url.database
    assert db_path is not None
    with sqlite3.connect(db_path) as other:
        other.execute("UPDATE particles SET status = 'RETRACTED' WHERE id = ?", (seeded[3].id,))
    rows = await _assert_same(file_db_session)
    assert seeded[3].id not in {p.id for p, _ in rows}


async def test_a_cache_the_journal_has_outrun_is_rebuilt(db_session: AsyncSession) -> None:
    await _seed(db_session)
    await load_active_candidates(db_session)
    head = await read_head(db_session)
    assert head is not None
    # Pruning removed the entry the cache was built at.
    await db_session.execute(text("DELETE FROM particle_changes"))
    await insert_particle(db_session, _particle("after the prune", 0.9), _vector(5))
    with _count_hydrations() as calls:
        await load_active_candidates(db_session)
        assert calls[0] == 10  # a full rebuild
    await _assert_same(db_session)


async def test_too_many_changes_rebuild(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(db_session)
    await load_active_candidates(db_session)
    monkeypatch.setattr(candidate_cache, "_DELTA_FLOOR", 1)
    for i in range(4):
        await insert_particle(db_session, _particle(f"bulk {i}", 0.9), _vector(50 + i))
    with _count_hydrations() as calls:
        await load_active_candidates(db_session)
        assert calls[0] == 13
    await _assert_same(db_session)


async def test_a_subject_query_never_builds_the_cache(db_session: AsyncSession) -> None:
    await _seed(db_session)
    await _assert_same(db_session, subject_id="subj-a")
    assert cached_rows(db_session) is None
    await load_active_candidates(db_session)
    assert cached_rows(db_session) == 9
    with _count_hydrations() as calls:
        await load_active_candidates(db_session, subject_id="subj-a")
        assert calls[0] == 0


async def test_off_without_the_config_or_the_triggers(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(db_session)
    get_config().query.candidate_cache = False
    await _assert_same(db_session)
    assert cached_rows(db_session) is None

    get_config().query.candidate_cache = True
    await db_session.execute(text("DROP TRIGGER trg_particles_journal_update"))
    await _assert_same(db_session)
    assert cached_rows(db_session) is None


async def test_cached_vectors_are_read_only(db_session: AsyncSession) -> None:
    await _seed(db_session)
    rows = await load_active_candidates(db_session)
    with pytest.raises(ValueError):
        rows[0][1][0] = 0.0


async def test_returned_particles_are_copies(db_session: AsyncSession) -> None:
    from particles.operations.query import retrieve_ranked

    await _seed(db_session)
    cached = {p.id: p for p, _ in await load_active_candidates(db_session)}
    ranked = await retrieve_ranked(
        db_session, QueryRequest(question="anything", top_k=5), use_embeddings=False
    )
    assert ranked
    for p, _, _ in ranked:
        assert p is not cached[p.id]
        assert p.model_dump() == cached[p.id].model_dump()


async def test_retrieve_ranked_is_identical_with_and_without_the_cache(
    db_session: AsyncSession,
) -> None:
    from particles import embeddings as ep
    from particles.operations.query import retrieve_ranked

    await _seed(db_session)

    class _Encoder:
        def encode(self, texts: list[str], **_: Any) -> list[Any]:
            return [np.asarray(_vector(3), dtype=np.float32) for _ in texts]

    original = ep._embedding_model
    ep.set_embedding_model(_Encoder())  # type: ignore[arg-type]
    try:
        request = QueryRequest(question="which claim?", top_k=6)
        get_config().query.candidate_cache = False
        direct = await retrieve_ranked(db_session, request)
        get_config().query.candidate_cache = True
        await retrieve_ranked(db_session, request)  # builds the cache
        cached = await retrieve_ranked(db_session, request)
    finally:
        ep.set_embedding_model(original)
    assert cached_rows(db_session) == 9
    assert [(p.id, sim, eff) for p, sim, eff in cached] == [
        (p.id, sim, eff) for p, sim, eff in direct
    ]


async def test_a_large_build_is_exempted_from_collection(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gc

    freezes = [0]
    real_freeze = gc.freeze

    def counting_freeze() -> None:
        freezes[0] += 1
        real_freeze()

    monkeypatch.setattr(candidate_cache.gc, "freeze", counting_freeze)
    await _seed(db_session)
    try:
        get_config().query.candidate_cache_freeze_rows = 100
        await load_active_candidates(db_session)  # 9 rows: under the threshold
        assert freezes[0] == 0
        candidate_cache.clear()
        get_config().query.candidate_cache_freeze_rows = 5
        await load_active_candidates(db_session)
        assert freezes[0] == 1 and gc.get_freeze_count() > 0
        # A refresh after a write brings the cache up without freezing again.
        await insert_particle(db_session, _particle("one more", 0.9), _vector(77))
        await load_active_candidates(db_session)
        await load_active_candidates(db_session)
        assert freezes[0] == 1
    finally:
        gc.unfreeze()
