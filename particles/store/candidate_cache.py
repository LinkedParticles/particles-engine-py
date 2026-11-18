# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The ACTIVE candidates of semantic query, decoded once per process.

A semantic query scores every ACTIVE particle that has an embedding.
:func:`~particles.store.particle_store.get_active_particles_with_embeddings`
reads them all from SQLite, ``json.loads`` each embedding and builds a
``Particle`` for each row, on every query; on the query-latency baseline that
was about three quarters of a 5.7 s query at 56k rows
(``docs/benchmarks/query-latency.md``). The rows rarely change between two
queries, so this module keeps the decoded rows of each store in memory and
re-reads only the rows that changed.

**How it knows what changed.** The store's change journal
(:mod:`particles.store.change_journal`) names every ``particles`` row that any
process inserted, updated or deleted, so a write by the session hooks, the CLI
or an older install invalidates exactly what it touched. Each query reads the
journal's newest entry (one indexed lookup); when it differs from the one the
cache was built at, the cache re-reads the rows named since then, or rebuilds
when it cannot tell what it missed. Nothing is ever served without that check,
so the cache returns what the uncached loader would return at the moment of the
query: the same rows, the same particles, the same vectors, in the same
``(confidence_value, rowid)`` order.

**What it holds.** Every ACTIVE row with an embedding, whatever its
confidence or embedding model. ``min_confidence``, the embedding-model check
and its warning, and the subject filter are applied per query, the subject
filter through the live ``particle_subjects`` table, which the journal does not
cover. Only this ACTIVE set is cached. The as-of lens and the window route read
across statuses with their own predicates and keep their own loaders.

**Who reads it.** Semantic query, and every other reader of the whole ACTIVE
set: each extraction pass and each ``reconcile_and_insert`` call
(``ingest/pipeline.py``), and the contradiction census, the update and
document-supersession sweeps, re-anchoring, the paraphrase sweep, utility
mining, co-evidential suggestion and the abstraction pass. On a 56k-row store
a warm read takes about 6 ms against 4.5 s for the loader, and an extraction
pass about 0.25 s against 4.9 s.

**Vectors on demand.** A full build reads every row without its embedding;
the vectors are read and decoded the first time a caller asks for them, and
kept from then on. Parsing the JSON vectors is about 2.9 s of a 7 s cold build
at 61k rows (2026-11-07), and the readers that rank by confidence alone (the
``MEMORY.md`` region, the documentation projection) never compare a vector, so
a short-lived process that only renders, such as a Claude Code hook, no longer
pays for them. :func:`load_active_candidate_particles` is that reader's entry
point; :func:`load_active_candidates` returns the vectors as before.

**Sharing.** The ``Particle`` objects and the vectors are shared by every
reader. The vectors are read-only arrays, and callers must treat the particles
as read-only too: the query path copies the ones it returns
(``operations/query/main.py``), and the other readers hand none out.

**Garbage collection.** A full build of a large cache ends by exempting the
cache from Python's cyclic garbage collector (``gc.freeze``), which otherwise
walks every cached object on most queries; see :func:`_exempt_from_collection`
and ``query.candidate_cache_freeze_rows``. This is a process-wide setting.

**When it stays out of the way.** No cache is used when
``query.candidate_cache`` is off, when the store is not SQLite or its journal
triggers are missing, or when the journal is empty. A query filtered by subject
never builds the cache: it reads its few rows directly, which is cheaper than
decoding the whole store, and it uses the cache only once an unfiltered query
has built it.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import weakref
from collections.abc import Collection
from dataclasses import dataclass
from operator import attrgetter
from typing import Any

import numpy as np
from sqlalchemy import literal_column, select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer
from sqlalchemy.sql.elements import ColumnClause

from particles.config import get_config
from particles.core.schema import Particle
from particles.core.status import Status
from particles.embeddings import get_embedding_model_id, row_norms
from particles.store.change_journal import changed_since, journal_installed, read_head, token_at
from particles.store.particle_store import (
    LEGACY_EMBEDDING_MODEL_ID,
    ParticleRow,
    decode_embedding,
    get_active_particles_with_embeddings,
    get_embeddings_by_ids,
    log_skipped_embedding_models,
)
from particles.store.subject_store import get_particles_for_subject

log = logging.getLogger(__name__)

Embedding = np.ndarray[Any, np.dtype[np.float32]]

#: Rows read per partition while building, so the JSON text and the ORM rows of
#: one partition are freed before the next is read.
_BUILD_PARTITION = 2_000
#: Ids per ``IN`` list when re-reading changed rows, or reading vectors by id.
_DELTA_CHUNK = 500
#: Read missing vectors with one scan of the ACTIVE rows, rather than by id,
#: when more than this many are wanted.
_VECTOR_SCAN_FLOOR = 5_000
#: Re-read at most this many changed rows (or a quarter of the cache, if
#: larger) before rebuilding instead.
_DELTA_FLOOR = 1_000


@dataclass(frozen=True, slots=True)
class _Entry:
    """One cached ACTIVE row."""

    particle: Particle
    confidence: float
    model_id: str
    order: tuple[float, int]


@dataclass(frozen=True, slots=True)
class _State:
    """A store's cached rows, valid as of journal entry ``head``."""

    head: tuple[int, bytes]
    by_id: dict[str, _Entry]
    ordered: tuple[_Entry, ...]
    #: Built from scratch, rather than brought up from an earlier state.
    rebuilt: bool
    #: The decoded vectors read so far, by particle id: filled on demand
    #: (:func:`_ensure_vectors`), so a row without one has not been asked for.
    vectors: dict[str, Embedding]


_STATES: weakref.WeakKeyDictionary[Engine, _State] = weakref.WeakKeyDictionary()

#: One lock per store and event loop, held while the cache is brought up to
#: date or its vectors read, so two readers that arrive together (an engine's
#: warm-up and the first request) build one copy, not two. An
#: asyncio lock belongs to one loop, and a CLI process runs several in turn.
_LOCKS: weakref.WeakKeyDictionary[Engine, tuple[asyncio.AbstractEventLoop, asyncio.Lock]] = (
    weakref.WeakKeyDictionary()
)


def _lock_for(engine: Engine) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    held = _LOCKS.get(engine)
    if held is None or held[0] is not loop:
        held = (loop, asyncio.Lock())
        _LOCKS[engine] = held
    return held[1]


def clear() -> None:
    """Drop every store's cache (tests, and after an operator reset)."""
    _STATES.clear()
    _MATRICES.clear()


def cached_rows(session: AsyncSession) -> int | None:
    """How many rows the session's store has cached, or ``None`` for no cache."""
    engine = _engine_of(session)
    state = _STATES.get(engine) if engine is not None else None
    return None if state is None else len(state.ordered)


async def load_active_candidates(
    session: AsyncSession,
    min_confidence: float = 0.0,
    subject_id: str | None = None,
) -> list[tuple[Particle, Embedding]]:
    """The cached equivalent of ``get_active_particles_with_embeddings``.

    Same arguments, same result, same order; see the module docstring for when
    it reads through the cache and when it falls back to the loader.
    """
    state = await _cached_state(session, subject_id)
    if state is None:
        return await get_active_particles_with_embeddings(session, min_confidence, subject_id)
    entries = _select(state, min_confidence, await _allowed(session, subject_id))
    await _ensure_vectors_locked(session, state, entries)
    # A row whose vector vanished after the head was read (retired, or its
    # embedding cleared) is left out, as the loader would leave it out now.
    return [
        (entry.particle, state.vectors[entry.particle.id])
        for entry in entries
        if entry.particle.id in state.vectors
    ]


async def load_active_candidate_particles(
    session: AsyncSession,
    min_confidence: float = 0.0,
    subject_id: str | None = None,
) -> list[Particle]:
    """:func:`load_active_candidates` without the vectors: the same particles, same order.

    For a reader that ranks without comparing vectors. It reads no vector, so
    a cache this call builds holds none until a caller asks for them.
    """
    state = await _cached_state(session, subject_id)
    if state is None:
        rows = await get_active_particles_with_embeddings(session, min_confidence, subject_id)
        return [p for p, _ in rows]
    return [e.particle for e in _select(state, min_confidence, await _allowed(session, subject_id))]


async def load_candidate_vectors(
    session: AsyncSession, particle_ids: Collection[str]
) -> dict[str, Embedding]:
    """The current-model vectors of the ACTIVE particles among ``particle_ids``.

    Read through the cache, so a caller asking for a few hundred vectors per
    session over many sessions decodes each one once per process; without a
    cache, read by id. Missing, retired and other-model ids are absent.
    """
    state = await _cached_state(session, None)
    if state is None:
        return await get_embeddings_by_ids(session, list(dict.fromkeys(particle_ids)))
    current_id = get_embedding_model_id()
    entries = [
        entry
        for pid in dict.fromkeys(particle_ids)
        if (entry := state.by_id.get(pid)) is not None and entry.model_id == current_id
    ]
    await _ensure_vectors_locked(session, state, entries)
    return {
        e.particle.id: state.vectors[e.particle.id]
        for e in entries
        if e.particle.id in state.vectors
    }


@dataclass(frozen=True, slots=True)
class CandidateMatrix:
    """Every current-model ACTIVE vector stacked into one matrix.

    For a reader that compares one vector with the whole ACTIVE set on every
    request, such as the subject suggestions, so a request is one product
    rather than a stack of 61k vectors. Row ``i`` is ``particles[i]``'s vector;
    ``norms`` are the rows' L2 norms. Read-only, like every cached vector.
    """

    particles: tuple[Particle, ...]
    vectors: Embedding
    norms: Embedding


#: Each store's matrix and the journal head it was stacked at.
_MATRICES: weakref.WeakKeyDictionary[Engine, tuple[tuple[int, bytes], CandidateMatrix]] = (
    weakref.WeakKeyDictionary()
)


def _stack(particles: list[Particle], vectors: list[Embedding]) -> CandidateMatrix:
    dim = len(vectors[0]) if vectors else 0
    matrix = np.stack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)
    matrix.flags.writeable = False
    norms = row_norms(matrix)
    norms.flags.writeable = False
    return CandidateMatrix(particles=tuple(particles), vectors=matrix, norms=norms)


async def load_candidate_matrix(session: AsyncSession) -> CandidateMatrix:
    """:func:`load_active_candidates` stacked, kept until the store changes.

    The same particles in the same order. With a cache, the matrix is stacked
    once per journal head, and the cache's vectors become views of its rows, so
    each vector is held once. Without one, it is stacked on every call.
    """
    state = await _cached_state(session, None)
    engine = _engine_of(session)
    if state is None or engine is None:
        rows = await get_active_particles_with_embeddings(session)
        return _stack([p for p, _ in rows], [v for _, v in rows])
    async with _lock_for(engine):
        held = _MATRICES.get(engine)
        if held is not None and held[0] == state.head:
            return held[1]
        entries = _select(state, 0.0, None)
        await _ensure_vectors(session, state, entries)
        kept = [e.particle for e in entries if e.particle.id in state.vectors]
        matrix = _stack(kept, [state.vectors[p.id] for p in kept])
        # The rows replace the separately decoded vectors, so the store's
        # vectors are held once, in the matrix.
        for i, particle in enumerate(kept):
            state.vectors[particle.id] = matrix.vectors[i]
        _MATRICES[engine] = (state.head, matrix)
        return matrix


async def _cached_state(session: AsyncSession, subject_id: str | None) -> _State | None:
    """The session's store's cache brought up to date, or ``None`` to read directly."""
    engine = _engine_of(session)
    if (
        engine is None
        or not get_config().query.candidate_cache
        or not await journal_installed(session)
    ):
        return None
    async with _lock_for(engine):
        previous = _STATES.get(engine)
        state = await _current_state(session, previous, may_rebuild=subject_id is None)
        if state is None:
            return None
        _STATES[engine] = state
    if state is not previous and state.rebuilt:
        del previous  # the replaced rows are freed before the collection below
        _exempt_from_collection(len(state.ordered))
    return state


async def _allowed(session: AsyncSession, subject_id: str | None) -> frozenset[str] | None:
    if subject_id is None:
        return None
    return frozenset(await get_particles_for_subject(session, subject_id))


async def _ensure_vectors_locked(
    session: AsyncSession, state: _State, entries: Collection[_Entry]
) -> None:
    """:func:`_ensure_vectors` under the store's lock, so concurrent readers decode once."""
    engine = _engine_of(session)
    if engine is None:
        await _ensure_vectors(session, state, entries)
        return
    async with _lock_for(engine):
        await _ensure_vectors(session, state, entries)


async def _ensure_vectors(
    session: AsyncSession, state: _State, entries: Collection[_Entry]
) -> None:
    """Read and decode the vectors of ``entries`` that the cache does not hold yet.

    Read after the journal head, like every row the cache holds, so a vector
    that changes in between is re-read by the next refresh, which drops the
    vectors of the rows it re-reads.
    """
    missing = [e.particle.id for e in entries if e.particle.id not in state.vectors]
    if not missing:
        return
    if len(missing) > _VECTOR_SCAN_FLOOR:
        wanted = set(missing)
        query = (
            select(ParticleRow.id, ParticleRow.embedding_json)
            .where(
                ParticleRow.status == Status.ACTIVE.value,
                ParticleRow.embedding_json.isnot(None),
            )
            .execution_options(yield_per=_BUILD_PARTITION)
        )
        result = await session.stream(query)
        async for partition in result.partitions():
            for pid, raw in partition:
                if pid in wanted:
                    state.vectors[pid] = _vector(raw or "[]")
        return
    for start in range(0, len(missing), _DELTA_CHUNK):
        chunk = missing[start : start + _DELTA_CHUNK]
        rows = await session.execute(
            select(ParticleRow.id, ParticleRow.embedding_json).where(
                ParticleRow.id.in_(chunk),
                ParticleRow.status == Status.ACTIVE.value,
                ParticleRow.embedding_json.isnot(None),
            )
        )
        for pid, raw in rows:
            state.vectors[pid] = _vector(raw or "[]")


def _vector(raw: str) -> Embedding:
    vector = decode_embedding(raw)
    vector.flags.writeable = False
    return vector


def _exempt_from_collection(rows: int) -> None:
    """Keep the cyclic garbage collector from walking a large cache on every pass.

    A cached row is about two dozen objects the collector tracks, so a store's
    cache is over a million of them at 56k rows. Every full collection walks
    them all (about 0.3 s at 56k), and a query allocates enough to set one off
    on most queries; on the first after-measurement it doubled the median warm
    query. ``gc.freeze`` moves every object alive now into a permanent
    generation the collector skips. Reference counting still frees those
    objects, which is how a replaced cache's rows go: a ``Particle`` holds no
    reference cycle. The collection before it, with the earlier frozen
    objects unfrozen, frees any cyclic garbage first, so nothing unreachable is
    frozen. It runs once per full build, and only for a cache of at least
    ``query.candidate_cache_freeze_rows`` rows (0 never freezes).
    """
    threshold = get_config().query.candidate_cache_freeze_rows
    if threshold <= 0 or rows < threshold:
        return
    gc.unfreeze()
    gc.collect()
    gc.freeze()


def _engine_of(session: AsyncSession) -> Engine | None:
    bind = session.get_bind()
    if isinstance(bind, Connection):
        return bind.engine
    return bind if isinstance(bind, Engine) else None


def _select(state: _State, min_confidence: float, allowed: Collection[str] | None) -> list[_Entry]:
    current_id = get_embedding_model_id()
    if allowed is None:
        entries: Collection[_Entry] = state.ordered
    else:
        linked = [state.by_id[pid] for pid in allowed if pid in state.by_id]
        entries = sorted(linked, key=attrgetter("order"))
    out: list[_Entry] = []
    skipped = 0
    for entry in entries:
        if entry.confidence < min_confidence:
            continue
        if entry.model_id != current_id:
            skipped += 1
            continue
        out.append(entry)
    if skipped:
        log_skipped_embedding_models(skipped, current_id)
    return out


async def _current_state(
    session: AsyncSession, state: _State | None, *, may_rebuild: bool
) -> _State | None:
    """``state`` brought up to the journal's head, or ``None`` to read directly.

    The head is read before any row, so every row read afterwards is at least
    as new as it: a change that lands in between is also in the journal after
    the head, and the next refresh re-reads it.
    """
    head = await read_head(session)
    if head is None:
        return None
    if state is not None and state.head == head:
        return state
    if state is not None and await token_at(session, state.head[0]) == state.head[1]:
        limit = max(_DELTA_FLOOR, len(state.by_id) // 4)
        changed = await changed_since(session, state.head[0], limit=limit)
        if changed is not None:
            return await _apply_changes(session, state, head, changed)
    if not may_rebuild:
        return None
    return await _build(session, head)


def _entry(row: ParticleRow, rowid: int) -> _Entry:
    return _Entry(
        particle=row.to_model(),
        confidence=row.confidence_value,
        model_id=row.embedding_model_id or LEGACY_EMBEDDING_MODEL_ID,
        order=(row.confidence_value, rowid),
    )


def _cacheable(row: ParticleRow) -> bool:
    return row.status == Status.ACTIVE.value and row.embedding_json is not None


_ROWID: ColumnClause[Any] = literal_column("particles.rowid")


async def _build(session: AsyncSession, head: tuple[int, bytes]) -> _State:
    query = (
        select(ParticleRow, _ROWID)
        .where(ParticleRow.status == Status.ACTIVE.value, ParticleRow.embedding_json.isnot(None))
        .order_by(ParticleRow.confidence_value, _ROWID)
        # The vectors are read on demand (_ensure_vectors).
        .options(defer(ParticleRow.embedding_json))
        .execution_options(yield_per=_BUILD_PARTITION)
    )
    ordered: list[_Entry] = []
    by_id: dict[str, _Entry] = {}
    result = await session.stream(query)
    async for partition in result.partitions():
        for row, rowid in partition:
            entry = _entry(row, rowid)
            ordered.append(entry)
            by_id[entry.particle.id] = entry
    return _State(head=head, by_id=by_id, ordered=tuple(ordered), rebuilt=True, vectors={})


async def _apply_changes(
    session: AsyncSession, state: _State, head: tuple[int, bytes], changed: set[str]
) -> _State:
    by_id = dict(state.by_id)
    vectors = {pid: v for pid, v in state.vectors.items() if pid not in changed}
    pending = sorted(changed)
    for start in range(0, len(pending), _DELTA_CHUNK):
        chunk = pending[start : start + _DELTA_CHUNK]
        result = await session.execute(select(ParticleRow, _ROWID).where(ParticleRow.id.in_(chunk)))
        found: dict[str, tuple[ParticleRow, int]] = {row.id: (row, rowid) for row, rowid in result}
        for pid in chunk:
            hit = found.get(pid)
            if hit is not None and _cacheable(hit[0]):
                by_id[pid] = _entry(*hit)
                vectors[pid] = _vector(hit[0].embedding_json or "[]")
            else:
                by_id.pop(pid, None)
    ordered = tuple(sorted(by_id.values(), key=attrgetter("order")))
    log.debug("candidate cache: re-read %d changed row(s), %d cached", len(changed), len(ordered))
    return _State(head=head, by_id=by_id, ordered=ordered, rebuilt=False, vectors=vectors)
