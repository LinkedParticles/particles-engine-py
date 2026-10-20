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
from sqlalchemy.sql.elements import ColumnClause

from particles.config import get_config
from particles.core.schema import Particle
from particles.core.status import Status
from particles.embeddings import get_embedding_model_id
from particles.store.change_journal import changed_since, journal_installed, read_head, token_at
from particles.store.particle_store import (
    LEGACY_EMBEDDING_MODEL_ID,
    ParticleRow,
    decode_embedding,
    get_active_particles_with_embeddings,
    log_skipped_embedding_models,
)
from particles.store.subject_store import get_particles_for_subject

log = logging.getLogger(__name__)

Embedding = np.ndarray[Any, np.dtype[np.float32]]

#: Rows read per partition while building, so the JSON text and the ORM rows of
#: one partition are freed before the next is read.
_BUILD_PARTITION = 2_000
#: Ids per ``IN`` list when re-reading changed rows.
_DELTA_CHUNK = 500
#: Re-read at most this many changed rows (or a quarter of the cache, if
#: larger) before rebuilding instead.
_DELTA_FLOOR = 1_000


@dataclass(frozen=True, slots=True)
class _Entry:
    """One cached ACTIVE row."""

    particle: Particle
    embedding: Embedding
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


_STATES: weakref.WeakKeyDictionary[Engine, _State] = weakref.WeakKeyDictionary()


def clear() -> None:
    """Drop every store's cache (tests, and after an operator reset)."""
    _STATES.clear()


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
    engine = _engine_of(session)
    if (
        engine is None
        or not get_config().query.candidate_cache
        or not await journal_installed(session)
    ):
        return await get_active_particles_with_embeddings(session, min_confidence, subject_id)
    previous = _STATES.get(engine)
    state = await _current_state(session, previous, may_rebuild=subject_id is None)
    if state is None:
        return await get_active_particles_with_embeddings(session, min_confidence, subject_id)
    _STATES[engine] = state
    if state is not previous and state.rebuilt:
        del previous  # the replaced rows are freed before the collection below
        _exempt_from_collection(len(state.ordered))
    allowed = (
        None
        if subject_id is None
        else frozenset(await get_particles_for_subject(session, subject_id))
    )
    return _select(state, min_confidence, allowed)


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


def _select(
    state: _State, min_confidence: float, allowed: Collection[str] | None
) -> list[tuple[Particle, Embedding]]:
    current_id = get_embedding_model_id()
    if allowed is None:
        entries: Collection[_Entry] = state.ordered
    else:
        linked = [state.by_id[pid] for pid in allowed if pid in state.by_id]
        entries = sorted(linked, key=attrgetter("order"))
    out: list[tuple[Particle, Embedding]] = []
    skipped = 0
    for entry in entries:
        if entry.confidence < min_confidence:
            continue
        if entry.model_id != current_id:
            skipped += 1
            continue
        out.append((entry.particle, entry.embedding))
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
    embedding = decode_embedding(row.embedding_json or "[]")
    embedding.flags.writeable = False
    return _Entry(
        particle=row.to_model(),
        embedding=embedding,
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
    return _State(head=head, by_id=by_id, ordered=tuple(ordered), rebuilt=True)


async def _apply_changes(
    session: AsyncSession, state: _State, head: tuple[int, bytes], changed: set[str]
) -> _State:
    by_id = dict(state.by_id)
    pending = sorted(changed)
    for start in range(0, len(pending), _DELTA_CHUNK):
        chunk = pending[start : start + _DELTA_CHUNK]
        result = await session.execute(select(ParticleRow, _ROWID).where(ParticleRow.id.in_(chunk)))
        found: dict[str, tuple[ParticleRow, int]] = {row.id: (row, rowid) for row, rowid in result}
        for pid in chunk:
            hit = found.get(pid)
            if hit is not None and _cacheable(hit[0]):
                by_id[pid] = _entry(*hit)
            else:
                by_id.pop(pid, None)
    ordered = tuple(sorted(by_id.values(), key=attrgetter("order")))
    log.debug("candidate cache: re-read %d changed row(s), %d cached", len(changed), len(ordered))
    return _State(head=head, by_id=by_id, ordered=ordered, rebuilt=False)
