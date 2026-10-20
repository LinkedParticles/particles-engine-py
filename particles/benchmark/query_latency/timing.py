# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-stage timing of ``retrieve_ranked``, from outside the library.

Nothing in the query path is instrumented. :func:`instrument` swaps the seams
``retrieve_ranked`` already calls through module globals for timed wrappers, and
restores them on exit:

* in :mod:`particles.operations.query.main`: ``_embed``, the loader
  ``load_active_candidates`` (the candidate cache in front of
  ``get_active_particles_with_embeddings``), ``_gather_scored``,
  ``load_source_rows``, ``stale_support_discounts`` and
  ``_collapse_co_evidential_top_k``;
* ``ParticleRow.to_model`` (hydration), counted only while the loader runs;
* the session class's ``execute`` for the duration of the loader call (the SQL fetch,
  which with SQLAlchemy's async session includes building the ORM rows, since
  the async result is pre-buffered), and its ``stream``, whose partition
  fetches count the same way (the candidate cache builds from one);
* ``particles.corpus.store.get_snapshot_source_meta`` and
  ``get_source_types_for_entries``, to count the ``IN`` lists they send (both
  are imported at call time, so the module attribute is what they resolve).

Each stage is a difference of ``perf_counter`` marks, so the stages and the
residual ``other`` sum to the measured total. ``decode`` is what the loader
spends outside the SQL fetch and hydration: the ``json.loads`` of each
embedding, the ``np.array`` around it and the model-id check, and on a query
the candidate cache answers, its selection over the cached rows. The SQL fetch
then covers the change-journal lookups, and hydration and the counted
embedding bytes cover only the rows the cache had to (re)read. The wrapper around
``to_model`` costs two clock reads per row (well under a microsecond), against a
hydration cost two orders of magnitude larger; that overhead is inside
``hydrate``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from particles.benchmark.query_latency.schema import STAGES, QuerySample


@dataclass
class _Marks:
    """One query's clock marks and counters."""

    start: float = 0.0
    embed: tuple[float, float] | None = None
    loader: tuple[float, float] | None = None
    sql: float = 0.0
    hydrate: float = 0.0
    source_rows: tuple[float, float] | None = None
    discounts: tuple[float, float] | None = None
    gather_end: float | None = None
    collapse: tuple[float, float] | None = None
    end: float = 0.0
    rows_read: int = 0
    models_hydrated: int = 0
    embedding_json_bytes: int = 0
    candidates_scored: int = 0
    snapshot_ids: int = 0
    entry_ids: int = 0
    in_loader: bool = False


@dataclass
class StageTimer:
    """Collects one query's stage marks while :func:`instrument` is active.

    ``query_vector`` set means the encoder is bypassed: ``_embed`` returns it
    instead of encoding the question (the encoder-free reference run).
    """

    query_vector: np.ndarray[Any, np.dtype[np.float32]] | None = None
    last_query: np.ndarray[Any, np.dtype[np.float32]] | None = None
    marks: _Marks = field(default_factory=_Marks)
    last_candidates: list[np.ndarray[Any, np.dtype[np.float32]]] = field(default_factory=list)
    keep_candidates: bool = False

    def begin(self) -> None:
        self.marks = _Marks(start=time.perf_counter())

    def finish(self, *, question_index: int, subject_id: str | None, results: int) -> QuerySample:
        m = self.marks
        m.end = time.perf_counter()
        total = m.end - m.start
        stages = dict.fromkeys(STAGES, 0.0)
        if m.embed is not None:
            stages["setup"] = m.embed[0] - m.start
            stages["embed"] = m.embed[1] - m.embed[0]
        if m.loader is not None:
            loader = m.loader[1] - m.loader[0]
            stages["sql_fetch"] = m.sql
            stages["hydrate"] = m.hydrate
            stages["decode"] = max(0.0, loader - m.sql - m.hydrate)
            if m.source_rows is not None:
                stages["filters"] = m.source_rows[0] - m.loader[1]
                stages["source_rows"] = m.source_rows[1] - m.source_rows[0]
        if m.discounts is not None:
            stages["support_discounts"] = m.discounts[1] - m.discounts[0]
            if m.gather_end is not None:
                stages["scoring"] = m.gather_end - m.discounts[1]
        if m.gather_end is not None and m.collapse is not None:
            stages["sort"] = m.collapse[0] - m.gather_end
            stages["collapse"] = m.collapse[1] - m.collapse[0]
        stages["other"] = max(0.0, total - sum(v for k, v in stages.items() if k != "other"))
        return QuerySample(
            question_index=question_index,
            subject_id=subject_id,
            total_seconds=total,
            stages=stages,
            rows_read=m.rows_read,
            embedding_json_bytes=m.embedding_json_bytes,
            models_hydrated=m.models_hydrated,
            candidates_scored=m.candidates_scored,
            snapshot_ids_looked_up=m.snapshot_ids,
            entry_ids_looked_up=m.entry_ids,
            results=results,
        )


class _TimedStream:
    """A streamed result whose partition fetches add to the SQL fetch stage."""

    def __init__(self, inner: Any, marks: _Marks) -> None:
        self._inner = inner
        self._marks = marks

    async def partitions(self, size: int | None = None) -> AsyncIterator[Any]:
        it = self._inner.partitions(size).__aiter__()
        while True:
            t0 = time.perf_counter()
            try:
                partition = await it.__anext__()
            except StopAsyncIteration:
                self._marks.sql += time.perf_counter() - t0
                return
            self._marks.sql += time.perf_counter() - t0
            yield partition

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _timed_span(
    fn: Callable[..., Awaitable[Any]],
    record: Callable[[float, float, Any], None],
) -> Callable[..., Awaitable[Any]]:
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = await fn(*args, **kwargs)
        record(t0, time.perf_counter(), out)
        return out

    return wrapper


@contextmanager
def instrument(timer: StageTimer) -> Iterator[StageTimer]:
    """Swap the query path's seams for timed wrappers; restore them on exit."""
    import particles.corpus.store as corpus_store
    import particles.operations.query.main as main
    from particles.store.particle_store import ParticleRow

    saved_main = {
        name: getattr(main, name)
        for name in (
            "_embed",
            "load_active_candidates",
            "_gather_scored",
            "load_source_rows",
            "stale_support_discounts",
            "_collapse_co_evidential_top_k",
        )
    }
    saved_to_model = ParticleRow.to_model
    saved_corpus = {
        name: getattr(corpus_store, name)
        for name in ("get_snapshot_source_meta", "get_source_types_for_entries")
    }

    def embed(text: str) -> np.ndarray[Any, np.dtype[np.float32]] | None:
        t0 = time.perf_counter()
        out: np.ndarray[Any, np.dtype[np.float32]] | None = (
            timer.query_vector if timer.query_vector is not None else saved_main["_embed"](text)
        )
        timer.marks.embed = (t0, time.perf_counter())
        timer.last_query = out
        return out

    def to_model(self: Any) -> Any:
        if not timer.marks.in_loader:
            return saved_to_model(self)
        t0 = time.perf_counter()
        out = saved_to_model(self)
        timer.marks.hydrate += time.perf_counter() - t0
        timer.marks.models_hydrated += 1
        timer.marks.embedding_json_bytes += len(self.embedding_json or "")
        return out

    async def loader(session: Any, *args: Any, **kwargs: Any) -> Any:
        # Patched on the class for the loader call only: the session type may
        # not take instance attributes, and nothing else runs meanwhile.
        session_type = type(session)
        saved_execute = session_type.execute
        saved_stream = session_type.stream

        async def execute(self: Any, *a: Any, **kw: Any) -> Any:
            t0 = time.perf_counter()
            result = await saved_execute(self, *a, **kw)
            timer.marks.sql += time.perf_counter() - t0
            return result

        async def stream(self: Any, *a: Any, **kw: Any) -> Any:
            # The candidate cache builds from a streamed result, so the SQL
            # fetch is the time spent waiting for each partition, interleaved
            # with the decode and hydration of the one before it.
            t0 = time.perf_counter()
            result = await saved_stream(self, *a, **kw)
            timer.marks.sql += time.perf_counter() - t0
            return _TimedStream(result, timer.marks)

        session_type.execute = execute
        session_type.stream = stream
        timer.marks.in_loader = True
        t0 = time.perf_counter()
        try:
            out = await saved_main["load_active_candidates"](session, *args, **kwargs)
        finally:
            timer.marks.loader = (t0, time.perf_counter())
            timer.marks.in_loader = False
            session_type.execute = saved_execute
            session_type.stream = saved_stream
        timer.marks.rows_read = len(out)
        if timer.keep_candidates:
            timer.last_candidates = [emb for _, emb in out]
        return out

    def on_gather(t0: float, t1: float, out: Any) -> None:
        timer.marks.gather_end = t1
        timer.marks.candidates_scored = len(out[0])

    def on_source_rows(t0: float, t1: float, _out: Any) -> None:
        timer.marks.source_rows = (t0, t1)

    def on_discounts(t0: float, t1: float, _out: Any) -> None:
        timer.marks.discounts = (t0, t1)

    def on_collapse(t0: float, t1: float, _out: Any) -> None:
        timer.marks.collapse = (t0, t1)

    async def snapshot_meta(session: Any, snapshot_ids: list[str]) -> Any:
        timer.marks.snapshot_ids = len(snapshot_ids)
        return await saved_corpus["get_snapshot_source_meta"](session, snapshot_ids)

    async def source_types(session: Any, entry_ids: list[str]) -> Any:
        timer.marks.entry_ids = len(entry_ids)
        return await saved_corpus["get_source_types_for_entries"](session, entry_ids)

    patches_main: dict[str, Any] = {
        "_embed": embed,
        "load_active_candidates": loader,
        "_gather_scored": _timed_span(saved_main["_gather_scored"], on_gather),
        "load_source_rows": _timed_span(saved_main["load_source_rows"], on_source_rows),
        "stale_support_discounts": _timed_span(saved_main["stale_support_discounts"], on_discounts),
        "_collapse_co_evidential_top_k": _timed_span(
            saved_main["_collapse_co_evidential_top_k"], on_collapse
        ),
    }
    for name, fn in patches_main.items():
        setattr(main, name, fn)
    setattr(ParticleRow, "to_model", to_model)  # noqa: B010 — a method swap mypy would refuse
    corpus_store.get_snapshot_source_meta = snapshot_meta
    corpus_store.get_source_types_for_entries = source_types
    try:
        yield timer
    finally:
        for name, fn in saved_main.items():
            setattr(main, name, fn)
        setattr(ParticleRow, "to_model", saved_to_model)  # noqa: B010
        for name, fn in saved_corpus.items():
            setattr(corpus_store, name, fn)


def cosine_reference(
    vectors: list[np.ndarray[Any, np.dtype[np.float32]]],
    query: np.ndarray[Any, np.dtype[np.float32]],
) -> tuple[float, float, float]:
    """``(loop, stack, matmul)`` seconds for the similarity alone over ``vectors``.

    ``loop`` calls the production :func:`~particles.embeddings.cosine_similarity`
    once per vector, as the scoring loop does; ``stack`` builds one matrix from
    the per-row arrays; ``matmul`` is one matrix-vector product over it. A
    reference measurement, outside every timed query.
    """
    from particles.embeddings import cosine_similarity

    t0 = time.perf_counter()
    for v in vectors:
        cosine_similarity(query, v)
    t1 = time.perf_counter()
    matrix = np.stack(vectors) if vectors else np.zeros((0, query.shape[0]), dtype=np.float32)
    t2 = time.perf_counter()
    sims = matrix @ query
    np.clip(sims, 0.0, 1.0, out=sims)
    t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2
