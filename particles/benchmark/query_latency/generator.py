# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The seeded synthetic store the query-latency benchmark reads.

Scoring cost does not depend on what the text says, so a store of random unit
vectors measures the retrieval path exactly as well as a real one, at any size,
with no LLM call and no encoder. What does matter is the *shape* of each row and
of the tables the path joins, so the generator copies the owner's store's
aggregate shape (measured 2026-10-17 over 56,196 ACTIVE rows; counts and column
sizes only, never content):

* every row is built as a real :class:`~particles.core.schema.Particle` and
  serialised by :meth:`ParticleRow.from_model`, so the columns, the embedding
  JSON format and the model id are the production writer's;
* subjects per particle follow the measured distribution (68% one, 22% two,
  5% none) and particles per subject a Zipf law (exponent 0.8) that reproduces
  the measured percentiles (median 2, p90 9, p99 about 50, one hub in the
  thousands at 56k);
* one corpus entry per 82 particles and one snapshot per 32, as measured;
* 21.6% of rows carry ``properties`` (the measured mix of gated subjects,
  document-meta scope, polarity and validity crumbs), 36.7% a structured claim;
* 20.4% extra rows that are not ACTIVE (stale, superseded, retracted), which
  the loader's SQL must skip, and CO_EVIDENTIAL pairs over 1.3% of rows.

A ``(seed, size)`` is the fixture: two builds produce the same rows in the same
order, and :attr:`StoreManifest.fingerprint` (a hash of every id) says so.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import event, insert
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from particles.benchmark.query_latency.schema import DIM, GENERATOR_VERSION, StoreManifest
from particles.core.schema import (
    AssertionModality,
    ClaimTerm,
    Confidence,
    ExtractorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    StructuredClaim,
    TermKind,
    UncertaintyNature,
    ValidKind,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.embeddings import get_embedding_model_id

#: Subjects linked per particle: P(0), P(1), … P(6). Measured; the tail past 6
#: (under 0.1%) is folded into 6.
SUBJECTS_PER_PARTICLE = (0.051, 0.683, 0.218, 0.036, 0.0076, 0.0027, 0.0022)
#: Distinct subjects per ACTIVE particle (14,519 / 56,196).
SUBJECTS_PER_ROW = 0.2584
#: Zipf exponent of particles per subject; fitted to the measured percentiles.
SUBJECT_ZIPF = 0.8
ROWS_PER_ENTRY = 82
ROWS_PER_SNAPSHOT = 32
#: Non-ACTIVE rows per ACTIVE row (11,458 / 56,241).
INACTIVE_PER_ACTIVE = 0.204
#: CO_EVIDENTIAL relations per ACTIVE row (724 / 56,196).
RELATIONS_PER_ROW = 0.0129
STRUCTURED_CLAIM_SHARE = 0.367
#: ``properties`` templates and their shares of all rows (21.6% in total).
PROPERTY_SHARES = (
    ("gated", 0.145),
    ("document_meta", 0.019),
    ("polarity", 0.015),
    ("valid_from", 0.037),
)
#: How many seeded query subjects the manifest records for the filtered variant.
QUERY_SUBJECTS = 64

CHUNK = 5_000

_WORD_TEXT = (
    "archive build cache client config corpus decision default deploy digest "
    "document engine export extractor field filter graph index ingest lens "
    "ledger memory model module network operator owner particle pipeline policy "
    "project provider query ranking record release review runtime schema "
    "session snapshot source store subject suite surface table token trust "
    "update vector version window worker"
)
_WORDS = _WORD_TEXT.split()

_K_PROBS = np.array(SUBJECTS_PER_PARTICLE) / sum(SUBJECTS_PER_PARTICLE)

_EPOCH = datetime(2023, 1, 1, tzinfo=UTC)
_SPAN_DAYS = 3 * 365


@dataclass
class _Shape:
    """The fixed tables a store's particles point into."""

    subject_ids: list[str]
    subject_cdf: np.ndarray[Any, np.dtype[np.float64]]
    entry_ids: list[str]
    snapshot_ids: list[str]
    snapshot_entry: list[int]
    linked: dict[str, int] = field(default_factory=dict)


def store_file_name(seed: int, size: int) -> str:
    """The cached store's file name: one file per ``(seed, size, generator)``."""
    return f"qlat-s{seed}-n{size}-g{GENERATOR_VERSION}.db"


def plan_counts(size: int) -> dict[str, int]:
    """The table sizes a store of ``size`` ACTIVE rows gets. Pure."""
    entries = max(1, round(size / ROWS_PER_ENTRY))
    return {
        "active": size,
        "inactive": round(size * INACTIVE_PER_ACTIVE),
        "subjects": max(1, round(size * SUBJECTS_PER_ROW)),
        "corpus_entries": entries,
        "snapshots": max(entries, round(size / ROWS_PER_SNAPSHOT)),
        "relations": round(size * RELATIONS_PER_ROW),
    }


def _uuid(rng: np.random.Generator) -> str:
    return str(uuid.UUID(bytes=rng.bytes(16), version=4))


def _unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    vecs = rng.standard_normal((n, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def query_vectors(seed: int, n: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """The fixed query vectors the encoder-free run scores against. Pure."""
    return _unit_vectors(np.random.default_rng([seed, 0x51]), n)


def questions(seed: int, n: int) -> list[str]:
    """Seeded question texts for the encoder run. Pure.

    Their wording does not matter to the cost (the store's vectors are random);
    they exist so the encoder encodes a realistic-length question.
    """
    rng = np.random.default_rng([seed, 0x9E])
    out = []
    for _ in range(n):
        words = [_WORDS[i] for i in rng.integers(0, len(_WORDS), size=int(rng.integers(5, 11)))]
        out.append(f"What do we know about the {' '.join(words)}?")
    return out


def _sentence(rng: np.random.Generator) -> str:
    target = int(np.clip(rng.normal(103, 30), 30, 240))
    words: list[str] = []
    length = 0
    while length < target:
        w = _WORDS[int(rng.integers(0, len(_WORDS)))]
        words.append(w)
        length += len(w) + 1
    return " ".join(words).capitalize() + "."


def _pick(roll: float, shares: list[float]) -> int | None:
    """Index of the share ``roll`` (in [0, 1)) falls in, or ``None`` past the last."""
    acc = 0.0
    for i, share in enumerate(shares):
        acc += share
        if roll < acc:
            return i
    return None


def _properties(rng: np.random.Generator) -> dict[str, object] | None:
    picked = _pick(float(rng.random()), [share for _, share in PROPERTY_SHARES])
    if picked is None:
        return None
    kind = PROPERTY_SHARES[picked][0]
    match kind:
        case "gated":
            names = [_WORDS[int(i)] for i in rng.integers(0, len(_WORDS), size=2)]
            return {"extraction:gated_subjects": names}
        case "document_meta":
            return {"extraction:scope": "DOCUMENT_META"}
        case "polarity":
            return {"extraction:polarity": "HYPOTHETICAL"}
        case _:
            return {
                "extraction:valid_from_basis": "coarse",
                "extraction:valid_from_cue": f"in {int(rng.integers(1990, 2026))}",
                "extraction:valid_from_granularity": "year",
                "extraction:valid_from_rule": "cadff2f3c5db",
            }


_INACTIVE = (
    (Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY, 0.79),
    (Status.SUPERSEDED, StatusReason.DUPLICATE_MERGED, 0.145),
    (Status.RETRACTED, StatusReason.SOURCE_RETRACTED, 0.065),
)


def _particle(rng: np.random.Generator, shape: _Shape, *, active: bool) -> Particle:
    k = int(rng.choice(len(SUBJECTS_PER_PARTICLE), p=_K_PROBS))
    picks = np.searchsorted(shape.subject_cdf, rng.random(k), side="right")
    subject_ids = list(dict.fromkeys(shape.subject_ids[int(i)] for i in picks))
    snap = int(rng.integers(0, len(shape.snapshot_ids)))
    asserted_at = _EPOCH + timedelta(seconds=float(rng.uniform(0, _SPAN_DAYS * 86_400)))
    properties = _properties(rng)
    valid_from = None
    if properties is not None and "extraction:valid_from_cue" in properties:
        year = int(str(properties["extraction:valid_from_cue"]).split()[-1])
        valid_from = datetime(year, 1, 1, tzinfo=UTC)
    status: Status = Status.ACTIVE
    reason: StatusReason | None = None
    if not active:
        # The last pair takes any roll past the listed shares.
        picked = _pick(float(rng.random()), [share for _, _, share in _INACTIVE])
        status, reason, _ = _INACTIVE[len(_INACTIVE) - 1 if picked is None else picked]
    structured = None
    if subject_ids and float(rng.random()) < STRUCTURED_CLAIM_SHARE:
        words = [_WORDS[int(i)] for i in rng.integers(0, len(_WORDS), size=6)]
        structured = StructuredClaim(
            subject=ClaimTerm(kind=TermKind.TOKEN, value=words[0]),
            predicate=ClaimTerm(kind=TermKind.TOKEN, value=f"{words[1]} {words[2]} {words[3]}"),
            object=ClaimTerm(kind=TermKind.LITERAL, value=f"{words[4]} {words[5]}"),
            subject_id=subject_ids[0],
            structurizer_id="content-structurizer",
            structurizer_version="1.0.0",
            generated_at=asserted_at,
        )
    modality_roll = float(rng.random())
    modality = (
        AssertionModality.CONSTITUTIVE
        if modality_roll < 0.029
        else AssertionModality.EVALUATIVE
        if modality_roll < 0.032
        else AssertionModality.FALSIFIABLE
    )
    return Particle(
        id=_uuid(rng),
        content=_sentence(rng),
        confidence=Confidence(
            value=round(float(rng.uniform(0.55, 0.98)), 2),
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        ),
        uncertainty_nature=(
            UncertaintyNature.ALEATORY
            if float(rng.random()) < 0.005
            else UncertaintyNature.EPISTEMIC
        ),
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=shape.entry_ids[shape.snapshot_entry[snap]],
                snapshot_id=shape.snapshot_ids[snap],
            )
        ],
        asserted_by="general-extractor",
        asserted_at=asserted_at,
        status=status,
        status_reason=reason,
        assertion_modality=modality,
        valid_from=valid_from,
        valid_kind=ValidKind.STATE if valid_from is not None else None,
        extractor_ref=ExtractorRef(
            name="general-extractor", version=f"0.{int(rng.integers(12, 19))}.0"
        ),
        subject_ids=subject_ids,
        properties=properties,
        structured_claim=structured,
    )


def _row(particle: Particle, vector: np.ndarray[Any, np.dtype[np.float32]]) -> dict[str, Any]:
    from particles.store.particle_store import ParticleRow

    orm = ParticleRow.from_model(particle, vector.tolist())
    row = {c.key: getattr(orm, c.key) for c in ParticleRow.__table__.columns}
    row["born_retired"] = False
    row["retired_at"] = None if particle.status is Status.ACTIVE else particle.asserted_at
    return row


def _shape(rng: np.random.Generator, counts: dict[str, int]) -> _Shape:
    n_subjects = counts["subjects"]
    weights = 1.0 / np.arange(1, n_subjects + 1, dtype=np.float64) ** SUBJECT_ZIPF
    cdf = np.cumsum(weights / weights.sum())
    cdf[-1] = 1.0
    entries = [_uuid(rng) for _ in range(counts["corpus_entries"])]
    snapshots = [_uuid(rng) for _ in range(counts["snapshots"])]
    # Every entry has at least one snapshot; the rest land on random entries.
    snapshot_entry = list(range(len(entries))) + [
        int(i) for i in rng.integers(0, len(entries), size=len(snapshots) - len(entries))
    ]
    return _Shape(
        subject_ids=[_uuid(rng) for _ in range(n_subjects)],
        subject_cdf=cdf,
        entry_ids=entries,
        snapshot_ids=snapshots,
        snapshot_entry=snapshot_entry,
    )


def _status_plan(rng: np.random.Generator, counts: dict[str, int]) -> Iterator[bool]:
    """Interleave the inactive rows among the active ones, as a store accrues them."""
    total = counts["active"] + counts["inactive"]
    remaining_active = counts["active"]
    for i in range(total):
        left = total - i
        is_active = float(rng.random()) < remaining_active / left
        remaining_active -= is_active
        yield is_active


async def _insert(conn: AsyncConnection, table: Any, rows: list[dict[str, Any]]) -> None:
    if rows:
        await conn.execute(insert(table), rows)


async def bring_schema_current(path: Path) -> None:
    """Add to a cached store the tables, indexes and triggers added since it was built.

    A store is written once by ``create_all`` and reused for every later run,
    so a schema change since (a new table, a new index, the change journal's
    triggers) would otherwise be measured against a store that lacks it. Only
    additions are made, and the rows are untouched, so the fixture and its
    fingerprint stay the same. Idempotent; a no-op on a current store.
    """
    import particles._orm_modules  # noqa: F401 — registers every ORM table on Base.metadata
    from particles.db import Base
    from particles.store.change_journal import install_journal

    def upgrade(sync_conn: Any) -> None:
        Base.metadata.create_all(sync_conn, checkfirst=True)
        for table in Base.metadata.sorted_tables:
            for index in table.indexes:
                index.create(sync_conn, checkfirst=True)
        install_journal(sync_conn)

    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(upgrade)
    finally:
        await engine.dispose()


async def build_store(
    path: Path,
    *,
    seed: int,
    size: int,
    progress: Callable[[str], None] | None = None,
) -> StoreManifest:
    """Build the ``(seed, size)`` store at ``path``, or return the cached one.

    The store is written to a ``.building`` file and renamed into place only when
    complete, with its manifest beside it, so an interrupted build is never
    mistaken for a finished one and the next launch starts it over.
    """
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() and manifest_path.exists():
        manifest = StoreManifest.model_validate_json(manifest_path.read_text())
        if (manifest.seed, manifest.size, manifest.generator_version) == (
            seed,
            size,
            GENERATOR_VERSION,
        ):
            await bring_schema_current(path)
            return manifest

    import particles._orm_modules  # noqa: F401 — registers every ORM table on Base.metadata
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.db import Base
    from particles.store.particle_store import ParticleRow, ProvenanceEdgeRow
    from particles.store.relation_store import ParticleRelationRow
    from particles.store.subject_store import ParticleSubjectRow, SubjectRow

    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    building = path.with_suffix(".building")
    building.unlink(missing_ok=True)
    rng = np.random.default_rng([seed, size, int(GENERATOR_VERSION)])
    counts = plan_counts(size)
    shape = _shape(rng, counts)
    created = _EPOCH

    engine = create_async_engine(f"sqlite+aiosqlite:///{building}")

    def _fast_build(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=OFF")
        cursor.execute("PRAGMA synchronous=OFF")
        cursor.close()

    event.listen(engine.sync_engine, "connect", _fast_build)
    digest = hashlib.sha256()
    embedding_bytes = 0
    links = 0
    relations = 0
    # The filtered variant's subjects: the first subject of seeded random ACTIVE
    # particles, so a subject is asked about in proportion to its use.
    sample_at = {
        int(pos): order
        for order, pos in enumerate(
            rng.choice(size, size=min(size, 4 * QUERY_SUBJECTS), replace=False)
        )
    }
    sampled: dict[int, str] = {}
    active_seen = 0
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await _insert(
                conn,
                CorpusEntryRow.__table__,
                [
                    {
                        "entry_id": eid,
                        "uri_r": f"file:///synthetic/source-{i}.md",
                        "source_type": "CONVERSATION" if i % 4 else "LOCAL_MARKDOWN",
                        "mutability": "STABLE",
                        "fetch_policy": "NEVER",
                        "created_at": created,
                        "deposited_by": "query-latency-generator",
                        "tags_json": "[]",
                    }
                    for i, eid in enumerate(shape.entry_ids)
                ],
            )
            snap_rows = []
            for i, sid in enumerate(shape.snapshot_ids):
                published = (
                    _EPOCH + timedelta(days=float(rng.uniform(0, _SPAN_DAYS)))
                    if float(rng.random()) < 0.79
                    else None
                )
                snap_rows.append(
                    {
                        "snapshot_id": sid,
                        "entry_id": shape.entry_ids[shape.snapshot_entry[i]],
                        "captured_at": created,
                        "content_hash": hashlib.sha256(sid.encode()).hexdigest(),
                        "warc_record_type": "RESPONSE",
                        "extraction_status": "COMPLETE",
                        "content_published_at": published,
                        "extraction_attempts": 1,
                        "resume_whole": False,
                    }
                )
            await _insert(conn, SnapshotRow.__table__, snap_rows)

        statuses = _status_plan(rng, counts)
        total = counts["active"] + counts["inactive"]
        relation_budget = counts["relations"]
        done = 0
        while done < total:
            n = min(CHUNK, total - done)
            vectors = _unit_vectors(rng, n)
            particle_rows: list[dict[str, Any]] = []
            edge_rows: list[dict[str, Any]] = []
            link_rows: list[dict[str, Any]] = []
            chunk_active: list[str] = []
            for j in range(n):
                active = next(statuses)
                p = _particle(rng, shape, active=active)
                row = _row(p, vectors[j])
                particle_rows.append(row)
                digest.update(p.id.encode())
                ref = p.provenance[0]
                edge_rows.append(
                    {
                        "particle_id": p.id,
                        "corpus_entry_id": ref.corpus_entry_id,
                        "snapshot_id": ref.snapshot_id,
                        "chunk_hash": None,
                    }
                )
                for sid in p.subject_ids:
                    link_rows.append({"particle_id": p.id, "subject_id": sid})
                if active:
                    if active_seen in sample_at and p.subject_ids:
                        sampled[sample_at[active_seen]] = p.subject_ids[0]
                    active_seen += 1
                    chunk_active.append(p.id)
                    embedding_bytes += len(row["embedding_json"])
                    for sid in p.subject_ids:
                        shape.linked[sid] = shape.linked.get(sid, 0) + 1
            rel_rows: list[dict[str, Any]] = []
            want = min(relation_budget, round(len(chunk_active) * RELATIONS_PER_ROW))
            for _ in range(want if len(chunk_active) >= 2 else 0):
                a, b = sorted(rng.choice(len(chunk_active), size=2, replace=False).tolist())
                rel_rows.append(
                    {
                        "particle_a": chunk_active[a],
                        "particle_b": chunk_active[b],
                        "relation_type": "CO_EVIDENTIAL",
                        "created_by": "query-latency-generator",
                        "created_at": created,
                        "confidence": 1.0,
                    }
                )
            rel_rows = list({(r["particle_a"], r["particle_b"]): r for r in rel_rows}.values())
            relation_budget -= len(rel_rows)
            relations += len(rel_rows)
            links += len(link_rows)
            async with engine.begin() as conn:
                await _insert(conn, ParticleRow.__table__, particle_rows)
                await _insert(conn, ProvenanceEdgeRow.__table__, edge_rows)
                await _insert(conn, ParticleSubjectRow.__table__, link_rows)
                await _insert(conn, ParticleRelationRow.__table__, rel_rows)
            done += n
            if progress is not None and (done // CHUNK) % 20 == 0:
                progress(f"  n={size}: {done}/{total} rows")

        async with engine.begin() as conn:
            await _insert(
                conn,
                SubjectRow.__table__,
                [
                    {
                        "id": sid,
                        "canonical_name": f"subject {i}",
                        "aliases_json": "[]",
                        "external_ids_json": "[]",
                        "created_at": created,
                        "asserted_by": "query-latency-generator",
                    }
                    for i, sid in enumerate(shape.subject_ids)
                    if sid in shape.linked
                ],
            )
    finally:
        await engine.dispose()

    query_subjects = [(sid, shape.linked[sid]) for _, sid in sorted(sampled.items())][
        :QUERY_SUBJECTS
    ]

    building.rename(path)
    manifest = StoreManifest(
        seed=seed,
        size=size,
        active=counts["active"],
        inactive=counts["inactive"],
        subjects=len(shape.linked),
        subject_links=links,
        corpus_entries=counts["corpus_entries"],
        snapshots=counts["snapshots"],
        relations=relations,
        embedding_json_bytes=embedding_bytes,
        db_bytes=path.stat().st_size,
        build_seconds=round(time.perf_counter() - started, 1),
        embedding_model_id=get_embedding_model_id(),
        fingerprint=digest.hexdigest(),
        query_subjects=query_subjects,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest
