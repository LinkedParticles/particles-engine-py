# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plan, store-manifest and result models for the query-latency benchmark.

A run is a grid of **cells**: store size × mode (cold, warm) × variant
(unfiltered, subject-filtered). Each cell runs in its own process and writes one
:class:`CellResult` JSON when it finishes, so a re-launched run skips the cells
already on disk and a crash or reboot loses only the cell in flight.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

#: Bump when a seed's generated store is meant to change; it is part of the
#: cached store's file name, so an old store is never reused under a new shape.
GENERATOR_VERSION = "1"

#: The embedding dimension of the default encoder (all-MiniLM-L6-v2).
DIM = 384

DEFAULT_SEED = 20261017

#: 56k is the owner's own store (56,196 ACTIVE rows with embeddings on
#: 2026-10-17); the others bracket it.
DEFAULT_SIZES = (10_000, 56_000, 100_000, 300_000, 1_000_000)

#: The ``retrieve_ranked`` stages, in pipeline order. ``other`` is the residual
#: (total minus the named stages), reported so the stages always sum to the total.
STAGES = (
    "setup",
    "embed",
    "sql_fetch",
    "decode",
    "hydrate",
    "filters",
    "source_rows",
    "support_discounts",
    "scoring",
    "sort",
    "collapse",
    "other",
)


class Mode(StrEnum):
    COLD = "cold"
    """A fresh process answering one question. The encoder load is timed apart;
    the OS page cache is not purged (that needs root), so "cold" means a cold
    Python process, not a cold disk."""
    WARM = "warm"
    """One process answering many questions after one untimed warm-up query."""


class Variant(StrEnum):
    UNFILTERED = "unfiltered"
    """No filter: every ACTIVE row with an embedding is a candidate."""
    SUBJECT = "subject"
    """``subject_id`` set: the one filter the loader applies in SQL."""


class StoreManifest(BaseModel):
    """What one generated store holds. Written beside the store when it is built."""

    seed: int
    size: int
    generator_version: str = GENERATOR_VERSION
    active: int
    inactive: int
    subjects: int
    subject_links: int
    corpus_entries: int
    snapshots: int
    relations: int
    embedding_json_bytes: int
    """Total length of the ACTIVE rows' ``embedding_json`` text."""
    db_bytes: int
    build_seconds: float
    embedding_model_id: str
    fingerprint: str
    """SHA-256 over every particle id in insertion order: two builds with one
    seed must agree."""
    query_subjects: list[tuple[str, int]] = Field(default_factory=list)
    """``(subject_id, ACTIVE particles linked)`` for the filtered variant: the
    first subject of seeded random particles, so a popular subject is asked
    about as often as it is used."""


class CellSpec(BaseModel):
    """One cell, as the orchestrator hands it to the child process."""

    cell_id: str
    size: int
    seed: int
    mode: Mode
    variant: Variant
    store_path: str
    result_path: str
    encoder: bool
    """Encode the question with the real encoder (the mini run) or use a fixed
    seeded query vector (the hardware-reference run; no model is loaded)."""
    question_offset: int = 0
    """Index of the first question; cold repetitions each take a different one."""
    questions: int = 1
    budget_seconds: float = 0.0
    """Warm cells stop asking once this is spent (after ``min_questions``); 0 = no budget."""
    min_questions: int = 5
    top_k: int = 40


class QuerySample(BaseModel):
    """One timed ``retrieve_ranked`` call."""

    question_index: int
    subject_id: str | None = None
    total_seconds: float
    stages: dict[str, float]
    """Seconds per stage; keys are :data:`STAGES`."""
    rows_read: int
    """Rows the loader's SQL returned (each hydrated into a model)."""
    embedding_json_bytes: int
    """Characters of embedding JSON decoded."""
    models_hydrated: int
    candidates_scored: int
    """Candidates that reached the scoring loop (after the list filters)."""
    snapshot_ids_looked_up: int
    """Size of the ``IN`` list ``load_source_rows`` sent for snapshots."""
    entry_ids_looked_up: int
    results: int


class CosineReference(BaseModel):
    """The cosine arithmetic alone, over the last query's candidate vectors.

    Not a stage of the path: a reference for how much of the scoring loop the
    similarity itself costs, measured two ways.
    """

    vectors: int
    loop_seconds: float
    """One ``cosine_similarity`` call per vector, as the scoring loop makes them."""
    stack_seconds: float
    """Stacking the vectors into one matrix (what a vectorised path would pay per
    query if it kept today's storage)."""
    matmul_seconds: float
    """One matrix-vector product over the stacked matrix."""


class Environment(BaseModel):
    """The reproduction tuple a published figure must carry."""

    git_sha: str | None
    package_version: str
    python: str
    numpy: str
    sqlite: str
    sqlalchemy: str
    pydantic: str
    platform: str
    machine: str
    cpu: str
    cpu_count: int | None
    memory_bytes: int | None
    load_average_before: tuple[float, float, float] | None
    load_average_after: tuple[float, float, float] | None = None
    bench_lock: str | None
    """``PARTICLES_BENCH_LOCK`` as the bench lock exports it, or ``None`` when
    the run did not hold the lock."""
    config: str
    """Which config the child loaded (the harness pins compiled defaults)."""


class CellStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    """The child exited non-zero (an exception, or a kill by the OS)."""
    TIMEOUT = "timeout"
    """The orchestrator killed the child at ``--cell-timeout``."""
    MEMORY_GUARD = "memory_guard"
    """The orchestrator killed the child because swap or free disk crossed the
    guard (a stand-in for out-of-memory on a host that swaps instead)."""
    SKIPPED = "skipped"
    """Not run: an earlier cell of the same size and variant failed."""


class CellResult(BaseModel):
    """One cell's result, written on completion (or by the orchestrator on failure)."""

    spec: CellSpec
    status: CellStatus = CellStatus.OK
    started_at: str
    finished_at: str
    wall_seconds: float
    import_seconds: float | None = None
    """Importing the query path in the fresh child process."""
    encoder_load_seconds: float | None = None
    """Loading the encoder (encoder runs only); never inside a query's ``embed``."""
    warmup_seconds: float | None = None
    """The untimed warm-up query (warm cells)."""
    samples: list[QuerySample] = Field(default_factory=list)
    cosine_reference: CosineReference | None = None
    peak_rss_bytes: int | None = None
    """The child's own peak resident set (``getrusage``)."""
    sampled_peak_rss_bytes: int | None = None
    """The orchestrator's sampled peak of the child's RSS (also known for a
    killed child)."""
    environment: Environment | None = None
    error: str | None = None
    notes: list[str] = Field(default_factory=list)
