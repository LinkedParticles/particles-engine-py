# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Memory-benchmark runner (§3): deposit → extract → query → judge.

The system under test is the pipeline the agent-memory wedge actually runs —
default config, default thresholds — not a benchmark-tuned variant. Per
question:

1. an **ephemeral scratch store** (a throwaway SQLite file in the run's
   working directory) is created, so no question's distractor set is
   contaminated by another's and no user store is ever touched;
2. each haystack session is deposited as a ``CONVERSATION`` corpus entry
   under ``longmemeval://<question_id>/session/<session_id>`` with the
   session's haystack date as the content date;
3. the standard extract runs, wrapped in a **candidate cache** keyed by
   session content hash — candidate production is Client-layer store-free,
   so repeated haystack sessions cost one LLM pass across the
   whole run while §6.6 reconciliation still runs per question store;
4. ``retrieve_ranked`` (the respond-free half of §9.3 query) returns the
   top-k, which is scored through real provenance chains against the labeled
   evidence sessions (condition i) — a derived particle resolves
   transitively through its premise links, so promotion is not penalized by
   a dead-ended chain (see :func:`_sessions_by_particle`).
   Zero-evidence abstention questions are unscoreable here: their
   per-question rows are marked, they are excluded from every retrieval
   aggregate, and the report discloses the excluded
   count (correction v1.74.2) — they stay fully in the QA family, which the
   dataset's protocol scores;
5. conditions ii–iv answer with **one pinned model** via the new
   ``llm.benchmark_answer`` purpose — a mismatched resolved model id raises
   :class:`SameModelViolation` (the same-model comparison is the validity
   condition, enforced structurally) — and the existing ``llm.benchmark``
   judge, pinned the same way, scores each answer per the dataset's
   per-question-type autoeval protocol (Anthropic judge + disclosure;
   owner-resolved 2026-07-12). The resolved ``llm.extraction`` and embedding
   model ids are recorded on the run tuple at run start, and a mid-run
   extraction-model change is refused (§5 comparability; correction
   v1.74.2).

Report-only: the harness never writes a user store; scratch stores are
deleted unless the caller keeps them. That includes the store's *content*:
deposits write blobs store-adjacent to the configured store, not to
the scratch SQLite file, so the run points ``storage.blob_dir`` at its own
work directory for its duration (:func:`isolated_blob_dir`).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import shutil
import tempfile
import time
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from particles.benchmark.memory.comparators import (
    MEMORY_KINDS,
    NotesWriter,
    run_comparator_question,
)
from particles.benchmark.memory.metrics import (
    accuracy_by_type,
    mean_by_type,
    parse_judge_verdict,
    precision_at_k,
    qa_accuracy,
    recall_at_k,
)
from particles.benchmark.memory.schema import (
    QA_CONDITIONS,
    QA_EXCLUSION_BUDGET,
    QA_EXCLUSION_INFRA,
    QA_EXCLUSION_KINDS,
    QA_EXCLUSION_UNRECORDED,
    MemoryBenchmarkReport,
    MemoryQuestion,
    MemorySession,
    QaConditionMetrics,
    QaQuestionResult,
    RetrievalQuestionResult,
    RetrievalStageMetrics,
    RunSelection,
)
from particles.config import ProviderSelection, TokenPrice, get_config
from particles.core.schema import (
    Mutability,
    Particle,
    QueryRequest,
    Snapshot,
    SourceType,
    SuggestMode,
)
from particles.core.scoring import recency_factor_from_params
from particles.corpus.deposit import deposit_text_versioned
from particles.embeddings import get_embedding_model_id
from particles.extraction.general import ExtractionResult
from particles.extraction.registry import ExtractorPlugin, select_extractor

# Tests patch these bindings on THIS module (tests/AGENTS.md § Mocking
# strategy: patch the caller's binding for module-top imports) — e.g.
# ``patch("particles.benchmark.memory.runner.extract_snapshot", ...)``.
from particles.ingest.pipeline import extract_snapshot
from particles.llm.registry import (
    CompletionRequest,
    EmptyCompletionError,
    LLMPurpose,
    complete,
    complete_many,
    get_provider,
)
from particles.operations.abstraction import is_derived, premise_ids_of, run_abstraction_pass
from particles.operations.consolidation import run_consolidation
from particles.operations.links_suggest import suggest_co_evidential
from particles.operations.query.main import retrieve_ranked
from particles.operations.query.source_info import SourceRow, load_source_rows
from particles.store.particle_store import get_particles_by_ids

log = logging.getLogger(__name__)

URI_SCHEME = "longmemeval"

#: Per-call output budgets. max_tokens is per-call by design.
#: Both are deliberately loose: an adaptive-thinking model (Sonnet 5+) spends
#: its thinking from the same budget as the answer and returns *no text block*
#: when the cap lands inside the thinking — the 2026-08-16 N=150 run lost
#: three full-context answers at 1024 and two abstention-judge verdicts at 16
#: that way, each scored incorrect. A higher cap never changes a reply that
#: already finished under the lower one, so raising it preserves comparability.
_ANSWER_MAX_TOKENS = 4096
_JUDGE_MAX_TOKENS = 1024

#: The answer scaffold's system turn, by version. The scaffold is shared by
#: conditions ii–iv (only the context block differs), so a change to it moves
#: every QA column at once and breaks comparability with runs made under the
#: previous text — which is why the version is a config knob
#: (``benchmark_memory.answer_scaffold``), recorded on the run tuple, and part
#: of the checkpoint key. v1 is the inaugural 2026-08-16 table's text; set the
#: knob to 1 to reproduce that table.
#:
#: v2 keeps the scaffold **question-type-blind** (the product never sees a
#: type label) and adds reader guidance for the failure classes the inaugural
#: run's error buckets share with the CoreSpeed LongMemEval write-up
#: (https://corespeed.io/blog/how-we-took-longmemeval-from-80-to-94-without-touching-retrieval,
#: whose label-free arm lost only 0.4 of its 2.4-point instruction gain):
#: conditional abstention (a factual gap is a refusal; a recommendation is a
#: personalisation task, never a refusal), an enumerate-merge-qualify-count
#: protocol for cross-session counting (Particles' ``subjects:`` field is the
#: merge key that write-up's readers had to reconstruct from prose), explicit
#: date arithmetic against the question date, and latest-dated-wins for
#: updated facts. The inaugural particles path answered 1 of 7 abstention
#: questions and 3 of 9 preference questions correctly, with retrieval recall
#: 1.0 on most of those failures — reader-side losses, not retrieval.
_ANSWER_SYSTEM_V1 = (
    "You are a helpful assistant answering a question about a user's prior "
    "chat history with an assistant. Answer concisely from the provided "
    "context. If the context does not contain the information needed, say "
    "that you don't have that information — do not guess."
)
_ANSWER_SYSTEM_V2 = (
    "You are a helpful assistant answering a question about a user's prior "
    "chat history with an assistant. Answer concisely from the provided "
    "context, following these rules:\n"
    "- Factual questions: if the context does not contain the information "
    "needed, say that you don't have that information — do not guess, and do "
    "not substitute a similar but different fact from the context.\n"
    "- Recommendation or advice questions: the user is asking you to apply "
    "what you know about them. Ground the answer in the preferences, "
    "constraints, and circumstances the context records about the user, even "
    "when the context says nothing about the specific subject asked about. "
    "Do not refuse for lack of subject-specific evidence.\n"
    "- Counting or enumeration questions: first list every candidate with its "
    "date, merge mentions that refer to the same real-world thing (lines that "
    "share a subject are the same thing), apply the question's qualifiers, "
    "and only then count.\n"
    "- Date and duration questions: treat the question date as today and "
    "compute from the dates in the context rather than estimating.\n"
    "- When the context records a value that later changed, answer with the "
    "latest dated statement."
)
_ANSWER_SYSTEMS: dict[int, str] = {1: _ANSWER_SYSTEM_V1, 2: _ANSWER_SYSTEM_V2}


def _answer_system() -> str:
    """The answer scaffold's system turn for the configured scaffold version."""
    from particles.config import get_config

    return _ANSWER_SYSTEMS[get_config().benchmark_memory.answer_scaffold]


class SameModelViolation(RuntimeError):
    """A pinned model resolution changed mid-run (§2/§5).

    Three pins, all enforced by refusal — the runner never
    warns-and-continues:

    * the **answer model** (``llm.benchmark_answer``) across conditions
      ii–iv — a same-model comparison is the validity condition of the QA
      family;
    * the **extraction model** (``llm.extraction``) across questions — the
      store's contents are a function of it, so a mid-run change silently
      splits the run into two incomparable pipelines (correction v1.74.2);
    * the **judge model** (``llm.benchmark``) across judged answers — one
      judge per table is what makes its accuracies comparable within it
      (correction v1.74.2).
    """


# ---------------------------------------------------------------------------
# URI scheme + session rendering
# ---------------------------------------------------------------------------


def session_uri(question_id: str, session_id: str) -> str:
    """The per-session URI-R: ``longmemeval://<question_id>/session/<session_id>``."""
    return f"{URI_SCHEME}://{question_id}/session/{session_id}"


def session_id_from_uri(uri_r: str | None) -> str | None:
    """Invert :func:`session_uri` — the provenance-chain scorer's last hop."""
    if not uri_r or not uri_r.startswith(f"{URI_SCHEME}://"):
        return None
    rest = uri_r[len(f"{URI_SCHEME}://") :]
    parts = rest.split("/session/", 1)
    if len(parts) != 2 or not parts[1]:
        return None
    return parts[1]


def render_session_text(session: MemorySession) -> str:
    """One haystack session as the speaker-turn transcript the pipeline deposits.

    The same material shape the harvester produces on SessionEnd —
    the benchmark exercises the real ingestion path, not a benchmark shim.
    """
    lines: list[str] = []
    if session.date:
        lines.append(f"Session date: {session.date}")
        lines.append("")
    for turn in session.turns:
        lines.append(f"{turn.role}: {turn.content}")
    return "\n".join(lines)


_DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d",
    "%Y-%m-%d",
)


def parse_session_date(raw: str | None) -> datetime | None:
    """Parse a LongMemEval session/question date string; ``None`` when unparseable.

    The dataset stamps dates like ``2023/05/20 (Sat) 02:21``. The parsed
    value feeds recency decay exactly as real harvests do; an unparseable
    date degrades to no content date rather than aborting the question.
    """
    if not raw:
        return None
    text = raw.strip()
    try:
        return datetime.fromisoformat(text).replace(tzinfo=UTC)
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Candidate cache (step 2)
# ---------------------------------------------------------------------------


class CachingExtractor:
    """Wrap an extractor with a content-hash candidate cache.

    Candidate production is Client-layer and store-free: the
    ``CandidateParticle[]`` for a session depends only on the session bytes,
    so the cache key is the snapshot content hash. Haystack sessions repeat
    across questions, turning the dominant LLM cost from
    (questions × sessions) into (unique sessions). Results are deep-copied on
    both store and replay — the pipeline mutates candidates in place
    (fingerprint stamping, subject gating), and one store's mutations must
    never leak into another's reconciliation.

    Only clean results are cached (``transient_error_count == 0``); a partial
    API failure is retried on the next occurrence rather than replayed.
    """

    def __init__(self, inner: ExtractorPlugin) -> None:
        self._inner = inner
        self._cache: dict[str, ExtractionResult] = {}
        # Per-content-hash locks: under a concurrent run (``concurrency > 1``)
        # two questions sharing a haystack session must not both pay for its
        # extraction — the second waits and replays the first's cached result.
        self._locks: dict[str, asyncio.Lock] = {}
        self.hits = 0
        self.misses = 0
        self.EXTRACTOR_ID = inner.EXTRACTOR_ID
        self.EXTRACTOR_VERSION = inner.EXTRACTOR_VERSION

    def accepts(self, source_type: str) -> bool:
        return self._inner.accepts(source_type)

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        key = snapshot.content_hash
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return copy.deepcopy(cached)
        async with self._locks.setdefault(key, asyncio.Lock()):
            cached = self._cache.get(key)
            if cached is not None:  # a concurrent holder filled it while we waited
                self.hits += 1
                return copy.deepcopy(cached)
            result = await self._inner.extract(snapshot, content, **kwargs)
            self.misses += 1
            if result.transient_error_count == 0:
                self._cache[key] = copy.deepcopy(result)
            return result


# ---------------------------------------------------------------------------
# Ephemeral scratch store
# ---------------------------------------------------------------------------


@asynccontextmanager
async def scratch_store(db_path: Path) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """A throwaway per-question SQLite store — never a registered user store.

    Creates the full ORM schema in a fresh file, yields a session factory,
    and disposes the engine on exit. The caller owns file deletion (kept
    only under an explicit ``--store-dir``).
    """
    from sqlalchemy import event
    from sqlalchemy.pool import NullPool

    import particles._orm_modules  # noqa: F401 — registers every ORM table on Base.metadata
    from particles.db import Base, _sqlite_set_pragmas

    db_path.parent.mkdir(parents=True, exist_ok=True)
    # NullPool: one connection per session, no pool cap. Pooled extraction
    # (``pooled=True``) parks a task per haystack session — ~50 per question,
    # times ``concurrency`` — and the default AsyncAdaptedQueuePool's 5 + 10
    # connections time out after 30 s under that fan-out (measured 2026-08-16:
    # every question failed with ``QueuePool limit ... reached``). A throwaway
    # store has nothing to gain from pooling. The same WAL + busy_timeout
    # pragmas the real engine sets (``particles.db.get_engine``) make the
    # concurrent writers block briefly instead of raising ``database is locked``.
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", poolclass=NullPool)
    event.listen(engine.sync_engine, "connect", _sqlite_set_pragmas)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


BLOB_SUBDIR = "blobs"
"""The run-local blob directory under a benchmark work dir (see :func:`isolated_blob_dir`)."""


@contextlib.contextmanager
def isolated_blob_dir(base: Path) -> Iterator[Path]:
    """Point ``storage.blob_dir`` at ``<base>/blobs`` for the block; yield that path.

    A scratch store isolates the SQLite rows but not the corpus content:
    :func:`~particles.corpus.deposit.save_blob` resolves ``blob_dir``
    store-adjacent to the *configured* default store, never to the
    scratch file. Without this, every haystack session a run deposits lands as
    a content-addressed blob in the operator's real ``corpus_blobs`` — breaking
    the report-only rule. An absolute path is honoured as-is by
    ``resolve_store_adjacent_path``, and the prior value is restored on exit,
    success or failure.

    The directory lives beside the stores deliberately: a kept store set
    (``--store-dir``) carries its blobs with it, so a ``--reuse-stores`` replay
    reads its snapshots from the same place the preparing run wrote them.
    Shared by every harness that builds scratch stores (the memory and
    memory-rot benchmarks).
    """
    blob_dir = (base / BLOB_SUBDIR).resolve()
    storage = get_config().storage
    prior = storage.blob_dir
    storage.blob_dir = str(blob_dir)
    try:
        yield blob_dir
    finally:
        storage.blob_dir = prior


def _remove_scratch_files(db_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Estimate / confirm gate (mirrored)
# ---------------------------------------------------------------------------


class EstimateCostComponent(BaseModel):
    """One priced slice of :class:`MemoryRunEstimate` — write, answer, or judge."""

    name: str
    #: The resolved ``llm.<purpose>`` model id the slice is priced against.
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    #: Whether the slice rides the Message Batches API (``--pooled`` /
    #: ``--batch-qa`` with ``llm.batch.enabled``), so the discount applies.
    batched: bool = False
    #: ``None`` when ``price_per_mtok`` has no entry for ``model``.
    cost_usd: float | None = None


class MemoryRunEstimate(BaseModel):
    """Projected LLM cost of a run, computed before any call is made.

    Extraction calls are projected from *unique* session byte counts (the
    candidate cache dedupes repeats) via the same chunk math as the audit's
    estimate; QA + judge calls are three each per question. ``estimated_tokens``
    is a magnitude signal (prompt bytes over a chars-per-token factor, ~4 by
    default and per-model via ``benchmark_memory.chars_per_token_by_model``)
    dominated by the full-context condition, which re-sends each question's
    whole haystack.
    """

    questions: int = 0
    unique_sessions: int = 0
    total_session_chars: int = 0
    estimated_extraction_calls: int = 0
    estimated_answer_calls: int = 0
    estimated_judge_calls: int = 0
    #: Upper bound on the calls made by an opted-in store-mutating pass
    #: (``--consolidation``). Bounded, not projected: the per-store caps
    #: (``consolidation.max_reconcile_probes`` + ``audit.max_contradiction_probes``)
    #: are what a saturated store spends, and a sparse one spends less.
    estimated_pass_calls: int = 0
    estimated_llm_calls: int = 0
    #: Projected *input* tokens (session bytes over the chars-per-token
    #: factor disclosed in ``assumption_sources``). The name predates the
    #: output projection below and is kept as-is; read it as the input side.
    #: It is a floor: the extraction system prompt is not counted (the
    #: 2026-09 provider survey measured it as the larger half of an
    #: extraction prompt).
    estimated_tokens: int = 0
    #: Cost components this projection cannot bound, disclosed rather than
    #: silently omitted — the confirm gate is worthless if a flag can route
    #: spend around it.
    unbounded_components: list[str] = Field(default_factory=list)
    #: Projected *output* tokens, per side and in total. Output is the larger
    #: half of an extraction run's bill (the survey measured ~6.7k output per
    #: claude-sonnet-5 extraction call against ~3.9k input), which is why an
    #: input-only projection under-estimated the inaugural run by nearly half.
    #: Driven by the ``benchmark_memory.estimate_output_tokens_per_*_call``
    #: assumptions, echoed in ``output_assumptions``. Ablation-pass probes are
    #: counted as calls only — their output is a short verdict each.
    estimated_write_output_tokens: int = 0
    estimated_answer_output_tokens: int = 0
    estimated_judge_output_tokens: int = 0
    estimated_output_tokens: int = 0
    #: The per-call output assumptions used, keyed ``extraction`` / ``answer``
    #: / ``judge`` — disclosed with the figures so a reader can re-derive them.
    output_assumptions: dict[str, int] = Field(default_factory=dict)
    #: Where each model-dependent figure came from, keyed
    #: ``extraction_output`` (the per-extraction-call output figure, keyed on
    #: the extraction model), ``write_chars_per_token`` (extraction model) and
    #: ``answer_chars_per_token`` (answer model) — each a per-model mapping
    #: entry or the scalar fallback, and the rendering says which.
    assumption_sources: dict[str, EstimateAssumption] = Field(default_factory=dict)
    #: Per-component dollar lines (write / answer / judge). Empty when the
    #: component makes no call.
    cost_components: list[EstimateCostComponent] = Field(default_factory=list)
    #: Sum of the priced components, or ``None`` when any component that makes
    #: a call has no price configured — a partial dollar figure reads as a
    #: total, so none is printed instead.
    estimated_cost_usd: float | None = None
    #: Resolved model ids for which ``benchmark_memory.price_per_mtok`` holds
    #: no entry. Rendered as "no price configured for <model>".
    unpriced_models: list[str] = Field(default_factory=list)


class EstimateAssumption(BaseModel):
    """One model-dependent estimate figure and where it came from.

    Every such figure is disclosed beside the number it produced, because a
    projection that silently used the Sonnet 5 output figure for a Haiku arm
    (or the 4-chars/token rule for a new-tokenizer model) is off by a third
    to a half and reads exactly like one that is right.
    """

    value: float
    #: The resolved model id the lookup was keyed on.
    model: str
    #: The ``benchmark_memory`` field (and, for a mapping hit, the key) that
    #: supplied the figure.
    source: str
    #: True when a ``*_by_model`` mapping entry supplied it; False when the
    #: scalar fallback did.
    per_model: bool = False

    def describe(self) -> str:
        """The provenance clause the rendered lines carry."""
        if self.per_model:
            return f"per-model entry for {self.model}"
        return f"scalar fallback for {self.model}, no per-model entry"


_T = TypeVar("_T")


def _by_model(mapping: Mapping[str, _T], provider: str, model: str) -> tuple[_T | None, str | None]:
    """``(entry, key)`` for a resolved selection: ``provider:model`` first, then the bare id.

    The one key-resolution convention every ``benchmark_memory`` per-model
    mapping shares — ``price_per_mtok``,
    ``estimate_output_tokens_per_extraction_call_by_model``,
    ``chars_per_token_by_model`` — so a key that prices a model is the key
    that selects its token assumptions, and the lookups cannot diverge.
    Membership, not truthiness: a legitimate ``0`` entry is an entry.
    """
    for key in (f"{provider}:{model}", model):
        if key in mapping:
            return mapping[key], key
    return None, None


def _lookup_price(provider: str, model: str) -> TokenPrice | None:
    """``price_per_mtok`` entry for a resolved selection: ``provider:model`` first, then bare id."""
    price, _ = _by_model(get_config().benchmark_memory.price_per_mtok, provider, model)
    return price


def _extraction_output_assumption(selection: ProviderSelection) -> EstimateAssumption:
    """Output tokens per extraction call for ``selection``: the per-model entry, else the scalar."""
    cfg = get_config().benchmark_memory
    value, key = _by_model(
        cfg.estimate_output_tokens_per_extraction_call_by_model,
        selection.provider,
        selection.model,
    )
    if value is not None:
        return EstimateAssumption(
            value=value,
            model=selection.model,
            source=f"estimate_output_tokens_per_extraction_call_by_model[{key}]",
            per_model=True,
        )
    return EstimateAssumption(
        value=cfg.estimate_output_tokens_per_extraction_call,
        model=selection.model,
        source="estimate_output_tokens_per_extraction_call",
    )


def _chars_per_token_assumption(selection: ProviderSelection) -> EstimateAssumption:
    """Chars per input token for ``selection``: the per-model entry, else the scalar.

    The scalar (4.0 by default) is the pre-4.7 tokenizer's rule of thumb;
    models on the newer tokenizer (Claude 4.7 and later, claude-sonnet-5
    among them) need a per-model entry or their input projection runs ~30 %
    low and the window check overstates its margin.
    """
    cfg = get_config().benchmark_memory
    value, key = _by_model(cfg.chars_per_token_by_model, selection.provider, selection.model)
    if value is not None:
        return EstimateAssumption(
            value=value,
            model=selection.model,
            source=f"chars_per_token_by_model[{key}]",
            per_model=True,
        )
    return EstimateAssumption(
        value=cfg.chars_per_token, model=selection.model, source="chars_per_token"
    )


def _cost_component(
    name: str,
    purpose: LLMPurpose,
    *,
    calls: int,
    input_tokens: int,
    output_tokens: int,
    batched: bool,
) -> EstimateCostComponent:
    """Price one slice against the model ``llm.<purpose>`` resolves to right now."""
    selection = get_config().llm.for_purpose(purpose)
    price = _lookup_price(selection.provider, selection.model)
    cost: float | None = None
    if price is not None:
        multiplier = 1.0 - get_config().benchmark_memory.batch_discount if batched else 1.0
        cost = multiplier * (
            input_tokens * price.input / 1_000_000 + output_tokens * price.output / 1_000_000
        )
    return EstimateCostComponent(
        name=name,
        model=selection.model,
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        batched=batched,
        cost_usd=cost,
    )


def estimate_run(
    questions: list[MemoryQuestion],
    *,
    qa: bool = True,
    memory: str = "particles",
    baselines: bool = True,
    reuse_stores: bool = False,
    consolidation: bool = False,
    dedup_judge: bool = False,
    pooled: bool = False,
    batch_qa: bool = False,
) -> MemoryRunEstimate:
    """Project LLM call counts, token volume, and (when priced) dollars.

    ``memory`` selects the memory under test (see
    :data:`~particles.benchmark.memory.comparators.MEMORY_KINDS`): the
    ``chunks`` comparator makes no write-time LLM call at all, ``notes`` makes
    exactly one per unique session. ``baselines=False`` (a comparator run
    reusing the particles run's baseline columns) drops the full-context and
    no-memory answer/judge calls from the projection.

    ``reuse_stores=True`` zeroes the write side outright: a replay
    deposits nothing and extracts nothing. The projection has to
    say so, or the confirm gate demands ``--yes`` for ~8.7k phantom calls on a
    run that makes none — and an operator who has learned to wave that through
    is exactly the operator who will wave through the run that *does* cost
    hundreds of dollars.

    Output tokens are projected per call from the
    ``benchmark_memory.estimate_output_tokens_per_*_call`` assumptions; they
    are the larger half of the bill and were the reason the inaugural run cost
    nearly twice its input-only projection. The extraction figure and the
    chars-per-token factor behind the input side are looked up per model
    first (``estimate_output_tokens_per_extraction_call_by_model``,
    ``chars_per_token_by_model`` — keyed exactly like ``price_per_mtok``) with
    the scalars as fallback, and ``assumption_sources`` records which was
    used. ``pooled`` / ``batch_qa`` mark
    the write side / the QA side as batch-priced, which only holds while
    ``llm.batch.enabled`` (the runner degrades to sequential calls
    otherwise). Dollars appear only for models with a
    ``benchmark_memory.price_per_mtok`` entry — the mapping ships empty
    because prices go stale — and a run with any unpriced component prints
    no total rather than a misleading partial one.
    """
    cfg = get_config().extraction
    unique_chars: dict[str, int] = {}
    haystack_chars_total = 0
    for question in questions:
        for session in question.sessions:
            text = render_session_text(session)
            haystack_chars_total += len(text)
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            unique_chars.setdefault(key, len(text))

    extraction_calls = 0
    if reuse_stores:
        pass  # replayed store: no deposit, no extraction, no write-time call
    elif memory == "notes":
        extraction_calls = sum(1 for n in unique_chars.values() if n > 0)
    elif memory == "particles":
        for n in unique_chars.values():
            if n <= 0:
                continue
            if n <= cfg.html_chunk_size:
                extraction_calls += 1
            else:
                extraction_calls += min(-(-n // cfg.html_chunk_size), cfg.max_llm_calls_per_source)

    # ablation passes. ``--consolidation`` is boundable from config:
    # every probe cap is per store, and this harness gives every question its
    # own. All three probe-bearing passes are summed: the same-subject update
    # sweep joined the cycle after this bound was written and was left out of
    # it, which routed up to 100 probes per store around the confirm gate.
    # ``--dedup-judge`` is not — the judge fans out per Subject cluster,
    # and how many clusters a store holds is a property of the extraction that
    # has not happened yet — so it is disclosed as unbounded instead of being
    # left out of the projection entirely.
    pass_calls = 0
    unbounded: list[str] = []
    if consolidation:
        cons_cfg = get_config().consolidation
        per_store = (
            cons_cfg.max_reconcile_probes
            + cons_cfg.max_update_probes
            + get_config().audit.max_contradiction_probes
        )
        pass_calls = per_store * len(questions)
    if dedup_judge:
        unbounded.append(
            "--dedup-judge: one judged batch per Subject candidate cluster per "
            "question — unbounded here (the cluster count is a property of the "
            "extracted store). Probe it with a --limit 5 --reuse-stores run and "
            "read 'dedup judge applied X of Y' in the quality notes."
        )

    conditions = 3 if baselines else 1
    answer_calls = conditions * len(questions) if qa else 0
    judge_calls = conditions * len(questions) if qa else 0
    # Input magnitude: unique-session write-time input + the full-context
    # condition re-reading every question's haystack once. Each side is
    # divided by the chars-per-token factor of the model that will read it
    # (the write side by the extraction model, the answer side by the answer
    # model) — the same factor check_context_window holds the window against.
    llm_cfg = get_config().llm
    extraction_sel = llm_cfg.for_purpose("extraction")
    answer_sel = llm_cfg.for_purpose("benchmark_answer")
    write_cpt = _chars_per_token_assumption(extraction_sel)
    answer_cpt = _chars_per_token_assumption(answer_sel)
    write_chars = 0 if (memory == "chunks" or reuse_stores) else sum(unique_chars.values())
    write_input_tokens = int(write_chars / write_cpt.value)
    qa_input_tokens = int(haystack_chars_total / answer_cpt.value) if qa and baselines else 0

    # Output side: per-call assumptions × projected calls. Zero write-side
    # output on --reuse-stores / chunks follows from zero write-side calls.
    # The extraction figure is per model (Sonnet 5 emits ~1.9x what Haiku 4.5
    # does per call); the QA-side figures are scaffold-bound, not model-bound.
    mem_cfg = get_config().benchmark_memory
    extraction_output = _extraction_output_assumption(extraction_sel)
    assumptions = {
        "extraction": int(extraction_output.value),
        "answer": mem_cfg.estimate_output_tokens_per_answer_call,
        "judge": mem_cfg.estimate_output_tokens_per_judge_call,
    }
    sources = {
        "extraction_output": extraction_output,
        "write_chars_per_token": write_cpt,
        "answer_chars_per_token": answer_cpt,
    }
    write_output = extraction_calls * assumptions["extraction"]
    answer_output = answer_calls * assumptions["answer"]
    judge_output = judge_calls * assumptions["judge"]

    # Dollars, per component, against whatever each purpose resolves to now.
    # The batch discount holds only while llm.batch.enabled — with it off the
    # runner degrades --pooled / --batch-qa to sequential full-price calls.
    batch_on = get_config().llm.batch.enabled
    components: list[EstimateCostComponent] = []
    if extraction_calls:
        components.append(
            _cost_component(
                "write (extraction / notes)",
                "extraction",
                calls=extraction_calls,
                input_tokens=write_input_tokens,
                output_tokens=write_output,
                batched=pooled and batch_on,
            )
        )
    if answer_calls:
        components.append(
            _cost_component(
                "answer",
                "benchmark_answer",
                calls=answer_calls,
                input_tokens=qa_input_tokens,
                output_tokens=answer_output,
                batched=batch_qa and batch_on,
            )
        )
    if judge_calls:
        components.append(
            _cost_component(
                "judge",
                "benchmark",
                calls=judge_calls,
                input_tokens=0,
                output_tokens=judge_output,
                batched=batch_qa and batch_on,
            )
        )
    unpriced = sorted({c.model for c in components if c.cost_usd is None})
    total: float | None = None
    if components and not unpriced:
        total = sum(c.cost_usd or 0.0 for c in components)

    return MemoryRunEstimate(
        questions=len(questions),
        unique_sessions=len(unique_chars),
        total_session_chars=sum(unique_chars.values()),
        estimated_extraction_calls=extraction_calls,
        estimated_answer_calls=answer_calls,
        estimated_judge_calls=judge_calls,
        estimated_pass_calls=pass_calls,
        estimated_llm_calls=extraction_calls + answer_calls + judge_calls + pass_calls,
        estimated_tokens=write_input_tokens + qa_input_tokens,
        unbounded_components=unbounded,
        estimated_write_output_tokens=write_output,
        estimated_answer_output_tokens=answer_output,
        estimated_judge_output_tokens=judge_output,
        estimated_output_tokens=write_output + answer_output + judge_output,
        output_assumptions=assumptions,
        assumption_sources=sources,
        cost_components=components,
        estimated_cost_usd=total,
        unpriced_models=unpriced,
    )


class ContextWindowExceeded(RuntimeError):
    """The ``qa_full_context`` baseline would not fit the answering model.

    Raised before any LLM call. Refusal rather than a warning, for the same
    reason :class:`SameModelViolation` is: a run that overflows the window
    does not fail, it *degrades* — the provider truncates or errors, and the
    baseline condition absorbs the whole loss while ``qa_no_memory``, whose
    prompt is three lines, sails through. That is a systematically
    anti-baseline result dressed as a measurement, and keeping the baseline
    honest is the whole point of the four-condition table. The
    ~500-session ``m`` variant is the case in hand; ``s`` (~115k tokens) fits a
    200k window with room.
    """


class ContextWindowCheck(BaseModel):
    """Whether the selected questions' full-context prompts fit the window.

    Token counts use the harness's chars-per-token heuristic — the answer
    model's ``benchmark_memory.chars_per_token_by_model`` entry, else the
    scalar, the same factor :func:`estimate_run` prices the answer side with
    — so this is a magnitude check, not an exact tokenizer. That is the right
    instrument for an order-of-magnitude overflow ("1.4M against a 200k
    budget"), but the factor has to be the model's own: at the ~4-chars/token
    default a new-tokenizer model's prompt reads ~27 % smaller than it is,
    which is the difference between a comfortable margin and none on the
    largest ``s`` haystacks. ``chars_per_token`` discloses the factor used.
    """

    variant: str
    #: The chars-per-token factor the counts were taken at, and its source.
    chars_per_token: EstimateAssumption | None = None
    checked_questions: int = 0
    window_tokens: int = 0
    reserved_output_tokens: int = 0
    largest_question_id: str | None = None
    largest_prompt_tokens: int = 0
    over_window_questions: int = 0

    @property
    def budget_tokens(self) -> int:
        """Input tokens available: the window less the reserved output budget."""
        return self.window_tokens - self.reserved_output_tokens

    @property
    def fits(self) -> bool:
        """True when every checked question's full-context prompt fits."""
        return self.over_window_questions == 0


def check_context_window(
    questions: list[MemoryQuestion],
    *,
    variant: str,
    qa: bool = True,
    baselines: bool = True,
) -> ContextWindowCheck:
    """Hold each question's ``qa_full_context`` prompt against the window.

    Only condition iii is unbounded: ``qa_particles`` is top-k bounded and
    ``qa_no_memory`` carries no context at all, so a run without the baselines
    (``qa=False``, or a comparator's ``baselines=False``) has nothing to
    check and returns a vacuously-fitting result.

    The window comes from ``benchmark_memory.answer_context_window_tokens``;
    ``_ANSWER_MAX_TOKENS`` is reserved out of it, because on the Anthropic
    Messages API the output budget is drawn from the same window as the
    input.
    """
    cfg = get_config().benchmark_memory
    cpt = _chars_per_token_assumption(get_config().llm.for_purpose("benchmark_answer"))
    check = ContextWindowCheck(
        variant=variant,
        chars_per_token=cpt,
        window_tokens=cfg.answer_context_window_tokens,
        reserved_output_tokens=_ANSWER_MAX_TOKENS,
    )
    if not (qa and baselines):
        return check
    system_tokens = int(len(_answer_system()) / cpt.value)
    for question in questions:
        check.checked_questions += 1
        prompt_chars = len(_answer_prompt(question, _full_context(question)))
        prompt_tokens = int(prompt_chars / cpt.value) + system_tokens
        if prompt_tokens > check.largest_prompt_tokens:
            check.largest_prompt_tokens = prompt_tokens
            check.largest_question_id = question.question_id
        if prompt_tokens > check.budget_tokens:
            check.over_window_questions += 1
    return check


def render_context_window_check(check: ContextWindowCheck) -> str:
    """One line stating the check's outcome — printed beside the estimate."""
    if not check.checked_questions:
        return "Context window: not checked — the qa_full_context baseline is not part of this run."
    verdict = "fits" if check.fits else "DOES NOT FIT"
    cpt = check.chars_per_token
    factor = (
        f", counted at ~{cpt.value:.2f} chars/token ({cpt.describe()}; "
        f"benchmark_memory.chars_per_token_by_model overrides the scalar)"
        if cpt is not None
        else ""
    )
    return (
        f"Context window ({check.variant} variant): largest qa_full_context prompt "
        f"~{check.largest_prompt_tokens:,} tokens "
        f"({check.largest_question_id}) against a {check.budget_tokens:,}-token input "
        f"budget ({check.window_tokens:,}-token window less {check.reserved_output_tokens:,} "
        f"reserved for output){factor} — {verdict}"
        + (
            ""
            if check.fits
            else f"; {check.over_window_questions} of {check.checked_questions} "
            f"question(s) over budget"
        )
    )


def _context_window_refusal(check: ContextWindowCheck) -> str:
    """The refusal message — says which variant, by how much, and the two fixes."""
    return (
        f"The qa_full_context baseline does not fit the answering model's context "
        f"window on the {check.variant!r} variant: {check.over_window_questions} of "
        f"{check.checked_questions} question(s) exceed the {check.budget_tokens:,}-token "
        f"input budget, the largest at ~{check.largest_prompt_tokens:,} tokens "
        f"({check.largest_question_id}). Running anyway would not produce a weaker "
        f"baseline — it would produce a truncated or errored one, and every lost call "
        f"would land on the baseline condition alone. Either route "
        f"llm.benchmark_answer to a model whose window fits and set "
        f"benchmark_memory.answer_context_window_tokens to match, or run the "
        f"conditions that are bounded (--no-baselines, or qa=False). Refusing to run."
    )


def render_estimate(estimate: MemoryRunEstimate) -> str:
    """Human rendering of the estimate — always printed before any LLM call."""
    parts = [
        f"Estimate: {estimate.questions} question(s), {estimate.unique_sessions} unique "
        f"haystack session(s) → ~{estimate.estimated_extraction_calls} write-time "
        f"(extraction / notes) call(s) + {estimate.estimated_answer_calls} answer call(s) + "
        f"{estimate.estimated_judge_calls} judge call(s)"
    ]
    if estimate.estimated_pass_calls:
        parts.append(f" + ≤{estimate.estimated_pass_calls} ablation-pass probe(s), capped")
    parts.append(
        f" = ~{estimate.estimated_llm_calls} LLM call(s), "
        f"~{estimate.estimated_tokens:,} input tokens (the full-context baseline "
        f"re-reads each question's whole haystack). Repeated sessions are candidate-cached."
    )
    src = estimate.assumption_sources
    write_cpt = src.get("write_chars_per_token")
    answer_cpt = src.get("answer_chars_per_token")
    if write_cpt is not None and answer_cpt is not None:
        parts.append(
            f"\n  Input tokens counted at ~{write_cpt.value:.2f} chars/token on the write "
            f"side ({write_cpt.describe()}) and ~{answer_cpt.value:.2f} on the answer side "
            f"({answer_cpt.describe()}); benchmark_memory.chars_per_token_by_model "
            f"overrides the scalar per model."
        )
    a = estimate.output_assumptions
    extraction_src = src.get("extraction_output")
    extraction_note = (
        f"; the extraction figure is the {extraction_src.describe()}"
        if extraction_src is not None
        else ""
    )
    parts.append(
        f"\n  Output tokens: ~{estimate.estimated_output_tokens:,} projected — "
        f"write ~{estimate.estimated_write_output_tokens:,} + answer "
        f"~{estimate.estimated_answer_output_tokens:,} + judge "
        f"~{estimate.estimated_judge_output_tokens:,}, assuming "
        f"{a.get('extraction', 0):,} / {a.get('answer', 0):,} / {a.get('judge', 0):,} "
        f"output tokens per extraction / answer / judge call "
        f"(benchmark_memory.estimate_output_tokens_per_*_call{extraction_note}). "
        f"Output is the larger half of an extraction run's bill; the input figure is a "
        f"floor (system prompts are not counted)."
    )
    if estimate.cost_components:
        parts.append("\n  " + render_cost_projection(estimate))
    for note in estimate.unbounded_components:
        parts.append(f"\n  NOT INCLUDED ABOVE — {note}")
    return "".join(parts)


def render_cost_projection(estimate: MemoryRunEstimate) -> str:
    """The dollar line: a total when every component is priced, else which model is not."""
    if estimate.unpriced_models:
        missing = ", ".join(f"no price configured for {m}" for m in estimate.unpriced_models)
        return (
            f"Cost: not projected — {missing} (set benchmark_memory.price_per_mtok; "
            f"prices go stale, so none is compiled in)."
        )
    pieces = []
    for c in estimate.cost_components:
        tag = ", batch" if c.batched else ""
        pieces.append(f"{c.name} US${c.cost_usd or 0.0:,.2f} ({c.model}{tag})")
    total = estimate.estimated_cost_usd or 0.0
    return f"Cost: ~US${total:,.2f} at configured list prices — " + " + ".join(pieces) + "."


# ---------------------------------------------------------------------------
# QA prompts + judge (§2/§3 steps 3–5)
# ---------------------------------------------------------------------------


def _answer_prompt(question: MemoryQuestion, context_block: str) -> str:
    """The shared answer scaffold — identical across conditions ii–iv.

    Only the context block differs (retrieved particles / full haystack /
    nothing), so the three conditions measure the memory, not the prompt.
    """
    date_line = f"\nQuestion date: {question.question_date}" if question.question_date else ""
    return (
        f"Context from the user's prior conversations:\n"
        f"{context_block}\n\n"
        f"Question:{date_line}\n{question.question}\n\n"
        f"Answer concisely."
    )


def _subject_rendering() -> str:
    """The configured ``qa_particles`` subject rendering."""
    from particles.config import get_config

    return get_config().benchmark_memory.subject_rendering


def _particles_context_lines(
    particles: list[Particle],
    pub_at_by_id: dict[str, datetime | None],
    budget_tokens: int | None = None,
    subject_names: Mapping[str, str] | None = None,
    rendering: str | None = None,
) -> list[tuple[str, str]]:
    """``(particle id, rendered line)`` pairs of condition ii's context, in rank order.

    The single source of the clamp, so the ids the report records as
    ``context_particle_ids`` are exactly the particles whose lines the
    answering model saw — a particle cut by the budget is absent from both.

    ``rendering`` selects how the subjects field is written
    (``benchmark_memory.subject_rendering``; the caller passes it so one run
    cannot straddle two renderings mid-loop):

    * ``uuids`` — the resolved ``subject_ids`` verbatim. What every published
      table through 1.146.5 was measured under, and the default.
    * ``names`` — each subject's ``canonical_name``, looked up in
      ``subject_names``. An id with no entry falls back to the id, so a
      partially-resolvable store degrades per subject instead of failing.
    * ``none`` — no subject field at all.

    A UUID is a poor thing to spend context on: it tokenizes at roughly half
    the characters-per-token of the name it stands for, and carries nothing the
    answering model can use. Measured over the 2026-09-18 s150 store set at
    ``top_k`` 40, ``names`` costs 33.6% fewer context tokens than ``uuids`` and
    ``none`` 52.2% fewer. The sibling extractor harness documents the same trap
    from the matching side (``equivalence.subject_qualified`` takes names, never
    ``particle.subject_ids``, because prefixing a claim with a UUID poisons the
    similarity matrix).
    """
    mode = rendering or "uuids"
    lines: list[tuple[str, str]] = []
    used_chars = 0
    budget_chars = None if budget_tokens is None else budget_tokens * 4
    for p in particles:
        pub_at = pub_at_by_id.get(p.id)
        date = pub_at.date().isoformat() if pub_at else "undated"
        if mode == "none":
            line = f"- [{date}] {p.content}"
        else:
            shown = list(p.subject_ids)
            if mode == "names" and subject_names is not None:
                shown = [subject_names.get(sid, sid) for sid in shown]
            subjects = ", ".join(shown) if shown else "-"
            line = f"- [{date}] {p.content} (subjects: {subjects})"
        if budget_chars is not None and lines and used_chars + len(line) > budget_chars:
            break
        lines.append((p.id, line))
        used_chars += len(line) + 1
    return lines


def _particles_context(
    particles: list[Particle],
    pub_at_by_id: dict[str, datetime | None],
    budget_tokens: int | None = None,
    subject_names: Mapping[str, str] | None = None,
    rendering: str | None = None,
) -> str:
    """Condition ii's context: top-k particle claims + subjects + dates.

    ``budget_tokens`` is the QA-at-budget clamp: particles are
    appended in rank order until the next line would exceed the budget
    (estimated at ~4 chars/token, the same heuristic as
    :func:`estimate_run`). ``None`` (default) keeps the full top-k context.
    The clamp applies to this condition only — the full-context baseline is
    deliberately unclamped (it is the ceiling, not the product).

    ``subject_names`` / ``rendering`` are passed through to
    :func:`_particles_context_lines`.
    """
    if not particles:
        return "(no memory claims were retrieved)"
    return "\n".join(
        line
        for _, line in _particles_context_lines(
            particles, pub_at_by_id, budget_tokens, subject_names, rendering
        )
    )


def _full_context(question: MemoryQuestion) -> str:
    """Condition iii's context: the full concatenated haystack."""
    blocks = [render_session_text(s) for s in question.sessions]
    return "\n\n---\n\n".join(blocks) if blocks else "(no history)"


_JUDGE_TYPE_GUIDANCE: dict[str, str] = {
    "single-session-user": ("The question asks about information the user stated in one session."),
    "single-session-assistant": (
        "The question asks about information the assistant stated in one session."
    ),
    "single-session-preference": (
        "The question asks for a response consistent with a preference the user "
        "expressed. Judge whether the response respects that preference; exact "
        "wording does not matter."
    ),
    "multi-session": (
        "Answering requires combining information across multiple sessions. Judge "
        "semantic equivalence with the reference answer."
    ),
    "temporal-reasoning": (
        "The question requires reasoning about dates or event order. Judge whether "
        "the response's dates/ordering agree with the reference answer; equivalent "
        "date formats count as correct."
    ),
    "knowledge-update": (
        "The information was updated across sessions. The response is correct only "
        "if it reflects the LATEST state, not an outdated earlier value."
    ),
}


#: Judge prompt protocol, by version (``benchmark_memory.judge_protocol``).
#: v1 is the 1.74.0 paraphrase of the dataset's autoeval prompts, which the
#: inaugural 2026-08-16 table was scored under. It deviates from the official
#: templates in ways that bite on specific categories: the abstention prompt
#: omitted the gold *explanation* and the official clause that a response
#: which offers other information while denying the asked fact still counts;
#: the preference prompt labelled the rubric a "reference answer" and dropped
#: "does not need to reflect all the points"; temporal-reasoning lost the
#: off-by-one leniency; knowledge-update was stricter than the official
#: "previous information alongside the updated answer is correct". v2 is the
#: official ``get_anscheck_prompt`` templates verbatim (LongMemEval,
#: ``src/evaluation/evaluate_qa.py``), still routed to the Anthropic judge —
#: the *model* deviation stays disclosed; the *prompt* deviation is gone. Like
#: the answer scaffold, the version rides the run tuple and the checkpoint
#: key: two runs judged under different protocols are not comparable.
_OFFICIAL_JUDGE_GENERIC = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
    "Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_OFFICIAL_JUDGE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for the "
    "number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct. \n\nQuestion: "
    "{}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
    "correct? Answer yes or no only."
)
_OFFICIAL_JUDGE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: "
    "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no "
    "only."
)
_OFFICIAL_JUDGE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: "
    "{}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? "
    "Answer yes or no only."
)
_OFFICIAL_JUDGE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from "
    "a model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some "
    "other information is given but the asked information is not.\n\nQuestion: "
    "{}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly "
    "identify the question as unanswerable? Answer yes or no only."
)
_OFFICIAL_JUDGE_BY_TYPE: dict[str, str] = {
    "single-session-user": _OFFICIAL_JUDGE_GENERIC,
    "single-session-assistant": _OFFICIAL_JUDGE_GENERIC,
    "multi-session": _OFFICIAL_JUDGE_GENERIC,
    "temporal-reasoning": _OFFICIAL_JUDGE_TEMPORAL,
    "knowledge-update": _OFFICIAL_JUDGE_KNOWLEDGE_UPDATE,
    "single-session-preference": _OFFICIAL_JUDGE_PREFERENCE,
}


def _judge_protocol() -> int:
    """The configured judge-prompt protocol version."""
    from particles.config import get_config

    return get_config().benchmark_memory.judge_protocol


def _judge_prompt_v1(question: MemoryQuestion, model_answer: str) -> str:
    """Protocol 1: the 1.74.0 paraphrase the inaugural table was scored under."""
    if question.is_abstention:
        return (
            "The following question CANNOT be answered from the user's chat "
            "history — the information was never mentioned. A correct response "
            "abstains: it says the information is not available (or declines to "
            "answer) instead of guessing.\n\n"
            f"Question: {question.question}\n\n"
            f"Response: {model_answer}\n\n"
            "Does the response correctly abstain? Answer yes or no only."
        )
    guidance = _JUDGE_TYPE_GUIDANCE.get(
        question.question_type,
        "Judge semantic equivalence with the reference answer.",
    )
    return (
        "You are grading a model's answer to a question about a user's chat "
        f"history. {guidance}\n\n"
        f"Question: {question.question}\n"
        f"Reference answer: {question.answer}\n"
        f"Model response: {model_answer}\n\n"
        "Is the model response correct? Answer yes or no only."
    )


def _judge_prompt_v2(question: MemoryQuestion, model_answer: str) -> str:
    """Protocol 2: the official autoeval templates verbatim.

    ``question.answer`` is what the dataset supplies in every case — the
    correct answer, the preference *rubric*, or the abstention *explanation*
    — and each official template names it accordingly. An unknown question
    type falls back to the generic template rather than raising, as the
    official script does.
    """
    if question.is_abstention:
        return _OFFICIAL_JUDGE_ABSTENTION.format(question.question, question.answer, model_answer)
    template = _OFFICIAL_JUDGE_BY_TYPE.get(question.question_type, _OFFICIAL_JUDGE_GENERIC)
    return template.format(question.question, question.answer, model_answer)


def judge_prompt(question: MemoryQuestion, model_answer: str) -> str:
    """Per-question-type autoeval judge prompt (step 5).

    Selected by ``benchmark_memory.judge_protocol``: 1 is the paraphrase the
    inaugural table was scored under, 2 the official templates verbatim (see
    the module comment above the templates). The judge *model* deviation
    (Anthropic, not the paper's OpenAI judge) is disclosed on the published
    page under either protocol — numbers are comparable within a table judged
    under one protocol and one model, not across leaderboards.
    """
    if _judge_protocol() == 1:
        return _judge_prompt_v1(question, model_answer)
    return _judge_prompt_v2(question, model_answer)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


class _CallOutcome(BaseModel):
    """One answer/judge call: the text it returned, or why there is none.

    ``excluded`` is set to a :data:`~particles.benchmark.memory.schema.
    QA_EXCLUSION_KINDS` member exactly when ``text`` is ``None`` — the pair is
    what lets a caller mark the condition *unscoreable* rather than wrong.
    """

    text: str | None = None
    excluded: str = ""
    detail: str = ""


async def _scored_call(
    purpose: LLMPurpose,
    prompt: str,
    *,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    system: str | None = None,
) -> _CallOutcome:
    """One answer/judge call: retry transient failures, classify what remains.

    The two failure causes are treated differently on purpose.

    * A reply that carried **no text** (:class:`EmptyCompletionError`) is
      deterministic at a fixed ``max_tokens``: an extended-thinking model that
      spent its whole budget thinking will do it again on an identical call, so
      retrying only spends the money twice — and on the full-context baseline a
      retry is ~115k input tokens. Reported immediately as
      :data:`QA_EXCLUSION_BUDGET`; the fix is the operator's cap.
    * Anything else is transient until proven otherwise — an overload, a
      timeout, a dropped connection — so it is retried ``retries`` times with a
      linear backoff before being reported as :data:`QA_EXCLUSION_INFRA`.

    Either way the caller gets a *typed absence*, never a verdict. Scoring
    these ``correct=False`` is the asymmetry this exists to remove: the
    ~115k-token baseline call fails far more often than the tiny
    ``qa_no_memory`` one, so a shared failure rate is a systematic
    thumb on the scale against the baseline.
    """
    attempt = 0
    while True:
        try:
            text = await complete(
                purpose,
                prompt,
                max_tokens=max_tokens,
                system=system,
                temperature=0.0,
            )
            return _CallOutcome(text=text)
        except EmptyCompletionError as exc:
            return _CallOutcome(excluded=QA_EXCLUSION_BUDGET, detail=repr(exc))
        except Exception as exc:  # noqa: BLE001 — one bad call must not abort the run
            if attempt >= retries:
                return _CallOutcome(excluded=QA_EXCLUSION_INFRA, detail=repr(exc))
            attempt += 1
            log.info("benchmark %s call failed (%r); retry %d/%d", purpose, exc, attempt, retries)
            if backoff_seconds:
                await asyncio.sleep(backoff_seconds * attempt)


def _excluded_note(question: MemoryQuestion, condition: str, stage: str, call: _CallOutcome) -> str:
    """The quality note for a call that produced no verdict."""
    return (
        f"Question {question.question_id} [{condition}]: {stage} call produced no "
        f"verdict ({call.excluded}: {call.detail}); excluded from the accuracy "
        f"denominator, not scored incorrect"
    )


class _QaAccumulator:
    """Per-condition accounting: verdicts, exclusions, and drill-down rows."""

    def __init__(self, condition: str) -> None:
        self.condition = condition
        self.pairs: list[tuple[str, bool]] = []
        self.rows: list[QaQuestionResult] = []
        self.excluded: dict[str, int] = dict.fromkeys(QA_EXCLUSION_KINDS, 0)

    def record(
        self,
        question: MemoryQuestion,
        correct: bool,
        *,
        answer: str | None = None,
        verdict: str | None = None,
        context_particle_ids: list[str] | None = None,
    ) -> None:
        self.pairs.append((question.question_type, correct))
        self.rows.append(
            QaQuestionResult(
                question_id=question.question_id,
                question_type=question.question_type,
                correct=correct,
                abstention=question.is_abstention,
                answer=answer,
                verdict=verdict,
                context_particle_ids=list(context_particle_ids or []),
            )
        )

    def record_excluded(
        self,
        question: MemoryQuestion,
        kind: str,
        *,
        answer: str | None = None,
        context_particle_ids: list[str] | None = None,
    ) -> None:
        """Record a question this condition could not score, by cause.

        Deliberately touches neither ``pairs`` (the accuracy denominator) nor
        ``accuracy_by_type``: an unscoreable call is absent from the rate, not
        a zero in it. ``answer`` is set when the answer call succeeded and
        only the judge produced nothing — the reviewer can still read what
        was said.
        """
        self.excluded[kind] = self.excluded.get(kind, 0) + 1
        self.rows.append(
            QaQuestionResult(
                question_id=question.question_id,
                question_type=question.question_type,
                correct=None,
                abstention=question.is_abstention,
                excluded=kind,
                answer=answer,
                context_particle_ids=list(context_particle_ids or []),
            )
        )

    def to_metrics(self, model_id: str) -> QaConditionMetrics:
        return QaConditionMetrics(
            condition=self.condition,
            model_id=model_id,
            questions=len(self.pairs),
            accuracy=qa_accuracy([c for _, c in self.pairs]),
            excluded_budget=self.excluded[QA_EXCLUSION_BUDGET],
            excluded_infra=self.excluded[QA_EXCLUSION_INFRA],
            excluded_unrecorded=self.excluded[QA_EXCLUSION_UNRECORDED],
            accuracy_by_type=accuracy_by_type(self.pairs),
            per_question=self.rows,
        )


def _resolve_answer_model() -> str:
    """The resolved ``llm.benchmark_answer`` "<provider>:<model>" id, right now."""
    return get_provider("benchmark_answer").provider_model


def _resolve_extraction_model() -> str:
    """The resolved ``llm.extraction`` "<provider>:<model>" id, right now."""
    return get_provider("extraction").provider_model


#: The one corpus source type this harness deposits (a haystack
#: session is a ``CONVERSATION`` entry). Every read-side policy that keys off
#: source type — decay above all — is therefore a property of *this* key alone.
BENCHMARK_SOURCE_TYPE = "CONVERSATION"


def _thresholds_snapshot() -> dict[str, float]:
    """The pipeline thresholds in effect — part of the recorded run tuple (§5).

    Every knob an ablation can flip has to be *visible* here, or two arms of
    that ablation record byte-identical tuples and the pair is unpublishable
    (results are comparable only against the same recorded tuple).

    Decay is the case that forced the point: it is
    configured per source type and this harness deposits exactly one, so the
    resolved ``CONVERSATION`` rule — not the whole ``content_age_decay``
    table — is what belongs on the tuple. ``half_life_days`` of ``0.0``
    encodes "no rule for this source type", i.e. decay is inert (see
    :func:`decay_quality_note`).
    """
    cfg = get_config()
    decay = cfg.content_age_decay.sources.get(BENCHMARK_SOURCE_TYPE)
    return {
        "extraction.similarity_threshold": cfg.extraction.similarity_threshold,
        "confidence.uncalibrated_cap.enabled": float(cfg.confidence.uncalibrated_cap.enabled),
        "confidence.uncalibrated_cap.cap_value": cfg.confidence.uncalibrated_cap.cap_value,
        "extraction.duplicate_suppression.enabled": float(
            cfg.extraction.duplicate_suppression.enabled
        ),
        "links_suggest.auto_merge.enabled": float(cfg.links_suggest.auto_merge.enabled),
        f"content_age_decay.sources.{BENCHMARK_SOURCE_TYPE}.half_life_days": (
            decay.half_life_days if decay is not None else 0.0
        ),
        f"content_age_decay.sources.{BENCHMARK_SOURCE_TYPE}.floor": (
            decay.floor if decay is not None else 1.0
        ),
    }


def decay_quality_note() -> str | None:
    """Disclose an inert decay policy, or ``None`` when decay is live.

    The shipped ``content_age_decay.sources`` table configures four web-ish
    source types and **not** ``CONVERSATION``, so under stock config
    :meth:`DecayPolicy.resolve` returns ``None`` for every particle this
    harness mints and the recency factor is a flat 1.0. A decay on/off
    ablation run in that state is *vacuous*: both arms are the "off" arm, and
    the pair reads as "decay changes nothing" when in truth decay never ran.
    The run says so in its own report rather than leaving it to whoever reads
    the table (the honesty framing is structural).
    """
    if BENCHMARK_SOURCE_TYPE in get_config().content_age_decay.sources:
        return None
    return (
        f"Decay is INERT for this run: content_age_decay.sources has no "
        f"{BENCHMARK_SOURCE_TYPE} rule, so every particle scores at recency "
        f"factor 1.0. A decay on/off ablation needs a "
        f"{BENCHMARK_SOURCE_TYPE} entry in config before the 'on' arm means "
        f"anything."
    )


#: Below this, the ranker's confidence term is mostly gone and a decay arm is
#: measuring its absence, not a preference for recent sessions.
_DECAY_SWAMPED_BELOW = 0.25


def decay_reference_note(questions: list[MemoryQuestion]) -> str | None:
    """Disclose where a *live* decay rule actually put the recency factors.

    The sibling of :func:`decay_quality_note` for the opposite failure. Decay
    is evaluated at the run's wall-clock instant, because retrieval passes no
    reference instant, while the haystacks are dated years earlier. A half-life
    that would separate last week from last quarter at the question's own date
    therefore scores every session near zero, the rank score's additive
    confidence term drops out, and the arm measures similarity-only ranking:
    two half-lives a factor of three apart returned identical tables on the
    inaugural store set. The factors are what tell the two readings
    apart, so the run reports them rather than leaving the half-life to imply
    a recency effect that did not happen.
    """
    rule = get_config().content_age_decay.sources.get(BENCHMARK_SOURCE_TYPE)
    if rule is None:
        return None
    now = datetime.now(UTC)
    factors = [
        recency_factor_from_params(published, rule.half_life_days, rule.floor, now)
        for question in questions
        for mem_session in question.sessions
        if (published := parse_session_date(mem_session.date)) is not None
    ]
    if not factors:
        return None
    note = (
        f"Decay is evaluated at this run's wall-clock instant, not at each "
        f"question's date: under the {BENCHMARK_SOURCE_TYPE} rule (half-life "
        f"{rule.half_life_days:g} d, floor {rule.floor:g}) the {len(factors)} dated "
        f"haystack session(s) score recency factors {min(factors):.2g} to "
        f"{max(factors):.2g}."
    )
    if max(factors) < _DECAY_SWAMPED_BELOW:
        note += (
            f" Every factor is below {_DECAY_SWAMPED_BELOW:g}, so the confidence "
            f"term of the rank score is largely suppressed: read this arm as "
            f"similarity-only ranking, not as a preference for recent sessions."
        )
    return note


def _read_side_ablation_knobs() -> dict[str, float]:
    """The config-driven read-side ablation knobs that are **not** at their default.

    The cap and decay arms are config overlays, not flags, so nothing on the
    ``run_memory_benchmark`` signature distinguishes them from the control.
    Left out of the checkpoint key, a cap-on arm run after a control at the
    same ``top_k`` restored all of the control's outcomes and reported "the
    cap changes nothing" when the cap never ran: the same silent
    ON-arm-is-an-OFF-arm shape as the inert decay rule and the shared
    consolidation lock. Empty under stock config, so the key every
    existing checkpoint was written under stays byte-identical.
    """
    snapshot = _thresholds_snapshot()
    cap_key = "confidence.uncalibrated_cap.enabled"
    decay_key = f"content_age_decay.sources.{BENCHMARK_SOURCE_TYPE}.half_life_days"
    knobs: dict[str, float] = {}
    if snapshot[cap_key]:
        knobs[cap_key] = snapshot[cap_key]
        knobs["confidence.uncalibrated_cap.cap_value"] = snapshot[
            "confidence.uncalibrated_cap.cap_value"
        ]
    if snapshot[decay_key]:
        floor_key = f"content_age_decay.sources.{BENCHMARK_SOURCE_TYPE}.floor"
        knobs[decay_key] = snapshot[decay_key]
        knobs[floor_key] = snapshot[floor_key]
    return knobs


# ---------------------------------------------------------------------------
# Persisted scratch stores — the ablation cost lever
# ---------------------------------------------------------------------------

#: Filename of the write-side manifest dropped beside a kept store set.
STORES_MANIFEST = "stores-manifest.json"


#: Write-side members that only *narrow* which questions were selected, and so
#: say nothing about what landed in any one question's store (see
#: load_stores_manifest). ``sample_seed`` is deliberately not one: a different
#: seed is a different draw, and stays a refusal even when the draws overlap.
_SELECTION_MEMBERS = frozenset({"question_limit", "question_types"})


class StoreReuseError(Exception):
    """A ``--reuse-stores`` run cannot safely replay the store set it was given."""


def _write_side_tuple(
    *,
    dataset_revision: str,
    variant: str,
    selection_seed: int,
    selection_limit: int | None,
    selection_types: list[str],
    extraction_model_id: str,
    embedding_model_id: str,
) -> dict[str, str]:
    """The half of the run tuple that determines what lands **in** the stores.

    Deliberately a strict subset of :class:`RunSelection`. ``top_k``, the
    context budget, the answer/judge models and every read-side threshold are
    excluded on purpose — varying those against one store set is exactly what
    the reuse path exists to make cheap. What is in here is what a replay
    cannot change after the fact: the dataset and question selection (which
    sessions were deposited), the extraction model (what claims were minted),
    the embedding model (the vectors those claims carry), and the two
    write-time reconciliation knobs that decide which candidates became
    particles at all.
    """
    cfg = get_config()
    return {
        "dataset_revision": dataset_revision,
        "variant": variant,
        "sample_seed": str(selection_seed),
        "question_limit": "all" if selection_limit is None else str(selection_limit),
        "question_types": ",".join(sorted(selection_types)),
        "extraction_model_id": extraction_model_id,
        "embedding_model_id": embedding_model_id,
        "extraction.similarity_threshold": str(cfg.extraction.similarity_threshold),
        "extraction.duplicate_suppression.enabled": str(
            cfg.extraction.duplicate_suppression.enabled
        ),
    }


def write_stores_manifest(
    work_dir: Path, write_side: dict[str, str], question_ids: list[str]
) -> None:
    """Stamp a kept store set with the write-side tuple that produced it."""
    payload = {
        "format": 1,
        "write_side": write_side,
        "question_ids": sorted(question_ids),
    }
    (work_dir / STORES_MANIFEST).write_text(json.dumps(payload, indent=2) + "\n")


def prepared_question_ids(work_dir: Path) -> frozenset[str] | None:
    """The questions a kept store set was prepared for, or ``None`` when unreadable.

    The stratified sampler does not nest: ``--limit 5`` under the pinned seed
    draws four questions the ``--limit 150`` draw never prepared, so a narrower
    ``--reuse-stores`` replay has to be drawn *from the prepared set* or it can
    never be a subset of it. ``None`` leaves the refusal to
    :func:`load_stores_manifest`, which words it properly.
    """
    try:
        payload = json.loads((work_dir / STORES_MANIFEST).read_text())
    except (OSError, ValueError):
        return None
    listed = payload.get("question_ids") if isinstance(payload, dict) else None
    if not isinstance(listed, list):
        return None
    return frozenset(str(q) for q in listed)


def load_stores_manifest(
    work_dir: Path, write_side: dict[str, str], question_ids: list[str]
) -> dict[str, str]:
    """Validate a store set against this run's write-side tuple; return the manifest's.

    Refuses rather than warns. A store set built under a different extraction
    model, a different question selection, or different write-time
    reconciliation knobs is a **different population**, and an ablation
    measured against it answers a question nobody asked — the failure mode
    the comparability contract exists to prevent, arriving here
    as a silent one because a replayed store deposits ``unchanged`` and
    re-extracts nothing. It costs nothing to check and a paid arm to miss.
    """
    path = work_dir / STORES_MANIFEST
    if not path.exists():
        raise StoreReuseError(
            f"{path} not found — --reuse-stores needs a store set persisted by an "
            f"earlier --store-dir run. Run the preparing arm first."
        )
    try:
        payload = json.loads(path.read_text())
    except ValueError as exc:
        raise StoreReuseError(f"{path} is not readable JSON: {exc}") from exc
    stored = payload.get("write_side")
    if not isinstance(stored, dict):
        raise StoreReuseError(f"{path} carries no write_side tuple.")
    # A subset replay is sound: every question owns its store, so which other
    # questions were selected beside it never reached its population. The
    # selection members are therefore waived when (and only when) every
    # selected question is one the manifest lists. Without this the documented
    # ``--limit 5 --reuse-stores`` cost probe was refused on ``question_limit``
    # drift and could not run at all.
    listed = payload.get("question_ids")
    is_subset = isinstance(listed, list) and set(question_ids) <= {str(q) for q in listed}
    waived = _SELECTION_MEMBERS if is_subset else frozenset()
    drift = sorted(
        f"{k}: stored {stored.get(k)!r} != this run {v!r}"
        for k, v in write_side.items()
        if k not in waived and stored.get(k) != v
    )
    if drift:
        raise StoreReuseError(
            "The persisted stores were built under a different write-side tuple, so "
            "replaying them would produce an incomparable arm:\n  " + "\n  ".join(drift)
        )
    missing = [qid for qid in question_ids if not (work_dir / f"{qid}.db").exists()]
    if missing:
        raise StoreReuseError(
            f"{len(missing)} selected question(s) have no persisted store under "
            f"{work_dir} (first: {missing[0]}). Re-run the preparing arm."
        )
    return {str(k): str(v) for k, v in stored.items()}


# ---------------------------------------------------------------------------
# Question-level checkpointing — long runs are interruptible
# ---------------------------------------------------------------------------

#: Version stamp of the checkpoint file format. 2 = outcome-affecting-only
#: key (v1.77.4); format-1 files carry the stricter key and are ignored.
_CHECKPOINT_FORMAT = 2


def _run_checkpoint_key(
    *,
    dataset_revision: str,
    variant: str,
    top_k: int,
    context_budget: int | None,
    abstraction: bool,
    qa: bool,
    extraction_model_id: str | None,
    embedding_model_id: str | None,
    answer_model_id: str,
    judge_model_id: str,
    memory: str = "particles",
    baselines: bool = True,
    consolidation: bool = False,
    dedup_judge: bool = False,
    reuse_stores: bool = False,
    answer_scaffold: int = 1,
    judge_protocol: int = 1,
    subject_rendering: str = "uuids",
    read_side_knobs: dict[str, float] | None = None,
) -> dict[str, object]:
    """The identity of one experiment — everything that AFFECTS an outcome.

    Deliberately **outcome-affecting knobs only** — dataset, variant, top_k,
    context budget, abstraction, qa, and the resolved models (they resolve
    from config without an API call). Question *selection* (seed / limit /
    types) is deliberately excluded: each question runs in its own scratch
    store, so its outcome is independent of which subset it ran in — a
    50-question run legitimately restores the overlap from an interrupted
    150-question run's checkpoint (v1.77.4; the first key over-included the
    question set and a changed ``--limit`` silently discarded paid work).
    Any change to an outcome-affecting knob hashes to a different file, so a
    checkpoint can never feed the wrong experiment.

    ``memory`` / ``baselines`` join the key **only when non-default**: a
    comparator run (or one skipping the baseline conditions) is a different
    experiment and hashes apart, while the default particles key is
    byte-identical to what every existing format-2 checkpoint was written
    under — no paid particles outcome becomes unrestorable.
    """
    key: dict[str, object] = {
        "format": _CHECKPOINT_FORMAT,
        "dataset_revision": dataset_revision,
        "variant": variant,
        "top_k": top_k,
        "context_budget": context_budget,
        "abstraction": abstraction,
        # Outcome-affecting: a consolidated / judged / replayed store yields a
        # different retrieval set, so these arms must never restore each
        # other's checkpoints. Emitted only when non-default, exactly as
        # ``memory`` / ``baselines`` are, so the particles key that predates
        # these knobs stays byte-identical and no paid outcome became
        # unrestorable.
        **({"consolidation": True} if consolidation else {}),
        **({"dedup_judge": True} if dedup_judge else {}),
        **({"reuse_stores": True} if reuse_stores else {}),
        # The answer scaffold text is shared by every QA condition, so a
        # different version is a different experiment for all three; v1 (the
        # inaugural text) is omitted so pre-1.141.2 checkpoints still restore.
        **({"answer_scaffold": answer_scaffold} if answer_scaffold != 1 else {}),
        # Same rule for the judge prompt protocol: a verdict under a different
        # protocol is a different outcome; v1 omitted so v1 checkpoints restore.
        **({"judge_protocol": judge_protocol} if judge_protocol != 1 else {}),
        # Same rule again for the subject rendering: it changes the text of
        # every qa_particles context, so an arm under a different rendering is
        # a different outcome. "uuids" omitted so every checkpoint written
        # before the knob existed still restores.
        **({"subject_rendering": subject_rendering} if subject_rendering != "uuids" else {}),
        # The cap and decay arms are config overlays with no flag of their own
        # (see _read_side_ablation_knobs); empty, hence omitted, under stock
        # config.
        **({"read_side_knobs": read_side_knobs} if read_side_knobs else {}),
        "qa": qa,
        "extraction_model_id": extraction_model_id,
        "embedding_model_id": embedding_model_id,
        "answer_model_id": answer_model_id,
        "judge_model_id": judge_model_id,
    }
    if memory != "particles":
        key["memory"] = memory
    if not baselines:
        key["baselines"] = False
    return key


def _checkpoint_path(checkpoint_dir: Path, key: dict[str, object]) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return checkpoint_dir / f"memory-run-{digest}.jsonl"


def _load_checkpoint(path: Path, key: dict[str, object]) -> dict[str, _QuestionOutcome]:
    """Previously completed outcomes, keyed by question id. Tolerant by design.

    A missing file, a header that does not match ``key`` (filename-hash
    collision or a hand-copied file), or an unparseable line (a crash mid-
    append) each degrade to "not restored" — a checkpoint can reduce spend,
    never corrupt a run.
    """
    if not path.exists():
        return {}
    restored: dict[str, _QuestionOutcome] = {}
    header_ok = False
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except ValueError:
            continue  # partial trailing line from an interrupted append
        if i == 0:
            header_ok = raw.get("key") == key
            if not header_ok:
                log.warning("Checkpoint %s belongs to a different run; ignoring it.", path)
                return {}
            continue
        try:
            qid = str(raw["question_id"])
            restored[qid] = _QuestionOutcome.model_validate(raw["outcome"])
        except (KeyError, ValueError):
            continue
    return restored if header_ok else {}


def _append_checkpoint(
    path: Path, key: dict[str, object], question_id: str, outcome: _QuestionOutcome
) -> None:
    """Append one completed outcome (writing the header line on first use)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ""
    if not path.exists():
        lines += json.dumps({"format": _CHECKPOINT_FORMAT, "key": key}) + "\n"
    lines += (
        json.dumps({"question_id": question_id, "outcome": outcome.model_dump(mode="json")}) + "\n"
    )
    with path.open("a") as fh:
        fh.write(lines)


async def run_memory_benchmark(
    questions: list[MemoryQuestion],
    *,
    variant: str,
    dataset_revision: str,
    selection_seed: int,
    selection_limit: int | None,
    selection_types: list[str] | None = None,
    questions_total: int | None = None,
    top_k: int | None = None,
    work_dir: Path | None = None,
    keep_stores: bool = False,
    qa: bool = True,
    context_budget: int | None = None,
    abstraction: bool = False,
    consolidation: bool = False,
    dedup_judge: bool = False,
    reuse_stores: bool = False,
    progress: Callable[[str], None] | None = None,
    concurrency: int = 1,
    checkpoint_dir: Path | None = None,
    fresh: bool = False,
    heartbeat_seconds: float | None = None,
    pooled: bool = False,
    batch_qa: bool = False,
    memory: str = "particles",
    baselines: bool = True,
) -> MemoryBenchmarkReport:
    """Run the four-condition memory benchmark over ``questions``.

    ``questions`` is the already-selected subset (see
    :func:`particles.benchmark.memory.loader.select_questions`);
    ``questions_total`` is the variant's full question count for the subset
    disclosure. ``qa=False`` runs the retrieval-stage family only — all three
    QA conditions then render ``not run`` (the baseline rows are never
    omitted). A question whose pipeline raises degrades to a quality note,
    never an aborted run (the sibling harnesses' robustness contract).

    ``progress`` (when given) receives one human-readable line per completed
    question — a completion counter, the question id, retrieval recall, and
    per-condition QA marks — so a multi-hour run is never silent between the
    cost confirmation and the final table. The CLI wires it to stderr; the
    harness itself stays report-only and prints nothing on its own.

    ``concurrency`` runs up to N questions at once. Safe by construction —
    each question owns an ephemeral scratch store, the candidate cache
    dedups concurrent same-session extraction behind per-hash locks, and the
    report is folded in question order so its contents are byte-identical to
    a sequential run's (completion order only affects progress-line order).
    Scheduling is deliberately NOT part of the RunSelection comparability
    tuple. The practical ceiling is the API rate tier: past ~4–8 the extra
    parallelism converts to 429 retries, not wall-clock.

    ``checkpoint_dir`` (when given) makes the run interruptible at question
    granularity: each completed question's outcome is appended to a JSONL
    file named by the hash of the experiment's identity (comparability tuple
    + models + exact question set), and a re-run of the identical experiment
    restores those outcomes instead of re-spending. Any knob change hashes
    to a different file — a checkpoint can never feed the wrong experiment.
    Failed questions are deliberately not checkpointed (a resume retries
    them). ``None`` (default) disables checkpointing. ``fresh`` discards a
    matching checkpoint before the run (then checkpoints normally).

    ``heartbeat_seconds`` (with ``progress``) emits a status line at that
    cadence even while no question has completed — questions on the larger
    variants take many extraction calls, and minutes of silence between the
    preamble and the first per-question line reads as a hang. The heartbeat
    reports questions complete, extraction calls so far (unique + cached
    replays), and elapsed time.

    ``pooled`` fans each question's haystack-session extractions out under
    one :class:`~particles.llm.CompletionPool` — the consolidation
    extract pass's shape — so the ~50 extraction requests a question needs
    merge into a single Message Batches job at the half price
    instead of ~50 serial full-price calls. Extraction is the dominant cost of
    a run (a chatty session emits several thousand output tokens), so this
    roughly halves the bill for a batch-eligible provider. It changes only
    *how* the same requests are dispatched — same model, prompt, and budget
    — so it is deliberately not part of the RunSelection comparability
    tuple, exactly like ``concurrency``. With ``llm.batch.enabled`` off (or
    a provider without batching) the pool degrades to the same sequential
    calls, so the flag is always safe. Off by default: pooled dispatch trades
    latency for price (a batch's floor is one poll interval), which is the
    wrong default for the fixture smoke and the right one for a paid run.

    ``batch_qa`` is the same lever for the QA family (conditions ii–iv):
    retrieval still runs per question (concurrently, under ``concurrency``),
    but once every question has retrieved, the answer calls for **all**
    questions in one condition are submitted as a single ``complete_many`` job,
    and the judge calls over those answers as a second — six
    batches total (answer + judge per condition), all at the 50 % Message
    Batches price. None of the answerer/judge calls is latency-sensitive
    (nobody waits on a benchmark), so this is the right trade for a paid run;
    like ``pooled`` it is off by default (a batch's floor is one poll interval,
    the wrong trade for the fixture smoke) and the two compose. It changes only
    *how* the same requests are dispatched — same model, prompt, and budget —
    so it is deliberately not part of the RunSelection comparability tuple,
    like ``pooled`` and ``concurrency``. The one-pinned-answer/judge-model
    guards (§2/§5) are re-checked once per condition batch rather than
    once per question, so a config flip between conditions still raises
    :class:`SameModelViolation`. With ``llm.batch.enabled`` off (or a provider
    without batching, or a sub-``min_requests`` set) the batches degrade to the
    same sequential calls, so the flag is always safe.

    ``memory`` selects the memory under test. ``"particles"`` (default) is
    the store; ``"chunks"`` and ``"notes"`` are the comparator memories of
    :mod:`particles.benchmark.memory.comparators` — same questions, answer
    scaffold, judge, and retrieval scoring, only the memory differs — and the
    report's ``selection.memory`` names which ran (its ``qa_particles`` slot
    then carries the ``qa_chunks`` / ``qa_notes`` condition). A comparator
    needs no scratch store and skips the extraction pipeline entirely; the
    checkpoint key carries the memory kind, so a comparator run can never
    restore a particles outcome. ``baselines=False`` skips the
    ``qa_full_context`` / ``qa_no_memory`` conditions (they render ``not
    run``): under an identical tuple those two conditions are the *same
    calls* whichever memory is under test, so a comparator run reuses the
    particles run's baseline columns instead of paying for them again — the
    published table says so.
    """
    if memory not in MEMORY_KINDS:
        raise ValueError(f"memory must be one of {MEMORY_KINDS}, got {memory!r}")
    cfg = get_config().benchmark_memory
    effective_top_k = top_k if top_k is not None else cfg.top_k
    answer_scaffold = cfg.answer_scaffold
    judge_protocol = cfg.judge_protocol
    subject_rendering = cfg.subject_rendering
    total = questions_total if questions_total is not None else len(questions)
    call_retries = cfg.call_retries
    call_backoff = cfg.call_retry_backoff_seconds

    # Pre-flight: refuse before spending anything if the full-context baseline
    # would overflow the answering model's window. Enforced here rather than in
    # the CLI so every caller — the integration smoke, a programmatic run —
    # gets the same refusal.
    window_check = check_context_window(questions, variant=variant, qa=qa, baselines=baselines)
    if not window_check.fits:
        raise ContextWindowExceeded(_context_window_refusal(window_check))

    # ablation: force the abstraction pass on (auto mode, age gate
    # zeroed — the scratch stores' particles are minutes old) for this run's
    # process config, restoring the operator's settings afterwards. The knobs
    # ride the RunSelection tuple so a clamped/consolidated run can never be
    # silently compared against a stock one.
    ab_cfg = get_config().consolidation.abstraction
    ab_saved = (ab_cfg.enabled, ab_cfg.mode, ab_cfg.min_source_age_days)
    if abstraction:
        ab_cfg.enabled = True
        ab_cfg.mode = "auto"
        ab_cfg.min_source_age_days = 0

    owns_work_dir = work_dir is None
    resolved_work_dir = (
        Path(tempfile.mkdtemp(prefix="particles-benchmark-memory-"))
        if work_dir is None
        else work_dir
    )
    resolved_work_dir.mkdir(parents=True, exist_ok=True)

    caching_extractor = CachingExtractor(select_extractor(SourceType.CONVERSATION))

    quality_notes: list[str] = []
    retrieval_rows: list[RetrievalQuestionResult] = []
    recall_pairs: list[tuple[str, float]] = []
    precision_values: list[float] = []
    abstention_count = 0
    memory_condition = f"qa_{memory}"
    qa_acc = {
        "qa_particles": _QaAccumulator(memory_condition),
        "qa_full_context": _QaAccumulator("qa_full_context"),
        "qa_no_memory": _QaAccumulator("qa_no_memory"),
    }
    answer_model_id: str | None = None
    judge_model_id: str | None = None

    # §5 comparability pins (correction v1.74.2): the store's contents are a
    # function of the extraction model and the ranking of the embedding
    # model, so both are resolved up front and recorded on the run tuple.
    # Extraction is re-checked per question and refused on drift, same style
    # as the answer-model pin.
    # A comparator memory has no extraction model: the notes writer's model
    # takes that slot (it is the memory-writing model, resolved from the same
    # purpose), and the chunks comparator records none.
    notes_writer: NotesWriter | None = None
    if memory == "notes":
        notes_writer = NotesWriter(
            cache_path=None if checkpoint_dir is None else checkpoint_dir / "notes-cache.jsonl"
        )
    extraction_model_id: str | None = (
        _resolve_extraction_model()
        if memory == "particles"
        else (notes_writer.model_id if notes_writer is not None else None)
    )
    embedding_model_id = get_embedding_model_id()

    # store reuse. A replay skips deposit and extract outright (see
    # :func:`_run_question`), so not one extraction call is made — which is the
    # entire cost lever: the write side is ~97 % of a run's bill and every
    # read-side ablation wants a *fixed* store population anyway. The manifest
    # gate is what keeps that from being a silent trap (see
    # :func:`load_stores_manifest`).
    reused_from: dict[str, str] | None = None
    if reuse_stores:
        if memory != "particles":
            raise StoreReuseError(
                f"--reuse-stores applies to the particles store; memory={memory!r} "
                f"builds no scratch store."
            )
        if work_dir is None:
            raise StoreReuseError("--reuse-stores requires --store-dir naming the persisted set.")
        reused_from = load_stores_manifest(
            work_dir,
            _write_side_tuple(
                dataset_revision=dataset_revision,
                variant=variant,
                selection_seed=selection_seed,
                selection_limit=selection_limit,
                selection_types=list(selection_types or []),
                extraction_model_id=extraction_model_id or "",
                embedding_model_id=embedding_model_id,
            ),
            [q.question_id for q in questions],
        )

    sem = asyncio.Semaphore(max(1, concurrency))
    outcomes: list[_QuestionOutcome | None] = [None] * len(questions)
    # ``batch_qa`` handoff: retrieval (phase A) stashes each live question's
    # three context blocks here for the post-retrieval batch phase to answer +
    # judge. In-memory only — never the checkpointed ``_QuestionOutcome`` (a
    # full-context block is the whole haystack; persisting 150 of them would
    # bloat the JSONL). Empty and unused when ``batch_qa`` is off.
    contexts_by_slot: dict[int, dict[str, str]] = {}
    completed_count = 0

    # Checkpoint restore (question granularity). The key embeds the full
    # experiment identity, so a restored outcome is definitionally from this
    # same experiment; the answer/judge pins are seeded from the key when
    # restored QA marks exist (they resolved from the same config).
    checkpoint_file: Path | None = None
    checkpoint_key: dict[str, object] | None = None
    restored_count = 0
    if checkpoint_dir is not None:
        checkpoint_key = _run_checkpoint_key(
            dataset_revision=dataset_revision,
            variant=variant,
            top_k=effective_top_k,
            context_budget=context_budget,
            abstraction=abstraction,
            consolidation=consolidation,
            dedup_judge=dedup_judge,
            reuse_stores=reuse_stores,
            subject_rendering=subject_rendering,
            answer_scaffold=answer_scaffold,
            judge_protocol=judge_protocol,
            read_side_knobs=_read_side_ablation_knobs(),
            qa=qa,
            extraction_model_id=extraction_model_id,
            embedding_model_id=embedding_model_id,
            answer_model_id=_resolve_answer_model(),
            judge_model_id=get_provider("benchmark").provider_model,
            memory=memory,
            baselines=baselines,
        )
        checkpoint_file = _checkpoint_path(checkpoint_dir, checkpoint_key)
        if fresh:
            checkpoint_file.unlink(missing_ok=True)
        restored = _load_checkpoint(checkpoint_file, checkpoint_key)
        for slot, question in enumerate(questions):
            outcome = restored.get(question.question_id)
            if outcome is not None:
                outcomes[slot] = outcome
                restored_count += 1
                if outcome.qa_marks:
                    answer_model_id = str(checkpoint_key["answer_model_id"])
                    judge_model_id = str(checkpoint_key["judge_model_id"])
        completed_count = restored_count
        if restored_count and progress is not None:
            progress(
                f"resuming: {restored_count}/{len(questions)} question(s) restored "
                f"from {checkpoint_file.name}"
            )

    async def _process(slot: int, question: MemoryQuestion) -> None:
        """One question end-to-end (scratch store → retrieval → QA), semaphore-gated.

        Writes its outcome into ``outcomes[slot]`` and emits one progress line
        on completion. Per-question failures degrade into the outcome; only a
        :class:`SameModelViolation` escapes (aborting the TaskGroup). The pin
        check-and-set blocks are synchronous, so they are atomic under
        asyncio's cooperative scheduling.
        """
        nonlocal completed_count, answer_model_id, judge_model_id
        async with sem:
            if memory == "particles":
                current_extraction = _resolve_extraction_model()
                if current_extraction != extraction_model_id:
                    raise SameModelViolation(
                        f"Extraction-model mismatch mid-run: llm.extraction resolved to "
                        f"{current_extraction!r} but the run is pinned to "
                        f"{extraction_model_id!r}. The store's contents are a function of "
                        f"the extraction model; refusing to continue."
                    )
            outcome = _QuestionOutcome()
            db_path = resolved_work_dir / f"{question.question_id}.db"
            try:
                if memory != "particles":
                    comparator = await run_comparator_question(
                        question,
                        memory=memory,
                        top_k=effective_top_k,
                        notes_writer=notes_writer,
                        context_budget=context_budget,
                    )
                    result = _QuestionResult(
                        retrieval=comparator.retrieval,
                        particles_context=comparator.context,
                        notes=comparator.notes,
                    )
                else:
                    async with (
                        scratch_store(db_path) as session_factory,
                        session_factory() as session,
                    ):
                        result = await _run_question(
                            session,
                            question,
                            caching_extractor,
                            top_k=effective_top_k,
                            context_budget=context_budget,
                            subject_rendering=subject_rendering,
                            abstraction=abstraction,
                            consolidation=consolidation,
                            dedup_judge=dedup_judge,
                            reuse_stores=reuse_stores,
                            db_path=db_path,
                            session_factory=session_factory if pooled else None,
                        )
            except SameModelViolation:
                raise
            except Exception as exc:  # noqa: BLE001 — one bad question must not abort the run
                outcome.failed_note = (
                    f"Question {question.question_id}: pipeline raised {exc!r}; skipped"
                )
                outcomes[slot] = outcome
                completed_count += 1
                if progress is not None:
                    progress(
                        f"[{completed_count}/{len(questions)}] {question.question_id}: "
                        f"FAILED ({exc!r}) — skipped"
                    )
                return
            finally:
                if not keep_stores:
                    _remove_scratch_files(db_path)

            outcome.retrieval = result.retrieval
            outcome.notes.extend(result.notes)
            outcome.context_particle_ids = list(result.context_particle_ids)

            if qa:
                # Conditions ii–iv share up to three context blocks (the
                # baselines are skipped under ``baselines=False``). Under
                # ``batch_qa`` the answer/judge calls are deferred to the
                # post-retrieval batch phase (:func:`_run_qa_batches`) —
                # retrieval-only here, and no checkpoint yet because QA is not
                # done. Otherwise QA runs inline, one pinned answer model
                # enforced per call.
                contexts = {"qa_particles": result.particles_context}
                if baselines:
                    contexts["qa_full_context"] = _full_context(question)
                    contexts["qa_no_memory"] = "(no context is available)"
                if batch_qa:
                    contexts_by_slot[slot] = contexts
                    outcomes[slot] = outcome
                    completed_count += 1
                    if progress is not None and outcome.retrieval is not None:
                        progress(
                            _progress_line(
                                completed_count, len(questions), outcome.retrieval, {}, qa=False
                            )
                        )
                    return
                for condition, context_block in contexts.items():
                    current_model = _resolve_answer_model()
                    if answer_model_id is None:
                        answer_model_id = current_model
                    elif current_model != answer_model_id:
                        raise SameModelViolation(
                            f"Answer-model mismatch across QA conditions: {condition} resolved "
                            f"llm.benchmark_answer to {current_model!r} but the run is pinned to "
                            f"{answer_model_id!r}. A same-model comparison is the validity "
                            f"condition of the QA family; refusing to continue."
                        )
                    answer = await _scored_call(
                        "benchmark_answer",
                        _answer_prompt(question, context_block),
                        max_tokens=_ANSWER_MAX_TOKENS,
                        system=_answer_system(),
                        retries=call_retries,
                        backoff_seconds=call_backoff,
                    )
                    if answer.text is None:
                        outcome.notes.append(_excluded_note(question, condition, "answer", answer))
                        outcome.qa_excluded[condition] = answer.excluded
                        continue
                    _record_qa_text(outcome, condition, answer=answer.text)

                    current_judge = get_provider("benchmark").provider_model
                    if judge_model_id is None:
                        judge_model_id = current_judge
                    elif current_judge != judge_model_id:
                        raise SameModelViolation(
                            f"Judge-model mismatch mid-run: llm.benchmark resolved to "
                            f"{current_judge!r} but the run is pinned to {judge_model_id!r}. "
                            f"One judge per table is what makes its accuracies comparable "
                            f"; refusing to continue."
                        )
                    verdict = await _scored_call(
                        "benchmark",
                        judge_prompt(question, answer.text),
                        max_tokens=_JUDGE_MAX_TOKENS,
                        retries=call_retries,
                        backoff_seconds=call_backoff,
                    )
                    if verdict.text is None:
                        outcome.notes.append(_excluded_note(question, condition, "judge", verdict))
                        outcome.qa_excluded[condition] = verdict.excluded
                        continue
                    outcome.qa_marks[condition] = parse_judge_verdict(verdict.text)
                    _record_qa_text(outcome, condition, verdict=verdict.text)

            outcomes[slot] = outcome
            if checkpoint_file is not None and checkpoint_key is not None:
                _append_checkpoint(checkpoint_file, checkpoint_key, question.question_id, outcome)
            completed_count += 1
            if progress is not None and outcome.retrieval is not None:
                progress(
                    _progress_line(
                        completed_count, len(questions), outcome.retrieval, outcome.qa_marks, qa=qa
                    )
                )

    async def _run_qa_batches() -> None:
        """Answer + judge the whole QA family through the batch path.

        Called once, after every question has retrieved (``batch_qa``). Each
        condition's answer calls go out as one ``complete_many`` job and the
        judge verdicts over those answers as a second — six batches (answer +
        judge per condition), each at the 50% Message Batches price. The
        one-model pins (§2/§5) are re-resolved once per condition batch
        (not per question) and still raise :class:`SameModelViolation` on a
        mid-run config flip.

        A ``None`` from ``complete_many`` carries **no cause** — a per-request
        failure, an expired batch, and a reply with no text block all arrive as
        the same ``None``. So that one call is re-issued through
        :func:`_scored_call` on the inline path, where the adapter raises a
        typed error: the retry both recovers a transient failure and, if it
        persists, classifies it as budget or infra for the disclosure. Only the
        handful of failures pay the full (non-batch) price.
        QA-complete questions are checkpointed here (deferred from
        :func:`_process`, which returns before its own checkpoint under
        ``batch_qa``).
        """
        nonlocal answer_model_id, judge_model_id
        live = [
            slot
            for slot in range(len(questions))
            if slot in contexts_by_slot
            and (o := outcomes[slot]) is not None
            and o.retrieval is not None
        ]
        if not live:
            return

        # The conditions actually built by _process — all three, or only the
        # memory-under-test slot when ``baselines=False``.
        active_conditions = [c for c in QA_CONDITIONS if c in contexts_by_slot[live[0]]]
        answers_by_condition: dict[str, list[_CallOutcome]] = {}
        for condition in active_conditions:
            current_model = _resolve_answer_model()
            if answer_model_id is None:
                answer_model_id = current_model
            elif current_model != answer_model_id:
                raise SameModelViolation(
                    f"Answer-model mismatch across QA conditions: {condition} resolved "
                    f"llm.benchmark_answer to {current_model!r} but the run is pinned to "
                    f"{answer_model_id!r}. A same-model comparison is the validity "
                    f"condition of the QA family; refusing to continue."
                )
            requests = [
                CompletionRequest(
                    prompt=_answer_prompt(questions[slot], contexts_by_slot[slot][condition]),
                    system=_answer_system(),
                )
                for slot in live
            ]
            if progress is not None:
                progress(
                    f"submitting answer batch [{condition}]: {len(requests)} request(s) "
                    f"via Message Batches (polling — the run is not hung)"
                )
            batch = await complete_many(
                "benchmark_answer",
                requests,
                max_tokens=_ANSWER_MAX_TOKENS,
                temperature=0.0,
                latency_tolerant=True,
            )
            answers = [
                _CallOutcome(text=text)
                if text is not None
                else await _scored_call(
                    "benchmark_answer",
                    requests[i].prompt,
                    max_tokens=_ANSWER_MAX_TOKENS,
                    system=_answer_system(),
                    retries=call_retries,
                    backoff_seconds=call_backoff,
                )
                for i, text in enumerate(batch)
            ]
            answers_by_condition[condition] = answers
            if progress is not None:
                progress(
                    f"answer batch [{condition}]: "
                    f"{sum(a.text is not None for a in answers)}/{len(requests)} answered"
                )

        for condition in active_conditions:
            current_judge = get_provider("benchmark").provider_model
            if judge_model_id is None:
                judge_model_id = current_judge
            elif current_judge != judge_model_id:
                raise SameModelViolation(
                    f"Judge-model mismatch mid-run: llm.benchmark resolved to "
                    f"{current_judge!r} but the run is pinned to {judge_model_id!r}. "
                    f"One judge per table is what makes its accuracies comparable "
                    f"; refusing to continue."
                )
            judge_requests: list[CompletionRequest] = []
            judged_slots: list[int] = []
            for slot, answer in zip(live, answers_by_condition[condition], strict=True):
                outcome = outcomes[slot]
                assert outcome is not None  # ``live`` filtered on this
                if answer.text is None:
                    outcome.notes.append(
                        _excluded_note(questions[slot], condition, "answer", answer)
                    )
                    outcome.qa_excluded[condition] = answer.excluded
                    continue
                _record_qa_text(outcome, condition, answer=answer.text)
                judge_requests.append(
                    CompletionRequest(prompt=judge_prompt(questions[slot], answer.text))
                )
                judged_slots.append(slot)
            if not judge_requests:
                continue
            if progress is not None:
                progress(
                    f"submitting judge batch [{condition}]: {len(judge_requests)} request(s) "
                    f"via Message Batches (polling — the run is not hung)"
                )
            verdict_batch = await complete_many(
                "benchmark",
                judge_requests,
                max_tokens=_JUDGE_MAX_TOKENS,
                temperature=0.0,
                latency_tolerant=True,
            )
            verdicts = [
                _CallOutcome(text=text)
                if text is not None
                else await _scored_call(
                    "benchmark",
                    judge_requests[i].prompt,
                    max_tokens=_JUDGE_MAX_TOKENS,
                    retries=call_retries,
                    backoff_seconds=call_backoff,
                )
                for i, text in enumerate(verdict_batch)
            ]
            for slot, verdict in zip(judged_slots, verdicts, strict=True):
                outcome = outcomes[slot]
                assert outcome is not None
                if verdict.text is None:
                    outcome.notes.append(
                        _excluded_note(questions[slot], condition, "judge", verdict)
                    )
                    outcome.qa_excluded[condition] = verdict.excluded
                else:
                    outcome.qa_marks[condition] = parse_judge_verdict(verdict.text)
                    _record_qa_text(outcome, condition, verdict=verdict.text)

        for slot in live:
            outcome = outcomes[slot]
            assert outcome is not None
            if checkpoint_file is not None and checkpoint_key is not None:
                _append_checkpoint(
                    checkpoint_file, checkpoint_key, questions[slot].question_id, outcome
                )
        if progress is not None:
            progress(
                f"QA batches complete: {len(live)} question(s) scored across "
                f"{len(QA_CONDITIONS)} condition(s)"
            )

    heartbeat_task: asyncio.Task[None] | None = None
    if progress is not None and heartbeat_seconds:
        started_monotonic = time.monotonic()
        progress_cb = progress

        async def _heartbeat() -> None:
            while True:
                await asyncio.sleep(heartbeat_seconds)
                elapsed = int(time.monotonic() - started_monotonic)
                if notes_writer is not None:
                    writes = (
                        f"notes written so far: {notes_writer.misses} unique + "
                        f"{notes_writer.hits} cached"
                    )
                elif memory == "particles":
                    writes = (
                        f"extraction calls so far: {caching_extractor.misses} unique + "
                        f"{caching_extractor.hits} cached"
                    )
                else:
                    writes = "no write-time LLM calls (chunks)"
                progress_cb(
                    f"  … {completed_count}/{len(questions)} question(s) complete; "
                    f"{writes}; elapsed {elapsed // 60}m{elapsed % 60:02d}s"
                )

        heartbeat_task = asyncio.create_task(_heartbeat())

    # Only the particles memory deposits; the comparators build no store.
    blob_isolation = (
        isolated_blob_dir(resolved_work_dir) if memory == "particles" else contextlib.nullcontext()
    )
    try:
        try:
            with blob_isolation:
                async with asyncio.TaskGroup() as tg:
                    for slot, question in enumerate(questions):
                        if outcomes[slot] is None:  # not restored from a checkpoint
                            tg.create_task(_process(slot, question))
        except* SameModelViolation as group:
            raise group.exceptions[0] from None
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        ab_cfg.enabled, ab_cfg.mode, ab_cfg.min_source_age_days = ab_saved
        if owns_work_dir and not keep_stores:
            shutil.rmtree(resolved_work_dir, ignore_errors=True)
        elif not keep_stores:
            # A caller-owned work dir outlives the run, but its blobs are
            # scratch like the per-question .db files already removed above.
            shutil.rmtree(resolved_work_dir / BLOB_SUBDIR, ignore_errors=True)
        elif keep_stores and memory == "particles" and not reuse_stores:
            # Stamp the kept set so a later --reuse-stores run can prove it is
            # replaying the population it thinks it is. Written in the finally
            # so an interrupted run still leaves its partial set usable — the
            # per-question .db existence check in load_stores_manifest is what
            # catches a set that is short.
            with contextlib.suppress(OSError):
                write_stores_manifest(
                    resolved_work_dir,
                    _write_side_tuple(
                        dataset_revision=dataset_revision,
                        variant=variant,
                        selection_seed=selection_seed,
                        selection_limit=selection_limit,
                        selection_types=list(selection_types or []),
                        extraction_model_id=extraction_model_id or "",
                        embedding_model_id=embedding_model_id,
                    ),
                    [
                        q.question_id
                        for q, o in zip(questions, outcomes, strict=True)
                        if o is not None and o.failed_note is None
                    ],
                )

    # QA batch phase: every question has now retrieved, so a whole condition's
    # answer calls go out as one Message Batches job and the judge
    # calls over them as a second. Deliberately after the retrieval TaskGroup's
    # finally — it needs neither the scratch stores nor the abstraction-config
    # override, only the in-memory context blocks — so a SameModelViolation it
    # raises propagates with the stores already cleaned up.
    if qa and batch_qa:
        await _run_qa_batches()

    # Fold outcomes in QUESTION order — the report is byte-identical to a
    # sequential run's regardless of completion order under concurrency.
    for question, outcome in zip(questions, outcomes, strict=True):
        if outcome is None:  # unreachable after a successful TaskGroup; defensive
            continue
        if outcome.failed_note is not None or outcome.retrieval is None:
            if outcome.failed_note is not None:
                quality_notes.append(outcome.failed_note)
            continue
        row = outcome.retrieval
        retrieval_rows.append(row)
        if row.recall_at_k is None or row.precision_at_k is None:
            # Zero-evidence (abstention) question: unscoreable at the
            # retrieval stage — excluded from the aggregates and disclosed
            # via abstention_questions, never blended as a vacuous 1.0 /
            # deterministic 0.0 (correction v1.74.2).
            abstention_count += 1
        else:
            recall_pairs.append((question.question_type, row.recall_at_k))
            precision_values.append(row.precision_at_k)
        quality_notes.extend(outcome.notes)
        if qa:
            # Iterate the fixed condition order (not the outcome dict's) so the
            # drill-down rows are ordered identically in every report.
            for condition in QA_CONDITIONS:
                # The context ids belong to the memory-under-test slot only:
                # the baselines see the haystack or nothing, never particles.
                context_ids = outcome.context_particle_ids if condition == "qa_particles" else []
                if condition in outcome.qa_excluded:
                    qa_acc[condition].record_excluded(
                        question,
                        outcome.qa_excluded[condition],
                        answer=outcome.qa_answers.get(condition),
                        context_particle_ids=context_ids,
                    )
                elif condition in outcome.qa_marks:
                    qa_acc[condition].record(
                        question,
                        outcome.qa_marks[condition],
                        answer=outcome.qa_answers.get(condition),
                        verdict=outcome.qa_verdicts.get(condition),
                        context_particle_ids=context_ids,
                    )

    if restored_count:
        quality_notes.append(
            f"Checkpoint: restored {restored_count} previously completed question(s); "
            f"outcomes accumulate in {checkpoint_file}."
        )

    if notes_writer is not None and (notes_writer.hits or notes_writer.misses):
        quality_notes.append(
            f"Notes writer ({notes_writer.model_id}): {notes_writer.misses} unique session "
            f"note(s) written, {notes_writer.hits} cache replay(s), "
            f"{notes_writer.failures} failure(s)."
        )
    if memory != "particles" and not baselines:
        quality_notes.append(
            f"Comparator run (memory={memory}): qa_full_context / qa_no_memory were not run — "
            f"under an identical tuple they are the same calls as the particles run's; "
            f"reuse those columns and say so."
        )
    if caching_extractor.hits or caching_extractor.misses:
        quality_notes.append(
            f"Candidate cache: {caching_extractor.misses} unique session extraction(s), "
            f"{caching_extractor.hits} cache replay(s)."
        )
    if reused_from is not None:
        quality_notes.append(
            f"Stores REUSED from {resolved_work_dir} — this run deposited and extracted "
            f"nothing; the particle population is the one that store set was built with "
            f"(extraction model {reused_from.get('extraction_model_id')}, "
            f"similarity_threshold {reused_from.get('extraction.similarity_threshold')}, "
            f"duplicate_suppression "
            f"{reused_from.get('extraction.duplicate_suppression.enabled')})."
        )
    for decay_note in (decay_quality_note(), decay_reference_note(questions)):
        if decay_note is not None:
            quality_notes.append(decay_note)

    selection = RunSelection(
        dataset_revision=dataset_revision,
        variant=variant,
        sample_seed=selection_seed,
        question_limit=selection_limit,
        question_types=list(selection_types or []),
        questions_selected=len(questions),
        questions_total=total,
        top_k=effective_top_k,
        answer_model_id=answer_model_id,
        judge_model_id=judge_model_id,
        extraction_model_id=extraction_model_id,
        embedding_model_id=embedding_model_id,
        context_budget_tokens=context_budget,
        abstraction=abstraction,
        consolidation=consolidation,
        dedup_judge=dedup_judge,
        stores_reused=reuse_stores,
        reused_from=reused_from,
        answer_scaffold=answer_scaffold,
        judge_protocol=judge_protocol,
        subject_rendering=subject_rendering,
        thresholds=_thresholds_snapshot(),
        memory=memory,
    )
    retrieval = RetrievalStageMetrics(
        questions=len(retrieval_rows) - abstention_count,
        abstention_questions=abstention_count,
        mean_recall_at_k=(
            sum(r for _, r in recall_pairs) / len(recall_pairs) if recall_pairs else 0.0
        ),
        mean_precision_at_k=(
            sum(precision_values) / len(precision_values) if precision_values else 0.0
        ),
        recall_by_type=mean_by_type(recall_pairs),
        per_question=retrieval_rows,
    )

    def _condition_metrics(condition: str) -> QaConditionMetrics | None:
        acc = qa_acc[condition]
        # ``rows``, not ``pairs``: a condition every one of whose calls was
        # excluded still *ran*, and must render with its exclusion counts
        # rather than as ``not run`` — the one case where those two readings
        # differ, and the dishonest one is the silent one.
        if not qa or not acc.rows or answer_model_id is None:
            return None
        return acc.to_metrics(answer_model_id)

    return MemoryBenchmarkReport(
        selection=selection,
        retrieval_stage=retrieval,
        qa_particles=_condition_metrics("qa_particles"),
        qa_full_context=_condition_metrics("qa_full_context"),
        qa_no_memory=_condition_metrics("qa_no_memory"),
        quality_notes=quality_notes,
    )


class _QuestionOutcome(BaseModel):
    """One question's full outcome, folded into the report in question order."""

    retrieval: RetrievalQuestionResult | None = None
    qa_marks: dict[str, bool] = Field(default_factory=dict)
    #: condition → why that condition produced no verdict (one of
    #: ``QA_EXCLUSION_KINDS``). Mutually exclusive with ``qa_marks`` per
    #: condition. A new *optional* field rather than a widened ``qa_marks``:
    #: checkpoints from earlier runs deserialize unchanged, so no paid outcome
    #: became unrestorable.
    qa_excluded: dict[str, str] = Field(default_factory=dict)
    #: condition → the answering model's reply / the judge's raw verdict text,
    #: and the ids of the particles that formed the ``qa_particles`` context
    #: block. The audit trail behind ``qa_marks``: without it a checkpointed
    #: (and hence never re-run) outcome could say *wrong* and nothing else.
    #: The text pair is empty when ``benchmark.record_claim_text`` is off —
    #: suppressed at production by :func:`_record_qa_text`, so the JSONL
    #: never holds it. All three are new *optional* fields, like
    #: ``qa_excluded``: an earlier checkpoint deserializes unchanged and its
    #: restored rows simply carry no text (the run is not re-paid to get it).
    #: The full-context block is deliberately NOT here — see
    #: ``contexts_by_slot`` in :func:`run_memory_benchmark`.
    qa_answers: dict[str, str] = Field(default_factory=dict)
    qa_verdicts: dict[str, str] = Field(default_factory=dict)
    context_particle_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    failed_note: str | None = None


def _record_qa_text(
    outcome: _QuestionOutcome,
    condition: str,
    *,
    answer: str | None = None,
    verdict: str | None = None,
) -> None:
    """Keep the answer / verdict text on the outcome — unless the operator said not to.

    ``benchmark.record_claim_text`` is honoured here, at production, for the
    same reason the extractor benchmark reads it in ``_record_emitted_claims``
    rather than at persist time: the outcome is checkpointed to JSONL and
    folded into a run file, and an answer quotes the corpus the model was
    shown. Suppressing at the one point the text enters the model means no
    downstream sink has to remember to scrub it.
    """
    if not get_config().benchmark.record_claim_text:
        return
    if answer is not None:
        outcome.qa_answers[condition] = answer
    if verdict is not None:
        outcome.qa_verdicts[condition] = verdict


class _QuestionResult(BaseModel):
    """Internal per-question pipeline output handed back to the run loop."""

    retrieval: RetrievalQuestionResult
    particles_context: str
    #: Ids of the particles whose lines make up ``particles_context``, in
    #: rank order and after the budget clamp. Empty for a comparator memory.
    context_particle_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _progress_line(
    index: int,
    total: int,
    row: RetrievalQuestionResult,
    qa_marks: dict[str, bool],
    *,
    qa: bool,
) -> str:
    """One per-question progress line: position, id, recall, QA marks."""
    recall = (
        "recall n/a (abstention)" if row.recall_at_k is None else f"recall {row.recall_at_k:.2f}"
    )
    line = f"[{index}/{total}] {row.question_id}: {row.particles_retrieved} particle(s), {recall}"
    if qa:
        marks = " ".join(
            ("✓" if qa_marks[c] else "✗") if c in qa_marks else "·"
            for c in ("qa_particles", "qa_full_context", "qa_no_memory")
        )
        line += f" · qa {marks}"
    return line


async def _sessions_by_particle(
    session: AsyncSession,
    particles: list[Particle],
    rows: dict[str, SourceRow],
) -> dict[str, set[str]]:
    """Haystack session ids each retrieved particle stands on.

    An extracted particle carries a SOURCE provenance ref, so ``rows`` already
    resolves it to one corpus entry and hence one session. A **
    derived particle carries no SOURCE ref at all** — only PARTICLE-typed
    premise refs (§3's field-reuse convention puts the premise particle id in
    ``corpus_entry_id``) — so the ``load_source_rows`` chain dead-ends on it
    and, before this function existed, every promoted abstraction that reached
    top-k scored as a retrieval non-hit. That penalized promotion structurally,
    however good the abstraction (Precision@10 1.000 → 0.986 in the
    oracle A/B of 2026-07-18).

    A derived particle is therefore credited with the **union** of its
    premises' sessions. The two metrics read that union differently, and the
    difference is deliberate:

    * **Recall** — every covered evidence session counts as hit. One
      abstraction summarising three evidence sessions is genuine retrieval of
      all three, and recall asks what share of the labeled evidence the top-k
      reached.
    * **Precision** — the derived particle is *one* retrieved item, hit when at
      least one premise session is in the evidence set. Precision's denominator
      is retrieved items, so a multi-session abstraction must not multiply-count
      itself into the numerator. The caller enforces this by appending exactly
      one representative session id per particle.

    Returns ``{particle_id: session_ids}``; a particle whose chain resolves
    nowhere is absent (never a spurious empty-string session).
    """
    out: dict[str, set[str]] = {}
    derived: list[Particle] = []
    for p in particles:
        _pub_at, _source_type, _entry_id, uri_r, _author = rows.get(
            p.id, (None, "", None, None, None)
        )
        sid = session_id_from_uri(uri_r)
        if sid is not None:
            out[p.id] = {sid}
        if is_derived(p):
            derived.append(p)
    if not derived:
        return out

    # One level of transitivity. A premise can itself be derived only when
    # consolidation.abstraction.max_depth > 1, which ships at 1;
    # raising it means recursing here.
    premise_ids = sorted({pid for d in derived for pid in premise_ids_of(d)})
    premises = await get_particles_by_ids(session, premise_ids)
    premise_rows = await load_source_rows(session, list(premises.values()))
    for d in derived:
        sids = {
            sid
            for pid in premise_ids_of(d)
            if (sid := session_id_from_uri(premise_rows.get(pid, (None, "", None, None, None))[3]))
            is not None
        }
        if sids:
            out.setdefault(d.id, set()).update(sids)
    return out


async def _extract_pooled(
    session_factory: async_sessionmaker[AsyncSession],
    deposits: list[tuple[str, str]],
    extractor: CachingExtractor,
) -> None:
    """Extract ``deposits`` concurrently under one ``CompletionPool``.

    Mirrors the consolidation extract pass: one asyncio task per deposit, each
    in its own session (an ``AsyncSession`` is not shareable across tasks),
    all registered as pool participants up front so no wave dispatches until
    every task has parked its request group. Any task's exception is re-raised
    after the gather so the caller's per-question failure handling sees it —
    the same contract as the serial loop, where the first raising deposit
    aborts the question.
    """
    from particles.llm import CompletionPool

    if not deposits:
        return
    pool = CompletionPool("extraction", expected_participants=len(deposits))

    async def _one(entry_id: str, snapshot_id: str) -> None:
        async with pool.participant(), session_factory() as task_session:
            await extract_snapshot(
                task_session,
                entry_id,
                snapshot_id,
                extractor=extractor,
                completion_pool=pool,
            )
            await task_session.commit()

    outcomes = await asyncio.gather(
        *(_one(entry_id, snapshot_id) for entry_id, snapshot_id in deposits),
        return_exceptions=True,
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome


async def _run_question(
    session: AsyncSession,
    question: MemoryQuestion,
    extractor: CachingExtractor,
    *,
    top_k: int,
    context_budget: int | None = None,
    subject_rendering: str = "uuids",
    abstraction: bool = False,
    consolidation: bool = False,
    dedup_judge: bool = False,
    reuse_stores: bool = False,
    db_path: Path | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> _QuestionResult:
    """Deposit → extract → retrieve → provenance-score one question (§3).

    ``session_factory`` (when given) selects pooled extraction: every deposit
    is extracted concurrently in its own scratch-store session under one
    ``CompletionPool`` (see :func:`run_memory_benchmark` ``pooled``). ``None``
    keeps the serial full-price path on the caller's session.

    ``reuse_stores`` skips the write side outright: nothing is deposited and
    nothing extracted, so the replay reads exactly the population the
    preparing run left. It used to re-deposit and rely on every session coming
    back ``unchanged``, which the dataset does not honour: four of the 150
    inaugural haystacks repeat one session id under two dates, the two
    renderings share a ``uri_r`` and differ in text, and so each replay
    re-versioned and re-extracted both: 8 paid calls the estimate projected as
    zero, and four stores re-rolled under every arm that was meant to hold the
    population fixed.
    """
    notes: list[str] = []

    # 1. Deposit each haystack session as a CONVERSATION corpus entry.
    deposits: list[tuple[str, str]] = []
    for mem_session in () if reuse_stores else question.sessions:
        entry_id, snapshot_id, unchanged = await deposit_text_versioned(
            session,
            text=render_session_text(mem_session),
            uri_r=session_uri(question.question_id, mem_session.session_id),
            source_type=SourceType.CONVERSATION,
            mutability=Mutability.STABLE,
            deposited_by="benchmark-memory",
            content_published_at=parse_session_date(mem_session.date),
        )
        await session.commit()
        if not unchanged:
            deposits.append((entry_id, snapshot_id))

    # 2. Standard extract (candidate-cached) + §6.6 reconciliation per store.
    if session_factory is None:
        for entry_id, snapshot_id in deposits:
            await extract_snapshot(session, entry_id, snapshot_id, extractor=extractor)
            await session.commit()
    else:
        await _extract_pooled(session_factory, deposits, extractor)

    # 2a. Ablation: the controlled instrument — run the
    # dream cycle's own pass list against this scratch store before retrieval,
    # so the "on" arm retrieves over the population reconcile + census left
    # behind. ``scope="store"`` because a fresh store has no prior run to take
    # a delta watermark from; the projection tail is skipped (there is no
    # MEMORY.md to render for a throwaway store); and the cycle lock is
    # per-store, or a concurrent run would silently skip every question after
    # the first (see run_consolidation's ``lock_path``).
    if consolidation:
        cons_report = await run_consolidation(
            session,
            store=f"benchmark-{question.question_id}",
            scope="store",
            actor="benchmark-memory-consolidate",
            projection_runner=None,
            projection_skip_reason="scratch store has no projection manifest",
            lock_path=(db_path.with_suffix(".consolidate.lock") if db_path is not None else None),
        )
        await session.commit()
        if cons_report.outcome == "skipped":
            notes.append(
                f"Question {question.question_id}: consolidation SKIPPED "
                f"({cons_report.skip_reason}) — this question's arm is not "
                f"consolidated"
            )
        else:
            notes.append(
                f"Question {question.question_id}: consolidation demoted "
                f"{cons_report.reconcile_demoted} (probed "
                f"{cons_report.reconcile_probes_run} of "
                f"{cons_report.reconcile_candidate_pairs} pair(s)); "
                f"{cons_report.duplicate_candidate_pairs_total} duplicate pair(s) seen"
            )

    # 2a-bis. Ablation: the co-evidential LLM judge in APPLY mode.
    # Judged PARAPHRASE pairs are linked CO_EVIDENTIAL, and the ranker collapses
    # each CO_EVIDENTIAL group *within* top-k — so this frees
    # slots that near-duplicate claims would otherwise occupy, which is what
    # makes it an instrument on Precision@k and on condition ii's context
    # rather than bookkeeping. ``confirmed=True`` because the interactive
    # apply-confirmation threshold has no meaning on a throwaway store.
    if dedup_judge:
        judge_report = await suggest_co_evidential(session, mode=SuggestMode.APPLY, confirmed=True)
        await session.commit()
        notes.append(
            f"Question {question.question_id}: dedup judge applied "
            f"{judge_report.applied_pairs} of {judge_report.judged_pairs} judged "
            f"({judge_report.total_candidates} candidate pair(s)) CO_EVIDENTIAL"
        )

    # 2b. Ablation: run the abstraction pass on the scratch store
    # between extract and retrieve (the run loop forced auto mode + a zero
    # age gate), so retrieval sees the consolidated population.
    if abstraction:
        ab_report = await run_abstraction_pass(session)
        await session.commit()
        if ab_report.promoted_particle_ids:
            notes.append(
                f"Question {question.question_id}: abstraction promoted "
                f"{len(ab_report.promoted_particle_ids)} derived belief(s) "
                f"from {ab_report.clusters_found} cluster(s)"
            )

    # 3. Top-k retrieval WITHOUT the NL respond step (that is condition ii's
    #    job, under the pinned answer model).
    scored = await retrieve_ranked(session, QueryRequest(question=question.question, top_k=top_k))
    particles = [p for p, _, _ in scored]

    # 4. Score retrieval through the real provenance chain:
    #    particle → corpus entry → uri_r → session id, resolving a
    #    derived particle through its premises (see _sessions_by_particle).
    # An abstention question is unscoreable at the retrieval stage by
    # protocol, not by label shape: the cleaned dataset labels every
    # ``*_abs`` question with its near-miss session (30/30 on ``s``), so
    # keying the exclusion off an empty ``answer_session_ids`` never fired
    # and blended those questions into the aggregates. Key off the
    # protocol flag (as the QA judge already does) and treat the labeled
    # near-miss session as no evidence.
    evidence = set() if question.is_abstention else set(question.answer_session_ids)
    rows = await load_source_rows(session, particles)
    sessions_by_id = await _sessions_by_particle(session, particles, rows)
    retrieved_session_ids: list[str | None] = []
    pub_at_by_id: dict[str, datetime | None] = {}
    for p in particles:
        pub_at, _source_type, _entry_id, _uri_r, _author = rows.get(
            p.id, (None, "", None, None, None)
        )
        # Precision counts one entry per retrieved particle, so a
        # multi-session abstraction is represented by a single session id:
        # an evidence-set member when it covers one (§3 "hit if at least one
        # premise session is labeled evidence"), else any resolved session.
        sids = sessions_by_id.get(p.id, set())
        representative = next(iter(sorted(sids & evidence)), None) or next(iter(sorted(sids)), None)
        retrieved_session_ids.append(representative)
        pub_at_by_id[p.id] = pub_at
    if particles and all(sid is None for sid in retrieved_session_ids):
        notes.append(
            f"Question {question.question_id}: no retrieved particle resolved to a "
            f"session id through its provenance chain"
        )

    # Recall credits every covered evidence session, so a derived particle
    # standing in for three sessions hits all three.
    hit_sessions = {sid for sids in sessions_by_id.values() for sid in sids} & evidence
    # An empty evidence set (the abstention variants) makes retrieval
    # unscoreable: recall_at_k / precision_at_k return None and the row is
    # marked, so the run loop can exclude it from the aggregates.
    retrieval = RetrievalQuestionResult(
        question_id=question.question_id,
        question_type=question.question_type,
        evidence_sessions=len(evidence),
        evidence_sessions_hit=len(hit_sessions),
        particles_retrieved=len(particles),
        recall_at_k=recall_at_k(hit_sessions, evidence),
        precision_at_k=precision_at_k(retrieved_session_ids, evidence),
        abstention=not evidence,
    )
    # Subject names are resolved only when the rendering asks for them: a
    # store-wide read per question is wasted work under the default, and the
    # scratch stores are small enough that one pass is cheaper than a
    # per-particle lookup when it is asked for.
    subject_names: dict[str, str] | None = None
    if subject_rendering == "names":
        from particles.store.subject_store import list_all_subjects

        subject_names = {s.id: s.canonical_name for s in await list_all_subjects(session)}
    context_lines = _particles_context_lines(
        particles, pub_at_by_id, context_budget, subject_names, subject_rendering
    )
    return _QuestionResult(
        retrieval=retrieval,
        particles_context=(
            "\n".join(line for _, line in context_lines)
            if particles
            else "(no memory claims were retrieved)"
        ),
        context_particle_ids=[pid for pid, _ in context_lines],
        notes=notes,
    )
