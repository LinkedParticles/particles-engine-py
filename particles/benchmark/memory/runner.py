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
   session content hash — candidate production is Client-layer store-free
   , so repeated haystack sessions cost one LLM pass across the
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
deleted unless the caller keeps them.
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
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

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
    MemoryBenchmarkReport,
    MemoryQuestion,
    MemorySession,
    QaConditionMetrics,
    QaQuestionResult,
    RetrievalQuestionResult,
    RetrievalStageMetrics,
    RunSelection,
)
from particles.config import get_config
from particles.core.schema import Mutability, Particle, QueryRequest, Snapshot, SourceType
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
    complete,
    complete_many,
    get_provider,
)
from particles.operations.abstraction import is_derived, premise_ids_of, run_abstraction_pass
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

_ANSWER_SYSTEM = (
    "You are a helpful assistant answering a question about a user's prior "
    "chat history with an assistant. Answer concisely from the provided "
    "context. If the context does not contain the information needed, say "
    "that you don't have that information — do not guess."
)


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


def _remove_scratch_files(db_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Estimate / confirm gate (mirrored)
# ---------------------------------------------------------------------------


class MemoryRunEstimate(BaseModel):
    """Projected LLM cost of a run, computed before any call is made.

    Extraction calls are projected from *unique* session byte counts (the
    candidate cache dedupes repeats) via the same chunk math as the audit's
    estimate; QA + judge calls are three each per question. ``estimated_tokens``
    is a magnitude signal (~4 chars/token) dominated by the full-context
    condition, which re-sends each question's whole haystack.
    """

    questions: int = 0
    unique_sessions: int = 0
    total_session_chars: int = 0
    estimated_extraction_calls: int = 0
    estimated_answer_calls: int = 0
    estimated_judge_calls: int = 0
    estimated_llm_calls: int = 0
    estimated_tokens: int = 0


def estimate_run(
    questions: list[MemoryQuestion],
    *,
    qa: bool = True,
    memory: str = "particles",
    baselines: bool = True,
) -> MemoryRunEstimate:
    """Project LLM call counts + token volume from session byte counts.

    ``memory`` selects the memory under test (see
    :data:`~particles.benchmark.memory.comparators.MEMORY_KINDS`): the
    ``chunks`` comparator makes no write-time LLM call at all, ``notes`` makes
    exactly one per unique session. ``baselines=False`` (a comparator run
    reusing the particles run's baseline columns) drops the full-context and
    no-memory answer/judge calls from the projection.
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
    if memory == "notes":
        extraction_calls = sum(1 for n in unique_chars.values() if n > 0)
    elif memory == "particles":
        for n in unique_chars.values():
            if n <= 0:
                continue
            if n <= cfg.html_chunk_size:
                extraction_calls += 1
            else:
                extraction_calls += min(-(-n // cfg.html_chunk_size), cfg.max_llm_calls_per_source)

    conditions = 3 if baselines else 1
    answer_calls = conditions * len(questions) if qa else 0
    judge_calls = conditions * len(questions) if qa else 0
    # Token magnitude: unique-session write-time input + the full-context
    # condition re-reading every question's haystack once.
    write_chars = 0 if memory == "chunks" else sum(unique_chars.values())
    token_chars = write_chars + (haystack_chars_total if qa and baselines else 0)
    return MemoryRunEstimate(
        questions=len(questions),
        unique_sessions=len(unique_chars),
        total_session_chars=sum(unique_chars.values()),
        estimated_extraction_calls=extraction_calls,
        estimated_answer_calls=answer_calls,
        estimated_judge_calls=judge_calls,
        estimated_llm_calls=extraction_calls + answer_calls + judge_calls,
        estimated_tokens=token_chars // 4,
    )


def render_estimate(estimate: MemoryRunEstimate) -> str:
    """Human rendering of the estimate — always printed before any LLM call."""
    return (
        f"Estimate: {estimate.questions} question(s), {estimate.unique_sessions} unique "
        f"haystack session(s) → ~{estimate.estimated_extraction_calls} write-time "
        f"(extraction / notes) call(s) + {estimate.estimated_answer_calls} answer call(s) + "
        f"{estimate.estimated_judge_calls} judge call(s) = "
        f"~{estimate.estimated_llm_calls} LLM call(s), "
        f"~{estimate.estimated_tokens:,} tokens (the full-context baseline re-reads "
        f"each question's whole haystack). Repeated sessions are candidate-cached."
    )


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


def _particles_context(
    particles: list[Particle],
    pub_at_by_id: dict[str, datetime | None],
    budget_tokens: int | None = None,
) -> str:
    """Condition ii's context: top-k particle claims + subjects + dates.

    ``budget_tokens`` is the QA-at-budget clamp: particles are
    appended in rank order until the next line would exceed the budget
    (estimated at ~4 chars/token, the same heuristic as
    :func:`estimate_run`). ``None`` (default) keeps the full top-k context.
    The clamp applies to this condition only — the full-context baseline is
    deliberately unclamped (it is the ceiling, not the product).
    """
    if not particles:
        return "(no memory claims were retrieved)"
    lines: list[str] = []
    used_chars = 0
    budget_chars = None if budget_tokens is None else budget_tokens * 4
    for p in particles:
        pub_at = pub_at_by_id.get(p.id)
        date = pub_at.date().isoformat() if pub_at else "undated"
        subjects = ", ".join(p.subject_ids) if p.subject_ids else "-"
        line = f"- [{date}] {p.content} (subjects: {subjects})"
        if budget_chars is not None and lines and used_chars + len(line) > budget_chars:
            break
        lines.append(line)
        used_chars += len(line) + 1
    return "\n".join(lines)


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


def judge_prompt(question: MemoryQuestion, model_answer: str) -> str:
    """Per-question-type autoeval-style judge prompt (step 5).

    Ported from the LongMemEval evaluation protocol to our provider; the
    judge deviation (Anthropic, not the paper's OpenAI judge) is disclosed
    on the published page — numbers are comparable within our table, not
    across leaderboards. Abstention-variant questions score per the dataset
    protocol: correct iff the model declined to answer.
    """
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


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


class _QaAccumulator:
    """Per-condition accounting: verdicts + drill-down rows."""

    def __init__(self, condition: str) -> None:
        self.condition = condition
        self.pairs: list[tuple[str, bool]] = []
        self.rows: list[QaQuestionResult] = []

    def record(self, question: MemoryQuestion, correct: bool) -> None:
        self.pairs.append((question.question_type, correct))
        self.rows.append(
            QaQuestionResult(
                question_id=question.question_id,
                question_type=question.question_type,
                correct=correct,
                abstention=question.is_abstention,
            )
        )

    def to_metrics(self, model_id: str) -> QaConditionMetrics:
        return QaConditionMetrics(
            condition=self.condition,
            model_id=model_id,
            questions=len(self.pairs),
            accuracy=qa_accuracy([c for _, c in self.pairs]),
            accuracy_by_type=accuracy_by_type(self.pairs),
            per_question=self.rows,
        )


def _resolve_answer_model() -> str:
    """The resolved ``llm.benchmark_answer`` "<provider>:<model>" id, right now."""
    return get_provider("benchmark_answer").provider_model


def _resolve_extraction_model() -> str:
    """The resolved ``llm.extraction`` "<provider>:<model>" id, right now."""
    return get_provider("extraction").provider_model


def _thresholds_snapshot() -> dict[str, float]:
    """The pipeline thresholds in effect — part of the recorded run tuple (§5)."""
    cfg = get_config()
    return {
        "extraction.similarity_threshold": cfg.extraction.similarity_threshold,
        "confidence.uncalibrated_cap.enabled": float(cfg.confidence.uncalibrated_cap.enabled),
        "confidence.uncalibrated_cap.cap_value": cfg.confidence.uncalibrated_cap.cap_value,
    }


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
    questions in one condition are submitted as a single ``complete_many`` job
    , and the judge calls over those answers as a second — six
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
    total = questions_total if questions_total is not None else len(questions)

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
                            abstraction=abstraction,
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
                    try:
                        answer_text = await complete(
                            "benchmark_answer",
                            _answer_prompt(question, context_block),
                            max_tokens=_ANSWER_MAX_TOKENS,
                            system=_ANSWER_SYSTEM,
                            temperature=0.0,
                        )
                    except SameModelViolation:
                        raise
                    except Exception as exc:  # noqa: BLE001 — degrade per condition, keep the run
                        outcome.notes.append(
                            f"Question {question.question_id} [{condition}]: answer call "
                            f"failed ({exc!r}); scored incorrect"
                        )
                        outcome.qa_marks[condition] = False
                        continue

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
                    try:
                        verdict_text = await complete(
                            "benchmark",
                            judge_prompt(question, answer_text),
                            max_tokens=_JUDGE_MAX_TOKENS,
                            temperature=0.0,
                        )
                        correct = parse_judge_verdict(verdict_text)
                    except Exception as exc:  # noqa: BLE001
                        outcome.notes.append(
                            f"Question {question.question_id} [{condition}]: judge call "
                            f"failed ({exc!r}); scored incorrect"
                        )
                        correct = False
                    outcome.qa_marks[condition] = correct

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
        mid-run config flip. A ``None`` from ``complete_many`` — a per-request
        failure or an expired batch — scores that condition incorrect, exactly
        as the inline path degrades an errored call. QA-complete questions are
        checkpointed here (deferred from :func:`_process`, which returns before
        its own checkpoint under ``batch_qa``).
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
        answers_by_condition: dict[str, list[str | None]] = {}
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
                    system=_ANSWER_SYSTEM,
                )
                for slot in live
            ]
            if progress is not None:
                progress(
                    f"submitting answer batch [{condition}]: {len(requests)} request(s) "
                    f"via Message Batches (polling — the run is not hung)"
                )
            answers = await complete_many(
                "benchmark_answer",
                requests,
                max_tokens=_ANSWER_MAX_TOKENS,
                temperature=0.0,
                latency_tolerant=True,
            )
            answers_by_condition[condition] = answers
            if progress is not None:
                progress(
                    f"answer batch [{condition}]: "
                    f"{sum(a is not None for a in answers)}/{len(requests)} answered"
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
            for slot, answer_text in zip(live, answers_by_condition[condition], strict=True):
                outcome = outcomes[slot]
                assert outcome is not None  # ``live`` filtered on this
                if answer_text is None:
                    outcome.notes.append(
                        f"Question {questions[slot].question_id} [{condition}]: answer call "
                        f"unavailable (batch); scored incorrect"
                    )
                    outcome.qa_marks[condition] = False
                    continue
                judge_requests.append(
                    CompletionRequest(prompt=judge_prompt(questions[slot], answer_text))
                )
                judged_slots.append(slot)
            if not judge_requests:
                continue
            if progress is not None:
                progress(
                    f"submitting judge batch [{condition}]: {len(judge_requests)} request(s) "
                    f"via Message Batches (polling — the run is not hung)"
                )
            verdicts = await complete_many(
                "benchmark",
                judge_requests,
                max_tokens=_JUDGE_MAX_TOKENS,
                temperature=0.0,
                latency_tolerant=True,
            )
            for slot, verdict_text in zip(judged_slots, verdicts, strict=True):
                outcome = outcomes[slot]
                assert outcome is not None
                if verdict_text is None:
                    outcome.notes.append(
                        f"Question {questions[slot].question_id} [{condition}]: judge call "
                        f"unavailable (batch); scored incorrect"
                    )
                    outcome.qa_marks[condition] = False
                else:
                    outcome.qa_marks[condition] = parse_judge_verdict(verdict_text)

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

    try:
        try:
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
            for condition, correct in outcome.qa_marks.items():
                qa_acc[condition].record(question, correct)

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
        if not qa or not acc.pairs or answer_model_id is None:
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
    notes: list[str] = Field(default_factory=list)
    failed_note: str | None = None


class _QuestionResult(BaseModel):
    """Internal per-question pipeline output handed back to the run loop."""

    retrieval: RetrievalQuestionResult
    particles_context: str
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
    however good the abstraction (Precision@10 1.000 → 0.986 in the oracle A/B of 2026-07-18).

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
    abstraction: bool = False,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> _QuestionResult:
    """Deposit → extract → retrieve → provenance-score one question (§3).

    ``session_factory`` (when given) selects pooled extraction: every deposit
    is extracted concurrently in its own scratch-store session under one
    ``CompletionPool`` (see :func:`run_memory_benchmark` ``pooled``). ``None``
    keeps the serial full-price path on the caller's session.
    """
    notes: list[str] = []

    # 1. Deposit each haystack session as a CONVERSATION corpus entry.
    deposits: list[tuple[str, str]] = []
    for mem_session in question.sessions:
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
    return _QuestionResult(
        retrieval=retrieval,
        particles_context=_particles_context(particles, pub_at_by_id, context_budget),
        notes=notes,
    )
