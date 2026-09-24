# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Replay a held-out question set through the query op and measure its gate.

Two stages, separately priced:

* :func:`replay_retrieval` — every question through ``retrieve_ranked`` (the
  query op's selection half, the memory-rot harness's adapter). Records the
  maximum raw cosine over the rendered top-k, which is exactly what the
  relevance floor reads. No LLM call; this alone yields the
  refusal curve.
* :func:`run_judged` — every question through the real ``query`` op **with the
  gate disabled for the run**, so the response step runs over the top-k the
  floor would have suppressed; a reference-free judge then labels the answer
  grounded-and-useful or not. Two LLM calls per question at most, behind the
  estimate/confirm gate the CLI owns.

Report-only, and read-only on the store: nothing here writes a particle, a
corpus entry, or a config value that outlives the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

# Shared with the sibling harnesses on purpose: one retry/classification rule
# for a scored call, and one price table, so the harnesses cannot disagree
# about what counts as a transient failure or what a model costs.
from particles.benchmark.memory.metrics import parse_judge_verdict
from particles.benchmark.memory.runner import _scored_call
from particles.benchmark.rot.runner import _chars_per_token, _price
from particles.config import get_config
from particles.core.schema import (
    AnswerFailureCause,
    Particle,
    QueryRequest,
    QueryResponse,
)
from particles.db import DEFAULT_STORE, StoreHandle, session_scope
from particles.embeddings import get_embedding_model, get_embedding_model_id
from particles.llm import data_fence_instruction, fence, get_provider, make_nonce
from particles.operations.query.main import query, retrieve_ranked
from particles.store.particle_store import count_particles_by_status

from .harvest import heldout_fingerprint
from .metrics import (
    answerable_rate,
    floor_sweep,
    refusal_curve,
    similarity_quantiles,
)
from .schema import (
    EXCLUSION_ANSWER,
    EXCLUSION_BUDGET,
    EXCLUSION_EMPTY,
    EXCLUSION_INFRA,
    RETRYABLE_EXCLUSIONS,
    HeldOutQuestion,
    QuestionResult,
    RelevanceFloorReport,
    RunSelection,
)

log = logging.getLogger(__name__)

_JUDGE_MAX_TOKENS = 1024
#: The query op's own answer-failure causes, mapped onto this suite's exclusion
#: vocabulary. Deliberately the memory benchmark's two classes and no third: an
#: answer the op could not produce because its token budget ran out is the same
#: kind of unscoreable as a judge call that ran out, and must not be laundered
#: as infrastructure noise (nor the reverse).
_ANSWER_EXCLUSION_BY_CAUSE: dict[AnswerFailureCause, str] = {
    AnswerFailureCause.BUDGET: EXCLUSION_BUDGET,
    AnswerFailureCause.PROVIDER: EXCLUSION_INFRA,
}
# Characters one rendered particle line adds beyond its content (bullet,
# modality marker) in the response prompt — for the cost projection only.
_LINE_OVERHEAD_CHARS = 24


class RelevanceFloorError(RuntimeError):
    """The run cannot measure anything — refused before any call is made."""


# ---------------------------------------------------------------------------
# The judge — reference-free, versioned
# ---------------------------------------------------------------------------

# A real question has no gold answer, so this is not the reference
# judge's prompt: it is the same judge *purpose* (``llm.benchmark``) asked a
# reference-free question. Never edit a version's text in place — add one.
_JUDGE_SYSTEMS: dict[int, str] = {
    1: (
        "You are grading one answer produced by a question-answering system over a "
        "private knowledge base. The user message carries, inside data fences, the "
        "question, the knowledge particles the system retrieved, and the answer it "
        "gave.\n\n"
        "Reply yes only if BOTH hold:\n"
        "1. Grounded: every substantive claim in the answer is supported by the "
        "retrieved particles.\n"
        "2. Useful: the answer resolves the question that was asked, in whole or in "
        "substantial part.\n\n"
        "Reply no if the answer says or implies that the knowledge base lacks the "
        "information; if it answers a different question from the one asked; if the "
        "question depends on context it does not supply (an unresolved 'this', "
        "'that', or 'it') so that no answer could be known to resolve it; or if its "
        "substantive claims are not supported by the particles.\n\n"
        "Reply with a single word: yes or no."
    ),
}


def judge_prompt(
    question: str, particle_lines: Sequence[str], answer: str, *, protocol: int
) -> tuple[str, str]:
    """``(system, user)`` for one judge call under ``protocol``.

    The question, the particles, and the answer are all untrusted text, so
    they travel behind one per-call nonce fence, as the response step's do.
    """
    if protocol not in _JUDGE_SYSTEMS:
        known = ", ".join(str(v) for v in sorted(_JUDGE_SYSTEMS))
        raise RelevanceFloorError(f"unknown judge_protocol {protocol}; known: {known}")
    nonce = make_nonce()
    system = _JUDGE_SYSTEMS[protocol] + "\n\n" + data_fence_instruction(nonce)
    user = (
        f"Question:\n{fence(question, nonce, label='question')}\n\n"
        f"Retrieved particles:\n{fence(chr(10).join(particle_lines), nonce, label='particles')}\n\n"
        f"Answer:\n{fence(answer, nonce, label='answer')}"
    )
    return system, user


def _context_lines(response: QueryResponse) -> list[str]:
    """The retrieved claims as the judge sees them — narrative hits expanded."""
    lines: list[str] = []
    for particle in response.particles:
        lines.append(f"- {particle.content}")
        lines.extend(
            f"    - {c.content}" for c in response.narrative_constituents.get(particle.id, [])
        )
    return lines


# ---------------------------------------------------------------------------
# The gate override
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def floor_disabled() -> Iterator[float]:
    """Set ``query.relevance_floor`` to 0.0 for the block; yield the prior value.

    0.0 is the documented off switch, so the query op runs its
    own response step over every top-k — no private seam, no forked pipeline.
    The prior value is restored on exit, success or failure.
    """
    cfg = get_config().query
    prior = cfg.relevance_floor
    cfg.relevance_floor = 0.0
    try:
        yield prior
    finally:
        cfg.relevance_floor = prior


# ---------------------------------------------------------------------------
# Stage 1 — retrieval replay ($0)
# ---------------------------------------------------------------------------


def _record_text() -> bool:
    return get_config().benchmark.record_claim_text


def _max_similarity(sims: Sequence[float]) -> float | None:
    return max(sims) if sims else None


def _context_chars(particles: Sequence[Particle]) -> int:
    return sum(len(p.content) + _LINE_OVERHEAD_CHARS for p in particles)


async def replay_retrieval(
    questions: Sequence[HeldOutQuestion],
    *,
    store: StoreHandle = DEFAULT_STORE,
    top_k: int,
    progress: Callable[[str], None] | None = None,
) -> list[QuestionResult]:
    """Replay every question through the selection half. No LLM call is made."""
    _require_encoder()
    rows: list[QuestionResult] = []
    record = _record_text()
    async with session_scope(store) as session:
        for index, held in enumerate(questions, start=1):
            scored = await retrieve_ranked(
                session, QueryRequest(question=held.question, top_k=top_k)
            )
            particles: list[Particle] = [p for p, _, _ in scored]
            rows.append(
                QuestionResult(
                    question_id=held.question_id,
                    source=held.source,
                    max_similarity=_max_similarity([sim for _, sim, _ in scored]),
                    hit_count=len(scored),
                    historical_refused=held.historical_refused,
                    excluded="" if scored else EXCLUSION_EMPTY,
                    context_chars=_context_chars(particles),
                    question_chars=len(held.question),
                    question=held.question if record else "",
                    context_particle_ids=[p.id for p in particles],
                )
            )
            if progress is not None and (index % 25 == 0 or index == len(questions)):
                progress(f"replayed {index}/{len(questions)}")
    return rows


def reuse_replay(
    saved: RelevanceFloorReport,
    questions: Sequence[HeldOutQuestion],
    selection: RunSelection,
) -> list[QuestionResult]:
    """The saved replay's rows for ``questions``, stripped of any judged label.

    The replay is free but slow on a large store, so a saved one may stand in
    for it — only under the same depth and encoder (the two things a recorded
    cosine depends on besides the store), and only if it covers every question
    asked for; a sample of a saved full replay is fine. Labels are dropped so
    a reused judged report can never pass its old verdicts off as this run's.
    """
    for name in ("top_k", "embedding_model_id"):
        if getattr(saved.selection, name) != getattr(selection, name):
            raise RelevanceFloorError(
                f"the saved replay was recorded under a different {name} "
                f"({getattr(saved.selection, name)!r}, not {getattr(selection, name)!r}); "
                f"re-run the replay instead of reusing it"
            )
    by_id = {row.question_id: row for row in saved.results}
    missing = [q.question_id for q in questions if q.question_id not in by_id]
    if missing:
        raise RelevanceFloorError(
            f"the saved replay does not cover {len(missing)} of the "
            f"{len(questions)} question(s) asked for; re-run the replay"
        )
    return [
        by_id[q.question_id].model_copy(
            update={"answerable": None, "responder_refused": None, "answer": "", "verdict": ""}
        )
        for q in questions
    ]


def _require_encoder() -> None:
    if get_embedding_model() is None:
        raise RelevanceFloorError(
            "no embedding model is available, so there is no cosine to hold to a "
            "floor (the gate is inert without one)"
        )


# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------


class JudgedStageEstimate(BaseModel):
    """Projected calls, tokens and — when both models are priced — dollars."""

    questions: int = 0
    answer_calls: int = 0
    judge_calls: int = 0
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    answer_model: str = ""
    judge_model: str = ""
    cost_usd: float | None = None
    assumptions: list[str] = Field(default_factory=list)

    @property
    def llm_calls(self) -> int:
        """Total projected LLM calls."""
        return self.answer_calls + self.judge_calls


def estimate_judged_stage(rows: Sequence[QuestionResult]) -> JudgedStageEstimate:
    """Project the judged stage from the replay's *measured* prompt sizes.

    Input is not assumed: each question's real top-k was just retrieved, so the
    response prompt's size is known. Only the two output sizes are assumptions,
    and the judge count is an upper bound — a question the responder itself
    refuses is labelled without a judge call.
    """
    cfg = get_config().benchmark_relevance_floor
    scoreable = [row for row in rows if row.max_similarity is not None]
    est = JudgedStageEstimate(
        questions=len(scoreable),
        answer_calls=len(scoreable),
        judge_calls=len(scoreable),
        answer_model=get_provider("query_response").provider_model,
        judge_model=get_provider("benchmark").provider_model,
    )
    answer_cpt = _chars_per_token("query_response")
    judge_cpt = _chars_per_token("benchmark")
    for row in scoreable:
        prompt_chars = row.context_chars + row.question_chars
        est.answer_input_tokens += (
            int(prompt_chars / answer_cpt) + cfg.estimate_answer_prompt_overhead_tokens
        )
        est.judge_input_tokens += (
            int(prompt_chars / judge_cpt)
            + cfg.estimate_answer_output_tokens
            + cfg.estimate_judge_prompt_overhead_tokens
        )
    est.answer_output_tokens = len(scoreable) * cfg.estimate_answer_output_tokens
    est.judge_output_tokens = len(scoreable) * cfg.estimate_judge_output_tokens
    est.assumptions = [
        "input tokens are measured from each question's retrieved top-k, not assumed",
        f"answer output assumed {cfg.estimate_answer_output_tokens} tokens/call, "
        f"judge output {cfg.estimate_judge_output_tokens}",
        "judge calls are an upper bound: a question the responder itself refuses "
        "is labelled without one",
        "transient-failure retries are not projected",
    ]
    answer_price = _price("query_response")
    judge_price = _price("benchmark")
    if answer_price is not None and judge_price is not None:
        est.cost_usd = (
            est.answer_input_tokens * answer_price[1]
            + est.answer_output_tokens * answer_price[2]
            + est.judge_input_tokens * judge_price[1]
            + est.judge_output_tokens * judge_price[2]
        ) / 1_000_000
    return est


def render_estimate(est: JudgedStageEstimate) -> str:
    """The pre-run projection, as the CLI prints it before any LLM call."""
    dollars = (
        f"~US${est.cost_usd:,.2f}"
        if est.cost_usd is not None
        else "no price configured (benchmark_memory.price_per_mtok)"
    )
    lines = [
        f"Judged stage — projected {est.llm_calls} LLM calls over {est.questions} questions:",
        f"  answer  {est.answer_calls:>5} calls  {est.answer_model}  "
        f"~{est.answer_input_tokens:,} in / ~{est.answer_output_tokens:,} out tokens",
        f"  judge   {est.judge_calls:>5} calls  {est.judge_model}  "
        f"~{est.judge_input_tokens:,} in / ~{est.judge_output_tokens:,} out tokens",
        f"  cost    {dollars}",
    ]
    lines.extend(f"  · {note}" for note in est.assumptions)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 2 — answer with the gate off, then judge (paid)
# ---------------------------------------------------------------------------


def _checkpoint_key(selection: RunSelection) -> str:
    """What a paid outcome is reusable under — models, protocol, depth, encoder."""
    payload = {
        "embedding_model_id": selection.embedding_model_id,
        "top_k": selection.top_k,
        "answer_model_id": selection.answer_model_id,
        "judge_model_id": selection.judge_model_id,
        "judge_protocol": selection.judge_protocol,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _load_checkpoint(path: Path, key: str) -> dict[str, QuestionResult]:
    restored: dict[str, QuestionResult] = {}
    if not path.exists():
        return restored
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            if row.get("key") == key:
                result = QuestionResult.model_validate(row["result"])
                if result.excluded not in RETRYABLE_EXCLUSIONS:
                    restored[result.question_id] = result
        except (ValueError, KeyError):
            log.warning("relevance-floor: skipping a malformed checkpoint line in %s", path)
    return restored


def _append_checkpoint(path: Path, key: str, result: QuestionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"key": key, "result": result.model_dump(mode="json")}) + "\n")


async def _answer_with_gate_off(
    held: HeldOutQuestion, *, store: StoreHandle, top_k: int
) -> QueryResponse:
    cfg = get_config().benchmark_relevance_floor
    attempt = 0
    while True:
        async with session_scope(store) as session:
            response = await query(session, QueryRequest(question=held.question, top_k=top_k))
        if response.answer_generation_error is None or attempt >= cfg.call_retries:
            return response
        attempt += 1
        log.info(
            "relevance-floor answer call failed (%s); retry %d/%d",
            response.answer_generation_error,
            attempt,
            cfg.call_retries,
        )
        if cfg.call_retry_backoff_seconds:
            await asyncio.sleep(cfg.call_retry_backoff_seconds * attempt)


async def _judge_one(
    held: HeldOutQuestion, *, store: StoreHandle, top_k: int, protocol: int
) -> QuestionResult:
    """One question end to end: answer with the gate off, then label it."""
    cfg = get_config().benchmark_relevance_floor
    record = _record_text()
    response = await _answer_with_gate_off(held, store=store, top_k=top_k)
    result = QuestionResult(
        question_id=held.question_id,
        source=held.source,
        hit_count=len(response.particles),
        historical_refused=held.historical_refused,
        context_chars=_context_chars(response.particles),
        question_chars=len(held.question),
        question=held.question if record else "",
        context_particle_ids=[p.id for p in response.particles],
    )
    if response.relevance is None:
        # Empty result (the encoder's presence was checked before the run).
        result.excluded = EXCLUSION_EMPTY
        return result
    result.max_similarity = response.relevance.max_similarity
    if response.answer_generation_error is not None:
        # The op types its own failures, so an exhausted answer budget lands in
        # the same bucket as an exhausted judge budget and a provider failure in
        # the same bucket as a transport one — which is the point of the split:
        # an operator's cap is actionable, infra noise is not. An untyped
        # failure (an engine predating the field) still falls back to its own
        # exclusion rather than being guessed into one of them.
        cause = response.answer_generation_error_cause
        result.excluded = (
            _ANSWER_EXCLUSION_BY_CAUSE.get(cause, EXCLUSION_ANSWER)
            if cause is not None
            else EXCLUSION_ANSWER
        )
        result.excluded_detail = response.answer_generation_error
        return result
    result.answer = response.answer if record else ""
    result.responder_refused = response.answer_refused
    if response.answer_refused:
        # With the floor off, a refusal can only be the responder's own §4
        # marker: the product's second gate says nothing retrieved bears on
        # the question. Labelled without paying a judge to agree.
        result.answerable = False
        return result
    system, user = judge_prompt(
        held.question, _context_lines(response), response.answer, protocol=protocol
    )
    call = await _scored_call(
        "benchmark",
        user,
        max_tokens=_JUDGE_MAX_TOKENS,
        retries=cfg.call_retries,
        backoff_seconds=cfg.call_retry_backoff_seconds,
        system=system,
    )
    if call.text is None:
        result.excluded = call.excluded
        result.excluded_detail = call.detail
        return result
    result.verdict = call.text if record else ""
    result.answerable = parse_judge_verdict(call.text)
    return result


async def run_judged(
    questions: Sequence[HeldOutQuestion],
    *,
    store: StoreHandle = DEFAULT_STORE,
    selection: RunSelection,
    checkpoint_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[QuestionResult]:
    """The paid stage over ``questions``; restores any checkpointed outcome first."""
    _require_encoder()
    protocol = selection.judge_protocol or get_config().benchmark_relevance_floor.judge_protocol
    key = _checkpoint_key(selection)
    done = _load_checkpoint(checkpoint_path, key) if checkpoint_path is not None else {}
    if done and progress is not None:
        progress(f"restored {len(done)} checkpointed outcome(s)")
    gate = asyncio.Semaphore(get_config().benchmark_relevance_floor.concurrency)
    finished = 0

    async def _one(held: HeldOutQuestion) -> QuestionResult:
        nonlocal finished
        if held.question_id in done:
            return done[held.question_id]
        async with gate:
            result = await _judge_one(held, store=store, top_k=selection.top_k, protocol=protocol)
        # A failed call is not checkpointed: a re-run should retry it. That
        # includes `budget` — sampling is the model's default once a model
        # rejects `temperature`, so an exhausted budget is no longer certain
        # to repeat.
        if checkpoint_path is not None and result.excluded not in RETRYABLE_EXCLUSIONS:
            _append_checkpoint(checkpoint_path, key, result)
        finished += 1
        if progress is not None and finished % 10 == 0:
            progress(f"judged {finished}/{len(questions) - len(done)}")
        return result

    with floor_disabled():
        return list(await asyncio.gather(*(_one(held) for held in questions)))


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


async def build_selection(
    questions: Sequence[HeldOutQuestion],
    *,
    store: StoreHandle = DEFAULT_STORE,
    top_k: int,
    floors: Sequence[float],
    sample_limit: int | None,
    judged: bool,
) -> RunSelection:
    """The run tuple, resolved once before anything is measured."""
    cfg = get_config()
    async with session_scope(store) as session:
        active = (await count_particles_by_status(session)).get("ACTIVE", 0)
    by_source: dict[str, int] = {}
    for question in questions:
        by_source[question.source.value] = by_source.get(question.source.value, 0) + 1
    return RunSelection(
        embedding_model_id=get_embedding_model_id(),
        top_k=top_k,
        configured_floor=cfg.query.relevance_floor,
        floors=list(floors),
        store_active_particles=active,
        heldout_questions=len(questions),
        heldout_by_source=by_source,
        heldout_fingerprint=heldout_fingerprint(questions),
        sample_limit=sample_limit,
        sample_seed=cfg.benchmark_relevance_floor.sample_seed,
        judged=judged,
        answer_model_id=get_provider("query_response").provider_model if judged else None,
        judge_model_id=get_provider("benchmark").provider_model if judged else None,
        judge_protocol=cfg.benchmark_relevance_floor.judge_protocol if judged else None,
    )


def build_report(
    selection: RunSelection, results: Sequence[QuestionResult]
) -> RelevanceFloorReport:
    """Assemble the report of record from recorded rows — pure, so re-sweepable."""
    rows = list(results)
    excluded: dict[str, int] = {}
    for result in rows:
        if result.excluded:
            excluded[result.excluded] = excluded.get(result.excluded, 0) + 1
    notes = [
        "Questions harvested from typed prompts (`user_prompt`) are a proxy for "
        "memory queries: real information needs in the store's domain that nobody "
        "addressed to the store. Many depend on conversational context and are "
        "unanswerable standalone, which inflates the unanswerable population; the "
        "answerable-but-refused numerator is unaffected. Read the per-source rows.",
        "The replay is against the store as it is now, not as it was when a question was asked.",
    ]
    if selection.judged:
        notes.append(
            "`answerable` is a label from a reference-free LLM judge (grounded in the "
            "retrieved particles AND useful for the question), not ground truth. A "
            "question the responder itself refused with the gate off is labelled "
            "unanswerable without a judge call."
        )
    if excluded:
        detail = ", ".join(f"{kind}: {count}" for kind, count in sorted(excluded.items()))
        notes.append(f"Excluded from every denominator — {detail}.")
    return RelevanceFloorReport(
        selection=selection,
        results=rows,
        similarity_quantiles=similarity_quantiles(
            [r.max_similarity for r in rows if r.max_similarity is not None]
        ),
        refusal_curve=refusal_curve(rows, selection.floors),
        sweep=floor_sweep(rows, selection.floors) if selection.judged else [],
        answerable=answerable_rate(rows),
        excluded=excluded,
        quality_notes=notes,
    )
