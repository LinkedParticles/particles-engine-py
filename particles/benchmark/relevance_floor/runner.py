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

from particles.benchmark.ceiling import (
    INTERRUPTED_DETAIL,
    UNSTARTED_DETAIL,
    CeilingReached,
    InFlight,
    QuestionGuard,
    Role,
    SpendCeiling,
    StoppedAtCeiling,
    stopped_note,
)
from particles.benchmark.memory.metrics import parse_judge_verdict
from particles.benchmark.memory.runner import _scored_call
from particles.benchmark.rot.runner import _chars_per_token, _price

# Shared with the sibling harnesses on purpose: one retry/classification rule
# for a scored call, and one price table, so the harnesses cannot disagree
# about what counts as a transient failure or what a model costs.
from particles.benchmark.spend import RunSpend
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
from particles.llm.usage import track_usage
from particles.operations.query.main import query, retrieve_ranked
from particles.store.particle_store import count_particles_by_status

from .harvest import heldout_fingerprint
from .metrics import (
    answerable_rate,
    floor_sweep,
    refusal_curve,
    responder_refusals,
    similarity_quantiles,
)
from .schema import (
    EXCLUSION_ANSWER,
    EXCLUSION_BUDGET,
    EXCLUSION_CEILING,
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
_ANSWER_RUBRIC_V1 = (
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
)
#: The grounded-and-useful judge of a composed answer, per protocol. Protocol 2
#: keeps protocol 1's text; it differs only in judging refusals (below).
_JUDGE_SYSTEMS: dict[int, str] = {1: _ANSWER_RUBRIC_V1, 2: _ANSWER_RUBRIC_V1}
#: The judge of a responder refusal, per protocol. Protocol 1 has
#: none: a question the responder refused is labelled unanswerable unjudged.
#: From protocol 2 the judge reads what the refusal withheld and says whether
#: an answer was there, so a false refusal is counted instead of assumed away.
_REFUSAL_JUDGE_SYSTEMS: dict[int, str] = {
    2: (
        "A question-answering system over a private knowledge base declined to "
        "answer a question, saying that nothing it retrieved bears on it. The user "
        "message carries, inside data fences, the question and the knowledge "
        "particles the system retrieved.\n\n"
        "Reply yes only if the retrieved particles contain information that "
        "answers the question that was asked, in whole or in substantial part, so "
        "that a grounded answer could have been written from them alone.\n\n"
        "Reply no if the particles are about related matters but do not answer "
        "this question; if answering would need facts the particles do not "
        "supply; or if the question depends on context it does not supply (an "
        "unresolved 'this', 'that', or 'it') so that no answer could be known to "
        "resolve it.\n\n"
        "Reply with a single word: yes or no."
    ),
}


def judges_refusals(protocol: int) -> bool:
    """Whether ``protocol`` judges a responder refusal instead of assuming it right."""
    return protocol in _REFUSAL_JUDGE_SYSTEMS


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


def refusal_judge_prompt(
    question: str, particle_lines: Sequence[str], *, protocol: int
) -> tuple[str, str]:
    """``(system, user)`` for judging one responder refusal under ``protocol``.

    The judge sees what the refusal withheld (the question and the retrieved
    particles, fenced as untrusted text) and never the refusal itself.
    """
    if protocol not in _REFUSAL_JUDGE_SYSTEMS:
        raise RelevanceFloorError(f"judge_protocol {protocol} does not judge refusals")
    nonce = make_nonce()
    system = _REFUSAL_JUDGE_SYSTEMS[protocol] + "\n\n" + data_fence_instruction(nonce)
    user = (
        f"Question:\n{fence(question, nonce, label='question')}\n\n"
        f"Retrieved particles:\n{fence(chr(10).join(particle_lines), nonce, label='particles')}"
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
        (
            "one judge call per question: a responder refusal is judged too "
            f"(judge protocol {cfg.judge_protocol})"
            if judges_refusals(cfg.judge_protocol)
            else "judge calls are an upper bound: a question the responder itself "
            "refuses is labelled without one"
        ),
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


def stop_calls(
    est: JudgedStageEstimate, roles: Sequence[Role]
) -> tuple[list[InFlight], list[InFlight]]:
    """``(in flight at the stop, every projected call)`` for the stop's terms.

    A question makes one call at a time, so at the stop at most
    ``concurrency`` calls are in flight; each is bounded as the dearer of an
    answer (with the query op's retry on an empty reply, which the stop cannot
    reach) and a judge call.
    """
    q = get_config().query
    by_name = {r.name: r for r in roles}
    answer_in = est.answer_input_tokens // max(est.answer_calls, 1)
    judge_in = est.judge_input_tokens // max(est.judge_calls, 1)
    retry = q.answer_retry_max_tokens if q.answer_retry_max_tokens > q.answer_max_tokens else 0
    flying = min(get_config().benchmark_relevance_floor.concurrency, est.questions)
    answer = InFlight(by_name["answer"], flying, answer_in, q.answer_max_tokens + retry)
    judge = InFlight(by_name["judge"], flying, judge_in, _JUDGE_MAX_TOKENS)

    def per_call(c: InFlight) -> float:
        if c.role.price is None:
            return 0.0
        return (c.input_tokens * c.role.price[0] + c.max_output_tokens * c.role.price[1]) / 1e6

    if answer.role.price is None and judge.role.price is None:
        in_flight = [answer]
    elif answer.role.price is None or judge.role.price is None:
        # One priced role bounds the dollars; the unpriced one the calls.
        in_flight = [answer, judge]
    else:
        in_flight = [answer if per_call(answer) >= per_call(judge) else judge]
    projected = [
        InFlight(by_name["answer"], est.answer_calls, answer_in, q.answer_max_tokens),
        InFlight(by_name["judge"], est.judge_calls, judge_in, _JUDGE_MAX_TOKENS),
    ]
    return in_flight, projected


def render_estimate(est: JudgedStageEstimate) -> str:
    """The pre-run projection, as the CLI prints it before any LLM call."""
    dollars = (
        f"~US${est.cost_usd:,.2f}"
        if est.cost_usd is not None
        else "no price configured (llm.price_per_mtok)"
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
    held: HeldOutQuestion,
    *,
    store: StoreHandle,
    top_k: int,
    before_call: Callable[[], None] | None = None,
) -> QueryResponse:
    cfg = get_config().benchmark_relevance_floor
    attempt = 0
    while True:
        if before_call is not None:
            before_call()
        async with session_scope(store) as session:
            # the recorded context keeps meaning what it meant on
            # every earlier run, so a marker refusal still lists its hits.
            response = await query(
                session,
                QueryRequest(question=held.question, top_k=top_k, include_refused_candidates=True),
            )
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


def _sentence_count(answer: str) -> int:
    # Deferred: the leakage package imports this one, so a module-top import
    # would be a cycle (AGENTS.md § Deferred imports, case 1).
    from particles.benchmark.leakage.metrics import split_sentences

    return len(split_sentences(answer))


async def _judge_one(
    held: HeldOutQuestion,
    *,
    store: StoreHandle,
    top_k: int,
    protocol: int,
    before_call: Callable[[], None] | None = None,
) -> QuestionResult:
    """One question end to end: answer with the gate off, then label it.

    ``before_call`` runs before every call (the spend stop); a
    stop it raises before the judge call carries the answer's sentence count.
    """
    cfg = get_config().benchmark_relevance_floor
    record = _record_text()
    response = await _answer_with_gate_off(held, store=store, top_k=top_k, before_call=before_call)
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
    result.answer_sentences = _sentence_count(response.answer)
    result.responder_refused = response.answer_refused
    if response.answer_refused:
        # With the floor off, a refusal can only be the responder's own §4
        # marker: the product's second gate says nothing retrieved bears on
        # the question. Protocol 1 takes it at its word; from protocol 2 the
        # judge reads the withheld top-k.
        if not judges_refusals(protocol):
            result.answerable = False
            return result
        system, user = refusal_judge_prompt(
            held.question, _context_lines(response), protocol=protocol
        )
    else:
        system, user = judge_prompt(
            held.question, _context_lines(response), response.answer, protocol=protocol
        )
    try:
        call = await _scored_call(
            "benchmark",
            user,
            max_tokens=_JUDGE_MAX_TOKENS,
            retries=cfg.call_retries,
            backoff_seconds=cfg.call_retry_backoff_seconds,
            system=system,
            before_attempt=before_call,
        )
    except CeilingReached as stop:
        stop.sentences = result.answer_sentences
        raise
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
    restored_out: list[str] | None = None,
    ceiling: SpendCeiling | None = None,
) -> list[QuestionResult]:
    """The paid stage over ``questions``; restores any checkpointed outcome first.

    ``restored_out``, when given, is extended with the ids of the questions
    restored from the checkpoint: they make no call, and the spend comparison
    says so.

    ``ceiling`` is the run's spend stop, checked before every
    call. Once it is reached no call starts; a question with a call left
    unstarted is excluded as ``budget_ceiling`` and never checkpointed, and
    ``ceiling.report`` describes the stop when the run returns.
    """
    _require_encoder()
    protocol = selection.judge_protocol or get_config().benchmark_relevance_floor.judge_protocol
    key = _checkpoint_key(selection)
    done = _load_checkpoint(checkpoint_path, key) if checkpoint_path is not None else {}
    if done and progress is not None:
        progress(f"restored {len(done)} checkpointed outcome(s)")
    if restored_out is not None:
        restored_out.extend(q.question_id for q in questions if q.question_id in done)
    gate = asyncio.Semaphore(get_config().benchmark_relevance_floor.concurrency)
    finished = 0

    async def _one(held: HeldOutQuestion) -> QuestionResult:
        nonlocal finished
        if held.question_id in done:
            return done[held.question_id]
        async with gate:
            guard = QuestionGuard(ceiling)
            try:
                # Per question, so an interrupted one's discarded spend is known.
                with track_usage() as question_usage:
                    result = await _judge_one(
                        held,
                        store=store,
                        top_k=selection.top_k,
                        protocol=protocol,
                        before_call=guard.before_call,
                    )
            except CeilingReached as stop:
                assert ceiling is not None  # only a stop raises it
                ceiling.record_stopped_question(
                    held.source.value,
                    started=guard.started,
                    usage=question_usage.snapshot(),
                    sentences=stop.sentences,
                )
                result = QuestionResult(
                    question_id=held.question_id,
                    source=held.source,
                    historical_refused=held.historical_refused,
                    question_chars=len(held.question),
                    excluded=EXCLUSION_CEILING,
                    excluded_detail=INTERRUPTED_DETAIL if guard.started else UNSTARTED_DETAIL,
                )
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
        results = list(await asyncio.gather(*(_one(held) for held in questions)))
    if ceiling is not None:
        complete_rows = [r for r in results if r.excluded != EXCLUSION_CEILING]
        ceiling.settle(
            finished_sources=[r.source.value for r in complete_rows],
            finished_sentences=[
                r.answer_sentences for r in complete_rows if r.answer_sentences is not None
            ],
        )
    return results


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
    selection: RunSelection,
    results: Sequence[QuestionResult],
    *,
    spend: RunSpend | None = None,
    stopped: StoppedAtCeiling | None = None,
) -> RelevanceFloorReport:
    """Assemble the report of record from recorded rows — pure, so re-sweepable.

    ``spend`` is the judged stage's projection beside its measured usage.
    """
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
    refusals_judged = selection.judge_protocol is not None and judges_refusals(
        selection.judge_protocol
    )
    if selection.judged:
        notes.append(
            "`answerable` is a label from a reference-free LLM judge (grounded in the "
            "retrieved particles AND useful for the question), not ground truth. "
            + (
                "A question the responder itself refused with the gate off is judged "
                "on whether the top-k it withheld answers the question, a different "
                "reading from the one an answer gets."
                if refusals_judged
                else "A question the responder itself refused with the gate off is "
                "labelled unanswerable without a judge call."
            )
        )
    if excluded:
        detail = ", ".join(f"{kind}: {count}" for kind, count in sorted(excluded.items()))
        notes.append(f"Excluded from every denominator — {detail}.")
    if stopped is not None:
        finished = sum(1 for r in rows if r.excluded != EXCLUSION_CEILING)
        notes.insert(0, stopped_note(stopped, finished=finished))
    return RelevanceFloorReport(
        selection=selection,
        results=rows,
        similarity_quantiles=similarity_quantiles(
            [r.max_similarity for r in rows if r.max_similarity is not None]
        ),
        refusal_curve=refusal_curve(rows, selection.floors),
        sweep=floor_sweep(rows, selection.floors) if selection.judged else [],
        answerable=answerable_rate(rows),
        responder=responder_refusals(rows) if refusals_judged else None,
        excluded=excluded,
        quality_notes=notes,
        spend=spend,
        stopped_at_ceiling=stopped,
    )
