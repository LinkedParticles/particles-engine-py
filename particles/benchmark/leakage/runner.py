# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Answer questions through the query op and judge every sentence.

Per question: the real ``query`` op runs exactly as configured (the relevance
floor included), the answer is split into sentences by the deterministic
splitter, and each sentence is put to the shared entailment judge
against the particles the answer was composed from — the rendered top-k, with
any narrative hit's constituents, which is precisely what the composer saw.
One answer call and one judge call per sentence, behind the estimate/confirm
gate the CLI owns.

The judge runs on the ``benchmark`` purpose and the composer on
``query_response``, so the two can be different models; by default a run
whose two purposes resolve to the same model is refused
(``benchmark_leakage.require_distinct_judge``), because a model judging its
own output shares the parametric background whose leakage is being measured.

A grounded run (``RunSelection.grounded``) asks the op for a
grounded answer, judges each sentence of a cited unit against the particles
that unit cites, and each uncited one against everything the composer saw, and
reports the composer's ``inference`` / ``background`` labels as their own
buckets beside the silent unsupported count.

Report-only, and read-only on the store.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

# Shared with the sibling harnesses on purpose: one retry/classification rule
# for a scored call, and one price table.
from particles.benchmark.memory.runner import _scored_call
from particles.benchmark.relevance_floor.harvest import heldout_fingerprint
from particles.benchmark.relevance_floor.schema import HeldOutQuestion
from particles.benchmark.rot.runner import _chars_per_token, _price
from particles.config import get_config
from particles.core.schema import (
    AnswerFailureCause,
    AttributionKind,
    AudienceHint,
    Particle,
    QueryRequest,
    QueryResponse,
)
from particles.db import DEFAULT_STORE, StoreHandle, session_scope
from particles.embeddings import get_embedding_model_id
from particles.llm import get_provider
from particles.operations.entailment import (
    EntailmentRubric,
    EntailmentVerdict,
    entailment_prompt,
    parse_entailment,
)
from particles.operations.query.grounding import composed_particles
from particles.operations.query.main import query
from particles.store.particle_store import count_particles_by_status

from .metrics import aggregate, split_sentences
from .schema import (
    EXCLUSION_ANSWER,
    EXCLUSION_BUDGET,
    EXCLUSION_EMPTY,
    EXCLUSION_INFRA,
    EXCLUSION_UNPARSEABLE,
    RETRYABLE_EXCLUSIONS,
    LeakageReport,
    QueryLeakage,
    RunSelection,
    SentenceResult,
    SentenceVerdict,
)

log = logging.getLogger(__name__)

_ANSWER_EXCLUSION_BY_CAUSE: dict[AnswerFailureCause, str] = {
    AnswerFailureCause.BUDGET: EXCLUSION_BUDGET,
    AnswerFailureCause.PROVIDER: EXCLUSION_INFRA,
}
_VERDICTS: dict[EntailmentVerdict, SentenceVerdict] = {
    EntailmentVerdict.ENTAILED: SentenceVerdict.SUPPORTED,
    EntailmentVerdict.NOT_ENTAILED: SentenceVerdict.UNSUPPORTED,
    EntailmentVerdict.NO_CLAIM: SentenceVerdict.NO_CLAIM,
}


class LeakageError(RuntimeError):
    """The run cannot measure anything — refused before any call is made."""


# ---------------------------------------------------------------------------
# The rubric — versioned; never edit a version's text in place
# ---------------------------------------------------------------------------

# Promotion's rubric fails a claim that drops a premise's date; this one does
# not, because an answer sentence that restates a dated claim without its date
# asserts nothing the knowledge base did not supply. What it does fail is the
# thing being measured: any content the particles did not supply, true or not.
#
# Protocol 1 asks for the verdict and then the reason, the order the
# abstraction judge's 1.169.10 fix found lets a judge commit before it reasons;
# it is kept, byte for byte, because the baseline was measured on it.
# Protocol 2 is the same rubric reason-first: it asks for the
# reasoning, then the claim flag, then the verdict, the order the shared
# schema already names. The parser reads the fields by name, so it reads both.
_RUBRIC_BODY = (
    "You are auditing one sentence of an answer that a question-answering "
    "system composed from a knowledge base. The user message carries, inside "
    "data fences, the question, possibly the answer text just before the "
    "sentence (context only), the SENTENCE under audit, and the knowledge-base "
    "claims the system was given to answer from.\n\n"
    "First decide whether the sentence asserts anything about the world. It "
    "does not when it only introduces or transitions ('Here is what I found:'), "
    "comments on the answer or on the knowledge base (what it contains or "
    "lacks, how confident it is), addresses the reader, or asks a question. "
    "Set asserts_claim to false for those.\n\n"
    "Otherwise set asserts_claim to true and decide whether every assertion "
    "in the sentence is supported by the knowledge-base claims taken together. "
    "Supported: a restatement, a paraphrase, a summary, a less specific "
    "version (dropping a date or a detail is fine), a combination of several "
    "claims, a hedge that matches a claim's own uncertainty, or an inference "
    "that follows directly from the claims. NOT supported: any fact, name, "
    "number, date, example, explanation, causal link, recommendation, or "
    "generalization the claims do not state or directly imply, including "
    "things that are true and well known. The question is whether the "
    "knowledge base supplied it, not whether it is correct. The question "
    "and the preceding answer text help you read the sentence (what 'it' "
    "refers to) and never count as support.\n\n"
)
_RUBRICS: dict[int, EntailmentRubric] = {
    2: EntailmentRubric(
        instructions=(
            _RUBRIC_BODY + "Write the reason first: say what the sentence asserts and which "
            "claims, if any, support each part. Then give the verdicts that "
            'reasoning reaches. Return a JSON object: {"reason": "...", '
            '"asserts_claim": true|false, "entailed": true|false}.'
        ),
        claim_heading="Sentence under audit",
        claim_label="sentence",
        premise_heading="Knowledge-base claim",
        premise_label="claim",
        allow_no_claim=True,
    ),
    1: EntailmentRubric(
        instructions=(
            "You are auditing one sentence of an answer that a question-answering "
            "system composed from a knowledge base. The user message carries, inside "
            "data fences, the question, possibly the answer text just before the "
            "sentence (context only), the SENTENCE under audit, and the knowledge-base "
            "claims the system was given to answer from.\n\n"
            "First decide whether the sentence asserts anything about the world. It "
            "does not when it only introduces or transitions ('Here is what I found:'), "
            "comments on the answer or on the knowledge base (what it contains or "
            "lacks, how confident it is), addresses the reader, or asks a question. "
            "Set asserts_claim to false for those.\n\n"
            "Otherwise set asserts_claim to true and decide whether every assertion "
            "in the sentence is supported by the knowledge-base claims taken together. "
            "Supported: a restatement, a paraphrase, a summary, a less specific "
            "version (dropping a date or a detail is fine), a combination of several "
            "claims, a hedge that matches a claim's own uncertainty, or an inference "
            "that follows directly from the claims. NOT supported: any fact, name, "
            "number, date, example, explanation, causal link, recommendation, or "
            "generalization the claims do not state or directly imply, including "
            "things that are true and well known. The question is whether the "
            "knowledge base supplied it, not whether it is correct. The question "
            "and the preceding answer text help you read the sentence (what 'it' "
            "refers to) and never count as support.\n\n"
            'Return a JSON object: {"asserts_claim": true|false, "entailed": '
            'true|false, "reason": "one sentence"}.'
        ),
        claim_heading="Sentence under audit",
        claim_label="sentence",
        premise_heading="Knowledge-base claim",
        premise_label="claim",
        allow_no_claim=True,
    ),
}


def rubric(protocol: int) -> EntailmentRubric:
    """The sentence-attribution rubric for ``protocol``."""
    if protocol not in _RUBRICS:
        known = ", ".join(str(v) for v in sorted(_RUBRICS))
        raise LeakageError(f"unknown judge_protocol {protocol}; known: {known}")
    return _RUBRICS[protocol]


def composed_premises(response: QueryResponse) -> list[str]:
    """The claims the composer answered from, in the order it saw them.

    Each rendered hit, and under a narrative hit its constituents —
    the response step expands those, so a sentence drawn from a constituent is
    drawn from the composer's own input.
    """
    premises: list[str] = []
    for particle in response.particles:
        premises.append(particle.content)
        premises.extend(c.content for c in response.narrative_constituents.get(particle.id, []))
    return premises


def judge_prompt(
    question: str,
    sentences: Sequence[str],
    index: int,
    premises: Sequence[str],
    *,
    protocol: int,
    context_sentences: int,
) -> tuple[str, str]:
    """``(system, user)`` for judging ``sentences[index]``."""
    preceding = " ".join(sentences[max(0, index - context_sentences) : index])
    context = [("Question", "question", question)]
    if preceding:
        context.append(("Preceding answer text (context only)", "preceding", preceding))
    return entailment_prompt(sentences[index], premises, rubric=rubric(protocol), context=context)


# ---------------------------------------------------------------------------
# Selection and the distinct-judge rule
# ---------------------------------------------------------------------------


def resolved_models() -> tuple[str, str]:
    """``(composer, judge)`` as each purpose resolves now (``provider:model``)."""
    return (
        get_provider("query_response").provider_model,
        get_provider("benchmark").provider_model,
    )


def check_distinct_judge() -> None:
    """Refuse a self-judging run unless the operator has turned the rule off."""
    composer, judge = resolved_models()
    if composer == judge and get_config().benchmark_leakage.require_distinct_judge:
        raise LeakageError(
            f"the judge (llm.benchmark) and the composer (llm.query_response) both "
            f"resolve to {composer}. A model judging its own answers shares the "
            f"background whose leakage is being measured; route llm.benchmark to a "
            f"different model, or set benchmark_leakage.require_distinct_judge: false "
            f"to measure the self-judged number on purpose"
        )


async def build_selection(
    questions: Sequence[HeldOutQuestion],
    *,
    store: StoreHandle = DEFAULT_STORE,
    top_k: int,
    ad_hoc: bool,
    sample_limit: int | None,
    grounded: bool | None = None,
) -> RunSelection:
    """The run tuple, resolved once before anything is measured.

    ``grounded`` ``None`` runs the op as configured (``query.grounded_answers``).
    """
    cfg = get_config()
    if grounded is None:
        grounded = cfg.query.grounded_answers
    async with session_scope(store) as session:
        active = (await count_particles_by_status(session)).get("ACTIVE", 0)
    by_source: dict[str, int] = {}
    for question in questions:
        by_source[question.source.value] = by_source.get(question.source.value, 0) + 1
    composer, judge = resolved_models()
    return RunSelection(
        store=str(store),
        store_active_particles=active,
        embedding_model_id=get_embedding_model_id(),
        top_k=top_k,
        audience=AudienceHint.GENERAL.value,
        configured_floor=cfg.query.relevance_floor,
        composer_model_id=composer,
        judge_model_id=judge,
        judge_protocol=cfg.benchmark_leakage.judge_protocol,
        distinct_judge=composer != judge,
        context_sentences=cfg.benchmark_leakage.context_sentences,
        questions=len(questions),
        by_source=by_source,
        fingerprint=heldout_fingerprint(questions),
        ad_hoc=ad_hoc,
        sample_limit=sample_limit,
        sample_seed=cfg.benchmark_leakage.sample_seed,
        grounded=grounded,
    )


# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------


class LeakageEstimate(BaseModel):
    """Projected calls, tokens and — when both models are priced — dollars."""

    questions: int = 0
    answer_calls: int = 0
    judge_calls: int = 0
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    composer_model: str = ""
    judge_model: str = ""
    cost_usd: float | None = None
    assumptions: list[str] = Field(default_factory=list)

    @property
    def llm_calls(self) -> int:
        """Total projected LLM calls."""
        return self.answer_calls + self.judge_calls


def estimate_run(questions: Sequence[HeldOutQuestion], *, top_k: int) -> LeakageEstimate:
    """Project a run from the configured assumptions, before anything is retrieved."""
    cfg = get_config().benchmark_leakage
    composer, judge = resolved_models()
    n = len(questions)
    sentences = round(n * cfg.estimate_sentences_per_answer)
    context_tokens = top_k * cfg.estimate_particle_tokens
    question_tokens = sum(
        int(len(q.question) / _chars_per_token("query_response")) for q in questions
    )
    est = LeakageEstimate(
        questions=n,
        answer_calls=n,
        judge_calls=sentences,
        composer_model=composer,
        judge_model=judge,
    )
    est.answer_input_tokens = (
        n * (context_tokens + cfg.estimate_answer_prompt_overhead_tokens) + question_tokens
    )
    est.answer_output_tokens = n * cfg.estimate_answer_output_tokens
    # Each judge call carries the premises, the question, a little preceding
    # text, and the sentence: the premises dominate.
    judge_context = int(context_tokens * _chars_per_token("query_response"))
    judge_context = int(judge_context / _chars_per_token("benchmark"))
    est.judge_input_tokens = sentences * (judge_context + cfg.estimate_judge_prompt_overhead_tokens)
    est.judge_output_tokens = sentences * cfg.estimate_judge_output_tokens
    est.assumptions = [
        f"top-k of {top_k} at ~{cfg.estimate_particle_tokens} tokens a particle (not "
        f"measured: nothing is retrieved before the projection)",
        f"~{cfg.estimate_sentences_per_answer:g} sentences an answer, answer output "
        f"{cfg.estimate_answer_output_tokens} tokens, judge output "
        f"{cfg.estimate_judge_output_tokens} tokens a call",
        "judge calls are an upper bound: a refused or empty answer is never judged",
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


def render_estimate(est: LeakageEstimate) -> str:
    """The pre-run projection, as the CLI prints it before any LLM call."""
    dollars = (
        f"~US${est.cost_usd:,.2f}"
        if est.cost_usd is not None
        else "no price configured (llm.price_per_mtok)"
    )
    lines = [
        f"Leakage run — projected {est.llm_calls} LLM calls over {est.questions} questions:",
        f"  answer  {est.answer_calls:>5} calls  {est.composer_model}  "
        f"~{est.answer_input_tokens:,} in / ~{est.answer_output_tokens:,} out tokens",
        f"  judge   {est.judge_calls:>5} calls  {est.judge_model}  "
        f"~{est.judge_input_tokens:,} in / ~{est.judge_output_tokens:,} out tokens",
        f"  cost    {dollars}",
    ]
    lines.extend(f"  · {note}" for note in est.assumptions)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _checkpoint_key(selection: RunSelection) -> str:
    """What a paid outcome is reusable under: the models, the rubric, and the read."""
    payload = {
        "store": selection.store,
        "embedding_model_id": selection.embedding_model_id,
        "top_k": selection.top_k,
        "audience": selection.audience,
        "configured_floor": selection.configured_floor,
        "composer_model_id": selection.composer_model_id,
        "judge_model_id": selection.judge_model_id,
        "judge_protocol": selection.judge_protocol,
        "context_sentences": selection.context_sentences,
    }
    if selection.grounded:
        # Added only when set, so an ungrounded run's key, and every
        # checkpoint recorded before grounded mode existed, is unchanged.
        payload["grounded"] = True
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _retryable(result: QueryLeakage) -> bool:
    return result.excluded in RETRYABLE_EXCLUSIONS or any(
        s.excluded in RETRYABLE_EXCLUSIONS for s in result.sentences
    )


def _load_checkpoint(path: Path, key: str) -> dict[str, QueryLeakage]:
    restored: dict[str, QueryLeakage] = {}
    if not path.exists():
        return restored
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            if row.get("key") == key:
                result = QueryLeakage.model_validate(row["result"])
                if not _retryable(result):
                    restored[result.question_id] = result
        except (ValueError, KeyError):
            log.warning("leakage: skipping a malformed checkpoint line in %s", path)
    return restored


def _append_checkpoint(path: Path, key: str, result: QueryLeakage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"key": key, "result": result.model_dump(mode="json")}) + "\n")


# ---------------------------------------------------------------------------
# One question end to end
# ---------------------------------------------------------------------------


def _record_text() -> bool:
    return get_config().benchmark.record_claim_text


async def _answer(
    held: HeldOutQuestion, *, store: StoreHandle, top_k: int, grounded: bool = False
) -> QueryResponse:
    """The real query op, retried on an answer failure like the sibling harnesses."""
    cfg = get_config().benchmark_leakage
    attempt = 0
    while True:
        async with session_scope(store) as session:
            response = await query(
                session,
                QueryRequest(
                    question=held.question,
                    top_k=top_k,
                    audience=AudienceHint.GENERAL,
                    grounded=grounded,
                ),
            )
        if response.answer_generation_error is None or attempt >= cfg.call_retries:
            return response
        attempt += 1
        log.info(
            "leakage answer call failed (%s); retry %d/%d",
            response.answer_generation_error,
            attempt,
            cfg.call_retries,
        )
        if cfg.call_retry_backoff_seconds:
            await asyncio.sleep(cfg.call_retry_backoff_seconds * attempt)


@dataclass(frozen=True)
class _Unit:
    """One sentence to judge, and what it is judged against."""

    premises: Sequence[str]
    attribution: AttributionKind | None = None
    cited_ids: tuple[str, ...] = ()
    invalid_citations: int = 0


def _premise_texts(particle: Particle, response: QueryResponse) -> list[str]:
    """A cited particle's content, and under a narrative its constituents' too."""
    return [particle.content] + [
        c.content for c in response.narrative_constituents.get(particle.id, [])
    ]


def grounded_units(response: QueryResponse) -> tuple[list[str], list[_Unit]]:
    """The sentences of a grounded answer, each with what it is judged against.

    Each attributed unit is split by the same deterministic splitter as an
    ungrounded answer, so both runs count sentences alike, and every sentence
    inherits its unit's attribution. A sentence of a ``CITED`` unit is judged
    against the particles it cites, specifically; every other sentence
    (``inference``, ``background``, ``unattributed``) against everything the
    composer saw, so a labelled sentence that the retrieved set does support
    is visible as such.
    """
    attribution = response.answer_attribution
    if attribution is None:
        raise ValueError("grounded_units needs a grounded response")
    everything = composed_premises(response)
    shown = composed_particles(response.particles, response.narrative_constituents)
    by_id = {p.id: p for p in shown}
    sentences: list[str] = []
    units: list[_Unit] = []
    for unit in attribution.sentences:
        if unit.kind is AttributionKind.CITED:
            premises = [t for pid in unit.cited_ids for t in _premise_texts(by_id[pid], response)]
        else:
            premises = everything
        for sentence in split_sentences(unit.text):
            sentences.append(sentence)
            units.append(
                _Unit(
                    premises=premises,
                    attribution=unit.kind,
                    cited_ids=tuple(unit.cited_ids),
                    invalid_citations=len(unit.invalid_citations),
                )
            )
    return sentences, units


async def _judge_sentence(
    question: str,
    sentences: Sequence[str],
    index: int,
    unit: _Unit,
    *,
    protocol: int,
) -> SentenceResult:
    cfg = get_config().benchmark_leakage
    record = _record_text()
    result = SentenceResult(
        index=index,
        text=sentences[index] if record else "",
        attribution=unit.attribution,
        cited_ids=list(unit.cited_ids),
        invalid_citations=unit.invalid_citations,
    )
    premises = unit.premises
    system, user = judge_prompt(
        question,
        sentences,
        index,
        premises,
        protocol=protocol,
        context_sentences=cfg.context_sentences,
    )
    call = await _scored_call(
        "benchmark",
        user,
        max_tokens=cfg.judge_max_tokens,
        retries=cfg.call_retries,
        backoff_seconds=cfg.call_retry_backoff_seconds,
        system=system,
    )
    if call.text is None:
        result.excluded = call.excluded
        result.excluded_detail = call.detail
        return result
    parsed = parse_entailment(call.text, rubric=rubric(protocol))
    if parsed is None:
        result.excluded = EXCLUSION_UNPARSEABLE
        result.excluded_detail = f"reply carried no usable verdict ({len(call.text)} chars)"
        return result
    verdict, reason = parsed
    result.verdict = _VERDICTS[verdict]
    result.reason = reason if record else ""
    return result


async def measure_question(
    held: HeldOutQuestion,
    *,
    store: StoreHandle,
    top_k: int,
    protocol: int,
    judge_gate: asyncio.Semaphore | None = None,
    grounded: bool = False,
) -> QueryLeakage:
    """One question: answer it through the op, then judge each sentence.

    ``grounded`` asks the op for a grounded answer and judges each sentence as
    :func:`grounded_units` describes.

    The sentences are judged concurrently under ``judge_gate``, which a run
    shares across every question so the in-flight judge calls stay bounded
    run-wide (``benchmark_leakage.judge_concurrency``). Results keep the
    answer's sentence order.
    """
    record = _record_text()
    response = await _answer(held, store=store, top_k=top_k, grounded=grounded)
    result = QueryLeakage(
        question_id=held.question_id,
        source=held.source,
        hit_count=len(response.particles),
        max_similarity=(
            response.relevance.max_similarity if response.relevance is not None else None
        ),
        question=held.question if record else "",
        context_particle_ids=[p.id for p in response.particles],
    )
    if not response.particles:
        result.excluded = EXCLUSION_EMPTY
        return result
    if response.answer_generation_error is not None:
        cause = response.answer_generation_error_cause
        result.excluded = (
            _ANSWER_EXCLUSION_BY_CAUSE.get(cause, EXCLUSION_ANSWER)
            if cause is not None
            else EXCLUSION_ANSWER
        )
        result.excluded_detail = response.answer_generation_error
        return result
    result.answer = response.answer if record else ""
    result.refused = response.answer_refused
    if response.answer_refused:
        return result
    if grounded and response.answer_attribution is not None:
        sentences, units = grounded_units(response)
    else:
        sentences = split_sentences(response.answer)
        everything = composed_premises(response)
        units = [_Unit(premises=everything) for _ in sentences]
    gate = judge_gate or asyncio.Semaphore(get_config().benchmark_leakage.judge_concurrency)

    async def _bounded(index: int) -> SentenceResult:
        async with gate:
            return await _judge_sentence(
                held.question, sentences, index, units[index], protocol=protocol
            )

    result.sentences = list(await asyncio.gather(*(_bounded(i) for i in range(len(sentences)))))
    return result


async def run_leakage(
    questions: Sequence[HeldOutQuestion],
    *,
    store: StoreHandle = DEFAULT_STORE,
    selection: RunSelection,
    checkpoint_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[QueryLeakage]:
    """Measure every question; restores any checkpointed outcome first."""
    check_distinct_judge()
    protocol = selection.judge_protocol
    rubric(protocol)  # refuse an unknown protocol before any call
    key = _checkpoint_key(selection)
    done = _load_checkpoint(checkpoint_path, key) if checkpoint_path is not None else {}
    if done and progress is not None:
        progress(f"restored {len(done)} checkpointed outcome(s)")
    gate = asyncio.Semaphore(get_config().benchmark_leakage.concurrency)
    judge_gate = asyncio.Semaphore(get_config().benchmark_leakage.judge_concurrency)
    finished = 0

    async def _one(held: HeldOutQuestion) -> QueryLeakage:
        nonlocal finished
        if held.question_id in done:
            return done[held.question_id]
        async with gate:
            try:
                result = await measure_question(
                    held,
                    store=store,
                    top_k=selection.top_k,
                    protocol=protocol,
                    judge_gate=judge_gate,
                    grounded=selection.grounded,
                )
            except Exception as exc:  # noqa: BLE001 — one bad question must not abort the run
                # A store or transport failure outside the scored calls (a
                # pool timeout, a dropped connection). Excluded as `infra`,
                # never checkpointed, so a re-run retries it.
                log.warning("leakage: question %s failed: %r", held.question_id, exc)
                result = QueryLeakage(
                    question_id=held.question_id,
                    source=held.source,
                    excluded=EXCLUSION_INFRA,
                    excluded_detail=repr(exc)[:300],
                )
        # A failed call is not checkpointed, so a re-run retries it.
        if checkpoint_path is not None and not _retryable(result):
            _append_checkpoint(checkpoint_path, key, result)
        finished += 1
        if progress is not None and (finished % 10 == 0 or finished == len(questions) - len(done)):
            progress(f"measured {finished}/{len(questions) - len(done)}")
        return result

    return list(await asyncio.gather(*(_one(held) for held in questions)))


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(selection: RunSelection, results: Sequence[QueryLeakage]) -> LeakageReport:
    """Assemble the report of record from recorded rows — pure, so re-aggregable."""
    notes = [
        "The judge reads only the particles the answer was composed from. A sentence "
        "it labels unsupported carries content the store did not supply for this "
        "answer, whether or not that content is true: this measures attribution, "
        "not accuracy.",
        "The rate's denominator is the claim-bearing sentences. Lead-ins, statements "
        "about what the knowledge base holds, and other sentences that assert nothing "
        "are counted apart (no_claim), as are sentences the judge could not score.",
        "A refused answer is never judged; refusals are reported as their own rate.",
    ]
    if selection.judge_protocol == 1:
        notes.append(
            "Judged under rubric protocol 1, which asks for the verdict before the "
            "reason. Protocol 2 asks for the reason first; compare a number only with "
            "one measured under the same protocol."
        )
    if selection.grounded:
        notes.append(
            "Grounded run: the composer cited particle ids per sentence and labelled "
            "its own contribution. A cited sentence is judged against the ids it cites, "
            "every other sentence against everything the composer saw. The headline "
            "counts only silent unsupported sentences (cited or unattributed); "
            "sentences labelled inference or background are reported beside it, over "
            "the same denominator, and never folded into it."
        )
    if not selection.distinct_judge:
        notes.append(
            "The judge and the composer are the same model, so the judge shares the "
            "background whose leakage is measured. Read this number as a lower bound."
        )
    if selection.by_source.get("user_prompt"):
        notes.append(
            "Questions harvested from typed prompts (`user_prompt`) are a proxy for "
            "memory queries, and many depend on conversational context. Read the "
            "per-source rows."
        )
    report = LeakageReport(selection=selection, results=list(results))
    report = aggregate(report)
    if report.excluded:
        detail = ", ".join(f"{kind}: {count}" for kind, count in report.excluded.items())
        notes.append(f"Questions excluded from every denominator: {detail}.")
    if report.sentences.judge_excluded:
        notes.append(
            f"{report.sentences.judge_excluded} sentence(s) the judge could not score are "
            "excluded from the rate's denominator, not counted unsupported."
        )
    report.quality_notes = notes
    return report
