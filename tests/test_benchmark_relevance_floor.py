# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the relevance-floor benchmark.

No API key, no network: the harvest and the sweep arithmetic are pure; the
runner tier drives the real query op over an in-memory store with a
deterministic bag-of-words encoder and scripted answer/judge providers.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.relevance_floor import (
    REFUSAL_PHRASE,
    HeldOutQuestion,
    QuestionResult,
    QuestionSource,
    RelevanceFloorError,
    RelevanceFloorReport,
    RunSelection,
    build_report,
    estimate_judged_stage,
    floor_disabled,
    floor_sweep,
    harvest_transcripts,
    heldout_fingerprint,
    judge_prompt,
    load_heldout,
    question_id,
    refusal_curve,
    render_report,
    replay_retrieval,
    reuse_replay,
    run_judged,
    sample_questions,
    similarity_quantiles,
    write_heldout,
)
from particles.benchmark.relevance_floor.harvest import prompt_questions, transcript_questions
from particles.benchmark.relevance_floor.metrics import refused_at
from particles.benchmark.relevance_floor.runner import _checkpoint_key
from particles.config import TokenPrice, get_config
from particles.core.schema import (
    CalibrationSource,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    RelevanceNote,
    Status,
    UncertaintyNature,
)
from particles.llm import CompletionError, override_providers
from particles.operations.query.respond import REFUSAL_MARKER

cli = CliRunner()
_ALL = frozenset(QuestionSource)
_BOUNDS = {"min_chars": 15, "max_chars": 300}


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------


def _line(record: dict[str, Any]) -> str:
    return json.dumps(record)


def _user(text: Any, **extra: Any) -> str:
    return _line({"type": "user", "message": {"role": "user", "content": text}, **extra})


def _tool_use(tool_id: str, name: str, tool_input: dict[str, Any]) -> str:
    block = {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}
    return _line({"type": "assistant", "message": {"role": "assistant", "content": [block]}})


def _tool_result(tool_id: str, text: str) -> str:
    block = {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": [{"type": "text", "text": text}],
    }
    return _line({"type": "user", "message": {"role": "user", "content": [block]}})


class TestHarvest:
    def test_mcp_query_records_the_historical_refusal(self) -> None:
        transcript = "\n".join(
            [
                _tool_use("t1", "mcp__particles__query", {"question": "Who maintains the gate?"}),
                _tool_result("t1", f'{{"answer": "… ({REFUSAL_PHRASE} 0.25) …"}}'),
                _tool_use(
                    "t2", "mcp__particles__query", {"question": "What is the floor default?"}
                ),
                _tool_result("t2", '{"answer": "0.25"}'),
                _tool_use("t3", "mcp__particles__query", {"question": "Never answered, was it?"}),
            ]
        )
        found = transcript_questions(transcript, sources=_ALL, **_BOUNDS)
        assert [(q.source, q.historical_refused) for q in found] == [
            (QuestionSource.MCP_QUERY, True),
            (QuestionSource.MCP_QUERY, False),
            (QuestionSource.MCP_QUERY, None),
        ]

    def test_refusal_phrase_is_the_query_ops_own(self) -> None:
        """The harvester's phrase is held against the builder, in both tenses."""
        from particles.operations.query.main import _below_floor_answer

        note = RelevanceNote(max_similarity=0.09, floor=0.25, below_floor=True)
        now = _below_floor_answer(QueryRequest(question="q"), note, 3)
        then = _below_floor_answer(
            QueryRequest(question="q", as_of=datetime(2026, 1, 1, tzinfo=UTC)), note, 3
        )
        assert REFUSAL_PHRASE in now
        assert REFUSAL_PHRASE in then

    def test_cli_query_is_harvested_unless_it_addresses_another_store(self) -> None:
        transcript = "\n".join(
            [
                _tool_use(
                    "b1", "Bash", {"command": 'uv run particles query "Which ADR owns the floor?"'}
                ),
                _tool_use(
                    "b2",
                    "Bash",
                    {"command": "DATABASE_URL=sqlite:///x.db particles query 'Is Pluto a planet?'"},
                ),
                _tool_use(
                    "b3", "Bash", {"command": "particles query 'Shared question?' --store team"}
                ),
                _tool_use("b4", "Bash", {"command": "particles query --predicates"}),
            ]
        )
        found = transcript_questions(transcript, sources=_ALL, **_BOUNDS)
        assert [(q.question, q.source) for q in found] == [
            ("Which ADR owns the floor?", QuestionSource.CLI_QUERY)
        ]

    def test_typed_prompts_yield_only_the_operators_own_questions(self) -> None:
        transcript = "\n".join(
            [
                _user(
                    "<system-reminder>Is this injected text a question?</system-reminder>"
                    "Fix the build. Why does the export gate reject this sentence?\n"
                    "```\nwhat = is_this_code_a_question?\n```"
                ),
                _user("/release what happens if I run this command?"),
                _user("Does a sidechain prompt count here at all?", isSidechain=True),
                _user("Does a meta record count here at all?", isMeta=True),
                _user([{"type": "text", "text": "How does the floor treat an empty result?"}]),
                _user("ok?"),
                _tool_result("zz", "Is a tool result ever a typed question?"),
            ]
        )
        found = transcript_questions(transcript, sources=_ALL, **_BOUNDS)
        assert [q.question for q in found] == [
            "Why does the export gate reject this sentence?",
            "How does the floor treat an empty result?",
        ]
        assert {q.source for q in found} == {QuestionSource.USER_PROMPT}
        assert all(q.historical_refused is None for q in found)

    def test_prompt_source_can_be_switched_off(self) -> None:
        transcript = _user("Why does the export gate reject this sentence?")
        explicit = frozenset({QuestionSource.MCP_QUERY, QuestionSource.CLI_QUERY})
        assert transcript_questions(transcript, sources=explicit, **_BOUNDS) == []

    def test_length_bounds(self) -> None:
        assert prompt_questions("Is it a bit short?", min_chars=30, max_chars=300) == []
        assert prompt_questions("Is this one much too long? " * 1, min_chars=1, max_chars=10) == []

    def test_dedup_prefers_the_explicit_source_and_keeps_a_refusal(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "one.jsonl").write_text(
            "\n".join(
                [
                    _user("What is the relevance floor default?"),
                    _tool_use(
                        "t1",
                        "mcp__particles__query",
                        {"question": "what is the relevance  floor default?"},
                    ),
                    _tool_result("t1", REFUSAL_PHRASE),
                ]
            )
        )
        (tmp_path / "two.jsonl").write_text(
            "\n".join(
                [
                    _tool_use(
                        "t9",
                        "mcp__particles__query",
                        {"question": "What is the relevance floor default?"},
                    ),
                    _tool_result("t9", "answered"),
                    "not json at all",
                ]
            )
        )
        harvested = harvest_transcripts(tmp_path, **_BOUNDS)
        assert harvested.transcripts_scanned == 2
        assert len(harvested.questions) == 1
        kept = harvested.questions[0]
        assert kept.source is QuestionSource.MCP_QUERY
        assert kept.historical_refused is True
        assert harvested.by_source == {"mcp_query": 1}
        assert (harvested.historical_results, harvested.historical_refusals) == (2, 1)

    def test_redaction_runs_before_the_id_is_taken(self, tmp_path: Path) -> None:
        (tmp_path / "t.jsonl").write_text(_user("Why does the key sk-SECRET123 fail to load?"))
        harvested = harvest_transcripts(
            tmp_path, redact=lambda text: text.replace("sk-SECRET123", "[REDACTED]"), **_BOUNDS
        )
        (kept,) = harvested.questions
        assert "SECRET" not in kept.question
        assert kept.question_id == question_id(kept.question)

    def test_heldout_round_trip_and_fingerprint(self, tmp_path: Path) -> None:
        questions = [
            HeldOutQuestion(
                question_id=question_id(t), question=t, source=QuestionSource.USER_PROMPT
            )
            for t in ("First question here?", "Second question here?")
        ]
        target = tmp_path / "nested" / "heldout.jsonl"
        assert write_heldout(target, questions) == 2
        assert load_heldout(target) == questions
        assert heldout_fingerprint(questions) == heldout_fingerprint(reversed(questions))
        assert heldout_fingerprint(questions) != heldout_fingerprint(questions[:1])


# ---------------------------------------------------------------------------
# Sweep arithmetic
# ---------------------------------------------------------------------------


def _row(
    sim: float | None,
    answerable: bool | None = None,
    source: QuestionSource = QuestionSource.USER_PROMPT,
) -> QuestionResult:
    return QuestionResult(
        question_id=f"q{sim}{answerable}{source}",
        source=source,
        max_similarity=sim,
        answerable=answerable,
    )


class TestMetrics:
    def test_the_gate_is_strict(self) -> None:
        assert refused_at(0.2499, 0.25)
        assert not refused_at(0.25, 0.25)
        assert not refused_at(0.0, 0.0)  # 0.0 is the off switch: it never refuses

    def test_refusal_curve_excludes_rows_without_a_cosine(self) -> None:
        rows = [
            _row(0.10),
            _row(0.30),
            _row(0.20, source=QuestionSource.MCP_QUERY),
            _row(None),
        ]
        (at_25,) = refusal_curve(rows, [0.25])
        assert (at_25.refused.numerator, at_25.refused.denominator) == (2, 3)
        assert at_25.refused_by_source["mcp_query"].value == 1.0
        assert at_25.refused_by_source["user_prompt"].value == 0.5

    def test_floor_sweep_is_the_two_by_two_table_both_ways(self) -> None:
        rows = [
            _row(0.20, True),  # answerable, refused at 0.25 — the false negative
            _row(0.50, True),
            _row(0.60, True),
            _row(0.10, False),  # unanswerable, refused — the gate working
            _row(0.30, False),  # unanswerable, passed
            _row(0.22, None),  # unjudged: joins nothing
            _row(None, True),  # no cosine: joins nothing
        ]
        (row,) = floor_sweep(rows, [0.25])
        assert (row.answerable_refused.numerator, row.answerable_refused.denominator) == (1, 3)
        assert (row.unanswerable_passed.numerator, row.unanswerable_passed.denominator) == (1, 2)
        assert (row.refused_were_answerable.numerator, row.refused_were_answerable.denominator) == (
            1,
            2,
        )
        assert (
            row.passed_were_unanswerable.numerator,
            row.passed_were_unanswerable.denominator,
        ) == (1, 3)

    def test_an_empty_denominator_is_none_never_a_number(self) -> None:
        (row,) = floor_sweep([_row(0.9, True)], [0.10])
        assert row.unanswerable_passed.value is None
        assert row.refused_were_answerable.value is None

    def test_quantiles(self) -> None:
        assert similarity_quantiles([]) == {}
        q = similarity_quantiles([0.1, 0.2, 0.3, 0.4, 0.5])
        assert (q["p0"], q["p50"], q["p100"]) == (0.1, 0.3, 0.5)

    def test_sample_is_deterministic_stratified_and_nested(self) -> None:
        questions = [
            HeldOutQuestion(
                question_id=f"u{i:03d}", question=f"u{i}?", source=QuestionSource.USER_PROMPT
            )
            for i in range(95)
        ] + [
            HeldOutQuestion(
                question_id=f"m{i:03d}", question=f"m{i}?", source=QuestionSource.MCP_QUERY
            )
            for i in range(5)
        ]
        assert sample_questions(questions, None, 0) == questions
        assert sample_questions(questions, 500, 0) == questions
        ten = sample_questions(questions, 10, 0)
        assert ten == sample_questions(questions, 10, 0)
        assert len(ten) == 10
        assert sum(q.source is QuestionSource.MCP_QUERY for q in ten) >= 1
        forty = {q.question_id for q in sample_questions(questions, 40, 0)}
        assert {q.question_id for q in ten} <= forty
        assert {q.question_id for q in sample_questions(questions, 10, 1)} != {
            q.question_id for q in ten
        }


# ---------------------------------------------------------------------------
# Runner — the real query op, scripted providers
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9']+")


class _BagOfWordsEncoder:
    """Deterministic stand-in encoder: hashed bag of words, L2-normalised."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
                out[i, h % 384] += 1.0
            norm = float(np.linalg.norm(out[i]))
            if norm:
                out[i] /= norm
        return out


class _Scripted:
    """A provider that answers from a function of the prompt, and counts calls."""

    def __init__(self, name: str, reply: Any) -> None:
        self._name = name
        self._reply = reply
        self.calls = 0

    @property
    def provider_model(self) -> str:
        return f"scripted:{self._name}"

    async def complete(self, prompt: str, **_: object) -> str:
        self.calls += 1
        out = self._reply(prompt) if callable(self._reply) else self._reply
        if isinstance(out, Exception):
            raise out
        return str(out)


_FACTS = (
    "The relevance floor default is 0.25 raw cosine.",
    "Pluto was reclassified as a dwarf planet in 2006.",
)
_ON_TOPIC = HeldOutQuestion(
    question_id="on",
    question="What is the relevance floor default?",
    source=QuestionSource.MCP_QUERY,
)
_OFF_TOPIC = HeldOutQuestion(
    question_id="off",
    question="Which football club won yesterday?",
    source=QuestionSource.USER_PROMPT,
)


@pytest.fixture
async def floor_store(
    db_session: Any, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[None, None]:
    """Two beliefs in the in-memory store, reachable through the runner's scope."""
    from particles import embeddings as ep
    from particles.store.particle_store import insert_particle

    encoder = _BagOfWordsEncoder()
    original = ep._embedding_model
    ep.set_embedding_model(encoder)  # type: ignore[arg-type]
    for content in _FACTS:
        particle = Particle(
            content=content,
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test-agent",
            status=Status.ACTIVE,
            provenance=[
                ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id="s")
            ],
        )
        await insert_particle(db_session, particle, encoder.encode([content])[0].tolist())
    await db_session.commit()

    @contextlib.asynccontextmanager
    async def _scope(store: str = "default", *, write: bool = False) -> AsyncGenerator[Any, None]:
        yield db_session

    monkeypatch.setattr("particles.benchmark.relevance_floor.runner.session_scope", _scope)
    # One shared session: serialise the judged stage so it is never used concurrently.
    get_config().benchmark_relevance_floor.concurrency = 1
    get_config().benchmark_relevance_floor.call_retry_backoff_seconds = 0.0
    try:
        yield
    finally:
        ep.set_embedding_model(original)


def _selection(**overrides: Any) -> RunSelection:
    base: dict[str, Any] = {
        "embedding_model_id": "bow",
        "top_k": 5,
        "configured_floor": 0.25,
        "floors": [0.10, 0.25, 0.40],
        "store_active_particles": 2,
        "heldout_questions": 2,
        "heldout_fingerprint": "f",
        "judged": True,
        "answer_model_id": "scripted:answer",
        "judge_model_id": "scripted:judge",
        "judge_protocol": 1,
    }
    return RunSelection(**{**base, **overrides})


class TestRunner:
    async def test_replay_is_free_and_records_what_the_floor_reads(self, floor_store: None) -> None:
        answer = _Scripted("answer", "unused")
        with override_providers({"query_response": answer, "benchmark": answer}):
            rows = await replay_retrieval([_ON_TOPIC, _OFF_TOPIC], top_k=5)
        assert answer.calls == 0
        on, off = rows
        assert on.max_similarity is not None and on.max_similarity > 0.4
        assert off.max_similarity is not None and off.max_similarity < 0.25
        assert on.answerable is None and off.answerable is None
        assert on.context_chars > 0 and on.question_chars == len(_ON_TOPIC.question)
        report = build_report(_selection(judged=False), [on, off])
        assert report.sweep == []
        at_25 = next(r for r in report.refusal_curve if r.floor == 0.25)
        assert (at_25.refused.numerator, at_25.refused.denominator) == (1, 2)

    async def test_judged_stage_answers_below_the_floor_and_restores_it(
        self, floor_store: None, tmp_path: Path
    ) -> None:
        answer = _Scripted("answer", "The default is 0.25.")
        # Keyed on the question: both prompts carry the same two retrieved beliefs.
        judge = _Scripted("judge", lambda prompt: "no" if "football" in prompt else "yes")
        checkpoint = tmp_path / "ck.jsonl"
        with override_providers({"query_response": answer, "benchmark": judge}):
            results = await run_judged(
                [_ON_TOPIC, _OFF_TOPIC], selection=_selection(), checkpoint_path=checkpoint
            )
            assert get_config().query.relevance_floor == 0.25  # restored
            # The off-topic question sits below the floor and was answered anyway.
            assert (answer.calls, judge.calls) == (2, 2)
            on, off = results
            assert (on.answerable, off.answerable) == (True, False)
            assert off.max_similarity is not None and off.max_similarity < 0.25
            assert off.responder_refused is False
            # A second run restores both outcomes and pays for nothing.
            again = await run_judged(
                [_ON_TOPIC, _OFF_TOPIC], selection=_selection(), checkpoint_path=checkpoint
            )
            assert (answer.calls, judge.calls) == (2, 2)
            assert again == results
            # A different judge model is a different key: nothing restores.
            await run_judged(
                [_ON_TOPIC],
                selection=_selection(judge_model_id="scripted:other"),
                checkpoint_path=checkpoint,
            )
            assert answer.calls == 3

    async def test_a_responder_refusal_is_labelled_without_a_judge_call(
        self, floor_store: None
    ) -> None:
        answer = _Scripted("answer", f"{REFUSAL_MARKER}\nNothing relevant.")
        judge = _Scripted("judge", "yes")
        with override_providers({"query_response": answer, "benchmark": judge}):
            (result,) = await run_judged([_OFF_TOPIC], selection=_selection())
        assert judge.calls == 0
        assert result.responder_refused is True
        assert result.answerable is False

    async def test_a_failed_call_is_excluded_not_scored_and_not_checkpointed(
        self, floor_store: None, tmp_path: Path
    ) -> None:
        get_config().benchmark_relevance_floor.call_retries = 1
        answer = _Scripted("answer", "An answer.")
        judge = _Scripted("judge", CompletionError("overloaded"))
        checkpoint = tmp_path / "ck.jsonl"
        with override_providers({"query_response": answer, "benchmark": judge}):
            (judge_failed,) = await run_judged(
                [_ON_TOPIC], selection=_selection(), checkpoint_path=checkpoint
            )
        assert judge.calls == 2  # one retry
        assert (judge_failed.excluded, judge_failed.answerable) == ("infra", None)
        assert "overloaded" in judge_failed.excluded_detail
        assert not checkpoint.exists()

        # A failure persisted by an older run is never restored: it is retried.
        checkpoint.write_text(
            json.dumps(
                {
                    "key": _checkpoint_key(_selection()),
                    "result": judge_failed.model_copy(update={"excluded": "budget"}).model_dump(
                        mode="json"
                    ),
                }
            )
            + "\n"
        )
        healthy = _Scripted("judge", "yes")
        with override_providers({"query_response": answer, "benchmark": healthy}):
            (retried,) = await run_judged(
                [_ON_TOPIC], selection=_selection(), checkpoint_path=checkpoint
            )
        assert (retried.excluded, retried.answerable, healthy.calls) == ("", True, 1)

        # The op types its own answer failures, so a provider failure lands in
        # `infra` beside the judge's — not in the untyped `answer_failed`
        # bucket, which now only catches an engine that carries no cause.
        broken = _Scripted("answer", CompletionError("down"))
        with override_providers({"query_response": broken, "benchmark": judge}):
            (answer_failed,) = await run_judged([_ON_TOPIC], selection=_selection())
        assert (answer_failed.excluded, answer_failed.answerable) == ("infra", None)
        assert "down" in answer_failed.excluded_detail
        report = build_report(_selection(), [judge_failed, answer_failed])
        assert report.excluded == {"infra": 2}
        assert report.answerable.denominator == 0

    async def test_answer_budget_failure_is_excluded_as_budget(self, floor_store: None) -> None:
        """An exhausted answer budget is the operator's cap, never infra noise.

        The distinction is the whole reason the op types its failures: `infra`
        says a re-run may succeed, `budget` says raise `query.answer_max_tokens`
        first. Folding the two would have hidden the 21 budget failures the
        first judged run turned up behind a transport-shaped label.
        """
        from particles.llm import EmptyCompletionError

        judge = _Scripted("judge", "yes")
        # Empty on every attempt, so the op's own larger-cap retry is spent too.
        starved = _Scripted("answer", EmptyCompletionError("no text block"))
        with override_providers({"query_response": starved, "benchmark": judge}):
            (result,) = await run_judged([_ON_TOPIC], selection=_selection())
        assert (result.excluded, result.answerable) == ("budget", None)
        assert build_report(_selection(), [result]).excluded == {"budget": 1}

    async def test_text_is_suppressed_at_production(self, floor_store: None) -> None:
        get_config().benchmark.record_claim_text = False
        answer = _Scripted("answer", "The default is 0.25.")
        judge = _Scripted("judge", "yes")
        with override_providers({"query_response": answer, "benchmark": judge}):
            (result,) = await run_judged([_ON_TOPIC], selection=_selection())
        assert (result.question, result.answer, result.verdict) == ("", "", "")
        assert result.answerable is True
        assert result.context_particle_ids

    async def test_no_encoder_is_refused_before_any_call(self, no_embedding_model: None) -> None:
        with pytest.raises(RelevanceFloorError, match="no embedding model"):
            await replay_retrieval([_ON_TOPIC], top_k=5)
        with pytest.raises(RelevanceFloorError, match="no embedding model"):
            await run_judged([_ON_TOPIC], selection=_selection())

    def test_floor_disabled_restores_on_failure(self) -> None:
        get_config().query.relevance_floor = 0.31
        with pytest.raises(RuntimeError), floor_disabled() as prior:
            assert prior == 0.31
            assert get_config().query.relevance_floor == 0.0
            raise RuntimeError("boom")
        assert get_config().query.relevance_floor == 0.31

    async def test_estimate_measures_input_and_prices_only_when_it_can(
        self, floor_store: None
    ) -> None:
        answer = _Scripted("answer", "unused")
        with override_providers({"query_response": answer, "benchmark": answer}):
            rows = await replay_retrieval([_ON_TOPIC, _OFF_TOPIC], top_k=5)
            unpriced = estimate_judged_stage(rows)
        assert (unpriced.answer_calls, unpriced.judge_calls, unpriced.llm_calls) == (2, 2, 4)
        assert unpriced.cost_usd is None
        assert unpriced.answer_input_tokens > 2 * 700  # the overhead plus the measured top-k
        assert answer.calls == 0

        selection = get_config().llm.for_purpose("query_response")
        get_config().benchmark_memory.price_per_mtok[selection.model] = TokenPrice(
            input=3.0, output=15.0
        )
        priced = estimate_judged_stage(rows)
        assert priced.cost_usd is not None and priced.cost_usd > 0

    def test_a_saved_replay_is_reused_only_under_the_same_depth_and_encoder(self) -> None:
        saved = build_report(
            _selection(),
            [_row(0.2, True).model_copy(update={"question_id": "on", "answer": "old"})],
        )
        (reused,) = reuse_replay(saved, [_ON_TOPIC], _selection())
        assert reused.max_similarity == 0.2
        assert (reused.answerable, reused.answer) == (None, "")  # old verdicts never carry over
        with pytest.raises(RelevanceFloorError, match="different top_k"):
            reuse_replay(saved, [_ON_TOPIC], _selection(top_k=10))
        with pytest.raises(RelevanceFloorError, match="different embedding_model_id"):
            reuse_replay(saved, [_ON_TOPIC], _selection(embedding_model_id="other"))
        with pytest.raises(RelevanceFloorError, match="does not cover 1 of the 2"):
            reuse_replay(saved, [_ON_TOPIC, _OFF_TOPIC], _selection())

    def test_unknown_judge_protocol_is_refused(self) -> None:
        with pytest.raises(RelevanceFloorError, match="unknown judge_protocol"):
            judge_prompt("q?", ["- p"], "a", protocol=99)

    def test_judge_prompt_fences_all_three_untrusted_texts(self) -> None:
        system, user = judge_prompt("the question?", ["- a particle"], "THE-ANSWER", protocol=1)
        assert "yes or no" in system
        for text in ("the question?", "- a particle", "THE-ANSWER"):
            assert text in user and text not in system


# ---------------------------------------------------------------------------
# Report shape and rendering
# ---------------------------------------------------------------------------


def _field_names(model: type[BaseModel], seen: set[type[BaseModel]] | None = None) -> set[str]:
    seen = seen if seen is not None else set()
    if model in seen:
        return set()
    seen.add(model)
    names = set(model.model_fields)
    for field in model.model_fields.values():
        for candidate in (field.annotation, *getattr(field.annotation, "__args__", ())):
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                names |= _field_names(candidate, seen)
    return names


class TestReport:
    def test_there_is_no_aggregate_score_field(self) -> None:
        assert not {n for n in _field_names(RelevanceFloorReport) if "score" in n.lower()}

    def test_the_rendered_table_carries_no_question_text(self) -> None:
        rows = [
            QuestionResult(
                question_id="a",
                source=QuestionSource.USER_PROMPT,
                max_similarity=0.2,
                answerable=True,
                question="PRIVATE-QUESTION-TEXT",
                answer="PRIVATE-ANSWER-TEXT",
                verdict="PRIVATE-VERDICT-TEXT",
            ),
            QuestionResult(
                question_id="b",
                source=QuestionSource.MCP_QUERY,
                max_similarity=0.7,
                answerable=False,
            ),
        ]
        rendered = render_report(
            build_report(_selection(heldout_by_source={"user_prompt": 1}), rows)
        )
        assert "PRIVATE" not in rendered
        assert "← configured" in rendered
        assert "answerable→refused" in rendered
        unjudged = render_report(build_report(_selection(judged=False), rows))
        assert "not run" in unjudged


# ---------------------------------------------------------------------------
# CLI — validation and the privacy guard
# ---------------------------------------------------------------------------


@pytest.fixture
def work_tree(tmp_path: Path) -> Generator[Path, None, None]:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    yield repo


class TestCLI:
    def test_harvest_refuses_to_write_inside_a_work_tree(
        self, tmp_path: Path, work_tree: Path
    ) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        (transcripts / "t.jsonl").write_text(
            _user("Why does the export gate reject this sentence?")
        )
        inside = work_tree / "data" / "heldout.jsonl"
        args = ["benchmark", "relevance-floor", "harvest", "--transcripts", str(transcripts)]
        refused = cli.invoke(app, [*args, "--output", str(inside)])
        assert refused.exit_code == 1
        assert "git work tree" in refused.output
        assert not inside.exists()

        allowed = cli.invoke(app, [*args, "--output", str(inside), "--allow-in-repo"])
        assert allowed.exit_code == 0, allowed.output
        assert "Held-out questions: 1" in allowed.output
        assert "export gate" not in allowed.output  # a census, never a question
        assert len(load_heldout(inside)) == 1

    def test_harvest_needs_a_transcript_directory(self, tmp_path: Path) -> None:
        result = cli.invoke(
            app,
            ["benchmark", "relevance-floor", "harvest", "--transcripts", str(tmp_path / "missing")],
        )
        assert result.exit_code == 1
        assert "no transcript directory" in result.output

    def test_run_needs_a_heldout_set(self, tmp_path: Path) -> None:
        result = cli.invoke(
            app, ["benchmark", "relevance-floor", "--heldout", str(tmp_path / "none.jsonl")]
        )
        assert result.exit_code == 1
        assert "harvest" in result.output

    def test_estimate_needs_judge(self, tmp_path: Path) -> None:
        heldout = tmp_path / "h.jsonl"
        write_heldout(heldout, [_ON_TOPIC])
        result = cli.invoke(
            app, ["benchmark", "relevance-floor", "--heldout", str(heldout), "--estimate"]
        )
        assert result.exit_code == 1
        assert "--judge" in result.output

    def test_json_report_is_refused_inside_a_work_tree(
        self, tmp_path: Path, work_tree: Path
    ) -> None:
        heldout = tmp_path / "h.jsonl"
        write_heldout(heldout, [_ON_TOPIC])
        result = cli.invoke(
            app,
            [
                "benchmark",
                "relevance-floor",
                "--heldout",
                str(heldout),
                "--format",
                "json",
                "--output",
                str(work_tree / "report.json"),
            ],
        )
        assert result.exit_code == 1
        assert "git work tree" in result.output

    def test_resweep_renders_a_saved_report_over_new_floors(self, tmp_path: Path) -> None:
        report = build_report(_selection(), [_row(0.2, True), _row(0.6, False)])
        saved = tmp_path / "report.json"
        saved.write_text(report.model_dump_json())
        result = cli.invoke(
            app,
            [
                "benchmark",
                "relevance-floor",
                "resweep",
                str(saved),
                "--floor",
                "0.5",
                "--floor",
                "0.05",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "0.05" in result.output and "0.50" in result.output
        assert "0.25 " not in result.output.split("JUDGED SWEEP")[1]

        saved.write_text("{}")
        bad = cli.invoke(app, ["benchmark", "relevance-floor", "resweep", str(saved)])
        assert bad.exit_code == 1
        assert "not a relevance-floor report" in bad.output
