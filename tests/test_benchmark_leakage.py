# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model-prior leakage benchmark and the shared entailment judge.

No API key, no network: the splitter, the judge's prompt and parser, and the
rates are pure; the runner tier drives the real query op over an in-memory
store with a deterministic bag-of-words encoder and scripted composer/judge
providers at the completion-provider port.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.leakage import (
    LeakageError,
    QueryLeakage,
    RunSelection,
    SentenceResult,
    SentenceVerdict,
    build_report,
    check_distinct_judge,
    estimate_run,
    judge_prompt,
    measure_question,
    render_report,
    rubric,
    run_leakage,
    split_sentences,
)
from particles.benchmark.leakage.runner import _checkpoint_key
from particles.benchmark.relevance_floor import HeldOutQuestion, QuestionSource, write_heldout
from particles.config import TokenPrice, get_config
from particles.core.schema import (
    AttributionKind,
    CalibrationSource,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Status,
    UncertaintyNature,
)
from particles.llm import CompletionError, override_providers
from particles.operations.entailment import (
    EntailmentRubric,
    EntailmentVerdict,
    entailment_prompt,
    parse_entailment,
)
from particles.operations.query.respond import REFUSAL_MARKER

cli = CliRunner()

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_prose_splits_at_terminal_punctuation(self) -> None:
        assert split_sentences("Pluto is small. It is far away! Is it cold? Yes.") == [
            "Pluto is small.",
            "It is far away!",
            "Is it cold?",
            "Yes.",
        ]

    def test_abbreviations_initials_decimals_and_versions_do_not_split(self) -> None:
        text = (
            "The floor is 0.25 in v1.2.3 (e.g. the default). Dr. Smith and J. R. Tolkien "
            "agree, approx. three times. Then it ends."
        )
        assert split_sentences(text) == [
            "The floor is 0.25 in v1.2.3 (e.g. the default).",
            "Dr. Smith and J. R. Tolkien agree, approx. three times.",
            "Then it ends.",
        ]

    def test_markdown_structure_makes_units(self) -> None:
        text = (
            "Here is what I found:\n\n"
            "## Status\n"
            "- **Pluto** is a dwarf planet. It was\n"
            "  reclassified in 2006.\n"
            "1. Run `uv sync`.\n"
            "> A quoted line.\n\n"
            "| field | value |\n"
            "|---|---|\n"
            "| floor | 0.25 |\n"
            "---\n"
            "```bash\n"
            "echo one. Two\n"
            "```\n"
            "***"
        )
        assert split_sentences(text) == [
            "Here is what I found:",
            "Status",
            "Pluto is a dwarf planet.",
            "It was reclassified in 2006.",
            "Run `uv sync`.",
            "A quoted line.",
            "field | value",
            "floor | 0.25",
            "echo one. Two",
        ]

    def test_units_without_a_letter_or_digit_are_dropped(self) -> None:
        assert split_sentences("...\n\n- \n\n— !") == []

    def test_deterministic(self) -> None:
        text = "One. Two.\n\n- Three. Four."
        assert split_sentences(text) == split_sentences(text) == ["One.", "Two.", "Three.", "Four."]


# ---------------------------------------------------------------------------
# The shared entailment judge
# ---------------------------------------------------------------------------

_BINARY = EntailmentRubric(
    instructions="Decide.",
    claim_heading="Claim",
    claim_label="claim",
    premise_heading="Premise",
    premise_label="premise",
)


class TestEntailmentJudge:
    def test_abstraction_rubric_keeps_its_original_prompt_and_schema(self) -> None:
        """The refactor that made the judge shared moved no promotion verdict."""
        from particles.operations.abstraction import _ENTAILMENT_RUBRIC

        assert _ENTAILMENT_RUBRIC.schema == {
            "type": "object",
            "properties": {"reason": {"type": "string"}, "entailed": {"type": "boolean"}},
            "required": ["reason", "entailed"],
            "additionalProperties": False,
        }
        system, user = entailment_prompt("G", ["a", "b"], rubric=_ENTAILMENT_RUBRIC)
        assert system.startswith("You are auditing a knowledge base. Decide whether the GENERAL")
        assert "Detail loss also fails this gate" in system
        assert "Write the reason first" in system
        assert re.fullmatch(
            r'General claim:\n<general nonce="(\w+)">\nG\n</general nonce="\1">\n\n'
            r'Specific claim 1:\n<specific_1 nonce="\1">\na\n</specific_1 nonce="\1">\n\n'
            r'Specific claim 2:\n<specific_2 nonce="\1">\nb\n</specific_2 nonce="\1">',
            user,
        )

    async def test_abstraction_still_calls_through_its_own_binding(self) -> None:
        """Its tests patch the abstraction module's seam; the shared judge keeps that seam."""
        from unittest.mock import AsyncMock, patch

        from particles.operations import abstraction as ab

        seam = AsyncMock(return_value='{"entailed": false, "reason": "broader"}')
        with patch.object(ab, "_llm_call", seam):
            assert await ab._check_entailment("G", ["a"]) is False
        assert seam.await_args is not None
        assert seam.await_args.kwargs["purpose"] == "abstraction"

    def test_context_blocks_are_fenced_ahead_of_the_claim(self) -> None:
        system, user = entailment_prompt(
            "THE-CLAIM", ["THE-PREMISE"], rubric=_BINARY, context=[("Question", "q", "THE-Q")]
        )
        assert user.index("THE-Q") < user.index("THE-CLAIM") < user.index("THE-PREMISE")
        for text in ("THE-Q", "THE-CLAIM", "THE-PREMISE"):
            assert text in user and text not in system

    def test_parse_reads_the_verdict_and_refuses_anything_unusable(self) -> None:
        three_way = rubric(1)
        assert parse_entailment('{"entailed": true, "reason": "r"}', rubric=_BINARY) == (
            EntailmentVerdict.ENTAILED,
            "r",
        )
        assert parse_entailment('```json\n{"entailed": false}\n```', rubric=_BINARY) == (
            EntailmentVerdict.NOT_ENTAILED,
            "",
        )
        assert parse_entailment(
            '{"asserts_claim": false, "entailed": true, "reason": "lead-in"}', rubric=three_way
        ) == (EntailmentVerdict.NO_CLAIM, "lead-in")
        # The binary rubric never yields NO_CLAIM, whatever the reply carries.
        assert parse_entailment('{"asserts_claim": false, "entailed": true}', rubric=_BINARY) == (
            EntailmentVerdict.ENTAILED,
            "",
        )
        # Prose carrying a brace of its own before the object still parses.
        assert parse_entailment(
            'see {a, b}: {"reason": "r", "entailed": true}', rubric=_BINARY
        ) == (
            EntailmentVerdict.ENTAILED,
            "r",
        )
        for unusable in (None, "", "yes", '{"entailed": "yes"}', "[true]"):
            assert parse_entailment(unusable, rubric=_BINARY) is None
        assert parse_entailment('{"entailed": true}', rubric=three_way) is None

    def test_three_way_schema_requires_the_claim_flag(self) -> None:
        schema = rubric(1).schema
        assert schema["required"] == ["reason", "asserts_claim", "entailed"]
        assert list(schema["properties"]) == ["reason", "asserts_claim", "entailed"]

    def test_protocol_one_is_kept_byte_for_byte(self) -> None:
        """The baseline was measured on this text; it never changes."""
        digest = hashlib.sha256(rubric(1).instructions.encode()).hexdigest()
        assert digest == "439c976ccf645ca186d46ffc735426b3243a44d52466eaa1af94128eaee21385"

    def test_protocol_two_asks_for_the_reason_first(self) -> None:
        """the same rubric, reasoning before the verdicts, and the default."""
        first, second = rubric(1).instructions, rubric(2).instructions
        shared = first[: first.index("Return a JSON object")]
        assert second.startswith(shared)
        assert "Write the reason first" in second and "Write the reason first" not in first
        shape = second[second.index("Return a JSON object") :]
        assert shape.index('"reason"') < shape.index('"asserts_claim"') < shape.index('"entailed"')
        assert get_config().benchmark_leakage.judge_protocol == 2
        # Field order is the prompt's business; the parser reads by name.
        reply = '{"reason": "restates claim 1", "asserts_claim": true, "entailed": true}'
        assert parse_entailment(reply, rubric=rubric(2)) == (
            EntailmentVerdict.ENTAILED,
            "restates claim 1",
        )

    def test_unknown_protocol_is_refused(self) -> None:
        with pytest.raises(LeakageError, match="unknown judge_protocol"):
            rubric(99)

    def test_judge_prompt_carries_preceding_context_but_not_as_a_premise(self) -> None:
        sentences = ["Pluto is small.", "It is cold.", "It is far.", "It was demoted."]
        system, user = judge_prompt(
            "Tell me about Pluto?",
            sentences,
            3,
            ["Pluto is a dwarf planet."],
            protocol=1,
            context_sentences=2,
        )
        assert "asserts_claim" in system
        assert "Tell me about Pluto?" in user
        assert "It is cold. It is far." in user
        assert "Pluto is small." not in user  # outside the two-sentence window
        assert user.index("It was demoted.") < user.index("Pluto is a dwarf planet.")
        _, first = judge_prompt("q?", sentences, 0, ["p"], protocol=1, context_sentences=2)
        assert "Preceding answer text" not in first


# ---------------------------------------------------------------------------
# Runner — the real query op, scripted providers
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9']+")
_SENTENCE = re.compile(r'<sentence nonce="\w+">\n(.*?)\n</sentence', re.S)


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
    """A provider that answers from a function of the prompt, and records each one."""

    def __init__(self, name: str, reply: Any) -> None:
        self._name = name
        self._reply = reply
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    @property
    def provider_model(self) -> str:
        return f"scripted:{self._name}"

    async def complete(self, prompt: str, **_: object) -> str:
        self.prompts.append(prompt)
        out = self._reply(prompt) if callable(self._reply) else self._reply
        if isinstance(out, Exception):
            raise out
        return str(out)


def _verdicts(prompt: str) -> str:
    """Judge by the sentence under audit: a lead-in, a supported claim, or leakage."""
    found = _SENTENCE.search(prompt)
    sentence = found.group(1) if found else ""
    if sentence.endswith(":"):
        return '{"asserts_claim": false, "entailed": false, "reason": "lead-in"}'
    supported = "0.25" in sentence
    return json.dumps({"asserts_claim": True, "entailed": supported, "reason": "r"})


def _sentence(prompt: str) -> str:
    found = _SENTENCE.search(prompt)
    return found.group(1) if found else ""


def _grounded_reply(prompt: str) -> str:
    """Cite the floor fact by the handle the composer was shown, and label the rest."""
    found = re.search(r"\[(p-[0-9a-f]+)\] The relevance floor", prompt)
    handle = found.group(1) if found else "p-missing"
    return (
        f"The floor defaults to 0.25 [{handle}]. It stops off-topic answers [inference]. "
        "Pluto is a planet [background]. Mars has two moons [p-0badf00d]."
    )


_FACTS = (
    "The relevance floor default is 0.25 raw cosine.",
    "Pluto was reclassified as a dwarf planet in 2006.",
)
_ANSWER = (
    "Here is what the knowledge base says:\n\n"
    "- The relevance floor defaults to 0.25.\n"
    "- It was introduced to stop off-topic answers in 2025."
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
async def leakage_store(
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

    monkeypatch.setattr("particles.benchmark.leakage.runner.session_scope", _scope)
    # One shared session: serialise the run so it is never used concurrently.
    get_config().benchmark_leakage.concurrency = 1
    get_config().benchmark_leakage.call_retry_backoff_seconds = 0.0
    get_config().query.relevance_floor = 0.0  # every question reaches the composer
    try:
        yield
    finally:
        ep.set_embedding_model(original)


def _selection(**overrides: Any) -> RunSelection:
    base: dict[str, Any] = {
        "store": "default",
        "store_active_particles": 2,
        "embedding_model_id": "bow",
        "top_k": 5,
        "audience": "GENERAL",
        "configured_floor": 0.0,
        "composer_model_id": "scripted:composer",
        "judge_model_id": "scripted:judge",
        "judge_protocol": 1,
        "distinct_judge": True,
        "context_sentences": 2,
        "questions": 1,
        "fingerprint": "f",
    }
    return RunSelection(**{**base, **overrides})


class TestRunner:
    async def test_each_sentence_is_judged_against_what_the_composer_saw(
        self, leakage_store: None
    ) -> None:
        composer = _Scripted("composer", _ANSWER)
        judge = _Scripted("judge", _verdicts)
        with override_providers({"query_response": composer, "benchmark": judge}):
            result = await measure_question(_ON_TOPIC, store="default", top_k=5, protocol=1)
        assert composer.calls == 1
        assert judge.calls == 3  # one call per sentence
        assert [s.verdict for s in result.sentences] == [
            SentenceVerdict.NO_CLAIM,
            SentenceVerdict.SUPPORTED,
            SentenceVerdict.UNSUPPORTED,
        ]
        assert (result.supported, result.unsupported, result.no_claim) == (1, 1, 1)
        assert result.unsupported_fraction == 0.5
        # The premises are exactly the retrieved particles the composer answered from.
        for fact in _FACTS:
            assert all(fact in prompt for prompt in judge.prompts)
        assert set(result.context_particle_ids) and result.hit_count == 2
        assert result.answer == _ANSWER and result.sentences[2].text.startswith("It was")

    async def test_a_refusal_is_never_judged(self, leakage_store: None) -> None:
        composer = _Scripted("composer", f"{REFUSAL_MARKER}\nNothing relevant.")
        judge = _Scripted("judge", _verdicts)
        with override_providers({"query_response": composer, "benchmark": judge}):
            (result,) = await run_leakage([_OFF_TOPIC], selection=_selection())
        assert judge.calls == 0
        assert result.refused is True and result.sentences == []
        report = build_report(_selection(), [result])
        assert report.unsupported_sentences.denominator == 0
        assert (report.refused.numerator, report.refused.denominator) == (1, 1)

    async def test_a_self_judging_run_is_refused_unless_turned_off(
        self, leakage_store: None
    ) -> None:
        same = _Scripted("same", _verdicts)
        with override_providers({"query_response": same, "benchmark": same}):
            with pytest.raises(LeakageError, match="both resolve to scripted:same"):
                await run_leakage([_ON_TOPIC], selection=_selection())
            assert same.calls == 0
            get_config().benchmark_leakage.require_distinct_judge = False
            check_distinct_judge()

    async def test_failures_are_excluded_never_scored_and_never_checkpointed(
        self, leakage_store: None, tmp_path: Path
    ) -> None:
        get_config().benchmark_leakage.call_retries = 1
        composer = _Scripted("composer", _ANSWER)
        broken = _Scripted("judge", CompletionError("overloaded"))
        checkpoint = tmp_path / "ck.jsonl"
        with override_providers({"query_response": composer, "benchmark": broken}):
            (result,) = await run_leakage(
                [_ON_TOPIC], selection=_selection(), checkpoint_path=checkpoint
            )
        assert broken.calls == 6  # three sentences, one retry each
        assert all(s.verdict is None and s.excluded == "infra" for s in result.sentences)
        assert result.unsupported_fraction is None
        assert not checkpoint.exists()

        garbled = _Scripted("judge", "I think it is fine.")
        with override_providers({"query_response": composer, "benchmark": garbled}):
            (unparsed,) = await run_leakage([_ON_TOPIC], selection=_selection())
        assert {s.excluded for s in unparsed.sentences} == {"unparseable"}
        report = build_report(_selection(), [unparsed])
        assert report.sentences.judge_excluded == 3
        assert report.unsupported_sentences.denominator == 0

        down = _Scripted("composer", CompletionError("down"))
        with override_providers({"query_response": down, "benchmark": garbled}):
            (answer_failed,) = await run_leakage([_ON_TOPIC], selection=_selection())
        assert (answer_failed.excluded, answer_failed.sentences) == ("infra", [])
        assert build_report(_selection(), [answer_failed]).excluded == {"infra": 1}

    async def test_an_unexpected_failure_excludes_one_question_not_the_run(
        self, leakage_store: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.benchmark.leakage import runner

        real = runner.measure_question

        async def flaky(held: HeldOutQuestion, **kwargs: Any) -> QueryLeakage:
            if held.question_id == "off":
                raise TimeoutError("QueuePool limit reached")
            return await real(held, **kwargs)

        monkeypatch.setattr(runner, "measure_question", flaky)
        composer = _Scripted("composer", _ANSWER)
        judge = _Scripted("judge", _verdicts)
        with override_providers({"query_response": composer, "benchmark": judge}):
            on, off = await run_leakage([_ON_TOPIC, _OFF_TOPIC], selection=_selection())
        assert on.excluded == "" and on.unsupported == 1
        assert off.excluded == "infra" and "QueuePool" in off.excluded_detail

    async def test_a_finished_question_is_restored_not_repaid(
        self, leakage_store: None, tmp_path: Path
    ) -> None:
        composer = _Scripted("composer", _ANSWER)
        judge = _Scripted("judge", _verdicts)
        checkpoint = tmp_path / "ck.jsonl"
        with override_providers({"query_response": composer, "benchmark": judge}):
            first = await run_leakage(
                [_ON_TOPIC], selection=_selection(), checkpoint_path=checkpoint
            )
            again = await run_leakage(
                [_ON_TOPIC], selection=_selection(), checkpoint_path=checkpoint
            )
            assert (composer.calls, judge.calls) == (1, 3)
            assert again == first
            # A different rubric is a different key: nothing restores.
            other = _selection(context_sentences=0)
            assert _checkpoint_key(other) != _checkpoint_key(_selection())
            await run_leakage([_ON_TOPIC], selection=other, checkpoint_path=checkpoint)
            assert composer.calls == 2

    async def test_a_grounded_run_judges_cited_sentences_against_their_citations(
        self, leakage_store: None
    ) -> None:
        """a cited sentence is checked against the ids it cites, specifically."""
        composer = _Scripted("composer", _grounded_reply)
        judge = _Scripted("judge", _verdicts)
        with override_providers({"query_response": composer, "benchmark": judge}):
            (result,) = await run_leakage([_ON_TOPIC], selection=_selection(grounded=True))
        assert "[p-" in composer.prompts[0]  # the composer was shown handles
        kinds = [s.attribution for s in result.sentences]
        assert kinds == [
            AttributionKind.CITED,
            AttributionKind.INFERENCE,
            AttributionKind.BACKGROUND,
            AttributionKind.UNATTRIBUTED,
        ]
        cited_prompt = next(p for p in judge.prompts if "defaults to 0.25" in _sentence(p))
        assert _FACTS[0] in cited_prompt and _FACTS[1] not in cited_prompt
        for prompt in judge.prompts:
            if "defaults to 0.25" not in _sentence(prompt):
                assert all(fact in prompt for fact in _FACTS)
        # The labelled sentences are disclosed: out of the silent numerator,
        # inside the denominator.
        assert (result.supported, result.unsupported) == (1, 1)
        assert (result.inference, result.background) == (1, 1)
        assert result.unsupported_fraction == 0.25
        assert result.sentences[3].invalid_citations == 1
        report = build_report(_selection(grounded=True), [result])
        assert (
            report.unsupported_sentences.numerator,
            report.unsupported_sentences.denominator,
        ) == (
            1,
            4,
        )
        assert report.inference_sentences.numerator == report.background_sentences.numerator == 1
        counts = report.sentences
        assert (counts.cited, counts.unattributed, counts.invalid_citations) == (1, 1, 1)
        assert (counts.unsupported_cited, counts.unsupported_unattributed) == (0, 1)
        table = render_report(report)
        assert "SILENT UNSUPPORTED SENTENCES" in table and "LABELLED BY THE COMPOSER" in table
        assert any("Grounded run" in note for note in report.quality_notes)

    async def test_grounded_is_its_own_checkpoint_key_and_ungrounded_keys_are_unchanged(
        self,
    ) -> None:
        """Checkpoints recorded before grounded mode existed still restore."""
        assert _checkpoint_key(_selection()) == "1cfda352abfadf38"
        assert _checkpoint_key(_selection(grounded=True)) != _checkpoint_key(_selection())

    async def test_text_is_suppressed_at_production(self, leakage_store: None) -> None:
        get_config().benchmark.record_claim_text = False
        composer = _Scripted("composer", _ANSWER)
        judge = _Scripted("judge", _verdicts)
        with override_providers({"query_response": composer, "benchmark": judge}):
            (result,) = await run_leakage([_ON_TOPIC], selection=_selection())
        assert (result.question, result.answer) == ("", "")
        assert all((s.text, s.reason) == ("", "") for s in result.sentences)
        assert result.unsupported == 1 and result.context_particle_ids

    async def test_estimate_prices_only_when_both_models_are_priced(
        self, leakage_store: None
    ) -> None:
        composer = _Scripted("composer", _ANSWER)
        judge = _Scripted("judge", _verdicts)
        get_config().llm.price_per_mtok.clear()
        with override_providers({"query_response": composer, "benchmark": judge}):
            unpriced = estimate_run([_ON_TOPIC, _OFF_TOPIC], top_k=40)
        assert (unpriced.answer_calls, unpriced.judge_calls) == (2, 16)
        assert unpriced.llm_calls == 18 and unpriced.cost_usd is None
        assert unpriced.judge_input_tokens > unpriced.answer_input_tokens
        assert composer.calls == judge.calls == 0
        for purpose in ("query_response", "benchmark"):
            model = get_config().llm.for_purpose(purpose).model
            get_config().llm.price_per_mtok[model] = TokenPrice(input=3.0, output=15.0)
        priced = estimate_run([_ON_TOPIC], top_k=40)
        assert priced.cost_usd is not None and priced.cost_usd > 0


# ---------------------------------------------------------------------------
# Report shape and rendering
# ---------------------------------------------------------------------------


def _answered(qid: str, source: QuestionSource, *verdicts: SentenceVerdict | None) -> QueryLeakage:
    return QueryLeakage(
        question_id=qid,
        source=source,
        hit_count=3,
        refused=False,
        question=f"SECRET-QUESTION-{qid}",
        answer=f"SECRET-ANSWER-{qid}",
        sentences=[
            SentenceResult(
                index=i,
                verdict=v,
                excluded="" if v is not None else "infra",
                text=f"SECRET-SENTENCE-{qid}-{i}",
                reason="SECRET-REASON",
            )
            for i, v in enumerate(verdicts)
        ],
    )


_S, _U, _N = SentenceVerdict.SUPPORTED, SentenceVerdict.UNSUPPORTED, SentenceVerdict.NO_CLAIM


def _rows() -> list[QueryLeakage]:
    return [
        _answered("a", QuestionSource.MCP_QUERY, _S, _S, _S, _U),  # 1/4
        _answered("b", QuestionSource.USER_PROMPT, _N, _U, _S, None),  # 1/2
        _answered("c", QuestionSource.USER_PROMPT, _N),  # no claim at all
        QueryLeakage(question_id="d", source=QuestionSource.USER_PROMPT, refused=True),
        QueryLeakage(question_id="e", source=QuestionSource.USER_PROMPT, excluded="empty"),
    ]


def _field_names(model: type[BaseModel], seen: set[type[BaseModel]] | None = None) -> set[str]:
    seen = seen if seen is not None else set()
    if model in seen:
        return set()
    seen.add(model)
    names: set[str] = set(model.model_fields)
    for field in model.model_fields.values():
        for arg in (field.annotation, *getattr(field.annotation, "__args__", ())):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                names |= _field_names(arg, seen)
    return names


class TestReport:
    def test_rates_pool_sentences_and_keep_every_exclusion_visible(self) -> None:
        report = build_report(_selection(questions=5), _rows())
        rate = report.unsupported_sentences
        assert (rate.numerator, rate.denominator) == (2, 6)
        by_source = report.unsupported_sentences_by_source
        assert (by_source["mcp_query"].numerator, by_source["mcp_query"].denominator) == (1, 4)
        assert (by_source["user_prompt"].numerator, by_source["user_prompt"].denominator) == (1, 2)
        assert report.mean_answer_fraction == pytest.approx((0.25 + 0.5) / 2)
        answers = report.answers_with_unsupported
        assert (answers.numerator, answers.denominator) == (2, 2)
        assert (report.refused.numerator, report.refused.denominator) == (1, 4)
        assert report.sentences.model_dump() == {
            "total": 9,
            "supported": 4,
            "unsupported": 2,
            "no_claim": 2,
            "judge_excluded": 1,
            # Grounded-only buckets stay zero on an ungrounded run.
            "inference": 0,
            "inference_entailed": 0,
            "background": 0,
            "background_entailed": 0,
            "cited": 0,
            "unattributed": 0,
            "unsupported_cited": 0,
            "unsupported_unattributed": 0,
            "invalid_citations": 0,
        }
        assert report.excluded == {"empty": 1}
        assert any("could not score" in note for note in report.quality_notes)

    def test_an_empty_run_is_none_never_a_number(self) -> None:
        report = build_report(_selection(), [])
        assert report.unsupported_sentences.value is None
        assert report.mean_answer_fraction is None

    def test_a_self_judged_run_says_its_number_is_a_lower_bound(self) -> None:
        report = build_report(_selection(distinct_judge=False), _rows())
        assert any("lower bound" in note for note in report.quality_notes)
        assert "SAME MODEL AS COMPOSER" in render_report(report)

    def test_there_is_no_blended_score_field(self) -> None:
        from particles.benchmark.leakage import LeakageReport

        assert not {"score", "overall", "total_score"} & _field_names(LeakageReport)

    def test_the_rendered_table_carries_no_question_answer_or_sentence_text(self) -> None:
        report = build_report(_selection(questions=5), _rows())
        for per_question in (False, True):
            table = render_report(report, per_question=per_question)
            assert "SECRET" not in table
            assert "33.3% (2/6)" in table
        assert "excluded (empty)" in render_report(report, per_question=True)
        assert "refused" in render_report(report, per_question=True)

    def test_a_saved_report_round_trips(self) -> None:
        from particles.benchmark.leakage import LeakageReport

        report = build_report(_selection(), _rows())
        again = LeakageReport.model_validate_json(report.model_dump_json())
        assert again.unsupported_sentences == report.unsupported_sentences
        assert again.results[0].unsupported_fraction == 0.25


# ---------------------------------------------------------------------------
# CLI — validation, the distinct-judge rule, the privacy guard
# ---------------------------------------------------------------------------


class TestCLI:
    def test_needs_a_heldout_set_or_a_question(self, tmp_path: Path) -> None:
        result = cli.invoke(
            app, ["benchmark", "leakage", "--heldout", str(tmp_path / "none.jsonl")]
        )
        assert result.exit_code == 1
        assert "--question" in result.output

    def test_question_replaces_the_heldout_set(self, tmp_path: Path) -> None:
        result = cli.invoke(
            app, ["benchmark", "leakage", "-q", "What?", "--heldout", str(tmp_path / "h.jsonl")]
        )
        assert result.exit_code == 1
        assert "replaces the held-out set" in result.output

    def test_estimate_prints_and_makes_no_call(self, tmp_path: Path) -> None:
        heldout = tmp_path / "h.jsonl"
        write_heldout(heldout, [_ON_TOPIC, _OFF_TOPIC])
        composer = _Scripted("composer", _ANSWER)
        judge = _Scripted("judge", _verdicts)
        with override_providers({"query_response": composer, "benchmark": judge}):
            result = cli.invoke(
                app, ["benchmark", "leakage", "--heldout", str(heldout), "--estimate"]
            )
        assert result.exit_code == 0, result.output
        assert "projected 18 LLM calls over 2 questions" in result.output
        assert "no LLM call was made" in result.output
        assert composer.calls == judge.calls == 0

    def test_a_self_judging_run_is_refused_before_the_estimate(self) -> None:
        same = _Scripted("same", _verdicts)
        with override_providers({"query_response": same, "benchmark": same}):
            result = cli.invoke(app, ["benchmark", "leakage", "-q", "What?", "--estimate"])
        assert result.exit_code == 1
        assert "require_distinct_judge" in result.output
        assert same.calls == 0

    def test_json_report_is_refused_inside_a_work_tree(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        result = cli.invoke(
            app,
            [
                "benchmark",
                "leakage",
                "-q",
                "What?",
                "--format",
                "json",
                "--output",
                str(repo / "report.json"),
            ],
        )
        assert result.exit_code == 1
        assert "git work tree" in result.output
