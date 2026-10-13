# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The probe output budget leaves a thinking model room to answer.

:data:`particles.llm.breaker.PROBE_MAX_TOKENS` is the budget every short probe
passes. On 2026-10-11 a claude-haiku-5-5 contradiction probe at 250 tokens spent
the whole budget thinking and returned no text block, so the re-anchor pass
left a restatement UNRESTATED. Each test below runs one probe call site against
a fake Anthropic client that thinks for :data:`_THINKING_TOKENS` before it
writes: more than the old budgets, well under the shared one. A probe passing
its old budget gets no text; at the shared budget it gets its verdict.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from particles.llm import set_client
from particles.llm.breaker import PROBE_MAX_TOKENS

#: What the fake model spends thinking before its first text token.
_THINKING_TOKENS = 1000


class _ThinkingClient:
    """An Anthropic SDK stand-in whose model thinks before it answers.

    A budget at or below :data:`_THINKING_TOKENS` is spent entirely on thinking:
    the reply has a thinking block, no text block, and ``stop_reason`` is
    ``max_tokens``, the shape claude-haiku-5-5 returned. A larger budget gets
    the thinking block and then ``reply``.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.budgets: list[int] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        budget = int(kwargs["max_tokens"])
        self.budgets.append(budget)
        thinking = SimpleNamespace(type="thinking", thinking="")
        if budget <= _THINKING_TOKENS:
            return SimpleNamespace(
                content=[thinking],
                stop_reason="max_tokens",
                usage=SimpleNamespace(input_tokens=300, output_tokens=budget),
            )
        return SimpleNamespace(
            content=[thinking, SimpleNamespace(type="text", text=self.reply)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=300, output_tokens=_THINKING_TOKENS + 60),
        )


@pytest.fixture
def thinking_client() -> Iterator[Any]:
    """Install a :class:`_ThinkingClient` with a given reply as the shared SDK client."""

    def install(reply: str) -> _ThinkingClient:
        client = _ThinkingClient(reply)
        set_client(client)
        return client

    try:
        yield install
    finally:
        set_client(None)


def test_the_probe_budget_exceeds_the_thinking_the_fake_model_spends() -> None:
    """The fixture is only a test of the budget while it thinks past the old ones."""
    assert 250 <= _THINKING_TOKENS < PROBE_MAX_TOKENS


@pytest.mark.asyncio
async def test_the_old_budget_returns_no_text_and_counts_a_failed_call(
    thinking_client: Any,
) -> None:
    """The 2026-10-11 failure, reproduced: 250 tokens of thinking and no answer."""
    from particles.llm.breaker import _llm_call, llm_failure_count

    thinking_client("REASON: ports differ.\nVERDICT: YES")
    before = llm_failure_count()
    assert await _llm_call("probe", max_tokens=250) is None
    assert llm_failure_count() - before == 1


@pytest.mark.asyncio
async def test_the_breaker_seam_defaults_to_the_probe_budget(thinking_client: Any) -> None:
    from particles.llm.breaker import _llm_call

    client = thinking_client("REASON: ports differ.\nVERDICT: YES")
    assert await _llm_call("probe") == "REASON: ports differ.\nVERDICT: YES"
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_extraction_contradiction_probe_answers(thinking_client: Any) -> None:
    from particles.ingest import pipeline

    client = thinking_client("REASON: port 8080 against port 9000.\nVERDICT: YES")
    assert await pipeline._llm_confirms_contradiction("Port 8080.", "Port 9000.") is True
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_reconcile_contradiction_probe_answers(thinking_client: Any) -> None:
    """The probe the re-anchor pass asks before it restates a retired claim."""
    from particles.operations.reconcile import _has_contradiction_signal

    client = thinking_client("REASON: port 8080 against port 9000.\nVERDICT: YES")
    assert await _has_contradiction_signal("Port 8080.", "Port 9000.") is True
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_update_probe_answers_without_a_retry(thinking_client: Any) -> None:
    """At the default budgets the first reading has the room its retry used to buy."""
    from particles.ingest import pipeline

    client = thinking_client("REASON: the host moved.\nSLOT: CHANGES\nVERDICT: YES")
    verdict = await pipeline._llm_slot_verdict("Hosted on Heroku.", "Hosted on Fly.io.")
    assert verdict is pipeline.SlotVerdict.CHANGES
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_census_contradiction_probe_answers(thinking_client: Any) -> None:
    from particles.operations.lint.contradictions import _llm_check_contradiction

    client = thinking_client("REASON: port 8080 against port 9000.\nVERDICT: YES")
    answer = await _llm_check_contradiction("Port 8080.", "Port 9000.")
    assert answer == "port 8080 against port 9000."
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_batched_census_probe_sends_the_probe_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from particles.operations.lint import contradictions

    call_many = AsyncMock(return_value=["REASON: both hold.\nVERDICT: NO"])
    monkeypatch.setattr(contradictions, "_llm_call_many", call_many)
    assert await contradictions._batch_check_contradictions([("A.", "B.")]) == [None]
    assert call_many.await_args is not None
    assert call_many.await_args.kwargs["max_tokens"] == PROBE_MAX_TOKENS


@pytest.mark.asyncio
async def test_the_granularity_probe_answers(thinking_client: Any) -> None:
    from particles.operations.lint.granularity import _llm_check_granularity

    client = thinking_client("YES: the port; the host")
    assert await _llm_check_granularity("Port 8080 on host x.") == "the port; the host"
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_equivalence_judge_answers_with_room_for_every_verdict(
    thinking_client: Any,
) -> None:
    from particles.operations.links_suggest import _judge_max_tokens, _judge_pairs

    client = thinking_client('{"p0": "PARAPHRASE", "p1": "DISTINCT"}')
    verdicts = await _judge_pairs([("A.", "A again."), ("B.", "C.")])
    assert [v.value if v else None for v in verdicts] == ["PARAPHRASE", "DISTINCT"]
    assert client.budgets == [_judge_max_tokens(2)]
    assert _judge_max_tokens(2) > PROBE_MAX_TOKENS


@pytest.mark.asyncio
async def test_the_paraphrase_judge_answers(thinking_client: Any) -> None:
    """The re-anchor pass's near-duplicate judge, on ``verification``."""
    from particles.operations.abstraction import _paraphrase_verdict

    client = thinking_client('{"verdict": "PARAPHRASE"}')
    verdict = await _paraphrase_verdict("A.", "A again.", purpose="verification")
    assert verdict.value == "PARAPHRASE"
    assert client.budgets == [PROBE_MAX_TOKENS]


# ---------------------------------------------------------------------------
# Short calls on the purposes that fall back to ``llm.default``. Those route to
# a model that thinks by default wherever ``llm.default`` does (claude-sonnet-5-5
# on the owner's store), so each takes the probe budget, not one sized to its
# answer.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_modality_classifier_answers(thinking_client: Any) -> None:
    """``extraction``; at the old 64 tokens the claim kept its stale default."""
    from particles.core.schema import AssertionModality
    from particles.extraction.modality import classify_modality

    client = thinking_client('{"assertion_modality": "EVALUATIVE"}')
    verdict = await classify_modality("The new layout is much nicer.")
    assert verdict is not None
    assert verdict.modality is AssertionModality.EVALUATIVE
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_benchmark_equivalence_judge_answers(thinking_client: Any) -> None:
    """``benchmark``; at the old 16 tokens every judged pair read as not-aligned."""
    from particles.benchmark.equivalence import _llm_pair_aligned
    from particles.benchmark.schema import ExpectedParticle
    from particles.core.schema import UncertaintyNature

    client = thinking_client("aligned")
    expected = ExpectedParticle(
        content="The server listens on port 8080.",
        confidence_min=0.5,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
    )
    assert await _llm_pair_aligned("Port 8080 is the server's port.", expected) is True
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_narrative_merge_label_is_synthesized(thinking_client: Any) -> None:
    """``extraction``; at the old 256 tokens the merge fell back to the first label."""
    from particles.ingest.narrative_merge import _merge_label

    client = thinking_client("A day that began in rain and ended at the coast.")
    notes: list[str] = []
    label = await _merge_label(["Morning in the rain.", "Evening at the coast."], notes)
    assert label == "A day that began in rain and ended at the coast."
    assert notes == []
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_structurizer_answers(thinking_client: Any) -> None:
    """``extraction``; the triple is a small JSON object, 512 left no room to think."""
    from particles.extraction.structure import structure_content

    client = thinking_client(
        '{"subject": "Server", "predicate": "listens on", "object": "port 8080"}'
    )
    claim = await structure_content("The server listens on port 8080.", ["Server"])
    assert claim is not None
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_abstraction_synthesis_answers(thinking_client: Any) -> None:
    """``abstraction``; the claim and its rationale, formerly at 400 tokens."""
    from particles.operations.abstraction import _synthesize_claim

    client = thinking_client(
        '{"claim": "Alice and Bob both prefer vim.", "rationale": "Each says so."}'
    )
    result = await _synthesize_claim(["Alice prefers vim.", "Bob prefers vim."])
    assert result == ("Alice and Bob both prefer vim.", "Each says so.")
    assert client.budgets == [PROBE_MAX_TOKENS]


@pytest.mark.asyncio
async def test_the_abstraction_entailment_judge_answers(thinking_client: Any) -> None:
    """``abstraction``; the reason-first reply, formerly at 600 tokens."""
    from particles.operations.abstraction import _check_entailment

    client = thinking_client('{"reason": "each premise states it.", "entailed": true}')
    verdict = await _check_entailment(
        "Alice and Bob both prefer vim.", ["Alice prefers vim.", "Bob prefers vim."]
    )
    assert verdict is True
    assert client.budgets == [PROBE_MAX_TOKENS]


def test_the_wikidata_judge_default_is_the_probe_budget() -> None:
    """Config cannot import ``particles.llm``, so its default is a literal kept equal here."""
    from particles.config import SubjectsConfig

    assert SubjectsConfig().wikidata_judge_max_tokens == PROBE_MAX_TOKENS


@pytest.mark.asyncio
async def test_the_wikidata_judge_answers(thinking_client: Any, db_session: Any) -> None:
    """``subject_resolution``; at the old 200 tokens an ambiguous name took the top hit."""
    from particles.ingest.authorities.wikidata_judge import judge_candidates

    client = thinking_client('{"qid": "Q575650"}')
    hits: list[dict[str, object]] = [
        {"id": "Q190", "label": "rust", "description": "iron oxide"},
        {"id": "Q575650", "label": "Rust", "description": "programming language"},
    ]
    judgement = await judge_candidates(db_session, "Rust", "We chose Rust over Go.", hits)
    assert judgement is not None
    assert judgement.index == 1
    assert client.budgets == [PROBE_MAX_TOKENS]
