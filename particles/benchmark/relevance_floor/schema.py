# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Held-out set and report models for the relevance-floor benchmark.

Two halves, deliberately separate:

* the **held-out set** — :class:`HeldOutQuestion` rows harvested from an
  operator's own agent transcripts by :mod:`.harvest`. It is private by
  construction (real questions a real person asked), so it is written outside
  any repository and never vendored;
* the **report** — :class:`RelevanceFloorReport`. Like the sibling reports it
  carries no aggregate "score": the refusal curve (what the gate *does*) and
  the judged sweep (whether it was *right*) are separate families, and every
  rate carries its own numerator and denominator so an exclusion can never
  hide inside a percentage.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from particles.benchmark.rot.schema import Rate

# ---------------------------------------------------------------------------
# Held-out set
# ---------------------------------------------------------------------------


class QuestionSource(StrEnum):
    """Where a held-out question came from — reported apart, never blended silently."""

    #: An explicit memory query through the MCP ``query`` tool.
    MCP_QUERY = "mcp_query"
    #: ``particles query "<question>"`` run in a shell tool call.
    CLI_QUERY = "cli_query"
    #: A question-shaped sentence the operator typed to their agent. A *proxy*
    #: for a memory query: it is a real information need in the store's domain,
    #: but nobody addressed it to the store.
    USER_PROMPT = "user_prompt"


class HeldOutQuestion(BaseModel):
    """One harvested question."""

    #: Stable id — a digest of the normalised question text, so a re-harvest
    #: keeps ids and a report row can be joined back to the private set.
    question_id: str
    question: str
    source: QuestionSource
    #: For the two explicit-query sources: whether the transcript recorded the
    #: below-floor refusal for this call. ``None`` when the result was not
    #: captured, and always for :attr:`QuestionSource.USER_PROMPT`.
    historical_refused: bool | None = None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

#: A question whose answer or judge call produced nothing to score — the memory
#: benchmark's pair, same strings, so a scored call's cause passes straight through.
EXCLUSION_INFRA = "infra"
EXCLUSION_BUDGET = "budget"
#: The query op reported an answer-generation failure it could not type. The
#: op now classifies its own failures (``answer_generation_error_cause``), so
#: a typed one is recorded under the matching cause above rather than here —
#: this is the residue: an engine too old to carry the field, or a cause this
#: runner does not know. Its reason string survives in ``excluded_detail``.
EXCLUSION_ANSWER = "answer_failed"
#: The store returned no candidate at all (the floor is inert on an empty result).
EXCLUSION_EMPTY = "empty"
EXCLUSION_KINDS = (EXCLUSION_INFRA, EXCLUSION_BUDGET, EXCLUSION_ANSWER, EXCLUSION_EMPTY)
#: Causes a re-run should try again: never checkpointed, never restored.
RETRYABLE_EXCLUSIONS = frozenset({EXCLUSION_INFRA, EXCLUSION_BUDGET, EXCLUSION_ANSWER})


class QuestionResult(BaseModel):
    """One held-out question, replayed.

    ``answerable`` is the judged label and is ``None`` — never ``False`` — for a
    row the judged stage did not reach or could not score, so re-blending an
    unjudged row into a rate fails strict typing instead of deflating it (the
    exclusion discipline).
    """

    question_id: str
    source: QuestionSource
    #: Maximum raw cosine over the rendered top-k — the quantity the query op
    #: holds to the floor. ``None`` with an ``excluded`` cause when there is none.
    max_similarity: float | None = None
    hit_count: int = 0
    historical_refused: bool | None = None
    #: The responder's own no-relevant-knowledge marker fired with the gate disabled.
    responder_refused: bool | None = None
    answerable: bool | None = None
    excluded: str = ""
    #: The failing call's own reason — a provider or op error string, never
    #: question or answer text.
    excluded_detail: str = ""
    #: Sizes of the rendered top-k and the question, in characters — what the
    #: judged stage's cost projection is measured from. Sizes, never text.
    context_chars: int = 0
    question_chars: int = 0
    #: Audit trail. Suppressed at production under
    #: ``benchmark.record_claim_text: false``; ids are kept regardless.
    question: str = ""
    answer: str = ""
    verdict: str = ""
    context_particle_ids: list[str] = Field(default_factory=list)


class RefusalRow(BaseModel):
    """One floor setting of the refusal curve — what the gate does, unjudged."""

    floor: float
    #: Share of replayed questions the product would refuse at this floor.
    refused: Rate = Field(default_factory=Rate)
    refused_by_source: dict[str, Rate] = Field(default_factory=dict)


class FloorRow(BaseModel):
    """One floor setting of the judged sweep — the 2×2 table and its four rates.

    Two conditionings of the same table, because they answer different
    questions and neither substitutes for the other:

    * **truth-conditional** (the pair) — of the answerable
      questions, the share refused; of the unanswerable, the share passed;
    * **gate-conditional** — of what the floor refused, the share that was
      answerable; of what it passed, the share that was not.
    """

    floor: float
    answerable_refused: Rate = Field(default_factory=Rate)
    unanswerable_passed: Rate = Field(default_factory=Rate)
    refused_were_answerable: Rate = Field(default_factory=Rate)
    passed_were_unanswerable: Rate = Field(default_factory=Rate)


class RunSelection(BaseModel):
    """The run tuple — what a number is comparable under."""

    embedding_model_id: str
    top_k: int
    #: ``query.relevance_floor`` as configured when the run started.
    configured_floor: float
    floors: list[float]
    store_active_particles: int
    heldout_questions: int
    heldout_by_source: dict[str, int] = Field(default_factory=dict)
    #: Digest over the sorted question ids — names the set without disclosing it.
    heldout_fingerprint: str
    sample_limit: int | None = None
    sample_seed: int = 0
    judged: bool = False
    answer_model_id: str | None = None
    judge_model_id: str | None = None
    judge_protocol: int | None = None


class RelevanceFloorReport(BaseModel):
    """The report of record. No aggregate score — by design, there is no such field."""

    selection: RunSelection
    results: list[QuestionResult] = Field(default_factory=list)
    #: Quantiles of ``max_similarity`` over the scoreable rows (p0 … p100).
    similarity_quantiles: dict[str, float] = Field(default_factory=dict)
    refusal_curve: list[RefusalRow] = Field(default_factory=list)
    #: Empty until the judged stage has run.
    sweep: list[FloorRow] = Field(default_factory=list)
    answerable: Rate = Field(default_factory=Rate)
    excluded: dict[str, int] = Field(default_factory=dict)
    quality_notes: list[str] = Field(default_factory=list)
