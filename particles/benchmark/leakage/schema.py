# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Report models for the model-prior leakage benchmark.

The number this package exists for is the **unsupported-sentence rate**: of
the answer sentences that assert something, the share the retrieved particles
do not entail. It is what a composer's parametric background looks like once
it has blended into an answer with no record of where it came from.

A grounded run has the composer cite particle ids per sentence and
label its own contribution. There a cited sentence is judged against the ids
it cites, and an uncited one against everything the composer saw. A sentence
the composer labelled ``inference`` or ``background`` is reported in its own
bucket: it is disclosed, so it is not *silent*. The headline then counts only
the silent unsupported sentences, cited or unattributed, over the same
claim-bearing denominator as an ungrounded run, which is what makes the two
runs' headlines comparable.

Every rate carries its numerator and denominator (the shared :class:`Rate`),
and every count that left a denominator is reported beside it, so an excluded
sentence or question can never hide inside a percentage. A sentence the judge
could not score is ``None``, never ``UNSUPPORTED``: re-blending one into a
rate fails strict typing instead of inflating it (the discipline).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

from particles.benchmark.relevance_floor.schema import QuestionSource
from particles.benchmark.rot.schema import Rate
from particles.core.schema import AttributionKind

#: Call outcomes that left nothing to score. The first two are the memory
#: benchmark's pair, same strings, so a scored call's cause passes through.
EXCLUSION_INFRA = "infra"
EXCLUSION_BUDGET = "budget"
#: The query op reported an answer failure it could not type.
EXCLUSION_ANSWER = "answer_failed"
#: The store returned no particle, so the op composed no answer from any.
EXCLUSION_EMPTY = "empty"
#: The judge replied, but not with a usable verdict.
EXCLUSION_UNPARSEABLE = "unparseable"
#: Causes a re-run should try again: never checkpointed, never restored.
RETRYABLE_EXCLUSIONS = frozenset(
    {EXCLUSION_INFRA, EXCLUSION_BUDGET, EXCLUSION_ANSWER, EXCLUSION_UNPARSEABLE}
)


class SentenceVerdict(StrEnum):
    """The judged label of one answer sentence."""

    #: Every assertion in it is entailed by the retrieved particles.
    SUPPORTED = "supported"
    #: It asserts something the retrieved particles do not entail.
    UNSUPPORTED = "unsupported"
    #: It asserts nothing about the world — a lead-in, a statement about what
    #: the knowledge base holds, a hedge. Outside every rate's denominator.
    NO_CLAIM = "no_claim"


class SentenceResult(BaseModel):
    """One sentence of one answer, judged."""

    index: int
    verdict: SentenceVerdict | None = None
    excluded: str = ""
    #: The failing call's own reason — never sentence text.
    excluded_detail: str = ""
    #: Audit trail; suppressed at production under
    #: ``benchmark.record_claim_text: false``.
    text: str = ""
    reason: str = ""
    #: Grounded runs only: the composer's attribution of the unit this
    #: sentence came from, the ids it cited (what the sentence was judged
    #: against), and how many of its citations named no retrieved particle.
    attribution: AttributionKind | None = None
    cited_ids: list[str] = Field(default_factory=list)
    invalid_citations: int = 0

    @property
    def labelled(self) -> bool:
        """The composer disclosed this sentence as its own contribution."""
        return self.attribution in _LABELLED


#: Attribution kinds that disclose the composer's own contribution.
_LABELLED = frozenset({AttributionKind.INFERENCE, AttributionKind.BACKGROUND})


class QueryLeakage(BaseModel):
    """One question through the real query op, its answer judged sentence by sentence."""

    question_id: str
    source: QuestionSource
    hit_count: int = 0
    max_similarity: float | None = None
    #: The op answered with a refusal (the relevance floor, or the responder's
    #: own no-relevant-knowledge marker). Nothing is judged; a refusal is an
    #: outcome, not an exclusion.
    refused: bool | None = None
    excluded: str = ""
    excluded_detail: str = ""
    sentences: list[SentenceResult] = Field(default_factory=list)
    #: Audit trail; text suppressed under ``benchmark.record_claim_text: false``,
    #: ids kept regardless.
    question: str = ""
    answer: str = ""
    context_particle_ids: list[str] = Field(default_factory=list)

    def _count(self, verdict: SentenceVerdict) -> int:
        return sum(1 for s in self.sentences if s.verdict is verdict and not s.labelled)

    def _claim_bearing(self, kind: AttributionKind) -> list[SentenceResult]:
        return [
            s
            for s in self.sentences
            if s.attribution is kind
            and s.verdict in (SentenceVerdict.SUPPORTED, SentenceVerdict.UNSUPPORTED)
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def supported(self) -> int:
        """Unlabelled sentences judged supported."""
        return self._count(SentenceVerdict.SUPPORTED)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unsupported(self) -> int:
        """Unlabelled sentences judged unsupported: the silent bucket."""
        return self._count(SentenceVerdict.UNSUPPORTED)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def inference(self) -> int:
        """Claim-bearing sentences the composer labelled its own inference."""
        return len(self._claim_bearing(AttributionKind.INFERENCE))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def background(self) -> int:
        """Claim-bearing sentences the composer labelled its own background."""
        return len(self._claim_bearing(AttributionKind.BACKGROUND))

    @property
    def claim_bearing(self) -> int:
        """Every sentence that asserts something and was scored."""
        return self.supported + self.unsupported + self.inference + self.background

    @computed_field  # type: ignore[prop-decorator]
    @property
    def no_claim(self) -> int:
        """Sentences that assert nothing."""
        return self._count(SentenceVerdict.NO_CLAIM)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def judge_excluded(self) -> int:
        """Sentences the judge could not score."""
        return sum(1 for s in self.sentences if s.verdict is None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unsupported_fraction(self) -> float | None:
        """Silent unsupported over claim-bearing sentences; ``None`` when there are none."""
        scored = self.claim_bearing
        return self.unsupported / scored if scored else None


class RunSelection(BaseModel):
    """The run tuple — what a number is comparable under."""

    store: str
    store_active_particles: int
    embedding_model_id: str
    top_k: int
    audience: str
    configured_floor: float
    composer_model_id: str
    judge_model_id: str
    judge_protocol: int
    #: Whether the judge resolved to a different model from the composer.
    distinct_judge: bool
    context_sentences: int
    questions: int
    by_source: dict[str, int] = Field(default_factory=dict)
    #: Digest over the sorted question ids — names the set without disclosing it.
    fingerprint: str
    #: ``True`` when the questions came from ``--question`` rather than a held-out set.
    ad_hoc: bool = False
    sample_limit: int | None = None
    sample_seed: int = 0
    #: The composer ran in grounded mode: per-sentence citations,
    #: with its own contribution labelled.
    grounded: bool = False


class SentenceCounts(BaseModel):
    """Every sentence the splitter produced, by outcome."""

    total: int = 0
    supported: int = 0
    unsupported: int = 0
    no_claim: int = 0
    judge_excluded: int = 0
    #: Grounded runs only. ``inference`` / ``background`` are claim-bearing
    #: sentences the composer labelled, and ``*_entailed`` how many of those the
    #: judge nonetheless found the retrieved particles support. The silent
    #: ``unsupported`` count splits into ``unsupported_cited`` (judged against
    #: the ids the sentence cited) and ``unsupported_unattributed`` (no valid
    #: citation and no label). ``invalid_citations`` counts citations that
    #: named no retrieved particle.
    inference: int = 0
    inference_entailed: int = 0
    background: int = 0
    background_entailed: int = 0
    cited: int = 0
    unattributed: int = 0
    unsupported_cited: int = 0
    unsupported_unattributed: int = 0
    invalid_citations: int = 0


class LeakageReport(BaseModel):
    """The report of record. Rates with their denominators; no blended score."""

    selection: RunSelection
    results: list[QueryLeakage] = Field(default_factory=list)
    #: The headline: silent unsupported over claim-bearing sentences, pooled
    #: over every judged answer. In an ungrounded run every sentence is silent.
    unsupported_sentences: Rate = Field(default_factory=Rate)
    #: Grounded runs: claim-bearing sentences the composer labelled
    #: ``inference`` / ``background``, over the same denominator. Reported
    #: beside the headline, never folded into it.
    inference_sentences: Rate = Field(default_factory=Rate)
    background_sentences: Rate = Field(default_factory=Rate)
    unsupported_sentences_by_source: dict[str, Rate] = Field(default_factory=dict)
    #: Mean of the per-answer fractions over answers with a claim-bearing
    #: sentence — weights each answer equally where the pooled rate weights
    #: each sentence equally. ``None`` when no answer had one.
    mean_answer_fraction: float | None = None
    #: Answers with at least one unsupported sentence, over answers judged.
    answers_with_unsupported: Rate = Field(default_factory=Rate)
    #: Refused answers over answered-or-refused questions.
    refused: Rate = Field(default_factory=Rate)
    sentences: SentenceCounts = Field(default_factory=SentenceCounts)
    #: Questions excluded from every denominator, by cause.
    excluded: dict[str, int] = Field(default_factory=dict)
    quality_notes: list[str] = Field(default_factory=list)
