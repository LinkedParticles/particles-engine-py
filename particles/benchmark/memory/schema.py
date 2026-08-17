"""Memory-benchmark datatypes + report renderer.

These are **not** the techspec §13.3 schema — that one
(:mod:`particles.benchmark.schema`) is frozen and has no answer / judge /
haystack concepts. This is the third parallel, SDK-local suite shape beside
the ``modality/`` and ``polarity/`` siblings — but where
those measure one extractor's output against gold *particles*, this harness
measures the whole pipeline (deposit → extract → reconcile → query) against
gold *answers* (LongMemEval; Wu et al., ICLR 2025).

Two renderer invariants bake the honesty framing into the
artifact rather than into authorial discipline:

* **The two measurement families are never merged.** ``retrieval-stage`` and
  ``end-to-end QA`` render as separately headed sections, and there is no
  single mergeable "score" field anywhere in the model
  (``tests/test_benchmark_memory.py`` walks the model tree to pin this).
* **QA numbers never render without the baseline rows.** A skipped
  ``qa_full_context`` / ``qa_no_memory`` renders as ``not run`` — the renderer
  has no flag to omit the rows.

Subset runs additionally carry the full selection tuple (seed + strata +
limit + variant + dataset revision + resolved answer/judge/extraction model
ids + the embedding model id + top_k + thresholds snapshot) and the renderer
states the subset status in the table header.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

# The six LongMemEval question types (abstention variants share the type;
# their question_id carries the ``_abs`` suffix). Used for stratified subset
# selection and the per-type accuracy breakdown.
QUESTION_TYPES: tuple[str, ...] = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)

#: The three end-to-end QA conditions (conditions ii–iv), in the
#: fixed render order. ``qa_full_context`` and ``qa_no_memory`` are the
#: baselines that must never be buried.
QA_CONDITIONS: tuple[str, ...] = ("qa_particles", "qa_full_context", "qa_no_memory")

#: The abstention-variant marker on LongMemEval question ids.
ABSTENTION_SUFFIX = "_abs"


# ---------------------------------------------------------------------------
# Dataset shape (the loader's output; one question = one benchmark unit)
# ---------------------------------------------------------------------------


class MemoryTurn(BaseModel):
    """One speaker turn of a haystack chat session."""

    role: str
    content: str
    # LongMemEval marks the evidence-bearing turns inside evidence sessions.
    # Informational here — retrieval is scored at session granularity (§3).
    has_answer: bool = False


class MemorySession(BaseModel):
    """One timestamped user–assistant haystack session.

    Maps to exactly one ``CONVERSATION`` corpus deposit (zero
    format adaptation — this is the same material shape the harvester deposits).
    """

    session_id: str
    date: str | None = None
    turns: list[MemoryTurn] = Field(default_factory=list)


class MemoryQuestion(BaseModel):
    """One LongMemEval question: haystack sessions + labeled evidence + answer."""

    question_id: str
    question_type: str
    question: str
    answer: str | None = None
    question_date: str | None = None
    sessions: list[MemorySession] = Field(default_factory=list)
    # The dataset's labeled evidence sessions — the retrieval-stage ground truth.
    answer_session_ids: list[str] = Field(default_factory=list)

    @property
    def is_abstention(self) -> bool:
        """True for abstention-variant questions (scored per dataset protocol)."""
        return self.question_id.endswith(ABSTENTION_SUFFIX)


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


class RunSelection(BaseModel):
    """The recorded run tuple — what makes two runs comparable.

    Everything the reproducibility contract names: dataset revision, variant,
    question selection (seed + types + limit), resolved answer/judge model
    ids, the resolved extraction + embedding model ids (correction v1.74.2:
    the store's contents are a function of the first and the ranking of the
    second, so omitting them let materially different pipelines record
    identical tuples), ``top_k``, and a snapshot of the thresholds in effect.
    """

    dataset_revision: str
    variant: str
    sample_seed: int
    # None ⇒ a full (``--all``) run over every question in the variant.
    question_limit: int | None = None
    # The strata (question types) the selection drew from.
    question_types: list[str] = Field(default_factory=list)
    questions_selected: int = 0
    questions_total: int = 0
    top_k: int = 10
    # Resolved "<provider>:<model>" ids. One answer model across conditions
    # ii–iv is enforced by the runner (SameModelViolation), so a rendered
    # report can only ever carry one.
    answer_model_id: str | None = None
    judge_model_id: str | None = None
    # Resolved "<provider>:<model>" id of ``llm.extraction`` (can
    # route it to a local model) and the embedding model id. The store's
    # contents are a function of the first and the ranking of the second, so
    # both are part of what makes two runs comparable. The runner resolves
    # them at run start and pins extraction (like the answer and judge
    # models) by refusal on mid-run drift.
    extraction_model_id: str | None = None
    embedding_model_id: str | None = None
    # ablation knobs — part of the comparability tuple: a
    # token-clamped qa_particles context or an abstraction-consolidated store
    # is a different experiment, so two runs are comparable only when these
    # match. ``context_budget_tokens`` clamps condition ii's context (the
    # QA-at-budget instrument); ``abstraction`` runs the pass
    # (mode=auto, age gate 0) on each scratch store between extract and
    # retrieve.
    context_budget_tokens: int | None = None
    abstraction: bool = False
    # Snapshot of the pipeline thresholds in effect for the run.
    thresholds: dict[str, float] = Field(default_factory=dict)
    # The memory under test: ``particles`` (the store) or one of the
    # comparator memories in :mod:`particles.benchmark.memory.comparators`
    # (``chunks`` — raw-transcript RAG; ``notes`` — LLM-written session
    # notes). Part of the tuple: a comparator report is a different
    # experiment, and its ``qa_particles`` slot carries a differently named
    # condition (``qa_chunks`` / ``qa_notes``), so nothing here can be read
    # as a Particles number by mistake.
    memory: str = "particles"

    @property
    def subset(self) -> bool:
        """True when the run covers a strict subset of the variant's questions."""
        return self.questions_selected < self.questions_total


class RetrievalQuestionResult(BaseModel):
    """Per-question retrieval-stage drill-down (condition i).

    ``recall_at_k`` / ``precision_at_k`` are ``None`` for a zero-evidence
    (abstention-variant) question — retrieval is unscoreable there, and the
    row says so explicitly rather than carrying a vacuous ``1.0`` /
    deterministic ``0.0``. ``abstention`` marks the row; the runner excludes
    it from every aggregate on :class:`RetrievalStageMetrics` and the
    renderer prints it as ``n/a (abstention)``.
    """

    question_id: str
    question_type: str
    evidence_sessions: int
    evidence_sessions_hit: int
    particles_retrieved: int
    recall_at_k: float | None
    precision_at_k: float | None
    abstention: bool = False


class RetrievalStageMetrics(BaseModel):
    """The retrieval-stage measurement family (condition i) — store + ranker only.

    A property of the particle store and its ranker: whether the top-k query
    result's provenance chains land on the labeled evidence sessions. It says
    nothing about answer accuracy — that is the end-to-end QA family's job,
    and the two are never merged.

    Zero-evidence (abstention-variant) questions contribute to **none** of
    the aggregates here — there is no labeled evidence to score, so blending
    them would inflate mean recall (vacuous 1.0s) and deflate mean precision
    (every retrieved particle a "false positive"). They appear in
    ``per_question`` (marked, unscored), are counted by
    ``abstention_questions``, and stay fully in the QA family, which the
    dataset's protocol does score (correction v1.74.2).
    """

    #: Evidence-bearing questions — the support of every aggregate below.
    questions: int = 0
    #: Zero-evidence (abstention) questions excluded from the aggregates.
    abstention_questions: int = 0
    mean_recall_at_k: float = 0.0
    mean_precision_at_k: float = 0.0
    recall_by_type: dict[str, float] = Field(default_factory=dict)
    per_question: list[RetrievalQuestionResult] = Field(default_factory=list)


class QaQuestionResult(BaseModel):
    """Per-question end-to-end QA drill-down for one condition."""

    question_id: str
    question_type: str
    correct: bool
    abstention: bool = False


class QaConditionMetrics(BaseModel):
    """One end-to-end QA condition (ii, iii, or iv) — answering model on top.

    ``model_id`` is the resolved answer-model id stamped by the runner; the
    same-model invariant across conditions ii–iv is enforced structurally
    (the runner refuses a mismatched set), so all three conditions of a
    rendered report carry the same value.
    """

    condition: str
    model_id: str
    questions: int = 0
    accuracy: float = 0.0
    accuracy_by_type: dict[str, float] = Field(default_factory=dict)
    per_question: list[QaQuestionResult] = Field(default_factory=list)


class MemoryBenchmarkReport(BaseModel):
    """Output of one memory-benchmark run.

    The two measurement families are separate sub-structures by design;
    there is deliberately **no single aggregate "score" field** anywhere in
    this model — merging retrieval-stage recall with QA accuracy is the
    conflation failure mode the ADR exists to exclude. A ``None`` QA
    condition renders as ``not run``; there is no way to omit its row.
    """

    benchmark: str = "longmemeval"
    selection: RunSelection
    # Family 1 — retrieval-stage (store + ranker).
    retrieval_stage: RetrievalStageMetrics = Field(default_factory=RetrievalStageMetrics)
    # Family 2 — end-to-end QA (answering model on top). None ⇒ not run.
    # ``qa_particles`` is the slot for the memory under test — for a
    # comparator run (``selection.memory != "particles"``) its ``condition``
    # reads ``qa_chunks`` / ``qa_notes`` and the renderer labels it so.
    qa_particles: QaConditionMetrics | None = None
    qa_full_context: QaConditionMetrics | None = None
    qa_no_memory: QaConditionMetrics | None = None
    quality_notes: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Renderer — the §6 invariants live here, not in the CLI
# ---------------------------------------------------------------------------

_CONDITION_LABELS: dict[str, str] = {
    "qa_particles": "qa_particles     (question + top-k retrieved particles)",
    "qa_chunks": "qa_chunks        (COMPARATOR: question + top-k raw transcript chunks)",
    "qa_notes": "qa_notes         (COMPARATOR: question + top-k LLM session notes)",
    "qa_full_context": "qa_full_context  (BASELINE: full concatenated haystack)",
    "qa_no_memory": "qa_no_memory     (BASELINE: question only)",
}


def _selection_header(selection: RunSelection) -> str:
    """One header line stating subset status + the full selection tuple."""
    scope = (
        (f"SUBSET run — {selection.questions_selected} of {selection.questions_total} questions")
        if selection.subset
        else f"FULL run — {selection.questions_selected} questions"
    )
    limit = "all" if selection.question_limit is None else str(selection.question_limit)
    strata = ",".join(selection.question_types) if selection.question_types else "all types"
    memory = "" if selection.memory == "particles" else f" | memory={selection.memory} (COMPARATOR)"
    return (
        f"{scope}{memory} | seed={selection.sample_seed} strata=[{strata}] limit={limit} "
        f"variant={selection.variant} revision={selection.dataset_revision} "
        f"top_k={selection.top_k} "
        f"answer_model={selection.answer_model_id or 'not run'} "
        f"judge_model={selection.judge_model_id or 'not run'} "
        f"extraction_model={selection.extraction_model_id or 'not recorded'} "
        f"embedding_model={selection.embedding_model_id or 'not recorded'}"
    )


def render_report_table(report: MemoryBenchmarkReport) -> str:
    """Render the four-condition table with the two families separately headed.

    The §6 structural invariants: the ``retrieval-stage`` and ``end-to-end
    QA`` families are separate sections (never merged into one number), and
    the QA section always renders all three condition rows — a skipped
    condition shows ``not run``. The header always states subset status and
    the full selection tuple. Excluded abstention questions are disclosed in
    the retrieval section — the count, plus each question id rendered as
    ``n/a (abstention)`` (correction v1.74.2).
    """
    lines: list[str] = []
    lines.append("Memory benchmark — LongMemEval")
    lines.append(_selection_header(report.selection))
    if report.selection.thresholds:
        thresholds = "  ".join(f"{k}={v:g}" for k, v in sorted(report.selection.thresholds.items()))
        lines.append(f"thresholds: {thresholds}")
    if report.selection.context_budget_tokens is not None or report.selection.abstraction:
        knobs: list[str] = []
        if report.selection.context_budget_tokens is not None:
            knobs.append(f"context_budget={report.selection.context_budget_tokens} tokens")
        if report.selection.abstraction:
            knobs.append("abstraction pass ON")
        lines.append(f"ablation: {'  '.join(knobs)} — compare only against a matching run")
    lines.append("")

    r = report.retrieval_stage
    stage = (
        "store + ranker"
        if report.selection.memory == "particles"
        else f"{report.selection.memory} index + ranker (COMPARATOR)"
    )
    lines.append(f"== Retrieval stage ({stage}; NOT answer accuracy) ==")
    lines.append(f"  questions scored : {r.questions}")
    if r.abstention_questions:
        lines.append(
            f"  {r.abstention_questions} abstention question(s) excluded from retrieval "
            f"aggregates — no labeled evidence to score:"
        )
        lines.extend(
            f"    {row.question_id}: recall/precision n/a (abstention)"
            for row in r.per_question
            if row.abstention
        )
    lines.append(f"  Recall@k         : {r.mean_recall_at_k:.3f}")
    lines.append(f"  Precision@k      : {r.mean_precision_at_k:.3f}")
    for qtype, recall in sorted(r.recall_by_type.items()):
        lines.append(f"    recall[{qtype}]: {recall:.3f}")
    lines.append("")

    lines.append("== End-to-end QA (answering model on top; NOT retrieval) ==")
    for condition in QA_CONDITIONS:
        metrics: QaConditionMetrics | None = getattr(report, condition)
        label = _CONDITION_LABELS[condition]
        if metrics is not None and metrics.condition in _CONDITION_LABELS:
            label = _CONDITION_LABELS[metrics.condition]
        if metrics is None:
            lines.append(f"  {label}: not run")
        else:
            lines.append(
                f"  {label}: accuracy {metrics.accuracy:.3f} "
                f"({metrics.questions} question(s), model {metrics.model_id})"
            )
            for qtype, acc in sorted(metrics.accuracy_by_type.items()):
                lines.append(f"      accuracy[{qtype}]: {acc:.3f}")
    lines.append("")

    if report.quality_notes:
        lines.append("Quality notes:")
        lines.extend(f"  - {note}" for note in report.quality_notes)

    return "\n".join(lines).rstrip() + "\n"
