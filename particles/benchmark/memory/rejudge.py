# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-judge a saved memory-benchmark report under the current judge protocol.

Since v1.141.1 every QA row records the answering model's reply
(``QaQuestionResult.answer``), and since v1.141.11 the judge prompt is
versioned (``benchmark_memory.judge_protocol``). Those two facts make the
judge stage separable from the answer stage: the verdicts of an existing
report can be re-scored under a *new* protocol (or a new judge model) by
re-running only the judge call over each row's stored answer. Re-answering
would cost the full-context baseline's ~115k input tokens per question
again *and* change the answers being judged — so a re-judge is both the
cheap path and the only one that isolates the judge's effect.

The output is a complete :class:`MemoryBenchmarkReport` of record, not a
scratch table:

* the retrieval stage is copied unchanged — nothing here touches a store;
* every QA condition present in the source is re-scored row by row, with
  ``accuracy`` / ``accuracy_by_type`` recomputed from the new verdicts;
* rows with no stored answer cannot be re-judged and stay excluded, under
  their original cause when they had one and as
  :data:`~particles.benchmark.memory.schema.QA_EXCLUSION_UNRECORDED` when the
  text was suppressed (``benchmark.record_claim_text: false``) — never
  scored, never silently dropped;
* ``selection.judge_protocol`` / ``selection.judge_model_id`` are set to what
  this pass used; the answer model is unchanged by re-judging and is carried
  over as recorded;
* the first quality note names the source report path, the source judge
  protocol and model, and the new ones, so the provenance is in the file.

The judge-model pin the runner enforces (``SameModelViolation`` on mid-run
drift) applies here too: one judge per table is what makes its accuracies
comparable, and a re-judge whose judge changed halfway is exactly the
mixed table the pin exists to refuse.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from particles.benchmark.memory.metrics import parse_judge_verdict
from particles.benchmark.memory.runner import (
    _JUDGE_MAX_TOKENS,
    SameModelViolation,
    _excluded_note,
    _QaAccumulator,
    _scored_call,
    judge_prompt,
)
from particles.benchmark.memory.schema import (
    QA_CONDITIONS,
    QA_EXCLUSION_UNRECORDED,
    MemoryBenchmarkReport,
    MemoryQuestion,
    QaConditionMetrics,
    QaQuestionResult,
)
from particles.config import get_config
from particles.llm.registry import get_provider

log = logging.getLogger(__name__)

#: Progress lines go out once per this many judged rows, and at each
#: condition's end — a 150-question run is ~450 sequential judge calls.
_PROGRESS_EVERY = 25


class RejudgeError(ValueError):
    """The source report cannot be re-judged as given.

    Raised before any LLM call: the report holds no stored answer at all
    (nothing to re-score), or a row names a question the supplied dataset
    does not contain (the judge prompt needs the question text and reference
    answer, which the report does not carry — the dataset must be the one
    the report was run against).
    """


def stored_answer_count(report: MemoryBenchmarkReport) -> int:
    """How many QA rows across all conditions carry an answer to re-judge."""
    return sum(
        1
        for metrics in _present_conditions(report)
        for row in metrics.per_question
        if row.answer is not None
    )


def _present_conditions(report: MemoryBenchmarkReport) -> list[QaConditionMetrics]:
    """The QA conditions the source ran, in the fixed render order."""
    present: list[QaConditionMetrics] = []
    for slot in QA_CONDITIONS:
        metrics = getattr(report, slot)
        if metrics is not None:
            present.append(metrics)
    return present


def _preserve_excluded(
    acc: _QaAccumulator, question: MemoryQuestion, row: QaQuestionResult
) -> None:
    """Carry a row with no stored answer into the new report, still excluded.

    The original cause is kept when the row had one (the answer call itself
    produced nothing, so there never was text); a row that has no cause and
    no text was suppressed at production, and is disclosed as such.
    """
    acc.record_excluded(
        question,
        row.excluded if row.excluded is not None else QA_EXCLUSION_UNRECORDED,
        answer=None,
        context_particle_ids=row.context_particle_ids,
    )


async def rejudge_report(
    source: MemoryBenchmarkReport,
    questions: list[MemoryQuestion],
    *,
    source_path: str,
    progress: Callable[[str], None] | None = None,
) -> MemoryBenchmarkReport:
    """Re-score every stored answer in ``source`` under the configured judge.

    ``questions`` is the dataset the source was run against (any superset
    of the report's rows will do — rows are matched by ``question_id``);
    the judge prompt needs each question's text, type, and reference
    answer, none of which the report carries. Sequential, with the
    runner's own retry/backoff (``benchmark_memory.call_retries`` /
    ``call_retry_backoff_seconds``); a judge call that still produces no
    verdict is excluded and disclosed exactly as the runner does it.

    Raises :class:`RejudgeError` before any LLM call when there is nothing
    to re-judge or the dataset does not cover the report, and
    :class:`SameModelViolation` if ``llm.benchmark`` resolves to a
    different model mid-pass.
    """
    conditions = _present_conditions(source)
    if stored_answer_count(source) == 0:
        raise RejudgeError(
            f"{source_path} holds no stored answers (no QA condition ran, or the run "
            f"was made with benchmark.record_claim_text off); there is nothing to "
            f"re-judge. Only a report written by v1.141.1 or later with answer text "
            f"recorded can be re-scored."
        )
    by_id = {q.question_id: q for q in questions}
    missing = sorted(
        {
            row.question_id
            for metrics in conditions
            for row in metrics.per_question
            if row.question_id not in by_id
        }
    )
    if missing:
        shown = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
        raise RejudgeError(
            f"The dataset does not contain {len(missing)} question(s) the report scores "
            f"({shown}); the re-judge needs the same variant and revision the report "
            f"was run against (variant={source.selection.variant!r}, "
            f"revision={source.selection.dataset_revision!r})."
        )

    cfg = get_config().benchmark_memory
    protocol = cfg.judge_protocol
    retries = cfg.call_retries
    backoff = cfg.call_retry_backoff_seconds
    record_text = get_config().benchmark.record_claim_text

    judge_model_id: str | None = None
    notes: list[str] = []
    judged = 0
    unrecorded = 0
    excluded_kept = 0
    revived = 0
    rescored: dict[str, QaConditionMetrics] = {}

    for metrics in conditions:
        acc = _QaAccumulator(metrics.condition)
        rows = metrics.per_question
        for index, row in enumerate(rows, start=1):
            question = by_id[row.question_id]
            if row.answer is None:
                if row.excluded is None:
                    unrecorded += 1
                else:
                    excluded_kept += 1
                _preserve_excluded(acc, question, row)
                continue
            if row.excluded is not None:
                # The answer existed; only the original judge call failed.
                # A stored answer is a stored answer — it gets its verdict now.
                revived += 1

            current_judge = get_provider("benchmark").provider_model
            if judge_model_id is None:
                judge_model_id = current_judge
            elif current_judge != judge_model_id:
                raise SameModelViolation(
                    f"Judge-model mismatch mid-run: llm.benchmark resolved to "
                    f"{current_judge!r} but the re-judge is pinned to {judge_model_id!r}. "
                    f"One judge per table is what makes its accuracies comparable "
                    f"; refusing to continue."
                )
            verdict = await _scored_call(
                "benchmark",
                judge_prompt(question, row.answer),
                max_tokens=_JUDGE_MAX_TOKENS,
                retries=retries,
                backoff_seconds=backoff,
            )
            judged += 1
            if verdict.text is None:
                notes.append(_excluded_note(question, metrics.condition, "judge", verdict))
                acc.record_excluded(
                    question,
                    verdict.excluded,
                    answer=row.answer,
                    context_particle_ids=row.context_particle_ids,
                )
            else:
                acc.record(
                    question,
                    parse_judge_verdict(verdict.text),
                    answer=row.answer,
                    verdict=verdict.text if record_text else None,
                    context_particle_ids=row.context_particle_ids,
                )
            if progress is not None and (index % _PROGRESS_EVERY == 0 or index == len(rows)):
                progress(f"[{metrics.condition}] re-judged {index}/{len(rows)} row(s)")
        rescored[metrics.condition] = acc.to_metrics(metrics.model_id)

    # ``judge_model_id`` is set iff at least one judge call was made, and
    # stored_answer_count > 0 guarantees that.
    assert judge_model_id is not None  # noqa: S101 — invariant, not validation

    condition_names = ", ".join(m.condition for m in conditions)
    provenance = (
        f"Re-judged from {source_path}: {judged} stored answer(s) across "
        f"{condition_names} were re-scored under judge protocol {protocol} by "
        f"{judge_model_id}; the source report was judged under protocol "
        f"{source.selection.judge_protocol} by "
        f"{source.selection.judge_model_id or 'not recorded'}. The answer model "
        f"({source.selection.answer_model_id or 'not recorded'}) and the retrieval "
        f"stage are unchanged; no answer call was made."
    )
    if excluded_kept or unrecorded:
        provenance += (
            f" {excluded_kept + unrecorded} row(s) had no stored answer and stay "
            f"excluded ({excluded_kept} excluded at answer time, {unrecorded} with "
            f"the answer text unrecorded)."
        )
    if revived:
        provenance += (
            f" {revived} row(s) whose original judge call produced no verdict "
            f"had a stored answer and were judged this time."
        )
    quality_notes = [provenance, *notes]
    if source.quality_notes:
        quality_notes.append(
            "Source report quality notes follow, retained verbatim (they describe "
            "the answer-time run; any judge exclusion they name was re-tried above):"
        )
        quality_notes.extend(source.quality_notes)

    def _slot(slot: str) -> QaConditionMetrics | None:
        metrics = getattr(source, slot)
        return None if metrics is None else rescored[metrics.condition]

    return MemoryBenchmarkReport(
        benchmark=source.benchmark,
        selection=source.selection.model_copy(
            update={"judge_protocol": protocol, "judge_model_id": judge_model_id}
        ),
        retrieval_stage=source.retrieval_stage.model_copy(deep=True),
        qa_particles=_slot("qa_particles"),
        qa_full_context=_slot("qa_full_context"),
        qa_no_memory=_slot("qa_no_memory"),
        quality_notes=quality_notes,
        generated_at=datetime.now(UTC),
    )
