# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Model-prior leakage benchmark — unsupported sentences in query answers.

The query op's response step is the engine's one inference over retrieved
knowledge, and the composing model's parametric background can blend into an
answer with no record. Knowledge-boundary recognition and the
coverage gap cover the case where the store is silent; this
measures the blended case: each answer sentence judged against the particles
it was composed from by the shared entailment judge, on a
different model from the composer. A grounded run judges each cited
sentence against the ids it cites and reports the composer's own labelled
inference and background beside the silent unsupported count.
"""

from .metrics import (
    aggregate,
    answers_with_unsupported,
    labelled_rate,
    mean_answer_fraction,
    refusal_rate,
    sentence_counts,
    split_sentences,
    unsupported_rate_by_source,
    unsupported_sentence_rate,
)
from .render import render_report
from .runner import (
    LeakageError,
    LeakageEstimate,
    build_report,
    build_selection,
    check_distinct_judge,
    composed_premises,
    estimate_run,
    grounded_units,
    judge_prompt,
    measure_question,
    render_estimate,
    resolved_models,
    rubric,
    run_leakage,
)
from .schema import (
    LeakageReport,
    QueryLeakage,
    RunSelection,
    SentenceCounts,
    SentenceResult,
    SentenceVerdict,
)

__all__ = [
    "LeakageError",
    "LeakageEstimate",
    "LeakageReport",
    "QueryLeakage",
    "RunSelection",
    "SentenceCounts",
    "SentenceResult",
    "SentenceVerdict",
    "aggregate",
    "answers_with_unsupported",
    "build_report",
    "build_selection",
    "check_distinct_judge",
    "composed_premises",
    "estimate_run",
    "grounded_units",
    "judge_prompt",
    "labelled_rate",
    "mean_answer_fraction",
    "measure_question",
    "refusal_rate",
    "render_estimate",
    "render_report",
    "resolved_models",
    "rubric",
    "run_leakage",
    "sentence_counts",
    "split_sentences",
    "unsupported_rate_by_source",
    "unsupported_sentence_rate",
]
