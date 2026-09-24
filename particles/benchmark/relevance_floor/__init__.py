# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Relevance-floor benchmark — the gate's error rates on real questions.

The memory-rot harness already sweeps the floor over synthetic probes.
This package is the other half: a held-out set harvested from
an operator's own agent transcripts, replayed through the query op, with the
response step optionally run over the floor-suppressed top-k and judged.
"""

from .harvest import (
    REFUSAL_PHRASE,
    HarvestResult,
    harvest_transcripts,
    heldout_fingerprint,
    load_heldout,
    question_id,
    write_heldout,
)
from .metrics import floor_sweep, refusal_curve, sample_questions, similarity_quantiles
from .render import render_report
from .runner import (
    JudgedStageEstimate,
    RelevanceFloorError,
    build_report,
    build_selection,
    estimate_judged_stage,
    floor_disabled,
    judge_prompt,
    render_estimate,
    replay_retrieval,
    reuse_replay,
    run_judged,
)
from .schema import (
    FloorRow,
    HeldOutQuestion,
    QuestionResult,
    QuestionSource,
    RefusalRow,
    RelevanceFloorReport,
    RunSelection,
)

__all__ = [
    "REFUSAL_PHRASE",
    "FloorRow",
    "HarvestResult",
    "HeldOutQuestion",
    "JudgedStageEstimate",
    "QuestionResult",
    "QuestionSource",
    "RefusalRow",
    "RelevanceFloorError",
    "RelevanceFloorReport",
    "RunSelection",
    "build_report",
    "build_selection",
    "estimate_judged_stage",
    "floor_disabled",
    "floor_sweep",
    "harvest_transcripts",
    "heldout_fingerprint",
    "judge_prompt",
    "load_heldout",
    "question_id",
    "refusal_curve",
    "render_estimate",
    "render_report",
    "replay_retrieval",
    "reuse_replay",
    "run_judged",
    "sample_questions",
    "similarity_quantiles",
    "write_heldout",
]
