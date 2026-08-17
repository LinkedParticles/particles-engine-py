"""Agent-memory benchmark evaluation — LongMemEval first.

The fourth measurement package under ``particles.benchmark`` and the third
*sibling* harness beside ``modality/`` and ``polarity/``
 — but with a different system under test. The content / modality /
polarity harnesses measure one **extractor's** output against gold particles;
this one measures the **whole pipeline** (deposit → extract → reconcile →
query) against gold *answers* on LongMemEval (Wu et al., ICLR 2025):
timestamped multi-session chat histories with labeled evidence sessions —
the same material shape the harvester deposits.

Every run reports four conditions in two explicitly labeled measurement
families:

* **retrieval-stage** — (i) evidence-session Recall@k / Precision@k of the
  store's top-k query result, scored through real provenance chains.
  Zero-evidence abstention questions are unscoreable here and are excluded
  from every retrieval aggregate with a disclosed count (correction
  v1.74.2) — they stay in the QA family, which the dataset's protocol
  scores;
* **end-to-end QA** — (ii) ``qa_particles``, (iii) ``qa_full_context`` (the
  baseline that must not be buried: the same model over the whole haystack),
  and (iv) ``qa_no_memory`` (the parametric floor), one pinned answer model
  across all three, enforced by a runner refusal. The judge and extraction
  models are pinned the same way, and the run tuple records the resolved
  extraction + embedding model ids (correction v1.74.2).

The techspec §13.3 ``BenchmarkSuite`` / ``ExpectedParticle`` schema is frozen
and has no answer / judge / haystack concepts, so this harness is
deliberately parallel: its own schema (:mod:`.schema`), loader
(:mod:`.loader`), pure metrics (:mod:`.metrics`), and runner
(:mod:`.runner`). **Report-only** like its siblings — per-question ephemeral
scratch stores mean a run never touches a user store. CLI verb:
``particles benchmark memory`` (its own sub-Typer group: the SUT is the
pipeline, not an extractor, so it does not live under ``particles
extractor``).
"""

from __future__ import annotations

from particles.benchmark.memory.loader import (
    MemoryDatasetLoadError,
    ensure_dataset,
    load_dataset_file,
    parse_question,
    select_questions,
)
from particles.benchmark.memory.metrics import (
    accuracy_by_type,
    parse_judge_verdict,
    precision_at_k,
    qa_accuracy,
    recall_at_k,
)
from particles.benchmark.memory.runner import (
    CachingExtractor,
    MemoryRunEstimate,
    SameModelViolation,
    estimate_run,
    render_estimate,
    run_memory_benchmark,
    session_id_from_uri,
    session_uri,
)
from particles.benchmark.memory.schema import (
    QA_CONDITIONS,
    QUESTION_TYPES,
    MemoryBenchmarkReport,
    MemoryQuestion,
    MemorySession,
    MemoryTurn,
    QaConditionMetrics,
    RetrievalStageMetrics,
    RunSelection,
    render_report_table,
)

__all__ = [
    "QA_CONDITIONS",
    "QUESTION_TYPES",
    "CachingExtractor",
    "MemoryBenchmarkReport",
    "MemoryDatasetLoadError",
    "MemoryQuestion",
    "MemoryRunEstimate",
    "MemorySession",
    "MemoryTurn",
    "QaConditionMetrics",
    "RetrievalStageMetrics",
    "RunSelection",
    "SameModelViolation",
    "accuracy_by_type",
    "ensure_dataset",
    "estimate_run",
    "load_dataset_file",
    "parse_judge_verdict",
    "parse_question",
    "precision_at_k",
    "qa_accuracy",
    "recall_at_k",
    "render_estimate",
    "render_report_table",
    "run_memory_benchmark",
    "select_questions",
    "session_id_from_uri",
    "session_uri",
]
