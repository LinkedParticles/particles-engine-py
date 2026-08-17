"""Journal-extractor modality benchmark.

A second, **additive** measurement axis alongside the techspec §13.3 content
benchmark (``particles.benchmark`` proper). Where the content harness measures
*correctness* — did the extractor emit the right facts, at the right
confidence — this harness measures **modality-classification quality**: given a
journal entry, did the extractor assign the right ``assertion_modality`` to
each claim?

It exists because the journal extractor inverts the usual modality
default — *"when unsure, prefer non-``FALSIFIABLE``"* — to keep feelings and
opinions out of the truth engine. That inversion raises one specifically
dangerous error: a genuinely falsifiable world-fact (a date, a count, a dose)
mis-tagged ``EVALUATIVE`` / ``EXPERIENTIAL`` and thereby exempted from
contradiction-arbitration. This suite measures that **false-non-``FALSIFIABLE``
rate** head-on, plus per-modality precision/recall and the whole-entry
**narrative-emission rate**.

The techspec §13.3 ``BenchmarkSuite`` / ``ExpectedParticle`` schema is frozen
and carries no modality field, so this harness is deliberately *parallel*: its
own suite schema (:mod:`.schema`), loader (:mod:`.loader`), pure metrics
(:mod:`.metrics`), and runner (:mod:`.runner`). It reuses the parent package's
embedding equivalence judge (:mod:`particles.benchmark.equivalence`) to align
emitted claims to gold labels, and is **report-only** like its sibling.
"""

from __future__ import annotations

from particles.benchmark.modality.loader import (
    ModalitySuiteLoadError,
    discover_modality_suites,
    load_modality_suite,
)
from particles.benchmark.modality.metrics import (
    confusion_counts,
    false_non_falsifiable_rate,
    modality_precision,
    modality_recall,
    narrative_emission_rate,
)
from particles.benchmark.modality.runner import (
    ModalityCaseResult,
    ModalityConfusionCell,
    ModalityReport,
    run_modality_benchmark,
)
from particles.benchmark.modality.schema import (
    ModalityCase,
    ModalityLabel,
    ModalitySuite,
)

__all__ = [
    "ModalityCase",
    "ModalityCaseResult",
    "ModalityConfusionCell",
    "ModalityLabel",
    "ModalityReport",
    "ModalitySuite",
    "ModalitySuiteLoadError",
    "confusion_counts",
    "discover_modality_suites",
    "false_non_falsifiable_rate",
    "load_modality_suite",
    "modality_precision",
    "modality_recall",
    "narrative_emission_rate",
    "run_modality_benchmark",
]
