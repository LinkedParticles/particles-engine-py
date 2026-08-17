"""General-extractor claim-polarity benchmark.

A third, **additive** measurement axis alongside the techspec §13.3 content
benchmark (``particles.benchmark`` proper) and the modality sibling
(``particles.benchmark.modality``). Where the content harness measures
*correctness* — did the extractor emit the right facts, at the right
confidence — this harness measures **claim-polarity classification quality**:
given a source passage, did the extractor correctly classify how the document
*frames* each proposition — ``ASSERTED`` (a held decision / claim), ``DECLINED``
(rejected / superseded / deferred / out-of-scope), or ``HYPOTHETICAL``
(counterfactual / motivational / worked example)?

It exists because cap. 1 keeps ``DECLINED`` / ``HYPOTHETICAL`` claims
off the default factual surface — query / projection / export. That makes one
error specifically dangerous: a real current decision (``ASSERTED``) wrongly
classified ``DECLINED`` is **silently hidden** — the precision risk that bears
directly on README-projection trust. This suite measures that
**wrong-``DECLINED`` rate** head-on (the headline), plus its superset the
wrong-hidden rate and per-polarity precision/recall.

The techspec §13.3 ``BenchmarkSuite`` / ``ExpectedParticle`` schema is frozen
and carries no polarity field, so this harness is deliberately *parallel*: its
own suite schema (:mod:`.schema`), loader (:mod:`.loader`), pure metrics
(:mod:`.metrics`), and runner (:mod:`.runner`). It reuses the parent package's
embedding equivalence judge (:mod:`particles.benchmark.equivalence`) to align
emitted claims to gold labels, and is **report-only** like its siblings.
"""

from __future__ import annotations

from particles.benchmark.polarity.loader import (
    PolaritySuiteLoadError,
    discover_polarity_suites,
    load_polarity_suite,
)
from particles.benchmark.polarity.metrics import (
    confusion_counts,
    polarity_precision,
    polarity_recall,
    wrong_declined_rate,
    wrong_hidden_rate,
)
from particles.benchmark.polarity.runner import (
    PolarityCaseResult,
    PolarityConfusionCell,
    PolarityReport,
    polarity_of,
    run_polarity_benchmark,
)
from particles.benchmark.polarity.schema import (
    ClaimPolarity,
    PolarityCase,
    PolarityLabel,
    PolaritySuite,
)

__all__ = [
    "ClaimPolarity",
    "PolarityCase",
    "PolarityCaseResult",
    "PolarityConfusionCell",
    "PolarityLabel",
    "PolarityReport",
    "PolaritySuite",
    "PolaritySuiteLoadError",
    "confusion_counts",
    "discover_polarity_suites",
    "load_polarity_suite",
    "polarity_of",
    "polarity_precision",
    "polarity_recall",
    "run_polarity_benchmark",
    "wrong_declined_rate",
    "wrong_hidden_rate",
]
