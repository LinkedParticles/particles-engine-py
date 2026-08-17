"""General-extractor event-anchored-validity benchmark.

A fourth, **additive** measurement axis alongside the techspec §13.3 content
benchmark (``particles.benchmark`` proper) and the modality
(``particles.benchmark.modality``) / polarity (``particles.benchmark.polarity``)
siblings. Where the content harness measures *correctness* — did the extractor
emit the right facts, at the right confidence — this harness measures
**event-anchored-validity quality**: given a source passage, did the general
extractor put a ``valid_until`` boundary on the claims that genuinely stop being
true at a date, and — the load-bearing question — did it **avoid** putting one on
durable facts that merely mention a date?

It exists because a ``valid_until`` is the one extraction signal that can flip a
particle out of ACTIVE (the §9.3 staleness lint retires it as
``VALIDITY_EXPIRED``). That makes one error specifically dangerous: a durable
fact wrongly assigned a boundary is **silently retired** — the over-eager-expiry
failure mode. This suite measures that **wrong-expiry rate** head-on (the
headline), plus existence precision/recall and date accuracy.

The techspec §13.3 ``BenchmarkSuite`` / ``ExpectedParticle`` schema is frozen and
carries no validity field, so this harness is deliberately *parallel*: its own
suite schema (:mod:`.schema`), loader (:mod:`.loader`), pure metrics
(:mod:`.metrics`), and runner (:mod:`.runner`). It reuses the parent package's
embedding equivalence judge (:mod:`particles.benchmark.equivalence`) to align
emitted claims to gold labels, and is **report-only** like its siblings.
"""

from __future__ import annotations

from particles.benchmark.validity.loader import (
    ValiditySuiteLoadError,
    discover_validity_suites,
    load_validity_suite,
)
from particles.benchmark.validity.metrics import (
    ValidityPair,
    date_accuracy,
    expiry_precision,
    expiry_recall,
    support_counts,
    wrong_expiry_rate,
)
from particles.benchmark.validity.runner import (
    ValidityCaseResult,
    ValidityReport,
    run_validity_benchmark,
    valid_until_date,
)
from particles.benchmark.validity.schema import (
    ValidityCase,
    ValidityLabel,
    ValiditySuite,
)

__all__ = [
    "ValidityCase",
    "ValidityCaseResult",
    "ValidityLabel",
    "ValidityPair",
    "ValidityReport",
    "ValiditySuite",
    "ValiditySuiteLoadError",
    "date_accuracy",
    "discover_validity_suites",
    "expiry_precision",
    "expiry_recall",
    "load_validity_suite",
    "run_validity_benchmark",
    "support_counts",
    "valid_until_date",
    "wrong_expiry_rate",
]
