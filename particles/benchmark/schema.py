"""BenchmarkSuite datatypes (techspec §13.3 frozen schema).

These are the **normative** types for benchmark suite YAML files. Field
names are taken verbatim from techspec §13.3 — do not rename without a
new techspec revision and a coordinated schema-version bump.

The :class:`BenchmarkCase` ``source_snapshot`` field is the techspec
default: an inline ``Snapshot`` describing the input. This repository
*also* supports a non-normative convenience form — ``fixture: <id>`` —
that references an existing fixture under ``tests/conformance/fixtures/``
by name. The loader resolves either form into a concrete
(snapshot, content_bytes) tuple before invoking the runner; nothing
outside :mod:`particles.benchmark.loader` sees the difference. The
``fixture`` form is what the bundled seed suite uses to avoid
duplicating the conformance corpus's input bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from particles.core.schema import Snapshot, UncertaintyNature


@dataclass(frozen=True)
class ExpectedParticle:
    """One gold-standard particle the extractor is expected to emit.

    ``confidence_min`` is the *minimum stated confidence* an emitted
    match should carry. An emitted particle whose content matches but
    whose stated confidence is below ``confidence_min`` is *demoted to
    an under-confidence partial match* — neither precision nor recall
    credit, but separately reported.

    ``required=True`` makes this particle count toward the recall
    denominator; optional particles influence precision (matching is a
    boon) but their absence is not a recall miss.
    """

    content: str
    confidence_min: float
    uncertainty_nature: UncertaintyNature
    required: bool = True


@dataclass(frozen=True)
class BenchmarkCase:
    """One input → expected-output pair within a suite.

    Exactly one of ``source_snapshot`` / ``fixture`` must be set. The
    loader translates ``fixture`` references against the conformance
    fixture corpus into the snapshot + content_bytes the runner needs.
    """

    case_id: str
    expected: list[ExpectedParticle]
    # Inline snapshot — techspec §13.3 default form
    source_snapshot: Snapshot | None = None
    inline_content: bytes | None = None
    # Convenience reference to a fixture in tests/conformance/fixtures/
    fixture: str | None = None


@dataclass(frozen=True)
class RequiredMetric:
    """A metric the suite says every runner must report.

    The reference runner always computes ``precision``, ``recall``, and
    ``calibration_error`` regardless of what ``metrics`` declares —
    those three are mandated by techspec §13.3. The list exists so a
    suite author can document additional, domain-specific metrics
    expected of downstream runners (the reference runner skips them
    with a quality note rather than erroring out).
    """

    name: str
    definition: str


@dataclass(frozen=True)
class BenchmarkSuite:
    """A named, versioned collection of benchmark cases (§13.3)."""

    suite_id: str
    name: str
    version: str
    domain: str
    source_types: list[str]
    cases: list[BenchmarkCase]
    metrics: list[RequiredMetric] = field(default_factory=list)
    published_by: str = ""
    published_at: datetime | None = None
