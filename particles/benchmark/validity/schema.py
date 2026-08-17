"""Validity-benchmark suite datatypes.

These are **not** the techspec §13.3 schema — that one
(:mod:`particles.benchmark.schema`) is frozen and carries no validity field.
This is a parallel, SDK-local suite shape whose gold standard is, per source
passage, a list of *claim text → expected validity boundary* labels, mirroring
the modality (:mod:`particles.benchmark.modality.schema`) and polarity
(:mod:`particles.benchmark.polarity.schema`) sibling harnesses.

A validity suite targets the general extractor, which accepts any
``source_type``, so it declares the single ``source_type`` the runner feeds to
``extract``. Each case also carries a ``reference_date`` — the anchor the runner
stamps onto the snapshot's ``content_published_at`` so the extractor resolves
relative boundaries ("the exam is tomorrow") against a fixed, reproducible
instant instead of the wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class ValidityLabel:
    """One gold-standard ``(claim text, expected validity boundary)`` pair.

    ``content`` is authored close to the general extractor's declarative output
    so the embedding judge can align it to an emitted claim. A label is exactly
    one of two kinds, and ``is_durable`` must agree with ``expected_valid_until``
    (the loader enforces it):

    * **durable** — ``is_durable=True`` and ``expected_valid_until=None``: the
      claim must carry **no** boundary (a durable fact, possibly one that merely
      *mentions* a date). These are the danger-probe decoys the headline
      :func:`~particles.benchmark.validity.metrics.wrong_expiry_rate` is computed
      over — the over-eager-expiry failure mode.
    * **date-bounded** — ``is_durable=False`` and ``expected_valid_until`` set:
      the claim genuinely stops being true at that date, so a correct extractor
      emits a ``valid_until`` near it.
    """

    content: str
    expected_valid_until: date | None
    is_durable: bool


@dataclass(frozen=True)
class ValidityCase:
    """One source passage plus its labeled claims (the runner's unit of work).

    ``document`` is the raw prose fed verbatim to the extractor — a passage that
    mixes genuinely date-bounded claims with durable facts that mention dates.
    ``reference_date`` is the anchor for relative boundaries: the runner stamps
    it onto the snapshot's ``content_published_at`` so "tomorrow" / "next week"
    resolve deterministically (and an archival relative cue resolves into the
    past → dropped as born-expired), independent of when the suite is run.
    """

    case_id: str
    document: str
    reference_date: date
    labels: list[ValidityLabel]

    def reference_datetime(self) -> datetime:
        """The reference date as a UTC-midnight datetime for the snapshot anchor."""
        from datetime import UTC

        return datetime(
            self.reference_date.year,
            self.reference_date.month,
            self.reference_date.day,
            tzinfo=UTC,
        )


@dataclass(frozen=True)
class ValiditySuite:
    """A named, versioned set of validity cases for one extractor."""

    suite_id: str
    name: str
    version: str
    domain: str
    source_type: str
    cases: list[ValidityCase]
    # Tolerance (in days) within which an emitted boundary counts as date-correct
    # for date_accuracy. Defaults loose enough to absorb "end of month" vs
    # "last day of month" granularity; a suite may tighten it.
    date_tolerance_days: int = 3
    published_by: str = ""
    published_at: datetime | None = None
    metrics: list[str] = field(default_factory=list)
