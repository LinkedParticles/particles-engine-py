"""Polarity-benchmark suite datatypes.

These are **not** the techspec §13.3 schema — that one
(:mod:`particles.benchmark.schema`) is frozen and carries no polarity field.
This is a parallel, SDK-local suite shape whose gold standard is a list of
*claim text → expected claim-polarity* labels per source passage, mirroring the
modality sibling harness (:mod:`particles.benchmark.modality.schema`). A polarity suite targets the general extractor (cap. 1),
which accepts any ``source_type``, so it declares the single ``source_type`` the
runner feeds to ``extract``.

The polarity vocabulary is the cap. 1 contract, defined once in
:mod:`particles.extraction.polarity` and re-used here so the gold labels can
never drift from the producer's value strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from particles.extraction.polarity import (
    POLARITY_ASSERTED,
    POLARITY_DECLINED,
    POLARITY_HYPOTHETICAL,
)


class ClaimPolarity(StrEnum):
    """The three claim-polarity classes (cap. 1).

    Values are bound to the canonical strings in
    :mod:`particles.extraction.polarity` so a benchmark label and an emitted
    particle's ``properties["extraction:polarity"]`` are compared on identical
    tokens —
    ``ASSERTED`` is also the meaning of the key's absence.
    """

    ASSERTED = POLARITY_ASSERTED
    DECLINED = POLARITY_DECLINED
    HYPOTHETICAL = POLARITY_HYPOTHETICAL


@dataclass(frozen=True)
class PolarityLabel:
    """One gold-standard ``(claim text, expected polarity)`` pair.

    ``content`` is authored close to the general extractor's declarative output
    so the embedding judge can align it to an emitted claim. ``polarity`` is the
    class a correct extractor should assign by how the source *frames* the
    proposition — never whether it is true (a rejected design may be truly
    rejected).
    """

    content: str
    polarity: ClaimPolarity


@dataclass(frozen=True)
class PolarityCase:
    """One source passage plus its labeled claims (the runner's unit of work).

    ``document`` is the raw prose fed verbatim to the extractor — an ADR /
    spec-style passage that mixes asserted decisions with rejected /
    superseded / deferred (``DECLINED``) and counterfactual / motivational
    (``HYPOTHETICAL``) prose.
    """

    case_id: str
    document: str
    labels: list[PolarityLabel]


@dataclass(frozen=True)
class PolaritySuite:
    """A named, versioned set of polarity cases for one extractor."""

    suite_id: str
    name: str
    version: str
    domain: str
    source_type: str
    cases: list[PolarityCase]
    published_by: str = ""
    published_at: datetime | None = None
    metrics: list[str] = field(default_factory=list)
