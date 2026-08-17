"""Modality-benchmark suite datatypes.

These are **not** the techspec §13.3 schema — that one
(:mod:`particles.benchmark.schema`) is frozen and carries no modality field.
This is a parallel, SDK-local suite shape whose gold standard is a list of
*reified claim → expected ``assertion_modality``* labels per journal entry. A
modality suite targets exactly one genre extractor, so it declares a single
``source_type`` (not the §13.3 list) — the runner feeds it to ``extract`` and
checks ``accepts(source_type)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from particles.core.schema import AssertionModality


@dataclass(frozen=True)
class ModalityLabel:
    """One gold-standard ``(reified claim text, expected modality)`` pair.

    ``content`` is authored close to the journal extractor's *reified*,
    third-person output (e.g. *"The author felt anxious"*, not *"I felt
    anxious"*) so the embedding judge can align it to an emitted claim.
    ``modality`` is the modality a correct extractor should assign.
    """

    content: str
    modality: AssertionModality


@dataclass(frozen=True)
class ModalityCase:
    """One journal entry plus its labeled claims (the runner's unit of work).

    ``entry`` is the raw first-person prose fed verbatim to the extractor.
    ``narrative_expected`` records whether a whole-entry ``NARRATIVE`` should
    be emitted — the denominator of the narrative-emission rate.
    """

    case_id: str
    entry: str
    labels: list[ModalityLabel]
    narrative_expected: bool = True


@dataclass(frozen=True)
class ModalitySuite:
    """A named, versioned set of modality cases for one genre extractor."""

    suite_id: str
    name: str
    version: str
    domain: str
    source_type: str
    cases: list[ModalityCase]
    published_by: str = ""
    published_at: datetime | None = None
    metrics: list[str] = field(default_factory=list)
