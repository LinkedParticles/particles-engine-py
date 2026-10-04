# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The vocabulary report's gather half.

Gather / decide: this module reads the plain rows (the classes of the subjects
the selected claims are about, every Subject's class and external links, the
operator log's counts by type) and hands them to the pure fold in
:mod:`particles.core.vocabulary_report`. Nothing is written, and the report is never
stored.

The claim candidate set is chosen by the caller, the structural query's
candidate selection, which applies the default exclusions and the project
observer like every other read that selects beliefs by something
other than their id.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ExternalRef, StructuredClaim, VocabularyReport
from particles.core.vocabulary_report import (
    NO_SUBJECT,
    UNCLASSED,
    alignments_from_documents,
    build_vocabulary_report,
)
from particles.store.event_store import OperatorEventType, count_events_by_type
from particles.store.particle_store import count_structured_claim_coverage
from particles.store.subject_store import get_subject_classes, list_subject_classes_and_links
from particles.store.vocabulary_store import get_adopted_documents


async def load_predicate_alignments(
    session: AsyncSession,
) -> tuple[dict[str, list[str]], str | None]:
    """What the adopted vocabulary documents align each canonical predicate to.

    Keyed by canonical form: the term a document mints for the form or
    confirms as an alias, then that term's outward alignments. The source
    names the adopted documents and their versions. A store that adopts none
    gets an empty map and no source, and the report shows every predicate
    unaligned.
    """
    return alignments_from_documents(await get_adopted_documents(session))


__all__ = [
    "MODELLING_DECISION_TYPES",
    "VocabularyInputs",
    "classed_claims",
    "gather_vocabulary_inputs",
    "vocabulary_report",
]

#: The operator events that record a modelling decision someone confirmed:
#: two claims are one (or not), a contradiction is settled, two subjects are
#: one (or two), a name or a class or an external link is right (or wrong).
MODELLING_DECISION_TYPES: tuple[str, ...] = tuple(
    t.value
    for t in (
        OperatorEventType.DUPLICATES_MERGED,
        OperatorEventType.DUPLICATES_UNMERGED,
        OperatorEventType.INCONSISTENCY_CLOSED,
        OperatorEventType.REVIEW_RESOLVED,
        OperatorEventType.SUBJECTS_MERGED,
        OperatorEventType.SUBJECTS_SPLIT,
        OperatorEventType.SUBJECT_ALIASED,
        OperatorEventType.SUBJECT_RECLASSIFIED,
        OperatorEventType.SUBJECT_LINK_CONFIRMED,
        OperatorEventType.SUBJECT_LINK_REMOVED,
    )
)


@dataclass
class VocabularyInputs:
    """One store's plain rows for the report; federation concatenates them."""

    claims: list[tuple[StructuredClaim, str]] = field(default_factory=list)
    subjects: list[tuple[str | None, list[ExternalRef]]] = field(default_factory=list)
    event_counts: Counter[str] = field(default_factory=Counter)
    structured_claims_total: int = 0

    def extend(self, other: VocabularyInputs) -> None:
        self.claims.extend(other.claims)
        self.subjects.extend(other.subjects)
        self.event_counts.update(other.event_counts)
        self.structured_claims_total += other.structured_claims_total


async def classed_claims(
    session: AsyncSession, claims: Sequence[StructuredClaim]
) -> list[tuple[StructuredClaim, str]]:
    """Each claim with the class label of the subject it is about.

    The subject a triple is about is its own ``subject_id``, the key predicate
    profiles use, never a particle-level link: a multi-subject
    particle is an edge, and its other subjects are not what the predicate
    attaches to.
    """
    classes = await get_subject_classes(session, {c.subject_id for c in claims if c.subject_id})
    rows: list[tuple[StructuredClaim, str]] = []
    for claim in claims:
        if claim.subject_id is None or claim.subject_id not in classes:
            rows.append((claim, NO_SUBJECT))
        else:
            rows.append((claim, classes[claim.subject_id] or UNCLASSED))
    return rows


async def gather_vocabulary_inputs(
    session: AsyncSession,
    claims: Sequence[StructuredClaim],
    *,
    as_of: datetime | None = None,
) -> VocabularyInputs:
    """Read one store's rows for the report over an already-selected claim set.

    ``as_of`` cuts the header at that instant: the Subjects created
    by then, as they stand now, and the decisions recorded by then. The claim
    set was already cut by the caller's as-of view.
    """
    coverage = await count_structured_claim_coverage(session)
    return VocabularyInputs(
        claims=await classed_claims(session, claims),
        subjects=await list_subject_classes_and_links(session, created_by=as_of),
        event_counts=Counter(await count_events_by_type(session, until=as_of)),
        structured_claims_total=int(coverage["annotated"]),
    )


async def vocabulary_report(
    session: AsyncSession, inputs: VocabularyInputs, *, as_of: datetime | None = None
) -> VocabularyReport:
    """Fold gathered rows into the report, with the alignments in force.

    ``session`` is the viewer's: under federation the viewer's adopted
    vocabulary document aligns every store's predicates, as the viewer's lens
    ranks every store's beliefs.
    """
    alignments, source = await load_predicate_alignments(session)
    return build_vocabulary_report(
        inputs.claims,
        inputs.subjects,
        structured_claims_total=inputs.structured_claims_total,
        event_counts=inputs.event_counts,
        decision_types=MODELLING_DECISION_TYPES,
        suppress_threshold=get_config().subjects.wikidata_link_suppress_threshold,
        alignments=alignments,
        alignment_source=source,
        as_of=as_of,
    )
