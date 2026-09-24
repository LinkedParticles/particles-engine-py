# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Same-subject update supersession — candidacy and the rung 2.5 inputs.

Two things the §6.6 ladder cannot do on its own:

* **Find the pair.** Extraction's own candidate search is per corpus entry and
  gated at ``extraction.similarity_threshold`` (0.80), so "the user lives in
  Denver" from today's session never meets "the user lives in Boston" from last
  week's: they are in different entries, and they score ~0.70 anyway. This
  module adds a second search, keyed on the claim's *about-subject*, across all
  entries, at a subject-scoped floor (:class:`SubjectIndex`,
  :func:`about_subject_name`).
* **Order the pair.** When the probe confirms the contradiction and trust
  cannot tell the sides apart, rung 2.5 prefers the strictly newer claim.
  :func:`update_order` decides whether the pair qualifies and in which
  direction; :func:`particles.core.conflict_resolution.resolve_conflict` applies
  it.

The about-subject is the entity a claim is *about*: the triple's
subject when the claim has one, else its sole subject. A particle's
``subject_ids`` also carry the entities it merely mentions ("moved from Boston
to Denver" names both cities), which must never become keys.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import numpy as np

from particles.core.schema import (
    CorpusEntry,
    Particle,
    ProvenanceRefType,
    is_truth_apt,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.stance import stance_holder
from particles.embeddings import cosine_similarity
from particles.extraction.general import CandidateParticle
from particles.extraction.polarity import is_non_asserted
from particles.extraction.scope import is_excluded_document_meta

Embedding = np.ndarray[Any, np.dtype[np.float32]]

#: The asserter-identity prefix of the agent write surface. A claim an
#: agent asserted is never retired by, nor retires, an extracted claim on the
#: strength of a timestamp (condition 4).
MCP_ASSERTER_PREFIX = "mcp:"


def candidate_subject_names(candidate: CandidateParticle) -> list[str]:
    """The subject name a candidate is paired on — the *about*-subject, or none.

    The triple's subject when it names one of the candidate's subjects (the
    same case-insensitive match :func:`particles.extraction.structure.bind_subject_id`
    applies after resolution), else the candidate's sole subject.

    Candidacy spanned **every** subject a candidate named for one release.
    Its own live run withdrew it, and the correction there
    records the measurement: on a kept store the wider key produced 167
    qualifying pairs, spent the whole probe cap on them and demoted 3, because
    a restatement of the *same* value outranks a genuine old→new pair on
    cosine (0.99 against 0.6–0.8) and the probe answers NO to nearly all of
    them. The list shape is kept — it is the plural the callers were rebuilt
    around, and a re-ranked wider key is the open question — but it
    now carries at most one name.
    """
    names = [n.strip() for n in candidate.subjects if n.strip()]
    if candidate.structured_claim is not None:
        wanted = candidate.structured_claim.subject.value.strip().casefold()
        about = next((n for n in names if n.casefold() == wanted), None)
        if about is not None:
            return [about]
    return names if len(names) == 1 else []


def particle_subject_ids(particle: Particle) -> list[str]:
    """The subject id a stored particle is paired on — its about-subject, or none."""
    if particle.structured_claim is not None and particle.structured_claim.subject_id:
        return [particle.structured_claim.subject_id]
    return list(particle.subject_ids) if len(particle.subject_ids) == 1 else []


def is_reconcilable(particle: Particle) -> bool:
    """Whether a stored particle may be a subject-scoped §6.6 candidate at all.

    The same exclusions the same-entry search applies: truth-apt only,
    no stances (a stance only ever pairs with a same-holder
    stance, which the same-entry search handles), and no DOCUMENT_META or
    non-asserted claims.
    """
    return (
        is_truth_apt(particle)
        and stance_holder(particle) is None
        and not is_excluded_document_meta(particle.properties)
        and not is_non_asserted(particle.properties)
    )


def is_extractor_asserted(particle: Particle) -> bool:
    """True for a machine-extracted claim — neither operator- nor agent-asserted.

    Operator beliefs carry ``HUMAN_REVIEW`` calibration (the guard
    reads the same field); agent writes carry ``AGENT_ASSERTED`` calibration and
    an ``mcp:`` asserter identity; an extracted claim has a SOURCE provenance ref.
    """
    if particle.confidence.calibration_source in (
        CalibrationSource.HUMAN_REVIEW,
        CalibrationSource.AGENT_ASSERTED,
    ):
        return False
    if particle.asserted_by.startswith(MCP_ASSERTER_PREFIX):
        return False
    return any(ref.type is ProvenanceRefType.SOURCE for ref in particle.provenance)


@dataclass
class SubjectIndex:
    """ACTIVE, reconcilable particles grouped by about-subject, with embeddings."""

    by_subject: dict[str, list[tuple[Particle, Embedding]]] = field(default_factory=dict)
    #: Particles retired earlier in the same pass — never offered again.
    retired: set[str] = field(default_factory=set)

    @classmethod
    def build(
        cls,
        pairs: Iterable[tuple[Particle, Embedding | None]],
        *,
        exclude_ids: Iterable[str] = (),
    ) -> SubjectIndex:
        """Index ``pairs`` (typically every ACTIVE particle with an embedding)."""
        excluded = set(exclude_ids)
        index = cls()
        for particle, emb in pairs:
            if emb is None or particle.id in excluded or not is_reconcilable(particle):
                continue
            for key in particle_subject_ids(particle):
                index.by_subject.setdefault(key, []).append((particle, emb))
        return index

    def candidates(
        self,
        subject_ids: Iterable[str],
        embedding: Embedding,
        *,
        floor: float,
        limit: int,
        skip_ids: Iterable[str] = (),
    ) -> list[Particle]:
        """Most similar particles sharing any of ``subject_ids``, best first."""
        skip = set(skip_ids) | self.retired
        seen: dict[str, float] = {}
        pool: dict[str, Particle] = {}
        for subject_id in subject_ids:
            for particle, emb in self.by_subject.get(subject_id, []):
                if particle.id in skip or particle.id in seen:
                    continue
                score = cosine_similarity(embedding, emb)
                if score >= floor:
                    seen[particle.id] = score
                    pool[particle.id] = particle
        best = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [pool[pid] for pid, _ in best]


def _snapshot_of(entry: CorpusEntry | None, snapshot_id: str | None) -> Any:
    if entry is None or not snapshot_id:
        return None
    return next((s for s in entry.snapshots if s.snapshot_id == snapshot_id), None)


def _authority(uri: str | None) -> str | None:
    if not uri:
        return None
    return urlparse(uri).netloc or None


def _contributors_key(entry: CorpusEntry) -> tuple[tuple[str, str], ...]:
    # Who and in what role — never *when*: the same contributor acting on two
    # dates is one lineage (ContributorRef.at defaults to the act's time).
    return tuple(sorted((c.id, c.role) for c in entry.contributors or []))


def _as_aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def own_assertion_order(new_particle: Particle, existing: Particle) -> int | None:
    """Rung 2.5 input for the assertion pathway.

    ``+1`` when both claims were asserted by the **same agent identity** and
    the new one is strictly later, else ``None``. An agent revising its own
    earlier belief is what ``particle_supersede`` already lets it do
    explicitly (own-beliefs-only); this is the automatic form and
    grants no further reach. Every other pair — against an extracted claim, an
    operator's, or another agent's — returns ``None`` and keeps failing closed.
    Ordered by ``asserted_at``: an assertion has no source content date.
    """
    agent = CalibrationSource.AGENT_ASSERTED
    if new_particle.confidence.calibration_source is not agent:
        return None
    if existing.confidence.calibration_source is not agent:
        return None
    if not new_particle.asserted_by or new_particle.asserted_by != existing.asserted_by:
        return None
    new_at, old_at = _as_aware(new_particle.asserted_at), _as_aware(existing.asserted_at)
    return 1 if new_at > old_at else None


async def latest_source_date(
    session: Any,
    particle: Particle,
    cache: dict[str, CorpusEntry | None] | None = None,
) -> datetime | None:
    """The date of ``particle``'s **latest** observation, across every SOURCE ref.

    A claim restated later keeps its original ref and *gains* another, in that
    later session's corpus entry (folds a re-observation into the
    particle it matches rather than minting a copy). Dating such a particle by
    its first ref makes a value that is current again look stale: measured on a
    rot world, where the sweep retired a reverted value because its
    first observation predated the value it had replaced. Every ref is
    consulted, each in its own entry, and the newest wins.
    """
    from particles.corpus.store import get_entry

    store = cache if cache is not None else {}
    dates: list[datetime] = []
    for ref in particle.provenance:
        if ref.type is not ProvenanceRefType.SOURCE:
            continue
        if ref.corpus_entry_id not in store:
            store[ref.corpus_entry_id] = await get_entry(session, ref.corpus_entry_id)
        entry = store[ref.corpus_entry_id]
        if entry is None:
            continue
        snap = next((s for s in entry.snapshots if s.snapshot_id == ref.snapshot_id), None)
        if snap is not None and snap.content_published_at is not None:
            dates.append(_as_aware(snap.content_published_at))
    return max(dates) if dates else None


def update_order(
    new_particle: Particle,
    new_entry: CorpusEntry | None,
    new_snapshot_id: str | None,
    existing: Particle,
    existing_entry: CorpusEntry | None,
    existing_snapshot_id: str | None,
    *,
    require_attribution: bool = False,
    new_date: datetime | None = None,
    existing_date: datetime | None = None,
) -> int | None:
    """The rung 2.5 input for one confirmed pair: ``+1``, ``-1``, or ``None``.

    ``+1`` when ``new`` is the strictly newer claim, ``-1`` when ``existing`` is,
    and ``None`` whenever the pair does not qualify:

    * either side is operator- or agent-asserted (:func:`is_extractor_asserted`);
    * the sides are distinguishable to trust — a different URL authority (the
      rule key), source type, snapshot author (the §6.4 AUTHOR tier),
      or contributor attribution. Only the entry itself may differ; a
      CORPUS_ENTRY-tier trust statement that separates two entries is rung 2's
      to act on, which runs first;
    * either side's source content date is unknown, or the dates are equal.
      ``new_date`` / ``existing_date`` override the snapshot's own date, which
      is how a caller supplies the **latest** observation of a re-stated claim
      (:func:`source_dates`) rather than its first;
    * ``require_attribution`` (the ``multi`` regime) and the
      lineage is anonymous — no snapshot author and no contributors on either
      side. In a shared store two unattributed claims cannot be assumed to
      share a principal, so the rung fails closed rather than letting one
      contributor's date retire another's belief.
    """
    if not (is_extractor_asserted(new_particle) and is_extractor_asserted(existing)):
        return None
    if new_entry is None or existing_entry is None:
        return None
    new_snap = _snapshot_of(new_entry, new_snapshot_id)
    old_snap = _snapshot_of(existing_entry, existing_snapshot_id)
    if new_snap is None or old_snap is None:
        return None
    same_lineage = (
        _authority(new_entry.uri_r) == _authority(existing_entry.uri_r)
        and new_entry.source_type == existing_entry.source_type
        and new_snap.author_id == old_snap.author_id
        and _contributors_key(new_entry) == _contributors_key(existing_entry)
    )
    if not same_lineage:
        return None
    if require_attribution and not (new_snap.author_id or _contributors_key(new_entry)):
        return None
    new_at = new_date or new_snap.content_published_at
    old_at = existing_date or old_snap.content_published_at
    if new_at is None or old_at is None:
        return None
    new_at, old_at = _as_aware(new_at), _as_aware(old_at)
    if new_at > old_at:
        return 1
    if new_at < old_at:
        return -1
    return None
