# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-pass exact-duplicate suppression at write time.

The prevention-side twin of Tier-A auto-merge, and the cross-pass
sibling of :mod:`particles.ingest.candidate_dedup` (which folds duplicates
*within* one extraction pass). Both share one notion of "the same
content" — :mod:`particles.core.duplicate_key`.

**The leak this closes.** ``extract_snapshot`` scopes §6.6 conflict resolution
to the *same corpus entry* (deliberate), so a claim already ACTIVE
from another entry is invisible to every rung; and widening that scope would not
help, because an exact duplicate reaches ``resolve_conflict`` step 2, finds no
contradiction signal, returns ``CORROBORATES``, and is written as a *second*
ACTIVE particle. Measured on the live store 2026-07-25: 3,411 of 21,650
particles minted over eight days were verbatim copies of claims already held
(15.8 %), one per re-extraction of re-deposited harvest material.

**The rung.** Before §6.6 sees a candidate, ask a deterministic question: *is
this exact claim already ACTIVE?* If so, do not mint a second particle — append
the candidate's provenance ref to the existing one
(:func:`particles.store.particle_store.append_provenance_ref`) so the new
source's evidence still lands, on the ACTIVE read surface rather than behind a
graph walk.

**Deliberately not similarity.** No cosine, no LLM, no threshold — ever.
Measurement showed that below exact identity cosine does not order
duplicate-likelihood (the 0.97–0.99 band hand-scores *worse* than 0.95–0.97),
and its worst false positive — ``claude-opus-4-6`` vs ``claude-opus-4-5`` —
sits at 0.9951. A threshold here would silently drop claims differing by one
load-bearing token.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.duplicate_key import content_hash, duplicate_key
from particles.core.schema import Particle, is_truth_apt
from particles.core.stance import holder_from_properties
from particles.core.status import Status, StatusReason
from particles.extraction.polarity import is_non_asserted
from particles.extraction.scope import is_excluded_document_meta
from particles.store.particle_store import (
    get_active_particles_by_content_hashes,
    get_particles_by_content_hashes_and_state,
)

log = logging.getLogger(__name__)

# The identity tuple two claims must share: normalized content, subject-id set,
# stance holder.
DuplicateKey = tuple[str, frozenset[str], str | None]


def is_suppression_eligible(particle: Particle) -> bool:
    """Whether a particle may participate in suppression at all.

    The same exclusions auto-merge inherits, for the same reasons:

    * **non-truth-apt** — an opinion or a document's constitutive
      rule has no shared truth to be "the same claim" about.
    * **DOCUMENT_META** / **non-asserted** — off the
      conflict surface entirely; these are written straight to ACTIVE and the
      duplicate machinery never touches them.

    Applied to *both* sides, so an ineligible candidate is never suppressed and
    an ineligible existing particle is never suppressed *into*.
    """
    if not is_truth_apt(particle):
        return False
    return not (
        is_excluded_document_meta(particle.properties) or is_non_asserted(particle.properties)
    )


def particle_key(particle: Particle) -> DuplicateKey:
    """The identity tuple for a stored or candidate particle."""
    return duplicate_key(
        particle.content,
        particle.subject_ids,
        holder_from_properties(particle.properties),
    )


class DuplicateIndex:
    """In-memory identity index for one write pass.

    Built once from a single indexed store probe, then **maintained in place**
    as particles are written — so a candidate suppresses against a sibling
    written earlier in the same pass, not only against pre-existing rows. This
    mirrors the ``candidate_cache`` discipline in
    :func:`particles.ingest.pipeline.reconcile_and_insert`.

    First-writer-wins: the earliest-registered particle for a key is the one
    later duplicates are suppressed into, which makes a re-run idempotent.
    """

    def __init__(
        self,
        existing: Iterable[Particle] = (),
        *,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._by_key: dict[DuplicateKey, Particle] = {}
        self._exclude_ids = exclude_ids
        for particle in existing:
            self.add(particle)

    def add(self, particle: Particle) -> None:
        """Register a written / existing ACTIVE particle as a suppression target."""
        if particle.id in self._exclude_ids or not is_suppression_eligible(particle):
            return
        self._by_key.setdefault(particle_key(particle), particle)

    def find(self, particle: Particle) -> Particle | None:
        """The ACTIVE particle already holding this exact claim, if any."""
        if not is_suppression_eligible(particle):
            return None
        return self._by_key.get(particle_key(particle))


async def build_duplicate_index(
    session: AsyncSession,
    contents: Sequence[str],
    *,
    exclude_ids: frozenset[str] = frozenset(),
) -> DuplicateIndex:
    """Load the suppression index for a pass in one indexed probe.

    ``exclude_ids`` is load-bearing, not an optimisation: a reindex passes the
    particles it is *about* to supersede, and suppressing a fresh candidate into
    one of those would retire the claim's only surviving copy — the claim would
    vanish from the store entirely. The §6.6 candidate set excludes the same ids
    for the analogous reason.
    """
    hashes = [content_hash(c) for c in contents]
    rows = await get_active_particles_by_content_hashes(session, hashes)
    return DuplicateIndex(rows, exclude_ids=exclude_ids)


# the retirements that encode a *judgment about the value* — a
# person (or a review-derived policy acting on a person's judgment) decided
# this claim should not be believed as stated. A re-emitted twin of any of
# these is quarantined for review rather than re-minted ACTIVE.
#
# Deliberately NOT in this set, because each encodes something other than a
# verdict on the value: SOURCE_RETRACTED (the *source* was withdrawn — another
# source making the same claim is new evidence), SUPERSEDED_BY_REINDEX and
# DOCUMENT_SUPERSEDED (a newer generation replaced it), DUPLICATE_MERGED (an
# identical copy folded away — the claim is still ACTIVE elsewhere),
# VALIDITY_EXPIRED (time, not judgment), and the trust-differential demotions
# TRUST_DEMOTED / LOWER_TRUST_SOURCE (a policy about sources, not a verdict on
# the claim). CONFLICT_PENDING is handled separately below: it is a hold
# awaiting judgment, not a judgment.
JUDGMENT_RETIREMENTS: tuple[tuple[Status, StatusReason], ...] = (
    (Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION),
    (Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION),
    (Status.PROVENANCE_STALE, StatusReason.CONFLICT_RESOLVED),
)

# A twin already sitting in quarantine (an ordinary §6.6 loser, or an earlier
# hold). Re-encountering it appends the new observation to the held
# row rather than opening a second hold — what makes a repeated reindex of an
# unchanged source idempotent instead of a wrapper per run.
PENDING_HOLD: tuple[tuple[Status, StatusReason], ...] = (
    (Status.PROVENANCE_STALE, StatusReason.CONFLICT_PENDING),
)


def is_judgment_retired(particle: Particle) -> bool:
    """True when the particle left ACTIVE by a judgment about its value."""
    return (particle.status, particle.status_reason) in JUDGMENT_RETIREMENTS


def is_pending_hold(particle: Particle) -> bool:
    """True for a quarantined row awaiting review."""
    return (particle.status, particle.status_reason) in PENDING_HOLD


async def build_retired_value_index(
    session: AsyncSession,
    contents: Sequence[str],
    *,
    exclude_ids: frozenset[str] = frozenset(),
) -> DuplicateIndex:
    """Load the retired-value index for a pass in one indexed probe.

    Registers, under the exact-duplicate identity key, every judgment-retired twin
    (:data:`JUDGMENT_RETIREMENTS`) and every pending hold (:data:`PENDING_HOLD`)
    of some candidate. Pending holds are registered **first** so that, where
    both exist for one key, a re-emitted candidate lands on the open hold (one
    review lifts everything) rather than opening a second one.

    Consulted *after* the ACTIVE index: a claim that is currently believed
    somewhere absorbs the observation before any retired twin is
    considered — the value is not in dispute if the store holds it ACTIVE.
    """
    hashes = [content_hash(c) for c in contents]
    pending = await get_particles_by_content_hashes_and_state(session, hashes, PENDING_HOLD)
    retired = await get_particles_by_content_hashes_and_state(session, hashes, JUDGMENT_RETIREMENTS)
    return DuplicateIndex([*pending, *retired], exclude_ids=exclude_ids)


def retired_value_note(quarantined: int, absorbed: int) -> str:
    """The disclosure line for an extraction pass's quality notes."""
    return (
        f"RETIRED_VALUE_QUARANTINED: {quarantined} candidate(s) re-asserted a claim an "
        f"operator or reviewer had retired; each was stored quarantined "
        f"(CONFLICT_PENDING) behind an INCONSISTENCY record for review instead of "
        f"re-entering ACTIVE, and {absorbed} further observation(s) were recorded on a "
        f"hold already awaiting review"
    )


def suppression_note(count: int) -> str:
    """The disclosure line for an extraction pass's quality notes.

    Suppression must never be silent — an operator has to be able to see that a
    pass "produced N particles and suppressed M" without querying the store.
    """
    return (
        f"DUPLICATE_SUPPRESSED: {count} candidate(s) already held verbatim by an "
        f"ACTIVE particle; provenance recorded on the existing particle(s) "
        f"instead of minting duplicates"
    )
