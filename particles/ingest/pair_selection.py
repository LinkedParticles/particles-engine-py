# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Which existing claims a candidate is paired with for §6.6: a pure decision (D2).

A write path gathers, per candidate, the similarity-gated nearest claim, the
subject-keyed pool, the observer precondition of each
and, in a ``multi`` store, whether rung 2.5 could act on each pool member.
Two pure decisions then run around the one expensive gather,
the LLM contradiction probe:

  1. :func:`plan_probes` names exactly the pairs to probe, and in which role;
  2. :func:`select_pairs` reads the probe results and picks the primary pair
     for the ladder, the rung 2.5 extras, and the declined pairs recorded as
     divergences.

After the ladder, :func:`plan_update_extras` says which extras rung 2.5
demotes, in order.

The two write paths state their different candidacy policies as arguments:

* **extraction** passes the same-entry nearest claim, the subject pool, the
  primed preconditions and, in a ``multi`` store, the lineage facts;
* **``reconcile_and_insert``** passes its store-wide nearest claim (or the
  single nearest subject match) with an empty pool, no preconditions (the
  ladder applies the precondition itself) and ``can_update=None``, so its one
  pair is always primary.

Performs no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from particles.core.observer_scope import PairPrecondition
from particles.core.schema import CorpusEntry, Particle, ProvenanceRefType
from particles.ingest.update_supersession import update_order


class PairRole(Enum):
    """Why a pair is probed."""

    NEAREST = "nearest"
    """The similarity-gated nearest claim: primary when confirmed, declined
    (recorded as a divergence) when confirmed and the precondition declines it."""
    SUBJECT = "subject"
    """A subject-pool member offered to the ladder when confirmed."""
    DECLINED = "declined"
    """A subject-pool member the precondition declines, probed only to record a
    divergence."""


@dataclass(frozen=True)
class CandidatePairs:
    """Every existing claim one candidate could be paired with, and the gathered facts."""

    nearest: Particle | None = None
    """The similarity-gated nearest claim: the same-entry search on the extract
    path, the store-wide one (or its subject fallback) on ``reconcile_and_insert``."""
    subject_pool: tuple[Particle, ...] = ()
    """Subject-keyed pool members, best first, ``nearest`` excluded."""
    precondition: Mapping[str, PairPrecondition] = field(default_factory=dict)
    """The observer precondition by existing id; a missing id reads ``RECONCILE``."""
    can_update: Mapping[str, bool] | None = None
    """By pool member id, whether rung 2.5 could act on the pair; ``None``
    outside the ``multi`` regime, where every pool member is offered."""

    def precondition_of(self, particle: Particle) -> PairPrecondition:
        return self.precondition.get(particle.id, PairPrecondition.RECONCILE)

    def pairable(self) -> list[Particle]:
        """Every existing claim this candidate could pair with: the precondition's inputs."""
        head = [self.nearest] if self.nearest is not None else []
        return head + list(self.subject_pool)


@dataclass(frozen=True)
class PairSelection:
    """The pairs one candidate is reconciled against."""

    primary: Particle | None = None
    """The pair the §6.6 ladder runs on, or ``None``."""
    signal: bool = False
    """Whether the probe confirmed a contradiction on ``primary``."""
    extras: tuple[Particle, ...] = ()
    """Further confirmed subject pairs, best first, settled by rung 2.5 alone."""
    declined: tuple[Particle, ...] = ()
    """Confirmed pairs the precondition declined, recorded as ``CONTRADICTS``."""


def _offered(pairs: CandidatePairs, other: Particle) -> bool:
    # in a ``multi`` store a pair rung 2.5 cannot settle has no
    # outcome but a review item, so it is neither probed nor offered.
    return pairs.can_update is None or pairs.can_update.get(other.id, False)


def plan_probes(pairs: CandidatePairs) -> list[tuple[Particle, PairRole]]:
    """The contradiction probes to run for one candidate, in order.

    The nearest claim is probed first, whatever its precondition: a declined
    pair is recorded only when the probe confirms it. Then each pool member in
    pool order: a declined one is probed to record a divergence, one rung 2.5
    could not act on in a ``multi`` store is skipped, and the rest are offered.
    """
    probes: list[tuple[Particle, PairRole]] = []
    if pairs.nearest is not None:
        probes.append((pairs.nearest, PairRole.NEAREST))
    for other in pairs.subject_pool:
        if pairs.precondition_of(other) is PairPrecondition.DECLINE:
            probes.append((other, PairRole.DECLINED))
        elif _offered(pairs, other):
            probes.append((other, PairRole.SUBJECT))
    return probes


def select_pairs(pairs: CandidatePairs, probes: Mapping[str, bool]) -> PairSelection:
    """Pick the primary pair, the rung 2.5 extras and the declined pairs.

    ``probes`` holds the result of every probe :func:`plan_probes` named, by
    existing id. The primary pair is the nearest claim unless the probe
    confirmed it and the precondition declines it; when the nearest is absent,
    declined or unconfirmed, the best confirmed pool member takes its place and
    the rest are extras. An unconfirmed nearest claim stays primary when no
    pool member is confirmed: the ladder then corroborates it.
    """
    declined: list[Particle] = []
    primary, signal = pairs.nearest, False
    if primary is not None:
        signal = probes.get(primary.id, False)
        if signal and pairs.precondition_of(primary) is PairPrecondition.DECLINE:
            declined.append(primary)
            primary, signal = None, False
    confirmed: list[Particle] = []
    for other in pairs.subject_pool:
        if pairs.precondition_of(other) is PairPrecondition.DECLINE:
            if probes.get(other.id, False):
                declined.append(other)
        elif _offered(pairs, other) and probes.get(other.id, False):
            confirmed.append(other)
    if (primary is None or not signal) and confirmed:
        primary, signal = confirmed.pop(0), True
    return PairSelection(
        primary=primary, signal=signal, extras=tuple(confirmed), declined=tuple(declined)
    )


def first_source_ref_entry_id(particle: Particle) -> str | None:
    """The corpus entry of ``particle``'s first SOURCE ref, the one rung 2.5 dates it by."""
    ref = next((r for r in particle.provenance if r.type is ProvenanceRefType.SOURCE), None)
    return ref.corpus_entry_id if ref is not None else None


def plan_update_extras(
    winner: Particle,
    winner_entry: CorpusEntry | None,
    winner_snapshot_id: str,
    others: Sequence[Particle],
    entries: Mapping[str, CorpusEntry | None],
    precondition: Mapping[str, PairPrecondition],
) -> list[str]:
    """The ids rung 2.5 demotes among the extra pairs, in order.

    The primary pair went through the full ladder and ``winner`` landed
    ``ACTIVE``. Each extra that qualifies is settled by source date: the older
    claim is demoted. A pair that does not qualify (no SOURCE ref, a
    precondition other than ``RECONCILE``, or no ``update_order``) is left as it
    was, so no second INCONSISTENCY is manufactured for one claim. The run
    stops at the winner: once an extra is newer, the winner is retired and has
    nothing left to settle.

    ``entries`` maps a corpus entry id to the entry (``None`` when missing), and
    ``precondition`` is by existing id, a missing id reading ``RECONCILE``.
    """
    demoted: list[str] = []
    for other in others:
        ref = next((r for r in other.provenance if r.type is ProvenanceRefType.SOURCE), None)
        if ref is None:
            continue
        # an extra another project observes, or a global claim, is
        # left as it was.
        if precondition.get(other.id, PairPrecondition.RECONCILE) is not PairPrecondition.RECONCILE:
            continue
        order = update_order(
            winner,
            winner_entry,
            winner_snapshot_id,
            other,
            entries.get(ref.corpus_entry_id),
            ref.snapshot_id,
        )
        if order is None:
            continue
        if order > 0:
            demoted.append(other.id)
        else:
            demoted.append(winner.id)
            break
    return demoted
