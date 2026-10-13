# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Which existing claims a candidate is paired with for §6.6: a pure decision (D2).

A write path gathers, per candidate, the similarity-gated nearest claim, the
subject-keyed pool, the observer precondition of each
and, in a ``multi`` store, whether rung 2.5 could act on each pool member.
Pure decisions then run around the two expensive gathers, the LLM update and
contradiction probes:

  0. :func:`plan_update_first` names the pairs rung 2.5 could act on, which
     the caller dates and asks the update question **first**, keeping only
     those :func:`admits_update_first` admits: rung 2.5 rests on the
     same-slot check, and a contradiction probe asks whether two claims can
     both be true at once, which successive values of one slot can;
  1. :func:`plan_probes` names exactly the pairs to probe, and in which role,
     skipping those the update answer has already settled;
  2. :func:`plan_update_probes` names the confirmed pairs rung 2.5 might
     settle that were not asked first, which the caller asks the update probe
     about once it has dated them;
  3. :func:`select_pairs` reads both probes' results and picks the primary
     pair for the ladder, the rung 2.5 extras, and the declined pairs recorded
     as divergences;
  4. :func:`plan_readings` names the probe-confirmed pairs the pass would act
     on, which the caller reads a second time, and
     :func:`apply_readings` lets each reading's verdict replace the probe's.

Before the write loop, :func:`write_order` says in which order the candidates
meet the ladder: one that may retire a claim by update goes first. After the
ladder, :func:`plan_update_extras` says which extras rung 2.5 demotes, in order.

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

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

from particles.core.conflict_resolution import (
    SlotVerdict,
    admits_update,
    newer_is_settled,
    offers_subject_pair,
)
from particles.core.observer_scope import PairPrecondition
from particles.core.past_value import records_past_value
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
    slots: Mapping[str, SlotVerdict] = field(default_factory=dict)
    """The update probe's verdict by existing id, for each pair it answered.
    Rung 2.5 acts on a pair only when the verdict is
    :attr:`~SlotVerdict.CHANGES`; a :attr:`~SlotVerdict.FIXED` pair reaches
    rung 3."""
    reading: str | None = None
    """The instruction a second reading confirmed ``primary``'s signal under,
    stamped on a rung-3 record as ``conflict:reading``. ``None``
    when no reading confirmed it: none ran, or it failed and the signal was
    kept (a same-entry pair)."""
    same_slot: frozenset[str] = frozenset()
    """The existing ids asked the update question first whose answer settled
    ``CHANGES``: rung 2.5 may act on these with no contradiction
    signal."""

    def slot_of(self, existing_id: str) -> SlotVerdict | None:
        """The update probe's verdict on the pair with ``existing_id``, if it gave one."""
        return self.slots.get(existing_id)

    def same_slot_update(self, existing_id: str) -> bool:
        """Whether rung 2.5 may act on this pair on the update probe's answer alone."""
        return existing_id in self.same_slot

    @property
    def updatable(self) -> frozenset[str]:
        """The existing ids rung 2.5 may act on: a slot that changes over time."""
        return frozenset(pid for pid, slot in self.slots.items() if admits_update(slot))


def _offered(pairs: CandidatePairs, other: Particle) -> bool:
    # in a ``multi`` store a pair rung 2.5 cannot settle has no
    # outcome but a review item, so it is neither probed nor offered.
    return pairs.can_update is None or pairs.can_update.get(other.id, False)


def plan_update_first(pairs: CandidatePairs) -> list[Particle]:
    """The pairs asked the update question before the contradiction probe.

    Every pair rung 2.5 could act on: the nearest claim and each offered pool
    member whose observer precondition reconciles it. A declined
    pair reaches no rung, and a project's claim contesting a global one goes to
    review whatever the update answer, so neither is asked first. The caller
    dates each one as rung 2.5 would, and asks only those it can order and
    whose older claim :func:`admits_update_first`; the rest take the
    contradiction probe first, as before.
    """
    out: list[Particle] = []
    if pairs.nearest is not None and pairs.precondition_of(pairs.nearest) is (
        PairPrecondition.RECONCILE
    ):
        out.append(pairs.nearest)
    for other in pairs.subject_pool:
        if pairs.precondition_of(other) is PairPrecondition.RECONCILE and _offered(pairs, other):
            out.append(other)
    return out


def admits_update_first(earlier: str) -> bool:
    """Whether a datable pair is asked the update question first (guard 2).

    Not when the earlier claim records a past value
    (:func:`~particles.core.past_value.records_past_value`): a statement about
    the past gives the slot no value to replace, the update probe misreads some
    of them, and the contradiction probe in front of it keeps them, as before.
    """
    return not records_past_value(earlier)


def plan_probes(
    pairs: CandidatePairs,
    slots: Mapping[str, SlotVerdict] | None = None,
    admitted: Container[str] = (),
) -> list[tuple[Particle, PairRole]]:
    """The contradiction probes to run for one candidate, in order.

    The nearest claim is probed first, whatever its precondition: a declined
    pair is recorded only when the probe confirms it. Then each pool member in
    pool order: a declined one is probed to record a divergence, one rung 2.5
    could not act on in a ``multi`` store is skipped, and the rest are offered.

    ``slots`` and ``admitted`` are the update-first answers. A
    pair asked first whose answer settled ``CHANGES`` is not probed: rung 2.5
    retires on that answer, and nothing the probe could say changes it. Nor is
    a pool member asked first that fills a different slot: such a member is not
    offered whatever the probe says. Every other pair is probed.
    """
    verdicts = slots or {}

    def settled(other: Particle, *, pool: bool) -> bool:
        if other.id not in admitted:
            return False
        verdict = verdicts.get(other.id)
        return admits_update(verdict) or (pool and verdict is SlotVerdict.DIFFERENT)

    probes: list[tuple[Particle, PairRole]] = []
    if pairs.nearest is not None and not settled(pairs.nearest, pool=False):
        probes.append((pairs.nearest, PairRole.NEAREST))
    for other in pairs.subject_pool:
        if pairs.precondition_of(other) is PairPrecondition.DECLINE:
            probes.append((other, PairRole.DECLINED))
        elif _offered(pairs, other) and not settled(other, pool=True):
            probes.append((other, PairRole.SUBJECT))
    return probes


def plan_update_probes(
    pairs: CandidatePairs, probes: Mapping[str, bool], asked: Container[str] = ()
) -> list[Particle]:
    """The confirmed pairs rung 2.5 might settle, in probe order.

    A contradiction is not yet an update. Rung 2.5 retires a claim on a date,
    so it needs a second verdict first: that the two claims fill one slot and
    the later one gives it a new value. The candidates for that question are
    the pairs the ladder or the extras could act on: a confirmed nearest claim
    the precondition does not decline, and each confirmed pool member that is
    offered. The caller dates each one and asks the update probe only about
    those rung 2.5 could order, so the question costs a call only where a
    retirement is otherwise one step away. A pair in ``asked`` was asked the
    update question first and is not asked again.
    """
    out: list[Particle] = []
    nearest = pairs.nearest
    if (
        nearest is not None
        and nearest.id not in asked
        and probes.get(nearest.id, False)
        and pairs.precondition_of(nearest) is not PairPrecondition.DECLINE
    ):
        out.append(nearest)
    for other in pairs.subject_pool:
        if (
            other.id not in asked
            and pairs.precondition_of(other) is not PairPrecondition.DECLINE
            and _offered(pairs, other)
            and probes.get(other.id, False)
        ):
            out.append(other)
    return out


def select_pairs(
    pairs: CandidatePairs,
    probes: Mapping[str, bool],
    slots: Mapping[str, SlotVerdict] | None = None,
    admitted: Container[str] = (),
) -> PairSelection:
    """Pick the primary pair, the rung 2.5 extras and the declined pairs.

    ``probes`` holds the result of every probe :func:`plan_probes` named, by
    existing id. The primary pair is the nearest claim unless the probe
    confirmed it and the precondition declines it; when the nearest is absent,
    declined or unconfirmed, the best confirmed pool member takes its place and
    the rest are extras. An unconfirmed nearest claim stays primary when no
    pool member is confirmed: the ladder then corroborates it.

    ``slots`` holds the update probe's verdict on each pair
    :func:`plan_update_probes` named and the caller could date. A
    pool member the update probe says fills a different slot is **not
    offered**: the subject-keyed search exists to find updates, and a
    same-subject pair about two attributes was expected to cost one probe and
    change nothing. Both claims stay ACTIVE, the outcome before
    that search existed. The nearest claim is not filtered: its pair
    reached the ladder before, and a different-slot verdict only
    keeps rung 2.5 off it. A pool member that fills the same slot is offered
    whatever the slot's kind: rung 2.5 settles one that changes over time, and
    a fixed one, whose two values contradict, falls to rung 3.

    ``admitted`` holds the ids asked the update question first. One
    whose answer settled ``CHANGES`` is offered with no contradiction signal
    (:attr:`PairSelection.same_slot`): rung 2.5's evidence is that answer. The
    primary pair's ``signal`` is always the contradiction probe's own result,
    so no rung that picks between sources runs on an update answer.
    """
    verdicts = slots or {}
    same_slot = frozenset(
        other.id
        for other in pairs.pairable()
        if other.id in admitted and admits_update(verdicts.get(other.id))
    )
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
        elif _offered(pairs, other) and (
            other.id in same_slot
            or (probes.get(other.id, False) and offers_subject_pair(verdicts.get(other.id)))
        ):
            confirmed.append(other)
    acts = primary is not None and (signal or primary.id in same_slot)
    if not acts and confirmed:
        primary = confirmed.pop(0)
        signal = probes.get(primary.id, False)
    return PairSelection(
        primary=primary,
        signal=signal,
        extras=tuple(confirmed),
        declined=tuple(declined),
        slots=dict(verdicts),
        same_slot=same_slot,
    )


def write_order(selections: Sequence[PairSelection]) -> list[int]:
    """The order the write loop takes one pass's candidates in: updates first.

    A candidate with a pair the update probe said changes over time may retire
    that claim at rung 2.5. A sibling of the same pass that only
    contradicts the claim then meets a retired claim and is written as a plain
    insert, the outcome the loop already gave when the update happened to come
    first. Writing it first makes that outcome independent of the
    order the extractor emitted the claims in. Without it, a sibling written
    before the update met the old claim still ACTIVE, went to rung 3, and the
    newer source's claim stayed quarantined behind a record whose other side
    the update had just retired.

    Stable: the extractor's order is kept within each group.
    """
    first = [i for i, selection in enumerate(selections) if selection.updatable]
    rest = [i for i, selection in enumerate(selections) if not selection.updatable]
    return first + rest


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
    updatable: Container[str],
) -> list[tuple[str, str]]:
    """The ``(retired id, winner id)`` pairs rung 2.5 settles among the extras.

    The primary pair went through the full ladder and ``winner`` landed
    ``ACTIVE``. Each extra that qualifies is settled by source date: the older
    claim is demoted. A pair that does not qualify (no SOURCE ref, a
    precondition other than ``RECONCILE``, or no ``update_order``) is left as it
    was, so no second INCONSISTENCY is manufactured for one claim. The run
    stops at the winner: once an extra is newer, the winner is retired and has
    nothing left to settle.

    ``entries`` maps a corpus entry id to the entry (``None`` when missing), and
    ``precondition`` is by existing id, a missing id reading ``RECONCILE``.
    ``updatable`` holds the ids the update probe confirmed fill a slot that
    changes over time; any other extra is left as it was. A fixed
    slot's extra is left too: rung 3 is the primary pair's alone, so no second
    INCONSISTENCY is manufactured for one claim, and the census reads the pair.

    The ALEATORY condition is the ladder's own
    (:func:`~particles.core.conflict_resolution.newer_is_settled`):
    a pair is settled only when its newer claim is not ALEATORY, so an
    EPISTEMIC winner retires an older ALEATORY extra and an ALEATORY winner
    retires nothing. Calling the one function keeps the two routes from
    drifting apart.

    Each pair names the claim that retired the loser, which the caller records
    (the winner's ``supersedes`` holds only one id).
    """
    demoted: list[tuple[str, str]] = []
    for other in others:
        if other.id not in updatable:
            continue
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
        if order is None or not newer_is_settled(other, winner, order):
            continue
        if order > 0:
            demoted.append((other.id, winner.id))
        else:
            demoted.append((winner.id, other.id))
            break
    return demoted


# ---------------------------------------------------------------------------
# The second reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairReading:
    """The second reading of one probe-confirmed pair."""

    confirmed: bool | None
    """Whether the reading confirmed the contradiction; ``None`` when it failed
    (breaker open, call error, or a reply still cut off after the retry)."""
    reading: str | None = None
    """The instruction the pair was read under (``standing/1``, ``observed/1``)."""


@dataclass(frozen=True)
class ReadingTally:
    """What the second readings did to one candidate's selection."""

    read: int = 0
    """Pairs sent to the reading."""
    cleared: int = 0
    """Pairs the reading did not confirm: their signal was dropped."""
    unread: int = 0
    """Pairs whose reading failed (``conflict_unread``)."""


def plan_readings(selection: PairSelection) -> list[Particle]:
    """The probe-confirmed pairs a selection would act on, to be read again.

    The primary pair when its signal is confirmed, then each declined pair:
    every write that rests on the probe's confirmation (a rung-3 record, a
    rung 1.5 or 2 retirement, a ``CONTRADICTS`` edge) inherits its precision.
    The rung 2.5 extras are not read: each already needs the update probe's
    YES before it retires anything.
    """
    out: list[Particle] = []
    if selection.primary is not None and selection.signal:
        out.append(selection.primary)
    out.extend(selection.declined)
    return out


def apply_readings(
    selection: PairSelection,
    readings: Mapping[str, PairReading],
    same_entry: Container[str],
) -> tuple[PairSelection, ReadingTally]:
    """Let each second reading's verdict replace the probe's (§2).

    ``readings`` holds the reading of each pair :func:`plan_readings` named,
    by existing id; a pair with no reading is left as the probe decided.

    * **Confirmed:** unchanged, and a confirmed primary records the
      instruction in :attr:`PairSelection.reading`.
    * **Not confirmed:** no signal. The primary routes as a probe NO (it
      corroborates) and a declined pair writes no edge.
    * **Failed:** a pair whose existing claim came from the candidate's own
      entry (``same_entry``) keeps the probe's signal, unstamped: the nightly
      census never discloses a same-source pair, so nothing else would read
      it. A cross-entry pair loses it: the census reads cross-source pairs
      every night.
    """
    read = cleared = unread = 0

    def keeps(other: Particle) -> tuple[bool, str | None]:
        nonlocal read, cleared, unread
        verdict = readings.get(other.id)
        if verdict is None:
            return True, None
        read += 1
        if verdict.confirmed is None:
            unread += 1
            return other.id in same_entry, None
        if not verdict.confirmed:
            cleared += 1
            return False, None
        return True, verdict.reading

    primary, signal, stamp = selection.primary, selection.signal, None
    if primary is not None and signal:
        signal, stamp = keeps(primary)
    declined = tuple(other for other in selection.declined if keeps(other)[0])
    return (
        replace(selection, signal=signal, declined=declined, reading=stamp),
        ReadingTally(read=read, cleared=cleared, unread=unread),
    )
