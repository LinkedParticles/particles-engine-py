# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The revision pair: two generations of one ``MUTABLE`` source.

A ``MUTABLE`` source is re-extracted in full when it changes, and the
cascade then retires the claims the new generation no longer states.
The §6.6 ladder runs first, and its same-entry search pairs a new candidate c
with the previous generation's claim x. When the probe confirms that pair and
rung 2.5 does not take it, rung 3 would quarantine c behind an INCONSISTENCY
record, and the cascade would then retire x in the same transaction: the store
would believe neither value while the source states the new one. Techspec §7.4
already decides the question for a ``MUTABLE`` source: only the generation
establishes which version is operative.

A pair (c, x) from the extraction of snapshot S of entry E is a **revision
pair** when all three hold:

* **E alone states x.** x's ``SOURCE`` refs name E's current generation, and
  no ref to another entry is a current statement by that entry (
  the read the cascade's attestation check makes).
* **S is newer.** S was captured after every snapshot of E that x's refs name,
  so an older snapshot extracted after a newer one forms none.
* **S does not restate x.** x is not in the pass's restated set: the claims S
  carried forward and the claims whose normalized content equals
  one of S's candidates' (the key's content half, subjects ignored).

Every condition is a property of x and the pass, so the result is a set of
existing claim ids, fixed before the write loop. On a revision pair the
pipeline writes an ``INCONSISTENT`` verdict as ``CORROBORATES`` (c is
inserted ``ACTIVE`` and the cascade retires x), and rung 2.5 takes its
direction from the generation (c is the newer claim). Gather is
:func:`load_revision_claims`; the decisions are pure.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.conflict_resolution import ConflictVerdict, LadderOutcome
from particles.core.duplicate_key import content_hash
from particles.core.schema import CorpusEntry, Mutability, Particle, ProvenanceRefType, Snapshot
from particles.corpus.store import EntryCurrency, get_entry_currency
from particles.store.observer_scope_join import SourceRef, attests

#: The quality-note key a pass's settled revision pairs are disclosed under.
REVISION_SETTLED = "REVISION_SETTLED"


def restated_ids(
    existing: Iterable[Particle],
    *,
    carry_forward_ids: Collection[str],
    candidate_contents: Iterable[str],
) -> frozenset[str]:
    """The existing claims the new generation restates verbatim.

    Two facts the pass already holds: the claims it carried forward, and the
    claims whose normalized content equals one of its candidates'. The second
    ignores subjects, so it errs toward "restated" and leaves such a pair to
    the ladder. A reworded restatement is not one, on purpose.
    """
    hashes = {content_hash(content) for content in candidate_contents}
    out = set(carry_forward_ids)
    out.update(p.id for p in existing if content_hash(p.content) in hashes)
    return frozenset(out)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def revision_claims(
    existing: Iterable[Particle],
    *,
    entry: CorpusEntry,
    snapshot: Snapshot,
    currency: Mapping[str, EntryCurrency],
    restated: Collection[str],
) -> frozenset[str]:
    """The ids of the existing claims that form a revision pair with this pass.

    ``currency`` holds :func:`~particles.corpus.store.get_entry_currency` for
    ``entry`` and every other entry the claims' refs name. An entry of any
    mutability but ``MUTABLE`` has no generations to settle, so
    it yields none. A ref this function cannot date (no snapshot id, or one
    the entry does not hold) keeps the claim out: the safe direction is
    today's disclosed outcome.
    """
    if entry.mutability != Mutability.MUTABLE:
        return frozenset()
    own = currency.get(entry.entry_id)
    current = own.latest_snapshot_id if own is not None else None
    if current is None:
        return frozenset()
    captured = {s.snapshot_id: _aware(s.captured_at) for s in entry.snapshots}
    s_at = _aware(snapshot.captured_at)

    def is_revision(particle: Particle) -> bool:
        if particle.id in restated:
            return False
        refs = [r for r in particle.provenance if r.type is ProvenanceRefType.SOURCE]
        own_refs = [r for r in refs if r.corpus_entry_id == entry.entry_id]
        if not any(r.snapshot_id == current for r in own_refs):
            return False
        for ref in refs:
            if ref.corpus_entry_id == entry.entry_id:
                continue
            # As the cascade reads it: an id with no corpus entry behind it
            # is not a source, and any other entry's current statement is.
            other = currency.get(ref.corpus_entry_id)
            ref_key = SourceRef(ref.corpus_entry_id, ref.snapshot_id)
            if other is not None and attests(ref_key, other):
                return False
        for ref in own_refs:
            at = captured.get(ref.snapshot_id) if ref.snapshot_id else None
            if at is None or not s_at > at:
                return False
        return True

    return frozenset(p.id for p in existing if is_revision(p))


async def load_generation_claims(
    session: AsyncSession,
    existing: Sequence[Particle],
    *,
    entry: CorpusEntry,
    snapshot: Snapshot,
    carry_forward_ids: Collection[str],
) -> frozenset[str]:
    """The claims stated only by an earlier generation of this entry, in one batched read.

    :func:`revision_claims` with the carried-forward claims as its only
    restated set: the claims the cascade would retire after this
    pass. The restatement match offers these; the revision pairs
    are these less the claims the pass restates (:func:`revision_from`). The
    read is made on every store, rescoped or not: the observer gate's prime
    loads scope only on a rescoped store, and the rule does not depend on
    scope.
    """
    if entry.mutability != Mutability.MUTABLE or not existing:
        return frozenset()
    entry_ids = {entry.entry_id} | {
        r.corpus_entry_id
        for p in existing
        for r in p.provenance
        if r.type is ProvenanceRefType.SOURCE and r.corpus_entry_id
    }
    currency = await get_entry_currency(session, entry_ids)
    return revision_claims(
        existing,
        entry=entry,
        snapshot=snapshot,
        currency=currency,
        restated=frozenset(carry_forward_ids),
    )


def revision_from(
    generation: Collection[str],
    existing: Iterable[Particle],
    *,
    candidate_contents: Iterable[str],
    carried: Collection[str] = frozenset(),
) -> frozenset[str]:
    """The revision pairs' claims: the generation's, less what the pass restates.

    A claim is restated when a candidate has its normalized content or the
    restatement match carried it; the carried-forward claims are
    already outside ``generation``.
    """
    restated = restated_ids(
        existing, carry_forward_ids=carried, candidate_contents=candidate_contents
    )
    return frozenset(generation) - restated


async def load_revision_claims(
    session: AsyncSession,
    existing: Sequence[Particle],
    *,
    entry: CorpusEntry,
    snapshot: Snapshot,
    carry_forward_ids: Collection[str],
    candidate_contents: Sequence[str],
) -> frozenset[str]:
    """Gather and decide :func:`revision_claims` for one pass, in one batched read.

    ``existing`` is every claim the pass's pairs name.
    """
    generation = await load_generation_claims(
        session, existing, entry=entry, snapshot=snapshot, carry_forward_ids=carry_forward_ids
    )
    return revision_from(generation, existing, candidate_contents=candidate_contents)


def settle_by_generation(outcome: LadderOutcome) -> LadderOutcome | None:
    """The outcome a revision pair is written with, or ``None`` when it is unchanged.

    Rung 3 does not apply to a revision pair: an ``INCONSISTENT``
    verdict, forced or not, is written as ``CORROBORATES`` is, and the cascade
    then retires the earlier claim. Every other verdict stands.
    """
    if outcome.verdict is ConflictVerdict.INCONSISTENT:
        return LadderOutcome(verdict=ConflictVerdict.CORROBORATES, record_divergence=False)
    return None


def generation_order(order: int | None) -> int | None:
    """Rung 2.5's order on a revision pair: the candidate is the newer claim.

    The update rung still decides whether the pair is an update (``None``
    stays ``None``); the generation decides which side is newer, whatever the
    two content dates say, so the earlier claim is the only one it retires.
    """
    return None if order is None else 1


def revision_settled_note(count: int) -> str:
    """The quality note for a pass that settled revision pairs by the generation."""
    return f"{REVISION_SETTLED}: {count} pair(s) settled by the generation"
