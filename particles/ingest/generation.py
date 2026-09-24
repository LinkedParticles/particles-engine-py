# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The snapshot-generation cascade for ``MUTABLE`` corpus entries.

The mutability contract: a new snapshot of a ``MUTABLE`` source sets the
prior snapshot's particles ``PROVENANCE_STALE`` and re-extracts. Until
the only implementation lived in ``corpus/fetch.py`` behind ``maybe_refetch``,
which had no production caller — so on the path every real deposit takes, a
revised document left its previous generation of beliefs ACTIVE forever. The
2026-07-24 dogfood census found 3,127 such particles, ~17% of the store's
ACTIVE beliefs.

**Ordering is the design.** The old implementation demoted *before* extraction.
That is wrong and is not carried over: the chunk-hash carry-forward looks
up **ACTIVE** particles, so pre-demoting blinds it, re-pays the LLM for every
unchanged paragraph, and mints duplicates of claims that never changed. The
cascade therefore runs **after** the new snapshot has been extracted —

    new RESPONSE snapshot (PENDING)
      → extract   — carry-forward keeps unchanged chunks' particles ACTIVE
      → cascade   — demote what is still anchored to a superseded snapshot

— so a particle that survived because its chunk is unchanged stays ACTIVE, and
only what the new generation did not reproduce is retired. Deletions and
rewrites are caught; restatements are deduped by carry-forward rather than
demoted-and-recreated.

Keying on *generation* rather than on semantics is what makes this work where
§6.6 could not: the intra-entry conflict ladder fires on a contradiction probe
between individual claims, which is blind to deletions and correctly declines to
retire a replaced-but-not-contradicted claim. It also means the cascade costs
**no LLM calls**.

Demote, never delete: ``PROVENANCE_STALE`` keeps the particle's content,
provenance, and confidence, surfaces it in the curation queue, and is
reversible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Mutability, ProvenanceRefType
from particles.core.status import Status, StatusReason

log = logging.getLogger(__name__)


async def cascade_superseded_generation(
    session: AsyncSession,
    *,
    entry_id: str,
    current_snapshot_id: str,
    exclude_ids: frozenset[str] = frozenset(),
) -> list[str]:
    """Demote ACTIVE particles anchored to a superseded snapshot of ``entry_id``.

    A no-op for every entry whose mutability is not ``MUTABLE``, so callers
    need not pre-filter: ``APPEND_ONLY`` content is additive by definition,
    ``STABLE`` never changes, and ``EPHEMERAL`` is not archived.

    Args:
        entry_id: The corpus entry whose generations are being reconciled.
        current_snapshot_id: The snapshot that has just been extracted. Every
            *other* snapshot of this entry is superseded by it.
        exclude_ids: Particle ids to leave ACTIVE regardless. Callers pass the
            carry-forward ids. A carried-forward particle's
            provenance edge keeps naming the snapshot it was originally
            extracted from (the edge is never re-pointed), so without this it
            would be misread as a stale generation and demoted. A belt since
            the re-observation is now also on the particle's
            provenance, which the attestation check below reads.

    A candidate some entry's *current* snapshot still attests is skipped—
    a claim a live source states is not ``PROVENANCE_STALE``
    under any reading of that status. The cascade was designed for
    single-entry particles; a claim folded from several sources
    retires only when none of them states it any more. "Current" is read as
    the scope join reads it: this entry's ``current_snapshot_id``, another
    ``MUTABLE`` entry's latest extracted snapshot, or any snapshot of an entry
    of another mutability.

    Returns:
        The ids demoted, in query order. Does not commit; the caller owns the
        transaction.
    """
    from particles.corpus.store import get_entry
    from particles.store.particle_store import (
        get_active_particle_ids_from_other_snapshots,
        update_particle_status,
    )

    entry = await get_entry(session, entry_id)
    if entry is None or entry.mutability != Mutability.MUTABLE:
        return []

    candidates = await get_active_particle_ids_from_other_snapshots(
        session, entry_id, current_snapshot_id
    )
    candidates = await _unattested(
        session,
        [pid for pid in candidates if pid not in exclude_ids],
        entry_id=entry_id,
        current_snapshot_id=current_snapshot_id,
    )
    demoted: list[str] = []
    for particle_id in candidates:
        await update_particle_status(
            session, particle_id, Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY
        )
        demoted.append(particle_id)

    if demoted:
        log.info(
            "Entry %s: demoted %d particle(s) from superseded snapshot generations "
            "(MUTABLE; current snapshot %s)",
            entry_id[:8],
            len(demoted),
            current_snapshot_id[:8],
        )
    return demoted


@dataclass
class GenerationBackfillReport:
    """What :func:`backfill_superseded_generations` found (and did)."""

    #: MUTABLE entries carrying more than one snapshot.
    entries_scanned: int = 0
    #: …of those, the ones with at least one demotable particle.
    entries_affected: int = 0
    #: Total particles demoted (or, when ``dry_run``, that would be).
    demoted: int = 0
    #: ``(entry_id, uri_r, count)`` per affected entry, largest first.
    per_entry: list[tuple[str, str | None, int]] = field(default_factory=list)
    #: Entries skipped because no snapshot of theirs has been extracted yet —
    #: demoting there would leave a hole rather than replace a generation.
    entries_unextracted: int = 0


async def backfill_superseded_generations(
    session: AsyncSession, *, dry_run: bool = True
) -> GenerationBackfillReport:
    """Apply the §2 cascade to entries whose snapshots moved before.

    The forward-looking cascade only fires on newly-extracted snapshots, so
    stores that predate it carry an accumulated backlog. This walks every
    ``MUTABLE`` entry with more than one snapshot and demotes what is anchored
    to a generation older than that entry's latest **COMPLETE** snapshot.

    "Latest COMPLETE" rather than "latest" is deliberate: if the newest snapshot
    is still ``PENDING``, the replacement beliefs do not exist yet, and retiring
    the old generation would leave the store with neither.
    "COMPLETE" here means an extracted RESPONSE snapshot: a REVISIT is COMPLETE
    too, but it records that the content did not change and carries no beliefs
    of its own, so it is never the current generation.

    ``dry_run`` counts without writing (it does not write-then-roll-back), so a
    caller may safely share its session with other work. Does not commit; the
    caller owns the transaction.
    """
    from particles.corpus.store import (
        get_latest_completed_snapshot_id,
        list_mutable_entries_with_multiple_snapshots,
    )
    from particles.store.particle_store import get_active_particle_ids_from_other_snapshots

    report = GenerationBackfillReport()
    for entry_id, uri_r in await list_mutable_entries_with_multiple_snapshots(session):
        report.entries_scanned += 1
        current = await get_latest_completed_snapshot_id(session, entry_id)
        if current is None:
            report.entries_unextracted += 1
            continue
        if dry_run:
            count = len(
                await _unattested(
                    session,
                    await get_active_particle_ids_from_other_snapshots(session, entry_id, current),
                    entry_id=entry_id,
                    current_snapshot_id=current,
                )
            )
        else:
            count = len(
                await cascade_superseded_generation(
                    session, entry_id=entry_id, current_snapshot_id=current
                )
            )
        if count:
            report.entries_affected += 1
            report.demoted += count
            report.per_entry.append((entry_id, uri_r, count))

    report.per_entry.sort(key=lambda row: row[2], reverse=True)
    if dry_run:
        log.info(
            "Generation backfill (dry run): %d particle(s) across %d entries would be demoted",
            report.demoted,
            report.entries_affected,
        )
    return report


async def _unattested(
    session: AsyncSession,
    particle_ids: list[str],
    *,
    entry_id: str,
    current_snapshot_id: str,
) -> list[str]:
    """``particle_ids`` that no entry's current snapshot attests, order kept."""
    if not particle_ids:
        return []
    from particles.corpus.store import get_entry_currency
    from particles.store.observer_scope_join import SourceRef, attests
    from particles.store.particle_store import get_particles_by_ids

    particles = await get_particles_by_ids(session, particle_ids)
    refs = {
        p.id: [
            SourceRef(r.corpus_entry_id, r.snapshot_id)
            for r in p.provenance
            if r.type is ProvenanceRefType.SOURCE
        ]
        for p in particles.values()
    }
    others = {r.entry_id for rs in refs.values() for r in rs if r.entry_id != entry_id}
    currency = await get_entry_currency(session, others)

    def attested(pid: str) -> bool:
        for ref in refs.get(pid, ()):
            if ref.entry_id == entry_id:
                if ref.snapshot_id == current_snapshot_id:
                    return True
            elif ref.entry_id in currency and attests(ref, currency[ref.entry_id]):
                return True
        return False

    return [pid for pid in particle_ids if not attested(pid)]
