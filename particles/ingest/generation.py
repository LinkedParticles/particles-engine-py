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

**"Current" is capture order, never extraction order**. The
snapshot just extracted is current only when no processed generation of the
entry was captured after it. An older snapshot extracted late (a forced
``extract --snapshot-id``, a programmatic call) is measured against the newer
generation, so the late pass retires its own unattested output and never the
current generation's claims.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Mutability, ProvenanceRefType
from particles.core.status import Status, StatusReason
from particles.core.validity import ensure_utc

log = logging.getLogger(__name__)

#: The quality-note prefix for a pass that extracted an older generation late.
LATE_GENERATION = "LATE_GENERATION"


@dataclass(frozen=True)
class CurrentGeneration:
    """Which record of a ``MUTABLE`` entry a cascade measures against."""

    snapshot_id: str
    #: The record is a processed ``ABSENT``: the entry states nothing, so
    #: snapshot-less provenance is retired too.
    absent: bool = False
    #: A processed generation was captured after the snapshot just extracted.
    late: bool = False


async def current_generation(
    session: AsyncSession, entry_id: str, extracted_snapshot_id: str
) -> CurrentGeneration:
    """The newer, by capture time, of ``extracted_snapshot_id`` and the entry's current generation.

    The snapshot just extracted is not yet ``COMPLETE`` when the cascade runs
    (it is marked in the same transaction, afterwards), so
    ``get_entry_currency`` still names its predecessor or nothing; the
    comparison is what makes the newer of the two current. On equal capture
    times the snapshot just extracted wins.
    """
    from particles.corpus.store import get_entry_currency, get_snapshot

    currency = (await get_entry_currency(session, [entry_id])).get(entry_id)
    if currency is None or currency.latest_snapshot_id in (None, extracted_snapshot_id):
        return CurrentGeneration(extracted_snapshot_id)
    held = await get_snapshot(session, currency.latest_snapshot_id)
    extracted = await get_snapshot(session, extracted_snapshot_id)
    if held is None or extracted is None:
        return CurrentGeneration(extracted_snapshot_id)
    if ensure_utc(held.captured_at) <= ensure_utc(extracted.captured_at):
        return CurrentGeneration(extracted_snapshot_id)
    return CurrentGeneration(held.snapshot_id, absent=currency.absent, late=True)


def late_generation_note(count: int) -> str:
    """The quality note for a pass that extracted an older generation late."""
    return (
        f"{LATE_GENERATION}: this snapshot was captured before the entry's current "
        f"generation; {count} claim(s) it stated that the current one does not were "
        f"retired, and none of the current generation's"
    )


async def cascade_superseded_generation(
    session: AsyncSession,
    *,
    entry_id: str,
    current_snapshot_id: str,
    exclude_ids: frozenset[str] = frozenset(),
    include_unsnapshotted: bool = False,
    late_out: list[str] | None = None,
) -> list[str]:
    """Demote ACTIVE particles anchored to a superseded snapshot of ``entry_id``.

    A no-op for every entry whose mutability is not ``MUTABLE``, so callers
    need not pre-filter: ``APPEND_ONLY`` content is additive by definition,
    ``STABLE`` never changes, and ``EPHEMERAL`` is not archived.

    Args:
        entry_id: The corpus entry whose generations are being reconciled.
        current_snapshot_id: The snapshot that has just been extracted (or the
            record absence processing is processing). It is current unless a
            processed generation of the entry was captured after it; then that
            generation is, and every other snapshot is superseded by it.
        exclude_ids: Particle ids to leave ACTIVE regardless. Callers pass the
            carry-forward ids. A carried-forward particle's
            provenance edge keeps naming the snapshot it was originally
            extracted from (the edge is never re-pointed), so without this it
            would be misread as a stale generation and demoted. A belt since
            the re-observation is now also on the particle's
            provenance, which the attestation check below reads.
        include_unsnapshotted: Also take provenance edges with no snapshot id.
            Absence processing passes it: an ``ABSENT`` current
            generation states nothing, so no edge of the entry survives it.
            Implied when the current generation turns out to be a processed
            absence.
        late_out: When the snapshot was extracted late, the demoted ids are
            also appended here, so the caller can disclose the pass.

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

    current = await current_generation(session, entry_id, current_snapshot_id)
    current_snapshot_id = current.snapshot_id
    include_unsnapshotted = include_unsnapshotted or current.absent

    candidates = await get_active_particle_ids_from_other_snapshots(
        session, entry_id, current_snapshot_id, include_unsnapshotted=include_unsnapshotted
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
            "(MUTABLE; current snapshot %s%s)",
            entry_id[:8],
            len(demoted),
            current_snapshot_id[:8],
            ", extracted late" if current.late else "",
        )
    if current.late and late_out is not None:
        late_out.extend(demoted)
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
    session: AsyncSession,
    *,
    dry_run: bool = True,
    entry_ids: Collection[str] | None = None,
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
    of its own, so it is never the current generation. The current generation
    is read through ``get_entry_currency``, so an entry whose current
    generation is a completed ``ABSENT`` record is processed as absence
    processing processes it, snapshot-less edges included.

    ``entry_ids`` scopes the walk; ``import vault`` passes the entries it just
    reclassified, to preview what the backfill would retire among them.
    ``None`` is store-wide.

    ``dry_run`` counts without writing (it does not write-then-roll-back), so a
    caller may safely share its session with other work. Does not commit; the
    caller owns the transaction.
    """
    from particles.corpus.store import (
        get_entry_currency,
        list_mutable_entries_with_multiple_snapshots,
    )
    from particles.store.particle_store import get_active_particle_ids_from_other_snapshots

    scope = set(entry_ids) if entry_ids is not None else None
    report = GenerationBackfillReport()
    for entry_id, uri_r in await list_mutable_entries_with_multiple_snapshots(session):
        if scope is not None and entry_id not in scope:
            continue
        report.entries_scanned += 1
        currency = (await get_entry_currency(session, [entry_id])).get(entry_id)
        current = currency.latest_snapshot_id if currency is not None else None
        if currency is None or current is None:
            report.entries_unextracted += 1
            continue
        if dry_run:
            count = len(
                await _unattested(
                    session,
                    await get_active_particle_ids_from_other_snapshots(
                        session, entry_id, current, include_unsnapshotted=currency.absent
                    ),
                    entry_id=entry_id,
                    current_snapshot_id=current,
                )
            )
        else:
            count = len(
                await cascade_superseded_generation(
                    session,
                    entry_id=entry_id,
                    current_snapshot_id=current,
                    include_unsnapshotted=currency.absent,
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
