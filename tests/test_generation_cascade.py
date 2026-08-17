"""Tests for the snapshot-generation cascade.

The defect this closes, measured on the dogfood store on 2026-07-24: 3,127
ACTIVE particles (~17% of the store) anchored to a *superseded* snapshot of a
MUTABLE local file, none of them a legitimate carry-forward. The
demotion was always specified; the only implementation lived behind
``maybe_refetch``, which had no production caller.

Two properties carry the design and are asserted here directly:

* **ordering** — carried-forward particles survive, because the cascade runs
  after extraction rather than before it; and
* **generation-keying** — a claim that the new snapshot simply stopped saying is
  retired, which is exactly what the §6.6 contradiction probe cannot do.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from particles.core.schema import (
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    FetchPolicy,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.corpus.store import CorpusEntryRow, SnapshotRow
from particles.ingest.generation import (
    backfill_superseded_generations,
    cascade_superseded_generation,
)
from particles.store.particle_store import get_particle, insert_particle


async def _entry(session: Any, *, mutability: Mutability = Mutability.MUTABLE) -> CorpusEntry:
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="LOCAL_MARKDOWN",
        uri_r=f"file:///tmp/{uuid.uuid4().hex}.md",
        fetch_policy=FetchPolicy.LAZY,
        mutability=mutability,
        deposited_by="test",
    )
    session.add(CorpusEntryRow.from_model(entry))
    await session.flush()
    return entry


async def _snapshot(
    session: Any,
    entry: CorpusEntry,
    *,
    age_days: int = 0,
    status: ExtractionStatus = ExtractionStatus.COMPLETE,
) -> Snapshot:
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC) - timedelta(days=age_days),
        content_hash=uuid.uuid4().hex * 2,
        warc_record_type=WarcRecordType.RESPONSE,
        extraction_status=status,
    )
    session.add(SnapshotRow.from_model(snap, entry.entry_id))
    await session.flush()
    return snap


async def _particle(
    session: Any, entry: CorpusEntry, snap: Snapshot, content: str, *, chunk_hash: str | None = None
) -> Particle:
    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry.entry_id,
                snapshot_id=snap.snapshot_id,
                chunk_hash=chunk_hash,
            )
        ],
    )
    await insert_particle(session, p)
    return p


class TestCascade:
    @pytest.mark.asyncio
    async def test_demotes_the_prior_generation(self, db_session: Any) -> None:
        """The motivating case: the old snapshot's claim stops being ACTIVE.

        Modelled on particle 20a27e4a — "pre-commit can be found by prefixing
        PATH=…" from a July 11 snapshot, still ACTIVE at confidence 1.0 beside
        the July 19 snapshot's "prepending PATH=… is forbidden by AGENTS.md".
        Both are truth-apt and neither logically contradicts the other, so the
        §6.6 probe correctly declined to choose; only the document generation
        says which is operative.
        """
        entry = await _entry(db_session)
        old = await _snapshot(db_session, entry, age_days=8)
        new = await _snapshot(db_session, entry)
        stale = await _particle(db_session, entry, old, "pre-commit is found by prefixing PATH=…")
        current = await _particle(db_session, entry, new, "prepending PATH=… is forbidden")
        await db_session.commit()

        demoted = await cascade_superseded_generation(
            db_session, entry_id=entry.entry_id, current_snapshot_id=new.snapshot_id
        )
        await db_session.commit()

        assert demoted == [stale.id]
        after = await get_particle(db_session, stale.id)
        assert after is not None
        assert after.status == Status.PROVENANCE_STALE
        assert after.status_reason == StatusReason.RETRACTED_DEPENDENCY
        # Demote, not delete: the claim is intact and reversible.
        assert after.content == stale.content
        assert after.confidence.value == pytest.approx(0.9)
        # The current generation is untouched.
        still = await get_particle(db_session, current.id)
        assert still is not None
        assert still.status == Status.ACTIVE

    @pytest.mark.asyncio
    async def test_carry_forward_ids_are_spared(self, db_session: Any) -> None:
        """The ordering property, asserted directly.

        A carry-forward keeps pointing at the snapshot it was first
        extracted from — provenance is deliberately not mutated — so without the
        exclusion it would read as a stale generation and be demoted. That would
        make the next re-extraction re-pay the LLM for text that never changed.
        """
        entry = await _entry(db_session)
        old = await _snapshot(db_session, entry, age_days=8)
        new = await _snapshot(db_session, entry)
        carried = await _particle(db_session, entry, old, "unchanged paragraph", chunk_hash="abc")
        dropped = await _particle(db_session, entry, old, "a claim the edit removed")
        await db_session.commit()

        demoted = await cascade_superseded_generation(
            db_session,
            entry_id=entry.entry_id,
            current_snapshot_id=new.snapshot_id,
            exclude_ids=frozenset({carried.id}),
        )
        await db_session.commit()

        assert demoted == [dropped.id]
        survivor = await get_particle(db_session, carried.id)
        assert survivor is not None
        assert survivor.status == Status.ACTIVE

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mutability", [Mutability.STABLE, Mutability.APPEND_ONLY, Mutability.EPHEMERAL]
    )
    async def test_no_op_for_non_mutable_entries(
        self, db_session: Any, mutability: Mutability
    ) -> None:
        """APPEND_ONLY content is additive; STABLE never changes. Neither retires."""
        entry = await _entry(db_session, mutability=mutability)
        old = await _snapshot(db_session, entry, age_days=8)
        new = await _snapshot(db_session, entry)
        p = await _particle(db_session, entry, old, "still true")
        await db_session.commit()

        assert (
            await cascade_superseded_generation(
                db_session, entry_id=entry.entry_id, current_snapshot_id=new.snapshot_id
            )
            == []
        )
        after = await get_particle(db_session, p.id)
        assert after is not None
        assert after.status == Status.ACTIVE

    @pytest.mark.asyncio
    async def test_single_snapshot_entry_is_untouched(self, db_session: Any) -> None:
        """A first extraction has no prior generation to retire."""
        entry = await _entry(db_session)
        only = await _snapshot(db_session, entry)
        p = await _particle(db_session, entry, only, "the first claim")
        await db_session.commit()

        assert (
            await cascade_superseded_generation(
                db_session, entry_id=entry.entry_id, current_snapshot_id=only.snapshot_id
            )
            == []
        )
        after = await get_particle(db_session, p.id)
        assert after is not None
        assert after.status == Status.ACTIVE

    @pytest.mark.asyncio
    async def test_is_idempotent(self, db_session: Any) -> None:
        """A nightly pass re-running must not thrash the status machine."""
        entry = await _entry(db_session)
        old = await _snapshot(db_session, entry, age_days=8)
        new = await _snapshot(db_session, entry)
        await _particle(db_session, entry, old, "retired claim")
        await db_session.commit()

        first = await cascade_superseded_generation(
            db_session, entry_id=entry.entry_id, current_snapshot_id=new.snapshot_id
        )
        await db_session.commit()
        second = await cascade_superseded_generation(
            db_session, entry_id=entry.entry_id, current_snapshot_id=new.snapshot_id
        )
        await db_session.commit()

        assert len(first) == 1
        assert second == []


class TestBackfill:
    @pytest.mark.asyncio
    async def test_dry_run_counts_without_writing(self, db_session: Any) -> None:
        entry = await _entry(db_session)
        old = await _snapshot(db_session, entry, age_days=8)
        await _snapshot(db_session, entry)
        p = await _particle(db_session, entry, old, "an old-generation claim")
        await db_session.commit()

        report = await backfill_superseded_generations(db_session, dry_run=True)

        assert report.entries_scanned == 1
        assert report.entries_affected == 1
        assert report.demoted == 1
        after = await get_particle(db_session, p.id)
        assert after is not None
        assert after.status == Status.ACTIVE  # nothing written

    @pytest.mark.asyncio
    async def test_apply_demotes_and_reports_per_entry(self, db_session: Any) -> None:
        entry = await _entry(db_session)
        old = await _snapshot(db_session, entry, age_days=8)
        await _snapshot(db_session, entry)
        p1 = await _particle(db_session, entry, old, "claim one")
        p2 = await _particle(db_session, entry, old, "claim two")
        await db_session.commit()

        report = await backfill_superseded_generations(db_session, dry_run=False)
        await db_session.commit()

        assert report.demoted == 2
        assert report.per_entry == [(entry.entry_id, entry.uri_r, 2)]
        for pid in (p1.id, p2.id):
            after = await get_particle(db_session, pid)
            assert after is not None
            assert after.status == Status.PROVENANCE_STALE

    @pytest.mark.asyncio
    async def test_skips_entries_whose_newest_snapshot_is_unextracted(
        self, db_session: Any
    ) -> None:
        """Never leave a hole: if the replacement beliefs do not exist yet, wait.

        A PENDING newest snapshot means the new generation has not been
        extracted. Retiring the old one now would leave the store with neither —
        so the entry is skipped and picked up once extraction lands.
        """
        entry = await _entry(db_session)
        old = await _snapshot(db_session, entry, age_days=8)
        await _snapshot(db_session, entry, status=ExtractionStatus.PENDING)
        p = await _particle(db_session, entry, old, "still the only generation extracted")
        await db_session.commit()

        report = await backfill_superseded_generations(db_session, dry_run=False)
        await db_session.commit()

        # The latest COMPLETE snapshot *is* `old`, so nothing is superseded yet.
        assert report.demoted == 0
        after = await get_particle(db_session, p.id)
        assert after is not None
        assert after.status == Status.ACTIVE
        assert old.extraction_status == ExtractionStatus.COMPLETE
