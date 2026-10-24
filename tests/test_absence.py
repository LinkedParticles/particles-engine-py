# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The ``ABSENT`` corpus record and every reader of an entry's current generation.

An ``ABSENT`` record says a completed vault walk looked for the note and did
not find it. Each reader the ADR's § 3.4 table names is pinned here, because a
reader that bypassed ``get_entry_currency`` would read a deleted note as still
current:

* the record itself: no blob, ``refers_to`` the last RESPONSE, the empty-body
  digest, never matched by a content-hash lookup, never matched by the
  unchanged-skip;
* the currency helper and ``attests``, now and as of an instant;
* the collapse in both directions, with the blob exemption;
* processing: the cascade with snapshot-less edges, ``COMPLETE``, no model call,
  and the keep for a claim another entry still states;
* ``reindex``, ``extract --snapshot-id``, ``stated_by``, the backfill, and the
  disclosure pass's ``stopped_by``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.core.schema import (
    ExtractionStatus,
    FetchPolicy,
    Mutability,
    ProvenanceRef,
    ProvenanceRefType,
    WarcRecordType,
)
from particles.core.status import Status, StatusReason
from particles.corpus.deposit import (
    EMPTY_BODY_SHA256,
    deposit_text_versioned,
    record_absence,
    save_blob,
    sha256,
    write_entry_and_snapshot,
)
from particles.corpus.store import (
    EntryCurrency,
    GenerationRow,
    SnapshotRow,
    get_entry_by_content_hash,
    get_entry_currency,
    get_snapshot,
    list_absent_entry_ids,
    list_snapshots_for_entry,
    snapshot_exists_by_hash,
    update_extraction_status,
)
from particles.ingest.absence import absences_last, process_absence
from particles.ingest.pending_collapse import collapse_superseded_pending, plan_collapse
from particles.store.observer_scope_join import SourceRef, attests
from particles.store.particle_store import get_particle
from tests._observer_scope import belief, generation

NOW = datetime(2026, 10, 21, tzinfo=UTC)


async def _absent(session: Any, entry_id: str, *, complete: bool = True) -> str:
    snaps = await list_snapshots_for_entry(session, entry_id)
    last = [s for s in snaps if s.warc_record_type is WarcRecordType.RESPONSE][-1]
    absence = await record_absence(session, entry_id=entry_id, refers_to=last.snapshot_id)
    if complete:
        await update_extraction_status(session, absence, ExtractionStatus.COMPLETE)
    await session.flush()
    return absence


# ---------------------------------------------------------------------------
# The record (§ 3.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecord:
    async def test_shape(self, db_session: Any) -> None:
        entry, snap = await generation(db_session, "note.md", "a note", [])
        absence_id = await _absent(db_session, entry, complete=False)
        absence = await get_snapshot(db_session, absence_id)
        assert absence is not None
        assert absence.warc_record_type is WarcRecordType.ABSENT
        assert absence.archive_path is None
        assert absence.refers_to == snap
        assert absence.content_hash == EMPTY_BODY_SHA256 == sha256(b"")
        assert absence.extraction_status is ExtractionStatus.PENDING

    async def test_no_content_hash_lookup_matches_an_absence(self, db_session: Any) -> None:
        entry, _ = await generation(db_session, "note.md", "a note", [])
        await _absent(db_session, entry)
        assert await get_entry_by_content_hash(db_session, EMPTY_BODY_SHA256) is None
        assert not await snapshot_exists_by_hash(db_session, EMPTY_BODY_SHA256)

    async def test_an_empty_file_never_deduplicates_onto_an_absence(self, db_session: Any) -> None:
        entry, _ = await generation(db_session, "note.md", "a note", [])
        await _absent(db_session, entry)
        empty_entry, _ = await write_entry_and_snapshot(
            db_session,
            uri_r="file:///elsewhere/empty.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.STABLE,
            fetch_policy=FetchPolicy.NEVER,
            content=b"",
            content_hash=EMPTY_BODY_SHA256,
            archive_path=save_blob(b"", EMPTY_BODY_SHA256),
            deposited_by="test",
            tags=[],
            warc_record_type=WarcRecordType.RESPONSE,
        )
        assert empty_entry != entry

    async def test_an_absent_latest_record_matches_no_deposit(self, db_session: Any) -> None:
        """§ 5: the unchanged-skip compares against the latest record, and an absence is none."""
        entry, _ = await generation(db_session, "note.md", "", [])
        await _absent(db_session, entry)
        unchanged: list[bool] = []
        again, snap = await write_entry_and_snapshot(
            db_session,
            uri_r="test://note.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.NEVER,
            content=b"",
            content_hash=EMPTY_BODY_SHA256,
            archive_path=save_blob(b"", EMPTY_BODY_SHA256),
            deposited_by="test",
            tags=[],
            warc_record_type=WarcRecordType.RESPONSE,
            out_unchanged=unchanged,
        )
        assert again == entry and unchanged == [False]
        assert (await get_snapshot(db_session, snap)).warc_record_type is WarcRecordType.RESPONSE  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# The current generation (§ 3.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCurrency:
    async def test_a_processed_absence_is_the_current_generation(self, db_session: Any) -> None:
        entry, _ = await generation(db_session, "note.md", "a note", [])
        absence = await _absent(db_session, entry)
        currency = (await get_entry_currency(db_session, [entry]))[entry]
        assert currency.absent and currency.latest_snapshot_id == absence

    async def test_a_pending_absence_is_not_yet(self, db_session: Any) -> None:
        entry, snap = await generation(db_session, "note.md", "a note", [])
        await _absent(db_session, entry, complete=False)
        currency = (await get_entry_currency(db_session, [entry]))[entry]
        assert not currency.absent and currency.latest_snapshot_id == snap
        assert entry in await list_absent_entry_ids(db_session, [entry])

    async def test_as_of_before_the_absence_reads_the_note(self, db_session: Any) -> None:
        entry, snap = await generation(
            db_session, "note.md", "a note", [], captured_at=NOW - timedelta(days=2)
        )
        absence = await _absent(db_session, entry)
        row = await db_session.get(SnapshotRow, absence)
        row.captured_at = NOW
        await db_session.flush()
        before = (await get_entry_currency(db_session, [entry], as_of=NOW - timedelta(days=1)))[
            entry
        ]
        assert not before.absent and before.latest_snapshot_id == snap
        after = (await get_entry_currency(db_session, [entry], as_of=NOW + timedelta(days=1)))[
            entry
        ]
        assert after.absent

    async def test_a_collapsed_absence_is_never_current(self, db_session: Any) -> None:
        entry, snap = await generation(db_session, "note.md", "a note", [])
        absence = await _absent(db_session, entry, complete=False)
        row = await db_session.get(SnapshotRow, absence)
        row.extraction_status = ExtractionStatus.COMPLETE.value
        row.superseded_by_snapshot_id = snap
        await db_session.flush()
        currency = (await get_entry_currency(db_session, [entry]))[entry]
        assert not currency.absent and currency.latest_snapshot_id == snap
        assert entry not in await list_absent_entry_ids(db_session, [entry])

    def test_attests_nothing_through_an_absent_entry(self) -> None:
        absent = EntryCurrency(tags=[], mutable=True, latest_snapshot_id="abs", absent=True)
        assert not attests(SourceRef("e", "s1"), absent)
        assert not attests(SourceRef("e", None), absent)
        # The first two early returns still hold.
        assert attests(SourceRef("e", "s1"), None)
        stable = EntryCurrency(tags=[], mutable=False, latest_snapshot_id=None)
        assert attests(SourceRef("e", "s1"), stable)
        # And a living entry attests through its current snapshot only.
        live = EntryCurrency(tags=[], mutable=True, latest_snapshot_id="s2")
        assert attests(SourceRef("e", "s2"), live) and not attests(SourceRef("e", "s1"), live)


# ---------------------------------------------------------------------------
# The collapse (§ 3.4)
# ---------------------------------------------------------------------------


def _row(
    sid: str, minute: int, status: ExtractionStatus, *, absent: bool = False, h: str = ""
) -> GenerationRow:
    return GenerationRow(
        entry_id="e",
        snapshot_id=sid,
        captured_at=NOW + timedelta(minutes=minute),
        extraction_status=status,
        content_hash=h or sid,
        collapsed=False,
        absent=absent,
    )


class TestCollapse:
    def test_a_pending_absence_collapses_older_unextracted_responses(self) -> None:
        rows = [
            _row("done", 0, ExtractionStatus.COMPLETE),
            _row("edit", 1, ExtractionStatus.PENDING),
            _row("gone", 2, ExtractionStatus.PENDING, absent=True),
        ]
        # The absence has no blob: it is exempt from the presence rule.
        assert plan_collapse(rows, present_blobs={"done", "edit"}) == [("edit", "gone")]

    def test_a_newer_response_collapses_a_pending_absence(self) -> None:
        rows = [
            _row("done", 0, ExtractionStatus.COMPLETE),
            _row("gone", 1, ExtractionStatus.PENDING, absent=True),
            _row("back", 2, ExtractionStatus.PENDING),
        ]
        assert plan_collapse(rows, present_blobs={"done", "back"}) == [("gone", "back")]

    def test_an_absence_never_collapses_a_complete_response(self) -> None:
        rows = [
            _row("done", 0, ExtractionStatus.COMPLETE),
            _row("gone", 1, ExtractionStatus.PENDING, absent=True),
        ]
        assert plan_collapse(rows, present_blobs={"done"}) == []

    def test_absences_run_after_the_response_work(self) -> None:
        assert absences_last(["a", "x", "b", "y"], [False, True, False, True]) == [
            "a",
            "b",
            "x",
            "y",
        ]

    @pytest.mark.asyncio
    async def test_collapse_reads_absences_from_the_store(self, db_session: Any) -> None:
        entry, _ = await generation(db_session, "note.md", "a note", [])
        await deposit_text_versioned(
            db_session,
            text="an edit",
            uri_r="test://note.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        absence = await _absent(db_session, entry, complete=False)
        await db_session.commit()
        report = await collapse_superseded_pending(db_session)
        assert [superseder for _, _, superseder in report.collapsed] == [absence]


# ---------------------------------------------------------------------------
# Processing (§ 3.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProcessing:
    async def test_retires_every_claim_including_snapshotless_refs(self, db_session: Any) -> None:
        entry, snap = await generation(db_session, "note.md", "a note", [])
        anchored = await belief(db_session, "anchored to the note", (entry, snap))
        loose = await belief(db_session, "a ref with no snapshot")
        from particles.store.particle_store import append_provenance_ref

        await append_provenance_ref(
            db_session,
            loose.id,
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry),
        )
        absence = await _absent(db_session, entry, complete=False)
        await db_session.commit()

        retired = await process_absence(db_session, entry_id=entry, snapshot_id=absence)

        assert set(retired) == {anchored.id, loose.id}
        for pid in (anchored.id, loose.id):
            p = await get_particle(db_session, pid)
            assert p is not None
            assert (p.status, p.status_reason) == (
                Status.PROVENANCE_STALE,
                StatusReason.RETRACTED_DEPENDENCY,
            )
        assert (await get_snapshot(db_session, absence)).extraction_status is (  # type: ignore[union-attr]
            ExtractionStatus.COMPLETE
        )

    async def test_keeps_a_claim_another_entry_still_states(self, db_session: Any) -> None:
        gone, gone_snap = await generation(db_session, "gone.md", "one", [])
        kept, kept_snap = await generation(db_session, "kept.md", "two", [])
        shared = await belief(db_session, "said by both", (gone, gone_snap), (kept, kept_snap))
        absence = await _absent(db_session, gone, complete=False)
        await db_session.commit()
        assert await process_absence(db_session, entry_id=gone, snapshot_id=absence) == []
        assert (await get_particle(db_session, shared.id)).status is Status.ACTIVE  # type: ignore[union-attr]

    async def test_no_claim_survives_a_second_absent_source(self, db_session: Any) -> None:
        """A claim two deleted notes stated retires once neither attests it."""
        a, a_snap = await generation(db_session, "a.md", "one", [])
        b, b_snap = await generation(db_session, "b.md", "two", [])
        shared = await belief(db_session, "said by both", (a, a_snap), (b, b_snap))
        await _absent(db_session, b)  # b already processed
        absence = await _absent(db_session, a, complete=False)
        await db_session.commit()
        assert await process_absence(db_session, entry_id=a, snapshot_id=absence) == [shared.id]

    async def test_backfill_processes_an_absent_current_generation(self, db_session: Any) -> None:
        from particles.ingest.generation import backfill_superseded_generations

        entry, snap = await generation(db_session, "note.md", "a note", [])
        claim = await belief(db_session, "still active", (entry, snap))
        await _absent(db_session, entry)  # processed, but the claim predates the cascade
        await db_session.commit()
        preview = await backfill_superseded_generations(db_session, dry_run=True)
        assert preview.demoted == 1
        await backfill_superseded_generations(db_session, dry_run=False)
        assert (await get_particle(db_session, claim.id)).status is Status.PROVENANCE_STALE  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# The other readers in the § 3.4 table
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReaders:
    async def test_reindex_skips_an_absent_entry_and_lists_it(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations import reindex as reindex_mod

        entry, snap = await generation(db_session, "note.md", "a note", [])
        claim = await belief(db_session, "a claim", (entry, snap))
        absence = await _absent(db_session, entry, complete=False)
        await db_session.commit()
        await process_absence(db_session, entry_id=entry, snapshot_id=absence)

        async def boom(*args: Any, **kwargs: Any) -> list[Any]:
            raise AssertionError("an absent entry was re-extracted")

        monkeypatch.setattr(reindex_mod, "extract_snapshot", boom)
        summary = await reindex_mod.reindex(db_session, entry_ids=[entry], run_post_lint=False)
        plan = summary["plan"]
        assert isinstance(plan, dict)
        assert plan["absent_entries"] == [entry] and summary["scope"] == 0
        assert (await get_particle(db_session, claim.id)).status is Status.PROVENANCE_STALE  # type: ignore[union-attr]

    async def test_reindex_auto_discovery_processes_a_pending_absence_last(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations import reindex as reindex_mod

        gone, _ = await generation(db_session, "gone.md", "one", [])
        absence = await _absent(db_session, gone, complete=False)
        pending_entry, pending_snap, _ = await deposit_text_versioned(
            db_session,
            text="new",
            uri_r="test://new.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.STABLE,
        )
        await db_session.commit()
        resolved = await reindex_mod.resolve_reindex_work(db_session, dry_run=True)
        order = [sid for _, sids in resolved.work for sid in sids]
        assert order == [pending_snap, absence]
        assert resolved.plan.absent_entries == []

    async def test_stated_by_credits_no_claim_after_an_absence(self, db_session: Any) -> None:
        from particles.operations.query.as_of import load_as_of_view
        from particles.operations.utility_exposure import stated_by

        entry, snap = await generation(
            db_session, "note.md", "a note", [], captured_at=NOW - timedelta(days=2)
        )
        claim = await belief(db_session, "a claim", (entry, snap))
        absence = await _absent(db_session, entry)
        row = await db_session.get(SnapshotRow, absence)
        row.captured_at = NOW
        await db_session.flush()
        before = await stated_by(
            db_session, [entry], await load_as_of_view(db_session, NOW - timedelta(days=1))
        )
        assert before == {claim.id: {entry}}
        after = await stated_by(
            db_session, [entry], await load_as_of_view(db_session, NOW + timedelta(days=1))
        )
        assert after == {}

    async def test_disclosure_names_the_absence_that_stopped_a_claim(self, db_session: Any) -> None:
        from particles.core.contradiction_disclosure import ConfirmedPair
        from particles.operations.contradiction_disclosure import run_disclosure

        n1 = await generation(db_session, "sept18.md", "exists", [], captured_at=NOW)
        n2 = await generation(db_session, "sept25.md", "404s", [], captured_at=NOW)
        a = await belief(db_session, "The endpoint exists", n1)
        b = await belief(db_session, "The endpoint returns not found", n2)
        await db_session.commit()
        await run_disclosure(
            db_session,
            confirmed=[ConfirmedPair(a=a.id, b=b.id, same_source=False, reason="disagree")],
            mint=True,
        )
        absence = await _absent(db_session, n2[0])
        await db_session.commit()
        report = await run_disclosure(db_session, confirmed=[], mint=False)
        [closed] = report.closed
        [dropped] = closed["dropped"]
        assert dropped["particle_id"] == b.id and dropped["state"] == "lapsed"
        assert dropped["stopped_by"] == [absence]


# ---------------------------------------------------------------------------
# extract --snapshot-id on an absent entry
# ---------------------------------------------------------------------------


class TestExtractRefusal:
    def test_naming_an_older_snapshot_of_an_absent_entry_needs_force(
        self, tmp_path: Path, cli_db: Path
    ) -> None:
        import asyncio

        from particles.api.cli import app
        from particles.db import session_scope

        async def seed() -> tuple[str, str]:
            async with session_scope() as session:
                entry, snap = await generation(session, "note.md", "a note", [])
                await _absent(session, entry)
                await session.commit()
                return entry, snap

        entry, snap = asyncio.run(seed())
        runner = CliRunner()
        refused = runner.invoke(app, ["extract", entry, "--snapshot-id", snap])
        assert refused.exit_code == 1
        assert "no longer found" in refused.output and "--force" in refused.output
