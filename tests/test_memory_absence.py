# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""A memory-directory walk records what it no longer finds.

The acceptance list of the record's § 8, through the real harvest, the absence
dispatch and the cascade, with no model call:

* which entries a walk may give an absence (§ 1) and the structural guards
  (§ 2): containment, the tags, ``MEMORY.md`` and the archive, an incomplete
  walk, an empty listing, a refused file, a listed-but-unread file;
* the judgement guards (§ 4): the first absence ever under a directory and a
  mass absence are refused and reported, ``--record`` confirms them, and the
  denominator leaves recorded absences out;
* processing (§ 3): a memory-file absence waits behind an unextracted memory
  file of its project, and a claim the merge target restates survives;
* the hook, pass 6, the verb, the SessionStart notice, ``hook doctor``, and the
  refresh ladder's count.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli._claude_code import walk_memory_dir
from particles.core.schema import (
    ExtractionStatus,
    Mutability,
    ProvenanceRef,
    ProvenanceRefType,
    WarcRecordType,
)
from particles.core.status import Status, StatusReason
from particles.corpus.deposit import deposit_text_versioned
from particles.corpus.refresh_outcome import RefreshOutcome, classify_refresh
from particles.corpus.store import (
    FileEntryRow,
    list_snapshots_for_entry,
    update_extraction_status,
)
from particles.operations.memory_absence import (
    MemoryWalk,
    preview_memory_absences,
    process_recorded_absences,
    qualifies,
    record_memory_absences,
    structural_skip,
    survey_memory_absences,
)
from particles.store.particle_store import get_particle
from tests._observer_scope import belief

PROJECT = "-p"
TAGS = ["claude-code", "memory-file", f"project:{PROJECT}"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def memdir(tmp_path: Path) -> Path:
    d = tmp_path / "projects" / PROJECT / "memory"
    d.mkdir(parents=True)
    return d


async def _harvest(
    session: Any,
    path: Path,
    text: str = "a fact",
    *,
    tags: list[str] | None = None,
    mutability: Mutability = Mutability.MUTABLE,
    extracted: bool = True,
) -> tuple[str, str]:
    """Write ``path`` and deposit it as the harvest would, extracted unless told not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    entry, snap, _ = await deposit_text_versioned(
        session,
        text=text,
        uri_r=path.resolve().as_uri(),
        source_type="LOCAL_MARKDOWN",
        mutability=mutability,
        tags=TAGS if tags is None else tags,
        deposited_by="claude-code-hook",
    )
    if extracted:
        await update_extraction_status(session, snap, ExtractionStatus.COMPLETE)
    await session.flush()
    return entry, snap


def _walk(memdir: Path, *, refused: int = 0) -> MemoryWalk:
    walk = walk_memory_dir(memdir)
    return MemoryWalk(
        directory=memdir.resolve(),
        project_key=PROJECT,
        listed=walk.listed_uris(),
        errors=walk.errors,
        refused_files=refused,
    )


async def _records(session: Any, entry_id: str) -> list[WarcRecordType]:
    return [s.warc_record_type for s in await list_snapshots_for_entry(session, entry_id)]


async def _confirmed_once(session: Any, memdir: Path) -> None:
    """Record one confirmed absence under ``memdir``, so the first-absence rule is past.

    Leaves ``anchor.md`` harvested and present, so the directory is never empty.
    """
    await _harvest(session, memdir / "anchor.md")
    gone = memdir / "confirmed-gone.md"
    await _harvest(session, gone)
    gone.unlink()
    report = await record_memory_absences(session, _walk(memdir), confirmed=True)
    assert len(report.absences) == 1


# ---------------------------------------------------------------------------
# § 1 who writes it, § 2 structural guards
# ---------------------------------------------------------------------------


def _row(uri: str, **kw: Any) -> FileEntryRow:
    return FileEntryRow(
        entry_id="e",
        uri_r=uri,
        source_type=kw.get("source_type", "LOCAL_MARKDOWN"),
        mutability=kw.get("mutability", Mutability.MUTABLE),
        latest_record_type=WarcRecordType.RESPONSE,
        last_response_id="s",
    )


class TestQualifies:
    def test_a_memory_file_of_the_project_qualifies(self, memdir: Path) -> None:
        uri = (memdir / "a.md").as_uri()
        assert qualifies(_row(uri), TAGS, directory=memdir, project_key=PROJECT)

    def test_the_projected_file_never_qualifies(self, memdir: Path) -> None:
        uri = (memdir / "MEMORY.md").as_uri()
        assert not qualifies(_row(uri), TAGS, directory=memdir, project_key=PROJECT)

    def test_the_archive_never_qualifies(self, memdir: Path) -> None:
        uri = (memdir / "a.md").as_uri()
        tags = [*TAGS, "memory-archive"]
        assert not qualifies(_row(uri), tags, directory=memdir, project_key=PROJECT)

    def test_a_sibling_directory_is_not_under_it(self, memdir: Path) -> None:
        uri = (memdir.parent / "memory-old" / "a.md").as_uri()
        assert not qualifies(_row(uri), TAGS, directory=memdir, project_key=PROJECT)

    @pytest.mark.parametrize(
        "tags",
        [["claude-code", f"project:{PROJECT}"], ["claude-code", "memory-file"], TAGS[:2]],
    )
    def test_both_harvest_tags_are_required(self, memdir: Path, tags: list[str]) -> None:
        uri = (memdir / "a.md").as_uri()
        assert not qualifies(_row(uri), tags, directory=memdir, project_key=PROJECT)

    def test_another_project_key_does_not_qualify(self, memdir: Path) -> None:
        uri = (memdir / "a.md").as_uri()
        assert not qualifies(_row(uri), TAGS, directory=memdir, project_key="-other")

    def test_stable_or_another_suffix_does_not_qualify(self, memdir: Path) -> None:
        uri = (memdir / "a.md").as_uri()
        stable = _row(uri, mutability=Mutability.STABLE)
        assert not qualifies(stable, TAGS, directory=memdir, project_key=PROJECT)
        txt = _row((memdir / "a.txt").as_uri())
        assert not qualifies(txt, TAGS, directory=memdir, project_key=PROJECT)


class TestStructuralSkip:
    def test_a_complete_listing_records(self, memdir: Path) -> None:
        (memdir / "a.md").write_text("x")
        assert structural_skip(_walk(memdir), enabled=True) is None

    def test_disabled(self, memdir: Path) -> None:
        (memdir / "a.md").write_text("x")
        assert structural_skip(_walk(memdir), enabled=False) == "disabled"

    def test_only_memory_md_is_an_empty_listing(self, memdir: Path) -> None:
        (memdir / "MEMORY.md").write_text("# Memory\n")
        assert structural_skip(_walk(memdir), enabled=True) == "empty_listing"

    def test_a_refused_file(self, memdir: Path) -> None:
        (memdir / "a.md").write_text("x")
        assert structural_skip(_walk(memdir, refused=1), enabled=True) == "file_refused"

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads an unreadable directory")
    def test_an_unreadable_subdirectory_makes_the_walk_incomplete(self, memdir: Path) -> None:
        (memdir / "a.md").write_text("x")
        locked = memdir / "sub"
        locked.mkdir()
        (locked / "b.md").write_text("y")
        locked.chmod(0)
        try:
            walk = walk_memory_dir(memdir)
            assert not walk.complete
            assert structural_skip(_walk(memdir), enabled=True) == "walk_incomplete"
        finally:
            locked.chmod(0o755)


@pytest.mark.asyncio
class TestRecord:
    async def test_a_deleted_file_gets_one_absence(self, db_session: Any, memdir: Path) -> None:
        await _confirmed_once(db_session, memdir)
        keep, _ = await _harvest(db_session, memdir / "keep.md")
        gone, _ = await _harvest(db_session, memdir / "gone.md")
        (memdir / "gone.md").unlink()
        report = await record_memory_absences(db_session, _walk(memdir))
        assert report.refusal is None
        assert len(report.absences) == 1
        assert (await _records(db_session, gone))[-1] is WarcRecordType.ABSENT
        assert (await _records(db_session, keep))[-1] is WarcRecordType.RESPONSE

    async def test_idempotent(self, db_session: Any, memdir: Path) -> None:
        await _confirmed_once(db_session, memdir)
        await _harvest(db_session, memdir / "keep.md")
        gone, _ = await _harvest(db_session, memdir / "gone.md")
        (memdir / "gone.md").unlink()
        await record_memory_absences(db_session, _walk(memdir))
        again = await record_memory_absences(db_session, _walk(memdir))
        assert again.absences == [] and again.missing == []
        assert (await _records(db_session, gone)).count(WarcRecordType.ABSENT) == 1

    async def test_exclusions_get_no_absence(self, db_session: Any, memdir: Path) -> None:
        await _confirmed_once(db_session, memdir)
        await _harvest(db_session, memdir / "keep.md")
        index, _ = await _harvest(db_session, memdir / "MEMORY.md")
        archive, _ = await _harvest(
            db_session, memdir / "archive.md", tags=[*TAGS, "memory-archive"]
        )
        untagged, _ = await _harvest(db_session, memdir / "untagged.md", tags=["claude-code"])
        other, _ = await _harvest(
            db_session, memdir / "other.md", tags=["claude-code", "memory-file", "project:-q"]
        )
        stable, _ = await _harvest(db_session, memdir / "stable.md", mutability=Mutability.STABLE)
        sibling, _ = await _harvest(db_session, memdir.parent / "memory-old" / "s.md")
        for name in ("MEMORY.md", "archive.md", "untagged.md", "other.md", "stable.md"):
            (memdir / name).unlink()
        (memdir.parent / "memory-old" / "s.md").unlink()
        report = await record_memory_absences(db_session, _walk(memdir))
        assert report.missing == []
        for entry in (index, archive, untagged, other, stable, sibling):
            assert WarcRecordType.ABSENT not in await _records(db_session, entry)

    async def test_a_listed_file_that_is_not_read_is_present(
        self, db_session: Any, memdir: Path
    ) -> None:
        """A symlink, an empty file, an unreadable one: listed, so never deleted (§ 1)."""
        await _confirmed_once(db_session, memdir)
        await _harvest(db_session, memdir / "keep.md")
        link, _ = await _harvest(db_session, memdir / "link.md")
        empty, _ = await _harvest(db_session, memdir / "empty.md")
        locked, _ = await _harvest(db_session, memdir / "locked.md")
        (memdir / "link.md").unlink()
        (memdir / "link.md").symlink_to(memdir / "keep.md")
        (memdir / "empty.md").write_text("")
        (memdir / "locked.md").chmod(0)
        try:
            report = await record_memory_absences(db_session, _walk(memdir))
        finally:
            (memdir / "locked.md").chmod(0o644)
        assert report.missing == []

    async def test_a_symlinked_directory_is_compared_by_its_target(
        self, db_session: Any, memdir: Path, tmp_path: Path
    ) -> None:
        await _confirmed_once(db_session, memdir)
        await _harvest(db_session, memdir / "keep.md")
        gone, _ = await _harvest(db_session, memdir / "gone.md")
        (memdir / "gone.md").unlink()
        link = tmp_path / "linked-memory"
        link.symlink_to(memdir)
        walk = walk_memory_dir(link)
        facts = MemoryWalk(directory=link.resolve(), project_key=PROJECT, listed=walk.listed_uris())
        report = await record_memory_absences(db_session, facts)
        assert [n.entry_id for n in report.missing] == [gone]
        assert len(report.absences) == 1

    async def test_an_incomplete_walk_records_nothing(self, db_session: Any, memdir: Path) -> None:
        await _confirmed_once(db_session, memdir)
        await _harvest(db_session, memdir / "keep.md")
        await _harvest(db_session, memdir / "gone.md")
        (memdir / "gone.md").unlink()
        facts = _walk(memdir)
        facts = MemoryWalk(
            directory=facts.directory,
            project_key=PROJECT,
            listed=facts.listed,
            errors=("sub: Permission denied",),
        )
        report = await record_memory_absences(db_session, facts)
        assert report.skipped == "walk_incomplete" and report.absences == []
        assert len(report.missing) == 1

    async def test_a_refused_file_records_nothing(self, db_session: Any, memdir: Path) -> None:
        await _confirmed_once(db_session, memdir)
        await _harvest(db_session, memdir / "keep.md")
        await _harvest(db_session, memdir / "gone.md")
        (memdir / "gone.md").unlink()
        report = await record_memory_absences(db_session, _walk(memdir, refused=1))
        assert report.skipped == "file_refused" and report.absences == []

    async def test_an_empty_listing_records_nothing(self, db_session: Any, memdir: Path) -> None:
        await _confirmed_once(db_session, memdir)
        await _harvest(db_session, memdir / "gone.md")
        (memdir / "gone.md").unlink()
        (memdir / "anchor.md").unlink()
        (memdir / "MEMORY.md").write_text("# Memory\n")
        report = await record_memory_absences(db_session, _walk(memdir))
        assert report.skipped == "empty_listing" and report.absences == []

    async def test_the_knob_off_records_nothing(
        self, db_session: Any, memdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _confirmed_once(db_session, memdir)
        await _harvest(db_session, memdir / "keep.md")
        await _harvest(db_session, memdir / "gone.md")
        (memdir / "gone.md").unlink()
        from particles.config import get_config

        monkeypatch.setattr(get_config().claude_code.harvest, "record_absence", False)
        report = await record_memory_absences(db_session, _walk(memdir))
        assert report.skipped == "disabled" and report.absences == []


# ---------------------------------------------------------------------------
# § 4 the judgement guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestJudgement:
    async def test_the_first_absence_is_refused_then_confirmed(
        self, db_session: Any, memdir: Path
    ) -> None:
        await _harvest(db_session, memdir / "keep.md")
        gone, _ = await _harvest(db_session, memdir / "gone.md")
        (memdir / "gone.md").unlink()
        refused = await record_memory_absences(db_session, _walk(memdir))
        assert refused.refusal == "first" and refused.absences == []
        assert refused.log_record()["absence_refused"]["reason"] == "first"
        confirmed = await record_memory_absences(db_session, _walk(memdir), confirmed=True)
        assert len(confirmed.absences) == 1
        # After that confirmation the hook records a later small deletion itself.
        later, _ = await _harvest(db_session, memdir / "later.md")
        (memdir / "later.md").unlink()
        hook = await record_memory_absences(db_session, _walk(memdir))
        assert [n.entry_id for n in hook.missing] == [later] and len(hook.absences) == 1
        assert WarcRecordType.ABSENT in await _records(db_session, gone)

    async def test_a_mass_absence_is_refused_and_reported_again(
        self, db_session: Any, memdir: Path
    ) -> None:
        await _confirmed_once(db_session, memdir)
        names = [f"f{i}.md" for i in range(8)]
        for name in names:
            await _harvest(db_session, memdir / name)
        for name in names[:6]:
            (memdir / name).unlink()
        first = await record_memory_absences(db_session, _walk(memdir))
        assert first.refusal == "mass" and first.absences == []
        assert first.previously_harvested == 9  # the eight and the anchor
        second = await record_memory_absences(db_session, _walk(memdir))
        assert second.refusal == "mass" and len(second.missing) == 6

    async def test_a_small_absence_is_recorded(self, db_session: Any, memdir: Path) -> None:
        await _confirmed_once(db_session, memdir)
        names = [f"f{i}.md" for i in range(8)]
        for name in names:
            await _harvest(db_session, memdir / name)
        for name in names[:2]:
            (memdir / name).unlink()
        report = await record_memory_absences(db_session, _walk(memdir))
        assert report.refusal is None and len(report.absences) == 2

    async def test_recorded_absences_leave_the_denominator(
        self, db_session: Any, memdir: Path
    ) -> None:
        """Twelve recorded absences do not dilute the test for a later mass deletion."""
        await _confirmed_once(db_session, memdir)
        for i in range(20):
            await _harvest(db_session, memdir / f"old{i}.md")
        for start in (0, 4, 8):  # three small deletions, each under the floor of five
            for i in range(start, start + 4):
                (memdir / f"old{i}.md").unlink()
            assert len((await record_memory_absences(db_session, _walk(memdir))).absences) == 4
        # Nine files remain. Five is over a quarter of nine, though under a quarter
        # of the 22 entries the directory ever held.
        for i in range(12, 17):
            (memdir / f"old{i}.md").unlink()
        report = await record_memory_absences(db_session, _walk(memdir))
        assert report.previously_harvested == 9
        assert report.refusal == "mass"

    async def test_a_directory_below_the_floor_is_guarded_only_by_the_first_rule(
        self, db_session: Any, memdir: Path
    ) -> None:
        await _confirmed_once(db_session, memdir)
        for i in range(4):
            await _harvest(db_session, memdir / f"s{i}.md")
        for i in range(3):
            (memdir / f"s{i}.md").unlink()
        report = await record_memory_absences(db_session, _walk(memdir))
        assert report.refusal is None and len(report.absences) == 3


# ---------------------------------------------------------------------------
# § 3 processing: the wait and what survives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProcessing:
    async def test_a_restated_claim_survives_and_a_dropped_one_retires(
        self, db_session: Any, memdir: Path
    ) -> None:
        """The merge: ``a`` deleted, ``b`` restates one of its claims."""
        await _confirmed_once(db_session, memdir)
        a, a_snap = await _harvest(db_session, memdir / "a.md", "claims X and Y")
        b, _ = await _harvest(db_session, memdir / "b.md", "claim Z")
        x = await belief(db_session, "X", (a, a_snap))
        y = await belief(db_session, "Y", (a, a_snap))
        # The session merged a into b: b's new generation restates X.
        _, b_new = await _harvest(db_session, memdir / "b.md", "claims Z and X", extracted=False)
        (memdir / "a.md").unlink()
        report = await record_memory_absences(db_session, _walk(memdir))
        assert len(report.absences) == 1

        # b is not extracted yet, so the absence waits.
        retired, holder = await process_recorded_absences(db_session, report)
        assert retired == [] and holder is not None and holder.snapshot_id == b_new

        # Extracting b folds X into the existing particle.
        from particles.store.particle_store import append_provenance_ref

        await append_provenance_ref(
            db_session,
            x.id,
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=b, snapshot_id=b_new),
        )
        await update_extraction_status(db_session, b_new, ExtractionStatus.COMPLETE)
        await db_session.commit()

        retired, holder = await process_recorded_absences(db_session, report)
        assert holder is None and retired == [y.id]
        kept = await get_particle(db_session, x.id)
        gone = await get_particle(db_session, y.id)
        assert kept is not None and kept.status is Status.ACTIVE
        assert gone is not None and gone.status is Status.PROVENANCE_STALE
        assert gone.status_reason is StatusReason.RETRACTED_DEPENDENCY

    async def test_the_pipeline_holds_the_absence_while_a_sibling_owes_extraction(
        self, db_session: Any, memdir: Path
    ) -> None:
        from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot

        await _confirmed_once(db_session, memdir)
        a, a_snap = await _harvest(db_session, memdir / "a.md", "claims X")
        await _harvest(db_session, memdir / "b.md", "claim Z")
        x = await belief(db_session, "X", (a, a_snap))
        _, b_new = await _harvest(db_session, memdir / "b.md", "Z, edited", extracted=False)
        (memdir / "a.md").unlink()
        report = await record_memory_absences(db_session, _walk(memdir))
        absence = report.absences[0]

        outcome = SnapshotOutcome()
        await extract_snapshot(db_session, a, absence, outcome_out=outcome)
        assert outcome.skipped == "waiting" and outcome.waiting_on == b_new
        assert outcome.waiting_on_memory_file and not outcome.waiting_on_failed
        assert (await get_particle(db_session, x.id)).status is Status.ACTIVE  # type: ignore[union-attr]

        await update_extraction_status(db_session, b_new, ExtractionStatus.COMPLETE)
        await db_session.commit()
        outcome = SnapshotOutcome()
        await extract_snapshot(db_session, a, absence, outcome_out=outcome)
        assert outcome.skipped is None and outcome.absence_retired == [x.id]

    async def test_a_failed_sibling_holds_the_absence_and_says_so(
        self, db_session: Any, memdir: Path
    ) -> None:
        from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot

        await _confirmed_once(db_session, memdir)
        a, _ = await _harvest(db_session, memdir / "a.md", "claims X")
        await _harvest(db_session, memdir / "b.md", "claim Z")
        _, b_new = await _harvest(db_session, memdir / "b.md", "Z, edited", extracted=False)
        await update_extraction_status(db_session, b_new, ExtractionStatus.FAILED)
        (memdir / "a.md").unlink()
        absence = (await record_memory_absences(db_session, _walk(memdir))).absences[0]
        for _ in range(2):  # still waiting on the next pass: a retry does not release it
            outcome = SnapshotOutcome()
            await extract_snapshot(db_session, a, absence, outcome_out=outcome)
            assert outcome.skipped == "waiting" and outcome.waiting_on_failed

    async def test_the_archives_failed_snapshot_does_not_hold_it(
        self, db_session: Any, memdir: Path, tmp_path: Path
    ) -> None:
        """The fold archive is append-only and no merge target (measured on the owner's store)."""
        from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot

        await _confirmed_once(db_session, memdir)
        a, _ = await _harvest(db_session, memdir / "a.md", "claims X")
        _, archived = await _harvest(
            db_session,
            tmp_path / "state" / "MEMORY.archive.md",
            "- [Folded](folded.md) — a folded pointer",
            tags=[*TAGS, "memory-archive"],
            mutability=Mutability.APPEND_ONLY,
            extracted=False,
        )
        await update_extraction_status(db_session, archived, ExtractionStatus.FAILED)
        (memdir / "a.md").unlink()
        absence = (await record_memory_absences(db_session, _walk(memdir))).absences[0]
        outcome = SnapshotOutcome()
        await extract_snapshot(db_session, a, absence, outcome_out=outcome)
        assert outcome.skipped is None and outcome.absence_retired == []

    async def test_another_projects_unextracted_file_does_not_hold_it(
        self, db_session: Any, memdir: Path, tmp_path: Path
    ) -> None:
        from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot

        await _confirmed_once(db_session, memdir)
        a, _ = await _harvest(db_session, memdir / "a.md", "claims X")
        await _harvest(db_session, memdir / "keep.md")
        await _harvest(
            db_session,
            tmp_path / "projects" / "-q" / "memory" / "q.md",
            tags=["claude-code", "memory-file", "project:-q"],
            extracted=False,
        )
        (memdir / "a.md").unlink()
        absence = (await record_memory_absences(db_session, _walk(memdir))).absences[0]
        outcome = SnapshotOutcome()
        await extract_snapshot(db_session, a, absence, outcome_out=outcome)
        assert outcome.skipped is None and outcome.absence_retired == []

    async def test_a_deleted_siblings_own_pending_snapshot_does_not_hold_it(
        self, db_session: Any, memdir: Path
    ) -> None:
        await _confirmed_once(db_session, memdir)
        await _harvest(db_session, memdir / "keep.md")
        await _harvest(db_session, memdir / "a.md")
        await _harvest(db_session, memdir / "b.md")
        await _harvest(db_session, memdir / "b.md", "b, edited", extracted=False)
        (memdir / "a.md").unlink()
        (memdir / "b.md").unlink()
        report = await record_memory_absences(db_session, _walk(memdir))
        assert len(report.absences) == 2
        _, holder = await process_recorded_absences(db_session, report)
        assert holder is None

    async def test_a_restore_before_processing_retires_nothing(
        self, db_session: Any, memdir: Path
    ) -> None:
        from particles.ingest.pending_collapse import collapse_superseded_pending

        await _confirmed_once(db_session, memdir)
        await _harvest(db_session, memdir / "keep.md")
        a, a_snap = await _harvest(db_session, memdir / "a.md", "claims X")
        x = await belief(db_session, "X", (a, a_snap))
        (memdir / "a.md").unlink()
        absence = (await record_memory_absences(db_session, _walk(memdir))).absences[0]
        await _harvest(db_session, memdir / "a.md", "claims X, restored", extracted=False)
        await collapse_superseded_pending(db_session)
        await db_session.commit()
        snaps = {s.snapshot_id: s for s in await list_snapshots_for_entry(db_session, a)}
        assert snaps[absence].extraction_status is ExtractionStatus.COMPLETE
        assert (await get_particle(db_session, x.id)).status is Status.ACTIVE  # type: ignore[union-attr]


@pytest.mark.asyncio
class TestPreview:
    async def test_the_dry_run_counts_what_record_retires(
        self, db_session: Any, memdir: Path
    ) -> None:
        await _harvest(db_session, memdir / "keep.md")
        a, a_snap = await _harvest(db_session, memdir / "a.md", "X and Y")
        b, b_snap = await _harvest(db_session, memdir / "b.md", "Y and W")
        keep_entry = (await survey_memory_absences(db_session, _walk(memdir))).previously_harvested
        assert keep_entry == 3
        x = await belief(db_session, "X", (a, a_snap))
        y = await belief(db_session, "Y", (a, a_snap), (b, b_snap))  # stated by both deleted
        await belief(db_session, "W", (b, b_snap))
        (memdir / "a.md").unlink()
        (memdir / "b.md").unlink()
        preview = await preview_memory_absences(db_session, _walk(memdir))
        assert preview.report.refusal == "first"
        assert x.id in preview.would_retire and y.id in preview.would_retire
        assert len(preview.would_retire) == 3

        report = await record_memory_absences(db_session, _walk(memdir), confirmed=True)
        retired, _ = await process_recorded_absences(db_session, report)
        assert sorted(retired) == sorted(preview.would_retire)

    async def test_the_dry_run_names_a_file_edited_after_the_deletion(
        self, db_session: Any, memdir: Path
    ) -> None:
        """A sibling captured after the deleted file's last snapshot still holds ``--record``."""
        await _harvest(db_session, memdir / "keep.md")
        await _harvest(db_session, memdir / "a.md", "claims X")
        _, edited = await _harvest(db_session, memdir / "keep.md", "kept, edited", extracted=False)
        (memdir / "a.md").unlink()
        preview = await preview_memory_absences(db_session, _walk(memdir))
        assert preview.holder is not None and preview.holder.snapshot_id == edited


# ---------------------------------------------------------------------------
# the refresh ladder
# ---------------------------------------------------------------------------


class TestRefreshOutcome:
    def test_missing_with_a_recorded_absence(self) -> None:
        assert classify_refresh("s", None, before_absent=True) is RefreshOutcome.ABSENT_RECORDED

    def test_missing_without_one(self) -> None:
        assert classify_refresh("s", None) is RefreshOutcome.MISSING


# ---------------------------------------------------------------------------
# the hook, the verb, the notice and the doctor, end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def hook_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_db: Path) -> Path:
    from tests._claude_projects import bind_hooks_to_store

    monkeypatch.setenv("HOME", str(tmp_path))
    bind_hooks_to_store(tmp_path, f"sqlite+aiosqlite:///{cli_db}")
    return tmp_path


def _project(home: Path) -> tuple[Path, Path]:
    project_dir = home / ".claude" / "projects" / PROJECT
    memory = project_dir / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    return project_dir, memory


def _session_end(runner: CliRunner, project_dir: Path, session: str) -> None:
    transcript = project_dir / f"{session}.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
    payload = json.dumps(
        {"session_id": session, "transcript_path": str(transcript), "cwd": "/some/project"}
    )
    result = runner.invoke(app, ["hook", "session-end", "--store", "default"], input=payload)
    assert result.exit_code == 0, result.output


def _session_start(runner: CliRunner, project_dir: Path) -> str:
    transcript = project_dir / "start.jsonl"
    transcript.write_text("")
    payload = json.dumps(
        {"session_id": "start", "transcript_path": str(transcript), "cwd": "/some/project"}
    )
    result = runner.invoke(app, ["hook", "session-start", "--store", "default"], input=payload)
    assert result.exit_code == 0, result.output
    return result.output


def _hook_log(home: Path) -> list[dict[str, Any]]:
    path = home / ".particles" / "claude-code" / "hooks.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _absent_uris() -> set[str]:
    async def _read() -> set[str]:
        from particles.corpus.store import list_entries
        from particles.db import session_scope

        async with session_scope() as session:
            entries = await list_entries(session, limit=500, source_type=None)
            out: set[str] = set()
            for e in entries:
                records = await _records(session, e.entry_id)
                if records and records[-1] is WarcRecordType.ABSENT:
                    out.add(e.uri_r or "")
            return out

    return asyncio.run(_read())


class TestHookEndToEnd:
    def test_first_deletion_is_reported_then_confirmed_then_small_ones_recorded(
        self, hook_home: Path
    ) -> None:
        runner = CliRunner()
        project_dir, memory = _project(hook_home)
        for name in ("a.md", "b.md", "c.md"):
            (memory / name).write_text(f"{name} says something\n")
        _session_end(runner, project_dir, "s1")

        (memory / "a.md").unlink()
        _session_end(runner, project_dir, "s2")
        record = _hook_log(hook_home)[-1]["memory_absences"]
        assert record["absence_refused"]["reason"] == "first"
        assert _absent_uris() == set()

        # The digest names it.
        assert "no longer found is not recorded" in _session_start(runner, project_dir)
        doctor = runner.invoke(app, ["hook", "doctor", "--store", "default"])
        assert "memory absences: ⚠ 1 file(s)" in doctor.output

        dry = runner.invoke(app, ["memory", "absences", "--json"])
        assert dry.exit_code == 0, dry.output
        listed = json.loads(dry.output)["directories"][0]
        assert listed["hook_refusal"] == "first" and len(listed["missing"]) == 1
        assert _absent_uris() == set()

        confirmed = runner.invoke(app, ["memory", "absences", "--record"])
        assert confirmed.exit_code == 0, confirmed.output
        assert "recorded 1 absence(s)" in confirmed.output
        assert _absent_uris() == {(memory / "a.md").resolve().as_uri()}
        assert "no longer found" not in _session_start(runner, project_dir)

        (memory / "b.md").unlink()
        _session_end(runner, project_dir, "s3")
        assert _hook_log(hook_home)[-1]["memory_absences"]["recorded"] == 1
        assert (memory / "b.md").resolve().as_uri() in _absent_uris()

    def test_a_remote_backend_records_nothing_and_says_so(
        self, hook_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli import hook

        _, memory = _project(hook_home)
        (memory / "a.md").write_text("a\n")

        class _Remote:
            remote = True

        monkeypatch.setattr("particles.api.client.get_backend", lambda: _Remote())
        record = asyncio.run(
            hook._record_memory_absences(
                "default", memory, PROJECT, walk_memory_dir(memory), refused_files=0
            )
        )
        assert record == {"skipped": "remote"}

    def test_pass_six_refuses_what_the_hook_refuses(
        self, hook_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli.memory import build_projection_runner
        from particles.config import get_config

        async def _no_render(*_a: Any, **_kw: Any) -> dict[str, Any]:
            return {}

        monkeypatch.setattr(get_config().agent_memory.projection, "enabled", True)
        monkeypatch.setattr("particles.api.cli._claude_code.projection_enabled", lambda: True)
        monkeypatch.setattr("particles.api.cli._memory_projection.run_projection_cycle", _no_render)
        runner = CliRunner()
        project_dir, memory = _project(hook_home)
        for name in ("a.md", "b.md"):
            (memory / name).write_text(f"{name}\n")
        _session_end(runner, project_dir, "s1")
        (memory / "a.md").unlink()
        run_pass, reason = build_projection_runner("default")
        assert run_pass is not None, reason
        telemetry = asyncio.run(run_pass())
        assert telemetry["absences_unrecorded"] == 1 and telemetry["absences_recorded"] == 0
        assert _absent_uris() == set()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads an unreadable directory")
    def test_the_verb_skips_an_unreadable_directory_with_exit_1(self, hook_home: Path) -> None:
        runner = CliRunner()
        project_dir, memory = _project(hook_home)
        (memory / "a.md").write_text("a\n")
        _session_end(runner, project_dir, "s1")
        (memory / "a.md").unlink()
        (memory / "keep.md").write_text("keep\n")
        locked = memory / "sub"
        locked.mkdir()
        locked.chmod(0)
        try:
            result = runner.invoke(app, ["memory", "absences", "--record"])
        finally:
            locked.chmod(0o755)
        assert result.exit_code == 1, result.output
        assert _absent_uris() == set()

    def test_the_verb_refuses_a_directory_claude_code_does_not_read(
        self, hook_home: Path, tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / "notes"
        elsewhere.mkdir()
        result = CliRunner().invoke(app, ["memory", "absences", "--memory-dir", str(elsewhere)])
        assert result.exit_code == 2
