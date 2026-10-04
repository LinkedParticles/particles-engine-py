# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""a vault export prunes and overwrites only the notes it wrote.

``export obsidian <vault>`` used to rglob every ``.md`` under its target and
unlink each one this run had not written, so pointing it at a real vault
deleted the user's notes; the Logseq exporter did the same over ``pages/``.
These tests pin the ownership ledger that replaced that, and the end-to-end
property the PDR asked for: a foreign note survives two consecutive exports.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from particles.core.schema import (
    CalibrationSource,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Status,
    Subject,
    UncertaintyNature,
)
from particles.db import session_scope
from particles.exporters.logseq import LogseqExporter
from particles.exporters.obsidian import ObsidianExporter
from particles.render.markdown import (
    EXPORT_JOURNAL_NAME,
    EXPORT_MANIFEST_NAME,
    ExportTargetNotOwnedError,
    MarkdownExportLedger,
)

# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def _seed_users_notes(root: Path) -> tuple[Path, Path]:
    """The PDR's reproduction: a vault holding a note and a journal entry."""
    ideas = root / "Ideas.md"
    journal = root / "Journal" / "2026-10-01.md"
    journal.parent.mkdir(parents=True)
    ideas.write_text("my ideas\n")
    journal.write_text("today\n")
    (root / ".obsidian").mkdir()
    return ideas, journal


class TestLedgerClaim:
    def test_empty_or_missing_target_is_claimed(self, tmp_path: Path) -> None:
        ledger = MarkdownExportLedger.claim(tmp_path / "new", exporter="obsidian", recursive=True)
        assert ledger.owned == set()

    def test_populated_target_without_manifest_is_refused(self, tmp_path: Path) -> None:
        _seed_users_notes(tmp_path)
        with pytest.raises(ExportTargetNotOwnedError) as excinfo:
            MarkdownExportLedger.claim(tmp_path, exporter="obsidian", recursive=True)
        message = str(excinfo.value)
        assert "2 Markdown file(s)" in message
        assert "Ideas.md" in message
        assert "--force" in message
        # Refusal writes nothing.
        assert not (tmp_path / EXPORT_MANIFEST_NAME).exists()

    def test_refusal_is_a_value_error(self) -> None:
        # Every export surface reports a ValueError as a usage error.
        assert issubclass(ExportTargetNotOwnedError, ValueError)

    def test_force_claims_a_populated_target(self, tmp_path: Path) -> None:
        _seed_users_notes(tmp_path)
        ledger = MarkdownExportLedger.claim(
            tmp_path, exporter="obsidian", recursive=True, force=True
        )
        assert ledger.owned == set()

    def test_manifest_present_means_no_refusal(self, tmp_path: Path) -> None:
        ideas, _ = _seed_users_notes(tmp_path)
        first = MarkdownExportLedger.claim(
            tmp_path, exporter="obsidian", recursive=True, force=True
        )
        first.write(tmp_path / "Coin.md", "coin\n")
        first.finish()
        # A note the user adds later does not make the next run refuse.
        later = tmp_path / "Later.md"
        later.write_text("added after the first export\n")
        second = MarkdownExportLedger.claim(tmp_path, exporter="obsidian", recursive=True)
        assert second.owned == {(tmp_path / "Coin.md").resolve()}
        assert ideas.resolve() not in second.owned

    def test_non_recursive_scope_ignores_subdirectories(self, tmp_path: Path) -> None:
        pages = tmp_path / "pages"
        (tmp_path / "journals").mkdir()
        (tmp_path / "journals" / "2026_10_01.md").write_text("- today\n")
        pages.mkdir()
        MarkdownExportLedger.claim(tmp_path, exporter="logseq", scope=pages, recursive=False)

    def test_legacy_marker_adopts_an_earlier_export(self, tmp_path: Path) -> None:
        old = tmp_path / "Old.md"
        old.write_text("---\ntags:\n  - particles/subject\n---\n# Old\n")
        ledger = MarkdownExportLedger.claim(
            tmp_path,
            exporter="obsidian",
            recursive=True,
            legacy_marker=lambda text: "particles/" in text,
        )
        assert ledger.owned == {old.resolve()}

    def test_other_exporters_manifest_grants_nothing(self, tmp_path: Path) -> None:
        note = tmp_path / "A.md"
        note.write_text("a\n")
        (tmp_path / EXPORT_MANIFEST_NAME).write_text(
            json.dumps({"exporter": "logseq", "version": 1, "files": ["A.md"]})
        )
        with pytest.raises(ExportTargetNotOwnedError):
            MarkdownExportLedger.claim(tmp_path, exporter="obsidian", recursive=True)

    def test_manifest_entries_outside_scope_are_ignored(self, tmp_path: Path) -> None:
        root = tmp_path / "vault"
        root.mkdir()
        outside = tmp_path / "precious.md"
        outside.write_text("not the export's\n")
        (root / EXPORT_MANIFEST_NAME).write_text(
            json.dumps({"exporter": "obsidian", "version": 1, "files": ["../precious.md", "x.txt"]})
        )
        ledger = MarkdownExportLedger.claim(root, exporter="obsidian", recursive=True)
        assert ledger.owned == set()
        ledger.finish()
        assert outside.exists()


class TestLedgerWriteAndFinish:
    def test_foreign_file_at_a_written_path_is_skipped(self, tmp_path: Path) -> None:
        ideas, _ = _seed_users_notes(tmp_path)
        ledger = MarkdownExportLedger.claim(
            tmp_path, exporter="obsidian", recursive=True, force=True
        )
        ledger.force = False  # the skip rule as a later, unforced run sees it
        assert ledger.write(ideas, "export body\n") is False
        assert ideas.read_text() == "my ideas\n"
        assert ledger.skipped == [ideas.resolve()]

    def test_force_overwrites_a_foreign_file_and_takes_ownership(self, tmp_path: Path) -> None:
        ideas, _ = _seed_users_notes(tmp_path)
        ledger = MarkdownExportLedger.claim(
            tmp_path, exporter="obsidian", recursive=True, force=True
        )
        assert ledger.write(ideas, "export body\n") is True
        ledger.finish()
        assert ideas.read_text() == "export body\n"
        manifest = json.loads((tmp_path / EXPORT_MANIFEST_NAME).read_text())
        assert manifest == {"exporter": "obsidian", "version": 1, "files": ["Ideas.md"]}

    def test_finish_prunes_only_owned_files_and_saves_the_manifest(self, tmp_path: Path) -> None:
        ideas, journal = _seed_users_notes(tmp_path)
        first = MarkdownExportLedger.claim(
            tmp_path, exporter="obsidian", recursive=True, force=True
        )
        first.write(tmp_path / "Kept.md", "kept\n")
        first.write(tmp_path / "shard" / "Gone.md", "gone\n")
        assert first.finish() == 0

        second = MarkdownExportLedger.claim(tmp_path, exporter="obsidian", recursive=True)
        second.write(tmp_path / "Kept.md", "kept\n")
        assert second.finish() == 1
        assert not (tmp_path / "shard" / "Gone.md").exists()
        assert not (tmp_path / "shard").exists()  # emptied by the prune
        assert ideas.exists() and journal.exists()
        assert (tmp_path / ".obsidian").is_dir()
        assert not (tmp_path / EXPORT_JOURNAL_NAME).exists()

    def test_interrupted_run_still_owns_what_it_wrote(self, tmp_path: Path) -> None:
        _seed_users_notes(tmp_path)
        first = MarkdownExportLedger.claim(
            tmp_path, exporter="obsidian", recursive=True, force=True
        )
        first.write(tmp_path / "Half.md", "written before the crash\n")
        # No finish(): the process died. The journal survives it.
        assert (tmp_path / EXPORT_JOURNAL_NAME).read_text() == "Half.md\n"

        second = MarkdownExportLedger.claim(tmp_path, exporter="obsidian", recursive=True)
        assert (tmp_path / "Half.md").resolve() in second.owned
        assert second.write(tmp_path / "Half.md", "rewritten\n") is True


# ---------------------------------------------------------------------------
# End to end: a foreign note survives two consecutive exports
# ---------------------------------------------------------------------------


def _particle(content: str) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
        asserted_by="stub-extractor",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        extractor_ref={"name": "stub-extractor", "version": "0.1.0"},
        subject_ids=[],
    )


async def _seed_subject(name: str) -> None:
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject, link_particle_to_subjects

    subject = Subject(id=str(uuid.uuid4()), canonical_name=name, asserted_by="test")
    particle = _particle(f"{name} is a test subject")
    async with session_scope() as session:
        await insert_subject(session, subject)
        await insert_particle(session, particle)
        await link_particle_to_subjects(session, particle.id, [subject.id])
        await session.commit()


class TestObsidianOwnership:
    @pytest.mark.asyncio
    async def test_users_notes_survive_two_exports(
        self, db_session: object, tmp_path: Path
    ) -> None:
        await _seed_subject("Pfennig")
        ideas, journal = _seed_users_notes(tmp_path)

        async with session_scope() as session:
            with pytest.raises(ExportTargetNotOwnedError):
                await ObsidianExporter().export(session, tmp_path, min_links=0)
        assert not (tmp_path / "Pfennig.md").exists()  # refused before any write

        async with session_scope() as session:
            await ObsidianExporter().export(session, tmp_path, min_links=0, force=True)
        assert (tmp_path / "Pfennig.md").exists()

        # Second run: no --force needed, and a suppressed subject's note is
        # still pruned while the user's notes stay.
        async with session_scope() as session:
            await ObsidianExporter().export(session, tmp_path, min_links=0, min_particles=99)
        assert not (tmp_path / "Pfennig.md").exists()
        assert ideas.read_text() == "my ideas\n"
        assert journal.read_text() == "today\n"
        assert (tmp_path / "_index.md").exists()

    @pytest.mark.asyncio
    async def test_users_note_at_a_subjects_path_is_not_overwritten(
        self, db_session: object, tmp_path: Path
    ) -> None:
        await _seed_subject("Pfennig")
        async with session_scope() as session:
            await ObsidianExporter().export(session, tmp_path, min_links=0)
        # Between runs the user writes a note, and the store gains a subject
        # whose note would take its path.
        ideas = tmp_path / "Ideas.md"
        ideas.write_text("my ideas\n")
        await _seed_subject("Ideas")
        async with session_scope() as session:
            await ObsidianExporter().export(session, tmp_path, min_links=0)
        assert ideas.read_text() == "my ideas\n"
        assert (tmp_path / "Pfennig.md").exists()
        async with session_scope() as session:
            await ObsidianExporter().export(session, tmp_path, min_links=0)
        assert ideas.read_text() == "my ideas\n"

    @pytest.mark.asyncio
    async def test_an_earlier_export_is_adopted_and_pruned(
        self, db_session: object, tmp_path: Path
    ) -> None:
        # A vault exported before the manifest existed: no refusal, and a
        # stale note it wrote is still pruned.
        stale = tmp_path / "Renamed Subject.md"
        stale.write_text("---\ntags:\n  - particles/subject\n---\n# Renamed Subject\n")
        (tmp_path / "_index.md").write_text("---\ntags: [particles/index]\n---\n# index\n")
        async with session_scope() as session:
            await ObsidianExporter().export(session, tmp_path, min_links=0)
        assert not stale.exists()
        assert (tmp_path / EXPORT_MANIFEST_NAME).exists()


class TestLogseqOwnership:
    @pytest.mark.asyncio
    async def test_users_pages_survive_two_exports(
        self, db_session: object, tmp_path: Path
    ) -> None:
        await _seed_subject("Pfennig")
        pages = tmp_path / "pages"
        pages.mkdir()
        own_page = pages / "Reading list.md"
        own_page.write_text("- a book\n")
        journal = tmp_path / "journals" / "2026_10_01.md"
        journal.parent.mkdir()
        journal.write_text("- today\n")

        async with session_scope() as session:
            with pytest.raises(ExportTargetNotOwnedError):
                await LogseqExporter().export(session, tmp_path, min_links=0)

        async with session_scope() as session:
            await LogseqExporter().export(session, tmp_path, min_links=0, force=True)
        assert (pages / "Pfennig.md").exists()
        assert (tmp_path / EXPORT_MANIFEST_NAME).exists()

        async with session_scope() as session:
            await LogseqExporter().export(session, tmp_path, min_links=0, min_particles=99)
        assert not (pages / "Pfennig.md").exists()
        assert own_page.read_text() == "- a book\n"
        assert journal.read_text() == "- today\n"
