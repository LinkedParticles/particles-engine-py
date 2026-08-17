"""Tests for ``particles corpus delete`` / ``corpus prune-orphans`` orphan cleanup.

The index tables (``particle_subjects``, ``particle_tag_edges``,
``particle_relations``, ``synthesis_cache``) carry no foreign key and SQLite
FK enforcement is off, so deleting particles / subjects elsewhere leaves
dangling rows unless they are swept explicitly. These tests pin that sweep.

Same harness as ``tests/test_cli.py``: a file-based SQLite DB (so state
survives across the fresh ``asyncio.run`` each CLI invocation spins up) and
``typer.testing.CliRunner``. State is seeded and re-inspected through
``session_scope()`` in async helpers.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    Snapshot,
    Subject,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, args: list[str]) -> Any:
    return runner.invoke(app, args, catch_exceptions=False)


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


# --- seeding helpers --------------------------------------------------------


async def _add_entry(uri: str) -> tuple[str, str]:
    """Insert a corpus entry + one COMPLETE snapshot. Returns (entry_id, snapshot_id)."""
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.db import session_scope

    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()), source_type="WEB_PAGE", uri_r=uri, deposited_by="test"
    )
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash="a" * 64,
        extraction_status=ExtractionStatus.COMPLETE,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    async with session_scope() as session:
        session.add(CorpusEntryRow.from_model(entry))
        session.add(SnapshotRow.from_model(snap, entry.entry_id))
        await session.commit()
    return entry.entry_id, snap.snapshot_id


async def _add_subject(name: str) -> str:
    from particles.db import session_scope
    from particles.store.subject_store import insert_subject

    subj = Subject(id=str(uuid.uuid4()), canonical_name=name, asserted_by="test")
    async with session_scope() as session:
        await insert_subject(session, subj)
        await session.commit()
    return subj.id


async def _add_particle(*, entry_id: str, snapshot_id: str, subject_ids: list[str]) -> str:
    """Insert an ACTIVE CLAIM particle. ``insert_particle`` also writes the
    provenance edge and the ``particle_subjects`` join rows."""
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=str(uuid.uuid4()),
        content="A test claim.",
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_id,
                snapshot_id=snapshot_id,
            )
        ],
        subject_ids=subject_ids,
    )
    async with session_scope() as session:
        await insert_particle(session, p)
        await session.commit()
    return p.id


async def _add_tag_edge(particle_id: str, tag: str) -> None:
    from particles.db import session_scope
    from particles.store.taxonomy_store import ParticleTagEdgeRow

    async with session_scope() as session:
        session.add(ParticleTagEdgeRow(particle_id=particle_id, tag=tag))
        await session.commit()


async def _add_relation(a: str, b: str) -> None:
    from particles.db import session_scope
    from particles.store.relation_store import create_relation

    async with session_scope() as session:
        await create_relation(
            session, a, b, RelationType.CO_EVIDENTIAL, RelationCreatedBy.EXTRACTOR_DIRECT
        )
        await session.commit()


async def _add_synth(subject_id: str) -> None:
    from particles.db import session_scope
    from particles.store.synthesis_cache_store import store_cached_article

    async with session_scope() as session:
        await store_cached_article(session, subject_id, "hash-" + subject_id[:6], "v1", "body")
        await session.commit()


# --- state inspection -------------------------------------------------------


async def _counts() -> dict[str, set[str]]:
    """Snapshot the ids present in each table we care about."""
    from sqlalchemy import select

    from particles.db import session_scope
    from particles.store.particle_store import ParticleRow
    from particles.store.relation_store import ParticleRelationRow
    from particles.store.subject_store import ParticleSubjectRow, SubjectRow
    from particles.store.synthesis_cache_store import SynthesisCacheRow
    from particles.store.taxonomy_store import ParticleTagEdgeRow

    async with session_scope() as session:
        particles = set((await session.execute(select(ParticleRow.id))).scalars())
        subjects = set((await session.execute(select(SubjectRow.id))).scalars())
        joins = {
            f"{pid}:{sid}"
            for pid, sid in (
                await session.execute(
                    select(ParticleSubjectRow.particle_id, ParticleSubjectRow.subject_id)
                )
            ).all()
        }
        tags = {
            f"{pid}:{tag}"
            for pid, tag in (
                await session.execute(
                    select(ParticleTagEdgeRow.particle_id, ParticleTagEdgeRow.tag)
                )
            ).all()
        }
        rels = {
            f"{a}:{b}"
            for a, b in (
                await session.execute(
                    select(ParticleRelationRow.particle_a, ParticleRelationRow.particle_b)
                )
            ).all()
        }
        synth = set((await session.execute(select(SynthesisCacheRow.subject_id))).scalars())
    return {
        "particles": particles,
        "subjects": subjects,
        "joins": joins,
        "tags": tags,
        "rels": rels,
        "synth": synth,
    }


# --- tests ------------------------------------------------------------------


class TestCorpusDeleteOrphanCleanup:
    def test_delete_sweeps_index_rows_and_orphan_subjects(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        entry_a, snap_a = _run_async(_add_entry("https://example.com/a"))
        entry_b, snap_b = _run_async(_add_entry("https://example.com/b"))

        # s_only is linked only to entry A's particle -> becomes an orphan.
        # s_shared is linked to particles in both entries -> must survive.
        s_only = _run_async(_add_subject("Only A"))
        s_shared = _run_async(_add_subject("Shared"))

        p1 = _run_async(_add_particle(entry_id=entry_a, snapshot_id=snap_a, subject_ids=[s_only]))
        p2 = _run_async(_add_particle(entry_id=entry_a, snapshot_id=snap_a, subject_ids=[s_shared]))
        p3 = _run_async(_add_particle(entry_id=entry_b, snapshot_id=snap_b, subject_ids=[s_shared]))

        _run_async(_add_tag_edge(p1, "topic/x"))
        _run_async(_add_tag_edge(p3, "topic/y"))
        # Relation spans a deleted (p2) and a surviving (p3) particle: still
        # removed because one endpoint vanishes.
        _run_async(_add_relation(p1, p2))
        _run_async(_add_relation(p2, p3))
        _run_async(_add_synth(s_only))
        _run_async(_add_synth(s_shared))

        result = _invoke(runner, ["corpus", "delete", entry_a[:8], "--yes"])
        assert result.exit_code == 0, result.output

        after = _run_async(_counts())

        # Deleted entry's particles gone; surviving entry's particle stays.
        assert after["particles"] == {p3}
        # s_only orphaned -> deleted; s_shared still linked to p3 -> survives.
        assert after["subjects"] == {s_shared}
        # Only the surviving particle's join row remains.
        assert after["joins"] == {f"{p3}:{s_shared}"}
        # Only the surviving particle's tag edge remains.
        assert after["tags"] == {f"{p3}:topic/y"}
        # Both relations touched a deleted particle -> all gone.
        assert after["rels"] == set()
        # Orphaned subject's synth row gone; surviving subject's stays.
        assert after["synth"] == {s_shared}

    def test_prune_orphans_sweeps_preexisting_cruft(self, runner: CliRunner, cli_db: Path) -> None:
        # Simulate a DB that accumulated dangling rows from older deletes:
        # index rows pointing at particle / subject ids that never existed.
        entry, snap = _run_async(_add_entry("https://example.com/live"))
        s_live = _run_async(_add_subject("Live"))
        p_live = _run_async(_add_particle(entry_id=entry, snapshot_id=snap, subject_ids=[s_live]))
        _run_async(_add_tag_edge(p_live, "topic/live"))
        _run_async(_add_synth(s_live))

        ghost_p = "ghost-particle"
        ghost_s = "ghost-subject"

        async def _seed_cruft() -> None:
            from particles.db import session_scope
            from particles.store.relation_store import ParticleRelationRow
            from particles.store.subject_store import ParticleSubjectRow, SubjectRow
            from particles.store.synthesis_cache_store import SynthesisCacheRow
            from particles.store.taxonomy_store import ParticleTagEdgeRow

            async with session_scope() as session:
                # Dangling join row (particle gone) -> also makes ghost_s an orphan.
                session.add(ParticleSubjectRow(particle_id=ghost_p, subject_id=ghost_s))
                # An orphan subject with no link at all.
                session.add(
                    SubjectRow(
                        id="lonely-subject",
                        canonical_name="Lonely",
                        aliases_json="[]",
                        external_ids_json="[]",
                        created_at=datetime.now(UTC),
                        asserted_by="test",
                    )
                )
                session.add(ParticleTagEdgeRow(particle_id=ghost_p, tag="topic/ghost"))
                session.add(
                    ParticleRelationRow(
                        particle_a=ghost_p,
                        particle_b=p_live,
                        relation_type=RelationType.CO_EVIDENTIAL.value,
                        created_by=RelationCreatedBy.EXTRACTOR_DIRECT.value,
                        created_at=datetime.now(UTC),
                        confidence=1.0,
                    )
                )
                session.add(
                    SynthesisCacheRow(
                        subject_id="vanished-subject",
                        input_hash="h",
                        prompt_version="v1",
                        article_body="x",
                        generated_at=datetime.now(UTC),
                    )
                )
                await session.commit()

        _run_async(_seed_cruft())

        result = _invoke(runner, ["corpus", "prune-orphans", "--yes"])
        assert result.exit_code == 0, result.output

        after = _run_async(_counts())

        # Live state untouched.
        assert after["particles"] == {p_live}
        assert s_live in after["subjects"]
        assert f"{p_live}:{s_live}" in after["joins"]
        assert f"{p_live}:topic/live" in after["tags"]
        assert s_live in after["synth"]

        # All cruft swept.
        assert "lonely-subject" not in after["subjects"]
        assert not any(j.startswith(f"{ghost_p}:") for j in after["joins"])
        assert not any(t.startswith(f"{ghost_p}:") for t in after["tags"])
        assert after["rels"] == set()
        assert "vanished-subject" not in after["synth"]

    def test_prune_orphans_clean_db_is_noop(self, runner: CliRunner, cli_db: Path) -> None:
        entry, snap = _run_async(_add_entry("https://example.com/clean"))
        s = _run_async(_add_subject("Clean"))
        _run_async(_add_particle(entry_id=entry, snapshot_id=snap, subject_ids=[s]))

        result = _invoke(runner, ["corpus", "prune-orphans", "--yes"])
        assert result.exit_code == 0, result.output
        assert "Nothing to prune." in result.output
