"""Cross-exporter `min_particle_confidence` contract.

Every shipped exporter MUST honor `min_particle_confidence` with one
agreed semantic: drop particles whose `effective_confidence` (trust-
weighted) falls below the threshold before any per-
exporter downstream step. This file is the normative behavior check.

The fixture corpus seeds two extractors with different trust weights:

* ``high-trust-extractor`` (trust 1.00) — every particle keeps its raw
  confidence as ``effective_confidence``.
* ``low-trust-extractor`` (trust 0.50) — every particle's
  ``effective_confidence`` is halved (mirrors Reddit weight).

Particles span the threshold from both sides per extractor so a
single ``min_particle_confidence=0.5`` run produces a deterministic
"some dropped, some kept" outcome per exporter.

The Anki-specific deprecation tests live in ``tests/test_anki.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from particles.core.schema import (
    ApplicabilityClause,
    Confidence,
    ExtractorRecord,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.db import session_scope
from particles.exporters.anki import AnkiExporter
from particles.exporters.obsidian import ObsidianExporter
from particles.exporters.wiki import WikiExporter
from particles.store.extractor_store import (
    invalidate_trust_cache,
    upsert_extractor_record,
)
from particles.store.particle_store import insert_particle
from particles.store.subject_store import insert_subject


# Make sure the wiki exporter never reaches the real Anthropic API even
# if ANTHROPIC_API_KEY is set in the shell. The structured-listing
# fallback is deterministic, which is what this contract test exercises.
@pytest.fixture(autouse=True)
def _no_anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    invalidate_trust_cache()


def _make_particle(
    content: str,
    *,
    confidence_value: float,
    extractor_id: str,
    subject_ids: list[str] | None = None,
) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(
            value=confidence_value,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="entry-1",
                snapshot_id="snap-1",
            )
        ],
        asserted_by=extractor_id,
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        extractor_ref={"name": extractor_id, "version": "0.1.0"},
        subject_ids=subject_ids or [],
    )


async def _seed_extractors_and_corpus() -> tuple[Subject, list[Particle], list[Particle]]:
    """Seed two extractors + one subject + a mix of particles.

    Returns ``(subject, kept_at_0_5, dropped_at_0_5)``. With
    threshold=0.5 the kept set has effective_confidence >= 0.5 and the
    dropped set is strictly below.

    Particles (trust_weight × raw = effective):
      kept:    high-trust 0.80 → 0.80
               high-trust 0.60 → 0.60
               low-trust  0.95 → 0.475 ← drops at 0.5 (CRITICAL — the
                                          ADR's semantic shift case)
               low-trust  1.00 → 0.50  ← exactly on the threshold, kept
      dropped: high-trust 0.40 → 0.40
               low-trust  0.30 → 0.15

    The 0.95-raw-low-trust particle is the canonical demonstration of
    "raw → effective" shift: a high *raw* number that the
    operator's trust policy properly discounts.
    """
    subj = Subject(
        id=str(uuid.uuid4()),
        canonical_name="TestSubject",
        asserted_by="test",
    )
    async with session_scope() as session:
        # Register the two extractors so their trust weights are loaded
        # by every exporter's populate_trust_cache call.
        for xid, weight in (("high-trust-extractor", 1.00), ("low-trust-extractor", 0.50)):
            await upsert_extractor_record(
                session,
                ExtractorRecord(
                    extractor_id=xid,
                    name=xid,
                    version="0.1.0",
                    applicability=[
                        ApplicabilityClause(
                            keyword="MAY",
                            domain_uri="http://example.org/test",
                            domain_label="test",
                            source_types=["TEST"],
                        )
                    ],
                    trust_weight=weight,
                ),
            )

        await insert_subject(session, subj)

        # subject_ids on the particle drives Obsidian's per-subject
        # bucketing; the join table is what wiki + anki use. Set both
        # so the test fixture matches a production-extracted particle.
        kept = [
            _make_particle(
                "Kept A — high-trust 0.80",
                confidence_value=0.80,
                extractor_id="high-trust-extractor",
                subject_ids=[subj.id],
            ),
            _make_particle(
                "Kept B — high-trust 0.60",
                confidence_value=0.60,
                extractor_id="high-trust-extractor",
                subject_ids=[subj.id],
            ),
            _make_particle(
                "Kept C — low-trust 1.00→0.50",
                confidence_value=1.00,
                extractor_id="low-trust-extractor",
                subject_ids=[subj.id],
            ),
        ]
        dropped = [
            _make_particle(
                "Dropped X — high-trust 0.40",
                confidence_value=0.40,
                extractor_id="high-trust-extractor",
                subject_ids=[subj.id],
            ),
            _make_particle(
                "Dropped Y — low-trust 0.95→0.475",
                confidence_value=0.95,
                extractor_id="low-trust-extractor",
                subject_ids=[subj.id],
            ),
            _make_particle(
                "Dropped Z — low-trust 0.30→0.15",
                confidence_value=0.30,
                extractor_id="low-trust-extractor",
                subject_ids=[subj.id],
            ),
        ]

        for p in (*kept, *dropped):
            # insert_particle auto-links via the join table when
            # subject_ids is non-empty (see particle_store.py:203).
            await insert_particle(session, p)
        await session.commit()

    return subj, kept, dropped


# ---------------------------------------------------------------------------
# Wiki exporter — contract
# ---------------------------------------------------------------------------


class TestWikiQualityThreshold:
    @pytest.mark.asyncio
    async def test_default_zero_threshold_does_not_filter(
        self, db_session: object, tmp_path: Path
    ) -> None:
        _, kept, dropped = await _seed_extractors_and_corpus()
        # Wiki defaults min_particles to 3 → the subject has 6 particles
        # at threshold 0.0 and renders.
        async with session_scope() as session:
            summary = await WikiExporter().export(session, tmp_path)
        assert summary.min_particle_confidence == 0.0
        assert summary.particles_dropped_below_threshold == 0
        assert summary.qualifying_subjects == 1
        article = (tmp_path / "TestSubject.md").read_text()
        # Every particle's content appears in the structured-listing
        # fallback (we run without an LLM, so the article is structured).
        for p in (*kept, *dropped):
            assert p.content in article

    @pytest.mark.asyncio
    async def test_threshold_drops_below_effective_confidence(
        self, db_session: object, tmp_path: Path
    ) -> None:
        _, kept, dropped = await _seed_extractors_and_corpus()
        async with session_scope() as session:
            summary = await WikiExporter().export(
                session,
                tmp_path,
                min_particle_confidence=0.5,
            )
        assert summary.min_particle_confidence == 0.5
        assert summary.particles_dropped_below_threshold == len(dropped)
        article = (tmp_path / "TestSubject.md").read_text()
        for p in kept:
            assert p.content in article
        for p in dropped:
            assert p.content not in article
        # The frontmatter records the threshold and drop count per ADR
        # 0065 § 5 wiki block.
        from particles.exporters.wiki import _parse_frontmatter

        fm = _parse_frontmatter(article)
        assert fm is not None
        assert fm["min_particle_confidence"] == 0.5
        assert fm["dropped_below_threshold"] == len(dropped)
        # particle_count is the post-filter count.
        assert fm["particle_count"] == len(kept)

    @pytest.mark.asyncio
    async def test_min_particles_count_runs_against_filtered_set(
        self, db_session: object, tmp_path: Path
    ) -> None:
        """A subject with too few high-confidence particles is suppressed."""
        _, _kept, _dropped = await _seed_extractors_and_corpus()
        # Raise the threshold so only 2 particles survive (Kept A 0.80 +
        # Kept B 0.60). With min_particles=3 the subject no longer
        # qualifies even though it had 6 ACTIVE particles before
        # filtering.
        async with session_scope() as session:
            summary = await WikiExporter().export(
                session,
                tmp_path,
                min_particle_confidence=0.55,
                min_particles=3,
            )
        assert summary.qualifying_subjects == 0
        assert not (tmp_path / "TestSubject.md").exists()


# ---------------------------------------------------------------------------
# Obsidian exporter — contract
# ---------------------------------------------------------------------------


class TestObsidianQualityThreshold:
    @pytest.mark.asyncio
    async def test_default_zero_threshold_does_not_filter(
        self, db_session: object, tmp_path: Path
    ) -> None:
        _, kept, dropped = await _seed_extractors_and_corpus()
        async with session_scope() as session:
            summary = await ObsidianExporter().export(session, tmp_path, min_links=0)
        assert summary.particles_dropped_below_threshold == 0
        assert summary.particles == len(kept) + len(dropped)
        note = (tmp_path / "TestSubject.md").read_text()
        for p in (*kept, *dropped):
            assert p.content in note

    @pytest.mark.asyncio
    async def test_threshold_drops_below_effective_confidence(
        self, db_session: object, tmp_path: Path
    ) -> None:
        _, kept, dropped = await _seed_extractors_and_corpus()
        async with session_scope() as session:
            summary = await ObsidianExporter().export(
                session,
                tmp_path,
                min_links=0,
                min_particle_confidence=0.5,
            )
        assert summary.particles_dropped_below_threshold == len(dropped)
        assert summary.particles == len(kept)
        note = (tmp_path / "TestSubject.md").read_text()
        for p in kept:
            assert p.content in note
        for p in dropped:
            assert p.content not in note


# ---------------------------------------------------------------------------
# Anki exporter — contract
# ---------------------------------------------------------------------------


class TestAnkiQualityThreshold:
    @pytest.mark.asyncio
    async def test_default_zero_threshold_does_not_filter(
        self, db_session: object, tmp_path: Path
    ) -> None:
        _, kept, dropped = await _seed_extractors_and_corpus()
        out: Path = tmp_path / "deck.txt"
        async with session_scope() as session:
            summary = await AnkiExporter().export(session, out)
        assert summary.particles_dropped_below_threshold == 0
        assert summary.cards_written == len(kept) + len(dropped)

    @pytest.mark.asyncio
    async def test_threshold_drops_below_effective_confidence(
        self, db_session: object, tmp_path: Path
    ) -> None:
        _, kept, dropped = await _seed_extractors_and_corpus()
        out: Path = tmp_path / "deck.txt"
        async with session_scope() as session:
            summary = await AnkiExporter().export(session, out, min_particle_confidence=0.5)
        assert summary.particles_dropped_below_threshold == len(dropped)
        assert summary.cards_written == len(kept)
        text = out.read_text()
        for p in kept:
            assert p.content in text
        for p in dropped:
            assert p.content not in text
        # Deck-file header records the threshold anki.
        assert "#comment:min_particle_confidence=0.5" in text


# ---------------------------------------------------------------------------
# Notion exporter — contract
# ---------------------------------------------------------------------------


class TestNotionQualityThreshold:
    """The first API-target exporter honours the filter.

    Exercised on the dry-run path: a dry run computes the full plan but makes
    ZERO Notion API writes, so the contract is checkable without
    mocking the HTTP surface. A readable token is still required (Notion has no
    anonymous mode), so the env var is set.
    """

    @pytest.mark.asyncio
    async def test_default_zero_threshold_does_not_filter(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.exporters.notion import NotionExporter

        monkeypatch.setenv("NOTION_API_KEY", "secret-test-token")
        _, kept, dropped = await _seed_extractors_and_corpus()
        async with session_scope() as session:
            summary = await NotionExporter().export(session, None, database_id="db-1", dry_run=True)
        assert summary.particles_dropped_below_threshold == 0
        assert summary.particles_synced == len(kept) + len(dropped)

    @pytest.mark.asyncio
    async def test_threshold_drops_below_effective_confidence(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.exporters.notion import NotionExporter

        monkeypatch.setenv("NOTION_API_KEY", "secret-test-token")
        _, kept, dropped = await _seed_extractors_and_corpus()
        async with session_scope() as session:
            summary = await NotionExporter().export(
                session, None, database_id="db-1", dry_run=True, min_particle_confidence=0.5
            )
        assert summary.particles_dropped_below_threshold == len(dropped)
        assert summary.particles_synced == len(kept)


# ---------------------------------------------------------------------------
# Graph exporter — contract
# ---------------------------------------------------------------------------


class TestGraphQualityThreshold:
    """The scoped graph exporter honours the filter.

    Scope is mandatory for this exporter (anti-hairball rule), so
    the contract is exercised through the subject-neighbourhood scope on the
    shared fixture subject. The filter applies on effective confidence before
    rendering; dropped particles appear in neither the graph payload nor the
    panel cargo, and the census records the drop count.
    """

    @pytest.mark.asyncio
    async def test_default_zero_threshold_does_not_filter(
        self, db_session: object, tmp_path: Path
    ) -> None:
        from particles.exporters.graph import GraphExporter

        subj, kept, dropped = await _seed_extractors_and_corpus()
        out: Path = tmp_path / "graph.html"
        async with session_scope() as session:
            summary = await GraphExporter().export(session, out, subject=subj.id)
        assert summary.particles_dropped_below_threshold == 0
        assert summary.particles == len(kept) + len(dropped)
        text = out.read_text()
        for p in (*kept, *dropped):
            assert p.content in text

    @pytest.mark.asyncio
    async def test_threshold_drops_below_effective_confidence(
        self, db_session: object, tmp_path: Path
    ) -> None:
        from particles.exporters.graph import GraphExporter

        subj, kept, dropped = await _seed_extractors_and_corpus()
        out: Path = tmp_path / "graph.html"
        async with session_scope() as session:
            summary = await GraphExporter().export(
                session, out, subject=subj.id, min_particle_confidence=0.5
            )
        assert summary.particles_dropped_below_threshold == len(dropped)
        assert summary.particles == len(kept)
        assert summary.min_particle_confidence == 0.5
        text = out.read_text()
        for p in kept:
            assert p.content in text
        for p in dropped:
            assert p.content not in text
