"""Tests for particles/store/particle_store.py — row (de)serialization.

Regression coverage for review finding F4.7: ``ParticleRow.from_model`` /
``to_model`` dropped ``ProvenanceRef.chunk_hash`` from ``provenance_json``,
so any consumer of the JSON column (interchange export, reconcile-and-insert)
saw a lossy provenance ref even though the edge table kept the hash.

Also covers the chunk-hash carry-forward extractor match (2026-06-11 review
§4.2): the lookup must exact-match the parsed ``extractor_ref`` name/version,
not substring-match the JSON column.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ParticleType,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.store.particle_store import (
    ParticleRow,
    get_active_particles_for_chunk_hash,
    get_active_particles_with_extractor_id,
    get_active_particles_with_extractor_version,
    get_inconsistency_backrefs,
    insert_particle,
)


def _particle() -> Particle:
    return Particle(
        content="Water is H2O.",
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="e1",
                snapshot_id="s1",
                location="bytes 0-120",
                chunk_hash="a" * 64,
            )
        ],
        asserted_by="general-extractor",
        subject_ids=["sid-water"],
    )


class TestQuarantineBirthSeam:
    """insert_particle enforces the §6.6 quarantine-birth condition.

    The transition table admits ``(new) → PROVENANCE_STALE`` keyed on status
    alone; the *reason* condition — permitted only with
    ``status_reason = CONFLICT_PENDING`` — lives at this persistence seam.
    """

    @pytest.mark.asyncio
    async def test_born_provenance_stale_without_conflict_pending_refused(
        self, db_session: Any
    ) -> None:
        from particles.core.status import Status, StatusReason

        p = _particle().model_copy(
            update={"status": Status.PROVENANCE_STALE, "status_reason": StatusReason.TRUST_DEMOTED}
        )
        with pytest.raises(ValueError, match="CONFLICT_PENDING"):
            await insert_particle(db_session, p)

    @pytest.mark.asyncio
    async def test_born_provenance_stale_without_any_reason_refused(self, db_session: Any) -> None:
        from particles.core.status import Status

        p = _particle().model_copy(update={"status": Status.PROVENANCE_STALE})
        with pytest.raises(ValueError, match="CONFLICT_PENDING"):
            await insert_particle(db_session, p)

    @pytest.mark.asyncio
    async def test_born_provenance_stale_with_conflict_pending_accepted(
        self, db_session: Any
    ) -> None:
        from particles.core.status import Status, StatusReason
        from particles.store.particle_store import get_particle

        p = _particle().model_copy(
            update={
                "status": Status.PROVENANCE_STALE,
                "status_reason": StatusReason.CONFLICT_PENDING,
            }
        )
        await insert_particle(db_session, p)
        stored = await get_particle(db_session, p.id)
        assert stored is not None
        assert stored.status is Status.PROVENANCE_STALE
        assert stored.status_reason is StatusReason.CONFLICT_PENDING

    @pytest.mark.asyncio
    async def test_born_active_and_born_inconsistency_unaffected(self, db_session: Any) -> None:
        """The seam's first validation must not break the existing birth paths."""
        from particles.core.status import Status
        from particles.store.particle_store import get_particle

        active = _particle()
        inc = _particle().model_copy(update={"status": Status.INCONSISTENCY})
        await insert_particle(db_session, active)
        await insert_particle(db_session, inc)
        assert (await get_particle(db_session, active.id)) is not None
        assert (await get_particle(db_session, inc.id)) is not None


def test_provenance_round_trips_chunk_hash_and_location() -> None:
    """model → row → model preserves every ProvenanceRef field (F4.7)."""
    p = _particle()
    rp = ParticleRow.from_model(p).to_model()

    assert len(rp.provenance) == 1
    ref = rp.provenance[0]
    assert ref.type == ProvenanceRefType.SOURCE
    assert ref.corpus_entry_id == "e1"
    assert ref.snapshot_id == "s1"
    assert ref.location == "bytes 0-120"
    assert ref.chunk_hash == "a" * 64


def test_provenance_reads_legacy_rows_without_chunk_hash() -> None:
    """Rows written before the F4.7 fix have no chunk_hash key — read as None."""
    row = ParticleRow.from_model(_particle())
    legacy = json.loads(row.provenance_json)
    for ref in legacy:
        del ref["chunk_hash"]
        del ref["location"]
    row.provenance_json = json.dumps(legacy)

    ref = row.to_model().provenance[0]
    assert ref.chunk_hash is None
    assert ref.location is None
    assert ref.corpus_entry_id == "e1"


async def _seed_with_extractor_ref(
    session: Any,
    *,
    extractor_id: str,
    extractor_version: str,
    chunk_hash: str = "0" * 64,
) -> str:
    """Insert an ACTIVE particle carrying a given ``extractor_ref`` (and chunk hash)."""
    p = Particle(
        content=f"seeded claim from {extractor_id}",
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=extractor_id,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="entry-1",
                snapshot_id="seed-snap",
                chunk_hash=chunk_hash,
            )
        ],
        extractor_ref={"name": extractor_id, "version": extractor_version},
    )
    await insert_particle(session, p, embedding=None)
    await session.commit()
    return p.id


class TestChunkHashExtractorMatch:
    """The carry-forward lookup exact-matches extractor id and version."""

    @pytest.mark.asyncio
    async def test_substring_extractor_id_is_not_a_hit(self, db_session: Any) -> None:
        """An id that is a substring of another extractor's id must miss.

        Regression for the 2026-06-11 review §4.2 nit: the lookup matched
        extractor id/version by JSON-text ``contains``, so a particle written
        by ``github-gist-extractor`` was a carry-forward hit for an extractor
        whose id is ``gist``.
        """
        chunk_hash = "b" * 64
        await _seed_with_extractor_ref(
            db_session,
            chunk_hash=chunk_hash,
            extractor_id="github-gist-extractor",
            extractor_version="1.0.0",
        )

        hits = await get_active_particles_for_chunk_hash(
            db_session, "entry-1", chunk_hash, "gist", "1.0.0"
        )
        assert hits == []

    @pytest.mark.asyncio
    async def test_substring_version_is_not_a_hit(self, db_session: Any) -> None:
        """A version that is a substring of the stored one (1.0 vs 1.0.0) must miss."""
        chunk_hash = "c" * 64
        await _seed_with_extractor_ref(
            db_session,
            chunk_hash=chunk_hash,
            extractor_id="test-extractor",
            extractor_version="1.0.0",
        )

        hits = await get_active_particles_for_chunk_hash(
            db_session, "entry-1", chunk_hash, "test-extractor", "1.0"
        )
        assert hits == []

    @pytest.mark.asyncio
    async def test_exact_match_is_a_hit(self, db_session: Any) -> None:
        """Positive control: the exact (id, version) pair still carries forward."""
        chunk_hash = "d" * 64
        pid = await _seed_with_extractor_ref(
            db_session,
            chunk_hash=chunk_hash,
            extractor_id="github-gist-extractor",
            extractor_version="1.0.0",
        )

        hits = await get_active_particles_for_chunk_hash(
            db_session, "entry-1", chunk_hash, "github-gist-extractor", "1.0.0"
        )
        assert [p.id for p in hits] == [pid]


class TestReindexExtractorSelectors:
    """the two reindex selector helpers exact-match the parsed key.

    ``get_active_particles_with_extractor_version`` /
    ``…_with_extractor_id`` used to select on ``extractor_ref_json.contains``
    alone — a raw substring match over the serialized JSON with no key
    discrimination, so a version false-hit on a *name* (and vice versa) and a
    nested id over-selected. These helpers back ``reindex
    --extractor-version`` / ``--extractor-id``, and reindex supersedes what it
    re-extracts, so over-selection silently retires particles the operator
    never scoped.
    """

    @pytest.mark.asyncio
    async def test_version_does_not_false_hit_on_extractor_name(self, db_session: Any) -> None:
        """A version string that appears inside the extractor *name* must miss."""
        await _seed_with_extractor_ref(
            db_session,
            extractor_id="legacy-2.0.0-extractor",
            extractor_version="1.0.0",
        )

        hits = await get_active_particles_with_extractor_version(db_session, "2.0.0")
        assert hits == []

    @pytest.mark.asyncio
    async def test_version_substring_is_not_a_hit(self, db_session: Any) -> None:
        """``1.0`` must not select a particle stamped ``1.0.0``."""
        await _seed_with_extractor_ref(
            db_session,
            extractor_id="test-extractor",
            extractor_version="1.0.0",
        )

        hits = await get_active_particles_with_extractor_version(db_session, "1.0")
        assert hits == []

    @pytest.mark.asyncio
    async def test_exact_version_is_a_hit(self, db_session: Any) -> None:
        """Positive control: the exact recorded version still selects."""
        pid = await _seed_with_extractor_ref(
            db_session,
            extractor_id="test-extractor",
            extractor_version="1.0.0",
        )

        hits = await get_active_particles_with_extractor_version(db_session, "1.0.0")
        assert [p.id for p in hits] == [pid]

    @pytest.mark.asyncio
    async def test_id_does_not_false_hit_on_version(self, db_session: Any) -> None:
        """An id string that appears inside the recorded *version* must miss."""
        await _seed_with_extractor_ref(
            db_session,
            extractor_id="general-extractor",
            extractor_version="2026.06-nightly",
        )

        hits = await get_active_particles_with_extractor_id(db_session, "nightly")
        assert hits == []

    @pytest.mark.asyncio
    async def test_nested_id_is_not_a_hit(self, db_session: Any) -> None:
        """``gist`` must not select a particle written by ``github-gist-extractor``."""
        await _seed_with_extractor_ref(
            db_session,
            extractor_id="github-gist-extractor",
            extractor_version="1.0.0",
        )

        hits = await get_active_particles_with_extractor_id(db_session, "gist")
        assert hits == []

    @pytest.mark.asyncio
    async def test_exact_id_is_a_hit(self, db_session: Any) -> None:
        """Positive control: the exact extractor id still selects, any version."""
        pid = await _seed_with_extractor_ref(
            db_session,
            extractor_id="github-gist-extractor",
            extractor_version="1.0.0",
        )

        hits = await get_active_particles_with_extractor_id(db_session, "github-gist-extractor")
        assert [p.id for p in hits] == [pid]


class TestInconsistencyBackrefs:
    """get_inconsistency_backrefs maps each particle referenced by
    an open INCONSISTENCY -> that INCONSISTENCY id, so the read tools can mark a
    returned ACTIVE belief as contested."""

    @pytest.mark.asyncio
    async def test_backrefs_map_referenced_particles_to_inconsistency(
        self, db_session: Any
    ) -> None:
        from particles.core.status import Status

        a = _particle().model_copy(update={"content": "Deploy key rotates monthly."})
        b = _particle().model_copy(update={"content": "Deploy key never rotates."})
        await insert_particle(db_session, a)

        inc = Particle(
            content=f"INCONSISTENCY: conflict.\nParticle A: {a.id}\nParticle B: {b.id}",
            confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE, corpus_entry_id=a.id, snapshot_id=a.id
                ),
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE, corpus_entry_id=b.id, snapshot_id=b.id
                ),
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1"
                ),
            ],
            asserted_by="extract-pipeline",
            status=Status.INCONSISTENCY,
        )
        await insert_particle(db_session, inc)

        backrefs = await get_inconsistency_backrefs(db_session)
        # Both conflicting sides resolve to the INCONSISTENCY; the SOURCE ref is ignored.
        assert backrefs.get(a.id) == inc.id
        assert backrefs.get(b.id) == inc.id
        assert "e1" not in backrefs

    @pytest.mark.asyncio
    async def test_uncontested_particle_absent(self, db_session: Any) -> None:
        a = _particle()
        await insert_particle(db_session, a)
        backrefs = await get_inconsistency_backrefs(db_session)
        assert backrefs.get(a.id) is None


class TestParticleIdsForEntries:
    """``get_particle_ids_for_entries`` — the harvested-scope lookup."""

    @pytest.mark.asyncio
    async def test_maps_entries_to_their_particles(self, db_session: Any) -> None:
        from particles.store.particle_store import get_particle_ids_for_entries

        a = _particle()  # provenance → e1
        b = _particle().model_copy(
            update={
                "provenance": [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e2")]
            }
        )
        c = _particle().model_copy(
            update={
                "provenance": [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e3")]
            }
        )
        for p in (a, b, c):
            await insert_particle(db_session, p)

        ids = await get_particle_ids_for_entries(db_session, ["e1", "e2"])
        assert ids == {a.id, b.id}
        # Duplicate entry ids collapse; unknown entries contribute nothing.
        assert await get_particle_ids_for_entries(db_session, ["e3", "e3", "nope"]) == {c.id}
        assert await get_particle_ids_for_entries(db_session, []) == set()


class TestGetActiveNarratives:
    """the NARRATIVE-only read the wiki exporter uses
    instead of paying a second full-store ACTIVE load."""

    @staticmethod
    def _p(content: str, *, particle_type: ParticleType, status: Status) -> Particle:
        return Particle(
            content=content,
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            particle_type=particle_type,
            status=status,
        )

    @pytest.mark.asyncio
    async def test_returns_only_active_narratives(self, db_session: Any) -> None:
        from particles.store.particle_store import get_active_narratives

        wanted = self._p("A hard day.", particle_type=ParticleType.NARRATIVE, status=Status.ACTIVE)
        retracted = self._p(
            "A retracted arc.", particle_type=ParticleType.NARRATIVE, status=Status.RETRACTED
        )
        claim = self._p("A plain claim.", particle_type=ParticleType.CLAIM, status=Status.ACTIVE)
        for p in (wanted, retracted, claim):
            await insert_particle(db_session, p)
        await db_session.commit()

        found = await get_active_narratives(db_session)
        assert [p.id for p in found] == [wanted.id]
