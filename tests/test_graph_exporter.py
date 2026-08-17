"""Tests for the scoped epistemic graph view.

Covers the operation (``operations.graph_view.build_graph_data``) and the
``graph`` exporter shell:
  - the anti-hairball invariant: scope is mandatory; the node / cargo caps
    truncate by rank and disclose (discipline; census correctness);
  - the epistemics annotation: effective confidence computed at render time,
    supersession ghosts + chains under ``--history``, the as-of lens
    (visible-with-note, asserted-after-T exclusion, fail-closed undatable
    disclosure), the contested basis and utility
    evidence, all display-only;
  - the self-contained HTML artifact: embedded JSON parses, script-breakout
    content stays escaped, the vendored Cytoscape build inlines cleanly,
    output is deterministic for a fixed render instant.

The ``min_particle_confidence`` contract sweep for this exporter
lives with the other exporters in ``tests/test_exporter_quality_threshold.py``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.graph_view import build_graph_data

EMB = (np.ones(4, dtype=np.float32) / 2.0).tolist()

T1980 = datetime(1980, 1, 1, tzinfo=UTC)
T2000 = datetime(2000, 1, 1, tzinfo=UTC)


def _subject(name: str) -> Subject:
    return Subject(id=str(uuid.uuid4()), canonical_name=name, asserted_by="test")


def _particle(
    content: str,
    *,
    subject_ids: list[str],
    status: Status = Status.ACTIVE,
    status_reason: StatusReason | None = None,
    supersedes: str | None = None,
    confidence: float = 0.9,
    asserted_at: datetime | None = None,
    provenance: list[ProvenanceRef] | None = None,
) -> Particle:
    kwargs: dict[str, Any] = {}
    if asserted_at is not None:
        kwargs["asserted_at"] = asserted_at
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=provenance
        or [
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="entry-1",
                snapshot_id="snap-1",
            )
        ],
        asserted_by="test-agent",
        status=status,
        status_reason=status_reason,
        supersedes=supersedes,
        subject_ids=subject_ids,
        **kwargs,
    )


async def _seed_neighbourhood(session: Any) -> tuple[Subject, Subject, Subject, list[Particle]]:
    """A ── B ── C chain: cargo on A, an A–B edge, a B–C edge."""
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject

    a, b, c = _subject("Alpha"), _subject("Beta"), _subject("Gamma")
    for s in (a, b, c):
        await insert_subject(session, s)
    cargo1 = _particle("Alpha is ancient", subject_ids=[a.id], confidence=0.9)
    cargo2 = _particle("Alpha is large", subject_ids=[a.id], confidence=0.5)
    edge_ab = _particle("Alpha borders Beta", subject_ids=[a.id, b.id])
    edge_bc = _particle("Beta borders Gamma", subject_ids=[b.id, c.id])
    for p in (cargo1, cargo2, edge_ab, edge_bc):
        await insert_particle(session, p, EMB)
    await session.flush()
    return a, b, c, [cargo1, cargo2, edge_ab, edge_bc]


@pytest.mark.asyncio
class TestScopeMandatory:
    async def test_no_scope_raises(self, db_session: Any) -> None:
        with pytest.raises(ValueError, match="scope is mandatory"):
            await build_graph_data(db_session)

    async def test_both_scopes_raise(self, db_session: Any) -> None:
        with pytest.raises(ValueError, match="scope is mandatory"):
            await build_graph_data(db_session, subject_id="s", query="q")

    async def test_unknown_subject_raises(self, db_session: Any) -> None:
        with pytest.raises(ValueError, match="unknown subject id"):
            await build_graph_data(db_session, subject_id="nope")

    async def test_unknown_id_explains_store_scoping(self, db_session: Any) -> None:
        """A missed UUID is usually another store's id — the error says so."""
        with pytest.raises(ValueError, match="store-scoped"):
            await build_graph_data(db_session, subject_id="b5f94c7b-4916-4311-9566-19f704f4a199")

    async def test_exact_name_resolves_to_anchor(self, db_session: Any) -> None:
        """An exact (case-insensitive) canonical name works as the anchor, and
        scope_ref canonicalizes to the resolved subject id."""
        a, _b, _c, _ps = await _seed_neighbourhood(db_session)
        data = await build_graph_data(db_session, subject_id="alpha")
        assert data.scope_ref == a.id
        assert any(n.subject_id == a.id and n.hop == 0 for n in data.nodes)

    async def test_unknown_name_suggests_matches(self, db_session: Any) -> None:
        """A pasted *name* gets did-you-mean suggestions with usable ids."""
        a, _b, _c, _ = await _seed_neighbourhood(db_session)
        with pytest.raises(ValueError, match=rf"Alpha \({a.id}\)"):
            await build_graph_data(db_session, subject_id="Alph")


@pytest.mark.asyncio
class TestSubjectScope:
    async def test_one_hop_neighbourhood(self, db_session: Any) -> None:
        a, b, c, (cargo1, cargo2, edge_ab, edge_bc) = await _seed_neighbourhood(db_session)
        data = await build_graph_data(db_session, subject_id=a.id, hops=1)

        assert data.scope_type == "subject"
        by_id = {n.subject_id: n for n in data.nodes}
        assert set(by_id) == {a.id, b.id}
        assert by_id[a.id].hop == 0 and by_id[b.id].hop == 1
        assert [e.particle_id for e in data.edges] == [edge_ab.id]
        # Cargo sorted by descending effective confidence.
        assert by_id[a.id].cargo == [cargo1.id, cargo2.id]
        # Gamma and its edge are outside the 1-hop scope.
        assert edge_bc.id not in data.particles
        assert data.census.rendered_subjects == 2
        assert data.disclosures == []

    async def test_two_hops_reach_gamma(self, db_session: Any) -> None:
        a, _b, c, (_c1, _c2, _ab, edge_bc) = await _seed_neighbourhood(db_session)
        data = await build_graph_data(db_session, subject_id=a.id, hops=2)
        by_id = {n.subject_id: n for n in data.nodes}
        assert c.id in by_id and by_id[c.id].hop == 2
        assert edge_bc.id in data.particles

    async def test_effective_confidence_is_annotated(self, db_session: Any) -> None:
        a, _b, _c, (cargo1, _c2, _ab, _bc) = await _seed_neighbourhood(db_session)
        data = await build_graph_data(db_session, subject_id=a.id)
        info = data.particles[cargo1.id]
        assert 0.0 < info.effective_confidence <= info.confidence == 0.9
        node = next(n for n in data.nodes if n.subject_id == a.id)
        assert node.max_effective_confidence == pytest.approx(
            max(data.particles[p].effective_confidence for p in node.cargo)
        )

    async def test_inconsistency_never_renders_but_contests(self, db_session: Any) -> None:
        from particles.store.particle_store import insert_particle

        a, _b, _c, (cargo1, _c2, _ab, _bc) = await _seed_neighbourhood(db_session)
        inconsistency = _particle(
            "conflict record",
            subject_ids=[a.id],
            status=Status.INCONSISTENCY,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE,
                    corpus_entry_id=cargo1.id,
                    snapshot_id="-",
                )
            ],
        )
        await insert_particle(db_session, inconsistency, EMB)
        data = await build_graph_data(db_session, subject_id=a.id)

        assert inconsistency.id not in data.particles  # never an ordinary element
        badge = data.particles[cargo1.id].contested
        assert badge is not None and "inconsistency" in badge.bases
        assert badge.inconsistency_id == inconsistency.id
        assert next(n for n in data.nodes if n.subject_id == a.id).contested


async def _seed_conflict(session: Any) -> tuple[Subject, Particle, Particle, Particle]:
    """One §6.6 conflict: ACTIVE winner, PROVENANCE_STALE quarantined loser,
    and the INCONSISTENCY record referencing both as its first two PARTICLE refs."""
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject

    s = _subject("Pluto")
    await insert_subject(session, s)
    winner = _particle("Pluto is a planet", subject_ids=[s.id])
    loser = _particle(
        "Pluto is not a planet",
        subject_ids=[s.id],
        status=Status.PROVENANCE_STALE,
        status_reason=StatusReason.CONFLICT_PENDING,
    )
    inc = _particle(
        "conflict record",
        subject_ids=[s.id],
        status=Status.INCONSISTENCY,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.PARTICLE, corpus_entry_id=winner.id, snapshot_id="-"
            ),
            ProvenanceRef(
                type=ProvenanceRefType.PARTICLE, corpus_entry_id=loser.id, snapshot_id="-"
            ),
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            ),
        ],
    )
    for p in (winner, loser, inc):
        await insert_particle(session, p, EMB)
    await session.flush()
    return s, winner, loser, inc


@pytest.mark.asyncio
class TestInconsistencyScope:
    """scope=inconsistency — a contradiction's evidence picture."""

    async def test_evidence_renders_anchor_and_both_disputants(self, db_session: Any) -> None:
        s, winner, loser, inc = await _seed_conflict(db_session)
        data = await build_graph_data(db_session, inconsistency_id=inc.id)

        assert data.scope_type == "inconsistency"
        assert data.scope_ref == inc.id
        # The anchor renders here BY DESIGN — the one exception to the §5.2
        # never-render rule ("and, in the deferred evidence scope, as the anchor").
        assert inc.id in data.particles
        # Both disputants render with their TRUE statuses — the quarantined
        # loser would never surface on an ordinary render, but it is half
        # the evidence.
        assert data.particles[winner.id].status is Status.ACTIVE
        assert data.particles[loser.id].status is Status.PROVENANCE_STALE
        # Anchor + disputants are the highlighted foreground set.
        for pid in (inc.id, winner.id, loser.id):
            assert data.particles[pid].retrieval_hit
        assert any(n.subject_id == s.id for n in data.nodes)

    async def test_prefix_resolves_to_full_anchor(self, db_session: Any) -> None:
        # The contested badge's prose truncates to 8 chars — that handle works.
        _s, _w, _l, inc = await _seed_conflict(db_session)
        data = await build_graph_data(db_session, inconsistency_id=inc.id[:8])
        assert data.scope_ref == inc.id

    async def test_unknown_id_raises(self, db_session: Any) -> None:
        with pytest.raises(ValueError, match="unknown inconsistency id"):
            await build_graph_data(db_session, inconsistency_id=str(uuid.uuid4()))

    async def test_non_inconsistency_particle_raises(self, db_session: Any) -> None:
        _s, winner, _l, _inc = await _seed_conflict(db_session)
        with pytest.raises(ValueError, match="unknown inconsistency id"):
            await build_graph_data(db_session, inconsistency_id=winner.id)

    async def test_trigger_particle_ref_is_not_a_disputant(self, db_session: Any) -> None:
        # A derived-particle conflict's trigger ref is PARTICLE-typed too; only
        # the FIRST TWO refs are the §6.6 pair (the cascade convention).
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        s = _subject("Derived")
        await insert_subject(db_session, s)
        a = _particle("claim A", subject_ids=[s.id])
        b = _particle(
            "claim B",
            subject_ids=[s.id],
            status=Status.PROVENANCE_STALE,
            status_reason=StatusReason.CONFLICT_PENDING,
        )
        trigger = _particle("premise", subject_ids=[s.id])
        inc = _particle(
            "conflict record",
            subject_ids=[s.id],
            status=Status.INCONSISTENCY,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE, corpus_entry_id=a.id, snapshot_id="-"
                ),
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE, corpus_entry_id=b.id, snapshot_id="-"
                ),
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE, corpus_entry_id=trigger.id, snapshot_id="-"
                ),
            ],
        )
        for p in (a, b, trigger, inc):
            await insert_particle(db_session, p, EMB)
        data = await build_graph_data(db_session, inconsistency_id=inc.id)
        # The trigger renders (incidental cargo on the shared subject) but is
        # not part of the highlighted evidence set.
        assert not data.particles[trigger.id].retrieval_hit
        assert data.particles[a.id].retrieval_hit and data.particles[b.id].retrieval_hit

    async def test_foreground_survives_cargo_cap(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A hub subject's incidental cargo must never truncate the disputants
        # out of the panel — the cap keeps the foreground first.
        from particles.config import get_config
        from particles.store.particle_store import insert_particle

        s, winner, loser, inc = await _seed_conflict(db_session)
        for i in range(4):
            await insert_particle(
                db_session, _particle(f"filler {i}", subject_ids=[s.id], confidence=0.99), EMB
            )
        monkeypatch.setattr(get_config().graph, "max_particles_per_subject", 3)
        data = await build_graph_data(db_session, inconsistency_id=inc.id)
        node = next(n for n in data.nodes if n.subject_id == s.id)
        for pid in (inc.id, winner.id, loser.id):
            assert pid in node.cargo
        assert any("max_particles_per_subject" in line for line in data.disclosures)

    async def test_subjectless_evidence_reaches_the_payload(self, db_session: Any) -> None:
        # Real stores hold pre-subject-binding conflicts: the INCONSISTENCY
        # and both disputants may carry NO subject_ids. The graph model has
        # nothing to hang them on (0 nodes is honest), but the evidence must
        # still reach the particles payload — panel-only, disclosed.
        from particles.store.particle_store import insert_particle

        a = _particle("old claim A", subject_ids=[])
        b = _particle(
            "old claim B",
            subject_ids=[],
            status=Status.PROVENANCE_STALE,
            status_reason=StatusReason.CONFLICT_PENDING,
        )
        inc = _particle(
            "conflict record",
            subject_ids=[],
            status=Status.INCONSISTENCY,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE, corpus_entry_id=a.id, snapshot_id="-"
                ),
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE, corpus_entry_id=b.id, snapshot_id="-"
                ),
            ],
        )
        for p in (a, b, inc):
            await insert_particle(db_session, p, EMB)
        data = await build_graph_data(db_session, inconsistency_id=inc.id)
        assert not data.nodes
        for pid in (inc.id, a.id, b.id):
            assert pid in data.particles
            assert data.particles[pid].retrieval_hit
        assert any("no linked subject" in line for line in data.disclosures)

    async def test_missing_disputant_is_disclosed(self, db_session: Any) -> None:
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        s = _subject("Orphaned")
        await insert_subject(db_session, s)
        a = _particle("surviving claim", subject_ids=[s.id])
        inc = _particle(
            "conflict record",
            subject_ids=[s.id],
            status=Status.INCONSISTENCY,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE, corpus_entry_id=a.id, snapshot_id="-"
                ),
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE,
                    corpus_entry_id=str(uuid.uuid4()),  # hard-deleted disputant
                    snapshot_id="-",
                ),
            ],
        )
        for p in (a, inc):
            await insert_particle(db_session, p, EMB)
        data = await build_graph_data(db_session, inconsistency_id=inc.id)
        assert any("disputant" in line for line in data.disclosures)


@pytest.mark.asyncio
class TestProjectionScope:
    """scope=projection — a manifest section's deterministic selection."""

    async def _seed_tagged(self, session: Any) -> tuple[Subject, Particle, Particle]:
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject
        from particles.store.taxonomy_store import set_particle_tags

        s = _subject("Tagged")
        await insert_subject(session, s)
        hit = _particle("tagged claim", subject_ids=[s.id])
        other = _particle("untagged claim", subject_ids=[s.id])
        for p in (hit, other):
            await insert_particle(session, p, EMB)
        # The tag cage reads the taxonomy edge table, not the tags_json column.
        await set_particle_tags(session, hit.id, ["graph-test"])
        await session.flush()
        return s, hit, other

    def _write_manifest(self, tmp_path: Path) -> Path:
        path = tmp_path / "m.yaml"
        path.write_text(
            "name: graph-test\n"
            "sections:\n"
            "  - title: Tagged claims\n"
            "    region: tagged-claims\n"
            "    tags: [graph-test]\n",
            encoding="utf-8",
        )
        return path

    async def test_section_selection_renders_as_foreground(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        s, hit, _other = await self._seed_tagged(db_session)
        path = self._write_manifest(tmp_path)
        data = await build_graph_data(db_session, manifest=str(path), section="tagged-claims")
        assert data.scope_type == "projection"
        assert data.scope_ref == f"{path}#tagged-claims"
        assert data.particles[hit.id].retrieval_hit
        assert any(n.subject_id == s.id for n in data.nodes)

    async def test_section_addressable_by_title(self, db_session: Any, tmp_path: Path) -> None:
        await self._seed_tagged(db_session)
        path = self._write_manifest(tmp_path)
        data = await build_graph_data(
            db_session,
            manifest=str(path),
            section="tagged claims",  # ci title
        )
        assert data.scope_ref == f"{path}#tagged-claims"  # resolved to the region

    async def test_unknown_manifest_raises(self, db_session: Any, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown manifest"):
            await build_graph_data(db_session, manifest=str(tmp_path / "missing.yaml"), section="x")

    async def test_unknown_section_lists_available(self, db_session: Any, tmp_path: Path) -> None:
        path = self._write_manifest(tmp_path)
        with pytest.raises(ValueError, match="unknown section.*tagged-claims"):
            await build_graph_data(db_session, manifest=str(path), section="nope")

    async def test_manifest_without_section_raises(self, db_session: Any) -> None:
        with pytest.raises(ValueError, match="both selectors"):
            await build_graph_data(db_session, manifest="m.yaml")


@pytest.mark.asyncio
class TestHistoryAndAsOf:
    async def _seed_chain(self, session: Any) -> tuple[Subject, Particle, Particle]:
        """old (asserted 1980, later retired) ⟶ new (ACTIVE successor)."""
        from particles.store.particle_store import insert_particle, update_particle_status
        from particles.store.subject_store import insert_subject

        s = _subject("Pluto")
        await insert_subject(session, s)
        old = _particle("Pluto is a planet", subject_ids=[s.id], asserted_at=T1980)
        await insert_particle(session, old, EMB)
        new = _particle("Pluto is a dwarf planet", subject_ids=[s.id], supersedes=old.id)
        await insert_particle(session, new, EMB)
        await update_particle_status(
            session, old.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
        )
        await session.flush()
        return s, old, new

    async def test_default_render_is_current_belief_only(self, db_session: Any) -> None:
        s, old, new = await self._seed_chain(db_session)
        data = await build_graph_data(db_session, subject_id=s.id)
        assert new.id in data.particles and old.id not in data.particles
        assert data.history is False

    async def test_history_includes_ghost_and_chain(self, db_session: Any) -> None:
        s, old, new = await self._seed_chain(db_session)
        data = await build_graph_data(db_session, subject_id=s.id, history=True)
        assert data.particles[old.id].ghost is True
        assert data.particles[old.id].status == Status.SUPERSEDED
        assert data.particles[new.id].ghost is False
        assert [(x.predecessor_id, x.successor_id) for x in data.supersessions] == [
            (old.id, new.id)
        ]

    async def test_as_of_shows_the_old_belief_with_note(self, db_session: Any) -> None:
        s, old, new = await self._seed_chain(db_session)
        data = await build_graph_data(db_session, subject_id=s.id, as_of=T2000)
        # At T2000 the old belief was in force; the successor did not exist yet.
        assert old.id in data.particles and new.id not in data.particles
        note = data.particles[old.id].as_of_note
        assert note is not None and note.status == Status.SUPERSEDED
        assert data.as_of == T2000

    async def test_undatable_retirement_is_disclosed(self, db_session: Any) -> None:
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        s = _subject("Mystery")
        await insert_subject(db_session, s)
        # Born SUPERSEDED with no retired_at, no successor, no event, no
        # valid_until: the §2b ladder ends at rung 4 → fail-closed + disclosed.
        p = _particle(
            "undatable retired belief",
            subject_ids=[s.id],
            status=Status.SUPERSEDED,
            status_reason=StatusReason.SUPERSEDED_BY_REINDEX,
            asserted_at=T1980,
        )
        await insert_particle(db_session, p, EMB)
        data = await build_graph_data(db_session, subject_id=s.id, as_of=T2000)
        assert p.id not in data.particles
        assert data.census.excluded_undatable == 1
        assert any("fail-closed" in line for line in data.disclosures)


@pytest.mark.asyncio
class TestCaps:
    async def test_max_nodes_truncates_and_discloses(self, db_session: Any) -> None:
        a, b, _c, _ = await _seed_neighbourhood(db_session)
        data = await build_graph_data(db_session, subject_id=a.id, hops=1, max_nodes=1)
        assert [n.subject_id for n in data.nodes] == [a.id]  # anchor outranks hop 1
        assert data.census.candidate_subjects == 2
        assert data.census.rendered_subjects == 1
        assert any("graph.max_nodes" in line for line in data.disclosures)
        # The A–B edge lost an endpoint: its particle becomes Alpha cargo.
        node = data.nodes[0]
        assert len(node.cargo) == 3 and data.edges == []

    async def test_cargo_cap_truncates_and_discloses(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        a, _b, _c, (cargo1, _c2, _ab, _bc) = await _seed_neighbourhood(db_session)
        monkeypatch.setattr(get_config().graph, "max_particles_per_subject", 1)
        data = await build_graph_data(db_session, subject_id=a.id)
        node = next(n for n in data.nodes if n.subject_id == a.id)
        assert node.cargo == [cargo1.id]  # highest effective confidence kept
        assert node.cargo_truncated == 1
        assert any("graph.max_particles_per_subject" in line for line in data.disclosures)

    async def test_hops_clamped_to_config_max(self, db_session: Any) -> None:
        a, _b, c, _ = await _seed_neighbourhood(db_session)
        # graph.max_hops defaults to 2, so hops=10 is clamped — Gamma (hop 2)
        # still renders but the request cannot widen past the config cap.
        data = await build_graph_data(db_session, subject_id=a.id, hops=10)
        assert "2 hops" in data.census.scope
        assert c.id in {n.subject_id for n in data.nodes}


@pytest.mark.asyncio
class TestQueryScope:
    async def test_retrieval_hits_are_flagged(self, db_session: Any) -> None:
        from particles import embeddings as ep

        a, _b, _c, (cargo1, cargo2, edge_ab, _bc) = await _seed_neighbourhood(db_session)
        model = MagicMock()
        model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
        original = ep._embedding_model
        ep.set_embedding_model(model)
        try:
            data = await build_graph_data(db_session, query="what borders alpha?")
        finally:
            ep.set_embedding_model(original)

        assert data.scope_type == "query"
        assert data.census.scope.startswith("query retrieval set")
        # Every hit subject renders at hop 0; hits are flagged.
        assert all(n.hop == 0 for n in data.nodes)
        hits = {pid for pid, info in data.particles.items() if info.retrieval_hit}
        assert cargo1.id in hits and edge_ab.id in hits

    async def test_query_scope_never_reaches_the_llm(self, db_session: Any) -> None:
        """retrieve_ranked generates no NL answer — no API key needed."""
        from particles import embeddings as ep

        await _seed_neighbourhood(db_session)
        model = MagicMock()
        model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
        original = ep._embedding_model
        ep.set_embedding_model(model)
        try:
            data = await build_graph_data(db_session, query="anything")
        finally:
            ep.set_embedding_model(original)
        assert data.census.rendered_particles > 0


@pytest.mark.asyncio
class TestUtilityEvidence:
    async def test_utility_scores_size_nodes_not_opacity(self, db_session: Any) -> None:
        from particles.store.utility_store import record_utility_events

        a, _b, _c, (cargo1, _c2, _ab, _bc) = await _seed_neighbourhood(db_session)
        baseline = await build_graph_data(db_session, subject_id=a.id)
        await record_utility_events(
            db_session, "session-1", {cargo1.id: "literal"}, observed_at=datetime.now(UTC)
        )
        await db_session.flush()
        data = await build_graph_data(db_session, subject_id=a.id)

        assert data.particles[cargo1.id].utility_score > 0.0
        node = next(n for n in data.nodes if n.subject_id == a.id)
        assert node.utility_score > 0.0
        # Display-only: utility never modulates effective confidence.
        assert (
            data.particles[cargo1.id].effective_confidence
            == baseline.particles[cargo1.id].effective_confidence
        )


@pytest.mark.asyncio
class TestHtmlArtifact:
    def _extract_payload(self, html_text: str) -> Any:
        marker = '<script type="application/json" id="graph-data">'
        start = html_text.index(marker) + len(marker)
        end = html_text.index("</script>", start)
        return json.loads(html_text[start:end])

    async def test_export_writes_self_contained_html(self, db_session: Any, tmp_path: Path) -> None:
        from particles.exporters.graph import GraphExporter

        a, _b, _c, particles = await _seed_neighbourhood(db_session)
        out = tmp_path / "graph.html"
        summary = await GraphExporter().export(db_session, out, subject=a.id)

        assert summary.format == "graph"
        assert summary.files_written == 1
        assert summary.subjects == 2 and summary.candidate_subjects == 2
        assert summary.particles_dropped_below_threshold == 0

        html_text = out.read_text(encoding="utf-8")
        assert "Cytoscape Consortium" in html_text  # vendored lib present
        payload = self._extract_payload(html_text)
        assert {n["subject_id"] for n in payload["nodes"]} == {a.id, _b.id}
        assert payload["census"]["rendered_subjects"] == 2

    async def test_script_breakout_content_stays_escaped(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        from particles.exporters.graph import GraphExporter
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        s = _subject("Evil")
        await insert_subject(db_session, s)
        evil = "</script><script>alert(1)</script>"
        await insert_particle(db_session, _particle(evil, subject_ids=[s.id]), EMB)
        out = tmp_path / "evil.html"
        await GraphExporter().export(db_session, out, subject=s.id)

        html_text = out.read_text(encoding="utf-8")
        payload = self._extract_payload(html_text)  # parse fails on a breakout
        [info] = payload["particles"].values()
        assert info["content"] == evil

    async def test_vendored_cytoscape_is_inline_safe(self) -> None:
        from particles.exporters.graph._assets import cytoscape_js, graph_assets_dir

        assert (graph_assets_dir() / "LICENSE.cytoscape.txt").exists()
        assert "</script" not in cytoscape_js().lower()

    async def test_render_is_deterministic(self, db_session: Any) -> None:
        from particles.exporters.graph.render import render_html

        a, _b, _c, _ = await _seed_neighbourhood(db_session)
        data1 = await build_graph_data(db_session, subject_id=a.id)
        data2 = await build_graph_data(db_session, subject_id=a.id)
        assert data1.model_dump() == data2.model_dump()
        stamp = datetime(2026, 7, 23, tzinfo=UTC)
        assert render_html(data1, generated_at=stamp) == render_html(data2, generated_at=stamp)

    async def test_exporter_requires_output_and_scope(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        from particles.exporters.graph import GraphExporter

        with pytest.raises(ValueError, match="output file path"):
            await GraphExporter().export(db_session, None, subject="s")
        with pytest.raises(ValueError, match="scope is mandatory"):
            await GraphExporter().export(db_session, tmp_path / "x.html")


class TestCliUsageErrors:
    """Exporter usage errors reach the operator as one clean stderr line
     with exit code 2 — never a traceback."""

    def test_unknown_subject_is_a_clean_cli_error(self, cli_db: Path, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(
            app,
            [
                "export",
                "graph",
                str(tmp_path / "g.html"),
                "--subject",
                "b5f94c7b-4916-4311-9566-19f704f4a199",
            ],
        )
        assert result.exit_code == 2
        assert "unknown subject id" in result.output
        assert "store-scoped" in result.output
        assert "Traceback" not in result.output

    def test_missing_scope_is_a_clean_cli_error(self, cli_db: Path, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["export", "graph", str(tmp_path / "g.html")])
        assert result.exit_code == 2
        assert "scope is mandatory" in result.output
        assert "Traceback" not in result.output
