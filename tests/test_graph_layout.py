# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure unit tests for the graph view's layout step (D2).

:func:`layout_graph` is the decide half of ``build_graph_data``: it takes the
gathered maps as plain values, so every layout rule is exercised here without
a store. The end-to-end behaviour through a seeded store stays covered by
``test_graph_exporter.py``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.operations.graph_view import GraphLayoutInputs, layout_graph


def _particle(pid: str, *subject_ids: str, supersedes: str | None = None) -> Particle:
    return Particle(
        id=pid,
        content=f"claim {pid}",
        confidence=Confidence(value=0.9),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=f"e-{pid}")],
        asserted_by="test",
        subject_ids=list(subject_ids),
        supersedes=supersedes,
    )


def _inputs(particles: list[Particle], **overrides: Any) -> GraphLayoutInputs:
    by_id = {p.id: p for p in particles}
    subject_ids = sorted({sid for p in particles for sid in p.subject_ids})
    base = GraphLayoutInputs(
        scope_type="query",
        scope_ref="q",
        scope_desc="query retrieval set (top 10)",
        as_of=None,
        history=False,
        particles=by_id,
        ghosts=frozenset(),
        hit_ids=frozenset(),
        hit_rank={},
        hop_by_subject={sid: 0 for sid in subject_ids},
        eff={pid: 0.5 for pid in by_id},
        badges={},
        utility={},
        source_uris={},
        as_of_notes={},
        subjects_by_id={
            sid: Subject(id=sid, canonical_name=sid.upper(), asserted_by="test")
            for sid in subject_ids
        },
        viewer_subject_ids=frozenset(),
        min_particle_confidence=0.0,
        dropped_below_threshold=0,
        excluded_undatable=0,
        missing_disputants=0,
    )
    return replace(base, **overrides)


def _layout(inputs: GraphLayoutInputs, *, max_nodes: int = 50, per_subject: int = 50) -> Any:
    return layout_graph(inputs, max_nodes=max_nodes, max_particles_per_subject=per_subject)


# -- node ranking -------------------------------------------------------------


def test_ranking_hop_then_support() -> None:
    particles = [_particle("p1", "a"), _particle("p2", "b"), _particle("p3", "c")]
    graph = _layout(
        _inputs(
            particles,
            scope_type="subject",
            scope_ref="c",
            hop_by_subject={"c": 0, "a": 1, "b": 1},
            eff={"p1": 0.2, "p2": 0.8, "p3": 0.1},
        )
    )
    # Hop distance first (anchor c), then descending support (b before a).
    assert [n.subject_id for n in graph.nodes] == ["c", "b", "a"]
    assert [n.hop for n in graph.nodes] == [0, 1, 1]
    assert graph.nodes[1].max_effective_confidence == 0.8


def test_ranking_viewer_adjacency_beats_support_within_a_hop() -> None:
    particles = [
        _particle("p1", "a"),
        _particle("p2", "b"),
        _particle("pv", "b", "viewer"),
    ]
    inputs = _inputs(
        particles,
        hop_by_subject={"a": 1, "b": 1, "viewer": 1},
        eff={"p1": 0.9, "p2": 0.1, "pv": 0.1},
    )
    assert [n.subject_id for n in _layout(inputs).nodes] == ["a", "b", "viewer"]
    adjacent = _layout(replace(inputs, viewer_subject_ids=frozenset({"viewer"})))
    # The viewer and the subject sharing an in-scope particle with them rank first.
    assert [n.subject_id for n in adjacent.nodes] == ["b", "viewer", "a"]


def test_ranking_hit_rank_before_support() -> None:
    particles = [_particle("p1", "a"), _particle("p2", "b")]
    graph = _layout(
        _inputs(
            particles,
            hit_ids=frozenset({"p1", "p2"}),
            hit_rank={"p2": 0, "p1": 1},
            eff={"p1": 0.9, "p2": 0.1},
        )
    )
    assert [n.subject_id for n in graph.nodes] == ["b", "a"]
    assert all(info.retrieval_hit for info in graph.particles.values())


def test_subject_anchor_always_renders() -> None:
    graph = _layout(
        _inputs(
            [_particle("p1", "a")],
            scope_type="subject",
            scope_ref="anchor",
            hop_by_subject={"anchor": 0, "a": 1},
        )
    )
    assert [n.subject_id for n in graph.nodes] == ["anchor", "a"]
    assert graph.nodes[0].label == "anchor"  # unknown subject falls back to its id
    assert graph.nodes[0].cargo == []


# -- max_nodes cut --------------------------------------------------------------


def test_max_nodes_cut_and_disclosure() -> None:
    particles = [_particle("p1", "a"), _particle("p2", "b"), _particle("p3", "c")]
    graph = _layout(
        _inputs(particles, eff={"p1": 0.9, "p2": 0.5, "p3": 0.1}),
        max_nodes=2,
    )
    assert [n.subject_id for n in graph.nodes] == ["a", "b"]
    assert "p3" not in graph.particles
    assert graph.census.candidate_subjects == 3
    assert graph.census.rendered_subjects == 2
    assert graph.census.candidate_particles == 3
    assert graph.census.rendered_particles == 2
    assert graph.disclosures == [
        "showing 2 of 3 subjects (graph.max_nodes = 2) — this view is a disclosed "
        "lower bound, not a census"
    ]


def test_no_disclosure_under_the_cap() -> None:
    assert _layout(_inputs([_particle("p1", "a")])).disclosures == []


# -- edge versus cargo ------------------------------------------------------------


def test_edge_versus_cargo_assignment() -> None:
    particles = [
        _particle("solo", "a"),
        _particle("pair", "a", "b"),
        _particle("tri", "a", "b", "c"),
    ]
    graph = _layout(_inputs(particles))
    assert [(e.source, e.target, e.particle_id) for e in graph.edges] == [
        ("a", "b", "pair"),
        ("a", "b", "tri"),
        ("a", "c", "tri"),
        ("b", "c", "tri"),
    ]
    cargo = {n.subject_id: n.cargo for n in graph.nodes}
    assert cargo == {"a": ["solo"], "b": [], "c": []}


def test_edge_degrades_to_cargo_when_one_end_is_cut() -> None:
    particles = [_particle("pair", "a", "b"), _particle("strong", "a")]
    graph = _layout(
        _inputs(particles, hop_by_subject={"a": 0, "b": 1}),
        max_nodes=1,
    )
    assert graph.edges == []
    assert graph.nodes[0].cargo == ["pair", "strong"]


def test_unrendered_particles_drop_unless_foreground() -> None:
    particles = [_particle("p1", "a"), _particle("orphan"), _particle("hit")]
    graph = _layout(_inputs(particles, hit_ids=frozenset({"hit"}), hit_rank={"hit": 0}))
    assert set(graph.particles) == {"p1", "hit"}
    assert graph.disclosures == [
        "1 foreground particle(s) have no linked subject — they appear in the "
        "detail panel, not on the canvas"
    ]


# -- per-subject cargo cap ------------------------------------------------------------


def test_cargo_cap_keeps_foreground_ahead_of_higher_confidence() -> None:
    particles = [
        _particle("weak-hit", "a"),
        _particle("strong", "a"),
        _particle("mid", "a"),
        _particle("pair", "a", "b"),
    ]
    graph = _layout(
        _inputs(
            particles,
            hit_ids=frozenset({"weak-hit"}),
            hit_rank={"weak-hit": 0},
            eff={"weak-hit": 0.1, "strong": 0.9, "mid": 0.5, "pair": 0.5},
        ),
        per_subject=2,
    )
    node_a = next(n for n in graph.nodes if n.subject_id == "a")
    assert node_a.cargo == ["weak-hit", "strong"]
    assert node_a.cargo_truncated == 1
    assert "mid" not in graph.particles
    assert "pair" in graph.particles  # edges are never cargo-capped
    assert graph.disclosures == [
        "1 particle(s) beyond the per-subject panel cap omitted "
        "(graph.max_particles_per_subject = 2)"
    ]


def test_cargo_order_is_confidence_then_id() -> None:
    particles = [_particle("b", "s"), _particle("a", "s"), _particle("c", "s")]
    graph = _layout(_inputs(particles, eff={"a": 0.5, "b": 0.5, "c": 0.9}))
    assert graph.nodes[0].cargo == ["c", "a", "b"]


# -- lineage edges ----------------------------------------------------------------


def test_supersession_edges_only_between_rendered_particles() -> None:
    particles = [
        _particle("old", "a"),
        _particle("new", "a", supersedes="old"),
        _particle("newer", "a", supersedes="gone"),
    ]
    graph = _layout(_inputs(particles, history=True, ghosts=frozenset({"old"})))
    assert [(s.predecessor_id, s.successor_id) for s in graph.supersessions] == [("old", "new")]
    assert graph.particles["old"].ghost
    assert not graph.particles["new"].ghost
    assert graph.history


def test_ghosts_do_not_count_toward_node_support() -> None:
    particles = [_particle("ghost", "a"), _particle("live", "a")]
    graph = _layout(
        _inputs(particles, ghosts=frozenset({"ghost"}), eff={"ghost": 0.9, "live": 0.3})
    )
    assert graph.nodes[0].max_effective_confidence == 0.3


# -- carried annotations and settled disclosures ------------------------------------


def test_gathered_counts_become_disclosures_and_census() -> None:
    graph = _layout(
        _inputs(
            [_particle("p1", "a")],
            scope_type="inconsistency",
            scope_ref="inc",
            min_particle_confidence=0.4,
            dropped_below_threshold=2,
            excluded_undatable=1,
            missing_disputants=1,
        )
    )
    assert graph.census.dropped_below_threshold == 2
    assert graph.census.excluded_undatable == 1
    assert graph.disclosures == [
        "2 particle(s) below min_particle_confidence = 0.4 dropped",
        "1 retired particle(s) excluded fail-closed: retirement instant not reconstructible",
        "1 disputant particle(s) referenced by this INCONSISTENCY no longer exist "
        "in the store — the evidence shown is incomplete",
    ]


def test_node_utility_sums_rendered_particles() -> None:
    particles = [_particle("p1", "a"), _particle("p2", "a")]
    graph = _layout(
        _inputs(particles, utility={"p1": 1.5, "p2": 0.5}, source_uris={"p1": "https://x"})
    )
    assert graph.nodes[0].utility_score == 2.0
    assert graph.particles["p1"].source_uri == "https://x"
    assert graph.particles["p2"].source_uri is None
