# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Rule-file sources stay out of the agent-memory region."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from particles.core.schema import QueryRequest
from particles.operations.projection import project_splice_body
from particles.operations.projection.manifest import DerivedSection, DocManifest
from particles.operations.query import retrieve_ranked
from particles.operations.query.source_tags import (
    excluded_by_source_tags,
    excluded_ids,
    held_only_by,
)
from particles.store.observer_scope_join import Attestation, AttestationGraph, load_attestations
from tests._observer_scope import belief, generation, project_source_tags, source

RULE = ["rule-file", "project:-proj-a"]
MEMORY = project_source_tags("-proj-a")


def attestation(now: dict[str, list[str]], ever: dict[str, list[str]] | None = None) -> Attestation:
    ever = ever if ever is not None else now
    return Attestation(ever=frozenset(ever), now=frozenset(now), tags={**ever, **now})


class TestHeldOnlyBy:
    def test_all_current_entries_tagged_is_held_only(self) -> None:
        assert held_only_by(attestation({"r1": ["rule-file"], "r2": ["rule-file"]}), {"rule-file"})

    def test_one_current_untagged_entry_keeps_it(self) -> None:
        held = attestation({"r": ["rule-file"], "m": ["memory-file"]})
        assert not held_only_by(held, {"rule-file"})

    def test_a_lapsed_belief_is_judged_by_the_entries_that_ever_stated_it(self) -> None:
        lapsed_rule = attestation({}, {"r": ["rule-file"]})
        lapsed_mixed = attestation({}, {"r": ["rule-file"], "m": ["memory-file"]})
        assert held_only_by(lapsed_rule, {"rule-file"})
        assert not held_only_by(lapsed_mixed, {"rule-file"})

    def test_current_entries_decide_over_old_ones(self) -> None:
        moved_to_memory = attestation({"m": ["memory-file"]}, {"r": ["rule-file"], "m": []})
        assert not held_only_by(moved_to_memory, {"rule-file"})

    def test_no_entry_at_all_is_never_held_only(self) -> None:
        assert not held_only_by(attestation({}), {"rule-file"})


class TestExcludedIds:
    def test_a_derived_belief_is_excluded_only_when_every_premise_is(self) -> None:
        graph = AttestationGraph(
            sourced={
                "rule": attestation({"r": ["rule-file"]}),
                "rule2": attestation({"r2": ["rule-file"]}),
                "memory": attestation({"m": ["memory-file"]}),
                "orphan": attestation({}),
            },
            premises={
                "all-rule": ["rule", "rule2"],
                "mixed": ["rule", "memory"],
                "with-orphan": ["rule", "orphan"],
                "second-order": ["all-rule"],
            },
        )
        ids = ["all-rule", "mixed", "with-orphan", "second-order", "rule", "memory"]
        assert excluded_ids(graph, ids, {"rule-file"}) == {"all-rule", "second-order", "rule"}

    def test_a_premise_cycle_is_kept(self) -> None:
        graph = AttestationGraph(sourced={}, premises={"x": ["y"], "y": ["x"]})
        assert excluded_ids(graph, ["x"], {"rule-file"}) == frozenset()

    def test_no_tags_excludes_nothing(self) -> None:
        graph = AttestationGraph(sourced={"rule": attestation({"r": ["rule-file"]})}, premises={})
        assert excluded_ids(graph, ["rule"], ()) == frozenset()


class TestLoadAttestations:
    async def test_ever_now_and_tags_follow_the_latest_generation(self, db_session: Any) -> None:
        r1 = await generation(db_session, "agents", "v1", RULE)
        stated = await belief(db_session, "stated by v1", r1)
        r2 = await generation(db_session, "agents", "v2", RULE)
        current = await belief(db_session, "stated by v2", r2)

        graph = await load_attestations(db_session, [stated, current])

        old = graph.sourced[stated.id]
        assert old.ever == {r1[0]} and old.now == frozenset()
        assert "rule-file" in old.tags[r1[0]]
        assert graph.sourced[current.id].now == {r2[0]}

    async def test_a_derived_belief_resolves_to_its_premises(self, db_session: Any) -> None:
        premise = await belief(db_session, "premise", await source(db_session, "m", MEMORY))
        derived = await belief(db_session, "derived", premises=(premise.id,))

        graph = await load_attestations(db_session, [derived])

        assert graph.premises[derived.id] == [premise.id]
        assert premise.id in graph.sourced


class TestTheRegionFilter:
    async def test_rule_file_only_beliefs_leave_and_memory_beliefs_stay(
        self, db_session: Any
    ) -> None:
        rule = await generation(db_session, "agents", "v1", RULE)
        memory = await generation(db_session, "memory", "v1", MEMORY)
        rule_only = await belief(db_session, "rule only", rule)
        both = await belief(db_session, "rule and memory", rule, memory)
        memory_only = await belief(db_session, "memory only", memory)
        unsourced = await belief(db_session, "asserted with no source")

        dropped = await excluded_by_source_tags(
            db_session, [rule_only, both, memory_only, unsourced], ["rule-file"]
        )

        assert dropped == {rule_only.id}

    async def test_a_memory_file_that_stopped_stating_it_no_longer_counts(
        self, db_session: Any
    ) -> None:
        rule = await generation(db_session, "agents", "v1", RULE)
        m1 = await generation(db_session, "memory", "v1", MEMORY)
        restated_once = await belief(db_session, "restated by memory v1 only", rule, m1)
        await generation(db_session, "memory", "v2", MEMORY)

        dropped = await excluded_by_source_tags(db_session, [restated_once], ["rule-file"])

        assert dropped == {restated_once.id}

    async def test_a_lapsed_rule_file_belief_leaves_on_a_store_never_rescoped(
        self, db_session: Any
    ) -> None:
        r1 = await generation(db_session, "agents", "v1", RULE)
        m1 = await generation(db_session, "memory", "v1", MEMORY)
        lapsed_rule = await belief(db_session, "dropped by the rule file", r1)
        lapsed_mixed = await belief(db_session, "dropped by both", r1, m1)
        await generation(db_session, "agents", "v2", RULE)
        await generation(db_session, "memory", "v2", MEMORY)

        dropped = await excluded_by_source_tags(
            db_session, [lapsed_rule, lapsed_mixed], ["rule-file"]
        )

        assert dropped == {lapsed_rule.id}

    async def test_a_derived_belief_over_an_unsourced_premise_is_kept(
        self, db_session: Any
    ) -> None:
        rule = await generation(db_session, "agents", "v1", RULE)
        from_rule = await belief(db_session, "from the rule file", rule)
        asserted = await belief(db_session, "asserted by an operator")
        over_rules = await belief(db_session, "over rules", premises=(from_rule.id,))
        mixed = await belief(
            db_session, "over a rule and an assertion", premises=(from_rule.id, asserted.id)
        )

        dropped = await excluded_by_source_tags(db_session, [over_rules, mixed], ["rule-file"])

        assert dropped == {over_rules.id}

    async def test_retrieve_ranked_drops_before_the_cut_and_query_is_untouched(
        self, db_session: Any
    ) -> None:
        rule = await generation(db_session, "agents", "v1", RULE)
        memory = await generation(db_session, "memory", "v1", MEMORY)
        rule_only = await belief(db_session, "rule only", rule)
        memory_only = await belief(db_session, "memory only", memory)
        request = QueryRequest(question="Memory index", top_k=1)

        filtered = await retrieve_ranked(
            db_session, request, use_embeddings=False, exclude_source_tags=["rule-file"]
        )
        unfiltered = await retrieve_ranked(
            db_session, request.model_copy(update={"top_k": 5}), use_embeddings=False
        )

        assert [p.id for p, _, _ in filtered] == [memory_only.id]
        assert {p.id for p, _, _ in unfiltered} == {rule_only.id, memory_only.id}


class TestTheAgentMemoryRegion:
    async def test_the_splice_body_filters_only_when_asked(self, db_session: Any) -> None:
        rule = await generation(db_session, "agents", "v1", RULE)
        memory = await generation(db_session, "memory", "v1", MEMORY)
        await belief(db_session, "Rule-file text the harness serves itself.", rule)
        await belief(db_session, "A lesson from a memory file.", memory)
        manifest = DocManifest(
            name="memory-index",
            sections=[DerivedSection(title="Memory index", render="bullets", top_k=60)],
        )

        region = await project_splice_body(
            db_session,
            manifest,
            base_dir=Path("."),
            synthesize=False,
            exclude_source_tags=["rule-file"],
        )
        docs = await project_splice_body(db_session, manifest, base_dir=Path("."), synthesize=False)

        assert "A lesson from a memory file." in region.document
        assert "Rule-file text" not in region.document
        assert "Rule-file text" in docs.document

    @pytest.mark.parametrize(("configured", "expected"), [(None, ["rule-file"]), ([], [])])
    async def test_the_region_renderer_reads_the_config_at_render_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        configured: list[str] | None,
        expected: list[str],
    ) -> None:
        import particles.operations.projection as projection
        from particles.api.cli import _memory_projection
        from particles.config import get_config
        from particles.operations.projection.memory_region import MemoryRegion, body_hash

        if configured is not None:
            get_config().agent_memory.projection.exclude_source_tags = configured
        seen: dict[str, Any] = {}

        class _Backend:
            async def memory_region(self, _store: str, **kwargs: Any) -> MemoryRegion:
                seen.update(kwargs)
                return MemoryRegion(body="", content_hash=body_hash(""))

        monkeypatch.setattr(projection, "load_manifest", lambda _path: None)
        monkeypatch.setattr(
            _memory_projection, "memory_manifest_path", lambda: tmp_path / "memory.yaml"
        )
        monkeypatch.setattr("particles.api.client.get_backend", lambda: _Backend())

        await _memory_projection._render_region_body("default")

        assert seen["exclude_source_tags"] == expected
