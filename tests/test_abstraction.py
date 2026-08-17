"""Tests for the abstraction-promotion pass (particles/operations/abstraction.py).

Covers the pure helpers (components, depth, shared subjects, parsers, the
derived-particle constructor contract), the four new store query shapes, the
§5 revalidation ladder's write paths (retire / structural refresh / entailment
refresh / supersede / budget deferral), and propose-mode candidate events.
LLM judges are mocked at the module binding
(``particles.operations.abstraction._llm_call`` is imported at module top, so
patch the *caller's* binding per tests/AGENTS.md § Mocking strategy).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from particles.core.conflict_resolution import build_inconsistency_particle
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations import abstraction as ab
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import (
    get_active_derived_particles,
    get_particle,
    get_particles_by_ids,
    get_superseding_particle,
    insert_particle,
    update_particle_provenance,
    update_particle_status,
)

OLD = datetime(2026, 1, 1, tzinfo=UTC)


def _specific(content: str, *, value: float = 0.9, subjects: list[str] | None = None) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=value, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
        asserted_by="general-extractor",
        asserted_at=OLD,
        subject_ids=subjects or ["subj-1"],
    )


def _derived(premises: list[Particle], content: str = "General claim.") -> Particle:
    return ab._build_derived_particle(claim=content, premises=premises, subject_ids=["subj-1"])


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestComponents:
    def test_transitive_grouping(self) -> None:
        comps = ab._components([("a", "b"), ("b", "c"), ("x", "y")])
        assert sorted(sorted(c) for c in comps) == [["a", "b", "c"], ["x", "y"]]

    def test_empty(self) -> None:
        assert ab._components([]) == []


class TestDerivedDepth:
    def test_non_derived_is_zero(self) -> None:
        p = _specific("s")
        assert ab._derived_depth(p, {}) == 0

    def test_first_level_is_one(self) -> None:
        premises = [_specific("a"), _specific("b"), _specific("c")]
        d = _derived(premises)
        assert ab._derived_depth(d, {d.id: d}) == 1

    def test_stacked_depth(self) -> None:
        premises = [_specific("a"), _specific("b"), _specific("c")]
        d1 = _derived(premises)
        d2 = _derived([d1, _specific("x"), _specific("y")], content="More general.")
        by_id = {d1.id: d1, d2.id: d2}
        assert ab._derived_depth(d2, by_id) == 2


class TestSharedSubjects:
    def test_intersection(self) -> None:
        a = _specific("a", subjects=["s1", "s2"])
        b = _specific("b", subjects=["s2", "s3"])
        assert ab._shared_subject_ids([a, b], "fallback") == ["s2"]

    def test_fallback_when_disjoint(self) -> None:
        a = _specific("a", subjects=["s1"])
        b = _specific("b", subjects=["s2"])
        assert ab._shared_subject_ids([a, b], "fallback") == ["fallback"]


class TestParseJsonObject:
    def test_plain(self) -> None:
        assert ab._parse_json_object('{"claim": "x"}') == {"claim": "x"}

    def test_fenced_with_prose(self) -> None:
        raw = 'Sure!\n```json\n{"claim": "x", "rationale": "y"}\n```'
        assert ab._parse_json_object(raw) == {"claim": "x", "rationale": "y"}

    def test_garbage_and_none(self) -> None:
        assert ab._parse_json_object("no json here") is None
        assert ab._parse_json_object(None) is None
        assert ab._parse_json_object("[1, 2]") is None


class TestDerivedParticleContract:
    """the lineage + confidence contract of a minted particle."""

    def test_constructor_contract(self) -> None:
        premises = [
            _specific("a", value=0.9),
            _specific("b", value=0.6),
            _specific("c", value=0.8),
        ]
        d = _derived(premises)
        assert d.confidence.calibration_source is CalibrationSource.DERIVED
        assert d.confidence.value == 0.6  # min of premises
        assert d.asserted_by == ab.ABSTRACTION_ACTOR
        assert d.extractor_ref is None
        assert d.status is Status.ACTIVE
        # One PARTICLE ref per premise, particle id in corpus_entry_id.
        assert [r.type for r in d.provenance] == [ProvenanceRefType.PARTICLE] * 3
        assert ab.premise_ids_of(d) == [p.id for p in premises]
        assert d.contributors is not None
        assert d.contributors[0].role == "agent"
        assert ab.is_derived(d) and not ab.is_derived(premises[0])


class TestInconsistencyTriggerRefType:
    """The wrapper's trigger ref stays type-honest for derived candidates."""

    def test_default_source_typed(self) -> None:
        inc = build_inconsistency_particle(
            _specific("a"), _specific("b"), corpus_entry_id="e9", snapshot_id="s9"
        )
        assert inc.provenance[2].type is ProvenanceRefType.SOURCE

    def test_particle_typed_trigger(self) -> None:
        a, b = _specific("a"), _specific("b")
        inc = build_inconsistency_particle(
            a,
            b,
            corpus_entry_id=b.id,
            snapshot_id="",
            trigger_ref_type=ProvenanceRefType.PARTICLE,
        )
        assert inc.provenance[2].type is ProvenanceRefType.PARTICLE
        assert inc.provenance[2].corpus_entry_id == b.id
        assert inc.provenance[2].snapshot_id is None  # "" normalised to None


# ---------------------------------------------------------------------------
# Store helpers (new query shapes)
# ---------------------------------------------------------------------------


class TestStoreHelpers:
    @pytest.mark.asyncio
    async def test_get_active_derived_particles(self, db_session) -> None:
        premises = [_specific("a"), _specific("b"), _specific("c")]
        for p in premises:
            await insert_particle(db_session, p)
        d = _derived(premises)
        await insert_particle(db_session, d)
        derived = await get_active_derived_particles(db_session)
        assert [p.id for p in derived] == [d.id]
        # A retired derived particle drops out.
        await update_particle_status(
            db_session, d.id, Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY
        )
        assert await get_active_derived_particles(db_session) == []

    @pytest.mark.asyncio
    async def test_get_particles_by_ids(self, db_session) -> None:
        a, b = _specific("a"), _specific("b")
        await insert_particle(db_session, a)
        await insert_particle(db_session, b)
        got = await get_particles_by_ids(db_session, [a.id, b.id, "missing", a.id])
        assert set(got) == {a.id, b.id}
        assert await get_particles_by_ids(db_session, []) == {}

    @pytest.mark.asyncio
    async def test_get_superseding_particle(self, db_session) -> None:
        old = _specific("old claim")
        await insert_particle(db_session, old)
        successor = _specific("new claim").model_copy(update={"supersedes": old.id})
        await insert_particle(db_session, successor)
        found = await get_superseding_particle(db_session, old.id)
        assert found is not None and found.id == successor.id
        assert await get_superseding_particle(db_session, "nope") is None

    @pytest.mark.asyncio
    async def test_update_particle_provenance(self, db_session) -> None:
        premises = [_specific("a"), _specific("b"), _specific("c")]
        for p in premises:
            await insert_particle(db_session, p)
        d = _derived(premises)
        await insert_particle(db_session, d)
        new_refs = ab._premise_refs([premises[0].id, premises[1].id])
        await update_particle_provenance(db_session, d.id, new_refs)
        reloaded = await get_particle(db_session, d.id)
        assert reloaded is not None
        assert ab.premise_ids_of(reloaded) == [premises[0].id, premises[1].id]
        with pytest.raises(ValueError, match="not found"):
            await update_particle_provenance(db_session, "missing", new_refs)


# ---------------------------------------------------------------------------
# Revalidation ladder (§5)
# ---------------------------------------------------------------------------


async def _seed_derived(db_session, n_premises: int = 3) -> tuple[Particle, list[Particle]]:
    premises = [_specific(f"specific claim {i}") for i in range(n_premises)]
    for p in premises:
        await insert_particle(db_session, p)
    d = _derived(premises)
    await insert_particle(db_session, d)
    return d, premises


def _enable(**overrides: object) -> None:
    """Turn the pass on in the live config singleton.

    The autouse fixture in conftest calls ``reset_config()`` before every
    test, so these mutations never leak (the test_confidence.py pattern —
    never call ``reset_config()`` mid-test; it disposes the DB engine
    registry and closes the in-memory database).
    """
    from particles.config import get_config

    cfg = get_config().consolidation.abstraction
    cfg.enabled = True
    for key, value in overrides.items():
        setattr(cfg, key, value)


class TestRevalidationLadder:
    @pytest.mark.asyncio
    async def test_untouched_derived_not_checked(self, db_session) -> None:
        _enable()
        await _seed_derived(db_session)
        report = ab.AbstractionReport()
        await ab._revalidate(db_session, report, ab._Budget(5))
        assert report.revalidation.checked == 0

    @pytest.mark.asyncio
    async def test_rung4_retire_below_floor(self, db_session) -> None:
        _enable()
        d, premises = await _seed_derived(db_session)
        # Retract two of three premises → |S′| = 1 < min_cluster_size (3).
        for p in premises[:2]:
            await update_particle_status(db_session, p.id, Status.RETRACTED)
        report = ab.AbstractionReport()
        await ab._revalidate(db_session, report, ab._Budget(5))
        assert report.revalidation.retired == 1
        assert report.llm_calls == 0  # no LLM on rung 4
        reloaded = await get_particle(db_session, d.id)
        assert reloaded is not None and reloaded.status is Status.PROVENANCE_STALE
        assert reloaded.status_reason is StatusReason.RETRACTED_DEPENDENCY

    @pytest.mark.asyncio
    async def test_rung1_structural_refresh(self, db_session) -> None:
        _enable()
        d, premises = await _seed_derived(db_session)
        # Supersede one premise with a content-identical revision.
        old = premises[0]
        successor = _specific(old.content).model_copy(update={"supersedes": old.id})
        await insert_particle(db_session, successor)
        await update_particle_status(
            db_session, old.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
        )
        report = ab.AbstractionReport()
        await ab._revalidate(db_session, report, ab._Budget(5))
        assert report.revalidation.refreshed_structural == 1
        assert report.llm_calls == 0
        reloaded = await get_particle(db_session, d.id)
        assert reloaded is not None and reloaded.status is Status.ACTIVE
        assert successor.id in ab.premise_ids_of(reloaded)
        assert old.id not in ab.premise_ids_of(reloaded)

    @pytest.mark.asyncio
    async def test_rung2_entailment_refresh(self, db_session) -> None:
        _enable()
        d, premises = await _seed_derived(db_session)
        old = premises[0]
        successor = _specific("reworded but compatible claim").model_copy(
            update={"supersedes": old.id}
        )
        await insert_particle(db_session, successor)
        await update_particle_status(
            db_session, old.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
        )
        entailed = json.dumps({"entailed": True, "reason": "still supported"})
        with patch.object(ab, "_llm_call", new=AsyncMock(return_value=entailed)):
            report = ab.AbstractionReport()
            await ab._revalidate(db_session, report, ab._Budget(5))
        assert report.revalidation.refreshed_entailed == 1
        reloaded = await get_particle(db_session, d.id)
        assert reloaded is not None
        assert successor.id in ab.premise_ids_of(reloaded)

    @pytest.mark.asyncio
    async def test_rung3_distinct_supersedes(self, db_session) -> None:
        _enable()
        d, premises = await _seed_derived(db_session)
        old = premises[0]
        successor = _specific("materially different claim").model_copy(
            update={"supersedes": old.id}
        )
        await insert_particle(db_session, successor)
        await update_particle_status(
            db_session, old.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
        )

        responses = iter(
            [
                json.dumps({"entailed": False, "reason": "no longer supported"}),
                json.dumps({"claim": "Updated general claim.", "rationale": "r"}),
                json.dumps({"verdict": "DISTINCT"}),
            ]
        )

        async def fake_llm(*args, **kwargs):
            return next(responses)

        # reconcile_and_insert re-embeds; make it a plain insert by mocking at
        # the ladder's deferred-import site is heavy — instead patch the
        # pipeline seam to a direct insert.
        async def fake_reconcile(session, particle, *args, **kwargs):
            await insert_particle(session, particle)
            return particle

        with (
            patch.object(ab, "_llm_call", new=fake_llm),
            patch("particles.ingest.pipeline.reconcile_and_insert", new=fake_reconcile),
        ):
            report = ab.AbstractionReport()
            await ab._revalidate(db_session, report, ab._Budget(5))

        assert report.revalidation.superseded == 1
        old_d = await get_particle(db_session, d.id)
        assert old_d is not None and old_d.status is Status.SUPERSEDED
        new_d = await get_superseding_particle(db_session, d.id)
        assert new_d is not None
        assert new_d.content == "Updated general claim."
        assert ab.is_derived(new_d)
        assert new_d.supersedes == d.id

    @pytest.mark.asyncio
    async def test_budget_exhausted_defers(self, db_session) -> None:
        _enable()
        d, premises = await _seed_derived(db_session)
        old = premises[0]
        successor = _specific("reworded claim").model_copy(update={"supersedes": old.id})
        await insert_particle(db_session, successor)
        await update_particle_status(
            db_session, old.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
        )
        report = ab.AbstractionReport()
        await ab._revalidate(db_session, report, ab._Budget(0))
        assert report.revalidation.deferred == 1
        # Untouched: still ACTIVE with the old premise refs (discount applies
        # at read time until a later cycle revalidates).
        reloaded = await get_particle(db_session, d.id)
        assert reloaded is not None and reloaded.status is Status.ACTIVE
        assert old.id in ab.premise_ids_of(reloaded)

    @pytest.mark.asyncio
    async def test_llm_unavailable_defers(self, db_session) -> None:
        _enable()
        d, premises = await _seed_derived(db_session)
        old = premises[0]
        successor = _specific("reworded claim").model_copy(update={"supersedes": old.id})
        await insert_particle(db_session, successor)
        await update_particle_status(
            db_session, old.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
        )
        with patch.object(ab, "_llm_call", new=AsyncMock(return_value=None)):
            report = ab.AbstractionReport()
            await ab._revalidate(db_session, report, ab._Budget(5))
        assert report.revalidation.deferred == 1


# ---------------------------------------------------------------------------
# Temporal-eligibility guard (§2 eligibility)
# ---------------------------------------------------------------------------


class TestIsTimeAnchored:
    """The content/valid_until detector behind ``exclude_time_anchored``."""

    def test_valid_until_bearer(self) -> None:
        p = _specific("a durable-looking claim").model_copy(
            update={"valid_until": datetime(2026, 12, 1, tzinfo=UTC)}
        )
        assert ab._is_time_anchored(p)

    @pytest.mark.parametrize(
        "content",
        [
            "The release shipped on 2026-07-18.",
            "The audit ran in July 2026.",
            "The sweep landed in December.",
            "Standup moved to Tuesday.",
            "The build broke at 14:30.",
            "Revenue rose in Q3.",
            "The migration completes tomorrow.",
            "The owner reviewed it yesterday.",
            "Cut the release next week.",
            "The key expired two days ago.",
            "As of the latest run, the gate holds.",
            "The extractor was upgraded on Jul 4.",
            "The freeze starts 12 Nov.",
        ],
    )
    def test_time_anchored_content(self, content: str) -> None:
        assert ab._is_time_anchored(_specific(content))

    @pytest.mark.parametrize(
        "content",
        [
            "Backgrounded commits fail at the GPG signing step.",
            "The operator prefers signed commits.",
            "This may fail when the key is missing.",
            "Version 1.77.4 fixed the interrupt handler.",
            "The pass is capped at 5 promotions per run.",
        ],
    )
    def test_durable_content_not_anchored(self, content: str) -> None:
        """No false positives on the durable claims the pass exists to gather.

        'may' as a modal verb and a bare version identifier are the two traps:
        both would neuter the pass if the detector fired on them.
        """
        assert not ab._is_time_anchored(_specific(content))


class _FakeSubject:
    def __init__(self, subject_id: str) -> None:
        self.id = subject_id


async def _clusters_over(members: list[Particle]) -> list[ab._Cluster]:
    """Run ``_find_clusters`` over a fixed member set with identical embeddings.

    The store reads are patched at the module binding (they are imported at
    the top of ``particles.operations.abstraction``, per tests/AGENTS.md
    § Mocking strategy), so eligibility is the only thing under test — every
    pair is maximally similar, so any 3 eligible members form one cluster.
    """
    import numpy as np

    emb = np.ones(4, dtype=np.float32)
    with (
        patch.object(ab, "get_inconsistency_backrefs", new=AsyncMock(return_value=set())),
        patch.object(ab, "get_active_derived_particles", new=AsyncMock(return_value=[])),
        patch.object(ab, "list_all_subjects", new=AsyncMock(return_value=[_FakeSubject("subj-1")])),
        patch.object(
            ab,
            "get_active_particles_with_embeddings",
            new=AsyncMock(return_value=[(p, emb) for p in members]),
        ),
    ):
        return await ab._find_clusters(None, scope_ids=None)  # type: ignore[arg-type]


class TestTemporalEligibility:
    """time-anchored premises never reach a cluster."""

    @pytest.mark.asyncio
    async def test_durable_members_cluster(self) -> None:
        _enable()
        members = [_specific(f"the durable claim {i}") for i in range(3)]
        clusters = await _clusters_over(members)
        assert len(clusters) == 1
        assert clusters[0].member_ids == frozenset(p.id for p in members)

    @pytest.mark.asyncio
    async def test_valid_until_bearer_excluded(self) -> None:
        _enable()
        members = [_specific(f"the durable claim {i}") for i in range(3)]
        members[0] = members[0].model_copy(
            update={"valid_until": datetime(2026, 12, 1, tzinfo=UTC)}
        )
        # Two survivors is below min_cluster_size (3) → no cluster at all.
        assert await _clusters_over(members) == []

    @pytest.mark.asyncio
    async def test_date_mention_excluded(self) -> None:
        _enable()
        members = [_specific(f"the durable claim {i}") for i in range(3)]
        members[0] = _specific("the deploy ran on 2026-07-18")
        assert await _clusters_over(members) == []

    @pytest.mark.asyncio
    async def test_knob_off_restores_old_behaviour(self) -> None:
        _enable(exclude_time_anchored=False)
        members = [_specific(f"the durable claim {i}") for i in range(3)]
        members[0] = members[0].model_copy(
            update={"valid_until": datetime(2026, 12, 1, tzinfo=UTC)}
        )
        members[1] = _specific("the deploy ran on 2026-07-18")
        clusters = await _clusters_over(members)
        assert len(clusters) == 1
        assert clusters[0].member_ids == frozenset(p.id for p in members)


# ---------------------------------------------------------------------------
# Promotion (§2 + §6 propose mode)
# ---------------------------------------------------------------------------


class TestPromotion:
    @pytest.mark.asyncio
    async def test_propose_mode_records_candidate_event(self, db_session) -> None:
        _enable()
        premises = [
            _specific("specific a", value=0.9),
            _specific("specific b", value=0.7),
            _specific("specific c", value=0.8),
        ]
        for p in premises:
            await insert_particle(db_session, p)
        cluster = ab._Cluster("subj-1", premises)

        responses = iter(
            [
                json.dumps({"claim": "General claim.", "rationale": "because"}),
                json.dumps({"entailed": True, "reason": "ok"}),
            ]
        )

        async def fake_llm(*args, **kwargs):
            return next(responses)

        report = ab.AbstractionReport(mode="propose")
        with (
            patch.object(ab, "_llm_call", new=fake_llm),
            patch.object(ab, "_duplicate_of", new=AsyncMock(return_value=None)),
        ):
            await ab._promote_cluster(db_session, cluster, report)

        assert report.candidates_synthesized == 1
        assert len(report.proposed_event_ids) == 1
        events = await list_events(db_session, event_type=OperatorEventType.ABSTRACTION_CANDIDATE)
        assert len(events) == 1
        payload = events[0].payload
        assert payload is not None
        assert payload["claim"] == "General claim."
        assert payload["premise_ids"] == [p.id for p in premises]
        assert payload["confidence_value"] == 0.7  # min of premises
        # No particle asserted in propose mode.
        assert await get_active_derived_particles(db_session) == []

    @pytest.mark.asyncio
    async def test_entailment_gate_rejects(self, db_session) -> None:
        _enable()
        premises = [_specific("a"), _specific("b"), _specific("c")]
        cluster = ab._Cluster("subj-1", premises)
        responses = iter(
            [
                json.dumps({"claim": "Overreaching claim.", "rationale": "r"}),
                json.dumps({"entailed": False, "reason": "broader than evidence"}),
            ]
        )

        async def fake_llm(*args, **kwargs):
            return next(responses)

        report = ab.AbstractionReport(mode="propose")
        with patch.object(ab, "_llm_call", new=fake_llm):
            await ab._promote_cluster(db_session, cluster, report)
        assert report.rejected_entailment == 1
        assert report.proposed_event_ids == []

    @pytest.mark.asyncio
    async def test_duplicate_gate_unavailable_discards_and_warns(
        self, db_session, no_embedding_model: None
    ) -> None:
        """an unrunnable duplicate check is not a passed duplicate check.

        Without an encoder the gate's pre-filter cannot run and `_duplicate_of`
        returns None — which the caller would read as "not a duplicate" and
        promote a paraphrase of an existing ACTIVE belief, leaving
        `rejected_duplicate` at a healthy-looking 0. Discard instead, matching
        how the entailment judge already handles its own unavailability.
        """
        _enable()
        premises = [_specific("a"), _specific("b"), _specific("c")]
        cluster = ab._Cluster("subj-1", premises)
        responses = iter(
            [
                json.dumps({"claim": "General claim.", "rationale": "r"}),
                json.dumps({"entailed": True, "reason": "ok"}),
            ]
        )

        async def fake_llm(*args, **kwargs):
            return next(responses)

        report = ab.AbstractionReport(mode="propose")
        with patch.object(ab, "_llm_call", new=fake_llm):
            await ab._promote_cluster(db_session, cluster, report)

        assert any("duplicate check unavailable" in w for w in report.warnings), report.warnings
        # Discarded, not promoted — and not miscounted as a duplicate rejection,
        # which would misreport why nothing shipped.
        assert report.proposed_event_ids == []
        assert report.rejected_duplicate == 0
        assert await get_active_derived_particles(db_session) == []

    @pytest.mark.asyncio
    async def test_auto_mode_asserts_via_reconcile(self, db_session) -> None:
        _enable(mode="auto")
        premises = [_specific("a"), _specific("b"), _specific("c")]
        for p in premises:
            await insert_particle(db_session, p)
        cluster = ab._Cluster("subj-1", premises)
        responses = iter(
            [
                json.dumps({"claim": "General claim.", "rationale": "r"}),
                json.dumps({"entailed": True, "reason": "ok"}),
            ]
        )

        async def fake_llm(*args, **kwargs):
            return next(responses)

        async def fake_reconcile(session, particle, *args, **kwargs):
            await insert_particle(session, particle)
            return particle

        report = ab.AbstractionReport(mode="auto")
        with (
            patch.object(ab, "_llm_call", new=fake_llm),
            patch.object(ab, "_duplicate_of", new=AsyncMock(return_value=None)),
            patch("particles.ingest.pipeline.reconcile_and_insert", new=fake_reconcile),
        ):
            await ab._promote_cluster(db_session, cluster, report)

        assert len(report.promoted_particle_ids) == 1
        derived = await get_active_derived_particles(db_session)
        assert len(derived) == 1
        assert derived[0].content == "General claim."
        assert ab.premise_ids_of(derived[0]) == [p.id for p in premises]

    @pytest.mark.asyncio
    async def test_run_pass_disabled_noop(self, db_session) -> None:
        # Default config: enabled=False (the autouse fixture reset config).
        report = await ab.run_abstraction_pass(db_session)
        assert report.enabled is False
        assert report.llm_calls == 0


# ---------------------------------------------------------------------------
# Read-time discount + projection suppression + lint branch (§5 / §10)
# ---------------------------------------------------------------------------


class TestStaleSupportDiscount:
    @pytest.mark.asyncio
    async def test_no_derived_no_queries(self, db_session) -> None:
        assert await ab.stale_support_discounts(db_session, [_specific("a")]) == {}

    @pytest.mark.asyncio
    async def test_discount_applies_when_premise_non_active(self, db_session) -> None:
        d, premises = await _seed_derived(db_session)
        await update_particle_status(db_session, premises[0].id, Status.RETRACTED)
        discounts = await ab.stale_support_discounts(db_session, [d, premises[1]])
        assert discounts == {d.id: 0.5}

    @pytest.mark.asyncio
    async def test_no_discount_when_all_premises_active(self, db_session) -> None:
        d, _premises = await _seed_derived(db_session)
        assert await ab.stale_support_discounts(db_session, [d]) == {}

    @pytest.mark.asyncio
    async def test_score_effective_confidence_applies_discount(self, db_session) -> None:
        from particles.operations.query.effective_confidence import (
            score_effective_confidence,
        )

        d, premises = await _seed_derived(db_session)
        baseline = await score_effective_confidence(db_session, [d])
        await update_particle_status(db_session, premises[0].id, Status.RETRACTED)
        discounted = await score_effective_confidence(db_session, [d])
        assert discounted[d.id] == pytest.approx(baseline[d.id] * 0.5)


class TestProjectionSuppression:
    @pytest.mark.asyncio
    async def test_premises_of_active_derived_suppressed(self, db_session) -> None:
        d, premises = await _seed_derived(db_session)
        suppressed = await ab.projection_suppressed_premise_ids(db_session)
        assert suppressed == frozenset(p.id for p in premises)

    @pytest.mark.asyncio
    async def test_retired_derived_releases_premises(self, db_session) -> None:
        d, _premises = await _seed_derived(db_session)
        await update_particle_status(
            db_session, d.id, Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY
        )
        assert await ab.projection_suppressed_premise_ids(db_session) == frozenset()

    @pytest.mark.asyncio
    async def test_source_demotion_none_disables(self, db_session) -> None:
        from particles.config import get_config

        await _seed_derived(db_session)
        get_config().consolidation.abstraction.source_demotion = "none"
        assert await ab.projection_suppressed_premise_ids(db_session) == frozenset()


class TestLintDerivedBranch:
    @pytest.mark.asyncio
    async def test_derived_gets_revalidation_finding_no_fix(self, db_session) -> None:
        from particles.operations.lint.staleness import _check_retraction_propagation

        d, premises = await _seed_derived(db_session)
        await update_particle_status(db_session, premises[0].id, Status.RETRACTED)
        findings = await _check_retraction_propagation(db_session, fix=True)
        by_type = {f.finding_type for f in findings}
        assert "DERIVED_REVALIDATION" in by_type
        assert "RETRACTION_CASCADE" not in by_type
        # No fix applied: still ACTIVE (keep-ACTIVE-and-discount).
        reloaded = await get_particle(db_session, d.id)
        assert reloaded is not None and reloaded.status is Status.ACTIVE

    @pytest.mark.asyncio
    async def test_non_derived_keeps_cascade_fix(self, db_session) -> None:
        from particles.operations.lint.staleness import _check_retraction_propagation

        target = _specific("victim")
        await insert_particle(db_session, target)
        dependent = Particle(
            content="depends on target",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=ab._premise_refs([target.id]),
            asserted_by="general-extractor",
            asserted_at=OLD,
        )
        await insert_particle(db_session, dependent)
        await update_particle_status(db_session, target.id, Status.RETRACTED)
        findings = await _check_retraction_propagation(db_session, fix=True)
        assert {f.finding_type for f in findings} == {"RETRACTION_CASCADE"}
        reloaded = await get_particle(db_session, dependent.id)
        assert reloaded is not None and reloaded.status is Status.PROVENANCE_STALE


# ---------------------------------------------------------------------------
# Propose-mode resolution: cards, accept / reject gestures (§6)
# ---------------------------------------------------------------------------


async def _seed_candidate_event(db_session, premises: list[Particle]) -> str:
    from particles.core.scoring.confidence import derive_abstraction_confidence
    from particles.store.event_store import EventRefKind, record_event

    event = await record_event(
        db_session,
        actor=ab.ABSTRACTION_ACTOR,
        event_type=OperatorEventType.ABSTRACTION_CANDIDATE,
        reason="because",
        refs=[(EventRefKind.PARTICLE, p.id) for p in premises],
        payload={
            "claim": "General claim.",
            "rationale": "because",
            "premise_ids": [p.id for p in premises],
            "subject_ids": ["subj-1"],
            "confidence_value": derive_abstraction_confidence(
                [p.confidence.value for p in premises]
            ),
        },
    )
    return event.event_id


class TestProposedAbstractionCard:
    def test_key_round_trip(self) -> None:
        from particles.operations.curation.cards import CardKind, CurationCard

        card = CurationCard(
            kind=CardKind.PROPOSED_ABSTRACTION,
            particle_ids=["a", "b", "c"],
            diagnostic="Proposed abstraction",
            suggested_gestures=["accept", "reject", "snooze"],
            candidate_event_id="ev-42",
        )
        assert card.key == "proposed_abstraction:ev-42"
        rebuilt = CurationCard.from_key(card.key)
        assert rebuilt.kind is CardKind.PROPOSED_ABSTRACTION
        assert rebuilt.candidate_event_id == "ev-42"
        assert "accept" in rebuilt.suggested_gestures

    @pytest.mark.asyncio
    async def test_collect_cards_surfaces_pending_candidate(self, db_session) -> None:
        from particles.operations.curation.cards import CardKind
        from particles.operations.curation.collect import collect_cards

        _d, premises = await _seed_derived(db_session)
        # A candidate over three fresh specifics (no derived particle exists
        # for them — use new premises so the card is genuinely pending).
        extra = [_specific(f"other {i}") for i in range(3)]
        for p in extra:
            await insert_particle(db_session, p)
        event_id = await _seed_candidate_event(db_session, extra)
        cards = await collect_cards(db_session, semantic=False)
        proposed = [c for c in cards if c.kind is CardKind.PROPOSED_ABSTRACTION]
        assert len(proposed) == 1
        assert proposed[0].candidate_event_id == event_id
        assert "General claim." in proposed[0].diagnostic
        assert proposed[0].particle_ids == [p.id for p in extra]


class TestAcceptRejectCandidate:
    @pytest.mark.asyncio
    async def test_accept_asserts_with_lineage(self, db_session) -> None:
        premises = [_specific(f"claim {i}", value=0.8 - i * 0.1) for i in range(3)]
        for p in premises:
            await insert_particle(db_session, p)
        event_id = await _seed_candidate_event(db_session, premises)

        async def fake_reconcile(session, particle, *args, **kwargs):
            await insert_particle(session, particle)
            return particle

        with patch("particles.ingest.pipeline.reconcile_and_insert", new=fake_reconcile):
            particle = await ab.accept_candidate(db_session, event_id, actor="curate")

        assert ab.is_derived(particle)
        assert particle.content == "General claim."
        assert ab.premise_ids_of(particle) == [p.id for p in premises]
        assert particle.confidence.value == pytest.approx(0.6)  # min of premises
        resolutions = await list_events(
            db_session, event_type=OperatorEventType.ABSTRACTION_RESOLVED
        )
        assert len(resolutions) == 1
        assert (resolutions[0].payload or {})["resolution"] == "accepted"
        # Resolved candidates stop being pending.
        assert await ab.pending_candidate_events(db_session) == []

    @pytest.mark.asyncio
    async def test_accept_stale_premises_raises(self, db_session) -> None:
        premises = [_specific(f"claim {i}") for i in range(3)]
        for p in premises:
            await insert_particle(db_session, p)
        event_id = await _seed_candidate_event(db_session, premises)
        await update_particle_status(db_session, premises[0].id, Status.RETRACTED)
        with pytest.raises(ab.CandidateStaleError):
            await ab.accept_candidate(db_session, event_id)

    @pytest.mark.asyncio
    async def test_reject_records_and_removes_from_pending(self, db_session) -> None:
        premises = [_specific(f"claim {i}") for i in range(3)]
        for p in premises:
            await insert_particle(db_session, p)
        event_id = await _seed_candidate_event(db_session, premises)
        await ab.reject_candidate(db_session, event_id, reason="too vague")
        resolutions = await list_events(
            db_session, event_type=OperatorEventType.ABSTRACTION_RESOLVED
        )
        assert (resolutions[0].payload or {})["resolution"] == "rejected"
        assert await ab.pending_candidate_events(db_session) == []
        # Double-resolution is refused.
        with pytest.raises(ValueError, match="already"):
            await ab.reject_candidate(db_session, event_id)

    @pytest.mark.asyncio
    async def test_apply_gesture_accept_dispatches(self, db_session) -> None:
        from particles.operations.curation.cards import CurationCard
        from particles.operations.curation.session import apply_gesture

        premises = [_specific(f"claim {i}") for i in range(3)]
        for p in premises:
            await insert_particle(db_session, p)
        event_id = await _seed_candidate_event(db_session, premises)
        card = CurationCard.from_key(f"proposed_abstraction:{event_id}")

        async def fake_reconcile(session, particle, *args, **kwargs):
            await insert_particle(session, particle)
            return particle

        with patch("particles.ingest.pipeline.reconcile_and_insert", new=fake_reconcile):
            result = await apply_gesture(db_session, card, "accept")
        assert result.startswith("Accepted")
        assert len(await get_active_derived_particles(db_session)) >= 1
