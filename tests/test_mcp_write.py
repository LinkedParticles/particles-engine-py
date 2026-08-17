"""write-surface enforcement tests for particles/mcp/tools/write.py.

These gate activation: they pin the security boundary — default-deny, server-side
field construction (§4a), the trust seed (§6a), the §6 mutation guards, and the
EXPLICIT_SUPERSESSION transition. Subject resolution is stubbed so the tests stay
offline (the live Wikidata authority would otherwise fire).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.db import DEFAULT_STORE, session_scope
from particles.store.particle_store import get_particle, insert_particle


def _enable_writes(**overrides: Any) -> None:
    """Enable MCP writes on the default (test-shared) store, in-place on the singleton."""
    w = get_config().mcp.write
    w.enabled_stores = [DEFAULT_STORE]
    for key, value in overrides.items():
        setattr(w, key, value)


@pytest.fixture
def stub_subjects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub subject resolution (write.py defers the import, so this patch reaches it)."""
    import particles.ingest.subject_resolver as sr

    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))


@pytest.fixture
def similar_embeddings() -> Any:
    """Every particle embeds to the same vector → cosine 1.0 → §6.6 conflict fires."""
    from unittest.mock import MagicMock

    import numpy as np

    from particles import embeddings as ep

    model = MagicMock()
    # Real models return numpy (convert_to_numpy=True); reconcile_and_insert calls
    # .tolist() on the result, so the mock must too.
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


async def _insert_foreign(db_session: Any, *, asserted_by: str, calib: CalibrationSource) -> str:
    p = Particle(
        content=f"belief by {asserted_by}",
        confidence=Confidence(value=0.7, calibration_source=calib),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=asserted_by,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
    )
    async with session_scope(DEFAULT_STORE) as s:
        await insert_particle(s, p)
        await s.commit()
    return p.id


class TestDefaultDeny:
    @pytest.mark.asyncio
    async def test_assert_rejected_when_no_store_enabled(self, db_session: Any) -> None:
        from particles.mcp.tools.write import particle_assert

        with pytest.raises(ValueError, match="MCP writes are disabled"):
            await particle_assert("X is true.", ["X"], 0.8, source_excerpt="we said X")

    @pytest.mark.asyncio
    async def test_non_allowlisted_store_rejected(self, db_session: Any) -> None:
        from particles.mcp.tools.write import particle_assert

        _enable_writes()
        with pytest.raises(ValueError, match="not write-enabled"):
            await particle_assert("X.", ["X"], 0.8, source_excerpt="x", store="other")


class TestServerSideConstruction:
    @pytest.mark.asyncio
    async def test_asserted_fields_are_server_owned_and_confidence_clamped(
        self, db_session: Any, stub_subjects: None
    ) -> None:
        from particles.mcp.tools.write import particle_assert

        _enable_writes(max_asserted_confidence=0.90, asserter_identity="mcp:test-agent")
        res = await particle_assert(
            "The deploy key rotates monthly.",
            ["deploy key"],
            0.99,  # above the 0.90 ceiling
            source_excerpt="we agreed the deploy key rotates monthly",
        )
        assert res["verdict"] == "ASSERTED"
        async with session_scope(DEFAULT_STORE) as s:
            p = await get_particle(s, res["asserted_particle_id"])
        assert p is not None
        assert p.confidence.calibration_source == CalibrationSource.AGENT_ASSERTED
        assert p.extractor_ref is None
        assert p.asserted_by == "mcp:test-agent"
        assert p.status == Status.ACTIVE
        assert p.confidence.value == pytest.approx(0.90)  # clamped
        assert p.provenance and p.provenance[0].type == ProvenanceRefType.SOURCE

    @pytest.mark.asyncio
    async def test_unprovenanced_assertion_rejected(
        self, db_session: Any, stub_subjects: None
    ) -> None:
        from particles.mcp.tools.write import particle_assert

        _enable_writes()
        with pytest.raises(ValueError, match="requires provenance"):
            await particle_assert("X.", ["X"], 0.8)


class TestGranularityGate:
    """The §3.3 claim-granularity soft-gate over assert + supersede."""

    @pytest.mark.asyncio
    async def test_oversized_content_rejected(self, db_session: Any, stub_subjects: None) -> None:
        from particles.mcp.tools.write import particle_assert

        _enable_writes(max_assertion_chars=40)
        with pytest.raises(ValueError, match="claim-granularity"):
            await particle_assert(
                "This single assertion is deliberately far too long to be one atomic claim.",
                ["X"],
                0.8,
                source_excerpt="src",
            )

    @pytest.mark.asyncio
    async def test_too_many_sentences_rejected(self, db_session: Any, stub_subjects: None) -> None:
        from particles.mcp.tools.write import particle_assert

        _enable_writes(max_assertion_chars=0, max_assertion_sentences=2)
        with pytest.raises(ValueError, match="claim-granularity"):
            await particle_assert(
                "First claim. Second claim. Third claim.",
                ["X"],
                0.8,
                source_excerpt="src",
            )

    @pytest.mark.asyncio
    async def test_supersede_uses_same_gate(self, db_session: Any, stub_subjects: None) -> None:
        from particles.mcp.tools.write import particle_assert, particle_supersede

        _enable_writes(max_assertion_chars=40)
        res = await particle_assert("Keys rotate monthly.", ["X"], 0.8, source_excerpt="src")
        with pytest.raises(ValueError, match="claim-granularity"):
            await particle_supersede(
                res["asserted_particle_id"],
                "This successor is deliberately too long to pass the claim-granularity gate.",
                ["X"],
                0.8,
                source_excerpt="src",
            )
        # The gate raises mid-supersede (after the prior is flushed to SUPERSEDED,
        # before commit); session_scope rolls back, so the prior stays ACTIVE.
        async with session_scope(DEFAULT_STORE) as s:
            prior = await get_particle(s, res["asserted_particle_id"])
        assert prior is not None
        assert prior.status == Status.ACTIVE

    @pytest.mark.asyncio
    async def test_atomic_content_passes(self, db_session: Any, stub_subjects: None) -> None:
        from particles.mcp.tools.write import particle_assert

        _enable_writes()  # defaults: 320 chars / 3 sentences
        res = await particle_assert(
            "Mercury is the closest planet to the Sun.",
            ["Mercury"],
            0.8,
            source_excerpt="src",
        )
        assert res["verdict"] == "ASSERTED"

    @pytest.mark.asyncio
    async def test_zero_threshold_disables_gate(self, db_session: Any, stub_subjects: None) -> None:
        from particles.mcp.tools.write import particle_assert

        _enable_writes(max_assertion_chars=0, max_assertion_sentences=0)
        res = await particle_assert(
            ("word " * 100).strip(),  # ~500 chars; would breach the default gate
            ["X"],
            0.8,
            source_excerpt="src",
        )
        assert res["verdict"] == "ASSERTED"


class TestTrustSeed:
    @pytest.mark.asyncio
    async def test_first_write_seeds_author_trust(
        self, db_session: Any, stub_subjects: None
    ) -> None:
        from particles.mcp.tools.write import particle_assert
        from particles.store.trust_store import get_trust_rank

        _enable_writes(agent_trust_rank=0.8, asserter_identity="mcp:test-agent")
        await particle_assert("X.", ["X"], 0.8, source_excerpt="x")
        async with session_scope(DEFAULT_STORE) as s:
            rank = await get_trust_rank(s, "agent-memory", "AUTHOR", "mcp:test-agent")
        assert rank == pytest.approx(0.8)


class TestMutationGuards:
    @pytest.mark.asyncio
    async def test_retract_own_belief(self, db_session: Any, stub_subjects: None) -> None:
        from particles.mcp.tools.write import particle_assert, particle_retract

        _enable_writes(asserter_identity="mcp:test-agent")
        res = await particle_assert("X.", ["X"], 0.8, source_excerpt="x")
        pid = res["asserted_particle_id"]
        out = await particle_retract(pid, reason="superseded by newer info")
        assert out["verdict"] == "RETRACTED"
        async with session_scope(DEFAULT_STORE) as s:
            p = await get_particle(s, pid)
        assert p is not None
        assert p.status == Status.RETRACTED
        assert p.status_reason == StatusReason.EXPLICIT_RETRACTION

    @pytest.mark.asyncio
    async def test_cannot_mutate_operator_particle(self, db_session: Any) -> None:
        from particles.mcp.tools.write import particle_retract

        _enable_writes(asserter_identity="mcp:test-agent")
        op_id = await _insert_foreign(
            db_session, asserted_by="operator", calib=CalibrationSource.HUMAN_REVIEW
        )
        with pytest.raises(ValueError, match="HUMAN_REVIEW"):
            await particle_retract(op_id, reason="nope")

    @pytest.mark.asyncio
    async def test_cannot_mutate_other_identity_without_flag(self, db_session: Any) -> None:
        from particles.mcp.tools.write import particle_retract

        _enable_writes(asserter_identity="mcp:test-agent", allow_cross_asserter=False)
        other_id = await _insert_foreign(
            db_session, asserted_by="mcp:other-agent", calib=CalibrationSource.EXTRACTOR_DIRECT
        )
        with pytest.raises(ValueError, match="cross-asserter"):
            await particle_retract(other_id, reason="nope")

    @pytest.mark.asyncio
    async def test_cross_asserter_allowed_with_flag(self, db_session: Any) -> None:
        from particles.mcp.tools.write import particle_retract

        _enable_writes(asserter_identity="mcp:test-agent", allow_cross_asserter=True)
        other_id = await _insert_foreign(
            db_session, asserted_by="mcp:other-agent", calib=CalibrationSource.EXTRACTOR_DIRECT
        )
        out = await particle_retract(other_id, reason="taking ownership")
        assert out["verdict"] == "RETRACTED"


class TestSupersede:
    @pytest.mark.asyncio
    async def test_supersede_transitions_predecessor(
        self, db_session: Any, stub_subjects: None
    ) -> None:
        from particles.mcp.tools.write import particle_assert, particle_supersede

        _enable_writes(asserter_identity="mcp:test-agent")
        first = await particle_assert("X is 5.", ["X"], 0.8, source_excerpt="x is 5")
        old_id = first["asserted_particle_id"]
        out = await particle_supersede(
            old_id, "X is 6.", ["X"], 0.85, source_excerpt="actually x is 6"
        )
        assert out["superseded_id"] == old_id
        async with session_scope(DEFAULT_STORE) as s:
            old = await get_particle(s, old_id)
            new = await get_particle(s, out["asserted_particle_id"])
        assert old is not None and new is not None
        assert old.status == Status.SUPERSEDED
        assert old.status_reason == StatusReason.EXPLICIT_SUPERSESSION
        assert new.status == Status.ACTIVE
        assert new.supersedes == old_id


class TestWriteSurfaceRegistration:
    @pytest.mark.asyncio
    async def test_write_tools_absent_by_default(self, db_session: Any) -> None:
        from particles.mcp import build_server

        names = {t.name for t in await build_server().list_tools()}
        assert "particle_assert" not in names  # default-deny → read-only surface

    @pytest.mark.asyncio
    async def test_write_tools_registered_when_enabled(self, db_session: Any) -> None:
        from particles.mcp import build_server

        _enable_writes()
        names = {t.name for t in await build_server().list_tools()}
        for tool in (
            "particle_assert",
            "particle_supersede",
            "particle_retract",
            "deposit_text",
            "link_add",
            "link_remove",
            "particle_tag",
            "particle_untag",
        ):
            assert tool in names


class TestMirrorTools:
    @pytest.mark.asyncio
    async def test_tag_and_untag(self, db_session: Any, stub_subjects: None) -> None:
        from particles.mcp.tools.write import particle_assert, particle_tag, particle_untag

        _enable_writes()
        pid = (await particle_assert("X.", ["X"], 0.8, source_excerpt="x"))["asserted_particle_id"]
        assert (await particle_tag(pid, ["alpha"]))["added"] == ["alpha"]
        assert (await particle_untag(pid, ["alpha"]))["removed"] == ["alpha"]

    @pytest.mark.asyncio
    async def test_link_add_and_remove(self, db_session: Any, stub_subjects: None) -> None:
        from particles.mcp.tools.write import link_add, link_remove, particle_assert

        _enable_writes()
        a = (await particle_assert("A.", ["X"], 0.8, source_excerpt="a"))["asserted_particle_id"]
        b = (await particle_assert("B.", ["X"], 0.8, source_excerpt="b"))["asserted_particle_id"]
        assert (await link_add(a, b))["verdict"] == "LINKED"
        assert (await link_remove(a, b))["removed"] is True

    @pytest.mark.asyncio
    async def test_link_rejects_non_co_evidential(self, db_session: Any) -> None:
        from particles.mcp.tools.write import link_add

        _enable_writes()
        with pytest.raises(ValueError, match="CO_EVIDENTIAL"):
            await link_add("a-id", "b-id", relation_type="CONTRADICTS")


class TestTrustBindingEndToEnd:
    """C1/§6a: the seeded AUTHOR trust actually binds at query-time resolution, so
    agent content ranks below operator content (not just that the seed is stored)."""

    @pytest.mark.asyncio
    async def test_agent_content_ranks_below_operator(
        self, db_session: Any, stub_subjects: None
    ) -> None:
        from particles.core.scoring.confidence import compute_effective_confidence
        from particles.mcp.tools.write import particle_assert
        from particles.operations.query.source_trust import load_trust_policy

        _enable_writes(agent_trust_rank=0.8, asserter_identity="mcp:test-agent")
        await particle_assert("X is five.", ["X"], 0.9, source_excerpt="x is five")

        async with session_scope(DEFAULT_STORE) as s:
            policy = await load_trust_policy(s)
        # The §6a binding: a CONVERSATION-sourced agent belief resolves to the
        # seeded 0.8 (not the neutral 1.0 it would get if the domain didn't bind).
        agent_rank = policy.evaluate(None, "CONVERSATION", None, "mcp:test-agent")
        assert agent_rank == pytest.approx(0.8)
        # Operator content carries no AUTHOR statement → neutral (None → 1.0).
        op_rank = policy.evaluate(None, "CONVERSATION", None, "operator")
        op_effective_rank = 1.0 if op_rank is None else op_rank

        agent_eff = compute_effective_confidence(0.9, source_trust_rank=agent_rank)
        op_eff = compute_effective_confidence(0.73, source_trust_rank=op_effective_rank)
        assert agent_eff == pytest.approx(0.72)  # 0.9 × 0.8
        # A modest operator belief outranks the most confident possible agent one.
        assert op_eff > agent_eff


class TestConsensusAndFailClosed:
    """The §6b/§6 engine behaviour exercised end-to-end through particle_assert."""

    @pytest.mark.asyncio
    async def test_contradiction_raises_inconsistency_not_supersede(
        self,
        db_session: Any,
        stub_subjects: None,
        similar_embeddings: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A write store reconciles in consensus mode (§6b): a confirmed contradiction
        surfaces as an INCONSISTENCY and the prior belief is NOT auto-superseded; the
        §7 contested marker then surfaces it on the recall path (M6)."""
        import particles.llm as llm
        from particles.mcp.tools.particles import particles_list
        from particles.mcp.tools.write import particle_assert
        from particles.store.particle_store import get_particle

        _enable_writes()  # default store is write-enabled -> resolves to multi
        monkeypatch.setattr(llm, "complete", AsyncMock(return_value="YES: they disagree"))

        a = await particle_assert(
            "Deploy key rotates monthly.", ["deploy key"], 0.8, source_excerpt="rotates monthly"
        )
        assert a["verdict"] == "ASSERTED"
        a_id = a["asserted_particle_id"]

        b = await particle_assert(
            "Deploy key never rotates.", ["deploy key"], 0.9, source_excerpt="never rotates"
        )
        assert b["verdict"] == "INCONSISTENCY_RAISED"
        inc_id = b["inconsistency_id"]
        cand_id = b["asserted_particle_id"]
        # M6 review fix: the belief id is the quarantined candidate, NOT the
        # INCONSISTENCY meta-particle's id.
        assert cand_id != inc_id

        async with session_scope(DEFAULT_STORE) as s:
            a_p = await get_particle(s, a_id)
            cand = await get_particle(s, cand_id)
            inc = await get_particle(s, inc_id)
        assert a_p is not None
        assert a_p.status == Status.ACTIVE  # prior belief preserved, not auto-superseded
        assert cand is not None and cand.status == Status.PROVENANCE_STALE  # quarantined belief
        assert inc is not None and inc.status == Status.INCONSISTENCY

        # M6: contested marker on the agent's own recall path.
        listing = await particles_list(status="ACTIVE")
        a_entry = next(p for p in listing["particles"] if p["id"] == a_id)
        assert a_entry["contested"] == inc_id

    @pytest.mark.asyncio
    async def test_probe_failure_fails_closed(
        self,
        db_session: Any,
        stub_subjects: None,
        similar_embeddings: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the contradiction probe cannot complete, the assertion fails closed
        (§6): quarantine + INCONSISTENCY, never two coexisting ACTIVE beliefs."""
        import particles.llm as llm
        from particles.mcp.tools.write import particle_assert

        _enable_writes()
        await particle_assert("Claim one.", ["X"], 0.8, source_excerpt="one")

        async def _boom(*args: Any, **kwargs: Any) -> str:
            raise RuntimeError("LLM unavailable")

        monkeypatch.setattr(llm, "complete", _boom)
        b = await particle_assert(
            "Claim one, restated differently.", ["X"], 0.8, source_excerpt="two"
        )
        assert b["verdict"] == "INCONSISTENCY_RAISED"
