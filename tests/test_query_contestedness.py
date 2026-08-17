"""Tests for per-claim contestedness (operations/query/contestedness.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    RelationCreatedBy,
    RelationType,
    TrustLensDefinition,
    TrustLensUrlRule,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource, merge_co_evidential_confidence
from particles.operations.query.contestedness import (
    MemberPolicy,
    compute_contestedness,
    load_member_policies,
    spread_for_group,
)
from particles.operations.query.source_trust import EMPTY_TRUST_POLICY, TrustPolicy

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Distinct valid-UUID ids per test to avoid cross-test collisions.
_P0 = "00000000-0000-0000-0000-0000000000f0"
_P1 = "00000000-0000-0000-0000-0000000000f1"
_P2 = "00000000-0000-0000-0000-0000000000f2"


def _claim(content: str, pid: str, entry_id: str, confidence: float = 0.9) -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)],
    )


def _local() -> MemberPolicy:
    return MemberPolicy(name="local", trust=EMPTY_TRUST_POLICY, extractor_weights={})


def _demoting_lens(name: str = "acme", domain_rank: float = 0.2) -> MemberPolicy:
    """A member that demotes the sketchy.example domain to ``domain_rank``."""
    return MemberPolicy(
        name=name,
        trust=TrustPolicy(
            statements={}, domain_scores={"sketchy.example": domain_rank}, url_patterns=()
        ),
        extractor_weights={},
    )


# Provenance row a particle on the lens-demoted domain resolves to.
_SKETCHY_ROW = (None, "WEB_PAGE", "e1", "https://sketchy.example/p", None)
_NEUTRAL_ROW = (None, "WEB_PAGE", "e2", "https://neutral.example/p", None)


# ---------------------------------------------------------------------------
# Pure / in-memory: MemberPolicy + spread_for_group (§3)
# ---------------------------------------------------------------------------


def test_member_effective_confidence_applies_trust_and_weight() -> None:
    p = _claim("c", _P0, "e1")
    # Local (neutral) renders the raw value; the demoting lens renders value × rank.
    assert _local().effective_confidence(p, _SKETCHY_ROW) == pytest.approx(0.9)
    assert _demoting_lens().effective_confidence(p, _SKETCHY_ROW) == pytest.approx(0.9 * 0.2)


def test_two_policy_spread_is_attributed_and_local_first() -> None:
    """The common per-claim case: local + one lens, singleton group."""
    p = _claim("c", _P0, "e1")
    members = [_local(), _demoting_lens()]
    reading = spread_for_group(members, [(p, "e1")], {p.id: _SKETCHY_ROW})
    assert reading is not None
    assert reading.spread == pytest.approx(0.9 - 0.18)
    # Renderings preserve member order — local first — so extremes are nameable.
    assert [r.policy for r in reading.renderings] == ["local", "acme"]
    assert reading.renderings[0].effective_confidence == pytest.approx(0.9)
    assert reading.renderings[1].effective_confidence == pytest.approx(0.18)


def test_fewer_than_two_members_is_none() -> None:
    """§3 degeneracy: a one-policy set mints no metric (absent, never 0.0)."""
    p = _claim("c", _P0, "e1")
    assert spread_for_group([_local()], [(p, "e1")], {p.id: _SKETCHY_ROW}) is None


def test_empty_group_is_none() -> None:
    assert spread_for_group([_local(), _demoting_lens()], [], {}) is None


def test_invariant_claim_has_zero_spread() -> None:
    """A claim no member demotes renders identically — spread 0.0 (a 'fact').

    Zero spread with ≥2 members is *measured invariance* (allowed), distinct from
    the §3 absence of a one-policy store.
    """
    p = _claim("c", _P0, "e2")
    reading = spread_for_group([_local(), _demoting_lens()], [(p, "e2")], {p.id: _NEUTRAL_ROW})
    assert reading is not None
    assert reading.spread == pytest.approx(0.0)


def test_group_merge_then_spread() -> None:
    """§1: noisy-OR merge over the group *first*, per policy, then spread."""
    p1 = _claim("c1", _P0, "e-a", confidence=0.9)
    p2 = _claim("c2", _P1, "e-b", confidence=0.8)
    members = [_local(), _demoting_lens()]
    group = [(p1, "e-a"), (p2, "e-b")]  # distinct sources → full independence
    rows = {
        p1.id: _SKETCHY_ROW,
        p2.id: (None, "WEB_PAGE", "e-b", "https://sketchy.example/q", None),
    }

    reading = spread_for_group(members, group, rows)
    assert reading is not None
    # Expected per-policy merged values, computed the same way the metric does.
    local_merged = merge_co_evidential_confidence([(0.9, "e-a"), (0.8, "e-b")])
    lens_merged = merge_co_evidential_confidence([(0.9 * 0.2, "e-a"), (0.8 * 0.2, "e-b")])
    assert reading.renderings[0].effective_confidence == pytest.approx(local_merged)
    assert reading.renderings[1].effective_confidence == pytest.approx(lens_merged)
    assert reading.spread == pytest.approx(local_merged - lens_merged)
    # Merge then spread ≠ spread then merge: the merged local value exceeds either
    # singleton, confirming the group merge ran before the range was taken.
    assert local_merged > 0.9


# ---------------------------------------------------------------------------
# DB integration: load_member_policies + compute_contestedness (§3)
# ---------------------------------------------------------------------------


def _lens_def(name: str = "acme-numismatics") -> TrustLensDefinition:
    return TrustLensDefinition(
        name=name,
        version=1,
        url_rules=[TrustLensUrlRule(scope="domain", pattern="sketchy.example", score=0.2)],
        extractor_weights={},
    )


async def _adopt(session: AsyncSession, lens: TrustLensDefinition) -> None:
    from particles.store.lens_store import adopt_lens, materialise_lens

    await materialise_lens(session, lens)
    await adopt_lens(session, lens.name)


async def _add_entry(session: AsyncSession, entry_id: str, uri_r: str) -> None:
    from datetime import UTC, datetime

    from particles.corpus.store import CorpusEntryRow

    session.add(
        CorpusEntryRow(
            entry_id=entry_id,
            uri_r=uri_r,
            source_type="WEB_PAGE",
            mutability="MUTABLE",
            fetch_policy="LAZY",
            created_at=datetime.now(UTC),
            deposited_by="test",
        )
    )
    await session.flush()


@pytest.mark.asyncio
async def test_load_member_policies_local_only(db_session: AsyncSession) -> None:
    """With no lens adopted the policy set is just the local member (§2/§3)."""
    members = await load_member_policies(db_session)
    assert [m.name for m in members] == ["local"]


@pytest.mark.asyncio
async def test_load_member_policies_lights_up_at_first_adoption(db_session: AsyncSession) -> None:
    await _adopt(db_session, _lens_def())
    await db_session.commit()
    members = await load_member_policies(db_session)
    assert [m.name for m in members] == ["local", "acme-numismatics"]


@pytest.mark.asyncio
async def test_compute_contestedness_absent_below_two_policies(db_session: AsyncSession) -> None:
    """§3: one-policy store → the metric is absent (empty), not a 0.0 reading."""
    p = _claim("c", _P0, "e1")
    members = await load_member_policies(db_session)
    assert await compute_contestedness(db_session, [p], members) == []


@pytest.mark.asyncio
async def test_compute_contestedness_two_policies_diverge(db_session: AsyncSession) -> None:
    from particles.store.particle_store import insert_particle

    await _adopt(db_session, _lens_def())
    await _add_entry(db_session, "e1", "https://sketchy.example/p")
    p = _claim("Sketchy claim.", _P0, "e1")
    await insert_particle(db_session, p)
    await db_session.commit()

    members = await load_member_policies(db_session)
    readings = await compute_contestedness(db_session, [p], members)
    assert len(readings) == 1
    reading = readings[0]
    assert reading.spread > 0.0
    by_policy = {r.policy: r.effective_confidence for r in reading.renderings}
    # The lens demotes the source; local stays neutral, so local renders higher.
    assert by_policy["local"] > by_policy["acme-numismatics"]
    assert by_policy["local"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_compute_contestedness_group_collapses_twin(db_session: AsyncSession) -> None:
    """A co-evidential twin is merged into the target's reading (§1 group merge)."""
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation

    await _adopt(db_session, _lens_def())
    await _add_entry(db_session, "e-a", "https://sketchy.example/a")
    await _add_entry(db_session, "e-b", "https://sketchy.example/b")
    p1 = _claim("phrasing A", _P0, "e-a", confidence=0.9)
    p2 = _claim("phrasing B", _P1, "e-b", confidence=0.8)
    await insert_particle(db_session, p1)
    await insert_particle(db_session, p2)
    await create_relation(
        db_session, p1.id, p2.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    members = await load_member_policies(db_session)
    readings = await compute_contestedness(db_session, [p1], members)
    assert len(readings) == 1
    # The local merge over the twin exceeds the strongest singleton — proof the
    # group was merged before the spread was taken.
    by_policy = {r.policy: r.effective_confidence for r in readings[0].renderings}
    assert by_policy["local"] > 0.9


# ---------------------------------------------------------------------------
# The §5 MUST: contestedness never moves ranking or effective confidence
# ---------------------------------------------------------------------------


def _mock_embeddings() -> object:
    from particles import embeddings as ep

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original = ep._embedding_model
    ep.set_embedding_model(mock_model)
    return original


@pytest.mark.asyncio
async def test_include_contestedness_never_affects_ranking(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§5 invariant: turning the metric on changes no effective confidence or order."""
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.store.particle_store import insert_particle

    await _adopt(db_session, _lens_def())
    await _add_entry(db_session, "e1", "https://sketchy.example/p")
    await _add_entry(db_session, "e2", "https://neutral.example/p")
    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    await insert_particle(db_session, _claim("Sketchy claim.", _P0, "e1", 0.9), emb)
    await insert_particle(db_session, _claim("Neutral claim.", _P1, "e2", 0.8), emb)
    await db_session.commit()

    original = _mock_embeddings()
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="ok"))
    try:
        plain = await qmain.query(db_session, QueryRequest(question="claims?", top_k=5))
        withc = await qmain.query(
            db_session,
            QueryRequest(question="claims?", top_k=5, include_contestedness=True),
        )
    finally:
        ep.set_embedding_model(original)  # type: ignore[arg-type]

    # Same order, same effective confidences — the metric is disclosure, not discount.
    assert [p.id for p in withc.particles] == [p.id for p in plain.particles]
    assert withc.effective_confidences == pytest.approx(plain.effective_confidences)
    # Plain run carries no readings; the opt-in run carries one per result.
    assert plain.contestedness == []
    assert len(withc.contestedness) == len(withc.particles)


@pytest.mark.asyncio
async def test_query_contestedness_absent_without_lens(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """include_contestedness on a one-policy store yields an empty list (§3)."""
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.store.particle_store import insert_particle

    await _add_entry(db_session, "e1", "https://sketchy.example/p")
    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    await insert_particle(db_session, _claim("A claim.", _P0, "e1", 0.9), emb)
    await db_session.commit()

    original = _mock_embeddings()
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="ok"))
    try:
        result = await qmain.query(
            db_session, QueryRequest(question="claim?", top_k=5, include_contestedness=True)
        )
    finally:
        ep.set_embedding_model(original)  # type: ignore[arg-type]

    assert result.contestedness == []
