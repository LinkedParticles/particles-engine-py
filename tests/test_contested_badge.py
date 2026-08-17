"""Tests for the composed contested badge (operations/query/contested.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from particles.core.schema import (
    Confidence,
    ContestedBadge,
    ContestednessReading,
    Particle,
    PolicyRendering,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    QueryResponse,
    RelationCreatedBy,
    RelationType,
    TrustLensDefinition,
    TrustLensUrlRule,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.stance import STANCE_HOLDER_KEY
from particles.core.status import Status
from particles.operations.query.contested import (
    _dispute_presence,
    compose_badge,
    compute_contested_badges,
)
from particles.operations.query.stance import AGREEMENT_CAVEAT

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Distinct valid-UUID ids per test to avoid cross-test collisions.
_T0 = "00000000-0000-0000-0000-0000000000e0"
_T1 = "00000000-0000-0000-0000-0000000000e1"
_S0 = "00000000-0000-0000-0000-0000000000e2"
_S1 = "00000000-0000-0000-0000-0000000000e3"
_I0 = "00000000-0000-0000-0000-0000000000e4"


def _claim(
    content: str,
    pid: str,
    entry_id: str = "ce-x",
    confidence: float = 0.9,
    properties: dict[str, object] | None = None,
) -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        properties=properties,
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)],
    )


def _inconsistency(pid: str, target_id: str) -> Particle:
    return Particle(
        id=pid,
        content="Conflict between beliefs.",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="lint",
        status=Status.INCONSISTENCY,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.PARTICLE, corpus_entry_id=target_id, snapshot_id=None
            )
        ],
    )


def _reading(spread: float) -> ContestednessReading:
    return ContestednessReading(
        spread=spread,
        renderings=[
            PolicyRendering(policy="local", effective_confidence=0.9),
            PolicyRendering(policy="acme", effective_confidence=0.9 - spread),
        ],
    )


# ---------------------------------------------------------------------------
# compose_badge — the pure §2 gates and §3 absence semantics
# ---------------------------------------------------------------------------


def test_stance_alone_fires_and_carries_m6_caveat() -> None:
    """§2: one DISPUTES position fires the stance basis; the M6 caveat MUST ride."""
    badge = compose_badge(has_dispute=True, reading=None, inconsistency_id=None)
    assert badge is not None
    assert badge.bases == ["stance"]
    assert badge.caveat == AGREEMENT_CAVEAT
    assert badge.inconsistency_id is None


def test_divergence_fires_at_callout_threshold() -> None:
    """§2: the divergence gate is the existing callout_threshold (default 0.2)."""
    badge = compose_badge(has_dispute=False, reading=_reading(0.2), inconsistency_id=None)
    assert badge is not None
    assert badge.bases == ["divergence"]
    assert badge.caveat is None  # M6 rides only with the stance basis


def test_divergence_below_threshold_does_not_fire() -> None:
    assert compose_badge(has_dispute=False, reading=_reading(0.19), inconsistency_id=None) is None


def test_divergence_absent_is_not_a_vote() -> None:
    """§3: reading=None (fewer than two policies) is absence, not non-firing —
    and with no other basis fired the claim carries no badge, never an
    explicit 'uncontested'."""
    assert compose_badge(has_dispute=False, reading=None, inconsistency_id=None) is None


def test_inconsistency_alone_fires_with_drilldown() -> None:
    badge = compose_badge(has_dispute=False, reading=None, inconsistency_id=_I0)
    assert badge is not None
    assert badge.bases == ["inconsistency"]
    assert badge.inconsistency_id == _I0
    assert badge.caveat is None


def test_disjunction_carries_every_fired_basis() -> None:
    """§1: the badge is a basis-carrying disjunction, never a blended scalar."""
    badge = compose_badge(has_dispute=True, reading=_reading(0.5), inconsistency_id=_I0)
    assert badge is not None
    assert badge.bases == ["stance", "divergence", "inconsistency"]
    assert badge.inconsistency_id == _I0
    assert badge.caveat == AGREEMENT_CAVEAT


def test_badge_model_rejects_empty_bases() -> None:
    """§4: a bare 'contested' with no basis is non-conforming."""
    with pytest.raises(ValueError):
        ContestedBadge(bases=[])


# ---------------------------------------------------------------------------
# _dispute_presence — the stance basis's presence question (rules)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endorse_only_never_fires(db_session: AsyncSession) -> None:
    """§2: endorsements alone never fire — agreement is not contest."""
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation

    t = _claim("A claim.", _T0)
    s = _claim("alice endorses.", _S0, properties={STANCE_HOLDER_KEY: "x:alice"})
    for p in (t, s):
        await insert_particle(db_session, p)
    await create_relation(
        db_session, s.id, t.id, RelationType.ENDORSES, RelationCreatedBy.EXTRACTOR_DIRECT
    )
    await db_session.commit()

    assert await _dispute_presence(db_session, [t]) == [False]
    badges = await compute_contested_badges(db_session, [t])
    assert badges == [None]


@pytest.mark.asyncio
async def test_active_dispute_fires_including_co_evidential_twin(
    db_session: AsyncSession,
) -> None:
    """A DISPUTES edge into the claim's CO_EVIDENTIAL group fires the basis."""
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation

    t = _claim("phrasing A", _T0)
    twin = _claim("phrasing B", _T1)
    s = _claim("bob disputes.", _S0, properties={STANCE_HOLDER_KEY: "x:bob"})
    for p in (t, twin, s):
        await insert_particle(db_session, p)
    await create_relation(
        db_session, t.id, twin.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, s.id, twin.id, RelationType.DISPUTES, RelationCreatedBy.EXTRACTOR_DIRECT
    )
    await db_session.commit()

    assert await _dispute_presence(db_session, [t]) == [True]
    badges = await compute_contested_badges(db_session, [t])
    assert badges[0] is not None
    assert badges[0].bases == ["stance"]
    assert badges[0].caveat == AGREEMENT_CAVEAT


@pytest.mark.asyncio
async def test_dangling_and_holderless_disputes_are_excluded(db_session: AsyncSession) -> None:
    """a retracted or holder-less stance contributes no position,
    so it cannot fire the basis."""
    from particles.store.particle_store import insert_particle, update_particle_status
    from particles.store.relation_store import create_relation

    t = _claim("A claim.", _T0)
    retracted = _claim("was disputed.", _S0, properties={STANCE_HOLDER_KEY: "x:carol"})
    holderless = _claim("dispute, unattributed.", _S1)
    for p in (t, retracted, holderless):
        await insert_particle(db_session, p)
    for sid in (retracted.id, holderless.id):
        await create_relation(
            db_session, sid, t.id, RelationType.DISPUTES, RelationCreatedBy.EXTRACTOR_DIRECT
        )
    await update_particle_status(db_session, retracted.id, Status.RETRACTED)
    await db_session.commit()

    assert await _dispute_presence(db_session, [t]) == [False]


# ---------------------------------------------------------------------------
# compute_contested_badges — inconsistency backref + divergence wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inconsistency_backref_fires_the_basis(db_session: AsyncSession) -> None:
    from particles.store.particle_store import insert_particle

    t = _claim("A claim.", _T0)
    await insert_particle(db_session, t)
    await insert_particle(db_session, _inconsistency(_I0, t.id))
    await db_session.commit()

    badges = await compute_contested_badges(db_session, [t])
    assert badges[0] is not None
    assert badges[0].bases == ["inconsistency"]
    assert badges[0].inconsistency_id == _I0


@pytest.mark.asyncio
async def test_divergence_absent_below_two_policies(db_session: AsyncSession) -> None:
    """§3: the zero-lens store never mints a divergence basis (and pays nothing)."""
    from particles.store.particle_store import insert_particle

    t = _claim("A claim.", _T0)
    await insert_particle(db_session, t)
    await db_session.commit()

    assert await compute_contested_badges(db_session, [t]) == [None]


async def _adopt_demoting_lens(session: AsyncSession) -> None:
    from particles.store.lens_store import adopt_lens, materialise_lens

    lens = TrustLensDefinition(
        name="acme",
        version=1,
        url_rules=[TrustLensUrlRule(scope="domain", pattern="sketchy.example", score=0.2)],
        extractor_weights={},
    )
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
async def test_divergence_fires_with_adopted_lens(db_session: AsyncSession) -> None:
    """With ≥2 policies and a spread ≥ callout_threshold the divergence basis fires."""
    from particles.store.particle_store import insert_particle

    await _adopt_demoting_lens(db_session)
    await _add_entry(db_session, "e1", "https://sketchy.example/p")
    t = _claim("Sketchy claim.", _T0, entry_id="e1")
    await insert_particle(db_session, t)
    await db_session.commit()

    badges = await compute_contested_badges(db_session, [t])
    assert badges[0] is not None
    assert badges[0].bases == ["divergence"]


# ---------------------------------------------------------------------------
# Query envelope: default-on attach, kill switch, never-affects-ranking (§4)
# ---------------------------------------------------------------------------


def _mock_embeddings() -> object:
    from particles import embeddings as ep

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original = ep._embedding_model
    ep.set_embedding_model(mock_model)
    return original


@pytest.mark.asyncio
async def test_query_attaches_badges_by_default_and_off_restores_ranking(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default-on: `contested` is parallel to `particles`; badge_enabled=False
    restores today's envelope exactly, and toggling the badge changes no
    ordering or effective confidence (§4 never-affects-ranking)."""
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation

    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    t = _claim("Disputed claim.", _T0, confidence=0.9)
    other = _claim("Quiet claim.", _T1, confidence=0.8)
    s = _claim("bob disputes.", _S0, properties={STANCE_HOLDER_KEY: "x:bob"})
    await insert_particle(db_session, t, emb)
    await insert_particle(db_session, other, emb)
    await insert_particle(db_session, s, emb)
    await create_relation(
        db_session, s.id, t.id, RelationType.DISPUTES, RelationCreatedBy.EXTRACTOR_DIRECT
    )
    await db_session.commit()

    original = _mock_embeddings()
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="ok"))
    try:
        on = await qmain.query(db_session, QueryRequest(question="claims?", top_k=5))
        get_config().contestedness.badge_enabled = False
        off = await qmain.query(db_session, QueryRequest(question="claims?", top_k=5))
    finally:
        ep.set_embedding_model(original)  # type: ignore[arg-type]

    # Default-on: one badge slot per ranked particle; the disputed claim carries
    # the stance basis + M6 caveat, the quiet claim carries None (never an
    # explicit "uncontested").
    assert len(on.contested) == len(on.particles)
    by_id = dict(zip([p.id for p in on.particles], on.contested, strict=True))
    badge = by_id[t.id]
    assert badge is not None and badge.bases == ["stance"]
    assert badge.caveat == AGREEMENT_CAVEAT
    assert by_id[other.id] is None

    # Kill switch: today's behavior exactly — no badges attached.
    assert off.contested == []
    # §4: the badge never feeds ranking, scores, or filtering.
    assert [p.id for p in off.particles] == [p.id for p in on.particles]
    assert off.effective_confidences == pytest.approx(on.effective_confidences)


@pytest.mark.asyncio
async def test_query_badge_reuses_optin_readings(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """include_contestedness + badge: the divergence basis agrees with the
    attached readings (single threshold, §2)."""
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.store.particle_store import insert_particle

    await _adopt_demoting_lens(db_session)
    await _add_entry(db_session, "e1", "https://sketchy.example/p")
    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    t = _claim("Sketchy claim.", _T0, entry_id="e1")
    await insert_particle(db_session, t, emb)
    await db_session.commit()

    original = _mock_embeddings()
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="ok"))
    try:
        resp = await qmain.query(
            db_session,
            QueryRequest(question="claim?", top_k=5, include_contestedness=True),
        )
    finally:
        ep.set_embedding_model(original)  # type: ignore[arg-type]

    assert len(resp.contestedness) == len(resp.particles) == 1
    badge = resp.contested[0]
    assert badge is not None and badge.bases == ["divergence"]
    assert resp.contestedness[0].spread >= 0.2


# ---------------------------------------------------------------------------
# Digest + MEMORY.md back-compat: bases added, off restores today's flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_badge_off_restores_inconsistency_only_flag(
    db_session: AsyncSession,
) -> None:
    from particles.config import get_config
    from particles.db import DEFAULT_STORE
    from particles.operations.digest import build_digest
    from particles.store.particle_store import insert_particle

    t = _claim("Deploy key rotates monthly.", _T0)
    await insert_particle(db_session, t)
    await insert_particle(db_session, _inconsistency(_I0, t.id))
    await db_session.commit()

    on = await build_digest(DEFAULT_STORE)
    assert f"contested (inconsistency) by `{_I0}`" in on

    get_config().contestedness.badge_enabled = False
    off = await build_digest(DEFAULT_STORE)
    assert f"contested by `{_I0}`" in off
    assert "contested (" not in off


def test_memory_bullet_renders_bases_and_keeps_drilldown() -> None:
    from particles.render.markdown import format_memory_bullet

    out = format_memory_bullet(
        "CI floors at 3.11.",
        "08d3e100",
        contested_by="9c447100",
        contested_bases=("stance", "inconsistency"),
    )
    assert out == (
        "- ⚠ contested (stance, inconsistency) — CI floors at 3.11. (vs. p-9c447100) `p-08d3e100`"
    )
    # Bases without an inconsistency drill-down: no "(vs. …)" detail.
    assert format_memory_bullet("X.", "aa", contested_bases=("divergence",)) == (
        "- ⚠ contested (divergence) — X. `p-aa`"
    )
    # No bases supplied → the pre-badge flag byte-for-byte (belt).
    assert format_memory_bullet("X.", "aa", contested_by="bb") == (
        "- ⚠ contested — X. (vs. p-bb) `p-aa`"
    )


# ---------------------------------------------------------------------------
# The composed [!contested] callout — one callout, per-basis attribution
# ---------------------------------------------------------------------------


def test_render_contested_callout_composed() -> None:
    from particles.core.schema import StancePosition
    from particles.render.markdown import render_contested_callout

    badge = ContestedBadge(
        bases=["stance", "divergence", "inconsistency"],
        inconsistency_id=_I0,
        caveat=AGREEMENT_CAVEAT,
    )
    positions = [
        StancePosition(
            kind=RelationType.DISPUTES,
            holder="x:bob",
            stance_particle_id=_S0,
            effective_confidence=0.7,
        )
    ]
    out = render_contested_callout(_reading(0.5), badge=badge, positions=positions)
    assert out.count("[!contested]") == 1
    assert "stance, divergence, inconsistency" in out
    assert "x:bob" in out  # stance attribution
    assert "count of keys, not of verified agents" in out  # M6 caveat travels
    assert "spread 0.50" in out  # divergence attribution
    assert f"`{_I0}`" in out  # inconsistency drill-down


def test_render_particles_one_callout_agreement_for_endorse_only() -> None:
    from particles.core.schema import StancePosition
    from particles.render.markdown import render_particles

    disputed = _claim("Disputed.", _T0)
    endorsed = _claim("Endorsed.", _T1)
    badge = ContestedBadge(bases=["stance"], caveat=AGREEMENT_CAVEAT)
    dists = [
        [
            StancePosition(
                kind=RelationType.DISPUTES,
                holder="x:bob",
                stance_particle_id=_S0,
                effective_confidence=0.7,
            )
        ],
        [
            StancePosition(
                kind=RelationType.ENDORSES,
                holder="x:alice",
                stance_particle_id=_S1,
                effective_confidence=0.9,
            )
        ],
    ]
    out = render_particles(
        [disputed, endorsed],
        agreement_distributions=dists,
        contested=[badge, None],
    )
    # One composed [!contested] for the disputed claim; [!agreement] remains
    # for the endorse-only (uncontested) distribution.
    assert out.count("[!contested]") == 1
    assert out.count("[!agreement]") == 1
    assert "x:bob" in out and "x:alice" in out


# ---------------------------------------------------------------------------
# MCP back-compat: `contested` key unchanged, `contested_bases` added
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_query_summary_carries_contested_bases(db_session: AsyncSession) -> None:
    from particles.mcp.tools.query import query as mcp_query

    p = _claim("atomic claim", _T0)
    fake_resp = QueryResponse(
        answer="The answer.",
        particles=[p],
        effective_confidences=[0.42],
        contested=[ContestedBadge(bases=["stance"], caveat=AGREEMENT_CAVEAT)],
    )
    with patch("particles.operations.query.query", new=AsyncMock(return_value=fake_resp)):
        result = await mcp_query("anything?", summary=True)

    hit = result["particles"][0]
    # The id-valued key keeps its meaning (not contested here) …
    assert hit["contested"] is None
    # … and the fired bases ride beside it.
    assert hit["contested_bases"] == ["stance"]


@pytest.mark.asyncio
async def test_mcp_particles_list_contested_bases(db_session: AsyncSession) -> None:
    from particles.config import get_config
    from particles.mcp.tools.particles import particles_list
    from particles.store.particle_store import insert_particle

    t = _claim("Contested belief.", _T0)
    quiet = _claim("Quiet belief.", _T1)
    await insert_particle(db_session, t)
    await insert_particle(db_session, quiet)
    await insert_particle(db_session, _inconsistency(_I0, t.id))
    await db_session.commit()

    result = await particles_list(status="ACTIVE")
    by_id = {e["id"]: e for e in result["particles"]}
    # `contested` keeps its current meaning (the INCONSISTENCY id) …
    assert by_id[t.id]["contested"] == _I0
    assert by_id[quiet.id]["contested"] is None
    # … with the fired bases beside it, composed by the composer.
    assert by_id[t.id]["contested_bases"] == ["inconsistency"]
    assert by_id[quiet.id]["contested_bases"] is None

    # Kill switch: today's entry shape exactly (no contested_bases key).
    get_config().contestedness.badge_enabled = False
    result_off = await particles_list(status="ACTIVE")
    assert all("contested_bases" not in e for e in result_off["particles"])
    assert {e["id"]: e["contested"] for e in result_off["particles"]}[t.id] == _I0


@pytest.mark.asyncio
async def test_mcp_listing_and_query_agree_on_a_stance_only_claim(
    db_session: AsyncSession,
) -> None:
    """the listing evaluates all three bases, like `query` does.

    Before the fix it hand-rolled ``["inconsistency"] if p.id in backrefs``, so
    a claim contested only by a stance rendered *uncontested* on the
    listing while the ``query`` tool on the same server badged it.
    """
    from particles.mcp.tools.particles import particles_list
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation

    t = _claim("Disputed by a stance only.", _T0)
    s = _claim("bob disputes.", _S0, properties={STANCE_HOLDER_KEY: "x:bob"})
    for p in (t, s):
        await insert_particle(db_session, p)
    await create_relation(
        db_session, s.id, t.id, RelationType.DISPUTES, RelationCreatedBy.EXTRACTOR_DIRECT
    )
    await db_session.commit()

    listing = {e["id"]: e for e in (await particles_list(status="ACTIVE"))["particles"]}
    composed = await compute_contested_badges(db_session, [t])

    assert composed[0] is not None and composed[0].bases == ["stance"]
    assert listing[t.id]["contested_bases"] == ["stance"]
    # The id-valued key is unmoved: no INCONSISTENCY, so no drill-down.
    assert listing[t.id]["contested"] is None
    # The stance particle itself is uncontested — the badge is per claim.
    assert listing[s.id]["contested_bases"] is None


# ---------------------------------------------------------------------------
# graph_view: the §7 kill switch reaches the graph surface too
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_view_honours_the_badge_kill_switch(db_session: AsyncSession) -> None:
    from particles.config import get_config
    from particles.core.schema import Subject
    from particles.operations.graph_view import build_graph_data
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject

    subj = Subject(canonical_name="Deploys", asserted_by="test")
    await insert_subject(db_session, subj)
    t = _claim("Contested in the graph.", _T0)
    t = t.model_copy(update={"subject_ids": [subj.id]})
    await insert_particle(db_session, t)
    await insert_particle(db_session, _inconsistency(_I0, t.id))
    await db_session.commit()

    on = await build_graph_data(db_session, subject_id=subj.id)
    assert on.particles[t.id].contested is not None
    assert on.particles[t.id].contested.bases == ["inconsistency"]
    assert any(n.contested for n in on.nodes)

    get_config().contestedness.badge_enabled = False
    off = await build_graph_data(db_session, subject_id=subj.id)
    assert off.particles[t.id].contested is None
    assert all(not n.contested for n in off.nodes)


# ---------------------------------------------------------------------------
# CLI: the default per-result badge line
# ---------------------------------------------------------------------------


def test_cli_query_prints_badge_line() -> None:
    from particles.api.cli import app

    p = _claim("Disputed claim about deploys.", _T0)
    resp = QueryResponse(
        answer="The answer.",
        particles=[p],
        effective_confidences=[0.9],
        contested=[
            ContestedBadge(
                bases=["stance", "inconsistency"],
                inconsistency_id=_I0,
                caveat=AGREEMENT_CAVEAT,
            )
        ],
    )
    backend = MagicMock()
    backend.remote = False
    backend.query = AsyncMock(return_value=resp)
    with patch("particles.api.cli.query.get_backend", return_value=backend):
        result = CliRunner().invoke(app, ["query", "anything?"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "⚠ contested (stance, inconsistency) — Disputed claim about deploys." in result.output
    assert f"(vs. p-{_I0[:8]})" in result.output
    assert "count of keys, not of verified agents" in result.output  # M6 note
