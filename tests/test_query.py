"""Tests for operations/query.py — §9.3."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from particles.core.schema import (
    AssertionModality,
    Confidence,
    ExtractorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource, compute_effective_confidence
from particles.core.status import Status


def _make_active_particle(
    content: str,
    confidence: float = 0.8,
    *,
    modality: AssertionModality = AssertionModality.FALSIFIABLE,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        assertion_modality=modality,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
    )


class TestEffectiveConfidence:
    def test_no_trust_weights(self) -> None:
        ec = compute_effective_confidence(0.8)
        assert ec == pytest.approx(0.8)

    def test_with_trust_weights(self) -> None:
        ec = compute_effective_confidence(0.8, extractor_trust_weight=0.9, source_trust_rank=0.85)
        assert ec == pytest.approx(0.8 * 0.9 * 0.85)

    def test_clamped_to_one(self) -> None:
        ec = compute_effective_confidence(1.1)
        assert ec == 1.0

    def test_clamped_to_zero(self) -> None:
        ec = compute_effective_confidence(-0.1)
        assert ec == 0.0


@pytest.mark.asyncio
async def test_query_returns_particles(db_session: object) -> None:
    """Query against a populated store returns results."""
    import numpy as np

    from particles import embeddings as ep
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    p = _make_active_particle("Water is composed of hydrogen and oxygen.", 0.95)
    emb = np.ones(4, dtype=np.float32)
    emb = emb / np.linalg.norm(emb)
    await insert_particle(session, p, emb.tolist())  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    # Mock the embedding model and LLM response
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])

    import anthropic

    mock_content = MagicMock()
    mock_content.text = "Water is composed of hydrogen and oxygen."
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)

    with MagicMock() as _m:
        import particles.operations.query as oq

        getattr(oq, "anthropic", None)
        try:
            # Patch the Anthropic client directly
            with __import__("unittest.mock", fromlist=["patch"]).patch(
                "anthropic.Anthropic", return_value=mock_client
            ):
                req = QueryRequest(question="What is water made of?", top_k=5)
                result = await query(session, req)  # type: ignore[arg-type]
                assert len(result.particles) >= 1
                assert result.particles[0].content == "Water is composed of hydrogen and oxygen."
        finally:
            ep.set_embedding_model(original_model)


@pytest.mark.asyncio
async def test_query_filters_by_assertion_modality(db_session: object) -> None:
    """an assertion_modality filter narrows the candidate set; unset returns all."""
    import numpy as np

    from particles import embeddings as ep
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    emb = (np.ones(4, dtype=np.float32) / np.linalg.norm(np.ones(4, dtype=np.float32))).tolist()
    fact = _make_active_particle("The store opened in 2020.")
    opinion = _make_active_particle(
        "The store has the best coffee in town.", modality=AssertionModality.EVALUATIVE
    )
    await insert_particle(session, fact, emb)  # type: ignore[arg-type]
    await insert_particle(session, opinion, emb)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    import anthropic

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    mock_content = MagicMock()
    mock_content.text = "An answer."
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "anthropic.Anthropic", return_value=mock_client
        ):
            # Unset → both modalities returned.
            req_all = QueryRequest(question="Tell me about the store.", top_k=5)
            ids_all = {p.id for p in (await query(session, req_all)).particles}  # type: ignore[arg-type]
            assert {fact.id, opinion.id} <= ids_all

            # FALSIFIABLE filter → the opinion is excluded.
            req_facts = QueryRequest(
                question="Tell me about the store.",
                top_k=5,
                assertion_modality=AssertionModality.FALSIFIABLE,
            )
            ids_facts = {p.id for p in (await query(session, req_facts)).particles}  # type: ignore[arg-type]
            assert fact.id in ids_facts
            assert opinion.id not in ids_facts
    finally:
        ep.set_embedding_model(original_model)


@pytest.mark.asyncio
async def test_query_include_ancestors_up_expansion(db_session: object) -> None:
    """--include-ancestors also matches particles tagged with a broader ancestor."""
    import numpy as np

    from particles import embeddings as ep
    from particles.core.schema import TagNode, TaxonomyDefinition
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle
    from particles.store.taxonomy_store import insert_taxonomy, set_particle_tags

    session = db_session  # type: ignore[assignment]
    await insert_taxonomy(  # type: ignore[arg-type]
        session,
        TaxonomyDefinition(
            name="Coins",
            version="1.0.0",
            author="tester",
            tags=[
                TagNode(tag="coins"),
                TagNode(tag="coins/by-region", parent="coins"),
                TagNode(tag="coins/by-region/germany", parent="coins/by-region"),
            ],
        ),
    )
    emb = (np.ones(4, dtype=np.float32) / np.linalg.norm(np.ones(4, dtype=np.float32))).tolist()
    leaf = _make_active_particle("A 1 Pfennig coin minted in Berlin.")
    broad = _make_active_particle("A general fact about coins.")
    await insert_particle(session, leaf, emb)  # type: ignore[arg-type]
    await insert_particle(session, broad, emb)  # type: ignore[arg-type]
    await set_particle_tags(session, leaf.id, ["coins/by-region/germany"])  # type: ignore[arg-type]
    await set_particle_tags(session, broad.id, ["coins"])  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    import anthropic

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    mock_content = MagicMock()
    mock_content.text = "An answer."
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "anthropic.Anthropic", return_value=mock_client
        ):
            # Without the flag: only the leaf-tagged particle matches.
            req = QueryRequest(
                question="coins from germany", top_k=5, tags=["coins/by-region/germany"]
            )
            ids = {p.id for p in (await query(session, req)).particles}  # type: ignore[arg-type]
            assert leaf.id in ids
            assert broad.id not in ids

            # With --include-ancestors: the ancestor-tagged particle is included too.
            req_anc = QueryRequest(
                question="coins from germany",
                top_k=5,
                tags=["coins/by-region/germany"],
                include_ancestors=True,
            )
            ids_anc = {p.id for p in (await query(session, req_anc)).particles}  # type: ignore[arg-type]
            assert {leaf.id, broad.id} <= ids_anc
    finally:
        ep.set_embedding_model(original_model)


@pytest.mark.asyncio
async def test_query_empty_store(db_session: object) -> None:
    """Query against empty store returns no particles and a helpful message."""
    from particles import embeddings as ep
    from particles.operations.query import query

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        with __import__("unittest.mock", fromlist=["patch"]).patch("anthropic.Anthropic"):
            req = QueryRequest(question="Anything?", top_k=5)
            result = await query(db_session, req)  # type: ignore[arg-type]
            assert result.particles == []
            assert "No relevant" in result.answer
    finally:
        ep.set_embedding_model(original_model)


async def test_query_federated_merges_stores(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """query_federated unions candidates from every store and ranks them under
    one viewer's lens."""
    from unittest.mock import AsyncMock

    import numpy as np

    import particles._orm_modules  # noqa: F401
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.db import DEFAULT_STORE, Base, get_engine, reset_engine, session_scope
    from particles.store.particle_store import insert_particle

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/default.db"
    cfg.storage.stores = {"other": f"sqlite+aiosqlite:///{tmp_path}/other.db"}

    for handle in (DEFAULT_STORE, "other"):
        engine = get_engine(handle)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    async with session_scope(DEFAULT_STORE) as s:
        await insert_particle(s, _make_active_particle("Default store fact.", 0.9), emb)
        await s.commit()
    async with session_scope("other") as s:
        await insert_particle(s, _make_active_particle("Other store fact.", 0.7), emb)
        await s.commit()

    # Fixed-vector embedding mock (matches the stored 4-dim vectors) so cosine is
    # equal for both and ranking falls to effective_confidence; mock the NL
    # responder so the test needs no LLM.
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="merged answer"))
    try:
        req = QueryRequest(question="facts?", top_k=10)
        result = await qmain.query_federated([DEFAULT_STORE, "other"], req)

        assert {p.content for p in result.particles} == {
            "Default store fact.",
            "Other store fact.",
        }
        # Higher-confidence default-store particle ranks first.
        assert result.particles[0].content == "Default store fact."
        assert result.answer == "merged answer"
    finally:
        ep.set_embedding_model(original_model)
        for handle in (DEFAULT_STORE, "other"):
            await get_engine(handle).dispose()
        reset_engine()


class TestRankingConfigKnobs:
    """The §9.3 ranking weights and truncation thresholds read config (P4-7)."""

    def test_combined_score_honors_configured_weights(self) -> None:
        from particles.config import get_config
        from particles.operations.query.main import _combined

        # Defaults: 0.6 × sim + 0.4 × eff_conf
        assert _combined(1.0, 0.0) == pytest.approx(0.6)
        assert _combined(0.0, 1.0) == pytest.approx(0.4)

        get_config().query.similarity_weight = 0.9
        get_config().query.confidence_weight = 0.1
        assert _combined(1.0, 0.0) == pytest.approx(0.9)
        assert _combined(0.0, 1.0) == pytest.approx(0.1)

    def test_truncation_warning_honors_configured_thresholds(self) -> None:
        from particles.config import get_config
        from particles.operations.query.main import _truncation_warning

        # Two near-identical scores straddle a top_k=1 cutoff → small gap →
        # warning with the defaults (gap < 0.05).
        scored = [
            (_make_active_particle("a"), 0.90, 0.90),
            (_make_active_particle("b"), 0.89, 0.89),
        ]
        top = scored[:1]
        assert _truncation_warning(scored, top, 1) is not None

        # Tightening the gap threshold below the actual gap (and keeping the
        # near-count above the near-cutoff tally) silences the warning.
        get_config().query.truncation_min_gap = 0.001
        get_config().query.truncation_near_count = 5
        assert _truncation_warning(scored, top, 1) is None


class TestConfidenceNote:
    """§6.3 OVERCONFIDENCE GUARD: uncalibrated includes AGENT_ASSERTED."""

    def _p(self, calib: CalibrationSource) -> Particle:
        return Particle(
            content="A claim.",
            confidence=Confidence(value=0.8, calibration_source=calib),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="x",
            status=Status.ACTIVE,
            provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1")],
        )

    def test_agent_asserted_triggers_provisional_note(self) -> None:
        from particles.operations.query.main import _confidence_note

        note = _confidence_note([self._p(CalibrationSource.AGENT_ASSERTED)], [0.7])
        assert "AGENT_ASSERTED" in note
        assert "provisional" in note

    def test_extractor_direct_still_triggers_provisional_note(self) -> None:
        from particles.operations.query.main import _confidence_note

        note = _confidence_note([self._p(CalibrationSource.EXTRACTOR_DIRECT)], [0.7])
        assert "EXTRACTOR_DIRECT" in note
        assert "provisional" in note

    def test_calibrated_high_confidence_no_note(self) -> None:
        from particles.operations.query.main import _confidence_note

        assert _confidence_note([self._p(CalibrationSource.CALIBRATED_BENCHMARK)], [0.8]) == ""

    def test_low_mean_confidence_note_takes_precedence(self) -> None:
        from particles.operations.query.main import _confidence_note

        note = _confidence_note([self._p(CalibrationSource.AGENT_ASSERTED)], [0.4])
        assert "below 0.6" in note

    def test_empty_returns_no_note(self) -> None:
        from particles.operations.query.main import _confidence_note

        assert _confidence_note([], []) == ""


# ---------------------------------------------------------------------------
# query-side narrative integration
# ---------------------------------------------------------------------------


def test_rank_score_demotes_narrative() -> None:
    """A NARRATIVE's combined score is multiplied by narrative_rank_weight at
    rank time, so it sorts below an equally-scored claim — but the reported
    effective_confidence (s[2]) is untouched."""
    from particles.config import get_config
    from particles.core.schema import ParticleType
    from particles.operations.query.main import _combined, _rank_score

    claim = _make_active_particle("a claim")
    narrative = _make_active_particle("a narrative").model_copy(
        update={"particle_type": ParticleType.NARRATIVE}
    )
    sim, eff = 0.9, 0.9
    weight = get_config().query.narrative_rank_weight
    assert _rank_score((claim, sim, eff)) == pytest.approx(_combined(sim, eff))
    assert _rank_score((narrative, sim, eff)) == pytest.approx(_combined(sim, eff) * weight)
    assert _rank_score((narrative, sim, eff)) < _rank_score((claim, sim, eff))


def test_is_code_symbol_keys_on_extractor_provenance() -> None:
    """a particle is a code symbol iff its extractor_ref names a
    structured code-symbol extractor — provenance the extractor already stamps."""
    from particles.operations.query.main import _is_code_symbol

    concept = _make_active_particle("a concept claim")
    assert not _is_code_symbol(concept)

    docstring = concept.model_copy(
        update={"extractor_ref": ExtractorRef(name="docstring-extractor", version="0.1.0")}
    )
    assert _is_code_symbol(docstring)


def test_rank_score_demotes_code_symbol_only_below_one() -> None:
    """a code-symbol particle's combined score is multiplied by
    code_symbol_weight at rank time. The default 1.0 is inert; below 1.0 it
    sorts below an equally-scored concept claim, which is itself unaffected."""
    from particles.operations.query.main import _combined, _rank_score

    concept = _make_active_particle("a concept claim")
    docstring = concept.model_copy(
        update={"extractor_ref": ExtractorRef(name="docstring-extractor", version="0.1.0")}
    )
    sim, eff = 0.9, 0.9

    # Default weight 1.0 is inert — docstring scores the same as the concept claim.
    assert _rank_score((docstring, sim, eff)) == pytest.approx(_combined(sim, eff))

    # Below 1.0, only the code-symbol particle is demoted.
    assert _rank_score((docstring, sim, eff), 0.3) == pytest.approx(_combined(sim, eff) * 0.3)
    assert _rank_score((docstring, sim, eff), 0.3) < _rank_score((concept, sim, eff), 0.3)
    assert _rank_score((concept, sim, eff), 0.3) == pytest.approx(_combined(sim, eff))


# ---------------------------------------------------------------------------
# document-precedence tie-break among detected conflicts.
# ---------------------------------------------------------------------------


def _key(year: int, ordinal: int = -1) -> tuple[datetime, int]:
    return (datetime(year, 1, 1, tzinfo=UTC), ordinal)


def test_rank_score_demotes_precedence_loser_only() -> None:
    """a particle in the precedence_demoted set has its combined score
    multiplied by document_precedence.rank_penalty at rank time. The default
    empty set is inert; a demoted particle sorts below an equally-scored,
    non-demoted one, whose score is unaffected — and effective_confidence (s[2])
    is untouched."""
    from particles.config import get_config
    from particles.operations.query.main import _combined, _rank_score

    older = _make_active_particle("older decision")
    newer = _make_active_particle("newer decision")
    sim, eff = 0.9, 0.9
    penalty = get_config().document_precedence.rank_penalty

    # Empty demotion set (default) is inert.
    assert _rank_score((older, sim, eff)) == pytest.approx(_combined(sim, eff))

    # With older in the demotion set, only it is demoted.
    demoted = frozenset({older.id})
    assert _rank_score((older, sim, eff), 1.0, demoted) == pytest.approx(
        _combined(sim, eff) * penalty
    )
    assert _rank_score((newer, sim, eff), 1.0, demoted) == pytest.approx(_combined(sim, eff))
    assert _rank_score((older, sim, eff), 1.0, demoted) < _rank_score(
        (newer, sim, eff), 1.0, demoted
    )


def test_precedence_demotions_demotes_older_in_conflict() -> None:
    """within a DETECTED conflict pair, the earlier-authored particle
    (smaller precedence key) is the demotion loser."""
    from particles.operations.query.precedence import precedence_demotions

    older = _make_active_particle("default was 50")
    newer = _make_active_particle("default is now 100")
    keys = {older.id: _key(2024, 56), newer.id: _key(2024, 57)}
    conflict = {frozenset({older.id, newer.id})}

    demoted = precedence_demotions([older, newer], conflict, keys)
    assert demoted == {older.id}


def test_precedence_demotions_does_not_reorder_non_conflicting() -> None:
    """two co-active complementary particles with NO detected conflict
    are never reordered, even when one is clearly later-authored."""
    from particles.operations.query.precedence import precedence_demotions

    older = _make_active_particle("topic A")
    newer = _make_active_particle("topic B")
    keys = {older.id: _key(2024, 56), newer.id: _key(2025, 99)}

    # No conflict pair → nothing demoted.
    assert precedence_demotions([older, newer], set(), keys) == set()


def test_precedence_demotions_inert_without_comparable_key() -> None:
    """a detected conflict where one side has no comparable precedence
    key is left alone — default-safe, do nothing rather than guess. A true tie
    (equal keys) is also inert."""
    from particles.operations.query.precedence import precedence_demotions

    a = _make_active_particle("claim a")
    b = _make_active_particle("claim b")
    conflict = {frozenset({a.id, b.id})}

    # b has no key → inert.
    assert precedence_demotions([a, b], conflict, {a.id: _key(2024, 56)}) == set()

    # Both keyed but equal → no defensible winner → inert.
    equal = {a.id: _key(2024, 56), b.id: _key(2024, 56)}
    assert precedence_demotions([a, b], conflict, equal) == set()


def test_precedence_demotions_inert_when_disabled() -> None:
    """with document_precedence.enabled=false the retrieve_ranked
    gate returns no demotions, reproducing pre-0166 ordering byte-for-byte. The
    gate lives in _precedence_demotions (the pure helper is config-free), and
    short-circuits before touching the session — so a None session is safe."""
    import asyncio

    from particles.config import get_config
    from particles.operations.query.main import _precedence_demotions

    older = _make_active_particle("older")
    newer = _make_active_particle("newer")
    scored = [(older, 0.9, 0.9), (newer, 0.9, 0.9)]
    conflict = {frozenset({older.id, newer.id})}

    # The autouse fixture calls reset_config() before each test, so this
    # mutation does not leak. enabled=False short-circuits to the empty set.
    get_config().document_precedence.enabled = False
    demoted = asyncio.run(_precedence_demotions(None, scored, {}, conflict))  # type: ignore[arg-type]
    assert demoted == frozenset()


def _particle_on_entry(content: str, entry_id: str) -> Particle:
    """An ACTIVE particle whose SOURCE provenance points at a specific entry."""
    return _make_active_particle(content).model_copy(
        update={
            "provenance": [
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=entry_id,
                    snapshot_id="snap-1",
                )
            ]
        }
    )


def test_build_precedence_keys_adr_ordinal_and_fallback() -> None:
    """the ADR id ordinal comes from the stamped supersession key;
    content_published_at is the general fallback; a particle with neither key
    component is omitted (no comparable key — default-safe)."""
    from particles.operations.query.precedence import build_precedence_keys

    adr = _particle_on_entry("an adr decision", "entry-adr")
    web = _particle_on_entry("a dated web claim", "entry-web")
    bare = _particle_on_entry("undated, non-genre", "entry-bare")

    pub = datetime(2025, 3, 1, tzinfo=UTC)
    keys = build_precedence_keys(
        [adr, web, bare],
        pub_at_by_id={adr.id: None, web.id: pub, bare.id: None},
        supersession_by_entry={"entry-adr": '{"key": "adr:0166"}'},
    )
    # adr: entry carries adr:0166, no pub_at → ordinal 166.
    assert keys[adr.id][1] == 166
    # web: pub_at only, no ADR genre key on its entry → (pub, -1).
    assert keys[web.id] == (pub, -1)
    # bare: no ADR key, no pub_at → omitted entirely (no comparable key).
    assert bare.id not in keys


class TestDocumentPrecedenceConfig:
    """config sub-model defaults + validation."""

    def test_defaults(self) -> None:
        from particles.config import DocumentPrecedenceConfig

        cfg = DocumentPrecedenceConfig()
        assert cfg.enabled is True
        assert cfg.rank_penalty == pytest.approx(0.6)

    def test_rank_penalty_bounds(self) -> None:
        import pydantic

        from particles.config import DocumentPrecedenceConfig

        # Valid endpoints.
        assert DocumentPrecedenceConfig(rank_penalty=0.0).rank_penalty == 0.0
        assert DocumentPrecedenceConfig(rank_penalty=1.0).rank_penalty == 1.0
        # Out of [0, 1] is rejected.
        with pytest.raises(pydantic.ValidationError):
            DocumentPrecedenceConfig(rank_penalty=1.5)
        with pytest.raises(pydantic.ValidationError):
            DocumentPrecedenceConfig(rank_penalty=-0.1)

    def test_wired_onto_root_config(self) -> None:
        from particles.config import ParticlesConfig

        assert ParticlesConfig().document_precedence.enabled is True


def test_particle_line_includes_modality_marker() -> None:
    """every answer-prompt line carries a <MODALITY> marker."""
    from particles.core.schema import AudienceHint
    from particles.operations.query.respond import _particle_line

    p = _make_active_particle("The author felt anxious.", modality=AssertionModality.EXPERIENTIAL)
    line = _particle_line(p, 0.9, AudienceHint.GENERAL)
    assert "<EXPERIENTIAL>" in line
    assert "The author felt anxious." in line


@pytest.mark.asyncio
async def test_query_demotes_and_expands_narrative(db_session: object) -> None:
    """End-to-end: a NARRATIVE hit ranks below an equally-scored claim and its
    SEQUENCE_IN constituents are returned in narrative_constituents."""
    import numpy as np

    from particles import embeddings as ep
    from particles.core.schema import ParticleType, RelationCreatedBy, RelationType
    from particles.llm import set_client
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation

    session = db_session  # type: ignore[assignment]
    emb = (np.ones(4, dtype=np.float32) / np.linalg.norm(np.ones(4))).tolist()

    plain = _make_active_particle("A plain claim about the day.")
    constituent = _make_active_particle("The author woke up tired.")
    narrative = _make_active_particle("A hard day the author got through.").model_copy(
        update={"particle_type": ParticleType.NARRATIVE}
    )
    for p in (plain, constituent, narrative):
        await insert_particle(session, p, emb)  # type: ignore[arg-type]
    await create_relation(
        session, constituent.id, narrative.id, RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
    )
    await session.commit()  # type: ignore[union-attr]

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    import anthropic

    mock_content = MagicMock()
    mock_content.text = "answer"
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(mock_client)
    try:
        result = await query(session, QueryRequest(question="how was the day?", top_k=5))  # type: ignore[arg-type]
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)

    # The narrative was expanded to its constituent.
    assert narrative.id in result.narrative_constituents
    assert [c.id for c in result.narrative_constituents[narrative.id]] == [constituent.id]

    # The narrative is demoted below an equally-scored non-narrative hit.
    ids = [p.id for p in result.particles]
    assert ids.index(narrative.id) > ids.index(plain.id)


# --- query relevance floor -----------------------------------------


def _mock_llm_client(answer: str) -> MagicMock:
    """A shared-client mock (particles.llm set_client seam) returning ``answer``."""
    import anthropic

    mock_content = MagicMock()
    mock_content.text = answer
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)
    return mock_client


@pytest.mark.asyncio
async def test_query_below_relevance_floor_refuses_without_llm(db_session: object) -> None:
    """Max top-k cosine below the floor → deterministic refusal, no LLM call,
    hits still returned, RelevanceNote populated (§3)."""
    import numpy as np

    from particles import embeddings as ep
    from particles.llm import set_client
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    p = _make_active_particle("Pluto was reclassified as a dwarf planet.", 0.9)
    await insert_particle(session, p, [1.0, 0.0, 0.0, 0.0])  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    # Query embedding orthogonal to the stored one → cosine 0.0 < floor 0.25.
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)])
    mock_client = _mock_llm_client("MUST NOT APPEAR")

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(mock_client)
    try:
        result = await query(session, QueryRequest(question="football", top_k=5))  # type: ignore[arg-type]
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)

    assert result.answer.startswith("The store holds no beliefs relevant to this question")
    mock_client.messages.create.assert_not_called()
    # Hits stay in the envelope as nearest-but-likely-unrelated transparency.
    assert len(result.particles) == 1
    assert result.relevance is not None
    assert result.relevance.below_floor is True
    assert result.relevance.max_similarity == 0.0
    assert result.relevance.floor == pytest.approx(0.25)
    assert result.answer_refused is True


@pytest.mark.asyncio
async def test_query_above_relevance_floor_answers_and_discloses(db_session: object) -> None:
    """An on-topic query answers via the LLM and carries a below_floor=False note."""
    import numpy as np

    from particles import embeddings as ep
    from particles.llm import set_client
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    p = _make_active_particle("Pluto was reclassified as a dwarf planet.", 0.9)
    emb = (np.ones(4, dtype=np.float32) / np.linalg.norm(np.ones(4))).tolist()
    await insert_particle(session, p, emb)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    mock_client = _mock_llm_client("Pluto is a dwarf planet.")

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(mock_client)
    try:
        result = await query(session, QueryRequest(question="Is Pluto a planet?", top_k=5))  # type: ignore[arg-type]
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)

    assert result.answer == "Pluto is a dwarf planet."
    assert result.relevance is not None
    assert result.relevance.below_floor is False
    assert result.relevance.max_similarity == pytest.approx(1.0, abs=1e-5)
    assert result.answer_generation_error is None
    assert result.answer_refused is False


@pytest.mark.asyncio
async def test_query_llm_refusal_marker_is_stripped_and_flagged(db_session: object) -> None:
    """The §4 responder-declared refusal: the NO_RELEVANT_KNOWLEDGE marker is
    stripped from the prose, ``answer_refused`` is set, and the truncation
    warning is suppressed (advice to widen top_k under a refusal is noise)."""
    import numpy as np

    from particles import embeddings as ep
    from particles.config import get_config
    from particles.llm import set_client
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    # Above-floor but non-bearing hits; more particles than top_k with tied
    # scores so the truncation heuristic would fire absent the suppression.
    emb = (np.ones(4, dtype=np.float32) / np.linalg.norm(np.ones(4))).tolist()
    for i in range(4):
        await insert_particle(  # type: ignore[arg-type]
            session, _make_active_particle(f"An unrelated engineering fact {i}.", 0.9), emb
        )
    await session.commit()  # type: ignore[union-attr]
    get_config().query.truncation_min_gap = 0.5  # tied scores → gap 0 < 0.5 → would warn

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    mock_client = _mock_llm_client(
        "NO_RELEVANT_KNOWLEDGE\nThe knowledge base holds nothing relevant to watermelons."
    )

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(mock_client)
    try:
        result = await query(session, QueryRequest(question="what is a watermelon?", top_k=2))  # type: ignore[arg-type]
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)

    assert result.answer == "The knowledge base holds nothing relevant to watermelons."
    assert result.answer_refused is True
    assert result.truncation_warning is None
    # The numeric verdict is untouched — this refusal came from the responder,
    # not the floor.
    assert result.relevance is not None
    assert result.relevance.below_floor is False


@pytest.mark.asyncio
async def test_query_marker_mid_answer_is_not_a_refusal(db_session: object) -> None:
    """A marker mention that is not the leading line stays prose — no flag."""
    import numpy as np

    from particles import embeddings as ep
    from particles.llm import set_client
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    p = _make_active_particle("Pluto was reclassified as a dwarf planet.", 0.9)
    emb = (np.ones(4, dtype=np.float32) / np.linalg.norm(np.ones(4))).tolist()
    await insert_particle(session, p, emb)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    answer_text = "Pluto is a dwarf planet. (NO_RELEVANT_KNOWLEDGE would look like this.)"
    mock_client = _mock_llm_client(answer_text)

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(mock_client)
    try:
        result = await query(session, QueryRequest(question="Is Pluto a planet?", top_k=5))  # type: ignore[arg-type]
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)

    assert result.answer == answer_text
    assert result.answer_refused is False


@pytest.mark.asyncio
async def test_query_llm_failure_is_disclosed_not_silent(db_session: object) -> None:
    """A failing answer generation is disclosed, never passed off as an answer.

    Historically ``respond.py`` caught every LLM failure (billing, network)
    and silently returned the concatenated particle contents as the "answer"
    — a bullet dump indistinguishable from a deliberate response. Now the
    fallback listing still renders, but the failure is named in the answer
    text itself and in ``QueryResponse.answer_generation_error``.
    """
    import numpy as np

    from particles import embeddings as ep
    from particles.llm import set_client
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    p = _make_active_particle("Pluto was reclassified as a dwarf planet.", 0.9)
    emb = (np.ones(4, dtype=np.float32) / np.linalg.norm(np.ones(4))).tolist()
    await insert_particle(session, p, emb)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    # An account-level provider failure at the shared-client seam (e.g.
    # Anthropic's 400 "credit balance is too low").
    import anthropic

    failing_client = MagicMock(spec=anthropic.Anthropic)
    failing_client.messages = MagicMock()
    failing_client.messages.create = MagicMock(
        side_effect=RuntimeError("Your credit balance is too low to access the Anthropic API.")
    )

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(failing_client)
    try:
        result = await query(session, QueryRequest(question="Is Pluto a planet?", top_k=5))  # type: ignore[arg-type]
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)

    # The failure is machine-readable…
    assert result.answer_generation_error is not None
    assert "credit balance" in result.answer_generation_error
    # …and the answer text says what it is: a disclosed listing, not prose.
    assert result.answer.startswith("[Answer generation unavailable:")
    assert "• Pluto was reclassified as a dwarf planet." in result.answer


@pytest.mark.asyncio
async def test_query_relevance_floor_zero_disables_gate(db_session: object) -> None:
    """floor=0.0 reproduces pre-0226 behaviour: the LLM answers even at sim 0."""
    import numpy as np

    from particles import embeddings as ep
    from particles.config import get_config
    from particles.llm import set_client
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    get_config().query.relevance_floor = 0.0
    p = _make_active_particle("Pluto was reclassified as a dwarf planet.", 0.9)
    await insert_particle(session, p, [1.0, 0.0, 0.0, 0.0])  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)])
    mock_client = _mock_llm_client("An answer.")

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(mock_client)
    try:
        result = await query(session, QueryRequest(question="football", top_k=5))  # type: ignore[arg-type]
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)

    assert result.answer == "An answer."
    assert result.relevance is not None
    assert result.relevance.below_floor is False


@pytest.mark.asyncio
async def test_query_relevance_inert_without_embedding_model(db_session: object) -> None:
    """No embedding model → sim column is the eff_conf fallback, not a
    relevance signal: the verdict must stay absent.

    Another change amends the *other* half of this case. The verdict staying absent
    is still right — there is no similarity to judge — but absence was the
    whole disclosure, and it reads identically to a legitimately inert query.
    The answer now carries the degradation note, so this pins both: relevance
    inert, degradation stated.
    """
    from unittest.mock import patch

    from particles.llm import set_client
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    p = _make_active_particle("Pluto was reclassified as a dwarf planet.", 0.9)
    await insert_particle(session, p, [1.0, 0.0, 0.0, 0.0])  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    mock_client = _mock_llm_client("An answer.")
    set_client(mock_client)
    try:
        with patch("particles.embeddings.get_embedding_model", return_value=None):
            result = await query(session, QueryRequest(question="football", top_k=5))  # type: ignore[arg-type]
    finally:
        set_client(None)

    assert result.relevance is None
    # The generated prose is preserved verbatim — the note is prepended, not a
    # replacement, because the beliefs really were retrieved and shown.
    assert result.answer.endswith("An answer.")
    assert result.ranking_degraded is not None
    assert "relevance floor" in result.ranking_degraded


@pytest.mark.asyncio
async def test_query_without_an_encoder_discloses_the_degradation(
    db_session: object, no_embedding_model: None
) -> None:
    """an encoder-free query must say the hits are not about the question.

    Without a query vector the ranking aliases similarity to effective
    confidence, so the top-k is the store's most-confident beliefs whatever was
    asked — and `relevance` is None, so the floor cannot refuse the
    answer either. Both facts have to reach the caller.
    """
    import numpy as np

    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    p = _make_active_particle("Ferrets are obligate carnivores.", 0.95)
    emb = np.ones(4, dtype=np.float32)
    await insert_particle(session, p, (emb / np.linalg.norm(emb)).tolist())  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    req = QueryRequest(question="What is the capital of Peru?", top_k=5)
    result = await query(session, req)  # type: ignore[arg-type]

    # The machine-readable disclosure, for UI banners and agent consumers.
    assert result.ranking_degraded is not None
    assert "no embedding model" in result.ranking_degraded.lower()
    # It must name the second half too: the guard that would have caught this
    # is the guard the same condition switched off.
    assert "relevance floor" in result.ranking_degraded.lower()
    # And the answer string itself, for plain-text consumers who see only that.
    assert result.answer.startswith("[")
    assert "not necessarily" in result.answer.lower()
    # `relevance is None` alone stays ambiguous — which is exactly why the
    # dedicated field has to exist rather than being inferred from it.
    assert result.relevance is None


@pytest.mark.asyncio
async def test_query_with_an_encoder_reports_no_degradation(db_session: object) -> None:
    """The disclosure must not fire on the normal path, or it means nothing."""
    import numpy as np

    from particles import embeddings as ep
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    p = _make_active_particle("Water is composed of hydrogen and oxygen.", 0.95)
    emb = np.ones(4, dtype=np.float32) / 2.0
    await insert_particle(session, p, emb.tolist())  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32) / 2.0])
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        result = await query(  # type: ignore[arg-type]
            session, QueryRequest(question="What is water made of?", top_k=5)
        )
    finally:
        ep.set_embedding_model(original_model)

    assert result.ranking_degraded is None
    assert not result.answer.startswith("[Semantic ranking unavailable")
