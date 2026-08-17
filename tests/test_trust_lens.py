"""Tests for trust lenses: extractor, store, composition, CLI."""

from __future__ import annotations

import json

import pytest

from particles.core.schema import (
    Snapshot,
    TrustLensDefinition,
    TrustLensStatement,
    TrustLensUrlRule,
)


def _lens(
    name: str = "acme-numismatics",
    version: int = 1,
    *,
    statements: list[TrustLensStatement] | None = None,
    url_rules: list[TrustLensUrlRule] | None = None,
    extractor_weights: dict[str, float] | None = None,
) -> TrustLensDefinition:
    if statements is None:
        statements = [
            TrustLensStatement(domain="numismatics", source_type="NUMISTA_API_COIN", trust_rank=0.4)
        ]
    if url_rules is None:
        url_rules = [
            TrustLensUrlRule(scope="domain", pattern="sketchy.example", score=0.2),
            TrustLensUrlRule(scope="url_pattern", pattern="/sponsored/", modifier=-0.3),
        ]
    if extractor_weights is None:
        extractor_weights = {"general-extractor": 0.8}
    return TrustLensDefinition(
        name=name,
        version=version,
        publisher="Acme Collectors Guild",
        description="Test lens.",
        statements=statements,
        url_rules=url_rules,
        extractor_weights=extractor_weights,
    )


class TestTrustLensModel:
    def test_kind_sentinel_round_trips(self) -> None:
        lens = _lens()
        parsed = TrustLensDefinition.model_validate_json(lens.model_dump_json())
        assert parsed.kind == "TrustLensDefinition"
        assert parsed.name == lens.name
        assert parsed.statements == lens.statements
        assert parsed.url_rules == lens.url_rules

    def test_url_rule_scope_validation(self) -> None:
        with pytest.raises(ValueError):
            TrustLensUrlRule(scope="domain", pattern="x.example", modifier=-0.1)
        with pytest.raises(ValueError):
            TrustLensUrlRule(scope="url_pattern", pattern="/x/", score=0.5)

    def test_extractor_weight_bounds(self) -> None:
        with pytest.raises(ValueError):
            TrustLensDefinition(name="bad", version=1, extractor_weights={"x": 1.5})

    def test_decay_rules_round_trip(self) -> None:
        from particles.core.schema import TrustLensDecayRule

        lens = TrustLensDefinition(
            name="d",
            version=1,
            decay_rules=[
                TrustLensDecayRule(
                    scope="source_type", pattern="REDDIT_POST", half_life_days=14.0, floor=0.05
                ),
                TrustLensDecayRule(
                    scope="url_pattern",
                    pattern=r"reddit\.com/r/x",
                    half_life_days=3650.0,
                    floor=0.5,
                ),
            ],
        )
        parsed = TrustLensDefinition.model_validate_json(lens.model_dump_json())
        assert parsed.decay_rules == lens.decay_rules

    def test_decay_rule_bounds(self) -> None:
        from particles.core.schema import TrustLensDecayRule

        with pytest.raises(ValueError):  # half-life must be > 0
            TrustLensDecayRule(scope="source_type", pattern="X", half_life_days=0.0, floor=0.1)
        with pytest.raises(ValueError):  # floor must be in [0, 1]
            TrustLensDecayRule(scope="source_type", pattern="X", half_life_days=10.0, floor=1.5)


def test_deposit_sentinel_detection() -> None:
    from particles.corpus.deposit import _is_taxonomy_definition, _is_trust_lens_definition

    lens_blob = _lens().model_dump_json().encode()
    assert _is_trust_lens_definition(lens_blob, ".json")
    assert not _is_trust_lens_definition(lens_blob, ".txt")
    assert not _is_taxonomy_definition(lens_blob, ".json")
    assert not _is_trust_lens_definition(b'{"name": "x", "version": "1", "tags": []}', ".json")
    assert not _is_trust_lens_definition(b"not json", ".json")


@pytest.mark.asyncio
async def test_decay_rules_survive_materialise_round_trip(db_session: object) -> None:
    """the fourth lens layer persists + reconstructs through the store."""
    from particles.core.schema import TrustLensDecayRule
    from particles.store.lens_store import get_lens, materialise_lens

    lens = _lens(name="decay-lens").model_copy(
        update={
            "decay_rules": [
                TrustLensDecayRule(
                    scope="source_type", pattern="REDDIT_POST", half_life_days=14.0, floor=0.05
                ),
                TrustLensDecayRule(
                    scope="url_pattern",
                    pattern=r"reddit\.com/r/x",
                    half_life_days=3650.0,
                    floor=0.5,
                ),
            ]
        }
    )
    await materialise_lens(db_session, lens)  # type: ignore[arg-type]
    stored = await get_lens(db_session, "decay-lens")  # type: ignore[arg-type]
    assert stored is not None
    got = sorted((d.scope, d.pattern, d.half_life_days, d.floor) for d in stored.decay_rules)
    assert got == sorted(
        [
            ("source_type", "REDDIT_POST", 14.0, 0.05),
            ("url_pattern", r"reddit\.com/r/x", 3650.0, 0.5),
        ]
    )


@pytest.mark.asyncio
async def test_extractor_materialises_and_supersedes(db_session: object) -> None:
    """v1 materialises; v2 replaces it; a stale re-deposit is rejected with a note."""
    from particles.extraction.trust_lens import TrustLensExtractor
    from particles.store.lens_store import get_lens

    session = db_session  # type: ignore[assignment]
    extractor = TrustLensExtractor()
    snapshot = Snapshot(corpus_entry_id="entry-lens", content_hash="h1")

    result = await extractor.extract(
        snapshot,
        _lens(version=1).model_dump_json().encode(),
        session=session,
        corpus_entry_id="entry-lens",
    )
    assert result.candidates == []
    stored = await get_lens(session, "acme-numismatics")  # type: ignore[arg-type]
    assert stored is not None and stored.version == 1
    assert stored.corpus_entry_id == "entry-lens"

    # v3 supersedes v1
    await extractor.extract(
        snapshot,
        _lens(version=3).model_dump_json().encode(),
        session=session,
        corpus_entry_id="entry-lens-2",
    )
    stored = await get_lens(session, "acme-numismatics")  # type: ignore[arg-type]
    assert stored is not None and stored.version == 3
    # full entry round-trip survives the replace
    assert stored.statements == _lens().statements
    assert stored.url_rules == _lens().url_rules
    assert stored.extractor_weights == {"general-extractor": 0.8}

    # stale version is rejected with a quality note, materialisation unchanged
    result = await extractor.extract(
        snapshot,
        _lens(version=2).model_dump_json().encode(),
        session=session,
        corpus_entry_id="entry-lens-3",
    )
    assert result.candidates == []
    assert any("not materialised" in n for n in result.quality_notes)
    stored = await get_lens(session, "acme-numismatics")  # type: ignore[arg-type]
    assert stored is not None and stored.version == 3


@pytest.mark.asyncio
async def test_extractor_invalid_json_is_a_note_not_an_error(db_session: object) -> None:
    from particles.extraction.trust_lens import TrustLensExtractor

    snapshot = Snapshot(corpus_entry_id="entry-x", content_hash="h1")
    result = await TrustLensExtractor().extract(
        snapshot, b'{"kind": "TrustLensDefinition", "version": -1}', session=db_session
    )
    assert result.candidates == []
    assert any("Invalid TrustLensDefinition" in n for n in result.quality_notes)


@pytest.mark.asyncio
async def test_adoption_lifecycle(db_session: object) -> None:
    from particles.store.lens_store import adopt_lens, list_lenses, materialise_lens, unadopt_lens

    session = db_session  # type: ignore[assignment]
    with pytest.raises(ValueError, match="No materialised lens"):
        await adopt_lens(session, "nope")

    assert await materialise_lens(session, _lens()) is None
    await adopt_lens(session, "acme-numismatics")
    with pytest.raises(ValueError, match="already adopted"):
        await adopt_lens(session, "acme-numismatics")
    assert [(row.name, adopted) for row, adopted in await list_lenses(session)] == [
        ("acme-numismatics", True)
    ]

    await unadopt_lens(session, "acme-numismatics")
    with pytest.raises(ValueError, match="not adopted"):
        await unadopt_lens(session, "acme-numismatics")


# ---------------------------------------------------------------------------
# Composition into the TrustPolicy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adopted_lens_composes_into_policy(db_session: object) -> None:
    """An adopted lens's entries appear in the snapshot; silence stays neutral."""
    from particles.operations.query.source_trust import load_trust_policy
    from particles.store.lens_store import adopt_lens, materialise_lens

    session = db_session  # type: ignore[assignment]
    await materialise_lens(session, _lens())

    # Materialised but NOT adopted → policy unaffected (neutrality regression)
    policy = await load_trust_policy(session)
    assert policy.evaluate("e1", "WEB_PAGE", "https://sketchy.example/p") is None

    await adopt_lens(session, "acme-numismatics")
    policy = await load_trust_policy(session)
    assert policy.statements[("numismatics", "SOURCE_TYPE", "NUMISTA_API_COIN")] == pytest.approx(
        0.4
    )
    assert policy.evaluate("e1", "WEB_PAGE", "https://sketchy.example/p") == pytest.approx(0.2)
    assert policy.evaluate("e1", "WEB_PAGE", "https://x.example/sponsored/p") == pytest.approx(0.7)
    # A source neither local policy nor the lens names stays neutral
    assert policy.evaluate("e1", "WEB_PAGE", "https://unrelated.example/p") is None


@pytest.mark.asyncio
async def test_local_policy_wins_over_lens(db_session: object) -> None:
    from particles.operations.query.source_trust import load_trust_policy
    from particles.store.lens_store import adopt_lens, materialise_lens
    from particles.store.trust_store import upsert_trust_rule

    session = db_session  # type: ignore[assignment]
    await materialise_lens(session, _lens())  # lens says sketchy.example = 0.2
    await adopt_lens(session, "acme-numismatics")
    await upsert_trust_rule(session, "domain", "sketchy.example", 0.9, None)  # type: ignore[arg-type]

    policy = await load_trust_policy(session)
    assert policy.evaluate("e1", "WEB_PAGE", "https://sketchy.example/p") == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_author_tier_orthogonal_to_lens_composition(db_session: object) -> None:
    """The §6.4 AUTHOR tier is per-statement-scope and composes orthogonally to
    lens adoption: a local AUTHOR-scoped statement outranks a
    lens-contributed SOURCE_TYPE statement (tier precedence, not local-wins —
    the keys differ), while particles by other authors still get the lens's
    SOURCE_TYPE rank under unchanged most-skeptical composition."""
    from particles.core.schema import (
        PolicyProvenance,
        SourceRef,
        SourceRefType,
        SourceTrustStatement,
    )
    from particles.operations.query.source_trust import load_trust_policy
    from particles.store.lens_store import adopt_lens, materialise_lens
    from particles.store.trust_store import insert_trust_statement

    session = db_session  # type: ignore[assignment]
    await materialise_lens(session, _lens())  # lens: NUMISTA_API_COIN = 0.4
    await adopt_lens(session, "acme-numismatics")
    await insert_trust_statement(
        session,  # type: ignore[arg-type]
        SourceTrustStatement(
            domain="numismatics",
            source_ref=SourceRef(type=SourceRefType.AUTHOR, value="numista:curator42"),
            trust_rank=0.95,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
        ),
    )

    policy = await load_trust_policy(session)
    # The configured author wins via tier 2 over the lens's SOURCE_TYPE rank …
    assert policy.evaluate("e1", "NUMISTA_API_COIN", None, "numista:curator42") == pytest.approx(
        0.95
    )
    # … other authors and author-less candidates get the lens rank at tier 3.
    assert policy.evaluate("e1", "NUMISTA_API_COIN", None, "numista:other") == pytest.approx(0.4)
    assert policy.evaluate("e1", "NUMISTA_API_COIN", None, None) == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_min_across_lenses(db_session: object) -> None:
    """Across multiple adopted lenses the most skeptical value wins per key."""
    from particles.operations.query.source_trust import load_trust_policy
    from particles.store.lens_store import adopt_lens, materialise_lens

    session = db_session  # type: ignore[assignment]
    await materialise_lens(
        session,
        _lens(
            "lens-a",
            url_rules=[
                TrustLensUrlRule(scope="domain", pattern="x.example", score=0.6),
                TrustLensUrlRule(scope="url_pattern", pattern="/ads/", modifier=-0.1),
            ],
            extractor_weights={"general-extractor": 0.9},
        ),
    )
    await materialise_lens(
        session,
        _lens(
            "lens-b",
            statements=[
                TrustLensStatement(
                    domain="numismatics", source_type="NUMISTA_API_COIN", trust_rank=0.7
                )
            ],
            url_rules=[
                TrustLensUrlRule(scope="domain", pattern="x.example", score=0.3),
                TrustLensUrlRule(scope="url_pattern", pattern="/ads/", modifier=-0.4),
            ],
            extractor_weights={"general-extractor": 0.5},
        ),
    )
    await adopt_lens(session, "lens-a")
    await adopt_lens(session, "lens-b")

    policy = await load_trust_policy(session)
    # domain rows: min(0.6, 0.3)
    assert policy.evaluate("e1", "WEB_PAGE", "https://x.example/p") == pytest.approx(0.3)
    # statements: lens-a's default statement (0.4) vs lens-b's 0.7 → 0.4
    assert policy.statements[("numismatics", "SOURCE_TYPE", "NUMISTA_API_COIN")] == pytest.approx(
        0.4
    )
    # url_pattern modifiers: min of per-lens sums → -0.4 (against neutral 1.0)
    assert policy.evaluate("e1", "WEB_PAGE", "https://y.example/ads/p") == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_extractor_weight_overlay(db_session: object) -> None:
    """Adopted lens extractor weights min-compose into get_trust_weight_map."""
    from particles.store.extractor_store import get_trust_weight_map
    from particles.store.lens_store import adopt_lens, materialise_lens, unadopt_lens

    session = db_session  # type: ignore[assignment]
    await materialise_lens(session, _lens(extractor_weights={"general-extractor": 0.6}))

    assert "general-extractor" not in await get_trust_weight_map(session)  # not adopted yet

    await adopt_lens(session, "acme-numismatics")
    weights = await get_trust_weight_map(session)
    assert weights["general-extractor"] == pytest.approx(0.6)

    await unadopt_lens(session, "acme-numismatics")
    assert "general-extractor" not in await get_trust_weight_map(session)


def test_schema_artifact_matches_model() -> None:
    """The committed normative artifact tracks the Pydantic model."""
    from pathlib import Path

    artifact = json.loads(Path("artifacts/schemas/trust_lens.schema.json").read_text())
    live = TrustLensDefinition.model_json_schema()
    assert artifact["properties"] == live["properties"]
    assert artifact.get("$defs", {}) == live.get("$defs", {})


@pytest.mark.asyncio
async def test_federated_viewer_adopted_lens(tmp_path: object, monkeypatch: object) -> None:
    """The viewer's adopted lens demotes another store's candidates."""
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, MagicMock

    import numpy as np

    import particles._orm_modules  # noqa: F401
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        QueryRequest,
        SourceType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status
    from particles.corpus.store import CorpusEntryRow
    from particles.db import DEFAULT_STORE, Base, get_engine, reset_engine, session_scope
    from particles.store.lens_store import adopt_lens, materialise_lens
    from particles.store.particle_store import insert_particle

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/viewer.db"
    cfg.storage.stores = {"other": f"sqlite+aiosqlite:///{tmp_path}/other.db"}
    for handle in (DEFAULT_STORE, "other"):
        engine = get_engine(handle)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def _particle(content: str, entry_id: str, confidence: float) -> Particle:
        return Particle(
            content=content,
            confidence=Confidence(
                value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
            ),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test-agent",
            status=Status.ACTIVE,
            provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)],
        )

    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    async with session_scope(DEFAULT_STORE) as s:
        # The VIEWER adopts a lens demoting the other store's source domain.
        await materialise_lens(
            s,
            _lens(
                "skeptics",
                url_rules=[
                    TrustLensUrlRule(scope="domain", pattern="otherfact.example", score=0.2)
                ],
                statements=[],
                extractor_weights={},
            ),
        )
        await adopt_lens(s, "skeptics")
        await s.commit()
    async with session_scope("other") as s:
        s.add(
            CorpusEntryRow(
                entry_id="o-entry",
                uri_r="https://otherfact.example/p",
                source_type=SourceType.WEB_PAGE.value,
                mutability="MUTABLE",
                fetch_policy="LAZY",
                created_at=datetime.now(UTC),
                deposited_by="test",
            )
        )
        await insert_particle(s, _particle("Other store fact.", "o-entry", 0.9), emb)
        await s.commit()

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original = ep._embedding_model
    ep.set_embedding_model(mock_model)
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="ok"))  # type: ignore[attr-defined]
    try:
        result = await qmain.query_federated(
            [DEFAULT_STORE, "other"], QueryRequest(question="facts?", top_k=10)
        )
        by_content = dict(
            zip([p.content for p in result.particles], result.effective_confidences, strict=True)
        )
        assert by_content["Other store fact."] == pytest.approx(0.9 * 0.2)
    finally:
        ep.set_embedding_model(original)  # type: ignore[arg-type]
        for handle in (DEFAULT_STORE, "other"):
            await get_engine(handle).dispose()
        reset_engine()
