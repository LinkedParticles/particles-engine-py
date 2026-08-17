"""Tests for operations/query/source_trust.py — query-time source_trust_rank."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    PolicyProvenance,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    SourceRef,
    SourceRefType,
    SourceTrustStatement,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.query.source_trust import (
    EMPTY_TRUST_POLICY,
    TrustPolicy,
    load_trust_policy,
)

# infer_domain("REDDIT_POST") → "social media" (MUST applicability clause);
# WEB_PAGE has no MUST clause → domain None → statement layers skipped.
REDDIT_DOMAIN = "social media"


def _policy(
    statements: dict[tuple[str, str, str], float] | None = None,
    domain_scores: dict[str, float] | None = None,
    url_patterns: tuple[tuple[re.Pattern[str], float], ...] = (),
) -> TrustPolicy:
    return TrustPolicy(
        statements=statements or {},
        domain_scores=domain_scores or {},
        url_patterns=url_patterns,
    )


class TestTrustPolicyEvaluate:
    """The resolve-or-None layered walk (§2), pure / in-memory."""

    def test_no_policy_is_none(self) -> None:
        assert EMPTY_TRUST_POLICY.evaluate("e1", "WEB_PAGE", "https://x.example/p") is None

    def test_file_uri_is_none_even_with_wildcard(self) -> None:
        policy = _policy(domain_scores={"*": 0.3})
        assert policy.evaluate("e1", "PDF", "file:///tmp/doc.pdf") is None

    def test_missing_uri_is_none(self) -> None:
        policy = _policy(domain_scores={"*": 0.3})
        assert policy.evaluate("e1", "WEB_PAGE", None) is None

    def test_corpus_entry_statement_wins_over_all(self) -> None:
        policy = _policy(
            statements={
                (REDDIT_DOMAIN, "CORPUS_ENTRY", "e1"): 0.1,
                (REDDIT_DOMAIN, "SOURCE_TYPE", "REDDIT_POST"): 0.5,
            },
            domain_scores={"reddit.com": 0.9},
        )
        rank = policy.evaluate("e1", "REDDIT_POST", "https://reddit.com/r/x")
        assert rank == pytest.approx(0.1)

    def test_source_type_statement_beats_url_rules(self) -> None:
        policy = _policy(
            statements={(REDDIT_DOMAIN, "SOURCE_TYPE", "REDDIT_POST"): 0.5},
            domain_scores={"reddit.com": 0.9},
        )
        rank = policy.evaluate("other-entry", "REDDIT_POST", "https://reddit.com/r/x")
        assert rank == pytest.approx(0.5)

    def test_none_domain_skips_statement_layers(self) -> None:
        # WEB_PAGE has no MUST clause → statements never match, URL layer applies.
        policy = _policy(
            statements={(REDDIT_DOMAIN, "SOURCE_TYPE", "WEB_PAGE"): 0.5},
            domain_scores={"x.example": 0.4},
        )
        assert policy.evaluate("e1", "WEB_PAGE", "https://x.example/p") == pytest.approx(0.4)

    def test_exact_domain_row_beats_wildcard(self) -> None:
        policy = _policy(domain_scores={"x.example": 0.4, "*": 0.8})
        assert policy.evaluate("e1", "WEB_PAGE", "https://x.example/p") == pytest.approx(0.4)
        assert policy.evaluate("e1", "WEB_PAGE", "https://y.example/p") == pytest.approx(0.8)

    def test_modifier_stacks_on_domain_baseline(self) -> None:
        policy = _policy(
            domain_scores={"x.example": 0.5},
            url_patterns=((re.compile(r"/blog/"), -0.2),),
        )
        rank = policy.evaluate("e1", "WEB_PAGE", "https://x.example/blog/post")
        assert rank == pytest.approx(0.3)

    def test_modifier_without_baseline_applies_against_neutral_one(self) -> None:
        # An explicit assertion bites even with no baseline row.
        policy = _policy(url_patterns=((re.compile(r"/forum/"), -0.4),))
        rank = policy.evaluate("e1", "WEB_PAGE", "https://x.example/forum/t")
        assert rank == pytest.approx(0.6)

    def test_unmatched_modifier_alone_is_none(self) -> None:
        policy = _policy(url_patterns=((re.compile(r"/forum/"), -0.4),))
        assert policy.evaluate("e1", "WEB_PAGE", "https://x.example/blog/p") is None

    def test_rank_clamped_to_unit_interval(self) -> None:
        policy = _policy(
            domain_scores={"x.example": 0.9},
            url_patterns=((re.compile(r"\.example"), 0.5),),
        )
        assert policy.evaluate("e1", "WEB_PAGE", "https://x.example/p") == 1.0

    def test_author_statement_applies(self) -> None:
        """§6.4 tier 2: an AUTHOR-scoped statement matches on the snapshot's author_id."""
        policy = _policy(statements={(REDDIT_DOMAIN, "AUTHOR", "reddit:u/expert"): 0.9})
        rank = policy.evaluate("e1", "REDDIT_POST", None, "reddit:u/expert")
        assert rank == pytest.approx(0.9)
        # A different author falls through (here to None — nothing else configured)
        assert policy.evaluate("e1", "REDDIT_POST", None, "reddit:u/other") is None

    def test_corpus_entry_beats_author_beats_source_type(self) -> None:
        """§6.4 first-match-wins precedence across all three statement tiers."""
        statements = {
            (REDDIT_DOMAIN, "CORPUS_ENTRY", "e1"): 0.1,
            (REDDIT_DOMAIN, "AUTHOR", "reddit:u/expert"): 0.9,
            (REDDIT_DOMAIN, "SOURCE_TYPE", "REDDIT_POST"): 0.5,
        }
        policy = _policy(statements=statements)
        # Entry-scoped wins even when the author is also configured
        assert policy.evaluate("e1", "REDDIT_POST", None, "reddit:u/expert") == pytest.approx(0.1)
        # No entry statement → AUTHOR beats SOURCE_TYPE
        assert policy.evaluate("e2", "REDDIT_POST", None, "reddit:u/expert") == pytest.approx(0.9)

    def test_no_author_id_falls_through_to_source_type(self) -> None:
        policy = _policy(
            statements={
                (REDDIT_DOMAIN, "AUTHOR", "reddit:u/expert"): 0.9,
                (REDDIT_DOMAIN, "SOURCE_TYPE", "REDDIT_POST"): 0.5,
            }
        )
        assert policy.evaluate("e1", "REDDIT_POST", None, None) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_load_trust_policy_snapshot(db_session: object) -> None:
    """load_trust_policy reads statements + domain rules; latest statement wins."""
    from particles.store.trust_store import insert_trust_statement, upsert_trust_rule

    session = db_session  # type: ignore[assignment]

    def _stmt(rank: float, asserted_at: datetime) -> SourceTrustStatement:
        return SourceTrustStatement(
            domain=REDDIT_DOMAIN,
            source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="REDDIT_POST"),
            trust_rank=rank,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
            asserted_at=asserted_at,
        )

    await insert_trust_statement(session, _stmt(0.7, datetime(2026, 1, 1, tzinfo=UTC)))  # type: ignore[arg-type]
    await insert_trust_statement(session, _stmt(0.3, datetime(2026, 6, 1, tzinfo=UTC)))  # type: ignore[arg-type]
    await upsert_trust_rule(session, "domain", "sketchy.example", 0.2, None)  # type: ignore[arg-type]
    await upsert_trust_rule(session, "url_pattern", r"/ads/", None, -0.3)  # type: ignore[arg-type]
    await upsert_trust_rule(session, "url_pattern", r"([invalid", None, -0.5)  # type: ignore[arg-type]

    policy = await load_trust_policy(session)  # type: ignore[arg-type]

    # Most recently asserted statement wins for the same key
    assert policy.statements[(REDDIT_DOMAIN, "SOURCE_TYPE", "REDDIT_POST")] == pytest.approx(0.3)
    assert policy.domain_scores == {"sketchy.example": 0.2}
    # The invalid regex was skipped at compile time
    assert len(policy.url_patterns) == 1
    assert policy.evaluate("e1", "REDDIT_POST", None) == pytest.approx(0.3)
    assert policy.evaluate("e1", "WEB_PAGE", "https://sketchy.example/p") == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# End-to-end: the query path consults the policy (§5)
# ---------------------------------------------------------------------------


def _particle(
    content: str,
    entry_id: str,
    confidence: float = 0.8,
    snapshot_id: str | None = None,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
    )


async def _add_entry(
    session: object, entry_id: str, uri_r: str | None, source_type: str = "WEB_PAGE"
) -> None:
    from particles.corpus.store import CorpusEntryRow

    session.add(  # type: ignore[attr-defined]
        CorpusEntryRow(
            entry_id=entry_id,
            uri_r=uri_r,
            source_type=source_type,
            mutability="MUTABLE",
            fetch_policy="LAZY",
            created_at=datetime.now(UTC),
            deposited_by="test",
        )
    )
    await session.flush()  # type: ignore[attr-defined]


async def _add_snapshot(
    session: object, snapshot_id: str, entry_id: str, author_id: str | None
) -> None:
    from particles.corpus.store import SnapshotRow

    session.add(  # type: ignore[attr-defined]
        SnapshotRow(
            snapshot_id=snapshot_id,
            entry_id=entry_id,
            captured_at=datetime.now(UTC),
            content_hash=f"hash-{snapshot_id}",
            warc_record_type="RESPONSE",
            extraction_status="COMPLETE",
            author_id=author_id,
        )
    )
    await session.flush()  # type: ignore[attr-defined]


def _mock_embeddings() -> tuple[MagicMock, object]:
    from particles import embeddings as ep

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original = ep._embedding_model
    ep.set_embedding_model(mock_model)
    return mock_model, original


@pytest.mark.asyncio
async def test_query_neutral_without_policy(db_session: object, monkeypatch: object) -> None:
    """Neutrality regression: zero trust configuration → effective confidence is
    exactly extractor_trust × recency, byte-identical to the pre-ADR-0113 path."""
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    await _add_entry(session, "entry-n", "https://anywhere.example/p")
    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    await insert_particle(session, _particle("Neutral fact.", "entry-n", 0.8), emb)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    _, original = _mock_embeddings()
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="ok"))  # type: ignore[attr-defined]
    try:
        result = await qmain.query(session, QueryRequest(question="fact?", top_k=5))  # type: ignore[arg-type]
        assert len(result.particles) == 1
        # No extractor row → trust weight default 1.0; no pub_at → decay 1.0;
        # no trust policy → source factor 1.0. Effective == raw value.
        assert result.effective_confidences[0] == pytest.approx(0.8)
    finally:
        ep.set_embedding_model(original)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_query_domain_rule_demotes_ranking(db_session: object, monkeypatch: object) -> None:
    """An asserted domain rule lowers effective confidence and flips
    ranking; the unconfigured source stays neutral."""
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.store.particle_store import insert_particle
    from particles.store.trust_store import upsert_trust_rule

    session = db_session  # type: ignore[assignment]
    await _add_entry(session, "entry-good", "https://solid.example/article")
    await _add_entry(session, "entry-bad", "https://sketchy.example/post")
    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    # The sketchy source asserts with HIGHER raw confidence …
    await insert_particle(session, _particle("Sketchy claim.", "entry-bad", 0.9), emb)  # type: ignore[arg-type]
    await insert_particle(session, _particle("Solid claim.", "entry-good", 0.8), emb)  # type: ignore[arg-type]
    await upsert_trust_rule(session, "domain", "sketchy.example", 0.2, None)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    _, original = _mock_embeddings()
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="ok"))  # type: ignore[attr-defined]
    try:
        result = await qmain.query(session, QueryRequest(question="claims?", top_k=5))  # type: ignore[arg-type]
        by_content = dict(
            zip([p.content for p in result.particles], result.effective_confidences, strict=True)
        )
        # … but the trust rule demotes it below the neutral source.
        assert by_content["Sketchy claim."] == pytest.approx(0.9 * 0.2)
        assert by_content["Solid claim."] == pytest.approx(0.8)
        assert result.particles[0].content == "Solid claim."
    finally:
        ep.set_embedding_model(original)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_query_federated_viewer_lens(tmp_path: object, monkeypatch: object) -> None:
    """The acceptance test: the viewer's trust policy is applied to
    every store's candidates; the origin store's own policy is ignored."""
    import particles._orm_modules  # noqa: F401
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.db import DEFAULT_STORE, Base, get_engine, reset_engine, session_scope
    from particles.store.particle_store import insert_particle
    from particles.store.trust_store import upsert_trust_rule

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/viewer.db"
    cfg.storage.stores = {"other": f"sqlite+aiosqlite:///{tmp_path}/other.db"}

    for handle in (DEFAULT_STORE, "other"):
        engine = get_engine(handle)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    async with session_scope(DEFAULT_STORE) as s:
        await _add_entry(s, "v-entry", "https://viewerfact.example/p")
        await insert_particle(s, _particle("Viewer store fact.", "v-entry", 0.8), emb)
        # The VIEWER demotes the other store's source …
        await upsert_trust_rule(s, "domain", "otherfact.example", 0.2, None)
        await s.commit()
    async with session_scope("other") as s:
        await _add_entry(s, "o-entry", "https://otherfact.example/p")
        await insert_particle(s, _particle("Other store fact.", "o-entry", 0.9), emb)
        # … while the ORIGIN store's own policy (demoting the viewer's
        # source) must NOT apply under the viewer's lens.
        await upsert_trust_rule(s, "domain", "viewerfact.example", 0.1, None)
        await s.commit()

    _, original = _mock_embeddings()
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="ok"))  # type: ignore[attr-defined]
    try:
        result = await qmain.query_federated(
            [DEFAULT_STORE, "other"], QueryRequest(question="facts?", top_k=10)
        )
        by_content = dict(
            zip([p.content for p in result.particles], result.effective_confidences, strict=True)
        )
        # Viewer's rule demotes the other store's higher-raw-confidence claim;
        # the other store's rule against the viewer's source has no effect.
        assert by_content["Other store fact."] == pytest.approx(0.9 * 0.2)
        assert by_content["Viewer store fact."] == pytest.approx(0.8)
        assert result.particles[0].content == "Viewer store fact."
    finally:
        ep.set_embedding_model(original)  # type: ignore[arg-type]
        for handle in (DEFAULT_STORE, "other"):
            await get_engine(handle).dispose()
        reset_engine()


# ---------------------------------------------------------------------------
# load_source_trust_ranks — the exporter-surface composition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_author_statement_demotes_ranking(
    db_session: object, monkeypatch: object
) -> None:
    """§6.4 tier 2 at query time: an AUTHOR-scoped statement changes effective
    confidence for particles whose SOURCE snapshot carries that author_id; a
    particle by another author on the same source type is untouched."""
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.store.particle_store import insert_particle
    from particles.store.trust_store import insert_trust_statement

    session = db_session  # type: ignore[assignment]
    await _add_entry(session, "entry-r1", "https://reddit.com/r/x/1", source_type="REDDIT_POST")
    await _add_entry(session, "entry-r2", "https://reddit.com/r/x/2", source_type="REDDIT_POST")
    await _add_snapshot(session, "snap-r1", "entry-r1", "reddit:u/troll")
    await _add_snapshot(session, "snap-r2", "entry-r2", "reddit:u/sage")
    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    await insert_particle(  # type: ignore[arg-type]
        session, _particle("Troll claim.", "entry-r1", 0.9, snapshot_id="snap-r1"), emb
    )
    await insert_particle(  # type: ignore[arg-type]
        session, _particle("Sage claim.", "entry-r2", 0.8, snapshot_id="snap-r2"), emb
    )
    await insert_trust_statement(
        session,  # type: ignore[arg-type]
        SourceTrustStatement(
            domain=REDDIT_DOMAIN,
            source_ref=SourceRef(type=SourceRefType.AUTHOR, value="reddit:u/troll"),
            trust_rank=0.2,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
        ),
    )
    await session.commit()  # type: ignore[union-attr]

    _, original = _mock_embeddings()
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="ok"))  # type: ignore[attr-defined]
    try:
        result = await qmain.query(session, QueryRequest(question="claims?", top_k=5))  # type: ignore[arg-type]
        by_content = dict(
            zip([p.content for p in result.particles], result.effective_confidences, strict=True)
        )
        assert by_content["Troll claim."] == pytest.approx(0.9 * 0.2)
        assert by_content["Sage claim."] == pytest.approx(0.8)
        assert result.particles[0].content == "Sage claim."
    finally:
        ep.set_embedding_model(original)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_load_source_trust_ranks(db_session: object) -> None:
    """Asserted ranks appear in the map; unconfigured particles are absent."""
    from particles.operations.query.source_trust import load_source_trust_ranks
    from particles.store.trust_store import upsert_trust_rule

    session = db_session  # type: ignore[assignment]
    await _add_entry(session, "entry-good", "https://solid.example/article")
    await _add_entry(session, "entry-bad", "https://sketchy.example/post")
    p_good = _particle("Solid claim.", "entry-good")
    p_bad = _particle("Sketchy claim.", "entry-bad")
    await upsert_trust_rule(session, "domain", "sketchy.example", 0.2, None)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    ranks = await load_source_trust_ranks(session, [p_good, p_bad])  # type: ignore[arg-type]
    assert ranks == {p_bad.id: pytest.approx(0.2)}


@pytest.mark.asyncio
async def test_load_source_trust_ranks_empty_policy_short_circuits(db_session: object) -> None:
    """No trust configuration → {} with no corpus-entry round trips."""
    from particles.operations.query.source_trust import load_source_trust_ranks

    session = db_session  # type: ignore[assignment]
    ranks = await load_source_trust_ranks(session, [_particle("Any claim.", "entry-x")])  # type: ignore[arg-type]
    assert ranks == {}


@pytest.mark.asyncio
async def test_load_source_trust_ranks_author_tier_batched(db_session: object) -> None:
    """Batched evaluation over mixed authors: each particle gets the rank its
    own §6.4 cascade walk produces — AUTHOR where the snapshot's author_id
    matches, SOURCE_TYPE fall-through where it doesn't or there is no author.
    This is the shared helper the anki/wiki exporters consume, so
    these numbers are exactly what query reports for the same particles."""
    from particles.operations.query.source_trust import load_source_trust_ranks
    from particles.store.trust_store import insert_trust_statement

    session = db_session  # type: ignore[assignment]
    await _add_entry(session, "entry-a", "https://reddit.com/r/x/a", source_type="REDDIT_POST")
    await _add_entry(session, "entry-b", "https://reddit.com/r/x/b", source_type="REDDIT_POST")
    await _add_entry(session, "entry-c", "https://reddit.com/r/x/c", source_type="REDDIT_POST")
    await _add_snapshot(session, "snap-a", "entry-a", "reddit:u/troll")
    await _add_snapshot(session, "snap-b", "entry-b", "reddit:u/sage")
    await _add_snapshot(session, "snap-c", "entry-c", None)  # no author recorded

    p_troll = _particle("Troll claim.", "entry-a", snapshot_id="snap-a")
    p_sage = _particle("Sage claim.", "entry-b", snapshot_id="snap-b")
    p_anon = _particle("Anonymous claim.", "entry-c", snapshot_id="snap-c")

    for ref_type, value, rank in [
        (SourceRefType.AUTHOR, "reddit:u/troll", 0.2),
        (SourceRefType.AUTHOR, "reddit:u/sage", 0.95),
        (SourceRefType.SOURCE_TYPE, "REDDIT_POST", 0.5),
    ]:
        await insert_trust_statement(
            session,  # type: ignore[arg-type]
            SourceTrustStatement(
                domain=REDDIT_DOMAIN,
                source_ref=SourceRef(type=ref_type, value=value),
                trust_rank=rank,
                policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
                asserted_by="operator",
            ),
        )
    await session.commit()  # type: ignore[union-attr]

    ranks = await load_source_trust_ranks(session, [p_troll, p_sage, p_anon])  # type: ignore[arg-type]
    assert ranks == {
        p_troll.id: pytest.approx(0.2),
        p_sage.id: pytest.approx(0.95),
        # No author_id on the snapshot → tier 2 skipped, SOURCE_TYPE applies
        p_anon.id: pytest.approx(0.5),
    }


def test_wiki_effective_confidences_apply_source_ranks() -> None:
    """The wiki exporter's pure helper multiplies the precomputed rank in;
    absence stays neutral."""
    from particles.exporters.wiki import _effective_confidences

    p_good = _particle("Solid claim.", "entry-good", 0.8)
    p_bad = _particle("Sketchy claim.", "entry-bad", 0.9)
    eff = _effective_confidences([p_good, p_bad], {}, {p_bad.id: 0.2})
    assert eff[p_good.id] == pytest.approx(0.8)
    assert eff[p_bad.id] == pytest.approx(0.9 * 0.2)


@pytest.mark.asyncio
async def test_min_confidence_filters_on_effective(db_session: object, monkeypatch: object) -> None:
    """§9.3 step 5: a particle whose raw confidence passes the floor
    but whose trust-demoted effective confidence falls below it is excluded."""
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.store.particle_store import insert_particle
    from particles.store.trust_store import upsert_trust_rule

    session = db_session  # type: ignore[assignment]
    await _add_entry(session, "entry-good", "https://solid.example/article")
    await _add_entry(session, "entry-bad", "https://sketchy.example/post")
    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    # Both pass the raw-value prefilter (0.9 and 0.8 ≥ 0.5) …
    await insert_particle(session, _particle("Sketchy claim.", "entry-bad", 0.9), emb)  # type: ignore[arg-type]
    await insert_particle(session, _particle("Solid claim.", "entry-good", 0.8), emb)  # type: ignore[arg-type]
    # … but the demoted one lands at 0.9 × 0.2 = 0.18 effective.
    await upsert_trust_rule(session, "domain", "sketchy.example", 0.2, None)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    _, original = _mock_embeddings()
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="ok"))  # type: ignore[attr-defined]
    try:
        result = await qmain.query(
            session,  # type: ignore[arg-type]
            QueryRequest(question="claims?", top_k=5, min_confidence=0.5),
        )
        assert [p.content for p in result.particles] == ["Solid claim."]
    finally:
        ep.set_embedding_model(original)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Per-member policy snapshots for contestedness (standalone, no overlay)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_local_trust_policy_excludes_lens(db_session: object) -> None:
    """The local member is the store's own policy *alone* — no lens overlay (§2)."""
    from particles.core.schema import TrustLensDefinition, TrustLensUrlRule
    from particles.operations.query.source_trust import (
        load_local_trust_policy,
        load_trust_policy,
    )
    from particles.store.lens_store import adopt_lens, materialise_lens

    session = db_session  # type: ignore[assignment]
    lens = TrustLensDefinition(
        name="acme",
        version=1,
        url_rules=[TrustLensUrlRule(scope="domain", pattern="sketchy.example", score=0.2)],
    )
    await materialise_lens(session, lens)  # type: ignore[arg-type]
    await adopt_lens(session, "acme")  # type: ignore[arg-type]

    # The composed (ranking) policy sees the lens; the local member does not.
    composed = await load_trust_policy(session)  # type: ignore[arg-type]
    local = await load_local_trust_policy(session)  # type: ignore[arg-type]
    assert composed.evaluate("e1", "WEB_PAGE", "https://sketchy.example/p") == pytest.approx(0.2)
    assert local.evaluate("e1", "WEB_PAGE", "https://sketchy.example/p") is None


def test_lens_to_trust_policy_is_standalone() -> None:
    """A lens converts to a policy that applies its own layers and nothing else (§2)."""
    from particles.core.schema import (
        TrustLensDefinition,
        TrustLensStatement,
        TrustLensUrlRule,
    )
    from particles.operations.query.source_trust import lens_to_trust_policy

    lens = TrustLensDefinition(
        name="acme",
        version=1,
        statements=[
            TrustLensStatement(domain="numismatics", source_type="NUMISTA_API_COIN", trust_rank=0.4)
        ],
        url_rules=[TrustLensUrlRule(scope="domain", pattern="sketchy.example", score=0.2)],
    )
    policy = lens_to_trust_policy(lens)
    assert policy.statements[("numismatics", "SOURCE_TYPE", "NUMISTA_API_COIN")] == pytest.approx(
        0.4
    )
    assert policy.evaluate("e1", "WEB_PAGE", "https://sketchy.example/p") == pytest.approx(0.2)
    # A source the lens does not name stays neutral — no synthetic baseline.
    assert policy.evaluate("e1", "WEB_PAGE", "https://unrelated.example/p") is None
