"""Tests for Extension B: source trust cascade."""

from __future__ import annotations

import pytest

from particles.core.schema import (
    PolicyProvenance,
    SourceRef,
    SourceRefType,
    SourceTrustStatement,
)
from particles.extraction.registry import infer_domain

# ---------------------------------------------------------------------------
# infer_domain
# ---------------------------------------------------------------------------


class TestInferDomain:
    def test_numista_infers_numismatics(self) -> None:
        domain = infer_domain("NUMISTA_API_COIN")
        assert domain == "numismatics"

    def test_nomisma_infers_numismatics(self) -> None:
        domain = infer_domain("NOMISMA_API")
        assert domain == "numismatics"

    def test_wikidata_infers_wikidata(self) -> None:
        domain = infer_domain("WIKIDATA_API")
        assert domain is not None  # Wikidata extractor has a MUST clause

    def test_unknown_source_type_returns_none(self) -> None:
        domain = infer_domain("WEB_PAGE")
        assert domain is None

    def test_pdf_returns_none(self) -> None:
        domain = infer_domain("PDF")
        assert domain is None


# ---------------------------------------------------------------------------
# SourceTrustStatement construction
# ---------------------------------------------------------------------------


class TestSourceTrustStatement:
    def test_operator_direct_statement(self) -> None:
        stmt = SourceTrustStatement(
            domain="numismatics",
            source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="NUMISTA_API_COIN"),
            trust_rank=0.9,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
        )
        assert stmt.domain == "numismatics"
        assert stmt.source_ref.type == SourceRefType.SOURCE_TYPE
        assert stmt.policy_provenance == PolicyProvenance.OPERATOR_DIRECT

    def test_reviewer_derived_statement(self) -> None:
        stmt = SourceTrustStatement(
            domain="numismatics",
            source_ref=SourceRef(type=SourceRefType.CORPUS_ENTRY, value="abc123"),
            trust_rank=0.8,
            policy_provenance=PolicyProvenance.REVIEWER_DERIVED,
            asserted_by="user@example.com",
        )
        assert stmt.policy_provenance == PolicyProvenance.REVIEWER_DERIVED


# ---------------------------------------------------------------------------
# cascade gate (DB integration)
# ---------------------------------------------------------------------------


class TestCascadeGate:
    @pytest.mark.asyncio
    async def test_operator_direct_always_passes(self, db_session: object) -> None:
        from particles.operations.cascade import _gate_passes
        from particles.store.trust_store import insert_trust_statement

        stmt = SourceTrustStatement(
            domain="numismatics",
            source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="NUMISTA_API_COIN"),
            trust_rank=0.9,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
        )
        await insert_trust_statement(db_session, stmt)  # type: ignore[arg-type]
        result = await _gate_passes(db_session, stmt)  # type: ignore[arg-type]
        assert result is True

    @pytest.mark.asyncio
    async def test_reviewer_derived_blocked_below_threshold(self, db_session: object) -> None:
        from particles.operations.cascade import _gate_passes
        from particles.store.trust_store import insert_trust_statement

        # Only 1 REVIEWER_DERIVED statement — should be blocked (need 3)
        stmt = SourceTrustStatement(
            domain="numismatics",
            source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="TEST_SOURCE"),
            trust_rank=0.8,
            policy_provenance=PolicyProvenance.REVIEWER_DERIVED,
            asserted_by="reviewer1",
        )
        await insert_trust_statement(db_session, stmt)  # type: ignore[arg-type]
        result = await _gate_passes(db_session, stmt)  # type: ignore[arg-type]
        assert result is False

    @pytest.mark.asyncio
    async def test_reviewer_derived_passes_at_threshold(self, db_session: object) -> None:
        from particles.operations.cascade import _gate_passes
        from particles.store.trust_store import insert_trust_statement

        domain = "finance"
        ref_type = SourceRefType.SOURCE_TYPE
        ref_value = "ANALYST_REPORT"
        stmts = []
        for i in range(3):
            s = SourceTrustStatement(
                domain=domain,
                source_ref=SourceRef(type=ref_type, value=ref_value),
                trust_rank=0.8,
                policy_provenance=PolicyProvenance.REVIEWER_DERIVED,
                asserted_by=f"reviewer{i}",
            )
            await insert_trust_statement(db_session, s)  # type: ignore[arg-type]
            stmts.append(s)

        result = await _gate_passes(db_session, stmts[-1])  # type: ignore[arg-type]
        assert result is True


# ---------------------------------------------------------------------------
# layered trust rank (DB integration)
# ---------------------------------------------------------------------------


class TestLayeredTrustRank:
    @pytest.mark.asyncio
    async def test_corpus_entry_scope_takes_precedence(self, db_session: object) -> None:
        from particles.store.trust_store import get_layered_trust_rank, insert_trust_statement

        entry_id = "test-entry-abc"
        domain = "numismatics"

        # SOURCE_TYPE-scoped at 0.6
        await insert_trust_statement(
            db_session,  # type: ignore[arg-type]
            SourceTrustStatement(
                domain=domain,
                source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="NUMISTA_API_COIN"),
                trust_rank=0.6,
                policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
                asserted_by="operator",
            ),
        )
        # CORPUS_ENTRY-scoped at 0.95 — should win
        await insert_trust_statement(
            db_session,  # type: ignore[arg-type]
            SourceTrustStatement(
                domain=domain,
                source_ref=SourceRef(type=SourceRefType.CORPUS_ENTRY, value=entry_id),
                trust_rank=0.95,
                policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
                asserted_by="operator",
            ),
        )

        rank = await get_layered_trust_rank(
            db_session,  # type: ignore[arg-type]
            domain,
            entry_id,
            "NUMISTA_API_COIN",
        )
        assert rank == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_source_type_scope_used_when_no_corpus_entry(self, db_session: object) -> None:
        from particles.store.trust_store import get_layered_trust_rank, insert_trust_statement

        domain = "numismatics"
        await insert_trust_statement(
            db_session,  # type: ignore[arg-type]
            SourceTrustStatement(
                domain=domain,
                source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="NUMISTA_API_COIN"),
                trust_rank=0.88,
                policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
                asserted_by="operator",
            ),
        )

        rank = await get_layered_trust_rank(
            db_session,  # type: ignore[arg-type]
            domain,
            "nonexistent-entry-id",
            "NUMISTA_API_COIN",
        )
        assert rank == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_fallback_to_adr_0045_when_no_statement(self, db_session: object) -> None:
        from particles.store.trust_store import get_layered_trust_rank

        # No statements written — should fall back to the domain score (0.50 default)
        rank = await get_layered_trust_rank(
            db_session,  # type: ignore[arg-type]
            "numismatics",
            "some-entry",
            "UNKNOWN_SOURCE_TYPE",
            uri_r=None,
        )
        assert rank == pytest.approx(0.50)

    @pytest.mark.asyncio
    async def test_author_scope_beats_source_type(self, db_session: object) -> None:
        """§6.4 tier 2: an AUTHOR-scoped statement wins over SOURCE_TYPE."""
        from particles.store.trust_store import get_layered_trust_rank, insert_trust_statement

        domain = "social media"
        await insert_trust_statement(
            db_session,  # type: ignore[arg-type]
            SourceTrustStatement(
                domain=domain,
                source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="REDDIT_POST"),
                trust_rank=0.4,
                policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
                asserted_by="operator",
            ),
        )
        await insert_trust_statement(
            db_session,  # type: ignore[arg-type]
            SourceTrustStatement(
                domain=domain,
                source_ref=SourceRef(type=SourceRefType.AUTHOR, value="reddit:u/expert"),
                trust_rank=0.9,
                policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
                asserted_by="operator",
            ),
        )

        rank = await get_layered_trust_rank(
            db_session,  # type: ignore[arg-type]
            domain,
            "no-entry-statement",
            "REDDIT_POST",
            author_id="reddit:u/expert",
        )
        assert rank == pytest.approx(0.9)

        # No author_id (non-UGC source) → tier 2 skipped, SOURCE_TYPE applies
        rank = await get_layered_trust_rank(
            db_session,  # type: ignore[arg-type]
            domain,
            "no-entry-statement",
            "REDDIT_POST",
            author_id=None,
        )
        assert rank == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_corpus_entry_scope_beats_author(self, db_session: object) -> None:
        """§6.4 first-match-wins: CORPUS_ENTRY beats AUTHOR, no aggregation."""
        from particles.store.trust_store import get_layered_trust_rank, insert_trust_statement

        domain = "social media"
        entry_id = "entry-with-statement"
        await insert_trust_statement(
            db_session,  # type: ignore[arg-type]
            SourceTrustStatement(
                domain=domain,
                source_ref=SourceRef(type=SourceRefType.AUTHOR, value="reddit:u/expert"),
                trust_rank=0.9,
                policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
                asserted_by="operator",
            ),
        )
        await insert_trust_statement(
            db_session,  # type: ignore[arg-type]
            SourceTrustStatement(
                domain=domain,
                source_ref=SourceRef(type=SourceRefType.CORPUS_ENTRY, value=entry_id),
                trust_rank=0.2,
                policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
                asserted_by="operator",
            ),
        )

        rank = await get_layered_trust_rank(
            db_session,  # type: ignore[arg-type]
            domain,
            entry_id,
            "REDDIT_POST",
            author_id="reddit:u/expert",
        )
        assert rank == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# run_trust_cascade (DB integration)
# ---------------------------------------------------------------------------


async def _seed_quarantined_conflict(db_session: object) -> tuple[object, object, object]:
    """Seed a quarantine-shaped conflict: ACTIVE A, quarantined B, open wrapper.

    Mirrors the entry/snapshot scaffolding of
    ``test_cascade_resolves_matching_inconsistency`` but births particle B
    PROVENANCE_STALE / CONFLICT_PENDING, the way the extract pipeline persists
    the INCONSISTENT-verdict loser. Returns ``(particle_a, particle_b, inc)``.
    """
    import uuid
    from datetime import UTC, datetime

    from particles.core.schema import (
        SCHEMA_VERSION,
        Confidence,
        ExtractionStatus,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status, StatusReason
    from particles.corpus.store import CorpusEntry, CorpusEntryRow, Snapshot, SnapshotRow
    from particles.store.particle_store import insert_particle

    entry_high = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="NUMISTA_API_COIN",
        uri_r="https://numista.com/coin/1",
        deposited_by="test",
    )
    entry_low = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="WEB_PAGE",
        uri_r="https://example.com/coin",
        deposited_by="test",
    )
    for e in [entry_high, entry_low]:
        db_session.add(CorpusEntryRow.from_model(e))  # type: ignore[attr-defined]
    snap_high = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        entry_id=entry_high.entry_id,
        captured_at=datetime.now(UTC),
        content_hash="aaa",
        extraction_status=ExtractionStatus.COMPLETE,
    )
    snap_low = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        entry_id=entry_low.entry_id,
        captured_at=datetime.now(UTC),
        content_hash="bbb",
        extraction_status=ExtractionStatus.COMPLETE,
    )
    db_session.add(SnapshotRow.from_model(snap_high, entry_high.entry_id))  # type: ignore[attr-defined]
    db_session.add(SnapshotRow.from_model(snap_low, entry_low.entry_id))  # type: ignore[attr-defined]
    await db_session.flush()  # type: ignore[attr-defined]

    conf = Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT)
    particle_a = Particle(
        content="Weight: 3.44 g (Numista)",
        confidence=conf,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_high.entry_id,
                snapshot_id=snap_high.snapshot_id,
            )
        ],
        asserted_by="numista-coin-extractor",
        status=Status.ACTIVE,
        schema_version=SCHEMA_VERSION,
    )
    particle_b = Particle(
        content="Weight: 3.5 g (web)",
        confidence=conf,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_low.entry_id,
                snapshot_id=snap_low.snapshot_id,
            )
        ],
        asserted_by="general-extractor",
        status=Status.PROVENANCE_STALE,
        status_reason=StatusReason.CONFLICT_PENDING,
        schema_version=SCHEMA_VERSION,
    )
    await insert_particle(db_session, particle_a)  # type: ignore[arg-type]
    await insert_particle(db_session, particle_b)  # type: ignore[arg-type]

    inc = Particle(
        content=f"INCONSISTENCY: {particle_a.id} vs {particle_b.id}",
        confidence=conf,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.PARTICLE,
                corpus_entry_id=particle_a.id,
                snapshot_id=particle_a.id,
            ),
            ProvenanceRef(
                type=ProvenanceRefType.PARTICLE,
                corpus_entry_id=particle_b.id,
                snapshot_id=particle_b.id,
            ),
        ],
        asserted_by="extract-pipeline",
        status=Status.INCONSISTENCY,
        schema_version=SCHEMA_VERSION,
    )
    await insert_particle(db_session, inc, domain_hint="numismatics")  # type: ignore[arg-type]
    await db_session.flush()  # type: ignore[attr-defined]
    return particle_a, particle_b, inc


class TestRunTrustCascade:
    @pytest.mark.asyncio
    async def test_cascade_resolves_matching_inconsistency(self, db_session: object) -> None:
        """An OPERATOR_DIRECT statement should auto-resolve a matching INCONSISTENCY."""
        import uuid
        from datetime import UTC, datetime

        from particles.core.schema import (
            SCHEMA_VERSION,
            Confidence,
            ExtractionStatus,
            Particle,
            ProvenanceRef,
            ProvenanceRefType,
            UncertaintyNature,
        )
        from particles.core.scoring.confidence import CalibrationSource
        from particles.core.status import Status
        from particles.corpus.store import CorpusEntry, CorpusEntryRow, Snapshot, SnapshotRow
        from particles.operations.cascade import run_trust_cascade
        from particles.store.particle_store import (
            get_particle,
            insert_particle,
        )
        from particles.store.trust_store import insert_trust_statement

        # Create two corpus entries with different source types
        entry_high = CorpusEntry(
            entry_id=str(uuid.uuid4()),
            source_type="NUMISTA_API_COIN",
            uri_r="https://numista.com/coin/1",
            deposited_by="test",
        )
        entry_low = CorpusEntry(
            entry_id=str(uuid.uuid4()),
            source_type="WEB_PAGE",
            uri_r="https://example.com/coin",
            deposited_by="test",
        )
        for e in [entry_high, entry_low]:
            db_session.add(CorpusEntryRow.from_model(e))  # type: ignore[attr-defined]

        snap_high = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            entry_id=entry_high.entry_id,
            captured_at=datetime.now(UTC),
            content_hash="aaa",
            extraction_status=ExtractionStatus.COMPLETE,
        )
        snap_low = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            entry_id=entry_low.entry_id,
            captured_at=datetime.now(UTC),
            content_hash="bbb",
            extraction_status=ExtractionStatus.COMPLETE,
        )
        db_session.add(SnapshotRow.from_model(snap_high, entry_high.entry_id))  # type: ignore[attr-defined]
        db_session.add(SnapshotRow.from_model(snap_low, entry_low.entry_id))  # type: ignore[attr-defined]

        await db_session.flush()  # type: ignore[attr-defined]

        # Create two constituent particles
        conf = Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT)
        particle_a = Particle(
            content="Weight: 3.44 g (Numista)",
            confidence=conf,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=entry_high.entry_id,
                    snapshot_id=snap_high.snapshot_id,
                )
            ],
            asserted_by="numista-coin-extractor",
            status=Status.ACTIVE,
            schema_version=SCHEMA_VERSION,
        )
        particle_b = Particle(
            content="Weight: 3.5 g (web)",
            confidence=conf,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=entry_low.entry_id,
                    snapshot_id=snap_low.snapshot_id,
                )
            ],
            asserted_by="general-extractor",
            status=Status.ACTIVE,
            schema_version=SCHEMA_VERSION,
        )
        await insert_particle(db_session, particle_a)  # type: ignore[arg-type]
        await insert_particle(db_session, particle_b)  # type: ignore[arg-type]

        # Create INCONSISTENCY particle with domain_hint
        inc = Particle(
            content=f"INCONSISTENCY: {particle_a.id} vs {particle_b.id}",
            confidence=conf,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE,
                    corpus_entry_id=particle_a.id,
                    snapshot_id=particle_a.id,
                ),
                ProvenanceRef(
                    type=ProvenanceRefType.PARTICLE,
                    corpus_entry_id=particle_b.id,
                    snapshot_id=particle_b.id,
                ),
            ],
            asserted_by="extract-pipeline",
            status=Status.INCONSISTENCY,
            schema_version=SCHEMA_VERSION,
        )
        await insert_particle(db_session, inc, domain_hint="numismatics")  # type: ignore[arg-type]
        await db_session.flush()  # type: ignore[attr-defined]

        # Write OPERATOR_DIRECT trust statement favouring NUMISTA_API_COIN
        stmt = SourceTrustStatement(
            domain="numismatics",
            source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="NUMISTA_API_COIN"),
            trust_rank=0.90,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
        )
        await insert_trust_statement(db_session, stmt)  # type: ignore[arg-type]

        resolved = await run_trust_cascade(db_session, stmt)  # type: ignore[arg-type]
        assert resolved == 1

        # Verify loser (particle_b) was demoted
        loser = await get_particle(db_session, particle_b.id)  # type: ignore[arg-type]
        assert loser is not None
        assert loser.status == Status.PROVENANCE_STALE

        # Verify INCONSISTENCY was resolved
        inc_after = await get_particle(db_session, inc.id)  # type: ignore[arg-type]
        assert inc_after is not None
        assert inc_after.status == Status.PROVENANCE_STALE

    @pytest.mark.asyncio
    async def test_cascade_over_quarantined_constituents(self, db_session: object) -> None:
        """both constituents exist (B quarantined), so the cascade's
        missing-constituent bail no longer fires. When the statement favours
        A's source, the quarantined loser B gets a reason-only flip to
        CONFLICT_RESOLVED (no status transition) and the wrapper closes."""
        from particles.core.status import Status, StatusReason
        from particles.operations.cascade import run_trust_cascade
        from particles.store.particle_store import get_particle
        from particles.store.trust_store import insert_trust_statement

        particle_a, particle_b, inc = await _seed_quarantined_conflict(db_session)

        # Favour A's source type (NUMISTA_API_COIN).
        stmt = SourceTrustStatement(
            domain="numismatics",
            source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="NUMISTA_API_COIN"),
            trust_rank=0.90,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
        )
        await insert_trust_statement(db_session, stmt)  # type: ignore[arg-type]

        resolved = await run_trust_cascade(db_session, stmt)  # type: ignore[arg-type]
        assert resolved == 1

        loser = await get_particle(db_session, particle_b.id)  # type: ignore[arg-type]
        assert loser is not None
        assert loser.status == Status.PROVENANCE_STALE
        assert loser.status_reason == StatusReason.CONFLICT_RESOLVED
        winner = await get_particle(db_session, particle_a.id)  # type: ignore[arg-type]
        assert winner is not None and winner.status == Status.ACTIVE
        inc_after = await get_particle(db_session, inc.id)  # type: ignore[arg-type]
        assert inc_after is not None and inc_after.status == Status.PROVENANCE_STALE

    @pytest.mark.asyncio
    async def test_cascade_promotes_quarantined_winner(self, db_session: object) -> None:
        """a cascade resolving in favour of the quarantined candidate
        mints a new ACTIVE particle (Reindex pattern), exactly as PREFER_B
        review would; the quarantined row is SUPERSEDED, never reactivated."""
        from particles.core.status import Status, StatusReason
        from particles.operations.cascade import run_trust_cascade
        from particles.store.particle_store import get_active_particles, get_particle
        from particles.store.trust_store import insert_trust_statement

        particle_a, particle_b, inc = await _seed_quarantined_conflict(db_session)

        # Favour B's source entry directly (CORPUS_ENTRY scope beats A's
        # source-type rank), so the quarantined B wins.
        b_entry_id = particle_b.provenance[0].corpus_entry_id
        stmt = SourceTrustStatement(
            domain="numismatics",
            source_ref=SourceRef(type=SourceRefType.CORPUS_ENTRY, value=b_entry_id),
            trust_rank=0.95,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
        )
        await insert_trust_statement(db_session, stmt)  # type: ignore[arg-type]

        resolved = await run_trust_cascade(db_session, stmt)  # type: ignore[arg-type]
        assert resolved == 1

        # Loser A demoted by transition; quarantined B superseded by the mint.
        loser = await get_particle(db_session, particle_a.id)  # type: ignore[arg-type]
        assert loser is not None and loser.status == Status.PROVENANCE_STALE
        assert loser.status_reason == StatusReason.CONFLICT_RESOLVED
        old_b = await get_particle(db_session, particle_b.id)  # type: ignore[arg-type]
        assert old_b is not None and old_b.status == Status.SUPERSEDED

        minted = [
            p
            for p in await get_active_particles(db_session)  # type: ignore[arg-type]
            if p.supersedes == particle_b.id
        ]
        assert len(minted) == 1
        assert minted[0].content == particle_b.content
        inc_after = await get_particle(db_session, inc.id)  # type: ignore[arg-type]
        assert inc_after is not None and inc_after.status == Status.PROVENANCE_STALE

    @pytest.mark.asyncio
    async def test_cascade_no_match_different_domain(self, db_session: object) -> None:
        """Cascade should not affect INCONSISTENCY particles in a different domain."""

        from particles.core.schema import SCHEMA_VERSION, Confidence, Particle, UncertaintyNature
        from particles.core.scoring.confidence import CalibrationSource
        from particles.core.status import Status
        from particles.operations.cascade import run_trust_cascade
        from particles.store.particle_store import insert_particle
        from particles.store.trust_store import insert_trust_statement

        # Inconsistency in "finance" domain
        conf = Confidence(value=0.7, calibration_source=CalibrationSource.EXTRACTOR_DIRECT)
        inc = Particle(
            content="INCONSISTENCY: finance conflict",
            confidence=conf,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[],
            asserted_by="extract-pipeline",
            status=Status.INCONSISTENCY,
            schema_version=SCHEMA_VERSION,
        )
        await insert_particle(db_session, inc, domain_hint="finance")  # type: ignore[arg-type]

        # Statement in "numismatics" domain — should not affect finance inconsistency
        stmt = SourceTrustStatement(
            domain="numismatics",
            source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="NUMISTA_API_COIN"),
            trust_rank=0.9,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
        )
        await insert_trust_statement(db_session, stmt)  # type: ignore[arg-type]

        resolved = await run_trust_cascade(db_session, stmt)  # type: ignore[arg-type]
        assert resolved == 0
