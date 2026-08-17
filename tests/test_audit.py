"""Tests for the first-run memory audit operation + renderer.

Pins the deterministic parts with injected cards / mocked finders: bucket
grouping and counts, exemplar ranking and claim-text attachment, the approved
§5 renderer phrasing (hedged labels, secondary lines, next verbs, disclosed
skips, the uncalibrated-confidence footnote), the §4 estimate math, the
extract-scoping, and the ``RECENCY_DECAY`` shared-seam card mapping. The
end-to-end fixture run with live extraction is the integration tier
(``tests/test_integration_audit.py``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    Confidence,
    ExtractionStatus,
    JudgeVerdictKind,
    LintFinding,
    LintReport,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    SuggestMode,
    SuggestReport,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.operations.audit import (
    AuditBucket,
    AuditReport,
    build_buckets,
    estimate_extraction,
    render_audit_report,
    render_estimate,
    run_memory_audit,
)
from particles.operations.curation.cards import (
    CardKind,
    CurationCard,
    DuplicateVerdict,
    ParticleBrief,
    gestures_for,
)
from particles.operations.curation.collect import collect_cards


def _card(
    kind: CardKind,
    *ids: str,
    leverage: float = 0.0,
    diagnostic: str = "diag",
    url: str | None = None,
    verdict: JudgeVerdictKind | None = None,
    bases: list[str] | None = None,
) -> CurationCard:
    return CurationCard(
        kind=kind,
        particle_ids=list(ids),
        corpus_url=url,
        diagnostic=diagnostic,
        suggested_gestures=gestures_for(kind),
        leverage=leverage,
        verdict=DuplicateVerdict(verdict=verdict) if verdict is not None else None,
        contested_bases=bases,
    )


def _brief(pid: str, content: str) -> ParticleBrief:
    return ParticleBrief(
        particle_id=pid, content=content, effective_confidence=0.5, status="ACTIVE"
    )


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


class TestEstimate:
    def test_small_source_is_one_call(self) -> None:
        est = estimate_extraction([100])
        assert est.entries == 1
        assert est.estimated_llm_calls == 1
        assert est.total_chars == 100
        assert est.estimated_tokens == 25

    def test_large_source_chunks(self) -> None:
        # html_chunk_size default 15000 → ceil(40000 / 15000) = 3 calls.
        assert estimate_extraction([40_000]).estimated_llm_calls == 3

    def test_capped_at_max_llm_calls_per_source(self) -> None:
        # max_llm_calls_per_source default 8; 200k chars would be 14 chunks.
        assert estimate_extraction([200_000]).estimated_llm_calls == 8

    def test_empty_texts_skipped(self) -> None:
        est = estimate_extraction([0, 100, 0])
        assert est.entries == 1
        assert est.estimated_llm_calls == 1

    def test_render_estimate_mentions_probes(self) -> None:
        line = render_estimate(estimate_extraction([100, 100]))
        assert "Estimate: 2 entries" in line
        assert "contradiction probes" in line


# ---------------------------------------------------------------------------
# Bucket assembly (census grouping + exemplar ranking)
# ---------------------------------------------------------------------------


class TestBuildBuckets:
    def test_counts_are_complete_and_exemplars_capped(self) -> None:
        cards = [_card(CardKind.STALE, f"p{i}", leverage=i / 10) for i in range(5)]
        buckets = build_buckets(cards, exemplars_per_class=3)
        assert len(buckets) == 1
        assert buckets[0].count == 5  # full census count, not the exemplar cap
        assert len(buckets[0].exemplars) == 3

    def test_exemplars_ranked_by_leverage(self) -> None:
        cards = [
            _card(CardKind.CONTRADICTION, "a", leverage=0.1),
            _card(CardKind.CONTRADICTION, "b", leverage=0.9),
            _card(CardKind.CONTRADICTION, "c", leverage=0.5),
        ]
        buckets = build_buckets(cards, exemplars_per_class=2)
        assert [c.particle_ids[0] for c in buckets[0].exemplars] == ["b", "c"]

    def test_headline_kinds_come_first(self) -> None:
        cards = [
            _card(CardKind.UNCITED_URL, url="https://x.example"),
            _card(CardKind.CONTRADICTION, "a"),
        ]
        kinds = [b.kind for b in build_buckets(cards, exemplars_per_class=3)]
        assert kinds.index(CardKind.CONTRADICTION) < kinds.index(CardKind.UNCITED_URL)


# ---------------------------------------------------------------------------
# The RECENCY_DECAY shared-seam extension
# ---------------------------------------------------------------------------


class TestRecencyDecayCard:
    @pytest.mark.asyncio
    async def test_recency_decay_finding_becomes_card(self, db_session: AsyncSession) -> None:
        report = LintReport(
            findings=[
                LintFinding(
                    particle_id="p-old",
                    finding_type="RECENCY_DECAY",
                    severity="INFO",
                    detail="aged past the decay floor",
                )
            ]
        )
        with patch(
            "particles.operations.curation.collect.run_lint",
            AsyncMock(return_value=report),
        ):
            cards = await collect_cards(db_session, semantic=False)
        recency = [c for c in cards if c.kind is CardKind.RECENCY_DECAY]
        assert len(recency) == 1
        assert recency[0].particle_ids == ["p-old"]
        assert recency[0].suggested_gestures == ["affirm", "supersede", "snooze"]

    @pytest.mark.asyncio
    async def test_duplicate_mode_override_decouples_judge_from_semantic(
        self, db_session: AsyncSession
    ) -> None:
        suggest = AsyncMock(return_value=SuggestReport(mode=SuggestMode.REPORT, clusters=[]))
        with (
            patch(
                "particles.operations.curation.collect.run_lint",
                AsyncMock(return_value=LintReport()),
            ),
            patch("particles.operations.curation.collect.suggest_co_evidential", suggest),
        ):
            await collect_cards(db_session, semantic=True, duplicate_mode=SuggestMode.REPORT)
            assert suggest.call_args.kwargs["mode"] is SuggestMode.REPORT
            # Default (None) keeps the coupling: semantic ⇒ LLM_JUDGE.
            await collect_cards(db_session, semantic=True)
            assert suggest.call_args.kwargs["mode"] is SuggestMode.LLM_JUDGE


# ---------------------------------------------------------------------------
# run_memory_audit — extract scoping, degradation flags, verdict census
# ---------------------------------------------------------------------------


class TestRunMemoryAudit:
    @pytest.mark.asyncio
    async def test_extracts_only_pending_snapshots_of_harvested_entries(
        self, db_session: AsyncSession
    ) -> None:
        from particles.corpus.store import update_extraction_status
        from particles.operations.deposit import deposit_text_versioned

        e1, s1, _ = await deposit_text_versioned(
            db_session,
            text="pending entry",
            uri_r="file:///mem/a.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        e2, s2, _ = await deposit_text_versioned(
            db_session,
            text="complete entry",
            uri_r="file:///mem/b.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        await update_extraction_status(db_session, s2, ExtractionStatus.COMPLETE)
        # A third entry exists but was NOT harvested by this audit run.
        e3, _s3, _ = await deposit_text_versioned(
            db_session,
            text="unrelated entry",
            uri_r="file:///other.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        await db_session.commit()

        extract = AsyncMock(return_value=[])
        with patch("particles.operations.extract.extract_snapshot", extract):
            report = await run_memory_audit(
                db_session,
                files_audited=2,
                harvested_entry_ids=[e1, e2],
                semantic=False,
            )
        assert extract.call_count == 1
        assert extract.call_args.args[1:3] == (e1, s1)
        assert report.extracted_snapshots == 1
        assert report.extraction_failures == 0
        assert e3 not in {c.args[1] for c in extract.call_args_list}

    @pytest.mark.asyncio
    async def test_extraction_failure_is_counted_not_fatal(self, db_session: AsyncSession) -> None:
        from particles.operations.deposit import deposit_text_versioned

        e1, _s1, _ = await deposit_text_versioned(
            db_session,
            text="will fail",
            uri_r="file:///mem/fail.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        await db_session.commit()
        with patch(
            "particles.operations.extract.extract_snapshot",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            report = await run_memory_audit(db_session, harvested_entry_ids=[e1], semantic=False)
        assert report.extraction_failures == 1
        assert report.extracted_snapshots == 0

    @pytest.mark.asyncio
    async def test_progress_events_emitted_per_snapshot_and_census(
        self, db_session: AsyncSession
    ) -> None:
        from particles.operations.audit import AuditProgress
        from particles.operations.deposit import deposit_text_versioned

        e1, _s1, _ = await deposit_text_versioned(
            db_session,
            text="first memory",
            uri_r="file:///mem/alpha.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        e2, _s2, _ = await deposit_text_versioned(
            db_session,
            text="second memory",
            uri_r="file:///mem/beta.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        await db_session.commit()

        events: list[AuditProgress] = []
        with patch(
            "particles.operations.extract.extract_snapshot",
            AsyncMock(return_value=["p1", "p2"]),
        ):
            await run_memory_audit(
                db_session,
                harvested_entry_ids=[e1, e2],
                semantic=False,
                on_progress=events.append,
            )

        extract_events = [e for e in events if e.phase == "extract"]
        assert [(e.done, e.total) for e in extract_events] == [(1, 2), (2, 2)]
        assert [e.label for e in extract_events] == ["alpha.md", "beta.md"]
        assert all(e.particles == 2 and not e.failed for e in extract_events)
        census_events = [e for e in events if e.phase == "census"]
        assert len(census_events) == 1
        assert census_events[0].label == "structural checks + duplicate scan"
        # Census event precedes nothing else emitted after extraction.
        assert events.index(census_events[0]) == len(events) - 1

    @pytest.mark.asyncio
    async def test_progress_failure_event(self, db_session: AsyncSession) -> None:
        from particles.operations.audit import AuditProgress
        from particles.operations.deposit import deposit_text_versioned

        e1, _s1, _ = await deposit_text_versioned(
            db_session,
            text="will fail",
            uri_r="file:///mem/broken.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        await db_session.commit()

        events: list[AuditProgress] = []
        with patch(
            "particles.operations.extract.extract_snapshot",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            report = await run_memory_audit(
                db_session,
                harvested_entry_ids=[e1],
                semantic=False,
                on_progress=events.append,
            )
        assert report.extraction_failures == 1
        failed = [e for e in events if e.phase == "extract"]
        assert len(failed) == 1
        assert failed[0].failed is True
        assert failed[0].label == "broken.md"
        assert failed[0].particles is None

    @pytest.mark.asyncio
    async def test_semantic_skip_reason_carried(self, db_session: AsyncSession) -> None:
        report = await run_memory_audit(
            db_session, semantic=False, semantic_skip_reason="no API key"
        )
        assert report.semantic_skipped is True
        assert report.semantic_skip_reason == "no API key"
        assert report.files_audited is None  # re-audit shape

    @pytest.mark.asyncio
    async def test_semantic_on_is_not_skipped(self, db_session: AsyncSession) -> None:
        with patch("particles.operations.audit.collect_cards", AsyncMock(return_value=[])):
            report = await run_memory_audit(db_session, semantic=True)
        assert report.semantic_skipped is False
        assert report.semantic_skip_reason is None

    @pytest.mark.asyncio
    async def test_judged_duplicate_verdict_census(self, db_session: AsyncSession) -> None:
        cards = [
            _card(CardKind.DUPLICATE_PAIR, "a", "b", verdict=JudgeVerdictKind.PARAPHRASE),
            _card(CardKind.DUPLICATE_PAIR, "c", "d", verdict=JudgeVerdictKind.DISTINCT),
            _card(CardKind.DUPLICATE_PAIR, "e", "f"),  # judge returned nothing → unsure
        ]
        with patch(
            "particles.operations.audit.collect_cards", AsyncMock(return_value=cards)
        ) as collect:
            report = await run_memory_audit(db_session, semantic=True, judge=True)
        assert collect.call_args.kwargs["duplicate_mode"] is SuggestMode.LLM_JUDGE
        assert report.judged is True
        assert report.duplicate_verdicts == {"PARAPHRASE": 1, "DISTINCT": 1, "UNSURE": 1}

    @pytest.mark.asyncio
    async def test_default_duplicates_stay_report_mode(self, db_session: AsyncSession) -> None:
        with patch(
            "particles.operations.audit.collect_cards", AsyncMock(return_value=[])
        ) as collect:
            report = await run_memory_audit(db_session, semantic=True, judge=False)
        assert collect.call_args.kwargs["duplicate_mode"] is SuggestMode.REPORT
        assert report.judged is False

    @pytest.mark.asyncio
    async def test_census_is_uncapped(self, db_session: AsyncSession) -> None:
        # Far more cards than curation.session_size (7) — the census keeps them all.
        cards = [_card(CardKind.STALE, f"p{i}") for i in range(30)]
        with patch("particles.operations.audit.collect_cards", AsyncMock(return_value=cards)):
            report = await run_memory_audit(db_session, semantic=False)
        assert report.count(CardKind.STALE) == 30


# ---------------------------------------------------------------------------
# Contradiction-probe bounding: cap, scope, per-pair
# progress, and the report census the §6 disclosures render from.
# ---------------------------------------------------------------------------

# Two orthogonal high-similarity embedding clusters (MiniLM is 384-dim; the
# 0.6 gate sits between ~1.0 within a cluster and 0.0 across them).
_EMB_ONE_A = [0.6, 0.8] + [0.0] * 382
_EMB_ONE_B = [0.61, 0.79] + [0.0] * 382
_EMB_TWO_A = [0.0, 0.0, 0.6, 0.8] + [0.0] * 380
_EMB_TWO_B = [0.0, 0.0, 0.61, 0.79] + [0.0] * 380


def _claim(content: str, entry_id: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)],
    )


class TestContradictionProbeBounding:
    @pytest.mark.asyncio
    async def test_probe_census_and_per_pair_progress(self, db_session: AsyncSession) -> None:
        """A store-scope audit reports the probe census and streams probe events."""
        from particles.operations.audit import AuditProgress
        from particles.store.particle_store import insert_particle

        p_a = _claim("The API listens on port 8000.", "ce-a")
        p_b = _claim("The API listens on port 9000.", "ce-b")
        await insert_particle(db_session, p_a, embedding=_EMB_ONE_A)
        await insert_particle(db_session, p_b, embedding=_EMB_ONE_B)
        await db_session.commit()

        events: list[AuditProgress] = []
        with patch(
            "particles.operations.lint.contradictions._llm_check_contradiction",
            return_value="YES: different ports",
        ):
            report = await run_memory_audit(db_session, semantic=True, on_progress=events.append)

        assert report.contradiction_probe_scope == "store"
        assert report.contradiction_candidate_pairs == 1
        assert report.contradiction_probes_run == 1
        assert report.count(CardKind.CONTRADICTION) == 1
        probe_events = [e for e in events if e.phase == "probe"]
        assert [(e.done, e.total) for e in probe_events] == [(1, 1)]
        assert probe_events[0].label == "contradiction probe"

    @pytest.mark.asyncio
    async def test_cap_binds_and_census_discloses(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``audit.max_contradiction_probes`` bounds the LLM spend; the census shows it."""
        from particles.config import get_config
        from particles.store.particle_store import insert_particle

        # Mutate the cached singleton (reset_config() would dispose the
        # db_session engine); the autouse reset restores it for the next test.
        monkeypatch.setattr(get_config().audit, "max_contradiction_probes", 1)

        # Three mutually-similar particles → three candidate pairs.
        for i, emb in enumerate((_EMB_ONE_A, _EMB_ONE_B, _EMB_ONE_A)):
            await insert_particle(db_session, _claim(f"Claim {i}.", f"ce-{i}"), embedding=emb)
        await db_session.commit()

        with patch(
            "particles.operations.lint.contradictions._llm_check_contradiction",
            return_value=None,
        ) as probe:
            report = await run_memory_audit(db_session, semantic=True)

        assert probe.call_count == 1
        assert report.contradiction_candidate_pairs == 3
        assert report.contradiction_probes_run == 1

    @pytest.mark.asyncio
    async def test_harvested_scope_limits_pairs_to_harvest(self, db_session: AsyncSession) -> None:
        """``contradiction_scope="harvested"`` drops pairs with no harvested side."""
        from particles.store.particle_store import insert_particle

        harvested = _claim("Harvested claim.", "ce-harvest")
        partner = _claim("Pre-existing near-duplicate.", "ce-old-1")
        old_a = _claim("Old claim Y.", "ce-old-2")
        old_b = _claim("Old claim not Y.", "ce-old-3")
        await insert_particle(db_session, harvested, embedding=_EMB_ONE_A)
        await insert_particle(db_session, partner, embedding=_EMB_ONE_B)
        await insert_particle(db_session, old_a, embedding=_EMB_TWO_A)
        await insert_particle(db_session, old_b, embedding=_EMB_TWO_B)
        await db_session.commit()

        with patch(
            "particles.operations.lint.contradictions._llm_check_contradiction",
            return_value=None,
        ) as probe:
            report = await run_memory_audit(
                db_session,
                semantic=True,
                harvested_entry_ids=["ce-harvest"],
                contradiction_scope="harvested",
            )

        # Only the (harvested, partner) pair qualifies; the old-store pair is
        # out of scope even though it clears the similarity gate.
        assert probe.call_count == 1
        assert report.contradiction_probe_scope == "harvested"
        assert report.contradiction_candidate_pairs == 1
        assert report.contradiction_probes_run == 1

    @pytest.mark.asyncio
    async def test_store_pollution_cannot_starve_intra_harvest_pairs(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """regression: the owner-dogfood 3-vs-0 asymmetry cannot recur.

        A harvested contradiction pair plus a store particle that is MORE
        similar to a harvested belief than the harvested beliefs are to each
        other. Under the old pure-similarity order a cap of 1 went to the
        coincidental mixed pair, so identical memory files reported 0
        contradictions on a populated store but found them on an empty one.
        With the two-tier order the intra-harvest pair wins the budget, so the
        intra-harvest findings are the same regardless of store population
        (given cap ≥ intra-harvest pair count). The unpolluted-store baseline is
        ``test_harvested_scope_limits_pairs_to_harvest`` above.
        """
        from particles.config import get_config
        from particles.store.particle_store import insert_particle

        monkeypatch.setattr(get_config().audit, "max_contradiction_probes", 1)

        # Intra-harvest pair at cosine ~0.99; the store particle sits at
        # ~0.99995 to h1 (and ~0.99 to h2) — the mixed pairs out-similar the
        # intra pair, exactly the starvation shape from the dogfood run.
        h1 = _claim("Memory says the port is 8000.", "ce-harvest")
        h2 = _claim("Memory says the port is 9000.", "ce-harvest")
        s1 = _claim("Unrelated store claim.", "ce-old")
        await insert_particle(db_session, h1, embedding=[1.0, 0.0] + [0.0] * 382)
        await insert_particle(db_session, h2, embedding=[0.99, 0.141] + [0.0] * 382)
        await insert_particle(db_session, s1, embedding=[0.9999, 0.0141] + [0.0] * 382)
        await db_session.commit()

        probed: list[tuple[str, str]] = []

        async def _judge(content_a: str, content_b: str) -> str | None:
            probed.append((content_a, content_b))
            return "YES: different ports"

        with patch(
            "particles.operations.lint.contradictions._llm_check_contradiction",
            side_effect=_judge,
        ):
            report = await run_memory_audit(
                db_session,
                semantic=True,
                harvested_entry_ids=["ce-harvest"],
                contradiction_scope="harvested",
            )

        # The single unit of budget went to the intra-harvest pair.
        assert probed == [("Memory says the port is 8000.", "Memory says the port is 9000.")]
        assert report.count(CardKind.CONTRADICTION) == 1
        assert report.contradiction_candidate_pairs == 3
        assert report.contradiction_intra_scope_pairs == 1
        assert report.contradiction_probes_run == 1
        # The capped disclosure names the tier split (§6 honesty stance).
        text = render_audit_report(report)
        assert "probed 1 of 3 candidate pairs" in text
        assert "1 intra-harvest, 2 cross-store" in text

    @pytest.mark.asyncio
    async def test_probe_census_zero_when_semantic_off(self, db_session: AsyncSession) -> None:
        report = await run_memory_audit(db_session, semantic=False)
        assert report.contradiction_probe_scope is None
        assert report.contradiction_candidate_pairs == 0
        assert report.contradiction_probes_run == 0


# ---------------------------------------------------------------------------
# Duplicate-scan harvest scoping: the headline / exemplars are
# partitioned to the harvest, the store-wide total is disclosed in a tail line.
# ---------------------------------------------------------------------------


def _subject_claim(content: str, entry_id: str, subject_id: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        subject_ids=[subject_id],
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)],
    )


class TestDuplicateScanScoping:
    async def _seed_two_pairs(self, session: AsyncSession) -> None:
        """One duplicate pair touching a harvested entry, one purely store-side.

        The store-side pair is the *planted out-of-scope duplicate*: it clears
        the similarity gate but neither side traces to the harvest, so a
        harvest-scoped headline must exclude it while the store-wide total counts
        it.
        """
        from particles.core.schema import Subject
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject

        s1 = Subject(canonical_name="Harvested topic", asserted_by="test")
        s2 = Subject(canonical_name="Store-only topic", asserted_by="test")
        await insert_subject(session, s1)
        await insert_subject(session, s2)

        # S1: a harvested particle + a near-duplicate → in-scope pair.
        await insert_particle(
            session, _subject_claim("Harvested claim.", "ce-harvest", s1.id), embedding=_EMB_ONE_A
        )
        await insert_particle(
            session,
            _subject_claim("Harvested near-duplicate.", "ce-old-1", s1.id),
            embedding=_EMB_ONE_B,
        )
        # S2: two pre-existing particles → out-of-scope pair (planted pollution).
        await insert_particle(
            session, _subject_claim("Old claim.", "ce-old-2", s2.id), embedding=_EMB_TWO_A
        )
        await insert_particle(
            session, _subject_claim("Old near-duplicate.", "ce-old-3", s2.id), embedding=_EMB_TWO_B
        )
        await session.commit()

    @pytest.mark.asyncio
    async def test_harvested_scope_partitions_headline_and_discloses_total(
        self, db_session: AsyncSession
    ) -> None:
        await self._seed_two_pairs(db_session)

        report = await run_memory_audit(
            db_session,
            semantic=False,
            harvested_entry_ids=["ce-harvest"],
            contradiction_scope="harvested",
        )

        # Headline counts only the harvest-touching pair; the store-wide total M
        # is retained for the tail disclosure.
        assert report.duplicate_scope == "harvested"
        assert report.count(CardKind.DUPLICATE_PAIR) == 1
        assert report.duplicate_candidate_pairs_total == 2

        text = render_audit_report(report)
        assert "1 likely-duplicate belief pairs" in text
        assert (
            "duplicate scan is store-wide; 2 candidate pairs total, 1 involve this harvest"
        ) in text

    @pytest.mark.asyncio
    async def test_store_scope_keeps_all_pairs_and_omits_tail(
        self, db_session: AsyncSession
    ) -> None:
        await self._seed_two_pairs(db_session)

        report = await run_memory_audit(
            db_session,
            semantic=False,
            harvested_entry_ids=["ce-harvest"],
            contradiction_scope="store",
        )

        # A re-audit / --scope store leaves the duplicate scan store-wide: both
        # pairs count and no tail line is printed.
        assert report.duplicate_scope == "store"
        assert report.count(CardKind.DUPLICATE_PAIR) == 2
        assert report.duplicate_candidate_pairs_total == 2
        assert "duplicate scan is store-wide;" not in render_audit_report(report)


# ---------------------------------------------------------------------------
# Renderer (§5/§6 — the approved copy)
# ---------------------------------------------------------------------------


def _report(**kwargs: object) -> AuditReport:
    defaults: dict[str, object] = {
        "store": "default",
        "files_audited": 23,
        "beliefs": 212,
        "subjects": 58,
    }
    defaults.update(kwargs)
    return AuditReport(**defaults)  # type: ignore[arg-type]


class TestRenderer:
    def test_header_matches_approved_copy(self) -> None:
        text = render_audit_report(_report())
        assert "Audited 23 memory files → 212 beliefs about 58 subjects." in text

    def test_headline_classes_always_shown_with_hedged_labels(self) -> None:
        text = render_audit_report(_report())
        # Zero counts still render — an audit never hides a class (§5).
        assert "0 potential contradictions" in text
        assert "0 likely-duplicate belief pairs" in text
        assert "0 probably-stale facts" in text
        assert "(0 cross-file, 0 contested at extract time)" in text
        assert "(unjudged similarity candidates; --judge to verify)" in text
        assert "(0 aged past their source's decay horizon, 0 expired)" in text

    def test_headline_counts_compose_the_kinds(self) -> None:
        buckets = [
            AuditBucket(kind=CardKind.CONTRADICTION, count=2),
            AuditBucket(kind=CardKind.CONTESTED, count=2),
            AuditBucket(kind=CardKind.DUPLICATE_PAIR, count=11),
            AuditBucket(kind=CardKind.STALE, count=2),
            AuditBucket(kind=CardKind.RECENCY_DECAY, count=3),
            AuditBucket(kind=CardKind.CONFIDENCE_DECAY, count=2),
        ]
        text = render_audit_report(_report(buckets=buckets))
        assert "4 potential contradictions" in text
        assert "(2 cross-file, 2 contested at extract time)" in text
        assert "11 likely-duplicate belief pairs" in text
        assert "7 probably-stale facts" in text
        assert "(5 aged past their source's decay horizon, 2 expired)" in text

    def test_secondary_lines_only_when_nonzero(self) -> None:
        assert "Also:" not in render_audit_report(_report())
        buckets = [
            AuditBucket(kind=CardKind.UNCITED_URL, count=3),
            AuditBucket(kind=CardKind.NO_SUBJECT, count=6),
        ]
        text = render_audit_report(_report(buckets=buckets))
        assert "Also: 3 cited sources never captured · 6 beliefs have no resolvable subject" in text
        assert "particles deposit <url>" in text

    def test_exemplars_carry_claim_text_partner_and_next_verb(self) -> None:
        card = _card(
            CardKind.CONTRADICTION,
            "aaaaaaaa-1111",
            "bbbbbbbb-2222",
            diagnostic="claims disagree about Jenkins",
        )
        card.particles = [
            _brief("aaaaaaaa-1111", "Staging runs on Jenkins."),
            _brief("bbbbbbbb-2222", "Jenkins no longer runs staging."),
        ]
        buckets = [AuditBucket(kind=CardKind.CONTRADICTION, count=1, exemplars=[card])]
        text = render_audit_report(_report(buckets=buckets))
        assert '"Staging runs on Jenkins."  [aaaaaaaa…]' in text
        assert '↔ "Jenkins no longer runs staging."  [bbbbbbbb…]' in text
        assert "claims disagree about Jenkins" in text
        assert "next: particles review · particles curate --kind contradiction" in text

    def test_every_headline_class_has_its_next_verb(self) -> None:
        buckets = [
            AuditBucket(
                kind=CardKind.DUPLICATE_PAIR,
                count=1,
                exemplars=[_card(CardKind.DUPLICATE_PAIR, "a", "b")],
            ),
            AuditBucket(kind=CardKind.STALE, count=1, exemplars=[_card(CardKind.STALE, "c")]),
        ]
        text = render_audit_report(_report(buckets=buckets))
        assert (
            "next: particles links suggest --judge · particles curate --kind duplicate_pair" in text
        )
        assert "next: particles curate --kind stale" in text

    def test_judged_relabels_verified_duplicates(self) -> None:
        buckets = [AuditBucket(kind=CardKind.DUPLICATE_PAIR, count=3)]
        text = render_audit_report(
            _report(
                buckets=buckets,
                judged=True,
                duplicate_verdicts={"PARAPHRASE": 1, "DISTINCT": 1},
            )
        )
        assert "3 verified duplicate belief pairs" in text
        assert "(LLM-judged: 1 paraphrase, 1 distinct, 1 unsure)" in text
        assert "unjudged similarity candidates" not in text

    def test_skip_line_mirrors_semantic_skipped(self) -> None:
        text = render_audit_report(
            _report(semantic_skipped=True, semantic_skip_reason="no API key")
        )
        assert "contradiction check skipped: no API key" in text

    def test_no_skip_line_when_semantic_ran(self) -> None:
        assert "skipped" not in render_audit_report(_report())

    def test_uncalibrated_confidence_footnote(self) -> None:
        text = render_audit_report(_report())
        assert (
            "confidence on this content is self-reported and capped, "
            "not benchmark-calibrated" in text
        )
        # No beliefs → nothing to disclose confidence about.
        assert "self-reported" not in render_audit_report(_report(beliefs=0))

    def test_footer_hands_off_to_curate(self) -> None:
        assert "Run 'particles curate' to work these down a few at a time." in render_audit_report(
            _report()
        )

    def test_projection_closing_line(self) -> None:
        assert "MEMORY.md" not in render_audit_report(_report())
        text = render_audit_report(_report(projection_rendered=True))
        assert "MEMORY.md was re-projected from the audited store" in text

    def test_reaudit_header_names_the_store(self) -> None:
        text = render_audit_report(_report(files_audited=None, store="memory"))
        assert "Re-audited store 'memory' → 212 beliefs about 58 subjects." in text

    def test_transcripts_counted_in_header(self) -> None:
        text = render_audit_report(_report(transcripts_audited=5))
        assert "Audited 23 memory files + 5 transcripts →" in text


class TestProbeFailureDisclosure:
    """§6 honesty: per-probe failures are counted and disclosed (v1.67.2)."""

    def test_renderer_discloses_probe_failures(self) -> None:
        text = render_audit_report(_report(semantic_probe_failures=3))
        assert (
            "3 semantic probes failed or were declined by the model and were "
            "skipped — contradiction and duplicate counts may read low"
        ) in text

    def test_renderer_singular_probe(self) -> None:
        text = render_audit_report(_report(semantic_probe_failures=1))
        assert "1 semantic probe failed" in text

    def test_zero_failures_render_nothing(self) -> None:
        assert "semantic probe" not in render_audit_report(_report())

    @pytest.mark.asyncio
    async def test_run_memory_audit_counts_probe_failures(self, db_session: AsyncSession) -> None:
        from particles.operations import _llm

        async def _failing_collect(*args: object, **kwargs: object) -> list[object]:
            # Simulate two contradiction probes failing inside the finder scan
            # (each failed _llm_call increments the process counter).
            _llm._failure_count += 2
            return []

        with patch("particles.operations.audit.collect_cards", _failing_collect):
            report = await run_memory_audit(db_session, semantic=False)
        assert report.semantic_probe_failures == 2


class TestProbeBoundingDisclosure:
    """§6 honesty for the bounded probe: a scoped or
    capped probe is disclosed as a lower bound, never a silent partial census."""

    def test_capped_probe_discloses_probed_x_of_y(self) -> None:
        text = render_audit_report(
            _report(
                contradiction_probe_scope="store",
                contradiction_candidate_pairs=400,
                contradiction_probes_run=50,
            )
        )
        assert "probed 50 of 400 candidate pairs" in text
        assert "audit.max_contradiction_probes" in text

    def test_harvested_scope_discloses_and_names_the_opt_out(self) -> None:
        text = render_audit_report(
            _report(
                contradiction_probe_scope="harvested",
                contradiction_candidate_pairs=3,
                contradiction_probes_run=3,
            )
        )
        assert "contradiction probe scoped to this harvest's beliefs" in text
        assert "intra-harvest pairs probed first" in text
        assert "--scope store" in text

    def test_capped_harvested_probe_discloses_tier_split(self) -> None:
        """a capped harvested-scope probe names the tier split, so the
        operator can see whether the budget ever reached the cross-store tier."""
        text = render_audit_report(
            _report(
                contradiction_probe_scope="harvested",
                contradiction_candidate_pairs=400,
                contradiction_intra_scope_pairs=12,
                contradiction_probes_run=50,
            )
        )
        assert "probed 50 of 400 candidate pairs" in text
        assert "12 intra-harvest, 388 cross-store" in text

    def test_capped_store_probe_has_no_tier_split(self) -> None:
        """Store-wide runs have no harvested set, so no tier split is rendered."""
        text = render_audit_report(
            _report(
                contradiction_probe_scope="store",
                contradiction_candidate_pairs=400,
                contradiction_probes_run=50,
            )
        )
        assert "probed 50 of 400 candidate pairs" in text
        assert "intra-harvest" not in text

    def test_uncapped_store_probe_discloses_nothing(self) -> None:
        text = render_audit_report(
            _report(
                contradiction_probe_scope="store",
                contradiction_candidate_pairs=7,
                contradiction_probes_run=7,
            )
        )
        assert "probed" not in text
        assert "scoped to this harvest" not in text

    def test_skipped_semantic_suppresses_the_capped_line(self) -> None:
        # When the semantic pass was skipped (no key / breaker), the skip line
        # owns the disclosure; a "probed 0 of N" line would double-report.
        text = render_audit_report(
            _report(
                semantic_skipped=True,
                contradiction_probe_scope="store",
                contradiction_candidate_pairs=5,
                contradiction_probes_run=0,
            )
        )
        assert "contradiction check skipped" in text
        assert "candidate pairs" not in text


class TestContestedBasisSplit:
    """the class widens; the contradictions headline does not."""

    def test_bucket_counts_each_basis(self) -> None:
        cards = [
            _card(CardKind.CONTESTED, "a", bases=["inconsistency"]),
            _card(CardKind.CONTESTED, "b", bases=["stance", "divergence"]),
            _card(CardKind.CONTESTED, "c", bases=["divergence", "inconsistency"]),
        ]
        bucket = build_buckets(cards, 3)[0]
        assert bucket.count == 3
        # A card firing two bases counts under both — they need not sum to count.
        assert bucket.bases == {"inconsistency": 2, "stance": 1, "divergence": 2}

    def test_observer_only_contested_stays_out_of_the_contradictions_headline(self) -> None:
        """A lens disagreement is not the store contradicting itself."""
        cards = [
            _card(CardKind.CONTRADICTION, "x", "y"),
            _card(CardKind.CONTESTED, "a", bases=["inconsistency"]),
            _card(CardKind.CONTESTED, "b", bases=["stance"]),
            _card(CardKind.CONTESTED, "c", bases=["divergence"]),
        ]
        text = render_audit_report(_report(buckets=build_buckets(cards, 3)))
        # 1 contradiction + 1 inconsistency-basis contested — not 1 + 3.
        assert "2 potential contradictions" in text
        assert "(1 cross-file, 1 contested at extract time)" in text
        assert "2 beliefs contested by observer signal (1 stance, 1 divergence)" in text

    def test_observer_line_absent_when_every_basis_is_inconsistency(self) -> None:
        """The line the dogfood store never sees: no new noise where nothing fired."""
        cards = [_card(CardKind.CONTESTED, "a", bases=["inconsistency"])]
        text = render_audit_report(_report(buckets=build_buckets(cards, 3)))
        assert "contested by observer signal" not in text
        assert "(0 cross-file, 1 contested at extract time)" in text

    def test_pre_0215_buckets_count_as_inconsistency(self) -> None:
        """A bucket with no recorded bases reads as the pre-widening class."""
        report = _report(buckets=[AuditBucket(kind=CardKind.CONTESTED, count=4)])
        assert report.contested_split() == (4, 0)
        assert "4 potential contradictions" in render_audit_report(report)
