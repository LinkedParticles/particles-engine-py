# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``reindex --estimate``.

The decisions are pure and tested as such: which snapshots may be sampled, the
seeded sample, how two claim lists line up, what counts as changed, the share's
interval, and the projection arithmetic. The spending half runs once against a
store with the extractor and the judge stubbed, to pin that it measures each
sample and writes nothing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    ExtractorRef,
    JudgeVerdictKind,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.corpus.deposit import save_blob, sha256
from particles.corpus.store import CorpusEntryRow, SnapshotRow
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.operations.reindex import ReindexPlan, SnapshotPlan
from particles.operations.reindex_estimate import (
    ClaimChange,
    ClaimPairing,
    EstimatePlan,
    choose_sample,
    claim_change,
    normalise_claim,
    pair_claims,
    plan_reindex_estimate,
    project_sweep,
    run_reindex_estimate,
    sample_population,
    wilson_interval,
)
from particles.store.particle_store import ParticleRow, get_particle, insert_particle


def _sp(snapshot_id: str, **kwargs: Any) -> SnapshotPlan:
    fields: dict[str, Any] = {
        "entry_id": f"e-{snapshot_id}",
        "snapshot_id": snapshot_id,
        "particles": 3,
        "source_bytes": 1000,
    }
    fields.update(kwargs)
    return SnapshotPlan(**fields)


# ---------------------------------------------------------------------------
# Pure decisions
# ---------------------------------------------------------------------------


class TestSamplePopulation:
    def test_only_snapshots_with_claims_and_a_blob_outside_a_replay_are_sampleable(self) -> None:
        plan = ReindexPlan(
            entries=4,
            snapshots=4,
            particles=6,
            missing_blobs=1,
            scope_description="x",
            snapshot_plans=[
                _sp("ok"),
                _sp("replay", replayed=True),
                _sp("gone", blob_missing=True),
                _sp("empty", particles=0),
            ],
        )
        population = sample_population(plan)
        assert [sp.snapshot_id for sp in population.eligible] == ["ok"]
        assert (population.replayed, population.blob_missing, population.no_claims) == (1, 1, 1)


class TestChooseSample:
    def test_seeded_and_independent_of_gather_order(self) -> None:
        population = [_sp(f"s{i:02d}") for i in range(30)]
        first = choose_sample(population, 5, seed=7)
        again = choose_sample(list(reversed(population)), 5, seed=7)
        assert first == again
        assert len(first) == 5
        assert choose_sample(population, 5, seed=8) != first

    def test_a_sample_at_least_the_population_takes_everything(self) -> None:
        population = [_sp("b"), _sp("a")]
        assert [sp.snapshot_id for sp in choose_sample(population, 10, seed=0)] == ["a", "b"]


class TestPairClaims:
    def test_verbatim_matches_ignore_case_spacing_and_a_final_stop(self) -> None:
        assert normalise_claim("  The  Bridge opened.") == "the bridge opened"
        pairing = pair_claims(["The bridge opened in 1932."], ["the bridge  opened in 1932"])
        assert pairing == ClaimPairing(exact=1, pairs=[], unmatched_stored=0, unmatched_new=0)

    def test_residuals_pair_with_their_closest_and_leftovers_are_unmatched(self) -> None:
        stored = ["Alice lives in Berlin.", "Bob works at Acme.", "Carol is a pilot."]
        new = ["Bob is employed at Acme.", "Alice lives in Berlin, Germany."]
        pairing = pair_claims(stored, new)
        assert pairing.exact == 0
        assert pairing.pairs == [
            ("Bob works at Acme.", "Bob is employed at Acme."),
            ("Alice lives in Berlin.", "Alice lives in Berlin, Germany."),
        ]
        assert (pairing.unmatched_stored, pairing.unmatched_new) == (1, 0)

    def test_duplicates_match_one_for_one(self) -> None:
        pairing = pair_claims(["X is Y.", "X is Y."], ["X is Y."])
        assert (pairing.exact, pairing.unmatched_stored, pairing.unmatched_new) == (1, 1, 0)


class TestClaimChange:
    def test_paraphrases_and_verbatim_matches_are_unchanged(self) -> None:
        pairing = ClaimPairing(exact=2, pairs=[("a", "b")], unmatched_stored=0, unmatched_new=0)
        change = claim_change(pairing, [JudgeVerdictKind.PARAPHRASE])
        assert change == ClaimChange(differing=0, total=6)
        assert not change.changed

    def test_unsure_and_distinct_count_both_claims_and_unmatched_count_once(self) -> None:
        pairing = ClaimPairing(
            exact=1, pairs=[("a", "b"), ("c", "d")], unmatched_stored=1, unmatched_new=2
        )
        change = claim_change(pairing, [JudgeVerdictKind.UNSURE, JudgeVerdictKind.DISTINCT])
        assert change == ClaimChange(differing=7, total=9)
        assert change.changed

    def test_a_verdict_count_mismatch_is_refused(self) -> None:
        pairing = ClaimPairing(exact=0, pairs=[("a", "b")], unmatched_stored=0, unmatched_new=0)
        with pytest.raises(ValueError, match="1 pair"):
            claim_change(pairing, [])


class TestWilsonInterval:
    def test_known_values(self) -> None:
        low, high = wilson_interval(5, 10)
        assert low == pytest.approx(0.2366, abs=1e-3)
        assert high == pytest.approx(0.7634, abs=1e-3)
        assert wilson_interval(0, 0) == (0.0, 1.0)
        low, high = wilson_interval(0, 12)
        assert low == 0.0 and 0.2 < high < 0.3


class TestProjectSweep:
    def test_share_and_cost_scale_by_population_and_bytes(self) -> None:
        outcomes = [
            (1000, ClaimChange(differing=2, total=10)),
            (3000, ClaimChange(differing=0, total=10)),
        ]
        projection = project_sweep(
            outcomes=outcomes, population=40, sweep_bytes=100_000, sample_extraction_usd=0.08
        )
        assert projection.measured == 2
        assert projection.changed == 1
        assert projection.share == 0.5
        assert projection.projected_changed == 20
        assert projection.claim_change_rate == pytest.approx(0.1)
        # $0.08 for 4,000 bytes → $2.00 for 100,000 bytes.
        assert projection.projected_sweep_usd == pytest.approx(2.0)

    def test_an_unpriced_or_empty_sample_projects_no_cost(self) -> None:
        unpriced = project_sweep(
            outcomes=[(10, ClaimChange(differing=0, total=2))],
            population=5,
            sweep_bytes=50,
            sample_extraction_usd=None,
        )
        assert unpriced.projected_sweep_usd is None
        empty = project_sweep(outcomes=[], population=5, sweep_bytes=50, sample_extraction_usd=0.0)
        assert empty.share is None
        assert empty.projected_changed is None
        assert empty.projected_sweep_usd is None


# ---------------------------------------------------------------------------
# Gather and apply against a store
# ---------------------------------------------------------------------------


async def _entry_with_claims(
    session: AsyncSession, text: str, claims: list[str], *, version: str = "0.15.0"
) -> tuple[CorpusEntry, Snapshot, list[Particle]]:
    raw = text.encode()
    content_hash = sha256(raw)
    save_blob(raw, content_hash)
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="WEB_PAGE",
        uri_r=f"https://example.com/{uuid.uuid4().hex[:6]}",
        deposited_by="test",
    )
    session.add(CorpusEntryRow.from_model(entry))
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash=content_hash,
        extraction_status=ExtractionStatus.COMPLETE,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    session.add(SnapshotRow.from_model(snap, entry.entry_id))
    await session.flush()
    particles = []
    for claim in claims:
        particle = Particle(
            id=str(uuid.uuid4()),
            content=claim,
            confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            asserted_at=datetime.now(UTC),
            status=Status.ACTIVE,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=entry.entry_id,
                    snapshot_id=snap.snapshot_id,
                )
            ],
            extractor_ref=ExtractorRef(name="general-extractor", version=version),
        )
        await insert_particle(session, particle)
        particles.append(particle)
    return entry, snap, particles


class _Extractor:
    """Answers each snapshot with a fixed claim list, keyed by its text."""

    EXTRACTOR_ID = "general-extractor"

    def __init__(self, answers: dict[bytes, list[str]]) -> None:
        self.answers = answers
        self.kwargs: list[dict[str, Any]] = []

    async def extract(self, snapshot: Snapshot, content: bytes, **kwargs: Any) -> ExtractionResult:
        self.kwargs.append(kwargs)
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=c, confidence_value=0.8, uncertainty_nature=UncertaintyNature.EPISTEMIC
                )
                for c in self.answers[content]
            ]
        )


class TestEstimateAgainstAStore:
    @pytest.mark.asyncio
    async def test_plan_samples_the_version_scope_and_prices_it(
        self, db_session: AsyncSession
    ) -> None:
        await _entry_with_claims(db_session, "Alpha text.", ["Alpha is first."])
        await _entry_with_claims(db_session, "Other text.", ["Other."], version="0.16.0")
        await db_session.commit()

        plan = await plan_reindex_estimate(
            db_session, extractor_version="0.15.0", include_failed=False, sample_size=5, seed=1
        )
        assert plan.population == 1
        assert [sp.source_bytes for sp in plan.sample] == [len(b"Alpha text.")]
        assert plan.seed == 1
        assert plan.sample_estimate_calls >= 1
        assert any("Estimate: re-extract 1 of 1" in line for line in plan.format_lines())

    @pytest.mark.asyncio
    async def test_run_measures_each_sample_and_writes_nothing(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, _, same = await _entry_with_claims(
            db_session, "Same text.", ["The bridge opened in 1932.", "It is steel."]
        )
        _, _, changed = await _entry_with_claims(
            db_session, "Changed text.", ["Maria lives near here."]
        )
        await db_session.commit()
        extractor = _Extractor(
            {
                b"Same text.": ["the bridge opened in 1932", "It is made of steel."],
                b"Changed text.": ["Maria lives in Kreuzberg, Berlin.", "Kreuzberg is in Berlin."],
            }
        )
        monkeypatch.setattr(
            "particles.operations.reindex_estimate.select_extractor", lambda _t: extractor
        )
        judged: list[tuple[str, str]] = []

        async def judge(pairs: list[tuple[str, str]], *, batch_size: int) -> list[JudgeVerdictKind]:
            judged.extend(pairs)
            return [
                JudgeVerdictKind.PARAPHRASE if "steel" in a else JudgeVerdictKind.DISTINCT
                for a, _ in pairs
            ]

        monkeypatch.setattr("particles.operations.reindex_estimate.judge_claim_pairs", judge)
        before = await db_session.scalar(select(func.count()).select_from(ParticleRow))

        plan = await plan_reindex_estimate(
            db_session, extractor_version="0.15.0", include_failed=False
        )
        report = await run_reindex_estimate(db_session, plan)

        by_snapshot = {s.stored_claims: s for s in report.samples}
        assert by_snapshot[2].changed is False
        assert (by_snapshot[2].exact, by_snapshot[2].judged) == (1, 1)
        assert by_snapshot[1].changed is True
        assert report.projection.measured == 2
        assert report.projection.changed == 1
        assert report.projection.share == 0.5
        # The stored claims were named superseded so carry-forward reads every chunk.
        supersede_sets = [k["supersede_ids"] for k in extractor.kwargs]
        assert frozenset(p.id for p in same) in supersede_sets
        # Nothing was written: same particle count, every stored claim still ACTIVE.
        after = await db_session.scalar(select(func.count()).select_from(ParticleRow))
        assert after == before
        for particle in [*same, *changed]:
            stored = await get_particle(db_session, particle.id)
            assert stored is not None and stored.status is Status.ACTIVE
        assert any("1 of 2 sampled snapshot(s) changed" in line for line in report.format_lines())

    @pytest.mark.asyncio
    async def test_a_failed_sample_is_reported_and_left_out_of_the_share(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _entry_with_claims(db_session, "Boom.", ["A claim."])
        await db_session.commit()

        class _Failing:
            async def extract(self, *args: Any, **kwargs: Any) -> ExtractionResult:
                return ExtractionResult(transient_error_count=1)

        monkeypatch.setattr(
            "particles.operations.reindex_estimate.select_extractor", lambda _t: _Failing()
        )
        plan = await plan_reindex_estimate(
            db_session, extractor_version="0.15.0", include_failed=False
        )
        report = await run_reindex_estimate(db_session, plan)
        assert report.samples[0].error == "1 extraction call(s) failed"
        assert report.projection.measured == 0
        assert isinstance(report.plan, EstimatePlan)
