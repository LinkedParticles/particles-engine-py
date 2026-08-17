"""Tests for the UtilityPolicy: lens round-trip, composition, apply_utility."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from particles.config import get_config
from particles.core.schema import TrustLensDefinition, TrustLensUtilityRule
from particles.operations.query.utility_policy import (
    EMPTY_UTILITY_POLICY,
    apply_utility,
    load_utility_policy,
)
from particles.store.lens_store import adopt_lens, get_lens, materialise_lens
from particles.store.utility_store import record_utility_events

_NOW = datetime(2026, 7, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_lens_utility_rules_round_trip(db_session: object) -> None:
    lens = TrustLensDefinition(
        name="acme",
        version=1,
        utility_rules=[
            TrustLensUtilityRule(half_life_uses_days=45, rank_lift=0.05),
            TrustLensUtilityRule(
                scope="url_pattern",
                pattern=r"claude-code://",
                half_life_uses_days=10,
                rank_lift=0.09,
            ),
        ],
    )
    await materialise_lens(db_session, lens)  # type: ignore[arg-type]
    loaded = await get_lens(db_session, "acme")  # type: ignore[arg-type]
    assert loaded is not None
    assert len(loaded.utility_rules) == 2
    by_scope = {r.scope: r for r in loaded.utility_rules}
    assert by_scope["default"].half_life_uses_days == 45
    assert by_scope["url_pattern"].pattern == r"claude-code://"
    assert by_scope["url_pattern"].rank_lift == 0.09


@pytest.mark.asyncio
async def test_disabled_config_yields_empty_policy(db_session: object) -> None:
    # utility.enabled = False ⇒ neutral — the pre-0190 ranking.
    # The autouse fixture calls reset_config() before the next test, so mutating
    # the cached config here does not leak (and reset_config() mid-test would
    # dispose the db_session engine).
    get_config().utility.enabled = False
    policy = await load_utility_policy(db_session)  # type: ignore[arg-type]
    assert policy is EMPTY_UTILITY_POLICY


@pytest.mark.asyncio
async def test_default_from_local_config(db_session: object) -> None:
    policy = await load_utility_policy(db_session)  # type: ignore[arg-type]
    assert policy.default is not None
    # The calibrated default: (half_life_uses_days, rank_lift), set there
    # and re-centred into the intersection of every rendered
    # head size's band; that intersection was re-measured after the
    # duplicate merge. It was re-centred again once the
    # duplicate ceiling was removed entirely: with no upper edge the
    # log-midpoint rule no longer applies, so 0.015 is chosen for margin above
    # the N=60 floor of 0.011, bounded by the omega coupling.
    assert policy.default == (30.0, 0.015)


@pytest.mark.asyncio
async def test_lens_makes_default_more_skeptical_only(db_session: object) -> None:
    # A lens offering MORE promotion than local (a higher rank_lift) cannot
    # inflate; a lens offering LESS makes the composed default more skeptical.
    lens = TrustLensDefinition(
        name="skeptic",
        version=1,
        utility_rules=[TrustLensUtilityRule(half_life_uses_days=10, rank_lift=0.005)],
    )
    await materialise_lens(db_session, lens)  # type: ignore[arg-type]
    await adopt_lens(db_session, "skeptic")  # type: ignore[arg-type]
    policy = await load_utility_policy(db_session)  # type: ignore[arg-type]
    # min each: half_life min(30,10)=10, rank_lift min(0.015,0.005)=0.005
    assert policy.default == (10.0, 0.005)


@pytest.mark.asyncio
async def test_apply_utility_promotes_reinforced_belief(db_session: object) -> None:
    # p-a has utility evidence; p-b has none → only p-a is promoted, p-b unchanged.
    await record_utility_events(
        db_session,  # type: ignore[arg-type]
        "sess-1",
        {"p-a": "literal"},
        observed_at=_NOW,
    )
    eff = {"p-a": 0.50, "p-b": 0.60}
    source_info = {"p-a": ("CONVERSATION", None), "p-b": ("CONVERSATION", None)}
    out = await apply_utility(db_session, eff, source_info)  # type: ignore[arg-type]
    assert out["p-a"] > 0.50  # promoted by its additive rank bonus
    assert math.isclose(out["p-b"], 0.60, abs_tol=1e-9)  # neutral (no evidence)


@pytest.mark.asyncio
async def test_apply_utility_noop_when_empty(db_session: object) -> None:
    out = await apply_utility(db_session, {}, {})  # type: ignore[arg-type]
    assert out == {}


@pytest.mark.asyncio
async def test_retrieve_ranked_promotes_below_cut_belief_into_topk(db_session: object) -> None:
    """efficacy: utility must lift an acted-upon belief tied at the
    confidence ceiling *into* the top_k, not just reorder an already-cut set.

    Regression for the truncation-order bug: 3 beliefs tie at effective
    confidence, top_k=2. Without utility the id-order cut drops one; a utility
    event on that dropped belief must pull it into the top-2.
    """
    import numpy as np

    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        QueryRequest,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.operations.query import retrieve_ranked
    from particles.store.particle_store import insert_particle

    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()

    def mk(pid: str) -> Particle:
        return Particle(
            id=pid,
            content=f"belief {pid}",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            provenance=[
                ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id="s")
            ],
        )

    # ids sort so p-z is last → without utility it falls outside top_k=2.
    for pid in ("p-a", "p-b", "p-z"):
        await insert_particle(db_session, mk(pid), emb)  # type: ignore[arg-type]
    await record_utility_events(db_session, "sess", {"p-z": "literal"}, observed_at=_NOW)  # type: ignore[arg-type]

    # use_embeddings=False ⇒ the question text is unused (pure eff-conf ranking).
    req = QueryRequest(
        question="rank", top_k=2, include_non_asserted=False, include_document_meta=False
    )

    plain = await retrieve_ranked(db_session, req, use_embeddings=False)  # type: ignore[arg-type]
    boosted = await retrieve_ranked(db_session, req, use_embeddings=False, apply_utility=True)  # type: ignore[arg-type]

    assert "p-z" not in {p.id for p, _, _ in plain}  # dropped by the id-order cut
    assert "p-z" in {p.id for p, _, _ in boosted}  # utility pulled it into the top-2


@pytest.mark.asyncio
async def test_kernel_gate_utility_only_when_flag_set(db_session: object) -> None:
    """The §6 boundary: score_effective_confidence applies utility ONLY on the flag.

    This is what keeps the semantic-search `query` path (which calls the kernel
    with the default `apply_utility_factor=False`) free of utility distortion,
    while the projection/digest path (flag=True) gets the promotion.
    """
    import numpy as np

    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.operations.query.effective_confidence import score_effective_confidence
    from particles.store.particle_store import insert_particle

    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    particle = Particle(
        content="Every commit needs `git commit -s`",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
    )
    await insert_particle(db_session, particle, emb)  # type: ignore[arg-type]
    await record_utility_events(db_session, "sess-1", {particle.id: "literal"}, observed_at=_NOW)  # type: ignore[arg-type]

    without = await score_effective_confidence(db_session, [particle])  # type: ignore[arg-type]
    with_util = await score_effective_confidence(db_session, [particle], apply_utility_factor=True)  # type: ignore[arg-type]

    # Default (query path): truth-only score, no utility promotion.
    assert math.isclose(without[particle.id], 0.5, abs_tol=1e-9)
    # Projection/digest path: the reinforced belief is promoted above its truth score.
    assert with_util[particle.id] > without[particle.id]


@pytest.mark.asyncio
async def test_high_use_low_confidence_outranks_low_use_high_confidence(
    db_session: object,
) -> None:
    """The exact defect the supersession fixed (report 6).

    The measured failure was **cap saturation discarding count magnitude**. With
    `weight=0.5` the old factor was ~0.95 of the way to `cap` by R≈6, so every
    belief used more than a handful of times pegged ×1.4 — ~1,542 of them in the
    dogfood store — and *within that tied cohort the projection reordered by base
    effective confidence*, the exact ranking utility was meant to override.

    So a belief acted on 24× but scored a hair lower lost to one acted on 6×:
        multiplier: 0.69×1.4 = 0.966  <  0.70×1.4 = 0.980   ← wrong winner
        additive:   0.69+λln25 = 0.754 > 0.70+λln7 = 0.739  ← right winner

    These are the real store's numbers: the uncalibrated cap ties
    essentially the whole head near 0.70, which is why the *separation* has to
    come from the utility term rather than from the confidence spread.
    """
    import numpy as np

    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.operations.query.effective_confidence import score_effective_confidence
    from particles.store.particle_store import insert_particle

    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()

    def mk(content: str, value: float) -> Particle:
        return Particle(
            content=content,
            confidence=Confidence(
                value=value, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
            ),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            provenance=[
                ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
            ],
        )

    load_bearing = mk("Never prepend `export PATH=...` to a command", 0.69)  # used 24×
    lesser = mk("The pre-commit hook runs ruff", 0.70)  # used 6×
    for p in (load_bearing, lesser):
        await insert_particle(db_session, p, emb)  # type: ignore[arg-type]

    for pid, uses in ((load_bearing.id, 24), (lesser.id, 6)):
        for n in range(uses):
            await record_utility_events(
                db_session,  # type: ignore[arg-type]
                f"sess-{pid[:4]}-{n}",
                {pid: "literal"},
                observed_at=_NOW,
            )

    # Truth axis alone ranks the marginally-more-confident belief first.
    truth = await score_effective_confidence(db_session, [load_bearing, lesser])  # type: ignore[arg-type]
    assert truth[lesser.id] > truth[load_bearing.id]

    # The projection key inverts them: 24 uses beat 6. Under the old
    # multiplier both pegged the cap and this assertion failed.
    rank = await score_effective_confidence(
        db_session,  # type: ignore[arg-type]
        [load_bearing, lesser],
        apply_utility_factor=True,
    )
    assert rank[load_bearing.id] > rank[lesser.id]

    # Explicitly: the superseded form would have got this wrong.
    assert 0.69 * 1.4 < 0.70 * 1.4

    # …and the truth-axis score itself was never mutated.
    assert math.isclose(truth[load_bearing.id], 0.69, abs_tol=1e-9)
    assert math.isclose(truth[lesser.id], 0.70, abs_tol=1e-9)


@pytest.mark.asyncio
async def test_digest_displays_confidence_but_ranks_by_lift(db_session: object) -> None:
    """§Consequences: rank_score is not a confidence.

    `score_confidence_and_rank` hands the digest both quantities from one pass —
    it orders by the lifted score but renders the untouched effective
    confidence, so a lifted belief is never labelled with a >1.0 "confidence".
    """
    import numpy as np

    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.operations.query.effective_confidence import score_confidence_and_rank
    from particles.store.particle_store import insert_particle

    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    particle = Particle(
        content="`git commit -s` adds a `Signed-off-by` trailer",
        confidence=Confidence(value=0.99, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
    )
    await insert_particle(db_session, particle, emb)  # type: ignore[arg-type]
    for n in range(30):
        await record_utility_events(
            db_session,  # type: ignore[arg-type]
            f"s-{n}",
            {particle.id: "literal"},
            observed_at=_NOW,
        )

    eff, rank = await score_confidence_and_rank(db_session, [particle])  # type: ignore[arg-type]
    assert math.isclose(eff[particle.id], 0.99, abs_tol=1e-9)  # displayed, untouched
    assert rank[particle.id] > eff[particle.id]  # ordered, lifted

    # With utility off the two maps agree, so a caller may use this always.
    eff2, rank2 = await score_confidence_and_rank(
        db_session,  # type: ignore[arg-type]
        [particle],
        with_utility=False,
    )
    assert eff2 == rank2
