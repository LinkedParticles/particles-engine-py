"""Tests for operations/links_suggest.py — `links suggest`.

Covers REPORT-mode candidate proposal (the logic that used to be the L-IDX-01
lint check), the LLM-judge verdict mapping, the --apply path, the
apply-confirm-threshold guard, and the legacy config-key deprecation shim.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from particles.core.schema import (
    AssertionModality,
    Confidence,
    JudgeVerdictKind,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    Subject,
    SuggestMode,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.operations.links_suggest import (
    ApplyConfirmationRequired,
    suggest_co_evidential,
)
from particles.store.particle_store import insert_particle
from particles.store.relation_store import create_relation, get_co_evidential_group
from particles.store.subject_store import insert_subject

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Two near-identical unit vectors (cosine sim ≈ 1.0) vs an orthogonal one.
_EMB_A = np.array([0.6, 0.8] + [0.0] * 382, dtype=np.float32)
_EMB_B = np.array([0.61, 0.79] + [0.0] * 382, dtype=np.float32)
_EMB_ORTHO = np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32)
_PID_A = "00000000-0000-0000-0000-00000000000a"
_PID_B = "00000000-0000-0000-0000-00000000000b"


def _mk_particle(
    content: str,
    particle_id: str,
    subject_id: str,
    *,
    modality: AssertionModality = AssertionModality.FALSIFIABLE,
) -> Particle:
    return Particle(
        id=particle_id,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        subject_ids=[subject_id],
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce-default")],
        assertion_modality=modality,
    )


async def _seed_pair(
    session: AsyncSession,
    *,
    emb_b: np.ndarray = _EMB_B,
    content_a: str = "Acme acquired Widget.",
    content_b: str = "Acme Corp bought Widget.",
) -> Subject:
    subj = Subject(canonical_name="Acme Corp", asserted_by="test")
    await insert_subject(session, subj)
    p_a = _mk_particle(content_a, _PID_A, subj.id)
    p_b = _mk_particle(content_b, _PID_B, subj.id)
    await insert_particle(session, p_a, embedding=_EMB_A.tolist())
    await insert_particle(session, p_b, embedding=emb_b.tolist())
    await session.commit()
    return subj


@pytest.mark.asyncio
async def test_report_proposes_high_similarity_pair(db_session: AsyncSession) -> None:
    """REPORT mode surfaces a same-Subject pair above threshold as a candidate."""
    subj = await _seed_pair(db_session)

    report = await suggest_co_evidential(db_session, subject_id=subj.id)

    assert report.mode is SuggestMode.REPORT
    assert report.total_candidates == 1
    assert len(report.clusters) == 1
    cluster = report.clusters[0]
    assert cluster.subject_id == subj.id
    assert cluster.subject_name == "Acme Corp"
    cand = cluster.candidates[0]
    assert {cand.particle_a, cand.particle_b} == {_PID_A, _PID_B}
    assert cand.similarity >= 0.92
    assert cand.verdict is None  # REPORT never judges
    assert cand.applied is False


@pytest.mark.asyncio
async def test_non_falsifiable_excluded_from_candidates(db_session: AsyncSession) -> None:
    """a near-identical EVALUATIVE particle is never a co-evidential candidate."""
    subj = Subject(canonical_name="Acme Corp", asserted_by="test")
    await insert_subject(db_session, subj)
    p_a = _mk_particle("Acme acquired Widget.", _PID_A, subj.id)
    p_b = _mk_particle(
        "Acme buying Widget was a great move.",
        _PID_B,
        subj.id,
        modality=AssertionModality.EVALUATIVE,
    )
    await insert_particle(db_session, p_a, embedding=_EMB_A.tolist())
    await insert_particle(db_session, p_b, embedding=_EMB_B.tolist())
    await db_session.commit()

    # Only one truth-apt particle remains in the subject → no pair, no candidate.
    report = await suggest_co_evidential(db_session, subject_id=subj.id)
    assert report.total_candidates == 0


@pytest.mark.asyncio
async def test_different_holder_stances_not_co_evidential(db_session: AsyncSession) -> None:
    """M2: two near-identical stances by DIFFERENT holders are never
    proposed as co-evidential — merging would collapse distinct positions."""
    from particles.core.stance import STANCE_HOLDER_KEY

    subj = Subject(canonical_name="Acme Corp", asserted_by="test")
    await insert_subject(db_session, subj)
    p_a = _mk_particle("alice endorses the claim.", _PID_A, subj.id).model_copy(
        update={"properties": {STANCE_HOLDER_KEY: "x:alice"}}
    )
    p_b = _mk_particle("bob endorses the claim.", _PID_B, subj.id).model_copy(
        update={"properties": {STANCE_HOLDER_KEY: "x:bob"}}
    )
    await insert_particle(db_session, p_a, embedding=_EMB_A.tolist())
    await insert_particle(db_session, p_b, embedding=_EMB_B.tolist())
    await db_session.commit()

    report = await suggest_co_evidential(db_session, subject_id=subj.id)
    assert report.total_candidates == 0


@pytest.mark.asyncio
async def test_same_holder_stances_remain_co_evidential(db_session: AsyncSession) -> None:
    """Two same-holder stances (two sources for one attitude) are still
    co-evidence candidates — M2 only excludes holder *mismatches*."""
    from particles.core.stance import STANCE_HOLDER_KEY

    subj = Subject(canonical_name="Acme Corp", asserted_by="test")
    await insert_subject(db_session, subj)
    p_a = _mk_particle("alice endorses the claim.", _PID_A, subj.id).model_copy(
        update={"properties": {STANCE_HOLDER_KEY: "x:alice"}}
    )
    p_b = _mk_particle("alice endorses the claim (mirror).", _PID_B, subj.id).model_copy(
        update={"properties": {STANCE_HOLDER_KEY: "x:alice"}}
    )
    await insert_particle(db_session, p_a, embedding=_EMB_A.tolist())
    await insert_particle(db_session, p_b, embedding=_EMB_B.tolist())
    await db_session.commit()

    report = await suggest_co_evidential(db_session, subject_id=subj.id)
    assert report.total_candidates == 1


@pytest.mark.asyncio
async def test_report_skips_already_linked_pairs(db_session: AsyncSession) -> None:
    """A pair already linked CO_EVIDENTIAL is not re-proposed."""
    subj = await _seed_pair(db_session)
    await create_relation(
        db_session, _PID_A, _PID_B, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    report = await suggest_co_evidential(db_session, subject_id=subj.id)
    assert report.total_candidates == 0


@pytest.mark.asyncio
async def test_report_respects_threshold(db_session: AsyncSession) -> None:
    """A dissimilar pair (orthogonal embeddings) is below threshold and skipped."""
    subj = await _seed_pair(db_session, emb_b=_EMB_ORTHO)

    report = await suggest_co_evidential(db_session, subject_id=subj.id)
    assert report.total_candidates == 0


@pytest.mark.asyncio
async def test_explicit_threshold_override(db_session: AsyncSession) -> None:
    """The explicit threshold argument overrides the config default both ways."""
    subj = await _seed_pair(db_session)

    # The pair's cosine similarity ≈ 0.9999: a low floor admits it, 1.0 excludes it.
    report_lo = await suggest_co_evidential(db_session, subject_id=subj.id, threshold=0.5)
    report_hi = await suggest_co_evidential(db_session, subject_id=subj.id, threshold=1.0)
    assert report_lo.total_candidates == 1
    assert report_hi.total_candidates == 0


@pytest.mark.asyncio
async def test_unknown_subject_yields_warning(db_session: AsyncSession) -> None:
    """A missing subject_id produces no candidates and a warning, not an error."""
    report = await suggest_co_evidential(db_session, subject_id="does-not-exist")
    assert report.total_candidates == 0
    assert any("not found" in w for w in report.warnings)


@pytest.mark.asyncio
async def test_llm_judge_assigns_verdicts(db_session: AsyncSession) -> None:
    """LLM_JUDGE maps the model's JSON verdict onto each candidate; no mutation."""
    subj = await _seed_pair(db_session)
    key = f"{_PID_A[:8]}+{_PID_B[:8]}"

    # Patch the shared LLM seam where links_suggest defers its import.
    with patch(
        "particles.operations._llm._llm_call",
        return_value=f'{{"{key}": "PARAPHRASE"}}',
    ) as mock_llm:
        report = await suggest_co_evidential(
            db_session, subject_id=subj.id, mode=SuggestMode.LLM_JUDGE
        )

    mock_llm.assert_called_once()
    assert report.judged_pairs == 1
    assert report.applied_pairs == 0
    assert report.clusters[0].candidates[0].verdict is JudgeVerdictKind.PARAPHRASE
    # LLM_JUDGE must not create a relation.
    assert await get_co_evidential_group(db_session, _PID_A) == {_PID_A}


@pytest.mark.asyncio
async def test_llm_judge_unreturned_pair_defaults_unsure(db_session: AsyncSession) -> None:
    """A candidate the LLM omits from its JSON defaults to UNSURE."""
    subj = await _seed_pair(db_session)

    with patch("particles.operations._llm._llm_call", return_value="{}"):
        report = await suggest_co_evidential(
            db_session, subject_id=subj.id, mode=SuggestMode.LLM_JUDGE
        )
    assert report.clusters[0].candidates[0].verdict is JudgeVerdictKind.UNSURE


@pytest.mark.asyncio
async def test_apply_links_paraphrase_pairs(db_session: AsyncSession) -> None:
    """APPLY links PARAPHRASE pairs via a CO_EVIDENTIAL relation (created_by=LLM_JUDGE)."""
    subj = await _seed_pair(db_session)
    key = f"{_PID_A[:8]}+{_PID_B[:8]}"

    with patch(
        "particles.operations._llm._llm_call",
        return_value=f'{{"{key}": "PARAPHRASE"}}',
    ):
        report = await suggest_co_evidential(db_session, subject_id=subj.id, mode=SuggestMode.APPLY)

    assert report.applied_pairs == 1
    assert report.clusters[0].candidates[0].applied is True
    group = await get_co_evidential_group(db_session, _PID_A)
    assert group == {_PID_A, _PID_B}
    from particles.store.relation_store import get_relations_for_particle

    rels = await get_relations_for_particle(db_session, _PID_A)
    assert rels[0].created_by is RelationCreatedBy.LLM_JUDGE


@pytest.mark.asyncio
async def test_apply_skips_non_paraphrase(db_session: AsyncSession) -> None:
    """DISTINCT / UNSURE verdicts are never auto-linked."""
    subj = await _seed_pair(db_session)
    key = f"{_PID_A[:8]}+{_PID_B[:8]}"

    with patch(
        "particles.operations._llm._llm_call",
        return_value=f'{{"{key}": "DISTINCT"}}',
    ):
        report = await suggest_co_evidential(db_session, subject_id=subj.id, mode=SuggestMode.APPLY)

    assert report.applied_pairs == 0
    assert await get_co_evidential_group(db_session, _PID_A) == {_PID_A}


@pytest.mark.asyncio
async def test_apply_confirm_threshold_guard(db_session: AsyncSession) -> None:
    """APPLY over the confirm threshold raises unless confirmed; nothing is linked."""
    # Force the threshold to 0 so a single paraphrase pair trips the guard.
    # The autouse config-reset fixture restores the default for the next test.
    from particles.config import get_config

    get_config().links_suggest.apply_confirm_threshold = 0

    subj = await _seed_pair(db_session)
    key = f"{_PID_A[:8]}+{_PID_B[:8]}"

    with patch(
        "particles.operations._llm._llm_call",
        return_value=f'{{"{key}": "PARAPHRASE"}}',
    ):
        with pytest.raises(ApplyConfirmationRequired) as excinfo:
            await suggest_co_evidential(db_session, subject_id=subj.id, mode=SuggestMode.APPLY)
        # No link was written on the guarded path.
        assert await get_co_evidential_group(db_session, _PID_A) == {_PID_A}

        # Re-run confirmed=True → it links.
        report = await suggest_co_evidential(
            db_session, subject_id=subj.id, mode=SuggestMode.APPLY, confirmed=True
        )

    assert excinfo.value.pair_count == 1
    assert excinfo.value.threshold == 0
    assert report.applied_pairs == 1


def test_legacy_config_key_migrates_with_warning(caplog) -> None:
    """The deprecated lint.co_evidential_candidate_threshold key is migrated + warned."""
    from particles.config import _migrate_legacy_keys

    raw: dict = {"lint": {"co_evidential_candidate_threshold": 0.81}}
    with caplog.at_level(logging.WARNING, logger="particles.config"):
        _migrate_legacy_keys(raw)

    assert raw["links_suggest"]["candidate_threshold"] == 0.81
    assert "co_evidential_candidate_threshold" not in raw["lint"]
    assert "deprecated" in caplog.text.lower()


def test_legacy_config_key_does_not_override_new(caplog) -> None:
    """When both keys are present, the new key wins and the old is dropped."""
    from particles.config import _migrate_legacy_keys

    raw: dict = {
        "lint": {"co_evidential_candidate_threshold": 0.81},
        "links_suggest": {"candidate_threshold": 0.95},
    }
    _migrate_legacy_keys(raw)
    assert raw["links_suggest"]["candidate_threshold"] == 0.95


@pytest.mark.asyncio
async def test_scope_particle_ids_bounds_judge_to_in_scope_pairs(
    db_session: AsyncSession,
) -> None:
    """LLM_JUDGE judges only pairs touching the scope set.

    Enumeration stays store-wide (all pairs remain in ``clusters`` /
    ``total_candidates``), so the audit can still report the store-wide total;
    only the verdict pass is bounded, keeping the ``--judge`` LLM cost scaled to
    the harvest.
    """
    subj = Subject(canonical_name="Acme Corp", asserted_by="test")
    await insert_subject(db_session, subj)
    pid_a = "00000000-0000-0000-0000-0000000000a1"
    pid_b = "00000000-0000-0000-0000-0000000000b1"
    pid_c = "00000000-0000-0000-0000-0000000000c1"
    emb_c = np.array([0.62, 0.78] + [0.0] * 382, dtype=np.float32)
    await insert_particle(db_session, _mk_particle("A.", pid_a, subj.id), embedding=_EMB_A.tolist())
    await insert_particle(db_session, _mk_particle("B.", pid_b, subj.id), embedding=_EMB_B.tolist())
    await insert_particle(db_session, _mk_particle("C.", pid_c, subj.id), embedding=emb_c.tolist())
    await db_session.commit()

    # Patch the LLM batch call to a no-op verdict map; judged candidates then
    # default to UNSURE, so a non-None verdict marks "was judged".
    with patch("particles.operations.links_suggest._judge_batch", return_value={}):
        report = await suggest_co_evidential(
            db_session,
            mode=SuggestMode.LLM_JUDGE,
            scope_particle_ids=frozenset({pid_a}),
        )

    # Enumeration is store-wide: all three pairs are candidates.
    assert report.total_candidates == 3
    candidates = {
        frozenset({c.particle_a, c.particle_b}): c
        for cluster in report.clusters
        for c in cluster.candidates
    }
    # The two pairs touching A are judged; (B, C) — no side in scope — is not.
    assert candidates[frozenset({pid_a, pid_b})].verdict is JudgeVerdictKind.UNSURE
    assert candidates[frozenset({pid_a, pid_c})].verdict is JudgeVerdictKind.UNSURE
    assert candidates[frozenset({pid_b, pid_c})].verdict is None
    assert report.judged_pairs == 2


@pytest.mark.asyncio
async def test_scope_judge_order_is_intra_scope_pairs_first(
    db_session: AsyncSession,
) -> None:
    """under a harvest scope, LLM_JUDGE consumes pairs in two tiers.

    The mixed (harvested ↔ store) pair is deliberately MORE similar than the
    (harvested ↔ harvested) pair, yet the intra-scope pair is judged in the
    first LLM batch and the mixed pairs in a later one — so a judge pass cut
    short (circuit breaker) has verdicted the pairs a memory audit is
    about before spending on coincidental cross-store neighbours.
    """
    subj = Subject(canonical_name="Acme Corp", asserted_by="test")
    await insert_subject(db_session, subj)
    pid_h1 = "00000000-0000-0000-0000-0000000000d1"
    pid_h2 = "00000000-0000-0000-0000-0000000000d2"
    pid_s = "00000000-0000-0000-0000-0000000000d3"
    # (h1, h2) ~0.99; (h1, s) ~0.99995 — the highest-similarity pair is mixed.
    emb_h1 = np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)
    emb_h2 = np.array([0.99, 0.141] + [0.0] * 382, dtype=np.float32)
    emb_s = np.array([0.9999, 0.0141] + [0.0] * 382, dtype=np.float32)
    await insert_particle(
        db_session, _mk_particle("H1.", pid_h1, subj.id), embedding=emb_h1.tolist()
    )
    await insert_particle(
        db_session, _mk_particle("H2.", pid_h2, subj.id), embedding=emb_h2.tolist()
    )
    await insert_particle(db_session, _mk_particle("S.", pid_s, subj.id), embedding=emb_s.tolist())
    await db_session.commit()

    batches: list[list[frozenset[str]]] = []

    def _record_batch(batch_candidates: list, content_for: dict) -> dict:  # type: ignore[type-arg]
        batches.append([frozenset({c.particle_a, c.particle_b}) for c in batch_candidates])
        return {}

    with patch("particles.operations.links_suggest._judge_batch", side_effect=_record_batch):
        report = await suggest_co_evidential(
            db_session,
            mode=SuggestMode.LLM_JUDGE,
            scope_particle_ids=frozenset({pid_h1, pid_h2}),
        )

    # First batch: the intra-scope pair alone, despite its lower similarity;
    # the mixed pairs ride a later batch. All three in-scope pairs are judged.
    assert batches[0] == [frozenset({pid_h1, pid_h2})]
    assert [p for batch in batches[1:] for p in batch] == [
        frozenset({pid_h1, pid_s}),
        frozenset({pid_h2, pid_s}),
    ]
    assert report.judged_pairs == 3


# --------------------------------------------------------------------------- #
# the store-wide fetch replaces the per-Subject N+1              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_store_wide_scan_issues_one_embedding_fetch(db_session: AsyncSession) -> None:
    """A store-wide scan fetches embeddings ONCE, not once per Subject.

    This loop used to issue one ``get_active_particles_with_embeddings(
    subject_id=…)`` per Subject. On the 2026-08-02 dogfood store that was 4,009
    queries costing 301.9 s of a 324 s call, and it was the single largest term
    in the 172 s curation-queue build. Guard the fix: if this
    assertion ever reads more than one call, the N+1 is back.
    """
    import particles.store.particle_store as ps

    await _seed_pair(db_session)
    # A second Subject, so a per-Subject implementation would fetch twice.
    other = Subject(canonical_name="Other Corp", asserted_by="test")
    await insert_subject(db_session, other)
    await insert_particle(
        db_session,
        _mk_particle("Other thing.", "00000000-0000-0000-0000-00000000000c", other.id),
        embedding=_EMB_ORTHO.tolist(),
    )
    await db_session.commit()

    real = ps.get_active_particles_with_embeddings
    calls: list[str | None] = []

    async def counting(session, *a, **kw):  # type: ignore[no-untyped-def]
        calls.append(kw.get("subject_id"))
        return await real(session, *a, **kw)

    with patch.object(ps, "get_active_particles_with_embeddings", counting):
        report = await suggest_co_evidential(db_session, mode=SuggestMode.REPORT)

    assert calls == [None], f"expected one store-wide fetch, got {len(calls)}: {calls}"
    # And the batched path still finds the pair the per-Subject one did.
    assert report.total_candidates == 1


@pytest.mark.asyncio
async def test_single_subject_scan_keeps_the_targeted_query(db_session: AsyncSession) -> None:
    """`--subject X` still issues one targeted join, not a store-wide scan.

    The batch path is for the store-wide scan; making a single-Subject lookup
    read every ACTIVE particle would trade one N+1 for a different regression.
    """
    import particles.store.particle_store as ps

    subj = await _seed_pair(db_session)
    real = ps.get_active_particles_with_embeddings
    calls: list[str | None] = []

    async def counting(session, *a, **kw):  # type: ignore[no-untyped-def]
        calls.append(kw.get("subject_id"))
        return await real(session, *a, **kw)

    with patch.object(ps, "get_active_particles_with_embeddings", counting):
        report = await suggest_co_evidential(db_session, subject_id=subj.id)

    assert calls == [subj.id]
    assert report.total_candidates == 1
