"""Tests for ``particles/operations/reconcile.py`` — the cross-entry
document-supersession reconcile sweep.

The sweep is the *activation* half of the fix: it runs §6.6 rung 1.5
cross-entry over already-extracted ACTIVE particles, demoting a superseded claim
the intra-entry extract path never reconciles. These tests pin the write path
(a real demotion to PROVENANCE_STALE / DOCUMENT_SUPERSEDED), the modality
independence (a superseded CONSTITUTIVE definition is reachable), the
conflict-gate (no replacement signal → keep both), idempotency, and the v1 gates
(disabled / multi-trust-order → no-op). The LLM replacement-signal probe is
mocked; embeddings are supplied directly so cosine similarity is deterministic.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    SourceType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.corpus.deposit import deposit_text
from particles.operations.reconcile import reconcile_supersession
from particles.store.particle_store import get_particle, insert_particle


def _adr(adr_id: str, *, supersedes: str | None = None, superseded_by: str | None = None) -> str:
    lines = ["---", "type: ADR", f'id: "{adr_id}"']
    if supersedes is not None:
        lines.append(f'supersedes: "{supersedes}"')
    if superseded_by is not None:
        lines.append(f'superseded_by: "{superseded_by}"')
    lines += ["---", f"# ADR {adr_id}", "", "Decision body."]
    return "\n".join(lines)


def _particle(
    *,
    content: str,
    entry_id: str,
    snapshot_id: str,
    modality: AssertionModality = AssertionModality.FALSIFIABLE,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_id,
                snapshot_id=snapshot_id,
            ),
        ],
        asserted_by="test",
        assertion_modality=modality,
    )


async def _setup_flagship(
    session: AsyncSession,
    *,
    superseded_modality: AssertionModality,
) -> tuple[str, str]:
    """Deposit 0116 (supersedes 0017) + 0017, insert one claim each with similar
    embeddings. Returns (superseded_particle_id, superseding_particle_id).
    """
    new_entry, new_snap = await deposit_text(
        session, _adr("0116", supersedes="0017"), source_type=SourceType.LOCAL_MARKDOWN
    )
    old_entry, old_snap = await deposit_text(
        session, _adr("0017", superseded_by="0116"), source_type=SourceType.LOCAL_MARKDOWN
    )

    superseded = _particle(
        content="effective_confidence is computed as calibrated_confidence × trust_weight × rank",
        entry_id=old_entry,
        snapshot_id=old_snap,
        modality=superseded_modality,
    )
    superseding = _particle(
        content="effective_confidence is computed as confidence.value × trust_weight × rank",
        entry_id=new_entry,
        snapshot_id=new_snap,
        modality=AssertionModality.FALSIFIABLE,
    )
    # Near-identical embeddings → cosine ≈ 1.0, well above the similarity floor.
    await insert_particle(session, superseded, embedding=[1.0, 0.0, 0.05])
    await insert_particle(session, superseding, embedding=[1.0, 0.0, 0.04])
    await session.commit()
    return superseded.id, superseding.id


def _probe(value: bool | None):  # type: ignore[no-untyped-def]
    async def _fake(_a: str, _b: str) -> bool | None:
        return value

    return _fake


class TestReconcileSupersessionSweep:
    @pytest.mark.asyncio
    async def test_constitutive_superseded_is_demoted(self, db_session: AsyncSession) -> None:
        # The flagship — the superseded claim is CONSTITUTIVE (non-truth-apt),
        # the exact case cap. 2 could not reach; the fix retires it.
        superseded_id, superseding_id = await _setup_flagship(
            db_session, superseded_modality=AssertionModality.CONSTITUTIVE
        )
        with patch("particles.operations.reconcile._has_contradiction_signal", _probe(True)):
            summary = await reconcile_supersession(db_session)

        assert summary["demoted"] == 1
        assert summary["scope_pairs"] >= 1
        loser = await get_particle(db_session, superseded_id)
        winner = await get_particle(db_session, superseding_id)
        assert loser is not None and winner is not None
        assert loser.status is Status.PROVENANCE_STALE
        assert loser.status_reason is StatusReason.DOCUMENT_SUPERSEDED
        assert winner.status is Status.ACTIVE  # the current claim stays ACTIVE

    @pytest.mark.asyncio
    async def test_falsifiable_superseded_is_demoted(self, db_session: AsyncSession) -> None:
        # Modality-independence cuts both ways: a FALSIFIABLE superseded claim is
        # retired too (this worked; guard it stays working).
        superseded_id, _ = await _setup_flagship(
            db_session, superseded_modality=AssertionModality.FALSIFIABLE
        )
        with patch("particles.operations.reconcile._has_contradiction_signal", _probe(True)):
            summary = await reconcile_supersession(db_session)
        assert summary["demoted"] == 1
        loser = await get_particle(db_session, superseded_id)
        assert loser is not None and loser.status is Status.PROVENANCE_STALE

    @pytest.mark.asyncio
    async def test_no_replacement_signal_keeps_both(self, db_session: AsyncSession) -> None:
        # Conflict-gate / cap. 2(c): no replacement signal → no demotion. A still
        # -true superseded-document claim is never blanket-demoted.
        superseded_id, _ = await _setup_flagship(
            db_session, superseded_modality=AssertionModality.CONSTITUTIVE
        )
        with patch("particles.operations.reconcile._has_contradiction_signal", _probe(False)):
            summary = await reconcile_supersession(db_session)
        assert summary["demoted"] == 0
        assert summary["candidate_pairs"] >= 1  # the pair was considered…
        loser = await get_particle(db_session, superseded_id)
        assert loser is not None and loser.status is Status.ACTIVE  # …but kept

    @pytest.mark.asyncio
    async def test_unavailable_probe_keeps_both(self, db_session: AsyncSession) -> None:
        # None (probe could not complete) is fail-open → keep both.
        superseded_id, _ = await _setup_flagship(
            db_session, superseded_modality=AssertionModality.CONSTITUTIVE
        )
        with patch("particles.operations.reconcile._has_contradiction_signal", _probe(None)):
            summary = await reconcile_supersession(db_session)
        assert summary["demoted"] == 0
        loser = await get_particle(db_session, superseded_id)
        assert loser is not None and loser.status is Status.ACTIVE

    @pytest.mark.asyncio
    async def test_dry_run_reports_without_mutating(self, db_session: AsyncSession) -> None:
        superseded_id, _ = await _setup_flagship(
            db_session, superseded_modality=AssertionModality.CONSTITUTIVE
        )
        with patch("particles.operations.reconcile._has_contradiction_signal", _probe(True)):
            summary = await reconcile_supersession(db_session, dry_run=True)
        assert summary["demoted"] == 1  # would demote
        assert summary["dry_run"] is True
        assert len(summary["demotions"]) == 1  # type: ignore[arg-type]
        loser = await get_particle(db_session, superseded_id)
        assert loser is not None and loser.status is Status.ACTIVE  # not mutated

    @pytest.mark.asyncio
    async def test_idempotent_second_run_demotes_nothing(self, db_session: AsyncSession) -> None:
        await _setup_flagship(db_session, superseded_modality=AssertionModality.CONSTITUTIVE)
        with patch("particles.operations.reconcile._has_contradiction_signal", _probe(True)):
            first = await reconcile_supersession(db_session)
            second = await reconcile_supersession(db_session)
        assert first["demoted"] == 1
        assert second["demoted"] == 0  # the loser is already off the ACTIVE surface

    @pytest.mark.asyncio
    async def test_no_supersession_edges_is_noop(self, db_session: AsyncSession) -> None:
        # Two unrelated ADRs with similar claims — no edge, so nothing demotes.
        e1, s1 = await deposit_text(db_session, _adr("0053"), source_type=SourceType.LOCAL_MARKDOWN)
        e2, s2 = await deposit_text(db_session, _adr("0113"), source_type=SourceType.LOCAL_MARKDOWN)
        await insert_particle(
            db_session,
            _particle(content="X is Y", entry_id=e1, snapshot_id=s1),
            embedding=[1.0, 0.0],
        )
        await insert_particle(
            db_session,
            _particle(content="X is Y", entry_id=e2, snapshot_id=s2),
            embedding=[1.0, 0.0],
        )
        await db_session.commit()
        with patch("particles.operations.reconcile._has_contradiction_signal", _probe(True)):
            summary = await reconcile_supersession(db_session)
        assert summary["scope_pairs"] == 0
        assert summary["demoted"] == 0

    @pytest.mark.asyncio
    async def test_disabled_is_noop(self, db_session: AsyncSession) -> None:
        # ``document_supersession.enabled = false`` reproduces today's behaviour
        # exactly. The autouse reset_config() fixture restores it next test.
        from particles.config import get_config

        get_config().document_supersession.enabled = False
        superseded_id, _ = await _setup_flagship(
            db_session, superseded_modality=AssertionModality.CONSTITUTIVE
        )
        summary = await reconcile_supersession(db_session)
        assert summary["enabled"] is False
        assert summary["demoted"] == 0
        loser = await get_particle(db_session, superseded_id)
        assert loser is not None and loser.status is Status.ACTIVE

    @pytest.mark.asyncio
    async def test_multi_trust_order_is_noop(self, db_session: AsyncSession) -> None:
        # rung 1.5 is single-trust-order only in v1.
        from particles.config import get_config

        get_config().reconciliation.store_mode = "multi"
        superseded_id, _ = await _setup_flagship(
            db_session, superseded_modality=AssertionModality.CONSTITUTIVE
        )
        summary = await reconcile_supersession(db_session)
        assert summary["single_trust_order"] is False
        assert summary["demoted"] == 0
        loser = await get_particle(db_session, superseded_id)
        assert loser is not None and loser.status is Status.ACTIVE
