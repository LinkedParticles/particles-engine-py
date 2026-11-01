# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
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
from particles.store.particle_store import get_particle, get_particles_by_ids, insert_particle
from tests._write_probe import store_accepts_a_writer


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
    async def test_constitutive_superseded_is_demoted(
        self, db_session: AsyncSession, candidate_cache_mode: str
    ) -> None:
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
    async def test_a_sweep_demotion_is_recorded_in_the_event_log(
        self, db_session: AsyncSession
    ) -> None:
        """Security review 2026-10-24-2, P34: every automatic retirement leaves an event."""
        from particles.store.event_store import OperatorEventType, list_events

        superseded_id, superseding_id = await _setup_flagship(
            db_session, superseded_modality=AssertionModality.FALSIFIABLE
        )
        with patch("particles.operations.reconcile._has_contradiction_signal", _probe(True)):
            await reconcile_supersession(db_session)
        [event] = await list_events(db_session, event_type=OperatorEventType.DOCUMENT_SUPERSESSION)
        assert event.payload["retired_id"] == superseded_id
        assert event.payload["winner_id"] == superseding_id
        assert event.payload["route"] == "sweep"

    @pytest.mark.asyncio
    async def test_a_dry_run_records_no_event(self, db_session: AsyncSession) -> None:
        from particles.store.event_store import OperatorEventType, list_events

        await _setup_flagship(db_session, superseded_modality=AssertionModality.FALSIFIABLE)
        with patch("particles.operations.reconcile._has_contradiction_signal", _probe(True)):
            await reconcile_supersession(db_session, dry_run=True)
        assert not await list_events(db_session, event_type=OperatorEventType.DOCUMENT_SUPERSESSION)

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


@pytest.mark.asyncio
async def test_a_demotion_is_committed_before_the_next_probe(
    file_db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer in another process can write while the sweep probes.

    Demotions used to stay uncommitted until the sweep ended, so SQLite's
    write lock was held across every later probe and a concurrent writer
    failed with ``database is locked``.
    """
    import particles.operations.reconcile as reconcile_mod
    from particles.config import get_config

    session = file_db_session
    get_config().extraction.similarity_threshold = 0.5
    sup = _particle(content="the superseding claim", entry_id="sup-e", snapshot_id="s1")
    subs = [
        _particle(content=f"old claim {i}", entry_id="sub-e", snapshot_id="s2") for i in range(3)
    ]
    for p in [sup, *subs]:
        await insert_particle(session, p)
    await session.commit()

    monkeypatch.setattr(
        "particles.operations.version_guard.assert_store_schema_current",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        reconcile_mod,
        "iter_supersession_entry_pairs",
        AsyncMock(return_value=[("sup-e", "sub-e")]),
    )
    sims = [0.95, 0.90, 0.85]
    monkeypatch.setattr(
        reconcile_mod,
        "load_active_candidates",
        AsyncMock(
            return_value=[(sup, np.array([1.0], dtype=np.float32))]
            + [(p, np.array([s], dtype=np.float32)) for p, s in zip(subs, sims, strict=True)]
        ),
    )
    monkeypatch.setattr(reconcile_mod, "_cosine", lambda a, b: float(a[0] * b[0]))
    writable_during_probes: list[bool] = []

    async def probe(_a: str, _b: str) -> bool | None:
        writable_during_probes.append(store_accepts_a_writer())
        return True  # replaces: every pair demotes

    monkeypatch.setattr(reconcile_mod, "_has_contradiction_signal", probe)

    summary = await reconcile_supersession(session)

    assert summary["demoted"] == 3
    assert writable_during_probes == [True, True, True]
    stored = await get_particles_by_ids(session, [p.id for p in subs])
    assert {p.status for p in stored.values()} == {Status.PROVENANCE_STALE}


# ---------------------------------------------------------------------------
# The sweeps read the regime of the store they sweep (security review
# 2026-10-27-1, G6)
# ---------------------------------------------------------------------------


def _multi_by_its_own_entry(session: AsyncSession, how: str) -> None:
    from particles.config import get_config
    from particles.db import DEFAULT_STORE, store_of

    cfg = get_config()
    cfg.reconciliation.store_mode = "single"
    if how == "per_store":
        cfg.reconciliation.per_store = {DEFAULT_STORE: "multi"}
    else:
        cfg.mcp.write.enabled_stores = [DEFAULT_STORE]
    assert cfg.reconciliation_mode_for(store_of(session)) == "multi"


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["per_store", "write_enabled"])
async def test_document_sweep_uses_the_stores_own_regime(
    db_session: AsyncSession, how: str
) -> None:
    """On a store whose own regime is multi, the document sweep runs no single-trust ordering."""
    _multi_by_its_own_entry(db_session, how)
    with patch("particles.operations.version_guard.assert_store_schema_current", AsyncMock()):
        summary = await reconcile_supersession(db_session, dry_run=True)
    assert summary["single_trust_order"] is False, summary


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["per_store", "write_enabled"])
async def test_update_sweep_requires_attribution_on_a_store_whose_own_regime_is_multi(
    db_session: AsyncSession, how: str
) -> None:
    """The update sweep's gather asks for attribution on a store multi by its own entry."""
    from particles.operations import reconcile

    _multi_by_its_own_entry(db_session, how)
    gathered = AsyncMock(return_value=reconcile._UpdateGather())
    with patch.object(reconcile, "_update_candidates", gathered):
        await reconcile.count_update_candidates(db_session, None)
    assert gathered.call_args.kwargs["require_attribution"] is True


# ---------------------------------------------------------------------------
# The same regime, read from a second registered store (security review
# 2026-10-28-1, O72). The tests above make the default store multi on the
# in-memory default engine, so they hold whether a sweep reads the store it was
# handed or the default store's entry. These register a second store through
# the config loader, make only it multi, keep the global mode and the default
# store single, and drive each pass on a session ``session_scope("team")``
# opens. The replacement probe is patched; no model call.
# ---------------------------------------------------------------------------


def _two_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    from particles.config import reset_config

    extra = (
        "reconciliation:\n  store_mode: single\n  per_store:\n    team: multi\n"
        if how == "per_store"
        else "reconciliation:\n  store_mode: single\nmcp:\n  write:\n    enabled_stores: [team]\n"
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\n"
        "storage:\n"
        f"  database_url: sqlite+aiosqlite:///{tmp_path / 'default.db'}\n"
        f"  blob_dir: {tmp_path / 'blobs'}\n"
        "  stores:\n"
        f"    team: sqlite+aiosqlite:///{tmp_path / 'team.db'}\n" + extra,
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PARTICLES_BLOB_DIR", raising=False)
    reset_config()


def _probe_true() -> Callable[[str, str], Awaitable[bool | None]]:
    async def _fake(_a: str, _b: str) -> bool | None:
        return True

    return _fake


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["per_store", "write_enabled"])
async def test_the_document_sweep_retires_nothing_on_a_named_store_whose_own_regime_is_multi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """A replacement signal on a named multi store demotes nothing; the default store still does."""
    from particles.db import create_tables, session_scope, store_of

    _two_stores(tmp_path, monkeypatch, how)
    await create_tables("default")
    await create_tables("team")
    with patch("particles.operations.reconcile._has_contradiction_signal", _probe_true()):
        async with session_scope("team") as team:
            assert store_of(team) == "team"
            old_id, _ = await _setup_flagship(
                team, superseded_modality=AssertionModality.FALSIFIABLE
            )
            summary = await reconcile_supersession(team)
            loser = await get_particle(team, old_id)
        async with session_scope("default") as default:
            d_old, _ = await _setup_flagship(
                default, superseded_modality=AssertionModality.FALSIFIABLE
            )
            d_summary = await reconcile_supersession(default)
            d_loser = await get_particle(default, d_old)
    assert summary["single_trust_order"] is False, summary
    assert summary.get("demoted", 0) == 0, summary
    assert loser is not None and loser.status is Status.ACTIVE
    # The control: the single default store's sweep does demote.
    assert d_summary["single_trust_order"] is True
    assert d_loser is not None and d_loser.status is Status.PROVENANCE_STALE


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["per_store", "write_enabled"])
async def test_the_update_sweep_requires_attribution_on_a_named_multi_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """The update sweep's gather on a named store multi by its own entry asks for attribution."""
    from particles.db import create_tables, session_scope
    from particles.operations import reconcile

    _two_stores(tmp_path, monkeypatch, how)
    await create_tables("team")
    gathered = AsyncMock(return_value=reconcile._UpdateGather())
    with patch.object(reconcile, "_update_candidates", gathered):
        async with session_scope("team") as team:
            await reconcile.count_update_candidates(team, None)
    assert gathered.call_args.kwargs["require_attribution"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["per_store", "write_enabled"])
async def test_the_paraphrase_pass_requires_attribution_on_a_named_multi_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """The paraphrase pass on a named store multi by its own entry asks for attribution."""
    from datetime import UTC, datetime

    from particles.db import create_tables, session_scope
    from particles.operations import paraphrases

    _two_stores(tmp_path, monkeypatch, how)
    await create_tables("team")
    retired = type("R", (), {"id": "p-retired"})()
    winner = object()
    pages: list[list[tuple[Any, datetime]]] = [[(retired, datetime(2026, 1, 1, tzinfo=UTC))], []]

    async def _page(*_a: Any, **_k: Any) -> list[tuple[Any, datetime]]:
        return pages.pop(0)

    seen: dict[str, object] = {}

    async def _cands(*_a: Any, **kw: Any) -> list[Any]:
        seen.update(kw)
        return []

    with (
        patch.object(paraphrases, "get_update_retirements_after", _page),
        patch.object(
            paraphrases, "get_embeddings_by_ids", AsyncMock(return_value={"p-retired": [1.0]})
        ),
        patch.object(paraphrases, "winner_of", AsyncMock(return_value=winner)),
        patch.object(paraphrases, "_winner_usable", lambda _w: True),
        patch.object(paraphrases, "_candidates", _cands),
    ):
        async with session_scope("team") as team:
            await paraphrases.run_paraphrases(team, cursor=None, dry_run=True)
    assert seen.get("require_attribution") is True, seen


@pytest.mark.asyncio
async def test_the_extraction_conflict_fallback_reads_a_named_stores_own_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extraction's section 6.6 ladder on a named multi store runs with no single trust order."""
    from particles.db import create_tables, session_scope
    from particles.ingest import pipeline

    _two_stores(tmp_path, monkeypatch, "per_store")
    await create_tables("team")

    class _Stop(Exception):
        pass

    seen: dict[str, object] = {}

    def _ladder(*_a: Any, **kw: Any) -> None:
        seen.update(kw)
        raise _Stop

    a = _particle(content="x is 1", entry_id="e1", snapshot_id="s1")
    b = _particle(content="x is 2", entry_id="e2", snapshot_id="s2")
    with (
        patch.object(pipeline, "decide_ladder", _ladder),
        patch.object(pipeline, "needs_rung_inputs", lambda *a, **k: False),
    ):
        async with session_scope("team") as team:
            with pytest.raises(_Stop):
                await pipeline._resolve_conflict(team, b, a, "e2", "s2", probe=True)
    assert seen["single_trust_order"] is False, seen
