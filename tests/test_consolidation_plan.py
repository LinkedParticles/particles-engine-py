# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``memory consolidate --dry-run``: gathers priced, nothing sent.

Pins ``particles/operations/consolidation_plan.py`` against a real in-memory
store: the dry run reports what each pass would extract or probe with a
list-price estimate, makes no LLM call (the Anthropic client is mocked through
the ``set_client`` seam and fails the test if touched), writes nothing, and
simulates ``consolidation.budget_usd`` with the run's own decision.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import anthropic
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import ProviderSelection, get_config
from particles.core.schema import (
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.operations.consolidation_plan import (
    MEASURED_LINE,
    ConsolidationPlan,
    PlannedPass,
    plan_consolidation,
    render_consolidation_plan,
)
from particles.store.particle_store import insert_particle

_HAIKU = "claude-haiku-4-5"


@pytest.fixture
def refusing_client(monkeypatch: pytest.MonkeyPatch) -> Generator[MagicMock, None, None]:
    """A key (so the LLM passes are available) and a client that fails if called."""
    from particles import llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = get_config().llm
    for purpose in ("extraction", "semantic_lint", "verification", "abstraction"):
        setattr(cfg, purpose, ProviderSelection(provider="anthropic", model=_HAIKU))
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = AssertionError("the dry run made an LLM call")
    client.messages.batches.create.side_effect = AssertionError("the dry run submitted a batch")
    llm.set_client(client)
    yield client
    llm.set_client(None)


async def _seed(session: AsyncSession) -> None:
    """Two PENDING snapshots with blobs and three ACTIVE beliefs."""
    from particles.corpus.deposit import save_blob, sha256
    from particles.corpus.store import CorpusEntryRow, SnapshotRow

    for i in range(2):
        content = (f"Fact {i}: the Pluto reclassification happened in 2006. " * 40).encode()
        digest = sha256(content)
        save_blob(content, digest)
        entry = CorpusEntry(
            entry_id=str(uuid.uuid4()),
            source_type="LOCAL_MARKDOWN",
            uri_r=f"file:///tmp/plan-{i}.md",
            deposited_by="test",
        )
        snap = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            captured_at=datetime.now(UTC),
            content_hash=digest,
            archive_path=str(digest),
            extraction_status=ExtractionStatus.PENDING,
            warc_record_type=WarcRecordType.RESPONSE,
        )
        session.add(CorpusEntryRow.from_model(entry))
        session.add(SnapshotRow.from_model(snap, entry.entry_id))
    for i in range(3):
        await insert_particle(
            session,
            Particle(
                content=f"Pluto is a dwarf planet, claim {i}.",
                confidence=Confidence(
                    value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1"
                    )
                ],
                asserted_by="general-extractor",
            ),
        )
    await session.commit()


async def _state(session: AsyncSession) -> dict[str, Any]:
    """Every row count and status the dry run must leave as it found it."""
    from particles.corpus.store import SnapshotRow
    from particles.store.event_store import OperatorEventRow
    from particles.store.particle_store import ParticleRow

    statuses = await session.execute(select(SnapshotRow.snapshot_id, SnapshotRow.extraction_status))
    particles = await session.execute(select(ParticleRow.id, ParticleRow.status))
    events = await session.scalar(select(func.count()).select_from(OperatorEventRow))
    return {
        "snapshots": sorted(statuses.all()),
        "particles": sorted(particles.all()),
        "events": events,
    }


def _by_name(plan: ConsolidationPlan) -> dict[str, PlannedPass]:
    return {p.name: p for p in plan.passes}


@pytest.mark.asyncio
async def test_the_dry_run_makes_no_call_and_writes_nothing(
    db_session: AsyncSession, refusing_client: MagicMock
) -> None:
    await _seed(db_session)
    before = await _state(db_session)

    plan = await plan_consolidation(db_session)

    assert refusing_client.messages.create.call_count == 0
    assert refusing_client.messages.batches.create.call_count == 0
    assert not db_session.new and not db_session.dirty and not db_session.deleted
    await db_session.rollback()
    assert await _state(db_session) == before

    passes = _by_name(plan)
    # The passes in the cycle's own order, each with a position.
    assert [p.name for p in plan.passes][:3] == ["refresh", "extract", "reconcile"]
    extract = passes["extract"]
    assert extract.action == "run"
    assert extract.detail.startswith("would extract 2 of 2 pending snapshots")
    assert extract.llm_priced and extract.calls >= 2
    assert extract.estimate_usd is not None and extract.estimate_usd > 0
    assert passes["census"].detail.endswith("would probe 0 (audit.max_contradiction_probes = 50)")
    assert not passes["disclose"].llm_priced
    assert plan.total_calls >= extract.calls
    assert plan.total_usd is not None


@pytest.mark.asyncio
async def test_the_report_says_how_to_read_the_estimate(
    db_session: AsyncSession, refusing_client: MagicMock
) -> None:
    await _seed(db_session)
    rendered = render_consolidation_plan(await plan_consolidation(db_session))

    assert "nothing written, no LLM call made" in rendered
    assert "list-price estimates over estimated tokens" in rendered
    # Names the figure the metered run prints afterwards, so the two compare.
    assert MEASURED_LINE == "LLM usage: … ≈ $X at list price."
    assert MEASURED_LINE in rendered
    assert "CONSOLIDATION_RUN" in rendered
    assert "Estimated total: ~" in rendered


@pytest.mark.asyncio
async def test_structural_only_plans_no_llm_pass(
    db_session: AsyncSession, refusing_client: MagicMock
) -> None:
    await _seed(db_session)
    plan = await plan_consolidation(db_session, structural_only=True)

    passes = _by_name(plan)
    assert plan.semantic_degraded_reason == "--structural-only"
    assert passes["extract"].action == "skip"
    assert "2 snapshots pending" in passes["extract"].detail
    assert passes["reconcile"].action == "skip"
    assert passes["census"].calls == 0
    assert plan.total_calls == 0
    assert plan.total_usd == 0.0


@pytest.mark.asyncio
async def test_the_budget_is_simulated_with_the_runs_own_decision(
    db_session: AsyncSession, refusing_client: MagicMock
) -> None:
    await _seed(db_session)
    get_config().consolidation.budget_usd = 0.0001
    plan = await plan_consolidation(db_session)

    extract = _by_name(plan)["extract"]
    assert extract.budget_skip is not None
    assert extract.budget_skip.startswith("would be skipped, US$0.00 of US$0.00 spent before it")
    # A skipped pass spends nothing, so it is out of the total: every priced
    # pass here would pass the budget, so nothing is left in it.
    assert extract.calls > 0
    assert plan.total_calls == 0
    rendered = render_consolidation_plan(plan)
    assert "Budget: consolidation.budget_usd = US$0.00" in rendered
    assert "the run would skip extract" in rendered


@pytest.mark.asyncio
async def test_a_generous_budget_skips_nothing(
    db_session: AsyncSession, refusing_client: MagicMock
) -> None:
    await _seed(db_session)
    get_config().consolidation.budget_usd = 1000.0
    plan = await plan_consolidation(db_session)

    assert all(p.budget_skip is None for p in plan.passes)
    assert "every pass fits at these estimates" in render_consolidation_plan(plan)


@pytest.mark.asyncio
async def test_the_projection_verdict_comes_from_the_surface(
    db_session: AsyncSession, refusing_client: MagicMock
) -> None:
    plan = await plan_consolidation(
        db_session, projection_skip_reason="agent_memory.projection.enabled is false"
    )
    projection = _by_name(plan)["projection"]
    assert projection.action == "skip"
    assert projection.detail == "agent_memory.projection.enabled is false"
