# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Agent-side extraction submissions through the real pipeline (§3-§6).

Each test deposits a transcript as an ``APPEND_ONLY`` entry, records an ask the
way the Stop hook does, and submits candidates the way the ``extraction_submit``
tool does. The embedding model returns a distinct vector per text, so no
candidate pairs with another and no contradiction probe runs; the background
extractor's LLM is a fake that records what it was sent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles import embeddings as ep
from particles.config import get_config
from particles.core.schema import ExtractionStatus, Mutability, UncertaintyNature
from particles.corpus.deposit import deposit_text_versioned
from particles.corpus.store import SnapshotRow
from particles.extraction.general import CandidateParticle
from particles.ingest import agent_extraction
from particles.ingest.agent_extraction import SubmissionRefused, apply_submission, record_ask
from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import get_particles_for_entry

URI = "claude-code://session/adr-0303"
FIRST = (
    "## User\n\nThe staging database runs Postgres 17.\n\n"
    "## Assistant\n\nThe deploy pipeline runs nightly at 02:00 UTC.\n"
)
LATER = "\n## User\n\nThe on-call rotation changes every Monday.\n"


def _vector(text: str) -> list[float]:
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest() * 4, dtype=np.uint8)
    vec = raw.astype(np.float32) - 127.5
    return list(vec / np.linalg.norm(vec))


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(
        self, text: str, *args: Any, **kwargs: Any
    ) -> tuple[list[CandidateParticle], list[str], bool]:
        self.calls.append(text)
        claim = CandidateParticle(
            content=f"Background claim {len(self.calls)}.",
            confidence_value=0.8,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
        )
        return [claim], [], False


@pytest.fixture
def llm() -> Iterator[FakeLLM]:
    fake = FakeLLM()
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, **_: np.array([_vector(t) for t in texts], dtype=np.float32)
    )
    original = ep._embedding_model
    ep.set_embedding_model(model)
    with (
        patch("particles.extraction.general._call_llm", fake),
        patch("particles.extraction.incremental._call_llm", fake),
    ):
        yield fake
    ep.set_embedding_model(original)


@pytest.fixture
def mode_on() -> None:
    cfg = get_config().claude_code.agent_extraction
    cfg.mode = "on"
    cfg.min_claims_per_kchar = 1.0


def _candidate(content: str, evidence: str) -> dict[str, Any]:
    return {
        "content": content,
        "subjects": [],
        "confidence_value": 0.9,
        "uncertainty_nature": "EPISTEMIC",
        "evidence": evidence,
    }


GOOD = [
    _candidate("The staging database runs Postgres 17.", "staging database runs Postgres 17"),
    _candidate("The deploy pipeline runs nightly at 02:00 UTC.", "runs nightly at 02:00 UTC"),
]


async def _deposit(session: AsyncSession, text: str) -> tuple[str, str]:
    entry_id, snapshot_id, unchanged = await deposit_text_versioned(
        session,
        text=text,
        uri_r=URI,
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
        deposited_by="test",
    )
    assert not unchanged
    await session.commit()
    return entry_id, snapshot_id


async def _ask(
    session: AsyncSession, entry_id: str, snapshot_id: str, start: int, end: int, **kw: Any
) -> str:
    event = await record_ask(
        session,
        entry_id=entry_id,
        snapshot_id=snapshot_id,
        session_id="adr-0303",
        store=kw.get("store", "default"),
        start=start,
        end=end,
        provider_model="claude-code:claude-opus-5-5",
        after_uuid="a1",
        mode=get_config().claude_code.agent_extraction.mode,
    )
    await session.commit()
    return event.event_id


async def _row(session: AsyncSession, snapshot_id: str) -> SnapshotRow:
    return (
        await session.execute(
            select(SnapshotRow)
            .where(SnapshotRow.snapshot_id == snapshot_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


class TestModes:
    async def test_shadow_writes_and_marks_nothing(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        get_config().claude_code.agent_extraction.mode = "shadow"
        entry_id, snapshot_id = await _deposit(db_session, FIRST)
        ask_id = await _ask(db_session, entry_id, snapshot_id, 0, len(FIRST.encode()))
        out = await apply_submission(db_session, ask_id=ask_id, store="default", candidates=GOOD)
        assert (out.mode, out.accepted, out.written) == ("shadow", 2, 0)
        assert await get_particles_for_entry(db_session, entry_id) == []
        row = await _row(db_session, snapshot_id)
        assert row.extraction_status == ExtractionStatus.PENDING.value
        events = await list_events(
            db_session, event_type=OperatorEventType.AGENT_EXTRACTION_SUBMITTED
        )
        assert events[0].payload is not None and events[0].payload["accepted"] == 2

    async def test_on_writes_agent_particles_and_closes_the_span(
        self, db_session: AsyncSession, llm: FakeLLM, mode_on: None
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, FIRST)
        end = len(FIRST.encode())
        ask_id = await _ask(db_session, entry_id, snapshot_id, 0, end)
        bad = [*GOOD, _candidate("Staging runs Postgres 16.", "Postgres 16")]
        out = await apply_submission(db_session, ask_id=ask_id, store="default", candidates=bad)
        assert (out.accepted, out.dropped_no_evidence, out.written) == (2, 1, 2)
        assert out.closed_span
        particles = await get_particles_for_entry(db_session, entry_id)
        assert {p.extractor_ref.name for p in particles if p.extractor_ref} == {"claude-code-agent"}
        assert {p.extraction_provider_model for p in particles} == {"claude-code:claude-opus-5-5"}
        assert all(p.provenance[0].location.startswith("bytes=") for p in particles)
        row = await _row(db_session, snapshot_id)
        assert row.extraction_status == ExtractionStatus.COMPLETE.value
        assert row.extracted_through == end
        assert llm.calls == []


class TestRefusals:
    async def test_unknown_expired_repeated_and_foreign_store(
        self, db_session: AsyncSession, llm: FakeLLM, mode_on: None
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, FIRST)
        with pytest.raises(SubmissionRefused, match="no agent-extraction ask"):
            await apply_submission(db_session, ask_id="nope", store="default", candidates=GOOD)
        ask_id = await _ask(db_session, entry_id, snapshot_id, 0, len(FIRST.encode()))
        with pytest.raises(SubmissionRefused, match="not made against store"):
            await apply_submission(db_session, ask_id=ask_id, store="other", candidates=GOOD)
        await apply_submission(db_session, ask_id=ask_id, store="default", candidates=GOOD)
        with pytest.raises(SubmissionRefused, match="already answered"):
            await apply_submission(db_session, ask_id=ask_id, store="default", candidates=GOOD)

    async def test_expired(
        self, db_session: AsyncSession, llm: FakeLLM, mode_on: None, monkeypatch: Any
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, FIRST)
        ask_id = await _ask(db_session, entry_id, snapshot_id, 0, len(FIRST.encode()))
        later = datetime.now(UTC) + timedelta(hours=1)

        class _Clock(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
                return later

        monkeypatch.setattr(agent_extraction, "datetime", _Clock)
        with pytest.raises(SubmissionRefused, match="expired"):
            await apply_submission(db_session, ask_id=ask_id, store="default", candidates=GOOD)


class TestCoverage:
    async def test_under_covered_with_a_background_extractor_reopens_the_snapshot(
        self, db_session: AsyncSession, llm: FakeLLM, mode_on: None, monkeypatch: Any
    ) -> None:
        get_config().claude_code.agent_extraction.min_claims_per_kchar = 50.0
        monkeypatch.setattr(agent_extraction, "purpose_servable", lambda purpose: True)
        entry_id, snapshot_id = await _deposit(db_session, FIRST)
        ask_id = await _ask(db_session, entry_id, snapshot_id, 0, len(FIRST.encode()))
        out = await apply_submission(db_session, ask_id=ask_id, store="default", candidates=GOOD)
        assert not out.covered and not out.closed_span and out.written == 2
        row = await _row(db_session, snapshot_id)
        assert row.extraction_status == ExtractionStatus.PENDING.value
        assert row.extracted_through is None

    async def test_under_covered_without_one_closes_anyway(
        self, db_session: AsyncSession, llm: FakeLLM, mode_on: None
    ) -> None:
        get_config().claude_code.agent_extraction.min_claims_per_kchar = 50.0
        entry_id, snapshot_id = await _deposit(db_session, FIRST)
        ask_id = await _ask(db_session, entry_id, snapshot_id, 0, len(FIRST.encode()))
        out = await apply_submission(db_session, ask_id=ask_id, store="default", candidates=GOOD)
        assert not out.covered and out.closed_span
        assert any("closed anyway" in n for n in out.notes)


class TestNoDoubleExtraction:
    async def test_earlier_pending_snapshots_are_closed(
        self, db_session: AsyncSession, llm: FakeLLM, mode_on: None
    ) -> None:
        entry_id, first_id = await _deposit(db_session, FIRST)
        _, second_id = await _deposit(db_session, FIRST + LATER)
        ask_id = await _ask(db_session, entry_id, second_id, 0, len((FIRST + LATER).encode()))
        candidates = [
            *GOOD,
            _candidate("The on-call rotation changes every Monday.", "changes every Monday"),
        ]
        await apply_submission(db_session, ask_id=ask_id, store="default", candidates=candidates)
        first = await _row(db_session, first_id)
        assert first.extraction_status == ExtractionStatus.COMPLETE.value

    async def test_the_background_pass_reads_only_what_the_agent_did_not_cover(
        self, db_session: AsyncSession, llm: FakeLLM, mode_on: None
    ) -> None:
        entry_id, first_id = await _deposit(db_session, FIRST)
        ask_id = await _ask(db_session, entry_id, first_id, 0, len(FIRST.encode()))
        await apply_submission(db_session, ask_id=ask_id, store="default", candidates=GOOD)
        _, second_id = await _deposit(db_session, FIRST + LATER)
        outcome = SnapshotOutcome()
        await extract_snapshot(db_session, entry_id, second_id, outcome_out=outcome)
        await db_session.commit()
        assert outcome.append_fallback == 0
        assert len(llm.calls) == 1
        assert "on-call rotation" in llm.calls[0]
        assert "Postgres 17" not in llm.calls[0]

    async def test_a_background_run_that_got_there_first_owns_its_text(
        self, db_session: AsyncSession, llm: FakeLLM, mode_on: None
    ) -> None:
        entry_id, first_id = await _deposit(db_session, FIRST)
        await extract_snapshot(db_session, entry_id, first_id)
        await db_session.commit()
        _, second_id = await _deposit(db_session, FIRST + LATER)
        ask_id = await _ask(db_session, entry_id, second_id, 0, len((FIRST + LATER).encode()))
        candidates = [
            *GOOD,
            _candidate("The on-call rotation changes every Monday.", "changes every Monday"),
        ]
        out = await apply_submission(
            db_session, ask_id=ask_id, store="default", candidates=candidates
        )
        assert out.dropped_already_extracted == 2
        assert out.accepted == 1


class TestCalibration:
    async def test_a_session_model_reads_the_api_fit_of_the_same_model(
        self, db_session: AsyncSession, llm: FakeLLM, mode_on: None
    ) -> None:
        from particles.core.schema import ExtractorCalibration
        from particles.core.scoring.confidence import CalibrationSource
        from particles.extraction.agent import EXTRACTOR_ID, EXTRACTOR_VERSION
        from particles.store.extractor_store import upsert_calibration

        await upsert_calibration(
            db_session,
            EXTRACTOR_ID,
            ExtractorCalibration(
                temperature=1.0,
                transform="isotonic",
                fitted_at=datetime(2026, 10, 9, tzinfo=UTC),
                benchmark_suite_id="prose-calibration-001",
                sample_size=200,
                calibration_error_before=0.2,
                calibration_error_after=0.05,
                provider_model="anthropic:claude-opus-5-5",
                extractor_version=EXTRACTOR_VERSION,
                breakpoints=[(0.5, 0.4), (0.95, 0.7)],
            ),
        )
        await db_session.commit()
        entry_id, snapshot_id = await _deposit(db_session, FIRST)
        ask_id = await _ask(db_session, entry_id, snapshot_id, 0, len(FIRST.encode()))
        await apply_submission(db_session, ask_id=ask_id, store="default", candidates=GOOD)
        particles = await get_particles_for_entry(db_session, entry_id)
        assert {p.confidence.calibration_source for p in particles} == {
            CalibrationSource.CALIBRATED_BENCHMARK
        }
        assert {p.confidence.calibration_method for p in particles} == {"isotonic_regression"}
        assert all(p.confidence.value < 0.9 for p in particles)
