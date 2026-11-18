# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Day-marked transcripts in the append-only read, and the re-read (§6).

* The first marked snapshot of a transcript extracted unmarked is read as a
  delta: the base is found on day-mark-reduced text, only the new turns are
  read, and the snapshot's offset is in its own bytes.
* A delta chunk whose shown text starts after a mark carries that mark's day
  as its REFERENCE DATE.
* ``plan_reread`` / ``run_reread`` replay a transcript from its first marked
  snapshot through the newest rendering: claims first cited before it stay,
  the rest are retired and read again, each new claim citing the snapshot
  that first held its passage, and the offsets written are the stored blobs'.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles import embeddings as ep
from particles.config import get_config
from particles.core.schema import UncertaintyNature
from particles.core.status import Status, StatusReason
from particles.extraction.general import CandidateParticle
from particles.operations.session_reread import (
    NOT_YET_DEPOSITED,
    REFUSE_NO_MARKS,
    REFUSE_PENDING,
    plan_reread,
    run_reread,
)
from particles.store.particle_store import get_particles_for_entry
from tests.test_append_base import _deposit, _extract, _extracted_through, _vector

SEP = "·"


class Recorder:
    """Records each extraction call's text and keywords; one claim per call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(
        self, text: str, *args: Any, **kwargs: Any
    ) -> tuple[list[CandidateParticle], list[str], bool]:
        self.calls.append((text, kwargs))
        claim = CandidateParticle(
            content=f"Claim {len(self.calls)}: {text.strip().splitlines()[-1][:60]}",
            confidence_value=0.8,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
        )
        return [claim], [], False

    def reference_days(self) -> list[date | None]:
        out: list[date | None] = []
        for _, kwargs in self.calls:
            moment = kwargs.get("reference_published_at")
            out.append(moment.date() if isinstance(moment, datetime) else None)
        return out


@pytest.fixture
def llm() -> Iterator[Recorder]:
    fake = Recorder()
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


@contextmanager
def _context_chars(chars: int) -> Iterator[None]:
    cfg = get_config().extraction
    before = cfg.append_context_chars
    cfg.append_context_chars = chars
    try:
        yield
    finally:
        cfg.append_context_chars = before


def _turn(speaker: str, tag: str, day: str | None = None) -> str:
    heading = f"## {speaker}" if day is None else f"## {speaker} {SEP} {day}"
    words = " ".join(f"{tag.lower()}{i}" for i in range(8))
    return f"{heading}\n\n{tag} {words}\n"


HEAD = "# Claude Code session reread\n"
DAY_ONE = HEAD + "\n" + _turn("User", "Alpha") + "\n" + _turn("Assistant", "Beta")
# Version-1 text of a session that went on to a second day: no mark.
UNMARKED = DAY_ONE + "\n" + _turn("User", "Gamma")
# Its version-2 rendering: the same turns, the day change marked.
MARKED = DAY_ONE + "\n" + _turn("User", "Gamma", "2026-09-02")
MARKED_ON = MARKED + "\n" + _turn("Assistant", "Delta", "2026-09-03")


class TestMarkedDelta:
    @pytest.mark.asyncio
    async def test_the_first_marked_snapshot_reads_only_its_new_turns(
        self, db_session: AsyncSession, llm: Recorder
    ) -> None:
        entry_id, s1 = await _deposit(db_session, UNMARKED)
        await _extract(db_session, entry_id, s1)
        _, s2 = await _deposit(db_session, MARKED_ON)
        llm.calls.clear()

        _, outcome = await _extract(db_session, entry_id, s2)

        [(text, _kwargs)] = llm.calls
        assert "Delta" in text and "Gamma" not in text and "Alpha" not in text
        assert outcome.append_fallback == 0 and outcome.append_marks_reduced == 1
        assert await _extracted_through(db_session, s2) == len(MARKED_ON.encode())

    @pytest.mark.asyncio
    async def test_a_re_deposit_with_only_marks_added_makes_no_call(
        self, db_session: AsyncSession, llm: Recorder
    ) -> None:
        entry_id, s1 = await _deposit(db_session, UNMARKED)
        await _extract(db_session, entry_id, s1)
        _, s2 = await _deposit(db_session, MARKED)
        llm.calls.clear()

        written, outcome = await _extract(db_session, entry_id, s2)

        assert llm.calls == [] and written == []
        assert outcome.append_fallback == 0 and outcome.append_marks_reduced == 1

    @pytest.mark.asyncio
    async def test_a_chunk_shown_after_a_mark_takes_its_day(
        self, db_session: AsyncSession, llm: Recorder
    ) -> None:
        entry_id, s1 = await _deposit(db_session, MARKED)
        await _extract(db_session, entry_id, s1)
        _, s2 = await _deposit(db_session, MARKED_ON)
        llm.calls.clear()

        with _context_chars(40):
            await _extract(db_session, entry_id, s2)

        assert llm.reference_days() == [date(2026, 9, 2)]


class TestReread:
    async def _two_snapshots(self, session: AsyncSession) -> tuple[str, str, str]:
        entry_id, s1 = await _deposit(session, DAY_ONE)
        await _extract(session, entry_id, s1)
        _, s2 = await _deposit(session, UNMARKED)
        await _extract(session, entry_id, s2)
        return entry_id, s1, s2

    @pytest.mark.asyncio
    async def test_plan_from_the_rendering_before_it_is_deposited(
        self, db_session: AsyncSession, llm: Recorder
    ) -> None:
        entry_id, s1, s2 = await self._two_snapshots(db_session)
        claims = {p.id: p for p in await get_particles_for_entry(db_session, entry_id)}

        plan = await plan_reread(db_session, entry_id, MARKED_ON.encode())

        assert plan.refusal is None
        assert plan.steps == [s2, NOT_YET_DEPOSITED]
        assert plan.first_base == s1
        assert set(plan.renderings) == {s2}
        assert f"{SEP} 2026-09-02".encode() in plan.renderings[s2]
        [retired] = plan.retire
        assert retired.provenance[0].snapshot_id == s2 and retired.id in claims
        assert plan.read_chars > 0 and plan.retire_with_state == 0
        with pytest.raises(ValueError):
            await run_reread(db_session, plan)

    @pytest.mark.asyncio
    async def test_reread_replays_from_the_first_marked_snapshot(
        self, db_session: AsyncSession, llm: Recorder
    ) -> None:
        entry_id, s1, s2 = await self._two_snapshots(db_session)
        before = {p.id: p for p in await get_particles_for_entry(db_session, entry_id)}
        _, s3 = await _deposit(db_session, MARKED_ON)
        plan = await plan_reread(db_session, entry_id)
        assert plan.steps == [s2, s3]
        llm.calls.clear()

        with _context_chars(40):
            retired = await run_reread(db_session, plan)

        assert retired == 1
        texts = [text for text, _ in llm.calls]
        assert len(texts) == 2
        assert f"## User {SEP} 2026-09-02" in texts[0] and "Gamma" in texts[0]
        assert "Delta" in texts[1]
        assert llm.reference_days()[1] == date(2026, 9, 2)

        after = {p.id: p for p in await get_particles_for_entry(db_session, entry_id)}
        old_s1 = [p for p in before.values() if p.provenance[0].snapshot_id == s1]
        old_s2 = [p for p in before.values() if p.provenance[0].snapshot_id == s2]
        assert all(after[p.id].status is Status.ACTIVE for p in old_s1)
        assert all(after[p.id].status is Status.SUPERSEDED for p in old_s2)
        assert all(after[p.id].status_reason is StatusReason.SUPERSEDED_BY_REINDEX for p in old_s2)
        fresh = [p for p in after.values() if p.id not in before]
        assert sorted(p.provenance[0].snapshot_id for p in fresh) == sorted([s2, s3])
        # The offsets are the stored blobs' own, so an ordinary read finds them.
        assert await _extracted_through(db_session, s2) == len(UNMARKED.encode())
        assert await _extracted_through(db_session, s3) == len(MARKED_ON.encode())

    @pytest.mark.asyncio
    async def test_a_transcript_with_no_mark_is_refused(
        self, db_session: AsyncSession, llm: Recorder
    ) -> None:
        entry_id, _, _ = await self._two_snapshots(db_session)
        plan = await plan_reread(db_session, entry_id, UNMARKED.encode())
        assert plan.refusal == REFUSE_NO_MARKS and plan.steps == []

    @pytest.mark.asyncio
    async def test_an_unextracted_snapshot_before_the_mark_is_refused(
        self, db_session: AsyncSession, llm: Recorder
    ) -> None:
        entry_id, _ = await _deposit(db_session, DAY_ONE)
        _, _ = await _deposit(db_session, UNMARKED)
        plan = await plan_reread(db_session, entry_id, MARKED_ON.encode())
        assert plan.refusal == REFUSE_PENDING
