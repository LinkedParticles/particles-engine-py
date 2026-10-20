# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Planning an extraction's calls before it runs.

Pins ``particles/operations/extraction_reads.py`` against the real pipeline:
each test plans a snapshot's reads, then extracts it with a recording fake LLM
and checks the plan named the calls the extraction actually made: the
append-only tail with its context, no call for a chunk carried
forward, none past ``extraction.max_llm_calls_per_source``.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import Mutability
from particles.corpus.deposit import deposit_text_versioned
from particles.ingest.pipeline import extract_snapshot
from particles.operations.extraction_reads import ReadSummary, plan_reads
from particles.operations.spend_estimate import planned_extraction_estimate
from tests.test_append_base import fake_extraction_llm


def _turn(tag: str, words: int = 8) -> str:
    """One transcript turn, a single paragraph."""
    return f"User: {tag} " + " ".join(f"{tag.lower()}{i}" for i in range(words)) + "\n"


async def _deposit(
    session: AsyncSession, text: str, *, uri: str, source_type: str, mutability: Mutability
) -> tuple[str, str]:
    entry_id, snapshot_id, unchanged = await deposit_text_versioned(
        session,
        text=text,
        uri_r=uri,
        source_type=source_type,
        mutability=mutability,
        deposited_by="test",
    )
    assert not unchanged
    await session.commit()
    return entry_id, snapshot_id


async def _extract(session: AsyncSession, entry_id: str, snapshot_id: str) -> None:
    await extract_snapshot(session, entry_id, snapshot_id)
    await session.commit()


@pytest.mark.asyncio
async def test_an_append_only_snapshot_is_planned_as_its_tail(db_session: AsyncSession) -> None:
    first = _turn("Alpha") + "\n" + _turn("Beta")
    second = first + "\n" + _turn("Gamma")
    with fake_extraction_llm() as llm:
        entry_id, s1 = await _deposit(
            db_session,
            first,
            uri="claude-code://session/plan-tail",
            source_type="CONVERSATION",
            mutability=Mutability.APPEND_ONLY,
        )
        await _extract(db_session, entry_id, s1)
        _, s2 = await _deposit(
            db_session,
            second,
            uri="claude-code://session/plan-tail",
            source_type="CONVERSATION",
            mutability=Mutability.APPEND_ONLY,
        )

        [read] = await plan_reads(db_session, [(entry_id, s2)])
        llm.reset()
        await _extract(db_session, entry_id, s2)

    assert read.delta
    assert read.whole_chars == len(second)
    [call] = read.calls
    [(text, context)] = llm.calls
    assert call.chars == len(text) < len(second)
    assert call.context_chars == len(context or "") > 0
    # A transcript writes less per character than the documents the rate was
    # fitted on (audit.estimate_output_scale_by_source_type).
    assert call.density == get_config().audit.estimate_output_scale_by_source_type["CONVERSATION"]


@pytest.mark.asyncio
async def test_carried_chunks_and_the_call_cap_make_no_call(db_session: AsyncSession) -> None:
    cfg = get_config().extraction
    cfg.html_chunk_size = 1000
    paragraphs = [_turn(f"P{i}", words=100) for i in range(5)]  # one chunk each
    uri = "file:///tmp/plan-carry.md"
    with fake_extraction_llm() as llm:
        entry_id, s1 = await _deposit(
            db_session,
            "\n".join(paragraphs),
            uri=uri,
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        await _extract(db_session, entry_id, s1)
        # Change the first and last paragraphs: the three between are carried.
        edited = [_turn("Changed", words=100), *paragraphs[1:4], _turn("Also", words=100)]
        _, s2 = await _deposit(
            db_session,
            "\n".join(edited),
            uri=uri,
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        [uncapped] = await plan_reads(db_session, [(entry_id, s2)])
        cap = cfg.max_llm_calls_per_source
        cfg.max_llm_calls_per_source = 1
        [capped] = await plan_reads(db_session, [(entry_id, s2)])
        cfg.max_llm_calls_per_source = cap
        llm.reset()
        await _extract(db_session, entry_id, s2)

    assert not uncapped.delta
    # The unchanged middle is carried: the plan and the run make the same calls.
    assert uncapped.carried >= 1
    assert len(uncapped.calls) == len(llm.calls) > 1
    # Under a cap of one, every call past the first is planned as never read.
    assert len(capped.calls) == 1
    assert capped.capped == len(uncapped.calls) - 1
    assert capped.calls[0].density == 1.0


@pytest.mark.asyncio
async def test_a_missing_snapshot_plans_no_call(db_session: AsyncSession) -> None:
    [read] = await plan_reads(db_session, [("no-entry", "no-snapshot")])
    assert read.calls == ()
    summary = ReadSummary.of([read])
    assert planned_extraction_estimate(summary.calls, entries=0, batched=True).calls == 0


def test_the_summary_adds_up_reads() -> None:
    from particles.operations.audit import ExtractionCall
    from particles.operations.extraction_reads import SnapshotRead

    reads = [
        SnapshotRead("a", (ExtractionCall(100, 40),), delta=True, whole_chars=500, carried=1),
        SnapshotRead("b", (ExtractionCall(200), ExtractionCall(50)), whole_chars=250, capped=2),
    ]
    summary = ReadSummary.of(reads)
    assert (summary.snapshots, summary.delta, summary.carried, summary.capped) == (2, 1, 1, 2)
    assert (summary.whole_chars, summary.read_chars, len(summary.calls)) == (750, 350, 3)
