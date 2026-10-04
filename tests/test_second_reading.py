# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Extraction reads a probe-confirmed contradiction a second time (§2, §4).

The first contradiction probe is scripted (``bow``, ``scripted_probe``); the
second reading is patched at ``particles.ingest.second_reading`` so each test
decides its verdict. The autouse ``second_reading_confirms`` default is
overridden here.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import tomllib
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.contradiction_disclosure import READING_KEY, READING_OBSERVED, READING_STANDING
from particles.core.schema import Mutability, Particle, SourceType
from particles.core.status import Status, StatusReason
from particles.ingest.pair_selection import (
    PairReading,
    PairSelection,
    apply_readings,
    plan_readings,
)
from particles.ingest.second_reading import (
    ClaimContext,
    ProbeVerdict,
    SourceKind,
    in_flight_context,
    reading_for,
)
from tests._upstream import REPO_ROOT, upstream_only
from tests.test_benchmark_rot import bow_encoder  # noqa: F401 — a fixture
from tests.test_observer_gate import scripted_probe  # noqa: F401 — a fixture
from tests.test_update_supersession import (  # noqa: F401 — ``bow`` is a fixture
    BOSTON,
    DENVER,
    T0,
    _by_content,
    _OneClaim,
    bow,
)

Reader = Callable[[ClaimContext, ClaimContext], Coroutine[Any, Any, ProbeVerdict | None]]

_YES = ProbeVerdict(usable=True, contradicts=True, description="the cities differ")
_NO = ProbeVerdict(usable=True, contradicts=False, description="a later session moved")


def _p(content: str, entry: str = "e1") -> Particle:
    from particles.core.schema import (
        Confidence,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource

    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry)],
        asserted_by="test",
    )


class _Reads:
    """A second reading returning ``verdict``, recording each pair it is shown."""

    def __init__(self, verdict: ProbeVerdict | None) -> None:
        self.verdict = verdict
        self.pairs: list[tuple[ClaimContext, ClaimContext]] = []

    async def __call__(self, a: ClaimContext, b: ClaimContext) -> ProbeVerdict | None:
        self.pairs.append((a, b))
        return self.verdict


def _reading(monkeypatch: pytest.MonkeyPatch, verdict: ProbeVerdict | None) -> _Reads:
    from particles.ingest import second_reading

    reads = _Reads(verdict)
    monkeypatch.setattr(second_reading, "_llm_verify_contradiction", reads)
    return reads


# ---------------------------------------------------------------------------
# The decision (pure)
# ---------------------------------------------------------------------------


class TestPlanAndApply:
    def test_reads_a_confirmed_primary_and_every_declined_pair_never_the_extras(self) -> None:
        primary, extra, declined = _p("a"), _p("b"), _p("c")
        selection = PairSelection(
            primary=primary, signal=True, extras=(extra,), declined=(declined,)
        )
        assert plan_readings(selection) == [primary, declined]

    def test_an_unconfirmed_primary_is_not_read(self) -> None:
        assert plan_readings(PairSelection(primary=_p("a"), signal=False)) == []

    def test_confirmed_keeps_the_signal_and_records_the_instruction(self) -> None:
        primary = _p("a")
        out, tally = apply_readings(
            PairSelection(primary=primary, signal=True),
            {primary.id: PairReading(confirmed=True, reading=READING_OBSERVED)},
            same_entry=set(),
        )
        assert (out.signal, out.reading) == (True, READING_OBSERVED)
        assert (tally.read, tally.cleared, tally.unread) == (1, 0, 0)

    def test_not_confirmed_is_the_probes_no(self) -> None:
        primary, declined = _p("a"), _p("b")
        out, tally = apply_readings(
            PairSelection(primary=primary, signal=True, declined=(declined,)),
            {
                primary.id: PairReading(confirmed=False, reading=READING_STANDING),
                declined.id: PairReading(confirmed=False, reading=READING_STANDING),
            },
            same_entry=set(),
        )
        assert out.primary is primary and out.signal is False and out.reading is None
        assert out.declined == ()
        assert (tally.read, tally.cleared, tally.unread) == (2, 2, 0)

    def test_a_failed_reading_fails_closed_for_one_entry_and_open_across_entries(self) -> None:
        same, cross = _p("a"), _p("b", entry="e2")
        failed = PairReading(confirmed=None, reading=READING_OBSERVED)
        kept, t1 = apply_readings(
            PairSelection(primary=same, signal=True), {same.id: failed}, same_entry={same.id}
        )
        assert (kept.signal, kept.reading) == (True, None)  # kept, unstamped
        dropped, t2 = apply_readings(
            PairSelection(primary=cross, signal=True, declined=(same,)),
            {cross.id: failed, same.id: failed},
            same_entry={same.id},
        )
        assert dropped.signal is False
        assert dropped.declined == (same,)
        assert (t1.unread, t2.unread) == (1, 2)


class TestInFlightContext:
    def test_an_append_only_record_is_read_as_an_observation_at_its_capture_time(self) -> None:
        when = T0 + timedelta(hours=5, minutes=7)
        ctx = in_flight_context(
            DENVER,
            text="day 1: " + BOSTON + "\n\nday 9: " + DENVER,
            chunk_hash=None,
            partner=BOSTON,
            source="session-1.jsonl",
            source_type=SourceType.CONVERSATION,
            mutability=Mutability.APPEND_ONLY,
            when=when,
        )
        assert ctx.kind is SourceKind.RECORD
        assert ctx.observed == "2026-06-01 05:07 UTC"
        assert ctx.source == "session-1.jsonl"
        assert DENVER in ctx.passage
        note = ClaimContext(content=BOSTON, source="MEMORY.md", date="2026-06-01", passage="")
        assert reading_for(note, ctx) == READING_OBSERVED

    def test_a_chunk_hash_narrows_the_passage_to_that_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import hashlib

        from particles.config import get_config
        from particles.extraction.general import (
            _normalise_for_hashing,
            _split_into_paragraph_chunks,
        )

        monkeypatch.setattr(get_config().extraction, "html_chunk_size", 40)
        text = "The home city is Boston.\n\n" + "Filler about lunch.\n\n" * 3 + DENVER
        chunks = _split_into_paragraph_chunks(_normalise_for_hashing(text), 40)
        last = hashlib.sha256(chunks[-1].encode()).hexdigest()
        ctx = in_flight_context(
            DENVER,
            text=text,
            chunk_hash=last,
            partner=BOSTON,
            source="s",
            source_type=None,
            mutability=None,
            when=None,
        )
        assert "Boston" not in ctx.passage and DENVER in ctx.passage
        assert ctx.kind is SourceKind.NOTE and ctx.observed == "unknown"


def test_a_rung_3_record_carries_the_instruction_only_when_given() -> None:
    from particles.core.schema import ProvenanceRefType
    from particles.ingest.conflict_plan import plan_quarantine

    existing, new = _p(BOSTON), _p(DENVER)
    kw: dict[str, Any] = {
        "corpus_entry_id": "e1",
        "snapshot_id": "s1",
        "trigger_ref_type": ProvenanceRefType.SOURCE,
    }
    _, stamped = plan_quarantine(existing, new, reading=READING_OBSERVED, **kw)
    _, bare = plan_quarantine(existing, new, **kw)
    assert (stamped.properties or {})[READING_KEY] == READING_OBSERVED
    assert READING_KEY not in (bare.properties or {})


# ---------------------------------------------------------------------------
# The write path: one transcript, re-extracted (a same-entry pair)
# ---------------------------------------------------------------------------


async def _transcript(
    session: AsyncSession,
    *,
    verify: bool = True,
    uri: str | None = None,
) -> tuple[list[Particle], Any]:
    """Extract two snapshots of one append-only transcript; the second says Denver.

    Returns what the second pass wrote and its :class:`SnapshotOutcome`.
    """
    from particles.config import get_config
    from particles.corpus.deposit import deposit_text_versioned
    from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot

    get_config().extraction.verify_conflicts = verify
    extractor = _OneClaim()
    uri = uri or f"claude-code://session/{uuid.uuid4()}"
    first = f"day 1: {BOSTON}"
    second = f"{first}\nday 9: {DENVER}"
    extractor.claims[first.encode()] = BOSTON
    extractor.claims[second.encode()] = DENVER
    outcome = SnapshotOutcome()
    written: list[Particle] = []
    for text in (first, second):
        entry_id, snapshot_id, _ = await deposit_text_versioned(
            session,
            text=text,
            uri_r=uri,
            source_type=SourceType.CONVERSATION,
            mutability=Mutability.APPEND_ONLY,
        )
        await session.commit()
        written = await extract_snapshot(
            session, entry_id, snapshot_id, extractor=extractor, outcome_out=outcome
        )
        await session.commit()
    return written, outcome


async def _records(session: AsyncSession) -> list[Particle]:
    from particles.store.particle_store import get_particles_by_status

    return await get_particles_by_status(session, Status.INCONSISTENCY)


@pytest.mark.asyncio
@pytest.mark.usefixtures("bow")
class TestSameEntry:
    async def test_confirmed_goes_to_rung_3_and_the_record_is_stamped(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _reading(monkeypatch, _YES)
        _, outcome = await _transcript(db_session)
        [record] = await _records(db_session)
        assert (record.properties or {})[READING_KEY] == READING_OBSERVED
        [denver] = await _by_content(db_session, DENVER)
        assert denver.status_reason is StatusReason.CONFLICT_PENDING
        assert len(reads.pairs) == 1
        assert (outcome.conflicts_read, outcome.conflicts_cleared) == (1, 0)

    async def test_not_confirmed_corroborates_with_no_record_and_no_quarantine(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reading(monkeypatch, _NO)
        _, outcome = await _transcript(db_session)
        assert await _records(db_session) == []
        [boston], [denver] = (
            await _by_content(db_session, BOSTON),
            await _by_content(db_session, DENVER),
        )
        assert (boston.status, denver.status) == (Status.ACTIVE, Status.ACTIVE)
        assert (outcome.conflicts_read, outcome.conflicts_cleared) == (1, 1)

    async def test_the_newcomer_is_read_as_the_records_observation(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.corpus.store import get_snapshot

        reads = _reading(monkeypatch, _YES)
        written, _ = await _transcript(db_session)
        [(existing, newcomer)] = reads.pairs
        assert existing.content == BOSTON and newcomer.content == DENVER
        assert newcomer.kind is SourceKind.RECORD
        assert newcomer.source_type == SourceType.CONVERSATION
        [denver] = await _by_content(db_session, DENVER)
        snapshot = await get_snapshot(db_session, denver.provenance[0].snapshot_id or "")
        assert snapshot is not None
        assert newcomer.observed == snapshot.captured_at.strftime("%Y-%m-%d %H:%M UTC")
        assert reading_for(existing, newcomer) == READING_OBSERVED

    async def test_a_failed_reading_keeps_the_signal_and_opens_an_unstamped_record(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reading(monkeypatch, None)
        _, outcome = await _transcript(db_session)
        [record] = await _records(db_session)
        assert READING_KEY not in (record.properties or {})
        assert outcome.conflict_unread == 1

    async def test_verify_conflicts_false_restores_the_probe_alone(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _reading(monkeypatch, _NO)
        _, outcome = await _transcript(db_session, verify=False)
        assert reads.pairs == []
        [record] = await _records(db_session)
        assert READING_KEY not in (record.properties or {})
        assert outcome.conflicts_read == 0

    async def test_no_llm_call_happens_inside_the_write_lock(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.ingest import pipeline

        held = [False]
        real_lock = pipeline.write_lock

        @contextlib.asynccontextmanager
        async def watched() -> AsyncIterator[None]:
            async with real_lock():
                held[0] = True
                try:
                    yield
                finally:
                    held[0] = False

        calls_under_lock: list[str] = []

        async def reading(a: ClaimContext, b: ClaimContext) -> ProbeVerdict:
            if held[0]:
                calls_under_lock.append("second reading")
            return _YES

        probe = pipeline._has_contradiction_signal

        async def watched_probe(a: str, b: str) -> bool | None:
            if held[0]:
                calls_under_lock.append("probe")
            return await probe(a, b)

        from particles.ingest import second_reading

        monkeypatch.setattr(pipeline, "write_lock", watched)
        monkeypatch.setattr(second_reading, "_llm_verify_contradiction", reading)
        # A scoped patch: ``bow`` owns the probe's patch, and a monkeypatch
        # would restore bow's mock after bow had already torn it down.
        with patch.object(pipeline, "_has_contradiction_signal", watched_probe):
            await _transcript(db_session)
        assert len(await _records(db_session)) == 1
        assert calls_under_lock == []


# ---------------------------------------------------------------------------
# Two sessions (a cross-entry pair, via the subject pool)
# ---------------------------------------------------------------------------


async def _two_sessions(session: AsyncSession) -> Any:
    from particles.corpus.deposit import deposit_text_versioned
    from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot

    extractor = _OneClaim()
    outcome = SnapshotOutcome()
    for day, claim in ((1, BOSTON), (9, DENVER)):
        text = f"day {day}: {claim}"
        extractor.claims[text.encode()] = claim
        entry_id, snapshot_id, _ = await deposit_text_versioned(
            session,
            text=text,
            uri_r=f"claude-code://session/{uuid.uuid4()}",
            source_type=SourceType.CONVERSATION,
            mutability=Mutability.STABLE,
            content_published_at=T0 + timedelta(days=day),
        )
        await session.commit()
        await extract_snapshot(
            session, entry_id, snapshot_id, extractor=extractor, outcome_out=outcome
        )
        await session.commit()
    return outcome


@pytest.mark.asyncio
@pytest.mark.usefixtures("bow")
class TestCrossEntry:
    async def test_a_failed_reading_drops_the_signal_and_writes_nothing(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reading(monkeypatch, None)
        outcome = await _two_sessions(db_session)
        [boston], [denver] = (
            await _by_content(db_session, BOSTON),
            await _by_content(db_session, DENVER),
        )
        # Without the reading this pair is an update: Boston would be retired.
        assert (boston.status, denver.status) == (Status.ACTIVE, Status.ACTIVE)
        assert await _records(db_session) == []
        assert outcome.conflict_unread == 1

    async def test_confirmed_acts_as_today(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reading(monkeypatch, _YES)
        await _two_sessions(db_session)
        [boston] = await _by_content(db_session, BOSTON)
        assert boston.status_reason is StatusReason.SUPERSEDED_BY_UPDATE


# ---------------------------------------------------------------------------
# A declined pair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("bow_encoder", "scripted_probe")
class TestDeclined:
    @pytest.mark.parametrize(("verdict", "edges"), [(_YES, 1), (_NO, 0)])
    async def test_a_contradicts_edge_is_written_only_when_the_reading_confirms(
        self,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
        verdict: ProbeVerdict,
        edges: int,
    ) -> None:
        from particles.corpus.deposit import deposit_text_versioned
        from particles.ingest.observer_gate import DivergenceTally
        from particles.ingest.pipeline import extract_snapshot
        from tests._observer_scope import rescoped
        from tests.test_observer_gate import (
            _CLOCK,
            _LineChunkExtractor,
            branch,
            harvest,
            relations,
        )

        reads = _reading(monkeypatch, verdict)
        extractor = _LineChunkExtractor()
        await rescoped(db_session)
        await harvest(db_session, "alpha", [branch("main")], extractor)
        entry_id, snapshot_id, _ = await deposit_text_versioned(
            db_session,
            text=branch("master") + "\n",
            uri_r="file:///home/me/.claude/projects/beta/memory/MEMORY.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
            tags=["claude-code", "memory-file", "project:beta"],
            content_published_at=_CLOCK[0] + timedelta(days=1),
        )
        await db_session.commit()
        tally = DivergenceTally()
        await extract_snapshot(
            db_session, entry_id, snapshot_id, extractor=extractor, divergences_out=tally
        )
        await db_session.commit()
        assert len(reads.pairs) == 1
        assert len(await relations(db_session)) == edges
        assert tally.declined == edges


# ---------------------------------------------------------------------------
# Layering: one reading, one breaker, no new pinned edge
# ---------------------------------------------------------------------------


def test_the_census_and_extraction_share_one_reading_and_one_breaker() -> None:
    import particles.ingest.pipeline as pipeline
    import particles.llm.breaker as breaker
    import particles.operations._llm as old_seam
    import particles.operations.lint.contradictions as census

    assert old_seam is breaker
    assert census._llm_verify_contradiction.__module__ == "particles.ingest.second_reading"
    assert pipeline.read_pair.__module__ == "particles.ingest.second_reading"
    assert census.claim_context is pipeline.claim_context


def test_the_reading_imports_nothing_from_operations() -> None:
    # Located through the import system, not the checkout: the breaker ships in
    # the Client distribution, so a published Engine tree has no file at its
    # repository path.
    for module in (
        "particles.ingest.second_reading",
        "particles.ingest.source_passage",
        "particles.llm.breaker",
    ):
        spec = importlib.util.find_spec(module)
        assert spec is not None and spec.origin is not None, module
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not {m for m in imported if m.startswith("particles.operations")}, module


@upstream_only
def test_the_reading_pins_no_import_linter_edge() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pinned = [
        edge
        for contract in config["tool"]["importlinter"]["contracts"]
        for edge in contract.get("ignore_imports", [])
    ]
    assert not [e for e in pinned if "second_reading" in e or "breaker" in e]
    assert not [e for e in pinned if "source_passage" in e]
