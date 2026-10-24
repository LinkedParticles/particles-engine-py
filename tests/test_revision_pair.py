# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A contradiction between two generations of one source is settled by the generation.

The defect, measured on a copy of the live store on 2026-10-21: a ``MUTABLE``
file's new version held a claim the probe found contradicts a claim of its old
version, rung 2.5 did not take the pair, rung 3 quarantined the new claim, and
the cascade then retired the old one in the same transaction. The
store believed neither value; 13 of 51 open write-time records were in that
state. The record's acceptance table is pinned here case by case, with a
stubbed extractor and scripted probes, and no LLM.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from particles.core.conflict_resolution import ConflictVerdict, LadderOutcome, SlotVerdict
from particles.core.schema import (
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    FetchPolicy,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.status import Status, StatusReason
from particles.corpus.store import EntryCurrency
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.ingest.pair_selection import plan_update_extras
from particles.ingest.pipeline import SnapshotOutcome
from particles.ingest.revision_pair import (
    REVISION_SETTLED,
    generation_order,
    restated_ids,
    revision_claims,
    revision_settled_note,
    settle_by_generation,
)
from particles.ingest.second_reading import ProbeVerdict

T0 = datetime(2026, 6, 1, tzinfo=UTC)

NOTION = "The author uses Notion for tasks."
THINGS = "The author moved their tasks to Things."
STILL_NOTION = "Tasks still live in Notion."

#: The scripted contradiction probe: these pairs contradict, nothing else does.
_CONTRADICTS = {frozenset((NOTION, THINGS)), frozenset((STILL_NOTION, THINGS))}


# ---------------------------------------------------------------------------
# The pure predicate
# ---------------------------------------------------------------------------


def _snap(snapshot_id: str, at: datetime, published: datetime | None = None) -> Snapshot:
    return Snapshot(
        snapshot_id=snapshot_id,
        captured_at=at,
        content_published_at=published,
        content_hash="0" * 64,
        warc_record_type=WarcRecordType.RESPONSE,
        extraction_status=ExtractionStatus.COMPLETE,
    )


def _corpus_entry(
    entry_id: str, snaps: list[Snapshot], mutability: Mutability = Mutability.MUTABLE
) -> CorpusEntry:
    return CorpusEntry(
        entry_id=entry_id,
        source_type="LOCAL_MARKDOWN",
        uri_r=f"file:///tmp/{entry_id}.md",
        fetch_policy=FetchPolicy.LAZY,
        mutability=mutability,
        deposited_by="test",
        snapshots=snaps,
    )


def _claim(content: str, *refs: tuple[str, str | None]) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=e, snapshot_id=s)
            for e, s in refs
        ],
    )


def _currency(latest: str | None, *, mutable: bool = True) -> EntryCurrency:
    return EntryCurrency(tags=[], mutable=mutable, latest_snapshot_id=latest)


class TestRevisionClaims:
    """Each of §1's three conditions, alone."""

    def _world(self) -> tuple[CorpusEntry, Snapshot, Snapshot]:
        old, new = _snap("v1", T0), _snap("v2", T0 + timedelta(days=1))
        return _corpus_entry("E", [old, new]), old, new

    def test_a_claim_only_the_previous_generation_states_is_one(self) -> None:
        entry, old, new = self._world()
        x = _claim(NOTION, ("E", "v1"))
        got = revision_claims(
            [x], entry=entry, snapshot=new, currency={"E": _currency("v1")}, restated=()
        )
        assert got == {x.id}

    def test_a_claim_another_entry_currently_states_is_not(self) -> None:
        entry, _old, new = self._world()
        x = _claim(NOTION, ("E", "v1"), ("F", "f1"))
        currency = {"E": _currency("v1"), "F": _currency(None, mutable=False)}
        assert not revision_claims([x], entry=entry, snapshot=new, currency=currency, restated=())

    def test_another_entrys_superseded_generation_does_not_attest(self) -> None:
        entry, _old, new = self._world()
        x = _claim(NOTION, ("E", "v1"), ("F", "f1"))
        currency = {"E": _currency("v1"), "F": _currency("f2")}
        got = revision_claims([x], entry=entry, snapshot=new, currency=currency, restated=())
        assert got == {x.id}

    def test_a_ref_to_no_corpus_entry_is_not_a_source(self) -> None:
        entry, _old, new = self._world()
        x = _claim(NOTION, ("E", "v1"), ("gone", "g1"))
        got = revision_claims(
            [x], entry=entry, snapshot=new, currency={"E": _currency("v1")}, restated=()
        )
        assert got == {x.id}

    def test_a_claim_not_naming_the_current_generation_is_not(self) -> None:
        older = _snap("v0", T0 - timedelta(days=1))
        entry, old, new = self._world()
        entry = _corpus_entry("E", [older, old, new])
        x = _claim(NOTION, ("E", "v0"))
        assert not revision_claims(
            [x], entry=entry, snapshot=new, currency={"E": _currency("v1")}, restated=()
        )

    def test_an_older_snapshot_extracted_late_is_not_newer(self) -> None:
        early, late = _snap("v1", T0), _snap("v3", T0 + timedelta(days=3))
        s = _snap("v2", T0 + timedelta(days=2))
        entry = _corpus_entry("E", [early, s, late])
        x = _claim(NOTION, ("E", "v1"), ("E", "v3"))
        assert not revision_claims(
            [x], entry=entry, snapshot=s, currency={"E": _currency("v3")}, restated=()
        )

    def test_a_restated_claim_is_not(self) -> None:
        entry, _old, new = self._world()
        x = _claim(NOTION, ("E", "v1"))
        assert not revision_claims(
            [x], entry=entry, snapshot=new, currency={"E": _currency("v1")}, restated={x.id}
        )

    def test_a_non_mutable_entry_has_no_generations(self) -> None:
        old, new = _snap("v1", T0), _snap("v2", T0 + timedelta(days=1))
        for mutability in (Mutability.STABLE, Mutability.APPEND_ONLY):
            entry = _corpus_entry("E", [old, new], mutability)
            x = _claim(NOTION, ("E", "v1"))
            assert not revision_claims(
                [x], entry=entry, snapshot=new, currency={"E": _currency("v1")}, restated=()
            )

    def test_restated_is_carry_forward_plus_verbatim_content(self) -> None:
        x = _claim(NOTION, ("E", "v1"))
        y = _claim("Something else.", ("E", "v1"))
        carried = _claim("A carried claim.", ("E", "v1"))
        got = restated_ids(
            [x, y, carried],
            carry_forward_ids=[carried.id],
            candidate_contents=[f"  {NOTION[:-1]}  ", THINGS],
        )
        assert got == {x.id, carried.id}

    def test_a_reworded_restatement_is_not_a_restatement(self) -> None:
        x = _claim(NOTION, ("E", "v1"))
        got = restated_ids([x], carry_forward_ids=[], candidate_contents=[STILL_NOTION])
        assert not got


class TestSettlement:
    def test_rung_three_is_written_as_corroboration(self) -> None:
        outcome = LadderOutcome(verdict=ConflictVerdict.INCONSISTENT, record_divergence=False)
        settled = settle_by_generation(outcome)
        assert settled is not None and settled.verdict is ConflictVerdict.CORROBORATES

    @pytest.mark.parametrize(
        "verdict",
        [v for v in ConflictVerdict if v is not ConflictVerdict.INCONSISTENT] + [None],
    )
    def test_every_other_verdict_stands(self, verdict: ConflictVerdict | None) -> None:
        assert settle_by_generation(LadderOutcome(verdict=verdict, record_divergence=False)) is None

    def test_the_generation_orders_rung_two_point_five(self) -> None:
        assert generation_order(-1) == 1
        assert generation_order(1) == 1
        assert generation_order(None) is None

    def test_the_note(self) -> None:
        expected = f"{REVISION_SETTLED}: 2 pair(s) settled by the generation"
        assert revision_settled_note(2) == expected


class TestExtras:
    def test_a_revision_extra_is_retired_even_when_its_date_is_later(self) -> None:
        """rung 2.5 on a revision pair may retire x, and never c."""
        newer_date = _snap("v1", T0, published=T0 + timedelta(days=5))
        s = _snap("v2", T0 + timedelta(days=1), published=T0)
        entry = _corpus_entry("E", [newer_date, s])
        winner = _claim(THINGS, ("E", "v2"))
        extra = _claim(NOTION, ("E", "v1"))
        args = (winner, entry, "v2", [extra], {"E": entry}, {}, {extra.id})
        assert plan_update_extras(*args) == [(winner.id, extra.id)]
        assert plan_update_extras(*args, revision={extra.id}) == [(extra.id, winner.id)]


# ---------------------------------------------------------------------------
# The pipeline: acceptance table
# ---------------------------------------------------------------------------


class _Constant:
    """Every text embeds to one vector, so every pair is the nearest pair."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        out = np.zeros((len(texts), 8), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class _Scripted:
    """An extractor emitting scripted claims per deposited text, keeping each result."""

    EXTRACTOR_ID = "test-extractor"
    EXTRACTOR_VERSION = "1"

    def __init__(self) -> None:
        self.claims: dict[bytes, list[str]] = {}
        self.results: list[ExtractionResult] = []

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> ExtractionResult:
        result = ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=claim,
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["the author"],
                )
                for claim in self.claims[content]
            ]
        )
        self.results.append(result)
        return result


async def _contradicts(a: str, b: str) -> bool:
    return frozenset((a, b)) in _CONTRADICTS


@pytest.fixture
def probes() -> Generator[dict[str, SlotVerdict], None, None]:
    """Both probes and the second reading scripted; yields the update verdict to give."""
    from particles import embeddings as ep

    slot = {"verdict": SlotVerdict.DIFFERENT}

    async def updates(earlier: str, later: str) -> SlotVerdict:
        return slot["verdict"]

    original = ep._embedding_model
    ep.set_embedding_model(_Constant())  # type: ignore[arg-type]
    try:
        with (
            patch(
                "particles.ingest.pipeline._has_contradiction_signal",
                AsyncMock(side_effect=_contradicts),
            ),
            patch("particles.ingest.pipeline._has_update_signal", AsyncMock(side_effect=updates)),
            patch(
                "particles.ingest.pipeline.read_pair",
                AsyncMock(return_value=ProbeVerdict(usable=True, contradicts=True)),
            ),
        ):
            yield slot  # type: ignore[misc]
    finally:
        ep.set_embedding_model(original)


class _Source:
    """One source under test: deposit a version, extract it, read what it left."""

    def __init__(
        self,
        session: Any,
        extractor: _Scripted,
        *,
        mutability: Mutability = Mutability.MUTABLE,
        uri: str | None = None,
    ) -> None:
        self.session = session
        self.extractor = extractor
        self.mutability = mutability
        self.uri = uri or f"file:///notes/{uuid.uuid4().hex}.md"
        self.entry_id: str | None = None

    async def deposit(self, claims: list[str], *, day: int) -> str:
        from particles.corpus.deposit import deposit_text_versioned

        text = f"version of day {day}: " + " ".join(claims)
        self.extractor.claims[text.encode()] = claims
        entry_id, snapshot_id, unchanged = await deposit_text_versioned(
            self.session,
            text=text,
            uri_r=self.uri,
            source_type="LOCAL_MARKDOWN",
            mutability=self.mutability,
            content_published_at=T0 + timedelta(days=day),
        )
        assert not unchanged
        self.entry_id = entry_id
        await self.session.commit()
        return snapshot_id

    async def extract(self, snapshot_id: str) -> SnapshotOutcome:
        from particles.ingest.pipeline import extract_snapshot

        assert self.entry_id is not None
        outcome = SnapshotOutcome()
        await extract_snapshot(
            self.session,
            self.entry_id,
            snapshot_id,
            extractor=self.extractor,
            outcome_out=outcome,
        )
        await self.session.commit()
        return outcome

    async def say(self, claims: list[str], *, day: int) -> SnapshotOutcome:
        return await self.extract(await self.deposit(claims, day=day))


async def _one(session: Any, content: str) -> Particle:
    from particles.store.particle_store import get_particles_by_status

    found = [
        p
        for status in Status
        for p in await get_particles_by_status(session, status)
        if p.content == content
    ]
    assert len(found) == 1, (content, found)
    return found[0]


async def _records(session: Any) -> list[Particle]:
    from particles.store.particle_store import get_particles_by_status

    return await get_particles_by_status(session, Status.INCONSISTENCY)


def _notes(extractor: _Scripted) -> list[str]:
    return [n for r in extractor.results for n in r.quality_notes if REVISION_SETTLED in n]


def _quarantined(p: Particle) -> bool:
    return p.status is Status.PROVENANCE_STALE and p.status_reason is StatusReason.CONFLICT_PENDING


@pytest.mark.asyncio
class TestAcceptance:
    async def test_case_1_an_edit_settles_by_the_generation(
        self, db_session: Any, probes: dict[str, SlotVerdict]
    ) -> None:
        ex = _Scripted()
        note = _Source(db_session, ex)
        await note.say([NOTION], day=1)
        outcome = await note.say([THINGS], day=2)

        c, x = await _one(db_session, THINGS), await _one(db_session, NOTION)
        assert c.status is Status.ACTIVE
        assert x.status is Status.PROVENANCE_STALE
        assert x.status_reason is StatusReason.RETRACTED_DEPENDENCY
        assert await _records(db_session) == []
        assert outcome.revision_settled == 1
        assert _notes(ex) == [revision_settled_note(1)]

    async def test_case_1_writes_no_dropped_candidate_event(
        self, db_session: Any, probes: dict[str, SlotVerdict]
    ) -> None:
        from particles.store.event_store import list_events

        note = _Source(db_session, _Scripted())
        await note.say([NOTION], day=1)
        await note.say([THINGS], day=2)
        events = [e.event_type.value for e in await list_events(db_session)]
        assert not [e for e in events if "DROPPED" in e or "SUPERSESSION" in e]

    async def test_case_2_an_update_retires_the_old_value(
        self, db_session: Any, probes: dict[str, SlotVerdict]
    ) -> None:
        probes["verdict"] = SlotVerdict.CHANGES
        ex = _Scripted()
        note = _Source(db_session, ex)
        await note.say([NOTION], day=1)
        outcome = await note.say([THINGS], day=2)

        c, x = await _one(db_session, THINGS), await _one(db_session, NOTION)
        assert c.status is Status.ACTIVE and c.supersedes == x.id
        assert x.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        assert await _records(db_session) == []
        assert outcome.revision_settled == 0
        assert _notes(ex) == []

    async def test_case_3_the_direction_comes_from_the_generation(
        self, db_session: Any, probes: dict[str, SlotVerdict]
    ) -> None:
        """A file restored from a backup carries an older mtime than the text it replaces."""
        probes["verdict"] = SlotVerdict.CHANGES
        ex = _Scripted()
        note = _Source(db_session, ex)
        await note.say([NOTION], day=5)
        outcome = await note.say([THINGS], day=1)

        c, x = await _one(db_session, THINGS), await _one(db_session, NOTION)
        assert c.status is Status.ACTIVE and c.supersedes == x.id
        assert x.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        assert outcome.revision_settled == 0
        assert _notes(ex) == []

    @pytest.mark.parametrize("order", [[NOTION, THINGS], [THINGS, NOTION]])
    async def test_case_4_a_version_stating_both_is_recorded(
        self, db_session: Any, probes: dict[str, SlotVerdict], order: list[str]
    ) -> None:
        ex = _Scripted()
        note = _Source(db_session, ex)
        await note.say([NOTION], day=1)
        outcome = await note.say(order, day=2)

        c, x = await _one(db_session, THINGS), await _one(db_session, NOTION)
        assert _quarantined(c)
        assert x.status is Status.ACTIVE
        assert len(await _records(db_session)) == 1
        assert outcome.revision_settled == 0
        assert _notes(ex) == []

    async def test_case_5_a_claim_another_source_states_is_recorded(
        self, db_session: Any, probes: dict[str, SlotVerdict]
    ) -> None:
        ex = _Scripted()
        note = _Source(db_session, ex)
        await note.say([NOTION], day=1)
        # The same claim, verbatim, from a second entry: folded into x.
        await _Source(db_session, ex).say([NOTION], day=0)
        outcome = await note.say([THINGS], day=2)

        c, x = await _one(db_session, THINGS), await _one(db_session, NOTION)
        assert len({r.corpus_entry_id for r in x.provenance}) == 2
        assert _quarantined(c)
        assert x.status is Status.ACTIVE
        assert len(await _records(db_session)) == 1
        assert outcome.revision_settled == 0
        assert _notes(ex) == []

    async def test_case_6_a_stable_entry_is_recorded(
        self, db_session: Any, probes: dict[str, SlotVerdict]
    ) -> None:
        ex = _Scripted()
        note = _Source(db_session, ex, mutability=Mutability.STABLE)
        await note.say([NOTION], day=1)
        outcome = await note.say([THINGS], day=2)

        c, x = await _one(db_session, THINGS), await _one(db_session, NOTION)
        assert _quarantined(c)
        assert x.status is Status.ACTIVE
        assert len(await _records(db_session)) == 1
        assert outcome.revision_settled == 0
        assert _notes(ex) == []

    @pytest.mark.parametrize("order", [[THINGS, STILL_NOTION], [STILL_NOTION, THINGS]])
    async def test_case_7_a_reworded_restatement_is_not_a_restatement(
        self, db_session: Any, probes: dict[str, SlotVerdict], order: list[str]
    ) -> None:
        ex = _Scripted()
        note = _Source(db_session, ex)
        await note.say([NOTION], day=1)
        outcome = await note.say(order, day=2)

        c = await _one(db_session, THINGS)
        reworded = await _one(db_session, STILL_NOTION)
        x = await _one(db_session, NOTION)
        assert c.status is Status.ACTIVE
        assert reworded.status is Status.ACTIVE
        assert x.status_reason is StatusReason.RETRACTED_DEPENDENCY
        assert await _records(db_session) == []
        assert outcome.revision_settled == 1
        assert _notes(ex) == [revision_settled_note(1)]

    async def test_case_8_a_collapsed_generation_changes_nothing(
        self, db_session: Any, probes: dict[str, SlotVerdict]
    ) -> None:
        from particles.corpus.store import get_snapshot_superseded_by
        from particles.ingest.pending_collapse import collapse_superseded_pending

        ex = _Scripted()
        note = _Source(db_session, ex)
        await note.say([NOTION], day=1)
        between = await note.deposit(["An interim draft of the task list."], day=2)
        latest = await note.deposit([THINGS], day=3)
        await collapse_superseded_pending(db_session)
        await db_session.commit()
        assert await get_snapshot_superseded_by(db_session, between) == latest
        outcome = await note.extract(latest)

        c, x = await _one(db_session, THINGS), await _one(db_session, NOTION)
        assert c.status is Status.ACTIVE
        assert x.status_reason is StatusReason.RETRACTED_DEPENDENCY
        assert await _records(db_session) == []
        assert outcome.revision_settled == 1
        assert _notes(ex) == [revision_settled_note(1)]

    async def test_case_9_an_older_snapshot_extracted_late_is_recorded(
        self, db_session: Any, probes: dict[str, SlotVerdict]
    ) -> None:
        ex = _Scripted()
        note = _Source(db_session, ex)
        await note.say([NOTION], day=1)
        older = await note.deposit([THINGS], day=2)
        newer = await note.deposit([NOTION, "The author reviews the list on Fridays."], day=3)
        await note.extract(newer)
        outcome = await note.extract(older)

        c = await _one(db_session, THINGS)
        assert _quarantined(c)
        assert len(await _records(db_session)) == 1
        assert outcome.revision_settled == 0
        assert _notes(ex) == []
