# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A restated claim is carried into the new generation.

The defect, measured on 2026-10-22: re-importing five lightly edited vault
notes retired 36 of their 54 claims and minted 46 reworded copies, because a
short note is read whole and the model rewords sentences the edit never
touched. These tests pin the pure pair selection and write rule, the shell's
use of the probe ledger, and the pipeline outcomes (carry, admission period,
lineage, off switch) with a stubbed extractor and scripted model answers.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from particles.config import get_config
from particles.core.conflict_resolution import SlotVerdict
from particles.core.rung_ladder import DuplicateVerdict
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.status import Status, StatusReason
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.ingest.pipeline import SnapshotOutcome
from particles.ingest.restatement import (
    RESTATED,
    RESTATED_AS_KEY,
    RESTATEMENT_SHADOW,
    RESTATEMENT_UNASKED,
    Carry,
    RestatementCandidate,
    RestatementOutcome,
    UpdateGuard,
    decide_carry,
    match_restatements,
    restatement_notes,
    select_pairs,
)
from particles.ingest.second_reading import ProbeVerdict

T0 = datetime(2026, 6, 1, tzinfo=UTC)

PODCAST = '"No hay negros en el Tibet" publica episodios un par de veces al mes.'
PODCAST_REWORDED = "El podcast No hay negros en el Tibet publica dos episodios al mes."
WEEKLY = "El podcast No hay negros en el Tibet publica un episodio cada semana."


def _vec(*values: float) -> Any:
    out = np.zeros(4, dtype=np.float32)
    out[: len(values)] = values
    return out / np.linalg.norm(out)


def _claim(content: str, *, holder: str | None = None) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.8),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        status=Status.ACTIVE,
        asserted_by="test",
        asserted_at=T0,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="E", snapshot_id="v1")
        ],
        properties={"stance:holder": holder} if holder else None,
    )


def _cand(content: str, emb: Any, *, holder: str | None = None) -> RestatementCandidate:
    return RestatementCandidate(content=content, holder=holder, embedding=emb)


# ---------------------------------------------------------------------------
# The pair selection
# ---------------------------------------------------------------------------


class TestSelectPairs:
    def test_one_to_one_by_descending_cosine(self) -> None:
        x, y = _claim("Alpha holds."), _claim("Beta holds.")
        candidates = [
            _cand("Alpha still holds.", _vec(1, 0.1)),
            _cand("Beta holds now.", _vec(0.1, 1)),
        ]
        pairs = select_pairs(candidates, [(x, _vec(1, 0)), (y, _vec(0, 1))], floor=0.5)
        assert {(p.index, p.existing.id) for p in pairs} == {(0, x.id), (1, y.id)}

    def test_a_claim_is_never_carried_by_two_candidates(self) -> None:
        x = _claim("Alpha holds.")
        candidates = [_cand("Alpha holds, again.", _vec(1, 0.2)), _cand("Alpha holds!", _vec(1, 0))]
        pairs = select_pairs(candidates, [(x, _vec(1, 0))], floor=0.5)
        assert [(p.index, p.existing.id) for p in pairs] == [(1, x.id)]

    def test_below_the_floor_is_not_a_pair(self) -> None:
        x = _claim("Alpha holds.")
        pairs = select_pairs([_cand("Gamma differs.", _vec(0, 1))], [(x, _vec(1, 0))], floor=0.7)
        assert pairs == []

    def test_a_key_equal_pair_needs_no_cosine_and_comes_first(self) -> None:
        x = _claim("“Quoted title” airs twice a month.")
        y = _claim("Something nearer in embedding space.")
        # The candidate embeds nearer y, but it is x's sentence under the key.
        candidates = [_cand('"Quoted title" airs twice a month', _vec(0, 1))]
        pairs = select_pairs(candidates, [(x, _vec(1, 0)), (y, _vec(0, 1))], floor=0.9)
        assert [(p.existing.id, p.keyed) for p in pairs] == [(x.id, True)]

    def test_a_different_holder_is_never_a_pair(self) -> None:
        x = _claim("I like tea.", holder="alice")
        pairs = select_pairs(
            [_cand("I like tea.", _vec(1, 0), holder="bob")], [(x, _vec(1, 0))], floor=0.1
        )
        assert pairs == []

    def test_a_skipped_candidate_takes_no_part(self) -> None:
        x = _claim("Alpha holds.")
        skipped = RestatementCandidate(
            content="Alpha holds.", holder=None, embedding=_vec(1, 0), skip=True
        )
        assert select_pairs([skipped], [(x, _vec(1, 0))], floor=0.1) == []


# ---------------------------------------------------------------------------
# The write rule
# ---------------------------------------------------------------------------

_QUIET = UpdateGuard(verdict=SlotVerdict.DIFFERENT, any_changes=False)


class TestDecideCarry:
    def test_a_key_equal_pair_always_carries(self) -> None:
        got = decide_carry(
            DuplicateVerdict.ABSTAIN, keyed=True, admitted=False, update=None, contradiction=None
        )
        assert got is Carry.CARRY

    def test_an_admitted_same_with_quiet_guards_carries(self) -> None:
        got = decide_carry(
            DuplicateVerdict.SAME, keyed=False, admitted=True, update=_QUIET, contradiction=False
        )
        assert got is Carry.CARRY

    @pytest.mark.parametrize(
        ("update", "contradiction"),
        [
            # A first reading said CHANGES, even though the second did not.
            (UpdateGuard(verdict=SlotVerdict.DIFFERENT, any_changes=True), False),
            (UpdateGuard(verdict=SlotVerdict.CHANGES, any_changes=True), False),
            # A reading that could not complete blocks the carry.
            (UpdateGuard(verdict=None, any_changes=False), False),
            (None, False),
            # The contradiction probe confirmed, or could not complete.
            (_QUIET, True),
            (_QUIET, None),
        ],
    )
    def test_any_dissenting_or_missing_reading_blocks_the_carry(
        self, update: UpdateGuard | None, contradiction: bool | None
    ) -> None:
        got = decide_carry(
            DuplicateVerdict.SAME,
            keyed=False,
            admitted=True,
            update=update,
            contradiction=contradiction,
        )
        assert got is Carry.NO

    def test_before_admission_a_same_is_only_counted(self) -> None:
        got = decide_carry(
            DuplicateVerdict.SAME, keyed=False, admitted=False, update=_QUIET, contradiction=None
        )
        assert got is Carry.SHADOW

    @pytest.mark.parametrize("verdict", [DuplicateVerdict.DIFFERENT, DuplicateVerdict.ABSTAIN])
    def test_anything_but_same_writes_as_before(self, verdict: DuplicateVerdict) -> None:
        got = decide_carry(verdict, keyed=False, admitted=True, update=_QUIET, contradiction=False)
        assert got is Carry.NO


def test_notes_disclose_every_outcome() -> None:
    outcome = RestatementOutcome(shadow=2, unasked=3)
    notes = restatement_notes(outcome, carried=1, lineage_written=0)
    assert [n.split(":")[0] for n in notes] == [RESTATED, RESTATEMENT_SHADOW, RESTATEMENT_UNASKED]
    assert restatement_notes(RestatementOutcome(), carried=0, lineage_written=0) == []


# ---------------------------------------------------------------------------
# The shell: judge, ledger, guards
# ---------------------------------------------------------------------------


def _judge_reply(*answers: str) -> Any:
    async def call(prompt: str, *, max_tokens: int) -> str:
        return json.dumps({f"p{i}": a for i, a in enumerate(answers)})

    return AsyncMock(side_effect=call)


def _slot(guard: UpdateGuard) -> Any:
    return AsyncMock(return_value=guard)


@pytest.mark.asyncio
class TestMatch:
    async def _run(
        self,
        session: Any,
        judge: Any,
        guard: UpdateGuard = _QUIET,
        contradiction: bool | None = False,
    ) -> tuple[RestatementOutcome, Particle, Any, Any]:
        x = _claim(PODCAST)
        slot = _slot(guard)
        probe = AsyncMock(return_value=contradiction)
        outcome = await match_restatements(
            session,
            [_cand(PODCAST_REWORDED, _vec(1, 0.1))],
            [(x, _vec(1, 0))],
            judge_call=judge,
            read_slot=slot,
            probe_contradiction=probe,
        )
        return outcome, x, slot, probe

    async def test_an_admitted_same_carries(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().generation.restatement, "judge_admitted", True)
        outcome, x, _, probe = await self._run(db_session, _judge_reply("PARAPHRASE"))
        assert outcome.carried == {0: x}
        assert outcome.judged == 1 and outcome.shadow == 0
        probe.assert_awaited_once()

    async def test_before_admission_a_same_is_counted_and_not_probed(self, db_session: Any) -> None:
        outcome, _, _, probe = await self._run(db_session, _judge_reply("PARAPHRASE"))
        assert outcome.carried == {} and outcome.shadow == 1
        probe.assert_not_awaited()

    async def test_a_changed_value_is_planned_as_lineage(self, db_session: Any) -> None:
        changed = UpdateGuard(verdict=SlotVerdict.CHANGES, any_changes=True)
        outcome, x, _, _ = await self._run(db_session, _judge_reply("DISTINCT"), guard=changed)
        assert outcome.carried == {} and outcome.lineage == {0: x}
        assert outcome.slot_verdicts == {(PODCAST, PODCAST_REWORDED): SlotVerdict.CHANGES}

    async def test_an_abstention_asks_nothing_more(self, db_session: Any) -> None:
        outcome, _, slot, _ = await self._run(db_session, _judge_reply("UNSURE"))
        assert outcome.carried == {} and outcome.lineage == {}
        slot.assert_not_awaited()

    async def test_the_verdict_is_remembered(self, db_session: Any) -> None:
        first = _judge_reply("PARAPHRASE")
        await self._run(db_session, first)
        second = _judge_reply("DISTINCT")
        outcome, _, _, _ = await self._run(db_session, second)
        first.assert_awaited_once()
        second.assert_not_awaited()
        assert outcome.shadow == 1  # the recorded SAME, not a fresh DISTINCT

    async def test_pairs_over_the_cap_are_unasked(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().generation.restatement, "max_judged", 0)
        judge = _judge_reply()
        outcome, _, _, _ = await self._run(db_session, judge)
        assert outcome.unasked == 1 and outcome.judged == 0
        judge.assert_not_awaited()

    async def test_off_matches_nothing(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().generation.restatement, "enabled", False)
        judge = _judge_reply("PARAPHRASE")
        outcome, _, _, _ = await self._run(db_session, judge)
        assert outcome == RestatementOutcome()
        judge.assert_not_awaited()


# ---------------------------------------------------------------------------
# The pipeline: two generations of one note
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
                    subjects=["the podcast"],
                )
                for claim in self.claims[content]
            ]
        )
        self.results.append(result)
        return result


class _Note:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.extractor = _Scripted()
        self.uri = f"file:///vault/{uuid.uuid4()}.md"
        self.entry_id: str | None = None

    async def say(self, claims: list[str], *, day: int) -> SnapshotOutcome:
        from particles.corpus.deposit import deposit_text_versioned
        from particles.ingest.pipeline import extract_snapshot

        text = f"version of day {day}: " + " ".join(claims)
        self.extractor.claims[text.encode()] = claims
        entry_id, snapshot_id, unchanged = await deposit_text_versioned(
            self.session,
            text=text,
            uri_r=self.uri,
            source_type="LOCAL_MARKDOWN",
            mutability="MUTABLE",
            content_published_at=T0 + timedelta(days=day),
        )
        assert not unchanged
        self.entry_id = entry_id
        await self.session.commit()
        outcome = SnapshotOutcome()
        await extract_snapshot(
            self.session, entry_id, snapshot_id, extractor=self.extractor, outcome_out=outcome
        )
        await self.session.commit()
        return outcome

    def notes(self, key: str) -> list[str]:
        return [n for r in self.extractor.results for n in r.quality_notes if n.startswith(key)]


@pytest.fixture
def scripted() -> Generator[dict[str, Any], None, None]:
    """The judge, the update readings and the contradiction probe, all scripted."""
    from particles import embeddings as ep

    script: dict[str, Any] = {
        "judge": "PARAPHRASE",
        "guard": _QUIET,
    }

    async def judge(prompt: str, *, max_tokens: int) -> str:
        return json.dumps({"p0": script["judge"]})

    async def slot(earlier: str, later: str) -> UpdateGuard:
        guard: UpdateGuard = script["guard"]
        return guard

    original = ep._embedding_model
    ep.set_embedding_model(_Constant())  # type: ignore[arg-type]
    try:
        with (
            patch(
                "particles.ingest.pipeline._restatement_judge_call", AsyncMock(side_effect=judge)
            ),
            patch("particles.ingest.pipeline._restatement_slot", AsyncMock(side_effect=slot)),
            patch(
                "particles.ingest.pipeline._has_contradiction_signal",
                AsyncMock(return_value=False),
            ),
            patch(
                "particles.ingest.pipeline._has_update_signal",
                AsyncMock(return_value=SlotVerdict.DIFFERENT),
            ),
            patch(
                "particles.ingest.pipeline.read_pair",
                AsyncMock(return_value=ProbeVerdict(usable=True, contradicts=False)),
            ),
        ):
            yield script
    finally:
        ep.set_embedding_model(original)


async def _all(session: Any, content: str) -> list[Particle]:
    from particles.store.particle_store import get_particles_by_status

    return [
        p
        for status in Status
        for p in await get_particles_by_status(session, status)
        if p.content == content
    ]


@pytest.mark.asyncio
class TestPipeline:
    async def test_an_admitted_restatement_keeps_the_earlier_claim(
        self, db_session: Any, scripted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().generation.restatement, "judge_admitted", True)
        note = _Note(db_session)
        await note.say([PODCAST], day=1)
        outcome = await note.say([PODCAST_REWORDED], day=2)

        [kept] = await _all(db_session, PODCAST)
        assert kept.status is Status.ACTIVE
        assert (kept.properties or {})[RESTATED_AS_KEY]["text"] == PODCAST_REWORDED
        assert len({r.snapshot_id for r in kept.provenance}) == 2
        assert await _all(db_session, PODCAST_REWORDED) == []
        assert outcome.restated == 1
        assert note.notes(RESTATED) == [f"{RESTATED}: 1 claim(s) carried into the new generation"]

    async def test_before_admission_the_pass_writes_as_before(
        self, db_session: Any, scripted: dict[str, Any]
    ) -> None:
        note = _Note(db_session)
        await note.say([PODCAST], day=1)
        outcome = await note.say([PODCAST_REWORDED], day=2)

        [old] = await _all(db_session, PODCAST)
        [new] = await _all(db_session, PODCAST_REWORDED)
        assert old.status is Status.PROVENANCE_STALE
        assert old.status_reason is StatusReason.RETRACTED_DEPENDENCY
        assert new.status is Status.ACTIVE
        assert outcome.restated == 0 and outcome.restatement_shadow == 1

    async def test_a_changed_value_records_lineage(
        self, db_session: Any, scripted: dict[str, Any]
    ) -> None:
        """The match's update reading reaches rung 2.5 too, so the pair is asked once.

        Here rung 2.5 takes the pair in the write loop (the match handed it the
        reading); the generation route writes the same lineage when it does not,
        which :class:`TestLineageWrite` pins.
        """
        scripted["judge"] = "DISTINCT"
        scripted["guard"] = UpdateGuard(verdict=SlotVerdict.CHANGES, any_changes=True)
        note = _Note(db_session)
        await note.say([PODCAST], day=1)
        await note.say([WEEKLY], day=2)

        [old] = await _all(db_session, PODCAST)
        [new] = await _all(db_session, WEEKLY)
        assert old.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        assert new.status is Status.ACTIVE and new.supersedes == old.id

    async def test_an_exact_twin_is_still_an_adr_0211_suppression(
        self, db_session: Any, scripted: dict[str, Any]
    ) -> None:
        note = _Note(db_session)
        await note.say([PODCAST], day=1)
        outcome = await note.say([PODCAST.replace('"', "")], day=2)

        [kept] = await _all(db_session, PODCAST)
        assert kept.status is Status.ACTIVE
        assert outcome.restated == 0  # disclosed as a suppression, not a restatement
        assert RESTATED_AS_KEY not in (kept.properties or {})

    async def test_off_restores_the_old_churn(
        self, db_session: Any, scripted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().generation.restatement, "enabled", False)
        monkeypatch.setattr(get_config().generation.restatement, "judge_admitted", True)
        note = _Note(db_session)
        await note.say([PODCAST], day=1)
        await note.say([PODCAST_REWORDED], day=2)

        [old] = await _all(db_session, PODCAST)
        assert old.status is Status.PROVENANCE_STALE
        assert len(await _all(db_session, PODCAST_REWORDED)) == 1


@pytest.mark.asyncio
class TestLineageWrite:
    async def _pair(self, session: Any) -> tuple[Particle, Particle]:
        from particles.store.particle_store import insert_particle

        old, new = _claim(PODCAST), _claim(WEEKLY)
        await insert_particle(session, old)
        await insert_particle(session, new)
        await session.flush()
        return old, new

    async def test_the_generation_route_retires_and_names(self, db_session: Any) -> None:
        from particles.ingest.pipeline import RESTATEMENT_ROUTE, _apply_restatement_lineage
        from particles.store.event_store import OperatorEventType, list_events
        from particles.store.particle_store import get_particle

        old, new = await self._pair(db_session)
        assert await _apply_restatement_lineage(db_session, old.id, new.id)
        retired, winner = (
            await get_particle(db_session, old.id),
            await get_particle(db_session, new.id),
        )
        assert retired is not None and winner is not None
        assert retired.status is Status.PROVENANCE_STALE
        assert retired.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        assert winner.supersedes == old.id
        events = await list_events(db_session, event_type=OperatorEventType.UPDATE_SUPERSESSION)
        assert [(e.payload or {}).get("route") for e in events] == [RESTATEMENT_ROUTE]

    async def test_nothing_when_the_pair_is_already_settled(self, db_session: Any) -> None:
        from particles.ingest.pipeline import _apply_restatement_lineage
        from particles.store.particle_store import set_supersedes_if_unset

        old, new = await self._pair(db_session)
        other = _claim("Another predecessor.")
        assert await set_supersedes_if_unset(db_session, new.id, other.id)
        assert not await _apply_restatement_lineage(db_session, old.id, new.id)
        assert not await _apply_restatement_lineage(db_session, old.id, old.id)
