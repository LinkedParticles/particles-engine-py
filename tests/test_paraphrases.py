# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The paraphrases pass: an update retires the other phrasings of the value.

Built on the trip-versus-move survivor: the sweep retired "The user lives in
Delhi, in Lajpat Nagar." in favour of the move to Mumbai, and "The user's home
is in Delhi." stayed ACTIVE because the contradiction probe it never passed
samples. Each world here is built straight into the store, with embeddings
chosen so every cosine to the retired claim is exact, and the update probe is
scripted at the pass's seam (``paraphrases._read_update_slot``), per
tests/AGENTS.md § Mocking strategy.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from particles.config import get_config
from particles.core.conflict_resolution import SlotVerdict
from particles.core.observer_scope import PairPrecondition
from particles.core.probe_verdict import ProbeKind, verdict_key
from particles.core.reanchor import Cursor
from particles.core.schema import (
    Confidence,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    SourceType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.ingest.pipeline import SlotReading, update_decided_by
from particles.store.event_store import OperatorEventType

T0 = datetime(2026, 9, 1, tzinfo=UTC)
DIM = 16

RESIDENCE = "The user lives in Delhi, in Lajpat Nagar."
HOME = "The user's home is in Delhi."
MOVE = "The user decided to move to Mumbai permanently."
CURRY = "Sandeep's Curry House is located near Lajpat Nagar."


def _vec(cos_to_r: float, axis: int) -> list[float]:
    """A unit vector at exactly ``cos_to_r`` to the retired claim's (axis 0)."""
    out = [0.0] * DIM
    out[0] = cos_to_r
    out[axis] = math.sqrt(max(0.0, 1.0 - cos_to_r**2))
    return out


def _axis(axis: int) -> list[float]:
    out = [0.0] * DIM
    out[axis] = 1.0
    return out


class _World:
    """Claims written straight into the store, one corpus entry per claim."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.axis = 1

    def next_axis(self) -> int:
        self.axis += 1
        return self.axis

    async def claim(
        self,
        content: str,
        *,
        day: int,
        embedding: list[float],
        subject: str = "user",
        subjects: list[str] | None = None,
        nature: UncertaintyNature = UncertaintyNature.EPISTEMIC,
        authority: str = "claude-code://session",
        source_type: str = SourceType.CONVERSATION,
    ) -> Particle:
        from particles.corpus.deposit import deposit_text_versioned
        from particles.store.particle_store import insert_particle

        entry_id, snapshot_id, _ = await deposit_text_versioned(
            self.session,
            text=f"Day {day}: {content} ({uuid.uuid4()})",
            uri_r=f"{authority}/{uuid.uuid4()}",
            source_type=source_type,
            mutability=Mutability.STABLE,
            content_published_at=T0 + timedelta(days=day),
        )
        particle = Particle(
            id=str(uuid.uuid4()),
            content=content,
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=nature,
            asserted_by="general-extractor",
            asserted_at=T0 + timedelta(days=day),
            subject_ids=subjects if subjects is not None else [subject],
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=entry_id,
                    snapshot_id=snapshot_id,
                )
            ],
        )
        await insert_particle(self.session, particle, embedding=embedding)
        await self.session.commit()
        return particle

    async def retire(self, retired: Particle, winner: Particle, *, record: str = "event") -> None:
        """Retire ``retired`` by update, its winner on record as ``record`` says.

        ``event`` writes the ``UPDATE_SUPERSESSION`` event (1.178.4 on);
        ``supersedes`` only points the winner at it (before the event existed);
        ``none`` leaves no trace of the winner.
        """
        from particles.ingest.conflict_plan import plan_update_event
        from particles.store.event_store import record_event
        from particles.store.particle_store import ParticleRow, update_particle_status

        await update_particle_status(
            self.session,
            retired.id,
            Status.PROVENANCE_STALE,
            StatusReason.SUPERSEDED_BY_UPDATE,
        )
        if record == "event":
            event = plan_update_event(retired.id, winner.id, route="sweep")
            await record_event(
                self.session,
                actor=event.actor,
                event_type=event.event_type,
                reason=event.reason,
                refs=list(event.refs),
                payload=dict(event.payload),
            )
        elif record == "supersedes":
            row = await self.session.get(ParticleRow, winner.id)
            row.supersedes = retired.id
        await self.session.commit()


async def _trip(session: Any, *, home_cos: float = 0.785) -> tuple[_World, dict[str, Particle]]:
    """The survivor's world: residence retired by the move, home and a mention ACTIVE."""
    world = _World(session)
    claims = {
        "residence": await world.claim(RESIDENCE, day=4, embedding=_axis(0)),
        "home": await world.claim(HOME, day=11, embedding=_vec(home_cos, world.next_axis())),
        "curry": await world.claim(CURRY, day=4, embedding=_vec(0.51, world.next_axis())),
        "move": await world.claim(MOVE, day=12, embedding=_vec(0.40, world.next_axis())),
    }
    await world.retire(claims["residence"], claims["move"])
    return world, claims


Verdicts = dict[str, SlotVerdict | None]


class _Probe:
    """The scripted update probe: a verdict per earlier claim, CHANGES by default."""

    def __init__(self, verdicts: Verdicts | None = None) -> None:
        self.verdicts = verdicts or {}
        self.asked: list[tuple[str, str]] = []

    async def __call__(self, earlier: str, later: str) -> SlotReading:
        self.asked.append((earlier, later))
        verdict = self.verdicts.get(earlier, SlotVerdict.CHANGES)
        purpose = "verification" if verdict is SlotVerdict.CHANGES else "semantic_lint"
        return SlotReading(verdict, purpose)  # type: ignore[arg-type]


@pytest.fixture
def probe() -> Generator[Callable[..., _Probe], None, None]:
    """Script the update probe, and fail if any contradiction probe is asked."""
    contradiction = AsyncMock(side_effect=AssertionError("no contradiction probe"))
    holder: list[_Probe] = []

    def make(verdicts: Verdicts | None = None) -> _Probe:
        scripted = _Probe(verdicts)
        holder.append(scripted)
        return scripted

    async def read(earlier: str, later: str) -> SlotReading:
        return await holder[-1](earlier, later)

    with (
        patch("particles.operations.paraphrases._read_update_slot", read),
        patch("particles.operations.reconcile._has_contradiction_signal", contradiction),
        patch("particles.ingest.pipeline._has_contradiction_signal", contradiction),
    ):
        yield make
    assert contradiction.await_count == 0


async def _status(session: Any, particle: Particle) -> tuple[Status, StatusReason | None]:
    from particles.store.particle_store import get_particle

    found = await get_particle(session, particle.id)
    assert found is not None
    return found.status, found.status_reason


async def _update_events(session: Any) -> list[dict[str, Any]]:
    from particles.store.event_store import list_events

    events = await list_events(session, event_type=OperatorEventType.UPDATE_SUPERSESSION, limit=100)
    return [e.payload or {} for e in events]


async def _run(session: Any, cursor: Cursor | None = None, **kwargs: Any) -> Any:
    from particles.operations.paraphrases import run_paraphrases

    report = await run_paraphrases(session, cursor=cursor, **kwargs)
    await session.commit()
    return report


# ---------------------------------------------------------------------------
# The retirement
# ---------------------------------------------------------------------------


class TestTheSurvivor:
    async def test_two_changes_retire_the_other_phrasing(
        self, db_session: Any, probe: Callable[..., _Probe], candidate_cache_mode: str
    ) -> None:
        _, c = await _trip(db_session)
        scripted = probe()
        report = await _run(db_session)

        assert scripted.asked == [(HOME, MOVE)]
        assert await _status(db_session, c["home"]) == (
            Status.PROVENANCE_STALE,
            StatusReason.SUPERSEDED_BY_UPDATE,
        )
        assert (await _status(db_session, c["curry"]))[0] is Status.ACTIVE
        assert report.probes == 1 and report.readings == 1
        assert report.retired == [
            {
                "retired": c["home"].id,
                "winner": c["move"].id,
                "trigger": c["residence"].id,
                "similarity": 0.785,
            }
        ]
        [event] = [e for e in await _update_events(db_session) if e["route"] == "paraphrase"]
        assert event == {
            "retired_id": c["home"].id,
            "winner_id": c["move"].id,
            "route": "paraphrase",
            "decided_by": update_decided_by(),
        }

    @pytest.mark.parametrize("verdict", [SlotVerdict.DIFFERENT, SlotVerdict.FIXED])
    async def test_another_verdict_leaves_it_and_ledgers_it(
        self, db_session: Any, probe: Callable[..., _Probe], verdict: SlotVerdict
    ) -> None:
        from particles.ingest.pipeline import update_prompt_hash
        from particles.operations.probe_ledger import recall

        _, c = await _trip(db_session)
        probe({HOME: verdict})
        report = await _run(db_session)

        assert (await _status(db_session, c["home"]))[0] is Status.ACTIVE
        assert report.kept == [
            {
                "particle": c["home"].id,
                "winner": c["move"].id,
                "trigger": c["residence"].id,
                "verdict": verdict.value,
            }
        ]
        key = verdict_key(ProbeKind.UPDATE_SLOT, HOME, MOVE)
        assert await recall(db_session, ProbeKind.UPDATE_SLOT, update_prompt_hash(), [key]) == {
            key: False
        }
        # The cleared pair is never asked again, even from the start.
        again = probe()
        rerun = await _run(db_session)
        assert again.asked == [] and rerun.previously_cleared == 1 and rerun.probes == 0

    async def test_an_unusable_reply_is_counted_and_passed(
        self, db_session: Any, probe: Callable[..., _Probe]
    ) -> None:
        _, c = await _trip(db_session)
        probe({HOME: None})
        report = await _run(db_session)

        assert (await _status(db_session, c["home"]))[0] is Status.ACTIVE
        assert report.unanswered == 1 and report.triggers == 1
        assert report.cursor is not None and report.cursor["particle_id"] == c["residence"].id

    async def test_an_unreachable_model_holds_the_cursor(
        self, db_session: Any, probe: Callable[..., _Probe], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations import paraphrases

        monkeypatch.setattr(paraphrases, "llm_circuit_open", lambda: True)
        _, c = await _trip(db_session)
        probe({HOME: None})
        report = await _run(db_session)

        assert (await _status(db_session, c["home"]))[0] is Status.ACTIVE
        assert report.cursor is None and report.triggers == 0
        assert report.warnings and "retried next run" in report.warnings[0]


# ---------------------------------------------------------------------------
# Candidacy
# ---------------------------------------------------------------------------


class TestCandidacy:
    async def test_below_the_floor_is_not_asked(
        self, db_session: Any, probe: Callable[..., _Probe], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _trip(db_session, home_cos=0.69)
        scripted = probe()
        report = await _run(db_session)
        assert scripted.asked == [] and report.above_floor == 0

        monkeypatch.setattr(
            get_config().reconciliation.update_supersession, "paraphrase_floor", 0.60
        )
        await _run(db_session)
        assert scripted.asked == [(HOME, MOVE)]

    async def test_at_most_max_paraphrases_highest_cosine_first(
        self, db_session: Any, probe: Callable[..., _Probe], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "max_paraphrases", 2)
        world, c = await _trip(db_session, home_cos=0.75)
        for content, cos in [("The user resides in Delhi.", 0.9), ("Delhi is home.", 0.8)]:
            await world.claim(content, day=10, embedding=_vec(cos, world.next_axis()))
        scripted = probe()
        report = await _run(db_session)

        assert [e for e, _ in scripted.asked] == ["The user resides in Delhi.", "Delhi is home."]
        assert report.above_floor == 3 and report.candidates == 2
        assert (await _status(db_session, c["home"]))[0] is Status.ACTIVE

    async def test_another_lineage_is_not_asked(
        self, db_session: Any, probe: Callable[..., _Probe]
    ) -> None:
        world, _ = await _trip(db_session, home_cos=0.60)
        await world.claim(
            "Our user's residence is Delhi.",
            day=10,
            embedding=_vec(0.9, world.next_axis()),
            authority="https://example.com/profile",
            source_type=SourceType.WEB_PAGE,
        )
        scripted = probe()
        report = await _run(db_session)
        assert scripted.asked == [] and report.unordered == 1

    async def test_a_phrasing_newer_than_the_winner_is_not_asked(
        self, db_session: Any, probe: Callable[..., _Probe]
    ) -> None:
        world, _ = await _trip(db_session, home_cos=0.60)
        await world.claim("Delhi is the user's home.", day=30, embedding=_vec(0.9, 9))
        scripted = probe()
        report = await _run(db_session)
        assert scripted.asked == [] and report.unordered == 1

    async def test_another_subject_is_not_asked(
        self, db_session: Any, probe: Callable[..., _Probe]
    ) -> None:
        world, _ = await _trip(db_session, home_cos=0.60)
        await world.claim("Delhi is home to many.", day=10, embedding=_vec(0.9, 9), subject="delhi")
        scripted = probe()
        report = await _run(db_session)
        assert scripted.asked == [] and report.above_floor == 0

    @pytest.mark.parametrize("side", ["candidate", "winner"])
    async def test_only_an_aleatory_winner_keeps_it_out(
        self, db_session: Any, probe: Callable[..., _Probe], side: str
    ) -> None:
        """the newer side decides; an older ALEATORY phrasing may be retired."""
        world = _World(db_session)
        aleatory, epistemic = UncertaintyNature.ALEATORY, UncertaintyNature.EPISTEMIC
        residence = await world.claim(RESIDENCE, day=4, embedding=_axis(0))
        home = await world.claim(
            HOME,
            day=11,
            embedding=_vec(0.785, 2),
            nature=aleatory if side == "candidate" else epistemic,
        )
        move = await world.claim(
            MOVE, day=12, embedding=_axis(3), nature=aleatory if side == "winner" else epistemic
        )
        await world.retire(residence, move)
        scripted = probe()
        report = await _run(db_session)

        if side == "winner":
            assert scripted.asked == [] and report.aleatory_excluded == 1
            assert (await _status(db_session, home))[0] is Status.ACTIVE
        else:
            assert scripted.asked == [(HOME, MOVE)] and report.aleatory_excluded == 0
            assert (await _status(db_session, home))[0] is Status.PROVENANCE_STALE

    async def test_the_observer_precondition_keeps_it_out(
        self, db_session: Any, probe: Callable[..., _Probe], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations import paraphrases

        class _Gate:
            engaged = True

            @classmethod
            async def open(cls, session: Any) -> _Gate:
                return cls()

            async def scope_of(self, session: Any, particle: Particle) -> object:
                return object()

            async def verdict(self, session: Any, scope: object, existing: Particle) -> Any:
                return PairPrecondition.DECLINE

        monkeypatch.setattr(paraphrases, "ObserverGate", _Gate)
        _, c = await _trip(db_session)
        scripted = probe()
        report = await _run(db_session)

        assert scripted.asked == [] and report.observer_declined == 1
        assert (await _status(db_session, c["home"]))[0] is Status.ACTIVE


# ---------------------------------------------------------------------------
# The winner, the cascade and the cap
# ---------------------------------------------------------------------------


class TestWalk:
    async def test_a_retirement_before_the_event_falls_back_to_supersedes(
        self, db_session: Any, probe: Callable[..., _Probe]
    ) -> None:
        world = _World(db_session)
        residence = await world.claim(RESIDENCE, day=4, embedding=_axis(0))
        home = await world.claim(HOME, day=11, embedding=_vec(0.785, 2))
        move = await world.claim(MOVE, day=12, embedding=_axis(3))
        await world.retire(residence, move, record="supersedes")
        scripted = probe()
        await _run(db_session)

        assert scripted.asked == [(HOME, MOVE)]
        assert (await _status(db_session, home))[0] is Status.PROVENANCE_STALE

    async def test_a_retirement_with_no_winner_is_skipped(
        self, db_session: Any, probe: Callable[..., _Probe]
    ) -> None:
        world = _World(db_session)
        residence = await world.claim(RESIDENCE, day=4, embedding=_axis(0))
        home = await world.claim(HOME, day=11, embedding=_vec(0.785, 2))
        move = await world.claim(MOVE, day=12, embedding=_axis(3))
        await world.retire(residence, move, record="none")
        scripted = probe()
        report = await _run(db_session)

        assert scripted.asked == [] and report.no_winner == 1 and report.triggers == 1
        assert report.cursor is not None and report.cursor["particle_id"] == residence.id
        assert (await _status(db_session, home))[0] is Status.ACTIVE

    async def test_a_retirement_made_here_has_its_own_phrasings_examined(
        self, db_session: Any, probe: Callable[..., _Probe]
    ) -> None:
        world = _World(db_session)
        residence = await world.claim(RESIDENCE, day=4, embedding=_axis(0))
        # Near the residence claim; and a third phrasing near it but not the residence.
        home = await world.claim(HOME, day=11, embedding=[0.8, 0.6] + [0.0] * (DIM - 2))
        third = await world.claim(
            "The user's address is in Delhi.",
            day=11,
            embedding=[0.3, 0.954] + [0.0] * (DIM - 2),
        )
        move = await world.claim(MOVE, day=12, embedding=_axis(3))
        await world.retire(residence, move)
        scripted = probe()
        report = await _run(db_session)

        assert scripted.asked == [(HOME, MOVE), ("The user's address is in Delhi.", MOVE)]
        assert [r["trigger"] for r in report.retired] == [residence.id, home.id]
        assert all(r["winner"] == move.id for r in report.retired)
        assert (await _status(db_session, third))[0] is Status.PROVENANCE_STALE

    async def test_the_run_cap_holds_the_cursor_back(
        self, db_session: Any, probe: Callable[..., _Probe], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().consolidation, "max_paraphrase_probes", 2)
        world = _World(db_session)
        first = await world.claim(RESIDENCE, day=4, embedding=_axis(0))
        a = await world.claim(HOME, day=11, embedding=_vec(0.8, 1))
        b = await world.claim("The user resides in Delhi.", day=11, embedding=_vec(0.8, 2))
        move = await world.claim(MOVE, day=12, embedding=_axis(3))
        second = await world.claim("The user works at Infosys.", day=4, embedding=_axis(4))
        c_emb = [0.0] * DIM
        c_emb[4], c_emb[5] = 0.9, math.sqrt(1 - 0.81)
        c = await world.claim("The user has a job at Infosys.", day=11, embedding=c_emb)
        job = await world.claim("The user joined Wipro.", day=12, embedding=_axis(6))
        await world.retire(first, move)
        await world.retire(second, job)
        scripted = probe()

        capped = await _run(db_session)
        assert capped.capped and capped.probes == 2 and capped.triggers == 1
        assert capped.cursor is not None and capped.cursor["particle_id"] == first.id
        assert {await _status(db_session, p) for p in (a, b)} == {
            (Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE)
        }
        assert (await _status(db_session, c))[0] is Status.ACTIVE

        resumed = await _run(db_session, Cursor.from_payload(capped.cursor))
        assert scripted.asked[-1] == ("The user has a job at Infosys.", "The user joined Wipro.")
        assert (await _status(db_session, c))[0] is Status.PROVENANCE_STALE
        assert not resumed.capped

    async def test_max_paraphrases_zero_is_a_no_op(
        self, db_session: Any, probe: Callable[..., _Probe], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "max_paraphrases", 0)
        _, c = await _trip(db_session)
        scripted = probe()
        report = await _run(db_session)

        assert scripted.asked == [] and not report.enabled
        assert report.skipped_reason == "reconciliation.update_supersession.max_paraphrases is 0"
        assert (await _status(db_session, c["home"]))[0] is Status.ACTIVE

    async def test_a_dry_run_asks_and_writes_nothing(
        self, db_session: Any, probe: Callable[..., _Probe]
    ) -> None:
        _, c = await _trip(db_session)
        scripted = probe()
        report = await _run(db_session, dry_run=True)

        assert scripted.asked == [] and report.probes == 1 and report.candidates == 1
        assert report.cursor is None
        assert (await _status(db_session, c["home"]))[0] is Status.ACTIVE


# ---------------------------------------------------------------------------
# The cycle, the plan and the store
# ---------------------------------------------------------------------------


class TestCycle:
    def test_the_pass_runs_between_the_sweep_and_re_anchoring(self) -> None:
        from particles.operations.consolidation import PASS_ORDER

        position = PASS_ORDER.index("paraphrases")
        assert PASS_ORDER[position - 1] == "reconcile_updates"
        assert PASS_ORDER[position + 1] == "reanchor"

    async def test_its_cursor_is_read_from_its_own_block(self, db_session: Any) -> None:
        from particles.operations.reanchor import prior_cursor
        from particles.store.event_store import record_event

        cursor = {"retired_at": T0.isoformat(), "particle_id": "p-1"}
        await record_event(
            db_session,
            actor="memory-consolidate",
            event_type=OperatorEventType.CONSOLIDATION_RUN,
            reason="run",
            refs=[],
            payload={"census": {"paraphrases": {"cursor": cursor}}},
        )
        await db_session.commit()
        found = await prior_cursor(db_session, "memory-consolidate", "paraphrases")
        assert found == Cursor(retired_at=T0, particle_id="p-1")
        assert await prior_cursor(db_session, "memory-consolidate") is None

    async def test_the_plan_reports_the_estimate(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations import consolidation_plan

        monkeypatch.setattr(consolidation_plan, "_semantic_availability", lambda _: (True, None))
        await _trip(db_session)
        plan = await consolidation_plan.plan_consolidation(db_session)
        [entry] = [p for p in plan.passes if p.name == "paraphrases"]

        assert entry.action == "run" and entry.llm_priced
        assert entry.calls == 2  # one first reading, and at most a second
        assert "would probe 1" in entry.detail and "1 claim above" in entry.detail

    async def test_a_degraded_store_skips_it(self, db_session: Any) -> None:
        from particles.operations.consolidation_plan import plan_consolidation

        plan = await plan_consolidation(db_session, structural_only=True)
        [entry] = [p for p in plan.passes if p.name == "paraphrases"]
        assert entry.action == "skip" and "LLM-priced" in entry.detail

    async def test_reconcile_updates_runs_it_after_the_sweep(
        self, db_session: Any, probe: Callable[..., _Probe]
    ) -> None:
        from particles.api.cli.reconcile import _paraphrases

        _, c = await _trip(db_session)
        probe()
        out = await _paraphrases(db_session, dry_run=False, progress=None)
        assert [r["retired"] for r in out["retired"]] == [c["home"].id]  # type: ignore[index]


class TestStore:
    async def test_embeddings_by_id_include_retired_claims(self, db_session: Any) -> None:
        from particles.store.particle_store import ParticleRow, get_embeddings_by_ids

        _, c = await _trip(db_session)
        row = await db_session.get(ParticleRow, c["curry"].id)
        row.embedding_model_id = "another-model"
        await db_session.commit()

        found = await get_embeddings_by_ids(
            db_session, [c["residence"].id, c["curry"].id, "missing"]
        )
        assert set(found) == {c["residence"].id}
        assert found[c["residence"].id][0] == pytest.approx(1.0)


class TestTheSpeakerKey:
    """the paraphrase search reads the widened key."""

    async def _world(self, session: Any) -> None:
        from particles.core.schema import Subject
        from particles.store.subject_store import insert_subject

        persona = Subject(canonical_name="the user", asserted_by="test")
        await insert_subject(session, persona)
        world = _World(session)
        # Two subjects and no triple: the residence's about key is empty, so
        # only the speaker key reaches the other phrasing.
        residence = await world.claim(
            RESIDENCE, day=4, embedding=_axis(0), subjects=["delhi", persona.id]
        )
        await world.claim(
            HOME, day=11, embedding=_vec(0.785, world.next_axis()), subject=persona.id
        )
        move = await world.claim(
            MOVE, day=12, embedding=_vec(0.40, world.next_axis()), subject=persona.id
        )
        await world.retire(residence, move)

    async def test_a_speaker_anchored_retirement_finds_its_other_phrasing(
        self, db_session: Any, probe: Callable[..., _Probe]
    ) -> None:
        await self._world(db_session)
        scripted = probe()
        await _run(db_session)
        assert scripted.asked == [(HOME, MOVE)]

    async def test_without_the_speaker_key_it_finds_none(
        self, db_session: Any, probe: Callable[..., _Probe], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().subjects, "persona_openers", [])
        await self._world(db_session)
        scripted = probe()
        await _run(db_session)
        assert scripted.asked == []
