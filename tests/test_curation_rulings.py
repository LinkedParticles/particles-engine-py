# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Demotion cards and the rulings they record.

A ``demotion`` card shows a claim a later claim retired as its replacement.
Affirming it (the replacement was right) or dismissing it (both hold) appends
one labelled pair to ``<benchmark.runs_dir>/demotion-rulings.jsonl``; snooze
records nothing, and no gesture changes either claim's status.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.duplicate_key import content_hash
from particles.core.probe_verdict import ProbeKind, verdict_key
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.curation import (
    CardKind,
    CurationCard,
    apply_gesture,
    build_curation_queue,
    load_rulings,
    rulings_path,
)
from particles.operations.curation.cards import describe_gesture
from particles.store.particle_store import (
    get_particle,
    get_recorded_demotions,
    insert_particle,
    update_particle_status,
)
from particles.store.probe_verdict_store import record_verdict, verdicts_for_pair
from particles.store.subject_store import insert_subject

_OLD = "The Kawai ES110 costs $299."
_NEW = "The Kawai ES110 costs $699 to $799."


def _particle(
    content: str,
    *,
    subject_id: str,
    supersedes: str | None = None,
) -> Particle:
    fields: dict[str, object] = {
        "content": content,
        "confidence": Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        "uncertainty_nature": UncertaintyNature.EPISTEMIC,
        "provenance": [
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id=None)
        ],
        "asserted_by": "general-extractor",
        "subject_ids": [subject_id],
        "supersedes": supersedes,
    }
    return Particle(**fields)  # type: ignore[arg-type]


async def _seed_demotion(
    session: AsyncSession, reason: StatusReason = StatusReason.SUPERSEDED_BY_UPDATE
) -> tuple[Particle, Particle]:
    subject = Subject(canonical_name="Kawai ES110", asserted_by="test")
    await insert_subject(session, subject)
    retired = _particle(_OLD, subject_id=subject.id)
    replacement = _particle(_NEW, subject_id=subject.id, supersedes=retired.id)
    await insert_particle(session, retired)
    await insert_particle(session, replacement)
    await _demote(session, retired.id, reason)
    return retired, replacement


async def _demote(session: AsyncSession, particle_id: str, reason: StatusReason) -> None:
    status = (
        Status.SUPERSEDED
        if reason is StatusReason.SUPERSEDED_BY_REANCHOR
        else (Status.PROVENANCE_STALE)
    )
    await update_particle_status(session, particle_id, status, reason)
    await session.flush()


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    get_config().benchmark.runs_dir = str(tmp_path / "runs")
    return tmp_path / "runs"


async def _demotion_card(session: AsyncSession) -> CurationCard:
    cards = (await build_curation_queue(session, semantic=False)).cards
    [card] = [c for c in cards if c.kind is CardKind.DEMOTION]
    return card


class TestRecordedDemotions:
    async def test_lists_each_retired_claim_with_its_replacement(
        self, db_session: AsyncSession
    ) -> None:
        retired, replacement = await _seed_demotion(db_session)
        [(got_retired, got_replacement)] = await get_recorded_demotions(db_session)
        assert (got_retired.id, got_replacement.id) == (retired.id, replacement.id)

    async def test_a_demotion_with_no_recorded_replacement_is_not_listed(
        self, db_session: AsyncSession
    ) -> None:
        subject = Subject(canonical_name="Kawai ES110", asserted_by="test")
        await insert_subject(db_session, subject)
        orphan = _particle(_OLD, subject_id=subject.id)
        # A retraction cascade names no replacement either, even when linked.
        cascade = _particle("Another claim.", subject_id=subject.id)
        linked = _particle("Its successor.", subject_id=subject.id, supersedes=cascade.id)
        for p in (orphan, cascade, linked):
            await insert_particle(db_session, p)
        await _demote(db_session, orphan.id, StatusReason.SUPERSEDED_BY_UPDATE)
        await _demote(db_session, cascade.id, StatusReason.RETRACTED_DEPENDENCY)
        assert await get_recorded_demotions(db_session) == []


class TestDemotionCard:
    async def test_the_queue_carries_the_pair_retired_first(self, db_session: AsyncSession) -> None:
        retired, replacement = await _seed_demotion(db_session)
        card = await _demotion_card(db_session)
        assert card.particle_ids == [retired.id, replacement.id]
        assert card.suggested_gestures == ["affirm", "dismiss", "snooze"]
        assert "SUPERSEDED_BY_UPDATE" in card.diagnostic

    async def test_the_card_leaves_the_queue_once_the_replacement_is_gone(
        self, db_session: AsyncSession
    ) -> None:
        _, replacement = await _seed_demotion(db_session)
        await update_particle_status(
            db_session, replacement.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.flush()
        cards = (await build_curation_queue(db_session, semantic=False)).cards
        assert not [c for c in cards if c.kind is CardKind.DEMOTION]

    def test_the_help_discloses_the_fixture_only_while_recording_is_on(self) -> None:
        card = CurationCard(kind=CardKind.DEMOTION, particle_ids=["a", "b"], diagnostic="")
        assert "benchmark fixture" in describe_gesture(card, "dismiss", snooze_days=14)
        assert "benchmark fixture" not in describe_gesture(card, "snooze", snooze_days=14)
        get_config().benchmark.record_demotion_rulings = False
        assert "benchmark fixture" not in describe_gesture(card, "dismiss", snooze_days=14)


class TestRulings:
    async def test_dismiss_records_a_coexist_ruling_and_changes_no_status(
        self, db_session: AsyncSession, runs_dir: Path
    ) -> None:
        retired, replacement = await _seed_demotion(db_session)
        key = verdict_key(ProbeKind.UPDATE_SLOT, _OLD, _NEW)
        await record_verdict(db_session, ProbeKind.UPDATE_SLOT, "v1", key, True, model="m")
        card = await _demotion_card(db_session)

        msg = await apply_gesture(db_session, card, "dismiss", actor="tester")
        await db_session.commit()

        assert msg.endswith("Recorded as a benchmark fixture.")
        path = runs_dir / "demotion-rulings.jsonl"
        assert path == rulings_path()
        [line] = path.read_text().splitlines()
        record = json.loads(line)
        assert record["format"] == "particles.demotion-ruling/1"
        assert record["ruling"] == "coexist"
        assert record["reason"] == "SUPERSEDED_BY_UPDATE"
        assert record["retired"]["content"] == _OLD
        assert record["retired"]["content_hash"] == content_hash(_OLD)
        assert record["replacement"]["content"] == _NEW
        assert record["replacement"]["particle_id"] == replacement.id
        assert record["subjects"][0]["name"] == "Kawai ES110"
        assert record["actor"] == "tester"
        assert record["card_key"] == card.key
        [verdict] = record["probe_verdicts"]
        assert verdict["probe_kind"] == "update_slot" and verdict["verdict"] is True
        # A side effect, never a reversal: both statuses stand.
        after = await get_particle(db_session, retired.id)
        assert after is not None and after.status is Status.PROVENANCE_STALE

    async def test_affirm_records_a_replacement_ruling(
        self, db_session: AsyncSession, runs_dir: Path
    ) -> None:
        await _seed_demotion(db_session)
        card = await _demotion_card(db_session)
        msg = await apply_gesture(db_session, card, "affirm")
        assert "Recorded as a benchmark fixture." in msg
        [ruling], notes = load_rulings(rulings_path())
        assert ruling.ruling == "replacement" and notes == []

    async def test_snooze_records_nothing(self, db_session: AsyncSession, runs_dir: Path) -> None:
        await _seed_demotion(db_session)
        card = await _demotion_card(db_session)
        msg = await apply_gesture(db_session, card, "snooze", days=7)
        assert "fixture" not in msg
        assert not (runs_dir / "demotion-rulings.jsonl").exists()

    async def test_the_switch_turns_recording_off(
        self, db_session: AsyncSession, runs_dir: Path
    ) -> None:
        get_config().benchmark.record_demotion_rulings = False
        await _seed_demotion(db_session)
        card = await _demotion_card(db_session)
        msg = await apply_gesture(db_session, card, "dismiss")
        assert "fixture" not in msg
        assert not (runs_dir / "demotion-rulings.jsonl").exists()

    async def test_a_dismiss_on_any_other_card_records_nothing(
        self, db_session: AsyncSession, runs_dir: Path
    ) -> None:
        card = CurationCard(
            kind=CardKind.DUPLICATE_PAIR,
            particle_ids=["a", "b"],
            diagnostic="",
            suggested_gestures=["merge", "dismiss", "snooze"],
        )
        msg = await apply_gesture(db_session, card, "dismiss")
        assert "fixture" not in msg
        assert not (runs_dir / "demotion-rulings.jsonl").exists()

    async def test_a_card_rebuilt_from_its_key_records_the_pair_in_order(
        self, db_session: AsyncSession, runs_dir: Path
    ) -> None:
        # A key sorts its ids, so the recorder re-derives which claim was retired.
        retired, replacement = await _seed_demotion(db_session)
        card = CurationCard.from_key(f"demotion:{replacement.id}|{retired.id}")
        await apply_gesture(db_session, card, "dismiss")
        [ruling], _ = load_rulings(rulings_path())
        assert ruling.retired.particle_id == retired.id
        assert ruling.replacement.particle_id == replacement.id


class TestLoadRulings:
    def test_the_newest_ruling_per_pair_wins_and_bad_lines_are_skipped(
        self, tmp_path: Path
    ) -> None:
        def line(ruling: str, at: str) -> str:
            return json.dumps(
                {
                    "format": "particles.demotion-ruling/1",
                    "ruling": ruling,
                    "reason": "SUPERSEDED_BY_UPDATE",
                    "retired": {"particle_id": "r", "content_hash": "h1", "content": _OLD},
                    "replacement": {"particle_id": "n", "content_hash": "h2", "content": _NEW},
                    "actor": "curate",
                    "store": "default",
                    "recorded_at": at,
                    "card_key": "demotion:n|r",
                }
            )

        path = tmp_path / "rulings.jsonl"
        path.write_text(
            "\n".join(
                [
                    line("coexist", "2026-10-01T00:00:00+00:00"),
                    "not json",
                    json.dumps({"format": "something-else/9"}),
                    line("replacement", "2026-10-02T00:00:00+00:00"),
                ]
            )
            + "\n"
        )
        rulings, notes = load_rulings(path)
        assert [r.ruling for r in rulings] == ["replacement"]
        assert rulings[0].recorded_at == datetime(2026, 10, 2, tzinfo=UTC)
        assert len(notes) == 2

    def test_a_missing_file_is_no_rulings(self, tmp_path: Path) -> None:
        assert load_rulings(tmp_path / "absent.jsonl") == ([], [])


class TestVerdictsForPair:
    async def test_every_kind_in_either_order(self, db_session: AsyncSession) -> None:
        a, b = content_hash(_OLD), content_hash(_NEW)
        await record_verdict(db_session, ProbeKind.UPDATE_SLOT, "v1", (a, b), True, model="m")
        await record_verdict(
            db_session, ProbeKind.RECONCILE_CONTRADICTION, "v1", (b, a), True, model="m"
        )
        await record_verdict(db_session, ProbeKind.CONTRADICTION, "v1", (a, "zz"), False, model="m")
        rows = await verdicts_for_pair(db_session, a, b)
        assert [r.probe_kind for r in rows] == ["update_slot", "reconcile_contradiction"]
