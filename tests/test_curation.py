# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the curation surface — operations/curation.

Covers the card shape + key round-trip, the ``count_active_dependents`` store
helper (a new query shape), leverage scoring, the session model (today's-N,
snooze / affirm filtering), and the gesture dispatch onto existing write ops.
The structural finders run with ``semantic=False``; the duplicate-pair /
uncited-url finders need embedding / url-mention seeding and are covered by their
own suites — curation only *composes* them. The LLM_JUDGE wiring of the
duplicate finder (mode selection, the verdict on the card, leverage demotion of
a DISTINCT verdict, and graceful degrade) is covered in ``TestDuplicateVerdict``,
mocking ``suggest_co_evidential`` / its ``_llm_call`` seam rather than a real
model.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles import llm
from particles.config import get_config
from particles.core.schema import (
    CandidateCluster,
    CoEvidentialCandidate,
    Confidence,
    JudgeVerdictKind,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    SuggestMode,
    SuggestReport,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.agent_write import AgentWriteResult
from particles.operations.curation import (
    BeliefRevision,
    CardKind,
    CurationCard,
    DuplicateVerdict,
    apply_gesture,
    build_curation_queue,
)
from particles.operations.curation.cards import KIND_TITLES, describe_gesture, gestures_for
from particles.operations.curation.collect import collect_cards
from particles.operations.curation.leverage import score_cards, stakes_of
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import (
    count_active_dependents,
    get_particle,
    insert_particle,
)
from particles.store.relation_store import get_co_evidential_group
from particles.store.subject_store import insert_subject
from particles.store.utility_store import record_utility_events


def _active(
    content: str,
    *,
    valid_until: datetime | None = None,
    asserted_at: datetime | None = None,
    dep_on: str | None = None,
    status: Status = Status.ACTIVE,
    status_reason: StatusReason | None = None,
) -> Particle:
    """A minimal particle. SOURCE snapshot_id is None so the corpus-link lint
    check doesn't fire spuriously; ``dep_on`` adds a PARTICLE provenance edge."""
    provenance = [
        ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id=None)
    ]
    if dep_on is not None:
        provenance.append(
            ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=dep_on, snapshot_id=None)
        )
    fields: dict[str, object] = {
        "content": content,
        "confidence": Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        "uncertainty_nature": UncertaintyNature.EPISTEMIC,
        "provenance": provenance,
        "asserted_by": "general-extractor",
        "subject_ids": ["sid"],
        "status": status,
    }
    if status_reason is not None:
        fields["status_reason"] = status_reason
    if valid_until is not None:
        fields["valid_until"] = valid_until
    if asserted_at is not None:
        fields["asserted_at"] = asserted_at
    return Particle(**fields)  # type: ignore[arg-type]


def _inconsistency(*about: str) -> Particle:
    """An INCONSISTENCY meta-particle referencing the contested belief(s)."""
    return Particle(
        content="Conflict between beliefs.",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=pid, snapshot_id=None)
            for pid in about
        ],
        asserted_by="lint",
        status=Status.INCONSISTENCY,
    )


_PAST = datetime(2000, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Card shape — pure, no DB                                                     #
# --------------------------------------------------------------------------- #


class TestCardKey:
    def test_belief_key_roundtrips(self) -> None:
        card = CurationCard(
            kind=CardKind.DUPLICATE_PAIR,
            particle_ids=["b", "a"],
            diagnostic="x",
            suggested_gestures=gestures_for(CardKind.DUPLICATE_PAIR),
        )
        # Key is order-independent (sorted) so the same pair yields one key.
        assert card.key == "duplicate_pair:a|b"
        rebuilt = CurationCard.from_key(card.key)
        assert rebuilt.kind is CardKind.DUPLICATE_PAIR
        assert sorted(rebuilt.particle_ids) == ["a", "b"]

    def test_url_key_roundtrips(self) -> None:
        card = CurationCard(
            kind=CardKind.UNCITED_URL,
            corpus_url="https://example.com/x",
            diagnostic="x",
        )
        assert card.key == "uncited_url:https://example.com/x"
        rebuilt = CurationCard.from_key(card.key)
        assert rebuilt.kind is CardKind.UNCITED_URL
        assert rebuilt.corpus_url == "https://example.com/x"

    def test_failed_snapshots_key(self) -> None:
        card = CurationCard(kind=CardKind.FAILED_SNAPSHOTS, diagnostic="x")
        assert card.key == "failed_snapshots"
        assert CurationCard.from_key("failed_snapshots").kind is CardKind.FAILED_SNAPSHOTS

    def test_unparseable_key_raises(self) -> None:
        with pytest.raises(ValueError):
            CurationCard.from_key("not-a-real-key")

    def test_every_kind_offers_gestures(self) -> None:
        for kind in CardKind:
            assert gestures_for(kind), f"{kind} has no gestures"


# --------------------------------------------------------------------------- #
# Store helper — count_active_dependents (new query shape)                     #
# --------------------------------------------------------------------------- #


class TestGestureDescriptions:
    """Every offered gesture is explained, so a curator can choose between them."""

    @pytest.mark.parametrize("kind", list(CardKind))
    def test_every_kind_has_a_title_and_every_gesture_a_description(self, kind: CardKind) -> None:
        assert KIND_TITLES[kind][0]
        card = CurationCard(kind=kind, particle_ids=["p-1"], diagnostic="")
        for g in gestures_for(kind):
            text = describe_gesture(card, g, snooze_days=14)
            assert text, f"{kind.value}/{g} has no description"
            assert "{" not in text  # every placeholder filled

    def test_resolve_names_this_cards_actions(self) -> None:
        # the help line names exactly the actions this card offers.
        card = CurationCard(
            kind=CardKind.INCONSISTENCY,
            diagnostic="",
            inconsistency_id="inc-9",
            resolve_actions=["PREFER_B", "BOTH_VALID", "DISCARD"],
        )
        text = describe_gesture(card, "resolve", snooze_days=14)
        assert "--action PREFER_B|BOTH_VALID|DISCARD" in text
        stale = CurationCard(kind=CardKind.STALE, particle_ids=["p-1"], diagnostic="")
        assert "30 days" in describe_gesture(stale, "snooze", snooze_days=30)

    def test_supersede_is_described_as_applied_with_its_flags(self) -> None:
        # supersede dispatches from `curate apply`, so its description
        # names the flags rather than routing the curator to HTTP / MCP.
        stale = CurationCard(kind=CardKind.STALE, particle_ids=["p-1"], diagnostic="")
        text = describe_gesture(stale, "supersede", snooze_days=14)
        assert "--content" in text and "--reason" in text and "--confidence" in text
        assert "Not applied" not in text and "POST" not in text

    def test_a_kind_specific_meaning_overrides_the_generic_one(self) -> None:
        dup = CurationCard(kind=CardKind.DUPLICATE_PAIR, particle_ids=["a", "b"], diagnostic="")
        assert "different claims" in describe_gesture(dup, "dismiss", snooze_days=14)


class TestCountActiveDependents:
    @pytest.mark.asyncio
    async def test_counts_provenance_dependents(self, db_session: AsyncSession) -> None:
        a = _active("base claim")
        await insert_particle(db_session, a)
        for i in range(3):
            await insert_particle(db_session, _active(f"rests on a #{i}", dep_on=a.id))
        independent = _active("unrelated")
        await insert_particle(db_session, independent)
        await db_session.flush()

        counts = await count_active_dependents(db_session, {a.id, independent.id})
        assert counts[a.id] == 3
        assert counts[independent.id] == 0

    @pytest.mark.asyncio
    async def test_empty_input(self, db_session: AsyncSession) -> None:
        assert await count_active_dependents(db_session, set()) == {}


# --------------------------------------------------------------------------- #
# Collect + session model                                                     #
# --------------------------------------------------------------------------- #


class TestQueue:
    @pytest.mark.asyncio
    async def test_surfaces_stale_and_contested(self, db_session: AsyncSession) -> None:
        stale = _active("expired belief", valid_until=_PAST)
        contested = _active("disputed belief")
        await insert_particle(db_session, stale)
        await insert_particle(db_session, contested)

        inc = _inconsistency(contested.id)
        await insert_particle(db_session, inc)
        await db_session.flush()

        cards = (await build_curation_queue(db_session, semantic=False)).cards
        by_kind = {c.kind: c for c in cards}
        assert CardKind.STALE in by_kind
        # a belief contested only by an open INCONSISTENCY gets no
        # CONTESTED card; each record it sits in gets its own conflict card.
        assert CardKind.CONTESTED not in by_kind
        assert by_kind[CardKind.STALE].particle_ids == [stale.id]
        conflicts = [c for c in cards if c.kind is CardKind.INCONSISTENCY]
        assert {c.inconsistency_id for c in conflicts} >= {inc.id}

    @pytest.mark.asyncio
    async def test_session_size_caps(self, db_session: AsyncSession) -> None:
        for i in range(9):
            await insert_particle(db_session, _active(f"expired #{i}", valid_until=_PAST))
        await db_session.flush()

        # Default session_size is 7 — the finite "today's N".
        assert len((await build_curation_queue(db_session, semantic=False)).cards) == 7
        assert len((await build_curation_queue(db_session, semantic=False, limit=3)).cards) == 3

    @pytest.mark.asyncio
    async def test_kind_filter(self, db_session: AsyncSession) -> None:
        stale = _active("expired", valid_until=_PAST)
        contested = _active("disputed")
        await insert_particle(db_session, stale)
        await insert_particle(db_session, contested)
        await insert_particle(db_session, _inconsistency(contested.id))
        await db_session.flush()

        cards = (
            await build_curation_queue(db_session, semantic=False, kind=CardKind.INCONSISTENCY)
        ).cards
        assert cards and {c.kind for c in cards} == {CardKind.INCONSISTENCY}

    @pytest.mark.asyncio
    async def test_surfaces_no_subject(self, db_session: AsyncSession) -> None:
        # an ACTIVE CLAIM with no subjects becomes a NO_SUBJECT card
        # whose resolving gesture is assign-subject. The orphan needs a SOURCE
        # provenance edge so the ORPHAN lint does not also fire.
        orphan = Particle(
            content="An orphaned claim about nothing in particular.",
            confidence=Confidence(value=0.7, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[
                ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id=None)
            ],
            asserted_by="general-extractor",
            subject_ids=[],
        )
        await insert_particle(db_session, orphan)
        await db_session.flush()

        cards = (
            await build_curation_queue(db_session, semantic=False, kind=CardKind.NO_SUBJECT)
        ).cards
        assert len(cards) == 1
        card = cards[0]
        assert card.kind is CardKind.NO_SUBJECT
        assert card.particle_ids == [orphan.id]
        assert "assign-subject" in card.suggested_gestures
        # The key round-trips through from_key (used by the apply path).
        assert card.key == f"no_subject:{orphan.id}"
        assert CurationCard.from_key(card.key).kind is CardKind.NO_SUBJECT

    @pytest.mark.asyncio
    async def test_cards_carry_particle_brief(self, db_session: AsyncSession) -> None:
        # each card carries a compact ParticleBrief of its particle(s)
        # — claim text + effective confidence + status — so a gesture (e.g. which
        # of a duplicate pair to keep) can be judged without a `particles show`
        # round-trip. Briefs are aligned to particle_ids.
        stale = _active("expired belief", valid_until=_PAST)
        await insert_particle(db_session, stale)
        await db_session.flush()

        cards = (await build_curation_queue(db_session, semantic=False)).cards
        stale_card = next(c for c in cards if c.kind is CardKind.STALE)
        assert len(stale_card.particles) == len(stale_card.particle_ids) == 1
        brief = stale_card.particles[0]
        assert brief.particle_id == stale.id
        assert brief.content == "expired belief"
        assert brief.status == Status.ACTIVE.value
        assert 0.0 < brief.effective_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_briefs_empty_for_particleless_card(self, db_session: AsyncSession) -> None:
        # A card with no particle (e.g. failed_snapshots) keeps the empty default.
        from particles.operations.curation.session import _attach_particle_briefs

        card = CurationCard(kind=CardKind.FAILED_SNAPSHOTS, diagnostic="2 failed")
        await _attach_particle_briefs(db_session, [card])
        assert card.particles == []

    @pytest.mark.asyncio
    async def test_contested_card_carries_both_sides_of_its_conflict(
        self, db_session: AsyncSession
    ) -> None:
        # "Does it stand?" cannot be judged from the flagged belief alone: the
        # card carries the INCONSISTENCY's claims A and B in review's order, so
        # a client can offer keep-A / keep-B without guessing the sides.
        a = _active("the baseline top 15 held no guidelines")
        b = _active("by iteration 5 no guideline had reached the top 15")
        await insert_particle(db_session, a)
        await insert_particle(db_session, b)
        inc = _inconsistency(a.id, b.id)
        await insert_particle(db_session, inc)
        await db_session.flush()

        cards = (
            await build_curation_queue(db_session, semantic=False, kind=CardKind.INCONSISTENCY)
        ).cards
        (card,) = cards
        assert card.key == f"inconsistency:{inc.id}"
        assert card.inconsistency_id == inc.id
        assert card.particle_ids == [a.id, b.id]
        assert card.conflict is not None
        assert card.conflict.inconsistency_id == inc.id
        assert card.conflict.a is not None and card.conflict.a.particle_id == a.id
        assert card.conflict.b is not None and card.conflict.b.content == b.content
        # The kind's title and question travel on the card for every client.
        assert (card.title, card.question) == KIND_TITLES[CardKind.INCONSISTENCY]
        dumped = card.model_dump(mode="json")
        assert dumped["title"] == "Open conflict" and dumped["question"]

    @pytest.mark.asyncio
    async def test_briefs_carry_the_source_uri(self, db_session: AsyncSession) -> None:
        # A belief with no subject is unplaceable without its source.
        from particles.core.schema import CorpusEntry, FetchPolicy, Mutability
        from particles.corpus.store import CorpusEntryRow

        db_session.add(
            CorpusEntryRow.from_model(
                CorpusEntry(
                    entry_id="e1",
                    uri_r="claude-code://session/abc",
                    source_type="CONVERSATION",
                    mutability=Mutability.STABLE,
                    fetch_policy=FetchPolicy.NEVER,
                    deposited_by="test",
                    tags=[],
                )
            )
        )
        stale = _active("expired belief", valid_until=_PAST)
        await insert_particle(db_session, stale)
        await db_session.flush()

        cards = (await build_curation_queue(db_session, semantic=False)).cards
        brief = next(c for c in cards if c.kind is CardKind.STALE).particles[0]
        assert brief.source_uri == "claude-code://session/abc"
        assert brief.asserted_at is not None

    @pytest.mark.asyncio
    async def test_open_count_is_the_backlog_behind_the_slice(
        self, db_session: AsyncSession
    ) -> None:
        # The served slice is capped; open_count says how much is behind it,
        # net of what the operator already snoozed.
        for i in range(9):
            await insert_particle(db_session, _active(f"expired #{i}", valid_until=_PAST))
        await db_session.flush()

        result = await build_curation_queue(db_session, semantic=False, kind=CardKind.STALE)
        assert (len(result.cards), result.open_count) == (7, 9)

        await apply_gesture(db_session, result.cards[0], "snooze", actor="test")
        await db_session.flush()
        after = await build_curation_queue(db_session, semantic=False, kind=CardKind.STALE)
        assert (len(after.cards), after.open_count) == (7, 8)

    @pytest.mark.asyncio
    async def test_contested_and_dependents_raise_leverage(self, db_session: AsyncSession) -> None:
        # A high-dependency stale belief outranks an isolated stale belief.
        heavy = _active("load-bearing", valid_until=_PAST, asserted_at=_PAST)
        light = _active("isolated", valid_until=_PAST, asserted_at=_PAST)
        await insert_particle(db_session, heavy)
        await insert_particle(db_session, light)
        for i in range(5):
            await insert_particle(db_session, _active(f"dep #{i}", dep_on=heavy.id))
        await db_session.flush()

        cards = (await build_curation_queue(db_session, semantic=False)).cards
        ranked = [c for c in cards if c.kind is CardKind.STALE]
        assert ranked[0].particle_ids == [heavy.id]
        heavy_card = next(c for c in ranked if c.particle_ids == [heavy.id])
        light_card = next(c for c in ranked if c.particle_ids == [light.id])
        assert heavy_card.leverage > light_card.leverage


# --------------------------------------------------------------------------- #
# Gesture dispatch                                                            #
# --------------------------------------------------------------------------- #


class TestGestures:
    @pytest.mark.asyncio
    async def test_affirm_records_event_and_suppresses(self, db_session: AsyncSession) -> None:
        stale = _active("expired", valid_until=_PAST)
        await insert_particle(db_session, stale)
        await db_session.flush()

        [card] = (await build_curation_queue(db_session, semantic=False)).cards
        msg = await apply_gesture(db_session, card, "affirm")
        await db_session.commit()
        assert "Affirmed" in msg

        events = await list_events(db_session, event_type=OperatorEventType.BELIEF_AFFIRMED)
        assert len(events) == 1
        assert events[0].payload == {"card_key": card.key, "kind": card.kind.value}
        # Affirmed card no longer surfaces.
        assert (await build_curation_queue(db_session, semantic=False)).cards == []

    @pytest.mark.asyncio
    async def test_snooze_suppresses_then_expires(self, db_session: AsyncSession) -> None:
        stale = _active("expired", valid_until=_PAST)
        await insert_particle(db_session, stale)
        await db_session.flush()

        [card] = (await build_curation_queue(db_session, semantic=False)).cards
        await apply_gesture(db_session, card, "snooze", days=14)
        await db_session.commit()
        assert (await build_curation_queue(db_session, semantic=False)).cards == []

        event = (await list_events(db_session, event_type=OperatorEventType.CURATION_CARD_SNOOZED))[
            0
        ]
        assert event.payload is not None
        assert event.payload["card_key"] == card.key
        assert event.payload["snooze_days"] == 14

    @pytest.mark.asyncio
    async def test_retract_transitions_belief(self, db_session: AsyncSession) -> None:
        stale = _active("expired", valid_until=_PAST)
        await insert_particle(db_session, stale)
        await db_session.flush()

        [card] = (await build_curation_queue(db_session, semantic=False)).cards
        msg = await apply_gesture(db_session, card, "retract", reason="no longer true")
        await db_session.commit()
        assert "Retracted" in msg

        updated = await get_particle(db_session, stale.id)
        assert updated is not None
        assert updated.status is Status.RETRACTED
        events = await list_events(db_session, event_type=OperatorEventType.PARTICLE_RETRACTED)
        assert events[0].reason == "no longer true"

    @pytest.mark.asyncio
    async def test_merge_links_pair(self, db_session: AsyncSession) -> None:
        a = _active("claim one")
        b = _active("claim two")
        await insert_particle(db_session, a)
        await insert_particle(db_session, b)
        await db_session.flush()

        card = CurationCard(
            kind=CardKind.DUPLICATE_PAIR,
            particle_ids=[a.id, b.id],
            diagnostic="dup",
            suggested_gestures=gestures_for(CardKind.DUPLICATE_PAIR),
        )
        await apply_gesture(db_session, card, "merge")
        await db_session.commit()
        assert b.id in await get_co_evidential_group(db_session, a.id)

    @pytest.mark.asyncio
    async def test_surfaced_gesture_points_to_command(self, db_session: AsyncSession) -> None:
        card = CurationCard(
            kind=CardKind.FAILED_SNAPSHOTS,
            diagnostic="x",
            suggested_gestures=gestures_for(CardKind.FAILED_SNAPSHOTS),
        )
        with pytest.raises(ValueError, match="particles reindex"):
            await apply_gesture(db_session, card, "reindex")

    @pytest.mark.asyncio
    async def test_comment_left_every_belief_card(self, db_session: AsyncSession) -> None:
        # a contradiction card is an unrecorded pair, so there is
        # no record for review to resolve; comment is no longer offered.
        card = CurationCard(
            kind=CardKind.CONTRADICTION,
            particle_ids=["p1"],
            diagnostic="x",
            suggested_gestures=gestures_for(CardKind.CONTRADICTION),
        )
        assert "comment" not in card.suggested_gestures
        with pytest.raises(ValueError, match="does not offer"):
            await apply_gesture(db_session, card, "comment")

    @pytest.mark.asyncio
    async def test_gesture_not_offered_raises(self, db_session: AsyncSession) -> None:
        card = CurationCard(
            kind=CardKind.UNCITED_URL,
            corpus_url="https://example.com",
            diagnostic="x",
            suggested_gestures=gestures_for(CardKind.UNCITED_URL),
        )
        with pytest.raises(ValueError, match="does not offer"):
            await apply_gesture(db_session, card, "retract")


# --------------------------------------------------------------------------- #
# supersede — dispatched from the operator's revision               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def _stub_embeddings() -> Any:
    """A constant embedding so the successor's §6.6 insert calls no real model."""
    import numpy as np

    from particles import embeddings as ep

    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


def _card_for(particle: Particle, kind: CardKind = CardKind.STALE) -> CurationCard:
    return CurationCard(
        kind=kind,
        particle_ids=[particle.id],
        diagnostic="x",
        suggested_gestures=gestures_for(kind),
    )


@pytest.mark.usefixtures("_stub_embeddings")
class TestSupersedeGesture:
    @pytest.mark.asyncio
    async def test_replaces_the_belief_and_inherits_its_subjects_by_id(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import particles.ingest.subject_resolver as sr

        # §5: not gated on the agent allowlist, so an empty one changes nothing.
        get_config().mcp.write.enabled_stores = []
        # §2: inherited subjects are linked by id; the resolver is never asked.
        resolve = AsyncMock(side_effect=AssertionError("inherited ids must not re-resolve"))
        monkeypatch.setattr(sr, "resolve_subject", resolve)
        subject = Subject(canonical_name="Snooze window", asserted_by="test")
        await insert_subject(db_session, subject)
        old = _active("The snooze window is 14 days.", valid_until=_PAST)
        old = old.model_copy(update={"subject_ids": [subject.id]})
        await insert_particle(db_session, old)
        await db_session.flush()

        msg = await apply_gesture(
            db_session,
            _card_for(old),
            "supersede",
            reason="Changed in 1.140",
            revision=BeliefRevision(content="The snooze window is 30 days.", confidence=0.8),
        )
        await db_session.commit()

        assert f"Superseded {old.id[:8]}" in msg and "ASSERTED" in msg
        prior = await get_particle(db_session, old.id)
        assert prior is not None and prior.status is Status.SUPERSEDED
        [event] = await list_events(db_session, event_type=OperatorEventType.PARTICLE_SUPERSEDED)
        assert event.actor == "curate"
        assert event.reason == "Changed in 1.140"
        assert event.payload is not None and event.payload["operator"] is True
        successor_id = next(r.ref_id for r in event.refs if r.ref_id != old.id)
        successor = await get_particle(db_session, successor_id)
        assert successor is not None
        assert successor.status is Status.ACTIVE
        assert successor.content == "The snooze window is 30 days."
        assert successor.supersedes == old.id
        assert successor.subject_ids == [subject.id]
        assert successor.confidence.value == pytest.approx(0.8)
        resolve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_subject_flags_replace_the_inherited_set(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import particles.ingest.subject_resolver as sr

        by_id = Subject(canonical_name="Curation", asserted_by="test")
        by_name = Subject(canonical_name="Snooze window", asserted_by="test")
        await insert_subject(db_session, by_id)
        await insert_subject(db_session, by_name)
        monkeypatch.setattr(sr, "resolve_subject", AsyncMock(return_value=by_name))
        old = _active("A claim filed under the wrong subject.", valid_until=_PAST)
        await insert_particle(db_session, old)
        await db_session.flush()

        await apply_gesture(
            db_session,
            _card_for(old),
            "supersede",
            reason="Refiled",
            revision=BeliefRevision(
                content="A claim filed under the right subjects.",
                confidence=0.7,
                subjects=[by_id.id, "Snooze window"],
            ),
        )
        [event] = await list_events(db_session, event_type=OperatorEventType.PARTICLE_SUPERSEDED)
        successor = await get_particle(
            db_session, next(r.ref_id for r in event.refs if r.ref_id != old.id)
        )
        assert successor is not None
        assert successor.subject_ids == [by_id.id, by_name.id]  # "sid" is not inherited

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("revision", "expected"),
        [
            (BeliefRevision(content="c", confidence=0.5), {"source_excerpt": "the reason"}),
            (
                BeliefRevision(content="c", confidence=0.5, source_excerpt="the release notes"),
                {"source_excerpt": "the release notes"},
            ),
            (
                BeliefRevision(content="c", confidence=0.5, corpus_entry_id="entry-9"),
                {"source_excerpt": None, "corpus_entry_id": "entry-9"},
            ),
        ],
    )
    async def test_provenance_defaults_to_the_reason(
        self, db_session: AsyncSession, revision: BeliefRevision, expected: dict[str, Any]
    ) -> None:
        old = _active("c0", valid_until=_PAST)
        await insert_particle(db_session, old)
        await db_session.flush()
        write = AsyncMock(
            return_value=AgentWriteResult(asserted_particle_id="succ-1", verdict="ASSERTED")
        )
        with patch("particles.operations.agent_write.supersede_belief", new=write):
            await apply_gesture(
                db_session, _card_for(old), "supersede", reason=" the reason ", revision=revision
            )
        kwargs = write.await_args.kwargs
        for key, value in expected.items():
            assert kwargs[key] == value
        assert kwargs["operator"] is True
        assert kwargs["actor"] == "curate"
        assert kwargs["reason"] == "the reason"
        assert kwargs["subject_ids"] == ["sid"]

    @pytest.mark.asyncio
    async def test_a_conflicting_successor_names_the_inconsistency(
        self, db_session: AsyncSession
    ) -> None:
        old = _active("c0", valid_until=_PAST)
        await insert_particle(db_session, old)
        await db_session.flush()
        write = AsyncMock(
            return_value=AgentWriteResult(
                asserted_particle_id="succ-1",
                verdict="INCONSISTENCY_RAISED",
                inconsistency_id="inc-12345678",
            )
        )
        with patch("particles.operations.agent_write.supersede_belief", new=write):
            msg = await apply_gesture(
                db_session,
                _card_for(old),
                "supersede",
                reason="r",
                revision=BeliefRevision(content="c", confidence=0.5),
            )
        assert "INCONSISTENCY_RAISED" in msg and "inc-1234" in msg and "particles review" in msg

    @pytest.mark.asyncio
    async def test_the_successor_is_the_operators(self, db_session: AsyncSession) -> None:
        # attributed to the operator principal, HUMAN_REVIEW, unclamped.
        get_config().curation.operator_identity = "operator:test"
        get_config().mcp.write.max_asserted_confidence = 0.9
        old = _active("c0", valid_until=_PAST)
        await insert_particle(db_session, old)
        await db_session.flush()
        msg = await apply_gesture(
            db_session,
            _card_for(old),
            "supersede",
            reason="r",
            revision=BeliefRevision(content="c1", confidence=0.95),
        )
        assert "capped" not in msg
        [event] = await list_events(db_session, event_type=OperatorEventType.PARTICLE_SUPERSEDED)
        successor = await get_particle(
            db_session, next(r.ref_id for r in event.refs if r.ref_id != old.id)
        )
        assert successor is not None
        assert successor.asserted_by == "operator:test"
        assert successor.confidence.calibration_source is CalibrationSource.HUMAN_REVIEW
        assert successor.confidence.value == pytest.approx(0.95)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reason", "revision", "match"),
        [
            ("r", None, "--content"),
            ("r", BeliefRevision(content="  ", confidence=0.5), "--content"),
            ("  ", BeliefRevision(content="c", confidence=0.5), "--reason"),
            (None, BeliefRevision(content="c", confidence=0.5), "--reason"),
        ],
    )
    async def test_refuses_without_content_or_reason(
        self,
        db_session: AsyncSession,
        reason: str | None,
        revision: BeliefRevision | None,
        match: str,
    ) -> None:
        old = _active("c0", valid_until=_PAST)
        await insert_particle(db_session, old)
        await db_session.flush()
        with pytest.raises(ValueError, match=match):
            await apply_gesture(
                db_session, _card_for(old), "supersede", reason=reason, revision=revision
            )
        still = await get_particle(db_session, old.id)
        assert still is not None and still.status is Status.ACTIVE

    @pytest.mark.asyncio
    async def test_an_orphan_needs_a_subject(self, db_session: AsyncSession) -> None:
        orphan = _active("An unsubjected claim.").model_copy(update={"subject_ids": []})
        await insert_particle(db_session, orphan)
        await db_session.flush()
        with pytest.raises(ValueError, match="assign-subject"):
            await apply_gesture(
                db_session,
                _card_for(orphan, CardKind.NO_SUBJECT),
                "supersede",
                reason="r",
                revision=BeliefRevision(content="c", confidence=0.5),
            )
        still = await get_particle(db_session, orphan.id)
        assert still is not None and still.status is Status.ACTIVE

    @pytest.mark.asyncio
    async def test_a_retired_belief_is_refused_by_the_operator_guard(
        self, db_session: AsyncSession
    ) -> None:
        gone = _active("c0", status=Status.RETRACTED)
        await insert_particle(db_session, gone)
        await db_session.flush()
        with pytest.raises(ValueError, match="ACTIVE"):
            await apply_gesture(
                db_session,
                _card_for(gone),
                "supersede",
                reason="r",
                revision=BeliefRevision(content="c", confidence=0.5),
            )


# --------------------------------------------------------------------------- #
# Projection-blocking leverage                                      #
# --------------------------------------------------------------------------- #


class TestProjectionBlocking:
    @pytest.mark.asyncio
    async def test_manifest_belief_outranks_isolated(
        self, db_session: AsyncSession, tmp_path
    ) -> None:
        """A belief feeding a configured projection manifest gets extra leverage."""
        from particles.config import get_config
        from particles.core.schema import Subject
        from particles.store.subject_store import insert_subject, link_particle_to_subjects

        subject = Subject(canonical_name="Projected", asserted_by="test")
        await insert_subject(db_session, subject)
        # Two equally-stale beliefs; only the first feeds the projected doc.
        featured = _active("load-bearing belief", valid_until=_PAST, asserted_at=_PAST)
        plain = _active("isolated belief", valid_until=_PAST, asserted_at=_PAST)
        # Projection selection loads via get_active_particles_with_embeddings, so
        # the selected belief needs a (non-null) embedding to be retrievable; the
        # value is unused under the key-free use_embeddings=False selection.
        emb = [0.1, 0.2, 0.3, 0.4]
        await insert_particle(db_session, featured, emb)
        await insert_particle(db_session, plain, emb)
        await link_particle_to_subjects(db_session, featured.id, [subject.id])
        await db_session.flush()

        manifest = tmp_path / "readme.yaml"
        manifest.write_text(
            "name: readme\nsections:\n  - title: Featured\n    subjects: [Projected]\n",
            encoding="utf-8",
        )
        # Point the curation config at the manifest by mutating the cached
        # singleton (the autouse reset_config() clears it before the next test);
        # avoids reloading PARTICLES_CONFIG, which would disturb the test DB engine.
        get_config().curation.projection_manifests = [str(manifest)]

        cards = (await build_curation_queue(db_session, semantic=False)).cards
        featured_card = next(c for c in cards if c.particle_ids == [featured.id])
        plain_card = next(c for c in cards if c.particle_ids == [plain.id])
        assert featured_card.leverage > plain_card.leverage

    @pytest.mark.asyncio
    async def test_inert_without_manifests(self, db_session: AsyncSession) -> None:
        """No configured manifests → projection-blocking contributes nothing."""
        a = _active("belief a", valid_until=_PAST, asserted_at=_PAST)
        b = _active("belief b", valid_until=_PAST, asserted_at=_PAST)
        await insert_particle(db_session, a)
        await insert_particle(db_session, b)
        await db_session.flush()

        cards = (await build_curation_queue(db_session, semantic=False)).cards
        stale = [c for c in cards if c.kind is CardKind.STALE]
        # Same age, no deps, no projection manifest → equal leverage.
        assert len({round(c.leverage, 6) for c in stale}) == 1


# --------------------------------------------------------------------------- #
# Stakes-weighted leverage                                          #
# --------------------------------------------------------------------------- #


async def _belief(
    session: AsyncSession, content: str, *, age_days: float, uses: int = 0
) -> Particle:
    """An ACTIVE belief asserted ``age_days`` ago and used in ``uses`` sessions.

    Every utility event is observed now, so its reinforcement score is exactly
    ``uses`` (no decay has elapsed).
    """
    now = datetime.now(UTC)
    belief = _active(content, asserted_at=now - timedelta(days=age_days))
    await insert_particle(session, belief)
    for i in range(uses):
        await record_utility_events(session, f"{belief.id}-s{i}", {belief.id: "literal"}, now)
    await session.flush()
    return belief


def _card(kind: CardKind, *particle_ids: str, url: str | None = None) -> CurationCard:
    return CurationCard(
        kind=kind,
        particle_ids=list(particle_ids),
        corpus_url=url,
        diagnostic="test",
        suggested_gestures=gestures_for(kind),
    )


class TestStakes:
    @pytest.mark.asyncio
    async def test_use_outranks_unused_conflict(self, db_session: AsyncSession) -> None:
        """An old unused conflict ranks below a new, heavily used duplicate."""
        a = await _belief(db_session, "unused side a", age_days=90)
        b = await _belief(db_session, "unused side b", age_days=90)
        used = await _belief(db_session, "heavily used", age_days=1, uses=20)
        conflict = _card(CardKind.CONTESTED, a.id, b.id)
        dup = _card(CardKind.DUPLICATE_PAIR, used.id)

        await score_cards(db_session, [conflict, dup])
        assert dup.leverage > conflict.leverage

    @pytest.mark.asyncio
    async def test_faint_use_does_not_beat_conflict(self, db_session: AsyncSession) -> None:
        """Pins the trade-off: one use is weaker than a conflict."""
        a = await _belief(db_session, "unused side a", age_days=90)
        b = await _belief(db_session, "unused side b", age_days=90)
        faint = await _belief(db_session, "used once", age_days=1, uses=1)
        conflict = _card(CardKind.CONTESTED, a.id, b.id)
        dup = _card(CardKind.DUPLICATE_PAIR, faint.id)

        await score_cards(db_session, [conflict, dup])
        assert conflict.leverage > dup.leverage

    @pytest.mark.asyncio
    async def test_stakes_is_the_max_over_members(self, db_session: AsyncSession) -> None:
        used = await _belief(db_session, "used", age_days=10, uses=20)
        unused = await _belief(db_session, "unused", age_days=10)
        pair = _card(CardKind.DUPLICATE_PAIR, used.id, unused.id)
        alone = _card(CardKind.DUPLICATE_PAIR, used.id)

        await score_cards(db_session, [pair, alone])
        assert pair.leverage == alone.leverage

    @pytest.mark.asyncio
    async def test_cold_start_keeps_the_adr_0139_order(self, db_session: AsyncSession) -> None:
        """No use and no dependents anywhere: stakes is the floor, same order."""
        beliefs = [
            await _belief(db_session, f"belief {i}", age_days=age)
            for i, age in enumerate((400, 200, 30, 1))
        ]

        def mixed() -> list[CurationCard]:
            return [
                _card(CardKind.CONTESTED, beliefs[1].id),
                _card(CardKind.STALE, beliefs[0].id),
                _card(CardKind.NO_SUBJECT, beliefs[2].id),
                _card(CardKind.DUPLICATE_PAIR, beliefs[3].id),
                _card(CardKind.UNCITED_URL, url="https://example.org/x"),
            ]

        def order(cards: list[CurationCard]) -> list[str]:
            return [c.key for c in sorted(cards, key=lambda c: -c.leverage)]

        with_stakes = mixed()
        await score_cards(db_session, with_stakes)
        get_config().curation.stakes.enabled = False
        without = mixed()
        await score_cards(db_session, without)
        assert order(with_stakes) == order(without)

    @pytest.mark.asyncio
    async def test_off_switches_reproduce_adr_0139(self, db_session: AsyncSession) -> None:
        a = await _belief(db_session, "side a", age_days=90, uses=5)
        b = await _belief(db_session, "side b", age_days=30)

        def cards() -> list[CurationCard]:
            return [_card(CardKind.CONTESTED, a.id, b.id), _card(CardKind.NO_SUBJECT, b.id)]

        get_config().curation.stakes.enabled = False
        disabled = cards()
        await score_cards(db_session, disabled)
        get_config().curation.stakes.enabled = True
        get_config().curation.stakes.floor = 1.0
        get_config().curation.stakes.base = 0.0
        neutral = cards()
        await score_cards(db_session, neutral)
        assert [c.leverage for c in neutral] == [c.leverage for c in disabled]
        assert disabled[0].leverage > 1.0  # the unscaled sum, contested + age

    @pytest.mark.asyncio
    async def test_belief_free_cards_take_the_floor(self, db_session: AsyncSession) -> None:
        faint = await _belief(db_session, "used once", age_days=1, uses=1)
        url = _card(CardKind.UNCITED_URL, url="https://example.org/uncited")
        no_subject = _card(CardKind.NO_SUBJECT, faint.id)

        await score_cards(db_session, [url, no_subject])
        stakes = get_config().curation.stakes
        assert url.leverage == round(stakes.floor * stakes.base, 6)
        assert no_subject.leverage > url.leverage

    @pytest.mark.asyncio
    async def test_dependents_are_reliance(self, db_session: AsyncSession) -> None:
        """A belief others derive from is relied on though no session used it."""
        base = await _belief(db_session, "premise", age_days=10)
        lone = await _belief(db_session, "lone", age_days=10)
        for i in range(20):
            await insert_particle(db_session, _active(f"derived #{i}", dep_on=base.id))
        await db_session.flush()

        cfg = get_config().curation
        assert stakes_of([base.id], {}, {base.id: 20}, cfg) == 1.0
        assert stakes_of([lone.id], {}, {}, cfg) == cfg.stakes.floor
        premise, isolated = _card(CardKind.NO_SUBJECT, base.id), _card(CardKind.NO_SUBJECT, lone.id)
        await score_cards(db_session, [premise, isolated])
        assert premise.leverage > isolated.leverage


# --------------------------------------------------------------------------- #
# Duplicate-pair LLM judge wiring                                   #
# --------------------------------------------------------------------------- #


def _fake_suggest(*, mode: SuggestMode, verdict: JudgeVerdictKind | None) -> AsyncMock:
    """An ``suggest_co_evidential`` mock that records its ``mode`` and returns one
    candidate carrying ``verdict`` (the shape ``collect_cards`` consumes)."""
    report = SuggestReport(
        mode=mode,
        clusters=[
            CandidateCluster(
                subject_id="sid",
                subject_name="A Subject",
                candidates=[
                    CoEvidentialCandidate(
                        particle_a="pa", particle_b="pb", similarity=0.94, verdict=verdict
                    )
                ],
            )
        ],
        total_candidates=1,
    )
    return AsyncMock(return_value=report)


def _mock_anthropic(text: str) -> MagicMock:
    """A stand-in Anthropic client whose ``messages.create`` returns ``text``.

    Injected via ``llm.set_client`` so finders that route through the real
    ``complete`` seam (rather than a patched ``_llm_call``) get a deterministic
    reply and make no network call. Mirrors ``_make_mock_anthropic`` in
    ``tests/test_llm.py``.
    """
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(return_value=resp)
    return client


class TestDuplicateVerdict:
    @pytest.mark.asyncio
    async def test_semantic_on_selects_llm_judge_and_carries_verdict(
        self, db_session: AsyncSession
    ) -> None:
        """semantic=True runs the duplicate finder in LLM_JUDGE and lands the
        verdict on the DUPLICATE_PAIR card."""
        mock = _fake_suggest(mode=SuggestMode.LLM_JUDGE, verdict=JudgeVerdictKind.DISTINCT)
        with patch("particles.operations.curation.collect.suggest_co_evidential", mock):
            cards = await collect_cards(db_session, semantic=True)

        # collect_cards asked for LLM_JUDGE mode (not REPORT).
        assert mock.await_args is not None
        assert mock.await_args.kwargs["mode"] is SuggestMode.LLM_JUDGE

        dup = next(c for c in cards if c.kind is CardKind.DUPLICATE_PAIR)
        assert isinstance(dup.verdict, DuplicateVerdict)
        assert dup.verdict.verdict is JudgeVerdictKind.DISTINCT

    @pytest.mark.asyncio
    async def test_semantic_off_stays_report_no_verdict(self, db_session: AsyncSession) -> None:
        """semantic=False keeps REPORT mode (similarity only) — no verdict on the
        card, exactly as before."""
        mock = _fake_suggest(mode=SuggestMode.REPORT, verdict=None)
        with patch("particles.operations.curation.collect.suggest_co_evidential", mock):
            cards = await collect_cards(db_session, semantic=False)

        assert mock.await_args is not None
        assert mock.await_args.kwargs["mode"] is SuggestMode.REPORT

        dup = next(c for c in cards if c.kind is CardKind.DUPLICATE_PAIR)
        assert dup.verdict is None

    @pytest.mark.asyncio
    async def test_unavailable_llm_degrades_to_unsure_no_demotion(
        self, db_session: AsyncSession
    ) -> None:
        """When the LLM is unavailable the judge defaults each pair to UNSURE;
        the card still surfaces with that verdict and is NOT demoted
        (only DISTINCT demotes) — graceful degrade, no crash."""
        mock = _fake_suggest(mode=SuggestMode.LLM_JUDGE, verdict=JudgeVerdictKind.UNSURE)
        with patch("particles.operations.curation.collect.suggest_co_evidential", mock):
            cards = await collect_cards(db_session, semantic=True)

        dup = next(c for c in cards if c.kind is CardKind.DUPLICATE_PAIR)
        assert dup.verdict is not None
        assert dup.verdict.verdict is JudgeVerdictKind.UNSURE

    @pytest.mark.asyncio
    async def test_distinct_verdict_demotes_leverage(self, db_session: AsyncSession) -> None:
        """leverage rule: a DISTINCT-verdict duplicate card scores lower
        than the same card with a PARAPHRASE / no verdict, so it sinks."""
        from particles.operations.curation.leverage import score_cards

        def _dup(verdict: JudgeVerdictKind | None) -> CurationCard:
            return CurationCard(
                kind=CardKind.DUPLICATE_PAIR,
                particle_ids=["x", "y"],
                subject_ids=["sid"],
                diagnostic="dup",
                suggested_gestures=gestures_for(CardKind.DUPLICATE_PAIR),
                verdict=None if verdict is None else DuplicateVerdict(verdict=verdict),
            )

        # Give the pair some real signal (a dependent) so the base score is > 0
        # and the multiplicative demotion is observable.
        base = _active("base claim")
        await insert_particle(db_session, base)
        await db_session.flush()

        distinct = _dup(JudgeVerdictKind.DISTINCT)
        distinct.particle_ids = [base.id, "y"]
        paraphrase = _dup(JudgeVerdictKind.PARAPHRASE)
        paraphrase.particle_ids = [base.id, "y"]
        none_card = _dup(None)
        none_card.particle_ids = [base.id, "y"]
        for i in range(5):
            await insert_particle(db_session, _active(f"dep #{i}", dep_on=base.id))
        await db_session.flush()

        await score_cards(db_session, [distinct, paraphrase, none_card])
        assert distinct.leverage < paraphrase.leverage
        assert paraphrase.leverage == none_card.leverage

    @pytest.mark.asyncio
    async def test_verdict_does_not_affect_card_key(self) -> None:
        """The verdict is advisory and must NOT participate in the snooze/affirm
        identity (``key`` / ``from_key``)."""
        with_v = CurationCard(
            kind=CardKind.DUPLICATE_PAIR,
            particle_ids=["a", "b"],
            diagnostic="x",
            suggested_gestures=gestures_for(CardKind.DUPLICATE_PAIR),
            verdict=DuplicateVerdict(verdict=JudgeVerdictKind.DISTINCT, rationale="differ"),
        )
        without_v = CurationCard(
            kind=CardKind.DUPLICATE_PAIR,
            particle_ids=["a", "b"],
            diagnostic="x",
            suggested_gestures=gestures_for(CardKind.DUPLICATE_PAIR),
        )
        assert with_v.key == without_v.key == "duplicate_pair:a|b"
        # Round-tripping from the key never recovers (or invents) a verdict.
        assert CurationCard.from_key(with_v.key).verdict is None

    @pytest.mark.asyncio
    async def test_end_to_end_distinct_via_llm_call_seam(self, db_session: AsyncSession) -> None:
        """End-to-end through the real finder: two same-Subject near-duplicate
        beliefs, the patched ``_llm_call`` returns a DISTINCT verdict, and the
        card surfaces carrying it. Exercises collect + the actual
        LLM_JUDGE path, not a mocked finder."""
        import numpy as np

        subject = Subject(canonical_name="RDF Schema", asserted_by="test")
        await insert_subject(db_session, subject)
        a = Particle(
            content="Dan Brickley and R.V. Guha authored RDF Schema.",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            subject_ids=[subject.id],
            provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1")],
        )
        b = Particle(
            content="The W3C published RDF Schema on 2004-02-10.",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            subject_ids=[subject.id],
            provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1")],
        )
        emb_a = np.array([0.6, 0.8] + [0.0] * 382, dtype=np.float32)
        emb_b = np.array([0.61, 0.79] + [0.0] * 382, dtype=np.float32)
        await insert_particle(db_session, a, embedding=emb_a.tolist())
        await insert_particle(db_session, b, embedding=emb_b.tolist())
        await db_session.flush()

        # ``collect_cards(semantic=True)`` also runs the contradiction and
        # granularity lint finders, which bind ``_llm_call`` at module top and so
        # escape the patch below (it only rebinds the canonical
        # ``particles.operations._llm._llm_call`` the duplicate finder resolves
        # lazily). Inject a benign Anthropic mock so those collateral finders hit
        # the test seam instead of a real API call — a live call here would error
        # account-level and trip the process-global circuit breaker,
        # leaking into later tests. The autouse ``clear_subject_cache`` fixture
        # clears the client again before the next test.
        llm.set_client(_mock_anthropic("NO"))

        pair_key = f"{a.id[:8]}+{b.id[:8]}"
        with patch(
            "particles.operations._llm._llm_call",
            return_value=f'{{"{pair_key}": "DISTINCT"}}',
        ):
            cards = await collect_cards(db_session, semantic=True)

        dup = next(c for c in cards if c.kind is CardKind.DUPLICATE_PAIR)
        assert dup.verdict is not None
        assert dup.verdict.verdict is JudgeVerdictKind.DISTINCT


class TestSecondReadingOnCardSurfaces:
    """a semantic card collection counts confirmed contradictions only."""

    @pytest.mark.asyncio
    async def test_semantic_collection_verifies_every_flag(self, db_session: AsyncSession) -> None:
        from particles.core.schema import LintReport

        lint = AsyncMock(return_value=LintReport())
        with patch("particles.operations.curation.collect.run_lint", lint):
            await collect_cards(db_session, semantic=True)

        control = lint.await_args.kwargs["contradiction_probe"]
        assert control.verify is True
        # Uncapped, like the probe: an unread flag would drop out of the queue
        # with no disclosure line to say so.
        assert control.max_verifications is None
        assert control.max_probes is None

    @pytest.mark.asyncio
    async def test_verification_off_or_structural_passes_no_control(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config
        from particles.core.schema import LintReport

        lint = AsyncMock(return_value=LintReport())
        with patch("particles.operations.curation.collect.run_lint", lint):
            await collect_cards(db_session, semantic=False)
        assert lint.await_args.kwargs["contradiction_probe"] is None

        monkeypatch.setattr(get_config().audit, "verify_contradictions", False)
        with patch("particles.operations.curation.collect.run_lint", lint):
            await collect_cards(db_session, semantic=True)
        assert lint.await_args.kwargs["contradiction_probe"].verify is False


# --------------------------------------------------------------------------- #
# one card per open conflict                                        #
# --------------------------------------------------------------------------- #


def _census_record(a_side: list[str], b_side: list[str]) -> Particle:
    """A census INCONSISTENCY record naming its members by side."""
    from particles.core.contradiction_disclosure import CENSUS_ORIGIN, ORIGIN_KEY, SIDES_KEY

    record = _inconsistency(*a_side, *b_side)
    return record.model_copy(
        update={"properties": {ORIGIN_KEY: CENSUS_ORIGIN, SIDES_KEY: {"a": a_side, "b": b_side}}}
    )


def _demoted(content: str) -> Particle:
    return _active(
        content, status=Status.PROVENANCE_STALE, status_reason=StatusReason.RETRACTED_DEPENDENCY
    )


def _quarantined(content: str) -> Particle:
    return _active(
        content, status=Status.PROVENANCE_STALE, status_reason=StatusReason.CONFLICT_PENDING
    )


async def _insert(session: AsyncSession, *particles: Particle) -> None:
    """Insert, reaching a non-quarantine demotion the way the store does: by a
    transition from ACTIVE (only a quarantined loser may be born demoted)."""
    from particles.store.particle_store import update_particle_status

    for p in particles:
        born_demoted = (
            p.status is Status.PROVENANCE_STALE
            and p.status_reason is not StatusReason.CONFLICT_PENDING
        )
        if born_demoted:
            await insert_particle(
                session, p.model_copy(update={"status": Status.ACTIVE, "status_reason": None})
            )
            await update_particle_status(session, p.id, p.status, p.status_reason)
        else:
            await insert_particle(session, p)
    await session.flush()


async def _conflict_cards(session: AsyncSession) -> dict[str, CurationCard]:
    cards = (
        await build_curation_queue(session, semantic=False, kind=CardKind.INCONSISTENCY, limit=50)
    ).cards
    return {c.inconsistency_id or "": c for c in cards}


class TestConflictCards:
    @pytest.mark.asyncio
    async def test_briefs_carry_the_status_reason(self, db_session: AsyncSession) -> None:
        # Both sides of a no-ACTIVE record read PROVENANCE_STALE; only the
        # reason tells a quarantined B (restorable) from a demoted A.
        a, b = _demoted("d A"), _quarantined("d B")
        await _insert(db_session, a, b)
        record = _inconsistency(a.id, b.id)
        await _insert(db_session, record)

        conflict = (await _conflict_cards(db_session))[record.id].conflict
        assert conflict is not None and conflict.a is not None and conflict.b is not None
        assert (conflict.a.status, conflict.a.status_reason) == (
            "PROVENANCE_STALE",
            "RETRACTED_DEPENDENCY",
        )
        assert (conflict.b.status, conflict.b.status_reason) == (
            "PROVENANCE_STALE",
            "CONFLICT_PENDING",
        )

    @pytest.mark.asyncio
    async def test_one_card_per_record_whatever_its_members(self, db_session: AsyncSession) -> None:
        from particles.operations.review import list_inconsistencies

        both = (_active("both A"), _active("both B"))
        quarantine = (_active("q A"), _quarantined("q B"))
        demoted = (_demoted("d A"), _quarantined("d B"))
        dangling_b = _active("dangling B")
        await _insert(db_session, *both, *quarantine, *demoted, dangling_b)
        records = [
            _inconsistency(both[0].id, both[1].id),
            _inconsistency(quarantine[0].id, quarantine[1].id),
            _inconsistency(demoted[0].id, demoted[1].id),
            _inconsistency("gone-from-the-store", dangling_b.id),
        ]
        await _insert(db_session, *records)

        cards = await _conflict_cards(db_session)
        # §7 parity: one card per record `particles review` lists.
        assert set(cards) == {r.id for r in records}
        assert len(cards) == len(await list_inconsistencies(db_session))
        # §3: the offered actions follow the members' statuses.
        assert cards[records[0].id].resolve_actions == [
            "PREFER_A",
            "PREFER_B",
            "BOTH_VALID",
            "DISCARD",
        ]
        assert cards[records[1].id].resolve_actions == [
            "PREFER_A",
            "PREFER_B",
            "BOTH_VALID",
            "DISCARD",
        ]
        assert cards[records[2].id].resolve_actions == ["PREFER_B", "BOTH_VALID", "DISCARD"]
        assert cards[records[3].id].resolve_actions == ["PREFER_B", "BOTH_VALID", "DISCARD"]

    @pytest.mark.asyncio
    async def test_a_belief_in_two_records_gets_two_cards(self, db_session: AsyncSession) -> None:
        m, x, y = _active("shared"), _active("x"), _active("y")
        await _insert(db_session, m, x, y)
        r1, r2 = _inconsistency(m.id, x.id), _inconsistency(m.id, y.id)
        await _insert(db_session, r1, r2)

        cards = await _conflict_cards(db_session)
        assert set(cards) == {r1.id, r2.id}
        assert cards[r1.id].key != cards[r2.id].key

    @pytest.mark.asyncio
    async def test_a_census_record_is_one_card_with_its_further_members(
        self, db_session: AsyncSession
    ) -> None:
        a1, a2, b1 = _active("a1"), _active("a2"), _active("b1")
        await _insert(db_session, a1, a2, b1)
        record = _census_record([a1.id, a2.id], [b1.id])
        await _insert(db_session, record)

        (card,) = (await _conflict_cards(db_session)).values()
        assert card.particle_ids == [a1.id, b1.id, a2.id]
        assert card.conflict is not None
        assert card.conflict.a is not None and card.conflict.a.particle_id == a1.id
        assert [b.particle_id for b in card.conflict.further_a] == [a2.id]
        assert card.conflict.further_b == []

    @pytest.mark.asyncio
    async def test_key_round_trips_to_the_record(self) -> None:
        card = CurationCard(kind=CardKind.INCONSISTENCY, inconsistency_id="rec-1", diagnostic="")
        assert card.key == "inconsistency:rec-1"
        rebuilt = CurationCard.from_key(card.key)
        assert rebuilt.kind is CardKind.INCONSISTENCY
        assert rebuilt.inconsistency_id == "rec-1"
        assert rebuilt.suggested_gestures == ["resolve", "snooze"]

    @pytest.mark.asyncio
    async def test_resolve_closes_the_record(self, db_session: AsyncSession) -> None:
        a, b = _active("a"), _active("b")
        await _insert(db_session, a, b)
        record = _inconsistency(a.id, b.id)
        await _insert(db_session, record)

        card = CurationCard.from_key(f"inconsistency:{record.id}")
        message = await apply_gesture(db_session, card, "resolve", action="BOTH_VALID")
        assert "BOTH_VALID" in message
        closed = await get_particle(db_session, record.id)
        assert closed is not None and closed.status is not Status.INCONSISTENCY
        events = await list_events(db_session, event_type=OperatorEventType.REVIEW_RESOLVED)
        assert events and events[0].payload["action"] == "BOTH_VALID"
        assert await _conflict_cards(db_session) == {}

    @pytest.mark.asyncio
    async def test_comment_is_an_alias_of_resolve(self, db_session: AsyncSession) -> None:
        a, b = _active("a"), _active("b")
        await _insert(db_session, a, b)
        record = _inconsistency(a.id, b.id)
        await _insert(db_session, record)

        card = CurationCard.from_key(f"inconsistency:{record.id}")
        await apply_gesture(db_session, card, "comment", action="DISCARD")
        closed = await get_particle(db_session, record.id)
        assert closed is not None and closed.status is not Status.INCONSISTENCY

    @pytest.mark.asyncio
    async def test_withheld_actions_are_refused(self, db_session: AsyncSession) -> None:
        a, b = _demoted("a"), _quarantined("b")
        await _insert(db_session, a, b)
        record = _inconsistency(a.id, b.id)
        await _insert(db_session, record)
        card = CurationCard.from_key(f"inconsistency:{record.id}")

        with pytest.raises(
            ValueError, match="PREFER_A is not offered.*claim A is PROVENANCE_STALE"
        ):
            await apply_gesture(db_session, card, "resolve", action="PREFER_A")
        with pytest.raises(ValueError, match="needs --action"):
            await apply_gesture(db_session, card, "resolve")
        still = await get_particle(db_session, record.id)
        assert still is not None and still.status is Status.INCONSISTENCY

    @pytest.mark.asyncio
    @pytest.mark.parametrize("gesture", ["affirm", "dismiss"])
    async def test_a_conflict_cannot_be_hidden_for_good(
        self, db_session: AsyncSession, gesture: str
    ) -> None:
        card = CurationCard.from_key("inconsistency:rec-1")
        with pytest.raises(ValueError, match="BOTH_VALID.*DISCARD"):
            await apply_gesture(db_session, card, gesture)

    @pytest.mark.asyncio
    async def test_liveness_follows_the_record_on_a_stored_collection(
        self, db_session: AsyncSession
    ) -> None:
        from particles.operations.curation import rebuild_curation_snapshot
        from particles.store.particle_store import update_particle_status

        m, x, y, z = _active("m"), _active("x"), _active("y"), _active("z")
        await _insert(db_session, m, x, y, z)
        shared_1, shared_2 = _inconsistency(m.id, x.id), _inconsistency(m.id, y.id)
        closed_elsewhere = _inconsistency(y.id, z.id)
        await _insert(db_session, shared_1, shared_2, closed_elsewhere)
        await rebuild_curation_snapshot(db_session, semantic=False)
        assert set(await _conflict_cards(db_session)) == {
            shared_1.id,
            shared_2.id,
            closed_elsewhere.id,
        }

        # Level 3 by record id only: resolving the first record names m in its
        # event, and must not hide the second record's card.
        await apply_gesture(
            db_session,
            CurationCard.from_key(f"inconsistency:{shared_1.id}"),
            "resolve",
            action="PREFER_A",
        )
        # DEFER leaves the record open, so its card stays.
        await apply_gesture(
            db_session,
            CurationCard.from_key(f"inconsistency:{shared_2.id}"),
            "resolve",
            action="DEFER",
            note="later",
        )
        # A close by another path (the second reading) emits no
        # REVIEW_RESOLVED; the record's own status drops the card.
        await update_particle_status(
            db_session, closed_elsewhere.id, Status.RETRACTED, StatusReason.CONFLICT_RESOLVED
        )
        await db_session.commit()

        assert set(await _conflict_cards(db_session)) == {shared_2.id}

    @pytest.mark.asyncio
    async def test_suppression_is_by_the_record_key(self, db_session: AsyncSession) -> None:
        from particles.store.event_store import record_event

        a, b = _active("a"), _active("b")
        await _insert(db_session, a, b)
        record = _inconsistency(a.id, b.id)
        await _insert(db_session, record)

        # An earlier affirm on the belief's contested card does not carry over.
        await record_event(
            db_session,
            actor="test",
            event_type=OperatorEventType.BELIEF_AFFIRMED,
            refs=[],
            payload={"card_key": f"contested:{a.id}", "kind": "contested"},
        )
        await db_session.flush()
        assert record.id in await _conflict_cards(db_session)

        await apply_gesture(
            db_session, CurationCard.from_key(f"inconsistency:{record.id}"), "snooze"
        )
        await db_session.flush()
        assert await _conflict_cards(db_session) == {}

    @pytest.mark.asyncio
    async def test_demoted_members_still_age_the_card(self, db_session: AsyncSession) -> None:
        # §2: each signal is the max over the members, demoted ones included.
        old = datetime(2000, 1, 1, tzinfo=UTC)
        a = _active(
            "a",
            status=Status.PROVENANCE_STALE,
            status_reason=StatusReason.RETRACTED_DEPENDENCY,
            asserted_at=old,
        )
        b = _active(
            "b",
            status=Status.PROVENANCE_STALE,
            status_reason=StatusReason.CONFLICT_PENDING,
            asserted_at=old,
        )
        await _insert(db_session, a, b)
        await _insert(db_session, _inconsistency(a.id, b.id))

        (card,) = (await _conflict_cards(db_session)).values()
        cfg = get_config().curation
        w = cfg.leverage_weights
        # Full age and contestedness; neither member was used, so the card
        # takes the floor like any unused conflict (test 1).
        urgency = w.contestedness + w.staleness_age
        assert card.leverage == pytest.approx(cfg.stakes.floor * (cfg.stakes.base + urgency))


class TestNoActiveConflictStakes:
    """a conflict with no ACTIVE member takes its members' recorded reliance."""

    @staticmethod
    async def _use(session: AsyncSession, particle_id: str, uses: int) -> None:
        now = datetime.now(UTC)
        for i in range(uses):
            await record_utility_events(
                session, f"{particle_id}-s{i}", {particle_id: "literal"}, now
            )
        await session.flush()

    @pytest.mark.asyncio
    async def test_recorded_use_of_a_demoted_member_sets_the_stakes(
        self, db_session: AsyncSession
    ) -> None:
        # Test 2: use recorded before A's demotion is still read, so R(A) = 20
        # takes full stakes; its unused twin takes the floor.
        used = (_demoted("used a"), _quarantined("used b"))
        unused = (_demoted("unused a"), _quarantined("unused b"))
        await _insert(db_session, *used, *unused)
        used_rec = _inconsistency(*(p.id for p in used))
        unused_rec = _inconsistency(*(p.id for p in unused))
        await _insert(db_session, used_rec, unused_rec)
        await self._use(db_session, used[0].id, 20)

        cards = await _conflict_cards(db_session)
        cfg = get_config().curation
        assert cards[used_rec.id].leverage == pytest.approx(
            cards[unused_rec.id].leverage / cfg.stakes.floor, rel=1e-4
        )

    @pytest.mark.asyncio
    async def test_no_special_case_for_a_conflict_with_no_active_member(
        self, db_session: AsyncSession
    ) -> None:
        # Test 3: equal reliance and ages score the same whether A is ACTIVE
        # or demoted.
        dead = (_demoted("dead a"), _quarantined("dead b"))
        live = (_active("live a"), _quarantined("live b"))
        await _insert(db_session, *dead, *live)
        dead_rec = _inconsistency(*(p.id for p in dead))
        live_rec = _inconsistency(*(p.id for p in live))
        await _insert(db_session, dead_rec, live_rec)
        await self._use(db_session, dead[0].id, 4)
        await self._use(db_session, live[0].id, 4)

        cards = await _conflict_cards(db_session)
        assert cards[dead_rec.id].leverage == pytest.approx(cards[live_rec.id].leverage, rel=1e-3)

    @pytest.mark.asyncio
    async def test_active_dependents_of_a_demoted_member_count(
        self, db_session: AsyncSession
    ) -> None:
        # Test 4, an edge case: a dependency cascade normally demotes these
        # dependents too, but when they are ACTIVE they are reliance.
        a, b = _demoted("a"), _quarantined("b")
        await _insert(db_session, a, b)
        children = [_active(f"child {i}", dep_on=a.id) for i in range(3)]
        await _insert(db_session, *children)
        rec = _inconsistency(a.id, b.id)
        await _insert(db_session, rec)

        cards = await _conflict_cards(db_session)
        cfg = get_config().curation
        dep_norm = math.log1p(3) / math.log1p(cfg.dependency_norm_cap)
        stakes = cfg.stakes.floor + (1.0 - cfg.stakes.floor) * dep_norm
        w = cfg.leverage_weights
        urgency = cards[rec.id].leverage / stakes - cfg.stakes.base
        # The dependency signal is in the urgency too.
        assert urgency == pytest.approx(w.contestedness + w.dependency_count * dep_norm, abs=0.01)

    @pytest.mark.asyncio
    async def test_an_unused_no_active_conflict_does_not_rank_last(
        self, db_session: AsyncSession
    ) -> None:
        # Test 5: the floor, not the bottom; a belief-free card still ranks below.
        a, b = _demoted("a"), _quarantined("b")
        await _insert(db_session, a, b)
        conflict = _card(CardKind.INCONSISTENCY, a.id, b.id)
        url = _card(CardKind.UNCITED_URL, url="https://example.org/x")
        await score_cards(db_session, [conflict, url])
        assert conflict.leverage > url.leverage


class TestContestedNarrowing:
    """§4, on the finding → card projection alone."""

    def _finding(self, bases: list[str]) -> Any:
        from particles.core.schema import LintFinding

        return LintFinding(
            particle_id="p-1",
            finding_type="CONTESTED",
            severity="INFO",
            detail="Contested",
            contested_bases=bases,
            inconsistency_id="rec-1" if "inconsistency" in bases else None,
        )

    def test_inconsistency_only_belief_gets_no_contested_card(self) -> None:
        from particles.operations.curation.collect import cards_from_findings

        assert cards_from_findings([self._finding(["inconsistency"])]) == []

    def test_observer_basis_keeps_its_card_and_points_at_the_conflict(self) -> None:
        from particles.operations.curation.collect import cards_from_findings

        (card,) = cards_from_findings([self._finding(["divergence", "inconsistency"])])
        assert card.kind is CardKind.CONTESTED
        assert card.suggested_gestures == ["affirm", "snooze"]
        assert "inconsistency:rec-1" in card.diagnostic


# ---------------------------------------------------------------------------
# recoverable orphans fold into one gated_subjects card
# ---------------------------------------------------------------------------


class TestGatedSubjectsCard:
    async def _orphans(self, session: AsyncSession) -> tuple[Particle, Particle, Particle]:
        from particles.core.schema import Mutability
        from particles.corpus.deposit import deposit_text_versioned

        entry_id, _snap, _same = await deposit_text_versioned(
            session,
            text="a transcript",
            uri_r="claude-code://session/gated",
            source_type="CONVERSATION",
            mutability=Mutability.APPEND_ONLY,
            tags=["claude-code", "project:k1"],
        )

        def orphan(content: str) -> Particle:
            return Particle(
                content=content,
                confidence=Confidence(
                    value=0.7, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=None
                    )
                ],
                asserted_by="general-extractor",
                subject_ids=[],
            )

        a = orphan("`codec.py` is the standard-tier core of the interchange format.")
        b = orphan("`record_event` writes one operator event.")
        plain = orphan("The value of memory is cross-session by definition.")
        for p in (a, b, plain):
            await insert_particle(session, p)
        await session.flush()
        return a, b, plain

    @pytest.mark.asyncio
    async def test_recoverable_orphans_become_one_card(self, db_session: AsyncSession) -> None:
        a, b, plain = await self._orphans(db_session)
        cards = await collect_cards(db_session, semantic=False)
        [gated] = [c for c in cards if c.kind is CardKind.GATED_SUBJECTS]
        assert gated.particle_ids == sorted([a.id, b.id])
        assert gated.key == "gated_subjects"
        assert gated.suggested_gestures == ["relink", "snooze"]
        no_subject = [c.particle_ids for c in cards if c.kind is CardKind.NO_SUBJECT]
        assert no_subject == [[plain.id]]
        assert CurationCard.from_key("gated_subjects").kind is CardKind.GATED_SUBJECTS

    @pytest.mark.asyncio
    async def test_relink_gesture_links_every_member_in_place(
        self, db_session: AsyncSession
    ) -> None:
        a, b, _plain = await self._orphans(db_session)
        [card] = (
            await build_curation_queue(db_session, semantic=False, kind=CardKind.GATED_SUBJECTS)
        ).cards
        # A batch card is briefed with a sample, never all of its beliefs.
        assert len(card.particles) == 2
        msg = await apply_gesture(db_session, card, "relink")
        await db_session.commit()
        assert msg.startswith("Relinked 2 belief(s) to 2 subject(s)")
        for pid in (a.id, b.id):
            same = await get_particle(db_session, pid)
            assert same is not None and same.status is Status.ACTIVE and same.subject_ids
        after = await build_curation_queue(db_session, semantic=False, kind=CardKind.GATED_SUBJECTS)
        assert after.cards == []

    @pytest.mark.asyncio
    async def test_snapshot_card_resolves_by_batch_event_not_membership(
        self, db_session: AsyncSession
    ) -> None:
        from particles.operations.curation import rebuild_curation_snapshot
        from particles.store.event_store import EventRefKind, record_event

        a, _b, _plain = await self._orphans(db_session)
        await rebuild_curation_snapshot(db_session, semantic=False)
        # An unrelated gesture touching one member does not hide the batch.
        await record_event(
            db_session,
            actor="test",
            event_type=OperatorEventType.PARTICLE_RETRACTED,
            refs=[(EventRefKind.PARTICLE, a.id)],
        )
        await db_session.commit()
        served = await build_curation_queue(db_session, kind=CardKind.GATED_SUBJECTS)
        assert [c.key for c in served.cards] == ["gated_subjects"]

        [card] = served.cards
        await apply_gesture(db_session, card, "relink")
        await db_session.commit()
        served = await build_curation_queue(db_session, kind=CardKind.GATED_SUBJECTS)
        assert served.cards == []

    def test_audit_counts_the_beliefs_a_batch_card_covers(self) -> None:
        from particles.operations.audit import build_buckets

        card = CurationCard(
            kind=CardKind.GATED_SUBJECTS,
            particle_ids=["p1", "p2", "p3"],
            diagnostic="",
            suggested_gestures=gestures_for(CardKind.GATED_SUBJECTS),
        )
        [bucket] = build_buckets([card], exemplars_per_class=3)
        assert bucket.kind is CardKind.GATED_SUBJECTS and bucket.count == 3
