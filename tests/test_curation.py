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

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles import llm
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
from particles.core.status import Status
from particles.operations.curation import (
    CardKind,
    CurationCard,
    DuplicateVerdict,
    apply_gesture,
    build_curation_queue,
)
from particles.operations.curation.cards import gestures_for
from particles.operations.curation.collect import collect_cards
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import (
    count_active_dependents,
    get_particle,
    insert_particle,
)
from particles.store.relation_store import get_co_evidential_group
from particles.store.subject_store import insert_subject


def _active(
    content: str,
    *,
    valid_until: datetime | None = None,
    asserted_at: datetime | None = None,
    dep_on: str | None = None,
    status: Status = Status.ACTIVE,
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
        await insert_particle(db_session, _inconsistency(contested.id))
        await db_session.flush()

        cards = (await build_curation_queue(db_session, semantic=False)).cards
        by_kind = {c.kind: c for c in cards}
        assert CardKind.STALE in by_kind
        assert CardKind.CONTESTED in by_kind
        assert by_kind[CardKind.STALE].particle_ids == [stale.id]
        assert by_kind[CardKind.CONTESTED].particle_ids == [contested.id]

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
            await build_curation_queue(db_session, semantic=False, kind=CardKind.CONTESTED)
        ).cards
        assert {c.kind for c in cards} == {CardKind.CONTESTED}

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
            kind=CardKind.CONTRADICTION,
            particle_ids=["p1"],
            diagnostic="x",
            suggested_gestures=gestures_for(CardKind.CONTRADICTION),
        )
        with pytest.raises(ValueError, match="particles review"):
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
        """When the LLM is unavailable the judge defaults each pair to UNSURE
        ; the card still surfaces with that verdict and is NOT demoted
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
