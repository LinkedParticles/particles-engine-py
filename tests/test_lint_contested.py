"""Tests for the composed-contested finder (operations/lint/contestedness.py).

The ADR's claim is that the hygiene surfaces stop disagreeing with recall
because they share one finder. These pin the three basis gates, the
disjunction, the two consumers, the ``badge_enabled`` restore,
and the §3 cost shape (a bounded query count, not a walk per belief).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    TrustLensDefinition,
    TrustLensUrlRule,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.curation.cards import CardKind
from particles.operations.curation.collect import collect_cards
from particles.operations.lint.contestedness import _check_contested, compute_store_contested
from particles.store.particle_store import insert_particle
from particles.store.relation_store import create_relation

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_A = "00000000-0000-0000-0000-00000000000a"
_B = "00000000-0000-0000-0000-00000000000b"
_INC = "00000000-0000-0000-0000-0000000000c0"
_STANCE = "00000000-0000-0000-0000-0000000000d0"


def _claim(content: str, pid: str, entry_id: str, confidence: float = 0.9) -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)],
    )


async def _add_entry(session: AsyncSession, entry_id: str, uri_r: str) -> None:
    from particles.corpus.store import CorpusEntryRow

    session.add(
        CorpusEntryRow(
            entry_id=entry_id,
            uri_r=uri_r,
            source_type="WEB_PAGE",
            mutability="MUTABLE",
            fetch_policy="LAZY",
            created_at=datetime.now(UTC),
            deposited_by="test",
        )
    )
    await session.flush()


async def _adopt_lens(session: AsyncSession) -> None:
    """Adopt one lens so the policy set reaches two and divergence is measurable."""
    from particles.store.lens_store import adopt_lens, materialise_lens

    lens = TrustLensDefinition(
        name="acme-numismatics",
        version=1,
        url_rules=[TrustLensUrlRule(scope="domain", pattern="sketchy.example", score=0.2)],
        extractor_weights={},
    )
    await materialise_lens(session, lens)
    await adopt_lens(session, lens.name)


async def _add_inconsistency(session: AsyncSession, target: str) -> None:
    """An open INCONSISTENCY particle referencing ``target`` (the marker)."""
    inc = Particle(
        id=_INC,
        content="INCONSISTENCY: conflict between two claims.",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=Status.INCONSISTENCY,
        provenance=[ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=target)],
    )
    await insert_particle(session, inc)


async def _add_dispute(session: AsyncSession, target: str) -> None:
    """A live DISPUTES stance (ACTIVE, with a holder) pointing at ``target``."""
    stance = _claim("Someone disputes it.", _STANCE, "e1")
    stance.properties = {"stance:holder": "reviewer@example.com"}
    await insert_particle(session, stance)
    await create_relation(
        session, _STANCE, target, RelationType.DISPUTES, RelationCreatedBy.HUMAN_REVIEW
    )


class TestBasisGates:
    """§2: each basis fires alone, and the badge is their disjunction."""

    @pytest.mark.asyncio
    async def test_no_basis_no_badge(self, db_session: AsyncSession) -> None:
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("A quiet claim.", _A, "e1"))
        await db_session.commit()

        census = await compute_store_contested(db_session)
        assert census.badges == {}
        assert [
            f for f in await _check_contested(db_session) if f.finding_type == "CONTESTED"
        ] == []

    @pytest.mark.asyncio
    async def test_inconsistency_basis_alone(self, db_session: AsyncSession) -> None:
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("A claim.", _A, "e1"))
        await _add_inconsistency(db_session, _A)
        await db_session.commit()

        census = await compute_store_contested(db_session)
        assert census.badges[_A].bases == ["inconsistency"]
        assert census.badges[_A].inconsistency_id == _INC

    @pytest.mark.asyncio
    async def test_stance_basis_alone_carries_the_caveat(self, db_session: AsyncSession) -> None:
        """binds this consumer: the M6 caveat travels with the badge."""
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("A disputed claim.", _A, "e1"))
        await _add_dispute(db_session, _A)
        await db_session.commit()

        badge = (await compute_store_contested(db_session)).badges[_A]
        assert badge.bases == ["stance"]
        assert badge.caveat

    @pytest.mark.asyncio
    async def test_endorsement_never_fires_the_stance_basis(self, db_session: AsyncSession) -> None:
        """§2: agreement is not contest."""
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("A supported claim.", _A, "e1"))
        stance = _claim("Someone endorses it.", _STANCE, "e1")
        stance.properties = {"stance:holder": "reviewer@example.com"}
        await insert_particle(db_session, stance)
        await create_relation(
            db_session, _STANCE, _A, RelationType.ENDORSES, RelationCreatedBy.HUMAN_REVIEW
        )
        await db_session.commit()

        assert (await compute_store_contested(db_session)).badges == {}

    @pytest.mark.asyncio
    async def test_holderless_stance_contributes_no_position(
        self, db_session: AsyncSession
    ) -> None:
        """a dangling / holder-less edge is not a declared position."""
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("A claim.", _A, "e1"))
        await insert_particle(db_session, _claim("No holder here.", _STANCE, "e1"))
        await create_relation(
            db_session, _STANCE, _A, RelationType.DISPUTES, RelationCreatedBy.HUMAN_REVIEW
        )
        await db_session.commit()

        assert (await compute_store_contested(db_session)).badges == {}

    @pytest.mark.asyncio
    async def test_divergence_absent_below_two_policies(self, db_session: AsyncSession) -> None:
        """§3: a one-policy store mints no fact-like divergence badge."""
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("Sketchy claim.", _A, "e1"))
        await db_session.commit()

        census = await compute_store_contested(db_session)
        assert census.badges == {}
        assert census.readings == {}

    @pytest.mark.asyncio
    async def test_divergence_basis_alone_with_a_lens(self, db_session: AsyncSession) -> None:
        await _adopt_lens(db_session)
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("Sketchy claim.", _A, "e1"))
        await db_session.commit()

        badge = (await compute_store_contested(db_session)).badges[_A]
        assert badge.bases == ["divergence"]
        assert badge.caveat is None

    @pytest.mark.asyncio
    async def test_bases_compose_as_a_disjunction(self, db_session: AsyncSession) -> None:
        """§1: the badge is a set of fired labels, in canonical order."""
        await _adopt_lens(db_session)
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("Sketchy, disputed claim.", _A, "e1"))
        await _add_dispute(db_session, _A)
        await _add_inconsistency(db_session, _A)
        await db_session.commit()

        badge = (await compute_store_contested(db_session)).badges[_A]
        assert badge.bases == ["stance", "divergence", "inconsistency"]
        assert badge.inconsistency_id == _INC
        assert badge.caveat


class TestFindingAndCard:
    """The finder's two consumers."""

    @pytest.mark.asyncio
    async def test_per_claim_finding_is_info_and_names_its_bases(
        self, db_session: AsyncSession
    ) -> None:
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("A claim.", _A, "e1"))
        await _add_inconsistency(db_session, _A)
        await db_session.commit()

        findings = [f for f in await _check_contested(db_session) if f.finding_type == "CONTESTED"]
        assert len(findings) == 1
        f = findings[0]
        # Disclosure, not defect — CONTRADICTION keeps ERROR.
        assert f.severity == "INFO"
        assert f.particle_id == _A
        assert f.contested_bases == ["inconsistency"]
        # The FULL id rides structurally (the detail prose truncates to 8
        # chars) so a client can link the evidence graph directly.
        assert f.inconsistency_id == _INC
        assert "inconsistency" in f.detail
        assert f.recommended_action and "particles review" in f.recommended_action

    @pytest.mark.asyncio
    async def test_card_carries_bases_and_a_basis_free_key(self, db_session: AsyncSession) -> None:
        """§2: the key stays `contested:<id>` so existing snoozes keep matching."""
        await _adopt_lens(db_session)
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("Sketchy claim.", _A, "e1"))
        await db_session.commit()

        cards = [
            c
            for c in await collect_cards(db_session, semantic=False)
            if c.kind is CardKind.CONTESTED
        ]
        assert len(cards) == 1
        assert cards[0].contested_bases == ["divergence"]
        # Divergence-only: no INCONSISTENCY exists, so no evidence link.
        assert cards[0].inconsistency_id is None
        assert cards[0].key == f"contested:{_A}"

    @pytest.mark.asyncio
    async def test_inconsistency_card_carries_the_full_evidence_id(
        self, db_session: AsyncSession
    ) -> None:
        """The CONTESTED card links the evidence graph structurally."""
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("A claim.", _A, "e1"))
        await _add_inconsistency(db_session, _A)
        await db_session.commit()

        cards = [
            c
            for c in await collect_cards(db_session, semantic=False)
            if c.kind is CardKind.CONTESTED
        ]
        assert len(cards) == 1
        assert cards[0].inconsistency_id == _INC

    @pytest.mark.asyncio
    async def test_divergence_only_card_offers_no_comment_gesture(
        self, db_session: AsyncSession
    ) -> None:
        """§6: `comment` routes to review.resolve, which needs an INCONSISTENCY."""
        await _adopt_lens(db_session)
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("Sketchy claim.", _A, "e1"))
        await db_session.commit()

        cards = [
            c
            for c in await collect_cards(db_session, semantic=False)
            if c.kind is CardKind.CONTESTED
        ]
        assert "comment" not in cards[0].suggested_gestures
        assert "affirm" in cards[0].suggested_gestures

    @pytest.mark.asyncio
    async def test_inconsistency_card_still_offers_comment(self, db_session: AsyncSession) -> None:
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("A claim.", _A, "e1"))
        await _add_inconsistency(db_session, _A)
        await db_session.commit()

        cards = [
            c
            for c in await collect_cards(db_session, semantic=False)
            if c.kind is CardKind.CONTESTED
        ]
        assert "comment" in cards[0].suggested_gestures

    @pytest.mark.asyncio
    async def test_retired_beliefs_get_no_card(self, db_session: AsyncSession) -> None:
        """The quarantined loser of a §6.6 conflict is not actionable curation work.

        Before the bespoke finder emitted a card for *every* id an
        INCONSISTENCY referenced, including the PROVENANCE_STALE side that no
        recall surface ever shows. Routing through the composer — which badges
        ACTIVE beliefs, exactly as recall does — is what removes it.
        """
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        loser = _claim("The quarantined side.", _B, "e1")
        loser.status = Status.PROVENANCE_STALE
        loser.status_reason = StatusReason.CONFLICT_PENDING
        await insert_particle(db_session, loser)
        await insert_particle(db_session, _claim("The surviving side.", _A, "e1"))
        inc = Particle(
            id=_INC,
            content="INCONSISTENCY: conflict between two claims.",
            confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            status=Status.INCONSISTENCY,
            provenance=[
                ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=_A),
                ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=_B),
            ],
        )
        await insert_particle(db_session, inc)
        await db_session.commit()

        badges = (await compute_store_contested(db_session)).badges
        assert set(badges) == {_A}


class TestSurfaceAgreement:
    """The point of the ADR: the hygiene surfaces agree, and the switch restores."""

    @pytest.mark.asyncio
    async def test_histogram_and_per_claim_finding_agree(self, db_session: AsyncSession) -> None:
        """One evaluation, two renderings — no singleton-vs-merge mismatch."""
        await _adopt_lens(db_session)
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await _add_entry(db_session, "e2", "https://trusted.example/p")
        await insert_particle(db_session, _claim("Sketchy claim.", _A, "e1"))
        await insert_particle(db_session, _claim("Trusted claim.", _B, "e2"))
        await db_session.commit()

        findings = await _check_contested(db_session)
        per_claim = {
            f.particle_id
            for f in findings
            if f.finding_type == "CONTESTED" and "divergence" in (f.contested_bases or [])
        }
        distribution = next(f for f in findings if f.finding_type == "CONTESTEDNESS_DISTRIBUTION")
        assert f"{len(per_claim)} with spread" in distribution.detail

    @pytest.mark.asyncio
    async def test_badge_disabled_restores_the_inconsistency_basis(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """off restores the pre-badge behaviour on hygiene too."""
        await _adopt_lens(db_session)
        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("Sketchy claim.", _A, "e1"))
        await _add_dispute(db_session, _A)
        await db_session.commit()

        assert (await compute_store_contested(db_session)).badges[_A].bases == [
            "stance",
            "divergence",
        ]

        # Flip the knob on the cached config object: a PARTICLES_CONFIG file
        # would force a reset_config() that re-resolves the engine registry and
        # closes this test's in-memory store out from under the session.
        monkeypatch.setattr(get_config().contestedness, "badge_enabled", False)
        census = await compute_store_contested(db_session)
        assert census.badges == {}  # no INCONSISTENCY here → no card, as before
        # The histogram predates the badge and is not gated by it.
        assert census.spreads

    @pytest.mark.asyncio
    async def test_evaluation_query_count_does_not_scale_with_the_store(
        self, db_session: AsyncSession
    ) -> None:
        """§3: the cost bound the ADR rests on — inverted bases, not a walk per belief.

        Ten times the beliefs must not mean ten times the round trips; a
        per-particle composer would issue a CO_EVIDENTIAL BFS each.
        """
        from sqlalchemy import event

        await _add_entry(db_session, "e1", "https://sketchy.example/p")
        await insert_particle(db_session, _claim("Claim 0.", _A, "e1"))
        await db_session.commit()

        counts: list[int] = []

        async def _measure() -> int:
            n = 0
            engine = db_session.get_bind()

            def _count(*_args: object, **_kwargs: object) -> None:
                nonlocal n
                n += 1

            event.listen(engine, "before_cursor_execute", _count)
            try:
                await compute_store_contested(db_session)
            finally:
                event.remove(engine, "before_cursor_execute", _count)
            return n

        counts.append(await _measure())
        for i in range(30):
            pid = f"00000000-0000-0000-0000-0000000{i:05d}"
            await insert_particle(db_session, _claim(f"Claim {i}.", pid, "e1"))
        await db_session.commit()
        counts.append(await _measure())

        assert counts[1] == counts[0], (
            f"query count grew with the store ({counts[0]} → {counts[1]}) — the "
            "per-basis inversion regressed to a per-particle walk"
        )
