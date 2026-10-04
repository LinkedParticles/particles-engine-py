# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for same-subject update supersession.

The failure this closes, as the rot benchmark measured it: a value
that changes across sessions ("I moved to Denver") never meets the value it
replaces — extraction reconciled per corpus entry and gated at 0.80 cosine —
so old and new both stayed ACTIVE and the old one ranked first more often than
not. Pinned here:

* **the keys** — the about-subject of a candidate and of a stored particle, and
  which particles may be candidates at all;
* **the lineage rule** — rung 2.5 fires only when trust cannot tell the sides
  apart, both are extractor-asserted, and both are strictly dated;
* **the pipeline** — a cross-entry update supersedes; an out-of-order older
  claim is stored demoted; ``multi`` stores do it too; a revert re-mints
  (outside the judgment set); a different lineage falls through; the
  switch really switches it off;
* **the slot rule** — a contradiction retires a claim only when the
  update probe says both claims fill one slot and the later one gives it a new
  value, on every route rung 2.5 runs on.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from particles.core.conflict_resolution import ConflictVerdict, SlotVerdict, resolve_conflict
from particles.core.schema import (
    ClaimTerm,
    Confidence,
    ContributorRef,
    CorpusEntry,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    SourceType,
    StructuredClaim,
    TermKind,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.ingest.duplicate_suppression import JUDGMENT_RETIREMENTS
from particles.ingest.pipeline import _has_update_signal as _real_update_signal
from particles.ingest.update_supersession import (
    SubjectIndex,
    candidate_subject_names,
    is_extractor_asserted,
    is_reconcilable,
    own_assertion_order,
    particle_subject_ids,
    update_order,
)

T0 = datetime(2026, 6, 1, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Keys and candidacy (pure)
# ---------------------------------------------------------------------------


def _cand(subjects: list[str], triple_subject: str | None = None) -> CandidateParticle:
    return CandidateParticle(
        content="x",
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=subjects,
        structured_claim=(
            StructuredClaim(
                subject=ClaimTerm(kind=TermKind.TOKEN, value=triple_subject),
                predicate=ClaimTerm(kind=TermKind.TOKEN, value="lives in"),
                object=ClaimTerm(kind=TermKind.TOKEN, value="Denver"),
                structurizer_id="t",
                structurizer_version="1",
            )
            if triple_subject
            else None
        ),
    )


def _p(
    content: str = "The user's home city is Boston.",
    *,
    subject_ids: list[str] | None = None,
    calibration: CalibrationSource = CalibrationSource.EXTRACTOR_DIRECT,
    asserted_by: str = "general-extractor",
    source_ref: bool = True,
    triple_subject_id: str | None = None,
    properties: dict[str, object] | None = None,
) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=calibration),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=asserted_by,
        asserted_at=T0,
        subject_ids=subject_ids if subject_ids is not None else ["u"],
        properties=properties,
        provenance=(
            [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id="s")]
            if source_ref
            else [ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id="p")]
        ),
        structured_claim=(
            StructuredClaim(
                subject=ClaimTerm(kind=TermKind.TOKEN, value="user"),
                predicate=ClaimTerm(kind=TermKind.TOKEN, value="lives in"),
                object=ClaimTerm(kind=TermKind.TOKEN, value="Boston"),
                subject_id=triple_subject_id,
                structurizer_id="t",
                structurizer_version="1",
            )
            if triple_subject_id
            else None
        ),
    )


class TestKeys:
    def test_the_triple_subject_is_the_key(self) -> None:
        cand = _cand(["Boston", "the user", "Denver"], triple_subject="The User")
        assert candidate_subject_names(cand) == ["the user"]

    def test_sole_subject_is_the_key_when_there_is_no_triple(self) -> None:
        assert candidate_subject_names(_cand(["the user"])) == ["the user"]

    def test_a_mentioned_entity_is_not_a_key(self) -> None:
        # The correction: pairing on every named subject flooded a
        # fixed probe budget with same-value restatements, which outrank a real
        # old→new pair on cosine. A candidate with no about-subject has no key.
        assert candidate_subject_names(_cand(["Boston", "Denver"])) == []

    def test_stored_subject_key_is_the_about_subject(self) -> None:
        assert particle_subject_ids(_p(subject_ids=["u", "boston"])) == []
        assert particle_subject_ids(_p(subject_ids=["u"], triple_subject_id="v")) == ["v"]
        assert particle_subject_ids(_p(subject_ids=["u"])) == ["u"]

    def test_reconcilable_excludes_what_the_same_entry_search_excludes(self) -> None:
        assert is_reconcilable(_p())
        assert not is_reconcilable(_p(properties={"extraction:scope": "DOCUMENT_META"}))
        assert not is_reconcilable(_p(properties={"extraction:polarity": "DECLINED"}))
        assert not is_reconcilable(_p(properties={"stance:holder": "someone"}))

    def test_extractor_asserted(self) -> None:
        assert is_extractor_asserted(_p())
        assert not is_extractor_asserted(_p(calibration=CalibrationSource.HUMAN_REVIEW))
        assert not is_extractor_asserted(_p(calibration=CalibrationSource.AGENT_ASSERTED))
        assert not is_extractor_asserted(_p(asserted_by="mcp:claude-code"))
        assert not is_extractor_asserted(_p(source_ref=False))


class TestSubjectIndex:
    @staticmethod
    def _emb(*xs: float) -> Any:
        v = np.array(xs, dtype=np.float32)
        return v / np.linalg.norm(v)

    def test_candidates_by_subject_floor_limit_and_skip(self) -> None:
        near, mid, far = _p("a"), _p("b"), _p("c")
        other_subject = _p("d", subject_ids=["v"])
        index = SubjectIndex.build(
            [
                (near, self._emb(1, 0.1)),
                (mid, self._emb(1, 1)),
                (far, self._emb(0, 1)),
                (other_subject, self._emb(1, 0)),
            ]
        )
        q = self._emb(1, 0)
        assert index.candidates(["u"], q, floor=0.5, limit=3) == [near, mid]
        assert index.candidates(["u"], q, floor=0.5, limit=1) == [near]
        assert index.candidates(["u"], q, floor=0.5, limit=3, skip_ids=[near.id]) == [mid]
        index.retired.add(mid.id)
        assert index.candidates(["u"], q, floor=0.5, limit=3) == [near]

    def test_several_keys_never_offer_one_particle_twice(self) -> None:
        # The plural signature outlived the breadth it was built
        # for: a key set still de-duplicates.
        shared = _p("a", subject_ids=["u"], triple_subject_id="boston")
        index = SubjectIndex.build([(shared, self._emb(1, 0.1))])
        q = self._emb(1, 0)
        assert index.candidates(["boston"], q, floor=0.5, limit=3) == [shared]
        assert index.candidates(["u", "boston"], q, floor=0.5, limit=3) == [shared]

    def test_build_indexes_under_the_about_subject_only(self) -> None:
        p = _p("a", subject_ids=["u"], triple_subject_id="boston")
        index = SubjectIndex.build([(p, self._emb(1, 0))])
        assert set(index.by_subject) == {"boston"}

    def test_build_excludes_ineligible_and_unembedded(self) -> None:
        keep, no_emb, excluded = _p("a"), _p("b"), _p("c")
        index = SubjectIndex.build(
            [(keep, self._emb(1, 0)), (no_emb, None), (excluded, self._emb(1, 0))],
            exclude_ids=[excluded.id],
        )
        assert [p for p, _ in index.by_subject["u"]] == [keep]


# ---------------------------------------------------------------------------
# The lineage + dating rule (pure)
# ---------------------------------------------------------------------------


def _entry(
    *,
    uri: str = "claude-code://session/a",
    source_type: str = "CONVERSATION",
    published: datetime | None = T0,
    author: str | None = None,
    contributors: list[ContributorRef] | None = None,
) -> tuple[CorpusEntry, str]:
    snap = Snapshot(content_hash="0" * 64, content_published_at=published, author_id=author)
    entry = CorpusEntry(
        uri_r=uri,
        source_type=source_type,
        snapshots=[snap],
        contributors=contributors,
        deposited_by="test",
    )
    return entry, snap.snapshot_id


class TestUpdateOrder:
    def _order(self, new_kw: dict[str, Any], old_kw: dict[str, Any], **particles: Any) -> Any:
        ne, ns = _entry(**new_kw)
        oe, os_ = _entry(**old_kw)
        return update_order(particles.get("new", _p()), ne, ns, particles.get("old", _p()), oe, os_)

    def test_newer_wins_either_way(self) -> None:
        later = {"uri": "claude-code://session/b", "published": T0 + timedelta(days=3)}
        assert self._order(later, {}) == 1
        assert self._order({}, later) == -1

    def test_equal_or_undated_does_not_qualify(self) -> None:
        assert self._order({}, {}) is None
        assert self._order({"published": None}, {}) is None

    @pytest.mark.parametrize(
        "diff",
        [
            {"uri": "https://example.org/page"},
            {"source_type": "WEB_PAGE"},
            {"author": "github:someone"},
            {"contributors": [ContributorRef(id="github:alice", role="author")]},
        ],
    )
    def test_any_trust_key_difference_breaks_the_lineage(self, diff: dict[str, Any]) -> None:
        newer = {"published": T0 + timedelta(days=1), **diff}
        assert self._order(newer, {}) is None

    def test_the_same_contributor_on_two_dates_is_one_lineage(self) -> None:
        def ref(day: int) -> ContributorRef:
            return ContributorRef(id="github:alice", role="author", at=T0 + timedelta(days=day))

        newer = {"published": T0 + timedelta(days=1), "contributors": [ref(1)]}
        assert self._order(newer, {"contributors": [ref(0)]}) == 1

    def test_operator_or_agent_side_never_qualifies(self) -> None:
        newer = {"published": T0 + timedelta(days=1)}
        human = _p(calibration=CalibrationSource.HUMAN_REVIEW)
        agent = _p(asserted_by="mcp:claude-code")
        assert self._order(newer, {}, old=human) is None
        assert self._order(newer, {}, new=agent) is None


class TestAttributionAndAgents:
    def test_multi_regime_requires_attribution(self) -> None:
        ne, ns = _entry(published=T0 + timedelta(days=1))
        oe, os_ = _entry()
        # single regime: an anonymous lineage is one principal by declaration
        assert update_order(_p(), ne, ns, _p(), oe, os_) == 1
        # multi regime: anonymous means unknown, so the rung fails closed
        assert update_order(_p(), ne, ns, _p(), oe, os_, require_attribution=True) is None

    def test_multi_regime_fires_when_attributed(self) -> None:
        ne, ns = _entry(published=T0 + timedelta(days=1), author="github:alice")
        oe, os_ = _entry(author="github:alice")
        assert update_order(_p(), ne, ns, _p(), oe, os_, require_attribution=True) == 1

    def test_an_agent_supersedes_only_its_own_earlier_assertion(self) -> None:
        def agent(at: datetime, who: str = "mcp:claude-code") -> Particle:
            p = _p(calibration=CalibrationSource.AGENT_ASSERTED, asserted_by=who)
            return p.model_copy(update={"asserted_at": at})

        old, new = agent(T0), agent(T0 + timedelta(hours=1))
        assert own_assertion_order(new, old) == 1
        assert own_assertion_order(old, new) is None  # older assertion never wins
        assert own_assertion_order(agent(T0 + timedelta(hours=1), "mcp:other"), old) is None
        assert own_assertion_order(new, _p()) is None  # vs an extracted claim
        assert own_assertion_order(new, _p(calibration=CalibrationSource.HUMAN_REVIEW)) is None


class TestLadderRung:
    def test_rung_is_not_gated_on_single_trust_order(self) -> None:
        a, b = _p(), _p()
        assert (
            resolve_conflict(a, b, update_order=1, single_trust_order=False)
            is ConflictVerdict.UPDATE_SUPERSEDES
        )

    def test_no_order_means_todays_behaviour(self) -> None:
        a, b = _p(), _p()
        assert resolve_conflict(a, b) is ConflictVerdict.INCONSISTENT

    def test_status_reason_is_not_a_judgment_retirement(self) -> None:
        assert not any(r is StatusReason.SUPERSEDED_BY_UPDATE for _, r in JUDGMENT_RETIREMENTS)


# ---------------------------------------------------------------------------
# Through the real pipeline
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9']+")
_CITIES = ("Boston", "Denver", "Lisbon")


class _BagOfWords:
    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                out[i, int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16) % 256] += 1.0
            out[i] /= max(float(np.linalg.norm(out[i])), 1e-9)
        return out


class _OneClaim:
    """An extractor emitting one scripted claim per deposited text."""

    EXTRACTOR_ID = "test-extractor"
    EXTRACTOR_VERSION = "1"

    def __init__(self) -> None:
        self.claims: dict[bytes, str] = {}

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> ExtractionResult:
        claim = self.claims[content]
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=claim,
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["the user"],
                )
            ]
        )


FAVOURITE = "Sandeep's Curry House is the user's favourite place to eat."
NOT_YET = "The user has not yet found a regular place to eat."
DELHI = "The user lives in Lajpat Nagar, Delhi."
MUMBAI = "The user is living in Bandra, Mumbai."

#: A fixed fact stated twice with different values: the sweep retired the
#: earlier figure for the later one on the kept stores.
KAWAI_EARLIER = "The Kawai ES110 digital piano costs $699 to $799."
KAWAI_LATER = "The Kawai ES110 digital piano costs $299."

#: Contradicting pairs outside the city rule, each with the update probe's
#: scripted verdict: whether the two claims fill one slot, and of
#: what kind. The first is the 2026-09-27 Delhi scenario: the
#: contradiction probe confirmed it, and rung 2.5 retired the favourite on its
#: date alone.
_SCRIPTED: dict[frozenset[str], SlotVerdict] = {
    frozenset((FAVOURITE, NOT_YET)): SlotVerdict.DIFFERENT,
    frozenset(
        (
            "The user's favourite restaurant is Sandeep's Curry House.",
            "The user's favourite restaurant is now Bombay Canteen.",
        )
    ): SlotVerdict.CHANGES,
    frozenset(("The user is vegetarian.", "The user eats fish now.")): SlotVerdict.CHANGES,
    frozenset((DELHI, MUMBAI)): SlotVerdict.CHANGES,
    frozenset((KAWAI_EARLIER, KAWAI_LATER)): SlotVerdict.FIXED,
}


async def _contradicts(a: str, b: str) -> bool:
    """Scripted probe: two claims contradict iff they name different cities."""
    if frozenset((a, b)) in _SCRIPTED:
        return True
    ca = {c for c in _CITIES if c in a}
    cb = {c for c in _CITIES if c in b}
    return bool(ca and cb and ca != cb)


async def _updates(earlier: str, later: str) -> SlotVerdict:
    """Scripted update probe: a home city is one slot that changes; the scripted pairs as listed."""
    scripted = _SCRIPTED.get(frozenset((earlier, later)))
    if scripted is not None:
        return scripted
    return SlotVerdict.CHANGES if await _contradicts(earlier, later) else SlotVerdict.DIFFERENT


@pytest.fixture
def bow() -> Generator[AsyncMock, None, None]:
    """A bag-of-words encoder, and both probes scripted; yields the update probe."""
    from particles import embeddings as ep

    original = ep._embedding_model
    ep.set_embedding_model(_BagOfWords())  # type: ignore[arg-type]
    update_probe = AsyncMock(side_effect=_updates)
    try:
        with (
            patch(
                "particles.ingest.pipeline._has_contradiction_signal",
                AsyncMock(side_effect=_contradicts),
            ),
            patch("particles.ingest.pipeline._has_update_signal", update_probe),
        ):
            yield update_probe
    finally:
        ep.set_embedding_model(original)


async def _say(
    session: Any,
    extractor: _OneClaim,
    claim: str,
    *,
    day: int,
    uri: str | None = None,
    source_type: str = SourceType.CONVERSATION,
    author: str | None = None,
) -> list[Particle]:
    from particles.corpus.deposit import deposit_text_versioned
    from particles.ingest.pipeline import extract_snapshot

    text = f"day {day}: {claim}"
    extractor.claims[text.encode()] = claim
    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text=text,
        uri_r=uri or f"claude-code://session/{uuid.uuid4()}",
        source_type=source_type,
        mutability=Mutability.STABLE,
        content_published_at=T0 + timedelta(days=day),
    )
    if author is not None:
        # No deposit path stamps attribution yet (leaves that to the
        # multi-user MVP), so the test writes it where a harvester would.
        from particles.corpus.store import SnapshotRow

        row = await session.get(SnapshotRow, snapshot_id)
        assert row is not None
        row.author_id = author
        await session.flush()
    await session.commit()
    written = await extract_snapshot(session, entry_id, snapshot_id, extractor=extractor)
    await session.commit()
    return written


async def _by_content(session: Any, content: str) -> list[Particle]:
    from particles.store.particle_store import get_particles_by_status

    out: list[Particle] = []
    for status in Status:
        out += [p for p in await get_particles_by_status(session, status) if p.content == content]
    return out


BOSTON = "The user's home city is Boston."
DENVER = "The user's home city is Denver."


@pytest.mark.asyncio
class TestPipeline:
    async def test_a_cross_session_update_supersedes(self, db_session: Any, bow: None) -> None:
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        [new] = await _say(db_session, ex, DENVER, day=9)
        [old] = await _by_content(db_session, BOSTON)
        assert old.status is Status.PROVENANCE_STALE
        assert old.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        assert new.status is Status.ACTIVE and new.supersedes == old.id

    async def test_an_out_of_order_older_claim_is_stored_demoted(
        self, db_session: Any, bow: None
    ) -> None:
        ex = _OneClaim()
        await _say(db_session, ex, DENVER, day=9)
        [late_arrival] = await _say(db_session, ex, BOSTON, day=1)
        assert late_arrival.status is Status.PROVENANCE_STALE
        assert late_arrival.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        [current] = await _by_content(db_session, DENVER)
        assert current.status is Status.ACTIVE

    async def test_multi_store_without_attribution_leaves_both_active(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """in a shared store an anonymous lineage is not one principal.

        Failing closed means *not acting* — the outcome before, both
        claims ACTIVE — not quarantining the newer claim behind a review item,
        which would leave the store on the older value.
        """
        from particles.config import get_config

        monkeypatch.setattr(get_config().reconciliation, "store_mode", "multi")
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        written = await _say(db_session, ex, DENVER, day=9)
        assert [p.status for p in written] == [Status.ACTIVE]
        [old] = await _by_content(db_session, BOSTON)
        assert old.status is Status.ACTIVE

    async def test_multi_store_with_attribution_updates(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        monkeypatch.setattr(get_config().reconciliation, "store_mode", "multi")
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1, author="github:alice")
        [new] = await _say(db_session, ex, DENVER, day=9, author="github:alice")
        assert new.status is Status.ACTIVE
        [old] = await _by_content(db_session, BOSTON)
        assert old.status_reason is StatusReason.SUPERSEDED_BY_UPDATE

    async def test_a_revert_re_mints_instead_of_being_held(
        self, db_session: Any, bow: None
    ) -> None:
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        await _say(db_session, ex, DENVER, day=9)
        [back] = await _say(db_session, ex, BOSTON, day=20)
        assert back.status is Status.ACTIVE
        [denver] = await _by_content(db_session, DENVER)
        assert denver.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        assert back.supersedes == denver.id

    async def test_a_different_lineage_never_supersedes(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        # multi, so rung 2 cannot settle it on trust either
        monkeypatch.setattr(get_config().reconciliation, "store_mode", "multi")
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        written = await _say(
            db_session,
            ex,
            DENVER,
            day=9,
            uri="https://people.example/profile",
            source_type=SourceType.WEB_PAGE,
        )
        assert [p.status for p in written] == [Status.ACTIVE]
        [old] = await _by_content(db_session, BOSTON)
        assert old.status is Status.ACTIVE
        assert old.status_reason is None

    async def test_the_switch_turns_it_off(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", False)
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        await _say(db_session, ex, DENVER, day=9)
        assert [p.status for p in await _by_content(db_session, BOSTON)] == [Status.ACTIVE]
        assert [p.status for p in await _by_content(db_session, DENVER)] == [Status.ACTIVE]

    async def test_several_stale_values_converge_in_one_pass(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        # Two stale values already ACTIVE side by side (a pre-0268 store).
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", False)
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        await _say(db_session, ex, DENVER, day=5)
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", True)
        [lisbon] = await _say(db_session, ex, "The user's home city is Lisbon.", day=9)
        assert lisbon.status is Status.ACTIVE
        for stale in (BOSTON, DENVER):
            [p] = await _by_content(db_session, stale)
            assert p.status_reason is StatusReason.SUPERSEDED_BY_UPDATE, stale


# ---------------------------------------------------------------------------
# an update retires a claim only when both claims fill one slot
# ---------------------------------------------------------------------------

#: The synthetic updates the slot rule must keep: each later claim gives a new
#: value for the earlier claim's slot.
SAME_SLOT_UPDATES = [
    (BOSTON, DENVER),
    (
        "The user's favourite restaurant is Sandeep's Curry House.",
        "The user's favourite restaurant is now Bombay Canteen.",
    ),
    ("The user is vegetarian.", "The user eats fish now."),
]


@pytest.mark.asyncio
class TestSlotRule:
    """The update probe gates rung 2.5 on every route it runs on.

    The contradiction probe stays scripted as before; what these tests pin is
    that its YES is no longer enough to retire a claim on a date.
    """

    @pytest.fixture(autouse=True)
    def _floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The bag-of-words encoder scores these short pairs lower than the
        # production encoder does (the Delhi pair met the 0.45 floor there).
        # The floor finds candidates and decides nothing, so lowering it only
        # makes sure every pair here reaches the probes.
        from particles.config import get_config

        monkeypatch.setattr(get_config().reconciliation.update_supersession, "subject_floor", 0.3)

    async def test_a_situational_claim_never_retires_a_lasting_preference(
        self, db_session: Any, bow: AsyncMock
    ) -> None:
        from particles.store.particle_store import get_particles_by_status

        ex = _OneClaim()
        await _say(db_session, ex, FAVOURITE, day=4)
        [new] = await _say(db_session, ex, NOT_YET, day=20)
        [favourite] = await _by_content(db_session, FAVOURITE)
        assert favourite.status is Status.ACTIVE and favourite.status_reason is None
        assert new.status is Status.ACTIVE and new.supersedes is None
        # The pair was asked, earlier claim first, and nothing went to review:
        # a cross-entry pair in different slots leaves the store as it was.
        bow.assert_awaited_once_with(FAVOURITE, NOT_YET)
        assert await get_particles_by_status(db_session, Status.INCONSISTENCY) == []

    @pytest.mark.parametrize(("old", "new"), SAME_SLOT_UPDATES)
    async def test_a_new_value_for_the_same_slot_still_supersedes(
        self, db_session: Any, bow: AsyncMock, old: str, new: str
    ) -> None:
        ex = _OneClaim()
        await _say(db_session, ex, old, day=4)
        [written] = await _say(db_session, ex, new, day=20)
        [stale] = await _by_content(db_session, old)
        assert stale.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        assert written.status is Status.ACTIVE and written.supersedes == stale.id
        bow.assert_awaited_once_with(old, new)

    async def test_an_out_of_order_claim_is_asked_in_date_order(
        self, db_session: Any, bow: AsyncMock
    ) -> None:
        ex = _OneClaim()
        await _say(db_session, ex, DENVER, day=9)
        [late_arrival] = await _say(db_session, ex, BOSTON, day=1)
        assert late_arrival.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        bow.assert_awaited_once_with(BOSTON, DENVER)

    async def test_an_incomplete_update_probe_retires_nothing(
        self, db_session: Any, bow: AsyncMock
    ) -> None:
        bow.side_effect = _real_update_signal
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        with patch("particles.ingest.pipeline._llm_slot_verdict", AsyncMock(return_value=None)):
            await _say(db_session, ex, DENVER, day=9)
        [old] = await _by_content(db_session, BOSTON)
        assert old.status is Status.ACTIVE

    async def test_a_pair_the_rung_cannot_order_is_never_asked(
        self, db_session: Any, bow: AsyncMock
    ) -> None:
        # A different lineage: rung 2.5 cannot fire, so the question would be spend.
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        await _say(
            db_session,
            ex,
            DENVER,
            day=9,
            uri="https://people.example/profile",
            source_type=SourceType.WEB_PAGE,
        )
        bow.assert_not_awaited()

    async def test_the_sweep_leaves_a_different_slot_pair_alone(
        self, db_session: Any, bow: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config
        from particles.operations import reconcile as reconcile_mod

        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", False)
        ex = _OneClaim()
        await _say(db_session, ex, FAVOURITE, day=4)
        await _say(db_session, ex, NOT_YET, day=20)
        await _say(db_session, ex, BOSTON, day=4)
        await _say(db_session, ex, DENVER, day=20)
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", True)
        monkeypatch.setattr(
            reconcile_mod, "_has_contradiction_signal", AsyncMock(side_effect=_contradicts)
        )
        update_probe = AsyncMock(side_effect=_updates)
        monkeypatch.setattr(reconcile_mod, "_has_update_signal", update_probe)

        summary = await reconcile_mod.reconcile_updates(db_session)
        assert summary["demoted"] == 1
        assert summary["different_slot"] == 1
        assert summary["update_probed"] == 2
        [favourite] = await _by_content(db_session, FAVOURITE)
        assert favourite.status is Status.ACTIVE
        [boston] = await _by_content(db_session, BOSTON)
        assert boston.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        update_probe.assert_any_await(FAVOURITE, NOT_YET)

    async def test_an_agent_keeps_its_own_belief_in_another_slot(
        self, db_session: Any, bow: AsyncMock
    ) -> None:
        agent = TestAgentAssertionPath()
        await agent._assert(db_session, FAVOURITE, at=T0)
        written = await agent._assert(db_session, NOT_YET, at=T0 + timedelta(hours=1))
        [favourite] = await _by_content(db_session, FAVOURITE)
        assert favourite.status is Status.ACTIVE
        # The assertion path fails closed: the pair goes to review.
        assert written.status is Status.INCONSISTENCY
        bow.assert_awaited_once_with(FAVOURITE, NOT_YET)


@pytest.mark.asyncio
class TestFixedSlotRule:
    """A fixed slot given two values goes to review, not to rung 2.5.

    The update probe confirms the pair fills one slot, as's
    residuals, and names the slot fixed: a later price for the same product as
    stated once is not a newer price. Every route must keep the earlier claim.
    """

    @pytest.fixture(autouse=True)
    def _floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import get_config

        monkeypatch.setattr(get_config().reconciliation.update_supersession, "subject_floor", 0.3)

    async def test_extraction_sends_the_pair_to_rung_3(
        self, db_session: Any, bow: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config
        from particles.store.particle_store import get_particles_by_status

        # The second reading is a separate gate with its own tests;
        # off here so the pair reaches the ladder on the scripted probes alone.
        monkeypatch.setattr(get_config().extraction, "verify_conflicts", False)
        ex = _OneClaim()
        await _say(db_session, ex, KAWAI_EARLIER, day=4)
        await _say(db_session, ex, KAWAI_LATER, day=20)
        [earlier] = await _by_content(db_session, KAWAI_EARLIER)
        assert earlier.status is Status.ACTIVE and earlier.status_reason is None
        [later] = await _by_content(db_session, KAWAI_LATER)
        # Rung 3 at a write: the record opens, and the newcomer waits behind it
        # for review. Nothing is retired by date.
        assert later.status_reason is StatusReason.CONFLICT_PENDING
        assert later.supersedes is None
        [record] = await get_particles_by_status(db_session, Status.INCONSISTENCY)
        named = [
            r.corpus_entry_id for r in record.provenance if r.type is ProvenanceRefType.PARTICLE
        ]
        assert named[:2] == [earlier.id, later.id]
        bow.assert_awaited_once_with(KAWAI_EARLIER, KAWAI_LATER)

    async def test_the_agent_path_never_retires_its_own_fixed_fact(
        self, db_session: Any, bow: AsyncMock
    ) -> None:
        agent = TestAgentAssertionPath()
        await agent._assert(db_session, KAWAI_EARLIER, at=T0)
        written = await agent._assert(db_session, KAWAI_LATER, at=T0 + timedelta(hours=1))
        [earlier] = await _by_content(db_session, KAWAI_EARLIER)
        assert earlier.status is Status.ACTIVE
        assert written.status is Status.INCONSISTENCY
        bow.assert_awaited_once_with(KAWAI_EARLIER, KAWAI_LATER)

    async def _backlog(self, session: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
        from particles.config import get_config
        from particles.operations import reconcile as reconcile_mod

        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", False)
        ex = _OneClaim()
        await _say(session, ex, KAWAI_EARLIER, day=4)
        await _say(session, ex, KAWAI_LATER, day=20)
        await _say(session, ex, BOSTON, day=4)
        await _say(session, ex, DENVER, day=20)
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", True)
        monkeypatch.setattr(
            reconcile_mod, "_has_contradiction_signal", AsyncMock(side_effect=_contradicts)
        )
        monkeypatch.setattr(reconcile_mod, "_has_update_signal", AsyncMock(side_effect=_updates))
        return reconcile_mod

    async def test_the_sweep_opens_a_review_and_keeps_both_active(
        self, db_session: Any, bow: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.core.contradiction_disclosure import ORIGIN_KEY
        from particles.store.particle_store import get_particles_by_status

        reconcile_mod = await self._backlog(db_session, monkeypatch)
        summary = await reconcile_mod.reconcile_updates(db_session)
        # The home city still updates; the price does not.
        assert summary["demoted"] == 1
        assert summary["fixed_slot"] == 1
        [earlier] = await _by_content(db_session, KAWAI_EARLIER)
        [later] = await _by_content(db_session, KAWAI_LATER)
        assert earlier.status is Status.ACTIVE and later.status is Status.ACTIVE
        assert later.supersedes is None
        [record] = await get_particles_by_status(db_session, Status.INCONSISTENCY)
        named = [
            r.corpus_entry_id for r in record.provenance if r.type is ProvenanceRefType.PARTICLE
        ]
        assert named[:2] == [earlier.id, later.id]
        assert record.asserted_by == reconcile_mod.UPDATE_SWEEP_ACTOR
        assert (record.properties or {})[ORIGIN_KEY] == reconcile_mod.FIXED_SLOT_ORIGIN
        [review] = summary["reviews"]
        assert review["record_id"] == record.id

        # Idempotent: the ledger remembers the pair may not retire, so a re-run
        # asks nothing and opens nothing.
        again = await reconcile_mod.reconcile_updates(db_session)
        assert again["fixed_slot"] == 0 and again["previously_cleared"] >= 1
        assert len(await get_particles_by_status(db_session, Status.INCONSISTENCY)) == 1

    async def test_a_pair_already_under_review_is_not_opened_twice(
        self, db_session: Any, bow: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.store.particle_store import get_particles_by_status

        reconcile_mod = await self._backlog(db_session, monkeypatch)
        await reconcile_mod.reconcile_updates(db_session)
        # A reworded prompt invalidates the ledger, so the pair is asked again.
        monkeypatch.setattr(reconcile_mod, "update_prompt_hash", lambda: "reworded")
        again = await reconcile_mod.reconcile_updates(db_session)
        [review] = again["reviews"]
        assert review.get("already_open") is True and "record_id" not in review
        assert len(await get_particles_by_status(db_session, Status.INCONSISTENCY)) == 1

    async def test_a_dry_run_opens_nothing(
        self, db_session: Any, bow: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.store.particle_store import get_particles_by_status

        reconcile_mod = await self._backlog(db_session, monkeypatch)
        dry = await reconcile_mod.reconcile_updates(db_session, dry_run=True)
        assert dry["fixed_slot"] == 1 and "record_id" not in dry["reviews"][0]
        assert await get_particles_by_status(db_session, Status.INCONSISTENCY) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("slot", "expected"),
    [
        (SlotVerdict.CHANGES, (True, True)),
        (SlotVerdict.FIXED, (True, False)),
        (SlotVerdict.DIFFERENT, (True, False)),
        (None, (True, None)),
    ],
)
async def test_update_checks_reports_whether_the_sweep_would_retire(
    monkeypatch: pytest.MonkeyPatch, slot: SlotVerdict | None, expected: tuple[bool, bool | None]
) -> None:
    """The real-pairs benchmark reads the sweep's decision; a fixed slot keeps both."""
    from particles.operations import reconcile as reconcile_mod

    monkeypatch.setattr(reconcile_mod, "_has_contradiction_signal", AsyncMock(return_value=True))
    monkeypatch.setattr(reconcile_mod, "_has_update_signal", AsyncMock(return_value=slot))
    assert await reconcile_mod.update_checks(KAWAI_EARLIER, KAWAI_LATER) == expected


class TestSlotVerdictParser:
    """The update probe's reply carries the slot's kind before its verdict."""

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("REASON: home city.\nSLOT: CHANGES\nVERDICT: YES", SlotVerdict.CHANGES),
            ("REASON: the author.\nSLOT: FIXED\nVERDICT: YES", SlotVerdict.FIXED),
            ("REASON: two attributes.\nSLOT: NONE\nVERDICT: NO", SlotVerdict.DIFFERENT),
            # A NO needs no slot line: it is not an update whatever the kind.
            ("REASON: two attributes.\nVERDICT: NO", SlotVerdict.DIFFERENT),
            ("REASON: x.\n**SLOT:** fixed\n**VERDICT:** yes", SlotVerdict.FIXED),
            # A YES without a usable kind is off protocol: no verdict, keep both.
            ("REASON: home city.\nVERDICT: YES", None),
            ("REASON: x.\nSLOT: NONE\nVERDICT: YES", None),
            ("REASON: x.\nSLOT: CHANGES\nSLOT: FIXED\nVERDICT: YES", None),
            # Cut before the verdict.
            ("REASON: the author.\nSLOT: FIXED", None),
        ],
    )
    def test_reply(self, reply: str, expected: SlotVerdict | None) -> None:
        from particles.ingest.pipeline import _slot_verdict

        assert _slot_verdict(reply) is expected

    def test_prompt_asks_for_the_kind_before_the_verdict(self) -> None:
        from particles.ingest.pipeline import _update_prompt

        prompt = _update_prompt("A.", "B.")
        assert prompt.index("SLOT: CHANGES") < prompt.index("VERDICT: YES")
        assert prompt.endswith("Claim A: A.\n\nClaim B: B.")


# ---------------------------------------------------------------------------
# the backlog sweep, tool turns, persona folding, restated claims
# ---------------------------------------------------------------------------


class TestToolTurns:
    def test_marks_transcript_and_distiller_forms(self) -> None:
        from particles.extraction.tool_turns import TOOL_TURN_LABEL, mark_tool_turns

        text = (
            "user: where do I live?\n"
            "assistant: let me check\n"
            "tool: Profile snippet — home city: Lagos\n"
            "[tool: web_search — the user's profile]\n"
        )
        marked, count = mark_tool_turns(text)
        assert count == 2
        assert marked.count(TOOL_TURN_LABEL) == 2
        assert "user: where do I live?" in marked
        assert "assistant: let me check" in marked
        # Idempotent: a second pass marks nothing new.
        assert mark_tool_turns(marked) == (marked, 0)

    def test_speaker_turns_are_untouched(self) -> None:
        from particles.extraction.tool_turns import mark_tool_turns

        text = "user: the tool said I live in Lagos\nassistant: noted\n"
        assert mark_tool_turns(text) == (text, 0)

    def test_the_rule_rides_only_a_source_that_has_one(self) -> None:
        from particles.extraction.general import _build_extract_prompt
        from particles.extraction.tool_turns import TOOL_TURN_RULE

        flags: dict[str, bool] = {
            "scope_enabled": False,
            "modality_enabled": False,
            "polarity_enabled": False,
        }
        assert TOOL_TURN_RULE not in _build_extract_prompt(**flags)
        assert TOOL_TURN_RULE in _build_extract_prompt(**flags, tool_turns_present=True)


class TestPersonaFolding:
    """One persona per store for conversational sources — and both paths must fold.

    The rule shipped off for one release because it measurably *cost* update
    supersessions (1 / 3 / 0 against 13 / 10 / 13 unfolded, over three fixed
    live extraction samples). The cause was the asymmetry the last test here
    pins: the write path folded the name, the §6.6 precompute
    looked up the raw one. With both folding the same samples give 12 / 15 /
    14, so the rule is on again.
    """

    @pytest.mark.parametrize("alias", ["user", "The User", "the speaker", "I", "me"])
    def test_conversational_aliases_fold(self, alias: str) -> None:
        from particles.ingest.subject_resolver import _persona_canonical

        assert _persona_canonical(alias, "CONVERSATION") == "the user"

    def test_other_names_and_other_sources_are_untouched(self) -> None:
        from particles.ingest.subject_resolver import _persona_canonical

        assert _persona_canonical("Hiroshi", "CONVERSATION") == "Hiroshi"
        assert _persona_canonical("user", "WEB_PAGE") == "user"

    def test_a_memory_file_folds_whatever_its_source_type(self) -> None:
        """a Claude Code memory file is LOCAL_MARKDOWN on disk."""
        from particles.ingest.subject_resolver import _persona_canonical
        from particles.store.subject_store import persona_alias_guard

        memory_file = ["claude-code", "memory-file"]
        assert _persona_canonical("user", "LOCAL_MARKDOWN", memory_file) == "the user"
        assert _persona_canonical("Mumbai", "LOCAL_MARKDOWN", memory_file) == "Mumbai"
        # The same file untagged is any other Markdown file: its "I" is not the speaker.
        assert _persona_canonical("user", "LOCAL_MARKDOWN", ["claude-code"]) == "user"
        assert persona_alias_guard("I", "LOCAL_MARKDOWN", memory_file) == ()
        assert persona_alias_guard("I", "LOCAL_MARKDOWN") == ("the user",)

    def test_an_empty_tag_list_keys_the_fold_on_source_type_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config
        from particles.ingest.subject_resolver import _persona_canonical

        monkeypatch.setattr(get_config().subjects, "persona_source_tags", [])
        assert _persona_canonical("user", "LOCAL_MARKDOWN", ["memory-file"]) == "user"
        assert _persona_canonical("user", "CONVERSATION") == "the user"

    @pytest.mark.asyncio
    async def test_the_readonly_lookup_folds_the_same_way_the_write_path_does(
        self, db_session: Any
    ) -> None:
        """the fold silently disabled rung 2.5 when the two disagreed.

        ``resolve_subject`` stores a conversational persona under one canonical
        Subject; the §6.6 precompute looked the *raw* surface form up, found
        nothing, and returned no key — so every claim about the speaker skipped
        the subject-keyed search entirely. Measured cost on three fixed live
        extraction samples: 1 / 3 / 0 update supersessions instead of
        13 / 10 / 13.
        """
        from particles.core.schema import Subject
        from particles.ingest.pipeline import _candidate_subject_ids_readonly
        from particles.store.subject_store import insert_subject

        subject = Subject(canonical_name="the user", asserted_by="test")
        await insert_subject(db_session, subject)
        candidate = _cand(["user"])

        # The write path folds "user" onto "the user"; so must this one.
        assert await _candidate_subject_ids_readonly(
            db_session, candidate, {}, source_type="CONVERSATION"
        ) == [subject.id]
        # A source type outside the fold keeps the raw name, and finds nothing.
        assert (
            await _candidate_subject_ids_readonly(db_session, candidate, {}, source_type="WEB_PAGE")
            == []
        )
        # A memory file folds by its tag, as the write path does.
        assert await _candidate_subject_ids_readonly(
            db_session,
            candidate,
            {},
            source_type="LOCAL_MARKDOWN",
            source_tags=["claude-code", "memory-file"],
        ) == [subject.id]
        assert (
            await _candidate_subject_ids_readonly(
                db_session, candidate, {}, source_type="LOCAL_MARKDOWN"
            )
            == []
        )


#: The moved-city fixture the integration tier harvests (tests/fixtures/moved_city).
MOVED_CITY = Path(__file__).parent / "fixtures" / "moved_city"

#: Each session's residence claim as the live extractor emitted it on
#: 2026-09-27: the same speaker named "user" in one file and "the user" in the
#: next. The triple names the speaker, so the speaker is the about-subject.
_SPLIT_SPEAKER: dict[str, tuple[str, list[str], datetime]] = {
    "session-04.md": (DELHI, ["user", "Lajpat Nagar", "Delhi"], datetime(2026, 9, 4, tzinfo=UTC)),
    "session-07.md": (
        MUMBAI,
        ["the user", "Bandra", "Mumbai"],
        datetime(2026, 9, 20, tzinfo=UTC),
    ),
}


class _WikidataLike:
    """A live authority answering "user" the way Wikidata did: "user account".

    Q3604202 at link confidence 0.22 clears the 0.15 abstain floor, so the
    speaker's "user" became a Subject of its own. It answers nothing else, and
    records every name it was asked.
    """

    NAMESPACE = "wikidata"
    PRIORITY = 0
    LIVE = True
    DEFAULT_LINK_CONFIDENCE = 0.95
    APPLICABILITY: list[Any] = []

    def __init__(self) -> None:
        self.asked: list[str] = []

    def uri_for(self, external_id: str) -> str | None:
        return None

    def recognize(self, name: str) -> Any:
        return None

    async def resolve(self, session: Any, name: str, **kwargs: Any) -> Any:
        from particles.core.schema import ExternalRef
        from particles.ingest.authorities import AuthorityResolution

        self.asked.append(name)
        if name.casefold() != "user":
            return None
        return AuthorityResolution(
            external_ref=ExternalRef(
                namespace="wikidata", id="Q3604202", uri=None, confidence=0.22375816106796265
            ),
            canonical_name="user account",
            aliases=["account", "user", "online account", "User"],
        )

    async def canonical_name_for(self, session: Any, external_id: str) -> str | None:
        return None


class _MemoryFiles:
    """An extractor emitting each moved-city session's scripted residence claim."""

    EXTRACTOR_ID = "test-extractor"
    EXTRACTOR_VERSION = "1"

    def __init__(self) -> None:
        self.by_content: dict[bytes, tuple[str, list[str]]] = {}

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> ExtractionResult:
        claim, subjects = self.by_content[content]
        speaker = subjects[0]
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=claim,
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=subjects,
                    structured_claim=StructuredClaim(
                        subject=ClaimTerm(kind=TermKind.TOKEN, value=speaker),
                        predicate=ClaimTerm(kind=TermKind.TOKEN, value="lives in"),
                        object=ClaimTerm(kind=TermKind.TOKEN, value=subjects[-1]),
                        structurizer_id="t",
                        structurizer_version="1",
                    ),
                )
            ]
        )


@pytest.mark.asyncio
class TestMemoryFileSpeaker:
    """one speaker across memory files, so the move supersedes.

    The 2026-09-27 replay on 1.159.0: ``particles audit`` harvested two
    session notes as ``LOCAL_MARKDOWN`` memory files. The persona fold keyed on
    source type alone, so it never ran; "user" went to Wikidata and came back
    as "user account", "the user" became a bare Subject, and "the user lives in
    Delhi" stayed ACTIVE beside "the user is living in Bandra" because the two
    claims shared no subject for rung 2.5 to pair them on.
    """

    @pytest.fixture(autouse=True)
    def _floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # As in TestSlotRule: the bag-of-words encoder scores this pair below
        # the production floor. The floor finds candidates and decides nothing.
        from particles.config import get_config

        monkeypatch.setattr(get_config().reconciliation.update_supersession, "subject_floor", 0.3)

    async def _harvest(
        self, session: Any, ex: _MemoryFiles, wikidata: _WikidataLike, name: str
    ) -> list[Particle]:
        from particles.corpus.deposit import deposit_text_versioned
        from particles.ingest.pipeline import extract_snapshot

        claim, subjects, dated = _SPLIT_SPEAKER[name]
        text = (MOVED_CITY / name).read_text()
        ex.by_content[text.encode()] = (claim, subjects)
        entry_id, snapshot_id, _ = await deposit_text_versioned(
            session,
            text=text,
            uri_r=f"file:///notes/{name}",
            source_type=SourceType.LOCAL_MARKDOWN,
            mutability=Mutability.MUTABLE,
            content_published_at=dated,
            # What `particles audit` and the SessionEnd harvest stamp.
            tags=["claude-code", "memory-file"],
        )
        await session.commit()
        with patch("particles.ingest.subject_resolver.get_authorities", return_value=[wikidata]):
            written = await extract_snapshot(session, entry_id, snapshot_id, extractor=ex)
        await session.commit()
        return written

    async def _speaker(self, session: Any, particle: Particle) -> str:
        from particles.store.subject_store import get_subject

        names = [
            s.canonical_name
            for s in [await get_subject(session, sid) for sid in particle.subject_ids]
            if s is not None
        ]
        return next(n for n in names if "user" in n.casefold())

    async def test_the_move_supersedes_across_two_memory_files(
        self, db_session: Any, bow: AsyncMock
    ) -> None:
        ex, wikidata = _MemoryFiles(), _WikidataLike()
        await self._harvest(db_session, ex, wikidata, "session-04.md")
        [mumbai] = await self._harvest(db_session, ex, wikidata, "session-07.md")

        [delhi] = await _by_content(db_session, DELHI)
        assert await self._speaker(db_session, delhi) == "the user"
        assert await self._speaker(db_session, mumbai) == "the user"
        assert delhi.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        assert mumbai.status is Status.ACTIVE and mumbai.supersedes == delhi.id
        bow.assert_awaited_once_with(DELHI, MUMBAI)

    async def test_the_speaker_never_reaches_a_live_authority(
        self, db_session: Any, bow: AsyncMock
    ) -> None:
        ex, wikidata = _MemoryFiles(), _WikidataLike()
        await self._harvest(db_session, ex, wikidata, "session-04.md")
        await self._harvest(db_session, ex, wikidata, "session-07.md")
        persona_forms = {"user", "the user"}
        assert not persona_forms & {n.casefold() for n in wikidata.asked}
        # The places in the same claims are still looked up.
        assert {"Delhi", "Mumbai"} <= set(wikidata.asked)

    async def test_without_the_tag_the_speaker_splits_as_it_did(
        self, db_session: Any, bow: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: this fixture reproduces the 1.159.0 split when the fold is off."""
        from particles.config import get_config

        monkeypatch.setattr(get_config().subjects, "persona_source_tags", [])
        ex, wikidata = _MemoryFiles(), _WikidataLike()
        await self._harvest(db_session, ex, wikidata, "session-04.md")
        [mumbai] = await self._harvest(db_session, ex, wikidata, "session-07.md")

        [delhi] = await _by_content(db_session, DELHI)
        assert await self._speaker(db_session, delhi) == "user account"
        assert await self._speaker(db_session, mumbai) == "the user"
        assert delhi.status is Status.ACTIVE and delhi.status_reason is None
        assert mumbai.supersedes is None
        bow.assert_not_awaited()


@pytest.mark.asyncio
class TestSweep:
    @pytest.fixture(autouse=True)
    def _probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Confirm every replacement probe without an API key.

        The sweep's probe leg is the shared L-SEM-01 call; a unit run must not
        need a key for it (tests/AGENTS.md § Integration tests). What these
        tests pin is the *rule* around it — candidacy, ordering, idempotence —
        so the probe answers YES and the phase-1 filter still does the work.
        """
        from particles.operations import reconcile as reconcile_mod

        monkeypatch.setattr(
            reconcile_mod, "_has_contradiction_signal", AsyncMock(return_value=True)
        )
        monkeypatch.setattr(reconcile_mod, "_has_update_signal", AsyncMock(side_effect=_updates))

    async def test_sweep_clears_a_backlog_and_is_idempotent(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config
        from particles.operations.reconcile import reconcile_updates

        # A store as it was before: updates written with the rung off.
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", False)
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        await _say(db_session, ex, DENVER, day=9)
        assert [p.status for p in await _by_content(db_session, BOSTON)] == [Status.ACTIVE]

        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", True)
        first = await reconcile_updates(db_session)
        assert first["demoted"] == 1
        [old] = await _by_content(db_session, BOSTON)
        assert old.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        [new] = await _by_content(db_session, DENVER)
        assert new.status is Status.ACTIVE and new.supersedes == old.id

        second = await reconcile_updates(db_session)
        assert second["demoted"] == 0

    async def test_sweep_respects_the_switch_and_dry_run(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config
        from particles.operations.reconcile import reconcile_updates

        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", False)
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        await _say(db_session, ex, DENVER, day=9)
        off = await reconcile_updates(db_session)
        assert off["enabled"] is False and off["demoted"] == 0

        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", True)
        dry = await reconcile_updates(db_session, dry_run=True)
        assert dry["demoted"] == 1
        assert [p.status for p in await _by_content(db_session, BOSTON)] == [Status.ACTIVE]

    async def test_a_restated_value_is_dated_by_its_latest_observation(
        self, db_session: Any, bow: None
    ) -> None:
        """A value restated after a change is current again.

        The regression this pins: the sweep dated a claim by its *first*
        provenance ref, so a reverted value — folded into the original
        particle — looked older than the value it had replaced and
        was retired, taking the current value off the ACTIVE surface.
        """
        from particles.config import get_config
        from particles.ingest.update_supersession import latest_source_date
        from particles.operations.reconcile import reconcile_updates

        # A store as before, so every value stays ACTIVE and the restatement
        # folds into the original particle instead of minting one.
        get_config().reconciliation.update_supersession.enabled = False
        ex = _OneClaim()
        await _say(db_session, ex, BOSTON, day=1)
        await _say(db_session, ex, DENVER, day=5)
        await _say(db_session, ex, BOSTON, day=20)
        get_config().reconciliation.update_supersession.enabled = True
        [boston] = await _by_content(db_session, BOSTON)
        assert len([r for r in boston.provenance if r.type.value == "SOURCE"]) == 2
        latest = await latest_source_date(db_session, boston)
        assert latest is not None and latest.day == (T0 + timedelta(days=20)).day

        await reconcile_updates(db_session)
        [boston] = await _by_content(db_session, BOSTON)
        assert boston.status is Status.ACTIVE, "the restated value must survive the sweep"

    async def test_a_generic_and_an_instance_claim_are_never_paired(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """an exception does not falsify "most", so no probe, no retirement.

        Both probes would say "a newer value for one slot that changes", which
        retires the earlier claim for any pair that reached them.
        """
        from particles.config import get_config
        from particles.operations import reconcile as reconcile_mod

        generic = "Users generally keep the editor in dark mode."
        instance = "The user keeps the editor in light mode."
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", False)
        ex = _OneClaim()
        await _say(db_session, ex, generic, day=1)
        await _say(db_session, ex, instance, day=9)
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", True)
        contradiction = AsyncMock(return_value=True)
        monkeypatch.setattr(reconcile_mod, "_has_contradiction_signal", contradiction)
        monkeypatch.setattr(
            reconcile_mod, "_has_update_signal", AsyncMock(return_value=SlotVerdict.CHANGES)
        )

        assert await reconcile_mod.count_update_candidates(db_session, None) == 0
        summary = await reconcile_mod.reconcile_updates(db_session)
        assert summary["demoted"] == 0
        contradiction.assert_not_awaited()
        for claim in (generic, instance):
            [p] = await _by_content(db_session, claim)
            assert p.status is Status.ACTIVE


@pytest.mark.asyncio
class TestAgentAssertionPath:
    """an agent may revise its own belief, and nothing else."""

    async def _assert(
        self, session: Any, content: str, *, at: datetime, who: str = "mcp:claude-code"
    ) -> Particle:
        from particles.ingest.pipeline import reconcile_and_insert

        p = _p(content, calibration=CalibrationSource.AGENT_ASSERTED, asserted_by=who).model_copy(
            update={"asserted_at": at}
        )
        out = await reconcile_and_insert(
            session,
            p,
            embedding=[float(x) for x in _BagOfWords().encode([content])[0]],
            single_trust_order=False,
        )
        await session.commit()
        assert out is not None
        return out

    async def _store(self, session: Any, particle: Particle) -> Particle:
        from particles.store.particle_store import insert_particle

        emb = [float(x) for x in _BagOfWords().encode([particle.content])[0]]
        await insert_particle(session, particle, emb)
        await session.commit()
        return particle

    async def test_an_agent_supersedes_its_own_earlier_assertion(
        self, db_session: Any, bow: None
    ) -> None:
        old = await self._assert(db_session, BOSTON, at=T0)
        new = await self._assert(db_session, DENVER, at=T0 + timedelta(hours=1))
        [stored_old] = await _by_content(db_session, BOSTON)
        assert stored_old.id == old.id
        assert stored_old.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        assert new.status is Status.ACTIVE and new.supersedes == old.id

    async def test_an_agent_never_supersedes_an_extracted_claim(
        self, db_session: Any, bow: None
    ) -> None:
        extracted = await self._store(db_session, _p(BOSTON))
        written = await self._assert(db_session, DENVER, at=T0 + timedelta(hours=1))
        [stored] = await _by_content(db_session, BOSTON)
        assert stored.id == extracted.id and stored.status is Status.ACTIVE
        # fail-closed: the agent's claim is quarantined behind a review record
        assert written.status is Status.INCONSISTENCY

    async def test_an_agent_never_supersedes_an_operator_belief(
        self, db_session: Any, bow: None
    ) -> None:
        operator = await self._store(
            db_session, _p(BOSTON, calibration=CalibrationSource.HUMAN_REVIEW)
        )
        await self._assert(db_session, DENVER, at=T0 + timedelta(hours=1))
        [stored] = await _by_content(db_session, BOSTON)
        assert stored.id == operator.id and stored.status is Status.ACTIVE

    async def test_an_agent_never_supersedes_another_agent(
        self, db_session: Any, bow: None
    ) -> None:
        await self._assert(db_session, BOSTON, at=T0, who="mcp:other-client")
        await self._assert(db_session, DENVER, at=T0 + timedelta(hours=1))
        [stored] = await _by_content(db_session, BOSTON)
        assert stored.status is Status.ACTIVE
