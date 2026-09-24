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
  switch really switches it off.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from particles.core.conflict_resolution import ConflictVerdict, resolve_conflict
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


async def _contradicts(a: str, b: str) -> bool:
    """Scripted probe: two claims contradict iff they name different cities."""
    ca = {c for c in _CITIES if c in a}
    cb = {c for c in _CITIES if c in b}
    return bool(ca and cb and ca != cb)


@pytest.fixture
def bow() -> Generator[None, None, None]:
    from particles import embeddings as ep

    original = ep._embedding_model
    ep.set_embedding_model(_BagOfWords())  # type: ignore[arg-type]
    try:
        with patch(
            "particles.ingest.pipeline._has_contradiction_signal",
            AsyncMock(side_effect=_contradicts),
        ):
            yield
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
