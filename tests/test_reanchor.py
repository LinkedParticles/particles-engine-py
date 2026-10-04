# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The re-anchor pass: restate claims that relied on a superseded state.

Built on the Delhi scenario: session 4 says the user lives in Lajpat Nagar and
that Sandeep's Curry House is a ten-minute walk from their flat; session 7 says
they moved to Mumbai. The residence claim is retired as an update, and the
walking-distance claim must stop reading as a fact about the user's present.

The pure decisions (:mod:`particles.core.reanchor`) are tested directly. The
pass's LLM legs are scripted at their module seams (``_probe``, ``_read``,
the paraphrase judge and the contradiction probe), per tests/AGENTS.md
§ Mocking strategy; the encoder is a bag of words so similarity is real and
deterministic.
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

from particles.core.conflict_resolution import SlotVerdict
from particles.core.reanchor import (
    Cursor,
    DependencyVerdict,
    Outcome,
    ProbeVerdict,
    Reading,
    build_restatement,
    decide_after_reading,
    decide_write,
    is_before_or_at,
    parse_probe_reply,
    parse_reading_reply,
)
from particles.core.schema import (
    Confidence,
    JudgeVerdictKind,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    SourceType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.extraction.general import CandidateParticle, ExtractionResult

T0 = datetime(2026, 9, 1, tzinfo=UTC)

RESIDENCE = "The user lives in Delhi, in Lajpat Nagar."
WALK = "Sandeep's Curry House is a ten-minute walk from the user's flat."
TIKKA = "The user considers Sandeep's Curry House's paneer tikka the best they have had."
VEG = "The user is vegetarian."
MOVED = "The user moved to Mumbai last month."
BANDRA = "The user is currently living in Bandra."
RESTATED = (
    "Sandeep's Curry House is a ten-minute walk from the flat in Lajpat Nagar, Delhi, "
    "where the user lived as of 2026-09-04."
)

_TOKEN = re.compile(r"[a-z0-9']+")


class _BagOfWords:
    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                out[i, int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16) % 256] += 1.0
            out[i] /= max(float(np.linalg.norm(out[i])), 1e-9)
        return out


class _Sessions:
    """An extractor emitting scripted claims, with subject names, per deposited text."""

    EXTRACTOR_ID = "test-extractor"
    EXTRACTOR_VERSION = "1"

    def __init__(self) -> None:
        self.claims: dict[bytes, list[tuple[str, list[str]]]] = {}

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> ExtractionResult:
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=claim,
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=subjects,
                )
                for claim, subjects in self.claims[content]
            ]
        )


_USER = "the user"
_PLACE = "Sandeep's Curry House"

SESSION_4 = [
    (RESIDENCE, [_USER]),
    (WALK, [_PLACE, _USER]),
    (TIKKA, [_PLACE, _USER]),
    (VEG, [_USER]),
]
SESSION_7 = [(MOVED, [_USER]), (BANDRA, [_USER])]


@pytest.fixture
def bow() -> Generator[None, None, None]:
    """A bag-of-words encoder, and write-time probes that find nothing."""
    from particles import embeddings as ep

    original = ep._embedding_model
    ep.set_embedding_model(_BagOfWords())  # type: ignore[arg-type]
    try:
        with (
            patch(
                "particles.ingest.pipeline._has_contradiction_signal",
                AsyncMock(return_value=False),
            ),
            patch(
                "particles.ingest.pipeline._has_update_signal",
                AsyncMock(return_value=SlotVerdict.DIFFERENT),
            ),
        ):
            yield
    finally:
        ep.set_embedding_model(original)


async def _session(
    session: Any, ex: _Sessions, claims: list[tuple[str, list[str]]], day: int
) -> None:
    from particles.corpus.deposit import deposit_text_versioned
    from particles.ingest.pipeline import extract_snapshot

    text = f"Session on day {day}.\n\n" + "\n\n".join(f"User: {c}" for c, _ in claims)
    ex.claims[text.encode()] = claims
    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text=text,
        uri_r=f"claude-code://session/{uuid.uuid4()}",
        source_type=SourceType.CONVERSATION,
        mutability=Mutability.STABLE,
        content_published_at=T0 + timedelta(days=day),
    )
    await session.commit()
    await extract_snapshot(session, entry_id, snapshot_id, extractor=ex)
    await session.commit()


async def _one(session: Any, content: str, status: Status | None = None) -> Particle:
    from particles.store.particle_store import get_particles_by_status

    found: list[Particle] = []
    for s in [status] if status else list(Status):
        found += [p for p in await get_particles_by_status(session, s) if p.content == content]
    assert len(found) == 1, (content, found)
    return found[0]


async def _delhi(session: Any, ex: _Sessions | None = None) -> _Sessions:
    """Sessions 4 and 7 extracted, and the residence retired as an update."""
    from particles.store.particle_store import update_particle_status

    ex = ex or _Sessions()
    await _session(session, ex, SESSION_4, day=4)
    await _session(session, ex, SESSION_7, day=20)
    residence = await _one(session, RESIDENCE)
    await update_particle_status(
        session, residence.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
    )
    await session.commit()
    return ex


def _script(
    *,
    depends: dict[str, str | None] | None = None,
    reading: Reading | None = Reading(depended=True, faithful=True, reason="ok"),
    paraphrase: JudgeVerdictKind = JudgeVerdictKind.DISTINCT,
    contradiction: bool | None = False,
) -> Any:
    """Patch the pass's four LLM seams; yields the probe mock."""
    depends = {WALK: RESTATED} if depends is None else depends

    async def probe(
        retired: Particle, passage: str, date: str | None, cands: list[Particle]
    ) -> Any:
        return [
            ProbeVerdict(c.id, DependencyVerdict.DEPENDS, depends[c.content], "moved")
            if c.content in depends
            else ProbeVerdict(c.id, DependencyVerdict.HOLDS, None, "holds")
            for c in cands
        ]

    probe_mock = AsyncMock(side_effect=probe)
    patches = [
        patch("particles.operations.reanchor._probe", probe_mock),
        patch("particles.operations.reanchor._read", AsyncMock(return_value=reading)),
        patch(
            "particles.operations.abstraction._paraphrase_verdict",
            AsyncMock(return_value=paraphrase),
        ),
        patch(
            "particles.operations.reconcile._has_contradiction_signal",
            AsyncMock(return_value=contradiction),
        ),
    ]
    return patches, probe_mock


class _Scripted:
    def __init__(self, **kwargs: Any) -> None:
        self.patches, self.probe = _script(**kwargs)

    def __enter__(self) -> AsyncMock:
        for p in self.patches:
            p.start()
        return self.probe

    def __exit__(self, *exc: object) -> None:
        for p in reversed(self.patches):
            p.stop()


# ---------------------------------------------------------------------------
# Pure decisions
# ---------------------------------------------------------------------------


class TestPure:
    def test_probe_reply_maps_labels_and_drops_unknowns(self) -> None:
        reply = (
            '{"claims": [{"claim": "c1", "verdict": "DEPENDS", "restatement": "R.", '
            '"reason": "x"}, {"claim": "c2", "verdict": "holds", "reason": "y"}, '
            '{"claim": "c9", "verdict": "HOLDS", "reason": "z"}, '
            '{"claim": "c2", "verdict": "DEPENDS", "reason": "dup"}]}'
        )
        verdicts = parse_probe_reply(reply, {"c1": "a", "c2": "b"})
        assert verdicts == [
            ProbeVerdict("a", DependencyVerdict.DEPENDS, "R.", "x"),
            ProbeVerdict("b", DependencyVerdict.HOLDS, None, "y"),
        ]
        assert parse_probe_reply("not json", {"c1": "a"}) is None

    def test_reading_reply_needs_both_booleans(self) -> None:
        assert parse_reading_reply('{"depended": true, "faithful": false, "reason": "r"}') == (
            Reading(depended=True, faithful=False, reason="r")
        )
        assert parse_reading_reply('{"depended": true}') is None

    def test_outcomes(self) -> None:
        assert decide_after_reading(None) is Outcome.UNRESTATED
        assert (
            decide_after_reading(Reading(depended=False, faithful=True, reason=""))
            is Outcome.DEPENDENCY_REJECTED
        )
        assert (
            decide_after_reading(Reading(depended=True, faithful=False, reason=""))
            is Outcome.UNRESTATED
        )
        assert decide_after_reading(Reading(depended=True, faithful=True, reason="")) is None
        assert decide_write(duplicate_of="d", conflict=True) is Outcome.MATCHED_EXISTING
        assert decide_write(duplicate_of=None, conflict=True) is Outcome.UNRESTATED
        assert decide_write(duplicate_of=None, conflict=False) is Outcome.RESTATED

    def test_not_observed_after(self) -> None:
        assert is_before_or_at(T0, T0)
        assert not is_before_or_at(T0 + timedelta(days=1), T0)
        assert not is_before_or_at(None, T0)

    def test_cursor_round_trip(self) -> None:
        cursor = Cursor(retired_at=T0, particle_id="p")
        assert Cursor.from_payload(cursor.payload()) == cursor
        assert Cursor.from_payload({"retired_at": "nope", "particle_id": "p"}) is None

    def test_restatement_record(self) -> None:
        def claim(content: str, value: float, snap: str) -> Particle:
            return Particle(
                content=content,
                confidence=Confidence(
                    value=value, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id=snap
                    )
                ],
                asserted_by="general-extractor",
                subject_ids=["s-place", "s-user"] if "Sandeep" in content else ["s-user"],
            )

        original = claim(WALK, 0.9, "s4")
        retired = claim(RESIDENCE, 0.8, "s4")
        now = T0 + timedelta(days=30)
        written = build_restatement(original, retired, RESTATED, provider_model="p:m", now=now)
        assert written.id != original.id
        assert written.supersedes == original.id
        assert written.confidence.value == 0.8
        assert written.confidence.calibration_source is CalibrationSource.EXTRACTOR_DIRECT
        assert [r.type for r in written.provenance] == [
            ProvenanceRefType.SOURCE,
            ProvenanceRefType.PARTICLE,
        ]
        assert written.provenance[1].corpus_entry_id == retired.id
        assert written.asserted_by == original.asserted_by
        assert written.asserted_at == now
        assert written.subject_ids == ["s-place", "s-user"]
        assert written.contributors is not None
        assert written.contributors[-1].id == "consolidation:reanchor"


# ---------------------------------------------------------------------------
# The pass, on the Delhi scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDelhi:
    async def test_the_walking_claim_is_restated(self, db_session: Any, bow: None) -> None:
        from particles.operations.reanchor import run_reanchor
        from particles.store.particle_store import get_particles_by_status

        await _delhi(db_session)
        walk = await _one(db_session, WALK)
        residence = await _one(db_session, RESIDENCE)
        with _Scripted() as probe:
            report = await run_reanchor(db_session, cursor=None)
            await db_session.commit()

        assert probe.await_count == 1
        [cands] = [call.args[3] for call in probe.await_args_list]
        assert {c.content for c in cands} == {WALK, TIKKA, VEG}
        assert report.triggers == 1 and len(report.restated) == 1 and report.holds == 2

        old = await _one(db_session, WALK)
        assert (old.status, old.status_reason) == (
            Status.SUPERSEDED,
            StatusReason.SUPERSEDED_BY_REANCHOR,
        )
        new = await _one(db_session, RESTATED)
        assert new.status is Status.ACTIVE
        assert new.supersedes == walk.id
        assert new.asserted_by == walk.asserted_by
        assert [
            r.corpus_entry_id for r in new.provenance if r.type is ProvenanceRefType.PARTICLE
        ] == [residence.id]
        assert {r.snapshot_id for r in new.provenance if r.type is ProvenanceRefType.SOURCE} == {
            r.snapshot_id for r in walk.provenance if r.type is ProvenanceRefType.SOURCE
        }
        assert new.confidence.value == min(walk.confidence.value, residence.confidence.value)

        active = {p.content for p in await get_particles_by_status(db_session, Status.ACTIVE)}
        assert WALK not in active and RESTATED in active
        assert {TIKKA, VEG, MOVED, BANDRA} <= active

    async def test_the_digest_carries_the_restatement(self, db_session: Any, bow: None) -> None:
        from particles.operations.digest import build_digest
        from particles.operations.reanchor import run_reanchor

        await _delhi(db_session)
        with _Scripted():
            await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        text = await build_digest("default")
        assert RESTATED in text
        assert WALK not in text

    async def test_a_rerun_examines_nothing_new(self, db_session: Any, bow: None) -> None:
        from particles.operations.reanchor import run_reanchor

        await _delhi(db_session)
        with _Scripted() as probe:
            first = await run_reanchor(db_session, cursor=None)
            await db_session.commit()
            cursor = Cursor.from_payload(first.cursor)
            second = await run_reanchor(db_session, cursor=cursor)
        assert cursor is not None and second.triggers == 0 and probe.await_count == 1

    async def test_the_restatement_is_not_re_examined(self, db_session: Any, bow: None) -> None:
        """A second update from the same passage finds the restatement ineligible."""
        from particles.operations.reanchor import run_reanchor
        from particles.store.particle_store import update_particle_status

        await _delhi(db_session)
        with _Scripted():
            first = await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        veg = await _one(db_session, VEG)
        await update_particle_status(
            db_session, veg.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
        )
        await db_session.commit()
        with _Scripted() as probe:
            await run_reanchor(db_session, cursor=Cursor.from_payload(first.cursor))
        [cands] = [call.args[3] for call in probe.await_args_list]
        assert RESTATED not in {c.content for c in cands}


@pytest.mark.asyncio
class TestCandidacy:
    async def test_claims_outside_the_passage_or_subject_are_not_sent(
        self, db_session: Any, bow: None
    ) -> None:
        from particles.operations.reanchor import run_reanchor

        ex = _Sessions()
        claims = [*SESSION_4, ("Delhi traffic is heavy in the evenings.", ["Delhi"])]
        await _session(db_session, ex, claims, day=4)
        await _session(db_session, ex, SESSION_7, day=20)
        from particles.store.particle_store import update_particle_status

        residence = await _one(db_session, RESIDENCE)
        await update_particle_status(
            db_session, residence.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
        )
        await db_session.commit()
        with _Scripted() as probe:
            await run_reanchor(db_session, cursor=None)
        [cands] = [call.args[3] for call in probe.await_args_list]
        sent = {c.content for c in cands}
        assert "Delhi traffic is heavy in the evenings." not in sent  # no shared subject
        assert MOVED not in sent and BANDRA not in sent  # another snapshot

    async def test_a_claim_restated_after_the_update_is_left_alone(
        self, db_session: Any, bow: None
    ) -> None:
        from particles.operations.reanchor import run_reanchor

        ex = await _delhi(db_session)
        # Session 9 restates the walking claim, which folds into the same
        # particle, so its latest observation is now later than R's.
        await _session(db_session, ex, [(WALK, [_PLACE, _USER])], day=25)
        with _Scripted() as probe:
            await run_reanchor(db_session, cursor=None)
        [cands] = [call.args[3] for call in probe.await_args_list]
        assert WALK not in {c.content for c in cands}

    async def test_an_agent_assertion_is_not_a_candidate(self, db_session: Any, bow: None) -> None:
        from particles.operations.reanchor import run_reanchor
        from particles.store.particle_store import ParticleRow

        await _delhi(db_session)
        walk = await _one(db_session, WALK)
        row = await db_session.get(ParticleRow, walk.id)
        row.confidence_calibration_source = CalibrationSource.AGENT_ASSERTED.value
        await db_session.commit()
        with _Scripted() as probe:
            await run_reanchor(db_session, cursor=None)
        [cands] = [call.args[3] for call in probe.await_args_list]
        assert WALK not in {c.content for c in cands}

    async def test_the_ninth_candidate_is_not_sent(self, db_session: Any, bow: None) -> None:
        from particles.config import get_config
        from particles.operations.reanchor import run_reanchor

        get_config().consolidation.reanchor.max_candidates_per_retirement = 2
        await _delhi(db_session)
        with _Scripted() as probe:
            report = await run_reanchor(db_session, cursor=None)
        [cands] = [call.args[3] for call in probe.await_args_list]
        assert len(cands) == 2 and report.candidates == 2


@pytest.mark.asyncio
class TestTriggers:
    async def test_only_update_retirements_trigger(self, db_session: Any, bow: None) -> None:
        from particles.operations.reanchor import run_reanchor
        from particles.store.particle_store import update_particle_status

        ex = _Sessions()
        await _session(db_session, ex, SESSION_4, day=4)
        residence = await _one(db_session, RESIDENCE)
        await update_particle_status(
            db_session, residence.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()
        with _Scripted() as probe:
            report = await run_reanchor(db_session, cursor=None)
        assert report.triggers == 0 and probe.await_count == 0


@pytest.mark.asyncio
class TestReadingAndChecks:
    async def test_a_rejected_dependency_writes_nothing(self, db_session: Any, bow: None) -> None:
        from particles.operations.reanchor import pending_unrestated_events, run_reanchor

        await _delhi(db_session)
        with _Scripted(reading=Reading(depended=False, faithful=True, reason="lasting")):
            report = await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        assert report.dependency_rejected == 1 and not report.restated
        assert (await _one(db_session, WALK)).status is Status.ACTIVE
        assert await pending_unrestated_events(db_session) == []

    @pytest.mark.parametrize(
        "reading",
        [Reading(depended=True, faithful=False, reason="added detail"), None],
    )
    async def test_an_unfaithful_or_failed_reading_raises_a_card(
        self, db_session: Any, bow: None, reading: Reading | None
    ) -> None:
        from particles.operations.curation.cards import CardKind
        from particles.operations.curation.collect import collect_cards
        from particles.operations.reanchor import pending_unrestated_events, run_reanchor

        await _delhi(db_session)
        with _Scripted(reading=reading):
            report = await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        walk = await _one(db_session, WALK)
        assert walk.status is Status.ACTIVE and len(report.unrestated) == 1
        [event] = await pending_unrestated_events(db_session)
        assert event.payload["particle_id"] == walk.id
        cards = [
            c
            for c in await collect_cards(db_session, semantic=False)
            if c.kind is CardKind.STALE_BASIS
        ]
        assert [c.particle_ids for c in cards] == [[walk.id]]
        assert cards[0].key == f"stale_basis:{walk.id}"

    async def test_a_conflicting_restatement_is_not_written(
        self, db_session: Any, bow: None
    ) -> None:
        from particles.config import get_config
        from particles.operations.reanchor import run_reanchor

        # Any claim sharing a subject is a contradiction candidate here.
        get_config().reconciliation.update_supersession.subject_floor = 0.0
        await _delhi(db_session)
        with _Scripted(contradiction=True):
            report = await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        assert not report.restated and len(report.unrestated) == 1
        assert "contradicts" in report.unrestated[0]["why"]
        assert (await _one(db_session, WALK)).status is Status.ACTIVE

    async def test_a_claim_of_the_other_kind_is_never_probed(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a generic and an instance claim are not an adjudicable pair.

        The detector is patched to read the restatement alone as generic, so
        every standing claim beside it is of the other kind and none is probed:
        the probe that would block the restatement is never asked.
        """
        from particles.config import get_config
        from particles.operations import reanchor
        from particles.operations.reanchor import run_reanchor

        get_config().reconciliation.update_supersession.subject_floor = 0.0
        monkeypatch.setattr(reanchor, "is_generic_claim", lambda text: text == RESTATED)
        await _delhi(db_session)
        with _Scripted(contradiction=True):
            report = await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        assert report.conflict_probes == 0
        assert len(report.restated) == 1 and not report.unrestated

    async def test_a_failed_contradiction_check_is_a_conflict(
        self, db_session: Any, bow: None
    ) -> None:
        from particles.config import get_config
        from particles.operations.reanchor import run_reanchor

        get_config().reconciliation.update_supersession.subject_floor = 0.0
        await _delhi(db_session)
        with _Scripted(contradiction=None):
            report = await run_reanchor(db_session, cursor=None)
        assert not report.restated and len(report.unrestated) == 1

    async def test_an_existing_statement_is_matched(self, db_session: Any, bow: None) -> None:
        from particles.operations.reanchor import run_reanchor

        ex = _Sessions()
        claims = [*SESSION_4, (RESTATED, [_PLACE, _USER])]
        await _session(db_session, ex, claims, day=4)
        await _session(db_session, ex, SESSION_7, day=20)
        from particles.store.particle_store import update_particle_status

        residence = await _one(db_session, RESIDENCE)
        await update_particle_status(
            db_session, residence.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
        )
        await db_session.commit()
        existing = await _one(db_session, RESTATED)
        with _Scripted():
            report = await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        assert [m["existing"] for m in report.matched_existing] == [existing.id]
        walk = await _one(db_session, WALK)
        assert walk.status_reason is StatusReason.SUPERSEDED_BY_REANCHOR
        assert (await _one(db_session, RESTATED)).id == existing.id  # nothing new written


@pytest.mark.asyncio
class TestInvalidationAndCaps:
    async def test_retracting_the_retired_claim_flags_the_restatement(
        self, db_session: Any, bow: None
    ) -> None:
        from particles.operations.lint.staleness import _check_retraction_propagation
        from particles.operations.reanchor import run_reanchor
        from particles.store.particle_store import update_particle_status

        await _delhi(db_session)
        with _Scripted():
            await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        new = await _one(db_session, RESTATED)
        assert new.id not in {
            f.particle_id for f in await _check_retraction_propagation(db_session, fix=False)
        }
        residence = await _one(db_session, RESIDENCE)
        await update_particle_status(
            db_session, residence.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()
        flagged = {
            f.particle_id for f in await _check_retraction_propagation(db_session, fix=False)
        }
        assert new.id in flagged

    async def test_the_cap_leaves_a_retirement_waiting(self, db_session: Any, bow: None) -> None:
        from particles.config import get_config
        from particles.operations.reanchor import run_reanchor
        from particles.store.particle_store import update_particle_status

        get_config().consolidation.reanchor.max_retirements_per_run = 1
        await _delhi(db_session)
        bandra = await _one(db_session, BANDRA)
        await update_particle_status(
            db_session, bandra.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
        )
        await db_session.commit()
        with _Scripted(depends={}):
            first = await run_reanchor(db_session, cursor=None)
            second = await run_reanchor(db_session, cursor=Cursor.from_payload(first.cursor))
        assert (first.triggers, first.waiting) == (1, 1)
        assert (second.triggers, second.waiting) == (1, 0)

    async def test_a_dry_run_makes_no_call(self, db_session: Any, bow: None) -> None:
        from particles.operations.reanchor import run_reanchor

        await _delhi(db_session)
        with _Scripted() as probe:
            report = await run_reanchor(db_session, cursor=None, dry_run=True)
        assert probe.await_count == 0 and report.probes == 1 and report.candidates == 3
        assert report.cursor is None
        assert (await _one(db_session, WALK)).status is Status.ACTIVE

    async def test_the_switch_skips_the_pass(self, db_session: Any, bow: None) -> None:
        from particles.config import get_config
        from particles.operations.reanchor import run_reanchor

        get_config().consolidation.reanchor.enabled = False
        await _delhi(db_session)
        report = await run_reanchor(db_session, cursor=None)
        assert report.skipped_reason is not None and report.triggers == 0

    async def test_the_cursor_is_read_from_the_last_run(self, db_session: Any) -> None:
        from particles.operations.reanchor import prior_cursor
        from particles.store.event_store import OperatorEventType, record_event

        cursor = Cursor(retired_at=T0, particle_id="p1")
        await record_event(
            db_session,
            actor="memory-consolidate",
            event_type=OperatorEventType.CONSOLIDATION_RUN,
            payload={"census": {"reanchor": {"cursor": cursor.payload()}}},
        )
        await db_session.commit()
        assert await prior_cursor(db_session, "memory-consolidate") == cursor
        assert await prior_cursor(db_session, "audit") is None


@pytest.mark.asyncio
class TestReindex:
    async def test_reindex_keeps_the_restatement(self, db_session: Any, bow: None) -> None:
        from particles.operations.reanchor import run_reanchor
        from particles.operations.reindex import _reanchored_ids
        from particles.store.particle_store import get_particles_by_status

        await _delhi(db_session)
        with _Scripted():
            await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        active = await get_particles_by_status(db_session, Status.ACTIVE)
        protected = await _reanchored_ids(db_session, active)
        assert {p.content for p in active if p.id in protected} == {RESTATED}


@pytest.mark.asyncio
class TestPassage:
    async def test_the_passage_carries_the_source_opening_line(
        self, db_session: Any, bow: None
    ) -> None:
        """A note's dated header is where the restatement's date comes from."""
        from particles.operations.reanchor import _passage_for

        await _delhi(db_session)
        residence = await _one(db_session, RESIDENCE)
        walk = await _one(db_session, WALK)
        located = await _passage_for(db_session, residence, [walk])
        assert located is not None
        passage, date = located
        assert passage.startswith("Session on day 4.")
        assert WALK in passage and RESIDENCE in passage
        assert date == "2026-09-05"


@pytest.mark.asyncio
class TestInterleaving:
    async def test_a_settled_verdict_is_committed_before_the_next_reading(
        self, file_db_session: Any, bow: None
    ) -> None:
        """A writer in another process can write while the pass waits on the LLM.

        The restatement used to stay uncommitted until the caller's commit, so
        SQLite's write lock was held through every later reading and probe.
        """
        from particles.operations.reanchor import run_reanchor
        from tests._write_probe import store_accepts_a_writer

        session = file_db_session
        await _delhi(session)
        tikka_restated = (
            "The user considered Sandeep's Curry House's paneer tikka the best they had "
            "had while living in Lajpat Nagar, Delhi, as of 2026-09-04."
        )
        writable_during_readings: list[bool] = []

        async def reading(*_: Any) -> Reading:
            writable_during_readings.append(store_accepts_a_writer())
            return Reading(depended=True, faithful=True, reason="ok")

        with (
            _Scripted(depends={WALK: RESTATED, TIKKA: tikka_restated}),
            patch("particles.operations.reanchor._read", AsyncMock(side_effect=reading)),
        ):
            report = await run_reanchor(session, cursor=None)

        assert len(report.restated) == 2
        assert writable_during_readings == [True, True]
        assert store_accepts_a_writer()
