# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The re-anchor pass: restate claims that relied on a superseded state.

Built on the Delhi scenario: session 4 says the user lives in Lajpat Nagar and
that Sandeep's Curry House is a ten-minute walk from their flat; session 7 says
they moved to Mumbai. The residence claim is retired as an update, and the
walking-distance claim must stop reading as a fact about the user's present.
Beside it, the Ledgerly case (a project moved from PostgreSQL to SQLite) pins
the write order that keeps the newer source's claim out of quarantine, and
that the pass's candidacy reaches a claim that presupposed the old state.

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
from pathlib import Path
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
    parse_json_object,
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
from particles.ingest.pipeline import update_decided_by

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
        #: Claims the extractor labels ALEATORY; every other claim is EPISTEMIC.
        self.aleatory: set[str] = set()

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> ExtractionResult:
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=claim,
                    confidence_value=0.9,
                    uncertainty_nature=(
                        UncertaintyNature.ALEATORY
                        if claim in self.aleatory
                        else UncertaintyNature.EPISTEMIC
                    ),
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
    def test_the_winner_is_exempt_only_when_the_restatement_is_dated_before_it(self) -> None:
        """the date guard, by calendar date, failing closed."""
        from particles.core.reanchor import exempts_winner

        moved = datetime(2026, 10, 9, 17, 0, tzinfo=UTC)
        assert exempts_winner("2026-10-08", moved)
        assert not exempts_winner("2026-10-09", moved)
        assert not exempts_winner("2026-10-11", moved)
        assert not exempts_winner(None, moved)
        assert not exempts_winner("2026-10-08", None)
        assert not exempts_winner("not a date", moved)

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

    def test_probe_reply_tolerates_how_the_label_is_echoed(self) -> None:
        reply = (
            '{"claims": [{"claim": "C1", "verdict": "HOLDS", "reason": "x"}, '
            '{"claim": "[c2]:", "verdict": "HOLDS", "reason": "y"}]}'
        )
        assert [v.candidate_id for v in parse_probe_reply(reply, {"c1": "a", "c2": "b"}) or []] == [
            "a",
            "b",
        ]

    def test_probe_reply_falls_back_to_the_claim_text(self) -> None:
        """claude-sonnet-5 put each claim's text where its label belonged (2026-10-04)."""
        reply = (
            '{"claims": [{"claim": "The user is vegetarian", "verdict": "HOLDS", '
            '"reason": "x"}, {"claim": "Not a candidate.", "verdict": "HOLDS", "reason": "y"}]}'
        )
        verdicts = parse_probe_reply(reply, {"c1": "a"}, {"a": VEG})
        assert verdicts == [ProbeVerdict("a", DependencyVerdict.HOLDS, None, "x")]
        assert parse_probe_reply(reply, {"c1": "a"}) == []

    def test_the_probe_schema_holds_the_claim_to_this_calls_labels(self) -> None:
        from particles.operations.reanchor import _PROBE_SCHEMA, _probe_schema

        schema = _probe_schema(["c1", "c2"])
        claim = schema["properties"]["claims"]["items"]["properties"]["claim"]
        assert claim == {"type": "string", "enum": ["c1", "c2"]}
        assert "enum" not in _PROBE_SCHEMA["properties"]["claims"]["items"]["properties"]["claim"]

    def test_reading_reply_needs_both_booleans(self) -> None:
        assert parse_reading_reply('{"depended": true, "faithful": false, "reason": "r"}') == (
            Reading(depended=True, faithful=False, reason="r")
        )
        assert parse_reading_reply('{"depended": true}') is None

    def test_reading_reply_missing_its_closing_quote_is_repaired(self) -> None:
        """claude-sonnet-5-5 dropped the quote that closes ``reason`` (2026-10-11).

        The reply verbatim from the captured run: ``end_turn``, no thinking, one
        text block. Only the closing quote is missing.
        """
        reply = (
            '{"depended": true, "faithful": true, "reason": "The original\'s \'the '
            "user's son's class' refers by possessive description to the circumstance "
            "named in the retired claim (Mrs Patel's Year 3 class), so it would point "
            "at a new class after the change; the restatement names that class, is "
            "dated 2025-10-20, uses past tense, and keeps the 28-pupil detail without "
            "adding anything else.}"
        )
        reading = parse_reading_reply(reply)
        assert reading is not None
        assert (reading.depended, reading.faithful) == (True, True)
        assert reading.reason.endswith("without adding anything else.")
        # Whitespace or a fence between the string and its brace is tolerated too.
        assert parse_reading_reply(
            '```json\n{"depended": false, "faithful": true, "reason": "r.\n}\n```'
        ) == Reading(depended=False, faithful=True, reason="r.")

    def test_a_repair_never_rescues_a_reply_that_is_still_broken(self) -> None:
        # Cut off inside the string: no closing brace, nothing to put the quote before.
        assert parse_reading_reply('{"depended": true, "faithful": true, "reason": "cut') is None
        # A quote lost mid-object fails as a missing delimiter, which is never repaired.
        assert parse_reading_reply('{"depended": true, "reason": "r, "faithful": true}') is None
        # A different defect (a bare word) is not an unterminated string.
        assert parse_reading_reply('{"depended": yes, "faithful": true, "reason": "r"}') is None
        # The probe reply: the quote goes before the whole run of closing brackets.
        assert parse_json_object('{"claims": [{"claim": "c1", "reason": "x.}]}') == {
            "claims": [{"claim": "c1", "reason": "x."}]
        }

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
        # guard 1: the mark rung 2.5 reads to leave it alone.
        from particles.core.reanchor import is_reanchor_restatement

        assert is_reanchor_restatement(written)
        assert not is_reanchor_restatement(original)

    def test_the_restatement_carries_this_sdks_stamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """D2: a minted record never copies its predecessor's stamp."""
        monkeypatch.setattr("particles.core.reanchor.SCHEMA_VERSION", "1.2.0")
        original = Particle(
            content=WALK,
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[
                ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id="s")
            ],
            asserted_by="general-extractor",
            schema_version="1.0.0",
        )
        retired = original.model_copy(update={"id": str(uuid.uuid4()), "content": RESIDENCE})
        written = build_restatement(original, retired, RESTATED, provider_model=None, now=T0)
        assert written.schema_version == "1.2.0"


# ---------------------------------------------------------------------------
# The pass, on the Delhi scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDelhi:
    async def test_the_walking_claim_is_restated(
        self, db_session: Any, bow: None, candidate_cache_mode: str
    ) -> None:
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

    async def test_a_candidate_with_no_verdict_is_asked_once_more(
        self, db_session: Any, bow: None
    ) -> None:
        from particles.operations.consolidation import render_reanchor_line
        from particles.operations.reanchor import run_reanchor

        await _delhi(db_session)
        calls: list[list[str]] = []

        async def partial(
            retired: Particle, passage: str, date: str | None, cands: list[Particle]
        ) -> list[ProbeVerdict]:
            calls.append([c.content for c in cands])
            # First ask: only the vegetarian claim is answered. Second ask: the
            # walk claim is answered, the tikka claim never is.
            return [
                ProbeVerdict(c.id, DependencyVerdict.HOLDS, None, "holds")
                for c in cands
                if c.content == (VEG if len(calls) == 1 else WALK)
            ]

        with (
            _Scripted() as _probe,
            patch("particles.operations.reanchor._probe", AsyncMock(side_effect=partial)),
        ):
            report = await run_reanchor(db_session, cursor=None)
            await db_session.commit()

        assert [sorted(c) for c in calls] == [sorted([WALK, TIKKA, VEG]), sorted([WALK, TIKKA])]
        assert report.probes == 2 and report.holds == 2 and report.unanswered == 1
        assert report.cursor is not None
        assert any("no verdict for 1 candidate" in w for w in report.warnings)
        assert report.payload()["unanswered"] == 1
        assert "1 candidate(s) the probe gave no verdict for" in render_reanchor_line(report)

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


# ---------------------------------------------------------------------------
# The Ledgerly case: a project moves from PostgreSQL to SQLite
# ---------------------------------------------------------------------------

#: Session 3 (2026-09-04) and session 9 (2026-09-20), as general-extractor
#: 0.16.0 cut them on the released 1.172.0 (the r/AIMemory Rung 1 thread's
#: Postgres-to-SQLite case). Session 9 is in the order the extractor emitted it:
#: the single-file claim comes before the claim that updates the database.
PG = "Ledgerly uses PostgreSQL 16 as its database."
CONTAINER = (
    "Before running Ledgerly's test suite, the database container must be started "
    "with `docker compose up -d db`, otherwise pytest fails to connect."
)
NO_CONTAINER = (
    "If the `db` container is not running, pytest fails to connect when running "
    "Ledgerly's test suite."
)
JSONB = "Ledgerly's migrations in `migrations/` use JSONB columns for the audit log."
PG_ONLY = (
    "Ledgerly's migrations in `migrations/` cannot be run against databases other "
    "than PostgreSQL (which supports JSONB)."
)
MOVED_PG = "Ledgerly was moved from PostgreSQL to SQLite in the week before 2026-09-20."
ONE_FILE = "Ledgerly's entire database is stored in a single file named `ledgerly.db`."
SQLITE = "Ledgerly uses SQLite as of 2026-09-20."
SIMPLER = "Ledgerly's deploys became simpler after the migration from PostgreSQL to SQLite."
CONTAINER_RESTATED = (
    "As of 2026-09-04, when Ledgerly used PostgreSQL 16, its test suite needed the "
    "database container started with `docker compose up -d db`, otherwise pytest "
    "failed to connect."
)

_LEDGERLY = "Ledgerly"
SESSION_3 = [(c, [_LEDGERLY]) for c in (PG, CONTAINER, NO_CONTAINER, JSONB, PG_ONLY)]
SESSION_9 = [(c, [_LEDGERLY]) for c in (MOVED_PG, ONE_FILE, SQLITE, SIMPLER)]
_DEPENDENTS = {CONTAINER, NO_CONTAINER, JSONB, PG_ONLY}

#: The write-time probes as the live run answered them: both newer claims
#: contradict the Postgres claim, and only "uses SQLite" fills its slot with a
#: new value. "A single file" is not an update of the database, so that pair
#: reaches rung 3 on its own.
_LEDGERLY_CONFLICTS: dict[frozenset[str], SlotVerdict] = {
    frozenset((PG, SQLITE)): SlotVerdict.CHANGES,
    frozenset((PG, ONE_FILE)): SlotVerdict.FIXED,
}


@pytest.fixture
def ledgerly_probes(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """A bag-of-words encoder and the write-time probes scripted for the Ledgerly pairs."""
    from particles import embeddings as ep
    from particles.config import get_config

    async def contradicts(a: str, b: str) -> bool:
        return frozenset((a, b)) in _LEDGERLY_CONFLICTS

    async def updates(earlier: str, later: str) -> SlotVerdict:
        return _LEDGERLY_CONFLICTS.get(frozenset((earlier, later)), SlotVerdict.DIFFERENT)

    # Every session-3 claim is in the subject pool, so the scripted probes
    # alone decide which pairs reach the ladder; the bag-of-words cosine
    # between differently worded claims would otherwise gate them out.
    update_cfg = get_config().reconciliation.update_supersession
    monkeypatch.setattr(update_cfg, "subject_floor", 0.0)
    monkeypatch.setattr(update_cfg, "max_candidates", 10)
    # The second reading has its own tests; off so the pairs reach
    # the ladder on the scripted probes.
    monkeypatch.setattr(get_config().extraction, "verify_conflicts", False)
    original = ep._embedding_model
    ep.set_embedding_model(_BagOfWords())  # type: ignore[arg-type]
    try:
        with (
            patch(
                "particles.ingest.pipeline._has_contradiction_signal",
                AsyncMock(side_effect=contradicts),
            ),
            patch("particles.ingest.pipeline._has_update_signal", AsyncMock(side_effect=updates)),
        ):
            yield
    finally:
        ep.set_embedding_model(original)


async def _ledgerly(session: Any) -> None:
    """Sessions 3 and 9 extracted, with rung 2.5 left to retire the Postgres claim."""
    ex = _Sessions()
    await _session(session, ex, SESSION_3, day=4)
    await _session(session, ex, SESSION_9, day=20)


@pytest.mark.asyncio
class TestLedgerly:
    async def test_the_update_settles_before_its_sibling_meets_the_old_claim(
        self, db_session: Any, ledgerly_probes: None
    ) -> None:
        """The newer source's claim is written ACTIVE, not quarantined behind the old one.

        On 1.172.0 the write loop took session 9 in the extractor's order: the
        single-file claim met "uses PostgreSQL 16" while it was still ACTIVE,
        went to rung 3 and was born ``CONFLICT_PENDING``; the next claim then
        retired the Postgres claim as an update, leaving the record with no
        ACTIVE side and the new fact held.
        """
        from particles.store.particle_store import get_particles_by_status

        await _ledgerly(db_session)

        pg = await _one(db_session, PG)
        assert (pg.status, pg.status_reason) == (
            Status.PROVENANCE_STALE,
            StatusReason.SUPERSEDED_BY_UPDATE,
        )
        sqlite = await _one(db_session, SQLITE)
        assert sqlite.status is Status.ACTIVE and sqlite.supersedes == pg.id
        one_file = await _one(db_session, ONE_FILE)
        assert (one_file.status, one_file.status_reason) == (Status.ACTIVE, None)
        assert await get_particles_by_status(db_session, Status.INCONSISTENCY) == []

    async def test_the_dependents_reach_the_probe(
        self, db_session: Any, ledgerly_probes: None
    ) -> None:
        """Candidacy is not what kept the Postgres-era instructions current.

        All four co-extracted claims share the snapshot and the subject with
        the retired claim, and are sent to the probe. On 1.172.0 the probe
        judged each one HOLDS: its rubric counts only a relative reference to
        the old state ("my flat") as dependence, and these presuppose it.
        """
        from particles.operations.reanchor import run_reanchor

        await _ledgerly(db_session)
        with _Scripted(depends={}) as probe:
            report = await run_reanchor(db_session, cursor=None, scope="store")
            await db_session.commit()

        assert probe.await_count == 1
        [(retired, cands)] = [(call.args[0], call.args[3]) for call in probe.await_args_list]
        assert retired.content == PG
        assert {c.content for c in cands} == _DEPENDENTS
        assert report.triggers == 1 and report.holds == 4 and report.restated == []

    async def test_a_presupposing_claim_judged_dependent_is_restated_as_history(
        self, db_session: Any, ledgerly_probes: None
    ) -> None:
        """Once the probe calls the container step dependent, the pass dates it.

        The write side needs no change for a dependency of this kind: the
        restatement names the old state and the date, and the instruction
        stops reading as a current step.
        """
        from particles.operations.reanchor import run_reanchor
        from particles.store.particle_store import get_particles_by_status

        await _ledgerly(db_session)
        container = await _one(db_session, CONTAINER)
        pg = await _one(db_session, PG)
        with _Scripted(depends={CONTAINER: CONTAINER_RESTATED}):
            report = await run_reanchor(db_session, cursor=None, scope="store")
            await db_session.commit()

        assert len(report.restated) == 1
        old = await _one(db_session, CONTAINER)
        assert (old.status, old.status_reason) == (
            Status.SUPERSEDED,
            StatusReason.SUPERSEDED_BY_REANCHOR,
        )
        new = await _one(db_session, CONTAINER_RESTATED)
        assert new.status is Status.ACTIVE and new.supersedes == container.id
        assert [
            r.corpus_entry_id for r in new.provenance if r.type is ProvenanceRefType.PARTICLE
        ] == [pg.id]
        active = [p.content for p in await get_particles_by_status(db_session, Status.ACTIVE)]
        current_steps = [c for c in active if "docker compose up -d db" in c and "As of" not in c]
        assert current_steps == []

    async def test_a_later_update_never_retires_the_restatement(
        self, db_session: Any, ledgerly_probes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """guard 1: dated history is never either side of rung 2.5.

        The restatement keeps the original's SOURCE refs, so it is
        extractor-asserted and in the Postgres claim's lineage. With both sweep
        probes answering yes to everything, every other same-subject pair may
        retire, but the restatement is never paired.
        """
        from particles.ingest.pipeline import SlotReading
        from particles.operations import reconcile
        from particles.operations.reanchor import run_reanchor

        await _ledgerly(db_session)
        with _Scripted(depends={CONTAINER: CONTAINER_RESTATED}):
            await run_reanchor(db_session, cursor=None, scope="store")
            await db_session.commit()
        restated = await _one(db_session, CONTAINER_RESTATED)
        assert restated.status is Status.ACTIVE

        asked: list[tuple[str, str]] = []

        async def read(earlier: str, later: str) -> SlotReading:
            asked.append((earlier, later))
            return SlotReading(SlotVerdict.CHANGES, "verification")

        monkeypatch.setattr(reconcile, "_has_contradiction_signal", AsyncMock(return_value=True))
        monkeypatch.setattr(reconcile, "_read_update_slot", read)
        await reconcile.reconcile_updates(db_session)
        await db_session.commit()

        assert (await _one(db_session, CONTAINER_RESTATED)).status is Status.ACTIVE
        assert asked, "the sweep had other pairs to ask"
        assert not any(CONTAINER_RESTATED in pair for pair in asked)


# ---------------------------------------------------------------------------
# Trip versus move: a settled report retires an earlier plan
# ---------------------------------------------------------------------------

#: The r/AIMemory commenter's case (``tests/fixtures/trip_vs_move/``): "I live
#: in Delhi, but I'm in Mumbai for a week", then "I moved to Mumbai". The claims
#: are the ones claude-sonnet-4-6 cut from the notes on 1.178.1, and the return
#: flight is labelled ALEATORY, as that extractor labelled it.
TRIP_FIXTURE = Path(__file__).parent / "fixtures" / "trip_vs_move"

TRIP = "The user is in Mumbai for a week for work."
HOTEL = "The user is staying at a hotel in Bandra, Mumbai."
FLIGHT = "The user flies back home to Delhi on Thursday, 2026-10-15."
HOME = "The user's home is in Delhi."
DINNER = "The user wants a vegetarian restaurant for dinner on the evening of 2026-10-08."
DECIDED = "The user has decided to move to Mumbai permanently."
NOT_BACK = "The user is not going back to Delhi."
LEASE = "The user has signed a lease on a flat in Bandra, Mumbai."
MOVED_LITERAL = "The user moved to Mumbai."
FLIGHT_RESTATED = (
    "When the user was in Mumbai for a week for work, they were scheduled to fly back "
    "home to Delhi on Thursday, 2026-10-15, as of 2026-10-08."
)

TRIP_SESSION_4 = [(RESIDENCE, [_USER]), (WALK, [_PLACE, _USER]), (VEG, [_USER])]
TRIP_SESSION_11 = [(c, [_USER]) for c in (TRIP, HOTEL, FLIGHT, HOME, DINNER)]
TRIP_SESSION_12 = [(c, [_USER]) for c in (DECIDED, NOT_BACK, LEASE)]
TRIP_SESSION_12_MOVED = [(MOVED_LITERAL, [_USER])]

#: The write-time probes as the 2026-10-09 replay answered them: the
#: contradiction probe said YES and the update probe CHANGES (3 of 3, both
#: readings) for the flight against "not going back"; the flight against the
#: decision to move fills a different slot, and nothing in session 11
#: contradicts the residence. The literal note retires the trip and the
#: residence, and leaves the flight to the re-anchor pass.
_TRIP_CONFLICTS: dict[frozenset[str], SlotVerdict] = {
    frozenset((TRIP, DECIDED)): SlotVerdict.CHANGES,
    frozenset((FLIGHT, NOT_BACK)): SlotVerdict.CHANGES,
    frozenset((TRIP, MOVED_LITERAL)): SlotVerdict.CHANGES,
    frozenset((RESIDENCE, MOVED_LITERAL)): SlotVerdict.CHANGES,
}


@pytest.fixture
def trip_probes(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """A bag-of-words encoder and the write-time probes scripted for the trip pairs."""
    from particles import embeddings as ep
    from particles.config import get_config

    async def contradicts(a: str, b: str) -> bool:
        return frozenset((a, b)) in _TRIP_CONFLICTS

    async def updates(earlier: str, later: str) -> SlotVerdict:
        return _TRIP_CONFLICTS.get(frozenset((earlier, later)), SlotVerdict.DIFFERENT)

    # As in the Ledgerly fixture: the scripted probes alone decide which pairs
    # reach the ladder, and the second reading has its own tests.
    update_cfg = get_config().reconciliation.update_supersession
    monkeypatch.setattr(update_cfg, "subject_floor", 0.0)
    monkeypatch.setattr(update_cfg, "max_candidates", 10)
    monkeypatch.setattr(get_config().extraction, "verify_conflicts", False)
    original = ep._embedding_model
    ep.set_embedding_model(_BagOfWords())  # type: ignore[arg-type]
    try:
        with (
            patch(
                "particles.ingest.pipeline._has_contradiction_signal",
                AsyncMock(side_effect=contradicts),
            ),
            patch("particles.ingest.pipeline._has_update_signal", AsyncMock(side_effect=updates)),
        ):
            yield
    finally:
        ep.set_embedding_model(original)


async def _note(
    session: Any, ex: _Sessions, name: str, claims: list[tuple[str, list[str]]], day: float
) -> None:
    """Deposit one fixture note, dated ``day``, and extract the scripted claims from it."""
    from particles.corpus.deposit import deposit_text_versioned
    from particles.ingest.pipeline import extract_snapshot

    text = (TRIP_FIXTURE / name).read_text()
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


async def _trip(session: Any) -> _Sessions:
    """Sessions 4 (2026-09-04) and 11 (2026-10-08) extracted."""
    ex = _Sessions()
    ex.aleatory.add(FLIGHT)
    await _note(session, ex, "session-04.md", TRIP_SESSION_4, day=3)
    await _note(session, ex, "session-11.md", TRIP_SESSION_11, day=37)
    return ex


async def _held(session: Any) -> list[Particle]:
    from particles.store.particle_store import get_particles_by_status

    stale = await get_particles_by_status(session, Status.PROVENANCE_STALE)
    return [p for p in stale if p.status_reason is StatusReason.CONFLICT_PENDING]


@pytest.mark.asyncio
class TestTripVersusMove:
    async def test_a_trip_leaves_the_residence_current(
        self, db_session: Any, trip_probes: None
    ) -> None:
        """Half one: a week in Mumbai is not a move."""
        from particles.store.particle_store import get_particles_by_status

        await _trip(db_session)
        residence = await _one(db_session, RESIDENCE)
        assert (residence.status, residence.status_reason) == (Status.ACTIVE, None)
        assert await _held(db_session) == []
        assert await get_particles_by_status(db_session, Status.INCONSISTENCY) == []
        flight = await _one(db_session, FLIGHT)
        assert flight.uncertainty_nature is UncertaintyNature.ALEATORY

    async def test_a_settled_report_retires_the_return_flight(
        self, db_session: Any, trip_probes: None
    ) -> None:
        """Half two: "not going back" retires the flight the same note made false.

        On 1.178.1 the flight's ALEATORY label kept rung 2.5 off the pair:
        "not going back to Delhi" was held ``CONFLICT_PENDING`` behind it, and
        the query said the user was due back on Thursday.
        """
        from particles.store.event_store import OperatorEventType, list_events
        from particles.store.particle_store import get_particles_by_status

        ex = await _trip(db_session)
        await _note(db_session, ex, "session-12.md", TRIP_SESSION_12, day=38)

        active = [p.content for p in await get_particles_by_status(db_session, Status.ACTIVE)]
        assert [c for c in active if "back home to Delhi" in c] == []
        not_back = await _one(db_session, NOT_BACK)
        assert (not_back.status, not_back.status_reason) == (Status.ACTIVE, None)
        assert await _held(db_session) == []

        flight = await _one(db_session, FLIGHT)
        assert (flight.status, flight.status_reason) == (
            Status.PROVENANCE_STALE,
            StatusReason.SUPERSEDED_BY_UPDATE,
        )
        events = await list_events(db_session, event_type=OperatorEventType.UPDATE_SUPERSESSION)
        assert {(e.payload["retired_id"], e.payload["winner_id"]) for e in events} == {
            ((await _one(db_session, TRIP)).id, (await _one(db_session, DECIDED)).id),
            (flight.id, not_back.id),
        }

    async def test_the_move_retires_both_phrasings_of_the_residence(
        self, db_session: Any, trip_probes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the second phrasing of the old residence goes with the first.

        As on 1.178.1, the sweep's contradiction probe confirms the residence
        against the move but misses "home is in Delhi" (it said YES 1 of 3 on
        the replay), so the sweep retires one phrasing. The paraphrases pass
        asks the update probe of the other, with no contradiction probe.
        """
        from particles.config import get_config
        from particles.ingest.pipeline import SlotReading
        from particles.operations import paraphrases, reconcile
        from particles.store.event_store import OperatorEventType, list_events

        ex = await _trip(db_session)
        await _note(db_session, ex, "session-12.md", TRIP_SESSION_12, day=38)
        assert (await _one(db_session, HOME)).status is Status.ACTIVE

        # The bag-of-words encoder scores the pair 0.516; the store's own
        # encoder scored it 0.785. Every other user claim is at 0.50 or below.
        update_cfg = get_config().reconciliation.update_supersession
        # This pins the paraphrases route as first built, with the
        # contradiction probe in front of the sweep. The sweep now asks the
        # update question first and retires the second phrasing itself (ADR
        # 0313; the parametrized test above).
        monkeypatch.setattr(update_cfg, "same_slot_admits", False)
        monkeypatch.setattr(update_cfg, "paraphrase_floor", 0.51)
        confirmed = {frozenset((RESIDENCE, DECIDED))}
        changes = {(RESIDENCE, DECIDED), (HOME, DECIDED)}

        async def contradicts(a: str, b: str) -> bool:
            return frozenset((a, b)) in confirmed

        async def read(earlier: str, later: str) -> SlotReading:
            if (earlier, later) in changes:
                return SlotReading(SlotVerdict.CHANGES, "verification")
            return SlotReading(SlotVerdict.DIFFERENT, "semantic_lint")

        contradiction = AsyncMock(side_effect=contradicts)
        monkeypatch.setattr(reconcile, "_has_contradiction_signal", contradiction)
        monkeypatch.setattr(reconcile, "_read_update_slot", read)
        monkeypatch.setattr(paraphrases, "_read_update_slot", read)

        await reconcile.reconcile_updates(db_session)
        await db_session.commit()
        assert (await _one(db_session, HOME)).status is Status.ACTIVE
        asked = contradiction.await_count
        report = await paraphrases.run_paraphrases(db_session, cursor=None)
        await db_session.commit()
        assert contradiction.await_count == asked

        decided = await _one(db_session, DECIDED)
        for content in (RESIDENCE, HOME):
            retired = await _one(db_session, content)
            assert (retired.status, retired.status_reason) == (
                Status.PROVENANCE_STALE,
                StatusReason.SUPERSEDED_BY_UPDATE,
            )
        events = await list_events(db_session, event_type=OperatorEventType.UPDATE_SUPERSESSION)
        winners = {e.payload["retired_id"]: e.payload for e in events}
        residence, home = await _one(db_session, RESIDENCE), await _one(db_session, HOME)
        assert winners[residence.id]["winner_id"] == decided.id
        assert winners[home.id] == {
            "retired_id": home.id,
            "winner_id": decided.id,
            "route": "paraphrase",
            "decided_by": update_decided_by(),
        }
        assert [r["retired"] for r in report.retired] == [home.id]

    async def test_the_literal_move_restates_the_flight(
        self, db_session: Any, trip_probes: None
    ) -> None:
        """The commenter's own phrasing: "I moved to Mumbai", with no "not going back".

        Nothing in that note contradicts the flight, so the re-anchor pass is
        the only route: the flight relied on the trip the note retired. It is
        now a candidate whatever its label, and its restatement
        keeps the label.
        """
        from particles.operations.reanchor import run_reanchor
        from particles.store.particle_store import get_particles_by_status

        ex = await _trip(db_session)
        await _note(db_session, ex, "session-12-moved.md", TRIP_SESSION_12_MOVED, day=38)
        trip = await _one(db_session, TRIP)
        assert trip.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        flight = await _one(db_session, FLIGHT)
        assert flight.status is Status.ACTIVE

        with _Scripted(depends={FLIGHT: FLIGHT_RESTATED}) as probe:
            report = await run_reanchor(db_session, cursor=None, scope="store")
            await db_session.commit()

        by_trigger = {call.args[0].content: call.args[3] for call in probe.await_args_list}
        assert FLIGHT in {c.content for c in by_trigger[TRIP]}
        assert len(report.restated) == 1

        old = await _one(db_session, FLIGHT)
        assert (old.status, old.status_reason) == (
            Status.SUPERSEDED,
            StatusReason.SUPERSEDED_BY_REANCHOR,
        )
        new = await _one(db_session, FLIGHT_RESTATED)
        assert new.status is Status.ACTIVE and new.supersedes == flight.id
        assert new.uncertainty_nature is UncertaintyNature.ALEATORY
        assert [
            r.corpus_entry_id for r in new.provenance if r.type is ProvenanceRefType.PARTICLE
        ] == [trip.id]
        active = [p.content for p in await get_particles_by_status(db_session, Status.ACTIVE)]
        assert FLIGHT not in active

    @pytest.mark.parametrize("admits", [True, False])
    async def test_a_move_the_contradiction_probe_clears_still_retires_the_trip(
        self, db_session: Any, trip_probes: None, monkeypatch: pytest.MonkeyPatch, admits: bool
    ) -> None:
        """: the ``claude-sonnet-5-5`` shape of the trip case.

        On 1.185.0 the contradiction probe cleared the dated move against the
        trip, the residence and the home claim in every run, so the update
        probe was never asked and nothing was retired. Rung 2.5 now rests on
        the update answer; with ``same_slot_admits`` off this reproduces the
        1.185.0 outcome.
        """
        from particles.config import get_config
        from particles.ingest.pipeline import SlotReading
        from particles.operations import reconcile

        monkeypatch.setattr(
            get_config().reconciliation.update_supersession, "same_slot_admits", admits
        )
        changes = {**_TRIP_CONFLICTS, frozenset((HOME, MOVED_LITERAL)): SlotVerdict.CHANGES}

        async def updates(earlier: str, later: str) -> SlotVerdict:
            return changes.get(frozenset((earlier, later)), SlotVerdict.DIFFERENT)

        async def read(earlier: str, later: str) -> SlotReading:
            return SlotReading(await updates(earlier, later), "verification")

        ex = await _trip(db_session)
        with (
            patch(
                "particles.ingest.pipeline._has_contradiction_signal",
                AsyncMock(return_value=False),
            ),
            patch("particles.ingest.pipeline._has_update_signal", AsyncMock(side_effect=updates)),
        ):
            await _note(db_session, ex, "session-12-moved.md", TRIP_SESSION_12_MOVED, day=38)
        monkeypatch.setattr(reconcile, "_has_contradiction_signal", AsyncMock(return_value=False))
        monkeypatch.setattr(reconcile, "_read_update_slot", read)
        await reconcile.reconcile_updates(db_session)
        await db_session.commit()

        for content in (TRIP, RESIDENCE, HOME):
            claim = await _one(db_session, content)
            if admits:
                assert claim.status_reason is StatusReason.SUPERSEDED_BY_UPDATE, content
            else:
                assert (claim.status, claim.status_reason) == (Status.ACTIVE, None), content
        assert (await _one(db_session, MOVED_LITERAL)).status is Status.ACTIVE
        assert await _held(db_session) == []

    @staticmethod
    async def _moved_and_reanchored(
        db_session: Any, *, moved_day: float, contradicts: Any
    ) -> tuple[Any, Particle]:
        """The literal move, then the pass with the restatement check scripted."""
        from particles.operations.reanchor import run_reanchor

        ex = await _trip(db_session)
        await _note(db_session, ex, "session-12-moved.md", TRIP_SESSION_12_MOVED, day=moved_day)
        moved = await _one(db_session, MOVED_LITERAL)
        with (
            _Scripted(depends={FLIGHT: FLIGHT_RESTATED}),
            patch(
                "particles.operations.reconcile._has_contradiction_signal",
                AsyncMock(side_effect=contradicts),
            ),
        ):
            report = await run_reanchor(db_session, cursor=None, scope="store")
            await db_session.commit()
        return report, moved

    async def test_the_restatement_is_not_checked_against_the_move_that_ended_the_trip(
        self, db_session: Any, trip_probes: None
    ) -> None:
        """: a dated past stay does not contradict the move.

        ``claude-sonnet-4-6`` judged each of the trip's restatements to
        contradict "The user moved to Mumbai." (item 5), the update
        the restatement is dated against. Session 11's heading dates the
        passage 2026-10-08; the move is dated 2026-10-09.
        """

        async def contradicts(a: str, b: str) -> bool:
            return MOVED_LITERAL in (a, b)

        report, moved = await self._moved_and_reanchored(
            db_session, moved_day=38, contradicts=contradicts
        )
        assert len(report.restated) == 1 and report.unrestated == []
        new = await _one(db_session, FLIGHT_RESTATED)
        assert new.status is Status.ACTIVE
        assert moved.status is Status.ACTIVE

    async def test_a_restatement_dated_after_the_move_is_still_checked_against_it(
        self, db_session: Any, trip_probes: None
    ) -> None:
        """The date guard: a restatement not dated before W gets no exemption."""

        async def contradicts(a: str, b: str) -> bool:
            return MOVED_LITERAL in (a, b)

        # The move noon on 2026-10-08: strictly after the trip, so rung 2.5
        # retires it, but not dated before the passage's 2026-10-08 heading.
        report, moved = await self._moved_and_reanchored(
            db_session, moved_day=37.5, contradicts=contradicts
        )
        assert report.restated == []
        [held] = report.unrestated
        assert held["why"] == f"the restatement contradicts {moved.id}"
        assert (await _one(db_session, FLIGHT)).status is Status.ACTIVE

    async def test_the_exemption_covers_the_winner_only(
        self, db_session: Any, trip_probes: None
    ) -> None:
        """Every other standing claim is still checked."""

        async def contradicts(a: str, b: str) -> bool:
            return True

        report, moved = await self._moved_and_reanchored(
            db_session, moved_day=38, contradicts=contradicts
        )
        assert report.restated == []
        [held] = report.unrestated
        assert held["why"].startswith("the restatement contradicts ")
        assert moved.id not in held["why"]

    async def test_a_return_with_no_relative_word_is_restated_as_the_end_of_the_trip(
        self, db_session: Any, trip_probes: None
    ) -> None:
        """The second acceptance run's wording: "flies back to Delhi", labelled EPISTEMIC.

        Without "home" the claim carries no relative reference, and the probe
        judged it HOLDS. Under form (c) it states how the retired
        trip ends. The verdict is the model's; this pins that the
        write side restates it exactly as it does any other dependent.
        """
        from particles.operations.reanchor import run_reanchor
        from particles.store.particle_store import get_particles_by_status

        ex = _Sessions()
        await _note(db_session, ex, "session-04.md", TRIP_SESSION_4, day=3)
        await _note(db_session, ex, "session-11.md", TRIP_SESSION_11_RUN_2, day=37)
        await _note(db_session, ex, "session-12-moved.md", TRIP_SESSION_12_MOVED, day=38)
        trip = await _one(db_session, TRIP)
        assert trip.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
        flight = await _one(db_session, FLIGHT_RUN_2)
        assert flight.status is Status.ACTIVE
        assert flight.uncertainty_nature is UncertaintyNature.EPISTEMIC

        with _Scripted(depends={FLIGHT_RUN_2: FLIGHT_RUN_2_RESTATED}):
            report = await run_reanchor(db_session, cursor=None, scope="store")
            await db_session.commit()

        assert len(report.restated) == 1
        old = await _one(db_session, FLIGHT_RUN_2)
        assert (old.status, old.status_reason) == (
            Status.SUPERSEDED,
            StatusReason.SUPERSEDED_BY_REANCHOR,
        )
        new = await _one(db_session, FLIGHT_RUN_2_RESTATED)
        assert new.status is Status.ACTIVE and new.supersedes == flight.id
        assert new.uncertainty_nature is UncertaintyNature.EPISTEMIC
        assert [
            r.corpus_entry_id for r in new.provenance if r.type is ProvenanceRefType.PARTICLE
        ] == [trip.id]
        active = [p.content for p in await get_particles_by_status(db_session, Status.ACTIVE)]
        assert FLIGHT_RUN_2 not in active


#: The second acceptance run of the settled-report rule, on claude-sonnet-4-6,
#: 2026-10-10: the flight came back without "home" and labelled
#: EPISTEMIC, and the probe judged it HOLDS against the retired trip.
FLIGHT_RUN_2 = "The user flies back to Delhi on Thursday, 2026-10-15."
FLIGHT_RUN_2_RESTATED = (
    "As of 2026-10-08, when the user was in Mumbai for a week for work, the user was "
    "due to fly back to Delhi on Thursday, 2026-10-15."
)
TRIP_SESSION_11_RUN_2 = [(c, [_USER]) for c in (TRIP, HOTEL, FLIGHT_RUN_2, HOME, DINNER)]


#: The first acceptance run of the settled-report rule, on
#: claude-sonnet-4-6, 2026-10-10: session 12's claims came back with the place
#: alone as their subject, no speaker. Nothing else in the run
#: differed from the 1.178.1 sample above.
TRIP_SESSION_12_RUN_1 = [(DECIDED, ["Mumbai"]), (NOT_BACK, ["Delhi"]), (LEASE, ["Bandra"])]


@pytest.mark.asyncio
class TestRunOneReplayed:
    """the run-1 extraction, replayed deterministically.

    The extractor left the speaker out of "The user is not going back to
    Delhi.", so the claim keyed on Delhi and never met the flight or the
    residence, which key on the persona. Rung 2.5 never ran.
    """

    async def test_the_speaker_is_attached_and_the_flight_retires(
        self, db_session: Any, trip_probes: None
    ) -> None:
        from particles.store.subject_store import get_subject

        ex = await _trip(db_session)
        await _note(db_session, ex, "session-12.md", TRIP_SESSION_12_RUN_1, day=38)

        not_back = await _one(db_session, NOT_BACK)
        names = {
            (await get_subject(db_session, sid)).canonical_name for sid in not_back.subject_ids
        }  # type: ignore[union-attr]
        assert names == {"Delhi", _USER}
        flight = await _one(db_session, FLIGHT)
        assert (flight.status, flight.status_reason) == (
            Status.PROVENANCE_STALE,
            StatusReason.SUPERSEDED_BY_UPDATE,
        )
        trip = await _one(db_session, TRIP)
        assert trip.status_reason is StatusReason.SUPERSEDED_BY_UPDATE

    async def test_without_the_rule_run_1_misses_the_pair(
        self, db_session: Any, trip_probes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: with the rule off this fixture reproduces run 1."""
        from particles.config import get_config

        monkeypatch.setattr(get_config().subjects, "persona_openers", [])
        ex = await _trip(db_session)
        await _note(db_session, ex, "session-12.md", TRIP_SESSION_12_RUN_1, day=38)
        flight = await _one(db_session, FLIGHT)
        assert (flight.status, flight.status_reason) == (Status.ACTIVE, None)

    async def test_the_sweep_pairs_them_too(
        self, db_session: Any, trip_probes: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The backlog sweep reads the same key.

        Session 12 is extracted with every write-time probe saying no, so the
        pair is left for the sweep. Its gather finds it under the persona key,
        and not without it.
        """
        from particles.config import get_config
        from particles.operations.reconcile import _update_candidates

        ex = await _trip(db_session)
        # Since the update probe is asked first at write time, so both
        # write-time probes say no here.
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
            await _note(db_session, ex, "session-12.md", TRIP_SESSION_12_RUN_1, day=38)
        flight, not_back = await _one(db_session, FLIGHT), await _one(db_session, NOT_BACK)
        assert flight.status is Status.ACTIVE

        async def pairs() -> set[tuple[str, str]]:
            gathered = await _update_candidates(
                db_session, scope_ids=None, subject_floor=0.0, require_attribution=False
            )
            return {(newer.id, older.id) for _, newer, older in gathered.candidates}

        assert (not_back.id, flight.id) in await pairs()
        monkeypatch.setattr(get_config().subjects, "persona_openers", [])
        assert (not_back.id, flight.id) not in await pairs()


class TestPrompts:
    """The probe and the second reading name both kinds of dependence.

    The verdict is the model's; what is pinned is that both prompts carry the
    rule it is asked to apply, its edge on both sides, and the faithfulness
    rule that bounds a wrong verdict of the new kind.
    """

    def test_both_prompts_frame_the_retired_claim_beyond_the_speaker(self) -> None:
        from particles.operations.reanchor import _PROBE_SYSTEM, _READING_SYSTEM

        for prompt in (_PROBE_SYSTEM, _READING_SYSTEM):
            assert "a project or a system" in prompt
            assert "about the speaker's circumstances" not in prompt

    def test_both_prompts_name_the_definite_description_and_the_stated_reason(self) -> None:
        from particles.operations.reanchor import _PROBE_SYSTEM, _READING_SYSTEM

        for prompt in (_PROBE_SYSTEM, _READING_SYSTEM):
            assert "relative or possessive reference" in prompt
            assert "definite description" in prompt
            assert "the passage itself gives" in prompt

    def test_a_description_points_at_what_the_retired_claim_names(self) -> None:
        """The owner's ruling on the Ledgerly container claims (2026-10-04)."""
        from particles.operations.reanchor import _PROBE_SYSTEM, _READING_SYSTEM

        for prompt in (_PROBE_SYSTEM, _READING_SYSTEM):
            assert "points at whatever the retired claim names" in prompt
        assert "it never cancels a reference the claim makes by description" in _READING_SYSTEM
        assert "though the claim never says MySQL" in _READING_SYSTEM

    def test_both_prompts_hold_a_named_but_current_circumstance_dependent(self) -> None:
        """The second reading rejected "the user's flat in Lajpat Nagar" in 2 of 4 samples
        on claude-sonnet-5 without this sentence (2026-10-04)."""
        from particles.operations.reanchor import _PROBE_SYSTEM, _READING_SYSTEM

        for prompt in (_PROBE_SYSTEM, _READING_SYSTEM):
            assert "'the user's flat in Lajpat Nagar'" in prompt

    def test_world_knowledge_is_not_dependence(self) -> None:
        from particles.operations.reanchor import _PROBE_SYSTEM, _READING_SYSTEM

        assert "never from what you know about the world" in _PROBE_SYSTEM
        assert "is not a dependence" in _READING_SYSTEM

    def test_the_probe_shows_a_same_subject_claim_that_holds(self) -> None:
        from particles.operations.reanchor import _PROBE_SYSTEM

        assert "'Orbit's deploy script is written in Bash' HOLDS" in _PROBE_SYSTEM

    def test_a_condition_clause_needs_a_tie_in_the_claim_or_passage(self) -> None:
        from particles.operations.reanchor import _READING_SYSTEM

        assert "as the condition the claim held under" in _READING_SYSTEM
        assert "a claim that does none of these adds a link" in _READING_SYSTEM

    def test_both_prompts_name_the_end_of_a_bounded_circumstance(self) -> None:
        """Form (c), and the HOLDS edge that bounds it (§4)."""
        from particles.operations.reanchor import _PROBE_SYSTEM, _READING_SYSTEM

        for prompt in (_PROBE_SYSTEM, _READING_SYSTEM):
            assert "with a stated limit or as temporary" in prompt
            assert "describes the circumstance by its end" in prompt
            assert "even though" in prompt and "names a place and a date" in prompt
            assert "only located where the circumstance began or will end" in prompt
            assert "another person's travel, a later visit" in prompt

    def test_the_reading_admits_a_condition_clause_on_a_claim_stating_the_end(self) -> None:
        """Without this the second reading rejects every form (c) restatement."""
        from particles.operations.reanchor import _READING_SYSTEM

        assert "refers to that circumstance, states how it ends, or the" in _READING_SYSTEM

    def test_the_reading_does_not_narrow_question_1_in_its_clause_rule(self) -> None:
        """Listing three ties read as exhaustive dropped a named-place claim (item 2)."""
        from particles.operations.reanchor import _READING_SYSTEM

        assert "these three ties are not" in _READING_SYSTEM
        assert "a narrower test than question 1" in _READING_SYSTEM

    def test_the_prompts_do_not_carry_the_acceptance_fixture(self) -> None:
        """The Ledgerly case is acceptance evidence, so neither prompt may quote it."""
        from particles.operations.reanchor import _PROBE_SYSTEM, _READING_SYSTEM

        for prompt in (_PROBE_SYSTEM, _READING_SYSTEM):
            assert "Ledgerly" not in prompt
            assert "PostgreSQL" not in prompt

    def test_the_prompts_do_not_carry_the_end_of_stay_measurement_set(self) -> None:
        """The form (c) example is a secondment, not a case the measurement set holds."""
        from particles.operations.reanchor import _PROBE_SYSTEM, _READING_SYSTEM

        for prompt in (_PROBE_SYSTEM, _READING_SYSTEM):
            for place in ("Mumbai", "Kyoto", "Porto", "KubeCon", "Bangalore", "Hooli"):
                assert place not in prompt


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

    async def _stamp(self, session: Any, content: str, stamp: str) -> Particle:
        from particles.store.particle_store import ParticleRow

        particle = await _one(session, content)
        row = await session.get(ParticleRow, particle.id)
        row.schema_version = stamp
        await session.commit()
        return particle

    async def test_a_claim_written_by_a_newer_minor_is_left_active(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D2: an older SDK never replaces a newer record with its own copy."""
        from particles.operations.consolidation import render_reanchor_line
        from particles.operations.reanchor import run_reanchor

        monkeypatch.setattr("particles.operations.version_guard.SCHEMA_VERSION", "1.2.0")
        await _delhi(db_session)
        walk = await self._stamp(db_session, WALK, "1.3.0")
        with _Scripted() as probe:
            report = await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        [cands] = [call.args[3] for call in probe.await_args_list]
        assert WALK not in {c.content for c in cands}
        assert report.skipped_newer_schema == 1
        assert report.payload()["skipped_newer_schema"] == 1
        assert "1 candidate(s) written by a newer schema" in render_reanchor_line(report)
        assert not report.restated
        kept = await _one(db_session, WALK)
        assert kept.id == walk.id and kept.status is Status.ACTIVE

    @pytest.mark.parametrize("stamp", ["1.1.0", "1.2.0"])
    async def test_a_claim_at_or_below_the_sdk_minor_is_restated(
        self, db_session: Any, bow: None, monkeypatch: pytest.MonkeyPatch, stamp: str
    ) -> None:
        from particles.operations.reanchor import run_reanchor

        monkeypatch.setattr("particles.operations.version_guard.SCHEMA_VERSION", "1.2.0")
        await _delhi(db_session)
        await self._stamp(db_session, WALK, stamp)
        with _Scripted():
            report = await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        assert report.skipped_newer_schema == 0
        assert len(report.restated) == 1
        old = await _one(db_session, WALK)
        assert (old.status, old.status_reason) == (
            Status.SUPERSEDED,
            StatusReason.SUPERSEDED_BY_REANCHOR,
        )


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

    async def test_the_premise_ref_is_not_a_broken_corpus_link(
        self, db_session: Any, bow: None
    ) -> None:
        """The premise ref carries a particle id, not a snapshot id.

        The corpus-link check used to resolve it as a snapshot, so every
        restatement surfaced as a ``broken_provenance`` card and ``lint --fix``
        demoted it to PROVENANCE_STALE.
        """
        from particles.operations.lint.staleness import _check_corpus_link_integrity
        from particles.operations.reanchor import run_reanchor

        await _delhi(db_session)
        with _Scripted():
            await run_reanchor(db_session, cursor=None)
            await db_session.commit()
        new = await _one(db_session, RESTATED)
        assert any(r.type is ProvenanceRefType.PARTICLE for r in new.provenance)
        flagged = {f.particle_id for f in await _check_corpus_link_integrity(db_session, fix=True)}
        assert new.id not in flagged
        assert (await _one(db_session, RESTATED)).status is Status.ACTIVE

        # A SOURCE ref beside the premise ref is still checked.
        from particles.store.particle_store import insert_particle

        dangling = ProvenanceRef(
            type=ProvenanceRefType.SOURCE,
            corpus_entry_id=str(uuid.uuid4()),
            snapshot_id=str(uuid.uuid4()),
        )
        broken = new.model_copy(
            update={
                "id": str(uuid.uuid4()),
                "content": "A restatement whose source snapshot is gone.",
                "provenance": [dangling, *new.provenance],
            }
        )
        await insert_particle(db_session, broken)
        await db_session.commit()
        flagged = {f.particle_id for f in await _check_corpus_link_integrity(db_session, fix=False)}
        assert flagged == {broken.id}

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


class TestStatedDateHelpers:
    """Which date a paragraph sits under. Pure."""

    _TWO_SESSIONS = (
        "# Session 3 (2026-09-04)\n\nLedgerly uses PostgreSQL 16.\n\n"
        "# Session 9 (2026-09-20)\n\nWe moved Ledgerly to SQLite.\n"
    )

    def test_a_paragraph_is_dated_by_the_nearest_dated_line_above_it(self) -> None:
        from particles.operations.reanchor import _stated_date_above

        text = self._TWO_SESSIONS
        assert _stated_date_above(text, text.find("PostgreSQL")) == "2026-09-04"
        assert _stated_date_above(text, text.find("SQLite")) == "2026-09-20"

    def test_a_dated_heading_below_the_paragraph_is_ignored(self) -> None:
        from particles.operations.reanchor import _stated_date_above

        text = "Some notes.\n\nA claim.\n\n# Later (2026-10-08)\n"
        assert _stated_date_above(text, text.find("A claim")) is None

    def test_a_whole_text_with_one_date_is_dated_by_it(self) -> None:
        from particles.operations.reanchor import _stated_date_of_whole

        text = (TRIP_FIXTURE / "session-11.md").read_text()
        assert _stated_date_of_whole(text) == "2026-10-08"

    def test_a_whole_text_with_two_dates_is_not_dated(self) -> None:
        """A multi-entry note's first heading must not date every section."""
        from particles.operations.reanchor import _stated_date_of_whole

        assert _stated_date_of_whole(self._TWO_SESSIONS) is None

    def test_a_whole_text_that_does_not_open_with_a_date_is_not_dated(self) -> None:
        from particles.operations.reanchor import _stated_date_of_whole

        assert _stated_date_of_whole("Notes\n\n# Later (2026-10-08)\n") is None


@pytest.mark.asyncio
class TestStatedDate:
    """The date the passage states reaches the probe and the second reading."""

    async def _home_retired(self, session: Any) -> tuple[Particle, Particle]:
        """Session 11 deposited as ``audit`` dated it (2026-10-11), and the home retired."""
        from particles.store.particle_store import update_particle_status

        ex = _Sessions()
        ex.aleatory.add(FLIGHT)
        # Day 40 after T0 is 2026-10-11: the mtime date the audit gave the note.
        await _note(session, ex, "session-11.md", TRIP_SESSION_11, day=40)
        home = await _one(session, HOME)
        await update_particle_status(
            session, home.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
        )
        await session.commit()
        return await _one(session, HOME), await _one(session, FLIGHT)

    async def test_the_passage_is_dated_by_its_heading(
        self, db_session: Any, trip_probes: None
    ) -> None:
        from particles.operations.reanchor import _passage_for

        home, flight = await self._home_retired(db_session)
        located = await _passage_for(db_session, home, [flight])
        assert located is not None
        passage, date = located
        assert passage.startswith("# Session 11 (2026-10-08): work trip")
        assert date == "2026-10-08"

    async def test_both_calls_are_given_the_stated_date(
        self, db_session: Any, trip_probes: None
    ) -> None:
        """item 4: the flight restated against the home's retirement."""
        import json

        from particles.operations.reanchor import run_reanchor

        await self._home_retired(db_session)
        prompts: dict[str, list[str]] = {"extraction": [], "verification": []}
        restated = (
            "As of 2026-10-08, when the user's home was in Delhi, the user was due to "
            "fly back to Delhi on Thursday, 2026-10-15."
        )

        async def llm(prompt: str, *args: Any, purpose: str, **kwargs: Any) -> str | None:
            prompts[purpose].append(prompt)
            if purpose != "extraction":
                return None  # an unusable reading: the claim is kept, nothing else is called
            labels = re.findall(r"^(c\d+):$", prompt, flags=re.MULTILINE)
            claims = [
                {"claim": label, "verdict": "DEPENDS", "restatement": restated, "reason": "r"}
                for label in labels
            ]
            return json.dumps({"claims": claims})

        with patch("particles.operations.reanchor._llm_call", AsyncMock(side_effect=llm)):
            report = await run_reanchor(db_session, cursor=None)

        assert report.probes == 1 and prompts["extraction"] and prompts["verification"]
        for prompt in prompts["extraction"] + prompts["verification"]:
            assert "Date of the source record: 2026-10-08" in prompt


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
