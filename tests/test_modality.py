# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The adjudicability default as a stamped, regenerable, lens-readable record."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sqlalchemy import select

from particles.core.modality import (
    LEGACY_MODALITY_CLASSIFIER,
    OPERATOR_CLASSIFIER,
    UNCLASSIFIED,
    LensModalityRule,
    ModalityFacts,
    ModalityReading,
    ModalityStamp,
    StampState,
    effective_modality,
    pair_adjudicable,
    resolve_stamp,
    stamp_state,
)
from particles.core.schema import (
    AssertionModality,
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    ExtractorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    TrustLensDefinition,
    TrustLensModalityRule,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.extraction.components import ComponentRecord
from particles.extraction.modality import (
    ModalityVerdict,
    current_modality_classifiers,
    parse_modality_reply,
    regeneration_classifier,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

F = AssertionModality.FALSIFIABLE
EV = AssertionModality.EVALUATIVE
EX = AssertionModality.EXPERIENTIAL
CO = AssertionModality.CONSTITUTIVE

_MODALITY = ("prompt.modality", "prompt.journal.rules")
_CLASSIFYING = ("general-extractor", "journal-extractor")


# ---------------------------------------------------------------------------
# The stamp, pure (§1)
# ---------------------------------------------------------------------------


def _resolve(**overrides: object) -> ModalityStamp:
    kwargs: dict[str, object] = {
        "stored_classifier": None,
        "stored_model": None,
        "stored_at": None,
        "extractor_name": "general-extractor",
        "provider_model": "anthropic:claude-x",
        "snapshot_components": None,
        "modality_components": _MODALITY,
        "classifying_extractors": _CLASSIFYING,
    }
    kwargs.update(overrides)
    return resolve_stamp(**kwargs)  # type: ignore[arg-type]


class TestResolveStamp:
    def test_a_stored_stamp_is_read_as_stored(self) -> None:
        at = datetime.now(UTC)
        stamp = _resolve(stored_classifier=OPERATOR_CLASSIFIER, stored_at=at)
        assert stamp == ModalityStamp(OPERATOR_CLASSIFIER, None, at)

    def test_an_assertion_is_unclassified(self) -> None:
        assert _resolve(extractor_name=None).classifier == UNCLASSIFIED

    def test_the_snapshot_record_names_the_rule(self) -> None:
        stamp = _resolve(snapshot_components={"prompt.modality": "abc", "path.chunked": "x"})
        assert stamp == ModalityStamp("prompt.modality@abc", "anthropic:claude-x")

    def test_a_record_without_a_modality_component_is_unclassified(self) -> None:
        assert _resolve(snapshot_components={"path.chunked": "x"}).classifier == UNCLASSIFIED

    def test_no_record_from_a_classifying_extractor_is_legacy(self) -> None:
        assert _resolve().classifier == LEGACY_MODALITY_CLASSIFIER

    def test_no_record_from_a_structured_extractor_is_unclassified(self) -> None:
        assert _resolve(extractor_name="numista-extractor").classifier == UNCLASSIFIED


class TestStampState:
    def test_states(self) -> None:
        current = {"prompt.modality@new"}
        assert stamp_state(ModalityStamp("prompt.modality@new"), current) == StampState.CURRENT
        assert stamp_state(ModalityStamp("prompt.modality@old"), current) == StampState.STALE
        assert stamp_state(ModalityStamp(LEGACY_MODALITY_CLASSIFIER), current) == StampState.STALE
        assert stamp_state(ModalityStamp(OPERATOR_CLASSIFIER), current) == StampState.OPERATOR
        assert stamp_state(ModalityStamp(UNCLASSIFIED), current) == StampState.UNCLASSIFIED

    def test_the_model_does_not_enter_the_state(self) -> None:
        current = {"prompt.modality@new"}
        assert (
            stamp_state(ModalityStamp("prompt.modality@new", "openai:other"), current)
            == StampState.CURRENT
        )

    def test_the_regeneration_classifier_is_a_current_identity(self) -> None:
        assert regeneration_classifier() in current_modality_classifiers()
        assert regeneration_classifier().startswith("prompt.modality@")


class TestParseReply:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"assertion_modality": "EVALUATIVE"}', EV),
            ('```json\n{"assertion_modality": "constitutive"}\n```', CO),
            ('Sure. {"assertion_modality": "FALSIFIABLE"}', F),
            ('{"assertion_modality": "OPINION"}', None),
            ('{"modality": "EVALUATIVE"}', None),
            ("not json", None),
        ],
    )
    def test_never_falls_back_to_a_default(
        self, raw: str, expected: AssertionModality | None
    ) -> None:
        assert parse_modality_reply(raw) == expected


# ---------------------------------------------------------------------------
# The lens composition, pure (§5)
# ---------------------------------------------------------------------------


def _rule(
    lens: str, scope: str, pattern: str, modality: AssertionModality, **kw: object
) -> LensModalityRule:
    return LensModalityRule(
        lens,
        TrustLensModalityRule(scope=scope, pattern=pattern, modality=modality, **kw),  # type: ignore[arg-type]
    )


_FACTS = ModalityFacts(
    particle_id="p1",
    stored=F,
    subject_ids=frozenset({"s1"}),
    source_types=frozenset({"LOCAL_MARKDOWN"}),
    source_uris=("https://www.pcgs.com/coin/123",),
)


class TestEffectiveModality:
    def test_silence_leaves_the_stored_default(self) -> None:
        assert effective_modality(_FACTS, []) == ModalityReading(F, "stored")

    def test_an_operator_verdict_wins(self) -> None:
        pinned = ModalityFacts(particle_id="p1", stored=F, operator_pinned=True)
        reading = effective_modality(pinned, [_rule("a", "particle", "p1", EV)])
        assert reading == ModalityReading(F, "operator")

    def test_within_one_lens_the_most_specific_rule_decides(self) -> None:
        # A lens carves an exception out of its own broader rule.
        rules = [
            _rule("a", "source_type", "LOCAL_MARKDOWN", CO),
            _rule("a", "particle", "p1", F),
        ]
        assert effective_modality(_FACTS, rules) == ModalityReading(F, "lens:a")

    def test_across_lenses_abstention_wins_whatever_the_scope(self) -> None:
        # Review Q3: lens a's narrow grant must not override lens b's broader
        # withholding, or adopting a lens could grant adjudication.
        rules = [
            _rule("a", "particle", "p1", F),
            _rule("b", "subject", "s1", EV),
        ]
        assert effective_modality(_FACTS, rules) == ModalityReading(EV, "lens:b")
        broader = [
            _rule("a", "subject", "s1", F),
            _rule("b", "source_type", "LOCAL_MARKDOWN", CO),
        ]
        assert effective_modality(_FACTS, broader) == ModalityReading(CO, "lens:b")

    def test_a_lens_that_only_grants_grants(self) -> None:
        rules = [_rule("a", "subject", "s1", F)]
        stored_evaluative = ModalityFacts(
            particle_id="p1", stored=EV, subject_ids=frozenset({"s1"})
        )
        assert effective_modality(stored_evaluative, rules) == ModalityReading(F, "lens:a")

    def test_abstention_wins_across_lenses_at_one_scope(self) -> None:
        rules = [
            _rule("permissive", "url_pattern", r"pcgs\.com", F),
            _rule("skeptic", "url_pattern", r"pcgs\.com", EX),
            _rule("other", "url_pattern", r"pcgs\.com", EV),
        ]
        # EVALUATIVE precedes EXPERIENTIAL in the enum, so it breaks the tie.
        assert effective_modality(_FACTS, rules) == ModalityReading(EV, "lens:other")

    def test_when_limits_a_rule_to_a_stored_value(self) -> None:
        evaluative = ModalityFacts(particle_id="p2", stored=EV, source_uris=("https://pcgs.com/x",))
        rule = _rule("grading", "url_pattern", r"pcgs\.com", F, when=EV)
        assert effective_modality(evaluative, [rule]) == ModalityReading(F, "lens:grading")
        assert effective_modality(_FACTS, [rule]).basis == "stored"

    def test_an_invalid_regex_matches_nothing(self) -> None:
        assert effective_modality(_FACTS, [_rule("a", "url_pattern", "(", EV)]).basis == "stored"

    def test_pair_adjudicable(self) -> None:
        assert pair_adjudicable(ModalityReading(F), ModalityReading(F))
        assert not pair_adjudicable(ModalityReading(F), ModalityReading(EV))


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------


async def _snapshot(session: AsyncSession, record: ComponentRecord | None) -> tuple[str, str]:
    from particles.corpus.store import CorpusEntryRow, SnapshotRow, update_extraction_status

    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="WEB_PAGE",
        uri_r=f"https://example.com/{uuid.uuid4()}",
        deposited_by="test",
    )
    session.add(CorpusEntryRow.from_model(entry))
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash="0" * 64,
        extraction_status=ExtractionStatus.IN_PROGRESS,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    session.add(SnapshotRow.from_model(snap, entry.entry_id))
    await session.flush()
    await update_extraction_status(
        session, snap.snapshot_id, ExtractionStatus.COMPLETE, components=record
    )
    return entry.entry_id, snap.snapshot_id


async def _claim(
    session: AsyncSession,
    content: str,
    *,
    record: ComponentRecord | None = None,
    modality: AssertionModality = F,
    embedding: list[float] | None = None,
) -> Particle:
    from particles.store.particle_store import insert_particle

    entry_id, snapshot_id = await _snapshot(session, record)
    particle = Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        assertion_modality=modality,
        extractor_ref=ExtractorRef(name="general-extractor", version="0.16.0"),
        extraction_provider_model="anthropic:claude-x",
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
    )
    await insert_particle(session, particle, embedding=embedding)
    return particle


def _current_record() -> ComponentRecord:
    identity = regeneration_classifier()
    name, digest = identity.split("@")
    return ComponentRecord(extractor="general-extractor", exercised={name: digest})


def _old_record() -> ComponentRecord:
    return ComponentRecord(
        extractor="general-extractor", exercised={"prompt.modality": "0000000000000000"}
    )


# ---------------------------------------------------------------------------
# A stale classifier stamp is reported (§1, §7)
# ---------------------------------------------------------------------------


class TestStaleStampReported:
    @pytest.mark.asyncio
    async def test_census_and_lint_name_the_stale_rule(self, db_session: AsyncSession) -> None:
        from particles.operations.lint.modality import _report_modality_classifiers
        from particles.operations.modality import modality_census

        await _claim(db_session, "The 1804 dollar was struck in 1834.", record=_current_record())
        await _claim(db_session, "The mint opened in 1792.", record=_old_record())
        await _claim(db_session, "Coins were hand struck.", record=_old_record())
        await _claim(db_session, "Silver is a metal.")  # no record: legacy
        await db_session.commit()

        census = await modality_census(db_session)
        assert census.total == 4
        assert census.by_state == {
            "current": 1,
            "stale": 3,
            "operator": 0,
            "unclassified": 0,
        }
        assert census.stale_by_classifier == {
            LEGACY_MODALITY_CLASSIFIER: 1,
            "prompt.modality@0000000000000000": 2,
        }

        findings = await _report_modality_classifiers(db_session)
        assert {f.finding_type for f in findings} == {"MODALITY_CLASSIFIER_STALE"}
        assert len(findings) == 2
        old = next(f for f in findings if "0000000000000000" in f.detail)
        assert old.severity == "INFO"
        assert old.detail.startswith("2 ACTIVE particle(s)")
        assert "particles modality" in (old.recommended_action or "")

    @pytest.mark.asyncio
    async def test_a_current_store_reports_nothing(self, db_session: AsyncSession) -> None:
        from particles.operations.lint.modality import _report_modality_classifiers

        await _claim(db_session, "The mint opened in 1792.", record=_current_record())
        await db_session.commit()
        assert await _report_modality_classifiers(db_session) == []

    @pytest.mark.asyncio
    async def test_particle_stamp_reads_one_claim(self, db_session: AsyncSession) -> None:
        from particles.operations.modality import particle_stamp

        p = await _claim(db_session, "The mint opened in 1792.", record=_old_record())
        await db_session.commit()
        stamped = await particle_stamp(db_session, p.id)
        assert stamped is not None
        stamp, state = stamped
        assert stamp == ModalityStamp("prompt.modality@0000000000000000", "anthropic:claude-x")
        assert state == StampState.STALE


# ---------------------------------------------------------------------------
# The operator verdict writes an event and changes §6.6 eligibility (§3, §4)
# ---------------------------------------------------------------------------


class TestReclassify:
    @pytest.mark.asyncio
    async def test_writes_the_event_and_changes_the_next_pairs_eligibility(
        self, db_session: AsyncSession
    ) -> None:
        from particles.core.conflict_resolution import ConflictVerdict, resolve_conflict
        from particles.core.schema import is_truth_apt
        from particles.ingest.pipeline import _find_conflict
        from particles.operations.consolidation import _delta_scope_ids
        from particles.operations.modality import reclassify_particle
        from particles.store.event_store import (
            EventRefKind,
            OperatorEventType,
            list_events,
        )
        from particles.store.particle_store import get_particle

        vec = [1.0, 0.0, 0.0]
        existing = await _claim(
            db_session, "Python is the best language.", record=_old_record(), embedding=vec
        )
        await db_session.commit()
        candidate = Particle(
            content="Rust is the best language.",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
        )
        emb = np.asarray(vec, dtype=np.float32)

        # Before: the stored default says adjudicable, so the pair is a §6.6 pair.
        before = await get_particle(db_session, existing.id)
        assert before is not None and is_truth_apt(before)
        assert _find_conflict(emb, [before], [emb]) is not None
        assert resolve_conflict(before, candidate) != ConflictVerdict.CORROBORATES

        watermark = datetime.now(UTC) - timedelta(seconds=1)
        result = await reclassify_particle(
            db_session,
            existing.id,
            EV,
            reason="a preference, not a fact",
            actor="cli:particle-reclassify",
        )
        await db_session.commit()
        assert (result.prior, result.modality) == (F, EV)
        assert result.prior_classifier == "prompt.modality@0000000000000000"

        # After: the next pair skips it, and the ladder abstains.
        after = await get_particle(db_session, existing.id)
        assert after is not None and not is_truth_apt(after)
        assert _find_conflict(emb, [after], [emb]) is None
        assert resolve_conflict(after, candidate) == ConflictVerdict.CORROBORATES

        # Content, confidence and status never move.
        assert (after.content, after.confidence.value, after.status) == (
            before.content,
            before.confidence.value,
            Status.ACTIVE,
        )

        events = await list_events(
            db_session,
            ref_kind=EventRefKind.PARTICLE,
            ref_id=existing.id,
            event_type=OperatorEventType.MODALITY_RECLASSIFIED,
        )
        assert len(events) == 1
        event = events[0]
        assert event.event_id == result.event_id
        assert event.actor == "cli:particle-reclassify"
        assert event.reason == "a preference, not a fact"
        assert event.payload == {
            "from": "FALSIFIABLE",
            "to": "EVALUATIVE",
            "prior_classifier": "prompt.modality@0000000000000000",
        }

        # The queue: the next consolidation run's delta scope selects it.
        assert existing.id in await _delta_scope_ids(db_session, watermark)

    @pytest.mark.asyncio
    async def test_a_verdict_needs_a_reason(self, db_session: AsyncSession) -> None:
        from particles.operations.modality import reclassify_particle

        p = await _claim(db_session, "x")
        with pytest.raises(ValueError, match="reason"):
            await reclassify_particle(db_session, p.id, EV, reason=" ", actor="t")


# ---------------------------------------------------------------------------
# Regeneration (§2)
# ---------------------------------------------------------------------------


class TestRegenerate:
    @pytest.mark.asyncio
    async def test_reclassifies_stale_claims_and_spares_operator_verdicts(
        self, db_session: AsyncSession
    ) -> None:
        from particles.operations.modality import (
            REGENERATE_ACTOR,
            reclassify_particle,
            regenerate_modality,
        )
        from particles.store.event_store import OperatorEventType, list_events
        from particles.store.particle_store import ParticleRow

        stale_changes = await _claim(
            db_session, "Tabs are nicer than spaces.", record=_old_record()
        )
        stale_confirms = await _claim(db_session, "The mint opened in 1792.", record=_old_record())
        stale_fails = await _claim(db_session, "Gibberish.", record=_old_record())
        current = await _claim(db_session, "Silver is a metal.", record=_current_record())
        pinned = await _claim(db_session, "I loved that coin.", record=_old_record())
        await reclassify_particle(db_session, pinned.id, F, reason="pinned", actor="t")
        await db_session.commit()

        asked: list[str] = []

        async def classify(content: str) -> ModalityVerdict | None:
            asked.append(content)
            if content == "Gibberish.":
                return None
            modality = EV if content.startswith("Tabs") else F
            return ModalityVerdict(modality, "anthropic:claude-y", regeneration_classifier())

        dry = await regenerate_modality(db_session, dry_run=True, classify=classify)
        assert dry.backlog == 3 and asked == []

        summary = await regenerate_modality(db_session, rate_limit_per_minute=0, classify=classify)
        assert sorted(asked) == sorted(
            [stale_changes.content, stale_confirms.content, stale_fails.content]
        )
        assert (summary.scope, summary.changed, summary.confirmed, summary.failed) == (3, 1, 1, 1)
        assert summary.remaining == 1  # the failed one stays stale

        rows = {
            r.id: r
            for r in (
                await db_session.execute(
                    select(ParticleRow).where(
                        ParticleRow.id.in_(
                            [stale_changes.id, stale_fails.id, pinned.id, current.id]
                        )
                    )
                )
            ).scalars()
        }
        assert rows[stale_changes.id].assertion_modality == "EVALUATIVE"
        assert rows[stale_changes.id].modality_classifier == regeneration_classifier()
        assert rows[stale_changes.id].modality_classifier_model == "anthropic:claude-y"
        assert rows[stale_fails.id].modality_classifier is None  # nothing written
        assert rows[pinned.id].modality_classifier == OPERATOR_CLASSIFIER
        assert rows[current.id].modality_classifier is None

        events = await list_events(db_session, event_type=OperatorEventType.MODALITY_RECLASSIFIED)
        batch = [e for e in events if e.actor == REGENERATE_ACTOR]
        assert len(batch) == 1
        assert batch[0].payload is not None
        assert batch[0].payload["changes"] == [
            {"particle_id": stale_changes.id, "from": "FALSIFIABLE", "to": "EVALUATIVE"}
        ]

    @pytest.mark.asyncio
    async def test_a_grant_is_queued_not_written(self, db_session: AsyncSession) -> None:
        # Review Q1: a flip to FALSIFIABLE would let the write path arbitrate the
        # claim, so regeneration queues it for an operator instead of writing it.
        from particles.operations.lint.modality import _report_pending_modality_grants
        from particles.operations.modality import (
            pending_grants,
            reclassify_particle,
            regenerate_modality,
        )
        from particles.store.particle_store import ParticleRow

        opinion = await _claim(db_session, "The grade is MS-65.", record=_old_record(), modality=EV)
        await db_session.commit()

        async def classify(content: str) -> ModalityVerdict | None:
            return ModalityVerdict(F, "anthropic:claude-y", regeneration_classifier())

        summary = await regenerate_modality(db_session, rate_limit_per_minute=0, classify=classify)
        assert (summary.changed, summary.queued) == (0, 1)
        row = await db_session.get(ParticleRow, opinion.id)
        assert row is not None
        assert (row.assertion_modality, row.modality_classifier) == ("EVALUATIVE", None)

        grants = await pending_grants(db_session)
        assert list(grants) == [opinion.id]
        findings = await _report_pending_modality_grants(db_session)
        assert [f.finding_type for f in findings] == ["MODALITY_GRANT_PENDING"]
        assert "particle reclassify" in (findings[0].recommended_action or "")

        # A second run does not pay to ask again.
        again = await regenerate_modality(db_session, rate_limit_per_minute=0, classify=classify)
        assert (again.backlog, again.scope) == (0, 0)

        # The operator's verdict settles it.
        await reclassify_particle(db_session, opinion.id, F, reason="graded", actor="t")
        await db_session.commit()
        assert await pending_grants(db_session) == {}

    @pytest.mark.asyncio
    async def test_journal_claims_are_out_of_scope(self, db_session: AsyncSession) -> None:
        # Review Q2: the general rule must not replace the journal prompt's verdict.
        from particles.operations.modality import regenerate_modality
        from particles.store.particle_store import insert_particle

        entry_id, snapshot_id = await _snapshot(
            db_session,
            ComponentRecord(
                extractor="journal-extractor",
                exercised={"prompt.journal.rules": "0000000000000000"},
            ),
        )
        journal_claim = Particle(
            content="I think the new role suits me.",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            assertion_modality=EV,
            extractor_ref=ExtractorRef(name="journal-extractor", version="0.5.0"),
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=entry_id,
                    snapshot_id=snapshot_id,
                )
            ],
        )
        await insert_particle(db_session, journal_claim)
        await db_session.commit()

        async def classify(content: str) -> ModalityVerdict | None:  # pragma: no cover
            raise AssertionError("a journal claim must not be classified")

        dry = await regenerate_modality(db_session, dry_run=True, classify=classify)
        assert dry.backlog == 0
        assert dry.excluded_journal == 1
        assert dry.census["by_state"]["stale"] == 1  # still reported stale

    @pytest.mark.asyncio
    async def test_an_operator_verdict_made_mid_run_survives(
        self, db_session: AsyncSession
    ) -> None:
        # Review Q5: the scope is computed once; the writer re-checks the row.
        from particles.operations.modality import reclassify_particle, regenerate_modality
        from particles.store.particle_store import ParticleRow

        claim = await _claim(db_session, "Python is the best language.", record=_old_record())
        await db_session.commit()

        async def classify(content: str) -> ModalityVerdict | None:
            # The operator rules while this run is waiting on the model.
            await reclassify_particle(db_session, claim.id, EX, reason="mid-run", actor="t")
            return ModalityVerdict(EV, "anthropic:claude-y", regeneration_classifier())

        summary = await regenerate_modality(db_session, rate_limit_per_minute=0, classify=classify)
        assert (summary.changed, summary.skipped_operator) == (0, 1)
        row = await db_session.get(ParticleRow, claim.id)
        assert row is not None
        assert (row.assertion_modality, row.modality_classifier) == ("EXPERIENTIAL", "operator")

    @pytest.mark.asyncio
    async def test_a_confirming_restamp_is_not_queued_for_re_pairing(
        self, db_session: AsyncSession
    ) -> None:
        # Review Q6: only a changed value enters the delta scope.
        from particles.operations.consolidation import _delta_scope_ids
        from particles.operations.modality import regenerate_modality

        confirmed = await _claim(db_session, "The mint opened in 1792.", record=_old_record())
        changed = await _claim(db_session, "Tabs are nicer than spaces.", record=_old_record())
        await db_session.commit()
        watermark = datetime.now(UTC)

        async def classify(content: str) -> ModalityVerdict | None:
            modality = EV if content.startswith("Tabs") else F
            return ModalityVerdict(modality, "anthropic:claude-y", regeneration_classifier())

        await regenerate_modality(db_session, rate_limit_per_minute=0, classify=classify)
        scope = await _delta_scope_ids(db_session, watermark)
        assert changed.id in scope
        assert confirmed.id not in scope


class TestStampTravelsWithACopy:
    @pytest.mark.asyncio
    async def test_quarantine_promotion_keeps_an_operator_pin(
        self, db_session: AsyncSession
    ) -> None:
        # Review Q7: a minted copy carries the value, so it carries the stamp;
        # otherwise an operator verdict silently becomes a regenerable default.
        from particles.core.status import StatusReason
        from particles.operations._quarantine import promote_quarantined
        from particles.operations.modality import particle_stamp, reclassify_particle
        from particles.store.particle_store import get_particle, update_particle_status

        loser = await _claim(db_session, "I felt the move was right.", record=_old_record())
        await reclassify_particle(db_session, loser.id, EX, reason="first person", actor="t")
        await update_particle_status(
            db_session, loser.id, Status.PROVENANCE_STALE, StatusReason.CONFLICT_PENDING
        )
        quarantined = await get_particle(db_session, loser.id)
        assert quarantined is not None
        minted = await promote_quarantined(db_session, quarantined)
        await db_session.commit()

        stamped = await particle_stamp(db_session, minted.id)
        assert stamped is not None
        assert stamped[0].classifier == OPERATOR_CLASSIFIER
        assert stamped[1] == StampState.OPERATOR
        assert minted.assertion_modality == EX


# ---------------------------------------------------------------------------
# A lens override changes the contested rendering, not stored state (§5)
# ---------------------------------------------------------------------------


class TestLensOverride:
    @pytest.mark.asyncio
    async def test_changes_the_badge_without_changing_stored_state(
        self, db_session: AsyncSession
    ) -> None:
        from particles.operations.query.contested import compute_contested_badges
        from particles.store.event_store import OperatorEventType, list_events
        from particles.store.lens_store import adopt_lens, materialise_lens
        from particles.store.particle_store import ParticleRow, insert_particle

        a = await _claim(
            db_session, "A Particle MUST carry a confidence.", record=_current_record()
        )
        b = await _claim(
            db_session, "A Particle need not carry a confidence.", record=_current_record()
        )
        record = Particle(
            content="Conflict between beliefs.",
            confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="lint",
            status=Status.INCONSISTENCY,
            provenance=[
                ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=a.id),
                ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=b.id),
            ],
        )
        await insert_particle(db_session, record)
        await db_session.commit()

        before = await compute_contested_badges(db_session, [a, b], readings=[])
        assert [x.bases if x else None for x in before] == [["inconsistency"]] * 2

        async def stored_state() -> list[tuple[str, str | None, object]]:
            rows = (
                await db_session.execute(
                    select(ParticleRow).where(ParticleRow.id.in_([a.id, b.id, record.id]))
                )
            ).scalars()
            return sorted(
                (r.id, r.assertion_modality, r.modality_classifier, r.status) for r in rows
            )

        state_before = await stored_state()
        events_before = await list_events(db_session, limit=500)

        lens = TrustLensDefinition(
            name="spec-readers",
            version=1,
            modality_rules=[TrustLensModalityRule(scope="particle", pattern=a.id, modality=CO)],
        )
        await materialise_lens(db_session, lens)
        await adopt_lens(db_session, lens.name)
        await db_session.commit()

        after = await compute_contested_badges(db_session, [a, b], readings=[])
        assert after == [None, None]

        # Stored state is untouched: same values, same stamps, same statuses,
        # and the only new events are the lens's own materialise / adopt.
        assert await stored_state() == state_before
        new_events = (await list_events(db_session, limit=500))[: -len(events_before) or None]
        assert {e.event_type for e in new_events} == {OperatorEventType.TRUST_CHANGED}

    @pytest.mark.asyncio
    async def test_an_operator_verdict_clears_the_badge_with_no_lens(
        self, db_session: AsyncSession
    ) -> None:
        # Review Q4: the filter runs on stored values when no lens is adopted.
        from particles.operations.modality import reclassify_particle
        from particles.operations.query.contested import compute_contested_badges
        from particles.store.particle_store import get_particle, insert_particle

        a = await _claim(db_session, "Spaces are better than tabs.", record=_current_record())
        b = await _claim(db_session, "Tabs are better than spaces.", record=_current_record())
        await insert_particle(
            db_session,
            Particle(
                content="Conflict between beliefs.",
                confidence=Confidence(
                    value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="lint",
                status=Status.INCONSISTENCY,
                provenance=[
                    ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=a.id),
                    ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=b.id),
                ],
            ),
        )
        await db_session.commit()
        before = await compute_contested_badges(db_session, [a, b], readings=[])
        assert [x.bases if x else None for x in before] == [["inconsistency"]] * 2

        await reclassify_particle(db_session, a.id, EV, reason="a preference", actor="t")
        await db_session.commit()
        reloaded = [await get_particle(db_session, a.id), await get_particle(db_session, b.id)]
        targets = [p for p in reloaded if p is not None]
        after = await compute_contested_badges(db_session, targets, readings=[])
        assert after == [None, None]

    @pytest.mark.asyncio
    async def test_lint_queues_a_lens_divergent_claim(self, db_session: AsyncSession) -> None:
        from particles.operations.lint.modality import _check_modality_lens_divergence
        from particles.store.lens_store import adopt_lens, materialise_lens

        assert await _check_modality_lens_divergence(db_session) == []
        opinion = await _claim(
            db_session, "MS-65 is the right grade.", record=_current_record(), modality=EV
        )
        fact = await _claim(db_session, "The coin weighs 26.73 g.", record=_current_record())
        lens = TrustLensDefinition(
            name="graders",
            version=1,
            modality_rules=[
                TrustLensModalityRule(scope="particle", pattern=opinion.id, modality=F, when=EV)
            ],
        )
        await materialise_lens(db_session, lens)
        await adopt_lens(db_session, lens.name)
        await db_session.commit()

        findings = await _check_modality_lens_divergence(db_session)
        assert [f.particle_id for f in findings] == [opinion.id]
        assert findings[0].finding_type == "MODALITY_LENS_DIVERGENCE"
        assert "lens:graders" in findings[0].detail
        assert "particle reclassify" in (findings[0].recommended_action or "")
        assert fact.id not in {f.particle_id for f in findings}

    @pytest.mark.asyncio
    async def test_an_operator_verdict_wins_over_a_lens(self, db_session: AsyncSession) -> None:
        from particles.operations.modality import load_modality_lens, reclassify_particle
        from particles.store.lens_store import adopt_lens, materialise_lens
        from particles.store.particle_store import get_particle

        p = await _claim(db_session, "I felt anxious.", record=_current_record())
        await reclassify_particle(db_session, p.id, EX, reason="first person", actor="t")
        lens = TrustLensDefinition(
            name="everything-counts",
            version=1,
            modality_rules=[TrustLensModalityRule(scope="particle", pattern=p.id, modality=F)],
        )
        await materialise_lens(db_session, lens)
        await adopt_lens(db_session, lens.name)
        await db_session.commit()

        loaded = await get_particle(db_session, p.id)
        assert loaded is not None
        reading = (await (await load_modality_lens(db_session)).readings(db_session, [loaded]))[
            p.id
        ]
        assert reading == ModalityReading(EX, "operator")


class TestLensRoundTrip:
    @pytest.mark.asyncio
    async def test_modality_rules_survive_materialisation(self, db_session: AsyncSession) -> None:
        from particles.store.lens_store import get_lens, materialise_lens

        rules = [
            TrustLensModalityRule(scope="particle", pattern="p-1", modality=EV),
            TrustLensModalityRule(scope="subject", pattern="s-1", modality=F, when=EV),
            TrustLensModalityRule(scope="url_pattern", pattern=r"pcgs\.com", modality=F),
            TrustLensModalityRule(scope="source_type", pattern="LOCAL_MARKDOWN", modality=CO),
        ]
        lens = TrustLensDefinition(name="m", version=1, modality_rules=rules)
        await materialise_lens(db_session, lens)
        loaded = await get_lens(db_session, "m")
        assert loaded is not None
        key = lambda r: (r.scope, r.pattern)  # noqa: E731
        assert sorted(loaded.modality_rules, key=key) == sorted(rules, key=key)
