"""Tests for the operator event log."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from particles.api.cli import app
from particles.db import session_scope
from particles.store.event_store import (
    EventRefKind,
    OperatorEventRow,
    OperatorEventType,
    get_event,
    list_events,
    record_event,
    ref_filter,
)


class TestRecordEvent:
    @pytest.mark.asyncio
    async def test_inserts_header_and_refs_atomically(self, db_session: AsyncSession) -> None:
        event = await record_event(
            db_session,
            actor="corpus-retract",
            event_type=OperatorEventType.SOURCE_RETRACTED,
            reason="publisher issued a correction",
            refs=[
                (EventRefKind.CORPUS_ENTRY, "entry-1"),
                (EventRefKind.PARTICLE, "p-1"),
                (EventRefKind.PARTICLE, "p-2"),
            ],
            payload={"skipped": {"SUPERSEDED": 1}},
        )
        await db_session.commit()

        fetched = await get_event(db_session, event.event_id)
        assert fetched is not None
        assert fetched.event_type is OperatorEventType.SOURCE_RETRACTED
        assert fetched.actor == "corpus-retract"
        assert fetched.reason == "publisher issued a correction"
        assert fetched.payload == {"skipped": {"SUPERSEDED": 1}}
        assert {(r.ref_kind, r.ref_id) for r in fetched.refs} == {
            (EventRefKind.CORPUS_ENTRY, "entry-1"),
            (EventRefKind.PARTICLE, "p-1"),
            (EventRefKind.PARTICLE, "p-2"),
        }

    @pytest.mark.asyncio
    async def test_dedupes_repeated_refs(self, db_session: AsyncSession) -> None:
        event = await record_event(
            db_session,
            actor="links-add",
            event_type=OperatorEventType.RELATION_ADDED,
            refs=[
                (EventRefKind.PARTICLE, "p-1"),
                (EventRefKind.PARTICLE, "p-1"),
            ],
        )
        await db_session.commit()
        fetched = await get_event(db_session, event.event_id)
        assert fetched is not None
        assert len(fetched.refs) == 1

    @pytest.mark.asyncio
    async def test_append_only_no_update_helper(self, db_session: AsyncSession) -> None:
        # The store exposes only record/get/list — there is no update or delete
        # helper. Assert the surface stays append-only.
        import particles.store.event_store as es

        assert not hasattr(es, "update_event")
        assert not hasattr(es, "delete_event")


class TestListEvents:
    @pytest.mark.asyncio
    async def test_filter_by_ref(self, db_session: AsyncSession) -> None:
        await record_event(
            db_session,
            actor="subjects-merge",
            event_type=OperatorEventType.SUBJECTS_MERGED,
            refs=[(EventRefKind.SUBJECT, "s-1"), (EventRefKind.SUBJECT, "s-2")],
        )
        await record_event(
            db_session,
            actor="subjects-alias",
            event_type=OperatorEventType.SUBJECT_ALIASED,
            refs=[(EventRefKind.SUBJECT, "s-3")],
        )
        await db_session.commit()

        only_s1 = await list_events(db_session, ref_kind=EventRefKind.SUBJECT, ref_id="s-1")
        assert len(only_s1) == 1
        assert only_s1[0].event_type is OperatorEventType.SUBJECTS_MERGED

    @pytest.mark.asyncio
    async def test_filter_by_type(self, db_session: AsyncSession) -> None:
        await record_event(
            db_session,
            actor="particle-tag",
            event_type=OperatorEventType.PARTICLE_TAGGED,
            refs=[(EventRefKind.PARTICLE, "p-9")],
        )
        await record_event(
            db_session,
            actor="particle-untag",
            event_type=OperatorEventType.PARTICLE_UNTAGGED,
            refs=[(EventRefKind.PARTICLE, "p-9")],
        )
        await db_session.commit()

        tagged = await list_events(db_session, event_type=OperatorEventType.PARTICLE_TAGGED)
        assert len(tagged) == 1
        assert tagged[0].event_type is OperatorEventType.PARTICLE_TAGGED

        # Distinct polar types: filtering by one never returns the other.
        all_for_p9 = await list_events(db_session, ref_kind=EventRefKind.PARTICLE, ref_id="p-9")
        assert {e.event_type for e in all_for_p9} == {
            OperatorEventType.PARTICLE_TAGGED,
            OperatorEventType.PARTICLE_UNTAGGED,
        }

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, db_session: AsyncSession) -> None:
        for i in range(5):
            await record_event(
                db_session,
                actor="links-add",
                event_type=OperatorEventType.RELATION_ADDED,
                refs=[(EventRefKind.PARTICLE, f"p-{i}")],
            )
        await db_session.commit()
        assert len(await list_events(db_session, limit=3)) == 3

    @pytest.mark.asyncio
    async def test_get_event_missing_returns_none(self, db_session: AsyncSession) -> None:
        assert await get_event(db_session, "does-not-exist") is None

    @pytest.mark.asyncio
    async def test_rows_persisted_in_both_tables(self, db_session: AsyncSession) -> None:
        await record_event(
            db_session,
            actor="trust-set",
            event_type=OperatorEventType.TRUST_CHANGED,
            reason="manual override",
            refs=[(EventRefKind.TRUST_STATEMENT, "ts-1")],
            payload={"old_rank": 0.3, "new_rank": 0.7},
        )
        await db_session.commit()
        headers = (await db_session.execute(select(OperatorEventRow))).scalars().all()
        assert len(headers) == 1
        assert headers[0].payload == {"old_rank": 0.3, "new_rank": 0.7}


class TestRefFilter:
    def test_none_set_returns_empty(self) -> None:
        assert ref_filter() == (None, None)

    def test_single_set_returns_pair(self) -> None:
        assert ref_filter(subject="s-1") == (EventRefKind.SUBJECT, "s-1")

    def test_more_than_one_raises(self) -> None:
        with pytest.raises(ValueError):
            ref_filter(particle="p-1", subject="s-1")


def _seed(
    events: list[tuple[OperatorEventType, list[tuple[EventRefKind, str]], str | None]],
) -> None:
    """Seed events into the (file-based cli_db) DB via a fresh event loop."""

    async def _go() -> None:
        async with session_scope() as session:
            for etype, refs, reason in events:
                await record_event(
                    session, actor="test", event_type=etype, reason=reason, refs=refs
                )
            await session.commit()

    asyncio.run(_go())


class TestEventsCLI:
    def test_list_empty(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["events", "list"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "No operator events." in result.stdout

    def test_list_shows_seeded_event(self, runner: CliRunner, cli_db: Path) -> None:
        _seed(
            [
                (
                    OperatorEventType.SOURCE_RETRACTED,
                    [(EventRefKind.CORPUS_ENTRY, "entry-abcdef12")],
                    "publisher correction",
                )
            ]
        )
        result = runner.invoke(app, ["events", "list"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "SOURCE_RETRACTED" in result.stdout
        assert "publisher correction" in result.stdout

    def test_list_conflicting_ref_flags_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["events", "list", "--particle", "p", "--subject", "s"])
        assert result.exit_code == 1

    def test_list_bad_type_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["events", "list", "--type", "NOPE"])
        assert result.exit_code == 1

    def test_show_missing_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["events", "show", "nope"])
        assert result.exit_code == 1


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestSubjectConsumers:
    """wiring of the subjects verbs (store-layer)."""

    @staticmethod
    async def _subject(
        session: AsyncSession, name: str, *, external_ids: list | None = None
    ) -> str:
        import uuid

        from particles.core.schema import Subject
        from particles.store.subject_store import insert_subject

        subj = Subject(
            id=str(uuid.uuid4()),
            canonical_name=name,
            external_ids=external_ids or [],
            asserted_by="t",
        )
        await insert_subject(session, subj)
        return subj.id

    @pytest.mark.asyncio
    async def test_alias_logs_event(self, db_session: AsyncSession) -> None:
        from particles.store.subject_store import add_aliases

        sid = await self._subject(db_session, "Acme")
        await add_aliases(db_session, sid, ["ACME Corp"])
        await db_session.commit()

        events = await list_events(db_session, ref_kind=EventRefKind.SUBJECT, ref_id=sid)
        assert [e.event_type for e in events] == [OperatorEventType.SUBJECT_ALIASED]
        assert events[0].payload == {"added": ["ACME Corp"]}

    @pytest.mark.asyncio
    async def test_alias_noop_logs_nothing(self, db_session: AsyncSession) -> None:
        from particles.store.subject_store import add_aliases

        sid = await self._subject(db_session, "Acme")
        await add_aliases(db_session, sid, ["acme"])  # already the canonical name
        await db_session.commit()
        assert await list_events(db_session, ref_kind=EventRefKind.SUBJECT, ref_id=sid) == []

    @pytest.mark.asyncio
    async def test_reclassify_logs_event(self, db_session: AsyncSession) -> None:
        from particles.store.subject_store import reclassify_subject

        sid = await self._subject(db_session, "1 Pfennig (1948-1950) GDR")
        updated, previous = await reclassify_subject(db_session, sid, "nmo:NumismaticObject")
        await db_session.commit()

        assert previous is None
        assert updated.subject_class == "nmo:NumismaticObject"
        events = await list_events(db_session, ref_kind=EventRefKind.SUBJECT, ref_id=sid)
        assert [e.event_type for e in events] == [OperatorEventType.SUBJECT_RECLASSIFIED]
        assert events[0].payload == {
            "previous_class": None,
            "new_class": "nmo:NumismaticObject",
        }

    @pytest.mark.asyncio
    async def test_reclassify_noop_logs_nothing(self, db_session: AsyncSession) -> None:
        from particles.store.subject_store import reclassify_subject

        sid = await self._subject(db_session, "Copper")
        await reclassify_subject(db_session, sid, "nmo:Material")
        _, previous = await reclassify_subject(db_session, sid, "nmo:Material")  # same value
        await db_session.commit()

        assert previous == "nmo:Material"
        # Only the first (state-changing) call logged.
        events = await list_events(db_session, ref_kind=EventRefKind.SUBJECT, ref_id=sid)
        assert [e.event_type for e in events] == [OperatorEventType.SUBJECT_RECLASSIFIED]

    @pytest.mark.asyncio
    async def test_confirm_and_unlink_log_distinct_types(self, db_session: AsyncSession) -> None:
        from particles.core.schema import ExternalRef
        from particles.store.subject_store import (
            remove_external_ref,
            set_external_ref_confidence,
        )

        sid = await self._subject(
            db_session,
            "Applied Optoelectronics",
            external_ids=[ExternalRef(namespace="wikidata", id="Q30297735", confidence=0.6)],
        )
        await set_external_ref_confidence(db_session, sid, "wikidata", "Q30297735", 1.0)
        await remove_external_ref(db_session, sid, "wikidata", "Q30297735")
        await db_session.commit()

        events = await list_events(db_session, ref_kind=EventRefKind.SUBJECT, ref_id=sid)
        assert {e.event_type for e in events} == {
            OperatorEventType.SUBJECT_LINK_CONFIRMED,
            OperatorEventType.SUBJECT_LINK_REMOVED,
        }

    @pytest.mark.asyncio
    async def test_merge_logs_event_with_both_subjects(self, db_session: AsyncSession) -> None:
        from particles.store.subject_store import merge_subjects

        src = await self._subject(db_session, "AAOI dup")
        tgt = await self._subject(db_session, "AAOI")
        await merge_subjects(db_session, src, tgt)
        await db_session.commit()

        events = await list_events(db_session, event_type=OperatorEventType.SUBJECTS_MERGED)
        assert len(events) == 1
        refs = {(r.ref_kind, r.ref_id) for r in events[0].refs}
        assert (EventRefKind.SUBJECT, src) in refs
        assert (EventRefKind.SUBJECT, tgt) in refs

    @pytest.mark.asyncio
    async def test_split_logs_event_with_moved_particle(self, db_session: AsyncSession) -> None:
        import uuid
        from datetime import UTC, datetime

        from particles.core.schema import Confidence, Particle, UncertaintyNature
        from particles.core.scoring.confidence import CalibrationSource
        from particles.core.status import Status
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import split_subject

        src = await self._subject(db_session, "Source")
        new = await self._subject(db_session, "New")
        pid = str(uuid.uuid4())
        await insert_particle(
            db_session,
            Particle(
                id=pid,
                content="A claim.",
                confidence=Confidence(
                    value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="t",
                asserted_at=datetime.now(UTC),
                status=Status.ACTIVE,
                subject_ids=[src],
            ),
        )
        relinked, _ = await split_subject(
            db_session, source_id=src, new_subject_id=new, particle_ids=[pid]
        )
        await db_session.commit()
        assert relinked == [pid]

        events = await list_events(db_session, event_type=OperatorEventType.SUBJECTS_SPLIT)
        assert len(events) == 1
        refs = {(r.ref_kind, r.ref_id) for r in events[0].refs}
        assert (EventRefKind.PARTICLE, pid) in refs
        assert (EventRefKind.SUBJECT, new) in refs


class TestTrustConsumers:
    """wiring of the trust verbs."""

    @pytest.mark.asyncio
    async def test_trust_set_rule_logs(self, db_session: AsyncSession) -> None:
        from particles.store.trust_store import upsert_trust_rule

        await upsert_trust_rule(
            db_session,
            scope="domain",
            pattern="en.wikipedia.org",
            score=0.8,
            modifier=None,
            rationale="reliable",
        )
        await db_session.commit()
        events = await list_events(db_session, event_type=OperatorEventType.TRUST_CHANGED)
        assert len(events) == 1
        assert events[0].payload is not None
        assert events[0].payload["kind"] == "rule"
        assert events[0].reason == "reliable"

    @pytest.mark.asyncio
    async def test_statement_set_logs_and_keeps_basis(self, db_session: AsyncSession) -> None:
        import uuid

        from particles.core.schema import (
            PolicyProvenance,
            SourceRef,
            SourceRefType,
            SourceTrustStatement,
        )
        from particles.operations.trust import set_trust_statement
        from particles.store.trust_store import get_trust_statements_for_domain

        sid = str(uuid.uuid4())
        stmt = SourceTrustStatement(
            statement_id=sid,
            domain="numismatics",
            source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="WEB_PAGE"),
            trust_rank=0.7,
            policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
            asserted_by="operator",
            basis="trusted dealer catalogue",
        )
        await set_trust_statement(db_session, stmt)
        await db_session.commit()

        events = await list_events(db_session, ref_kind=EventRefKind.TRUST_STATEMENT, ref_id=sid)
        assert len(events) == 1
        assert events[0].event_type is OperatorEventType.TRUST_CHANGED
        assert events[0].payload is not None
        assert events[0].payload["basis"] == "trusted dealer catalogue"
        # basis stays on the statement record (not relocated into the log)
        stored = await get_trust_statements_for_domain(db_session, "numismatics")
        assert stored[0].basis == "trusted dealer catalogue"

    @pytest.mark.asyncio
    async def test_insert_trust_statement_alone_logs_nothing(
        self, db_session: AsyncSession
    ) -> None:
        # The shared store helper (also used by review) must NOT log a
        # TRUST_CHANGED on its own — only the operator operation does.
        import uuid

        from particles.core.schema import (
            PolicyProvenance,
            SourceRef,
            SourceRefType,
            SourceTrustStatement,
        )
        from particles.store.trust_store import insert_trust_statement

        stmt = SourceTrustStatement(
            statement_id=str(uuid.uuid4()),
            domain="d",
            source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="WEB_PAGE"),
            trust_rank=0.5,
            policy_provenance=PolicyProvenance.REVIEWER_DERIVED,
            asserted_by="reviewer",
        )
        await insert_trust_statement(db_session, stmt)
        await db_session.commit()
        assert await list_events(db_session, event_type=OperatorEventType.TRUST_CHANGED) == []


class TestReviewConsumer:
    """wiring of the §9.6 review operation."""

    @staticmethod
    async def _active(session: AsyncSession, content: str) -> str:
        import uuid
        from datetime import UTC, datetime

        from particles.core.schema import (
            Confidence,
            Particle,
            ProvenanceRef,
            ProvenanceRefType,
            UncertaintyNature,
        )
        from particles.core.scoring.confidence import CalibrationSource
        from particles.core.status import Status
        from particles.store.particle_store import insert_particle

        pid = str(uuid.uuid4())
        await insert_particle(
            session,
            Particle(
                id=pid,
                content=content,
                confidence=Confidence(
                    value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="t",
                asserted_at=datetime.now(UTC),
                status=Status.ACTIVE,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1"
                    )
                ],
            ),
        )
        return pid

    @pytest.mark.asyncio
    async def test_resolve_logs_review_resolved_not_trust_changed(
        self, db_session: AsyncSession
    ) -> None:
        import uuid
        from datetime import UTC, datetime

        from particles.core.schema import (
            Confidence,
            Particle,
            ProvenanceRef,
            ProvenanceRefType,
            ResolutionAction,
            UncertaintyNature,
        )
        from particles.core.scoring.confidence import CalibrationSource
        from particles.core.status import Status
        from particles.operations.review import resolve
        from particles.store.particle_store import insert_particle

        a = await self._active(db_session, "claim A")
        b = await self._active(db_session, "claim B")
        inc_id = str(uuid.uuid4())
        await insert_particle(
            db_session,
            Particle(
                id=inc_id,
                content=f"INCONSISTENCY between {a} and {b}",
                confidence=Confidence(
                    value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="t",
                asserted_at=datetime.now(UTC),
                status=Status.INCONSISTENCY,
                provenance=[
                    ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=a),
                    ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=b),
                ],
            ),
        )
        await db_session.commit()

        await resolve(db_session, inc_id, ResolutionAction.PREFER_A, "reviewer-1")

        review_events = await list_events(db_session, event_type=OperatorEventType.REVIEW_RESOLVED)
        assert len(review_events) == 1
        refs = {(r.ref_kind, r.ref_id) for r in review_events[0].refs}
        assert (EventRefKind.PARTICLE, inc_id) in refs
        # review writes a SourceTrustStatement but via insert_trust_statement —
        # so it must NOT also log a standalone TRUST_CHANGED event.
        assert await list_events(db_session, event_type=OperatorEventType.TRUST_CHANGED) == []


class TestLinksConsumer:
    """wiring of `links add` / `links remove`."""

    @pytest.mark.asyncio
    async def test_manual_link_logs_added_then_removed(self, db_session: AsyncSession) -> None:
        from particles.core.schema import RelationCreatedBy, RelationType
        from particles.store.relation_store import create_relation, delete_relation

        await create_relation(
            db_session, "p-a", "p-b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.MANUAL_CLI
        )
        await delete_relation(db_session, "p-a", "p-b", RelationType.CO_EVIDENTIAL)
        await db_session.commit()

        types = [e.event_type for e in await list_events(db_session)]
        assert OperatorEventType.RELATION_ADDED in types
        assert OperatorEventType.RELATION_REMOVED in types

    @pytest.mark.asyncio
    async def test_llm_judge_link_logs_nothing(self, db_session: AsyncSession) -> None:
        # auto-suggest (LLM_JUDGE) is not an operator decision.
        from particles.core.schema import RelationCreatedBy, RelationType
        from particles.store.relation_store import create_relation

        await create_relation(
            db_session, "p-a", "p-b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.LLM_JUDGE
        )
        await db_session.commit()
        assert await list_events(db_session, event_type=OperatorEventType.RELATION_ADDED) == []


class TestParticleTagConsumer:
    """wiring of `particle tag` / `particle untag`."""

    @pytest.mark.asyncio
    async def test_tag_then_untag_log_distinct_types(self, db_session: AsyncSession) -> None:
        import uuid
        from datetime import UTC, datetime

        from particles.core.schema import Confidence, Particle, UncertaintyNature
        from particles.core.scoring.confidence import CalibrationSource
        from particles.core.status import Status
        from particles.store.particle_store import insert_particle
        from particles.store.taxonomy_store import add_particle_tags, remove_particle_tags

        pid = str(uuid.uuid4())
        await insert_particle(
            db_session,
            Particle(
                id=pid,
                content="A claim.",
                confidence=Confidence(
                    value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="t",
                asserted_at=datetime.now(UTC),
                status=Status.ACTIVE,
            ),
        )
        await add_particle_tags(db_session, pid, ["coins/germany"])
        await remove_particle_tags(db_session, pid, ["coins/germany"])
        await db_session.commit()

        events = await list_events(db_session, ref_kind=EventRefKind.PARTICLE, ref_id=pid)
        assert {e.event_type for e in events} == {
            OperatorEventType.PARTICLE_TAGGED,
            OperatorEventType.PARTICLE_UNTAGGED,
        }
