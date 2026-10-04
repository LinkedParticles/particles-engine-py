# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The vocabulary lifecycle: create, propose, confirm or decline,
align, export and import, adopt. Every revision is a new corpus deposit that the
vocabulary extractor materialises; a proposal lives only in the event log.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles import embeddings as ep
from particles.core.predicate_profile import PredicateRole, SlotKind
from particles.core.schema import (
    ClaimTerm,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    StructuredClaim,
    Subject,
    TermKind,
    UncertaintyNature,
)
from particles.core.vocabulary import Alignment, MatchStrength, Ruling
from particles.core.vocabulary_jsonld import dumps, loads
from particles.corpus.store import CorpusEntryRow
from particles.operations.vocabulary import (
    VocabularyError,
    adopt,
    align,
    confirm,
    create_document,
    decline,
    export_document,
    import_document,
    pending_proposals,
    propose,
)
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import insert_particle
from particles.store.subject_store import insert_subject
from particles.store.vocabulary_store import get_adopted_documents, get_document, list_versions

T0 = datetime(2026, 10, 2, tzinfo=UTC)
NS = "https://vocab.example.org/acme/"
RECORD = "artifact:record"
FILE = "artifact:file"

#: Fake encoder geometry: the forms that should cluster share a direction.
_VECTORS = {
    "contains": (1.0, 0.05, 0.0),
    "includes": (1.0, 0.10, 0.0),
    "holds": (1.0, 0.08, 0.0),
    "does not contain": (1.0, 0.06, 0.0),
    "moved to": (0.0, 1.0, 0.0),
    "relocated to": (0.0, 1.0, 0.05),
    "moved from": (0.0, 1.0, 0.02),
}


class _FakeEncoder(ep.EmbeddingModel):
    def encode(
        self, texts: list[str], convert_to_numpy: bool = True, normalize_embeddings: bool = True
    ) -> list[Any]:
        out = []
        for text in texts:
            v = np.array(_VECTORS.get(text, (0.0, 0.0, 1.0)), dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return out


@pytest.fixture
def encoder() -> Generator[None, None, None]:
    original = ep._embedding_model
    ep.set_embedding_model(_FakeEncoder())
    try:
        yield
    finally:
        ep.set_embedding_model(original)


async def _subject(session: AsyncSession, cls: str | None) -> str:
    subject = Subject(
        canonical_name=f"s-{uuid.uuid4().hex[:6]}", subject_class=cls, asserted_by="t"
    )
    await insert_subject(session, subject)
    return subject.id


async def _claims(session: AsyncSession, subject_id: str, predicate: str, n: int) -> None:
    for _ in range(n):
        await insert_particle(
            session,
            Particle(
                id=str(uuid.uuid4()),
                content=f"x {predicate} y",
                confidence=Confidence(value=0.9),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="t",
                asserted_at=T0,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id="s"
                    )
                ],
                structured_claim=StructuredClaim(
                    subject=ClaimTerm(kind=TermKind.TOKEN, value="x"),
                    predicate=ClaimTerm(kind=TermKind.TOKEN, value=predicate),
                    object=ClaimTerm(kind=TermKind.TOKEN, value="y"),
                    subject_id=subject_id,
                    structurizer_id="t",
                    structurizer_version="1",
                ),
            ),
        )


async def _store(session: AsyncSession) -> None:
    """A small store: records move between directories, files contain things."""
    record = await _subject(session, RECORD)
    file = await _subject(session, FILE)
    unclassed = await _subject(session, None)
    await _claims(session, record, "moved to", 6)
    await _claims(session, record, "relocated to", 2)
    await _claims(session, record, "moved from", 3)
    await _claims(session, file, "contains", 9)
    await _claims(session, file, "includes", 4)
    await _claims(session, file, "does not contain", 2)
    await _claims(session, unclassed, "contains", 5)


async def _create(session: AsyncSession) -> None:
    await create_document(session, name="acme", prefix="acme", namespace=NS, actor="t")


def _keys(report: Any, kind: str) -> dict[tuple[str, str], str]:
    return {(r["subject_class"], r["form"]): r["key"] for r in report.recorded if r["kind"] == kind}


class TestCreate:
    async def test_create_deposits_and_materialises_version_one(
        self, db_session: AsyncSession
    ) -> None:
        doc = await create_document(
            db_session, name="acme", prefix="acme", namespace=NS, publisher="Acme", actor="t"
        )
        assert (doc.version, doc.publisher) == (1, "Acme")
        assert doc.corpus_entry_id is not None
        entry = await db_session.get(CorpusEntryRow, doc.corpus_entry_id)
        assert entry is not None and entry.source_type == "VOCABULARY_DOCUMENT"

    async def test_a_name_is_created_once(self, db_session: AsyncSession) -> None:
        await _create(db_session)
        with pytest.raises(VocabularyError, match="already exists"):
            await _create(db_session)

    async def test_a_bad_namespace_is_an_operator_error(self, db_session: AsyncSession) -> None:
        with pytest.raises(VocabularyError, match="namespace"):
            await create_document(db_session, name="acme", prefix="acme", namespace="nope")


class TestPropose:
    async def test_proposals_come_from_classed_claims_only(
        self, db_session: AsyncSession, encoder: None
    ) -> None:
        await _store(db_session)
        await _create(db_session)
        report = await propose(db_session, "acme", actor="t")
        assert (report.claims, report.classed_claims) == (31, 26)
        aliases = {
            (r["subject_class"], r["form"]): r["aliases"]
            for r in report.recorded
            if r["kind"] == "alias"
        }
        # Records: "moved to" with "relocated to"; "moved from" is the other direction.
        assert aliases[(RECORD, "mov to")] == ["relocat to"]
        # Files: "contains" with "includes"; the negated form stays apart.
        assert aliases[(FILE, "contain")] == ["includ"]
        assert report.alias_coverage() == 4
        profiles = _keys(report, "profile")
        assert (RECORD, "mov to") in profiles and (FILE, "contain") in profiles
        recorded = await list_events(db_session, event_type=OperatorEventType.VOCABULARY_PROPOSED)
        assert len(recorded) == len(report.recorded)
        # The document is untouched until a reviewer confirms.
        doc = await get_document(db_session, "acme")
        assert doc is not None and (doc.version, doc.terms) == (1, [])

    async def test_a_dry_run_records_nothing(self, db_session: AsyncSession, encoder: None) -> None:
        await _store(db_session)
        await _create(db_session)
        report = await propose(db_session, "acme", dry_run=True)
        assert report.recorded
        assert await list_events(db_session, event_type=OperatorEventType.VOCABULARY_PROPOSED) == []

    async def test_a_key_is_proposed_once(self, db_session: AsyncSession, encoder: None) -> None:
        await _store(db_session)
        await _create(db_session)
        first = await propose(db_session, "acme")
        again = await propose(db_session, "acme")
        assert first.recorded and again.recorded == []

    async def test_without_an_encoder_only_profiles_are_proposed(
        self, db_session: AsyncSession, no_embedding_model: None
    ) -> None:
        await _store(db_session)
        await _create(db_session)
        report = await propose(db_session, "acme", dry_run=True)
        assert report.encoder_missing
        assert {r["kind"] for r in report.recorded} == {"profile"}

    async def test_filters_and_limit(self, db_session: AsyncSession, encoder: None) -> None:
        await _store(db_session)
        await _create(db_session)
        report = await propose(
            db_session,
            "acme",
            kinds=("profile",),
            subject_class=RECORD,
            form="was moved to",
            dry_run=True,
        )
        assert [(r["kind"], r["form"]) for r in report.recorded] == [("profile", "mov to")]
        limited = await propose(db_session, "acme", limit=1, dry_run=True)
        assert sum(r["kind"] == "profile" for r in limited.recorded) == 1

    async def test_an_unknown_document_is_refused(self, db_session: AsyncSession) -> None:
        with pytest.raises(VocabularyError, match="No vocabulary"):
            await propose(db_session, "nope")


class TestRulings:
    async def test_confirming_an_alias_and_a_profile_publishes_one_version(
        self, db_session: AsyncSession, encoder: None
    ) -> None:
        await _store(db_session)
        await _create(db_session)
        report = await propose(db_session, "acme")
        alias = _keys(report, "alias")[(RECORD, "mov to")]
        profile = _keys(report, "profile")[(RECORD, "mov to")]
        await confirm(db_session, [alias], actor="reviewer", reason="same move")
        doc = await confirm(
            db_session,
            [profile],
            kind=SlotKind.TIMELESS_SINGLE,
            past_forms=["moved from"],
            actor="reviewer",
        )
        assert doc.version == 3
        assert [v.version for v in await list_versions(db_session, "acme")] == [1, 2, 3]
        term = doc.terms[0]
        assert (term.form, term.subject_class, term.label) == ("mov to", RECORD, "moved to")
        assert [a.form for a in term.aliases] == ["relocat to", "mov from"]
        assert term.profile is not None
        assert term.profile.roles == {"mov from": PredicateRole.PAST}
        assert term.aliases[0].ruling.confirmed_by == "reviewer"
        assert term.aliases[0].ruling.evidence["reason"] == "same move"
        assert term.aliases[0].ruling.proposal_event_id is not None
        rulings = await list_events(db_session, event_type=OperatorEventType.VOCABULARY_RULED)
        assert {(e.payload or {})["version"] for e in rulings} == {2, 3}
        assert all((e.payload or {})["resolution"] == "confirmed" for e in rulings)

        # Not in force until adopted.
        assert await get_adopted_documents(db_session) == []
        await adopt(db_session, "acme", actor="t")
        [adopted] = await get_adopted_documents(db_session)
        assert adopted.name == "acme" and adopted.version == 3

    async def test_several_keys_confirm_into_one_version(
        self, db_session: AsyncSession, encoder: None
    ) -> None:
        await _store(db_session)
        await _create(db_session)
        report = await propose(db_session, "acme")
        keys = list(_keys(report, "alias").values())
        doc = await confirm(db_session, keys, actor="reviewer")
        assert doc.version == 2 and len(doc.terms) == 2

    async def test_a_profile_needs_a_kind(self, db_session: AsyncSession, encoder: None) -> None:
        await _store(db_session)
        await _create(db_session)
        report = await propose(db_session, "acme")
        key = _keys(report, "profile")[(FILE, "contain")]
        with pytest.raises(VocabularyError, match="--kind"):
            await confirm(db_session, [key])

    async def test_a_declined_proposal_is_never_confirmed_or_proposed_again(
        self, db_session: AsyncSession, encoder: None
    ) -> None:
        await _store(db_session)
        await _create(db_session)
        report = await propose(db_session, "acme")
        key = _keys(report, "alias")[(FILE, "contain")]
        await decline(db_session, [key], actor="reviewer", reason="different relations")
        with pytest.raises(VocabularyError, match="already declined"):
            await confirm(db_session, [key])
        assert key not in {e.payload["key"] for e in await pending_proposals(db_session, "acme")}
        assert key not in {r["key"] for r in (await propose(db_session, "acme")).recorded}
        doc = await get_document(db_session, "acme")
        assert doc is not None and doc.version == 1

    async def test_an_unknown_key_is_refused(self, db_session: AsyncSession) -> None:
        with pytest.raises(VocabularyError, match="No vocabulary proposal"):
            await confirm(db_session, ["vp-000000000000"])

    async def test_pending_proposals_are_ranked_by_claims(
        self, db_session: AsyncSession, encoder: None
    ) -> None:
        await _store(db_session)
        await _create(db_session)
        await propose(db_session, "acme")
        claims = [int(e.payload["claims"]) for e in await pending_proposals(db_session)]
        assert claims == sorted(claims, reverse=True)


class TestAlign:
    async def test_alignment_maps_outward_and_keeps_the_iri(
        self, db_session: AsyncSession, encoder: None
    ) -> None:
        await _store(db_session)
        await _create(db_session)
        report = await propose(db_session, "acme")
        doc = await confirm(
            db_session,
            [_keys(report, "profile")[(RECORD, "mov to")]],
            kind=SlotKind.ONE_AT_A_TIME,
        )
        local = doc.terms[0].local_name
        aligned = await align(
            db_session,
            "acme",
            f"acme:{local}",
            Alignment(
                target="wdt:P551",
                match=MatchStrength.EXACT,
                confidence=0.8,
                ruling=Ruling(confirmed_by="reviewer", evidence={"basis": "label match"}),
            ),
            actor="reviewer",
        )
        assert aligned.terms[0].local_name == local
        assert aligned.terms[0].alignments[0].target == "wdt:P551"
        ruled = await list_events(db_session, event_type=OperatorEventType.VOCABULARY_RULED)
        assert ruled[0].payload is not None and ruled[0].payload["resolution"] == "aligned"

    async def test_an_unknown_term_is_refused(self, db_session: AsyncSession) -> None:
        await _create(db_session)
        alignment = Alignment(
            target="wdt:P551",
            match=MatchStrength.EXACT,
            confidence=0.9,
            ruling=Ruling(confirmed_by="r"),
        )
        with pytest.raises(VocabularyError, match="no term"):
            await align(db_session, "acme", "nope", alignment)


class TestExportImport:
    async def test_export_round_trips_and_import_is_idempotent(
        self, db_session: AsyncSession, encoder: None
    ) -> None:
        await _store(db_session)
        await _create(db_session)
        report = await propose(db_session, "acme")
        await confirm(db_session, list(_keys(report, "alias").values()))
        text = await export_document(db_session, "acme")
        current = await get_document(db_session, "acme")
        assert current is not None and dumps(loads(text)) == dumps(current) == text
        assert await export_document(db_session, "acme", 1) != text
        # Importing a version the store already holds is a no-op.
        again = await import_document(db_session, text)
        assert again.version == 2

    async def test_import_of_another_operators_document(self, db_session: AsyncSession) -> None:
        theirs = loads(await self._other_store_document())
        imported = await import_document(db_session, dumps(theirs), actor="t")
        assert (imported.name, imported.version, imported.publisher) == ("hr", 4, "HR")
        entries = (await db_session.execute(select(CorpusEntryRow))).scalars().all()
        assert [e.source_type for e in entries] == ["VOCABULARY_DOCUMENT"]
        with pytest.raises(VocabularyError, match="different v4"):
            changed = theirs.model_copy(update={"description": "edited"})
            await import_document(db_session, dumps(changed))

    async def test_import_refuses_a_rewind(self, db_session: AsyncSession) -> None:
        theirs = loads(await self._other_store_document())
        await import_document(db_session, dumps(theirs))
        older = theirs.model_copy(update={"version": 2})
        with pytest.raises(VocabularyError, match="monotonic"):
            await import_document(db_session, dumps(older))

    async def test_import_of_something_else_is_refused(self, db_session: AsyncSession) -> None:
        with pytest.raises(VocabularyError, match="not a vocabulary document"):
            await import_document(db_session, '{"@type": "skos:ConceptScheme"}')

    async def _other_store_document(self) -> str:
        from particles.core.vocabulary import new_document, set_profile

        doc = new_document(
            name="hr", prefix="hr", namespace="https://hr.example.org/vocab/", publisher="HR"
        )
        for kind in (SlotKind.ONE_AT_A_TIME, SlotKind.MANY_AT_ONCE, SlotKind.TIMELESS_SINGLE):
            doc = set_profile(
                doc,
                form="report to",
                subject_class="person",
                kind=kind,
                ruling=Ruling(confirmed_by="hr-reviewer"),
            )
        return dumps(doc)
