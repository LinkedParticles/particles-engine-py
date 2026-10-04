# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The vocabulary store: append-only versions, the revision rules,
adoption store-wide or with a lens, the extractor's sink, the deposit sentinel,
and the predicate census.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.predicate_profile import PredicateRole, SlotKind
from particles.core.schema import (
    ClaimTerm,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    SourceType,
    StructuredClaim,
    TermKind,
    UncertaintyNature,
)
from particles.core.status import Status
from particles.core.vocabulary import (
    Ruling,
    VocabularyDocument,
    add_alias,
    new_document,
    set_profile,
)
from particles.core.vocabulary_jsonld import dumps
from particles.extraction.vocabulary import VocabularyExtractor
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import insert_particle, predicate_census
from particles.store.vocabulary_store import (
    adopt_document,
    get_adopted_documents,
    get_document,
    list_documents,
    list_versions,
    materialise_document,
    unadopt_document,
)

T0 = datetime(2026, 10, 2, tzinfo=UTC)
RULING = Ruling(confirmed_by="reviewer", confirmed_at=T0)
RECORD = "artifact:record"


def _v1(name: str = "acme") -> VocabularyDocument:
    return new_document(
        name=name, prefix="acme", namespace="https://vocab.example.org/acme/", issued=T0
    )


def _profiled(name: str = "acme") -> VocabularyDocument:
    doc = add_alias(
        _v1(name), form="mov to", subject_class=RECORD, aliases=["mov from"], ruling=RULING
    )
    return set_profile(
        doc,
        form="mov to",
        subject_class=RECORD,
        kind=SlotKind.TIMELESS_SINGLE,
        roles={"mov from": PredicateRole.PAST},
        ruling=RULING,
    )


class TestMaterialise:
    async def test_versions_are_appended_and_the_highest_is_current(
        self, db_session: AsyncSession
    ) -> None:
        v1 = _v1()
        v3 = _profiled()
        assert await materialise_document(db_session, v1, "entry-1") is None
        assert await materialise_document(db_session, v3, "entry-3") is None
        current = await get_document(db_session, "acme")
        assert current is not None and current.version == 3
        assert current.corpus_entry_id == "entry-3"
        old = await get_document(db_session, "acme", 1)
        assert old is not None and old.terms == []
        assert [r.version for r in await list_versions(db_session, "acme")] == [1, 3]
        events = await list_events(db_session, event_type=OperatorEventType.VOCABULARY_CHANGED)
        assert {(e.payload or {}).get("version") for e in events} == {1, 3}

    async def test_a_version_is_never_reused_or_rewound(self, db_session: AsyncSession) -> None:
        await materialise_document(db_session, _profiled(), None)
        refusal = await materialise_document(db_session, _v1(), None)
        assert refusal is not None and "monotonic" in refusal

    async def test_the_namespace_never_moves(self, db_session: AsyncSession) -> None:
        await materialise_document(db_session, _v1(), None)
        moved = _profiled().model_copy(update={"namespace": "https://elsewhere.example/"})
        refusal = await materialise_document(db_session, moved, None)
        assert refusal is not None and "namespace" in refusal

    async def test_a_minted_term_is_never_dropped(self, db_session: AsyncSession) -> None:
        doc = _profiled()
        await materialise_document(db_session, doc, None)
        dropped = doc.model_copy(update={"version": doc.version + 1, "terms": []})
        refusal = await materialise_document(db_session, dropped, None)
        assert refusal is not None and "drops term" in refusal

    async def test_a_minted_term_is_never_renamed(self, db_session: AsyncSession) -> None:
        doc = _profiled()
        await materialise_document(db_session, doc, None)
        term = doc.terms[0].model_copy(
            update={"form": "relocat to", "aliases": [], "profile": None}
        )
        renamed = VocabularyDocument.model_validate(
            {**doc.model_dump(), "version": doc.version + 1, "terms": [term.model_dump()]}
        )
        refusal = await materialise_document(db_session, renamed, None)
        assert refusal is not None and "renames" in refusal


class TestAdoption:
    async def test_adopt_and_unadopt(self, db_session: AsyncSession) -> None:
        await materialise_document(db_session, _profiled(), None)
        assert await get_adopted_documents(db_session) == []
        await adopt_document(db_session, "acme", actor="t")
        assert [d.name for d in await get_adopted_documents(db_session)] == ["acme"]
        with pytest.raises(ValueError, match="already adopted"):
            await adopt_document(db_session, "acme")
        rows = await list_documents(db_session)
        assert [(r.name, a) for r, a in rows] == [("acme", [""])]
        await unadopt_document(db_session, "acme")
        assert await get_adopted_documents(db_session) == []
        with pytest.raises(ValueError, match="not adopted"):
            await unadopt_document(db_session, "acme")

    async def test_an_unknown_document_cannot_be_adopted(self, db_session: AsyncSession) -> None:
        with pytest.raises(ValueError, match="No vocabulary"):
            await adopt_document(db_session, "nope")

    async def test_a_lens_adoption_is_in_force_only_with_its_lens(
        self, db_session: AsyncSession
    ) -> None:
        from particles.core.schema import TrustLensDefinition
        from particles.store.lens_store import adopt_lens, materialise_lens

        await materialise_document(db_session, _profiled(), None)
        await adopt_document(db_session, "acme", lens="hr")
        assert await get_adopted_documents(db_session) == []
        await materialise_lens(db_session, TrustLensDefinition(name="hr", version=1))
        await adopt_lens(db_session, "hr")
        assert [d.name for d in await get_adopted_documents(db_session)] == ["acme"]


class TestExtractor:
    def _snapshot(self) -> Snapshot:
        return Snapshot(
            snapshot_id="s",
            entry_id="e",
            content_hash="h",
            captured_at=T0,
            archive_path="/dev/null",
        )

    def test_accepts_only_its_source_type(self) -> None:
        ex = VocabularyExtractor()
        assert ex.accepts(SourceType.VOCABULARY_DOCUMENT)
        assert not ex.accepts(SourceType.TRUST_LENS_DEFINITION)

    async def test_parses_without_a_session_and_emits_no_particles(self) -> None:
        result = await VocabularyExtractor().extract(self._snapshot(), dumps(_profiled()).encode())
        assert result.candidates == []
        assert "parsed (1 terms)" in result.quality_notes[0]

    async def test_an_invalid_document_is_a_quality_note(self) -> None:
        result = await VocabularyExtractor().extract(self._snapshot(), b'{"@type": "x"}')
        assert result.candidates == []
        assert result.quality_notes[0].startswith("Invalid vocabulary document")

    async def test_the_sink_materialises_and_reports_a_refusal(
        self, db_session: AsyncSession
    ) -> None:
        content = dumps(_profiled()).encode()
        ok = await VocabularyExtractor().extract(
            self._snapshot(), content, session=db_session, corpus_entry_id="e1"
        )
        assert ok.quality_notes == []
        again = await VocabularyExtractor().extract(
            self._snapshot(), content, session=db_session, corpus_entry_id="e2"
        )
        assert "monotonic" in again.quality_notes[0]


class TestDepositSentinel:
    @pytest.mark.parametrize("suffix", [".json", ".jsonld"])
    def test_a_vocabulary_file_is_detected(self, tmp_path: Path, suffix: str) -> None:
        from particles.corpus.deposit import _resolve_source_type

        path = tmp_path / f"acme{suffix}"
        path.write_text(dumps(_v1()))
        assert _resolve_source_type(path, path.read_bytes(), None) == "VOCABULARY_DOCUMENT"

    def test_other_json_ld_is_not(self, tmp_path: Path) -> None:
        from particles.corpus.deposit import _is_vocabulary_document

        assert not _is_vocabulary_document(b'{"@type": "skos:ConceptScheme"}', ".jsonld")
        assert not _is_vocabulary_document(dumps(_v1()).encode(), ".ttl")
        assert not _is_vocabulary_document(b"[1]", ".json")


def _claim_particle(
    predicate: str, subject_id: str | None, status: Status = Status.ACTIVE
) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=f"x {predicate} y",
        confidence=Confidence(value=0.9),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="t",
        asserted_at=T0,
        status=status,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id="s")
        ],
        structured_claim=StructuredClaim(
            subject=ClaimTerm(kind=TermKind.TOKEN, value="x"),
            predicate=ClaimTerm(kind=TermKind.TOKEN, value=predicate),
            object=ClaimTerm(kind=TermKind.TOKEN, value="y"),
            subject_id=subject_id,
            structurizer_id="t",
            structurizer_version="1",
        ),
    )


class TestPredicateCensus:
    async def test_reads_active_claims_with_their_subject(self, db_session: AsyncSession) -> None:
        await insert_particle(db_session, _claim_particle("moved to", "s1"))
        await insert_particle(db_session, _claim_particle("contains", None))
        await insert_particle(db_session, _claim_particle("retired", "s1", Status.SUPERSEDED))
        rows = await predicate_census(db_session)
        assert sorted(rows, key=lambda r: r[1]) == [(None, "contains"), ("s1", "moved to")]
