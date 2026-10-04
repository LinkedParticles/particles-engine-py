# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy ORM and repository for vocabulary documents.

A vocabulary document arrives as a deposited corpus artefact, the trust-lens
pattern: the ``VocabularyExtractor`` hands the parsed document to
:func:`materialise_document` through the sink registered at
the bottom of this module. Unlike a lens, whose materialisation keeps only
the latest version, **every version is a row**: ``vocabulary_versions`` is
append-only, as the corpus entries behind it are, so ``vocab show --version``
reads any past ruling set without re-parsing a blob. The current version of a
name is its highest.

Adoption is store state, one row per adopted name. A row with an empty
``lens_name`` adopts the document for the whole store; a row naming a trust
lens adopts it only while that lens is adopted, so two departments can
disagree about a profile the way lenses let them disagree about sources.

Two revision rules protect what a published document already said
(:func:`_revision_refusal`): a later version keeps the name's namespace and
prefix, and keeps every minted term with its form and class. A term gains
aliases, alignments and profiles; it is never renamed or dropped, so an IRI
another store or document cites keeps resolving to the same relation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.vocabulary import VocabularyDocument
from particles.core.vocabulary_jsonld import dumps, loads
from particles.db import Base
from particles.store.event_store import OperatorEventType, record_event
from particles.store.lens_store import LensAdoptionRow

log = logging.getLogger(__name__)

#: The ``lens_name`` of a store-wide adoption.
STORE_WIDE = ""


class VocabularyVersionRow(Base):
    """One materialised version of a vocabulary document. Rows are only appended."""

    __tablename__ = "vocabulary_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_vocabulary_name_version"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    prefix: Mapped[str] = mapped_column(String, nullable=False)
    namespace: Mapped[str] = mapped_column(String, nullable=False)
    publisher: Mapped[str | None] = mapped_column(String, nullable=True)
    term_count: Mapped[int] = mapped_column(Integer, nullable=False)
    document_json: Mapped[str] = mapped_column(Text, nullable=False)
    corpus_entry_id: Mapped[str | None] = mapped_column(String, nullable=True)
    materialised_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class VocabularyAdoptionRow(Base):
    """One adoption: a document name, store-wide or riding one trust lens."""

    __tablename__ = "vocabulary_adoptions"

    vocabulary_name: Mapped[str] = mapped_column(String, primary_key=True)
    lens_name: Mapped[str] = mapped_column(String, primary_key=True, default=STORE_WIDE)
    adopted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    adopted_by: Mapped[str] = mapped_column(String, nullable=False)


def _row_to_document(row: VocabularyVersionRow) -> VocabularyDocument:
    doc = loads(row.document_json)
    return doc.model_copy(update={"corpus_entry_id": row.corpus_entry_id})


async def _latest_row(session: AsyncSession, name: str) -> VocabularyVersionRow | None:
    return (
        await session.execute(
            select(VocabularyVersionRow)
            .where(VocabularyVersionRow.name == name)
            .order_by(VocabularyVersionRow.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


def _revision_refusal(previous: VocabularyDocument, doc: VocabularyDocument) -> str | None:
    """Why ``doc`` may not follow ``previous``, or ``None`` when it may."""
    if doc.version <= previous.version:
        return (
            f"vocabulary {doc.name!r} v{doc.version} not materialised: v{previous.version} "
            "is already current (versions are monotonic)."
        )
    if (doc.namespace, doc.prefix) != (previous.namespace, previous.prefix):
        return (
            f"vocabulary {doc.name!r} v{doc.version} not materialised: it changes the "
            f"namespace or prefix of v{previous.version} ({previous.prefix}: "
            f"{previous.namespace}), which would move every published term IRI."
        )
    now = {t.local_name: t for t in doc.terms}
    for term in previous.terms:
        kept = now.get(term.local_name)
        if kept is None:
            return (
                f"vocabulary {doc.name!r} v{doc.version} not materialised: it drops term "
                f"{term.local_name!r}. A minted term is never removed; map it outward instead."
            )
        if (kept.form, kept.subject_class) != (term.form, term.subject_class):
            return (
                f"vocabulary {doc.name!r} v{doc.version} not materialised: it renames term "
                f"{term.local_name!r}. A minted term keeps its form and class."
            )
    return None


async def revision_refusal(session: AsyncSession, doc: VocabularyDocument) -> str | None:
    """Why the store would refuse ``doc`` as the next version of its name, or ``None``."""
    latest = await _latest_row(session, doc.name)
    return _revision_refusal(_row_to_document(latest), doc) if latest is not None else None


async def materialise_document(
    session: AsyncSession, doc: VocabularyDocument, corpus_entry_id: str | None = None
) -> str | None:
    """Append one version; returns a human-readable rejection, or ``None`` on success."""
    latest = await _latest_row(session, doc.name)
    if latest is not None:
        refusal = _revision_refusal(_row_to_document(latest), doc)
        if refusal is not None:
            return refusal
    session.add(
        VocabularyVersionRow(
            name=doc.name,
            version=doc.version,
            prefix=doc.prefix,
            namespace=doc.namespace,
            publisher=doc.publisher,
            term_count=len(doc.terms),
            document_json=dumps(doc),
            corpus_entry_id=corpus_entry_id,
            materialised_at=datetime.now(UTC),
        )
    )
    await session.flush()
    await record_event(
        session,
        actor="vocabulary-extractor",
        event_type=OperatorEventType.VOCABULARY_CHANGED,
        payload={
            "kind": "materialised",
            "name": doc.name,
            "version": doc.version,
            "previous_version": latest.version if latest is not None else None,
            "terms": len(doc.terms),
            "corpus_entry_id": corpus_entry_id,
        },
    )
    return None


async def get_document(
    session: AsyncSession, name: str, version: int | None = None
) -> VocabularyDocument | None:
    """A version of a document (the current one by default), or ``None``."""
    if version is None:
        row = await _latest_row(session, name)
    else:
        row = (
            await session.execute(
                select(VocabularyVersionRow).where(
                    VocabularyVersionRow.name == name, VocabularyVersionRow.version == version
                )
            )
        ).scalar_one_or_none()
    return _row_to_document(row) if row is not None else None


async def list_versions(session: AsyncSession, name: str) -> list[VocabularyVersionRow]:
    """Every materialised version of a name, oldest first."""
    return list(
        (
            await session.execute(
                select(VocabularyVersionRow)
                .where(VocabularyVersionRow.name == name)
                .order_by(VocabularyVersionRow.version)
            )
        )
        .scalars()
        .all()
    )


async def list_documents(
    session: AsyncSession,
) -> list[tuple[VocabularyVersionRow, list[str]]]:
    """The current version of every name, with its adoptions (``""`` for store-wide)."""
    newest = (
        select(VocabularyVersionRow.name, func.max(VocabularyVersionRow.version).label("v"))
        .group_by(VocabularyVersionRow.name)
        .subquery()
    )
    rows = (
        (
            await session.execute(
                select(VocabularyVersionRow)
                .join(
                    newest,
                    (VocabularyVersionRow.name == newest.c.name)
                    & (VocabularyVersionRow.version == newest.c.v),
                )
                .order_by(VocabularyVersionRow.name)
            )
        )
        .scalars()
        .all()
    )
    adoptions: dict[str, list[str]] = {}
    for a in (await session.execute(select(VocabularyAdoptionRow))).scalars().all():
        adoptions.setdefault(a.vocabulary_name, []).append(a.lens_name)
    return [(row, sorted(adoptions.get(row.name, []))) for row in rows]


async def adopt_document(
    session: AsyncSession, name: str, *, lens: str | None = None, actor: str = "operator"
) -> None:
    """Adopt a document store-wide, or only while ``lens`` is adopted.

    Raises ``ValueError`` for an unknown document or a repeated adoption. A
    lens need not be adopted yet: the row waits for it, the way an adopted lens
    waits for a re-deposit.
    """
    latest = await _latest_row(session, name)
    if latest is None:
        raise ValueError(f"No vocabulary named {name!r}: create or import it first.")
    key = (name, lens or STORE_WIDE)
    if await session.get(VocabularyAdoptionRow, key) is not None:
        where = f"with lens {lens!r}" if lens else "store-wide"
        raise ValueError(f"Vocabulary {name!r} is already adopted {where}.")
    session.add(
        VocabularyAdoptionRow(
            vocabulary_name=name,
            lens_name=lens or STORE_WIDE,
            adopted_at=datetime.now(UTC),
            adopted_by=actor,
        )
    )
    await session.flush()
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.VOCABULARY_CHANGED,
        payload={"kind": "adopt", "name": name, "version": latest.version, "lens": lens},
    )


async def unadopt_document(
    session: AsyncSession, name: str, *, lens: str | None = None, actor: str = "operator"
) -> None:
    """Remove one adoption; raises ``ValueError`` when it does not exist."""
    row = await session.get(VocabularyAdoptionRow, (name, lens or STORE_WIDE))
    if row is None:
        where = f"with lens {lens!r}" if lens else "store-wide"
        raise ValueError(f"Vocabulary {name!r} is not adopted {where}.")
    await session.delete(row)
    await session.flush()
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.VOCABULARY_CHANGED,
        payload={"kind": "unadopt", "name": name, "lens": lens},
    )


async def get_adopted_documents(session: AsyncSession) -> list[VocabularyDocument]:
    """The current version of every document in force for this store.

    In force means adopted store-wide, or adopted with a trust lens the store
    currently adopts. No adoption at all, the state of every store until an
    operator adopts one, returns an empty list after one query.
    """
    adoptions = (await session.execute(select(VocabularyAdoptionRow))).scalars().all()
    if not adoptions:
        return []
    lenses = {r.lens_name for r in (await session.execute(select(LensAdoptionRow))).scalars()}
    names = sorted(
        {a.vocabulary_name for a in adoptions if a.lens_name == STORE_WIDE or a.lens_name in lenses}
    )
    docs: list[VocabularyDocument] = []
    for name in names:
        doc = await get_document(session, name)
        if doc is not None:
            docs.append(doc)
        else:  # an adoption outlived its document; be loud, not fatal
            log.warning("Adopted vocabulary %r has no materialised version; ignoring.", name)
    return docs


# ---------------------------------------------------------------------------
# register the Engine-side sink with the Client-side extractor.
# ---------------------------------------------------------------------------
from particles.extraction.vocabulary import register_vocabulary_sink  # noqa: E402

register_vocabulary_sink(materialise_document)
