# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The vocabulary document's lifecycle (§4).

A vocabulary document is where a store's reviewed modelling rulings live:
canonical predicates, their aliases, their outward alignments, and the kind of
slot each fills. This module is the one entry point the ``particles vocab``
verbs drive:

* :func:`create_document` mints version 1 of a new namespace;
* :func:`propose` reads the store's predicate census, suggests alias and
  profile candidates, and records each as a ``VOCABULARY_PROPOSED`` event, the
  proposal's only persistence (the ``ABSTRACTION_CANDIDATE`` pattern);
* :func:`confirm` and :func:`decline` are the reviewer's ruling on a proposal,
  each a ``VOCABULARY_RULED`` event; a confirmation also publishes the next
  version of the document;
* :func:`align` records an outward alignment, a direct ruling;
* :func:`import_document` and :func:`export_document` carry a document
  between stores; :func:`adopt` / :func:`unadopt` put one in force.

**Every revision is a new deposit.** :func:`publish_version` deposits the
canonical JSON-LD as a ``VOCABULARY_DOCUMENT`` corpus entry and extracts it,
so the vocabulary extractor materialises the version through the same path a
document deposited by hand takes. Nothing edits a version once written.

**A document is a prior, never a gate.** Nothing here touches extraction: a
claim whose predicate no document knows is extracted and reconciled exactly as
before (the report-only stance, applied to modelling). An adopted
document's alignments feed the vocabulary report; nothing on the §6.6 ladder
reads a document. Profiles steering the update rung were proposed and
declined after measurement.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.predicate_profile import PredicateRole, SlotKind, normalise_predicate
from particles.core.schema import SourceType
from particles.core.vocabulary import (
    Alignment,
    CensusRow,
    ProposalPlan,
    Ruling,
    VocabularyDocument,
    add_alias,
    add_alignment,
    find_term,
    form_stats,
    new_document,
    plan_proposals,
    set_profile,
)
from particles.core.vocabulary_jsonld import dumps, loads
from particles.corpus.deposit import deposit_text
from particles.embeddings import get_embedding_model
from particles.ingest.pipeline import extract_snapshot
from particles.store.event_store import (
    OperatorEvent,
    OperatorEventType,
    list_events,
    record_event,
)
from particles.store.particle_store import predicate_census
from particles.store.subject_store import get_subject_classes
from particles.store.vocabulary_store import (
    adopt_document,
    get_document,
    revision_refusal,
    unadopt_document,
)

log = logging.getLogger(__name__)

__all__ = [
    "ProposeReport",
    "VocabularyError",
    "adopt",
    "align",
    "confirm",
    "create_document",
    "decline",
    "export_document",
    "import_document",
    "pending_proposals",
    "propose",
    "publish_version",
    "unadopt",
]

#: Large enough to read every proposal and ruling a store has recorded; the
#: event log is the proposal's persistence, so a truncated read would re-propose.
_EVENT_SCAN = 1_000_000


class VocabularyError(ValueError):
    """An operator-facing refusal: unknown document, stale proposal, bad ruling."""


# ---------------------------------------------------------------------------
# Publishing a version
# ---------------------------------------------------------------------------


async def publish_version(
    session: AsyncSession, doc: VocabularyDocument, *, deposited_by: str, text: str | None = None
) -> VocabularyDocument:
    """Deposit one version and materialise it through the vocabulary extractor.

    ``text`` is the bytes to archive; the canonical serialisation by default,
    or an imported file exactly as received. An identical version already
    materialised is returned unchanged, so a repeated import is a no-op.
    Raises :class:`VocabularyError` when the store would refuse the version.
    Does not commit.
    """
    existing = await get_document(session, doc.name, doc.version)
    if existing is not None:
        if dumps(existing) == dumps(doc):
            return existing
        raise VocabularyError(
            f"vocabulary {doc.name!r} already has a different v{doc.version}; versions are "
            "monotonic and a published version is never replaced."
        )
    refusal = await revision_refusal(session, doc)
    if refusal is not None:
        raise VocabularyError(refusal)
    entry_id, snapshot_id = await deposit_text(
        session,
        text if text is not None else dumps(doc),
        deposited_by=deposited_by,
        source_type=SourceType.VOCABULARY_DOCUMENT,
        tags=[f"vocabulary:{doc.name}"],
    )
    await extract_snapshot(session, entry_id, snapshot_id)
    stored = await get_document(session, doc.name, doc.version)
    if stored is None:
        raise VocabularyError(
            f"vocabulary {doc.name!r} v{doc.version} was deposited (corpus entry {entry_id}) "
            "but not materialised; see the snapshot's quality notes."
        )
    return stored


async def _current(session: AsyncSession, name: str) -> VocabularyDocument:
    doc = await get_document(session, name)
    if doc is None:
        raise VocabularyError(f"No vocabulary named {name!r}: create or import it first.")
    return doc


async def create_document(
    session: AsyncSession,
    *,
    name: str,
    prefix: str,
    namespace: str,
    publisher: str | None = None,
    description: str | None = None,
    actor: str = "operator",
) -> VocabularyDocument:
    """Mint version 1 of a new vocabulary. Raises :class:`VocabularyError` if the name exists."""
    if await get_document(session, name) is not None:
        raise VocabularyError(f"A vocabulary named {name!r} already exists.")
    try:
        doc = new_document(
            name=name,
            prefix=prefix,
            namespace=namespace,
            publisher=publisher,
            description=description,
        )
    except ValueError as exc:
        raise VocabularyError(str(exc)) from exc
    stored = await publish_version(session, doc, deposited_by=actor)
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.VOCABULARY_CHANGED,
        payload={"kind": "created", "name": name, "version": 1, "namespace": namespace},
    )
    return stored


async def import_document(
    session: AsyncSession, text: str, *, actor: str = "operator"
) -> VocabularyDocument:
    """Deposit another operator's document exactly as received, and materialise it.

    The name, namespace and every ruling travel unchanged; the corpus keeps the
    received bytes as the import's provenance. Importing does not adopt.
    """
    try:
        doc = loads(text)
    except ValueError as exc:
        raise VocabularyError(str(exc)) from exc
    stored = await publish_version(session, doc, deposited_by=actor, text=text)
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.VOCABULARY_CHANGED,
        payload={
            "kind": "imported",
            "name": doc.name,
            "version": doc.version,
            "publisher": doc.publisher,
        },
    )
    return stored


async def export_document(session: AsyncSession, name: str, version: int | None = None) -> str:
    """The canonical JSON-LD of one version (the current one by default)."""
    doc = await get_document(session, name, version)
    if doc is None:
        which = f" v{version}" if version is not None else ""
        raise VocabularyError(f"No vocabulary {name!r}{which}.")
    return dumps(doc)


async def adopt(
    session: AsyncSession, name: str, *, lens: str | None = None, actor: str = "operator"
) -> None:
    """Put a document in force for this store, or only while ``lens`` is adopted."""
    try:
        await adopt_document(session, name, lens=lens, actor=actor)
    except ValueError as exc:
        raise VocabularyError(str(exc)) from exc


async def unadopt(
    session: AsyncSession, name: str, *, lens: str | None = None, actor: str = "operator"
) -> None:
    """Remove one adoption."""
    try:
        await unadopt_document(session, name, lens=lens, actor=actor)
    except ValueError as exc:
        raise VocabularyError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


@dataclass
class ProposeReport:
    """What one proposal run read, found, and recorded."""

    document: str
    claims: int = 0
    distinct_predicates: int = 0
    classed_claims: int = 0
    plan: ProposalPlan = field(default_factory=ProposalPlan)
    recorded: list[dict[str, Any]] = field(default_factory=list)
    encoder_missing: bool = False

    def alias_coverage(self) -> int:
        """Distinct surface predicates some alias candidate covers, over every class."""
        return len(
            {surface for p in self.plan.aliases for m in p.members for surface, _ in m.surfaces}
        )


async def _proposal_events(
    session: AsyncSession, document: str
) -> tuple[list[OperatorEvent], dict[str, OperatorEvent]]:
    """This document's proposal events, and its rulings keyed by proposal event id."""
    proposals = [
        e
        for e in await list_events(
            session, event_type=OperatorEventType.VOCABULARY_PROPOSED, limit=_EVENT_SCAN
        )
        if (e.payload or {}).get("document") == document
    ]
    rulings: dict[str, OperatorEvent] = {}
    for e in await list_events(
        session, event_type=OperatorEventType.VOCABULARY_RULED, limit=_EVENT_SCAN
    ):
        pid = (e.payload or {}).get("proposal_event_id")
        if pid:
            rulings[str(pid)] = e
    return proposals, rulings


async def pending_proposals(
    session: AsyncSession, document: str | None = None
) -> list[OperatorEvent]:
    """Unruled proposals, highest claim count first; all documents when ``document`` is None."""
    proposals = await list_events(
        session, event_type=OperatorEventType.VOCABULARY_PROPOSED, limit=_EVENT_SCAN
    )
    _, rulings = await _proposal_events(session, document or "")
    pending = [
        e
        for e in proposals
        if e.event_id not in rulings
        and (document is None or (e.payload or {}).get("document") == document)
    ]
    pending.sort(key=lambda e: (-int((e.payload or {}).get("claims", 0)), e.occurred_at))
    return pending


async def _census(
    session: AsyncSession,
) -> tuple[list[CensusRow], int, int, int]:
    """The classed census, plus total claims, distinct predicates and classed claims."""
    pairs = await predicate_census(session)
    classes = await get_subject_classes(session, {sid for sid, _ in pairs if sid})
    counts: Counter[tuple[str, str]] = Counter()
    for sid, predicate in pairs:
        cls = classes.get(sid) if sid else None
        if cls:
            counts[(cls, predicate)] += 1
    rows = [CensusRow(subject_class=c, predicate=p, claims=n) for (c, p), n in counts.items()]
    return rows, len(pairs), len({p for _, p in pairs}), sum(counts.values())


def _embed(texts: Sequence[str]) -> dict[str, np.ndarray[Any, Any]] | None:
    """Unit vectors for each distinct text with the local encoder, or None without one."""
    unique = sorted(set(texts))
    if not unique:
        return {}  # nothing to embed: never load the encoder for an empty census
    model = get_embedding_model()
    if model is None:
        return None
    vectors = model.encode(list(unique), convert_to_numpy=True, normalize_embeddings=True)
    return {
        text: np.asarray(vec, dtype=np.float32) for text, vec in zip(unique, vectors, strict=True)
    }


async def propose(
    session: AsyncSession,
    name: str,
    *,
    limit: int | None = None,
    threshold: float | None = None,
    kinds: Collection[str] = ("alias", "profile"),
    subject_class: str | None = None,
    form: str | None = None,
    dry_run: bool = False,
    actor: str = "operator",
) -> ProposeReport:
    """Suggest alias and profile candidates from the store's predicates (§4).

    Only claims whose triple is about a classed subject take part, since a
    canonical predicate is a form on a class. Alias candidates come from the
    local encoder at ``threshold`` within one class and never join opposite
    polarity or direction; profile candidates are the canonical predicates no
    document profiles yet. The top ``limit`` of each kind are recorded as
    ``VOCABULARY_PROPOSED`` events unless ``dry_run``; a key already proposed
    for this document, pending or ruled on, is never proposed again. A
    proposal is a candidate: nothing enters the document until confirmed.
    Does not commit.
    """
    cfg = get_config().vocabulary
    limit = cfg.propose_limit if limit is None else limit
    threshold = cfg.alias_similarity if threshold is None else threshold
    doc = await _current(session, name)

    rows, claims, distinct, classed = await _census(session)
    if subject_class is not None:
        rows = [r for r in rows if r.subject_class == subject_class]
    stats = form_stats(rows)
    if form is not None:
        wanted = normalise_predicate(form)
        stats = [s for s in stats if s.form == wanted]

    report = ProposeReport(
        document=name, claims=claims, distinct_predicates=distinct, classed_claims=classed
    )
    vectors: dict[tuple[str, str], np.ndarray[Any, Any]] = {}
    if "alias" in kinds:
        embedded = _embed([s.display for s in stats])
        if embedded is None:
            report.encoder_missing = True
        else:
            vectors = {(s.form, s.subject_class): embedded[s.display] for s in stats}

    proposals, _ = await _proposal_events(session, name)
    excluded = frozenset(str((e.payload or {}).get("key")) for e in proposals)
    plan = plan_proposals(stats, vectors, threshold=threshold, documents=[doc], excluded=excluded)
    if "alias" not in kinds:
        plan.aliases = []
    if "profile" not in kinds:
        plan.profiles = []
    report.plan = plan

    payloads: list[dict[str, Any]] = []
    for a in plan.aliases[:limit]:
        payloads.append(
            {
                "document": name,
                "document_version": doc.version,
                "key": a.key,
                "kind": "alias",
                "subject_class": a.subject_class,
                "form": a.canonical,
                "aliases": list(a.aliases),
                "label": next(m.display for m in a.members if m.form == a.canonical),
                "claims": a.claims,
                "evidence": a.evidence(),
            }
        )
    for p in plan.profiles[:limit]:
        payloads.append(
            {
                "document": name,
                "document_version": doc.version,
                "key": p.key,
                "kind": "profile",
                "subject_class": p.subject_class,
                "form": p.form,
                "label": p.surfaces[0][0],
                "claims": p.claims,
                "evidence": p.evidence(),
            }
        )
    report.recorded = payloads
    if not dry_run:
        for payload in payloads:
            await record_event(
                session,
                actor=actor,
                event_type=OperatorEventType.VOCABULARY_PROPOSED,
                payload=payload,
            )
    return report


# ---------------------------------------------------------------------------
# Rulings
# ---------------------------------------------------------------------------


async def _load_pending(
    session: AsyncSession, keys: Sequence[str]
) -> tuple[str, list[OperatorEvent]]:
    """The pending proposal event for each key, all of one document."""
    proposals = await list_events(
        session, event_type=OperatorEventType.VOCABULARY_PROPOSED, limit=_EVENT_SCAN
    )
    by_key: dict[str, OperatorEvent] = {}
    for e in proposals:
        key = str((e.payload or {}).get("key"))
        by_key.setdefault(key, e)  # newest first: the latest proposal of a key
    _, rulings = await _proposal_events(session, "")
    events: list[OperatorEvent] = []
    for key in keys:
        event = by_key.get(key)
        if event is None:
            raise VocabularyError(f"No vocabulary proposal {key!r}.")
        ruling = rulings.get(event.event_id)
        if ruling is not None:
            raise VocabularyError(
                f"Proposal {key!r} was already {(ruling.payload or {}).get('resolution')}."
            )
        events.append(event)
    documents = {str((e.payload or {}).get("document")) for e in events}
    if len(documents) != 1:
        raise VocabularyError(
            f"Proposals {', '.join(keys)} belong to different documents: {sorted(documents)}."
        )
    return documents.pop(), events


async def confirm(
    session: AsyncSession,
    keys: Sequence[str],
    *,
    kind: SlotKind | None = None,
    past_forms: Sequence[str] = (),
    actor: str = "operator",
    reason: str | None = None,
) -> VocabularyDocument:
    """Confirm proposals into the next version of their document (one version for all).

    An alias proposal adds its forms as ``skos:altLabel`` of its canonical
    predicate. A profile proposal needs ``kind``; ``past_forms`` names forms
    that record a past value of the slot (``move from`` beside ``move to``),
    added to the term with the past role: the role mapping offered in place of
    an alias across directions. Each proposal gets a
    ``VOCABULARY_RULED`` event naming the version it produced. Does not commit.
    """
    name, events = await _load_pending(session, keys)
    doc = await _current(session, name)
    past = [normalise_predicate(f) for f in past_forms]
    for event in events:
        payload = event.payload or {}
        evidence = dict(payload.get("evidence") or {})
        if reason:
            evidence["reason"] = reason
        ruling = Ruling(confirmed_by=actor, evidence=evidence, proposal_event_id=event.event_id)
        cls = str(payload["subject_class"])
        form = str(payload["form"])
        label = payload.get("label")
        try:
            if payload.get("kind") == "alias":
                doc = add_alias(
                    doc,
                    form=form,
                    subject_class=cls,
                    aliases=[str(a) for a in payload.get("aliases") or []],
                    ruling=ruling,
                    label=label,
                )
            else:
                if kind is None:
                    raise VocabularyError(
                        f"Proposal {payload.get('key')!r} is a profile: say which kind of slot "
                        "it fills (--kind timeless_single | one_at_a_time | many_at_once)."
                    )
                if past:
                    doc = add_alias(
                        doc,
                        form=form,
                        subject_class=cls,
                        aliases=past,
                        ruling=ruling,
                        label=label,
                    )
                doc = set_profile(
                    doc,
                    form=form,
                    subject_class=cls,
                    kind=kind,
                    roles={f: PredicateRole.PAST for f in past},
                    ruling=ruling,
                    label=label,
                )
        except ValueError as exc:
            if isinstance(exc, VocabularyError):
                raise
            raise VocabularyError(str(exc)) from exc
    # Each helper bumped the version; one ruling set is one published version.
    original = await _current(session, name)
    doc = VocabularyDocument.model_validate({**doc.model_dump(), "version": original.version + 1})
    stored = await publish_version(session, doc, deposited_by=actor)
    for event in events:
        payload = event.payload or {}
        term = find_term(stored, str(payload["form"]), str(payload["subject_class"]))
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.VOCABULARY_RULED,
            reason=reason,
            payload={
                "resolution": "confirmed",
                "proposal_event_id": event.event_id,
                "key": payload.get("key"),
                "kind": payload.get("kind"),
                "document": name,
                "version": stored.version,
                "term": term.local_name if term is not None else None,
                "slot_kind": kind.value
                if kind is not None and payload.get("kind") == "profile"
                else None,
            },
        )
    return stored


async def decline(
    session: AsyncSession,
    keys: Sequence[str],
    *,
    actor: str = "operator",
    reason: str | None = None,
) -> None:
    """Record proposals as declined; the document is unchanged and never re-proposes them."""
    name, events = await _load_pending(session, keys)
    for event in events:
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.VOCABULARY_RULED,
            reason=reason,
            payload={
                "resolution": "declined",
                "proposal_event_id": event.event_id,
                "key": (event.payload or {}).get("key"),
                "kind": (event.payload or {}).get("kind"),
                "document": name,
            },
        )


async def align(
    session: AsyncSession,
    name: str,
    term_ref: str,
    alignment: Alignment,
    *,
    actor: str = "operator",
) -> VocabularyDocument:
    """Link a term outward to an external property, in the next version.

    The term keeps its IRI: a later external match is an equivalence link, never
    a rename. ``alignment.ruling`` carries who asserted it and on what basis.
    """
    doc = await _current(session, name)
    try:
        revised = add_alignment(doc, term_ref=term_ref, alignment=alignment)
    except ValueError as exc:
        raise VocabularyError(str(exc)) from exc
    stored = await publish_version(session, revised, deposited_by=actor)
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.VOCABULARY_RULED,
        payload={
            "resolution": "aligned",
            "document": name,
            "version": stored.version,
            "term": term_ref,
            "target": alignment.target,
            "match": alignment.match.value,
            "confidence": alignment.confidence,
            "evidence": alignment.ruling.evidence,
        },
    )
    return stored
