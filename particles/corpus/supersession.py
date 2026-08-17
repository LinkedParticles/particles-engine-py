"""Document-supersession relation between corpus entries (capability 2).

A corpus entry MAY declare that its source **document** supersedes another
document — an authored, machine-readable editorial fact (*"this decision
replaces that one"*). frontmatter ``supersedes: "0017"`` is the
motivating instance. This module is the Engine-side substrate for that
relation, in two halves:

* **The genre adapter** (:func:`document_relation_for_content`) — reads a
  document's structure into a :class:`DocumentRelation`. Capability 2 ships the
  **ADR** instance only: it reads the ``type: ADR`` frontmatter's ``id`` plus
  its ``supersedes:`` / ``superseded_by:`` fields. New genres (RFC ``Obsoletes:``,
  spec version lineage) are additive future adapters keyed off the same
  frontmatter parse; capability 3's ``section_roles`` hook would join here.
  Returns ``None`` for any document that is not a recognised genre — the cheap,
  common case — so the hook is a no-op on web pages, UGC, PDFs, etc.

* **The relation store + transitive predicate** (:func:`set_document_relation`,
  :func:`entry_supersedes`). The relation is captured at ingest on the corpus
  entry (``CorpusEntryRow.document_supersession_json``) and followed
  **transitively** over the document-*key* graph (so 0116 beats 0009 *through*
  0017) — keying on document identity, not entry id, makes it independent of
  deposit order and of re-deposits. This is distinct from the particle
  ``RelationType`` registry (which joins particles); it is a
  source-document editorial edge between corpus entries.

The §6.6 rung-1.5 prior (cap. 2) consults :func:`entry_supersedes` to
prefer the superseding document's claim when two claims conflict.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.corpus.store import CorpusEntryRow


@dataclass(frozen=True)
class DocumentRelation:
    """A document's canonical key and its direct supersession edges.

    ``supersedes`` and ``superseded_by`` are recorded from *whichever* side the
    source document annotates — the ADR convention writes both (the new ADR's
    ``supersedes:`` and the old ADR's ``superseded_by:``), but reading either
    one alone recovers the same directed edge, so an asymmetrically-annotated
    corpus still resolves.
    """

    key: str
    supersedes: tuple[str, ...] = ()
    superseded_by: tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps(
            {
                "key": self.key,
                "supersedes": list(self.supersedes),
                "superseded_by": list(self.superseded_by),
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> DocumentRelation | None:
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(obj, dict) or not obj.get("key"):
            return None
        return cls(
            key=str(obj["key"]),
            supersedes=tuple(str(k) for k in obj.get("supersedes", []) if k),
            superseded_by=tuple(str(k) for k in obj.get("superseded_by", []) if k),
        )


# --- Genre adapter: ADR instance -------------------------------------------

_ADR_KEY_PREFIX = "adr"


def _parse_frontmatter(content: bytes) -> dict[str, Any] | None:
    """Return the leading YAML frontmatter block as a dict, or ``None``.

    Recognises a ``---``-fenced block at the very start of a UTF-8 text
    document (the Markdown / Obsidian / ADR convention). Mirrors the extractor's
    :func:`particles.extraction.general._strip_obsidian_frontmatter` detection,
    kept Engine-side so the corpus layer needn't import the Client extractor.
    Bounded to the first 8 KiB and tolerant of malformed YAML (returns ``None``).
    """
    # Cheap reject: only text documents that *open* with a fence can carry it.
    head = content[:8192]
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return None
    after_open = 4 if text.startswith("---\n") else 5
    close = text.find("\n---", after_open)
    if close == -1:
        return None
    yaml_block = text[after_open:close]
    try:
        import yaml

        parsed = yaml.safe_load(yaml_block) if yaml_block.strip() else None
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalise_adr_id(value: Any) -> str | None:
    """``"0017"`` / ``17`` / an ``ADR``-prefixed string → ``"adr:0017"`` (None if unusable)."""
    if value is None:
        return None
    token = str(value).strip().lower().removeprefix("adr").strip(" :-")
    if not token:
        return None
    return f"{_ADR_KEY_PREFIX}:{token}"


def _coerce_id_list(value: Any) -> tuple[str, ...]:
    """Frontmatter ``supersedes`` may be a scalar or a list; normalise to keys."""
    if value is None:
        return ()
    items = value if isinstance(value, list) else [value]
    keys = [_normalise_adr_id(item) for item in items]
    return tuple(k for k in keys if k is not None)


def _adr_relation(meta: dict[str, Any]) -> DocumentRelation | None:
    """The ADR genre adapter: ``type: ADR`` frontmatter → a DocumentRelation."""
    if str(meta.get("type", "")).strip().upper() != "ADR":
        return None
    key = _normalise_adr_id(meta.get("id"))
    if key is None:
        return None
    return DocumentRelation(
        key=key,
        supersedes=_coerce_id_list(meta.get("supersedes")),
        superseded_by=_coerce_id_list(meta.get("superseded_by")),
    )


def document_relation_for_content(content: bytes) -> DocumentRelation | None:
    """Genre-adapter entry point: derive a document's supersession relation.

    Capability 2 recognises the **ADR** genre only. Returns ``None`` for every
    other source, so the deposit-time hook is a cheap no-op on non-ADR content.
    """
    meta = _parse_frontmatter(content)
    if meta is None:
        return None
    return _adr_relation(meta)


# --- Relation store + transitive predicate ---------------------------------


async def set_document_relation(
    session: AsyncSession, entry_id: str, relation: DocumentRelation | None
) -> None:
    """Stamp (or clear) a corpus entry's document-supersession relation.

    Idempotent — re-depositing the same document overwrites the prior value, so
    an edited ``supersedes:`` frontmatter is picked up on the next deposit. A
    ``None`` relation clears the column (the entry is no longer a recognised
    genre). No-op when neither the stored nor the new value changes anything.
    """
    row = await session.get(CorpusEntryRow, entry_id)
    if row is None:
        return
    new_value = relation.to_json() if relation is not None else None
    if row.document_supersession_json != new_value:
        row.document_supersession_json = new_value
        await session.flush()


async def _key_of(session: AsyncSession, entry_id: str) -> str | None:
    row = await session.get(CorpusEntryRow, entry_id)
    if row is None or row.document_supersession_json is None:
        return None
    rel = DocumentRelation.from_json(row.document_supersession_json)
    return rel.key if rel is not None else None


async def _supersession_graph(session: AsyncSession) -> dict[str, set[str]]:
    """Forward adjacency ``superseding_key → {superseded_key, …}``.

    Built from every entry that declares a relation, merging edges from both
    annotation directions (``supersedes`` on the new doc, ``superseded_by`` on
    the old doc) so the graph is the same regardless of which side annotated.
    Scoped to entries carrying the column, so it stays cheap at the motivating
    scale (a few hundred ADRs).
    """
    result = await session.execute(
        select(CorpusEntryRow.document_supersession_json).where(
            CorpusEntryRow.document_supersession_json.is_not(None)
        )
    )
    graph: dict[str, set[str]] = {}
    for (raw,) in result.all():
        rel = DocumentRelation.from_json(raw) if raw else None
        if rel is None:
            continue
        for superseded in rel.supersedes:
            graph.setdefault(rel.key, set()).add(superseded)
        for superseding in rel.superseded_by:
            graph.setdefault(superseding, set()).add(rel.key)
    return graph


async def entry_supersedes(
    session: AsyncSession,
    *,
    superseding_entry_id: str,
    superseded_entry_id: str,
) -> bool:
    """True if ``superseding_entry_id``'s document (transitively) supersedes the other's.

    Resolves each entry to its document key, then walks the supersession key
    graph forward from the candidate superseder. Returns ``False`` when either
    entry carries no genre relation, when the two share a key (same document —
    not a supersession), or when no path connects them. Cycle-safe via a visited
    set.
    """
    src = await _key_of(session, superseding_entry_id)
    dst = await _key_of(session, superseded_entry_id)
    if src is None or dst is None or src == dst:
        return False
    graph = await _supersession_graph(session)
    if src not in graph:
        return False
    seen: set[str] = {src}
    queue: deque[str] = deque(graph[src])
    while queue:
        node = queue.popleft()
        if node == dst:
            return True
        if node in seen:
            continue
        seen.add(node)
        queue.extend(graph.get(node, ()))
    return False


async def iter_supersession_entry_pairs(
    session: AsyncSession,
) -> list[tuple[str, str]]:
    """Every ``(superseding_entry_id, superseded_entry_id)`` transitive pair.

    Resolves the full document-supersession graph **once** (unlike repeated
    :func:`entry_supersedes` calls, each of which rebuilds it) and maps document
    keys back to the corpus entries that carry them, so the cross-entry
    reconcile sweep can enumerate its candidate entry pairs in a
    single pass. Each ordered pair means the first entry's document
    (transitively) supersedes the second's. A key that maps to several deposited
    entries (a re-deposit) yields a pair for each; self-pairs (same key) are
    excluded. Returns ``[]`` when no entry carries a genre relation.
    """
    result = await session.execute(
        select(CorpusEntryRow.entry_id, CorpusEntryRow.document_supersession_json).where(
            CorpusEntryRow.document_supersession_json.is_not(None)
        )
    )
    key_to_entries: dict[str, list[str]] = {}
    entry_key: dict[str, str] = {}
    for entry_id, raw in result.all():
        rel = DocumentRelation.from_json(raw) if raw else None
        if rel is None:
            continue
        key_to_entries.setdefault(rel.key, []).append(entry_id)
        entry_key[entry_id] = rel.key
    if not entry_key:
        return []

    graph = await _supersession_graph(session)
    pairs: list[tuple[str, str]] = []
    for sup_entry_id, src_key in entry_key.items():
        if src_key not in graph:
            continue
        # BFS forward from src_key over the transitive supersession graph.
        seen: set[str] = {src_key}
        queue: deque[str] = deque(graph[src_key])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            for sub_entry_id in key_to_entries.get(node, ()):
                if sub_entry_id != sup_entry_id:
                    pairs.append((sup_entry_id, sub_entry_id))
            queue.extend(graph.get(node, ()))
    return pairs
