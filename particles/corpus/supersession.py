# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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

**A document's key is read within its own collection** (security review
2026-10-24-2, P4). The number an ADR declares (``id: "0017"``, ``supersedes:
"0017"``) names a record of the collection the document belongs to, not of the
store, so the key graph's nodes are ``(lineage, key)`` pairs and an edge joins
only nodes of one lineage. The lineage is what the engine recorded about the
entry, never what the document says about itself: its depositor, its source
type, its URI's scheme and authority, and its collection root
(:func:`_document_lineage`). A document from another repository or host that
declares it supersedes ``0017`` therefore supersedes nothing of the owner's,
and a conflict between the two falls through to the trust rung.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core import bounded_yaml
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
    Aliases count toward a node budget linear in the block, so a short header
    cannot expand into a large value (security review 2026-10-26-1, W7).
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
        parsed = bounded_yaml.load(yaml_block) if yaml_block.strip() else None
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalise_adr_id(value: Any) -> str | None:
    """``"0017"`` / ``17`` / an ``ADR``-prefixed string → ``"adr:0017"`` (None if unusable).

    Only a string or an integer is an id; any other value (a list, a mapping,
    a boolean) is unusable rather than stringified.
    """
    if isinstance(value, bool) or not isinstance(value, str | int):
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
    doc_type = meta.get("type")
    if not isinstance(doc_type, str) or doc_type.strip().upper() != "ADR":
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


#: One document collection: who deposited it, as what, from where.
Lineage = tuple[str, str, str, str]

#: A node of the supersession graph: a document key inside its collection.
_Node = tuple[Lineage, str]


def _document_lineage(row: CorpusEntryRow) -> Lineage:
    """The collection an entry's document belongs to, from what the engine recorded.

    ``(deposited_by, source_type, scheme://authority, collection root)``. The
    collection root is the directory above the one holding the document, so an
    ADR tree whose lifecycle subdirectories (``active/``, ``superseded/``) sit
    side by side is one collection, while another checkout, repository or host
    is another. An entry with no URI-R has an empty root, and is one collection
    with every other such entry of its depositor and type.
    """
    parsed = urlparse(row.uri_r or "")
    path = PurePosixPath(unquote(parsed.path)) if parsed.path else None
    root = str(path.parent.parent) if path is not None else ""
    return (row.deposited_by, row.source_type, f"{parsed.scheme}://{parsed.netloc}", root)


async def _node_of(session: AsyncSession, entry_id: str) -> _Node | None:
    row = await session.get(CorpusEntryRow, entry_id)
    if row is None or row.document_supersession_json is None:
        return None
    rel = DocumentRelation.from_json(row.document_supersession_json)
    return (_document_lineage(row), rel.key) if rel is not None else None


async def _relation_rows(
    session: AsyncSession,
) -> list[tuple[str, Lineage, DocumentRelation]]:
    """``(entry_id, lineage, relation)`` for every entry that declares a relation."""
    result = await session.execute(
        select(CorpusEntryRow).where(CorpusEntryRow.document_supersession_json.is_not(None))
    )
    rows: list[tuple[str, Lineage, DocumentRelation]] = []
    for row in result.scalars().all():
        raw = row.document_supersession_json
        rel = DocumentRelation.from_json(raw) if raw else None
        if rel is not None:
            rows.append((row.entry_id, _document_lineage(row), rel))
    return rows


def _graph_of(rows: list[tuple[str, Lineage, DocumentRelation]]) -> dict[_Node, set[_Node]]:
    """Forward adjacency ``superseding node → {superseded node, …}``, per lineage.

    Merges edges from both annotation directions (``supersedes`` on the new
    doc, ``superseded_by`` on the old doc) so the graph is the same regardless
    of which side annotated. Every edge stays inside the lineage of the entry
    that declared it, so no collection can add an edge to another's graph.
    """
    graph: dict[_Node, set[_Node]] = {}
    for _entry_id, lineage, rel in rows:
        for superseded in rel.supersedes:
            graph.setdefault((lineage, rel.key), set()).add((lineage, superseded))
        for superseding in rel.superseded_by:
            graph.setdefault((lineage, superseding), set()).add((lineage, rel.key))
    return graph


async def _supersession_graph(session: AsyncSession) -> dict[_Node, set[_Node]]:
    """The supersession graph over every entry carrying the column.

    Scoped to entries carrying the column, so it stays cheap at the motivating
    scale (a few hundred ADRs).
    """
    return _graph_of(await _relation_rows(session))


async def entry_supersedes(
    session: AsyncSession,
    *,
    superseding_entry_id: str,
    superseded_entry_id: str,
) -> bool:
    """True if ``superseding_entry_id``'s document (transitively) supersedes the other's.

    Resolves each entry to its document node, then walks the supersession
    graph forward from the candidate superseder. Returns ``False`` when either
    entry carries no genre relation, when the two belong to different
    collections (:func:`_document_lineage`), when they share a key (same
    document, not a supersession), or when no path connects them. Cycle-safe
    via a visited set.
    """
    src = await _node_of(session, superseding_entry_id)
    dst = await _node_of(session, superseded_entry_id)
    if src is None or dst is None or src[0] != dst[0] or src == dst:
        return False
    graph = await _supersession_graph(session)
    if src not in graph:
        return False
    seen: set[_Node] = {src}
    queue: deque[_Node] = deque(graph[src])
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
    rows = await _relation_rows(session)
    node_to_entries: dict[_Node, list[str]] = {}
    entry_node: dict[str, _Node] = {}
    for entry_id, lineage, rel in rows:
        node = (lineage, rel.key)
        node_to_entries.setdefault(node, []).append(entry_id)
        entry_node[entry_id] = node
    if not entry_node:
        return []

    graph = _graph_of(rows)
    pairs: list[tuple[str, str]] = []
    for sup_entry_id, src_node in entry_node.items():
        if src_node not in graph:
            continue
        # BFS forward from src_node over the transitive supersession graph,
        # which never leaves the source's lineage.
        seen: set[_Node] = {src_node}
        queue: deque[_Node] = deque(graph[src_node])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            for sub_entry_id in node_to_entries.get(node, ()):
                if sub_entry_id != sup_entry_id:
                    pairs.append((sup_entry_id, sub_entry_id))
            queue.extend(graph.get(node, ()))
    return pairs
