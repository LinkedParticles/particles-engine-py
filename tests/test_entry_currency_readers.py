# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""No reader of a ``MUTABLE`` entry's current snapshot bypasses ``get_entry_currency``.

An entry's current generation may be an ``ABSENT`` record: the vault note was
deleted. A reader that picks "the newest extracted RESPONSE" for itself reads
that note as still current, which is how ``reindex`` would have resurrected a
deleted note's claims. Every reader is routed through the helper,
and this scan fails on a new one.

Two signatures are flagged in ``particles/``:

* a function whose code names both ``superseded_by_snapshot_id`` and
  ``COMPLETE``: the shape of a hand-written current-generation query;
* a call of ``get_latest_completed_snapshot_id`` or
  ``get_newest_response_snapshot``, which mean "newest extracted RESPONSE",
  never "current".

Each allowed site carries the reason it is not a currency read. Adding one is a
decision: route the new reader through ``get_entry_currency`` instead, unless
it really asks a different question.
"""

from __future__ import annotations

import ast
from pathlib import Path

from particles.corpus import store as corpus_store

#: The installed package, found through an Engine module so the scan runs where
#: the readers it audits are (this file never ships in the Client tree).
PACKAGE = Path(corpus_store.__file__).resolve().parent.parent

_HAND_WRITTEN_QUERY: dict[tuple[str, str], str] = {
    ("corpus/store.py", "get_entry_currency"): "the helper itself",
    ("corpus/store.py", "get_latest_completed_snapshot_id"): (
        "defines 'the newest extracted RESPONSE', the last text a deleted note held"
    ),
    ("corpus/store.py", "list_pending_snapshots_for_catchup"): (
        "orders the work queue by owed-since; reads no currency"
    ),
    ("corpus/store.py", "mark_snapshot_superseded"): "writes the collapse mark",
    ("corpus/store.py", "entry_ids_ever_extracted"): (
        "orders extract --all-pending's queue by whether the store has read an entry "
        "at all; reads no currency"
    ),
    ("corpus/store.py", "list_replaced_mutable_snapshot_ids"): (
        "the empty-COMPLETE lint's 'has a newer extracted sibling' test"
    ),
    ("operations/utility_exposure.py", "stated_by"): (
        "MUTABLE entries go through the helper; its own query reads non-MUTABLE entries only"
    ),
}

_NEWEST_RESPONSE_CALLERS: dict[tuple[str, str], str] = {
    ("operations/reindex.py", "_gather_named"): (
        "resolves a named entry's last text; _drop_absent_entries skips absent entries after"
    ),
    ("operations/projection/memory_region.py", "fold_verdicts"): (
        "the fold over memory files, which no vault walk ever marks absent"
    ),
}

_NEWEST_RESPONSE_FUNCTIONS = {"get_latest_completed_snapshot_id", "get_newest_response_snapshot"}


def _functions() -> list[tuple[str, str, ast.AST]]:
    found: list[tuple[str, str, ast.AST]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.append((rel, node.name, node))
    return found


def _names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            out.add(child.attr)
        elif isinstance(child, ast.Name):
            out.add(child.id)
    return out


def test_no_hand_written_current_generation_query() -> None:
    flagged = {
        (rel, name)
        for rel, name, node in _functions()
        if {"superseded_by_snapshot_id", "COMPLETE"} <= _names(node)
    }
    unexpected = sorted(flagged - set(_HAND_WRITTEN_QUERY))
    assert not unexpected, (
        "A function reads a MUTABLE entry's current snapshot without get_entry_currency, "
        f"so it would read a deleted vault note as current: {unexpected}"
    )


def test_no_new_caller_reads_the_newest_response_as_current() -> None:
    flagged = {
        (rel, name)
        for rel, name, node in _functions()
        if name not in _NEWEST_RESPONSE_FUNCTIONS and _names(node) & _NEWEST_RESPONSE_FUNCTIONS
    }
    unexpected = sorted(flagged - set(_NEWEST_RESPONSE_CALLERS))
    assert not unexpected, (
        "A new caller of the newest-extracted-RESPONSE helpers; if it means 'current', "
        f"read get_entry_currency instead: {unexpected}"
    )


def test_the_allowlists_name_real_sites() -> None:
    """A stale entry would let a renamed reader through unnoticed."""
    present = {(rel, name) for rel, name, _ in _functions()}
    assert set(_HAND_WRITTEN_QUERY) <= present
    assert set(_NEWEST_RESPONSE_CALLERS) <= present
