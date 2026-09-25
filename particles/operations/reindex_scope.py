# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure scope decisions for the §9.5 Reindex operation (D2).

``operations/reindex.py`` gathers the store reads (prefix matches, latest
COMPLETE snapshots, selector matches, FAILED/PENDING pairs, stale-schema
pairs, the collapse plan) and hands them here as plain values. Everything in
this module is a function of its arguments: no session, no clock, no
filesystem, so the scope rules, including the intersection, are
testable without a database.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence, Set
from dataclasses import dataclass
from typing import Literal, NamedTuple

#: Length of a full corpus entry id (a UUID string). Anything shorter is a
#: display prefix the shell must resolve against the store.
FULL_ENTRY_ID_LENGTH = 36

Pair = tuple[str, str]


class PrefixResolution(NamedTuple):
    """The outcome of resolving one operator-supplied entry id."""

    entry_id: str | None
    problem: Literal["ambiguous", "not_found"] | None


def is_prefix(raw_id: str) -> bool:
    """True when ``raw_id`` is shorter than a full id and needs a prefix lookup."""
    return len(raw_id) < FULL_ENTRY_ID_LENGTH


def resolve_prefix(raw_id: str, matches: Sequence[str]) -> PrefixResolution:
    """Resolve ``raw_id`` given the entry ids its prefix lookup ``matches``.

    A full-length id resolves to itself and ``matches`` is ignored (the shell
    does not look it up). A prefix resolves only when exactly one entry
    matches; more than one is ambiguous and none is not found, and both are
    skipped rather than guessed.
    """
    if not is_prefix(raw_id):
        return PrefixResolution(raw_id, None)
    if len(matches) == 1:
        return PrefixResolution(matches[0], None)
    if matches:
        return PrefixResolution(None, "ambiguous")
    return PrefixResolution(None, "not_found")


def union_selectors(*selections: Iterable[Pair] | None) -> set[Pair] | None:
    """Union of the pairs selected by the particle-matching flags.

    Each argument is one flag's matches, or ``None`` when that flag was not
    passed. Returns ``None``, distinct from an empty set, when no flag was
    passed at all, so "no filter requested" never reads as "filter requested,
    matched nothing".
    """
    requested = [s for s in selections if s is not None]
    if not requested:
        return None
    return {pair for selection in requested for pair in selection}


@dataclass(frozen=True)
class ReindexScope:
    """The decided scope: the snapshot pairs to re-extract, and any narrowing."""

    pairs: list[Pair]
    #: ``(kept, named)`` when intersecting the named entries with the
    #: particle-selector union dropped any of them; ``None`` otherwise.
    narrowed: tuple[int, int] | None = None


def decide_reindex_scope(
    *,
    named: Sequence[Pair] | None,
    selected: Set[Pair] | None,
    failed_or_pending: Sequence[Pair],
    collapsed: Set[str],
    stale_schema: Sequence[Pair],
) -> ReindexScope:
    """Decide the ``(entry_id, snapshot_id)`` pairs a reindex re-extracts.

    Args:
        named: the named entries resolved to their latest COMPLETE snapshot,
            or ``None`` when the operator named no entries (auto-discovery).
        selected: the particle-selector union (``union_selectors``), or
            ``None`` when no particle-matching flag was passed.
        failed_or_pending: FAILED/PENDING snapshot pairs; empty unless the
            auto-discovery path asked for them.
        collapsed: snapshot ids the collapse marks as superseded,
            which no scope retries.
        stale_schema: pairs whose ACTIVE particles carry an old schema version.

    **Named entries** are intersected with ``selected`` when both are given.
    Before that fix the named branch discarded every other flag,
    which widened a superseding verb. The intersection errs narrower, and any
    entry it drops is reported through ``narrowed``. The store-wide unions
    (FAILED/PENDING, stale schema) never apply here: naming entries must not
    hand the operator the rest of the store.

    **Auto-discovery** is the union of the uncollapsed FAILED/PENDING pairs,
    the selector matches and the stale-schema pairs.
    """
    if named is not None:
        deduped = list(dict.fromkeys(named))
        if selected is None:
            return ReindexScope(deduped)
        kept = [pair for pair in deduped if pair in selected]
        narrowed = (len(kept), len(deduped)) if len(kept) != len(deduped) else None
        return ReindexScope(kept, narrowed)

    scope: list[Pair] = [pair for pair in failed_or_pending if pair[1] not in collapsed]
    if selected is not None:
        scope.extend(selected)
    scope.extend(stale_schema)
    return ReindexScope(list(set(scope)))
