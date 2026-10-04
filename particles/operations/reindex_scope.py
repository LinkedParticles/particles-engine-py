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

from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Literal, NamedTuple

from particles.extraction.components import ComponentRecord, ComponentTable

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


# ---------------------------------------------------------------------------
# select only the snapshots whose exercised components changed.
# ---------------------------------------------------------------------------

#: Whether ``reindex --only-changed-components`` may narrow a scope. Off until
#: the first ``EXTRACTOR_VERSION`` bump after the component record shipped
#: (1.168.14) has run over a store: every snapshot extracted before that has no
#: record, reads as "every component exercised", and would be selected anyway,
#: so the flag could only mislead an operator into thinking it narrowed
#: something. Turning it on is the action trigger names.
ONLY_CHANGED_COMPONENTS_ENABLED = False

#: What a refused ``--only-changed-components`` says, on every surface.
ONLY_CHANGED_COMPONENTS_REFUSAL = (
    "--only-changed-components is not enabled yet. Snapshots record "
    "the extraction components they exercised from 1.168.14 on, and a snapshot "
    "extracted earlier has no record, so it counts as having exercised every "
    "component and would be re-extracted regardless. The flag is turned on "
    "once an extractor version bump has run over a stamped store. Until then, "
    "run the full version scope, or `particles reindex --estimate` to measure "
    "how much of it a bump changes."
)


def component_change_reasons(
    tables: Mapping[str, ComponentTable], record: ComponentRecord | None
) -> tuple[str, ...]:
    """Why a snapshot's extraction could differ under the current components.

    Empty means it could not: every component the snapshot's extraction
    exercised still has the digest it had, and no component the current code
    would add to its extraction is new to it. Pure, and conservative in every
    branch: anything the record cannot vouch for is a reason.

    Args:
        tables: each extractor's current :class:`ComponentTable`, by
            registered id, with any pipeline-level component (the subject
            gate) merged in.
        record: the snapshot's stored record, or ``None`` when it has none.

    Reasons, in order:

    1. **No record**, or an incomplete one (a carried-forward claim from a
       snapshot with none): nothing is known, so every component counts as
       exercised.
    2. **No table** for the record's extractor: nothing to compare against.
    3. An exercised component whose **digest changed**, or that the current
       table **no longer has** (removed, renamed, or disabled in config).
    4. A component the current table puts on **every** extraction that this
       one did not exercise (a new rule, or one newly enabled in config).
    5. A source-dependent component that is **new since the record** (absent
       from its ``available``): the record cannot say whether this source
       would reach it.

    The comparison is against the snapshot's own record, not against the
    previous version's table, so it is cumulative: a snapshot skipped by one
    bump is compared against everything that changed since it was extracted,
    the next time any bump asks.
    """
    if record is None:
        return ("no component record",)
    if not record.complete:
        return ("incomplete component record",)
    table = tables.get(record.extractor or "")
    if table is None:
        return (f"no component table for extractor {record.extractor!r}",)
    reasons: list[str] = []
    for name, digest in sorted(record.exercised.items()):
        current = table.digests.get(name)
        if current is None:
            reasons.append(f"{name}: no longer in the extractor's components")
        elif current != digest:
            reasons.append(f"{name}: changed")
    reasons.extend(
        f"{name}: now on every extraction"
        for name in sorted(table.always - record.exercised.keys())
    )
    known = set(record.available) | set(record.exercised)
    reasons.extend(
        f"{name}: new since the record"
        for name in sorted(set(table.digests) - table.always - known)
    )
    return tuple(reasons)


@dataclass(frozen=True)
class ComponentSelection:
    """A scope split by :func:`select_changed_components`."""

    #: Pairs whose extraction could differ, with the reasons for each.
    kept: dict[Pair, tuple[str, ...]]
    #: Pairs whose every exercised component is unchanged.
    skipped: list[Pair]


def select_changed_components(
    pairs: Sequence[Pair],
    records: Mapping[str, ComponentRecord | None],
    tables: Mapping[str, ComponentTable],
) -> ComponentSelection:
    """Split a version scope into the snapshots a bump could change and the rest.

    ``records`` is keyed by snapshot id; a snapshot missing from it has no
    record. A pair is kept when :func:`component_change_reasons` gives any
    reason, and skipped otherwise.

    **A skip is never silently permanent.** Skipping removes the pair from
    this run's scope and writes nothing: the snapshot's particles keep the
    ``extractor_version`` they were stamped with, so a later
    ``--extractor-version`` scope over that version selects them again, and
    the comparison then runs against everything that changed since their
    record (see :func:`component_change_reasons`). The caller must report the
    skipped count, never drop it from the plan line.
    """
    kept: dict[Pair, tuple[str, ...]] = {}
    skipped: list[Pair] = []
    for pair in pairs:
        reasons = component_change_reasons(tables, records.get(pair[1]))
        if reasons:
            kept[pair] = reasons
        else:
            skipped.append(pair)
    return ComponentSelection(kept=kept, skipped=skipped)
