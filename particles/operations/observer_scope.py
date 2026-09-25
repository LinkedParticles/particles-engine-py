# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer-scope administration — ``rescope``, ``assign``, ``widen`` (§2/§3).

The write half of observer scope is small on purpose. Keys are stamped on
corpus entries at deposit; what is left for an operator is (a) bringing a store
that predates the stamping up to date, and (b) the standing judgement that a
belief applies everywhere.

Everything here is **additive**: a rescope only ever adds a ``project:`` tag
beside whatever an entry already carries, so the history of how a source was
first attributed is never lost, and a second run changes nothing.

How an entry maps to a project key is a harness's path convention, which the
Engine does not know: the caller supplies ``key_for``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.observer_scope import EntryScope, classify_entry, project_keys, project_tag
from particles.corpus.store import add_entry_tags, get_entry, list_entry_tag_rows
from particles.store.event_store import EventRefKind, OperatorEventType, record_event
from particles.store.observer_scope_store import ScopeTarget, add_widening, remove_widening
from particles.store.particle_store import get_particle

#: ``(entry_id, uri_r, tags) -> project key``, or ``None`` to leave the entry alone.
KeyResolver = Callable[[str, str | None, list[str]], str | None]


@dataclass
class RescopeReport:
    """What one ``rescope`` pass found and did."""

    dry_run: bool
    examined: int = 0
    added: list[tuple[str, str]] = field(default_factory=list)
    """``(entry_id, key)`` for every key added (or that would be, on a dry run)."""
    unattributed: list[str] = field(default_factory=list)
    """Harness-harvested entries still carrying no key: in view for no project."""
    entries_per_key: Counter[str] = field(default_factory=Counter)
    keys_by_entry: dict[str, frozenset[str]] = field(default_factory=dict)
    """Every keyed entry's keys after the pass — what a caller that knows which
    keys still name a live project needs to find entries no session can see."""


# Known deviation: decision logic is interleaved with I/O in this function. Extract it with the
# next substantive change here (D2).
async def rescope(
    session: AsyncSession,
    *,
    key_for: KeyResolver,
    default_key: str | None = None,
    dry_run: bool = False,
    actor: str = "rescope",
) -> RescopeReport:
    """Add the canonical project key to every entry ``key_for`` can attribute.

    ``default_key`` is given to whatever is still unattributed afterwards — on
    a one-project machine that is one flag. Records one
    ``OBSERVER_SCOPE_RESCOPED`` event even when nothing changed: that event is
    what lets a project observer engage on this store at all.

    Does not commit; the caller owns the transaction.
    """
    harness_tags = get_config().observer_scope.harness_tags
    report = RescopeReport(dry_run=dry_run)
    for entry_id, uri_r, tags in await list_entry_tag_rows(session):
        report.examined += 1
        keys = set(project_keys(tags))
        key = key_for(entry_id, uri_r, tags)
        if (
            key is None
            and default_key
            and classify_entry(tags, harness_tags) is (EntryScope.UNATTRIBUTED)
        ):
            key = default_key
        if key is not None and key not in keys:
            report.added.append((entry_id, key))
            keys.add(key)
            if not dry_run:
                await add_entry_tags(session, entry_id, [project_tag(key)])
        if keys:
            report.entries_per_key.update(keys)
            report.keys_by_entry[entry_id] = frozenset(keys)
        elif classify_entry(tags, harness_tags) is EntryScope.UNATTRIBUTED:
            report.unattributed.append(entry_id)

    if not dry_run:
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.OBSERVER_SCOPE_RESCOPED,
            payload={
                "examined": report.examined,
                "keys_added": len(report.added),
                "unattributed": len(report.unattributed),
                "default_key": default_key,
            },
        )
    return report


async def assign_key(session: AsyncSession, entry_id: str, key: str, *, actor: str) -> bool:
    """Give one corpus entry a project key. Returns ``False`` if it already had it.

    Refuses a **global** entry: adding a key to a hand deposit would take it out
    of view for every other project, which is narrowing — a different
    judgement, and not one a repair verb should make as a side effect.
    """
    entry = await get_entry(session, entry_id)
    if entry is None:
        raise ValueError(f"No corpus entry {entry_id!r}.")
    harness_tags = get_config().observer_scope.harness_tags
    if classify_entry(entry.tags, harness_tags) is EntryScope.GLOBAL:
        raise ValueError(
            f"Entry {entry_id!r} is global (keyless, not harness-harvested); giving it a project "
            "key would hide it from every other project. Nothing was changed."
        )
    added = await add_entry_tags(session, entry_id, [project_tag(key)])
    if added:
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.OBSERVER_SCOPE_KEY_ASSIGNED,
            refs=[(EventRefKind.CORPUS_ENTRY, entry_id)],
            payload={"assigned": key},
        )
    return bool(added)


async def widen(
    session: AsyncSession, kind: ScopeTarget, target_id: str, *, actor: str, revoke: bool = False
) -> bool:
    """Put a belief, or a whole source, in view for every project — or take that back.

    Lens-side policy: the claim and its provenance are untouched. Returns
    whether anything changed. Operator-only by design; there is no agent
    surface for it.
    """
    if kind is ScopeTarget.PARTICLE:
        if await get_particle(session, target_id) is None:
            raise ValueError(f"No particle {target_id!r}.")
        ref_kind = EventRefKind.PARTICLE
    else:
        if await get_entry(session, target_id) is None:
            raise ValueError(f"No corpus entry {target_id!r}.")
        ref_kind = EventRefKind.CORPUS_ENTRY

    if revoke:
        changed = await remove_widening(session, kind, target_id)
        event_type = OperatorEventType.OBSERVER_SCOPE_WIDEN_REVOKED
    else:
        changed = await add_widening(session, kind, target_id, created_by=actor)
        event_type = OperatorEventType.OBSERVER_SCOPE_WIDENED
    if changed:
        await record_event(
            session, actor=actor, event_type=event_type, refs=[(ref_kind, target_id)]
        )
    return changed
