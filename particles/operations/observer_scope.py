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
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.observer_scope import (
    UNATTRIBUTED_MARKER,
    EntryScope,
    classify_entry,
    project_keys,
    project_tag,
)
from particles.core.schema import SourceType
from particles.corpus.store import (
    add_entry_tags,
    get_entry,
    list_entry_attribution_rows,
    list_entry_tag_rows,
    remove_entry_tags,
)
from particles.store.event_store import EventRefKind, OperatorEventType, list_events, record_event
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
    marked: list[str] = field(default_factory=list)
    """Keyless agent writes marked ``scope:unattributed``, which neither the
    resolver nor ``default_key`` attributes: their project is
    not known, so only ``--assign`` or ``widen`` places them."""
    entries_per_key: Counter[str] = field(default_factory=Counter)
    keys_by_entry: dict[str, frozenset[str]] = field(default_factory=dict)
    """Every keyed entry's keys after the pass — what a caller that knows which
    keys still name a live project needs to find entries no session can see."""


def plan_rescope(
    rows: Iterable[tuple[str, str | None, list[str]]],
    *,
    key_for: KeyResolver,
    default_key: str | None,
    harness_tags: Collection[str],
    dry_run: bool = False,
) -> RescopeReport:
    """Decide what a ``rescope`` pass adds, from the entries' tag rows alone.

    The pure half of :func:`rescope` (D2): ``rows`` are
    ``(entry_id, uri_r, tags)``, and the report's ``added`` is the whole plan.
    """
    report = RescopeReport(dry_run=dry_run)
    for entry_id, uri_r, tags in rows:
        report.examined += 1
        keys = set(project_keys(tags))
        if not keys and UNATTRIBUTED_MARKER in tags:
            report.marked.append(entry_id)
            continue
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
        if keys:
            report.entries_per_key.update(keys)
            report.keys_by_entry[entry_id] = frozenset(keys)
        elif classify_entry(tags, harness_tags) is EntryScope.UNATTRIBUTED:
            report.unattributed.append(entry_id)
    return report


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
    what lets a project observer engage on this store at all. Its payload
    names every ``(entry, key)`` pair it added, so the store's
    attribution history can be read back from the event log.

    Does not commit; the caller owns the transaction.
    """
    report = plan_rescope(
        await list_entry_tag_rows(session),
        key_for=key_for,
        default_key=default_key,
        harness_tags=get_config().observer_scope.harness_tags,
        dry_run=dry_run,
    )
    if dry_run:
        return report
    for entry_id, key in report.added:
        await add_entry_tags(session, entry_id, [project_tag(key)])
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.OBSERVER_SCOPE_RESCOPED,
        payload={
            "examined": report.examined,
            "keys_added": len(report.added),
            "unattributed": len(report.unattributed),
            "marked_unattributed": len(report.marked),
            "default_key": default_key,
            "added": [[entry_id, key] for entry_id, key in report.added],
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


def agent_identities(extra: Iterable[str] = ()) -> list[str]:
    """The depositor identities whose keyless writes are agent material.

    The native MCP surface's and the memory-server facade's configured
    asserters, plus any the operator names (``--identity``) for writes made
    under an identity since reconfigured.
    """
    cfg = get_config().mcp
    return list(
        dict.fromkeys([cfg.write.asserter_identity, cfg.memory_compat.asserter_identity, *extra])
    )


@dataclass
class MarkReport:
    """What one ``rescope --mark-agent-writes`` pass found and did."""

    dry_run: bool
    identities: list[str]
    marked: list[tuple[str, str]] = field(default_factory=list)
    """``(entry_id, identity)`` for every entry marked (or that would be)."""
    conversation_by_depositor: Counter[str] = field(default_factory=Counter)
    """Every keyless, non-harness ``CONVERSATION`` entry, by depositor: what the
    identity rule can and cannot see, so a miss is shown rather than silent."""

    @property
    def unmatched(self) -> dict[str, int]:
        return {
            who: n
            for who, n in self.conversation_by_depositor.items()
            if who not in self.identities
        }


def plan_mark_agent_writes(
    rows: Iterable[tuple[str, list[str], str, str]],
    *,
    identities: Collection[str],
    harness_tags: Collection[str],
    dry_run: bool = False,
) -> MarkReport:
    """Which global entries are agent writes, from ``(entry, tags, depositor, type)`` rows.

    An entry is agent material when it is global by omission (keyless, no
    harness tag, no marker) and its depositor is a named identity, never
    because its content looks like an agent's.
    """
    report = MarkReport(dry_run=dry_run, identities=list(identities))
    for entry_id, tags, deposited_by, source_type in rows:
        if classify_entry(tags, harness_tags) is not EntryScope.GLOBAL:
            continue
        if source_type == SourceType.CONVERSATION.value:
            report.conversation_by_depositor[deposited_by] += 1
        if deposited_by in identities:
            report.marked.append((entry_id, deposited_by))
    return report


async def mark_agent_writes(
    session: AsyncSession,
    *,
    identities: Collection[str],
    dry_run: bool = False,
    actor: str,
) -> MarkReport:
    """Mark each existing keyless agent write ``scope:unattributed``.

    Nothing is reclassified until this runs: the marker on a new agent write
    is set at deposit, while an old one changes only here, one recorded
    ``OBSERVER_SCOPE_MARKED_UNATTRIBUTED`` event per entry. Does not commit.
    """
    report = plan_mark_agent_writes(
        await list_entry_attribution_rows(session),
        identities=identities,
        harness_tags=get_config().observer_scope.harness_tags,
        dry_run=dry_run,
    )
    if dry_run:
        return report
    for entry_id, identity in report.marked:
        await add_entry_tags(session, entry_id, [UNATTRIBUTED_MARKER])
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.OBSERVER_SCOPE_MARKED_UNATTRIBUTED,
            refs=[(EventRefKind.CORPUS_ENTRY, entry_id)],
            payload={"identity": identity},
        )
    return report


async def unmark(session: AsyncSession, entry_id: str, *, actor: str) -> bool:
    """Reverse one recorded ``--mark-agent-writes`` mark. Returns ``False`` if already reversed.

    Refuses an entry no ``OBSERVER_SCOPE_MARKED_UNATTRIBUTED`` event names: a
    write marked at deposit, or by ``--withdraw --leave-unattributed``, was not
    marked by mistake, and this verb restores an earlier class rather than
    deciding a new one. Also refuses a marked entry that has
    since been given a key: the marker then records where the entry began.
    """
    if await get_entry(session, entry_id) is None:
        raise ValueError(f"No corpus entry {entry_id!r}.")
    marks = await list_events(
        session,
        ref_kind=EventRefKind.CORPUS_ENTRY,
        ref_id=entry_id,
        event_type=OperatorEventType.OBSERVER_SCOPE_MARKED_UNATTRIBUTED,
        limit=1,
    )
    if not marks:
        raise ValueError(
            f"Entry {entry_id!r} was not marked by `rescope --mark-agent-writes`, so there is "
            "no mark to reverse. Nothing was changed."
        )
    entry = await get_entry(session, entry_id)
    if entry is not None and project_keys(entry.tags):
        # A key outranks the marker, which then stays as the record that the
        # entry began as an agent write (decision 2).
        raise ValueError(
            f"Entry {entry_id!r} has a project key now, so its marker stays as the record "
            "that it began as an agent write. Withdraw the key first if the mark was wrong. "
            "Nothing was changed."
        )
    removed = await remove_entry_tags(session, entry_id, [UNATTRIBUTED_MARKER])
    if removed:
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.OBSERVER_SCOPE_UNMARKED,
            refs=[(EventRefKind.CORPUS_ENTRY, entry_id)],
        )
    return bool(removed)


async def withdraw_key(
    session: AsyncSession,
    entry_id: str,
    key: str,
    *,
    actor: str,
    leave_unattributed: bool = False,
) -> bool:
    """Remove one project key from one entry: the repair for a wrong key.

    Returns ``False`` if the entry did not carry the key. **Refuses to remove
    an entry's last key**, whatever its class: a keyed rule file carries no
    harness tag, so emptying it would make it global, visible in every
    project, through a verb meant for repair. With ``leave_unattributed`` the
    last key may go and the entry is left unattributed: a harness entry keeps
    its harness tag, and any other entry gains the ``scope:unattributed``
    marker. No correction verb makes an entry global; that is ``widen``.
    """
    entry = await get_entry(session, entry_id)
    if entry is None:
        raise ValueError(f"No corpus entry {entry_id!r}.")
    keys = project_keys(entry.tags)
    if key not in keys:
        return False
    marked = False
    if keys == {key}:
        if not leave_unattributed:
            raise ValueError(
                f"{key!r} is the last project key on entry {entry_id!r}; removing it would "
                "change who sees the entry rather than repair it. Assign the right key first, "
                "or pass --leave-unattributed to leave the entry in view for no project. "
                "Nothing was changed."
            )
        harness_tags = get_config().observer_scope.harness_tags
        without = [tag for tag in entry.tags if tag != project_tag(key)]
        if classify_entry(without, harness_tags) is EntryScope.GLOBAL:
            marked = bool(await add_entry_tags(session, entry_id, [UNATTRIBUTED_MARKER]))
    await remove_entry_tags(session, entry_id, [project_tag(key)])
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.OBSERVER_SCOPE_KEY_WITHDRAWN,
        refs=[(EventRefKind.CORPUS_ENTRY, entry_id)],
        payload={"withdrawn": key, **({"marked": True} if marked else {})},
    )
    return True


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
