# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Forget a recorded rule-source root.

``particles rules forget ROOT…`` removes rows from ``rule_source_roots``, the
roots the local refresh reads rule files under. It removes rows
only: no belief is retracted, no corpus entry is touched, and no entry's
``fetch_policy`` changes. Before anything is deleted, every ``LAZY`` ``file://``
entry under each root is judged twice by the refresh's own predicate, with the
recorded rows as they are and without the rows being forgotten, so the
operator sees which sources stop being refreshed and which another route still
admits. Each removed row writes one ``RULE_SOURCE_ROOT_FORGOTTEN`` event.

Gather (the rows and the entries), decide (:func:`plan_forget`, pure), apply
(:func:`forget_rule_source_roots` deletes and records).
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import is_broad_root
from particles.corpus.fetch import admission_route, local_refusal, path_from_file_uri
from particles.corpus.rule_roots import (
    RuleSourceRootRow,
    delete_rule_source_root,
    find_rule_source_root,
    list_rule_source_roots,
)
from particles.corpus.store import CorpusEntryRow, list_refreshable_local_entries
from particles.corpus.worktree import verified_main_checkout
from particles.store.event_store import OperatorEventType, record_event

#: The actor the CLI verb records.
FORGET_ACTOR = "cli:rules-forget"


@dataclass(frozen=True)
class LocalEntry:
    """One ``LAZY`` ``file://`` entry, as the decision needs it."""

    entry_id: str
    path: Path
    deposited_by: str | None


@dataclass(frozen=True)
class RootForgetPlan:
    """What forgetting one recorded root does to the entries under it."""

    path: str
    """The stored key, verbatim."""
    recorded_by: str
    recorded_at: datetime
    becomes_refused: int = 0
    """Entries the refresh reads today and refuses once the row is gone."""
    still_admitted: dict[str, int] = field(default_factory=dict)
    """Entries another route still admits, counted by that route's name."""
    already_refused: int = 0
    """Entries under the root the refresh refuses either way."""

    @property
    def still_admitted_total(self) -> int:
        """Every entry another route still admits."""
        return sum(self.still_admitted.values())


@dataclass
class ForgetReport:
    """The outcome of one ``rules forget`` call."""

    plans: list[RootForgetPlan] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    """Arguments that name no recorded row."""
    dry_run: bool = False


RefusalFn = Callable[[Path, Sequence[Path], str | None], object | None]
RouteFn = Callable[[Path, Sequence[Path], str | None], str | None]


def plan_forget(
    rows: Sequence[RuleSourceRootRow],
    recorded: Sequence[Path],
    entries: Sequence[LocalEntry],
    *,
    refusal: RefusalFn,
    route: RouteFn,
    derived: Mapping[str, Sequence[Path]] | None = None,
) -> list[RootForgetPlan]:
    """Judge each entry under each root with and without the rows being forgotten.

    Pure over its arguments: ``refusal`` and ``route`` are the refresh's own
    predicates, passed in so the decision reads no config or filesystem of its
    own beyond what they read. The "without" set drops every row being
    forgotten at once, since that is the state the call leaves.

    ``derived`` maps a stored key to the roots that row admits beyond its own
    directory: the verified main checkout of a recorded worktree root.
    An entry under one of them that no forgotten root holds is counted
    against the first row whose derived root holds it, when the refresh admits
    it today, so the counts name every source the forget stops refreshing
    (security review 2026-10-25-1, Q49). Such an entry the refresh already
    refuses is not counted, since the row never reached it.
    """
    forgotten = {row.path for row in rows}
    remaining = [r for r in recorded if str(r) not in forgotten]
    derived = derived or {}

    def _under(path: Path, root: Path) -> bool:
        return path == root or path.is_relative_to(root)

    claimed: set[str] = {
        entry.entry_id
        for entry in entries
        if any(_under(entry.path, Path(row.path)) for row in rows)
    }
    plans: list[RootForgetPlan] = []
    for row in rows:
        root = Path(row.path)
        refused = 0
        already = 0
        admitted: Counter[str] = Counter()
        for entry in entries:
            if _under(entry.path, root):
                if refusal(entry.path, recorded, entry.deposited_by) is not None:
                    already += 1
                    continue
            elif entry.entry_id not in claimed and any(
                _under(entry.path, d) for d in derived.get(row.path, ())
            ):
                claimed.add(entry.entry_id)
                if refusal(entry.path, recorded, entry.deposited_by) is not None:
                    continue
            else:
                continue
            if (name := route(entry.path, remaining, entry.deposited_by)) is None:
                refused += 1
            else:
                admitted[name] += 1
        plans.append(
            RootForgetPlan(
                path=row.path,
                recorded_by=row.recorded_by,
                recorded_at=row.recorded_at,
                becomes_refused=refused,
                still_admitted=dict(sorted(admitted.items())),
                already_refused=already,
            )
        )
    return plans


async def _local_entries(session: AsyncSession) -> list[LocalEntry]:
    out: list[LocalEntry] = []
    for entry_id, uri_r in await list_refreshable_local_entries(session):
        try:
            raw = path_from_file_uri(uri_r)
        except ValueError:
            continue
        row = await session.get(CorpusEntryRow, entry_id)
        out.append(
            LocalEntry(
                entry_id=entry_id,
                path=Path(os.path.realpath(raw)),
                deposited_by=row.deposited_by if row is not None else None,
            )
        )
    return out


def _refusal(path: Path, recorded: Sequence[Path], deposited_by: str | None) -> object | None:
    return local_refusal(path, recorded_roots=recorded, deposited_by=deposited_by)


def _route(path: Path, recorded: Sequence[Path], deposited_by: str | None) -> str | None:
    return admission_route(path, recorded_roots=recorded, deposited_by=deposited_by)


def _derived_roots(rows: Sequence[RuleSourceRootRow]) -> dict[str, list[Path]]:
    """Each row's verified main checkout, the root the refresh derives from it.

    Derived the way :func:`~particles.corpus.fetch.rule_file_roots` derives it,
    so a broad main checkout, which admits nothing, is left out.
    """
    out: dict[str, list[Path]] = {}
    for row in rows:
        main = verified_main_checkout(Path(row.path))
        if main is not None and not is_broad_root(main):
            out[row.path] = [main]
    return out


async def forget_rule_source_roots(
    session: AsyncSession,
    paths: Sequence[str],
    *,
    dry_run: bool = False,
    actor: str = FORGET_ACTOR,
) -> ForgetReport:
    """Remove the recorded rows ``paths`` name, reporting what each removal changes.

    An argument that names no row lands in :attr:`ForgetReport.unmatched` and
    changes nothing. With ``dry_run`` nothing is deleted and no event is
    written. Does not commit: the caller commits the deletes and their events
    together.
    """
    report = ForgetReport(dry_run=dry_run)
    rows: list[RuleSourceRootRow] = []
    for raw in paths:
        row = await find_rule_source_root(session, raw)
        if row is None:
            report.unmatched.append(raw)
        elif row.path not in {r.path for r in rows}:
            rows.append(row)
    if not rows:
        return report

    recorded = await list_rule_source_roots(session)
    entries = await _local_entries(session)
    report.plans = plan_forget(
        rows, recorded, entries, refusal=_refusal, route=_route, derived=_derived_roots(rows)
    )
    if dry_run:
        return report

    by_path = {row.path: row for row in rows}
    for plan in report.plans:
        await delete_rule_source_root(session, by_path[plan.path])
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.RULE_SOURCE_ROOT_FORGOTTEN,
            payload={
                "path": plan.path,
                "recorded_by": plan.recorded_by,
                "recorded_at": plan.recorded_at.isoformat(),
                "becomes_refused": plan.becomes_refused,
                "still_admitted": plan.still_admitted_total,
                "still_admitted_by_route": plan.still_admitted,
            },
        )
    return report
