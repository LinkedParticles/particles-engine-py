# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Which host a ``file://`` corpus entry's path names.

A ``file://`` URI-R carries no host, so each such entry records
``origin_host``: the host id (:mod:`particles.host_identity`) of the machine
whose filesystem the path names. This module holds the rules around it:

- **Who stamps what** (§1.2). An in-process deposit stamps this host's id. An
  HTTP deposit runs under :func:`deposit_origin`, which the engine's routes set
  to the presenting key's ``host_id`` (``None`` for a key that records none).
- **Which entry a deposit may join** (§3.3). A deposit that names a ``file://``
  entry joins it when the entry's origin is the deposit's, adopts it when the
  entry has none, and is refused (:class:`OtherHostEntryRefused`) otherwise.
- **Counting** (§2.2, §3.7) and the one-time stamp and repair verbs (§5).

The local refresh reads only entries whose origin is this host's
(``corpus/fetch.py``), so a store served to a client on another machine, or
moved between machines, never has a same-named path read from the wrong disk.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass

from sqlalchemy import func, inspect, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from particles.core.schema import FetchPolicy
from particles.corpus.store import CorpusEntryRow
from particles.host_identity import local_host_id

log = logging.getLogger(__name__)

#: The rows a ``file://`` entry's URI-R starts with.
FILE_SCHEME = "file://"

#: ``(set, origin)``: whether a scope chose the origin, and the one it chose.
#: Unset means an in-process deposit, which stamps this host's id.
_origin: ContextVar[tuple[bool, str | None]] = ContextVar(
    "particles_deposit_origin", default=(False, None)
)
#: Whether the scope's deposit is an operator's, which may write a client's
#: ``file://`` entry whatever its origin, but never one of this engine's host.
_operator: ContextVar[bool] = ContextVar("particles_deposit_origin_operator", default=False)


@contextlib.contextmanager
def deposit_origin(host_id: str | None, *, operator: bool = False) -> Iterator[None]:
    """Stamp ``host_id`` as the origin of every ``file://`` entry a deposit in this scope makes.

    The engine's HTTP deposit routes wrap each deposit in it with the presenting
    key's recorded ``host_id``, never a value from the request.
    ``None`` stamps no origin: the entry is then read by no host until a deposit
    with an origin adopts it. ``operator`` (an operator key, which records no
    host) lets the deposit write an entry of any client's host, as an operator
    may write any principal's entry, and still never one of this engine's own
    host, which is the case security review N19 refused.
    """
    token = _origin.set((True, host_id))
    op_token = _operator.set(operator)
    try:
        yield
    finally:
        _operator.reset(op_token)
        _origin.reset(token)


def current_deposit_origin() -> str | None:
    """The origin a ``file://`` deposit in this scope stamps: the scope's, else this host's."""
    chosen, host_id = _origin.get()
    return host_id if chosen else local_host_id()


def is_file_uri(uri_r: str | None) -> bool:
    return uri_r is not None and uri_r.startswith(FILE_SCHEME)


class OtherHostEntryRefused(Exception):
    """A deposit named a ``file://`` entry that another host's filesystem holds.

    ``engine_own`` is set when that host is the one running this process, which
    over HTTP means the store was stamped on the engine's host.
    """

    def __init__(self, uri_r: str, entry_origin: str, depositor_origin: str | None) -> None:
        self.uri_r = uri_r
        self.entry_origin = entry_origin
        self.depositor_origin = depositor_origin
        self.engine_own = entry_origin == local_host_id()
        super().__init__(self.describe())

    def describe(self) -> str:
        mine = self.depositor_origin or "none"
        if self.engine_own:
            return (
                f"{self.uri_r} is tracked for this engine's own host ({self.entry_origin}), "
                f"which refreshes it from its own disk; this deposit's host id is {mine}. If "
                "these are the client's files, the store was stamped on the engine host: run "
                "`particles corpus reorigin --from <engine host id> --to <client host id>` "
                "on the engine host."
            )
        return (
            f"{self.uri_r} is tracked for another host ({self.entry_origin}); this deposit's "
            f"host id is {mine}. Two hosts tracking one path in one store is refused "
            "; check the key's --host-id with `particles hook doctor`."
        )


async def entry_origin(session: AsyncSession, entry_id: str) -> str | None:
    """The stored origin of an entry, ``None`` when it has none or the entry is gone."""
    row = await session.get(CorpusEntryRow, entry_id)
    return row.origin_host if row is not None else None


async def claim_named_entry(session: AsyncSession, entry_id: str, uri_r: str | None) -> None:
    """Check a deposit may write the ``file://`` entry its URI-R names; adopt one with no origin.

    Raises :class:`OtherHostEntryRefused` when the entry's origin is another
    host's than the deposit's. An entry with no origin takes the deposit's
    (adoption, §5.2); a deposit with no origin adopts nothing.
    """
    if not is_file_uri(uri_r):
        return
    row = await session.get(CorpusEntryRow, entry_id)
    if row is None:
        return
    mine = current_deposit_origin()
    if row.origin_host is None:
        if mine is not None:
            row.origin_host = mine
            await session.flush()
            log.info("Entry %s adopted as host %s's file", entry_id[:8], mine)
        return
    if row.origin_host != mine:
        assert uri_r is not None
        if _operator.get() and mine is None and row.origin_host != local_host_id():
            return
        raise OtherHostEntryRefused(uri_r, row.origin_host, mine)


def stamp_new_entry(row: CorpusEntryRow) -> None:
    """Stamp a new ``file://`` entry's origin with the deposit's (§1.2)."""
    if is_file_uri(row.uri_r):
        row.origin_host = current_deposit_origin()


def foreign_origin(existing_origin: str | None) -> bool:
    """Whether a content-hash match onto an entry of this origin must not be joined (§1.3)."""
    return existing_origin is not None and existing_origin != current_deposit_origin()


# --- counting --------------------------------------------------------------------


@dataclass(frozen=True)
class OriginCounts:
    """``file://`` entries by origin, relative to this host."""

    this_host: int = 0
    other_hosts: int = 0
    unstamped: int = 0

    def line(self) -> str:
        return (
            f"{self.this_host} file:// entr(ies) are this host's, {self.other_hosts} another "
            f"host's, {self.unstamped} unstamped"
        )


async def count_file_entries_by_origin(
    session: AsyncSession, *, lazy_only: bool = False
) -> OriginCounts:
    """How many ``file://`` entries are this host's, another host's, or unstamped."""
    me = local_host_id()
    stmt = select(CorpusEntryRow.origin_host, func.count()).where(
        CorpusEntryRow.uri_r.is_not(None), CorpusEntryRow.uri_r.startswith(FILE_SCHEME)
    )
    if lazy_only:
        stmt = stmt.where(CorpusEntryRow.fetch_policy == FetchPolicy.LAZY.value)
    rows = (await session.execute(stmt.group_by(CorpusEntryRow.origin_host))).all()
    this_host = other = unstamped = 0
    for origin, n in rows:
        if origin is None:
            unstamped += int(n)
        elif me is not None and origin == me:
            this_host += int(n)
        else:
            other += int(n)
    return OriginCounts(this_host=this_host, other_hosts=other, unstamped=unstamped)


# --- the one-time stamp and the repair (§5.1, §5.3) -------------------------------


@dataclass(frozen=True)
class StampPlan:
    """What the one-time stamp writes, by class: rule files, memory files, the rest."""

    host_id: str | None
    rule_files: int = 0
    memory_files: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return self.rule_files + self.memory_files + self.other


async def has_origin_column(session: AsyncSession) -> bool:
    """Whether the store's schema has ``corpus_entries.origin_host`` yet (migration 062)."""

    def _check(sync_session: Session) -> bool:
        columns = inspect(sync_session.connection()).get_columns("corpus_entries")
        return any(column["name"] == "origin_host" for column in columns)

    return await session.run_sync(_check)


async def plan_origin_stamp(session: AsyncSession) -> StampPlan:
    """Count the unstamped ``file://`` entries the one-time stamp would give this host's id.

    Before migration 062 every ``file://`` entry is unstamped, so a preview run
    ahead of ``db init`` counts them all without reading the column.
    """
    rule = memory = other = 0
    stmt = select(CorpusEntryRow.tags_json).where(
        CorpusEntryRow.uri_r.is_not(None), CorpusEntryRow.uri_r.startswith(FILE_SCHEME)
    )
    if await has_origin_column(session):
        stmt = stmt.where(CorpusEntryRow.origin_host.is_(None))
    result = await session.execute(stmt)
    for (tags_json,) in result.all():
        tags = set(json.loads(tags_json or "[]"))
        if "rule-file" in tags:
            rule += 1
        elif tags & {"memory-file", "memory-archive"}:
            memory += 1
        else:
            other += 1
    return StampPlan(host_id=local_host_id(), rule_files=rule, memory_files=memory, other=other)


async def apply_origin_stamp(session: AsyncSession, plan: StampPlan) -> int:
    """Give every unstamped ``file://`` entry ``plan.host_id``. Returns the count; no commit."""
    if plan.host_id is None:
        return 0
    result = await session.execute(
        update(CorpusEntryRow)
        .where(
            CorpusEntryRow.uri_r.is_not(None),
            CorpusEntryRow.uri_r.startswith(FILE_SCHEME),
            CorpusEntryRow.origin_host.is_(None),
        )
        .values(origin_host=plan.host_id)
    )
    return int(getattr(result, "rowcount", 0) or 0)


async def count_reorigin(
    session: AsyncSession, *, from_host: str, uri_prefix: str | None = None
) -> int:
    """How many ``file://`` entries of ``from_host`` (under ``uri_prefix``) a repair would move."""
    stmt = select(func.count()).where(
        CorpusEntryRow.uri_r.is_not(None),
        CorpusEntryRow.uri_r.startswith(uri_prefix or FILE_SCHEME),
        CorpusEntryRow.origin_host == from_host,
    )
    return int((await session.execute(stmt)).scalar_one())


async def apply_reorigin(
    session: AsyncSession, *, from_host: str, to_host: str, uri_prefix: str | None = None
) -> int:
    """Move ``from_host``'s ``file://`` entries (under ``uri_prefix``) to ``to_host``. No commit."""
    result = await session.execute(
        update(CorpusEntryRow)
        .where(
            CorpusEntryRow.uri_r.is_not(None),
            CorpusEntryRow.uri_r.startswith(uri_prefix or FILE_SCHEME),
            CorpusEntryRow.origin_host == from_host,
        )
        .values(origin_host=to_host)
    )
    return int(getattr(result, "rowcount", 0) or 0)
