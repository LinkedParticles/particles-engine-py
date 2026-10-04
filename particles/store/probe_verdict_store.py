# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The probe-verdict ledger: every pairwise probe answer, kept.

A pairwise probe (the census contradiction probe, its second reading, the
reconcile sweeps' contradiction and update-slot probes) costs one LLM call per
pair. Before this ledger only a YES left a trace, as the relation or record it
caused, so a pair the probe cleared was asked again on the next store-wide run
at full price. On the 2026-09-20 LongMemEval cycle that was 30,000 of 36,627
calls.

Each row is a record (D1): the answer a model gave about two claim
contents at one prompt version. It is keyed by the probe kind, the prompt
hash, and both claims' content hashes (:mod:`particles.core.probe_verdict`),
and carries the verdict, the model and the time. Rows are only appended. A
probe asked again that gives the same answer adds nothing; one that gives a
different answer adds a row, and the newest row is the one read. An edited
claim has a new content hash and a changed prompt a new prompt hash, so either
misses the old rows rather than invalidating them.

One kind is not a pairwise yes or no: the subject-link judge picks one of a
name's Wikidata candidates or none. Its rows carry that answer in
``answer``, a JSON object naming the chosen QID (or null) and every candidate
it was offered, so a rejected candidate stays on record without ever being
attached to a Subject. ``verdict`` is True when a candidate was chosen. Its
key includes the model (:func:`lookup_answer`), because a pick among
candidates is a judgement a different model may make differently.

The functions flush and never commit: the caller owns the transaction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.probe_verdict import ProbeKind, VerdictKey
from particles.db import Base

__all__ = [
    "ProbeVerdictRow",
    "count_verdicts",
    "lookup_answer",
    "lookup_verdicts",
    "record_answer",
    "record_verdict",
    "record_verdicts",
    "verdicts_for_pair",
]

#: Pairs per ``IN`` clause: two bound parameters each, well under SQLite's limit.
_CHUNK = 400


class ProbeVerdictRow(Base):
    """One recorded answer to one pairwise probe."""

    __tablename__ = "probe_verdicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    probe_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    hash_a: Mapped[str] = mapped_column(String(64), nullable=False)
    hash_b: Mapped[str] = mapped_column(String(64), nullable=False)
    verdict: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: The full answer when a kind has more than a yes or no to say (JSON text),
    #: else ``None``. Only :attr:`ProbeKind.SUBJECT_LINK` writes it.
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str] = mapped_column(String, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_probe_verdicts_key", "probe_kind", "prompt_hash", "hash_a", "hash_b"),
    )


async def lookup_verdicts(
    session: AsyncSession,
    kind: ProbeKind,
    prompt_hash: str,
    keys: Iterable[VerdictKey],
) -> dict[VerdictKey, bool]:
    """The newest recorded verdict for each key already answered under ``prompt_hash``.

    Keys never answered under this prompt version are absent from the result.
    """
    wanted = list(dict.fromkeys(keys))
    found: dict[VerdictKey, tuple[int, bool]] = {}
    for start in range(0, len(wanted), _CHUNK):
        chunk = wanted[start : start + _CHUNK]
        rows = await session.execute(
            select(
                ProbeVerdictRow.id,
                ProbeVerdictRow.hash_a,
                ProbeVerdictRow.hash_b,
                ProbeVerdictRow.verdict,
            ).where(
                ProbeVerdictRow.probe_kind == kind.value,
                ProbeVerdictRow.prompt_hash == prompt_hash,
                tuple_(ProbeVerdictRow.hash_a, ProbeVerdictRow.hash_b).in_(chunk),
            )
        )
        for row_id, hash_a, hash_b, verdict in rows:
            key = (hash_a, hash_b)
            if key not in found or row_id > found[key][0]:
                found[key] = (row_id, verdict)
    return {key: verdict for key, (_, verdict) in found.items()}


async def record_verdicts(
    session: AsyncSession,
    kind: ProbeKind,
    prompt_hash: str,
    verdicts: Mapping[VerdictKey, bool],
    *,
    model: str,
    now: datetime | None = None,
) -> int:
    """Append each verdict whose key has no record, or whose newest record differs.

    Returns the number of rows written. Flushes; the caller commits.
    """
    if not verdicts:
        return 0
    known = await lookup_verdicts(session, kind, prompt_hash, verdicts)
    at = now or datetime.now(UTC)
    rows = [
        ProbeVerdictRow(
            probe_kind=kind.value,
            prompt_hash=prompt_hash,
            hash_a=key[0],
            hash_b=key[1],
            verdict=verdict,
            model=model,
            recorded_at=at,
        )
        for key, verdict in verdicts.items()
        if known.get(key) is not verdict
    ]
    session.add_all(rows)
    await session.flush()
    return len(rows)


async def record_verdict(
    session: AsyncSession,
    kind: ProbeKind,
    prompt_hash: str,
    key: VerdictKey,
    verdict: bool,
    *,
    model: str,
    now: datetime | None = None,
) -> bool:
    """Record one verdict; True when a row was written."""
    return bool(
        await record_verdicts(session, kind, prompt_hash, {key: verdict}, model=model, now=now)
    )


async def lookup_answer(
    session: AsyncSession,
    kind: ProbeKind,
    prompt_hash: str,
    key: VerdictKey,
    *,
    model: str,
) -> str | None:
    """The newest recorded answer for ``key`` from ``model``, or ``None`` when none is.

    For a kind whose answer is more than a yes or no. The model is part of the
    key here, unlike :func:`lookup_verdicts`.
    """
    row = (
        await session.execute(
            select(ProbeVerdictRow.answer)
            .where(
                ProbeVerdictRow.probe_kind == kind.value,
                ProbeVerdictRow.prompt_hash == prompt_hash,
                ProbeVerdictRow.hash_a == key[0],
                ProbeVerdictRow.hash_b == key[1],
                ProbeVerdictRow.model == model,
                ProbeVerdictRow.answer.is_not(None),
            )
            .order_by(ProbeVerdictRow.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return row


async def record_answer(
    session: AsyncSession,
    kind: ProbeKind,
    prompt_hash: str,
    key: VerdictKey,
    *,
    verdict: bool,
    answer: str,
    model: str,
    now: datetime | None = None,
) -> bool:
    """Append ``answer`` unless it is already the newest one recorded; True when written.

    Flushes; the caller commits.
    """
    if await lookup_answer(session, kind, prompt_hash, key, model=model) == answer:
        return False
    session.add(
        ProbeVerdictRow(
            probe_kind=kind.value,
            prompt_hash=prompt_hash,
            hash_a=key[0],
            hash_b=key[1],
            verdict=verdict,
            answer=answer,
            model=model,
            recorded_at=now or datetime.now(UTC),
        )
    )
    await session.flush()
    return True


async def verdicts_for_pair(
    session: AsyncSession, hash_x: str, hash_y: str
) -> list[ProbeVerdictRow]:
    """Every recorded answer about two claims, in either key order, oldest first.

    Spans every probe kind and prompt version: a caller reporting what produced
    a decision wants the whole trail, not the answer one prompt would reuse.
    """
    rows = await session.execute(
        select(ProbeVerdictRow)
        .where(
            tuple_(ProbeVerdictRow.hash_a, ProbeVerdictRow.hash_b).in_(
                [(hash_x, hash_y), (hash_y, hash_x)]
            )
        )
        .order_by(ProbeVerdictRow.id)
    )
    return list(rows.scalars())


async def count_verdicts(session: AsyncSession, kind: ProbeKind | None = None) -> int:
    """Rows in the ledger, optionally for one probe kind."""
    stmt = select(func.count()).select_from(ProbeVerdictRow)
    if kind is not None:
        stmt = stmt.where(ProbeVerdictRow.probe_kind == kind.value)
    return int((await session.execute(stmt)).scalar_one())
