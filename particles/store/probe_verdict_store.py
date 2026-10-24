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

**Rungs, not only model probes**. A row may also carry the
stamp of the rung that produced it (``rung``, ``name@version``; a model
probe's version is its prompt hash), the verdict in the ladder's own
vocabulary (``rung_verdict``), and a ``shadow`` flag for a rung that has not
met its bar, whose verdict was recorded and never returned. A shadow rung's
rows sit under a kind of their own (:func:`~particles.core.probe_verdict.rung_kind`),
keyed on the same content hashes as the probe whose question it answers, so
its verdicts and the probe's answers on the same inputs join
(:func:`shadow_standings`); those rows are what its admission bar accumulates.
An admitted deterministic rung writes no row: nothing reads
one, its load is counted on the run record, and on a large store it decides
more pairs per census than the ledger holds probe answers. A rung row records
a decision, never an abstention: an abstaining rung is counted on the run
record and leaves no row.

The functions flush and never commit: the caller owns the transaction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    false,
    func,
    select,
    tuple_,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.probe_verdict import RUNG_KIND_PREFIX, ProbeKind, VerdictKey, rung_kind
from particles.core.rung_ladder import RungStamp
from particles.db import Base
from particles.llm.usage import NO_SAMPLING

__all__ = [
    "POSITIVE_VERDICTS",
    "ProbeVerdictRow",
    "ShadowStanding",
    "count_verdicts",
    "lookup_answer",
    "lookup_stamped_verdicts",
    "lookup_verdicts",
    "record_answer",
    "record_verdict",
    "record_rung_verdicts",
    "record_stamped_verdicts",
    "record_verdicts",
    "shadow_standings",
    "verdicts_for_pair",
]

#: The ladder verdicts a row's ``verdict`` boolean reads as True: the answer
#: that would act (a contradiction, a same slot, one claim, an entailment).
POSITIVE_VERDICTS = frozenset({"yes", "changes", "fixed", "same", "entailed"})

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
    #: The stamp of the rung that produced the row, ``name@version`` (
    #: § 6); ``None`` on a row recorded before the ladder.
    rung: Mapped[str | None] = mapped_column(String(96), nullable=True)
    #: The verdict in the ladder's own vocabulary, when ``verdict`` cannot carry it.
    rung_verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    #: The rung ran in shadow: its verdict was recorded and never returned.
    shadow: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    #: What the answering call sent: ``sent``, ``dropped``,
    #: ``unset``, or ``none`` for a rung that makes no model call. ``sent``
    #: says a fixed temperature was received, never that the answer is
    #: reproducible. ``NULL`` on a row recorded before the column, or when the
    #: answering provider reported nothing (an in-process one).
    temperature: Mapped[str | None] = mapped_column(String(16), nullable=True)

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
    rung: RungStamp | None = None,
    temperature: str | None = None,
) -> int:
    """Append each verdict whose key has no record, or whose newest record differs.

    ``rung`` stamps the rows with the ladder rung that asked.
    ``temperature`` is what the answering calls sent, as
    :func:`particles.llm.observe_sampling` observed it.
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
            rung=None if rung is None else str(rung),
            temperature=temperature,
        )
        for key, verdict in verdicts.items()
        if known.get(key) is not verdict
    ]
    session.add_all(rows)
    await session.flush()
    return len(rows)


async def lookup_stamped_verdicts(
    session: AsyncSession,
    kind: ProbeKind,
    prompt_hash: str,
    keys: Iterable[VerdictKey],
) -> dict[VerdictKey, tuple[bool, str | None]]:
    """The newest recorded verdict for each key under ``prompt_hash``, with its rung.

    As :func:`lookup_verdicts`, plus the ``name@version`` stamp of the rung that
    answered (``None`` on a row written without one). A caller that reads more
    than yes or no from a ladder, such as which rung settled an update reading,
    reads it here.
    """
    wanted = list(dict.fromkeys(keys))
    found: dict[VerdictKey, tuple[int, bool, str | None]] = {}
    for start in range(0, len(wanted), _CHUNK):
        chunk = wanted[start : start + _CHUNK]
        rows = await session.execute(
            select(
                ProbeVerdictRow.id,
                ProbeVerdictRow.hash_a,
                ProbeVerdictRow.hash_b,
                ProbeVerdictRow.verdict,
                ProbeVerdictRow.rung,
            ).where(
                ProbeVerdictRow.probe_kind == kind.value,
                ProbeVerdictRow.prompt_hash == prompt_hash,
                tuple_(ProbeVerdictRow.hash_a, ProbeVerdictRow.hash_b).in_(chunk),
            )
        )
        for row_id, hash_a, hash_b, verdict, rung in rows:
            key = (hash_a, hash_b)
            if key not in found or row_id > found[key][0]:
                found[key] = (row_id, verdict, rung)
    return {key: (verdict, rung) for key, (_, verdict, rung) in found.items()}


async def record_stamped_verdicts(
    session: AsyncSession,
    kind: ProbeKind,
    prompt_hash: str,
    verdicts: Mapping[VerdictKey, tuple[bool, RungStamp | None]],
    *,
    model: str,
    now: datetime | None = None,
) -> int:
    """Append each verdict whose newest record differs in its value **or its rung**.

    :func:`record_verdicts` skips a key whose newest verdict is unchanged, which
    loses the rung that answered when a second reading now settles what a first
    reading settled before. Returns the rows written. Flushes; the caller commits.
    """
    if not verdicts:
        return 0
    known = await lookup_stamped_verdicts(session, kind, prompt_hash, verdicts)
    at = now or datetime.now(UTC)
    rows = []
    for key, (verdict, rung) in verdicts.items():
        stamp = None if rung is None else str(rung)
        if known.get(key) == (verdict, stamp):
            continue
        rows.append(
            ProbeVerdictRow(
                probe_kind=kind.value,
                prompt_hash=prompt_hash,
                hash_a=key[0],
                hash_b=key[1],
                verdict=verdict,
                model=model,
                recorded_at=at,
                rung=stamp,
            )
        )
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
    temperature: str | None = None,
) -> bool:
    """Record one verdict; True when a row was written."""
    return bool(
        await record_verdicts(
            session,
            kind,
            prompt_hash,
            {key: verdict},
            model=model,
            now=now,
            temperature=temperature,
        )
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
    temperature: str | None = None,
) -> bool:
    """Append ``answer`` unless it is already the newest one recorded; True when written.

    ``temperature`` is what the answering call sent. Flushes;
    the caller commits.
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
            temperature=temperature,
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


async def _newest_rung_verdicts(
    session: AsyncSession, kind: str, version: str, shadow: bool, keys: Sequence[VerdictKey]
) -> dict[VerdictKey, str]:
    found: dict[VerdictKey, tuple[int, str]] = {}
    for start in range(0, len(keys), _CHUNK):
        chunk = keys[start : start + _CHUNK]
        rows = await session.execute(
            select(
                ProbeVerdictRow.id,
                ProbeVerdictRow.hash_a,
                ProbeVerdictRow.hash_b,
                ProbeVerdictRow.rung_verdict,
            ).where(
                ProbeVerdictRow.probe_kind == kind,
                ProbeVerdictRow.prompt_hash == version,
                ProbeVerdictRow.shadow == shadow,
                tuple_(ProbeVerdictRow.hash_a, ProbeVerdictRow.hash_b).in_(chunk),
            )
        )
        for row_id, hash_a, hash_b, value in rows:
            key = (hash_a, hash_b)
            if value is not None and (key not in found or row_id > found[key][0]):
                found[key] = (row_id, value)
    return {key: value for key, (_, value) in found.items()}


async def record_rung_verdicts(
    session: AsyncSession,
    stamp: RungStamp,
    verdicts: Mapping[VerdictKey, str],
    *,
    shadow: bool = False,
    now: datetime | None = None,
) -> int:
    """Append a deterministic rung's verdicts, one row per input it decided.

    ``verdicts`` maps each input's key to the rung's verdict value; pass only
    decisions, never an abstention. The rows sit under :func:`rung_kind` with the
    rung's version as the prompt hash, so a new version is a new record set. A
    verdict equal to the newest one already recorded for its key adds nothing.
    Returns the rows written. Flushes; the caller commits.

    The census calls this for its shadow rungs only (``shadow=True``), the
    rows :func:`shadow_standings` reads; an admitted rung's decisions are
    counted on the run record and never written here.
    """
    if not verdicts:
        return 0
    kind = rung_kind(stamp.name)
    wanted = list(dict.fromkeys(verdicts))
    known = await _newest_rung_verdicts(session, kind, stamp.version, shadow, wanted)
    at = now or datetime.now(UTC)
    rows = [
        ProbeVerdictRow(
            probe_kind=kind,
            prompt_hash=stamp.version,
            hash_a=key[0],
            hash_b=key[1],
            verdict=value in POSITIVE_VERDICTS,
            model=str(stamp),
            recorded_at=at,
            rung=str(stamp),
            rung_verdict=value,
            shadow=shadow,
            temperature=NO_SAMPLING,
        )
        for key, value in verdicts.items()
        if known.get(key) != value
    ]
    session.add_all(rows)
    await session.flush()
    return len(rows)


@dataclass(frozen=True)
class ShadowStanding:
    """A shadow rung's running precision for one verdict it returns."""

    rung: str
    """The rung's stamp, ``name@version``."""
    verdict: str
    """The verdict value this line measures (``yes``, ``no``, …)."""
    recorded: int
    """Inputs the rung gave this verdict on."""
    labelled: int
    """…of those, inputs the labels cover: the bar's input count."""
    correct: int
    """…of those, inputs where the label agrees."""

    @property
    def precision(self) -> float | None:
        """``correct / labelled``, or ``None`` with no labelled input yet."""
        return self.correct / self.labelled if self.labelled else None


async def shadow_standings(
    session: AsyncSession,
    label_kinds: Sequence[ProbeKind] = (ProbeKind.SECOND_READING, ProbeKind.CONTRADICTION),
) -> list[ShadowStanding]:
    """Each shadow rung's running precision against the probes' answers.

    A shadow rung's verdicts accrue night after night beside the probe's
    answers on the same inputs. The label of an input is the newest answer of
    the first kind in ``label_kinds`` that answered it, under any prompt
    version: for the contradiction ladder, the second reading where one ran,
    else the census probe. A shadow ``no`` is correct when the label is NO,
    any other verdict when it is YES. The owner's labels in the kept fixture
    are keyed on particle ids, not contents, and are joined by the milestone's
    benchmark, not here.
    """
    rows = (
        await session.execute(
            select(
                ProbeVerdictRow.id,
                ProbeVerdictRow.rung,
                ProbeVerdictRow.rung_verdict,
                ProbeVerdictRow.hash_a,
                ProbeVerdictRow.hash_b,
            ).where(
                ProbeVerdictRow.shadow.is_(True),
                ProbeVerdictRow.probe_kind.startswith(RUNG_KIND_PREFIX),
            )
        )
    ).all()
    newest: dict[tuple[str, VerdictKey], tuple[int, str]] = {}
    for row_id, rung, value, hash_a, hash_b in rows:
        if rung is None or value is None:
            continue
        slot = (rung, (hash_a, hash_b))
        if slot not in newest or row_id > newest[slot][0]:
            newest[slot] = (row_id, value)
    keys = list({key for _, key in newest})
    labels: dict[VerdictKey, bool] = {}
    for kind in label_kinds:
        missing = [k for k in keys if k not in labels]
        if not missing:
            break
        for start in range(0, len(missing), _CHUNK):
            chunk = missing[start : start + _CHUNK]
            found = await session.execute(
                select(
                    ProbeVerdictRow.id,
                    ProbeVerdictRow.hash_a,
                    ProbeVerdictRow.hash_b,
                    ProbeVerdictRow.verdict,
                )
                .where(
                    ProbeVerdictRow.probe_kind == kind.value,
                    tuple_(ProbeVerdictRow.hash_a, ProbeVerdictRow.hash_b).in_(chunk),
                )
                .order_by(ProbeVerdictRow.id)
            )
            for _, hash_a, hash_b, verdict in found:
                labels[(hash_a, hash_b)] = verdict
    tally: dict[tuple[str, str], list[int]] = {}
    for (rung, key), (_, value) in newest.items():
        cell = tally.setdefault((rung, value), [0, 0, 0])
        cell[0] += 1
        if key in labels:
            cell[1] += 1
            cell[2] += int(labels[key] == (value in POSITIVE_VERDICTS))
    return [
        ShadowStanding(rung=rung, verdict=value, recorded=r, labelled=lab, correct=ok)
        for (rung, value), (r, lab, ok) in sorted(tally.items())
    ]
