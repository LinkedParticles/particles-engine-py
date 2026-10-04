# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Demotion rulings kept as labelled benchmark pairs.

A ``demotion`` card shows a claim a later claim retired as its replacement.
When the operator affirms it (the replacement was right) or dismisses it (both
claims hold), that judgment is the label the memory-rot benchmark's live
update checks are scored against, and the store keeps it nowhere reusable:
the gesture's event names only the card. This module writes it to an
operator-owned, append-only JSONL file under ``benchmark.runs_dir``, one
record per ruling.

The record is a side effect of the gesture, never part of its meaning. No
status changes: a demotion encodes a judgment, so it is not reversible,
and a "coexist" ruling leaves the retired claim retired. A later
ruling on the same pair is appended, not merged; readers take the newest one
per pair (:func:`load_rulings`).

The file holds claim text from the operator's own store and lives under their
home directory. ``benchmark.record_demotion_rulings`` turns the writing off.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.duplicate_key import content_hash
from particles.core.schema import Particle
from particles.store.particle_store import REPLACEMENT_DEMOTIONS, get_particles_by_ids
from particles.store.probe_verdict_store import verdicts_for_pair
from particles.store.subject_store import get_subject

from .cards import DEMOTION_RULINGS, CardKind, CurationCard

log = logging.getLogger(__name__)

__all__ = [
    "RULINGS_FILENAME",
    "RULING_FORMAT",
    "DemotionRuling",
    "RulingClaim",
    "RulingProbeVerdict",
    "RulingSubject",
    "build_ruling",
    "load_rulings",
    "record_demotion_ruling",
    "rulings_path",
]

#: The fixture file's name inside ``benchmark.runs_dir``.
RULINGS_FILENAME = "demotion-rulings.jsonl"

#: The record format stamp. A reader skips a line under any other stamp.
RULING_FORMAT = "particles.demotion-ruling/1"

#: What the operator ruled: the replacement was right, or both claims hold.
Ruling = Literal["replacement", "coexist"]


class RulingClaim(BaseModel):
    """One side of a ruled pair."""

    particle_id: str
    content_hash: str
    content: str
    asserted_at: datetime | None = None


class RulingSubject(BaseModel):
    """A subject the retired claim is about."""

    subject_id: str
    name: str | None = None


class RulingProbeVerdict(BaseModel):
    """One recorded probe answer about the pair (the ledger row)."""

    probe_kind: str
    prompt_hash: str
    hash_a: str
    hash_b: str
    verdict: bool
    model: str
    recorded_at: datetime


class DemotionRuling(BaseModel):
    """One operator ruling on a demotion: a labelled pair."""

    format: str = RULING_FORMAT
    ruling: Ruling
    #: The demotion's status reason, e.g. ``SUPERSEDED_BY_UPDATE``.
    reason: str
    retired: RulingClaim
    replacement: RulingClaim
    subjects: list[RulingSubject] = Field(default_factory=list)
    probe_verdicts: list[RulingProbeVerdict] = Field(default_factory=list)
    actor: str
    store: str
    recorded_at: datetime
    card_key: str

    @property
    def pair_key(self) -> tuple[str, str]:
        """``(retired hash, replacement hash)``: the identity a newer ruling replaces."""
        return (self.retired.content_hash, self.replacement.content_hash)


def rulings_path() -> Path:
    """Where rulings are written: ``<benchmark.runs_dir>/demotion-rulings.jsonl``."""
    return Path(get_config().benchmark.runs_dir).expanduser() / RULINGS_FILENAME


def _split_pair(particles: list[Particle]) -> tuple[Particle, Particle] | None:
    """``(retired, replacement)`` from a demotion card's two claims, in either order."""
    if len(particles) != 2:
        return None
    a, b = particles
    for retired, replacement in ((a, b), (b, a)):
        if retired.status_reason in REPLACEMENT_DEMOTIONS and replacement.supersedes == retired.id:
            return retired, replacement
    for retired, replacement in ((a, b), (b, a)):
        if retired.status_reason in REPLACEMENT_DEMOTIONS:
            return retired, replacement
    return None


def _claim(p: Particle) -> RulingClaim:
    return RulingClaim(
        particle_id=p.id,
        content_hash=content_hash(p.content),
        content=p.content,
        asserted_at=p.asserted_at,
    )


async def build_ruling(
    session: AsyncSession,
    card: CurationCard,
    ruling: Ruling,
    *,
    actor: str,
    store: str,
    now: datetime | None = None,
) -> DemotionRuling | None:
    """The record for one ruling on a demotion card, or ``None`` if the pair is gone.

    Reads both claims by id (a card key sorts its ids, so the order is
    re-derived from the store), the retired claim's subjects, and every probe
    verdict the ledger holds for the two contents.
    """
    found = await get_particles_by_ids(session, card.particle_ids)
    pair = _split_pair([found[pid] for pid in card.particle_ids if pid in found])
    if pair is None:
        return None
    retired, replacement = pair
    subjects: list[RulingSubject] = []
    for sid in retired.subject_ids:
        subject = await get_subject(session, sid)
        subjects.append(
            RulingSubject(
                subject_id=sid, name=subject.canonical_name if subject is not None else None
            )
        )
    rows = await verdicts_for_pair(
        session, content_hash(retired.content), content_hash(replacement.content)
    )
    assert retired.status_reason is not None  # _split_pair requires one
    return DemotionRuling(
        ruling=ruling,
        reason=retired.status_reason.value,
        retired=_claim(retired),
        replacement=_claim(replacement),
        subjects=subjects,
        probe_verdicts=[
            RulingProbeVerdict(
                probe_kind=r.probe_kind,
                prompt_hash=r.prompt_hash,
                hash_a=r.hash_a,
                hash_b=r.hash_b,
                verdict=r.verdict,
                model=r.model,
                recorded_at=r.recorded_at,
            )
            for r in rows
        ],
        actor=actor,
        store=store,
        recorded_at=now or datetime.now(UTC),
        card_key=card.key,
    )


def _append(record: DemotionRuling, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


async def record_demotion_ruling(
    session: AsyncSession,
    card: CurationCard,
    gesture: str,
    *,
    actor: str,
    store: str,
) -> str | None:
    """Append the ruling a gesture on a demotion card makes; return the disclosure.

    ``None`` when nothing applies: not a demotion card, a gesture that rules
    nothing (snooze), or recording switched off. A write that fails is logged
    and disclosed, never raised: the gesture itself has already succeeded.
    """
    ruling = DEMOTION_RULINGS.get(gesture)
    if card.kind is not CardKind.DEMOTION or ruling is None:
        return None
    if not get_config().benchmark.record_demotion_rulings:
        return None
    record = await build_ruling(session, card, ruling, actor=actor, store=store)
    if record is None:
        return "The pair is no longer in the store, so no benchmark fixture was recorded."
    path = rulings_path()
    try:
        _append(record, path)
    except OSError as exc:
        log.warning("Could not record demotion ruling to %s: %s", path, exc)
        return f"Could not record the benchmark fixture ({exc})."
    return "Recorded as a benchmark fixture."


def load_rulings(path: Path) -> tuple[list[DemotionRuling], list[str]]:
    """The newest ruling per pair in ``path``, plus a note per line skipped.

    A missing file is no rulings. A line that does not parse, or carries
    another format stamp, is skipped and named in the notes, never fatal.
    """
    if not path.exists():
        return [], []
    latest: dict[tuple[str, str], DemotionRuling] = {}
    notes: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if raw.get("format") != RULING_FORMAT:
                    notes.append(f"{path.name}:{lineno}: format {raw.get('format')!r} skipped")
                    continue
                record = DemotionRuling.model_validate(raw)
            except (ValueError, ValidationError, AttributeError) as exc:
                notes.append(f"{path.name}:{lineno}: unreadable ({type(exc).__name__}), skipped")
                continue
            latest[record.pair_key] = record
    return list(latest.values()), notes
