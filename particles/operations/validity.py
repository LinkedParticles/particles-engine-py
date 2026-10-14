# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The world-time record on a claim: the prose default and the operator verdict.

Three writers keep ``valid_from`` and ``valid_kind``: extraction (at insert,
basis ``stated``), the deterministic prose-date pass here
(:func:`regenerate_validity`, the ``particles validity`` verb; bases
``prose``, ``as-of``, ``coarse``) and the operator verdict here
(:func:`date_particle`, ``particles particle date``; basis ``operator``). The
last two share one writer, ``set_validity_record``, which never overwrites a
``stated`` or ``operator`` value from the regeneration pass.

The pass is the pattern without the model: it reads the absolute
dates the prose rules already wrote into claim text, so it
costs nothing and runs in one sitting. What makes a prose default stale is
the cue rule, never the prose, so each value carries the rule digest and the
pass regenerates a row whose digest differs from the current one.

Nothing here reaches reconciliation, ranking or confidence: the record is
read-time material for the window route.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ValidKind
from particles.core.validity import (
    REGENERABLE_BASES,
    ProseAnchor,
    ValidFromBasis,
    anchor_from_prose,
    prose_rule_digest,
)
from particles.store.event_store import EventRefKind, OperatorEventType, record_event
from particles.store.particle_store import (
    ValidityRecordRow,
    get_validity_records,
    set_validity_record,
)

log = logging.getLogger(__name__)

#: The bases the regeneration pass leaves alone.
PRESERVED_BASES: tuple[str, ...] = (ValidFromBasis.STATED.value, ValidFromBasis.OPERATOR.value)


@dataclass
class ValidityCensus:
    """How the store's ACTIVE claims stand on the world-time axis."""

    total: int = 0
    dated: int = 0
    by_basis: dict[str, int] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)
    stale: int = 0
    """Prose defaults whose rule digest is not the current one."""
    rule: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "dated": self.dated,
            "undated": self.total - self.dated,
            "coverage": round(self.dated / self.total, 4) if self.total else 0.0,
            "by_basis": dict(sorted(self.by_basis.items())),
            "by_kind": dict(sorted(self.by_kind.items())),
            "stale": self.stale,
            "rule": self.rule,
        }


def _census(rows: list[ValidityRecordRow], rule: str) -> ValidityCensus:
    by_basis: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    census = ValidityCensus(total=len(rows), rule=rule)
    for row in rows:
        if row.valid_from is None:
            continue
        census.dated += 1
        by_basis[row.basis or "unrecorded"] += 1
        by_kind[row.valid_kind or "unkinded"] += 1
        if row.basis in REGENERABLE_BASES and row.rule != rule:
            census.stale += 1
    census.by_basis = dict(by_basis)
    census.by_kind = dict(by_kind)
    return census


def _in_scope(row: ValidityRecordRow, rule: str) -> bool:
    """Undated, or a prose default from another rule."""
    if row.valid_from is None:
        return row.basis not in PRESERVED_BASES
    return row.basis in REGENERABLE_BASES and row.rule != rule


@dataclass
class RegenerateSummary:
    """What one regeneration run did, or (dry run) would do."""

    dry_run: bool
    scope: int = 0
    set: int = 0
    cleared: int = 0
    """Stale prose defaults whose prose now yields no anchor."""
    skipped: dict[str, int] = field(default_factory=dict)
    by_basis: dict[str, int] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)
    failed: int = 0
    failed_ids: list[str] = field(default_factory=list)
    rule: str = ""
    census: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """A JSON-ready form for the verb's report."""
        return {
            "dry_run": self.dry_run,
            "scope": self.scope,
            "set": self.set,
            "cleared": self.cleared,
            "skipped": dict(sorted(self.skipped.items())),
            "by_basis": dict(sorted(self.by_basis.items())),
            "by_kind": dict(sorted(self.by_kind.items())),
            "failed": self.failed,
            "failed_ids": list(self.failed_ids),
            "rule": self.rule,
            "census": self.census,
        }


async def regenerate_validity(
    session: AsyncSession,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> RegenerateSummary:
    """Set the prose default on every ACTIVE claim in scope.

    Scope is every ACTIVE claim with no ``valid_from`` whose basis is not
    ``stated`` or ``operator``, plus every prose default stamped with a rule
    digest other than the current one. Each is read with
    :func:`anchor_from_prose`; an anchor is written through
    ``set_validity_record`` (which re-reads the row and preserves a stated or
    operator value recorded since), a skip is counted by reason, and a stale
    default whose prose no longer yields an anchor is cleared. Commits every
    ``validity_backfill.commit_interval`` rows and once at the end, so an
    interrupt costs seconds. A dry run reports the scope and the census and
    writes nothing.
    """
    rule = prose_rule_digest()
    rows = await get_validity_records(session)
    summary = RegenerateSummary(dry_run=dry_run, rule=rule)
    scope = [row for row in rows if _in_scope(row, rule)]
    summary.scope = len(scope)
    skipped: Counter[str] = Counter()
    by_basis: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    if dry_run:
        for row in scope:
            anchor = anchor_from_prose(row.content)
            if isinstance(anchor, ProseAnchor):
                summary.set += 1
                by_basis[anchor.basis.value] += 1
                by_kind[anchor.valid_kind.value] += 1
            else:
                skipped[anchor.reason] += 1
        summary.skipped, summary.by_basis, summary.by_kind = (
            dict(skipped),
            dict(by_basis),
            dict(by_kind),
        )
        summary.census = _census(rows, rule).as_dict()
        return summary

    interval = get_config().validity_backfill.commit_interval
    cap = len(scope) if not limit else min(limit, len(scope))
    for done, row in enumerate(scope[:cap], start=1):
        anchor = anchor_from_prose(row.content)
        try:
            if isinstance(anchor, ProseAnchor):
                if anchor.valid_kind is ValidKind.EVENT and row.valid_until is not None:
                    # A moment has one end: an EVENT cue on a claim that
                    # carries a boundary reads as the span's start.
                    kind = ValidKind.STATE
                else:
                    kind = anchor.valid_kind
                written = await set_validity_record(
                    session,
                    row.particle_id,
                    valid_from=anchor.valid_from,
                    valid_kind=kind,
                    basis=anchor.basis.value,
                    cue=anchor.cue,
                    granularity=anchor.granularity,
                    rule=rule,
                    preserve=PRESERVED_BASES,
                )
                if written:
                    summary.set += 1
                    by_basis[anchor.basis.value] += 1
                    by_kind[kind.value] += 1
                else:
                    skipped["preserved"] += 1
            elif row.valid_from is not None:
                # A stale default whose prose no longer anchors: cleared, so
                # the claim reads as undated rather than as placed by a rule
                # that no longer exists.
                written = await set_validity_record(
                    session,
                    row.particle_id,
                    valid_from=None,
                    valid_kind=None,
                    basis="",
                    preserve=PRESERVED_BASES,
                )
                if written:
                    summary.cleared += 1
                else:
                    skipped["preserved"] += 1
            else:
                skipped[anchor.reason] += 1
        except ValueError as exc:
            summary.failed += 1
            summary.failed_ids.append(row.particle_id)
            log.warning("validity: %s", exc)
        if progress is not None:
            progress(f"{done}/{cap} {row.particle_id[:8]}")
        if done % interval == 0:
            await session.commit()
    await session.commit()
    summary.skipped, summary.by_basis, summary.by_kind = (
        dict(skipped),
        dict(by_basis),
        dict(by_kind),
    )
    summary.census = _census(await get_validity_records(session), rule).as_dict()
    return summary


async def validity_census(session: AsyncSession) -> ValidityCensus:
    """The coverage census alone (acceptance item 3)."""
    return _census(await get_validity_records(session), prose_rule_digest())


@dataclass(frozen=True)
class DateResult:
    particle_id: str
    valid_from: datetime | None
    valid_kind: ValidKind | None


#: Event actor for the CLI operator verdict.
DATE_ACTOR = "cli:particle-date"


async def date_particle(
    session: AsyncSession,
    particle_id: str,
    *,
    valid_from: datetime | None,
    valid_kind: ValidKind | None,
    reason: str,
    actor: str = DATE_ACTOR,
) -> DateResult:
    """Set one claim's world-time record by operator verdict.

    Basis ``operator``, which ``particles validity`` never overwrites. A
    ``None`` start clears the record. Records a ``PARTICLE_DATED`` event with
    the reason and the values. Flushes; the caller commits.
    """
    await set_validity_record(
        session,
        particle_id,
        valid_from=valid_from,
        valid_kind=valid_kind,
        basis=ValidFromBasis.OPERATOR.value if valid_from is not None else "",
    )
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.PARTICLE_DATED,
        reason=reason,
        refs=[(EventRefKind.PARTICLE, particle_id)],
        payload={
            "valid_from": valid_from.isoformat() if valid_from else None,
            "valid_kind": valid_kind.value if valid_kind else None,
        },
    )
    return DateResult(particle_id=particle_id, valid_from=valid_from, valid_kind=valid_kind)
