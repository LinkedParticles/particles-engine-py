# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``import vault``: a vault note is a living document.

One run of the verb, local arm. It walks the root, reclassifies the older
``STABLE`` note entries under it ``MUTABLE`` (with an event, retiring nothing),
deposits every listed note ``MUTABLE`` and identified by its path, and records
an ``ABSENT`` corpus record for each previously imported note the walk no
longer finds, when the guards allow. Nothing here retires a claim: an edit's
dropped claims and a deletion's claims are retired at the next extraction,
with no model call (``ingest/absence.py``).

Gather / decide / apply (D2): the walk and the store reads gather,
:func:`decide_absences` is the pure guard of §3.3, and the writes apply it.

Notes are deposited newest-edited first (:func:`newest_first`),
so the extraction queue, which follows deposit order, reads the notes a person
is working in before the rest, and the report prices what the import queued.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import Mutability, SourceType, WarcRecordType
from particles.corpus.deposit import (
    VAULT_SUFFIXES,
    deposit_vault,
    record_absence,
    vault_part_ignored,
    walk_vault,
)
from particles.corpus.store import (
    FileEntryRow,
    list_entry_ids_with_absences,
    list_file_entries_under,
    set_entry_mutability,
)
from particles.ingest.generation import backfill_superseded_generations
from particles.operations.extract_run import PendingSnapshot, RunPlan, plan_run
from particles.store.event_store import EventRefKind, OperatorEventType, record_event

log = logging.getLogger(__name__)

#: A mass absence is at least this many notes. A constant, not
#: a knob: below it, the first-absence confirmation is the guard.
ABSENCE_FLOOR = 5

#: A walk this size or larger is checked for rewritten modification times.
FLAT_MTIME_MIN_NOTES = 20

#: The actor on the reclassification event.
IMPORT_ACTOR = "import"

Refusal = Literal["first", "always", "mass"]


@dataclass(frozen=True)
class MissingNote:
    """A previously imported note the walk did not find."""

    entry_id: str
    uri_r: str
    path: Path
    refers_to: str | None


@dataclass(frozen=True)
class AbsenceDecision:
    """What :func:`decide_absences` concluded."""

    record: bool
    refusal: Refusal | None = None


def decide_absences(
    *,
    missing: int,
    previously_imported: int,
    first_under_root: bool,
    walk_complete: bool,
    confirmed: bool,
    keep_missing: bool,
    confirm_policy: Literal["first", "always"],
    max_fraction: float,
) -> AbsenceDecision:
    """Whether a walk may record the absences it found (§3.3). Pure.

    Nothing is recorded when there is nothing missing, the walk met an OS
    error, or the operator passed ``--keep-missing``. Otherwise the first
    absence under a root (every absence, under ``always``) and a mass absence
    (more than ``max_fraction`` of the previously imported notes, and at least
    :data:`ABSENCE_FLOOR`) are refused unless ``confirmed``.
    """
    if missing == 0 or not walk_complete or keep_missing:
        return AbsenceDecision(record=False)
    if confirmed:
        return AbsenceDecision(record=True)
    if confirm_policy == "always":
        return AbsenceDecision(record=False, refusal="always")
    if first_under_root:
        return AbsenceDecision(record=False, refusal="first")
    if missing >= ABSENCE_FLOOR and missing > max_fraction * previously_imported:
        return AbsenceDecision(record=False, refusal="mass")
    return AbsenceDecision(record=True)


@dataclass
class VaultImportReport:
    """What one ``import vault`` run did, for the verb's summary."""

    results: list[tuple[str, str]] = field(default_factory=list)
    """``(entry_id, snapshot_id)`` per deposited note, in walk order."""
    unchanged: list[bool] = field(default_factory=list)
    """Parallel to ``results``: ``True`` for a note skipped as unchanged."""
    unreadable: list[Path] = field(default_factory=list)
    """Listed notes that could not be read: skipped, and counted present."""
    reclassified: list[str] = field(default_factory=list)
    """Entries reclassified ``STABLE`` → ``MUTABLE`` this run (§1.3)."""
    backfill_entries: int = 0
    """…of those, the entries holding more than one generation (§1.3 preview)."""
    backfill_claims: int = 0
    """The claims ``corpus refresh --backfill-cascade`` would retire among them."""
    missing: list[MissingNote] = field(default_factory=list)
    """Previously imported notes the walk did not find."""
    absences: list[str] = field(default_factory=list)
    """The ``ABSENT`` records written, one per missing note when recorded."""
    refusal: Refusal | None = None
    """Why the missing notes were not recorded, when a guard refused."""
    kept_missing: bool = False
    walk_errors: list[str] = field(default_factory=list)
    export_dirs: list[Path] = field(default_factory=list)
    previously_imported: int = 0
    """The notes under the root imported before this walk and not already absent."""
    flat_mtimes: bool = False
    """More than half the walked notes share one minute of mtime."""
    queued: RunPlan | None = None
    """The extraction the new and changed notes owe, priced."""

    @property
    def changed(self) -> int:
        return sum(1 for flag in self.unchanged if not flag)

    @property
    def unchanged_count(self) -> int:
        return sum(self.unchanged)

    @property
    def absent_fraction(self) -> float:
        """The missing notes as a fraction of the previously imported ones."""
        if not self.previously_imported:
            return 0.0
        return len(self.missing) / self.previously_imported


def newest_first(files: Sequence[Path], mtimes: Sequence[float]) -> list[Path]:
    """``files`` by descending modification time, ties by path."""
    return [
        path for _, path in sorted(zip(mtimes, files, strict=True), key=lambda m: (-m[0], m[1]))
    ]


def mtimes_look_rewritten(mtimes: Sequence[float]) -> bool:
    """Whether a copy or a sync rewrote the walk's mtimes.

    True when at least :data:`FLAT_MTIME_MIN_NOTES` notes were walked and more
    than half of them carry an mtime in the same minute. Newest-first order is
    then path order, which the import says.
    """
    if len(mtimes) < FLAT_MTIME_MIN_NOTES:
        return False
    _, most = Counter(int(m // 60) for m in mtimes).most_common(1)[0]
    return most > len(mtimes) / 2


def _mtime(path: Path) -> float:
    """``path``'s mtime, or 0 when it cannot be read (it then deposits last)."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def path_of_file_uri(uri: str) -> Path:
    """The local path a ``file://`` URI-R names."""
    return Path(unquote(urlparse(uri).path))


def _in_walk_scope(path: Path, root: Path, export_dirs: Collection[Path]) -> bool:
    """Whether the walk's ignore policy would list ``path`` if it existed (§3.2)."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    if any(vault_part_ignored(part) for part in rel.parts):
        return False
    if path.suffix.lower() not in VAULT_SUFFIXES:
        return False
    return not any(d == path.parent or d in path.parents for d in export_dirs)


def icloud_placeholder(path: Path) -> Path:
    """The placeholder iCloud's optimised storage leaves for an evicted file."""
    return path.with_name(f".{path.name}.icloud")


async def import_vault(
    session: AsyncSession,
    vault_dir: Path,
    *,
    deposited_by: str = "operator",
    tags: list[str] | None = None,
    mutability: Mutability = Mutability.MUTABLE,
    confirm_absent: bool = False,
    keep_missing: bool = False,
    progress: Callable[[str], None] | None = None,
) -> VaultImportReport:
    """Walk ``vault_dir``, reclassify, deposit, and record what the walk no longer finds.

    Does not commit; the caller commits, including when the guard refuses,
    since a refusal still deposits every note it found.
    """
    if not vault_dir.exists() or not vault_dir.is_dir():
        raise ValueError(f"Vault directory not found: {vault_dir}")
    root = vault_dir.resolve()
    report = VaultImportReport()

    # Gather.
    walk = walk_vault(root)
    report.walk_errors = list(walk.errors)
    report.export_dirs = list(walk.export_dirs)
    notes = [
        row
        for row in await list_file_entries_under(session, root.as_uri())
        if row.source_type == SourceType.LOCAL_MARKDOWN
        and _in_walk_scope(path_of_file_uri(row.uri_r), root, walk.export_dirs)
    ]

    # §1.3: reclassify, with an event, retiring nothing.
    if mutability is Mutability.MUTABLE:
        report.reclassified = await _reclassify(session, notes)
        if report.reclassified:
            preview = await backfill_superseded_generations(
                session, dry_run=True, entry_ids=report.reclassified
            )
            report.backfill_entries = preview.entries_scanned
            report.backfill_claims = preview.demoted
    reclassified = set(report.reclassified)

    mtimes = [_mtime(path) for path in walk.files]
    report.flat_mtimes = mtimes_look_rewritten(mtimes)
    report.results = await deposit_vault(
        session,
        root,
        deposited_by=deposited_by,
        tags=tags,
        progress=progress,
        out_unchanged=report.unchanged,
        mutability=mutability,
        files=newest_first(walk.files, mtimes),
        unreadable=report.unreadable,
    )
    owed = [
        PendingSnapshot(entry_id, snapshot_id)
        for (entry_id, snapshot_id), unchanged in zip(report.results, report.unchanged, strict=True)
        if not unchanged
    ]
    if owed:
        await session.flush()
        report.queued = await plan_run(session, owed)

    listed = {path.resolve().as_uri() for path in walk.files}
    living = [
        row
        for row in notes
        if (row.mutability is Mutability.MUTABLE or row.entry_id in reclassified)
        and row.latest_record_type is not WarcRecordType.ABSENT
    ]
    report.previously_imported = len(living)
    report.missing = [
        MissingNote(
            entry_id=row.entry_id,
            uri_r=row.uri_r,
            path=path_of_file_uri(row.uri_r),
            refers_to=row.last_response_id,
        )
        for row in living
        if row.uri_r not in listed and not icloud_placeholder(path_of_file_uri(row.uri_r)).exists()
    ]
    first_under_root = bool(report.missing) and not await list_entry_ids_with_absences(
        session, [row.entry_id for row in notes]
    )

    # Decide.
    cfg = get_config().import_vault
    decision = decide_absences(
        missing=len(report.missing),
        previously_imported=report.previously_imported,
        first_under_root=first_under_root,
        walk_complete=walk.complete,
        confirmed=confirm_absent,
        keep_missing=keep_missing,
        confirm_policy=cfg.confirm_absent,
        max_fraction=cfg.max_absent_fraction,
    )
    report.refusal = decision.refusal
    report.kept_missing = keep_missing and bool(report.missing)

    # Apply.
    if decision.record:
        for note in report.missing:
            report.absences.append(
                await record_absence(session, entry_id=note.entry_id, refers_to=note.refers_to)
            )
    if report.missing:
        log.info(
            "import vault %s: %d note(s) no longer found, %.1f%% of %d previously imported; %s",
            root,
            len(report.missing),
            100 * report.absent_fraction,
            report.previously_imported,
            "recorded"
            if decision.record
            else f"not recorded ({decision.refusal or 'kept or walk incomplete'})",
        )
    return report


async def _reclassify(session: AsyncSession, notes: list[FileEntryRow]) -> list[str]:
    """Reclassify every ``STABLE`` note entry ``MUTABLE``; one event each (§1.3)."""
    done: list[str] = []
    for row in notes:
        if row.mutability is not Mutability.STABLE:
            continue
        before = await set_entry_mutability(session, row.entry_id, Mutability.MUTABLE)
        if before is None:
            continue
        await record_event(
            session,
            actor=IMPORT_ACTOR,
            event_type=OperatorEventType.CORPUS_ENTRY_RECLASSIFIED,
            refs=[(EventRefKind.CORPUS_ENTRY, row.entry_id)],
            payload={"from": before.value, "to": Mutability.MUTABLE.value},
        )
        done.append(row.entry_id)
    if done:
        log.info("import vault: reclassified %d STABLE note entr(ies) MUTABLE", len(done))
    return done
