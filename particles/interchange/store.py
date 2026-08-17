"""Store-aware interchange export / import / restore (Part B).

Wraps the pure codec with the DB I/O the wire format needs:
  - **export** resolves a particle's subjects from the session and encodes units;
  - **import** resolves each unit's subject refs (by external reference, then by
    name, creating bare-local subjects as needed), assigns the resulting
    subject_ids, and inserts the particle through the §6.6 ladder
    (``reconcile_and_insert``) — so an imported claim reconciles against the
    target store exactly as a freshly extracted one would;
  - **restore** is the faithful, id-preserving sibling of import:
    it reconstructs *the very store a bundle was exported from* into an **empty**
    target, inserting each particle / subject **directly** with its origin id
    preserved verbatim — no fingerprint reconcile, no fresh id. It refuses a
    non-empty target (a collision is a reconstruction error, never a silent
    merge), which is exactly what keeps the invariant (cross-store identity is
    claim-fingerprint, not UUID) intact: restore is reconstruct-only and can
    never become a UUID-smuggling cross-store merge.

Provenance refs travel as-is and point at the *source* store's corpus entries
(origin metadata); the corpus itself moves only via store-export (Part C).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import SCHEMA_VERSION, Particle, Subject
from particles.interchange.codec import (
    FORMAT_VERSION,
    SubjectRef,
    from_unit,
    subject_from_unit,
    subject_to_unit,
    to_unit,
)
from particles.interchange.jsonl import read_jsonl, write_jsonl
from particles.interchange.yaml_ld import read_yaml_ld, write_yaml_ld


async def export_particles(
    session: AsyncSession, particles: Sequence[Particle]
) -> list[dict[str, Any]]:
    """Encode particles (resolving each one's subjects from the store) to units."""
    from particles.store.subject_store import get_subjects_for_particle

    units: list[dict[str, Any]] = []
    for particle in particles:
        subjects = await get_subjects_for_particle(session, particle.id)
        units.append(to_unit(particle, {s.id: s for s in subjects}))
    return units


async def export_active(session: AsyncSession) -> list[dict[str, Any]]:
    """Encode every ACTIVE particle in the store to interchange units."""
    from particles.store.particle_store import get_active_particles

    return await export_particles(session, await get_active_particles(session))


@dataclass
class ImportSummary:
    imported: int = 0
    dropped: int = 0
    subjects_created: int = 0


async def import_units(session: AsyncSession, units: Iterable[dict[str, Any]]) -> ImportSummary:
    """Import interchange units into the current store (single-store write, §6.6).

    Subjects are resolved by external reference first (the cross-store join key),
    then by canonical name; an unmatched named subject is created bare-local.
    Each particle is reconciled via the §6.6 ladder; a particle dropped by trust
    resolution is counted in ``dropped``.
    """
    from particles.ingest.pipeline import (
        load_active_conflict_candidates,
        reconcile_and_insert,
    )

    summary = ImportSummary()
    # Load the §6.6 conflict-candidate set once for the whole batch and let
    # reconcile_and_insert maintain it in place, so an N-unit import runs one
    # full-store embedding scan instead of N (F4.3).
    candidate_cache = await load_active_conflict_candidates(session)
    for unit in units:
        parsed = from_unit(unit)
        subject_ids: list[str] = []
        for ref in parsed.subjects:
            sid, created = await _resolve_subject_ref(session, ref)
            if sid is not None:
                subject_ids.append(sid)
            if created:
                summary.subjects_created += 1

        particle = parsed.particle.model_copy(update={"subject_ids": subject_ids})
        inserted = await reconcile_and_insert(session, particle, candidate_cache=candidate_cache)
        if inserted is None:
            summary.dropped += 1
        else:
            summary.imported += 1
    return summary


async def _resolve_subject_ref(session: AsyncSession, ref: SubjectRef) -> tuple[str | None, bool]:
    """Resolve a subject ref to a target subject_id, creating one if needed.

    Returns ``(subject_id, created)``. Resolution order: shared external ref →
    canonical name → create bare-local. A ref with neither an external ref nor a
    name is unresolvable (returns ``(None, False)``) — the particle imports with
    one fewer subject link.
    """
    from particles.store.subject_store import (
        find_by_external_ref,
        find_by_name,
        insert_subject,
    )

    for ext in ref.external_refs:
        existing = await find_by_external_ref(session, ext.namespace, ext.id)
        if existing is not None:
            return existing.id, False

    if ref.canonical_name:
        by_name = await find_by_name(session, ref.canonical_name)
        if by_name is not None:
            return by_name.id, False
        subject = Subject(
            canonical_name=ref.canonical_name,
            aliases=ref.aliases,
            external_ids=ref.external_refs,
            subject_class=ref.subject_class,
            asserted_by="interchange-import",
        )
        await insert_subject(session, subject)
        return subject.id, True

    return None, False


MANIFEST_NAME = "manifest.json"
PARTICLES_BASE = "particles"
SUBJECTS_BASE = "subjects"
#: Canonical JSONL member names, kept for back-compat (restore's single-member
#: scan and callers that name the members directly).
PARTICLES_MEMBER = f"{PARTICLES_BASE}.jsonl"
SUBJECTS_MEMBER = f"{SUBJECTS_BASE}.jsonl"

#: Container format -> (member extension, writer). JSONL is canonical / lossless;
#: YAML-LD is the human-editable sibling. Both share the same codec
#: units and ``manifest.json`` envelope — only the concrete member syntax differs.
_CONTAINER_WRITERS = {
    "jsonl": (".jsonl", write_jsonl),
    "yaml": (".yaml", write_yaml_ld),
}
#: Member extension -> reader. Import / restore auto-detect the container from a
#: member's extension, so a bundle written either way re-imports with no flag.
#: JSONL is listed first so it wins the unlikely case a directory holds both.
_MEMBER_READERS = {
    ".jsonl": read_jsonl,
    ".yaml": read_yaml_ld,
    ".yml": read_yaml_ld,
}


def _read_member(files: dict[str, str], base: str) -> list[dict[str, Any]]:
    """Read the ``base`` bundle member in whichever container it was written.

    Tries ``base.jsonl`` then ``base.yaml`` / ``base.yml``; returns ``[]`` when
    no member with that base is present (an absent optional member, e.g. a
    subjects-less corpus JSONL).
    """
    for ext, reader in _MEMBER_READERS.items():
        content = files.get(base + ext)
        if content is not None:
            return reader(content)
    return []


def _read_by_extension(name: str, content: str) -> list[dict[str, Any]]:
    """Parse a bundle member by its filename extension.

    Returns ``[]`` for a member with no recognised interchange extension (it is
    not an interchange container). Used by restore's single-member scan, which
    accepts an arbitrarily named ``*.corpus.jsonl`` / ``*.corpus.yaml`` file.
    """
    for ext, reader in _MEMBER_READERS.items():
        if name.endswith(ext):
            return reader(content)
    return []


async def export_store_bundle(session: AsyncSession, *, container: str = "jsonl") -> dict[str, str]:
    """Export a whole store to a portable bundle.

    Returns a filename -> content map: a ``manifest.json`` envelope plus a
    particles and a subjects member (the knowledge-graph core). ``container``
    selects the member serialization — ``"jsonl"`` (canonical, one unit per
    line) or ``"yaml"`` (the human-editable YAML-LD sibling); both
    carry the identical codec units and round-trip through the same import path.
    Trust statements, the event log, and corpus blobs are deferred bundle
    members; the manifest's ``members`` list names what is present.
    """
    if container not in _CONTAINER_WRITERS:
        raise ValueError(
            f"unknown interchange container {container!r}; "
            f"expected one of {sorted(_CONTAINER_WRITERS)}."
        )
    ext, writer = _CONTAINER_WRITERS[container]
    particles_member = f"{PARTICLES_BASE}{ext}"
    subjects_member = f"{SUBJECTS_BASE}{ext}"

    from particles.store.subject_store import list_all_subjects

    particle_units = await export_active(session)
    subject_units = [subject_to_unit(s) for s in await list_all_subjects(session)]
    manifest = {
        "formatVersion": FORMAT_VERSION,
        "schemaVersion": SCHEMA_VERSION,
        "exportedAt": datetime.now(UTC).isoformat(),
        "members": [particles_member, subjects_member],
        "counts": {"particles": len(particle_units), "subjects": len(subject_units)},
    }
    return {
        MANIFEST_NAME: json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        particles_member: writer(particle_units),
        subjects_member: writer(subject_units),
    }


async def import_store_bundle(session: AsyncSession, files: dict[str, str]) -> ImportSummary:
    """Import a store-export bundle into the current store (single-store writes).

    Subjects are imported first (so standalone subjects exist and particle refs
    resolve to them), then particles via the §6.6 ladder.
    """
    summary = ImportSummary()
    for unit in _read_member(files, SUBJECTS_BASE):
        if await _import_subject_unit(session, unit):
            summary.subjects_created += 1
    particle_summary = await import_units(session, _read_member(files, PARTICLES_BASE))
    summary.imported = particle_summary.imported
    summary.dropped = particle_summary.dropped
    summary.subjects_created += particle_summary.subjects_created
    return summary


async def _import_subject_unit(session: AsyncSession, unit: dict[str, Any]) -> bool:
    """Insert a standalone subject unless it already exists (by external ref or
    canonical name). Returns whether a new subject was created.
    """
    from particles.store.subject_store import (
        find_by_external_ref,
        find_by_name,
        insert_subject,
    )

    subject = subject_from_unit(unit)
    for ext in subject.external_ids:
        if await find_by_external_ref(session, ext.namespace, ext.id) is not None:
            return False
    if await find_by_name(session, subject.canonical_name) is not None:
        return False
    await insert_subject(session, subject)
    return True


# ---------------------------------------------------------------------------
# Restore — id-preserving, no-reconcile reconstruction
# ---------------------------------------------------------------------------


class RestoreError(RuntimeError):
    """Restore was asked to reconstruct a bundle into a non-empty store.

    Restore is reconstruct-only: it preserves origin UUIDs verbatim and bypasses
    the §6.6 ladder, so it is sound **only** into an empty target. A populated
    target risks an id collision and would let restore become a UUID-smuggling
    cross-store merge — exactly what the invariant forbids. The empty-target guard
    raises this instead, naming what was found so an operator can pick a fresh
    target store.
    """


@dataclass
class RestoreSummary:
    """Outcome of a faithful, id-preserving restore."""

    particles: int = 0
    subjects: int = 0


async def _assert_empty_target(session: AsyncSession) -> None:
    """Raise :class:`RestoreError` unless the target store has no particles.

    The guard that keeps the invariant intact: restore preserves origin ids and
    skips reconcile, so it is only sound into an empty store. Any existing
    particle (of any status) means the target is populated — refuse rather than
    risk an id collision or a silent merge.
    """
    from particles.store.particle_store import count_particles_by_status

    counts = await count_particles_by_status(session)
    total = sum(counts.values())
    if total:
        raise RestoreError(
            f"restore requires an empty target store, but it already holds {total} "
            f"particle(s) ({counts}). Restore reconstructs a bundle's own store with "
            "origin ids preserved; merging into a populated store is "
            "`interchange import` (claim-fingerprint identity). Restore "
            "into a fresh store instead."
        )


async def restore_store_bundle(session: AsyncSession, files: dict[str, str]) -> RestoreSummary:
    """Faithfully reconstruct a store-export bundle into an **empty** store.

    The id-preserving, no-reconcile sibling of :func:`import_store_bundle`.
    Subjects are restored first (origin ids preserved) so particle subject links
    resolve, then particles (origin ids preserved, subject_ids rebuilt against the
    restored subject ids) are inserted **directly** via the store — bypassing the
    §6.6 ladder, because the bundle is already-reconciled state, not new claims.

    Handles both bundle shapes:
      - the full ``export_store_bundle`` layout (``subjects.jsonl`` +
        ``particles.jsonl``); and
      - a single particles member (one ``*.corpus.jsonl`` / ``*.corpus.yaml``
        file, units of ``@type: Particle``) whose subjects ride only as inline
        refs on the particles — the projection-gate bundle. Inline
        subject refs carrying a ``sourceSubjectId`` are restored as subjects
        with that id.

    Refuses a non-empty target with :class:`RestoreError` (the guard
    rail). Flushes through the store inserts; the caller commits.
    """
    await _assert_empty_target(session)

    summary = RestoreSummary()

    # Collect every particle unit. The full bundle keeps them in a named member;
    # a single-member corpus JSONL is found by @type so the gate's *.corpus.jsonl
    # (whatever its member name) is handled without a naming convention.
    particle_units: list[dict[str, Any]] = list(_read_member(files, PARTICLES_BASE))
    if not particle_units:
        subjects_members = {SUBJECTS_BASE + ext for ext in _MEMBER_READERS}
        for name, content in files.items():
            if name == MANIFEST_NAME or name in subjects_members:
                continue
            for unit in _read_by_extension(name, content):
                if unit.get("@type") == "Particle":
                    particle_units.append(unit)

    parsed_particles = [from_unit(unit) for unit in particle_units]

    # 1. Restore subjects first, origin ids preserved. Standalone subject units
    #    (full bundle) take precedence; inline subject refs (corpus JSONL) fill in
    #    any subject a particle references but no standalone unit carried. A given
    #    origin id is restored exactly once.
    restored_subject_ids: set[str] = set()
    for unit in _read_member(files, SUBJECTS_BASE):
        sid = unit.get("sourceSubjectId")
        if sid is None or sid in restored_subject_ids:
            continue
        await _restore_subject(session, subject_from_unit(unit), sid)
        restored_subject_ids.add(sid)
        summary.subjects += 1

    for parsed in parsed_particles:
        for ref in parsed.subjects:
            sid = ref.source_subject_id
            if sid is None or sid in restored_subject_ids:
                continue
            await _restore_subject(session, _subject_from_ref(ref), sid)
            restored_subject_ids.add(sid)
            summary.subjects += 1

    # 2. Restore particles, origin ids preserved, subject_ids rebuilt against the
    #    restored subject ids (which are the origin ids — preserved verbatim).
    from particles.store.particle_store import insert_particle

    for parsed in parsed_particles:
        pid = parsed.source_particle_id
        if pid is None:
            raise RestoreError(
                "restore bundle carries a particle unit with no sourceParticleId; a "
                "faithful restore requires the origin id. The bundle is "
                "malformed for restore — use `interchange import` for fingerprint merge."
            )
        subject_ids = [
            ref.source_subject_id for ref in parsed.subjects if ref.source_subject_id is not None
        ]
        particle = parsed.particle.model_copy(update={"id": pid, "subject_ids": subject_ids})
        await insert_particle(session, particle)
        summary.particles += 1

    return summary


def _subject_from_ref(ref: SubjectRef) -> Subject:
    """Reconstruct a Subject from an inline particle subject ref.

    The corpus-JSONL bundle carries subjects only as inline refs on particles
    (no standalone ``subjects.jsonl``); this rebuilds the Subject so the restored
    store has the node the particle links to. The origin id is applied by the
    caller via :func:`_restore_subject`.
    """
    return Subject(
        canonical_name=ref.canonical_name or "(unnamed)",
        aliases=ref.aliases,
        external_ids=ref.external_refs,
        subject_class=ref.subject_class,
        asserted_by="interchange-restore",
    )


async def _restore_subject(session: AsyncSession, subject: Subject, origin_id: str) -> None:
    """Insert a subject with its origin id preserved verbatim."""
    from particles.store.subject_store import insert_subject

    await insert_subject(session, subject.model_copy(update={"id": origin_id}))
