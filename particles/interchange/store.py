# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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

Provenance refs travel as origin metadata and point at the *source* store's
corpus entries; the corpus itself moves only via store-export (Part C). A ref
that names a record of the *target* store is prefixed on the way in, so a
bundle never chooses which of this store's sources attest its claims.

Import and restore both refuse a bundle whose manifest ``schemaVersion`` is
incompatible with the SDK's before any member is read, and both report, per
key, the unit keys the codec dropped (D6).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.reanchor import REANCHOR_ACTOR
from particles.core.schema import (
    AssertionModality,
    ContributorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.extraction.polarity import POLARITY_KEY
from particles.extraction.scope import SCOPE_KEY
from particles.ingest.routing import skips_conflict_resolution
from particles.ingest.update_supersession import MCP_ASSERTER_PREFIX
from particles.interchange.codec import (
    FORMAT_VERSION,
    SubjectRef,
    check_schema_version,
    from_unit,
    particle_schema_default,
    subject_from_unit,
    subject_to_unit,
    to_unit,
)
from particles.interchange.jsonl import read_jsonl, write_jsonl
from particles.interchange.yaml_ld import read_yaml_ld, write_yaml_ld
from particles.store.taxonomy_store import RESERVED_TAGS


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
    """Outcome of an interchange import.

    ``dropped`` counts *particles* the §6.6 ladder declined to insert.
    ``dropped_keys`` counts unit *keys* this SDK does not know, per key name:
    the codec drops them on decode (D6), so a unit written by a newer
    minor of the schema imports as a record of this SDK's contract, and the
    report says what was left behind. ``restamped`` counts particles and
    Subjects whose unit named a reserved asserter, imported under
    :data:`IMPORT_ASSERTER` instead (Q34). ``retagged`` counts particles whose
    unit carried the memory-server migration's reserved tag, imported under
    :data:`CLAIMED_TAG_PREFIX` instead (the Q6 residual). ``reanchored``
    counts particles whose provenance named a record of the target store,
    imported with that id prefixed :data:`ORIGIN_REF_PREFIX` instead (R23).
    ``recalibrated`` counts particles whose unit claimed an operator's or an
    agent's calibration, imported as ``IMPORTED`` instead (R43).
    """

    imported: int = 0
    dropped: int = 0
    subjects_created: int = 0
    dropped_keys: dict[str, int] = field(default_factory=dict)
    restamped: int = 0
    retagged: int = 0
    reanchored: int = 0
    recalibrated: int = 0


#: The asserter an import stamps on a record it creates, and on a unit whose own
#: asserter is a reserved identity (security review 2026-10-25-1, Q34).
IMPORT_ASSERTER = "interchange-import"

#: The asserter a restored Subject carries (no key may hold it).
RESTORE_ASSERTER = "interchange-restore"


def _reserved_asserters() -> frozenset[str]:
    """The identities whose ``asserted_by`` is authority on this store.

    The memory-server façade owns, tombstones and relates by
    ``mcp.memory_compat.asserter_identity`` (N30, P50), and the agent verbs
    mutate what ``mcp.write.asserter_identity`` asserted. Only
    those servers may write under them. A bundle is bytes its author chose, so
    a unit naming either is attribution an import keeps, never authority it
    grants (security review 2026-10-25-1, Q34).
    """
    mcp = get_config().mcp
    return frozenset({mcp.write.asserter_identity, mcp.memory_compat.asserter_identity})


def _demote_reserved(
    asserted_by: str, contributors: list[ContributorRef] | None, at: datetime
) -> tuple[str, list[ContributorRef] | None] | None:
    """Re-stamp a reserved asserter, keeping the claimed one as a contributor.

    Reserved means one of the two configured server identities, or any
    ``mcp:``-prefixed one: the update rung and the agent verbs read that prefix
    as an agent's write (security review 2026-10-25-2, R43).
    Returns ``None`` when ``asserted_by`` is not reserved, so the record is
    imported as written. Otherwise returns :data:`IMPORT_ASSERTER` and the
    contributors with the unit's own asserter appended in the ``author`` role:
    provenance a reader can see, which no write surface keys on.
    """
    if not _is_reserved_asserter(asserted_by):
        return None
    return _restamp_asserter(asserted_by, contributors, at)


def _is_reserved_asserter(asserted_by: str) -> bool:
    return asserted_by in _reserved_asserters() or asserted_by.startswith(MCP_ASSERTER_PREFIX)


def _restamp_asserter(
    asserted_by: str, contributors: list[ContributorRef] | None, at: datetime
) -> tuple[str, list[ContributorRef] | None]:
    """:data:`IMPORT_ASSERTER`, with the claimed asserter appended as an ``author`` contributor."""
    if asserted_by == IMPORT_ASSERTER:
        return IMPORT_ASSERTER, contributors
    claimed = ContributorRef(id=asserted_by, role="author", at=at)
    return IMPORT_ASSERTER, [*(contributors or []), claimed]


#: The prefix an import puts on a reserved tag a unit carried, so the claim
#: stays visible as provenance and no reader keys on it (the Q6 residual).
CLAIMED_TAG_PREFIX = "interchange-claimed:"

#: Tags only a Particles server stamps, read as authority by the façade: the
#: memory-server migration's tag, which with ``CalibrationSource.IMPORTED``
#: makes a belief the façade's own memory (P50). The same set the tag
#: verbs refuse.
_RESERVED_TAGS = RESERVED_TAGS


def _restamp_reserved_tags(tags: list[str] | None) -> list[str] | None:
    """Prefix each reserved tag with :data:`CLAIMED_TAG_PREFIX`, order kept.

    Returns ``None`` when ``tags`` holds no reserved tag, so the particle is
    imported as written. The memory-server migration stamps its tag
    on what it extracts from a memory-server export, and the façade reads it with
    ``CalibrationSource.IMPORTED`` as its own memory. A bundle is bytes its
    author chose, and its unit may carry both, so the tag an import keeps is
    attribution, never the façade's mark (security review 2026-10-25-1, the
    interchange variant of Q6, beside Q34). The calibration source is left as
    written: it says how the confidence was set, and no reader treats it alone
    as authority.
    """
    if not tags or not _RESERVED_TAGS.intersection(tags):
        return None
    return [f"{CLAIMED_TAG_PREFIX}{t}" if t in _RESERVED_TAGS else t for t in tags]


#: The prefix an import puts on a provenance id that names a record of the
#: target store, so the ref stays as origin metadata and attests nothing here
#: (security review 2026-10-25-2, R23).
ORIGIN_REF_PREFIX = "interchange-origin:"

#: Calibration sources that mark a belief as the operator's or an agent's own:
#: the agent verbs refuse to mutate a ``HUMAN_REVIEW`` belief, and the update
#: rung never retires either. A bundle cannot claim them (R43).
_RESERVED_CALIBRATIONS = frozenset(
    {CalibrationSource.HUMAN_REVIEW, CalibrationSource.AGENT_ASSERTED}
)


#: The properties a reconciliation router reads to keep a claim off the
#: conflict surface (a document-meta scope, a non-asserted polarity). A bundle
#: that sets them keeps its claim out of arbitration and contradiction lint, so
#: an import keeps them only under the claimed prefix (security review
#: 2026-10-27-1, G29).
_ROUTING_PROPERTY_KEYS = (SCOPE_KEY, POLARITY_KEY)

#: The property an import keeps a bundle's declared modality under, beside the
#: ``FALSIFIABLE`` default it stores (G29).
CLAIMED_MODALITY_KEY = f"{CLAIMED_TAG_PREFIX}assertion_modality"


def _claimed_routing(particle: Particle) -> dict[str, object] | None:
    """``particle``'s properties with its arbitration exemptions kept as claims, or None.

    A non-``FALSIFIABLE`` modality and the routing properties each keep a claim
    out of the section 6.6 ladder and contradiction lint (``is_truth_apt``,
    ``skips_conflict_resolution``). A bundle chooses both, so the imported
    belief is stored ``FALSIFIABLE`` with neither exemption, and the declared
    values stay in its properties under :data:`CLAIMED_TAG_PREFIX`. The
    adjudicability default is regenerable: ``particles modality``
    or an operator verdict may withdraw it again (G29).
    """
    exempt = skips_conflict_resolution(particle.properties)
    modal = particle.assertion_modality is not AssertionModality.FALSIFIABLE
    if not exempt and not modal:
        return None
    properties = dict(particle.properties or {})
    if exempt:
        for key in _ROUTING_PROPERTY_KEYS:
            if key in properties:
                properties[f"{CLAIMED_TAG_PREFIX}{key}"] = properties.pop(key)
    if modal:
        properties[CLAIMED_MODALITY_KEY] = particle.assertion_modality.value
    return properties


@dataclass(frozen=True)
class _TargetRecords:
    """The ids an untrusted unit's provenance or predecessor must not name: this store's."""

    entry_ids: frozenset[str]
    particle_ids: frozenset[str]


async def _target_records(session: AsyncSession, particles: Iterable[Particle]) -> _TargetRecords:
    """Which ids these particles' provenance or ``supersedes`` name are this store's records."""
    from sqlalchemy import select

    from particles.corpus.store import CorpusEntryRow
    from particles.store.particle_store import ParticleRow

    entry_ids: set[str] = set()
    particle_ids: set[str] = set()
    for particle in particles:
        if particle.supersedes is not None and not particle.supersedes.startswith(
            ORIGIN_REF_PREFIX
        ):
            particle_ids.add(particle.supersedes)
        for ref in particle.provenance:
            if ref.corpus_entry_id.startswith(ORIGIN_REF_PREFIX):
                continue
            target = particle_ids if ref.type is ProvenanceRefType.PARTICLE else entry_ids
            target.add(ref.corpus_entry_id)

    async def _present(column: Any, ids: set[str]) -> frozenset[str]:
        found: set[str] = set()
        listed = sorted(ids)
        for start in range(0, len(listed), 500):
            rows = await session.execute(
                select(column).where(column.in_(listed[start : start + 500]))
            )
            found.update(rows.scalars())
        return frozenset(found)

    return _TargetRecords(
        entry_ids=await _present(CorpusEntryRow.entry_id, entry_ids),
        particle_ids=await _present(ParticleRow.id, particle_ids),
    )


def _reanchor_refs(
    provenance: list[ProvenanceRef], targets: _TargetRecords
) -> list[ProvenanceRef] | None:
    """Prefix every ref id naming a target-store record with :data:`ORIGIN_REF_PREFIX`.

    Returns ``None`` when no ref names one, so the provenance is kept as
    written. A bundle's refs point at the store it was exported from. A ref that
    names an entry or a particle of *this* store would let the bundle choose
    which of its sources attest the claim, and so which projects see it and
    which source trust it borrows (security review 2026-10-25-2, R23). The
    prefixed id keeps the origin visible and resolves to nothing here.
    """

    def _names_target(ref: ProvenanceRef) -> bool:
        ids = targets.particle_ids if ref.type is ProvenanceRefType.PARTICLE else targets.entry_ids
        return ref.corpus_entry_id in ids

    if not any(_names_target(ref) for ref in provenance):
        return None
    return [
        ref.model_copy(
            update={
                "corpus_entry_id": f"{ORIGIN_REF_PREFIX}{ref.corpus_entry_id}",
                "snapshot_id": (
                    f"{ORIGIN_REF_PREFIX}{ref.snapshot_id}" if ref.snapshot_id else None
                ),
            }
        )
        if _names_target(ref)
        else ref
        for ref in provenance
    ]


def _restamp_reserved_contributors(
    contributors: list[ContributorRef] | None,
) -> list[ContributorRef] | None:
    """Prefix a contributor that names the re-anchor pass, so no reader keys on it.

    Returns ``None`` when no contributor does. The update rung exempts a belief
    the re-anchor pass wrote, and reads that from its contributors, so a bundle
    listing the pass would exempt its own claims from retirement (R43).
    """
    if not contributors or not any(c.id == REANCHOR_ACTOR for c in contributors):
        return None
    return [
        c.model_copy(update={"id": f"{CLAIMED_TAG_PREFIX}{c.id}"}) if c.id == REANCHOR_ACTOR else c
        for c in contributors
    ]


@dataclass
class _Neutralised:
    """What :func:`_neutralise_particle` changed on one particle."""

    restamped: bool = False
    retagged: bool = False
    reanchored: bool = False
    recalibrated: bool = False


def _neutralise_particle(
    particle: Particle, targets: _TargetRecords
) -> tuple[Particle, _Neutralised]:
    """Turn every server-owned mark an untrusted unit carries into attribution.

    A bundle is bytes its author chose. Each mark a reader keys on as authority
    (an asserter, the migration tag, the re-anchor contributor, an operator's
    or an agent's calibration, a ref naming this store's records, a
    predecessor in this store, a modality or routing property that exempts the
    claim from arbitration) is kept where a reader can see it and taken out of
    what any reader keys on (Q34, Q6, R23, R26, R43, G29).

    Every asserter is re-stamped :data:`IMPORT_ASSERTER`, not only a reserved
    one: under per-client keys any principal name can be an agent's, and that
    agent's own-beliefs guard keys on ``asserted_by`` (security
    review 2026-10-26-1, W38). ``restamped`` still counts only the reserved
    ones, which the import notice names.
    """
    update: dict[str, Any] = {}
    done = _Neutralised()
    contributors = particle.contributors
    if particle.asserted_by != IMPORT_ASSERTER:
        update["asserted_by"], contributors = _restamp_asserter(
            particle.asserted_by, contributors, particle.asserted_at
        )
        done.restamped = _is_reserved_asserter(particle.asserted_by)
    restamped = _restamp_reserved_contributors(contributors)
    if restamped is not None:
        contributors = restamped
        done.restamped = True
    if contributors is not particle.contributors:
        update["contributors"] = contributors
    retagged = _restamp_reserved_tags(particle.tags)
    if retagged is not None:
        update["tags"] = retagged
        done.retagged = True
    reanchored = _reanchor_refs(particle.provenance, targets)
    if reanchored is not None:
        update["provenance"] = reanchored
        done.reanchored = True
    if particle.supersedes is not None and particle.supersedes in targets.particle_ids:
        # A predecessor naming a belief of this store would let the bundle date
        # that belief's retirement and stand as its successor (W39).
        update["supersedes"] = f"{ORIGIN_REF_PREFIX}{particle.supersedes}"
        done.reanchored = True
    claimed = _claimed_routing(particle)
    if claimed is not None:
        update["properties"] = claimed
        update["assertion_modality"] = AssertionModality.FALSIFIABLE
    if particle.confidence.calibration_source in _RESERVED_CALIBRATIONS:
        update["confidence"] = particle.confidence.model_copy(
            update={"calibration_source": CalibrationSource.IMPORTED}
        )
        done.recalibrated = True
    return (particle.model_copy(update=update) if update else particle), done


def _count_neutralised(summary: ImportSummary | RestoreSummary, done: _Neutralised) -> None:
    summary.restamped += done.restamped
    summary.retagged += done.retagged
    summary.reanchored += done.reanchored
    summary.recalibrated += done.recalibrated


def _count_dropped_keys(target: dict[str, int], keys: Iterable[str]) -> None:
    """Add one to ``target[key]`` for every key in ``keys``."""
    for key in keys:
        target[key] = target.get(key, 0) + 1


async def import_units(
    session: AsyncSession,
    units: Iterable[dict[str, Any]],
    *,
    store: str | None = None,
    subject_units: Iterable[dict[str, Any]] = (),
) -> ImportSummary:
    """Import interchange units into the current store (single-store write, §6.6).

    Subjects are resolved by external reference first (the cross-store join key),
    then by canonical name; an unmatched named subject is created bare-local.
    Each particle is reconciled via the §6.6 ladder; a particle dropped by trust
    resolution is counted in ``dropped``. Unit keys the codec does not know are
    counted per key in ``dropped_keys``. A unit asserted by a reserved
    identity (the façade's, the agent's, or any ``mcp:`` one) is re-stamped
    :data:`IMPORT_ASSERTER`, its asserter kept as an ``author`` contributor,
    and counted in ``restamped`` (security review 2026-10-25-1, Q34), as is
    one listing the re-anchor pass as a contributor (R43). A unit carrying the
    memory-server migration's tag has it prefixed with
    :data:`CLAIMED_TAG_PREFIX`, and is counted in ``retagged`` (the Q6
    residual; see :func:`_restamp_reserved_tags`). A provenance ref naming a
    corpus entry or a particle of the target store is prefixed
    :data:`ORIGIN_REF_PREFIX` (``reanchored``), and a duplicate of an ACTIVE
    belief appends none of its refs to it (R23). An operator's or an agent's
    calibration source is imported as ``IMPORTED`` (``recalibrated``, R43).

    Every unit's subjects are resolved first, committed under the store's
    writer lock in one transaction with ``subject_units`` (a bundle's
    standalone subjects), so a subject reference that cannot be built into a
    Subject refuses the batch with none of them committed (security review
    2026-10-31-2, AE7). Then each particle's embedding and model probes run before
    the lock, and its write commits under it, so no probe runs while this
    import holds the lock or SQLite's write lock (security review 2026-10-31-1,
    AD15). The function therefore commits.

    Each unit commits on its own, so other writers commit between two units.
    A unit's apply searches the store as it stands under the lock: a claim
    retired meanwhile is no partner, and a claim committed meanwhile that its
    gather never probed holds the imported claim for review behind an
    INCONSISTENCY, never corroborated (security review 2026-10-31-2, AE1).
    The units committed before a failure stay committed.

    The §6.6 trust regime is ``store``'s own (``reconciliation_mode_for``;
    ``None`` is the default store), not
    the process-global ``reconciliation.store_mode``, so a bundle imported into
    a write-enabled store reconciles in ``multi`` as an agent's assertion would
    (security review 2026-10-26-1, the interchange half of P30).

    Raises:
        IncompatibleSchemaVersionError: If any unit's ``schemaVersion`` is
            incompatible with the SDK's (D6). Every unit is checked
            before the first write, so a refused batch writes nothing.
    """
    from particles.db import write_transaction
    from particles.ingest.pipeline import apply_reconcile, gather_reconcile

    units = list(units)
    for unit in units:
        check_schema_version(
            unit.get("schemaVersion", particle_schema_default()), what="interchange unit"
        )

    summary = ImportSummary()
    single_trust_order = get_config().reconciliation_mode_for(store) == "single"
    parsed_units = [from_unit(unit) for unit in units]
    targets = await _target_records(session, (parsed.particle for parsed in parsed_units))
    unit_subjects: list[list[str]] = []
    async with write_transaction(session, store):
        for subject_unit in subject_units:
            created, restamped = await _import_subject_unit(session, subject_unit)
            summary.subjects_created += created
            summary.restamped += restamped
        for parsed in parsed_units:
            subject_ids: list[str] = []
            for ref in parsed.subjects:
                sid, created = await _resolve_subject_ref(session, ref)
                if sid is not None:
                    subject_ids.append(sid)
                if created:
                    summary.subjects_created += 1
            unit_subjects.append(subject_ids)
    for parsed, subject_ids in zip(parsed_units, unit_subjects, strict=True):
        _count_dropped_keys(summary.dropped_keys, parsed.dropped_keys)
        particle, done = _neutralise_particle(parsed.particle, targets)
        _count_neutralised(summary, done)
        particle = particle.model_copy(update={"subject_ids": subject_ids})
        # An imported duplicate adds nothing to the belief it duplicates: the
        # bundle's refs are origin metadata, and appending them would let a
        # bundle widen that belief's sources (R23). Each unit's gather reads the
        # store as it stands, through the process-wide candidate cache, never a
        # list loaded before an earlier unit's commit, and its apply searches
        # the store again under the lock (security review 2026-10-31-2, AE1).
        gathered = await gather_reconcile(session, particle)
        async with write_transaction(session, store):
            inserted = await apply_reconcile(
                session,
                particle,
                gathered,
                append_provenance=False,
                single_trust_order=single_trust_order,
            )
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
        persona_alias_guard,
    )

    for ext in ref.external_refs:
        existing = await find_by_external_ref(session, ext.namespace, ext.id)
        if existing is not None:
            return existing.id, False

    if ref.canonical_name:
        # A binding path: an imported Subject named "User" or "I" is not this
        # store's conversational persona, whatever forms its fold has recorded.
        by_name = await find_by_name(
            session,
            ref.canonical_name,
            skip_aliases_of=persona_alias_guard(ref.canonical_name, None),
        )
        if by_name is not None:
            return by_name.id, False
        subject = Subject(
            canonical_name=ref.canonical_name,
            aliases=ref.aliases,
            external_ids=ref.external_refs,
            subject_class=ref.subject_class,
            asserted_by=IMPORT_ASSERTER,
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
        "schemaVersion": particle_schema_default(),
        "exportedAt": datetime.now(UTC).isoformat(),
        "members": [particles_member, subjects_member],
        "counts": {"particles": len(particle_units), "subjects": len(subject_units)},
    }
    return {
        MANIFEST_NAME: json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        particles_member: writer(particle_units),
        subjects_member: writer(subject_units),
    }


def _check_bundle_manifest(files: dict[str, str]) -> None:
    """Refuse a bundle whose manifest is stamped with an incompatible schema (D6).

    Runs before any member is read, so an incompatible bundle writes nothing. A
    bundle with no manifest (restore's single-member corpus JSONL) or a
    manifest without ``schemaVersion`` passes here; each unit is still checked
    as it is decoded.

    Raises:
        IncompatibleSchemaVersionError: If the manifest's ``schemaVersion`` is
            of another major, major ``0``, or malformed.
        ValueError: If the manifest is not a JSON object.
    """
    raw = files.get(MANIFEST_NAME)
    if raw is None:
        return
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"bundle {MANIFEST_NAME} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"bundle {MANIFEST_NAME} must be a JSON object")
    if "schemaVersion" in manifest:
        check_schema_version(manifest["schemaVersion"], what="bundle manifest")


async def import_store_bundle(
    session: AsyncSession, files: dict[str, str], *, store: str | None = None
) -> ImportSummary:
    """Import a store-export bundle into the current store (single-store writes).

    Subjects are imported first (so standalone subjects exist and particle refs
    resolve to them), then particles via the §6.6 ladder. A manifest stamped
    with an incompatible ``schemaVersion`` is refused before any member is read
    (D6).
    """
    _check_bundle_manifest(files)
    subject_units = list(_read_member(files, SUBJECTS_BASE))
    particle_units = list(_read_member(files, PARTICLES_BASE))
    # The standalone subjects commit with the particles' subject references, in
    # one transaction under the store's writer lock before any particle's probes
    # run (security review 2026-10-31-1, AD15), so a refused bundle writes
    # nothing: every unit is checked and decoded before it, and a reference that
    # cannot be built rolls the subjects back (2026-10-31-2, AE7).
    return await import_units(session, particle_units, store=store, subject_units=subject_units)


async def _import_subject_unit(session: AsyncSession, unit: dict[str, Any]) -> tuple[bool, bool]:
    """Insert a standalone subject unless it already exists (by external ref or
    canonical name). Returns whether a new subject was created, and whether its
    reserved asserter was re-stamped (Q34; see :func:`_demote_reserved`).
    """
    from particles.store.subject_store import (
        find_by_external_ref,
        find_by_name,
        insert_subject,
        persona_alias_guard,
    )

    subject = subject_from_unit(unit)
    for ext in subject.external_ids:
        if await find_by_external_ref(session, ext.namespace, ext.id) is not None:
            return False, False
    guard = persona_alias_guard(subject.canonical_name, None)
    if await find_by_name(session, subject.canonical_name, skip_aliases_of=guard) is not None:
        return False, False
    demoted = _demote_reserved(subject.asserted_by, subject.contributors, subject.created_at)
    if demoted is not None:
        asserted_by, contributors = demoted
        subject = subject.model_copy(
            update={"asserted_by": asserted_by, "contributors": contributors}
        )
    await insert_subject(session, subject)
    return True, demoted is not None


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
    #: Unit keys this SDK does not know, counted per key (D6).
    dropped_keys: dict[str, int] = field(default_factory=dict)
    #: Server-owned marks an untrusted restore turned into attribution, counted
    #: as on :class:`ImportSummary` (R26). All zero under ``trust_bundle``.
    restamped: int = 0
    retagged: int = 0
    reanchored: int = 0
    recalibrated: int = 0


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


async def restore_store_bundle(
    session: AsyncSession, files: dict[str, str], *, trust_bundle: bool = False
) -> RestoreSummary:
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
    rail), and a manifest stamped with an incompatible ``schemaVersion`` with
    :class:`~particles.interchange.codec.IncompatibleSchemaVersionError` before
    any member is read (D6). Flushes through the store inserts; the
    caller commits.

    Unless ``trust_bundle`` is set, every server-owned mark a unit carries is
    turned into attribution exactly as :func:`import_units` does (a reserved
    asserter on a particle or a Subject, the migration tag, the re-anchor
    contributor, an operator's or an agent's calibration, a ref naming a corpus
    entry the target already holds), and counted on the summary. A bundle a
    user is handed would otherwise restore as the memory-server façade's own
    memory and Subjects (security review 2026-10-25-2, R26). ``trust_bundle``
    restores the marks verbatim, for a bundle the operator exported from their
    own store.
    """
    _check_bundle_manifest(files)
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
    for parsed in parsed_particles:
        _count_dropped_keys(summary.dropped_keys, parsed.dropped_keys)

    # 1. Restore subjects first, origin ids preserved. Standalone subject units
    #    (full bundle) take precedence; inline subject refs (corpus JSONL) fill in
    #    any subject a particle references but no standalone unit carried. A given
    #    origin id is restored exactly once.
    restored_subject_ids: set[str] = set()
    for unit in _read_member(files, SUBJECTS_BASE):
        sid = unit.get("sourceSubjectId")
        if sid is None or sid in restored_subject_ids:
            continue
        subject = subject_from_unit(unit)
        demoted = (
            None
            if trust_bundle
            else _demote_reserved(subject.asserted_by, subject.contributors, subject.created_at)
        )
        if demoted is not None:
            asserted_by, contributors = demoted
            subject = subject.model_copy(
                update={"asserted_by": asserted_by, "contributors": contributors}
            )
            summary.restamped += 1
        await _restore_subject(session, subject, sid)
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

    # The target holds no particle (checked above), so only its corpus entries
    # can be named; a PARTICLE ref names another restored particle by its
    # preserved origin id, which is the bundle's own record.
    targets = _TargetRecords(entry_ids=frozenset(), particle_ids=frozenset())
    if not trust_bundle:
        found = await _target_records(session, (parsed.particle for parsed in parsed_particles))
        targets = _TargetRecords(entry_ids=found.entry_ids, particle_ids=frozenset())
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
        particle = parsed.particle
        if not trust_bundle:
            particle, done = _neutralise_particle(particle, targets)
            _count_neutralised(summary, done)
        particle = particle.model_copy(update={"id": pid, "subject_ids": subject_ids})
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
        asserted_by=RESTORE_ASSERTER,
    )


async def _restore_subject(session: AsyncSession, subject: Subject, origin_id: str) -> None:
    """Insert a subject with its origin id preserved verbatim."""
    from particles.store.subject_store import insert_subject

    await insert_subject(session, subject.model_copy(update={"id": origin_id}))
