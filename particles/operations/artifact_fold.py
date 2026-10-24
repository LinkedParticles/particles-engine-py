# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fold a file's other spellings into its path Subject.

The shell around :func:`particles.core.artifact_fold.plan_artifact_fold`:
gather one namespace's file Subjects and the store's dotted-name Subjects,
decide, then merge each fold as its own ``SUBJECTS_MERGED`` event. The
repository's file list is the operator's to supply (the CLI reads it from a
checkout), because the store sees only part of a repository and its own paths
are a poor oracle for which bare name is unique.

Every fold is undoable from its event alone: the event records the rule, the
source Subject as it was, the refs it carried and the links the target
already had, and :func:`undo_fold` reverses it with
:func:`~particles.store.subject_store.restore_merged_source`. ``subjects
split`` cannot, since it leaves a carried ref on the target and the next
mention would merge the two again.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.artifact_fold import (
    ARTIFACT_REF_NAMESPACE,
    FileSubject,
    Fold,
    FoldPlan,
    ModuleCandidate,
    Refusal,
    artifact_ref_name,
    artifact_ref_project,
    is_absolute_path,
    module_files,
    plan_artifact_fold,
)
from particles.core.schema import ExternalRef, Subject
from particles.core.status import Status
from particles.extraction.subject_gate import is_file_extension
from particles.ingest.artifact_namespace import artifact_relative_path
from particles.store.event_store import EventRefKind, OperatorEventType, get_event, list_events
from particles.store.particle_store import ParticleRow
from particles.store.subject_store import (
    ParticleSubjectRow,
    SubjectRow,
    get_subject,
    merge_subjects_many,
    restore_merged_source,
)

#: The actor every fold and its undo records.
FOLD_ACTOR = "subjects-fold-artifacts"

#: The ``subject_class`` of the Subjects a fold merges into.
FILE_SUBJECT_CLASS = "artifact:file"

_IN_CHUNK = 500


@dataclass(frozen=True)
class ClaimSpan:
    """How many ACTIVE claims a Subject has, and when they were asserted."""

    claims: int = 0
    first: datetime | None = None
    last: datetime | None = None


@dataclass
class FoldReport:
    """A namespace's fold plan with the evidence an operator reads it by."""

    namespace_key: str
    repo_files: int | None
    folds: list[Fold] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    spans: dict[str, ClaimSpan] = field(default_factory=dict)
    applied_events: list[str] = field(default_factory=list)


async def gather_fold_inputs(
    session: AsyncSession, namespace_key: str
) -> tuple[list[FileSubject], list[ModuleCandidate], set[str]]:
    """The namespace's file Subjects, the dotted-name candidates, and foreign paths.

    A file Subject's path is the name its ``artifact`` ref carries, which is
    the normalized spelling its identity was minted from. A path with a file
    Subject in another namespace is foreign, for the §5 single-project rule.
    """
    rows = await session.execute(
        select(
            SubjectRow.id,
            SubjectRow.canonical_name,
            SubjectRow.subject_class,
            SubjectRow.external_ids_json,
        )
    )
    files: list[FileSubject] = []
    modules: list[ModuleCandidate] = []
    foreign: set[str] = set()
    for sid, name, subject_class, refs_json in rows.all():
        refs: list[dict[str, object]] = json.loads(refs_json)
        artifact = [str(r["id"]) for r in refs if r.get("namespace") == ARTIFACT_REF_NAMESPACE]
        if subject_class == FILE_SUBJECT_CLASS and artifact:
            for ref_id in artifact:
                if artifact_ref_project(ref_id) == namespace_key:
                    files.append(FileSubject(subject_id=sid, path=artifact_ref_name(ref_id)))
                    break
            else:
                foreign.update(artifact_ref_name(r) for r in artifact)
        if module_files(name) is not None:
            modules.append(
                ModuleCandidate(
                    subject_id=sid,
                    name=name,
                    subject_class=subject_class,
                    has_external_refs=bool(refs),
                )
            )
    return files, modules, foreign


async def plan_namespace_fold(
    session: AsyncSession, namespace_key: str, repo_files: Collection[str] | None
) -> FoldPlan:
    """Gather and decide one namespace's fold. Reads only."""
    files, modules, foreign = await gather_fold_inputs(session, namespace_key)
    relative: dict[str, str] = {}
    for f in files:
        if is_absolute_path(f.path):
            rel = artifact_relative_path(namespace_key, f.path)
            if rel:
                relative[f.subject_id] = rel
    return plan_artifact_fold(
        files,
        relative_paths=relative,
        repo_files=repo_files,
        modules=modules,
        foreign_paths=foreign,
        is_file_extension=is_file_extension,
    )


async def claim_spans(session: AsyncSession, subject_ids: Sequence[str]) -> dict[str, ClaimSpan]:
    """ACTIVE claim count and assertion-date range per Subject."""
    spans: dict[str, ClaimSpan] = {}
    ids = list(dict.fromkeys(subject_ids))
    for i in range(0, len(ids), _IN_CHUNK):
        chunk = ids[i : i + _IN_CHUNK]
        rows = await session.execute(
            select(
                ParticleSubjectRow.subject_id,
                func.count(),
                func.min(ParticleRow.asserted_at),
                func.max(ParticleRow.asserted_at),
            )
            .join(ParticleRow, ParticleRow.id == ParticleSubjectRow.particle_id)
            .where(
                ParticleSubjectRow.subject_id.in_(chunk),
                ParticleRow.status == Status.ACTIVE.value,
            )
            .group_by(ParticleSubjectRow.subject_id)
        )
        for sid, n, first, last in rows.all():
            spans[sid] = ClaimSpan(claims=int(n), first=first, last=last)
    return spans


def _source_record(subject: Subject) -> dict[str, object]:
    return {
        "id": subject.id,
        "canonical_name": subject.canonical_name,
        "description": subject.description,
        "aliases": list(subject.aliases),
        "external_ids": [r.model_dump(mode="json") for r in subject.external_ids],
        "subject_class": subject.subject_class,
        "asserted_by": subject.asserted_by,
        "created_at": subject.created_at.isoformat(),
    }


async def apply_fold(
    session: AsyncSession, plan: FoldPlan, *, actor: str = FOLD_ACTOR
) -> list[str]:
    """Merge each planned fold, in plan order, one event per source.

    A source already gone (folded by an earlier step, or merged meanwhile) is
    skipped. Returns the ``SUBJECTS_MERGED`` event ids. Caller commits.
    """
    event_ids: list[str] = []
    for fold in plan.folds:
        source = await get_subject(session, fold.source_id)
        if source is None or await get_subject(session, fold.target_id) is None:
            continue
        await merge_subjects_many(
            session,
            [fold.source_id],
            fold.target_id,
            actor=actor,
            extra_payload={"rule": fold.rule.value, "source_record": _source_record(source)},
        )
        merged = await list_events(
            session,
            ref_kind=EventRefKind.SUBJECT,
            ref_id=fold.source_id,
            event_type=OperatorEventType.SUBJECTS_MERGED,
            limit=1,
        )
        if merged:
            event_ids.append(merged[0].event_id)
    return event_ids


async def build_fold_report(
    session: AsyncSession,
    namespace_key: str,
    repo_files: Collection[str] | None,
    *,
    apply: bool = False,
    actor: str = FOLD_ACTOR,
) -> FoldReport:
    """Plan one namespace's fold, with claim spans, and apply it when asked."""
    plan = await plan_namespace_fold(session, namespace_key, repo_files)
    ids = [f.source_id for f in plan.folds] + [f.target_id for f in plan.folds]
    report = FoldReport(
        namespace_key=namespace_key,
        repo_files=None if repo_files is None else len(repo_files),
        folds=plan.folds,
        refusals=plan.refusals,
        spans=await claim_spans(session, ids),
    )
    if apply:
        report.applied_events = await apply_fold(session, plan, actor=actor)
    return report


async def undo_fold(session: AsyncSession, event_id: str, *, actor: str = FOLD_ACTOR) -> list[str]:
    """Reverse one fold from its ``SUBJECTS_MERGED`` event. Caller commits.

    Returns the particle ids moved back onto the recreated source.

    Raises:
        ValueError: The event is not a fold, or it was already undone, or its
            target no longer exists.
    """
    event = await get_event(session, event_id)
    payload = (event.payload or {}) if event is not None else {}
    record = payload.get("source_record")
    if (
        event is None
        or event.event_type != OperatorEventType.SUBJECTS_MERGED
        or not isinstance(record, dict)
    ):
        raise ValueError(f"Event {event_id} is not an artifact fold")
    source = Subject(
        id=str(record["id"]),
        canonical_name=str(record["canonical_name"]),
        description=record.get("description"),
        aliases=list(record.get("aliases") or []),
        external_ids=[ExternalRef.model_validate(r) for r in record.get("external_ids") or []],
        subject_class=record.get("subject_class"),
        asserted_by=str(record.get("asserted_by") or actor),
        created_at=datetime.fromisoformat(str(record["created_at"])),
    )
    particle_ids = [r.ref_id for r in event.refs if r.ref_kind == EventRefKind.PARTICLE]
    return await restore_merged_source(
        session,
        source=source,
        target_id=str(payload["target"]),
        particle_ids=particle_ids,
        keep_on_target=list(payload.get("already_on_target") or []),
        aliases_added=list(payload.get("aliases_added") or []),
        refs_carried=list(payload.get("refs_carried") or []),
        actor=actor,
        payload={"reverses": event_id, "rule": payload.get("rule")},
    )
