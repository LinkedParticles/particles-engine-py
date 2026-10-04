# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Attach Subjects to subjectless beliefs in place.

Two callers share one write. The batch path recovers the names the non-entity
gate stripped from existing orphans and relinks them, scoped by
their project (``particles subjects relink-gated``, the ``gated_subjects``
curation card). The per-card path attaches the one Subject an operator chose
(the ``assign-subject`` gesture). Neither supersedes: a subject link is an
annotation, changed in place, so the belief keeps its id, its
utility evidence and its observer scope.

Recovery is deterministic and reads only what the particle itself carries, in
tier order, stopping at the first tier that yields a name:

1. the ``extraction:gated_subjects`` record (particles minted from 1.159.4);
2. the structured claim's subject term, when it is still an unbound
   ``TOKEN`` the gate would qualify;
3. the claim's backtick spans the gate would qualify.

It never invents a name the particle does not spell, and a name with no
namespace key is not recovered (the fail-closed rule).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import SubjectGateConfig, get_config
from particles.core.observer_scope import project_keys
from particles.core.schema import ParticleType
from particles.extraction.subject_gate import (
    GATED_SUBJECTS_KEY,
    QUALIFY,
    classify_non_entity,
    is_qualifiable,
)
from particles.extraction.subject_scope import subject_expected
from particles.ingest.artifact_namespace import artifact_namespace_for
from particles.ingest.authorities.registry import RecognizeContext
from particles.ingest.subject_resolver import resolve_qualified_subject
from particles.store.event_store import EventRefKind, OperatorEventType, record_event

#: The actor a batch relink records when the caller names none.
RELINK_ACTOR = "subjects-relink-gated"

_BACKTICK_SPAN = re.compile(r"`([^`\n]+)`")


@dataclass(frozen=True)
class RecoveredName:
    """One subject name recovered from an orphan, with how it was found."""

    name: str
    token_class: str
    tier: int


def _qualifies(name: str, cfg: SubjectGateConfig) -> str | None:
    """The gate class of ``name`` when the gate is configured to qualify it."""
    token_class = classify_non_entity(name, cli_binaries=cfg.cli_binaries, allowlist=cfg.allowlist)
    if token_class is None or cfg.dispositions.get(token_class) != QUALIFY:
        return None
    return token_class if is_qualifiable(name, token_class) else None


def recover_names(
    *,
    properties: Mapping[str, object] | None,
    structured_subject: Mapping[str, object] | None,
    content: str,
    tiers: Sequence[int],
    cfg: SubjectGateConfig,
) -> list[RecoveredName]:
    """The names the first productive tier recovers from one orphan. Pure.

    ``structured_subject`` is the stored structured claim's subject term
    (``{"kind": ..., "value": ...}``) or ``None``. Names are deduplicated in
    order. An empty result means no enabled tier found a qualifiable name.
    """
    for tier in sorted(set(tiers)):
        raw: list[str] = []
        if tier == 1:
            record = (properties or {}).get(GATED_SUBJECTS_KEY)
            if isinstance(record, list):
                raw = [str(e.get("name", "")) for e in record if isinstance(e, dict)]
        elif tier == 2:
            if structured_subject and structured_subject.get("kind") == "TOKEN":
                raw = [str(structured_subject.get("value", ""))]
        elif tier == 3:
            raw = _BACKTICK_SPAN.findall(content)
        found: list[RecoveredName] = []
        seen: set[str] = set()
        for name in raw:
            name = name.strip()
            if not name or name in seen:
                continue
            token_class = _qualifies(name, cfg)
            if token_class is not None:
                seen.add(name)
                found.append(RecoveredName(name=name, token_class=token_class, tier=tier))
        if found:
            return found
    return []


@dataclass
class RelinkItem:
    """One orphan the plan would relink."""

    particle_id: str
    content: str
    namespace_key: str
    names: list[RecoveredName]


@dataclass
class RelinkPlan:
    """What a relink would do, computed read-only.

    ``items`` are the orphans with a recovered name and a namespace key.
    ``fail_closed_no_key`` / ``fail_closed_several`` count orphans whose names
    were recovered but whose entry has no project key, or tags that fold to
    more than one project, so the gate suppresses them.
    """

    tiers: list[int]
    orphans: int = 0
    items: list[RelinkItem] = field(default_factory=list)
    by_tier: Counter[int] = field(default_factory=Counter)
    by_class: Counter[str] = field(default_factory=Counter)
    fail_closed_no_key: int = 0
    fail_closed_several: int = 0
    unrecovered: int = 0

    def sample(self, n: int, seed: int) -> list[RelinkItem]:
        """A reproducible random sample of ``n`` items, for the precision check."""
        if n >= len(self.items):
            return list(self.items)
        return random.Random(seed).sample(self.items, n)


async def plan_gated_relink(
    session: AsyncSession,
    *,
    tiers: Sequence[int] | None = None,
    particle_ids: Sequence[str] | None = None,
) -> RelinkPlan:
    """Recover names for every subjectless belief that owes a subject. Read-only.

    Walks the ACTIVE particles with no subject link that ``subject_expected``
    says should have one, so DOCUMENT_META and non-asserted claims
    are never touched. ``tiers`` defaults to ``subject_gate.relink_tiers``.
    ``particle_ids`` narrows the walk (a test seam and a future per-entry
    card). No write, no LLM, no network.
    """
    from particles.corpus.store import list_entry_tag_rows
    from particles.store.particle_store import ParticleRow
    from particles.store.subject_store import ParticleSubjectRow

    cfg = get_config().subject_gate
    chosen = list(tiers) if tiers is not None else list(cfg.relink_tiers)
    plan = RelinkPlan(tiers=sorted(set(chosen)))

    linked = select(ParticleSubjectRow.particle_id).where(
        ParticleSubjectRow.particle_id == ParticleRow.id
    )
    query = select(
        ParticleRow.id,
        ParticleRow.content,
        ParticleRow.particle_type,
        ParticleRow.properties_json,
        ParticleRow.structured_claim_json,
        ParticleRow.provenance_json,
    ).where(
        ParticleRow.status == "ACTIVE",
        ParticleRow.subject_ids_json == "[]",
        ~linked.exists(),
    )
    if particle_ids is not None:
        query = query.where(ParticleRow.id.in_(list(particle_ids)))
    rows = (await session.execute(query)).all()
    if not rows:
        return plan

    entries = {eid: (uri, tags) for eid, uri, tags in await list_entry_tag_rows(session)}
    keys: dict[str, str | None] = {}

    for pid, content, ptype, props_json, sc_json, prov_json in rows:
        properties = json.loads(props_json) if props_json else None
        try:
            particle_type = ParticleType(ptype)
        except ValueError:
            continue
        if not subject_expected(particle_type, properties):
            continue
        plan.orphans += 1
        subject_term = json.loads(sc_json).get("subject") if sc_json else None
        names = recover_names(
            properties=properties,
            structured_subject=subject_term if isinstance(subject_term, dict) else None,
            content=content,
            tiers=plan.tiers,
            cfg=cfg,
        )
        if not names:
            plan.unrecovered += 1
            continue
        entry_id = _first_source_entry(prov_json) or ""
        uri, tags = entries.get(entry_id, (None, []))
        if entry_id not in keys:
            keys[entry_id] = artifact_namespace_for(tags, uri)
        key = keys[entry_id]
        if key is None:
            if project_keys(tags):
                plan.fail_closed_several += 1
            else:
                plan.fail_closed_no_key += 1
            continue
        plan.items.append(
            RelinkItem(particle_id=pid, content=content, namespace_key=key, names=names)
        )
        plan.by_tier[names[0].tier] += 1
        for n in names:
            plan.by_class[n.token_class] += 1
    return plan


def _first_source_entry(provenance_json: str) -> str | None:
    try:
        refs = json.loads(provenance_json)
    except ValueError:
        return None
    for ref in refs if isinstance(refs, list) else []:
        if isinstance(ref, dict) and ref.get("type") == "SOURCE" and ref.get("corpus_entry_id"):
            return str(ref["corpus_entry_id"])
    return None


@dataclass
class RelinkResult:
    """What an applied relink wrote."""

    relinked: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    event_id: str | None = None


async def _attach_and_bind(
    session: AsyncSession, particle_id: str, subject_ids: list[str], names: list[str]
) -> list[str]:
    """Attach in place and bind the structured claim's subject term when it names one."""
    from particles.extraction.structure import bind_subject_id
    from particles.store.particle_store import get_particle, set_structured_claim
    from particles.store.subject_store import attach_subjects_in_place

    linked = await attach_subjects_in_place(session, particle_id, subject_ids)
    particle = await get_particle(session, particle_id)
    if (
        particle is not None
        and particle.structured_claim is not None
        and particle.structured_claim.subject_id is None
        and names
    ):
        bound = bind_subject_id(particle.structured_claim, names, subject_ids)
        if bound.subject_id is not None:
            await set_structured_claim(session, particle_id, bound)
    return linked


async def apply_gated_relink(
    session: AsyncSession, plan: RelinkPlan, *, actor: str = RELINK_ACTOR
) -> RelinkResult:
    """Relink every planned orphan in place, recording one event.

    Each recovered name resolves through the artifact authority, scoped by its
    item's namespace key. An orphan that another writer linked since the plan
    was built is skipped, never reassigned. Does not commit.
    """
    result = RelinkResult()
    # particle id → [[name, gate class, tier], …]: compact, as a run can link thousands.
    links: dict[str, list[list[object]]] = {}
    for item in plan.items:
        subject_ids: list[str] = []
        names: list[str] = []
        for recovered in item.names:
            context = RecognizeContext(
                token_class=recovered.token_class, namespace_key=item.namespace_key
            )
            subject = await resolve_qualified_subject(
                session, recovered.name, context, asserted_by=actor
            )
            if subject is None:
                continue
            subject_ids.append(subject.id)
            names.append(recovered.name)
        if not subject_ids:
            result.skipped.append(item.particle_id)
            continue
        try:
            linked = await _attach_and_bind(session, item.particle_id, subject_ids, names)
        except ValueError:
            result.skipped.append(item.particle_id)
            continue
        result.relinked.append(item.particle_id)
        for sid in linked:
            if sid not in result.subjects:
                result.subjects.append(sid)
        links[item.particle_id] = [[r.name, r.token_class, r.tier] for r in item.names]
    if result.relinked:
        event = await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.SUBJECTS_RELINKED,
            refs=[
                *[(EventRefKind.PARTICLE, pid) for pid in result.relinked],
                *[(EventRefKind.SUBJECT, sid) for sid in result.subjects],
            ],
            payload={
                "batch": True,
                "tiers": plan.tiers,
                "relinked": len(result.relinked),
                "skipped": len(result.skipped),
                "links": links,
            },
        )
        result.event_id = event.event_id
    return result


async def relink_subjects(
    session: AsyncSession,
    particle_id: str,
    subject_ids: list[str],
    *,
    actor: str,
    names: list[str] | None = None,
) -> RelinkResult:
    """Attach operator-chosen Subjects to one unlinked belief in place.

    The ``assign-subject`` write. ``names`` are the operator's spellings, used
    only to bind the structured claim's subject term. Raises ``ValueError``
    when the particle is missing or already linked. Does not commit.
    """
    linked = await _attach_and_bind(session, particle_id, subject_ids, names or [])
    event = await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.SUBJECTS_RELINKED,
        refs=[
            (EventRefKind.PARTICLE, particle_id),
            *[(EventRefKind.SUBJECT, sid) for sid in linked],
        ],
        payload={"batch": False, "relinked": 1, "subject_ids": linked},
    )
    return RelinkResult(relinked=[particle_id], subjects=linked, event_id=event.event_id)


class RelinkReportName(BaseModel):
    """One recovered name in a report."""

    name: str
    token_class: str
    tier: int


class RelinkReportItem(BaseModel):
    """One sampled orphan and the names a relink would link it to."""

    particle_id: str
    content: str
    namespace_key: str
    names: list[RelinkReportName]


class RelinkReport(BaseModel):
    """A relink's plan, and what it wrote when applied.

    The dry run and ``--apply`` print this same shape from the same plan.
    ``sample`` is reproducible for a given ``seed``: it is the list an operator
    hand-checks before accepting a tier.
    """

    tiers: list[int]
    orphans: int
    recoverable: int
    by_tier: dict[str, int]
    by_class: dict[str, int]
    fail_closed_no_key: int
    fail_closed_several: int
    unrecovered: int
    sample: list[RelinkReportItem] = []
    applied: bool = False
    relinked: int = 0
    subjects: int = 0
    skipped: int = 0
    event_id: str | None = None


def build_report(
    plan: RelinkPlan,
    *,
    sample: int = 0,
    seed: int = 0,
    result: RelinkResult | None = None,
) -> RelinkReport:
    """Render a plan (and an applied result, when there is one) as a report."""
    return RelinkReport(
        tiers=plan.tiers,
        orphans=plan.orphans,
        recoverable=len(plan.items),
        by_tier={str(t): n for t, n in sorted(plan.by_tier.items())},
        by_class=dict(sorted(plan.by_class.items())),
        fail_closed_no_key=plan.fail_closed_no_key,
        fail_closed_several=plan.fail_closed_several,
        unrecovered=plan.unrecovered,
        sample=[
            RelinkReportItem(
                particle_id=item.particle_id,
                content=item.content,
                namespace_key=item.namespace_key,
                names=[
                    RelinkReportName(name=n.name, token_class=n.token_class, tier=n.tier)
                    for n in item.names
                ],
            )
            for item in (plan.sample(sample, seed) if sample > 0 else [])
        ],
        applied=result is not None,
        relinked=len(result.relinked) if result is not None else 0,
        subjects=len(result.subjects) if result is not None else 0,
        skipped=len(result.skipped) if result is not None else 0,
        event_id=result.event_id if result is not None else None,
    )
