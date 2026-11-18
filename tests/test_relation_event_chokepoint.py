# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Every relation a writer creates is disclosed where the store says (RA3).

A ``CO_EVIDENTIAL`` edge pools its pair at query time: the lower-ranked member
drops out of a result and the representative carries the merged confidence.
The 2026-11-17 release audit (RA3) found that a model PARAPHRASE verdict
applied by ``links suggest --apply`` wrote such an edge with no operator
event, so a verdict a source author can steer changed what queries return
without a record in the event log.

Two layers of test:

1. **Structural.** An AST walk over ``particles/`` finds every
   ``create_relation(`` call and reads its ``created_by`` argument. It must be
   a literal ``RelationCreatedBy.<member>``, and the ``(file, member)`` pair
   must appear in :data:`ALLOWED`, which states where that call's event is
   written. The allowlist must agree with
   :data:`particles.store.relation_store.RELATION_EVENT_SITE`, and a caller
   that writes its own merge event must call ``record_event`` with
   ``DUPLICATES_MERGED`` in the same function.

2. **Behavioural.** The audit's failing proof, ported: links APPLY on a
   scratch store leaves a ``RELATION_ADDED`` event naming both particles, the
   judge's rung and the run.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import particles
from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    Subject,
    SuggestMode,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.ingest.duplicate_judge import duplicate_judge_stamp
from particles.operations.links_suggest import suggest_co_evidential
from particles.store.event_store import EventRefKind, OperatorEventType, list_events
from particles.store.particle_store import insert_particle
from particles.store.relation_store import (
    RELATION_EVENT_SITE,
    RelationEventSite,
    create_relation,
    get_co_evidential_group,
)
from particles.store.subject_store import insert_subject
from tests import _source_tree

_PKG = Path(particles.__file__).parent

#: Every ``create_relation`` call site, by file and ``created_by`` member, with
#: where its event is written. A new call site, or a new kind at an old one,
#: fails until it is listed here with a site that matches the store's map.
ALLOWED: dict[tuple[str, RelationCreatedBy], RelationEventSite] = {
    # `particles links add` and the curation merge gestures.
    ("operations/agent_write.py", RelationCreatedBy.MANUAL_CLI): (
        RelationEventSite.CREATE_RELATION
    ),
    ("operations/curation/session.py", RelationCreatedBy.MANUAL_CLI): (
        RelationEventSite.CREATE_RELATION
    ),
    # An MCP agent's `link_add`.
    ("operations/agent_write.py", RelationCreatedBy.AGENT_ASSERTED): (
        RelationEventSite.CREATE_RELATION
    ),
    # `links suggest --apply`: a model PARAPHRASE verdict (RA3).
    ("operations/links_suggest.py", RelationCreatedBy.LLM_JUDGE): (
        RelationEventSite.CREATE_RELATION_VERDICT
    ),
    # `links dedup --apply` and its re-read variant:
    # one DUPLICATES_MERGED event per group, naming every member.
    ("operations/links_suggest.py", RelationCreatedBy.EXACT_DUPLICATE): (
        RelationEventSite.CALLER_MERGE_EVENT
    ),
    ("operations/reread_merge.py", RelationCreatedBy.REREAD_DUPLICATE): (
        RelationEventSite.CALLER_MERGE_EVENT
    ),
    # The extraction path keeps no operator events: stance and
    # narrative edges, and a declined pair's CONTRADICTS edge.
    ("ingest/pipeline.py", RelationCreatedBy.EXTRACTOR_DIRECT): (
        RelationEventSite.EXTRACTION_UNLOGGED
    ),
    ("store/relation_store.py", RelationCreatedBy.OBSERVER_DIVERGENCE): (
        RelationEventSite.EXTRACTION_UNLOGGED
    ),
}


def _is_create_relation(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "create_relation"
    return isinstance(func, ast.Attribute) and func.attr == "create_relation"


def _created_by(call: ast.Call) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "created_by":
            return kw.value
    # create_relation(session, particle_a, particle_b, relation_type, created_by, ...)
    return call.args[4] if len(call.args) > 4 else None


def _member(node: ast.expr | None) -> RelationCreatedBy | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "RelationCreatedBy"
        and node.attr in RelationCreatedBy.__members__
    ):
        return RelationCreatedBy[node.attr]
    return None


def _calls_merge_event(func: ast.AST) -> bool:
    """Whether ``func`` calls ``record_event`` with ``DUPLICATES_MERGED``."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name != "record_event":
            continue
        for kw in node.keywords:
            if (
                kw.arg == "event_type"
                and isinstance(kw.value, ast.Attribute)
                and kw.value.attr == OperatorEventType.DUPLICATES_MERGED.value
            ):
                return True
    return False


def _call_sites() -> list[tuple[str, int, ast.Call, ast.AST]]:
    """``(relative path, line, call, enclosing function)`` for every call."""
    out: list[tuple[str, int, ast.Call, ast.AST]] = []
    for path in sorted(_PKG.rglob("*.py")):
        rel = path.relative_to(_PKG).as_posix()
        tree = _source_tree.parse(path.read_text())
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Call) and _is_create_relation(node):
                    out.append((rel, node.lineno, node, func))
    # A call nested in an inner function is seen once per enclosing function;
    # keep the innermost, the last one walked.
    seen: dict[tuple[str, int], tuple[str, int, ast.Call, ast.AST]] = {}
    for site in out:
        seen[(site[0], site[1])] = site
    return list(seen.values())


def test_create_relation_is_only_ever_called_by_name() -> None:
    """An alias or a reference passed as a value would hide a call from the scan."""
    problems: list[str] = []
    for path in sorted(_PKG.rglob("*.py")):
        rel = path.relative_to(_PKG).as_posix()
        tree = _source_tree.parse(path.read_text())
        called = {
            id(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _is_create_relation(node)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "create_relation" and alias.asname not in (
                        None,
                        "create_relation",
                    ):
                        problems.append(f"{rel}:{node.lineno}: imported as {alias.asname}")
            elif isinstance(node, ast.Name | ast.Attribute) and id(node) not in called:
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name == "create_relation" and isinstance(node.ctx, ast.Load):
                    problems.append(f"{rel}:{node.lineno}: create_relation used as a value")
    assert not problems, "\n".join(problems)


def test_every_kind_has_an_event_site() -> None:
    assert set(RELATION_EVENT_SITE) == set(RelationCreatedBy)


def test_allowlist_agrees_with_the_store() -> None:
    wrong = {
        key: (site, RELATION_EVENT_SITE[key[1]])
        for key, site in ALLOWED.items()
        if site is not RELATION_EVENT_SITE[key[1]]
    }
    assert not wrong, f"ALLOWED disagrees with RELATION_EVENT_SITE: {wrong}"


def test_every_create_relation_call_is_allowlisted() -> None:
    sites = _call_sites()
    assert sites, "the scan found no create_relation call; it is broken"
    problems: list[str] = []
    used: set[tuple[str, RelationCreatedBy]] = set()
    for rel, line, call, func in sites:
        member = _member(_created_by(call))
        if member is None:
            problems.append(
                f"{rel}:{line}: created_by is not a literal RelationCreatedBy member, "
                "so where its event is written cannot be checked"
            )
            continue
        key = (rel, member)
        used.add(key)
        if key not in ALLOWED:
            problems.append(
                f"{rel}:{line}: create_relation(created_by={member.value}) is not in "
                "ALLOWED; list it with the site its event is written at"
            )
            continue
        if ALLOWED[key] is RelationEventSite.CALLER_MERGE_EVENT and not _calls_merge_event(func):
            problems.append(
                f"{rel}:{line}: {member.value} is disclosed by its caller's "
                "DUPLICATES_MERGED event, and this function writes none"
            )
    stale = set(ALLOWED) - used
    assert not problems, "\n".join(problems)
    assert not stale, f"ALLOWED lists call sites that no longer exist: {sorted(stale)}"


# ---------------------------------------------------------------------------
# Behavioural: the audit's proof (RA3), ported
# ---------------------------------------------------------------------------

_EMB_A = np.array([0.6, 0.8] + [0.0] * 382, dtype=np.float32)
_EMB_B = np.array([0.61, 0.79] + [0.0] * 382, dtype=np.float32)
_TRUSTED = "00000000-0000-0000-0000-0000000000a1"
_HOSTILE = "00000000-0000-0000-0000-0000000000b1"


def _p(content: str, pid: str, sid: str, entry: str) -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        subject_ids=[sid],
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry)],
        assertion_modality=AssertionModality.FALSIFIABLE,
    )


@pytest.mark.asyncio
async def test_applied_llm_judge_link_is_recorded_in_event_log(db_session: AsyncSession) -> None:
    subj = Subject(canonical_name="Acme Corp", asserted_by="test")
    await insert_subject(db_session, subj)
    await insert_particle(
        db_session,
        _p("Acme's support line is 555-0100.", _TRUSTED, subj.id, "ce-vendor"),
        embedding=_EMB_A.tolist(),
    )
    await insert_particle(
        db_session,
        _p("Acme's support line is 555-0199.", _HOSTILE, subj.id, "ce-forum"),
        embedding=_EMB_B.tolist(),
    )
    await db_session.commit()

    # The judge's PARAPHRASE stands in for a verdict the hostile claim steered.
    with patch("particles.operations._llm._llm_call", return_value='{"p0": "PARAPHRASE"}'):
        report = await suggest_co_evidential(
            db_session, subject_id=subj.id, mode=SuggestMode.APPLY, actor="test:apply"
        )

    assert report.applied_pairs == 1
    assert await get_co_evidential_group(db_session, _TRUSTED) == {_TRUSTED, _HOSTILE}

    events = await list_events(db_session, ref_kind=EventRefKind.PARTICLE, ref_id=_TRUSTED)
    assert len(events) == 1, "a model verdict merged two claims and no event discloses it"
    (event,) = events
    assert event.event_type is OperatorEventType.RELATION_ADDED
    assert event.actor == "test:apply"
    assert {r.ref_id for r in event.refs if r.ref_kind is EventRefKind.PARTICLE} == {
        _TRUSTED,
        _HOSTILE,
    }
    payload = event.payload or {}
    assert payload["created_by"] == RelationCreatedBy.LLM_JUDGE.value
    assert payload["relation_type"] == RelationType.CO_EVIDENTIAL.value
    assert payload["decided_by"] == str(duplicate_judge_stamp())
    assert isinstance(payload["run_id"], str) and payload["run_id"]


@pytest.mark.asyncio
async def test_verdict_relation_without_its_verdict_is_refused(db_session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="decided_by and run_id"):
        await create_relation(
            db_session, "p-a", "p-b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.LLM_JUDGE
        )
    assert await list_events(db_session) == []
