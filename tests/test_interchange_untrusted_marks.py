# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A bundle's server-owned marks are attribution, never authority (security review 2026-10-25-2).

An interchange bundle is bytes its author chose. Import and restore keep every
mark a reader keys on as authority where a reader can see it, and take it out
of what any reader keys on: a provenance ref naming a record of the target
store (R23), the re-anchor contributor, an ``mcp:`` asserter and an operator's
or an agent's calibration (R43), and, on restore, the reserved asserters and
the migration tag the import path already re-stamped (R26). Adopted from the
review's proof tests; in-process, against a scratch store with a stub encoder.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from particles.config import get_config
from particles.core.schema import (
    AssertionModality,
    Confidence,
    ContributorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.db import DEFAULT_STORE
from particles.interchange.codec import to_unit
from particles.mcp.memory_compat import ops


def _encode(texts: Any, **_kwargs: Any) -> Any:
    items = [texts] if isinstance(texts, str) else list(texts)
    return np.array(
        [[(b / 127.5) - 1.0 for b in hashlib.sha256(str(t).encode()).digest()[:16]] for t in items],
        dtype=np.float32,
    )


@pytest.fixture
def stub_encoder() -> Generator[None, None, None]:
    from particles import embeddings as ep

    model = MagicMock()
    model.encode = MagicMock(side_effect=_encode)
    original = ep._embedding_model
    ep.set_embedding_model(model)
    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    try:
        yield
    finally:
        ep.set_embedding_model(original)


async def _keyed_entry(session: Any, name: str, key: str) -> tuple[str, str]:
    from particles.core.schema import Mutability
    from particles.corpus.deposit import deposit_text_versioned

    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text=f"transcript of {name}",
        uri_r=f"claude-code://session/{name}",
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
        tags=["claude-code", f"project:{key}"],
        deposited_by="claude-code-hook",
    )
    return entry_id, snapshot_id


def _particle(content: str, entry_id: str, snapshot_id: str | None, **fields: Any) -> Particle:
    calibration = fields.pop("calibration", CalibrationSource.EXTRACTOR_DIRECT)
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=calibration),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
        asserted_by=fields.pop("asserted_by", "bundle-author"),
        **fields,
    )


def _unit(content: str, entry_id: str, snapshot_id: str | None, **fields: Any) -> dict[str, Any]:
    return to_unit(_particle(content, entry_id, snapshot_id, **fields), {})


async def _active(session: Any, content: str) -> Particle:
    from particles.store.particle_store import get_active_particles

    found = [p for p in await get_active_particles(session) if p.content == content]
    assert found, "the imported belief is not ACTIVE"
    return found[0]


# ---------------------------------------------------------------------------
# R23: provenance refs naming the target store's records
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_imported_unit_does_not_take_the_scope_of_an_entry_its_refs_name(
    db_session: Any, stub_encoder: None
) -> None:
    """A claim imported from a bundle is not placed in a project's view by a ref
    naming one of that project's corpus entries (R23)."""
    from particles.core.observer_scope import visible
    from particles.interchange.store import ORIGIN_REF_PREFIX, import_units
    from particles.store.observer_scope_join import load_scopes
    from tests._observer_scope import rescoped

    a_entry, a_snap = await _keyed_entry(db_session, "aaaa", "proj-a")
    await rescoped(db_session)
    await db_session.commit()
    content = "Project A's release is frozen until further notice."
    summary = await import_units(db_session, [_unit(content, a_entry, a_snap)])
    await db_session.commit()
    assert summary.reanchored == 1
    imported = await _active(db_session, content)
    scope = (await load_scopes(db_session, [imported]))[imported.id]
    assert not visible(scope, "proj-a"), scope
    # The origin id stays visible as metadata and resolves to nothing here.
    assert imported.provenance[0].corpus_entry_id == f"{ORIGIN_REF_PREFIX}{a_entry}"
    assert imported.provenance[0].snapshot_id == f"{ORIGIN_REF_PREFIX}{a_snap}"


@pytest.mark.asyncio
async def test_an_imported_duplicate_does_not_widen_an_existing_belief_into_another_project(
    db_session: Any, stub_encoder: None
) -> None:
    """Importing a unit identical to a project-A belief, whose ref names a project-B
    entry, leaves the belief out of project B's view and its provenance as it was (R23)."""
    from particles.core.observer_scope import visible
    from particles.interchange.store import import_units
    from particles.store.observer_scope_join import load_scopes
    from particles.store.particle_store import get_particle
    from tests._observer_scope import belief, rescoped

    a_entry, a_snap = await _keyed_entry(db_session, "aaaa", "proj-a")
    b_entry, b_snap = await _keyed_entry(db_session, "bbbb", "proj-b")
    content = "Project A's deploy key lives in vault path a/deploy."
    existing = await belief(db_session, content, (a_entry, a_snap))
    await rescoped(db_session)
    await db_session.commit()
    await import_units(db_session, [_unit(content, b_entry, b_snap)])
    await db_session.commit()
    reread = await get_particle(db_session, existing.id)
    assert reread is not None
    assert reread.provenance == existing.provenance
    after = (await load_scopes(db_session, [reread]))[existing.id]
    assert not visible(after, "proj-b"), after


@pytest.mark.asyncio
async def test_an_imported_derived_unit_does_not_take_a_target_belief_as_its_premise(
    db_session: Any, stub_encoder: None
) -> None:
    """A PARTICLE ref naming a belief of the target store is prefixed too, so a bundle
    cannot inherit a project's scope as that belief's derivation (R23)."""
    from particles.interchange.store import ORIGIN_REF_PREFIX, import_units
    from particles.store.particle_store import insert_particle

    a_entry, a_snap = await _keyed_entry(db_session, "aaaa", "proj-a")
    premise = _particle("Project A ships on Tuesdays.", a_entry, a_snap)
    await insert_particle(db_session, premise, _encode(premise.content)[0].tolist())
    await db_session.commit()
    derived = _particle("A derived claim.", premise.id, None)
    derived = derived.model_copy(
        update={
            "provenance": [
                ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=premise.id)
            ]
        }
    )
    await import_units(db_session, [to_unit(derived, {})])
    await db_session.commit()
    imported = await _active(db_session, "A derived claim.")
    assert imported.provenance[0].corpus_entry_id == f"{ORIGIN_REF_PREFIX}{premise.id}"


@pytest.mark.asyncio
async def test_a_ref_into_the_source_store_is_kept_as_written(
    db_session: Any, stub_encoder: None
) -> None:
    """A ref naming an entry this store does not hold (the ordinary cross-store case)
    imports unchanged: it is origin metadata already."""
    from particles.interchange.store import import_units

    summary = await import_units(db_session, [_unit("Elsewhere.", "entry-x", "snap-x")])
    await db_session.commit()
    assert summary.reanchored == 0
    imported = await _active(db_session, "Elsewhere.")
    assert imported.provenance[0].corpus_entry_id == "entry-x"


# ---------------------------------------------------------------------------
# R43: the other server-owned marks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_imported_unit_is_not_read_as_a_reanchor_restatement(
    db_session: Any, stub_encoder: None
) -> None:
    """A unit listing the re-anchor pass as a contributor does not exempt its claim from
    the update rung (R43); the claimed contributor stays visible, prefixed."""
    from datetime import UTC, datetime

    from particles.core.reanchor import REANCHOR_ACTOR, is_reanchor_restatement
    from particles.interchange.store import CLAIMED_TAG_PREFIX, import_units

    reanchor = ContributorRef(id=REANCHOR_ACTOR, role="author", at=datetime(2026, 1, 1, tzinfo=UTC))
    unit = _unit("The office moved to the fourth floor.", "e", "s", contributors=[reanchor])
    summary = await import_units(db_session, [unit])
    await db_session.commit()
    imported = await _active(db_session, "The office moved to the fourth floor.")
    assert not is_reanchor_restatement(imported)
    assert summary.restamped == 1
    # The claimed asserter is kept as an author contributor beside it (W38).
    assert [c.id for c in imported.contributors or []] == [
        f"{CLAIMED_TAG_PREFIX}{REANCHOR_ACTOR}",
        "bundle-author",
    ]


@pytest.mark.asyncio
async def test_an_imported_mcp_prefixed_asserter_is_restamped(
    db_session: Any, stub_encoder: None
) -> None:
    """Any ``mcp:`` asserter, not only the two configured identities, is imported under
    the import identity: the update rung and agent verbs read the prefix (R43)."""
    from particles.ingest.update_supersession import is_extractor_asserted
    from particles.interchange.store import IMPORT_ASSERTER, import_units

    unit = _unit("The lab closes at six.", "e", "s", asserted_by="mcp:someone-else")
    summary = await import_units(db_session, [unit])
    await db_session.commit()
    imported = await _active(db_session, "The lab closes at six.")
    assert imported.asserted_by == IMPORT_ASSERTER
    assert summary.restamped == 1
    assert "mcp:someone-else" in [c.id for c in imported.contributors or []]
    assert is_extractor_asserted(imported)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claimed", [CalibrationSource.HUMAN_REVIEW, CalibrationSource.AGENT_ASSERTED]
)
async def test_an_imported_operator_or_agent_calibration_is_imported(
    db_session: Any, stub_encoder: None, claimed: CalibrationSource
) -> None:
    """A unit claiming an operator's or an agent's calibration is imported as IMPORTED,
    so it is neither exempt from retirement nor guarded from agent mutation (R43)."""
    from particles.interchange.store import import_units

    content = f"The canteen serves soup ({claimed.value})."
    summary = await import_units(db_session, [_unit(content, "e", "s", calibration=claimed)])
    await db_session.commit()
    imported = await _active(db_session, content)
    assert imported.confidence.calibration_source is CalibrationSource.IMPORTED
    assert imported.confidence.value == 0.9
    assert summary.recalibrated == 1


# ---------------------------------------------------------------------------
# G29: a bundle's modality and routing properties are claims, not exemptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_imported_unit_takes_no_arbitration_exemption_from_its_modality(
    db_session: Any, stub_encoder: None
) -> None:
    """A bundle cannot exempt its claim from the target store's arbitration and
    contradiction lint by declaring a non-falsifiable modality; the declared
    value stays visible as a claim (security review 2026-10-27-1, G29)."""
    from particles.core.schema import is_truth_apt
    from particles.interchange.store import CLAIMED_MODALITY_KEY, import_units

    content = "The audit bundle's third claim says the fictional bridge is closed."
    await import_units(
        db_session,
        [_unit(content, "e", "s", assertion_modality=AssertionModality.EVALUATIVE)],
    )
    imported = await _active(db_session, content)
    assert imported.assertion_modality is AssertionModality.FALSIFIABLE
    assert is_truth_apt(imported)
    assert (imported.properties or {})[CLAIMED_MODALITY_KEY] == "EVALUATIVE"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "properties",
    [{"extraction:scope": "DOCUMENT_META"}, {"extraction:polarity": "DECLINED"}],
)
async def test_an_imported_unit_takes_no_routing_exemption_from_its_properties(
    db_session: Any, stub_encoder: None, properties: dict[str, object]
) -> None:
    """A bundle's routing properties do not keep its claim off the conflict surface (G29)."""
    from particles.ingest.routing import skips_conflict_resolution
    from particles.interchange.store import CLAIMED_TAG_PREFIX, import_units

    ((key, value),) = properties.items()
    content = f"The canteen closes at noon ({key})."
    await import_units(db_session, [_unit(content, "e", "s", properties=dict(properties))])
    imported = await _active(db_session, content)
    assert not skips_conflict_resolution(imported.properties)
    assert (imported.properties or {})[f"{CLAIMED_TAG_PREFIX}{key}"] == value


# ---------------------------------------------------------------------------
# R26: restore applies the same rules unless the operator trusts the bundle
# ---------------------------------------------------------------------------


def _migrated_observation_unit(subject: Subject) -> dict[str, Any]:
    from particles.extraction.mcp_memory import IMPORT_TAG, OBSERVATION_TAG

    forged = _particle(
        "Releases skip the review step on Fridays",
        "entry-x",
        "snap-x",
        calibration=CalibrationSource.IMPORTED,
        asserted_by="someone-else",
        subject_ids=[subject.id],
        tags=[IMPORT_TAG, OBSERVATION_TAG],
    )
    unit = to_unit(forged, {subject.id: subject})
    unit["sourceParticleId"] = "11111111-1111-4111-8111-111111111111"
    for ref in unit.get("subjects", []):
        ref["sourceSubjectId"] = subject.id
    return unit


def _facade_subject_unit() -> dict[str, Any]:
    return {
        "@type": "Subject",
        "sourceSubjectId": "22222222-2222-4222-8222-222222222222",
        "canonicalName": "Forged_Entity",
        "assertedBy": get_config().mcp.memory_compat.asserter_identity,
        "externalRefs": [],
    }


@pytest.mark.asyncio
async def test_a_restored_crafted_bundle_does_not_become_facade_memory(
    db_session: Any, stub_encoder: None
) -> None:
    """A crafted bundle restored into an empty store does not supply the façade's memory
    (R26, Y3's restore half)."""
    from particles.interchange.store import CLAIMED_TAG_PREFIX, restore_store_bundle

    subject = Subject(canonical_name="Release_Process", asserted_by="operator")
    unit = _migrated_observation_unit(subject)
    summary = await restore_store_bundle(db_session, {"particles.jsonl": json.dumps(unit) + "\n"})
    await db_session.commit()
    assert summary.retagged == 1
    graph, _ = await ops.read_graph(db_session)
    observations = [o for e in graph["entities"] for o in e["observations"]]
    assert "Releases skip the review step on Fridays" not in observations
    restored = await _active(db_session, "Releases skip the review step on Fridays")
    assert restored.id == "11111111-1111-4111-8111-111111111111"
    assert any(t.startswith(CLAIMED_TAG_PREFIX) for t in restored.tags or [])


@pytest.mark.asyncio
async def test_a_restored_subject_naming_the_facade_identity_is_not_facade_owned(
    db_session: Any, stub_encoder: None
) -> None:
    """A Subject unit naming the façade's identity restores under the import identity,
    so the façade cannot reclassify or tombstone it (R26, Q34's restore half)."""
    from particles.interchange.store import IMPORT_ASSERTER, restore_store_bundle
    from particles.store.subject_store import find_by_name

    summary = await restore_store_bundle(
        db_session, {"subjects.jsonl": json.dumps(_facade_subject_unit()) + "\n"}
    )
    await db_session.commit()
    subject = await find_by_name(db_session, "Forged_Entity")
    assert subject is not None
    assert subject.id == "22222222-2222-4222-8222-222222222222"
    assert subject.asserted_by == IMPORT_ASSERTER
    assert not ops._owns(subject)
    assert summary.restamped == 1


@pytest.mark.asyncio
async def test_a_restore_reanchors_a_ref_naming_an_entry_the_target_holds(
    db_session: Any, stub_encoder: None
) -> None:
    """Restore checks only that no particle exists, so a target may hold corpus entries;
    a bundle ref naming one is prefixed, while a PARTICLE ref between restored particles
    keeps its preserved origin id (R23, R26)."""
    from particles.interchange.store import ORIGIN_REF_PREFIX, restore_store_bundle

    a_entry, a_snap = await _keyed_entry(db_session, "aaaa", "proj-a")
    await db_session.commit()
    first = _particle("First restored.", a_entry, a_snap)
    second = _particle("Second restored.", "e", None).model_copy(
        update={
            "provenance": [ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=first.id)]
        }
    )
    units = []
    for particle in (first, second):
        unit = to_unit(particle, {})
        unit["sourceParticleId"] = particle.id
        units.append(json.dumps(unit))
    summary = await restore_store_bundle(db_session, {"particles.jsonl": "\n".join(units) + "\n"})
    await db_session.commit()
    assert summary.reanchored == 1
    restored_first = await _active(db_session, "First restored.")
    restored_second = await _active(db_session, "Second restored.")
    assert restored_first.provenance[0].corpus_entry_id == f"{ORIGIN_REF_PREFIX}{a_entry}"
    assert restored_second.provenance[0].corpus_entry_id == first.id


@pytest.mark.asyncio
async def test_trust_bundle_restores_the_marks_verbatim(
    db_session: Any, stub_encoder: None
) -> None:
    """``trust_bundle`` keeps a faithful restore of the operator's own export: the marks
    come back as written and nothing is counted."""
    from particles.extraction.mcp_memory import IMPORT_TAG
    from particles.interchange.store import restore_store_bundle
    from particles.store.subject_store import find_by_name

    subject = Subject(canonical_name="Release_Process", asserted_by="operator")
    files = {
        "particles.jsonl": json.dumps(_migrated_observation_unit(subject)) + "\n",
        "subjects.jsonl": json.dumps(_facade_subject_unit()) + "\n",
    }
    summary = await restore_store_bundle(db_session, files, trust_bundle=True)
    await db_session.commit()
    assert (summary.restamped, summary.retagged, summary.reanchored) == (0, 0, 0)
    restored = await _active(db_session, "Releases skip the review step on Fridays")
    assert IMPORT_TAG in (restored.tags or [])
    forged = await find_by_name(db_session, "Forged_Entity")
    assert forged is not None and ops._owns(forged)


def test_cli_restore_reports_the_marks_and_honours_trust_bundle(cli_db: Any, tmp_path: Any) -> None:
    """`interchange restore` says what it re-stamped and how to keep it; `--trust-bundle`
    restores the bundle as written."""
    from typer.testing import CliRunner

    from particles.api.cli import app

    runner = CliRunner()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "subjects.jsonl").write_text(json.dumps(_facade_subject_unit()) + "\n")
    res = runner.invoke(app, ["interchange", "restore", str(bundle)])
    assert res.exit_code == 0, res.output
    assert "1 record(s) named a Particles server identity" in res.output
    assert "--trust-bundle" in res.output

    # A Subject-only restore leaves the store with no particle, so it can run again.
    own = tmp_path / "own"
    own.mkdir()
    unit = {
        **_facade_subject_unit(),
        "sourceSubjectId": "33333333-3333-4333-8333-333333333333",
        "canonicalName": "Own_Entity",
    }
    (own / "subjects.jsonl").write_text(json.dumps(unit) + "\n")
    res = runner.invoke(app, ["interchange", "restore", str(own), "--trust-bundle"])
    assert res.exit_code == 0, res.output
    assert "server identity" not in res.output


# ---------------------------------------------------------------------------
# Security review 2026-10-26-1: a predecessor naming the target store (W39), and
# an asserter naming an agent key's principal (W38)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_imported_unit_never_names_a_target_particle_as_its_predecessor(
    db_session: Any, stub_encoder: None
) -> None:
    """An imported unit's ``supersedes`` never names a particle of the target
    store, so a bundle cannot date a target belief's retirement in the as-of
    lens or draw itself as that belief's successor (W39)."""
    from datetime import UTC, datetime

    from particles.core.status import Status
    from particles.interchange.store import ORIGIN_REF_PREFIX, import_units
    from particles.operations.query.as_of import load_retirement_index
    from particles.store.particle_store import insert_particle

    target = _particle(
        "The release train leaves on Tuesdays.", "bundle-entry", None, status=Status.SUPERSEDED
    )
    await insert_particle(db_session, target)
    await db_session.flush()

    content = "The release train leaves on Thursdays."
    unit = _unit(
        content,
        "bundle-entry",
        None,
        supersedes=target.id,
        asserted_at=datetime(2001, 1, 1, tzinfo=UTC),
    )
    summary = await import_units(db_session, [unit])
    assert summary.imported == 1
    await db_session.flush()

    index = await load_retirement_index(db_session)
    assert index.successor_by_predecessor.get(target.id) is None
    imported = await _active(db_session, content)
    assert imported.supersedes == f"{ORIGIN_REF_PREFIX}{target.id}"


@pytest.mark.asyncio
async def test_an_imported_unit_never_keeps_an_agent_key_principal_as_asserter(
    db_session: Any, stub_encoder: None
) -> None:
    """An imported unit naming any asserter, such as a live agent key's
    principal, is re-stamped, so that agent's own-beliefs guard never reaches a
    belief a bundle author wrote; the claimed asserter stays as an author
    contributor (W38)."""
    from particles.interchange.store import IMPORT_ASSERTER, import_units

    content = "The nightly job retries three times before paging."
    unit = _unit(content, "bundle-entry", None, asserted_by="agent:ci-bot")
    summary = await import_units(db_session, [unit])
    await db_session.flush()
    imported = await _active(db_session, content)
    assert imported.asserted_by == IMPORT_ASSERTER
    assert [(c.id, c.role) for c in imported.contributors or []] == [("agent:ci-bot", "author")]
    # Not a server identity, so the import notice does not count it as one.
    assert summary.restamped == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "per_store", "expected"),
    [([DEFAULT_STORE], {}, False), ([], {DEFAULT_STORE: "multi"}, False), ([], {}, True)],
    ids=["write-enabled", "per-store-multi", "global-single"],
)
async def test_import_reconciles_under_the_target_stores_own_trust_regime(
    db_session: Any,
    stub_encoder: None,
    enabled: list[str],
    per_store: dict[str, str],
    expected: bool,
) -> None:
    """Import resolves the §6.6 regime for its target store, not from the global
    ``store_mode`` alone (the interchange half of P30)."""
    from unittest.mock import AsyncMock, patch

    from particles.interchange.store import import_units

    cfg = get_config()
    cfg.reconciliation.store_mode = "single"
    cfg.reconciliation.per_store = per_store
    cfg.mcp.write.enabled_stores = enabled
    seen = AsyncMock(return_value=None)
    with patch("particles.ingest.pipeline.apply_reconcile", seen):
        await import_units(db_session, [_unit("The canteen opens at noon.", "e", None)])
    assert seen.await_args is not None
    assert seen.await_args.kwargs["single_trust_order"] is expected


# ---------------------------------------------------------------------------
# AF3 / AF4: a status the import kept is the bundle's, not a judgment here
# ---------------------------------------------------------------------------


def _kept_unit(content: str, status: str, reason: str) -> dict[str, Any]:
    unit = _unit(content, "bundle-entry", "bundle-snapshot")
    unit["status"], unit["statusReason"] = status, reason
    return unit


async def _statuses(session: Any, content: str) -> list[tuple[Any, Any]]:
    from particles.core.status import Status
    from particles.store.particle_store import get_particles_by_status

    return [
        (p.status, p.status_reason)
        for status in Status
        for p in await get_particles_by_status(session, status)
        if p.content == content
    ]


async def _open_records(session: Any) -> list[Particle]:
    from particles.store.particle_store import get_inconsistency_particles

    return list(await get_inconsistency_particles(session))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("RETRACTED", "EXPLICIT_RETRACTION"),
        ("SUPERSEDED", "EXPLICIT_SUPERSESSION"),
        ("PROVENANCE_STALE", "CONFLICT_PENDING"),
    ],
)
async def test_a_kept_non_active_unit_takes_no_part_in_reconciling_a_later_unit(
    db_session: Any, stub_encoder: None, status: str, reason: str
) -> None:
    """A unit the import keeps non-ACTIVE neither holds nor absorbs the bundle's later
    unit of the same claim: that unit lands ACTIVE, and no record names the kept one
    (security review 2026-10-31-3, AF3, AF4)."""
    from particles.core.status import Status
    from particles.interchange.store import import_units

    content = "The kept widget weighs two kilograms."
    await import_units(
        db_session, [_kept_unit(content, status, reason), _unit(content, "e2", "s2")]
    )
    await db_session.commit()
    assert (Status.ACTIVE, None) in await _statuses(db_session, content)
    assert await _open_records(db_session) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "reason"),
    [("RETRACTED", "EXPLICIT_RETRACTION"), ("PROVENANCE_STALE", "CONFLICT_PENDING")],
)
async def test_a_status_a_bundle_chose_does_not_hold_a_later_writers_claim(
    db_session: Any, stub_encoder: None, status: str, reason: str
) -> None:
    """A unit a bundle delivers retired by judgment, or as a pending hold, is not read
    as a judgment made in this store: a later writer's assertion of the same claim
    lands ACTIVE, not held behind it (security review 2026-10-31-3, AF3, AF4)."""
    from particles.core.status import Status
    from particles.db import write_transaction
    from particles.ingest.pipeline import reconcile_and_insert
    from particles.interchange.store import import_units

    content = "The planted widget weighs four kilograms."
    await import_units(db_session, [_kept_unit(content, status, reason)])
    await db_session.commit()
    async with write_transaction(db_session):
        await reconcile_and_insert(
            db_session, _particle(content, "agent-entry", "agent-snapshot", asserted_by="mcp:agent")
        )
    assert (Status.ACTIVE, None) in await _statuses(db_session, content)
    assert await _open_records(db_session) == []


@pytest.mark.asyncio
async def test_an_imported_claim_this_store_retracts_still_holds_its_reassertion(
    db_session: Any, stub_encoder: None
) -> None:
    """Control: an imported claim the operator retracts here is a judgment made here, so
    a later re-assertion is held behind it as any retired value is."""
    from particles.core.status import Status, StatusReason
    from particles.db import write_transaction
    from particles.ingest.pipeline import reconcile_and_insert
    from particles.interchange.store import import_units
    from particles.store.particle_store import update_particle_status

    content = "The imported widget weighs five kilograms."
    await import_units(db_session, [_unit(content, "bundle-entry", "bundle-snapshot")])
    await db_session.commit()
    imported = await _active(db_session, content)
    await update_particle_status(
        db_session, imported.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await db_session.commit()
    async with write_transaction(db_session):
        await reconcile_and_insert(
            db_session, _particle(content, "agent-entry", "agent-snapshot", asserted_by="mcp:agent")
        )
    assert (Status.ACTIVE, None) not in await _statuses(db_session, content)
    assert len(await _open_records(db_session)) == 1


def test_the_retired_value_index_names_the_asserter_the_import_stamps() -> None:
    """The index's copy of the import asserter is the import's own."""
    from particles.ingest.duplicate_suppression import IMPORTED_BY
    from particles.interchange.store import IMPORT_ASSERTER

    assert IMPORTED_BY == IMPORT_ASSERTER
