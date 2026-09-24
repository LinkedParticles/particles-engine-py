# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the retired-value quarantine rung.

The failure this closes: retraction is keyed on a *record*, and every
reconciliation path compared candidates against ACTIVE particles only, so
re-extracting or re-asserting a claim an operator had retracted minted a fresh
ACTIVE particle with no trace that the value had been judged wrong. The Atlas
calls the sequence "memory laundering"; its test list is the spec here:

* **re-extraction proof** — retract, re-extract, the value stays off ACTIVE;
* **the laundering sequence** — assert X, supersede with Y, restate X;
* **scope of the judgment** — only retirements that encode a verdict on the
  value fire; a retracted *source*, a reindex, an auto-merge do not;
* **an ACTIVE twin wins** — a claim the store currently believes absorbs the
  observation before any retired twin is consulted;
* **idempotence under repetition** — a second re-assertion lands on the open
  hold, not a second record;
* **the lift is a review** — PREFER_B mints the ACTIVE particle, PREFER_A
  keeps the retirement, and neither writes a trust statement or cascades;
* **the flag is a true off-switch**.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from particles.core.conflict_resolution import RETIRED_VALUE_KEY, build_inconsistency_particle
from particles.core.duplicate_key import content_hash
from particles.core.schema import (
    Confidence,
    Particle,
    PolicyProvenance,
    ProvenanceRef,
    ProvenanceRefType,
    ResolutionAction,
    SourceRef,
    SourceRefType,
    SourceTrustStatement,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.ingest.duplicate_suppression import (
    JUDGMENT_RETIREMENTS,
    build_retired_value_index,
)
from particles.store.particle_store import (
    get_particle,
    get_particles_by_content_hashes_and_state,
    get_particles_by_status,
    insert_particle,
    update_particle_status,
)

EMB = [0.1, 0.2, 0.3, 0.4]


def _particle(
    content: str,
    *,
    status: Status = Status.ACTIVE,
    reason: StatusReason | None = None,
    entry_id: str = "entry-1",
    snapshot_id: str = "snap-1",
    subject_ids: list[str] | None = None,
) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        asserted_at=datetime.now(UTC),
        status=status,
        status_reason=reason,
        subject_ids=subject_ids or [],
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
    )


async def _active_with(session: Any, content: str) -> list[Particle]:
    return [
        p for p in await get_particles_by_status(session, Status.ACTIVE) if p.content == content
    ]


async def _reconcile(session: Any, particle: Particle) -> Particle | None:
    from particles.ingest.pipeline import reconcile_and_insert

    return await reconcile_and_insert(session, particle, embedding=EMB)


# ---------------------------------------------------------------------------
# The index — which retirements count as a judgment
# ---------------------------------------------------------------------------


async def _insert_in_state(
    session: Any, claim: str, status: Status, reason: StatusReason | None
) -> Particle:
    """Insert ACTIVE, then transition — the way every retired row actually arises."""
    p = _particle(claim)
    await insert_particle(session, p, EMB)
    if status is not Status.ACTIVE:
        await update_particle_status(session, p.id, status, reason)
    await session.flush()
    return p


@pytest.mark.asyncio
async def test_probe_returns_only_rows_in_the_requested_states(db_session: Any) -> None:
    claim = "The OTLP port is 4318."
    await _insert_in_state(db_session, claim, Status.ACTIVE, None)
    retracted = await _insert_in_state(
        db_session, claim, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await _insert_in_state(db_session, claim, Status.RETRACTED, StatusReason.SOURCE_RETRACTED)
    await _insert_in_state(db_session, claim, Status.SUPERSEDED, StatusReason.SUPERSEDED_BY_REINDEX)

    found = await get_particles_by_content_hashes_and_state(
        db_session, [content_hash(claim)], JUDGMENT_RETIREMENTS
    )
    assert [p.id for p in found] == [retracted.id]


@pytest.mark.parametrize(
    ("status", "reason", "fires"),
    [
        (Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION, True),
        (Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION, True),
        (Status.PROVENANCE_STALE, StatusReason.CONFLICT_RESOLVED, True),
        (Status.PROVENANCE_STALE, StatusReason.CONFLICT_PENDING, True),  # an open hold
        (Status.ACTIVE, None, False),  # the index's business, not this one's
        (Status.RETRACTED, StatusReason.SOURCE_RETRACTED, False),
        (Status.SUPERSEDED, StatusReason.SUPERSEDED_BY_REINDEX, False),
        (Status.SUPERSEDED, StatusReason.DUPLICATE_MERGED, False),
        (Status.PROVENANCE_STALE, StatusReason.DOCUMENT_SUPERSEDED, False),
        (Status.PROVENANCE_STALE, StatusReason.VALIDITY_EXPIRED, False),
        (Status.PROVENANCE_STALE, StatusReason.LOWER_TRUST_SOURCE, False),
        (Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY, False),
    ],
)
@pytest.mark.asyncio
async def test_index_holds_judgment_retirements_and_open_holds_only(
    db_session: Any, status: Status, reason: StatusReason | None, fires: bool
) -> None:
    claim = "Coverage comes from `audit.py`."
    await _insert_in_state(db_session, claim, status, reason)
    index = await build_retired_value_index(db_session, [claim])
    assert (index.find(_particle(claim)) is not None) is fires


# ---------------------------------------------------------------------------
# The rung on the reconcile path (agent asserts, interchange import)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasserting_a_retracted_claim_is_held_not_reminted(db_session: Any) -> None:
    claim = "There are four commits on branch X."
    twin = _particle(claim)
    await insert_particle(db_session, twin, EMB)
    await update_particle_status(
        db_session, twin.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await db_session.commit()

    candidate = _particle(claim, entry_id="entry-2", snapshot_id="snap-2")
    returned = await _reconcile(db_session, candidate)
    await db_session.commit()

    assert returned is not None and returned.status is Status.INCONSISTENCY
    assert returned.properties == {RETIRED_VALUE_KEY: "EXPLICIT_RETRACTION"}
    refs = [r.corpus_entry_id for r in returned.provenance if r.type is ProvenanceRefType.PARTICLE]
    assert refs == [twin.id, candidate.id]
    assert "retired by judgment" in returned.content

    held = await get_particle(db_session, candidate.id)
    assert held is not None
    assert held.status is Status.PROVENANCE_STALE
    assert held.status_reason is StatusReason.CONFLICT_PENDING
    assert held.content == claim  # stored in full — nothing was dropped
    assert await _active_with(db_session, claim) == []
    # The twin is untouched: its retirement, and its stamp, stand.
    reloaded = await get_particle(db_session, twin.id)
    assert reloaded is not None and reloaded.status is Status.RETRACTED


@pytest.mark.asyncio
async def test_laundering_sequence_keeps_the_superseded_value_off_active(db_session: Any) -> None:
    """Verel's red-team finding: reject → supersede with another value →
    restate the original. The original must not reach ACTIVE."""
    x = _particle("The default port is 8000.")
    await insert_particle(db_session, x, EMB)
    y = _particle("The default port is 8080.")
    await insert_particle(db_session, y, [0.9, 0.1, 0.0, 0.0])
    await update_particle_status(
        db_session, x.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
    )
    await db_session.commit()

    returned = await _reconcile(db_session, _particle("The default port is 8000."))
    await db_session.commit()

    assert returned is not None and returned.status is Status.INCONSISTENCY
    assert returned.properties == {RETIRED_VALUE_KEY: "EXPLICIT_SUPERSESSION"}
    assert await _active_with(db_session, "The default port is 8000.") == []
    assert len(await _active_with(db_session, "The default port is 8080.")) == 1


@pytest.mark.asyncio
async def test_non_judgment_retirement_does_not_fire(db_session: Any) -> None:
    """A retracted *source* is not a verdict on the value: a second source
    making the same claim is new evidence and lands ACTIVE as before."""
    claim = "Model `claude-opus-4-6` is current."
    twin = _particle(claim)
    await insert_particle(db_session, twin, EMB)
    await update_particle_status(
        db_session, twin.id, Status.RETRACTED, StatusReason.SOURCE_RETRACTED
    )
    await db_session.commit()

    returned = await _reconcile(db_session, _particle(claim, entry_id="entry-2"))
    await db_session.commit()

    assert returned is not None and returned.status is Status.ACTIVE
    assert len(await _active_with(db_session, claim)) == 1


@pytest.mark.asyncio
async def test_active_twin_absorbs_before_any_retired_twin_is_consulted(db_session: Any) -> None:
    """runs first: a value the store currently believes is not in dispute."""
    claim = "The live store is named `particles.db`."
    live = _particle(claim)
    retired = _particle(claim, status=Status.RETRACTED, reason=StatusReason.EXPLICIT_RETRACTION)
    await insert_particle(db_session, live, EMB)
    await insert_particle(db_session, retired)
    await db_session.commit()

    returned = await _reconcile(db_session, _particle(claim, entry_id="entry-2"))
    await db_session.commit()

    assert returned is not None and returned.id == live.id
    assert len(await _active_with(db_session, claim)) == 1
    assert await get_particles_by_status(db_session, Status.INCONSISTENCY) == []


@pytest.mark.asyncio
async def test_second_reassertion_lands_on_the_open_hold(db_session: Any) -> None:
    """Idempotence: a repeated reindex of an unchanged source opens one hold, not one per run."""
    claim = "Coverage comes from `test_audit.py`."
    twin = _particle(claim)
    await insert_particle(db_session, twin, EMB)
    await update_particle_status(
        db_session, twin.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await db_session.commit()

    first = await _reconcile(db_session, _particle(claim, entry_id="entry-2", snapshot_id="s2"))
    await db_session.commit()
    second = await _reconcile(db_session, _particle(claim, entry_id="entry-3", snapshot_id="s3"))
    await db_session.commit()

    assert first is not None and first.status is Status.INCONSISTENCY
    assert second is not None and second.status is Status.PROVENANCE_STALE
    assert second.status_reason is StatusReason.CONFLICT_PENDING
    held = await get_particle(db_session, second.id)
    assert held is not None
    assert {r.corpus_entry_id for r in held.provenance} == {"entry-2", "entry-3"}
    assert len(await get_particles_by_status(db_session, Status.INCONSISTENCY)) == 1


@pytest.mark.asyncio
async def test_normalization_variants_are_the_same_value(db_session: Any) -> None:
    twin = _particle("The  OTLP  port is 4318.")
    await insert_particle(db_session, twin, EMB)
    await update_particle_status(
        db_session, twin.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await db_session.commit()

    returned = await _reconcile(db_session, _particle("The OTLP port is 4318", entry_id="entry-2"))
    assert returned is not None and returned.status is Status.INCONSISTENCY


@pytest.mark.asyncio
async def test_disabled_flag_restores_reminting(db_session: Any) -> None:
    from particles.config import get_config

    get_config().extraction.retired_value_quarantine.enabled = False
    claim = "A claim the operator retracted."
    twin = _particle(claim)
    await insert_particle(db_session, twin, EMB)
    await update_particle_status(
        db_session, twin.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await db_session.commit()

    returned = await _reconcile(db_session, _particle(claim, entry_id="entry-2"))
    assert returned is not None and returned.status is Status.ACTIVE


# ---------------------------------------------------------------------------
# The lift is a review; the cascade stays out of it
# ---------------------------------------------------------------------------


async def _held_pair(session: Any, claim: str) -> tuple[Particle, Particle, Particle]:
    """Insert a retracted twin, re-assert the claim, return (twin, held, wrapper)."""
    twin = _particle(claim)
    await insert_particle(session, twin, EMB)
    await update_particle_status(
        session, twin.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await session.commit()
    candidate = _particle(claim, entry_id="entry-2", snapshot_id="snap-2")
    wrapper = await _reconcile(session, candidate)
    await session.commit()
    assert wrapper is not None and wrapper.status is Status.INCONSISTENCY
    held = await get_particle(session, candidate.id)
    assert held is not None
    return twin, held, wrapper


@pytest.mark.asyncio
async def test_prefer_b_lifts_the_hold_and_writes_no_trust_statement(db_session: Any) -> None:
    from particles.operations.review import resolve
    from particles.store.trust_store import get_trust_statements_for_domain

    claim = "Rust is the better choice for the daemon."
    twin, held, wrapper = await _held_pair(db_session, claim)

    review = await resolve(db_session, wrapper.id, ResolutionAction.PREFER_B, "reviewer-1")

    assert review.trust_statement_id is None
    assert await get_trust_statements_for_domain(db_session, "general") == []
    lifted = await _active_with(db_session, claim)
    assert len(lifted) == 1 and lifted[0].supersedes == held.id
    reloaded_twin = await get_particle(db_session, twin.id)
    assert reloaded_twin is not None and reloaded_twin.status is Status.RETRACTED
    closed = await get_particle(db_session, wrapper.id)
    assert closed is not None and closed.status is Status.RETRACTED


@pytest.mark.asyncio
async def test_prefer_a_keeps_the_retirement(db_session: Any) -> None:
    from particles.operations.review import resolve
    from particles.store.trust_store import get_trust_statements_for_domain

    claim = "The daemon listens on port 9000."
    twin, held, wrapper = await _held_pair(db_session, claim)

    review = await resolve(db_session, wrapper.id, ResolutionAction.PREFER_A, "reviewer-1")

    assert review.trust_statement_id is None
    assert await get_trust_statements_for_domain(db_session, "general") == []
    assert await _active_with(db_session, claim) == []
    resolved = await get_particle(db_session, held.id)
    assert resolved is not None
    assert resolved.status is Status.PROVENANCE_STALE
    assert resolved.status_reason is StatusReason.CONFLICT_RESOLVED
    reloaded_twin = await get_particle(db_session, twin.id)
    assert reloaded_twin is not None and reloaded_twin.status is Status.RETRACTED


@pytest.mark.asyncio
async def test_trust_cascade_leaves_retired_value_records_to_a_person(db_session: Any) -> None:
    from particles.operations.cascade import run_trust_cascade
    from particles.store.trust_store import insert_trust_statement

    twin = _particle(
        "A held claim.", status=Status.RETRACTED, reason=StatusReason.EXPLICIT_RETRACTION
    )
    held = _particle(
        "A held claim.",
        status=Status.PROVENANCE_STALE,
        reason=StatusReason.CONFLICT_PENDING,
        entry_id="entry-2",
    )
    wrapper = build_inconsistency_particle(
        twin, held, corpus_entry_id="entry-2", snapshot_id="snap-2", retired_twin=True
    )
    for p in (twin, held):
        await insert_particle(db_session, p)
    await insert_particle(db_session, wrapper, domain_hint="general")
    stmt = SourceTrustStatement(
        domain="general",
        source_ref=SourceRef(type=SourceRefType.SOURCE_TYPE, value="WEB_PAGE"),
        trust_rank=0.95,
        policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
        asserted_by="operator",
    )
    await insert_trust_statement(db_session, stmt)
    await db_session.commit()

    assert await run_trust_cascade(db_session, stmt) == 0
    still_open = await get_particle(db_session, wrapper.id)
    assert still_open is not None and still_open.status is Status.INCONSISTENCY


# ---------------------------------------------------------------------------
# End-to-end through the extraction pipeline
# ---------------------------------------------------------------------------


def _mock_llm(contents: list[str]) -> MagicMock:
    import anthropic

    payload = MagicMock()
    payload.text = json.dumps(
        [
            {"content": c, "confidence_value": 0.9, "uncertainty_nature": "EPISTEMIC"}
            for c in contents
        ]
    )
    resp = MagicMock()
    resp.content = [payload]
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(return_value=resp)
    return client


async def _deposit(session: Any, tmp_path: Path, name: str, text: str) -> tuple[str, str]:
    from particles.corpus.deposit import deposit_file

    doc = tmp_path / name
    doc.write_text(text)
    entry_id, snapshot_id = await deposit_file(session, doc, deposited_by="test")
    await session.commit()
    return entry_id, snapshot_id


@pytest.mark.asyncio
async def test_reextraction_of_a_retracted_claim_is_held_and_disclosed(
    db_session: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The Atlas's named scenario: a source that still asserts a value the
    operator retracted is re-extracted. The value stays off ACTIVE, the
    candidate is stored quarantined behind a review record, and the pass says so."""
    from particles import embeddings as ep
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client

    claim = "The live store is named `particles.db`."
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([EMB], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    set_client(_mock_llm([claim]))
    try:
        e1, s1 = await _deposit(db_session, tmp_path, "a.md", "first source")
        first = await extract_snapshot(db_session, e1, s1)
        await db_session.commit()
        assert len(first) == 1
        await update_particle_status(
            db_session, first[0].id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()

        e2, s2 = await _deposit(db_session, tmp_path, "b.md", "second source, same claim")
        with caplog.at_level(logging.INFO, logger="particles.ingest.pipeline"):
            second = await extract_snapshot(db_session, e2, s2)
        await db_session.commit()

        assert len(second) == 1 and second[0].status is Status.INCONSISTENCY
        assert second[0].properties == {RETIRED_VALUE_KEY: "EXPLICIT_RETRACTION"}
        assert await _active_with(db_session, claim) == []
        held = [
            p
            for p in await get_particles_by_status(db_session, Status.PROVENANCE_STALE)
            if p.content == claim
        ]
        assert len(held) == 1 and held[0].status_reason is StatusReason.CONFLICT_PENDING
        assert held[0].provenance[0].corpus_entry_id == e2
        assert "RETIRED_VALUE_QUARANTINED: 1 candidate(s)" in caplog.text
    finally:
        set_client(None)
        ep.set_embedding_model(original)
