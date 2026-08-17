"""Tests for context fingerprinting."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.store.particle_store import (
    compute_context_fingerprint,
    get_particles_by_context_fingerprint,
    insert_particle,
    update_particle_status,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _mk_particle(content: str, particle_id: str | None = None) -> Particle:
    """Build a minimal ACTIVE particle for tests."""
    kwargs: dict[str, object] = {
        "content": content,
        "confidence": Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        "uncertainty_nature": UncertaintyNature.EPISTEMIC,
        "asserted_by": "test",
        "provenance": [
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce-test"),
        ],
    }
    if particle_id is not None:
        kwargs["id"] = particle_id
    return Particle(**kwargs)


@pytest.mark.asyncio
async def test_fingerprint_of_empty_store_is_sha256_of_empty(db_session: AsyncSession) -> None:
    """An empty store fingerprints to SHA-256("")."""
    fp = await compute_context_fingerprint(db_session)
    assert fp == hashlib.sha256(b"").hexdigest()


@pytest.mark.asyncio
async def test_fingerprint_is_deterministic(db_session: AsyncSession) -> None:
    """Same ACTIVE set → same fingerprint, regardless of insertion order."""
    p1 = _mk_particle("alpha", particle_id="11111111-1111-1111-1111-111111111111")
    p2 = _mk_particle("beta", particle_id="22222222-2222-2222-2222-222222222222")
    p3 = _mk_particle("gamma", particle_id="33333333-3333-3333-3333-333333333333")

    await insert_particle(db_session, p1)
    await insert_particle(db_session, p2)
    await insert_particle(db_session, p3)
    fp_forward = await compute_context_fingerprint(db_session)

    # Fingerprint must equal the SHA-256 of sorted UUIDs concatenated.
    expected = hashlib.sha256("".join(sorted([p1.id, p2.id, p3.id])).encode()).hexdigest()
    assert fp_forward == expected


@pytest.mark.asyncio
async def test_fingerprint_ignores_non_active_particles(db_session: AsyncSession) -> None:
    """RETRACTED / SUPERSEDED particles MUST NOT contribute to the fingerprint."""
    p1 = _mk_particle("active claim", particle_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    p2 = _mk_particle("about to retract", particle_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

    await insert_particle(db_session, p1)
    await insert_particle(db_session, p2)
    await db_session.commit()

    fp_with_both = await compute_context_fingerprint(db_session)

    await update_particle_status(
        db_session, p2.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await db_session.commit()

    fp_after_retract = await compute_context_fingerprint(db_session)

    assert fp_with_both != fp_after_retract
    # Only p1 remains ACTIVE
    assert fp_after_retract == hashlib.sha256(p1.id.encode()).hexdigest()


@pytest.mark.asyncio
async def test_fingerprint_changes_when_active_set_grows(db_session: AsyncSession) -> None:
    """Adding a new ACTIVE particle changes the fingerprint."""
    p1 = _mk_particle("first")
    await insert_particle(db_session, p1)
    fp_one = await compute_context_fingerprint(db_session)

    p2 = _mk_particle("second")
    await insert_particle(db_session, p2)
    fp_two = await compute_context_fingerprint(db_session)

    assert fp_one != fp_two


@pytest.mark.asyncio
async def test_get_particles_by_context_fingerprint(db_session: AsyncSession) -> None:
    """The reverse lookup returns every particle stamped with a fingerprint."""
    fp = "deadbeef" * 8  # 64 hex chars

    p1 = _mk_particle("first")
    p2 = _mk_particle("second")
    p3 = _mk_particle("unrelated")

    # Manually stamp the same fingerprint on two particles, different on a third.
    p1 = p1.model_copy(update={"context_fingerprint": fp})
    p2 = p2.model_copy(update={"context_fingerprint": fp})
    p3 = p3.model_copy(update={"context_fingerprint": "cafef00d" * 8})

    await insert_particle(db_session, p1)
    await insert_particle(db_session, p2)
    await insert_particle(db_session, p3)

    matches = await get_particles_by_context_fingerprint(db_session, fp)
    assert {m.id for m in matches} == {p1.id, p2.id}


@pytest.mark.asyncio
async def test_fingerprint_round_trips_via_orm(db_session: AsyncSession) -> None:
    """A stamped fingerprint survives ORM round-trip through ParticleRow."""
    from particles.store.particle_store import get_particle

    fp = "abc" + "0" * 61  # 64 hex chars
    p = _mk_particle("stamped").model_copy(update={"context_fingerprint": fp})
    await insert_particle(db_session, p)
    await db_session.commit()

    fetched = await get_particle(db_session, p.id)
    assert fetched is not None
    assert fetched.context_fingerprint == fp


@pytest.mark.asyncio
async def test_fingerprint_null_for_pre_adr_particle(db_session: AsyncSession) -> None:
    """A particle without an explicit fingerprint defaults to None and persists as NULL."""
    from particles.store.particle_store import get_particle

    p = _mk_particle("no fingerprint")
    assert p.context_fingerprint is None

    await insert_particle(db_session, p)
    await db_session.commit()

    fetched = await get_particle(db_session, p.id)
    assert fetched is not None
    assert fetched.context_fingerprint is None
