"""Graded, observer-relative claim-equivalence.

Covers the `effective_equivalence` lens (identity MVP + reserved hook) and the
threshold-gated co-evidential collapse that finally respects the edge confidence
the BFS previously ignored.
"""

from __future__ import annotations

import pytest

from particles.core.equivalence import effective_equivalence


def test_effective_equivalence_identity_and_reserved_hook() -> None:
    # MVP: identity lens.
    assert effective_equivalence(0.7) == 0.7
    # Reserved per-observer hook (provisional multiply); clamped.
    assert effective_equivalence(0.8, observer_trust=0.5) == pytest.approx(0.4)
    assert effective_equivalence(0.8, observer_trust=2.0) == 1.0


async def test_co_evidential_group_is_threshold_gated(db_session: object) -> None:
    import particles._orm_modules  # noqa: F401
    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        RelationCreatedBy,
        RelationType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation, get_co_evidential_group

    session = db_session  # type: ignore[assignment]

    def _p(content: str) -> Particle:
        return Particle(
            content=content,
            confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="t",
            provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1")],
        )

    a, b = _p("Water is H2O."), _p("Water = dihydrogen monoxide.")
    await insert_particle(session, a)  # type: ignore[arg-type]
    await insert_particle(session, b)  # type: ignore[arg-type]
    await create_relation(
        session,  # type: ignore[arg-type]
        a.id,
        b.id,
        RelationType.CO_EVIDENTIAL,
        RelationCreatedBy.AUTO_CLUSTER_V1,
        confidence=0.5,
    )
    await session.commit()  # type: ignore[attr-defined]

    # Default threshold (0.0) collapses on any link — pre-0106 behaviour.
    assert await get_co_evidential_group(session, a.id) == {a.id, b.id}  # type: ignore[arg-type]
    # A threshold above the edge confidence drops the pair to singletons.
    assert await get_co_evidential_group(session, a.id, min_confidence=0.6) == {a.id}  # type: ignore[arg-type]
    # A threshold below the edge confidence keeps the group.
    assert await get_co_evidential_group(session, a.id, min_confidence=0.4) == {a.id, b.id}  # type: ignore[arg-type]
