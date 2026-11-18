# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The exact identifier-token rank addend in the query op."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ParticleType,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status


def _particle(content: str, confidence: float = 0.8) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
    )


class TestRankScore:
    def test_lift_is_added_after_the_weights(self) -> None:
        """``μ · X`` is not scaled by the NARRATIVE weight (§1's slot)."""
        from particles.operations.query.main import _combined, _rank_score

        narrative = _particle("RFC 9110 narrative").model_copy(
            update={"particle_type": ParticleType.NARRATIVE}
        )
        weight = get_config().query.narrative_rank_weight
        lift = {narrative.id: 0.1}
        assert _rank_score((narrative, 0.5, 0.5), exact_lift=lift) == pytest.approx(
            _combined(0.5, 0.5) * weight + 0.1
        )

    def test_no_lift_is_the_dense_score(self) -> None:
        from particles.operations.query.main import _rank_score

        p = _particle("a claim")
        assert _rank_score((p, 0.7, 0.6), exact_lift={}) == _rank_score((p, 0.7, 0.6))
        assert _rank_score((p, 0.7, 0.6), exact_lift={"other": 1.0}) == _rank_score((p, 0.7, 0.6))


class TestExactTokenLift:
    def test_inert_at_zero_weight(self) -> None:
        from particles.operations.query.main import _exact_token_lift

        p = _particle("RFC 9110 is implementable.")
        assert get_config().query.exact_token_weight == 0.0
        assert _exact_token_lift("Is RFC 9110 implementable?", [(p, 0.5, 0.5)]) == {}

    def test_inert_without_an_identifier_token(self) -> None:
        from particles.operations.query.main import _exact_token_lift

        get_config().query.exact_token_weight = 0.1
        p = _particle("The top 40 hits.")
        assert _exact_token_lift("what are the top 40 hits?", [(p, 0.5, 0.5)]) == {}
        assert _exact_token_lift(None, [(p, 0.5, 0.5)]) == {}

    def test_lift_is_weight_times_share(self) -> None:
        from particles.operations.query.main import _exact_token_lift

        get_config().query.exact_token_weight = 0.2
        both = _particle("RFC 9110 was decided in 2026.")
        year = _particle("Something in 2026.")
        none = _particle("RFC 9112 is another record.")
        lift = _exact_token_lift(
            "what did we decide about RFC 9110 in 2026?",
            [(both, 0.5, 0.5), (year, 0.5, 0.5), (none, 0.5, 0.5)],
        )
        assert lift == {both.id: pytest.approx(0.2), year.id: pytest.approx(0.1)}

    def test_min_length_is_read_from_config(self) -> None:
        from particles.operations.query.main import _exact_token_lift

        get_config().query.exact_token_weight = 0.2
        get_config().query.exact_token_min_length = 2
        p = _particle("PR 12 merged.")
        assert _exact_token_lift("did PR 12 merge?", [(p, 0.5, 0.5)]) == {p.id: 0.2}


def _mock_model(vector: list[float]) -> MagicMock:
    model = MagicMock()
    model.encode = MagicMock(return_value=[np.array(vector, dtype=np.float32)])
    return model


async def _seed(session: object) -> tuple[Particle, Particle, Particle]:
    """Three beliefs: two close to the question's vector, the record's one far."""
    from particles.store.particle_store import insert_particle

    near_a = _particle("RFC 9112 settled the split package.", 0.9)
    near_b = _particle("The split package ships in two wheels.", 0.85)
    record = _particle("RFC 9110 is implementable once its review lands.", 0.6)
    await insert_particle(session, near_a, [1.0, 0.0, 0.0, 0.0])  # type: ignore[arg-type]
    await insert_particle(session, near_b, [0.95, 0.31, 0.0, 0.0])  # type: ignore[arg-type]
    await insert_particle(session, record, [0.6, 0.8, 0.0, 0.0])  # type: ignore[arg-type]
    await session.commit()  # type: ignore[attr-defined]
    return near_a, near_b, record


@pytest.mark.asyncio
async def test_no_identifier_token_leaves_ranking_byte_identical(db_session: object) -> None:
    """With ``X = 0`` for every belief the ranking is the dense one, value for value."""
    from particles import embeddings as ep
    from particles.operations.query import retrieve_ranked

    await _seed(db_session)
    original = ep._embedding_model
    ep.set_embedding_model(_mock_model([1.0, 0.0, 0.0, 0.0]))
    try:
        request = QueryRequest(question="where does the split package ship, top 40?", top_k=3)
        dense = await retrieve_ranked(db_session, request)  # type: ignore[arg-type]
        get_config().query.exact_token_weight = 1.0
        lifted = await retrieve_ranked(db_session, request)  # type: ignore[arg-type]
    finally:
        ep.set_embedding_model(original)
    assert [(p.id, sim, eff) for p, sim, eff in lifted] == [
        (p.id, sim, eff) for p, sim, eff in dense
    ]


@pytest.mark.asyncio
async def test_identifier_token_lifts_the_naming_belief(db_session: object) -> None:
    """The belief naming the record rises into the cut, its confidence unchanged."""
    from particles import embeddings as ep
    from particles.operations.query import retrieve_ranked

    _, _, record = await _seed(db_session)
    original = ep._embedding_model
    ep.set_embedding_model(_mock_model([1.0, 0.0, 0.0, 0.0]))
    try:
        request = QueryRequest(question="Is RFC 9110 implementable?", top_k=2)
        dense = await retrieve_ranked(db_session, request)  # type: ignore[arg-type]
        get_config().query.exact_token_weight = 0.5
        lifted = await retrieve_ranked(db_session, request)  # type: ignore[arg-type]
    finally:
        ep.set_embedding_model(original)
    assert record.id not in {p.id for p, _, _ in dense}
    assert lifted[0][0].id == record.id
    # Read-time ordering only: the reported effective confidence is untouched.
    dense_all = {p.id: eff for p, _, eff in dense}
    assert all(dense_all.get(p.id, eff) == eff for p, _, eff in lifted)


@pytest.mark.asyncio
async def test_query_op_applies_the_lift(db_session: object, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import particles.operations.query.main as qmain
    from particles import embeddings as ep

    _, _, record = await _seed(db_session)
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="answer"))
    get_config().query.exact_token_weight = 0.5
    original = ep._embedding_model
    ep.set_embedding_model(_mock_model([1.0, 0.0, 0.0, 0.0]))
    try:
        result = await qmain.query(
            db_session,  # type: ignore[arg-type]
            QueryRequest(question="Is RFC 9110 implementable?", top_k=2),
        )
    finally:
        ep.set_embedding_model(original)
    assert result.particles[0].id == record.id


async def test_federated_union_takes_the_viewers_weight(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The viewer's ``μ · X`` reaches every member store's candidates before the
    union is cut. A member store has no query configuration of its
    own: the only weight the federated path reads is the process's, the viewer's.
    """
    import particles._orm_modules  # noqa: F401
    import particles.operations.query.main as qmain
    from particles import embeddings as ep
    from particles.db import DEFAULT_STORE, Base, get_engine, reset_engine, session_scope
    from particles.store.particle_store import insert_particle

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/default.db"
    cfg.storage.stores = {"other": f"sqlite+aiosqlite:///{tmp_path}/other.db"}
    for handle in (DEFAULT_STORE, "other"):
        async with get_engine(handle).begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    near = _particle("RFC 9112 settled the split package.", 0.9)
    record = _particle("RFC 9110 is implementable once its review lands.", 0.6)
    async with session_scope(DEFAULT_STORE) as s:
        await insert_particle(s, near, [1.0, 0.0, 0.0, 0.0])
        await s.commit()
    async with session_scope("other") as s:
        await insert_particle(s, record, [0.6, 0.8, 0.0, 0.0])
        await s.commit()

    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="merged"))
    original = ep._embedding_model
    ep.set_embedding_model(_mock_model([1.0, 0.0, 0.0, 0.0]))
    request = QueryRequest(question="Is RFC 9110 implementable?", top_k=1)
    try:
        dense = await qmain.query_federated([DEFAULT_STORE, "other"], request)
        cfg.query.exact_token_weight = 0.5
        lifted = await qmain.query_federated([DEFAULT_STORE, "other"], request)
    finally:
        ep.set_embedding_model(original)
        for handle in (DEFAULT_STORE, "other"):
            await get_engine(handle).dispose()
        reset_engine()
    assert [p.id for p in dense.particles] == [near.id]
    assert [p.id for p in lifted.particles] == [record.id]
