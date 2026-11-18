# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Subject suggestions for a belief.

The pure half pins the tokenizer, the name runs, the two-of-twenty rule and
the merge; the shell half pins what the store gathers for each signal, the
persona guard, the observer filter and the route's contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle, Subject
from particles.core.scoring.confidence import CalibrationSource
from particles.operations.curation.subject_suggestions import (
    MAX_NAME_WORDS,
    NamedMatch,
    NeighbourSupport,
    Suggestion,
    claim_tokens,
    name_keys,
    neighbour_support,
    rank_suggestions,
    suggest_subjects,
)
from particles.store.particle_store import insert_particle
from particles.store.subject_store import insert_subject
from tests._observer_scope import belief, project_source_tags, rescoped, source

# --- The tokenizer and the name runs ------------------------------------------


@pytest.mark.parametrize(
    ("raw", "token"),
    [
        ('"quoted"', "quoted"),
        ("'single'", "single"),
        ("“curly”", "curly"),
        ("(round)", "round"),
        ("[square]", "square"),
        ("{brace}", "brace"),
        ("comma,", "comma"),
        ("colon:", "colon"),
        ("semi;", "semi"),
        ("why?", "why"),
        ("bang!", "bang"),
        ("*bold*", "bold"),
        ("`code`", "code"),
        ("stop.", "stop"),
        ("trust'.", "trust"),
        ("(`uv run`).", "uv run"),
        ("v1.2.3", "v1.2.3"),
        ("particles/", "particles/"),
    ],
)
def test_each_token_loses_its_surrounding_punctuation(raw: str, token: str) -> None:
    assert claim_tokens(raw) == token.split()


def test_a_token_of_punctuation_alone_is_dropped() -> None:
    assert claim_tokens('a "" (.) b') == ["a", "b"]


def test_name_keys_fold_and_skip_short_runs_longest_first() -> None:
    keys = name_keys(["Uv", "Run", "x"])
    assert keys == ["uv run x", "uv run", "run x", "run"]


def test_a_ten_word_name_is_a_key_and_an_eleven_word_one_is_not() -> None:
    words = [f"w{i}" for i in range(MAX_NAME_WORDS + 1)]
    keys = name_keys(words)
    assert " ".join(words[:MAX_NAME_WORDS]) in keys
    assert " ".join(words) not in keys


# --- The neighbour signal and the merge -----------------------------------------


def test_a_neighbour_subject_needs_the_support_of_two_neighbours() -> None:
    found = neighbour_support(
        [(0.9, {"a", "b"}), (0.8, {"a"}), (0.7, {"c"}), (0.6, {"c", "b"})], min_support=2
    )
    assert [(s.subject_id, s.count) for s in found] == [("a", 2), ("b", 2), ("c", 2)]
    assert found[0].score == pytest.approx(1.7)
    alone = neighbour_support([(0.9, {"a"}), (0.8, {"b"})], min_support=2)
    assert alone == []


def test_the_merge_puts_three_named_first_then_neighbours_then_the_rest() -> None:
    named = [
        NamedMatch("short", 5, 1),
        NamedMatch("long-common", 20, 50),
        NamedMatch("long-rare", 20, 2),
        NamedMatch("mid", 10, 3),
        NamedMatch("tail", 4, 1),
    ]
    neighbours = [NeighbourSupport("n1", 2.0, 3), NeighbourSupport("mid", 1.5, 2)]
    out = rank_suggestions(named, neighbours, exclude=set(), limit=20)
    assert [s.subject_id for s in out] == ["long-rare", "long-common", "mid", "n1", "short", "tail"]
    assert out[2] == Suggestion("mid", named=True, neighbours=2)
    assert out[3] == Suggestion("n1", named=False, neighbours=3)


def test_the_merge_cuts_to_the_limit_and_never_offers_a_carried_subject() -> None:
    named = [NamedMatch("a", 9, 1), NamedMatch("b", 8, 1)]
    neighbours = [NeighbourSupport("b", 1.0, 2), NeighbourSupport("c", 0.9, 2)]
    out = rank_suggestions(named, neighbours, exclude={"a"}, limit=2)
    assert [s.subject_id for s in out] == ["b", "c"]


# --- The shell ------------------------------------------------------------------


def _vec(*xs: float) -> list[float]:
    v = np.asarray(xs, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()  # type: ignore[no-any-return]


async def _subject(session: AsyncSession, name: str, aliases: list[str] | None = None) -> Subject:
    s = Subject(canonical_name=name, aliases=aliases or [], asserted_by="test")
    await insert_subject(session, s)
    return s


async def _belief(
    session: AsyncSession,
    content: str,
    vector: list[float] | None,
    subjects: list[Subject] | None = None,
    tags: list[str] | None = None,
) -> Particle:
    from particles.core.schema import (
        Confidence,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )

    entry_id, snapshot_id = await source(session, content[:20], tags or ["web"])
    p = Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        subject_ids=[s.id for s in subjects or []],
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
    )
    await insert_particle(session, p, vector)
    return p


async def _names(session: AsyncSession, p: Particle, **kwargs: Any) -> list[tuple[str, bool, int]]:
    out = await suggest_subjects(session, p, **kwargs)
    return [(s.subject.canonical_name, s.named, s.neighbours) for s in out]


async def test_named_subjects_by_canonical_name_and_alias(
    db_session: AsyncSession, candidate_cache_mode: str
) -> None:
    await _subject(db_session, "uv run lint-imports")
    # The alias is shorter than "uv run lint-imports" so the order is decided by
    # length alone. Keep it free of the upstream repository's name: the export
    # rewrites that name to the target repo's, which can be exactly as long and
    # leave the order to the random subject ids.
    await _subject(db_session, "Particles", aliases=["particles-sdk"])
    ten = " ".join(f"w{i}" for i in range(10))
    await _subject(db_session, ten)
    claim = await _belief(
        db_session, f"Run `uv run lint-imports` in particles-sdk; then {ten}.", None
    )
    names = await _names(db_session, claim)
    assert names == [
        (ten, True, 0),
        ("uv run lint-imports", True, 0),
        ("Particles", True, 0),
    ]


async def test_a_subject_the_belief_carries_is_never_suggested(db_session: AsyncSession) -> None:
    gate = await _subject(db_session, "gate checks")
    await _subject(db_session, "mypy")
    claim = await _belief(db_session, "The gate checks run mypy.", None, [gate])
    assert await _names(db_session, claim) == [("mypy", True, 0)]


async def test_a_persona_form_is_suggested_only_inside_a_persona_source(
    db_session: AsyncSession,
) -> None:
    await _subject(db_session, "the user", aliases=["speaker"])
    web = await _belief(db_session, "Speaker notes are kept in the deck.", None, tags=["web"])
    mine = await _belief(db_session, "Speaker prefers tabs.", None, tags=["memory-file"])
    assert await _names(db_session, web) == []
    assert await _names(db_session, mine) == [("the user", True, 0)]


async def test_neighbour_subjects_need_two_of_the_nearest_and_no_vector_gets_named_only(
    db_session: AsyncSession, candidate_cache_mode: str
) -> None:
    shared = await _subject(db_session, "shared theme")
    lone = await _subject(db_session, "one-off")
    await _subject(db_session, "named thing")
    await _belief(db_session, "near one", _vec(1, 0.1, 0), [shared])
    await _belief(db_session, "near two", _vec(1, 0.2, 0), [shared, lone])
    await _belief(db_session, "far away", _vec(0, 0, 1), [lone])
    orphan = await _belief(db_session, "An orphan about the named thing.", _vec(1, 0.15, 0))
    unembedded = await _belief(db_session, "Another orphan about the named thing.", None)

    assert await _names(db_session, orphan) == [
        ("named thing", True, 0),
        ("shared theme", False, 2),
    ]
    assert await _names(db_session, unembedded) == [("named thing", True, 0)]
    assert await _names(db_session, orphan, limit=1) == [("named thing", True, 0)]


async def test_an_observer_project_keeps_out_of_view_neighbours_from_the_count(
    db_session: AsyncSession,
) -> None:
    in_a = await source(db_session, "a", project_source_tags("-proj-a"))
    in_b = await source(db_session, "b", project_source_tags("-proj-b"))
    ours = await _subject(db_session, "our theme")
    theirs = await _subject(db_session, "their theme")
    for i in range(2):
        await belief(db_session, f"A claim {i}.", in_a, subject_ids=[ours.id])
        await belief(db_session, f"B claim {i}.", in_b, subject_ids=[theirs.id])
    orphan = await belief(db_session, "An orphan in A.", in_a)
    await rescoped(db_session)
    await db_session.commit()

    everyone = {name for name, _, _ in await _names(db_session, orphan)}
    assert everyone == {"our theme", "their theme"}
    seen_from_a = await _names(db_session, orphan, observer_project="-proj-a")
    assert seen_from_a == [("our theme", False, 2)]


# --- The route ---------------------------------------------------------------------


def test_the_route_answers_404_for_an_unknown_belief(api_client: Any) -> None:
    resp = api_client.get("/particles/00000000-0000-4000-8000-000000000000/subject-suggestions")
    assert resp.status_code == 404


def test_the_route_returns_each_subject_with_its_reasons(api_client: Any) -> None:
    import asyncio

    from particles.db import session_scope

    async def seed() -> str:
        async with session_scope() as session:
            await _subject(session, "gate checks")
            claim = await _belief(session, "The gate checks run first.", None)
            await session.commit()
            return claim.id

    pid = asyncio.run(seed())
    resp = api_client.get(f"/particles/{pid}/subject-suggestions", params={"limit": 50})
    assert resp.status_code == 200
    body = resp.json()
    assert [(i["subject"]["canonical_name"], i["named"], i["neighbours"]) for i in body] == [
        ("gate checks", True, 0)
    ]
    assert (
        api_client.get(f"/particles/{pid}/subject-suggestions", params={"limit": 0}).status_code
        == 422
    )


@pytest.fixture
def api_client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A keyed engine over a file store with its tables created (as ``tests/test_app.py``)."""
    import asyncio

    import particles._orm_modules  # noqa: F401
    from particles.config import reset_config
    from particles.db import Base, get_engine
    from tests._engine_client import keyed_engine_client

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    reset_config()

    async def create() -> None:
        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(create())
    yield keyed_engine_client(monkeypatch)

    async def dispose() -> None:
        await get_engine().dispose()

    asyncio.run(dispose())
    reset_config()
