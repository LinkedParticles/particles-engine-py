# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A keyless engine serves reads only, and every gate runs first.

The route walk pins the route classification: every route names a
gate in its decorator, every non-``GET`` route is a write route behind a write
gate unless it is listed below as a read or a branch writer, and no read or
branch-writing route takes a write session. The behavioural half calls every
write route on a keyless engine from loopback, with and without a bearer, and
asserts the one 503 and that no store lock was taken (security review
2026-10-25, Q12). The two hand-kept sets are deliberate friction:
classifying a route as a read is a reviewed edit here, never a default.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import sqlite3
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from particles.api import app as app_module
from particles.api import auth
from particles.api.app import app

_ENGINE = "http://127.0.0.1:8000"

#: Routes with no gate at all (app shell and its redirect, /health).
OPEN = {("GET", "/health"), ("GET", "/")}

#: Non-GET routes that change nothing. Each entry is a reviewed decision.
READ_ONLY_POSTS = {
    ("POST", "/query"),
    ("POST", "/corpus/links/suggest"),
    ("POST", "/particles/contested-badges"),
    ("POST", "/memory/projection/{store}"),
}

#: Routes that write on one branch only, with a body that takes the write branch.
BRANCH_WRITERS: dict[tuple[str, str], dict[str, Any]] = {
    ("POST", "/lint"): {"fix": True},
    ("POST", "/links/suggest"): {"mode": "APPLY", "confirmed": True},
    ("POST", "/reconcile"): {"dry_run": False},
    ("POST", "/subjects/merge"): {"source_ids": ["a"], "target_id": "b", "dry_run": False},
    ("POST", "/corpus/{entry_id}/retract"): {"dry_run": False},
    ("POST", "/subjects/relink-gated"): {"dry_run": False},
    ("POST", "/reindex"): {"dry_run": False},
}

_GATES: dict[Callable[..., Any], str] = {
    auth._read_gate: "read",
    auth._auth_gate: "auth",
    auth._agent_gate: "write",
    auth._operator_gate: "operator",
}
_WRITE_SESSIONS = {"get_write_session", "_store_write_session"}


def _routes() -> list[tuple[str, str, APIRoute]]:
    out = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            for method in sorted(route.methods):
                out.append((method, route.path, route))
    return out


def _first_gate(route: APIRoute) -> str | None:
    deps = route.dependant.dependencies
    return _GATES.get(deps[0].call) if deps else None  # type: ignore[arg-type]


def _dependency_names(dependant: Dependant) -> set[str]:
    names: set[str] = set()
    for dep in dependant.dependencies:
        names.add(getattr(dep.call, "__name__", ""))
        names |= _dependency_names(dep)
    return names


def _write_routes() -> list[tuple[str, str]]:
    return [
        (method, path)
        for method, path, route in _routes()
        if _first_gate(route) in {"write", "operator"}
    ]


# --- the walk -------------------------------------------------------------------


def test_every_listed_route_exists() -> None:
    """A stale entry in a hand-kept set would hide a route that moved."""
    present = {(method, path) for method, path, _ in _routes()}
    assert present >= READ_ONLY_POSTS
    assert set(BRANCH_WRITERS) <= present
    # The root redirect exists only when the web-UI bundle is on disk to mount,
    # which a published engine tree is not.
    mounted = app_module._web_ui_static_app is not None
    assert present >= (OPEN if mounted else OPEN - {("GET", "/")})


def test_every_gated_route_names_its_gate_first() -> None:
    """The gate is the first dependency, so it refuses before any session opens (Q12)."""
    missing = [
        (method, path)
        for method, path, route in _routes()
        if (method, path) not in OPEN and _first_gate(route) is None
    ]
    assert missing == []


def test_every_non_get_route_is_a_gated_write_or_a_listed_exception() -> None:
    unclassified = []
    for method, path, route in _routes():
        if method == "GET" or (method, path) in OPEN:
            continue
        gate = _first_gate(route)
        if gate in {"write", "operator"}:
            assert (method, path) not in READ_ONLY_POSTS, (method, path)
            assert (method, path) not in BRANCH_WRITERS, (method, path)
        elif (method, path) in READ_ONLY_POSTS or (method, path) in BRANCH_WRITERS:
            assert gate in {"read", "auth"}, (method, path, gate)
        else:
            unclassified.append((method, path, gate))
    assert unclassified == [], "a write route without a write gate, or an unlisted read POST"


def test_no_read_or_branch_writer_takes_a_write_session() -> None:
    offenders = []
    for method, path, route in _routes():
        listed = (method, path) in READ_ONLY_POSTS or (method, path) in BRANCH_WRITERS
        if (method == "GET" or listed) and _dependency_names(route.dependant) & _WRITE_SESSIONS:
            offenders.append((method, path))
    assert offenders == []


@pytest.mark.parametrize("key", sorted(READ_ONLY_POSTS))
def test_a_read_post_handler_writes_nothing(key: tuple[str, str]) -> None:
    route = next(r for m, p, r in _routes() if (m, p) == key)
    source = inspect.getsource(route.endpoint)
    assert not re.search(r"commit\(|write_transaction", source), key


# --- behaviour on a keyless engine ----------------------------------------------


@pytest.fixture
def lock_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every acquisition of the store's writer lock."""
    import particles.db as db

    calls: list[str] = []
    original = db.write_lock

    @asynccontextmanager
    async def _spy(*args: Any, **kwargs: Any) -> AsyncIterator[None]:
        calls.append("lock")
        async with original(*args, **kwargs):
            yield

    monkeypatch.setattr(db, "write_lock", _spy)
    return calls


@pytest.fixture
def keyless(cli_db: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    yield TestClient(app, base_url=_ENGINE, client=("127.0.0.1", 50000))


def _fill(path: str) -> str:
    return re.sub(r"\{[^}:]+(:path)?\}", "x", path)


def _call(client: TestClient, method: str, path: str, body: dict[str, Any], bearer: bool) -> Any:
    headers = {"Authorization": "Bearer anything"} if bearer else {}
    return client.request(method, _fill(path), json=body, headers=headers)


@pytest.mark.parametrize("bearer", [False, True], ids=["no-bearer", "bearer"])
def test_every_write_route_refuses_a_keyless_engine_and_takes_no_lock(
    keyless: TestClient, lock_calls: list[str], bearer: bool
) -> None:
    routes = _write_routes()
    assert len(routes) >= 30, "the walk found too few write routes to be the real table"
    wrong = []
    for method, path in routes:
        resp = _call(keyless, method, path, {}, bearer)
        if resp.status_code != 503 or resp.json()["detail"] != auth.KEYLESS_WRITE_REFUSAL:
            wrong.append((method, path, resp.status_code))
    assert wrong == []
    assert lock_calls == []


@pytest.mark.parametrize("key", sorted(BRANCH_WRITERS))
def test_a_branch_writers_write_branch_refuses_a_keyless_engine(
    keyless: TestClient, lock_calls: list[str], key: tuple[str, str]
) -> None:
    method, path = key
    resp = _call(keyless, method, path, BRANCH_WRITERS[key], bearer=False)
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == auth.KEYLESS_WRITE_REFUSAL
    assert lock_calls == []


def test_control_a_keyed_write_takes_the_lock_the_spy_watches(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch, lock_calls: list[str]
) -> None:
    """Without this, the no-lock assertions above could pass on a spy that sees nothing."""
    monkeypatch.setenv("PARTICLES_API_KEY", "a-real-secret")
    client = TestClient(app, base_url=_ENGINE, client=("127.0.0.1", 50000))
    resp = client.post(
        "/corpus/deposit/text",
        json={"text": "A note."},
        headers={"Authorization": "Bearer a-real-secret"},
    )
    assert resp.status_code == 200, resp.text
    assert lock_calls


def test_a_keyed_engine_answers_a_missing_bearer_with_401_not_503(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status alone tells "engine has no key" from "your token is wrong"."""
    monkeypatch.setenv("PARTICLES_API_KEY", "a-real-secret")
    client = TestClient(app, base_url=_ENGINE, client=("127.0.0.1", 50000))
    resp = client.post("/corpus/deposit/text", json={"text": "x"})
    assert resp.status_code == 401


def test_a_keyless_engine_still_serves_reads(keyless: TestClient) -> None:
    assert keyless.get("/corpus").status_code == 200
    assert keyless.get("/events?limit=1").status_code == 200
    assert keyless.post("/particles/contested-badges", json={"particle_ids": []}).status_code == 200


def _row_counts(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}


async def _seed_particle() -> None:
    from particles.core.schema import (
        CalibrationSource,
        Confidence,
        Particle,
        Status,
        UncertaintyNature,
    )
    from particles.db import get_engine, session_scope
    from particles.store.particle_store import insert_particle

    particle = Particle(
        id=str(uuid.uuid4()),
        content="The deploy runs nightly.",
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
    )
    try:
        async with session_scope() as session:
            await insert_particle(session, particle, (np.ones(4, dtype=np.float32) / 2.0).tolist())
            await session.commit()
    finally:
        await get_engine().dispose()


_READ_POST_BODIES: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {
    ("POST", "/query"): ("/query", {"question": "When does the deploy run?"}),
    ("POST", "/corpus/links/suggest"): ("/corpus/links/suggest", {}),
    ("POST", "/particles/contested-badges"): ("/particles/contested-badges", {"particle_ids": []}),
    ("POST", "/memory/projection/{store}"): (
        "/memory/projection/default",
        {
            "manifest": {
                "name": "memory-index",
                "sections": [{"title": "Memory index", "render": "bullets", "top_k": 60}],
            }
        },
    ),
}


@pytest.mark.parametrize("key", sorted(READ_ONLY_POSTS))
def test_a_read_post_changes_no_table(
    keyless: TestClient, cli_db: Path, key: tuple[str, str]
) -> None:
    """Check 5: a route filed as a read writes nothing on a seeded store."""
    from particles import embeddings

    assert set(_READ_POST_BODIES) == READ_ONLY_POSTS
    asyncio.run(_seed_particle())
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32))
    original = embeddings._embedding_model
    embeddings.set_embedding_model(model)
    try:
        before = _row_counts(cli_db)
        path, body = _READ_POST_BODIES[key]
        resp = keyless.post(path, json=body)
        assert resp.status_code == 200, resp.text
        assert _row_counts(cli_db) == before
    finally:
        embeddings.set_embedding_model(original)


# --- semantic reports on a keyless engine (security review 2026-10-25-2, R5) ----

#: Read routes whose report can reach the model, and so the probe-verdict ledger.
SEMANTIC_REPORTS: dict[tuple[str, str], dict[str, Any]] = {
    ("POST", "/lint"): {"json": {"fix": False, "semantic": True}},
    ("GET", "/curation"): {"params": {"semantic": "true", "no_snapshot": "true"}},
}


class _ScriptedContradiction:
    """Answers every probe with a contradiction verdict; counts the calls."""

    calls = 0

    @property
    def provider_model(self) -> str:
        return "test:scripted"

    async def complete(self, prompt: str, **_kw: Any) -> str:
        type(self).calls += 1
        return "REASON: scripted\nSLOT: CHANGES\nVERDICT: YES"


async def _seed_contradicting_pair() -> None:
    from particles.core.schema import (
        CalibrationSource,
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    async with session_scope() as session:
        for i, (content, x, y) in enumerate(
            [("The port is 8080.", 0.6, 0.8), ("The port is 9090.", 0.61, 0.79)]
        ):
            particle = Particle(
                content=content,
                confidence=Confidence(
                    value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="test-agent",
                asserted_at=datetime.now(UTC),
                provenance=[
                    ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=f"entry-{i}")
                ],
            )
            vec = np.array([x, y] + [0.0] * 382, dtype=np.float32)
            await insert_particle(session, particle, embedding=vec.tolist())
        await session.commit()


async def _semantic_report(key: tuple[str, str], bearer: str | None) -> Any:
    import httpx

    from particles.llm import override_providers

    method, path = key
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    scripted = _ScriptedContradiction()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 50000))
    purposes = ("semantic_lint", "verification", "use_judge", "abstraction")
    with override_providers({p: scripted for p in purposes}):  # type: ignore[misc]
        async with httpx.AsyncClient(transport=transport, base_url=_ENGINE) as client:
            return await client.request(method, path, headers=headers, **SEMANTIC_REPORTS[key])


@pytest.mark.parametrize("key", sorted(SEMANTIC_REPORTS))
@pytest.mark.asyncio
async def test_a_keyless_semantic_report_changes_no_table(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch, lock_calls: list[str], key: tuple[str, str]
) -> None:
    """A report the model answers is still a read on a keyless engine, and takes no lock."""
    from particles.db import get_engine

    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    await _seed_contradicting_pair()
    lock_calls.clear()
    before = _row_counts(cli_db)
    _ScriptedContradiction.calls = 0
    try:
        resp = await _semantic_report(key, bearer=None)
    finally:
        await get_engine().dispose()
    assert resp.status_code == 200, resp.text
    assert _ScriptedContradiction.calls > 0, "the model was never asked: the test saw nothing"
    assert _row_counts(cli_db) == before
    assert lock_calls == []


@pytest.mark.asyncio
async def test_control_a_bearer_holders_semantic_lint_keeps_its_verdicts(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this, the no-change assertion above could pass on a census that records nothing."""
    from particles.db import get_engine

    monkeypatch.setenv("PARTICLES_API_KEY", "a-real-secret")
    await _seed_contradicting_pair()
    before = _row_counts(cli_db)["probe_verdicts"]
    try:
        resp = await _semantic_report(("POST", "/lint"), bearer="a-real-secret")
    finally:
        await get_engine().dispose()
    assert resp.status_code == 200, resp.text
    assert _row_counts(cli_db)["probe_verdicts"] > before
