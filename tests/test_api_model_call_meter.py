# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What one rate-limit token buys on the HTTP engine (security review 2026-10-25-2, R2, R6).

The limiter used to count requests, and one admitted request could run a model
pass over the whole store. The property pinned here: a caller without the
engine's bearer makes at most ``api.rate_limit_per_minute`` model calls a
minute, whichever route it calls, and a refused call is disclosed on the
response. Every model purpose is routed to an in-process scripted provider and
``network.egress`` is ``none``, so no call leaves the process.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import pytest_asyncio

from particles.api._rate_limit import REFUSED_CALLS_HEADER
from particles.llm import override_providers

_ENGINE = "http://127.0.0.1:8000"
_LOOPBACK = ("127.0.0.1", 50000)
_PURPOSES = (
    "extraction",
    "semantic_lint",
    "query_response",
    "synthesis",
    "abstraction",
    "verification",
    "subject_resolution",
    "use_judge",
)
_KEY = "a-real-secret"
_SUBJECTS = 6


class _Scripted:
    """Answers every prompt with a parseable no; counts the calls."""

    calls = 0

    @property
    def provider_model(self) -> str:
        return "test:scripted"

    async def complete(self, prompt: str, **_kw: Any) -> str:
        type(self).calls += 1
        return "[]" if "JSON" in prompt else "NO"


def _particle(content: str, entry: str) -> Any:
    from particles.core.schema import (
        CalibrationSource,
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )

    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        asserted_at=datetime.now(UTC) - timedelta(days=1),
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry)],
    )


async def _make_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rate: int, *, write_enabled: bool = True
) -> None:
    """A store with long claims (each a granularity probe) and six two-claim Subjects.

    ``write_enabled=False`` leaves the default store out of
    ``mcp.write.enabled_stores`` and pins it ``single`` through the config
    loader, a configuration the loader accepts (security review 2026-10-28-1,
    O72).
    """
    import particles._orm_modules  # noqa: F401
    from particles.config import reset_config
    from particles.core.schema import Subject
    from particles.db import Base, get_engine, session_scope
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject, link_particle_to_subjects

    cfg = tmp_path / "config.yaml"
    regime = (
        "mcp:\n  write:\n    enabled_stores: [default]\n"
        if write_enabled
        else "reconciliation:\n  per_store:\n    default: single\n"
    )
    cfg.write_text(
        f"network:\n  egress: none\napi:\n  rate_limit_per_minute: {rate}\n" + regime,
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    reset_config()
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with session_scope() as session:
        for s in range(_SUBJECTS):
            subject = Subject(canonical_name=f"Service {s}", asserted_by="test-agent")
            await insert_subject(session, subject)
            vec = np.zeros(384, dtype=np.float32)
            vec[s] = 1.0
            for variant in ("listens on", "is bound to"):
                p = _particle(
                    f"Service {s} {variant} port {8000 + s} on every host, and its nightly "
                    f"job rotates the logs at {s} o'clock every day of the week.",
                    f"entry-{s}-{variant}",
                )
                await insert_particle(session, p, embedding=vec.tolist())
                await link_particle_to_subjects(session, p.id, [subject.id])
        await session.commit()


@pytest_asyncio.fixture
async def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    from particles.config import reset_config
    from particles.db import get_engine

    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    await _make_store(tmp_path, monkeypatch, rate=1)
    try:
        yield
    finally:
        await get_engine().dispose()
        reset_config()


async def _request(method: str, path: str, **kw: Any) -> httpx.Response:
    from particles.api.app import app

    transport = httpx.ASGITransport(app=app, client=_LOOPBACK)
    with override_providers({p: _Scripted() for p in _PURPOSES}):  # type: ignore[misc]
        async with httpx.AsyncClient(transport=transport, base_url=_ENGINE) as client:
            return await client.request(method, path, **kw)


#: The model-reaching reports a keyless engine serves, each admitted on one token.
_SPENDING_READS: list[tuple[str, str, dict[str, Any]]] = [
    ("POST", "/lint", {"json": {"fix": False, "semantic": True}}),
    ("GET", "/curation", {"params": {"semantic": "true", "no_snapshot": "true"}}),
    ("POST", "/links/suggest", {"json": {"mode": "LLM_JUDGE"}}),
]


@pytest.mark.parametrize(("method", "path", "kw"), _SPENDING_READS, ids=lambda v: str(v))
@pytest.mark.asyncio
async def test_one_token_buys_at_most_one_model_call(
    store: None, method: str, path: str, kw: dict[str, Any]
) -> None:
    """With api.rate_limit_per_minute=1 a keyless caller makes at most one call a minute."""
    _Scripted.calls = 0
    first = await _request(method, path, **kw)
    assert first.status_code == 200, first.text
    assert _Scripted.calls == 1, "the route made no model call: the test saw nothing"
    # The pass wanted more than the one call; the response says so.
    assert int(first.headers[REFUSED_CALLS_HEADER]) >= 1
    again = await _request(method, path, **kw)
    assert again.status_code == 429
    assert _Scripted.calls == 1


@pytest.mark.asyncio
async def test_a_judged_link_suggestion_still_lists_every_candidate(store: None) -> None:
    """A judge pass cut short by the meter still reports the candidates it found."""
    resp = await _request("POST", "/links/suggest", json={"mode": "LLM_JUDGE"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["total_candidates"] >= _SUBJECTS


@pytest_asyncio.fixture
async def single_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """The ``store`` fixture's data on a default store that is not write-enabled and is single."""
    from particles.config import reset_config
    from particles.db import get_engine

    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    await _make_store(tmp_path, monkeypatch, rate=1, write_enabled=False)
    try:
        yield
    finally:
        await get_engine().dispose()
        reset_config()


@pytest.mark.asyncio
async def test_a_reconcile_dry_run_is_metered(
    single_store: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /reconcile dry_run spends one call per probe from the same allowance."""
    import particles.operations.reconcile as reconcile
    from particles.config import get_config

    candidates = [
        (
            0.99 - i / 100,
            _particle(f"The service listens on port {8000 + i}.", f"old-{i}"),
            _particle(f"The service listens on port {9000 + i}.", f"new-{i}"),
            f"new-{i}",
            f"old-{i}",
        )
        for i in range(5)
    ]

    async def _pairs(_session: Any) -> list[tuple[str, str]]:
        return [(f"new-{i}", f"old-{i}") for i in range(5)]

    async def _candidates(*_a: Any) -> list[Any]:
        return list(candidates)

    async def _current(_session: Any) -> None:
        return None

    # A write-enabled store reconciles multi, which makes the document sweep a
    # no-op (security review 2026-10-27-1, G6), and the loader refuses one
    # pinned single. This store is not write-enabled and is single by its
    # loaded configuration (security review 2026-10-28-1, O72).
    assert get_config().reconciliation_mode_for("default") == "single"
    assert "default" not in get_config().mcp.write.enabled_stores
    monkeypatch.setattr(reconcile, "iter_supersession_entry_pairs", _pairs)
    monkeypatch.setattr(reconcile, "_supersession_candidates", _candidates)
    monkeypatch.setattr("particles.operations.version_guard.assert_store_schema_current", _current)
    _Scripted.calls = 0
    resp = await _request("POST", "/reconcile", json={"dry_run": True})
    assert resp.status_code == 200, resp.text
    assert resp.json()["probed"] == 5
    assert _Scripted.calls == 1
    assert resp.headers[REFUSED_CALLS_HEADER] == "4"


@pytest.mark.asyncio
async def test_a_bearer_holder_is_not_metered_by_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bearer carries the operator's authority: its report runs whole, on one token."""
    from particles.config import reset_config
    from particles.db import get_engine

    monkeypatch.setenv("PARTICLES_API_KEY", _KEY)
    await _make_store(tmp_path, monkeypatch, rate=1)
    try:
        _Scripted.calls = 0
        resp = await _request(
            "POST",
            "/links/suggest",
            json={"mode": "LLM_JUDGE"},
            headers={"Authorization": f"Bearer {_KEY}"},
        )
        assert resp.status_code == 200, resp.text
        assert _Scripted.calls >= _SUBJECTS
        assert REFUSED_CALLS_HEADER not in resp.headers
    finally:
        await get_engine().dispose()
        reset_config()


@pytest.mark.asyncio
async def test_agent_text_deposit_is_rate_limited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6: a bearer holder's agent text deposits, which extraction later reads, are rate-limited."""
    from particles.config import reset_config
    from particles.db import get_engine

    monkeypatch.setenv("PARTICLES_API_KEY", _KEY)
    await _make_store(tmp_path, monkeypatch, rate=1)
    try:
        statuses = []
        for i in range(3):
            resp = await _request(
                "POST",
                "/particles/deposit",
                json={"text": f"Conversation fragment {i}: we chose port {9000 + i}."},
                headers={"Authorization": f"Bearer {_KEY}"},
            )
            statuses.append(resp.status_code)
        assert statuses == [200, 429, 429]
    finally:
        await get_engine().dispose()
        reset_config()
