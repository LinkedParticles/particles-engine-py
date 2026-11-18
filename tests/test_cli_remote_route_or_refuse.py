# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Route or refuse against a remote engine.

With ``engine.base_url`` set, a CLI verb either reaches the engine through the
``Backend`` seam or refuses with the message. It never reads or writes
the local store ``DATABASE_URL`` names, and never creates an empty one there.

* ``curate`` routes: the queue over ``GET /curation``, ``--card`` over
  ``GET /curation/cards/{key}``, ``--refresh`` over ``POST /curation/rebuild``.
  The engine is an ``httpx.MockTransport`` serving canned JSON; the response
  parsing is also checked against the real app over an in-process transport.
* ``curate apply``, ``curate --precision``, ``reconcile``, ``inbox
  process|watch``, ``interchange`` and ``synthesis-cache`` refuse.
* With no engine configured, each of them still runs against the local store.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.client.http import HttpBackend
from particles.core.schema import QualityReport
from particles.operations.curation import CardKind, CurationCard
from particles.operations.curation.cards import ParticleBrief, gestures_for

runner = CliRunner()

_REFUSAL = "is not available against a remote engine yet"

_ENGINE_URL = "https://engine.test"


def _plain(text: str) -> str:
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())


@pytest.fixture
def remote_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``engine.base_url`` set, and ``DATABASE_URL`` naming a store that does not exist.

    Returns that store's path: a verb that opened the local store would create
    the file, so its absence after the run shows the verb stayed off it.
    """
    local_db = tmp_path / "local-should-stay-absent.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{local_db}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", _ENGINE_URL)
    from particles.config import reset_config

    reset_config()
    return local_db


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

_REFUSED: list[tuple[list[str], str]] = [
    (["curate", "apply", "affirm", "contested:p-1"], "curate apply"),
    (["curate", "apply", "snooze", "stale:p-1", "--days", "3"], "curate apply"),
    (["curate", "--precision"], "curate --precision"),
    (["reconcile"], "reconcile"),
    (["db", "backup", "{tmp}/backup"], "db backup"),
    (["reconcile", "--dry-run"], "reconcile"),
    (["reconcile", "--updates"], "reconcile"),
    (["reconcile", "--dependents"], "reconcile"),
    (["inbox", "process"], "inbox process"),
    (["inbox", "watch"], "inbox watch"),
    (["interchange", "export", "-o", "{tmp}/out"], "interchange export"),
    (["interchange", "import", "{tmp}"], "interchange import"),
    (["interchange", "restore", "{tmp}"], "interchange restore"),
    (["synthesis-cache", "list"], "synthesis-cache list"),
    (["synthesis-cache", "show", "abc"], "synthesis-cache show"),
    (["synthesis-cache", "vacuum"], "synthesis-cache vacuum"),
    (["synthesis-cache", "vacuum", "--dry-run"], "synthesis-cache vacuum"),
    (["synthesis-cache", "evict", "abc", "--yes"], "synthesis-cache evict"),
    # The memory verbs that open a store (found by the 2026-11-07 cutover rehearsal:
    # `memory rescope` and `memory redate` opened the local store from a thin client).
    (["memory", "rescope", "--dry-run"], "memory rescope"),
    (["memory", "rescope", "--assign", "entry-1", "proj"], "memory rescope"),
    (["memory", "rescope", "--mark-agent-writes", "--dry-run"], "memory rescope"),
    (["memory", "widen", "p-1"], "memory widen"),
    (["memory", "redate", "--dry-run"], "memory redate"),
    (["memory", "useful", "p-1"], "memory useful"),
    (["memory", "rebuild-utility", "--yes"], "memory rebuild-utility"),
    (["memory", "sweep-rank-lift"], "memory sweep-rank-lift"),
    (["memory", "sweep-owner-lift"], "memory sweep-owner-lift"),
]


@pytest.mark.parametrize(
    ("argv", "verb"), _REFUSED, ids=[" ".join(a).replace("{tmp}", "TMP") for a, _ in _REFUSED]
)
def test_verb_refuses_against_a_remote_engine(
    argv: list[str], verb: str, remote_mode: Path, tmp_path: Path
) -> None:
    args = [a.replace("{tmp}", str(tmp_path)) for a in argv]
    result = runner.invoke(app, args)
    assert result.exit_code == 1, result.output
    out = _plain(result.output)
    assert f"`{verb}` {_REFUSAL}" in out
    assert "unset engine.base_url" in out
    assert not remote_mode.exists(), "the verb opened the local store"


def test_inbox_status_still_runs_remotely(
    remote_mode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``inbox status`` reads the inbox file only, never the store, so it is not refused."""
    inbox = tmp_path / "inbox.txt"
    inbox.write_text("https://example.com/a\n")
    monkeypatch.setenv("INBOX_FILE_PATH", str(inbox))
    from particles.config import reset_config

    reset_config()
    result = runner.invoke(app, ["inbox", "status"])
    assert result.exit_code == 0, result.output
    assert "Pending:   1" in result.output
    assert not remote_mode.exists()


# ---------------------------------------------------------------------------
# Local mode is unchanged
# ---------------------------------------------------------------------------


class TestStillLocalWithoutAnEngine:
    """With no ``engine.base_url`` the refusals are no-ops and each verb runs locally."""

    def test_reconcile_dry_run(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["reconcile", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert _REFUSAL not in result.output
        assert json.loads(result.output)["dry_run"] is True

    def test_synthesis_cache_list(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["synthesis-cache", "list"])
        assert result.exit_code == 0, result.output
        assert _REFUSAL not in result.output

    def test_synthesis_cache_vacuum_dry_run(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["synthesis-cache", "vacuum", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert _REFUSAL not in result.output

    def test_interchange_export_then_import(self, cli_db: Path, tmp_path: Path) -> None:
        out = tmp_path / "bundle"
        exported = runner.invoke(app, ["interchange", "export", "-o", str(out)])
        assert exported.exit_code == 0, exported.output
        assert "Wrote bundle" in exported.output
        imported = runner.invoke(app, ["interchange", "import", str(out)])
        assert imported.exit_code == 0, imported.output
        assert "Imported 0 particle(s)" in imported.output

    def test_inbox_process_reaches_its_own_checks(
        self, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Past the remote gate, the verb's own config check speaks (exit 2).
        monkeypatch.delenv("INBOX_FILE_PATH", raising=False)
        result = runner.invoke(app, ["inbox", "process"])
        assert result.exit_code == 2, result.output
        assert _REFUSAL not in result.output
        assert "inbox.file_path" in result.output

    def test_curate_queue(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["curate"])
        assert result.exit_code == 0, result.output
        assert "Store:" in result.output
        assert _REFUSAL not in result.output

    def test_curate_precision(self, cli_db: Path) -> None:
        result = runner.invoke(app, ["curate", "--precision"])
        assert result.exit_code == 0, result.output
        assert _REFUSAL not in result.output

    def test_curate_apply_reaches_the_operation(self, cli_db: Path) -> None:
        # An unparseable key is the operation's own refusal, not the remote one.
        result = runner.invoke(app, ["curate", "apply", "affirm", "not-a-key"])
        assert result.exit_code == 1
        assert _REFUSAL not in result.output


# ---------------------------------------------------------------------------
# curate routes over the engine
# ---------------------------------------------------------------------------


def _quality_json() -> dict[str, Any]:
    return QualityReport(
        active_particles=12,
        inconsistency_particles=1,
        calibration=[],
        extractor_direct_fraction=1.0,
        total_entries=3,
        snapshots_pending=0,
        snapshots_in_progress=0,
        snapshots_complete=3,
        snapshots_failed=2,
        total_subjects=4,
        subjects_without_particles=1,
    ).model_dump(mode="json")


def _card() -> CurationCard:
    return CurationCard(
        kind=CardKind.STALE,
        particle_ids=["p-remote"],
        particles=[
            ParticleBrief(
                particle_id="p-remote",
                content="The engine holds this belief.",
                subject_labels=["engine"],
                effective_confidence=0.6,
                status="ACTIVE",
            )
        ],
        diagnostic="A stale belief served by the engine.",
        suggested_gestures=gestures_for(CardKind.STALE),
        leverage=0.8,
    )


class _Engine:
    """A mock engine: canned JSON per route, and a log of every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.card = _card()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path == "/quality":
            return httpx.Response(200, json=_quality_json())
        if request.method == "GET" and path == "/curation":
            return httpx.Response(
                200,
                json={
                    "cards": [self.card.model_dump(mode="json")],
                    "count": 1,
                    "semantic_skipped": False,
                    "source": "snapshot",
                    "snapshot_id": "snap-1",
                    "built_at": "2026-11-07T03:30:00Z",
                    "age_seconds": 600.0,
                    "stale": False,
                    "scope": "store",
                    "per_kind_scope": {},
                    "collection_size": 9,
                    "open_count": 7,
                    "kind_as_of": {},
                },
            )
        if request.method == "GET" and path.startswith("/curation/cards/"):
            if path.endswith("stale:p-remote"):
                return httpx.Response(200, json=self.card.model_dump(mode="json"))
            return httpx.Response(404, json={"detail": "No open card."})
        if request.method == "POST" and path == "/curation/rebuild":
            return httpx.Response(
                200,
                json={
                    "snapshot_id": "snap-2",
                    "built_at": "2026-11-07T04:00:00Z",
                    "collection_size": 9,
                    "scope": "store",
                    "semantic": False,
                },
            )
        if request.method == "GET" and path.startswith("/particles/"):
            return httpx.Response(404, json={"detail": "Not found."})
        return httpx.Response(500, json={"detail": f"unexpected {request.method} {path}"})

    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]


@pytest.fixture
def engine(remote_mode: Path, monkeypatch: pytest.MonkeyPatch) -> _Engine:
    """Wire ``HttpBackend`` to the mock engine (its client is built via egress_client)."""
    fake = _Engine()
    real_async_client = httpx.AsyncClient

    def _client(
        *, base_url: str, timeout: float, headers: dict[str, str], purpose: str = ""
    ) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.MockTransport(fake.handler), base_url=base_url, headers=headers
        )

    monkeypatch.setattr("particles.api.client.http.egress_client", _client)
    return fake


class TestCurateRoutes:
    def test_queue_is_served_by_the_engine(self, engine: _Engine, remote_mode: Path) -> None:
        result = runner.invoke(app, ["curate", "--kind", "stale", "-n", "3"])
        assert result.exit_code == 0, result.output
        assert "Store: 12 active" in result.output
        assert "Curation queue: 1 of 7 open cards" in result.output
        assert "The engine holds this belief." in result.output
        assert "key: stale:p-remote" in result.output
        assert ("GET", "/curation") in engine.calls()
        sent = next(r for r in engine.requests if r.url.path == "/curation")
        assert sent.url.params["kind"] == "stale"
        assert sent.url.params["limit"] == "3"
        # The flag's default is off, sent explicitly so the engine's
        # curation.semantic default does not turn the paid finders on.
        assert sent.url.params["semantic"] == "false"
        assert sent.url.params["no_snapshot"] == "false"
        assert not remote_mode.exists(), "curate opened the local store"

    def test_no_snapshot_is_forwarded(self, engine: _Engine) -> None:
        result = runner.invoke(app, ["curate", "--no-snapshot"])
        assert result.exit_code == 0, result.output
        sent = next(r for r in engine.requests if r.url.path == "/curation")
        assert sent.url.params["no_snapshot"] == "true"

    def test_refresh_rebuilds_on_the_engine_first(self, engine: _Engine, remote_mode: Path) -> None:
        result = runner.invoke(app, ["curate", "--refresh"])
        assert result.exit_code == 0, result.output
        calls = engine.calls()
        assert calls.index(("POST", "/curation/rebuild")) < calls.index(("GET", "/curation"))
        assert not remote_mode.exists()

    def test_card_is_served_by_the_engine(self, engine: _Engine, remote_mode: Path) -> None:
        result = runner.invoke(app, ["curate", "--card", "stale:p-remote"])
        assert result.exit_code == 0, result.output
        assert "The engine holds this belief." in result.output
        assert any(p.startswith("/curation/cards/") for _, p in engine.calls())
        assert not remote_mode.exists()

    def test_unknown_card_is_a_plain_not_found(self, engine: _Engine) -> None:
        result = runner.invoke(app, ["curate", "--card", "stale:p-missing"])
        assert result.exit_code == 1
        assert "No open card with key" in _plain(result.output)


@pytest.fixture
def asgi_backend(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[HttpBackend, None, None]:
    """An ``HttpBackend`` wired to the real app in-process, over the ``cli_db`` store."""
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://127.0.0.1:8000")
    from particles.api.app import app as fastapi_app
    from particles.config import reset_config

    reset_config()
    real_async_client = httpx.AsyncClient

    def _client(
        *, base_url: str, timeout: float, headers: dict[str, str], purpose: str = ""
    ) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.ASGITransport(app=fastapi_app), base_url=base_url, headers=headers
        )

    monkeypatch.setattr("particles.api.client.http.egress_client", _client)
    yield HttpBackend()


class TestHttpBackendCurationAgainstTheRealApp:
    """The response models of the real routes parse into the local result types."""

    def test_queue_round_trips(self, asgi_backend: HttpBackend) -> None:
        result = asyncio.run(
            asgi_backend.curation_queue(limit=5, kind=None, semantic=False, no_snapshot=True)
        )
        assert result.cards == []
        assert result.source == "live"
        assert result.count == 0
        assert result.semantic is False

    def test_unknown_card_is_none(self, asgi_backend: HttpBackend) -> None:
        assert asyncio.run(asgi_backend.curation_card("stale:p-none")) is None
