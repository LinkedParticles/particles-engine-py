# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the keys-and-roles findings of security review 2026-10-28-1.

O44: a harvester deposit never stores a URI holding a raw control character.
O46: a project key that renders as another, or spells it in another
normalisation form, is never a distinct binding.
O47 (E17, E18, Z38): every refusal on the agent and operator write routes
arrives before the store's writer lock, and a useful press commits under it.
O48: revoking the last live per-client key never turns a running engine keyless.
O49: the forwarded-hop walk reads every ``X-Forwarded-For`` field line.
O50 (Z35): the key file is never read through a link, nor from a directory
another user may write into.
O51: the rank-lift monitor reads only the exposure rows the session frame trusts.
O79: a hand-written key record ``check_mint`` refuses never authenticates.
O80: a URL deposit's response names a follow it did not make.

(O45, the mint's non-terminal refusal, is in ``tests/test_engine_keys.py``.)

Each test uses a scratch store and key file under ``tmp_path``, sets
configuration only through the loader, and makes no model call or outside
connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import unicodedata
from collections.abc import AsyncIterator, Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

LEGACY = "sec8-legacy-operator-key"


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str = "") -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n" + body, encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))


def _create_tables() -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    asyncio.run(_go())


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """A scratch store, a key file and ``PARTICLES_API_KEY`` set to a real key."""
    from particles.api import _keys
    from particles.config import reset_config

    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sec8.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", LEGACY)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    _write_config(tmp_path, monkeypatch)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    yield keydir
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


def _enable_belief_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from particles.config import get_config, reset_config

    _write_config(tmp_path, monkeypatch, "mcp:\n  write:\n    enabled_stores: [default]\n")
    reset_config()
    assert get_config().mcp.write.enabled_stores == ["default"]


@pytest.fixture
def lock_log(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every acquisition of a store's writer lock."""
    import particles.db as db

    taken: list[str] = []
    real = db.write_lock

    @contextlib.asynccontextmanager
    async def _recording(store: str = "default") -> AsyncIterator[None]:
        taken.append(store)
        async with real(store):
            yield

    monkeypatch.setattr(db, "write_lock", _recording)
    return taken


def _tokens() -> dict[str, str]:
    from particles.api import _keys
    from particles.api._keys import Role
    from particles.config import get_config

    operator = get_config().curation.operator_identity
    return {
        "reader": _keys.mint(Role.READER, "viewer")[0],
        "agent": _keys.mint(Role.AGENT, "agent:ci-bot")[0],
        "harvester": _keys.mint(Role.AGENT, "claude-code-hook", reserved_ok=True)[0],
        "writer-label": _keys.mint(Role.AGENT, "general-extractor", reserved_ok=True)[0],
        "mcp": _keys.mint(Role.AGENT, "mcp:claude-code")[0],
        "operator": _keys.mint(Role.OPERATOR, operator)[0],
        "legacy": LEGACY,
    }


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# O44: the harvest URI is judged as stored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri",
    [
        "file:///home/u/.claude/projects/x/memory/a\r\nb.md",
        "file:///home/u/.claude/projects/x/memory/a\tb.md",
        "file:///home/u/.claude/projects/x/memory/a.md\n",
        "claude-code://session/abc\n",
    ],
)
def test_the_harvest_shape_refuses_a_raw_control_character(uri: str) -> None:
    from particles.api.app import _is_harvest_uri

    assert not _is_harvest_uri(uri), repr(uri)


def test_the_harvest_shape_still_admits_a_memory_file() -> None:
    from particles.api.app import _is_harvest_uri

    assert _is_harvest_uri("file:///home/u/.claude/projects/x/memory/a.md")


def test_a_harvester_deposit_never_stores_a_uri_holding_a_raw_control_character(
    store: Path,
) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    token, _ = _keys.mint(Role.AGENT, "claude-code-hook", reserved_ok=True)
    resp = _client().post(
        "/corpus/deposit/text",
        json={
            "text": "memory text",
            "uri_r": "file:///home/u/.claude/projects/x/memory/a\r\nb\t.md",
            "source_type": "LOCAL_MARKDOWN",
        },
        headers=_bearer(token),
    )

    async def _uris() -> list[str]:
        from particles.corpus.store import list_entries
        from particles.db import get_engine, session_scope

        async with session_scope() as s:
            out = [e.uri_r or "" for e in await list_entries(s)]
        await get_engine().dispose()
        return out

    assert resp.status_code == 403, resp.text
    assert not any(ch in uri for uri in asyncio.run(_uris()) for ch in "\r\n\t")


# ---------------------------------------------------------------------------
# O46: one project, one binding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    ["acme​", "ac­me", "acme️", "﻿acme", "ac⁠me"],
    ids=["zero-width-space", "soft-hyphen", "variation-selector", "bom", "word-joiner"],
)
def test_a_project_key_that_renders_as_another_is_never_a_distinct_binding(
    store: Path, spelling: str
) -> None:
    from particles.api import _keys
    from particles.api._keys import KeyRuleError, Role

    try:
        _token, record = _keys.mint(Role.AGENT, "agent:ci-bot", project=spelling)
    except KeyRuleError:
        assert not _keys.live_keys()
        return
    assert record.project == "acme", repr(record.project)


def test_one_project_in_two_normalisation_forms_is_one_project(store: Path) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    composed = "café"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    _token, record = _keys.mint(Role.AGENT, "agent:ci-bot", project=decomposed)
    assert record.project == composed


def test_a_key_file_record_binding_a_lookalike_project_makes_the_file_unusable(
    store: Path,
) -> None:
    from particles.api import _keys
    from particles.api._keys import KeyFileUnusable, Role

    _keys.mint(Role.AGENT, "agent:ci-bot", project="acme")
    path = store / "engine-keys.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["keys"][0]["project"] = "acme​"
    path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(path, 0o600)
    _keys.reset_key_cache()
    with pytest.raises(KeyFileUnusable, match="invisible"):
        _keys.all_keys()


def test_a_key_file_record_in_decomposed_form_binds_the_composed_project(store: Path) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    _keys.mint(Role.AGENT, "agent:ci-bot", project="café")
    path = store / "engine-keys.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["keys"][0]["project"] = unicodedata.normalize("NFD", "café")
    path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(path, 0o600)
    _keys.reset_key_cache()
    assert [r.project for r in _keys.live_keys()] == ["café"]


def test_a_bound_key_reads_its_project_in_any_normalisation_form() -> None:
    from fastapi import HTTPException

    from particles.api import app as app_mod
    from particles.api._keys import Role
    from particles.api.auth import Principal, _current_principal

    token = _current_principal.set(
        Principal(id="agent:ci-bot", role=Role.AGENT, key_id="k", project="café")
    )
    try:
        assert app_mod._bound_project(unicodedata.normalize("NFD", "café")) == "café"
        for body in ("café​", "cafe"):
            with pytest.raises(HTTPException) as info:
                app_mod._bound_project(body)
            assert info.value.status_code == 403
    finally:
        _current_principal.reset(token)


def test_a_project_key_body_holding_an_invisible_character_is_refused() -> None:
    from pydantic import ValidationError

    from particles.api.app import AgentTagRequest

    with pytest.raises(ValidationError):
        AgentTagRequest(particle_id="p", tags=["t"], project_key="acme​")
    decomposed = unicodedata.normalize("NFD", "café")
    req = AgentTagRequest(particle_id="p", tags=["t"], project_key=decomposed)
    assert req.project_key == "café"
    assert AgentTagRequest(particle_id="p", tags=["t"], project_key="​").project_key is None


def test_an_agent_deposit_tag_naming_a_lookalike_project_is_refused(store: Path) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    token, _ = _keys.mint(Role.AGENT, "agent:ci-bot")
    resp = _client().post(
        "/corpus/deposit/text",
        json={"text": "a note", "tags": ["project:acme​"]},
        headers=_bearer(token),
    )
    assert resp.status_code == 403, resp.text


# ---------------------------------------------------------------------------
# O47: refusals before the writer lock, and the useful press under it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "caller", ["agent", "harvester", "writer-label", "mcp", "operator", "legacy"]
)
def test_the_agent_retract_refusal_takes_no_writer_lock(
    store: Path, lock_log: list[str], caller: str
) -> None:
    token = _tokens()[caller]
    lock_log.clear()
    resp = _client().post(
        "/particles/retract", json={"particle_id": "x", "reason": "r"}, headers=_bearer(token)
    )
    assert resp.status_code == 403, resp.text
    assert lock_log == [], f"{caller}: the writer lock was taken before the 403"


@pytest.mark.parametrize("caller", ["operator", "legacy"])
def test_the_operator_retract_refusal_takes_no_writer_lock(
    store: Path, lock_log: list[str], caller: str
) -> None:
    token = _tokens()[caller]
    lock_log.clear()
    resp = _client().post("/particles/x/retract", json={"reason": "r"}, headers=_bearer(token))
    assert resp.status_code == 403, resp.text
    assert lock_log == []


_EXPOSURE = {
    "session_id": "s",
    "recorded_at": "2026-10-27T00:00:00Z",
    "source": "startup",
    "action": "full",
    "project_key": "other-project",
    "observer_scope_applied": True,
    "beliefs": [],
}


def test_a_refused_exposure_takes_no_writer_lock(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_log: list[str]
) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    _enable_belief_writes(tmp_path, monkeypatch)
    token, _ = _keys.mint(Role.AGENT, "agent:ci-bot", project="acme")
    lock_log.clear()
    resp = _client().post("/session-exposures", json=_EXPOSURE, headers=_bearer(token))
    assert resp.status_code == 403, resp.text
    assert lock_log == []


def test_a_malformed_exposure_body_takes_no_writer_lock(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_log: list[str]
) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    _enable_belief_writes(tmp_path, monkeypatch)
    token, _ = _keys.mint(Role.AGENT, "agent:ci-bot")
    lock_log.clear()
    resp = _client().post("/session-exposures", json={"session_id": 1}, headers=_bearer(token))
    assert resp.status_code == 422, resp.text
    assert lock_log == []


def test_an_accepted_exposure_commits_under_the_writer_lock(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_log: list[str]
) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    _enable_belief_writes(tmp_path, monkeypatch)
    token, _ = _keys.mint(Role.AGENT, "agent:ci-bot", project="acme")
    lock_log.clear()
    resp = _client().post(
        "/session-exposures", json={**_EXPOSURE, "project_key": "acme"}, headers=_bearer(token)
    )
    assert resp.status_code == 200, resp.text
    assert lock_log == ["default"]


_SIX_ROUTES = (
    (
        "/particles/supersede",
        {"supersedes_id": "x", "content": "c", "subject_names": ["S"], "confidence": 0.5},
    ),
    ("/particles/retract", {"particle_id": "x", "reason": "r"}),
    ("/particles/link", {"particle_a": "x", "particle_b": "y"}),
    ("/particles/unlink", {"particle_a": "x", "particle_b": "y"}),
    ("/particles/tag", {"particle_id": "x", "tags": ["t"]}),
    ("/particles/untag", {"particle_id": "x", "tags": ["t"]}),
)


def test_a_writer_label_refusal_takes_no_writer_lock(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_log: list[str]
) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    _enable_belief_writes(tmp_path, monkeypatch)
    token, _ = _keys.mint(Role.AGENT, "general-extractor", reserved_ok=True)
    took = []
    for path, body in _SIX_ROUTES:
        lock_log.clear()
        resp = _client().post(path, json=body, headers=_bearer(token))
        assert resp.status_code == 403, (path, resp.text)
        assert "store writer" in resp.text, (path, resp.text)
        if lock_log:
            took.append(path)
    assert took == []


def test_a_bound_keys_foreign_project_refusal_takes_no_writer_lock(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_log: list[str]
) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    _enable_belief_writes(tmp_path, monkeypatch)
    token, _ = _keys.mint(Role.AGENT, "agent:ci-bot", project="acme")
    took = []
    routes = [*_SIX_ROUTES[1:], ("/particles/deposit", {"text": "a note"})]
    for path, body in routes:
        lock_log.clear()
        resp = _client().post(path, json={**body, "project_key": "other"}, headers=_bearer(token))
        assert resp.status_code == 403, (path, resp.text)
        if lock_log:
            took.append(path)
    assert took == []


def test_a_useful_press_commits_under_the_writer_lock(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_log: list[str]
) -> None:
    import particles.operations.utility_feedback as uf

    _enable_belief_writes(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []

    async def _fake(session: Any, particle_id: str, **kwargs: Any) -> Any:
        calls.append({"particle_id": particle_id, **kwargs})
        return uf.MarkUsefulResult(
            particle_id=particle_id,
            credit_key="k" if kwargs.get("operator") else None,
            counted=bool(kwargs.get("operator")),
            event_id="e",
        )

    monkeypatch.setattr(uf, "mark_belief_useful", _fake)
    tokens = _tokens()
    for caller, operator in (("agent", False), ("operator", True), ("legacy", True)):
        lock_log.clear()
        calls.clear()
        resp = _client().post(
            "/memory/useful", json={"particle_id": "p-1"}, headers=_bearer(tokens[caller])
        )
        assert resp.status_code == 200, (caller, resp.text)
        assert calls[0]["operator"] is operator, caller
        assert lock_log == ["default"], caller
    lock_log.clear()
    resp = _client().post(
        "/memory/useful", json={"particle_id": "p-1"}, headers=_bearer(tokens["reader"])
    )
    assert resp.status_code == 403
    assert lock_log == []


# ---------------------------------------------------------------------------
# O48, O50, O79: the key file
# ---------------------------------------------------------------------------


@pytest.fixture
def per_client_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, Path], None, None]:
    """An engine on per-client keys only, with reads requiring a key."""
    from particles.api import _keys
    from particles.config import get_config, reset_config

    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    _write_config(tmp_path, monkeypatch, "api:\n  require_auth_for_reads: true\n")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'keys.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    assert get_config().api.require_auth_for_reads is True
    _create_tables()
    try:
        yield _client(), keydir
    finally:
        _keys.reset_key_cache()
        _keys.forget_key_files()
        reset_config()


def test_revoking_the_last_key_never_turns_a_running_engine_keyless(
    per_client_engine: tuple[TestClient, Path],
) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    engine, _ = per_client_engine
    token, record = _keys.mint(Role.READER, "viewer")
    assert engine.get("/events", headers=_bearer(token)).status_code == 200
    assert engine.get("/events").status_code == 401
    _keys.revoke(record.key_id)
    assert engine.get("/events").status_code in {401, 503}, "the engine turned keyless"
    assert engine.get("/events", headers=_bearer(token)).status_code == 401


def test_a_file_that_never_held_a_live_key_leaves_the_engine_keyless(
    per_client_engine: tuple[TestClient, Path],
) -> None:
    """Control: only a live key read by this process holds the engine keyed."""
    from particles.api import _keys
    from particles.api._keys import Role
    from particles.api.auth import engine_is_keyless

    _token, record = _keys.mint(Role.READER, "viewer")
    _keys.revoke(record.key_id)
    _keys.reset_key_cache()
    _keys.forget_key_files()
    assert engine_is_keyless()


def test_a_key_directory_others_may_write_into_is_refused(
    per_client_engine: tuple[TestClient, Path],
) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    engine, keydir = per_client_engine
    token, _ = _keys.mint(Role.READER, "viewer")
    for mode in (0o777, 0o770):
        os.chmod(keydir, mode)
        _keys.reset_key_cache()
        resp = engine.get("/events", headers=_bearer(token))
        assert resp.status_code == 503, oct(mode)
    os.chmod(keydir, 0o1777)  # sticky: only a file's owner may remove or rename it
    _keys.reset_key_cache()
    assert engine.get("/events", headers=_bearer(token)).status_code == 200
    os.chmod(keydir, 0o700)


def test_a_setgid_directory_of_a_group_only_the_engine_holds_is_allowed(
    per_client_engine: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape of a Kubernetes fsGroup volume: group-writable, setgid, a group no one else holds.

    The host's own groups are shared with other accounts (staff on macOS), so
    the test stands in a dedicated group by answering the membership question
    (security review 2026-10-28-2, AA49). The operator names the group in
    ``api.keys_dir_group``; unnamed, the shape is refused (2026-10-30-1, AB53).
    """
    from particles.api import _keys
    from particles.api._keys import Role
    from particles.config import get_config

    engine, keydir = per_client_engine
    token, _ = _keys.mint(Role.READER, "viewer")
    os.chown(keydir, -1, os.getegid())
    os.chmod(keydir, 0o2770)
    monkeypatch.setattr(_keys, "_group_shared_with_others", lambda gid: False)
    _keys.reset_key_cache()
    assert engine.get("/events", headers=_bearer(token)).status_code == 503
    monkeypatch.setattr(get_config().api, "keys_dir_group", os.getegid())
    _keys.reset_key_cache()
    assert engine.get("/events", headers=_bearer(token)).status_code == 200
    monkeypatch.setattr(_keys, "_group_shared_with_others", lambda gid: True)
    _keys.reset_key_cache()
    assert engine.get("/events", headers=_bearer(token)).status_code == 503
    os.chmod(keydir, 0o700)


def test_a_link_at_the_key_path_is_never_followed(
    per_client_engine: tuple[TestClient, Path], tmp_path: Path
) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    engine, keydir = per_client_engine
    token, _ = _keys.mint(Role.READER, "viewer")
    path = keydir / "engine-keys.json"
    elsewhere = tmp_path / "older-copy.json"
    elsewhere.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    os.chmod(elsewhere, 0o600)
    path.unlink()
    path.symlink_to(elsewhere)
    _keys.reset_key_cache()
    assert engine.get("/events", headers=_bearer(token)).status_code == 503


def test_a_hand_written_agent_record_under_the_operators_name_never_authenticates(
    per_client_engine: tuple[TestClient, Path],
) -> None:
    from particles.api import _keys
    from particles.api._keys import KeyRuleError, Role
    from particles.config import get_config

    engine, keydir = per_client_engine
    operator = get_config().curation.operator_identity
    with pytest.raises(KeyRuleError):
        _keys.mint(Role.AGENT, operator, reserved_ok=True)
    token, record = _keys.mint(Role.AGENT, "agent:ci-bot")
    path = keydir / "engine-keys.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data["keys"]:
        if item["key_id"] == record.key_id:
            item["principal"] = operator
            item["reserved_ok"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(path, 0o600)
    _keys.reset_key_cache()
    assert _keys.lookup(token) is None
    assert engine.get("/auth/whoami", headers=_bearer(token)).status_code in {401, 503}


# ---------------------------------------------------------------------------
# O49: every forwarded field line
# ---------------------------------------------------------------------------

_PROXY = "10.0.0.5"


def _forwarded(lines: list[str]) -> Request:
    headers = [(b"host", b"127.0.0.1:8000")]
    headers += [(b"x-forwarded-for", line.encode()) for line in lines]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/particles",
            "headers": headers,
            "client": (_PROXY, 40000),
            "server": ("10.0.0.2", 8000),
            "query_string": b"",
        }
    )


@pytest.fixture
def behind_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from particles.config import get_config, reset_config

    _write_config(tmp_path, monkeypatch, f"api:\n  trusted_proxies: ['{_PROXY}']\n")
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(tmp_path / "engine-keys.json"))
    reset_config()
    assert get_config().api.trusted_proxies == [_PROXY]


def test_a_second_forwarded_line_written_by_the_proxy_names_the_client(
    behind_proxy: None,
) -> None:
    from particles.api.auth import _client_is_loopback, _real_client_host

    request = _forwarded(["127.0.0.1", "203.0.113.7"])
    assert _real_client_host(request) == "203.0.113.7"
    assert not _client_is_loopback(request)


def test_a_keyless_engine_behind_a_proxy_refuses_a_remote_client_whatever_lines_it_sends(
    behind_proxy: None,
) -> None:
    from particles.api.auth import request_bound_refusal

    assert request_bound_refusal(_forwarded(["203.0.113.7"])) is not None
    assert request_bound_refusal(_forwarded(["127.0.0.1", "203.0.113.7"])) is not None


def test_a_remote_client_cannot_pick_its_rate_limit_bucket(behind_proxy: None) -> None:
    from particles.api._rate_limit import _client_bucket

    one = _client_bucket(_forwarded(["198.51.100.1", "203.0.113.7"]))
    two = _client_bucket(_forwarded(["198.51.100.2", "203.0.113.7"]))
    assert one == two


# ---------------------------------------------------------------------------
# O51: the monitor reads only trusted exposure rows
# ---------------------------------------------------------------------------


def test_the_rank_lift_monitor_ignores_another_agents_rows(store: Path) -> None:
    async def _go() -> tuple[dict[str, set[str]], set[str]]:
        from particles.db import get_engine, session_scope
        from particles.operations.rank_lift_monitor import _shown_by_session, live_project_keys
        from particles.store.session_exposure_store import (
            SessionExposure,
            ShownBelief,
            record_session_exposure,
        )

        def _exposure(project: str, belief: str) -> SessionExposure:
            return SessionExposure(
                session_id="sess-1",
                recorded_at=datetime.now(UTC),
                source="startup",
                action="full",
                project_key=project,
                observer_scope_applied=True,
                beliefs=[ShownBelief(particle_id=belief, shown_as="digest")],
            )

        async with session_scope() as s:
            await record_session_exposure(
                s, _exposure("chosen-by-agent", "p-agent-pick"), recorded_by="agent:ci-bot"
            )
            await record_session_exposure(
                s, _exposure("harvested", "p-harvested"), recorded_by="claude-code-hook"
            )
            await record_session_exposure(s, _exposure("in-process", "p-local"))
            await s.commit()
        async with session_scope() as s:
            shown = await _shown_by_session(s, {"sess-1"})
            keys = await live_project_keys(s, datetime.now(UTC) - timedelta(days=1))
        await get_engine().dispose()
        return shown, keys

    shown, keys = asyncio.run(_go())
    assert shown == {"sess-1": {"harvested", "local"}}  # stored without the p- prefix
    assert keys == {"harvested", "in-process"}


# ---------------------------------------------------------------------------
# O80: a refused follow is reported
# ---------------------------------------------------------------------------


def test_the_deposit_response_can_report_a_refused_follow() -> None:
    from particles.api.app import DepositResponse

    assert DepositResponse(entry_id="e", snapshot_id="s").follow_refused == []
    assert "follow_refused" in DepositResponse.model_fields


def test_a_refused_follow_reaches_the_callers_record(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import particles.corpus.deposit as dep
    import particles.corpus.store as corpus_store
    from particles.corpus.deposit import ForeignEntryRefused

    class _Importer:
        def primary_url(self, content: bytes) -> str:
            return "https://example.org/held-by-another"

    async def _refusing(*args: Any, **kwargs: Any) -> tuple[str, str]:
        raise ForeignEntryRefused("another principal's entry")

    async def _entry(session: Any, entry_id: str) -> Any:
        return SimpleNamespace(
            snapshots=[SimpleNamespace(snapshot_id="snap-1", content_hash="0" * 64)]
        )

    @contextlib.asynccontextmanager
    async def _unit(session: Any) -> AsyncIterator[None]:
        yield

    monkeypatch.setattr(dep, "deposit_url", _refusing)
    monkeypatch.setattr(dep, "load_blob", lambda content_hash: b"{}")
    monkeypatch.setattr(dep, "_follow_unit", _unit)
    monkeypatch.setattr(dep.egress, "allows_url", lambda url: True)
    monkeypatch.setattr(corpus_store, "get_entry", _entry)
    refused: list[str] = []
    targets: list[tuple[str, str, str]] = []

    async def _go() -> None:
        await dep._maybe_follow_primary_url(
            None,  # type: ignore[arg-type]
            _Importer(),
            entry_id="entry-1",
            snapshot_id="snap-1",
            deposited_by="agent:ci-bot",
            tags=[],
            follow_post_links=True,
            out_follow_targets=targets,
            out_follow_refused=refused,
        )

    asyncio.run(_go())
    assert refused == ["https://example.org/held-by-another"]
    assert targets == []
