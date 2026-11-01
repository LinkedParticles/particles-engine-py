# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-client engine keys with roles.

The key file (``particles/api/_keys.py``), principal resolution and the role
gates (``particles/api/auth.py``), and the provenance an agent key's writes
record (``particles/api/app.py``). Each test pins one property of the record's
test plan; the legacy ``PARTICLES_API_KEY`` is the operator key the rest of the
suite already runs under (``tests/_engine_client.py``).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from particles.api import _keys
from particles.api._keys import KeyRuleError, Role
from tests._engine_client import TEST_ENGINE_KEY, keyed_engine_client
from tests.test_write_route_gate import BRANCH_WRITERS, _first_gate, _routes

AGENT = "agent:laptop-mcp"
OTHER_AGENT = "agent:ci-bot"
HOOK = "claude-code-hook"


@pytest.fixture
def keys_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """A key file under this test's own ``tmp_path`` (the suite's default is never created)."""
    from particles.config import reset_config

    path = tmp_path / "engine-keys.json"
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(path))
    reset_config()
    _keys.reset_key_cache()
    yield path
    _keys.reset_key_cache()
    _keys.forget_key_files()


@pytest.fixture
def engine(
    cli_db: Path, keys_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    """A keyed engine, belief writes on the default store; the legacy key is the default bearer."""
    import particles.ingest.subject_resolver as sr
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.db import DEFAULT_STORE

    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield keyed_engine_client(monkeypatch)
    finally:
        ep.set_embedding_model(original)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


#: The client directories the harvester keys these tests mint record,
#: matching the ``file:///home/u/…`` URIs the tests send.
HOOK_ROOTS: dict[str, Any] = {
    "memory_roots": ["/home/u/.claude/projects"],
    "archive_roots": ["/home/u/.particles/claude-code/projects"],
}


def _mint(role: str, principal: str, **kwargs: Any) -> str:
    token, _record = _keys.mint(Role(role), principal, **kwargs)
    return token


def _entries() -> list[Any]:
    async def _go() -> list[Any]:
        from particles.corpus.store import list_entries
        from particles.db import get_engine, session_scope

        try:
            async with session_scope() as s:
                return list(await list_entries(s))
        finally:
            await get_engine().dispose()

    return asyncio.run(_go())


# --- the key file ------------------------------------------------------------


class TestMintingRules:
    """rule 5 and §4: the keys `keys create` refuses."""

    @pytest.mark.parametrize(
        ("role", "principal", "kwargs"),
        [
            (Role.OPERATOR, "operator:phone", {}),
            (Role.AGENT, "operator:local", {}),
            (Role.AGENT, "operator", {}),
            (Role.READER, "api-user", {}),
            (Role.AGENT, HOOK, {}),
            (Role.AGENT, "mcp:memory-compat", {}),
            (Role.READER, "viewer", {"project": "repo"}),
            (Role.AGENT, AGENT, {"project": "  "}),
            (Role.AGENT, "  ", {}),
        ],
    )
    def test_refused(
        self, keys_file: Path, role: Role, principal: str, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(KeyRuleError):
            _keys.mint(role, principal, **kwargs)
        assert not keys_file.exists()

    def test_reserved_name_needs_the_acknowledgement(self, keys_file: Path) -> None:
        _keys.mint(Role.AGENT, HOOK, reserved_ok=True)
        assert [r.principal for r in _keys.live_keys()] == [HOOK]

    def test_operator_key_carries_the_operator_identity(self, keys_file: Path) -> None:
        _token, record = _keys.mint(Role.OPERATOR, "operator:local")
        assert record.principal == "operator:local"


class TestKeyFile:
    def test_no_secret_at_rest(self, keys_file: Path) -> None:
        token = _mint("agent", AGENT)
        secret = token.rsplit("_", 1)[1]
        text = keys_file.read_text()
        assert secret not in text
        assert token not in text
        assert oct(keys_file.stat().st_mode & 0o777) == "0o600"

    def test_token_round_trips_and_a_wrong_secret_does_not(self, keys_file: Path) -> None:
        token = _mint("agent", AGENT)
        record = _keys.lookup(token)
        assert record is not None and record.principal == AGENT
        forged = token[:-1] + ("A" if token[-1] != "A" else "B")
        assert _keys.lookup(forged) is None

    def test_a_prefix_that_disagrees_with_the_record_is_refused(self, keys_file: Path) -> None:
        token = _mint("agent", AGENT)
        assert _keys.lookup(token.replace("ptk_agent_", "ptk_op_", 1)) is None

    def test_revocation_is_seen_without_a_restart(self, keys_file: Path) -> None:
        token, record = _keys.mint(Role.AGENT, AGENT)
        assert _keys.lookup(token) is not None
        _keys.revoke(record.key_id)
        assert _keys.lookup(token) is None
        assert not _keys.all_keys()[record.key_id].live

    def test_an_in_place_edit_is_seen_on_the_next_lookup(self, keys_file: Path) -> None:
        token, record = _keys.mint(Role.AGENT, AGENT)
        assert _keys.lookup(token) is not None
        st = keys_file.stat()
        # A hand edit that keeps the inode, the size and the mtime; the change
        # time is in the cache's stamp (security review 2026-10-27-1, G76).
        text = keys_file.read_text().replace('"revoked_at": null', '"revoked_at": "1" ')
        assert len(text) == len(keys_file.read_text())
        with keys_file.open("r+") as handle:
            handle.write(text)
        os.utime(keys_file, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert _keys.lookup(token) is None

    def test_an_unsafe_file_is_unusable_and_never_keyless(self, keys_file: Path) -> None:
        from particles.api.auth import engine_is_keyless

        _mint("agent", AGENT)
        _keys.revoke(next(iter(_keys.all_keys())))
        # Every key revoked, then a restart: keyless (a running engine stays
        # keyed, security review 2026-10-28-1, O48).
        _keys.forget_key_files()
        assert engine_is_keyless()
        keys_file.chmod(0o666)
        _keys.reset_key_cache()
        with pytest.raises(_keys.KeyFileUnusable):
            _keys.live_keys()
        assert not engine_is_keyless()

    def test_a_corrupt_file_is_unusable(self, keys_file: Path) -> None:
        keys_file.write_text("{not json")
        keys_file.chmod(0o600)
        with pytest.raises(_keys.KeyFileUnusable):
            _keys.live_keys()


# --- roles by route ----------------------------------------------------------


def _gated(kind: str) -> list[tuple[str, str]]:
    return [(m, p) for m, p, route in _routes() if _first_gate(route) == kind]


class TestRolesByRoute:
    """§9's table: who may call which class of route."""

    def test_an_agent_key_is_refused_on_every_operator_route(self, engine: TestClient) -> None:
        token = _mint("agent", AGENT)
        wrong = []
        for method, path in _gated("operator"):
            url = path.replace("{", "x").replace("}", "")
            resp = engine.request(method, url, json={}, headers=_bearer(token))
            if resp.status_code != 403:
                wrong.append((method, path, resp.status_code))
        assert wrong == []

    def test_an_agent_key_is_refused_on_every_branch_writers_write_branch(
        self, engine: TestClient
    ) -> None:
        token = _mint("agent", AGENT)
        wrong = []
        for (method, path), body in BRANCH_WRITERS.items():
            url = path.replace("{entry_id}", "e-missing")
            resp = engine.request(method, url, json=body, headers=_bearer(token))
            if resp.status_code != 403:
                wrong.append((method, path, resp.status_code, resp.text[:120]))
        assert wrong == []

    def test_a_reader_key_is_refused_on_every_agent_write(self, engine: TestClient) -> None:
        token = _mint("reader", "viewer")
        wrong = []
        for method, path in _gated("write"):
            resp = engine.request(method, path, json={}, headers=_bearer(token))
            if resp.status_code != 403:
                wrong.append((method, path, resp.status_code))
        assert wrong == []

    def test_the_refusal_names_the_key_and_what_it_needs(self, engine: TestClient) -> None:
        token = _mint("agent", AGENT)
        resp = engine.post("/subjects", json={}, headers=_bearer(token))
        assert resp.status_code == 403
        assert f"agent, principal {AGENT}" in resp.json()["detail"]
        assert "operator key" in resp.json()["detail"]

    def test_every_role_reads(self, engine: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import get_config

        get_config().api.require_auth_for_reads = True
        for role, principal in (("reader", "viewer"), ("agent", AGENT)):
            token = _mint(role, principal)
            assert engine.get("/subjects", headers=_bearer(token)).status_code == 200
            assert engine.get("/events", headers=_bearer(token)).status_code == 200
        bare = TestClient(engine.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 1))
        assert bare.get("/subjects").status_code == 401

    def test_an_unknown_and_a_revoked_key_get_the_same_401(self, engine: TestClient) -> None:
        token, record = _keys.mint(Role.AGENT, AGENT)
        unknown = "ptk_agent_000000000000_" + "A" * 43
        assert engine.post("/session-exposures", json={}, headers=_bearer(token)).status_code != 401
        _keys.revoke(record.key_id)
        revoked = engine.get("/events", headers=_bearer(token))
        never = engine.get("/events", headers=_bearer(unknown))
        assert revoked.status_code == never.status_code == 401
        assert revoked.json() == never.json()

    def test_an_unusable_key_file_refuses_per_client_keys_and_keeps_the_legacy_key(
        self, engine: TestClient, keys_file: Path
    ) -> None:
        token = _mint("agent", AGENT)
        # The engine starts on a usable file (a request runs its serving checks),
        # and the file goes bad while it serves.
        assert engine.get("/events", headers=_bearer(token)).status_code == 200
        keys_file.chmod(0o666)
        _keys.reset_key_cache()
        resp = engine.get("/events", headers=_bearer(token))
        assert resp.status_code == 503
        assert "PARTICLES_API_KEY still works" in resp.json()["detail"]
        assert engine.get("/events").status_code == 200  # the legacy bearer

    def test_a_file_of_revoked_keys_and_no_legacy_key_is_keyless(
        self, engine: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.auth import KEYLESS_WRITE_REFUSAL

        token, record = _keys.mint(Role.AGENT, AGENT)
        _keys.revoke(record.key_id)
        monkeypatch.delenv("PARTICLES_API_KEY")
        # Keyless after a restart; a running engine stays keyed (O48).
        _keys.forget_key_files()
        resp = engine.post("/session-exposures", json={}, headers=_bearer(token))
        assert resp.status_code == 503
        assert resp.json()["detail"] == KEYLESS_WRITE_REFUSAL

    def test_an_engine_keyed_by_the_file_alone_accepts_an_agent_write(
        self, engine: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = _mint("agent", AGENT)
        monkeypatch.delenv("PARTICLES_API_KEY")
        resp = engine.post(
            "/corpus/deposit/text", json={"text": "Keyed by the file."}, headers=_bearer(token)
        )
        assert resp.status_code == 200, resp.text


# --- provenance from the key -------------------------------------------------


class TestStampedProvenance:
    """an agent key's deposits are recorded as its principal."""

    def test_an_absent_depositor_is_the_principal(self, engine: TestClient) -> None:
        token = _mint("agent", AGENT)
        resp = engine.post(
            "/corpus/deposit/text", json={"text": "Stamped."}, headers=_bearer(token)
        )
        assert resp.status_code == 200, resp.text
        (entry,) = [e for e in _entries() if e.entry_id == resp.json()["entry_id"]]
        assert entry.deposited_by == AGENT

    @pytest.mark.parametrize(
        "body",
        [
            {"deposited_by": "operator"},
            {"deposited_by": "operator:local"},
            {"deposited_by": "api-user"},
            {"author_id": "someone-else"},
            {"deposited_by": AGENT, "author_id": "operator:local"},
        ],
    )
    def test_naming_anyone_else_is_refused_and_stores_nothing(
        self, engine: TestClient, body: dict[str, str]
    ) -> None:
        token = _mint("agent", AGENT)
        before = len(_entries())
        resp = engine.post(
            "/corpus/deposit/text",
            json={"text": "A planted attribution.", **body},
            headers=_bearer(token),
        )
        assert resp.status_code == 403, resp.text
        assert len(_entries()) == before

    def test_an_operator_key_with_no_depositor_still_records_api_user(
        self, engine: TestClient
    ) -> None:
        resp = engine.post("/corpus/deposit/text", json={"text": "From the operator."})
        assert resp.status_code == 200, resp.text
        (entry,) = [e for e in _entries() if e.entry_id == resp.json()["entry_id"]]
        assert entry.deposited_by == "api-user"

    @pytest.mark.parametrize(
        "uri", ["https://example.org/page", "file:///etc/hosts", "claude-code://session/x"]
    )
    def test_an_ordinary_agent_names_no_uri(self, engine: TestClient, uri: str) -> None:
        token = _mint("agent", AGENT)
        resp = engine.post(
            "/corpus/deposit/text",
            json={"text": "x", "uri_r": uri, "mutability": "STABLE"},
            headers=_bearer(token),
        )
        assert resp.status_code == 403

    def test_a_harvester_names_its_two_schemes_and_no_web_uri(self, engine: TestClient) -> None:
        token = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)
        ok = engine.post(
            "/corpus/deposit/text",
            json={
                "text": "A session.",
                "uri_r": "claude-code://session/abc",
                "mutability": "APPEND_ONLY",
                "tags": ["claude-code", "memory-file"],
            },
            headers=_bearer(token),
        )
        assert ok.status_code == 200, ok.text
        web = engine.post(
            "/corpus/deposit/text",
            json={"text": "x", "uri_r": "https://example.org/", "mutability": "STABLE"},
            headers=_bearer(token),
        )
        assert web.status_code == 403
        memory = engine.post(
            "/corpus/deposit/text",
            json={
                "text": "# Memory",
                "uri_r": "file:///home/u/.claude/projects/p/memory/MEMORY.md",
                "source_type": "LOCAL_MARKDOWN",
                "mutability": "MUTABLE",
            },
            headers=_bearer(token),
        )
        assert memory.status_code == 200, memory.text

    @pytest.mark.parametrize("source_type", ["WEB_PAGE", "ACADEMIC_PAPER", "LOCAL_MARKDOWN"])
    def test_an_agent_names_no_source_type_of_its_own(
        self, engine: TestClient, source_type: str
    ) -> None:
        token = _mint("agent", AGENT)
        resp = engine.post(
            "/corpus/deposit/text",
            json={"text": "x", "source_type": source_type},
            headers=_bearer(token),
        )
        assert resp.status_code == 403

    def test_the_harvest_tags_need_a_harvester(self, engine: TestClient) -> None:
        token = _mint("agent", AGENT)
        resp = engine.post(
            "/corpus/deposit/text",
            json={"text": "x", "tags": ["memory-file"]},
            headers=_bearer(token),
        )
        assert resp.status_code == 403

    def test_an_upload_typed_unlike_its_bytes_is_refused(self, engine: TestClient) -> None:
        token = _mint("agent", AGENT)
        resp = engine.post(
            "/corpus/deposit/file",
            files={"file": ("notes.md", b"# notes\n", "text/markdown")},
            data={"source_type": "ACADEMIC_PAPER"},
            headers=_bearer(token),
        )
        assert resp.status_code == 403


class TestOwnEntriesOnly:
    """rule 3: an agent's deposit never writes another principal's entry."""

    def test_matching_bytes_get_an_entry_of_the_agents_own(self, engine: TestClient) -> None:
        text = "Bytes the operator deposited first."
        first = engine.post("/corpus/deposit/text", json={"text": text, "deposited_by": "operator"})
        assert first.status_code == 200, first.text
        token = _mint("agent", AGENT)
        second = engine.post("/corpus/deposit/text", json={"text": text}, headers=_bearer(token))
        assert second.status_code == 200, second.text
        assert second.json()["entry_id"] != first.json()["entry_id"]

    def test_a_uri_another_principal_holds_answers_409_and_writes_nothing(
        self, engine: TestClient
    ) -> None:
        uri = "claude-code://session/held"
        held = engine.post(
            "/corpus/deposit/text",
            json={"text": "v1", "uri_r": uri, "mutability": "APPEND_ONLY", "deposited_by": "x"},
        )
        assert held.status_code == 200, held.text
        before = len(_entries())
        token = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)
        resp = engine.post(
            "/corpus/deposit/text",
            json={"text": "v2", "uri_r": uri, "mutability": "APPEND_ONLY"},
            headers=_bearer(token),
        )
        assert resp.status_code == 409, resp.text
        assert len(_entries()) == before

    def test_the_two_harvester_depositors_count_as_one_owner(self, engine: TestClient) -> None:
        uri = "file:///home/u/.claude/projects/p/memory/MEMORY.md"
        audit = engine.post(
            "/corpus/deposit/text",
            json={
                "text": "# v1",
                "uri_r": uri,
                "source_type": "LOCAL_MARKDOWN",
                "mutability": "MUTABLE",
                "deposited_by": "audit",
            },
        )
        assert audit.status_code == 200, audit.text
        token = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)
        resp = engine.post(
            "/corpus/deposit/text",
            json={
                "text": "# v2",
                "uri_r": uri,
                "source_type": "LOCAL_MARKDOWN",
                "mutability": "MUTABLE",
            },
            headers=_bearer(token),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["entry_id"] == audit.json()["entry_id"]


class TestAgentRoutes:
    """rule 2 and rule 6: the asserter and the event actor are the key's."""

    def _assert(self, engine: TestClient, token: str, content: str) -> str:
        resp = engine.post(
            "/particles/assert",
            json={
                "content": content,
                "subject_names": ["release train"],
                "confidence": 0.8,
                "source_excerpt": content,
            },
            headers=_bearer(token),
        )
        assert resp.status_code == 200, resp.text
        return str(resp.json()["asserted_particle_id"])

    def test_an_agent_asserts_as_its_principal_and_its_event_names_the_key(
        self, engine: TestClient
    ) -> None:
        token, record = _keys.mint(Role.AGENT, OTHER_AGENT)
        pid = self._assert(engine, token, "The release train leaves on Tuesdays.")
        assert engine.get(f"/particles/{pid}").json()["asserted_by"] == OTHER_AGENT
        events = engine.get("/events", params={"key": record.key_id}).json()
        assert events, "the assertion's event is listed under its key"
        assert all(e["payload"]["http_principal"] == OTHER_AGENT for e in events)
        assert all(e["payload"]["http_key_id"] == record.key_id for e in events)
        assert engine.get("/events", params={"key": "000000000000"}).json() == []

    def test_one_agent_cannot_retract_anothers_belief(self, engine: TestClient) -> None:
        mine = _mint("agent", OTHER_AGENT)
        theirs = _mint("agent", AGENT)
        pid = self._assert(engine, mine, "The build runs nightly.")
        resp = engine.post(
            "/particles/retract",
            json={"particle_id": pid, "reason": "not yours"},
            headers=_bearer(theirs),
        )
        assert resp.status_code == 400, resp.text


class TestRateBuckets:
    """an authenticated request spends from its principal's bucket."""

    def test_two_keys_of_one_principal_share_a_bucket_and_principals_do_not(
        self, engine: TestClient
    ) -> None:
        from particles.api._rate_limit import reset_rate_limiter
        from particles.config import get_config

        get_config().api.rate_limit_per_minute = 1
        reset_rate_limiter()
        first, second = _mint("agent", AGENT), _mint("agent", AGENT)
        other = _mint("agent", OTHER_AGENT)

        def deposit(token: str, text: str) -> int:
            return engine.post(
                "/particles/deposit", json={"text": text}, headers=_bearer(token)
            ).status_code

        assert deposit(first, "one") == 200
        assert deposit(second, "two") == 429
        assert deposit(other, "three") == 200

    def test_each_generic_deposit_route_spends_from_the_principals_bucket(
        self, engine: TestClient
    ) -> None:
        """The activation gate's R25 half: the three generic deposits are rate-limited too."""
        from particles.api._rate_limit import reset_rate_limiter
        from particles.config import get_config

        get_config().api.rate_limit_per_minute = 1
        token = _mint("agent", AGENT)
        attempts = {
            "text": lambda: engine.post(
                "/corpus/deposit/text", json={"text": "t"}, headers=_bearer(token)
            ),
            "file": lambda: engine.post(
                "/corpus/deposit/file",
                files={"file": ("n.md", b"# n\n", "text/markdown")},
                headers=_bearer(token),
            ),
            "url": lambda: engine.post(
                "/corpus/deposit/url", json={"url": "https://example.org/"}, headers=_bearer(token)
            ),
        }
        for name, attempt in attempts.items():
            reset_rate_limiter()
            # Spend the principal's one token on a deposit it may make, then try the route.
            spent = engine.post("/particles/deposit", json={"text": name}, headers=_bearer(token))
            assert spent.status_code == 200, spent.text
            assert attempt().status_code == 429, name

    def test_bad_keys_never_touch_a_principals_bucket(self, engine: TestClient) -> None:
        from particles.api._rate_limit import reset_rate_limiter
        from particles.config import get_config

        get_config().api.rate_limit_per_minute = 1
        reset_rate_limiter()
        good = _mint("agent", AGENT)
        for i in range(3):
            bad = f"ptk_agent_{i:012x}_" + "A" * 43
            engine.post("/particles/deposit", json={"text": "x"}, headers=_bearer(bad))
        resp = engine.post("/particles/deposit", json={"text": "y"}, headers=_bearer(good))
        assert resp.status_code == 200, resp.text


# --- startup and the CLI -----------------------------------------------------


class TestStartup:
    def test_a_legacy_key_with_the_per_client_prefix_is_refused(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.auth import enforce_key_material_on_startup

        monkeypatch.setenv("PARTICLES_API_KEY", "ptk_op_not_really")
        with pytest.raises(RuntimeError, match="ptk_"):
            enforce_key_material_on_startup()

    def test_an_unusable_key_file_stops_the_engine(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.auth import enforce_key_material_on_startup

        monkeypatch.setenv("PARTICLES_API_KEY", TEST_ENGINE_KEY)
        _mint("agent", AGENT)
        keys_file.chmod(0o666)
        _keys.reset_key_cache()
        with pytest.raises(RuntimeError, match="unusable"):
            enforce_key_material_on_startup()


class TestKeysCli:
    def _run(self, *args: str, claude_code: bool = False) -> Any:
        from particles.api.cli import app

        # The suite may itself run in a Claude Code shell; each test says which
        # shell it models.
        env = {"CLAUDECODE": "1" if claude_code else None}
        return CliRunner().invoke(app, ["engine", "keys", *args], env=env)

    @staticmethod
    def _written_token(keys_file: Path, output: str) -> str:
        line = next(line for line in output.splitlines() if line.startswith("Wrote the key to "))
        path = Path(line.removeprefix("Wrote the key to ").split(" (mode 0600)")[0])
        assert path.parent == keys_file.parent
        assert path.stat().st_mode & 0o777 == 0o600
        return path.read_text(encoding="utf-8").strip()

    def test_a_claude_code_shell_gets_no_token(self, keys_file: Path) -> None:
        """Security review 2026-10-26-1, W1: a minted key never reaches a session transcript."""
        result = self._run("create", "--role", "agent", "--principal", AGENT, claude_code=True)
        assert result.exit_code == 0, result.output
        assert "ptk_" not in result.output
        assert self._written_token(keys_file, result.output).startswith("ptk_agent_")

    def test_keys_create_never_prints_a_key_without_print_token(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Security review 2026-10-28-1, O45, and 2026-10-28-2, AA45: any harness that keeps stdout.

        A harness that runs its shell on a pseudo-terminal looks like a person's
        terminal, so the key never goes to standard output by default.
        """
        monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")  # another harness's shell marker
        result = self._run("create", "--role", "agent", "--principal", AGENT)
        assert result.exit_code == 0, result.output
        assert "ptk_agent_" not in result.output
        token = self._written_token(keys_file, result.output)
        assert _keys.lookup(token) is not None

    def test_output_writes_the_key_to_a_new_private_file(
        self, keys_file: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "phone.key"
        result = self._run(
            "create", "--role", "agent", "--principal", AGENT, "--output", str(target)
        )
        assert result.exit_code == 0, result.output
        assert "ptk_" not in result.output
        assert target.stat().st_mode & 0o777 == 0o600
        assert _keys.lookup(target.read_text(encoding="utf-8").strip()) is not None

    def test_output_never_writes_over_an_existing_file_or_mints(
        self, keys_file: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "taken.key"
        target.write_text("already here", encoding="utf-8")
        result = self._run(
            "create", "--role", "agent", "--principal", AGENT, "--output", str(target)
        )
        assert result.exit_code == 1
        assert target.read_text(encoding="utf-8") == "already here"
        assert not keys_file.exists()

    def test_print_token_prints_it(self, keys_file: Path) -> None:
        result = self._run("create", "--role", "agent", "--principal", AGENT, "--print-token")
        assert result.exit_code == 0, result.output
        assert any(line.startswith("ptk_agent_") for line in result.output.splitlines())
        assert not list(keys_file.parent.glob("*.key"))

    def test_print_token_prints_it_in_a_claude_code_shell(self, keys_file: Path) -> None:
        result = self._run(
            "create", "--role", "agent", "--principal", AGENT, "--print-token", claude_code=True
        )
        assert result.exit_code == 0, result.output
        assert any(line.startswith("ptk_agent_") for line in result.output.splitlines())

    def test_create_list_revoke(self, keys_file: Path) -> None:
        created = self._run("create", "--role", "agent", "--principal", AGENT, "--label", "laptop")
        assert created.exit_code == 0, created.output
        token = self._written_token(keys_file, created.output)
        key_id = token.split("_")[2]
        listed = self._run("list")
        assert key_id in listed.output and "laptop" in listed.output
        assert _keys.all_keys()[key_id].digest not in listed.output
        revoked = self._run("revoke", key_id)
        assert revoked.exit_code == 0, revoked.output
        assert "revoked" in self._run("list").output

    def test_the_mcp_asserter_identity_needs_no_acknowledgement(self, keys_file: Path) -> None:
        """The HTTP MCP server's key shares the in-process server's principal."""
        from particles.config import get_config

        principal = get_config().mcp.write.asserter_identity
        assert principal == "mcp:claude-code"
        created = self._run("create", "--role", "agent", "--principal", principal)
        assert created.exit_code == 0, created.output
        assert [r.principal for r in _keys.live_keys()] == [principal]

    def test_an_extractor_id_still_needs_the_acknowledgement(self, keys_file: Path) -> None:
        refused = self._run("create", "--role", "agent", "--principal", "general-extractor")
        assert refused.exit_code == 1
        assert not keys_file.exists()
        created = self._run(
            "create", "--role", "agent", "--principal", "general-extractor", "--reserved-principal"
        )
        assert created.exit_code == 0, created.output

    def test_a_refused_key_exits_1(self, keys_file: Path) -> None:
        result = self._run("create", "--role", "operator", "--principal", "operator:phone")
        assert result.exit_code == 1
        assert "operator principal" in result.output

    def test_a_reader_key_says_it_reads_everything(self, keys_file: Path) -> None:
        result = self._run("create", "--role", "reader", "--principal", "viewer")
        assert result.exit_code == 0, result.output
        assert "not confidential" in result.output

    def test_refused_on_a_thin_client(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://127.0.0.1:9")
        reset_config()
        result = self._run("list")
        assert result.exit_code == 1


def test_route_walk_still_sees_four_gates() -> None:
    """Every gated route names one of the four async gates."""
    from particles.api.app import app

    kinds = {
        _first_gate(route)
        for route in app.routes
        if isinstance(route, APIRoute) and route.dependant.dependencies
    }
    assert {"read", "auth", "write", "operator"} <= kinds


# --- whoami, the warnings, and the callers that must not break ----


class TestWhoAmI:
    def test_each_key_says_who_it_is(self, engine: TestClient) -> None:
        token, record = _keys.mint(Role.AGENT, AGENT, project="repo")
        body = engine.get("/auth/whoami", headers=_bearer(token)).json()
        assert body == {
            "anonymous": False,
            "principal": AGENT,
            "role": "agent",
            "key_id": record.key_id,
            "project": "repo",
            "memory_roots": None,
            "archive_roots": None,
        }
        legacy = engine.get("/auth/whoami").json()
        assert legacy["role"] == "operator" and legacy["key_id"] == "env"

    def test_a_keyed_engine_wants_a_key(self, engine: TestClient) -> None:
        bare = TestClient(engine.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 1))
        assert bare.get("/auth/whoami").status_code == 401


class TestTokenWarnings:
    @pytest.mark.parametrize(
        ("token", "legacy", "agent_side", "warns"),
        [
            (None, None, True, False),
            ("not-a-per-client-key", None, False, True),
            ("same-as-legacy", "same-as-legacy", False, True),
            ("ptk_op_0123456789ab_" + "A" * 43, None, False, False),
            ("ptk_op_0123456789ab_" + "A" * 43, None, True, True),
            ("ptk_agent_0123456789ab_" + "A" * 43, None, True, False),
            ("ptk_read_0123456789ab_" + "A" * 43, None, True, False),
        ],
    )
    def test_client_token_warning(
        self, token: str | None, legacy: str | None, agent_side: bool, warns: bool
    ) -> None:
        warning = _keys.client_token_warning(token, legacy, agent_side=agent_side)
        assert (warning is not None) is warns

    def test_the_posture_line_counts_the_keys(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.auth import http_write_posture

        monkeypatch.setenv("PARTICLES_API_KEY", TEST_ENGINE_KEY)
        _mint("agent", AGENT)
        _mint("reader", "viewer")
        line = http_write_posture()
        assert "1 reader, 1 agent, 0 operator" in line
        assert "legacy PARTICLES_API_KEY (operator)" in line

    def test_the_doctor_warns_on_an_operator_key_in_the_hooks(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli.hook import _engine_line
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://engine.test")
        reset_config()
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "ptk_op_0123456789ab_" + "A" * 43)
        assert "⚠" in _engine_line() and "operator key" in _engine_line()
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "ptk_agent_0123456789ab_" + "A" * 43)
        assert "⚠" not in _engine_line()
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "the-old-shared-key")
        assert "legacy shared key" in _engine_line()

    def test_config_validate_names_the_client_token(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli.config import _client_token_lines
        from particles.config import get_config, reset_config

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://engine.test")
        reset_config()
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "ptk_read_0123456789ab_" + "A" * 43)
        assert _client_token_lines(get_config()) == ["client token: a per-client reader key"]
        monkeypatch.setenv("PARTICLES_API_KEY", "shared")
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "shared")
        (line,) = _client_token_lines(get_config())
        assert line.startswith("client token: ⚠")

    def test_the_mcp_server_warns_on_an_operator_key(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.config import reset_config
        from particles.mcp.server import _warn_on_an_operator_token

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://engine.test")
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "legacy-shared")
        reset_config()
        with caplog.at_level("WARNING"):
            _warn_on_an_operator_token()
        assert "legacy shared key" in caplog.text


@pytest.fixture
def remote(
    cli_db: Path, keys_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Any, None, None]:
    """``backend_for(token)``: an ``HttpBackend`` over the in-process engine sending ``token``."""
    import httpx

    import particles.ingest.subject_resolver as sr
    from particles import embeddings as ep
    from particles.api.client.http import HttpBackend
    from particles.config import get_config, reset_config
    from particles.db import DEFAULT_STORE

    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://engine.test")
    monkeypatch.setenv("PARTICLES_API_KEY", TEST_ENGINE_KEY)
    reset_config()
    get_config().api.allowed_hosts = ["engine.test"]
    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)

    from particles.api.app import app as fastapi_app

    real_async_client = httpx.AsyncClient

    def _asgi_client(
        *, base_url: str, timeout: float, headers: dict[str, str], purpose: str = ""
    ) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.ASGITransport(app=fastapi_app), base_url=base_url, headers=headers
        )

    monkeypatch.setattr("particles.api.client.http.egress_client", _asgi_client)

    def backend_for(token: str) -> HttpBackend:
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", token)
        return HttpBackend()

    try:
        yield backend_for
    finally:
        ep.set_embedding_model(original)


def _mcp_backend_calls() -> set[str]:
    """Every backend method an MCP tool or the server calls, read from the source."""
    import re

    root = Path(__file__).resolve().parents[1] / "particles" / "mcp"
    names: set[str] = set()
    for path in root.rglob("*.py"):
        names |= set(re.findall(r"get_backend\(\)\.(\w+)\(", path.read_text()))
    return names


def _agent_side_calls() -> dict[str, Any]:
    """How the agent-side callers call each method, with arguments a test store accepts."""
    from particles.core.schema import SuggestMode
    from particles.db import DEFAULT_STORE as S

    p = "p-missing"
    return {
        "agent_links_add": lambda b: b.agent_links_add(
            p, "p-b", relation_type="CO_EVIDENTIAL", store=S
        ),
        "agent_links_remove": lambda b: b.agent_links_remove(
            p, "p-b", relation_type="CO_EVIDENTIAL", store=S
        ),
        "agent_particle_tag": lambda b: b.agent_particle_tag(p, ["t"], store=S),
        "agent_particle_untag": lambda b: b.agent_particle_untag(p, ["t"], store=S),
        "corpus_links_suggest": lambda b: b.corpus_links_suggest(limit=5, min_sources=None),
        "deposit_text": lambda b: b.deposit_text(text="An agent note.", tags=None, store=S),
        "digest": lambda b: b.digest(S),
        "event_show": lambda b: b.event_show("e-missing"),
        "events_list": lambda b: b.events_list(
            particle=None, subject=None, entry=None, event_type=None, limit=5
        ),
        "graph": lambda b: b.graph(subject_id="s-missing"),
        "links_suggest": lambda b: b.links_suggest(
            subject_id=None, threshold=None, mode=SuggestMode.REPORT, confirmed=False
        ),
        "lint": lambda b: b.lint(fix=False, semantic=False, low_coverage_threshold=3),
        "list_corpus_entries": lambda b: b.list_corpus_entries(limit=5, source_type=None),
        "list_taxonomies": lambda b: b.list_taxonomies(),
        "particle_assert": lambda b: b.particle_assert(
            content="The agent's own claim.",
            subject_names=["release train"],
            confidence=0.7,
            source_excerpt="The agent's own claim.",
            corpus_entry_id=None,
            uncertainty_nature="EPISTEMIC",
            tags=None,
            store=S,
        ),
        "particle_detail": lambda b: b.particle_detail(p),
        "particle_retract": lambda b: b.particle_retract(particle_id=p, reason="r", store=S),
        "particle_supersede": lambda b: b.particle_supersede(
            supersedes_id=p,
            content="c",
            subject_names=["S"],
            confidence=0.5,
            source_excerpt="c",
            corpus_entry_id=None,
            uncertainty_nature="EPISTEMIC",
            tags=None,
            store=S,
            reason="r",
        ),
        "particles_by_fingerprint": lambda b: b.particles_by_fingerprint("0" * 16, limit=5),
        "quality": lambda b: b.quality(),
        "subject_detail": lambda b: b.subject_detail("s-missing", particle_id_limit=5),
        "subjects_list": lambda b: b.subjects_list(limit=5),
        "subjects_search": lambda b: b.subjects_search("release", limit=5),
    }


def _outcome(coro: Any) -> str:
    """``ok``, or the status an ``EngineHttpError`` reports, or another failure's type."""
    from particles.api.client.http import EngineHttpError

    try:
        asyncio.run(coro)
    except EngineHttpError as exc:
        return str(exc).split(" ", 1)[0]
    except Exception as exc:  # noqa: BLE001 - a 404 or a 400 is fine here; only 401/403 are not
        return type(exc).__name__
    return "ok"


class TestNoFlagDay:
    """Test plan item 16: the agent-side callers call nothing an agent key refuses."""

    def test_every_mcp_backend_call_is_classified(self) -> None:
        unclassified = _mcp_backend_calls() - set(_agent_side_calls())
        assert unclassified == set(), (
            "an MCP tool calls a backend method this test does not run with an agent key; "
            "add it to _agent_side_calls, or move the tool to an operator route on purpose"
        )

    def test_the_mcp_tools_meet_no_refusal_with_an_agent_key(self, remote: Any) -> None:
        backend = remote(_mint("agent", AGENT))
        refused = {
            name: outcome
            for name, call in _agent_side_calls().items()
            if (outcome := _outcome(call(backend))) in {"401", "403"}
        }
        assert refused == {}

    def test_the_hooks_meet_no_refusal_with_the_harvest_key(self, remote: Any) -> None:
        from datetime import UTC, datetime

        from particles.db import DEFAULT_STORE
        from particles.store.session_exposure_store import SessionExposure

        backend = remote(_mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS))
        transcript = backend.deposit_text_at_uri(
            text="A session transcript.",
            uri_r="claude-code://session/s-1",
            source_type="CONVERSATION",
            mutability="APPEND_ONLY",
            tags=["claude-code", "session:s-1", "project:repo"],
            deposited_by=HOOK,
            content_published_at=datetime.now(UTC),
            store=DEFAULT_STORE,
        )
        assert _outcome(transcript) == "ok"
        memory = backend.deposit_text_at_uri(
            text="# Memory",
            uri_r="file:///home/u/.claude/projects/repo/memory/MEMORY.md",
            source_type="LOCAL_MARKDOWN",
            mutability="MUTABLE",
            tags=["claude-code", "memory-file", "project:repo"],
            deposited_by=HOOK,
            content_published_at=datetime.now(UTC),
            store=DEFAULT_STORE,
        )
        assert _outcome(memory) == "ok"
        exposure = SessionExposure(
            session_id="s-1",
            recorded_at=datetime.now(UTC),
            source="startup",
            action="full",
            project_key="repo",
            observer_scope_applied=False,
        )
        assert _outcome(backend.record_session_exposure(DEFAULT_STORE, exposure)) not in {
            "401",
            "403",
        }

    def test_the_operator_cli_verbs_are_what_an_agent_key_meets_403_on(self, remote: Any) -> None:
        from particles.core.schema import SuggestMode

        backend = remote(_mint("agent", AGENT))
        operator_calls = {
            "extract": backend.extract("e-missing", "s-missing", agent_id="x"),
            "reindex": backend.reindex(
                entry_ids=None,
                extractor_version=None,
                extractor_id=None,
                include_failed=False,
                dry_run=False,
            ),
            "lint --fix": backend.lint(fix=True, semantic=False, low_coverage_threshold=3),
            "links suggest --apply": backend.links_suggest(
                subject_id=None, threshold=None, mode=SuggestMode.APPLY, confirmed=True
            ),
            "subjects merge": backend.subject_merge(["s-a"], "s-b"),
            "corpus retract": backend.corpus_retract("e-missing", reason="r", dry_run=False),
            "corpus links dismiss": backend.corpus_links_dismiss(
                url="https://example.org/", snooze_days=None
            ),
        }
        outcomes = {name: _outcome(call) for name, call in operator_calls.items()}
        assert outcomes == dict.fromkeys(operator_calls, "403")

    def test_the_report_branches_the_mcp_tools_use_are_literals(self) -> None:
        import ast

        root = Path(__file__).resolve().parents[1] / "particles" / "mcp" / "tools"
        lint = ast.parse((root / "lint.py").read_text())
        suggest = ast.parse((root / "links_suggest.py").read_text())

        def kwargs(tree: ast.AST, method: str) -> dict[str, str]:
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == method
                ):
                    return {k.arg or "": ast.unparse(k.value) for k in node.keywords}
            raise AssertionError(f"no {method}() call")

        assert kwargs(lint, "lint")["fix"] == "False"
        assert kwargs(suggest, "links_suggest")["mode"] == "SuggestMode.REPORT"


class TestProjectBinding:
    """Test plan item 7: a bound agent key writes into its own project only."""

    def _entry(self, entry_id: str) -> Any:
        (entry,) = [e for e in _entries() if e.entry_id == entry_id]
        return entry

    def test_a_deposit_lands_in_the_keys_project(self, engine: TestClient) -> None:
        token = _mint("agent", AGENT, project="repo")
        resp = engine.post("/corpus/deposit/text", json={"text": "Bound."}, headers=_bearer(token))
        assert resp.status_code == 200, resp.text
        assert "project:repo" in self._entry(resp.json()["entry_id"]).tags

    def test_a_deposit_naming_another_project_is_refused(self, engine: TestClient) -> None:
        token = _mint("agent", AGENT, project="repo")
        before = len(_entries())
        resp = engine.post(
            "/corpus/deposit/text",
            json={"text": "Elsewhere.", "tags": ["project:other"]},
            headers=_bearer(token),
        )
        assert resp.status_code == 403, resp.text
        assert len(_entries()) == before

    def test_an_agent_route_takes_the_project_from_the_key(self, engine: TestClient) -> None:
        token = _mint("agent", AGENT, project="repo")
        ok = engine.post(
            "/particles/deposit", json={"text": "From the bound agent."}, headers=_bearer(token)
        )
        assert ok.status_code == 200, ok.text
        assert "project:repo" in self._entry(ok.json()["entry_id"]).tags
        before = len(_entries())
        other = engine.post(
            "/particles/deposit",
            json={"text": "Into another project.", "project_key": "other"},
            headers=_bearer(token),
        )
        assert other.status_code == 403, other.text
        assert len(_entries()) == before

    def test_an_unbound_key_keeps_the_body_project(self, engine: TestClient) -> None:
        token = _mint("agent", AGENT)
        resp = engine.post(
            "/particles/deposit",
            json={"text": "Unbound.", "project_key": "other"},
            headers=_bearer(token),
        )
        assert resp.status_code == 200, resp.text
        assert "project:other" in self._entry(resp.json()["entry_id"]).tags


# --- security review 2026-10-26-1 (sixth delta audit), keys and roles -----------


def _rows(model: Any) -> list[Any]:
    async def _go() -> list[Any]:
        from sqlalchemy import select

        from particles.db import get_engine, session_scope

        try:
            async with session_scope() as s:
                return list((await s.execute(select(model))).scalars())
        finally:
            await get_engine().dispose()

    return asyncio.run(_go())


_ZERO_WIDTH = ["​", "‌", "‍", "⁠", "﻿", "᠎"]


class TestStoreWriterLabels:
    """W5: no key holds a label the store's own writers stamp by accident."""

    @pytest.mark.parametrize(
        "principal",
        [
            "general-extractor",
            "journal-extractor",
            "reddit-extractor",
            "extract-pipeline",
            "memory-consolidate",
            "interchange-import",
            "interchange-restore",
            "web-clipper",
        ],
    )
    def test_mint_refuses_a_writer_label_without_acknowledgement(
        self, keys_file: Path, principal: str
    ) -> None:
        with pytest.raises(KeyRuleError, match="reserved-principal"):
            _keys.mint(Role.AGENT, principal)
        _token, record = _keys.mint(Role.AGENT, principal, reserved_ok=True)
        assert record.principal == principal

    def test_an_agent_key_cannot_retract_an_extracted_belief(self, engine: TestClient) -> None:
        from sqlalchemy import update

        from particles.db import get_engine, session_scope
        from particles.store.particle_store import ParticleRow

        writer = _mint("agent", OTHER_AGENT)
        resp = engine.post(
            "/particles/assert",
            json={
                "content": "The staging cluster runs in eu-west-1.",
                "subject_names": ["staging cluster"],
                "confidence": 0.8,
                "source_excerpt": "The staging cluster runs in eu-west-1.",
            },
            headers=_bearer(writer),
        )
        assert resp.status_code == 200, resp.text
        pid = resp.json()["asserted_particle_id"]

        async def _as_extracted() -> None:
            # The same row, stamped as the pipeline stamps what it extracts.
            try:
                async with session_scope() as s:
                    await s.execute(
                        update(ParticleRow)
                        .where(ParticleRow.id == pid)
                        .values(
                            asserted_by="general-extractor",
                            confidence_calibration_source="EXTRACTOR_DIRECT",
                        )
                    )
                    await s.commit()
            finally:
                await get_engine().dispose()

        asyncio.run(_as_extracted())
        # Even a key minted under the extractor's id on purpose owns nothing it extracted.
        squatter = _mint("agent", "general-extractor", reserved_ok=True)
        for path, body in (
            ("/particles/retract", {"particle_id": pid, "reason": "mine"}),
            ("/particles/tag", {"particle_id": pid, "tags": ["mine"]}),
        ):
            resp = engine.post(path, json=body, headers=_bearer(squatter))
            assert resp.status_code in (400, 403), (path, resp.status_code, resp.text)
        assert engine.get(f"/particles/{pid}").json()["status"] == "ACTIVE"


class TestSessionExposureRoute:
    """W6: the exposure route is bound, attributed, write-gated, bounded and rate-limited."""

    @staticmethod
    def _body(project: str, n: int = 0) -> dict[str, Any]:
        return {
            "session_id": "s-1",
            "recorded_at": "2020-01-01T00:00:00+00:00",
            "source": "startup",
            "action": "full",
            "project_key": project,
            "observer_scope_applied": True,
            "beliefs": [{"particle_id": f"p-{i:08x}", "shown_as": "digest"} for i in range(n)],
        }

    def test_a_bound_key_cannot_record_exposure_for_another_project(
        self, engine: TestClient
    ) -> None:
        from particles.store.session_exposure_store import SessionExposureRow

        token = _mint("agent", OTHER_AGENT, project="alpha")
        resp = engine.post("/session-exposures", json=self._body("beta"), headers=_bearer(token))
        assert resp.status_code == 403, resp.text
        assert _rows(SessionExposureRow) == []

    def test_a_row_records_the_principal_the_binding_and_the_engines_time(
        self, engine: TestClient
    ) -> None:
        from particles.store.session_exposure_store import SessionExposureRow

        token = _mint("agent", OTHER_AGENT, project="alpha")
        resp = engine.post("/session-exposures", json=self._body("", 2), headers=_bearer(token))
        assert resp.status_code == 200, resp.text
        (row,) = _rows(SessionExposureRow)
        assert row.recorded_by == OTHER_AGENT
        assert row.project_key == "alpha"
        assert row.recorded_at.year >= 2026

    def test_a_store_without_belief_writes_refuses_it(self, engine: TestClient) -> None:
        from particles.config import get_config

        get_config().mcp.write.enabled_stores = []
        token = _mint("agent", OTHER_AGENT)
        resp = engine.post("/session-exposures", json=self._body("alpha"), headers=_bearer(token))
        assert resp.status_code == 403, resp.text

    def test_the_belief_list_is_bounded(self, engine: TestClient) -> None:
        from particles.config import get_config

        get_config().storage.max_page_size = 3
        token = _mint("agent", OTHER_AGENT)
        resp = engine.post(
            "/session-exposures", json=self._body("alpha", 4), headers=_bearer(token)
        )
        assert resp.status_code == 422, resp.text

    def test_the_route_is_rate_limited(self, engine: TestClient) -> None:
        from particles.config import get_config

        get_config().api.rate_limit_per_minute = 1
        token = _mint("agent", OTHER_AGENT)
        first = engine.post("/session-exposures", json=self._body("a"), headers=_bearer(token))
        second = engine.post("/session-exposures", json=self._body("a"), headers=_bearer(token))
        assert first.status_code == 200, first.text
        assert second.status_code == 429, second.text


class TestInvisibleProjectKeys:
    """W53 / W72 / E9: a project key of only invisible characters names no project."""

    @pytest.mark.parametrize("key", _ZERO_WIDTH)
    def test_a_zero_width_project_cannot_be_minted(self, keys_file: Path, key: str) -> None:
        with pytest.raises(KeyRuleError):
            _keys.check_mint(Role.AGENT, OTHER_AGENT, key, reserved_ok=False)
        with pytest.raises(KeyRuleError):
            _keys.mint(Role.AGENT, OTHER_AGENT, project=key)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", _ZERO_WIDTH)
    async def test_a_zero_width_project_key_is_refused_on_a_rescoped_store(
        self, key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import particles.operations.agent_write as aw

        monkeypatch.setattr(aw, "lens_may_engage", AsyncMock(return_value=True))
        with pytest.raises(ValueError):
            await aw.require_project_binding(object(), key)

    @pytest.mark.parametrize("key", _ZERO_WIDTH)
    def test_a_body_project_of_invisible_characters_reads_as_omitted(self, key: str) -> None:
        from particles.api.app import ParticleRetractRequest

        req = ParticleRetractRequest(particle_id="p", reason="r", project_key=key * 2)
        assert req.project_key is None

    def test_a_key_file_project_of_empty_string_does_not_bind_to_global(
        self, keys_file: Path
    ) -> None:
        import json as _json

        keys_file.write_text(
            _json.dumps(
                {
                    "version": 1,
                    "keys": [
                        {
                            "key_id": "aaaaaaaaaaaa",
                            "principal": OTHER_AGENT,
                            "role": "agent",
                            "project": "",
                            "label": None,
                            "created_at": "2026-10-26T00:00:00+00:00",
                            "revoked_at": None,
                            "digest": "0" * 64,
                        }
                    ],
                }
            )
        )
        os.chmod(keys_file, 0o600)
        try:
            records = _keys.all_keys()
        except _keys.KeyFileUnusable:
            return
        assert records["aaaaaaaaaaaa"].project is None


class TestKeyFileHardening:
    """W54, W76, W14: the key file's lock, its unreadable forms, and its path."""

    def test_revocation_survives_a_concurrent_mint(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import threading

        _victim_token, victim = _keys.mint(Role.AGENT, "agent:victim")
        real_token_hex = _keys.secrets.token_hex
        revoker: list[threading.Thread] = []

        def _interleave(n: int) -> str:
            # The second `keys create` has loaded the file; a `keys revoke` in
            # another shell starts before it writes.
            if not revoker:
                thread = threading.Thread(target=_keys.revoke, args=(victim.key_id,))
                revoker.append(thread)
                thread.start()
                thread.join(timeout=0.3)
            return real_token_hex(n)

        monkeypatch.setattr(_keys.secrets, "token_hex", _interleave)
        _keys.mint(Role.AGENT, "agent:other")
        revoker[0].join(timeout=10)
        assert not revoker[0].is_alive()
        _keys.reset_key_cache()
        assert not _keys.all_keys()[victim.key_id].live

    def test_legacy_key_keeps_working_when_the_key_file_goes_bad(
        self, cli_db: Path, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.app import app

        monkeypatch.setenv("PARTICLES_API_KEY", "legacy-operator-key")
        client = TestClient(
            app,
            base_url="http://127.0.0.1:8000",
            client=("127.0.0.1", 50000),
            raise_server_exceptions=False,
        )
        token = _mint("reader", "viewer")
        assert client.get("/auth/whoami", headers=_bearer("legacy-operator-key")).status_code == 200
        keys_file.write_bytes(b"\xff\xfe not a key file")
        os.chmod(keys_file, 0o600)
        _keys.reset_key_cache()
        resp = client.get("/auth/whoami", headers=_bearer("legacy-operator-key"))
        assert resp.status_code == 200, (resp.status_code, resp.text[:200])
        # A per-client key answers 503 (unusable file), never 500.
        assert client.get("/auth/whoami", headers=_bearer(token)).status_code == 503

    def test_a_key_file_that_is_a_directory_is_unusable(self, keys_file: Path) -> None:
        keys_file.mkdir()
        os.chmod(keys_file, 0o700)
        with pytest.raises(_keys.KeyFileUnusable):
            _keys.all_keys()
        assert _keys.has_live_keys()

    def test_relative_keys_file_is_never_read_from_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil

        from particles.config import reset_config

        # A key list a cloned repository ships: written here, owned by the user
        # who cloned it, not group- or world-writable.
        planted = tmp_path / "planted.json"
        monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(planted))
        reset_config()
        _keys.reset_key_cache()
        token, _record = _keys.mint(Role.OPERATOR, "operator:local")
        clone = tmp_path / "clone"
        clone.mkdir()
        shutil.copy2(planted, clone / "engine-keys.json")
        monkeypatch.chdir(clone)
        monkeypatch.setenv("PARTICLES_API_KEYS_FILE", "engine-keys.json")
        reset_config()
        _keys.reset_key_cache()
        try:
            with pytest.raises(_keys.KeyFileUnusable, match="relative"):
                _keys.keys_file_path()
            with pytest.raises(_keys.KeyFileUnusable):
                _keys.lookup(token)
            with pytest.raises(_keys.KeyFileUnusable):
                _keys.mint(Role.AGENT, OTHER_AGENT)
            # Refused, it still never makes the engine look keyless.
            assert _keys.has_live_keys()
        finally:
            _keys.reset_key_cache()


class TestForwardedHops:
    """W78: a forwarded hop that is not an address never names a bucket."""

    def test_forwarded_hop_never_names_a_principal_bucket(self) -> None:
        from starlette.requests import Request

        from particles.api._rate_limit import _client_bucket
        from particles.config import get_config

        get_config().api.trusted_proxies = ["10.0.0.0/8"]
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/query",
            "headers": [(b"x-forwarded-for", b"principal:operator:local, 10.0.0.7")],
            "client": ("10.0.0.2", 40000),
            "query_string": b"",
        }
        bucket = _client_bucket(Request(scope))
        assert not bucket.startswith("principal:"), bucket


class TestDoctorRoleLabel:
    """W77 / W68: `hook doctor` names the role of the token it judges."""

    def test_doctor_engine_line_names_a_reader_key_as_a_reader(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli.hook import _engine_line
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://127.0.0.1:8000")
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "ptk_read_0123456789ab_" + "R" * 43)
        reset_config()
        line = _engine_line()
        assert "an agent key" not in line
        assert "⚠" in line and "reader key" in line


# --- security review 2026-10-27-1 ------------------------------------------------


def _assert_belief(client: TestClient, token: str, text: str = "The build runs nightly.") -> str:
    resp = client.post(
        "/particles/assert",
        json={
            "content": text,
            "subject_names": ["build"],
            "confidence": 0.8,
            "source_excerpt": text,
        },
        headers=_bearer(token),
    )
    assert resp.status_code == 200, resp.text
    return str(resp.json()["asserted_particle_id"])


def _restamp(pid: str, **values: Any) -> None:
    from sqlalchemy import update

    from particles.db import get_engine, session_scope
    from particles.store.particle_store import ParticleRow

    async def _go() -> None:
        try:
            async with session_scope() as s:
                await s.execute(update(ParticleRow).where(ParticleRow.id == pid).values(**values))
                await s.commit()
        finally:
            await get_engine().dispose()

    asyncio.run(_go())


class TestReservedNamesOwnNoImports:
    """G30, G39, G44, G42: a reserved name grants no belief or entry it did not write."""

    @pytest.mark.parametrize("label", ["general-extractor", "interchange-import"])
    @pytest.mark.parametrize("calibration", ["IMPORTED", "AGENT_ASSERTED"])
    def test_an_agent_key_under_a_writer_label_never_retracts_an_imported_belief(
        self, engine: TestClient, label: str, calibration: str
    ) -> None:
        pid = _assert_belief(engine, _mint("agent", OTHER_AGENT))
        # As a migration's extraction or an interchange import stamps it, and
        # as an import that kept a bundle's agent calibration left it.
        _restamp(pid, asserted_by=label, confidence_calibration_source=calibration)
        squatter = _mint("agent", label, reserved_ok=True)
        for path, body in (
            ("/particles/retract", {"particle_id": pid, "reason": "mine"}),
            ("/particles/tag", {"particle_id": pid, "tags": ["mine"]}),
        ):
            resp = engine.post(path, json=body, headers=_bearer(squatter))
            assert resp.status_code in (400, 403), (path, resp.status_code, resp.text)
        assert engine.get(f"/particles/{pid}").json()["status"] == "ACTIVE"

    @pytest.mark.parametrize("label", ["general-extractor", "interchange-import"])
    def test_every_belief_changing_route_refuses_a_writer_label_key(
        self, engine: TestClient, label: str
    ) -> None:
        """Z6, G30: a writer-label agent key gets 403, exactly, on all six
        belief-changing routes for beliefs it asserted itself, so only the
        identity refusal stands between it and the change, and nothing changes."""
        token = _mint("agent", label, reserved_ok=True)
        a = _assert_belief(engine, token, "The build runs nightly.")
        b = _assert_belief(engine, token, "The deploy runs weekly.")
        before = {pid: engine.get(f"/particles/{pid}").json()["status"] for pid in (a, b)}
        claim = "The build runs hourly."
        calls: list[tuple[str, dict[str, Any]]] = [
            (
                "/particles/supersede",
                {
                    "supersedes_id": a,
                    "content": claim,
                    "subject_names": ["build"],
                    "confidence": 0.8,
                    "source_excerpt": claim,
                },
            ),
            ("/particles/retract", {"particle_id": a, "reason": "mine"}),
            ("/particles/link", {"particle_a": a, "particle_b": b}),
            ("/particles/unlink", {"particle_a": a, "particle_b": b}),
            ("/particles/tag", {"particle_id": a, "tags": ["mine"]}),
            ("/particles/untag", {"particle_id": a, "tags": ["mine"]}),
        ]
        for path, body in calls:
            resp = engine.post(path, json=body, headers=_bearer(token))
            assert resp.status_code == 403, (path, resp.status_code, resp.text)
        for pid in (a, b):
            got = engine.get(f"/particles/{pid}").json()
            assert got["status"] == before[pid], (pid, before, got["status"])
            assert "mine" not in (got.get("tags") or [])

    def test_an_imported_belief_is_still_the_memory_facades_own(self) -> None:
        from particles.config import get_config
        from particles.core.scoring.confidence import CalibrationSource
        from particles.operations.agent_write import _agent_owns_calibration

        facade = get_config().mcp.memory_compat.asserter_identity
        assert _agent_owns_calibration(CalibrationSource.IMPORTED, facade)
        assert _agent_owns_calibration(CalibrationSource.AGENT_ASSERTED, OTHER_AGENT)
        assert not _agent_owns_calibration(CalibrationSource.IMPORTED, "interchange-import")

    def test_a_key_record_under_a_reserved_name_never_authenticates_unacknowledged(
        self, keys_file: Path
    ) -> None:
        import json as _json

        token, _record = _keys.mint(Role.AGENT, "agent:placeholder")
        data = _json.loads(keys_file.read_text())
        # The same record under a name reserved after it was minted.
        data["keys"][0]["principal"] = "web-clipper"
        keys_file.write_text(_json.dumps(data))
        os.chmod(keys_file, 0o600)
        _keys.reset_key_cache()
        assert "web-clipper" in _keys.reserved_principals()
        assert _keys.lookup(token) is None

    def test_a_reserved_name_minted_on_purpose_still_authenticates(self, keys_file: Path) -> None:
        token, record = _keys.mint(Role.AGENT, HOOK, reserved_ok=True)
        assert record.reserved_ok
        _keys.reset_key_cache()
        found = _keys.lookup(token)
        assert found is not None and found.principal == HOOK

    def test_every_depositor_label_a_store_writer_records_is_reserved(
        self, keys_file: Path
    ) -> None:
        with pytest.raises(KeyRuleError):
            _keys.check_mint(Role.AGENT, "inbox", None, reserved_ok=False)


class TestKeyFileFailsClosed:
    """G37, G43, G72, G76, G77: a damaged key file never serves keys or leaks the host."""

    def test_a_key_file_with_a_non_object_record_is_unusable_not_a_crash(
        self, keys_file: Path
    ) -> None:
        import json as _json

        keys_file.write_text(_json.dumps({"version": 1, "keys": ["not-a-record"]}))
        os.chmod(keys_file, 0o600)
        _keys.reset_key_cache()
        with pytest.raises(_keys.KeyFileUnusable):
            _keys.all_keys()

    def test_a_non_object_record_leaves_per_client_keys_at_503_not_500(
        self, cli_db: Path, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json as _json

        from particles.api.app import app

        monkeypatch.setenv("PARTICLES_API_KEY", "legacy-operator-key")
        client = TestClient(
            app,
            base_url="http://127.0.0.1:8000",
            client=("127.0.0.1", 50000),
            raise_server_exceptions=False,
        )
        token = _mint("reader", "viewer")
        assert client.get("/auth/whoami", headers=_bearer(token)).status_code == 200
        keys_file.write_text(_json.dumps({"version": 1, "keys": [7]}))
        os.chmod(keys_file, 0o600)
        _keys.reset_key_cache()
        assert client.get("/auth/whoami", headers=_bearer("legacy-operator-key")).status_code == 200
        assert client.get("/auth/whoami", headers=_bearer(token)).status_code == 503

    def test_an_unusable_key_file_refusal_names_no_path_or_uid_to_an_unauthenticated_caller(
        self, cli_db: Path, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.app import app

        monkeypatch.setenv("PARTICLES_API_KEY", "legacy-operator-key")
        _mint("reader", "viewer")
        client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
        assert client.get("/auth/whoami", headers=_bearer("legacy-operator-key")).status_code == 200
        os.chmod(keys_file, 0o666)
        _keys.reset_key_cache()
        forged = "ptk_read_" + "0" * 12 + "_" + "A" * 43
        resp = client.get("/auth/whoami", headers=_bearer(forged))
        assert resp.status_code in (401, 503)
        detail = resp.json()["detail"]
        assert str(keys_file) not in detail and str(os.geteuid()) not in detail, detail

    def test_a_key_file_made_group_writable_is_refused_on_the_next_request(
        self, keys_file: Path
    ) -> None:
        _mint("agent", OTHER_AGENT)
        _keys.all_keys()  # the engine's cached copy
        os.chmod(keys_file, 0o666)
        with pytest.raises(_keys.KeyFileUnusable):
            _keys.all_keys()

    def test_the_lock_file_is_never_followed_through_a_symlink(
        self, keys_file: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "elsewhere" / "created-by-mint"
        target.parent.mkdir()
        keys_file.parent.mkdir(parents=True, exist_ok=True)
        keys_file.with_name(f".{keys_file.name}.lock").symlink_to(target)
        with pytest.raises(OSError):
            _mint("agent", OTHER_AGENT)
        assert not target.exists()

    def test_an_engine_with_per_client_keys_never_turns_keyless_when_its_file_disappears(
        self, cli_db: Path, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.app import app
        from particles.config import get_config

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        get_config().api.require_auth_for_reads = True
        _mint("reader", "viewer")
        client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
        assert client.get("/particles/no-such-id").status_code == 401
        keys_file.unlink()
        _keys.reset_key_cache()
        assert client.get("/particles/no-such-id").status_code == 401

    def test_a_key_file_replaced_by_a_link_to_nothing_is_unusable(
        self, keys_file: Path, tmp_path: Path
    ) -> None:
        keys_file.symlink_to(tmp_path / "missing.json")
        with pytest.raises(_keys.KeyFileUnusable):
            _keys.all_keys()


class TestKeysNeverLapseAtRuntime:
    """O5: a key file replaced at runtime by one holding no live key never opens the reads.

    The revocation, directory and link cases are pinned with O48 and O50 in
    ``tests/test_security_keys_roles_20261028.py``.
    """

    def test_a_key_file_replaced_by_one_with_no_live_key_never_opens_the_reads(
        self, cli_db: Path, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.app import app
        from particles.config import get_config

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        monkeypatch.delenv("CLAUDECODE", raising=False)
        get_config().api.require_auth_for_reads = True
        client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
        _keys.mint(Role.READER, "viewer")
        assert client.get("/particles").status_code == 401
        replacement = keys_file.with_name("engine-keys.json.new")
        replacement.write_text(json.dumps({"version": 1, "keys": []}), encoding="utf-8")
        os.chmod(replacement, 0o600)
        os.replace(replacement, keys_file)
        assert client.get("/particles").status_code in (401, 503)


class TestExposureFrameRecorders:
    """G40: only a session's harvester (or the operator) credits beliefs to its frame."""

    @staticmethod
    def _exposure(session_id: str, project: str, pids: list[str]) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "recorded_at": "2026-10-26T00:00:00+00:00",
            "source": "startup",
            "action": "full",
            "project_key": project,
            "observer_scope_applied": True,
            "beliefs": [{"particle_id": p, "shown_as": "digest"} for p in pids],
        }

    def test_another_principals_exposure_rows_never_enter_a_sessions_frame(
        self, engine: TestClient
    ) -> None:
        from particles.db import get_engine, session_scope
        from particles.operations.utility_exposure import session_frame

        hook = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)
        stranger = _mint("agent", OTHER_AGENT)
        for token, project, pid in ((hook, "alpha", "p-aaaa"), (stranger, "beta", "p-bbbb")):
            resp = engine.post(
                "/session-exposures",
                json=self._exposure("sess-1", project, [pid]),
                headers=_bearer(token),
            )
            assert resp.status_code == 200, resp.text

        async def _frame() -> Any:
            try:
                async with session_scope() as s:
                    return await session_frame(s, "sess-1")
            finally:
                await get_engine().dispose()

        frame = asyncio.run(_frame())
        assert frame is not None
        assert "aaaa" in frame.recorded and "alpha" in frame.project_keys
        assert "bbbb" not in frame.recorded and "beta" not in frame.project_keys, frame


class TestHarvestUriShape:
    """G41: a harvester's ``file://`` URI must match the whole harvest shape."""

    @pytest.mark.parametrize(
        "uri",
        [
            "file:///x/projects/s/memory/a.md%0A",
            "file:///x/projects/s/memory/a.md%00",
            "file:///x/projects/s/MEMORY.archive.md%0A",
            "file:///x/projects/s/memory/a.md/extra",
        ],
    )
    def test_a_harvester_file_uri_with_a_trailing_control_character_is_refused(
        self, uri: str
    ) -> None:
        from particles.api.app import _is_harvest_uri

        assert not _is_harvest_uri(uri)

    @pytest.mark.parametrize(
        "uri",
        [
            "file:///home/u/.claude/projects/-home-u-repo/memory/MEMORY.md",
            "file:///home/u/.claude/projects/-home-u-repo/memory/topic/notes.md",
            "file:///home/u/.particles/claude-code/projects/-home-u-repo/MEMORY.archive.md",
        ],
    )
    def test_the_harvests_own_file_uris_are_admitted(self, uri: str) -> None:
        from particles.api.app import _is_harvest_uri

        assert _is_harvest_uri(uri)


class TestForwardedHopsWithPorts:
    """W78: a forwarded hop with a port or brackets never lets a client-written hop through."""

    @staticmethod
    def _request(xff: str, peer: str, host: bytes = b"127.0.0.1:8000") -> Any:
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/particles",
            "headers": [(b"host", host), (b"x-forwarded-for", xff.encode())],
            "client": (peer, 40000),
            "server": ("127.0.0.1", 8000),
            "query_string": b"",
        }
        return Request(scope)

    @pytest.mark.parametrize("hop", ["203.0.113.7:4711", "[2001:db8::7]", "[2001:db8::7]:443"])
    def test_a_remote_client_forwarded_with_a_port_is_never_read_as_loopback(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch, hop: str
    ) -> None:
        from particles.api.auth import _client_is_loopback, request_bound_refusal
        from particles.config import get_config

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        get_config().api.trusted_proxies = ["127.0.0.0/8"]
        assert not _client_is_loopback(self._request(hop, "127.0.0.1"))
        assert request_bound_refusal(self._request(hop, "127.0.0.1")) is not None

    @pytest.mark.parametrize("hop", ["203.0.113.7:4711", "[2001:db8::7]:443", "unknown"])
    def test_a_client_written_loopback_hop_left_of_a_forwarded_hop_is_never_read_as_loopback(
        self, monkeypatch: pytest.MonkeyPatch, hop: str
    ) -> None:
        from particles.api.auth import _client_is_loopback
        from particles.config import ApiConfig, get_config

        validated = ApiConfig(trusted_proxies=["10.0.0.0/8"]).trusted_proxies
        monkeypatch.setattr(get_config().api, "trusted_proxies", validated)
        request = self._request(f"127.0.0.1, {hop}", "10.0.0.5", b"engine.example:8000")
        assert not _client_is_loopback(request)

    def test_a_forwarded_hop_with_a_port_names_the_clients_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.auth import _real_client_host
        from particles.config import get_config

        get_config().api.trusted_proxies = ["10.0.0.0/8"]
        assert _real_client_host(self._request("203.0.113.7:4711", "10.0.0.5")) == "203.0.113.7"
        assert _real_client_host(self._request("[2001:db8::7]:443", "10.0.0.5")) == "2001:db8::7"


class TestBlankRenderingProjectKeys:
    """G65, G74: a project key with no character a reader can see names no project."""

    @pytest.mark.parametrize(
        "key",
        [
            "ㅤ",  # Hangul filler
            "ᅟᅠ",  # Hangul choseong and jungseong fillers
            "ﾠ",  # halfwidth Hangul filler
            "⠀",  # braille pattern blank
            "͏",  # combining grapheme joiner
            "́́",  # lone combining marks
            "\u0001",  # a control character
            "\ue000",  # private use
        ],
    )
    def test_a_project_key_that_renders_blank_names_no_project(
        self, keys_file: Path, key: str
    ) -> None:
        from particles.core.observer_scope import names_a_project

        assert not names_a_project(key)
        with pytest.raises(KeyRuleError):
            _keys.check_mint(Role.AGENT, OTHER_AGENT, key, reserved_ok=False)

    @pytest.mark.parametrize("key", ["alpha", "café", "項目", "repo-1", "-"])
    def test_a_visible_project_key_still_names_a_project(self, key: str) -> None:
        from particles.core.observer_scope import names_a_project

        assert names_a_project(key)


class TestUsefulCreditIsOperatorOnly:
    """G75: only an operator press earns the credit (§6, amended 2026-10-27)."""

    @staticmethod
    def _useful(engine: TestClient, pid: str, token: str | None = None) -> dict[str, Any]:
        headers = _bearer(token) if token is not None else None
        resp = engine.post("/memory/useful", json={"particle_id": pid}, headers=headers)
        assert resp.status_code == 200, resp.text
        return dict(resp.json())

    @staticmethod
    def _credits(pid: str) -> list[Any]:
        from particles.store.utility_store import SOURCE_EXPLICIT, UtilityEventRow

        rows = _rows(UtilityEventRow)
        return [r for r in rows if r.particle_id == pid and r.source == SOURCE_EXPLICIT]

    def test_an_agent_press_earns_no_credit_and_names_its_principal(
        self, engine: TestClient
    ) -> None:
        from particles.store.event_store import OperatorEventRow

        pid = _assert_belief(engine, _mint("agent", OTHER_AGENT))
        body = self._useful(engine, pid, _mint("agent", AGENT))

        assert body["counted"] is False
        assert body["credit_key"] is None
        assert self._credits(pid) == []
        (event,) = [r for r in _rows(OperatorEventRow) if r.event_type == "BELIEF_MARKED_USEFUL"]
        assert event.payload["http_principal"] == AGENT
        assert event.payload["operator_credit"] is False
        assert "credit_key" not in event.payload

    def test_an_agent_press_never_spends_the_operators_daily_credit(
        self, engine: TestClient
    ) -> None:
        pid = _assert_belief(engine, _mint("agent", OTHER_AGENT))
        self._useful(engine, pid, _mint("agent", AGENT))
        # The operator's own press, with the legacy key the fixture sends by default.
        operator = self._useful(engine, pid)

        assert operator["counted"] is True, operator
        assert len(self._credits(pid)) == 1

    @pytest.mark.parametrize("per_client_key", [False, True])
    def test_an_operator_press_earns_one_credit_per_belief_per_day(
        self, engine: TestClient, per_client_key: bool
    ) -> None:
        pid = _assert_belief(engine, _mint("agent", OTHER_AGENT))
        from particles.config import get_config

        operator_id = get_config().curation.operator_identity
        token = _mint("operator", operator_id) if per_client_key else None
        first = self._useful(engine, pid, token)
        second = self._useful(engine, pid, token)

        assert first["counted"] is True
        assert first["credit_key"].startswith("explicit:http:/memory/useful:")
        assert second["counted"] is False
        assert len(self._credits(pid)) == 1
