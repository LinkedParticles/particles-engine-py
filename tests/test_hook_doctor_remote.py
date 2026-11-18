# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""``particles hook doctor`` against a remote engine.

With ``engine.base_url`` set the doctor probes the engine instead of judging a
local SQLite file: ``GET /health`` (reachability, latency against the hooks'
deadline, the client-versus-engine version) and an authenticated
``GET /digest/{store}``, then one line each for the harvest, the Stop hook
and the projection. The engine is a mocked transport behind the engine
client's checked ``egress_client`` seam; no socket is opened.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import particles
from particles.api.cli import app
from particles.config import get_config, reset_config
from tests._claude_projects import isolate_claude_code

_TOKEN = "engine-token-doctor-0123456789"

Handler = Callable[[httpx.Request], httpx.Response]


def _engine(
    *,
    version: str | None = None,
    digest_status: int = 200,
    health_status: int = 200,
    status: str = "ok",
    whoami: dict[str, Any] | None = None,
) -> Handler:
    """A fake engine answering the three routes the doctor reads."""

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            if health_status != 200:
                return httpx.Response(health_status, json={"detail": "health broke"})
            return httpx.Response(
                200, json={"status": status, "version": version or particles.__version__}
            )
        if path == "/digest/default":
            if digest_status != 200:
                return httpx.Response(digest_status, json={"detail": "Invalid or missing token."})
            assert request.headers.get("Authorization") == f"Bearer {_TOKEN}"
            return httpx.Response(200, json={"markdown": "# Memory digest\n- a belief\n"})
        if path.startswith("/digest/"):
            return httpx.Response(404, json={"detail": "Unknown store."})
        if path == "/auth/whoami":
            return httpx.Response(200, json=whoami or {"principal": "operator", "role": "operator"})
        return httpx.Response(404, json={"detail": "no such route"})

    return handle


@pytest.fixture
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Handler], None]]:
    """A thin client: an engine configured, and no local database file at all."""
    isolate_claude_code(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'absent.db'}")
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://engine.test")
    monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", _TOKEN)
    reset_config()

    def install(handler: Handler) -> None:
        real = httpx.AsyncClient

        def _client(
            *, base_url: str, timeout: float, headers: dict[str, str], purpose: str = ""
        ) -> httpx.AsyncClient:
            return real(transport=httpx.MockTransport(handler), base_url=base_url, headers=headers)

        monkeypatch.setattr("particles.api.client.http.egress_client", _client)

    yield install
    reset_config()


def _doctor() -> Any:
    return CliRunner().invoke(app, ["hook", "doctor", "--store", "default"])


def test_a_healthy_engine_passes_on_a_thin_client_with_no_local_file(
    remote: Callable[[Handler], None],
) -> None:
    remote(_engine())
    result = _doctor()
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "reachability:   GET /health answered in" in out
    assert f"version:        engine and client both {particles.__version__}" in out
    assert "digest:         GET /digest/default served" in out
    assert "this environment's token" in out
    assert "✓ The engine answers and serves store 'default'" in out
    # The local-store checks say they do not apply rather than failing.
    assert "Database file does not exist" not in out
    assert "local store:    not checked" in out
    assert "schema:         not checked from this client" in out
    assert "corpus tables:" not in out


def test_an_unreachable_engine_fails(remote: Callable[[Handler], None]) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    remote(refuse)
    result = _doctor()
    assert result.exit_code == 1, result.output
    assert "reachability:   ✗ Could not reach the Particles engine at https://engine.test" in (
        result.stdout
    )
    assert "✗ The hooks cannot use store 'default' on the engine" in result.stdout
    assert "✓" not in result.stdout.split("reachability:")[1]


def test_a_failing_health_route_fails(remote: Callable[[Handler], None]) -> None:
    remote(_engine(health_status=500))
    result = _doctor()
    assert result.exit_code == 1, result.output
    assert "reachability:   ✗ GET /health answered 500: health broke" in result.stdout


def test_a_refused_token_is_told_apart_from_success(remote: Callable[[Handler], None]) -> None:
    remote(_engine(digest_status=401))
    result = _doctor()
    assert result.exit_code == 1, result.output
    assert "digest:         ✗ GET /digest/default refused (401)" in result.stdout
    assert "wrong or revoked" in result.stdout
    assert _TOKEN not in result.stdout


def test_a_missing_token_is_named(
    remote: Callable[[Handler], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN")
    reset_config()
    remote(_engine(digest_status=401))
    result = _doctor()
    assert result.exit_code == 1, result.output
    assert "this environment holds no PARTICLES_ENGINE_TOKEN" in result.stdout


def test_a_keyless_refusal_fails(remote: Callable[[Handler], None]) -> None:
    remote(_engine(digest_status=503))
    result = _doctor()
    assert result.exit_code == 1, result.output
    assert "digest:         ✗ GET /digest/default answered 503" in result.stdout


def test_a_token_the_hooks_do_not_pin_is_disclosed(
    remote: Callable[[Handler], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    remote(_engine())
    monkeypatch.setattr(
        "particles.api.cli.hook._hook_pinned_tokens", lambda: ["another-token-0123456789"]
    )
    result = _doctor()
    assert result.exit_code == 0, result.output
    assert "the probe used this environment's token, not the one the installed hooks pin" in (
        result.stdout
    )
    assert "another-token-0123456789" not in result.stdout


def test_an_engine_only_the_environment_names_keeps_the_relative_store_refusal(
    remote: Callable[[Handler], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hooks refuse a working-directory store unless the pinned config names the engine."""
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./relative.db")
    reset_config()
    remote(_engine())
    result = _doctor()
    assert result.exit_code == 1, result.output
    assert "hook refusal:   ✗ every hook refuses and does nothing" in result.stdout
    assert "✓ The engine answers" not in result.stdout


def test_a_store_the_engine_does_not_configure_fails(remote: Callable[[Handler], None]) -> None:
    remote(_engine())
    result = CliRunner().invoke(app, ["hook", "doctor", "--store", "elsewhere"])
    assert result.exit_code == 1, result.output
    assert "the engine does not configure store 'elsewhere'" in result.stdout


def test_a_version_mismatch_warns_without_failing(remote: Callable[[Handler], None]) -> None:
    remote(_engine(version="0.0.1"))
    result = _doctor()
    assert result.exit_code == 0, result.output
    assert f"version:        ⚠ engine 0.0.1, this client {particles.__version__}" in result.stdout
    assert "lockstep" in result.stdout


def test_a_degraded_engine_is_disclosed(remote: Callable[[Handler], None]) -> None:
    remote(_engine(status="degraded"))
    result = _doctor()
    assert result.exit_code == 0, result.output
    assert "the engine reports status 'degraded'" in result.stdout


def test_a_slow_engine_is_measured_against_the_hook_deadline(
    remote: Callable[[Handler], None],
) -> None:
    remote(_engine())
    get_config().claude_code.hook_deadline_seconds = 1e-9
    result = _doctor()
    assert result.exit_code == 0, result.output
    assert "over the hooks' 1e-09 s deadline" in result.stdout


def test_the_harvest_off_is_reported_and_the_projection_with_it(
    remote: Callable[[Handler], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    remote(_engine())
    monkeypatch.setattr("particles.api.cli.hook.projection_enabled", lambda: True)
    assert get_config().claude_code.harvest.allow_remote is False
    result = _doctor()
    assert result.exit_code == 0, result.output
    assert "harvest:        ⚠ off: claude_code.harvest.allow_remote is false" in result.stdout
    assert "projection:     ⚠ MEMORY.md is never re-rendered" in result.stdout
    assert "stop hook:      none installed" in result.stdout


def test_the_harvest_on_names_the_engine_render(
    remote: Callable[[Handler], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    remote(_engine())
    monkeypatch.setattr("particles.api.cli.hook.projection_enabled", lambda: True)
    get_config().claude_code.harvest.allow_remote = True
    result = _doctor()
    assert result.exit_code == 0, result.output
    assert "harvest:        SessionEnd deposits transcripts and memory files on the engine" in (
        result.stdout
    )
    assert "utility mining and memory-absence recording need a local store" in result.stdout
    assert "the engine renders the MEMORY.md region (POST /memory/projection/default)" in (
        result.stdout
    )


def test_the_local_path_is_unchanged_without_an_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_claude_code(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'absent.db'}")
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    reset_config()
    try:
        result = _doctor()
    finally:
        reset_config()
    assert result.exit_code == 1, result.output
    assert "Database file does not exist" in result.stdout
    assert "reachability:" not in result.stdout


# ---------------------------------------------------------------------------
# the projection line names what the manifest's file sections send.
# ---------------------------------------------------------------------------

_BLOCK_MANIFEST = """\
name: memory-index
sections:
  - block: intro.md
  - title: "Memory index"
    render: bullets
    top_k: 60
"""


def _recording(paths: list[str]) -> Handler:
    inner = _engine()

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return inner(request)

    return handle


def _block_manifest(text: str = "A hand-written intro.\n") -> Path:
    from particles.api.cli._claude_code import memory_manifest_path

    path = memory_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_BLOCK_MANIFEST)
    (path.parent / "intro.md").write_text(text)
    return path


def test_the_projection_line_counts_the_file_sections_and_sends_nothing(
    remote: Callable[[Handler], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths: list[str] = []
    remote(_recording(paths))
    monkeypatch.setattr("particles.api.cli.hook.projection_enabled", lambda: True)
    get_config().claude_code.harvest.allow_remote = True
    _block_manifest("A hand-written intro.\n")
    result = _doctor()
    assert result.exit_code == 0, result.output
    assert "file sections: 1 file(s), 22 bytes sent with each render (limit 65536)" in (
        result.stdout
    )
    assert "A hand-written intro." not in result.stdout
    assert not any(p.startswith("/memory/projection") for p in paths)


def test_the_projection_line_names_a_missing_file_section(
    remote: Callable[[Handler], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    remote(_engine())
    monkeypatch.setattr("particles.api.cli.hook.projection_enabled", lambda: True)
    get_config().claude_code.harvest.allow_remote = True
    _block_manifest().with_name("intro.md").unlink()
    result = _doctor()
    assert (
        "⚠ file sections: the render is refused until this is fixed: a file section names "
        "a file that does not exist"
    ) in result.stdout
    assert "intro.md" not in result.stdout.split("projection:")[1]


def test_the_projection_line_names_a_total_over_the_cap(
    remote: Callable[[Handler], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.operations.projection.memory_region import MAX_REMOTE_BLOCK_BYTES

    remote(_engine())
    monkeypatch.setattr("particles.api.cli.hook.projection_enabled", lambda: True)
    get_config().claude_code.harvest.allow_remote = True
    _block_manifest("x" * (MAX_REMOTE_BLOCK_BYTES + 1))
    result = _doctor()
    assert f"over the {MAX_REMOTE_BLOCK_BYTES}-byte limit for a remote render" in result.stdout
    assert "⚠ file sections: the render is refused until this is fixed" in result.stdout


def test_the_projection_line_names_the_harvest_off_for_file_sections(
    remote: Callable[[Handler], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    remote(_engine())
    monkeypatch.setattr("particles.api.cli.hook.projection_enabled", lambda: True)
    assert get_config().claude_code.harvest.allow_remote is False
    _block_manifest()
    result = _doctor()
    assert "file sections: the manifest has some, and their text is sent only with " in (
        result.stdout
    )
    assert "claude_code.harvest.allow_remote on, so each SessionStart pushes the full digest" in (
        result.stdout
    )


def test_the_projection_line_says_nothing_more_without_file_sections(
    remote: Callable[[Handler], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    remote(_engine())
    monkeypatch.setattr("particles.api.cli.hook.projection_enabled", lambda: True)
    get_config().claude_code.harvest.allow_remote = True
    path = _block_manifest()
    path.write_text(_BLOCK_MANIFEST.replace("  - block: intro.md\n", ""))
    result = _doctor()
    assert "file sections" not in result.stdout


# --- the write switch ---------------------------------------------


def _who(stores: list[str] | None, role: str = "agent") -> dict[str, Any]:
    body: dict[str, Any] = {"anonymous": False, "principal": "claude-code-hooks", "role": role}
    if stores is not None:
        body["belief_write_stores"] = stores
    return body


def test_a_store_that_takes_writes_is_named(remote: Callable[[Handler], None]) -> None:
    remote(_engine(whoami=_who(["default"])))
    result = _doctor()
    assert result.exit_code == 0, result.output
    assert "writes:         the engine accepts belief writes on store 'default'" in result.stdout


def test_a_store_the_engine_refuses_writes_on_warns_without_failing(
    remote: Callable[[Handler], None],
) -> None:
    """The digest read passes, so only whoami tells the doctor about the 403s."""
    remote(_engine(whoami=_who(["memory"])))
    result = _doctor()
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "writes:         ⚠ the engine refuses belief writes on store 'default'" in out
    assert "mcp.write.enabled_stores" in out
    assert "✓ The engine answers and serves store 'default'" in out


def test_a_reader_key_is_named_as_the_cause(remote: Callable[[Handler], None]) -> None:
    remote(_engine(whoami=_who([], role="reader")))
    result = _doctor()
    assert result.exit_code == 0, result.output
    assert "the key is a reader key" in result.stdout


def test_an_older_engine_is_not_judged(remote: Callable[[Handler], None]) -> None:
    remote(_engine(whoami=_who(None)))
    result = _doctor()
    assert result.exit_code == 0, result.output
    assert "writes:         not checked; the engine's /auth/whoami does not name" in result.stdout


def test_the_doctor_never_writes_to_check_the_switch(remote: Callable[[Handler], None]) -> None:
    seen: list[str] = []
    inner = _engine(whoami=_who(["default"]))

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return inner(request)

    remote(record)
    assert _doctor().exit_code == 0
    assert set(seen) == {"GET"}


def test_a_refused_digest_skips_the_write_line(remote: Callable[[Handler], None]) -> None:
    remote(_engine(digest_status=401, whoami=_who(["default"])))
    result = _doctor()
    assert "writes:" not in result.stdout
