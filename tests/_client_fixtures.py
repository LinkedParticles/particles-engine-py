# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Client-layer pytest fixtures, shared by both exported trees (D4).

The split-package build gives the Client (``linkedparticles-core``) and Engine
(``linkedparticles``) distributions a test suite each, and both need the same
process-hygiene fixtures: the env defaults, the logger-level restore, the
env-leak assertion, the config/LLM-client reset between tests, the live-LLM
guard, and the encoder-absent seam. Every fixture here touches **only**
Client-layer modules (``particles.config``, ``particles.embeddings``,
``particles.llm``) plus the standard library and ``httpx`` (a Client
dependency), so the file rides both repos unchanged.

It is one shared body rather than a hand-copied overlay conftest precisely so
the two trees cannot drift. Each tree's ``conftest.py`` re-exports these names;
the private (Engine) ``conftest.py`` additionally defines the Engine fixtures
(``db_session``, ``cli_db``, and the Engine half of the per-test reset).

**Adding a fixture here** means asserting it is Client-pure — the export's
import-purity scan enforces that mechanically, so an Engine reference added by
accident drops this file out of the Client tree and the Client suite loses
every fixture at once. Put Engine-touching fixtures in ``conftest.py``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from collections.abc import Generator

import httpx
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Set default test environment before any tests are collected.

    The lazy SQLAlchemy engine reads ``DATABASE_URL`` on its first
    ``get_engine()`` call, which is invoked from inside fixtures — so setting
    the env here (before collection) is sufficient. The autouse fixture below
    calls ``reset_config()``, which discards the cached engine, between every
    test. Only the Engine tree opens a database, but the defaults are set here
    because the ``PARTICLES_CONFIG`` override below has to happen once, before
    collection, for either suite.

    ``PARTICLES_CONFIG`` is unconditionally pointed at a non-existent path so
    the developer's local ``./config.yaml`` (e.g. ``inbox.file_path``,
    ``obsidian.default_output_path``) does not leak into tests. Tests that
    need specific config values monkeypatch the individual env-var overrides
    registered in ``_ENV_OVERRIDES`` (those take precedence over the file
    loader, which is missing).

    Under ``pytest-xdist`` (``-n auto``) this runs once per worker process, each
    with a distinct ``PYTEST_XDIST_WORKER`` (``gw0``, ``gw1``, …). The blob
    store is the only filesystem-shared resource (``DATABASE_URL`` is per-process
    ``:memory:``; ``cli_db`` / ``api_db`` use per-test ``tmp_path``), so it is
    suffixed per worker to keep parallel workers from racing on the same
    content-addressed directory.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    blob_dir = f"/tmp/particles_test_blobs{('_' + worker) if worker else ''}"
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    os.environ.setdefault("PARTICLES_BLOB_DIR", blob_dir)
    os.environ["PARTICLES_CONFIG"] = "/nonexistent-test-config.yaml"
    # Benchmark-run persistence is on by default; keep test runs out of the
    # developer's real ~/.particles/benchmark/runs (worker-suffixed like the
    # blob dir).
    os.environ.setdefault(
        "BENCHMARK_RUNS_DIR",
        f"/tmp/particles_test_benchmark_runs{('_' + worker) if worker else ''}",
    )


@pytest.fixture(autouse=True)
def reset_client_state() -> None:
    """Drop the process-global Client state a previous test may have set.

    ``reset_config()`` reloads the config singleton — which also clears the
    cached DB engine and, via the seam's reset hook, the circuit
    breaker — and ``set_client(None)`` drops any Anthropic client a test
    injected, so the ``ANTHROPIC_API_KEY`` check in ``get_client()`` applies
    uniformly. The Engine half of the per-test reset (subject-resolver cache,
    CLI output settings) lives in the Engine tree's own ``conftest.py``.
    """
    from particles.config import reset_config
    from particles.llm import set_client

    reset_config()
    set_client(None)


@pytest.fixture(autouse=True)
def restore_logger_levels() -> Generator[None, None, None]:
    """Restore the process-global logger levels a CLI verb may have mutated.

    ``configure_output`` / ``configure_logging`` set the level of the
    ``particles`` logger family for the whole process — ``--quiet`` pins it at
    ERROR — and nothing puts it back. A later test in the same worker that
    asserts on a ``particles.*`` WARNING through ``caplog`` then sees an empty
    log, because the record is dropped at the logger before it ever reaches the
    capture handler. That is what made
    ``test_observability.py::test_extra_absent_warns_and_noops`` fail
    intermittently under ``-n auto``: it passed or failed depending on whether
    xdist happened to schedule a ``--quiet`` CLI test into the same worker
    first. Snapshot-and-restore here so the leak cannot cross a test boundary.
    """
    names = ("particles", "httpx", "httpcore")
    saved = {name: logging.getLogger(name).level for name in names}
    root_level = logging.root.level
    try:
        yield
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)
        logging.root.setLevel(root_level)


@pytest.fixture(autouse=True)
def no_env_leak() -> Generator[None, None, None]:
    """Fail the test that leaks a process-env var, not the one that trips over it.

    Sibling of ``restore_logger_levels`` above, for the same failure class: a
    process-global a CLI verb mutates, surfacing only when ``-n auto`` happens
    to schedule the victim into the same worker.

    The concrete case is ``PARTICLES_API_BIND_HOST``. ``engine serve`` writes it
    into ``os.environ`` on purpose — the documented bootstrap path, so
    the fail-closed gate sees the interface uvicorn is about to bind — and under
    ``CliRunner`` that write lands in the pytest process instead of dying with a
    launcher. ``monkeypatch.delenv(..., raising=False)`` does not protect
    against it: when the variable is absent, monkeypatch registers no restore,
    so there is nothing to undo a write that happens later in the test body.
    `main` went red on 2026-08-03 with
    ``test_config.py::TestApiBindHost::test_default_is_loopback`` asserting
    ``'0.0.0.0' == '127.0.0.1'`` — a test that touches none of this.

    Restoring silently would hide the next such leak, so this asserts. Keep the
    watched set to variables **product code writes**; monkeypatch already covers
    everything a test sets on itself.
    """
    watched = ("PARTICLES_API_BIND_HOST", "PARTICLES_DAEMON_ENABLED")
    before = {name: os.environ.get(name) for name in watched}
    try:
        yield
    finally:
        leaked = {n: os.environ.get(n) for n in watched if os.environ.get(n) != before[n]}
        for name, original in before.items():
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original
        if leaked:
            raise AssertionError(
                f"test leaked process-env var(s) into the session: {leaked}. "
                "Restore them in the test (or a local fixture) — a later test in "
                "the same xdist worker will otherwise fail for no visible reason."
            )


#: Provider credentials a unit test must never see (``particles/secrets.py``).
#: ``PARTICLES_LLM_API_KEY_<NAME>`` is per named ``llm.providers`` entry, so it
#: is matched by prefix.
_PROVIDER_KEY_VARS = ("ANTHROPIC_API_KEY", "PARTICLES_LOCAL_LLM_API_KEY")
_PROVIDER_KEY_PREFIX = "PARTICLES_LLM_API_KEY_"


def _is_anthropic_request(request: httpx.Request) -> bool:
    """True for a request the Anthropic SDK sent, whatever base URL it used.

    The SDK stamps ``anthropic-version`` on every request, so this also catches
    a developer shell with ``ANTHROPIC_BASE_URL`` pointed at a gateway.
    """
    return "anthropic-version" in request.headers or request.url.host.endswith("anthropic.com")


@pytest.fixture(autouse=True)
def no_live_llm(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Keep every unit test off the live, billed LLM API.

    Sibling of ``no_env_leak`` above: a guard against a failure the suite cannot
    otherwise see. Most LLM call sites degrade gracefully when the provider call
    fails, so a test that forgets to mock the ``set_client`` seam still passes
    with no key set, and quietly makes live calls when the developer's shell has
    one. That cost real money and most of the unit suite's wall time: with a key
    set, the suite ran in about 360 s instead of about 70 s, with identical
    results.

    Two layers. The provider keys are removed from the environment, so an
    unmocked call site takes the same no-key path it takes in CI. A test that
    sets a fake key on purpose, to reach a key-gated branch, still builds a real
    SDK client, so the Anthropic SDK's HTTP traffic is also intercepted: the
    request gets an immediate 400 (not retried, so no backoff delay) and the
    test fails at teardown naming the request. Tests marked ``integration`` are
    exempt; they exist to make live calls.
    """
    if request.node.get_closest_marker("integration") is not None:
        yield
        return

    for name in list(os.environ):
        if name in _PROVIDER_KEY_VARS or name.startswith(_PROVIDER_KEY_PREFIX):
            monkeypatch.delenv(name)

    blocked: list[str] = []

    def _refuse(req: httpx.Request) -> httpx.Response:
        blocked.append(f"{req.method} {req.url}")
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "live LLM call refused in a unit test",
                },
            },
            request=req,
        )

    real_send = httpx.Client.send
    real_async_send = httpx.AsyncClient.send

    # The parameter is spelled ``request`` because httpx passes it by keyword
    # on some paths (``send(request=...)``); it shadows the fixture's own.
    def send(
        self: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        if _is_anthropic_request(request):
            return _refuse(request)
        return real_send(self, request, *args, **kwargs)

    async def async_send(
        self: httpx.AsyncClient, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        if _is_anthropic_request(request):
            return _refuse(request)
        return await real_async_send(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    yield
    if blocked:
        raise AssertionError(
            f"unit test reached the live LLM API ({len(blocked)} request(s), refused): "
            f"{blocked}. Mock it at the particles.llm.set_client seam "
            "(tests/AGENTS.md § Mocking strategy), or mark the test integration."
        )


@pytest.fixture
def no_embedding_model(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Make ``get_embedding_model()`` genuinely return ``None`` everywhere.

    Patching the function reaches only call sites that defer their import, and
    several of the paths under test bind it at module top — so this blocks the
    ``sentence_transformers`` import instead, which is what an install without
    the package actually looks like. Every call site then takes its real
    encoder-free branch, with no per-module patch to keep in sync.

    ``sys.modules[name] = None`` is the documented way to make ``import name``
    raise ``ImportError``; the singleton is cleared on both sides so a cached
    model neither hides the effect nor leaks out of the test.
    """
    import sys

    from particles import embeddings as ep

    original = ep._embedding_model
    ep.set_embedding_model(None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # type: ignore[arg-type]
    try:
        yield
    finally:
        ep.set_embedding_model(original)


def stream_via_create(client: Any) -> Any:
    """Answer ``client.messages.stream(...)`` with what ``messages.create`` returns.

    The Anthropic adapter streams any call whose ``max_tokens`` is above the
    SDK's non-streaming ceiling, which the extraction budgets are, and reads the
    reply with ``get_final_message()``. A test that scripts its replies on a
    mocked ``messages.create`` calls this once on the mock, and the same script
    then serves both paths: each ``stream`` call is routed through ``create``
    (so its call list, ``side_effect`` sequence and raised errors all apply) and
    the result comes back as the stream's final message. Which path a budget
    takes is pinned by the adapter's own tests in ``tests/test_llm.py``.
    """

    def _stream(**kwargs: Any) -> MagicMock:
        message = client.messages.create(**kwargs)
        manager = MagicMock()
        manager.__enter__.return_value.get_final_message.return_value = message
        manager.__exit__.return_value = False
        return manager

    client.messages.stream = MagicMock(side_effect=_stream)
    return client
