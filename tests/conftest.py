# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures — the Engine (private monorepo) half.

The Client-pure fixtures live in ``tests/_client_fixtures.py`` and are
re-exported below; both exported trees receive that module verbatim so the two
suites cannot drift (D4). Everything defined *here* touches Engine
modules (``particles.db``, the stores, ``particles.ingest``,
``particles.api.cli``) and therefore rides the Engine repo only.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pwd
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

# Re-exported, not redefined: pytest picks the `pytest_configure` hook and the
# autouse fixtures up off this module's namespace. Keep the list exhaustive —
# a name dropped here silently disables that fixture for the whole suite.
from tests._client_fixtures import (  # noqa: F401
    no_embedding_model,
    no_env_leak,
    no_live_llm,
    pytest_configure,
    reset_client_state,
    restore_logger_levels,
)


@pytest.fixture(autouse=True)
def clear_subject_cache(reset_client_state: None) -> None:  # noqa: F811
    """Engine half of the per-test global reset.

    The Client half — ``reset_config()`` + ``set_client(None)`` — is
    ``reset_client_state`` in ``tests/_client_fixtures.py``; it runs for both
    trees. This fixture adds the Engine-only globals on top, and takes the
    Client half as a parameter so the two halves keep the ordering the single
    pre-split fixture had (config reset first, then the Engine caches). pytest
    resolves fixtures by parameter name, so that parameter shadowing the
    re-exported name above is the idiom rather than a redefinition — hence the
    ``F811`` suppression.
    """
    from particles.ingest.subject_resolver import clear_cache

    # Importing the seam registers its reset hook: the
    # reset_config() in reset_client_state then clears the circuit
    # breaker and the probe-failure counter uniformly — no per-global conftest
    # poking needed.
    from particles.operations import _llm  # noqa: F401

    clear_cache()
    # Reset the CLI output settings context var: it is set by
    # configure_output() and, being a process-global context var, would otherwise
    # leak an explicit --progress/--quiet from one test into the next (e.g. into a
    # heartbeat no-op-off-a-TTY assertion).
    from particles.api.cli._output import _CURRENT, OutputSettings

    _CURRENT.set(OutputSettings())


@pytest.fixture(autouse=True)
def second_reading_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extraction's second reading of a contradiction confirms by default.

    The ladder tests script the first probe and nothing else; with no key the
    reading would fail, which drops a cross-entry pair's signal. The default
    here is the reading that changes nothing, so those tests keep asserting
    today's outcomes. A test of the reading itself patches
    ``particles.ingest.second_reading._llm_verify_contradiction`` again (or
    ``read_pair``); the census's own calls go through
    ``operations.lint.contradictions`` and are untouched.
    """
    from particles.ingest import second_reading

    async def _confirm(
        a: second_reading.ClaimContext, b: second_reading.ClaimContext
    ) -> second_reading.ProbeVerdict:
        return second_reading.ProbeVerdict(
            usable=True, contradicts=True, description="confirmed (test default)"
        )

    monkeypatch.setattr(second_reading, "_llm_verify_contradiction", _confirm)


# The engine's bearer-token pair: the key the server expects, and the
# token an HTTP client sends to it.
_ENGINE_BEARER_VARS = ("PARTICLES_API_KEY", "PARTICLES_ENGINE_TOKEN")


@pytest.fixture(autouse=True)
def no_engine_bearer(no_env_leak: None, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """Start every test from an unset engine bearer, whatever the shell exports.

    With ``PARTICLES_API_KEY`` exported, auth stops being the ``dev-key``
    no-op the suite assumes, and every in-process request the tests make
    without a bearer came back 401: 129 failures, and the pre-commit unit
    suite refused every commit. A test that exercises auth sets the variable
    itself, which overrides this. The fixture lives here rather than in
    ``_client_fixtures.py`` because only Engine code (``particles.api``)
    reads either variable.

    It takes ``no_env_leak`` as a parameter to pin the order: pytest sets up
    autouse fixtures alphabetically, which would otherwise instantiate
    ``monkeypatch`` before ``no_env_leak``. ``monkeypatch`` would then undo a
    test's own ``setenv`` only after the leak check had already run, so every
    test that sets a watched variable would be reported as leaking it.
    """
    for name in _ENGINE_BEARER_VARS:
        monkeypatch.delenv(name, raising=False)


class _NoWaitLimiter:
    """A rate limiter that never waits, for an authority that never calls out."""

    async def acquire(self) -> None:
        return None


@contextlib.asynccontextmanager
async def _wikidata_unreachable(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Any, None]:
    raise httpx.ConnectError("live Wikidata refused in a unit test")
    yield  # pragma: no cover - unreachable; makes this an async generator


@pytest.fixture(autouse=True)
def no_live_wikidata(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every unit test off the live Wikidata API.

    Subject resolution falls through to the Wikidata authority whenever a name
    is not already in the local store, so any test that extracts or asserts a
    particle without stubbing ``_wikidata_candidates`` used to make real HTTP
    calls. Measured 2026-10-01: 26 tests in five files did, about 70 requests
    per run. Each paid a round trip plus the authority's 2 rps throttle, so the
    three ``normative-rule-file`` cases in test_scope_exemption.py took over
    4 s apiece, and the suite's result depended on Wikidata being up.

    The authority now sees Wikidata as unreachable, which is the path it
    already handles: the search and alias helpers catch the error and return
    nothing, so the subject resolves as a bare local one. The throttle is
    replaced too, because an instantly refused call still waits out the
    limiter first. A test that exercises the authority patches
    ``_wikidata_candidates`` / ``_wikidata_aliases`` or ``particles_client`` itself,
    which overrides this. Tests marked ``integration`` are exempt.
    """
    if request.node.get_closest_marker("integration") is not None:
        return
    from particles.ingest.authorities import wikidata

    monkeypatch.setattr(wikidata, "particles_client", _wikidata_unreachable)
    monkeypatch.setattr(wikidata, "_wikidata_limiter", _NoWaitLimiter)


#: The real ``~/.claude/projects`` of the account running the suite, taken from
#: the password database rather than ``$HOME``: a test that swapped ``HOME``
#: and still reached the real directory is exactly the case to catch.
_REAL_CLAUDE_PROJECTS = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, ".claude", "projects")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_PATH_EVENTS = {"os.mkdir": 1, "os.remove": 1, "os.rmdir": 1, "os.truncate": 1}
_PAIR_EVENTS = {"os.rename", "os.symlink", "os.link"}
#: Writes refused during the running test; ``None`` while no test is running.
_claude_projects_writes: list[str] | None = None


def _under_real_claude_projects(path: object) -> bool:
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False  # an fd-relative call; nothing here writes that way
    absolute = os.path.abspath(os.fsdecode(path))
    return absolute == _REAL_CLAUDE_PROJECTS or absolute.startswith(_REAL_CLAUDE_PROJECTS + os.sep)


def _claude_projects_audit_hook(event: str, args: tuple[object, ...]) -> None:
    """Refuse any write under the real ``~/.claude/projects`` while a test runs.

    ``open`` covers ``io.open`` and ``os.open`` (both pass the OS flags);
    ``os.rename`` covers ``os.replace`` and so the projection's atomic write.
    """
    if _claude_projects_writes is None:
        return
    if event == "open":
        flags = args[2]
        targets = (args[0],) if isinstance(flags, int) and flags & _WRITE_FLAGS else ()
    elif event in _PAIR_EVENTS:
        targets = args[:2]
    elif event in _PATH_EVENTS:
        targets = args[:1]
    else:
        return
    for target in targets:
        if _under_real_claude_projects(target):
            _claude_projects_writes.append(f"{event} {os.fsdecode(target)}")  # type: ignore[arg-type]
            raise PermissionError(f"test wrote under the real {_REAL_CLAUDE_PROJECTS}: {target!r}")


sys.addaudithook(_claude_projects_audit_hook)


@pytest.fixture(autouse=True)
def no_real_claude_projects_writes() -> Generator[None, None, None]:
    """Fail any test that writes under the real ``~/.claude/projects``.

    Those files are the live agent memory of the account running the suite:
    every Claude Code session in a project loads its ``MEMORY.md`` as facts
    about the user. A test once ran ``particles memory consolidate`` against a
    scratch store with the real ``HOME``, and its synthetic beliefs replaced
    three real memory files. The product now refuses that
    (``_claude_code.projection_refusal``); this is the second line.

    The write is refused before it happens (``PermissionError`` from an audit
    hook), and the test fails at teardown naming it, because the projection
    contains its own failures and would otherwise pass. Integration tests are
    **not** exempt: they are the tier that runs the real cycle. A test that
    needs a harness directory builds a fake one under ``tmp_path`` with a fake
    ``HOME`` (``tests/_claude_projects.py``).
    """
    global _claude_projects_writes
    _claude_projects_writes = []
    try:
        yield
    finally:
        refused, _claude_projects_writes = _claude_projects_writes, None
    if refused:
        pytest.fail(
            "wrote under the real ~/.claude/projects (refused): " + "; ".join(refused),
            pytrace=False,
        )


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    # Maintained in one place — see particles/_orm_modules.py. Manually
    # listing modules here drifts whenever a new ORM module lands
    # (e.g. synthesis_cache); the central registry is the
    # single source of truth.
    import particles._orm_modules  # noqa: F401
    from particles.db import Base, get_engine, session_scope

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_scope() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    # Dispose the engine while the event loop is still alive. The autouse
    # ``reset_client_state`` fixture calls ``reset_config()`` →
    # ``reset_engine()`` between tests, which only nulls the cached engine —
    # it does not close the aiosqlite Connection. Without this dispose, the
    # Connection becomes unreachable after pytest-asyncio tears the loop
    # down, and aiosqlite's ``__del__`` raises
    # ``RuntimeError: Event loop is closed`` when it tries to schedule
    # cleanup work on the dead loop.
    await engine.dispose()


@pytest_asyncio.fixture
async def file_db_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[AsyncSession, None]:
    """``db_session`` over a file-backed SQLite store, so a second connection sees it.

    For the tests that ask whether another writer could write mid-pass
    (``tests/_write_probe.py``): an in-memory store is private to
    its connection, and the write lock is a no-op there.
    """
    import particles._orm_modules  # noqa: F401
    from particles.config import reset_config
    from particles.db import Base, get_engine, session_scope

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    reset_config()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        async with session_scope() as session:
            yield session
    finally:
        await engine.dispose()


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """File-based SQLite for CLI tests.

    CLI commands wrap their async impl in ``asyncio.run(...)``, which spins a
    fresh event loop and opens its own session via ``session_scope()``. With
    ``:memory:`` SQLite, each connection gets its own database — state would
    not survive between CLI invocations within a single test. A file-based
    DB shares state across asyncio.run boundaries.

    ``PARTICLES_CONFIG`` is already overridden at session level by
    ``pytest_configure`` so the dev's local ``./config.yaml`` does not leak
    into any test (CLI or otherwise).
    """
    db_path = tmp_path / "cli.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))

    from particles.config import reset_config

    reset_config()

    async def _create_tables() -> None:
        # Maintained in one place — see particles/_orm_modules.py.
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create_tables())
    yield db_path

    # Dispose the engine before reset_config() drops the cached pointer.
    # See the matching note on the db_session fixture above — without this
    # the aiosqlite Connection outlives every loop it was bound to and its
    # __del__ raises RuntimeError: Event loop is closed.
    async def _dispose() -> None:
        from particles.db import get_engine

        await get_engine().dispose()

    asyncio.run(_dispose())
    reset_config()
