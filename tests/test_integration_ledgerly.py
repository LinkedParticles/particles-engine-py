# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Live run of the Ledgerly scenario (integration tier).

The second moved-fact case, proposed on the r/AIMemory Rung 1 thread: a
project's notes say it runs on PostgreSQL 16, that the tests need the database
container started first, and that the migrations use JSONB; a later note says
the project moved to SQLite. The sibling of the moved-city scenario
(``tests/test_integration_moved_city.py``), with a dependence of another kind:
the setup steps name no relative reference ("my flat"); they refer to the
old database by description ("the database container").

The fixture (``tests/fixtures/ledgerly/``, synthetic) is two sessions, run the
way that report ran them: audit session 3, add session 9 and audit again, then
one ``particles memory consolidate --scope store``.

What is pinned, and who owns each link:

* **Passing (update supersession).** "Ledgerly uses PostgreSQL 16" is no
  longer ACTIVE after the move, whether extraction writes a "uses SQLite"
  claim or only "was moved from PostgreSQL to SQLite": a reported move gives
  the slot its new value, confirmed by a second reading before it
  retires anything.
* **Passing (write order, 1.172.2).** No claim about the single-file database
  is held ``CONFLICT_PENDING`` behind the PostgreSQL claim. On 1.172.0 the
  extractor emitted it before the claim that updates the database, it met the
  PostgreSQL claim while that was still ACTIVE, and it stayed quarantined
  after the update retired its other side.
* **Expected to fail (a recorded limit).** No ACTIVE claim
  presents ``docker compose up -d db`` as a current step. The re-anchor pass
  reaches a claim that names the retired database by description, but the
  models judge a step about the container that runs it a procedure not tied
  to the database engine, so it stays current. Non-strict, since
  a sample that restates it is not a failure.
* **Passing (the passage, not world knowledge).** The JSONB claim stays
  ACTIVE. The passage ties it to no database; that JSONB is a PostgreSQL
  column type is world knowledge, which never makes a claim dependent.

The assertions read claim text, which is model behaviour; they are kept to the
command, the file name and the engine names, never to wording or counts.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.config import reset_config
from particles.secrets import get_anthropic_api_key_optional
from tests._claude_projects import isolate_claude_code

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        get_anthropic_api_key_optional() is None,
        reason="integration tier requires ANTHROPIC_API_KEY (tests/AGENTS.md § Integration tests)",
    ),
]

FIXTURE = Path(__file__).parent / "fixtures" / "ledgerly"

_COMMAND = "docker compose up -d db"
_ONE_FILE = ("ledgerly.db", "single file")
# Words that present a setup step as past rather than current.
# Only the markers a dated restatement carries: a present-tense claim can say
# "was" or "were" ("pytest fails when the container was not started").
_PAST = ("as of", "when ledgerly used")


def _rows(db_path: Path) -> list[tuple[str, str, str | None]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT content, status, status_reason FROM particles").fetchall()
    return [(str(c), str(s), r) for c, s, r in rows]


def _has(text: str, words: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(w in low for w in words)


@pytest.fixture(scope="module")
def ledgerly_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Audit session 3, then sessions 3 and 9, then one consolidation cycle.

    Module-scoped so the assertions share one set of LLM calls (cost
    discipline per tests/AGENTS.md); the store wiring mirrors the moved-city
    fixture.
    """
    import asyncio

    work = tmp_path_factory.mktemp("ledgerly")
    db_path = work / "particles.db"
    notes = work / "notes"
    notes.mkdir()

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    mp.setenv("PARTICLES_BLOB_DIR", str(work / "blobs"))
    # A scratch store must never reach the real ~/.claude/projects.
    isolate_claude_code(mp, work)
    try:
        reset_config()

        async def _create_tables() -> None:
            import particles._orm_modules  # noqa: F401
            from particles.db import Base, get_engine

            async with get_engine().begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        asyncio.run(_create_tables())

        runner = CliRunner()
        shutil.copy(FIXTURE / "session-03.md", notes / "session-03.md")
        first = runner.invoke(app, ["audit", str(notes), "--yes"])
        assert first.exit_code == 0, first.output

        shutil.copy(FIXTURE / "session-09.md", notes / "session-09.md")
        second = runner.invoke(app, ["audit", str(notes), "--yes"])
        assert second.exit_code == 0, second.output

        cycle = runner.invoke(app, ["memory", "consolidate", "--scope", "store"])
        assert cycle.exit_code == 0, cycle.output
    finally:

        async def _dispose() -> None:
            from particles.db import get_engine

            await get_engine().dispose()

        asyncio.run(_dispose())
        mp.undo()
        reset_config()
    return db_path


def test_postgres_is_no_longer_current(ledgerly_store: Path) -> None:
    current = [
        c
        for c, status, _ in _rows(ledgerly_store)
        if status == "ACTIVE" and _has(c, ("uses postgresql",))
    ]
    assert not current, f"the PostgreSQL claim is still ACTIVE after the move: {current}"


def test_the_single_file_claim_is_not_held(ledgerly_store: Path) -> None:
    held = [
        c
        for c, _, reason in _rows(ledgerly_store)
        if reason == "CONFLICT_PENDING" and _has(c, _ONE_FILE)
    ]
    assert not held, f"the newer source's claim is quarantined: {held}"


@pytest.mark.xfail(
    strict=False,
    reason="a step about the container that runs a retired value stays current",
)
def test_the_container_step_is_not_current(ledgerly_store: Path) -> None:
    rows = _rows(ledgerly_store)
    assert any(_has(c, (_COMMAND,)) for c, _, _ in rows), (
        "precondition: the container step was never extracted"
    )
    current = [
        c
        for c, status, _ in rows
        if status == "ACTIVE" and _has(c, (_COMMAND,)) and not _has(c, _PAST)
    ]
    assert not current, f"ACTIVE claims present the PostgreSQL-era step as current: {current}"


def test_the_jsonb_claim_is_left_current(ledgerly_store: Path) -> None:
    rows = _rows(ledgerly_store)
    jsonb = [c for c, _, _ in rows if _has(c, ("jsonb",)) and not _has(c, ("postgresql",))]
    assert jsonb, "precondition: the JSONB claim was never extracted"
    current = [c for c, status, _ in rows if status == "ACTIVE" and c in jsonb]
    assert current, f"the JSONB claim was re-anchored on world knowledge: {jsonb}"
