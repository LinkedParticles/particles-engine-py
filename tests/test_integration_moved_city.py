# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Live run of the moved-city scenario (integration tier).

The market's own example of memory rot: a user tells an agent where they live
and names a favourite restaurant "a ten-minute walk from my flat", then moves
to another city. The agent later recommends the old restaurant as a short walk
from the new flat, because the stored walk-distance claim said "the user's
flat" and nothing in the store placed the restaurant anywhere.

The fixture (``tests/fixtures/moved_city/``, synthetic) is two sessions. This
test harvests session 4 with ``particles audit``, then adds session 7 and
audits again, the same path the SessionEnd harvest takes, and reads the store.

What is pinned, and who owns each link:

* **Passing (the extractor's link).** The store holds an ACTIVE claim placing
  Sandeep's Curry House in Delhi or Lajpat Nagar, and no ACTIVE claim ties the
  restaurant to the user's flat without naming where that flat is. The general
  extractor resolves "my flat" against the same source before it writes the
  claim, and emits the locating claim the source implies.
* **Passing (update supersession).** The move still supersedes "the user lives
  in Delhi" (a claim that lost its speaker subject would never be paired with
  it), and a lasting preference ("favourite place to eat") survives a
  situational statement ("has not found a regular place to eat yet"), which
  update supersession gated on a same-slot probe in 1.157.0.
* **Passing (re-anchoring).** When "the user lives in Delhi" is
  superseded, a claim whose truth rested on it (the walk distance from the
  user's Lajpat Nagar flat) is restated by the nightly cycle, anchored to the
  old flat and dated, rather than staying ACTIVE in the present tense. The
  fixture runs one ``particles memory consolidate`` after the second harvest,
  since the audit never runs the maintenance passes.

The assertions read claim text, which is model behaviour; they are kept to the
entity and place names the scenario is about, never to wording or counts. Run
by hand, with the store and a query answer printed, via
``uv run python scripts/scenario_moved_city.py``.
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

FIXTURE = Path(__file__).parent / "fixtures" / "moved_city"

_RESTAURANT = "sandeep"
_OLD_PLACE = ("delhi", "lajpat nagar")
# Words that tie a claim to where the user lives.
_HOME = ("flat", "home", "apartment", "walk", "where the user lives", "near the user")
_FAVOURITE = ("favourite", "favorite")
# Words that present a residence as past rather than current.
_PAST = ("former", "previous", "used to", "old flat", "lived", "until", "before the move")


def _active_claims(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT content FROM particles WHERE status = 'ACTIVE'").fetchall()
    return [str(r[0]) for r in rows]


def _all_claims(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT content FROM particles").fetchall()
    return [str(r[0]) for r in rows]


def _has(text: str, words: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(w in low for w in words)


@pytest.fixture(scope="module")
def moved_city_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Audit session 4, then sessions 4 and 7, into one scratch store.

    Module-scoped so the three assertions below share one pair of extraction
    calls (cost discipline per tests/AGENTS.md). ``cli_db`` is function-scoped,
    so the store wiring is repeated here with a module-level monkeypatch.
    """
    import asyncio

    work = tmp_path_factory.mktemp("moved_city")
    db_path = work / "particles.db"
    notes = work / "notes"
    notes.mkdir()

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    mp.setenv("PARTICLES_BLOB_DIR", str(work / "blobs"))
    # A scratch store must never reach the real ~/.claude/projects: a fake HOME,
    # its own state directory, and the MEMORY.md projection off.
    isolate_claude_code(mp, work)
    try:
        reset_config()

        async def _create_tables() -> None:
            # Maintained in one place; see particles/_orm_modules.py.
            import particles._orm_modules  # noqa: F401
            from particles.db import Base, get_engine

            async with get_engine().begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        asyncio.run(_create_tables())

        runner = CliRunner()
        shutil.copy(FIXTURE / "session-04.md", notes / "session-04.md")
        first = runner.invoke(app, ["audit", str(notes), "--yes"])
        assert first.exit_code == 0, first.output

        shutil.copy(FIXTURE / "session-07.md", notes / "session-07.md")
        second = runner.invoke(app, ["audit", str(notes), "--yes"])
        assert second.exit_code == 0, second.output

        # The audit is a census and never runs the maintenance passes. The
        # nightly cycle is what a harness store takes next, and it is where
        # the re-anchor pass restates a claim whose state was retired.
        cycle = runner.invoke(app, ["memory", "consolidate", "--scope", "store"])
        assert cycle.exit_code == 0, cycle.output
    finally:
        # The same teardown as ``cli_db``: dispose before reset_config() drops
        # the cached engine, or the aiosqlite connection outlives its loop.
        async def _dispose() -> None:
            from particles.db import get_engine

            await get_engine().dispose()

        asyncio.run(_dispose())
        mp.undo()
        reset_config()
    return db_path


def test_restaurant_is_placed_in_delhi(moved_city_store: Path) -> None:
    active = _active_claims(moved_city_store)
    placed = [c for c in active if _has(c, (_RESTAURANT,)) and _has(c, _OLD_PLACE)]
    assert placed, f"no ACTIVE claim places Sandeep's Curry House in Delhi: {active}"


def test_no_active_claim_ties_restaurant_to_an_unnamed_flat(moved_city_store: Path) -> None:
    active = _active_claims(moved_city_store)
    unanchored = [
        c for c in active if _has(c, (_RESTAURANT,)) and _has(c, _HOME) and not _has(c, _OLD_PLACE)
    ]
    assert not unanchored, f"ACTIVE claims tie the restaurant to an unnamed flat: {unanchored}"


def test_old_residence_is_superseded(moved_city_store: Path) -> None:
    active = _active_claims(moved_city_store)
    residence = [
        c
        for c in active
        if _has(c, ("lives in", "living in"))
        and _has(c, _OLD_PLACE)
        and not _has(c, (_RESTAURANT,))
    ]
    assert not residence, f"the Delhi residence is still ACTIVE after the move: {residence}"


def test_favourite_survives_a_situational_statement(moved_city_store: Path) -> None:
    active = _active_claims(moved_city_store)
    favourite = [c for c in active if _has(c, (_RESTAURANT,)) and _has(c, _FAVOURITE)]
    assert favourite, f"the favourite-restaurant claim was retired: {active}"


def test_dependent_claim_is_reexamined_after_the_move(moved_city_store: Path) -> None:
    walk = [c for c in _all_claims(moved_city_store) if _has(c, (_RESTAURANT,)) and _has(c, _HOME)]
    assert walk, "precondition: the walk-distance claim was never extracted"
    stale = [
        c
        for c in _active_claims(moved_city_store)
        if _has(c, (_RESTAURANT,)) and _has(c, _HOME) and not _has(c, _PAST)
    ]
    assert not stale, f"claims resting on the superseded residence are still ACTIVE: {stale}"
