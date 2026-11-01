# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""A savepoint outside ``write_transaction`` nests in a real transaction (2026-10-28-2 review).

AA5 and AA56: the usefulness credit and its event share one transaction, so a
press whose event write fails leaves no credit; a web-clipper scan under a
plain session that fails after a capture keeps nothing. Each runs on the
store's real aiosqlite engine.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_a_useful_press_whose_event_write_fails_leaves_no_credit(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credit and its event share one transaction: an event failure leaves no credit row."""

    async def _go() -> int:
        from sqlalchemy import func, select

        import particles.operations.utility_feedback as uf
        from particles.core.schema import (
            CalibrationSource,
            Confidence,
            Particle,
            ProvenanceRef,
            ProvenanceRefType,
            UncertaintyNature,
        )
        from particles.db import get_engine, session_scope
        from particles.store.particle_store import insert_particle
        from particles.store.utility_store import UtilityEventRow

        p = Particle(
            content="A claim.",
            confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="t",
            provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e")],
        )
        async with session_scope() as s:
            await insert_particle(s, p, embedding=[0.0] * 384)
            await s.commit()

        async def _boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("event write failed")

        monkeypatch.setattr(uf, "record_event", _boom)
        async with session_scope(write=True) as s:
            with pytest.raises(RuntimeError):
                await uf.mark_belief_useful(s, p.id, actor="http", operator=True)
            await s.rollback()
        async with session_scope() as s:
            n = (await s.execute(select(func.count()).select_from(UtilityEventRow))).scalar_one()
        await get_engine().dispose()
        return int(n)

    assert asyncio.run(_go()) == 0, "a credit row was committed without its event"


def test_a_useful_press_that_fails_before_commit_keeps_no_credit(cli_db: Path) -> None:
    """A credit recorded first in a plain write session is undone when the session fails."""
    from sqlalchemy import select

    from particles.db import get_engine, session_scope
    from particles.store.utility_store import UtilityEventRow, record_explicit_utility_event

    async def go() -> int:
        try:
            try:
                async with session_scope(write=True) as session:
                    await record_explicit_utility_event(
                        session, "p-0000", "explicit:operator:2026-10-28", None
                    )
                    raise RuntimeError("the event write fails")
            except RuntimeError:
                pass
            async with session_scope() as session:
                return len((await session.execute(select(UtilityEventRow.id))).all())
        finally:
            await get_engine().dispose()

    assert asyncio.run(go()) == 0


def test_a_web_clipper_import_that_fails_after_its_first_capture_keeps_nothing(
    cli_db: Path, tmp_path: Path
) -> None:
    """A scan under a plain session that fails before its commit leaves the store as it was."""
    from sqlalchemy import select

    from particles.corpus.deposit import deposit_web_clipper
    from particles.corpus.store import CorpusEntryRow
    from particles.db import get_engine, session_scope

    captures = tmp_path / "caps"
    captures.mkdir()
    (captures / "a.md").write_text("---\nsource: https://example.org/a\n---\nbody a\n")

    async def go() -> int:
        try:
            try:
                async with session_scope() as session:
                    await deposit_web_clipper(session, captures, deposited_by="web-clipper")
                    raise RuntimeError("the run fails before its commit")
            except RuntimeError:
                pass
            async with session_scope() as session:
                return len((await session.execute(select(CorpusEntryRow.entry_id))).all())
        finally:
            await get_engine().dispose()

    assert asyncio.run(go()) == 0
