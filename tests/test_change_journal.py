# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The particle change journal: what its triggers record, and its readers."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from particles.store.change_journal import (
    JOURNAL_RETENTION,
    changed_since,
    journal_installed,
    read_head,
    token_at,
)


async def _entries(session: AsyncSession) -> list[str]:
    rows = await session.execute(text("SELECT particle_id FROM particle_changes ORDER BY seq"))
    return list(rows.scalars())


async def _insert(session: AsyncSession, pid: str) -> None:
    await session.execute(
        text(
            "INSERT INTO particles (id, content, confidence_value, confidence_calibration_source, "
            "uncertainty_nature, asserted_by, asserted_at, status, schema_version, "
            "particle_type, assertion_modality, provenance_json, subject_ids_json, "
            "canonical_form, born_retired) VALUES (:id, 'c', 0.5, 'EXTRACTOR_DIRECT', "
            "'EPISTEMIC', 't', '2026-10-17', 'ACTIVE', '1.0.0', 'CLAIM', 'FALSIFIABLE', "
            "'[]', '[]', 'PROSE', 0)"
        ),
        {"id": pid},
    )


async def test_a_new_store_has_a_marker(db_session: AsyncSession) -> None:
    assert await journal_installed(db_session)
    assert await _entries(db_session) == [""]
    head = await read_head(db_session)
    assert head is not None and len(head[1]) == 8


async def test_every_write_names_its_row(db_session: AsyncSession) -> None:
    await _insert(db_session, "p1")
    await _insert(db_session, "p2")
    await db_session.execute(text("UPDATE particles SET confidence_value = 0.6 WHERE id = 'p1'"))
    await db_session.execute(text("UPDATE particles SET id = 'p3' WHERE id = 'p2'"))
    await db_session.execute(text("DELETE FROM particles WHERE id = 'p1'"))
    assert await _entries(db_session) == ["", "p1", "p2", "p1", "p2", "p3", "p1"]


async def test_tokens_are_fresh_per_entry(db_session: AsyncSession) -> None:
    for i in range(20):
        await _insert(db_session, f"p{i}")
    tokens = (await db_session.execute(text("SELECT token FROM particle_changes"))).scalars()
    assert len(set(tokens)) == 21


async def test_readers(db_session: AsyncSession) -> None:
    start = await read_head(db_session)
    assert start is not None
    for pid in ("a", "b", "c"):
        await _insert(db_session, pid)
    await db_session.execute(text("UPDATE particles SET content = 'x' WHERE id = 'a'"))
    assert await token_at(db_session, start[0]) == start[1]
    assert await token_at(db_session, 10_000) is None
    assert await changed_since(db_session, start[0], limit=3) == {"a", "b", "c"}
    assert await changed_since(db_session, start[0], limit=2) is None
    # The marker entry's empty id is never a change.
    assert await changed_since(db_session, 0, limit=3) == {"a", "b", "c"}


async def test_missing_triggers_turn_it_off(db_session: AsyncSession) -> None:
    await db_session.execute(text("DROP TRIGGER trg_particle_changes_prune"))
    assert not await journal_installed(db_session)


async def test_the_journal_keeps_only_its_newest_entries(db_session: AsyncSession) -> None:
    await db_session.execute(
        text(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < :k) "
            "INSERT INTO particle_changes (particle_id, token) SELECT 'x', randomblob(8) FROM n"
        ),
        {"k": JOURNAL_RETENTION + 50},
    )
    count = (await db_session.execute(text("SELECT count(*) FROM particle_changes"))).scalar_one()
    assert count == JOURNAL_RETENTION
