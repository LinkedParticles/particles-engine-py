# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Read SQLite's plan for the statements a store function actually sends.

A plan test that retypes the SQL tests the retyped copy. :func:`plans_of`
records every statement ``call`` executes on the session's engine, then asks
SQLite for each one's ``EXPLAIN QUERY PLAN`` with the same parameters.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession


async def plans_of(
    session: AsyncSession, call: Callable[[], Awaitable[Any]]
) -> list[tuple[str, list[str]]]:
    """``(statement, plan detail lines)`` for each SELECT ``call`` runs."""
    engine = session.get_bind()
    sent: list[tuple[str, Any]] = []

    def record(_conn: Any, _cursor: Any, statement: str, parameters: Any, *_: Any) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            sent.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", record)
    try:
        await call()
    finally:
        event.remove(engine, "before_cursor_execute", record)
    out = []
    conn = await session.connection()
    for statement, parameters in sent:
        rows = await conn.exec_driver_sql(f"EXPLAIN QUERY PLAN {statement}", parameters)
        out.append((statement, [row[-1] for row in rows]))
    return out
