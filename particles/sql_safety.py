"""SQL-injection-adjacent safety helpers.

Currently just ``escape_like_pattern`` — SQLAlchemy already parameterises
the value passed to ``column.like(...)`` (so the value cannot be SQL
injection), but ``%`` and ``_`` in user input are still LIKE wildcards
that expand the match. A user typing ``%`` as a prefix would match every
row; ``_`` would match any single character. Both are dialect-portable
escape concerns rather than injection vulnerabilities, but worth fixing
for behavioural correctness on user input.

Usage:

    from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern

    prefix_clean = escape_like_pattern(user_input)
    col.like(f"{prefix_clean}%", escape=LIKE_ESCAPE)
"""

from __future__ import annotations

# Backslash is the SQLAlchemy default and is portable across SQLite + Postgres.
LIKE_ESCAPE = "\\"


def escape_like_pattern(value: str) -> str:
    """Escape ``%``, ``_``, and the escape character itself in a LIKE pattern.

    The caller is responsible for adding the surrounding wildcards (e.g.
    ``f"{escape_like_pattern(prefix)}%"``) and passing ``escape=LIKE_ESCAPE``
    to ``column.like(...)``.
    """
    # Escape the escape character first so we don't double-escape the others.
    return (
        value.replace(LIKE_ESCAPE, LIKE_ESCAPE + LIKE_ESCAPE)
        .replace("%", LIKE_ESCAPE + "%")
        .replace("_", LIKE_ESCAPE + "_")
    )
