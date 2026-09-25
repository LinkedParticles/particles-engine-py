# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Derived views are not stored as columns (D1, gated per D6).

Walks every table registered on ``Base.metadata`` and fails on a column whose
*name* marks a read-time derived view: ``effective_confidence``,
``rank_score``, or an observer-scope field.

The check reads **names only**. A derived value stored under an unrelated name
passes it, which is why D6 records this principle as partly checked.

The stored embedding (``particles.embedding_json`` plus the
``embedding_model_id`` that keys it) is the one sanctioned derived column on a
record row D1's boundary cases, and is allowlisted below. Any
other ``embedding*`` column fails, so a second one needs its own ruling.
"""

from __future__ import annotations

import re

import sqlalchemy as sa

import particles._orm_modules  # noqa: F401  (registers every ORM table)
from particles.db import Base

# Column-name fragments that mark a derived view.
DERIVED_NAME_PATTERN = re.compile(
    r"effective_confidence|rank_score|observer_scope|in_view|visible_to|embedding"
)

# The one sanctioned derived column on a record row (D1: the stored
# embedding is a disposable cache keyed on its model id, rebuildable by
# re-encoding the claim).
SANCTIONED = {
    ("particles", "embedding_json"),
    ("particles", "embedding_model_id"),
}


def _derived_columns(metadata: sa.MetaData) -> list[str]:
    return [
        f"{table.name}.{column.name}"
        for table in metadata.sorted_tables
        for column in table.columns
        if DERIVED_NAME_PATTERN.search(column.name) and (table.name, column.name) not in SANCTIONED
    ]


def test_no_derived_view_stored_as_column() -> None:
    offenders = _derived_columns(Base.metadata)
    assert not offenders, (
        "Derived views must be computed at read time, not stored (D1): " + ", ".join(offenders)
    )


def test_sanctioned_embedding_columns_still_exist() -> None:
    # A stale allowlist entry would silently widen the exception's name space.
    present = {(t.name, c.name) for t in Base.metadata.sorted_tables for c in t.columns}
    assert present >= SANCTIONED


def test_check_is_not_vacuous() -> None:
    metadata = sa.MetaData()
    sa.Table(
        "particles",
        metadata,
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("embedding_json", sa.Text),
        sa.Column("effective_confidence", sa.Float),
        sa.Column("cached_rank_score", sa.Float),
        sa.Column("observer_scope_json", sa.Text),
    )
    sa.Table("other", metadata, sa.Column("embedding_json", sa.Text))
    assert _derived_columns(metadata) == [
        "other.embedding_json",
        "particles.effective_confidence",
        "particles.cached_rank_score",
        "particles.observer_scope_json",
    ]
