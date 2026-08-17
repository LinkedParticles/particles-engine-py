"""Prefix the engine-side annotation ``properties`` keys.

Renames three bare keys inside ``particles.properties_json`` to the
``extraction:`` namespace the spec requires:

* ``polarity``     -> ``extraction:polarity``      (cap. 1)
* ``scope``        -> ``extraction:scope``
* ``scope_action`` -> ``extraction:scope_action``

**Why the data must move, not just the constants.** ``polarity`` and ``scope``
are *visibility* levers: ``is_non_asserted`` and ``is_excluded_document_meta``
are what keep DECLINED / HYPOTHETICAL / DOCUMENT_META particles off the default
factual surface (query, projection, export, §6.6, L-SEM-01, L-IDX-01). Those
predicates read the prefixed key only, so a store that upgraded the package
without this migration would silently surface every already-stored non-asserted
and document-meta particle. Nothing errors; the results are just wrong. That
silent failure mode is the reason this is a data migration rather than a
dual-read in the predicates (and § Alternatives for the rejected
one).

**Why it is not scoped by extractor id.** ``general.py::_call_llm`` is the
default carry-forward seam (``extraction/incremental.py``), so the Reddit,
Hacker News, GitHub repo/gist/Pages, and Mastodon extractors emit these keys
too. Filtering on ``asserted_by = 'general-extractor'`` would leave most of the
population behind. The rewrite therefore keys on the property name alone.

**Scope of the edit.** Only the three exact top-level keys, only when present,
value preserved verbatim; every other key, and the JSON of every row without
one of them, is untouched. A row already carrying the prefixed key keeps it
(the legacy value is dropped rather than clobbering the current one) — a row
holding both spellings is malformed, and the prefixed one is what the
predicates mean. ``downgrade()`` is the exact inverse.

Revision ID: 035
Revises: 034
"""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa

from alembic import op

revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None

# Inlined rather than imported from ``particles.extraction`` on purpose: a
# migration is a frozen record of one schema step, and must keep describing
# this rename even after the application constants move again.
_RENAMES = {
    "polarity": "extraction:polarity",
    "scope": "extraction:scope",
    "scope_action": "extraction:scope_action",
}


def rewrite_properties_keys(bind: sa.engine.Connection, mapping: dict[str, str]) -> None:
    """Rename ``mapping``'s keys inside every ``particles.properties_json``.

    Takes the connection rather than calling ``op.get_bind()`` internally so the
    rewrite is exercisable outside an Alembic context — this migration edits
    operator data, which the unit suite (``create_all`` on an in-memory DB)
    otherwise never covers. See ``tests/test_migration_035_properties_keys.py``.
    """
    rows = bind.execute(
        sa.text(
            "SELECT id, properties_json FROM particles "
            "WHERE properties_json IS NOT NULL AND properties_json != ''"
        )
    ).fetchall()

    for particle_id, raw in rows:
        try:
            properties = json.loads(raw)
        except (TypeError, ValueError):
            # Unparseable JSON is left exactly as found: this migration renames
            # keys, and repairing malformed rows is not its job.
            continue
        if not isinstance(properties, dict):
            continue
        if not mapping.keys() & properties.keys():
            continue

        renamed: dict[str, Any] = dict(properties)
        for old, new in mapping.items():
            if old in renamed:
                value = renamed.pop(old)
                renamed.setdefault(new, value)

        bind.execute(
            sa.text("UPDATE particles SET properties_json = :props WHERE id = :id"),
            {"props": json.dumps(renamed, ensure_ascii=False), "id": particle_id},
        )


def upgrade() -> None:
    rewrite_properties_keys(op.get_bind(), _RENAMES)


def downgrade() -> None:
    rewrite_properties_keys(op.get_bind(), {new: old for old, new in _RENAMES.items()})
