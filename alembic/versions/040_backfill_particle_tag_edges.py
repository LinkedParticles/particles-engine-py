# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Backfill ``particle_tag_edges`` from ``particles.tags_json``.

A particle's tags live in two places: ``particles.tags_json`` (the canonical
list, round-tripped to ``Particle.tags``) and ``particle_tag_edges`` (one row
per tag, the index every tag-filtered read joins on). Until this release
``insert_particle`` wrote only the JSON column; the edge rows came only from
``set_particle_tags`` / ``add_particle_tags``. A particle born with tags (an
agent assertion with ``tags``, a supersession or subject-assign successor
inheriting its predecessor's tags, a conversation deposit) was therefore
invisible to every tag-filtered query.

**The backfill.** For every particle with a non-empty ``tags_json``, insert the
edge rows it is missing. Rows already present are left alone, so the migration
is idempotent and safe on a store where some particles were retagged by hand.
Malformed JSON is skipped, not repaired: the column is read as-is by
``ParticleRow.to_model``, and this migration does not change what a particle
says about itself.

``downgrade()`` is a deliberate no-op. A backfilled edge cannot be told apart
from one ``set_particle_tags`` wrote, and each one agrees with the particle's
own tag list, so removing them would only reintroduce the defect.

Revision ID: 040
Revises: 039
"""

from __future__ import annotations

import json

import sqlalchemy as sa

from alembic import op

revision = "040"
down_revision = "039"
branch_labels = None
depends_on = None


def backfill_tag_edges(bind: sa.engine.Connection) -> int:
    """Insert the edge rows each tagged particle is missing.

    Takes the connection rather than calling ``op.get_bind()`` so the backfill
    is exercisable outside an Alembic context, as in migration 037. See
    ``tests/test_migration_040_tag_edges.py``.

    Returns the number of edge rows inserted.
    """
    existing = {
        (pid, tag)
        for pid, tag in bind.execute(
            sa.text("SELECT particle_id, tag FROM particle_tag_edges")
        ).fetchall()
    }
    rows = bind.execute(
        sa.text("SELECT id, tags_json FROM particles WHERE tags_json IS NOT NULL")
    ).fetchall()
    missing: list[dict[str, str]] = []
    for pid, tags_json in rows:
        try:
            tags = json.loads(tags_json)
        except (TypeError, ValueError):
            continue
        if not isinstance(tags, list):
            continue
        for tag in dict.fromkeys(t for t in tags if isinstance(t, str) and t):
            if (pid, tag) not in existing:
                existing.add((pid, tag))
                missing.append({"pid": pid, "tag": tag})
    if missing:
        bind.execute(
            sa.text("INSERT INTO particle_tag_edges (particle_id, tag) VALUES (:pid, :tag)"),
            missing,
        )
    return len(missing)


def upgrade() -> None:
    backfill_tag_edges(op.get_bind())


def downgrade() -> None:
    pass
