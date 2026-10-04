# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Mark rows born retired, so the as-of lens never reads them as beliefs.

One non-null boolean column on ``particles``:

* ``born_retired`` — true for a row inserted in a status other than
  ``ACTIVE``: an INCONSISTENCY record, or the losing candidate of a §6.6
  INCONSISTENT verdict persisted ``PROVENANCE_STALE`` / ``CONFLICT_PENDING``.
  ``insert_particle`` stamps it from this revision on.

Why a column. The as-of lens hid these rows by their
current status and reason (``INCONSISTENCY``, ``CONFLICT_PENDING``), and
resolving a conflict changes both. The loser moves to ``CONFLICT_RESOLVED``
(PREFER_A, the trust cascade), ``SUPERSEDED`` (promotion) or ``RETRACTED``
(DISCARD); the record moves to ``RETRACTED`` or ``PROVENANCE_STALE``. Each then
looked like a retired belief, dated from the review, so an as-of query between
its birth and the review returned a claim the store never held. Nothing else
on the row survives the resolution: ``retired_at`` stays NULL (the row never
left ACTIVE), but it is NULL on pre-029 retirements too.

**The backfill** is exact on every store this SDK wrote. It marks:

1. every row still ``INCONSISTENCY`` or ``CONFLICT_PENDING``;
2. every resolved INCONSISTENCY record: content opening ``INCONSISTENCY: ``
   (the one template prefix every version has written), not ACTIVE, no
   ``retired_at``;
3. every row an INCONSISTENCY record, open or resolved, names as its B claim
   (the second PARTICLE provenance ref), when the row exists, is not ACTIVE,
   and has no ``retired_at``. Since the only writer of an
   INCONSISTENCY record is the quarantine plan, and its B is the quarantined
   candidate; before the B ref was a dangling id, so it matches no
   row.

The ``retired_at`` guard skips any row stamped on leaving ACTIVE, which a
born-retired row never is.

``downgrade()`` drops the column.

Revision ID: 042
Revises: 041
"""

from __future__ import annotations

import json

import sqlalchemy as sa

from alembic import op

revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None

#: The content prefix of every INCONSISTENCY record template this SDK has
#: written (``core.conflict_resolution.build_inconsistency_particle``).
_RECORD_PREFIX = "INCONSISTENCY: "


def _b_claim_id(provenance_json: str | None) -> str | None:
    """The B claim an INCONSISTENCY record names: its second PARTICLE ref."""
    try:
        refs = json.loads(provenance_json or "[]")
    except (TypeError, ValueError):
        return None
    if not isinstance(refs, list):
        return None
    particle_refs = [
        r.get("corpus_entry_id")
        for r in refs
        if isinstance(r, dict) and r.get("type") == "PARTICLE"
    ]
    b = particle_refs[1] if len(particle_refs) > 1 else None
    return b if isinstance(b, str) else None


def backfill_born_retired(bind: sa.engine.Connection) -> int:
    """Set ``born_retired`` on every born-retired row already stored.

    Takes the connection rather than calling ``op.get_bind()`` so the backfill
    is exercisable outside an Alembic context, as in migration 040. See
    ``tests/test_migration_042_born_retired.py``.

    Returns the number of rows marked.
    """
    marked = {
        pid
        for (pid,) in bind.execute(
            sa.text(
                "SELECT id FROM particles"
                " WHERE status = 'INCONSISTENCY' OR status_reason = 'CONFLICT_PENDING'"
            )
        ).fetchall()
    }
    # Rows that left no trace of ever being believed: not ACTIVE, never stamped.
    unstamped = bind.execute(
        sa.text(
            "SELECT id, content, provenance_json FROM particles"
            " WHERE status != 'ACTIVE' AND retired_at IS NULL"
        )
    ).fetchall()
    unstamped_ids = {pid for pid, _content, _prov in unstamped}
    records = [
        (pid, prov) for pid, content, prov in unstamped if content.startswith(_RECORD_PREFIX)
    ]
    marked.update(pid for pid, _prov in records)
    open_records = bind.execute(
        sa.text("SELECT provenance_json FROM particles WHERE status = 'INCONSISTENCY'")
    ).fetchall()
    record_provenance = [prov for (prov,) in open_records] + [prov for _pid, prov in records]
    b_ids = {b for prov in record_provenance if (b := _b_claim_id(prov)) is not None}
    marked |= b_ids & unstamped_ids
    if marked:
        bind.execute(
            sa.text("UPDATE particles SET born_retired = :t WHERE id = :pid"),
            [{"t": True, "pid": pid} for pid in sorted(marked)],
        )
    return len(marked)


def upgrade() -> None:
    with op.batch_alter_table("particles") as batch:
        batch.add_column(
            sa.Column(
                "born_retired",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
    backfill_born_retired(op.get_bind())


def downgrade() -> None:
    with op.batch_alter_table("particles") as batch:
        batch.drop_column("born_retired")
