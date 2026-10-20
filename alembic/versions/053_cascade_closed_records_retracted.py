# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Retract the INCONSISTENCY records the trust cascade closed as PROVENANCE_STALE.

Data only; no schema change. The §6.6 wrapper-terminal rule says
every non-DEFER resolution retracts the record (``RETRACTED`` /
``CONFLICT_RESOLVED``). Review, the held-claim release and the disclosure
sweep always did; the trust cascade (Extension B) moved the record
to ``PROVENANCE_STALE`` / ``CONFLICT_RESOLVED`` instead, through an
``INCONSISTENCY → PROVENANCE_STALE`` edge §6.6 does not list for a record.
From 1.192.15 the cascade retracts, and the edge is gone from the transition
table, so this moves the records it already closed to the state review writes.

**Which rows.** A record is a ``CLAIM`` like any other (there is no
INCONSISTENCY particle type), and a demoted loser of the same cascade carries
the same status and reason, so the status pair alone cannot select records.
A row is moved when all of these hold:

1. ``status = 'PROVENANCE_STALE'`` and ``status_reason = 'CONFLICT_RESOLVED'``;
2. its content opens ``INCONSISTENCY: ``, the one prefix every record template
   has written (the discriminator migration 042 uses);
3. ``born_retired`` is set and ``retired_at`` is NULL: the row was never
   believed. A believed loser was stamped when it left ACTIVE.

The cascade was the only writer of that state on a record at every version
(review has retracted since and left the record open before it).

**What is kept.** Only ``status`` changes. ``status_reason`` stays
``CONFLICT_RESOLVED``, ``retired_at`` stays NULL (write-once; a
record never left ACTIVE, so there is nothing to stamp), and ``born_retired``
stays set, which is what the as-of lens reads: a record is never visible at
any instant, before this migration or after it.

**Idempotent.** A second run finds no row in the old state.

**``downgrade()`` is a no-op**, by decision. The older code reads a retracted
record exactly as it reads one review closed, which is the state this
migration writes, so nothing downstream of a downgrade needs the old status
back. Restoring it would reintroduce the state the specification forbids.

Revision ID: 053
Revises: 052
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "053"
down_revision = "052"
branch_labels = None
depends_on = None

#: The content prefix of every INCONSISTENCY record template this SDK has
#: written (``core.conflict_resolution.build_inconsistency_particle`` and the
#: census record), as in migration 042.
_RECORD_PREFIX = "INCONSISTENCY: "


def retract_cascade_closed_records(bind: sa.engine.Connection) -> int:
    """Move every cascade-closed record to ``RETRACTED``; return how many moved.

    Takes the connection rather than calling ``op.get_bind()`` so it is
    exercisable outside an Alembic context, as in migration 042. See
    ``tests/test_migration_053_cascade_closed_records.py``.
    """
    rows = bind.execute(
        sa.text(
            "SELECT id, content FROM particles"
            " WHERE status = 'PROVENANCE_STALE' AND status_reason = 'CONFLICT_RESOLVED'"
            " AND born_retired AND retired_at IS NULL"
        )
    ).fetchall()
    ids = sorted(pid for pid, content in rows if content.startswith(_RECORD_PREFIX))
    if ids:
        bind.execute(
            sa.text("UPDATE particles SET status = 'RETRACTED' WHERE id = :pid"),
            [{"pid": pid} for pid in ids],
        )
    return len(ids)


def upgrade() -> None:
    retract_cascade_closed_records(op.get_bind())


def downgrade() -> None:
    # Deliberately empty; see the module docstring.
    pass
