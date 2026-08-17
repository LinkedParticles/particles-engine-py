"""Utility-event channel discriminator — the explicit operator gesture.

A second producer is added to ``utility_events``: an explicit operator
usefulness gesture (``particles memory useful <id>``), which exists because the action-miner is structurally blind to prohibitive and stance guidelines
— compliance with "never do X" is the *absence* of a tool call.

Three changes on ``utility_events``:

1. A new ``source`` column — ``"mined"`` | ``"explicit"``. It decides the
   event's weight at read time and, just as importantly, scopes
   the miner's re-mine delete so a re-mine can never destroy an operator's
   gesture. Existing rows are all mined, hence the server default.
2. ``match_basis`` becomes **nullable**. It names which *miner tier* matched
   (``literal`` / ``behavioural``) and has no meaning off the miner, so explicit
   rows carry ``NULL`` rather than overloading it with a third value that would
   conflate "how the miner matched" with "who produced the event".
3. A **unique** index on ``(particle_id, session_id)``. This pair was already
   the table's natural key, enforced at write time only; the explicit channel
   leans on it harder — it synthesises ``session_id`` as
   ``explicit:<actor>:<date>``, which is what bounds the gesture to one credit
   per belief per principal per day. Any pre-existing duplicate
   pairs (never expected — the miner has always deleted-then-inserted per
   session) are collapsed to their first row before the constraint is added, so
   the upgrade cannot fail on legacy data.

No SCHEMA_VERSION bump: this is storage-layer utility evidence, not the
normative particle shape (own boundary, carried forward).

Revision ID: 032
Revises: 031
"""

import sqlalchemy as sa

from alembic import op

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None

_DEDUPE = """
DELETE FROM utility_events
 WHERE id NOT IN (
       SELECT MIN(id) FROM utility_events GROUP BY particle_id, session_id
 )
"""


def upgrade() -> None:
    with op.batch_alter_table("utility_events") as batch:
        batch.add_column(
            sa.Column("source", sa.String(), nullable=False, server_default="mined"),
        )
        batch.alter_column("match_basis", existing_type=sa.String(), nullable=True)
    op.execute(_DEDUPE)
    op.create_index(
        "uq_utility_events_particle_session",
        "utility_events",
        ["particle_id", "session_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_utility_events_particle_session", table_name="utility_events")
    # Explicit rows carry a NULL match_basis, which the pre-0216 NOT NULL column
    # cannot hold; drop that channel rather than inventing a tier for it. The
    # gestures survive in the operator event log and a rebuild re-derives them.
    op.execute("DELETE FROM utility_events WHERE source = 'explicit'")
    with op.batch_alter_table("utility_events") as batch:
        batch.alter_column(
            "match_basis",
            existing_type=sa.String(),
            nullable=False,
            server_default="literal",
        )
        batch.drop_column("source")
