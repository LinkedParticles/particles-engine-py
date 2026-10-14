# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The world-time axis on a claim: valid_from and valid_kind.

Two nullable columns on ``particles``:

* ``valid_from`` — when the claim begins to hold, the counterpart of
  ``valid_until``;
* ``valid_kind`` — ``EVENT`` (a moment) or ``STATE`` (a span), which is how
  the window route reads the pair.

plus an index on ``(status, valid_from)`` for the window route's candidate
scan and the backfill's "ACTIVE and undated" scope.

**No backfill here.** The crumbs that say how a value was set live in
``properties_json``, and the deterministic prose-date default is the
``particles validity`` verb, which an operator runs once; a
migration that read every claim's prose would hold the store for as long as
that pass takes, and would stamp no rule digest.

Revision ID: 051
Revises: 050
"""

import sqlalchemy as sa

from alembic import op

revision = "051"
down_revision = "050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("particles") as batch:
        batch.add_column(sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("valid_kind", sa.String(), nullable=True))
    op.create_index("ix_particles_status_valid_from", "particles", ["status", "valid_from"])


def downgrade() -> None:
    op.drop_index("ix_particles_status_valid_from", table_name="particles")
    with op.batch_alter_table("particles") as batch:
        batch.drop_column("valid_kind")
        batch.drop_column("valid_from")
