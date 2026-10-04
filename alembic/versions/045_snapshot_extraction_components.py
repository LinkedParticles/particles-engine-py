# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Record which extraction components each snapshot's extraction exercised.

One nullable text column on ``snapshots``:

* ``extraction_components_json`` — the components the snapshot's latest
  extraction exercised (each prompt section, the chunking path, the vision
  channel, the subject gate), each named and stamped with a content hash of
  the text that defines it. Written in the transaction that marks the snapshot
  COMPLETE after an extraction, beside ``extracted_through``.

Why a column. An ``EXTRACTOR_VERSION`` bump re-runs every snapshot stamped with
the old version, though most bumps change one prompt section or one rule. The
record is what lets a later bump select only the snapshots that reached the
changed part. The selection itself is not enabled by this migration.

**No backfill.** Every existing row stays NULL, and NULL reads as "every
component exercised": nothing is known about an extraction that ran before
the record, so it is never skipped on the record's account.

Revision ID: 045
Revises: 044
"""

import sqlalchemy as sa

from alembic import op

revision = "045"
down_revision = "044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("snapshots") as batch:
        batch.add_column(sa.Column("extraction_components_json", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("snapshots") as batch:
        batch.drop_column("extraction_components_json")
