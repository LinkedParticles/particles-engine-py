# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Stamp the assertion_modality record; let a lens carry modality rules.

Three nullable columns on ``particles``, the stamp of the adjudicability
default:

* ``modality_classifier`` — the classifier identity (``<component>@<digest>``,
  or ``operator``) that last wrote the value after insert;
* ``modality_classifier_model`` — the ``"<provider>:<model>"`` that ran it;
* ``modality_classified_at`` — when. Also read by the delta scope,
  which is how a reclassified claim is re-paired by the next consolidation run.

plus an index on ``(status, modality_classifier)`` for the regeneration scope.

Two nullable columns on ``trust_lens_entries`` for the ``modality_<scope>``
entry kind: ``modality`` and ``modality_when``.

**No backfill.** Extraction never writes the stamp: a NULL stamp is resolved
through the minting snapshot's component record at read time, and a
row with no such record reads as a pre-stamp classification. Every existing
row is therefore already readable, and every existing lens is silent about
modality.

Revision ID: 049
Revises: 048
"""

import sqlalchemy as sa

from alembic import op

revision = "049"
down_revision = "048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("particles") as batch:
        batch.add_column(sa.Column("modality_classifier", sa.Text(), nullable=True))
        batch.add_column(sa.Column("modality_classifier_model", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("modality_classified_at", sa.DateTime(timezone=True), nullable=True)
        )
    op.create_index(
        "ix_particles_status_modality_classifier",
        "particles",
        ["status", "modality_classifier"],
    )
    with op.batch_alter_table("trust_lens_entries") as batch:
        batch.add_column(sa.Column("modality", sa.String(), nullable=True))
        batch.add_column(sa.Column("modality_when", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("trust_lens_entries") as batch:
        batch.drop_column("modality_when")
        batch.drop_column("modality")
    op.drop_index("ix_particles_status_modality_classifier", table_name="particles")
    with op.batch_alter_table("particles") as batch:
        batch.drop_column("modality_classified_at")
        batch.drop_column("modality_classifier_model")
        batch.drop_column("modality_classifier")
