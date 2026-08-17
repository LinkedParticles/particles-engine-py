"""Structured-claim annotation columns.

Five columns on ``particles``, split on the embedding precedent:
the derived triple lives in a JSON payload column, its generator stamp in three
columns of its own so the backfill's scope query and the coverage report are
SQL rather than a JSON scan.

* ``structured_claim_json`` — the S-P-O triple + resolved subject id.
* ``structurizer_id`` / ``structurizer_version`` / ``structured_claim_generated_at``
  — what produced it, at which version, and when.
* ``canonical_form`` — which of the prose/structured pair is the assertion.
  NOT NULL with a ``PROSE`` server default, so every existing row reads back as
  prose-canonical (which it is).

**No backfill.** Unlike migration 031's derived hash, a structured claim cannot
be computed in SQL — it takes an LLM call per particle. Every pre-migration row
is simply un-annotated, which the spec establishes as a legal *permanent*
state: no operation degrades when the annotation is absent, and nothing lints
for its absence. The operator populates rows deliberately, and at their own
pace, via ``particles structure``.

There is deliberately **no legacy tier** for the stamp (the one place this
diverges): embeddings predated their model marker and so
needed ``LEGACY_EMBEDDING_MODEL_ID``, whereas a structured claim is *born*
stamped — ``ParticleRow.from_model`` writes payload and stamp together or
writes neither — so a payload with a NULL stamp is unreachable and needs no
grandfathered default.

SCHEMA_VERSION is unchanged at ``1.0.0``: both new Particle
fields are additive and Optional with defaults that make every pre-existing
particle deserialize unchanged, and the guard in ``operations/version_guard.py``
compares versions with exact equality — a bump would make ``query`` /
``extract`` / ``review`` / ``reindex`` refuse to run on every existing store.

Revision ID: 033
Revises: 032
"""

import sqlalchemy as sa

from alembic import op

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("particles") as batch:
        batch.add_column(sa.Column("structured_claim_json", sa.Text(), nullable=True))
        batch.add_column(sa.Column("structurizer_id", sa.Text(), nullable=True))
        batch.add_column(sa.Column("structurizer_version", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "structured_claim_generated_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch.add_column(
            sa.Column(
                "canonical_form",
                sa.String(),
                nullable=False,
                server_default="PROSE",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("particles") as batch:
        batch.drop_column("canonical_form")
        batch.drop_column("structured_claim_generated_at")
        batch.drop_column("structurizer_version")
        batch.drop_column("structurizer_id")
        batch.drop_column("structured_claim_json")
