"""Per-particle extraction-model provenance.

One nullable column plus its index on ``particles``:

* ``extraction_provider_model`` — the ``"<provider>:<model>"`` pairing that
  produced the particle. The disclosure key and the calibration key, now recorded on the belief itself rather than computed
  transiently at extraction and thrown away.
* ``ix_particles_status_provider_model`` — the ``reindex --provider-model``
  scope query is the reason the column exists, so it is indexed on
  ``(status, extraction_provider_model)``, mirroring the three existing
  status-leading indexes.

The column is deliberately **not** a key inside ``extractor_ref_json``
: ``extractor_ref`` names the code, this names the runtime
substrate that code invoked, and one extractor version runs under many
models — folding them would make a model swap read as an extractor upgrade
to ``reindex --extractor-version``. The JSON column is also selected by SQL
substring, and pairings nest (``openai:gpt-5.6`` is a prefix of
``openai:gpt-5.6-luna``), so a nested scope would over-select the very
sibling model it exists to separate.

**No backfill, and no legacy tier** — this is where the design diverges
from the embedding precedent it otherwise follows. That marker
coalesces NULL to ``LEGACY_EMBEDDING_MODEL_ID`` because one known model had
embedded every pre-existing row. Here the premise is false, and its falsity
is the whole motivation: the 2026-08-01 provider trial left the store holding
Claude- and Luna-extracted particles side by side, so coalescing NULL to any
single pairing would manufacture provenance that is wrong for roughly half
the affected rows. An extraction pairing is *historical*, not derived — the
information was destroyed at mint time and no later computation recovers it
(``asserted_at`` is a heuristic, not a record). NULL therefore means
UNRECORDED, permanently, and the operator's remedy is re-extraction.

SCHEMA_VERSION is unchanged at ``1.0.0``: the field is
additive and Optional, and ``operations/version_guard.py`` compares versions
with exact equality — a bump would make ``query`` / ``extract`` / ``review``
/ ``reindex`` refuse to run on every existing store.

Revision ID: 034
Revises: 033
"""

import sqlalchemy as sa

from alembic import op

revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("particles") as batch:
        batch.add_column(sa.Column("extraction_provider_model", sa.Text(), nullable=True))
    op.create_index(
        "ix_particles_status_provider_model",
        "particles",
        ["status", "extraction_provider_model"],
    )


def downgrade() -> None:
    op.drop_index("ix_particles_status_provider_model", table_name="particles")
    with op.batch_alter_table("particles") as batch:
        batch.drop_column("extraction_provider_model")
