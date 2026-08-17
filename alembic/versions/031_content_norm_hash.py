"""Normalized-content hash index for extract-time duplicate suppression.

The extract path declines to mint a particle whose claim is already held verbatim by an
ACTIVE particle with the same subjects and stance holder. The lookup has to be
an indexed probe — a scan over every ACTIVE particle on each extraction pass
would put an O(N) read on the write path.

Two changes on ``particles``:

1. A new nullable ``content_norm_hash`` column — SHA-256 over the *normalized*
   content (``particles.core.duplicate_key.content_hash``: whitespace collapsed,
   sentence-final punctuation trimmed, case and wording preserved).
2. A composite ``(status, content_norm_hash)`` index, status first because the
   rung only ever asks about ACTIVE particles.

**Backfill.** The column is populated for every existing row here rather than
left to lazy rewrite: an unbackfilled row is invisible to the suppression
lookup, so a claim already in the store would be re-minted once before the
column caught up — precisely the leak this ADR closes. The backfill is done in
Python (not SQL) because the normalization is Python-defined; a SQL expression
would be a second implementation of it, free to drift.

The hash is **derived, never authoritative**: ``content`` remains the source of
truth and ``ParticleRow.from_model`` recomputes the hash on every write, so a
downgrade loses only an index, never data.

SCHEMA_VERSION is unchanged — this is storage metadata, not a Core particle
field, and it is never serialized to the schema artifacts or interchange.

Revision ID: 031
Revises: 030
"""

import hashlib

import sqlalchemy as sa

from alembic import op

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None

# Vendored from particles.core.duplicate_key so this migration is a frozen
# historical artifact: it must keep reproducing the hash it wrote even if the
# live normalization is later revised (a revision would ship its own migration).
_TRAILING_PUNCT = ".,;:!?\"'"


def _content_hash(content: str) -> str:
    collapsed = " ".join(content.split())
    normalized = collapsed.rstrip(_TRAILING_PUNCT).rstrip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def upgrade() -> None:
    with op.batch_alter_table("particles") as batch:
        batch.add_column(sa.Column("content_norm_hash", sa.String(length=64), nullable=True))

    # Backfill in batches so a large store does not build one enormous
    # transaction. Reading (id, content) only keeps the working set small.
    conn = op.get_bind()
    particles = sa.table(
        "particles",
        sa.column("id", sa.String),
        sa.column("content", sa.Text),
        sa.column("content_norm_hash", sa.String),
    )
    rows = conn.execute(sa.select(particles.c.id, particles.c.content)).fetchall()
    for i in range(0, len(rows), 500):
        for row_id, content in rows[i : i + 500]:
            conn.execute(
                particles.update()
                .where(particles.c.id == row_id)
                .values(content_norm_hash=_content_hash(content or ""))
            )

    op.create_index(
        "ix_particles_status_content_hash",
        "particles",
        ["status", "content_norm_hash"],
    )


def downgrade() -> None:
    op.drop_index("ix_particles_status_content_hash", table_name="particles")
    with op.batch_alter_table("particles") as batch:
        batch.drop_column("content_norm_hash")
