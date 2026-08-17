"""Utility as an additive rank-lift — lens vocabulary λ replaces weight/floor/cap.

This change supersedes the earlier design: the bounded,
saturating *multiplier* on
effective confidence becomes an additive term on the projection ordering key,
``rank_score = effective_confidence + λ·ln(1 + R)``. The three tunables that
parameterised the multiplier collapse into one — ``λ`` — carried on a lens
``utility_rules`` entry as ``rank_lift``.

Two changes on ``trust_lens_entries``:

1. A new nullable ``rank_lift`` column — the ``λ`` for a
   ``utility_default`` / ``utility_source_type`` / ``utility_url_pattern`` entry.
2. Drop ``cap``, which existed solely for the superseded utility multiplier's
   upper clamp. (``weight`` and ``floor`` are **kept** — they are still carried
   by ``extractor_weight`` and ``decay_*`` entries respectively; only their
   utility-rule *use* is retired. ``half_life_uses_days`` is unchanged: it
   parameterises ``R``, which is carried forward untouched.)

**Compat.** Utility-rule rows materialised under the old vocabulary keep their
``half_life_uses_days`` but have ``rank_lift`` NULL, so ``lens_store`` skips
them: the lens reads as *silent about utility* and the store's local ``utility``
config applies — the conservative, most-skeptical degradation rather than a
silently reinterpreted parameter (a stored ``weight = 0.5`` means something
entirely different as a ``λ``). Re-deposit an affected lens to restore its
utility layer under the new vocabulary. Stores with no utility-bearing lens —
the overwhelmingly common case — are unaffected either way.

SCHEMA_VERSION is unchanged: the lens artifact is Extension B and the particle
interchange is untouched (the ``TrustLensDefinition`` JSON Schema swaps
``weight``/``floor``/``cap`` for ``rank_lift``, synced in the activation commit).

Revision ID: 030
Revises: 029
"""

import sqlalchemy as sa

from alembic import op

revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("trust_lens_entries") as batch:
        batch.add_column(sa.Column("rank_lift", sa.Float(), nullable=True))
        batch.drop_column("cap")


def downgrade() -> None:
    with op.batch_alter_table("trust_lens_entries") as batch:
        batch.add_column(sa.Column("cap", sa.Float(), nullable=True))
        batch.drop_column("rank_lift")
